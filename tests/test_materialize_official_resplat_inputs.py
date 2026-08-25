from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_official_resplat_inputs.py"
SPEC = importlib.util.spec_from_file_location(
    "materialize_official_resplat_inputs", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MATERIALIZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATERIALIZE
SPEC.loader.exec_module(MATERIALIZE)


class FakeCV2:
    INTER_AREA = 3
    INTER_LINEAR = 1

    def __init__(self) -> None:
        self.undistort_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.resize_calls: list[tuple[tuple[int, int], int]] = []

    def undistort(
        self, image: np.ndarray, camera: np.ndarray, distortion: np.ndarray
    ) -> np.ndarray:
        self.undistort_calls.append((image.copy(), camera.copy(), distortion.copy()))
        return image.copy()

    def resize(
        self, image: np.ndarray, size: tuple[int, int], *, interpolation: int
    ) -> np.ndarray:
        self.resize_calls.append((size, interpolation))
        width, height = size
        ys = np.linspace(0, image.shape[0] - 1, height).round().astype(int)
        xs = np.linspace(0, image.shape[1] - 1, width).round().astype(int)
        return image[ys][:, xs]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, *, include_distortion: bool = True) -> tuple[Path, Path]:
    image_root = root / "rgb"
    image_root.mkdir(parents=True)
    rows = []
    for index in range(3):
        yy, xx = np.mgrid[:6, :8]
        image = np.stack(
            (
                (xx * 20 + index) % 256,
                (yy * 30 + 2 * index) % 256,
                ((xx + yy) * 10 + 3 * index) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        path = image_root / f"{index}.png"
        Image.fromarray(image, mode="RGB").save(path)
        row = {
            "index": index,
            "frame": f"rgb/{index}.png",
            "timestamp": f"{index / 30:.9f}",
            "rgb_path": str(path.resolve()),
            "fx": 4.0,
            "fy": 5.0,
            "cx": 3.0,
            "cy": 2.0,
        }
        if include_distortion:
            row.update(
                {"k1": 0.1, "k2": -0.2, "p1": 0.01, "p2": -0.02, "k3": 0.3}
            )
        rows.append(row)
    csv_path = root / "frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = root / "net_g_latest_batch_8_no_NYU.pth"
    checkpoint.write_bytes(b"small official-EVSSM test fixture")
    return csv_path, checkpoint


class TestMaterializeOfficialReSplatInputs(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _allow_checkpoint(self, checkpoint: Path) -> mock._patch:
        return mock.patch.object(
            MATERIALIZE, "OFFICIAL_UNBLUR_EVSSM_SHA256", _sha(checkpoint)
        )

    def test_indices_file_raw_fallback_and_preprocessed_evssm_provenance(self) -> None:
        csv_path, checkpoint = _fixture(self.root)
        indices_file = self.root / "indices.txt"
        indices_file.write_text("2, 0  # preserve requested order\n", encoding="utf-8")
        tensors = self.root / "sharp"
        tensors.mkdir()
        evssm = torch.zeros(1, 3, 3, 4, dtype=torch.float32)
        evssm[:, 0] = 1.0
        evssm[:, 1] = 0.5
        torch.save(
            {
                "tensor": evssm,
                "shape": evssm.shape,
                "dtype": evssm.dtype,
                "timestamp": 2,
            },
            tensors / "2.pt",
        )

        real_load = torch.load
        loads: list[dict[str, object]] = []

        def audited_load(*args: object, **kwargs: object) -> object:
            loads.append(dict(kwargs))
            return real_load(*args, **kwargs)

        fake_cv2 = FakeCV2()
        output = self.root / "resplat_inputs"
        with self._allow_checkpoint(checkpoint), mock.patch.object(
            torch, "load", side_effect=audited_load
        ):
            manifest_path = MATERIALIZE.materialize(
                frames_csv=csv_path,
                output_dir=output,
                evssm_checkpoint=checkpoint,
                indices_file=indices_file,
                evssm_tensor_dir=tensors,
                width=4,
                height=3,
                _cv2_module=fake_cv2,
            )

        self.assertEqual(manifest_path, output / "manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], MATERIALIZE.SCHEMA)
        self.assertEqual(manifest["selection"]["source_indices"], [2, 0])
        self.assertEqual(
            manifest["provider_counts"],
            {"official_unblur_evssm": 1, "raw_undistorted": 1},
        )
        self.assertEqual(manifest["camera"]["width"], 4)
        self.assertEqual(manifest["camera"]["height"], 3)
        self.assertEqual(
            manifest["camera"]["K"],
            [[2.0, 0.0, 1.5], [0.0, 2.5, 1.0], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(loads, [{"map_location": "cpu", "weights_only": True}])

        # Only RAW fallback enters OpenCV.  The tracker tensor is already
        # undistorted/resized and must not be processed for a second time.
        self.assertEqual(len(fake_cv2.undistort_calls), 1)
        self.assertEqual(len(fake_cv2.resize_calls), 1)
        np.testing.assert_allclose(
            fake_cv2.undistort_calls[0][1],
            [[4.0, 0.0, 3.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]],
        )
        np.testing.assert_allclose(
            fake_cv2.undistort_calls[0][2], [0.1, -0.2, 0.01, -0.02, 0.3]
        )

        by_index = {frame["source_index"]: frame for frame in manifest["frames"]}
        evssm_record = by_index[2]
        raw_record = by_index[0]
        self.assertEqual(evssm_record["provider"], "official_unblur_evssm")
        self.assertEqual(
            evssm_record["preprocessing"],
            {
                "performed_by_materializer": False,
                "operations": ["TUM undistort", "tracker resize/crop"],
                "second_undistort_forbidden": True,
                "status": "preprocessed_upstream",
            },
        )
        self.assertIs(evssm_record["tensor"]["weights_only"], True)
        self.assertEqual(
            evssm_record["tensor"]["container_schema"], "tracker_safe_tensor_v1"
        )
        self.assertEqual(
            evssm_record["tensor"]["container"],
            {
                "schema": "tracker_safe_tensor_v1",
                "source_index_bound": True,
                "timestamp": 2,
            },
        )
        self.assertEqual(evssm_record["tensor_sha256"], _sha(tensors / "2.pt"))
        self.assertEqual(raw_record["provider"], "raw_undistorted")
        self.assertIsNone(raw_record["tensor_sha256"])
        self.assertEqual(raw_record["raw_sha256"], _sha(self.root / "rgb" / "0.png"))
        self.assertEqual(
            raw_record["distortion"]["vector"], [0.1, -0.2, 0.01, -0.02, 0.3]
        )
        self.assertEqual(raw_record["intrinsics"]["scale"], {"x": 0.5, "y": 0.5})
        self.assertTrue(
            all(frame["camera_reference"] == "#/camera" for frame in manifest["frames"])
        )
        self.assertTrue(
            all(
                frame["evssm_checkpoint_sha256"] == _sha(checkpoint)
                for frame in manifest["frames"]
            )
        )
        for frame in manifest["frames"]:
            png = Path(frame["output"]["path"])
            self.assertEqual(frame["png_sha256"], _sha(png))
            with Image.open(png) as image:
                self.assertEqual(image.size, (4, 3))
        with Image.open(evssm_record["output"]["path"]) as image:
            np.testing.assert_array_equal(np.asarray(image)[0, 0], [255, 128, 0])

    def test_global_distortion_override_and_no_overwrite(self) -> None:
        csv_path, checkpoint = _fixture(self.root, include_distortion=False)
        output = self.root / "immutable"
        with self._allow_checkpoint(checkpoint):
            manifest = MATERIALIZE.materialize(
                frames_csv=csv_path,
                output_dir=output,
                evssm_checkpoint=checkpoint,
                indices=[1],
                width=4,
                height=3,
                distortion=(1.0, 2.0, 3.0, 4.0, 5.0),
                _cv2_module=FakeCV2(),
            )
            manifest_before = manifest.read_bytes()
            png = output / "images" / "000001.png"
            png_before = png.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                MATERIALIZE.materialize(
                    frames_csv=csv_path,
                    output_dir=output,
                    evssm_checkpoint=checkpoint,
                    indices=[1],
                    width=4,
                    height=3,
                    distortion=(1.0, 2.0, 3.0, 4.0, 5.0),
                    _cv2_module=FakeCV2(),
                )
        payload = json.loads(manifest_before)
        self.assertEqual(
            payload["frames"][0]["distortion"]["coefficients"],
            {"k1": 1.0, "k2": 2.0, "p1": 3.0, "p2": 4.0, "k3": 5.0},
        )
        self.assertEqual(manifest.read_bytes(), manifest_before)
        self.assertEqual(png.read_bytes(), png_before)

    def test_rejects_wrong_or_forbidden_checkpoint(self) -> None:
        csv_path, checkpoint = _fixture(self.root)
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
            MATERIALIZE.materialize(
                frames_csv=csv_path,
                output_dir=self.root / "bad_hash",
                evssm_checkpoint=checkpoint,
                indices=[0],
                width=4,
                height=3,
                _cv2_module=FakeCV2(),
            )

        turtle_dir = self.root / "TURTLE"
        turtle_dir.mkdir()
        forbidden = turtle_dir / "checkpoint.pth"
        forbidden.write_bytes(checkpoint.read_bytes())
        with mock.patch.object(
            MATERIALIZE, "OFFICIAL_UNBLUR_EVSSM_SHA256", _sha(forbidden)
        ), self.assertRaisesRegex(ValueError, "forbidden 'turtle'"):
            MATERIALIZE.materialize(
                frames_csv=csv_path,
                output_dir=self.root / "turtle_out",
                evssm_checkpoint=forbidden,
                indices=[0],
            )

        gopro_tensors = self.root / "GoPro_tensors"
        gopro_tensors.mkdir()
        with self._allow_checkpoint(checkpoint), self.assertRaisesRegex(
            ValueError, "forbidden 'gopro'"
        ):
            MATERIALIZE.materialize(
                frames_csv=csv_path,
                output_dir=self.root / "gopro_out",
                evssm_checkpoint=checkpoint,
                indices=[0],
                evssm_tensor_dir=gopro_tensors,
            )

    def test_rejects_malformed_tensors_and_cleans_staging(self) -> None:
        valid = torch.zeros(1, 3, 3, 4)
        cases = [
            ({"tensor": valid}, "must contain exactly keys"),
            (
                {
                    "tensor": valid,
                    "shape": valid.shape,
                    "dtype": valid.dtype,
                    "timestamp": 0,
                    "extra": True,
                },
                "must contain exactly keys",
            ),
            (
                {
                    "tensor": "not-a-tensor",
                    "shape": valid.shape,
                    "dtype": valid.dtype,
                    "timestamp": 0,
                },
                "safe dict tensor must be a torch.Tensor",
            ),
            (
                {
                    "tensor": valid,
                    "shape": torch.Size((1, 3, 2, 4)),
                    "dtype": valid.dtype,
                    "timestamp": 0,
                },
                "shape does not match",
            ),
            (
                {
                    "tensor": valid,
                    "shape": valid.shape,
                    "dtype": torch.float64,
                    "timestamp": 0,
                },
                "dtype does not match",
            ),
            (
                {
                    "tensor": valid,
                    "shape": valid.shape,
                    "dtype": valid.dtype,
                    "timestamp": 1,
                },
                "timestamp must be the exact integer source_index 0",
            ),
            (
                {
                    "tensor": valid,
                    "shape": valid.shape,
                    "dtype": valid.dtype,
                    "timestamp": True,
                },
                "timestamp must be the exact integer source_index 0",
            ),
            (torch.zeros(3, 3, 4), r"shape \[1,3,H,W\]"),
            (torch.zeros(1, 3, 3, 4, dtype=torch.int64), "floating dtype"),
            (torch.zeros(1, 3, 2, 4), "already match ReSplat output size"),
            (torch.full((1, 3, 3, 4), 1.1), r"values must be in \[0,1\]"),
            (torch.full((1, 3, 3, 4), float("nan")), "non-finite"),
        ]
        for case_index, (payload, error) in enumerate(cases):
            with self.subTest(error=error):
                case_root = self.root / f"case_{case_index}"
                case_root.mkdir()
                csv_path, checkpoint = _fixture(case_root)
                tensors = case_root / "sharp"
                tensors.mkdir()
                torch.save(payload, tensors / "0.pt")
                output = case_root / "failed_bundle"
                with self._allow_checkpoint(checkpoint), self.assertRaisesRegex(
                    ValueError, error
                ):
                    MATERIALIZE.materialize(
                        frames_csv=csv_path,
                        output_dir=output,
                        evssm_checkpoint=checkpoint,
                        indices=[0],
                        evssm_tensor_dir=tensors,
                        width=4,
                        height=3,
                        _cv2_module=FakeCV2(),
                    )
                self.assertFalse(output.exists())
                self.assertEqual(list(case_root.glob(".failed_bundle.staging-*")), [])

    def test_indices_validation_and_single_mapping_camera(self) -> None:
        csv_path, checkpoint = _fixture(self.root)
        indices_file = self.root / "indices.txt"
        indices_file.write_text("0\n99\n", encoding="utf-8")
        with self._allow_checkpoint(checkpoint), self.assertRaisesRegex(
            ValueError, "absent from TUM source CSV"
        ):
            MATERIALIZE.materialize(
                frames_csv=csv_path,
                output_dir=self.root / "missing",
                evssm_checkpoint=checkpoint,
                indices_file=indices_file,
            )

        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[1]["fx"] = "4.5"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self._allow_checkpoint(checkpoint), self.assertRaisesRegex(
            ValueError, "single mapping camera"
        ):
            MATERIALIZE.materialize(
                frames_csv=csv_path,
                output_dir=self.root / "mixed_camera",
                evssm_checkpoint=checkpoint,
                indices=[0, 1],
                width=4,
                height=3,
                _cv2_module=FakeCV2(),
            )
        self.assertFalse((self.root / "mixed_camera").exists())


if __name__ == "__main__":
    unittest.main()
