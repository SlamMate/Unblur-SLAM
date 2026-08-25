#!/usr/bin/env python3
"""CPU-only contracts for the official cvg/ReSplat TUM scene exporter."""

from __future__ import annotations

import csv
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_tum_official_resplat_scene.py"
SPEC = importlib.util.spec_from_file_location(
    "export_tum_official_resplat_scene", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _read_colmap_images(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split()
        if len(fields) != 10:
            raise AssertionError(f"invalid images.txt pose row: {line}")
        if index + 1 >= len(lines) or lines[index + 1] != "":
            raise AssertionError("every COLMAP image needs an empty POINTS2D line")
        result.append(
            {
                "id": int(fields[0]),
                "qvec": np.asarray(fields[1:5], dtype=np.float64),
                "tvec": np.asarray(fields[5:8], dtype=np.float64),
                "camera_id": int(fields[8]),
                "name": fields[9],
            }
        )
        index += 2
    return result


def _colmap_record_to_c2w(record: dict[str, object]) -> np.ndarray:
    rotation_w2c = _qvec_to_rotation(np.asarray(record["qvec"]))
    translation_w2c = np.asarray(record["tvec"], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation_w2c.T
    c2w[:3, 3] = -rotation_w2c.T @ translation_w2c
    return c2w


class OfficialReSplatSceneExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "rgb"
        self.images.mkdir()
        self.repo = self._make_official_repo_fixture()
        self.rows = self._make_rows()
        self.csv_path = self.root / "estimated_frames.csv"
        self._write_csv(self.rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_official_repo_fixture(self) -> Path:
        repo = self.root / "resplat"
        (repo / "scripts").mkdir(parents=True)
        (repo / "scripts" / "infer_colmap.py").write_text(
            "# official inference fixture\n", encoding="utf-8"
        )
        (repo / "MODEL_ZOO.md").write_text("# model zoo fixture\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "remote", "add", "origin", EXPORT.OFFICIAL_RESPLAT_URL],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            [
                "git", "-C", str(repo),
                "-c", "user.name=CPU Test",
                "-c", "user.email=cpu-test@example.invalid",
                "commit", "-q", "-m", "fixture",
            ],
            check=True,
        )
        return repo

    def _make_rows(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        specifications = (
            (10, 1.0, 2.0, 3.0, 90.0),
            (2, -0.25, 0.5, 1.5, 0.0),
            (7, 4.0, -1.0, 0.25, -45.0),
        )
        for source_index, tx, ty, tz, degrees in specifications:
            yy, xx = np.mgrid[:8, :12]
            rgb = np.stack(
                [
                    (xx * 11 + source_index) % 255,
                    (yy * 17 + source_index) % 255,
                    (xx + yy * 5 + source_index) % 255,
                ],
                axis=-1,
            ).astype(np.uint8)
            image = self.images / f"raw_{source_index}.png"
            Image.fromarray(rgb, mode="RGB").save(image)
            half = math.radians(degrees) / 2.0
            result.append(
                {
                    "index": source_index,
                    "frame": image.name,
                    "timestamp": source_index / 30.0,
                    "rgb_path": str(image),
                    "tx": tx,
                    "ty": ty,
                    "tz": tz,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": math.sin(half),
                    "qw": math.cos(half),
                    "fx": 525.0,
                    "fy": 526.0,
                    "cx": 5.5,
                    "cy": 3.5,
                    "pose_source": "droid_traj_est_not_align",
                    "uses_ground_truth_pose": "false",
                }
            )
        return result

    def _write_csv(self, rows: list[dict[str, object]]) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _export(self, output: Path, **overrides: object) -> Path:
        arguments: dict[str, object] = {
            "frames_csv": self.csv_path,
            "output_dir": output,
            "selected_indices": [10, 2],
            "selection_provenance": {"kind": "test"},
            "image_mode": "raw",
            "images_json": None,
            "image_root": None,
            "resplat_repo": self.repo,
            "model_preset": "dl3dv_8v_256x448_small",
        }
        arguments.update(overrides)
        return EXPORT.export_scene(**arguments)

    def _csv_c2w(self) -> dict[int, np.ndarray]:
        frames, _, _ = EXPORT.load_frames_csv(self.csv_path)
        return {frame.source_index: frame.c2w for frame in frames}

    def test_colmap_roundtrip_sorted_filenames_atomic_and_no_gt_manifest(self) -> None:
        indices = self.root / "indices.txt"
        indices.write_text("10, 2 # deliberate request order\n", encoding="utf-8")
        selected, selection_provenance = EXPORT.parse_indices(
            indices=None, indices_file=indices
        )
        self.assertEqual(selected, [2, 10])
        output = self._export(
            self.root / "scene",
            selected_indices=selected,
            selection_provenance=selection_provenance,
        )

        camera_fields = [
            line for line in (output / "sparse/0/cameras.txt").read_text().splitlines()
            if line and not line.startswith("#")
        ][0].split()
        self.assertEqual(camera_fields[:4], ["1", "PINHOLE", "12", "8"])
        np.testing.assert_allclose(
            np.asarray(camera_fields[4:], dtype=np.float64), [525, 526, 5.5, 3.5]
        )

        records = _read_colmap_images(output / "sparse/0/images.txt")
        self.assertEqual([record["name"] for record in records], ["00000002.png", "00000010.png"])
        expected = self._csv_c2w()
        for source_index, record in zip((2, 10), records):
            np.testing.assert_allclose(
                _colmap_record_to_c2w(record), expected[source_index], atol=1.0e-12
            )

        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], EXPORT.SCHEMA)
        self.assertEqual(manifest["selection"]["source_indices"], [2, 10])
        self.assertEqual(manifest["selection"]["provenance"]["sha256"], _sha256(indices))
        self.assertIs(manifest["ground_truth_contract"]["uses_ground_truth_pose"], False)
        self.assertIs(manifest["ground_truth_contract"]["ground_truth_file_read"], False)
        self.assertIs(manifest["images"]["undistortion_claimed"], False)
        self.assertIs(
            manifest["images"]["generator_or_checkpoint_verified_by_exporter"], False
        )
        for frame in manifest["frames"]:
            copied = output / frame["selected_image"]["exported_relative_path"]
            self.assertEqual(_sha256(copied), frame["selected_image"]["source_sha256"])
            self.assertFalse(frame["effective_pose"]["uses_ground_truth_pose"])

        with self.assertRaisesRegex(FileExistsError, "overwrite"):
            self._export(output)
        self.assertFalse(any(self.root.glob(".scene.*")))

    def test_formal_nonraw_requires_verified_mapping_and_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-raw"):
            self._export(self.root / "raw_formal", formal_smoke=True)
        with self.assertRaisesRegex(ValueError, "requires --images-json"):
            self._export(
                self.root / "no_mapping", image_mode="evssm", formal_smoke=False
            )

        deblur_dir = self.root / "evssm"
        deblur_dir.mkdir()
        mappings = []
        for row in self.rows:
            source_index = int(row["index"])
            source = Path(str(row["rgb_path"]))
            target = deblur_dir / f"evssm_{source_index}.png"
            with Image.open(source) as image:
                image.resize((6, 4), Image.Resampling.BILINEAR).save(target)
            mappings.append(
                {
                    "source_index": source_index,
                    "output": {"path": str(target), "sha256": _sha256(target)},
                }
            )
        mapping_path = self.root / "evssm_images.json"
        mapping_path.write_text(
            json.dumps(
                {
                    "schema": "test",
                    "camera": {
                        "width": 6,
                        "height": 4,
                        "K": [[262.5, 0.0, 2.75], [0.0, 263.0, 1.75], [0.0, 0.0, 1.0]],
                    },
                    "frames": mappings,
                }
            ),
            encoding="utf-8",
        )
        bad_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        bad_payload["frames"][0]["output"].pop("sha256")
        bad_mapping = self.root / "bad_images.json"
        bad_mapping.write_text(json.dumps(bad_payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "declare a full SHA-256"):
            self._export(
                self.root / "bad_hash",
                image_mode="evssm",
                images_json=bad_mapping,
            )

        checkpoint_bytes = b"deterministic official checkpoint fixture"
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        preset_name = "cpu_test_official_preset"
        checkpoint_name = f"test-checkpoint-{checkpoint_hash[:8]}.pth"
        checkpoint = self.root / checkpoint_name
        checkpoint.write_bytes(checkpoint_bytes)
        EXPORT.PRESET_SPECS[preset_name] = {
            "num_context": 2,
            "num_refine": 1,
            "checkpoint_filename": checkpoint_name,
            "checkpoint_sha256_prefix": checkpoint_hash[:8],
        }
        try:
            output = self._export(
                self.root / "formal_scene",
                selected_indices=[2, 7, 10],
                image_mode="evssm",
                images_json=mapping_path,
                formal_smoke=True,
                model_preset=preset_name,
                checkpoint=checkpoint,
                expected_checkpoint_sha256=checkpoint_hash,
            )
            sidecar_poses = np.repeat(np.eye(4)[None], 11, axis=0)
            sidecar_npz = self.root / "formal_sidecar_estimate.npz"
            np.savez(
                sidecar_npz,
                full_ba_c2w=sidecar_poses,
                traj_ref_poses=sidecar_poses,
                pose_source=np.asarray("full_ba_estimate_not_align"),
                uses_ground_truth_pose=np.asarray(False),
            )
            with self.assertRaisesRegex(ValueError, "GT/reference sidecars"):
                self._export(
                    self.root / "formal_sidecar_scene",
                    selected_indices=[2, 7, 10],
                    image_mode="evssm",
                    images_json=mapping_path,
                    formal_smoke=True,
                    model_preset=preset_name,
                    checkpoint=checkpoint,
                    expected_checkpoint_sha256=checkpoint_hash,
                    trajectory_npz=sidecar_npz,
                    trajectory_key="full_ba_c2w",
                )
        finally:
            EXPORT.PRESET_SPECS.pop(preset_name, None)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["formal_smoke"])
        self.assertTrue(manifest["images"]["mapped_hashes_verified"])
        self.assertEqual(
            manifest["official_resplat"]["checkpoint"]["actual"]["sha256"],
            checkpoint_hash,
        )
        self.assertEqual(
            [frame["selected_image"]["mode_label"] for frame in manifest["frames"]],
            ["evssm", "evssm", "evssm"],
        )
        self.assertEqual(manifest["camera"]["source"], "images_json")
        self.assertEqual(manifest["camera"]["width"], 6)
        self.assertEqual(manifest["camera"]["height"], 4)
        np.testing.assert_allclose(
            manifest["camera"]["K"],
            [[262.5, 0.0, 2.75], [0.0, 263.0, 1.75], [0.0, 0.0, 1.0]],
        )
        camera_fields = [
            line for line in (output / "sparse/0/cameras.txt").read_text().splitlines()
            if line and not line.startswith("#")
        ][0].split()
        self.assertEqual(camera_fields[:4], ["1", "PINHOLE", "6", "4"])
        np.testing.assert_allclose(
            np.asarray(camera_fields[4:], dtype=np.float64), [262.5, 263.0, 2.75, 1.75]
        )

    def test_full_ba_npz_overrides_effective_pose_but_preserves_csv_audit(self) -> None:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], 11, axis=0)
        for index in range(len(poses)):
            poses[index, :3, 3] = [index * 0.25, -index * 0.1, 2.0]
        trajectory = self.root / "full_ba_estimate.npz"
        np.savez(
            trajectory,
            full_ba_c2w=poses,
            traj_ref_poses=np.repeat(np.eye(4)[None], len(poses), axis=0),
            pose_source=np.asarray("full_ba_estimate_not_align"),
            uses_ground_truth_pose=np.asarray(False),
        )
        output = self._export(
            self.root / "ba_scene",
            trajectory_npz=trajectory,
            trajectory_key="full_ba_c2w",
        )
        records = _read_colmap_images(output / "sparse/0/images.txt")
        for source_index, record in zip((2, 10), records):
            np.testing.assert_allclose(
                _colmap_record_to_c2w(record), poses[source_index], atol=1.0e-12
            )
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["pose_override"]["sha256"], _sha256(trajectory))
        self.assertEqual(manifest["pose_override"]["shape"], [11, 4, 4])
        self.assertTrue(manifest["pose_override"]["contains_ground_truth_sidecar"])
        self.assertEqual(
            manifest["pose_override"]["ground_truth_sidecar_keys"], ["traj_ref_poses"]
        )
        self.assertTrue(manifest["ground_truth_contract"]["ground_truth_file_read"])
        self.assertTrue(
            manifest["ground_truth_contract"]["contains_ground_truth_sidecar"]
        )
        self.assertFalse(
            manifest["ground_truth_contract"]["ground_truth_sidecar_arrays_accessed"]
        )
        self.assertEqual(manifest["effective_pose_source"], "full_ba_estimate_not_align")
        csv_expected = self._csv_c2w()
        for frame in manifest["frames"]:
            source_index = frame["source_index"]
            np.testing.assert_allclose(
                frame["csv_pose_audit"]["c2w_opencv"], csv_expected[source_index]
            )
            np.testing.assert_allclose(
                frame["effective_pose"]["c2w_opencv"], poses[source_index]
            )

        short = self.root / "short_ba.npz"
        np.savez(
            short,
            full_ba_c2w=poses[:10],
            pose_source=np.asarray("full_ba_estimate_not_align"),
            uses_ground_truth_pose=np.asarray(False),
        )
        with self.assertRaisesRegex(ValueError, "at least 11 poses"):
            self._export(
                self.root / "short_scene",
                trajectory_npz=short,
                trajectory_key="full_ba_c2w",
            )
        unsafe = self.root / "unsafe_ba.npz"
        np.savez(
            unsafe,
            full_ba_c2w=poses,
            pose_source=np.asarray("aligned_to_gt"),
            uses_ground_truth_pose=np.asarray(False),
        )
        with self.assertRaisesRegex(ValueError, "non-GT"):
            self._export(
                self.root / "unsafe_scene",
                trajectory_npz=unsafe,
                trajectory_key="full_ba_c2w",
            )

    def test_audits_all_csv_rows_and_leaves_no_partial_output(self) -> None:
        inconsistent = [dict(row) for row in self.rows]
        inconsistent[-1]["fx"] = 999.0
        self._write_csv(inconsistent)
        output = self.root / "inconsistent_scene"
        with self.assertRaisesRegex(ValueError, "intrinsics K"):
            self._export(output, selected_indices=[2])
        self.assertFalse(output.exists())

        unsafe = [dict(row) for row in self.rows]
        unsafe[-1]["uses_ground_truth_pose"] = "true"
        self._write_csv(unsafe)
        with self.assertRaisesRegex(ValueError, "explicitly be false"):
            self._export(self.root / "unsafe_scene", selected_indices=[2])

        self._write_csv(self.rows)
        Path(str(self.rows[-1]["rgb_path"])).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            self._export(self.root / "missing_scene", selected_indices=[2])

    def test_cli_requires_explicit_indices_and_returns_failure_without_them(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                EXPORT.parse_args(
                    [
                        "--frames-csv", str(self.csv_path),
                        "--output-dir", str(self.root / "scene"),
                        "--image-mode", "raw",
                        "--resplat-repo", str(self.repo),
                    ]
                )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            EXPORT.parse_indices(indices="2,2", indices_file=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
