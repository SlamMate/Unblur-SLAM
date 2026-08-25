#!/usr/bin/env python3
"""CPU-only contracts for audited FrameCrafter EVSSM precomputation."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "precompute_framecrafter_evssm.py"
SPEC = importlib.util.spec_from_file_location("precompute_framecrafter_evssm", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PRECOMPUTE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PRECOMPUTE
SPEC.loader.exec_module(PRECOMPUTE)


def _write_fixture(root: Path, *, pose_source: str = "droid_traj_est_not_align") -> Path:
    images = root / "rgb"
    images.mkdir(parents=True)
    rows = []
    for index, (width, height) in enumerate(((11, 7), (12, 8), (13, 9))):
        yy, xx = np.mgrid[:height, :width]
        rgb = np.stack(
            (
                (17 * xx + index) % 256,
                (29 * yy + 3 * index) % 256,
                (11 * (xx + yy) + 7 * index) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        path = images / f"frame_{index}.png"
        Image.fromarray(rgb, mode="RGB").save(path)
        rows.append(
            {
                "index": index,
                "frame": path.name,
                "timestamp": f"{index * 0.033:.6f}",
                "rgb_path": str(path.resolve()),
                "pose_source": pose_source,
                "uses_ground_truth_pose": "false",
            }
        )
    csv_path = root / "frames.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


class TestFrameCrafterEVSSMPrecompute(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _checkpoint(self) -> Path:
        checkpoint = self.root / "dummy_evssm.pth"
        checkpoint.write_bytes(b"identity-contract-checkpoint")
        return checkpoint

    def test_identity_subset_writes_strict_nonproduction_metadata(self) -> None:
        csv_path = _write_fixture(self.root)
        checkpoint = self._checkpoint()
        output = self.root / "out"

        metadata_path = PRECOMPUTE.precompute(
            frames_csv=csv_path,
            checkpoint=checkpoint,
            output_dir=output,
            source_indices=("2,0",),
            test_only_identity=True,
        )

        self.assertEqual(metadata_path, (output / "metadata.json").resolve())
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], PRECOMPUTE.METADATA_SCHEMA)
        self.assertEqual(payload["artifact_class"], "test_only_identity")
        self.assertIs(payload["test_only"], True)
        self.assertIs(payload["production_eligible"], False)
        self.assertIs(payload["uses_ground_truth_pose"], False)
        self.assertEqual(payload["pose_source"], "droid_traj_est_not_align")
        # CSV stream order is retained even if the request is unordered.
        self.assertEqual(payload["selection"]["source_indices"], [0, 2])
        self.assertEqual(payload["selection"]["count"], 2)
        self.assertEqual(len(payload["checkpoint_sha256"]), 64)
        self.assertEqual(len(payload["implementation"]["sha256"]), 64)
        self.assertEqual(payload["accepted_count"], 0)

        for record in payload["frames"]:
            self.assertEqual(record["schema"], PRECOMPUTE.METADATA_SCHEMA)
            self.assertTrue(Path(record["raw"]["path"]).is_absolute())
            self.assertTrue(Path(record["output"]["path"]).is_absolute())
            self.assertEqual(
                PRECOMPUTE.sha256_file(record["raw"]["path"]),
                record["raw"]["sha256"],
            )
            self.assertEqual(
                PRECOMPUTE.sha256_file(record["output"]["path"]),
                record["output"]["sha256"],
            )
            self.assertEqual(record["checkpoint_sha256"], payload["checkpoint_sha256"])
            self.assertEqual(
                record["implementation_sha256"], payload["implementation"]["sha256"]
            )
            self.assertEqual(len(record["cache_key"]), 64)
            self.assertIs(record["uses_ground_truth_pose"], False)
            self.assertIs(record["test_only"], True)
            self.assertIs(record["production_eligible"], False)
            self.assertEqual(record["provider"], "test_only_identity")
            self.assertIs(record["accepted"], False)
            self.assertEqual(record["confidence"], 0.0)
            self.assertIn("test_only_identity", record["failures"])
            self.assertAlmostEqual(record["sharpness_gain"], 1.0)
            self.assertAlmostEqual(record["image_consistency"], 1.0)
            with Image.open(record["raw_path"]) as raw, Image.open(
                record["output_path"]
            ) as generated:
                self.assertEqual(generated.size, raw.size)
                np.testing.assert_array_equal(
                    np.asarray(generated), np.asarray(raw.convert("RGB"))
                )

    def test_all_selection_threshold_audit_and_no_silent_overwrite(self) -> None:
        csv_path = _write_fixture(self.root)
        checkpoint = self._checkpoint()
        output = self.root / "out"
        metadata = PRECOMPUTE.precompute(
            frames_csv=csv_path,
            checkpoint=checkpoint,
            output_dir=output,
            source_indices=("all",),
            min_sharpness_gain=1.1,
            test_only_identity=True,
        )
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertEqual(payload["selection"]["source_indices"], [0, 1, 2])
        self.assertTrue(
            all(record["metric_gates_passed"] is False for record in payload["frames"])
        )
        self.assertTrue(
            all("sharpness_gain" in record["failures"] for record in payload["frames"])
        )
        self.assertEqual(payload["thresholds"]["min_sharpness_gain"], 1.1)

        with self.assertRaisesRegex(FileExistsError, "metadata already exists"):
            PRECOMPUTE.precompute(
                frames_csv=csv_path,
                checkpoint=checkpoint,
                output_dir=output,
                test_only_identity=True,
            )

    def test_rejects_gt_pose_provenance_and_bad_source_requests(self) -> None:
        unsafe_csv = _write_fixture(
            self.root / "unsafe", pose_source="aligned_to_gt"
        )
        checkpoint = self._checkpoint()
        with self.assertRaisesRegex(ValueError, "non-GT"):
            PRECOMPUTE.precompute(
                frames_csv=unsafe_csv,
                checkpoint=checkpoint,
                output_dir=self.root / "unsafe_out",
                test_only_identity=True,
            )

        safe_csv = _write_fixture(self.root / "safe")
        with self.assertRaisesRegex(ValueError, "absent from CSV"):
            PRECOMPUTE.precompute(
                frames_csv=safe_csv,
                checkpoint=checkpoint,
                output_dir=self.root / "missing_out",
                source_indices=("99",),
                test_only_identity=True,
            )

    def test_cli_identity_mode_is_explicit(self) -> None:
        csv_path = _write_fixture(self.root)
        checkpoint = self._checkpoint()
        output = self.root / "cli_out"
        self.assertEqual(
            PRECOMPUTE.main(
                [
                    "--frames-csv",
                    str(csv_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output),
                    "--source-indices",
                    "1",
                    "--test-only-identity",
                ]
            ),
            0,
        )
        payload = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["selection"]["source_indices"], [1])
        self.assertEqual(payload["backend"], "test_only_identity")
        self.assertEqual(payload["accepted_count"], 0)


if __name__ == "__main__":
    unittest.main()
