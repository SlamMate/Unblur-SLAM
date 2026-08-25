#!/usr/bin/env python3
"""CPU-only contract tests for advanced FrameCrafter evidence rendering."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "visualize_framecrafter_advanced.py"
REPORT_SCHEMA = "unblur_slam.framecrafter_preprocess_report.v1"
MANIFEST_SCHEMA = "unblur_slam.framecrafter_manifest.v1"


def write_image(path: Path, base: int, detail_x: int) -> None:
    array = np.full((48, 64, 3), base, dtype=np.uint8)
    array[8:40, detail_x : detail_x + 12] = 255 - base
    array[8:40:2, detail_x : detail_x + 12] = base // 3
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


class AdvancedFrameCrafterVisualizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.snapshot = self.root / "advanced_snapshot"
        self.snapshot.mkdir()
        self.signature = "a" * 64
        self.generation_id = "b" * 32
        self.report_path = self.snapshot / (
            f"preprocess_report_{self.signature}_{self.generation_id}.json"
        )
        self.manifest_path = self.snapshot / (
            f"manifest_{self.signature}_{self.generation_id}.json"
        )
        self.originals = []
        for index, base in enumerate((20, 50, 80, 110)):
            path = self.snapshot / "originals" / f"source_{index}.png"
            write_image(path, base, 8 + index * 4)
            self.originals.append(path)
        self.raw_context = self.snapshot / "contexts" / "raw.png"
        self.evssm_context = self.snapshot / "contexts" / "evssm.png"
        self.fallback_context = self.snapshot / "contexts" / "fallback.png"
        write_image(self.raw_context, 60, 10)
        write_image(self.evssm_context, 90, 20)
        write_image(self.fallback_context, 120, 30)
        self.conditioning = [
            {
                "source_index": 0,
                "role": "local_blurry_before",
                "requested_mode": "hybrid",
                "resolved_mode": "raw",
                "resolved_path": str(self.raw_context),
                "fallback_reason": None,
                "score": {"overlap": 0.72, "total": 0.61},
            },
            {
                "source_index": 1,
                "role": "sharp_before",
                "requested_mode": "hybrid",
                "resolved_mode": "evssm",
                "resolved_path": str(self.evssm_context),
                "fallback_reason": None,
                "score": {"overlap": 0.84, "total": 0.79},
            },
            {
                "source_index": 3,
                "role": "sharp_after",
                "requested_mode": "hybrid",
                "resolved_mode": "raw",
                "resolved_path": str(self.fallback_context),
                "fallback_reason": "evssm_consistency",
                "score": {"overlap": 0.67, "total": 0.70},
            },
        ]
        self.generated: dict[str, Path] = {}
        for index, target_id in enumerate(("target_sharp", "target_geometry", "target_rejected")):
            path = self.snapshot / "generated" / f"{target_id}.png"
            write_image(path, 140 + 20 * index, 16 + 4 * index)
            self.generated[target_id] = path
        self._write_advanced_snapshot()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_outcome(self, target_id: str, partition: str) -> dict[str, object]:
        record: dict[str, object] = {
            "target_id": target_id,
            "batch_id": "batch_00000",
            # Deliberately not equal to planned endpoints: this verifies that
            # the visualizer uses the local gate support recorded by the gate.
            "gate_support_source_indices": [1, 2],
            "acceptance_class": partition,
            "metrics": {
                "sharpness_gain": 1.20 if partition == "sharp_accepted" else 0.91,
                "depth_coverage": 0.88,
                "depth_consistency": 0.93,
                "photometric_error": 0.04,
                "reprojection_error_px": 0.45,
                "reprojection_valid_ratio": 0.82,
            },
            "geometry_failures": [] if partition != "rejected" else ["depth_consistency"],
            "sharp_failures": [] if partition == "sharp_accepted" else ["sharpness_gain"],
        }
        if partition == "sharp_accepted":
            record["rgb_path"] = str(self.generated[target_id])
            record["failures"] = []
        elif partition == "geometry_only":
            record["rgb_path"] = str(self.generated[target_id])
            record["failures"] = ["sharpness_gain"]
        else:
            record["candidate_rgb_path"] = str(self.generated[target_id])
            record["failures"] = ["sharpness_gain", "depth_consistency"]
        return record

    def _write_advanced_snapshot(self) -> None:
        target_names = ("target_sharp", "target_geometry", "target_rejected")
        planned = [
            {
                "target_id": target_id,
                "left_index": 0,
                "right_index": 3,
                "batch_id": "batch_00000",
                "conditioning": self.conditioning,
                "reasons": ["low_depth_overlap"],
            }
            for target_id in target_names
        ]
        sharp = self._make_outcome("target_sharp", "sharp_accepted")
        geometry = self._make_outcome("target_geometry", "geometry_only")
        rejected = self._make_outcome("target_rejected", "rejected")
        report = {
            "schema": REPORT_SCHEMA,
            "scene": "fr2_xyz_test",
            "preprocess_signature": self.signature,
            "generation_id": self.generation_id,
            "acceptance_mode": "geometry",
            "source_frame_count": 4,
            "selected_target_count": 3,
            "accepted_target_count": 2,
            "rejected_target_count": 1,
            "manifest": str(self.manifest_path),
            "planned": planned,
            "accepted": [sharp, geometry],
            "rejected": [rejected],
            "generation_batches": [
                {
                    "batch_id": "batch_00000",
                    "target_ids": list(target_names),
                    "conditioning": self.conditioning,
                }
            ],
            "quality_partition": {
                "sharp_accepted": [sharp],
                "geometry_only": [geometry],
                "rejected": [rejected],
            },
        }
        frames: list[dict[str, object]] = [
            {
                "kind": "original",
                "source_index": index,
                "timestamp": float(index),
                "rgb_path": str(path),
                "depth_path": None,
            }
            for index, path in enumerate(self.originals)
        ]
        for target_id, partition in (
            ("target_sharp", "sharp_accepted"),
            ("target_geometry", "geometry_only"),
        ):
            frames.append(
                {
                    "kind": "synthetic",
                    "target_id": target_id,
                    "rgb_path": str(self.generated[target_id]),
                    "acceptance_class": partition,
                }
            )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "preprocess_signature": self.signature,
            "generation_id": self.generation_id,
            "source_frame_count": 4,
            "generated_frame_count": 2,
            "preprocess_report_path": str(self.report_path),
            "frames": frames,
        }
        self.report_path.write_text(json.dumps(report), encoding="utf-8")
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_cli_renders_all_partitions_and_auditable_ratios(self) -> None:
        output = self.root / "visualization"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(self.report_path),
                "--output-dir",
                str(output),
                "--crop-size",
                "16",
                "--conditioning-panel-width",
                "100",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        stdout = json.loads(completed.stdout)
        self.assertEqual(stdout["visualized_target_count"], 3)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["scene"], "fr2_xyz_test")
        self.assertEqual(
            (
                summary["sharp_accepted_count"],
                summary["geometry_only_count"],
                summary["rejected_count"],
            ),
            (1, 1, 1),
        )
        self.assertEqual(summary["actual_injected_count"], 2)
        self.assertEqual(summary["actual_injected_sharp_count"], 1)
        self.assertEqual(summary["actual_injected_geometry_only_count"], 1)
        self.assertAlmostEqual(summary["actual_injected_over_source"], 0.5)
        self.assertAlmostEqual(summary["actual_injected_over_source_plus_injected"], 1 / 3)
        self.assertAlmostEqual(summary["actual_injected_over_selected"], 2 / 3)
        # Conditioning is counted once for the shared generation batch.  The
        # intentional HYBRID raw role is not an EVSSM attempt.
        self.assertEqual(summary["conditioning_view_count_unique_batches"], 3)
        self.assertEqual(summary["evssm_attempted_count_unique_batches"], 2)
        self.assertEqual(summary["evssm_fallback_count_unique_batches"], 1)
        self.assertAlmostEqual(summary["evssm_fallback_ratio"], 0.5)

        by_partition = {item["quality_partition"]: item for item in summary["targets"]}
        expected_colors = {
            "sharp_accepted": (29, 150, 73),
            "geometry_only": (220, 155, 0),
            "rejected": (210, 48, 48),
        }
        for partition, target in by_partition.items():
            self.assertEqual(target["gate_support_source_indices"], [1, 2])
            self.assertEqual(target["conditioning_count"], 3)
            self.assertEqual(Path(target["left_support_path"]), self.originals[1])
            for key in (
                "conditioning_sheet_path",
                "gate_full_path",
                "gate_detail100_path",
            ):
                path = Path(target[key])
                self.assertTrue(path.is_absolute() and path.is_file())
                with Image.open(path) as image:
                    self.assertEqual(image.getpixel((0, 0)), expected_colors[partition])
            with Image.open(target["conditioning_sheet_path"]) as contact:
                self.assertEqual(contact.width, 300)
            with Image.open(target["gate_full_path"]) as full:
                self.assertEqual(full.width, 64 * 3)
            with Image.open(target["gate_detail100_path"]) as detail:
                self.assertEqual(detail.width, 16 * 3)
            self.assertEqual(target["actually_injected"], partition != "rejected")

        with (output / "summary.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["actual_injected_count"], "2")
        self.assertEqual(rows[0]["evssm_fallback_ratio"], "0.5")

    def test_manifest_input_resolves_paired_report_and_target_filter(self) -> None:
        output = self.root / "manifest_input"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(self.manifest_path),
                "--output-dir",
                str(output),
                "--target",
                "target_geometry",
                "--crop-size",
                "12",
                "--conditioning-panel-width",
                "80",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(completed.stdout)["visualized_target_count"], 1)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(summary["targets"]), 1)
        self.assertEqual(summary["targets"][0]["quality_partition"], "geometry_only")

    def test_old_report_fails_with_actionable_message_by_default(self) -> None:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        report.pop("quality_partition")
        self.report_path.write_text(json.dumps(report), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(self.report_path),
                "--output-dir",
                str(self.root / "legacy_failure"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("no advanced quality_partition", completed.stderr)
        self.assertIn("--legacy-policy compat", completed.stderr)


if __name__ == "__main__":
    unittest.main()
