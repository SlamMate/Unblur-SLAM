#!/usr/bin/env python3
"""CPU-only tests for FrameCrafter result summaries and evidence images."""

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
SCRIPT = ROOT / "scripts" / "summarize_framecrafter_results.py"
REPORT_SCHEMA = "unblur_slam.framecrafter_preprocess_report.v1"
MANIFEST_SCHEMA = "unblur_slam.framecrafter_manifest.v1"


def write_rgb(path: Path, base: int, *, detail_x: int) -> None:
    array = np.full((48, 64, 3), base, dtype=np.uint8)
    array[12:36, detail_x : detail_x + 12] = 255 - base
    array[12:36:2, detail_x : detail_x + 12] = base // 2
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def original(index: int, rgb_path: Path) -> dict[str, object]:
    return {
        "kind": "original",
        "source_index": index,
        "timestamp": float(index),
        "rgb_path": str(rgb_path.resolve()),
        "depth_path": None,
    }


class SummarizeFrameCrafterResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "source_frames"
        self.sources.mkdir()
        self.source_paths: list[Path] = []
        for index, value in enumerate((30, 70, 110)):
            path = self.sources / f"source_{index}.png"
            write_rgb(path, value, detail_x=20)
            self.source_paths.append(path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_scene(
        self,
        scene: str,
        *,
        accepted_gains: list[float],
        rejected_gains: list[float],
        planned_total: int,
        source_count: int = 3,
    ) -> tuple[Path, Path]:
        snapshot = self.root / scene / "immutable_snapshot"
        artifact_root = snapshot / "artifacts"
        signature = ("a" if accepted_gains else "b") * 64
        generation_id = ("1" if accepted_gains else "2") * 32
        report_path = snapshot / f"preprocess_report_{signature}_{generation_id}.json"
        manifest_path = snapshot / f"manifest_{signature}_{generation_id}.json"
        planned: list[dict[str, object]] = []
        accepted: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        synthetic_frames: list[dict[str, object]] = []

        outcome_gains = [("accepted", gain) for gain in accepted_gains]
        outcome_gains.extend(("rejected", gain) for gain in rejected_gains)
        for index, (status, gain) in enumerate(outcome_gains):
            target_id = f"{scene}_target_{index}"
            reasons = ["consecutive_blurry_region"]
            if index == 0:
                reasons.append("large_pose_gap")
            planned.append(
                {
                    "target_id": target_id,
                    "left_index": index % 2,
                    "right_index": index % 2 + 1,
                    "alpha": 0.5,
                    "reasons": reasons,
                }
            )
            metrics = {
                "sharpness_gain": gain,
                "depth_consistency": 0.91 - index * 0.01,
                "photometric_error": 0.03 + index * 0.01,
                "reprojection_error_px": 0.4 + index * 0.1,
            }
            image_path = artifact_root / status / f"{target_id}.png"
            write_rgb(image_path, 130 + index * 10, detail_x=20)
            # Simulate a relocated immutable snapshot whose recorded artifact
            # paths still point at the machine on which it was generated.
            stale_path = f"/old/machine/run/artifacts/{status}/{target_id}.png"
            if status == "accepted":
                accepted.append(
                    {"target_id": target_id, "metrics": metrics, "rgb_path": stale_path}
                )
                synthetic_frames.append(
                    {
                        "kind": "synthetic",
                        "target_id": target_id,
                        "rgb_path": stale_path,
                        "left_index": index % 2,
                        "right_index": index % 2 + 1,
                        "reasons": reasons,
                        "gate_metrics": metrics,
                    }
                )
            else:
                rejected.append(
                    {
                        "target_id": target_id,
                        "failures": ["sharpness_gain"],
                        "metrics": metrics,
                        "candidate_rgb_path": stale_path,
                    }
                )

        report = {
            "schema": REPORT_SCHEMA,
            "backend": "python_api",
            "preprocess_signature": signature,
            "generation_id": generation_id,
            "source_frame_count": source_count,
            "planned_total_before_cap": planned_total,
            "planned_target_count": len(planned),
            "selected_target_count": len(planned),
            "accepted_target_count": len(accepted),
            "rejected_target_count": len(rejected),
            "manifest": f"/stale/snapshot/{manifest_path.name}",
            "planned": planned,
            "accepted": accepted,
            "rejected": rejected,
        }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "preprocess_signature": signature,
            "generation_id": generation_id,
            "source_frame_count": source_count,
            "generated_frame_count": len(accepted),
            "preprocess_report_path": f"/stale/snapshot/{report_path.name}",
            "frames": [
                *(
                    original(index, self.source_paths[index % len(self.source_paths)])
                    for index in range(source_count)
                ),
                *synthetic_frames,
            ],
        }
        snapshot.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return snapshot, manifest_path

    def test_cli_summarises_ratios_ranking_and_rejected_fallback(self) -> None:
        accepted_snapshot, _ = self.make_scene(
            "fr1_desk",
            accepted_gains=[1.20, 1.45],
            rejected_gains=[0.95],
            planned_total=5,
        )
        _, rejected_manifest = self.make_scene(
            "fr2_xyz",
            accepted_gains=[],
            rejected_gains=[0.82, 0.99],
            planned_total=4,
        )
        output = (self.root / "summary_output").resolve()
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--scene",
                f"fr1_desk={accepted_snapshot}",
                "--scene",
                f"fr2_xyz={rejected_manifest}",
                "--output-dir",
                str(output),
                "--top-k",
                "2",
                "--zoom-size",
                "16",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        stdout = json.loads(completed.stdout)
        self.assertEqual(stdout["scene_count"], 2)
        self.assertTrue(Path(stdout["summary_json"]).is_absolute())
        self.assertTrue(Path(stdout["summary_csv"]).is_absolute())

        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        by_scene = {scene["scene"]: scene for scene in summary["scenes"]}
        fr1 = by_scene["fr1_desk"]
        self.assertEqual(
            (fr1["source"], fr1["planned"], fr1["selected"]), (3, 5, 3)
        )
        self.assertEqual((fr1["accepted"], fr1["rejected"]), (2, 1))
        self.assertEqual(fr1["not_evaluated"], 2)
        self.assertEqual(fr1["selected_without_gate_outcome"], 0)
        self.assertAlmostEqual(fr1["accepted_over_source"], 2 / 3)
        self.assertAlmostEqual(fr1["accepted_over_source_plus_accepted"], 2 / 5)
        self.assertAlmostEqual(fr1["accepted_over_planned"], 2 / 5)
        self.assertEqual(
            [item["sharpness_gain"] for item in fr1["accepted_ranked_by_sharpness_gain"]],
            [1.45, 1.20],
        )
        self.assertEqual([item["status"] for item in fr1["visuals"]], ["ACCEPTED"] * 2)
        self.assertTrue(all(item["counted_as_added"] for item in fr1["visuals"]))
        self.assertEqual(fr1["selected_reason_counts"]["large_pose_gap"], 1)
        self.assertNotIn("planned_reason_counts", fr1)

        fr2 = by_scene["fr2_xyz"]
        self.assertEqual(fr2["accepted"], 0)
        self.assertEqual(fr2["accepted_over_source"], 0.0)
        self.assertEqual(len(fr2["visuals"]), 1)
        fallback = fr2["visuals"][0]
        self.assertEqual(fallback["status"], "REJECTED")
        self.assertFalse(fallback["counted_as_added"])
        self.assertEqual(fallback["metrics"]["sharpness_gain"], 0.99)
        self.assertIn("sharpness_gain", fallback["failures"])

        for scene in (fr1, fr2):
            self.assertTrue(Path(scene["report_path"]).is_absolute())
            self.assertTrue(Path(scene["manifest_path"]).is_absolute())
            for visual in scene["visuals"]:
                full = Path(visual["full_triptych_path"])
                zoom = Path(visual["zoom100_triptych_path"])
                self.assertTrue(full.is_absolute() and full.is_file())
                self.assertTrue(zoom.is_absolute() and zoom.is_file())
                expected_accent = (
                    (29, 150, 73)
                    if visual["status"] == "ACCEPTED"
                    else (210, 48, 48)
                )
                with Image.open(full) as full_image, Image.open(zoom) as zoom_image:
                    self.assertEqual(full_image.width, 64 * 3)
                    self.assertEqual(zoom_image.width, 16 * 3)
                    self.assertEqual(full_image.getpixel((0, 0)), expected_accent)

        with (output / "summary.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["scene"] for row in rows], ["fr1_desk", "fr2_xyz"])
        self.assertEqual(rows[1]["best_visual_status"], "REJECTED")
        self.assertEqual(rows[1]["best_visual_sharpness_gain"], "0.99")

    def test_real_probe_shape_counts_pre_cap_targets_as_not_evaluated(self) -> None:
        snapshot, _ = self.make_scene(
            "fr1_real_probe",
            accepted_gains=[1.06],
            rejected_gains=[0.96, 0.91, 0.88],
            planned_total=135,
            source_count=592,
        )
        output = (self.root / "probe_summary").resolve()
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--scene",
                f"fr1_real_probe={snapshot}",
                "--output-dir",
                str(output),
                "--top-k",
                "1",
                "--zoom-size",
                "16",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        scene = json.loads(
            (output / "summary.json").read_text(encoding="utf-8")
        )["scenes"][0]
        self.assertEqual(scene["source"], 592)
        self.assertEqual(scene["planned"], 135)
        self.assertEqual(scene["selected"], 4)
        self.assertEqual(scene["not_evaluated"], 131)
        self.assertEqual(scene["selected_without_gate_outcome"], 0)
        self.assertAlmostEqual(scene["accepted_over_planned"], 1 / 135)
        self.assertAlmostEqual(scene["accepted_over_selected"], 1 / 4)
        self.assertIn("selected_reason_counts", scene)
        self.assertNotIn("planned_reason_counts", scene)

        with (output / "summary.csv").open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["not_evaluated"], "131")
        self.assertEqual(row["selected_without_gate_outcome"], "0")


if __name__ == "__main__":
    unittest.main()
