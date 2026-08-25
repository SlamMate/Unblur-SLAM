#!/usr/bin/env python3
"""CPU contract tests for ATE filtering and FrameCrafter trajectory export."""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framecrafter_pipeline import load_frames_csv  # noqa: E402
from src.utils.datasets import TUM_RGB  # noqa: E402
from src.utils.eval_traj import (  # noqa: E402
    align_full_traj,
    align_kf_traj,
    build_evaluation_trajectory_pairs,
)


def _load_export_module():
    path = ROOT / "scripts" / "export_framecrafter_trajectory.py"
    spec = importlib.util.spec_from_file_location(
        "export_framecrafter_trajectory", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPORT = _load_export_module()


def pose(tx: float = 0.0, angle_deg: float = 0.0) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    result[0, 3] = tx
    return result


def pose_knots(midpoint_tx: float) -> np.ndarray:
    return np.stack([pose(midpoint_tx - 1.0), pose(midpoint_tx + 1.0)])


class GuardedPoses:
    """Pose table that proves eval=false rows are never accessed as GT."""

    def __init__(self, values: list[np.ndarray], forbidden: set[int]):
        self.values = values
        self.forbidden = forbidden
        self.read_indices: list[int] = []

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> np.ndarray:
        index = int(index)
        if index in self.forbidden:
            raise AssertionError(f"synthetic pose {index} was read as ground truth")
        self.read_indices.append(index)
        return self.values[index]


class EvaluationPairingTests(unittest.TestCase):
    def test_eval_false_is_filtered_before_gt_access_and_pairing_stays_positional(self):
        stream = type("Stream", (), {})()
        stream.poses = GuardedPoses(
            [pose_knots(1.0), pose_knots(999.0), pose_knots(3.0), pose_knots(4.0)],
            forbidden={1},
        )
        stream.frame_metadata = [
            {"eval": True, "synthetic": False},
            {"eval": False, "synthetic": True},
            {"eval": True, "synthetic": False},
            {"eval": True, "synthetic": False},
        ]
        # Deliberately non-monotonic indices catch zip/order mistakes.  The
        # estimate at position 1 is the synthetic frame and must be removed
        # from both sides of the pair before its poison GT pose is read.
        estimates = np.stack([pose(20.0), pose(999.0), pose(10.0), pose(30.0)])
        paired_est, paired_ref, timestamps, kept = build_evaluation_trajectory_pairs(
            estimates, [2, 1, 0, 3], stream
        )

        np.testing.assert_allclose(paired_est[:, 0, 3], [20.0, 10.0, 30.0])
        np.testing.assert_allclose(paired_ref[:, 0, 3], [3.0, 1.0, 4.0])
        np.testing.assert_allclose(timestamps, [2.0, 0.0, 3.0])
        np.testing.assert_array_equal(kept, [2, 0, 3])
        self.assertEqual(stream.poses.read_indices, [2, 0, 3])

    def test_pairing_rejects_ambiguous_or_misaligned_indices(self):
        stream = type("Stream", (), {})()
        stream.poses = [pose_knots(0.0), pose_knots(1.0)]
        stream.frame_metadata = [{"eval": True}, {"eval": True}]

        with self.assertRaisesRegex(ValueError, "length mismatch"):
            build_evaluation_trajectory_pairs(np.stack([pose(), pose()]), [0], stream)
        with self.assertRaisesRegex(ValueError, "dataset index"):
            build_evaluation_trajectory_pairs(np.stack([pose()]), [0.25], stream)
        with self.assertRaises(IndexError):
            build_evaluation_trajectory_pairs(np.stack([pose()]), [2], stream)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_evaluation_trajectory_pairs(
                np.stack([pose(), pose()]), [0, 0], stream
            )

    def test_public_full_and_keyframe_alignment_both_apply_eval_filter(self):
        reference_poses = []
        for xyz in ((0, 0, 0), (99, 99, 99), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
            value = np.eye(4, dtype=np.float64)
            value[:3, 3] = xyz
            reference_poses.append(value)
        stream = type("Stream", (), {})()
        stream.poses = GuardedPoses(
            [np.stack([value, value]) for value in reference_poses], forbidden={1}
        )
        stream.frame_metadata = [{"eval": index != 1} for index in range(5)]
        estimates = np.stack(reference_poses)

        _, _, _, full_est, full_ref = align_full_traj(estimates, stream, None)
        np.testing.assert_allclose(full_est.timestamps, [0.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(full_ref.timestamps, [0.0, 2.0, 3.0, 4.0])

        with tempfile.TemporaryDirectory() as directory:
            keyframes = Path(directory) / "keyframes.npz"
            np.savez(keyframes, poses=estimates, timestamps=np.arange(5))
            _, _, _, kf_est, kf_ref = align_kf_traj(keyframes, stream)
        np.testing.assert_allclose(kf_est.timestamps, [0.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(kf_ref.timestamps, [0.0, 2.0, 3.0, 4.0])


class FrameCrafterTrajectoryExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "rgb").mkdir()
        (self.root / "depth").mkdir()

        rgb_rows = ["# timestamp filename"]
        depth_rows = ["# timestamp filename"]
        for index, timestamp in enumerate((0.0, 0.02, 0.04, 0.08)):
            rgb = self.root / "rgb" / f"{index:06d}.png"
            depth = self.root / "depth" / f"{index:06d}.png"
            rgb.touch()
            depth.touch()
            rgb_rows.append(f"{timestamp:.3f} rgb/{rgb.name}")
            depth_rows.append(f"{timestamp + 0.003:.3f} depth/{depth.name}")
        (self.root / "rgb.txt").write_text("\n".join(rgb_rows) + "\n", encoding="utf-8")
        (self.root / "depth.txt").write_text(
            "\n".join(depth_rows) + "\n", encoding="utf-8"
        )
        # The export must succeed even though a normal TUM loader could not
        # parse this file.  It is evaluation-only and must never supply poses.
        (self.root / "groundtruth.txt").write_text(
            "THIS FILE MUST NOT BE READ BY THE EXPORTER\n", encoding="utf-8"
        )

        self.poses = np.stack([pose(0.25), pose(1.5, 90.0), pose(2.75, -30.0)])
        self.trajectory = self.root / "traj_full_full_traj.npz"
        np.savez(
            self.trajectory,
            traj_est_not_align=self.poses,
            traj_est_not_align_timestamps=np.arange(3, dtype=np.float64),
            traj_est_not_align_eval_mask=np.ones(3, dtype=np.bool_),
            # A poison reference trajectory demonstrates which array won.
            traj_ref_poses=np.stack([pose(900.0), pose(901.0), pose(902.0)]),
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_cli_export_uses_unaligned_estimate_and_exact_tum_rgb_depth_order(self):
        output = self.root / "framecrafter_frames.csv"
        exit_code = EXPORT.main(
            [
                "--trajectory-npz", str(self.trajectory),
                "--tum-root", str(self.root),
                "--output", str(output),
                "--fx", "525",
                "--fy", "526",
                "--cx", "319.5",
                "--cy", "239.5",
            ]
        )
        self.assertEqual(exit_code, 0)

        frames = load_frames_csv(
            output,
            pose_convention="c2w",
            compute_missing_sharpness=False,
            expected_pose_source="droid_traj_est_not_align",
            require_pose_provenance=True,
        )
        self.assertEqual([frame.source_index for frame in frames], [0, 1, 2])
        np.testing.assert_allclose([frame.timestamp for frame in frames], [0.0, 0.04, 0.08])
        for actual, expected in zip(frames, self.poses):
            np.testing.assert_allclose(actual.c2w, expected, atol=1.0e-10)
            self.assertTrue(actual.eval)
            self.assertTrue(actual.rgb_path.is_absolute())
            self.assertTrue(actual.depth_path.is_absolute())
        self.assertEqual(frames[1].rgb_path.name, "000002.png")

        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["pose_source"] for row in rows}, {"droid_traj_est_not_align"})
        self.assertEqual({row["uses_ground_truth_pose"] for row in rows}, {"false"})
        self.assertEqual({row["trajectory_key"] for row in rows}, {"traj_est_not_align"})

        mismatched_csv = self.root / "mismatched_estimate.csv"
        mismatched_rows = [dict(row) for row in rows]
        mismatched_rows[0]["tx"] = "123"
        with mismatched_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(mismatched_rows)
        with self.assertRaisesRegex(ValueError, "does not numerically match"):
            load_frames_csv(
                mismatched_csv,
                pose_convention="c2w",
                compute_missing_sharpness=False,
                expected_pose_source="droid_traj_est_not_align",
                require_pose_provenance=True,
            )

        # The CSV is bound to the actual trajectory bytes, not merely a safe
        # filename/string. Replacing the NPZ invalidates production loading.
        with self.trajectory.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "trajectory_path/trajectory_sha256"):
            load_frames_csv(
                output,
                pose_convention="c2w",
                compute_missing_sharpness=False,
                expected_pose_source="droid_traj_est_not_align",
                require_pose_provenance=True,
            )

    def test_float32_se3_quaternion_canonicalisation_keeps_pose_binding(self):
        # Real lietorch trajectories are float32 and can carry a few 1e-7 of
        # orthogonality error.  CSV uses a unit quaternion, so loading projects
        # that harmless error back onto SO(3) and must not reject its own
        # exporter output.
        float32_poses = self.poses.astype(np.float32)
        float32_poses[1, 0, 0] += np.float32(2.0e-7)
        np.savez(
            self.trajectory,
            traj_est_not_align=float32_poses,
            traj_est_not_align_timestamps=np.arange(3, dtype=np.float64),
            traj_est_not_align_eval_mask=np.ones(3, dtype=np.bool_),
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
        )
        output = self.root / "float32_framecrafter_frames.csv"
        exit_code = EXPORT.main(
            [
                "--trajectory-npz", str(self.trajectory),
                "--tum-root", str(self.root),
                "--output", str(output),
                "--fx", "525", "--fy", "526",
                "--cx", "319.5", "--cy", "239.5",
            ]
        )
        self.assertEqual(exit_code, 0)
        frames = load_frames_csv(
            output,
            pose_convention="c2w",
            compute_missing_sharpness=False,
            expected_pose_source="droid_traj_est_not_align",
            require_pose_provenance=True,
        )
        self.assertEqual(len(frames), 3)
        np.testing.assert_allclose(frames[1].c2w, float32_poses[1], atol=5.0e-7)

    def test_tum_loader_preserves_selected_rgb_timestamps(self):
        tum = self.root / "valid_tum"
        (tum / "rgb").mkdir(parents=True)
        (tum / "depth").mkdir()
        rgb_rows = []
        depth_rows = []
        gt_rows = ["# timestamp tx ty tz qx qy qz qw"]
        for index, timestamp in enumerate((0.0, 0.04, 0.08)):
            rgb = tum / "rgb" / f"{index}.png"
            depth = tum / "depth" / f"{index}.png"
            rgb.write_bytes(b"rgb")
            depth.write_bytes(b"depth")
            rgb_rows.append(f"{timestamp:.3f} rgb/{index}.png")
            depth_rows.append(f"{timestamp + 0.003:.3f} depth/{index}.png")
            gt_rows.append(f"{timestamp:.3f} {index} 0 0 0 0 0 1")
        (tum / "rgb.txt").write_text("\n".join(rgb_rows) + "\n", encoding="utf-8")
        (tum / "depth.txt").write_text(
            "\n".join(depth_rows) + "\n", encoding="utf-8"
        )
        (tum / "groundtruth.txt").write_text(
            "\n".join(gt_rows) + "\n", encoding="utf-8"
        )
        loader = object.__new__(TUM_RGB)
        loader.num_control_knots = 2
        images, depths, poses, timestamps = TUM_RGB.loadtum(
            loader, str(tum), frame_rate=32
        )
        self.assertEqual(len(images), 3)
        self.assertEqual(len(depths), 3)
        self.assertEqual(len(poses), 3)
        np.testing.assert_allclose(timestamps, [0.0, 0.04, 0.08])

    def test_loader_never_falls_back_to_reference_or_aligned_trajectory(self):
        reference_only = self.root / "reference_only.npz"
        np.savez(reference_only, traj_ref_poses=self.poses)
        with self.assertRaisesRegex(KeyError, "no explicitly unaligned"):
            EXPORT.load_unaligned_trajectory(reference_only)

        aligned_only = self.root / "aligned_only.npz"
        np.savez(aligned_only, traj_est_poses=self.poses)
        with self.assertRaisesRegex(KeyError, "no explicitly unaligned"):
            EXPORT.load_unaligned_trajectory(aligned_only)
        with self.assertRaisesRegex(ValueError, "not an allowed unaligned"):
            EXPORT.load_unaligned_trajectory(
                self.trajectory, trajectory_key="traj_est_poses"
            )

        unsafe_source = self.root / "unsafe_source.npz"
        np.savez(
            unsafe_source,
            traj_est_not_align=self.poses,
            pose_source=np.asarray("aligned_to_gt"),
            uses_ground_truth_pose=np.asarray(False),
        )
        with self.assertRaisesRegex(ValueError, "non-GT"):
            EXPORT.load_unaligned_trajectory(unsafe_source)

        missing_source = self.root / "missing_source.npz"
        np.savez(
            missing_source,
            traj_est_not_align=self.poses,
            uses_ground_truth_pose=np.asarray(False),
        )
        with self.assertRaisesRegex(ValueError, "declare pose_source"):
            EXPORT.load_unaligned_trajectory(missing_source)

    def test_declared_gt_bad_indices_and_length_mismatch_are_hard_failures(self):
        declared_gt = self.root / "declared_gt.npz"
        np.savez(
            declared_gt,
            traj_est_not_align=self.poses,
            uses_ground_truth_pose=np.asarray(True),
        )
        with self.assertRaisesRegex(ValueError, "uses_ground_truth_pose=true"):
            EXPORT.load_unaligned_trajectory(declared_gt)

        bad_indices = self.root / "bad_indices.npz"
        np.savez(
            bad_indices,
            traj_est_not_align=self.poses,
            traj_est_not_align_timestamps=np.asarray([0.0, 2.0, 1.0]),
            traj_est_not_align_eval_mask=np.ones(3, dtype=np.bool_),
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
        )
        with self.assertRaisesRegex(ValueError, "dataset indices 0..N-1"):
            EXPORT.load_unaligned_trajectory(bad_indices)

        second_pass = self.root / "second_pass_with_synthetic.npz"
        np.savez(
            second_pass,
            traj_est_not_align=self.poses,
            traj_est_not_align_timestamps=np.arange(3, dtype=np.float64),
            traj_est_not_align_eval_mask=np.asarray([True, False, True]),
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
        )
        with self.assertRaisesRegex(ValueError, "eval=false synthetic"):
            EXPORT.load_unaligned_trajectory(second_pass)

        associations = EXPORT.select_dataset_associations(
            EXPORT.associate_tum_rgb_depth(
                EXPORT.read_tum_list(self.root / "rgb.txt", self.root),
                EXPORT.read_tum_list(self.root / "depth.txt", self.root),
            )
        )
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            EXPORT.write_framecrafter_csv(
                self.root / "mismatch.csv",
                self.poses[:2],
                associations,
                {"fx": 525, "fy": 526, "cx": 319.5, "cy": 239.5},
                trajectory_path=self.trajectory,
                trajectory_key="traj_est_not_align",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
