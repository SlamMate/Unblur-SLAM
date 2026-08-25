#!/usr/bin/env python3
"""Pure-CPU synthetic tests for the FrameCrafter preprocessing pipeline."""

from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framecrafter_pipeline import (  # noqa: E402
    FrameRecord,
    FrameCrafterGenerationBatch,
    GateConfig,
    SyntheticFrameResult,
    TargetView,
    TestOnlyBlendBackend,
    build_manifest,
    c2w_to_opencv_w2c,
    evaluate_candidate,
    framecrafter_input_arrays,
    fuse_bilateral_depth,
    interpolate_c2w,
    load_frames_csv,
    plan_interpolated_targets,
    plan_framecrafter_generation_batches,
    save_framecrafter_npz,
    select_real_contexts,
    select_scene_wide_targets,
    select_shared_real_contexts,
    targets_from_planner_json,
    validate_pose_source,
    validate_manifest_payload,
    write_manifest,
)
from scripts.run_framecrafter_preprocess import (  # noqa: E402
    compute_preprocess_signature,
)


def checkerboard(height: int = 12, width: int = 16) -> np.ndarray:
    y, x = np.indices((height, width))
    base = ((x + y) % 2).astype(np.float32)
    return np.stack([base, 1.0 - base, 0.25 + 0.5 * base], axis=-1)


def save_test_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intrinsic(height: int = 12, width: int = 16) -> np.ndarray:
    return np.array(
        [[20.0, 0.0, (width - 1) / 2], [0.0, 20.0, (height - 1) / 2], [0, 0, 1]],
        dtype=np.float64,
    )


def pose(center_x: float = 0.0, angle_deg: float = 0.0) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[0, 3] = center_x
    return result


class FrameCrafterPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = checkerboard()
        self.k = intrinsic()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_frames(self, count: int = 7) -> list[FrameRecord]:
        frames = []
        for index in range(count):
            rgb = self.root / f"rgb_{index:03d}.png"
            depth = self.root / f"depth_{index:03d}.npy"
            save_test_rgb(rgb, self.image)
            np.save(depth, np.ones(self.image.shape[:2], dtype=np.float32))
            frames.append(
                FrameRecord(
                    source_index=index,
                    frame_id=rgb.name,
                    timestamp=float(index),
                    rgb_path=rgb,
                    depth_path=depth,
                    c2w=pose(0.01 * index),
                    intrinsics=self.k,
                    sharpness=float(index + 1),
                )
            )
        return frames

    def make_target(self, frames: list[FrameRecord], left: int = 2, right: int = 3) -> TargetView:
        return TargetView(
            target_id="synthetic_test",
            left_index=frames[left].source_index,
            right_index=frames[right].source_index,
            left_position=left,
            right_position=right,
            timestamp=(frames[left].timestamp + frames[right].timestamp) / 2,
            alpha=0.5,
            c2w=interpolate_c2w(frames[left].c2w, frames[right].c2w, 0.5),
            intrinsics=self.k,
            reasons=("consecutive_blurry_region",),
        )

    def test_camera_center_and_slerp_interpolation(self) -> None:
        middle = interpolate_c2w(pose(0.0, 0.0), pose(2.0, 90.0), 0.5)
        np.testing.assert_allclose(middle[:3, 3], [1.0, 0.0, 0.0], atol=1e-7)
        expected = pose(1.0, 45.0)
        np.testing.assert_allclose(middle[:3, :3], expected[:3, :3], atol=1e-7)
        np.testing.assert_allclose(c2w_to_opencv_w2c(middle) @ middle, np.eye(4), atol=1e-6)

    def test_load_w2c_csv_converts_translation_to_camera_center(self) -> None:
        rgb = self.root / "frame.png"
        save_test_rgb(rgb, self.image)
        csv_path = self.root / "frames.csv"
        csv_path.write_text(
            "frame,timestamp,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,laplacian\n"
            f"{rgb.name},0,-1,0,0,0,0,0,1,20,20,7.5,5.5,2\n",
            encoding="utf-8",
        )
        frame = load_frames_csv(
            csv_path, image_root=self.root, pose_convention="w2c"
        )[0]
        np.testing.assert_allclose(frame.c2w[:3, 3], [1, 0, 0], atol=1e-7)

    def test_direct_and_json_planning_create_real_target_poses(self) -> None:
        frames = self.make_frames(3)
        frames[0].sharpness = 0.0
        frames[1].sharpness = 0.0
        frames[2].sharpness = 10.0
        planned = plan_interpolated_targets(
            frames,
            laplacian_threshold=1.0,
            translation_step=10.0,
            rotation_step_deg=180.0,
            blur_region_inserts=1,
        )
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].left_index, 0)
        self.assertIn("consecutive_blurry_region", planned[0].reasons)

        payload = {
            "segments": [
                {
                    "left_frame": frames[1].frame_id,
                    "right_frame": frames[2].frame_id,
                    "alphas": [0.25, 0.75],
                    "reasons": ["large_pose_gap"],
                }
            ]
        }
        from_json = targets_from_planner_json(payload, frames)
        self.assertEqual(len(from_json), 2)
        self.assertAlmostEqual(from_json[0].c2w[0, 3], 0.0125)
        self.assertAlmostEqual(from_json[1].c2w[0, 3], 0.0175)

    def test_context_selection_npz_and_test_backend_contract(self) -> None:
        frames = self.make_frames()
        target = self.make_target(frames)
        contexts = select_real_contexts(frames, target, context_count=4, min_contexts=3)
        self.assertEqual(len(contexts), 4)
        self.assertIn(target.left_index, {frame.source_index for frame in contexts})
        self.assertIn(target.right_index, {frame.source_index for frame in contexts})
        w2c, ks = framecrafter_input_arrays(contexts, [target])
        self.assertEqual(w2c.shape, (5, 4, 4))
        self.assertEqual(ks.shape, (5, 3, 3))
        npz_path = save_framecrafter_npz(self.root / "input.npz", contexts, [target])
        data = np.load(npz_path)
        self.assertEqual(set(data.files), {"w2c_poses", "intrinsics"})
        with self.assertRaises(RuntimeError):
            TestOnlyBlendBackend()
        blended = TestOnlyBlendBackend(allow_test_only=True).generate(contexts, target)
        np.testing.assert_allclose(blended, self.image, atol=1 / 255 + 1e-6)

    def test_scene_wide_cap_and_safe_greedy_batches(self) -> None:
        frames = self.make_frames(13)

        def target(left: int, reasons: tuple[str, ...], ordinal: int) -> TargetView:
            right = left + 1
            return TargetView(
                target_id=f"target_{ordinal:02d}",
                left_index=frames[left].source_index,
                right_index=frames[right].source_index,
                left_position=left,
                right_position=right,
                timestamp=(frames[left].timestamp + frames[right].timestamp) / 2,
                alpha=0.5,
                c2w=interpolate_c2w(frames[left].c2w, frames[right].c2w, 0.5),
                intrinsics=self.k,
                reasons=reasons,
            )

        all_targets = [
            target(
                index,
                ("large_pose_gap",) if index in (4, 7) else ("consecutive_blurry_region",),
                index,
            )
            for index in range(12)
        ]
        capped = select_scene_wide_targets(all_targets, 5)
        self.assertEqual(len(capped), 5)
        self.assertEqual([item.timestamp for item in capped], sorted(item.timestamp for item in capped))
        capped_ids = {item.target_id for item in capped}
        self.assertIn("target_00", capped_ids)
        self.assertIn("target_11", capped_ids)
        self.assertIn("target_04", capped_ids)
        self.assertIn("target_07", capped_ids)

        priority_only = [
            target(index, ("large_pose_gap",), index) for index in range(6)
        ]
        sampled_priority = select_scene_wide_targets(priority_only, 3)
        self.assertEqual(len(sampled_priority), 3)
        self.assertEqual(sampled_priority[0].target_id, "target_00")
        self.assertEqual(sampled_priority[-1].target_id, "target_05")
        self.assertTrue(
            all("large_pose_gap" in item.reasons for item in sampled_priority)
        )

        sparse_targets = [
            target(0, ("large_pose_gap",), 20),
            target(2, ("large_pose_gap",), 21),
            target(4, ("large_pose_gap",), 22),
            target(6, ("large_pose_gap",), 23),
            target(7, ("large_pose_gap",), 24),
        ]
        batches = plan_framecrafter_generation_batches(
            frames, sparse_targets, context_count=6, min_contexts=3
        )
        self.assertEqual([len(batch.targets) for batch in batches], [3, 2])
        for batch in batches:
            self.assertIsInstance(batch, FrameCrafterGenerationBatch)
            self.assertLessEqual(len(batch.targets), 4)
            self.assertLessEqual(len(batch.contexts) + len(batch.targets), 10)
            context_indices = {frame.source_index for frame in batch.contexts}
            for item in batch.targets:
                self.assertIn(item.left_index, context_indices)
                self.assertIn(item.right_index, context_indices)
        nearest = select_shared_real_contexts(
            frames, sparse_targets[-2:], context_count=6, min_contexts=3
        )
        self.assertEqual(len(nearest), 6)
        self.assertTrue(all(frame.kind == "original" for frame in nearest))

        # A scene-wide cap deliberately leaves temporally distant candidates;
        # they must not be mistaken for a local multi-target diffusion batch.
        far_frames = self.make_frames(41)
        far_targets = []
        for index in range(40):
            far_targets.append(
                TargetView(
                    target_id=f"far_{index:02d}",
                    left_index=index,
                    right_index=index + 1,
                    left_position=index,
                    right_position=index + 1,
                    timestamp=index + 0.5,
                    alpha=0.5,
                    c2w=interpolate_c2w(
                        far_frames[index].c2w, far_frames[index + 1].c2w, 0.5
                    ),
                    intrinsics=self.k,
                    reasons=("consecutive_blurry_region",),
                )
            )
        capped_far = select_scene_wide_targets(far_targets, 4)
        self.assertEqual(
            [item.target_id for item in capped_far],
            ["far_00", "far_13", "far_26", "far_39"],
        )
        far_batches = plan_framecrafter_generation_batches(
            far_frames, capped_far, context_count=6, min_contexts=3
        )
        self.assertEqual([len(batch.targets) for batch in far_batches], [1, 1, 1, 1])
        self.assertTrue(
            all(batch.endpoint_position_span <= 12 for batch in far_batches)
        )

        local_targets = []
        for ordinal, alpha in enumerate((0.2, 0.4, 0.6, 0.8)):
            local_targets.append(
                TargetView(
                    target_id=f"local_{ordinal}",
                    left_index=10,
                    right_index=11,
                    left_position=10,
                    right_position=11,
                    timestamp=10.0 + alpha,
                    alpha=alpha,
                    c2w=interpolate_c2w(
                        far_frames[10].c2w, far_frames[11].c2w, alpha
                    ),
                    intrinsics=self.k,
                    reasons=("large_pose_gap",),
                )
            )
        local_batches = plan_framecrafter_generation_batches(
            far_frames, local_targets, context_count=6, min_contexts=3
        )
        self.assertEqual([len(batch.targets) for batch in local_batches], [4])
        self.assertEqual(local_batches[0].endpoint_position_span, 1)

    def test_bilateral_depth_and_all_acceptance_gates(self) -> None:
        frames = self.make_frames(2)
        # Identity geometry makes expected projection/gating values exact.
        frames[0].c2w = np.eye(4)
        frames[1].c2w = np.eye(4)
        target = self.make_target(frames, 0, 1)
        depth = np.ones(self.image.shape[:2], dtype=np.float32)
        fusion = fuse_bilateral_depth(
            self.image,
            depth,
            frames[0].c2w,
            self.k,
            self.image,
            depth,
            frames[1].c2w,
            self.k,
            target.c2w,
            self.k,
        )
        self.assertAlmostEqual(fusion.metrics["depth_coverage"], 1.0)
        self.assertAlmostEqual(fusion.metrics["depth_consistency"], 1.0)
        result = evaluate_candidate(
            self.image,
            frames[0],
            frames[1],
            target,
            left_depth=depth,
            right_depth=depth,
            config=GateConfig(
                min_sharpness_gain=0.99,
                min_depth_coverage=0.9,
                min_depth_consistency=0.9,
                max_photometric_error=0.01,
                max_reprojection_error_px=0.1,
                min_reprojection_valid_ratio=0.9,
            ),
        )
        self.assertTrue(result.accepted, result.failures)
        self.assertGreater(result.confidence, 0.95)
        # Source PNGs are quantized to uint8 while the synthetic array remains
        # float, so the residual is bounded by one quantization step.
        self.assertLess(result.metrics["photometric_error"], 1.0 / 255.0)
        self.assertAlmostEqual(result.metrics["reprojection_error_px"], 0.0, places=6)

        rejected = evaluate_candidate(
            self.image,
            frames[0],
            frames[1],
            target,
            left_depth=depth,
            right_depth=2.0 * depth,
            config=GateConfig(min_sharpness_gain=0.9, min_depth_coverage=0.1),
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("depth_coverage", rejected.failures)

    def test_manifest_schema_order_and_gt_guard(self) -> None:
        frames = self.make_frames(2)
        target = self.make_target(frames, 0, 1)
        rgb_path = self.root / "synthetic.png"
        save_test_rgb(rgb_path, self.image)
        depth_path = self.root / "synthetic.npy"
        np.save(depth_path, np.ones(self.image.shape[:2], dtype=np.float32))
        synthetic = SyntheticFrameResult(
            target=target,
            rgb_path=rgb_path,
            depth_path=depth_path,
            confidence=0.8,
            source_ids=(frames[0].frame_id, frames[1].frame_id),
            gate_metrics={"sharpness_gain": 1.2},
        )
        manifest = build_manifest(
            frames, [synthetic], pose_source="droid_traj_est_not_align"
        )
        self.assertEqual(manifest["schema"], "unblur_slam.framecrafter_manifest.v1")
        self.assertFalse(manifest["uses_ground_truth_pose"])
        self.assertEqual([item["kind"] for item in manifest["frames"]],
                         ["original", "synthetic", "original"])
        generated = manifest["frames"][1]
        self.assertIsNone(generated["source_index"])
        self.assertFalse(generated["eval"])
        self.assertTrue(generated["fixed_pose"])
        self.assertTrue(Path(generated["rgb_path"]).is_absolute())
        manifest_path = write_manifest(self.root / "manifest.json", manifest)
        reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(reloaded["generated_frame_count"], 1)
        with self.assertRaises(ValueError):
            validate_pose_source("TUM_groundtruth")
        with self.assertRaises(ValueError):
            validate_pose_source("aligned_to_gt")

    def test_preprocess_signature_is_stable_and_invalidates_material_changes(self) -> None:
        image_root = self.root / "images"
        depth_root = self.root / "depth"
        base_model_dir = self.root / "base_model"
        framecrafter_repo = self.root / "framecrafter_repo"
        for directory in (image_root, depth_root, base_model_dir, framecrafter_repo):
            directory.mkdir()
        save_test_rgb(image_root / "a.png", self.image)
        np.save(depth_root / "a.npy", np.ones(self.image.shape[:2], dtype=np.float32))
        (framecrafter_repo / "model.py").write_text("VERSION = 1\n", encoding="utf-8")
        (base_model_dir / "config.json").write_text("{}\n", encoding="utf-8")
        frames_csv = self.root / "signature_frames.csv"
        header = "frame,timestamp,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,sharpness,depth_path"
        frames_csv.write_text(
            header + "\n" + "a.png,0,0,0,0,0,0,0,1,20,20,7.5,5.5,1,a.npy\n",
            encoding="utf-8",
        )
        planner_json = self.root / "signature_plan.json"
        planner_json.write_text('{"segments": []}\n', encoding="utf-8")
        checkpoint = self.root / "framecrafter.safetensors"
        checkpoint.write_bytes(b"checkpoint-v1")
        trajectory = self.root / "traj_full_full_traj.npz"
        np.savez(
            trajectory,
            traj_est_not_align=np.eye(4)[None],
            traj_est_not_align_timestamps=np.arange(1, dtype=np.float64),
            traj_est_not_align_eval_mask=np.ones(1, dtype=np.bool_),
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
        )
        trajectory_hash = sha256_file(trajectory)
        header = (
            "frame,timestamp,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,"
            "sharpness,depth_path,pose_source,uses_ground_truth_pose,"
            "trajectory_path,trajectory_sha256,trajectory_key"
        )
        row_suffix = (
            ",droid_traj_est_not_align,false,"
            f"{trajectory},{trajectory_hash},traj_est_not_align"
        )
        frames_csv.write_text(
            header
            + "\n"
            + "a.png,0,0,0,0,0,0,0,1,20,20,7.5,5.5,1,a.npy"
            + row_suffix
            + "\n",
            encoding="utf-8",
        )
        values = {
            "frames_csv": frames_csv,
            "planner_json": planner_json,
            "image_root": image_root,
            "depth_root": depth_root,
            "framecrafter_repo": framecrafter_repo,
            "checkpoint": checkpoint,
            "base_model_dir": base_model_dir,
            "output_dir": self.root / "generated",
            "min_sharpness_gain": 1.05,
            "blur_quantile": 0.30,
            "seed": 42,
            "pose_source": "droid_traj_est_not_align",
        }
        signature = compute_preprocess_signature(SimpleNamespace(**values))
        self.assertEqual(
            signature, compute_preprocess_signature(SimpleNamespace(**values))
        )
        self.assertRegex(signature, r"^[0-9a-f]{64}$")

        frames_csv.write_text(
            header
            + "\n"
            + "a.png,1,0,0,0,0,0,0,1,20,20,7.5,5.5,1,a.npy"
            + row_suffix
            + "\n",
            encoding="utf-8",
        )
        changed_csv = compute_preprocess_signature(SimpleNamespace(**values))
        self.assertNotEqual(signature, changed_csv)

        planner_json.write_text('{"segments": [{"alphas": [0.5]}]}\n', encoding="utf-8")
        changed_plan = compute_preprocess_signature(SimpleNamespace(**values))
        self.assertNotEqual(changed_csv, changed_plan)

        changed_gate_values = dict(values, min_sharpness_gain=1.10)
        self.assertNotEqual(
            changed_plan,
            compute_preprocess_signature(SimpleNamespace(**changed_gate_values)),
        )
        changed_local_gate_values = dict(
            values, evssm_local_max_tile_mae=0.12
        )
        self.assertNotEqual(
            changed_plan,
            compute_preprocess_signature(
                SimpleNamespace(**changed_local_gate_values)
            ),
        )

        checkpoint.write_bytes(b"checkpoint-v2-with-a-different-size")
        self.assertNotEqual(
            changed_plan, compute_preprocess_signature(SimpleNamespace(**values))
        )

        before_image_change = compute_preprocess_signature(SimpleNamespace(**values))
        save_test_rgb(image_root / "a.png", np.flipud(self.image))
        self.assertNotEqual(
            before_image_change,
            compute_preprocess_signature(SimpleNamespace(**values)),
        )

        before_depth_change = compute_preprocess_signature(SimpleNamespace(**values))
        np.save(depth_root / "a.npy", np.full(self.image.shape[:2], 2.0, dtype=np.float32))
        self.assertNotEqual(
            before_depth_change,
            compute_preprocess_signature(SimpleNamespace(**values)),
        )

        before_repo_change = compute_preprocess_signature(SimpleNamespace(**values))
        (framecrafter_repo / "model.py").write_text("VERSION = 2\n", encoding="utf-8")
        self.assertNotEqual(
            before_repo_change,
            compute_preprocess_signature(SimpleNamespace(**values)),
        )

        large_shard = base_model_dir / "diffusion_pytorch_model.safetensors"
        large_shard.write_bytes(b"A" * (9 * 1024 * 1024))
        before_large_content_change = compute_preprocess_signature(
            SimpleNamespace(**values)
        )
        previous_stat = large_shard.stat()
        with large_shard.open("r+b") as handle:
            handle.seek(4 * 1024 * 1024)
            handle.write(b"B")
        os.utime(
            large_shard,
            ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns),
        )
        self.assertNotEqual(
            before_large_content_change,
            compute_preprocess_signature(SimpleNamespace(**values)),
        )

    def test_cli_cpu_smoke_planner_to_npz_gates_and_manifest(self) -> None:
        trajectory = self.root / "traj_full_full_traj.npz"
        trajectory_poses = np.repeat(np.eye(4)[None], 3, axis=0)
        np.savez(
            trajectory,
            traj_est_not_align=trajectory_poses,
            traj_est_not_align_timestamps=np.arange(3, dtype=np.float64),
            traj_est_not_align_eval_mask=np.ones(3, dtype=np.bool_),
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
        )
        trajectory_hash = sha256_file(trajectory)
        rows = [
            "frame,timestamp,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,sharpness,"
            "depth_path,pose_source,uses_ground_truth_pose,trajectory_path,"
            "trajectory_sha256,trajectory_key"
        ]
        for index, score in enumerate((0.0, 0.0, 10.0)):
            rgb = self.root / f"cli_rgb_{index}.png"
            depth = self.root / f"cli_depth_{index}.npy"
            save_test_rgb(rgb, self.image)
            np.save(depth, np.ones(self.image.shape[:2], dtype=np.float32))
            rows.append(
                f"{rgb.name},{index},0,0,0,0,0,0,1,20,20,7.5,5.5,"
                f"{score},{depth.name},droid_traj_est_not_align,false,"
                f"{trajectory},{trajectory_hash},traj_est_not_align"
            )
        csv_path = self.root / "estimated_droid_frames.csv"
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        # A tiny CPU fake implementing the public FrameCrafter constructor
        # and generate signature. This validates our dynamic Python-API bridge
        # without downloading or pretending to execute the 14B model.
        fake_repo = self.root / "fake_official_framecrafter"
        fake_repo.mkdir()
        (fake_repo / "model.py").write_text(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "class FrameCrafter:\n"
            "    def __init__(self, checkpoint_path, device='cuda', vram_limit=20, base_model_dir=None):\n"
            "        self.checkpoint_path = checkpoint_path\n"
            "        self.call_log = Path(checkpoint_path).parent.parent / 'fake_generate_calls.txt'\n"
            "    def generate(self, images, w2c_poses, intrinsics, **kwargs):\n"
            "        target_count = len(w2c_poses) - len(images)\n"
            "        prior = int(self.call_log.read_text()) if self.call_log.exists() else 0\n"
            "        self.call_log.write_text(str(prior + 1))\n"
            "        source = np.asarray(images[0]).copy()\n"
            "        outputs = []\n"
            "        for target_position in range(target_count):\n"
            "            result = source.copy()\n"
            "            if target_position == 1:\n"
            "                result[:] = 0\n"
            "            elif target_position == 2:\n"
            "                result[..., 2] = np.clip(result[..., 2].astype(np.int16) + 20, 0, 255)\n"
            "            outputs.append(Image.fromarray(result.astype(np.uint8)))\n"
            "        return list(images) + outputs\n",
            encoding="utf-8",
        )
        fake_checkpoint = fake_repo / "framecrafter.safetensors"
        fake_checkpoint.write_bytes(b"CPU contract fixture; not model weights")
        fake_base_model = self.root / "fake_wan_base"
        fake_base_model.mkdir()
        (fake_base_model / "config.json").write_text("{}\n", encoding="utf-8")
        output = self.root / "cli_output"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_framecrafter_preprocess.py"),
            "--frames-csv", str(csv_path),
            "--image-root", str(self.root),
            "--depth-root", str(self.root),
            "--depth-scale", "1",
            "--output-depth-scale", "5000",
            "--laplacian-threshold", "1",
            "--translation-step", "10",
            "--rotation-step-deg", "180",
            "--blur-region-inserts", "3",
            "--context-count", "3",
            "--min-contexts", "3",
            "--backend", "python_api",
            "--framecrafter-repo", str(fake_repo),
            "--checkpoint", str(fake_checkpoint),
            "--base-model-dir", str(fake_base_model),
            "--device", "cpu",
            "--min-sharpness-gain", "0.99",
            "--min-depth-coverage", "0.9",
            "--min-depth-consistency", "0.9",
            "--max-photometric-error", "0.05",
            "--max-reprojection-error-px", "0.1",
            "--min-reprojection-valid-ratio", "0.9",
            "--output-dir", str(output),
        ]
        completed = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True
        )
        self.assertIn('"accepted_target_count": 2', completed.stdout)
        self.assertIn('"backend": "python_api"', completed.stdout)
        manifest_paths = list(output.glob("manifest_*.json"))
        report_paths = list(output.glob("preprocess_report_*.json"))
        self.assertEqual(len(manifest_paths), 1)
        self.assertEqual(len(report_paths), 1)
        manifest_path = manifest_paths[0]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["generated_frame_count"], 2)
        report = json.loads(report_paths[0].read_text(encoding="utf-8"))
        self.assertEqual(report["planned_total_before_cap"], 3)
        self.assertEqual(report["selected_target_count"], 3)
        self.assertEqual(report["generation_batch_count"], 1)
        self.assertEqual(report["backend_generate_call_count"], 1)
        self.assertEqual(len(report["generation_batches"]), 1)
        self.assertEqual(report["generation_batches"][0]["target_count"], 3)
        self.assertEqual(report["generation_batches"][0]["total_view_count"], 6)
        self.assertEqual(report["generation_batches"][0]["endpoint_position_span"], 1)
        self.assertEqual(report["generation_batches"][0]["max_endpoint_position_span"], 6)
        self.assertEqual(len({item["batch_id"] for item in report["planned"]}), 1)
        self.assertEqual(
            [item["batch_target_position"] for item in report["planned"]],
            [0, 1, 2],
        )
        self.assertEqual(
            [item["batch_target_position"] for item in report["accepted"]],
            [0, 2],
        )
        self.assertEqual(report["rejected"][0]["batch_target_position"], 1)
        self.assertIn("sharpness_gain", report["rejected"][0]["failures"])
        self.assertEqual(
            (self.root / "fake_generate_calls.txt").read_text(encoding="utf-8"),
            "1",
        )
        self.assertRegex(manifest["preprocess_signature"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            manifest["preprocess_signature"], report["preprocess_signature"]
        )
        self.assertFalse(manifest["backend_test_only"])
        self.assertEqual(manifest["backend"], "python_api")
        self.assertEqual(
            manifest["accepted_output_sha256"], report["accepted_output_sha256"]
        )
        self.assertEqual(
            manifest["source_input_sha256"], report["source_input_sha256"]
        )
        self.assertRegex(manifest["preprocess_report_sha256"], r"^[0-9a-f]{64}$")
        validate_manifest_payload(
            manifest, manifest_path=manifest_path, require_provenance=True
        )
        tampered = json.loads(json.dumps(manifest))
        tampered_synthetic = next(
            item for item in tampered["frames"] if item["kind"] == "synthetic"
        )
        tampered_synthetic["c2w"][0][3] += 99.0
        with self.assertRaisesRegex(ValueError, "frame-contract digest"):
            validate_manifest_payload(
                tampered,
                manifest_path=manifest_path,
                require_provenance=True,
            )
        invalid_batch_report = json.loads(json.dumps(report))
        invalid_batch_report["generation_batches"][0]["target_ids"] = list(
            reversed(invalid_batch_report["generation_batches"][0]["target_ids"])
        )
        invalid_batch_report_path = output / "invalid_batch_report.json"
        invalid_batch_report_path.write_text(
            json.dumps(invalid_batch_report), encoding="utf-8"
        )
        invalid_batch_manifest = json.loads(json.dumps(manifest))
        invalid_batch_manifest["preprocess_report_path"] = str(
            invalid_batch_report_path
        )
        invalid_batch_manifest["preprocess_report_sha256"] = sha256_file(
            invalid_batch_report_path
        )
        with self.assertRaisesRegex(ValueError, "batch|planned target order"):
            validate_manifest_payload(
                invalid_batch_manifest,
                manifest_path=manifest_path,
                require_provenance=True,
            )
        unsafe_report = dict(report)
        unsafe_report["uses_ground_truth_pose"] = True
        unsafe_report_path = output / "unsafe_preprocess_report.json"
        unsafe_report_path.write_text(json.dumps(unsafe_report), encoding="utf-8")
        unsafe_manifest = json.loads(json.dumps(manifest))
        unsafe_manifest["preprocess_report_path"] = str(unsafe_report_path)
        unsafe_manifest["preprocess_report_sha256"] = sha256_file(unsafe_report_path)
        with self.assertRaisesRegex(ValueError, "report"):
            validate_manifest_payload(
                unsafe_manifest,
                manifest_path=manifest_path,
                require_provenance=True,
            )
        synthetic = next(item for item in manifest["frames"] if item["kind"] == "synthetic")
        synthetics = [item for item in manifest["frames"] if item["kind"] == "synthetic"]
        self.assertEqual(
            [item["batch_target_position"] for item in synthetics], [0, 2]
        )
        first_pixels = np.asarray(Image.open(synthetics[0]["rgb_path"]))
        third_pixels = np.asarray(Image.open(synthetics[1]["rgb_path"]))
        np.testing.assert_array_equal(third_pixels[..., :2], first_pixels[..., :2])
        np.testing.assert_array_equal(
            third_pixels[..., 2],
            np.clip(first_pixels[..., 2].astype(np.int16) + 20, 0, 255).astype(np.uint8),
        )
        self.assertTrue(synthetic["depth_path"].endswith(".png"))
        encoded_depth = np.asarray(Image.open(synthetic["depth_path"]))
        self.assertEqual(int(encoded_depth.max()), 5000)
        npz_files = list((output / "artifacts").glob("*/*/framecrafter_npz/*.npz"))
        self.assertEqual(len(npz_files), 1)
        arrays = np.load(npz_files[0])
        self.assertEqual(arrays["w2c_poses"].shape, (6, 4, 4))

        first_rgb_path = Path(synthetic["rgb_path"])
        first_rgb_hash = sha256_file(first_rgb_path)
        subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True
        )
        self.assertEqual(
            (self.root / "fake_generate_calls.txt").read_text(encoding="utf-8"),
            "2",
        )
        regenerated_manifests = list(output.glob("manifest_*.json"))
        self.assertEqual(len(regenerated_manifests), 2)
        second_manifest_path = next(
            path for path in regenerated_manifests if path != manifest_path
        )
        second_manifest = json.loads(
            second_manifest_path.read_text(encoding="utf-8")
        )
        second_synthetic = next(
            item
            for item in second_manifest["frames"]
            if item["kind"] == "synthetic"
        )
        self.assertNotEqual(second_manifest["generation_id"], manifest["generation_id"])
        self.assertNotEqual(second_synthetic["rgb_path"], str(first_rgb_path))
        self.assertEqual(sha256_file(first_rgb_path), first_rgb_hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)
