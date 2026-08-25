#!/usr/bin/env python3
"""CPU contracts for the official GoPro TURTLE stream materializer."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_tum_turtle_stream import (  # noqa: E402
    FR2_XYZ_DROID_KEYFRAMES,
    FR2_XYZ_DISTORTION,
    FR2_XYZ_HEIGHT_EDGE,
    FR2_XYZ_WIDTH_EDGE,
    OFFICIAL_GOPRO_CACHE_NON_NULL_MASK,
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    SCHEMA,
    TURTLE_CACHE_CONTRACT,
    _cache_audit,
    _preprocess_raw,
    load_contiguous_source_frames,
    materialize_tum_turtle_stream,
    sha256_pixels,
)


class FakeCv2:
    __version__ = "fake-cpu-contract"
    IMREAD_COLOR = 1
    INTER_LINEAR = 1

    def __init__(self):
        self.undistort_calls = 0
        self.resize_calls = 0

    def imread(self, path, flags):
        del flags
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def undistort(self, image, camera, distortion):
        self.undistort_calls += 1
        assert camera.shape == (3, 3)
        assert distortion.shape == (5,)
        return image.copy()

    def resize(self, image, size, interpolation):
        del interpolation
        self.resize_calls += 1
        width, height = size
        if (image.shape[1], image.shape[0]) == (width, height):
            return image.copy()
        return np.asarray(
            Image.fromarray(image, mode="RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )


class FakeOfficialBackend:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None
        self.frames_seen = 0
        self.cache_updates = 0
        self.reset_count = 0
        self.last_timestamp = None
        self.resolution = None
        self.calls = []

    def reset(self):
        self.k_cache = None
        self.v_cache = None
        self.last_timestamp = None
        self.resolution = None
        self.reset_count += 1

    def step(self, image, timestamp=None):
        prior_cache = self.k_cache is not None
        self.calls.append(
            {
                "timestamp": timestamp,
                "prior_cache": prior_cache,
                "image": image.detach().clone(),
            }
        )
        self.frames_seen += 1
        self.cache_updates += 1
        self.last_timestamp = timestamp
        self.resolution = tuple(image.shape[-2:])
        marker = torch.tensor([self.frames_seen], dtype=torch.float32)
        self.k_cache = [
            marker.clone() if present else None
            for present in OFFICIAL_GOPRO_CACHE_NON_NULL_MASK
        ]
        self.v_cache = [
            marker.clone() if present else None
            for present in OFFICIAL_GOPRO_CACHE_NON_NULL_MASK
        ]
        return (image + self.frames_seen / 255.0).clamp(0.0, 1.0)

    def state_info(self):
        return {
            "frames_seen": self.frames_seen,
            "cache_updates": self.cache_updates,
            "reset_count": self.reset_count,
            "resolution": self.resolution,
            "last_timestamp": self.last_timestamp,
            "has_cache": self.k_cache is not None and self.v_cache is not None,
        }


def official_record():
    return {
        "implementation": "official_ascend_research_turtle",
        "repo": {"commit": PINNED_TURTLE_COMMIT},
        "architecture": {"sha256": PINNED_TURTLE_ARCH_SHA256},
        "config": {"sha256": PINNED_TURTLE_CONFIG_SHA256},
        "checkpoint": {
            "sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
            "metadata": {"kind": "official_gopro"},
        },
        "cache_contract": TURTLE_CACHE_CONTRACT,
    }


class TurtleStreamMaterializerTest(unittest.TestCase):
    def make_source(self, root: Path, *, count=4, gt_index=None, timestamps=None):
        fields = [
            "index",
            "timestamp",
            "rgb_path",
            "fx",
            "fy",
            "cx",
            "cy",
            "pose_source",
            "uses_ground_truth_pose",
        ]
        rows = []
        for index in range(count):
            pixels = np.full((8, 8, 3), 20 + index * 10, dtype=np.uint8)
            pixels[:, :, 1] += index
            image_path = root / f"{index:06d}.png"
            Image.fromarray(pixels, mode="RGB").save(image_path)
            rows.append(
                {
                    "index": index,
                    "timestamp": (
                        timestamps[index]
                        if timestamps is not None
                        else 1000.0 + index * 0.1
                    ),
                    "rgb_path": str(image_path),
                    "fx": 8.0,
                    "fy": 8.0,
                    "cx": 4.0,
                    "cy": 4.0,
                    "pose_source": "droid_traj_est_not_align",
                    "uses_ground_truth_pose": index == gt_index,
                }
            )
        csv_path = root / "frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_gap_free_steps_emit_only_selection_with_full_hash_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = self.make_source(root)
            destination = root / "output"
            backend = FakeOfficialBackend()
            cv2 = FakeCv2()
            manifest_path = materialize_tum_turtle_stream(
                frames_csv=csv_path,
                output_dir=destination,
                start_index=0,
                end_index=3,
                emitted_source_indices=(0, 2, 3),
                width=8,
                height=8,
                device="cpu",
                progress_every=0,
                _backend=backend,
                _turtle_record=official_record(),
                _cv2_module=cv2,
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["schema"], SCHEMA)
            self.assertEqual(payload["stream"]["processed_count"], 4)
            self.assertEqual(payload["stream"]["step_count"], 4)
            self.assertEqual(payload["stream"]["cache_updates"], 4)
            self.assertEqual(payload["stream"]["reset_count"], 1)
            self.assertEqual(payload["stream"]["processed_source_indices"], [0, 1, 2, 3])
            self.assertFalse(payload["stream"]["gaps_skipped"])
            self.assertEqual(
                payload["selection"]["emitted_source_indices"], [0, 2, 3]
            )
            self.assertEqual([frame["source_index"] for frame in payload["frames"]], [0, 2, 3])
            self.assertEqual(len(payload["stream"]["steps"]), 4)
            self.assertEqual(
                payload["stream"]["official_gopro_cache_non_null_mask"],
                [False, False, False, True, True, True, True, True],
            )
            self.assertEqual(payload["stream"]["k_cache_non_null_count"], 5)
            self.assertEqual(len(backend.calls), 4)
            self.assertEqual(
                [call["prior_cache"] for call in backend.calls],
                [False, True, True, True],
            )
            self.assertEqual(
                [call["timestamp"] for call in backend.calls],
                [1000.0, 1000.1, 1000.2, 1000.3],
            )
            self.assertEqual(cv2.undistort_calls, 4)
            self.assertEqual(cv2.resize_calls, 4)
            self.assertFalse(payload["safety"]["ground_truth_images_used"])
            self.assertFalse(payload["safety"]["ground_truth_poses_used"])
            self.assertFalse(payload["safety"]["custom_causal_evssm_used"])
            self.assertEqual(payload["camera"]["K"], [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])

            for frame in payload["frames"]:
                output = Path(frame["output"]["path"])
                self.assertTrue(output.is_file())
                pixels = np.asarray(Image.open(output).convert("RGB"), dtype=np.uint8)
                self.assertEqual(frame["output"]["pixel_sha256"], sha256_pixels(pixels))
                self.assertEqual(len(frame["output"]["sha256"]), 64)
            self.assertFalse((destination / "images/000001.png").exists())
            for step in payload["stream"]["steps"]:
                self.assertEqual(len(step["input_file_sha256"]), 64)
                self.assertEqual(len(step["input_rgb_u8_pixel_sha256"]), 64)
                self.assertEqual(len(step["output_rgb_u8_pixel_sha256"]), 64)
                self.assertEqual(step["k_cache_non_null_count_after"], 5)
                self.assertEqual(step["v_cache_non_null_count_after"], 5)
                self.assertEqual(
                    step["k_cache_non_null_mask_after"],
                    [False, False, False, True, True, True, True, True],
                )

    def test_cache_audit_accepts_only_the_exact_official_sparse_layout(self):
        marker = torch.ones(1)
        official = [
            marker if present else None
            for present in OFFICIAL_GOPRO_CACHE_NON_NULL_MASK
        ]
        audit = _cache_audit(official, "key")
        self.assertEqual(audit["slot_count"], 8)
        self.assertEqual(audit["non_null_count"], 5)
        self.assertEqual(
            audit["non_null_mask"],
            [False, False, False, True, True, True, True, True],
        )
        with self.assertRaisesRegex(RuntimeError, "cache mask mismatch"):
            _cache_audit([marker for _ in range(8)], "key")
        bad_type = list(official)
        bad_type[3] = "not-a-tensor"
        with self.assertRaisesRegex(RuntimeError, "Tensor or None"):
            _cache_audit(bad_type, "value")

    def test_rejects_missing_index_nonmonotonic_time_and_gt_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = self.make_source(root, count=3)
            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows((rows[0], rows[2]))
            with self.assertRaisesRegex(ValueError, "missing source indices"):
                materialize_tum_turtle_stream(
                    frames_csv=csv_path,
                    output_dir=root / "gap",
                    start_index=0,
                    end_index=2,
                    emitted_source_indices=(0,),
                    width=8,
                    height=8,
                    device="cpu",
                    _backend=FakeOfficialBackend(),
                    _turtle_record=official_record(),
                    _cv2_module=FakeCv2(),
                )

            time_root = root / "time"
            time_root.mkdir()
            time_csv = self.make_source(time_root, count=3, timestamps=(1.0, 2.0, 2.0))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                materialize_tum_turtle_stream(
                    frames_csv=time_csv,
                    output_dir=root / "bad_time",
                    start_index=0,
                    end_index=2,
                    emitted_source_indices=(0,),
                    width=8,
                    height=8,
                    device="cpu",
                    _backend=FakeOfficialBackend(),
                    _turtle_record=official_record(),
                    _cv2_module=FakeCv2(),
                )

            gt_root = root / "gt"
            gt_root.mkdir()
            gt_csv = self.make_source(gt_root, count=2, gt_index=1)
            with self.assertRaisesRegex(ValueError, "must explicitly be false"):
                materialize_tum_turtle_stream(
                    frames_csv=gt_csv,
                    output_dir=root / "bad_gt",
                    start_index=0,
                    end_index=1,
                    emitted_source_indices=(0,),
                    width=8,
                    height=8,
                    device="cpu",
                    _backend=FakeOfficialBackend(),
                    _turtle_record=official_record(),
                    _cv2_module=FakeCv2(),
                )

    def test_refuses_overwrite_before_backend_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = self.make_source(root, count=1)
            destination = root / "exists"
            destination.mkdir()
            backend = FakeOfficialBackend()
            with self.assertRaises(FileExistsError):
                materialize_tum_turtle_stream(
                    frames_csv=csv_path,
                    output_dir=destination,
                    start_index=0,
                    end_index=0,
                    emitted_source_indices=(0,),
                    width=8,
                    height=8,
                    device="cpu",
                    _backend=backend,
                    _turtle_record=official_record(),
                    _cv2_module=FakeCv2(),
                )
            self.assertEqual(backend.calls, [])

    def test_formal_fr2_profile_is_the_fixed_42_keyframe_partition(self):
        self.assertEqual(len(FR2_XYZ_DROID_KEYFRAMES), 42)
        self.assertEqual(FR2_XYZ_DROID_KEYFRAMES[0], 0)
        self.assertEqual(FR2_XYZ_DROID_KEYFRAMES[-1], 2764)
        self.assertEqual(len(set(FR2_XYZ_DROID_KEYFRAMES)), 42)
        self.assertEqual(tuple(sorted(FR2_XYZ_DROID_KEYFRAMES)), FR2_XYZ_DROID_KEYFRAMES)
        self.assertEqual(FR2_XYZ_WIDTH_EDGE, 8)
        self.assertEqual(FR2_XYZ_HEIGHT_EDGE, 8)

    def test_tracker_resize_then_crop_contract_updates_effective_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = self.make_source(root, count=1)
            manifest_path = materialize_tum_turtle_stream(
                frames_csv=csv_path,
                output_dir=root / "cropped",
                start_index=0,
                end_index=0,
                emitted_source_indices=(0,),
                width=8,
                height=8,
                width_edge=2,
                height_edge=2,
                device="cpu",
                progress_every=0,
                _backend=FakeOfficialBackend(),
                _turtle_record=official_record(),
                _cv2_module=FakeCv2(),
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["camera"]["resize_before_crop_width"], 12)
            self.assertEqual(payload["camera"]["resize_before_crop_height"], 12)
            self.assertEqual(
                payload["camera"]["crop_edges"],
                {"left": 2, "right": 2, "top": 2, "bottom": 2},
            )
            self.assertEqual(
                payload["camera"]["K"],
                [[12.0, 0.0, 4.0], [0.0, 12.0, 4.0], [0.0, 0.0, 1.0]],
            )
            with Image.open(payload["frames"][0]["output"]["path"]) as output:
                self.assertEqual(output.size, (8, 8))

    def test_script_imports_only_the_official_turtle_backend(self):
        source_path = ROOT / "scripts/materialize_tum_turtle_stream.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertIn("src.turtle_backend", imported_modules)
        self.assertFalse(
            any("causal" in module or "evssm" in module for module in imported_modules)
        )


class PinnedFr2TrackerAlignmentTest(unittest.TestCase):
    """Optional real-data proof that the materializer matches BaseDataset pixels."""

    FRAMES_CSV = Path(
        "/srv/szha0669/unblur-slam/framecrafter_preprocess/fr2_xyz/estimated_frames.csv"
    )
    CLEAR_GT_ROOT = Path(
        "/srv/szha0669/unblur-slam/slam_smoke/fr2_xyz_resplat/baseline/"
        "freiburg2_xyz/refinement_checkpoints/iter_000400/clear_gt_renders"
    )

    def test_all_42_clear_gt_left_panels_match_tracker_preprocessing_exactly(self):
        if not self.FRAMES_CSV.is_file() or not self.CLEAR_GT_ROOT.is_dir():
            self.skipTest("pinned fr2 smoke artifacts are unavailable")
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is unavailable in this CPU test environment")
        frames, _ = load_contiguous_source_frames(
            self.FRAMES_CSV, start_index=0, end_index=2764
        )
        by_index = {frame.source_index: frame for frame in frames}
        for source_index in FR2_XYZ_DROID_KEYFRAMES:
            _, pixels, _ = _preprocess_raw(
                by_index[source_index],
                width=512,
                height=384,
                width_edge=FR2_XYZ_WIDTH_EDGE,
                height_edge=FR2_XYZ_HEIGHT_EDGE,
                distortion=np.asarray(FR2_XYZ_DISTORTION, dtype=np.float64),
                cv2_module=cv2,
            )
            comparison_path = (
                self.CLEAR_GT_ROOT / f"source_{source_index:06d}_gt_render.png"
            )
            self.assertTrue(comparison_path.is_file())
            with Image.open(comparison_path) as image:
                clear_gt_left = np.asarray(image.convert("RGB"), dtype=np.uint8)[
                    :, :512
                ]
            self.assertTrue(
                np.array_equal(pixels, clear_gt_left),
                f"tracker-space mismatch at source_index={source_index}",
            )


if __name__ == "__main__":
    unittest.main()
