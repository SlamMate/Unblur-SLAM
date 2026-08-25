#!/usr/bin/env python3
"""Synthetic CPU tests for overlap-driven FrameCrafter planning."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framecrafter_overlap import (  # noqa: E402
    OverlapPlanningConfig,
    approximate_frustum_overlap,
    bidirectional_depth_overlap,
    match_image_overlap_ransac,
    plan_overlap_deficit,
)


def intrinsics(height: int, width: int, focal: float = 80.0) -> np.ndarray:
    return np.array(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pose(*, center_x: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    angle = math.radians(yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )
    result[0, 3] = center_x
    return result


class FrameCrafterOverlapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.height = 48
        self.width = 64
        self.k = intrinsics(self.height, self.width)
        self.depth = np.ones((self.height, self.width), dtype=np.float32)
        self.identity = pose()

    def depth_overlap(self, c2w_b: np.ndarray, depth_b: np.ndarray | None = None):
        return bidirectional_depth_overlap(
            self.depth,
            self.k,
            self.identity,
            self.depth if depth_b is None else depth_b,
            self.k,
            c2w_b,
            sample_stride=2,
            depth_abs_tolerance=1.0e-5,
            depth_rel_tolerance=1.0e-5,
        )

    def test_identical_cameras_have_full_bidirectional_visibility(self) -> None:
        result = self.depth_overlap(self.identity)
        self.assertAlmostEqual(result.a_to_b.frustum_ratio, 1.0)
        self.assertAlmostEqual(result.b_to_a.frustum_ratio, 1.0)
        self.assertAlmostEqual(result.symmetric_visible_overlap, 1.0)
        self.assertAlmostEqual(result.symmetric_target_coverage, 1.0)
        self.assertEqual(result.a_to_b.occluded_count, 0)

    def test_nearby_but_opposite_facing_cameras_do_not_overlap(self) -> None:
        opposite = pose(yaw_deg=180.0)
        result = self.depth_overlap(opposite)
        self.assertEqual(result.symmetric_frustum_overlap, 0.0)
        self.assertEqual(result.symmetric_visible_overlap, 0.0)

        coarse = approximate_frustum_overlap(
            self.k,
            self.identity,
            self.depth.shape,
            self.k,
            opposite,
            self.depth.shape,
            depth_range_a=(0.5, 2.0),
            depth_range_b=(0.5, 2.0),
        )
        self.assertEqual(coarse.symmetric_overlap, 0.0)

    def test_translation_reduces_frustum_overlap(self) -> None:
        same = approximate_frustum_overlap(
            self.k,
            self.identity,
            self.depth.shape,
            self.k,
            self.identity,
            self.depth.shape,
            depth_range_a=(1.0, 2.0),
            depth_range_b=(1.0, 2.0),
        )
        shifted = approximate_frustum_overlap(
            self.k,
            self.identity,
            self.depth.shape,
            self.k,
            pose(center_x=0.4),
            self.depth.shape,
            depth_range_a=(1.0, 2.0),
            depth_range_b=(1.0, 2.0),
        )
        self.assertAlmostEqual(same.symmetric_overlap, 1.0)
        self.assertGreater(shifted.symmetric_overlap, 0.0)
        self.assertLess(shifted.symmetric_overlap, same.symmetric_overlap)

    def test_destination_occluder_reduces_visible_overlap(self) -> None:
        depth_b = self.depth.copy()
        depth_b[:, : self.width // 2] = 0.5
        result = self.depth_overlap(self.identity, depth_b)
        self.assertGreater(result.a_to_b.occluded_count, 0)
        self.assertAlmostEqual(result.a_to_b.visible_ratio, 0.5, delta=0.03)
        self.assertAlmostEqual(result.symmetric_visible_overlap, 0.5, delta=0.03)

    def test_invalid_camera_depth_and_nonfinite_inputs_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "depth_a contains non-finite"):
            invalid = self.depth.copy()
            invalid[0, 0] = np.nan
            bidirectional_depth_overlap(
                invalid,
                self.k,
                self.identity,
                self.depth,
                self.k,
                self.identity,
            )
        with self.assertRaisesRegex(ValueError, "intrinsics_a must have shape"):
            bidirectional_depth_overlap(
                self.depth,
                np.eye(4),
                self.identity,
                self.depth,
                self.k,
                self.identity,
            )
        with self.assertRaisesRegex(ValueError, "c2w_b rotation"):
            invalid_pose = self.identity.copy()
            invalid_pose[0, 0] = 2.0
            self.depth_overlap(invalid_pose)
        with self.assertRaisesRegex(ValueError, "image_shape_a"):
            approximate_frustum_overlap(
                self.k,
                self.identity,
                (0, self.width),
                self.k,
                self.identity,
                self.depth.shape,
            )

    def test_overlap_deficit_recommends_even_inserts_or_submap(self) -> None:
        config = OverlapPlanningConfig(
            target_pair_overlap=0.7,
            hard_submap_overlap=0.05,
            max_inserts=4,
        )
        sufficient = plan_overlap_deficit(0.8, config)
        self.assertEqual(sufficient.insert_count, 0)
        self.assertFalse(sufficient.split_submap)

        bridge = plan_overlap_deficit(0.25, config)
        self.assertEqual(bridge.required_inserts, 3)
        self.assertEqual(bridge.insert_count, 3)
        self.assertEqual(bridge.alphas, (0.25, 0.5, 0.75))
        self.assertFalse(bridge.split_submap)

        disconnected = plan_overlap_deficit(0.01, config)
        self.assertTrue(disconnected.split_submap)
        self.assertEqual(disconnected.insert_count, 0)
        self.assertEqual(disconnected.reason, "hard_overlap_discontinuity")

        over_budget = plan_overlap_deficit(0.1, config)
        self.assertTrue(over_budget.budget_exceeded)
        self.assertTrue(over_budget.split_submap)

    def test_missing_opencv_is_auditable_and_optionally_required(self) -> None:
        image = np.zeros((32, 32), dtype=np.uint8)
        with mock.patch(
            "src.framecrafter_overlap._import_cv2",
            side_effect=ModuleNotFoundError("cv2"),
        ):
            result = match_image_overlap_ransac(image, image)
            self.assertFalse(result.available)
            self.assertFalse(result.success)
            self.assertIn("skipped", result.message)
            with self.assertRaisesRegex(RuntimeError, "OpenCV is required"):
                match_image_overlap_ransac(image, image, require_opencv=True)

    def test_feature_ransac_reports_matched_image_coverage(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is optional")
        generator = np.random.default_rng(7)
        image_a = generator.integers(0, 256, size=(160, 200), dtype=np.uint8)
        image_b = np.zeros_like(image_a)
        image_b[5:, 8:] = image_a[:-5, :-8]
        result = match_image_overlap_ransac(
            image_a,
            image_b,
            detector="orb",
            model_type="homography",
            max_features=1500,
            ratio_test=0.8,
            ransac_threshold_px=1.0,
        )
        self.assertTrue(result.available)
        self.assertTrue(result.success, result.message)
        self.assertGreater(result.inliers, 20)
        self.assertGreater(result.inlier_ratio, 0.8)
        # ORB deliberately excludes a border around the image, so convex-hull
        # coverage is conservative even for this near-full translated view.
        self.assertGreater(result.symmetric_coverage, 0.3)
        self.assertGreater(result.overlap_score, 0.3)
        self.assertEqual(result.model.shape, (3, 3))

    def test_feature_input_shape_and_finiteness_fail_before_optional_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "image_a must have shape"):
            match_image_overlap_ransac(np.zeros((2, 2, 2, 2)), np.zeros((2, 2)))
        bad = np.zeros((8, 8), dtype=np.float32)
        bad[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "image_a contains non-finite"):
            match_image_overlap_ransac(bad, np.zeros_like(bad))


if __name__ == "__main__":
    unittest.main()
