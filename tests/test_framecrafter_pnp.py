#!/usr/bin/env python3
"""CPU contract tests for RGB-D feature/PnP pose refinement."""

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

from src.framecrafter_pnp import (  # noqa: E402
    PnPRefinementGateConfig,
    _MatchedFeatures,
    gate_pnp_refinement,
    refine_rgbd_pose_pnp,
)


def intrinsics(height: int, width: int, focal: float = 100.0) -> np.ndarray:
    return np.array(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def yaw_pose(yaw_deg: float, translation: tuple[float, float, float]) -> np.ndarray:
    angle = math.radians(yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )
    result[:3, 3] = translation
    return result


class FrameCrafterPnPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.height, self.width = 120, 160
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.depth = np.full((self.height, self.width), 2.0, dtype=np.float64)
        self.K = intrinsics(self.height, self.width)
        self.identity = np.eye(4, dtype=np.float64)

    def test_image_depth_scale_mismatch_and_missing_depth_are_explicit(self) -> None:
        mismatch = refine_rgbd_pose_pnp(
            self.image,
            self.depth[:-1],
            self.K,
            self.identity,
            self.image,
            self.K,
        )
        self.assertFalse(mismatch.success)
        self.assertEqual(mismatch.failure_code, "image_depth_scale_mismatch")

        missing = refine_rgbd_pose_pnp(
            self.image,
            np.zeros_like(self.depth),
            self.K,
            self.identity,
            self.image,
            self.K,
        )
        self.assertFalse(missing.success)
        self.assertEqual(missing.failure_code, "missing_depth")

    def test_missing_opencv_is_auditable_and_optionally_required(self) -> None:
        with mock.patch(
            "src.framecrafter_pnp._import_cv2",
            side_effect=ModuleNotFoundError("cv2"),
        ):
            result = refine_rgbd_pose_pnp(
                self.image,
                self.depth,
                self.K,
                self.identity,
                self.image,
                self.K,
            )
            self.assertFalse(result.available)
            self.assertEqual(result.failure_code, "opencv_unavailable")
            with self.assertRaisesRegex(RuntimeError, "OpenCV is required"):
                refine_rgbd_pose_pnp(
                    self.image,
                    self.depth,
                    self.K,
                    self.identity,
                    self.image,
                    self.K,
                    require_opencv=True,
                )

    def test_synthetic_rgbd_correspondences_recover_pose_and_gate_correction(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is optional")

        true_pose_b = yaw_pose(4.0, (0.08, -0.02, 0.03))
        initial_pose_b = yaw_pose(6.0, (0.11, -0.02, 0.03))
        pixels_a: list[list[float]] = []
        points_b: list[list[float]] = []
        depth = np.zeros_like(self.depth)
        rng = np.random.default_rng(19)
        while len(pixels_a) < 36:
            u = int(rng.integers(18, self.width - 18))
            v = int(rng.integers(18, self.height - 18))
            if depth[v, u] > 0.0:
                continue
            z = float(rng.uniform(1.4, 3.4))
            world = np.linalg.inv(self.K) @ np.array([u, v, 1.0]) * z
            point_b = true_pose_b[:3, :3].T @ (world - true_pose_b[:3, 3])
            projected = self.K @ point_b
            uv_b = projected[:2] / projected[2]
            if (
                point_b[2] > 0.0
                and 2.0 <= uv_b[0] < self.width - 2.0
                and 2.0 <= uv_b[1] < self.height - 2.0
            ):
                depth[v, u] = z
                pixels_a.append([float(u), float(v)])
                points_b.append(uv_b.tolist())

        matches = _MatchedFeatures(
            points_a=np.asarray(pixels_a, dtype=np.float64),
            points_b=np.asarray(points_b, dtype=np.float64),
            keypoints_a=80,
            keypoints_b=78,
            tentative_matches=len(pixels_a),
        )
        with mock.patch("src.framecrafter_pnp._match_features", return_value=matches):
            result = refine_rgbd_pose_pnp(
                self.image,
                depth,
                self.K,
                self.identity,
                self.image,
                self.K,
                c2w_b=initial_pose_b,
                min_matches=8,
            )

        self.assertTrue(result.success, result.message)
        self.assertGreaterEqual(result.inliers, 34)
        self.assertGreater(result.inlier_ratio, 0.94)
        self.assertLess(result.reprojection_rmse_px or 99.0, 1.0e-3)
        np.testing.assert_allclose(result.refined_c2w_b, true_pose_b, atol=2.0e-4)
        self.assertAlmostEqual(result.rotation_correction_deg or 0.0, 2.0, delta=0.02)
        self.assertAlmostEqual(result.translation_correction or 0.0, 0.03, delta=0.002)

        accepted = gate_pnp_refinement(
            result,
            PnPRefinementGateConfig(
                max_rotation_correction_deg=3.0,
                max_translation_correction=0.04,
                min_inliers=20,
                min_inlier_ratio=0.8,
                max_reprojection_rmse_px=0.1,
            ),
        )
        self.assertTrue(accepted.accepted)
        rejected = gate_pnp_refinement(
            result,
            PnPRefinementGateConfig(
                max_rotation_correction_deg=1.0,
                max_translation_correction=0.01,
                min_inliers=20,
                min_inlier_ratio=0.8,
                max_reprojection_rmse_px=0.1,
            ),
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("rotation_correction_too_large", rejected.failures)
        self.assertIn("translation_correction_too_large", rejected.failures)

    def test_featureless_image_fails_with_specific_code(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is optional")
        result = refine_rgbd_pose_pnp(
            self.image,
            self.depth,
            self.K,
            self.identity,
            self.image,
            self.K,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, "insufficient_features_a")


if __name__ == "__main__":
    unittest.main()
