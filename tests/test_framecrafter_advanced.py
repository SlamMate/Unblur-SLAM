#!/usr/bin/env python3
"""CPU contracts for overlap-aware planning and role-aware batching."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from src.framecrafter_advanced import (
    AdvancedPlannerConfig,
    _propagate_gap_se3_correction,
    build_role_aware_batches,
    load_anchor_source_indices,
    plan_anchor_overlap_targets,
)
from src.framecrafter_context import ContextSelectionConfig
from src.framecrafter_pipeline import FrameRecord, interpolate_c2w
from src.framecrafter_pnp import PnPRefinementResult


class AdvancedFrameCrafterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frames: list[FrameRecord] = []
        intrinsics = np.array(
            [[70.0, 0.0, 39.5], [0.0, 70.0, 29.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        for index in range(9):
            image = np.full((60, 80, 3), 40 + index * 8, dtype=np.uint8)
            image[10:50:2, 12 + index : 55 + index] = 220
            rgb = self.root / f"rgb_{index}.png"
            depth = self.root / f"depth_{index}.npy"
            Image.fromarray(image).save(rgb)
            np.save(depth, np.full((60, 80), 4.0, dtype=np.float32))
            c2w = np.eye(4, dtype=np.float64)
            c2w[0, 3] = index * 0.06
            self.frames.append(
                FrameRecord(
                    source_index=index,
                    frame_id=f"frame_{index}",
                    timestamp=float(index),
                    rgb_path=rgb,
                    depth_path=depth,
                    c2w=c2w,
                    intrinsics=intrinsics,
                    sharpness=float(index + 1),
                )
            )
        self.anchors = self.root / "anchors.txt"
        self.anchors.write_text("0\n8\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_anchor_overlap_planning_and_same_gap_context_batch(self) -> None:
        self.assertEqual(load_anchor_source_indices(self.anchors, self.frames), (0, 8))
        result = plan_anchor_overlap_targets(
            self.frames,
            self.anchors,
            depth_scale=1.0,
            config=AdvancedPlannerConfig(
                target_pair_overlap=0.95,
                hard_submap_overlap=0.01,
                max_inserts=4,
                include_blurry_regions=False,
            ),
        )
        self.assertEqual(len(result.pairs), 1)
        self.assertGreater(len(result.targets), 0)
        self.assertTrue(
            all((target.left_index, target.right_index) == (0, 8) for target in result.targets)
        )
        batches, provenance = build_role_aware_batches(
            self.frames,
            result.targets,
            context_config=ContextSelectionConfig(
                context_budget=6,
                min_contexts=3,
                local_blurry_count=2,
                sharp_context_count=2,
                min_sharp_overlap=0.0,
                image_mode="raw",
            ),
            depth_scale=1.0,
            context_search_radius=8,
        )
        self.assertTrue(batches)
        for batch in batches:
            self.assertLessEqual(len(batch.targets), 4)
            self.assertTrue(
                all(
                    (target.left_position, target.right_position) == (0, 8)
                    for target in batch.targets
                )
            )
            selected = provenance[batch.batch_id]["conditioning"]
            roles = {record["role"] for record in selected}
            self.assertIn("endpoint_left", roles)
            self.assertIn("endpoint_right", roles)
            self.assertTrue(any(role.startswith("sharp") for role in roles))
            self.assertTrue(all(record["resolved_mode"] == "raw" for record in selected))
            self.assertTrue(all("evssm_local_gate" in record for record in selected))
            self.assertTrue(all(record["evssm_local_gate"] is None for record in selected))

    def test_anchor_file_rejects_non_integral_or_missing_indices(self) -> None:
        self.anchors.write_text("0\n4.5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integer"):
            load_anchor_source_indices(self.anchors, self.frames)
        self.anchors.write_text("0\n99\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "absent"):
            load_anchor_source_indices(self.anchors, self.frames)

    @staticmethod
    def overlap_result(value: float) -> SimpleNamespace:
        return SimpleNamespace(
            symmetric_visible_overlap=value,
            symmetric_frustum_overlap=min(1.0, value + 0.1),
            symmetric_target_coverage=max(0.0, value - 0.1),
        )

    @staticmethod
    def pnp_result(
        refined_pose: np.ndarray,
        *,
        translation_correction: float,
        rotation_correction_deg: float = 0.0,
    ) -> PnPRefinementResult:
        return PnPRefinementResult(
            available=True,
            success=True,
            failure_code=None,
            message="synthetic refinement",
            detector="orb",
            keypoints_a=80,
            keypoints_b=78,
            tentative_matches=42,
            depth_supported_matches=40,
            inliers=36,
            inlier_ratio=0.9,
            reprojection_rmse_px=0.35,
            refined_c2w_b=refined_pose,
            rotation_correction_deg=rotation_correction_deg,
            translation_correction=translation_correction,
        )

    def test_accepted_pnp_recomputes_overlap_and_smooths_full_se3_correction(self) -> None:
        refined = self.frames[8].c2w.copy()
        angle = np.deg2rad(2.0)
        refined[:3, :3] = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        refined[0, 3] += 0.08
        pnp = self.pnp_result(
            refined,
            translation_correction=0.08,
            rotation_correction_deg=2.0,
        )
        overlap = [self.overlap_result(0.40), self.overlap_result(0.45)]
        frustum = [SimpleNamespace(symmetric_overlap=0.52), SimpleNamespace(symmetric_overlap=0.41)]
        with (
            mock.patch(
                "src.framecrafter_advanced.refine_rgbd_pose_pnp",
                return_value=pnp,
            ) as refine_mock,
            mock.patch(
                "src.framecrafter_advanced.bidirectional_depth_overlap",
                side_effect=overlap,
            ) as depth_overlap_mock,
            mock.patch(
                "src.framecrafter_advanced.approximate_frustum_overlap",
                side_effect=frustum,
            ) as frustum_mock,
        ):
            result = plan_anchor_overlap_targets(
                self.frames,
                self.anchors,
                depth_scale=1.0,
                config=AdvancedPlannerConfig(
                    target_pair_overlap=0.80,
                    hard_submap_overlap=0.01,
                    max_inserts=4,
                    include_blurry_regions=False,
                    pnp_refinement=True,
                    pnp_ambiguity_low=0.0,
                    pnp_ambiguity_high=1.0,
                    pnp_min_inliers=20,
                    pnp_min_inlier_ratio=0.8,
                    pnp_max_reprojection_rmse_px=1.0,
                    pnp_max_rotation_correction_deg=2.0,
                    pnp_max_translation_correction=0.10,
                ),
            )

        refine_mock.assert_called_once()
        self.assertEqual(depth_overlap_mock.call_count, 2)
        self.assertEqual(frustum_mock.call_count, 2)
        pair = result.pairs[0]
        self.assertTrue(pair.pnp_attempted)
        self.assertTrue(pair.pnp_success)
        self.assertTrue(pair.pnp_accepted)
        self.assertIsNone(pair.pnp_failure)
        self.assertEqual(pair.pnp_inliers, 36)
        self.assertAlmostEqual(pair.pnp_inlier_ratio or 0.0, 0.9)
        self.assertAlmostEqual(pair.pnp_reprojection_rmse_px or 0.0, 0.35)
        self.assertAlmostEqual(pair.pnp_rotation_correction_deg or 0.0, 2.0)
        self.assertAlmostEqual(pair.depth_visible_overlap, 0.45)
        self.assertAlmostEqual(pair.coarse_frustum_overlap, 0.41)
        np.testing.assert_allclose(np.asarray(pair.pnp_refined_right_c2w), refined)
        self.assertTrue(result.targets)
        for target in result.targets:
            self.assertIn("pnp_pose_refined", target.reasons)
            uncorrected = np.eye(4)
            uncorrected[0, 3] = target.timestamp * 0.06
            full_correction = refined @ np.linalg.inv(self.frames[8].c2w)
            smooth_correction = interpolate_c2w(
                np.eye(4), full_correction, target.alpha
            )
            np.testing.assert_allclose(
                target.c2w, smooth_correction @ uncorrected, atol=1.0e-9
            )
        left_endpoint = _propagate_gap_se3_correction(
            self.frames[0].c2w, self.frames[8].c2w, refined, 0.0
        )
        right_endpoint = _propagate_gap_se3_correction(
            self.frames[8].c2w, self.frames[8].c2w, refined, 1.0
        )
        np.testing.assert_allclose(left_endpoint, self.frames[0].c2w, atol=1.0e-9)
        np.testing.assert_allclose(right_endpoint, refined, atol=1.0e-9)

    def test_rejected_pnp_is_audited_and_never_changes_pose_or_overlap(self) -> None:
        unsafe = self.frames[8].c2w.copy()
        unsafe[0, 3] += 0.80
        pnp = self.pnp_result(unsafe, translation_correction=0.80)
        overlap = self.overlap_result(0.40)
        with (
            mock.patch(
                "src.framecrafter_advanced.refine_rgbd_pose_pnp",
                return_value=pnp,
            ),
            mock.patch(
                "src.framecrafter_advanced.bidirectional_depth_overlap",
                return_value=overlap,
            ) as depth_overlap_mock,
            mock.patch(
                "src.framecrafter_advanced.approximate_frustum_overlap",
                return_value=SimpleNamespace(symmetric_overlap=0.52),
            ) as frustum_mock,
        ):
            result = plan_anchor_overlap_targets(
                self.frames,
                self.anchors,
                depth_scale=1.0,
                config=AdvancedPlannerConfig(
                    target_pair_overlap=0.80,
                    hard_submap_overlap=0.01,
                    max_inserts=4,
                    include_blurry_regions=False,
                    pnp_refinement=True,
                    pnp_ambiguity_low=0.0,
                    pnp_ambiguity_high=1.0,
                    pnp_min_inliers=20,
                    pnp_min_inlier_ratio=0.8,
                    pnp_max_reprojection_rmse_px=1.0,
                    pnp_max_rotation_correction_deg=2.0,
                    pnp_max_translation_correction=0.10,
                ),
            )

        self.assertEqual(depth_overlap_mock.call_count, 1)
        self.assertEqual(frustum_mock.call_count, 1)
        pair = result.pairs[0]
        self.assertTrue(pair.pnp_attempted)
        self.assertTrue(pair.pnp_success)
        self.assertFalse(pair.pnp_accepted)
        self.assertIn("translation_correction_too_large", pair.pnp_failure or "")
        self.assertIsNone(pair.pnp_refined_right_c2w)
        self.assertFalse(pair.rotation_refined)
        self.assertAlmostEqual(pair.depth_visible_overlap, 0.40)
        self.assertTrue(result.targets)
        for target in result.targets:
            self.assertNotIn("pnp_pose_refined", target.reasons)
            self.assertAlmostEqual(target.c2w[0, 3], target.timestamp * 0.06)


if __name__ == "__main__":
    unittest.main()
