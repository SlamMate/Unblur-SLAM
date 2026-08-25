"""Overlap-aware FrameCrafter planning and conditioning orchestration.

This module joins the CPU geometry and context-selection primitives without
loading FrameCrafter itself.  Sparse-view candidates are planned between
DROID tracking anchors, while continuous blurry regions remain an independent
source of candidates.  Context images are selected by role and retain explicit
RAW/EVSSM provenance.

All camera poses consumed here are estimated, unaligned ``c2w`` transforms.
The anchor file contributes *indices only*; it never replaces estimated poses
from the provenance-bound frames CSV.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .framecrafter_context import (
    ContextFrameMetadata,
    ContextSelectionConfig,
    ContextSelectionResult,
    EVSSMImageCandidate,
    EVSSMResolver,
    select_framecrafter_contexts,
)
from .framecrafter_overlap import (
    OverlapPlanningConfig,
    approximate_frustum_overlap,
    bidirectional_depth_overlap,
    match_image_overlap_ransac,
    plan_overlap_deficit,
)
from .framecrafter_pipeline import (
    FrameCrafterGenerationBatch,
    FrameRecord,
    TargetView,
    interpolate_c2w,
    laplacian_sharpness,
    read_depth,
    read_rgb,
)
from .framecrafter_pnp import (
    PnPRefinementGateConfig,
    gate_pnp_refinement,
    refine_rgbd_pose_pnp,
)


_EPS = 1.0e-12


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "frame"


def _rotation_delta_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _depth_range(depth: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(depth, dtype=np.float64)
    valid = valid[np.isfinite(valid) & (valid > 1.0e-4)]
    if valid.size < 16:
        return (0.1, 5.0)
    near, far = np.quantile(valid, [0.02, 0.98])
    near = max(1.0e-3, float(near))
    far = max(near + 1.0e-3, float(far))
    return near, far


def _target_id(left: FrameRecord, right: FrameRecord, ordinal: int, alpha: float) -> str:
    return (
        f"syn_{_safe_id(left.frame_id)}_{_safe_id(right.frame_id)}_"
        f"{ordinal:02d}_{alpha:.6f}"
    )


def _propagate_gap_se3_correction(
    c2w: np.ndarray,
    original_right_c2w: np.ndarray,
    refined_right_c2w: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply identity-to-full right-anchor correction at normalized gap time."""

    correction = refined_right_c2w @ np.linalg.inv(original_right_c2w)
    smooth_correction = interpolate_c2w(np.eye(4), correction, float(alpha))
    return smooth_correction @ c2w


@dataclass(frozen=True)
class AdvancedPlannerConfig:
    target_pair_overlap: float = 0.65
    hard_submap_overlap: float = 0.05
    max_inserts: int = 4
    sample_stride: int = 4
    depth_abs_tolerance: float = 0.03
    depth_rel_tolerance: float = 0.03
    include_blurry_regions: bool = True
    blur_quantile: float = 0.30
    laplacian_threshold: Optional[float] = None
    blur_region_inserts: int = 1
    feature_refinement: bool = False
    feature_detector: str = "orb"
    feature_model: str = "essential"
    feature_ambiguity_low: float = 0.15
    feature_ambiguity_high: float = 0.75
    feature_overlap_weight: float = 0.20
    feature_refine_rotation: bool = False
    feature_min_inlier_ratio: float = 0.35
    feature_max_rotation_correction_deg: float = 12.0
    pnp_refinement: bool = False
    pnp_detector: str = "orb"
    pnp_max_features: int = 3000
    pnp_ratio_test: float = 0.75
    pnp_mutual_check: bool = True
    pnp_min_keypoints: int = 12
    pnp_min_matches: int = 8
    pnp_min_depth: float = 1.0e-4
    pnp_max_depth: float = float("inf")
    pnp_min_laplacian_variance: float = 0.0
    pnp_ambiguity_low: float = 0.15
    pnp_ambiguity_high: float = 0.75
    pnp_ransac_reprojection_error_px: float = 3.0
    pnp_ransac_confidence: float = 0.999
    pnp_ransac_iterations: int = 200
    pnp_min_inliers: int = 8
    pnp_min_inlier_ratio: float = 0.35
    pnp_max_reprojection_rmse_px: float = 2.0
    pnp_max_rotation_correction_deg: float = 12.0
    pnp_max_translation_correction: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < float(self.target_pair_overlap) < 1.0:
            raise ValueError("target_pair_overlap must be in (0, 1)")
        if not 0.0 <= float(self.hard_submap_overlap) < float(
            self.target_pair_overlap
        ):
            raise ValueError("hard_submap_overlap must be below target overlap")
        if not 0 <= int(self.max_inserts) <= 4:
            raise ValueError("max_inserts must be in [0, 4]")
        if int(self.sample_stride) < 1:
            raise ValueError("sample_stride must be positive")
        if not 0.0 <= float(self.blur_quantile) <= 1.0:
            raise ValueError("blur_quantile must be in [0, 1]")
        if int(self.blur_region_inserts) < 0:
            raise ValueError("blur_region_inserts cannot be negative")
        if not 0.0 <= float(self.feature_ambiguity_low) <= float(
            self.feature_ambiguity_high
        ) <= 1.0:
            raise ValueError("feature ambiguity interval must be inside [0, 1]")
        if not 0.0 <= float(self.feature_overlap_weight) <= 1.0:
            raise ValueError("feature_overlap_weight must be in [0, 1]")
        if str(self.pnp_detector).lower() not in {"orb", "sift"}:
            raise ValueError("pnp_detector must be 'orb' or 'sift'")
        if (
            int(self.pnp_max_features) != self.pnp_max_features
            or self.pnp_max_features < 1
        ):
            raise ValueError("pnp_max_features must be positive")
        if not 0.0 < float(self.pnp_ratio_test) < 1.0:
            raise ValueError("pnp_ratio_test must be in (0, 1)")
        if (
            int(self.pnp_min_keypoints) != self.pnp_min_keypoints
            or int(self.pnp_min_matches) != self.pnp_min_matches
            or self.pnp_min_keypoints < 4
            or self.pnp_min_matches < 4
        ):
            raise ValueError("PnP keypoint/match minima must be at least 4")
        if not 0.0 < float(self.pnp_min_depth) < float(self.pnp_max_depth):
            raise ValueError("PnP depth bounds must satisfy 0 < min < max")
        if float(self.pnp_min_laplacian_variance) < 0.0:
            raise ValueError("pnp_min_laplacian_variance cannot be negative")
        if not 0.0 <= float(self.pnp_ambiguity_low) <= float(
            self.pnp_ambiguity_high
        ) <= 1.0:
            raise ValueError("PnP ambiguity interval must be inside [0, 1]")
        if float(self.pnp_ransac_reprojection_error_px) <= 0.0:
            raise ValueError("pnp_ransac_reprojection_error_px must be positive")
        if not 0.0 < float(self.pnp_ransac_confidence) < 1.0:
            raise ValueError("pnp_ransac_confidence must be in (0, 1)")
        if (
            int(self.pnp_ransac_iterations) != self.pnp_ransac_iterations
            or self.pnp_ransac_iterations < 1
        ):
            raise ValueError("pnp_ransac_iterations must be positive")
        PnPRefinementGateConfig(
            max_rotation_correction_deg=self.pnp_max_rotation_correction_deg,
            max_translation_correction=self.pnp_max_translation_correction,
            min_inliers=self.pnp_min_inliers,
            min_inlier_ratio=self.pnp_min_inlier_ratio,
            max_reprojection_rmse_px=self.pnp_max_reprojection_rmse_px,
        )


@dataclass(frozen=True)
class PairPlanningRecord:
    left_index: int
    right_index: int
    left_position: int
    right_position: int
    depth_visible_overlap: float
    depth_frustum_overlap: float
    depth_target_coverage: float
    coarse_frustum_overlap: float
    feature_attempted: bool
    feature_available: bool
    feature_success: bool
    feature_overlap: Optional[float]
    feature_inlier_ratio: Optional[float]
    feature_inliers: Optional[int]
    pnp_attempted: bool
    pnp_available: bool
    pnp_success: bool
    pnp_accepted: bool
    pnp_failure: Optional[str]
    pnp_inliers: Optional[int]
    pnp_inlier_ratio: Optional[float]
    pnp_reprojection_rmse_px: Optional[float]
    pnp_rotation_correction_deg: Optional[float]
    pnp_translation_correction: Optional[float]
    pnp_refined_right_c2w: Optional[tuple[tuple[float, ...], ...]]
    measured_overlap: float
    insert_count: int
    alphas: tuple[float, ...]
    split_submap: bool
    budget_exceeded: bool
    reason: str
    rotation_refined: bool
    rotation_correction_deg: Optional[float]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["alphas"] = list(self.alphas)
        return payload


@dataclass(frozen=True)
class AdvancedPlanningResult:
    targets: tuple[TargetView, ...]
    pairs: tuple[PairPlanningRecord, ...]
    anchor_source_indices: tuple[int, ...]
    blur_threshold: float
    sparse_target_count: int
    blurry_target_count: int
    submap_boundaries: tuple[tuple[int, int], ...]


def load_anchor_source_indices(
    path: Path | str, frames: Sequence[FrameRecord]
) -> tuple[int, ...]:
    """Load DROID anchor indices from ``video.npz`` timestamps or a text list."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"DROID anchor file does not exist: {resolved}")
    if resolved.suffix.lower() == ".npz":
        with np.load(resolved, allow_pickle=False) as arrays:
            if "timestamps" not in arrays:
                raise ValueError("DROID video.npz is missing timestamps")
            values = np.asarray(arrays["timestamps"], dtype=np.float64).reshape(-1)
    else:
        tokens = resolved.read_text(encoding="utf-8").replace(",", " ").split()
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("DROID anchor list needs at least two finite indices")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1.0e-6):
        raise ValueError("DROID anchor timestamps must map to integer source indices")
    indices = tuple(int(value) for value in rounded)
    if list(indices) != sorted(set(indices)):
        raise ValueError("DROID anchor indices must be unique and increasing")
    valid = {int(frame.source_index) for frame in frames}
    missing = [index for index in indices if index not in valid]
    if missing:
        raise ValueError(f"DROID anchors are absent from frames CSV: {missing[:8]}")
    return indices


def _target(
    frames: Sequence[FrameRecord],
    left_position: int,
    right_position: int,
    alpha: float,
    reasons: Sequence[str],
    ordinal: int,
    *,
    refined_right_c2w: Optional[np.ndarray] = None,
) -> TargetView:
    left, right = frames[left_position], frames[right_position]
    timestamp = (1.0 - alpha) * left.timestamp + alpha * right.timestamp
    # Follow the estimated full-frame trajectory between DROID anchors instead
    # of assuming a single straight SE(3) chord across a long sparse gap.  The
    # target generally falls between two original video timestamps and remains
    # a novel view, while its camera is locally consistent with RGB-D support.
    timestamps = np.asarray(
        [frame.timestamp for frame in frames[left_position : right_position + 1]],
        dtype=np.float64,
    )
    relative_right = int(np.searchsorted(timestamps, timestamp, side="right"))
    support_right_position = min(
        right_position, left_position + max(1, relative_right)
    )
    support_left_position = max(left_position, support_right_position - 1)
    support_left = frames[support_left_position]
    support_right = frames[support_right_position]
    denominator = support_right.timestamp - support_left.timestamp
    local_alpha = (
        0.0
        if abs(denominator) <= _EPS
        else float(np.clip((timestamp - support_left.timestamp) / denominator, 0.0, 1.0))
    )
    c2w = interpolate_c2w(support_left.c2w, support_right.c2w, local_alpha)
    if refined_right_c2w is not None:
        # Propagate the complete right-anchor SE(3) correction over the whole
        # local full-frame trajectory.  With camera-to-world transforms the
        # left-multiplicative correction below is identity at gap alpha=0 and
        # exactly maps the original right anchor onto the refined one at 1.
        c2w = _propagate_gap_se3_correction(
            c2w, right.c2w, refined_right_c2w, alpha
        )
    return TargetView(
        target_id=_target_id(left, right, ordinal, alpha),
        left_index=left.source_index,
        right_index=right.source_index,
        left_position=left_position,
        right_position=right_position,
        timestamp=timestamp,
        alpha=alpha,
        c2w=c2w,
        intrinsics=(
            (1.0 - local_alpha) * support_left.intrinsics
            + local_alpha * support_right.intrinsics
        ),
        reasons=tuple(dict.fromkeys(str(reason) for reason in reasons)),
    )


def plan_anchor_overlap_targets(
    frames: Sequence[FrameRecord],
    anchor_path: Path | str,
    *,
    depth_scale: float,
    config: AdvancedPlannerConfig = AdvancedPlannerConfig(),
    only_gap: Optional[tuple[int, int]] = None,
) -> AdvancedPlanningResult:
    """Plan sparse and blur-region views from estimated poses and RGB-D overlap."""

    if len(frames) < 2:
        raise ValueError("advanced planning requires at least two source frames")
    if not math.isfinite(float(depth_scale)) or float(depth_scale) <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    positions = {int(frame.source_index): position for position, frame in enumerate(frames)}
    anchors = load_anchor_source_indices(anchor_path, frames)
    scores = np.asarray(
        [
            float(frame.sharpness)
            if frame.sharpness is not None
            else laplacian_sharpness(read_rgb(frame.rgb_path))
            for frame in frames
        ],
        dtype=np.float64,
    )
    threshold = (
        float(config.laplacian_threshold)
        if config.laplacian_threshold is not None
        else float(np.quantile(scores, config.blur_quantile))
    )
    depth_cache: dict[int, np.ndarray] = {}
    rgb_cache: dict[int, np.ndarray] = {}

    def rgb(index: int) -> np.ndarray:
        if index not in rgb_cache:
            rgb_cache[index] = read_rgb(frames[positions[index]].rgb_path)
        return rgb_cache[index]

    def depth(index: int) -> np.ndarray:
        if index not in depth_cache:
            frame = frames[positions[index]]
            if frame.depth_path is None:
                raise ValueError(f"overlap planning requires depth for source {index}")
            depth_cache[index] = read_depth(frame.depth_path, depth_scale)
        return depth_cache[index]

    planning_config = OverlapPlanningConfig(
        target_pair_overlap=config.target_pair_overlap,
        hard_submap_overlap=config.hard_submap_overlap,
        max_inserts=config.max_inserts,
        split_if_budget_exceeded=True,
    )
    pair_records: list[PairPlanningRecord] = []
    targets_by_gap: dict[tuple[int, int], dict[float, set[str]]] = {}
    refined_right_poses: dict[tuple[int, int], np.ndarray] = {}
    submap_boundaries: list[tuple[int, int]] = []

    for left_index, right_index in zip(anchors[:-1], anchors[1:]):
        if only_gap is not None and (left_index, right_index) != tuple(only_gap):
            continue
        left_position, right_position = positions[left_index], positions[right_index]
        left, right = frames[left_position], frames[right_position]
        left_depth, right_depth = depth(left_index), depth(right_index)
        depth_overlap = bidirectional_depth_overlap(
            left_depth,
            left.intrinsics,
            left.c2w,
            right_depth,
            right.intrinsics,
            right.c2w,
            sample_stride=config.sample_stride,
            depth_abs_tolerance=config.depth_abs_tolerance,
            depth_rel_tolerance=config.depth_rel_tolerance,
        )
        image_shape_left, image_shape_right = left_depth.shape, right_depth.shape
        frustum = approximate_frustum_overlap(
            left.intrinsics,
            left.c2w,
            image_shape_left,
            right.intrinsics,
            right.c2w,
            image_shape_right,
            depth_range_a=_depth_range(left_depth),
            depth_range_b=_depth_range(right_depth),
        )
        measured = float(depth_overlap.symmetric_visible_overlap)
        pnp_attempted = bool(
            config.pnp_refinement
            and config.pnp_ambiguity_low <= measured <= config.pnp_ambiguity_high
        )
        pnp_result = None
        pnp_gate = None
        pnp_accepted = False
        pnp_failure = None
        refined_right_pose = None
        if pnp_attempted:
            pnp_result = refine_rgbd_pose_pnp(
                rgb(left_index),
                left_depth,
                left.intrinsics,
                left.c2w,
                rgb(right_index),
                right.intrinsics,
                c2w_b=right.c2w,
                detector=config.pnp_detector,
                max_features=config.pnp_max_features,
                ratio_test=config.pnp_ratio_test,
                mutual_check=config.pnp_mutual_check,
                min_keypoints=config.pnp_min_keypoints,
                min_matches=config.pnp_min_matches,
                min_depth=config.pnp_min_depth,
                max_depth=config.pnp_max_depth,
                min_laplacian_variance=config.pnp_min_laplacian_variance,
                ransac_reprojection_error_px=(
                    config.pnp_ransac_reprojection_error_px
                ),
                ransac_confidence=config.pnp_ransac_confidence,
                ransac_iterations=config.pnp_ransac_iterations,
            )
            if pnp_result.success:
                pnp_gate = gate_pnp_refinement(
                    pnp_result,
                    PnPRefinementGateConfig(
                        max_rotation_correction_deg=(
                            config.pnp_max_rotation_correction_deg
                        ),
                        max_translation_correction=(
                            config.pnp_max_translation_correction
                        ),
                        min_inliers=config.pnp_min_inliers,
                        min_inlier_ratio=config.pnp_min_inlier_ratio,
                        max_reprojection_rmse_px=(
                            config.pnp_max_reprojection_rmse_px
                        ),
                    ),
                )
                pnp_accepted = bool(pnp_gate.accepted)
            if not pnp_result.success:
                pnp_failure = pnp_result.failure_code or "refinement_failed"
            elif not pnp_accepted:
                pnp_failure = ",".join(pnp_gate.failures) if pnp_gate else "gate_failed"
            if pnp_accepted:
                if pnp_result.refined_c2w_b is None:
                    raise RuntimeError("accepted PnP refinement has no refined c2w")
                refined_right_pose = np.asarray(
                    pnp_result.refined_c2w_b, dtype=np.float64
                )
                refined_right_poses[(left_position, right_position)] = (
                    refined_right_pose
                )
                # PnP changes both translation and rotation, so all geometry
                # evidence used by insertion planning must be recomputed.
                depth_overlap = bidirectional_depth_overlap(
                    left_depth,
                    left.intrinsics,
                    left.c2w,
                    right_depth,
                    right.intrinsics,
                    refined_right_pose,
                    sample_stride=config.sample_stride,
                    depth_abs_tolerance=config.depth_abs_tolerance,
                    depth_rel_tolerance=config.depth_rel_tolerance,
                )
                frustum = approximate_frustum_overlap(
                    left.intrinsics,
                    left.c2w,
                    image_shape_left,
                    right.intrinsics,
                    refined_right_pose,
                    image_shape_right,
                    depth_range_a=_depth_range(left_depth),
                    depth_range_b=_depth_range(right_depth),
                )
                measured = float(depth_overlap.symmetric_visible_overlap)
        feature_attempted = bool(
            config.feature_refinement
            and config.feature_ambiguity_low <= measured <= config.feature_ambiguity_high
        )
        feature = None
        refined_rotation = None
        correction = None
        if feature_attempted:
            feature = match_image_overlap_ransac(
                rgb(left_index),
                rgb(right_index),
                detector=config.feature_detector,
                model_type=config.feature_model,
                intrinsics_a=left.intrinsics,
                intrinsics_b=right.intrinsics,
            )
            if feature.success:
                weight = float(config.feature_overlap_weight)
                measured = float(
                    np.clip(
                        (1.0 - weight) * measured + weight * feature.overlap_score,
                        0.0,
                        1.0,
                    )
                )
                if (
                    config.feature_refine_rotation
                    and not pnp_accepted
                    and feature.relative_rotation is not None
                    and feature.inlier_ratio >= config.feature_min_inlier_ratio
                ):
                    candidate_rotation = left.c2w[:3, :3] @ feature.relative_rotation.T
                    candidate_pose = right.c2w.copy()
                    candidate_pose[:3, :3] = candidate_rotation
                    correction = _rotation_delta_deg(right.c2w, candidate_pose)
                    if correction <= config.feature_max_rotation_correction_deg:
                        refined_rotation = candidate_rotation
                        feature_refined_pose = right.c2w.copy()
                        feature_refined_pose[:3, :3] = candidate_rotation
                        refined_right_poses[(left_position, right_position)] = (
                            feature_refined_pose
                        )
        insertion = plan_overlap_deficit(measured, planning_config)
        if insertion.split_submap:
            submap_boundaries.append((left_index, right_index))
        else:
            gap = targets_by_gap.setdefault((left_position, right_position), {})
            for alpha in insertion.alphas:
                reasons = {"low_view_overlap", "anchor_sparse_gap"}
                if pnp_accepted:
                    reasons.add("pnp_pose_refined")
                gap.setdefault(float(alpha), set()).update(reasons)
        pair_records.append(
            PairPlanningRecord(
                left_index=left_index,
                right_index=right_index,
                left_position=left_position,
                right_position=right_position,
                depth_visible_overlap=float(depth_overlap.symmetric_visible_overlap),
                depth_frustum_overlap=float(depth_overlap.symmetric_frustum_overlap),
                depth_target_coverage=float(depth_overlap.symmetric_target_coverage),
                coarse_frustum_overlap=float(frustum.symmetric_overlap),
                feature_attempted=feature_attempted,
                feature_available=False if feature is None else bool(feature.available),
                feature_success=False if feature is None else bool(feature.success),
                feature_overlap=(
                    None if feature is None or not feature.success else float(feature.overlap_score)
                ),
                feature_inlier_ratio=(
                    None if feature is None or not feature.success else float(feature.inlier_ratio)
                ),
                feature_inliers=(
                    None if feature is None or not feature.success else int(feature.inliers)
                ),
                pnp_attempted=pnp_attempted,
                pnp_available=False if pnp_result is None else bool(pnp_result.available),
                pnp_success=False if pnp_result is None else bool(pnp_result.success),
                pnp_accepted=pnp_accepted,
                pnp_failure=pnp_failure,
                pnp_inliers=(
                    None if pnp_result is None else int(pnp_result.inliers)
                ),
                pnp_inlier_ratio=(
                    None if pnp_result is None else float(pnp_result.inlier_ratio)
                ),
                pnp_reprojection_rmse_px=(
                    None
                    if pnp_result is None
                    else pnp_result.reprojection_rmse_px
                ),
                pnp_rotation_correction_deg=(
                    None
                    if pnp_result is None
                    else pnp_result.rotation_correction_deg
                ),
                pnp_translation_correction=(
                    None
                    if pnp_result is None
                    else pnp_result.translation_correction
                ),
                pnp_refined_right_c2w=(
                    None
                    if not pnp_accepted or refined_right_pose is None
                    else tuple(
                        tuple(float(value) for value in row)
                        for row in refined_right_pose
                    )
                ),
                measured_overlap=measured,
                insert_count=insertion.insert_count,
                alphas=insertion.alphas,
                split_submap=insertion.split_submap,
                budget_exceeded=insertion.budget_exceeded,
                reason=insertion.reason,
                rotation_refined=refined_rotation is not None or pnp_accepted,
                rotation_correction_deg=(
                    pnp_result.rotation_correction_deg
                    if pnp_accepted and pnp_result is not None
                    else correction
                ),
            )
        )

    sparse_target_count = sum(len(values) for values in targets_by_gap.values())
    blurry_added = 0
    if config.include_blurry_regions and only_gap is None:
        inserts = min(config.max_inserts, max(0, int(config.blur_region_inserts)))
        for left_position in range(len(frames) - 1):
            right_position = left_position + 1
            if scores[left_position] >= threshold or scores[right_position] >= threshold:
                continue
            gap = targets_by_gap.setdefault((left_position, right_position), {})
            for ordinal in range(inserts):
                alpha = float((ordinal + 1) / (inserts + 1))
                reasons = gap.setdefault(alpha, set())
                before = bool(reasons)
                reasons.add("consecutive_blurry_region")
                if not before:
                    blurry_added += 1

    targets: list[TargetView] = []
    for (left_position, right_position), alpha_reasons in sorted(
        targets_by_gap.items(), key=lambda item: item[0]
    ):
        refined = refined_right_poses.get((left_position, right_position))
        for ordinal, alpha in enumerate(sorted(alpha_reasons)):
            targets.append(
                _target(
                    frames,
                    left_position,
                    right_position,
                    alpha,
                    sorted(alpha_reasons[alpha]),
                    ordinal,
                    refined_right_c2w=refined,
                )
            )
    targets.sort(key=lambda value: (value.timestamp, value.left_position, value.alpha))
    return AdvancedPlanningResult(
        targets=tuple(targets),
        pairs=tuple(pair_records),
        anchor_source_indices=anchors,
        blur_threshold=threshold,
        sparse_target_count=sparse_target_count,
        blurry_target_count=blurry_added,
        submap_boundaries=tuple(submap_boundaries),
    )


def select_advanced_scene_targets(
    targets: Sequence[TargetView], max_targets: Optional[int]
) -> list[TargetView]:
    """Cap targets scene-wide while retaining sparse-overlap gaps first."""

    chronological = sorted(
        targets, key=lambda value: (value.timestamp, value.left_position, value.alpha)
    )
    if max_targets is None or len(chronological) <= int(max_targets):
        return chronological
    count = max(0, int(max_targets))
    if count == 0:
        return []
    priority = [value for value in chronological if "low_view_overlap" in value.reasons]
    ordinary = [value for value in chronological if "low_view_overlap" not in value.reasons]

    def uniform(values: Sequence[TargetView], amount: int) -> list[TargetView]:
        if amount <= 0:
            return []
        if amount >= len(values):
            return list(values)
        positions = np.rint(np.linspace(0, len(values) - 1, amount)).astype(int)
        return [values[int(position)] for position in positions]

    selected = (
        uniform(priority, count)
        if len(priority) >= count
        else [*priority, *uniform(ordinary, count - len(priority))]
    )
    return sorted(selected, key=lambda value: (value.timestamp, value.left_position, value.alpha))


def load_evssm_resolver(
    metadata_path: Optional[Path | str], *, require_production: bool
) -> tuple[Optional[EVSSMResolver], dict[int, Mapping[str, Any]]]:
    """Load audited EVSSM precompute metadata into the context resolver."""

    if metadata_path in (None, ""):
        return None, {}
    path = Path(metadata_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "unblur_slam.framecrafter_evssm_precompute.v1":
        raise ValueError("unsupported FrameCrafter EVSSM metadata schema")
    if payload.get("uses_ground_truth_pose") is not False:
        raise ValueError("EVSSM metadata must declare uses_ground_truth_pose=false")
    if require_production and (
        payload.get("test_only") is not False
        or payload.get("production_eligible") is not True
    ):
        raise ValueError("test-only/ineligible EVSSM metadata cannot condition production")
    records = payload.get("frames")
    if not isinstance(records, list):
        raise ValueError("EVSSM metadata frames must be a list")
    mapping: dict[object, EVSSMImageCandidate] = {}
    by_index: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("EVSSM frame record must be an object")
        source_index = int(record["source_index"])
        output = Path(str(record["output_path"])).expanduser().resolve()
        if not output.is_file() or _sha256_file(output) != str(record["output_sha256"]):
            raise ValueError(f"EVSSM output hash mismatch for source {source_index}")
        candidate = EVSSMImageCandidate(
            path=output,
            confidence=float(record["confidence"]),
            sharpness_gain=float(record["sharpness_gain"]),
            consistency=float(record.get("consistency", record["image_consistency"])),
            provider="evssm_precompute",
        )
        mapping[source_index] = candidate
        mapping[str(record["frame_id"])] = candidate
        by_index[source_index] = record
    return EVSSMResolver(precomputed=mapping), by_index


def _candidate_target_overlap(
    frame: FrameRecord,
    target: TargetView,
    image_shape: tuple[int, int],
    depth_range: tuple[float, float],
) -> float:
    result = approximate_frustum_overlap(
        frame.intrinsics,
        frame.c2w,
        image_shape,
        target.intrinsics,
        target.c2w,
        image_shape,
        depth_range_a=depth_range,
        depth_range_b=depth_range,
        grid_shape=(8, 12),
        depth_samples=3,
    )
    return float(result.symmetric_overlap)


def build_role_aware_batches(
    frames: Sequence[FrameRecord],
    targets: Sequence[TargetView],
    *,
    context_config: ContextSelectionConfig,
    depth_scale: float,
    context_search_radius: int = 32,
    evssm_resolver: Optional[EVSSMResolver] = None,
    evssm_records: Mapping[int, Mapping[str, Any]] = {},
) -> tuple[list[FrameCrafterGenerationBatch], dict[str, dict[str, object]]]:
    """Batch only alphas of one gap and attach role-aware conditioning images."""

    if not targets:
        return [], {}
    if int(context_search_radius) < 1:
        raise ValueError("context_search_radius must be positive")
    grouped: dict[tuple[int, int], list[TargetView]] = {}
    for target in targets:
        grouped.setdefault((target.left_position, target.right_position), []).append(target)
    sharpness_values = np.asarray(
        [
            float(frame.sharpness)
            if frame.sharpness is not None
            else laplacian_sharpness(read_rgb(frame.rgb_path))
            for frame in frames
        ],
        dtype=np.float64,
    )
    blur_cutoff = float(np.quantile(sharpness_values, context_config.blur_quantile))
    batches: list[FrameCrafterGenerationBatch] = []
    reports: dict[str, dict[str, object]] = {}
    batch_number = 0
    for gap, gap_targets in sorted(
        grouped.items(), key=lambda item: min(value.timestamp for value in item[1])
    ):
        left_position, right_position = gap
        gap_targets = sorted(gap_targets, key=lambda value: value.alpha)
        maximum_targets = min(4, 10 - int(context_config.context_budget))
        for start in range(0, len(gap_targets), maximum_targets):
            chunk = gap_targets[start : start + maximum_targets]
            representative = min(chunk, key=lambda value: abs(value.alpha - 0.5))
            begin = max(0, left_position - int(context_search_radius))
            end = min(len(frames), right_position + int(context_search_radius) + 1)
            endpoint = frames[left_position]
            if endpoint.depth_path is None:
                target_depth_range = (0.1, 5.0)
                image_shape = read_rgb(endpoint.rgb_path).shape[:2]
            else:
                endpoint_depth = read_depth(endpoint.depth_path, depth_scale)
                target_depth_range = _depth_range(endpoint_depth)
                image_shape = endpoint_depth.shape
            metadata: list[ContextFrameMetadata] = []
            for position in range(begin, end):
                frame = frames[position]
                evssm = evssm_records.get(frame.source_index, {})
                metadata.append(
                    ContextFrameMetadata(
                        frame=frame,
                        position=position,
                        overlap=_candidate_target_overlap(
                            frame, representative, image_shape, target_depth_range
                        ),
                        reliability=1.0,
                        sharpness=float(sharpness_values[position]),
                        is_blurry=bool(sharpness_values[position] <= blur_cutoff),
                        evssm_path=evssm.get("output_path"),
                        evssm_confidence=evssm.get("confidence"),
                        evssm_sharpness_gain=evssm.get("sharpness_gain"),
                        evssm_consistency=evssm.get(
                            "consistency", evssm.get("image_consistency")
                        ),
                        evssm_provider="evssm_precompute",
                    )
                )
            selection: ContextSelectionResult = select_framecrafter_contexts(
                representative,
                metadata,
                context_config,
                evssm_resolver,
            )
            batch_id = f"batch_{batch_number:05d}"
            batch_number += 1
            batch = FrameCrafterGenerationBatch(
                batch_id=batch_id,
                contexts=selection.frame_records,
                targets=tuple(chunk),
                max_endpoint_position_span=max(1, right_position - left_position),
            )
            batches.append(batch)
            conditioning = []
            for selected in selection.contexts:
                conditioning.append(
                    {
                        "source_index": selected.frame.source_index,
                        "frame_id": selected.frame.frame_id,
                        "role": selected.role,
                        "position": selected.position,
                        "score": selected.score.as_dict(),
                        "requested_mode": selected.provenance.requested_mode,
                        "resolved_mode": selected.provenance.resolved_mode,
                        "provider": selected.provenance.provider,
                        "raw_path": str(selected.provenance.raw_path),
                        "resolved_path": str(selected.provenance.resolved_path),
                        "resolved_sha256": _sha256_file(selected.provenance.resolved_path),
                        "fallback_reason": selected.provenance.fallback_reason,
                        "evssm_confidence": selected.provenance.evssm_confidence,
                        "evssm_sharpness_gain": selected.provenance.evssm_sharpness_gain,
                        "evssm_consistency": selected.provenance.evssm_consistency,
                        "evssm_local_gate": (
                            None
                            if selected.provenance.evssm_local_gate is None
                            else selected.provenance.evssm_local_gate.as_dict()
                        ),
                    }
                )
            reports[batch_id] = {
                "batch_policy": "same_gap_multi_alpha_v1",
                "context_selection_policy": "role_aware_overlap_v1",
                "representative_target_id": representative.target_id,
                "requested_image_mode": context_config.image_mode,
                "conditioning": conditioning,
            }
    return batches, reports


__all__ = [
    "AdvancedPlannerConfig",
    "AdvancedPlanningResult",
    "PairPlanningRecord",
    "build_role_aware_batches",
    "load_anchor_source_indices",
    "load_evssm_resolver",
    "plan_anchor_overlap_targets",
    "select_advanced_scene_targets",
]
