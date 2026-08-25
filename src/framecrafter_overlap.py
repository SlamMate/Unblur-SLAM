"""CPU-only view-overlap estimates for FrameCrafter target planning.

The functions in this module deliberately operate on estimated camera poses and
depth maps supplied by the caller.  They do not load, align, or otherwise use
ground-truth trajectories.  Camera poses follow the rest of the FrameCrafter
preprocessor: ``c2w`` is an OpenCV camera-to-world transform (x right, y down,
z forward).

Three levels of evidence are exposed:

* :func:`bidirectional_depth_overlap` projects measured depth in both
  directions and checks visibility against the destination depth buffer.
* :func:`approximate_frustum_overlap` cheaply projects a lattice of frustum
  samples when dense depth is unavailable or a first-stage screen is desired.
* :func:`match_image_overlap_ransac` optionally performs an OpenCV feature and
  RANSAC refinement.  OpenCV is imported lazily, so geometry-only deployments
  do not depend on it.

All geometry is NumPy-only and device independent.  Public entry points reject
invalid shapes and non-finite values rather than silently producing a planning
decision from corrupt camera state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


_EPS = 1.0e-12
_PIXEL_EPS = 1.0e-7


def _finite_array(value: Any, *, name: str, ndim: Optional[int] = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _intrinsics(value: Any, name: str) -> np.ndarray:
    result = _finite_array(value, name=name)
    if result.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {result.shape}")
    if result[0, 0] <= 0.0 or result[1, 1] <= 0.0:
        raise ValueError(f"{name} focal lengths must be positive")
    if abs(float(np.linalg.det(result))) <= _EPS:
        raise ValueError(f"{name} must be invertible")
    return result


def _c2w(value: Any, name: str) -> np.ndarray:
    result = _finite_array(value, name=name)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {result.shape}")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6):
        raise ValueError(f"{name} must be a homogeneous rigid transform")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-4):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-4):
        raise ValueError(f"{name} rotation must be right-handed")
    return result


def _depth(value: Any, name: str) -> np.ndarray:
    result = _finite_array(value, name=name, ndim=2)
    if result.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.any(result < 0.0):
        raise ValueError(f"{name} cannot contain negative depths")
    return result


def _image_shape(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain (height, width)")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return height, width


def _unit_interval(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    return result


@dataclass(frozen=True)
class DirectionalDepthOverlap:
    """Visibility of depth samples from one camera in another camera."""

    source_valid_count: int
    in_frustum_count: int
    depth_supported_count: int
    visible_count: int
    occluded_count: int
    inconsistent_count: int
    frustum_ratio: float
    depth_supported_ratio: float
    visible_ratio: float
    target_coverage_ratio: float


@dataclass(frozen=True)
class BidirectionalDepthOverlap:
    """Conservative, bidirectional measured scene overlap."""

    a_to_b: DirectionalDepthOverlap
    b_to_a: DirectionalDepthOverlap
    symmetric_frustum_overlap: float
    symmetric_visible_overlap: float
    symmetric_target_coverage: float


@dataclass(frozen=True)
class FrustumOverlap:
    """Depth-range-based coarse overlap of two camera frusta."""

    a_to_b: float
    b_to_a: float
    symmetric_overlap: float
    samples_per_direction: int


@dataclass(frozen=True)
class FeatureOverlapResult:
    """Optional image-space overlap refinement returned by OpenCV/RANSAC."""

    available: bool
    success: bool
    detector: str
    model_type: str
    keypoints_a: int
    keypoints_b: int
    tentative_matches: int
    inliers: int
    inlier_ratio: float
    coverage_a: float
    coverage_b: float
    symmetric_coverage: float
    overlap_score: float
    model: Optional[np.ndarray] = None
    relative_rotation: Optional[np.ndarray] = None
    relative_translation_direction: Optional[np.ndarray] = None
    message: str = ""


@dataclass(frozen=True)
class OverlapPlanningConfig:
    """Policy for turning a pairwise overlap deficit into inserted views."""

    target_pair_overlap: float = 0.65
    hard_submap_overlap: float = 0.05
    max_inserts: int = 4
    split_if_budget_exceeded: bool = True

    def __post_init__(self) -> None:
        target = _unit_interval(self.target_pair_overlap, "target_pair_overlap")
        hard = _unit_interval(self.hard_submap_overlap, "hard_submap_overlap")
        if target <= 0.0 or target >= 1.0:
            raise ValueError("target_pair_overlap must be strictly between 0 and 1")
        if hard >= target:
            raise ValueError("hard_submap_overlap must be below target_pair_overlap")
        if int(self.max_inserts) != self.max_inserts or self.max_inserts < 0:
            raise ValueError("max_inserts must be a non-negative integer")


@dataclass(frozen=True)
class OverlapInsertionPlan:
    """An auditable pairwise insertion/submap decision."""

    measured_overlap: float
    target_pair_overlap: float
    overlap_deficit: float
    required_inserts: int
    insert_count: int
    alphas: tuple[float, ...]
    split_submap: bool
    budget_exceeded: bool
    reason: str


def _project_depth_direction(
    source_depth: np.ndarray,
    source_intrinsics: np.ndarray,
    source_c2w: np.ndarray,
    target_depth: np.ndarray,
    target_intrinsics: np.ndarray,
    target_c2w: np.ndarray,
    *,
    sample_stride: int,
    min_depth: float,
    max_depth: float,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
) -> DirectionalDepthOverlap:
    source_height, source_width = source_depth.shape
    target_height, target_width = target_depth.shape

    rows = np.arange(0, source_height, sample_stride, dtype=np.int64)
    cols = np.arange(0, source_width, sample_stride, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(cols, rows)
    sampled_depth = source_depth[grid_y, grid_x]
    source_valid = (sampled_depth >= min_depth) & (sampled_depth <= max_depth)
    source_valid_count = int(source_valid.sum())
    if source_valid_count == 0:
        return DirectionalDepthOverlap(
            source_valid_count=0,
            in_frustum_count=0,
            depth_supported_count=0,
            visible_count=0,
            occluded_count=0,
            inconsistent_count=0,
            frustum_ratio=0.0,
            depth_supported_ratio=0.0,
            visible_ratio=0.0,
            target_coverage_ratio=0.0,
        )

    x = grid_x[source_valid].astype(np.float64)
    y = grid_y[source_valid].astype(np.float64)
    z = sampled_depth[source_valid]
    pixels = np.stack([x, y, np.ones_like(x)], axis=0)
    source_points = (np.linalg.inv(source_intrinsics) @ pixels) * z[None, :]
    world_points = (
        source_c2w[:3, :3] @ source_points + source_c2w[:3, 3, None]
    )
    target_points = target_c2w[:3, :3].T @ (
        world_points - target_c2w[:3, 3, None]
    )
    target_z = target_points[2]
    projected = target_intrinsics @ target_points
    safe_z = np.where(np.abs(projected[2]) > _EPS, projected[2], 1.0)
    projected_x = projected[0] / safe_z
    projected_y = projected[1] / safe_z
    in_frustum = (
        (target_z > _EPS)
        & (projected_x >= -_PIXEL_EPS)
        & (projected_x <= target_width - 1 + _PIXEL_EPS)
        & (projected_y >= -_PIXEL_EPS)
        & (projected_y <= target_height - 1 + _PIXEL_EPS)
    )
    in_frustum_count = int(in_frustum.sum())
    if in_frustum_count == 0:
        return DirectionalDepthOverlap(
            source_valid_count=source_valid_count,
            in_frustum_count=0,
            depth_supported_count=0,
            visible_count=0,
            occluded_count=0,
            inconsistent_count=0,
            frustum_ratio=0.0,
            depth_supported_ratio=0.0,
            visible_ratio=0.0,
            target_coverage_ratio=0.0,
        )

    nearest_x = np.clip(
        np.rint(projected_x[in_frustum]), 0, target_width - 1
    ).astype(np.int64)
    nearest_y = np.clip(
        np.rint(projected_y[in_frustum]), 0, target_height - 1
    ).astype(np.int64)
    projected_z = target_z[in_frustum]
    measured_z = target_depth[nearest_y, nearest_x]
    supported = (measured_z >= min_depth) & (measured_z <= max_depth)
    depth_supported_count = int(supported.sum())

    difference = projected_z - measured_z
    tolerance = depth_abs_tolerance + depth_rel_tolerance * np.maximum(
        projected_z, measured_z
    )
    visible = supported & (np.abs(difference) <= tolerance)
    occluded = supported & (difference > tolerance)
    inconsistent = supported & (difference < -tolerance)
    visible_count = int(visible.sum())

    # Count destination coverage in stride-sized cells.  This keeps the metric
    # comparable when source projection itself is subsampled.
    target_valid_y, target_valid_x = np.nonzero(
        (target_depth >= min_depth) & (target_depth <= max_depth)
    )
    if target_valid_y.size:
        cell_columns = int(math.ceil(target_width / sample_stride))
        valid_cells = np.unique(
            (target_valid_y // sample_stride) * cell_columns
            + target_valid_x // sample_stride
        )
        visible_cells = np.unique(
            (nearest_y[visible] // sample_stride) * cell_columns
            + nearest_x[visible] // sample_stride
        )
        target_coverage = float(visible_cells.size / max(1, valid_cells.size))
    else:
        target_coverage = 0.0

    return DirectionalDepthOverlap(
        source_valid_count=source_valid_count,
        in_frustum_count=in_frustum_count,
        depth_supported_count=depth_supported_count,
        visible_count=visible_count,
        occluded_count=int(occluded.sum()),
        inconsistent_count=int(inconsistent.sum()),
        frustum_ratio=float(in_frustum_count / source_valid_count),
        depth_supported_ratio=float(depth_supported_count / source_valid_count),
        visible_ratio=float(visible_count / source_valid_count),
        target_coverage_ratio=float(np.clip(target_coverage, 0.0, 1.0)),
    )


def bidirectional_depth_overlap(
    depth_a: Any,
    intrinsics_a: Any,
    c2w_a: Any,
    depth_b: Any,
    intrinsics_b: Any,
    c2w_b: Any,
    *,
    sample_stride: int = 4,
    min_depth: float = 1.0e-4,
    max_depth: float = math.inf,
    depth_abs_tolerance: float = 0.03,
    depth_rel_tolerance: float = 0.03,
) -> BidirectionalDepthOverlap:
    """Measure conservative scene overlap by projecting depth both ways.

    Zeros are treated as missing depth.  Other non-finite or negative values
    are rejected.  ``symmetric_visible_overlap`` is the minimum directional
    visible ratio, intentionally preventing a small foreground view from being
    mistaken for high mutual coverage of a much wider view.
    """

    depth_a_array = _depth(depth_a, "depth_a")
    depth_b_array = _depth(depth_b, "depth_b")
    intrinsics_a_array = _intrinsics(intrinsics_a, "intrinsics_a")
    intrinsics_b_array = _intrinsics(intrinsics_b, "intrinsics_b")
    c2w_a_array = _c2w(c2w_a, "c2w_a")
    c2w_b_array = _c2w(c2w_b, "c2w_b")
    if int(sample_stride) != sample_stride or sample_stride <= 0:
        raise ValueError("sample_stride must be a positive integer")
    sample_stride = int(sample_stride)
    values = {
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "depth_abs_tolerance": float(depth_abs_tolerance),
        "depth_rel_tolerance": float(depth_rel_tolerance),
    }
    # Positive infinity is a useful default maximum; all other values must be
    # finite and tolerances cannot be negative.
    if not math.isfinite(values["min_depth"]) or values["min_depth"] <= 0.0:
        raise ValueError("min_depth must be finite and positive")
    if not (math.isfinite(values["max_depth"]) or values["max_depth"] == math.inf):
        raise ValueError("max_depth must be finite or +inf")
    if values["max_depth"] <= values["min_depth"]:
        raise ValueError("max_depth must be greater than min_depth")
    for key in ("depth_abs_tolerance", "depth_rel_tolerance"):
        if not math.isfinite(values[key]) or values[key] < 0.0:
            raise ValueError(f"{key} must be finite and non-negative")

    common = dict(
        sample_stride=sample_stride,
        min_depth=values["min_depth"],
        max_depth=values["max_depth"],
        depth_abs_tolerance=values["depth_abs_tolerance"],
        depth_rel_tolerance=values["depth_rel_tolerance"],
    )
    a_to_b = _project_depth_direction(
        depth_a_array,
        intrinsics_a_array,
        c2w_a_array,
        depth_b_array,
        intrinsics_b_array,
        c2w_b_array,
        **common,
    )
    b_to_a = _project_depth_direction(
        depth_b_array,
        intrinsics_b_array,
        c2w_b_array,
        depth_a_array,
        intrinsics_a_array,
        c2w_a_array,
        **common,
    )
    return BidirectionalDepthOverlap(
        a_to_b=a_to_b,
        b_to_a=b_to_a,
        symmetric_frustum_overlap=min(a_to_b.frustum_ratio, b_to_a.frustum_ratio),
        symmetric_visible_overlap=min(a_to_b.visible_ratio, b_to_a.visible_ratio),
        symmetric_target_coverage=min(
            a_to_b.target_coverage_ratio, b_to_a.target_coverage_ratio
        ),
    )


def _frustum_direction(
    source_intrinsics: np.ndarray,
    source_c2w: np.ndarray,
    source_shape: tuple[int, int],
    source_depth_range: tuple[float, float],
    target_intrinsics: np.ndarray,
    target_c2w: np.ndarray,
    target_shape: tuple[int, int],
    *,
    grid_rows: int,
    grid_cols: int,
    depth_samples: int,
) -> float:
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    xs = np.linspace(0.0, source_width - 1.0, grid_cols)
    ys = np.linspace(0.0, source_height - 1.0, grid_rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pixels = np.stack(
        [grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)], axis=0
    )
    near, far = source_depth_range
    depths = np.geomspace(near, far, depth_samples)
    rays = np.linalg.inv(source_intrinsics) @ pixels
    points = (rays[:, :, None] * depths[None, None, :]).reshape(3, -1)
    world = source_c2w[:3, :3] @ points + source_c2w[:3, 3, None]
    target = target_c2w[:3, :3].T @ (world - target_c2w[:3, 3, None])
    projection = target_intrinsics @ target
    safe_z = np.where(np.abs(projection[2]) > _EPS, projection[2], 1.0)
    x = projection[0] / safe_z
    y = projection[1] / safe_z
    inside = (
        (target[2] > _EPS)
        & (x >= -_PIXEL_EPS)
        & (x <= target_width - 1.0 + _PIXEL_EPS)
        & (y >= -_PIXEL_EPS)
        & (y <= target_height - 1.0 + _PIXEL_EPS)
    )
    return float(inside.mean())


def approximate_frustum_overlap(
    intrinsics_a: Any,
    c2w_a: Any,
    image_shape_a: Sequence[int],
    intrinsics_b: Any,
    c2w_b: Any,
    image_shape_b: Sequence[int],
    *,
    depth_range_a: tuple[float, float] = (0.1, 5.0),
    depth_range_b: tuple[float, float] = (0.1, 5.0),
    grid_shape: tuple[int, int] = (12, 16),
    depth_samples: int = 3,
) -> FrustumOverlap:
    """Approximate mutual frustum coverage using only cameras and depth range."""

    k_a = _intrinsics(intrinsics_a, "intrinsics_a")
    k_b = _intrinsics(intrinsics_b, "intrinsics_b")
    pose_a = _c2w(c2w_a, "c2w_a")
    pose_b = _c2w(c2w_b, "c2w_b")
    shape_a = _image_shape(image_shape_a, "image_shape_a")
    shape_b = _image_shape(image_shape_b, "image_shape_b")
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain (rows, columns)")
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows < 2 or cols < 2:
        raise ValueError("grid_shape dimensions must be at least 2")
    if int(depth_samples) != depth_samples or depth_samples <= 0:
        raise ValueError("depth_samples must be a positive integer")
    depth_samples = int(depth_samples)

    def validate_range(value: tuple[float, float], name: str) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain (near, far)")
        near, far = float(value[0]), float(value[1])
        if not math.isfinite(near) or not math.isfinite(far):
            raise ValueError(f"{name} must be finite")
        if near <= 0.0 or far <= near:
            raise ValueError(f"{name} requires 0 < near < far")
        return near, far

    range_a = validate_range(depth_range_a, "depth_range_a")
    range_b = validate_range(depth_range_b, "depth_range_b")
    a_to_b = _frustum_direction(
        k_a,
        pose_a,
        shape_a,
        range_a,
        k_b,
        pose_b,
        shape_b,
        grid_rows=rows,
        grid_cols=cols,
        depth_samples=depth_samples,
    )
    b_to_a = _frustum_direction(
        k_b,
        pose_b,
        shape_b,
        range_b,
        k_a,
        pose_a,
        shape_a,
        grid_rows=rows,
        grid_cols=cols,
        depth_samples=depth_samples,
    )
    return FrustumOverlap(
        a_to_b=a_to_b,
        b_to_a=b_to_a,
        symmetric_overlap=min(a_to_b, b_to_a),
        samples_per_direction=rows * cols * depth_samples,
    )


def _import_cv2() -> Any:
    import cv2  # type: ignore

    return cv2


def _gray_u8(image: Any, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"{name} must have shape HxW or HxWxC, got {array.shape}")
    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{name} cannot be empty")
    numeric = np.asarray(array, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains non-finite values")
    if numeric.ndim == 3:
        if numeric.shape[2] == 1:
            numeric = numeric[..., 0]
        else:
            # RGB luminance.  Alpha, if present, is deliberately ignored.
            numeric = (
                0.299 * numeric[..., 0]
                + 0.587 * numeric[..., 1]
                + 0.114 * numeric[..., 2]
            )
    if np.issubdtype(array.dtype, np.floating) and numeric.max(initial=0.0) <= 1.0:
        numeric = numeric * 255.0
    return np.rint(np.clip(numeric, 0.0, 255.0)).astype(np.uint8)


def _feature_failure(
    *,
    available: bool,
    detector: str,
    model_type: str,
    keypoints_a: int = 0,
    keypoints_b: int = 0,
    tentative_matches: int = 0,
    message: str,
) -> FeatureOverlapResult:
    return FeatureOverlapResult(
        available=available,
        success=False,
        detector=detector,
        model_type=model_type,
        keypoints_a=keypoints_a,
        keypoints_b=keypoints_b,
        tentative_matches=tentative_matches,
        inliers=0,
        inlier_ratio=0.0,
        coverage_a=0.0,
        coverage_b=0.0,
        symmetric_coverage=0.0,
        overlap_score=0.0,
        message=message,
    )


def match_image_overlap_ransac(
    image_a: Any,
    image_b: Any,
    *,
    detector: str = "orb",
    model_type: str = "fundamental",
    intrinsics_a: Optional[Any] = None,
    intrinsics_b: Optional[Any] = None,
    max_features: int = 2000,
    ratio_test: float = 0.75,
    ransac_threshold_px: float = 1.5,
    ransac_confidence: float = 0.999,
    require_opencv: bool = False,
) -> FeatureOverlapResult:
    """Refine overlap with feature matching and a robust geometric model.

    ``model_type`` may be ``"homography"``, ``"fundamental"``, or
    ``"essential"``.  Essential-matrix mode also returns a relative rotation
    and translation direction when pose recovery succeeds.  With the default
    ``require_opencv=False``, a missing OpenCV install yields an explicit
    ``available=False`` result rather than breaking the geometry-only planner.
    """

    gray_a = _gray_u8(image_a, "image_a")
    gray_b = _gray_u8(image_b, "image_b")
    detector = str(detector).lower()
    model_type = str(model_type).lower()
    if detector not in {"orb", "sift"}:
        raise ValueError("detector must be 'orb' or 'sift'")
    if model_type not in {"homography", "fundamental", "essential"}:
        raise ValueError("model_type must be homography, fundamental, or essential")
    if int(max_features) != max_features or max_features <= 0:
        raise ValueError("max_features must be a positive integer")
    ratio_test = float(ratio_test)
    ransac_threshold_px = float(ransac_threshold_px)
    ransac_confidence = float(ransac_confidence)
    if not math.isfinite(ratio_test) or not 0.0 < ratio_test < 1.0:
        raise ValueError("ratio_test must be finite and in (0, 1)")
    if not math.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0.0:
        raise ValueError("ransac_threshold_px must be finite and positive")
    if not math.isfinite(ransac_confidence) or not 0.0 < ransac_confidence < 1.0:
        raise ValueError("ransac_confidence must be finite and in (0, 1)")

    k_a = _intrinsics(intrinsics_a, "intrinsics_a") if intrinsics_a is not None else None
    k_b = _intrinsics(intrinsics_b, "intrinsics_b") if intrinsics_b is not None else None
    if model_type == "essential" and (k_a is None or k_b is None):
        raise ValueError("essential model requires both intrinsics matrices")

    try:
        cv2 = _import_cv2()
    except (ImportError, ModuleNotFoundError) as exc:
        if require_opencv:
            raise RuntimeError("OpenCV is required for feature refinement") from exc
        return _feature_failure(
            available=False,
            detector=detector,
            model_type=model_type,
            message="OpenCV is unavailable; feature refinement was skipped",
        )

    if detector == "orb":
        extractor = cv2.ORB_create(nfeatures=int(max_features))
        norm = cv2.NORM_HAMMING
    else:
        if not hasattr(cv2, "SIFT_create"):
            return _feature_failure(
                available=True,
                detector=detector,
                model_type=model_type,
                message="this OpenCV build does not provide SIFT",
            )
        extractor = cv2.SIFT_create(nfeatures=int(max_features))
        norm = cv2.NORM_L2
    keypoints_a, descriptors_a = extractor.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = extractor.detectAndCompute(gray_b, None)
    count_a, count_b = len(keypoints_a), len(keypoints_b)
    if descriptors_a is None or descriptors_b is None:
        return _feature_failure(
            available=True,
            detector=detector,
            model_type=model_type,
            keypoints_a=count_a,
            keypoints_b=count_b,
            message="not enough detected features",
        )

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    neighbours = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    matches = [
        pair[0]
        for pair in neighbours
        if len(pair) == 2 and pair[0].distance < ratio_test * pair[1].distance
    ]
    minimum = {"homography": 4, "fundamental": 8, "essential": 5}[model_type]
    if len(matches) < minimum:
        return _feature_failure(
            available=True,
            detector=detector,
            model_type=model_type,
            keypoints_a=count_a,
            keypoints_b=count_b,
            tentative_matches=len(matches),
            message=f"RANSAC needs at least {minimum} ratio-test matches",
        )

    points_a = np.float64([keypoints_a[item.queryIdx].pt for item in matches])
    points_b = np.float64([keypoints_b[item.trainIdx].pt for item in matches])
    rotation = None
    translation = None
    if model_type == "homography":
        model, mask = cv2.findHomography(
            points_a,
            points_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px,
            confidence=ransac_confidence,
        )
    elif model_type == "fundamental":
        model, mask = cv2.findFundamentalMat(
            points_a,
            points_b,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_threshold_px,
            confidence=ransac_confidence,
        )
    else:
        assert k_a is not None and k_b is not None
        homogeneous_a = np.column_stack([points_a, np.ones(len(points_a))])
        homogeneous_b = np.column_stack([points_b, np.ones(len(points_b))])
        normalized_a = (np.linalg.inv(k_a) @ homogeneous_a.T).T[:, :2]
        normalized_b = (np.linalg.inv(k_b) @ homogeneous_b.T).T[:, :2]
        focal_scale = max(
            float(k_a[0, 0]), float(k_a[1, 1]), float(k_b[0, 0]), float(k_b[1, 1])
        )
        model, mask = cv2.findEssentialMat(
            normalized_a,
            normalized_b,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=ransac_confidence,
            threshold=ransac_threshold_px / focal_scale,
        )
        if model is not None and mask is not None:
            # findEssentialMat can stack multiple 3x3 solutions.  recoverPose
            # consumes one; use the first deterministic hypothesis.
            essential = np.asarray(model, dtype=np.float64).reshape(-1, 3, 3)[0]
            _, recovered_rotation, recovered_translation, pose_mask = cv2.recoverPose(
                essential, normalized_a, normalized_b, mask=mask
            )
            model = essential
            if recovered_rotation is not None and recovered_translation is not None:
                rotation = np.asarray(recovered_rotation, dtype=np.float64)
                translation = np.asarray(recovered_translation, dtype=np.float64).reshape(3)
            if pose_mask is not None:
                mask = pose_mask

    if model is None or mask is None:
        return _feature_failure(
            available=True,
            detector=detector,
            model_type=model_type,
            keypoints_a=count_a,
            keypoints_b=count_b,
            tentative_matches=len(matches),
            message="RANSAC did not find a valid model",
        )
    mask_array = np.asarray(mask).reshape(-1).astype(bool)
    # Some OpenCV versions may return a shorter mask for a degenerate model.
    if mask_array.size != len(matches):
        return _feature_failure(
            available=True,
            detector=detector,
            model_type=model_type,
            keypoints_a=count_a,
            keypoints_b=count_b,
            tentative_matches=len(matches),
            message="OpenCV returned an invalid RANSAC mask",
        )
    inlier_points_a = points_a[mask_array]
    inlier_points_b = points_b[mask_array]
    inlier_count = int(mask_array.sum())
    if inlier_count < minimum:
        return _feature_failure(
            available=True,
            detector=detector,
            model_type=model_type,
            keypoints_a=count_a,
            keypoints_b=count_b,
            tentative_matches=len(matches),
            message="too few geometrically consistent matches",
        )

    def hull_coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
        if len(points) < 3:
            return 0.0
        hull = cv2.convexHull(np.float32(points))
        area = float(cv2.contourArea(hull))
        image_area = float(max(1, (shape[0] - 1) * (shape[1] - 1)))
        return float(np.clip(area / image_area, 0.0, 1.0))

    coverage_a = hull_coverage(inlier_points_a, gray_a.shape)
    coverage_b = hull_coverage(inlier_points_b, gray_b.shape)
    inlier_ratio = float(inlier_count / len(matches))
    symmetric_coverage = min(coverage_a, coverage_b)
    return FeatureOverlapResult(
        available=True,
        success=True,
        detector=detector,
        model_type=model_type,
        keypoints_a=count_a,
        keypoints_b=count_b,
        tentative_matches=len(matches),
        inliers=inlier_count,
        inlier_ratio=inlier_ratio,
        coverage_a=coverage_a,
        coverage_b=coverage_b,
        symmetric_coverage=symmetric_coverage,
        overlap_score=float(symmetric_coverage * inlier_ratio),
        model=np.asarray(model, dtype=np.float64),
        relative_rotation=rotation,
        relative_translation_direction=translation,
        message="ok",
    )


def plan_overlap_deficit(
    measured_overlap: float,
    config: OverlapPlanningConfig = OverlapPlanningConfig(),
) -> OverlapInsertionPlan:
    """Recommend interpolated views, or a submap split, from overlap deficit.

    The estimate assumes overlap decays approximately exponentially with view
    displacement.  Splitting a baseline into ``n + 1`` equal intervals then
    changes expected adjacent overlap from ``o`` to ``o ** (1 / (n + 1))``.
    This gives an auditable insertion count instead of an arbitrary linear bin.
    """

    overlap = _unit_interval(measured_overlap, "measured_overlap")
    target = config.target_pair_overlap
    deficit = max(0.0, target - overlap)
    if overlap >= target:
        return OverlapInsertionPlan(
            measured_overlap=overlap,
            target_pair_overlap=target,
            overlap_deficit=0.0,
            required_inserts=0,
            insert_count=0,
            alphas=(),
            split_submap=False,
            budget_exceeded=False,
            reason="sufficient_overlap",
        )
    if overlap <= config.hard_submap_overlap or overlap <= _EPS:
        return OverlapInsertionPlan(
            measured_overlap=overlap,
            target_pair_overlap=target,
            overlap_deficit=deficit,
            required_inserts=config.max_inserts + 1,
            insert_count=0,
            alphas=(),
            split_submap=True,
            budget_exceeded=True,
            reason="hard_overlap_discontinuity",
        )

    required = max(1, int(math.ceil(math.log(overlap) / math.log(target))) - 1)
    budget_exceeded = required > config.max_inserts
    split = bool(budget_exceeded and config.split_if_budget_exceeded)
    count = 0 if split else min(required, config.max_inserts)
    alphas = tuple((index + 1) / (count + 1) for index in range(count))
    return OverlapInsertionPlan(
        measured_overlap=overlap,
        target_pair_overlap=target,
        overlap_deficit=deficit,
        required_inserts=required,
        insert_count=count,
        alphas=alphas,
        split_submap=split,
        budget_exceeded=budget_exceeded,
        reason="insertion_budget_exceeded" if split else "overlap_deficit",
    )


__all__ = [
    "BidirectionalDepthOverlap",
    "DirectionalDepthOverlap",
    "FeatureOverlapResult",
    "FrustumOverlap",
    "OverlapInsertionPlan",
    "OverlapPlanningConfig",
    "approximate_frustum_overlap",
    "bidirectional_depth_overlap",
    "match_image_overlap_ransac",
    "plan_overlap_deficit",
]
