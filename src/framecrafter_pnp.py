"""RGB-D feature matching and PnP pose refinement for FrameCrafter.

The source image contributes metric (or SLAM-scale) 3-D points through its
depth map and estimated ``c2w`` pose.  Matched pixels in the destination image
then constrain its world-to-camera transform with ``cv2.solvePnPRansac``.

OpenCV is imported lazily.  Expected data failures (missing depth, featureless
images, image/depth scale mismatches, and RANSAC failure) are returned as
auditable result codes; malformed camera matrices still raise ``ValueError``.
Camera convention is OpenCV throughout: x right, y down, z forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


_EPS = 1.0e-12


@dataclass(frozen=True)
class PnPRefinementResult:
    """Result of RGB-D correspondence matching and robust PnP refinement."""

    available: bool
    success: bool
    failure_code: Optional[str]
    message: str
    detector: str
    keypoints_a: int
    keypoints_b: int
    tentative_matches: int
    depth_supported_matches: int
    inliers: int
    inlier_ratio: float
    reprojection_rmse_px: Optional[float]
    refined_c2w_b: Optional[np.ndarray]
    rotation_correction_deg: Optional[float]
    translation_correction: Optional[float]
    laplacian_variance_a: Optional[float] = None
    laplacian_variance_b: Optional[float] = None


@dataclass(frozen=True)
class PnPRefinementGateConfig:
    """Safety and quality limits applied before adopting a refined pose."""

    max_rotation_correction_deg: Optional[float] = 12.0
    max_translation_correction: Optional[float] = 0.25
    min_inliers: int = 8
    min_inlier_ratio: float = 0.35
    max_reprojection_rmse_px: float = 2.0

    def __post_init__(self) -> None:
        if (
            self.max_rotation_correction_deg is not None
            and float(self.max_rotation_correction_deg) < 0.0
        ):
            raise ValueError("max_rotation_correction_deg cannot be negative")
        if (
            self.max_translation_correction is not None
            and float(self.max_translation_correction) < 0.0
        ):
            raise ValueError("max_translation_correction cannot be negative")
        if int(self.min_inliers) != self.min_inliers or self.min_inliers < 0:
            raise ValueError("min_inliers must be a non-negative integer")
        if not 0.0 <= float(self.min_inlier_ratio) <= 1.0:
            raise ValueError("min_inlier_ratio must be in [0, 1]")
        if float(self.max_reprojection_rmse_px) < 0.0:
            raise ValueError("max_reprojection_rmse_px cannot be negative")


@dataclass(frozen=True)
class PnPRefinementGateResult:
    """Decision record for :func:`gate_pnp_refinement`."""

    accepted: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class _MatchedFeatures:
    points_a: np.ndarray
    points_b: np.ndarray
    keypoints_a: int
    keypoints_b: int
    tentative_matches: int


def _import_cv2():
    import cv2  # type: ignore

    return cv2


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _intrinsics(value: Any, *, name: str) -> np.ndarray:
    result = _finite_array(value, name=name)
    if result.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {result.shape}")
    if result[0, 0] <= 0.0 or result[1, 1] <= 0.0:
        raise ValueError(f"{name} focal lengths must be positive")
    if abs(float(np.linalg.det(result))) <= _EPS:
        raise ValueError(f"{name} must be invertible")
    return result


def _c2w(value: Any, *, name: str) -> np.ndarray:
    result = _finite_array(value, name=name)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {result.shape}")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6):
        raise ValueError(f"{name} must be a homogeneous transform")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-4):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-4):
        raise ValueError(f"{name} rotation must be right-handed")
    return result


def _image(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim not in (2, 3):
        raise ValueError(f"{name} must be HxW or HxWxC, got {result.shape}")
    if result.ndim == 3 and result.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} channel count must be 1, 3, or 4")
    if result.shape[0] <= 0 or result.shape[1] <= 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"{name} must contain numeric pixels")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite pixels")
    return result


def _depth(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.size == 0:
        raise ValueError(f"depth_a must be a non-empty HxW array, got {result.shape}")
    return result


def _failure(
    code: str,
    message: str,
    *,
    detector: str,
    available: bool = True,
    keypoints_a: int = 0,
    keypoints_b: int = 0,
    tentative_matches: int = 0,
    depth_supported_matches: int = 0,
    inliers: int = 0,
    inlier_ratio: float = 0.0,
    laplacian_variance_a: Optional[float] = None,
    laplacian_variance_b: Optional[float] = None,
) -> PnPRefinementResult:
    return PnPRefinementResult(
        available=available,
        success=False,
        failure_code=code,
        message=message,
        detector=detector,
        keypoints_a=keypoints_a,
        keypoints_b=keypoints_b,
        tentative_matches=tentative_matches,
        depth_supported_matches=depth_supported_matches,
        inliers=inliers,
        inlier_ratio=float(inlier_ratio),
        reprojection_rmse_px=None,
        refined_c2w_b=None,
        rotation_correction_deg=None,
        translation_correction=None,
        laplacian_variance_a=laplacian_variance_a,
        laplacian_variance_b=laplacian_variance_b,
    )


def _to_gray_u8(image: np.ndarray, cv2: Any) -> np.ndarray:
    working = image
    if working.dtype != np.uint8:
        working = working.astype(np.float64)
        minimum, maximum = float(working.min()), float(working.max())
        if minimum >= 0.0 and maximum <= 1.0 + 1.0e-6:
            working = working * 255.0
        working = np.clip(working, 0.0, 255.0).astype(np.uint8)
    if working.ndim == 2:
        return working
    if working.shape[2] == 1:
        return working[..., 0]
    conversion = cv2.COLOR_RGBA2GRAY if working.shape[2] == 4 else cv2.COLOR_RGB2GRAY
    return cv2.cvtColor(working, conversion)


def _make_detector(
    cv2: Any,
    detector: str,
    max_features: int,
    detector_params: Optional[Mapping[str, Any]],
):
    params = dict(detector_params or {})
    params.setdefault("nfeatures", int(max_features))
    if detector == "orb":
        return cv2.ORB_create(**params), cv2.NORM_HAMMING
    if detector == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("this OpenCV build does not provide SIFT")
        return cv2.SIFT_create(**params), cv2.NORM_L2
    raise ValueError("detector must be 'orb' or 'sift'")


def _match_features(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    *,
    cv2: Any,
    detector: str,
    max_features: int,
    detector_params: Optional[Mapping[str, Any]],
    ratio_test: float,
    mutual_check: bool,
) -> _MatchedFeatures:
    feature_detector, norm = _make_detector(
        cv2, detector, max_features, detector_params
    )
    keypoints_a, descriptors_a = feature_detector.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = feature_detector.detectAndCompute(gray_b, None)
    count_a, count_b = len(keypoints_a), len(keypoints_b)
    if descriptors_a is None or descriptors_b is None:
        return _MatchedFeatures(
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            count_a,
            count_b,
            0,
        )

    matcher = cv2.BFMatcher(norm, crossCheck=False)

    def ratio_matches(left: np.ndarray, right: np.ndarray) -> list[Any]:
        accepted: list[Any] = []
        for candidates in matcher.knnMatch(left, right, k=2):
            if len(candidates) == 2 and candidates[0].distance < ratio_test * candidates[1].distance:
                accepted.append(candidates[0])
        return accepted

    forward = ratio_matches(descriptors_a, descriptors_b)
    if mutual_check and forward:
        backward = ratio_matches(descriptors_b, descriptors_a)
        reverse_pairs = {(match.queryIdx, match.trainIdx) for match in backward}
        forward = [
            match
            for match in forward
            if (match.trainIdx, match.queryIdx) in reverse_pairs
        ]
    points_a = np.asarray(
        [keypoints_a[match.queryIdx].pt for match in forward], dtype=np.float64
    ).reshape(-1, 2)
    points_b = np.asarray(
        [keypoints_b[match.trainIdx].pt for match in forward], dtype=np.float64
    ).reshape(-1, 2)
    return _MatchedFeatures(points_a, points_b, count_a, count_b, len(forward))


def _rotation_delta_deg(c2w_a: np.ndarray, c2w_b: np.ndarray) -> float:
    relative = c2w_a[:3, :3].T @ c2w_b[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def refine_rgbd_pose_pnp(
    image_a: Any,
    depth_a: Any,
    K_a: Any,
    c2w_a: Any,
    image_b: Any,
    K_b: Any,
    *,
    c2w_b: Optional[Any] = None,
    detector: str = "orb",
    max_features: int = 3000,
    detector_params: Optional[Mapping[str, Any]] = None,
    ratio_test: float = 0.75,
    mutual_check: bool = True,
    min_keypoints: int = 12,
    min_matches: int = 8,
    min_depth: float = 1.0e-4,
    max_depth: float = float("inf"),
    min_laplacian_variance: float = 0.0,
    ransac_reprojection_error_px: float = 3.0,
    ransac_confidence: float = 0.999,
    ransac_iterations: int = 200,
    refine_iterative: bool = True,
    distortion_b: Optional[Any] = None,
    require_opencv: bool = False,
) -> PnPRefinementResult:
    """Refine camera B using RGB-D pixels in A and feature matches in B.

    ``depth_a`` must have exactly the same pixel grid as ``image_a``.  This
    deliberate contract avoids silently applying an unknown resize/crop scale
    to depth or intrinsics.  Images A and B may have different resolutions when
    their respective intrinsics describe those grids.
    """

    detector = str(detector).lower()
    if detector not in {"orb", "sift"}:
        raise ValueError("detector must be 'orb' or 'sift'")
    if int(max_features) != max_features or max_features < 1:
        raise ValueError("max_features must be a positive integer")
    if not 0.0 < float(ratio_test) < 1.0:
        raise ValueError("ratio_test must be in (0, 1)")
    if int(min_keypoints) != min_keypoints or min_keypoints < 4:
        raise ValueError("min_keypoints must be an integer of at least 4")
    if int(min_matches) != min_matches or min_matches < 4:
        raise ValueError("min_matches must be an integer of at least 4")
    if not 0.0 < float(min_depth) < float(max_depth):
        raise ValueError("depth bounds must satisfy 0 < min_depth < max_depth")
    if float(min_laplacian_variance) < 0.0:
        raise ValueError("min_laplacian_variance cannot be negative")
    if float(ransac_reprojection_error_px) <= 0.0:
        raise ValueError("ransac_reprojection_error_px must be positive")
    if not 0.0 < float(ransac_confidence) < 1.0:
        raise ValueError("ransac_confidence must be in (0, 1)")
    if int(ransac_iterations) != ransac_iterations or ransac_iterations < 1:
        raise ValueError("ransac_iterations must be a positive integer")

    array_a = _image(image_a, name="image_a")
    array_b = _image(image_b, name="image_b")
    depth = _depth(depth_a)
    intrinsics_a = _intrinsics(K_a, name="K_a")
    intrinsics_b = _intrinsics(K_b, name="K_b")
    pose_a = _c2w(c2w_a, name="c2w_a")
    pose_b = None if c2w_b is None else _c2w(c2w_b, name="c2w_b")

    if depth.shape != array_a.shape[:2]:
        return _failure(
            "image_depth_scale_mismatch",
            f"image_a grid {array_a.shape[:2]} does not match depth_a grid {depth.shape}",
            detector=detector,
        )
    valid_depth = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    if not np.any(valid_depth):
        return _failure(
            "missing_depth",
            "depth_a has no finite samples inside the requested depth range",
            detector=detector,
        )

    try:
        cv2 = _import_cv2()
    except (ImportError, ModuleNotFoundError) as error:
        if require_opencv:
            raise RuntimeError("OpenCV is required for RGB-D PnP refinement") from error
        return _failure(
            "opencv_unavailable",
            f"RGB-D PnP skipped because OpenCV is unavailable: {error}",
            detector=detector,
            available=False,
        )

    gray_a, gray_b = _to_gray_u8(array_a, cv2), _to_gray_u8(array_b, cv2)
    lap_a = float(cv2.Laplacian(gray_a, cv2.CV_64F).var())
    lap_b = float(cv2.Laplacian(gray_b, cv2.CV_64F).var())
    if lap_a < min_laplacian_variance:
        return _failure(
            "blurred_image_a",
            f"image_a Laplacian variance {lap_a:.6g} is below {min_laplacian_variance:.6g}",
            detector=detector,
            laplacian_variance_a=lap_a,
            laplacian_variance_b=lap_b,
        )
    if lap_b < min_laplacian_variance:
        return _failure(
            "blurred_image_b",
            f"image_b Laplacian variance {lap_b:.6g} is below {min_laplacian_variance:.6g}",
            detector=detector,
            laplacian_variance_a=lap_a,
            laplacian_variance_b=lap_b,
        )

    try:
        matched = _match_features(
            gray_a,
            gray_b,
            cv2=cv2,
            detector=detector,
            max_features=int(max_features),
            detector_params=detector_params,
            ratio_test=float(ratio_test),
            mutual_check=bool(mutual_check),
        )
    except RuntimeError as error:
        return _failure(
            "detector_unavailable",
            str(error),
            detector=detector,
            laplacian_variance_a=lap_a,
            laplacian_variance_b=lap_b,
        )
    if matched.keypoints_a < min_keypoints:
        return _failure(
            "insufficient_features_a",
            f"image_a produced {matched.keypoints_a} keypoints; need {min_keypoints}",
            detector=detector,
            keypoints_a=matched.keypoints_a,
            keypoints_b=matched.keypoints_b,
            tentative_matches=matched.tentative_matches,
            laplacian_variance_a=lap_a,
            laplacian_variance_b=lap_b,
        )
    if matched.keypoints_b < min_keypoints:
        return _failure(
            "insufficient_features_b",
            f"image_b produced {matched.keypoints_b} keypoints; need {min_keypoints}",
            detector=detector,
            keypoints_a=matched.keypoints_a,
            keypoints_b=matched.keypoints_b,
            tentative_matches=matched.tentative_matches,
            laplacian_variance_a=lap_a,
            laplacian_variance_b=lap_b,
        )
    if matched.tentative_matches < min_matches:
        return _failure(
            "insufficient_matches",
            f"feature matching produced {matched.tentative_matches} correspondences; need {min_matches}",
            detector=detector,
            keypoints_a=matched.keypoints_a,
            keypoints_b=matched.keypoints_b,
            tentative_matches=matched.tentative_matches,
            laplacian_variance_a=lap_a,
            laplacian_variance_b=lap_b,
        )

    rounded = np.rint(matched.points_a).astype(np.int64)
    height, width = depth.shape
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    sampled = np.full(len(rounded), np.nan, dtype=np.float64)
    sampled[inside] = depth[rounded[inside, 1], rounded[inside, 0]]
    supported = inside & np.isfinite(sampled) & (sampled >= min_depth) & (sampled <= max_depth)
    supported_count = int(supported.sum())
    if supported_count < min_matches:
        return _failure(
            "insufficient_valid_depth",
            f"only {supported_count}/{matched.tentative_matches} matches have valid depth; need {min_matches}",
            detector=detector,
            keypoints_a=matched.keypoints_a,
            keypoints_b=matched.keypoints_b,
            tentative_matches=matched.tentative_matches,
            depth_supported_matches=supported_count,
            laplacian_variance_a=lap_a,
            laplacian_variance_b=lap_b,
        )

    pixels_a = matched.points_a[supported]
    pixels_b = matched.points_b[supported]
    depths = sampled[supported]
    homogeneous = np.column_stack((pixels_a, np.ones(supported_count))).T
    camera_points_a = (np.linalg.inv(intrinsics_a) @ homogeneous) * depths[None, :]
    world_points = (
        pose_a[:3, :3] @ camera_points_a + pose_a[:3, 3, None]
    ).T.astype(np.float64)
    image_points_b = pixels_b.astype(np.float64)
    if distortion_b is None:
        distortion = np.zeros((4, 1), dtype=np.float64)
    else:
        distortion = _finite_array(distortion_b, name="distortion_b").reshape(-1, 1)

    solved, rvec, tvec, inlier_indices = cv2.solvePnPRansac(
        world_points,
        image_points_b,
        intrinsics_b,
        distortion,
        iterationsCount=int(ransac_iterations),
        reprojectionError=float(ransac_reprojection_error_px),
        confidence=float(ransac_confidence),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not solved or inlier_indices is None or len(inlier_indices) < 4:
        inliers = 0 if inlier_indices is None else int(len(inlier_indices))
        return _failure(
            "pnp_ransac_failed",
            f"solvePnPRansac failed with {inliers} inliers",
            detector=detector,
            keypoints_a=matched.keypoints_a,
            keypoints_b=matched.keypoints_b,
            tentative_matches=matched.tentative_matches,
            depth_supported_matches=supported_count,
            inliers=inliers,
            inlier_ratio=inliers / supported_count,
            laplacian_variance_a=lap_a,
            laplacian_variance_b=lap_b,
        )

    indices = np.asarray(inlier_indices, dtype=np.int64).reshape(-1)
    if refine_iterative and len(indices) >= 4:
        refined, candidate_rvec, candidate_tvec = cv2.solvePnP(
            world_points[indices],
            image_points_b[indices],
            intrinsics_b,
            distortion,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if refined:
            rvec, tvec = candidate_rvec, candidate_tvec

    projected, _ = cv2.projectPoints(
        world_points[indices], rvec, tvec, intrinsics_b, distortion
    )
    residuals = projected.reshape(-1, 2) - image_points_b[indices]
    rmse = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    world_to_camera_rotation, _ = cv2.Rodrigues(rvec)
    refined_pose = np.eye(4, dtype=np.float64)
    refined_pose[:3, :3] = world_to_camera_rotation.T
    refined_pose[:3, 3] = -world_to_camera_rotation.T @ np.asarray(tvec).reshape(3)
    rotation_correction = (
        None if pose_b is None else _rotation_delta_deg(pose_b, refined_pose)
    )
    translation_correction = (
        None
        if pose_b is None
        else float(np.linalg.norm(refined_pose[:3, 3] - pose_b[:3, 3]))
    )
    inlier_count = int(len(indices))
    return PnPRefinementResult(
        available=True,
        success=True,
        failure_code=None,
        message="RGB-D PnP refinement succeeded",
        detector=detector,
        keypoints_a=matched.keypoints_a,
        keypoints_b=matched.keypoints_b,
        tentative_matches=matched.tentative_matches,
        depth_supported_matches=supported_count,
        inliers=inlier_count,
        inlier_ratio=float(inlier_count / supported_count),
        reprojection_rmse_px=rmse,
        refined_c2w_b=refined_pose,
        rotation_correction_deg=rotation_correction,
        translation_correction=translation_correction,
        laplacian_variance_a=lap_a,
        laplacian_variance_b=lap_b,
    )


def gate_pnp_refinement(
    result: PnPRefinementResult,
    config: Optional[PnPRefinementGateConfig] = None,
) -> PnPRefinementGateResult:
    """Accept a PnP pose only when all configured correction gates pass."""

    policy = config or PnPRefinementGateConfig()
    failures: list[str] = []
    if not result.available:
        failures.append("unavailable")
    if not result.success:
        failures.append(result.failure_code or "refinement_failed")
    if result.inliers < policy.min_inliers:
        failures.append("insufficient_inliers")
    if result.inlier_ratio < policy.min_inlier_ratio:
        failures.append("low_inlier_ratio")
    if (
        result.reprojection_rmse_px is None
        or not np.isfinite(result.reprojection_rmse_px)
        or result.reprojection_rmse_px > policy.max_reprojection_rmse_px
    ):
        failures.append("high_reprojection_rmse")
    if policy.max_rotation_correction_deg is not None:
        if result.rotation_correction_deg is None:
            failures.append("rotation_correction_unavailable")
        elif result.rotation_correction_deg > policy.max_rotation_correction_deg:
            failures.append("rotation_correction_too_large")
    if policy.max_translation_correction is not None:
        if result.translation_correction is None:
            failures.append("translation_correction_unavailable")
        elif result.translation_correction > policy.max_translation_correction:
            failures.append("translation_correction_too_large")
    return PnPRefinementGateResult(accepted=not failures, failures=tuple(failures))


# Alternate verb order retained as a discoverable convenience for callers.
refine_pose_rgbd_pnp = refine_rgbd_pose_pnp

