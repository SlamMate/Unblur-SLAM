"""Coordinate-safe bridge from native ReSplat Gaussians to Unblur-SLAM.

ReSplat normalizes every inference bundle to the middle context camera.  Its
``means`` and ``covariances`` are consequently expressed in that normalized
(OpenCV) world frame.  The snapshot middle-camera ``c2w`` is the exact rigid
transform back to the mapper world frame.  The pinned official configuration,
however, sets ``no_rotate_sh=true``: degree>0 SH remains in each source-camera
block's local basis.  The production bridge therefore imports invariant DC
only and explicitly drops all higher-order coefficients.

The native ReSplat ``scales`` and ``rotations`` must *not* be transformed and
copied independently.  Upstream explicitly documents that those tensors are
not rotated to world space, while ``covariances`` are.  This module therefore
treats covariance as authoritative and deterministically factorizes the
world-space covariance into the log-scale / wxyz representation consumed by
Unblur-SLAM's GaussianModel.

This bridge makes a payload representationally importable.  It does not decide
map ownership, remove overlapping native Unblur Gaussians, resolve stale pose
revisions, or mutate an optimizer; those are separate active-map merge gates.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


WORLD_ARTIFACT_SCHEMA = "unblur_slam.official_resplat_unblur_world_gaussians.v1"
SUPPORTED_SH_DIMS = (1, 4, 9, 16)
PSD_RELATIVE_SPECTRAL_TOLERANCE = 2e-6
PSD_EIGENVALUE_FLOOR = 1e-12


def _rigid_c2w(value: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("pivot c2w must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("pivot c2w bottom row must be [0,0,0,1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError("pivot c2w rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-4):
        raise ValueError("pivot c2w rotation must have determinant +1")
    return matrix


def _real_sh_basis(directions: np.ndarray, dimension: int) -> np.ndarray:
    """Evaluate the real SH basis used by gsplat/Graphdeco (degrees 0--3)."""

    if dimension not in SUPPORTED_SH_DIMS:
        raise ValueError(
            f"unsupported SH dimension {dimension}; expected one of {SUPPORTED_SH_DIMS}"
        )
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("SH directions must have shape [sample,3]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("SH directions must be finite and nonzero")
    x, y, z = (values / norms).T
    result = [np.full_like(x, 0.28209479177387814)]
    if dimension >= 4:
        c1 = 0.4886025119029199
        result.extend((-c1 * y, c1 * z, -c1 * x))
    if dimension >= 9:
        c2 = (
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396,
        )
        result.extend(
            (
                c2[0] * x * y,
                c2[1] * y * z,
                c2[2] * (2.0 * z * z - x * x - y * y),
                c2[3] * x * z,
                c2[4] * (x * x - y * y),
            )
        )
    if dimension >= 16:
        c3 = (
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435,
        )
        result.extend(
            (
                c3[0] * y * (3.0 * x * x - y * y),
                c3[1] * x * y * z,
                c3[2] * y * (4.0 * z * z - x * x - y * y),
                c3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y),
                c3[4] * x * (4.0 * z * z - x * x - y * y),
                c3[5] * z * (x * x - y * y),
                c3[6] * x * (x * x - 3.0 * y * y),
            )
        )
    return np.stack(result, axis=-1)


def _fibonacci_directions(count: int) -> np.ndarray:
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / float(count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = index * (math.pi * (3.0 - math.sqrt(5.0)))
    return np.stack((radius * np.cos(theta), radius * np.sin(theta), z), axis=-1)


def sh_local_to_world_matrix(
    local_to_world_rotation: Sequence[Sequence[float]], dimension: int
) -> np.ndarray:
    """Return the coefficient transform preserving gsplat-rendered colors.

    For column directions ``d_world = R @ d_local``.  The returned matrix ``T``
    obeys ``Y(d_world) @ (T @ c_local) == Y(R.T @ d_world) @ c_local``.
    Computing the small degree<=3 representation from the renderer's explicit
    polynomial basis avoids silently mixing e3nn and Graphdeco basis ordering.
    """

    rotation = np.asarray(local_to_world_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("local-to-world rotation must be 3x3")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError("local-to-world rotation must be orthonormal")
    sample_count = max(64, 4 * dimension)
    world_directions = _fibonacci_directions(sample_count)
    # Row-vector equivalent of R.T @ d_world is d_world @ R.
    local_directions = world_directions @ rotation
    world_basis = _real_sh_basis(world_directions, dimension)
    local_basis = _real_sh_basis(local_directions, dimension)
    transform, _, rank, _ = np.linalg.lstsq(world_basis, local_basis, rcond=None)
    if int(rank) != dimension:
        raise RuntimeError("SH rotation basis solve was rank deficient")
    residual = np.max(np.abs(world_basis @ transform - local_basis))
    if not np.isfinite(residual) or residual > 5e-12:
        raise RuntimeError(f"SH rotation solve residual is too large: {residual}")
    return transform


def rotate_harmonics_local_to_world(
    harmonics: np.ndarray, local_to_world_rotation: Sequence[Sequence[float]]
) -> np.ndarray:
    coefficients = np.asarray(harmonics)
    if coefficients.ndim != 3 or coefficients.shape[1] != 3:
        raise ValueError("harmonics must have shape [gaussian,3,d_sh]")
    dimension = int(coefficients.shape[2])
    transform = sh_local_to_world_matrix(local_to_world_rotation, dimension)
    result = np.einsum(
        "ij,ncj->nci", transform, coefficients.astype(np.float64), optimize=True
    )
    return np.ascontiguousarray(result.astype(np.float32))


def _canonicalize_eigenvectors(rotations: np.ndarray) -> np.ndarray:
    """Resolve eigh column signs deterministically and enforce SO(3)."""

    result = np.asarray(rotations, dtype=np.float64).copy()
    for column in range(3):
        vectors = result[:, :, column]
        pivots = np.argmax(np.abs(vectors), axis=1)
        signs = np.take_along_axis(vectors, pivots[:, None], axis=1)[:, 0]
        result[:, :, column] *= np.where(signs < 0.0, -1.0, 1.0)[:, None]
    negative = np.linalg.det(result) < 0.0
    # Flipping one eigenvector preserves covariance.  Use the smallest-axis
    # column (eigh ordering) so the operation is deterministic.
    result[negative, :, 0] *= -1.0
    return result


def _rotation_matrices_to_wxyz(rotations: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to canonical unit wxyz quaternions."""

    # scipy's implementation is robust near pi and is already a dependency of
    # both the official ReSplat and Unblur environments.
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(rotations).as_quat()
    wxyz = np.concatenate((xyzw[:, 3:4], xyzw[:, :3]), axis=1)
    wxyz *= np.where(wxyz[:, :1] < 0.0, -1.0, 1.0)
    return np.ascontiguousarray(wxyz.astype(np.float32))


def covariance_to_unblur_scale_rotation(
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, float]]:
    """Repair float32 PSD roundoff, then factor into scale and wxyz rotation."""

    covariance = np.asarray(covariances, dtype=np.float64)
    if covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError("covariances must have shape [gaussian,3,3]")
    if not np.isfinite(covariance).all():
        raise ValueError("covariances contain non-finite values")
    symmetry_error = float(np.max(np.abs(covariance - covariance.transpose(0, 2, 1))))
    if symmetry_error > 1e-5:
        raise ValueError(f"covariances are not symmetric: max error {symmetry_error}")
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    raw_eigenvalues = eigenvalues.copy()
    largest = np.maximum(eigenvalues[:, -1], PSD_EIGENVALUE_FLOOR)
    significant_negative = eigenvalues[:, 0] < (
        -PSD_RELATIVE_SPECTRAL_TOLERANCE * largest
    )
    if np.any(significant_negative):
        worst = float(np.min(eigenvalues[:, 0]))
        worst_ratio = float(
            np.max(-eigenvalues[significant_negative, 0] / largest[significant_negative])
        )
        raise ValueError(
            "covariances contain negative spectrum beyond float32 tolerance: "
            f"min={worst}, max_relative_negative={worst_ratio}"
        )
    clamped_mask = eigenvalues < PSD_EIGENVALUE_FLOOR
    eigenvalues = np.maximum(eigenvalues, PSD_EIGENVALUE_FLOOR)
    rotations = _canonicalize_eigenvectors(eigenvectors)
    scales = np.sqrt(eigenvalues)
    repaired_covariance = np.einsum(
        "nij,nj,nkj->nik", rotations, scales * scales, rotations, optimize=True
    )
    correction = np.abs(repaired_covariance - covariance)
    max_correction = float(np.max(correction))
    # Verify factorization against the repaired covariance, not the raw
    # float32 matrix whose tiny negative eigenvalues prompted the repair.
    reconstructed = np.einsum(
        "nij,nj,nkj->nik", rotations, scales * scales, rotations, optimize=True
    )
    absolute_error = float(np.max(np.abs(reconstructed - repaired_covariance)))
    denominator = max(
        float(np.max(np.abs(repaired_covariance))), np.finfo(np.float64).tiny
    )
    relative_error = absolute_error / denominator
    if relative_error > 5e-6:
        raise RuntimeError(
            f"covariance factorization residual is too large: relative {relative_error}"
        )
    return (
        np.ascontiguousarray(repaired_covariance.astype(np.float64)),
        np.ascontiguousarray(scales.astype(np.float32)),
        _rotation_matrices_to_wxyz(rotations),
        {
            "max_input_symmetry_error": symmetry_error,
            "relative_spectral_tolerance": PSD_RELATIVE_SPECTRAL_TOLERANCE,
            "eigenvalue_floor": PSD_EIGENVALUE_FLOOR,
            "minimum_raw_eigenvalue": float(np.min(raw_eigenvalues)),
            "clamped_eigenvalue_count": int(np.count_nonzero(clamped_mask)),
            "clamped_gaussian_count": int(np.count_nonzero(np.any(clamped_mask, axis=1))),
            "significant_negative_gaussian_count": int(
                np.count_nonzero(significant_negative)
            ),
            "max_psd_correction": max_correction,
            "max_reconstruction_absolute_error": absolute_error,
            "max_reconstruction_relative_error": relative_error,
        },
    )


def build_unblur_world_arrays(
    *,
    means_local: np.ndarray,
    covariances_local: np.ndarray,
    harmonics_local: np.ndarray,
    opacities: np.ndarray,
    pivot_c2w: Sequence[Sequence[float]],
    owner_frame_ids: Sequence[int],
    owner_sequence_ordinals: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convert one fixed-topology ReSplat state into snapshot-world arrays."""

    transform = _rigid_c2w(pivot_c2w)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    means = np.asarray(means_local, dtype=np.float64)
    covariance = np.asarray(covariances_local, dtype=np.float64)
    harmonics = np.asarray(harmonics_local)
    opacity = np.asarray(opacities, dtype=np.float64).reshape(-1)
    count = int(means.shape[0]) if means.ndim == 2 else -1
    if means.shape != (count, 3):
        raise ValueError("means_local must have shape [gaussian,3]")
    if covariance.shape != (count, 3, 3):
        raise ValueError("covariances_local shape does not match means")
    if harmonics.ndim != 3 or harmonics.shape[:2] != (count, 3):
        raise ValueError("harmonics_local shape does not match means")
    if opacity.shape != (count,):
        raise ValueError("opacities shape does not match means")
    if not np.isfinite(means).all() or not np.isfinite(opacity).all():
        raise ValueError("means/opacities contain non-finite values")
    if not np.all((opacity > 0.0) & (opacity < 1.0)):
        raise ValueError("opacities must be strictly inside (0,1) for logit import")
    frame_ids = np.asarray(owner_frame_ids, dtype=np.int64)
    ordinals = np.asarray(owner_sequence_ordinals, dtype=np.int64)
    context_count = int(frame_ids.shape[0])
    if frame_ids.ndim != 1 or ordinals.shape != frame_ids.shape or context_count < 1:
        raise ValueError("owner frame ids/ordinals must be equal nonempty vectors")
    if count % context_count:
        raise ValueError("fixed topology cannot be divided across context owners")
    per_view = count // context_count

    means_world = means @ rotation.T + translation
    covariance_world = np.einsum(
        "ij,njk,lk->nil", rotation, covariance, rotation, optimize=True
    )
    # The pinned official config has no_rotate_sh=true.  Higher-order
    # coefficients are consequently in eight different source-camera bases,
    # not in the middle-local world basis.  DC is rotation invariant and is
    # the only safe representation for the default Unblur max_sh_degree=0.
    source_sh_dimension = int(harmonics.shape[2])
    harmonics_world = np.ascontiguousarray(harmonics[:, :, :1].astype(np.float32))
    (
        covariance_world_repaired,
        scales_world,
        rotations_world,
        factorization,
    ) = covariance_to_unblur_scale_rotation(covariance_world)
    owner_slots = np.repeat(np.arange(context_count, dtype=np.int16), per_view)
    owner_ids = np.repeat(frame_ids, per_view)
    owner_ordinals = np.repeat(ordinals, per_view)
    opacity_f32 = np.ascontiguousarray(opacity.astype(np.float32))
    eps = np.finfo(np.float32).eps
    clamped_opacity = np.clip(opacity, eps, 1.0 - eps)

    arrays = {
        # Retained only to make the rigid covariance conversion independently
        # auditable.  Active-map import consumes the *_world tensors below.
        "source_covariances_local": np.ascontiguousarray(covariance.astype(np.float32)),
        "means_world": np.ascontiguousarray(means_world.astype(np.float32)),
        "covariances_world": covariance_world_repaired,
        "harmonics_world": harmonics_world,
        "opacities": opacity_f32,
        "scales_world": scales_world,
        "rotations_world_wxyz": rotations_world,
        # Direct GaussianModel parameter layout (raw, before activations).
        "unblur_features_dc": np.ascontiguousarray(
            harmonics_world[:, :, :1].transpose(0, 2, 1)
        ),
        "unblur_features_rest": np.ascontiguousarray(
            harmonics_world[:, :, 1:].transpose(0, 2, 1)
        ),
        "unblur_log_scales": np.ascontiguousarray(np.log(scales_world).astype(np.float32)),
        "unblur_logit_opacities": np.ascontiguousarray(
            np.log(clamped_opacity / (1.0 - clamped_opacity)).astype(np.float32)[:, None]
        ),
        "owner_context_slots": owner_slots,
        "owner_frame_ids": owner_ids,
        "owner_sequence_ordinals": owner_ordinals,
    }
    metadata: dict[str, Any] = {
        "schema": WORLD_ARTIFACT_SCHEMA,
        "gaussian_count": count,
        "context_count": context_count,
        "gaussians_per_context_view": per_view,
        "coordinate_frame": "snapshot_unblur_world_opencv",
        "local_frame": "middle_context_camera_local_opencv",
        "local_to_world_c2w_opencv": transform.tolist(),
        "mean_formula": "p_world=R_world_from_local@p_local+t_world_from_local",
        "covariance_formula": (
            "C_world=PSD_floor(R_world_from_local@C_local@R_world_from_local.T)"
        ),
        "covariance_source": "official_refined_gaussians.covariances",
        "scale_rotation_source": "deterministic_eigendecomposition_of_covariances_world",
        "native_scale_rotation_copied": False,
        "official_no_rotate_sh": True,
        "source_harmonic_dimension": source_sh_dimension,
        "imported_harmonic_dimension": 1,
        "dropped_higher_order_harmonics": source_sh_dimension - 1,
        "harmonic_basis": "rotation_invariant_dc_only",
        "harmonic_conversion": "native_dc_copied_exactly;degree_gt_0_dropped",
        "factorization": dict(factorization),
        "raw_transformed_covariance_retained_implicitly": (
            "source_covariances_local plus local_to_world_c2w_opencv"
        ),
        "representationally_importable_by_unblur": True,
        "active_map_ownership_decided": False,
        "optimizer_state_included": False,
        "unconditional_append_safe": False,
        "recommended_active_import": {
            "zero_sh_rest": True,
            "higher_order_sh_imported": False,
            "max_sh_degree_zero_behavior": "import_dc_only_and_allocate_zero_rest",
            "reason": "preserve_geometry_first_and_avoid_cross_renderer_sh_basis_risk",
        },
    }
    return arrays, metadata


def array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }


__all__ = [
    "SUPPORTED_SH_DIMS",
    "WORLD_ARTIFACT_SCHEMA",
    "array_manifest",
    "build_unblur_world_arrays",
    "covariance_to_unblur_scale_rotation",
    "rotate_harmonics_local_to_world",
    "sh_local_to_world_matrix",
]
