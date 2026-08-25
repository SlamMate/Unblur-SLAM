"""Pose-aware FrameCrafter preprocessing utilities for Unblur-SLAM.

This module deliberately has no dependency on the official FrameCrafter code.
The 14B model is imported only when :class:`PythonAPIFrameCrafterBackend` is
constructed.  Geometry, planning, gating, manifest generation, and the clearly
labelled test-only backend all run on CPU with NumPy/Pillow.

Pose convention
---------------
``FrameRecord.c2w`` and ``TargetView.c2w`` use OpenCV camera axes (x right,
y down, z forward).  Official FrameCrafter input is obtained by inverting
those matrices to OpenCV world-to-camera (w2c) matrices.  Callers must not feed
ground-truth/evaluation-aligned poses into this preprocessing path.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image


_EPS = 1.0e-8
_SAFE_UNALIGNED_POSE_KEYS = {
    "traj_est_not_align",
    "traj_est_unaligned",
    "poses_not_align",
    "poses_unaligned",
    "trajectory_not_aligned",
}
_UNALIGNED_AUX_KEYS = {
    "traj_est_not_align": (
        "traj_est_not_align_timestamps",
        "traj_est_not_align_eval_mask",
    ),
    "traj_est_unaligned": (
        "traj_est_unaligned_timestamps",
        "traj_est_unaligned_eval_mask",
    ),
    "poses_not_align": (
        "poses_not_align_timestamps",
        "poses_not_align_eval_mask",
    ),
    "poses_unaligned": (
        "poses_unaligned_timestamps",
        "poses_unaligned_eval_mask",
    ),
    "trajectory_not_aligned": (
        "trajectory_not_aligned_timestamps",
        "trajectory_not_aligned_eval_mask",
    ),
}


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_c2w(value: Any, name: str = "c2w") -> np.ndarray:
    result = _array(value, (4, 4), name)
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-5):
        raise ValueError(f"{name} must be a homogeneous rigid transform")
    should_be_identity = result[:3, :3].T @ result[:3, :3]
    if not np.allclose(should_be_identity, np.eye(3), atol=2.0e-4):
        raise ValueError(f"{name} rotation is not orthonormal")
    if np.linalg.det(result[:3, :3]) < 0.999:
        raise ValueError(f"{name} rotation must be right-handed")
    return result


def _absolute(path: Optional[Path | str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


@dataclass
class FrameRecord:
    """A real input frame and its non-GT estimated camera state."""

    source_index: int
    frame_id: str
    timestamp: float
    rgb_path: Path
    c2w: np.ndarray
    intrinsics: np.ndarray
    sharpness: Optional[float] = None
    depth_path: Optional[Path] = None
    eval: bool = True
    kind: str = "original"

    def __post_init__(self) -> None:
        self.source_index = int(self.source_index)
        self.frame_id = str(self.frame_id)
        self.timestamp = float(self.timestamp)
        self.rgb_path = _absolute(self.rgb_path)  # type: ignore[assignment]
        self.depth_path = _absolute(self.depth_path)
        self.c2w = _validate_c2w(self.c2w, f"c2w[{self.frame_id}]")
        self.intrinsics = _array(
            self.intrinsics, (3, 3), f"intrinsics[{self.frame_id}]"
        )
        if self.kind != "original":
            raise ValueError("FrameCrafter context records must be real/original frames")
        if self.sharpness is not None:
            self.sharpness = float(self.sharpness)


@dataclass
class TargetView:
    """A requested synthetic view between two original frames."""

    target_id: str
    left_index: int
    right_index: int
    left_position: int
    right_position: int
    timestamp: float
    alpha: float
    c2w: np.ndarray
    intrinsics: np.ndarray
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.target_id = str(self.target_id)
        self.left_index = int(self.left_index)
        self.right_index = int(self.right_index)
        self.left_position = int(self.left_position)
        self.right_position = int(self.right_position)
        self.timestamp = float(self.timestamp)
        self.alpha = float(self.alpha)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"target alpha must be in (0, 1), got {self.alpha}")
        self.c2w = _validate_c2w(self.c2w, f"c2w[{self.target_id}]")
        self.intrinsics = _array(
            self.intrinsics, (3, 3), f"intrinsics[{self.target_id}]"
        )
        self.reasons = tuple(str(item) for item in self.reasons)


@dataclass
class WarpResult:
    rgb: np.ndarray
    depth: np.ndarray
    valid: np.ndarray


@dataclass
class DepthFusionResult:
    depth: np.ndarray
    valid: np.ndarray
    left_warp: WarpResult
    right_warp: WarpResult
    metrics: dict[str, float]


@dataclass
class GateConfig:
    min_sharpness_gain: float = 1.0
    min_depth_coverage: float = 0.05
    min_depth_consistency: float = 0.50
    max_photometric_error: float = 0.20
    max_reprojection_error_px: float = 2.0
    min_reprojection_valid_ratio: float = 0.05
    depth_abs_tolerance: float = 0.03
    depth_rel_tolerance: float = 0.03
    require_depth: bool = True


@dataclass
class GateResult:
    accepted: bool
    confidence: float
    metrics: dict[str, Optional[float]]
    failures: tuple[str, ...]
    fused_depth: Optional[np.ndarray] = None
    fused_depth_valid: Optional[np.ndarray] = None


@dataclass
class SyntheticFrameResult:
    target: TargetView
    rgb_path: Path
    depth_path: Optional[Path]
    confidence: float
    source_ids: tuple[str, ...]
    gate_metrics: Mapping[str, Optional[float]]
    batch_id: Optional[str] = None
    batch_target_ids: tuple[str, ...] = field(default_factory=tuple)
    batch_target_position: Optional[int] = None
    acceptance_class: str = "sharp_accepted"

    def __post_init__(self) -> None:
        self.rgb_path = _absolute(self.rgb_path)  # type: ignore[assignment]
        self.depth_path = _absolute(self.depth_path)
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.batch_id is not None:
            self.batch_id = str(self.batch_id)
        self.acceptance_class = str(self.acceptance_class)
        if self.acceptance_class not in {"sharp_accepted", "geometry_only"}:
            raise ValueError(
                "acceptance_class must be sharp_accepted or geometry_only"
            )
        self.batch_target_ids = tuple(str(value) for value in self.batch_target_ids)
        if self.batch_target_position is not None:
            self.batch_target_position = int(self.batch_target_position)
            if not 0 <= self.batch_target_position < len(self.batch_target_ids):
                raise ValueError("batch_target_position is outside batch_target_ids")
            if self.batch_target_ids[self.batch_target_position] != self.target.target_id:
                raise ValueError("batch target ordering does not match SyntheticFrameResult")


@dataclass(frozen=True)
class FrameCrafterGenerationBatch:
    """One official M-context to N-target generation request.

    All contexts are immutable real input frames.  Targets remain independent
    observations after generation; the batch only amortizes one diffusion call.
    """

    batch_id: str
    contexts: tuple[FrameRecord, ...]
    targets: tuple[TargetView, ...]
    max_endpoint_position_span: int

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("FrameCrafter batch_id cannot be empty")
        if not self.contexts or not self.targets:
            raise ValueError("FrameCrafter batch requires contexts and targets")
        if len(self.targets) > 4 or len(self.contexts) + len(self.targets) > 10:
            raise ValueError("FrameCrafter batch violates N<=4 or M+N<=10")
        if any(frame.kind != "original" for frame in self.contexts):
            raise ValueError("FrameCrafter batches may use only real/original contexts")
        context_positions = {frame.source_index for frame in self.contexts}
        endpoint_positions = {
            position
            for target in self.targets
            for position in (target.left_position, target.right_position)
        }
        endpoint_span = max(endpoint_positions) - min(endpoint_positions)
        if (
            int(self.max_endpoint_position_span) < 1
            or endpoint_span > int(self.max_endpoint_position_span)
        ):
            raise ValueError(
                f"batch {self.batch_id} endpoint span {endpoint_span} exceeds "
                f"local window {self.max_endpoint_position_span}"
            )
        for target in self.targets:
            if target.left_index not in context_positions or target.right_index not in context_positions:
                raise ValueError(
                    f"batch {self.batch_id} omits endpoints for {target.target_id}"
                )

    @property
    def endpoint_position_span(self) -> int:
        positions = [
            position
            for target in self.targets
            for position in (target.left_position, target.right_position)
        ]
        return max(positions) - min(positions)


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert an xyzw quaternion to a right-handed 3x3 rotation matrix."""

    q = _array(quaternion, (4,), "quaternion").copy()
    norm = float(np.linalg.norm(q))
    if norm <= _EPS:
        raise ValueError("zero quaternion is invalid")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized xyzw quaternion."""

    r = _array(rotation, (3, 3), "rotation")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
             (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        )
    else:
        diagonal = np.diag(r)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(max(_EPS, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
            q = np.array([0.25 * s, (r[0, 1] + r[1, 0]) / s,
                          (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s])
        elif index == 1:
            s = math.sqrt(max(_EPS, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
            q = np.array([(r[0, 1] + r[1, 0]) / s, 0.25 * s,
                          (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s])
        else:
            s = math.sqrt(max(_EPS, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
            q = np.array([(r[0, 2] + r[2, 0]) / s,
                          (r[1, 2] + r[2, 1]) / s, 0.25 * s,
                          (r[1, 0] - r[0, 1]) / s])
    q /= max(_EPS, float(np.linalg.norm(q)))
    return q


def slerp_rotation(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation between two rotation matrices."""

    alpha = float(alpha)
    q0 = matrix_to_quaternion_xyzw(left)
    q1 = matrix_to_quaternion_xyzw(right)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        q /= max(_EPS, float(np.linalg.norm(q)))
    else:
        theta = math.acos(dot)
        sine = math.sin(theta)
        q = (math.sin((1.0 - alpha) * theta) / sine) * q0
        q += (math.sin(alpha * theta) / sine) * q1
    return quaternion_xyzw_to_matrix(q)


def interpolate_c2w(left_c2w: np.ndarray, right_c2w: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate camera centers linearly and orientations with SO(3) SLERP."""

    left = _validate_c2w(left_c2w, "left_c2w")
    right = _validate_c2w(right_c2w, "right_c2w")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = slerp_rotation(left[:3, :3], right[:3, :3], alpha)
    result[:3, 3] = (1.0 - alpha) * left[:3, 3] + alpha * right[:3, 3]
    return result


def c2w_to_opencv_w2c(c2w: np.ndarray) -> np.ndarray:
    """Invert an OpenCV-axis c2w transform for official FrameCrafter input."""

    return np.linalg.inv(_validate_c2w(c2w)).astype(np.float32)


def rotation_delta_deg(left_c2w: np.ndarray, right_c2w: np.ndarray) -> float:
    relative = left_c2w[:3, :3].T @ right_c2w[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def read_rgb(path: Path | str) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return image


def save_rgb(path: Path | str, image: np.ndarray) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB").save(destination)
    return destination


def read_depth(path: Path | str, depth_scale: float = 1.0) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        depth = np.load(source)
    else:
        depth = np.asarray(Image.open(source))
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    return depth / float(depth_scale)


def save_depth_npy(path: Path | str, depth: np.ndarray) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, np.asarray(depth, dtype=np.float32))
    return destination


def save_depth_png(
    path: Path | str, depth_metres: np.ndarray, depth_scale: float = 5000.0
) -> Path:
    """Save metric depth as uint16 PNG compatible with BaseDataset.depthloader."""

    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(depth_metres, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    encoded[valid] = np.clip(
        np.rint(depth[valid] * float(depth_scale)), 1, np.iinfo(np.uint16).max
    ).astype(np.uint16)
    Image.fromarray(encoded).save(destination)
    return destination


def laplacian_sharpness(image: np.ndarray) -> float:
    """Mean absolute 4-neighbour Laplacian on a [0,1] RGB/gray image."""

    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    if array.ndim != 2:
        raise ValueError(f"image must be HW or HWC, got {array.shape}")
    padded = np.pad(array, 1, mode="edge")
    laplacian = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.mean(np.abs(laplacian)))


def _resolve_relative(root: Optional[Path], value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (root if root is not None else Path.cwd()) / path
    return path.resolve()


def _npz_declares_false(value: Any, label: str) -> None:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be a scalar false value")
    item = array.reshape(()).item()
    normalized = str(item).strip().lower()
    if normalized not in {"0", "false", "no"}:
        raise ValueError(f"{label} must be false, got {item!r}")


def load_bound_unaligned_trajectory(
    path: Path, key: str
) -> tuple[np.ndarray, str]:
    """Load the exact safe c2w array that a production CSV claims to export."""

    path = validate_pose_input_path(path)
    with np.load(path, allow_pickle=False) as payload:
        available = set(payload.files)
        if key not in _UNALIGNED_AUX_KEYS:
            raise ValueError(f"unsupported unaligned trajectory key {key!r}")
        timestamp_key, eval_mask_key = _UNALIGNED_AUX_KEYS[key]
        required = {
            key,
            timestamp_key,
            eval_mask_key,
            "pose_source",
            "uses_ground_truth_pose",
        }
        missing = required - available
        if missing:
            raise ValueError(
                f"trajectory NPZ is missing provenance fields: {sorted(missing)}"
            )
        _npz_declares_false(
            payload["uses_ground_truth_pose"], "trajectory uses_ground_truth_pose"
        )
        pose_source = validate_pose_source(
            str(np.asarray(payload["pose_source"]).reshape(()).item())
        )
        poses = np.asarray(payload[key], dtype=np.float64)
        source_indices = np.asarray(payload[timestamp_key], dtype=np.float64).reshape(-1)
        eval_mask = np.asarray(payload[eval_mask_key])
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError(
            f"trajectory {key!r} must be a non-empty Nx4x4 c2w array"
        )
    for index, pose in enumerate(poses):
        _validate_c2w(pose, f"trajectory[{key}][{index}]")
    expected_indices = np.arange(len(poses), dtype=np.float64)
    if len(source_indices) != len(poses) or not np.allclose(
        source_indices, expected_indices, rtol=0.0, atol=1.0e-5
    ):
        raise ValueError(
            f"{timestamp_key} must map the full trajectory to source indices 0..N-1"
        )
    if eval_mask.shape != (len(poses),) or not np.issubdtype(
        eval_mask.dtype, np.bool_
    ):
        raise ValueError(f"{eval_mask_key} must contain one boolean per pose")
    if not bool(np.all(eval_mask)):
        raise ValueError(
            "trajectory contains eval=false synthetic frames; only an "
            "original-stream first-pass trajectory is allowed"
        )
    return poses, pose_source


def load_frames_csv(
    csv_path: Path | str,
    *,
    image_root: Optional[Path | str] = None,
    depth_root: Optional[Path | str] = None,
    default_intrinsics: Optional[np.ndarray] = None,
    pose_convention: str = "c2w",
    compute_missing_sharpness: bool = True,
    expected_pose_source: Optional[str] = None,
    require_pose_provenance: bool = False,
) -> list[FrameRecord]:
    """Load planner-compatible CSV plus optional RGB-D/intrinsics columns.

    Required pose columns are ``frame,timestamp,tx,ty,tz,qx,qy,qz,qw``.
    Optional columns: ``index``, ``rgb_path``, ``depth_path``, ``sharpness`` or
    ``laplacian``, ``fx,fy,cx,cy``, and ``eval``.  Production callers set
    ``require_pose_provenance`` and require every row to declare the same safe
    ``pose_source`` plus ``uses_ground_truth_pose=false``.  ``pose_convention``
    is explicit to prevent accidentally treating a w2c translation as a camera
    center.
    """

    csv_path = validate_pose_input_path(csv_path)
    pose_convention = pose_convention.lower()
    if pose_convention not in {"c2w", "w2c"}:
        raise ValueError("pose_convention must be 'c2w' or 'w2c'")
    image_root_path = _absolute(image_root)
    depth_root_path = _absolute(depth_root)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"frame", "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"}
    if not rows:
        raise ValueError(f"no frame rows in {csv_path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing CSV columns: {sorted(missing)}")
    expected_source = (
        None
        if expected_pose_source is None
        else validate_pose_source(expected_pose_source)
    )
    if require_pose_provenance:
        provenance_columns = {
            "pose_source",
            "uses_ground_truth_pose",
            "trajectory_path",
            "trajectory_sha256",
            "trajectory_key",
        }
        missing_provenance = provenance_columns - set(rows[0])
        if missing_provenance:
            raise ValueError(
                "production pose CSV is missing provenance columns: "
                f"{sorted(missing_provenance)}"
            )

    result: list[FrameRecord] = []
    observed_pose_source: Optional[str] = None
    observed_trajectory: Optional[tuple[str, str, str]] = None
    trajectory_hash_cache: dict[Path, str] = {}
    trajectory_contract_cache: dict[tuple[Path, str], tuple[np.ndarray, str]] = {}
    for position, row in enumerate(rows):
        source_index = int(row.get("index", "") or position)
        bound_trajectory_poses: Optional[np.ndarray] = None
        if require_pose_provenance:
            row_pose_source = validate_pose_source(row.get("pose_source", ""))
            if observed_pose_source is None:
                observed_pose_source = row_pose_source
            elif row_pose_source != observed_pose_source:
                raise ValueError("pose_source must be identical on every CSV row")
            if expected_source is not None and row_pose_source != expected_source:
                raise ValueError(
                    "CSV pose_source disagrees with configured pose_source: "
                    f"{row_pose_source!r} != {expected_source!r}"
                )
            uses_gt = str(row.get("uses_ground_truth_pose", "")).strip().lower()
            if uses_gt not in {"0", "false", "no"}:
                raise ValueError(
                    "production pose CSV must explicitly declare "
                    "uses_ground_truth_pose=false on every row"
                )
            trajectory_path = validate_pose_input_path(
                _resolve_relative(
                    csv_path.parent, str(row.get("trajectory_path", "")).strip()
                )
            )
            declared_trajectory_hash = str(
                row.get("trajectory_sha256", "")
            ).strip()
            trajectory_key = str(row.get("trajectory_key", "")).strip()
            if trajectory_key not in _SAFE_UNALIGNED_POSE_KEYS:
                raise ValueError(
                    f"unsafe or unknown unaligned trajectory key {trajectory_key!r}"
                )
            if (
                len(declared_trajectory_hash) != 64
                or any(ch not in "0123456789abcdef" for ch in declared_trajectory_hash)
                or not trajectory_path.is_file()
            ):
                raise ValueError(
                    "CSV trajectory_path/trajectory_sha256 provenance mismatch"
                )
            actual_trajectory_hash = trajectory_hash_cache.get(trajectory_path)
            if actual_trajectory_hash is None:
                actual_trajectory_hash = _file_sha256(trajectory_path)
                trajectory_hash_cache[trajectory_path] = actual_trajectory_hash
            if actual_trajectory_hash != declared_trajectory_hash:
                raise ValueError(
                    "CSV trajectory_path/trajectory_sha256 provenance mismatch"
                )
            trajectory_identity = (
                str(trajectory_path),
                declared_trajectory_hash,
                trajectory_key,
            )
            if observed_trajectory is None:
                observed_trajectory = trajectory_identity
            elif trajectory_identity != observed_trajectory:
                raise ValueError(
                    "trajectory provenance must be identical on every CSV row"
                )
            contract_key = (trajectory_path, trajectory_key)
            trajectory_contract = trajectory_contract_cache.get(contract_key)
            if trajectory_contract is None:
                trajectory_contract = load_bound_unaligned_trajectory(
                    trajectory_path, trajectory_key
                )
                trajectory_contract_cache[contract_key] = trajectory_contract
            bound_trajectory_poses, npz_pose_source = trajectory_contract
            if npz_pose_source != row_pose_source:
                raise ValueError(
                    "CSV pose_source disagrees with trajectory NPZ pose_source"
                )
            if not 0 <= source_index < len(bound_trajectory_poses):
                raise ValueError(
                    f"CSV source_index={source_index} is outside the bound trajectory"
                )
        frame_id = row["frame"]
        rgb_value = row.get("rgb_path", "").strip() or frame_id
        rgb_path = _resolve_relative(image_root_path, rgb_value)
        depth_value = row.get("depth_path", "").strip()
        depth_path = _resolve_relative(depth_root_path, depth_value) if depth_value else None
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quaternion_xyzw_to_matrix(
            [float(row[key]) for key in ("qx", "qy", "qz", "qw")]
        )
        transform[:3, 3] = [float(row[key]) for key in ("tx", "ty", "tz")]
        c2w = transform if pose_convention == "c2w" else np.linalg.inv(transform)
        # DROID/lietorch exports SE(3) matrices as float32.  Serialising the
        # rotation through the CSV quaternion necessarily projects tiny
        # float32 orthogonality error back onto SO(3); real trajectories show
        # at most ~2.3e-7 elementwise change.  Keep the binding far tighter
        # than any meaningful pose edit while accepting that canonicalisation.
        if bound_trajectory_poses is not None and not np.allclose(
            c2w,
            bound_trajectory_poses[source_index],
            rtol=1.0e-8,
            atol=5.0e-7,
        ):
            raise ValueError(
                "CSV pose does not numerically match the declared unaligned "
                f"trajectory at source_index={source_index}"
            )

        if all(row.get(key, "").strip() for key in ("fx", "fy", "cx", "cy")):
            intrinsics = np.array(
                [[float(row["fx"]), 0.0, float(row["cx"])],
                 [0.0, float(row["fy"]), float(row["cy"])],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        elif default_intrinsics is not None:
            intrinsics = _array(default_intrinsics, (3, 3), "default_intrinsics").copy()
        else:
            raise ValueError("CSV lacks fx/fy/cx/cy and no default intrinsics were supplied")

        sharp_value = row.get("sharpness", "").strip() or row.get("laplacian", "").strip()
        sharpness = float(sharp_value) if sharp_value else None
        if sharpness is None and compute_missing_sharpness:
            sharpness = laplacian_sharpness(read_rgb(rgb_path))
        eval_value = row.get("eval", "true").strip().lower()
        is_eval = eval_value not in {"0", "false", "no"}
        result.append(
            FrameRecord(
                source_index=source_index,
                frame_id=frame_id,
                timestamp=float(row["timestamp"]),
                rgb_path=rgb_path,
                depth_path=depth_path,
                c2w=c2w,
                intrinsics=intrinsics,
                sharpness=sharpness,
                eval=is_eval,
            )
        )
    if len({frame.source_index for frame in result}) != len(result):
        raise ValueError("source_index values must be unique")
    return result


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "frame"


def _make_target(
    frames: Sequence[FrameRecord],
    left_position: int,
    right_position: int,
    alpha: float,
    reasons: Sequence[str],
    ordinal: int,
) -> TargetView:
    left = frames[left_position]
    right = frames[right_position]
    target_id = (
        f"syn_{_safe_id(left.frame_id)}_{_safe_id(right.frame_id)}_"
        f"{ordinal:02d}_{alpha:.6f}"
    )
    return TargetView(
        target_id=target_id,
        left_index=left.source_index,
        right_index=right.source_index,
        left_position=left_position,
        right_position=right_position,
        timestamp=(1.0 - alpha) * left.timestamp + alpha * right.timestamp,
        alpha=alpha,
        c2w=interpolate_c2w(left.c2w, right.c2w, alpha),
        intrinsics=(1.0 - alpha) * left.intrinsics + alpha * right.intrinsics,
        reasons=tuple(reasons),
    )


def plan_interpolated_targets(
    frames: Sequence[FrameRecord],
    *,
    laplacian_threshold: Optional[float] = None,
    blur_quantile: float = 0.30,
    translation_step: float = 0.08,
    rotation_step_deg: float = 6.0,
    blur_region_inserts: int = 1,
    max_inserts: int = 4,
) -> list[TargetView]:
    """Plan targets directly using camera-center distance and blur scores."""

    if len(frames) < 2:
        return []
    scores = np.asarray(
        [frame.sharpness if frame.sharpness is not None else laplacian_sharpness(read_rgb(frame.rgb_path))
         for frame in frames],
        dtype=np.float64,
    )
    threshold = (
        float(laplacian_threshold)
        if laplacian_threshold is not None
        else float(np.quantile(scores, float(np.clip(blur_quantile, 0.0, 1.0))))
    )
    targets: list[TargetView] = []
    for left_position, (left, right) in enumerate(zip(frames[:-1], frames[1:])):
        right_position = left_position + 1
        translation = float(np.linalg.norm(right.c2w[:3, 3] - left.c2w[:3, 3]))
        rotation = rotation_delta_deg(left.c2w, right.c2w)
        translation_ratio = translation / max(_EPS, float(translation_step))
        rotation_ratio = rotation / max(_EPS, float(rotation_step_deg))
        pose_inserts = max(0, int(math.ceil(max(translation_ratio, rotation_ratio))) - 1)
        both_blurry = bool(scores[left_position] < threshold and scores[right_position] < threshold)
        blur_inserts = max(0, int(blur_region_inserts)) if both_blurry else 0
        insert_count = min(max(0, int(max_inserts)), max(pose_inserts, blur_inserts))
        reasons: list[str] = []
        if blur_inserts:
            reasons.append("consecutive_blurry_region")
        if pose_inserts:
            reasons.append("large_pose_gap")
        for ordinal in range(insert_count):
            alpha = (ordinal + 1.0) / (insert_count + 1.0)
            targets.append(
                _make_target(
                    frames, left_position, right_position, alpha, reasons, ordinal
                )
            )
    return targets


def _uniform_sequence_sample(
    values: Sequence[tuple[int, TargetView]], count: int
) -> list[tuple[int, TargetView]]:
    """Return an endpoint-covering deterministic uniform subsequence."""

    requested = max(0, int(count))
    if requested == 0:
        return []
    if requested >= len(values):
        return list(values)
    if requested == 1:
        return [values[0]]
    positions = np.rint(
        np.linspace(0, len(values) - 1, num=requested, dtype=np.float64)
    ).astype(np.int64)
    # With requested<=len(values), linspace spacing is >=1 and rounded indices
    # are unique.  Keep the assertion explicit because exact cardinality is a
    # provenance/reporting requirement.
    if len(set(int(position) for position in positions)) != requested:
        raise RuntimeError("uniform FrameCrafter target selection lost cardinality")
    return [values[int(position)] for position in positions]


def select_scene_wide_targets(
    targets: Sequence[TargetView], max_targets: Optional[int]
) -> list[TargetView]:
    """Apply a deterministic, scene-wide cap without prefix bias.

    ``large_pose_gap`` candidates are rare/high-value and are retained first.
    If they alone exceed the cap they are uniformly sampled across the scene;
    otherwise all are retained and remaining capacity is uniformly filled from
    other candidates.  The returned target geometry is untouched and restored
    to chronological order.
    """

    chronological = sorted(
        enumerate(targets),
        key=lambda item: (
            item[1].timestamp,
            item[1].left_position,
            item[1].right_position,
            item[1].alpha,
            item[0],
        ),
    )
    if max_targets is None:
        return [target for _, target in chronological]
    limit = max(0, int(max_targets))
    if len(chronological) <= limit:
        return [target for _, target in chronological]
    if limit == 0:
        return []

    priority = [
        item for item in chronological if "large_pose_gap" in item[1].reasons
    ]
    ordinary = [
        item for item in chronological if "large_pose_gap" not in item[1].reasons
    ]
    if len(priority) >= limit:
        selected = _uniform_sequence_sample(priority, limit)
    else:
        selected = priority + _uniform_sequence_sample(
            ordinary, limit - len(priority)
        )
    selected.sort(
        key=lambda item: (
            item[1].timestamp,
            item[1].left_position,
            item[1].right_position,
            item[1].alpha,
            item[0],
        )
    )
    if len(selected) != limit:
        raise RuntimeError("scene-wide FrameCrafter cap produced the wrong target count")
    return [target for _, target in selected]


def _frame_aliases(frame: FrameRecord) -> set[str]:
    return {frame.frame_id, str(frame.rgb_path), frame.rgb_path.name}


def targets_from_planner_json(
    planner: Path | str | Mapping[str, Any], frames: Sequence[FrameRecord]
) -> list[TargetView]:
    """Convert the existing planner JSON's segment/alpha entries to poses."""

    if isinstance(planner, Mapping):
        payload = dict(planner)
    else:
        with open(planner, encoding="utf-8") as handle:
            payload = json.load(handle)
    alias_to_positions: dict[str, list[int]] = {}
    for position, frame in enumerate(frames):
        for alias in _frame_aliases(frame):
            alias_to_positions.setdefault(alias, []).append(position)

    def resolve(value: str) -> int:
        candidates = alias_to_positions.get(str(value), [])
        if len(candidates) != 1:
            raise ValueError(
                f"planner frame {value!r} resolves to {len(candidates)} source frames"
            )
        return candidates[0]

    targets: list[TargetView] = []
    for segment_index, segment in enumerate(payload.get("segments", [])):
        left_position = resolve(segment["left_frame"])
        right_position = resolve(segment["right_frame"])
        if right_position <= left_position:
            raise ValueError("planner segments must preserve source-frame order")
        reasons = tuple(segment.get("reasons", ()))
        for ordinal, alpha in enumerate(segment.get("alphas", ())):
            targets.append(
                _make_target(
                    frames,
                    left_position,
                    right_position,
                    float(alpha),
                    reasons,
                    segment_index * 100 + ordinal,
                )
            )
    return targets


def select_real_contexts(
    frames: Sequence[FrameRecord],
    target: TargetView,
    *,
    context_count: int = 6,
    min_contexts: int = 3,
) -> list[FrameRecord]:
    """Select real contexts using sharpness, pose proximity, and time proximity.

    The two real bracketing frames are retained when room permits.  Generated
    frames are not accepted by ``FrameRecord`` and therefore cannot recursively
    become context.
    """

    context_count = min(max(int(context_count), int(min_contexts)), len(frames))
    if len(frames) < min_contexts:
        raise ValueError(f"FrameCrafter needs at least {min_contexts} real contexts")
    sharpness = np.asarray(
        [frame.sharpness if frame.sharpness is not None else laplacian_sharpness(read_rgb(frame.rgb_path))
         for frame in frames],
        dtype=np.float64,
    )
    sharp_range = float(np.ptp(sharpness))
    sharp_norm = (sharpness - float(np.min(sharpness))) / max(_EPS, sharp_range)
    translations = np.asarray(
        [np.linalg.norm(frame.c2w[:3, 3] - target.c2w[:3, 3]) for frame in frames]
    )
    nonzero_translation = translations[translations > _EPS]
    translation_scale = (
        float(np.median(nonzero_translation)) if nonzero_translation.size else 1.0
    )
    rotations = np.asarray([rotation_delta_deg(frame.c2w, target.c2w) for frame in frames])
    time_distance = np.asarray([abs(frame.timestamp - target.timestamp) for frame in frames])
    nonzero_time = time_distance[time_distance > _EPS]
    time_scale = float(np.median(nonzero_time)) if nonzero_time.size else 1.0
    pose_proximity = np.exp(-translations / max(_EPS, translation_scale))
    pose_proximity *= np.exp(-rotations / 45.0)
    temporal_proximity = np.exp(-time_distance / max(_EPS, time_scale))
    scores = 0.55 * sharp_norm + 0.30 * pose_proximity + 0.15 * temporal_proximity

    ranked = list(np.argsort(-scores, kind="stable"))
    mandatory = [target.left_position, target.right_position]
    selected: list[int] = []
    for position in mandatory + ranked:
        if position not in selected:
            selected.append(position)
        if len(selected) == context_count:
            break
    selected.sort(key=lambda position: (frames[position].timestamp, frames[position].source_index))
    return [frames[position] for position in selected]


def select_shared_real_contexts(
    frames: Sequence[FrameRecord],
    targets: Sequence[TargetView],
    *,
    context_count: int = 6,
    min_contexts: int = 3,
) -> list[FrameRecord]:
    """Select one real context set that contains every batched endpoint.

    Endpoint frames are mandatory.  Remaining slots are filled with the real
    frames nearest in timestamp to any target, with source order as the stable
    tie breaker.  Generated frames can never enter this function because the
    :class:`FrameRecord` contract accepts only ``kind='original'``.
    """

    if not targets:
        raise ValueError("shared FrameCrafter contexts require at least one target")
    minimum = int(min_contexts)
    if minimum < 1:
        raise ValueError("min_contexts must be positive")
    if len(frames) < minimum:
        raise ValueError(f"FrameCrafter needs at least {minimum} real contexts")
    selected_count = min(max(int(context_count), minimum), len(frames))
    mandatory = sorted(
        {
            position
            for target in targets
            for position in (target.left_position, target.right_position)
        }
    )
    if any(position < 0 or position >= len(frames) for position in mandatory):
        raise ValueError("FrameCrafter target endpoint position is outside source frames")
    if len(mandatory) > selected_count:
        raise ValueError(
            f"{len(mandatory)} endpoint contexts exceed context_count={selected_count}"
        )

    target_timestamps = tuple(float(target.timestamp) for target in targets)
    remaining = [position for position in range(len(frames)) if position not in mandatory]
    remaining.sort(
        key=lambda position: (
            min(
                abs(float(frames[position].timestamp) - timestamp)
                for timestamp in target_timestamps
            ),
            position,
        )
    )
    selected = mandatory + remaining[: selected_count - len(mandatory)]
    selected.sort(
        key=lambda position: (frames[position].timestamp, frames[position].source_index)
    )
    return [frames[position] for position in selected]


def plan_framecrafter_generation_batches(
    frames: Sequence[FrameRecord],
    targets: Sequence[TargetView],
    *,
    context_count: int = 6,
    min_contexts: int = 3,
    max_targets_per_batch: int = 4,
    max_total_views: int = 10,
    max_endpoint_position_span: Optional[int] = None,
) -> list[FrameCrafterGenerationBatch]:
    """Greedily group adjacent chronological candidates into safe M-to-N calls.

    This changes neither the candidate set nor any target geometry.  Stable
    sorting supplies the required temporal execution order; each group is the
    longest adjacent prefix whose endpoint union fits M, with N<=4 and M+N<=10.
    """

    if not targets:
        return []
    minimum = int(min_contexts)
    if minimum < 1 or len(frames) < minimum:
        raise ValueError(f"FrameCrafter needs at least {minimum} real contexts")
    selected_count = min(max(int(context_count), minimum), len(frames))
    local_span_limit = (
        2 * selected_count
        if max_endpoint_position_span is None
        else int(max_endpoint_position_span)
    )
    if local_span_limit < 1:
        raise ValueError("max_endpoint_position_span must be positive")
    total_view_limit = min(10, int(max_total_views))
    target_limit = min(int(max_targets_per_batch), total_view_limit - selected_count)
    if target_limit < 1:
        raise ValueError(
            "FrameCrafter requires room for at least one target after contexts "
            f"(M={selected_count}, max_total_views={total_view_limit})"
        )
    target_limit = min(target_limit, 4)
    chronological = [
        target
        for _, target in sorted(
            enumerate(targets),
            key=lambda item: (
                item[1].timestamp,
                item[1].left_position,
                item[1].right_position,
                item[1].alpha,
                item[0],
            ),
        )
    ]

    groups: list[list[TargetView]] = []
    current: list[TargetView] = []
    current_endpoints: set[int] = set()
    for target in chronological:
        target_endpoints = {target.left_position, target.right_position}
        combined_endpoints = current_endpoints | target_endpoints
        combined_span = max(combined_endpoints) - min(combined_endpoints)
        can_append = bool(current) and (
            len(current) < target_limit
            and len(combined_endpoints) <= selected_count
            and combined_span <= local_span_limit
        )
        if current and not can_append:
            groups.append(current)
            current = []
            current_endpoints = set()
        current.append(target)
        current_endpoints.update(target_endpoints)
        current_span = max(current_endpoints) - min(current_endpoints)
        if len(current_endpoints) > selected_count or current_span > local_span_limit:
            raise ValueError(
                f"target {target.target_id} endpoints exceed context capacity/local window"
            )
    if current:
        groups.append(current)

    batches: list[FrameCrafterGenerationBatch] = []
    for batch_index, group in enumerate(groups):
        contexts = select_shared_real_contexts(
            frames,
            group,
            context_count=selected_count,
            min_contexts=minimum,
        )
        batches.append(
            FrameCrafterGenerationBatch(
                batch_id=f"batch_{batch_index:05d}",
                contexts=tuple(contexts),
                targets=tuple(group),
                max_endpoint_position_span=local_span_limit,
            )
        )
    return batches


def framecrafter_input_arrays(
    contexts: Sequence[FrameRecord], targets: Sequence[TargetView]
) -> tuple[np.ndarray, np.ndarray]:
    """Return official FrameCrafter arrays in context-first, target-last order."""

    if not contexts:
        raise ValueError("at least one context is required")
    w2c = np.stack(
        [c2w_to_opencv_w2c(frame.c2w) for frame in contexts]
        + [c2w_to_opencv_w2c(target.c2w) for target in targets]
    ).astype(np.float32)
    intrinsics = np.stack(
        [frame.intrinsics for frame in contexts]
        + [target.intrinsics for target in targets]
    ).astype(np.float32)
    return w2c, intrinsics


def save_framecrafter_npz(
    path: Path | str, contexts: Sequence[FrameRecord], targets: Sequence[TargetView]
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    w2c, intrinsics = framecrafter_input_arrays(contexts, targets)
    np.savez(destination, w2c_poses=w2c, intrinsics=intrinsics)
    return destination


class PythonAPIFrameCrafterBackend:
    """Dynamic adapter for the public ``model.FrameCrafter`` Python API."""

    # This adapter checks the public FrameCrafter Python contract.  Repository
    # authenticity is recorded by content digests, so do not label an arbitrary
    # compatible checkout as "official" based on a class name alone.
    backend_name = "python_api"
    test_only = False

    def __init__(
        self,
        *,
        repo_path: Path | str,
        checkpoint_path: Path | str,
        device: str = "cuda",
        vram_limit: float = 48.0,
        base_model_dir: Optional[Path | str] = None,
        num_inference_steps: int = 50,
        seed: int = 42,
        cfg_scale: float = 1.0,
    ) -> None:
        repo = Path(repo_path).expanduser().resolve()
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        model_file = repo / "model.py"
        if not model_file.is_file():
            raise FileNotFoundError(f"FrameCrafter-compatible model.py not found: {model_file}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FrameCrafter checkpoint not found: {checkpoint}")
        repo_string = str(repo)
        if repo_string not in sys.path:
            # Keep the path available: official model.py performs absolute
            # imports from its bundled camera_utils/diffsynth packages.
            sys.path.insert(0, repo_string)
        module_name = f"_unblur_framecrafter_api_{abs(hash(repo_string))}"
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import FrameCrafter API from {model_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        model_class = getattr(module, "FrameCrafter", None)
        if model_class is None:
            raise ImportError(f"{model_file} does not export FrameCrafter")
        self.model = model_class(
            checkpoint_path=str(checkpoint),
            device=device,
            vram_limit=float(vram_limit),
            base_model_dir=str(Path(base_model_dir).expanduser().resolve())
            if base_model_dir is not None
            else None,
        )
        self.num_inference_steps = int(num_inference_steps)
        self.seed = int(seed)
        self.cfg_scale = float(cfg_scale)
        self.generate_call_count = 0

    def generate_many(
        self,
        contexts: Sequence[FrameRecord],
        targets: Sequence[TargetView],
        *,
        height: int = 480,
        width: int = 832,
        resize_mode: Optional[str] = "stretch",
        restore_size_hws: Optional[Sequence[tuple[int, int]]] = None,
    ) -> list[np.ndarray]:
        """Generate N targets in one official call and preserve target ordering."""

        if not targets:
            raise ValueError("generate_many requires at least one target")
        if len(targets) > 4 or len(contexts) + len(targets) > 10:
            raise ValueError("FrameCrafter generate_many requires N<=4 and M+N<=10")
        if restore_size_hws is not None and len(restore_size_hws) != len(targets):
            raise ValueError("restore_size_hws must contain one shape per target")
        images: list[Image.Image] = []
        for frame in contexts:
            with Image.open(frame.rgb_path) as source:
                images.append(source.convert("RGB"))
        w2c, intrinsics = framecrafter_input_arrays(contexts, targets)
        self.generate_call_count += 1
        video = self.model.generate(
            images=images,
            w2c_poses=w2c,
            intrinsics=intrinsics,
            height=int(height),
            width=int(width),
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            cfg_scale=self.cfg_scale,
            resize_mode=resize_mode,
        )
        expected_count = len(contexts) + len(targets)
        if len(video) < expected_count:
            raise RuntimeError(
                f"FrameCrafter API returned {len(video)} frames for "
                f"{len(contexts)} contexts and {len(targets)} targets"
            )

        outputs: list[np.ndarray] = []
        for target_position in range(len(targets)):
            output = video[len(contexts) + target_position].convert("RGB")
            restore_size_hw = (
                None if restore_size_hws is None else restore_size_hws[target_position]
            )
            if restore_size_hw is not None:
                restore_h, restore_w = restore_size_hw
                if resize_mode == "crop" and output.size != (restore_w, restore_h):
                    raise ValueError(
                        "center-crop preprocessing cannot be inverted to full source FOV"
                    )
                if output.size != (restore_w, restore_h):
                    output = output.resize(
                        (restore_w, restore_h), Image.Resampling.BILINEAR
                    )
            outputs.append(np.asarray(output, dtype=np.float32) / 255.0)
        return outputs

    def generate(
        self,
        contexts: Sequence[FrameRecord],
        target: TargetView,
        *,
        height: int = 480,
        width: int = 832,
        resize_mode: Optional[str] = "stretch",
        restore_size_hw: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        return self.generate_many(
            contexts,
            [target],
            height=height,
            width=width,
            resize_mode=resize_mode,
            restore_size_hws=None if restore_size_hw is None else [restore_size_hw],
        )[0]


class TestOnlyBlendBackend:
    """Deterministic endpoint blend used only by CPU tests/smoke plumbing.

    It is intentionally impossible to instantiate without an explicit opt-in,
    and its name contains ``test_only`` so manifests/logs cannot confuse its
    output with a learned FrameCrafter result.
    """

    backend_name = "test_only_endpoint_blend"
    test_only = True

    def __init__(self, *, allow_test_only: bool = False) -> None:
        if not allow_test_only:
            raise RuntimeError(
                "TestOnlyBlendBackend is not FrameCrafter; pass allow_test_only=True "
                "only for tests/smoke plumbing"
            )
        self.generate_call_count = 0

    def generate_many(
        self,
        contexts: Sequence[FrameRecord],
        targets: Sequence[TargetView],
        **kwargs: Any,
    ) -> list[np.ndarray]:
        if not targets:
            raise ValueError("generate_many requires at least one target")
        if len(targets) > 4 or len(contexts) + len(targets) > 10:
            raise ValueError("FrameCrafter generate_many requires N<=4 and M+N<=10")
        self.generate_call_count += 1
        return [self._generate_one(contexts, target, **kwargs) for target in targets]

    def _generate_one(
        self,
        contexts: Sequence[FrameRecord],
        target: TargetView,
        **_: Any,
    ) -> np.ndarray:
        by_index = {frame.source_index: frame for frame in contexts}
        if target.left_index not in by_index or target.right_index not in by_index:
            raise ValueError("test-only blend requires both bracketing frames as contexts")
        left = read_rgb(by_index[target.left_index].rgb_path)
        right = read_rgb(by_index[target.right_index].rgb_path)
        if left.shape != right.shape:
            raise ValueError("test-only blend endpoints must share a resolution")
        return ((1.0 - target.alpha) * left + target.alpha * right).astype(np.float32)

    def generate(
        self,
        contexts: Sequence[FrameRecord],
        target: TargetView,
        **_: Any,
    ) -> np.ndarray:
        self.generate_call_count += 1
        return self._generate_one(contexts, target)


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image, dtype=np.float32)
    if result.ndim != 3 or result.shape[2] != 3:
        raise ValueError(f"RGB image must have shape HxWx3, got {result.shape}")
    if result.max(initial=0.0) > 1.0:
        result = result / 255.0
    return np.clip(result, 0.0, 1.0)


def project_rgbd_to_target(
    source_rgb: np.ndarray,
    source_depth: np.ndarray,
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
    source_intrinsics: np.ndarray,
    target_intrinsics: np.ndarray,
    *,
    target_shape: Optional[tuple[int, int]] = None,
) -> WarpResult:
    """Forward-project RGB-D with a nearest-surface z-buffer."""

    rgb = _ensure_rgb(source_rgb)
    depth = np.asarray(source_depth, dtype=np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"source depth/RGB shape mismatch: {depth.shape} vs {rgb.shape[:2]}")
    source_c2w = _validate_c2w(source_c2w, "source_c2w")
    target_w2c = np.linalg.inv(_validate_c2w(target_c2w, "target_c2w"))
    ks = _array(source_intrinsics, (3, 3), "source_intrinsics")
    kt = _array(target_intrinsics, (3, 3), "target_intrinsics")
    target_h, target_w = target_shape or depth.shape

    v, u = np.indices(depth.shape, dtype=np.float64)
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > _EPS)
    source_flat_indices = np.flatnonzero(valid.ravel())
    if source_flat_indices.size == 0:
        return WarpResult(
            rgb=np.zeros((target_h, target_w, 3), dtype=np.float32),
            depth=np.zeros((target_h, target_w), dtype=np.float32),
            valid=np.zeros((target_h, target_w), dtype=bool),
        )
    u_valid = u.ravel()[source_flat_indices]
    v_valid = v.ravel()[source_flat_indices]
    z_valid = z.ravel()[source_flat_indices]
    x = (u_valid - ks[0, 2]) * z_valid / ks[0, 0]
    y = (v_valid - ks[1, 2]) * z_valid / ks[1, 1]
    points_source = np.stack([x, y, z_valid, np.ones_like(z_valid)], axis=0)
    points_world = source_c2w @ points_source
    points_target = target_w2c @ points_world
    z_target = points_target[2]
    positive = z_target > _EPS
    u_target = np.rint(kt[0, 0] * points_target[0] / np.maximum(z_target, _EPS) + kt[0, 2]).astype(np.int64)
    v_target = np.rint(kt[1, 1] * points_target[1] / np.maximum(z_target, _EPS) + kt[1, 2]).astype(np.int64)
    inside = (
        positive
        & (u_target >= 0)
        & (u_target < target_w)
        & (v_target >= 0)
        & (v_target < target_h)
    )
    source_flat_indices = source_flat_indices[inside]
    z_target = z_target[inside]
    destination_flat = (v_target[inside] * target_w + u_target[inside]).astype(np.int64)

    output_rgb = np.zeros((target_h * target_w, 3), dtype=np.float32)
    output_depth = np.zeros(target_h * target_w, dtype=np.float32)
    output_valid = np.zeros(target_h * target_w, dtype=bool)
    if destination_flat.size:
        order = np.argsort(z_target, kind="stable")
        ordered_destinations = destination_flat[order]
        unique_destinations, first = np.unique(ordered_destinations, return_index=True)
        chosen = order[first]
        chosen_sources = source_flat_indices[chosen]
        output_rgb[unique_destinations] = rgb.reshape(-1, 3)[chosen_sources]
        output_depth[unique_destinations] = z_target[chosen].astype(np.float32)
        output_valid[unique_destinations] = True
    return WarpResult(
        rgb=output_rgb.reshape(target_h, target_w, 3),
        depth=output_depth.reshape(target_h, target_w),
        valid=output_valid.reshape(target_h, target_w),
    )


def fuse_bilateral_depth(
    left_rgb: np.ndarray,
    left_depth: np.ndarray,
    left_c2w: np.ndarray,
    left_intrinsics: np.ndarray,
    right_rgb: np.ndarray,
    right_depth: np.ndarray,
    right_c2w: np.ndarray,
    right_intrinsics: np.ndarray,
    target_c2w: np.ndarray,
    target_intrinsics: np.ndarray,
    *,
    target_shape: Optional[tuple[int, int]] = None,
    abs_tolerance: float = 0.03,
    rel_tolerance: float = 0.03,
    include_single_sided: bool = False,
) -> DepthFusionResult:
    """Project both endpoint depths and keep z-consistent target pixels."""

    left_warp = project_rgbd_to_target(
        left_rgb, left_depth, left_c2w, target_c2w, left_intrinsics,
        target_intrinsics, target_shape=target_shape
    )
    right_warp = project_rgbd_to_target(
        right_rgb, right_depth, right_c2w, target_c2w, right_intrinsics,
        target_intrinsics, target_shape=target_shape
    )
    both = left_warp.valid & right_warp.valid
    tolerance = np.maximum(
        float(abs_tolerance),
        float(rel_tolerance) * np.maximum(left_warp.depth, right_warp.depth),
    )
    agreement = both & (np.abs(left_warp.depth - right_warp.depth) <= tolerance)
    fused = np.zeros_like(left_warp.depth, dtype=np.float32)
    fused_valid = agreement.copy()
    fused[agreement] = 0.5 * (
        left_warp.depth[agreement] + right_warp.depth[agreement]
    )
    if include_single_sided:
        left_only = left_warp.valid & ~right_warp.valid
        right_only = right_warp.valid & ~left_warp.valid
        fused[left_only] = left_warp.depth[left_only]
        fused[right_only] = right_warp.depth[right_only]
        fused_valid |= left_only | right_only
    pixel_count = max(1, fused.size)
    overlap_count = int(both.sum())
    metrics = {
        "left_coverage": float(left_warp.valid.sum() / pixel_count),
        "right_coverage": float(right_warp.valid.sum() / pixel_count),
        "overlap_coverage": float(overlap_count / pixel_count),
        "depth_consistency": float(agreement.sum() / max(1, overlap_count)),
        "depth_coverage": float(fused_valid.sum() / pixel_count),
    }
    return DepthFusionResult(fused, fused_valid, left_warp, right_warp, metrics)


def _target_to_source_reprojection_stats(
    target_depth: np.ndarray,
    target_valid: np.ndarray,
    target_c2w: np.ndarray,
    target_intrinsics: np.ndarray,
    source_depth: np.ndarray,
    source_c2w: np.ndarray,
    source_intrinsics: np.ndarray,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> tuple[float, float, float]:
    """Return median cycle error, valid ratio, and source-depth consistency."""

    target_depth = np.asarray(target_depth, dtype=np.float64)
    target_valid = np.asarray(target_valid, dtype=bool)
    source_depth = np.asarray(source_depth, dtype=np.float64)
    kt = _array(target_intrinsics, (3, 3), "target_intrinsics")
    ks = _array(source_intrinsics, (3, 3), "source_intrinsics")
    target_c2w = _validate_c2w(target_c2w, "target_c2w")
    source_w2c = np.linalg.inv(_validate_c2w(source_c2w, "source_c2w"))
    v, u = np.indices(target_depth.shape, dtype=np.float64)
    target_indices = np.flatnonzero(target_valid.ravel() & (target_depth.ravel() > _EPS))
    if target_indices.size == 0:
        return float("inf"), 0.0, 0.0
    ut = u.ravel()[target_indices]
    vt = v.ravel()[target_indices]
    zt = target_depth.ravel()[target_indices]
    xt = (ut - kt[0, 2]) * zt / kt[0, 0]
    yt = (vt - kt[1, 2]) * zt / kt[1, 1]
    target_points = np.stack([xt, yt, zt, np.ones_like(zt)], axis=0)
    source_points = source_w2c @ (target_c2w @ target_points)
    expected_z = source_points[2]
    us = np.rint(ks[0, 0] * source_points[0] / np.maximum(expected_z, _EPS) + ks[0, 2]).astype(np.int64)
    vs = np.rint(ks[1, 1] * source_points[1] / np.maximum(expected_z, _EPS) + ks[1, 2]).astype(np.int64)
    inside = (
        (expected_z > _EPS)
        & (us >= 0)
        & (us < source_depth.shape[1])
        & (vs >= 0)
        & (vs < source_depth.shape[0])
    )
    if not inside.any():
        return float("inf"), 0.0, 0.0
    us, vs = us[inside], vs[inside]
    expected_z = expected_z[inside]
    ut, vt = ut[inside], vt[inside]
    observed_z = source_depth[vs, us]
    observed_valid = np.isfinite(observed_z) & (observed_z > _EPS)
    if not observed_valid.any():
        return float("inf"), 0.0, 0.0
    us = us[observed_valid].astype(np.float64)
    vs = vs[observed_valid].astype(np.float64)
    expected_z = expected_z[observed_valid]
    observed_z = observed_z[observed_valid]
    ut = ut[observed_valid]
    vt = vt[observed_valid]
    tolerance = np.maximum(
        float(abs_tolerance),
        float(rel_tolerance) * np.maximum(expected_z, observed_z),
    )
    depth_ok = np.abs(expected_z - observed_z) <= tolerance
    valid_ratio = float(observed_valid.sum() / max(1, target_indices.size))
    depth_consistency = float(depth_ok.sum() / max(1, depth_ok.size))
    if not depth_ok.any():
        return float("inf"), valid_ratio, depth_consistency

    xs = (us[depth_ok] - ks[0, 2]) * observed_z[depth_ok] / ks[0, 0]
    ys = (vs[depth_ok] - ks[1, 2]) * observed_z[depth_ok] / ks[1, 1]
    source_observed = np.stack(
        [xs, ys, observed_z[depth_ok], np.ones_like(xs)], axis=0
    )
    target_w2c = np.linalg.inv(target_c2w)
    cycled = target_w2c @ (_validate_c2w(source_c2w) @ source_observed)
    u_cycle = kt[0, 0] * cycled[0] / np.maximum(cycled[2], _EPS) + kt[0, 2]
    v_cycle = kt[1, 1] * cycled[1] / np.maximum(cycled[2], _EPS) + kt[1, 2]
    errors = np.sqrt((u_cycle - ut[depth_ok]) ** 2 + (v_cycle - vt[depth_ok]) ** 2)
    return float(np.median(errors)), valid_ratio, depth_consistency


def evaluate_candidate(
    generated_rgb: np.ndarray,
    left_frame: FrameRecord,
    right_frame: FrameRecord,
    target: TargetView,
    *,
    left_depth: Optional[np.ndarray],
    right_depth: Optional[np.ndarray],
    config: GateConfig = GateConfig(),
) -> GateResult:
    """Apply sharpness, RGB-D consistency, photometric, and reprojection gates."""

    generated = _ensure_rgb(generated_rgb)
    left_rgb = read_rgb(left_frame.rgb_path)
    right_rgb = read_rgb(right_frame.rgb_path)
    if generated.shape != left_rgb.shape or generated.shape != right_rgb.shape:
        raise ValueError(
            "gating requires generated and bracketing RGB at a common resolution: "
            f"generated={generated.shape}, left={left_rgb.shape}, right={right_rgb.shape}"
        )
    generated_sharpness = laplacian_sharpness(generated)
    reference_sharpness = max(laplacian_sharpness(left_rgb), laplacian_sharpness(right_rgb))
    if reference_sharpness <= _EPS:
        sharpness_gain = 1.0 if generated_sharpness <= _EPS else generated_sharpness / _EPS
    else:
        sharpness_gain = generated_sharpness / reference_sharpness
    metrics: dict[str, Optional[float]] = {
        "generated_sharpness": generated_sharpness,
        "reference_sharpness": reference_sharpness,
        "sharpness_gain": float(sharpness_gain),
    }
    failures: list[str] = []
    confidence_terms: list[float] = []
    if sharpness_gain < config.min_sharpness_gain:
        failures.append("sharpness_gain")
    confidence_terms.append(
        float(np.clip(sharpness_gain / max(_EPS, config.min_sharpness_gain), 0.0, 1.0))
    )

    if left_depth is None or right_depth is None:
        metrics.update(
            depth_coverage=None,
            depth_consistency=None,
            photometric_error=None,
            reprojection_error_px=None,
            reprojection_valid_ratio=None,
        )
        if config.require_depth:
            failures.append("missing_depth")
        return GateResult(
            accepted=not failures,
            confidence=float(np.mean(confidence_terms)),
            metrics=metrics,
            failures=tuple(failures),
        )

    fusion = fuse_bilateral_depth(
        left_rgb,
        left_depth,
        left_frame.c2w,
        left_frame.intrinsics,
        right_rgb,
        right_depth,
        right_frame.c2w,
        right_frame.intrinsics,
        target.c2w,
        target.intrinsics,
        target_shape=generated.shape[:2],
        abs_tolerance=config.depth_abs_tolerance,
        rel_tolerance=config.depth_rel_tolerance,
    )
    depth_coverage = fusion.metrics["depth_coverage"]
    depth_consistency = fusion.metrics["depth_consistency"]
    if depth_coverage < config.min_depth_coverage:
        failures.append("depth_coverage")
    if depth_consistency < config.min_depth_consistency:
        failures.append("depth_consistency")
    confidence_terms.extend(
        [
            float(np.clip(depth_coverage / max(_EPS, config.min_depth_coverage), 0.0, 1.0)),
            float(np.clip(depth_consistency / max(_EPS, config.min_depth_consistency), 0.0, 1.0)),
        ]
    )

    photo_mask = fusion.valid
    if photo_mask.any():
        left_error = np.abs(generated[photo_mask] - fusion.left_warp.rgb[photo_mask]).mean()
        right_error = np.abs(generated[photo_mask] - fusion.right_warp.rgb[photo_mask]).mean()
        photometric_error = float(0.5 * (left_error + right_error))
    else:
        photometric_error = float("inf")
    if not np.isfinite(photometric_error) or photometric_error > config.max_photometric_error:
        failures.append("photometric_error")
    confidence_terms.append(
        0.0
        if not np.isfinite(photometric_error)
        else float(np.clip(1.0 - photometric_error / max(_EPS, config.max_photometric_error), 0.0, 1.0))
    )

    left_reprojection = _target_to_source_reprojection_stats(
        fusion.depth,
        fusion.valid,
        target.c2w,
        target.intrinsics,
        left_depth,
        left_frame.c2w,
        left_frame.intrinsics,
        abs_tolerance=config.depth_abs_tolerance,
        rel_tolerance=config.depth_rel_tolerance,
    )
    right_reprojection = _target_to_source_reprojection_stats(
        fusion.depth,
        fusion.valid,
        target.c2w,
        target.intrinsics,
        right_depth,
        right_frame.c2w,
        right_frame.intrinsics,
        abs_tolerance=config.depth_abs_tolerance,
        rel_tolerance=config.depth_rel_tolerance,
    )
    finite_errors = [item[0] for item in (left_reprojection, right_reprojection) if np.isfinite(item[0])]
    reprojection_error = float(np.mean(finite_errors)) if finite_errors else float("inf")
    reprojection_valid_ratio = float(0.5 * (left_reprojection[1] + right_reprojection[1]))
    if (
        not np.isfinite(reprojection_error)
        or reprojection_error > config.max_reprojection_error_px
    ):
        failures.append("reprojection_error")
    if reprojection_valid_ratio < config.min_reprojection_valid_ratio:
        failures.append("reprojection_valid_ratio")
    confidence_terms.extend(
        [
            0.0
            if not np.isfinite(reprojection_error)
            else float(np.clip(1.0 - reprojection_error / max(_EPS, config.max_reprojection_error_px), 0.0, 1.0)),
            float(np.clip(reprojection_valid_ratio / max(_EPS, config.min_reprojection_valid_ratio), 0.0, 1.0)),
        ]
    )
    metrics.update(
        depth_coverage=float(depth_coverage),
        depth_consistency=float(depth_consistency),
        overlap_coverage=float(fusion.metrics["overlap_coverage"]),
        photometric_error=None if not np.isfinite(photometric_error) else photometric_error,
        reprojection_error_px=None if not np.isfinite(reprojection_error) else reprojection_error,
        reprojection_valid_ratio=reprojection_valid_ratio,
        reprojection_depth_consistency=float(
            0.5 * (left_reprojection[2] + right_reprojection[2])
        ),
    )
    return GateResult(
        accepted=not failures,
        confidence=float(np.clip(np.mean(confidence_terms), 0.0, 1.0)),
        metrics=metrics,
        failures=tuple(failures),
        fused_depth=fusion.depth,
        fused_depth_valid=fusion.valid,
    )


def validate_pose_source(pose_source: str) -> str:
    """Reject names that indicate GT or evaluation-aligned trajectory leakage."""

    value = str(pose_source).strip()
    lowered = value.lower()
    forbidden = ("groundtruth", "ground_truth", "traj_ref", "gt_pose", "aligned_to_gt")
    if not value or any(token in lowered for token in forbidden):
        raise ValueError(
            "pose_source must name a non-GT, non-evaluation-aligned estimate "
            "(for example droid_traj_est_not_align)"
        )
    return value


def validate_pose_input_path(path: Path | str) -> Path:
    """Refuse trajectory filenames that advertise GT/reference pose content."""

    resolved = Path(path).expanduser().resolve()
    lowered = resolved.name.lower()
    forbidden = (
        "groundtruth",
        "ground_truth",
        "traj_ref",
        "gt_pose",
        "aligned_to_gt",
    )
    if any(token in lowered for token in forbidden):
        raise ValueError(
            f"refusing possible ground-truth pose input {resolved}; export an explicitly "
            "non-aligned DROID trajectory CSV instead"
        )
    return resolved


def build_manifest(
    originals: Sequence[FrameRecord],
    synthetics: Sequence[SyntheticFrameResult],
    *,
    pose_source: str,
) -> dict[str, Any]:
    """Build the augmented-stream manifest consumed by the dataset adapter."""

    pose_source = validate_pose_source(pose_source)
    entries: list[tuple[float, int, dict[str, Any]]] = []
    for frame in originals:
        entry = {
            "kind": "original",
            "source_index": frame.source_index,
            "rgb_path": str(frame.rgb_path.resolve()),
            "depth_path": str(frame.depth_path.resolve()) if frame.depth_path else None,
            "c2w": frame.c2w.tolist(),
            "confidence": 1.0,
            "eval": bool(frame.eval),
            "fixed_pose": False,
            "reasons": [],
            "left_index": None,
            "right_index": None,
            "alpha": None,
            "timestamp": frame.timestamp,
        }
        entries.append((frame.timestamp, 0, entry))
    for result in synthetics:
        target = result.target
        entry = {
            "kind": "synthetic",
            "target_id": target.target_id,
            "source_index": None,
            "rgb_path": str(result.rgb_path.resolve()),
            "depth_path": str(result.depth_path.resolve()) if result.depth_path else None,
            "c2w": target.c2w.tolist(),
            "confidence": result.confidence,
            "eval": False,
            "fixed_pose": True,
            "reasons": list(target.reasons),
            "left_index": target.left_index,
            "right_index": target.right_index,
            "alpha": target.alpha,
            "timestamp": target.timestamp,
            "source_ids": list(result.source_ids),
            "gate_metrics": dict(result.gate_metrics),
            "batch_id": result.batch_id,
            "batch_target_ids": list(result.batch_target_ids),
            "batch_target_position": result.batch_target_position,
            "acceptance_class": result.acceptance_class,
        }
        entries.append((target.timestamp, 1, entry))
    entries.sort(key=lambda item: (item[0], item[1]))
    return {
        "schema": "unblur_slam.framecrafter_manifest.v1",
        "source_frame_count": len(originals),
        "generated_frame_count": len(synthetics),
        "pose_source": pose_source,
        "uses_ground_truth_pose": False,
        "frames": [entry for _, _, entry in entries],
    }


def _canonical_frame_contract(
    entries: Sequence[Mapping[str, Any]], *, kind: str
) -> list[dict[str, Any]]:
    """Return the exact frame fields cryptographically bound to the report."""

    records: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or entry.get("kind") != kind:
            continue
        common = {
            "kind": kind,
            "source_index": entry.get("source_index"),
            "timestamp": entry.get("timestamp"),
            "rgb_path": entry.get("rgb_path"),
            "depth_path": entry.get("depth_path"),
            "rgb_sha256": entry.get("rgb_sha256"),
            "depth_sha256": entry.get("depth_sha256"),
            "c2w": entry.get("c2w"),
            "confidence": entry.get("confidence"),
            "eval": entry.get("eval"),
            "fixed_pose": entry.get("fixed_pose"),
        }
        if kind == "synthetic":
            common.update(
                target_id=entry.get("target_id"),
                left_index=entry.get("left_index"),
                right_index=entry.get("right_index"),
                alpha=entry.get("alpha"),
                reasons=entry.get("reasons"),
                source_ids=entry.get("source_ids"),
                gate_metrics=entry.get("gate_metrics"),
                batch_id=entry.get("batch_id"),
                batch_target_ids=entry.get("batch_target_ids"),
                batch_target_position=entry.get("batch_target_position"),
            )
            if "acceptance_class" in entry:
                common["acceptance_class"] = entry.get("acceptance_class")
        common["manifest_position"] = position
        records.append(common)
    return records


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def synthetic_output_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    """Bind every accepted synthetic pose, source relation, gate and RGB-D file."""

    return _canonical_sha256(_canonical_frame_contract(entries, kind="synthetic"))


def source_input_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    """Bind every original pose/timestamp and input RGB-D artifact."""

    return _canonical_sha256(_canonical_frame_contract(entries, kind="original"))


def _manifest_file(path_value: Any, manifest_dir: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_batch_preprocess_report(
    report: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    *,
    report_dir: Path,
) -> None:
    """Cross-check M-to-N batching against every per-target report record."""

    batches = report.get("generation_batches")
    planned = report.get("planned")
    accepted = report.get("accepted")
    rejected = report.get("rejected")
    if not all(isinstance(value, list) for value in (batches, planned, accepted, rejected)):
        raise ValueError("batched FrameCrafter report lists are missing or invalid")
    if int(report.get("generation_batch_count", -1)) != len(batches):
        raise ValueError("FrameCrafter generation_batch_count is inconsistent")
    if int(report.get("backend_generate_call_count", -1)) != len(batches):
        raise ValueError("FrameCrafter report must record one official call per batch")
    planned_count = int(report.get("planned_target_count", -1))
    target_selection_policy = str(report.get("target_selection_policy", ""))
    if (
        int(report.get("selected_target_count", -1)) != planned_count
        or int(report.get("planned_total_before_cap", -1)) < planned_count
        or target_selection_policy
        not in {
            "scene_wide_large_pose_gap_priority_then_uniform_v1",
            "scene_wide_overlap_priority_then_uniform_v1",
        }
    ):
        raise ValueError("FrameCrafter scene-wide target selection provenance is invalid")

    batch_by_id: dict[str, Mapping[str, Any]] = {}
    flattened_target_ids: list[str] = []
    for batch_position, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise ValueError(f"generation_batches[{batch_position}] must be an object")
        batch_id = str(batch.get("batch_id", "")).strip()
        target_ids = batch.get("target_ids")
        context_indices = batch.get("context_source_indices")
        context_ids = batch.get("context_ids")
        if (
            not batch_id
            or batch_id in batch_by_id
            or not isinstance(target_ids, list)
            or not isinstance(context_indices, list)
            or not isinstance(context_ids, list)
        ):
            raise ValueError("FrameCrafter batch identity/context metadata is invalid")
        target_ids = [str(value) for value in target_ids]
        context_indices = [int(value) for value in context_indices]
        context_ids = [str(value) for value in context_ids]
        if (
            not 1 <= len(target_ids) <= 4
            or len(set(target_ids)) != len(target_ids)
            or len(context_indices) != len(context_ids)
            or len(set(context_indices)) != len(context_indices)
            or int(batch.get("target_count", -1)) != len(target_ids)
            or int(batch.get("context_count", -1)) != len(context_indices)
            or int(batch.get("total_view_count", -1))
            != len(context_indices) + len(target_ids)
            or len(context_indices) + len(target_ids) > 10
        ):
            raise ValueError("FrameCrafter M-to-N batch limits/counts are inconsistent")
        endpoint_min = int(batch.get("endpoint_position_min", -1))
        endpoint_max = int(batch.get("endpoint_position_max", -1))
        endpoint_span = int(batch.get("endpoint_position_span", -1))
        endpoint_span_limit = int(batch.get("max_endpoint_position_span", -1))
        batch_policy = str(batch.get("batch_policy", "legacy_local_multi_gap_v1"))
        if (
            endpoint_min < 0
            or endpoint_max < endpoint_min
            or endpoint_span != endpoint_max - endpoint_min
            or endpoint_span > endpoint_span_limit
        ):
            raise ValueError("FrameCrafter batch is not source-position local")
        if batch_policy == "legacy_local_multi_gap_v1":
            if endpoint_span_limit != 2 * len(context_indices):
                raise ValueError("legacy FrameCrafter batch span limit is invalid")
        elif batch_policy == "same_gap_multi_alpha_v1":
            if endpoint_span_limit != max(1, endpoint_span):
                raise ValueError("same-gap FrameCrafter batch span contract is invalid")
            conditioning = batch.get("conditioning")
            if not isinstance(conditioning, list) or len(conditioning) != len(
                context_indices
            ):
                raise ValueError("role-aware FrameCrafter conditioning is missing")
            conditioning_indices = []
            for item in conditioning:
                if not isinstance(item, Mapping):
                    raise ValueError("conditioning provenance must be an object")
                source_index = int(item.get("source_index", -1))
                conditioning_indices.append(source_index)
                if str(item.get("resolved_mode")) not in {"raw", "evssm"}:
                    raise ValueError("conditioning resolved_mode is invalid")
                artifact = _manifest_file(
                    item.get("resolved_path"), report_dir, "conditioning.resolved_path"
                )
                expected_hash = str(item.get("resolved_sha256", ""))
                if len(expected_hash) != 64 or _file_sha256(artifact) != expected_hash:
                    raise ValueError("conditioning image hash mismatch")
            if conditioning_indices != context_indices:
                raise ValueError("conditioning order disagrees with batch contexts")
        else:
            raise ValueError(f"unknown FrameCrafter batch policy {batch_policy!r}")
        pose_path = _manifest_file(
            batch.get("poses_npz"), report_dir, f"{batch_id}.poses_npz"
        )
        expected_pose_hash = str(batch.get("poses_npz_sha256", ""))
        if len(expected_pose_hash) != 64 or _file_sha256(pose_path) != expected_pose_hash:
            raise ValueError(f"FrameCrafter batch NPZ hash mismatch for {batch_id}")
        try:
            with np.load(pose_path, allow_pickle=False) as arrays:
                expected_views = len(context_indices) + len(target_ids)
                if (
                    set(arrays.files) != {"w2c_poses", "intrinsics"}
                    or arrays["w2c_poses"].shape != (expected_views, 4, 4)
                    or arrays["intrinsics"].shape != (expected_views, 3, 3)
                    or not np.isfinite(arrays["w2c_poses"]).all()
                    or not np.isfinite(arrays["intrinsics"]).all()
                ):
                    raise ValueError("unexpected batch NPZ arrays")
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid FrameCrafter batch NPZ for {batch_id}") from error
        batch_by_id[batch_id] = batch
        flattened_target_ids.extend(target_ids)

    if len(flattened_target_ids) != planned_count or len(set(flattened_target_ids)) != planned_count:
        raise ValueError("FrameCrafter batch targets do not match planned_target_count")
    if len(planned) != planned_count:
        raise ValueError("FrameCrafter planned target records have wrong cardinality")
    planned_endpoint_positions: dict[str, list[int]] = {
        batch_id: [] for batch_id in batch_by_id
    }
    for target_position, (record, target_id) in enumerate(zip(planned, flattened_target_ids)):
        if not isinstance(record, Mapping) or str(record.get("target_id")) != target_id:
            raise ValueError("FrameCrafter planned target order disagrees with batches")
        batch_id = str(record.get("batch_id", ""))
        batch = batch_by_id.get(batch_id)
        if batch is None:
            raise ValueError("FrameCrafter planned target names an unknown batch")
        batch_target_ids = [str(value) for value in batch["target_ids"]]
        batch_target_position = int(record.get("batch_target_position", -1))
        context_indices = [int(value) for value in batch["context_source_indices"]]
        left_position = int(record.get("left_position", -1))
        right_position = int(record.get("right_position", -1))
        if (
            not 0 <= batch_target_position < len(batch_target_ids)
            or batch_target_ids[batch_target_position] != target_id
            or record.get("batch_target_ids") != batch["target_ids"]
            or record.get("context_source_indices") != batch["context_source_indices"]
            or record.get("context_ids") != batch["context_ids"]
            or record.get("poses_npz") != batch.get("poses_npz")
            or record.get("poses_npz_sha256") != batch.get("poses_npz_sha256")
            or int(record.get("left_index", -1)) not in context_indices
            or int(record.get("right_index", -1)) not in context_indices
            or not 0 <= left_position < right_position
        ):
            raise ValueError("FrameCrafter planned target batch mapping is invalid")
        planned_endpoint_positions[batch_id].extend((left_position, right_position))

    for batch_id, positions in planned_endpoint_positions.items():
        batch = batch_by_id[batch_id]
        if (
            min(positions) != int(batch["endpoint_position_min"])
            or max(positions) != int(batch["endpoint_position_max"])
        ):
            raise ValueError("FrameCrafter batch endpoint span disagrees with targets")
        if str(batch.get("batch_policy", "legacy_local_multi_gap_v1")) == (
            "same_gap_multi_alpha_v1"
        ):
            pairs = {
                (
                    int(record["left_position"]),
                    int(record["right_position"]),
                )
                for record in planned
                if isinstance(record, Mapping)
                and str(record.get("batch_id")) == batch_id
            }
            if len(pairs) != 1:
                raise ValueError("same-gap FrameCrafter batch contains multiple gaps")

    accepted_count = int(report.get("accepted_target_count", -1))
    rejected_count = int(report.get("rejected_target_count", -1))
    if len(accepted) != accepted_count or len(rejected) != rejected_count:
        raise ValueError("FrameCrafter per-target gate report counts are inconsistent")
    outcome_records = [*accepted, *rejected]
    outcome_ids = [
        str(record.get("target_id", "")) if isinstance(record, Mapping) else ""
        for record in outcome_records
    ]
    if len(outcome_ids) != planned_count or set(outcome_ids) != set(flattened_target_ids):
        raise ValueError("FrameCrafter gate outcomes do not partition planned targets")

    synthetic_entries = {
        str(entry.get("target_id")): entry
        for entry in entries
        if entry.get("kind") == "synthetic"
    }
    accepted_by_id = {
        str(record.get("target_id")): record
        for record in accepted
        if isinstance(record, Mapping)
    }
    if set(synthetic_entries) != set(accepted_by_id):
        raise ValueError("FrameCrafter accepted report targets disagree with manifest")
    for target_id, entry in synthetic_entries.items():
        record = accepted_by_id[target_id]
        if any(
            (
                entry.get("batch_id") != record.get("batch_id"),
                entry.get("batch_target_ids") != record.get("batch_target_ids"),
                entry.get("batch_target_position")
                != record.get("batch_target_position"),
                entry.get("source_ids") != record.get("context_ids"),
                entry.get("gate_metrics") != record.get("metrics"),
                entry.get("rgb_path") != record.get("rgb_path"),
                entry.get("depth_path") != record.get("depth_path"),
                entry.get("acceptance_class", "sharp_accepted")
                != record.get("acceptance_class", "sharp_accepted"),
            )
        ):
            raise ValueError("FrameCrafter accepted target batch/gate provenance disagrees")
    for record in rejected:
        if not isinstance(record, Mapping):
            raise ValueError("FrameCrafter rejected target record must be an object")
        artifact = _manifest_file(
            record.get("candidate_rgb_path"), report_dir, "rejected.candidate_rgb_path"
        )
        expected_hash = str(record.get("candidate_rgb_sha256", ""))
        if len(expected_hash) != 64 or _file_sha256(artifact) != expected_hash:
            raise ValueError("FrameCrafter rejected candidate RGB hash mismatch")


def validate_manifest_payload(
    payload: Any,
    *,
    manifest_path: Path | str,
    require_provenance: bool = True,
) -> Mapping[str, Any]:
    """Validate the complete production manifest contract, fail closed.

    This validator is shared by the parent-process preflight and each dataset
    worker.  It deliberately verifies declared counts against frame content,
    keeps every original observation in-order/evaluable, and prevents a
    test-only backend or movable synthetic pose from entering SLAM.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("FrameCrafter manifest root must be a JSON object")
    if payload.get("schema") != "unblur_slam.framecrafter_manifest.v1":
        raise ValueError(
            f"unsupported FrameCrafter manifest schema {payload.get('schema')!r}"
        )
    if payload.get("uses_ground_truth_pose") is not False:
        raise ValueError("FrameCrafter manifest must declare uses_ground_truth_pose=false")
    validate_pose_source(payload.get("pose_source", ""))
    try:
        source_count = int(payload["source_frame_count"])
        generated_count = int(payload["generated_frame_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "FrameCrafter manifest counts must be finite integers"
        ) from error
    if source_count < 1 or generated_count < 0:
        raise ValueError(
            "FrameCrafter production manifest requires at least one original "
            "and a non-negative generated count"
        )
    entries = payload.get("frames")
    if not isinstance(entries, list) or not entries:
        raise ValueError("FrameCrafter manifest frames must be a non-empty list")

    manifest = Path(manifest_path).expanduser().resolve()
    manifest_dir = manifest.parent
    original_indices: list[int] = []
    actual_generated = 0
    synthetic_target_ids: list[str] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"manifest frame {position} must be an object")
        kind = str(entry.get("kind", "")).lower()
        _validate_c2w(entry.get("c2w"), f"frames[{position}].c2w")
        timestamp = float(entry.get("timestamp"))
        if not np.isfinite(timestamp):
            raise ValueError(f"frames[{position}].timestamp must be finite")
        _manifest_file(entry.get("rgb_path"), manifest_dir, f"frames[{position}].rgb_path")
        if entry.get("depth_path") not in (None, ""):
            _manifest_file(
                entry.get("depth_path"), manifest_dir, f"frames[{position}].depth_path"
            )

        if kind == "original":
            source_index = int(entry.get("source_index"))
            original_indices.append(source_index)
            if entry.get("eval") is not True:
                raise ValueError(
                    f"original frame {source_index} must remain eval=true"
                )
            if bool(entry.get("fixed_pose", False)):
                raise ValueError(
                    f"original frame {source_index} cannot be fixed_pose"
                )
            if require_provenance:
                for path_key, hash_key in (
                    ("rgb_path", "rgb_sha256"),
                    ("depth_path", "depth_sha256"),
                ):
                    if entry.get(path_key) in (None, ""):
                        if path_key == "depth_path":
                            continue
                        raise ValueError(f"original frame is missing {path_key}")
                    artifact = _manifest_file(
                        entry[path_key], manifest_dir, f"original.{path_key}"
                    )
                    expected_hash = str(entry.get(hash_key, ""))
                    if len(expected_hash) != 64 or _file_sha256(artifact) != expected_hash:
                        raise ValueError(
                            f"original artifact hash mismatch for {artifact}"
                        )
        elif kind == "synthetic":
            actual_generated += 1
            target_id = str(entry.get("target_id", "")).strip()
            if not target_id:
                raise ValueError("synthetic frame requires a non-empty target_id")
            synthetic_target_ids.append(target_id)
            if entry.get("source_index") is not None:
                raise ValueError("synthetic frame source_index must be null")
            if entry.get("eval") is not False:
                raise ValueError("synthetic frames must be eval=false")
            if entry.get("fixed_pose") is not True:
                raise ValueError("synthetic frames must be fixed_pose=true")
            if str(entry.get("acceptance_class", "sharp_accepted")) not in {
                "sharp_accepted",
                "geometry_only",
            }:
                raise ValueError("synthetic acceptance_class is invalid")
            batch_id = str(entry.get("batch_id", "")).strip()
            batch_target_ids = entry.get("batch_target_ids")
            batch_target_position = entry.get("batch_target_position")
            has_batch_provenance = any(
                value not in (None, "", [])
                for value in (batch_id, batch_target_ids, batch_target_position)
            )
            if has_batch_provenance:
                if not batch_id or not isinstance(batch_target_ids, list):
                    raise ValueError("synthetic FrameCrafter batch provenance is incomplete")
                try:
                    batch_target_position = int(batch_target_position)
                except (TypeError, ValueError) as error:
                    raise ValueError("synthetic batch_target_position must be an integer") from error
                if (
                    not 0 <= batch_target_position < len(batch_target_ids)
                    or str(batch_target_ids[batch_target_position]) != target_id
                    or len(batch_target_ids) > 4
                ):
                    raise ValueError("synthetic FrameCrafter batch target mapping is invalid")
            left, right = int(entry.get("left_index")), int(entry.get("right_index"))
            alpha = float(entry.get("alpha"))
            if not (0 <= left < right < source_count and 0.0 < alpha < 1.0):
                raise ValueError(
                    "synthetic frame requires valid left/right source indices and 0<alpha<1"
                )
            if require_provenance:
                for path_key, hash_key in (
                    ("rgb_path", "rgb_sha256"),
                    ("depth_path", "depth_sha256"),
                ):
                    if entry.get(path_key) in (None, ""):
                        if path_key == "depth_path":
                            continue
                        raise ValueError(f"synthetic frame is missing {path_key}")
                    artifact = _manifest_file(
                        entry[path_key], manifest_dir, f"synthetic.{path_key}"
                    )
                    expected_hash = str(entry.get(hash_key, ""))
                    if len(expected_hash) != 64 or _file_sha256(artifact) != expected_hash:
                        raise ValueError(
                            f"synthetic artifact hash mismatch for {artifact}"
                        )
        else:
            raise ValueError(f"unknown manifest frame kind {kind!r}")

    if original_indices != list(range(source_count)):
        raise ValueError(
            "manifest must contain every original frame exactly once and in source order"
        )
    if actual_generated != generated_count:
        raise ValueError(
            "generated_frame_count does not match actual synthetic entries: "
            f"declared={generated_count}, actual={actual_generated}"
        )
    if len(set(synthetic_target_ids)) != len(synthetic_target_ids):
        raise ValueError("synthetic target_id values must be unique")

    if require_provenance:
        signature = str(payload.get("preprocess_signature", ""))
        if len(signature) != 64 or any(ch not in "0123456789abcdef" for ch in signature):
            raise ValueError("production manifest requires a lowercase SHA-256 signature")
        generation_id = str(payload.get("generation_id", ""))
        if len(generation_id) != 32 or any(
            ch not in "0123456789abcdef" for ch in generation_id
        ):
            raise ValueError(
                "production manifest requires a lowercase 128-bit generation_id"
            )
        if payload.get("backend") != "python_api" or bool(
            payload.get("backend_test_only", True)
        ):
            raise ValueError(
                "real SLAM accepts only a non-test FrameCrafter Python API backend"
            )
        report_path = _manifest_file(
            payload.get("preprocess_report_path"),
            manifest_dir,
            "preprocess_report_path",
        )
        expected_report_hash = str(payload.get("preprocess_report_sha256", ""))
        if len(expected_report_hash) != 64 or _file_sha256(report_path) != expected_report_hash:
            raise ValueError("FrameCrafter preprocess report hash mismatch")
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        try:
            report_mismatch = not isinstance(report, Mapping) or any(
                (
                    report.get("schema")
                    != "unblur_slam.framecrafter_preprocess_report.v1",
                    report.get("uses_ground_truth_pose") is not False,
                    validate_pose_source(report.get("pose_source", ""))
                    != payload.get("pose_source"),
                    report.get("preprocess_signature") != signature,
                    report.get("generation_id") != generation_id,
                    report.get("backend") != payload.get("backend"),
                    bool(report.get("backend_test_only", True)),
                    int(report.get("source_frame_count", -1)) != source_count,
                    int(report.get("accepted_target_count", -1)) != generated_count,
                    Path(str(report.get("manifest", ""))).expanduser().resolve()
                    != manifest,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "FrameCrafter preprocess report has invalid provenance"
            ) from error
        if report_mismatch:
            raise ValueError("FrameCrafter preprocess report disagrees with manifest")
        has_batch_provenance = bool(report.get("generation_batches")) or any(
            entry.get("kind") == "synthetic" and entry.get("batch_id")
            for entry in entries
        )
        if has_batch_provenance:
            _validate_batch_preprocess_report(
                report, entries, report_dir=report_path.parent
            )
        accepted_digest = synthetic_output_digest(entries)
        source_digest = source_input_digest(entries)
        if any(
            (
                payload.get("accepted_output_sha256") != accepted_digest,
                report.get("accepted_output_sha256") != accepted_digest,
                payload.get("source_input_sha256") != source_digest,
                report.get("source_input_sha256") != source_digest,
            )
        ):
            raise ValueError(
                "FrameCrafter manifest/report frame-contract digest mismatch"
            )
    return payload


def write_manifest(path: Path | str, manifest: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(manifest), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def frame_by_source_index(frames: Sequence[FrameRecord], index: int) -> FrameRecord:
    matches = [frame for frame in frames if frame.source_index == int(index)]
    if len(matches) != 1:
        raise KeyError(f"source index {index} resolves to {len(matches)} frames")
    return matches[0]


__all__ = [
    "DepthFusionResult",
    "FrameRecord",
    "FrameCrafterGenerationBatch",
    "GateConfig",
    "GateResult",
    "PythonAPIFrameCrafterBackend",
    "SyntheticFrameResult",
    "TargetView",
    "TestOnlyBlendBackend",
    "build_manifest",
    "c2w_to_opencv_w2c",
    "evaluate_candidate",
    "frame_by_source_index",
    "framecrafter_input_arrays",
    "fuse_bilateral_depth",
    "interpolate_c2w",
    "laplacian_sharpness",
    "load_bound_unaligned_trajectory",
    "load_frames_csv",
    "plan_interpolated_targets",
    "plan_framecrafter_generation_batches",
    "project_rgbd_to_target",
    "read_depth",
    "read_rgb",
    "save_depth_npy",
    "save_depth_png",
    "save_framecrafter_npz",
    "save_rgb",
    "select_scene_wide_targets",
    "select_real_contexts",
    "select_shared_real_contexts",
    "source_input_digest",
    "synthetic_output_digest",
    "targets_from_planner_json",
    "validate_pose_source",
    "validate_pose_input_path",
    "validate_manifest_payload",
    "write_manifest",
]
