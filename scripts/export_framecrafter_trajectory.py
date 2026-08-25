#!/usr/bin/env python3
"""Export a first-pass Unblur-SLAM trajectory as FrameCrafter frame CSV.

Only explicitly *unaligned estimated* trajectory keys are accepted.  The TUM
``groundtruth.txt`` file is never opened: RGB/depth paths are associated from
``rgb.txt`` and ``depth.txt`` and paired positionally with the full DROID
``traj_est_not_align`` output.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Mapping, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framecrafter_pipeline import (  # noqa: E402
    load_bound_unaligned_trajectory,
    validate_pose_source,
)

UNALIGNED_POSE_KEYS = (
    "traj_est_not_align",
    "traj_est_unaligned",
    "poses_not_align",
    "poses_unaligned",
    "trajectory_not_aligned",
)
UNALIGNED_TIMESTAMP_KEYS = (
    "traj_est_not_align_timestamps",
    "traj_est_unaligned_timestamps",
    "poses_not_align_timestamps",
    "poses_unaligned_timestamps",
)
UNALIGNED_EVAL_MASK_KEYS = (
    "traj_est_not_align_eval_mask",
    "traj_est_unaligned_eval_mask",
    "poses_not_align_eval_mask",
    "poses_unaligned_eval_mask",
)
FORBIDDEN_PROVENANCE_TOKENS = (
    "groundtruth",
    "ground_truth",
    "traj_ref",
    "reference_pose",
    "gt_pose",
    "aligned_to_gt",
)


@dataclass(frozen=True)
class TumListEntry:
    timestamp: float
    relative_path: str
    absolute_path: Path


@dataclass(frozen=True)
class TumAssociation:
    rgb: TumListEntry
    depth: TumListEntry
    raw_rgb_index: int
    raw_depth_index: int


def _forbid_gt_provenance(value: str, what: str) -> None:
    lowered = str(value).lower()
    if any(token in lowered for token in FORBIDDEN_PROVENANCE_TOKENS):
        raise ValueError(f"{what} advertises ground-truth/reference poses: {value}")


def _scalar_bool(value) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("uses_ground_truth_pose must be a scalar")
    item = array.reshape(()).item()
    if isinstance(item, str):
        normalized = item.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
        raise ValueError(f"invalid uses_ground_truth_pose value {item!r}")
    if item not in (0, 1, False, True):
        raise ValueError(f"invalid uses_ground_truth_pose value {item!r}")
    return bool(item)


def _validate_rigid_poses(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError(f"unaligned trajectory must be Nx4x4, got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("unaligned trajectory contains non-finite values")
    if not np.allclose(poses[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        raise ValueError("unaligned trajectory contains invalid homogeneous transforms")
    rotations = poses[:, :3, :3]
    orthogonality = np.swapaxes(rotations, 1, 2) @ rotations
    if not np.allclose(orthogonality, np.eye(3), atol=3e-4):
        raise ValueError("unaligned trajectory rotations are not orthonormal")
    if np.any(np.linalg.det(rotations) < 0.999):
        raise ValueError("unaligned trajectory contains a left-handed rotation")
    return poses


def load_unaligned_trajectory(
    npz_path: Path | str, trajectory_key: Optional[str] = None
) -> tuple[np.ndarray, str, str]:
    """Load only a key whose contract explicitly says it is unaligned."""

    path = Path(npz_path).expanduser().resolve()
    _forbid_gt_provenance(path.name, "trajectory filename")
    with np.load(path, allow_pickle=False) as payload:
        available = set(payload.files)
        if trajectory_key is not None:
            if trajectory_key not in UNALIGNED_POSE_KEYS:
                raise ValueError(
                    f"trajectory key {trajectory_key!r} is not an allowed unaligned estimate key"
                )
            key = trajectory_key
        else:
            key = next((name for name in UNALIGNED_POSE_KEYS if name in available), None)
            if key is None:
                raise KeyError(
                    "NPZ has no explicitly unaligned estimated trajectory key; "
                    f"expected one of {UNALIGNED_POSE_KEYS}, found {sorted(available)}"
                )
        if key not in available:
            raise KeyError(f"trajectory key {key!r} not found in {path}")
        if "uses_ground_truth_pose" not in available:
            raise ValueError(
                "unaligned trajectory NPZ must explicitly declare "
                "uses_ground_truth_pose=false"
            )
        if _scalar_bool(payload["uses_ground_truth_pose"]):
            raise ValueError("trajectory declares uses_ground_truth_pose=true")
        if "pose_source" not in available:
            raise ValueError(
                "unaligned trajectory NPZ must explicitly declare pose_source"
            )
        source = validate_pose_source(
            str(np.asarray(payload["pose_source"]).reshape(()).item())
        )
        poses = _validate_rigid_poses(payload[key])

        timestamp_key = next(
            (name for name in UNALIGNED_TIMESTAMP_KEYS if name in available), None
        )
        if timestamp_key is not None:
            indices = np.asarray(payload[timestamp_key], dtype=np.float64).reshape(-1)
            expected = np.arange(len(poses), dtype=np.float64)
            if len(indices) != len(poses) or not np.allclose(indices, expected, atol=1e-5):
                raise ValueError(
                    f"{timestamp_key} must map the full trajectory to dataset indices 0..N-1"
                )
        eval_mask_key = next(
            (name for name in UNALIGNED_EVAL_MASK_KEYS if name in available), None
        )
        if eval_mask_key is not None:
            eval_mask = np.asarray(payload[eval_mask_key])
            if eval_mask.shape != (len(poses),):
                raise ValueError(
                    f"{eval_mask_key} must have one flag per trajectory pose"
                )
            if not np.issubdtype(eval_mask.dtype, np.bool_):
                raise ValueError(f"{eval_mask_key} must contain boolean flags")
            if not bool(np.all(eval_mask)):
                raise ValueError(
                    "trajectory contains eval=false synthetic frames; export the "
                    "original-stream first-pass trajectory instead"
                )
    # Reuse the exact production CSV contract so exporter and loader cannot
    # drift on source-index or eval-mask provenance rules.
    poses, source = load_bound_unaligned_trajectory(path, key)
    return poses, key, source


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tum_list(path: Path | str, root: Path | str) -> list[TumListEntry]:
    """Read one TUM timestamp/path list without consulting any pose file."""

    path = Path(path).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"invalid TUM list row {path}:{line_number}: {line!r}")
            timestamp = float(fields[0])
            if not np.isfinite(timestamp):
                raise ValueError(f"non-finite TUM timestamp at {path}:{line_number}")
            relative = fields[1].strip()
            resolved = (root / relative).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(f"TUM path escapes dataset root: {relative}") from error
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            entries.append(TumListEntry(timestamp, relative, resolved))
    if not entries:
        raise ValueError(f"no entries found in {path}")
    timestamps = np.asarray([entry.timestamp for entry in entries])
    if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
        raise ValueError(f"TUM timestamps are not ordered in {path}")
    return entries


def associate_tum_rgb_depth(
    rgb_entries: Iterable[TumListEntry],
    depth_entries: Iterable[TumListEntry],
    *,
    max_delta: float = 0.08,
) -> list[TumAssociation]:
    """Match each RGB frame to its nearest depth frame, as TUM_RGB does."""

    rgb_entries = list(rgb_entries)
    depth_entries = list(depth_entries)
    if max_delta <= 0:
        raise ValueError("max_delta must be positive")
    depth_times = np.asarray([entry.timestamp for entry in depth_entries])
    associations = []
    for rgb_index, rgb in enumerate(rgb_entries):
        insertion = int(np.searchsorted(depth_times, rgb.timestamp))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(depth_entries)]
        depth_index = min(candidates, key=lambda index: abs(depth_times[index] - rgb.timestamp))
        if abs(depth_times[depth_index] - rgb.timestamp) < max_delta:
            associations.append(
                TumAssociation(rgb, depth_entries[depth_index], rgb_index, depth_index)
            )
    if not associations:
        raise ValueError("no RGB/depth pairs satisfy the association threshold")
    return associations


def select_dataset_associations(
    associations: Iterable[TumAssociation],
    *,
    frame_rate: float = 32.0,
    stride: int = 1,
    max_frames: int = -1,
) -> list[TumAssociation]:
    """Mirror TUM_RGB's frame-rate selection followed by max_frames/stride."""

    associations = list(associations)
    if not associations:
        raise ValueError("cannot select from an empty association list")
    if stride < 1:
        raise ValueError("stride must be positive")
    if frame_rate > 0:
        selected = [associations[0]]
        min_interval = 1.0 / frame_rate
        for association in associations[1:]:
            if association.rgb.timestamp - selected[-1].rgb.timestamp > min_interval:
                selected.append(association)
    else:
        selected = associations
    if max_frames >= 0:
        selected = selected[:max_frames]
    return selected[::stride]


def rotation_matrix_to_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized XYZW quaternion."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion


def write_framecrafter_csv(
    output_path: Path | str,
    poses_c2w: np.ndarray,
    associations: Iterable[TumAssociation],
    intrinsics: Mapping[str, float],
    *,
    pose_source: str = "droid_traj_est_not_align",
    trajectory_path: Path | str,
    trajectory_key: str,
) -> Path:
    """Write the CSV contract consumed by ``load_frames_csv``."""

    output_path = Path(output_path).expanduser().resolve()
    _forbid_gt_provenance(output_path.name, "output CSV filename")
    pose_source = validate_pose_source(pose_source)
    trajectory_path = Path(trajectory_path).expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory provenance file not found: {trajectory_path}")
    if trajectory_key not in UNALIGNED_POSE_KEYS:
        raise ValueError(f"unsafe trajectory provenance key {trajectory_key!r}")
    trajectory_sha256 = _sha256_file(trajectory_path)
    associations = list(associations)
    poses_c2w = _validate_rigid_poses(poses_c2w)
    if len(poses_c2w) != len(associations):
        raise ValueError(
            "trajectory/TUM association length mismatch: "
            f"{len(poses_c2w)} poses for {len(associations)} frames; "
            "check frame-rate, stride, and max-frames against the first-pass config"
        )
    required_intrinsics = {key: float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")}
    if not np.isfinite(list(required_intrinsics.values())).all():
        raise ValueError("intrinsics must be finite")

    fieldnames = [
        "index", "frame", "timestamp", "rgb_path", "depth_path",
        "tx", "ty", "tz", "qx", "qy", "qz", "qw",
        "fx", "fy", "cx", "cy", "eval", "pose_source",
        "uses_ground_truth_pose", "trajectory_path", "trajectory_sha256",
        "trajectory_key",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, (pose, association) in enumerate(zip(poses_c2w, associations)):
                qx, qy, qz, qw = rotation_matrix_to_xyzw(pose[:3, :3])
                writer.writerow(
                    {
                        "index": index,
                        "frame": association.rgb.relative_path,
                        "timestamp": f"{association.rgb.timestamp:.9f}",
                        "rgb_path": str(association.rgb.absolute_path),
                        "depth_path": str(association.depth.absolute_path),
                        "tx": f"{pose[0, 3]:.12g}",
                        "ty": f"{pose[1, 3]:.12g}",
                        "tz": f"{pose[2, 3]:.12g}",
                        "qx": f"{qx:.12g}",
                        "qy": f"{qy:.12g}",
                        "qz": f"{qz:.12g}",
                        "qw": f"{qw:.12g}",
                        **{key: f"{value:.12g}" for key, value in required_intrinsics.items()},
                        "eval": "true",
                        "pose_source": pose_source,
                        "uses_ground_truth_pose": "false",
                        "trajectory_path": str(trajectory_path),
                        "trajectory_sha256": trajectory_sha256,
                        "trajectory_key": trajectory_key,
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output_path


def _resolve_from_config(args: argparse.Namespace) -> None:
    if args.config is None:
        return
    from thirdparty.glorie_slam.config import load_config

    config = load_config(str(args.config), str(ROOT / "configs" / "unblur_slam.yaml"))
    if args.tum_root is None:
        args.tum_root = Path(config["data"]["dataset_root"]) / config["data"]["input_folder"]
    for key in ("fx", "fy", "cx", "cy"):
        if getattr(args, key) is None:
            setattr(args, key, float(config["cam"][key]))
    if args.stride is None:
        args.stride = int(config.get("stride", 1))
    if args.max_frames is None:
        args.max_frames = int(config.get("max_frames", -1))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-npz", type=Path, required=True)
    parser.add_argument("--trajectory-key", choices=UNALIGNED_POSE_KEYS)
    parser.add_argument("--tum-root", type=Path)
    parser.add_argument("--config", type=Path, help="Optional Unblur-SLAM scene config")
    parser.add_argument("--output", type=Path, required=True)
    for key in ("fx", "fy", "cx", "cy"):
        parser.add_argument(f"--{key}", type=float)
    parser.add_argument("--max-rgb-depth-dt", type=float, default=0.08)
    parser.add_argument("--frame-rate", type=float, default=32.0)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    _resolve_from_config(args)
    if args.tum_root is None:
        raise ValueError("--tum-root is required unless --config supplies it")
    missing_intrinsics = [key for key in ("fx", "fy", "cx", "cy") if getattr(args, key) is None]
    if missing_intrinsics:
        raise ValueError(f"missing intrinsics: {missing_intrinsics}; pass them or use --config")
    stride = 1 if args.stride is None else args.stride
    max_frames = -1 if args.max_frames is None else args.max_frames

    poses, selected_key, pose_source = load_unaligned_trajectory(
        args.trajectory_npz, trajectory_key=args.trajectory_key
    )
    tum_root = args.tum_root.expanduser().resolve()
    rgb = read_tum_list(tum_root / "rgb.txt", tum_root)
    depth = read_tum_list(tum_root / "depth.txt", tum_root)
    associations = select_dataset_associations(
        associate_tum_rgb_depth(rgb, depth, max_delta=args.max_rgb_depth_dt),
        frame_rate=args.frame_rate,
        stride=stride,
        max_frames=max_frames,
    )
    output = write_framecrafter_csv(
        args.output,
        poses,
        associations,
        {key: getattr(args, key) for key in ("fx", "fy", "cx", "cy")},
        pose_source=pose_source,
        trajectory_path=args.trajectory_npz,
        trajectory_key=selected_key,
    )
    print(
        f"exported {len(associations)} FrameCrafter frames from {selected_key} "
        f"to {output} (groundtruth.txt was not read)"
    )
    return output


def main(argv: Optional[list[str]] = None) -> int:
    try:
        run(parse_args(argv))
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
