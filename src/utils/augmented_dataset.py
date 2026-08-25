"""Load a FrameCrafter augmentation manifest into an existing dataset.

The augmentation is intentionally represented as a thin dataset view.  The
original dataset parser remains responsible for calibration and image
preprocessing, while this module only replaces its ordered frame tables.  A
synthetic frame is an additional training observation; it is never ground
truth and is excluded from evaluation by default.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "unblur_slam.framecrafter_manifest.v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_path(value: str | None, manifest_dir: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    return str(path.resolve())


def _pose_like(c2w: Any, template: np.ndarray) -> np.ndarray:
    pose = np.asarray(c2w, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Synthetic c2w must be 4x4, got {pose.shape}")
    if template.ndim == 3:
        return np.repeat(pose[None], template.shape[0], axis=0)
    return pose


def _validate_original(entry: dict[str, Any], frame_count: int) -> int:
    if "source_index" not in entry:
        raise ValueError("Original manifest entries require source_index")
    source_index = int(entry["source_index"])
    if not 0 <= source_index < frame_count:
        raise IndexError(
            f"source_index={source_index} outside original dataset of {frame_count} frames"
        )
    return source_index


def apply_framecrafter_manifest(
    dataset,
    manifest_path: str | Path,
    *,
    expected_signature: str | None = None,
):
    """Mutate ``dataset`` to expose the augmented sequence described by JSON.

    The manifest is applied independently in the tracker and mapper processes,
    so it contains paths and matrices rather than live tensors.  This keeps the
    multiprocessing boundary deterministic and avoids serializing generated
    images through a pipe.
    """

    path = Path(manifest_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    from src.framecrafter_pipeline import validate_manifest_payload

    validate_manifest_payload(
        payload, manifest_path=path, require_provenance=True
    )
    if (
        expected_signature is not None
        and payload.get("preprocess_signature") != expected_signature
    ):
        raise ValueError(
            "FrameCrafter manifest changed after parent preflight: "
            f"expected {expected_signature}, got {payload.get('preprocess_signature')!r}"
        )

    entries = payload.get("frames")
    if not isinstance(entries, list) or not entries:
        raise ValueError("FrameCrafter manifest must contain a non-empty frames list")

    original_color_paths = list(dataset.color_paths)
    original_depth_paths = (
        None if dataset.depth_paths is None else list(dataset.depth_paths)
    )
    original_poses = None if dataset.poses is None else np.asarray(dataset.poses)
    original_gt_paths = (
        list(dataset.gt_paths) if hasattr(dataset, "gt_paths") else None
    )
    original_timestamps = (
        None
        if getattr(dataset, "image_timestamps", None) is None
        else np.asarray(dataset.image_timestamps)
    )
    dataset_name = str(
        getattr(
            dataset,
            "name",
            getattr(dataset, "config", {}).get("dataset", ""),
        )
    ).lower()
    if dataset_name in {"tumrgbd", "tumrgb"} and original_timestamps is None:
        raise ValueError(
            "TUM FrameCrafter augmentation requires the exact selected RGB "
            "timestamps for second-pass provenance validation"
        )

    if original_poses is None:
        raise ValueError(
            "FrameCrafter augmentation requires non-GT estimated poses in the manifest "
            "and a pose-shaped dataset table"
        )
    if original_gt_paths is not None and len(original_gt_paths) != len(original_color_paths):
        raise ValueError(
            "FrameCrafter augmentation currently supports one GT/observation path per "
            "frame; multi-exposure datasets must be converted to a manifest first"
        )

    colors: list[str] = []
    depths: list[str] | None = [] if original_depth_paths is not None else None
    poses: list[np.ndarray] = []
    gt_paths: list[str] | None = [] if original_gt_paths is not None else None
    timestamps: list[float] | None = [] if original_timestamps is not None else None
    metadata: list[dict[str, Any]] = []
    original_sequence: list[int] = []
    manifest_dir = path.parent

    for augmented_index, entry in enumerate(entries):
        kind = str(entry.get("kind", "")).lower()
        if kind == "original":
            source_index = _validate_original(entry, len(original_color_paths))
            original_sequence.append(source_index)
            color_path = original_color_paths[source_index]
            depth_path = (
                None if original_depth_paths is None else original_depth_paths[source_index]
            )
            pose = original_poses[source_index]
            gt_path = (
                None if original_gt_paths is None else original_gt_paths[source_index]
            )
            timestamp = (
                None if original_timestamps is None else float(original_timestamps[source_index])
            )
            if _sha256_file(color_path) != entry.get("rgb_sha256"):
                raise ValueError(
                    "FrameCrafter source RGB does not match the second-pass dataset "
                    f"at source_index={source_index}"
                )
            if original_depth_paths is not None:
                if depth_path is None or _sha256_file(depth_path) != entry.get(
                    "depth_sha256"
                ):
                    raise ValueError(
                        "FrameCrafter source depth does not match the second-pass "
                        f"dataset at source_index={source_index}"
                    )
            if timestamp is not None and not np.isclose(
                timestamp, float(entry.get("timestamp")), rtol=0.0, atol=1.0e-7
            ):
                raise ValueError(
                    "FrameCrafter source timestamp does not match the second-pass "
                    f"dataset at source_index={source_index}"
                )
            synthetic = False
            confidence = 1.0
            eval_frame = bool(entry.get("eval", True))
        elif kind == "synthetic":
            source_index = -1
            color_path = _absolute_path(entry.get("rgb_path"), manifest_dir)
            depth_path = _absolute_path(entry.get("depth_path"), manifest_dir)
            if color_path is None or not Path(color_path).is_file():
                raise FileNotFoundError(f"Missing synthetic RGB for frame {augmented_index}: {color_path}")
            if original_depth_paths is not None and (
                depth_path is None or not Path(depth_path).is_file()
            ):
                raise FileNotFoundError(
                    "RGB-D augmentation requires a gated synthetic depth map; "
                    f"missing for frame {augmented_index}: {depth_path}"
                )
            pose = _pose_like(entry.get("c2w"), original_poses[0])
            gt_path = color_path if original_gt_paths is not None else None
            left = int(entry["left_index"])
            right = int(entry["right_index"])
            alpha = float(entry["alpha"])
            if original_timestamps is None:
                timestamp = None
            else:
                timestamp = float(
                    (1.0 - alpha) * original_timestamps[left]
                    + alpha * original_timestamps[right]
                )
            synthetic = True
            confidence = float(entry.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Synthetic confidence must be in [0,1], got {confidence}")
            if bool(entry.get("eval", False)):
                raise ValueError("Synthetic FrameCrafter observations cannot be eval frames")
            eval_frame = False
        else:
            raise ValueError(f"Unknown manifest frame kind {kind!r}")

        colors.append(str(color_path))
        if depths is not None:
            depths.append(str(depth_path))
        poses.append(np.asarray(pose))
        if gt_paths is not None:
            gt_paths.append(str(gt_path))
        if timestamps is not None:
            timestamps.append(float(timestamp))
        metadata.append(
            {
                **entry,
                "augmented_index": augmented_index,
                "source_index": source_index,
                "synthetic": synthetic,
                "eval": eval_frame,
                "confidence": confidence,
            }
        )

    expected_original_sequence = list(range(len(original_color_paths)))
    if original_sequence != expected_original_sequence:
        raise ValueError(
            "FrameCrafter manifest must contain every original source frame "
            "exactly once and in order; got "
            f"{original_sequence}, expected {expected_original_sequence}"
        )

    dataset.color_paths = colors
    dataset.depth_paths = depths
    dataset.poses = np.asarray(poses)
    if gt_paths is not None:
        dataset.gt_paths = gt_paths
    if timestamps is not None:
        dataset.image_timestamps = np.asarray(timestamps)
    dataset.n_img = len(colors)
    dataset.frame_metadata = metadata
    dataset.framecrafter_manifest = str(path)
    dataset.original_frame_count = len(original_color_paths)
    return dataset
