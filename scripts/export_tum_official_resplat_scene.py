#!/usr/bin/env python3
"""Export audited TUM estimates as an official cvg/ReSplat COLMAP scene.

This is deliberately a one-way, standalone bridge.  It does not modify the
Unblur-SLAM mapper and it does not describe any custom replay/refinement logic
as ReSplat.  Camera poses are read as OpenCV camera-to-world transforms from an
``estimated_frames.csv``-style file and written as COLMAP world-to-camera
``qvec``/``tvec`` records.

The exporter is fail-closed: source indices are mandatory, ground-truth or
evaluation-aligned pose provenance is rejected, image/checkpoint bytes are
hashed, an existing output is never reused, and a complete scene is first
built in a sibling temporary directory before it is renamed into place.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from PIL import Image


SCHEMA = "unblur_slam.official_resplat_colmap_scene.v1"
OFFICIAL_RESPLAT_URL = "https://github.com/cvg/resplat"
FALSE_VALUES = {"0", "false", "no"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_POSE_TOKENS = (
    "groundtruth",
    "ground_truth",
    "traj_ref",
    "reference_pose",
    "gt_pose",
    "aligned_to_gt",
)

# These names and view counts come from the official cvg/ReSplat
# scripts/infer_colmap.py presets.  The eight hexadecimal characters embedded
# in each release filename are treated as a minimum content-integrity prefix;
# a caller may additionally pin the complete SHA-256.
PRESET_SPECS: dict[str, dict[str, Any]] = {
    "dl3dv_8v_512x960": {
        "num_context": 8,
        "num_refine": 4,
        "checkpoint_filename": "resplat-base-dl3dv-512x960-view8-8179ed87.pth",
        "checkpoint_sha256_prefix": "8179ed87",
    },
    "dl3dv_16v_540x960": {
        "num_context": 16,
        "num_refine": 2,
        "checkpoint_filename": "resplat-base-dl3dv-540x960-view16-a72dc6d0.pth",
        "checkpoint_sha256_prefix": "a72dc6d0",
    },
    "dl3dv_8v_256x448": {
        "num_context": 8,
        "num_refine": 4,
        "checkpoint_filename": "resplat-base-dl3dv-256x448-view8-1934a04c.pth",
        "checkpoint_sha256_prefix": "1934a04c",
    },
    "dl3dv_16v_256x448": {
        "num_context": 16,
        "num_refine": 4,
        "checkpoint_filename": "resplat-base-dl3dv-256x448-view16-f38bf984.pth",
        "checkpoint_sha256_prefix": "f38bf984",
    },
    "dl3dv_32v_256x448": {
        "num_context": 32,
        "num_refine": 4,
        "checkpoint_filename": "resplat-base-dl3dv-256x448-view32-439b63a6.pth",
        "checkpoint_sha256_prefix": "439b63a6",
    },
    "dl3dv_8v_256x448_small": {
        "num_context": 8,
        "num_refine": 4,
        "checkpoint_filename": "resplat-small-dl3dv-256x448-view8-548993fe.pth",
        "checkpoint_sha256_prefix": "548993fe",
    },
    "dl3dv_8v_256x448_large": {
        "num_context": 8,
        "num_refine": 0,
        "checkpoint_filename": "resplat-large-dl3dv-256x448-view8-62f1703a.pth",
        "checkpoint_sha256_prefix": "62f1703a",
    },
}


@dataclass(frozen=True)
class CsvFrame:
    source_index: int
    csv_row: int
    frame_id: str
    timestamp: float
    raw_image_path: Path
    c2w: np.ndarray
    intrinsics: np.ndarray
    pose_source: str


@dataclass(frozen=True)
class ImageBinding:
    source_index: int
    path: Path
    sha256: str
    declared_sha256: Optional[str]
    width: int
    height: int


@dataclass(frozen=True)
class CameraBinding:
    width: int
    height: int
    intrinsics: np.ndarray


@dataclass(frozen=True)
class MappedImage:
    path: Path
    sha256: str
    camera: CameraBinding


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _validate_pose_source(value: object, label: str) -> str:
    source = str(value).strip()
    lowered = source.lower()
    if not source or any(token in lowered for token in FORBIDDEN_POSE_TOKENS):
        raise ValueError(
            f"{label} must identify a non-GT, non-evaluation-aligned pose source"
        )
    return source


def _is_ground_truth_sidecar_key(value: str) -> bool:
    lowered = str(value).strip().lower()
    return (
        any(token in lowered for token in FORBIDDEN_POSE_TOKENS)
        or lowered in {"gt", "reference", "reference_poses", "ref_poses"}
        or lowered.startswith("gt_")
        or lowered.endswith("_gt")
        or "_gt_" in lowered
    )


def _require_false(value: object, label: str) -> None:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be a scalar false value")
    item = array.reshape(()).item()
    if str(item).strip().lower() not in FALSE_VALUES:
        raise ValueError(f"{label} must explicitly be false, got {item!r}")


def _resolve_path(value: str, root: Path, label: str) -> Path:
    if not str(value).strip():
        raise ValueError(f"{label} is empty")
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _parse_finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a number: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def quaternion_xyzw_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    """Return a proper rotation matrix for a unit XYZW quaternion."""

    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("c2w quaternion must contain four finite XYZW values")
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 0.0 or abs(norm - 1.0) > 1.0e-4:
        raise ValueError(f"c2w quaternion must be unit length, got norm {norm:.9g}")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_qvec_qwxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to COLMAP's canonical QW,QX,QY,QZ."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"rotation must be a finite 3x3 matrix, got {rotation.shape}")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError("rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-5):
        raise ValueError("rotation is not proper/right-handed")

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    result = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    result /= np.linalg.norm(result)
    if result[0] < 0.0:
        result = -result
    return result


def _validate_c2w(c2w: np.ndarray, label: str) -> np.ndarray:
    pose = np.asarray(c2w, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{label} must be a finite 4x4 c2w matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation_to_qvec_qwxyz(pose[:3, :3])
    return pose


def c2w_to_colmap(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV c2w into COLMAP world-to-camera qvec and tvec."""

    pose = _validate_c2w(c2w, "c2w")
    rotation_w2c = pose[:3, :3].T
    translation_w2c = -rotation_w2c @ pose[:3, 3]
    return rotation_to_qvec_qwxyz(rotation_w2c), translation_w2c


def _csv_pose(row: Mapping[str, str], row_number: int) -> np.ndarray:
    q = [
        _parse_finite(row[name], f"CSV row {row_number} {name}")
        for name in ("qx", "qy", "qz", "qw")
    ]
    translation = np.asarray(
        [
            _parse_finite(row[name], f"CSV row {row_number} {name}")
            for name in ("tx", "ty", "tz")
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_xyzw_to_rotation(q)
    result[:3, 3] = translation
    return _validate_c2w(result, f"CSV row {row_number} c2w")


def load_frames_csv(
    csv_path: Path | str, *, image_root: Path | str | None = None
) -> tuple[list[CsvFrame], str, np.ndarray]:
    """Load and audit every CSV row; no row may advertise GT poses."""

    source = Path(csv_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"frames CSV does not exist: {source}")
    root = source.parent if image_root is None else Path(image_root).expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"frames CSV is empty: {source}")
    index_columns = columns & {"index", "source_index"}
    if len(index_columns) != 1:
        raise ValueError(
            "frames CSV must contain exactly one explicit source index column: "
            "index or source_index"
        )
    index_column = next(iter(index_columns))
    required = {
        "frame",
        "tx", "ty", "tz", "qx", "qy", "qz", "qw",
        "fx", "fy", "cx", "cy",
        "pose_source", "uses_ground_truth_pose",
    }
    missing = required - columns
    if missing:
        raise ValueError(f"frames CSV is missing required columns: {sorted(missing)}")

    frames: list[CsvFrame] = []
    observed_source: Optional[str] = None
    observed_k: Optional[np.ndarray] = None
    seen_indices: set[int] = set()
    for position, row in enumerate(rows, 2):
        try:
            source_index = int(str(row[index_column]).strip())
        except (TypeError, ValueError) as error:
            raise ValueError(f"CSV row {position} has an invalid source index") from error
        if source_index < 0:
            raise ValueError(f"CSV row {position} source index must be non-negative")
        if source_index in seen_indices:
            raise ValueError(f"duplicate source index in CSV: {source_index}")
        seen_indices.add(source_index)

        _require_false(
            row.get("uses_ground_truth_pose", ""),
            f"CSV row {position} uses_ground_truth_pose",
        )
        pose_source = _validate_pose_source(
            row.get("pose_source", ""), f"CSV row {position} pose_source"
        )
        if observed_source is None:
            observed_source = pose_source
        elif pose_source != observed_source:
            raise ValueError("pose_source must be identical on every CSV row")

        values = [
            _parse_finite(row[name], f"CSV row {position} {name}")
            for name in ("fx", "fy", "cx", "cy")
        ]
        if values[0] <= 0.0 or values[1] <= 0.0:
            raise ValueError(f"CSV row {position} fx/fy must be positive")
        k = np.asarray(
            [[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        if observed_k is None:
            observed_k = k
        elif not np.allclose(k, observed_k, rtol=0.0, atol=1.0e-9):
            raise ValueError("camera intrinsics K must be identical on every CSV row")

        frame_id = str(row.get("frame", "")).strip()
        rgb_value = str(row.get("rgb_path", "")).strip() or frame_id
        raw_path = _resolve_path(rgb_value, root, f"CSV row {position} RGB path")
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"CSV row {position} RGB image does not exist: {raw_path}"
            )
        timestamp_text = str(row.get("timestamp", "")).strip()
        timestamp = (
            float(source_index)
            if not timestamp_text
            else _parse_finite(timestamp_text, f"CSV row {position} timestamp")
        )
        frames.append(
            CsvFrame(
                source_index=source_index,
                csv_row=position,
                frame_id=frame_id,
                timestamp=timestamp,
                raw_image_path=raw_path,
                c2w=_csv_pose(row, position),
                intrinsics=k,
                pose_source=pose_source,
            )
        )
    assert observed_source is not None and observed_k is not None
    return frames, observed_source, observed_k


def _tokenize_indices(text: str) -> list[str]:
    pieces: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0]
        pieces.extend(piece for piece in re.split(r"[\s,]+", line.strip()) if piece)
    return pieces


def parse_indices(*, indices: Optional[str], indices_file: Optional[Path]) -> tuple[list[int], dict[str, Any]]:
    if (indices is None) == (indices_file is None):
        raise ValueError("provide exactly one of --indices or --indices-file")
    provenance: dict[str, Any]
    if indices_file is not None:
        path = indices_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"indices file does not exist: {path}")
        tokens = _tokenize_indices(path.read_text(encoding="utf-8"))
        provenance = {"kind": "file", "path": str(path), "sha256": sha256_file(path)}
    else:
        assert indices is not None
        tokens = _tokenize_indices(indices)
        provenance = {"kind": "cli", "value": indices}
    if not tokens:
        raise ValueError("source-index selection is empty")
    try:
        values = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError("source-index selection contains a non-integer") from error
    if any(value < 0 for value in values):
        raise ValueError("source indices must be non-negative")
    if len(set(values)) != len(values):
        raise ValueError("source-index selection contains duplicates")
    return sorted(values), provenance


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception as error:
        raise ValueError(f"invalid or unreadable image: {path}") from error


def _parse_mapping_camera(value: object, label: str) -> CameraBinding:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object containing width, height, and K")
    try:
        width_value = value["width"]
        height_value = value["height"]
        width = int(width_value)
        height = int(height_value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must declare integer width and height") from error
    try:
        if float(width_value) != width or float(height_value) != height:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} width/height must be exact integers") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} width/height must be positive")
    if "K" not in value:
        raise ValueError(f"{label} must declare a 3x3 K matrix")
    try:
        intrinsics = np.asarray(value["K"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} K must be numeric") from error
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"{label} K must be a finite 3x3 matrix")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError(f"{label} K must have positive fx/fy")
    pinhole_template = np.asarray(
        [
            [intrinsics[0, 0], 0.0, intrinsics[0, 2]],
            [0.0, intrinsics[1, 1], intrinsics[1, 2]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.allclose(intrinsics, pinhole_template, rtol=0.0, atol=1.0e-9):
        raise ValueError(f"{label} K is not representable by COLMAP PINHOLE")
    return CameraBinding(width=width, height=height, intrinsics=intrinsics)


def _camera_matches(left: CameraBinding, right: CameraBinding) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and np.allclose(
            left.intrinsics, right.intrinsics, rtol=0.0, atol=1.0e-9
        )
    )


def _mapping_records(payload: object) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("images JSON must contain an object")
    if "frames" in payload:
        frames = payload["frames"]
        if not isinstance(frames, list):
            raise ValueError("images JSON frames must be a list")
        if not all(isinstance(item, Mapping) for item in frames):
            raise ValueError("every images JSON frame must be an object")
        return list(frames)
    mapping = payload.get("images", payload)
    if not isinstance(mapping, Mapping):
        raise ValueError("images JSON images must be an object keyed by source index")
    records = []
    for key, value in mapping.items():
        try:
            source_index = int(str(key))
        except ValueError:
            # Top-level metadata is allowed only when an explicit "images"
            # object was provided.  A direct map must be unambiguous.
            if mapping is payload:
                raise ValueError("direct images JSON keys must all be source indices")
            raise
        if isinstance(value, str):
            records.append({"source_index": source_index, "path": value})
        elif isinstance(value, Mapping):
            records.append({"source_index": source_index, **dict(value)})
        else:
            raise ValueError(f"images JSON entry {key!r} must be a path or object")
    return records


def load_images_mapping(
    path: Path | str,
) -> tuple[dict[int, MappedImage], dict[str, Any], CameraBinding]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"images JSON does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"images JSON is invalid: {source}") from error

    top_level_camera = (
        None
        if not isinstance(payload, Mapping) or payload.get("camera") is None
        else _parse_mapping_camera(payload["camera"], "images JSON top-level camera")
    )
    result: dict[int, MappedImage] = {}
    observed_camera: Optional[CameraBinding] = None
    for position, record in enumerate(_mapping_records(payload)):
        try:
            source_index = int(record["source_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"images JSON record {position} lacks a valid source_index") from error
        if source_index < 0 or source_index in result:
            raise ValueError(f"images JSON has invalid/duplicate source_index={source_index}")
        nested = record.get("output")
        if nested is not None and not isinstance(nested, Mapping):
            raise ValueError(f"images JSON output for {source_index} must be an object")
        nested = nested if isinstance(nested, Mapping) else {}
        image_value = (
            nested.get("path")
            or record.get("output_path")
            or record.get("path")
            or record.get("image_path")
        )
        declared_hash = (
            nested.get("sha256")
            or record.get("output_sha256")
            or record.get("sha256")
        )
        image_path = _resolve_path(
            str(image_value or ""), source.parent, f"images JSON path for {source_index}"
        )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"mapped image for source_index={source_index} does not exist: {image_path}"
            )
        normalized_hash = str(declared_hash or "").strip().lower()
        if not SHA256_RE.fullmatch(normalized_hash):
            raise ValueError(
                f"mapped image source_index={source_index} must declare a full SHA-256"
            )
        actual_hash = sha256_file(image_path)
        if actual_hash != normalized_hash:
            raise ValueError(
                f"mapped image SHA-256 mismatch for source_index={source_index}"
            )
        camera_value = record.get("camera")
        if camera_value is None:
            if top_level_camera is None:
                raise ValueError(
                    f"mapped image source_index={source_index} must declare camera "
                    "width,height,K, or inherit a top-level camera"
                )
            camera = top_level_camera
        else:
            camera = _parse_mapping_camera(
                camera_value, f"images JSON camera for source_index={source_index}"
            )
            if top_level_camera is not None and not _camera_matches(
                camera, top_level_camera
            ):
                raise ValueError(
                    f"mapped camera disagrees with top-level camera for source_index={source_index}"
                )
        if observed_camera is None:
            observed_camera = camera
        elif not _camera_matches(camera, observed_camera):
            raise ValueError("all mapped images must declare one identical camera and K")
        actual_size = _image_size(image_path)
        if actual_size != (camera.width, camera.height):
            raise ValueError(
                f"mapped image size disagrees with declared camera for "
                f"source_index={source_index}: {actual_size} != "
                f"{(camera.width, camera.height)}"
            )
        result[source_index] = MappedImage(image_path, actual_hash, camera)
    if not result:
        raise ValueError("images JSON contains no image mappings")
    assert observed_camera is not None
    return (
        result,
        {
            "path": str(source),
            "sha256": sha256_file(source),
            "camera": {
                "width": observed_camera.width,
                "height": observed_camera.height,
                "K": observed_camera.intrinsics.tolist(),
            },
        },
        observed_camera,
    )


def bind_images(
    frames: Sequence[CsvFrame],
    *,
    image_mode: str,
    images_json: Path | None,
) -> tuple[
    dict[int, ImageBinding],
    Optional[dict[str, Any]],
    CameraBinding,
]:
    if image_mode == "raw":
        if images_json is not None:
            raise ValueError("--images-json is not allowed with --image-mode raw")
        mapping = None
        mapping_provenance = None
        mapping_camera = None
    else:
        if images_json is None:
            raise ValueError(
                f"--image-mode {image_mode} requires --images-json with path+SHA-256 mappings"
            )
        mapping, mapping_provenance, mapping_camera = load_images_mapping(images_json)

    bindings: dict[int, ImageBinding] = {}
    for frame in frames:
        if mapping is None:
            image_path = frame.raw_image_path
            declared = None
            actual = sha256_file(image_path)
        else:
            if frame.source_index not in mapping:
                raise ValueError(
                    f"images JSON has no mapping for selected source_index={frame.source_index}"
                )
            mapped = mapping[frame.source_index]
            image_path, actual = mapped.path, mapped.sha256
            declared = actual
        width, height = _image_size(image_path)
        bindings[frame.source_index] = ImageBinding(
            source_index=frame.source_index,
            path=image_path,
            sha256=actual,
            declared_sha256=declared,
            width=width,
            height=height,
        )
    sizes = {(binding.width, binding.height) for binding in bindings.values()}
    if len(sizes) != 1:
        raise ValueError("all exported images must have one identical resolution")
    if mapping_camera is None:
        first = next(iter(bindings.values()))
        camera = CameraBinding(
            width=first.width,
            height=first.height,
            intrinsics=np.asarray(frames[0].intrinsics, dtype=np.float64),
        )
    else:
        camera = mapping_camera
        if sizes != {(camera.width, camera.height)}:
            raise ValueError("mapped image sizes disagree with the declared camera")
    return bindings, mapping_provenance, camera


def load_pose_override(
    path: Path | str,
    key: str,
    *,
    minimum_length: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"trajectory NPZ does not exist: {source}")
    lowered_identity = f"{source.name} {key}".lower()
    if any(token in lowered_identity for token in FORBIDDEN_POSE_TOKENS):
        raise ValueError("trajectory path/key advertises GT or aligned/reference poses")
    with np.load(source, allow_pickle=False) as payload:
        available = set(payload.files)
        ground_truth_sidecars = sorted(
            name
            for name in available
            if name != "uses_ground_truth_pose" and _is_ground_truth_sidecar_key(name)
        )
        required = {key, "pose_source", "uses_ground_truth_pose"}
        missing = required - available
        if missing:
            raise ValueError(f"trajectory NPZ is missing fields: {sorted(missing)}")
        _require_false(
            payload["uses_ground_truth_pose"],
            "trajectory uses_ground_truth_pose",
        )
        pose_source_array = np.asarray(payload["pose_source"])
        if pose_source_array.size != 1:
            raise ValueError("trajectory pose_source must be a scalar string")
        pose_source = _validate_pose_source(
            pose_source_array.reshape(()).item(), "trajectory pose_source"
        )
        poses = np.asarray(payload[key], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"trajectory {key!r} must have shape Nx4x4")
    if len(poses) < minimum_length:
        raise ValueError(
            f"trajectory {key!r} needs at least {minimum_length} poses, got {len(poses)}"
        )
    for index, pose in enumerate(poses):
        _validate_c2w(pose, f"trajectory {key}[{index}]")
    return poses, {
        "path": str(source),
        "sha256": sha256_file(source),
        "key": key,
        "shape": list(poses.shape),
        "pose_source": pose_source,
        "uses_ground_truth_pose": False,
        "contains_ground_truth_sidecar": bool(ground_truth_sidecars),
        "ground_truth_sidecar_keys": ground_truth_sidecars,
        "ground_truth_sidecar_arrays_accessed": False,
    }


def _git(repo: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot inspect ReSplat git repository {repo}") from error
    return completed.stdout.strip()


def _normalize_git_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized == "git@github.com:cvg/resplat":
        normalized = OFFICIAL_RESPLAT_URL
    return normalized.lower()


def inspect_official_resplat_repo(path: Path | str) -> dict[str, Any]:
    repo = Path(path).expanduser().resolve()
    inference_script = repo / "scripts" / "infer_colmap.py"
    model_zoo = repo / "MODEL_ZOO.md"
    if not inference_script.is_file() or not model_zoo.is_file():
        raise ValueError(
            f"ReSplat repo lacks official scripts/infer_colmap.py or MODEL_ZOO.md: {repo}"
        )
    origin = _git(repo, "remote", "get-url", "origin")
    if _normalize_git_url(origin) != _normalize_git_url(OFFICIAL_RESPLAT_URL):
        raise ValueError(f"ReSplat origin is not official cvg/resplat: {origin}")
    commit = _git(repo, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("ReSplat repository has an invalid commit id")
    dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))
    if dirty:
        raise ValueError("official ReSplat repository has tracked modifications")
    return {
        "path": str(repo),
        "origin": origin,
        "expected_origin": OFFICIAL_RESPLAT_URL,
        "commit": commit,
        "tracked_worktree_clean": True,
        "infer_colmap_path": str(inference_script.resolve()),
        "infer_colmap_sha256": sha256_file(inference_script),
        "model_zoo_path": str(model_zoo.resolve()),
        "model_zoo_sha256": sha256_file(model_zoo),
    }


def checkpoint_metadata(
    *,
    model_preset: str,
    checkpoint: Path | None,
    expected_checkpoint_sha256: Optional[str],
    formal_smoke: bool,
) -> dict[str, Any]:
    try:
        spec = dict(PRESET_SPECS[model_preset])
    except KeyError as error:
        raise ValueError(f"unknown official ReSplat model preset: {model_preset}") from error
    expected_full = None
    if expected_checkpoint_sha256 is not None:
        expected_full = expected_checkpoint_sha256.strip().lower()
        if not SHA256_RE.fullmatch(expected_full):
            raise ValueError("--expected-checkpoint-sha256 must contain 64 lowercase hex digits")
    actual = None
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"ReSplat checkpoint does not exist: {resolved}")
        actual_hash = sha256_file(resolved)
        if resolved.name != spec["checkpoint_filename"]:
            raise ValueError(
                "checkpoint filename does not match the official model preset: "
                f"{resolved.name} != {spec['checkpoint_filename']}"
            )
        if not actual_hash.startswith(str(spec["checkpoint_sha256_prefix"])):
            raise ValueError("checkpoint bytes do not match the official filename SHA prefix")
        if expected_full is not None and actual_hash != expected_full:
            raise ValueError("checkpoint SHA-256 does not match the explicit expected value")
        actual = {"path": str(resolved), "filename": resolved.name, "sha256": actual_hash}
    elif formal_smoke:
        raise ValueError("--formal-smoke requires --checkpoint")
    elif expected_full is not None:
        raise ValueError("--expected-checkpoint-sha256 requires --checkpoint")
    return {
        "model_preset": model_preset,
        "num_context": int(spec["num_context"]),
        "num_refine": int(spec["num_refine"]),
        "expected_filename": spec["checkpoint_filename"],
        "expected_sha256_prefix": spec["checkpoint_sha256_prefix"],
        "explicit_expected_sha256": expected_full,
        "actual": actual,
    }


def _format_numbers(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def export_scene(
    *,
    frames_csv: Path,
    output_dir: Path,
    selected_indices: Sequence[int],
    selection_provenance: Mapping[str, Any],
    image_mode: str,
    images_json: Path | None,
    image_root: Path | None,
    resplat_repo: Path,
    model_preset: str,
    checkpoint: Path | None = None,
    expected_checkpoint_sha256: Optional[str] = None,
    trajectory_npz: Path | None = None,
    trajectory_key: Optional[str] = None,
    formal_smoke: bool = False,
) -> Path:
    """Build a complete official-ReSplat-compatible COLMAP text scene."""

    if image_mode not in {"raw", "evssm", "turtle"}:
        raise ValueError("image_mode must be raw, evssm, or turtle")
    if formal_smoke and image_mode == "raw":
        raise ValueError("--formal-smoke requires a non-raw image mode")
    indices = sorted(int(value) for value in selected_indices)
    if not indices or any(value < 0 for value in indices):
        raise ValueError("selected source indices must be non-empty and non-negative")
    if len(set(indices)) != len(indices):
        raise ValueError("selected source indices contain duplicates")
    if (trajectory_npz is None) != (trajectory_key is None):
        raise ValueError("--trajectory-npz and --trajectory-key must be provided together")

    destination = output_dir.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    csv_resolved = frames_csv.expanduser().resolve()
    frames, csv_pose_source, csv_common_k = load_frames_csv(
        csv_resolved, image_root=image_root
    )
    by_index = {frame.source_index: frame for frame in frames}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"selected source indices are absent from CSV: {missing}")
    selected = [by_index[index] for index in indices]
    bindings, mapping_provenance, effective_camera = bind_images(
        selected, image_mode=image_mode, images_json=images_json
    )
    repo_metadata = inspect_official_resplat_repo(resplat_repo)
    ckpt_metadata = checkpoint_metadata(
        model_preset=model_preset,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        formal_smoke=formal_smoke,
    )
    if formal_smoke and len(selected) < ckpt_metadata["num_context"]:
        raise ValueError(
            "formal smoke has fewer selected views than the official preset requires: "
            f"{len(selected)} < {ckpt_metadata['num_context']}"
        )

    pose_override = None
    pose_override_metadata = None
    effective_pose_source = csv_pose_source
    if trajectory_npz is not None:
        assert trajectory_key is not None
        pose_override, pose_override_metadata = load_pose_override(
            trajectory_npz, trajectory_key, minimum_length=max(indices) + 1
        )
        if formal_smoke and pose_override_metadata["contains_ground_truth_sidecar"]:
            raise ValueError(
                "formal smoke refuses a trajectory NPZ archive containing GT/reference sidecars"
            )
        effective_pose_source = str(pose_override_metadata["pose_source"])

    width, height = effective_camera.width, effective_camera.height
    effective_k = effective_camera.intrinsics
    fx, fy, cx, cy = (
        float(effective_k[0, 0]),
        float(effective_k[1, 1]),
        float(effective_k[0, 2]),
        float(effective_k[1, 2]),
    )
    source_csv_metadata = {
        "path": str(csv_resolved),
        "sha256": sha256_file(csv_resolved),
        "row_count": len(frames),
        "pose_source": csv_pose_source,
        "uses_ground_truth_pose": False,
        "K": csv_common_k.tolist(),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    installed = False
    try:
        images_dir = temporary / "images"
        sparse_dir = temporary / "sparse" / "0"
        images_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)

        cameras_text = (
            "# Camera list with one line of data per camera:\n"
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            "# Number of cameras: 1\n"
            f"1 PINHOLE {width} {height} {_format_numbers([fx, fy, cx, cy])}\n"
        )
        cameras_path = sparse_dir / "cameras.txt"
        cameras_path.write_text(cameras_text, encoding="utf-8")

        image_lines = [
            "# Image list with two lines of data per image:",
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
            "#   POINTS2D[] as (X, Y, POINT3D_ID)",
            f"# Number of images: {len(selected)}, mean observations per image: 0",
        ]
        frame_manifests: list[dict[str, Any]] = []
        filename_width = max(8, len(str(max(indices))))
        for image_id, frame in enumerate(selected, 1):
            binding = bindings[frame.source_index]
            suffix = binding.path.suffix.lower() or ".png"
            image_name = f"{frame.source_index:0{filename_width}d}{suffix}"
            exported_image = images_dir / image_name
            shutil.copyfile(binding.path, exported_image)
            exported_hash = sha256_file(exported_image)
            if exported_hash != binding.sha256:
                raise RuntimeError(
                    f"copied image hash mismatch for source_index={frame.source_index}"
                )

            effective_c2w = (
                frame.c2w
                if pose_override is None
                else pose_override[frame.source_index]
            )
            qvec, tvec = c2w_to_colmap(effective_c2w)
            image_lines.append(
                f"{image_id} {_format_numbers(qvec)} {_format_numbers(tvec)} 1 {image_name}"
            )
            image_lines.append("")
            raw_hash = sha256_file(frame.raw_image_path)
            frame_manifests.append(
                {
                    "source_index": frame.source_index,
                    "csv_row": frame.csv_row,
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp,
                    "camera_id": 1,
                    "colmap_image_id": image_id,
                    "image_name": image_name,
                    "raw_csv_image": {
                        "path": str(frame.raw_image_path),
                        "sha256": raw_hash,
                    },
                    "selected_image": {
                        "mode_label": image_mode,
                        "source_path": str(binding.path),
                        "source_sha256": binding.sha256,
                        "declared_sha256": binding.declared_sha256,
                        "exported_relative_path": f"images/{image_name}",
                        "exported_sha256": exported_hash,
                        "width": binding.width,
                        "height": binding.height,
                    },
                    "K": effective_k.tolist(),
                    "csv_camera_audit": {
                        "K": frame.intrinsics.tolist(),
                        "raw_image_width": _image_size(frame.raw_image_path)[0],
                        "raw_image_height": _image_size(frame.raw_image_path)[1],
                    },
                    "csv_pose_audit": {
                        "pose_source": frame.pose_source,
                        "uses_ground_truth_pose": False,
                        "c2w_opencv": frame.c2w.tolist(),
                        "c2w_sha256": _canonical_json_sha256(frame.c2w.tolist()),
                    },
                    "effective_pose": {
                        "pose_source": effective_pose_source,
                        "uses_ground_truth_pose": False,
                        "source": "trajectory_npz" if pose_override is not None else "frames_csv",
                        "c2w_opencv": effective_c2w.tolist(),
                        "c2w_sha256": _canonical_json_sha256(effective_c2w.tolist()),
                        "colmap_w2c_qvec_qwxyz": qvec.tolist(),
                        "colmap_w2c_tvec": tvec.tolist(),
                    },
                }
            )

        images_path = sparse_dir / "images.txt"
        images_path.write_text("\n".join(image_lines) + "\n", encoding="utf-8")
        points_path = sparse_dir / "points3D.txt"
        points_path.write_text(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
            "# Number of points: 0, mean track length: 0\n",
            encoding="utf-8",
        )

        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "exporter": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "artifact_class": "official_cvg_resplat_colmap_input_scene",
            "formal_smoke": bool(formal_smoke),
            "selection": {
                "source_indices": indices,
                "count": len(indices),
                "provenance": dict(selection_provenance),
            },
            "source_csv": source_csv_metadata,
            "pose_override": pose_override_metadata,
            "effective_pose_source": effective_pose_source,
            "camera": {
                "model": "PINHOLE",
                "camera_id": 1,
                "width": width,
                "height": height,
                "K": effective_k.tolist(),
                "source": "images_json" if image_mode != "raw" else "frames_csv",
                "intrinsics_identical_on_all_csv_rows": True,
            },
            "images": {
                "mode_label": image_mode,
                "mapping_manifest": mapping_provenance,
                "mapped_hashes_verified": image_mode != "raw",
                "generator_or_checkpoint_verified_by_exporter": False,
                "undistortion_claimed": False,
                "note": (
                    "image_mode is a caller-supplied provenance label; this exporter "
                    "only verifies paths, bytes, dimensions, and declared hashes"
                ),
            },
            "official_resplat": {
                "repository": repo_metadata,
                "checkpoint": ckpt_metadata,
                "integration": "standalone_official_infer_colmap_input",
            },
            "ground_truth_contract": {
                "uses_ground_truth_pose": False,
                "ground_truth_pose_used": False,
                "ground_truth_file_read": bool(
                    pose_override_metadata is not None
                    and pose_override_metadata["contains_ground_truth_sidecar"]
                ),
                "contains_ground_truth_sidecar": bool(
                    pose_override_metadata is not None
                    and pose_override_metadata["contains_ground_truth_sidecar"]
                ),
                "ground_truth_sidecar_arrays_accessed": False,
                "csv_rows_all_declared_false": True,
                "effective_pose_declared_false": True,
            },
            "colmap": {
                "pose_convention": "world_to_camera",
                "qvec_order": "qw_qx_qy_qz",
                "camera_coordinate_convention": "opencv_y_down_z_forward",
                "sparse_model": "sparse/0",
                "cameras_sha256": sha256_file(cameras_path),
                "images_sha256": sha256_file(images_path),
                "points3D_sha256": sha256_file(points_path),
            },
            "frames": frame_manifests,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        for path in (
            cameras_path,
            images_path,
            points_path,
            manifest_path,
            *(images_dir / frame["image_name"] for frame in frame_manifests),
        ):
            _fsync_file(path)
        _fsync_directory(images_dir)
        _fsync_directory(sparse_dir)
        _fsync_directory(sparse_dir.parent)
        _fsync_directory(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing concurrent overwrite of output: {destination}")
        os.rename(temporary, destination)
        installed = True
        _fsync_directory(destination.parent)
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)
    return destination


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--indices", help="Comma/space-separated source indices")
    selection.add_argument("--indices-file", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--image-mode", choices=("raw", "evssm", "turtle"), required=True)
    parser.add_argument("--images-json", type=Path)
    parser.add_argument("--trajectory-npz", type=Path)
    parser.add_argument("--trajectory-key")
    parser.add_argument("--resplat-repo", type=Path, required=True)
    parser.add_argument(
        "--model-preset", choices=tuple(PRESET_SPECS),
        default="dl3dv_8v_256x448_small",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--formal-smoke", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    indices, selection_provenance = parse_indices(
        indices=args.indices, indices_file=args.indices_file
    )
    output = export_scene(
        frames_csv=args.frames_csv,
        output_dir=args.output_dir,
        selected_indices=indices,
        selection_provenance=selection_provenance,
        image_mode=args.image_mode,
        images_json=args.images_json,
        image_root=args.image_root,
        resplat_repo=args.resplat_repo,
        model_preset=args.model_preset,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        trajectory_npz=args.trajectory_npz,
        trajectory_key=args.trajectory_key,
        formal_smoke=args.formal_smoke,
    )
    print(
        f"exported {len(indices)} audited frames to {output} for official "
        f"cvg/ReSplat preset {args.model_preset}; no GT pose file was read"
    )
    return output


def main(argv: Optional[list[str]] = None) -> int:
    try:
        run(parse_args(argv))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
