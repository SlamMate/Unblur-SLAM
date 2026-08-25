#!/usr/bin/env python3
"""Materialize audited TUM RGB inputs for the official cvg/ReSplat model.

EVSSM tensors accepted by this tool are the tensors delivered to the tracker:
they have already undergone the TUM undistort and tracker resize/crop.  They
therefore replace the processed raw image *after* geometry preprocessing and
must already match ``--width``/``--height``.  A raw image is undistorted and
resized only when its corresponding EVSSM tensor is absent.

The command is deliberately fail-closed.  It accepts only the official
Unblur-SLAM EVSSM checkpoint by content hash, never loads tensors through
unrestricted pickle, and never overwrites an existing output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from PIL import Image


SCHEMA = "unblur_slam.official_resplat_inputs.v1"
OFFICIAL_UNBLUR_EVSSM_SHA256 = (
    "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
)
DISTORTION_KEYS = ("k1", "k2", "p1", "p2", "k3")
FORBIDDEN_BACKEND_MARKERS = ("turtle", "gopro")
_TOKEN_SPLIT = re.compile(r"[\s,]+")


def sha256_file(path: Path | str) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object, label: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


def _resolve_rgb_path(csv_path: Path, row: Mapping[str, str], index: int) -> Path:
    value = str(row.get("rgb_path", "")).strip() or str(
        row.get("frame", "")
    ).strip()
    if not value:
        raise ValueError(f"source index {index} has neither rgb_path nor frame")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = csv_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"raw RGB for source index {index} not found: {path}")
    return path


def _load_rows(frames_csv: Path) -> tuple[dict[int, dict[str, str]], list[int]]:
    if not frames_csv.is_file():
        raise FileNotFoundError(f"TUM source CSV not found: {frames_csv}")
    with frames_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)
    required = {"index", "fx", "fy", "cx", "cy"}
    missing = required - fieldnames
    if missing:
        raise ValueError(f"TUM source CSV is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError(f"TUM source CSV is empty: {frames_csv}")

    by_index: dict[int, dict[str, str]] = {}
    order: list[int] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            index = int(str(row.get("index", "")).strip())
        except ValueError as error:
            raise ValueError(f"invalid integer index at CSV line {line_number}") from error
        if index < 0:
            raise ValueError(f"negative source index at CSV line {line_number}: {index}")
        if index in by_index:
            raise ValueError(f"duplicate source index in CSV: {index}")
        by_index[index] = dict(row)
        order.append(index)
    return by_index, order


def parse_indices_file(path: Path | str) -> list[int]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"indices file not found: {source}")
    tokens: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        tokens.extend(token for token in _TOKEN_SPLIT.split(line.split("#", 1)[0]) if token)
    return _validate_indices(tokens, source_label="indices file")


def parse_indices(values: Sequence[str]) -> list[int]:
    tokens = [
        token
        for value in values
        for token in _TOKEN_SPLIT.split(str(value).strip())
        if token
    ]
    return _validate_indices(tokens, source_label="--indices")


def _validate_indices(values: Sequence[object], *, source_label: str) -> list[int]:
    if not values:
        raise ValueError(f"{source_label} contains no indices")
    try:
        indices = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source_label} must contain only integer indices") from error
    if any(index < 0 for index in indices):
        raise ValueError(f"{source_label} contains a negative index")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{source_label} contains duplicate indices")
    return indices


def _checkpoint_record(checkpoint: Path) -> dict[str, str]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EVSSM checkpoint not found: {checkpoint}")
    lowered = checkpoint.as_posix().lower()
    marker = next(
        (item for item in FORBIDDEN_BACKEND_MARKERS if item in lowered), None
    )
    if marker is not None:
        raise ValueError(
            f"forbidden {marker!r} checkpoint/backend path for official Unblur-SLAM EVSSM"
        )
    digest = sha256_file(checkpoint)
    if digest != OFFICIAL_UNBLUR_EVSSM_SHA256:
        raise ValueError(
            "EVSSM checkpoint SHA-256 mismatch: expected official Unblur-SLAM "
            f"{OFFICIAL_UNBLUR_EVSSM_SHA256}, got {digest}"
        )
    return {"path": str(checkpoint), "sha256": digest}


def _distortion_for_row(
    row: Mapping[str, str], override: Sequence[float] | None, index: int
) -> tuple[np.ndarray, dict[str, float]]:
    if override is not None:
        if len(override) != len(DISTORTION_KEYS):
            raise ValueError("distortion override must contain k1 k2 p1 p2 k3")
        values = [
            _finite_float(value, f"distortion override {key}")
            for key, value in zip(DISTORTION_KEYS, override)
        ]
    else:
        missing = [key for key in DISTORTION_KEYS if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(
                f"source index {index} has no complete distortion; missing {missing}; "
                "pass --distortion k1 k2 p1 p2 k3"
            )
        values = [
            _finite_float(row[key], f"source index {index} {key}")
            for key in DISTORTION_KEYS
        ]
    named = dict(zip(DISTORTION_KEYS, values))
    return np.asarray(values, dtype=np.float64), named


def _intrinsics_for_row(row: Mapping[str, str], index: int) -> dict[str, float]:
    intrinsics = {
        key: _finite_float(row.get(key, ""), f"source index {index} {key}")
        for key in ("fx", "fy", "cx", "cy")
    }
    if intrinsics["fx"] <= 0.0 or intrinsics["fy"] <= 0.0:
        raise ValueError(f"source index {index} focal lengths must be positive")
    return intrinsics


def _camera_matrix(intrinsics: Mapping[str, float]) -> np.ndarray:
    return np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _scaled_intrinsics(
    intrinsics: Mapping[str, float], scale_x: float, scale_y: float
) -> dict[str, float]:
    return {
        "fx": float(intrinsics["fx"] * scale_x),
        "fy": float(intrinsics["fy"] * scale_y),
        "cx": float(intrinsics["cx"] * scale_x),
        "cy": float(intrinsics["cy"] * scale_y),
    }


def _load_raw_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _load_evssm_tensor(
    path: Path, *, width: int, height: int, source_index: int
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production dependency error
        raise RuntimeError("PyTorch is required to load EVSSM tensors") from error

    # Do not add a compatibility fallback here: unrestricted pickle is forbidden.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    container_schema: str
    container_timestamp: int | None
    if isinstance(payload, torch.Tensor):
        tensor_payload = payload
        container_schema = "bare_tensor"
        container_timestamp = None
    elif type(payload) is dict:
        required_keys = {"tensor", "shape", "dtype", "timestamp"}
        actual_keys = set(payload)
        if actual_keys != required_keys:
            raise ValueError(
                "EVSSM safe dict must contain exactly keys "
                f"{sorted(required_keys)}, got {sorted(map(str, actual_keys))}"
            )
        tensor_payload = payload["tensor"]
        if not isinstance(tensor_payload, torch.Tensor):
            raise ValueError("EVSSM safe dict tensor must be a torch.Tensor")
        if payload["shape"] != tensor_payload.shape:
            raise ValueError(
                "EVSSM safe dict shape does not match tensor.shape: "
                f"{payload['shape']!r} != {tensor_payload.shape!r}"
            )
        if payload["dtype"] != tensor_payload.dtype:
            raise ValueError(
                "EVSSM safe dict dtype does not match tensor.dtype: "
                f"{payload['dtype']!r} != {tensor_payload.dtype!r}"
            )
        timestamp = payload["timestamp"]
        if type(timestamp) is not int or timestamp != source_index:
            raise ValueError(
                "EVSSM safe dict timestamp must be the exact integer source_index "
                f"{source_index}, got {timestamp!r}"
            )
        container_schema = "tracker_safe_tensor_v1"
        container_timestamp = timestamp
    else:
        raise ValueError(
            "EVSSM artifact must be a bare torch.Tensor or exact tracker safe dict, "
            f"got {type(payload).__name__}"
        )
    if tensor_payload.ndim != 4 or tuple(tensor_payload.shape[:2]) != (1, 3):
        raise ValueError(
            "EVSSM tensor must have shape [1,3,H,W], got "
            f"{tuple(tensor_payload.shape)}"
        )
    if not tensor_payload.dtype.is_floating_point:
        raise ValueError(
            f"EVSSM tensor must have floating dtype, got {tensor_payload.dtype}"
        )
    if tuple(tensor_payload.shape[-2:]) != (height, width):
        raise ValueError(
            "preprocessed EVSSM tensor must already match ReSplat output size "
            f"[{height},{width}], got {tuple(tensor_payload.shape[-2:])}"
        )
    tensor = tensor_payload.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("EVSSM tensor contains non-finite values")
    minimum = float(tensor.min().item())
    maximum = float(tensor.max().item())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"EVSSM tensor values must be in [0,1], got [{minimum},{maximum}]"
        )
    image = tensor[0].permute(1, 2, 0).numpy()
    return image, {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(tensor_payload.shape),
        "dtype": str(tensor_payload.dtype),
        "value_range": [minimum, maximum],
        "weights_only": True,
        "preprocessed_upstream": True,
        "preprocessing": "TUM undistort then tracker resize/crop",
        "container_schema": container_schema,
        "container": {
            "schema": container_schema,
            "timestamp": container_timestamp,
            "source_index_bound": container_schema == "tracker_safe_tensor_v1",
        },
    }


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV (cv2) is required for raw TUM fallback frames") from error
    return cv2


def _process_raw(
    raw: np.ndarray,
    intrinsics: Mapping[str, float],
    distortion: np.ndarray,
    *,
    width: int,
    height: int,
    cv2_module: Any | None,
) -> np.ndarray:
    cv2 = cv2_module if cv2_module is not None else _import_cv2()
    undistorted = cv2.undistort(raw, _camera_matrix(intrinsics), distortion)
    interpolation = (
        cv2.INTER_AREA
        if width <= raw.shape[1] and height <= raw.shape[0]
        else cv2.INTER_LINEAR
    )
    return cv2.resize(undistorted, (width, height), interpolation=interpolation)


def _uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"output image must be HWC RGB, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("output image contains non-finite values")
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0)
    return np.asarray(np.clip(array, 0, 255), dtype=np.uint8)


def _atomic_save_png(path: Path, image: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.png"
    try:
        with temporary.open("xb") as handle:
            Image.fromarray(_uint8_rgb(image), mode="RGB").save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize(
    *,
    frames_csv: Path | str,
    output_dir: Path | str,
    evssm_checkpoint: Path | str,
    indices: Sequence[int] | None = None,
    indices_file: Path | str | None = None,
    evssm_tensor_dir: Path | str | None = None,
    width: int = 512,
    height: int = 384,
    distortion: Sequence[float] | None = None,
    _cv2_module: Any | None = None,
) -> Path:
    """Create one immutable ReSplat image bundle and return its manifest."""

    if (indices is None) == (indices_file is None):
        raise ValueError("provide exactly one of indices or indices_file")
    selected = (
        _validate_indices(indices if indices is not None else (), source_label="indices")
        if indices_file is None
        else parse_indices_file(indices_file)
    )
    if width <= 0 or height <= 0:
        raise ValueError("output width and height must be positive")

    csv_path = Path(frames_csv).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    checkpoint = Path(evssm_checkpoint).expanduser().resolve()
    tensor_root = (
        None
        if evssm_tensor_dir is None
        else Path(evssm_tensor_dir).expanduser().resolve()
    )
    for label, path in (("EVSSM tensor directory", tensor_root),):
        if path is not None and not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")
    if tensor_root is not None:
        lowered = tensor_root.as_posix().lower()
        marker = next(
            (item for item in FORBIDDEN_BACKEND_MARKERS if item in lowered), None
        )
        if marker is not None:
            raise ValueError(f"forbidden {marker!r} EVSSM tensor directory")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {destination}")

    checkpoint_metadata = _checkpoint_record(checkpoint)
    rows, csv_order = _load_rows(csv_path)
    missing = [index for index in selected if index not in rows]
    if missing:
        raise ValueError(f"requested indices are absent from TUM source CSV: {missing}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        records: list[dict[str, Any]] = []
        provider_counts = {"raw_undistorted": 0, "official_unblur_evssm": 0}
        canonical_camera: dict[str, Any] | None = None
        for index in selected:
            row = rows[index]
            raw_path = _resolve_rgb_path(csv_path, row, index)
            raw_sha256 = sha256_file(raw_path)
            raw = _load_raw_rgb(raw_path)
            raw_height, raw_width = raw.shape[:2]
            intrinsics = _intrinsics_for_row(row, index)
            distortion_array, distortion_named = _distortion_for_row(
                row, distortion, index
            )
            scale_x = width / raw_width
            scale_y = height / raw_height
            output_intrinsics = _scaled_intrinsics(intrinsics, scale_x, scale_y)
            frame_camera = {
                "model": "PINHOLE",
                "width": width,
                "height": height,
                "K": _camera_matrix(output_intrinsics).tolist(),
                **output_intrinsics,
            }
            if canonical_camera is None:
                canonical_camera = frame_camera
            elif not np.allclose(
                np.asarray(canonical_camera["K"], dtype=np.float64),
                np.asarray(frame_camera["K"], dtype=np.float64),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "selected TUM rows do not share one output camera K; official "
                    "ReSplat export requires a single mapping camera"
                )

            tensor_path = None if tensor_root is None else tensor_root / f"{index}.pt"
            if tensor_path is not None and tensor_path.is_file():
                pixels, tensor_metadata = _load_evssm_tensor(
                    tensor_path,
                    width=width,
                    height=height,
                    source_index=index,
                )
                provider = "official_unblur_evssm"
                preprocessing = {
                    "performed_by_materializer": False,
                    "status": "preprocessed_upstream",
                    "operations": ["TUM undistort", "tracker resize/crop"],
                    "second_undistort_forbidden": True,
                }
            else:
                pixels = _process_raw(
                    raw,
                    intrinsics,
                    distortion_array,
                    width=width,
                    height=height,
                    cv2_module=_cv2_module,
                )
                tensor_metadata = {
                    "path": None,
                    "sha256": None,
                    "shape": None,
                    "dtype": None,
                    "weights_only": None,
                    "preprocessed_upstream": False,
                    "container_schema": None,
                    "container": None,
                }
                provider = "raw_undistorted"
                preprocessing = {
                    "performed_by_materializer": True,
                    "status": "materialized_from_raw",
                    "operations": ["cv2.undistort", "cv2.resize"],
                    "second_undistort_forbidden": False,
                }
            provider_counts[provider] += 1

            relative_output = Path("images") / f"{index:06d}.png"
            staged_output = staging / relative_output
            _atomic_save_png(staged_output, pixels)
            output_sha256 = sha256_file(staged_output)
            final_output = destination / relative_output
            record = {
                "schema": SCHEMA,
                "source_index": index,
                "timestamp": (
                    _finite_float(row["timestamp"], f"source index {index} timestamp")
                    if str(row.get("timestamp", "")).strip()
                    else None
                ),
                "provider": provider,
                "raw": {
                    "path": str(raw_path),
                    "sha256": raw_sha256,
                    "width": raw_width,
                    "height": raw_height,
                },
                "raw_sha256": raw_sha256,
                "tensor": tensor_metadata,
                "tensor_sha256": tensor_metadata["sha256"],
                "output": {
                    "path": str(final_output),
                    "relative_path": relative_output.as_posix(),
                    "sha256": output_sha256,
                    "width": width,
                    "height": height,
                },
                "png_sha256": output_sha256,
                "intrinsics": {
                    "raw": intrinsics,
                    "output": output_intrinsics,
                    "scale": {"x": scale_x, "y": scale_y},
                },
                "camera_reference": "#/camera",
                "distortion": {
                    "model": "opencv_radial_tangential",
                    "order": list(DISTORTION_KEYS),
                    "coefficients": distortion_named,
                    "vector": distortion_array.tolist(),
                },
                "preprocessing": preprocessing,
                "evssm_checkpoint": dict(checkpoint_metadata),
                "evssm_checkpoint_path": checkpoint_metadata["path"],
                "evssm_checkpoint_sha256": checkpoint_metadata["sha256"],
            }
            records.append(record)

        assert canonical_camera is not None
        manifest = {
            "schema": SCHEMA,
            "artifact_class": "official_cvg_resplat_inputs",
            "frames_csv": str(csv_path),
            "frames_csv_sha256": sha256_file(csv_path),
            "selection": {
                "source_indices": selected,
                "count": len(selected),
                "csv_stream_order": csv_order,
            },
            "output": {
                "directory": str(destination),
                "width": width,
                "height": height,
                "encoding": "rgb_uint8_png",
            },
            "provider_counts": provider_counts,
            "camera": canonical_camera,
            "evssm_checkpoint": dict(checkpoint_metadata),
            "forbidden_backends": list(FORBIDDEN_BACKEND_MARKERS),
            "frames": records,
        }
        _atomic_write_json(staging / "manifest.json", manifest)
        try:
            os.rename(staging, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite concurrently-created output: {destination}"
            ) from error
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evssm-checkpoint", type=Path, required=True)
    parser.add_argument("--evssm-tensor-dir", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--indices", nargs="+", help="source indices (space/comma separated)"
    )
    selection.add_argument("--indices-file", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument(
        "--distortion",
        nargs=5,
        type=float,
        metavar=("K1", "K2", "P1", "P2", "K3"),
        help="global TUM distortion override; otherwise read CSV columns",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chosen_indices = None if args.indices is None else parse_indices(args.indices)
    manifest = materialize(
        frames_csv=args.frames_csv,
        output_dir=args.output_dir,
        evssm_checkpoint=args.evssm_checkpoint,
        indices=chosen_indices,
        indices_file=args.indices_file,
        evssm_tensor_dir=args.evssm_tensor_dir,
        width=args.width,
        height=args.height,
        distortion=args.distortion,
    )
    print(json.dumps({"manifest": str(manifest), "sha256": sha256_file(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
