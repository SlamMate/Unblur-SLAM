#!/usr/bin/env python3
"""Materialize a gap-free official TURTLE stream for TUM/ReSplat.

The formal fr2_xyz smoke consumes source frames 0..2764 in strict timestamp
order.  Every source frame advances the upstream TURTLE K/V state exactly
once, while only the fixed 42 DROID keyframes are written as PNGs.  This is
important: feeding only sparse keyframes would not reproduce the online
``stream_every_frame`` history used by Unblur-SLAM.

No pose, depth or ground-truth image is supplied to TURTLE.  The source CSV is
used only to bind source indices/timestamps/raw RGB paths and camera metadata.
The official checkout, architecture, configuration and GoPro checkpoint are
content-addressed by :mod:`src.turtle_backend` before CUDA inference starts.
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
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence
import uuid

import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.turtle_backend import (  # noqa: E402
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    TURTLE_CACHE_CONTRACT,
    TurtleStreamingBackend,
    build_turtle_model,
    sha256_file,
    validate_turtle_artifacts,
)


SCHEMA = "unblur_slam.turtle_stream_materialization.v1"
OFFICIAL_TURTLE_ORIGIN = "https://github.com/Ascend-Research/Turtle.git"
DEFAULT_TURTLE_REPO = Path("/srv/szha0669/unblur-slam/external/TURTLE")
DEFAULT_TURTLE_CONFIG = DEFAULT_TURTLE_REPO / "options/Turtle_Deblur_Gopro.yml"
DEFAULT_TURTLE_CHECKPOINT = Path(
    "/srv/szha0669/unblur-slam/pretrained/turtle/GoPro_Deblur.pth"
)
FR2_XYZ_DROID_KEYFRAMES = (
    0,
    9,
    15,
    49,
    58,
    72,
    89,
    109,
    125,
    166,
    220,
    319,
    374,
    407,
    435,
    470,
    483,
    523,
    568,
    704,
    750,
    789,
    827,
    926,
    1004,
    1160,
    1251,
    1342,
    1409,
    1460,
    1553,
    1692,
    1795,
    1889,
    1978,
    2055,
    2206,
    2282,
    2358,
    2425,
    2590,
    2764,
)
FR2_XYZ_DISTORTION = (0.2312, -0.7849, -0.0033, -0.0001, 0.9172)
FR2_XYZ_HEIGHT_EDGE = 8
FR2_XYZ_WIDTH_EDGE = 8
# The GoPro YAML selects non-caching Reduced/Channel attention for the three
# encoder slots, then two latent FHR and three decoder CHM cache slots.  The
# official model therefore returns an exact sparse eight-slot layout rather
# than eight non-null tensors.  Pinning the mask catches both an architecture
# drift and an accidentally substituted cache implementation.
OFFICIAL_GOPRO_CACHE_NON_NULL_MASK = (
    False,
    False,
    False,
    True,
    True,
    True,
    True,
    True,
)


@dataclass(frozen=True)
class SourceFrame:
    source_index: int
    timestamp: float
    rgb_path: Path
    intrinsics: tuple[float, float, float, float]
    pose_source: str


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _false_flag(value: object, label: str) -> None:
    normalized = str(value).strip().lower()
    if normalized not in {"false", "0", "no"}:
        raise ValueError(f"{label} must explicitly be false, got {value!r}")


def _resolve_rgb(csv_path: Path, value: object, label: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = csv_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_contiguous_source_frames(
    frames_csv: Path | str,
    *,
    start_index: int,
    end_index: int,
) -> tuple[list[SourceFrame], dict[str, Any]]:
    """Bind one explicit, gap-free, non-GT source-index interval."""

    source = Path(frames_csv).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"frames CSV does not exist: {source}")
    if start_index < 0 or end_index < start_index:
        raise ValueError("source interval must satisfy 0 <= start_index <= end_index")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {
            "index",
            "timestamp",
            "fx",
            "fy",
            "cx",
            "cy",
            "pose_source",
            "uses_ground_truth_pose",
        }
        if not ({"rgb_path", "frame"} & fields):
            required.add("rgb_path")
        missing = required - fields
        if missing:
            raise ValueError(f"frames CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    by_index: dict[int, tuple[int, Mapping[str, str]]] = {}
    for csv_row, row in enumerate(rows, start=2):
        try:
            index = int(str(row.get("index", "")).strip())
        except ValueError as error:
            raise ValueError(f"invalid source index at CSV row {csv_row}") from error
        if index < 0 or index in by_index:
            raise ValueError(f"invalid or duplicate source index {index}")
        by_index[index] = (csv_row, row)

    wanted = list(range(start_index, end_index + 1))
    missing_indices = [index for index in wanted if index not in by_index]
    if missing_indices:
        raise ValueError(
            "continuous TURTLE stream has missing source indices: "
            f"{missing_indices[:16]}"
        )

    result: list[SourceFrame] = []
    prior_timestamp: Optional[float] = None
    pose_sources: set[str] = set()
    for expected_step, index in enumerate(wanted):
        csv_row, row = by_index[index]
        timestamp = _finite_float(row["timestamp"], f"CSV row {csv_row} timestamp")
        if prior_timestamp is not None and timestamp <= prior_timestamp:
            raise ValueError(
                "continuous TURTLE timestamps must be strictly increasing: "
                f"source_index={index}, {timestamp} <= {prior_timestamp}"
            )
        prior_timestamp = timestamp
        _false_flag(
            row["uses_ground_truth_pose"],
            f"CSV row {csv_row} uses_ground_truth_pose",
        )
        pose_source = str(row["pose_source"]).strip()
        if not pose_source:
            raise ValueError(f"CSV row {csv_row} pose_source is empty")
        pose_sources.add(pose_source)
        rgb_value = str(row.get("rgb_path", "")).strip() or str(
            row.get("frame", "")
        ).strip()
        intrinsics = tuple(
            _finite_float(row[key], f"CSV row {csv_row} {key}")
            for key in ("fx", "fy", "cx", "cy")
        )
        if intrinsics[0] <= 0.0 or intrinsics[1] <= 0.0:
            raise ValueError(f"CSV row {csv_row} focal lengths must be positive")
        result.append(
            SourceFrame(
                source_index=index,
                timestamp=timestamp,
                rgb_path=_resolve_rgb(
                    source, rgb_value, f"CSV row {csv_row} raw RGB"
                ),
                intrinsics=intrinsics,
                pose_source=pose_source,
            )
        )
        if expected_step != index - start_index:  # pragma: no cover - defensive
            raise AssertionError("source-index/step binding drifted")
    if len(pose_sources) != 1:
        raise ValueError(
            f"source interval contains multiple pose_source values: {sorted(pose_sources)}"
        )
    return result, {
        "path": str(source),
        "sha256": sha256_file(source),
        "pose_source_declared_but_not_consumed": next(iter(pose_sources)),
        "uses_ground_truth_pose": False,
        "poses_consumed_by_turtle": False,
        "depth_consumed_by_turtle": False,
        "ground_truth_images_consumed_by_turtle": False,
    }


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot audit official TURTLE checkout: {repo}") from error
    return completed.stdout.strip()


def official_artifact_record(artifacts: Any, model: torch.nn.Module) -> dict[str, Any]:
    """Turn already validated artifacts into a fail-closed audit record."""

    origin = _git(artifacts.repo, "remote", "get-url", "origin")
    if origin != OFFICIAL_TURTLE_ORIGIN:
        raise ValueError(
            f"TURTLE origin mismatch: expected {OFFICIAL_TURTLE_ORIGIN}, got {origin}"
        )
    status = _git(artifacts.repo, "status", "--porcelain")
    if status:
        raise ValueError("official TURTLE checkout must be clean for formal materialization")
    expected = {
        "commit": (artifacts.commit, PINNED_TURTLE_COMMIT),
        "architecture_sha256": (
            artifacts.architecture_sha256,
            PINNED_TURTLE_ARCH_SHA256,
        ),
        "config_sha256": (artifacts.config_sha256, PINNED_TURTLE_CONFIG_SHA256),
        "checkpoint_sha256": (
            artifacts.checkpoint_sha256,
            PINNED_TURTLE_CHECKPOINT_SHA256,
        ),
    }
    mismatches = {
        key: (actual, wanted)
        for key, (actual, wanted) in expected.items()
        if str(actual).lower() != wanted
    }
    if mismatches:
        raise ValueError(f"official TURTLE artifact mismatch: {mismatches}")
    metadata = dict(artifacts.checkpoint_metadata)
    if metadata.get("kind") != "official_gopro":
        raise ValueError("formal smoke requires the official GoPro TURTLE checkpoint")
    return {
        "implementation": "official_ascend_research_turtle",
        "repo": {
            "path": str(artifacts.repo),
            "origin": origin,
            "commit": artifacts.commit,
            "clean": True,
        },
        "architecture": {
            "path": str(artifacts.architecture),
            "sha256": artifacts.architecture_sha256,
        },
        "config": {
            "path": str(artifacts.config),
            "sha256": artifacts.config_sha256,
            "num_frames_tocache": int(artifacts.options["num_frames_tocache"]),
            "use_both_input": bool(artifacts.options["use_both_input"]),
        },
        "checkpoint": {
            "path": str(artifacts.checkpoint),
            "sha256": artifacts.checkpoint_sha256,
            "metadata": metadata,
        },
        "model": {
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "training": bool(model.training),
        },
        "cache_contract": TURTLE_CACHE_CONTRACT,
        "call_contract": {
            "one_model_call_per_source_frame": True,
            "input": "B=1,T=2,C=3,H,W; [previous,current]",
            "first_pair": "[frame_0,frame_0]",
            "returns": "restored,k_cache[8],v_cache[8]",
            "persistent_kv_forwarded": True,
        },
    }


def _validate_injected_turtle_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the private CPU-test seam without weakening the production CLI."""

    result = json.loads(json.dumps(record))
    try:
        expected = {
            "commit": result["repo"]["commit"],
            "architecture": result["architecture"]["sha256"],
            "config": result["config"]["sha256"],
            "checkpoint": result["checkpoint"]["sha256"],
            "kind": result["checkpoint"]["metadata"]["kind"],
            "cache": result["cache_contract"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError("injected TURTLE audit record is incomplete") from error
    wanted = {
        "commit": PINNED_TURTLE_COMMIT,
        "architecture": PINNED_TURTLE_ARCH_SHA256,
        "config": PINNED_TURTLE_CONFIG_SHA256,
        "checkpoint": PINNED_TURTLE_CHECKPOINT_SHA256,
        "kind": "official_gopro",
        "cache": TURTLE_CACHE_CONTRACT,
    }
    if expected != wanted:
        raise ValueError(f"injected TURTLE audit record is not official GoPro: {expected}")
    return result


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment failure
        raise RuntimeError("OpenCV is required for TUM undistortion") from error
    return cv2


def _preprocess_raw(
    frame: SourceFrame,
    *,
    width: int,
    height: int,
    width_edge: int,
    height_edge: int,
    distortion: np.ndarray,
    cv2_module: Any,
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int]]:
    cv2 = cv2_module
    raw_bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if raw_bgr is None or raw_bgr.ndim != 3 or raw_bgr.shape[2] != 3:
        raise ValueError(f"cannot decode raw RGB image: {frame.rgb_path}")
    raw_height, raw_width = map(int, raw_bgr.shape[:2])
    fx, fy, cx, cy = frame.intrinsics
    camera = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    undistorted = cv2.undistort(raw_bgr, camera, distortion)
    resize_width = width + 2 * width_edge
    resize_height = height + 2 * height_edge
    resized_with_edge = cv2.resize(
        undistorted,
        (resize_width, resize_height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = resized_with_edge[
        height_edge : height_edge + height,
        width_edge : width_edge + width,
    ]
    if tuple(resized.shape[:2]) != (height, width):
        raise RuntimeError("TUM resize/crop did not produce the requested tracker shape")
    rgb_u8 = np.ascontiguousarray(resized[:, :, ::-1], dtype=np.uint8)
    tensor = (
        torch.from_numpy(rgb_u8)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    return tensor, rgb_u8, (raw_width, raw_height)


def _rgb_u8(tensor: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(tensor) or tuple(tensor.shape[:2]) != (1, 3):
        raise ValueError(f"TURTLE output must be 1x3xHxW, got {tuple(tensor.shape)}")
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)[0]
        .mul(255.0)
        .round()
        .to(device="cpu", dtype=torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    return array


def sha256_pixels(array: np.ndarray) -> str:
    pixels = np.ascontiguousarray(array, dtype=np.uint8)
    return hashlib.sha256(pixels.tobytes(order="C")).hexdigest()


def _atomic_save_png(path: Path, pixels: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.png"
    try:
        with temporary.open("xb") as handle:
            Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(
                handle, format="PNG"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite JSON: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cache_audit(cache: Any, label: str) -> dict[str, Any]:
    if not isinstance(cache, (tuple, list)) or len(cache) != 8:
        raise RuntimeError(f"official TURTLE {label} cache must contain exactly eight slots")
    invalid = [
        index
        for index, value in enumerate(cache)
        if value is not None and not torch.is_tensor(value)
    ]
    if invalid:
        raise RuntimeError(
            f"official TURTLE {label} cache slots must be Tensor or None; invalid={invalid}"
        )
    mask = tuple(torch.is_tensor(value) for value in cache)
    if mask != OFFICIAL_GOPRO_CACHE_NON_NULL_MASK:
        raise RuntimeError(
            f"official GoPro TURTLE {label} cache mask mismatch: "
            f"expected {list(OFFICIAL_GOPRO_CACHE_NON_NULL_MASK)}, got {list(mask)}"
        )
    return {
        "slot_count": len(cache),
        "non_null_count": int(sum(mask)),
        "non_null_mask": list(mask),
        "slot_types": ["tensor" if present else "none" for present in mask],
    }


def materialize_tum_turtle_stream(
    *,
    frames_csv: Path | str,
    output_dir: Path | str,
    turtle_repo: Path | str = DEFAULT_TURTLE_REPO,
    turtle_config: Path | str = DEFAULT_TURTLE_CONFIG,
    turtle_checkpoint: Path | str = DEFAULT_TURTLE_CHECKPOINT,
    start_index: int,
    end_index: int,
    emitted_source_indices: Sequence[int],
    width: int = 512,
    height: int = 384,
    width_edge: int = 0,
    height_edge: int = 0,
    distortion: Sequence[float] = FR2_XYZ_DISTORTION,
    device: Any = "cuda:0",
    progress_every: int = 100,
    _backend: Any = None,
    _turtle_record: Optional[Mapping[str, Any]] = None,
    _cv2_module: Any = None,
) -> Path:
    """Run one audited causal stream and return the final manifest path."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {destination}"
        )
    if width <= 0 or height <= 0 or width % 8 or height % 8:
        raise ValueError("TURTLE output dimensions must be positive multiples of eight")
    if width_edge < 0 or height_edge < 0:
        raise ValueError("TUM crop edges must be non-negative")
    if len(distortion) != 5 or not all(math.isfinite(float(value)) for value in distortion):
        raise ValueError("TUM distortion must contain five finite coefficients")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    emitted = [int(value) for value in emitted_source_indices]
    if not emitted or len(set(emitted)) != len(emitted):
        raise ValueError("emitted_source_indices must be non-empty and unique")
    if emitted != sorted(emitted):
        raise ValueError("emitted_source_indices must be sorted")
    if emitted[0] < start_index or emitted[-1] > end_index:
        raise ValueError("every emitted source index must lie in the processed interval")
    emitted_set = set(emitted)

    frames, source_record = load_contiguous_source_frames(
        frames_csv, start_index=start_index, end_index=end_index
    )
    expected_count = end_index - start_index + 1
    if len(frames) != expected_count:
        raise AssertionError("continuous source loader returned the wrong frame count")

    actual_device = torch.device(device)
    if _backend is None:
        artifacts = validate_turtle_artifacts(
            {
                "turtle_repo": str(turtle_repo),
                "turtle_config": str(turtle_config),
                "turtle_checkpoint": str(turtle_checkpoint),
                "turtle_repo_commit": PINNED_TURTLE_COMMIT,
                "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
                "turtle_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
            },
            load_weights=False,
        )
        model = build_turtle_model(artifacts, device=actual_device)
        backend = TurtleStreamingBackend(model, device=actual_device)
        turtle_record = official_artifact_record(artifacts, model)
    else:
        if _turtle_record is None:
            raise ValueError("private backend injection requires an audited TURTLE record")
        backend = _backend
        turtle_record = _validate_injected_turtle_record(_turtle_record)

    cv2 = _cv2_module if _cv2_module is not None else _import_cv2()
    distortion_array = np.asarray(distortion, dtype=np.float64)
    torch.manual_seed(0)
    if actual_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(actual_device)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        backend.reset()
        reset_events = [
            {
                "before_source_index": start_index,
                "reason": "explicit_sequence_start",
                "reset_ordinal": 1,
            }
        ]
        steps: list[dict[str, Any]] = []
        output_frames: list[dict[str, Any]] = []
        latencies: list[float] = []
        canonical_raw_size: Optional[tuple[int, int]] = None
        canonical_intrinsics: Optional[tuple[float, float, float, float]] = None
        stream_started = time.perf_counter()

        for step_index, frame in enumerate(frames):
            raw_sha256 = sha256_file(frame.rgb_path)
            model_input, input_pixels, raw_size = _preprocess_raw(
                frame,
                width=width,
                height=height,
                width_edge=width_edge,
                height_edge=height_edge,
                distortion=distortion_array,
                cv2_module=cv2,
            )
            if canonical_raw_size is None:
                canonical_raw_size = raw_size
                canonical_intrinsics = frame.intrinsics
            elif raw_size != canonical_raw_size or not np.allclose(
                np.asarray(frame.intrinsics),
                np.asarray(canonical_intrinsics),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError("processed stream must use one raw camera and resolution")

            had_cache_before = backend.k_cache is not None and backend.v_cache is not None
            expected_cache_before = step_index > 0
            if had_cache_before != expected_cache_before:
                raise RuntimeError(
                    f"unexpected cache state before source_index={frame.source_index}"
                )
            _synchronize(actual_device)
            started = time.perf_counter()
            restored = backend.step(model_input, timestamp=frame.timestamp)
            _synchronize(actual_device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            output_pixels = _rgb_u8(restored)
            input_pixels_sha256 = sha256_pixels(input_pixels)
            output_pixels_sha256 = sha256_pixels(output_pixels)
            k_audit = _cache_audit(backend.k_cache, "key")
            v_audit = _cache_audit(backend.v_cache, "value")
            if k_audit["non_null_mask"] != v_audit["non_null_mask"]:
                raise RuntimeError("official TURTLE key/value sparse cache masks disagree")
            expected_updates = step_index + 1
            state = dict(backend.state_info())
            if (
                int(state["frames_seen"]) != expected_updates
                or int(state["cache_updates"]) != expected_updates
                or int(state["reset_count"]) != 1
                or state["last_timestamp"] != frame.timestamp
                or not bool(state["has_cache"])
            ):
                raise RuntimeError(
                    f"TURTLE stream-state audit failed at source_index={frame.source_index}: "
                    f"{state}"
                )

            emitted_now = frame.source_index in emitted_set
            png_record: Optional[dict[str, Any]] = None
            if emitted_now:
                relative_output = Path("images") / f"{frame.source_index:06d}.png"
                staged_output = staging / relative_output
                _atomic_save_png(staged_output, output_pixels)
                png_sha256 = sha256_file(staged_output)
                final_output = destination / relative_output
                png_record = {
                    "path": str(final_output),
                    "relative_path": relative_output.as_posix(),
                    "sha256": png_sha256,
                    "pixel_sha256": output_pixels_sha256,
                    "width": width,
                    "height": height,
                    "encoding": "rgb_uint8_png",
                }
                output_frames.append(
                    {
                        "schema": SCHEMA,
                        "provider": "official_turtle_gopro_streaming",
                        "source_index": frame.source_index,
                        "timestamp": frame.timestamp,
                        "step_index": step_index,
                        "input": {
                            "path": str(frame.rgb_path),
                            "sha256": raw_sha256,
                            "preprocessed_pixel_sha256": input_pixels_sha256,
                        },
                        "output": png_record,
                        "output_sha256": png_sha256,
                        "camera_reference": "#/camera",
                        "stream_audit": {
                            "cache_present_before": had_cache_before,
                            "cache_present_after": True,
                            "cache_update_ordinal": expected_updates,
                            "reset_count": int(state["reset_count"]),
                        },
                    }
                )

            steps.append(
                {
                    "step_index": step_index,
                    "source_index": frame.source_index,
                    "timestamp": frame.timestamp,
                    "input_path": str(frame.rgb_path),
                    "input_file_sha256": raw_sha256,
                    "input_rgb_u8_pixel_sha256": input_pixels_sha256,
                    "output_rgb_u8_pixel_sha256": output_pixels_sha256,
                    "hash_encoding": "contiguous_rgb_uint8_hwc_bytes",
                    "emitted_png": emitted_now,
                    "emitted_png_sha256": (
                        None if png_record is None else png_record["sha256"]
                    ),
                    "cache_present_before": had_cache_before,
                    "cache_present_after": True,
                    "k_cache_slots_after": k_audit["slot_count"],
                    "v_cache_slots_after": v_audit["slot_count"],
                    "k_cache_non_null_count_after": k_audit["non_null_count"],
                    "v_cache_non_null_count_after": v_audit["non_null_count"],
                    "k_cache_non_null_mask_after": k_audit["non_null_mask"],
                    "v_cache_non_null_mask_after": v_audit["non_null_mask"],
                    "cache_update_ordinal": expected_updates,
                    "reset_count": int(state["reset_count"]),
                    "latency_ms": latency_ms,
                }
            )
            if progress_every and (
                expected_updates % progress_every == 0 or expected_updates == expected_count
            ):
                print(
                    json.dumps(
                        {
                            "event": "turtle_stream_progress",
                            "processed": expected_updates,
                            "total": expected_count,
                            "source_index": frame.source_index,
                            "emitted": len(output_frames),
                        }
                    ),
                    flush=True,
                )

        wall_seconds = time.perf_counter() - stream_started
        final_state = dict(backend.state_info())
        if [record["source_index"] for record in output_frames] != emitted:
            raise RuntimeError("emitted PNG order/count differs from the declared selection")
        if canonical_raw_size is None or canonical_intrinsics is None:
            raise AssertionError("empty stream")
        raw_width, raw_height = canonical_raw_size
        fx, fy, cx, cy = canonical_intrinsics
        resize_width = width + 2 * width_edge
        resize_height = height + 2 * height_edge
        scale_x, scale_y = resize_width / raw_width, resize_height / raw_height
        output_k = [
            [fx * scale_x, 0.0, cx * scale_x - width_edge],
            [0.0, fy * scale_y, cy * scale_y - height_edge],
            [0.0, 0.0, 1.0],
        ]
        peak_memory = (
            int(torch.cuda.max_memory_allocated(actual_device))
            if actual_device.type == "cuda"
            else None
        )
        cuda_device_name = (
            torch.cuda.get_device_name(actual_device)
            if actual_device.type == "cuda"
            else None
        )
        manifest = {
            "schema": SCHEMA,
            "artifact_class": "official_turtle_stream_inputs_for_cvg_resplat",
            "source": source_record,
            "camera": {
                "model": "PINHOLE",
                "width": width,
                "height": height,
                "K": output_k,
                "fx": output_k[0][0],
                "fy": output_k[1][1],
                "cx": output_k[0][2],
                "cy": output_k[1][2],
                "raw_width": raw_width,
                "raw_height": raw_height,
                "resize_before_crop_width": resize_width,
                "resize_before_crop_height": resize_height,
                "crop_edges": {
                    "left": width_edge,
                    "right": width_edge,
                    "top": height_edge,
                    "bottom": height_edge,
                },
                "distortion": {
                    "model": "opencv_radial_tangential",
                    "order": ["k1", "k2", "p1", "p2", "k3"],
                    "vector": [float(value) for value in distortion_array],
                },
                "preprocessing": [
                    "cv2.imread_color",
                    "cv2.undistort_same_K",
                    f"cv2.resize_inter_linear_{resize_width}x{resize_height}",
                    (
                        f"crop_l{width_edge}_r{width_edge}_"
                        f"t{height_edge}_b{height_edge}"
                    ),
                    "bgr_to_rgb",
                    "uint8_to_float32_div_255",
                ],
            },
            "turtle": turtle_record,
            "stream": {
                "processed_range": {
                    "start_source_index": start_index,
                    "end_source_index": end_index,
                    "inclusive": True,
                },
                "processed_source_indices": list(range(start_index, end_index + 1)),
                "processed_count": expected_count,
                "step_count": int(final_state["frames_seen"]),
                "cache_updates": int(final_state["cache_updates"]),
                "reset_count": int(final_state["reset_count"]),
                "reset_events": reset_events,
                "strictly_increasing_source_indices": True,
                "strictly_increasing_timestamps": True,
                "gaps_skipped": False,
                "first_pair": "self",
                "one_step_per_source_frame": True,
                "persistent_kv": True,
                "k_cache_slots": 8,
                "v_cache_slots": 8,
                "k_cache_non_null_count": int(
                    sum(OFFICIAL_GOPRO_CACHE_NON_NULL_MASK)
                ),
                "v_cache_non_null_count": int(
                    sum(OFFICIAL_GOPRO_CACHE_NON_NULL_MASK)
                ),
                "official_gopro_cache_non_null_mask": list(
                    OFFICIAL_GOPRO_CACHE_NON_NULL_MASK
                ),
                "cache_contract": TURTLE_CACHE_CONTRACT,
                "steps": steps,
            },
            "selection": {
                "emitted_source_indices": emitted,
                "emitted_count": len(emitted),
                "processed_but_not_emitted_count": expected_count - len(emitted),
            },
            "output": {
                "directory": str(destination),
                "encoding": "rgb_uint8_png",
                "width": width,
                "height": height,
            },
            "performance": {
                "device": str(actual_device),
                "cuda_device_name": cuda_device_name,
                "seed": 0,
                "latency_scope": "official TURTLE step including device synchronization",
                "latency_ms": {
                    "mean": float(np.mean(latencies)),
                    "median": float(np.median(latencies)),
                    "p95": float(np.percentile(latencies, 95)),
                    "max": float(max(latencies)),
                },
                "stream_wall_seconds": wall_seconds,
                "peak_cuda_memory_allocated_bytes": peak_memory,
            },
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "numpy": np.__version__,
                "pillow": PIL_VERSION,
                "opencv": str(cv2.__version__),
                "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "safety": {
                "ground_truth_images_used": False,
                "ground_truth_poses_used": False,
                "depth_used": False,
                "custom_causal_evssm_used": False,
                "sliding_window_recomputation_used": False,
            },
            "frames": output_frames,
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
    parser.add_argument("--turtle-repo", type=Path, default=DEFAULT_TURTLE_REPO)
    parser.add_argument("--turtle-config", type=Path, default=DEFAULT_TURTLE_CONFIG)
    parser.add_argument(
        "--turtle-checkpoint", type=Path, default=DEFAULT_TURTLE_CHECKPOINT
    )
    parser.add_argument(
        "--profile",
        choices=("fr2_xyz_42kf_0_2764",),
        required=True,
        help="Pinned formal stream/selection contract",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.profile != "fr2_xyz_42kf_0_2764":  # pragma: no cover - argparse guards
        raise ValueError(f"unsupported profile: {args.profile}")
    manifest = materialize_tum_turtle_stream(
        frames_csv=args.frames_csv,
        output_dir=args.output_dir,
        turtle_repo=args.turtle_repo,
        turtle_config=args.turtle_config,
        turtle_checkpoint=args.turtle_checkpoint,
        start_index=0,
        end_index=2764,
        emitted_source_indices=FR2_XYZ_DROID_KEYFRAMES,
        width=args.width,
        height=args.height,
        width_edge=FR2_XYZ_WIDTH_EDGE,
        height_edge=FR2_XYZ_HEIGHT_EDGE,
        distortion=FR2_XYZ_DISTORTION,
        device=args.device,
        progress_every=args.progress_every,
    )
    print(json.dumps({"manifest": str(manifest), "sha256": sha256_file(manifest)}))


if __name__ == "__main__":
    main()
