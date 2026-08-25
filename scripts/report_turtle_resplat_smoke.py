#!/usr/bin/env python3
"""CPU-only audit/report for the official TURTLE + official ReSplat smoke.

The frame sets and metric protocol are pre-registered in
``docs/TURTLE_RESPLAT_SMOKE_ACCEPTANCE_ZH.md``.  This script consumes immutable
inference artifacts, never imports either model, never selects frames from
metric values, and refuses to overwrite an existing report directory.

The 42 TUM references are published clear-frame protocol images.  Raw is
therefore an exact-reference control; the frontend section measures
clear-frame preservation, not deblurring quality.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


SCHEMA = "unblur_slam.turtle_resplat_smoke_audit.v1"
TURTLE_SCHEMA = "unblur_slam.turtle_stream_materialization.v1"
RESPLAT_SCHEMA = "unblur_slam.paired_official_resplat_smoke.v1"
EXPECTED_TURTLE_PROVIDER = "official_turtle_gopro_streaming"
EXPECTED_TURTLE_CHECKPOINT_SHA256 = (
    "10334b3e81d0416bcde5ccaca960dc81dbfb5b6d23e53fadaf7896d72b580c82"
)
EXPECTED_RESPLAT_CHECKPOINT_SHA256 = (
    "548993fede0d9536d2d914cbe51e0ebea0ad6f88c898c909e02127d59bb2be9a"
)
EXPECTED_EVSSM_CHECKPOINT_SHA256 = (
    "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
)

CLEAR_42 = (
    0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220, 319, 374, 407,
    435, 470, 483, 523, 568, 704, 750, 789, 827, 926, 1004, 1160,
    1251, 1342, 1409, 1460, 1553, 1692, 1795, 1889, 1978, 2055,
    2206, 2282, 2358, 2425, 2590, 2764,
)
EVSSM_26 = (
    49, 72, 166, 319, 435, 470, 483, 750, 827, 1004, 1160, 1251,
    1342, 1409, 1460, 1692, 1795, 1889, 1978, 2055, 2206, 2282,
    2358, 2425, 2590, 2764,
)
FIXED_FIVE = (49, 483, 1342, 2055, 2764)
FRONTEND_SIZE = (512, 384)
RESPLAT_SIZE = (448, 320)
TRACKER_K = (
    (429.7425, 0.0, 260.2075),
    (0.0, 434.1666666666667, 200.08333333333334),
    (0.0, 0.0, 1.0),
)
FLOW_FB_THRESHOLD_PX = 1.0
FLOW_MIN_VALID_FRACTION = 0.25
REALTIME_30FPS_P95_MS = 1000.0 / 30.0


class AuditError(RuntimeError):
    """Raised when an artifact violates a pre-registered contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def load_json(path: Path | str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    require(source.is_file(), f"missing JSON: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSON: {source}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {source}")
    return value


def sha256_file(path: Path | str) -> str:
    source = Path(path).expanduser().resolve()
    require(source.is_file(), f"missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path | str, size: Optional[tuple[int, int]] = None) -> Image.Image:
    source = Path(path).expanduser().resolve()
    require(source.is_file(), f"missing image: {source}")
    try:
        with Image.open(source) as opened:
            opened.load()
            image = opened.convert("RGB")
    except Exception as error:  # pragma: no cover - decode failure path
        raise AuditError(f"cannot decode RGB image {source}: {error}") from error
    if size is not None:
        require(image.size == size, f"{source}: expected {size}, got {image.size}")
    return image


def rgb_chw(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)


def pixels_sha256(image: Image.Image) -> str:
    array = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _finite_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _finite_median(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def psnr_ssim(reference: Image.Image, candidate: Image.Image) -> dict[str, Any]:
    require(reference.size == candidate.size, "metric image sizes differ")
    reference_array = rgb_chw(reference)
    candidate_array = rgb_chw(candidate)
    mse = float(np.mean((reference_array - candidate_array) ** 2, dtype=np.float64))
    exact = bool(mse == 0.0)
    psnr: float | str = "Infinity" if exact else float(-10.0 * math.log10(mse))
    ssim = float(
        structural_similarity(
            reference_array,
            candidate_array,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
    )
    return {"mse": mse, "psnr_db": psnr, "ssim": ssim, "exact_match": exact}


class LpipsVggCpu:
    """Lazy VGG LPIPS evaluator matching official ReSplat's metric definition."""

    def __init__(self, enabled: bool, batch_size: int = 4) -> None:
        self.enabled = bool(enabled)
        self.batch_size = max(1, int(batch_size))
        self._model: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if not self.enabled or self._model is not None:
            return
        import torch
        from lpips import LPIPS

        torch.set_grad_enabled(False)
        self._torch = torch
        self._model = LPIPS(net="vgg").eval().to(torch.device("cpu"))

    def evaluate(self, pairs: Sequence[tuple[Image.Image, Image.Image]]) -> list[Optional[float]]:
        if not self.enabled:
            return [None] * len(pairs)
        self._load()
        torch = self._torch
        results: list[Optional[float]] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            reference = torch.from_numpy(np.stack([rgb_chw(a) for a, _ in chunk]))
            candidate = torch.from_numpy(np.stack([rgb_chw(b) for _, b in chunk]))
            with torch.inference_mode():
                values = self._model.forward(reference, candidate, normalize=True)
            results.extend(float(value) for value in values[:, 0, 0, 0].cpu().tolist())
        return results


def metric_rows(
    references: Mapping[int, Image.Image],
    candidates: Mapping[int, Image.Image],
    indices: Sequence[int],
    lpips: LpipsVggCpu,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = [int(index) for index in indices]
    require(all(index in references for index in ordered), "reference frame missing")
    require(all(index in candidates for index in ordered), "candidate frame missing")
    rows = []
    pairs = [(references[index], candidates[index]) for index in ordered]
    lpips_values = lpips.evaluate(pairs)
    for index, value in zip(ordered, lpips_values):
        row = {"source_index": index, **psnr_ssim(references[index], candidates[index])}
        row["lpips_vgg"] = value
        rows.append(row)
    numeric_psnr = [
        float(row["psnr_db"])
        for row in rows
        if isinstance(row["psnr_db"], (float, int))
    ]
    all_exact = all(bool(row["exact_match"]) for row in rows)
    summary = {
        "count": len(rows),
        "mean_psnr_db": "Infinity" if all_exact else _finite_mean(numeric_psnr),
        "median_psnr_db": "Infinity" if all_exact else _finite_median(numeric_psnr),
        "mean_ssim": _finite_mean(row["ssim"] for row in rows),
        "median_ssim": _finite_median(row["ssim"] for row in rows),
        "mean_lpips_vgg": _finite_mean(row["lpips_vgg"] for row in rows),
        "median_lpips_vgg": _finite_median(row["lpips_vgg"] for row in rows),
        "exact_match_count": sum(bool(row["exact_match"]) for row in rows),
    }
    return rows, summary


def exact_reference_control_rows(
    references: Mapping[int, Image.Image], indices: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Record identity metrics exactly, without wasting a VGG forward pass."""

    rows = [
        {
            "source_index": int(index),
            "mse": 0.0,
            "psnr_db": "Infinity",
            "ssim": 1.0,
            "exact_match": True,
            "lpips_vgg": 0.0,
        }
        for index in indices
    ]
    require(all(int(index) in references for index in indices), "reference control frame missing")
    return rows, {
        "count": len(rows),
        "mean_psnr_db": "Infinity",
        "median_psnr_db": "Infinity",
        "mean_ssim": 1.0,
        "median_ssim": 1.0,
        "mean_lpips_vgg": 0.0,
        "median_lpips_vgg": 0.0,
        "exact_match_count": len(rows),
    }


def clear_references(clear_pair_dir: Path | str) -> tuple[dict[int, Image.Image], dict[str, str]]:
    root = Path(clear_pair_dir).expanduser().resolve()
    require(root.is_dir(), f"missing clear-GT pair directory: {root}")
    references: dict[int, Image.Image] = {}
    hashes: dict[str, str] = {}
    expected_names = {f"source_{index:06d}_gt_render.png" for index in CLEAR_42}
    actual_names = {path.name for path in root.glob("source_*_gt_render.png") if path.is_file()}
    require(actual_names == expected_names, "clear-GT pair filenames do not exactly match CLEAR_42")
    for index in CLEAR_42:
        path = root / f"source_{index:06d}_gt_render.png"
        pair = load_rgb(path, (1024, 384))
        references[index] = pair.crop((0, 0, 512, 384))
        hashes[path.name] = sha256_file(path)
    return references, hashes


def _turtle_checkpoint_sha(manifest: Mapping[str, Any]) -> Optional[str]:
    turtle = manifest.get("turtle")
    if not isinstance(turtle, Mapping):
        return None
    checkpoint = turtle.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        return str(checkpoint.get("sha256") or "") or None
    return None


def load_turtle_artifact(
    manifest_path: Path | str,
    old_input_manifest: Mapping[str, Any],
    references: Mapping[int, Image.Image],
) -> tuple[dict[int, Image.Image], dict[str, Any], dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = load_json(path)
    require(manifest.get("schema") == TURTLE_SCHEMA, "wrong TURTLE manifest schema")
    require(_turtle_checkpoint_sha(manifest) == EXPECTED_TURTLE_CHECKPOINT_SHA256,
            "TURTLE checkpoint SHA does not match pinned official GoPro checkpoint")
    camera = manifest.get("camera")
    require(isinstance(camera, Mapping), "TURTLE camera contract missing")
    require(int(camera.get("width", -1)) == 512 and int(camera.get("height", -1)) == 384,
            "TURTLE tracker output shape is not 512x384")
    require(int(camera.get("resize_before_crop_width", -1)) == 528 and
            int(camera.get("resize_before_crop_height", -1)) == 400,
            "TURTLE did not use the tracker 528x400 pre-crop resize")
    edges = camera.get("crop_edges")
    require(isinstance(edges, Mapping) and
            [int(edges.get(side, -1)) for side in ("left", "right", "top", "bottom")] == [8, 8, 8, 8],
            "TURTLE did not use tracker crop8")
    require(np.allclose(np.asarray(camera.get("K"), dtype=np.float64),
                        np.asarray(TRACKER_K, dtype=np.float64), rtol=0.0, atol=1.0e-9),
            "TURTLE K differs from the exact fr2 tracker K")
    safety = manifest.get("safety")
    require(isinstance(safety, Mapping), "TURTLE manifest omits safety contract")
    for key in ("ground_truth_images_used", "ground_truth_poses_used", "depth_used"):
        require(safety.get(key) is False, f"TURTLE safety contract failed: {key}")

    stream = manifest.get("stream")
    selection = manifest.get("selection")
    frames = manifest.get("frames")
    require(isinstance(stream, Mapping), "TURTLE stream record missing")
    require(isinstance(selection, Mapping), "TURTLE selection record missing")
    require(isinstance(frames, list), "TURTLE frames record missing")
    require(stream.get("processed_source_indices") == list(range(2765)),
            "TURTLE did not process the exact contiguous 0..2764 range")
    require(int(stream.get("processed_count", -1)) == 2765, "wrong processed_count")
    require(int(stream.get("step_count", -1)) == 2765, "wrong step_count")
    require(int(stream.get("cache_updates", -1)) == 2765, "wrong cache update count")
    require(int(stream.get("reset_count", -1)) == 1, "TURTLE must reset exactly once")
    require(stream.get("first_pair") == "self", "TURTLE first frame must self-pair")
    require(stream.get("persistent_kv") is True, "persistent TURTLE K/V is not declared")
    require(selection.get("emitted_source_indices") == list(CLEAR_42),
            "TURTLE emitted selection differs from CLEAR_42")
    require(int(selection.get("emitted_count", -1)) == len(CLEAR_42),
            "wrong emitted frame count")
    steps = stream.get("steps")
    require(isinstance(steps, list) and len(steps) == 2765, "TURTLE step audit incomplete")
    for ordinal, step in enumerate(steps):
        require(isinstance(step, Mapping), f"invalid TURTLE step {ordinal}")
        require(int(step.get("step_index", -1)) == ordinal, "step_index discontinuity")
        require(int(step.get("source_index", -1)) == ordinal, "source_index discontinuity")
        require(bool(step.get("cache_present_after")), "cache absent after step")
        require(bool(step.get("cache_present_before")) == (ordinal > 0),
                "cache-before sequence contract failed")
        require(int(step.get("cache_update_ordinal", -1)) == ordinal + 1,
                "cache update ordinal discontinuity")
        require(int(step.get("reset_count", -1)) == 1, "unexpected mid-stream reset")

    old_by_index = {
        int(frame["source_index"]): frame
        for frame in old_input_manifest.get("frames", [])
        if isinstance(frame, Mapping) and "source_index" in frame
    }
    require(set(old_by_index) == set(CLEAR_42), "old input manifest is not the fixed 42-frame set")
    images: dict[int, Image.Image] = {}
    frame_hashes: dict[str, str] = {}
    require(len(frames) == len(CLEAR_42), "TURTLE PNG manifest count mismatch")
    for record in frames:
        require(isinstance(record, Mapping), "invalid TURTLE frame record")
        index = int(record.get("source_index", -1))
        require(index in CLEAR_42 and index not in images, "invalid/duplicate TURTLE source index")
        require(record.get("provider") == EXPECTED_TURTLE_PROVIDER, "wrong TURTLE provider")
        output = record.get("output")
        input_record = record.get("input")
        require(isinstance(output, Mapping) and isinstance(input_record, Mapping),
                "TURTLE frame omits input/output record")
        output_path = Path(str(output.get("path", ""))).expanduser().resolve()
        digest = sha256_file(output_path)
        require(digest == output.get("sha256") == record.get("output_sha256"),
                f"TURTLE PNG SHA mismatch at {index}")
        image = load_rgb(output_path, FRONTEND_SIZE)
        require(pixels_sha256(image) == output.get("pixel_sha256"),
                f"TURTLE pixel SHA mismatch at {index}")
        require(str(input_record.get("sha256")) == str(old_by_index[index].get("raw_sha256")),
                f"raw source SHA mismatch at {index}")
        require(str(input_record.get("preprocessed_pixel_sha256")) == pixels_sha256(references[index]),
                f"TURTLE preprocessed input is not pixel-identical to clear reference at {index}")
        images[index] = image
        frame_hashes[f"{index:06d}.png"] = digest

    latency = [float(step["latency_ms"]) for step in steps]
    require(all(math.isfinite(value) and value >= 0 for value in latency),
            "invalid TURTLE latency")
    steady = latency[1:]
    performance = manifest.get("performance") if isinstance(manifest.get("performance"), Mapping) else {}
    latency_summary = {
        "scope": "official TURTLE step including synchronization",
        "sample_count_all": len(latency),
        "first_frame_ms": latency[0],
        "steady_state_excludes_first_frame": True,
        "steady_state_sample_count": len(steady),
        "steady_state_mean_ms": float(np.mean(steady)),
        "steady_state_median_ms": float(np.median(steady)),
        "steady_state_p95_ms": float(np.percentile(steady, 95)),
        "steady_state_max_ms": float(max(steady)),
        "steady_state_throughput_fps_from_mean": float(1000.0 / np.mean(steady)),
        "thirty_fps_p95_threshold_ms": REALTIME_30FPS_P95_MS,
        "thirty_fps_p95_feasible": bool(np.percentile(steady, 95) <= REALTIME_30FPS_P95_MS),
        "producer_all_frame_summary": performance.get("latency_ms"),
        "stream_wall_seconds": performance.get("stream_wall_seconds"),
        "peak_cuda_memory_allocated_bytes": performance.get("peak_cuda_memory_allocated_bytes"),
    }
    provenance = {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "checkpoint_sha256": _turtle_checkpoint_sha(manifest),
        "frame_png_sha256": frame_hashes,
        "safety": dict(safety),
        "camera": camera,
        "preprocessed_inputs_pixel_identical_to_clear_reference": True,
    }
    return images, latency_summary, {"manifest": manifest, "provenance": provenance}


def load_evssm_26(manifest_path: Path | str) -> tuple[dict[int, Image.Image], dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = load_json(path)
    checkpoint = manifest.get("evssm_checkpoint")
    require(isinstance(checkpoint, Mapping), "EVSSM input manifest omits checkpoint")
    require(checkpoint.get("sha256") == EXPECTED_EVSSM_CHECKPOINT_SHA256,
            "EVSSM checkpoint SHA mismatch")
    frames = manifest.get("frames")
    require(isinstance(frames, list), "EVSSM input manifest omits frames")
    images: dict[int, Image.Image] = {}
    providers: dict[int, str] = {}
    for frame in frames:
        require(isinstance(frame, Mapping), "invalid EVSSM input frame")
        index = int(frame["source_index"])
        provider = str(frame.get("provider", ""))
        providers[index] = provider
        if provider != "official_unblur_evssm":
            continue
        output = frame.get("output")
        require(isinstance(output, Mapping), "EVSSM frame omits output")
        image_path = Path(str(output.get("path", ""))).expanduser().resolve()
        require(sha256_file(image_path) == output.get("sha256") == frame.get("png_sha256"),
                f"EVSSM PNG SHA mismatch at {index}")
        images[index] = load_rgb(image_path, FRONTEND_SIZE)
    require(tuple(sorted(images)) == EVSSM_26, "official EVSSM coverage differs from EVSSM_26")
    require(set(providers) == set(CLEAR_42), "old input manifest coverage differs from CLEAR_42")
    require(sum(value == "raw_undistorted" for value in providers.values()) == 16,
            "expected exactly 16 raw fallbacks in historical mixed stream")
    camera_caveat = {
        "status": "HISTORICAL_CAMERA_CONTRACT_INCOHERENT",
        "official_evssm_tracker_tensor_count": 26,
        "raw_direct_resize_fallback_count": 16,
        "reason": (
            "tracker EVSSM tensors use 528x400 resize then crop8, raw fallbacks use "
            "direct 512x384 resize, while the scene declares one direct-resize K"
        ),
        "fair_turtle_vs_historical_mix_attribution_allowed": False,
    }
    return images, {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "providers": {str(key): value for key, value in sorted(providers.items())},
        "historical_camera_caveat": camera_caveat,
    }


def _temporal_unwarped(
    previous: Image.Image,
    current: Image.Image,
    previous_reference: Image.Image,
    current_reference: Image.Image,
) -> dict[str, float]:
    p = rgb_chw(previous)
    c = rgb_chw(current)
    pg = rgb_chw(previous_reference)
    cg = rgb_chw(current_reference)
    change = c - p
    reference_change = cg - pg
    return {
        "adjacent_change_l1": float(np.mean(np.abs(change), dtype=np.float64)),
        "reference_temporal_difference_error_l1": float(
            np.mean(np.abs(change - reference_change), dtype=np.float64)
        ),
    }


def _remap(array: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.remap(
        array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _reference_flow_warped(
    previous: Image.Image,
    current: Image.Image,
    previous_reference: Image.Image,
    current_reference: Image.Image,
) -> dict[str, Any]:
    """Reference-only Farneback flow diagnostic with forward/backward mask."""

    import cv2

    previous_gt_u8 = np.asarray(previous_reference.convert("RGB"), dtype=np.uint8)
    current_gt_u8 = np.asarray(current_reference.convert("RGB"), dtype=np.uint8)
    previous_gray = cv2.cvtColor(previous_gt_u8, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_gt_u8, cv2.COLOR_RGB2GRAY)
    params = dict(
        pyr_scale=0.5,
        levels=5,
        winsize=21,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    # backward maps each current pixel to a previous-frame coordinate.
    backward = cv2.calcOpticalFlowFarneback(current_gray, previous_gray, None, **params)
    forward = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, **params)
    height, width = current_gray.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = grid_x + backward[..., 0]
    map_y = grid_y + backward[..., 1]
    inside = (
        (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    sampled_forward = _remap(forward, map_x, map_y)
    fb_error = np.linalg.norm(backward + sampled_forward, axis=2)
    valid = inside & np.isfinite(fb_error) & (fb_error <= FLOW_FB_THRESHOLD_PX)
    valid_fraction = float(valid.mean())
    if valid_fraction < FLOW_MIN_VALID_FRACTION:
        return {
            "available": False,
            "valid_fraction": valid_fraction,
            "minimum_valid_fraction": FLOW_MIN_VALID_FRACTION,
            "forward_backward_threshold_px": FLOW_FB_THRESHOLD_PX,
            "warped_adjacent_change_l1": None,
            "reference_flow_warped_difference_error_l1": None,
        }

    p = np.asarray(previous.convert("RGB"), dtype=np.float32) / 255.0
    c = np.asarray(current.convert("RGB"), dtype=np.float32) / 255.0
    pg = previous_gt_u8.astype(np.float32) / 255.0
    cg = current_gt_u8.astype(np.float32) / 255.0
    warped_p = _remap(p, map_x, map_y)
    warped_pg = _remap(pg, map_x, map_y)
    candidate_residual = c - warped_p
    reference_residual = cg - warped_pg
    mask = np.repeat(valid[..., None], 3, axis=2)
    return {
        "available": True,
        "valid_fraction": valid_fraction,
        "minimum_valid_fraction": FLOW_MIN_VALID_FRACTION,
        "forward_backward_threshold_px": FLOW_FB_THRESHOLD_PX,
        "warped_adjacent_change_l1": float(np.mean(np.abs(candidate_residual[mask]))),
        "reference_flow_warped_difference_error_l1": float(
            np.mean(np.abs((candidate_residual - reference_residual)[mask]))
        ),
    }


def temporal_rows(
    references: Mapping[int, Image.Image],
    candidates: Mapping[str, Mapping[int, Image.Image]],
    indices: Sequence[int],
    *,
    flow_warp: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = [int(index) for index in indices]
    rows: list[dict[str, Any]] = []
    for previous_index, current_index in zip(ordered[:-1], ordered[1:]):
        for method, images in candidates.items():
            require(previous_index in images and current_index in images,
                    f"{method} missing temporal pair")
            row: dict[str, Any] = {
                "method": method,
                "previous_source_index": previous_index,
                "current_source_index": current_index,
                "source_index_gap": current_index - previous_index,
                **_temporal_unwarped(
                    images[previous_index],
                    images[current_index],
                    references[previous_index],
                    references[current_index],
                ),
            }
            row["flow"] = (
                _reference_flow_warped(
                    images[previous_index],
                    images[current_index],
                    references[previous_index],
                    references[current_index],
                )
                if flow_warp
                else {
                    "available": False,
                    "reason": "disabled_by_cli",
                    "valid_fraction": None,
                    "warped_adjacent_change_l1": None,
                    "reference_flow_warped_difference_error_l1": None,
                }
            )
            rows.append(row)
    summary: dict[str, Any] = {}
    for method in candidates:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            "pair_count": len(subset),
            "mean_adjacent_change_l1": _finite_mean(
                row["adjacent_change_l1"] for row in subset
            ),
            "mean_reference_temporal_difference_error_l1": _finite_mean(
                row["reference_temporal_difference_error_l1"] for row in subset
            ),
            "flow_available_pair_count": sum(bool(row["flow"]["available"]) for row in subset),
            "mean_flow_valid_fraction": _finite_mean(row["flow"]["valid_fraction"] for row in subset),
            "mean_warped_adjacent_change_l1": _finite_mean(
                row["flow"]["warped_adjacent_change_l1"] for row in subset
            ),
            "mean_reference_flow_warped_difference_error_l1": _finite_mean(
                row["flow"]["reference_flow_warped_difference_error_l1"]
                for row in subset
            ),
        }
    return rows, summary


def _resplat_checkpoint_sha(manifest: Mapping[str, Any]) -> Optional[str]:
    official = manifest.get("official_resplat")
    if not isinstance(official, Mapping):
        return None
    checkpoint = official.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        return None
    return str(checkpoint.get("sha256") or "") or None


def load_resplat_run(root: Path | str) -> dict[str, Any]:
    directory = Path(root).expanduser().resolve()
    manifest_path = directory / "run_manifest.json"
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == RESPLAT_SCHEMA, f"wrong ReSplat schema: {directory}")
    require(_resplat_checkpoint_sha(manifest) == EXPECTED_RESPLAT_CHECKPOINT_SHA256,
            f"wrong official ReSplat checkpoint: {directory}")
    contract = manifest.get("paired_contract")
    require(isinstance(contract, Mapping), "ReSplat paired contract missing")
    require(int(contract.get("encoder_forward_calls", -1)) == 1, "ReSplat encoder call count != 1")
    require(int(contract.get("forward_update_calls", -1)) == 1, "ReSplat update call count != 1")
    require(contract.get("init_object_passed_directly_to_forward_update") is True,
            "ReSplat init object was not passed directly")
    require(contract.get("initial_state_in_place_mutation_detected") is False,
            "ReSplat init state mutated in place")
    official = manifest.get("official_resplat")
    require(isinstance(official, Mapping) and int(official.get("num_refine", -1)) == 4,
            "ReSplat run is not refine4")
    selection = manifest.get("selection")
    require(isinstance(selection, Mapping), "ReSplat selection missing")
    context_names = list(selection.get("context_names", []))
    target_names = list(selection.get("target_names", []))
    require(len(context_names) == 8 and len(target_names) == 34, "ReSplat must be 8 context / 34 target")
    require(set(context_names).isdisjoint(target_names), "context and targets overlap")
    require(
        {int(Path(name).stem) for name in context_names + target_names} == set(CLEAR_42),
        "ReSplat selection does not partition CLEAR_42",
    )
    init_metrics = load_json(directory / "paired_init0" / "metrics.json")
    refine_metrics = load_json(directory / "paired_refine4" / "metrics.json")
    for label, metrics in (("init0", init_metrics), ("refine4", refine_metrics)):
        require(isinstance(metrics.get("mean"), Mapping), f"{label} official mean missing")
        require(len(metrics.get("per_view", [])) == 34, f"{label} per-view count != 34")
        require([row["name"] for row in metrics["per_view"]] == target_names,
                f"{label} metric names differ from selection")
    mean0 = init_metrics["mean"]
    mean4 = refine_metrics["mean"]
    primary = {
        "metric_target": "this run's own frontend stream",
        "cross_frontend_absolute_difference_is_attributable": False,
        "init0": dict(mean0),
        "refine4": dict(mean4),
        "refine4_minus_init0": {
            "psnr_db": float(mean4["psnr"]) - float(mean0["psnr"]),
            "ssim": float(mean4["ssim"]) - float(mean0["ssim"]),
            "lpips": float(mean4["lpips"]) - float(mean0["lpips"]),
        },
    }
    return {
        "root": directory,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "context_names": context_names,
        "target_names": target_names,
        "init_metrics": init_metrics,
        "refine_metrics": refine_metrics,
        "primary": primary,
    }


def audit_resplat_pair(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, Any]:
    require(new["context_names"] == old["context_names"], "new/old context names differ")
    require(new["target_names"] == old["target_names"], "new/old target names differ")
    new_manifest = new["manifest"]
    old_manifest = old["manifest"]
    new_official = new_manifest["official_resplat"]
    old_official = old_manifest["official_resplat"]
    new_repo = new_official["repository"]
    old_repo = old_official["repository"]
    require(new_repo.get("commit") == old_repo.get("commit"), "ReSplat commits differ")
    require(_resplat_checkpoint_sha(new_manifest) == _resplat_checkpoint_sha(old_manifest),
            "ReSplat checkpoint SHAs differ")
    require(new_manifest.get("image_shape") == old_manifest.get("image_shape") == [320, 448],
            "ReSplat image shapes differ")
    new_scene = new_manifest.get("scene")
    old_scene = old_manifest.get("scene")
    require(isinstance(new_scene, Mapping) and isinstance(old_scene, Mapping), "scene records missing")
    require(int(new_scene.get("image_count", -1)) == int(old_scene.get("image_count", -1)) == 42,
            "scene image counts differ")
    return {
        "same_context_names": True,
        "same_target_names": True,
        "same_official_resplat_commit": True,
        "same_official_resplat_checkpoint": True,
        "same_model_image_shape": True,
        "same_42_source_partition": True,
        "cross_run_attribution_blocked_by_historical_mixed_camera_contract": True,
    }


def _scene_frame_map(scene_manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    frames = scene_manifest.get("frames")
    require(isinstance(frames, list) and len(frames) == 42, "scene manifest must contain 42 frames")
    result: dict[int, Mapping[str, Any]] = {}
    for frame in frames:
        require(isinstance(frame, Mapping), "invalid scene frame record")
        index = int(frame.get("source_index", -1))
        require(index in CLEAR_42 and index not in result, "invalid/duplicate scene source index")
        result[index] = frame
    return result


def audit_scene_pair_and_turtle_binding(
    new_run: Mapping[str, Any],
    old_run: Mapping[str, Any],
    turtle_record: Mapping[str, Any],
) -> dict[str, Any]:
    new_scene_path = Path(str(new_run["manifest"]["scene"]["manifest_path"])).resolve()
    old_scene_path = Path(str(old_run["manifest"]["scene"]["manifest_path"])).resolve()
    new_scene = load_json(new_scene_path)
    old_scene = load_json(old_scene_path)
    require(sha256_file(new_scene_path) == new_run["manifest"]["scene"]["manifest_sha256"],
            "new scene manifest SHA binding failed")
    require(sha256_file(old_scene_path) == old_run["manifest"]["scene"]["manifest_sha256"],
            "old scene manifest SHA binding failed")
    new_frames = _scene_frame_map(new_scene)
    old_frames = _scene_frame_map(old_scene)
    pose_sha: dict[int, str] = {}
    for index in CLEAR_42:
        new_pose = new_frames[index].get("effective_pose")
        old_pose = old_frames[index].get("effective_pose")
        require(isinstance(new_pose, Mapping) and isinstance(old_pose, Mapping), "scene pose audit missing")
        require(new_pose.get("uses_ground_truth_pose") is False and
                old_pose.get("uses_ground_truth_pose") is False,
                "ground-truth pose entered a ReSplat scene")
        require(new_pose.get("c2w_sha256") == old_pose.get("c2w_sha256"),
                f"new/old DROID C2W differs at source {index}")
        pose_sha[index] = str(new_pose.get("c2w_sha256"))

    turtle_manifest = turtle_record["manifest"]
    turtle_frames = {int(frame["source_index"]): frame for frame in turtle_manifest["frames"]}
    for index in CLEAR_42:
        selected = new_frames[index].get("selected_image")
        require(isinstance(selected, Mapping), "new scene selected-image audit missing")
        turtle_output = turtle_frames[index]["output"]
        require(selected.get("source_sha256") == turtle_output.get("sha256"),
                f"new scene is not bound to TURTLE PNG at source {index}")
        require(selected.get("exported_sha256") == turtle_output.get("sha256"),
                f"new scene copied TURTLE PNG bytes incorrectly at source {index}")

    return {
        "new_scene_manifest": str(new_scene_path),
        "new_scene_manifest_sha256": sha256_file(new_scene_path),
        "old_scene_manifest": str(old_scene_path),
        "old_scene_manifest_sha256": sha256_file(old_scene_path),
        "all_42_non_gt_droid_pose_sha_identical": True,
        "same_pose_verified_from_scene_manifests": True,
        "new_scene_all_42_pngs_bound_to_turtle_manifest": True,
        "pose_c2w_sha256": {str(index): pose_sha[index] for index in CLEAR_42},
    }


def resplat_clear_gt_metrics(
    run: Mapping[str, Any],
    references: Mapping[int, Image.Image],
    lpips: LpipsVggCpu,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[int, Image.Image]]]:
    indices = [int(Path(name).stem) for name in run["target_names"]]
    resized_references = {
        index: references[index].resize(RESPLAT_SIZE, Image.Resampling.LANCZOS)
        for index in indices
    }
    arms: dict[str, dict[int, Image.Image]] = {}
    summaries: dict[str, Any] = {}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for label, directory in (("init0", "paired_init0"), ("refine4", "paired_refine4")):
        images: dict[int, Image.Image] = {}
        for name, index in zip(run["target_names"], indices):
            images[index] = load_rgb(run["root"] / directory / "rendered" / name, RESPLAT_SIZE)
        rows, summary = metric_rows(resized_references, images, indices, lpips)
        arms[label] = images
        rows_by_arm[label] = rows
        summaries[label] = summary

    def numeric(value: Any) -> float:
        require(isinstance(value, (int, float)), "unexpected infinite ReSplat metric")
        return float(value)

    summaries["refine4_minus_init0"] = {
        "mean_psnr_db": numeric(summaries["refine4"]["mean_psnr_db"])
        - numeric(summaries["init0"]["mean_psnr_db"]),
        "mean_ssim": float(summaries["refine4"]["mean_ssim"])
        - float(summaries["init0"]["mean_ssim"]),
        "mean_lpips_vgg": (
            None
            if summaries["init0"]["mean_lpips_vgg"] is None
            else float(summaries["refine4"]["mean_lpips_vgg"])
            - float(summaries["init0"]["mean_lpips_vgg"])
        ),
    }
    init_by_index = {int(row["source_index"]): row for row in rows_by_arm["init0"]}
    refine_by_index = {int(row["source_index"]): row for row in rows_by_arm["refine4"]}
    merged = []
    for index in indices:
        initial = init_by_index[index]
        refined = refine_by_index[index]
        merged.append(
            {
                "source_index": index,
                "init0_psnr_db": initial["psnr_db"],
                "init0_ssim": initial["ssim"],
                "init0_lpips_vgg": initial["lpips_vgg"],
                "refine4_psnr_db": refined["psnr_db"],
                "refine4_ssim": refined["ssim"],
                "refine4_lpips_vgg": refined["lpips_vgg"],
                "delta_psnr_db": float(refined["psnr_db"]) - float(initial["psnr_db"]),
                "delta_ssim": float(refined["ssim"]) - float(initial["ssim"]),
                "delta_lpips_vgg": (
                    None
                    if initial["lpips_vgg"] is None
                    else float(refined["lpips_vgg"]) - float(initial["lpips_vgg"])
                ),
            }
        )
    return summaries, merged, arms


def _geometry_stats_from_arrays(initial: np.ndarray, refined: np.ndarray) -> dict[str, Any]:
    initial = np.asarray(initial, dtype=np.float64)
    refined = np.asarray(refined, dtype=np.float64)
    require(initial.ndim == 2 and initial.shape[1] == 3, "initial PLY XYZ shape invalid")
    require(refined.shape == initial.shape, "init/refine PLY vertex topology differs")
    require(bool(np.isfinite(initial).all()) and bool(np.isfinite(refined).all()),
            "non-finite native Gaussian position")
    displacement = np.linalg.norm(refined - initial, axis=1)
    maximum_index = int(np.argmax(displacement))
    return {
        "same_index_diagnostic_only": True,
        "vertex_count_init0": int(len(initial)),
        "vertex_count_refine4": int(len(refined)),
        "all_positions_finite": True,
        "nonzero_displacement_count": int(np.count_nonzero(displacement > 0.0)),
        "position_displacement_m_assuming_scene_scale": {
            "mean": float(np.mean(displacement)),
            "median": float(np.median(displacement)),
            "p95": float(np.percentile(displacement, 95)),
            "max": float(displacement[maximum_index]),
            "max_vertex_index": maximum_index,
            "count_gt_1m": int(np.count_nonzero(displacement > 1.0)),
            "count_gt_5m": int(np.count_nonzero(displacement > 5.0)),
            "init_max_vertex_xyz": initial[maximum_index].tolist(),
            "refine_max_vertex_xyz": refined[maximum_index].tolist(),
        },
        "interpretation_limit": (
            "Same-index movement validates a native update path; without GT geometry it does "
            "not establish correspondence, metric-scale accuracy, or geometric improvement."
        ),
    }


def resplat_geometry_update(run: Mapping[str, Any]) -> dict[str, Any]:
    from plyfile import PlyData

    def xyz(path: Path) -> np.ndarray:
        require(path.is_file(), f"missing native Gaussian PLY: {path}")
        vertices = PlyData.read(str(path))["vertex"].data
        return np.column_stack((vertices["x"], vertices["y"], vertices["z"]))

    initial_path = run["root"] / "paired_init0" / "gaussians.ply"
    refined_path = run["root"] / "paired_refine4" / "gaussians.ply"
    return {
        **_geometry_stats_from_arrays(xyz(initial_path), xyz(refined_path)),
        "init0_ply": str(initial_path),
        "init0_ply_sha256": sha256_file(initial_path),
        "refine4_ply": str(refined_path),
        "refine4_ply_sha256": sha256_file(refined_path),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _labelled_panel(image: Image.Image, label: str, width: int = 320) -> Image.Image:
    height = int(round(image.height * width / image.width))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height + 30), (18, 18, 18))
    panel.paste(resized, (0, 30))
    ImageDraw.Draw(panel).text((8, 8), label, fill=(255, 255, 255))
    return panel


def write_frontend_montage(
    path: Path,
    references: Mapping[int, Image.Image],
    evssm: Mapping[int, Image.Image],
    turtle: Mapping[int, Image.Image],
) -> None:
    rows = []
    for index in FIXED_FIVE:
        panels = [
            _labelled_panel(references[index], f"Raw/reference src {index}"),
            _labelled_panel(evssm[index], "Official EVSSM"),
            _labelled_panel(turtle[index], "Official TURTLE GoPro"),
        ]
        row = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)))
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)
    canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), (12, 12, 12))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(path)


def write_resplat_montage(
    path: Path,
    references: Mapping[int, Image.Image],
    old_arms: Mapping[str, Mapping[int, Image.Image]],
    new_arms: Mapping[str, Mapping[int, Image.Image]],
) -> None:
    rows = []
    for index in FIXED_FIVE:
        reference = references[index].resize(RESPLAT_SIZE, Image.Resampling.LANCZOS)
        panels = [
            _labelled_panel(reference, f"Clear reference src {index}", 280),
            _labelled_panel(old_arms["init0"][index], "Historical mix init0", 280),
            _labelled_panel(old_arms["refine4"][index], "Historical mix refine4", 280),
            _labelled_panel(new_arms["init0"][index], "TURTLE stream init0", 280),
            _labelled_panel(new_arms["refine4"][index], "TURTLE stream refine4", 280),
        ]
        row = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)))
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)
    canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), (12, 12, 12))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(path)


def _direction_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "psnr_improved": sum(float(row["delta_psnr_db"]) > 0 for row in rows),
        "psnr_unchanged": sum(float(row["delta_psnr_db"]) == 0 for row in rows),
        "psnr_worsened": sum(float(row["delta_psnr_db"]) < 0 for row in rows),
        "ssim_improved": sum(float(row["delta_ssim"]) > 0 for row in rows),
        "ssim_unchanged": sum(float(row["delta_ssim"]) == 0 for row in rows),
        "ssim_worsened": sum(float(row["delta_ssim"]) < 0 for row in rows),
        "lpips_improved": sum(
            row.get("delta_lpips_vgg") is not None and float(row["delta_lpips_vgg"]) < 0
            for row in rows
        ),
        "lpips_unchanged": sum(
            row.get("delta_lpips_vgg") is not None and float(row["delta_lpips_vgg"]) == 0
            for row in rows
        ),
        "lpips_worsened": sum(
            row.get("delta_lpips_vgg") is not None and float(row["delta_lpips_vgg"]) > 0
            for row in rows
        ),
    }


def _frontend_csv_rows(
    scope: str,
    method: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [{"scope": scope, "method": method, **dict(row)} for row in rows]


def run(args: argparse.Namespace) -> Path:
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite report output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    installed = False
    try:
        references, clear_hashes = clear_references(args.clear_gt_pairs)
        old_input_manifest = load_json(args.evssm_input_manifest)
        evssm, evssm_provenance = load_evssm_26(args.evssm_input_manifest)
        turtle, turtle_latency, turtle_record = load_turtle_artifact(
            args.turtle_manifest, old_input_manifest, references
        )
        require(set(turtle) == set(CLEAR_42), "TURTLE image coverage differs from CLEAR_42")
        require(set(evssm) == set(EVSSM_26), "EVSSM image coverage differs from EVSSM_26")
        require(all(index in EVSSM_26 for index in FIXED_FIVE), "fixed views leave EVSSM intersection")

        lpips = LpipsVggCpu(not args.skip_lpips, batch_size=args.lpips_batch_size)
        raw42_rows, raw42_summary = exact_reference_control_rows(references, CLEAR_42)
        turtle42_rows, turtle42_summary = metric_rows(references, turtle, CLEAR_42, lpips)
        raw26_rows, raw26_summary = exact_reference_control_rows(references, EVSSM_26)
        evssm26_rows, evssm26_summary = metric_rows(references, evssm, EVSSM_26, lpips)
        turtle26_rows, turtle26_summary = metric_rows(references, turtle, EVSSM_26, lpips)
        require(raw42_summary["exact_match_count"] == 42, "raw/reference control is not exact")
        require(raw26_summary["exact_match_count"] == 26, "raw/reference intersection is not exact")

        temporal, temporal_summary = temporal_rows(
            references,
            {"raw_reference_control": references, "official_evssm": evssm, "official_turtle_gopro": turtle},
            EVSSM_26,
            flow_warp=not args.skip_flow,
        )

        old_run = load_resplat_run(args.previous_resplat_root)
        new_run = load_resplat_run(args.turtle_resplat_root)
        resplat_contract = audit_resplat_pair(new_run, old_run)
        scene_contract = audit_scene_pair_and_turtle_binding(new_run, old_run, turtle_record)
        old_clear_summary, old_clear_rows, old_arms = resplat_clear_gt_metrics(old_run, references, lpips)
        new_clear_summary, new_clear_rows, new_arms = resplat_clear_gt_metrics(new_run, references, lpips)
        old_geometry = resplat_geometry_update(old_run)
        new_geometry = resplat_geometry_update(new_run)

        frontend_rows = (
            _frontend_csv_rows("clear42", "raw_reference_control", raw42_rows)
            + _frontend_csv_rows("clear42", "official_turtle_gopro", turtle42_rows)
            + _frontend_csv_rows("evssm_intersection26", "raw_reference_control", raw26_rows)
            + _frontend_csv_rows("evssm_intersection26", "official_evssm", evssm26_rows)
            + _frontend_csv_rows("evssm_intersection26", "official_turtle_gopro", turtle26_rows)
        )
        _write_csv(staging / "frontend_per_frame.csv", frontend_rows)
        _write_csv(staging / "frontend_temporal_pairs.csv", temporal)
        _write_csv(
            staging / "resplat_clear_gt_per_view.csv",
            [{"run": "historical_evssm_raw_mix", **row} for row in old_clear_rows]
            + [{"run": "turtle_gopro_stream", **row} for row in new_clear_rows],
        )
        write_frontend_montage(staging / "fixed5_frontend.png", references, evssm, turtle)
        write_resplat_montage(staging / "fixed5_resplat_clear_gt.png", references, old_arms, new_arms)

        summary = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "audit_result": "PASS",
            "protocol": {
                "pre_registered_document": str(
                    (Path(__file__).resolve().parents[1] / "docs" / "TURTLE_RESPLAT_SMOKE_ACCEPTANCE_ZH.md")
                ),
                "clear_42_source_indices": list(CLEAR_42),
                "official_evssm_intersection_26": list(EVSSM_26),
                "fixed_visualization_five": list(FIXED_FIVE),
                "frames_selected_from_new_metric_values": False,
                "quality_thresholds_selected_after_results": False,
            },
            "frontend_clear_frame_preservation": {
                "not_a_blurry_frame_deblurring_benchmark": True,
                "raw_is_pixel_identical_to_reference_by_protocol": True,
                "raw_control_interpretation": (
                    "Raw PSNR=Infinity, SSIM=1 and LPIPS=0 are reference-control values, "
                    "not a deblurring result."
                ),
                "metric_definition": {
                    "psnr": "RGB full-frame MSE in [0,1]",
                    "ssim": "win_size=11, gaussian_weights=True, channel_axis=0, data_range=1",
                    "lpips": (
                        "VGG LPIPS normalize=True on CPU, matching official ReSplat"
                        if not args.skip_lpips else "not computed (--skip-lpips)"
                    ),
                },
                "all_clear_42": {
                    "raw_reference_control": raw42_summary,
                    "official_turtle_gopro_zero_shot": turtle42_summary,
                },
                "fair_three_way_intersection_26": {
                    "raw_reference_control": raw26_summary,
                    "official_unblur_evssm": evssm26_summary,
                    "official_turtle_gopro_zero_shot": turtle26_summary,
                },
                "temporal_intersection_26": {
                    "scope": (
                        "25 adjacent pairs in the sorted sparse EVSSM intersection; source-index "
                        "gaps are irregular, so all temporal values are diagnostic only"
                    ),
                    "flow": {
                        "enabled": not args.skip_flow,
                        "provider": "reference-only OpenCV Farneback backward flow",
                        "forward_backward_threshold_px": FLOW_FB_THRESHOLD_PX,
                        "minimum_valid_fraction": FLOW_MIN_VALID_FRACTION,
                    },
                    "mean": temporal_summary,
                },
                "turtle_performance": turtle_latency,
            },
            "official_resplat": {
                "contract": {**resplat_contract, **scene_contract},
                "primary_frontend_stream_metrics": {
                    "cross_run_absolute_values_have_different_pixel_targets": True,
                    "historical_evssm_raw_mix": old_run["primary"],
                    "turtle_gopro_stream": new_run["primary"],
                    "allowed_interpretation": "within-run refine4-minus-init0 only",
                },
                "posthoc_common_clear_gt_34": {
                    "resize": "PIL LANCZOS from 512x384 reference to 448x320",
                    "historical_evssm_raw_mix": {
                        **old_clear_summary,
                        "direction_counts": _direction_counts(old_clear_rows),
                    },
                    "turtle_gopro_stream": {
                        **new_clear_summary,
                        "direction_counts": _direction_counts(new_clear_rows),
                    },
                    "cross_run_delta_attributable_to_turtle": False,
                    "reason": evssm_provenance["historical_camera_caveat"]["reason"],
                },
                "native_gaussian_geometry_update": {
                    "historical_evssm_raw_mix": old_geometry,
                    "turtle_gopro_stream": new_geometry,
                    "cross_run_geometry_accuracy_ranking_allowed": False,
                    "ground_truth_geometry_used": False,
                },
            },
            "provenance": {
                "turtle": turtle_record["provenance"],
                "historical_evssm_mix": evssm_provenance,
                "new_resplat_run_manifest": {
                    "path": str(new_run["manifest_path"]),
                    "sha256": new_run["manifest_sha256"],
                },
                "old_resplat_run_manifest": {
                    "path": str(old_run["manifest_path"]),
                    "sha256": old_run["manifest_sha256"],
                },
                "clear_gt_pair_sha256": clear_hashes,
            },
            "claim_boundaries": {
                "official_turtle_gopro_zero_shot_used": True,
                "replica_blurry_finetuned_turtle_used": False,
                "causal_persistent_kv_stream_contract_validated": True,
                "turtle_quality_on_blurry_tum_frames_established": False,
                "slam_ate_or_tracking_stability_established": False,
                "formal_26k_result_present": False,
                "legacy_iter400_used_as_26k": False,
                "legacy_iter400_right_half_render_used_in_frontend_metrics": False,
                "historical_evssm_mix_is_camera_consistent": False,
                "official_resplat_directly_refined_unblur_26k_state": False,
            },
            "artifacts": {
                "frontend_per_frame_csv": "frontend_per_frame.csv",
                "frontend_temporal_pairs_csv": "frontend_temporal_pairs.csv",
                "resplat_clear_gt_per_view_csv": "resplat_clear_gt_per_view.csv",
                "fixed5_frontend": "fixed5_frontend.png",
                "fixed5_resplat_clear_gt": "fixed5_resplat_clear_gt.png",
            },
        }
        with (staging / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing concurrent overwrite: {destination}")
        os.rename(staging, destination)
        installed = True
        return destination
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turtle-manifest", type=Path, required=True)
    parser.add_argument("--evssm-input-manifest", type=Path, required=True)
    parser.add_argument("--clear-gt-pairs", type=Path, required=True)
    parser.add_argument("--turtle-resplat-root", type=Path, required=True)
    parser.add_argument("--previous-resplat-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lpips-batch-size", type=int, default=4)
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Contract-test only; formal report must compute VGG LPIPS",
    )
    parser.add_argument(
        "--skip-flow",
        action="store_true",
        help="Contract-test only; formal report should compute reference-flow diagnostics",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.lpips_batch_size < 1:
        raise ValueError("--lpips-batch-size must be positive")
    output = run(args)
    print(json.dumps({"output": str(output), "summary": str(output / "summary.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
