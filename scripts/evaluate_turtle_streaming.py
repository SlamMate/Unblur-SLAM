#!/usr/bin/env python3
"""Evaluate raw, optional EVSSM, streaming TURTLE, and sharp GT frames.

The evaluator resets TURTLE's official K/V state at every JSONL sequence
boundary.  It reports spatial quality, two explicitly non-flow-warped temporal
diagnostics, per-frame TURTLE latency, peak CUDA allocation, and triptych or
quad comparison images.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_turtle_streaming import (  # noqa: E402
    DEFAULT_TURTLE_CHECKPOINT,
    DEFAULT_TURTLE_CONFIG,
    DEFAULT_TURTLE_REPO,
    SequenceRecord,
    choose_device,
    load_sequence_manifest,
    read_rgb_tensor,
)
from src.turtle_backend import (  # noqa: E402
    PINNED_TURTLE_CHECKPOINT_SHA256,
    TurtleStreamingBackend,
    load_turtle_model,
    sha256_file,
)
from src.video_deblur.dataset import load_evssm_precompute_report  # noqa: E402


FORMAL_WARMUP_STEPS = 1
FORMAL_STEADY_FRAME_INDEX_MIN = 3
FORMAL_HISTORY_CONTROL_FRAME_INDICES = (3, 19, 39, 59, 79, 99)


def image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """PSNR, RGB SSIM and L1 for CHW tensors in [0, 1]."""

    prediction = prediction.detach().float().clamp(0.0, 1.0).cpu()
    target = target.detach().float().clamp(0.0, 1.0).cpu()
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("image metrics require matching CHW tensors")
    mse = float(torch.mean((prediction - target) ** 2).item())
    prediction_np = prediction.permute(1, 2, 0).numpy()
    target_np = target.permute(1, 2, 0).numpy()
    minimum_side = min(prediction_np.shape[:2])
    if minimum_side < 3:
        ssim = float("nan")
    else:
        window = min(7, minimum_side if minimum_side % 2 else minimum_side - 1)
        ssim = float(
            structural_similarity(
                prediction_np,
                target_np,
                data_range=1.0,
                channel_axis=2,
                win_size=window,
            )
        )
    return {
        "psnr": -10.0 * math.log10(max(mse, 1.0e-12)),
        "ssim": ssim,
        "l1": float(torch.mean(torch.abs(prediction - target)).item()),
    }


def temporal_metrics(
    current: torch.Tensor,
    previous: torch.Tensor,
    current_gt: torch.Tensor,
    previous_gt: torch.Tensor,
) -> Dict[str, float]:
    """Adjacent change and GT difference error; neither uses optical flow."""

    current = current.detach().float().clamp(0.0, 1.0)
    previous = previous.detach().float().clamp(0.0, 1.0)
    current_gt = current_gt.detach().float().clamp(0.0, 1.0)
    previous_gt = previous_gt.detach().float().clamp(0.0, 1.0)
    predicted_difference = current - previous
    gt_difference = current_gt - previous_gt
    return {
        "adjacent_change_l1": float(predicted_difference.abs().mean().item()),
        "gt_temporal_difference_error_l1": float(
            (predicted_difference - gt_difference).abs().mean().item()
        ),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    return float(np.percentile(values, percentile)) if values else None


def _metric_delta(
    candidate: Mapping[str, Optional[float]],
    reference: Mapping[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    """Subtract two identically aggregated RGB metric mappings."""

    result: Dict[str, Optional[float]] = {}
    for metric in ("psnr", "ssim", "l1"):
        left = candidate.get(metric)
        right = reference.get(metric)
        result[metric] = (
            None if left is None or right is None else float(left) - float(right)
        )
    return result


def _pad_to_multiple(image: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, int, int]:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("TURTLE evaluation input must be 1x3xHxW")
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if not pad_height and not pad_width:
        return image, int(height), int(width)
    mode = (
        "reflect"
        if height > pad_height and width > pad_width and height > 1 and width > 1
        else "replicate"
    )
    return F.pad(image, (0, pad_width, 0, pad_height), mode=mode), int(height), int(width)


def _to_pil(tensor: torch.Tensor, *, panel_width: int) -> Image.Image:
    array = (
        tensor.detach().float().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    )
    image = Image.fromarray(np.round(array * 255.0).astype(np.uint8))
    if panel_width > 0 and image.width > panel_width:
        height = max(1, int(round(image.height * panel_width / image.width)))
        image = image.resize((panel_width, height), Image.Resampling.LANCZOS)
    return image


def write_montage(
    path: Path,
    items: Sequence[Tuple[str, torch.Tensor]],
    *,
    subtitle: str,
    panel_width: int = 480,
) -> None:
    images = [(label, _to_pil(tensor, panel_width=panel_width)) for label, tensor in items]
    width = sum(image.width for _, image in images)
    body_height = max(image.height for _, image in images)
    canvas = Image.new("RGB", (width, body_height + 58), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in images:
        canvas.paste(image, (x, 32))
        draw.text((x + 8, 8), label, fill=(255, 255, 255))
        x += image.width
    draw.text((8, body_height + 39), subtitle, fill=(100, 220, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def prepare_output_directory(path: Path) -> Path:
    """Create a new output directory, refusing any non-empty destination."""

    path = path.expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"evaluation output is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return normalized or "sequence"


def resolve_evssm_paths(
    records: Sequence[SequenceRecord],
    *,
    evssm_manifest: Optional[Path] = None,
    data_root: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Tuple[Path, ...]]], Optional[Dict[str, Any]]]:
    """Resolve complete optional EVSSM paths without silently mixing domains."""

    if evssm_manifest is not None:
        auxiliary = load_sequence_manifest(evssm_manifest, root=data_root)
        by_name = {record.name: record for record in auxiliary}
        if set(by_name) != {record.name for record in records}:
            raise ValueError("EVSSM and raw manifests must contain identical sequence names")
        result: Dict[str, Tuple[Path, ...]] = {}
        for record in records:
            source = by_name[record.name]
            paths = source.teacher if source.teacher is not None else source.blurry
            if len(paths) != len(record.blurry):
                raise ValueError(f"EVSSM frame count mismatch for {record.name!r}")
            result[record.name] = tuple(paths)
        manifest_path = evssm_manifest.expanduser().resolve()
        return result, {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }

    has_teacher = [record.teacher is not None for record in records]
    if any(has_teacher) and not all(has_teacher):
        raise ValueError("teacher paths must be complete for every evaluated sequence")
    if all(has_teacher):
        return {
            record.name: tuple(record.teacher or ()) for record in records
        }, {"manifest": "embedded_in_raw_manifest"}
    return None, None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def history_control_outputs(
    backend: Any,
    current: torch.Tensor,
    past_frames: Sequence[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Construct same-checkpoint counterfactuals for official K/V history.

    The GoPro configuration ignores the left image in its two-frame wrapper;
    only the explicit K/V state contains temporal information.  Every control
    therefore starts with K/V=None and builds only the history stated by its
    name, never a future frame.
    """

    if current.ndim != 4 or current.shape[:2] != (1, 3):
        raise ValueError("history controls require one BCHW RGB frame")
    # Direct cache capacity is two or three frames, but every cached feature
    # was itself produced from earlier recurrent state.  Exact reconstruction
    # therefore requires the complete past prefix, not merely the last three
    # RGB inputs.
    history = list(past_frames)
    for frame in history:
        if frame.ndim != 4 or frame.shape != current.shape:
            raise ValueError("history-control frames must match current BCHW shape")

    reset, _, _ = backend.replay_step(current)

    repeat_k = repeat_v = None
    repeat_previous = None
    for _ in history:
        _, repeat_k, repeat_v = backend.replay_step(
            current,
            k_cache=repeat_k,
            v_cache=repeat_v,
            previous_frame=repeat_previous,
        )
        repeat_previous = current
    repeat, _, _ = backend.replay_step(
        current,
        k_cache=repeat_k,
        v_cache=repeat_v,
        previous_frame=repeat_previous,
    )

    # Preserve the complete past-frame multiset while changing only its order.
    shuffled_history = history[1:] + history[:1] if len(history) > 1 else history
    shuffled_k = shuffled_v = None
    shuffled_previous = None
    for frame in shuffled_history:
        _, shuffled_k, shuffled_v = backend.replay_step(
            frame,
            k_cache=shuffled_k,
            v_cache=shuffled_v,
            previous_frame=shuffled_previous,
        )
        shuffled_previous = frame
    shuffled, _, _ = backend.replay_step(
        current,
        k_cache=shuffled_k,
        v_cache=shuffled_v,
        previous_frame=shuffled_previous,
    )
    return {
        "turtle_reset_cache": reset,
        "turtle_repeat_current": repeat,
        "turtle_shuffled_history": shuffled,
    }


def evaluate_sequences(
    records: Sequence[SequenceRecord],
    backend: Any,
    *,
    device: torch.device,
    output_dir: Path,
    max_visuals: int = 12,
    visual_panel_width: int = 480,
    evssm_paths: Optional[Mapping[str, Sequence[Path]]] = None,
    checkpoint_metadata: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    history_controls: bool = False,
    warmup_steps: int = FORMAL_WARMUP_STEPS,
    history_control_frame_indices: Sequence[int] = FORMAL_HISTORY_CONTROL_FRAME_INDICES,
) -> Dict[str, Any]:
    """Run separate timing and quality/history streams and aggregate metrics.

    A first complete normal stream is timing-only.  A second complete normal
    stream supplies quality and history comparisons, so controls and metrics
    cannot perturb adjacent latency samples.
    Ordered replay is an incremental second stream (linear cost); expensive
    repeat/shuffle counterfactuals are computed only at the frozen within-
    sequence indices.  Every path calls ``backend.replay_step`` and therefore
    shares the normal stream's exact autocast implementation.
    """

    frame_total = sum(len(record.blurry) for record in records)
    if frame_total <= 0:
        raise ValueError("cannot evaluate an empty sequence collection")
    if warmup_steps != FORMAL_WARMUP_STEPS:
        raise ValueError(f"formal TURTLE evaluation fixes warmup_steps={FORMAL_WARMUP_STEPS}")
    frozen_control_indices = tuple(int(value) for value in history_control_frame_indices)
    if frozen_control_indices != FORMAL_HISTORY_CONTROL_FRAME_INDICES:
        raise ValueError(
            "formal history-control frame subset changed: "
            f"{frozen_control_indices!r}"
        )
    visual_count = min(max(0, int(max_visuals)), frame_total)
    visual_indices = (
        set(int(value) for value in np.linspace(0, frame_total - 1, visual_count))
        if visual_count
        else set()
    )
    # One target-independent unmeasured model step initializes kernels and the
    # allocator.  The stream is reset afterwards, so validation frame zero
    # still begins with empty K/V.
    first_raw = read_rgb_tensor(records[0].blurry[0], device=device)
    first_padded, _, _ = _pad_to_multiple(first_raw.unsqueeze(0))
    backend.reset()
    for warmup_index in range(warmup_steps):
        _synchronize(device)
        backend.step(first_padded, timestamp=warmup_index)
        _synchronize(device)
    backend.reset()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Pass 1 is the only timed pass.  It opens blurry inputs but never opens a
    # sharp target and never executes replay/history controls.  Keeping the
    # complete timed stream contiguous prevents metric and counterfactual
    # workloads between adjacent latency samples from changing the benchmark.
    latencies: List[float] = []
    for record in records:
        backend.reset()
        for frame_index, blurry_path in enumerate(record.blurry):
            raw = read_rgb_tensor(blurry_path, device=device)
            padded, _, _ = _pad_to_multiple(raw.unsqueeze(0))
            _synchronize(device)
            started = time.perf_counter()
            backend.step(padded, timestamp=frame_index)
            _synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)
    if len(latencies) != frame_total:
        raise RuntimeError("dedicated TURTLE timing-pass coverage changed")
    # Capture normal-stream allocation before the intentionally much more
    # expensive quality/history pass can allocate replay-control state.
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    # Pass 2 computes quality and history diagnostics.  None of its forwards
    # contribute to the latency samples recorded above.
    rows: List[Dict[str, Any]] = []
    global_index = 0
    temporal_pairs = 0
    ordered_replay_max_abs = 0.0
    for record in records:
        backend.reset()
        previous_sources: Optional[Dict[str, torch.Tensor]] = None
        previous_gt: Optional[torch.Tensor] = None
        past_padded_frames: List[torch.Tensor] = []
        ordered_k = ordered_v = None
        ordered_previous: Optional[torch.Tensor] = None
        sequence_evssm = evssm_paths.get(record.name) if evssm_paths is not None else None
        if sequence_evssm is not None and len(sequence_evssm) != len(record.blurry):
            raise ValueError(f"EVSSM frame count mismatch for {record.name!r}")

        for frame_index, (blurry_path, sharp_path) in enumerate(
            zip(record.blurry, record.sharp)
        ):
            raw = read_rgb_tensor(blurry_path, device=device)
            gt = read_rgb_tensor(sharp_path, device=device)
            if raw.shape != gt.shape:
                raise ValueError(f"raw/GT shape mismatch in {record.name!r} frame {frame_index}")
            padded, height, width = _pad_to_multiple(raw.unsqueeze(0))
            # Deliberately unmeasured: this is the normal stream used for
            # quality and as the reference for the history controls below.
            turtle_padded = backend.step(padded, timestamp=frame_index)
            latency_ms = latencies[global_index]
            turtle = turtle_padded[0, :, :height, :width].clamp(0.0, 1.0)

            sources: Dict[str, torch.Tensor] = {"raw": raw, "turtle": turtle}
            if history_controls:
                ordered_padded, ordered_k, ordered_v = backend.replay_step(
                    padded,
                    k_cache=ordered_k,
                    v_cache=ordered_v,
                    previous_frame=ordered_previous,
                )
                ordered_previous = padded.detach()
                sources["turtle_replayed_ordered"] = ordered_padded[
                    0, :, :height, :width
                ]
                ordered_replay_max_abs = max(
                    ordered_replay_max_abs,
                    float(
                        torch.max(
                            torch.abs(sources["turtle"] - sources["turtle_replayed_ordered"])
                        ).item()
                    ),
                )
                if frame_index in frozen_control_indices:
                    controls = history_control_outputs(
                        backend,
                        padded,
                        past_padded_frames,
                    )
                    for name, value in controls.items():
                        sources[name] = value[0, :, :height, :width]
            if sequence_evssm is not None:
                evssm = read_rgb_tensor(Path(sequence_evssm[frame_index]), device=device)
                if evssm.shape != gt.shape:
                    raise ValueError(
                        f"EVSSM/GT shape mismatch in {record.name!r} frame {frame_index}"
                    )
                sources["evssm"] = evssm
            metrics = {name: image_metrics(image, gt) for name, image in sources.items()}
            temporal = None
            temporal_source_names = ["raw", "turtle"]
            if "turtle_replayed_ordered" in sources:
                temporal_source_names.append("turtle_replayed_ordered")
            if "evssm" in sources:
                temporal_source_names.append("evssm")
            if previous_sources is not None and previous_gt is not None:
                temporal_pairs += 1
                temporal = {
                    name: temporal_metrics(image, previous_sources[name], gt, previous_gt)
                    for name, image in sources.items()
                    if name in temporal_source_names
                }
            row: Dict[str, Any] = {
                "sequence": record.name,
                "frame_index": frame_index,
                "global_index": global_index,
                "raw_path": str(blurry_path),
                "gt_path": str(sharp_path),
                "turtle_latency_ms": latency_ms,
                "history_control_selected": frame_index in frozen_control_indices,
                "metrics": metrics,
                "temporal": temporal,
            }
            if sequence_evssm is not None:
                row["evssm_path"] = str(sequence_evssm[frame_index])
            rows.append(row)

            if global_index in visual_indices:
                items: List[Tuple[str, torch.Tensor]] = [("Raw blurry", raw)]
                if "evssm" in sources:
                    items.append(("EVSSM", sources["evssm"]))
                items.extend((("TURTLE", turtle), ("Sharp GT", gt)))
                scores = " | ".join(
                    f"{name} {metrics[name]['psnr']:.3f} dB"
                    for name in ("raw", "evssm", "turtle")
                    if name in metrics
                )
                write_montage(
                    output_dir
                    / "visuals"
                    / f"{global_index:05d}_{_safe_name(record.name)}_{frame_index:05d}.png",
                    items,
                    subtitle=scores,
                    panel_width=visual_panel_width,
                )

            previous_sources = {
                name: sources[name].detach() for name in temporal_source_names
            }
            previous_gt = gt.detach()
            past_padded_frames.append(padded.detach())
            global_index += 1

    subset_control_names = (
        ["turtle_reset_cache", "turtle_repeat_current", "turtle_shuffled_history"]
        if history_controls
        else []
    )
    persistent_control_names = ["turtle_replayed_ordered"] if history_controls else []
    source_names = ["raw", "turtle"] + persistent_control_names + (
        ["evssm"] if evssm_paths is not None else []
    )
    means = {
        source: {
            metric: _mean([row["metrics"][source][metric] for row in rows])
            for metric in ("psnr", "ssim", "l1")
        }
        for source in source_names
    }
    temporal_means: Dict[str, Dict[str, Optional[float]]] = {}
    for source in source_names:
        temporal_means[source] = {
            metric: _mean(
                [
                    row["temporal"][source][metric]
                    for row in rows
                    if row["temporal"] is not None
                ]
            )
            for metric in (
                "adjacent_change_l1",
                "gt_temporal_difference_error_l1",
            )
        }

    steady_rows = [
        row for row in rows if int(row["frame_index"]) >= FORMAL_STEADY_FRAME_INDEX_MIN
    ]
    steady_means = {
        source: {
            metric: _mean([row["metrics"][source][metric] for row in steady_rows])
            for metric in ("psnr", "ssim", "l1")
        }
        for source in source_names
    }
    per_sequence: Dict[str, Any] = {}
    for record in records:
        sequence_rows = [row for row in rows if row["sequence"] == record.name]
        sequence_steady = [
            row
            for row in sequence_rows
            if int(row["frame_index"]) >= FORMAL_STEADY_FRAME_INDEX_MIN
        ]
        per_sequence[record.name] = {
            "frame_count": len(sequence_rows),
            "steady_frame_count": len(sequence_steady),
            "mean": {
                source: {
                    metric: _mean(
                        [row["metrics"][source][metric] for row in sequence_rows]
                    )
                    for metric in ("psnr", "ssim", "l1")
                }
                for source in source_names
            },
            "steady_mean": {
                source: {
                    metric: _mean(
                        [row["metrics"][source][metric] for row in sequence_steady]
                    )
                    for metric in ("psnr", "ssim", "l1")
                }
                for source in source_names
            },
        }

    all_latency = {
        "mean": _mean(latencies),
        "median": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
        "max": max(latencies) if latencies else None,
        "frames": len(latencies),
    }
    steady_latencies = [
        float(row["turtle_latency_ms"]) for row in steady_rows
    ]
    steady_latency = {
        "mean": _mean(steady_latencies),
        "median": _percentile(steady_latencies, 50),
        "p95": _percentile(steady_latencies, 95),
        "max": max(steady_latencies) if steady_latencies else None,
        "frames": len(steady_latencies),
    }
    raw_baseline = {
        "registration": {
            "source": "decoded_blurry_RGB_tensor_scored_against_same_sharp_target",
            "per_frame_rows_present": True,
            "shared_metric_implementation": "evaluate_turtle_streaming.image_metrics",
            "aggregation": "arithmetic_mean_of_per_frame_metrics",
            "steady_frame_index_min": FORMAL_STEADY_FRAME_INDEX_MIN,
        },
        "all_frames": means["raw"],
        "steady": steady_means["raw"],
        "per_sequence": {
            name: {
                "all_frames": values["mean"]["raw"],
                "steady": values["steady_mean"]["raw"],
                "frame_count": values["frame_count"],
                "steady_frame_count": values["steady_frame_count"],
            }
            for name, values in per_sequence.items()
        },
    }
    turtle_minus_raw = {
        "all_frames": _metric_delta(means["turtle"], means["raw"]),
        "steady": _metric_delta(steady_means["turtle"], steady_means["raw"]),
        "per_sequence": {
            name: {
                "all_frames": _metric_delta(
                    values["mean"]["turtle"], values["mean"]["raw"]
                ),
                "steady": _metric_delta(
                    values["steady_mean"]["turtle"],
                    values["steady_mean"]["raw"],
                ),
            }
            for name, values in per_sequence.items()
        },
    }
    summary: Dict[str, Any] = {
        "schema": "unblur_slam.turtle_streaming_evaluation.v1",
        "frame_count": len(rows),
        "sequence_count": len(records),
        "temporal_pair_count": temporal_pairs,
        "sources": source_names + subset_control_names + ["gt"],
        "mean": means,
        "steady_mean": steady_means,
        "per_sequence": per_sequence,
        "raw_baseline": raw_baseline,
        "model_minus_raw": turtle_minus_raw,
        "turtle_minus_raw": {
            metric: turtle_minus_raw["all_frames"][metric]
            for metric in ("psnr", "ssim", "l1")
        },
        "temporal": {
            "protocol": {
                "optical_flow_warp_used": False,
                "adjacent_change_l1": (
                    "raw adjacent-frame L1; includes genuine camera/object motion"
                ),
                "gt_temporal_difference_error_l1": (
                    "L1 between predicted and GT adjacent differences; not flow-warped"
                ),
            },
            "mean": temporal_means,
        },
        "performance": {
            "warmup": {
                "unmeasured_calls": warmup_steps,
                "input": "first_validation_blurry_only",
                "target_or_metric_used": False,
                "stream_reset_after_warmup": True,
            },
            "latency_scope": (
                "one normal TURTLE model/backend step in a dedicated timing-only "
                "ordered-stream pass with pre/post device synchronization; excludes "
                "image decode, padding, sharp-target access, quality forwards, "
                "metrics, reporting, and every history-control/replay forward"
            ),
            "pass_separation": {
                "timing_pass": {
                    "normal_stream_model_steps": len(rows),
                    "sharp_target_images_opened": False,
                    "metrics_computed": False,
                    "history_or_replay_control_forwards": 0,
                },
                "quality_history_pass": {
                    "normal_stream_model_steps": len(rows),
                    "timed_model_steps": 0,
                    "history_controls_enabled": bool(history_controls),
                },
                "passes_are_distinct_complete_dataset_traversals": True,
            },
            "turtle_latency_ms": all_latency,
            "steady_turtle_latency_ms": steady_latency,
            "history_controls_timed": False,
            "quality_history_pass_timed": False,
            "peak_cuda_memory_allocated_bytes": peak_memory,
            "peak_memory_scope": "dedicated_timing_only_normal_stream_pass",
            "compute_precision": {
                "backend_inference_precision": str(
                    getattr(backend, "inference_precision", "unspecified")
                ),
                "normal_and_control_forward_autocast": (
                    "CUDA_FP16"
                    if getattr(backend, "inference_precision", None) == "fp16"
                    else "disabled_FP32"
                ),
                "cuda_matmul_allow_tf32": (
                    bool(torch.backends.cuda.matmul.allow_tf32)
                    if device.type == "cuda"
                    else None
                ),
                "cudnn_allow_tf32": (
                    bool(torch.backends.cudnn.allow_tf32)
                    if device.type == "cuda"
                    else None
                ),
            },
        },
        "checkpoint_metadata": dict(checkpoint_metadata or {}),
        "provenance": dict(provenance or {}),
        "frames": rows,
    }
    if history_controls:
        control_rows = [
            row
            for row in rows
            if row.get("history_control_selected") is True
            and int(row["frame_index"]) >= FORMAL_STEADY_FRAME_INDEX_MIN
        ]
        control_means = {
            source: {
                metric: _mean(
                    [row["metrics"][source][metric] for row in control_rows]
                )
                for metric in ("psnr", "ssim", "l1")
            }
            for source in ["turtle"] + subset_control_names
        }
        control_per_sequence: Dict[str, Any] = {}
        for record in records:
            selected = [row for row in control_rows if row["sequence"] == record.name]
            control_per_sequence[record.name] = {
                "frame_indices": [int(row["frame_index"]) for row in selected],
                "frame_count": len(selected),
                "mean": {
                    source: {
                        metric: _mean(
                            [row["metrics"][source][metric] for row in selected]
                        )
                        for metric in ("psnr", "ssim", "l1")
                    }
                    for source in ["turtle"] + subset_control_names
                },
            }
        prefix_control_forwards = sum(
            int(row["frame_index"]) + 1 for row in control_rows
        )
        quality_history_forward_total = (
            2 * len(rows) + len(control_rows) + 2 * prefix_control_forwards
        )
        combined_two_pass_forward_total = len(rows) + quality_history_forward_total
        summary["performance"]["pass_separation"]["quality_history_pass"].update(
            {
                "total_model_steps_including_controls": quality_history_forward_total,
            }
        )
        summary["performance"]["pass_separation"][
            "forward_accounting_excluding_warmup"
        ] = {
            "timing_only_model_steps": len(rows),
            "quality_history_model_steps": quality_history_forward_total,
            "combined_model_steps": combined_two_pass_forward_total,
        }
        summary["history_ablation"] = {
            "protocol": {
                "cache_source": "official TURTLE K/V only; use_both_input=false",
                "cache_slots_per_kind": 8,
                "populated_cache_slots_per_kind": 5,
                "direct_cache_capacity_frames": [3, 3, 3, 3, 2],
                "effective_history": "recurrent_full_prefix",
                "ordered_control": (
                    "independent_incremental_full_stream_replay_at_every_frame"
                ),
                "repeat_current_control": (
                    "reset_then_replay_current_once_per_complete_past_frame"
                ),
                "shuffled_control": "cyclic_left_shift_of_complete_past_prefix",
                "steady_frame_index_min": FORMAL_STEADY_FRAME_INDEX_MIN,
                "expensive_control_frame_indices_per_sequence": list(
                    frozen_control_indices
                ),
                "expensive_controls_aggregation": (
                    "unweighted arithmetic frame mean over the frozen 6xsequence subset"
                ),
                "future_frames_used": False,
                "normal_and_all_controls_share_backend_autocast_path": True,
                "controls_in_normal_latency": False,
            },
            "ordered_replay_max_abs": ordered_replay_max_abs,
            "ordered_replay_matches_stream": ordered_replay_max_abs <= 1.0e-6,
            "ordered_replay_frame_count": len(rows),
            "control_frame_count": len(control_rows),
            "forward_accounting_excluding_warmup": {
                "dedicated_timing_normal_stream": len(rows),
                "quality_normal_stream": len(rows),
                # Backward-compatible name for the normal stream inside the
                # quality/history pass (the frozen 16,280-forward budget).
                "normal_stream": len(rows),
                "incremental_ordered_replay": len(rows),
                "reset": len(control_rows),
                "repeat_prefix_plus_current": sum(
                    int(row["frame_index"]) + 1 for row in control_rows
                ),
                "shuffled_prefix_plus_current": sum(
                    int(row["frame_index"]) + 1 for row in control_rows
                ),
                "total": quality_history_forward_total,
                "total_including_dedicated_timing_pass": combined_two_pass_forward_total,
                "timing_pass_history_or_replay_controls": 0,
            },
            "control_mean": control_means,
            "control_per_sequence": control_per_sequence,
            "steady_normal_minus_control": {
                control: {
                    "psnr": control_means["turtle"]["psnr"]
                    - control_means[control]["psnr"],
                    "ssim": control_means["turtle"]["ssim"]
                    - control_means[control]["ssim"],
                    "l1": control_means["turtle"]["l1"]
                    - control_means[control]["l1"],
                }
                for control in subset_control_names
            },
        }
    if "evssm" in means:
        summary["turtle_minus_evssm"] = {
            metric: means["turtle"][metric] - means["evssm"][metric]
            for metric in ("psnr", "ssim", "l1")
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    evssm = parser.add_mutually_exclusive_group()
    evssm.add_argument("--evssm-manifest", type=Path)
    evssm.add_argument("--evssm-precompute-report", type=Path)
    parser.add_argument("--turtle-repo", type=Path, default=DEFAULT_TURTLE_REPO)
    parser.add_argument("--turtle-config", type=Path, default=DEFAULT_TURTLE_CONFIG)
    parser.add_argument(
        "--turtle-checkpoint", type=Path, default=DEFAULT_TURTLE_CHECKPOINT
    )
    parser.add_argument("--turtle-checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-visuals", type=int, default=12)
    parser.add_argument("--visual-panel-width", type=int, default=480)
    parser.add_argument("--history-controls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_visuals < 0 or args.visual_panel_width < 0:
        raise ValueError("visual counts and widths must be non-negative")
    output_dir = prepare_output_directory(args.output_dir)
    device = choose_device(args.device)
    manifest = args.manifest.expanduser().resolve()
    records = load_sequence_manifest(manifest, root=args.data_root)

    evssm_manifest = args.evssm_manifest
    evssm_provenance: Optional[Dict[str, Any]] = None
    if args.evssm_precompute_report is not None:
        report = load_evssm_precompute_report(
            str(args.evssm_precompute_report), verify_teacher_artifacts=True
        )
        evssm_manifest = Path(str(report["teacher_manifest"]))
        evssm_provenance = dict(report)
    evssm_paths, evssm_manifest_provenance = resolve_evssm_paths(
        records, evssm_manifest=evssm_manifest, data_root=args.data_root
    )
    if evssm_provenance is None:
        evssm_provenance = evssm_manifest_provenance

    checkpoint = args.turtle_checkpoint.expanduser().resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    configured_sha = args.turtle_checkpoint_sha256
    if checkpoint_sha256 != PINNED_TURTLE_CHECKPOINT_SHA256:
        if configured_sha is None:
            raise ValueError(
                "fine-tuned TURTLE evaluation requires --turtle-checkpoint-sha256 "
                f"{checkpoint_sha256}"
            )
        if configured_sha.lower() != checkpoint_sha256:
            raise ValueError("--turtle-checkpoint-sha256 does not match checkpoint bytes")
    model, checkpoint_metadata = load_turtle_model(
        args.turtle_repo,
        checkpoint,
        config=args.turtle_config,
        device=device,
        checkpoint_sha256=configured_sha,
    )
    if args.history_controls and bool(getattr(model, "use_both_input", True)):
        raise ValueError("history controls require pinned TURTLE use_both_input=false")
    backend = TurtleStreamingBackend(model, device=device)
    provenance = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "evssm": evssm_provenance,
        "device": str(device),
    }
    summary = evaluate_sequences(
        records,
        backend,
        device=device,
        output_dir=output_dir,
        max_visuals=args.max_visuals,
        visual_panel_width=args.visual_panel_width,
        evssm_paths=evssm_paths,
        checkpoint_metadata=checkpoint_metadata,
        provenance=provenance,
        history_controls=args.history_controls,
    )
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "metrics": str(metrics_path),
                "frame_count": summary["frame_count"],
                "mean": summary["mean"],
                "performance": summary["performance"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
