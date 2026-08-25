#!/usr/bin/env python3
"""Paired FP32/FP16 runtime probe for the pinned official TURTLE backend.

This is a systems benchmark only.  It reuses already materialized 512x384 RGB
frames and never opens ground-truth images or computes deblurring quality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.turtle_backend import (  # noqa: E402
    TurtleStreamingBackend,
    build_turtle_model,
    sha256_file,
    validate_turtle_artifacts,
)


SCHEMA = "unblur_slam.turtle_runtime_precision_benchmark.v1"


def _load_frames(manifest_path: Path, count: int) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("frames")
    if not isinstance(records, list) or len(records) < count:
        raise ValueError(f"manifest must contain at least {count} frame records")
    tensors: list[torch.Tensor] = []
    provenance: list[dict[str, Any]] = []
    for record in records[:count]:
        output = record.get("output") if isinstance(record, dict) else None
        if not isinstance(output, dict):
            raise ValueError("every manifest frame must contain an output object")
        path = Path(str(output.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(output.get("sha256", "")).lower()
        actual = sha256_file(path)
        if expected != actual:
            raise ValueError(f"materialized frame SHA mismatch: {path}")
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        if pixels.shape != (384, 512, 3):
            raise ValueError(f"benchmark frame must be 512x384 RGB: {path}")
        tensors.append(torch.from_numpy(pixels / 255.0).permute(2, 0, 1).unsqueeze(0))
        provenance.append(
            {
                "source_index": int(record["source_index"]),
                "path": str(path),
                "sha256": actual,
            }
        )
    return tensors, provenance


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _run(
    model: torch.nn.Module,
    frames: list[torch.Tensor],
    *,
    device: str,
    precision: str,
    warmup: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    backend = TurtleStreamingBackend(
        model,
        device=device,
        inference_precision=precision,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    outputs: list[torch.Tensor] = []
    elapsed_ms: list[float] = []
    for index, frame in enumerate(frames):
        torch.cuda.synchronize()
        started = time.perf_counter()
        restored = backend.step(frame, timestamp=index)
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        outputs.append(restored.cpu())
    steady = elapsed_ms[warmup:]
    return outputs, {
        "precision": precision,
        "frames_total": len(frames),
        "warmup_frames_excluded": warmup,
        "frames_timed_steady": len(steady),
        "mean_ms": statistics.fmean(steady),
        "median_ms": statistics.median(steady),
        "p95_ms": _percentile(steady, 0.95),
        "max_ms": max(steady),
        "fps_from_mean": 1000.0 / statistics.fmean(steady),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "backend_state": dict(backend.state_info()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("paired precision benchmark requires a CUDA device")
    if args.frames < 4 or not 0 <= args.warmup < args.frames:
        raise ValueError("require frames>=4 and 0<=warmup<frames")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    config = {
        "turtle_repo": str(args.repo),
        "turtle_config": str(args.config),
        "turtle_checkpoint": str(args.checkpoint),
    }
    artifacts = validate_turtle_artifacts(config, load_weights=False)
    frames, frame_provenance = _load_frames(args.manifest.resolve(), args.frames)
    model = build_turtle_model(artifacts, device=args.device)
    fp32_outputs, fp32 = _run(
        model,
        frames,
        device=args.device,
        precision="fp32",
        warmup=args.warmup,
    )
    fp16_outputs, fp16 = _run(
        model,
        frames,
        device=args.device,
        precision="fp16",
        warmup=args.warmup,
    )

    max_abs = 0.0
    psnr_values: list[float] = []
    exact_match_frames = 0
    for reference, candidate in zip(fp32_outputs, fp16_outputs):
        max_abs = max(max_abs, float((reference - candidate).abs().max()))
        mse = float(((reference - candidate) ** 2).mean())
        if mse == 0.0:
            exact_match_frames += 1
        else:
            psnr_values.append(-10.0 * np.log10(mse))

    report = {
        "schema": SCHEMA,
        "artifact_class": "runtime_only_no_ground_truth",
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest.resolve()),
        },
        "frames": frame_provenance,
        "official_turtle": {
            "repo": str(artifacts.repo),
            "commit": artifacts.commit,
            "architecture_sha256": artifacts.architecture_sha256,
            "config_sha256": artifacts.config_sha256,
            "checkpoint_sha256": artifacts.checkpoint_sha256,
        },
        "runtime": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": args.device,
            "gpu_name": torch.cuda.get_device_name(torch.device(args.device)),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "fp32": fp32,
        "fp16": fp16,
        "fp16_vs_fp32": {
            "max_abs": max_abs,
            "exact_match_frames": exact_match_frames,
            "mean_psnr_db_nonexact": (
                statistics.fmean(psnr_values) if psnr_values else None
            ),
            "min_psnr_db_nonexact": min(psnr_values) if psnr_values else None,
            "all_outputs_finite": all(
                bool(torch.isfinite(value).all()) for value in fp16_outputs
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
