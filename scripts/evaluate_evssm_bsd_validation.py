#!/usr/bin/env python3
"""Direct-float official EVSSM evaluation on BSD 3ms24ms validation.

No prediction is serialized to PNG before scoring.  The one-frame EVSSM call
is timed with device synchronization after one target-independent warm-up;
decode, metric calculation, reporting, and all targets are outside that scope.
The entry point has no test-split argument.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import (  # noqa: E402
    inspect_bsd_sequence_manifest,
    load_contract,
    sha256_file,
    validate_protocol,
)
from scripts.bsd_dpdd_runtime import require_gpu1_a6000  # noqa: E402
from scripts.evaluate_turtle_streaming import (  # noqa: E402
    FORMAL_STEADY_FRAME_INDEX_MIN,
    FORMAL_WARMUP_STEPS,
    _mean,
    _metric_delta,
    _pad_to_multiple,
    _percentile,
    _synchronize,
    image_metrics,
)
from scripts.train_turtle_streaming import (  # noqa: E402
    choose_device,
    load_sequence_manifest,
    read_rgb_tensor,
)
from src.deblur_backends import EVSSMBackend  # noqa: E402


SCHEMA = "unblur_slam.evssm_bsd3ms24ms_direct_float_validation.v1"


def _load_official_evssm(
    definition: Mapping[str, Any], *, device: torch.device
) -> tuple[EVSSMBackend, Mapping[str, Any]]:
    checkpoint = Path(str(definition["checkpoint"])).expanduser().resolve()
    architecture = Path(str(definition["architecture"])).expanduser().resolve()
    backend_source = Path(str(definition["backend_source"])).expanduser().resolve()
    if sha256_file(checkpoint) != definition["checkpoint_sha256"]:
        raise ValueError("official EVSSM checkpoint identity changed")
    if sha256_file(architecture) != definition["architecture_sha256"]:
        raise ValueError("official EVSSM architecture identity changed")
    if sha256_file(backend_source) != definition["backend_source_sha256"]:
        raise ValueError("official EVSSM backend source identity changed")

    from thirdparty.EVSSM.models.EVSSM import EVSSM

    model = EVSSM().to(device)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("official EVSSM checkpoint payload must be a mapping")
    state = payload.get("params", payload)
    if not isinstance(state, Mapping):
        raise ValueError("official EVSSM checkpoint has no params mapping")
    incompatible = model.load_state_dict(dict(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("official EVSSM strict state load failed")
    model.eval().requires_grad_(False)
    return EVSSMBackend(model, device), {
        "kind": "official_unblur_slam_evssm",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": definition["checkpoint_sha256"],
        "architecture": str(architecture),
        "architecture_sha256": definition["architecture_sha256"],
        "backend_source": str(backend_source),
        "backend_source_sha256": definition["backend_source_sha256"],
        "strict_load_missing": 0,
        "strict_load_unexpected": 0,
        "precision": "CUDA_FP32",
        "autocast": "disabled",
        "tf32_runtime_flags_recorded_in_results": True,
    }


def _fresh_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite EVSSM output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _latency(values: Sequence[float]) -> Mapping[str, Any]:
    return {
        "mean": _mean(values),
        "median": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values),
        "frames": len(values),
    }


@torch.no_grad()
def evaluate(
    records: Sequence[Any],
    backend: EVSSMBackend,
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    if not records:
        raise ValueError("EVSSM BSD validation cannot be empty")
    # Warm-up consumes blurry RGB only; its output and latency are discarded.
    warmup = read_rgb_tensor(records[0].blurry[0], device=device).unsqueeze(0)
    _synchronize(device)
    warmup_output = backend(warmup, timestamp=0)
    _synchronize(device)
    if tuple(warmup_output.shape) != tuple(warmup.shape):
        raise RuntimeError("EVSSM warm-up changed image shape")

    # Dedicated timing-only traversal: blurry RGB is decoded before the timer;
    # no sharp target is opened and no metric/quality work is interleaved
    # between synchronized latency samples.
    latencies: list[float] = []
    for record in records:
        for frame_index, blurry_path in enumerate(record.blurry):
            blurry = read_rgb_tensor(blurry_path, device=device)
            _synchronize(device)
            started = time.perf_counter()
            restored = backend(blurry.unsqueeze(0), timestamp=frame_index)
            _synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            prediction = restored[0]
            if tuple(prediction.shape) != tuple(blurry.shape) or not bool(
                torch.isfinite(prediction).all()
            ):
                raise RuntimeError("EVSSM timing pass returned an invalid image")
            latencies.append(latency_ms)

    # Independent quality traversal.  The model is stateless, so no history
    # reset/control applies; all forwards in this pass are deliberately
    # unmeasured.
    rows: list[dict[str, Any]] = []
    global_index = 0
    for record in records:
        for frame_index, (blurry_path, sharp_path) in enumerate(
            zip(record.blurry, record.sharp)
        ):
            blurry = read_rgb_tensor(blurry_path, device=device)
            sharp = read_rgb_tensor(sharp_path, device=device)
            if tuple(blurry.shape) != tuple(sharp.shape):
                raise ValueError("BSD blurry/sharp RGB shapes differ")
            restored = backend(blurry.unsqueeze(0), timestamp=frame_index)
            prediction = restored[0]
            if tuple(prediction.shape) != tuple(sharp.shape) or not bool(
                torch.isfinite(prediction).all()
            ):
                raise RuntimeError("EVSSM quality pass returned an invalid direct-float image")
            raw_metric = image_metrics(blurry, sharp)
            evssm_metric = image_metrics(prediction, sharp)
            rows.append(
                {
                    "sequence": record.name,
                    "frame_index": frame_index,
                    "global_index": global_index,
                    "raw_path": str(blurry_path),
                    "gt_path": str(sharp_path),
                    "blurry_path": str(blurry_path),
                    "sharp_path": str(sharp_path),
                    "raw_metrics": raw_metric,
                    "metrics": evssm_metric,
                    "evssm_latency_ms": latencies[global_index],
                }
            )
            global_index += 1
    if len(latencies) != len(rows):
        raise RuntimeError("EVSSM timing/quality coverage differs")
    metric_names = ("psnr", "ssim", "l1")
    mean = {
        metric: _mean([row["metrics"][metric] for row in rows])
        for metric in metric_names
    }
    raw_mean = {
        metric: _mean([row["raw_metrics"][metric] for row in rows])
        for metric in metric_names
    }
    steady = [
        row
        for row in rows
        if int(row["frame_index"]) >= FORMAL_STEADY_FRAME_INDEX_MIN
    ]
    steady_mean = {
        metric: _mean([row["metrics"][metric] for row in steady])
        for metric in metric_names
    }
    raw_steady_mean = {
        metric: _mean([row["raw_metrics"][metric] for row in steady])
        for metric in metric_names
    }
    per_sequence: dict[str, Any] = {}
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
                metric: _mean([row["metrics"][metric] for row in sequence_rows])
                for metric in metric_names
            },
            "steady_mean": {
                metric: _mean(
                    [row["metrics"][metric] for row in sequence_steady]
                )
                for metric in metric_names
            },
            "raw_mean": {
                metric: _mean(
                    [row["raw_metrics"][metric] for row in sequence_rows]
                )
                for metric in metric_names
            },
            "raw_steady_mean": {
                metric: _mean(
                    [row["raw_metrics"][metric] for row in sequence_steady]
                )
                for metric in metric_names
            },
        }
    steady_latencies = [
        float(row["evssm_latency_ms"])
        for row in steady
    ]
    raw_per_sequence = {
        name: {
            "all_frames": values["raw_mean"],
            "steady": values["raw_steady_mean"],
            "frame_count": values["frame_count"],
            "steady_frame_count": values["steady_frame_count"],
        }
        for name, values in per_sequence.items()
    }
    minus_raw_per_sequence = {
        name: {
            "all_frames": _metric_delta(values["mean"], values["raw_mean"]),
            "steady": _metric_delta(
                values["steady_mean"], values["raw_steady_mean"]
            ),
        }
        for name, values in per_sequence.items()
    }
    return {
        "frame_count": len(rows),
        "sequence_count": len(records),
        "mean": mean,
        "steady_mean": steady_mean,
        "per_sequence": per_sequence,
        "raw_baseline": {
            "registration": {
                "source": "decoded_blurry_RGB_tensor_scored_against_same_sharp_target",
                "per_frame_rows_present": True,
                "shared_metric_implementation": "evaluate_turtle_streaming.image_metrics",
                "aggregation": "arithmetic_mean_of_per_frame_metrics",
                "steady_frame_index_min": FORMAL_STEADY_FRAME_INDEX_MIN,
            },
            "all_frames": raw_mean,
            "steady": raw_steady_mean,
            "per_sequence": raw_per_sequence,
        },
        "model_minus_raw": {
            "all_frames": _metric_delta(mean, raw_mean),
            "steady": _metric_delta(steady_mean, raw_steady_mean),
            "per_sequence": minus_raw_per_sequence,
        },
        "performance": {
            "warmup": {
                "unmeasured_calls": FORMAL_WARMUP_STEPS,
                "input": "first_validation_blurry_only",
                "target_or_metric_used": False,
                "output_and_latency_discarded": True,
            },
            "latency_scope": (
                "one EVSSM FP32 model/backend call in a dedicated timing-only "
                "stateless pass with pre/post CUDA synchronization; excludes image "
                "decode, target access, quality forwards, metrics, reporting, and SLAM"
            ),
            "pass_separation": {
                "timing_pass": {
                    "stateless_model_steps": len(rows),
                    "sharp_target_images_opened": False,
                    "metrics_computed": False,
                    "history_or_replay_control_forwards": 0,
                },
                "quality_pass": {
                    "stateless_model_steps": len(rows),
                    "timed_model_steps": 0,
                },
                "passes_are_distinct_complete_dataset_traversals": True,
                "forward_accounting_excluding_warmup": {
                    "timing_only_model_steps": len(rows),
                    "quality_model_steps": len(rows),
                    "combined_model_steps": 2 * len(rows),
                },
            },
            "evssm_latency_ms": _latency(latencies),
            "steady_evssm_latency_ms": _latency(steady_latencies),
            "compute_precision": {
                "model_and_input": "CUDA_FP32",
                "autocast": "disabled",
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
        "frames": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-limit", type=int, choices=(0, 2), default=0)
    parser.add_argument("--reference-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract_path, contract, contract_sha = load_contract(
        args.contract, expected_sha256=args.expected_contract_sha256
    )
    validate_protocol(
        contract, allow_template=False, reference_only=args.reference_only
    )
    runtime = require_gpu1_a6000(args.device)
    bsd = contract["data"]["bsd"]
    inventory = inspect_bsd_sequence_manifest(
        bsd["validation_manifest"],
        dataset_root=bsd["dataset_root"],
        expected_sha256=bsd["validation_manifest_sha256"],
        expected_split="validation",
        expected_sequences=20,
        expected_frames=2000,
        expected_per_exposure_sequences=20,
        require_assets=True,
        verify_content=True,
    )
    records = load_sequence_manifest(inventory.path, root=Path(bsd["dataset_root"]))
    if args.sequence_limit:
        records = records[: args.sequence_limit]
    device = choose_device(args.device)
    backend, model_identity = _load_official_evssm(
        contract["models"]["evssm_E"], device=device
    )
    evaluated = evaluate(records, backend, device=device)
    expected_sequences = args.sequence_limit or 20
    expected_frames = expected_sequences * 100
    if (
        evaluated["sequence_count"] != expected_sequences
        or evaluated["frame_count"] != expected_frames
    ):
        raise RuntimeError("EVSSM BSD validation coverage changed")
    passes = evaluated.get("performance", {}).get("pass_separation", {})
    if (
        passes.get("timing_pass", {}).get("stateless_model_steps") != expected_frames
        or passes.get("timing_pass", {}).get("sharp_target_images_opened") is not False
        or passes.get("timing_pass", {}).get("metrics_computed") is not False
        or passes.get("timing_pass", {}).get("history_or_replay_control_forwards") != 0
        or passes.get("quality_pass", {}).get("stateless_model_steps") != expected_frames
        or passes.get("quality_pass", {}).get("timed_model_steps") != 0
    ):
        raise RuntimeError("EVSSM dedicated timing/quality pass separation changed")
    raw = evaluated.get("raw_baseline", {})
    if (
        raw.get("registration", {}).get("per_frame_rows_present") is not True
        or set(raw.get("per_sequence", {})) != {record.name for record in records}
    ):
        raise RuntimeError("EVSSM raw-baseline registration changed")
    payload = {
        "schema": SCHEMA,
        "formal": True,
        "arm": "E",
        "interpretation": (
            "external stateless single-frame restoration-module reference; not a "
            "same-method arm and no causal-history or SLAM claim"
        ),
        "protocol": {
            "contract": str(contract_path),
            "contract_sha256": contract_sha,
            "manifest": str(inventory.path),
            "manifest_sha256": inventory.sha256,
            "selected_split": "validation",
            "sequence_limit": args.sequence_limit,
            "formal_full_validation": args.sequence_limit == 0,
            "prediction_representation_for_metrics": "direct_float32_tensor_no_png_roundtrip",
            "metrics": "per-frame RGB PSNR/SSIM/L1 then arithmetic frame mean",
            "raw_common_baseline_registered": True,
            "reference_role": "external_stateless_single_frame_EVSSM_reference",
            "history_state": "none",
            "claim_scope": "restoration_module_only",
            "slam_quality_or_speed_claim": False,
            "bsd_test_pixels_opened": False,
        },
        "runtime": dict(runtime),
        "model": dict(model_identity),
        "results": evaluated,
    }
    output = _fresh_output(args.output_dir)
    report = output / "metrics.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(report, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        report.unlink(missing_ok=True)
        raise
    print(json.dumps({"metrics": str(report), "arm": "E", "frames": expected_frames}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
