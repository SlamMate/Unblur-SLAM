#!/usr/bin/env python3
"""Formal DPDD-validation evaluator for TURTLE arms G, O, B, and BD.

Each of the 74 validation images begins with empty K/V and discards returned
state.  The script has no test option and reports current-frame spatial quality
only; useful history must be established on ordered BSD video controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import SEEDS, load_contract, validate_protocol  # noqa: E402
from scripts.bsd_dpdd_runtime import require_gpu1_a6000  # noqa: E402
from scripts.evaluate_turtle_bsd_streaming import validate_trained_metadata  # noqa: E402
from scripts.evaluate_turtle_single_image_defocus import (  # noqa: E402
    _metrics,
    _load_lpips_metric,
    evaluate_raw_pairs,
    load_dpdd_evaluation_dataset_contract,
    load_single_image_manifest,
    paired_arm_delta,
    pad_to_multiple,
    prepare_output_directory,
    read_pair_tensor,
    summarize_rows,
)
from scripts.train_turtle_streaming import choose_device  # noqa: E402
from src.turtle_backend import (  # noqa: E402
    PINNED_TURTLE_CHECKPOINT_SHA256,
    TurtleStreamingBackend,
    load_turtle_model,
    sha256_file,
)
from src.turtle_official_bsd_backend import (  # noqa: E402
    PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
    load_official_bsd_turtle_model,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_timing_pass(
    records: Sequence[Any],
    backend: Any,
    *,
    device: torch.device,
    padding_multiple: int = 8,
    require_png_rgb16: bool = True,
) -> list[float]:
    """Time only reset current-frame model steps, without opening targets."""

    if not records:
        raise ValueError("cannot time an empty DPDD validation set")
    first = read_pair_tensor(
        records[0].blurry,
        device=device,
        require_png_rgb16=require_png_rgb16,
    ).unsqueeze(0)
    first, _, _, _, _ = pad_to_multiple(first, padding_multiple)
    backend.reset()
    _synchronize(device)
    backend.step(first, timestamp=0)
    _synchronize(device)
    backend.reset()

    latencies: list[float] = []
    for record in records:
        blurry = read_pair_tensor(
            record.blurry, device=device, require_png_rgb16=require_png_rgb16
        )
        padded, _, _, _, _ = pad_to_multiple(
            blurry.unsqueeze(0), padding_multiple
        )
        backend.reset()
        require_empty = dict(backend.state_info())
        if bool(require_empty.get("has_cache")):
            raise RuntimeError(f"pair {record.name!r} timing pass began with K/V")
        _synchronize(device)
        started = time.perf_counter()
        backend.step(padded, timestamp=0)
        _synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


@torch.no_grad()
def evaluate_quality_pass(
    records: Sequence[Any],
    backend: Any,
    *,
    device: torch.device,
    lpips_metric: Any,
    latencies: Sequence[float],
    padding_multiple: int = 8,
    require_png_rgb16: bool = True,
) -> list[dict[str, Any]]:
    """Run the independent, wholly unmeasured DPDD quality traversal."""

    if len(records) != len(latencies):
        raise ValueError("DPDD timing/quality coverage differs")
    rows: list[dict[str, Any]] = []
    for index, (record, latency_ms) in enumerate(zip(records, latencies)):
        blurry = read_pair_tensor(
            record.blurry, device=device, require_png_rgb16=require_png_rgb16
        )
        sharp = read_pair_tensor(
            record.sharp, device=device, require_png_rgb16=require_png_rgb16
        )
        if blurry.shape != sharp.shape:
            raise ValueError(f"pair {record.name!r} changed shape after RGB decoding")
        padded, height, width, pad_height, pad_width = pad_to_multiple(
            blurry.unsqueeze(0), padding_multiple
        )
        backend.reset()
        if bool(backend.state_info().get("has_cache")):
            raise RuntimeError(f"pair {record.name!r} quality pass began with K/V")
        restored_padded = backend.step(padded, timestamp=0)
        if not bool(backend.state_info().get("has_cache")):
            raise RuntimeError(f"pair {record.name!r} did not produce official K/V")
        restored = restored_padded[0, :, :height, :width]
        rows.append(
            {
                "index": index,
                "name": record.name,
                "blurry_path": str(record.blurry),
                "sharp_path": str(record.sharp),
                "source_sha256": record.source_sha256,
                "target_sha256": record.target_sha256,
                "source_height": height,
                "source_width": width,
                "pad_bottom": pad_height,
                "pad_right": pad_width,
                "cache_empty_before_call": True,
                "cache_populated_after_call": True,
                "metrics": _metrics(restored, sharp, lpips_metric),
                "latency_ms": float(latency_ms),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--arm", choices=("G", "O", "B", "BD"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lpips-device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path, contract, contract_sha = load_contract(
        args.contract,
        expected_sha256=args.expected_contract_sha256,
    )
    validate_protocol(contract, allow_template=False)
    runtime = require_gpu1_a6000(args.device)
    dpdd = contract["data"]["dpdd"]
    manifest = Path(dpdd["validation_manifest"]).expanduser().resolve()
    dataset_provenance = load_dpdd_evaluation_dataset_contract(
        Path(dpdd["dataset_manifest"]),
        expected_dataset_manifest_sha256=dpdd["dataset_manifest_sha256"],
        validation_manifest=manifest,
        expected_validation_manifest_sha256=dpdd["validation_manifest_sha256"],
    )
    records = load_single_image_manifest(
        manifest,
        root=Path(dpdd["root"]),
        expected_split="validation",
        canonical_contract=True,
        verify_content=True,
    )
    if len(records) != 74:
        raise ValueError("formal DPDD validation requires exactly 74 pairs")
    device = choose_device(args.device)
    if device.type != "cuda":
        raise ValueError("formal DPDD evaluation requires CUDA FP16")
    output = prepare_output_directory(args.output_dir)

    models = contract["models"]
    if args.arm in {"G", "O"}:
        if args.seed is not None or args.checkpoint is not None or args.checkpoint_sha256 is not None:
            raise ValueError("frozen G/O identity comes only from the bound contract")
        definition = models[f"turtle_{args.arm}"]
        checkpoint = Path(definition["checkpoint"])
        checkpoint_sha = definition["checkpoint_sha256"]
    else:
        if args.seed not in SEEDS or args.checkpoint is None or args.checkpoint_sha256 is None:
            raise ValueError("formal B/BD requires seed, checkpoint, and SHA256")
        checkpoint = args.checkpoint.expanduser().resolve()
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != args.checkpoint_sha256.lower():
            raise ValueError("B/BD checkpoint SHA256 mismatch")

    if args.arm == "O":
        if checkpoint_sha != PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256:
            raise ValueError("O checkpoint identity changed")
        model, metadata = load_official_bsd_turtle_model(
            checkpoint,
            device=device,
            checkpoint_sha256=checkpoint_sha,
        )
    else:
        if args.arm == "G" and checkpoint_sha != PINNED_TURTLE_CHECKPOINT_SHA256:
            raise ValueError("G checkpoint identity changed")
        model, metadata = load_turtle_model(
            models["turtle_shared_repo"]["repo"],
            checkpoint,
            config=models["turtle_G"]["config"],
            device=device,
            checkpoint_sha256=(None if args.arm == "G" else checkpoint_sha),
        )
        if args.arm in {"B", "BD"}:
            validate_trained_metadata(
                metadata,
                arm=args.arm,
                seed=int(args.seed),
                contract_sha256=contract_sha,
                bsd_train_sha256=contract["data"]["bsd"]["train_manifest_sha256"],
                dpdd_train_sha256=dpdd["train_manifest_sha256"],
            )
    if bool(getattr(model, "use_both_input", True)):
        raise ValueError("independent-image reset protocol requires use_both_input=false")
    backend = TurtleStreamingBackend(model, device=device, inference_precision="fp16")
    # Run the complete timing-only pass before loading LPIPS or opening any
    # sharp target.  Quality and raw-baseline work happens in pass two.
    latencies = evaluate_timing_pass(
        records,
        backend,
        device=device,
        padding_multiple=8,
        require_png_rgb16=True,
    )
    if len(latencies) != 74:
        raise RuntimeError("formal DPDD timing-pass coverage changed")
    lpips_device = torch.device(args.lpips_device)
    lpips = _load_lpips_metric(lpips_device)
    raw_rows = evaluate_raw_pairs(
        records,
        device=device,
        lpips_metric=lpips,
        require_png_rgb16=True,
    )
    for raw_row, record in zip(raw_rows, records):
        raw_row["source_sha256"] = record.source_sha256
        raw_row["target_sha256"] = record.target_sha256
    rows = evaluate_quality_pass(
        records,
        backend,
        device=device,
        lpips_metric=lpips,
        latencies=latencies,
        padding_multiple=8,
        require_png_rgb16=True,
    )
    raw_summary = summarize_rows(raw_rows)
    model_summary = summarize_rows(rows)
    payload = {
        "schema": "unblur_slam.turtle_bsd_dpdd_validation_arm.v1",
        "formal": True,
        "interpretation": "single-image current-frame spatial restoration only; K/V reset before every image",
        "protocol": {
            "contract": str(contract_path),
            "contract_sha256": contract_sha,
            "manifest": str(manifest),
            "manifest_sha256": dpdd["validation_manifest_sha256"],
            "selected_split": "validation",
            "pair_count": 74,
            "cache_boundary": "hard_reset_before_every_image",
            "two_frame_wrapper": "pair=(current,current)",
            "precision": "CUDA_FP16",
            "padding": "right_bottom_to_multiple_8_then_crop_back",
            "warmup_steps": 1,
            "latency_scope": (
                "one reset TURTLE current-frame model/backend step in a dedicated "
                "timing-only pass with pre/post CUDA synchronization; excludes K/V "
                "reset, PNG16 decode, padding, sharp-target access, quality forwards, "
                "metrics, LPIPS, reporting, and SLAM"
            ),
            "latency_comparability": (
                "same dedicated timing-only synchronized model/backend-step boundary "
                "as EVSSM; TURTLE uses FP16 autocast while EVSSM uses FP32 with TF32 "
                "runtime flags disclosed, so values are descriptive architecture-"
                "specific restoration latency, not end-to-end online SLAM speed"
            ),
            "pass_separation": {
                "timing_pass": {
                    "reset_model_steps": 74,
                    "sharp_target_images_opened": False,
                    "metrics_or_lpips_computed": False,
                    "warmup_steps_excluded": 1,
                },
                "quality_pass": {
                    "reset_model_steps": 74,
                    "timed_model_steps": 0,
                },
                "passes_are_distinct_complete_dataset_traversals": True,
                "forward_accounting_excluding_warmup": {
                    "timing_only_model_steps": 74,
                    "quality_model_steps": 74,
                    "combined_model_steps": 148,
                },
            },
            "compute_precision": {
                "model_parameters": "FP32",
                "forward_autocast": "CUDA_FP16",
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            },
            "raw_common_baseline_registered": True,
            "claim_scope": "restoration_module_only",
            "slam_quality_or_speed_claim": False,
            "dpdd_dataset": dataset_provenance,
            "dpdd_test_pixels_opened": False,
        },
        "arm": args.arm,
        "seed": args.seed,
        "runtime_identity": dict(runtime),
        "arm_definition": dict(contract["arms"][args.arm]),
        "architecture_variant": (
            models[f"turtle_{args.arm}"]["architecture_variant"]
            if args.arm in {"G", "O"}
            else "t1"
        ),
        "reference_comparison_disclosure": {
            "O_t0_vs_G_t1_is_same_architecture": False,
            "O_t0_vs_G_t1_training_data_matched": False,
            "O_t0_vs_G_t1_training_budget_matched": False,
            "O_t0_vs_G_t1_is_causal_effect_estimate": False,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_metadata": dict(metadata),
        "raw_baseline": {
            "registration": {
                "per_image_rows_present": True,
                "identity_fields": [
                    "name",
                    "blurry_path",
                    "sharp_path",
                    "source_sha256",
                    "target_sha256",
                ],
                "metric_implementation": "evaluate_turtle_single_image_defocus._metrics",
            },
            "summary": raw_summary,
            "images": raw_rows,
        },
        "raw": {"summary": raw_summary, "images": raw_rows},
        "model": {"summary": model_summary, "images": rows},
        "model_minus_raw": paired_arm_delta(rows, raw_rows),
    }
    report = output / "metrics.json"
    with report.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"metrics": str(report), "arm": args.arm, "pairs": 74}))


if __name__ == "__main__":
    main()
