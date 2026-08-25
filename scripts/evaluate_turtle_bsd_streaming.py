#!/usr/bin/env python3
"""Strict BSD 3ms24ms streaming evaluation for G, O, B, and BD.

The entry point has no test-split argument.  It validates the bound BSD
validation manifest before loading a model, resets K/V at every sequence, and
uses the shared normal/reset/repeat/ordered-replay/shuffled-history controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import (  # noqa: E402
    SEEDS,
    inspect_bsd_sequence_manifest,
    load_contract,
    validate_protocol,
)
from scripts.bsd_dpdd_runtime import require_gpu1_a6000  # noqa: E402
from scripts.evaluate_turtle_streaming import (  # noqa: E402
    FORMAL_HISTORY_CONTROL_FRAME_INDICES,
    evaluate_sequences,
    prepare_output_directory,
)
from scripts.train_turtle_streaming import choose_device, load_sequence_manifest  # noqa: E402
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


def validate_trained_metadata(
    metadata: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    contract_sha256: str,
    bsd_train_sha256: str,
    dpdd_train_sha256: str,
) -> None:
    if metadata.get("kind") != "finetuned" or metadata.get("mode") != arm:
        raise ValueError(f"checkpoint is not formal arm {arm}")
    if metadata.get("schema") != "unblur_slam.turtle_bsd3ms24ms_dpdd_training.v1":
        raise ValueError("B/BD checkpoint training schema changed")
    if metadata.get("contract_sha256") != contract_sha256:
        raise ValueError("B/BD checkpoint contract lineage changed")
    training = metadata.get("training", {})
    expected = {
        "seed": seed,
        "optimizer_steps": 300,
        "attempted_optimizer_steps": 300,
        "executed_optimizer_steps": 300,
        "amp_skipped_optimizer_steps": 0,
        "bsd_passes": 5,
        "bsd_clip_length": 5,
        "dpdd_backward_steps": 0 if arm == "B" else 70,
        "dpdd_pairs": 0 if arm == "B" else 350,
        "validation_during_training": False,
        "test_pixels_or_metrics_read": False,
    }
    if training != expected:
        raise ValueError(f"formal {arm} training metadata changed")
    manifests = metadata.get("manifests", {})
    if manifests.get("bsd_train_sha256") != bsd_train_sha256:
        raise ValueError(f"formal {arm} BSD train lineage changed")
    if manifests.get("bsd_validation_read") is not False or manifests.get("bsd_test_read") is not False:
        raise ValueError(f"formal {arm} touched sealed BSD data during training")
    wanted_dpdd = dpdd_train_sha256 if arm == "BD" else None
    if manifests.get("dpdd_train_sha256") != wanted_dpdd:
        raise ValueError(f"formal {arm} DPDD lineage changed")
    if manifests.get("dpdd_test_read") is not False:
        raise ValueError(f"formal {arm} touched DPDD test")


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
    parser.add_argument("--precision", choices=("fp16",), default="fp16")
    parser.add_argument("--sequence-limit", type=int, choices=(0, 2), default=0)
    parser.add_argument("--reference-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path, contract, contract_sha = load_contract(
        args.contract,
        expected_sha256=args.expected_contract_sha256,
    )
    if args.reference_only and args.arm not in {"G", "O"}:
        raise ValueError("reference-only execution permits only G and O")
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
    )
    records = load_sequence_manifest(
        inventory.path,
        root=Path(bsd["dataset_root"]),
    )
    if args.sequence_limit:
        records = records[: args.sequence_limit]
    output = prepare_output_directory(args.output_dir)
    device = choose_device(args.device)
    if device.type != "cuda":
        raise ValueError("formal BSD streaming evaluation requires CUDA FP16")

    models = contract["models"]
    configured_checkpoint = args.checkpoint
    configured_sha = args.checkpoint_sha256
    if args.arm in {"G", "O"}:
        if args.seed is not None or configured_checkpoint is not None or configured_sha is not None:
            raise ValueError("frozen G/O identities come only from the bound contract")
        definition = models[f"turtle_{args.arm}"]
        checkpoint = Path(definition["checkpoint"])
        checkpoint_sha = definition["checkpoint_sha256"]
    else:
        if args.seed not in SEEDS or configured_checkpoint is None or configured_sha is None:
            raise ValueError("formal B/BD requires seed, checkpoint, and checkpoint SHA256")
        checkpoint = configured_checkpoint.expanduser().resolve()
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != configured_sha.lower():
            raise ValueError("B/BD checkpoint SHA256 mismatch")
    if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_sha:
        raise ValueError(f"formal {args.arm} checkpoint bytes changed")

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
                bsd_train_sha256=bsd["train_manifest_sha256"],
                dpdd_train_sha256=contract["data"]["dpdd"]["train_manifest_sha256"],
            )
    if bool(getattr(model, "use_both_input", True)):
        raise ValueError("formal history controls require use_both_input=false")
    backend = TurtleStreamingBackend(model, device=device, inference_precision="fp16")

    provenance = {
        "contract": str(contract_path),
        "contract_sha256": contract_sha,
        "arm": args.arm,
        "seed": args.seed,
        "manifest": str(inventory.path),
        "manifest_sha256": inventory.sha256,
        "selected_split": "validation",
        "sequence_limit": args.sequence_limit,
        "formal_full_validation": args.sequence_limit == 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "runtime": dict(runtime),
        "bsd_test_pixels_opened": False,
        "arm_definition": dict(contract["arms"][args.arm]),
        "architecture_variant": models[f"turtle_{args.arm}"]["architecture_variant"]
        if args.arm in {"G", "O"}
        else "t1",
        "evaluation_claim_scope": (
            "restoration_module_quality_and_synchronized_model_step_latency_only"
        ),
        "slam_quality_or_speed_claim": False,
    }
    metrics = evaluate_sequences(
        records,
        backend,
        device=device,
        output_dir=output,
        max_visuals=0,
        checkpoint_metadata=metadata,
        provenance=provenance,
        history_controls=True,
    )
    expected_sequences = args.sequence_limit or 20
    expected_frames = expected_sequences * 100
    if metrics.get("sequence_count") != expected_sequences or metrics.get("frame_count") != expected_frames:
        raise RuntimeError("BSD evaluation coverage changed")
    history = metrics.get("history_ablation", {})
    expected_control_frames = expected_sequences * len(FORMAL_HISTORY_CONTROL_FRAME_INDICES)
    if (
        history.get("ordered_replay_frame_count") != expected_frames
        or history.get("control_frame_count") != expected_control_frames
        or history.get("ordered_replay_matches_stream") is not True
    ):
        raise RuntimeError("BSD history-control coverage/replay contract changed")
    performance = metrics.get("performance", {})
    passes = performance.get("pass_separation", {})
    timing_pass = passes.get("timing_pass", {})
    quality_pass = passes.get("quality_history_pass", {})
    if (
        timing_pass.get("normal_stream_model_steps") != expected_frames
        or timing_pass.get("sharp_target_images_opened") is not False
        or timing_pass.get("metrics_computed") is not False
        or timing_pass.get("history_or_replay_control_forwards") != 0
        or quality_pass.get("normal_stream_model_steps") != expected_frames
        or quality_pass.get("timed_model_steps") != 0
        or performance.get("history_controls_timed") is not False
    ):
        raise RuntimeError("BSD dedicated timing/quality pass separation changed")
    accounting = history.get("forward_accounting_excluding_warmup", {})
    if (
        accounting.get("dedicated_timing_normal_stream") != expected_frames
        or accounting.get("timing_pass_history_or_replay_controls") != 0
        or accounting.get("total_including_dedicated_timing_pass")
        != int(accounting.get("total", -expected_frames)) + expected_frames
    ):
        raise RuntimeError("BSD timing/history forward accounting changed")
    raw = metrics.get("raw_baseline", {})
    if (
        raw.get("registration", {}).get("per_frame_rows_present") is not True
        or raw.get("all_frames") != metrics.get("mean", {}).get("raw")
        or raw.get("steady") != metrics.get("steady_mean", {}).get("raw")
    ):
        raise RuntimeError("BSD raw-baseline registration changed")
    metrics_path = output / "metrics.json"
    with metrics_path.open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"metrics": str(metrics_path), "arm": args.arm, "frames": expected_frames}))


if __name__ == "__main__":
    main()
