#!/usr/bin/env python3
"""Formal B/BD trainer for the preregistered BSD 3ms24ms validation study.

This launcher has no validation or test option.  B and BD share the exact same
five-pass BSD clip schedule.  BD inserts one deterministic DPDD pass into 70
approximately-even BSD steps, restores torch RNG before the matched BSD
forward, and still executes exactly one optimizer/scheduler update per step.
The BSD-only B arm never resolves or reads the DPDD manifest or pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import (  # noqa: E402
    SEEDS,
    approximately_even_positions,
    deterministic_bsd_schedule,
    inspect_bsd_sequence_manifest,
    load_contract,
    validate_protocol,
)
from scripts.bsd_dpdd_runtime import require_gpu1_a6000  # noqa: E402
from scripts.train_turtle_mixed_defocus import (  # noqa: E402
    CountingAdamW,
    configure_parameter_scopes,
    deterministic_single_schedule,
    load_dpdd_dataset_contract,
    load_single_batch,
    source_scoped_optimizer_step,
)
from scripts.evaluate_turtle_single_image_defocus import (  # noqa: E402
    load_single_image_manifest,
)
from scripts.train_turtle_streaming import (  # noqa: E402
    HISTORY_ATTENTION_PARAMETER_COUNT,
    HISTORY_ATTENTION_PARAMETER_TENSORS,
    PairedSequenceDataset,
    _choose_transform,
    _read_transformed,
    choose_device,
    save_checkpoint,
    set_seed,
)
from src.turtle_backend import (  # noqa: E402
    FINETUNED_CHECKPOINT_FORMAT,
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    TURTLE_CACHE_CONTRACT,
    load_turtle_model,
    sha256_file,
)


TRAINING_SCHEMA = "unblur_slam.turtle_bsd3ms24ms_dpdd_training.v1"
FORMAL_STEPS = 300
FORMAL_BSD_PASSES = 5
FORMAL_BSD_SEQUENCES = 60
FORMAL_BSD_FRAMES_PER_SEQUENCE = 100
FORMAL_CLIP_LENGTH = 5
FORMAL_CROP_SIZE = 192
FORMAL_DPDD_STEPS = 70
FORMAL_DPDD_BATCH = 5
FORMAL_DPDD_PAIRS = 350
SPATIAL_HEAD_PARAMETER_TENSORS = 30
SPATIAL_HEAD_PARAMETER_COUNT = 105_283


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def optimizer_step_mode(arm: str, *, has_dpdd: bool) -> str:
    """Select the existing source-scoped helper contract for one formal step."""

    if arm == "B":
        if has_dpdd:
            raise ValueError("B must never receive a DPDD batch")
        return "V"
    if arm == "BD":
        return "M" if has_dpdd else "V"
    raise ValueError("formal training arm must be B or BD")


def build_formal_schedules(
    sequence_names: Sequence[str],
    sequence_lengths: Sequence[int],
    dpdd_names: Sequence[str] | None,
    *,
    seed: int,
) -> Mapping[str, Any]:
    if len(sequence_names) != FORMAL_BSD_SEQUENCES:
        raise ValueError("formal BSD training requires exactly 60 sequences")
    if set(sequence_lengths) != {FORMAL_BSD_FRAMES_PER_SEQUENCE}:
        raise ValueError("formal BSD training requires exactly 100 frames per sequence")
    bsd = deterministic_bsd_schedule(
        sequence_names,
        sequence_lengths,
        seed=seed,
        passes=FORMAL_BSD_PASSES,
        clip_length=FORMAL_CLIP_LENGTH,
    )
    if len(bsd) != FORMAL_STEPS:
        raise RuntimeError("formal BSD schedule did not produce exactly 300 steps")
    if dpdd_names is None:
        return {"bsd": bsd, "dpdd": [], "dpdd_positions": ()}
    if len(dpdd_names) != FORMAL_DPDD_PAIRS:
        raise ValueError("formal BD training requires exactly 350 DPDD pairs")
    dpdd = deterministic_single_schedule(
        dpdd_names,
        seed=seed,
        steps=FORMAL_DPDD_STEPS,
        batch_size=FORMAL_DPDD_BATCH,
    )
    positions = approximately_even_positions(
        total_steps=FORMAL_STEPS,
        selected_steps=FORMAL_DPDD_STEPS,
    )
    return {"bsd": bsd, "dpdd": dpdd, "dpdd_positions": positions}


def load_bsd_clip(
    dataset: PairedSequenceDataset,
    *,
    sequence_index: int,
    pass_index: int,
    start: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    record = dataset.records[sequence_index]
    stop = int(start) + FORMAL_CLIP_LENGTH
    if start < 0 or stop > len(record.blurry):
        raise ValueError("BSD clip bounds escape the canonical sequence")
    transform_seed = int(seed) + 30_000 + int(pass_index) * 1_000_003 + int(sequence_index) * 9_176
    transform = _choose_transform(
        record,
        crop_size=FORMAL_CROP_SIZE,
        augment=True,
        rng=random.Random(transform_seed),
    )
    blurry = torch.stack(
        [_read_transformed(path, transform) for path in record.blurry[start:stop]]
    )
    sharp = torch.stack(
        [_read_transformed(path, transform) for path in record.sharp[start:stop]]
    )
    if tuple(blurry.shape) != (FORMAL_CLIP_LENGTH, 3, FORMAL_CROP_SIZE, FORMAL_CROP_SIZE):
        raise RuntimeError("BSD clip tensor contract changed")
    return blurry, sharp, {
        "sequence": record.name,
        "sequence_index": int(sequence_index),
        "pass_index": int(pass_index),
        "start": int(start),
        "stop_exclusive": int(stop),
        "transform_rng_seed": transform_seed,
        "crop_box": list(transform.crop_box),
        "flip_horizontal": transform.flip_horizontal,
        "flip_vertical": transform.flip_vertical,
        "quarter_turns": transform.quarter_turns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--arm", choices=("B", "BD"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path, contract, contract_sha = load_contract(
        args.contract,
        expected_sha256=args.expected_contract_sha256,
    )
    validate_protocol(contract, allow_template=False)
    runtime = require_gpu1_a6000(args.device)
    if args.seed not in SEEDS:
        raise ValueError("formal B/BD seed must be 17, 42, or 73")
    training = contract["training"]
    if (
        training.get("optimizer_steps") != FORMAL_STEPS
        or training.get("bsd_passes") != FORMAL_BSD_PASSES
        or training.get("bsd_clip_length") != FORMAL_CLIP_LENGTH
        or training.get("crop_size") != FORMAL_CROP_SIZE
    ):
        raise ValueError("bound contract training budget changed")

    output = args.output.expanduser().resolve()
    digest_output = output.with_name(output.name + ".sha256")
    if output.exists() or digest_output.exists():
        raise FileExistsError(f"formal trainer refuses overwrite: {output}")

    bsd = contract["data"]["bsd"]
    bsd_root = Path(bsd["dataset_root"]).expanduser().resolve()
    bsd_train_manifest = Path(bsd["train_manifest"]).expanduser().resolve()
    inspect_bsd_sequence_manifest(
        bsd_train_manifest,
        dataset_root=bsd_root,
        expected_sha256=bsd["train_manifest_sha256"],
        expected_split="train",
        expected_sequences=60,
        expected_frames=6000,
        expected_per_exposure_sequences=60,
        require_assets=True,
    )
    # No validation path is resolved or read by this trainer.
    video_dataset = PairedSequenceDataset(
        bsd_train_manifest,
        root=bsd_root,
        crop_size=FORMAL_CROP_SIZE,
        augment=True,
        seed=args.seed,
    )
    if len(video_dataset.records) != FORMAL_BSD_SEQUENCES:
        raise ValueError("BSD train manifest record count changed")

    dpdd = None
    dpdd_manifest = None
    dpdd_records = []
    if args.arm == "BD":
        dpdd = contract["data"]["dpdd"]
        dpdd_manifest = Path(dpdd["train_manifest"]).expanduser().resolve()
        if sha256_file(dpdd_manifest) != dpdd["train_manifest_sha256"]:
            raise ValueError("DPDD train manifest SHA256 changed")
        load_dpdd_dataset_contract(
            Path(dpdd["dataset_manifest"]),
            expected_dataset_manifest_sha256=dpdd["dataset_manifest_sha256"],
            train_manifest=dpdd_manifest,
            expected_train_manifest_sha256=dpdd["train_manifest_sha256"],
        )
        dpdd_records = load_single_image_manifest(
            dpdd_manifest,
            root=Path(dpdd["root"]),
            expected_split="train",
            canonical_contract=True,
            verify_content=True,
        )
    schedules = build_formal_schedules(
        [record.name for record in video_dataset.records],
        [len(record.blurry) for record in video_dataset.records],
        [record.name for record in dpdd_records] if args.arm == "BD" else None,
        seed=args.seed,
    )

    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type != "cuda" or not args.amp:
        raise ValueError("formal B/BD training requires CUDA with --amp")
    turtle = contract["models"]["turtle_G"]
    model, base_metadata = load_turtle_model(
        contract["models"]["turtle_shared_repo"]["repo"],
        turtle["checkpoint"],
        config=turtle["config"],
        device=device,
    )
    if base_metadata.get("kind") != "official_gopro":
        raise ValueError("B/BD must start from the official G checkpoint")
    mode = "V" if args.arm == "B" else "M"
    scopes = configure_parameter_scopes(model, mode)
    model.train()
    optimizer = CountingAdamW(
        [
            {"params": scopes.history, "lr": 1.0e-5, "weight_decay": 1.0e-3},
            {"params": scopes.spatial, "lr": 1.0e-5, "weight_decay": 1.0e-4},
        ],
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FORMAL_STEPS, eta_min=1.0e-7
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=True,
        init_scale=1024.0,
        growth_interval=2000,
    )
    dpdd_position_to_batch = {
        position: index for index, position in enumerate(schedules["dpdd_positions"])
    }
    bsd_audits: List[Mapping[str, Any]] = []
    dpdd_audits: List[Mapping[str, Any]] = []
    for step, (pass_index, sequence_index, start) in enumerate(schedules["bsd"]):
        video_blurry, video_sharp, bsd_audit = load_bsd_clip(
            video_dataset,
            sequence_index=sequence_index,
            pass_index=pass_index,
            start=start,
            seed=args.seed,
        )
        bsd_audits.append(bsd_audit)
        single_batch = None
        if args.arm == "BD" and step in dpdd_position_to_batch:
            dpdd_step = dpdd_position_to_batch[step]
            single_blurry, single_sharp, audit = load_single_batch(
                dpdd_records,
                schedules["dpdd"][dpdd_step],
                seed=args.seed,
                step=dpdd_step,
                crop_size=FORMAL_CROP_SIZE,
            )
            single_batch = (single_blurry, single_sharp)
            dpdd_audits.extend(audit)
        row = source_scoped_optimizer_step(
            model,
            scopes,
            optimizer,
            mode=optimizer_step_mode(args.arm, has_dpdd=single_batch is not None),
            device=device,
            single_batch=single_batch,
            video_batch=(video_blurry, video_sharp),
            amp_enabled=True,
            scaler=scaler,
            scheduler=scheduler,
        )
        print(json.dumps({"step": step + 1, "arm": args.arm, **row}), flush=True)
    if optimizer.step_count != FORMAL_STEPS or scheduler.last_epoch != FORMAL_STEPS:
        raise RuntimeError("formal optimizer/scheduler did not execute exactly 300 steps")
    if args.arm == "BD" and len(dpdd_audits) != FORMAL_DPDD_PAIRS:
        raise RuntimeError("BD did not consume exactly one DPDD train pass")
    if args.arm == "B" and dpdd_audits:
        raise RuntimeError("B unexpectedly consumed DPDD")

    metadata = {
        "format": FINETUNED_CHECKPOINT_FORMAT,
        "schema": TRAINING_SCHEMA,
        "base_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
        "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
        "input_domain": "raw",
        "cache_contract": TURTLE_CACHE_CONTRACT,
        "mode": args.arm,
        "contract": str(contract_path),
        "contract_sha256": contract_sha,
        "runtime_identity": dict(runtime),
        "training": {
            "seed": args.seed,
            "optimizer_steps": optimizer.step_count,
            "attempted_optimizer_steps": FORMAL_STEPS,
            "executed_optimizer_steps": optimizer.step_count,
            "amp_skipped_optimizer_steps": 0,
            "bsd_passes": FORMAL_BSD_PASSES,
            "bsd_clip_length": FORMAL_CLIP_LENGTH,
            "dpdd_backward_steps": FORMAL_DPDD_STEPS if args.arm == "BD" else 0,
            "dpdd_pairs": FORMAL_DPDD_PAIRS if args.arm == "BD" else 0,
            "validation_during_training": False,
            "test_pixels_or_metrics_read": False,
        },
        "manifests": {
            "bsd_train": str(bsd_train_manifest),
            "bsd_train_sha256": bsd["train_manifest_sha256"],
            "bsd_selected_split": "train",
            "bsd_validation_read": False,
            "bsd_test_read": False,
            "dpdd_train": str(dpdd_manifest) if args.arm == "BD" else None,
            "dpdd_train_sha256": dpdd["train_manifest_sha256"] if args.arm == "BD" else None,
            "dpdd_selected_split": "train" if args.arm == "BD" else None,
            "dpdd_test_read": False,
        },
        "parameter_scopes": {
            "history_names": scopes.history_names,
            "history_parameters": HISTORY_ATTENTION_PARAMETER_COUNT,
            "history_tensors": HISTORY_ATTENTION_PARAMETER_TENSORS,
            "spatial_names": scopes.spatial_names,
            "spatial_parameters": SPATIAL_HEAD_PARAMETER_COUNT,
            "spatial_tensors": SPATIAL_HEAD_PARAMETER_TENSORS,
            "dpdd_history_gradient": "forbidden_and_asserted_none",
        },
        "sampling_audit": {
            "bsd_schedule_sha256": _json_sha256(schedules["bsd"]),
            "bsd_transforms_sha256": _json_sha256(bsd_audits),
            "dpdd_positions": list(schedules["dpdd_positions"]) if args.arm == "BD" else [],
            "dpdd_schedule_names_sha256": (
                _json_sha256(
                    [[dpdd_records[index].name for index in batch] for batch in schedules["dpdd"]]
                )
                if args.arm == "BD"
                else None
            ),
            "dpdd_transforms_sha256": _json_sha256(dpdd_audits) if args.arm == "BD" else None,
            "bsd_transforms": bsd_audits,
            "dpdd_transforms": dpdd_audits,
        },
        "loss": dict(training["bsd_objective"]),
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.99],
            "scheduler": "CosineAnnealingLR_Tmax300_eta_min_1e-7",
            "groupwise_clip": 1.0,
        },
    }
    digest = save_checkpoint(output, model=model, metadata=metadata, overwrite=False)
    print(json.dumps({"checkpoint": str(output), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
