#!/usr/bin/env python3
"""CPU-only fail-closed preflight for strict three-stage TURTLE training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_turtle_unblur_stable import (
    CHECKPOINT_FORMAT,
    _load_sources,
    _read_contract,
    configure_full_scopes,
)
from scripts.train_turtle_streaming import (
    DEFAULT_TURTLE_CHECKPOINT,
    DEFAULT_TURTLE_CONFIG,
    DEFAULT_TURTLE_REPO,
)
from src.turtle_backend import build_turtle_model_from_scratch, sha256_file


SCHEMA = "unblur_slam.turtle_unblur_stable_preflight.v1"


def _publish(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    contract_path = args.contract.expanduser().resolve()
    observed = sha256_file(contract_path)
    if observed != args.contract_sha256.strip().lower():
        raise ValueError("contract SHA mismatch")
    contract = _read_contract(contract_path)
    output_root = Path(str(contract["output_dir"])).expanduser().resolve()
    storage_policy = contract.get("storage_policy")
    if not isinstance(storage_policy, dict) or storage_policy.get("fresh_no_overwrite") is not True:
        raise ValueError("output storage policy is missing or unsafe")
    output_parent = Path(str(storage_policy.get("output_root_parent", ""))).expanduser().resolve()
    if output_parent == Path("/"):
        raise ValueError("output parent must not be filesystem root")
    try:
        output_root.relative_to(output_parent)
    except ValueError as error:
        raise ValueError("output root lies outside contracted parent") from error
    resume_payload = None
    if args.resume is None:
        if args.resume_sha256 is not None:
            raise ValueError("resume SHA was supplied without --resume")
        if output_root.exists() or output_root.is_symlink():
            raise FileExistsError(f"fresh output root already exists: {output_root}")
    else:
        resume = args.resume.expanduser().resolve()
        if args.resume_sha256 is None or sha256_file(resume) != args.resume_sha256.lower():
            raise ValueError("resume checkpoint SHA mismatch")
        if not output_root.is_dir() or output_root.is_symlink():
            raise FileNotFoundError("resume output root is missing")
        try:
            resume.relative_to(output_root / "checkpoints")
        except ValueError as error:
            raise ValueError("resume checkpoint lies outside the output root") from error
        loaded = torch.load(resume, map_location="cpu")
        if loaded.get("contract", {}).get("sha256") != observed:
            raise ValueError("resume checkpoint belongs to another contract")
        resume_payload = {
            "path": str(resume), "sha256": args.resume_sha256.lower(),
            "step": int(loaded.get("step", -1)),
        }
    sources = _load_sources(contract, int(contract["seed"]))
    inventory = []
    source_contracts = {str(item["name"]): item for item in contract["sources"]}
    for source in sources:
        declared = source_contracts[source.name]
        common_inventory = {
            "name": source.name,
            "kind": source.kind,
            "weight": source.weight,
            "manifest": declared["manifest"],
            "manifest_sha256": declared["manifest_sha256"],
            "provenance": declared["provenance"],
        }
        if source.kind == "video":
            inventory.append({
                **common_inventory,
                "sequence_records": len(source.dataset.records),
                "paired_frames": sum(len(record.blurry) for record in source.dataset.records),
            })
        else:
            inventory.append({
                **common_inventory,
                "paired_images": len(source.dataset),
            })
    model, metadata = build_turtle_model_from_scratch(
        DEFAULT_TURTLE_REPO, config=DEFAULT_TURTLE_CONFIG, device=torch.device("cpu"),
    )
    if metadata.get("kind") != "random_initialization":
        raise ValueError("scratch TURTLE architecture identity changed")
    initialization = contract["initialization"]
    prior_identity = None
    if initialization["kind"] == "completed_prior_stage":
        prior_path = Path(initialization["checkpoint"]).expanduser().resolve()
        prior = torch.load(prior_path, map_location="cpu")
        if (prior.get("format") != CHECKPOINT_FORMAT
                or prior.get("stage") != initialization["stage"]
                or int(prior.get("step", -1)) != 300_000):
            raise ValueError("prior-stage checkpoint lineage is invalid")
        model.load_state_dict(prior["model"], strict=True)
        prior_identity = {"path": str(prior_path), "sha256": sha256_file(prior_path),
                          "stage": prior["stage"], "step": int(prior["step"])}
    scopes = configure_full_scopes(model)
    payload = {
        "schema": SCHEMA,
        "status": "pass",
        "gpu_queried_or_started": False,
        "contract": {"path": str(contract_path), "sha256": observed},
        "stage": contract["stage"],
        "output_root": str(output_root),
        "output_root_absent": args.resume is None,
        "resume": resume_payload,
        "test_pixels_permitted": False,
        "photometric_transform": contract["photometric_transform"],
        "training_protocol": {
            key: contract[key]
            for key in (
                "total_steps",
                "crop_size",
                "clip_length",
                "batch_size_per_gpu",
                "global_batch_size",
                "ddp_world_size",
                "ddp_backend",
                "gradient_reduction",
                "amp_overflow_policy",
                "amp_max_same_batch_retries",
                "amp_initial_scale",
                "amp_growth_interval",
                "ddp_source_choice",
                "ddp_rank_data_rng",
                "distributed_topology",
                "launch_workflow",
                "video_batch_implementation",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "betas",
                "scheduler",
                "scheduler_t_max",
                "scheduler_eta_min",
                "gradient_clip_norm",
                "fft_weight",
                "seed",
                "checkpoint_every",
                "amp",
                "validation_during_training",
            )
        },
        "data_mix_disclosure": contract["data_mix_disclosure"],
        "sources": inventory,
        "parameter_scope": {
            "history": sum(parameter.numel() for parameter in scopes.history),
            "spatial": sum(parameter.numel() for parameter in scopes.spatial),
            "total": sum(parameter.numel() for parameter in model.parameters()),
        },
        "initialization": contract["initialization"],
        "prior_checkpoint_verified": prior_identity,
        "scratch_architecture_metadata": metadata,
        "implementation": contract["implementation"],
    }
    if args.output is not None:
        _publish(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
