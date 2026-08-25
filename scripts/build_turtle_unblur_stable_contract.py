#!/usr/bin/env python3
"""Freeze one stage of the strict three-stage scratch TURTLE protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_turtle_unblur_stable import (  # noqa: E402
    BATCH_SIZE_PER_GPU,
    GLOBAL_BATCH_SIZE,
    DDP_WORLD_SIZE,
    SUPPORTED_DDP_WORLD_SIZES,
    DDP_BACKEND,
    AMP_MAX_SAME_BATCH_RETRIES,
    AMP_INITIAL_SCALE,
    AMP_GROWTH_INTERVAL,
    CLIP_LENGTH,
    CROP_SIZE,
    DEFOCUS_REHEARSAL_LR,
    FFT_WEIGHT,
    MOTION_BASE_LR,
    PHOTOMETRIC_TRANSFORM,
    REPLICA_LR,
    SCHEMA,
    TOTAL_STEPS,
)
from src.turtle_backend import (  # noqa: E402
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    sha256_file,
)


DATA_ROOT = Path(
    os.environ.get("UNBLUR_ARTIFACT_ROOT", "/srv/szha0669/unblur-slam")
).expanduser().resolve()
MANIFEST_ROOT = DATA_ROOT / "training_manifests/stable_turtle_v1"
OUTPUT_ROOT = DATA_ROOT / "turtle_finetune/unblur_stable_three_stage_ddp_v3"
OUTPUTS = {stage: OUTPUT_ROOT / f"{stage}_seed42" for stage in (
    "motion_base", "replica", "defocus_rehearsal"
)}
STAGE_LR = {"motion_base": MOTION_BASE_LR, "replica": REPLICA_LR,
            "defocus_rehearsal": DEFOCUS_REHEARSAL_LR}


def pin(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def common(stage: str, output: Path, ddp_world_size: int = DDP_WORLD_SIZE) -> dict:
    if ddp_world_size not in SUPPORTED_DDP_WORLD_SIZES:
        raise ValueError(f"unsupported DDP world size: {ddp_world_size}")
    batch_size_per_gpu = GLOBAL_BATCH_SIZE // ddp_world_size
    implementation_paths = [
        ROOT / "scripts/train_turtle_unblur_stable.py",
        ROOT / "scripts/preflight_turtle_unblur_stable.py",
        ROOT / "scripts/build_turtle_unblur_stable_contract.py",
        ROOT / "scripts/evaluate_turtle_single_image_defocus.py",
        ROOT / "scripts/train_turtle_mixed_defocus.py",
        ROOT / "scripts/train_turtle_streaming.py",
        ROOT / "scripts/execute_pinned_dual_gpu_command.py",
        ROOT / "scripts/run_turtle_unblur_ddp_probe_and_train.py",
        ROOT / "scripts/offline_fair_gpu_contract.py",
        ROOT / "src/turtle_backend.py",
        ROOT / "training_file_extra/motion.yaml",
        ROOT / "training_file_extra/defocus.yaml",
        ROOT / "training_file_extra/reverse_gamma_whole_dataset/reverse.py",
    ]
    if stage in {"replica", "defocus_rehearsal"}:
        implementation_paths.append(ROOT / "scripts/materialize_replica_blurry_office3.py")
    return {
        "schema": SCHEMA,
        "stage": stage,
        "output_dir": str(output.expanduser().resolve()),
        "storage_policy": {
            "output_root_parent": str(OUTPUT_ROOT.expanduser().resolve()),
            "fresh_no_overwrite": True,
        },
        "test_pixels_permitted": False,
        "validation_during_training": False,
        "implementation": [pin(path) for path in implementation_paths],
        "total_steps": TOTAL_STEPS,
        "crop_size": CROP_SIZE,
        "clip_length": CLIP_LENGTH,
        "batch_size_per_gpu": batch_size_per_gpu,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "ddp_world_size": ddp_world_size,
        "ddp_backend": DDP_BACKEND,
        "gradient_reduction": "ddp_mean_equivalent_to_global_batch6_mean",
        "amp_overflow_policy": "lower_scale_and_retry_exact_same_batch_without_scheduler_or_sample_advance",
        "amp_max_same_batch_retries": AMP_MAX_SAME_BATCH_RETRIES,
        "amp_initial_scale": AMP_INITIAL_SCALE,
        "amp_growth_interval": AMP_GROWTH_INTERVAL,
        "ddp_source_choice": "identical_seed_and_allreduce_identity_check",
        "ddp_rank_data_rng": "seed_plus_rank_times_1000000_plus_source_offset",
        "distributed_topology": "single_node_exact_visible_gpu_count",
        "launch_workflow": "same_dual_gpu_lease_step1_probe_then_exact_checkpoint_resume_to_300000",
        "video_batch_implementation": (
            f"ddp{ddp_world_size}_local_batch{batch_size_per_gpu}_global_batch6_"
            "five_frame_bptt_one_optimizer_step"
        ),
        "optimizer": "AdamW",
        "learning_rate": STAGE_LR[stage],
        "weight_decay": 1e-3,
        "betas": [0.9, 0.9],
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": TOTAL_STEPS,
        "scheduler_eta_min": 1e-7,
        "gradient_clip_norm": 1.0,
        "fft_weight": FFT_WEIGHT,
        "photometric_transform": PHOTOMETRIC_TRANSFORM,
        "initialization_root": "random_scratch_pinned_turtle_architecture",
        "official_gopro_checkpoint_used_for_initialization": False,
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
        "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
        "seed": 42,
        "checkpoint_every": 6_000,
        "amp": True,
        "data_mix_disclosure": {
            "paper_discloses_exact_weights": False,
            "weights_are_preregistered_extension": True,
            "sampling": "strict_stage_order_then_one_source_per_step_ddp_global_batch6",
            "model_selection_or_validation_during_training": False,
            "bsd_dpdd_validation_and_tum_are_held_out": True,
            "bsd_training_pixels_used": False,
            "tum_training_pixels_used": False,
        },
    }


def source(
    name: str,
    kind: str,
    root: Path,
    manifest: Path,
    weight: float,
    *,
    repository: str,
    revision: str,
    provenance_artifacts: list[Path],
    **extra,
) -> dict:
    root = root.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(root)
    payload = {
        "name": name,
        "kind": kind,
        "root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "split": "train",
        "weight": weight,
        "alignment_radius": 0,
        "bit_depth": 8,
        "provenance": {
            "repository": repository,
            "revision": revision,
            "artifacts": [pin(path) for path in provenance_artifacts],
        },
    }
    payload.update(extra)
    return payload


def motion_base_contract(ddp_world_size: int = DDP_WORLD_SIZE) -> dict:
    payload = common("motion_base", OUTPUTS["motion_base"], ddp_world_size)
    payload["initialization"] = {"kind": "random_scratch_pinned_turtle_architecture"}
    payload["sources"] = [
        source(
            "reds",
            "video",
            DATA_ROOT / "training_data/reds_hf_snah/materialized",
            MANIFEST_ROOT / "reds_train.content_addressed.v1.jsonl",
            0.55,
            repository="snah/REDS",
            revision="62dc25d16e6f43d2214f1b365023abda86f7a0ae",
            provenance_artifacts=[
                DATA_ROOT / "training_data/reds_hf_snah/train_blur.zip",
                DATA_ROOT / "training_data/reds_hf_snah/train_sharp.zip",
            ],
        ),
        source(
            "gopro_blur_gamma",
            "video",
            DATA_ROOT / "training_data/gopro_large_hf_snah/materialized",
            MANIFEST_ROOT / "gopro_large_blur_gamma_train.content_addressed.v1.jsonl",
            0.45,
            repository="snah/GOPRO_Large",
            revision="592978466ae510d2734b199cad2fc79a346bda1c",
            provenance_artifacts=[
                DATA_ROOT / "training_data/gopro_large_hf_snah/GOPRO_Large.zip",
            ],
        ),
    ]
    return payload


def prior_initialization(stage: str, initialization: Path) -> dict:
    initialization = initialization.expanduser().resolve()
    return {
        "kind": "completed_prior_stage",
        "stage": stage,
        "checkpoint": str(initialization),
        "checkpoint_sha256": sha256_file(initialization),
    }


def replica_contract(initialization: Path, office_root: Path, office_manifest: Path,
                     ddp_world_size: int = DDP_WORLD_SIZE) -> dict:
    payload = common("replica", OUTPUTS["replica"], ddp_world_size)
    payload["initialization"] = prior_initialization("motion_base", initialization)
    payload["sources"] = [source(
        "replica_blurry_office3", "video", office_root, office_manifest, 1.0,
        repository="qizhangslam/Unblur_slam_traning_dataset",
        revision="1f9d98158c3f27f6ec6de45ee2874c9caf2a2c59",
        provenance_artifacts=[office_root / "materialization_audit.json"],
    )]
    return payload


def defocus_contract(initialization: Path, office_root: Path, office_manifest: Path,
                     ddp_world_size: int = DDP_WORLD_SIZE) -> dict:
    payload = common("defocus_rehearsal", OUTPUTS["defocus_rehearsal"], ddp_world_size)
    payload["initialization"] = prior_initialization("replica", initialization)
    payload["sources"] = [
        source(
            "unblur_hf_defocus",
            "single",
            DATA_ROOT / "training_data/unblur_slam_official_rev1f9d981",
            MANIFEST_ROOT / "unblur_hf_defocus_train.paired_image.v1.jsonl",
            0.80,
            repository="qizhangslam/Unblur_slam_traning_dataset",
            revision="1f9d98158c3f27f6ec6de45ee2874c9caf2a2c59",
            provenance_artifacts=[
                DATA_ROOT / "training_data/unblur_slam_official_rev1f9d981/.gitattributes",
            ],
            manifest_kind="paired_image",
        ),
        source(
            "replica_blurry_office3", "video", office_root, office_manifest, 0.20,
            repository="qizhangslam/Unblur_slam_traning_dataset",
            revision="1f9d98158c3f27f6ec6de45ee2874c9caf2a2c59",
            provenance_artifacts=[office_root / "materialization_audit.json"],
        ),
    ]
    return payload


def main() -> None:
    global DATA_ROOT, MANIFEST_ROOT, OUTPUT_ROOT, OUTPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("motion_base", "replica", "defocus_rehearsal"), required=True)
    parser.add_argument("--office-root", type=Path)
    parser.add_argument("--office-manifest", type=Path)
    parser.add_argument("--initialization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ddp-world-size", type=int, choices=SUPPORTED_DDP_WORLD_SIZES,
                        default=DDP_WORLD_SIZE)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    DATA_ROOT = args.data_root.expanduser().resolve()
    MANIFEST_ROOT = args.manifest_root.expanduser().resolve()
    OUTPUT_ROOT = args.output_root.expanduser().resolve()
    if OUTPUT_ROOT == Path("/"):
        raise ValueError("output root must not be filesystem root")
    OUTPUTS = {stage: OUTPUT_ROOT / f"{stage}_seed42" for stage in (
        "motion_base", "replica", "defocus_rehearsal"
    )}
    if args.stage == "motion_base":
        if args.initialization is not None or args.office_root is not None or args.office_manifest is not None:
            raise ValueError("motion_base accepts no prior checkpoint or Office3 input")
        payload = motion_base_contract(args.ddp_world_size)
    elif args.stage == "replica":
        if args.initialization is None or args.office_root is None or args.office_manifest is None:
            raise ValueError("replica requires prior motion_base and Office3")
        payload = replica_contract(args.initialization, args.office_root, args.office_manifest,
                                   args.ddp_world_size)
    else:
        if args.initialization is None or args.office_root is None or args.office_manifest is None:
            raise ValueError("defocus_rehearsal requires prior replica and Office3 rehearsal")
        payload = defocus_contract(args.initialization, args.office_root, args.office_manifest,
                                   args.ddp_world_size)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"path": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
