#!/usr/bin/env python3
"""Train the causal video-deblur network on ordered JSONL sequences.

The v3 objective treats frozen single-frame EVSSM as the safe baseline.  The
temporal network predicts a bounded residual and is trained with newest-frame
reconstruction, EVSSM fidelity, last-pair temporal-delta, and spatial-gradient
constraints.  Cached EVSSM outputs are supported so this training script does
not need to run the single-frame teacher online.
"""

import argparse
from contextlib import nullcontext
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_deblur import VideoDeblurJsonlDataset, build_causal_video_deblur
from src.video_deblur.dataset import (
    TEACHER_PROVENANCE_SCHEMA,
    sha256_file,
)


CHECKPOINT_FORMAT_V3 = "unblur_slam.causal_video_deblur.v3"
OBJECTIVE_SCHEMA_V3 = "unblur_slam.causal_video_deblur.objective.v3"
REFINEMENT_SCHEMA_V3 = "unblur_slam.causal_video_deblur.refinement.v3"
TRAINING_CONTRACT_SCHEMA_V3 = "unblur_slam.causal_video_deblur.training.v3"
CHECKPOINT_FORMAT_V4 = "unblur_slam.causal_video_deblur.v4"
OBJECTIVE_SCHEMA_V4 = "unblur_slam.causal_video_deblur.objective.v4"
REFINEMENT_SCHEMA_V4 = "unblur_slam.causal_video_deblur.refinement.v4"
TRAINING_CONTRACT_SCHEMA_V4 = "unblur_slam.causal_video_deblur.training.v4"
OPTIMIZATION_SCHEMA_V4 = "unblur_slam.causal_video_deblur.optimization.v4"
WARM_START_SCHEMA_V4 = "unblur_slam.causal_video_deblur.warm_start.v4"
RNG_STATE_SCHEMA_V4 = "unblur_slam.causal_video_deblur.rng_state.v4"
NUMPY_RNG_ENCODING_V4 = (
    "numpy.random.RandomState.MT19937.keys_torch_int64.v1"
)
V4_TOTAL_STEPS = 600
V4_ALIGNMENT_ONLY_STEPS = 100
V4_BASE_LR = 2.0e-5
V4_ALIGNMENT_LR = 2.0e-4
V4_WEIGHT_DECAY = 1.0e-3
V4_REGISTERED_CONTRACT_SCHEMA = (
    "unblur_slam.causal_evssm_alignment_replica424_experiment.v4"
)
V4_REGISTERED_CONTRACT_SHA256 = (
    "511dbcce9bad94ef10b3b5af9615d1bfed1300cf273cac5a9b57779c0413563d"
)
V4_REGISTERED_CONTRACT_PATH = (
    ROOT / "configs/local/causal_evssm_v4_alignment_replica424_contract.json"
)
V4_WARM_START_SHA256 = (
    "8338d007762d9e626bd7f85722140a70ff0084e58f1bbe8dfd338daca346b0e4"
)
V4_EXPECTED_TRAIN_CLIPS = 234
V4_EXPECTED_TRAIN_SEQUENCES = 127
V4_EXPECTED_REAL_TRANSITION_SLOTS = 169
V4_EXPECTED_UNIQUE_REAL_TRANSITIONS = 107
V4_EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "bd7caa189374683c8ffd7e8fce83cb62e5f69b73f6048808c4808dc2b4ecd2ba"
)
V4_EXPECTED_TRAIN_PRECOMPUTE_SHA256 = (
    "9fad0d8c90e64fc5ef471bef85c374b5a09393f33ca16fb3dabb5a1bb206a3e0"
)
V4_EXPECTED_TRAIN_TEACHER_MANIFEST_SHA256 = (
    "1e1f9ab0d28ec3d7f391c9d4bcb6184ea275829af3fc824c7e42195bbba1f24e"
)
V4_EXPECTED_VAL_MANIFEST_SHA256 = (
    "1aa8cc7a01b82c7d759c3db70e6c7e796a26d09398f3a1fd1592d787db9f886b"
)
V4_EXPECTED_VAL_PRECOMPUTE_SHA256 = (
    "2a394089ead9b6ef069fab1885b20d11805b90b83d32f3cbd180fd4490cd8d4a"
)
V4_EXPECTED_VAL_TEACHER_MANIFEST_SHA256 = (
    "b9f2b86a18705bb427799fc1823491cf7dd6f9a7e3c54af9889a09fe1073e6fc"
)
V4_EXPECTED_EVSSM_SHA256 = (
    "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
)
V4_ALIGNMENT_STATE_PREFIXES = ("motion_aligner.",)
V4_ALIGNMENT_STATE_EXACT = ("motion_alignment_gate",)
V4_MOTION_ALIGNMENT_CONFIG = {
    "mode": "coarse_local_correlation_v1",
    "match_channels": 16,
    "radius": 8,
    "temperature": 0.05,
}
V4_CLI_ONLY_FIELDS = (
    "motion_alignment_v4",
    "warm_start_v3",
    "v4_alignment_only_steps",
    "v4_base_lr",
    "v4_alignment_lr",
    "v4_alignment_photo_weight",
    "v4_alignment_gradient_weight",
    "v4_alignment_smooth_weight",
    "v4_joint_alignment_weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument(
        "--precompute-report",
        "--train-precompute-report",
        dest="train_precompute_report",
        type=Path,
        help=(
            "validated precompute_report.json; when --train-manifest is omitted, "
            "the report's content-bound output manifest is used"
        ),
    )
    parser.add_argument("--val-precompute-report", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--motion-alignment-v4",
        action="store_true",
        help=(
            "enable the preregistered motion-aligned v4 training path; the "
            "legacy v3 path is unchanged when this flag is absent"
        ),
    )
    parser.add_argument(
        "--warm-start-v3",
        type=Path,
        help="strict v3 checkpoint used to initialize only shared v4 model keys",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")

    parser.add_argument("--history", type=int, default=5)
    parser.add_argument(
        "--input-domain",
        choices=("raw", "evssm"),
        default="raw",
        help=(
            "raw trains a standalone causal model; evssm trains a causal "
            "temporal adapter on frozen single-frame EVSSM outputs"
        ),
    )
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument(
        "--max-residual",
        type=float,
        default=8.0 / 255.0,
        help=(
            "hard per-pixel RGB correction bound in normalized [0,1] units; "
            "the v3 default is 8/255 around the EVSSM input"
        ),
    )
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--clip-stride", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help=(
            "optimizer-step budget; 0 uses epochs*loader length. The cosine "
            "schedule advances per step, matching EVSSM's iteration recipe"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--grad-accumulation",
        type=int,
        default=1,
        help="number of micro-batches per optimizer step",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="linear LR warmup measured in optimizer steps",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument(
        "--v4-alignment-only-steps",
        type=int,
        default=V4_ALIGNMENT_ONLY_STEPS,
        help="v4 phase-1 optimizer steps with every base parameter frozen",
    )
    parser.add_argument("--v4-base-lr", type=float, default=V4_BASE_LR)
    parser.add_argument("--v4-alignment-lr", type=float, default=V4_ALIGNMENT_LR)
    parser.add_argument("--v4-alignment-photo-weight", type=float, default=1.0)
    parser.add_argument("--v4-alignment-gradient-weight", type=float, default=0.2)
    parser.add_argument("--v4-alignment-smooth-weight", type=float, default=0.01)
    parser.add_argument("--v4-joint-alignment-weight", type=float, default=0.05)
    parser.add_argument("--fft-weight", type=float, default=0.1)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--evssm-fidelity-weight",
        type=float,
        default=0.25,
        help="full-prefix L1 penalty to the frozen/cached EVSSM sequence",
    )
    parser.add_argument(
        "--temporal-delta-weight",
        type=float,
        default=0.1,
        help="L1 penalty on the newest adjacent-frame delta versus sharp GT",
    )
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0.05,
        help="GT spatial-gradient and runtime-style Laplacian L1 on the newest frame",
    )
    parser.add_argument(
        "--laplacian-gate-weight",
        type=float,
        default=0.01,
        help=(
            "runtime-aligned relative log-variance hinge versus EVSSM; kept "
            "separate from GT edge fidelity to expose domain-shift spikes"
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="stop after one optimization step")

    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="single-frame EVSSM .pth containing a params state dict",
    )
    parser.add_argument(
        "--teacher-input",
        action="store_true",
        help=(
            "unsupported for deployable models; use --distill-weight with an "
            "EVSSM checkpoint or cached teacher frames instead"
        ),
    )
    parser.add_argument(
        "--teacher-chunk",
        type=int,
        default=1,
        help="EVSSM requires batch=1; values other than 1 are rejected",
    )
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def validate_teacher_options(
    teacher_checkpoint: Optional[Path],
    teacher_input: bool,
    distill_weight: float,
    input_domain: str = "raw",
    evssm_fidelity_weight: float = 0.0,
) -> None:
    """Keep the trained model compatible with the one-input streaming runtime."""
    if distill_weight < 0.0:
        raise ValueError("--distill-weight must be non-negative")
    if evssm_fidelity_weight < 0.0:
        raise ValueError("--evssm-fidelity-weight must be non-negative")
    if teacher_input:
        raise ValueError(
            "--teacher-input is not deployable by the one-input streaming runtime; "
            "use EVSSM distillation with a positive --distill-weight instead"
        )
    normalized_domain = str(input_domain).lower()
    if normalized_domain not in {"raw", "evssm"}:
        raise ValueError("--input-domain must be raw or evssm")
    if (
        teacher_checkpoint
        and distill_weight <= 0.0
        and evssm_fidelity_weight <= 0.0
        and normalized_domain != "evssm"
    ):
        raise ValueError(
            "--teacher-checkpoint has no effect unless a positive "
            "--distill-weight/--evssm-fidelity-weight is configured or "
            "--input-domain=evssm"
        )


def validate_v3_hyperparameters(
    *,
    max_residual: float,
    fft_weight: float,
    distill_weight: float,
    evssm_fidelity_weight: float,
    temporal_delta_weight: float,
    edge_weight: float,
    laplacian_gate_weight: float,
) -> None:
    values = {
        "--max-residual": max_residual,
        "--fft-weight": fft_weight,
        "--distill-weight": distill_weight,
        "--evssm-fidelity-weight": evssm_fidelity_weight,
        "--temporal-delta-weight": temporal_delta_weight,
        "--edge-weight": edge_weight,
        "--laplacian-gate-weight": laplacian_gate_weight,
    }
    for label, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
    if max_residual <= 0.0 or max_residual > 1.0:
        raise ValueError("--max-residual must be in (0, 1] for v3 training")
    for label in (
        "--fft-weight",
        "--distill-weight",
        "--evssm-fidelity-weight",
        "--temporal-delta-weight",
        "--edge-weight",
        "--laplacian-gate-weight",
    ):
        if values[label] < 0.0:
            raise ValueError(f"{label} must be non-negative")


def validate_v4_options(args: argparse.Namespace) -> None:
    """Fail closed around the preregistered 100+500 step v4 protocol."""

    enabled = bool(getattr(args, "motion_alignment_v4", False))
    warm_start = getattr(args, "warm_start_v3", None)
    if not enabled:
        if warm_start is not None:
            raise ValueError("--warm-start-v3 requires --motion-alignment-v4")
        return
    resume = getattr(args, "resume", None)
    if (warm_start is None) == (resume is None):
        raise ValueError(
            "motion-aligned v4 requires exactly one of --warm-start-v3 or --resume"
        )
    if int(args.history) < 2:
        raise ValueError("motion-aligned v4 requires history >= 2")
    if str(args.input_domain) != "evssm":
        raise ValueError("motion-aligned v4 is defined only in the EVSSM domain")
    if bool(args.teacher_input):
        raise ValueError("motion-aligned v4 does not support teacher input features")
    if int(args.max_steps) not in (0, V4_TOTAL_STEPS):
        raise ValueError(
            f"motion-aligned v4 requires exactly {V4_TOTAL_STEPS} optimizer steps"
        )
    if int(args.v4_alignment_only_steps) != V4_ALIGNMENT_ONLY_STEPS:
        raise ValueError(
            "motion-aligned v4 requires exactly 100 alignment-only steps"
        )
    if int(args.warmup_steps) != 0:
        raise ValueError("motion-aligned v4 uses fixed phase learning rates, no warmup")

    exact_values = {
        "--weight-decay": (float(args.weight_decay), V4_WEIGHT_DECAY),
        "--v4-base-lr": (float(args.v4_base_lr), V4_BASE_LR),
        "--v4-alignment-lr": (
            float(args.v4_alignment_lr),
            V4_ALIGNMENT_LR,
        ),
        "--v4-alignment-photo-weight": (
            float(args.v4_alignment_photo_weight),
            1.0,
        ),
        "--v4-alignment-gradient-weight": (
            float(args.v4_alignment_gradient_weight),
            0.2,
        ),
        "--v4-alignment-smooth-weight": (
            float(args.v4_alignment_smooth_weight),
            0.01,
        ),
        "--v4-joint-alignment-weight": (
            float(args.v4_joint_alignment_weight),
            0.05,
        ),
    }
    for label, (actual, expected) in exact_values.items():
        if not math.isfinite(actual) or actual != expected:
            raise ValueError(
                f"motion-aligned v4 preregisters {label}={expected}, got {actual}"
            )

    exact_integer_values = {
        "--history": (int(args.history), 3),
        "--channels": (int(args.channels), 32),
        "--heads": (int(args.heads), 4),
        "--blocks": (int(args.blocks), 2),
        "--crop-size": (int(args.crop_size), 192),
        "--batch-size": (int(args.batch_size), 4),
        "--grad-accumulation": (int(args.grad_accumulation), 2),
        "--workers": (int(args.workers), 0),
        "--clip-stride": (int(args.clip_stride), 1),
        "--seed": (int(args.seed), 42),
    }
    for label, (actual, expected) in exact_integer_values.items():
        if actual != expected:
            raise ValueError(
                f"motion-aligned v4 preregisters {label}={expected}, got {actual}"
            )
    exact_base_loss_values = {
        "--max-residual": (float(args.max_residual), 8.0 / 255.0),
        "--fft-weight": (float(args.fft_weight), 0.1),
        "--distill-weight": (float(args.distill_weight), 0.0),
        "--evssm-fidelity-weight": (float(args.evssm_fidelity_weight), 0.1),
        "--temporal-delta-weight": (float(args.temporal_delta_weight), 0.05),
        "--edge-weight": (float(args.edge_weight), 0.05),
        "--laplacian-gate-weight": (float(args.laplacian_gate_weight), 0.02),
        "--grad-clip": (float(args.grad_clip), 1.0),
    }
    for label, (actual, expected) in exact_base_loss_values.items():
        if not math.isfinite(actual) or actual != expected:
            raise ValueError(
                f"motion-aligned v4 preregisters {label}={expected}, got {actual}"
            )
    if bool(args.amp):
        raise ValueError("motion-aligned v4 preregisters CPU execution with AMP off")
    if bool(args.dry_run):
        raise ValueError("formal motion-aligned v4 forbids --dry-run checkpoints")
    if str(args.device) != "cpu":
        raise ValueError("motion-aligned v4 requires explicit --device cpu")


def dataset_transition_summary(dataset: VideoDeblurJsonlDataset) -> Dict[str, int]:
    """Count real adjacent transition slots without loading any image tensors."""

    slots = 0
    unique = set()
    for sequence_index, frame_indices in dataset.clips:
        for left, right in zip(frame_indices[:-1], frame_indices[1:]):
            if int(right) == int(left) + 1:
                slots += 1
                unique.add((int(sequence_index), int(left), int(right)))
    return {
        "train_clips": int(len(dataset)),
        "train_sequences": int(len(dataset.sequences)),
        "real_transition_slots": int(slots),
        "unique_real_transitions": int(len(unique)),
        "alignment_sampler_clips": int(len(real_transition_clip_indices(dataset))),
    }


def validate_v4_registered_contract() -> Dict[str, str]:
    """Bind a formal run to the immutable pre-training contract bytes."""

    if not V4_REGISTERED_CONTRACT_PATH.is_file():
        raise FileNotFoundError(
            f"registered v4 contract is missing: {V4_REGISTERED_CONTRACT_PATH}"
        )
    actual = sha256_file(V4_REGISTERED_CONTRACT_PATH)
    if actual != V4_REGISTERED_CONTRACT_SHA256:
        raise ValueError(
            "registered v4 experiment contract SHA-256 mismatch; refuse training"
        )
    return {
        "schema": V4_REGISTERED_CONTRACT_SCHEMA,
        "path": str(V4_REGISTERED_CONTRACT_PATH.resolve()),
        "sha256": actual,
    }


def validate_v4_training_inventory(summary: Mapping[str, int]) -> None:
    expected = {
        "train_clips": V4_EXPECTED_TRAIN_CLIPS,
        "train_sequences": V4_EXPECTED_TRAIN_SEQUENCES,
        "real_transition_slots": V4_EXPECTED_REAL_TRANSITION_SLOTS,
        "unique_real_transitions": V4_EXPECTED_UNIQUE_REAL_TRANSITIONS,
        "alignment_sampler_clips": V4_EXPECTED_UNIQUE_REAL_TRANSITIONS,
    }
    for key, value in expected.items():
        actual = int(summary.get(key, -1))
        if actual != value:
            raise ValueError(
                f"v4 training inventory {key} must be {value}, got {actual}"
            )


def validate_v4_data_identity(
    args: argparse.Namespace,
    *,
    train_set: VideoDeblurJsonlDataset,
    val_set: Optional[VideoDeblurJsonlDataset],
    teacher_provenance: Mapping[str, object],
) -> Dict[str, str]:
    """Pin both optimization and temporal-validation data before training."""

    if args.teacher_checkpoint is not None:
        raise ValueError(
            "formal v4 requires the preregistered cached official EVSSM outputs, "
            "not a runtime teacher checkpoint"
        )
    if args.train_precompute_report is None or args.val_precompute_report is None:
        raise ValueError("formal v4 requires exact train/val EVSSM precompute reports")
    if val_set is None:
        raise ValueError("formal v4 requires the preregistered temporal validation set")
    train_provenance = train_set.teacher_provenance
    val_provenance = val_set.teacher_provenance
    if not isinstance(train_provenance, dict) or not isinstance(val_provenance, dict):
        raise ValueError("formal v4 requires content-bound cached teacher provenance")

    def source_manifest_digest(report_value: Path, label: str) -> str:
        report_path = report_value.expanduser().resolve()
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if not isinstance(report, dict):
            raise ValueError(f"formal v4 {label} precompute report is malformed")
        digest = str(report.get("input_manifest_sha256", "")).lower()
        source_value = report.get("input_manifest")
        if not source_value:
            raise ValueError(f"formal v4 {label} report has no input_manifest")
        source_path = Path(str(source_value)).expanduser()
        if not source_path.is_absolute():
            source_path = report_path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_file() or sha256_file(source_path) != digest:
            raise ValueError(
                f"formal v4 {label} source manifest is missing or SHA-mismatched"
            )
        return digest

    actual = {
        "train_manifest_sha256": source_manifest_digest(
            args.train_precompute_report, "train"
        ),
        "train_precompute_report_sha256": str(
            train_provenance.get("precompute_report_sha256", "")
        ),
        "train_teacher_manifest_sha256": str(
            sha256_file(train_set.manifest)
        ),
        "val_manifest_sha256": source_manifest_digest(
            args.val_precompute_report, "validation"
        ),
        "val_precompute_report_sha256": str(
            val_provenance.get("precompute_report_sha256", "")
        ),
        "val_teacher_manifest_sha256": str(
            sha256_file(val_set.manifest)
        ),
        "evssm_checkpoint_sha256": str(
            teacher_provenance.get("evssm_checkpoint_sha256", "")
        ),
    }
    expected = {
        "train_manifest_sha256": V4_EXPECTED_TRAIN_MANIFEST_SHA256,
        "train_precompute_report_sha256": V4_EXPECTED_TRAIN_PRECOMPUTE_SHA256,
        "train_teacher_manifest_sha256": (
            V4_EXPECTED_TRAIN_TEACHER_MANIFEST_SHA256
        ),
        "val_manifest_sha256": V4_EXPECTED_VAL_MANIFEST_SHA256,
        "val_precompute_report_sha256": V4_EXPECTED_VAL_PRECOMPUTE_SHA256,
        "val_teacher_manifest_sha256": V4_EXPECTED_VAL_TEACHER_MANIFEST_SHA256,
        "evssm_checkpoint_sha256": V4_EXPECTED_EVSSM_SHA256,
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(
                f"formal v4 data identity {key} mismatch: {actual[key]} != {value}"
            )
    if train_provenance.get("storage") != "precomputed_png_rgb8" or (
        val_provenance.get("storage") != "precomputed_png_rgb8"
    ):
        raise ValueError("formal v4 teacher storage must be precomputed PNG RGB8")
    if str(train_provenance.get("teacher_manifest_sha256", "")) != actual[
        "train_teacher_manifest_sha256"
    ] or str(val_provenance.get("teacher_manifest_sha256", "")) != actual[
        "val_teacher_manifest_sha256"
    ]:
        raise ValueError("formal v4 output teacher manifest provenance mismatch")
    if str(val_provenance.get("evssm_checkpoint_sha256", "")) != (
        V4_EXPECTED_EVSSM_SHA256
    ):
        raise ValueError("formal v4 validation EVSSM checkpoint SHA-256 mismatch")
    return actual


def real_transition_clip_indices(
    dataset: VideoDeblurJsonlDataset,
) -> List[int]:
    """Indices of clips containing at least one physical adjacent edge."""

    indices: List[int] = []
    for clip_index, (_, frame_indices) in enumerate(dataset.clips):
        if any(
            int(right) == int(left) + 1
            for left, right in zip(frame_indices[:-1], frame_indices[1:])
        ):
            indices.append(int(clip_index))
    return indices


def is_v4_alignment_state_key(name: str) -> bool:
    return name in V4_ALIGNMENT_STATE_EXACT or name.startswith(
        V4_ALIGNMENT_STATE_PREFIXES
    )


def split_v4_named_parameters(
    model: nn.Module,
) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
    """Return exact base/alignment parameter partitions for phase control."""

    base: List[Tuple[str, nn.Parameter]] = []
    alignment: List[Tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if is_v4_alignment_state_key(name):
            alignment.append((name, parameter))
        else:
            base.append((name, parameter))
    alignment_names = {name for name, _ in alignment}
    expected = {
        "motion_alignment_gate",
        "motion_aligner.match_projection.weight",
    }
    if alignment_names != expected:
        raise ValueError(
            "v4 model alignment parameters do not match the registered API: "
            f"{sorted(alignment_names)} != {sorted(expected)}"
        )
    if not base:
        raise ValueError("v4 model has no base parameters")
    return base, alignment


def build_v4_optimizer(
    model: nn.Module,
    *,
    base_lr: float = V4_BASE_LR,
    alignment_lr: float = V4_ALIGNMENT_LR,
    weight_decay: float = V4_WEIGHT_DECAY,
) -> torch.optim.Optimizer:
    base, alignment = split_v4_named_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for _, parameter in base],
                "lr": float(base_lr),
                "group_name": "base",
            },
            {
                "params": [parameter for _, parameter in alignment],
                "lr": float(alignment_lr),
                "group_name": "alignment",
            },
        ],
        weight_decay=float(weight_decay),
        betas=(0.9, 0.9),
    )
    configure_v4_phase(
        model,
        optimizer,
        "alignment_only",
        base_lr=base_lr,
        alignment_lr=alignment_lr,
    )
    return optimizer


def configure_v4_phase(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    phase: str,
    *,
    base_lr: float = V4_BASE_LR,
    alignment_lr: float = V4_ALIGNMENT_LR,
) -> None:
    """Freeze/unfreeze the exact parameter partitions and set phase LRs."""

    if phase not in {"alignment_only", "joint"}:
        raise ValueError(f"unsupported v4 phase {phase!r}")
    base, alignment = split_v4_named_parameters(model)
    base_trainable = phase == "joint"
    for _, parameter in base:
        parameter.requires_grad_(base_trainable)
    for name, parameter in alignment:
        parameter.requires_grad_(
            base_trainable or name == "motion_aligner.match_projection.weight"
        )
    groups = {str(group.get("group_name", "")): group for group in optimizer.param_groups}
    if set(groups) != {"base", "alignment"}:
        raise ValueError("v4 optimizer must contain exact base/alignment groups")
    groups["base"]["lr"] = float(base_lr) if base_trainable else 0.0
    groups["alignment"]["lr"] = float(alignment_lr)


def v4_phase_for_step(
    optimizer_step: int, alignment_only_steps: int = V4_ALIGNMENT_ONLY_STEPS
) -> str:
    if optimizer_step < 0:
        raise ValueError("optimizer_step cannot be negative")
    return "alignment_only" if optimizer_step < alignment_only_steps else "joint"


def v4_epoch_end_step(epoch: int) -> int:
    """Exact legal epoch-boundary step under 13/29 updates per full loader."""

    if 0 <= epoch <= 6:
        return (epoch + 1) * 13
    if epoch == 7:
        return V4_ALIGNMENT_ONLY_STEPS
    if 8 <= epoch <= 24:
        return V4_ALIGNMENT_ONLY_STEPS + (epoch - 7) * 29
    if epoch == 25:
        return V4_TOTAL_STEPS
    raise ValueError("v4 checkpoint epoch must be in [0,25]")


def state_key_digest(state: Mapping[str, object]) -> str:
    """Digest ordered state keys plus tensor shape/dtype, not tensor values."""

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("model state must map string keys to tensors")
        descriptor = json.dumps(
            {
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(descriptor.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def v3_base_config_from_v4(model_config: Mapping[str, object]) -> Dict[str, object]:
    config = dict(model_config)
    motion = config.pop("motion_alignment", None)
    if motion != V4_MOTION_ALIGNMENT_CONFIG:
        raise ValueError(
            "v4 model_config.motion_alignment does not match the registered config"
        )
    return config


def _load_checkpoint_payload(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    role: str = "checkpoint",
) -> Tuple[Dict[str, object], str, Path]:
    """Hash one immutable byte snapshot, then deserialize it weights-only.

    Formal v4 checkpoint inputs are never loaded through Python's unrestricted
    pickle path.  Reading the complete file once also binds a registered digest
    to the exact bytes passed to ``torch.load`` instead of checking one path
    snapshot and later loading another.
    """

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
    serialized = resolved.read_bytes()
    actual_sha256 = hashlib.sha256(serialized).hexdigest()
    if expected_sha256 is not None:
        registered_sha256 = str(expected_sha256).lower()
        if (
            len(registered_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in registered_sha256
            )
        ):
            raise ValueError(f"{role} expected SHA-256 is malformed")
        if actual_sha256 != registered_sha256:
            raise ValueError(
                f"{role} SHA-256 mismatch: expected {registered_sha256}, "
                f"got {actual_sha256}"
            )
    try:
        payload = torch.load(
            io.BytesIO(serialized),
            map_location="cpu",
            weights_only=True,
        )
    except (pickle.UnpicklingError, RuntimeError, EOFError, TypeError) as error:
        raise ValueError(
            f"{role} was rejected by torch.load(weights_only=True); formal v4 "
            "never falls back to unsafe pickle loading. Migrate legacy v4 "
            "checkpoints containing a NumPy ndarray RNG state with "
            "scripts/migrate_v4_rng_checkpoint.py before --resume"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return payload, actual_sha256, resolved


def load_v4_warm_start(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    identity_atol: float = 1.0e-6,
    expected_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Strictly copy v3 model tensors and prove the zero-gate v4 identity."""

    if identity_atol <= 0.0 or not math.isfinite(float(identity_atol)):
        raise ValueError("identity_atol must be finite and positive")
    bare = model.module if hasattr(model, "module") else model
    if not hasattr(bare, "config_dict"):
        raise ValueError("v4 model must expose config_dict")
    target_config = dict(bare.config_dict())
    expected_v3_config = v3_base_config_from_v4(target_config)

    checkpoint, source_sha256, resolved = _load_checkpoint_payload(
        checkpoint_path,
        expected_sha256=expected_sha256,
        role="v4 warm-start checkpoint",
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT_V3:
        raise ValueError("--warm-start-v3 requires an exact v3 checkpoint format")
    source_config = checkpoint.get("model_config")
    if not isinstance(source_config, dict) or dict(source_config) != expected_v3_config:
        raise ValueError(
            "warm-start v3 model_config does not exactly match the v4 base config"
        )
    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("warm-start checkpoint is missing a model state dictionary")
    # Validate tensor/key structure before calling load_state_dict so malformed
    # or optimizer-like mappings cannot partially mutate the target model.
    source_key_digest = state_key_digest(source_state)
    target_state = bare.state_dict()
    expected_alignment_keys = {
        "motion_alignment_gate",
        "motion_aligner.match_projection.weight",
        "motion_aligner.offsets",
    }
    actual_alignment_keys = {
        key for key in target_state if is_v4_alignment_state_key(key)
    }
    if actual_alignment_keys != expected_alignment_keys:
        raise ValueError(
            "v4 target alignment state keys do not match the registered API"
        )
    expected_source_keys = set(target_state) - expected_alignment_keys
    if set(source_state) != expected_source_keys:
        missing = sorted(expected_source_keys - set(source_state))
        unexpected = sorted(set(source_state) - expected_source_keys)
        raise ValueError(
            "warm-start source model keys are not exactly the v4 shared keys; "
            f"missing={missing}, unexpected={unexpected}"
        )

    incompatible = bare.load_state_dict(source_state, strict=False)
    if set(incompatible.missing_keys) != expected_alignment_keys or (
        incompatible.unexpected_keys
    ):
        raise ValueError(
            "warm-start load produced a non-registered missing/unexpected key set"
        )
    loaded_state = bare.state_dict()
    for key in expected_source_keys:
        if not torch.equal(loaded_state[key].cpu(), source_state[key].cpu()):
            raise RuntimeError(f"warm-start tensor copy was not exact for {key}")
    gate = loaded_state["motion_alignment_gate"]
    if gate.numel() != 1 or float(gate.detach().cpu().item()) != 0.0:
        raise ValueError("v4 motion_alignment_gate must remain exactly zero at warm-start")

    try:
        target_device = next(bare.parameters()).device
    except StopIteration as error:
        raise ValueError("v4 model has no parameters") from error
    source_model = build_causal_video_deblur(expected_v3_config).to(
        target_device
    ).eval()
    source_model.load_state_dict(source_state, strict=True)
    target_was_training = bare.training
    bare.eval()
    history = int(expected_v3_config["max_history"])
    probe_shape = (1, history, 3, 12, 16)
    probe = torch.linspace(
        0.0,
        1.0,
        steps=math.prod(probe_shape),
        dtype=torch.float32,
    ).reshape(probe_shape).to(target_device)
    with torch.no_grad():
        source_output = source_model.forward_sequence(probe)
        target_output = bare.forward_sequence(probe)
    if not bool(torch.isfinite(source_output).all()) or not bool(
        torch.isfinite(target_output).all()
    ):
        raise RuntimeError("warm-start identity probe produced non-finite output")
    max_abs_difference = float(
        (source_output - target_output).abs().max().detach().cpu().item()
    )
    passed = bool(
        torch.allclose(
            source_output,
            target_output,
            atol=float(identity_atol),
            rtol=0.0,
        )
    )
    if target_was_training:
        bare.train()
    if not passed:
        raise RuntimeError(
            "zero-gate v4 does not reproduce the warm-start v3 checkpoint; "
            f"max_abs_difference={max_abs_difference}"
        )
    return {
        "schema": WARM_START_SCHEMA_V4,
        "source_path": str(resolved),
        "source_sha256": source_sha256,
        "source_format": CHECKPOINT_FORMAT_V3,
        "source_model_config": dict(source_config),
        "source_state_key_digest_sha256": source_key_digest,
        "copied_key_count": int(len(expected_source_keys)),
        "allowed_missing_alignment_keys": sorted(expected_alignment_keys),
        "optimizer_state_loaded": False,
        "identity_probe": {
            "shape": list(probe_shape),
            "atol": float(identity_atol),
            "rtol": 0.0,
            "max_abs_difference": max_abs_difference,
            "passed": True,
        },
    }


def _no_teacher_provenance() -> Dict[str, object]:
    return {
        "schema": TEACHER_PROVENANCE_SCHEMA,
        "storage": "none",
        "teacher_domain": "none",
        "evssm_checkpoint_sha256": None,
    }


def resolve_teacher_provenance(
    *,
    dataset: VideoDeblurJsonlDataset,
    teacher_checkpoint: Optional[Path],
    distill_weight: float,
    input_domain: str,
    evssm_fidelity_weight: float = 0.0,
) -> Dict[str, object]:
    """Describe the EVSSM signal that is actually consumed during training."""

    uses_teacher = (
        float(distill_weight) > 0.0
        or float(evssm_fidelity_weight) > 0.0
        or str(input_domain) == "evssm"
    )
    cached = dataset.teacher_provenance
    if teacher_checkpoint is not None:
        checkpoint = teacher_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"EVSSM teacher checkpoint does not exist: {checkpoint}")
        checkpoint_sha = sha256_file(checkpoint)
        if cached is not None and cached["evssm_checkpoint_sha256"] != checkpoint_sha:
            raise ValueError(
                "runtime EVSSM teacher SHA-256 does not match the precompute report"
            )
        return {
            "schema": TEACHER_PROVENANCE_SCHEMA,
            "storage": "runtime_evssm_float_tensor",
            "teacher_domain": "evssm_restored_rgb_0_1",
            "evssm_checkpoint": str(checkpoint),
            "evssm_checkpoint_sha256": checkpoint_sha,
        }
    if uses_teacher:
        if cached is None:
            raise ValueError(
                "cached EVSSM training requires --precompute-report so the "
                "teacher checkpoint SHA, storage, and manifest are verified"
            )
        return dict(cached)
    return _no_teacher_provenance()


def validate_validation_teacher_provenance(
    training: Dict[str, object],
    validation_set: Optional[VideoDeblurJsonlDataset],
    *,
    teacher_checkpoint: Optional[Path],
    input_domain: str,
) -> None:
    if validation_set is None or str(input_domain) != "evssm":
        return
    if teacher_checkpoint is not None:
        return
    validation = validation_set.teacher_provenance
    if validation is None:
        raise ValueError(
            "EVSSM-domain validation with cached teachers requires "
            "--val-precompute-report"
        )
    if validation["evssm_checkpoint_sha256"] != training["evssm_checkpoint_sha256"]:
        raise ValueError("train/validation cached EVSSM checkpoint SHA-256 mismatch")


def set_seed(seed: int, *, seed_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_v4_rng_state(
    train_loader_generator: torch.Generator,
    alignment_loader_generator: torch.Generator,
) -> Dict[str, object]:
    """Snapshot every RNG that affects the next CPU epoch."""

    numpy_random_state = np.random.get_state()
    # PyTorch 2.3 exposes ``torch.uint32`` but cannot serialize its storage.
    # Store the MT19937 uint32 words losslessly in a weights-only-safe int64
    # tensor instead of retaining NumPy's pickle-dependent ndarray.
    safe_numpy_random_state = (
        numpy_random_state[0],
        torch.from_numpy(numpy_random_state[1].astype(np.int64, copy=True)),
        int(numpy_random_state[2]),
        int(numpy_random_state[3]),
        float(numpy_random_state[4]),
    )

    return {
        "schema": RNG_STATE_SCHEMA_V4,
        "checkpoint_boundary": "epoch_end_no_pending_accumulation",
        "python_random_state": random.getstate(),
        "numpy_random_state": safe_numpy_random_state,
        "numpy_random_state_encoding": NUMPY_RNG_ENCODING_V4,
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "train_loader_generator_state": train_loader_generator.get_state().clone(),
        "alignment_loader_generator_state": (
            alignment_loader_generator.get_state().clone()
        ),
    }


def restore_v4_rng_state(
    state: object,
    *,
    train_loader_generator: torch.Generator,
    alignment_loader_generator: torch.Generator,
) -> None:
    state = validate_v4_rng_state(state)
    try:
        random.setstate(state["python_random_state"])
        np.random.set_state(numpy_rng_state_as_numpy(state["numpy_random_state"]))
        torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
        train_loader_generator.set_state(
            state["train_loader_generator_state"].cpu()
        )
        alignment_loader_generator.set_state(
            state["alignment_loader_generator_state"].cpu()
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("v4 RNG state could not be restored") from error


def validate_numpy_rng_state(state: object) -> Tuple[object, ...]:
    """Validate either the safe tensor or legacy ndarray MT19937 payload."""

    if not isinstance(state, tuple) or len(state) != 5:
        raise ValueError("v4 NumPy RNG state must be a five-item tuple")
    algorithm, keys, position, has_gauss, cached_gaussian = state
    if algorithm != "MT19937":
        raise ValueError("v4 NumPy RNG state must use MT19937")
    if isinstance(keys, torch.Tensor):
        if (
            keys.dtype != torch.int64
            or keys.device.type != "cpu"
            or keys.ndim != 1
            or keys.numel() != 624
            or not keys.is_contiguous()
        ):
            raise ValueError(
                "safe v4 NumPy RNG keys must be a contiguous CPU "
                "624-element int64 tensor"
            )
        if bool((keys < 0).any().item()) or bool(
            (keys > int(np.iinfo(np.uint32).max)).any().item()
        ):
            raise ValueError("safe v4 NumPy RNG keys exceed the uint32 range")
    elif isinstance(keys, np.ndarray):
        if keys.dtype != np.uint32 or keys.ndim != 1 or keys.size != 624:
            raise ValueError(
                "legacy v4 NumPy RNG keys must be a 624-element uint32 ndarray"
            )
    else:
        raise ValueError(
            "v4 NumPy RNG keys must be a safe int64 tensor or legacy uint32 ndarray"
        )
    if type(position) is not int or not 0 <= position <= 624:
        raise ValueError("v4 NumPy RNG position must be an integer in [0,624]")
    if type(has_gauss) is not int or has_gauss not in {0, 1}:
        raise ValueError("v4 NumPy RNG has_gauss must be 0 or 1")
    if type(cached_gaussian) is not float or not math.isfinite(
        float(cached_gaussian)
    ):
        raise ValueError("v4 NumPy RNG cached Gaussian must be finite")
    return state


def numpy_rng_state_as_safe_tensor(state: object) -> Tuple[object, ...]:
    """Return a torch-2.3 weights-only-safe, lossless MT19937 tuple."""

    algorithm, keys, position, has_gauss, cached_gaussian = (
        validate_numpy_rng_state(state)
    )
    if isinstance(keys, torch.Tensor):
        safe_keys = keys.detach().cpu().clone()
    else:
        safe_keys = torch.from_numpy(keys.astype(np.int64, copy=True))
    return (
        algorithm,
        safe_keys,
        int(position),
        int(has_gauss),
        float(cached_gaussian),
    )


def numpy_rng_state_as_numpy(state: object) -> Tuple[object, ...]:
    """Return the exact tuple shape accepted by ``np.random.set_state``."""

    algorithm, keys, position, has_gauss, cached_gaussian = (
        validate_numpy_rng_state(state)
    )
    if isinstance(keys, torch.Tensor):
        numpy_keys = keys.detach().cpu().numpy().astype(np.uint32, copy=True)
    else:
        numpy_keys = keys.copy()
    return (
        algorithm,
        numpy_keys,
        int(position),
        int(has_gauss),
        float(cached_gaussian),
    )


def validate_v4_rng_state(state: object) -> Dict[str, object]:
    if not isinstance(state, dict) or state.get("schema") != RNG_STATE_SCHEMA_V4:
        raise ValueError("v4 resume is missing the registered RNG state")
    if state.get("checkpoint_boundary") != "epoch_end_no_pending_accumulation":
        raise ValueError("v4 resume is not at a clean epoch/accumulation boundary")
    required_tensors = (
        "torch_cpu_rng_state",
        "train_loader_generator_state",
        "alignment_loader_generator_state",
    )
    for key in required_tensors:
        value = state.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.device.type != "cpu"
            or value.ndim != 1
            or value.numel() < 1
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"v4 RNG state {key} must be a nonempty contiguous CPU "
                "uint8 vector"
            )
    if not isinstance(state.get("python_random_state"), tuple):
        raise ValueError("v4 Python RNG state is malformed")
    numpy_random_state = validate_numpy_rng_state(
        state.get("numpy_random_state")
    )
    numpy_keys = numpy_random_state[1]
    encoding = state.get("numpy_random_state_encoding")
    if isinstance(numpy_keys, torch.Tensor):
        if encoding != NUMPY_RNG_ENCODING_V4:
            raise ValueError("safe v4 NumPy RNG state has no exact encoding tag")
    elif encoding not in {None, NUMPY_RNG_ENCODING_V4}:
        raise ValueError("legacy v4 NumPy RNG state has an unknown encoding tag")
    return dict(state)


def fft_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Real/imaginary FFT L1 with an explicit orthonormal scale contract."""
    prediction_fft = torch.fft.fft2(
        prediction.float(), dim=(-2, -1), norm="ortho"
    )
    target_fft = torch.fft.fft2(target.float(), dim=(-2, -1), norm="ortho")
    prediction_parts = torch.stack((prediction_fft.real, prediction_fft.imag), dim=-1)
    target_parts = torch.stack((target_fft.real, target_fft.imag), dim=-1)
    return F.l1_loss(prediction_parts, target_parts)


def rolling_window_temporal_delta_l1_loss(
    current_prediction: torch.Tensor,
    previous_prediction: torch.Tensor,
    sharp_sequence: torch.Tensor,
) -> torch.Tensor:
    """Match two adjacent full-history streaming outputs to the GT delta.

    ``sharp_sequence`` contains H+1 ordered frames.  The caller evaluates the
    model on ``frames[:, :-1]`` and ``frames[:, 1:]`` separately, exactly as
    two consecutive runtime calls with H-frame rolling windows would do.
    """
    if current_prediction.shape != previous_prediction.shape:
        raise ValueError("rolling predictions must have matching BCHW shapes")
    if current_prediction.dim() != 4:
        raise ValueError("rolling predictions must be BCHW tensors")
    if sharp_sequence.dim() != 5 or sharp_sequence.shape[1] < 2:
        raise ValueError("sharp_sequence must be BTCHW with at least two frames")
    if sharp_sequence[:, -1].shape != current_prediction.shape:
        raise ValueError("sharp_sequence and rolling predictions do not match")
    prediction_delta = current_prediction - previous_prediction
    target_delta = sharp_sequence[:, -1] - sharp_sequence[:, -2]
    return F.l1_loss(prediction_delta, target_delta)


def stitch_rolling_pairwise_diagnostics(
    previous_window: torch.Tensor, current_window: torch.Tensor
) -> torch.Tensor:
    """Cover every H+1 transition once from two H-frame diagnostic calls."""

    if previous_window.dim() < 2 or current_window.dim() != previous_window.dim():
        raise ValueError("rolling diagnostics must have matching B,T,... ranks")
    if previous_window.shape != current_window.shape:
        raise ValueError("rolling diagnostics must have matching shapes")
    if previous_window.shape[1] == 0:
        # H1 has no within-window pair in either call.  v4 rejects H1, while
        # this exact empty contract remains useful to dataset/unit tests.
        return previous_window
    return torch.cat((previous_window, current_window[:, -1:]), dim=1)


def warp_with_pixel_flow(
    source: torch.Tensor, flow_current_to_source: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiably warp BCHW source with (dx,dy) flow in source pixels."""

    if source.dim() != 4:
        raise ValueError("source must be BCHW")
    if flow_current_to_source.dim() != 4 or flow_current_to_source.shape[1] != 2:
        raise ValueError("flow must have shape [B,2,H,W]")
    if source.shape[0] != flow_current_to_source.shape[0] or source.shape[-2:] != (
        flow_current_to_source.shape[-2:]
    ):
        raise ValueError("source/flow batch and spatial shapes must match")
    batch, _, height, width = source.shape
    flow = flow_current_to_source.to(dtype=source.dtype)
    base_x = torch.arange(width, dtype=source.dtype, device=source.device).reshape(
        1, 1, 1, width
    )
    base_y = torch.arange(height, dtype=source.dtype, device=source.device).reshape(
        1, 1, height, 1
    )
    sample_x = base_x + flow[:, 0:1]
    sample_y = base_y + flow[:, 1:2]
    valid = (
        (sample_x >= 0.0)
        & (sample_x <= float(width - 1))
        & (sample_y >= 0.0)
        & (sample_y <= float(height - 1))
    )
    grid_x = 2.0 * (sample_x + 0.5) / float(width) - 1.0
    grid_y = 2.0 * (sample_y + 0.5) / float(height) - 1.0
    grid = torch.stack((grid_x[:, 0], grid_y[:, 0]), dim=-1)
    warped = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return warped, valid.to(dtype=source.dtype)


def _differentiable_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    zero_sources: Sequence[torch.Tensor],
) -> torch.Tensor:
    expanded_mask = mask.to(dtype=values.dtype).expand_as(values)
    denominator = expanded_mask.sum()
    if bool((denominator.detach() > 0).item()):
        return (values * expanded_mask).sum() / denominator
    zero = values.sum() * 0.0
    for source in zero_sources:
        zero = zero + source.sum() * 0.0
    return zero


def motion_alignment_auxiliary_losses(
    sharp_sequence: torch.Tensor,
    flow_current_to_previous: torch.Tensor,
    transition_valid: torch.Tensor,
    diagnostic_valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Photo/gradient/smooth losses on real, non-padding adjacent edges only."""

    if sharp_sequence.dim() != 5 or sharp_sequence.shape[2] != 3:
        raise ValueError("sharp_sequence must be [B,T+1,3,H,W]")
    if flow_current_to_previous.dim() != 5 or flow_current_to_previous.shape[2] != 2:
        raise ValueError("flow must be [B,T,2,h,w]")
    batch, transitions = flow_current_to_previous.shape[:2]
    if sharp_sequence.shape[0] != batch or sharp_sequence.shape[1] != transitions + 1:
        raise ValueError("sharp sequence length must be flow transitions + 1")
    if transition_valid.shape != (batch, transitions):
        raise ValueError("transition_valid must have shape [B,T]")
    if diagnostic_valid is not None and diagnostic_valid.shape != (
        batch,
        transitions,
        1,
        flow_current_to_previous.shape[-2],
        flow_current_to_previous.shape[-1],
    ):
        raise ValueError("diagnostic_valid shape does not match flow")

    if transitions == 0:
        zero = flow_current_to_previous.sum() * 0.0 + sharp_sequence.sum() * 0.0
        return zero, zero, zero, zero.detach()

    flow_height, flow_width = flow_current_to_previous.shape[-2:]
    sharp_low = F.interpolate(
        sharp_sequence.reshape(
            batch * (transitions + 1),
            3,
            sharp_sequence.shape[-2],
            sharp_sequence.shape[-1],
        ),
        size=(flow_height, flow_width),
        mode="area",
    ).reshape(batch, transitions + 1, 3, flow_height, flow_width)
    previous = sharp_low[:, :-1].reshape(
        batch * transitions, 3, flow_height, flow_width
    )
    current = sharp_low[:, 1:].reshape(
        batch * transitions, 3, flow_height, flow_width
    )
    flat_flow = flow_current_to_previous.reshape(
        batch * transitions, 2, flow_height, flow_width
    )
    warped_previous, geometric_valid = warp_with_pixel_flow(previous, flat_flow)
    warped_previous = warped_previous.reshape(
        batch, transitions, 3, flow_height, flow_width
    )
    current = current.reshape(batch, transitions, 3, flow_height, flow_width)
    geometric_valid = geometric_valid.reshape(
        batch, transitions, 1, flow_height, flow_width
    )
    mask = transition_valid.reshape(batch, transitions, 1, 1, 1).to(
        dtype=geometric_valid.dtype
    ) * geometric_valid
    if diagnostic_valid is not None:
        mask = mask * diagnostic_valid.to(dtype=mask.dtype)

    previous_gray = warped_previous.mean(dim=2, keepdim=True)
    current_gray = current.mean(dim=2, keepdim=True)
    photo_values = torch.sqrt(
        (previous_gray - current_gray).square() + 1.0e-6
    )
    photo = _differentiable_masked_mean(
        photo_values, mask, (flow_current_to_previous,)
    )

    gradient_terms: List[torch.Tensor] = []
    if flow_width > 1:
        mask_x = mask[..., 1:] * mask[..., :-1]
        previous_dx = previous_gray[..., 1:] - previous_gray[..., :-1]
        current_dx = current_gray[..., 1:] - current_gray[..., :-1]
        gradient_terms.append(
            _differentiable_masked_mean(
                (previous_dx - current_dx).abs(),
                mask_x,
                (flow_current_to_previous,),
            )
        )
    if flow_height > 1:
        mask_y = mask[..., 1:, :] * mask[..., :-1, :]
        previous_dy = previous_gray[..., 1:, :] - previous_gray[..., :-1, :]
        current_dy = current_gray[..., 1:, :] - current_gray[..., :-1, :]
        gradient_terms.append(
            _differentiable_masked_mean(
                (previous_dy - current_dy).abs(),
                mask_y,
                (flow_current_to_previous,),
            )
        )
    gradient = (
        sum(gradient_terms) / float(len(gradient_terms))
        if gradient_terms
        else photo * 0.0
    )

    flow = flow_current_to_previous
    smooth_terms: List[torch.Tensor] = []
    if flow_width > 1:
        flow_dx = flow[..., 1:] - flow[..., :-1]
        image_dx = (current_gray[..., 1:] - current_gray[..., :-1]).abs()
        smooth_terms.append(
            _differentiable_masked_mean(
                flow_dx.abs() * torch.exp(-10.0 * image_dx),
                mask[..., 1:] * mask[..., :-1],
                (flow_current_to_previous,),
            )
        )
    if flow_height > 1:
        flow_dy = flow[..., 1:, :] - flow[..., :-1, :]
        image_dy = (current_gray[..., 1:, :] - current_gray[..., :-1, :]).abs()
        smooth_terms.append(
            _differentiable_masked_mean(
                flow_dy.abs() * torch.exp(-10.0 * image_dy),
                mask[..., 1:, :] * mask[..., :-1, :],
                (flow_current_to_previous,),
            )
        )
    smooth = (
        sum(smooth_terms) / float(len(smooth_terms))
        if smooth_terms
        else photo * 0.0
    )
    real_transition_count = transition_valid.to(dtype=photo.dtype).sum().detach()
    return photo, gradient, smooth, real_transition_count


def motion_compensated_temporal_delta_l1_loss(
    current_prediction: torch.Tensor,
    previous_prediction: torch.Tensor,
    current_sharp: torch.Tensor,
    previous_sharp: torch.Tensor,
    flow_current_to_previous: torch.Tensor,
    transition_valid: torch.Tensor,
    diagnostic_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Match motion-compensated deltas while preventing task loss flow gaming."""

    tensors = (
        current_prediction,
        previous_prediction,
        current_sharp,
        previous_sharp,
    )
    if any(tensor.dim() != 4 for tensor in tensors):
        raise ValueError("prediction/sharp inputs must be BCHW")
    if any(tensor.shape != current_prediction.shape for tensor in tensors[1:]):
        raise ValueError("prediction/sharp inputs must have matching shapes")
    if flow_current_to_previous.dim() != 4 or flow_current_to_previous.shape[1] != 2:
        raise ValueError("flow must be [B,2,h,w]")
    batch, _, height, width = current_prediction.shape
    if flow_current_to_previous.shape[0] != batch:
        raise ValueError("flow batch size mismatch")
    if transition_valid.shape != (batch,):
        raise ValueError("transition_valid must have shape [B]")

    flow_height, flow_width = flow_current_to_previous.shape[-2:]
    full_flow = F.interpolate(
        flow_current_to_previous.detach(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    full_flow = torch.stack(
        (
            full_flow[:, 0] * (float(width) / float(flow_width)),
            full_flow[:, 1] * (float(height) / float(flow_height)),
        ),
        dim=1,
    )
    warped_prediction, geometric_valid = warp_with_pixel_flow(
        previous_prediction, full_flow
    )
    warped_sharp, _ = warp_with_pixel_flow(previous_sharp, full_flow)
    mask = transition_valid.reshape(batch, 1, 1, 1).to(
        dtype=geometric_valid.dtype
    ) * geometric_valid
    if diagnostic_valid is not None:
        if diagnostic_valid.shape != (batch, 1, flow_height, flow_width):
            raise ValueError("diagnostic_valid shape mismatch")
        mask = mask * F.interpolate(
            diagnostic_valid.to(dtype=mask.dtype),
            size=(height, width),
            mode="nearest",
        )
    prediction_delta = current_prediction - warped_prediction
    target_delta = current_sharp - warped_sharp
    return _differentiable_masked_mean(
        (prediction_delta - target_delta).abs(),
        mask,
        (current_prediction, previous_prediction),
    )


def compute_v4_batch_losses(
    model: nn.Module,
    model_input_sequence: torch.Tensor,
    sharp_sequence: torch.Tensor,
    transition_valid: torch.Tensor,
    teacher: Optional[torch.Tensor],
    teacher_available: torch.Tensor,
    args: argparse.Namespace,
    phase: str,
) -> Dict[str, torch.Tensor]:
    """Compute one v4 micro-batch with exact rolling diagnostics and phases."""

    if phase not in {"alignment_only", "joint"}:
        raise ValueError(f"unsupported v4 phase {phase!r}")
    if model_input_sequence.dim() != 5 or sharp_sequence.shape != model_input_sequence.shape:
        raise ValueError("v4 model/sharp sequences must be matching BTCHW tensors")
    batch, clip_length = model_input_sequence.shape[:2]
    if transition_valid.shape != (batch, clip_length - 1):
        raise ValueError("v4 transition_valid must cover every H+1 adjacent slot")
    if teacher_available.shape != (batch,):
        raise ValueError("teacher_available must have shape [B]")
    if not hasattr(model, "forward_sequence_with_motion_diagnostics"):
        raise ValueError("v4 model is missing motion diagnostics API")

    previous_input = model_input_sequence[:, :-1]
    current_input = model_input_sequence[:, 1:]
    previous_diagnostics = model.forward_sequence_with_motion_diagnostics(
        previous_input
    )
    current_diagnostics = model.forward_sequence_with_motion_diagnostics(
        current_input
    )
    previous_prediction_sequence, previous_flow, previous_confidence, previous_valid = (
        previous_diagnostics
    )
    prediction_sequence, current_flow, current_confidence, current_valid = (
        current_diagnostics
    )
    flow = stitch_rolling_pairwise_diagnostics(previous_flow, current_flow)
    confidence = stitch_rolling_pairwise_diagnostics(
        previous_confidence, current_confidence
    )
    motion_valid = stitch_rolling_pairwise_diagnostics(
        previous_valid, current_valid
    )
    photo, motion_gradient, motion_smooth, real_transition_count = (
        motion_alignment_auxiliary_losses(
            sharp_sequence,
            flow,
            transition_valid,
            diagnostic_valid=motion_valid,
        )
    )
    alignment_objective = (
        float(args.v4_alignment_photo_weight) * photo
        + float(args.v4_alignment_gradient_weight) * motion_gradient
        + float(args.v4_alignment_smooth_weight) * motion_smooth
    )
    prediction = prediction_sequence[:, -1]
    previous_prediction = previous_prediction_sequence[:, -1]
    zero = prediction.sum() * 0.0

    l1 = zero
    fft = zero
    temporal_delta = zero
    sharp_edge = zero
    laplacian_gate_hinge = zero
    evssm_fidelity = zero
    distill = zero
    if phase == "alignment_only":
        loss = alignment_objective
    else:
        if current_flow.shape[1] < 1:
            raise RuntimeError("joint v4 phase requires at least one motion transition")
        l1 = F.l1_loss(prediction, sharp_sequence[:, -1])
        fft = fft_l1_loss(prediction, sharp_sequence[:, -1])
        temporal_delta = motion_compensated_temporal_delta_l1_loss(
            prediction,
            previous_prediction,
            sharp_sequence[:, -1],
            sharp_sequence[:, -2],
            current_flow[:, -1],
            transition_valid[:, -1],
            diagnostic_valid=current_valid[:, -1],
        )
        sharp_edge = spatial_gradient_l1_loss(
            prediction, sharp_sequence[:, -1]
        )
        if teacher is not None and bool(teacher_available.all().item()):
            laplacian_gate_hinge = runtime_laplacian_logvar_hinge_loss(
                prediction, teacher[:, -1]
            )
        if float(args.evssm_fidelity_weight) > 0.0:
            if teacher is None:
                raise RuntimeError("v4 EVSSM fidelity requires teacher frames")
            evssm_fidelity = evssm_fidelity_l1_loss(
                prediction_sequence, teacher[:, 1:], teacher_available
            )
        loss = (
            l1
            + float(args.fft_weight) * fft
            + float(args.temporal_delta_weight) * temporal_delta
            + float(args.edge_weight) * sharp_edge
            + float(args.laplacian_gate_weight) * laplacian_gate_hinge
            + float(args.evssm_fidelity_weight) * evssm_fidelity
            + float(args.v4_joint_alignment_weight) * alignment_objective
        )
        if float(args.distill_weight) > 0.0:
            if teacher is None:
                raise RuntimeError("positive distillation requires teacher frames")
            if bool(teacher_available.any().item()):
                distill = F.l1_loss(
                    prediction[teacher_available], teacher[teacher_available, -1]
                )
                loss = loss + float(args.distill_weight) * distill

    confidence_mask = transition_valid.reshape(batch, clip_length - 1, 1, 1, 1)
    confidence_mean = _differentiable_masked_mean(
        confidence,
        confidence_mask.to(dtype=confidence.dtype) * motion_valid,
        (flow,),
    )
    return {
        "loss": loss,
        "prediction_sequence": prediction_sequence,
        "prediction": prediction,
        "previous_prediction": previous_prediction,
        "flow": flow,
        "confidence": confidence,
        "motion_valid": motion_valid,
        "l1": l1,
        "fft": fft,
        "temporal_delta": temporal_delta,
        "sharp_edge": sharp_edge,
        "laplacian_gate_hinge": laplacian_gate_hinge,
        "evssm_fidelity": evssm_fidelity,
        "distill": distill,
        "motion_photo": photo,
        "motion_gradient": motion_gradient,
        "motion_smooth": motion_smooth,
        "motion_alignment_objective": alignment_objective,
        "real_transition_count": real_transition_count,
        "motion_confidence_mean": confidence_mean,
    }


def spatial_gradient_l1_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """First-order gradient plus Laplacian edge fidelity for BCHW tensors."""
    if prediction.shape != target.shape or prediction.dim() != 4:
        raise ValueError("spatial-gradient inputs must be matching BCHW tensors")
    losses = []
    if prediction.shape[-1] > 1:
        prediction_dx = prediction[..., 1:] - prediction[..., :-1]
        target_dx = target[..., 1:] - target[..., :-1]
        losses.append(F.l1_loss(prediction_dx, target_dx))
    if prediction.shape[-2] > 1:
        prediction_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        losses.append(F.l1_loss(prediction_dy, target_dy))
    losses.append(
        F.l1_loss(
            runtime_laplacian_response(prediction),
            runtime_laplacian_response(target),
        )
    )
    if not losses:
        return prediction.sum() * 0.0
    return sum(losses) / float(len(losses))


def runtime_laplacian_response(image: torch.Tensor) -> torch.Tensor:
    """Match ``variance_of_laplacian``: RGB mean then zero-padded convolution."""
    if image.dim() != 4:
        raise ValueError("Laplacian input must be BCHW")
    grayscale = image.mean(dim=1, keepdim=True)
    kernel = image.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).reshape(1, 1, 3, 3)
    return F.conv2d(grayscale, kernel, padding=1)


def runtime_laplacian_variance(image: torch.Tensor) -> torch.Tensor:
    """Per-image variance using the runtime gate's unbiased variance rule."""
    response = runtime_laplacian_response(image).flatten(1)
    if response.shape[1] < 2:
        return response.new_zeros((response.shape[0],))
    return response.var(dim=1, unbiased=True)


def runtime_laplacian_logvar_hinge_loss(
    prediction: torch.Tensor, evssm_reference: torch.Tensor
) -> torch.Tensor:
    """Penalize candidates that would fail the runtime EVSSM sharpness gate."""
    if prediction.shape != evssm_reference.shape or prediction.dim() != 4:
        raise ValueError("Laplacian-retention inputs must be matching BCHW tensors")
    prediction_variance = runtime_laplacian_variance(prediction).clamp_min(1.0e-6)
    evssm_variance = runtime_laplacian_variance(evssm_reference).clamp_min(1.0e-6)
    return F.relu(evssm_variance.log() - prediction_variance.log()).mean()


def evssm_fidelity_l1_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    """Keep the complete predicted prefix close to its EVSSM baseline."""
    if prediction.shape != teacher.shape or prediction.dim() != 5:
        raise ValueError("EVSSM-fidelity inputs must be matching BTCHW tensors")
    if available.dim() != 1 or available.shape[0] != prediction.shape[0]:
        raise ValueError("EVSSM availability must contain one value per batch item")
    if not bool(available.any().item()):
        return prediction.sum() * 0.0
    return F.l1_loss(prediction[available], teacher[available])


def local_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a small dependency-free local SSIM diagnostic for BCHW images."""
    if prediction.shape != target.shape or prediction.dim() != 4:
        raise ValueError("SSIM inputs must be matching BCHW tensors")
    window = min(11, int(prediction.shape[-2]), int(prediction.shape[-1]))
    if window % 2 == 0:
        window -= 1
    window = max(1, window)
    mu_prediction = F.avg_pool2d(prediction, window, stride=1)
    mu_target = F.avg_pool2d(target, window, stride=1)
    prediction_variance = F.avg_pool2d(prediction * prediction, window, stride=1)
    prediction_variance = prediction_variance - mu_prediction.square()
    target_variance = F.avg_pool2d(target * target, window, stride=1)
    target_variance = target_variance - mu_target.square()
    covariance = F.avg_pool2d(prediction * target, window, stride=1)
    covariance = covariance - mu_prediction * mu_target
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_prediction * mu_target + c1) * (
        2.0 * covariance + c2
    )
    denominator = (
        mu_prediction.square() + mu_target.square() + c1
    ) * (prediction_variance + target_variance + c2)
    return (numerator / denominator.clamp_min(1.0e-12)).mean()


def build_objective_contract(args: argparse.Namespace) -> Dict[str, object]:
    if bool(getattr(args, "motion_alignment_v4", False)):
        return {
            "schema": OBJECTIVE_SCHEMA_V4,
            "primary_reconstruction": {
                "frames": "latest_only",
                "l1_weight": 1.0,
                "fft_l1_weight": float(args.fft_weight),
                "fft_normalization": "ortho",
                "phase": "joint_only",
            },
            "evssm_fidelity": {
                "frames": "all_causal_prefix_positions",
                "loss": "l1",
                "weight": float(args.evssm_fidelity_weight),
                "phase": "joint_only",
            },
            "motion_alignment": {
                "flow_direction": "current_to_previous",
                "flow_units": "quarter_resolution_pixels",
                "real_transitions_only": True,
                "padding_transition_policy": "excluded_by_transition_valid",
                "photometric_weight": float(args.v4_alignment_photo_weight),
                "gradient_weight": float(args.v4_alignment_gradient_weight),
                "smooth_weight": float(args.v4_alignment_smooth_weight),
                "joint_phase_scale": float(args.v4_joint_alignment_weight),
                "confidence_weighted": False,
                "phase": "alignment_only_unscaled_and_joint_scaled",
            },
            "temporal_delta": {
                "frames": "two_shifted_full_history_windows",
                "reference": "motion_compensated_sharp_gt_difference",
                "loss": "l1",
                "weight": float(args.temporal_delta_weight),
                "detached_flow": True,
                "flow_direction": "current_to_previous",
                "real_transitions_only": True,
                "phase": "joint_only",
            },
            "edge": {
                "frames": "latest_only",
                "operator": (
                    "first_order_xy_plus_runtime_grayscale_zero_pad_laplacian"
                ),
                "loss": "sharp_gradient_and_laplacian_l1",
                "weight": float(args.edge_weight),
                "phase": "joint_only",
            },
            "laplacian_gate": {
                "reference": "evssm_latest_frame",
                "loss": "relative_log_variance_hinge",
                "minimum_relative_gain": 0.0,
                "variance_floor": 1.0e-6,
                "runtime_gate_alignment": (
                    "rgb_mean_then_four_neighbour_zero_pad_laplacian_unbiased_variance"
                ),
                "weight": float(args.laplacian_gate_weight),
                "phase": "joint_only",
            },
            "legacy_latest_evssm_distillation": {
                "loss": "l1",
                "weight": float(args.distill_weight),
                "phase": "joint_only",
            },
        }
    return {
        "schema": OBJECTIVE_SCHEMA_V3,
        "primary_reconstruction": {
            "frames": "latest_only",
            "l1_weight": 1.0,
            "fft_l1_weight": float(args.fft_weight),
            "fft_normalization": "ortho",
        },
        "evssm_fidelity": {
            "frames": "all_causal_prefix_positions",
            "loss": "l1",
            "weight": float(args.evssm_fidelity_weight),
            "teacher_availability": "required_for_every_sample_when_weight_positive",
        },
        "temporal_delta": {
            "frames": "two_shifted_full_history_windows",
            "reference": "sharp_gt_difference_without_flow_warp",
            "loss": "l1",
            "weight": float(args.temporal_delta_weight),
            "window_contract": "H_plus_1_dataset_clip_to_two_H_frame_calls",
        },
        "edge": {
            "frames": "latest_only",
            "operator": (
                "first_order_xy_plus_runtime_grayscale_zero_pad_laplacian"
            ),
            "loss": "sharp_gradient_and_laplacian_l1",
            "weight": float(args.edge_weight),
        },
        "laplacian_gate": {
            "reference": "evssm_latest_frame",
            "loss": "relative_log_variance_hinge",
            "minimum_relative_gain": 0.0,
            "variance_floor": 1.0e-6,
            "runtime_gate_alignment": (
                "rgb_mean_then_four_neighbour_zero_pad_laplacian_unbiased_variance"
            ),
            "weight": float(args.laplacian_gate_weight),
        },
        "legacy_latest_evssm_distillation": {
            "loss": "l1",
            "weight": float(args.distill_weight),
        },
    }


def build_refinement_contract(model_config: Dict[str, object]) -> Dict[str, object]:
    input_domain = str(model_config["input_domain"])
    if "motion_alignment" in model_config:
        return {
            "schema": REFINEMENT_SCHEMA_V4,
            "base": "frozen_evssm_input" if input_domain == "evssm" else "raw_input",
            "formula": "output = input + max_residual * tanh(residual_logits)",
            "max_residual": float(model_config["max_residual"]),
            "bound_scope": "per_pixel_per_rgb_channel_normalized_0_1",
            "motion_alignment": dict(model_config["motion_alignment"]),
            "flow_bound": {
                "per_component_quarter_resolution_pixels": 16.0,
                "per_component_input_pixels": 64.0,
            },
            "warm_start_identity": (
                "zero_motion_alignment_gate_reproduces_source_v3_with_probe"
            ),
        }
    return {
        "schema": REFINEMENT_SCHEMA_V3,
        "base": "frozen_evssm_input" if input_domain == "evssm" else "raw_input",
        "formula": "output = input + max_residual * tanh(residual_logits)",
        "max_residual": float(model_config["max_residual"]),
        "bound_scope": "per_pixel_per_rgb_channel_normalized_0_1",
        "identity_safe_initialization": "zero_weight_and_bias_output_head",
        "initial_output": "bit_exact_input",
    }


def build_optimization_contract(
    args: argparse.Namespace,
    *,
    max_steps: int,
    optimizer_steps_per_epoch: int,
    execution_device: Optional[str] = None,
    amp_effective: Optional[bool] = None,
) -> Dict[str, object]:
    if bool(getattr(args, "motion_alignment_v4", False)):
        if execution_device != "cpu" or amp_effective is not False:
            raise ValueError("formal v4 optimization requires CPU with AMP disabled")
        alignment_only_steps = int(args.v4_alignment_only_steps)
        return {
            "schema": OPTIMIZATION_SCHEMA_V4,
            "optimizer": "AdamW",
            "total_optimizer_steps": int(max_steps),
            "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
            "gradient_accumulation_micro_batches": int(args.grad_accumulation),
            "batch_size": int(args.batch_size),
            "effective_batch_size": int(
                args.batch_size * args.grad_accumulation
            ),
            "num_workers": int(args.workers),
            "schedule_unit": "optimizer_step",
            "lr_schedule": "fixed_by_phase",
            "optimizer_reset_at_phase_boundary": False,
            "optimizer_state_from_v3_loaded": False,
            "weight_decay": float(args.weight_decay),
            "execution_device": "cpu",
            "amp_requested": bool(args.amp),
            "amp_effective": False,
            "drop_last": True,
            "alignment_loader_clips_per_epoch": 104,
            "alignment_loader_dropped_clips_per_epoch": 3,
            "joint_loader_clips_per_epoch": 232,
            "joint_loader_dropped_clips_per_epoch": 2,
            "alignment_micro_batches_per_epoch": 26,
            "joint_micro_batches_per_epoch": 58,
            "drop_incomplete_accumulation_group": False,
            "loader_generator_seeds": {
                "joint": int(args.seed) + 1000,
                "alignment_only": int(args.seed) + 2000,
            },
            "resume_boundary": "epoch_end_no_pending_accumulation",
            "resume_rng_state": "python_numpy_torch_cpu_and_both_loader_generators",
            "phases": [
                {
                    "name": "alignment_only",
                    "start_step_inclusive": 0,
                    "end_step_exclusive": alignment_only_steps,
                    "optimizer_steps": alignment_only_steps,
                    "base_trainable": False,
                    "base_lr": 0.0,
                    "alignment_lr": float(args.v4_alignment_lr),
                    "trainable_parameters": [
                        "motion_aligner.match_projection.weight"
                    ],
                },
                {
                    "name": "joint",
                    "start_step_inclusive": alignment_only_steps,
                    "end_step_exclusive": int(max_steps),
                    "optimizer_steps": int(max_steps) - alignment_only_steps,
                    "base_trainable": True,
                    "base_lr": float(args.v4_base_lr),
                    "alignment_lr": float(args.v4_alignment_lr),
                    "trainable_parameters": [
                        "base_parameters",
                        "motion_aligner.match_projection.weight",
                        "motion_alignment_gate",
                    ],
                },
            ],
        }
    return {
        "schema": "unblur_slam.causal_video_deblur.optimization.v3",
        "optimizer": "AdamW",
        "optimizer_step_budget": int(max_steps),
        "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
        "gradient_accumulation_micro_batches": int(args.grad_accumulation),
        "lr_schedule": "linear_warmup_then_cosine",
        "warmup_steps": int(args.warmup_steps),
        "schedule_unit": "optimizer_step",
        "base_lr": float(args.lr),
        "minimum_lr": 1.0e-7,
        "weight_decay": float(args.weight_decay),
    }


def load_single_frame_teacher(checkpoint: Path, device: torch.device) -> nn.Module:
    """Load the repository's single-frame EVSSM without coupling the main model to it."""
    try:
        from thirdparty.EVSSM.models.EVSSM import EVSSM
    except Exception as error:
        raise RuntimeError(
            "loading an EVSSM teacher requires the thirdparty/EVSSM dependencies"
        ) from error
    teacher = EVSSM().to(device)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("params", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"unsupported teacher checkpoint structure: {checkpoint}")
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def run_teacher(
    teacher: nn.Module, frames: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    if int(chunk_size) != 1:
        raise ValueError(
            "EVSSM's grid operator requires teacher micro-batch 1; "
            "set --teacher-chunk 1"
        )
    batch, timesteps, channels, height, width = frames.shape
    flat = frames.reshape(batch * timesteps, channels, height, width)
    h_pad = (4 - height % 4) % 4
    w_pad = (4 - width % 4) % 4
    outputs = []
    for start in range(flat.shape[0]):
        chunk = flat[start : start + 1]
        if h_pad or w_pad:
            chunk = F.pad(chunk, (0, w_pad, 0, h_pad), mode="reflect")
        output = teacher(chunk)
        if isinstance(output, (tuple, list)):
            output = output[0]
        outputs.append(output[:, :, :height, :width].clamp(0.0, 1.0))
    return torch.cat(outputs).reshape(batch, timesteps, 3, height, width)


def teacher_for_batch(
    batch: Dict[str, object],
    blurry: torch.Tensor,
    teacher_model: Optional[nn.Module],
    teacher_chunk: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if teacher_model is not None:
        teacher = run_teacher(teacher_model, blurry, teacher_chunk)
        available = torch.ones(blurry.shape[0], dtype=torch.bool, device=device)
        return teacher, available

    available = batch["has_teacher"].to(device=device, dtype=torch.bool)
    if bool(available.any().item()):
        cached = batch["teacher"].to(device=device, dtype=torch.float32, non_blocking=True)
        # Missing cached teachers are replaced by the raw frames.  The mask
        # excludes them from distillation while keeping batched teacher input valid.
        mask = available.view(-1, 1, 1, 1, 1)
        return torch.where(mask, cached, blurry), available
    return None, available


def build_training_contract(
    model_config: Mapping[str, object],
    *,
    motion_alignment_v4: bool,
    transition_summary: Optional[Mapping[str, int]] = None,
) -> Dict[str, object]:
    history = int(model_config["max_history"])
    if not motion_alignment_v4:
        return {
            "schema": TRAINING_CONTRACT_SCHEMA_V3,
            "supervised_output": "latest_frame",
            "temporal_output": "rolling_two_window_forward",
            "training_clip_length": history + 1,
            "rolling_window_length": history,
            "stream_prefix_padding": "repeat_first_frame_on_left",
            "causality": "strict_upper_triangular_temporal_attention_mask",
            "fft_normalization": "ortho",
        }
    if transition_summary is None:
        raise ValueError("v4 training contract requires a transition summary")
    return {
        "schema": TRAINING_CONTRACT_SCHEMA_V4,
        "supervised_output": "latest_frame_in_joint_phase",
        "temporal_output": "rolling_two_window_motion_diagnostics",
        "training_clip_length": history + 1,
        "rolling_window_length": history,
        "stream_prefix_padding": "repeat_first_frame_on_left",
        "frame_index_field": "frame_indices",
        "transition_mask_field": "transition_valid",
        "transition_valid_definition": "right_frame_index_equals_left_plus_one",
        "padding_transition_policy": "never_supervised",
        "train_clips": int(transition_summary["train_clips"]),
        "train_sequences": int(transition_summary["train_sequences"]),
        "alignment_sampler_clips": int(
            transition_summary["alignment_sampler_clips"]
        ),
        "alignment_sampler_policy": "clips_with_at_least_one_real_transition",
        "transition_weighting": "real_clip_slot_frequency_not_unique_edge_uniform",
        "dropped_tail_policy": (
            "shuffle_then_drop_incomplete_microbatch_each_epoch"
        ),
        "terminal_checkpoint_policy": (
            "unconditional_atomic_save_at_exact_optimizer_step_600_before_exit"
        ),
        "resume_rng_policy": (
            "epoch_boundary_python_numpy_torch_cpu_and_loader_generators"
        ),
        "real_transition_slots": int(transition_summary["real_transition_slots"]),
        "unique_real_transitions": int(
            transition_summary["unique_real_transitions"]
        ),
        "diagnostic_method": "forward_sequence_with_motion_diagnostics",
        "diagnostic_tuple": [
            "prediction_sequence",
            "adjacent_flow_current_to_previous",
            "adjacent_confidence",
            "adjacent_valid",
        ],
        "alignment_disabled_method": "forward_sequence_alignment_disabled",
        "flow_units": "quarter_resolution_pixels",
        "causality": (
            "strict_pairwise_previous_current_alignment_plus_"
            "strict_upper_triangular_temporal_attention"
        ),
        "fft_normalization": "ortho",
        "phases": [
            {
                "name": "alignment_only",
                "optimizer_steps": V4_ALIGNMENT_ONLY_STEPS,
                "base_frozen": True,
                "trainable_parameters": [
                    "motion_aligner.match_projection.weight"
                ],
                "loss": "real_transition_motion_alignment_auxiliary_only",
            },
            {
                "name": "joint",
                "optimizer_steps": V4_TOTAL_STEPS - V4_ALIGNMENT_ONLY_STEPS,
                "base_frozen": False,
                "trainable_parameters": [
                    "base_parameters",
                    "motion_aligner.match_projection.weight",
                    "motion_alignment_gate",
                ],
                "loss": "bounded_v3_losses_plus_motion_compensated_delta_and_alignment",
            },
        ],
    }


def model_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    step: int,
    best_psnr: float,
    best_ssim_at_best_psnr: float,
    input_domain: str,
    teacher_provenance: Dict[str, object],
    objective_contract: Dict[str, object],
    optimization_contract: Dict[str, object],
    validation_metrics: Dict[str, float],
    motion_alignment_v4: bool = False,
    warm_start_provenance: Optional[Dict[str, object]] = None,
    training_contract: Optional[Dict[str, object]] = None,
    registered_contract: Optional[Dict[str, str]] = None,
    data_identity: Optional[Dict[str, str]] = None,
    rng_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    bare = model.module if hasattr(model, "module") else model
    model_config = bare.config_dict()
    model_config["input_domain"] = str(input_domain)
    payload = {
        "format": CHECKPOINT_FORMAT_V4 if motion_alignment_v4 else CHECKPOINT_FORMAT_V3,
        "model": bare.state_dict(),
        "model_config": model_config,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_psnr": best_psnr,
        "best_ssim_at_best_psnr": best_ssim_at_best_psnr,
        "validation_metrics": dict(validation_metrics),
        "teacher_provenance": dict(teacher_provenance),
        "objective_contract": dict(objective_contract),
        "optimization_contract": dict(optimization_contract),
        "refinement_contract": build_refinement_contract(model_config),
        "checkpoint_selection": {
            "metric": "val_psnr",
            "mode": "max",
            "ssim_role": "diagnostic_recorded_at_selected_psnr",
            "deployment_status": "not_deployment_selected",
            "required_deployment_selector": (
                "motion_aligned_evssm_multimetric_gate.v1"
                if motion_alignment_v4
                else "evssm_relative_multimetric_gate.v1"
            ),
        },
        "training_contract": (
            training_contract
            if motion_alignment_v4
            else build_training_contract(
                model_config, motion_alignment_v4=False
            )
        ),
    }
    if motion_alignment_v4:
        if int(step) != v4_epoch_end_step(int(epoch)):
            raise ValueError("v4 checkpoint is not at a legal epoch/step boundary")
        if (
            not isinstance(warm_start_provenance, dict)
            or warm_start_provenance.get("schema") != WARM_START_SCHEMA_V4
        ):
            raise ValueError("v4 checkpoint requires warm_start_provenance")
        if (
            not isinstance(training_contract, dict)
            or training_contract.get("schema") != TRAINING_CONTRACT_SCHEMA_V4
        ):
            raise ValueError("v4 checkpoint requires a v4 training_contract")
        payload["warm_start_provenance"] = dict(warm_start_provenance)
        payload["training_phase"] = v4_phase_for_step(int(step))
        if registered_contract != {
            "schema": V4_REGISTERED_CONTRACT_SCHEMA,
            "path": str(V4_REGISTERED_CONTRACT_PATH.resolve()),
            "sha256": V4_REGISTERED_CONTRACT_SHA256,
        }:
            raise ValueError("v4 checkpoint requires the registered contract binding")
        payload["registered_contract"] = dict(registered_contract)
        expected_data_identity = {
            "train_manifest_sha256": V4_EXPECTED_TRAIN_MANIFEST_SHA256,
            "train_precompute_report_sha256": V4_EXPECTED_TRAIN_PRECOMPUTE_SHA256,
            "train_teacher_manifest_sha256": (
                V4_EXPECTED_TRAIN_TEACHER_MANIFEST_SHA256
            ),
            "val_manifest_sha256": V4_EXPECTED_VAL_MANIFEST_SHA256,
            "val_precompute_report_sha256": V4_EXPECTED_VAL_PRECOMPUTE_SHA256,
            "val_teacher_manifest_sha256": V4_EXPECTED_VAL_TEACHER_MANIFEST_SHA256,
            "evssm_checkpoint_sha256": V4_EXPECTED_EVSSM_SHA256,
        }
        if data_identity != expected_data_identity:
            raise ValueError("v4 checkpoint requires exact data identity binding")
        payload["data_identity"] = dict(data_identity)
        try:
            serialized_rng_state = validate_v4_rng_state(rng_state)
        except ValueError as error:
            raise ValueError(
                "v4 checkpoint requires an epoch-boundary RNG state"
            ) from error
        # Even when continuing from a trusted legacy checkpoint, every newly
        # written checkpoint uses the tensor-only NumPy representation.
        serialized_rng_state["numpy_random_state"] = (
            numpy_rng_state_as_safe_tensor(
                serialized_rng_state["numpy_random_state"]
            )
        )
        serialized_rng_state["numpy_random_state_encoding"] = (
            NUMPY_RNG_ENCODING_V4
        )
        payload["rng_state"] = serialized_rng_state
    return payload


def validate_v4_warm_start_provenance(
    provenance: object,
    *,
    expected_base_config: Mapping[str, object],
) -> Dict[str, object]:
    """Validate the immutable v3 origin carried through every v4 resume."""

    if not isinstance(provenance, dict) or provenance.get("schema") != (
        WARM_START_SCHEMA_V4
    ):
        raise ValueError("v4 checkpoint has invalid warm_start_provenance")
    if provenance.get("source_sha256") != V4_WARM_START_SHA256:
        raise ValueError("v4 warm start source SHA-256 is not preregistered")
    if provenance.get("source_format") != CHECKPOINT_FORMAT_V3:
        raise ValueError("v4 warm start source format is not v3")
    if provenance.get("source_model_config") != dict(expected_base_config):
        raise ValueError("v4 warm start source model_config mismatch")
    if provenance.get("optimizer_state_loaded") is not False:
        raise ValueError("v4 warm start must not load v3 optimizer state")
    expected_missing = {
        "motion_alignment_gate",
        "motion_aligner.match_projection.weight",
        "motion_aligner.offsets",
    }
    allowed_missing = provenance.get("allowed_missing_alignment_keys")
    if not isinstance(allowed_missing, list) or set(allowed_missing) != expected_missing:
        raise ValueError("v4 warm start missing-key provenance mismatch")
    digest = str(provenance.get("source_state_key_digest_sha256", "")).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("v4 warm start key digest is not SHA-256")
    copied = provenance.get("copied_key_count")
    if type(copied) is not int or copied < 1:
        raise ValueError("v4 warm start copied_key_count must be positive")
    probe = provenance.get("identity_probe")
    if not isinstance(probe, dict) or probe.get("passed") is not True:
        raise ValueError("v4 warm start identity probe did not pass")
    try:
        atol = float(probe["atol"])
        rtol = float(probe["rtol"])
        difference = float(probe["max_abs_difference"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("v4 warm start identity probe is malformed") from error
    if (
        not math.isfinite(atol)
        or not math.isfinite(rtol)
        or not math.isfinite(difference)
        or atol != 1.0e-6
        or rtol != 0.0
        or difference < 0.0
        or difference > atol
    ):
        raise ValueError("v4 warm start identity probe tolerance mismatch")
    return dict(provenance)


def load_v4_resume(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_config: Mapping[str, object],
    objective_contract: Mapping[str, object],
    optimization_contract: Mapping[str, object],
    training_contract: Mapping[str, object],
    teacher_provenance: Mapping[str, object],
    registered_contract: Mapping[str, str],
    data_identity: Mapping[str, str],
    train_loader_generator: torch.Generator,
    alignment_loader_generator: torch.Generator,
    base_lr: float = V4_BASE_LR,
    alignment_lr: float = V4_ALIGNMENT_LR,
) -> Tuple[int, int, float, float, Dict[str, object]]:
    """Strict v4->v4 resume, including optimizer and phase continuity."""

    checkpoint, _, _ = _load_checkpoint_payload(
        checkpoint_path,
        role="v4 resume checkpoint",
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT_V4:
        raise ValueError("--resume with v4 requires an exact v4 checkpoint")
    exact_metadata = {
        "model_config": dict(model_config),
        "objective_contract": dict(objective_contract),
        "optimization_contract": dict(optimization_contract),
        "training_contract": dict(training_contract),
        "teacher_provenance": dict(teacher_provenance),
        "refinement_contract": build_refinement_contract(dict(model_config)),
        "registered_contract": dict(registered_contract),
        "data_identity": dict(data_identity),
    }
    for key, expected in exact_metadata.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"v4 resume {key} mismatch")
    warm_start = validate_v4_warm_start_provenance(
        checkpoint.get("warm_start_provenance"),
        expected_base_config=v3_base_config_from_v4(model_config),
    )
    step = checkpoint.get("step")
    if type(step) is not int or step < 0 or step > V4_TOTAL_STEPS:
        raise ValueError("v4 resume step is outside [0,600]")
    phase = v4_phase_for_step(step)
    if checkpoint.get("training_phase") != phase:
        raise ValueError("v4 resume training_phase disagrees with checkpoint step")
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int:
        raise ValueError("v4 resume epoch is invalid")
    if step != v4_epoch_end_step(epoch):
        raise ValueError("v4 resume epoch/step is not a legal phase boundary")
    validate_v4_rng_state(checkpoint.get("rng_state"))

    bare = model.module if hasattr(model, "module") else model
    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("v4 resume is missing model state")
    state_key_digest(source_state)
    target_state = bare.state_dict()
    if set(source_state) != set(target_state):
        raise ValueError("v4 resume model state keys mismatch")
    for key, target in target_state.items():
        source = source_state[key]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(f"v4 resume tensor contract mismatch for {key}")

    optimizer_state = checkpoint.get("optimizer")
    scheduler_state = checkpoint.get("scheduler")
    if not isinstance(optimizer_state, dict) or not isinstance(scheduler_state, dict):
        raise ValueError("v4 resume is missing optimizer/scheduler state")
    bare.load_state_dict(source_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    groups = {
        str(group.get("group_name", "")): group for group in optimizer.param_groups
    }
    if set(groups) != {"base", "alignment"}:
        raise ValueError("v4 resume optimizer groups mismatch")
    expected_base_lr = 0.0 if phase == "alignment_only" else float(base_lr)
    if float(groups["base"]["lr"]) != expected_base_lr or float(
        groups["alignment"]["lr"]
    ) != float(alignment_lr):
        raise ValueError("v4 resume optimizer learning rates disagree with phase")
    configure_v4_phase(
        model,
        optimizer,
        phase,
        base_lr=base_lr,
        alignment_lr=alignment_lr,
    )
    best_psnr = float(checkpoint.get("best_psnr", float("-inf")))
    best_ssim = float(checkpoint.get("best_ssim_at_best_psnr", float("nan")))
    restore_v4_rng_state(
        checkpoint.get("rng_state"),
        train_loader_generator=train_loader_generator,
        alignment_loader_generator=alignment_loader_generator,
    )
    return epoch + 1, step, best_psnr, best_ssim, warm_start


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    teacher_model: Optional[nn.Module],
    teacher_input: bool,
    teacher_chunk: int,
    input_domain: str,
) -> Dict[str, float]:
    model.eval()
    squared_error = 0.0
    pixel_count = 0
    ssim_sum = 0.0
    image_count = 0
    for batch in loader:
        blurry = batch["blurry"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["sharp"][:, -1].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        teacher = None
        teacher_available = torch.zeros(
            blurry.shape[0], dtype=torch.bool, device=device
        )
        if teacher_input or input_domain == "evssm":
            teacher, teacher_available = teacher_for_batch(
                batch, blurry, teacher_model, teacher_chunk, device
            )
            if teacher is None or (
                input_domain == "evssm" and not bool(teacher_available.all().item())
            ):
                raise RuntimeError(
                    "EVSSM-domain validation requires an EVSSM output for every frame"
                )
        model_input = teacher if input_domain == "evssm" else blurry
        prediction = model(
            model_input, teacher if teacher_input else None
        ).clamp(0.0, 1.0)
        squared_error += float(F.mse_loss(prediction, target, reduction="sum").item())
        pixel_count += target.numel()
        ssim_sum += float(local_ssim(prediction, target).item()) * int(
            target.shape[0]
        )
        image_count += int(target.shape[0])
    mse = squared_error / max(1, pixel_count)
    model.train()
    return {
        "psnr": -10.0 * math.log10(max(mse, 1.0e-12)),
        "ssim": ssim_sum / max(1, image_count),
    }


def save_training_checkpoint(
    payload: Mapping[str, object],
    path: Path,
    *,
    atomic: bool,
) -> None:
    """Use same-directory replace for formal v4; retain legacy v3 save exactly."""

    if not atomic:
        torch.save(dict(payload), path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    args = parse_args()
    if args.train_manifest is None and args.train_precompute_report is None:
        raise ValueError("--train-manifest or --precompute-report is required")
    if args.teacher_chunk != 1:
        raise ValueError(
            "--teacher-chunk must be 1 because EVSSM requires batch=1"
        )
    if args.grad_accumulation < 1:
        raise ValueError("--grad-accumulation must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    validate_teacher_options(
        args.teacher_checkpoint,
        args.teacher_input,
        args.distill_weight,
        args.input_domain,
        args.evssm_fidelity_weight,
    )
    validate_v3_hyperparameters(
        max_residual=args.max_residual,
        fft_weight=args.fft_weight,
        distill_weight=args.distill_weight,
        evssm_fidelity_weight=args.evssm_fidelity_weight,
        temporal_delta_weight=args.temporal_delta_weight,
        edge_weight=args.edge_weight,
        laplacian_gate_weight=args.laplacian_gate_weight,
    )
    validate_v4_options(args)
    set_seed(args.seed, seed_cuda=not bool(args.motion_alignment_v4))
    device = choose_device(args.device)
    if args.motion_alignment_v4 and device.type != "cpu":
        raise ValueError(
            "formal motion-aligned v4 is preregistered for --device cpu"
        )
    registered_v4_contract: Optional[Dict[str, str]] = (
        validate_v4_registered_contract() if args.motion_alignment_v4 else None
    )

    root_value = str(args.data_root) if args.data_root else None
    train_set = VideoDeblurJsonlDataset(
        str(args.train_manifest) if args.train_manifest is not None else None,
        # H+1 frames form the two adjacent, full H-frame windows that the
        # streaming runtime would evaluate at t-1 and t.
        clip_length=args.history + 1,
        stride=args.clip_stride,
        crop_size=args.crop_size,
        augment=True,
        root=root_value,
        precompute_report=(
            str(args.train_precompute_report)
            if args.train_precompute_report is not None
            else None
        ),
    )
    transition_summary = dataset_transition_summary(train_set)
    alignment_clip_indices: List[int] = []
    if args.motion_alignment_v4:
        validate_v4_training_inventory(transition_summary)
        if int(transition_summary["unique_real_transitions"]) < 1:
            raise ValueError(
                "motion-aligned v4 training requires at least one real adjacent "
                "transition after prefix-padding exclusion"
            )
        alignment_clip_indices = real_transition_clip_indices(train_set)
        if not alignment_clip_indices:
            raise RuntimeError(
                "v4 transition accounting is inconsistent: positive transition "
                "count but no trainable clips"
            )
    val_set = (
        VideoDeblurJsonlDataset(
            str(args.val_manifest) if args.val_manifest is not None else None,
            clip_length=args.history,
            stride=args.clip_stride,
            crop_size=args.crop_size,
            augment=False,
            root=root_value,
            precompute_report=(
                str(args.val_precompute_report)
                if args.val_precompute_report is not None
                else None
            ),
        )
        if args.val_manifest is not None or args.val_precompute_report is not None
        else None
    )
    teacher_provenance = resolve_teacher_provenance(
        dataset=train_set,
        teacher_checkpoint=args.teacher_checkpoint,
        distill_weight=args.distill_weight,
        input_domain=args.input_domain,
        evssm_fidelity_weight=args.evssm_fidelity_weight,
    )
    validate_validation_teacher_provenance(
        teacher_provenance,
        val_set,
        teacher_checkpoint=args.teacher_checkpoint,
        input_domain=args.input_domain,
    )
    v4_data_identity = (
        validate_v4_data_identity(
            args,
            train_set=train_set,
            val_set=val_set,
            teacher_provenance=teacher_provenance,
        )
        if args.motion_alignment_v4
        else None
    )
    train_loader_generator: Optional[torch.Generator] = None
    alignment_loader_generator: Optional[torch.Generator] = None
    if args.motion_alignment_v4:
        train_loader_generator = torch.Generator(device="cpu")
        train_loader_generator.manual_seed(int(args.seed) + 1000)
        alignment_loader_generator = torch.Generator(device="cpu")
        alignment_loader_generator.manual_seed(int(args.seed) + 2000)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=bool(args.motion_alignment_v4),
        generator=train_loader_generator,
    )
    alignment_loader = (
        DataLoader(
            Subset(train_set, alignment_clip_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            generator=alignment_loader_generator,
        )
        if args.motion_alignment_v4
        else None
    )
    if args.motion_alignment_v4:
        if len(train_loader) != 58 or alignment_loader is None or len(
            alignment_loader
        ) != 26:
            raise ValueError(
                "v4 loader inventory must be joint=58 and alignment=26 "
                "micro-batches per epoch"
            )
        if train_loader_generator is None or alignment_loader_generator is None:
            raise RuntimeError("v4 loader generators were not initialized")
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        if val_set is not None
        else None
    )
    if not args.motion_alignment_v4:
        args.output.mkdir(parents=True, exist_ok=True)

    config = {
        "channels": args.channels,
        "num_heads": args.heads,
        "num_blocks": args.blocks,
        "max_history": args.history,
        "use_teacher_input": args.teacher_input,
        "input_domain": args.input_domain,
        "max_residual": args.max_residual,
    }
    if args.motion_alignment_v4:
        config["motion_alignment"] = dict(V4_MOTION_ALIGNMENT_CONFIG)
    model = build_causal_video_deblur(config)
    warm_start_provenance = (
        load_v4_warm_start(
            model,
            args.warm_start_v3,
            expected_sha256=V4_WARM_START_SHA256,
        )
        if args.motion_alignment_v4 and args.warm_start_v3 is not None
        else None
    )
    if warm_start_provenance is not None and warm_start_provenance.get(
        "source_sha256"
    ) != V4_WARM_START_SHA256:
        raise ValueError(
            "--warm-start-v3 is not the preregistered v3 H3 epoch-20 checkpoint"
        )
    model = model.to(device)
    config = model.config_dict()
    objective_contract = build_objective_contract(args)
    teacher_model = (
        load_single_frame_teacher(args.teacher_checkpoint, device)
        if args.teacher_checkpoint
        else None
    )

    optimizer = (
        build_v4_optimizer(
            model,
            base_lr=args.v4_base_lr,
            alignment_lr=args.v4_alignment_lr,
            weight_decay=args.weight_decay,
        )
        if args.motion_alignment_v4
        else torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.9),
        )
    )
    optimizer_steps_per_epoch = max(
        1, math.ceil(len(train_loader) / int(args.grad_accumulation))
    )
    max_steps = (
        V4_TOTAL_STEPS
        if args.motion_alignment_v4
        else (
            int(args.max_steps)
            if int(args.max_steps) > 0
            else int(args.epochs * optimizer_steps_per_epoch)
        )
    )
    if args.warmup_steps >= max_steps and args.warmup_steps > 0:
        raise ValueError("--warmup-steps must be smaller than the optimizer-step budget")
    minimum_lr = 1.0e-7
    minimum_factor = min(1.0, minimum_lr / float(args.lr))

    def lr_multiplier(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        cosine_steps = max(1, max_steps - int(args.warmup_steps))
        progress = float(step - int(args.warmup_steps)) / float(cosine_steps)
        progress = min(1.0, max(0.0, progress))
        return minimum_factor + 0.5 * (1.0 - minimum_factor) * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lambda _: 1.0, lambda _: 1.0]
        )
        if args.motion_alignment_v4
        else torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_multiplier
        )
    )
    if args.motion_alignment_v4:
        # The v4 scheduler exists only to preserve the checkpoint container ABI;
        # its fixed lambdas are never stepped. Reapply the explicit phase after
        # LambdaLR initialization so no implicit LR policy can leak in.
        configure_v4_phase(
            model,
            optimizer,
            "alignment_only",
            base_lr=args.v4_base_lr,
            alignment_lr=args.v4_alignment_lr,
        )
        assert alignment_loader is not None
        alignment_steps_per_epoch = max(
            1,
            math.ceil(len(alignment_loader) / int(args.grad_accumulation)),
        )
        minimum_v4_epochs = math.ceil(
            V4_ALIGNMENT_ONLY_STEPS / alignment_steps_per_epoch
        ) + math.ceil(
            (V4_TOTAL_STEPS - V4_ALIGNMENT_ONLY_STEPS)
            / optimizer_steps_per_epoch
        )
        if not args.dry_run and int(args.epochs) < int(minimum_v4_epochs):
            raise ValueError(
                "--epochs cannot reach the registered 600 v4 optimizer steps; "
                f"need at least {minimum_v4_epochs}, got {args.epochs}"
            )
    optimization_contract = build_optimization_contract(
        args,
        max_steps=max_steps,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        execution_device=(str(device) if args.motion_alignment_v4 else None),
        amp_effective=(
            bool(args.amp and device.type == "cuda")
            if args.motion_alignment_v4
            else None
        ),
    )
    training_contract = build_training_contract(
        config,
        motion_alignment_v4=bool(args.motion_alignment_v4),
        transition_summary=(transition_summary if args.motion_alignment_v4 else None),
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch, global_step, best_psnr = 0, 0, float("-inf")
    best_ssim_at_best_psnr = float("nan")

    if args.resume and args.motion_alignment_v4:
        assert registered_v4_contract is not None
        assert train_loader_generator is not None
        assert alignment_loader_generator is not None
        (
            start_epoch,
            global_step,
            best_psnr,
            best_ssim_at_best_psnr,
            warm_start_provenance,
        ) = load_v4_resume(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=config,
            objective_contract=objective_contract,
            optimization_contract=optimization_contract,
            training_contract=training_contract,
            teacher_provenance=teacher_provenance,
            registered_contract=registered_v4_contract,
            data_identity=v4_data_identity,
            train_loader_generator=train_loader_generator,
            alignment_loader_generator=alignment_loader_generator,
            base_lr=args.v4_base_lr,
            alignment_lr=args.v4_alignment_lr,
        )
    elif args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        saved_config = dict(checkpoint.get("model_config", {}))
        # v1 model configs omitted max_residual and therefore used the legacy
        # unbounded output head.  Normalize only for an explicit legacy resume;
        # the safe v3 default cannot silently reinterpret those weights.
        saved_config.setdefault("max_residual", 0.0)
        if dict(saved_config) != config:
            raise ValueError(f"resume model_config mismatch: {saved_config} != {config}")
        saved_objective = checkpoint.get("objective_contract")
        if saved_objective is not None and saved_objective != objective_contract:
            raise ValueError("resume objective_contract mismatch")
        saved_optimization = checkpoint.get("optimization_contract")
        if (
            saved_optimization is not None
            and saved_optimization != optimization_contract
        ):
            raise ValueError("resume optimization_contract mismatch")
        if checkpoint.get("teacher_provenance") != teacher_provenance:
            raise ValueError("resume teacher_provenance mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("step", 0))
        best_psnr = float(checkpoint.get("best_psnr", float("-inf")))
        best_ssim_at_best_psnr = float(
            checkpoint.get("best_ssim_at_best_psnr", float("nan"))
        )

    if args.motion_alignment_v4:
        warm_start_provenance = validate_v4_warm_start_provenance(
            warm_start_provenance,
            expected_base_config=v3_base_config_from_v4(config),
        )
        if registered_v4_contract is None:
            raise RuntimeError("v4 registered contract was not validated")
        # Formal v4 creates no output path until every data, warm-start/resume,
        # model, objective, optimization, and provenance contract has passed.
        args.output.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    if not args.motion_alignment_v4:
        # Keep the serialized legacy invocation byte-for-byte in field content:
        # v4-only parser defaults are intentionally absent on the v3 path.
        for field in V4_CLI_ONLY_FIELDS:
            run_config.pop(field, None)
    run_config["device"] = str(device)
    run_config["resolved_train_manifest"] = str(train_set.manifest)
    run_config["training_clip_length"] = args.history + 1
    run_config["rolling_window_length"] = args.history
    run_config["teacher_provenance"] = teacher_provenance
    run_config["model_config"] = config
    run_config["objective_contract"] = objective_contract
    run_config["optimization_contract"] = optimization_contract
    run_config["refinement_contract"] = build_refinement_contract(config)
    if args.motion_alignment_v4:
        run_config["training_contract"] = training_contract
        run_config["transition_summary"] = dict(transition_summary)
        run_config["alignment_train_clips"] = len(alignment_clip_indices)
        run_config["warm_start_provenance"] = warm_start_provenance
        run_config["registered_contract"] = registered_v4_contract
        run_config["data_identity"] = v4_data_identity
        run_config["amp_requested"] = bool(args.amp)
        run_config["amp_effective"] = bool(amp_enabled)
    run_config["checkpoint_selection"] = {
        "metric": "val_psnr",
        "mode": "max",
        "ssim_role": "diagnostic_recorded_at_selected_psnr",
        "deployment_status": "not_deployment_selected",
        "required_deployment_selector": (
            "motion_aligned_evssm_multimetric_gate.v1"
            if args.motion_alignment_v4
            else "evssm_relative_multimetric_gate.v1"
        ),
    }
    for key, value in list(run_config.items()):
        if isinstance(value, Path):
            run_config[key] = str(value)
    with (args.output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
        handle.write("\n")

    startup_record: Dict[str, object] = {
        "device": str(device),
        "train_clips": len(train_set),
        "val_clips": len(val_set) if val_set is not None else 0,
        "model_config": config,
        "objective_contract": objective_contract,
        "optimization_contract": optimization_contract,
        "refinement_contract": build_refinement_contract(config),
        "teacher": (
            str(args.teacher_checkpoint) if args.teacher_checkpoint else "cached/none"
        ),
        "teacher_provenance": teacher_provenance,
    }
    if args.motion_alignment_v4:
        startup_record.update(
            {
                "training_contract": training_contract,
                "transition_summary": dict(transition_summary),
                "alignment_train_clips": len(alignment_clip_indices),
                "warm_start_provenance": warm_start_provenance,
                "registered_contract": registered_v4_contract,
                "data_identity": v4_data_identity,
                "amp_requested": bool(args.amp),
                "amp_effective": bool(amp_enabled),
            }
        )
    print(json.dumps(startup_record), flush=True)

    for epoch in range(start_epoch, args.epochs):
        if global_step >= max_steps:
            break
        phase = (
            v4_phase_for_step(global_step)
            if args.motion_alignment_v4
            else "v3_joint"
        )
        if args.motion_alignment_v4:
            configure_v4_phase(
                model,
                optimizer,
                phase,
                base_lr=args.v4_base_lr,
                alignment_lr=args.v4_alignment_lr,
            )
            if phase == "alignment_only":
                assert alignment_loader is not None
                active_loader = alignment_loader
            else:
                active_loader = train_loader
        else:
            active_loader = train_loader
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        batch_index = -1
        for batch_index, batch in enumerate(active_loader):
            blurry_sequence = batch["blurry"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            sharp_sequence = batch["sharp"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            transition_valid = (
                batch["transition_valid"].to(
                    device=device, dtype=torch.bool, non_blocking=True
                )
                if args.motion_alignment_v4
                else None
            )
            target = sharp_sequence[:, -1]
            teacher = None
            teacher_available = torch.zeros(
                blurry_sequence.shape[0], dtype=torch.bool, device=device
            )
            if (
                args.teacher_input
                or args.distill_weight > 0.0
                or args.evssm_fidelity_weight > 0.0
                or args.input_domain == "evssm"
            ):
                teacher, teacher_available = teacher_for_batch(
                    batch,
                    blurry_sequence,
                    teacher_model,
                    args.teacher_chunk,
                    device,
                )
            if args.teacher_input and teacher is None:
                raise RuntimeError(
                    "--teacher-input requires --teacher-checkpoint or teacher paths in the manifest"
                )
            if args.distill_weight > 0.0 and teacher is None:
                raise RuntimeError(
                    "positive --distill-weight requires --teacher-checkpoint "
                    "or teacher paths in the manifest"
                )
            if args.evssm_fidelity_weight > 0.0 and (
                teacher is None or not bool(teacher_available.all().item())
            ):
                raise RuntimeError(
                    "positive --evssm-fidelity-weight requires a verified EVSSM "
                    "output for every sample (use --teacher-checkpoint or a "
                    "complete precompute report)"
                )
            if args.input_domain == "evssm" and (
                teacher is None or not bool(teacher_available.all().item())
            ):
                raise RuntimeError(
                    "--input-domain=evssm requires an EVSSM teacher output for "
                    "every frame (use --teacher-checkpoint or complete cached "
                    "teacher paths)"
                )

            autocast_context = (
                torch.cuda.amp.autocast(enabled=True) if amp_enabled else nullcontext()
            )
            with autocast_context:
                model_input_sequence = (
                    teacher if args.input_domain == "evssm" else blurry_sequence
                )
                model_input = model_input_sequence[:, 1:]
                if args.motion_alignment_v4:
                    assert transition_valid is not None
                    v4_losses = compute_v4_batch_losses(
                        model,
                        model_input_sequence,
                        sharp_sequence,
                        transition_valid,
                        teacher,
                        teacher_available,
                        args,
                        phase,
                    )
                    loss = v4_losses["loss"]
                    prediction_sequence = v4_losses["prediction_sequence"]
                    prediction = v4_losses["prediction"]
                    l1 = v4_losses["l1"]
                    fft = v4_losses["fft"]
                    temporal_delta = v4_losses["temporal_delta"]
                    sharp_edge = v4_losses["sharp_edge"]
                    laplacian_gate_hinge = v4_losses["laplacian_gate_hinge"]
                    evssm_fidelity = v4_losses["evssm_fidelity"]
                    distill = v4_losses["distill"]
                else:
                    previous_input = model_input_sequence[:, :-1]
                    previous_teacher = (
                        teacher[:, :-1]
                        if args.teacher_input and teacher is not None
                        else None
                    )
                    current_teacher = (
                        teacher[:, 1:]
                        if args.teacher_input and teacher is not None
                        else None
                    )
                    prediction_sequence = model.forward_sequence(
                        model_input, current_teacher
                    )
                    prediction = prediction_sequence[:, -1]
                    previous_prediction = model(previous_input, previous_teacher)
                    l1 = F.l1_loss(prediction, target)
                    fft = fft_l1_loss(prediction, target)
                    temporal_delta = rolling_window_temporal_delta_l1_loss(
                        prediction, previous_prediction, sharp_sequence
                    )
                    sharp_edge = spatial_gradient_l1_loss(prediction, target)
                    laplacian_gate_hinge = prediction.new_zeros(())
                    if teacher is not None and bool(teacher_available.all().item()):
                        laplacian_gate_hinge = runtime_laplacian_logvar_hinge_loss(
                            prediction, teacher[:, -1]
                        )
                    evssm_fidelity = prediction.new_zeros(())
                    if (
                        args.evssm_fidelity_weight > 0.0
                        and teacher is not None
                    ):
                        evssm_fidelity = evssm_fidelity_l1_loss(
                            prediction_sequence, teacher[:, 1:], teacher_available
                        )
                    loss = (
                        l1
                        + args.fft_weight * fft
                        + args.temporal_delta_weight * temporal_delta
                        + args.edge_weight * sharp_edge
                        + args.laplacian_gate_weight * laplacian_gate_hinge
                        + args.evssm_fidelity_weight * evssm_fidelity
                    )
                    distill = prediction.new_zeros(())
                    if (
                        args.distill_weight > 0.0
                        and teacher is not None
                        and bool(teacher_available.any().item())
                    ):
                        distill = F.l1_loss(
                            prediction[teacher_available], teacher[teacher_available, -1]
                        )
                        loss = loss + args.distill_weight * distill

            group_start = (
                batch_index // int(args.grad_accumulation)
            ) * int(args.grad_accumulation)
            group_size = min(
                int(args.grad_accumulation), len(active_loader) - group_start
            )
            group_position = batch_index - group_start + 1
            scaler.scale(loss / float(group_size)).backward()
            running_loss += float(loss.detach().item())
            if group_position < group_size:
                continue

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if not args.motion_alignment_v4:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % args.log_every == 0 or global_step == 1:
                log_record: Dict[str, object] = {
                    "epoch": epoch,
                    "step": global_step,
                    "micro_batches_in_step": group_size,
                    "loss": float(loss.detach().item()),
                    "l1": float(l1.detach().item()),
                    "fft": float(fft.detach().item()),
                    "temporal_delta": float(temporal_delta.detach().item()),
                    "sharp_edge": float(sharp_edge.detach().item()),
                    "laplacian_gate_hinge": float(
                        laplacian_gate_hinge.detach().item()
                    ),
                    "evssm_fidelity": float(evssm_fidelity.detach().item()),
                    "distill": float(distill.detach().item()),
                    "max_abs_residual": float(
                        (prediction_sequence - model_input)
                        .detach()
                        .abs()
                        .max()
                        .item()
                    ),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if args.motion_alignment_v4:
                    groups = {
                        str(group["group_name"]): group
                        for group in optimizer.param_groups
                    }
                    log_record.update(
                        {
                            "phase": phase,
                            "trainable_parameters": (
                                ["motion_aligner.match_projection.weight"]
                                if phase == "alignment_only"
                                else [
                                    "base_parameters",
                                    "motion_aligner.match_projection.weight",
                                    "motion_alignment_gate",
                                ]
                            ),
                            "base_lr": groups["base"]["lr"],
                            "alignment_lr": groups["alignment"]["lr"],
                            "motion_photo": float(
                                v4_losses["motion_photo"].detach().item()
                            ),
                            "motion_gradient": float(
                                v4_losses["motion_gradient"].detach().item()
                            ),
                            "motion_smooth": float(
                                v4_losses["motion_smooth"].detach().item()
                            ),
                            "motion_alignment_objective": float(
                                v4_losses["motion_alignment_objective"]
                                .detach()
                                .item()
                            ),
                            "real_transition_count": int(
                                v4_losses["real_transition_count"]
                                .detach()
                                .item()
                            ),
                            "motion_confidence_mean": float(
                                v4_losses["motion_confidence_mean"]
                                .detach()
                                .item()
                            ),
                        }
                    )
                print(json.dumps(log_record), flush=True)
            if args.dry_run:
                break
            if global_step >= max_steps:
                break
            if (
                args.motion_alignment_v4
                and phase == "alignment_only"
                and global_step >= V4_ALIGNMENT_ONLY_STEPS
            ):
                # Start the joint phase with the full loader and a fresh
                # accumulation group, while retaining the same optimizer state.
                break

        if args.motion_alignment_v4:
            # Checkpoints describe the phase that will consume the next
            # optimizer step.  At the exact 100-step boundary this serializes
            # joint LRs/requires_grad without resetting AdamW moments.
            configure_v4_phase(
                model,
                optimizer,
                v4_phase_for_step(global_step),
                base_lr=args.v4_base_lr,
                alignment_lr=args.v4_alignment_lr,
            )
        val_metrics = (
            validate(
                model,
                val_loader,
                device,
                teacher_model,
                args.teacher_input,
                args.teacher_chunk,
                args.input_domain,
            )
            if val_loader is not None
            else {"psnr": float("nan"), "ssim": float("nan")}
        )
        val_psnr = float(val_metrics["psnr"])
        val_ssim = float(val_metrics["ssim"])
        v4_rng_state = None
        if args.motion_alignment_v4:
            assert train_loader_generator is not None
            assert alignment_loader_generator is not None
            v4_rng_state = capture_v4_rng_state(
                train_loader_generator,
                alignment_loader_generator,
            )
        if not math.isnan(val_psnr) and val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim_at_best_psnr = val_ssim
            best_payload = model_payload(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_psnr,
                best_ssim_at_best_psnr,
                args.input_domain,
                teacher_provenance,
                objective_contract,
                optimization_contract,
                val_metrics,
                motion_alignment_v4=bool(args.motion_alignment_v4),
                warm_start_provenance=warm_start_provenance,
                training_contract=training_contract,
                registered_contract=registered_v4_contract,
                data_identity=v4_data_identity,
                rng_state=v4_rng_state,
            )
            save_training_checkpoint(
                best_payload,
                args.output / "best.pth",
                atomic=bool(args.motion_alignment_v4),
            )
        terminal_v4 = bool(
            args.motion_alignment_v4 and global_step == V4_TOTAL_STEPS
        )
        if (epoch + 1) % args.save_every == 0 or args.dry_run or terminal_v4:
            payload = model_payload(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_psnr,
                best_ssim_at_best_psnr,
                args.input_domain,
                teacher_provenance,
                objective_contract,
                optimization_contract,
                val_metrics,
                motion_alignment_v4=bool(args.motion_alignment_v4),
                warm_start_provenance=warm_start_provenance,
                training_contract=training_contract,
                registered_contract=registered_v4_contract,
                data_identity=v4_data_identity,
                rng_state=v4_rng_state,
            )
            save_training_checkpoint(
                payload,
                args.output / "latest.pth",
                atomic=bool(args.motion_alignment_v4),
            )
            save_training_checkpoint(
                payload,
                args.output / f"epoch_{epoch + 1:04d}.pth",
                atomic=bool(args.motion_alignment_v4),
            )
            if terminal_v4:
                save_training_checkpoint(
                    payload,
                    args.output / "terminal_final.pth",
                    atomic=True,
                )
        epoch_record: Dict[str, object] = {
            "epoch": epoch,
            "mean_train_loss": running_loss / max(1, batch_index + 1),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "best_psnr": best_psnr,
            "best_ssim_at_best_psnr": best_ssim_at_best_psnr,
            "checkpoint_selection": "max_val_psnr",
        }
        if args.motion_alignment_v4:
            epoch_record.update(
                {
                    "phase_executed": phase,
                    "next_phase": v4_phase_for_step(global_step),
                    "amp_effective": bool(amp_enabled),
                }
            )
        print(json.dumps(epoch_record), flush=True)
        if args.dry_run:
            break
        if global_step >= max_steps:
            break


if __name__ == "__main__":
    main()
