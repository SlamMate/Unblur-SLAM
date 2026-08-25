#!/usr/bin/env python3
"""Resumable Unblur-SLAM-style adaptation of the pinned causal TURTLE model.

The trainer deliberately separates ordered video sources from independent
single-image defocus pairs.  Video clips keep the official K/V graph attached
for five causal frames.  Single-image examples reset K/V and freeze only the
five history-attention modules for that optimizer step, while updating the
entire remaining spatial network.

The JSON contract is the authority for data paths, SHA256 identities, source
weights, optimizer settings and output directory.  Test splits are rejected.
This file is a trainer, not a GPU launcher; resource isolation belongs in the
launch preflight.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_turtle_single_image_defocus import (  # noqa: E402
    SingleImagePair,
    load_single_image_manifest,
)
from scripts.train_turtle_mixed_defocus import (  # noqa: E402
    CountingAdamW,
    _transform_tensor_pair,
    execute_checked_optimizer_step,
)
from scripts.train_turtle_streaming import (  # noqa: E402
    DEFAULT_TURTLE_CHECKPOINT,
    DEFAULT_TURTLE_CONFIG,
    DEFAULT_TURTLE_REPO,
    HISTORY_ATTENTION_PARAMETER_COUNT,
    HISTORY_ATTENTION_PARAMETER_PREFIXES,
    HISTORY_ATTENTION_PARAMETER_TENSORS,
    PairedSequenceDataset,
    SequenceRecord,
    _choose_transform,
    _read_transformed,
    _validate_forward_result,
    fft_l1_loss,
    set_seed,
)
from src.turtle_backend import (  # noqa: E402
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    TURTLE_CACHE_CONTRACT,
    build_turtle_model_from_scratch,
    load_turtle_model,
    sha256_file,
)


SCHEMA = "unblur_slam.turtle_unblur_stable_training.v3"
TOTAL_STEPS = 300_000
CLIP_LENGTH = 5
CROP_SIZE = 128
MOTION_BASE_LR = 1e-3
REPLICA_LR = 1e-5
DEFOCUS_REHEARSAL_LR = 1e-7
BASE_LR = MOTION_BASE_LR
FFT_WEIGHT = 0.1
BATCH_SIZE_PER_GPU = 3
GLOBAL_BATCH_SIZE = 6
DDP_WORLD_SIZE = 2
SUPPORTED_DDP_WORLD_SIZES = (2, 3, 6)
DDP_BACKEND = "nccl"
AMP_MAX_SAME_BATCH_RETRIES = 8
AMP_INITIAL_SCALE = 1024.0
AMP_GROWTH_INTERVAL = 2000
PHOTOMETRIC_TRANSFORM = "exact_srgb_to_linear_before_model_and_loss"
CHECKPOINT_FORMAT = "unblur_slam.turtle_unblur_stable_checkpoint.v3"
DEPLOY_CHECKPOINT_FORMAT = "unblur_slam.turtle_unblur_stable_deploy.v2"
PAIRED_IMAGE_SCHEMA = "unblur_slam.paired_image_train.v1"
PAIRED_VIDEO_SCHEMA = "unblur_slam.paired_video_train.v1"


def _sha(value: Any, label: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{label} must be one SHA256")
    return normalized


def _json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _next_amp_retry_scale(previous_scale: float) -> float:
    """Return the shared retry scale without permitting an optimizer update."""
    if not math.isfinite(previous_scale) or previous_scale <= 0.0:
        raise ValueError("AMP scale must be finite and positive")
    return max(previous_scale / 2.0, 1.0)


def _read_contract(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise ValueError(f"training contract must use schema {SCHEMA}")
    if payload.get("test_pixels_permitted") is not False:
        raise ValueError("training contract must explicitly forbid test pixels")
    implementation = payload.get("implementation")
    if not isinstance(implementation, list) or not implementation:
        raise ValueError("content-addressed implementation pins are required")
    pinned_paths = set()
    for pin in implementation:
        if not isinstance(pin, Mapping):
            raise ValueError("each implementation pin must be an object")
        path = Path(str(pin.get("path", ""))).expanduser().resolve()
        if path in pinned_paths or not path.is_file():
            raise ValueError(f"implementation path missing or duplicated: {path}")
        pinned_paths.add(path)
        if sha256_file(path) != _sha(pin.get("sha256"), f"implementation {path}"):
            raise ValueError(f"implementation SHA mismatch: {path}")
    stage = str(payload.get("stage", ""))
    stage_protocol = {
        "motion_base": {
            "learning_rate": MOTION_BASE_LR,
            "scheduler_eta_min": 1e-7,
            "source_weights": {"reds": 0.55, "gopro_blur_gamma": 0.45},
        },
        "replica": {
            "learning_rate": REPLICA_LR,
            "scheduler_eta_min": 1e-7,
            "source_weights": {"replica_blurry_office3": 1.0},
        },
        "defocus_rehearsal": {
            "learning_rate": DEFOCUS_REHEARSAL_LR,
            "scheduler_eta_min": DEFOCUS_REHEARSAL_LR,
            "source_weights": {
                "unblur_hf_defocus": 0.80,
                "replica_blurry_office3": 0.20,
            },
        },
    }
    if stage not in stage_protocol:
        raise ValueError("training stage must be motion_base, replica, or defocus_rehearsal")
    protocol = stage_protocol[stage]
    fixed = {
        "total_steps": TOTAL_STEPS,
        "crop_size": CROP_SIZE,
        "clip_length": CLIP_LENGTH,
        "optimizer": "AdamW",
        "learning_rate": protocol["learning_rate"],
        "weight_decay": 1e-3,
        "betas": [0.9, 0.9],
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": TOTAL_STEPS,
        "scheduler_eta_min": protocol["scheduler_eta_min"],
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
        "global_batch_size": GLOBAL_BATCH_SIZE,
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
        "amp": True,
        "validation_during_training": False,
    }
    mismatches = {
        key: (payload.get(key), wanted)
        for key, wanted in fixed.items()
        if payload.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"fixed Unblur-style training fields changed: {mismatches}")
    world_size = int(payload.get("ddp_world_size", -1))
    local_batch_size = int(payload.get("batch_size_per_gpu", -1))
    if world_size not in SUPPORTED_DDP_WORLD_SIZES:
        raise ValueError(
            f"ddp_world_size must be one of {SUPPORTED_DDP_WORLD_SIZES} to preserve global batch 6"
        )
    if local_batch_size * world_size != GLOBAL_BATCH_SIZE:
        raise ValueError("local batch times DDP world size must equal global batch 6")
    expected_implementation = (
        f"ddp{world_size}_local_batch{local_batch_size}_global_batch6_"
        "five_frame_bptt_one_optimizer_step"
    )
    if payload.get("video_batch_implementation") != expected_implementation:
        raise ValueError("DDP video batch implementation identity changed")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("training contract has no sources")
    weight_sum = 0.0
    names = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("each source must be an object")
        name = str(source.get("name", ""))
        if not name or name in names:
            raise ValueError("source names must be non-empty and unique")
        names.add(name)
        if source.get("kind") not in {"video", "single"}:
            raise ValueError(f"source {name}: kind must be video or single")
        allowed_kinds = (
            {"video"}
            if stage in {"motion_base", "replica"}
            else {"video", "single"}
        )
        if source.get("kind") not in allowed_kinds:
            raise ValueError(f"source {name}: kind is invalid for stage {stage}")
        if source.get("kind") == "single" and source.get("manifest_kind") not in {
            "dpdd", "paired_image"
        }:
            raise ValueError(
                f"source {name}: single manifest_kind must be dpdd or paired_image"
            )
        split = str(source.get("split", "")).lower()
        if split not in {"train", "training"}:
            raise ValueError(f"source {name}: only train split is permitted")
        weight = float(source.get("weight", 0.0))
        if not (weight > 0.0):
            raise ValueError(f"source {name}: weight must be positive")
        weight_sum += weight
        radius = int(source.get("alignment_radius", 0))
        if radius < 0 or radius > 4:
            raise ValueError(f"source {name}: alignment_radius must be in [0,4]")
        if int(source.get("bit_depth", 8)) not in {8, 16}:
            raise ValueError(f"source {name}: bit_depth must be 8 or 16")
        manifest = Path(str(source.get("manifest", ""))).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"source {name}: manifest not found: {manifest}")
        if sha256_file(manifest) != _sha(source.get("manifest_sha256"), name):
            raise ValueError(f"source {name}: manifest SHA256 mismatch")
        expected_provenance = {
            "reds": ("snah/REDS", "62dc25d16e6f43d2214f1b365023abda86f7a0ae"),
            "gopro_blur_gamma": ("snah/GOPRO_Large", "592978466ae510d2734b199cad2fc79a346bda1c"),
            "replica_blurry_office3": (
                "qizhangslam/Unblur_slam_traning_dataset",
                "1f9d98158c3f27f6ec6de45ee2874c9caf2a2c59",
            ),
            "unblur_hf_defocus": (
                "qizhangslam/Unblur_slam_traning_dataset",
                "1f9d98158c3f27f6ec6de45ee2874c9caf2a2c59",
            ),
        }[name]
        provenance = source.get("provenance")
        if not isinstance(provenance, Mapping) or (
            provenance.get("repository"), provenance.get("revision")
        ) != expected_provenance:
            raise ValueError(f"source {name}: repository provenance changed")
        artifacts = provenance.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"source {name}: provenance artifacts are required")
        for artifact in artifacts:
            artifact_path = Path(str(artifact.get("path", ""))).expanduser().resolve()
            if not artifact_path.is_file() or artifact_path.is_symlink():
                raise FileNotFoundError(f"source {name}: provenance artifact missing")
            if sha256_file(artifact_path) != _sha(
                artifact.get("sha256"), f"source {name} provenance artifact"
            ):
                raise ValueError(f"source {name}: provenance artifact drifted")
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"source weights must sum to 1, got {weight_sum}")
    expected_source_weights = protocol["source_weights"]
    observed_source_weights = {
        str(source["name"]): float(source["weight"]) for source in sources
    }
    if observed_source_weights != expected_source_weights:
        raise ValueError(
            "stable training source identities/weights changed: "
            f"{observed_source_weights}"
        )
    if any(int(source.get("alignment_radius", 0)) != 0 for source in sources):
        raise ValueError("formal held-out-BSD training must not use residual alignment")
    disclosure = payload.get("data_mix_disclosure")
    required_disclosure = {
        "paper_discloses_exact_weights": False,
        "weights_are_preregistered_extension": True,
        "sampling": "strict_stage_order_then_one_source_per_step_ddp_global_batch6",
        "model_selection_or_validation_during_training": False,
        "bsd_dpdd_validation_and_tum_are_held_out": True,
        "bsd_training_pixels_used": False,
        "tum_training_pixels_used": False,
    }
    if disclosure != required_disclosure:
        raise ValueError("stable training data-mix disclosure changed")
    initialization = payload.get("initialization")
    if not isinstance(initialization, Mapping):
        raise ValueError("training initialization contract is required")
    if stage == "motion_base":
        if initialization != {"kind": "random_scratch_pinned_turtle_architecture"}:
            raise ValueError("motion_base must initialize TURTLE from scratch")
    else:
        expected_prior = "motion_base" if stage == "replica" else "replica"
        if initialization.get("kind") != "completed_prior_stage" or initialization.get("stage") != expected_prior:
            raise ValueError(f"{stage} must initialize from completed {expected_prior}")
        checkpoint = Path(str(initialization.get("checkpoint", ""))).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("defocus initialization checkpoint is missing")
        if sha256_file(checkpoint) != _sha(
            initialization.get("checkpoint_sha256"), "initialization checkpoint"
        ):
            raise ValueError("defocus initialization checkpoint SHA mismatch")
    return payload


@dataclass
class ParameterScopes:
    history: List[nn.Parameter]
    spatial: List[nn.Parameter]
    history_names: List[str]
    spatial_names: List[str]


def load_paired_image_train_manifest(manifest: Path, root: Path) -> List[SingleImagePair]:
    """Load a content-addressed generic 8-bit training-pair manifest."""

    records: List[SingleImagePair] = []
    names = set()
    root = root.expanduser().resolve()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if payload.get("schema") != PAIRED_IMAGE_SCHEMA:
                raise ValueError(f"paired-image schema mismatch at line {line_number}")
            if str(payload.get("split", "")).lower() not in {"train", "training"}:
                raise ValueError("paired-image manifest contains a non-train row")
            name = str(payload.get("name", ""))
            if not name or name in names:
                raise ValueError("paired-image names must be non-empty and unique")
            names.add(name)
            paths = []
            for key in ("blurry", "sharp"):
                relative = Path(str(payload.get(key, "")))
                lowered = {part.lower() for part in relative.parts}
                if relative.is_absolute() or lowered & {
                    "test", "testing", "validation", "valid", "val"
                }:
                    raise ValueError("paired-image paths must be relative train paths")
                resolved = (root / relative).resolve()
                try:
                    resolved.relative_to(root)
                except ValueError as error:
                    raise ValueError("paired-image path escapes its root") from error
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                paths.append(resolved)
            blurry, sharp = paths
            source_hash = _sha(payload.get("source_sha256"), "source_sha256")
            target_hash = _sha(payload.get("target_sha256"), "target_sha256")
            if sha256_file(blurry) != source_hash or sha256_file(sharp) != target_hash:
                raise ValueError(f"paired-image content SHA mismatch for {name}")
            with Image.open(blurry) as low, Image.open(sharp) as high:
                if low.size != high.size or low.mode != "RGB" or high.mode != "RGB":
                    raise ValueError(f"paired-image RGB/size mismatch for {name}")
            records.append(SingleImagePair(
                name=name, blurry=blurry, sharp=sharp, split="train",
                source_sha256=source_hash, target_sha256=target_hash,
            ))
    if not records:
        raise ValueError("paired-image manifest is empty")
    return records


def configure_full_scopes(model: nn.Module) -> ParameterScopes:
    """Train the whole model, while retaining a switchable history group."""

    model.requires_grad_(True)
    history_items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in HISTORY_ATTENTION_PARAMETER_PREFIXES)
    ]
    if (
        len(history_items) != HISTORY_ATTENTION_PARAMETER_TENSORS
        or sum(parameter.numel() for _, parameter in history_items)
        != HISTORY_ATTENTION_PARAMETER_COUNT
    ):
        raise RuntimeError("pinned TURTLE history parameter scope changed")
    history_ids = {id(parameter) for _, parameter in history_items}
    spatial_items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) not in history_ids
    ]
    if not spatial_items or not history_items:
        raise RuntimeError("full TURTLE parameter partition is empty")
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    partitioned = sum(parameter.numel() for _, parameter in history_items + spatial_items)
    if all_parameters != partitioned:
        raise RuntimeError("TURTLE parameter partition is not exhaustive")
    return ParameterScopes(
        history=[parameter for _, parameter in history_items],
        spatial=[parameter for _, parameter in spatial_items],
        history_names=[name for name, _ in history_items],
        spatial_names=[name for name, _ in spatial_items],
    )


def verify_content_addressed_video_manifest(manifest: Path, root: Path) -> None:
    """Fail closed on split leakage, path escape, or video-asset drift."""

    root = root.expanduser().resolve()
    seen_names = set()
    seen_paths = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if payload.get("schema") != PAIRED_VIDEO_SCHEMA:
                raise ValueError(f"video schema mismatch at line {line_number}")
            if str(payload.get("split", "")).lower() not in {"train", "training"}:
                raise ValueError("video manifest contains a non-train record")
            name = str(payload.get("sequence", ""))
            if not name or name in seen_names:
                raise ValueError("video sequence names must be non-empty and unique")
            seen_names.add(name)
            for path_key, sha_key in (
                ("blurry", "blurry_sha256"), ("sharp", "sharp_sha256")
            ):
                values = payload.get(path_key)
                hashes = payload.get(sha_key)
                if not isinstance(values, list) or not isinstance(hashes, list):
                    raise ValueError(f"sequence {name}: paths and SHA arrays are required")
                if len(values) < CLIP_LENGTH or len(values) != len(hashes):
                    raise ValueError(f"sequence {name}: invalid {path_key} length")
                for value, declared in zip(values, hashes):
                    relative = Path(str(value))
                    if relative.is_absolute():
                        raise ValueError("video paths must be relative")
                    resolved = (root / relative).resolve()
                    try:
                        resolved.relative_to(root)
                    except ValueError as error:
                        raise ValueError("video path escapes its root") from error
                    lowered = {part.lower() for part in relative.parts}
                    if lowered & {"test", "testing", "validation", "valid", "val"}:
                        raise ValueError("video path reaches a sealed split")
                    if resolved in seen_paths:
                        raise ValueError("video assets must not be reused across records/sides")
                    seen_paths.add(resolved)
                    if not resolved.is_file() or resolved.is_symlink():
                        raise FileNotFoundError(resolved)
                    if sha256_file(resolved) != _sha(declared, f"{name}.{sha_key}"):
                        raise ValueError(f"video asset SHA mismatch: {resolved}")
    if not seen_names:
        raise ValueError("video manifest is empty")


def _set_grad(parameters: Sequence[nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def best_integer_alignment(
    prediction: torch.Tensor,
    target: torch.Tensor,
    radius: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Select one detached global residual shift and return valid overlap."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("alignment inputs must be matching BCHW tensors")
    radius = int(radius)
    if radius <= 0:
        return prediction, target, (0, 0)
    height, width = prediction.shape[-2:]
    if 2 * radius >= min(height, width):
        raise ValueError("alignment radius is too large for the crop")
    best: Optional[Tuple[float, int, int]] = None
    with torch.no_grad():
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                py0, py1 = max(0, dy), min(height, height + dy)
                px0, px1 = max(0, dx), min(width, width + dx)
                ty0, ty1 = max(0, -dy), min(height, height - dy)
                tx0, tx1 = max(0, -dx), min(width, width - dx)
                error = float(
                    F.l1_loss(
                        prediction[..., py0:py1, px0:px1],
                        target[..., ty0:ty1, tx0:tx1],
                    )
                )
                candidate = (error, dy, dx)
                if best is None or candidate < best:
                    best = candidate
    assert best is not None
    _, dy, dx = best
    py0, py1 = max(0, dy), min(height, height + dy)
    px0, px1 = max(0, dx), min(width, width + dx)
    ty0, ty1 = max(0, -dy), min(height, height - dy)
    tx0, tx1 = max(0, -dx), min(width, width - dx)
    return (
        prediction[..., py0:py1, px0:px1],
        target[..., ty0:ty1, tx0:tx1],
        (dy, dx),
    )


def restoration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    alignment_radius: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    prediction_view, target_view, shift = best_integer_alignment(
        prediction, target, alignment_radius
    )
    l1 = F.l1_loss(prediction_view, target_view)
    fft = fft_l1_loss(prediction_view, target_view)
    return l1 + FFT_WEIGHT * fft, {
        "l1": float(l1.detach()),
        "fft": float(fft.detach()),
        "alignment_shift_yx": list(shift),
    }


def causal_clip_loss(
    model: nn.Module,
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    *,
    device: torch.device,
    alignment_radius: int,
    amp: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if blurry.shape != sharp.shape or blurry.ndim not in {4, 5}:
        raise ValueError("video clip must be matching TCHW or BTCHW tensors")
    if blurry.ndim == 4:
        blurry = blurry.unsqueeze(0)
        sharp = sharp.unsqueeze(0)
    if int(blurry.shape[1]) != CLIP_LENGTH:
        raise ValueError(f"video clip must have exactly {CLIP_LENGTH} frames")
    k_cache = v_cache = None
    previous = None
    terms: List[torch.Tensor] = []
    rows: List[Dict[str, Any]] = []
    context = torch.cuda.amp.autocast(enabled=True) if amp else nullcontext()
    with context:
        for frame_index in range(CLIP_LENGTH):
            current = blurry[:, frame_index].to(device, non_blocking=True)
            target = sharp[:, frame_index].to(device, non_blocking=True)
            prior = current if previous is None else previous
            pair = torch.stack((prior, current), dim=1)
            prediction, k_cache, v_cache = _validate_forward_result(
                model(pair, k_cache, v_cache), current
            )
            sample_losses = []
            sample_rows = []
            for sample_index in range(int(current.shape[0])):
                sample_loss, sample_row = restoration_loss(
                    prediction[sample_index : sample_index + 1],
                    target[sample_index : sample_index + 1],
                    alignment_radius=alignment_radius,
                )
                sample_losses.append(sample_loss)
                sample_rows.append({"sample": sample_index, **sample_row})
            loss = torch.stack(sample_losses).mean()
            terms.append(loss)
            rows.append({"frame": frame_index, "samples": sample_rows})
            previous = current
    return torch.stack(terms).mean(), {"frames": rows}


def single_batch_loss(
    model: nn.Module,
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    *,
    device: torch.device,
    alignment_radius: int,
    amp: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    blurry = blurry.to(device, non_blocking=True)
    sharp = sharp.to(device, non_blocking=True)
    pair = torch.stack((blurry, blurry), dim=1)
    context = torch.cuda.amp.autocast(enabled=True) if amp else nullcontext()
    with context:
        prediction, _, _ = _validate_forward_result(model(pair, None, None), blurry)
        return restoration_loss(
            prediction, sharp, alignment_radius=alignment_radius
        )


@dataclass
class RuntimeSource:
    name: str
    kind: str
    weight: float
    alignment_radius: int
    bit_depth: int
    dataset: Any
    rng: random.Random


def srgb_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the exact IEC sRGB inverse transfer function without quantizing."""

    if not tensor.is_floating_point():
        raise TypeError("sRGB tensor must be floating point")
    if torch.any((tensor < 0) | (tensor > 1)):
        raise ValueError("sRGB tensor must be in [0,1]")
    return torch.where(
        tensor <= 0.04045,
        tensor / 12.92,
        torch.pow((tensor + 0.055) / 1.055, 2.4),
    )


def _load_sources(
    contract: Mapping[str, Any], seed: int, *, rank: int = 0
) -> List[RuntimeSource]:
    sources: List[RuntimeSource] = []
    for index, spec in enumerate(contract["sources"]):
        manifest = Path(str(spec["manifest"])).expanduser().resolve()
        root = Path(str(spec["root"])).expanduser().resolve()
        kind = str(spec["kind"])
        if kind == "video":
            verify_content_addressed_video_manifest(manifest, root)
            dataset: Any = PairedSequenceDataset(
                manifest, root=root, crop_size=CROP_SIZE, augment=True,
                seed=seed + rank * 1_000_000 + 100_000 + index * 10_000,
            )
            if any(len(record.blurry) < CLIP_LENGTH for record in dataset.records):
                raise ValueError(f"video source {spec['name']} has a short sequence")
        else:
            if spec.get("manifest_kind") == "dpdd":
                dataset = load_single_image_manifest(
                    manifest,
                    root=root,
                    expected_split="train",
                    canonical_contract=True,
                    verify_content=True,
                )
            else:
                dataset = load_paired_image_train_manifest(manifest, root)
        sources.append(RuntimeSource(
            name=str(spec["name"]), kind=kind, weight=float(spec["weight"]),
            alignment_radius=int(spec.get("alignment_radius", 0)), dataset=dataset,
            bit_depth=int(spec.get("bit_depth", 8)),
            rng=random.Random(seed + rank * 1_000_000 + 200_000 + index * 10_000),
        ))
    return sources


def _choose_source(sources: Sequence[RuntimeSource], rng: random.Random) -> RuntimeSource:
    value = rng.random()
    cumulative = 0.0
    for source in sources:
        cumulative += source.weight
        if value < cumulative:
            return source
    return sources[-1]


def _video_batch(
    source: RuntimeSource, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    dataset: PairedSequenceDataset = source.dataset
    blurry_batch = []
    sharp_batch = []
    samples = []
    for _ in range(batch_size):
        sequence_index = source.rng.randrange(len(dataset.records))
        record = dataset.records[sequence_index]
        start = source.rng.randrange(len(record.blurry) - CLIP_LENGTH + 1)
        transform = _choose_transform(
            record, crop_size=CROP_SIZE, augment=True, rng=source.rng
        )
        blurry_batch.append(torch.stack([
            _read_transformed(path, transform)
            for path in record.blurry[start : start + CLIP_LENGTH]
        ]))
        sharp_batch.append(torch.stack([
            _read_transformed(path, transform)
            for path in record.sharp[start : start + CLIP_LENGTH]
        ]))
        samples.append({
            "sequence": record.name,
            "sequence_index": sequence_index,
            "start": start,
            "crop_box": list(transform.crop_box),
        })
    return srgb_to_linear(torch.stack(blurry_batch)), srgb_to_linear(torch.stack(sharp_batch)), {
        "samples": samples,
        "batch_size": batch_size,
        "photometric_transform": PHOTOMETRIC_TRANSFORM,
    }


def _single_batch(
    source: RuntimeSource, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    records: Sequence[SingleImagePair] = source.dataset
    blurry: List[torch.Tensor] = []
    sharp: List[torch.Tensor] = []
    rows: List[Dict[str, Any]] = []
    for _ in range(batch_size):
        index = source.rng.randrange(len(records))
        record = records[index]
        if source.bit_depth == 16:
            low, high, audit = _transform_tensor_pair(
                record, crop_size=CROP_SIZE, rng=source.rng
            )
        else:
            sequence = SequenceRecord(
                name=record.name,
                blurry=(record.blurry,),
                sharp=(record.sharp,),
            )
            transform = _choose_transform(
                sequence, crop_size=CROP_SIZE, augment=True, rng=source.rng
            )
            low = _read_transformed(record.blurry, transform)
            high = _read_transformed(record.sharp, transform)
            audit = {
                "name": record.name,
                "crop_box": list(transform.crop_box),
                "flip_horizontal": transform.flip_horizontal,
                "flip_vertical": transform.flip_vertical,
                "quarter_turns": transform.quarter_turns,
            }
        blurry.append(low)
        sharp.append(high)
        rows.append({"index": index, **audit})
    return srgb_to_linear(torch.stack(blurry)), srgb_to_linear(torch.stack(sharp)), {
        "pairs": rows,
        "photometric_transform": PHOTOMETRIC_TRANSFORM,
    }


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint(
    *,
    output_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    contract_path: Path,
    contract: Mapping[str, Any],
    scopes: ParameterScopes,
    source_draws: Mapping[str, int],
    sources: Sequence[RuntimeSource],
    chooser: random.Random,
    distributed_rank_states: Optional[Sequence[Mapping[str, Any]]] = None,
    amp_overflow_retries_total: int = 0,
) -> Path:
    path = output_dir / "checkpoints" / f"step_{step:06d}.pth"
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if digest_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint digest: {digest_path}")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "stage": str(contract["stage"]),
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
            "canonical_sha256": _json_sha(contract),
        },
        "source_draws": dict(source_draws),
        "sampling_rng": {
            "chooser": chooser.getstate(),
            "sources": {source.name: source.rng.getstate() for source in sources},
        },
        "distributed": {
            "world_size": int(contract["ddp_world_size"]),
            "backend": DDP_BACKEND,
            "local_batch_size": int(contract["batch_size_per_gpu"]),
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "rank_states": list(distributed_rank_states or []),
            "amp_overflow_retries_total": int(amp_overflow_retries_total),
            "amp_overflow_policy": "lower_scale_and_retry_exact_same_batch_without_scheduler_or_sample_advance",
        },
        "parameter_scope": {
            "history_tensors": len(scopes.history),
            "history_parameters": sum(p.numel() for p in scopes.history),
            "spatial_tensors": len(scopes.spatial),
            "spatial_parameters": sum(p.numel() for p in scopes.spatial),
        },
        "architecture_lineage": {
            "initialization_root": "random_scratch_pinned_turtle_architecture",
            "official_gopro_checkpoint_used_for_initialization": False,
            "repo_commit": PINNED_TURTLE_COMMIT,
            "arch_sha256": PINNED_TURTLE_ARCH_SHA256,
            "config_sha256": PINNED_TURTLE_CONFIG_SHA256,
            "cache_contract": TURTLE_CACHE_CONTRACT,
        },
    }
    _atomic_torch_save(payload, path)
    digest = sha256_file(path)
    fd = os.open(digest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(digest_path, 0o444)
    if step == TOTAL_STEPS and str(contract["stage"]) == "defocus_rehearsal":
        deploy_path = output_dir / "turtle_unblur_stable_deploy.pth"
        deploy_digest_path = deploy_path.with_suffix(deploy_path.suffix + ".sha256")
        if deploy_digest_path.exists():
            raise FileExistsError(
                f"refusing to overwrite deploy checkpoint digest: {deploy_digest_path}"
            )
        deploy_metadata = {
            "format": DEPLOY_CHECKPOINT_FORMAT,
            "stage": str(contract["stage"]),
            "step": step,
            "initialization_root": "random_scratch_pinned_turtle_architecture",
            "official_gopro_checkpoint_used_for_initialization": False,
            "turtle_repo_commit": PINNED_TURTLE_COMMIT,
            "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
            "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
            "input_domain": "linear_srgb",
            "output_domain": "linear_srgb",
            "photometric_transform": PHOTOMETRIC_TRANSFORM,
            "cache_contract": TURTLE_CACHE_CONTRACT,
            "training_contract_path": str(contract_path),
            "training_contract_sha256": sha256_file(contract_path),
            "training_contract_canonical_sha256": _json_sha(contract),
        }
        _atomic_torch_save(
            {"params": model.state_dict(), "metadata": deploy_metadata},
            deploy_path,
        )
        deploy_digest = sha256_file(deploy_path)
        fd = os.open(
            deploy_digest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(f"{deploy_digest}  {deploy_path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(deploy_digest_path, 0o444)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not all(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise RuntimeError("formal v3 training must be launched with torchrun")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    contract_path = args.contract.expanduser().resolve()
    if sha256_file(contract_path) != _sha(args.contract_sha256, "contract"):
        raise ValueError("contract SHA256 mismatch")
    contract = _read_contract(contract_path)
    expected_world_size = int(contract["ddp_world_size"])
    if world_size != expected_world_size or rank not in range(world_size) or local_rank not in range(world_size):
        raise RuntimeError(
            f"formal training requires exactly {expected_world_size} local DDP ranks"
        )
    if args.device not in {"cuda", "cuda:0"}:
        raise ValueError("DDP trainer derives cuda device from LOCAL_RANK")
    torch.cuda.set_device(local_rank)
    if torch.cuda.device_count() != world_size:
        raise RuntimeError(
            f"visible CUDA device count {torch.cuda.device_count()} does not equal DDP world size {world_size}"
        )
    dist.init_process_group(backend=DDP_BACKEND, init_method="env://")
    device = torch.device("cuda", local_rank)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal training requires CUDA")
    seed = int(contract["seed"])
    set_seed(seed)
    output_dir = Path(str(contract["output_dir"])).expanduser().resolve()
    storage_policy = contract.get("storage_policy")
    if not isinstance(storage_policy, Mapping) or storage_policy.get("fresh_no_overwrite") is not True:
        raise ValueError("training storage policy is missing or unsafe")
    output_parent = Path(str(storage_policy.get("output_root_parent", ""))).expanduser().resolve()
    if output_parent == Path("/"):
        raise ValueError("training output parent must not be filesystem root")
    try:
        output_dir.relative_to(output_parent)
    except ValueError as error:
        raise ValueError("training output lies outside its contracted parent") from error
    if rank == 0 and args.resume is None:
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"fresh training output already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    if args.resume is not None:
        resume_candidate = args.resume.expanduser().resolve()
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise FileNotFoundError("resume requires the existing non-symlink output directory")
        try:
            resume_candidate.relative_to(output_dir / "checkpoints")
        except ValueError as error:
            raise ValueError("resume checkpoint must belong to this output directory") from error
    model, base_metadata = build_turtle_model_from_scratch(
        DEFAULT_TURTLE_REPO, config=DEFAULT_TURTLE_CONFIG, device=device,
    )
    if base_metadata.get("kind") != "random_initialization":
        raise ValueError("training must construct the pinned TURTLE architecture from scratch")
    initialization = contract["initialization"]
    if initialization["kind"] == "completed_prior_stage":
        prior = torch.load(
            Path(str(initialization["checkpoint"])).expanduser().resolve(),
            map_location="cpu",
        )
        if (
            prior.get("format") != CHECKPOINT_FORMAT
            or int(prior.get("step", -1)) != TOTAL_STEPS
            or prior.get("stage") != initialization["stage"]
        ):
            raise ValueError("initialization is not the declared completed prior-stage checkpoint")
        model.load_state_dict(prior["model"], strict=True)
        # Stage-local sampling/crop randomness begins from the same declared seed;
        # strict state loading must not consume the formal training RNG stream.
        set_seed(seed)
    scopes = configure_full_scopes(model)
    distributed_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    optimizer = CountingAdamW(
        [
            {"params": scopes.history, "lr": float(contract["learning_rate"]), "weight_decay": 1e-3},
            {"params": scopes.spatial, "lr": float(contract["learning_rate"]), "weight_decay": 1e-3},
        ],
        betas=(0.9, 0.9),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_STEPS, eta_min=float(contract["scheduler_eta_min"])
    )
    amp = bool(contract.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp,
        init_scale=float(contract["amp_initial_scale"]),
        growth_interval=int(contract["amp_growth_interval"]),
    )
    sources = _load_sources(contract, seed, rank=rank)
    chooser = random.Random(seed + 300_000)
    source_draws = {source.name: 0 for source in sources}
    amp_overflow_retries_total = 0
    start_step = 0
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        resume = torch.load(resume_path, map_location="cpu")
        if resume.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("resume checkpoint format mismatch")
        if resume.get("contract", {}).get("sha256") != sha256_file(contract_path):
            raise ValueError("resume checkpoint belongs to another contract")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        distributed = resume.get("distributed")
        if not isinstance(distributed, Mapping) or int(distributed.get("world_size", -1)) != world_size:
            raise ValueError("resume checkpoint does not contain the two-rank DDP state")
        rank_states = distributed.get("rank_states")
        if not isinstance(rank_states, list) or len(rank_states) != world_size:
            raise ValueError("resume checkpoint rank-state set is incomplete")
        rank_state = rank_states[rank]
        random.setstate(rank_state["python_rng"])
        torch.random.set_rng_state(rank_state["torch_cpu_rng"])
        torch.cuda.set_rng_state(rank_state["torch_cuda_rng"], device=device)
        source_draws = dict(resume["source_draws"])
        chooser.setstate(rank_state["chooser_rng"])
        source_states = rank_state["source_rngs"]
        if set(source_states) != {source.name for source in sources}:
            raise ValueError("resume source RNG set differs from the contract")
        for source in sources:
            source.rng.setstate(source_states[source.name])
        start_step = int(resume["step"])
        amp_overflow_retries_total = int(distributed.get("amp_overflow_retries_total", 0))
        if optimizer.step_count != start_step:
            # CountingAdamW's custom counter is not in the vanilla state dict.
            optimizer.step_count = start_step
    stop = TOTAL_STEPS if args.stop_after_step is None else int(args.stop_after_step)
    if stop <= start_step or stop > TOTAL_STEPS:
        raise ValueError("stop-after-step must be in (resume_step, 300000]")
    distributed_model.train()
    checkpoint_every = int(contract["checkpoint_every"])
    batch_size = int(contract["batch_size_per_gpu"])
    log_path = output_dir / "training.jsonl"
    if rank == 0 and start_step == 0 and log_path.exists():
        raise FileExistsError(f"refusing to append an existing fresh log: {log_path}")
    for step_index in range(start_step, stop):
        source = _choose_source(sources, chooser)
        source_index = next(index for index, item in enumerate(sources) if item.name == source.name)
        source_bounds = torch.tensor([source_index, source_index], device=device, dtype=torch.int64)
        dist.all_reduce(source_bounds[:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(source_bounds[1:], op=dist.ReduceOp.MAX)
        if int(source_bounds[0]) != int(source_bounds[1]):
            raise RuntimeError("DDP ranks selected different sources")
        source_draws[source.name] += 1
        if source.kind == "video":
            blurry, sharp, sample_audit = _video_batch(source, batch_size)
        else:
            blurry, sharp, sample_audit = _single_batch(source, batch_size)
        retry_count = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            if source.kind == "video":
                loss, loss_audit = causal_clip_loss(
                    distributed_model, blurry, sharp, device=device,
                    alignment_radius=source.alignment_radius, amp=amp,
                )
            else:
                loss, loss_audit = single_batch_loss(
                    distributed_model, blurry, sharp, device=device,
                    alignment_radius=source.alignment_radius, amp=amp,
                )
            if amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if amp:
                scaler.unscale_(optimizer)
            if source.kind == "single":
                for parameter in scopes.history:
                    parameter.grad = None
            active = scopes.spatial + (scopes.history if source.kind == "video" else [])
            gradients = [parameter.grad for parameter in active if parameter.grad is not None]
            locally_finite = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
            finite_flag = torch.tensor(1 if locally_finite else 0, device=device, dtype=torch.int32)
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if int(finite_flag) == 1:
                break
            if not amp or retry_count >= int(contract["amp_max_same_batch_retries"]):
                raise FloatingPointError("DDP gradients remained non-finite after same-batch AMP retries")
            # A rank-local scaler.step() is unsafe here: one rank may be finite
            # while its peer overflowed, which would update only half the DDP
            # replicas.  The globally reduced flag makes both ranks lower the
            # identical scale without stepping, then recompute this exact batch.
            previous_scale = float(scaler.get_scale())
            scaler.update(new_scale=_next_amp_retry_scale(previous_scale))
            if float(scaler.get_scale()) >= previous_scale and previous_scale > 1.0:
                raise RuntimeError("AMP retry did not lower the shared loss scale")
            retry_count += 1
            amp_overflow_retries_total += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(
            active, max_norm=1.0, error_if_nonfinite=True
        )
        executed = execute_checked_optimizer_step(
            optimizer, amp_enabled=amp, scaler=scaler, scheduler=scheduler
        )
        completed = step_index + 1
        if executed != completed:
            raise RuntimeError("optimizer execution counter drifted")
        mean_loss = loss.detach().float().clone()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss /= world_size
        row = {
            "step": completed,
            "source": source.name,
            "kind": source.kind,
            "loss": float(mean_loss),
            "grad_norm_before_clip": float(grad_norm),
            "lr": [group["lr"] for group in optimizer.param_groups],
            "rank0_sample": sample_audit if rank == 0 else None,
            "rank0_loss_detail": loss_audit,
            "ddp": {"world_size": world_size, "local_batch_size": batch_size,
                    "global_batch_size": batch_size * world_size},
            "amp_same_batch_retries": retry_count,
            "amp_overflow_retries_total": amp_overflow_retries_total,
        }
        if rank == 0:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        if rank == 0 and completed % int(contract.get("print_every", 10)) == 0:
            print(json.dumps({
                "step": completed,
                "source": source.name,
                "loss": float(mean_loss),
                "grad_norm_before_clip": float(grad_norm),
                "lr": row["lr"],
                "amp_same_batch_retries": retry_count,
                "amp_overflow_retries_total": amp_overflow_retries_total,
            }, sort_keys=True), flush=True)
        if completed % checkpoint_every == 0 or completed == stop:
            local_state = {
                "rank": rank,
                "python_rng": random.getstate(),
                "torch_cpu_rng": torch.random.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state(device).cpu(),
                "chooser_rng": chooser.getstate(),
                "source_rngs": {item.name: item.rng.getstate() for item in sources},
            }
            gathered_states: Optional[List[Any]] = [None] * world_size if rank == 0 else None
            dist.gather_object(local_state, gathered_states, dst=0)
            if rank == 0:
                checkpoint = _checkpoint(
                    output_dir=output_dir, step=completed, model=model,
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    contract_path=contract_path, contract=contract, scopes=scopes,
                    source_draws=source_draws, sources=sources, chooser=chooser,
                    distributed_rank_states=gathered_states,
                    amp_overflow_retries_total=amp_overflow_retries_total,
                )
                print(json.dumps({"checkpoint": str(checkpoint), "step": completed}), flush=True)
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
