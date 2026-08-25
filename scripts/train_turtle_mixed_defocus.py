#!/usr/bin/env python3
"""Minimal source-scoped TURTLE fine-tuning for Replica424 plus DPDD.

The preregistered mixed arm has 78 *joint* optimizer steps.  At each step a
DPDD batch is backpropagated with only ``refinement`` and ``ending`` trainable,
then one Replica sequence is backpropagated through the complete causal K/V
prefix with the five history attentions and the spatial head trainable.  One
group-wise clip, optimizer step, and scheduler step follow the two backwards.

This file intentionally has no test-split option and accepts only the canonical
content-addressed DPDD train manifest.  It is not a GPU launcher.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_turtle_single_image_defocus import (  # noqa: E402
    CANONICAL_PAIR_SCHEMA,
    SingleImagePair,
    load_single_image_manifest,
    read_pair_tensor,
)
from scripts.train_turtle_streaming import (  # noqa: E402
    DEFAULT_TURTLE_CHECKPOINT,
    DEFAULT_TURTLE_CONFIG,
    DEFAULT_TURTLE_REPO,
    HISTORY_ATTENTION_PARAMETER_COUNT,
    HISTORY_ATTENTION_PARAMETER_PREFIXES,
    HISTORY_ATTENTION_PARAMETER_TENSORS,
    PairedSequenceDataset,
    _choose_transform,
    _read_transformed,
    _replay_current_from_past,
    _validate_forward_result,
    choose_device,
    cyclically_shuffled_past_indices,
    fft_l1_loss,
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


TRAINING_SCHEMA = "unblur_slam.turtle_replica424_dpdd_hf_mixed_training.v3"
SPATIAL_HEAD_PREFIXES = ("refinement.", "ending.")
SPATIAL_HEAD_PARAMETER_TENSORS = 30
SPATIAL_HEAD_PARAMETER_COUNT = 105_283
FORMAL_STEPS = 78
FORMAL_DPDD_BATCH = 5
FORMAL_DPDD_TRAIN_COUNT = 350
FORMAL_VIDEO_SEQUENCE_COUNT = 26
FORMAL_SEEDS = frozenset({17, 42, 73})
FORMAL_VIDEO_MANIFEST_SHA256 = (
    "bd7caa189374683c8ffd7e8fce83cb62e5f69b73f6048808c4808dc2b4ecd2ba"
)
DPDD_DATASET_MANIFEST_SCHEMA = "unblur_slam.dpdd_hf_png16_materialization.v1"
DPDD_REPOSITORY = "JacobLinCool/DPDD"
DPDD_REVISION = "52e4035a045ea1763313b9ce2b27cf2e620cfc30"
DPDD_CONFIG = "combined"


def _normalized_sha256(value: Any, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{label} must be one SHA256")
    return normalized


def load_dpdd_dataset_contract(
    dataset_manifest: Path,
    *,
    expected_dataset_manifest_sha256: str,
    train_manifest: Path,
    expected_train_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Verify and reduce the immutable DPDD materialization provenance."""

    path = Path(dataset_manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DPDD dataset manifest does not exist: {path}")
    expected_dataset_hash = _normalized_sha256(
        expected_dataset_manifest_sha256,
        label="DPDD dataset-manifest SHA256",
    )
    actual_dataset_hash = sha256_file(path)
    if actual_dataset_hash != expected_dataset_hash:
        raise ValueError("DPDD dataset-manifest SHA256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("DPDD dataset manifest is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("DPDD dataset manifest must be a JSON object")
    fixed = {
        "schema": DPDD_DATASET_MANIFEST_SCHEMA,
        "repository": DPDD_REPOSITORY,
        "revision": DPDD_REVISION,
        "config": DPDD_CONFIG,
        "splits": {"train": 350, "validation": 74},
    }
    mismatches = {
        key: (payload.get(key), wanted)
        for key, wanted in fixed.items()
        if payload.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"DPDD dataset provenance mismatch: {mismatches}")

    distribution = payload.get("distribution")
    if not isinstance(distribution, Mapping):
        raise ValueError("DPDD dataset manifest has no distribution provenance")
    license_claim = distribution.get("dataset_card_declared_license")
    license_warning = distribution.get("license_scope_warning")
    if str(license_claim).strip().lower() != "mit" or not isinstance(
        license_warning, str
    ) or not license_warning.strip():
        raise ValueError("DPDD dataset license claim/warning is incomplete")

    disclosure = payload.get("test_disclosure")
    if not isinstance(disclosure, Mapping):
        raise ValueError("DPDD dataset manifest has no test disclosure")
    required_disclosure = {
        "metadata_pristine": False,
        "images_decoded": False,
        "pixels_opened": False,
        "metrics_opened": False,
        "split_supported_by_this_materializer": False,
    }
    disclosure_mismatches = {
        key: (disclosure.get(key), wanted)
        for key, wanted in required_disclosure.items()
        if disclosure.get(key) is not wanted
    }
    if disclosure_mismatches:
        raise ValueError(f"DPDD test disclosure mismatch: {disclosure_mismatches}")

    canonical = payload.get("canonical_manifests")
    if not isinstance(canonical, Mapping):
        raise ValueError("DPDD canonical manifest index is missing")
    dataset_root = path.parent.resolve()
    normalized_train_hash = _normalized_sha256(
        expected_train_manifest_sha256,
        label="DPDD train-manifest SHA256",
    )
    for split, expected_rows in (("train", 350), ("validation", 74)):
        entry = canonical.get(split)
        if not isinstance(entry, Mapping):
            raise ValueError(f"DPDD canonical {split} manifest metadata is missing")
        raw_relative = Path(str(entry.get("path", "")))
        if raw_relative.is_absolute():
            raise ValueError(f"DPDD canonical {split} path must be relative")
        resolved = (dataset_root / raw_relative).resolve()
        try:
            resolved.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(f"DPDD canonical {split} path escapes dataset root") from error
        if (
            entry.get("schema") != CANONICAL_PAIR_SCHEMA
            or entry.get("rows") != expected_rows
            or entry.get("paths_relative_to") != "dataset_root"
        ):
            raise ValueError(f"DPDD canonical {split} contract mismatch")
        recorded_hash = _normalized_sha256(
            entry.get("sha256"), label=f"DPDD canonical {split} SHA256"
        )
        if not resolved.is_file() or sha256_file(resolved) != recorded_hash:
            raise ValueError(f"DPDD canonical {split} manifest content mismatch")
        if split == "train":
            if resolved != Path(train_manifest).expanduser().resolve():
                raise ValueError("--dpdd-pairs is not the dataset manifest's train split")
            if recorded_hash != normalized_train_hash:
                raise ValueError("DPDD train-manifest SHA256 provenance mismatch")

    return {
        "path": str(path),
        "sha256": actual_dataset_hash,
        "schema": payload["schema"],
        "repository": payload["repository"],
        "revision": payload["revision"],
        "config": payload["config"],
        "dataset_card_declared_license": license_claim,
        "license_scope_warning": license_warning,
        "test_metadata_pristine": disclosure["metadata_pristine"],
        "test_pixels_opened": disclosure["pixels_opened"],
        "test_metrics_opened": disclosure["metrics_opened"],
        "canonical_train_manifest_sha256": normalized_train_hash,
    }


@dataclass
class ParameterScopes:
    history: List[nn.Parameter]
    spatial: List[nn.Parameter]
    history_names: List[str]
    spatial_names: List[str]


class CountingAdamW(torch.optim.AdamW):
    """Expose executed steps so AMP overflow skips cannot masquerade as updates."""

    def __init__(self, parameters: Any, **kwargs: Any):
        super().__init__(parameters, **kwargs)
        self.step_count = 0

    def step(self, closure: Optional[Any] = None):
        result = super().step(closure=closure)
        self.step_count += 1
        return result


def configure_parameter_scopes(model: nn.Module, mode: str) -> ParameterScopes:
    """Freeze everything except exact pinned history/spatial parameter sets."""

    normalized = str(mode).strip().upper()
    if normalized not in {"V", "S", "M"}:
        raise ValueError("training mode must be V, S, or M; G is the frozen base")
    model.requires_grad_(False)
    history_items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in HISTORY_ATTENTION_PARAMETER_PREFIXES)
    ]
    spatial_items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in SPATIAL_HEAD_PREFIXES)
    ]
    if (
        len(history_items) != HISTORY_ATTENTION_PARAMETER_TENSORS
        or sum(parameter.numel() for _, parameter in history_items)
        != HISTORY_ATTENTION_PARAMETER_COUNT
    ):
        raise RuntimeError("pinned TURTLE history scope changed")
    if (
        len(spatial_items) != SPATIAL_HEAD_PARAMETER_TENSORS
        or sum(parameter.numel() for _, parameter in spatial_items)
        != SPATIAL_HEAD_PARAMETER_COUNT
    ):
        raise RuntimeError("pinned TURTLE spatial-head scope changed")
    history = [parameter for _, parameter in history_items]
    spatial = [parameter for _, parameter in spatial_items]
    for parameter in spatial:
        parameter.requires_grad_(True)
    if normalized in {"V", "M"}:
        for parameter in history:
            parameter.requires_grad_(True)
    return ParameterScopes(
        history=history,
        spatial=spatial,
        history_names=[name for name, _ in history_items],
        spatial_names=[name for name, _ in spatial_items],
    )


def deterministic_single_schedule(
    names: Sequence[str], *, seed: int, steps: int = FORMAL_STEPS, batch_size: int = FORMAL_DPDD_BATCH
) -> List[List[int]]:
    if not names or len(set(names)) != len(names):
        raise ValueError("DPDD names must be non-empty and unique")
    order = list(range(len(names)))
    random.Random(int(seed) + 30_000).shuffle(order)
    needed = int(steps) * int(batch_size)
    expanded = [order[index % len(order)] for index in range(needed)]
    return [expanded[start : start + batch_size] for start in range(0, needed, batch_size)]


def deterministic_video_schedule(
    eligible_indices: Sequence[int], *, seed: int, steps: int = FORMAL_STEPS
) -> List[Tuple[int, int]]:
    if not eligible_indices:
        raise ValueError("Replica schedule requires eligible sequences")
    schedule: List[Tuple[int, int]] = []
    pass_index = 0
    while len(schedule) < steps:
        order = list(eligible_indices)
        random.Random(int(seed) + 10_000 + pass_index).shuffle(order)
        schedule.extend((pass_index, index) for index in order)
        pass_index += 1
    return schedule[:steps]


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _transform_tensor_pair(
    record: SingleImagePair,
    *,
    crop_size: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    blurry = read_pair_tensor(
        record.blurry, device=torch.device("cpu"), require_png_rgb16=True
    )
    sharp = read_pair_tensor(
        record.sharp, device=torch.device("cpu"), require_png_rgb16=True
    )
    if blurry.shape != sharp.shape:
        raise ValueError(f"DPDD pair {record.name!r} is not aligned")
    height, width = (int(value) for value in blurry.shape[-2:])
    if crop_size > min(height, width):
        raise ValueError(f"crop exceeds DPDD pair {record.name!r}")
    top = rng.randrange(height - crop_size + 1)
    left = rng.randrange(width - crop_size + 1)
    blurry = blurry[:, top : top + crop_size, left : left + crop_size]
    sharp = sharp[:, top : top + crop_size, left : left + crop_size]
    horizontal = bool(rng.randrange(2))
    vertical = bool(rng.randrange(2))
    quarter_turns = rng.randrange(4)
    if horizontal:
        blurry, sharp = torch.flip(blurry, (-1,)), torch.flip(sharp, (-1,))
    if vertical:
        blurry, sharp = torch.flip(blurry, (-2,)), torch.flip(sharp, (-2,))
    if quarter_turns:
        blurry = torch.rot90(blurry, quarter_turns, (-2, -1))
        sharp = torch.rot90(sharp, quarter_turns, (-2, -1))
    audit = {
        "name": record.name,
        "crop_box_left_top_size": [left, top, crop_size],
        "flip_horizontal": horizontal,
        "flip_vertical": vertical,
        "quarter_turns": quarter_turns,
    }
    return blurry, sharp, audit


def load_single_batch(
    records: Sequence[SingleImagePair],
    indices: Sequence[int],
    *,
    seed: int,
    step: int,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    blurry: List[torch.Tensor] = []
    sharp: List[torch.Tensor] = []
    audits: List[Dict[str, Any]] = []
    for position, index in enumerate(indices):
        rng = random.Random(
            int(seed) + 40_000 + int(step) * 1_000_003 + int(index) * 9_176 + position
        )
        source, target, audit = _transform_tensor_pair(
            records[index], crop_size=crop_size, rng=rng
        )
        blurry.append(source)
        sharp.append(target)
        audits.append(audit)
    return torch.stack(blurry), torch.stack(sharp), audits


def load_video_sample(
    dataset: PairedSequenceDataset,
    *,
    sequence_index: int,
    pass_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    record = dataset.records[sequence_index]
    rng_seed = dataset.seed + int(pass_index) * 1_000_003 + int(sequence_index) * 9_176
    transform = _choose_transform(
        record,
        crop_size=dataset.crop_size,
        augment=dataset.augment,
        rng=random.Random(rng_seed),
    )
    blurry = torch.stack([_read_transformed(path, transform) for path in record.blurry])
    sharp = torch.stack([_read_transformed(path, transform) for path in record.sharp])
    audit = {
        "sequence": record.name,
        "pass_index": int(pass_index),
        "sequence_index": int(sequence_index),
        "transform_rng_seed": int(rng_seed),
        "crop_box": list(transform.crop_box),
        "flip_horizontal": transform.flip_horizontal,
        "flip_vertical": transform.flip_vertical,
        "quarter_turns": transform.quarter_turns,
    }
    return blurry, sharp, audit


def single_image_objective(
    model: nn.Module,
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    *,
    device: torch.device,
    fft_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    blurry = blurry.to(device=device, dtype=torch.float32, non_blocking=True)
    sharp = sharp.to(device=device, dtype=torch.float32, non_blocking=True)
    pair = torch.stack((blurry, blurry), dim=1)
    prediction, _, _ = _validate_forward_result(model(pair, None, None), blurry)
    l1 = F.l1_loss(prediction, sharp)
    fft = fft_l1_loss(prediction, sharp)
    loss = l1 + float(fft_weight) * fft
    return loss, {"single_l1": float(l1.detach()), "single_fft": float(fft.detach())}


def video_objective(
    model: nn.Module,
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    *,
    device: torch.device,
    fft_weight: float,
    temporal_delta_weight: float,
    order_contrast_weight: float,
    order_contrast_margin: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    frame_count = int(blurry.shape[0])
    if frame_count < 3 or sharp.shape != blurry.shape:
        raise ValueError("video objective requires a matching sequence of at least 3 frames")
    k_cache = v_cache = None
    previous = None
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    l1_terms: List[torch.Tensor] = []
    fft_terms: List[torch.Tensor] = []
    temporal_terms: List[torch.Tensor] = []
    for frame_index in range(frame_count):
        current = blurry[frame_index : frame_index + 1].to(device=device)
        target = sharp[frame_index : frame_index + 1].to(device=device)
        prior = current if previous is None else previous
        pair = torch.stack((prior[0], current[0]), dim=0).unsqueeze(0)
        prediction, k_cache, v_cache = _validate_forward_result(
            model(pair, k_cache, v_cache), current
        )
        predictions.append(prediction)
        targets.append(target)
        if frame_index >= 1:
            l1_terms.append(F.l1_loss(prediction, target))
            fft_terms.append(fft_l1_loss(prediction, target))
            temporal_terms.append(
                F.l1_loss(prediction - predictions[-2], target - targets[-2])
            )
        previous = current
    l1 = torch.stack(l1_terms).mean()
    fft = torch.stack(fft_terms).mean()
    temporal = torch.stack(temporal_terms).mean()
    anchor = frame_count - 1
    shuffled = _replay_current_from_past(
        model,
        blurry,
        anchor_index=anchor,
        past_indices=cyclically_shuffled_past_indices(anchor),
        device=device,
    )
    ordered_error = F.l1_loss(predictions[anchor], targets[anchor])
    shuffled_error = F.l1_loss(shuffled, targets[anchor])
    order_gap = shuffled_error - ordered_error
    rank = F.relu(float(order_contrast_margin) - order_gap)
    loss = (
        l1
        + float(fft_weight) * fft
        + float(temporal_delta_weight) * temporal
        + float(order_contrast_weight) * rank
    )
    return loss, {
        "video_l1": float(l1.detach()),
        "video_fft": float(fft.detach()),
        "video_temporal_delta": float(temporal.detach()),
        "video_order_gap": float(order_gap.detach()),
        "video_order_rank": float(rank.detach()),
    }


def _set_requires_grad(parameters: Sequence[nn.Parameter], value: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(value)


def _capture_torch_rng() -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    return cpu_state, cuda_states


def _restore_torch_rng(
    state: Tuple[torch.Tensor, Optional[List[torch.Tensor]]]
) -> None:
    cpu_state, cuda_states = state
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def execute_checked_optimizer_step(
    optimizer: torch.optim.Optimizer,
    *,
    amp_enabled: bool,
    scaler: Optional[Any],
    scheduler: Optional[Any],
) -> int:
    """Execute one update and fail if AMP silently skipped it."""

    if not hasattr(optimizer, "step_count"):
        raise TypeError("formal optimizer must expose executed step_count")
    before = int(optimizer.step_count)
    if amp_enabled:
        if scaler is None:
            raise ValueError("AMP requires a GradScaler")
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    after = int(optimizer.step_count)
    if after != before + 1:
        raise FloatingPointError(
            "GradScaler skipped the optimizer update; formal 78-step budget aborted"
        )
    if scheduler is not None:
        scheduler.step()
    return after


def source_scoped_optimizer_step(
    model: nn.Module,
    scopes: ParameterScopes,
    optimizer: torch.optim.Optimizer,
    *,
    mode: str,
    device: torch.device,
    single_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
    video_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
    fft_weight: float = 0.1,
    temporal_delta_weight: float = 0.1,
    order_contrast_weight: float = 1.0,
    order_contrast_margin: float = 0.0001,
    history_grad_clip: float = 1.0,
    spatial_grad_clip: float = 1.0,
    amp_enabled: bool = False,
    scaler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Perform at most two backwards followed by exactly one optimizer step."""

    normalized = mode.upper()
    if normalized == "M" and (single_batch is None or video_batch is None):
        raise ValueError("M requires both DPDD and Replica batches")
    if normalized == "V" and (single_batch is not None or video_batch is None):
        raise ValueError("V requires only one Replica batch")
    if normalized == "S" and (single_batch is None or video_batch is not None):
        raise ValueError("S requires only one DPDD batch")
    if amp_enabled and scaler is None:
        raise ValueError("AMP requires a GradScaler")
    optimizer.zero_grad(set_to_none=True)
    if not hasattr(optimizer, "step_count"):
        raise TypeError("formal optimizer must expose executed step_count")
    metrics: Dict[str, Any] = {}
    autocast = torch.cuda.amp.autocast if amp_enabled else None

    if single_batch is not None:
        # In M, DPDD must not advance the random stream subsequently consumed
        # by the matched Replica video forward. This keeps V/M video RNG
        # trajectories identical even if a future pinned module becomes
        # stochastic in train mode.
        pre_single_rng = _capture_torch_rng() if normalized == "M" else None
        _set_requires_grad(scopes.history, False)
        _set_requires_grad(scopes.spatial, True)
        context = autocast(enabled=True) if autocast is not None else nullcontext()
        with context:
            single_loss, single_metrics = single_image_objective(
                model,
                single_batch[0],
                single_batch[1],
                device=device,
                fft_weight=fft_weight,
            )
        if amp_enabled:
            scaler.scale(single_loss).backward()
        else:
            single_loss.backward()
        if any(parameter.grad is not None for parameter in scopes.history):
            raise RuntimeError("DPDD single-image backward reached history parameters")
        metrics.update(single_metrics)
        metrics["single_loss"] = float(single_loss.detach())
        if pre_single_rng is not None:
            _restore_torch_rng(pre_single_rng)

    if video_batch is not None:
        _set_requires_grad(scopes.history, True)
        _set_requires_grad(scopes.spatial, True)
        context = autocast(enabled=True) if autocast is not None else nullcontext()
        with context:
            video_loss, video_metrics = video_objective(
                model,
                video_batch[0],
                video_batch[1],
                device=device,
                fft_weight=fft_weight,
                temporal_delta_weight=temporal_delta_weight,
                order_contrast_weight=order_contrast_weight,
                order_contrast_margin=order_contrast_margin,
            )
        if amp_enabled:
            scaler.scale(video_loss).backward()
        else:
            video_loss.backward()
        if not any(parameter.grad is not None for parameter in scopes.history):
            raise RuntimeError("Replica video backward did not reach history parameters")
        metrics.update(video_metrics)
        metrics["video_loss"] = float(video_loss.detach())

    active_history = scopes.history if video_batch is not None else []
    if amp_enabled:
        scaler.unscale_(optimizer)
    for label, parameters in (
        ("history", active_history),
        ("spatial", scopes.spatial),
    ):
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if parameters and not gradients:
            raise RuntimeError(f"{label} parameter group has no gradients")
        if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError(f"{label} parameter group has non-finite gradients")
    # Separate group clipping prevents the DPDD spatial gradient from changing
    # the history-gradient normalization used by the matched V arm.
    history_norm = (
        torch.nn.utils.clip_grad_norm_(
            active_history, history_grad_clip, error_if_nonfinite=True
        )
        if active_history and history_grad_clip > 0
        else torch.tensor(0.0)
    )
    spatial_norm = (
        torch.nn.utils.clip_grad_norm_(
            scopes.spatial, spatial_grad_clip, error_if_nonfinite=True
        )
        if spatial_grad_clip > 0
        else torch.tensor(0.0)
    )
    executed_steps = execute_checked_optimizer_step(
        optimizer,
        amp_enabled=amp_enabled,
        scaler=scaler,
        scheduler=scheduler,
    )
    metrics["history_grad_norm_before_group_clip"] = float(history_norm)
    metrics["spatial_grad_norm_before_group_clip"] = float(spatial_norm)
    metrics["optimizer_step_executed"] = True
    metrics["executed_optimizer_steps"] = executed_steps
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("V", "S", "M"), required=True)
    parser.add_argument("--video-manifest", type=Path)
    parser.add_argument(
        "--video-root",
        type=Path,
        help="root used to resolve relative paths in --video-manifest (required for V/M)",
    )
    parser.add_argument("--dpdd-pairs", type=Path)
    parser.add_argument("--dpdd-root", type=Path)
    parser.add_argument("--dpdd-manifest-sha256")
    parser.add_argument("--dpdd-dataset-manifest", type=Path)
    parser.add_argument("--dpdd-dataset-manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turtle-repo", type=Path, default=DEFAULT_TURTLE_REPO)
    parser.add_argument("--turtle-config", type=Path, default=DEFAULT_TURTLE_CONFIG)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_TURTLE_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=FORMAL_STEPS)
    parser.add_argument("--dpdd-batch-size", type=int, default=FORMAL_DPDD_BATCH)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode.upper()
    if args.steps != FORMAL_STEPS or args.dpdd_batch_size != FORMAL_DPDD_BATCH:
        raise ValueError("formal v3 contract fixes 78 steps and DPDD batch size 5")
    if args.seed not in FORMAL_SEEDS:
        raise ValueError("formal v3 seed must be 17, 42, or 73")
    if args.crop_size != 128:
        raise ValueError("formal v3 contract fixes crop size 128")
    if mode in {"V", "M"} and (
        args.video_manifest is None or args.video_root is None
    ):
        raise ValueError(f"mode {mode} requires --video-manifest and --video-root")
    if mode in {"S", "M"} and (
        args.dpdd_pairs is None
        or args.dpdd_manifest_sha256 is None
        or args.dpdd_dataset_manifest is None
        or args.dpdd_dataset_manifest_sha256 is None
    ):
        raise ValueError(
            f"mode {mode} requires content-addressed --dpdd-pairs and "
            "--dpdd-dataset-manifest"
        )
    if mode == "V" and any(
        value is not None
        for value in (
            args.dpdd_pairs,
            args.dpdd_root,
            args.dpdd_manifest_sha256,
            args.dpdd_dataset_manifest,
            args.dpdd_dataset_manifest_sha256,
        )
    ):
        raise ValueError("V must not receive any DPDD arguments")
    if mode == "S" and (args.video_manifest is not None or args.video_root is not None):
        raise ValueError("S must not receive Replica --video-manifest/--video-root")
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type != "cuda" or not args.amp:
        raise ValueError("formal V/S/M training requires CUDA with --amp")
    output = args.output.expanduser().resolve()
    digest_output = output.with_name(output.name + ".sha256")
    if output.exists() or digest_output.exists():
        raise FileExistsError(f"formal trainer refuses overwrite: {output}")

    video_dataset = None
    video_schedule: List[Tuple[int, int]] = []
    if mode in {"V", "M"}:
        if sha256_file(args.video_manifest) != FORMAL_VIDEO_MANIFEST_SHA256:
            raise ValueError("Replica424 training manifest SHA256 mismatch")
        video_dataset = PairedSequenceDataset(
            args.video_manifest,
            root=args.video_root,
            crop_size=args.crop_size,
            augment=True,
            seed=args.seed + 20_000,
        )
        eligible = [
            index for index, record in enumerate(video_dataset.records) if len(record.blurry) >= 3
        ]
        if len(eligible) != FORMAL_VIDEO_SEQUENCE_COUNT:
            raise ValueError(f"expected 26 eligible Replica sequences, got {len(eligible)}")
        video_schedule = deterministic_video_schedule(eligible, seed=args.seed)

    single_records: List[SingleImagePair] = []
    single_schedule: List[List[int]] = []
    dpdd_manifest_hash = None
    dpdd_dataset_provenance = None
    if mode in {"S", "M"}:
        dpdd_manifest_hash = sha256_file(args.dpdd_pairs)
        if dpdd_manifest_hash != args.dpdd_manifest_sha256.lower():
            raise ValueError("DPDD canonical manifest SHA256 mismatch")
        dpdd_dataset_provenance = load_dpdd_dataset_contract(
            args.dpdd_dataset_manifest,
            expected_dataset_manifest_sha256=args.dpdd_dataset_manifest_sha256,
            train_manifest=args.dpdd_pairs,
            expected_train_manifest_sha256=dpdd_manifest_hash,
        )
        single_records = load_single_image_manifest(
            args.dpdd_pairs,
            root=args.dpdd_root,
            expected_split="train",
            canonical_contract=True,
            verify_content=True,
        )
        if len(single_records) != FORMAL_DPDD_TRAIN_COUNT:
            raise ValueError(f"expected DPDD train350, got {len(single_records)}")
        single_schedule = deterministic_single_schedule(
            [record.name for record in single_records], seed=args.seed
        )

    model, base_metadata = load_turtle_model(
        args.turtle_repo,
        args.base_checkpoint,
        config=args.turtle_config,
        device=device,
    )
    if base_metadata.get("kind") != "official_gopro":
        raise ValueError("V/S/M must start from the pinned official GoPro checkpoint")
    scopes = configure_parameter_scopes(model, mode)
    model.train()
    groups = [{"params": scopes.spatial, "lr": 1e-5, "weight_decay": 1e-4}]
    if mode in {"V", "M"}:
        groups.insert(0, {"params": scopes.history, "lr": 1e-5, "weight_decay": 1e-3})
    optimizer = CountingAdamW(groups, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FORMAL_STEPS, eta_min=1e-7
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled,
        init_scale=1024.0,
        growth_interval=2000,
    )
    video_audits: List[Dict[str, Any]] = []
    single_audits: List[List[Dict[str, Any]]] = []

    for step in range(FORMAL_STEPS):
        single_batch = None
        if mode in {"S", "M"}:
            single_blur, single_sharp, audit = load_single_batch(
                single_records,
                single_schedule[step],
                seed=args.seed,
                step=step,
                crop_size=args.crop_size,
            )
            single_batch = (single_blur, single_sharp)
            single_audits.append(audit)
        video_batch = None
        if mode in {"V", "M"}:
            pass_index, sequence_index = video_schedule[step]
            video_blur, video_sharp, audit = load_video_sample(
                video_dataset,
                sequence_index=sequence_index,
                pass_index=pass_index,
            )
            video_batch = (video_blur, video_sharp)
            video_audits.append(audit)
        row = source_scoped_optimizer_step(
            model,
            scopes,
            optimizer,
            mode=mode,
            device=device,
            single_batch=single_batch,
            video_batch=video_batch,
            amp_enabled=amp_enabled,
            scaler=scaler,
            scheduler=scheduler,
        )
        print(json.dumps({"step": step + 1, "mode": mode, **row}), flush=True)

    if optimizer.step_count != FORMAL_STEPS or scheduler.last_epoch != FORMAL_STEPS:
        raise RuntimeError(
            "formal optimizer/scheduler execution count mismatch: "
            f"optimizer={optimizer.step_count}, scheduler={scheduler.last_epoch}"
        )

    metadata = {
        "format": FINETUNED_CHECKPOINT_FORMAT,
        "schema": TRAINING_SCHEMA,
        "base_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
        "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
        "input_domain": "raw",
        "cache_contract": TURTLE_CACHE_CONTRACT,
        "uses_paired_sharp_ground_truth_rgb": True,
        "uses_gt_pose": False,
        "uses_gt_depth": False,
        "mode": mode,
        "training": {
            "seed": args.seed,
            "optimizer_steps": FORMAL_STEPS,
            "attempted_optimizer_steps": FORMAL_STEPS,
            "executed_optimizer_steps": optimizer.step_count,
            "amp_skipped_optimizer_steps": 0,
            "mixed_step": "two_backward_one_joint_step" if mode == "M" else None,
            "mixed_dpdd_rng": (
                "torch_cpu_and_all_cuda_rng_saved_before_dpdd_and_restored_before_video"
                if mode == "M"
                else None
            ),
            "amp": amp_enabled,
            "grad_scaler": {
                "init_scale": 1024.0,
                "growth_interval": 2000,
                "growth_disabled_within_78_steps": True,
                "overflow_policy": "fail_closed_no_checkpoint",
            },
        },
        "manifests": {
            "video": str(args.video_manifest.resolve()) if args.video_manifest else None,
            "video_sha256": sha256_file(args.video_manifest) if args.video_manifest else None,
            "video_root": str(args.video_root.resolve()) if args.video_manifest else None,
            "dpdd_pairs": str(args.dpdd_pairs.resolve()) if args.dpdd_pairs else None,
            "dpdd_pairs_sha256": dpdd_manifest_hash,
            "dpdd_selected_split": "train" if args.dpdd_pairs else None,
            "dpdd_dataset": dpdd_dataset_provenance,
            "test_pixels_or_metrics_read": False,
        },
        "parameter_scopes": {
            "history_names": scopes.history_names,
            "history_parameters": HISTORY_ATTENTION_PARAMETER_COUNT,
            "spatial_names": scopes.spatial_names,
            "spatial_parameters": SPATIAL_HEAD_PARAMETER_COUNT,
            "dpdd_history_gradient": "forbidden_and_asserted_none",
        },
        "sampling_audit": {
            "video_order_seed": args.seed + 10_000,
            "video_transform_base_seed": args.seed + 20_000,
            "dpdd_order_seed": args.seed + 30_000,
            "dpdd_transform_base_seed": args.seed + 40_000,
            "video_schedule_sha256": _json_sha256(video_schedule),
            "dpdd_schedule_names_sha256": _json_sha256(
                [[single_records[index].name for index in batch] for batch in single_schedule]
            ),
            "video_transforms_sha256": _json_sha256(video_audits),
            "dpdd_transforms_sha256": _json_sha256(single_audits),
            "video_transforms": video_audits,
            "dpdd_transforms": single_audits,
        },
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.99],
            "scheduler": "CosineAnnealingLR_78_joint_steps",
            "history_group_clip": 1.0,
            "spatial_group_clip": 1.0,
        },
        "loss": {
            "l1_weight": 1.0,
            "fft_weight": 0.1,
            "video_temporal_delta_weight": 0.1,
            "video_order_rank_weight": 1.0,
            "video_order_margin": 0.0001,
            "dpdd_temporal_weight": 0.0,
        },
    }
    digest = save_checkpoint(
        output, model=model, metadata=metadata, overwrite=False
    )
    print(json.dumps({"checkpoint": str(output), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
