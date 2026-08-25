#!/usr/bin/env python3
"""Low-LR fine-tuning for the pinned, incremental TURTLE deblurrer.

Each JSONL record is an independent causal stream.  Frame zero is supplied as
both elements of TURTLE's two-frame input, then the eight official K/V slots
are carried only within that record.  The default full-record mode keeps the
cache graph attached until one sequence-level optimizer step; the legacy
one-frame mode explicitly detaches after every update.  Neither mode carries
state across a JSONL boundary.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


ARTIFACT_ROOT = Path(
    os.environ.get("UNBLUR_ARTIFACT_ROOT", "/srv/szha0669/unblur-slam")
).expanduser().resolve()
DEFAULT_TURTLE_REPO = Path(
    os.environ.get("UNBLUR_TURTLE_REPO", str(ARTIFACT_ROOT / "external/TURTLE"))
).expanduser().resolve()
DEFAULT_TURTLE_CONFIG = DEFAULT_TURTLE_REPO / "options/Turtle_Deblur_Gopro.yml"
DEFAULT_TURTLE_CHECKPOINT = Path(
    os.environ.get(
        "UNBLUR_TURTLE_GOPRO_CHECKPOINT",
        str(ARTIFACT_ROOT / "pretrained/turtle/GoPro_Deblur.pth"),
    )
).expanduser().resolve()

HISTORY_ATTENTION_PARAMETER_PREFIXES = (
    "latent.transformer_blocks.0.attn.",
    "latent.transformer_blocks.10.attn.",
    "decoder_level3.transformer_blocks.9.attn.",
    "decoder_level2.transformer_blocks.5.attn.",
    "decoder_level1.transformer_blocks.1.attn.",
)
HISTORY_ATTENTION_PARAMETER_TENSORS = 56
HISTORY_ATTENTION_PARAMETER_COUNT = 3_475_994


@dataclass(frozen=True)
class SequenceRecord:
    """One ordered, gap-free sequence from a paired JSONL manifest."""

    name: str
    blurry: Tuple[Path, ...]
    sharp: Tuple[Path, ...]
    teacher: Optional[Tuple[Path, ...]] = None


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError("missing one of: " + ", ".join(keys))


def _optional_first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_sequence_manifest(
    manifest: Path | str,
    *,
    root: Optional[Path | str] = None,
) -> List[SequenceRecord]:
    """Load ordered JSONL without ever joining records across a temporal gap."""

    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    data_root = (
        Path(root).expanduser().resolve() if root is not None else manifest_path.parent
    )
    records: List[SequenceRecord] = []
    names = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on {manifest_path}:{line_number}: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            name = str(payload.get("sequence", payload.get("name", f"line_{line_number}")))
            if name in names:
                raise ValueError(f"duplicate sequence name in manifest: {name!r}")
            names.add(name)

            if "frames" in payload:
                frames = payload["frames"]
                if not isinstance(frames, list) or not frames:
                    raise ValueError(f"sequence {name!r} has no frames")
                blurry_values: List[Any] = []
                sharp_values: List[Any] = []
                teacher_values: List[Any] = []
                teacher_complete = True
                for frame in frames:
                    if not isinstance(frame, dict):
                        raise ValueError(f"sequence {name!r} contains a non-object frame")
                    blurry_values.append(_first(frame, ("blurry", "blur", "input", "lq")))
                    sharp_values.append(_first(frame, ("sharp", "target", "gt")))
                    teacher = _optional_first(frame, ("teacher", "single_frame_teacher"))
                    if teacher is None:
                        teacher_complete = False
                    else:
                        teacher_values.append(teacher)
            else:
                blurry_value = _first(payload, ("blurry", "blur", "input", "lq"))
                sharp_value = _first(payload, ("sharp", "target", "gt"))
                if not isinstance(blurry_value, list) or not isinstance(sharp_value, list):
                    raise ValueError(f"sequence {name!r} paths must be arrays")
                blurry_values = list(blurry_value)
                sharp_values = list(sharp_value)
                teacher_value = _optional_first(
                    payload, ("teacher", "single_frame_teacher")
                )
                teacher_complete = teacher_value is not None
                if teacher_value is not None and not isinstance(teacher_value, list):
                    raise ValueError(f"sequence {name!r} teacher paths must be an array")
                teacher_values = list(teacher_value or [])

            if not blurry_values:
                raise ValueError(f"sequence {name!r} has no frames")
            if len(blurry_values) != len(sharp_values):
                raise ValueError(f"sequence {name!r} has mismatched blurry/sharp lengths")
            if teacher_complete and len(teacher_values) != len(blurry_values):
                raise ValueError(f"sequence {name!r} has incomplete teacher frames")
            if not teacher_complete and teacher_values:
                raise ValueError(f"sequence {name!r} has partially specified teacher frames")

            record = SequenceRecord(
                name=name,
                blurry=tuple(_resolve_path(value, data_root) for value in blurry_values),
                sharp=tuple(_resolve_path(value, data_root) for value in sharp_values),
                teacher=(
                    tuple(_resolve_path(value, data_root) for value in teacher_values)
                    if teacher_complete
                    else None
                ),
            )
            paths = record.blurry + record.sharp + (record.teacher or ())
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"missing frame in sequence {record.name!r}: {path}"
                    )
            records.append(record)
    if not records:
        raise ValueError(f"manifest contains no sequences: {manifest_path}")
    return records


@dataclass(frozen=True)
class _Transform:
    source_size: Tuple[int, int]
    crop_box: Tuple[int, int, int, int]
    flip_horizontal: bool
    flip_vertical: bool
    quarter_turns: int


def _image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _choose_transform(
    record: SequenceRecord,
    *,
    crop_size: int,
    augment: bool,
    rng: random.Random,
) -> _Transform:
    width, height = _image_size(record.blurry[0])
    if crop_size:
        if crop_size > width or crop_size > height:
            raise ValueError(
                f"crop_size={crop_size} exceeds {record.name!r} image size {width}x{height}"
            )
        left = rng.randrange(width - crop_size + 1) if augment else (width - crop_size) // 2
        top = rng.randrange(height - crop_size + 1) if augment else (height - crop_size) // 2
        crop_box = (left, top, left + crop_size, top + crop_size)
    else:
        crop_box = (0, 0, width, height)
    return _Transform(
        source_size=(width, height),
        crop_box=crop_box,
        flip_horizontal=bool(augment and rng.randrange(2)),
        flip_vertical=bool(augment and rng.randrange(2)),
        quarter_turns=(rng.randrange(4) if augment else 0),
    )


def _read_transformed(path: Path, transform: _Transform) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != transform.source_size:
            raise ValueError(
                f"sequence frames have inconsistent sizes: {path} is {image.size}, "
                f"expected {transform.source_size}"
            )
        image = image.crop(transform.crop_box)
        if transform.flip_horizontal:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if transform.flip_vertical:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rotations = (
            None,
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        )
        rotation = rotations[transform.quarter_turns]
        if rotation is not None:
            image = image.transpose(rotation)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def read_rgb_tensor(path: Path, *, device: Any = "cpu") -> torch.Tensor:
    """Read an RGB image as a CHW float tensor in [0, 1]."""

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


class PairedSequenceDataset:
    """Sequence-level loader with one shared crop/flip/rotation per record."""

    def __init__(
        self,
        manifest: Path | str,
        *,
        root: Optional[Path | str] = None,
        crop_size: int = 192,
        augment: bool = True,
        seed: int = 42,
    ):
        if crop_size < 0:
            raise ValueError("crop_size cannot be negative")
        if crop_size and crop_size % 8:
            raise ValueError("TURTLE crop_size must be divisible by 8")
        self.manifest = Path(manifest).expanduser().resolve()
        self.records = load_sequence_manifest(self.manifest, root=root)
        self.crop_size = int(crop_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        # This arithmetic seed is stable across Python processes and workers.
        rng = random.Random(self.seed + self.epoch * 1_000_003 + int(index) * 9_176)
        transform = _choose_transform(
            record, crop_size=self.crop_size, augment=self.augment, rng=rng
        )
        blurry = torch.stack([_read_transformed(path, transform) for path in record.blurry])
        sharp = torch.stack([_read_transformed(path, transform) for path in record.sharp])
        height, width = blurry.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError(
                f"TURTLE input must be divisible by 8, got {height}x{width}; "
                "configure --crop-size"
            )
        return {"sequence": record.name, "blurry": blurry, "sharp": sharp}


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fft_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """EVSSM's L1 loss over real and imaginary 2-D FFT components."""

    prediction_fft = torch.fft.fft2(prediction.float(), dim=(-2, -1))
    target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
    prediction_parts = torch.stack((prediction_fft.real, prediction_fft.imag), dim=-1)
    target_parts = torch.stack((target_fft.real, target_fft.imag), dim=-1)
    return F.l1_loss(prediction_parts, target_parts)


def _validate_forward_result(
    result: Any, current: torch.Tensor
) -> Tuple[torch.Tensor, Sequence[Any], Sequence[Any]]:
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise RuntimeError("TURTLE must return (restored, k_cache, v_cache)")
    restored, k_cache, v_cache = result
    if not torch.is_tensor(restored) or tuple(restored.shape) != tuple(current.shape):
        raise RuntimeError(
            f"TURTLE output shape mismatch: expected {tuple(current.shape)}, "
            f"got {getattr(restored, 'shape', None)}"
        )
    for label, cache in (("k_cache", k_cache), ("v_cache", v_cache)):
        if not isinstance(cache, (tuple, list)) or len(cache) != 8:
            raise RuntimeError(f"official TURTLE {label} must contain eight tensors")
        if any(value is not None and not torch.is_tensor(value) for value in cache):
            raise RuntimeError(f"official TURTLE {label} contains a non-tensor")
        mask = [value is not None for value in cache]
        if mask != [False, False, False, True, True, True, True, True]:
            raise RuntimeError(
                f"official TURTLE {label} population mask changed: {mask}"
            )
    return restored, k_cache, v_cache


def _detach_cache(cache: Sequence[Any]) -> List[Any]:
    return [None if value is None else value.detach() for value in cache]


def cyclically_shuffled_past_indices(anchor_index: int) -> Tuple[int, ...]:
    """Return a deterministic non-identity permutation of an anchor's past.

    ``anchor_index`` is also the number of strictly earlier frames.  The
    cyclic left shift is identical to the evaluator's shuffled-history arm,
    contains no current/future frame, and is defined only when temporal order
    can actually change (at least two past frames).
    """

    if anchor_index < 2:
        raise ValueError("order contrast requires an anchor with two past frames")
    ordered = tuple(range(int(anchor_index)))
    return ordered[1:] + ordered[:1]


def _replay_current_from_past(
    model: nn.Module,
    blurry: torch.Tensor,
    *,
    anchor_index: int,
    past_indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Replay an explicitly listed past and predict one current frame.

    The function deliberately validates that ``past_indices`` is a
    permutation of *all and only* ``[0, anchor_index)``.  This makes future
    leakage and dropped/duplicated history a fail-closed error.  K/V tensors
    remain attached, so the counterfactual order loss backpropagates through
    the complete replayed prefix.
    """

    frame_count = int(blurry.shape[0])
    if blurry.ndim != 4 or blurry.shape[1] != 3:
        raise ValueError("TURTLE replay input must be a TCHW RGB tensor")
    if anchor_index < 2 or anchor_index >= frame_count:
        raise ValueError("order-contrast anchor must have two past frames and a current frame")
    normalized = tuple(int(index) for index in past_indices)
    if len(normalized) != anchor_index or sorted(normalized) != list(range(anchor_index)):
        raise ValueError("replayed history must contain every past frame exactly once")

    k_cache: Optional[Sequence[Any]] = None
    v_cache: Optional[Sequence[Any]] = None
    previous: Optional[torch.Tensor] = None
    for frame_index in normalized:
        current = blurry[frame_index : frame_index + 1].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        prior = current if previous is None else previous
        pair = torch.stack((prior[0], current[0]), dim=0).unsqueeze(0)
        _, k_cache, v_cache = _validate_forward_result(
            model(pair, k_cache, v_cache), current
        )
        previous = current

    current = blurry[anchor_index : anchor_index + 1].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    prior = current if previous is None else previous
    pair = torch.stack((prior[0], current[0]), dim=0).unsqueeze(0)
    prediction, _, _ = _validate_forward_result(
        model(pair, k_cache, v_cache), current
    )
    return prediction


def configure_trainable_scope(model: nn.Module, scope: str) -> Dict[str, Any]:
    """Select either the complete official model or its five history attentions.

    The history-only scope is intentionally fail-closed against the pinned
    TURTLE architecture.  It prevents a Replica smoke from silently becoming
    another unrestricted single-image spatial fine-tune.
    """

    normalized = str(scope).strip().lower()
    if normalized not in {"all", "history_attention"}:
        raise ValueError("trainable scope must be all or history_attention")
    model.requires_grad_(normalized == "all")
    if normalized == "history_attention":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                any(name.startswith(prefix) for prefix in HISTORY_ATTENTION_PARAMETER_PREFIXES)
            )
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    tensor_count = len(selected)
    parameter_count = sum(parameter.numel() for _, parameter in selected)
    if normalized == "history_attention" and (
        tensor_count != HISTORY_ATTENTION_PARAMETER_TENSORS
        or parameter_count != HISTORY_ATTENTION_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "pinned TURTLE history-attention parameter contract changed: "
            f"tensors={tensor_count}, parameters={parameter_count}"
        )
    if not selected:
        raise RuntimeError("training scope selected no TURTLE parameters")
    return {
        "scope": normalized,
        "parameter_tensors": tensor_count,
        "parameter_count": parameter_count,
        "parameter_prefixes": (
            list(HISTORY_ATTENTION_PARAMETER_PREFIXES)
            if normalized == "history_attention"
            else None
        ),
    }


def train_sequence_full_bptt(
    model: nn.Module,
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    fft_weight: float = 0.1,
    grad_clip: float = 1.0,
    amp_enabled: bool = False,
    scaler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    loss_start_frame: int = 1,
    temporal_delta_weight: float = 0.0,
    order_contrast_weight: float = 0.0,
    order_contrast_margin: float = 0.0,
    order_anchor_index: Optional[int] = None,
) -> Dict[str, float]:
    """Run one official causal stream and update once after the whole sequence.

    K/V tensors remain attached until the sequence-level loss is backpropagated.
    This preserves the central upstream causal-training property (later-frame
    loss can assign credit to parameters that produced earlier cache state),
    while this smoke deliberately uses each real JSONL record at its natural
    length and supervises only frames selected by ``loss_start_frame``.

    The optional v2 objectives remain strictly causal. ``temporal_delta``
    matches adjacent output/GT changes in the ordered stream. ``order_contrast``
    compares the final (or explicitly selected) current frame against a second
    cache built from the same complete past multiset in cyclically shifted
    order.  No current or future frame is inserted into that cache.
    """

    if blurry.ndim != 4 or sharp.shape != blurry.shape or blurry.shape[1] != 3:
        raise ValueError("blurry/sharp sequences must be matching TCHW RGB tensors")
    frame_count = int(blurry.shape[0])
    if frame_count < 1:
        raise ValueError("training sequence is empty")
    if loss_start_frame < 0 or loss_start_frame >= frame_count:
        raise ValueError("loss_start_frame must select at least one sequence frame")
    if amp_enabled and scaler is None:
        raise ValueError("AMP training requires a GradScaler")
    if temporal_delta_weight < 0 or order_contrast_weight < 0:
        raise ValueError("temporal objective weights cannot be negative")
    if order_contrast_margin < 0:
        raise ValueError("order contrast margin cannot be negative")
    if order_contrast_weight > 0 and frame_count < 3:
        raise ValueError("order contrast requires a sequence of at least three frames")

    optimizer.zero_grad(set_to_none=True)
    k_cache: Optional[Sequence[Any]] = None
    v_cache: Optional[Sequence[Any]] = None
    previous: Optional[torch.Tensor] = None
    l1_terms: List[torch.Tensor] = []
    fft_terms: List[torch.Tensor] = []
    temporal_delta_terms: List[torch.Tensor] = []
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    autocast = torch.cuda.amp.autocast(enabled=True) if amp_enabled else nullcontext()
    with autocast:
        for frame_index in range(frame_count):
            current = blurry[frame_index : frame_index + 1].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            target = sharp[frame_index : frame_index + 1].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            prior = current if previous is None else previous
            pair = torch.stack((prior[0], current[0]), dim=0).unsqueeze(0)
            prediction, k_cache, v_cache = _validate_forward_result(
                model(pair, k_cache, v_cache), current
            )
            predictions.append(prediction)
            targets.append(target)
            if frame_index >= loss_start_frame:
                l1_terms.append(F.l1_loss(prediction, target))
                fft_terms.append(fft_l1_loss(prediction, target))
                if frame_index > 0:
                    temporal_delta_terms.append(
                        F.l1_loss(
                            prediction - predictions[frame_index - 1],
                            target - targets[frame_index - 1],
                        )
                    )
            previous = current
        l1 = torch.stack(l1_terms).mean()
        fft = torch.stack(fft_terms).mean()
        temporal_delta = (
            torch.stack(temporal_delta_terms).mean()
            if temporal_delta_terms
            else l1.new_zeros(())
        )

        ordered_anchor_error = l1.new_zeros(())
        shuffled_anchor_error = l1.new_zeros(())
        order_gap = l1.new_zeros(())
        order_rank = l1.new_zeros(())
        selected_anchor = -1
        if order_contrast_weight > 0:
            selected_anchor = (
                frame_count - 1
                if order_anchor_index is None
                else int(order_anchor_index)
            )
            if selected_anchor < max(2, loss_start_frame) or selected_anchor >= frame_count:
                raise ValueError(
                    "order anchor must be a supervised frame with at least two past frames"
                )
            shuffled_prediction = _replay_current_from_past(
                model,
                blurry,
                anchor_index=selected_anchor,
                past_indices=cyclically_shuffled_past_indices(selected_anchor),
                device=device,
            )
            ordered_anchor_error = F.l1_loss(
                predictions[selected_anchor], targets[selected_anchor]
            )
            shuffled_anchor_error = F.l1_loss(
                shuffled_prediction, targets[selected_anchor]
            )
            order_gap = shuffled_anchor_error - ordered_anchor_error
            order_rank = F.relu(float(order_contrast_margin) - order_gap)

        loss = (
            l1
            + float(fft_weight) * fft
            + float(temporal_delta_weight) * temporal_delta
            + float(order_contrast_weight) * order_rank
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if scaler is not None and amp_enabled:
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return {
        "frames": frame_count,
        "supervised_frames": len(l1_terms),
        "loss": float(loss.detach().item()),
        "l1": float(l1.detach().item()),
        "fft": float(fft.detach().item()),
        "temporal_delta_l1": float(temporal_delta.detach().item()),
        "order_anchor_index": selected_anchor,
        "ordered_anchor_l1": float(ordered_anchor_error.detach().item()),
        "shuffled_anchor_l1": float(shuffled_anchor_error.detach().item()),
        "shuffled_minus_ordered_anchor_l1": float(order_gap.detach().item()),
        "order_rank_hinge": float(order_rank.detach().item()),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def train_sequence(
    model: nn.Module,
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    step_budget: int,
    fft_weight: float = 0.1,
    grad_clip: float = 1.0,
    amp_enabled: bool = False,
    scaler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> List[Dict[str, float]]:
    """Optimize at most ``step_budget`` ordered frames from one sequence."""

    if blurry.ndim != 4 or sharp.shape != blurry.shape or blurry.shape[1] != 3:
        raise ValueError("blurry/sharp sequences must be matching TCHW RGB tensors")
    if step_budget < 0:
        raise ValueError("step_budget cannot be negative")
    if amp_enabled and scaler is None:
        raise ValueError("AMP training requires a GradScaler")

    k_cache: Optional[Sequence[Any]] = None
    v_cache: Optional[Sequence[Any]] = None
    previous: Optional[torch.Tensor] = None
    rows: List[Dict[str, float]] = []
    for frame_index in range(min(int(blurry.shape[0]), int(step_budget))):
        current = blurry[frame_index : frame_index + 1].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        target = sharp[frame_index : frame_index + 1].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        prior = current if previous is None else previous
        pair = torch.stack((prior[0], current[0]), dim=0).unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.cuda.amp.autocast(enabled=True) if amp_enabled else nullcontext()
        )
        with autocast:
            prediction, new_k, new_v = _validate_forward_result(
                model(pair, k_cache, v_cache), current
            )
            l1 = F.l1_loss(prediction, target)
            fft = fft_l1_loss(prediction, target)
            loss = l1 + float(fft_weight) * fft

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Truncated causal training: retain values but never the old graph.
        k_cache = _detach_cache(new_k)
        v_cache = _detach_cache(new_v)
        previous = current.detach()
        rows.append(
            {
                "frame_index": frame_index,
                "loss": float(loss.detach().item()),
                "l1": float(l1.detach().item()),
                "fft": float(fft.detach().item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
    return rows


@torch.no_grad()
def validate_model(
    model: nn.Module,
    dataset: PairedSequenceDataset,
    device: torch.device,
) -> float:
    """PSNR over every frame, resetting causal state for each JSONL record."""

    was_training = model.training
    model.eval()
    squared_error = 0.0
    pixel_count = 0
    for sequence_index in range(len(dataset)):
        sample = dataset[sequence_index]
        blurry = sample["blurry"]
        sharp = sample["sharp"]
        k_cache: Optional[Sequence[Any]] = None
        v_cache: Optional[Sequence[Any]] = None
        previous: Optional[torch.Tensor] = None
        for frame_index in range(int(blurry.shape[0])):
            current = blurry[frame_index : frame_index + 1].to(device=device)
            target = sharp[frame_index : frame_index + 1].to(device=device)
            prior = current if previous is None else previous
            pair = torch.stack((prior[0], current[0]), dim=0).unsqueeze(0)
            prediction, k_cache, v_cache = _validate_forward_result(
                model(pair, k_cache, v_cache), current
            )
            squared_error += float(F.mse_loss(prediction, target, reduction="sum").item())
            pixel_count += target.numel()
            k_cache = _detach_cache(k_cache)
            v_cache = _detach_cache(v_cache)
            previous = current
    if was_training:
        model.train()
    mse = squared_error / max(1, pixel_count)
    return -10.0 * math.log10(max(mse, 1.0e-12))


def build_checkpoint_metadata(
    *,
    base_metadata: Mapping[str, Any],
    train_manifest: Path,
    val_manifest: Optional[Path],
    seed: int,
    steps: int,
    crop_size: int,
    augment: bool,
    fft_weight: float,
    learning_rate: float,
    weight_decay: float,
    betas: Tuple[float, float],
    amp: bool,
    best_val_psnr: Optional[float],
    grad_clip: float = 1.0,
    scheduler_eta_min: float = 1.0e-7,
    bptt_mode: str = "one_frame_truncated",
    trainable_scope: Optional[Mapping[str, Any]] = None,
    minimum_sequence_length: int = 1,
    loss_start_frame: int = 0,
    temporal_delta_weight: float = 0.0,
    order_contrast_weight: float = 0.0,
    order_contrast_margin: float = 0.0,
    order_anchor_policy: str = "disabled",
) -> Dict[str, Any]:
    """Build metadata accepted by the strict fine-tuned checkpoint loader."""

    if base_metadata.get("base_checkpoint_sha256") != PINNED_TURTLE_CHECKPOINT_SHA256:
        raise ValueError("fine-tuning base is not the pinned TURTLE GoPro checkpoint")
    train_manifest = train_manifest.expanduser().resolve()
    val_manifest = val_manifest.expanduser().resolve() if val_manifest else None
    return {
        "format": FINETUNED_CHECKPOINT_FORMAT,
        "base_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
        "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
        "input_domain": "raw",
        "cache_contract": TURTLE_CACHE_CONTRACT,
        # `uses_gt` refers to SLAM pose/depth leakage. Paired sharp RGB is the
        # explicit restoration target and is recorded separately below.
        "uses_gt": False,
        "uses_gt_pose": False,
        "uses_gt_depth": False,
        "uses_sharp_rgb_supervision": True,
        "manifests": {
            "train": str(train_manifest),
            "train_sha256": sha256_file(train_manifest),
            "validation": str(val_manifest) if val_manifest else None,
            "validation_sha256": sha256_file(val_manifest) if val_manifest else None,
        },
        "loss": {
            "name": "l1_plus_fft_l1",
            "l1_weight": 1.0,
            "fft_weight": float(fft_weight),
            "temporal_order_v2": {
                "temporal_delta_l1_weight": float(temporal_delta_weight),
                "ordered_vs_shuffled_rank_weight": float(order_contrast_weight),
                "ordered_vs_shuffled_l1_margin": float(order_contrast_margin),
                "anchor_policy": str(order_anchor_policy),
                "shuffle_policy": "cyclic_left_shift_complete_past_prefix",
                "counterfactual_past_multiset_preserved": True,
                "future_frames_used": False,
            },
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "betas": [float(betas[0]), float(betas[1])],
            "scheduler": "CosineAnnealingLR",
            "scheduler_eta_min": float(scheduler_eta_min),
            "scheduler_t_max_optimizer_steps": int(steps),
            "gradient_clip_norm": float(grad_clip),
        },
        "history": {
            "mode": "official_incremental_kv",
            "cache_tensors_per_kind": 8,
            "pair_context_frames": 2,
            "num_frames_tocache": 3,
            "sequence_boundary": "hard_reset",
            "startup": "repeat_first_frame_on_left",
            "backpropagation": str(bptt_mode),
        },
        "trainable_scope": dict(trainable_scope or {"scope": "all"}),
        "sequence_filter": {
            "minimum_length": int(minimum_sequence_length),
            "loss_start_frame": int(loss_start_frame),
        },
        "augmentation": {
            "enabled": bool(augment),
            "shared_across_sequence_and_modalities": True,
            "crop_size": int(crop_size),
            "horizontal_flip": bool(augment),
            "vertical_flip": bool(augment),
            "quarter_turn_rotation": bool(augment),
            "sampling_policy": (
                "shared_per_record; hflip_p=0.5; vflip_p=0.5; "
                "quarter_turn_uniform_0_1_2_3"
                if augment
                else "deterministic_center_crop_no_flip_no_rotation"
            ),
        },
        "training": {
            "seed": int(seed),
            "steps": int(steps),
            "amp": bool(amp),
            "best_validation_psnr": (
                None if best_val_psnr is None else float(best_val_psnr)
            ),
        },
    }


def save_checkpoint(
    output: Path,
    *,
    model: nn.Module,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> str:
    """Atomically save params+metadata and return the content SHA-256."""

    output = output.expanduser().resolve()
    digest_path = output.with_name(output.name + ".sha256")
    if not overwrite and (output.exists() or digest_path.exists()):
        raise FileExistsError(f"refusing to overwrite checkpoint artifacts: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    bare = model.module if hasattr(model, "module") else model
    payload = {"params": bare.state_dict(), "metadata": dict(metadata)}
    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256_file(output)
    digest_temporary = digest_path.with_name(digest_path.name + ".tmp")
    digest_temporary.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    os.replace(digest_temporary, digest_path)
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--turtle-repo", type=Path, default=DEFAULT_TURTLE_REPO)
    parser.add_argument("--turtle-config", type=Path, default=DEFAULT_TURTLE_CONFIG)
    parser.add_argument(
        "--base-checkpoint", type=Path, default=DEFAULT_TURTLE_CHECKPOINT
    )
    parser.add_argument("--base-checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--fft-weight", type=float, default=0.1)
    parser.add_argument("--temporal-delta-weight", type=float, default=0.0)
    parser.add_argument("--order-contrast-weight", type=float, default=0.0)
    parser.add_argument("--order-contrast-margin", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eta-min", type=float, default=1.0e-7)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument(
        "--bptt-mode",
        choices=("full_sequence", "one_frame_truncated"),
        default="full_sequence",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=("history_attention", "all"),
        default="history_attention",
    )
    parser.add_argument("--min-sequence-length", type=int, default=2)
    parser.add_argument("--loss-start-frame", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.crop_size < 0 or (args.crop_size and args.crop_size % 8):
        raise ValueError("--crop-size must be zero or a positive multiple of 8")
    if (
        args.lr <= 0
        or args.weight_decay < 0
        or args.fft_weight < 0
        or args.temporal_delta_weight < 0
        or args.order_contrast_weight < 0
        or args.order_contrast_margin < 0
        or args.grad_clip < 0
        or args.eta_min < 0
    ):
        raise ValueError("learning rate must be positive and loss/decay non-negative")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        raise ValueError("AdamW beta values must be in [0, 1)")
    if args.log_every < 1 or args.val_every < 0:
        raise ValueError("--log-every must be positive and --val-every non-negative")
    if args.min_sequence_length < 1:
        raise ValueError("--min-sequence-length must be positive")
    if args.loss_start_frame < 0:
        raise ValueError("--loss-start-frame cannot be negative")
    if args.bptt_mode == "one_frame_truncated" and args.loss_start_frame != 0:
        raise ValueError("one-frame truncated mode requires --loss-start-frame 0")
    if args.bptt_mode != "full_sequence" and (
        args.temporal_delta_weight > 0 or args.order_contrast_weight > 0
    ):
        raise ValueError("temporal v2 objectives require --bptt-mode full_sequence")
    if args.order_contrast_weight > 0 and args.min_sequence_length < 3:
        raise ValueError("order contrast requires --min-sequence-length at least 3")

    set_seed(args.seed)
    device = choose_device(args.device)
    output = args.output.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    if output == base_checkpoint:
        raise ValueError("--output may not replace the immutable GoPro base checkpoint")
    digest_output = output.with_name(output.name + ".sha256")
    if output.exists() and output.is_dir():
        raise FileExistsError(f"checkpoint output is a directory: {output}")
    if not args.overwrite and (output.exists() or digest_output.exists()):
        raise FileExistsError(f"refusing to overwrite checkpoint artifacts: {output}")

    train_dataset = PairedSequenceDataset(
        args.train_manifest,
        root=args.data_root,
        crop_size=args.crop_size,
        augment=not args.no_augment,
        seed=args.seed,
    )
    val_dataset = (
        PairedSequenceDataset(
            args.val_manifest,
            root=args.data_root,
            crop_size=args.crop_size,
            augment=False,
            seed=args.seed,
        )
        if args.val_manifest is not None
        else None
    )
    model, base_metadata = load_turtle_model(
        args.turtle_repo,
        base_checkpoint,
        config=args.turtle_config,
        device=device,
        checkpoint_sha256=args.base_checkpoint_sha256,
    )
    if base_metadata.get("kind") != "official_gopro":
        raise ValueError("training must start from the pinned official GoPro checkpoint")
    trainable_scope = configure_trainable_scope(model, args.trainable_scope)
    model.train()

    betas = (float(args.beta1), float(args.beta2))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=betas,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.eta_min
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    global_step = 0
    epoch = 0
    best_val_psnr: Optional[float] = None
    last_val_psnr: Optional[float] = None
    last_val_step: Optional[int] = None
    next_val_step = args.val_every if args.val_every > 0 else None
    eligible_indices = [
        index
        for index, record in enumerate(train_dataset.records)
        if len(record.blurry) >= args.min_sequence_length
    ]
    if not eligible_indices:
        raise ValueError("sequence filter removed every training record")
    eligible_frames = sum(
        len(train_dataset.records[index].blurry) for index in eligible_indices
    )
    print(
        json.dumps(
            {
                "training_contract": {
                    "bptt_mode": args.bptt_mode,
                    "trainable_scope": trainable_scope,
                    "eligible_sequences": len(eligible_indices),
                    "eligible_frames": eligible_frames,
                    "minimum_sequence_length": args.min_sequence_length,
                    "loss_start_frame": args.loss_start_frame,
                    "temporal_delta_weight": args.temporal_delta_weight,
                    "order_contrast_weight": args.order_contrast_weight,
                    "order_contrast_margin": args.order_contrast_margin,
                    "order_anchor_policy": "last_supervised_frame",
                }
            }
        ),
        flush=True,
    )
    while global_step < args.max_steps:
        train_dataset.set_epoch(epoch)
        order = list(eligible_indices)
        random.Random(args.seed + epoch).shuffle(order)
        for sequence_index in order:
            if global_step >= args.max_steps:
                break
            sample = train_dataset[sequence_index]
            if args.bptt_mode == "full_sequence":
                rows = [
                    train_sequence_full_bptt(
                        model,
                        sample["blurry"],
                        sample["sharp"],
                        optimizer,
                        device=device,
                        fft_weight=args.fft_weight,
                        grad_clip=args.grad_clip,
                        amp_enabled=amp_enabled,
                        scaler=scaler,
                        scheduler=scheduler,
                        loss_start_frame=args.loss_start_frame,
                        temporal_delta_weight=args.temporal_delta_weight,
                        order_contrast_weight=args.order_contrast_weight,
                        order_contrast_margin=args.order_contrast_margin,
                    )
                ]
            else:
                rows = train_sequence(
                    model,
                    sample["blurry"],
                    sample["sharp"],
                    optimizer,
                    device=device,
                    step_budget=args.max_steps - global_step,
                    fft_weight=args.fft_weight,
                    grad_clip=args.grad_clip,
                    amp_enabled=amp_enabled,
                    scaler=scaler,
                    scheduler=scheduler,
                )
            for row in rows:
                global_step += 1
                if global_step == 1 or global_step % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "sequence": sample["sequence"],
                                **row,
                            }
                        ),
                        flush=True,
                    )
            # Validate only between JSONL streams. This preserves the exact
            # training K/V state inside a stream and reports the actual model
            # step, even when a short sequence crosses the requested interval.
            if (
                val_dataset is not None
                and next_val_step is not None
                and global_step >= next_val_step
            ):
                last_val_psnr = validate_model(model, val_dataset, device)
                last_val_step = global_step
                best_val_psnr = (
                    last_val_psnr
                    if best_val_psnr is None
                    else max(best_val_psnr, last_val_psnr)
                )
                print(
                    json.dumps(
                        {"step": global_step, "validation_psnr": last_val_psnr}
                    ),
                    flush=True,
                )
                while next_val_step <= global_step:
                    next_val_step += args.val_every
        epoch += 1

    if val_dataset is not None and last_val_step != global_step:
        last_val_psnr = validate_model(model, val_dataset, device)
        last_val_step = global_step
        best_val_psnr = (
            last_val_psnr
            if best_val_psnr is None
            else max(best_val_psnr, last_val_psnr)
        )

    metadata = build_checkpoint_metadata(
        base_metadata=base_metadata,
        train_manifest=train_dataset.manifest,
        val_manifest=(val_dataset.manifest if val_dataset is not None else None),
        seed=args.seed,
        steps=global_step,
        crop_size=args.crop_size,
        augment=not args.no_augment,
        fft_weight=args.fft_weight,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        betas=betas,
        amp=amp_enabled,
        best_val_psnr=best_val_psnr,
        grad_clip=args.grad_clip,
        scheduler_eta_min=args.eta_min,
        bptt_mode=args.bptt_mode,
        trainable_scope=trainable_scope,
        minimum_sequence_length=args.min_sequence_length,
        loss_start_frame=args.loss_start_frame,
        temporal_delta_weight=args.temporal_delta_weight,
        order_contrast_weight=args.order_contrast_weight,
        order_contrast_margin=args.order_contrast_margin,
        order_anchor_policy=(
            "last_supervised_frame"
            if args.order_contrast_weight > 0
            else "disabled"
        ),
    )
    checkpoint_sha256 = save_checkpoint(
        output, model=model, metadata=metadata, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "checkpoint_sha256": checkpoint_sha256,
                "steps": global_step,
                "last_validation_psnr": last_val_psnr,
                "best_validation_psnr": best_val_psnr,
                "device": str(device),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
