"""A small, genuinely causal temporal-attention deblurring network.

The public inference contract intentionally matches ``CausalTorchScriptBackend``:

    blurry frames [B, T, 3, H, W] -> deblurred last frame [B, 3, H, W]

``forward_sequence`` is exported as well.  It is useful for training and for
verifying causality: output at time ``t`` is invariant to every input after
``t``.  Temporal attention is applied at one-quarter resolution and uses a
strict upper-triangular mask, so no future key/value is visible.

For the causal-EVSSM configuration the input is the frozen EVSSM result.  The
network predicts only a bounded correction to that input.  The correction
head is zero initialized, so a newly constructed model is exactly EVSSM (not
merely close to EVSSM) while still receiving a useful gradient on its first
optimization step.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .alignment import CoarseLocalCorrelationAligner


DEFAULT_MAX_RESIDUAL = 8.0 / 255.0


def _group_count(channels: int) -> int:
    """Return a small GroupNorm group count that divides ``channels``."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = F.gelu(self.norm1(image))
        residual = self.conv1(residual)
        residual = F.gelu(self.norm2(residual))
        residual = self.conv2(residual)
        return image + residual


class CausalTemporalAttention(nn.Module):
    """Multi-head attention over time at each spatial feature location."""

    def __init__(self, channels: int, num_heads: int, max_history: int):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = int(channels // num_heads)
        self.max_history = int(max_history)
        self.scale = float(self.head_dim) ** -0.5

        self.position = nn.Parameter(torch.zeros(max_history, channels))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: B,T,C,H,W.  Spatial positions are independent attention
        # batches; only the temporal axis participates in attention.
        batch, timesteps, channels, height, width = features.shape
        if channels != self.channels:
            raise ValueError("temporal attention channel mismatch")
        if timesteps > self.max_history:
            raise ValueError("input history exceeds configured max_history")

        tokens = features.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, timesteps, channels
        )
        tokens = tokens + self.position[:timesteps].unsqueeze(0)

        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized).reshape(
            batch * height * width,
            timesteps,
            3,
            self.num_heads,
            self.head_dim,
        )
        queries = qkv[:, :, 0].permute(0, 2, 1, 3)
        keys = qkv[:, :, 1].permute(0, 2, 1, 3)
        values = qkv[:, :, 2].permute(0, 2, 1, 3)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        future_mask = torch.ones(
            (timesteps, timesteps), dtype=torch.bool, device=features.device
        ).triu(1)
        scores = scores.masked_fill(future_mask.unsqueeze(0).unsqueeze(0), -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, values)
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch * height * width, timesteps, channels
        )

        tokens = tokens + self.proj(attended)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.reshape(batch, height, width, timesteps, channels).permute(
            0, 3, 4, 1, 2
        )


class CausalVideoDeblur(nn.Module):
    """U-Net-like video deblurrer with causal bottleneck attention.

    ``teacher_frames`` is optional.  When ``use_teacher_input`` is enabled,
    cached outputs from a frozen single-frame model can be supplied during
    training.  A zero-initialized gate makes that path opt-in and preserves a
    raw-video-only initialization.  Knowledge distillation itself is handled
    by the training script and does not change the deployment signature.
    """

    def __init__(
        self,
        channels: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_history: int = 5,
        use_teacher_input: bool = False,
        input_domain: str = "raw",
        max_residual: float = DEFAULT_MAX_RESIDUAL,
    ):
        super().__init__()
        if channels < 4:
            raise ValueError("channels must be at least 4")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.num_blocks = int(num_blocks)
        self.max_history = int(max_history)
        self.use_teacher_input = bool(use_teacher_input)
        self.input_domain = str(input_domain).lower()
        if self.input_domain not in {"raw", "evssm"}:
            raise ValueError("input_domain must be raw or evssm")
        self.max_residual = float(max_residual)
        if not math.isfinite(self.max_residual):
            raise ValueError("max_residual must be finite")
        if self.max_residual < 0.0 or self.max_residual > 1.0:
            raise ValueError("max_residual must be in [0, 1]")

        c1, c2, c3 = channels, channels * 2, channels * 4
        self.stem = nn.Conv2d(3, c1, 3, padding=1)
        self.teacher_stem = nn.Conv2d(3, c1, 3, padding=1)
        self.teacher_gate = nn.Parameter(torch.zeros(()))
        self.encoder1 = nn.Sequential(*[ResidualBlock(c1) for _ in range(num_blocks)])
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.encoder2 = nn.Sequential(*[ResidualBlock(c2) for _ in range(num_blocks)])
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.encoder3 = nn.Sequential(*[ResidualBlock(c3) for _ in range(num_blocks)])

        self.temporal = CausalTemporalAttention(c3, num_heads, max_history)

        self.up2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.decoder2 = nn.Sequential(*[ResidualBlock(c2) for _ in range(num_blocks)])
        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.decoder1 = nn.Sequential(*[ResidualBlock(c1) for _ in range(num_blocks)])
        self.output = nn.Conv2d(c1, 3, 3, padding=1)

        # Begin as an identity residual predictor.  This makes training stable
        # when initializing the temporal adapter from single-frame outputs.
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def _validate_inputs(
        self, frames: torch.Tensor, teacher_frames: Optional[torch.Tensor]
    ) -> None:
        if frames.dim() != 5:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        if frames.shape[2] != 3:
            raise ValueError("frames must contain three RGB channels")
        if frames.shape[1] < 1:
            raise ValueError("frames must contain at least one timestep")
        if frames.shape[1] > self.max_history:
            raise ValueError("frames exceed configured max_history")
        if teacher_frames is not None:
            if teacher_frames.shape != frames.shape:
                raise ValueError("teacher_frames must match frames exactly")

    @torch.jit.export
    def forward_sequence(
        self,
        frames: torch.Tensor,
        teacher_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return one deblurred image per input timestep as ``[B,T,3,H,W]``."""
        self._validate_inputs(frames, teacher_frames)
        batch, timesteps, _, height, width = frames.shape
        flat_frames = frames.reshape(batch * timesteps, 3, height, width)

        level1 = self.stem(flat_frames)
        if self.use_teacher_input and teacher_frames is not None:
            flat_teacher = teacher_frames.reshape(batch * timesteps, 3, height, width)
            teacher_features = self.teacher_stem(flat_teacher)
            level1 = level1 + torch.tanh(self.teacher_gate) * teacher_features
        level1 = self.encoder1(level1)
        level2 = self.encoder2(self.down1(level1))
        level3 = self.encoder3(self.down2(level2))

        low_height, low_width = level3.shape[-2], level3.shape[-1]
        temporal = level3.reshape(
            batch, timesteps, level3.shape[1], low_height, low_width
        )
        temporal = self.temporal(temporal)
        decoded = temporal.reshape(batch * timesteps, level3.shape[1], low_height, low_width)

        decoded = F.interpolate(
            decoded, size=(level2.shape[-2], level2.shape[-1]), mode="bilinear", align_corners=False
        )
        decoded = self.decoder2(self.up2(decoded) + level2)
        decoded = F.interpolate(
            decoded, size=(level1.shape[-2], level1.shape[-1]), mode="bilinear", align_corners=False
        )
        decoded = self.decoder1(self.up1(decoded) + level1)
        residual_logits = self.output(decoded)
        if self.max_residual > 0.0:
            # This is a per-channel, per-pixel hard bound around the input
            # frame.  In the EVSSM domain, the input frame is precisely the
            # frozen single-frame EVSSM result.
            residual = self.max_residual * torch.tanh(residual_logits)
        else:
            # Compatibility path for v1 checkpoints whose model_config did
            # not contain max_residual.  Keeping the old unbounded formula is
            # essential: silently applying a new scale to old weights would
            # change their predictions.
            residual = residual_logits
        restored = flat_frames + residual
        return restored.reshape(batch, timesteps, 3, height, width)

    def forward(
        self,
        frames: torch.Tensor,
        teacher_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return only the newest deblurred frame as ``[B,3,H,W]``."""
        return self.forward_sequence(frames, teacher_frames)[:, -1]

    @torch.jit.unused
    def config_dict(self) -> Dict[str, Any]:
        return {
            "channels": self.channels,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "max_history": self.max_history,
            "use_teacher_input": self.use_teacher_input,
            "input_domain": self.input_domain,
            "max_residual": self.max_residual,
        }


class MotionAlignedCausalVideoDeblurV4(CausalVideoDeblur):
    """V4 adapter with recursively motion-aligned causal bottleneck memory.

    The inherited v3 modules retain their exact names and shapes.  Motion is
    an additive, zero-gated branch: a v3 state dict can therefore be copied
    without renaming any base key, and a freshly initialized v4 branch returns
    exactly the v3 temporal bottleneck while the gate is zero.

    Alignment is estimated only for adjacent original encoder features.  At
    timestep ``t`` the current-to-previous flow from ``t`` to ``t-1`` warps
    every accumulated memory feature from the ``t-1`` coordinate system into
    the current coordinate system.  This recursive construction is strictly
    causal and needs only ``T-1`` flow estimates.
    """

    __constants__ = [
        "motion_alignment_mode",
        "motion_match_channels",
        "motion_radius",
        "motion_temperature",
    ]

    def __init__(
        self,
        channels: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_history: int = 5,
        use_teacher_input: bool = False,
        input_domain: str = "raw",
        max_residual: float = DEFAULT_MAX_RESIDUAL,
        motion_match_channels: int = 16,
        motion_radius: int = 8,
        motion_temperature: float = 0.05,
    ) -> None:
        super().__init__(
            channels=channels,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_history=max_history,
            use_teacher_input=use_teacher_input,
            input_domain=input_domain,
            max_residual=max_residual,
        )
        self.motion_alignment_mode = "coarse_local_correlation_v1"
        self.motion_match_channels = int(motion_match_channels)
        self.motion_radius = int(motion_radius)
        self.motion_temperature = float(motion_temperature)
        self.motion_alignment_gate = nn.Parameter(torch.zeros(()))
        self.motion_aligner = CoarseLocalCorrelationAligner(
            input_channels=channels * 4,
            match_channels=self.motion_match_channels,
            radius=self.motion_radius,
            temperature=self.motion_temperature,
        )

    def _encode_sequence(
        self,
        frames: torch.Tensor,
        teacher_frames: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = frames.size(0)
        timesteps = frames.size(1)
        height = frames.size(3)
        width = frames.size(4)
        flat_frames = frames.reshape(batch * timesteps, 3, height, width)
        level1 = self.stem(flat_frames)
        if self.use_teacher_input and teacher_frames is not None:
            flat_teacher = teacher_frames.reshape(batch * timesteps, 3, height, width)
            teacher_features = self.teacher_stem(flat_teacher)
            level1 = level1 + torch.tanh(self.teacher_gate) * teacher_features
        level1 = self.encoder1(level1)
        level2 = self.encoder2(self.down1(level1))
        level3 = self.encoder3(self.down2(level2))
        return flat_frames, level1, level2, level3

    def _decode_sequence(
        self,
        flat_frames: torch.Tensor,
        level1: torch.Tensor,
        level2: torch.Tensor,
        level3: torch.Tensor,
        temporal: torch.Tensor,
        batch: int,
        timesteps: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        low_height = level3.size(2)
        low_width = level3.size(3)
        decoded = temporal.reshape(
            batch * timesteps, level3.size(1), low_height, low_width
        )
        decoded = F.interpolate(
            decoded,
            size=(level2.size(2), level2.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.decoder2(self.up2(decoded) + level2)
        decoded = F.interpolate(
            decoded,
            size=(level1.size(2), level1.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.decoder1(self.up1(decoded) + level1)
        residual_logits = self.output(decoded)
        if self.max_residual > 0.0:
            residual = self.max_residual * torch.tanh(residual_logits)
        else:
            residual = residual_logits
        restored = flat_frames + residual
        return restored.reshape(batch, timesteps, 3, height, width)

    def _pad_for_motion_alignment(self, feature: torch.Tensor) -> torch.Tensor:
        height = feature.size(2)
        width = feature.size(3)
        padded_height = max(2, height + height % 2)
        padded_width = max(2, width + width % 2)
        return F.pad(
            feature,
            (0, padded_width - width, 0, padded_height - height),
            mode="replicate",
        )

    def _recursive_aligned_temporal(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = features.size(0)
        timesteps = features.size(1)
        height = features.size(3)
        width = features.size(4)

        memory = torch.jit.annotate(List[torch.Tensor], [])
        aligned_latest = torch.jit.annotate(List[torch.Tensor], [])
        adjacent_flows = torch.jit.annotate(List[torch.Tensor], [])
        adjacent_confidences = torch.jit.annotate(List[torch.Tensor], [])
        adjacent_valids = torch.jit.annotate(List[torch.Tensor], [])

        memory.append(features[:, 0])
        aligned_latest.append(self.temporal(features[:, 0:1])[:, -1])

        for timestep in range(1, timesteps):
            previous_original = self._pad_for_motion_alignment(
                features[:, timestep - 1]
            )
            current_original = self._pad_for_motion_alignment(features[:, timestep])
            flow_padded, confidence_padded = self.motion_aligner.estimate_flow(
                previous_original, current_original
            )
            _, valid_padded = self.motion_aligner.warp_source(
                previous_original, flow_padded
            )

            warped_memory = torch.jit.annotate(List[torch.Tensor], [])
            for memory_feature in memory:
                memory_padded = self._pad_for_motion_alignment(memory_feature)
                warped_padded, _ = self.motion_aligner.warp_source(
                    memory_padded, flow_padded
                )
                blend = (confidence_padded * valid_padded).to(
                    dtype=memory_padded.dtype
                )
                aligned_padded = memory_padded + blend * (
                    warped_padded - memory_padded
                )
                warped_memory.append(aligned_padded[:, :, :height, :width])
            memory = warped_memory
            memory.append(features[:, timestep])
            aligned_prefix = torch.stack(memory, dim=1)
            aligned_latest.append(self.temporal(aligned_prefix)[:, -1])
            adjacent_flows.append(flow_padded[:, :, :height, :width])
            adjacent_confidences.append(
                confidence_padded[:, :, :height, :width]
            )
            adjacent_valids.append(valid_padded[:, :, :height, :width])

        aligned_temporal = torch.stack(aligned_latest, dim=1)
        if timesteps > 1:
            flows = torch.stack(adjacent_flows, dim=1)
            confidences = torch.stack(adjacent_confidences, dim=1)
            valids = torch.stack(adjacent_valids, dim=1)
        else:
            flows = features.new_empty((batch, 0, 2, height, width))
            confidences = features.new_empty((batch, 0, 1, height, width))
            valids = features.new_empty((batch, 0, 1, height, width))
        return aligned_temporal, flows, confidences, valids

    @torch.jit.export
    def forward_sequence_with_motion_diagnostics(
        self,
        frames: torch.Tensor,
        teacher_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return predictions plus differentiable adjacent motion diagnostics."""
        self._validate_inputs(frames, teacher_frames)
        batch = frames.size(0)
        timesteps = frames.size(1)
        height = frames.size(3)
        width = frames.size(4)
        flat_frames, level1, level2, level3 = self._encode_sequence(
            frames, teacher_frames
        )
        low_height = level3.size(2)
        low_width = level3.size(3)
        features = level3.reshape(
            batch, timesteps, level3.size(1), low_height, low_width
        )
        baseline_temporal = self.temporal(features)
        aligned_temporal, flows, confidences, valids = (
            self._recursive_aligned_temporal(features)
        )
        temporal = baseline_temporal + torch.tanh(self.motion_alignment_gate) * (
            aligned_temporal - baseline_temporal
        )
        restored = self._decode_sequence(
            flat_frames,
            level1,
            level2,
            level3,
            temporal,
            batch,
            timesteps,
            height,
            width,
        )
        return restored, flows, confidences, valids

    @torch.jit.export
    def forward_sequence_alignment_disabled(
        self,
        frames: torch.Tensor,
        teacher_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the exact inherited-v3 branch from the same v4 artifact."""
        self._validate_inputs(frames, teacher_frames)
        batch = frames.size(0)
        timesteps = frames.size(1)
        height = frames.size(3)
        width = frames.size(4)
        flat_frames, level1, level2, level3 = self._encode_sequence(
            frames, teacher_frames
        )
        temporal = level3.reshape(
            batch,
            timesteps,
            level3.size(1),
            level3.size(2),
            level3.size(3),
        )
        temporal = self.temporal(temporal)
        return self._decode_sequence(
            flat_frames,
            level1,
            level2,
            level3,
            temporal,
            batch,
            timesteps,
            height,
            width,
        )

    @torch.jit.export
    def forward_sequence(
        self,
        frames: torch.Tensor,
        teacher_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_sequence_with_motion_diagnostics(
            frames, teacher_frames
        )[0]

    def forward(
        self,
        frames: torch.Tensor,
        teacher_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_sequence(frames, teacher_frames)[:, -1]

    @torch.jit.unused
    def config_dict(self) -> Dict[str, Any]:
        config = super().config_dict()
        config["motion_alignment"] = {
            "mode": self.motion_alignment_mode,
            "match_channels": self.motion_match_channels,
            "radius": self.motion_radius,
            "temperature": self.motion_temperature,
        }
        return config


def build_causal_video_deblur(config: Optional[Dict[str, Any]] = None) -> CausalVideoDeblur:
    """Construct a model from checkpoint-compatible configuration values."""
    values: Dict[str, Any] = {} if config is None else dict(config)
    # Old v1 checkpoint configs have no max_residual.  They must retain the
    # original unbounded residual semantics.  Fresh models use the safe v3
    # default; v3 training always serializes the explicit value.
    default_max_residual = DEFAULT_MAX_RESIDUAL if config is None else 0.0
    common = {
        "channels": int(values.get("channels", 32)),
        "num_heads": int(values.get("num_heads", 4)),
        "num_blocks": int(values.get("num_blocks", 2)),
        "max_history": int(values.get("max_history", 5)),
        "use_teacher_input": bool(values.get("use_teacher_input", False)),
        "input_domain": str(values.get("input_domain", "raw")),
        "max_residual": float(values.get("max_residual", default_max_residual)),
    }
    motion_config = values.get("motion_alignment")
    if motion_config is None:
        return CausalVideoDeblur(**common)
    if not isinstance(motion_config, dict):
        raise ValueError("motion_alignment must be a configuration object")
    allowed_motion_keys = {"mode", "match_channels", "radius", "temperature"}
    unknown_motion_keys = set(motion_config) - allowed_motion_keys
    if unknown_motion_keys:
        raise ValueError(
            "unsupported motion_alignment keys: "
            + ", ".join(sorted(str(key) for key in unknown_motion_keys))
        )
    mode = str(motion_config.get("mode", ""))
    if mode != "coarse_local_correlation_v1":
        raise ValueError(f"unsupported motion_alignment mode {mode!r}")
    return MotionAlignedCausalVideoDeblurV4(
        **common,
        motion_match_channels=int(motion_config.get("match_channels", 16)),
        motion_radius=int(motion_config.get("radius", 8)),
        motion_temperature=float(motion_config.get("temperature", 0.05)),
    )
