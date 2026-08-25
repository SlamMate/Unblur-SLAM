"""Strictly pairwise feature alignment for a future causal EVSSM adapter.

This module is intentionally independent of the current video-deblur model.
It estimates a bounded current-to-previous sampling flow from two quarter-
resolution feature maps:

    previous/current [B,C,H,W]
      -> local correlation at [B,Cm,H/2,W/2]
      -> current-to-previous flow [B,2,H,W], in quarter-resolution pixels
      -> previous features warped into the current coordinate system

Only ``previous`` and ``current`` are accepted, so the primitive cannot access
future frames.  Integration, temporal fusion, and training losses deliberately
remain outside this file while the existing causal-EVSSM experiment is being
audited.
"""

import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class CoarseLocalCorrelationAligner(nn.Module):
    """Estimate bounded pairwise flow with coarse local correlation.

    The input feature maps are expected at one-quarter image resolution.  A
    shared learned projection is average-pooled once, producing one-eighth
    resolution matching features.  For every current location, cosine
    correlation is evaluated against a square window in the previous feature
    map.  A masked soft-argmax returns a flow proposal, which is bilinearly
    upsampled back to quarter resolution.

    Flow convention:

      ``flow[:, 0]`` is dx and ``flow[:, 1]`` is dy.  At current coordinate
      ``(x, y)``, the warp samples the previous map at
      ``(x + dx, y + dy)``.

    With the fixed downsample factor of two, each component is bounded by
    ``2 * radius`` quarter-resolution pixels.  The default radius eight thus
    covers +/-16 quarter-resolution pixels, or +/-64 input-image pixels.

    ``forward(previous, current)`` returns ``(aligned_previous, flow,
    confidence)``.  ``confidence`` is normalized inverse correlation entropy
    in [0, 1].  ``estimate_flow`` and ``warp_source`` are exported separately
    for diagnostics and future training losses.
    """

    __constants__ = [
        "input_channels",
        "match_channels",
        "radius",
        "window_size",
        "temperature",
        "downsample_factor",
        "max_flow",
    ]

    def __init__(
        self,
        input_channels: int = 128,
        match_channels: int = 16,
        radius: int = 8,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if match_channels < 1:
            raise ValueError("match_channels must be positive")
        if radius < 1:
            raise ValueError("radius must be positive")
        if not math.isfinite(float(temperature)) or not 0.0 < temperature:
            raise ValueError("temperature must be finite and positive")

        self.input_channels = int(input_channels)
        self.match_channels = int(match_channels)
        self.radius = int(radius)
        self.window_size = int(2 * radius + 1)
        self.temperature = float(temperature)
        self.downsample_factor = 2
        self.max_flow = float(self.downsample_factor * radius)

        # The projection is shared across previous/current features, keeping
        # the correlation symmetric before the direction-specific lookup.
        # A bias would be removed by neither cosine normalization nor the
        # shared spatial transform, so omit it intentionally.
        self.match_projection = nn.Conv2d(
            input_channels, match_channels, kernel_size=1, bias=False
        )

        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            indexing="ij",
        )
        offsets = torch.stack(
            (offsets_x.reshape(-1), offsets_y.reshape(-1)), dim=1
        )
        self.register_buffer("offsets", offsets, persistent=True)

    def _validate_pair(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> None:
        if previous.dim() != 4 or current.dim() != 4:
            raise ValueError("previous and current must be BCHW tensors")
        if previous.size(0) != current.size(0):
            raise ValueError("previous/current batch sizes must match")
        if previous.size(1) != current.size(1):
            raise ValueError("previous/current channel counts must match")
        if previous.size(2) != current.size(2) or previous.size(3) != current.size(3):
            raise ValueError("previous/current spatial sizes must match")
        if previous.size(1) != self.input_channels:
            raise ValueError("feature channel count does not match input_channels")
        if previous.size(2) < 2 or previous.size(3) < 2:
            raise ValueError("quarter-resolution features must be at least 2x2")
        if previous.size(2) % 2 != 0 or previous.size(3) % 2 != 0:
            raise ValueError(
                "quarter-resolution feature height/width must be even for the fixed 1/8 contract"
            )

    @torch.jit.export
    def estimate_flow(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current-to-previous flow and confidence at 1/4 resolution."""

        self._validate_pair(previous, current)
        batch = previous.size(0)
        height = previous.size(2)
        width = previous.size(3)

        previous_coarse = F.avg_pool2d(
            previous, kernel_size=self.downsample_factor, stride=self.downsample_factor
        )
        current_coarse = F.avg_pool2d(
            current, kernel_size=self.downsample_factor, stride=self.downsample_factor
        )
        previous_match = F.normalize(
            self.match_projection(previous_coarse).float(),
            p=2.0,
            dim=1,
            eps=1.0e-6,
        )
        current_match = F.normalize(
            self.match_projection(current_coarse).float(),
            p=2.0,
            dim=1,
            eps=1.0e-6,
        )

        coarse_height = previous_match.size(2)
        coarse_width = previous_match.size(3)
        coarse_pixels = coarse_height * coarse_width
        candidates = self.window_size * self.window_size

        # patches: B,Cm,K,N; queries: B,Cm,1,N.  The candidate order matches
        # the registered (dx,dy) offsets buffer.
        patches = F.unfold(
            previous_match,
            kernel_size=self.window_size,
            padding=self.radius,
        ).reshape(batch, self.match_channels, candidates, coarse_pixels)
        queries = current_match.reshape(
            batch, self.match_channels, 1, coarse_pixels
        )
        scores = (patches * queries).sum(dim=1) / self.temperature

        # Zero padding is not a real matching candidate.  Explicitly mask it
        # because a zero correlation could otherwise beat a negative valid
        # cosine score near image borders.
        support = F.unfold(
            torch.ones(
                (batch, 1, coarse_height, coarse_width),
                dtype=previous_match.dtype,
                device=previous_match.device,
            ),
            kernel_size=self.window_size,
            padding=self.radius,
        ).reshape(batch, candidates, coarse_pixels)
        valid_candidates = support > 0.5
        scores = scores.masked_fill(~valid_candidates, -1.0e4)
        probabilities = torch.softmax(scores, dim=1)

        offsets = self.offsets.to(dtype=probabilities.dtype)
        flow_coarse = torch.matmul(
            probabilities.transpose(1, 2), offsets
        ).transpose(1, 2)
        flow_coarse = flow_coarse.reshape(
            batch, 2, coarse_height, coarse_width
        )

        # Inverse normalized entropy gives a dependency-free confidence.  A
        # location with only one valid candidate is assigned zero confidence:
        # it contains no matching evidence even though its entropy is zero.
        safe_probabilities = probabilities.clamp_min(1.0e-12)
        entropy = -(
            probabilities * safe_probabilities.log()
        ).sum(dim=1, keepdim=True)
        valid_count = valid_candidates.sum(dim=1, keepdim=True).to(
            dtype=probabilities.dtype
        )
        maximum_entropy = valid_count.clamp_min(2.0).log()
        confidence_coarse = 1.0 - entropy / maximum_entropy
        confidence_coarse = torch.where(
            valid_count > 1.0,
            confidence_coarse,
            torch.zeros_like(confidence_coarse),
        ).clamp(0.0, 1.0)
        confidence_coarse = confidence_coarse.reshape(
            batch, 1, coarse_height, coarse_width
        )

        flow = F.interpolate(
            flow_coarse,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ) * float(self.downsample_factor)
        confidence = F.interpolate(
            confidence_coarse,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        # Bilinear interpolation is a convex combination, so the bound is
        # already guaranteed analytically.  Clamp as a numerical contract for
        # low-precision inference and future refactors.
        flow = flow.clamp(-self.max_flow, self.max_flow)
        return flow, confidence

    @torch.jit.export
    def warp_source(
        self, source: torch.Tensor, flow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Warp a 1/4-resolution source with current-to-source pixel flow.

        Returns ``(warped, valid)``.  ``valid`` has shape ``[B,1,H,W]`` and is
        one only where the requested source coordinate lies inside the image.
        Border padding prevents black wedges in the returned feature map; the
        validity tensor lets a later integration ignore extrapolated samples.
        """

        if source.dim() != 4:
            raise ValueError("source must be a BCHW tensor")
        if flow.dim() != 4 or flow.size(1) != 2:
            raise ValueError("flow must have shape [B,2,H,W]")
        if source.size(0) != flow.size(0):
            raise ValueError("source/flow batch sizes must match")
        if source.size(2) != flow.size(2) or source.size(3) != flow.size(3):
            raise ValueError("source/flow spatial sizes must match")
        if not source.is_floating_point() or not flow.is_floating_point():
            raise ValueError("source and flow must be floating-point tensors")

        batch = source.size(0)
        height = source.size(2)
        width = source.size(3)
        flow_for_grid = flow.to(dtype=source.dtype)

        base_x = torch.arange(
            width, dtype=source.dtype, device=source.device
        ).reshape(1, 1, 1, width)
        base_y = torch.arange(
            height, dtype=source.dtype, device=source.device
        ).reshape(1, 1, height, 1)
        sample_x = base_x + flow_for_grid[:, 0:1]
        sample_y = base_y + flow_for_grid[:, 1:2]
        valid = (
            (sample_x >= 0.0)
            & (sample_x <= float(width - 1))
            & (sample_y >= 0.0)
            & (sample_y <= float(height - 1))
        ).to(dtype=source.dtype)

        # Pixel-center normalization for align_corners=False.
        normalized_x = 2.0 * (sample_x + 0.5) / float(width) - 1.0
        normalized_y = 2.0 * (sample_y + 0.5) / float(height) - 1.0
        grid = torch.stack(
            (normalized_x[:, 0], normalized_y[:, 0]), dim=-1
        )
        warped = F.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return warped, valid

    def forward(
        self, previous: torch.Tensor, current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return aligned previous features, bounded flow, and confidence."""

        flow, confidence = self.estimate_flow(previous, current)
        aligned_previous, _ = self.warp_source(previous, flow)
        return aligned_previous, flow, confidence


__all__ = ["CoarseLocalCorrelationAligner"]
