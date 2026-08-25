"""Deterministic, role-aware context selection for FrameCrafter.

This module keeps *where to generate* separate from *which real observations
condition generation*.  Given a planned :class:`TargetView`, it constructs a
small context set containing:

* the two real endpoint observations;
* nearby blurry observations that preserve local temporal motion; and
* sharp observations before/after the target that still see the target area.

The selector consumes overlap/reliability metadata produced by the geometry
planner.  It deliberately does not run feature matching, EVSSM, or any GPU
model.  EVSSM images can instead be supplied as audited, precomputed records
or by a lightweight callback.  Every selected image records its actual source
and any fallback, so an ``evssm`` experiment cannot silently become ``raw``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from .framecrafter_pipeline import FrameRecord, TargetView


_EPS = 1.0e-12
_IMAGE_MODES = {"raw", "evssm", "hybrid"}
_CONTEXT_ROLES = {
    "endpoint_left",
    "endpoint_right",
    "endpoint_both",
    "local_blurry_before",
    "local_blurry_after",
    "local_blurry_inside",
    "sharp_before",
    "sharp_after",
    "sharp_context",
    "supplemental_context",
}


@dataclass(frozen=True)
class EVSSMLocalGateMetric:
    """Worst tile for one local RAW-versus-EVSSM degradation metric."""

    value: Optional[float]
    threshold: float
    comparison: str
    tile_xyxy: Optional[tuple[int, int, int, int]]
    eligible_tile_count: int
    failed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "threshold": float(self.threshold),
            "comparison": self.comparison,
            "tile_xyxy": None if self.tile_xyxy is None else list(self.tile_xyxy),
            "eligible_tile_count": int(self.eligible_tile_count),
            "failed": bool(self.failed),
        }


@dataclass(frozen=True)
class EVSSMLocalGateResult:
    """Auditable tile-level decision for one precomputed EVSSM image."""

    passed: bool
    image_shape_hw: Optional[tuple[int, int]]
    tile_size: int
    tile_stride: int
    tile_count: int
    dark_luma_threshold: float
    min_raw_luma: float
    min_raw_edge: float
    min_raw_laplacian: float
    metrics: Mapping[str, EVSSMLocalGateMetric]
    failure_reasons: tuple[str, ...]
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "unblur_slam.evssm_local_gate.v1",
            "passed": bool(self.passed),
            "image_shape_hw": (
                None if self.image_shape_hw is None else list(self.image_shape_hw)
            ),
            "tile_size": int(self.tile_size),
            "tile_stride": int(self.tile_stride),
            "tile_count": int(self.tile_count),
            "dark_luma_threshold": float(self.dark_luma_threshold),
            "eligibility_thresholds": {
                "min_raw_luma": float(self.min_raw_luma),
                "min_raw_edge": float(self.min_raw_edge),
                "min_raw_laplacian": float(self.min_raw_laplacian),
            },
            "metrics": {
                name: metric.as_dict() for name, metric in self.metrics.items()
            },
            "failure_reasons": list(self.failure_reasons),
            "error": self.error,
        }


def _unit_interval(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return result


def _optional_finite(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _absolute(path: Optional[Path | str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


@dataclass
class ContextFrameMetadata:
    """Target-relative metadata for one real source observation.

    ``overlap`` is the bidirectional visible overlap with the requested target
    view, normally computed from intrinsics, extrinsics, and depth.  Optional
    ``pairwise_overlap`` values make the redundancy penalty more accurate; the
    selector falls back to a pose-based similarity when they are unavailable.
    """

    frame: FrameRecord
    position: int
    overlap: float
    reliability: float = 1.0
    sharpness: Optional[float] = None
    is_blurry: Optional[bool] = None
    pairwise_overlap: Mapping[int, float] = field(default_factory=dict)
    evssm_path: Optional[Path] = None
    evssm_confidence: Optional[float] = None
    evssm_sharpness_gain: Optional[float] = None
    evssm_consistency: Optional[float] = None
    evssm_provider: str = "metadata"

    def __post_init__(self) -> None:
        if self.frame.kind != "original":
            raise ValueError("FrameCrafter contexts must be real/original frames")
        self.position = int(self.position)
        self.overlap = _unit_interval(self.overlap, "overlap")
        self.reliability = _unit_interval(self.reliability, "reliability")
        if self.sharpness is None:
            self.sharpness = self.frame.sharpness
        self.sharpness = _optional_finite(self.sharpness, "sharpness")
        if self.sharpness is not None and self.sharpness < 0.0:
            raise ValueError("sharpness cannot be negative")
        if self.is_blurry is not None:
            self.is_blurry = bool(self.is_blurry)
        cleaned_overlap: dict[int, float] = {}
        for source_index, value in self.pairwise_overlap.items():
            cleaned_overlap[int(source_index)] = _unit_interval(
                value, f"pairwise_overlap[{source_index}]"
            )
        self.pairwise_overlap = cleaned_overlap
        self.evssm_path = _absolute(self.evssm_path)
        self.evssm_confidence = _optional_finite(
            self.evssm_confidence, "evssm_confidence"
        )
        self.evssm_sharpness_gain = _optional_finite(
            self.evssm_sharpness_gain, "evssm_sharpness_gain"
        )
        self.evssm_consistency = _optional_finite(
            self.evssm_consistency, "evssm_consistency"
        )
        self.evssm_provider = str(self.evssm_provider)


@dataclass(frozen=True)
class EVSSMImageCandidate:
    """One already-computed EVSSM result and its CPU-side gate metadata."""

    path: Path
    confidence: float
    sharpness_gain: float
    consistency: float
    provider: str = "precomputed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _absolute(self.path))
        object.__setattr__(
            self, "confidence", _unit_interval(self.confidence, "confidence")
        )
        gain = float(self.sharpness_gain)
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("sharpness_gain must be finite and non-negative")
        object.__setattr__(self, "sharpness_gain", gain)
        object.__setattr__(
            self, "consistency", _unit_interval(self.consistency, "consistency")
        )
        object.__setattr__(self, "provider", str(self.provider))


EVSSMCallback = Callable[[ContextFrameMetadata], Optional[EVSSMImageCandidate]]


@dataclass
class EVSSMResolver:
    """Resolve precomputed EVSSM images without importing an EVSSM model.

    Resolution order is metadata, explicit mapping, precomputed root, then the
    callback.  Mapping keys may be source indices, frame ids, or source names.
    A root uses ``path_template`` with ``source_index``, ``frame_id``, ``name``,
    ``stem``, and ``suffix`` fields.  Root/metadata candidates take their gate
    metrics from :class:`ContextFrameMetadata`.
    """

    precomputed: Mapping[object, EVSSMImageCandidate] = field(default_factory=dict)
    precomputed_root: Optional[Path] = None
    path_template: str = "{name}"
    callback: Optional[EVSSMCallback] = None

    def __post_init__(self) -> None:
        self.precomputed_root = _absolute(self.precomputed_root)
        self.path_template = str(self.path_template)

    @staticmethod
    def _from_metadata(
        metadata: ContextFrameMetadata, path: Optional[Path], provider: str
    ) -> tuple[Optional[EVSSMImageCandidate], Optional[str]]:
        if path is None or not path.is_file():
            return None, "missing_evssm_file"
        metrics = (
            metadata.evssm_confidence,
            metadata.evssm_sharpness_gain,
            metadata.evssm_consistency,
        )
        if any(value is None for value in metrics):
            return None, "missing_evssm_gate_metrics"
        return (
            EVSSMImageCandidate(
                path=path,
                confidence=float(metrics[0]),
                sharpness_gain=float(metrics[1]),
                consistency=float(metrics[2]),
                provider=provider,
            ),
            None,
        )

    def resolve(
        self, metadata: ContextFrameMetadata
    ) -> tuple[Optional[EVSSMImageCandidate], Optional[str]]:
        """Return a candidate and an audit reason when no candidate exists."""

        reasons: list[str] = []
        if metadata.evssm_path is not None:
            candidate, reason = self._from_metadata(
                metadata, metadata.evssm_path, metadata.evssm_provider
            )
            if candidate is not None:
                return candidate, None
            if reason:
                reasons.append(reason)

        keys = (
            metadata.frame.source_index,
            metadata.frame.frame_id,
            metadata.frame.rgb_path.name,
        )
        for key in keys:
            candidate = self.precomputed.get(key)
            if candidate is not None:
                if not candidate.path.is_file():
                    return None, "missing_evssm_file"
                return candidate, None

        if self.precomputed_root is not None:
            raw = metadata.frame.rgb_path
            try:
                relative = self.path_template.format(
                    source_index=metadata.frame.source_index,
                    frame_id=metadata.frame.frame_id,
                    name=raw.name,
                    stem=raw.stem,
                    suffix=raw.suffix,
                )
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid EVSSM path_template: {error}") from error
            candidate, reason = self._from_metadata(
                metadata,
                (self.precomputed_root / relative).resolve(),
                "precomputed_root",
            )
            if candidate is not None:
                return candidate, None
            if reason:
                reasons.append(reason)

        if self.callback is not None:
            candidate = self.callback(metadata)
            if candidate is not None:
                if not isinstance(candidate, EVSSMImageCandidate):
                    raise TypeError("EVSSM callback must return EVSSMImageCandidate or None")
                if not candidate.path.is_file():
                    return None, "missing_evssm_file"
                return candidate, None
            reasons.append("callback_returned_none")

        if not reasons:
            reasons.append("evssm_not_configured")
        return None, ";".join(dict.fromkeys(reasons))


@dataclass(frozen=True)
class ContextSelectionConfig:
    """Selection, scoring, and image-source policy."""

    context_budget: int = 6
    min_contexts: int = 3
    local_blurry_count: int = 2
    sharp_context_count: int = 2
    local_radius: int = 8
    min_sharp_overlap: float = 0.25
    blur_quantile: float = 0.35
    sharp_quantile: float = 0.65
    overlap_weight: float = 0.35
    sharpness_weight: float = 0.25
    reliability_weight: float = 0.15
    view_diversity_weight: float = 0.15
    redundancy_weight: float = 0.10
    locality_weight: float = 0.25
    translation_diversity_scale: float = 0.10
    rotation_diversity_scale_deg: float = 15.0
    seed: int = 0
    fill_budget: bool = True
    image_mode: str = "raw"
    evssm_min_confidence: float = 0.70
    evssm_min_sharpness_gain: float = 1.0
    evssm_min_consistency: float = 0.80
    hybrid_evssm_roles: tuple[str, ...] = (
        "local_blurry_inside",
    )
    evssm_local_gate_enabled: bool = True
    evssm_local_tile_size: int = 32
    evssm_local_tile_stride: int = 16
    evssm_local_max_brightness_drop: float = 0.30
    evssm_local_min_edge_retention: float = 0.50
    evssm_local_min_laplacian_retention: float = 0.50
    evssm_local_max_tile_mae: float = 0.20
    evssm_local_max_dark_expansion: float = 0.30
    evssm_local_dark_luma_threshold: float = 96.0 / 255.0
    evssm_local_min_raw_luma: float = 0.10
    evssm_local_min_raw_edge: float = 0.01
    evssm_local_min_raw_laplacian: float = 0.01

    def __post_init__(self) -> None:
        if int(self.context_budget) < 2:
            raise ValueError("context_budget must leave room for both endpoints")
        if not 2 <= int(self.min_contexts) <= int(self.context_budget):
            raise ValueError("min_contexts must be in [2, context_budget]")
        if int(self.local_blurry_count) < 0 or int(self.sharp_context_count) < 0:
            raise ValueError("context role counts cannot be negative")
        if 2 + int(self.local_blurry_count) + int(self.sharp_context_count) > int(
            self.context_budget
        ):
            raise ValueError(
                "endpoint/local/sharp quotas exceed context_budget; lower a quota"
            )
        if int(self.local_radius) < 1:
            raise ValueError("local_radius must be positive")
        _unit_interval(self.min_sharp_overlap, "min_sharp_overlap")
        _unit_interval(self.blur_quantile, "blur_quantile")
        _unit_interval(self.sharp_quantile, "sharp_quantile")
        if self.blur_quantile > self.sharp_quantile:
            raise ValueError("blur_quantile cannot exceed sharp_quantile")
        weights = (
            self.overlap_weight,
            self.sharpness_weight,
            self.reliability_weight,
            self.view_diversity_weight,
            self.redundancy_weight,
            self.locality_weight,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights):
            raise ValueError("context scoring weights must be finite and non-negative")
        if self.translation_diversity_scale <= 0.0:
            raise ValueError("translation_diversity_scale must be positive")
        if self.rotation_diversity_scale_deg <= 0.0:
            raise ValueError("rotation_diversity_scale_deg must be positive")
        if self.image_mode not in _IMAGE_MODES:
            raise ValueError(f"image_mode must be one of {sorted(_IMAGE_MODES)}")
        _unit_interval(self.evssm_min_confidence, "evssm_min_confidence")
        if self.evssm_min_sharpness_gain < 0.0:
            raise ValueError("evssm_min_sharpness_gain cannot be negative")
        _unit_interval(self.evssm_min_consistency, "evssm_min_consistency")
        roles = tuple(str(role).strip() for role in self.hybrid_evssm_roles)
        if any(not role for role in roles):
            raise ValueError("hybrid_evssm_roles cannot contain empty roles")
        if len(set(roles)) != len(roles):
            raise ValueError("hybrid_evssm_roles cannot contain duplicates")
        unknown_roles = sorted(set(roles) - _CONTEXT_ROLES)
        if unknown_roles:
            raise ValueError(f"unknown hybrid EVSSM context roles: {unknown_roles}")
        object.__setattr__(self, "hybrid_evssm_roles", roles)
        if int(self.evssm_local_tile_size) < 4:
            raise ValueError("evssm_local_tile_size must be at least 4")
        if int(self.evssm_local_tile_stride) < 1:
            raise ValueError("evssm_local_tile_stride must be positive")
        _unit_interval(
            self.evssm_local_max_brightness_drop,
            "evssm_local_max_brightness_drop",
        )
        _unit_interval(
            self.evssm_local_min_edge_retention,
            "evssm_local_min_edge_retention",
        )
        _unit_interval(
            self.evssm_local_min_laplacian_retention,
            "evssm_local_min_laplacian_retention",
        )
        _unit_interval(self.evssm_local_max_tile_mae, "evssm_local_max_tile_mae")
        _unit_interval(
            self.evssm_local_max_dark_expansion,
            "evssm_local_max_dark_expansion",
        )
        _unit_interval(
            self.evssm_local_dark_luma_threshold,
            "evssm_local_dark_luma_threshold",
        )
        _unit_interval(self.evssm_local_min_raw_luma, "evssm_local_min_raw_luma")
        for value, name in (
            (self.evssm_local_min_raw_edge, "evssm_local_min_raw_edge"),
            (
                self.evssm_local_min_raw_laplacian,
                "evssm_local_min_raw_laplacian",
            ),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _luma_and_local_derivatives(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    luma = np.asarray(rgb, dtype=np.float32) @ np.asarray(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    dx = np.zeros_like(luma)
    dy = np.zeros_like(luma)
    dx[:, 1:-1] = 0.5 * (luma[:, 2:] - luma[:, :-2])
    dx[:, 0] = luma[:, 1] - luma[:, 0]
    dx[:, -1] = luma[:, -1] - luma[:, -2]
    dy[1:-1, :] = 0.5 * (luma[2:, :] - luma[:-2, :])
    dy[0, :] = luma[1, :] - luma[0, :]
    dy[-1, :] = luma[-1, :] - luma[-2, :]
    edge = np.hypot(dx, dy)
    laplacian = np.zeros_like(luma)
    laplacian[1:-1, 1:-1] = np.abs(
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
    )
    return luma, edge, laplacian


def _local_gate_error(
    config: ContextSelectionConfig, reason: str, error: str
) -> EVSSMLocalGateResult:
    return EVSSMLocalGateResult(
        passed=False,
        image_shape_hw=None,
        tile_size=int(config.evssm_local_tile_size),
        tile_stride=int(config.evssm_local_tile_stride),
        tile_count=0,
        dark_luma_threshold=float(config.evssm_local_dark_luma_threshold),
        min_raw_luma=float(config.evssm_local_min_raw_luma),
        min_raw_edge=float(config.evssm_local_min_raw_edge),
        min_raw_laplacian=float(config.evssm_local_min_raw_laplacian),
        metrics={},
        failure_reasons=(reason,),
        error=error,
    )


def audit_evssm_local_degradation(
    raw_path: Path | str,
    evssm_path: Path | str,
    config: ContextSelectionConfig,
) -> EVSSMLocalGateResult:
    """Compare spatially aligned RAW/EVSSM images over overlapping tiles.

    Global Laplacian and consistency scores can hide a small collapsed object.
    This audit therefore rejects the entire EVSSM conditioning frame when any
    tile severely loses brightness/edges/Laplacian energy, exceeds RGB MAE, or
    expands its dark-pixel support.  Very dark or textureless RAW tiles are
    excluded from ratio metrics so numerical noise cannot trigger a fallback.
    """

    try:
        with Image.open(raw_path) as image:
            raw = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(evssm_path) as image:
            output = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except (OSError, ValueError) as error:
        return _local_gate_error(
            config, "read_error", f"{type(error).__name__}: {error}"
        )
    if raw.shape != output.shape:
        return _local_gate_error(
            config,
            "shape_mismatch",
            f"RAW shape {raw.shape} != EVSSM shape {output.shape}",
        )
    height, width = raw.shape[:2]
    if height < 2 or width < 2:
        return _local_gate_error(
            config, "image_too_small", f"image shape {(height, width)} is too small"
        )

    tile_height = min(int(config.evssm_local_tile_size), height)
    tile_width = min(int(config.evssm_local_tile_size), width)
    y_starts = _tile_starts(height, tile_height, int(config.evssm_local_tile_stride))
    x_starts = _tile_starts(width, tile_width, int(config.evssm_local_tile_stride))
    raw_luma, raw_edge, raw_laplacian = _luma_and_local_derivatives(raw)
    output_luma, output_edge, output_laplacian = _luma_and_local_derivatives(output)

    maxima: dict[str, tuple[float, tuple[int, int, int, int], int]] = {}
    minima: dict[str, tuple[float, tuple[int, int, int, int], int]] = {}

    def update_maximum(name: str, value: float, tile: tuple[int, int, int, int]) -> None:
        previous = maxima.get(name)
        count = 1 if previous is None else previous[2] + 1
        if previous is None or value > previous[0]:
            maxima[name] = (float(value), tile, count)
        else:
            maxima[name] = (previous[0], previous[1], count)

    def update_minimum(name: str, value: float, tile: tuple[int, int, int, int]) -> None:
        previous = minima.get(name)
        count = 1 if previous is None else previous[2] + 1
        if previous is None or value < previous[0]:
            minima[name] = (float(value), tile, count)
        else:
            minima[name] = (previous[0], previous[1], count)

    for y0 in y_starts:
        for x0 in x_starts:
            y1, x1 = y0 + tile_height, x0 + tile_width
            tile = (x0, y0, x1, y1)
            raw_rgb_tile = raw[y0:y1, x0:x1]
            output_rgb_tile = output[y0:y1, x0:x1]
            raw_luma_tile = raw_luma[y0:y1, x0:x1]
            output_luma_tile = output_luma[y0:y1, x0:x1]
            raw_mean = float(np.mean(raw_luma_tile))
            output_mean = float(np.mean(output_luma_tile))
            if raw_mean >= float(config.evssm_local_min_raw_luma):
                update_maximum(
                    "brightness_drop",
                    max(0.0, raw_mean - output_mean) / max(raw_mean, _EPS),
                    tile,
                )
            raw_edge_mean = float(np.mean(raw_edge[y0:y1, x0:x1]))
            if raw_edge_mean >= float(config.evssm_local_min_raw_edge):
                update_minimum(
                    "edge_retention",
                    float(np.mean(output_edge[y0:y1, x0:x1]))
                    / max(raw_edge_mean, _EPS),
                    tile,
                )
            raw_laplacian_mean = float(np.mean(raw_laplacian[y0:y1, x0:x1]))
            if raw_laplacian_mean >= float(config.evssm_local_min_raw_laplacian):
                update_minimum(
                    "laplacian_retention",
                    float(np.mean(output_laplacian[y0:y1, x0:x1]))
                    / max(raw_laplacian_mean, _EPS),
                    tile,
                )
            update_maximum(
                "tile_mae",
                float(np.mean(np.abs(raw_rgb_tile - output_rgb_tile))),
                tile,
            )
            raw_dark = float(
                np.mean(raw_luma_tile < config.evssm_local_dark_luma_threshold)
            )
            output_dark = float(
                np.mean(output_luma_tile < config.evssm_local_dark_luma_threshold)
            )
            update_maximum("dark_expansion", max(0.0, output_dark - raw_dark), tile)

    definitions = (
        (
            "brightness_drop",
            maxima.get("brightness_drop"),
            float(config.evssm_local_max_brightness_drop),
            "max",
        ),
        (
            "edge_retention",
            minima.get("edge_retention"),
            float(config.evssm_local_min_edge_retention),
            "min",
        ),
        (
            "laplacian_retention",
            minima.get("laplacian_retention"),
            float(config.evssm_local_min_laplacian_retention),
            "min",
        ),
        (
            "tile_mae",
            maxima.get("tile_mae"),
            float(config.evssm_local_max_tile_mae),
            "max",
        ),
        (
            "dark_expansion",
            maxima.get("dark_expansion"),
            float(config.evssm_local_max_dark_expansion),
            "max",
        ),
    )
    metrics: dict[str, EVSSMLocalGateMetric] = {}
    failures = []
    for name, record, threshold, comparison in definitions:
        value = None if record is None else float(record[0])
        failed = value is not None and (
            value > threshold if comparison == "max" else value < threshold
        )
        if failed:
            failures.append(name)
        metrics[name] = EVSSMLocalGateMetric(
            value=value,
            threshold=threshold,
            comparison=comparison,
            tile_xyxy=None if record is None else record[1],
            eligible_tile_count=0 if record is None else int(record[2]),
            failed=bool(failed),
        )
    return EVSSMLocalGateResult(
        passed=not failures,
        image_shape_hw=(height, width),
        tile_size=int(config.evssm_local_tile_size),
        tile_stride=int(config.evssm_local_tile_stride),
        tile_count=len(y_starts) * len(x_starts),
        dark_luma_threshold=float(config.evssm_local_dark_luma_threshold),
        min_raw_luma=float(config.evssm_local_min_raw_luma),
        min_raw_edge=float(config.evssm_local_min_raw_edge),
        min_raw_laplacian=float(config.evssm_local_min_raw_laplacian),
        metrics=metrics,
        failure_reasons=tuple(failures),
    )


@dataclass(frozen=True)
class ContextScore:
    overlap: float
    sharpness: float
    reliability: float
    view_diversity: float
    redundancy: float
    locality: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "overlap": self.overlap,
            "sharpness": self.sharpness,
            "reliability": self.reliability,
            "view_diversity": self.view_diversity,
            "redundancy": self.redundancy,
            "locality": self.locality,
            "total": self.total,
        }


@dataclass(frozen=True)
class ImageProvenance:
    requested_mode: str
    resolved_mode: str
    provider: str
    raw_path: Path
    resolved_path: Path
    fallback_reason: Optional[str]
    evssm_confidence: Optional[float]
    evssm_sharpness_gain: Optional[float]
    evssm_consistency: Optional[float]
    evssm_local_gate: Optional[EVSSMLocalGateResult]


@dataclass(frozen=True)
class SelectedContext:
    """One selected real camera observation and its actual conditioning image."""

    frame: FrameRecord
    position: int
    role: str
    image_path: Path
    score: ContextScore
    provenance: ImageProvenance

    def as_frame_record(self) -> FrameRecord:
        """Clone the camera metadata while substituting the resolved RGB path."""

        return FrameRecord(
            source_index=self.frame.source_index,
            frame_id=self.frame.frame_id,
            timestamp=self.frame.timestamp,
            rgb_path=self.image_path,
            depth_path=self.frame.depth_path,
            c2w=self.frame.c2w.copy(),
            intrinsics=self.frame.intrinsics.copy(),
            sharpness=self.frame.sharpness,
            eval=self.frame.eval,
            kind="original",
        )


@dataclass(frozen=True)
class ContextSelectionResult:
    target_id: str
    contexts: tuple[SelectedContext, ...]
    requested_image_mode: str
    seed: int

    @property
    def frame_records(self) -> tuple[FrameRecord, ...]:
        return tuple(context.as_frame_record() for context in self.contexts)

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(context.frame.source_index for context in self.contexts)


def _stable_tie_breaker(seed: int, target_id: str, source_index: int, role: str) -> int:
    payload = f"{int(seed)}\0{target_id}\0{int(source_index)}\0{role}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rotation_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _pose_distance(
    left: ContextFrameMetadata,
    right: ContextFrameMetadata,
    config: ContextSelectionConfig,
) -> float:
    translation = float(
        np.linalg.norm(left.frame.c2w[:3, 3] - right.frame.c2w[:3, 3])
    )
    rotation = _rotation_angle_degrees(left.frame.c2w, right.frame.c2w)
    translation_term = min(1.0, translation / config.translation_diversity_scale)
    rotation_term = min(1.0, rotation / config.rotation_diversity_scale_deg)
    return 0.5 * translation_term + 0.5 * rotation_term


def _deduplicate_metadata(
    metadata: Sequence[ContextFrameMetadata],
) -> list[ContextFrameMetadata]:
    """Deterministically keep the most informative duplicate observation."""

    grouped: dict[int, list[ContextFrameMetadata]] = {}
    for item in metadata:
        grouped.setdefault(item.frame.source_index, []).append(item)
    result: list[ContextFrameMetadata] = []
    for source_index in sorted(grouped):
        group = grouped[source_index]
        positions = {item.position for item in group}
        frame_ids = {item.frame.frame_id for item in group}
        if len(positions) != 1 or len(frame_ids) != 1:
            raise ValueError(
                f"conflicting metadata for source_index={source_index}: "
                "position/frame_id differ"
            )
        result.append(
            max(
                group,
                key=lambda item: (
                    item.overlap,
                    item.reliability,
                    -1.0 if item.sharpness is None else item.sharpness,
                    str(item.frame.rgb_path),
                ),
            )
        )
    return result


def _sharpness_normalization(
    metadata: Sequence[ContextFrameMetadata],
) -> tuple[dict[int, float], float, float]:
    known = [item.sharpness for item in metadata if item.sharpness is not None]
    if not known:
        return ({item.frame.source_index: 0.5 for item in metadata}, 0.0, 0.0)
    minimum = float(min(known))
    maximum = float(max(known))
    span = maximum - minimum
    normalized = {
        item.frame.source_index: (
            0.5
            if item.sharpness is None or span <= _EPS
            else float((item.sharpness - minimum) / span)
        )
        for item in metadata
    }
    return normalized, minimum, maximum


def _score_candidate(
    item: ContextFrameMetadata,
    selected: Sequence[ContextFrameMetadata],
    sharpness_normalized: Mapping[int, float],
    target: TargetView,
    config: ContextSelectionConfig,
) -> ContextScore:
    if selected:
        pose_distances = [_pose_distance(item, other, config) for other in selected]
        view_diversity = float(min(pose_distances))
        redundancies = []
        for other, pose_distance in zip(selected, pose_distances):
            overlap = item.pairwise_overlap.get(other.frame.source_index)
            if overlap is None:
                overlap = other.pairwise_overlap.get(item.frame.source_index)
            redundancies.append(float(overlap) if overlap is not None else 1.0 - pose_distance)
        redundancy = float(max(redundancies))
    else:
        view_diversity = 0.0
        redundancy = 0.0
    if item.position < target.left_position:
        temporal_distance = target.left_position - item.position
    elif item.position > target.right_position:
        temporal_distance = item.position - target.right_position
    else:
        temporal_distance = 0
    locality = max(0.0, 1.0 - temporal_distance / max(1, config.local_radius))
    sharpness = float(sharpness_normalized[item.frame.source_index])
    total = (
        config.overlap_weight * item.overlap
        + config.sharpness_weight * sharpness
        + config.reliability_weight * item.reliability
        + config.view_diversity_weight * view_diversity
        - config.redundancy_weight * redundancy
    )
    return ContextScore(
        overlap=item.overlap,
        sharpness=sharpness,
        reliability=item.reliability,
        view_diversity=view_diversity,
        redundancy=redundancy,
        locality=float(locality),
        total=float(total),
    )


def _choose_best(
    candidates: Sequence[ContextFrameMetadata],
    selected: Sequence[ContextFrameMetadata],
    sharpness_normalized: Mapping[int, float],
    target: TargetView,
    config: ContextSelectionConfig,
    role: str,
    favor_locality: bool = False,
) -> tuple[ContextFrameMetadata, ContextScore]:
    if not candidates:
        raise ValueError("cannot choose from an empty context candidate set")
    ranked = []
    for item in candidates:
        score = _score_candidate(item, selected, sharpness_normalized, target, config)
        rank_score = score.total + (
            config.locality_weight * score.locality if favor_locality else 0.0
        )
        ranked.append(
            (
                -rank_score,
                _stable_tie_breaker(
                    config.seed, target.target_id, item.frame.source_index, role
                ),
                item.frame.source_index,
                item,
                score,
            )
        )
    _, _, _, item, score = min(ranked, key=lambda value: value[:3])
    return item, score


def _evssm_gate_failure(
    candidate: EVSSMImageCandidate, config: ContextSelectionConfig
) -> Optional[str]:
    failures = []
    if candidate.confidence < config.evssm_min_confidence:
        failures.append("evssm_confidence")
    if candidate.sharpness_gain < config.evssm_min_sharpness_gain:
        failures.append("evssm_sharpness_gain")
    if candidate.consistency < config.evssm_min_consistency:
        failures.append("evssm_consistency")
    return ",".join(failures) if failures else None


def _resolve_image(
    metadata: ContextFrameMetadata,
    role: str,
    config: ContextSelectionConfig,
    resolver: Optional[EVSSMResolver],
) -> tuple[Path, ImageProvenance]:
    raw_path = metadata.frame.rgb_path
    wants_evssm = config.image_mode == "evssm" or (
        config.image_mode == "hybrid" and role in config.hybrid_evssm_roles
    )
    if not wants_evssm:
        provider = "raw" if config.image_mode == "raw" else "hybrid_raw_role"
        return raw_path, ImageProvenance(
            requested_mode=config.image_mode,
            resolved_mode="raw",
            provider=provider,
            raw_path=raw_path,
            resolved_path=raw_path,
            fallback_reason=None,
            evssm_confidence=None,
            evssm_sharpness_gain=None,
            evssm_consistency=None,
            evssm_local_gate=None,
        )

    if resolver is None:
        candidate, missing_reason = None, "evssm_resolver_not_configured"
    else:
        candidate, missing_reason = resolver.resolve(metadata)
    local_gate = None
    if candidate is not None:
        gate_failure = _evssm_gate_failure(candidate, config)
        if gate_failure is None:
            if config.evssm_local_gate_enabled:
                local_gate = audit_evssm_local_degradation(
                    raw_path, candidate.path, config
                )
                if not local_gate.passed:
                    gate_failure = ",".join(
                        f"evssm_local_{reason}"
                        for reason in local_gate.failure_reasons
                    )
            if gate_failure is None:
                return candidate.path, ImageProvenance(
                    requested_mode=config.image_mode,
                    resolved_mode="evssm",
                    provider=candidate.provider,
                    raw_path=raw_path,
                    resolved_path=candidate.path,
                    fallback_reason=None,
                    evssm_confidence=candidate.confidence,
                    evssm_sharpness_gain=candidate.sharpness_gain,
                    evssm_consistency=candidate.consistency,
                    evssm_local_gate=local_gate,
                )
        missing_reason = gate_failure
    return raw_path, ImageProvenance(
        requested_mode=config.image_mode,
        resolved_mode="raw",
        provider="raw_fallback",
        raw_path=raw_path,
        resolved_path=raw_path,
        fallback_reason=missing_reason,
        evssm_confidence=None if candidate is None else candidate.confidence,
        evssm_sharpness_gain=None if candidate is None else candidate.sharpness_gain,
        evssm_consistency=None if candidate is None else candidate.consistency,
        evssm_local_gate=local_gate,
    )


def select_framecrafter_contexts(
    target: TargetView,
    metadata: Sequence[ContextFrameMetadata],
    config: ContextSelectionConfig = ContextSelectionConfig(),
    evssm_resolver: Optional[EVSSMResolver] = None,
) -> ContextSelectionResult:
    """Select a bounded, deterministic context set for one target view.

    The returned tuple is temporally ordered for FrameCrafter.  ``role`` and
    selection scores preserve why each observation was chosen.  Endpoint
    observations are mandatory, selection is de-duplicated by ``source_index``,
    and a fixed seed affects only exact-score tie breaking.
    """

    candidates = _deduplicate_metadata(metadata)
    if len(candidates) < config.min_contexts:
        raise ValueError(
            f"need at least {config.min_contexts} unique real contexts, got "
            f"{len(candidates)}"
        )
    by_index = {item.frame.source_index: item for item in candidates}
    missing_endpoints = [
        index
        for index in (target.left_index, target.right_index)
        if index not in by_index
    ]
    if missing_endpoints:
        raise ValueError(f"target endpoint metadata missing: {missing_endpoints}")

    sharpness_normalized, _, _ = _sharpness_normalization(candidates)
    known_sharpness = np.asarray(
        [item.sharpness for item in candidates if item.sharpness is not None],
        dtype=np.float64,
    )
    if known_sharpness.size:
        blur_cutoff = float(np.quantile(known_sharpness, config.blur_quantile))
        sharp_cutoff = float(np.quantile(known_sharpness, config.sharp_quantile))
    else:
        blur_cutoff = -math.inf
        sharp_cutoff = math.inf

    def is_blurry(item: ContextFrameMetadata) -> bool:
        if item.is_blurry is not None:
            return item.is_blurry
        return item.sharpness is not None and item.sharpness <= blur_cutoff

    def is_sharp(item: ContextFrameMetadata) -> bool:
        if item.is_blurry is not None:
            return not item.is_blurry
        return item.sharpness is not None and item.sharpness >= sharp_cutoff

    selected: dict[int, tuple[ContextFrameMetadata, str, ContextScore]] = {}

    def add(item: ContextFrameMetadata, role: str, score: ContextScore) -> None:
        if item.frame.source_index not in selected:
            selected[item.frame.source_index] = (item, role, score)

    # Endpoints are immutable real anchors irrespective of their blur score.
    left = by_index[target.left_index]
    right = by_index[target.right_index]
    add(
        left,
        "endpoint_left",
        _score_candidate(left, [], sharpness_normalized, target, config),
    )
    if right.frame.source_index == left.frame.source_index:
        selected[left.frame.source_index] = (
            left,
            "endpoint_both",
            selected[left.frame.source_index][2],
        )
    else:
        add(
            right,
            "endpoint_right",
            _score_candidate(right, [left], sharpness_normalized, target, config),
        )

    # Preserve local temporal evidence, balancing observations on both sides.
    local_pool = [
        item
        for item in candidates
        if item.frame.source_index not in selected
        and is_blurry(item)
        and target.left_position - config.local_radius
        <= item.position
        <= target.right_position + config.local_radius
    ]
    local_slots = config.local_blurry_count
    preferred_sides = ("before", "after") if local_slots > 1 else ("before",)
    for side in preferred_sides:
        if local_slots <= 0:
            break
        if side == "before":
            pool = [item for item in local_pool if item.position < target.left_position]
            role = "local_blurry_before"
        else:
            pool = [item for item in local_pool if item.position > target.right_position]
            role = "local_blurry_after"
        if not pool:
            continue
        item, score = _choose_best(
            pool,
            [value[0] for value in selected.values()],
            sharpness_normalized,
            target,
            config,
            role,
            favor_locality=True,
        )
        add(item, role, score)
        local_pool = [
            value
            for value in local_pool
            if value.frame.source_index != item.frame.source_index
        ]
        local_slots -= 1
    while local_slots > 0 and local_pool:
        item, score = _choose_best(
            local_pool,
            [value[0] for value in selected.values()],
            sharpness_normalized,
            target,
            config,
            "local_blurry",
            favor_locality=True,
        )
        if item.position < target.left_position:
            role = "local_blurry_before"
        elif item.position > target.right_position:
            role = "local_blurry_after"
        else:
            role = "local_blurry_inside"
        add(item, role, score)
        local_pool = [
            value
            for value in local_pool
            if value.frame.source_index != item.frame.source_index
        ]
        local_slots -= 1

    # Sharp guides require overlap with the target.  Reserve one on each side
    # when both exist, then greedily fill any remaining sharp quota.
    sharp_pool = [
        item
        for item in candidates
        if item.frame.source_index not in selected
        and is_sharp(item)
        and item.overlap >= config.min_sharp_overlap
    ]
    sharp_slots = config.sharp_context_count
    preferred_sharp_sides = (
        ("before", "sharp_before"),
        ("after", "sharp_after"),
    )
    for side, role in preferred_sharp_sides:
        if sharp_slots <= 0:
            break
        if side == "before":
            pool = [item for item in sharp_pool if item.position < target.left_position]
        else:
            pool = [item for item in sharp_pool if item.position > target.right_position]
        if not pool:
            continue
        item, score = _choose_best(
            pool,
            [value[0] for value in selected.values()],
            sharpness_normalized,
            target,
            config,
            role,
        )
        add(item, role, score)
        sharp_pool = [
            value
            for value in sharp_pool
            if value.frame.source_index != item.frame.source_index
        ]
        sharp_slots -= 1
    while sharp_slots > 0 and sharp_pool:
        item, score = _choose_best(
            sharp_pool,
            [value[0] for value in selected.values()],
            sharpness_normalized,
            target,
            config,
            "sharp_context",
        )
        add(item, "sharp_context", score)
        sharp_pool = [
            value
            for value in sharp_pool
            if value.frame.source_index != item.frame.source_index
        ]
        sharp_slots -= 1

    if config.fill_budget:
        while len(selected) < min(config.context_budget, len(candidates)):
            pool = [
                item
                for item in candidates
                if item.frame.source_index not in selected
            ]
            if not pool:
                break
            item, score = _choose_best(
                pool,
                [value[0] for value in selected.values()],
                sharpness_normalized,
                target,
                config,
                "supplemental_context",
            )
            add(item, "supplemental_context", score)

    if len(selected) < config.min_contexts:
        raise ValueError(
            f"context selection produced {len(selected)} frames, below "
            f"min_contexts={config.min_contexts}"
        )
    if len(selected) > config.context_budget:
        raise AssertionError("context selection exceeded its budget")

    resolved = []
    for item, role, score in sorted(
        selected.values(),
        key=lambda value: (value[0].position, value[0].frame.source_index),
    ):
        image_path, provenance = _resolve_image(
            item, role, config, evssm_resolver
        )
        resolved.append(
            SelectedContext(
                frame=item.frame,
                position=item.position,
                role=role,
                image_path=image_path,
                score=score,
                provenance=provenance,
            )
        )
    return ContextSelectionResult(
        target_id=target.target_id,
        contexts=tuple(resolved),
        requested_image_mode=config.image_mode,
        seed=int(config.seed),
    )


__all__ = [
    "ContextFrameMetadata",
    "ContextScore",
    "ContextSelectionConfig",
    "ContextSelectionResult",
    "EVSSMCallback",
    "EVSSMImageCandidate",
    "EVSSMLocalGateMetric",
    "EVSSMLocalGateResult",
    "EVSSMResolver",
    "ImageProvenance",
    "SelectedContext",
    "audit_evssm_local_degradation",
    "select_framecrafter_contexts",
]
