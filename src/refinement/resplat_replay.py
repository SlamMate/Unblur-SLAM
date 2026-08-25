"""Residual-guided replay for long-sequence Unblur-SLAM refinement.

This module is **ReSplat-inspired**: it borrows the idea of repeatedly using
rendering error as feedback, but it is not the official learned ReSplat model
and does not claim checkpoint compatibility with it.  The sampler is agnostic
to the renderer.  A caller probes a small set of training keyframes, reports
their signals with :meth:`observe`, and asks for the next frames with
:meth:`sample` or :meth:`sample_many`.

Sampling and priority updates are O(log N) through Fenwick trees.  In
particular, selecting refinement views does not require rescoring every frame
of a long TUM sequence on every iteration.  ``uniform_probability`` is an
explicit exploration floor, so even low-residual frames keep a non-zero chance
of being revisited.

The sampler never consumes clear-GT membership or metric scores: it only sees
training-view render residuals.  This keeps the paper evaluation labels out of
the replay policy even when an online SLAM observation also belongs to the
published clear-frame reporting subset.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union


FrameId = Union[int, str]

# Online mapping already optimizes two background views per mapping iteration.
# Replay is a replacement policy for those views, not an extra optimization
# budget, so the fair online contract deliberately fixes this count at two.
ONLINE_REPLAY_VIEW_COUNT = 2


def validate_resplat_config(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Validate the replay feature gates without constructing CUDA objects.

    This function is shared by the CLI preflight and ``Mapper`` so an invalid
    online experiment cannot silently fall back to uniform sampling or change
    the number of views optimized per online mapping iteration.
    """

    config = config or {}
    enabled = bool(config.get("enabled", False))
    online_enabled = bool(config.get("online_enabled", False))
    backend = str(config.get("backend", "residual_replay"))

    if enabled and backend != "residual_replay":
        raise ValueError(
            "Only mapping.resplat.backend=residual_replay is implemented"
        )
    if online_enabled and not enabled:
        raise ValueError(
            "mapping.resplat.online_enabled=true requires "
            "mapping.resplat.enabled=true"
        )
    if online_enabled and backend != "residual_replay":
        raise ValueError(
            "mapping.resplat.online_enabled=true requires "
            "mapping.resplat.backend=residual_replay"
        )
    if online_enabled:
        raw_count = config.get("online_replay_views", ONLINE_REPLAY_VIEW_COUNT)
        if isinstance(raw_count, bool):
            raise ValueError(
                "mapping.resplat.online_replay_views must be the integer 2"
            )
        try:
            view_count = int(raw_count)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "mapping.resplat.online_replay_views must be the integer 2"
            ) from error
        try:
            is_exact_integer = float(raw_count) == float(view_count)
        except (TypeError, ValueError):
            is_exact_integer = False
        if not is_exact_integer or view_count != ONLINE_REPLAY_VIEW_COUNT:
            raise ValueError(
                "mapping.resplat.online_replay_views must remain 2 so replay "
                "replaces the two baseline background views instead of changing "
                "the online optimization budget"
            )
    return config


class _FenwickTree:
    """Dynamic Fenwick tree supporting O(log N) weighted draws."""

    def __init__(self) -> None:
        self._tree = [0.0]  # one-indexed
        self._values: list[float] = []

    def __len__(self) -> int:
        return len(self._values)

    def prefix_sum(self, count: int) -> float:
        """Return the sum of the first ``count`` values."""
        count = min(max(0, int(count)), len(self._values))
        result = 0.0
        while count:
            result += self._tree[count]
            count -= count & -count
        return result

    @property
    def total(self) -> float:
        return self.prefix_sum(len(self._values))

    def append(self, value: float) -> None:
        value = _finite_nonnegative(value, "Fenwick value")
        index = len(self._values) + 1
        lowbit = index & -index
        # A newly appended Fenwick node owns the existing suffix
        # [index-lowbit+1, index-1] plus the new value.
        owned_existing = self.prefix_sum(index - 1) - self.prefix_sum(index - lowbit)
        self._values.append(value)
        self._tree.append(owned_existing + value)

    def set(self, index: int, value: float) -> None:
        value = _finite_nonnegative(value, "Fenwick value")
        if not 0 <= index < len(self._values):
            raise IndexError(index)
        delta = value - self._values[index]
        self._values[index] = value
        tree_index = index + 1
        while tree_index < len(self._tree):
            self._tree[tree_index] += delta
            tree_index += tree_index & -tree_index

    def get(self, index: int) -> float:
        return self._values[index]

    def find(self, target: float) -> int:
        """Find the index whose half-open cumulative interval contains target."""
        total = self.total
        if not 0.0 <= target < total:
            raise ValueError(f"target must be in [0, {total}), got {target}")
        index = 0
        accumulated = 0.0
        bit = 1 << (len(self._values).bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate <= len(self._values):
                candidate_sum = accumulated + self._tree[candidate]
                if candidate_sum <= target:
                    index = candidate
                    accumulated = candidate_sum
            bit >>= 1
        return index  # zero-indexed because `index` is the preceding count


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _finite_nonnegative(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _unit(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return value


def _tuple_tree(value: Any) -> Any:
    """Convert JSON-restored RNG state lists back to tuples recursively."""
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration for :class:`ResidualReplaySampler`.

    ``coverage`` observations mean the fraction of the view already explained
    by the map, so priority uses ``1 - coverage``.  Residual and Laplacian gap
    are non-negative and are squashed by their respective scales.  Novelty and
    reliability are confidence-like values in ``[0, 1]``.
    """

    ema_alpha: float = 0.20
    uniform_probability: float = 0.15
    residual_weight: float = 1.00
    laplacian_gap_weight: float = 0.35
    coverage_gap_weight: float = 0.30
    novelty_weight: float = 0.20
    residual_scale: float = 0.10
    laplacian_gap_scale: float = 0.10
    min_priority: float = 1e-6

    def __post_init__(self) -> None:
        _unit(self.ema_alpha, "ema_alpha")
        _unit(self.uniform_probability, "uniform_probability")
        for name in (
            "residual_weight",
            "laplacian_gap_weight",
            "coverage_gap_weight",
            "novelty_weight",
            "min_priority",
        ):
            _finite_nonnegative(getattr(self, name), name)
        for name in ("residual_scale", "laplacian_gap_scale"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass
class ReplayFrameState:
    residual: float = 0.0
    laplacian_gap: float = 0.0
    coverage: float = 0.0
    novelty: float = 1.0
    reliability: float = 1.0
    observations: int = 0
    visits: int = 0
    last_observed_step: Optional[int] = None
    last_sampled_step: Optional[int] = None


class ResidualReplaySampler:
    """Incremental recurrent-residual replay sampler.

    The class deliberately contains no neural network, renderer, image loader,
    or Gaussian mutation.  This keeps the scheduling policy testable on CPU and
    lets Unblur-SLAM retain its native loss and Gaussian representation.
    """

    STATE_SCHEMA = "unblur_slam.resplat_inspired_replay.v1"
    LOG_FIELDS = [
        "event",
        "step",
        "frame_id",
        "mode",
        "residual_ema",
        "laplacian_gap_ema",
        "coverage_ema",
        "novelty_ema",
        "reliability_ema",
        "priority",
        "observations",
        "visits",
    ]

    def __init__(
        self,
        frame_ids: Iterable[FrameId] = (),
        *,
        config: Optional[ReplayConfig] = None,
        seed: int = 2026,
        log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config or ReplayConfig()
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._frame_ids: list[FrameId] = []
        self._index: dict[FrameId, int] = {}
        self._states: list[ReplayFrameState] = []
        self._priority_tree = _FenwickTree()
        self._uniform_tree = _FenwickTree()
        self._event_step = 0
        self.log_path = Path(log_path) if log_path is not None else None
        for frame_id in frame_ids:
            self.register(frame_id)

    def __len__(self) -> int:
        return len(self._frame_ids)

    @property
    def frame_ids(self) -> tuple[FrameId, ...]:
        return tuple(self._frame_ids)

    def register(self, frame_id: FrameId) -> None:
        """Register one training frame. Duplicate registration is a no-op."""
        if not isinstance(frame_id, (int, str)) or isinstance(frame_id, bool):
            raise TypeError("frame_id must be an int or str")
        if frame_id in self._index:
            return
        self._index[frame_id] = len(self._frame_ids)
        self._frame_ids.append(frame_id)
        state = ReplayFrameState()
        self._states.append(state)
        self._priority_tree.append(self._priority(state))
        self._uniform_tree.append(1.0)

    def state_for(self, frame_id: FrameId) -> ReplayFrameState:
        """Return a copy of the current EMA state for diagnostics."""
        index = self._index[frame_id]
        return ReplayFrameState(**asdict(self._states[index]))

    @staticmethod
    def _squash(value: float, scale: float) -> float:
        value = max(0.0, value)
        return value / (value + scale)

    def _priority(self, state: ReplayFrameState) -> float:
        cfg = self.config
        score = (
            cfg.residual_weight * self._squash(state.residual, cfg.residual_scale)
            + cfg.laplacian_gap_weight
            * self._squash(state.laplacian_gap, cfg.laplacian_gap_scale)
            + cfg.coverage_gap_weight * (1.0 - state.coverage)
            + cfg.novelty_weight * state.novelty
        )
        return cfg.min_priority + state.reliability * score

    def priority_for(self, frame_id: FrameId) -> float:
        return self._priority_tree.get(self._index[frame_id])

    def _ema(self, previous: float, current: float, observations: int) -> float:
        if observations == 0:
            return current
        alpha = self.config.ema_alpha
        return (1.0 - alpha) * previous + alpha * current

    def observe(
        self,
        frame_id: FrameId,
        *,
        residual: Optional[float] = None,
        laplacian_gap: Optional[float] = None,
        coverage: Optional[float] = None,
        novelty: Optional[float] = None,
        reliability: Optional[float] = None,
        step: Optional[int] = None,
    ) -> float:
        """Update a frame's EMAs and return its new replay priority.

        Omitted signals retain their previous values.  ``residual`` and
        ``laplacian_gap`` must be non-negative.  Coverage, novelty, and
        reliability are in ``[0, 1]``.
        """
        self.register(frame_id)
        index = self._index[frame_id]
        state = self._states[index]
        observations = state.observations

        if residual is not None:
            value = _finite_nonnegative(residual, "residual")
            state.residual = self._ema(state.residual, value, observations)
        if laplacian_gap is not None:
            value = _finite_nonnegative(laplacian_gap, "laplacian_gap")
            state.laplacian_gap = self._ema(state.laplacian_gap, value, observations)
        if coverage is not None:
            value = _unit(coverage, "coverage")
            state.coverage = self._ema(state.coverage, value, observations)
        if novelty is not None:
            value = _unit(novelty, "novelty")
            state.novelty = self._ema(state.novelty, value, observations)
        if reliability is not None:
            value = _unit(reliability, "reliability")
            state.reliability = self._ema(state.reliability, value, observations)

        state.observations += 1
        event_step = self._resolve_step(step)
        state.last_observed_step = event_step
        priority = self._priority(state)
        self._priority_tree.set(index, priority)
        self._log("observe", event_step, frame_id, "", state, priority)
        return priority

    def _resolve_step(self, step: Optional[int]) -> int:
        if step is None:
            self._event_step += 1
            return self._event_step
        step = int(step)
        if step < 0:
            raise ValueError("step must be non-negative")
        self._event_step = max(self._event_step, step)
        return step

    def _draw_index(self) -> tuple[int, str]:
        if self._uniform_tree.total <= 0.0:
            raise IndexError("cannot sample from an empty replay set")
        use_uniform = (
            self._priority_tree.total <= 0.0
            or self._rng.random() < self.config.uniform_probability
        )
        tree = self._uniform_tree if use_uniform else self._priority_tree
        total = tree.total
        # random() is in [0, 1), matching Fenwick.find's half-open interval.
        return tree.find(self._rng.random() * total), "uniform" if use_uniform else "priority"

    def sample(self, *, step: Optional[int] = None) -> FrameId:
        """Sample one frame in O(log N)."""
        return self.sample_many(1, step=step)[0]

    def sample_many(self, count: int, *, step: Optional[int] = None) -> list[FrameId]:
        """Sample up to ``count`` distinct frames without a full-frame scan."""
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        count = min(count, len(self._frame_ids))
        if count == 0:
            return []

        event_step = self._resolve_step(step)
        selected: list[tuple[int, str, float, float]] = []
        try:
            for _ in range(count):
                index, mode = self._draw_index()
                priority = self._priority_tree.get(index)
                uniform = self._uniform_tree.get(index)
                selected.append((index, mode, priority, uniform))
                # Temporarily remove the item from both distributions to make
                # the batch unique. Each change is O(log N).
                self._priority_tree.set(index, 0.0)
                self._uniform_tree.set(index, 0.0)
        finally:
            for index, _, priority, uniform in selected:
                self._priority_tree.set(index, priority)
                self._uniform_tree.set(index, uniform)

        result = []
        for index, mode, priority, _ in selected:
            state = self._states[index]
            state.visits += 1
            state.last_sampled_step = event_step
            frame_id = self._frame_ids[index]
            self._log("sample", event_step, frame_id, mode, state, priority)
            result.append(frame_id)
        return result

    def sample_many_from(
        self,
        frame_ids: Iterable[FrameId],
        count: int,
        *,
        step: Optional[int] = None,
    ) -> list[FrameId]:
        """Sample distinct frames while restricting a draw to an active set.

        Online mapping changes its current window over time.  Frames in that
        window must not be selected again as background views, even though
        their replay statistics should remain available after they leave the
        window.  The temporary mask preserves those statistics and the normal
        uniform/priority mixture.  In the mapper, only the small current-window
        complement is normally masked; the offline ``sample_many`` path is
        unchanged.
        """

        eligible = []
        eligible_indices = set()
        for frame_id in frame_ids:
            if frame_id not in self._index:
                raise KeyError(f"unregistered replay frame {frame_id!r}")
            index = self._index[frame_id]
            if index in eligible_indices:
                continue
            eligible.append(frame_id)
            eligible_indices.add(index)

        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0 or not eligible:
            return []

        masked: list[tuple[int, float, float]] = []
        try:
            for index in range(len(self._frame_ids)):
                if index in eligible_indices:
                    continue
                priority = self._priority_tree.get(index)
                uniform = self._uniform_tree.get(index)
                masked.append((index, priority, uniform))
                self._priority_tree.set(index, 0.0)
                self._uniform_tree.set(index, 0.0)
            return self.sample_many(min(count, len(eligible)), step=step)
        finally:
            for index, priority, uniform in masked:
                self._priority_tree.set(index, priority)
                self._uniform_tree.set(index, uniform)

    def _log(
        self,
        event: str,
        step: int,
        frame_id: FrameId,
        mode: str,
        state: ReplayFrameState,
        priority: float,
    ) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.log_path.exists() or self.log_path.stat().st_size == 0
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.LOG_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "event": event,
                    "step": step,
                    "frame_id": frame_id,
                    "mode": mode,
                    "residual_ema": f"{state.residual:.9g}",
                    "laplacian_gap_ema": f"{state.laplacian_gap:.9g}",
                    "coverage_ema": f"{state.coverage:.9g}",
                    "novelty_ema": f"{state.novelty:.9g}",
                    "reliability_ema": f"{state.reliability:.9g}",
                    "priority": f"{priority:.9g}",
                    "observations": state.observations,
                    "visits": state.visits,
                }
            )

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable exact continuation state."""
        return {
            "schema": self.STATE_SCHEMA,
            "config": asdict(self.config),
            "seed": self.seed,
            "event_step": self._event_step,
            "frame_ids": self._frame_ids,
            "frames": [asdict(state) for state in self._states],
            "rng_state": self._rng.getstate(),
        }

    @classmethod
    def from_state_dict(
        cls,
        payload: dict[str, Any],
        *,
        log_path: Optional[Union[str, Path]] = None,
    ) -> "ResidualReplaySampler":
        if payload.get("schema") != cls.STATE_SCHEMA:
            raise ValueError(f"Unsupported replay state schema: {payload.get('schema')!r}")
        frame_ids = payload.get("frame_ids", [])
        frames = payload.get("frames", [])
        if len(frame_ids) != len(frames):
            raise ValueError("Replay state frame_ids/frames length mismatch")
        sampler = cls(
            frame_ids,
            config=ReplayConfig(**payload["config"]),
            seed=int(payload["seed"]),
            log_path=log_path,
        )
        sampler._states = [ReplayFrameState(**item) for item in frames]
        sampler._priority_tree = _FenwickTree()
        sampler._uniform_tree = _FenwickTree()
        for state in sampler._states:
            sampler._priority_tree.append(sampler._priority(state))
            sampler._uniform_tree.append(1.0)
        sampler._event_step = int(payload.get("event_step", 0))
        sampler._rng.setstate(_tuple_tree(payload["rng_state"]))
        return sampler

    def save_state(self, path: Union[str, Path]) -> None:
        """Atomically save sampler state as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state_dict(), handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load_state(
        cls,
        path: Union[str, Path],
        *,
        log_path: Optional[Union[str, Path]] = None,
    ) -> "ResidualReplaySampler":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_state_dict(payload, log_path=log_path)


__all__ = [
    "FrameId",
    "ONLINE_REPLAY_VIEW_COUNT",
    "ReplayConfig",
    "ReplayFrameState",
    "ResidualReplaySampler",
    "validate_resplat_config",
]
