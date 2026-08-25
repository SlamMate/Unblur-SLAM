"""Minimal submap contracts for a possible future LoopSplat-style backend.

This file implements boundary decisions, frame membership, checkpoint metadata,
and application of an externally supplied rigid correction.  It intentionally
does **not** implement place recognition, Gaussian registration, loop-edge
verification, pose-graph optimization, or Gaussian merging.  A caller must
provide those algorithms and may use ``gaussian_applier`` to update its native
3DGS tensors after a correction has been accepted.

LoopSplat is an RGB-D SE(3) system.  Monocular Unblur-SLAM may need Sim(3)
corrections instead; this module rejects non-rigid matrices rather than hiding
that research boundary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union


FrameId = Union[int, str]
Matrix4 = tuple[tuple[float, float, float, float], ...]


def _matrix4(value: Sequence[Sequence[float]], *, rigid: bool = False) -> Matrix4:
    if len(value) != 4 or any(len(row) != 4 for row in value):
        raise ValueError("transform must be a 4x4 matrix")
    result = tuple(tuple(float(item) for item in row) for row in value)
    if any(not math.isfinite(item) for row in result for item in row):
        raise ValueError("transform entries must be finite")
    if rigid:
        tolerance = 1e-4
        if any(abs(result[3][column] - expected) > tolerance for column, expected in enumerate((0, 0, 0, 1))):
            raise ValueError("rigid transform bottom row must be [0, 0, 0, 1]")
        rotation = [row[:3] for row in result[:3]]
        for i in range(3):
            for j in range(3):
                dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
                expected = 1.0 if i == j else 0.0
                if abs(dot - expected) > tolerance:
                    raise ValueError("transform rotation must be orthonormal (SE(3), not Sim(3))")
        determinant = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if abs(determinant - 1.0) > tolerance:
            raise ValueError("transform rotation must have determinant +1")
    return result


def _matmul(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


IDENTITY4: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def pose_delta(anchor_c2w: Sequence[Sequence[float]], current_c2w: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Return camera-center distance and geodesic rotation difference in degrees."""
    anchor = _matrix4(anchor_c2w, rigid=True)
    current = _matrix4(current_c2w, rigid=True)
    translation = math.sqrt(sum((current[i][3] - anchor[i][3]) ** 2 for i in range(3)))
    # trace(R_anchor^T R_current), without requiring NumPy or Torch.
    trace = sum(anchor[k][i] * current[k][i] for i in range(3) for k in range(3))
    cosine = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    rotation_deg = math.degrees(math.acos(cosine))
    return translation, rotation_deg


@dataclass(frozen=True)
class BoundaryDecision:
    start_new_submap: bool
    reasons: tuple[str, ...]
    translation_delta: float
    rotation_delta_deg: float


@dataclass(frozen=True)
class SubmapBoundaryPolicy:
    """Motion/keyframe-count boundary policy using estimated poses only."""

    min_keyframes: int = 8
    max_keyframes: int = 64
    translation_threshold: float = 0.5
    rotation_threshold_deg: float = 50.0

    def __post_init__(self) -> None:
        if self.min_keyframes < 1:
            raise ValueError("min_keyframes must be positive")
        if self.max_keyframes < self.min_keyframes:
            raise ValueError("max_keyframes must be >= min_keyframes")
        if self.translation_threshold <= 0.0 or self.rotation_threshold_deg <= 0.0:
            raise ValueError("motion thresholds must be positive")

    def decide(
        self,
        anchor_c2w: Sequence[Sequence[float]],
        current_c2w: Sequence[Sequence[float]],
        keyframe_count: int,
    ) -> BoundaryDecision:
        keyframe_count = int(keyframe_count)
        translation, rotation = pose_delta(anchor_c2w, current_c2w)
        reasons = []
        if keyframe_count >= self.max_keyframes:
            reasons.append("max_keyframes")
        if keyframe_count >= self.min_keyframes:
            if translation >= self.translation_threshold:
                reasons.append("translation")
            if rotation >= self.rotation_threshold_deg:
                reasons.append("rotation")
        return BoundaryDecision(bool(reasons), tuple(reasons), translation, rotation)


@dataclass
class SubmapRecord:
    """JSON-safe membership and checkpoint metadata for one local map."""

    submap_id: int
    anchor_frame_id: FrameId
    anchor_c2w: Matrix4
    local_to_global: Matrix4 = IDENTITY4
    frame_ids: list[FrameId] = field(default_factory=list)
    keyframe_ids: list[FrameId] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    gaussian_count: Optional[int] = None
    closed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.submap_id = int(self.submap_id)
        self.anchor_c2w = _matrix4(self.anchor_c2w, rigid=True)
        self.local_to_global = _matrix4(self.local_to_global, rigid=True)
        self.add_frame(self.anchor_frame_id, is_keyframe=True)

    def add_frame(self, frame_id: FrameId, *, is_keyframe: bool = False) -> None:
        if self.closed:
            raise RuntimeError("cannot add a frame to a closed submap")
        if frame_id not in self.frame_ids:
            self.frame_ids.append(frame_id)
        if is_keyframe and frame_id not in self.keyframe_ids:
            self.keyframe_ids.append(frame_id)

    def close(
        self,
        checkpoint_path: Union[str, Path],
        *,
        gaussian_count: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if gaussian_count is not None and int(gaussian_count) < 0:
            raise ValueError("gaussian_count must be non-negative")
        self.checkpoint_path = str(checkpoint_path)
        self.gaussian_count = None if gaussian_count is None else int(gaussian_count)
        if metadata:
            self.metadata.update(metadata)
        self.closed = True

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return metadata only; no Gaussian tensors are serialized here."""
        return {
            "schema": "unblur_slam.submap_metadata.v1",
            "submap_id": self.submap_id,
            "anchor_frame_id": self.anchor_frame_id,
            "anchor_c2w": self.anchor_c2w,
            "local_to_global": self.local_to_global,
            "frame_ids": self.frame_ids,
            "keyframe_ids": self.keyframe_ids,
            "checkpoint_path": self.checkpoint_path,
            "gaussian_count": self.gaussian_count,
            "closed": self.closed,
            "metadata": self.metadata,
            "registration_implemented": False,
        }

    def save_checkpoint_metadata(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.checkpoint_metadata(), handle, indent=2)
            handle.write("\n")


@dataclass(frozen=True)
class RigidCorrection:
    """An already verified external SE(3) correction for one submap."""

    submap_id: int
    transform: Matrix4
    source: str = "external"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "submap_id", int(self.submap_id))
        object.__setattr__(self, "transform", _matrix4(self.transform, rigid=True))
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        object.__setattr__(self, "confidence", confidence)


GaussianCorrectionApplier = Callable[[SubmapRecord, Matrix4], None]


def apply_rigid_correction(
    record: SubmapRecord,
    correction: RigidCorrection,
    *,
    gaussian_applier: Optional[GaussianCorrectionApplier] = None,
) -> Matrix4:
    """Apply an accepted correction to metadata and optionally native tensors.

    ``gaussian_applier`` is the only bridge to a scene representation.  It must
    rotate/translate means and rotations consistently; this module does not
    pretend to provide LoopSplat's Gaussian registration.
    """
    if record.submap_id != correction.submap_id:
        raise ValueError("correction submap_id does not match record")
    if gaussian_applier is not None:
        gaussian_applier(record, correction.transform)
    record.local_to_global = _matmul(correction.transform, record.local_to_global)
    history = record.metadata.setdefault("rigid_corrections", [])
    history.append(
        {
            "source": correction.source,
            "confidence": correction.confidence,
            "transform": correction.transform,
        }
    )
    return record.local_to_global


__all__ = [
    "BoundaryDecision",
    "GaussianCorrectionApplier",
    "IDENTITY4",
    "RigidCorrection",
    "SubmapBoundaryPolicy",
    "SubmapRecord",
    "apply_rigid_correction",
    "pose_delta",
]
