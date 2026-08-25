"""Fail-closed contracts for asynchronous official cvg/ReSplat sidecars.

The sidecar consumes a *closed* submap snapshot containing exactly eight
past-only mapper keyframes.  Work is handed to a fresh official-ReSplat Python
process through files, never through CUDA tensors or the active
``GaussianModel``.  A completed native ReSplat scene is published atomically
only after snapshot binding, pose-staleness, runtime, and geometry gates pass.

This module publishes a rigorously converted snapshot-world candidate, but the
sidecar queue itself deliberately has no unconditional active-map mutation
path.  ReSplat and Unblur still have different topology, ownership and
optimizer-state contracts.  Requesting injection through this queue therefore
raises ``UnsupportedActiveMapMerge``; a mapper must apply the separate
ownership/staleness/quality gates before using the converted payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from src.submaps import pose_delta


FrameId = Union[int, str]
SNAPSHOT_SCHEMA = "unblur_slam.official_resplat_closed_submap_snapshot.v1"
RESULT_SCHEMA = "unblur_slam.official_resplat_closed_submap_result.v1"
GATE_SCHEMA = "unblur_slam.official_resplat_sidecar_gate.v1"
QUEUE_EVENT_SCHEMA = "unblur_slam.official_resplat_sidecar_queue_event.v1"
OFFICIAL_CONTEXT_KEYFRAMES = 8
OFFICIAL_REFINEMENT_UPDATES = 4
MIN_OFFICIAL_REFINEMENT_UPDATES = 1
MAX_OFFICIAL_REFINEMENT_UPDATES = 4
OFFICIAL_PRESET = "dl3dv_8v_256x448_small"
OFFICIAL_LATENT_DOWNSAMPLE = 4


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _matrix4(value: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if len(value) != 4 or any(len(row) != 4 for row in value):
        raise ValueError(f"{name} must be a 4x4 matrix")
    result = [[float(item) for item in row] for row in value]
    if any(not math.isfinite(item) for row in result for item in row):
        raise ValueError(f"{name} must contain finite values")
    expected_bottom = (0.0, 0.0, 0.0, 1.0)
    if any(abs(result[3][i] - expected_bottom[i]) > 1e-4 for i in range(4)):
        raise ValueError(f"{name} bottom row must be [0,0,0,1]")
    return result


def _matrix3(value: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if len(value) != 3 or any(len(row) != 3 for row in value):
        raise ValueError(f"{name} must be a 3x3 matrix")
    result = [[float(item) for item in row] for row in value]
    if any(not math.isfinite(item) for row in result for item in row):
        raise ValueError(f"{name} must contain finite values")
    if result[0][0] <= 0.0 or result[1][1] <= 0.0:
        raise ValueError(f"{name} focal lengths must be positive")
    if any(abs(result[2][i] - expected) > 1e-6 for i, expected in enumerate((0, 0, 1))):
        raise ValueError(f"{name} bottom row must be [0,0,1]")
    return result


def pose_hash(c2w: Sequence[Sequence[float]]) -> str:
    matrix = _matrix4(c2w, "c2w")
    # Reuse the submap SE(3) validator; hashing a finite affine/Sim(3) matrix
    # would otherwise disguise an invalid pose as a valid immutable pose.
    pose_delta(matrix, matrix)
    return hashlib.sha256(_canonical_json(matrix)).hexdigest()


@dataclass(frozen=True)
class SidecarFrameInput:
    """One online mapper keyframe before filesystem materialization."""

    frame_id: FrameId
    sequence_ordinal: int
    c2w: Sequence[Sequence[float]]
    intrinsics_px: Sequence[Sequence[float]]
    image: Any


@dataclass(frozen=True)
class SidecarConfig:
    """Validated sidecar settings; enabling it never enables map injection."""

    enabled: bool = False
    mode: str = "sidecar_only"
    context_keyframes: int = OFFICIAL_CONTEXT_KEYFRAMES
    queue_capacity: int = 1
    output_root: str = ""
    python_executable: str = ""
    runner_script: str = ""
    resplat_repo: str = ""
    checkpoint: str = ""
    expected_checkpoint_sha256: str = ""
    model_preset: str = OFFICIAL_PRESET
    refinement_updates: int = OFFICIAL_REFINEMENT_UPDATES
    cuda_visible_devices: str = ""
    process_device: str = "cuda:0"
    near: float = 0.01
    far: float = 200.0
    max_runtime_seconds: float = 30.0
    final_drain_timeout_seconds: float = 30.0
    max_pose_revision_lag: int = 10000
    max_pose_translation_drift: float = 0.05
    max_pose_rotation_drift_deg: float = 2.0
    min_gaussian_count: int = 1
    max_gaussian_count: int = 2_000_000
    min_finite_fraction: float = 1.0
    max_p95_distance: float = 50.0
    max_distance: float = 200.0
    max_p95_scale: float = 5.0
    max_scale: float = 25.0
    max_quaternion_norm_deviation: float = 1e-3
    active_map_merge: bool = False

    def __post_init__(self) -> None:
        if self.mode != "sidecar_only":
            raise ValueError("official ReSplat online mode must be sidecar_only")
        if self.context_keyframes != OFFICIAL_CONTEXT_KEYFRAMES:
            raise ValueError("official small8v ReSplat requires exactly 8 keyframes")
        if self.model_preset != OFFICIAL_PRESET:
            raise ValueError(
                f"online contract is pinned to official preset {OFFICIAL_PRESET}"
            )
        if (
            isinstance(self.refinement_updates, bool)
            or not MIN_OFFICIAL_REFINEMENT_UPDATES
            <= int(self.refinement_updates)
            <= MAX_OFFICIAL_REFINEMENT_UPDATES
        ):
            raise ValueError(
                "refinement_updates must request between 1 and 4 actual recurrent states"
            )
        if self.active_map_merge:
            raise UnsupportedActiveMapMerge(
                "active-map merge is unsupported; publish native sidecars only"
            )
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        if not 0.0 < self.near < self.far:
            raise ValueError("near/far must satisfy 0 < near < far")
        if not 0.0 < self.max_runtime_seconds <= 3600.0:
            raise ValueError("max_runtime_seconds must be in (0,3600]")
        if not 0.0 <= self.final_drain_timeout_seconds <= 60.0:
            raise ValueError("final_drain_timeout_seconds must be in [0,60]")
        if self.max_pose_revision_lag < 0:
            raise ValueError("max_pose_revision_lag must be non-negative")
        if self.max_pose_translation_drift < 0.0:
            raise ValueError("max_pose_translation_drift must be non-negative")
        if self.max_pose_rotation_drift_deg < 0.0:
            raise ValueError("max_pose_rotation_drift_deg must be non-negative")
        if not 0.0 <= self.min_finite_fraction <= 1.0:
            raise ValueError("min_finite_fraction must be in [0,1]")
        if self.min_gaussian_count < 1 or self.max_gaussian_count < self.min_gaussian_count:
            raise ValueError("invalid Gaussian-count gate")
        for name in (
            "max_p95_distance",
            "max_distance",
            "max_p95_scale",
            "max_scale",
            "max_quaternion_norm_deviation",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.max_distance < self.max_p95_distance:
            raise ValueError("max_distance must be >= max_p95_distance")
        if self.max_scale < self.max_p95_scale:
            raise ValueError("max_scale must be >= max_p95_scale")
        if self.enabled:
            required = {
                "output_root": self.output_root,
                "python_executable": self.python_executable,
                "runner_script": self.runner_script,
                "resplat_repo": self.resplat_repo,
                "checkpoint": self.checkpoint,
                "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
                "cuda_visible_devices": self.cuda_visible_devices,
            }
            missing = [name for name, value in required.items() if not str(value)]
            if missing:
                raise ValueError(
                    "enabled official ReSplat sidecar is missing: " + ", ".join(missing)
                )
            digest = self.expected_checkpoint_sha256.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("expected_checkpoint_sha256 must be 64 lowercase hex")

    @classmethod
    def from_dict(
        cls, value: Optional[Mapping[str, Any]], *, default_output_root: Path | str
    ) -> "SidecarConfig":
        raw = dict(value or {})
        if not raw.get("output_root"):
            raw["output_root"] = str(Path(default_output_root))
        return cls(**raw)


class UnsupportedActiveMapMerge(RuntimeError):
    """Raised whenever a caller tries to inject a native ReSplat scene."""


def active_map_merge_assessment() -> dict[str, Any]:
    """Return why direct queue-side append remains disabled after conversion."""

    return {
        "supported": False,
        "decision": "reject",
        "reason_codes": [
            "fixed_pixel_aligned_resplat_topology_not_arbitrary_unblur_topology",
            "no_unblur_keyframe_ownership",
            "no_unblur_optimizer_state",
            "no_active_map_version_binding_or_conflict_resolution",
        ],
        "coordinate_conversion_available": True,
        "safe_use": "verified_snapshot_world_candidate_then_mapper_owned_merge_gate",
    }


def reject_active_map_merge() -> None:
    assessment = active_map_merge_assessment()
    raise UnsupportedActiveMapMerge(
        "official ReSplat native output cannot be merged into the active Unblur "
        "map: " + ", ".join(assessment["reason_codes"])
    )


def _image_to_rgb_uint8(image: Any) -> "Any":
    import numpy as np

    if hasattr(image, "detach"):
        image = image.detach().float().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("sidecar image must be HWC or CHW RGB")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError("sidecar image must have exactly three channels")
    if not np.isfinite(array).all():
        raise ValueError("sidecar image contains non-finite values")
    if np.issubdtype(array.dtype, np.floating):
        if float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6:
            raise ValueError("floating sidecar image must be in [0,1]")
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0)
    else:
        if int(array.min()) < 0 or int(array.max()) > 255:
            raise ValueError("integer sidecar image must be in [0,255]")
    return np.ascontiguousarray(array.astype(np.uint8))


def _validate_snapshot_data_provenance(payload: Mapping[str, Any]) -> None:
    """Allow conditioned membership only when every forbidden GT input is explicit.

    The fixed-11KF experiment inherited a schedule selected by a historical
    clear-image protocol.  That fact is metadata about membership, not license
    for the sidecar to consume clear pixels, GT pose/depth, or clear-GT metrics.
    """

    legacy_pose = bool(payload.get("uses_ground_truth_pose", True))
    if legacy_pose or bool(payload.get("uses_ground_truth_pose_or_depth", legacy_pose)):
        raise ValueError("ground-truth pose/depth is forbidden in online sidecars")
    if bool(payload.get("uses_independent_clear_pixels", False)):
        raise ValueError("independent clear pixels are forbidden in online sidecars")
    if bool(payload.get("uses_clear_gt_metrics", False)):
        raise ValueError("clear-GT metrics are forbidden in online sidecars")

    membership = bool(payload.get("uses_clear_gt_membership", True))
    disclosed = payload.get(
        "selection_membership_clear_gt_conditioned", membership
    )
    if disclosed is not membership:
        raise ValueError("clear-GT-conditioned membership disclosure mismatch")
    if not membership:
        return
    if payload.get("integration_mode") != "online_mapper":
        raise ValueError("conditioned membership is allowed only for online_mapper")
    provenance = payload.get("source_provenance") or {}
    expected = {
        "selection_membership_clear_gt_conditioned": True,
        "uses_ground_truth_pose_or_depth": False,
        "uses_independent_clear_pixels": False,
        "uses_clear_gt_metrics": False,
    }
    for key, value in expected.items():
        if provenance.get(key) is not value:
            raise ValueError(f"conditioned-membership provenance {key} drifted")


def materialize_closed_submap_snapshot(
    *,
    snapshots_root: Path | str,
    submap_id: int,
    record_keyframe_ids: Sequence[FrameId],
    frames: Sequence[SidecarFrameInput],
    closure_sequence_ordinal: int,
    pose_revision: int,
    integration_mode: str = "online_mapper",
    selection_source: str = "online_mapper_closed_submap_membership",
    source_provenance: Optional[Mapping[str, Any]] = None,
    uses_clear_gt_membership: bool = False,
    uses_independent_clear_pixels: bool = False,
) -> Path:
    """Atomically persist the latest eight eligible frames of a closed submap.

    Selection uses mapper membership and sequence order only.  A caller may
    truthfully disclose that this membership came from a historically
    clear-GT-conditioned frozen schedule; no clear pixels, GT pose/depth,
    clear-GT metric, image-quality score, or future frame is accepted here.
    """

    from PIL import Image

    allowed_modes = {
        "online_mapper": "online_mapper_closed_submap_membership",
        "independent_queue_smoke": "droid_motion_filter_first_closed_8kf_prefix",
    }
    if integration_mode not in allowed_modes:
        raise ValueError(f"unsupported sidecar integration_mode: {integration_mode}")
    if selection_source != allowed_modes[integration_mode]:
        raise ValueError("selection_source does not match sidecar integration_mode")
    if integration_mode == "independent_queue_smoke" and not source_provenance:
        raise ValueError("independent queue smoke requires bound source provenance")
    root = Path(snapshots_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    membership = set(record_keyframe_ids)
    if len(membership) < OFFICIAL_CONTEXT_KEYFRAMES:
        raise ValueError("closed submap has fewer than 8 unique keyframes")
    eligible = [frame for frame in frames if frame.frame_id in membership]
    eligible.sort(key=lambda item: int(item.sequence_ordinal))
    if len({int(item.sequence_ordinal) for item in eligible}) != len(eligible):
        raise ValueError("sidecar sequence ordinals must be unique")
    if len({item.frame_id for item in eligible}) != len(eligible):
        raise ValueError("sidecar frame ids must be unique")
    if any(int(item.sequence_ordinal) > int(closure_sequence_ordinal) for item in eligible):
        raise ValueError("future keyframe entered a past-only closed-submap snapshot")
    selected = eligible[-OFFICIAL_CONTEXT_KEYFRAMES:]
    if len(selected) != OFFICIAL_CONTEXT_KEYFRAMES:
        raise ValueError("exactly 8 materializable closed-submap keyframes are required")

    staging = Path(tempfile.mkdtemp(prefix=".snapshot.staging.", dir=str(root)))
    installed = False
    try:
        image_dir = staging / "images"
        image_dir.mkdir()
        records: list[dict[str, Any]] = []
        for index, frame in enumerate(selected):
            rgb = _image_to_rgb_uint8(frame.image)
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            filename = f"{index:02d}_frame_{str(frame.frame_id)}.png"
            path = image_dir / filename
            Image.fromarray(rgb, mode="RGB").save(path)
            c2w = _matrix4(frame.c2w, f"frame {frame.frame_id} c2w")
            intrinsics_px = _matrix3(
                frame.intrinsics_px, f"frame {frame.frame_id} intrinsics"
            )
            intrinsics_normalized = [row[:] for row in intrinsics_px]
            intrinsics_normalized[0][0] /= width
            intrinsics_normalized[0][2] /= width
            intrinsics_normalized[1][1] /= height
            intrinsics_normalized[1][2] /= height
            records.append(
                {
                    "frame_id": frame.frame_id,
                    "sequence_ordinal": int(frame.sequence_ordinal),
                    "image_path": f"images/{filename}",
                    "image_sha256": sha256_file(path),
                    "image_size_wh": [width, height],
                    "c2w_opencv": c2w,
                    "pose_hash": pose_hash(c2w),
                    "intrinsics_px": intrinsics_px,
                    "intrinsics_normalized": intrinsics_normalized,
                }
            )
        payload: dict[str, Any] = {
            "schema": SNAPSHOT_SCHEMA,
            "artifact_class": "past_only_closed_submap_8kf",
            "integration_mode": integration_mode,
            "submap_id": int(submap_id),
            "closed": True,
            "selection_source": selection_source,
            "uses_ground_truth_pose": False,
            "uses_ground_truth_pose_or_depth": False,
            "uses_clear_gt_membership": bool(uses_clear_gt_membership),
            "selection_membership_clear_gt_conditioned": bool(
                uses_clear_gt_membership
            ),
            "uses_independent_clear_pixels": bool(uses_independent_clear_pixels),
            "uses_clear_gt_metrics": False,
            "context_keyframes": OFFICIAL_CONTEXT_KEYFRAMES,
            "closure_sequence_ordinal": int(closure_sequence_ordinal),
            "pose_revision": int(pose_revision),
            "frames": records,
            "active_map_state_included": False,
            "source_provenance": dict(source_provenance or {}),
        }
        _validate_snapshot_data_provenance(payload)
        digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
        payload["snapshot_sha256"] = digest
        payload["snapshot_id"] = f"submap-{int(submap_id):04d}-{digest[:16]}"
        _atomic_write_json(staging / "snapshot_manifest.json", payload)
        destination = root / str(payload["snapshot_id"])
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"snapshot already exists: {destination}")
        os.rename(staging, destination)
        installed = True
        return destination
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)


def load_snapshot(snapshot_dir: Path | str, *, verify_images: bool = True) -> dict[str, Any]:
    root = Path(snapshot_dir).expanduser().resolve()
    path = root / "snapshot_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid sidecar snapshot manifest: {path}") from error
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("wrong official ReSplat sidecar snapshot schema")
    mode = payload.get("integration_mode")
    allowed_modes = {
        "online_mapper": "online_mapper_closed_submap_membership",
        "independent_queue_smoke": "droid_motion_filter_first_closed_8kf_prefix",
    }
    if mode not in allowed_modes or payload.get("selection_source") != allowed_modes[mode]:
        raise ValueError("snapshot integration mode/selection source mismatch")
    if mode == "independent_queue_smoke" and not payload.get("source_provenance"):
        raise ValueError("independent queue snapshot lacks source provenance")
    _validate_snapshot_data_provenance(payload)
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != OFFICIAL_CONTEXT_KEYFRAMES:
        raise ValueError("snapshot must contain exactly 8 frames")
    unsigned = dict(payload)
    expected = str(unsigned.pop("snapshot_sha256", ""))
    unsigned.pop("snapshot_id", None)
    actual = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("snapshot manifest SHA-256 binding failed")
    if payload.get("snapshot_id") != f"submap-{int(payload['submap_id']):04d}-{actual[:16]}":
        raise ValueError("snapshot_id does not match snapshot digest")
    closure = int(payload["closure_sequence_ordinal"])
    ordinals = [int(frame["sequence_ordinal"]) for frame in frames]
    if ordinals != sorted(ordinals) or len(set(ordinals)) != len(ordinals):
        raise ValueError("snapshot frames are not in unique sequence order")
    if any(value > closure for value in ordinals):
        raise ValueError("snapshot is not past-only")
    for frame in frames:
        c2w = _matrix4(frame["c2w_opencv"], "snapshot c2w")
        if pose_hash(c2w) != frame.get("pose_hash"):
            raise ValueError("snapshot pose hash mismatch")
        relative = Path(str(frame["image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("snapshot image path must remain inside the snapshot")
        image_path = root / relative
        if verify_images and sha256_file(image_path) != frame.get("image_sha256"):
            raise ValueError("snapshot image hash mismatch")
    return payload


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reasons: tuple[str, ...]
    measurements: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GATE_SCHEMA,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "measurements": dict(self.measurements),
            "active_map_merge": active_map_merge_assessment(),
        }


def evaluate_staleness(
    snapshot: Mapping[str, Any],
    *,
    current_poses: Mapping[FrameId, Sequence[Sequence[float]]],
    current_pose_revision: int,
    config: SidecarConfig,
) -> GateDecision:
    reasons: list[str] = []
    revision_lag = int(current_pose_revision) - int(snapshot["pose_revision"])
    if revision_lag < 0:
        reasons.append("current_pose_revision_precedes_snapshot")
    if revision_lag > config.max_pose_revision_lag:
        reasons.append("pose_revision_lag_exceeded")
    max_translation = 0.0
    max_rotation = 0.0
    changed_pose_hashes = 0
    for frame in snapshot["frames"]:
        frame_id = frame["frame_id"]
        if frame_id not in current_poses:
            reasons.append(f"missing_current_pose:{frame_id}")
            continue
        current = _matrix4(current_poses[frame_id], f"current pose {frame_id}")
        current_hash = pose_hash(current)
        if current_hash == frame["pose_hash"]:
            # An identical immutable matrix has exactly zero drift.  Avoid a
            # small false geodesic angle when a float32 DROID rotation is only
            # approximately orthonormal within the accepted SE(3) tolerance.
            translation, rotation = 0.0, 0.0
        else:
            translation, rotation = pose_delta(frame["c2w_opencv"], current)
        max_translation = max(max_translation, translation)
        max_rotation = max(max_rotation, rotation)
        if current_hash != frame["pose_hash"]:
            changed_pose_hashes += 1
    if max_translation > config.max_pose_translation_drift:
        reasons.append("pose_translation_drift_exceeded")
    if max_rotation > config.max_pose_rotation_drift_deg:
        reasons.append("pose_rotation_drift_exceeded")
    return GateDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        measurements={
            "pose_revision_lag": revision_lag,
            "changed_pose_hash_count": changed_pose_hashes,
            "max_translation_drift": max_translation,
            "max_rotation_drift_deg": max_rotation,
        },
    )


def evaluate_result(
    result: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any],
    elapsed_seconds: float,
    config: SidecarConfig,
) -> GateDecision:
    reasons: list[str] = []
    if result.get("schema") != RESULT_SCHEMA:
        reasons.append("wrong_result_schema")
    if result.get("snapshot_sha256") != snapshot.get("snapshot_sha256"):
        reasons.append("snapshot_digest_mismatch")
    if result.get("snapshot_id") != snapshot.get("snapshot_id"):
        reasons.append("snapshot_id_mismatch")
    if result.get("artifact_class") != "native_official_resplat_closed_submap_sidecar":
        reasons.append("wrong_artifact_class")
    if result.get("integration_mode") != snapshot.get("integration_mode"):
        reasons.append("integration_mode_mismatch")
    if not bool(result.get("past_only", False)):
        reasons.append("result_not_declared_past_only")
    if bool(result.get("future_views_used", True)):
        reasons.append("future_views_were_used")
    if bool(result.get("ground_truth_used", True)):
        reasons.append("ground_truth_was_used")
    conditioned_membership = bool(snapshot.get("uses_clear_gt_membership", False))
    if result.get(
        "selection_membership_clear_gt_conditioned", False
    ) is not conditioned_membership:
        reasons.append("conditioned_membership_disclosure_mismatch")
    if bool(result.get("ground_truth_pose_or_depth_used", conditioned_membership)):
        reasons.append("ground_truth_pose_or_depth_was_used")
    if bool(result.get("independent_clear_pixels_used", conditioned_membership)):
        reasons.append("independent_clear_pixels_were_used")
    if bool(result.get("clear_gt_metrics_used", conditioned_membership)):
        reasons.append("clear_gt_metrics_were_used")
    if int(result.get("source_pose_revision", -1)) != int(snapshot["pose_revision"]):
        reasons.append("source_pose_revision_mismatch")
    expected_pose_hashes = [frame["pose_hash"] for frame in snapshot["frames"]]
    if result.get("source_pose_hashes") != expected_pose_hashes:
        reasons.append("source_pose_hashes_mismatch")
    official = result.get("official_resplat") or {}
    if official.get("model_preset") != OFFICIAL_PRESET:
        reasons.append("wrong_official_model_preset")
    if int(official.get("num_context", -1)) != OFFICIAL_CONTEXT_KEYFRAMES:
        reasons.append("wrong_context_count")
    if int(official.get("num_refine", -1)) != int(config.refinement_updates):
        reasons.append("wrong_refinement_count")
    repository = official.get("repository") or {}
    expected_origin = str(repository.get("expected_origin", "")).rstrip("/").lower()
    if expected_origin != "https://github.com/cvg/resplat":
        reasons.append("not_bound_to_official_cvg_resplat")
    if not bool(repository.get("tracked_worktree_clean", False)):
        reasons.append("official_resplat_checkout_not_clean")
    checkpoint = official.get("checkpoint") or {}
    if checkpoint.get("sha256") != config.expected_checkpoint_sha256:
        reasons.append("official_checkpoint_digest_mismatch")
    contract = result.get("execution_contract") or {}
    if int(contract.get("encoder_forward_calls", -1)) != 1:
        reasons.append("official_initializer_call_count_not_one")
    if int(contract.get("forward_update_calls", -1)) != 1:
        reasons.append("official_forward_update_call_count_not_one")
    if not bool(contract.get("init_object_passed_directly", False)):
        reasons.append("official_init_object_not_passed_directly")
    if config.refinement_updates != OFFICIAL_REFINEMENT_UPDATES or (
        "requested_recurrent_updates" in contract
    ):
        requested = int(contract.get("requested_recurrent_updates", -1))
        returned = int(contract.get("returned_recurrent_states", -1))
        selected = int(contract.get("selected_state_index_zero_based", -1))
        if requested != int(config.refinement_updates):
            reasons.append("requested_recurrent_update_count_mismatch")
        if returned != int(config.refinement_updates):
            reasons.append("returned_recurrent_state_count_mismatch")
        if selected != int(config.refinement_updates) - 1:
            reasons.append("selected_recurrent_state_index_mismatch")
        if bool(contract.get("fourth_state_computed", True)) != (
            int(config.refinement_updates) >= 4
        ):
            reasons.append("fourth_state_execution_disclosure_mismatch")
    if bool(result.get("active_map_merge_performed", True)):
        reasons.append("active_map_merge_was_attempted")
    conversion_performed = bool(
        result.get("native_to_unblur_conversion_performed", False)
    )
    if conversion_performed and not isinstance(
        result.get("unblur_world_artifact"), Mapping
    ):
        reasons.append("unblur_world_conversion_missing_contract")
    local_contract = result.get("local_coordinate_contract") or {}
    if bool(local_contract.get("safe_for_active_unblur_map_merge", True)):
        reasons.append("native_local_state_misdeclared_merge_safe")
    if float(elapsed_seconds) > config.max_runtime_seconds:
        reasons.append("runtime_gate_exceeded")

    geometry = result.get("geometry") or {}
    count = int(geometry.get("gaussian_count", -1))
    image_shape = result.get("image_shape_hw")
    expected_topology_count = -1
    if (
        not isinstance(image_shape, list)
        or len(image_shape) != 2
        or any(isinstance(value, bool) for value in image_shape)
    ):
        reasons.append("invalid_official_image_shape")
    else:
        height, width = (int(image_shape[0]), int(image_shape[1]))
        if (
            height < OFFICIAL_LATENT_DOWNSAMPLE
            or width < OFFICIAL_LATENT_DOWNSAMPLE
            or height % 64
            or width % 64
        ):
            reasons.append("invalid_official_image_shape")
        else:
            expected_topology_count = (
                OFFICIAL_CONTEXT_KEYFRAMES
                * (height // OFFICIAL_LATENT_DOWNSAMPLE)
                * (width // OFFICIAL_LATENT_DOWNSAMPLE)
            )
            if count != expected_topology_count:
                reasons.append("fixed_official_topology_count_mismatch")
    finite_fraction = float(geometry.get("finite_fraction", -1.0))
    p95_distance = float(geometry.get("p95_distance_from_local_origin", math.inf))
    max_distance = float(geometry.get("max_distance_from_local_origin", math.inf))
    p95_scale = float(geometry.get("p95_scale", math.inf))
    max_scale = float(geometry.get("max_scale", math.inf))
    quaternion_deviation = float(
        geometry.get("max_quaternion_norm_deviation", math.inf)
    )
    numeric = (
        finite_fraction,
        p95_distance,
        max_distance,
        p95_scale,
        max_scale,
        quaternion_deviation,
    )
    if not all(math.isfinite(value) for value in numeric):
        reasons.append("non_finite_geometry_summary")
    if not config.min_gaussian_count <= count <= config.max_gaussian_count:
        reasons.append("gaussian_count_out_of_range")
    if finite_fraction < config.min_finite_fraction:
        reasons.append("finite_fraction_below_gate")
    if p95_distance > config.max_p95_distance:
        reasons.append("p95_distance_gate_exceeded")
    if max_distance > config.max_distance:
        reasons.append("max_distance_gate_exceeded")
    if p95_scale > config.max_p95_scale:
        reasons.append("p95_scale_gate_exceeded")
    if max_scale > config.max_scale:
        reasons.append("max_scale_gate_exceeded")
    if quaternion_deviation > config.max_quaternion_norm_deviation:
        reasons.append("quaternion_norm_gate_exceeded")
    return GateDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        measurements={
            "elapsed_seconds": float(elapsed_seconds),
            "gaussian_count": count,
            "expected_fixed_topology_count": expected_topology_count,
            "finite_fraction": finite_fraction,
            "p95_distance_from_local_origin": p95_distance,
            "max_distance_from_local_origin": max_distance,
            "p95_scale": p95_scale,
            "max_scale": max_scale,
            "max_quaternion_norm_deviation": quaternion_deviation,
        },
    )


def verify_result_artifacts(
    result: Mapping[str, Any], result_root: Path | str
) -> GateDecision:
    """Bind the published manifest to an immutable native Gaussian payload."""

    reasons: list[str] = []
    outputs = result.get("outputs") or {}
    relative = Path(str(outputs.get("native_gaussians_npz", "")))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        reasons.append("invalid_native_gaussian_artifact_path")
        return GateDecision(False, tuple(reasons))
    artifact = Path(result_root).resolve() / relative
    if not artifact.is_file():
        reasons.append("native_gaussian_artifact_missing")
        return GateDecision(False, tuple(reasons), {"path": str(artifact)})
    expected = str(outputs.get("native_gaussians_npz_sha256", ""))
    actual = sha256_file(artifact)
    if actual != expected:
        reasons.append("native_gaussian_artifact_sha256_mismatch")
    arrays = outputs.get("npz_arrays") or {}
    conversion_performed = bool(
        result.get("native_to_unblur_conversion_performed", False)
    )
    required = {"means", "scales", "rotations", "harmonics", "opacities"}
    if conversion_performed:
        required.add("covariances")
    if set(arrays) != required:
        reasons.append("native_gaussian_array_contract_incomplete")
    actual_geometry: dict[str, Any] = {}
    try:
        import numpy as np

        with np.load(artifact, allow_pickle=False) as archive:
            if set(archive.files) != required:
                reasons.append("native_gaussian_npz_arrays_incomplete")
            else:
                loaded = {name: np.asarray(archive[name]) for name in required}
                means = loaded["means"]
                scales = loaded["scales"]
                rotations = loaded["rotations"]
                harmonics = loaded["harmonics"]
                opacities = loaded["opacities"]
                covariances = loaded.get("covariances")
                count = int(means.shape[0]) if means.ndim == 2 else -1
                valid_shapes = (
                    means.ndim == 2
                    and means.shape[1] == 3
                    and scales.shape == means.shape
                    and rotations.shape == (count, 4)
                    and harmonics.ndim == 3
                    and harmonics.shape[:2] == (count, 3)
                    and opacities.shape in ((count,), (count, 1))
                    and (
                        not conversion_performed
                        or (
                            covariances is not None
                            and covariances.shape == (count, 3, 3)
                        )
                    )
                )
                if not valid_shapes:
                    reasons.append("native_gaussian_npz_shape_contract_failed")
                else:
                    if not bool(np.all(scales > 0.0)):
                        reasons.append("native_gaussian_scale_not_positive")
                    if not bool(
                        np.all((opacities >= 0.0) & (opacities <= 1.0))
                    ):
                        reasons.append("native_gaussian_opacity_out_of_range")
                    if conversion_performed and not bool(
                        np.allclose(
                            covariances,
                            np.swapaxes(covariances, -1, -2),
                            rtol=1e-5,
                            atol=1e-6,
                        )
                    ):
                        reasons.append("native_gaussian_covariance_not_symmetric")
                    for name, value in loaded.items():
                        declaration = arrays.get(name) or {}
                        if declaration.get("shape") != list(value.shape) or declaration.get(
                            "dtype"
                        ) != str(value.dtype):
                            reasons.append(
                                f"native_gaussian_array_declaration_mismatch:{name}"
                            )
                    values = tuple(loaded.values())
                    total = sum(int(value.size) for value in values)
                    finite = sum(int(np.isfinite(value).sum()) for value in values)
                    distances = np.linalg.norm(means.astype(np.float64), axis=-1)
                    scale_values = scales.astype(np.float64)
                    quaternion_norms = np.linalg.norm(
                        rotations.astype(np.float64), axis=-1
                    )
                    actual_geometry = {
                        "gaussian_count": count,
                        "finite_fraction": float(finite / total) if total else 0.0,
                        "p95_distance_from_local_origin": float(
                            np.quantile(distances, 0.95)
                        ),
                        "max_distance_from_local_origin": float(np.max(distances)),
                        "p95_scale": float(np.quantile(scale_values, 0.95)),
                        "max_scale": float(np.max(scale_values)),
                        "max_quaternion_norm_deviation": float(
                            np.max(np.abs(quaternion_norms - 1.0))
                        ),
                    }
                    declared_geometry = result.get("geometry") or {}
                    for name, actual_value in actual_geometry.items():
                        try:
                            declared_value = float(declared_geometry[name])
                        except (KeyError, TypeError, ValueError):
                            reasons.append(f"geometry_summary_missing:{name}")
                            continue
                        if not math.isclose(
                            declared_value,
                            float(actual_value),
                            rel_tol=1e-5,
                            abs_tol=1e-6,
                        ):
                            reasons.append(f"geometry_summary_mismatch:{name}")
    except (OSError, ValueError) as error:
        reasons.append(f"native_gaussian_npz_unreadable:{type(error).__name__}")
    return GateDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        measurements={
            "artifact_relative_path": str(relative),
            "sha256": actual,
            "geometry_recomputed_from_npz": actual_geometry,
        },
    )


def verify_unblur_world_artifact(
    result: Mapping[str, Any],
    result_root: Path | str,
    snapshot: Mapping[str, Any],
    expected_refinement_updates: int = 3,
) -> GateDecision:
    """Verify the coordinate-converted state before any active-map mutation.

    This gate independently recomputes the middle-pivot mean/covariance/SH
    transforms from the native artifact, checks the Unblur raw-parameter
    layout and fixed owner blocks, and binds all of it to the immutable source
    snapshot.  Passing this gate proves representation and provenance only;
    overlap removal, staleness and optimizer mutation are mapper concerns.
    """

    import numpy as np

    from src.refinement.resplat_unblur_bridge import (
        SUPPORTED_SH_DIMS,
        WORLD_ARTIFACT_SCHEMA,
        covariance_to_unblur_scale_rotation,
    )

    reasons: list[str] = []
    measurements: dict[str, Any] = {}
    if int((result.get("official_resplat") or {}).get("num_refine", -1)) != int(
        expected_refinement_updates
    ):
        reasons.append("world_artifact_refinement_count_mismatch")
    contract = result.get("unblur_world_artifact") or {}
    if contract.get("schema") != WORLD_ARTIFACT_SCHEMA:
        reasons.append("wrong_unblur_world_artifact_schema")
    bindings = {
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "source_pose_revision": int(snapshot.get("pose_revision", -1)),
        "source_pose_hashes": [frame["pose_hash"] for frame in snapshot.get("frames", [])],
        "refinement_state": int(expected_refinement_updates),
    }
    for name, expected in bindings.items():
        if contract.get(name) != expected:
            reasons.append(f"world_artifact_binding_mismatch:{name}")
    frames = snapshot.get("frames") or []
    if len(frames) != OFFICIAL_CONTEXT_KEYFRAMES:
        reasons.append("world_artifact_snapshot_context_count_mismatch")
        return GateDecision(False, tuple(reasons), measurements)
    middle = frames[OFFICIAL_CONTEXT_KEYFRAMES // 2]
    if (
        contract.get("pivot_context_index_zero_based")
        != OFFICIAL_CONTEXT_KEYFRAMES // 2
        or contract.get("pivot_frame_id") != middle.get("frame_id")
        or contract.get("pivot_pose_hash") != middle.get("pose_hash")
    ):
        reasons.append("world_artifact_middle_pivot_binding_mismatch")
    expected_frame_ids = [int(frame["frame_id"]) for frame in frames]
    expected_ordinals = [int(frame["sequence_ordinal"]) for frame in frames]
    if contract.get("owner_frame_ids") != expected_frame_ids:
        reasons.append("world_artifact_owner_frame_ids_mismatch")
    if contract.get("owner_sequence_ordinals") != expected_ordinals:
        reasons.append("world_artifact_owner_ordinals_mismatch")
    if contract.get("coordinate_frame") != "snapshot_unblur_world_opencv":
        reasons.append("wrong_unblur_world_coordinate_frame")
    if contract.get("local_frame") != "middle_context_camera_local_opencv":
        reasons.append("wrong_resplat_local_coordinate_frame")
    if contract.get("covariance_source") != "official_refined_gaussians.covariances":
        reasons.append("world_covariance_not_bound_to_official_covariance")
    if bool(contract.get("native_scale_rotation_copied", True)):
        reasons.append("unsafe_native_scale_rotation_copy_declared")
    if contract.get("representationally_importable_by_unblur") is not True:
        reasons.append("world_artifact_not_declared_importable")
    if contract.get("unconditional_append_safe") is not False:
        reasons.append("world_artifact_unconditional_append_misdeclared_safe")
    if contract.get("optimizer_state_included") is not False:
        reasons.append("world_artifact_optimizer_state_disclosure_mismatch")
    if (
        contract.get("official_no_rotate_sh") is not True
        or int(contract.get("source_harmonic_dimension", -1)) != 16
        or int(contract.get("imported_harmonic_dimension", -1)) != 1
        or int(contract.get("dropped_higher_order_harmonics", -1)) != 15
        or contract.get("harmonic_conversion")
        != "native_dc_copied_exactly;degree_gt_0_dropped"
    ):
        reasons.append("unsafe_or_incomplete_dc_only_sh_contract")
    import_recommendation = contract.get("recommended_active_import") or {}
    if (
        import_recommendation.get("zero_sh_rest") is not True
        or import_recommendation.get("higher_order_sh_imported") is not False
        or import_recommendation.get("max_sh_degree_zero_behavior")
        != "import_dc_only_and_allocate_zero_rest"
    ):
        reasons.append("unsafe_or_missing_active_sh_import_recommendation")

    outputs = result.get("outputs") or {}
    root = Path(result_root).expanduser().resolve()

    def bound_path(name: str, sha_name: str) -> Optional[Path]:
        relative = Path(str(outputs.get(name, "")))
        if not str(relative) or relative.is_absolute() or ".." in relative.parts:
            reasons.append(f"invalid_artifact_path:{name}")
            return None
        path = root / relative
        if not path.is_file():
            reasons.append(f"artifact_missing:{name}")
            return None
        actual = sha256_file(path)
        if actual != str(outputs.get(sha_name, "")):
            reasons.append(f"artifact_sha256_mismatch:{name}")
        return path

    world_path = bound_path(
        "unblur_world_gaussians_npz", "unblur_world_gaussians_npz_sha256"
    )
    native_path = bound_path(
        "native_gaussians_npz", "native_gaussians_npz_sha256"
    )
    if world_path is None or native_path is None:
        return GateDecision(False, tuple(reasons), measurements)

    required = {
        "source_covariances_local",
        "means_world",
        "covariances_world",
        "harmonics_world",
        "opacities",
        "scales_world",
        "rotations_world_wxyz",
        "unblur_features_dc",
        "unblur_features_rest",
        "unblur_log_scales",
        "unblur_logit_opacities",
        "owner_context_slots",
        "owner_frame_ids",
        "owner_sequence_ordinals",
    }
    declared_arrays = outputs.get("unblur_world_npz_arrays") or {}
    if set(declared_arrays) != required:
        reasons.append("unblur_world_array_manifest_incomplete")
    try:
        with np.load(world_path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                reasons.append("unblur_world_npz_arrays_incomplete")
                return GateDecision(False, tuple(reasons), measurements)
            arrays = {name: np.asarray(archive[name]) for name in required}
        with np.load(native_path, allow_pickle=False) as archive:
            native = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        reasons.append(f"unblur_world_npz_unreadable:{type(error).__name__}")
        return GateDecision(False, tuple(reasons), measurements)

    means = arrays["means_world"]
    count = int(means.shape[0]) if means.ndim == 2 else -1
    harmonics = arrays["harmonics_world"]
    sh_dim = int(harmonics.shape[2]) if harmonics.ndim == 3 else -1
    valid_shapes = (
        means.shape == (count, 3)
        and arrays["source_covariances_local"].shape == (count, 3, 3)
        and arrays["covariances_world"].shape == (count, 3, 3)
        and harmonics.shape == (count, 3, sh_dim)
        and sh_dim in SUPPORTED_SH_DIMS
        and arrays["opacities"].shape == (count,)
        and arrays["scales_world"].shape == (count, 3)
        and arrays["rotations_world_wxyz"].shape == (count, 4)
        and arrays["unblur_features_dc"].shape == (count, 1, 3)
        and arrays["unblur_features_rest"].shape == (count, sh_dim - 1, 3)
        and arrays["unblur_log_scales"].shape == (count, 3)
        and arrays["unblur_logit_opacities"].shape == (count, 1)
        and arrays["owner_context_slots"].shape == (count,)
        and arrays["owner_frame_ids"].shape == (count,)
        and arrays["owner_sequence_ordinals"].shape == (count,)
    )
    if not valid_shapes:
        reasons.append("unblur_world_npz_shape_contract_failed")
        return GateDecision(False, tuple(reasons), measurements)
    expected_native_arrays = {
        "means",
        "covariances",
        "scales",
        "rotations",
        "harmonics",
        "opacities",
    }
    if set(native) != expected_native_arrays:
        reasons.append("conversion_native_gaussian_six_array_contract_failed")
    for name, value in arrays.items():
        declaration = declared_arrays.get(name) or {}
        if declaration.get("shape") != list(value.shape) or declaration.get(
            "dtype"
        ) != str(value.dtype):
            reasons.append(f"unblur_world_array_declaration_mismatch:{name}")
    total = sum(int(value.size) for value in arrays.values())
    finite = sum(int(np.isfinite(value).sum()) for value in arrays.values())
    finite_fraction = float(finite / total) if total else 0.0
    if finite_fraction != 1.0:
        reasons.append("unblur_world_non_finite_values")
    if not np.all(arrays["scales_world"] > 0.0):
        reasons.append("unblur_world_nonpositive_scale")
    if not np.all((arrays["opacities"] > 0.0) & (arrays["opacities"] < 1.0)):
        reasons.append("unblur_world_opacity_out_of_open_unit_interval")
    quaternion_error = float(
        np.max(
            np.abs(
                np.linalg.norm(arrays["rotations_world_wxyz"].astype(np.float64), axis=1)
                - 1.0
            )
        )
    )
    if quaternion_error > 1e-4:
        reasons.append("unblur_world_quaternion_norm_failed")

    if count % OFFICIAL_CONTEXT_KEYFRAMES:
        reasons.append("unblur_world_fixed_owner_block_not_divisible")
        per_view = -1
    else:
        per_view = count // OFFICIAL_CONTEXT_KEYFRAMES
        expected_slots = np.repeat(
            np.arange(OFFICIAL_CONTEXT_KEYFRAMES, dtype=np.int16), per_view
        )
        if not np.array_equal(arrays["owner_context_slots"], expected_slots):
            reasons.append("unblur_world_owner_slot_blocks_mismatch")
        if not np.array_equal(
            arrays["owner_frame_ids"], np.repeat(expected_frame_ids, per_view)
        ):
            reasons.append("unblur_world_owner_frame_blocks_mismatch")
        if not np.array_equal(
            arrays["owner_sequence_ordinals"],
            np.repeat(expected_ordinals, per_view),
        ):
            reasons.append("unblur_world_owner_ordinal_blocks_mismatch")

    pivot = np.asarray(middle["c2w_opencv"], dtype=np.float64)
    declared_pivot = np.asarray(contract.get("local_to_world_c2w_opencv"), dtype=np.float64)
    if declared_pivot.shape != (4, 4) or not np.allclose(
        declared_pivot, pivot, rtol=0.0, atol=1e-7
    ):
        reasons.append("unblur_world_pivot_matrix_mismatch")
    else:
        rotation, translation = pivot[:3, :3], pivot[:3, 3]
        native_means = native.get("means")
        native_harmonics = native.get("harmonics")
        native_opacities = native.get("opacities")
        if native_means is None or native_means.shape != (count, 3):
            reasons.append("native_means_missing_for_world_audit")
        else:
            expected_means = native_means.astype(np.float64) @ rotation.T + translation
            mean_error = float(np.max(np.abs(expected_means - means.astype(np.float64))))
            measurements["max_mean_transform_error"] = mean_error
            if mean_error > 2e-5:
                reasons.append("unblur_world_mean_transform_mismatch")
        native_covariances = native.get("covariances")
        source_covariances = arrays["source_covariances_local"]
        if (
            native_covariances is None
            or native_covariances.shape != (count, 3, 3)
        ):
            reasons.append("native_covariances_missing_for_world_audit")
            local_covariance = source_covariances.astype(np.float64)
        else:
            if not np.array_equal(source_covariances, native_covariances):
                reasons.append("world_source_covariances_not_exact_native_copy")
            # Recompute the world state from the official selected-state array,
            # never from the duplicated world-artifact provenance field.
            local_covariance = native_covariances.astype(np.float64)
        raw_expected_covariance = np.einsum(
            "ij,njk,lk->nil", rotation, local_covariance, rotation, optimize=True
        )
        try:
            (
                expected_covariance,
                _expected_scales,
                _expected_rotations,
                expected_factorization,
            ) = covariance_to_unblur_scale_rotation(raw_expected_covariance)
        except ValueError:
            reasons.append("unblur_world_source_covariance_failed_psd_policy")
            expected_covariance = raw_expected_covariance
            expected_factorization = {}
        covariance_error = float(
            np.max(np.abs(expected_covariance - arrays["covariances_world"]))
        )
        measurements["max_covariance_transform_error"] = covariance_error
        if covariance_error > 1e-9:
            reasons.append("unblur_world_covariance_transform_mismatch")
        declared_factorization = contract.get("factorization") or {}
        for name in (
            "clamped_eigenvalue_count",
            "clamped_gaussian_count",
            "significant_negative_gaussian_count",
        ):
            if declared_factorization.get(name) != expected_factorization.get(name):
                reasons.append(f"world_psd_factorization_count_mismatch:{name}")
        for name in (
            "relative_spectral_tolerance",
            "eigenvalue_floor",
            "minimum_raw_eigenvalue",
            "max_psd_correction",
        ):
            try:
                declared_value = float(declared_factorization[name])
                expected_value = float(expected_factorization[name])
            except (KeyError, TypeError, ValueError):
                reasons.append(f"world_psd_factorization_value_missing:{name}")
                continue
            if not math.isclose(
                declared_value, expected_value, rel_tol=1e-8, abs_tol=1e-12
            ):
                reasons.append(f"world_psd_factorization_value_mismatch:{name}")
        if (
            native_harmonics is None
            or native_harmonics.ndim != 3
            or native_harmonics.shape[:2] != (count, 3)
            or native_harmonics.shape[2] != 16
            or harmonics.shape != (count, 3, 1)
        ):
            reasons.append("native_harmonics_missing_for_world_audit")
        else:
            harmonic_error = float(
                np.max(np.abs(native_harmonics[:, :, :1] - harmonics))
            )
            measurements["max_dc_copy_error"] = harmonic_error
            if harmonic_error != 0.0:
                reasons.append("unblur_world_dc_copy_mismatch")
        if native_opacities is None or native_opacities.reshape(-1).shape != (count,):
            reasons.append("native_opacities_missing_for_world_audit")
        elif not np.allclose(
            native_opacities.reshape(-1), arrays["opacities"], rtol=0.0, atol=1e-7
        ):
            reasons.append("unblur_world_opacity_copy_mismatch")

    # Reconstruct covariance from exactly the scale/quaternion convention used
    # by Unblur's build_scaling_rotation (wxyz, column rotation axes).
    try:
        from scipy.spatial.transform import Rotation

        wxyz = arrays["rotations_world_wxyz"].astype(np.float64)
        xyzw = np.concatenate((wxyz[:, 1:], wxyz[:, :1]), axis=1)
        rotation_matrices = Rotation.from_quat(xyzw).as_matrix()
        reconstructed_covariance = np.einsum(
            "nij,nj,nkj->nik",
            rotation_matrices,
            arrays["scales_world"].astype(np.float64) ** 2,
            rotation_matrices,
            optimize=True,
        )
        factor_error = float(
            np.max(
                np.abs(
                    reconstructed_covariance
                    - arrays["covariances_world"].astype(np.float64)
                )
            )
        )
        measurements["max_covariance_factorization_error"] = factor_error
        if factor_error > 3e-5:
            reasons.append("unblur_world_scale_rotation_factorization_mismatch")
    except ValueError:
        reasons.append("unblur_world_quaternion_factorization_failed")

    if not np.allclose(
        np.exp(arrays["unblur_log_scales"].astype(np.float64)),
        arrays["scales_world"].astype(np.float64),
        rtol=2e-5,
        atol=1e-8,
    ):
        reasons.append("unblur_raw_log_scale_mismatch")
    logits = arrays["unblur_logit_opacities"].astype(np.float64)[:, 0]
    activated_opacity = 1.0 / (1.0 + np.exp(-logits))
    if not np.allclose(
        activated_opacity,
        arrays["opacities"].astype(np.float64),
        rtol=2e-5,
        atol=1e-7,
    ):
        reasons.append("unblur_raw_logit_opacity_mismatch")
    reconstructed_harmonics = np.concatenate(
        (
            arrays["unblur_features_dc"],
            arrays["unblur_features_rest"],
        ),
        axis=1,
    ).transpose(0, 2, 1)
    if not np.array_equal(reconstructed_harmonics, harmonics):
        reasons.append("unblur_feature_layout_mismatch")

    measurements.update(
        {
            "gaussian_count": count,
            "gaussians_per_context_view": per_view,
            "sh_dimension": sh_dim,
            "finite_fraction": finite_fraction,
            "max_quaternion_norm_deviation": quaternion_error,
            "world_artifact_sha256": sha256_file(world_path),
        }
    )
    return GateDecision(not reasons, tuple(reasons), measurements)


@dataclass
class _ActiveProcess:
    snapshot_dir: Path
    raw_output_dir: Path
    process: subprocess.Popen
    started_monotonic: float
    stdout_handle: Any
    stderr_handle: Any


class OfficialReSplatSidecarQueue:
    """One-worker nonblocking subprocess queue for immutable native sidecars."""

    def __init__(self, config: SidecarConfig) -> None:
        self.config = config
        self.root = Path(config.output_root).expanduser().resolve()
        self.pending: list[Path] = []
        self.active: Optional[_ActiveProcess] = None
        self._submitted: set[str] = set()
        if config.enabled:
            for name in ("snapshots", "raw", "published", "rejected", "logs"):
                (self.root / name).mkdir(parents=True, exist_ok=True)

    def _event(self, event: str, **values: Any) -> dict[str, Any]:
        return {"schema": QUEUE_EVENT_SCHEMA, "event": event, **values}

    def submit(self, snapshot_dir: Path | str) -> dict[str, Any]:
        if not self.config.enabled:
            return self._event("disabled")
        source = Path(snapshot_dir).expanduser().resolve()
        snapshot = load_snapshot(source)
        snapshot_id = str(snapshot["snapshot_id"])
        if snapshot_id in self._submitted:
            return self._event("duplicate_rejected", snapshot_id=snapshot_id)
        in_flight = len(self.pending) + (1 if self.active is not None else 0)
        if in_flight >= self.config.queue_capacity:
            rejection = self.root / "rejected" / f"{snapshot_id}-queue-full.json"
            _atomic_write_json(
                rejection,
                self._event("queue_full_rejected", snapshot_id=snapshot_id),
            )
            return self._event("queue_full_rejected", snapshot_id=snapshot_id)
        self._submitted.add(snapshot_id)
        self.pending.append(source)
        self._start_next()
        return self._event("submitted", snapshot_id=snapshot_id)

    def _command(self, snapshot_dir: Path, raw_output: Path) -> list[str]:
        return [
            self.config.python_executable,
            self.config.runner_script,
            "--snapshot-dir",
            str(snapshot_dir),
            "--output-dir",
            str(raw_output),
            "--resplat-repo",
            self.config.resplat_repo,
            "--checkpoint",
            self.config.checkpoint,
            "--expected-checkpoint-sha256",
            self.config.expected_checkpoint_sha256,
            "--model-preset",
            self.config.model_preset,
            "--num-refine",
            str(int(self.config.refinement_updates)),
            "--device",
            self.config.process_device,
            "--near",
            str(self.config.near),
            "--far",
            str(self.config.far),
        ]

    def _start_next(self) -> None:
        if self.active is not None or not self.pending:
            return
        snapshot_dir = self.pending.pop(0)
        snapshot = load_snapshot(snapshot_dir, verify_images=False)
        snapshot_id = str(snapshot["snapshot_id"])
        raw_output = self.root / "raw" / snapshot_id
        stdout_path = self.root / "logs" / f"{snapshot_id}.stdout.log"
        stderr_path = self.root / "logs" / f"{snapshot_id}.stderr.log"
        stdout_handle = stdout_path.open("wb")
        stderr_handle = stderr_path.open("wb")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        try:
            process = subprocess.Popen(
                self._command(snapshot_dir, raw_output),
                cwd=str(Path(self.config.runner_script).resolve().parents[1]),
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            raise
        self.active = _ActiveProcess(
            snapshot_dir=snapshot_dir,
            raw_output_dir=raw_output,
            process=process,
            started_monotonic=time.monotonic(),
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )

    def _install_rejection(
        self, active: _ActiveProcess, snapshot: Mapping[str, Any], reasons: Sequence[str]
    ) -> Path:
        snapshot_id = str(snapshot["snapshot_id"])
        source = active.raw_output_dir
        if not source.exists():
            source = Path(
                tempfile.mkdtemp(prefix=f".{snapshot_id}.rejected.", dir=str(self.root / "raw"))
            )
        _atomic_write_json(
            source / "gate_decision.json",
            GateDecision(False, tuple(reasons)).to_dict(),
        )
        destination = self.root / "rejected" / snapshot_id
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"rejection destination exists: {destination}")
        os.rename(source, destination)
        return destination

    def poll(
        self,
        *,
        current_poses: Mapping[FrameId, Sequence[Sequence[float]]],
        current_pose_revision: int,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        active = self.active
        if active is None:
            self._start_next()
            return events
        elapsed = time.monotonic() - active.started_monotonic
        return_code = active.process.poll()
        timed_out = return_code is None and elapsed > self.config.max_runtime_seconds
        if timed_out:
            active.process.terminate()
            try:
                active.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                active.process.kill()
                active.process.wait(timeout=5.0)
            return_code = active.process.returncode
        if return_code is None:
            return events
        active.stdout_handle.close()
        active.stderr_handle.close()
        snapshot = load_snapshot(active.snapshot_dir, verify_images=False)
        snapshot_id = str(snapshot["snapshot_id"])
        try:
            if timed_out:
                destination = self._install_rejection(
                    active, snapshot, ("runtime_gate_exceeded",)
                )
                events.append(
                    self._event("rejected", snapshot_id=snapshot_id, path=str(destination))
                )
            elif return_code != 0:
                destination = self._install_rejection(
                    active, snapshot, (f"runner_exit_code:{return_code}",)
                )
                events.append(
                    self._event("rejected", snapshot_id=snapshot_id, path=str(destination))
                )
            else:
                manifest_path = active.raw_output_dir / "run_manifest.json"
                try:
                    result = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError("runner produced no valid run_manifest.json") from error
                result_gate = evaluate_result(
                    result,
                    snapshot=snapshot,
                    elapsed_seconds=elapsed,
                    config=self.config,
                )
                artifact_gate = verify_result_artifacts(
                    result, active.raw_output_dir
                )
                if bool(result.get("native_to_unblur_conversion_performed", False)):
                    world_artifact_gate = verify_unblur_world_artifact(
                        result,
                        active.raw_output_dir,
                        snapshot,
                        expected_refinement_updates=int(
                            self.config.refinement_updates
                        ),
                    )
                else:
                    world_artifact_gate = GateDecision(True, ())
                stale_gate = evaluate_staleness(
                    snapshot,
                    current_poses=current_poses,
                    current_pose_revision=current_pose_revision,
                    config=self.config,
                )
                reasons = (
                    *result_gate.reasons,
                    *artifact_gate.reasons,
                    *world_artifact_gate.reasons,
                    *stale_gate.reasons,
                )
                accepted = not reasons
                combined = GateDecision(
                    accepted=accepted,
                    reasons=tuple(reasons),
                    measurements={
                        "result": dict(result_gate.measurements),
                        "artifact": dict(artifact_gate.measurements),
                        "world_artifact": dict(
                            world_artifact_gate.measurements
                        ),
                        "staleness": dict(stale_gate.measurements),
                    },
                )
                _atomic_write_json(
                    active.raw_output_dir / "gate_decision.json", combined.to_dict()
                )
                bucket = "published" if accepted else "rejected"
                destination = self.root / bucket / snapshot_id
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(f"sidecar destination exists: {destination}")
                os.rename(active.raw_output_dir, destination)
                events.append(
                    self._event(
                        "published" if accepted else "rejected",
                        snapshot_id=snapshot_id,
                        path=str(destination),
                        reasons=list(reasons),
                    )
                )
        except Exception as error:
            destination = self._install_rejection(
                active, snapshot, (f"publication_contract_error:{type(error).__name__}",)
            )
            events.append(
                self._event(
                    "rejected",
                    snapshot_id=snapshot_id,
                    path=str(destination),
                    reasons=[str(error)],
                )
            )
        self.active = None
        self._start_next()
        return events

    def drain(
        self,
        *,
        current_pose_provider: Callable[
            [], tuple[Mapping[FrameId, Sequence[Sequence[float]]], int]
        ],
    ) -> list[dict[str, Any]]:
        """Wait at most the configured final timeout, then leave work unmerged."""

        events: list[dict[str, Any]] = []
        deadline = time.monotonic() + self.config.final_drain_timeout_seconds
        while (self.active is not None or self.pending) and time.monotonic() <= deadline:
            poses, revision = current_pose_provider()
            events.extend(
                self.poll(current_poses=poses, current_pose_revision=revision)
            )
            if self.active is not None:
                time.sleep(0.05)
        if self.active is not None or self.pending:
            abandoned = list(self.pending)
            self.pending.clear()
            for snapshot_dir in abandoned:
                snapshot = load_snapshot(snapshot_dir, verify_images=False)
                snapshot_id = str(snapshot["snapshot_id"])
                destination = (
                    self.root / "rejected" / f"{snapshot_id}-drain-timeout.json"
                )
                _atomic_write_json(
                    destination,
                    self._event(
                        "not_launched_before_drain_timeout",
                        snapshot_id=snapshot_id,
                    ),
                )
                events.append(
                    self._event(
                        "rejected",
                        snapshot_id=snapshot_id,
                        path=str(destination),
                        reasons=["not_launched_before_drain_timeout"],
                    )
                )
            if self.active is not None:
                self.active.process.terminate()
                try:
                    self.active.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.active.process.kill()
                    self.active.process.wait(timeout=5.0)
                poses, revision = current_pose_provider()
                events.extend(
                    self.poll(current_poses=poses, current_pose_revision=revision)
                )
            events.append(
                self._event(
                    "drain_timeout_all_unfinished_sidecars_rejected",
                    pending_rejected_without_launch=len(abandoned),
                )
            )
        return events


__all__ = [
    "GATE_SCHEMA",
    "OFFICIAL_CONTEXT_KEYFRAMES",
    "MAX_OFFICIAL_REFINEMENT_UPDATES",
    "MIN_OFFICIAL_REFINEMENT_UPDATES",
    "OFFICIAL_PRESET",
    "OFFICIAL_REFINEMENT_UPDATES",
    "OfficialReSplatSidecarQueue",
    "RESULT_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "GateDecision",
    "SidecarConfig",
    "SidecarFrameInput",
    "UnsupportedActiveMapMerge",
    "active_map_merge_assessment",
    "evaluate_result",
    "evaluate_staleness",
    "load_snapshot",
    "materialize_closed_submap_snapshot",
    "pose_hash",
    "reject_active_map_merge",
    "sha256_file",
    "verify_result_artifacts",
    "verify_unblur_world_artifact",
]
