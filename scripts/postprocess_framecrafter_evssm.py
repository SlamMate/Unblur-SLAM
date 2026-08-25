#!/usr/bin/env python3
"""Fail-closed FrameCrafter -> EVSSM post-processing.

This command consumes one *completed* production FrameCrafter manifest/report,
runs the repository EVSSM checkpoint on every accepted synthetic RGB, and
emits a new immutable snapshot.  It never edits the upstream snapshot.

An EVSSM result replaces its FrameCrafter input only when all of the following
hold:

* exact-PNG global Laplacian sharpness improves;
* the shared tile audit detects no local brightness/edge/Laplacian/MAE/dark
  expansion regression; and
* the original ``evaluate_candidate`` RGB-D/photometric/reprojection gates
  still pass (sharpness is classified separately).

The default failure policy retains the pre-EVSSM FrameCrafter RGB.  ``reject``
instead removes a failed synthetic observation from the augmented stream.  In
both cases the EVSSM candidate is retained as a content-addressed audit asset.

Production inference is CUDA-only and strict-loads the supplied checkpoint.
Tests may inject a fake callable only with ``test_only=True``; such manifests
are marked ``backend_test_only`` and are rejected by the normal SLAM validator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.precompute_framecrafter_evssm import (  # noqa: E402
    build_evssm_inference,
)
from src.framecrafter_context import (  # noqa: E402
    ContextSelectionConfig,
    audit_evssm_local_degradation,
)
from src.framecrafter_pipeline import (  # noqa: E402
    FrameRecord,
    GateConfig,
    TargetView,
    evaluate_candidate,
    laplacian_sharpness,
    read_depth,
    read_rgb,
    source_input_digest,
    synthetic_output_digest,
    validate_manifest_payload,
    validate_pose_source,
)


POSTPROCESS_SCHEMA = "unblur_slam.framecrafter_evssm_postprocess.v1"
REPORT_SCHEMA = "unblur_slam.framecrafter_evssm_postprocess_report.v1"
OFFICIAL_UNBLUR_SLAM_EVSSM_SHA256 = (
    "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EPS = 1.0e-12


Inference = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class PostprocessConfig:
    failure_policy: str = "fallback"
    min_post_vs_pre_sharpness_gain: float = 1.0
    min_reference_sharpness_gain: float = 1.05
    depth_scale: float = 5000.0
    min_depth_coverage: float = 0.05
    min_depth_consistency: float = 0.50
    max_photometric_error: float = 0.20
    max_reprojection_error_px: float = 2.0
    min_reprojection_valid_ratio: float = 0.05
    depth_abs_tolerance: float = 0.03
    depth_rel_tolerance: float = 0.03
    require_depth: bool = True
    local_tile_size: int = 32
    local_tile_stride: int = 16
    local_max_brightness_drop: float = 0.30
    local_min_edge_retention: float = 0.50
    local_min_laplacian_retention: float = 0.50
    local_max_tile_mae: float = 0.20
    local_max_dark_expansion: float = 0.30
    local_dark_luma_threshold: float = 96.0 / 255.0
    local_min_pre_luma: float = 0.10
    local_min_pre_edge: float = 0.01
    local_min_pre_laplacian: float = 0.01

    def __post_init__(self) -> None:
        if self.failure_policy not in {"fallback", "reject"}:
            raise ValueError("failure_policy must be fallback or reject")
        for name, value in (
            (
                "min_post_vs_pre_sharpness_gain",
                self.min_post_vs_pre_sharpness_gain,
            ),
            ("min_reference_sharpness_gain", self.min_reference_sharpness_gain),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(float(self.depth_scale)) or self.depth_scale <= 0.0:
            raise ValueError("depth_scale must be finite and positive")
        for name, value in (
            ("min_depth_coverage", self.min_depth_coverage),
            ("min_depth_consistency", self.min_depth_consistency),
            ("min_reprojection_valid_ratio", self.min_reprojection_valid_ratio),
        ):
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
        for name, value in (
            ("max_photometric_error", self.max_photometric_error),
            ("max_reprojection_error_px", self.max_reprojection_error_px),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("depth_abs_tolerance", self.depth_abs_tolerance),
            ("depth_rel_tolerance", self.depth_rel_tolerance),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        # Reuse the authoritative validators for both gate families.
        self.gate_config()
        self.local_gate_config()

    def gate_config(self) -> GateConfig:
        return GateConfig(
            min_sharpness_gain=float(self.min_reference_sharpness_gain),
            min_depth_coverage=float(self.min_depth_coverage),
            min_depth_consistency=float(self.min_depth_consistency),
            max_photometric_error=float(self.max_photometric_error),
            max_reprojection_error_px=float(self.max_reprojection_error_px),
            min_reprojection_valid_ratio=float(self.min_reprojection_valid_ratio),
            depth_abs_tolerance=float(self.depth_abs_tolerance),
            depth_rel_tolerance=float(self.depth_rel_tolerance),
            require_depth=bool(self.require_depth),
        )

    def local_gate_config(self) -> ContextSelectionConfig:
        return ContextSelectionConfig(
            evssm_local_gate_enabled=True,
            evssm_local_tile_size=int(self.local_tile_size),
            evssm_local_tile_stride=int(self.local_tile_stride),
            evssm_local_max_brightness_drop=float(
                self.local_max_brightness_drop
            ),
            evssm_local_min_edge_retention=float(self.local_min_edge_retention),
            evssm_local_min_laplacian_retention=float(
                self.local_min_laplacian_retention
            ),
            evssm_local_max_tile_mae=float(self.local_max_tile_mae),
            evssm_local_max_dark_expansion=float(self.local_max_dark_expansion),
            evssm_local_dark_luma_threshold=float(
                self.local_dark_luma_threshold
            ),
            evssm_local_min_raw_luma=float(self.local_min_pre_luma),
            evssm_local_min_raw_edge=float(self.local_min_pre_edge),
            evssm_local_min_raw_laplacian=float(
                self.local_min_pre_laplacian
            ),
        )


@dataclass(frozen=True)
class _SyntheticJob:
    """All CPU-validated inputs needed for one EVSSM inference."""

    accepted_record: dict[str, Any]
    left: FrameRecord
    right: FrameRecord
    target: TargetView
    pre_path: Path
    pre_sha256: str
    pre_rgb: np.ndarray
    intrinsics_provenance: dict[str, Any]


def sha256_file(path: Path | str) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable JSON: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # The UUID filename is unique, and hard-linking gives no-replace
        # publication semantics even on filesystems where rename would replace.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _encode_rgb_png(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"EVSSM output must be HWC RGB, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("EVSSM output contains non-finite values")
    encoded = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(path, format="PNG")


def _store_content_addressed_rgb(
    root: Path, image: np.ndarray
) -> tuple[Path, str]:
    staging = root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"{uuid.uuid4().hex}.png"
    try:
        _encode_rgb_png(temporary, image)
        digest = sha256_file(temporary)
        destination = root / "artifacts" / "evssm_rgb" / digest[:2] / f"{digest}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != digest:
                raise RuntimeError(
                    f"content-address collision or corrupt artifact: {destination}"
                )
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if sha256_file(destination) != digest:
                    raise RuntimeError(
                        f"concurrent content-address collision: {destination}"
                    )
        return destination.resolve(), digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite_metrics(metrics: Mapping[str, Optional[float]]) -> dict[str, Optional[float]]:
    cleaned: dict[str, Optional[float]] = {}
    for name, value in metrics.items():
        if value is None:
            cleaned[str(name)] = None
        else:
            number = float(value)
            cleaned[str(name)] = number if math.isfinite(number) else None
    return cleaned


def _intrinsics_from_entry(entry: Mapping[str, Any]) -> Optional[np.ndarray]:
    value = entry.get("intrinsics")
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("manifest intrinsics must be a finite 3x3 matrix")
    return matrix


class _BatchGeometry:
    def __init__(self, report: Mapping[str, Any], report_dir: Path):
        self.report = report
        self.report_dir = report_dir
        self.batches = {
            str(item["batch_id"]): item
            for item in report.get("generation_batches", [])
            if isinstance(item, Mapping)
        }
        self.planned = {
            str(item["target_id"]): item
            for item in report.get("planned", [])
            if isinstance(item, Mapping)
        }
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _load(self, batch_id: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._arrays.get(batch_id)
        if cached is not None:
            return cached
        batch = self.batches.get(batch_id)
        if batch is None:
            raise ValueError(f"target refers to unknown batch {batch_id!r}")
        path = Path(str(batch["poses_npz"])).expanduser()
        if not path.is_absolute():
            path = self.report_dir / path
        path = path.resolve()
        with np.load(path, allow_pickle=False) as payload:
            poses = np.asarray(payload["w2c_poses"], dtype=np.float64).copy()
            intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64).copy()
        self._arrays[batch_id] = (poses, intrinsics)
        return poses, intrinsics

    def target_and_support_intrinsics(
        self,
        target_entry: Mapping[str, Any],
        support_indices: tuple[int, int],
        originals: Mapping[int, Mapping[str, Any]],
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], dict[str, Any]]:
        direct_target = _intrinsics_from_entry(target_entry)
        batch_id = str(target_entry.get("batch_id", "")).strip()
        sources: dict[str, Any] = {}
        batch: Optional[Mapping[str, Any]] = None
        context_indices: list[int] = []
        batch_intrinsics: Optional[np.ndarray] = None
        if batch_id:
            batch = self.batches.get(batch_id)
            if batch is None:
                raise ValueError(f"unknown generation batch {batch_id!r}")
            _, batch_intrinsics = self._load(batch_id)
            context_indices = [int(value) for value in batch["context_source_indices"]]
            target_position = int(target_entry["batch_target_position"])
            target_k = batch_intrinsics[len(context_indices) + target_position]
            sources["target"] = "generation_batch_npz"
            if direct_target is not None and not np.allclose(
                direct_target, target_k, rtol=0.0, atol=1.0e-6
            ):
                raise ValueError("manifest and batch target intrinsics disagree")
        elif direct_target is not None:
            target_k = direct_target
            sources["target"] = "manifest"
        else:
            raise ValueError(
                "cannot reconstruct target intrinsics: missing batch NPZ and "
                "manifest intrinsics"
            )

        support_k: list[np.ndarray] = []
        support_sources: list[str] = []
        for source_index in support_indices:
            direct = _intrinsics_from_entry(originals[source_index])
            if direct is not None:
                support_k.append(direct)
                support_sources.append("manifest")
            elif batch_intrinsics is not None and source_index in context_indices:
                support_k.append(batch_intrinsics[context_indices.index(source_index)])
                support_sources.append("generation_batch_npz_context")
            else:
                # Unblur-SLAM/TUM streams use one calibrated K for the full
                # sequence.  A local gate support need not itself be a model
                # conditioning frame, so the target K is the only K stored by
                # the upstream production snapshot in that case.
                support_k.append(target_k)
                support_sources.append("generation_batch_target_shared_calibration")
        sources["supports"] = support_sources
        return target_k, (support_k[0], support_k[1]), sources

    def target_positions(self, target_entry: Mapping[str, Any]) -> tuple[int, int]:
        planned = self.planned.get(str(target_entry["target_id"]))
        if planned is None:
            return int(target_entry["left_index"]), int(target_entry["right_index"])
        return int(planned["left_position"]), int(planned["right_position"])


def _frame_record(
    entry: Mapping[str, Any], source_index: int, intrinsics: np.ndarray
) -> FrameRecord:
    return FrameRecord(
        source_index=source_index,
        frame_id=f"original_{source_index:06d}",
        timestamp=float(entry["timestamp"]),
        rgb_path=Path(str(entry["rgb_path"])),
        depth_path=(
            None
            if entry.get("depth_path") in (None, "")
            else Path(str(entry["depth_path"]))
        ),
        c2w=np.asarray(entry["c2w"], dtype=np.float64),
        intrinsics=intrinsics,
        eval=True,
    )


def _target_view(
    entry: Mapping[str, Any], intrinsics: np.ndarray, positions: tuple[int, int]
) -> TargetView:
    return TargetView(
        target_id=str(entry["target_id"]),
        left_index=int(entry["left_index"]),
        right_index=int(entry["right_index"]),
        left_position=positions[0],
        right_position=positions[1],
        timestamp=float(entry["timestamp"]),
        alpha=float(entry["alpha"]),
        c2w=np.asarray(entry["c2w"], dtype=np.float64),
        intrinsics=intrinsics,
        reasons=tuple(str(value) for value in entry.get("reasons", [])),
    )


def _evaluate_one(
    rgb: np.ndarray,
    left: FrameRecord,
    right: FrameRecord,
    target: TargetView,
    config: PostprocessConfig,
) -> dict[str, Any]:
    try:
        left_depth = (
            None
            if left.depth_path is None
            else read_depth(left.depth_path, config.depth_scale)
        )
        right_depth = (
            None
            if right.depth_path is None
            else read_depth(right.depth_path, config.depth_scale)
        )
        gate = evaluate_candidate(
            rgb,
            left,
            right,
            target,
            left_depth=left_depth,
            right_depth=right_depth,
            config=config.gate_config(),
        )
        failures = list(gate.failures)
        geometry_failures = [value for value in failures if value != "sharpness_gain"]
        sharp_failures = [value for value in failures if value == "sharpness_gain"]
        return {
            "metrics": _finite_metrics(gate.metrics),
            "confidence": float(gate.confidence),
            "failures": failures,
            "geometry_failures": geometry_failures,
            "sharp_failures": sharp_failures,
            "geometry_passed": not geometry_failures,
            "sharpness_passed": not sharp_failures,
            "error": None,
        }
    except Exception as error:
        return {
            "metrics": {},
            "confidence": 0.0,
            "failures": ["evaluation_error"],
            "geometry_failures": ["evaluation_error"],
            "sharp_failures": [],
            "geometry_passed": False,
            "sharpness_passed": False,
            "error": f"{type(error).__name__}: {error}",
        }


def evaluate_evssm_candidate(
    *,
    pre_path: Path,
    post_path: Path,
    left: FrameRecord,
    right: FrameRecord,
    target: TargetView,
    config: PostprocessConfig,
) -> dict[str, Any]:
    """Return the complete double-gate decision for one stored candidate."""

    pre = read_rgb(pre_path)
    post = read_rgb(post_path)
    if post.shape != pre.shape:
        raise ValueError(f"EVSSM changed image shape: {post.shape} != {pre.shape}")
    pre_sharpness = laplacian_sharpness(pre)
    post_sharpness = laplacian_sharpness(post)
    post_vs_pre = post_sharpness / max(pre_sharpness, _EPS)
    improved = bool(
        post_sharpness > pre_sharpness + _EPS
        and post_vs_pre >= float(config.min_post_vs_pre_sharpness_gain)
    )
    local = audit_evssm_local_degradation(
        pre_path, post_path, config.local_gate_config()
    )
    pre_gate = _evaluate_one(pre, left, right, target, config)
    post_gate = _evaluate_one(post, left, right, target, config)
    failures: list[str] = []
    if not improved:
        failures.append("post_sharpness_not_improved")
    if not local.passed:
        failures.extend(f"local_{value}" for value in local.failure_reasons)
    if not post_gate["geometry_passed"]:
        failures.extend(
            f"geometry_{value}" for value in post_gate["geometry_failures"]
        )
    return {
        "pre_laplacian_sharpness": float(pre_sharpness),
        "post_laplacian_sharpness": float(post_sharpness),
        "post_vs_pre_sharpness_gain": float(post_vs_pre),
        "global_sharpness_improved": improved,
        "local_gate": local.as_dict(),
        "pre_evaluate_candidate": pre_gate,
        "post_evaluate_candidate": post_gate,
        "replacement_passed": not failures,
        "replacement_failures": list(dict.fromkeys(failures)),
    }


def _accepted_record_by_id(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in report.get("accepted", []):
        if not isinstance(record, Mapping):
            raise ValueError("upstream accepted records must be objects")
        target_id = str(record.get("target_id", ""))
        if not target_id or target_id in result:
            raise ValueError("upstream accepted target IDs must be unique/non-empty")
        result[target_id] = copy.deepcopy(dict(record))
    return result


def _preflight_synthetic_jobs(
    upstream_manifest: Mapping[str, Any],
    upstream_report: Mapping[str, Any],
    *,
    report_file: Path,
) -> tuple[dict[int, Mapping[str, Any]], dict[str, _SyntheticJob]]:
    """Resolve every synthetic dependency before an inference model is loaded.

    The upstream manifest validator binds the batch files and RGB-D artifacts,
    but accepted-record fields such as ``gate_support_source_indices`` are
    consumed only by this post-processing stage.  Resolve the complete job set
    eagerly so a malformed final target cannot waste all preceding GPU calls.
    """

    original_entries: dict[int, Mapping[str, Any]] = {
        int(entry["source_index"]): entry
        for entry in upstream_manifest["frames"]
        if entry.get("kind") == "original"
    }
    accepted_records = _accepted_record_by_id(upstream_report)
    geometry = _BatchGeometry(upstream_report, report_file.parent)
    jobs: dict[str, _SyntheticJob] = {}

    for upstream_entry in upstream_manifest["frames"]:
        kind = upstream_entry.get("kind")
        if kind == "original":
            continue
        if kind != "synthetic":
            raise ValueError(f"unsupported frame kind {kind!r}")

        target_id = str(upstream_entry["target_id"])
        accepted_record = accepted_records.get(target_id)
        if accepted_record is None:
            raise ValueError(
                f"synthetic target {target_id!r} has no upstream accepted record"
            )
        support_values = accepted_record.get("gate_support_source_indices") or [
            upstream_entry["left_index"],
            upstream_entry["right_index"],
        ]
        if not isinstance(support_values, list) or len(support_values) != 2:
            raise ValueError(f"invalid gate supports for {target_id}")
        support_indices = (int(support_values[0]), int(support_values[1]))
        if any(index not in original_entries for index in support_indices):
            raise ValueError(f"gate supports for {target_id} are not original frames")

        target_k, support_k, intrinsics_provenance = (
            geometry.target_and_support_intrinsics(
                upstream_entry, support_indices, original_entries
            )
        )
        left = _frame_record(
            original_entries[support_indices[0]], support_indices[0], support_k[0]
        )
        right = _frame_record(
            original_entries[support_indices[1]], support_indices[1], support_k[1]
        )
        target = _target_view(
            upstream_entry,
            target_k,
            geometry.target_positions(upstream_entry),
        )

        pre_path = Path(str(upstream_entry["rgb_path"])).expanduser().resolve()
        pre_sha256 = sha256_file(pre_path)
        if pre_sha256 != upstream_entry.get("rgb_sha256"):
            raise ValueError(f"upstream synthetic RGB hash changed: {pre_path}")
        pre_rgb = read_rgb(pre_path)
        jobs[target_id] = _SyntheticJob(
            accepted_record=accepted_record,
            left=left,
            right=right,
            target=target,
            pre_path=pre_path,
            pre_sha256=pre_sha256,
            pre_rgb=pre_rgb,
            intrinsics_provenance=intrinsics_provenance,
        )

    if set(jobs) != set(accepted_records):
        missing = sorted(set(accepted_records) - set(jobs))
        raise ValueError(
            "upstream accepted records disagree with synthetic manifest targets: "
            f"extra accepted targets={missing}"
        )
    return original_entries, jobs


def _build_signature(
    *,
    manifest_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    implementation_sha256: str,
    device: str,
    config: PostprocessConfig,
    test_only: bool,
) -> str:
    return _sha256_json(
        {
            "schema": POSTPROCESS_SCHEMA,
            "upstream_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "upstream_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            },
            "implementation_sha256": implementation_sha256,
            "device": "test_injected" if test_only else str(device),
            "config": asdict(config),
            "output_encoding": "rgb_uint8_png_content_addressed_sha256",
        }
    )


def _decision_summary(
    decision: Mapping[str, Any], action: str, selected_sha256: Optional[str]
) -> dict[str, Any]:
    return {
        "target_id": decision["target_id"],
        "action": action,
        "replacement_passed": decision["quality"]["replacement_passed"],
        "replacement_failures": decision["quality"]["replacement_failures"],
        "pre_laplacian_sharpness": decision["quality"][
            "pre_laplacian_sharpness"
        ],
        "post_laplacian_sharpness": decision["quality"][
            "post_laplacian_sharpness"
        ],
        "post_vs_pre_sharpness_gain": decision["quality"][
            "post_vs_pre_sharpness_gain"
        ],
        "selected_rgb_sha256": selected_sha256,
    }


def postprocess(
    *,
    manifest_path: Path | str,
    report_path: Path | str,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    device: str = "cuda:0",
    expected_checkpoint_sha256: Optional[str] = None,
    config: PostprocessConfig = PostprocessConfig(),
    infer: Optional[Inference] = None,
    test_only: bool = False,
) -> dict[str, Any]:
    """Create one new immutable postprocessed snapshot and return its summary."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"EVSSM checkpoint does not exist: {checkpoint_file}")
    if infer is not None and not test_only:
        raise ValueError("injected inference is test-only; pass test_only=True")
    if infer is None and test_only:
        raise ValueError("test_only=True requires an injected inference callable")

    upstream_manifest = _load_json(manifest_file, "upstream manifest")
    upstream_report = _load_json(report_file, "upstream report")
    validate_manifest_payload(
        upstream_manifest, manifest_path=manifest_file, require_provenance=True
    )
    declared_report = Path(
        str(upstream_manifest.get("preprocess_report_path", ""))
    ).expanduser().resolve()
    if declared_report != report_file:
        raise ValueError(
            "explicit --report is not the report cryptographically bound by --manifest"
        )
    if upstream_manifest.get("preprocess_report_sha256") != sha256_file(report_file):
        raise ValueError("upstream manifest/report SHA-256 mismatch")
    if upstream_manifest.get("uses_ground_truth_pose") is not False or upstream_report.get(
        "uses_ground_truth_pose"
    ) is not False:
        raise ValueError("ground-truth-derived preprocessing is forbidden")
    validate_pose_source(str(upstream_manifest.get("pose_source", "")))

    checkpoint_sha256 = sha256_file(checkpoint_file)
    if not _SHA256_RE.fullmatch(checkpoint_sha256):
        raise RuntimeError("internal checkpoint SHA-256 failure")
    expected_contract_hash = expected_checkpoint_sha256
    if expected_contract_hash is None and not test_only:
        expected_contract_hash = OFFICIAL_UNBLUR_SLAM_EVSSM_SHA256
    if expected_contract_hash is not None:
        expected = str(expected_contract_hash).strip().lower()
        if not _SHA256_RE.fullmatch(expected):
            raise ValueError("expected checkpoint SHA-256 must be 64 lowercase hex")
        if checkpoint_sha256 != expected:
            raise ValueError(
                "EVSSM checkpoint SHA-256 mismatch: "
                f"{checkpoint_sha256} != {expected}"
            )

    implementation_path = Path(__file__).resolve()
    implementation_sha256 = sha256_file(implementation_path)
    signature = _build_signature(
        manifest_path=manifest_file,
        report_path=report_file,
        checkpoint_path=checkpoint_file,
        checkpoint_sha256=checkpoint_sha256,
        implementation_sha256=implementation_sha256,
        device=device,
        config=config,
        test_only=test_only,
    )
    generation_id = uuid.uuid4().hex
    snapshot_dir = destination / "snapshots" / signature / generation_id
    new_manifest_path = snapshot_dir / f"manifest_{signature}_{generation_id}.json"
    new_report_path = snapshot_dir / (
        f"preprocess_report_{signature}_{generation_id}.json"
    )
    if new_manifest_path.exists() or new_report_path.exists():
        raise FileExistsError("postprocess snapshot identity already exists")

    # Resolve every target, support, batch-intrinsics source, and input image
    # before loading the CUDA model.  The inference loop below must not expose
    # a structural/provenance failure only after earlier GPU calls have run.
    _, synthetic_jobs = _preflight_synthetic_jobs(
        upstream_manifest,
        upstream_report,
        report_file=report_file,
    )
    inference = (
        infer
        if infer is not None
        else build_evssm_inference(checkpoint_file, device)
    )
    new_entries: list[dict[str, Any]] = []
    new_accepted: list[dict[str, Any]] = []
    newly_rejected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    checkpoint_provenance = {
        "path": str(checkpoint_file),
        "sha256": checkpoint_sha256,
        "expected_sha256": (
            None
            if expected_contract_hash is None
            else str(expected_contract_hash).strip().lower()
        ),
        "official_unblur_slam_motion_checkpoint": bool(
            checkpoint_sha256 == OFFICIAL_UNBLUR_SLAM_EVSSM_SHA256
        ),
        "strict_state_dict_load": not test_only,
        "backend": "test_injected" if test_only else "evssm",
        "device": "test_injected" if test_only else str(device),
    }

    for upstream_entry in upstream_manifest["frames"]:
        entry = copy.deepcopy(dict(upstream_entry))
        if entry.get("kind") == "original":
            # Originals are deliberately byte/path/pose/eval invariant.
            new_entries.append(entry)
            continue
        if entry.get("kind") != "synthetic":
            raise ValueError(f"unsupported frame kind {entry.get('kind')!r}")
        target_id = str(entry["target_id"])
        job = synthetic_jobs[target_id]
        accepted_record = job.accepted_record
        left = job.left
        right = job.right
        target = job.target
        pre_path = job.pre_path
        pre_sha256 = job.pre_sha256
        pre = job.pre_rgb
        post_float = np.asarray(
            inference(pre, float(entry["timestamp"])), dtype=np.float32
        )
        if post_float.shape != pre.shape:
            raise ValueError(
                f"EVSSM changed {target_id} shape: {post_float.shape} != {pre.shape}"
            )
        candidate_path, candidate_sha256 = _store_content_addressed_rgb(
            destination, post_float
        )
        quality = evaluate_evssm_candidate(
            pre_path=pre_path,
            post_path=candidate_path,
            left=left,
            right=right,
            target=target,
            config=config,
        )
        passed = bool(quality["replacement_passed"])
        action = "replace_with_evssm" if passed else (
            "fallback_to_framecrafter"
            if config.failure_policy == "fallback"
            else "reject_synthetic"
        )
        decision = {
            "schema": POSTPROCESS_SCHEMA,
            "target_id": target_id,
            "action": action,
            "upstream": {
                "manifest_path": str(manifest_file),
                "manifest_sha256": sha256_file(manifest_file),
                "report_path": str(report_file),
                "report_sha256": sha256_file(report_file),
                "rgb_path": str(pre_path),
                "rgb_sha256": pre_sha256,
                "depth_path": entry.get("depth_path"),
                "depth_sha256": entry.get("depth_sha256"),
            },
            "candidate": {
                "rgb_path": str(candidate_path),
                "rgb_sha256": candidate_sha256,
                "encoding": "rgb_uint8_png",
                "content_addressed": candidate_path.stem == candidate_sha256,
            },
            "evssm_checkpoint": dict(checkpoint_provenance),
            "implementation": {
                "path": str(implementation_path),
                "sha256": implementation_sha256,
            },
            "gate_support_source_indices": [left.source_index, right.source_index],
            "intrinsics_provenance": job.intrinsics_provenance,
            "thresholds": asdict(config),
            "quality": quality,
            "uses_ground_truth_pose": False,
            "eval": False,
            "fixed_pose": True,
        }
        decisions.append(decision)

        if passed:
            post_gate = quality["post_evaluate_candidate"]
            acceptance_class = (
                "sharp_accepted"
                if post_gate["sharpness_passed"]
                else "geometry_only"
            )
            entry["rgb_path"] = str(candidate_path)
            entry["rgb_sha256"] = candidate_sha256
            entry["gate_metrics"] = copy.deepcopy(post_gate["metrics"])
            entry["confidence"] = float(post_gate["confidence"])
            entry["acceptance_class"] = acceptance_class
        elif config.failure_policy == "reject":
            rejected_record = copy.deepcopy(accepted_record)
            rejected_record.pop("rgb_path", None)
            rejected_record.pop("depth_path", None)
            rejected_record.update(
                acceptance_class="rejected",
                failures=list(quality["replacement_failures"]),
                geometry_failures=list(
                    quality["post_evaluate_candidate"]["geometry_failures"]
                ),
                sharp_failures=list(
                    quality["post_evaluate_candidate"]["sharp_failures"]
                ),
                metrics=copy.deepcopy(
                    quality["post_evaluate_candidate"]["metrics"]
                ),
                candidate_rgb_path=str(candidate_path),
                candidate_rgb_sha256=candidate_sha256,
                evssm_postprocess=decision,
            )
            newly_rejected.append(rejected_record)
            continue

        # A fallback keeps all upstream selected fields exactly.  Both paths
        # record the complete decision as an additive, report-bound provenance.
        entry["evssm_postprocess"] = decision
        new_entries.append(entry)
        selected_record = copy.deepcopy(accepted_record)
        selected_record.update(
            acceptance_class=entry.get("acceptance_class", "sharp_accepted"),
            confidence=entry["confidence"],
            raw_gate_confidence=(
                quality["post_evaluate_candidate"]["confidence"]
                if passed
                else accepted_record.get("raw_gate_confidence", entry["confidence"])
            ),
            geometry_failures=(
                quality["post_evaluate_candidate"]["geometry_failures"]
                if passed
                else accepted_record.get("geometry_failures", [])
            ),
            sharp_failures=(
                quality["post_evaluate_candidate"]["sharp_failures"]
                if passed
                else accepted_record.get("sharp_failures", [])
            ),
            metrics=copy.deepcopy(entry["gate_metrics"]),
            rgb_path=entry["rgb_path"],
            depth_path=entry.get("depth_path"),
            evssm_postprocess=decision,
        )
        new_accepted.append(selected_record)

    # Preserve the upstream temporal order.  Reject policy simply removes the
    # failed synthetic entry; it can never remove or reorder an original.
    generated_count = sum(entry.get("kind") == "synthetic" for entry in new_entries)
    if [
        int(entry["source_index"])
        for entry in new_entries
        if entry.get("kind") == "original"
    ] != list(range(int(upstream_manifest["source_frame_count"]))):
        raise RuntimeError("postprocess changed original-frame order")
    if any(
        entry.get("eval") is not True or bool(entry.get("fixed_pose", False))
        for entry in new_entries
        if entry.get("kind") == "original"
    ):
        raise RuntimeError("postprocess changed original eval/fixed_pose semantics")
    if any(
        entry.get("eval") is not False or entry.get("fixed_pose") is not True
        for entry in new_entries
        if entry.get("kind") == "synthetic"
    ):
        raise RuntimeError("postprocess changed synthetic eval/fixed_pose semantics")

    accepted_digest = synthetic_output_digest(new_entries)
    source_digest = source_input_digest(new_entries)
    existing_rejected = copy.deepcopy(list(upstream_report.get("rejected", [])))
    all_rejected = [*existing_rejected, *newly_rejected]
    sharp_records = [
        record
        for record in new_accepted
        if record.get("acceptance_class", "sharp_accepted") == "sharp_accepted"
    ]
    geometry_records = [
        record
        for record in [*new_accepted, *all_rejected]
        if record.get("acceptance_class") == "geometry_only"
    ]
    geometry_rejected = [
        record
        for record in all_rejected
        if record.get("acceptance_class") == "rejected"
    ]

    report = copy.deepcopy(upstream_report)
    report.update(
        preprocess_signature=signature,
        generation_id=generation_id,
        backend_test_only=bool(test_only),
        accepted_target_count=len(new_accepted),
        rejected_target_count=len(all_rejected),
        sharp_accepted_target_count=len(sharp_records),
        geometry_only_target_count=len(geometry_records),
        geometry_rejected_target_count=len(geometry_rejected),
        accepted_output_sha256=accepted_digest,
        source_input_sha256=source_digest,
        manifest=str(new_manifest_path.resolve()),
        accepted=new_accepted,
        rejected=all_rejected,
        quality_partition={
            "sharp_accepted": sharp_records,
            "geometry_only": geometry_records,
            "rejected": geometry_rejected,
        },
        postprocess_schema=REPORT_SCHEMA,
        postprocess={
            "schema": REPORT_SCHEMA,
            "signature": signature,
            "generation_id": generation_id,
            "upstream_manifest": {
                "path": str(manifest_file),
                "sha256": sha256_file(manifest_file),
            },
            "upstream_report": {
                "path": str(report_file),
                "sha256": sha256_file(report_file),
            },
            "checkpoint": dict(checkpoint_provenance),
            "implementation": {
                "path": str(implementation_path),
                "sha256": implementation_sha256,
            },
            "config": asdict(config),
            "test_only": bool(test_only),
            "production_eligible": not test_only,
            "input_synthetic_count": len(decisions),
            "replace_count": sum(
                item["action"] == "replace_with_evssm" for item in decisions
            ),
            "fallback_count": sum(
                item["action"] == "fallback_to_framecrafter" for item in decisions
            ),
            "postprocess_reject_count": len(newly_rejected),
            "decisions": decisions,
        },
    )
    # Keep the legacy report schema so the existing dataset validator can
    # validate the complete upstream batching contract plus this new stage.
    report["schema"] = "unblur_slam.framecrafter_preprocess_report.v1"
    _atomic_write_json(new_report_path, report)

    manifest = copy.deepcopy(upstream_manifest)
    manifest.update(
        source_frame_count=int(upstream_manifest["source_frame_count"]),
        generated_frame_count=generated_count,
        frames=new_entries,
        preprocess_signature=signature,
        generation_id=generation_id,
        backend_test_only=bool(test_only),
        accepted_output_sha256=accepted_digest,
        source_input_sha256=source_digest,
        preprocess_report_path=str(new_report_path.resolve()),
        preprocess_report_sha256=sha256_file(new_report_path),
        postprocess_schema=POSTPROCESS_SCHEMA,
        postprocess={
            "schema": POSTPROCESS_SCHEMA,
            "signature": signature,
            "checkpoint_sha256": checkpoint_sha256,
            "upstream_manifest_sha256": sha256_file(manifest_file),
            "report_path": str(new_report_path.resolve()),
            "report_sha256": sha256_file(new_report_path),
            "production_eligible": not test_only,
        },
    )
    _atomic_write_json(new_manifest_path, manifest)

    if not test_only:
        validate_manifest_payload(
            manifest, manifest_path=new_manifest_path, require_provenance=True
        )
    summary = {
        "schema": POSTPROCESS_SCHEMA,
        "manifest": str(new_manifest_path.resolve()),
        "manifest_sha256": sha256_file(new_manifest_path),
        "report": str(new_report_path.resolve()),
        "report_sha256": sha256_file(new_report_path),
        "signature": signature,
        "generation_id": generation_id,
        "checkpoint_sha256": checkpoint_sha256,
        "input_synthetic_count": len(decisions),
        "replace_count": sum(
            item["action"] == "replace_with_evssm" for item in decisions
        ),
        "fallback_count": sum(
            item["action"] == "fallback_to_framecrafter" for item in decisions
        ),
        "reject_count": len(newly_rejected),
        "output_synthetic_count": generated_count,
        "production_eligible": not test_only,
        "decisions": [
            _decision_summary(
                item,
                str(item["action"]),
                (
                    str(item["candidate"]["rgb_sha256"])
                    if item["action"] == "replace_with_evssm"
                    else str(item["upstream"]["rgb_sha256"])
                    if item["action"] == "fallback_to_framecrafter"
                    else None
                ),
            )
            for item in decisions
        ],
    }
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Immutable FrameCrafter-to-EVSSM synthetic-frame postprocess"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help=(
            "Explicit checkpoint identity. Production defaults to the official "
            "Unblur-SLAM motion EVSSM SHA-256."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--failure-policy", choices=("fallback", "reject"), default="fallback"
    )
    parser.add_argument("--min-post-vs-pre-sharpness-gain", type=float, default=1.0)
    parser.add_argument("--min-reference-sharpness-gain", type=float, default=1.05)
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--min-depth-coverage", type=float, default=0.05)
    parser.add_argument("--min-depth-consistency", type=float, default=0.50)
    parser.add_argument("--max-photometric-error", type=float, default=0.20)
    parser.add_argument("--max-reprojection-error-px", type=float, default=2.0)
    parser.add_argument("--min-reprojection-valid-ratio", type=float, default=0.05)
    parser.add_argument("--depth-abs-tolerance", type=float, default=0.03)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.03)
    parser.add_argument("--allow-missing-depth-gates", action="store_true")
    parser.add_argument("--local-tile-size", type=int, default=32)
    parser.add_argument("--local-tile-stride", type=int, default=16)
    parser.add_argument("--local-max-brightness-drop", type=float, default=0.30)
    parser.add_argument("--local-min-edge-retention", type=float, default=0.50)
    parser.add_argument("--local-min-laplacian-retention", type=float, default=0.50)
    parser.add_argument("--local-max-tile-mae", type=float, default=0.20)
    parser.add_argument("--local-max-dark-expansion", type=float, default=0.30)
    parser.add_argument(
        "--local-dark-luma-threshold", type=float, default=96.0 / 255.0
    )
    parser.add_argument("--local-min-pre-luma", type=float, default=0.10)
    parser.add_argument("--local-min-pre-edge", type=float, default=0.01)
    parser.add_argument("--local-min-pre-laplacian", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = PostprocessConfig(
        failure_policy=args.failure_policy,
        min_post_vs_pre_sharpness_gain=args.min_post_vs_pre_sharpness_gain,
        min_reference_sharpness_gain=args.min_reference_sharpness_gain,
        depth_scale=args.depth_scale,
        min_depth_coverage=args.min_depth_coverage,
        min_depth_consistency=args.min_depth_consistency,
        max_photometric_error=args.max_photometric_error,
        max_reprojection_error_px=args.max_reprojection_error_px,
        min_reprojection_valid_ratio=args.min_reprojection_valid_ratio,
        depth_abs_tolerance=args.depth_abs_tolerance,
        depth_rel_tolerance=args.depth_rel_tolerance,
        require_depth=not args.allow_missing_depth_gates,
        local_tile_size=args.local_tile_size,
        local_tile_stride=args.local_tile_stride,
        local_max_brightness_drop=args.local_max_brightness_drop,
        local_min_edge_retention=args.local_min_edge_retention,
        local_min_laplacian_retention=args.local_min_laplacian_retention,
        local_max_tile_mae=args.local_max_tile_mae,
        local_max_dark_expansion=args.local_max_dark_expansion,
        local_dark_luma_threshold=args.local_dark_luma_threshold,
        local_min_pre_luma=args.local_min_pre_luma,
        local_min_pre_edge=args.local_min_pre_edge,
        local_min_pre_laplacian=args.local_min_pre_laplacian,
    )
    summary = postprocess(
        manifest_path=args.manifest,
        report_path=args.report,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
