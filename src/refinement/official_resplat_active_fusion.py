"""Fail-closed protocol helpers for one synchronous ReSplat state3 fusion.

These helpers deliberately contain no renderer and create no CUDA context.
The mapper supplies reconstruction summaries computed from its eight online
context observations; clear/reference images are never accepted as gate input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence


ACTIVE_FUSION_CONFIG_SCHEMA = (
    "unblur_slam.official_resplat_active_fusion_config.v1"
)
FORCED_COMMIT_DIAGNOSTIC_CONFIG_SCHEMA = (
    "unblur_slam.official_resplat_active_fusion_config.v2_forced_commit_diagnostic"
)
COUNT_AGNOSTIC_FORCED_COMMIT_CONFIG_SCHEMA = (
    "unblur_slam.official_resplat_active_fusion_config."
    "v3_count_agnostic_forced_commit_diagnostic"
)
VISIBILITY_CACHE_REFRESH_FORCED_COMMIT_CONFIG_SCHEMA = (
    "unblur_slam.official_resplat_active_fusion_config."
    "v4_visibility_cache_refresh_forced_commit_diagnostic"
)
ACTIVE_FUSION_AUDIT_SCHEMA = "unblur_slam.official_resplat_active_fusion_audit.v1"
EXPECTED_CONTEXT_KEYFRAMES = 8
EXPECTED_REFINEMENT_UPDATES = 3
EXPECTED_SELECTED_STATE_INDEX = 2
EXPECTED_MAPPED_SOURCE_INDICES = (15, 49, 58, 72, 89, 109, 125, 166)


def canonical_contract_sha256(
    value: Mapping[str, Any], *, digest_field: str = "event_sha256"
) -> str:
    """Hash one JSON contract while excluding its self-referential digest.

    The function is intentionally CPU-only and shared by the mapper, reporter,
    and tests.  ``allow_nan=False`` makes every persisted link fail closed on
    non-finite values rather than silently producing non-standard JSON.
    """

    payload = dict(value)
    payload.pop(str(digest_field), None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stamp_contract_sha256(
    value: Mapping[str, Any], *, digest_field: str = "event_sha256"
) -> dict[str, Any]:
    """Return a copy carrying its canonical self-digest."""

    result = dict(value)
    result[str(digest_field)] = canonical_contract_sha256(
        result, digest_field=digest_field
    )
    return result


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    try:
        exact = float(value) == float(result)
    except (TypeError, ValueError):
        exact = False
    if not exact or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class ActiveFusionConfig:
    """Validated experiment settings, including every preregistered gate."""

    enabled: bool
    output_root: str
    python_executable: str
    runner_script: str
    resplat_repo: str
    resplat_repo_commit: str
    checkpoint: str
    expected_checkpoint_sha256: str
    cuda_visible_devices: str
    process_device: str
    near: float
    far: float
    max_runtime_seconds: float
    geometry_gate: Mapping[str, Any]
    sidecar_quality_gate: Mapping[str, Any]
    merge: Mapping[str, Any]
    postmerge_quality_gate: Mapping[str, Any]
    schema: str = ACTIVE_FUSION_CONFIG_SCHEMA
    synchronous: bool = True
    trigger_keyframe_count: int = EXPECTED_CONTEXT_KEYFRAMES
    trigger_source_index: int = 166
    expected_mapped_source_indices: Sequence[int] = EXPECTED_MAPPED_SOURCE_INDICES
    max_fusion_attempts: int = 1
    refinement_updates: int = EXPECTED_REFINEMENT_UPDATES
    selected_state_index_zero_based: int = EXPECTED_SELECTED_STATE_INDEX
    fourth_state_computed: bool = False
    uses_ground_truth: bool = False
    selection_membership_clear_gt_conditioned: bool = True
    uses_ground_truth_poses_or_depths_for_fusion: bool = False
    uses_independent_clear_pixels_for_fusion: bool = False
    uses_clear_gt_metrics: bool = False
    model_preset: str = "dl3dv_8v_256x448_small"
    max_pose_revision_lag: int = 0
    max_pose_translation_drift: float = 0.0
    max_pose_rotation_drift_deg: float = 0.0
    posthoc_after_v2_rejection: bool = False
    unsafe_not_deployable: bool = False
    gate_thresholds_unchanged: bool = False
    force_commit_after_postmerge_rejection: bool = False
    expected_forced_commit_gaussian_count: int = 0
    v2_rejection_audit_sha256: str = "0" * 64
    posthoc_after_v3_count_mismatch: bool = False
    count_agnostic_forced_commit: bool = False
    v3_count_mismatch_audit_sha256: str = "0" * 64
    posthoc_after_v4_visibility_cache_mismatch: bool = False
    refresh_occ_aware_visibility_after_forced_commit: bool = False
    v4_visibility_cache_failure_audit_sha256: str = "0" * 64

    def __post_init__(self) -> None:
        diagnostic_flags = (
            self.posthoc_after_v2_rejection,
            self.unsafe_not_deployable,
            self.gate_thresholds_unchanged,
            self.force_commit_after_postmerge_rejection,
        )
        diagnostic = any(diagnostic_flags)
        count_agnostic = bool(
            self.posthoc_after_v3_count_mismatch
            or self.count_agnostic_forced_commit
            or self.v3_count_mismatch_audit_sha256 != "0" * 64
        )
        cache_refresh = bool(
            self.posthoc_after_v4_visibility_cache_mismatch
            or self.refresh_occ_aware_visibility_after_forced_commit
            or self.v4_visibility_cache_failure_audit_sha256 != "0" * 64
        )
        if cache_refresh and not count_agnostic:
            raise ValueError(
                "cache-refresh diagnostic requires the complete count-agnostic "
                "forced-commit contract"
            )
        if cache_refresh:
            if self.posthoc_after_v4_visibility_cache_mismatch is not True:
                raise ValueError(
                    "cache-refresh diagnostic must disclose the v4 mismatch"
                )
            if self.refresh_occ_aware_visibility_after_forced_commit is not True:
                raise ValueError(
                    "cache-refresh diagnostic must explicitly enable refresh"
                )
            v4_digest = str(self.v4_visibility_cache_failure_audit_sha256)
            if v4_digest == "0" * 64 or len(v4_digest) != 64 or any(
                ch not in "0123456789abcdef" for ch in v4_digest
            ):
                raise ValueError(
                    "v4 visibility-cache failure audit digest must be lowercase "
                    "SHA-256"
                )
        expected_schema = (
            VISIBILITY_CACHE_REFRESH_FORCED_COMMIT_CONFIG_SCHEMA
            if cache_refresh
            else (
                COUNT_AGNOSTIC_FORCED_COMMIT_CONFIG_SCHEMA
                if count_agnostic
                else (
                    FORCED_COMMIT_DIAGNOSTIC_CONFIG_SCHEMA
                    if diagnostic
                    else ACTIVE_FUSION_CONFIG_SCHEMA
                )
            )
        )
        if self.schema != expected_schema:
            raise ValueError("wrong official active-fusion config schema")
        if diagnostic:
            if not all(flag is True for flag in diagnostic_flags):
                raise ValueError(
                    "forced-commit diagnostic requires every unsafe/posthoc flag true"
                )
            if self.enabled is not True:
                raise ValueError("forced-commit diagnostic must enable active fusion")
            if count_agnostic:
                if self.posthoc_after_v3_count_mismatch is not True:
                    raise ValueError(
                        "count-agnostic diagnostic must disclose the v3 count mismatch"
                    )
                if self.count_agnostic_forced_commit is not True:
                    raise ValueError(
                        "count-agnostic diagnostic flag must be explicitly true"
                    )
                if self.expected_forced_commit_gaussian_count != 0:
                    raise ValueError(
                        "count-agnostic diagnostic forbids a cross-run exact count"
                    )
                v3_digest = str(self.v3_count_mismatch_audit_sha256)
                if v3_digest == "0" * 64 or len(v3_digest) != 64 or any(
                    ch not in "0123456789abcdef" for ch in v3_digest
                ):
                    raise ValueError(
                        "v3 count-mismatch audit digest must be lowercase SHA-256"
                    )
                if not cache_refresh and (
                    self.posthoc_after_v4_visibility_cache_mismatch is not False
                    or self.refresh_occ_aware_visibility_after_forced_commit
                    is not False
                    or self.v4_visibility_cache_failure_audit_sha256 != "0" * 64
                ):
                    raise ValueError(
                        "v4 count-only diagnostic cannot carry v5 cache metadata"
                    )
            elif _positive_int(
                self.expected_forced_commit_gaussian_count,
                "expected_forced_commit_gaussian_count",
            ) != 5_554:
                raise ValueError("forced-commit diagnostic is pinned to 5554 candidates")
            v2_digest = str(self.v2_rejection_audit_sha256)
            if len(v2_digest) != 64 or any(
                ch not in "0123456789abcdef" for ch in v2_digest
            ):
                raise ValueError("v2 rejection audit digest must be lowercase SHA-256")
        elif (
            self.expected_forced_commit_gaussian_count != 0
            or self.v2_rejection_audit_sha256 != "0" * 64
            or self.posthoc_after_v3_count_mismatch is not False
            or self.count_agnostic_forced_commit is not False
            or self.v3_count_mismatch_audit_sha256 != "0" * 64
            or self.posthoc_after_v4_visibility_cache_mismatch is not False
            or self.refresh_occ_aware_visibility_after_forced_commit is not False
            or self.v4_visibility_cache_failure_audit_sha256 != "0" * 64
            or any(flag is not False for flag in diagnostic_flags)
        ):
            raise ValueError("ordinary active fusion cannot carry forced-commit metadata")
        if not isinstance(self.enabled, bool):
            raise ValueError("active-fusion enabled must be boolean")
        if self.synchronous is not True:
            raise ValueError("first experiment requires synchronous fusion")
        if self.trigger_keyframe_count != EXPECTED_CONTEXT_KEYFRAMES:
            raise ValueError("active fusion must trigger after exactly 8 mapped keyframes")
        if self.trigger_source_index != 166:
            raise ValueError("fixed-11KF experiment must trigger on mapped source index 166")
        observed_sources = tuple(int(value) for value in self.expected_mapped_source_indices)
        if observed_sources != EXPECTED_MAPPED_SOURCE_INDICES:
            raise ValueError(
                "active fusion must use the preregistered first eight actually mapped "
                f"sources {EXPECTED_MAPPED_SOURCE_INDICES}"
            )
        if self.max_fusion_attempts != 1:
            raise ValueError("active-fusion experiment permits exactly one attempt")
        if self.refinement_updates != EXPECTED_REFINEMENT_UPDATES:
            raise ValueError("active fusion must select official ReSplat state3")
        if self.selected_state_index_zero_based != EXPECTED_SELECTED_STATE_INDEX:
            raise ValueError("state3 must use zero-based recurrent state index 2")
        if self.fourth_state_computed is not False:
            raise ValueError("state3 experiment forbids computing recurrent state4")
        if self.selection_membership_clear_gt_conditioned is not True:
            raise ValueError(
                "fixed schedule must disclose historical clear-GT-conditioned membership"
            )
        if (
            self.uses_ground_truth
            or self.uses_ground_truth_poses_or_depths_for_fusion
            or self.uses_independent_clear_pixels_for_fusion
            or self.uses_clear_gt_metrics
        ):
            raise ValueError(
                "fusion may not consume GT pose/depth, independent clear pixels, or clear metrics"
            )
        if self.model_preset != "dl3dv_8v_256x448_small":
            raise ValueError("active fusion is pinned to official small8v preset")
        if self.process_device != "cuda:0":
            raise ValueError("ReSplat child process device must be cuda:0")
        if not 0.0 < _finite(self.near, "near") < _finite(self.far, "far"):
            raise ValueError("near/far must satisfy 0 < near < far")
        if not 0.0 < _finite(self.max_runtime_seconds, "max_runtime_seconds") <= 3600.0:
            raise ValueError("max_runtime_seconds must be in (0,3600]")
        if self.max_pose_revision_lag != 0:
            raise ValueError("synchronous fusion requires zero pose revision lag")
        if self.max_pose_translation_drift != 0.0 or self.max_pose_rotation_drift_deg != 0.0:
            raise ValueError("synchronous fusion requires exact pose-hash stability")
        digest = str(self.expected_checkpoint_sha256)
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("official ReSplat checkpoint digest must be lowercase SHA-256")
        if len(str(self.resplat_repo_commit)) != 40 or any(
            ch not in "0123456789abcdef" for ch in str(self.resplat_repo_commit)
        ):
            raise ValueError("official ReSplat commit must be 40 lowercase hex")
        if self.enabled:
            required = (
                "python_executable",
                "runner_script",
                "resplat_repo",
                "checkpoint",
                "cuda_visible_devices",
            )
            missing = [name for name in required if not str(getattr(self, name))]
            if missing:
                raise ValueError("enabled active fusion is missing: " + ", ".join(missing))
        self._validate_geometry_gate()
        self._validate_sidecar_quality_gate()
        self._validate_merge()
        self._validate_postmerge_gate()

    def _validate_geometry_gate(self) -> None:
        gate = dict(self.geometry_gate)
        expected_count = _positive_int(
            gate.get("expected_gaussian_count"), "expected_gaussian_count"
        )
        if expected_count != 71_680:
            raise ValueError("official 8x320x448/4 topology must contain 71680 Gaussians")
        if _finite(gate.get("min_finite_fraction"), "min_finite_fraction") != 1.0:
            raise ValueError("geometry gate requires every value to be finite")
        expected = {
            "max_p95_distance_from_pivot": 50.0,
            "max_distance_from_pivot": 200.0,
            "max_p95_scale": 5.0,
            "max_scale": 25.0,
            "max_quaternion_norm_deviation": 0.001,
        }
        for name, value in expected.items():
            if _finite(gate.get(name), name) != value:
                raise ValueError(f"preregistered geometry gate {name} drifted")

    def _validate_sidecar_quality_gate(self) -> None:
        gate = dict(self.sidecar_quality_gate)
        if gate.get("require_metrics") is not True:
            raise ValueError("state3 context reconstruction metrics are required")
        if gate.get("metric_inputs") != "eight_context_observations":
            raise ValueError("sidecar quality gate may use only eight context observations")
        if gate.get("require_state3_composite_nonworse_than_init0") is not True:
            raise ValueError("state3 must be non-worse than init0 on the preregistered score")
        expected = {
            "minimum_state3_mean_psnr_db": 12.0,
            "minimum_state3_mean_ssim": 0.50,
            "maximum_state3_mean_l1": 0.30,
            "composite_l1_weight": 0.80,
            "composite_one_minus_ssim_weight": 0.20,
        }
        for name, value in expected.items():
            if _finite(gate.get(name), name) != value:
                raise ValueError(f"preregistered sidecar quality gate {name} drifted")
        l1_weight = _finite(gate.get("composite_l1_weight"), "composite_l1_weight")
        ssim_weight = _finite(
            gate.get("composite_one_minus_ssim_weight"),
            "composite_one_minus_ssim_weight",
        )
        if not math.isclose(l1_weight + ssim_weight, 1.0, abs_tol=1e-12):
            raise ValueError("sidecar composite weights must sum to one")

    def _validate_merge(self) -> None:
        from src.refinement.active_map_merge import ActiveMapMergeConfig

        parsed = ActiveMapMergeConfig.from_value(self.merge)
        expected = {
            "mode": "append",
            "min_opacity": 0.5,
            "min_scale": 1.0e-6,
            "max_scale": 0.2,
            "voxel_size": 0.01,
            "max_new_gaussians": 20_000,
            "min_new_gaussians": 1_024,
            "zero_sh_rest": True,
            "max_abs_position": 200.0,
        }
        record = asdict(parsed)
        for key, value in expected.items():
            if record[key] != value:
                raise ValueError(f"preregistered merge setting {key} drifted")

    def _validate_postmerge_gate(self) -> None:
        gate = dict(self.postmerge_quality_gate)
        if gate.get("metric_inputs") != "same_eight_context_observations":
            raise ValueError("postmerge gate must reuse the same eight observations")
        if (
            gate.get("renderer_proxy")
            != "exposure_compensated_midpoint_without_learned_blur"
            or gate.get("purpose") != "isolate_active_gaussian_map_change"
        ):
            raise ValueError("postmerge renderer-proxy disclosure drifted")
        if gate.get("rollback_on_rejection") is not True:
            raise ValueError("postmerge rejection must roll back the append")
        expected = {
            "l1_weight": 0.80,
            "one_minus_ssim_weight": 0.20,
            "maximum_relative_mean_composite_increase": 0.005,
            "maximum_absolute_per_view_composite_increase": 0.02,
        }
        for name, value in expected.items():
            if _finite(gate.get(name), name) != value:
                raise ValueError(f"preregistered postmerge gate {name} drifted")
        l1_weight = _finite(gate.get("l1_weight"), "postmerge l1_weight")
        ssim_weight = _finite(
            gate.get("one_minus_ssim_weight"), "postmerge one_minus_ssim_weight"
        )
        if not math.isclose(l1_weight + ssim_weight, 1.0, abs_tol=1e-12):
            raise ValueError("postmerge composite weights must sum to one")
        if _finite(
            gate.get("maximum_relative_mean_composite_increase"),
            "maximum_relative_mean_composite_increase",
        ) < 0.0:
            raise ValueError("postmerge relative tolerance must be non-negative")
        if _finite(
            gate.get("maximum_absolute_per_view_composite_increase"),
            "maximum_absolute_per_view_composite_increase",
        ) < 0.0:
            raise ValueError("postmerge per-view tolerance must be non-negative")

    @classmethod
    def from_dict(
        cls, value: Optional[Mapping[str, Any]], *, default_output_root: Path | str
    ) -> "ActiveFusionConfig":
        defaults: dict[str, Any] = {
            "enabled": False,
            "output_root": str(Path(default_output_root).expanduser().resolve()),
            "python_executable": "",
            "runner_script": "",
            "resplat_repo": "",
            "resplat_repo_commit": "0" * 40,
            "checkpoint": "",
            "expected_checkpoint_sha256": "0" * 64,
            "cuda_visible_devices": "",
            "process_device": "cuda:0",
            "near": 0.01,
            "far": 200.0,
            "max_runtime_seconds": 60.0,
            "geometry_gate": {
                "expected_gaussian_count": 71_680,
                "min_finite_fraction": 1.0,
                "max_p95_distance_from_pivot": 50.0,
                "max_distance_from_pivot": 200.0,
                "max_p95_scale": 5.0,
                "max_scale": 25.0,
                "max_quaternion_norm_deviation": 0.001,
            },
            "sidecar_quality_gate": {
                "require_metrics": True,
                "metric_inputs": "eight_context_observations",
                "minimum_state3_mean_psnr_db": 12.0,
                "minimum_state3_mean_ssim": 0.50,
                "maximum_state3_mean_l1": 0.30,
                "require_state3_composite_nonworse_than_init0": True,
                "composite_l1_weight": 0.80,
                "composite_one_minus_ssim_weight": 0.20,
            },
            "merge": {
                "mode": "append",
                "min_opacity": 0.5,
                "min_scale": 1.0e-6,
                "max_scale": 0.2,
                "voxel_size": 0.01,
                "max_new_gaussians": 20_000,
                "min_new_gaussians": 1_024,
                "zero_sh_rest": True,
                "max_abs_position": 200.0,
            },
            "postmerge_quality_gate": {
                "metric_inputs": "same_eight_context_observations",
                "renderer_proxy": "exposure_compensated_midpoint_without_learned_blur",
                "purpose": "isolate_active_gaussian_map_change",
                "l1_weight": 0.80,
                "one_minus_ssim_weight": 0.20,
                "maximum_relative_mean_composite_increase": 0.005,
                "maximum_absolute_per_view_composite_increase": 0.02,
                "rollback_on_rejection": True,
            },
            "posthoc_after_v2_rejection": False,
            "unsafe_not_deployable": False,
            "gate_thresholds_unchanged": False,
            "force_commit_after_postmerge_rejection": False,
            "expected_forced_commit_gaussian_count": 0,
            "v2_rejection_audit_sha256": "0" * 64,
            "posthoc_after_v3_count_mismatch": False,
            "count_agnostic_forced_commit": False,
            "v3_count_mismatch_audit_sha256": "0" * 64,
            "posthoc_after_v4_visibility_cache_mismatch": False,
            "refresh_occ_aware_visibility_after_forced_commit": False,
            "v4_visibility_cache_failure_audit_sha256": "0" * 64,
        }
        raw = {**defaults, **dict(value or {})}
        if not raw.get("output_root"):
            raw["output_root"] = str(Path(default_output_root).expanduser().resolve())
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("unknown active-fusion config fields: " + ", ".join(sorted(unknown)))
        return cls(**raw)


def forced_commit_count_contract(
    *,
    accepted_count: int,
    before_count: int,
    after_count: int,
    config: ActiveFusionConfig,
) -> dict[str, Any]:
    """Evaluate the executable v3/v4 forced-commit count contract on CPU."""

    values = {}
    for name, value in (
        ("accepted_count", accepted_count),
        ("before_count", before_count),
        ("after_count", after_count),
    ):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be an integer") from error
        if parsed < 0 or parsed != value:
            raise ValueError(f"{name} must be a nonnegative integer")
        values[name] = parsed
    minimum = int(config.merge["min_new_gaussians"])
    maximum = int(config.merge["max_new_gaussians"])
    within_bounds = minimum <= values["accepted_count"] <= maximum
    exact_cross_run_match = (
        None
        if config.count_agnostic_forced_commit
        else values["accepted_count"]
        == int(config.expected_forced_commit_gaussian_count)
    )
    algebra_matches = (
        values["after_count"] - values["before_count"]
        == values["accepted_count"]
    )
    reasons = []
    if not within_bounds:
        reasons.append("accepted_count_outside_preregistered_merge_bounds")
    if exact_cross_run_match is False:
        reasons.append("accepted_count_does_not_match_exact_v3_contract")
    if not algebra_matches:
        reasons.append("after_minus_before_does_not_equal_accepted_count")
    return {
        "schema": "unblur_slam.forced_commit_count_contract.v1",
        "accepted": not reasons,
        "kind": (
            "fresh_run_merge_bounds_and_internal_count_algebra"
            if config.count_agnostic_forced_commit
            else "cross_run_exact_count_and_internal_count_algebra"
        ),
        "cross_run_exact_count_required": not bool(
            config.count_agnostic_forced_commit
        ),
        "minimum_accepted_count": minimum,
        "maximum_accepted_count": maximum,
        "fresh_run_accepted_count": values["accepted_count"],
        "before_count": values["before_count"],
        "after_count": values["after_count"],
        "within_preregistered_merge_bounds": within_bounds,
        "exact_cross_run_count_requirement_satisfied": exact_cross_run_match,
        "after_minus_before_equals_accepted_count": algebra_matches,
        "reasons": reasons,
    }


def context_reconstruction_gate(
    manifest: Mapping[str, Any],
    config: ActiveFusionConfig,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate state3 self-reconstruction without reading clear/reference GT."""

    reasons: list[str] = []
    record = manifest.get("context_reconstruction") or {}
    if record.get("uses_clear_gt") is not False:
        reasons.append("context_reconstruction_not_declared_clear_gt_free")
    if record.get("inputs") != "eight_context_observations":
        reasons.append("wrong_context_reconstruction_inputs")
    if record.get("same_observations_for_init0_and_state3") is not True:
        reasons.append("init0_state3_observation_binding_missing")
    summaries: dict[str, dict[str, float]] = {}
    identities: dict[str, list[tuple[Any, int]]] = {}
    for stage in ("init0", "state3"):
        raw = record.get(stage) or {}
        try:
            declared = {
                "mean_psnr_db": _finite(raw.get("mean_psnr_db"), f"{stage} PSNR"),
                "mean_ssim": _finite(raw.get("mean_ssim"), f"{stage} SSIM"),
                "mean_l1": _finite(raw.get("mean_l1"), f"{stage} L1"),
            }
        except ValueError as error:
            reasons.append(f"invalid_{stage}_metrics:{error}")
            declared = {"mean_psnr_db": -math.inf, "mean_ssim": -math.inf, "mean_l1": math.inf}
        per_view = raw.get("per_view")
        if not isinstance(per_view, list) or len(per_view) != EXPECTED_CONTEXT_KEYFRAMES:
            reasons.append(f"{stage}_per_view_count_not_8")
            identities[stage] = []
            summary = {"mean_psnr_db": -math.inf, "mean_ssim": -math.inf, "mean_l1": math.inf}
        else:
            try:
                identities[stage] = [
                    (item["frame_id"], int(item["sequence_ordinal"]))
                    for item in per_view
                ]
            except (KeyError, TypeError, ValueError):
                reasons.append(f"{stage}_per_view_identity_invalid")
                identities[stage] = []
            if len(set(identities[stage])) != EXPECTED_CONTEXT_KEYFRAMES:
                reasons.append(f"{stage}_per_view_identity_not_unique")
            try:
                view_metrics = [
                    {
                        "psnr_db": _finite(item.get("psnr_db"), f"{stage} view PSNR"),
                        "ssim": _finite(item.get("ssim"), f"{stage} view SSIM"),
                        "l1": _finite(item.get("l1"), f"{stage} view L1"),
                    }
                    for item in per_view
                ]
            except (AttributeError, ValueError) as error:
                reasons.append(f"invalid_{stage}_per_view_metrics:{error}")
                view_metrics = []
            if view_metrics and any(
                item["psnr_db"] < 0.0
                or not -1.0 <= item["ssim"] <= 1.0
                or not 0.0 <= item["l1"] <= 1.0
                for item in view_metrics
            ):
                reasons.append(f"{stage}_per_view_metrics_out_of_range")
                view_metrics = []
            if view_metrics:
                summary = {
                    "mean_psnr_db": sum(item["psnr_db"] for item in view_metrics)
                    / len(view_metrics),
                    "mean_ssim": sum(item["ssim"] for item in view_metrics)
                    / len(view_metrics),
                    "mean_l1": sum(item["l1"] for item in view_metrics)
                    / len(view_metrics),
                }
                for name in summary:
                    if not math.isclose(
                        summary[name],
                        declared[name],
                        rel_tol=0.0,
                        abs_tol=1e-5 if name == "mean_psnr_db" else 1e-6,
                    ):
                        reasons.append(f"{stage}_{name}_does_not_match_per_view")
            else:
                summary = {
                    "mean_psnr_db": -math.inf,
                    "mean_ssim": -math.inf,
                    "mean_l1": math.inf,
                }
        summaries[stage] = summary

    if identities.get("init0") != identities.get("state3"):
        reasons.append("init0_state3_view_identity_or_order_mismatch")
    snapshot_frames = snapshot.get("frames")
    if not isinstance(snapshot_frames, list) or len(snapshot_frames) != EXPECTED_CONTEXT_KEYFRAMES:
        reasons.append("snapshot_context_identity_not_8")
        expected_identity: list[tuple[Any, int]] = []
    else:
        try:
            expected_identity = [
                (item["frame_id"], int(item["sequence_ordinal"]))
                for item in snapshot_frames
            ]
        except (KeyError, TypeError, ValueError):
            expected_identity = []
            reasons.append("snapshot_context_identity_invalid")
    if identities.get("init0") != expected_identity:
        reasons.append("context_reconstruction_snapshot_identity_mismatch")

    gate = dict(config.sidecar_quality_gate)
    state3 = summaries["state3"]
    if state3["mean_psnr_db"] < _finite(
        gate["minimum_state3_mean_psnr_db"], "minimum state3 PSNR"
    ):
        reasons.append("state3_mean_psnr_below_gate")
    if state3["mean_ssim"] < _finite(
        gate["minimum_state3_mean_ssim"], "minimum state3 SSIM"
    ):
        reasons.append("state3_mean_ssim_below_gate")
    if state3["mean_l1"] > _finite(
        gate["maximum_state3_mean_l1"], "maximum state3 L1"
    ):
        reasons.append("state3_mean_l1_above_gate")
    l1_weight = float(gate["composite_l1_weight"])
    ssim_weight = float(gate["composite_one_minus_ssim_weight"])

    def composite(summary: Mapping[str, float]) -> float:
        return l1_weight * summary["mean_l1"] + ssim_weight * (
            1.0 - summary["mean_ssim"]
        )

    init_score = composite(summaries["init0"])
    state3_score = composite(state3)
    if not math.isfinite(init_score) or not math.isfinite(state3_score):
        reasons.append("non_finite_context_composite")
    elif state3_score > init_score + 1e-8:
        reasons.append("state3_composite_worse_than_init0")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "uses_clear_gt": False,
        "inputs": "eight_context_observations",
        "same_observations_for_init0_and_state3": True,
        "view_identity": [
            {"frame_id": frame_id, "sequence_ordinal": ordinal}
            for frame_id, ordinal in identities.get("state3", [])
        ],
        "init0": summaries["init0"],
        "state3": summaries["state3"],
        "init0_composite": init_score,
        "state3_composite": state3_score,
    }


def postmerge_reconstruction_gate(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    config: ActiveFusionConfig,
) -> dict[str, Any]:
    """Decide whether a trial append remains in the active map."""

    before_views = before.get("per_view")
    after_views = after.get("per_view")
    if not isinstance(before_views, list) or not isinstance(after_views, list):
        raise ValueError("postmerge summaries require per_view lists")
    if len(before_views) != EXPECTED_CONTEXT_KEYFRAMES or len(after_views) != len(before_views):
        raise ValueError("postmerge gate requires the same eight views before and after")
    before_ids = [item.get("source_index") for item in before_views]
    after_ids = [item.get("source_index") for item in after_views]
    if before_ids != after_ids or len(set(before_ids)) != EXPECTED_CONTEXT_KEYFRAMES:
        raise ValueError("postmerge view identity/order changed")
    before_mean = _finite(before.get("mean_composite"), "before mean composite")
    after_mean = _finite(after.get("mean_composite"), "after mean composite")
    gate = dict(config.postmerge_quality_gate)
    relative = (after_mean - before_mean) / max(abs(before_mean), 1e-12)
    deltas = [
        _finite(right.get("composite"), "after view composite")
        - _finite(left.get("composite"), "before view composite")
        for left, right in zip(before_views, after_views)
    ]
    max_delta = max(deltas)
    reasons = []
    if relative > float(gate["maximum_relative_mean_composite_increase"]):
        reasons.append("relative_mean_composite_increase_exceeded")
    if max_delta > float(gate["maximum_absolute_per_view_composite_increase"]):
        reasons.append("per_view_composite_increase_exceeded")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "uses_clear_gt": False,
        "inputs": "same_eight_context_observations",
        "renderer_proxy": "exposure_compensated_midpoint_without_learned_blur",
        "before_mean_composite": before_mean,
        "after_mean_composite": after_mean,
        "relative_mean_composite_change": relative,
        "maximum_per_view_composite_increase": max_delta,
        "per_view_composite_deltas": deltas,
    }


def owner_ids_for_fixed_topology(
    frame_ids: Sequence[int], gaussian_count: int
) -> list[int]:
    """Bind each contiguous official-view block to its online mapper owner."""

    ids = [int(value) for value in frame_ids]
    if len(ids) != EXPECTED_CONTEXT_KEYFRAMES or len(set(ids)) != len(ids):
        raise ValueError("official state3 ownership needs eight unique frame ids")
    count = _positive_int(gaussian_count, "gaussian_count")
    if count % EXPECTED_CONTEXT_KEYFRAMES:
        raise ValueError("official fixed topology is not divisible by eight views")
    per_view = count // EXPECTED_CONTEXT_KEYFRAMES
    return [frame_id for frame_id in ids for _ in range(per_view)]


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "ACTIVE_FUSION_AUDIT_SCHEMA",
    "ACTIVE_FUSION_CONFIG_SCHEMA",
    "ActiveFusionConfig",
    "atomic_write_json",
    "context_reconstruction_gate",
    "owner_ids_for_fixed_topology",
    "postmerge_reconstruction_gate",
]
