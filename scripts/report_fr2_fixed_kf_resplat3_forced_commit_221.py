#!/usr/bin/env python3
"""Fail-closed report for the unsafe post-hoc v3 forced-commit diagnostic.

The only fresh arm is v3 EVSSM+official-ReSplat.  The baseline and safe
rollback arm are immutable v2 references, not contemporaneous reruns.  This
report therefore exposes diagnostic deltas but never labels them a fair pair,
a deployable method, or proof that any individual imported Gaussian survived.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_module(
    "forced_commit_v3_runner_contract",
    REPO_ROOT / "scripts/run_fr2_fixed_kf_resplat3_forced_commit_221.py",
)
V2_REPORT = _load_module(
    "forced_commit_v2_report_contract",
    REPO_ROOT / "scripts/report_fr2_fixed_kf_resplat3_fusion_221.py",
)
BASE = V2_REPORT.BASE
ContractError = BASE.ContractError
DEFAULT_ROOT = RUNNER.OUTPUT_ROOT
SCENE = "freiburg2_xyz"


def _expect(value: Any, expected: Any, label: str) -> None:
    BASE._expect(value, expected, label)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return BASE._load_json(path, label)


def _verify_file_binding(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    observed = BASE._sha256_file(path)
    _expect(record.get("sha256"), observed, f"{label} SHA-256")
    return path


def _verify_event(path: Path, sequence: int, event_type: str) -> dict[str, Any]:
    from src.refinement.official_resplat_active_fusion import (
        canonical_contract_sha256,
    )

    event = _load_json(path, f"forced chain event {sequence}")
    _expect(event.get("schema"), "unblur_slam.forced_commit_chain_event.v1", f"event {sequence} schema")
    _expect(event.get("sequence"), sequence, f"event {sequence} sequence")
    _expect(event.get("event_type"), event_type, f"event {sequence} type")
    _expect(
        event.get("event_sha256"),
        canonical_contract_sha256(event),
        f"event {sequence} canonical digest",
    )
    return event


def _validate_v3_config_and_preflight(
    cfg: Mapping[str, Any], preflight: Mapping[str, Any], arm_root: Path
) -> dict[str, Any]:
    _, v2 = RUNNER._load_configs()
    fusion, differences = RUNNER._validate_v3_config(cfg, v2)
    _expect(
        Path(str((cfg.get("data") or {}).get("output", ""))).resolve(),
        arm_root.resolve(),
        "v3 resolved output root",
    )
    _expect(
        preflight.get("schema"),
        "unblur_slam.fr2_xyz_resplat3_forced_commit_221_preflight.v3",
        "v3 preflight schema",
    )
    scope = preflight.get("scope") or {}
    for field in ("posthoc_after_v2_rejection", "unsafe_not_deployable"):
        _expect(scope.get(field), True, f"v3 preflight {field}")
    _expect(scope.get("single_new_fused_arm_only"), True, "v3 single-arm scope")
    _expect(scope.get("fresh_paired_comparison"), False, "v3 non-paired scope")
    scientific = preflight.get("scientific_contract") or {}
    required_true = (
        "v3_reruns_official_resplat",
        "force_only_after_complete_postmerge_gate_rejection",
        "gate_thresholds_unchanged",
        "merge_filters_unchanged",
        "geometry_gate_unchanged",
        "sidecar_quality_gate_unchanged",
        "postmerge_quality_gate_unchanged",
        "fresh_candidate_identity_equal_to_v2_not_claimed",
    )
    for field in required_true:
        _expect(scientific.get(field), True, f"v3 scientific contract {field}")
    _expect(
        scientific.get("v2_resplat_runtime_artifacts_reused"),
        False,
        "v3 ReSplat artifact reuse",
    )
    _expect(
        scientific.get("fresh_append_required_to_reproduce_v2_accepted_count"),
        5554,
        "v3 expected accepted count",
    )
    _expect(
        scientific.get("allowed_v3_minus_v2_rollback_resolved_config_differences"),
        differences,
        "v3/v2 config difference binding",
    )
    execution = preflight.get("execution") or {}
    _expect(execution.get("selected_arms"), ["v3_forced_fused"], "v3 single execution arm")
    _expect(execution.get("v3_canonical_resolved_config_sha256"), BASE._canonical_sha256(cfg), "v3 resolved config digest")
    _expect(execution.get("online_timer_includes_commit_and_source220_chain_overhead"), True, "v3 online audit timing")
    _expect(execution.get("total_timer_includes_commit_source220_and_final100_chain_overhead"), True, "v3 total audit timing")
    _expect(execution.get("terminal_validation_publication_after_internal_total_timer"), True, "v3 terminal timing boundary")
    _expect(execution.get("external_wall_timer_includes_terminal_validation_publication"), True, "v3 external terminal timing")

    frozen = preflight.get("frozen_v2_references") or {}
    current = RUNNER._validate_frozen_v2_references(cfg)
    _expect(frozen, current, "frozen v2 dual-reference binding")
    implementation = preflight.get("implementation_provenance") or {}
    _expect(implementation.get("schema"), "unblur_slam.implementation_provenance.v1", "v3 implementation schema")
    files = implementation.get("files") or {}
    _expect(set(files), set(RUNNER.CODE_PROVENANCE_FILES), "v3 implementation file set")
    for label, record in files.items():
        _verify_file_binding(record, f"v3 implementation {label}")
    instrumentation = ((preflight.get("forced_commit_chain_contract") or {}).get("instrumentation") or {})
    _expect(instrumentation.get("accepted"), True, "v3 forced instrumentation")
    _expect(instrumentation.get("mapper_sha256"), files["mapper"]["sha256"], "v3 mapper instrumentation binding")
    _expect(instrumentation.get("slam_sha256"), files["slam_terminal_writer"]["sha256"], "v3 slam instrumentation binding")
    return {"fusion": fusion, "differences": differences, "implementation": implementation}


def _validate_forced_premerge_contract(
    audit: Mapping[str, Any],
    cfg: Mapping[str, Any],
    terminal_manifest_path: Path,
) -> dict[str, Any]:
    """Validate every gate and lineage prerequisite before forced merge."""

    official = audit.get("official_state") or {}
    _expect(official.get("requested_recurrent_updates"), 3, "v3 audit state count")
    _expect(official.get("selected_state_index_zero_based"), 2, "v3 audit state index")
    _expect(official.get("fourth_state_computed"), False, "v3 audit fourth state")
    premerge = audit.get("premerge_gates") or {}
    expected_gates = {
        "world_artifact",
        "context_reconstruction",
        "repository_provenance",
        "data_lineage",
    }
    _expect(set(premerge), expected_gates, "v3 premerge gate set")
    for name in sorted(expected_gates):
        _expect((premerge.get(name) or {}).get("accepted"), True, f"v3 premerge gate {name}")
    repository = premerge.get("repository_provenance") or {}
    _expect(repository.get("expected_commit"), RUNNER.V2.EXPECTED_RESPLAT_COMMIT, "v3 premerge expected repo")
    _expect(repository.get("observed_commit"), RUNNER.V2.EXPECTED_RESPLAT_COMMIT, "v3 premerge observed repo")
    lineage = premerge.get("data_lineage") or {}
    _expect(lineage.get("selection_membership_clear_gt_conditioned"), True, "v3 premerge conditioned membership")
    for field in (
        "ground_truth_pose_or_depth_used",
        "independent_clear_pixels_used",
        "clear_gt_metrics_used",
    ):
        _expect(lineage.get(field), False, f"v3 premerge lineage {field}")
    context = premerge.get("context_reconstruction") or {}
    _expect(context.get("uses_clear_gt"), False, "v3 context gate clear-GT use")

    published = audit.get("published_result") or {}
    if not published:
        raise ContractError("v3 forced audit lacks a published state3 result")
    manifest_path = Path(str(published.get("manifest_path", ""))).expanduser().resolve()
    _expect(manifest_path, terminal_manifest_path.resolve(), "v3 published/terminal manifest path")
    _expect(published.get("manifest_sha256"), BASE._sha256_file(manifest_path), "v3 published manifest digest")
    result_root = Path(str(published.get("path", ""))).expanduser().resolve()
    if manifest_path.parent != result_root:
        raise ContractError("v3 published result root does not own its manifest")
    if RUNNER.V2_ROOT in result_root.parents:
        raise ContractError("v3 published result unexpectedly resides under v2")

    sh_import = audit.get("sh_import") or {}
    expected_sh = {
        "native_harmonic_dimension": 16,
        "active_harmonic_dimension": 1,
        "dropped_higher_order_coefficients": 15,
        "mapper_side_truncation_performed": False,
        "dc_only": True,
    }
    for field, expected in expected_sh.items():
        _expect(sh_import.get(field), expected, f"v3 SH import {field}")
    merge = audit.get("merge") or {}
    fusion_cfg = ((cfg.get("mapping") or {}).get("official_resplat_active_fusion") or {})
    _expect(merge.get("config"), fusion_cfg.get("merge"), "v3 merge config/runtime binding")
    _expect(merge.get("optimizer_state_shapes_valid"), True, "v3 merge optimizer shapes")
    _expect(merge.get("active_map_changed"), True, "v3 merge trial change")
    return {
        "manifest_path": manifest_path,
        "result_root": result_root,
        "premerge_gate_names": sorted(expected_gates),
    }


def _validate_recorded_postmerge_decision(
    postmerge: Mapping[str, Any], fusion_config: Any
) -> dict[str, Any]:
    """Recompute the unchanged gate and require the recorded decision exactly."""

    from src.refinement.official_resplat_active_fusion import (
        postmerge_reconstruction_gate,
    )

    recorded = postmerge.get("decision") or {}
    try:
        recomputed = postmerge_reconstruction_gate(
            postmerge.get("before") or {},
            postmerge.get("after") or {},
            fusion_config,
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"cannot recompute v3 postmerge gate: {error}") from error
    _expect(recorded, recomputed, "v3 recomputed postmerge decision")
    return recomputed


def _validate_terminal_final_state_link(
    terminal: Mapping[str, Any], final100: Mapping[str, Any]
) -> None:
    _expect(
        terminal.get("final_in_memory_state"),
        final100.get("post_refinement_state"),
        "terminal/final100 full state",
    )


def _validate_terminal_code_bindings(
    code_bindings: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    expected_code_labels = {
        "mapper",
        "active_fusion_helper",
        "active_map_merge",
        "sidecar_bridge",
        "world_bridge",
        "official_sidecar_runner",
        "gaussian_model",
        "slam_terminal_writer",
    }
    _expect(set(code_bindings), expected_code_labels, "terminal code binding set")
    implementation_files = (
        (preflight.get("implementation_provenance") or {}).get("files") or {}
    )
    implementation_map = {
        "mapper": "mapper",
        "active_fusion_helper": "active_fusion_helper",
        "active_map_merge": "active_map_merge",
        "sidecar_bridge": "sidecar_verifier",
        "world_bridge": "world_bridge",
        "official_sidecar_runner": "sidecar_runner",
        "gaussian_model": "gaussian_model",
        "slam_terminal_writer": "slam_terminal_writer",
    }
    for label, binding in code_bindings.items():
        _verify_file_binding(binding, f"terminal code {label}")
        expected = implementation_files.get(implementation_map[label]) or {}
        _expect(
            binding.get("sha256"),
            expected.get("sha256"),
            f"terminal/preflight code binding {label}",
        )
        _expect(
            Path(str(binding.get("path", ""))).resolve(),
            Path(str(expected.get("path", ""))).resolve(),
            f"terminal/preflight code path {label}",
        )


def _validate_resplat_artifact_lineage(
    audit: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Cross-bind native/world artifacts and the content-addressed snapshot."""

    outputs = manifest.get("outputs") or {}
    manifest_path = artifact_paths["official_resplat_manifest"]
    native_path = artifact_paths["official_resplat_native_gaussians"]
    world_path = artifact_paths["official_resplat_world_gaussians"]
    native_relative = Path(str(outputs.get("native_gaussians_npz", "")))
    if native_relative.is_absolute() or ".." in native_relative.parts or not str(native_relative):
        raise ContractError("v3 manifest native-artifact path is invalid")
    _expect(native_path, manifest_path.parent / native_relative, "manifest/terminal native artifact path")
    native_sha = BASE._sha256_file(native_path)
    _expect(outputs.get("native_gaussians_npz_sha256"), native_sha, "manifest/current native artifact digest")

    world_relative = Path(str(outputs.get("unblur_world_gaussians_npz", "")))
    if world_relative.is_absolute() or ".." in world_relative.parts or not str(world_relative):
        raise ContractError("v3 manifest world-artifact path is invalid")
    _expect(world_path, manifest_path.parent / world_relative, "manifest/terminal world artifact path")
    world_sha = BASE._sha256_file(world_path)
    _expect(outputs.get("unblur_world_gaussians_npz_sha256"), world_sha, "manifest/current world artifact digest")
    _expect(
        (
            ((audit.get("premerge_gates") or {}).get("world_artifact") or {})
            .get("measurements", {})
            .get("world_artifact_sha256")
        ),
        world_sha,
        "premerge/current world artifact digest",
    )
    arrays = outputs.get("npz_arrays") or {}
    _expect(
        set(arrays),
        {"means", "covariances", "scales", "rotations", "harmonics", "opacities"},
        "v3 official native six-array contract",
    )

    audit_snapshot = audit.get("snapshot") or {}
    expected_snapshot_manifest = (
        Path(str(audit_snapshot.get("path", ""))).expanduser().resolve()
        / "snapshot_manifest.json"
    )
    _expect(
        artifact_paths["snapshot_manifest"],
        expected_snapshot_manifest,
        "audit/terminal snapshot-manifest path",
    )
    snapshot_manifest = _load_json(expected_snapshot_manifest, "v3 snapshot manifest")
    from src.refinement.official_resplat_sidecar import load_snapshot

    try:
        verified_snapshot = load_snapshot(expected_snapshot_manifest.parent)
    except (OSError, RuntimeError, ValueError) as error:
        raise ContractError(f"v3 snapshot failed content-address verification: {error}") from error
    manifest_snapshot_fields = {
        "snapshot_id": "snapshot_id",
        "snapshot_sha256": "snapshot_sha256",
        "pose_revision": "source_pose_revision",
    }
    for field, manifest_field in manifest_snapshot_fields.items():
        _expect(snapshot_manifest.get(field), audit_snapshot.get(field), f"snapshot manifest/audit {field}")
        _expect(manifest.get(manifest_field), audit_snapshot.get(field), f"ReSplat manifest/audit snapshot {field}")
        _expect(verified_snapshot.get(field), audit_snapshot.get(field), f"verified snapshot/audit {field}")
    return {
        "native_sha256": native_sha,
        "world_sha256": world_sha,
        "snapshot_id": str(audit_snapshot.get("snapshot_id")),
        "snapshot_sha256": str(audit_snapshot.get("snapshot_sha256")),
        "source_pose_revision": int(audit_snapshot.get("pose_revision")),
    }
def _read_forced_chain(
    scene_root: Path, cfg: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    fusion_root = scene_root / "official_resplat_active_fusion"
    audit_path = fusion_root / "fusion_audit.json"
    final_contract_path = fusion_root / "fusion_final_contract.json"
    chain_root = fusion_root / "forced_commit_chain"
    audit = _load_json(audit_path, "v3 fusion audit")
    final_contract = _load_json(final_contract_path, "v3 online fusion contract")
    commit = _verify_event(
        chain_root / "00_forced_commit.json", 0, "postgate_rejected_forced_commit"
    )
    source220 = _verify_event(
        chain_root / "01_source220_complete.json", 1, "source220_mapping_complete"
    )
    final100 = _verify_event(
        chain_root / "02_final100_complete.json", 2, "final100_complete"
    )
    terminal = _load_json(chain_root / "terminal_contract.json", "v3 terminal contract")
    from src.refinement.official_resplat_active_fusion import canonical_contract_sha256

    _expect(
        terminal.get("terminal_sha256"),
        canonical_contract_sha256(terminal, digest_field="terminal_sha256"),
        "v3 terminal canonical digest",
    )
    _expect(audit.get("schema"), "unblur_slam.official_resplat_active_fusion_audit.v1", "v3 fusion audit schema")
    _expect(audit.get("status"), "postmerge_gate_rejected_forced_commit_unsafe", "v3 forced status")
    _expect(audit.get("active_map_changed_final"), True, "v3 forced active-map change")
    trigger = audit.get("trigger") or {}
    _expect(trigger.get("after_fully_mapped_keyframe_count"), 8, "v3 trigger count")
    _expect(trigger.get("source_index"), 166, "v3 trigger source")
    _expect(
        tuple(trigger.get("source_indices", ())),
        RUNNER.EXPECTED_CONTEXT,
        "v3 trigger context",
    )
    protocol = audit.get("diagnostic_protocol") or {}
    for field in (
        "posthoc_after_v2_rejection",
        "unsafe_not_deployable",
        "gate_thresholds_unchanged",
        "force_only_after_postmerge_gate_rejected",
    ):
        _expect(protocol.get(field), True, f"v3 audit protocol {field}")
    _expect(protocol.get("expected_forced_commit_gaussian_count"), 5554, "v3 protocol accepted count")
    _expect(protocol.get("v2_rejection_audit_sha256"), RUNNER.V2_ROLLBACK_AUDIT_SHA256, "v3 audit v2 lineage")
    postmerge = audit.get("postmerge_gate") or {}
    decision = postmerge.get("decision") or {}
    from src.refinement.official_resplat_active_fusion import ActiveFusionConfig

    fusion_config = ActiveFusionConfig.from_dict(
        ((cfg.get("mapping") or {}).get("official_resplat_active_fusion") or {}),
        default_output_root=scene_root / "official_resplat_active_fusion",
    )
    recomputed_decision = _validate_recorded_postmerge_decision(
        postmerge, fusion_config
    )
    _expect(decision.get("accepted"), False, "v3 unchanged post-gate rejection")
    reasons = list(decision.get("reasons") or [])
    if not reasons:
        raise ContractError("v3 forced commit lacks explicit post-gate rejection reasons")
    merge = audit.get("merge") or {}
    before = audit.get("active_state_before") or {}
    trial = audit.get("active_state_trial") or {}
    committed = audit.get("active_state_final") or {}
    _expect(merge.get("mode"), "append", "v3 merge mode")
    _expect(merge.get("accepted_count"), 5554, "v3 fresh accepted count")
    _expect(merge.get("before_count"), before.get("gaussian_count"), "v3 before count lineage")
    _expect(merge.get("after_count"), trial.get("gaussian_count"), "v3 trial count lineage")
    _expect(int(trial["gaussian_count"]), int(before["gaussian_count"]) + 5554, "v3 append equation")
    _expect(committed.get("gaussian_count"), trial.get("gaussian_count"), "v3 committed/trial count")
    _expect(committed.get("sha256"), trial.get("sha256"), "v3 committed/trial full-state SHA")
    if committed.get("sha256") == before.get("sha256"):
        raise ContractError("v3 committed full state did not differ from premerge")
    _expect(committed.get("byte_identical_to_trial"), True, "v3 committed/trial identity")
    _expect((audit.get("timing") or {}).get("rollback_seconds"), 0.0, "v3 rollback timing")
    override = audit.get("unsafe_forced_commit") or {}
    _expect(override.get("postmerge_gate_decision_recorded"), True, "v3 override decision record")
    _expect(override.get("postmerge_gate_accepted"), False, "v3 override rejected gate")
    _expect(override.get("ordinary_action"), "rollback", "v3 ordinary action")
    _expect(override.get("diagnostic_override_action"), "commit_without_rollback", "v3 override action")
    _expect(override.get("rollback_performed"), False, "v3 override rollback")
    _expect(override.get("committed_gaussian_count"), 5554, "v3 override count")
    _expect(override.get("uses_ground_truth"), False, "v3 override GT use")
    _expect(override.get("uses_clear_gt_metrics"), False, "v3 override clear metric use")

    _expect(commit.get("previous_event_sha256"), None, "commit chain genesis")
    _expect(commit.get("source_index"), 166, "commit source")
    _expect(commit.get("fusion_audit", {}).get("sha256"), BASE._sha256_file(audit_path), "commit/audit binding")
    _expect(commit.get("accepted_gaussian_count"), 5554, "commit accepted count")
    _expect(commit.get("before_state", {}).get("gaussian_count"), before.get("gaussian_count"), "commit before count")
    _expect(commit.get("trial_state", {}).get("gaussian_count"), trial.get("gaussian_count"), "commit trial count")
    _expect(commit.get("committed_state", {}).get("gaussian_count"), committed.get("gaussian_count"), "commit final count")
    _expect(commit.get("before_state", {}).get("full_active_state_sha256"), before.get("sha256"), "commit before state")
    _expect(commit.get("trial_state", {}).get("full_active_state_sha256"), trial.get("sha256"), "commit trial state")
    _expect(commit.get("committed_state", {}).get("full_active_state_sha256"), committed.get("sha256"), "commit final state")
    _expect(commit.get("trial_equals_committed"), True, "commit trial/final equality")
    _expect(commit.get("committed_differs_from_before"), True, "commit differs from before")
    _expect(commit.get("postmerge_gate_accepted"), False, "commit post-gate decision")
    _expect(commit.get("postmerge_gate_reasons"), reasons, "commit/audit rejection reasons")
    _expect(commit.get("rollback_performed"), False, "commit rollback flag")
    _expect(commit.get("rollback_seconds"), 0.0, "commit rollback seconds")
    _expect(commit.get("ground_truth_pose_depth_or_clear_pixels_used_for_decision"), False, "commit GT lineage")
    _expect(commit.get("individual_imported_gaussian_survival_not_claimed"), True, "commit survival caveat")

    _expect(source220.get("previous_event_sha256"), commit["event_sha256"], "source220 previous link")
    _expect(source220.get("source_index"), 220, "source220 source")
    _expect(source220.get("frame_id"), 10, "source220 frame id")
    _expect(source220.get("entry_state"), commit.get("committed_state"), "source220 entry/commit state")
    _expect(source220.get("entry_captured_before_any_source220_map_mutation"), True, "source220 entry ordering")
    _expect(source220.get("mapping_iterations_completed"), 100, "source220 mapping iterations")
    _expect(source220.get("prune_passes_completed"), 1, "source220 prune pass")
    _expect(source220.get("iteration_count_delta"), 101, "source220 iteration delta")
    _expect(
        int(source220.get("iteration_count_after", -1))
        - int(source220.get("iteration_count_before", -1)),
        101,
        "source220 counter arithmetic",
    )
    _expect(source220.get("individual_imported_gaussian_survival_after_mapping_not_tracked"), True, "source220 survival caveat")
    _expect(source220.get("committed_import_batch_was_live_at_source220_entry"), True, "source220 batch live")
    _expect(source220.get("committed_import_batch_participated_in_downstream_map_computation"), True, "source220 batch participation")

    _expect(final100.get("previous_event_sha256"), source220["event_sha256"], "final100 previous link")
    _expect(final100.get("entry_state"), source220.get("post_mapping_state"), "final100/source220 state link")
    _expect(final100.get("configured_iterations"), 100, "final100 configured count")
    _expect(final100.get("executed_iterations"), 100, "final100 executed count")
    _expect(final100.get("iteration_count_delta"), 100, "final100 iteration delta")
    _expect(
        int(final100.get("iteration_count_after", -1))
        - int(final100.get("iteration_count_before", -1)),
        100,
        "final100 counter arithmetic",
    )
    _expect(final100.get("individual_imported_gaussian_survival_after_final100_not_tracked"), True, "final100 survival caveat")
    _expect(final100.get("committed_import_batch_participated_in_final100_computation"), True, "final100 batch participation")
    checkpoint_path = _verify_file_binding(final100.get("checkpoint_ply") or {}, "final100 checkpoint PLY")
    _expect(
        (final100.get("checkpoint_ply") or {}).get("vertex_count"),
        (final100.get("post_refinement_state") or {}).get("gaussian_count"),
        "final100 checkpoint count",
    )

    _expect(terminal.get("schema"), "unblur_slam.forced_commit_terminal_contract.v1", "v3 terminal schema")
    _expect(terminal.get("status"), "complete_unsafe_posthoc_diagnostic", "v3 terminal status")
    _expect(terminal.get("previous_event_sha256"), final100["event_sha256"], "terminal previous link")
    chain = terminal.get("chain") or {}
    _expect(chain.get("commit_event_sha256"), commit["event_sha256"], "terminal commit link")
    _expect(chain.get("source220_event_sha256"), source220["event_sha256"], "terminal source220 link")
    _expect(chain.get("final100_event_sha256"), final100["event_sha256"], "terminal final100 link")
    _expect(chain.get("tip_sha256"), final100["event_sha256"], "terminal chain tip")
    for field in (
        "posthoc_after_v2_rejection",
        "unsafe_not_deployable",
        "gate_thresholds_unchanged",
        "merge_filters_unchanged",
        "v3_reran_official_resplat",
        "source220_mapping_completed",
        "final100_completed",
        "final_serialization_completed",
        "individual_imported_gaussian_survival_in_final_model_not_tracked",
        "individual_imported_gaussian_survival_in_final_model_not_claimed",
    ):
        _expect(terminal.get(field), True, f"terminal {field}")
    _expect(terminal.get("v2_resplat_runtime_artifacts_reused"), False, "terminal v2 artifact reuse")
    _expect(terminal.get("committed_import_batch_proven_live_at_source220_entry"), True, "terminal batch live")
    _expect(terminal.get("committed_import_batch_participated_in_source220_and_final100"), True, "terminal batch participation")
    _expect(terminal.get("uses_gt_for_forced_commit_decision"), False, "terminal decision GT use")
    _expect(terminal.get("clear_gt_metrics_bound_posthoc_for_evaluation"), True, "terminal posthoc metric binding")
    _expect(terminal.get("clear_gt_values_used_for_commit_or_checkpoint_selection"), False, "terminal metric selection leakage")
    final_ply_path = _verify_file_binding(terminal.get("final_model_ply") or {}, "terminal final model")
    _expect((terminal.get("final_model_ply") or {}).get("byte_identical_to_iter100_checkpoint"), True, "terminal/checkpoint equality")
    _expect(BASE._sha256_file(final_ply_path), BASE._sha256_file(checkpoint_path), "checkpoint/final-model byte equality")
    _expect((terminal.get("final_model_ply") or {}).get("vertex_count"), (terminal.get("final_in_memory_state") or {}).get("gaussian_count"), "terminal live/PLY count")
    _validate_terminal_final_state_link(terminal, final100)

    artifact_paths = {}
    for label, binding in (terminal.get("artifact_bindings") or {}).items():
        artifact_paths[label] = _verify_file_binding(binding, f"terminal artifact {label}")
    required_artifacts = {
        "resolved_config",
        "fusion_audit",
        "fusion_final_contract",
        "snapshot_manifest",
        "official_resplat_manifest",
        "official_resplat_native_gaussians",
        "official_resplat_world_gaussians",
        "final_metrics",
        "runtime_stats",
        "final_model_ply",
    }
    _expect(set(artifact_paths), required_artifacts, "terminal artifact binding set")
    _expect(artifact_paths["resolved_config"], (scene_root / "cfg.yaml").resolve(), "terminal resolved config path")
    _expect(artifact_paths["fusion_audit"], audit_path.resolve(), "terminal audit path")
    _expect(artifact_paths["fusion_final_contract"], final_contract_path.resolve(), "terminal online contract path")
    _expect(
        artifact_paths["final_metrics"],
        (scene_root / "psnr/after_refine/final_result.json").resolve(),
        "terminal final metrics path",
    )
    _expect(
        artifact_paths["runtime_stats"],
        (scene_root / "runtime_stats.json").resolve(),
        "terminal runtime-stats path",
    )
    _expect(artifact_paths["final_model_ply"], final_ply_path, "terminal final model binding")
    for name in (
        "official_resplat_manifest",
        "official_resplat_native_gaussians",
        "official_resplat_world_gaussians",
    ):
        path = artifact_paths[name]
        if RUNNER.V2_ROOT in path.parents or fusion_root not in path.parents:
            raise ContractError(f"v3 runtime artifact {name} was not freshly rooted in v3")
    code_bindings = terminal.get("code_bindings") or {}
    _validate_terminal_code_bindings(code_bindings, preflight)

    _validate_forced_premerge_contract(
        audit, cfg, artifact_paths["official_resplat_manifest"]
    )

    manifest = _load_json(artifact_paths["official_resplat_manifest"], "fresh v3 ReSplat manifest")
    repository = ((manifest.get("official_resplat") or {}).get("repository") or {})
    _expect(repository.get("commit"), RUNNER.V2.EXPECTED_RESPLAT_COMMIT, "v3 ReSplat commit")
    execution = manifest.get("execution_contract") or {}
    _expect(execution.get("requested_recurrent_updates"), 3, "v3 ReSplat updates")
    _expect(execution.get("returned_recurrent_states"), 3, "v3 ReSplat returned states")
    _expect(execution.get("selected_state_index_zero_based"), 2, "v3 ReSplat state index")
    _expect(execution.get("fourth_state_computed"), False, "v3 ReSplat fourth state")
    _validate_resplat_artifact_lineage(audit, manifest, artifact_paths)

    _expect(final_contract.get("fusion_status"), audit.get("status"), "online contract/audit status")
    _expect(
        tuple(final_contract.get("actually_mapped_source_indices", ())),
        RUNNER.EXPECTED_CONTEXT + RUNNER.EXPECTED_DOWNSTREAM,
        "online actually-mapped sequence",
    )
    _expect(
        tuple(final_contract.get("trigger_context_source_indices", ())),
        RUNNER.EXPECTED_CONTEXT,
        "online trigger context",
    )
    _expect(final_contract.get("trigger_source_index"), 166, "online trigger source")
    _expect(
        tuple(final_contract.get("downstream_online_mapped_source_indices_after_fusion", ())),
        RUNNER.EXPECTED_DOWNSTREAM,
        "online downstream sources",
    )
    _expect(final_contract.get("fusion_completed_before_downstream_online_mapping"), True, "online fusion ordering")
    _expect(final_contract.get("fusion_committed_to_active_map_before_downstream_mapping"), True, "online forced commit")
    _expect(final_contract.get("source220_downstream_mapping_completed"), True, "online source220 completion")
    _expect(final_contract.get("source220_chain_event_sha256"), source220["event_sha256"], "online/source220 chain link")
    _expect(final_contract.get("final_refinement_completion_pending_at_contract_write"), True, "online contract timing boundary")
    timing_disclosure = terminal.get("timing_disclosure") or {}
    _expect(timing_disclosure.get("online_time_includes_commit_and_source220_chain_overhead"), True, "terminal online timing disclosure")
    _expect(timing_disclosure.get("total_time_includes_commit_source220_and_final100_chain_overhead"), True, "terminal total timing disclosure")
    _expect(timing_disclosure.get("diagnostic_overhead_not_subtracted_from_reported_times"), True, "terminal timing subtraction disclosure")
    stage_seconds = timing_disclosure.get("recorded_stage_seconds") or {}
    _expect(
        set(stage_seconds),
        {
            "commit_validation_and_event_publication",
            "source220_entry_capture",
            "source220_complete_capture_and_event_publication",
            "final100_entry_capture",
            "final100_complete_capture_and_event_publication",
        },
        "terminal recorded internal timing stages",
    )
    for label, value in stage_seconds.items():
        if BASE._finite_float(value, f"terminal timing {label}") < 0.0:
            raise ContractError(f"terminal timing {label} is negative")
    return {
        "status": audit["status"],
        "unsafe_forced_commit": True,
        "fresh_accepted_count": 5554,
        "fresh_candidate_identity_equal_to_v2_not_claimed": True,
        "before_gaussian_count": int(before["gaussian_count"]),
        "committed_gaussian_count": int(committed["gaussian_count"]),
        "postmerge_gate_accepted": False,
        "postmerge_gate_reasons": reasons,
        "audit_path": str(audit_path),
        "audit_sha256": BASE._sha256_file(audit_path),
        "chain_event_sha256": {
            "commit": commit["event_sha256"],
            "source220": source220["event_sha256"],
            "final100": final100["event_sha256"],
            "terminal": terminal["terminal_sha256"],
        },
        "source220_and_final100_participation_proven": True,
        "individual_imported_gaussian_survival_not_tracked_or_claimed": True,
        "official_resplat_manifest": str(artifact_paths["official_resplat_manifest"]),
        "official_resplat_world_gaussians": str(artifact_paths["official_resplat_world_gaussians"]),
        "final_model": str(final_ply_path),
        "final_model_sha256": BASE._sha256_file(final_ply_path),
        "timing": audit.get("timing") or {},
        "terminal_timing_disclosure": terminal.get("timing_disclosure") or {},
    }


def _read_v3_arm(root: Path) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    arm_root = root / "evssm_resplat3_forced_commit"
    scene_root = arm_root / SCENE
    cfg = BASE._load_yaml(scene_root / "cfg.yaml", "v3 resolved config")
    preflight = _load_json(arm_root / "preflight.json", "v3 launch preflight")
    _validate_v3_config_and_preflight(cfg, preflight, arm_root)
    launcher = _load_json(arm_root / "launcher_runtime.json", "v3 launcher runtime")
    _expect(launcher.get("schema"), "unblur_slam.external_wall_runtime.v1", "v3 launcher schema")
    _expect(launcher.get("experiment_revision"), "v3_forced_commit", "v3 launcher revision")
    _expect(launcher.get("arm"), "v3_forced_fused", "v3 launcher arm")
    _expect(launcher.get("exit_code"), 0, "v3 launcher exit code")
    _expect(launcher.get("posthoc_after_v2_rejection"), True, "v3 launcher posthoc disclosure")
    _expect(launcher.get("unsafe_not_deployable"), True, "v3 launcher safety disclosure")
    runtime = BASE._read_runtime(arm_root, scene_root, "v3_forced_fused")
    metrics = BASE._read_metrics(arm_root, scene_root, "v3_forced_fused")
    keyframes = BASE._read_keyframes(scene_root, metrics, "v3_forced_fused")
    _expect(tuple(keyframes["source_indices"]), tuple(RUNNER.EXPECTED_FIXED_SOURCE_KEYFRAMES), "v3 exact fixed keyframes")
    frontend = BASE._read_frontend(arm_root, keyframes["source_indices"], "baseline")
    trajectory = BASE._read_trajectory(scene_root, keyframes["count"], "v3_forced_fused")
    gpu = V2_REPORT._read_gpu_monitor(arm_root, "fused")
    fusion = _read_forced_chain(scene_root, cfg, preflight)
    runtime_stats = _load_json(scene_root / "runtime_stats.json", "v3 runtime stats")
    _expect(runtime_stats.get("official_resplat_active_fusion_enabled"), True, "v3 runtime fusion switch")
    _expect(runtime_stats.get("official_resplat_active_fusion_status"), fusion["status"], "v3 runtime fusion status")
    audit_wall = BASE._finite_float(
        (fusion.get("timing") or {}).get("total_wall_seconds"),
        "v3 audit fusion wall",
    )
    runtime_wall = BASE._finite_float(
        runtime_stats.get("official_resplat_active_fusion_total_wall_seconds"),
        "v3 runtime fusion wall",
    )
    if not math.isclose(audit_wall, runtime_wall, rel_tol=0.0, abs_tol=1e-9):
        raise ContractError("v3 runtime/audit fusion wall timing mismatch")
    timing_fields = {
        "forced_commit_chain_commit_event_seconds": "commit_validation_and_event_publication",
        "forced_commit_chain_source220_entry_seconds": "source220_entry_capture",
        "forced_commit_chain_source220_complete_seconds": "source220_complete_capture_and_event_publication",
        "forced_commit_chain_final100_entry_seconds": "final100_entry_capture",
        "forced_commit_chain_final100_complete_seconds": "final100_complete_capture_and_event_publication",
    }
    terminal_stages = (
        (fusion.get("terminal_timing_disclosure") or {}).get(
            "recorded_stage_seconds"
        )
        or {}
    )
    for field, terminal_field in timing_fields.items():
        value = BASE._finite_float(runtime_stats.get(field), f"v3 diagnostic overhead {field}")
        if value < 0.0:
            raise ContractError(f"v3 diagnostic overhead {field} is negative")
        terminal_value = BASE._finite_float(
            terminal_stages.get(terminal_field),
            f"v3 terminal diagnostic overhead {terminal_field}",
        )
        if not math.isclose(value, terminal_value, rel_tol=0.0, abs_tol=1e-12):
            raise ContractError(
                f"v3 runtime/terminal diagnostic overhead mismatch: {field}"
            )
    if float((fusion.get("timing") or {}).get("total_wall_seconds", 0.0)) > runtime["official_timer_online_seconds"]:
        raise ContractError("v3 ReSplat/merge wall time is outside online timer")
    return (
        {
            "output_root": str(arm_root),
            "frontend": "official_unblur_slam_evssm",
            "runtime": runtime,
            "frontend_activity": frontend,
            "rendering_and_depth": metrics,
            "keyframes": keyframes,
            "trajectory": trajectory,
            "whole_device_gpu": gpu,
            "forced_active_resplat_fusion": fusion,
        },
        cfg,
        preflight,
    )


def _delta(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, float]:
    nr, orr = new["runtime"], old["runtime"]
    nm, om = new["rendering_and_depth"], old["rendering_and_depth"]
    nt, ot = new["trajectory"], old["trajectory"]
    return {
        "online_seconds": nr["official_timer_online_seconds"] - orr["official_timer_online_seconds"],
        "online_time_ratio": nr["official_timer_online_seconds"] / orr["official_timer_online_seconds"],
        "online_speed_ratio": orr["official_timer_online_seconds"] / nr["official_timer_online_seconds"],
        "derived_prefix_online_fps": nr["derived_prefix_online_fps"] - orr["derived_prefix_online_fps"],
        "psnr_db": nm["psnr_db"] - om["psnr_db"],
        "ssim": nm["ssim"] - om["ssim"],
        "lpips": nm["lpips"] - om["lpips"],
        "depth_l1": nm["depth_l1"] - om["depth_l1"],
        "full_trajectory_ate_rmse_m": nt["full_trajectory_ate_rmse_m"] - ot["full_trajectory_ate_rmse_m"],
        "keyframe_trajectory_ate_rmse_m": nt["keyframe_trajectory_ate_rmse_m"] - ot["keyframe_trajectory_ate_rmse_m"],
    }


def build_report(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = root.expanduser().resolve()
    baseline, _, _ = V2_REPORT._read_arm(RUNNER.V2_ROOT, "baseline")
    rollback, _, _ = V2_REPORT._read_arm(RUNNER.V2_ROOT, "fused")
    v2_published = _load_json(RUNNER.V2_EXPERIMENT_AUDIT, "frozen v2 experiment audit")
    _expect(v2_published.get("arms", {}).get("baseline"), baseline, "v2 baseline numeric path")
    _expect(v2_published.get("arms", {}).get("fused"), rollback, "v2 rollback numeric path")
    forced, _, preflight = _read_v3_arm(root)
    return {
        "schema": "unblur_slam.fr2_xyz_resplat3_forced_commit_221_report.v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(root),
        "scope": {
            "posthoc_after_v2_rejection": True,
            "unsafe_not_deployable": True,
            "single_fresh_v3_forced_arm": True,
            "v2_baseline_and_rollback_are_cross_revision_frozen_references": True,
            "fresh_paired_comparison": False,
            "paper_table_or_26k_result": False,
            "clear_gt_prefix_metric_only": True,
        },
        "reference_integrity": {
            "v2_dual_reference": preflight["frozen_v2_references"],
            "v2_baseline_numeric_path_unchanged_and_read_only": True,
            "v2_rollback_numeric_path_unchanged_and_read_only": True,
            "v2_published_report_sha256": RUNNER.V2_EXPERIMENT_AUDIT_SHA256,
            "v2_forced_arm_rerun_official_resplat_independently": True,
            "v2_resplat_runtime_artifacts_reused": False,
        },
        "arms": {
            "v2_baseline_reference": baseline,
            "v2_safe_rollback_reference": rollback,
            "v3_forced_commit_diagnostic": forced,
        },
        "diagnostic_comparisons_not_fair_paired_estimates": {
            "v3_forced_minus_v2_baseline_reference": _delta(forced, baseline),
            "v3_forced_minus_v2_safe_rollback_reference": _delta(forced, rollback),
        },
        "interpretation_notes": [
            "The unchanged postmerge gate rejected the fresh append; v3 kept it only through an explicitly unsafe post-hoc override.",
            "The fresh run must reproduce accepted_count=5554, but candidate identity equality with v2 is neither tracked nor claimed.",
            "The hash chain proves the committed batch was live at source220 entry and participated in source220 mapping and final100; it does not tag or prove survival of individual imported Gaussians in final_model.ply.",
            "Baseline and safe rollback values are SHA-frozen v2 references, not reruns under the v3 code revision, so speed/quality deltas are diagnostic and not a fair paired estimate.",
            "Internal online time includes commit/source220 chain audit overhead; internal total time also includes final100 chain overhead. Terminal validation/publication occurs later and is included only in external launcher wall time. No recorded audit overhead is subtracted.",
            "Clear-GT values are bound only for post-hoc evaluation and were not used for the forced-commit decision or checkpoint selection.",
        ],
    }


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, arm in report["arms"].items():
        runtime = arm["runtime"]
        metrics = arm["rendering_and_depth"]
        trajectory = arm["trajectory"]
        rows.append(
            {
                "arm": label,
                "unsafe_not_deployable": label == "v3_forced_commit_diagnostic",
                "cross_revision_reference": label != "v3_forced_commit_diagnostic",
                "online_seconds": runtime["official_timer_online_seconds"],
                "total_seconds": runtime["official_timer_total_seconds"],
                "external_wall_seconds": runtime["external_launcher_wall_seconds"],
                "derived_prefix_online_fps": runtime["derived_prefix_online_fps"],
                "psnr_db": metrics["psnr_db"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "depth_l1": metrics["depth_l1"],
                "full_trajectory_ate_rmse_m": trajectory["full_trajectory_ate_rmse_m"],
                "keyframe_trajectory_ate_rmse_m": trajectory["keyframe_trajectory_ate_rmse_m"],
                "fair_paired_comparison": False,
                "individual_imported_gaussian_survival_proven": False,
            }
        )
    return rows


def write_report(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "forced_commit_221_v3_audit.json"
    csv_path = output_dir / "forced_commit_221_v3_metrics.csv"
    for path in (json_path, csv_path):
        if path.exists() or path.is_symlink():
            raise ContractError(f"refusing to overwrite v3 report: {path}")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = _csv_rows(report)
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args.root)
        if args.output_dir is None:
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        else:
            json_path, csv_path = write_report(report, args.output_dir)
            print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))
        return 0
    except (ContractError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
