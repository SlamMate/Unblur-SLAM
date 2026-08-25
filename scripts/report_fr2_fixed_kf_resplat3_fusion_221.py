#!/usr/bin/env python3
"""Fail-closed v2 report for fixed-11KF EVSSM vs active ReSplat-state3 fusion.

This auditor performs no rendering and creates no CUDA context.  It verifies
the completed paired-run records, exact implementation hashes, continuous
whole-device GPU sampling, the one-shot fusion transaction, and either an
accepted active-map append or an explicitly byte-identical rollback.  Metrics
remain an 11-frame clear-GT prefix diagnostic, not a paper Table-6/26K result.
The report defaults exclusively to the new ``_v2`` root and does not reinterpret
or overwrite the fail-closed v1 run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import subprocess
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


BASE = _load_module(
    "fusion221_report_base",
    REPO_ROOT / "scripts/report_fr2_official_online_budget_paired_221.py",
)
RUNNER = _load_module(
    "fusion221_runner_contract",
    REPO_ROOT / "scripts/run_fr2_fixed_kf_resplat3_fusion_221.py",
)
ContractError = BASE.ContractError

DEFAULT_ROOT = RUNNER.OUTPUT_ROOT
SCENE = "freiburg2_xyz"
SOURCE_COUNT = 221
EXPECTED_FIXED = tuple(RUNNER.EXPECTED_FIXED_SOURCE_KEYFRAMES)
EXPECTED_CONTEXT = tuple(RUNNER.EXPECTED_FIRST_EIGHT_MAPPED_SOURCES)
EXPECTED_DOWNSTREAM = tuple(RUNNER.EXPECTED_DOWNSTREAM_MAPPED_SOURCES)
ARMS = {"baseline": "evssm_no_fusion", "fused": "evssm_resplat3_active_fusion"}
TERMINAL_FUSION_STATUSES = {
    "accepted",
    "sidecar_rejected",
    "premerge_gate_rejected",
    "merge_gate_rejected",
    "postmerge_gate_rejected_rolled_back",
}


def _expect(value: Any, expected: Any, label: str) -> None:
    BASE._expect(value, expected, label)


def _nested(value: Mapping[str, Any], path: str) -> Any:
    return BASE._nested(value, path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return BASE._load_json(path, label)


def _validate_config(cfg: Mapping[str, Any], arm: str, arm_root: Path) -> None:
    expected = {
        "dataset": "tumrgbd",
        "scene": SCENE,
        "stride": 1,
        "max_frames": SOURCE_COUNT,
        "setup_seed": 43,
        "device": "cuda:0",
        "warmup_mapper": True,
        "clear_init": False,
        "cam.W_out": 512,
        "cam.H_out": 384,
        "mapping.Training.init_itr_num": 1050,
        "mapping.Training.mapping_itr_num": 100,
        "mapping.Training.tracking_itr_num": 100,
        "mapping.final_refine_iters": 100,
        "mapping.online_plotting": False,
        "mapping.eval_before_final_ba": False,
        "mapping.hydrate_missing_droid_keyframes": True,
        "mapping.resplat.enabled": False,
        "mapping.resplat.online_enabled": False,
        "mapping.resplat.extra_iters": 0,
        "submaps.enabled": False,
        "submaps.official_resplat_sidecar.enabled": False,
        "framecrafter.enabled": False,
        "deblur.frontend": "evssm",
        "deblur.causal_checkpoint": "",
        "evssm_checkpoint_sha256": BASE.EXPECTED_EVSSM_SHA256,
        "evaluation.clear_gt_scope": "prefix_smoke",
    }
    for path, wanted in expected.items():
        _expect(_nested(cfg, path), wanted, f"{arm} config {path}")
    _expect(
        tuple(int(value) for value in _nested(cfg, "evaluation.expected_clear_gt_source_indices")),
        EXPECTED_FIXED,
        f"{arm} clear-GT source indices",
    )
    _expect(
        Path(str(_nested(cfg, "data.output"))).expanduser().resolve(),
        arm_root.resolve(),
        f"{arm} output root",
    )
    fixed = _nested(cfg, "tracking.fixed_source_keyframes")
    _expect(fixed.get("enabled"), True, f"{arm} fixed schedule enabled")
    _expect(fixed.get("strict_exact"), True, f"{arm} fixed schedule strict")
    _expect(
        tuple(int(value) for value in fixed.get("source_indices", ())),
        EXPECTED_FIXED,
        f"{arm} fixed schedule",
    )
    disclosure = _nested(cfg, "fixed_kf_resplat3_active_fusion_221")
    _expect(
        disclosure.get("schema"),
        "unblur_slam.fr2_xyz_fixed_kf_resplat3_active_fusion_221.v2",
        f"{arm} v2 disclosure schema",
    )
    exact_disclosure = {
        "experiment_revision": "v2",
        "supersedes_v1_mapper_hook_binding_bug": True,
        "mapped_viewpoint_binding_contract": "captured_before_pose_window_loop",
        "v1_output_preserved_and_not_reused": True,
        "conditional_on_frozen_evssm_baseline_schedule": True,
        "selection_membership_clear_gt_conditioned": True,
        "uses_ground_truth_poses_or_depths_for_fusion": False,
        "uses_independent_clear_pixels_for_fusion": False,
        "uses_clear_gt_metrics_for_fusion_gate": False,
        "trigger_after_mapped_keyframe_count": 8,
        "expected_first_eight_mapped_source_indices": list(EXPECTED_CONTEXT),
        "trigger_source_index": 166,
        "downstream_online_mapped_source_indices_after_fusion": list(EXPECTED_DOWNSTREAM),
        "maximum_fusion_attempts": 1,
        "synchronous_blocking_timing": True,
        "subprocess_gate_and_merge_inside_online_timer": True,
        "official_recurrent_updates": 3,
        "official_selected_state_index_zero_based": 2,
        "fourth_recurrent_state_computed": False,
    }
    for field, wanted in exact_disclosure.items():
        _expect(disclosure.get(field), wanted, f"{arm} disclosure {field}")
    fusion = _nested(cfg, "mapping.official_resplat_active_fusion")
    _expect(fusion.get("enabled"), arm == "fused", f"{arm} fusion switch")
    _expect(fusion.get("trigger_source_index"), 166, f"{arm} trigger source")
    _expect(
        tuple(int(value) for value in fusion.get("expected_mapped_source_indices", ())),
        EXPECTED_CONTEXT,
        f"{arm} expected mapped sources",
    )
    _expect(fusion.get("refinement_updates"), 3, f"{arm} recurrent updates")
    _expect(fusion.get("fourth_state_computed"), False, f"{arm} fourth state")
    _expect(
        fusion.get("selection_membership_clear_gt_conditioned"),
        True,
        f"{arm} conditioned membership disclosure",
    )


def _validate_preflight(
    preflight: Mapping[str, Any], cfg: Mapping[str, Any], arm: str
) -> None:
    _expect(
        preflight.get("schema"),
        "unblur_slam.fr2_xyz_fixed_kf_resplat3_active_fusion_221_preflight.v2",
        f"{arm} preflight schema",
    )
    scope = preflight.get("scope") or {}
    _expect(scope.get("experiment_revision"), "v2", f"{arm} experiment revision")
    _expect(
        Path(str(scope.get("superseded_v1_output_root", ""))).expanduser().resolve(),
        RUNNER.SUPERSEDED_V1_OUTPUT_ROOT,
        f"{arm} superseded v1 root",
    )
    _expect(
        scope.get("v1_outputs_preserved_and_not_reused"),
        True,
        f"{arm} v1 preservation",
    )
    _expect(scope.get("source_count"), SOURCE_COUNT, f"{arm} source count")
    _expect(
        tuple(scope.get("fixed_source_keyframes", ())),
        EXPECTED_FIXED,
        f"{arm} preflight fixed schedule",
    )
    fusion = preflight.get("fusion_contract") or {}
    binding = fusion.get("mapped_viewpoint_binding") or {}
    _expect(
        binding.get("schema"),
        "unblur_slam.mapper_mapped_viewpoint_hook_binding.v2",
        f"{arm} mapper hook binding schema",
    )
    _expect(binding.get("accepted"), True, f"{arm} mapper hook binding")
    _expect(binding.get("hook_argument"), "mapped_viewpoint", f"{arm} hook argument")
    _expect(
        binding.get("runtime_identity_and_source_guards"),
        True,
        f"{arm} mapped-viewpoint runtime guards",
    )
    mapper_implementation = (
        ((preflight.get("implementation_provenance") or {}).get("files") or {}).get(
            "mapper"
        )
        or {}
    )
    _expect(
        binding.get("mapper_sha256"),
        mapper_implementation.get("sha256"),
        f"{arm} mapper hook/code provenance binding",
    )
    _expect(
        tuple(fusion.get("expected_first_eight_actually_mapped_source_indices", ())),
        EXPECTED_CONTEXT,
        f"{arm} preflight mapped context",
    )
    _expect(fusion.get("trigger_source_index"), 166, f"{arm} preflight trigger")
    _expect(
        tuple(fusion.get("downstream_online_mapped_source_indices_after_fusion", ())),
        EXPECTED_DOWNSTREAM,
        f"{arm} preflight downstream mapping",
    )
    _expect(
        fusion.get("selection_membership_clear_gt_conditioned"),
        True,
        f"{arm} preflight conditioned membership",
    )
    for field in (
        "ground_truth_poses_or_depths_consumed",
        "independent_clear_pixels_consumed",
        "clear_gt_metrics_consumed_by_gate",
    ):
        _expect(fusion.get(field), False, f"{arm} preflight {field}")
    paired = preflight.get("paired_contract") or {}
    _expect(
        paired.get(f"{arm}_resolved_sha256"),
        BASE._canonical_sha256(cfg),
        f"{arm} resolved config digest",
    )
    bridge = (preflight.get("artifacts") or {}).get(
        "real_71680_gaussian_cpu_bridge_probe"
    ) or {}
    _expect(bridge.get("accepted"), True, f"{arm} real bridge probe")
    _expect(bridge.get("gpu_used"), False, f"{arm} bridge probe GPU use")
    _expect(
        bridge.get("experiment_uses_this_artifact_as_runtime_input"),
        False,
        f"{arm} historical probe runtime exclusion",
    )
    _expect(
        bridge.get("experiment_frontend_both_arms"),
        "official_unblur_slam_evssm",
        f"{arm} experiment frontend disclosure",
    )
    _expect(bridge.get("gaussian_count"), 71_680, f"{arm} bridge topology")
    _expect(
        tuple(bridge.get("world_harmonics_shape", ())),
        (71_680, 3, 1),
        f"{arm} bridge DC shape",
    )
    if int(bridge.get("capped_candidate_count_before_active_map_collision", 0)) < 1_024:
        raise ContractError(f"{arm} real bridge probe did not clear static merge minimum")


def _read_gpu_monitor(arm_root: Path, arm: str) -> dict[str, Any]:
    summary = _load_json(
        arm_root / "gpu_monitor_summary.json", f"{arm} whole-device GPU summary"
    )
    _expect(
        summary.get("schema"),
        "unblur_slam.whole_device_gpu_monitor.v1",
        f"{arm} GPU summary schema",
    )
    _expect(summary.get("physical_gpu"), 1, f"{arm} monitored physical GPU")
    _expect(summary.get("requested_interval_ms"), 250, f"{arm} GPU interval")
    _expect(
        summary.get("continuous_subprocess_sampling"), True, f"{arm} continuous GPU sampling"
    )
    _expect(summary.get("is_exact_instantaneous_peak"), False, f"{arm} GPU peak caveat")
    csv_path = arm_root / "gpu_monitor.csv"
    BASE._require_file(csv_path, f"{arm} raw GPU samples")
    samples = []
    for line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            index = int(fields[1])
            memory = int(fields[3])
            utilization = int(fields[4])
        except ValueError:
            continue
        if index == 1:
            samples.append((fields[0], fields[2], memory, utilization))
    if not samples:
        raise ContractError(f"{arm} GPU monitor contains no valid physical-GPU-1 samples")
    _expect(summary.get("sample_count"), len(samples), f"{arm} GPU sample count")
    _expect(
        summary.get("maximum_memory_used_mib"),
        max(item[2] for item in samples),
        f"{arm} GPU memory maximum",
    )
    _expect(
        summary.get("maximum_utilization_percent"),
        max(item[3] for item in samples),
        f"{arm} GPU utilization maximum",
    )
    return {
        "sampling_interval_ms": 250,
        "sample_count": len(samples),
        "whole_device_peak_lower_bound_mib": int(summary["maximum_memory_used_mib"]),
        "maximum_utilization_percent": int(summary["maximum_utilization_percent"]),
        "is_exact_instantaneous_peak": False,
        "raw_csv": str(csv_path),
    }


def _read_fusion(scene_root: Path, arm: str) -> dict[str, Any]:
    fusion_root = scene_root / "official_resplat_active_fusion"
    audit_path = fusion_root / "fusion_audit.json"
    final_path = fusion_root / "fusion_final_contract.json"
    if arm == "baseline":
        if audit_path.exists() or final_path.exists():
            raise ContractError("baseline unexpectedly contains active-fusion records")
        return {
            "enabled": False,
            "attempt_count": 0,
            "fusion_committed_to_active_map_before_downstream_mapping": False,
            "active_map_changed_at_commit": False,
            "imported_gaussian_survival_in_serialized_final_model_not_separately_tracked": True,
        }

    audit = _load_json(audit_path, "fused transaction audit")
    final = _load_json(final_path, "fused final online-mapping contract")
    _expect(
        audit.get("schema"),
        "unblur_slam.official_resplat_active_fusion_audit.v1",
        "fusion audit schema",
    )
    status = str(audit.get("status", ""))
    if status not in TERMINAL_FUSION_STATUSES:
        raise ContractError(f"fusion did not reach a valid fail-closed terminal status: {status}")
    trigger = audit.get("trigger") or {}
    _expect(trigger.get("after_fully_mapped_keyframe_count"), 8, "fusion trigger count")
    _expect(trigger.get("source_index"), 166, "fusion trigger source")
    _expect(
        tuple(int(value) for value in trigger.get("source_indices", ())),
        EXPECTED_CONTEXT,
        "fusion trigger context",
    )
    official = audit.get("official_state") or {}
    _expect(official.get("requested_recurrent_updates"), 3, "fusion state count")
    _expect(official.get("selected_state_index_zero_based"), 2, "fusion selected state")
    _expect(official.get("fourth_state_computed"), False, "fusion fourth state")
    lineage = audit.get("data_lineage") or {}
    _expect(
        lineage.get("selection_membership_clear_gt_conditioned"),
        True,
        "fusion conditioned membership",
    )
    for field in (
        "ground_truth_poses_or_depths_consumed_by_fusion",
        "independent_clear_pixels_consumed_by_fusion",
        "clear_gt_metrics_consumed_by_fusion",
    ):
        _expect(lineage.get(field), False, f"fusion lineage {field}")
    _expect(
        final.get("schema"),
        "unblur_slam.official_resplat_active_fusion_final_contract.v1",
        "fusion final contract schema",
    )
    _expect(final.get("fusion_attempt_count"), 1, "fusion final attempt count")
    _expect(
        tuple(final.get("actually_mapped_source_indices", ())),
        EXPECTED_CONTEXT + EXPECTED_DOWNSTREAM,
        "actually mapped online sequence",
    )
    _expect(
        tuple(final.get("trigger_context_source_indices", ())),
        EXPECTED_CONTEXT,
        "final trigger contexts",
    )
    _expect(
        tuple(final.get("downstream_online_mapped_source_indices_after_fusion", ())),
        EXPECTED_DOWNSTREAM,
        "downstream online mapping after fusion",
    )
    _expect(final.get("fusion_completed_before_downstream_online_mapping"), True, "fusion ordering")
    _expect(final.get("fusion_status"), status, "final/audit fusion status")
    _expect(
        final.get("fusion_committed_to_active_map_before_downstream_mapping"),
        status == "accepted",
        "fusion commit disclosure",
    )
    _expect(
        final.get(
            "imported_gaussian_survival_in_serialized_final_model_not_separately_tracked"
        ),
        True,
        "final imported-Gaussian survival disclosure",
    )
    _expect(final.get("fusion_audit_sha256"), BASE._sha256_file(audit_path), "fusion audit digest")

    timing = audit.get("timing") or {}
    timing_values = {
        name: BASE._finite_float(timing.get(name), f"fusion timing {name}")
        for name in (
            "snapshot_seconds",
            "subprocess_and_publication_seconds",
            "premerge_active_render_seconds",
            "merge_seconds",
            "postmerge_active_render_seconds",
            "rollback_seconds",
            "total_wall_seconds",
        )
    }
    if timing_values["total_wall_seconds"] <= 0.0:
        raise ContractError("fusion total wall timing is not positive")

    premerge = audit.get("premerge_gates") or {}
    published = audit.get("published_result") or {}
    if status != "sidecar_rejected" and not published:
        raise ContractError(f"fusion status {status} lacks its bound published manifest")
    manifest_record: Optional[dict[str, Any]] = None
    if published:
        manifest_path = Path(str(published.get("manifest_path", ""))).expanduser().resolve()
        _expect(published.get("manifest_sha256"), BASE._sha256_file(manifest_path), "published manifest digest")
        manifest = _load_json(manifest_path, "published state3 manifest")
        repository = ((manifest.get("official_resplat") or {}).get("repository") or {})
        _expect(repository.get("commit"), RUNNER.EXPECTED_RESPLAT_COMMIT, "runtime ReSplat commit")
        _expect(
            manifest.get("selection_membership_clear_gt_conditioned"),
            True,
            "runtime manifest conditioned membership",
        )
        for field in (
            "ground_truth_pose_or_depth_used",
            "independent_clear_pixels_used",
            "clear_gt_metrics_used",
        ):
            _expect(manifest.get(field), False, f"runtime manifest {field}")
        execution = manifest.get("execution_contract") or {}
        _expect(execution.get("requested_recurrent_updates"), 3, "manifest state count")
        _expect(execution.get("returned_recurrent_states"), 3, "manifest returned states")
        _expect(
            execution.get("selected_state_index_zero_based"),
            2,
            "manifest selected state",
        )
        _expect(execution.get("fourth_state_computed"), False, "manifest fourth state")
        arrays = (manifest.get("outputs") or {}).get("npz_arrays") or {}
        _expect(
            set(arrays),
            {"means", "covariances", "scales", "rotations", "harmonics", "opacities"},
            "production native NPZ six-array contract",
        )
        manifest_record = {
            "path": str(manifest_path),
            "sha256": published["manifest_sha256"],
            "repository_commit": repository["commit"],
            "native_npz_array_names": sorted(arrays),
        }
        repository_gate = premerge.get("repository_provenance") or {}
        _expect(repository_gate.get("accepted"), True, "runtime repository gate")
        _expect(repository_gate.get("observed_commit"), RUNNER.EXPECTED_RESPLAT_COMMIT, "repository gate commit")
        data_gate = premerge.get("data_lineage") or {}
        gate_decisions = {
            "world_artifact": (premerge.get("world_artifact") or {}).get("accepted"),
            "context_reconstruction": (
                premerge.get("context_reconstruction") or {}
            ).get("accepted"),
            "repository_provenance": repository_gate.get("accepted"),
            "data_lineage": data_gate.get("accepted"),
        }
        if status == "premerge_gate_rejected":
            if not all(value is True or value is False for value in gate_decisions.values()):
                raise ContractError("premerge rejection gate decisions are incomplete")
            if all(value is True for value in gate_decisions.values()):
                raise ContractError("premerge rejection has no explicitly rejected gate")
        elif status in {
            "merge_gate_rejected",
            "postmerge_gate_rejected_rolled_back",
            "accepted",
        }:
            for name, value in gate_decisions.items():
                _expect(value, True, f"runtime premerge gate {name}")

    changed = bool(audit.get("active_map_changed_final", False))
    effective = status == "accepted"
    if effective:
        _expect(changed, True, "accepted fusion map change")
        merge = audit.get("merge") or {}
        _expect(merge.get("active_map_changed"), True, "accepted merge trial")
        _expect(merge.get("mode"), "append", "accepted merge mode")
        post = ((audit.get("postmerge_gate") or {}).get("decision") or {})
        _expect(post.get("accepted"), True, "accepted postmerge gate")
        final_state = audit.get("active_state_final") or {}
        _expect(final_state.get("byte_identical_to_premerge"), False, "accepted final map identity")
    else:
        _expect(changed, False, "rejected fusion final map change")
        if status in {"merge_gate_rejected", "postmerge_gate_rejected_rolled_back"}:
            before = audit.get("active_state_before") or {}
            final_state = audit.get("active_state_final") or {}
            _expect(final_state.get("byte_identical_to_premerge"), True, "rejected merge rollback identity")
            _expect(final_state.get("sha256"), before.get("sha256"), "rollback state digest")
            _expect(final_state.get("gaussian_count"), before.get("gaussian_count"), "rollback Gaussian count")
        if status == "merge_gate_rejected":
            merge = audit.get("merge") or {}
            _expect(merge.get("active_map_changed"), False, "merge-gate rejection change")
        if status == "postmerge_gate_rejected_rolled_back":
            post = ((audit.get("postmerge_gate") or {}).get("decision") or {})
            _expect(post.get("accepted"), False, "postmerge rejection decision")

    return {
        "enabled": True,
        "attempt_count": 1,
        "status": status,
        "fusion_committed_to_active_map_before_downstream_mapping": effective,
        "active_map_changed_at_commit": changed,
        "imported_gaussian_survival_in_serialized_final_model_not_separately_tracked": True,
        "trigger_context_source_indices": list(EXPECTED_CONTEXT),
        "trigger_source_index": 166,
        "downstream_online_mapped_source_indices": list(EXPECTED_DOWNSTREAM),
        "timing": timing_values,
        "published_manifest": manifest_record,
        "rejection_reasons": list(audit.get("rejection_reasons") or []),
        "audit_path": str(audit_path),
        "audit_sha256": BASE._sha256_file(audit_path),
    }


def _read_arm(root: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    arm_root = root / ARMS[arm]
    scene_root = arm_root / SCENE
    cfg = BASE._load_yaml(scene_root / "cfg.yaml", f"{arm} resolved config")
    _validate_config(cfg, arm, arm_root)
    preflight = _load_json(arm_root / "preflight.json", f"{arm} preflight")
    _validate_preflight(preflight, cfg, arm)
    launcher = _load_json(arm_root / "launcher_runtime.json", f"{arm} launcher runtime")
    _expect(launcher.get("experiment_revision"), "v2", f"{arm} launcher revision")
    runtime = BASE._read_runtime(arm_root, scene_root, arm)
    metrics = BASE._read_metrics(arm_root, scene_root, arm)
    keyframes = BASE._read_keyframes(scene_root, metrics, arm)
    _expect(tuple(keyframes["source_indices"]), EXPECTED_FIXED, f"{arm} exact final keyframes")
    frontend = BASE._read_frontend(arm_root, keyframes["source_indices"], "baseline")
    trajectory = BASE._read_trajectory(scene_root, keyframes["count"], arm)
    gpu = _read_gpu_monitor(arm_root, arm)
    fusion = _read_fusion(scene_root, arm)
    runtime_stats = _load_json(scene_root / "runtime_stats.json", f"{arm} runtime stats")
    if arm == "baseline":
        if bool(runtime_stats.get("official_resplat_active_fusion_enabled", False)):
            raise ContractError("baseline runtime claims active fusion")
    else:
        _expect(runtime_stats.get("official_resplat_active_fusion_enabled"), True, "fused runtime switch")
        _expect(runtime_stats.get("official_resplat_active_fusion_status"), fusion["status"], "fused runtime status")
        recorded_wall = BASE._finite_float(
            runtime_stats.get("official_resplat_active_fusion_total_wall_seconds"),
            "fused runtime fusion wall",
        )
        if not math.isclose(recorded_wall, fusion["timing"]["total_wall_seconds"], abs_tol=1e-9):
            raise ContractError("fusion wall timing disagrees between runtime and audit")
        if recorded_wall > runtime["official_timer_online_seconds"]:
            raise ContractError("fusion wall time is not contained in online timer")
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
            "active_resplat_fusion": fusion,
        },
        cfg,
        preflight,
    )


def _validate_implementation_provenance(
    preflights: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    left = preflights["baseline"].get("implementation_provenance") or {}
    right = preflights["fused"].get("implementation_provenance") or {}
    _expect(left, right, "per-arm implementation provenance")
    _expect(left.get("schema"), "unblur_slam.implementation_provenance.v1", "implementation schema")
    files = left.get("files") or {}
    _expect(set(files), set(RUNNER.CODE_PROVENANCE_FILES), "implementation file set")
    verified = {}
    for label, record in files.items():
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        observed = BASE._sha256_file(path)
        _expect(record.get("sha256"), observed, f"implementation hash {label}")
        verified[label] = {"path": str(path), "sha256": observed}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    _expect(left.get("git_head"), head, "implementation git HEAD")
    return {
        "git_head": head,
        "git_worktree_clean_at_launch": bool(left.get("git_worktree_clean")),
        "git_dirty_entry_count_at_launch": int(left.get("git_dirty_entry_count", -1)),
        "git_dirty_disclosure_at_launch": left.get("git_dirty_disclosure"),
        "implementation_files_sha256_verified_at_report_time": True,
        "files": verified,
    }


def build_report(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    arms: dict[str, Any] = {}
    configs: dict[str, Mapping[str, Any]] = {}
    preflights: dict[str, Mapping[str, Any]] = {}
    for arm in ("baseline", "fused"):
        arms[arm], configs[arm], preflights[arm] = _read_arm(root, arm)
    raw_differences = BASE._pair_differences(configs["baseline"], configs["fused"])
    differences = {
        path: {"baseline": record.get("baseline"), "fused": record.get("turtle")}
        for path, record in raw_differences.items()
    }
    _expect(set(differences), RUNNER.ALLOWED_PAIR_DIFFS, "resolved config difference set")
    for path in differences:
        if path not in RUNNER.ALLOWED_PAIR_DIFFS:
            raise ContractError(f"unexpected paired config difference: {path}")
    for arm in ("baseline", "fused"):
        _expect(
            (preflights[arm].get("paired_contract") or {}).get(
                "allowed_resolved_config_differences"
            ),
            differences,
            f"{arm} preflight/resolved config differences",
        )
    implementation = _validate_implementation_provenance(preflights)

    baseline, fused = arms["baseline"], arms["fused"]
    br, fr = baseline["runtime"], fused["runtime"]
    bm, fm = baseline["rendering_and_depth"], fused["rendering_and_depth"]
    bt, ft = baseline["trajectory"], fused["trajectory"]
    comparison = {
        "same_exact_eleven_final_keyframes": baseline["keyframes"]["source_indices"]
        == fused["keyframes"]["source_indices"]
        == list(EXPECTED_FIXED),
        "fusion_committed_to_active_map_before_downstream_mapping": fused[
            "active_resplat_fusion"
        ][
            "fusion_committed_to_active_map_before_downstream_mapping"
        ],
        "imported_gaussian_survival_in_serialized_final_model_not_separately_tracked": True,
        "runtime": {
            "fused_over_baseline_online_time_x": fr["official_timer_online_seconds"]
            / br["official_timer_online_seconds"],
            "fused_over_baseline_online_speed_x": br["official_timer_online_seconds"]
            / fr["official_timer_online_seconds"],
            "online_seconds_delta_fused_minus_baseline": fr["official_timer_online_seconds"]
            - br["official_timer_online_seconds"],
            "derived_prefix_online_fps_delta_fused_minus_baseline": fr[
                "derived_prefix_online_fps"
            ]
            - br["derived_prefix_online_fps"],
            "whole_device_peak_lower_bound_mib_delta_fused_minus_baseline": fused[
                "whole_device_gpu"
            ]["whole_device_peak_lower_bound_mib"]
            - baseline["whole_device_gpu"]["whole_device_peak_lower_bound_mib"],
        },
        "quality_delta_fused_minus_baseline": {
            "psnr_db": fm["psnr_db"] - bm["psnr_db"],
            "ssim": fm["ssim"] - bm["ssim"],
            "lpips": fm["lpips"] - bm["lpips"],
            "depth_l1": fm["depth_l1"] - bm["depth_l1"],
            "full_trajectory_ate_rmse_m": ft["full_trajectory_ate_rmse_m"]
            - bt["full_trajectory_ate_rmse_m"],
            "keyframe_trajectory_ate_rmse_m": ft["keyframe_trajectory_ate_rmse_m"]
            - bt["keyframe_trajectory_ate_rmse_m"],
        },
    }
    return {
        "schema": "unblur_slam.fr2_xyz_fixed_kf_resplat3_active_fusion_221_report.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(root),
        "scope": {
            "experiment_revision": "v2",
            "superseded_v1_output_root": str(RUNNER.SUPERSEDED_V1_OUTPUT_ROOT),
            "v1_outputs_preserved_and_not_reused": True,
            "v1_fused_run_failed_closed_due_to_mapper_hook_binding": True,
            "scene": SCENE,
            "source_first": 0,
            "source_last": 220,
            "source_count": SOURCE_COUNT,
            "fixed_final_keyframes": list(EXPECTED_FIXED),
            "actually_mapped_before_fusion": list(EXPECTED_CONTEXT),
            "fusion_trigger_source_index": 166,
            "downstream_online_mapping_after_fusion": list(EXPECTED_DOWNSTREAM),
            "clear_gt_metric_scope": "clear_gt_prefix_smoke",
            "selection_membership_clear_gt_conditioned": True,
            "fusion_gates_consume_clear_gt_pixels_or_metrics": False,
            "fusion_consumes_ground_truth_pose_or_depth": False,
            "final_refinement_iterations": 100,
            "paper_table_6_fps_comparable": False,
            "paper_26k_offline_refinement": False,
            "single_scene_single_seed_prefix_diagnostic": True,
        },
        "pair_contract": {
            "same_official_evssm_both_arms": True,
            "same_seed_resolution_schedule_and_optimization_budgets": True,
            "allowed_resolved_config_differences": differences,
            "fusion_subprocess_gate_merge_and_rollback_inside_online_timer": True,
            "whole_device_gpu_sampling_interval_ms_both_arms": 250,
        },
        "implementation_provenance": implementation,
        "arms": arms,
        "comparison": comparison,
        "interpretation_notes": [
            "Fusion is committed to the active map only when status=accepted; every rejected trial is reported as rejected, not as an improvement.",
            "An accepted import participates in source-220 mapping and final-100 optimization, but survival of each imported Gaussian in the saved final map is not tracked.",
            "The synchronous ReSplat subprocess, validation, merge, post-gate, and any rollback are charged to online_inference_time.",
            "Fusion occurs after mapped source 166 and before one downstream online mapping update at source 220, then before the ordinary final-100 refinement.",
            "The frozen 11-keyframe schedule was historically conditioned on clear-GT protocol membership; fusion itself consumes no GT pose/depth, independent clear pixels, or clear-GT metric.",
            "The historical Turtle-named 71680-Gaussian file is used only as a CPU numerical bridge probe and is not an input to either experiment arm; both arms run official EVSSM.",
            "Derived prefix FPS is 221/online_inference_time and is not comparable to the paper's Table 6 FPS.",
            "Continuous 250 ms nvidia-smi maxima are lower bounds, not exact instantaneous whole-device peaks.",
            "PSNR/SSIM are higher-is-better; LPIPS, depth L1, and ATE are lower-is-better.",
        ],
    }


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for arm in ("baseline", "fused"):
        record = report["arms"][arm]
        runtime = record["runtime"]
        metrics = record["rendering_and_depth"]
        trajectory = record["trajectory"]
        fusion = record["active_resplat_fusion"]
        rows.append(
            {
                "arm": arm,
                "fusion_status": fusion.get("status", "disabled"),
                "fusion_committed_before_downstream_mapping": fusion[
                    "fusion_committed_to_active_map_before_downstream_mapping"
                ],
                "imported_gaussian_survival_in_serialized_final_model_not_separately_tracked": True,
                "source_frames": SOURCE_COUNT,
                "fixed_keyframes": ";".join(map(str, EXPECTED_FIXED)),
                "online_seconds": runtime["official_timer_online_seconds"],
                "total_seconds": runtime["official_timer_total_seconds"],
                "external_wall_seconds": runtime["external_launcher_wall_seconds"],
                "derived_prefix_online_fps": runtime["derived_prefix_online_fps"],
                "mapper_torch_peak_gpu_gib": runtime["mapper_torch_peak_gpu_gib"],
                "whole_device_peak_lower_bound_mib": record["whole_device_gpu"][
                    "whole_device_peak_lower_bound_mib"
                ],
                "psnr_db": metrics["psnr_db"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "depth_l1": metrics["depth_l1"],
                "full_trajectory_ate_rmse_m": trajectory[
                    "full_trajectory_ate_rmse_m"
                ],
                "keyframe_trajectory_ate_rmse_m": trajectory[
                    "keyframe_trajectory_ate_rmse_m"
                ],
                "paper_table_6_comparable": False,
                "paper_26k_refinement": False,
            }
        )
    return rows


def write_report(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "fused_221_v2_audit.json"
    csv_path = destination / "fused_221_v2_metrics.csv"
    for path in (json_path, csv_path):
        if path.exists() or path.is_symlink():
            raise ContractError(f"refusing to overwrite report output: {path}")
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
    except ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
