#!/usr/bin/env python3
"""Fail-closed report for the isolated unsafe v6 virtual-count bugfix run."""

from __future__ import annotations

import argparse
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
    "virtual_count_bugfix_v6_runner_contract",
    REPO_ROOT / "scripts/run_fr2_fixed_kf_resplat3_virtual_count_bugfix_221.py",
)
V5_REPORT = _load_module(
    "virtual_count_bugfix_v5_report_contract",
    REPO_ROOT
    / "scripts/report_fr2_fixed_kf_resplat3_visibility_cache_refresh_forced_commit_221.py",
)
BASE = V5_REPORT.BASE
V2_REPORT = V5_REPORT.V2_REPORT
ContractError = BASE.ContractError
DEFAULT_ROOT = RUNNER.OUTPUT_ROOT
SCENE = "freiburg2_xyz"


def _expect(value: Any, expected: Any, label: str) -> None:
    BASE._expect(value, expected, label)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return BASE._load_json(path, label)


def _validate_v6_preflight(
    cfg: Mapping[str, Any], preflight: Mapping[str, Any], arm_root: Path
) -> None:
    v6, v5 = RUNNER._load_configs()
    differences = RUNNER._validate_config(v6, v5)
    _expect(cfg, v6, "v6 persisted resolved config")
    _expect(
        Path(str((cfg.get("data") or {}).get("output", ""))).resolve(),
        arm_root.resolve(),
        "v6 output root",
    )
    _expect(
        preflight.get("schema"),
        "unblur_slam.fr2_xyz_resplat3_virtual_count_bugfix_221_preflight.v6",
        "v6 preflight schema",
    )
    scope = preflight.get("scope") or {}
    for field in (
        "single_fresh_v6_fused_arm_only",
        "unsafe_not_deployable",
    ):
        _expect(scope.get(field), True, f"v6 preflight scope {field}")
    _expect(scope.get("fresh_paired_comparison"), False, "v6 paired scope")
    _expect(scope.get("v5_runtime_artifacts_reused"), False, "v6 v5 reuse")
    scientific = preflight.get("scientific_contract") or {}
    _expect(
        scientific.get("direct_v5_inheritance"),
        RUNNER._validate_direct_v5_inheritance(),
        "v6 direct-v5 inheritance",
    )
    _expect(
        scientific.get("allowed_v6_v5_resolved_differences"),
        differences,
        "v6/v5 resolved differences",
    )
    _expect(
        scientific.get("only_code_change_from_v5"),
        RUNNER._validate_bugfix(),
        "v6 exact code bugfix",
    )
    _expect(
        scientific.get(
            "mapping_model_schedule_gates_merge_count_and_cache_contract_exactly_equal"
        ),
        True,
        "v6 scientific equality",
    )
    _expect(scientific.get("official_resplat_fresh_rerun"), True, "v6 fresh ReSplat")
    _expect(
        preflight.get("frozen_v5_failure_lineage"),
        RUNNER._v5_failure_binding(),
        "v6 frozen v5 failure lineage",
    )
    _expect(
        preflight.get("v5_cpu_preflight_reused_as_read_only_validation"),
        RUNNER.V5.preflight(check_output_available=False),
        "v6 nested v5 read-only preflight",
    )
    _expect(
        preflight.get("execution"),
        {
            "selected_arms": ["v6_virtual_count_bugfix_fused"],
            "physical_gpu": int(RUNNER.PHYSICAL_GPU),
            "process_visible_device": "cuda:0",
            "output_root": str(RUNNER.OUTPUT_ROOT),
            "config": str(RUNNER.CONFIG),
        },
        "v6 execution contract",
    )
    _expect(
        preflight.get("implementation_provenance"),
        RUNNER._implementation_provenance(),
        "v6 implementation provenance",
    )
    files = ((preflight.get("implementation_provenance") or {}).get("files") or {})
    for label in (
        "mapper", "active_fusion_helper", "active_map_merge", "world_bridge",
        "sidecar_runner", "sidecar_verifier", "gaussian_model",
        "slam_terminal_writer", "experiment_runner_v6", "experiment_report_v6",
    ):
        binding = files.get(label) or {}
        path = Path(str(binding.get("path", ""))).resolve()
        _expect(binding.get("sha256"), BASE._sha256_file(path), f"v6 code {label}")


def _read_v6_arm(root: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    arm_root = root / "evssm_resplat3_virtual_count_bugfix"
    scene_root = arm_root / SCENE
    cfg = BASE._load_yaml(scene_root / "cfg.yaml", "v6 resolved config")
    preflight = _load_json(arm_root / "preflight.json", "v6 launch preflight")
    _validate_v6_preflight(cfg, preflight, arm_root)
    launcher = _load_json(arm_root / "launcher_runtime.json", "v6 launcher runtime")
    _expect(launcher.get("schema"), "unblur_slam.external_wall_runtime.v1", "v6 launcher schema")
    _expect(launcher.get("experiment_revision"), "v6_virtual_count_bugfix", "v6 launcher revision")
    _expect(launcher.get("arm"), "v6_virtual_count_bugfix_fused", "v6 launcher arm")
    _expect(launcher.get("exit_code"), 0, "v6 launcher exit code")
    _expect(launcher.get("unsafe_not_deployable"), True, "v6 safety disclosure")
    _expect(launcher.get("v5_runtime_artifacts_reused"), False, "v6 artifact reuse")
    _expect(
        launcher.get("only_code_change_from_v5"),
        "bind_positive_virtual_count_before_context_metric_virtual_extrinsics_validation",
        "v6 launcher code change",
    )
    runtime = BASE._read_runtime(arm_root, scene_root, "v6_virtual_count_bugfix_fused")
    metrics = BASE._read_metrics(arm_root, scene_root, "v6_virtual_count_bugfix_fused")
    keyframes = BASE._read_keyframes(scene_root, metrics, "v6_virtual_count_bugfix_fused")
    _expect(
        tuple(keyframes["source_indices"]),
        RUNNER.EXPECTED_FIXED_SOURCE_KEYFRAMES,
        "v6 exact fixed keyframes",
    )
    frontend = BASE._read_frontend(arm_root, keyframes["source_indices"], "baseline")
    trajectory = BASE._read_trajectory(
        scene_root, keyframes["count"], "v6_virtual_count_bugfix_fused"
    )
    gpu = V2_REPORT._read_gpu_monitor(arm_root, "fused")
    fusion = V5_REPORT._read_forced_chain(scene_root, cfg, preflight)
    runtime_stats = _load_json(scene_root / "runtime_stats.json", "v6 runtime stats")
    _expect(runtime_stats.get("official_resplat_active_fusion_enabled"), True, "v6 fusion enabled")
    _expect(
        runtime_stats.get("official_resplat_active_fusion_status"),
        fusion["status"],
        "v6 runtime fusion status",
    )
    audit_wall = BASE._finite_float(
        (fusion.get("timing") or {}).get("total_wall_seconds"), "v6 audit fusion wall"
    )
    runtime_wall = BASE._finite_float(
        runtime_stats.get("official_resplat_active_fusion_total_wall_seconds"),
        "v6 runtime fusion wall",
    )
    if not math.isclose(audit_wall, runtime_wall, rel_tol=0.0, abs_tol=1e-9):
        raise ContractError("v6 runtime/audit fusion timing mismatch")
    cache_audit = BASE._finite_float(
        (fusion.get("timing") or {}).get("visibility_cache_refresh_seconds"),
        "v6 audit cache time",
    )
    cache_runtime = BASE._finite_float(
        runtime_stats.get("official_resplat_active_fusion_visibility_cache_refresh_seconds"),
        "v6 runtime cache time",
    )
    cache_report = BASE._finite_float(
        (fusion.get("visibility_cache_refresh") or {}).get("elapsed_seconds"),
        "v6 report cache time",
    )
    if not (
        math.isclose(cache_audit, cache_runtime, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(cache_audit, cache_report, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ContractError("v6 cache timing bindings disagree")
    timing_fields = {
        "forced_commit_chain_commit_event_seconds": "commit_validation_and_event_publication",
        "forced_commit_chain_source220_entry_seconds": "source220_entry_capture",
        "forced_commit_chain_source220_complete_seconds": "source220_complete_capture_and_event_publication",
        "forced_commit_chain_final100_entry_seconds": "final100_entry_capture",
        "forced_commit_chain_final100_complete_seconds": "final100_complete_capture_and_event_publication",
    }
    terminal_stages = (
        (fusion.get("terminal_timing_disclosure") or {}).get("recorded_stage_seconds")
        or {}
    )
    for field, terminal_field in timing_fields.items():
        value = BASE._finite_float(runtime_stats.get(field), f"v6 overhead {field}")
        terminal_value = BASE._finite_float(
            terminal_stages.get(terminal_field),
            f"v6 terminal overhead {terminal_field}",
        )
        if value < 0.0 or not math.isclose(
            value, terminal_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ContractError(f"v6 runtime/terminal overhead mismatch: {field}")
    if audit_wall > runtime["official_timer_online_seconds"]:
        raise ContractError("v6 ReSplat/merge wall time escaped online timer")
    return ({
        "output_root": str(arm_root),
        "frontend": "official_unblur_slam_evssm",
        "runtime": runtime,
        "frontend_activity": frontend,
        "rendering_and_depth": metrics,
        "keyframes": keyframes,
        "trajectory": trajectory,
        "whole_device_gpu": gpu,
        "forced_active_resplat_fusion": fusion,
    }, preflight)


def build_report(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = root.expanduser().resolve()
    baseline, _, _ = V2_REPORT._read_arm(RUNNER.V5.V2_ROOT, "baseline")
    rollback, _, _ = V2_REPORT._read_arm(RUNNER.V5.V2_ROOT, "fused")
    v2_published = _load_json(RUNNER.V5.V2_EXPERIMENT_AUDIT, "frozen v2 audit")
    _expect(v2_published.get("arms", {}).get("baseline"), baseline, "v2 baseline path")
    _expect(v2_published.get("arms", {}).get("fused"), rollback, "v2 rollback path")
    v6, preflight = _read_v6_arm(root)
    return {
        "schema": "unblur_slam.fr2_xyz_resplat3_virtual_count_bugfix_221_report.v6",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(root),
        "scope": {
            "unsafe_not_deployable": True,
            "single_fresh_v6_arm": True,
            "fresh_paired_comparison": False,
            "v5_failure_is_frozen_read_only_lineage": True,
            "v5_runtime_artifacts_reused": False,
            "only_code_change_from_v5_is_virtual_count_binding_and_positive_guard": True,
        },
        "reference_integrity": {
            "v5_failure": preflight["frozen_v5_failure_lineage"],
            "v2_baseline_and_rollback_are_frozen_cross_revision_references": True,
        },
        "arms": {
            "v2_baseline_reference": baseline,
            "v2_safe_rollback_reference": rollback,
            "v6_virtual_count_bugfix_diagnostic": v6,
        },
        "diagnostic_comparisons_not_fair_paired_estimates": {
            "v6_minus_v2_baseline_reference": V5_REPORT._delta(v6, baseline),
            "v6_minus_v2_safe_rollback_reference": V5_REPORT._delta(v6, rollback),
        },
        "interpretation_notes": [
            "v5 failed before merge because virtual_count was unbound; its active map remained unchanged.",
            "v6 directly inherits v5 and changes only the local virtual-count binding plus a fail-closed positive-count guard; every scientific numeric contract is unchanged.",
            "v6 reruns official ReSplat from a fresh snapshot and consumes no v5 runtime artifact.",
            "References are cross-revision and therefore quality/speed deltas are diagnostic, not a fair paired estimate.",
            "The underlying forced-commit protocol remains explicitly unsafe and is not deployable.",
        ],
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fr2_xyz_resplat3_virtual_count_bugfix_221_report.v6.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite v6 report: {path}")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args.root)
        output = args.output_dir or args.root / "_report"
        path = write_report(report, output)
        print(json.dumps({"report": str(path)}, indent=2))
        return 0
    except (
        ContractError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
