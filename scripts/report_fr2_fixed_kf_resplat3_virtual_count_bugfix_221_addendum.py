#!/usr/bin/env python3
"""Build or validate the immutable post-run audit addendum for v6.

The primary v6 report is intentionally left byte-for-byte unchanged.  This
addendum makes the frozen v2 references, selection lineage, FPS semantics,
and terminal artifact bindings self-contained for external review.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
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


REPORT = _load_module(
    "virtual_count_bugfix_v6_primary_report_for_addendum",
    REPO_ROOT / "scripts/report_fr2_fixed_kf_resplat3_virtual_count_bugfix_221.py",
)
RUNNER = REPORT.RUNNER

DEFAULT_ROOT = RUNNER.OUTPUT_ROOT
PRIMARY_REPORT_NAME = "fr2_xyz_resplat3_virtual_count_bugfix_221_report.v6.json"
ADDENDUM_NAME = "fr2_xyz_resplat3_virtual_count_bugfix_221_report.v6.addendum.v1.json"
PRIMARY_REPORT_SHA256 = "f93621dee6d28f97f2b6ce0af40b44f8367c4e39e3d717fca27c44bffb347c73"
PRIMARY_REPORT_SIZE_BYTES = 18_313
V6_ARM_TREE_SHA256 = "a7b6ff427c733276ace0d7582e50944cfe2f45410826b8b23b6aa70783e82cba"
V6_ARM_TREE_FILE_COUNT = 104
V6_ARM_TREE_TOTAL_BYTES = 95_380_540
FULL_PREFIX_FRAME_COUNT = 221
PERSISTED_PREFLIGHT_DIRTY_ENTRY_COUNT = 198
ADDENDUM_AUDIT_CODE_PATHS = (
    "scripts/report_fr2_fixed_kf_resplat3_virtual_count_bugfix_221_addendum.py",
    "tests/test_fr2_fixed_kf_resplat3_virtual_count_bugfix_221_addendum.py",
)


class AddendumError(RuntimeError):
    """Raised when a persisted post-run contract fails closed."""


def _expect(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise AddendumError(f"{label} mismatch: observed={value!r}, expected={expected!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise AddendumError(f"required regular file is missing or symlinked: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AddendumError(f"cannot load {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise AddendumError(f"{label} is not a JSON object: {path}")
    return value


def _tree_binding(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise AddendumError(f"v6 arm root is missing or symlinked: {root}")
    records = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AddendumError(f"v6 arm tree contains a symlink: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "size": size,
            }
        )
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _expect(digest, V6_ARM_TREE_SHA256, "v6 arm canonical tree SHA-256")
    _expect(len(records), V6_ARM_TREE_FILE_COUNT, "v6 arm file count")
    _expect(total_bytes, V6_ARM_TREE_TOTAL_BYTES, "v6 arm total bytes")
    return {
        "schema": "unblur_slam.canonical_regular_file_tree_binding.v1",
        "root": str(root),
        "tree_sha256": digest,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "record_fields": ["path", "sha256", "size"],
        "records_sorted_by_path": True,
        "symlinks_rejected": True,
    }


def _post_report_code_extension_binding() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    entries = tuple(line for line in completed.stdout.splitlines() if line)
    expected_extension_entries = tuple(f"?? {path}" for path in ADDENDUM_AUDIT_CODE_PATHS)
    for entry in expected_extension_entries:
        _expect(entries.count(entry), 1, f"post-report audit code status {entry}")
    current = RUNNER.V5._implementation_provenance()
    current_count = int(current.get("git_dirty_entry_count", -1))
    _expect(
        current_count,
        PERSISTED_PREFLIGHT_DIRTY_ENTRY_COUNT + len(ADDENDUM_AUDIT_CODE_PATHS),
        "post-report audit code dirty-entry delta",
    )
    status_sha256 = hashlib.sha256(
        json.dumps(entries, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "unblur_slam.post_report_audit_code_extension.v1",
        "persisted_preflight_git_dirty_entry_count": PERSISTED_PREFLIGHT_DIRTY_ENTRY_COUNT,
        "current_git_dirty_entry_count": current_count,
        "exact_new_untracked_paths": list(ADDENDUM_AUDIT_CODE_PATHS),
        "exact_dirty_entry_count_delta": len(ADDENDUM_AUDIT_CODE_PATHS),
        "current_git_status_short_all_entries_sha256": status_sha256,
        "normalization_scope_for_primary_report_regeneration": (
            "git_dirty_entry_count_only; all implementation file paths and SHA-256 "
            "bindings remain exact"
        ),
    }


@contextmanager
def _persisted_preflight_provenance_view():
    """Recreate the pre-addendum dirty-count view without hiding code drift.

    The primary report exactly bound every implementation file but also stored
    the total dirty-entry count.  Adding this generator and its test increases
    only that count by two.  The two exact untracked paths and the +2 delta are
    verified before the historical count is substituted temporarily.
    """

    extension = _post_report_code_extension_binding()
    original = RUNNER.V5._implementation_provenance

    def normalized() -> dict[str, Any]:
        provenance = original()
        _expect(
            provenance.get("git_dirty_entry_count"),
            extension["current_git_dirty_entry_count"],
            "current implementation dirty-entry count",
        )
        provenance["git_dirty_entry_count"] = PERSISTED_PREFLIGHT_DIRTY_ENTRY_COUNT
        return provenance

    RUNNER.V5._implementation_provenance = normalized
    try:
        yield extension
    finally:
        RUNNER.V5._implementation_provenance = original


def _primary_report_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "_report" / PRIMARY_REPORT_NAME
    binding = _file_binding(path)
    _expect(binding["sha256"], PRIMARY_REPORT_SHA256, "primary report SHA-256")
    _expect(binding["size_bytes"], PRIMARY_REPORT_SIZE_BYTES, "primary report size")
    primary = _load_json(path, "primary v6 report")
    _expect(
        primary.get("schema"),
        "unblur_slam.fr2_xyz_resplat3_virtual_count_bugfix_221_report.v6",
        "primary report schema",
    )
    with _persisted_preflight_provenance_view():
        rebuilt = REPORT.build_report(root)
    primary_without_time = deepcopy(primary)
    rebuilt_without_time = deepcopy(rebuilt)
    primary_without_time.pop("generated_at_utc", None)
    rebuilt_without_time.pop("generated_at_utc", None)
    _expect(primary_without_time, rebuilt_without_time, "primary report regenerated content")
    generated = primary.get("generated_at_utc")
    if not isinstance(generated, str):
        raise AddendumError("primary report generated_at_utc is missing")
    try:
        datetime.fromisoformat(generated)
    except ValueError as error:
        raise AddendumError("primary report generated_at_utc is invalid") from error
    binding.update(
        {
            "schema": primary["schema"],
            "generated_at_utc": generated,
            "regenerated_content_exact_except_timestamp": True,
        }
    )
    return binding, primary


def _verified_preflight(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    arm_root = RUNNER.OUTPUT
    if arm_root.parent != root:
        raise AddendumError("configured v6 arm is outside the requested experiment root")
    preflight_path = arm_root / "preflight.json"
    preflight = _load_json(preflight_path, "v6 persisted preflight")
    _expect(
        preflight.get("schema"),
        "unblur_slam.fr2_xyz_resplat3_virtual_count_bugfix_221_preflight.v6",
        "v6 persisted preflight schema",
    )
    _expect(
        (preflight.get("execution") or {}).get("output_root"),
        str(root),
        "v6 persisted preflight output root",
    )
    return preflight, _file_binding(preflight_path)


def _verify_bound_files(bindings: Mapping[str, Any], label: str) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for name, binding in sorted(bindings.items()):
        if not isinstance(binding, Mapping):
            raise AddendumError(f"{label}.{name} is not a binding")
        path = Path(str(binding.get("path", ""))).expanduser().resolve()
        current = _file_binding(path)
        _expect(current["sha256"], binding.get("sha256"), f"{label}.{name} SHA-256")
        verified[name] = dict(binding)
    return verified


def _chain_bindings(scene_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fusion_root = scene_root / "official_resplat_active_fusion"
    chain_root = fusion_root / "forced_commit_chain"
    event_specs = (
        ("commit", "00_forced_commit.json", 0, "postgate_rejected_forced_commit"),
        ("source220", "01_source220_complete.json", 1, "source220_mapping_complete"),
        ("final100", "02_final100_complete.json", 2, "final100_complete"),
    )
    from src.refinement.official_resplat_active_fusion import canonical_contract_sha256

    result: dict[str, Any] = {}
    previous: Optional[str] = None
    events: dict[str, Any] = {}
    for label, filename, sequence, event_type in event_specs:
        path = chain_root / filename
        event = _load_json(path, f"v6 {label} event")
        _expect(event.get("sequence"), sequence, f"{label} sequence")
        _expect(event.get("event_type"), event_type, f"{label} event type")
        _expect(event.get("previous_event_sha256"), previous, f"{label} previous digest")
        canonical = canonical_contract_sha256(event, digest_field="event_sha256")
        _expect(event.get("event_sha256"), canonical, f"{label} canonical digest")
        result[label] = {
            **_file_binding(path),
            "canonical_event_sha256": canonical,
            "sequence": sequence,
            "event_type": event_type,
        }
        events[label] = event
        previous = canonical

    terminal_path = chain_root / "terminal_contract.json"
    terminal = _load_json(terminal_path, "v6 terminal contract")
    _expect(terminal.get("previous_event_sha256"), previous, "terminal previous digest")
    canonical_terminal = canonical_contract_sha256(
        terminal, digest_field="terminal_sha256"
    )
    _expect(terminal.get("terminal_sha256"), canonical_terminal, "terminal canonical digest")
    result["terminal"] = {
        **_file_binding(terminal_path),
        "canonical_terminal_sha256": canonical_terminal,
    }
    events["terminal"] = terminal
    return result, events


def _selection_and_gt_lineage(
    audit: Mapping[str, Any],
    final_contract: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    audit_lineage = audit.get("data_lineage") or {}
    premerge_lineage = ((audit.get("premerge_gates") or {}).get("data_lineage") or {})
    expected = {
        "selection_membership_clear_gt_conditioned": True,
        "ground_truth_poses_or_depths_consumed_by_fusion": False,
        "independent_clear_pixels_consumed_by_fusion": False,
        "clear_gt_metrics_consumed_by_fusion": False,
    }
    _expect(audit_lineage, expected, "fusion audit data lineage")
    _expect(premerge_lineage.get("accepted"), True, "premerge data-lineage gate")
    for field, value in expected.items():
        premerge_name = {
            "ground_truth_poses_or_depths_consumed_by_fusion": "ground_truth_pose_or_depth_used",
            "independent_clear_pixels_consumed_by_fusion": "independent_clear_pixels_used",
            "clear_gt_metrics_consumed_by_fusion": "clear_gt_metrics_used",
        }.get(field, field)
        _expect(premerge_lineage.get(premerge_name), value, f"premerge lineage {field}")
    _expect(
        final_contract.get("selection_membership_clear_gt_conditioned"),
        True,
        "final-contract conditioned membership",
    )
    _expect(final_contract.get("ground_truth_poses_or_depths_consumed_by_fusion"), False, "final-contract GT pose/depth")
    _expect(final_contract.get("independent_clear_pixels_consumed_by_fusion"), False, "final-contract clear pixels")
    _expect(final_contract.get("clear_gt_metrics_consumed_by_fusion"), False, "final-contract clear metrics")
    _expect(terminal.get("clear_gt_metrics_bound_posthoc_for_evaluation"), True, "terminal posthoc clear metrics")
    _expect(terminal.get("clear_gt_values_used_for_commit_or_checkpoint_selection"), False, "terminal clear-GT selection")
    _expect(terminal.get("uses_gt_for_forced_commit_decision"), False, "terminal GT decision")
    return {
        "selection_membership_clear_gt_conditioned": True,
        "fusion_consumes_ground_truth_pose_or_depth": False,
        "fusion_consumes_independent_clear_pixels": False,
        "fusion_consumes_clear_gt_metrics": False,
        "clear_gt_metrics_bound_posthoc_for_evaluation": True,
        "clear_gt_values_used_for_commit_or_checkpoint_selection": False,
        "distinction": (
            "the frozen 11-keyframe membership was historically clear-GT-protocol "
            "conditioned; the fusion gate and forced-commit decision consume no GT "
            "pose/depth, independent clear pixels, or clear-GT metric"
        ),
    }


def _speed_semantics(primary: Mapping[str, Any]) -> dict[str, Any]:
    arms = primary.get("arms") or {}
    observed: dict[str, Any] = {}
    for label in (
        "v2_baseline_reference",
        "v2_safe_rollback_reference",
        "v6_virtual_count_bugfix_diagnostic",
    ):
        arm = arms.get(label) or {}
        runtime = arm.get("runtime") or {}
        trajectory = arm.get("trajectory") or {}
        _expect(
            trajectory.get("full_trajectory_frame_count"),
            FULL_PREFIX_FRAME_COUNT,
            f"{label} full-prefix frame count",
        )
        online = float(runtime.get("official_timer_online_seconds"))
        fps = float(runtime.get("derived_prefix_online_fps"))
        expected_fps = FULL_PREFIX_FRAME_COUNT / online
        if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1e-15):
            raise AddendumError(f"{label} derived-prefix FPS arithmetic mismatch")
        gpu = arm.get("whole_device_gpu") or {}
        observed[label] = {
            "official_timer_online_seconds": online,
            "derived_prefix_online_fps": fps,
            "whole_device_peak_lower_bound_mib": gpu.get(
                "whole_device_peak_lower_bound_mib"
            ),
        }
    return {
        "derived_prefix_frame_count": FULL_PREFIX_FRAME_COUNT,
        "derived_prefix_fps_formula": "221 / official_timer_online_seconds",
        "paper_table_6_fps_comparable": False,
        "fresh_paired_comparison": False,
        "cross_revision_deltas_are_diagnostic_only": True,
        "whole_device_gpu_peak_is_250ms_sampled_lower_bound_not_exact_peak": True,
        "arms": observed,
    }


def _numeric_results(primary: Mapping[str, Any]) -> dict[str, Any]:
    arms = primary.get("arms") or {}
    result: dict[str, Any] = {}
    for label, arm in arms.items():
        result[label] = {
            "runtime": deepcopy(arm.get("runtime") or {}),
            "rendering_and_depth": deepcopy(arm.get("rendering_and_depth") or {}),
            "trajectory": deepcopy(arm.get("trajectory") or {}),
            "whole_device_gpu": deepcopy(arm.get("whole_device_gpu") or {}),
        }
    return {
        "arms": result,
        "diagnostic_comparisons_not_fair_paired_estimates": deepcopy(
            primary.get("diagnostic_comparisons_not_fair_paired_estimates") or {}
        ),
        "all_values_copied_exactly_from_primary_report": True,
    }


def _material(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    _expect(root, DEFAULT_ROOT, "addendum experiment root")
    primary_binding, primary = _primary_report_binding(root)
    preflight, preflight_binding = _verified_preflight(root)
    v5_validation = preflight.get("v5_cpu_preflight_reused_as_read_only_validation") or {}
    v2_references = deepcopy(v5_validation.get("frozen_v2_references") or {})
    _expect(
        v2_references.get("schema"),
        "unblur_slam.v2_frozen_dual_reference.v1",
        "complete frozen v2 dual-reference schema",
    )

    arm_root = RUNNER.OUTPUT
    scene_root = arm_root / REPORT.SCENE
    fusion_root = scene_root / "official_resplat_active_fusion"
    audit = _load_json(fusion_root / "fusion_audit.json", "v6 fusion audit")
    final_contract = _load_json(
        fusion_root / "fusion_final_contract.json", "v6 fusion final contract"
    )
    chain, events = _chain_bindings(scene_root)
    terminal = events["terminal"]
    _expect(audit.get("status"), "postmerge_gate_rejected_forced_commit_unsafe", "fusion status")
    _expect(audit.get("active_map_changed_final"), True, "forced active-map change")
    _expect((audit.get("postmerge_gate") or {}).get("decision", {}).get("accepted"), False, "postmerge gate")
    _expect((audit.get("unsafe_forced_commit") or {}).get("ordinary_action"), "rollback", "ordinary policy")
    _expect((audit.get("unsafe_forced_commit") or {}).get("rollback_performed"), False, "forced rollback")
    _expect(terminal.get("individual_imported_gaussian_survival_in_final_model_not_tracked"), True, "terminal survival tracking")
    _expect(terminal.get("individual_imported_gaussian_survival_in_final_model_not_claimed"), True, "terminal survival claim")

    terminal_artifacts = _verify_bound_files(
        terminal.get("artifact_bindings") or {}, "terminal artifacts"
    )
    terminal_code = _verify_bound_files(
        terminal.get("code_bindings") or {}, "terminal code"
    )
    launcher_binding = _file_binding(arm_root / "launcher_runtime.json")
    gpu_binding = _file_binding(arm_root / "gpu_monitor_summary.json")
    launcher = _load_json(arm_root / "launcher_runtime.json", "v6 launcher")
    _expect(launcher.get("exit_code"), 0, "v6 launcher exit code")

    return {
        "schema": "unblur_slam.fr2_xyz_resplat3_virtual_count_bugfix_221_report_addendum.v1",
        "primary_report": primary_binding,
        "post_report_audit_code_extension": _post_report_code_extension_binding(),
        "v6_arm_frozen_tree": _tree_binding(arm_root),
        "reference_integrity": {
            "v2_dual_reference": v2_references,
            "v2_references_revalidated_against_current_read_only_preflight": True,
            "v5_failure": deepcopy(
                (primary.get("reference_integrity") or {}).get("v5_failure") or {}
            ),
        },
        "selection_and_gt_lineage": _selection_and_gt_lineage(
            audit, final_contract, terminal
        ),
        "speed_semantics": _speed_semantics(primary),
        "postrun_integrity": {
            "preflight": preflight_binding,
            "launcher_runtime": launcher_binding,
            "gpu_monitor_summary": gpu_binding,
            "fusion_status": audit["status"],
            "postmerge_gate_accepted": False,
            "postmerge_gate_reasons": deepcopy(
                (audit.get("postmerge_gate") or {}).get("decision", {}).get("reasons")
                or []
            ),
            "unsafe_forced_commit": True,
            "ordinary_policy_would_rollback": True,
            "chain": chain,
            "terminal_artifact_bindings": terminal_artifacts,
            "terminal_code_bindings": terminal_code,
            "source220_and_final100_participation_proven": True,
            "individual_imported_gaussian_survival_not_tracked_or_claimed": True,
        },
        "numeric_results": _numeric_results(primary),
        "addendum_generator": _file_binding(Path(__file__).resolve()),
        "interpretation_notes": [
            "This addendum supplements but does not replace or modify the SHA-bound primary v6 report.",
            "The unchanged postmerge gate rejected the append; the active-map result exists only because an explicitly unsafe post-hoc diagnostic override forced the commit.",
            "The cross-revision v2 comparisons are diagnostic, not fresh paired estimates, and derived prefix FPS is not comparable to paper Table 6 FPS.",
            "The chain proves batch participation in source220 and final100 but does not identify or prove survival of individual imported Gaussians in the final model.",
        ],
    }


def build_addendum(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    record = _material(root)
    record["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def validate_addendum(
    record: Mapping[str, Any], root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise AddendumError("addendum is not an object")
    generated = record.get("generated_at_utc")
    if not isinstance(generated, str):
        raise AddendumError("addendum generated_at_utc is missing")
    try:
        datetime.fromisoformat(generated)
    except ValueError as error:
        raise AddendumError("addendum generated_at_utc is invalid") from error
    observed = deepcopy(dict(record))
    observed.pop("generated_at_utc", None)
    expected = _material(root)
    _expect(observed, expected, "published addendum content")
    return dict(record)


def write_addendum(record: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ADDENDUM_NAME
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite v6 addendum: {path}")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or args.root / "_report"
    path = output_dir / ADDENDUM_NAME
    try:
        if args.validate_only:
            record = _load_json(path, "published v6 addendum")
            validate_addendum(record, args.root)
            print(json.dumps({"validated_addendum": str(path)}, indent=2))
            return 0
        record = build_addendum(args.root)
        path = write_addendum(record, output_dir)
        print(json.dumps({"addendum": str(path)}, indent=2))
        return 0
    except (
        AddendumError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        REPORT.ContractError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
