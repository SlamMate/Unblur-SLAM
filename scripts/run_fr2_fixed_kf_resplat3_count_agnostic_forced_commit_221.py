#!/usr/bin/env python3
"""CPU-preflight or run the isolated unsafe v4 count-agnostic diagnostic.

This is not a deployable method and not a fresh paired comparison.  It runs
only one new EVSSM+official-ReSplat arm.  Official ReSplat is executed again;
no v2/v3 sidecar output is consumed.  A forced commit is authorized only after
the unchanged postmerge gate explicitly rejects a fresh append whose accepted
count is within [1024, 20000] and equals ``after_count-before_count``.  The
failed v3 exact-count attempt is bound as read-only lineage only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/unblur_slam.yaml"
V2_RUNNER_PATH = REPO_ROOT / "scripts/run_fr2_fixed_kf_resplat3_fusion_221.py"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = _load_module("forced_commit_v2_runner", V2_RUNNER_PATH)
BASE = V2.BASE
PHYSICAL_GPU = V2.PHYSICAL_GPU
EXPECTED_FIXED_SOURCE_KEYFRAMES = tuple(V2.EXPECTED_FIXED_SOURCE_KEYFRAMES)
EXPECTED_CONTEXT = tuple(V2.EXPECTED_FIRST_EIGHT_MAPPED_SOURCES)
EXPECTED_DOWNSTREAM = tuple(V2.EXPECTED_DOWNSTREAM_MAPPED_SOURCES)

V2_ROOT = V2.OUTPUT_ROOT
V2_BASELINE_ROOT = (V2_ROOT / "evssm_no_fusion").resolve()
V2_ROLLBACK_ROOT = (V2_ROOT / "evssm_resplat3_active_fusion").resolve()
V2_EXPERIMENT_AUDIT = (V2_ROOT / "_audit/fused_221_v2_audit.json").resolve()
V2_ROLLBACK_AUDIT = (
    V2_ROLLBACK_ROOT
    / "freiburg2_xyz/official_resplat_active_fusion/fusion_audit.json"
).resolve()
V2_BASELINE_TREE_SHA256 = (
    "54fa5800205dc980ce0cb8fc3f23798cdd07fe2b532fad1f6a3c723d9ee04ecc"
)
V2_ROLLBACK_TREE_SHA256 = (
    "5256a557e26f09a02438f241cae56ebad1bc654455d50d2a83a7951b62246eca"
)
V2_EXPERIMENT_AUDIT_SHA256 = (
    "f95b9fdf74e0458e7e7103640c764f27c2b126e368db0b3508fa354deae53bc3"
)
V2_ROLLBACK_AUDIT_SHA256 = (
    "a8e65b6851c998c0038bf228a337de18af9cc182cfa4d711e5a92d9543053c3d"
)
V2_BASELINE_CFG_YAML_SHA256 = (
    "705cc02064fe4219a0b48b26fde32067a53cbf36def8333f10500375c21f5133"
)
V2_BASELINE_CANONICAL_CONFIG_SHA256 = (
    "222abf4c8b7a96d04b93afc63a4a64dc706c0376070cb74be8ce9af1355d4956"
)
V2_ROLLBACK_CANONICAL_CONFIG_SHA256 = (
    "fe0b2709bd780a6b0f1b2f3e6693c6bd568617522b7e02d0e97a3b89a9648c52"
)
V3_FAILED_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_paired/"
    "fr2_xyz_fixed_kf_resplat3_fusion_221_v3_forced_commit/"
    "evssm_resplat3_forced_commit"
).resolve()
V3_FAILED_AUDIT = (
    V3_FAILED_ROOT
    / "freiburg2_xyz/official_resplat_active_fusion/fusion_audit.json"
).resolve()
V3_FAILED_AUDIT_SHA256 = (
    "1f0ef9f3657fdca2b4dec317de27e22c4d80bb4523df56a9a584b389fe863864"
)
V3_FAILED_TREE_SHA256 = (
    "9e12c3d3254efa2d579b4ee8bb32876eb6235821919ea24744b048c5c9e10ed0"
)
V3_FAILED_LAUNCHER_SHA256 = (
    "9dee7cb8a6d61d3350e685481945167bd7679cb0a30eb42e9e2a34c9cdbb3c89"
)

CONFIG = (
    REPO_ROOT
    / "configs/local/fr2_xyz_fixed_kf_resplat3_fusion_221_v4_count_agnostic_forced_commit"
    / "evssm_resplat3_count_agnostic_forced_commit.yaml"
)
OUTPUT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_paired/"
    "fr2_xyz_fixed_kf_resplat3_fusion_221_v4_count_agnostic_forced_commit"
).resolve()
OUTPUT = (OUTPUT_ROOT / "evssm_resplat3_count_agnostic_forced_commit").resolve()

ALLOWED_V4_V2_CONFIG_DIFFERENCES = {
    "data.output",
    "fixed_kf_resplat3_active_fusion_221_v4",
    "mapping.official_resplat_active_fusion.count_agnostic_forced_commit",
    "mapping.official_resplat_active_fusion.expected_forced_commit_gaussian_count",
    "mapping.official_resplat_active_fusion.force_commit_after_postmerge_rejection",
    "mapping.official_resplat_active_fusion.gate_thresholds_unchanged",
    "mapping.official_resplat_active_fusion.posthoc_after_v2_rejection",
    "mapping.official_resplat_active_fusion.posthoc_after_v3_count_mismatch",
    "mapping.official_resplat_active_fusion.schema",
    "mapping.official_resplat_active_fusion.unsafe_not_deployable",
    "mapping.official_resplat_active_fusion.v2_rejection_audit_sha256",
    "mapping.official_resplat_active_fusion.v3_count_mismatch_audit_sha256",
}

CODE_PROVENANCE_FILES = {
    "experiment_runner_v4": Path(__file__).resolve(),
    "experiment_report_v4": REPO_ROOT
    / "scripts/report_fr2_fixed_kf_resplat3_count_agnostic_forced_commit_221.py",
    "mapper": REPO_ROOT / "src/mapper.py",
    "slam_terminal_writer": REPO_ROOT / "src/slam.py",
    "active_fusion_helper": REPO_ROOT
    / "src/refinement/official_resplat_active_fusion.py",
    "active_map_merge": REPO_ROOT / "src/refinement/active_map_merge.py",
    "world_bridge": REPO_ROOT / "src/refinement/resplat_unblur_bridge.py",
    "sidecar_runner": REPO_ROOT / "scripts/run_official_resplat_sidecar.py",
    "sidecar_verifier": REPO_ROOT / "src/refinement/official_resplat_sidecar.py",
    "gaussian_model": REPO_ROOT
    / "thirdparty/gaussian_splatting/scene/gaussian_model.py",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_binding(root: Path, expected_sha256: str) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"frozen v2 reference root is missing: {root}")
    records = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"frozen reference contains a symlink: {path}")
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
    observed = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"frozen v2 tree digest drifted for {root}: {observed}")
    return {
        "root": str(root),
        "tree_sha256": observed,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "all_regular_file_sha256_values_bound": True,
        "read_only_reference_for_report": True,
        "consumed_by_v4_runtime": False,
    }


def _v3_failure_binding() -> dict[str, Any]:
    """Bind the failed v3 audit as lineage; never expose it to runtime input."""

    tree = _tree_binding(V3_FAILED_ROOT, V3_FAILED_TREE_SHA256)
    launcher_path = V3_FAILED_ROOT / "launcher_runtime.json"
    if _sha256_file(launcher_path) != V3_FAILED_LAUNCHER_SHA256:
        raise ValueError("v3 failure launcher digest drifted")
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    if (
        launcher.get("schema") != "unblur_slam.external_wall_runtime.v1"
        or launcher.get("experiment_revision") != "v3_forced_commit"
        or launcher.get("arm") != "v3_forced_fused"
        or launcher.get("exit_code") != 1
    ):
        raise ValueError("v3 failed-attempt launcher contract drifted")
    if not V3_FAILED_AUDIT.is_file():
        raise FileNotFoundError(f"v3 failure audit is missing: {V3_FAILED_AUDIT}")
    observed_sha = _sha256_file(V3_FAILED_AUDIT)
    if observed_sha != V3_FAILED_AUDIT_SHA256:
        raise ValueError(f"v3 failure audit digest drifted: {observed_sha}")
    audit = json.loads(V3_FAILED_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "forced_commit_candidate_count_mismatch_rolled_back":
        raise ValueError("v3 lineage is not the preregistered count-mismatch rollback")
    merge = audit.get("merge") or {}
    before = audit.get("active_state_before") or {}
    trial = audit.get("active_state_trial") or {}
    final = audit.get("active_state_final") or {}
    decision = ((audit.get("postmerge_gate") or {}).get("decision") or {})
    accepted = int(merge.get("accepted_count", -1))
    before_count = int(before.get("gaussian_count", -1))
    trial_count = int(trial.get("gaussian_count", -1))
    if (
        decision.get("accepted") is not False
        or accepted != 5716
        or not (1024 <= accepted <= 20000)
        or before_count != 72645
        or trial_count != 78361
        or int(merge.get("before_count", -1)) != before_count
        or int(merge.get("after_count", -1)) != trial_count
        or trial_count - before_count != accepted
        or int(final.get("gaussian_count", -1)) != before_count
        or final.get("sha256") != before.get("sha256")
        or final.get("byte_identical_to_premerge") is not True
    ):
        raise ValueError("v3 failed-attempt count/rollback lineage drifted")
    return {
        "root": str(V3_FAILED_ROOT),
        "frozen_tree": tree,
        "frozen_tree_sha256": V3_FAILED_TREE_SHA256,
        "launcher_runtime": {
            "path": str(launcher_path),
            "sha256": V3_FAILED_LAUNCHER_SHA256,
            "exit_code": 1,
        },
        "fusion_audit_path": str(V3_FAILED_AUDIT),
        "fusion_audit_sha256": observed_sha,
        "status": audit["status"],
        "postmerge_gate_accepted": False,
        "accepted_count": accepted,
        "before_count": before_count,
        "trial_count": trial_count,
        "rollback_byte_identical_to_before": True,
        "read_only_lineage_only": True,
        "consumed_by_v4_runtime": False,
        "v3_sidecar_or_active_state_reused": False,
    }


def _resolved_differences(left: Any, right: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                result[path] = {"v2_rollback": None, "v4_count_agnostic": right[key]}
            elif key not in right:
                result[path] = {"v2_rollback": left[key], "v4_count_agnostic": None}
            else:
                result.update(_resolved_differences(left[key], right[key], path))
        return result
    if left != right:
        return {prefix: {"v2_rollback": left, "v4_count_agnostic": right}}
    return {}


def _load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from thirdparty.glorie_slam import config as config_io

    v4 = config_io.load_config(str(CONFIG), str(DEFAULT_CONFIG))
    v2 = V2._load_configs()["fused"]
    return v4, v2


def _validate_v4_config(v4: Mapping[str, Any], v2: Mapping[str, Any]) -> Any:
    differences = _resolved_differences(v2, v4)
    if set(differences) != ALLOWED_V4_V2_CONFIG_DIFFERENCES:
        missing = sorted(ALLOWED_V4_V2_CONFIG_DIFFERENCES - set(differences))
        extra = sorted(set(differences) - ALLOWED_V4_V2_CONFIG_DIFFERENCES)
        raise ValueError(
            f"v4/v2 resolved difference whitelist drifted; missing={missing}, extra={extra}"
        )
    fusion_v2 = ((v2.get("mapping") or {}).get("official_resplat_active_fusion") or {})
    fusion_v4 = ((v4.get("mapping") or {}).get("official_resplat_active_fusion") or {})
    for field in (
        "geometry_gate",
        "sidecar_quality_gate",
        "merge",
        "postmerge_quality_gate",
    ):
        if fusion_v4.get(field) != fusion_v2.get(field):
            raise ValueError(f"v4 changed preregistered {field}")
    forbidden_prefixes = (
        "mapping.official_resplat_active_fusion.geometry_gate.",
        "mapping.official_resplat_active_fusion.sidecar_quality_gate.",
        "mapping.official_resplat_active_fusion.merge.",
        "mapping.official_resplat_active_fusion.postmerge_quality_gate.",
    )
    if any(path.startswith(forbidden_prefixes) for path in differences):
        raise ValueError("v4 difference whitelist reached a gate or merge numeric field")
    if Path(str((v4.get("data") or {}).get("output", ""))).resolve() != OUTPUT:
        raise ValueError("v4 output path drifted")
    disclosure = v4.get("fixed_kf_resplat3_active_fusion_221_v4") or {}
    expected_disclosure = {
        "schema": "unblur_slam.fr2_xyz_fixed_kf_resplat3_count_agnostic_forced_commit_221.v4",
        "experiment_revision": "v4_count_agnostic_forced_commit",
        "posthoc_after_v2_rejection": True,
        "posthoc_after_v3_count_mismatch": True,
        "unsafe_not_deployable": True,
        "gate_thresholds_unchanged": True,
        "merge_filters_unchanged": True,
        "force_only_after_postmerge_gate_rejection": True,
        "cross_run_exact_candidate_count_required": False,
        "fresh_accepted_count_minimum": 1024,
        "fresh_accepted_count_maximum": 20000,
        "fresh_after_minus_before_must_equal_accepted_count": True,
        "v4_reruns_official_resplat": True,
        "v2_resplat_runtime_artifacts_reused": False,
        "v3_resplat_runtime_artifacts_reused": False,
        "single_v4_fused_arm_only": True,
        "cross_revision_references_not_fresh_paired_reruns": True,
        "v2_baseline_resolved_cfg_yaml_sha256": V2_BASELINE_CFG_YAML_SHA256,
        "v2_baseline_canonical_resolved_config_sha256": V2_BASELINE_CANONICAL_CONFIG_SHA256,
        "v2_baseline_frozen_tree_sha256": V2_BASELINE_TREE_SHA256,
        "v2_rollback_fusion_audit_sha256": V2_ROLLBACK_AUDIT_SHA256,
        "v2_rollback_canonical_resolved_config_sha256": V2_ROLLBACK_CANONICAL_CONFIG_SHA256,
        "v2_rollback_frozen_tree_sha256": V2_ROLLBACK_TREE_SHA256,
        "v2_experiment_audit_sha256": V2_EXPERIMENT_AUDIT_SHA256,
        "v3_failed_attempt_fusion_audit_sha256": V3_FAILED_AUDIT_SHA256,
        "v3_failed_attempt_frozen_tree_sha256": V3_FAILED_TREE_SHA256,
        "v3_failed_attempt_launcher_runtime_sha256": V3_FAILED_LAUNCHER_SHA256,
        "v3_failed_attempt_status": "forced_commit_candidate_count_mismatch_rolled_back",
        "v3_failed_attempt_accepted_count": 5716,
        "v3_failed_attempt_before_count": 72645,
        "v3_failed_attempt_trial_count": 78361,
    }
    for name, expected in expected_disclosure.items():
        if disclosure.get(name) != expected:
            raise ValueError(f"v4 disclosure {name} drifted")
    if Path(str(disclosure.get("v2_baseline_reference_root", ""))).resolve() != V2_BASELINE_ROOT:
        raise ValueError("v2 baseline reference path drifted")
    if Path(str(disclosure.get("v2_rollback_reference_root", ""))).resolve() != V2_ROLLBACK_ROOT:
        raise ValueError("v2 rollback reference path drifted")
    if Path(str(disclosure.get("v3_failed_attempt_root", ""))).resolve() != V3_FAILED_ROOT:
        raise ValueError("v3 failed-attempt reference path drifted")

    from src.refinement.official_resplat_active_fusion import ActiveFusionConfig

    fusion = ActiveFusionConfig.from_dict(
        fusion_v4,
        default_output_root=OUTPUT / "freiburg2_xyz/official_resplat_active_fusion",
    )
    if (
        fusion.enabled is not True
        or fusion.posthoc_after_v2_rejection is not True
        or fusion.unsafe_not_deployable is not True
        or fusion.gate_thresholds_unchanged is not True
        or fusion.force_commit_after_postmerge_rejection is not True
        or fusion.posthoc_after_v3_count_mismatch is not True
        or fusion.count_agnostic_forced_commit is not True
        or int(fusion.expected_forced_commit_gaussian_count) != 0
        or fusion.v2_rejection_audit_sha256 != V2_ROLLBACK_AUDIT_SHA256
        or fusion.v3_count_mismatch_audit_sha256 != V3_FAILED_AUDIT_SHA256
        or int(fusion.merge["min_new_gaussians"]) != 1024
        or int(fusion.merge["max_new_gaussians"]) != 20000
    ):
        raise ValueError("v4 forced-commit ActiveFusionConfig drifted")
    return fusion, differences


def validate_forced_commit_instrumentation(
    mapper_source: Optional[str] = None, slam_source: Optional[str] = None
) -> dict[str, Any]:
    """AST/source-order guard for source220, final100, and terminal hooks."""

    mapper_path = REPO_ROOT / "src/mapper.py"
    slam_path = REPO_ROOT / "src/slam.py"
    mapper_text = mapper_path.read_text(encoding="utf-8") if mapper_source is None else str(mapper_source)
    slam_text = slam_path.read_text(encoding="utf-8") if slam_source is None else str(slam_source)
    mapper_tree = ast.parse(mapper_text, filename=str(mapper_path))
    mapper_class = next(
        node for node in mapper_tree.body if isinstance(node, ast.ClassDef) and node.name == "Mapper"
    )
    methods = {
        node.name: node
        for node in mapper_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    run = methods.get("run")
    final_refine = methods.get("final_refine")
    hook = methods.get("_active_fusion_after_mapped_keyframe")
    if run is None or final_refine is None or hook is None:
        raise ValueError("required Mapper forced-commit methods are missing")

    def calls(method: ast.AST, name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        ]

    entry_calls = calls(run, "_forced_commit_source220_entry")
    metadata_calls = calls(run, "frame_info")
    recv_calls = calls(run, "recv")
    end_assignments = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "is_finished"
            for target in node.targets
        )
    ]
    end_guards = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "is_finished"
        and any(isinstance(child, ast.Break) for child in ast.walk(node))
    ]
    if len(entry_calls) != 1 or len(metadata_calls) < 1 or len(recv_calls) < 1:
        raise ValueError("source220 entry hook cardinality/order anchors drifted")
    if len(end_assignments) != 1 or len(end_guards) != 1:
        raise ValueError("mapper end-of-stream assignment/guard anchors drifted")
    entry = entry_calls[0]
    if not (
        min(node.lineno for node in recv_calls)
        < end_assignments[0].lineno
        < end_guards[0].lineno
        < entry.lineno
        < min(node.lineno for node in metadata_calls)
    ):
        raise ValueError(
            "source220 entry hook must follow the end guard and precede frame metadata/preparation"
        )
    if [ast.unparse(argument) for argument in entry.args] != [
        "int(idx)",
        "int(video_idx)",
    ]:
        raise ValueError("source220 entry hook arguments drifted")
    complete_calls = calls(hook, "_forced_commit_source220_complete")
    if len(complete_calls) != 1 or [
        ast.unparse(argument) for argument in complete_calls[0].args
    ] != ["viewpoint"]:
        raise ValueError("source220 completion hook drifted")
    final_entry = calls(final_refine, "_forced_commit_final100_entry")
    hydration = calls(final_refine, "_hydrate_missing_droid_keyframes_for_final_refine")
    final_complete = calls(final_refine, "_forced_commit_final100_complete")
    if len(final_entry) != 1 or len(hydration) != 1 or len(final_complete) != 1:
        raise ValueError("final100 hook cardinality drifted")
    if not final_entry[0].lineno < hydration[0].lineno < final_complete[0].lineno:
        raise ValueError("final100 entry/complete ordering drifted")
    required_mapper_fragments = (
        '"sequence": 0',
        '"sequence": 1',
        '"sequence": 2',
        '"trial_equals_committed": True',
        '"individual_imported_gaussian_survival_in_final_model_not_claimed": True',
    )
    if any(fragment not in mapper_text for fragment in required_mapper_fragments):
        raise ValueError("forced-commit chain disclosure fragment drifted")
    slam_tree = ast.parse(slam_text, filename=str(slam_path))
    save_lines = [
        node.lineno
        for node in ast.walk(slam_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save_ply"
    ]
    terminal_lines = [
        node.lineno
        for node in ast.walk(slam_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "finalize_forced_commit_terminal"
    ]
    if len(terminal_lines) != 1 or not any(line < terminal_lines[0] for line in save_lines):
        raise ValueError("final PLY/terminal publication ordering drifted")
    return {
        "schema": "unblur_slam.forced_commit_instrumentation_binding.v1",
        "accepted": True,
        "source220_entry_before_frame_metadata_pose_depth_deformation_mapping": True,
        "source220_completion_after_regular_mapping_and_prune_hook": True,
        "final100_entry_before_hydration_pose_depth_deformation": True,
        "final100_completion_and_terminal_ply_binding_present": True,
        "mapper_sha256": hashlib.sha256(mapper_text.encode("utf-8")).hexdigest(),
        "slam_sha256": hashlib.sha256(slam_text.encode("utf-8")).hexdigest(),
    }


def _implementation_provenance() -> dict[str, Any]:
    files = {}
    for label, path in CODE_PROVENANCE_FILES.items():
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"implementation file missing: {resolved}")
        files[label] = {"path": str(resolved), "sha256": _sha256_file(resolved)}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT, check=True, text=True, capture_output=True,
    ).stdout.splitlines()
    return {
        "schema": "unblur_slam.implementation_provenance.v1",
        "git_head": head,
        "git_worktree_clean": not status,
        "git_dirty_entry_count": len(status),
        "git_dirty_disclosure": "dirty_or_untracked_files_present" if status else "clean",
        "files": files,
    }


def _validate_frozen_v2_references(v4: Mapping[str, Any]) -> dict[str, Any]:
    baseline = _tree_binding(V2_BASELINE_ROOT, V2_BASELINE_TREE_SHA256)
    rollback = _tree_binding(V2_ROLLBACK_ROOT, V2_ROLLBACK_TREE_SHA256)
    if _sha256_file(V2_EXPERIMENT_AUDIT) != V2_EXPERIMENT_AUDIT_SHA256:
        raise ValueError("v2 experiment report/audit digest drifted")
    if _sha256_file(V2_ROLLBACK_AUDIT) != V2_ROLLBACK_AUDIT_SHA256:
        raise ValueError("v2 rollback fusion audit digest drifted")
    rollback_audit = json.loads(V2_ROLLBACK_AUDIT.read_text(encoding="utf-8"))
    decision = ((rollback_audit.get("postmerge_gate") or {}).get("decision") or {})
    merge = rollback_audit.get("merge") or {}
    if (
        rollback_audit.get("status") != "postmerge_gate_rejected_rolled_back"
        or decision.get("accepted") is not False
        or int(merge.get("accepted_count", -1)) != 5554
        or int(merge.get("before_count", -1)) != 72208
        or int(merge.get("after_count", -1)) != 77762
    ):
        raise ValueError("v2 rollback-arm scientific reference drifted")
    expected_reasons = {
        "relative_mean_composite_increase_exceeded",
        "per_view_composite_increase_exceeded",
    }
    if set(decision.get("reasons") or ()) != expected_reasons:
        raise ValueError("v2 post-gate rejection reasons drifted")
    baseline_cfg = V2_BASELINE_ROOT / "freiburg2_xyz/cfg.yaml"
    if _sha256_file(baseline_cfg) != V2_BASELINE_CFG_YAML_SHA256:
        raise ValueError("v2 baseline resolved cfg.yaml digest drifted")
    baseline_preflight = json.loads(
        (V2_BASELINE_ROOT / "preflight.json").read_text(encoding="utf-8")
    )
    paired = baseline_preflight.get("paired_contract") or {}
    if (
        paired.get("baseline_resolved_sha256") != V2_BASELINE_CANONICAL_CONFIG_SHA256
        or paired.get("fused_resolved_sha256") != V2_ROLLBACK_CANONICAL_CONFIG_SHA256
    ):
        raise ValueError("v2 canonical resolved-config digests drifted")
    return {
        "schema": "unblur_slam.v2_frozen_dual_reference.v1",
        "baseline": baseline,
        "rollback_arm": rollback,
        "v2_experiment_audit": {
            "path": str(V2_EXPERIMENT_AUDIT),
            "sha256": V2_EXPERIMENT_AUDIT_SHA256,
        },
        "v2_rollback_fusion_audit": {
            "path": str(V2_ROLLBACK_AUDIT),
            "sha256": V2_ROLLBACK_AUDIT_SHA256,
            "status": rollback_audit["status"],
            "accepted_count": 5554,
            "before_count": 72208,
            "trial_count": 77762,
            "postmerge_gate_accepted": False,
            "rejection_reasons": sorted(expected_reasons),
        },
        "reference_semantics": {
            "cross_revision": True,
            "fresh_pair_rerun": False,
            "baseline_numeric_path_unchanged_and_read_only": True,
            "rollback_arm_numeric_path_unchanged_and_read_only": True,
            "used_only_for_posthoc_diagnostic_comparison": True,
            "consumed_by_v4_slam_or_resplat_runtime": False,
        },
    }


def preflight(*, check_output_available: bool = True) -> dict[str, Any]:
    v4, v2 = _load_configs()
    fusion, differences = _validate_v4_config(v4, v2)
    if check_output_available and (OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink()):
        raise FileExistsError(f"refusing to overwrite v4 output root: {OUTPUT_ROOT}")
    # Reuse the already-audited CPU dataset/checkpoint/repository/real-artifact
    # checks, but never consume any v2 runtime sidecar output.
    v2_cpu_gate = V2.preflight(arms=("fused",), check_output_available=False)
    frozen = _validate_frozen_v2_references(v4)
    v3_failure = _v3_failure_binding()
    instrumentation = validate_forced_commit_instrumentation()
    implementation = _implementation_provenance()
    if instrumentation["mapper_sha256"] != implementation["files"]["mapper"]["sha256"]:
        raise ValueError("forced instrumentation is not bound to mapper provenance")
    if instrumentation["slam_sha256"] != implementation["files"]["slam_terminal_writer"]["sha256"]:
        raise ValueError("forced terminal hook is not bound to slam provenance")
    sidecar_root = Path(fusion.output_root).expanduser().resolve()
    if V2_ROOT == OUTPUT_ROOT or V2_ROOT in sidecar_root.parents:
        raise ValueError("v4 sidecar output would overlap a v2 runtime root")
    if V3_FAILED_ROOT == OUTPUT_ROOT or V3_FAILED_ROOT in sidecar_root.parents:
        raise ValueError("v4 sidecar output would overlap the failed v3 runtime root")
    return {
        "schema": "unblur_slam.fr2_xyz_resplat3_count_agnostic_forced_commit_221_preflight.v4",
        "scope": {
            "experiment_revision": "v4_count_agnostic_forced_commit",
            "posthoc_after_v2_rejection": True,
            "posthoc_after_v3_count_mismatch": True,
            "unsafe_not_deployable": True,
            "single_new_fused_arm_only": True,
            "cross_revision_v2_references": True,
            "cross_revision_v3_failed_attempt_lineage": True,
            "fresh_paired_comparison": False,
            "fixed_source_keyframes": list(EXPECTED_FIXED_SOURCE_KEYFRAMES),
            "trigger_context": list(EXPECTED_CONTEXT),
            "trigger_source_index": 166,
            "downstream_source_indices": list(EXPECTED_DOWNSTREAM),
            "final_refine_iterations": 100,
        },
        "scientific_contract": {
            "v4_reruns_official_resplat": True,
            "v2_resplat_runtime_artifacts_reused": False,
            "v3_resplat_runtime_artifacts_reused": False,
            "force_only_after_complete_postmerge_gate_rejection": True,
            "cross_run_exact_candidate_count_required": False,
            "fresh_accepted_count_minimum": 1024,
            "fresh_accepted_count_maximum": 20000,
            "fresh_after_minus_before_must_equal_accepted_count": True,
            "fresh_candidate_identity_equal_to_v2_or_v3_not_claimed": True,
            "gate_thresholds_unchanged": True,
            "merge_filters_unchanged": True,
            "geometry_gate_unchanged": True,
            "sidecar_quality_gate_unchanged": True,
            "postmerge_quality_gate_unchanged": True,
            "merge": dict(fusion.merge),
            "geometry_gate": dict(fusion.geometry_gate),
            "sidecar_quality_gate": dict(fusion.sidecar_quality_gate),
            "postmerge_quality_gate": dict(fusion.postmerge_quality_gate),
            "allowed_v4_minus_v2_rollback_resolved_config_differences": differences,
        },
        "forced_commit_chain_contract": {
            "instrumentation": instrumentation,
            "commit_requires_trial_full_state_equal_committed_and_different_from_before": True,
            "source220_entry_before_any_source220_map_mutation": True,
            "source220_requires_100_mapping_iterations_plus_one_prune_pass": True,
            "final100_requires_exactly_100_iterations": True,
            "final_model_ply_must_equal_iter100_checkpoint_bytes": True,
            "individual_imported_gaussian_survival_not_tracked_or_claimed": True,
            "uses_gt_for_forced_commit_decision": False,
            "clear_gt_metrics_bound_posthoc_for_evaluation": True,
            "clear_gt_values_used_for_commit_or_checkpoint_selection": False,
        },
        "frozen_v2_references": frozen,
        "frozen_v3_failed_attempt_lineage": v3_failure,
        "v2_cpu_preflight_reused_as_read_only_validation": {
            "schema": v2_cpu_gate["schema"],
            "artifacts": v2_cpu_gate["artifacts"],
            "fusion_contract": v2_cpu_gate["fusion_contract"],
        },
        "implementation_provenance": implementation,
        "execution": {
            "selected_arms": ["v4_count_agnostic_fused"],
            "output_root": str(OUTPUT_ROOT),
            "arm_output": str(OUTPUT),
            "new_sidecar_output_root": str(sidecar_root),
            "sequential_physical_gpu": int(PHYSICAL_GPU),
            "process_device": "cuda:0",
            "online_timer_includes_commit_and_source220_chain_overhead": True,
            "total_timer_includes_commit_source220_and_final100_chain_overhead": True,
            "terminal_validation_publication_after_internal_total_timer": True,
            "external_wall_timer_includes_terminal_validation_publication": True,
            "recorded_internal_diagnostic_overhead_not_subtracted": True,
            "v4_canonical_resolved_config_sha256": BASE._canonical_sha256(v4),
        },
    }


def _interrupt(process: subprocess.Popen[str], log: Any) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
    try:
        return int(process.wait(timeout=30))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        return int(process.wait())


def run() -> int:
    audit = preflight(check_output_available=True)
    V2.FIXED._assert_physical_gpu_free()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "preflight.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    command = [sys.executable, str(REPO_ROOT / "run.py"), str(CONFIG)]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": PHYSICAL_GPU,
            "PYTHONUNBUFFERED": "1",
            "UNBLUR_SKIP_NR_IQA": "1",
        }
    )
    started = time.monotonic()
    code = -1
    monitor = V2._WholeDeviceGpuMonitor(OUTPUT / "gpu_monitor.csv")
    with (OUTPUT / "launch.log").open("x", encoding="utf-8", buffering=1) as log:
        log.write("[launcher] revision=v4_count_agnostic_forced_commit arm=fused_only unsafe_not_deployable=true\n")
        log.write("[launcher] posthoc_after_v2_rejection=true posthoc_after_v3_count_mismatch=true gate_thresholds_unchanged=true merge_filters_unchanged=true\n")
        log.write("[launcher] official_resplat_rerun=true v2_v3_runtime_artifact_reuse=false exact_cross_run_count_required=false accepted_count_bounds=[1024,20000] internal_count_algebra_required=true\n")
        log.write("[launcher] source166_commit source220=100+prune final=100 internal_chain_overhead_in_internal_timers=true terminal_audit_in_external_wall_only=true\n")
        monitor.start()
        process: Optional[subprocess.Popen[str]] = None
        try:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            code = int(process.wait())
        except KeyboardInterrupt:
            code = 130 if process is None else _interrupt(process, log)
        finally:
            gpu = monitor.stop()
            (OUTPUT / "gpu_monitor_summary.json").write_text(
                json.dumps(gpu, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        log.write(f"[launcher] exit_code={code}\n")
    (OUTPUT / "launcher_runtime.json").write_text(
        json.dumps(
            {
                "schema": "unblur_slam.external_wall_runtime.v1",
                "experiment_revision": "v4_count_agnostic_forced_commit",
                "arm": "v4_count_agnostic_fused",
                "wall_runtime_seconds": time.monotonic() - started,
                "exit_code": code,
                "physical_gpu": int(PHYSICAL_GPU),
                "process_device": "cuda:0",
                "posthoc_after_v2_rejection": True,
                "posthoc_after_v3_count_mismatch": True,
                "cross_run_exact_candidate_count_required": False,
                "unsafe_not_deployable": True,
                "whole_device_gpu_monitor_summary": str(
                    OUTPUT / "gpu_monitor_summary.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return code


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true", help="CPU-only validation (default)")
    action.add_argument("--run", action="store_true", help="launch the one unsafe v4 fused arm")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.run:
            return run()
        print(json.dumps(preflight(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
