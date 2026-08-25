#!/usr/bin/env python3
"""CPU-preflight or run the isolated v6 virtual-count bugfix diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/unblur_slam.yaml"
V5_RUNNER_PATH = (
    REPO_ROOT
    / "scripts/run_fr2_fixed_kf_resplat3_visibility_cache_refresh_forced_commit_221.py"
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V5 = _load_module("virtual_count_bugfix_v5_runner_contract", V5_RUNNER_PATH)
PHYSICAL_GPU = V5.PHYSICAL_GPU
EXPECTED_FIXED_SOURCE_KEYFRAMES = tuple(V5.EXPECTED_FIXED_SOURCE_KEYFRAMES)
EXPECTED_CONTEXT = tuple(V5.EXPECTED_CONTEXT)
EXPECTED_DOWNSTREAM = tuple(V5.EXPECTED_DOWNSTREAM)

V5_FAILED_ROOT = V5.OUTPUT.resolve()
V5_FAILED_AUDIT = (
    V5_FAILED_ROOT
    / "freiburg2_xyz/official_resplat_active_fusion/fusion_audit.json"
).resolve()
V5_FAILED_TREE_SHA256 = "3c26d3ba67431091cee2c8865e11cee49f7614bd100279512b30a589f9bb107b"
V5_FAILED_AUDIT_SHA256 = "66a76384988c033d1bbf3d801c5a180df1a624c31509b916175b9ad39a2ddc1a"
V5_FAILED_LAUNCHER_SHA256 = "862aeee20e876327b276b5be33832dd7b2040fe0fbfa2cc80640866e99f9dac2"
V5_FAILED_LAUNCH_LOG_SHA256 = "1b0b31b3404ca64dae2fe9a263ceac2fd7ddd3db24c4dfdd99f418813dd50db5"
V5_FAILED_PREFLIGHT_SHA256 = "fa4274fd5b03e5ffec3f5761bfa2caaaf6f3dee9e66248a2cc28e4fb7f3adc26"
V5_OLD_MAPPER_SHA256 = "18954a694a113f527918070ade4e28226d1760b5e84787f0134220961c390ec9"
V5_FAILURE_SIGNATURE = "NameError: name 'virtual_count' is not defined"

CONFIG = (
    REPO_ROOT
    / "configs/local/fr2_xyz_fixed_kf_resplat3_fusion_221_v6_virtual_count_bugfix"
    / "evssm_resplat3_virtual_count_bugfix.yaml"
)
V5_CONFIG = V5.CONFIG
OUTPUT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_paired/"
    "fr2_xyz_fixed_kf_resplat3_fusion_221_v6_virtual_count_bugfix"
).resolve()
OUTPUT = (OUTPUT_ROOT / "evssm_resplat3_virtual_count_bugfix").resolve()

ALLOWED_V6_V5_CONFIG_DIFFERENCES = {
    "data.output",
    "fixed_kf_resplat3_active_fusion_221_v6",
}
APPROVED_BUGFIX_BLOCK = (
    "                    virtual_count = int(viewpoint.n_virtual_cams)\n"
    "                    if virtual_count <= 0:\n"
    "                        raise RuntimeError(\n"
    "                            \"deblur-fail context requires at least one virtual camera\"\n"
    "                        )\n"
)


def _sha256_file(path: Path) -> str:
    return V5._sha256_file(path)


def _load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from thirdparty.glorie_slam import config as config_io

    return (
        config_io.load_config(str(CONFIG), str(DEFAULT_CONFIG)),
        config_io.load_config(str(V5_CONFIG), str(DEFAULT_CONFIG)),
    )


def _validate_direct_v5_inheritance() -> dict[str, Any]:
    common = CONFIG.parent / "common.yaml"
    raw = yaml.safe_load(common.read_text(encoding="utf-8")) or {}
    expected = (
        "../fr2_xyz_fixed_kf_resplat3_fusion_221_v5_visibility_cache_refresh_"
        "forced_commit/common.yaml"
    )
    if not isinstance(raw, Mapping) or raw.get("inherit_from") != expected:
        raise ValueError("v6 common must directly inherit the frozen v5 common")
    parent = (common.parent / expected).resolve()
    expected_parent = (V5_CONFIG.parent / "common.yaml").resolve()
    if parent != expected_parent or not parent.is_file():
        raise ValueError("v6 direct-v5 inheritance path escaped or disappeared")
    return {
        "schema": "unblur_slam.direct_config_inheritance_binding.v1",
        "child_common": str(common.resolve()),
        "inherit_from_lexical": expected,
        "resolved_parent_common": str(parent),
        "directly_inherits_v5_common": True,
        "v5_common_sha256": _sha256_file(parent),
    }


def _validate_config(v6: Mapping[str, Any], v5: Mapping[str, Any]) -> dict[str, Any]:
    differences = V5._resolved_differences(v5, v6, left_label="v5", right_label="v6")
    if set(differences) != ALLOWED_V6_V5_CONFIG_DIFFERENCES:
        raise ValueError(
            "v6/v5 resolved difference whitelist drifted: "
            f"{sorted(differences)}"
        )
    if Path(str((v6.get("data") or {}).get("output", ""))).resolve() != OUTPUT:
        raise ValueError("v6 output path drifted")
    if (v6.get("mapping") or {}) != (v5.get("mapping") or {}):
        raise ValueError("v6 changed mapping/model/gate/merge configuration")
    disclosure = v6.get("fixed_kf_resplat3_active_fusion_221_v6") or {}
    expected = {
        "schema": "unblur_slam.fr2_xyz_fixed_kf_resplat3_virtual_count_bugfix_221.v6",
        "experiment_revision": "v6_virtual_count_bugfix",
        "direct_v5_inheritance": True,
        "only_code_change_from_v5": "bind_positive_virtual_count_before_context_metric_virtual_extrinsics_validation",
        "all_scientific_values_unchanged": True,
        "gate_thresholds_unchanged": True,
        "merge_filters_unchanged": True,
        "count_contract_unchanged": True,
        "cache_refresh_contract_unchanged": True,
        "v6_reruns_official_resplat_fresh": True,
        "v5_runtime_artifacts_reused": False,
        "v5_failed_attempt_root": str(V5_FAILED_ROOT),
        "v5_failed_attempt_frozen_tree_sha256": V5_FAILED_TREE_SHA256,
        "v5_failed_attempt_fusion_audit_sha256": V5_FAILED_AUDIT_SHA256,
        "v5_failed_attempt_launcher_runtime_sha256": V5_FAILED_LAUNCHER_SHA256,
        "v5_failed_attempt_launch_log_sha256": V5_FAILED_LAUNCH_LOG_SHA256,
        "v5_failed_attempt_status": "error_rejected",
        "v5_failure_signature": V5_FAILURE_SIGNATURE,
        "v5_active_map_changed_final": False,
        "v5_merge_started": False,
    }
    if disclosure != expected:
        raise ValueError("v6 disclosure/failed-v5 lineage drifted")
    return differences


def _validate_bugfix() -> dict[str, Any]:
    mapper = REPO_ROOT / "src/mapper.py"
    source = mapper.read_text(encoding="utf-8")
    if source.count(APPROVED_BUGFIX_BLOCK) != 1:
        raise ValueError("approved virtual-count bugfix block must occur exactly once")
    anchor = "                    R, t, theta, rho = viewpoint.get_virtual_extrinsics()\n"
    if source.count(anchor) < 1 or source.index(APPROVED_BUGFIX_BLOCK) > source.index(anchor):
        raise ValueError("virtual-count binding must precede virtual extrinsics")
    reverted = source.replace(APPROVED_BUGFIX_BLOCK, "", 1).encode("utf-8")
    reverted_sha = hashlib.sha256(reverted).hexdigest()
    if reverted_sha != V5_OLD_MAPPER_SHA256:
        raise ValueError("mapper contains changes beyond the approved v6 bugfix block")
    block_sha = hashlib.sha256(APPROVED_BUGFIX_BLOCK.encode("utf-8")).hexdigest()
    return {
        "schema": "unblur_slam.virtual_count_context_metric_bugfix.v1",
        "assignment_added": True,
        "positive_count_guard_added": True,
        "approved_block_occurrences": 1,
        "approved_block_sha256": block_sha,
        "v5_mapper_sha256_after_reversion": reverted_sha,
        "v6_mapper_sha256": _sha256_file(mapper),
        "v5_mapper_size_bytes": len(reverted),
        "v6_mapper_size_bytes": mapper.stat().st_size,
        "render_and_all_numeric_contracts_unchanged": True,
    }


def _v5_failure_binding() -> dict[str, Any]:
    tree = V5._tree_binding(V5_FAILED_ROOT, V5_FAILED_TREE_SHA256)
    audit_path = V5_FAILED_AUDIT
    launcher_path = V5_FAILED_ROOT / "launcher_runtime.json"
    log_path = V5_FAILED_ROOT / "launch.log"
    preflight_path = V5_FAILED_ROOT / "preflight.json"
    for path, expected, label in (
        (audit_path, V5_FAILED_AUDIT_SHA256, "audit"),
        (launcher_path, V5_FAILED_LAUNCHER_SHA256, "launcher"),
        (log_path, V5_FAILED_LAUNCH_LOG_SHA256, "launch log"),
        (preflight_path, V5_FAILED_PREFLIGHT_SHA256, "preflight"),
    ):
        if _sha256_file(path) != expected:
            raise ValueError(f"frozen v5 {label} SHA drifted")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    old_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "error_rejected"
        or audit.get("active_map_changed_final") is not False
        or audit.get("rejection_reasons") != [V5_FAILURE_SIGNATURE]
    ):
        raise ValueError("frozen v5 failure audit semantics drifted")
    gates = audit.get("premerge_gates") or {}
    if not all((gates.get(name) or {}).get("accepted") is True for name in (
        "world_artifact", "context_reconstruction", "repository_provenance", "data_lineage"
    )):
        raise ValueError("v5 did not pass every premerge gate before the code error")
    timing = audit.get("timing") or {}
    for field in (
        "merge_seconds", "premerge_active_render_seconds",
        "postmerge_active_render_seconds", "rollback_seconds",
        "visibility_cache_refresh_seconds",
    ):
        if float(timing.get(field, -1.0)) != 0.0:
            raise ValueError(f"v5 unexpectedly reached {field}")
    for field in (
        "merge", "active_state_before", "active_state_trial", "postmerge_gate",
        "visibility_cache_refresh", "unsafe_forced_commit",
    ):
        if field in audit:
            raise ValueError(f"v5 unexpectedly published post-premerge field {field}")
    if (V5_FAILED_AUDIT.parent / "forced_commit_chain").exists():
        raise ValueError("v5 unexpectedly published a forced-commit chain")
    for path in (
        V5_FAILED_AUDIT.parent / "fusion_final_contract.json",
        V5_FAILED_ROOT / "freiburg2_xyz/runtime_stats.json",
        V5_FAILED_ROOT / "freiburg2_xyz/final_model.ply",
    ):
        if path.exists() or path.is_symlink():
            raise ValueError(f"v5 unexpectedly published terminal artifact {path.name}")
    if (
        launcher.get("schema") != "unblur_slam.external_wall_runtime.v1"
        or launcher.get("experiment_revision") != "v5_visibility_cache_refresh_forced_commit"
        or launcher.get("arm") != "v5_visibility_cache_refresh_fused"
        or launcher.get("exit_code") != 1
    ):
        raise ValueError("frozen v5 launcher failure semantics drifted")
    log = log_path.read_text(encoding="utf-8")
    if (
        log.count("forced diagnostic did not reproduce the complete post-gate rejection")
        != 1
        or "_initialize_forced_commit_chain" not in log
    ):
        raise ValueError("v5 launch log no longer proves fail-closed worker termination")
    old_mapper = (((old_preflight.get("implementation_provenance") or {}).get("files") or {}).get("mapper") or {})
    if old_mapper.get("sha256") != V5_OLD_MAPPER_SHA256:
        raise ValueError("v5 preflight old mapper binding drifted")
    return {
        "schema": "unblur_slam.v5_virtual_count_failure_lineage.v1",
        "root": str(V5_FAILED_ROOT),
        "frozen_tree": tree,
        "audit_sha256": V5_FAILED_AUDIT_SHA256,
        "launcher_runtime_sha256": V5_FAILED_LAUNCHER_SHA256,
        "launch_log_sha256": V5_FAILED_LAUNCH_LOG_SHA256,
        "preflight_sha256": V5_FAILED_PREFLIGHT_SHA256,
        "status": "error_rejected",
        "exit_code": 1,
        "reason": V5_FAILURE_SIGNATURE,
        "premerge_gates_all_accepted": True,
        "merge_started": False,
        "active_map_changed": False,
        "runtime_artifacts_reused_by_v6": False,
    }


def _implementation_provenance() -> dict[str, Any]:
    provenance = V5._implementation_provenance()
    files = dict(V5.CODE_PROVENANCE_FILES)
    files.pop("experiment_runner_v5", None)
    files.pop("experiment_report_v5", None)
    files.update({
        "experiment_runner_v6": Path(__file__).resolve(),
        "experiment_report_v6": REPO_ROOT / "scripts/report_fr2_fixed_kf_resplat3_virtual_count_bugfix_221.py",
    })
    provenance["files"] = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in sorted(files.items())
    }
    return provenance


def preflight(*, check_output_available: bool = True) -> dict[str, Any]:
    v6, v5 = _load_configs()
    differences = _validate_config(v6, v5)
    direct = _validate_direct_v5_inheritance()
    bugfix = _validate_bugfix()
    failure = _v5_failure_binding()
    if check_output_available and (OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink()):
        raise FileExistsError(f"refusing to overwrite v6 output root: {OUTPUT_ROOT}")
    if OUTPUT_ROOT == V5.OUTPUT_ROOT or OUTPUT_ROOT in V5_FAILED_ROOT.parents:
        raise ValueError("v6 output overlaps frozen v5 failure")
    v5_validation = V5.preflight(check_output_available=False)
    return {
        "schema": "unblur_slam.fr2_xyz_resplat3_virtual_count_bugfix_221_preflight.v6",
        "scope": {
            "single_fresh_v6_fused_arm_only": True,
            "unsafe_not_deployable": True,
            "fresh_paired_comparison": False,
            "v5_runtime_artifacts_reused": False,
        },
        "scientific_contract": {
            "direct_v5_inheritance": direct,
            "only_code_change_from_v5": bugfix,
            "allowed_v6_v5_resolved_differences": differences,
            "mapping_model_schedule_gates_merge_count_and_cache_contract_exactly_equal": True,
            "official_resplat_fresh_rerun": True,
        },
        "frozen_v5_failure_lineage": failure,
        "implementation_provenance": _implementation_provenance(),
        "execution": {
            "selected_arms": ["v6_virtual_count_bugfix_fused"],
            "physical_gpu": int(PHYSICAL_GPU),
            "process_visible_device": "cuda:0",
            "output_root": str(OUTPUT_ROOT),
            "config": str(CONFIG),
        },
        "v5_cpu_preflight_reused_as_read_only_validation": v5_validation,
    }


def run() -> int:
    audit = preflight(check_output_available=True)
    V5.V2.FIXED._assert_physical_gpu_free()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "preflight.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    command = [sys.executable, str(REPO_ROOT / "run.py"), str(CONFIG)]
    environment = os.environ.copy()
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": PHYSICAL_GPU,
        "PYTHONUNBUFFERED": "1",
        "UNBLUR_SKIP_NR_IQA": "1",
    })
    started = time.monotonic()
    code = -1
    monitor = V5.V2._WholeDeviceGpuMonitor(OUTPUT / "gpu_monitor.csv")
    with (OUTPUT / "launch.log").open("x", encoding="utf-8", buffering=1) as log:
        log.write("[launcher] revision=v6_virtual_count_bugfix fresh_resplat=true v5_artifact_reuse=false unsafe=true\n")
        log.write("[launcher] only_code_change=bind_virtual_count_and_positive_guard all_scientific_values_unchanged=true\n")
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
            code = 130 if process is None else V5._interrupt(process, log)
        finally:
            gpu = monitor.stop()
            (OUTPUT / "gpu_monitor_summary.json").write_text(
                json.dumps(gpu, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        log.write(f"[launcher] exit_code={code}\n")
    (OUTPUT / "launcher_runtime.json").write_text(
        json.dumps({
            "schema": "unblur_slam.external_wall_runtime.v1",
            "experiment_revision": "v6_virtual_count_bugfix",
            "arm": "v6_virtual_count_bugfix_fused",
            "wall_runtime_seconds": time.monotonic() - started,
            "exit_code": code,
            "physical_gpu": int(PHYSICAL_GPU),
            "process_device": "cuda:0",
            "unsafe_not_deployable": True,
            "v5_runtime_artifacts_reused": False,
            "only_code_change_from_v5": "bind_positive_virtual_count_before_context_metric_virtual_extrinsics_validation",
            "whole_device_gpu_monitor_summary": str(OUTPUT / "gpu_monitor_summary.json"),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return code


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
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
