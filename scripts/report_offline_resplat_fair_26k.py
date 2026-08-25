#!/usr/bin/env python3
"""Build the atomic three-scene quality/runtime report for offline ReSplat v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from execute_offline_resplat_plan import validate_plan as validate_execution_plan


CONTRACT_SCHEMA = "unblur_slam.offline_resplat_fair_26k.v1"
MEASUREMENT_SCHEMA = "unblur_slam.offline_fair_26k_measurement.v1"
MILESTONE_SCHEMA = "unblur_slam.offline_fair_milestone.v1"
PLAN_SCHEMA = "unblur_slam.offline_resplat_multisubmap_plan.v1"
EXECUTION_SCHEMA = "unblur_slam.offline_resplat_execution.v1"
COMMON_SCHEMA = "unblur_slam.offline_fair_common_rgb_metrics.v1"
RUNNER_SCHEMA = "unblur_slam.paired_official_resplat_smoke.v1"
PINNED_GPU_COMMAND_SCHEMA = "unblur_slam.pinned_gpu_command.v1"
REPORT_SCHEMA = "unblur_slam.offline_resplat_fair_26k_final.v1"
SCENE_SCHEMA = "unblur_slam.official_resplat_colmap_scene.v1"
MILESTONES = (8000, 12000, 26000)
ARMS = ("U8", "U12", "U26", "R4-multisubmap")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _git(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot inspect frozen repository {repository}") from error
    return result.stdout.strip()


def _validate_static_provenance(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck the launch-time code/checkpoint identities before authorizing claims."""

    baseline = contract["unblur_baseline"]
    worktree = Path(str(baseline["worktree"])).resolve()
    head = _git(worktree, "rev-parse", "HEAD")
    if head != baseline["commit"]:
        raise ValueError("pristine-Unblur worktree commit changed after preregistration")
    status_lines = _git(
        worktree, "status", "--porcelain", "--untracked-files=all"
    ).splitlines()
    observed_paths = {line[2:].strip() for line in status_lines if len(line) >= 3}
    allowed = baseline["measurement_hook"]["allowed_worktree_paths"]
    if observed_paths != set(allowed):
        raise ValueError("pristine-Unblur worktree no longer has only the frozen hook diff")
    measurement_hashes: dict[str, str] = {}
    for relative, expected in allowed.items():
        path = worktree / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"pristine-Unblur measurement hook changed: {relative}")
        measurement_hashes[relative] = expected

    official = contract["official_resplat"]
    resplat_repo = Path(str(official["repository"])).resolve()
    if _git(resplat_repo, "rev-parse", "HEAD") != official["commit"]:
        raise ValueError("official ReSplat commit changed after preregistration")
    if _git(resplat_repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("official ReSplat tracked worktree is dirty")
    checkpoint = Path(str(official["checkpoint"])).resolve()
    runner = Path(str(official["runner"])).resolve()
    if sha256_file(checkpoint) != official["checkpoint_sha256"]:
        raise ValueError("official ReSplat checkpoint changed")
    if sha256_file(runner) != official["runner_sha256"]:
        raise ValueError("official ReSplat runner changed")

    tooling_hashes: dict[str, str] = {}
    for label, record in contract["protocol_tooling"].items():
        path = Path(str(record["path"])).resolve()
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"frozen protocol tool changed: {label}")
        tooling_hashes[label] = str(record["sha256"])

    asset_hashes: dict[str, str] = {}
    for label in ("evssm_checkpoint", "droid_checkpoint", "omnidata_checkpoint"):
        record = contract["assets"][label]
        path = Path(str(record["path"])).resolve()
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"frozen Unblur asset changed: {label}")
        asset_hashes[label] = str(record["sha256"])
    return {
        "unblur_commit": head,
        "measurement_hook_sha256": measurement_hashes,
        "resplat_commit": str(official["commit"]),
        "resplat_checkpoint_sha256": str(official["checkpoint_sha256"]),
        "resplat_runner_sha256": str(official["runner_sha256"]),
        "protocol_tooling_sha256": tooling_hashes,
        "unblur_asset_sha256": asset_hashes,
    }


def _same_gpu(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(
        str(observed.get(key)) == str(expected.get(key))
        for key in (
            "physical_index", "visible_devices", "logical_device", "name", "uuid", "serial"
        )
    )


def _argv_sha256(command: Sequence[str]) -> str:
    encoded = json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_pinned_audit(
    *,
    audit: Mapping[str, Any],
    label: str,
    expected_command: Sequence[str],
    expected_cwd: Path,
    expected_wrapper: Mapping[str, Any],
    expected_gpu: Mapping[str, Any],
    expected_lock: str,
) -> None:
    command = audit.get("command")
    if not isinstance(command, list) or not all(
        isinstance(value, str) and value for value in command
    ):
        raise ValueError(f"{label} pinned-wrapper audit has no argv-safe child command")
    if command != list(expected_command) or audit.get("command_sha256") != _argv_sha256(command):
        raise ValueError(f"{label} pinned-wrapper audit is not bound to the expected child")
    if Path(str(audit.get("working_directory", ""))).resolve() != expected_cwd.resolve():
        raise ValueError(f"{label} pinned-wrapper audit has the wrong working directory")
    if (
        audit.get("schema") != PINNED_GPU_COMMAND_SCHEMA
        or audit.get("status") != "complete"
        or audit.get("returncode") != 0
        or float(audit.get("full_child_process_wall_seconds", 0.0)) <= 0.0
        or audit.get("sequential_execution") is not True
        or audit.get("exclusive_lock") != expected_lock
        or audit.get("child_environment", {}).get("CUDA_VISIBLE_DEVICES")
        != str(expected_gpu["visible_devices"])
        or audit.get("child_environment", {}).get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        or audit.get("wrapper", {}).get("path") != expected_wrapper["path"]
        or audit.get("wrapper", {}).get("sha256") != expected_wrapper["sha256"]
        or not _same_gpu(audit.get("gpu", {}), expected_gpu)
        or audit.get("gpu", {}).get("idle_verified_before_launch") is not True
        or audit.get("gpu", {}).get("active_compute_pids_before_launch") != []
    ):
        raise ValueError(f"{label} did not use the exact pinned serial GPU wrapper")


def _metric_gates(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    return {
        "psnr": float(baseline["psnr"]) - float(candidate["psnr"])
        <= float(thresholds["psnr_drop_db_max"]),
        "ssim": float(baseline["ssim"]) - float(candidate["ssim"])
        <= float(thresholds["ssim_drop_max"]),
        "lpips": float(candidate["lpips"]) - float(baseline["lpips"])
        <= float(thresholds["lpips_increase_max"]),
    }


def _mean_views(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("cannot aggregate an empty metric view list")
    return {
        key: sum(float(record[key]) for record in records) / len(records)
        for key in ("psnr", "ssim", "lpips")
    }


def _validate_plan_against_contract(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    validated, _ = validate_execution_plan(plan_path)
    if validated.get("schema") != plan.get("schema"):
        raise ValueError("strict plan validator did not return the loaded formal plan")
    required_tools = (
        "materialize_frozen_bundle",
        "scene_exporter",
        "common_rgb_evaluator",
        "sequential_wall_executor",
        "pinned_gpu_wrapper",
    )
    for label in required_tools:
        if plan.get("protocol_tooling", {}).get(label) != contract.get(
            "protocol_tooling", {}
        ).get(label):
            raise ValueError(f"plan tool is not the preregistered tool: {label}")
    official_plan = plan.get("official_resplat", {})
    official_contract = contract["official_resplat"]
    if (
        Path(str(official_plan.get("repository", ""))).resolve()
        != Path(str(official_contract["repository"])).resolve()
        or Path(str(official_plan.get("python", ""))).resolve()
        != Path(str(official_contract["python"])).resolve()
        or official_plan.get("commit") != official_contract["commit"]
        or official_plan.get("model_preset") != official_contract["preset"]
        or int(official_plan.get("context_count", -1)) != 8
        or int(official_plan.get("recurrent_updates", -1)) != 4
        or official_plan.get("checkpoint")
        != {
            "path": official_contract["checkpoint"],
            "sha256": official_contract["checkpoint_sha256"],
        }
        or official_plan.get("runner")
        != {
            "path": official_contract["runner"],
            "sha256": official_contract["runner_sha256"],
        }
        or Path(str(plan.get("unblur_python", ""))).resolve()
        != Path(str(contract["unblur_baseline"]["python"])).resolve()
        or Path(str(plan.get("exclusive_lock", ""))).resolve()
        != Path(str(contract["execution_hardware"]["exclusive_lock"])).resolve()
    ):
        raise ValueError("plan executable/model/lock records differ from preregistration")


def _validate_exported_scene(
    *,
    plan: Mapping[str, Any],
    bundle: Mapping[str, Any],
    expected_exporter: Mapping[str, Any],
) -> tuple[Path, str]:
    scene_dir = Path(str(plan.get("scene_dir", ""))).resolve()
    manifest_path = scene_dir / "manifest.json"
    manifest = _load_json(manifest_path, "exported official-ReSplat scene manifest")
    if (
        manifest.get("schema") != SCENE_SCHEMA
        or manifest.get("artifact_class")
        != "official_cvg_resplat_colmap_input_scene"
        or manifest.get("formal_smoke") is not True
        or manifest.get("exporter") != expected_exporter
        or manifest.get("ground_truth_contract", {}).get("uses_ground_truth_pose")
        is not False
        or manifest.get("ground_truth_contract", {}).get("ground_truth_pose_used")
        is not False
    ):
        raise ValueError("exported scene provenance/ground-truth contract is invalid")
    task_root = Path(str(plan.get("task_root", ""))).resolve()
    frames_csv = task_root / "frames.csv"
    images_json = task_root / "images.json"
    if (
        manifest.get("source_csv", {}).get("path") != str(frames_csv)
        or manifest.get("source_csv", {}).get("sha256") != sha256_file(frames_csv)
        or manifest.get("images", {}).get("mapping_manifest", {}).get("path")
        != str(images_json)
        or manifest.get("images", {}).get("mapping_manifest", {}).get("sha256")
        != sha256_file(images_json)
        or manifest.get("images", {}).get("mapped_hashes_verified") is not True
    ):
        raise ValueError("exported scene is not bound to the frozen materializer inputs")
    records = bundle.get("records", [])
    sources = [int(record["source_index"]) for record in records]
    if manifest.get("selection", {}).get("source_indices") != sources:
        raise ValueError("exported scene source selection differs from the frozen bundle")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or len(frames) != len(records):
        raise ValueError("exported scene does not contain every frozen source frame")
    for record, frame in zip(records, frames):
        source = int(record["source_index"])
        selected = frame.get("selected_image", {})
        exported = scene_dir / str(selected.get("exported_relative_path", ""))
        if (
            int(frame.get("source_index", -1)) != source
            or selected.get("mode_label") != "evssm"
            or selected.get("source_sha256") != record["png_sha256"]
            or selected.get("declared_sha256") != record["png_sha256"]
            or selected.get("exported_sha256") != record["png_sha256"]
            or not exported.is_file()
            or sha256_file(exported) != record["png_sha256"]
            or frame.get("effective_pose", {}).get("uses_ground_truth_pose") is not False
            or frame.get("effective_pose", {}).get("source") != "frames_csv"
            or frame.get("effective_pose", {}).get("c2w_opencv")
            != record["c2w_opencv"]
            or frame.get("K") != record["intrinsics_px"]
        ):
            raise ValueError(f"exported scene frame differs from frozen source {source}")
    return manifest_path, sha256_file(manifest_path)


def _validate_run_manifests(
    *,
    plan: Mapping[str, Any],
    execution: Mapping[str, Any],
    official: Mapping[str, Any],
    gpu: Mapping[str, Any],
    scene_manifest_path: Path,
    scene_manifest_sha256: str,
) -> tuple[float, int]:
    execution_by_id = {
        int(record["submap_id"]): record for record in execution.get("tasks", [])
    }
    tasks = plan.get("tasks", [])
    if set(execution_by_id) != set(range(len(tasks))):
        raise ValueError("execution report does not exactly cover plan submaps")
    primary_seconds = 0.0
    primary_peak = 0
    for position, task in enumerate(tasks):
        if int(task.get("submap_id", -1)) != position:
            raise ValueError("plan submap IDs are not unique and contiguous")
        contexts = [int(value) for value in task.get("context_source_indices", [])]
        if len(contexts) != 8 or len(set(contexts)) != 8:
            raise ValueError("formal R4 submap does not contain eight unique contexts")
        aggregate = [int(value) for value in task.get("aggregate_target_source_indices", [])]
        target_sources = [int(value) for value in task["runner_target_source_indices"]]
        if any(value not in contexts for value in aggregate):
            raise ValueError("formal R4 aggregate query is not a context view")
        execution_record = execution_by_id[position]
        if (
            execution_record.get("returncode") != 0
            or execution_record.get("command_sha256")
            != _argv_sha256(task.get("command", []))
        ):
            raise ValueError("R4 subprocess failed")
        manifest_path = Path(str(task["output"])) / "run_manifest.json"
        if execution_record.get("run_manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("R4 run manifest changed after execution")
        manifest = _load_json(manifest_path, "R4 run manifest")
        model = manifest.get("official_resplat", {})
        if (
            manifest.get("schema") != RUNNER_SCHEMA
            or manifest.get("runner", {}).get("sha256") != official["runner_sha256"]
            or model.get("repository", {}).get("commit") != official["commit"]
            or model.get("checkpoint", {}).get("sha256") != official["checkpoint_sha256"]
            or model.get("model_preset") != official["preset"]
            or int(model.get("num_context", -1)) != 8
            or int(model.get("num_refine", -1)) != 4
            or manifest.get("image_shape") != [320, 448]
            or Path(str(manifest.get("scene", {}).get("manifest_path", ""))).resolve()
            != scene_manifest_path
            or manifest.get("scene", {}).get("manifest_sha256")
            != scene_manifest_sha256
        ):
            raise ValueError("R4 runner/model/checkpoint/refine/shape identity mismatch")
        if not _same_gpu(manifest.get("gpu_binding", {}), gpu):
            raise ValueError("R4 submap ran on a different physical GPU")
        selection = manifest.get("selection", {})
        if (
            [int(value) for value in selection.get("context_source_indices", [])] != contexts
            or [int(value) for value in selection.get("target_source_indices", [])]
            != target_sources
        ):
            raise ValueError("R4 manifest source selection differs from the frozen plan")
        if manifest.get("paired_contract", {}).get(
            "target_rgb_passed_to_forward_update"
        ) is not False:
            raise ValueError("R4 recurrent update saw target RGB")
        if manifest.get("metrics", {}).get("reference", {}).get(
            "passed_to_encoder_or_forward_update"
        ) is not False:
            raise ValueError("R4 model saw a metric reference")
        timing = manifest.get("terminal_reconstruction", {})
        seconds = float(timing.get("wall_seconds", 0.0))
        peak = int(timing.get("peak_allocated_bytes", 0))
        if (
            timing.get("primary") is not True
            or timing.get("wall_scope")
            != "terminal_backend_setup_plus_core_no_metrics_or_artifact_io"
            or timing.get("peak_scope")
            != "process_setup_plus_core_before_metrics_and_artifact_io"
            or seconds <= 0
            or peak <= 0
            or "metric computation" not in timing.get("excludes", [])
            or "output PNG/PLY artifact I/O" not in timing.get("excludes", [])
        ):
            raise ValueError("R4 primary time/peak boundary is invalid")
        if (
            not math.isclose(
                float(execution_record.get("primary_terminal_reconstruction_seconds", -1)),
                seconds,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            or int(execution_record.get("primary_terminal_peak_allocated_bytes", -1))
            != peak
            or float(execution_record.get("full_process_wall_seconds_secondary", 0.0))
            < seconds
        ):
            raise ValueError("R4 per-process primary/secondary timing audit mismatch")
        primary_seconds += seconds
        primary_peak = max(primary_peak, peak)
        artifacts = {
            int(record["source_index"]): record
            for record in manifest.get("outputs", {}).get(
                "paired_refine4_rendered", []
            )
        }
        if set(artifacts) != set(target_sources):
            raise ValueError("R4 render hashes do not exactly cover runner targets")
        for source, artifact in artifacts.items():
            path = Path(str(task["output"])) / str(artifact["relative_path"])
            if (
                artifact.get("quantization")
                != "official_resplat_mul255_astype_uint8_floor"
                or not path.is_file()
                or sha256_file(path) != artifact.get("sha256")
            ):
                raise ValueError(f"R4 rendered PNG integrity failure: {source}")
    return primary_seconds, primary_peak


def build_report(contract_path: Path) -> dict[str, Any]:
    contract = _load_json(contract_path, "preregistered contract")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("wrong preregistered contract schema")
    static_provenance = _validate_static_provenance(contract)
    output_root = Path(contract["assets"]["output_root"]).resolve()
    gpu = contract["execution_hardware"]["gpu"]
    gpu_lock = str(contract["execution_hardware"]["exclusive_lock"])
    unblur_contract = contract["unblur_baseline"]
    wrapper_contract = contract["protocol_tooling"]["pinned_gpu_wrapper"]
    official = {
        "commit": contract["official_resplat"]["commit"],
        "checkpoint_sha256": contract["official_resplat"]["checkpoint_sha256"],
        "runner_sha256": contract["official_resplat"]["runner_sha256"],
        "preset": contract["official_resplat"]["preset"],
    }
    thresholds = contract["reporting"]["noninferiority"]
    pooled: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in ARMS}
    scenes_report: dict[str, Any] = {}
    total_submaps = 0
    total_eval = 0
    aggregate_primary_seconds = {arm: 0.0 for arm in ARMS}
    aggregate_peak_bytes = {arm: 0 for arm in ARMS}
    r4_full_process_secondary = 0.0
    unblur_full_process_secondary = 0.0
    all_scene_gates = True

    for scene in contract["scenes"]:
        name = str(scene["name"])
        expected_eval = int(scene["eval_count"])
        expected_sources = int(scene["expected_keyframe_count"])
        expected_submaps = int(scene["expected_resplat_submaps"])
        unblur = output_root / "unblur_pristine" / name / "offline_fair_26k"
        tasks_root = output_root / "resplat_tasks" / name
        measurement_path = unblur / "measurement_manifest.json"
        bundle_path = unblur / "frozen_inputs" / "bundle.json"
        plan_path = tasks_root / "plan.json"
        execution_path = tasks_root / "execution_audit" / "execution.json"
        common_path = tasks_root / "common_rgb_metrics" / "report.json"
        unblur_gpu_audit_path = output_root / "gpu_audits" / f"unblur_{name}.json"
        common_gpu_audit_path = tasks_root / "common_rgb_gpu_execution.json"
        measurement = _load_json(measurement_path, f"{name} U measurement")
        bundle = _load_json(bundle_path, f"{name} frozen bundle")
        plan = _load_json(plan_path, f"{name} R4 plan")
        execution = _load_json(execution_path, f"{name} R4 execution")
        common = _load_json(common_path, f"{name} common RGB report")
        unblur_gpu_audit = _load_json(unblur_gpu_audit_path, f"{name} U GPU audit")
        common_gpu_audit = _load_json(common_gpu_audit_path, f"{name} metric GPU audit")

        _validate_plan_against_contract(
            plan_path=plan_path,
            plan=plan,
            contract=contract,
        )
        if (
            Path(str(plan.get("task_root", ""))).resolve() != tasks_root
            or Path(str(plan.get("scene_dir", ""))).resolve()
            != output_root / "resplat_scenes" / name
        ):
            raise ValueError(f"{name} plan task/scene roots differ from preregistration")
        scene_manifest_path, scene_manifest_sha256 = _validate_exported_scene(
            plan=plan,
            bundle=bundle,
            expected_exporter=contract["protocol_tooling"]["scene_exporter"],
        )

        expected_u_command = [
            str(unblur_contract["python"]),
            "run.py",
            str(scene["config"]),
        ]
        _validate_pinned_audit(
            audit=unblur_gpu_audit,
            label=f"{name} Unblur",
            expected_command=expected_u_command,
            expected_cwd=Path(str(unblur_contract["worktree"])),
            expected_wrapper=wrapper_contract,
            expected_gpu=gpu,
            expected_lock=gpu_lock,
        )
        common_outer_command = plan.get("common_rgb_evaluation_command")
        if not isinstance(common_outer_command, list) or "--" not in common_outer_command:
            raise ValueError(f"{name} plan has no pinned common-RGB wrapper command")
        separator = common_outer_command.index("--")
        if (
            separator < 2
            or common_outer_command[1] != wrapper_contract["path"]
            or common_outer_command[0] != contract["official_resplat"]["python"]
        ):
            raise ValueError(f"{name} common-RGB command uses the wrong wrapper/interpreter")
        _validate_pinned_audit(
            audit=common_gpu_audit,
            label=f"{name} common RGB",
            expected_command=common_outer_command[separator + 1 :],
            expected_cwd=Path(str(plan.get("workspace", ""))),
            expected_wrapper=wrapper_contract,
            expected_gpu=gpu,
            expected_lock=gpu_lock,
        )

        if (
            measurement.get("schema") != MEASUREMENT_SCHEMA
            or measurement.get("status") != "complete"
            or measurement.get("single_ordinary_26k_trajectory") is not True
            or measurement.get("milestones") != list(MILESTONES)
            or measurement.get("official_resplat_executed_in_process") is not False
            or measurement.get("residual_replay_executed") is not False
            or not _same_gpu(measurement.get("gpu_binding", {}), gpu)
        ):
            raise ValueError(f"{name} U trajectory manifest is not formal/complete")
        if (
            bundle.get("png_quantization")
            != "official_resplat_mul255_astype_uint8_floor"
            or int(bundle.get("mapped_keyframe_count", -1)) != expected_sources
            or len(bundle.get("eval_source_indices", [])) != expected_eval
            or len(bundle.get("context_windows", [])) != expected_submaps
            or not _same_gpu(bundle.get("gpu_binding", {}), gpu)
        ):
            raise ValueError(f"{name} frozen bundle count/GPU/quantizer mismatch")
        if (
            plan.get("schema") != PLAN_SCHEMA
            or plan.get("scene_name") != name
            or not isinstance(plan.get("workspace"), str)
            or not plan.get("workspace")
            or Path(str(plan.get("workspace", ""))).resolve()
            != Path(wrapper_contract["path"]).resolve().parents[1]
            or int(plan.get("source_keyframe_count", -1)) != expected_sources
            or int(plan.get("eval_query_count", -1)) != expected_eval
            or int(plan.get("submap_count", -1)) != expected_submaps
            or plan.get("all_formal_queries_are_context_mapped_training_views") is not True
            or int(plan.get("formal_query_is_not_context_count", -1)) != 0
            or plan.get("active_map_merge") is not False
            or plan.get("reads_unblur_gaussian_state") is not False
            or not _same_gpu(plan.get("gpu_contract", {}), gpu)
        ):
            raise ValueError(f"{name} R4 plan identity/count/scope mismatch")
        if (
            plan.get("source_bundle", {}).get("sha256") != sha256_file(bundle_path)
            or Path(str(plan.get("source_bundle", {}).get("path", ""))).resolve()
            != bundle_path
        ):
            raise ValueError(f"{name} plan is not bound to its frozen bundle")
        if (
            plan.get("source_unblur_measurement", {}).get("sha256")
            != sha256_file(measurement_path)
            or Path(str(plan.get("source_unblur_measurement", {}).get("path", ""))).resolve()
            != measurement_path
        ):
            raise ValueError(f"{name} plan is not bound to its completed U trajectory")
        if (
            execution.get("schema") != EXECUTION_SCHEMA
            or execution.get("status") != "complete"
            or execution.get("fresh_process_per_submap") is not True
            or execution.get("sequential_execution") is not True
            or int(execution.get("submap_count", -1)) != expected_submaps
            or int(execution.get("completed_submaps", -1)) != expected_submaps
            or execution.get("plan", {}).get("sha256") != sha256_file(plan_path)
            or not _same_gpu(execution.get("gpu_binding", {}), gpu)
            or execution.get("gpu_binding", {}).get("idle_verified_before_launch")
            is not True
            or execution.get("gpu_binding", {}).get("active_compute_pids_before_launch")
            != []
            or execution.get("exclusive_lock") != gpu_lock
            or execution.get("child_environment", {}).get("CUDA_VISIBLE_DEVICES")
            != str(gpu["visible_devices"])
            or execution.get("child_environment", {}).get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or execution.get("executor")
            != contract["protocol_tooling"]["sequential_wall_executor"]
        ):
            raise ValueError(f"{name} R4 execution is not complete/fresh/serial")
        r_seconds, r_peak = _validate_run_manifests(
            plan=plan,
            execution=execution,
            official=official,
            gpu=gpu,
            scene_manifest_path=scene_manifest_path,
            scene_manifest_sha256=scene_manifest_sha256,
        )
        if not math.isclose(
            r_seconds,
            float(execution.get("primary_terminal_reconstruction_seconds", -1)),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ) or r_peak != int(execution.get("primary_terminal_peak_allocated_bytes", -1)):
            raise ValueError(f"{name} execution primary time/peak aggregate mismatch")
        recomputed_secondary = sum(
            float(record.get("full_process_wall_seconds_secondary", 0.0))
            for record in execution.get("tasks", [])
        )
        if (
            recomputed_secondary <= 0.0
            or not math.isclose(
                recomputed_secondary,
                float(
                    execution.get(
                        "sequential_total_full_process_wall_seconds_secondary", -1
                    )
                ),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            or float(execution.get("executor_elapsed_wall_seconds_secondary", -1))
            < recomputed_secondary
        ):
            raise ValueError(f"{name} execution full-process secondary audit mismatch")

        u_runtime: dict[str, Any] = {}
        previous_seconds = -1.0
        for milestone in MILESTONES:
            arm = f"U{milestone // 1000}"
            metrics_path = unblur / "milestones" / f"iter_{milestone:06d}" / "metrics.json"
            metrics = _load_json(metrics_path, f"{name} {arm} milestone")
            seconds = float(metrics.get("optimizer_cumulative_seconds_excluding_measurement", 0))
            peak = int(metrics.get("optimizer_peak_allocated_bytes_excluding_measurement", 0))
            if (
                metrics.get("schema") != MILESTONE_SCHEMA
                or int(metrics.get("iteration", -1)) != milestone
                or metrics.get("same_ordinary_26k_trajectory") is not True
                or metrics.get("png_quantization")
                != "official_resplat_mul255_astype_uint8_floor"
                or int(metrics.get("num_frames", -1)) != expected_eval
                or seconds <= previous_seconds
                or peak <= 0
                or not _same_gpu(metrics.get("gpu_binding", {}), gpu)
            ):
                raise ValueError(f"{name} {arm} milestone is invalid")
            previous_seconds = seconds
            u_runtime[arm] = {
                "primary_terminal_seconds": seconds,
                "primary_peak_allocated_bytes": peak,
                "manifest": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            }
            aggregate_primary_seconds[arm] += seconds
            aggregate_peak_bytes[arm] = max(aggregate_peak_bytes[arm], peak)
        if (
            not math.isclose(
                float(
                    measurement.get(
                        "optimizer_cumulative_seconds_excluding_measurement", -1
                    )
                ),
                float(u_runtime["U26"]["primary_terminal_seconds"]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            or int(
                measurement.get(
                    "optimizer_peak_allocated_bytes_excluding_measurement", -1
                )
            )
            != int(u_runtime["U26"]["primary_peak_allocated_bytes"])
        ):
            raise ValueError(f"{name} U measurement does not terminate at U26")

        if (
            common.get("schema") != COMMON_SCHEMA
            or common.get("common_resolution_hw") != [320, 448]
            or common.get("resize") != "official_cvg_resplat_PIL_LANCZOS"
            or common.get("metric_implementation")
            != "official_cvg_resplat_compute_metrics"
            or common.get("prediction_source") != "saved_quantized_png_for_every_arm"
            or common.get("frozen_reference_artifacts_passed_to_model") is not False
            or common.get("depth_l1_formal_gate", {}).get("available") is not False
            or common.get("scope", {}).get("query_is_context_count") != expected_eval
            or common.get("scope", {}).get("query_is_not_context_count") != 0
            or common.get("inputs", {}).get("bundle", {}).get("sha256")
            != sha256_file(bundle_path)
            or common.get("inputs", {}).get("plan", {}).get("sha256")
            != sha256_file(plan_path)
            or common.get("inputs", {}).get("execution", {}).get("sha256")
            != sha256_file(execution_path)
            or not _same_gpu(common.get("gpu_binding", {}), gpu)
            or common.get("noninferiority", {}).get("thresholds") != thresholds
        ):
            raise ValueError(f"{name} common RGB report integrity/scope mismatch")
        scene_metrics: dict[str, Any] = {}
        expected_view_names = [
            f"{int(source):08d}.png" for source in bundle.get("eval_source_indices", [])
        ]
        for arm in ARMS:
            result = common.get("metrics", {}).get(arm, {})
            views = result.get("per_view", [])
            if (
                len(views) != expected_eval
                or [str(record.get("name", "")) for record in views]
                != expected_view_names
                or any(
                    not math.isfinite(float(record.get(key, math.nan)))
                    for record in views
                    for key in ("psnr", "ssim", "lpips")
                )
            ):
                raise ValueError(f"{name} {arm} does not contain every evaluation view")
            recomputed = _mean_views(views)
            if any(
                not math.isclose(
                    recomputed[key], float(result.get("mean", {}).get(key, math.nan)),
                    rel_tol=1e-6, abs_tol=1e-6
                )
                for key in recomputed
            ):
                raise ValueError(f"{name} {arm} mean does not match its per-view values")
            pooled[arm].extend(views)
            scene_metrics[arm] = result["mean"]
        gates = _metric_gates(scene_metrics["U26"], scene_metrics["R4-multisubmap"], thresholds)
        if common.get("noninferiority", {}).get("gates") != gates:
            raise ValueError(f"{name} common report gate recomputation mismatch")
        scene_passed = all(gates.values())
        all_scene_gates = all_scene_gates and scene_passed
        aggregate_primary_seconds["R4-multisubmap"] += r_seconds
        aggregate_peak_bytes["R4-multisubmap"] = max(
            aggregate_peak_bytes["R4-multisubmap"], r_peak
        )
        secondary = float(
            execution.get("sequential_total_full_process_wall_seconds_secondary", 0.0)
        )
        r4_full_process_secondary += secondary
        unblur_child_secondary = float(
            unblur_gpu_audit["full_child_process_wall_seconds"]
        )
        if unblur_child_secondary < float(
            u_runtime["U26"]["primary_terminal_seconds"]
        ):
            raise ValueError(f"{name} Unblur full-process wall is below its primary timer")
        unblur_full_process_secondary += unblur_child_secondary
        total_submaps += expected_submaps
        total_eval += expected_eval
        scenes_report[name] = {
            "counts": {
                "mapped_keyframes": expected_sources,
                "evaluation_context_views": expected_eval,
                "non_context_views": 0,
                "fresh_serial_resplat_subprocesses": expected_submaps,
            },
            "metrics": scene_metrics,
            "noninferiority": {"gates": gates, "passed": scene_passed},
            "runtime": {
                **u_runtime,
                "R4-multisubmap": {
                    "primary_terminal_seconds": r_seconds,
                    "primary_peak_allocated_bytes": r_peak,
                    "full_process_wall_seconds_secondary": secondary,
                },
                "unblur_full_process_wall_seconds_secondary": unblur_child_secondary,
                "r4_vs_u26_primary_speedup": u_runtime["U26"]["primary_terminal_seconds"] / r_seconds,
            },
            "artifacts": {
                "measurement": {"path": str(measurement_path), "sha256": sha256_file(measurement_path)},
                "bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
                "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
                "exported_scene_manifest": {
                    "path": str(scene_manifest_path),
                    "sha256": scene_manifest_sha256,
                },
                "execution": {"path": str(execution_path), "sha256": sha256_file(execution_path)},
                "common_rgb": {"path": str(common_path), "sha256": sha256_file(common_path)},
                "unblur_gpu_audit": {
                    "path": str(unblur_gpu_audit_path),
                    "sha256": sha256_file(unblur_gpu_audit_path),
                },
                "common_rgb_gpu_audit": {
                    "path": str(common_gpu_audit_path),
                    "sha256": sha256_file(common_gpu_audit_path),
                },
            },
        }

    expected_total_submaps = int(contract["reporting"]["expected_total_resplat_submaps"])
    expected_total_eval = int(contract["reporting"]["aggregate_eval_count"])
    if total_submaps != expected_total_submaps or total_eval != expected_total_eval:
        raise ValueError("three-scene exact total count gate failed")
    pooled_metrics = {arm: _mean_views(records) for arm, records in pooled.items()}
    if any(len(records) != expected_total_eval for records in pooled.values()):
        raise ValueError("pooled metric aggregation does not contain exactly 113 views per arm")
    pooled_gates = _metric_gates(
        pooled_metrics["U26"], pooled_metrics["R4-multisubmap"], thresholds
    )
    r4_primary = aggregate_primary_seconds["R4-multisubmap"]
    u26_primary = aggregate_primary_seconds["U26"]
    speedup_gate = r4_primary < u26_primary
    claim_allowed = (
        all_scene_gates
        and all(pooled_gates.values())
        and speedup_gate
        and total_submaps == 38
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "static_provenance": static_provenance,
        "claim_boundary": {
            "candidate": "independent official ReSplat chronological 8-view terminal submaps",
            "active_unblur_map_merge": False,
            "u_plus_r_fused_arm": False,
            "evaluation_scope": "100% context mapped-training-view rendering; not novel-view",
        },
        "gpu": gpu,
        "scenes": scenes_report,
        "aggregate": {
            "weighting": "pooled_per_evaluation_view_across_three_scenes",
            "evaluation_view_count": total_eval,
            "metrics": pooled_metrics,
            "noninferiority": {"gates": pooled_gates, "passed": all(pooled_gates.values())},
            "runtime": {
                "primary_terminal_seconds": aggregate_primary_seconds,
                "primary_peak_allocated_bytes": aggregate_peak_bytes,
                "r4_full_process_wall_seconds_secondary": r4_full_process_secondary,
                "unblur_full_process_wall_seconds_secondary": unblur_full_process_secondary,
                "r4_vs_u26_primary_speedup": u26_primary / r4_primary,
            },
            "fresh_serial_resplat_subprocesses": total_submaps,
        },
        "speedup_claim": {
            "allowed": claim_allowed,
            "all_scene_metric_gates_passed": all_scene_gates,
            "pooled_metric_gates_passed": all(pooled_gates.values()),
            "r4_primary_time_lower_than_u26": speedup_gate,
            "exact_38_fresh_serial_subprocesses": total_submaps == 38,
            "wording_if_allowed": (
                "On the preregistered mapped-training-view protocol, the independent "
                "official ReSplat terminal renderer is noninferior to U26 and faster "
                "in primary terminal reconstruction time."
                if claim_allowed else None
            ),
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Offline ReSplat fair 26K final report",
        "",
        "All R4 queries are context mapped-training views; these are not novel-view results.",
        "",
        "| Scene | U26 PSNR | R4 PSNR | U26 s | R4 primary s | Speedup | Gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, scene in report["scenes"].items():
        metrics = scene["metrics"]
        runtime = scene["runtime"]
        lines.append(
            f"| {name} | {metrics['U26']['psnr']:.4f} | "
            f"{metrics['R4-multisubmap']['psnr']:.4f} | "
            f"{runtime['U26']['primary_terminal_seconds']:.3f} | "
            f"{runtime['R4-multisubmap']['primary_terminal_seconds']:.3f} | "
            f"{runtime['r4_vs_u26_primary_speedup']:.3f}x | "
            f"{'PASS' if scene['noninferiority']['passed'] else 'FAIL'} |"
        )
    aggregate = report["aggregate"]
    lines.extend(
        [
            "",
            f"Pooled views: {aggregate['evaluation_view_count']}; fresh serial R4 processes: "
            f"{aggregate['fresh_serial_resplat_subprocesses']}.",
            f"Final speedup claim: {'ALLOWED' if report['speedup_claim']['allowed'] else 'NOT ALLOWED'}.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(contract_path: Path, destination: Path) -> Path:
    destination = destination.expanduser().resolve()
    if not str(destination).startswith("/srv/"):
        raise ValueError("final report output must be under /srv")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite final report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    installed = False
    try:
        report = build_report(contract_path.expanduser().resolve())
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "summary.md").write_text(_markdown(report), encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing concurrent overwrite: {destination}")
        os.rename(staging, destination)
        installed = True
        return destination
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = write_report(args.contract, args.output_dir)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, ZeroDivisionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"final offline ReSplat fair report saved atomically at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
