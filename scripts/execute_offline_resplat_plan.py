#!/usr/bin/env python3
"""Validate or sequentially execute an audited offline ReSplat submap plan.

The subprocess wall clock surrounds each fresh official runner process, so it
includes Python startup, checkpoint/model load, preprocessing, initialization,
four recurrent updates, target rendering, metrics, and artifact I/O.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from offline_fair_gpu_contract import validate_gpu_contract


PLAN_SCHEMA = "unblur_slam.offline_resplat_multisubmap_plan.v1"
REPORT_SCHEMA = "unblur_slam.offline_resplat_execution.v1"
BUNDLE_SCHEMA = "unblur_slam.offline_fair_frozen_bundle.v1"
GLOBAL_GPU_LOCK = Path("/srv/szha0669/unblur-slam/locks/physical_gpu1.lock")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def argv_sha256(command: Sequence[str]) -> str:
    encoded = json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _exact_tool_record(payload: Mapping[str, Any], label: str) -> tuple[Path, str]:
    record = payload.get("protocol_tooling", {}).get(label, {})
    path = Path(str(record.get("path", ""))).resolve()
    expected = str(record.get("sha256", ""))
    if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"frozen plan tool changed or is absent: {label}")
    return path, expected


def _chronological_context_windows(source_indices: Sequence[int]) -> list[list[int]]:
    ordered = [int(value) for value in source_indices]
    if ordered != sorted(set(ordered)) or len(ordered) < 8:
        raise ValueError("frozen source indices must be sorted, unique, and at least eight")
    windows = [ordered[start : start + 8] for start in range(0, len(ordered), 8)]
    if len(windows[-1]) < 8:
        windows[-1] = ordered[-8:]
    return windows


def _route_source_index(source_index: int, windows: Sequence[Sequence[int]]) -> int:
    exact = [position for position, window in enumerate(windows) if source_index in window]
    if exact:
        return exact[0]
    distances = [
        min(abs(int(source_index) - int(context)) for context in window)
        for window in windows
    ]
    return min(range(len(distances)), key=lambda position: (distances[position], position))


def _gpu_arguments(gpu: Mapping[str, Any]) -> list[str]:
    return [
        "--expected-physical-index", str(gpu["physical_index"]),
        "--expected-cuda-visible-devices", str(gpu["visible_devices"]),
        "--expected-gpu-name", str(gpu["name"]),
        "--expected-gpu-uuid", str(gpu["uuid"]),
        "--expected-gpu-serial", str(gpu["serial"]),
    ]


def _validate_exact_plan_commands(
    *,
    payload: Mapping[str, Any],
    plan_path: Path,
    bundle: Mapping[str, Any],
) -> None:
    task_root = Path(str(payload.get("task_root", ""))).resolve()
    scene_dir = Path(str(payload.get("scene_dir", ""))).resolve()
    workspace = Path(str(payload.get("workspace", ""))).resolve()
    if task_root != plan_path.parent or not str(scene_dir).startswith("/srv/"):
        raise ValueError("plan task/scene roots are not the frozen /srv locations")
    official = payload.get("official_resplat", {})
    checkpoint = official.get("checkpoint", {})
    runner = official.get("runner", {})
    gpu = payload.get("gpu_contract", {})
    required_gpu = {
        "physical_index", "visible_devices", "logical_device", "name", "uuid", "serial"
    }
    if set(gpu) != required_gpu or str(gpu.get("logical_device")) != "cuda:0":
        raise ValueError("plan lacks the exact six-field GPU contract")
    if str(gpu.get("physical_index")) != "1" or str(gpu.get("visible_devices")) != "1":
        raise ValueError("formal plan must expose only physical GPU1 as logical cuda:0")
    if Path(str(payload.get("exclusive_lock", ""))).resolve() != GLOBAL_GPU_LOCK:
        raise ValueError("formal plan does not use the shared physical-GPU1 lock")

    tools = {
        label: _exact_tool_record(payload, label)[0]
        for label in (
            "materialize_frozen_bundle",
            "scene_exporter",
            "common_rgb_evaluator",
            "sequential_wall_executor",
            "pinned_gpu_wrapper",
        )
    }
    if tools["sequential_wall_executor"] != Path(__file__).resolve():
        raise ValueError("plan points to a different sequential executor")
    scene_exporter = payload.get("scene_exporter", {})
    if (
        Path(str(scene_exporter.get("path", ""))).resolve() != tools["scene_exporter"]
        or scene_exporter.get("sha256")
        != payload["protocol_tooling"]["scene_exporter"]["sha256"]
    ):
        raise ValueError("plan scene-exporter records disagree")
    runner_path = Path(str(runner.get("path", ""))).resolve()
    runner_sha = str(runner.get("sha256", ""))
    if (
        not runner_path.is_file()
        or sha256_file(runner_path) != runner_sha
        or runner_sha != str(official.get("runner_sha256", ""))
    ):
        raise ValueError("plan official ReSplat runner changed")
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve()
    checkpoint_sha = str(checkpoint.get("sha256", ""))
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != checkpoint_sha
        or checkpoint_sha != str(official.get("checkpoint_sha256", ""))
    ):
        raise ValueError("plan official ReSplat checkpoint changed")
    if (
        int(official.get("context_count", -1)) != 8
        or int(official.get("recurrent_updates", -1)) != 4
        or official.get("model_preset") != "dl3dv_8v_256x448_small"
    ):
        raise ValueError("plan is not the frozen official 8-context/refine4 preset")

    source_indices = [int(record["source_index"]) for record in bundle.get("records", [])]
    expected_export = [
        str(payload["unblur_python"]),
        str(tools["scene_exporter"]),
        "--frames-csv", str(task_root / "frames.csv"),
        "--output-dir", str(scene_dir),
        "--indices", ",".join(str(value) for value in source_indices),
        "--image-mode", "evssm",
        "--images-json", str(task_root / "images.json"),
        "--resplat-repo", str(Path(str(official["repository"])).resolve()),
        "--checkpoint", str(checkpoint_path),
        "--expected-checkpoint-sha256", checkpoint_sha,
        "--model-preset", "dl3dv_8v_256x448_small",
        "--formal-smoke",
    ]
    if payload.get("export_command") != expected_export:
        raise ValueError("plan export argv is not the frozen scene-exporter command")

    for task in payload["tasks"]:
        contexts = [int(value) for value in task["context_source_indices"]]
        targets = [int(value) for value in task["runner_target_source_indices"]]
        output = Path(str(task["output"])).resolve()
        expected = [
            str(Path(str(official["python"])).resolve()),
            str(runner_path),
            "--scene-path", str(scene_dir),
            "--scene-manifest", str(scene_dir / "manifest.json"),
            "--resplat-repo", str(Path(str(official["repository"])).resolve()),
            "--checkpoint", str(checkpoint_path),
            "--expected-checkpoint-sha256", checkpoint_sha,
            "--output-dir", str(output),
            "--model-preset", "dl3dv_8v_256x448_small",
            "--expected-resplat-commit", str(official["commit"]),
            "--device", "cuda:0",
            "--image-shape", "320", "448",
            *_gpu_arguments(gpu),
            "--expected-target-count", str(len(targets)),
            "--max-save-images", str(len(targets)),
            "--save-ply",
            "--target-reference-json", str(task_root / "references.json"),
            "--context-source-indices", *[str(value) for value in contexts],
            "--target-source-indices", *[str(value) for value in targets],
        ]
        if task.get("command") != expected:
            raise ValueError(f"submap {task.get('submap_id')} argv is not exactly frozen")

    common_child = [
        str(Path(str(official["python"])).resolve()),
        str(tools["common_rgb_evaluator"]),
        "--bundle", str(Path(str(payload["source_bundle"]["path"])).resolve()),
        "--resplat-plan", str(plan_path),
        "--execution-report", str(task_root / "execution_audit" / "execution.json"),
        "--workspace", str(workspace),
        "--resplat-repo", str(Path(str(official["repository"])).resolve()),
        "--output-dir", str(task_root / "common_rgb_metrics"),
        "--device", "cuda:0",
        *_gpu_arguments(gpu),
    ]
    expected_common = [
        str(Path(str(official["python"])).resolve()),
        str(tools["pinned_gpu_wrapper"]),
        "--lock-file", str(GLOBAL_GPU_LOCK),
        "--audit-report", str(task_root / "common_rgb_gpu_execution.json"),
        *_gpu_arguments(gpu),
        "--",
        *common_child,
    ]
    if payload.get("common_rgb_evaluation_command") != expected_common:
        raise ValueError("plan common-RGB argv is not exactly frozen")
    expected_sequential = [
        str(Path(str(official["python"])).resolve()),
        str(tools["sequential_wall_executor"]),
        "--plan", str(plan_path),
        "--report-dir", str(task_root / "execution_audit"),
        "--temporary-root", str(task_root / "tmp"),
        "--lock-file", str(GLOBAL_GPU_LOCK),
        *_gpu_arguments(gpu),
        "--execute",
    ]
    if payload.get("sequential_execution_command") != expected_sequential:
        raise ValueError("plan sequential-executor argv is not exactly frozen")


def validate_plan(plan_path: Path) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    plan_path = plan_path.expanduser().resolve()
    payload = _load_json_object(plan_path, "ReSplat plan")
    if payload.get("schema") != PLAN_SCHEMA:
        raise ValueError("wrong ReSplat execution-plan schema")
    if payload.get("artifact_class") != "independent_official_resplat_terminal_multisubmap":
        raise ValueError("plan is not the independent official terminal backend")
    if payload.get("active_map_merge") is not False:
        raise ValueError("plan attempts an undeclared active-map merge")
    if payload.get("reads_unblur_gaussian_state") is not False:
        raise ValueError("plan attempts to read arbitrary Unblur Gaussian state")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("ReSplat execution plan has no tasks")
    declared_count = int(payload.get("submap_count", -1))
    expected_count = int(payload.get("expected_submap_count", -1))
    if declared_count != len(tasks) or expected_count != len(tasks):
        raise ValueError("ReSplat plan submap count is not the preregistered exact count")
    if payload.get("all_formal_queries_are_context_mapped_training_views") is not True:
        raise ValueError("formal plan must declare its all-context training-view scope")
    if int(payload.get("formal_query_is_not_context_count", -1)) != 0:
        raise ValueError("formal plan cannot advertise a nonexistent non-context stratum")
    source_bundle = payload.get("source_bundle", {})
    bundle_path = Path(str(source_bundle.get("path", ""))).resolve()
    if (
        not bundle_path.is_file()
        or source_bundle.get("sha256") != sha256_file(bundle_path)
    ):
        raise ValueError("plan source bundle is absent or changed")
    bundle = _load_json_object(bundle_path, "frozen source bundle")
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("plan source bundle has the wrong schema")
    sources = [int(record["source_index"]) for record in bundle.get("records", [])]
    expected_windows = _chronological_context_windows(sources)
    observed_windows = [
        [int(value) for value in window] for window in bundle.get("context_windows", [])
    ]
    if observed_windows != expected_windows:
        raise ValueError("frozen bundle context windows are not chronological groups of eight")
    eval_sources = [int(value) for value in bundle.get("eval_source_indices", [])]
    if eval_sources != sorted(set(eval_sources)) or not set(eval_sources).issubset(sources):
        raise ValueError("frozen bundle evaluation indices are invalid")
    expected_routes = {
        source: _route_source_index(source, expected_windows) for source in eval_sources
    }
    observed_routes = {
        int(key): int(value) for key, value in bundle.get("eval_routes", {}).items()
    }
    if observed_routes != expected_routes:
        raise ValueError("frozen bundle evaluation router is not the chronological router")
    measurement = payload.get("source_unblur_measurement", {})
    measurement_path = Path(str(measurement.get("path", ""))).resolve()
    if (
        not measurement_path.is_file()
        or measurement.get("sha256") != sha256_file(measurement_path)
    ):
        raise ValueError("plan source Unblur measurement is absent or changed")
    if (
        int(payload.get("source_keyframe_count", -1)) != len(sources)
        or int(payload.get("expected_source_keyframe_count", -1)) != len(sources)
        or int(payload.get("eval_query_count", -1)) != len(eval_sources)
        or int(payload.get("expected_eval_query_count", -1)) != len(eval_sources)
        or len(tasks) != len(expected_windows)
    ):
        raise ValueError("plan counts differ from the frozen bundle")
    normalized: list[dict[str, Any]] = []
    outputs: set[Path] = set()
    for position, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ValueError(f"task {position} is not an object")
        if int(task.get("submap_id", -1)) != position:
            raise ValueError("submap IDs must be unique and contiguous in execution order")
        contexts = [int(value) for value in task.get("context_source_indices", [])]
        if contexts != expected_windows[position]:
            raise ValueError(f"task {position} is not the frozen chronological context")
        aggregate_targets = [
            int(value) for value in task.get("aggregate_target_source_indices", [])
        ]
        expected_aggregate = [
            source for source in eval_sources if expected_routes[source] == position
        ]
        if aggregate_targets != expected_aggregate:
            raise ValueError("task aggregate targets differ from the frozen router")
        expected_runner_targets = expected_aggregate or [contexts[0]]
        runner_targets = [
            int(value) for value in task.get("runner_target_source_indices", [])
        ]
        if runner_targets != expected_runner_targets:
            raise ValueError("task runner targets differ from the frozen probe policy")
        if bool(task.get("probe_only_no_aggregate_target", False)) != (not expected_aggregate):
            raise ValueError("task probe-only marker differs from the frozen router")
        if [int(value) for value in task.get("context_target_overlap", [])] != sorted(
            set(contexts) & set(runner_targets)
        ):
            raise ValueError("task context/target overlap audit is inconsistent")
        command = task.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(value, str) and value for value in command
        ):
            raise ValueError(f"task {position} has no argv-safe command")
        output = Path(str(task["output"])).resolve()
        if not str(output).startswith("/srv/"):
            raise ValueError(f"task {position} output is not under /srv: {output}")
        if output != plan_path.parent / f"submap_{position:03d}":
            raise ValueError(f"task {position} output is not the frozen submap path")
        if output in outputs:
            raise ValueError(f"duplicate task output: {output}")
        outputs.add(output)
        if "--output-dir" not in command:
            raise ValueError(f"task {position} command has no explicit output directory")
        declared_output = Path(command[command.index("--output-dir") + 1]).resolve()
        if declared_output != output:
            raise ValueError(f"task {position} command/output mismatch")
        for required in (
            "--context-source-indices",
            "--target-source-indices",
            "--target-reference-json",
        ):
            if required not in command:
                raise ValueError(f"task {position} is missing {required}")
        context_position = command.index("--context-source-indices")
        target_position = command.index("--target-source-indices")
        command_contexts = [
            int(value) for value in command[context_position + 1 : target_position]
        ]
        command_targets = [int(value) for value in command[target_position + 1 :]]
        if command_contexts != contexts or command_targets != runner_targets:
            raise ValueError("task metadata and explicit runner source selection disagree")
        if command[command.index("--device") + 1] != "cuda:0":
            raise ValueError("formal ReSplat task must address logical cuda:0")
        normalized.append(
            {
                "submap_id": int(task["submap_id"]),
                "command": command,
                "command_sha256": argv_sha256(command),
                "output": output,
                "probe_only_no_aggregate_target": bool(
                    task.get("probe_only_no_aggregate_target", False)
                ),
            }
        )
    _validate_exact_plan_commands(payload=payload, plan_path=plan_path, bundle=bundle)
    return payload, normalized


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    plan_path = args.plan.expanduser().resolve()
    plan, tasks = validate_plan(plan_path)
    base_report = {
        "schema": REPORT_SCHEMA,
        "executor": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "artifact_class": plan["artifact_class"],
        "active_map_merge": False,
        "reads_unblur_gaussian_state": False,
        "fresh_process_per_submap": True,
        "submap_count": len(tasks),
    }
    if not args.execute:
        return {**base_report, "status": "validated_cpu_only_not_executed"}, 0

    report_dir = args.report_dir.expanduser().resolve()
    temporary_root = args.temporary_root.expanduser().resolve()
    lock_file = args.lock_file.expanduser().resolve()
    if not all(
        str(path).startswith("/srv/")
        for path in (report_dir, temporary_root, lock_file)
    ):
        raise ValueError("execution report, TMPDIR, and GPU lock must all be under /srv")
    if (
        lock_file != GLOBAL_GPU_LOCK
        or lock_file != Path(str(plan.get("exclusive_lock", ""))).resolve()
        or report_dir != plan_path.parent / "execution_audit"
        or temporary_root != plan_path.parent / "tmp"
    ):
        raise ValueError("executor output/TMP/lock arguments differ from the frozen plan")
    if report_dir.exists() or report_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite execution report: {report_dir}")
    existing = [str(task["output"]) for task in tasks if task["output"].exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite task outputs: {existing}")
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{report_dir.name}.", dir=str(report_dir.parent))
    )
    installed = False
    records = []
    overall_start = time.perf_counter()
    exit_code = 0
    try:
        expected_gpu = {
            "physical_index": args.expected_physical_index,
            "visible_devices": args.expected_cuda_visible_devices,
            "logical_device": "cuda:0",
            "name": args.expected_gpu_name,
            "uuid": args.expected_gpu_uuid,
            "serial": args.expected_gpu_serial,
        }
        if any(
            str(plan.get("gpu_contract", {}).get(key)) != str(value)
            for key, value in expected_gpu.items()
        ):
            raise ValueError("executor GPU arguments disagree with the frozen plan")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(args.expected_cuda_visible_devices)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["TMPDIR"] = str(temporary_root)
        environment["PYTHONPYCACHEPREFIX"] = str(temporary_root / "pycache")
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with lock_file.open("a+", encoding="utf-8") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"pinned GPU lock is already held: {lock_file}") from error
            gpu_binding = validate_gpu_contract(
                expected_gpu, require_visible_mask=False, require_idle=True
            )
            for task in tasks:
                log = staging / f"submap_{task['submap_id']:03d}.log"
                started = time.perf_counter()
                with log.open("wb") as handle:
                    result = subprocess.run(
                        task["command"],
                        stdin=subprocess.DEVNULL,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        check=False,
                    )
                wall = time.perf_counter() - started
                manifest = task["output"] / "run_manifest.json"
                primary_seconds = None
                primary_peak = None
                if manifest.is_file():
                    try:
                        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
                        primary = manifest_payload.get("terminal_reconstruction", {})
                        if (
                            primary.get("primary") is not True
                            or primary.get("wall_scope")
                            != "terminal_backend_setup_plus_core_no_metrics_or_artifact_io"
                            or primary.get("peak_scope")
                            != "process_setup_plus_core_before_metrics_and_artifact_io"
                        ):
                            raise ValueError("wrong primary timing/peak scope")
                        primary_seconds = float(primary["wall_seconds"])
                        primary_peak = int(primary["peak_allocated_bytes"])
                        if primary_seconds <= 0.0 or primary_peak <= 0:
                            raise ValueError("non-positive primary time/process peak")
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        primary_seconds = None
                        primary_peak = None
                record = {
                    "submap_id": task["submap_id"],
                    "command_sha256": task["command_sha256"],
                    "probe_only_no_aggregate_target": task[
                        "probe_only_no_aggregate_target"
                    ],
                    "full_process_wall_seconds_secondary": wall,
                    "primary_terminal_reconstruction_seconds": primary_seconds,
                    "primary_terminal_peak_allocated_bytes": primary_peak,
                    "returncode": result.returncode,
                    "log": str(log),
                    "log_sha256": sha256_file(log),
                    "output": str(task["output"]),
                    "run_manifest_sha256": (
                        sha256_file(manifest) if manifest.is_file() else None
                    ),
                }
                records.append(record)
                if (
                    result.returncode != 0
                    or not manifest.is_file()
                    or primary_seconds is None
                    or primary_peak is None
                ):
                    exit_code = result.returncode or 2
                    break
        total_wall = time.perf_counter() - overall_start
        complete = exit_code == 0 and len(records) == len(tasks)
        report = {
            **base_report,
            "status": "complete" if complete else "failed",
            "gpu_binding": gpu_binding,
            "exclusive_lock": str(lock_file),
            "sequential_execution": True,
            "child_environment": {
                "CUDA_VISIBLE_DEVICES": str(args.expected_cuda_visible_devices),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            },
            "secondary_full_process_wall_definition": (
                "fresh subprocess from fork through exit; includes startup, checkpoint/model "
                "load, preprocessing, init0, four updates, rendering, metrics, and artifact I/O"
            ),
            "primary_terminal_wall_definition": (
                "synchronized terminal-backend setup plus core inference; excludes metric "
                "reference loading, metric computation, and artifact I/O"
            ),
            "peak_allocated_definition": (
                "process allocator peak across setup plus core, stopped before metrics and "
                "artifact I/O"
            ),
            "sequential_total_full_process_wall_seconds_secondary": sum(
                float(record["full_process_wall_seconds_secondary"]) for record in records
            ),
            "executor_elapsed_wall_seconds_secondary": total_wall,
            "primary_terminal_reconstruction_seconds": (
                sum(float(record["primary_terminal_reconstruction_seconds"]) for record in records)
                if complete else None
            ),
            "primary_terminal_peak_allocated_bytes": (
                max(int(record["primary_terminal_peak_allocated_bytes"]) for record in records)
                if complete else None
            ),
            "completed_submaps": len(records),
            "tasks": records,
        }
        (staging / "execution.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if report_dir.exists() or report_dir.is_symlink():
            raise FileExistsError(f"refusing concurrent overwrite: {report_dir}")
        os.rename(staging, report_dir)
        installed = True
        return report, exit_code
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--expected-physical-index", type=int)
    parser.add_argument("--expected-cuda-visible-devices")
    parser.add_argument("--expected-gpu-name")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--expected-gpu-serial")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch GPU tasks; omitted means CPU validation only",
    )
    args = parser.parse_args(argv)
    required = (
        args.report_dir,
        args.temporary_root,
        args.lock_file,
        args.expected_physical_index,
        args.expected_cuda_visible_devices,
        args.expected_gpu_name,
        args.expected_gpu_uuid,
        args.expected_gpu_serial,
    )
    if args.execute and any(value is None for value in required):
        parser.error("--execute requires report/TMP/lock and the complete GPU contract")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report, exit_code = execute(parse_args(argv))
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
