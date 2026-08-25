#!/usr/bin/env python3
"""Turn a frozen pristine-Unblur bundle into audited ReSplat task commands.

This is CPU-only.  It writes a COLMAP-export input contract and one explicit
eight-view official-ReSplat task per chronological submap; it does not execute
the CUDA model.
"""

from __future__ import annotations

import argparse
import csv
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

import numpy as np


BUNDLE_SCHEMA = "unblur_slam.offline_fair_frozen_bundle.v1"
PLAN_SCHEMA = "unblur_slam.offline_resplat_multisubmap_plan.v1"
REFERENCE_SCHEMA = "unblur_slam.offline_fair_metric_references.v1"
MEASUREMENT_SCHEMA = "unblur_slam.offline_fair_26k_measurement.v1"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_to_xyzw(rotation: Sequence[Sequence[float]]) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
        raise ValueError("rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-5):
        raise ValueError("rotation is not proper")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion.tolist()


def chronological_context_windows(
    source_indices: Sequence[int], context_count: int = 8
) -> list[list[int]]:
    ordered = [int(value) for value in source_indices]
    if ordered != sorted(set(ordered)):
        raise ValueError("source indices must be strictly increasing and unique")
    if len(ordered) < context_count:
        raise ValueError("not enough frozen keyframes for one ReSplat context")
    windows = [
        ordered[start : start + context_count]
        for start in range(0, len(ordered), context_count)
    ]
    if len(windows[-1]) < context_count:
        windows[-1] = ordered[-context_count:]
    return windows


def route_source_index(source_index: int, windows: Sequence[Sequence[int]]) -> int:
    exact = [index for index, window in enumerate(windows) if source_index in window]
    if exact:
        return exact[0]
    scores = [
        min(abs(int(source_index) - int(context)) for context in window)
        for window in windows
    ]
    return min(range(len(scores)), key=lambda index: (scores[index], index))


def validate_bundle(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[list[int]]]:
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("wrong frozen-bundle schema")
    if payload.get("ground_truth_used_for_context_selection") is not False:
        raise ValueError("context selection must not use ground truth")
    if payload.get("metric_used_for_context_selection") is not False:
        raise ValueError("context selection must not use metrics")
    if payload.get("evaluation_references_loaded_after_fixed_index_selection") is not True:
        raise ValueError("evaluation references were not frozen after index selection")
    if payload.get("evaluation_reference_artifacts_are_model_inputs") is not False:
        raise ValueError("evaluation reference artifacts must remain metric-only")
    if int(payload.get("context_count", -1)) != 8:
        raise ValueError("official small ReSplat requires exactly eight contexts")
    records = payload.get("records")
    windows = payload.get("context_windows")
    if not isinstance(records, list) or not isinstance(windows, list):
        raise ValueError("frozen bundle has no records/context windows")
    if not records or not windows:
        raise ValueError("frozen bundle must contain records and context windows")
    sources = [int(record["source_index"]) for record in records]
    if sources != sorted(set(sources)):
        raise ValueError("frozen records must be strictly source-index ordered")
    normalized_windows = [[int(value) for value in window] for window in windows]
    if any(len(window) != 8 for window in normalized_windows):
        raise ValueError("every official ReSplat context window must contain eight views")
    if set().union(*(set(window) for window in normalized_windows)) != set(sources):
        raise ValueError("context windows do not cover every frozen keyframe")
    expected_windows = chronological_context_windows(sources, context_count=8)
    if normalized_windows != expected_windows:
        raise ValueError("context windows are not the preregistered chronological partition")
    eval_sources = [int(value) for value in payload.get("eval_source_indices", [])]
    if eval_sources != sorted(set(eval_sources)):
        raise ValueError("evaluation sources must be sorted and unique")
    if not set(eval_sources).issubset(sources):
        raise ValueError("every evaluation source must be a frozen mapped keyframe")
    routes = {int(key): int(value) for key, value in payload.get("eval_routes", {}).items()}
    if set(routes) != set(eval_sources):
        raise ValueError("eval routes do not exactly cover declared evaluation sources")
    if any(route < 0 or route >= len(normalized_windows) for route in routes.values()):
        raise ValueError("eval route references a nonexistent submap")
    expected_routes = {
        source: route_source_index(source, normalized_windows) for source in eval_sources
    }
    if routes != expected_routes:
        raise ValueError("evaluation routes do not match the preregistered temporal router")
    for record in records:
        source = int(record["source_index"])
        if source in set(eval_sources) and not isinstance(
            record.get("evaluation_reference"), Mapping
        ):
            raise ValueError(f"evaluation source {source} has no frozen metric reference")
    return records, normalized_windows


def build_task_plan(
    *,
    bundle: Mapping[str, Any],
    workspace: Path,
    scene_dir: Path,
    task_root: Path,
    resplat_repo: Path,
    resplat_python: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    paired_runner: Path,
    paired_runner_sha256: str,
    scene_exporter: Path,
    scene_exporter_sha256: str,
    unblur_python: Path,
    scene_name: str,
    expected_source_count: int,
    expected_eval_count: int,
    expected_submap_count: int,
    resplat_commit: str,
    gpu_contract: Mapping[str, Any],
) -> dict[str, Any]:
    records, windows = validate_bundle(bundle)
    eval_sources = [int(value) for value in bundle["eval_source_indices"]]
    routes = {int(key): int(value) for key, value in bundle["eval_routes"].items()}
    if len(records) != int(expected_source_count):
        raise ValueError("frozen source-keyframe count differs from preregistration")
    if len(eval_sources) != int(expected_eval_count):
        raise ValueError("frozen evaluation count differs from preregistration")
    if len(windows) != int(expected_submap_count):
        raise ValueError("frozen ReSplat submap count differs from preregistration")
    if any(source not in windows[routes[source]] for source in eval_sources):
        raise ValueError("formal mapped-training-view query must be a context view")
    tasks = []
    for submap_id, contexts in enumerate(windows):
        aggregate_targets = [source for source in eval_sources if routes[source] == submap_id]
        runner_targets = aggregate_targets or [contexts[0]]
        output = task_root / f"submap_{submap_id:03d}"
        command = [
            str(resplat_python),
            str(paired_runner),
            "--scene-path", str(scene_dir),
            "--scene-manifest", str(scene_dir / "manifest.json"),
            "--resplat-repo", str(resplat_repo),
            "--checkpoint", str(checkpoint),
            "--expected-checkpoint-sha256", checkpoint_sha256,
            "--output-dir", str(output),
            "--model-preset", "dl3dv_8v_256x448_small",
            "--expected-resplat-commit", str(resplat_commit),
            "--device", "cuda:0",
            "--image-shape", "320", "448",
            "--expected-physical-index", str(gpu_contract["physical_index"]),
            "--expected-cuda-visible-devices", str(gpu_contract["visible_devices"]),
            "--expected-gpu-name", str(gpu_contract["name"]),
            "--expected-gpu-uuid", str(gpu_contract["uuid"]),
            "--expected-gpu-serial", str(gpu_contract["serial"]),
            "--expected-target-count", str(len(runner_targets)),
            "--max-save-images", str(len(runner_targets)),
            "--save-ply",
            "--target-reference-json", str(task_root / "references.json"),
            "--context-source-indices",
            *[str(value) for value in contexts],
            "--target-source-indices",
            *[str(value) for value in runner_targets],
        ]
        tasks.append(
            {
                "submap_id": submap_id,
                "context_source_indices": contexts,
                "aggregate_target_source_indices": aggregate_targets,
                "runner_target_source_indices": runner_targets,
                "probe_only_no_aggregate_target": not aggregate_targets,
                "context_target_overlap": sorted(set(contexts) & set(runner_targets)),
                "command": command,
                "output": str(output),
            }
        )
    all_sources = [int(record["source_index"]) for record in records]
    export_command = [
        str(unblur_python),
        str(scene_exporter),
        "--frames-csv", str(task_root / "frames.csv"),
        "--output-dir", str(scene_dir),
        "--indices", ",".join(str(value) for value in all_sources),
        "--image-mode", "evssm",
        "--images-json", str(task_root / "images.json"),
        "--resplat-repo", str(resplat_repo),
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", checkpoint_sha256,
        "--model-preset", "dl3dv_8v_256x448_small",
        "--formal-smoke",
    ]
    return {
        "schema": PLAN_SCHEMA,
        "scene_name": scene_name,
        "workspace": str(workspace.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "task_root": str(task_root.resolve()),
        "artifact_class": "independent_official_resplat_terminal_multisubmap",
        "active_map_merge": False,
        "reads_unblur_gaussian_state": False,
        "context_partition_uses_pixels_or_metrics": False,
        "formal_rgb_metric_resolution_hw": [320, 448],
        "formal_rgb_metric_note": (
            "R4 is scored against frozen metric references at the official ReSplat "
            "shape; U8/U12/U26 must be rescored from saved PNGs at the same shape"
        ),
        "depth_l1_formal_gate_available": False,
        "all_formal_queries_are_context_mapped_training_views": True,
        "formal_query_is_not_context_count": 0,
        "png_quantization": "official_resplat_mul255_astype_uint8_floor",
        "submap_count": len(tasks),
        "expected_submap_count": int(expected_submap_count),
        "source_keyframe_count": len(records),
        "expected_source_keyframe_count": int(expected_source_count),
        "eval_query_count": len(eval_sources),
        "expected_eval_query_count": int(expected_eval_count),
        "official_resplat": {
            "commit": str(resplat_commit),
            "repository": str(resplat_repo.resolve()),
            "python": str(resplat_python.resolve()),
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_sha256,
            },
            "runner": {
                "path": str(paired_runner.resolve()),
                "sha256": paired_runner_sha256,
            },
            "checkpoint_sha256": checkpoint_sha256,
            "runner_sha256": paired_runner_sha256,
            "model_preset": "dl3dv_8v_256x448_small",
            "context_count": 8,
            "recurrent_updates": 4,
        },
        "unblur_python": str(unblur_python.resolve()),
        "scene_exporter": {
            "path": str(scene_exporter.resolve()),
            "sha256": scene_exporter_sha256,
        },
        "gpu_contract": dict(gpu_contract),
        "required_child_environment": {
            "CUDA_VISIBLE_DEVICES": str(gpu_contract["visible_devices"]),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        },
        "export_command": export_command,
        "tasks": tasks,
    }


def materialize(args: argparse.Namespace) -> Path:
    bundle_path = args.bundle.expanduser().resolve()
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    records, _ = validate_bundle(payload)
    measurement_path = bundle_path.parents[1] / "measurement_manifest.json"
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    if (
        measurement.get("schema") != MEASUREMENT_SCHEMA
        or measurement.get("status") != "complete"
        or measurement.get("single_ordinary_26k_trajectory") is not True
        or measurement.get("milestones") != [8000, 12000, 26000]
    ):
        raise ValueError("pristine U trajectory is not complete; refusing to plan R4")
    destination = args.output_dir.expanduser().resolve()
    scene_dir = args.scene_dir.expanduser().resolve()
    lock_file = args.lock_file.expanduser().resolve()
    if not str(destination).startswith("/srv/"):
        raise ValueError("all materialized outputs must be under /srv")
    if not str(scene_dir).startswith("/srv/") or not str(lock_file).startswith("/srv/"):
        raise ValueError("scene output and GPU lock must both be under /srv")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    installed = False
    try:
        unblur_python = args.unblur_python.expanduser().resolve()
        resplat_python = args.resplat_python.expanduser().resolve()
        checkpoint = args.checkpoint.expanduser().resolve()
        paired_runner = args.paired_runner.expanduser().resolve()
        scene_exporter = args.scene_exporter.expanduser().resolve()
        gpu_wrapper = args.gpu_wrapper.expanduser().resolve()
        for label, path in (
            ("Unblur Python", unblur_python),
            ("ReSplat Python", resplat_python),
            ("checkpoint", checkpoint),
            ("paired runner", paired_runner),
            ("scene exporter", scene_exporter),
            ("GPU wrapper", gpu_wrapper),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} is missing: {path}")
        if sha256_file(checkpoint) != args.checkpoint_sha256:
            raise ValueError("official ReSplat checkpoint hash mismatch")
        if sha256_file(paired_runner) != args.paired_runner_sha256:
            raise ValueError("paired runner hash mismatch")
        if sha256_file(scene_exporter) != args.scene_exporter_sha256:
            raise ValueError("scene exporter hash mismatch")
        gpu_contract = {
            "physical_index": args.expected_physical_index,
            "visible_devices": args.expected_cuda_visible_devices,
            "logical_device": "cuda:0",
            "name": args.expected_gpu_name,
            "uuid": args.expected_gpu_uuid,
            "serial": args.expected_gpu_serial,
        }
        if payload.get("png_quantization") != "official_resplat_mul255_astype_uint8_floor":
            raise ValueError("frozen bundle does not use the common official PNG quantizer")
        bundle_gpu = payload.get("gpu_binding", {})
        if any(str(bundle_gpu.get(key)) != str(value) for key, value in gpu_contract.items()):
            raise ValueError("frozen bundle GPU identity differs from the formal contract")
        if any(
            str(measurement.get("gpu_binding", {}).get(key)) != str(value)
            for key, value in gpu_contract.items()
        ):
            raise ValueError("completed U trajectory used a different physical GPU")
        first = records[0]
        camera = {
            "width": int(first["width"]),
            "height": int(first["height"]),
            "K": first["intrinsics_px"],
        }
        images_payload = {"camera": camera, "frames": []}
        references_payload = {
            "schema": REFERENCE_SCHEMA,
            "camera": camera,
            "selection_fixed_before_reference_loading": True,
            "frames": [],
        }
        eval_sources = {int(value) for value in payload["eval_source_indices"]}
        fieldnames = [
            "index", "frame", "rgb_path", "timestamp",
            "tx", "ty", "tz", "qx", "qy", "qz", "qw",
            "fx", "fy", "cx", "cy", "pose_source", "uses_ground_truth_pose",
        ]
        with (staging / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                png = Path(record["png"]).resolve()
                if sha256_file(png) != record["png_sha256"]:
                    raise ValueError(f"frozen PNG hash mismatch: {png}")
                if [int(record["width"]), int(record["height"])] != [camera["width"], camera["height"]]:
                    raise ValueError("frozen records do not share one processed camera size")
                if not np.allclose(record["intrinsics_px"], camera["K"], atol=1e-9, rtol=0):
                    raise ValueError("frozen records do not share one processed K")
                c2w = np.asarray(record["c2w_opencv"], dtype=np.float64)
                quaternion = rotation_to_xyzw(c2w[:3, :3])
                source = int(record["source_index"])
                writer.writerow(
                    {
                        "index": source,
                        "frame": str(png),
                        "rgb_path": str(png),
                        "timestamp": source,
                        "tx": c2w[0, 3], "ty": c2w[1, 3], "tz": c2w[2, 3],
                        "qx": quaternion[0], "qy": quaternion[1],
                        "qz": quaternion[2], "qw": quaternion[3],
                        "fx": camera["K"][0][0], "fy": camera["K"][1][1],
                        "cx": camera["K"][0][2], "cy": camera["K"][1][2],
                        "pose_source": "droid_final_ba_not_gt_not_eval_aligned",
                        "uses_ground_truth_pose": "false",
                    }
                )
                images_payload["frames"].append(
                    {"source_index": source, "path": str(png), "sha256": record["png_sha256"]}
                )
                if source in eval_sources:
                    reference = record["evaluation_reference"]
                    reference_png = Path(reference["png"]).resolve()
                    reference_hash = str(reference["png_sha256"])
                    aggregate = True
                else:
                    # Empty-target submaps still run one explicit probe so every
                    # planned official model invocation is auditable.  Probe
                    # metrics are excluded from the formal aggregate.
                    reference_png = png
                    reference_hash = str(record["png_sha256"])
                    aggregate = False
                if sha256_file(reference_png) != reference_hash:
                    raise ValueError(f"frozen metric-reference hash mismatch: {reference_png}")
                references_payload["frames"].append(
                    {
                        "source_index": source,
                        "path": str(reference_png),
                        "sha256": reference_hash,
                        "included_in_formal_aggregate": aggregate,
                    }
                )
        (staging / "images.json").write_text(
            json.dumps(images_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "references.json").write_text(
            json.dumps(references_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        plan = build_task_plan(
            bundle=payload,
            workspace=args.workspace.expanduser().resolve(),
            scene_dir=scene_dir,
            task_root=destination,
            resplat_repo=args.resplat_repo.expanduser().resolve(),
            resplat_python=resplat_python,
            checkpoint=checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            paired_runner=paired_runner,
            paired_runner_sha256=args.paired_runner_sha256,
            scene_exporter=scene_exporter,
            scene_exporter_sha256=args.scene_exporter_sha256,
            unblur_python=unblur_python,
            scene_name=args.scene_name,
            expected_source_count=args.expected_source_count,
            expected_eval_count=args.expected_eval_count,
            expected_submap_count=args.expected_submap_count,
            resplat_commit=args.resplat_commit,
            gpu_contract=gpu_contract,
        )
        plan["source_bundle"] = {"path": str(bundle_path), "sha256": sha256_file(bundle_path)}
        plan["source_unblur_measurement"] = {
            "path": str(measurement_path),
            "sha256": sha256_file(measurement_path),
        }
        common_evaluator = (
            args.workspace.expanduser().resolve()
            / "scripts"
            / "evaluate_offline_fair_common_rgb.py"
        )
        sequential_executor = (
            args.workspace.expanduser().resolve()
            / "scripts"
            / "execute_offline_resplat_plan.py"
        )
        materializer = Path(__file__).resolve()
        plan["protocol_tooling"] = {
            "materialize_frozen_bundle": {
                "path": str(materializer),
                "sha256": sha256_file(materializer),
            },
            "scene_exporter": {
                "path": str(scene_exporter),
                "sha256": sha256_file(scene_exporter),
            },
            "common_rgb_evaluator": {
                "path": str(common_evaluator),
                "sha256": sha256_file(common_evaluator),
            },
            "sequential_wall_executor": {
                "path": str(sequential_executor),
                "sha256": sha256_file(sequential_executor),
            },
            "pinned_gpu_wrapper": {
                "path": str(gpu_wrapper),
                "sha256": sha256_file(gpu_wrapper),
            },
        }
        plan["exclusive_lock"] = str(lock_file)
        common_evaluator_command = [
            str(resplat_python),
            str(common_evaluator),
            "--bundle", str(bundle_path),
            "--resplat-plan", str(destination / "plan.json"),
            "--execution-report", str(destination / "execution_audit" / "execution.json"),
            "--workspace", str(args.workspace.expanduser().resolve()),
            "--resplat-repo", str(args.resplat_repo.expanduser().resolve()),
            "--output-dir", str(destination / "common_rgb_metrics"),
            "--device", "cuda:0",
            "--expected-physical-index", str(gpu_contract["physical_index"]),
            "--expected-cuda-visible-devices", str(gpu_contract["visible_devices"]),
            "--expected-gpu-name", str(gpu_contract["name"]),
            "--expected-gpu-uuid", str(gpu_contract["uuid"]),
            "--expected-gpu-serial", str(gpu_contract["serial"]),
        ]
        plan["common_rgb_evaluation_command"] = [
            str(resplat_python),
            str(gpu_wrapper),
            "--lock-file", str(lock_file),
            "--audit-report", str(destination / "common_rgb_gpu_execution.json"),
            "--expected-physical-index", str(gpu_contract["physical_index"]),
            "--expected-cuda-visible-devices", str(gpu_contract["visible_devices"]),
            "--expected-gpu-name", str(gpu_contract["name"]),
            "--expected-gpu-uuid", str(gpu_contract["uuid"]),
            "--expected-gpu-serial", str(gpu_contract["serial"]),
            "--",
            *common_evaluator_command,
        ]
        plan["sequential_execution_command"] = [
            str(resplat_python),
            str(sequential_executor),
            "--plan", str(destination / "plan.json"),
            "--report-dir", str(destination / "execution_audit"),
            "--temporary-root", str(destination / "tmp"),
            "--lock-file", str(lock_file),
            "--expected-physical-index", str(gpu_contract["physical_index"]),
            "--expected-cuda-visible-devices", str(gpu_contract["visible_devices"]),
            "--expected-gpu-name", str(gpu_contract["name"]),
            "--expected-gpu-uuid", str(gpu_contract["uuid"]),
            "--expected-gpu-serial", str(gpu_contract["serial"]),
            "--execute",
        ]
        (staging / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing concurrent overwrite of {destination}")
        os.rename(staging, destination)
        installed = True
        return destination
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--expected-source-count", type=int, required=True)
    parser.add_argument("--expected-eval-count", type=int, required=True)
    parser.add_argument("--expected-submap-count", type=int, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--resplat-repo", type=Path, required=True)
    parser.add_argument("--resplat-python", type=Path, required=True)
    parser.add_argument("--unblur-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--paired-runner", type=Path, required=True)
    parser.add_argument("--paired-runner-sha256", required=True)
    parser.add_argument("--scene-exporter", type=Path, required=True)
    parser.add_argument("--scene-exporter-sha256", required=True)
    parser.add_argument("--resplat-commit", required=True)
    parser.add_argument("--gpu-wrapper", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--expected-physical-index", type=int, required=True)
    parser.add_argument("--expected-cuda-visible-devices", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-gpu-serial", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        output = materialize(parse_args(argv))
    except (FileExistsError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"offline ReSplat task plan materialized atomically at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
