#!/usr/bin/env python3
"""Score U8/U12/U26 and R4 PNG artifacts at one official 320x448 protocol.

This evaluator never runs or modifies either reconstruction backend.  It loads
only already-saved prediction PNGs and metric-only frozen references, applies
the pinned official ReSplat PIL/Lanczos preprocessing, and calls the official
PSNR/SSIM/LPIPS implementation for every arm.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from offline_fair_gpu_contract import validate_gpu_contract


BUNDLE_SCHEMA = "unblur_slam.offline_fair_frozen_bundle.v1"
PLAN_SCHEMA = "unblur_slam.offline_resplat_multisubmap_plan.v1"
EXECUTION_SCHEMA = "unblur_slam.offline_resplat_execution.v1"
RUNNER_SCHEMA = "unblur_slam.paired_official_resplat_smoke.v1"
MEASUREMENT_SCHEMA = "unblur_slam.offline_fair_26k_measurement.v1"
REPORT_SCHEMA = "unblur_slam.offline_fair_common_rgb_metrics.v1"
COMMON_HEIGHT = 320
COMMON_WIDTH = 448
MILESTONES = (8000, 12000, 26000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def collect_prediction_reference_pairs(
    bundle_path: Path, plan_path: Path, execution_path: Path
) -> tuple[dict[str, list[tuple[Path, Path, str]]], dict[str, Any]]:
    bundle = _load_json(bundle_path, "frozen bundle")
    plan = _load_json(plan_path, "ReSplat plan")
    execution = _load_json(execution_path, "ReSplat execution report")
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("wrong frozen-bundle schema")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("wrong ReSplat-plan schema")
    if execution.get("schema") != EXECUTION_SCHEMA or execution.get("status") != "complete":
        raise ValueError("ReSplat execution is not formally complete")
    if plan.get("active_map_merge") is not False or plan.get(
        "reads_unblur_gaussian_state"
    ) is not False:
        raise ValueError("formal R4 arm must remain an independent terminal backend")
    if plan.get("formal_rgb_metric_resolution_hw") != [COMMON_HEIGHT, COMMON_WIDTH]:
        raise ValueError("plan does not preregister the common 320x448 RGB shape")
    if plan.get("depth_l1_formal_gate_available") is not False:
        raise ValueError("R4 depth-L1 must not be advertised as an available formal gate")
    if plan.get("png_quantization") != "official_resplat_mul255_astype_uint8_floor":
        raise ValueError("plan does not use the common official PNG quantizer")
    if plan.get("all_formal_queries_are_context_mapped_training_views") is not True:
        raise ValueError("formal evaluation scope is not all-context mapped training views")
    source_bundle = plan.get("source_bundle", {})
    if Path(str(source_bundle.get("path", ""))).resolve() != bundle_path.resolve():
        raise ValueError("plan points to a different frozen bundle")
    if source_bundle.get("sha256") != sha256_file(bundle_path):
        raise ValueError("frozen bundle changed after plan materialization")
    source_measurement = plan.get("source_unblur_measurement", {})
    measurement_path = Path(str(source_measurement.get("path", ""))).resolve()
    if source_measurement.get("sha256") != sha256_file(measurement_path):
        raise ValueError("U trajectory manifest changed after plan materialization")
    measurement = _load_json(measurement_path, "U trajectory manifest")
    if (
        measurement.get("schema") != MEASUREMENT_SCHEMA
        or measurement.get("status") != "complete"
        or measurement.get("single_ordinary_26k_trajectory") is not True
        or measurement.get("milestones") != list(MILESTONES)
    ):
        raise ValueError("U8/U12/U26 are not a complete single trajectory")
    execution_plan = execution.get("plan", {})
    if Path(str(execution_plan.get("path", ""))).resolve() != plan_path.resolve():
        raise ValueError("execution report points to a different plan")
    if execution_plan.get("sha256") != sha256_file(plan_path):
        raise ValueError("plan changed after ReSplat execution")
    tasks = plan.get("tasks", [])
    if (
        execution.get("fresh_process_per_submap") is not True
        or execution.get("sequential_execution") is not True
        or int(execution.get("completed_submaps", -1)) != len(tasks)
        or int(execution.get("submap_count", -1)) != len(tasks)
        or len(execution.get("tasks", [])) != len(tasks)
    ):
        raise ValueError("execution report does not prove complete fresh serial submaps")
    execution_by_id = {
        int(record["submap_id"]): record for record in execution["tasks"]
    }
    if set(execution_by_id) != set(range(len(tasks))):
        raise ValueError("execution report has missing or duplicate submap IDs")

    eval_sources = [int(value) for value in bundle.get("eval_source_indices", [])]
    if not eval_sources or eval_sources != sorted(set(eval_sources)):
        raise ValueError("evaluation sources must be non-empty, sorted, and unique")
    records = {int(record["source_index"]): record for record in bundle["records"]}
    references: dict[int, Path] = {}
    for source in eval_sources:
        try:
            reference_record = records[source]["evaluation_reference"]
            reference = Path(reference_record["png"]).resolve()
            reference_sha256 = str(reference_record["png_sha256"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"evaluation source {source} lacks a frozen reference") from error
        if not reference.is_file():
            raise FileNotFoundError(f"frozen metric reference is missing: {reference}")
        if sha256_file(reference) != reference_sha256:
            raise ValueError(f"frozen metric-reference hash mismatch: {reference}")
        references[source] = reference

    benchmark_root = bundle_path.resolve().parents[1]
    arms: dict[str, list[tuple[Path, Path, str]]] = {}
    for milestone in MILESTONES:
        arm = f"U{milestone // 1000}"
        milestone_root = benchmark_root / "milestones" / f"iter_{milestone:06d}"
        milestone_metrics = _load_json(
            milestone_root / "metrics.json", f"{arm} native milestone manifest"
        )
        declared_predictions = {
            int(record["source_index"]): str(record["render_png_sha256"])
            for record in milestone_metrics.get("per_frame", [])
        }
        if set(declared_predictions) != set(eval_sources):
            raise ValueError(f"{arm} manifest does not exactly cover evaluation sources")
        pairs = []
        for source in eval_sources:
            prediction = (
                milestone_root
                / "renders"
                / f"{source:08d}.png"
            )
            if not prediction.is_file():
                raise FileNotFoundError(f"{arm} prediction is missing: {prediction}")
            if sha256_file(prediction) != declared_predictions[source]:
                raise ValueError(f"{arm} prediction hash mismatch: {prediction}")
            pairs.append((prediction, references[source], f"{source:08d}.png"))
        arms[arm] = pairs

    routed: dict[int, tuple[Path, bool]] = {}
    official_contract = plan.get("official_resplat", {})
    gpu_contract = plan.get("gpu_contract", {})
    for task in tasks:
        contexts = {int(value) for value in task["context_source_indices"]}
        output = Path(task["output"]).resolve()
        manifest_path = output / "run_manifest.json"
        execution_record = execution_by_id[int(task["submap_id"])]
        if execution_record.get("returncode") != 0:
            raise ValueError("R4 execution contains a failed subprocess")
        if execution_record.get("run_manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("R4 run manifest changed after execution")
        run_manifest = _load_json(manifest_path, "R4 run manifest")
        if run_manifest.get("schema") != RUNNER_SCHEMA:
            raise ValueError("wrong R4 run-manifest schema")
        expected_runner_targets = [
            int(value) for value in task["runner_target_source_indices"]
        ]
        observed_runner_targets = [
            int(value)
            for value in run_manifest.get("selection", {}).get(
                "target_source_indices", []
            )
        ]
        if observed_runner_targets != expected_runner_targets:
            raise ValueError("R4 run-manifest targets disagree with preregistered plan")
        observed_contexts = [
            int(value)
            for value in run_manifest.get("selection", {}).get(
                "context_source_indices", []
            )
        ]
        if observed_contexts != [int(value) for value in task["context_source_indices"]]:
            raise ValueError("R4 run-manifest contexts disagree with preregistered plan")
        if run_manifest.get("paired_contract", {}).get(
            "target_rgb_passed_to_forward_update"
        ) is not False:
            raise ValueError("R4 run passed target RGB into recurrent refinement")
        reference_contract = run_manifest.get("metrics", {}).get("reference", {})
        if reference_contract.get("passed_to_encoder_or_forward_update") is not False:
            raise ValueError("R4 metric references were exposed to the model")
        observed_official = run_manifest.get("official_resplat", {})
        if (
            observed_official.get("repository", {}).get("commit")
            != official_contract.get("commit")
            or observed_official.get("checkpoint", {}).get("sha256")
            != official_contract.get("checkpoint_sha256")
            or observed_official.get("model_preset")
            != official_contract.get("model_preset")
            or int(observed_official.get("num_context", -1)) != 8
            or int(observed_official.get("num_refine", -1)) != 4
            or run_manifest.get("runner", {}).get("sha256")
            != official_contract.get("runner_sha256")
            or run_manifest.get("image_shape") != [COMMON_HEIGHT, COMMON_WIDTH]
        ):
            raise ValueError("R4 runner/model/checkpoint/shape differs from the frozen plan")
        observed_gpu = run_manifest.get("gpu_binding", {})
        if any(
            str(observed_gpu.get(key)) != str(value)
            for key, value in gpu_contract.items()
        ):
            raise ValueError("R4 run used a different physical GPU")
        terminal = run_manifest.get("terminal_reconstruction", {})
        if (
            terminal.get("primary") is not True
            or terminal.get("wall_scope")
            != "terminal_backend_setup_plus_core_no_metrics_or_artifact_io"
            or terminal.get("peak_scope")
            != "process_setup_plus_core_before_metrics_and_artifact_io"
            or float(terminal.get("wall_seconds", 0.0)) <= 0.0
            or int(terminal.get("peak_allocated_bytes", 0)) <= 0
            or "metric computation" not in terminal.get("excludes", [])
            or "output PNG/PLY artifact I/O" not in terminal.get("excludes", [])
        ):
            raise ValueError("R4 primary timing/peak boundary is invalid")
        artifacts = {
            int(record["source_index"]): record
            for record in run_manifest.get("outputs", {}).get(
                "paired_refine4_rendered", []
            )
        }
        if set(artifacts) != set(expected_runner_targets):
            raise ValueError("R4 manifest does not hash every saved target render")
        for source, record in artifacts.items():
            artifact_path = output / str(record["relative_path"])
            if (
                record.get("quantization")
                != "official_resplat_mul255_astype_uint8_floor"
                or not artifact_path.is_file()
                or sha256_file(artifact_path) != record.get("sha256")
            ):
                raise ValueError(f"R4 rendered artifact integrity failure: {source}")
        for source in task["aggregate_target_source_indices"]:
            source = int(source)
            if source in routed:
                raise ValueError(f"R4 aggregate target is routed more than once: {source}")
            prediction = output / str(artifacts[source]["relative_path"])
            routed[source] = (prediction, source in contexts)
    if set(routed) != set(eval_sources):
        raise ValueError("R4 aggregate routes do not exactly cover evaluation sources")
    r4_pairs = []
    context_count = 0
    for source in eval_sources:
        prediction, is_context = routed[source]
        if not prediction.is_file():
            raise FileNotFoundError(f"R4 prediction is missing: {prediction}")
        context_count += int(is_context)
        r4_pairs.append((prediction, references[source], f"{source:08d}.png"))
    arms["R4-multisubmap"] = r4_pairs
    if context_count != len(eval_sources):
        raise ValueError("formal protocol requires every mapped-training query to be context")
    return arms, {
        "eval_source_indices": eval_sources,
        "query_is_context_count": context_count,
        "query_is_not_context_count": 0,
        "evaluation_scope": "all_context_mapped_training_keyframes_not_novel_views",
        "execution_report": {
            "path": str(execution_path.resolve()),
            "sha256": sha256_file(execution_path),
        },
    }


def _import_runner(workspace: Path) -> Any:
    script = workspace / "scripts" / "run_paired_official_resplat_smoke.py"
    specification = importlib.util.spec_from_file_location(
        "_offline_fair_paired_runner", script
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import paired runner: {script}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def evaluate(args: argparse.Namespace) -> Path:
    bundle = args.bundle.expanduser().resolve()
    plan = args.resplat_plan.expanduser().resolve()
    execution = args.execution_report.expanduser().resolve()
    arms, scope = collect_prediction_reference_pairs(bundle, plan, execution)
    destination = args.output_dir.expanduser().resolve()
    if not str(destination).startswith("/srv/"):
        raise ValueError("common RGB metric output must be under /srv")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    installed = False
    try:
        expected_gpu = {
            "physical_index": args.expected_physical_index,
            "visible_devices": args.expected_cuda_visible_devices,
            "logical_device": "cuda:0",
            "name": args.expected_gpu_name,
            "uuid": args.expected_gpu_uuid,
            "serial": args.expected_gpu_serial,
        }
        plan_payload = _load_json(plan, "ReSplat plan")
        if any(
            str(plan_payload.get("gpu_contract", {}).get(key)) != str(value)
            for key, value in expected_gpu.items()
        ):
            raise ValueError("evaluator GPU arguments disagree with the frozen plan")
        gpu_binding = validate_gpu_contract(
            expected_gpu, require_visible_mask=True, require_idle=False
        )
        runner = _import_runner(args.workspace.expanduser().resolve())
        repo_record = runner.inspect_official_repo(args.resplat_repo)
        if repo_record.get("commit") != plan_payload.get("official_resplat", {}).get("commit"):
            raise ValueError("metric evaluator sees a different ReSplat commit")
        official = runner.import_official_infer(args.resplat_repo)
        import torch
        metric_module = importlib.import_module("src.evaluation.metrics")

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested common RGB metric device has no CUDA")
        metrics: dict[str, Any] = {}
        for arm, pairs in arms.items():
            predictions = official.load_and_preprocess_images(
                [str(item[0]) for item in pairs], COMMON_HEIGHT, COMMON_WIDTH
            ).to(args.device)
            references = official.load_and_preprocess_images(
                [str(item[1]) for item in pairs], COMMON_HEIGHT, COMMON_WIDTH
            ).to(args.device)
            names = [item[2] for item in pairs]
            arm_dir = staging / arm
            arm_dir.mkdir()
            with torch.no_grad():
                psnr = metric_module.compute_psnr(references, predictions)
                ssim = metric_module.compute_ssim(references, predictions)
                lpips_parts = [
                    metric_module.compute_lpips(
                        references[start : start + 4], predictions[start : start + 4]
                    )
                    for start in range(0, len(pairs), 4)
                ]
                lpips = torch.cat(lpips_parts)
            result = {
                "mean": {
                    "psnr": float(psnr.mean().item()),
                    "ssim": float(ssim.mean().item()),
                    "lpips": float(lpips.mean().item()),
                },
                "per_view": [
                    {
                        "name": name,
                        "psnr": float(psnr[index].item()),
                        "ssim": float(ssim[index].item()),
                        "lpips": float(lpips[index].item()),
                    }
                    for index, name in enumerate(names)
                ],
            }
            (arm_dir / "metrics.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            metrics[arm] = result

        baseline = metrics["U26"]["mean"]
        candidate = metrics["R4-multisubmap"]["mean"]
        gates = {
            "psnr": baseline["psnr"] - candidate["psnr"] <= args.psnr_drop_db_max,
            "ssim": baseline["ssim"] - candidate["ssim"] <= args.ssim_drop_max,
            "lpips": candidate["lpips"] - baseline["lpips"]
            <= args.lpips_increase_max,
        }
        report = {
            "schema": REPORT_SCHEMA,
            "common_resolution_hw": [COMMON_HEIGHT, COMMON_WIDTH],
            "resize": "official_cvg_resplat_PIL_LANCZOS",
            "metric_implementation": "official_cvg_resplat_compute_metrics",
            "official_resplat_repository": repo_record,
            "gpu_binding": gpu_binding,
            "inputs": {
                "bundle": {"path": str(bundle), "sha256": sha256_file(bundle)},
                "plan": {"path": str(plan), "sha256": sha256_file(plan)},
                "execution": {
                    "path": str(execution), "sha256": sha256_file(execution)
                },
            },
            "prediction_source": "saved_quantized_png_for_every_arm",
            "frozen_reference_artifacts_passed_to_model": False,
            "depth_l1_formal_gate": {
                "available": False,
                "reason": "R4 has no audited raw metric-depth artifact in this protocol",
            },
            "scope": scope,
            "metrics": metrics,
            "noninferiority": {
                "thresholds": {
                    "psnr_drop_db_max": args.psnr_drop_db_max,
                    "ssim_drop_max": args.ssim_drop_max,
                    "lpips_increase_max": args.lpips_increase_max,
                },
                "gates": gates,
                "passed": all(gates.values()),
            },
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, destination)
        installed = True
        return destination
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--resplat-plan", type=Path, required=True)
    parser.add_argument("--execution-report", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--resplat-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-physical-index", type=int, required=True)
    parser.add_argument("--expected-cuda-visible-devices", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-gpu-serial", required=True)
    parser.add_argument("--psnr-drop-db-max", type=float, default=0.1)
    parser.add_argument("--ssim-drop-max", type=float, default=0.005)
    parser.add_argument("--lpips-increase-max", type=float, default=0.005)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        output = evaluate(parse_args(argv))
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"common 320x448 RGB metrics saved atomically at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
