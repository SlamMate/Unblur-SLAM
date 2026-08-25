#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "report_offline_resplat_fair_26k.py"
SPEC = importlib.util.spec_from_file_location("report_offline_resplat_fair_26k", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FinalOfflineReportTest(unittest.TestCase):
    def test_build_report_validates_three_scene_end_to_end_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            gpu = {
                "physical_index": 1,
                "visible_devices": "1",
                "logical_device": "cuda:0",
                "name": "NVIDIA RTX A6000",
                "uuid": "GPU-test",
                "serial": "123",
            }
            official = {
                "commit": "c" * 40,
                "checkpoint_sha256": "d" * 64,
                "runner_sha256": "e" * 64,
                "preset": "dl3dv_8v_256x448_small",
            }
            gpu_lock = "/srv/test/offline-fair-gpu1.lock"
            wrapper_contract = {
                "path": "/repo/scripts/execute_pinned_gpu_command.py",
                "sha256": "f" * 64,
            }
            executor_contract = {
                "path": "/repo/scripts/execute_offline_resplat_plan.py",
                "sha256": "a" * 64,
            }
            exporter_contract = {
                "path": "/repo/scripts/export_tum_official_resplat_scene.py",
                "sha256": "b" * 64,
            }
            unblur_contract = {
                "python": "/env/unblur-python",
                "worktree": "/worktree",
            }
            scene_contracts = []
            for scene_number in range(3):
                name = f"scene_{scene_number}"
                scene_contracts.append(
                    {
                        "name": name,
                        "config": f"configs/local/offline_resplat_fair_26k_v1/{name}.yaml",
                        "eval_count": 1,
                        "expected_keyframe_count": 8,
                        "expected_resplat_submaps": 1,
                    }
                )
                unblur = output / "unblur_pristine" / name / "offline_fair_26k"
                tasks_root = output / "resplat_tasks" / name
                measurement = unblur / "measurement_manifest.json"
                bundle = unblur / "frozen_inputs" / "bundle.json"
                write_json(
                    measurement,
                    {
                        "schema": MODULE.MEASUREMENT_SCHEMA,
                        "status": "complete",
                        "single_ordinary_26k_trajectory": True,
                        "milestones": list(MODULE.MILESTONES),
                        "official_resplat_executed_in_process": False,
                        "residual_replay_executed": False,
                        "optimizer_cumulative_seconds_excluding_measurement": 26.0,
                        "optimizer_peak_allocated_bytes_excluding_measurement": 100,
                        "gpu_binding": gpu,
                    },
                )
                u_command = [
                    unblur_contract["python"],
                    "run.py",
                    f"configs/local/offline_resplat_fair_26k_v1/{name}.yaml",
                ]
                pinned_common = {
                    "schema": MODULE.PINNED_GPU_COMMAND_SCHEMA,
                    "status": "complete",
                    "returncode": 0,
                    "wrapper": wrapper_contract,
                    "command": u_command,
                    "command_sha256": MODULE._argv_sha256(u_command),
                    "working_directory": unblur_contract["worktree"],
                    "full_child_process_wall_seconds": 30.0,
                    "sequential_execution": True,
                    "exclusive_lock": gpu_lock,
                    "child_environment": {
                        "CUDA_VISIBLE_DEVICES": "1",
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    },
                    "gpu": {
                        **gpu,
                        "idle_verified_before_launch": True,
                        "active_compute_pids_before_launch": [],
                    },
                }
                write_json(
                    output / "gpu_audits" / f"unblur_{name}.json", pinned_common
                )
                write_json(
                    bundle,
                    {
                        "png_quantization": "official_resplat_mul255_astype_uint8_floor",
                        "mapped_keyframe_count": 8,
                        "eval_source_indices": [0],
                        "context_windows": [list(range(8))],
                        "gpu_binding": gpu,
                    },
                )
                for milestone in MODULE.MILESTONES:
                    write_json(
                        unblur
                        / "milestones"
                        / f"iter_{milestone:06d}"
                        / "metrics.json",
                        {
                            "schema": MODULE.MILESTONE_SCHEMA,
                            "iteration": milestone,
                            "same_ordinary_26k_trajectory": True,
                            "png_quantization": "official_resplat_mul255_astype_uint8_floor",
                            "num_frames": 1,
                            "optimizer_cumulative_seconds_excluding_measurement": milestone / 1000,
                            "optimizer_peak_allocated_bytes_excluding_measurement": 100,
                            "gpu_binding": gpu,
                        },
                    )
                r4 = tasks_root / "submap_000"
                task_command = ["/env/resplat-python", "/repo/runner.py", "--synthetic"]
                scene_manifest_path = output / "resplat_scenes" / name / "manifest.json"
                scene_manifest_sha256 = "9" * 64
                rendered = r4 / "paired_refine4" / "rendered" / "00000000.png"
                rendered.parent.mkdir(parents=True, exist_ok=True)
                rendered.write_bytes(b"render")
                manifest = r4 / "run_manifest.json"
                write_json(
                    manifest,
                    {
                        "schema": MODULE.RUNNER_SCHEMA,
                        "runner": {"sha256": official["runner_sha256"]},
                        "official_resplat": {
                            "repository": {"commit": official["commit"]},
                            "checkpoint": {"sha256": official["checkpoint_sha256"]},
                            "model_preset": official["preset"],
                            "num_context": 8,
                            "num_refine": 4,
                        },
                        "image_shape": [320, 448],
                        "scene": {
                            "manifest_path": str(scene_manifest_path),
                            "manifest_sha256": scene_manifest_sha256,
                        },
                        "gpu_binding": gpu,
                        "selection": {
                            "context_source_indices": list(range(8)),
                            "target_source_indices": [0],
                        },
                        "paired_contract": {"target_rgb_passed_to_forward_update": False},
                        "metrics": {
                            "reference": {"passed_to_encoder_or_forward_update": False}
                        },
                        "terminal_reconstruction": {
                            "primary": True,
                            "wall_seconds": 1.0,
                            "peak_allocated_bytes": 10,
                            "wall_scope": "terminal_backend_setup_plus_core_no_metrics_or_artifact_io",
                            "peak_scope": "process_setup_plus_core_before_metrics_and_artifact_io",
                            "excludes": [
                                "metric computation",
                                "output PNG/PLY artifact I/O",
                            ],
                        },
                        "outputs": {
                            "paired_refine4_rendered": [
                                {
                                    "source_index": 0,
                                    "relative_path": str(rendered.relative_to(r4)),
                                    "sha256": MODULE.sha256_file(rendered),
                                    "quantization": "official_resplat_mul255_astype_uint8_floor",
                                }
                            ]
                        },
                    },
                )
                plan = tasks_root / "plan.json"
                common_child = [
                    "/env/resplat-python",
                    "/repo/scripts/evaluate_offline_fair_common_rgb.py",
                    "--output-dir",
                    str(tasks_root / "common_rgb_metrics"),
                ]
                write_json(
                    plan,
                    {
                        "schema": MODULE.PLAN_SCHEMA,
                        "scene_name": name,
                        "workspace": "/repo",
                        "task_root": str(tasks_root),
                        "scene_dir": str(output / "resplat_scenes" / name),
                        "source_keyframe_count": 8,
                        "eval_query_count": 1,
                        "submap_count": 1,
                        "all_formal_queries_are_context_mapped_training_views": True,
                        "formal_query_is_not_context_count": 0,
                        "active_map_merge": False,
                        "reads_unblur_gaussian_state": False,
                        "gpu_contract": gpu,
                        "source_bundle": {
                            "path": str(bundle),
                            "sha256": MODULE.sha256_file(bundle),
                        },
                        "source_unblur_measurement": {
                            "path": str(measurement),
                            "sha256": MODULE.sha256_file(measurement),
                        },
                        "common_rgb_evaluation_command": [
                            "/env/resplat-python",
                            wrapper_contract["path"],
                            "--lock-file",
                            gpu_lock,
                            "--",
                            *common_child,
                        ],
                        "tasks": [
                            {
                                "submap_id": 0,
                                "context_source_indices": list(range(8)),
                                "aggregate_target_source_indices": [0],
                                "runner_target_source_indices": [0],
                                "output": str(r4),
                                "command": task_command,
                            }
                        ],
                    },
                )
                execution = tasks_root / "execution_audit" / "execution.json"
                write_json(
                    execution,
                    {
                        "schema": MODULE.EXECUTION_SCHEMA,
                        "executor": executor_contract,
                        "status": "complete",
                        "fresh_process_per_submap": True,
                        "sequential_execution": True,
                        "submap_count": 1,
                        "completed_submaps": 1,
                        "plan": {"sha256": MODULE.sha256_file(plan)},
                        "gpu_binding": {
                            **gpu,
                            "idle_verified_before_launch": True,
                            "active_compute_pids_before_launch": [],
                        },
                        "exclusive_lock": gpu_lock,
                        "child_environment": {
                            "CUDA_VISIBLE_DEVICES": "1",
                            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        },
                        "primary_terminal_reconstruction_seconds": 1.0,
                        "primary_terminal_peak_allocated_bytes": 10,
                        "sequential_total_full_process_wall_seconds_secondary": 2.0,
                        "executor_elapsed_wall_seconds_secondary": 2.5,
                        "tasks": [
                            {
                                "submap_id": 0,
                                "command_sha256": MODULE._argv_sha256(task_command),
                                "returncode": 0,
                                "primary_terminal_reconstruction_seconds": 1.0,
                                "primary_terminal_peak_allocated_bytes": 10,
                                "full_process_wall_seconds_secondary": 2.0,
                                "run_manifest_sha256": MODULE.sha256_file(manifest),
                            }
                        ],
                    },
                )
                one_view = [{"name": "00000000.png", "psnr": 30.0, "ssim": 0.9, "lpips": 0.1}]
                metrics = {
                    arm: {"mean": {"psnr": 30.0, "ssim": 0.9, "lpips": 0.1}, "per_view": one_view}
                    for arm in MODULE.ARMS
                }
                common = tasks_root / "common_rgb_metrics" / "report.json"
                write_json(
                    common,
                    {
                        "schema": MODULE.COMMON_SCHEMA,
                        "common_resolution_hw": [320, 448],
                        "resize": "official_cvg_resplat_PIL_LANCZOS",
                        "metric_implementation": "official_cvg_resplat_compute_metrics",
                        "prediction_source": "saved_quantized_png_for_every_arm",
                        "frozen_reference_artifacts_passed_to_model": False,
                        "depth_l1_formal_gate": {"available": False},
                        "scope": {"query_is_context_count": 1, "query_is_not_context_count": 0},
                        "gpu_binding": gpu,
                        "inputs": {
                            "bundle": {"sha256": MODULE.sha256_file(bundle)},
                            "plan": {"sha256": MODULE.sha256_file(plan)},
                            "execution": {"sha256": MODULE.sha256_file(execution)},
                        },
                        "metrics": metrics,
                        "noninferiority": {
                            "thresholds": {
                                "psnr_drop_db_max": 0.1,
                                "ssim_drop_max": 0.005,
                                "lpips_increase_max": 0.005,
                            },
                            "gates": {"psnr": True, "ssim": True, "lpips": True}
                        },
                    },
                )
                common_audit = {
                    **pinned_common,
                    "command": common_child,
                    "command_sha256": MODULE._argv_sha256(common_child),
                    "working_directory": "/repo",
                }
                write_json(tasks_root / "common_rgb_gpu_execution.json", common_audit)
            contract = root / "contract.json"
            write_json(
                contract,
                {
                    "schema": MODULE.CONTRACT_SCHEMA,
                    "assets": {"output_root": str(output)},
                    "unblur_baseline": unblur_contract,
                    "protocol_tooling": {
                        "pinned_gpu_wrapper": wrapper_contract,
                        "sequential_wall_executor": executor_contract,
                        "scene_exporter": exporter_contract,
                    },
                    "execution_hardware": {"gpu": gpu, "exclusive_lock": gpu_lock},
                    "official_resplat": {**official, "python": "/env/resplat-python"},
                    "reporting": {
                        "noninferiority": {
                            "psnr_drop_db_max": 0.1,
                            "ssim_drop_max": 0.005,
                            "lpips_increase_max": 0.005,
                        },
                        "expected_total_resplat_submaps": 3,
                        "aggregate_eval_count": 3,
                    },
                    "scenes": scene_contracts,
                },
            )
            provenance = mock.patch.object(
                MODULE, "_validate_static_provenance", return_value={"verified": True}
            )
            exported_scene = mock.patch.object(
                MODULE,
                "_validate_exported_scene",
                side_effect=lambda **kwargs: (
                    output
                    / "resplat_scenes"
                    / kwargs["plan"]["scene_name"]
                    / "manifest.json",
                    "9" * 64,
                ),
            )
            with provenance, mock.patch.object(
                MODULE, "_validate_plan_against_contract"
            ), exported_scene:
                report = MODULE.build_report(contract)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["aggregate"]["evaluation_view_count"], 3)
            self.assertEqual(report["aggregate"]["fresh_serial_resplat_subprocesses"], 3)
            self.assertFalse(report["speedup_claim"]["allowed"])

            # A syntactically valid wrapper audit for an unrelated child must
            # not be usable as post-hoc proof that the formal U run held the lock.
            audit_path = output / "gpu_audits" / "unblur_scene_0.json"
            dummy = json.loads(audit_path.read_text(encoding="utf-8"))
            dummy["command"] = ["/bin/true"]
            dummy["command_sha256"] = MODULE._argv_sha256(dummy["command"])
            write_json(audit_path, dummy)
            with mock.patch.object(
                MODULE, "_validate_static_provenance", return_value={"verified": True}
            ), mock.patch.object(
                MODULE, "_validate_plan_against_contract"
            ), mock.patch.object(
                MODULE,
                "_validate_exported_scene",
                side_effect=lambda **kwargs: (
                    output
                    / "resplat_scenes"
                    / kwargs["plan"]["scene_name"]
                    / "manifest.json",
                    "9" * 64,
                ),
            ), self.assertRaisesRegex(ValueError, "expected child"):
                MODULE.build_report(contract)


if __name__ == "__main__":
    unittest.main()
