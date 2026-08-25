#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_offline_fair_common_rgb.py"
SPEC = importlib.util.spec_from_file_location("evaluate_offline_fair_common_rgb", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CommonRgbPairTest(unittest.TestCase):
    def test_collects_exact_u_and_independent_r4_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "scene" / "offline_fair_26k"
            frozen = benchmark / "frozen_inputs"
            frozen.mkdir(parents=True)
            records = []
            for source in (10, 20):
                reference = frozen / f"reference_{source}.png"
                reference.write_bytes(b"reference")
                records.append(
                    {
                        "source_index": source,
                        "evaluation_reference": {
                            "png": str(reference),
                            "png_sha256": MODULE.sha256_file(reference),
                        },
                    }
                )
                for milestone in MODULE.MILESTONES:
                    prediction = (
                        benchmark
                        / "milestones"
                        / f"iter_{milestone:06d}"
                        / "renders"
                        / f"{source:08d}.png"
                    )
                    prediction.parent.mkdir(parents=True, exist_ok=True)
                    prediction.write_bytes(b"unblur")
            for milestone in MODULE.MILESTONES:
                milestone_root = benchmark / "milestones" / f"iter_{milestone:06d}"
                (milestone_root / "metrics.json").write_text(
                    json.dumps(
                        {
                            "per_frame": [
                                {
                                    "source_index": source,
                                    "render_png_sha256": MODULE.sha256_file(
                                        milestone_root / "renders" / f"{source:08d}.png"
                                    ),
                                }
                                for source in (10, 20)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            bundle = frozen / "bundle.json"
            bundle.write_text(
                json.dumps(
                    {
                        "schema": MODULE.BUNDLE_SCHEMA,
                        "eval_source_indices": [10, 20],
                        "records": records,
                    }
                ),
                encoding="utf-8",
            )
            measurement = root / "measurement.json"
            measurement.write_text(
                json.dumps(
                    {
                        "schema": MODULE.MEASUREMENT_SCHEMA,
                        "status": "complete",
                        "single_ordinary_26k_trajectory": True,
                        "milestones": list(MODULE.MILESTONES),
                    }
                ),
                encoding="utf-8",
            )
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
                "model_preset": "dl3dv_8v_256x448_small",
            }
            r4 = root / "r4"
            for source in (10, 20):
                prediction = (
                    r4 / "paired_refine4" / "rendered" / f"{source:08d}.png"
                )
                prediction.parent.mkdir(parents=True, exist_ok=True)
                prediction.write_bytes(b"resplat")
            artifacts = []
            for source in (10, 20):
                prediction = r4 / "paired_refine4" / "rendered" / f"{source:08d}.png"
                artifacts.append(
                    {
                        "source_index": source,
                        "relative_path": str(prediction.relative_to(r4)),
                        "sha256": MODULE.sha256_file(prediction),
                        "quantization": "official_resplat_mul255_astype_uint8_floor",
                    }
                )
            manifest_path = r4 / "run_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": MODULE.RUNNER_SCHEMA,
                        "runner": {"sha256": official["runner_sha256"]},
                        "official_resplat": {
                            "repository": {"commit": official["commit"]},
                            "checkpoint": {"sha256": official["checkpoint_sha256"]},
                            "model_preset": official["model_preset"],
                            "num_context": 8,
                            "num_refine": 4,
                        },
                        "image_shape": [320, 448],
                        "gpu_binding": gpu,
                        "selection": {
                            "context_source_indices": [10, 11, 12, 13, 14, 15, 16, 20],
                            "target_source_indices": [10, 20],
                        },
                        "paired_contract": {
                            "target_rgb_passed_to_forward_update": False
                        },
                        "terminal_reconstruction": {
                            "primary": True,
                            "wall_seconds": 1.0,
                            "peak_allocated_bytes": 1,
                            "wall_scope": "terminal_backend_setup_plus_core_no_metrics_or_artifact_io",
                            "peak_scope": "process_setup_plus_core_before_metrics_and_artifact_io",
                            "excludes": [
                                "metric computation",
                                "output PNG/PLY artifact I/O",
                            ],
                        },
                        "metrics": {
                            "reference": {
                                "passed_to_encoder_or_forward_update": False
                            }
                        },
                        "outputs": {"paired_refine4_rendered": artifacts},
                    }
                ),
                encoding="utf-8",
            )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema": MODULE.PLAN_SCHEMA,
                        "active_map_merge": False,
                        "reads_unblur_gaussian_state": False,
                        "png_quantization": "official_resplat_mul255_astype_uint8_floor",
                        "all_formal_queries_are_context_mapped_training_views": True,
                        "formal_rgb_metric_resolution_hw": [320, 448],
                        "depth_l1_formal_gate_available": False,
                        "source_bundle": {
                            "path": str(bundle),
                            "sha256": MODULE.sha256_file(bundle),
                        },
                        "source_unblur_measurement": {
                            "path": str(measurement),
                            "sha256": MODULE.sha256_file(measurement),
                        },
                        "official_resplat": official,
                        "gpu_contract": gpu,
                        "tasks": [
                            {
                                "submap_id": 0,
                                "context_source_indices": [10, 11, 12, 13, 14, 15, 16, 20],
                                "aggregate_target_source_indices": [10, 20],
                                "runner_target_source_indices": [10, 20],
                                "output": str(r4),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            execution = root / "execution.json"
            execution.write_text(
                json.dumps(
                    {
                        "schema": MODULE.EXECUTION_SCHEMA,
                        "status": "complete",
                        "plan": {
                            "path": str(plan),
                            "sha256": MODULE.sha256_file(plan),
                        },
                        "fresh_process_per_submap": True,
                        "sequential_execution": True,
                        "submap_count": 1,
                        "completed_submaps": 1,
                        "tasks": [
                            {
                                "submap_id": 0,
                                "returncode": 0,
                                "run_manifest_sha256": MODULE.sha256_file(manifest_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            arms, scope = MODULE.collect_prediction_reference_pairs(bundle, plan, execution)
            self.assertEqual(set(arms), {"U8", "U12", "U26", "R4-multisubmap"})
            self.assertEqual(len(arms["R4-multisubmap"]), 2)
            self.assertEqual(scope["query_is_context_count"], 2)
            self.assertEqual(scope["query_is_not_context_count"], 0)


if __name__ == "__main__":
    unittest.main()
