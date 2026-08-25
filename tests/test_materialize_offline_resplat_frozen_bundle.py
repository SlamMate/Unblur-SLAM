#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_offline_resplat_frozen_bundle.py"
SPEC = importlib.util.spec_from_file_location("materialize_offline_resplat_frozen_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FrozenBundleMaterializerTest(unittest.TestCase):
    def test_identity_rotation_is_xyzw_identity(self) -> None:
        self.assertTrue(
            np.allclose(MODULE.rotation_to_xyzw(np.eye(3)), [0.0, 0.0, 0.0, 1.0])
        )

    def test_plan_keeps_resplat_independent_and_fills_probe_target(self) -> None:
        sources = list(range(8))
        records = [{"source_index": value} for value in sources]
        bundle = {
            "schema": MODULE.BUNDLE_SCHEMA,
            "ground_truth_used_for_context_selection": False,
            "metric_used_for_context_selection": False,
            "evaluation_references_loaded_after_fixed_index_selection": True,
            "evaluation_reference_artifacts_are_model_inputs": False,
            "context_count": 8,
            "records": records,
            "context_windows": [sources],
            "eval_source_indices": [],
            "eval_routes": {},
        }
        plan = MODULE.build_task_plan(
            bundle=bundle,
            workspace=Path("/workspace"),
            scene_dir=Path("/srv/scene"),
            task_root=Path("/srv/tasks"),
            resplat_repo=Path("/srv/resplat"),
            resplat_python=Path("/srv/resplat-python"),
            checkpoint=Path("/srv/checkpoint"),
            checkpoint_sha256="a" * 64,
            paired_runner=Path("/workspace/runner.py"),
            paired_runner_sha256="b" * 64,
            scene_exporter=Path("/workspace/exporter.py"),
            scene_exporter_sha256="d" * 64,
            unblur_python=Path("/srv/unblur-python"),
            scene_name="synthetic",
            expected_source_count=8,
            expected_eval_count=0,
            expected_submap_count=1,
            resplat_commit="c" * 40,
            gpu_contract={
                "physical_index": 1,
                "visible_devices": "1",
                "logical_device": "cuda:0",
                "name": "NVIDIA RTX A6000",
                "uuid": "GPU-test",
                "serial": "123",
            },
        )
        self.assertFalse(plan["active_map_merge"])
        self.assertFalse(plan["reads_unblur_gaussian_state"])
        self.assertEqual(plan["tasks"][0]["runner_target_source_indices"], [0])
        self.assertTrue(plan["tasks"][0]["probe_only_no_aggregate_target"])
        self.assertIn("--target-reference-json", plan["tasks"][0]["command"])
        self.assertEqual(plan["formal_rgb_metric_resolution_hw"], [320, 448])
        self.assertFalse(plan["depth_l1_formal_gate_available"])
        self.assertTrue(plan["all_formal_queries_are_context_mapped_training_views"])
        self.assertEqual(plan["expected_submap_count"], 1)
        self.assertEqual(plan["gpu_contract"]["uuid"], "GPU-test")
        self.assertIn("--expected-resplat-commit", plan["tasks"][0]["command"])
        self.assertEqual(plan["tasks"][0]["command"][-1], "0")
        indices_position = plan["export_command"].index("--indices")
        self.assertEqual(
            plan["export_command"][indices_position + 1],
            "0,1,2,3,4,5,6,7",
        )


if __name__ == "__main__":
    unittest.main()
