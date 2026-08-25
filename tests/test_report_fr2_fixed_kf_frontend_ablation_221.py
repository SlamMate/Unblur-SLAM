#!/usr/bin/env python3
"""CPU-only contracts for the fixed-keyframe paired result auditor."""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/report_fr2_fixed_kf_frontend_ablation_221.py"
SPEC = importlib.util.spec_from_file_location("fixed_kf_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


EXPECTED = REPORT.EXPECTED_FIXED_SOURCE_KEYFRAMES


def _fixed_config() -> dict:
    return {
        "tracking": {
            "fixed_source_keyframes": {
                "enabled": True,
                "schema": "unblur_slam.fixed_source_keyframes.v1",
                "coordinate_domain": "dataset_source_index",
                "strict_exact": True,
                "selection_source": "frozen_prior_evssm_baseline_schedule",
                "runtime_baseline_artifact_dependency": False,
                "uses_ground_truth_poses": False,
                "source_indices": list(EXPECTED),
            }
        },
        "fixed_kf_frontend_ablation_221": {
            "schema": "unblur_slam.fr2_xyz_fixed_kf_frontend_ablation_221.v1",
            "conditional_on_frozen_evssm_baseline_schedule": True,
            "shares_pose_estimates_between_arms": False,
            "uses_ground_truth_poses": False,
            "runtime_baseline_artifact_dependency": False,
            "provenance_verified_during_cpu_preflight": True,
            "frozen_baseline_video_npz": str(REPORT.FROZEN_BASELINE_VIDEO),
            "frozen_baseline_video_npz_sha256": REPORT.FROZEN_BASELINE_VIDEO_SHA256,
        },
    }


def _guard() -> dict:
    return {
        "physical_gpu": 1,
        "uuid": "GPU-test",
        "name": "NVIDIA RTX A6000",
        "memory_total_mib": 49140,
        "memory_used_mib": 2,
        "utilization_percent_snapshot": 0,
        "compute_processes": [],
        "max_idle_memory_mib": 64,
        "passed": True,
    }


class FixedKeyframeReportTests(unittest.TestCase):
    def test_fixed_config_requires_exact_nonleaky_schedule(self) -> None:
        parsed = REPORT._fixed_contract_from_config(_fixed_config(), "baseline")
        self.assertEqual(tuple(parsed["fixed"]["source_indices"]), EXPECTED)
        for key, value in (
            ("source_indices", list(EXPECTED) + [206]),
            ("uses_ground_truth_poses", True),
            ("runtime_baseline_artifact_dependency", True),
        ):
            cfg = copy.deepcopy(_fixed_config())
            cfg["tracking"]["fixed_source_keyframes"][key] = value
            with self.assertRaises(REPORT.ContractError):
                REPORT._fixed_contract_from_config(cfg, "baseline")

    def test_gpu_guard_is_fail_closed(self) -> None:
        valid = {"execution": {"gpu_free_guard_before_launch": _guard()}}
        self.assertTrue(REPORT._gpu_guard(valid, "baseline")["passed"])
        for mutation in (
            {"memory_used_mib": 65},
            {"compute_processes": [{"pid": 1}]},
            {"passed": False},
        ):
            bad = _guard()
            bad.update(mutation)
            with self.assertRaises(REPORT.ContractError):
                REPORT._gpu_guard(
                    {"execution": {"gpu_free_guard_before_launch": bad}},
                    "baseline",
                )

    def test_normalization_removes_only_arm_specific_guard(self) -> None:
        preflight = {
            "schema": "fixed",
            "execution": {
                "selected_arms": ["baseline", "turtle"],
                "gpu_free_guard_before_launch": _guard(),
            },
        }
        normalized = REPORT._normalized_preflight(preflight)
        self.assertNotIn("gpu_free_guard_before_launch", normalized["execution"])
        self.assertIn("gpu_free_guard_before_launch", preflight["execution"])
        self.assertEqual(normalized["schema"], "fixed")

    def test_incomplete_outputs_fail_without_creating_report_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "missing_run"
            report_dir = Path(directory) / "report"
            with self.assertRaises(REPORT.ContractError):
                REPORT.build_report(root)
            self.assertFalse(report_dir.exists())

    def test_script_is_cpu_only_and_default_output_is_separate(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertNotIn("torch", imports)
        args = REPORT.parse_args([])
        self.assertEqual(args.root, REPORT.DEFAULT_ROOT)
        self.assertEqual(args.output_dir, REPORT.DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
