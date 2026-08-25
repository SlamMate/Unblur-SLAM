#!/usr/bin/env python3
"""CPU contracts for the fixed-11KF EVSSM/TURTLE ablation."""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_fr2_fixed_kf_frontend_ablation_221.py"
SPEC = importlib.util.spec_from_file_location("fixed_kf_221", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ABLATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ABLATION
SPEC.loader.exec_module(ABLATION)

from src.utils.fixed_keyframes import (  # noqa: E402
    FIXED_SOURCE_KEYFRAME_SCHEMA,
    assert_exact_fixed_source_keyframes,
    parse_fixed_source_keyframe_contract,
)


EXPECTED = ABLATION.EXPECTED_FIXED_SOURCE_KEYFRAMES


def _contract(indices=EXPECTED) -> dict:
    return {
        "tracking": {
            "fixed_source_keyframes": {
                "enabled": True,
                "schema": FIXED_SOURCE_KEYFRAME_SCHEMA,
                "coordinate_domain": "dataset_source_index",
                "strict_exact": True,
                "selection_source": "frozen_prior_evssm_baseline_schedule",
                "runtime_baseline_artifact_dependency": False,
                "uses_ground_truth_poses": False,
                "source_indices": list(indices),
            }
        }
    }


class FixedKeyframeFrontendAblation221Tests(unittest.TestCase):
    def test_contract_accepts_exact_preregistered_schedule(self) -> None:
        parsed = parse_fixed_source_keyframe_contract(_contract())
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["source_indices"], EXPECTED)
        self.assertIn(153, range(221))
        self.assertNotIn(153, parsed["source_index_set"])
        self.assertNotIn(206, parsed["source_index_set"])

    def test_contract_rejects_ambiguous_or_leaky_schedules(self) -> None:
        mutations = (
            lambda cfg: cfg["tracking"]["fixed_source_keyframes"].update(
                strict_exact=False
            ),
            lambda cfg: cfg["tracking"]["fixed_source_keyframes"].update(
                runtime_baseline_artifact_dependency=True
            ),
            lambda cfg: cfg["tracking"]["fixed_source_keyframes"].update(
                uses_ground_truth_poses=True
            ),
            lambda cfg: cfg["tracking"]["fixed_source_keyframes"].update(
                source_indices=[0, 9, 9]
            ),
            lambda cfg: cfg["tracking"]["fixed_source_keyframes"].update(
                source_indices=[9, 15]
            ),
        )
        for mutate in mutations:
            cfg = copy.deepcopy(_contract())
            mutate(cfg)
            with self.assertRaises(ValueError):
                parse_fixed_source_keyframe_contract(cfg)

    def test_postrun_check_is_order_and_membership_exact(self) -> None:
        assert_exact_fixed_source_keyframes(EXPECTED, EXPECTED)
        for actual in (
            EXPECTED + (206,),
            tuple(value for value in EXPECTED if value != 166),
            tuple(reversed(EXPECTED)),
        ):
            with self.assertRaises(RuntimeError):
                assert_exact_fixed_source_keyframes(EXPECTED, actual)

    def test_resolved_repository_configs_share_only_one_schedule(self) -> None:
        configs, fixed = ABLATION._load_and_validate_configs()
        self.assertEqual(fixed["source_indices"], EXPECTED)
        differences = ABLATION.BASE._pair_differences(
            configs["baseline"], configs["turtle"]
        )
        self.assertLessEqual(set(differences), ABLATION.BASE.ALLOWED_PAIR_DIFFS)
        self.assertTrue(
            {"data.output", "deblur.frontend", "evssm_checkpoint"}.issubset(
                differences
            )
        )
        self.assertFalse(
            configs["baseline"]["fixed_kf_frontend_ablation_221"][
                "shares_pose_estimates_between_arms"
            ]
        )

    def test_runner_import_and_default_cli_are_gpu_inert(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertNotIn("torch", imports)
        args = ABLATION.parse_args([])
        self.assertFalse(args.run)
        self.assertEqual(args.arm, "all")

    def test_run_path_guards_gpu_before_atomic_output_creation(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        run_pair = source[source.index("def run_pair"):]
        self.assertLess(
            run_pair.index("_assert_physical_gpu_free()"),
            run_pair.index("_run_arm(arm, arm_audit)"),
        )
        run_arm = source[source.index("def _run_arm"):source.index("def _selected_arms")]
        self.assertIn("output.mkdir(parents=True, exist_ok=False)", run_arm)

    def test_motion_filter_has_fail_closed_fixed_policy_branch(self) -> None:
        source = (ROOT / "thirdparty/glorie_slam/motion_filter.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("if fixed_keyframe_policy:", source)
        self.assertIn("is_keyframe = is_tracking_anchor", source)
        self.assertIn("and motion_keyframe", source)


if __name__ == "__main__":
    unittest.main()
