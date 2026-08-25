#!/usr/bin/env python3
"""Standard-library CPU contracts for the direct 221-frame paired launcher."""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_fr2_official_online_budget_paired_221.py"
SPEC = importlib.util.spec_from_file_location("paired_221", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PAIRED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PAIRED
SPEC.loader.exec_module(PAIRED)


def _arm(frontend: str, output: Path) -> dict:
    turtle = frontend == "turtle_streaming"
    return {
        "dataset": "tumrgbd",
        "scene": "freiburg2_xyz",
        "max_frames": 221,
        "stride": 1,
        "setup_seed": 43,
        "device": "cuda:0",
        "warmup_mapper": True,
        "clear_init": False,
        "cam": {"W_out": 512, "H_out": 384},
        "data": {"output": str(output)},
        "evssm_checkpoint": "" if turtle else "/weights/evssm.pth",
        "evssm_checkpoint_sha256": "" if turtle else PAIRED.EXPECTED_SHA256["evssm"],
        "framecrafter": {"enabled": False},
        "submaps": {
            "enabled": False,
            "official_resplat_sidecar": {"enabled": False},
        },
        "evaluation": {
            "clear_gt_scope": "prefix_smoke",
            "expected_clear_gt_source_indices": list(PAIRED.EXPECTED_PREFIX),
        },
        "mapping": {
            "online_plotting": False,
            "eval_before_final_ba": False,
            "hydrate_missing_droid_keyframes": True,
            "final_refine_iters": 100,
            "Training": {
                "init_itr_num": 1050,
                "mapping_itr_num": 100,
                "tracking_itr_num": 100,
            },
            "resplat": {"enabled": False, "online_enabled": False, "extra_iters": 0},
        },
        "deblur": {
            "frontend": frontend,
            "causal_checkpoint": "",
            "stream_every_frame": True,
            "stream_apply_to_tracking": True,
            "stream_replace_sharp": False,
            "stream_min_laplacian_gain": 0.02,
            "turtle_inference_precision": "fp16" if turtle else "fp32",
            "turtle_repo": "/official/TURTLE" if turtle else "",
            "turtle_repo_commit": "pinned" if turtle else "",
            "turtle_config": "/official/Turtle.yml" if turtle else "",
            "turtle_config_sha256": "config-sha" if turtle else "",
            "turtle_checkpoint": "/weights/GoPro.pth" if turtle else "",
            "turtle_checkpoint_sha256": "checkpoint-sha" if turtle else "",
        },
        "paired_official_online_budget_221": {
            "schema": "unblur_slam.fr2_xyz_paired_official_online_budget_221.v1",
            "official_online_optimization_budget": True,
            "complete_three_sequence_paper_benchmark": False,
            "paper_26k_offline_refinement": False,
        },
    }


class PairedOfficialOnlineBudget221Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = _arm("evssm", PAIRED.OUTPUTS["baseline"])
        self.turtle = _arm("turtle_streaming", PAIRED.OUTPUTS["turtle"])

    def test_valid_pair_differs_only_in_declared_frontend_fields(self) -> None:
        differences = PAIRED._validate_pair_contract(self.baseline, self.turtle)
        self.assertEqual(
            set(differences),
            {
                "data.output",
                "deblur.frontend",
                "deblur.turtle_checkpoint",
                "deblur.turtle_checkpoint_sha256",
                "deblur.turtle_config",
                "deblur.turtle_config_sha256",
                "deblur.turtle_inference_precision",
                "deblur.turtle_repo",
                "deblur.turtle_repo_commit",
                "evssm_checkpoint",
                "evssm_checkpoint_sha256",
            },
        )

    def test_rejects_online_budget_or_final_budget_drift(self) -> None:
        for path, value in (
            (("mapping", "Training", "init_itr_num"), 100),
            (("mapping", "Training", "mapping_itr_num"), 10),
            (("mapping", "Training", "tracking_itr_num"), 10),
            (("mapping", "final_refine_iters"), 26000),
        ):
            changed = copy.deepcopy(self.turtle)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                PAIRED._validate_pair_contract(self.baseline, changed)

    def test_rejects_replay_causal_framecrafter_or_submaps(self) -> None:
        mutations = (
            lambda cfg: cfg["mapping"]["resplat"].update(enabled=True),
            lambda cfg: cfg["deblur"].update(causal_checkpoint="adapter.pt"),
            lambda cfg: cfg["framecrafter"].update(enabled=True),
            lambda cfg: cfg["submaps"].update(enabled=True),
        )
        for mutate in mutations:
            changed = copy.deepcopy(self.turtle)
            mutate(changed)
            with self.assertRaises(ValueError):
                PAIRED._validate_pair_contract(self.baseline, changed)

    def test_rejects_any_unlisted_cross_arm_change(self) -> None:
        changed = copy.deepcopy(self.turtle)
        changed["mapping"]["Training"]["window_size"] = 99
        self.baseline["mapping"]["Training"]["window_size"] = 10
        with self.assertRaisesRegex(ValueError, "outside frontend"):
            PAIRED._validate_pair_contract(self.baseline, changed)

    def test_import_and_default_cli_are_gpu_inert(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        top_level_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_level_imports.append(node.module or "")
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("src.turtle_backend", top_level_imports)
        args = PAIRED.parse_args([])
        self.assertFalse(args.run)
        self.assertEqual(args.arm, "all")

    def test_repository_configs_disclose_bounded_nonpaper_scope(self) -> None:
        common = (
            ROOT
            / "configs/local/fr2_xyz_official_online_budget_221/common.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("init_itr_num: 1050", common)
        self.assertIn("mapping_itr_num: 100", common)
        self.assertIn("tracking_itr_num: 100", common)
        self.assertIn("final_refine_iters: 100", common)
        self.assertIn("complete_three_sequence_paper_benchmark: false", common)
        self.assertIn("paper_26k_offline_refinement: false", common)


if __name__ == "__main__":
    unittest.main()

