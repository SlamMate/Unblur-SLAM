#!/usr/bin/env python3
"""CPU/source contracts for the 221-frame official TURTLE-only SLAM arm."""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_fr2_official_turtle_smoke.py"
SPEC = importlib.util.spec_from_file_location("fr2_official_turtle_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


class Fr2OfficialTurtleSmokeTests(unittest.TestCase):
    def test_repository_config_is_turtle_only_221_and_uniform100(self) -> None:
        cfg = SMOKE._load()
        SMOKE._validate_static_contract(cfg)
        self.assertEqual(cfg["deblur"]["frontend"], "turtle_streaming")
        self.assertEqual(cfg["max_frames"], 221)
        self.assertEqual(cfg["mapping"]["final_refine_iters"], 100)
        self.assertFalse(cfg["mapping"]["resplat"]["enabled"])
        self.assertFalse(cfg["mapping"]["resplat"]["online_enabled"])
        self.assertEqual(cfg["mapping"]["resplat"]["extra_iters"], 0)
        self.assertEqual(cfg["evssm_checkpoint"], "")
        self.assertEqual(cfg["deblur"]["causal_checkpoint"], "")
        self.assertEqual(
            tuple(cfg["evaluation"]["expected_clear_gt_source_indices"]),
            SMOKE.EXPECTED_PREFIX,
        )

    def test_rejects_enabling_old_replay_causal_or_evssm(self) -> None:
        base = SMOKE._load()
        for mutate in (
            lambda cfg: cfg["mapping"]["resplat"].update(enabled=True),
            lambda cfg: cfg["mapping"]["resplat"].update(online_enabled=True),
            lambda cfg: cfg["mapping"]["resplat"].update(extra_iters=1),
            lambda cfg: cfg["deblur"].update(frontend="causal_evssm"),
            lambda cfg: cfg["deblur"].update(causal_checkpoint="adapter.pt"),
            lambda cfg: cfg.update(evssm_checkpoint="evssm.pth"),
        ):
            cfg = copy.deepcopy(base)
            mutate(cfg)
            with self.assertRaises(ValueError):
                SMOKE._validate_static_contract(cfg)

    def test_rejects_compute_or_metric_scope_drift(self) -> None:
        base = SMOKE._load()
        variants = []
        changed = copy.deepcopy(base)
        changed["mapping"]["final_refine_iters"] = 101
        variants.append(changed)
        changed = copy.deepcopy(base)
        changed["max_frames"] = 42
        variants.append(changed)
        changed = copy.deepcopy(base)
        changed["evaluation"]["expected_clear_gt_source_indices"] = [0]
        variants.append(changed)
        for cfg in variants:
            with self.assertRaises(ValueError):
                SMOKE._validate_static_contract(cfg)

    def test_artifact_validator_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weight.bin"
            path.write_bytes(b"official bytes")
            digest = SMOKE.sha256_file(path)
            self.assertEqual(
                SMOKE._require_artifact(path, digest, digest, "test artifact"),
                path.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "not pinned"):
                SMOKE._require_artifact(path, "0" * 64, digest, "test artifact")

    def test_launcher_source_does_not_import_old_refinement_or_causal_code(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any("resplat_replay" in name for name in imported))
        self.assertFalse(any("causal" in name for name in imported))
        self.assertFalse(any("EVSSM" in name for name in imported))

    def test_official_hashes_and_gpu_mapping_are_fixed_in_source(self) -> None:
        cfg = SMOKE._load()
        deblur = cfg["deblur"]
        self.assertEqual(deblur["turtle_repo_commit"], SMOKE.PINNED_TURTLE_COMMIT)
        self.assertEqual(
            deblur["turtle_config_sha256"], SMOKE.PINNED_TURTLE_CONFIG_SHA256
        )
        self.assertEqual(
            deblur["turtle_checkpoint_sha256"], SMOKE.PINNED_TURTLE_CHECKPOINT_SHA256
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"CUDA_VISIBLE_DEVICES": "1"', source)
        self.assertIn('"CUDA_DEVICE_ORDER": "PCI_BUS_ID"', source)


if __name__ == "__main__":
    unittest.main()
