#!/usr/bin/env python3
"""CPU-only launch contracts for the online three-arm main experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/run_fr2_online_mip_resplat_main_v1.py"
SPEC = importlib.util.spec_from_file_location("online_mip_main", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class OnlineMipMainContracts(unittest.TestCase):
    def test_formal_preflight_and_exact_differences(self):
        result = RUNNER.preflight()
        self.assertTrue(result["passed"])
        self.assertFalse(result["gpu_started"])
        self.assertTrue(result["output_root_absent"])
        self.assertEqual(
            set(result["resolved_differences"]["mip_splatting"]),
            {"data.output", "mapping.mip_splatting.enabled"},
        )
        self.assertEqual(
            set(
                result["resolved_differences"][
                    "mip_splatting_plus_safe_resplat"
                ]
            ),
            {
                "data.output",
                "mapping.mip_splatting.enabled",
                "mapping.official_resplat_active_fusion.enabled",
            },
        )

    def test_commands_use_shared_lock_and_physical_gpu1(self):
        for arm, cfg in RUNNER.ARMS:
            command = RUNNER._wrapper_command(arm, cfg)
            self.assertIn(str(RUNNER.LOCK), command)
            self.assertEqual(
                command[command.index("--expected-cuda-visible-devices") + 1],
                "1",
            )
            self.assertEqual(command[-1], str(cfg))


if __name__ == "__main__":
    unittest.main()
