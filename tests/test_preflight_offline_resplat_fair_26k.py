#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_offline_resplat_fair_26k.py"
SPEC = importlib.util.spec_from_file_location("preflight_offline_resplat_fair_26k", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


class OfflineFairPreflightTest(unittest.TestCase):
    def test_context_windows_cover_short_tail_without_padding_ids(self) -> None:
        windows = PREFLIGHT.chronological_context_windows(list(range(18)))
        self.assertEqual(windows, [list(range(8)), list(range(8, 16)), list(range(10, 18))])
        self.assertEqual(set().union(*(set(window) for window in windows)), set(range(18)))

    def test_contract_rejects_fake_fused_arm(self) -> None:
        payload = {
            "schema": PREFLIGHT.SCHEMA,
            "claim_boundary": {
                "official_resplat_refines_existing_unblur_map": False,
                "official_resplat_reads_unblur_gaussian_or_optimizer_state": False,
                "active_map_merge": False,
                "single_global_resplat_map": False,
                "residual_replay_is_official_resplat": False,
            },
            "unblur_baseline": {
                "milestones_from_one_trajectory": [8000, 12000, 26000],
                "legacy_bpn": False,
                "residual_replay": False,
            },
            "official_resplat": {"context_count": 8, "recurrent_updates": 4},
            "reporting": {"combined_U_plus_R_arm": "unsafe_fake_fusion"},
            "scenes": [{}, {}, {}],
        }
        errors = PREFLIGHT.validate_contract(payload)
        self.assertIn("a fused U+R arm is forbidden without a trained adapter", errors)


if __name__ == "__main__":
    unittest.main()
