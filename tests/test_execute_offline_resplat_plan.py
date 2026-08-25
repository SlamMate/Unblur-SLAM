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
SCRIPT = ROOT / "scripts" / "execute_offline_resplat_plan.py"
SPEC = importlib.util.spec_from_file_location("execute_offline_resplat_plan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExecutePlanValidationTest(unittest.TestCase):
    def test_validate_only_rejects_unbound_command_without_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = Path("/srv/offline-fair-test/submap_000")
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "schema": MODULE.PLAN_SCHEMA,
                        "artifact_class": "independent_official_resplat_terminal_multisubmap",
                        "active_map_merge": False,
                        "reads_unblur_gaussian_state": False,
                        "all_formal_queries_are_context_mapped_training_views": True,
                        "formal_query_is_not_context_count": 0,
                        "submap_count": 1,
                        "expected_submap_count": 1,
                        "tasks": [
                            {
                                "submap_id": 0,
                                "output": str(output),
                                "probe_only_no_aggregate_target": False,
                                "context_source_indices": list(range(8)),
                                "aggregate_target_source_indices": [0],
                                "runner_target_source_indices": [0],
                                "command": [
                                    "/definitely/not/executed",
                                    "--output-dir", str(output),
                                    "--target-reference-json", "/srv/references.json",
                                    "--device", "cuda:0",
                                    "--context-source-indices", "0", "1", "2", "3",
                                    "4", "5", "6", "7",
                                    "--target-source-indices", "0",
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = MODULE.parse_args(["--plan", str(plan_path)])
            with mock.patch.object(MODULE.subprocess, "run") as run_mock:
                with self.assertRaises(ValueError):
                    MODULE.execute(args)
            run_mock.assert_not_called()

    def test_chronological_tail_and_router_are_deterministic(self) -> None:
        windows = MODULE._chronological_context_windows(list(range(18)))
        self.assertEqual(
            windows,
            [list(range(8)), list(range(8, 16)), list(range(10, 18))],
        )
        self.assertEqual(MODULE._route_source_index(10, windows), 1)


if __name__ == "__main__":
    unittest.main()
