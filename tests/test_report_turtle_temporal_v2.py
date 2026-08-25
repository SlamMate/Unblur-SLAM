#!/usr/bin/env python3
"""CPU-only fail-closed contracts for the TURTLE temporal v2 reporter."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_turtle_temporal_v2 import (  # noqa: E402
    _load_addendum,
    _load_contract,
    _load_reporting_fix,
    _parse_seed_metrics,
    _seed_statistics,
)
from scripts.report_turtle_history_smoke import HistorySmokeContractError  # noqa: E402


def _contract(*, room2_read: bool = False) -> dict:
    return {
        "schema": "unblur_slam.turtle_replica_temporal_order_multiseed.v2",
        "status": "preregistered_before_gpu_training",
        "protocol": {
            "future_frames_used": False,
            "room2_frame_pixels_read_before_validation_pass": room2_read,
            "room2_metrics_evaluated_before_validation_pass": False,
            "room2_manifest_bytes_already_read": True,
        },
    }


class TurtleTemporalV2ReportTest(unittest.TestCase):
    def test_seed_metrics_parser_requires_unique_explicit_seed_bindings(self) -> None:
        parsed = _parse_seed_metrics(["17=/srv/a.json", "42=/srv/b.json"])
        self.assertEqual(parsed, {17: Path("/srv/a.json"), 42: Path("/srv/b.json")})
        with self.assertRaisesRegex(ValueError, "SEED="):
            _parse_seed_metrics(["/srv/a.json"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _parse_seed_metrics(["17=/srv/a.json", "17=/srv/b.json"])

    def test_seed_statistics_are_population_statistics(self) -> None:
        result = _seed_statistics([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertAlmostEqual(
            result["population_std"], (2.0 / 3.0) ** 0.5, places=12
        )
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 3.0)

    def test_contract_refuses_room2_read_before_all_seed_validation_passes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = root / "accepted.json"
            accepted.write_text(json.dumps(_contract()) + "\n", encoding="utf-8")
            payload, source, digest = _load_contract(accepted)
            self.assertFalse(
                payload["protocol"][
                    "room2_frame_pixels_read_before_validation_pass"
                ]
            )
            self.assertEqual(source, accepted.resolve())
            self.assertEqual(len(digest), 64)

            rejected = root / "rejected.json"
            rejected.write_text(
                json.dumps(_contract(room2_read=True)) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(HistorySmokeContractError, "Room2"):
                _load_contract(rejected)

    def test_motion_only_addendum_binds_contract_and_forbids_clear_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.json"
            contract.write_text(json.dumps(_contract()) + "\n", encoding="utf-8")
            _, contract_source, contract_sha = _load_contract(contract)
            addendum = root / "addendum.json"
            payload = {
                "schema": "unblur_slam.turtle_temporal_v2_motion_only_addendum.v1",
                "status": "frozen_before_any_v2_validation_evaluation",
                "binds": {
                    "contract_path": str(contract_source),
                    "contract_sha256": contract_sha,
                },
                "selection_policy": {
                    "tum_keyframe_selection": "motion_only_selection_independent",
                    "clear_gt_membership_used": False,
                    "gt_pose_or_depth_used_during_selection": False,
                    "legacy_clear_conditioned_smoke_permitted": False,
                    "open_gt_metrics_only_after_outputs_frozen": True,
                },
                "implementation_pin_overrides": {
                    "report_script_sha256": "a" * 64,
                    "report_test_sha256": "b" * 64,
                },
            }
            addendum.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            loaded, source, digest = _load_addendum(
                addendum,
                contract_source=contract_source,
                contract_sha=contract_sha,
            )
            self.assertFalse(
                loaded["selection_policy"]["clear_gt_membership_used"]
            )
            self.assertEqual(source, addendum.resolve())
            self.assertEqual(len(digest), 64)

            payload["selection_policy"][
                "legacy_clear_conditioned_smoke_permitted"
            ] = True
            addendum.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                HistorySmokeContractError, "selection policy"
            ):
                _load_addendum(
                    addendum,
                    contract_source=contract_source,
                    contract_sha=contract_sha,
                )

    def test_reporting_fix_is_bound_and_cannot_change_metrics_or_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fix.json"
            payload = {
                "schema": "unblur_slam.turtle_temporal_v2_reporting_fix.v1",
                "status": "frozen_after_seed17_validation_before_multiseed_report",
                "binds": {
                    "contract_sha256": "a" * 64,
                    "motion_addendum_sha256": "b" * 64,
                },
                "metric_values_or_gates_changed": False,
                "implementation_pin_overrides": {
                    "report_script_sha256": "c" * 64,
                    "report_test_sha256": "d" * 64,
                },
            }
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            loaded, source, digest = _load_reporting_fix(
                path,
                contract_sha="a" * 64,
                motion_addendum_sha="b" * 64,
            )
            self.assertFalse(loaded["metric_values_or_gates_changed"])
            self.assertEqual(source, path.resolve())
            self.assertEqual(len(digest), 64)
            payload["metric_values_or_gates_changed"] = True
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                HistorySmokeContractError, "metric values"
            ):
                _load_reporting_fix(
                    path,
                    contract_sha="a" * 64,
                    motion_addendum_sha="b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
