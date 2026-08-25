#!/usr/bin/env python3
"""CPU-only contracts for the final TURTLE v4 validation reporter."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_turtle_dpdd_mixed_validation_v4 import (  # noqa: E402
    AcceptanceError,
    aggregate_temporal_frames,
    gate,
    require_complete_result_set,
    write_json_exclusive,
)


class FinalReporterContractTest(unittest.TestCase):
    def test_reporter_imports_only_the_standard_library(self) -> None:
        script = ROOT / "scripts/report_turtle_dpdd_mixed_validation_v4.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertTrue(imported_roots <= set(sys.stdlib_module_names) | {"__future__"})
        self.assertNotIn("torch", imported_roots)
        self.assertNotIn("cv2", imported_roots)

    def test_missing_temporal_matrix_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "preregistered_contract.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(AcceptanceError, "incomplete validation result set"):
                require_complete_result_set(root)

    def test_exclusive_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "validation_only_report.json"
            write_json_exclusive(output, {"first": True})
            before = output.read_bytes()
            with self.assertRaisesRegex(AcceptanceError, "refusing to overwrite"):
                write_json_exclusive(output, {"second": True})
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(json.loads(before), {"first": True})

    def test_gate_boundaries_are_explicit(self) -> None:
        self.assertTrue(gate("ge", 0.1, ">=", 0.1)["pass"])
        self.assertTrue(gate("le", 0.0, "<=", 0.0)["pass"])
        self.assertFalse(gate("strict", 0.0, ">", 0.0)["pass"])

    def test_temporal_aggregation_pools_frames_and_keeps_sequences_separate(self) -> None:
        frames = []
        global_index = 0
        for sequence, base in (("room1_a", 10.0), ("room1_b", 20.0)):
            for frame_index in range(5):
                metrics = {}
                for source_index, source in enumerate(
                    (
                        "raw",
                        "turtle",
                        "turtle_reset_cache",
                        "turtle_repeat_current",
                        "turtle_replayed_ordered",
                        "turtle_shuffled_history",
                    )
                ):
                    value = base + frame_index + source_index
                    metrics[source] = {"psnr": value, "ssim": value / 100.0, "l1": value / 1000.0}
                frames.append(
                    {
                        "sequence": sequence,
                        "frame_index": frame_index,
                        "global_index": global_index,
                        "metrics": metrics,
                    }
                )
                global_index += 1
        result = aggregate_temporal_frames(frames, steady_index_min=3)
        self.assertEqual(result["all_frame_count"], 10)
        self.assertEqual(result["steady_frame_count"], 4)
        self.assertEqual(result["per_sequence"]["room1_a"]["steady_frame_count"], 2)
        self.assertEqual(result["per_sequence"]["room1_b"]["steady_frame_count"], 2)
        self.assertAlmostEqual(result["steady_pooled_mean"]["turtle"]["psnr"], 19.5)
        self.assertAlmostEqual(result["per_sequence"]["room1_a"]["mean"]["turtle"]["psnr"], 14.5)


if __name__ == "__main__":
    unittest.main()
