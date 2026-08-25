#!/usr/bin/env python3
"""CPU-only contracts for the two-A6000 lock-first launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import execute_pinned_dual_gpu_command as launcher


class DualGpuLauncherContracts(unittest.TestCase):
    def test_pair_identity_and_atomic_audit(self) -> None:
        inventory = [
            {**item, "memory_total_mib": 49140} for item in launcher.EXPECTED
        ]
        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                audit_report=root / "audit.json",
                command=["--", "/bin/true"],
            )
            completed = subprocess.CompletedProcess(["/bin/true"], 0)
            with mock.patch.object(launcher, "LOCKS", (root / "gpu0.lock", root / "gpu1.lock")), \
                    mock.patch.object(launcher, "inspect_physical_gpus", return_value=inventory), \
                    mock.patch.object(launcher, "compute_processes_for_uuid", return_value=[]), \
                    mock.patch.object(launcher.subprocess, "run", return_value=completed) as run_mock:
                report, returncode = launcher.run(args)
            self.assertEqual(returncode, 0)
            self.assertEqual(report["distributed"]["world_size"], 2)
            self.assertEqual(
                run_mock.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0,1"
            )
            self.assertEqual(json.loads(args.audit_report.read_text())["status"], "complete")
            with self.assertRaises(FileExistsError):
                launcher.run(args)

    def test_busy_either_gpu_fails_before_child(self) -> None:
        inventory = [
            {**item, "memory_total_mib": 49140} for item in launcher.EXPECTED
        ]
        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as temporary:
            root = Path(temporary)
            args = argparse.Namespace(audit_report=root / "audit.json", command=["/bin/true"])
            def active(uuid: str) -> list[int]:
                return [1234] if uuid == launcher.EXPECTED[1]["uuid"] else []
            with mock.patch.object(launcher, "LOCKS", (root / "gpu0.lock", root / "gpu1.lock")), \
                    mock.patch.object(launcher, "inspect_physical_gpus", return_value=inventory), \
                    mock.patch.object(launcher, "compute_processes_for_uuid", side_effect=active), \
                    mock.patch.object(launcher.subprocess, "run") as run_mock:
                with self.assertRaisesRegex(RuntimeError, "not idle"):
                    launcher.run(args)
            run_mock.assert_not_called()
            self.assertFalse(args.audit_report.exists())


if __name__ == "__main__":
    unittest.main()
