#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


GPU = load("offline_fair_gpu_contract", "offline_fair_gpu_contract.py")
WRAPPER = load("execute_pinned_gpu_command", "execute_pinned_gpu_command.py")


class PinnedGpuContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = {
            "physical_index": 1,
            "visible_devices": "1",
            "logical_device": "cuda:0",
            "name": "NVIDIA RTX A6000",
            "uuid": "GPU-test",
            "serial": "123",
        }

    def test_identity_and_idle_are_fail_closed(self) -> None:
        def rows(query: str):
            if query.startswith("gpu="):
                return ["1, NVIDIA RTX A6000, GPU-test, 123, 49140"]
            return []

        with mock.patch.object(GPU, "_run_nvidia_smi", side_effect=rows), mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            clear=False,
        ):
            observed = GPU.validate_gpu_contract(
                self.expected, require_visible_mask=True, require_idle=True
            )
        self.assertEqual(observed["uuid"], "GPU-test")
        self.assertTrue(observed["idle_verified_before_launch"])

    def test_wrong_visibility_mask_is_rejected_before_cuda(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "wrong CUDA_VISIBLE_DEVICES"):
                GPU.validate_gpu_contract(
                    self.expected, require_visible_mask=True, require_idle=False
                )

    def test_wrapper_rejects_non_srv_audit_and_lock(self) -> None:
        args = WRAPPER.parse_args(
            [
                    "--lock-file", "/home/szha0669/offline-fair-test.lock",
                    "--audit-report", "/home/szha0669/offline-fair-test.json",
                    "--expected-physical-index", "1",
                    "--expected-cuda-visible-devices", "1",
                    "--expected-gpu-name", "NVIDIA RTX A6000",
                    "--expected-gpu-uuid", "GPU-test",
                    "--expected-gpu-serial", "123",
                    "--", "/bin/true",
            ]
        )
        with self.assertRaisesRegex(ValueError, "under /srv"):
            WRAPPER.run(args)

    def test_atomic_audit_never_replaces_an_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "audit.json"
            destination.write_text('{"owner":"first"}\n', encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "concurrent overwrite"):
                WRAPPER._atomic_json(destination, {"owner": "second"})
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"owner": "first"},
            )


if __name__ == "__main__":
    unittest.main()
