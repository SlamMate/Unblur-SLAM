#!/usr/bin/env python3
"""CPU/mocked tests for the physical-GPU1 v4 executor boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from scripts import bsd_dpdd_runtime as runtime
from scripts.execute_turtle_bsd_dpdd_v1 import _validate_final_report_payload


class _Completed:
    def __init__(self, stdout: str, *, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _nvidia_runner(*, processes: str = ""):
    def run(argv, **kwargs):
        del kwargs
        joined = " ".join(argv)
        if "--query-gpu=" in joined:
            return _Completed(
                "1, NVIDIA RTX A6000, "
                "GPU-3501b285-78cd-1494-87f1-ccac2136866e, "
                "1711224002341, 49140\n"
            )
        if "--query-compute-apps=" in joined:
            return _Completed(processes)
        raise AssertionError(f"unexpected nvidia-smi command: {argv!r}")

    return run


class _FakeCudnn:
    allow_tf32 = True
    benchmark = True
    deterministic = False

    @staticmethod
    def version():
        return 8902


class _FakeMatmul:
    allow_tf32 = True


class _FakeCuda:
    def __init__(self) -> None:
        self.matmul = _FakeMatmul()

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_name(index):
        if index != 0:
            raise AssertionError(index)
        return "NVIDIA RTX A6000"

    @staticmethod
    def get_device_properties(index):
        if index != 0:
            raise AssertionError(index)
        return types.SimpleNamespace(total_memory=48 * 1024**3)


class _FakeTorch:
    __version__ = "2.3.1"
    version = types.SimpleNamespace(cuda="12.1")

    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.backends = types.SimpleNamespace(cuda=self.cuda, cudnn=_FakeCudnn())
        self._deterministic = False
        self._warn_only = False
        self._debug = 0
        self._precision = "highest"

    def use_deterministic_algorithms(self, enabled, *, warn_only):
        self._deterministic = bool(enabled)
        self._warn_only = bool(warn_only)

    def set_deterministic_debug_mode(self, mode):
        self._debug = 2 if mode == "error" else int(mode)

    def set_float32_matmul_precision(self, value):
        self._precision = str(value)

    def are_deterministic_algorithms_enabled(self):
        return self._deterministic

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self._warn_only

    def get_deterministic_debug_mode(self):
        return self._debug

    def get_float32_matmul_precision(self):
        return self._precision


class RuntimeExecutorV4Tests(unittest.TestCase):
    def test_physical_identity_and_idle_are_exact(self) -> None:
        identity = runtime.inspect_physical_gpu1(
            require_idle=True, command_runner=_nvidia_runner()
        )
        self.assertEqual(identity["physical_index"], 1)
        self.assertEqual(identity["uuid"], runtime.EXPECTED_GPU_UUID)
        self.assertEqual(identity["serial"], runtime.EXPECTED_GPU_SERIAL)
        self.assertEqual(identity["compute_processes"], [])

        busy = (
            f"{runtime.EXPECTED_GPU_UUID}, 9191, rogue, 128\n"
        )
        with self.assertRaisesRegex(runtime.RuntimeContractError, "not idle"):
            runtime.inspect_physical_gpu1(
                require_idle=True, command_runner=_nvidia_runner(processes=busy)
            )

    def test_unfamiliar_compute_pid_fails_closed(self) -> None:
        rows = f"{runtime.EXPECTED_GPU_UUID}, 222, other-job, 10\n"
        with self.assertRaisesRegex(runtime.RuntimeContractError, "unfamiliar"):
            runtime.require_only_known_compute_pids(
                [111], phase="unit", command_runner=_nvidia_runner(processes=rows)
            )

    def test_code_bundle_rehash_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            source.write_text("frozen\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            rows = [{"pin": "source_sha256", "path": str(source), "sha256": digest}]
            bundle = {
                "schema": runtime.CODE_BUNDLE_SCHEMA,
                "files": {"source_sha256": str(source)},
                "bundle_sha256": hashlib.sha256(
                    json.dumps(
                        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode("utf-8")
                ).hexdigest(),
            }
            verified = runtime.rehash_frozen_code_bundle(
                implementation_pins={"source_sha256": digest},
                code_bundle=bundle,
                expected_path_map={"source_sha256": source},
            )
            self.assertEqual(verified["bundle_sha256"], bundle["bundle_sha256"])
            source.write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(runtime.RuntimeContractError, "pin changed"):
                runtime.rehash_frozen_code_bundle(
                    implementation_pins={"source_sha256": digest},
                    code_bundle=bundle,
                    expected_path_map={"source_sha256": source},
                )

    def test_logical_identity_requires_lock_and_deterministic_runtime(self) -> None:
        lock_parent = Path("/srv/szha0669/unblur-slam/locks")
        lock_parent.mkdir(parents=True, exist_ok=True)
        physical = {
            "physical_index": 1,
            "name": runtime.EXPECTED_GPU_NAME,
            "uuid": runtime.EXPECTED_GPU_UUID,
            "serial": runtime.EXPECTED_GPU_SERIAL,
            "nvidia_smi_total_memory_bytes": 48 * 1024**3,
        }
        environment = {
            "CUDA_VISIBLE_DEVICES": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
        with tempfile.TemporaryDirectory(dir=lock_parent, prefix="runtime-v4-") as directory:
            with mock.patch.dict(os.environ, environment, clear=False):
                with runtime.exclusive_gpu1_lock(Path(directory) / "gpu1.lock") as lock:
                    identity = runtime.require_gpu1_a6000(
                        "cuda:0",
                        lock_identity=lock,
                        physical_identity=physical,
                        environment_fingerprint_sha256="a" * 64,
                        code_bundle_sha256="b" * 64,
                        torch_module=_FakeTorch(),
                    )
        self.assertEqual(identity["logical_device"], "cuda:0")
        self.assertEqual(identity["gpu_uuid"], runtime.EXPECTED_GPU_UUID)
        self.assertTrue(identity["torch_runtime"]["deterministic_algorithms"])
        self.assertFalse(identity["torch_runtime"]["cuda_matmul_allow_tf32"])

    def test_final_report_is_schema_and_claim_strict(self) -> None:
        runtime_identity = {
            "schema": runtime.RUNTIME_IDENTITY_SCHEMA,
            "identity_sha256": "c" * 64,
        }
        keys = [
            "schema",
            "status",
            "contract_sha256",
            "arms",
            "raw_baseline",
            "claims_policy",
            "runtime_identity",
            "bsd_test_pixels_opened",
            "bsd_test_authorized",
            "slam_quality_or_speed_claim",
        ]
        specification = {
            "schema": "unit.final.v1",
            "status": "complete",
            "contract_sha256": "d" * 64,
            "arms": ["E", "G", "O"],
            "raw_baseline_required": True,
            "forbidden_claims": ["forbidden statement"],
            "runtime_identity_required": True,
            "exact_top_level_keys": keys,
        }
        payload = {
            "schema": "unit.final.v1",
            "status": "complete",
            "contract_sha256": "d" * 64,
            "arms": ["E", "G", "O"],
            "raw_baseline": {
                "source": "common_blurry_input_vs_sharp_ground_truth",
                "identical_across_reported_arms": True,
                "all_frames": {"psnr": 20.0, "ssim": 0.7, "l1": 0.1},
                "steady": {"psnr": 21.0, "ssim": 0.8, "l1": 0.09},
                "per_sequence": {
                    f"sequence_{index:02d}": {
                        "all_frames": {"psnr": 20.0, "ssim": 0.7, "l1": 0.1},
                        "steady": {"psnr": 21.0, "ssim": 0.8, "l1": 0.09},
                        "frame_count": 100,
                        "steady_frame_count": 97,
                    }
                    for index in range(20)
                },
                "registration_sha256": "e" * 64,
                "per_frame_E_G_O_identity_equal": True,
                "per_frame_E_G_O_metrics_abs_tolerance": 1.0e-12,
            },
            "claims_policy": {
                "forbidden_claims": ["forbidden statement"],
                "forbidden_claims_made": [],
                "all_forbidden_claims_excluded": True,
            },
            "runtime_identity": runtime_identity,
            "bsd_test_pixels_opened": False,
            "bsd_test_authorized": False,
            "slam_quality_or_speed_claim": False,
        }
        _validate_final_report_payload(
            payload,
            specification=specification,
            runtime_identity=runtime_identity,
        )
        payload["claims_policy"]["forbidden_claims_made"] = ["forbidden statement"]
        with self.assertRaisesRegex(Exception, "made a forbidden claim"):
            _validate_final_report_payload(
                payload,
                specification=specification,
                runtime_identity=runtime_identity,
            )


if __name__ == "__main__":
    unittest.main()
