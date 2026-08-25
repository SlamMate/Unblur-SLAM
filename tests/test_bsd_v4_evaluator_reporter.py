#!/usr/bin/env python3
"""CPU-only focused contracts for the v4 BSD reference reporter."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import ContractError  # noqa: E402
from scripts.report_turtle_bsd_dpdd_v1 import (  # noqa: E402
    build_reference_gate_receipt,
    build_reference_validation_report,
    write_new_json,
)


CONTRACT_PATH = ROOT / "configs/local/turtle_finetune/bsd_dpdd_causal_v1_preregistered.json"
RAW = {"psnr": 20.0, "ssim": 0.4, "l1": 0.2}
RUNTIME = {
    "physical_gpu": 1,
    "visible_device": "1",
    "logical_device": "cuda:0",
    "hardware_queried": True,
    "gpu_name": "NVIDIA RTX A6000",
    "total_memory_bytes": 50_465_865_728,
}


def _metric(psnr: float) -> dict[str, float]:
    return {"psnr": psnr, "ssim": 0.5, "l1": 0.1}


def _raw_baseline() -> dict:
    return {
        "registration": {"per_frame_rows_present": True},
        "all_frames": dict(RAW),
        "steady": dict(RAW),
        "per_sequence": {
            f"sequence_{sequence:02d}": {
                "all_frames": dict(RAW),
                "steady": dict(RAW),
                "frame_count": 100,
                "steady_frame_count": 97,
            }
            for sequence in range(20)
        },
    }


def _per_sequence(model: dict[str, float], *, evssm: bool) -> dict:
    result = {}
    for sequence in range(20):
        name = f"sequence_{sequence:02d}"
        if evssm:
            result[name] = {
                "mean": dict(model),
                "steady_mean": dict(model),
                "raw_mean": dict(RAW),
                "raw_steady_mean": dict(RAW),
                "frame_count": 100,
                "steady_frame_count": 97,
            }
        else:
            result[name] = {
                "mean": {"raw": dict(RAW), "turtle": dict(model)},
                "steady_mean": {"raw": dict(RAW), "turtle": dict(model)},
                "frame_count": 100,
                "steady_frame_count": 97,
            }
    return result


def _identity_rows(model: dict[str, float], *, evssm: bool) -> list[dict]:
    rows = []
    for sequence in range(20):
        name = f"sequence_{sequence:02d}"
        for frame in range(100):
            raw_path = f"/synthetic/validation/{name}/blur/{frame:08d}.png"
            gt_path = f"/synthetic/validation/{name}/sharp/{frame:08d}.png"
            if evssm:
                rows.append(
                    {
                        "sequence": name,
                        "frame_index": frame,
                        "raw_path": raw_path,
                        "gt_path": gt_path,
                        "raw_metrics": dict(RAW),
                        "metrics": dict(model),
                    }
                )
            else:
                rows.append(
                    {
                        "sequence": name,
                        "frame_index": frame,
                        "raw_path": raw_path,
                        "gt_path": gt_path,
                        "metrics": {"raw": dict(RAW), "turtle": dict(model)},
                    }
                )
    return rows


def _evssm_payload(contract: dict, contract_sha: str) -> dict:
    model = _metric(25.0)
    return {
        "schema": "unblur_slam.evssm_bsd3ms24ms_direct_float_validation.v1",
        "formal": True,
        "arm": "E",
        "protocol": {
            "contract_sha256": contract_sha,
            "manifest_sha256": contract["data"]["bsd"]["validation_manifest_sha256"],
            "formal_full_validation": True,
            "prediction_representation_for_metrics": "direct_float32_tensor_no_png_roundtrip",
        },
        "runtime": dict(RUNTIME),
        "model": {"checkpoint_sha256": contract["models"]["evssm_E"]["checkpoint_sha256"]},
        "results": {
            "sequence_count": 20,
            "frame_count": 2000,
            "mean": model,
            "steady_mean": model,
            "per_sequence": _per_sequence(model, evssm=True),
            "raw_baseline": _raw_baseline(),
            "performance": {
                "warmup": {"unmeasured_calls": 1},
                "pass_separation": {
                    "timing_pass": {
                        "stateless_model_steps": 2000,
                        "sharp_target_images_opened": False,
                        "metrics_computed": False,
                        "history_or_replay_control_forwards": 0,
                    },
                    "quality_pass": {"stateless_model_steps": 2000, "timed_model_steps": 0},
                    "forward_accounting_excluding_warmup": {
                        "timing_only_model_steps": 2000,
                        "quality_model_steps": 2000,
                        "combined_model_steps": 4000,
                    },
                },
                "evssm_latency_ms": {"frames": 2000},
                "steady_evssm_latency_ms": {"frames": 1940},
                "compute_precision": {
                    "model_and_input": "CUDA_FP32",
                    "autocast": "disabled",
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": True,
                },
            },
            "frames": _identity_rows(model, evssm=True),
        },
    }


def _turtle_payload(contract: dict, contract_sha: str, arm: str, psnr: float) -> dict:
    model = _metric(psnr)
    controls = {
        name: dict(model)
        for name in (
            "turtle_reset_cache",
            "turtle_repeat_current",
            "turtle_shuffled_history",
        )
    }
    return {
        "schema": "unblur_slam.turtle_streaming_evaluation.v1",
        "sequence_count": 20,
        "frame_count": 2000,
        "mean": {"raw": dict(RAW), "turtle": model},
        "steady_mean": {"raw": dict(RAW), "turtle": model},
        "per_sequence": _per_sequence(model, evssm=False),
        "raw_baseline": _raw_baseline(),
        "frames": _identity_rows(model, evssm=False),
        "provenance": {
            "arm": arm,
            "contract_sha256": contract_sha,
            "manifest_sha256": contract["data"]["bsd"]["validation_manifest_sha256"],
            "formal_full_validation": True,
            "checkpoint_sha256": contract["models"][f"turtle_{arm}"]["checkpoint_sha256"],
            "runtime": dict(RUNTIME),
        },
        "performance": {
            "warmup": {"unmeasured_calls": 1},
            "history_controls_timed": False,
            "turtle_latency_ms": {"frames": 2000},
            "steady_turtle_latency_ms": {"frames": 1940},
            "compute_precision": {
                "backend_inference_precision": "fp16",
                "normal_and_control_forward_autocast": "CUDA_FP16",
            },
            "pass_separation": {
                "timing_pass": {
                    "normal_stream_model_steps": 2000,
                    "sharp_target_images_opened": False,
                    "metrics_computed": False,
                    "history_or_replay_control_forwards": 0,
                },
                "quality_history_pass": {
                    "normal_stream_model_steps": 2000,
                    "timed_model_steps": 0,
                    "total_model_steps_including_controls": 16280,
                },
                "forward_accounting_excluding_warmup": {
                    "timing_only_model_steps": 2000,
                    "quality_history_model_steps": 16280,
                    "combined_model_steps": 18280,
                },
            },
        },
        "history_ablation": {
            "ordered_replay_frame_count": 2000,
            "ordered_replay_matches_stream": True,
            "ordered_replay_max_abs": 0.0,
            "control_frame_count": 120,
            "forward_accounting_excluding_warmup": {
                "total": 16280,
                "dedicated_timing_normal_stream": 2000,
                "quality_normal_stream": 2000,
                "timing_pass_history_or_replay_controls": 0,
                "total_including_dedicated_timing_pass": 18280,
            },
            "protocol": {
                "expensive_control_frame_indices_per_sequence": [3, 19, 39, 59, 79, 99],
                "normal_and_all_controls_share_backend_autocast_path": True,
            },
            "steady_normal_minus_control": controls,
        },
    }


class BsdV4ReporterTests(unittest.TestCase):
    def test_bad_o_quality_is_descriptive_but_raw_mismatch_fails(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        contract_sha = "c" * 64
        with tempfile.TemporaryDirectory(prefix="bsd-v4-report-") as directory:
            root = Path(directory)
            e_path = write_new_json(root / "E.json", _evssm_payload(contract, contract_sha))
            g_path = write_new_json(root / "G.json", _turtle_payload(contract, contract_sha, "G", 30.0))
            bad_o = _turtle_payload(contract, contract_sha, "O", 1.0)
            o_path = write_new_json(root / "O.json", bad_o)
            gate = build_reference_gate_receipt(
                contract,
                contract_sha,
                e_report=e_path,
                g_report=g_path,
                o_report=o_path,
            )
            self.assertEqual(gate["status"], "pass")
            self.assertFalse(gate["quality_selection"]["O_quality_used_to_authorize_training"])
            gate_path = write_new_json(root / "gate.json", gate)
            report = build_reference_validation_report(
                contract,
                contract_sha,
                e_report=e_path,
                g_report=g_path,
                o_report=o_path,
                gate_receipt=gate_path,
            )
            self.assertTrue(report["raw_baseline"]["identical_across_reported_arms"])
            self.assertEqual(set(report["model_minus_raw"]), {"E", "G", "O"})
            self.assertLess(report["descriptive_deltas"]["O_minus_G_steady"]["psnr"], 0.0)
            self.assertFalse(report["slam_quality_or_speed_claim"])

            mismatched_o = copy.deepcopy(bad_o)
            mismatched_o["frames"][0]["metrics"]["raw"]["psnr"] += 1.0e-6
            mismatch_path = write_new_json(root / "O_mismatch.json", mismatched_o)
            with self.assertRaises(ContractError):
                build_reference_gate_receipt(
                    contract,
                    contract_sha,
                    e_report=e_path,
                    g_report=g_path,
                    o_report=mismatch_path,
                )


if __name__ == "__main__":
    unittest.main()
