#!/usr/bin/env python3
"""Standard-library CPU tests for the BSD preregistration and t0 loader."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import (  # noqa: E402
    ContractError,
    approximately_even_positions,
    deterministic_bsd_schedule,
    inspect_bsd_sequence_manifest,
    load_contract,
    validate_code_bundle_declaration,
    validate_protocol,
)
from scripts.train_turtle_bsd_dpdd import (  # noqa: E402
    build_formal_schedules,
    optimizer_step_mode,
)
from scripts.evaluate_turtle_streaming import evaluate_sequences  # noqa: E402
from scripts.evaluate_evssm_bsd_validation import evaluate as evaluate_evssm_bsd  # noqa: E402
from scripts.bsd_dpdd_runtime import (  # noqa: E402
    RuntimeContractError,
    exclusive_gpu1_lock,
    require_gpu1_a6000,
)
from scripts.report_turtle_bsd_dpdd_v1 import (  # noqa: E402
    build_reference_gate_receipt,
    build_reference_validation_report,
    write_new_json,
)
from scripts.run_turtle_bsd_dpdd_v1 import build_plan, build_reference_plan  # noqa: E402
from scripts.train_turtle_streaming import SequenceRecord  # noqa: E402
from src.turtle_backend import TurtleStreamingBackend, load_turtle_model  # noqa: E402
from src.turtle_official_bsd_backend import (  # noqa: E402
    PINNED_OFFICIAL_BSD_CHECKPOINT,
    PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
    PINNED_OFFICIAL_BSD_PARAMETERS,
    PINNED_OFFICIAL_BSD_STATE_TENSORS,
    load_official_bsd_turtle_model,
    validate_official_bsd_artifacts,
)
from tests.test_bsd_v4_evaluator_reporter import (  # noqa: E402
    _evssm_payload as _v4_evssm_payload,
    _turtle_payload as _v4_turtle_payload,
)


CONTRACT = ROOT / "configs/local/turtle_finetune/bsd_dpdd_causal_v1_preregistered.json"


class _ReplayModel(torch.nn.Module):
    """Tiny deterministic model with the official eight-slot cache shape."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, pair, k_cache, v_cache):
        current = pair[:, 1]
        prior = (
            torch.zeros_like(current[:, :1, :1, :1])
            if k_cache is None
            else k_cache[3]
        )
        state = prior + current.mean(dim=(1, 2, 3), keepdim=True)
        restored = current + state * 0.001 + self.anchor * 0.0
        k_new = [None, None, None] + [state.clone() for _ in range(5)]
        v_new = [None, None, None] + [state.clone() for _ in range(5)]
        return restored, k_new, v_new


class _FloatEvssmBackend:
    def __call__(self, image, timestamp=None):
        del timestamp
        # Deliberately non-8-bit-exact to prove scoring retains float output.
        return (image + 0.0013).clamp(0.0, 1.0)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BsdDpddContractTests(unittest.TestCase):
    def test_frozen_bundle_declaration_fails_closed_on_pin_or_path_tamper(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        validate_code_bundle_declaration(contract)

        pin_tamper = copy.deepcopy(contract)
        pin_tamper["implementation_pins"]["streaming_trainer_dependency_sha256"] = (
            "0" * 64
        )
        with self.assertRaises(ContractError):
            validate_code_bundle_declaration(pin_tamper)

        path_tamper = copy.deepcopy(contract)
        path_tamper["code_bundle"]["files"]["runtime_guard_sha256"] = (
            "/tmp/substituted_runtime.py"
        )
        with self.assertRaises(ContractError):
            validate_code_bundle_declaration(path_tamper)

    def test_gpu1_runtime_guard_and_global_lock_fail_closed_without_gpu_query(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
            identity = require_gpu1_a6000("cuda:0", query_hardware=False)
            self.assertEqual(identity["physical_gpu"], 1)
            self.assertEqual(identity["logical_device"], "cuda:0")
            with self.assertRaises(RuntimeContractError):
                require_gpu1_a6000("cuda:1", query_hardware=False)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            with self.assertRaises(RuntimeContractError):
                require_gpu1_a6000("cuda:0", query_hardware=False)

        lock_parent = Path("/srv/szha0669/unblur-slam/locks")
        lock_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=lock_parent, prefix="bsd-lock-cpu-") as directory:
            lock = Path(directory) / "study.lock"
            with exclusive_gpu1_lock(lock):
                with self.assertRaises(RuntimeContractError):
                    with exclusive_gpu1_lock(lock):
                        pass

    def test_incremental_ordered_replay_uses_exact_normal_forward_path(self) -> None:
        backend = TurtleStreamingBackend(
            _ReplayModel(), device="cpu", inference_precision="fp32"
        )
        frames = [torch.full((1, 3, 8, 8), index / 20.0) for index in range(6)]
        replay_k = replay_v = replay_previous = None
        normal = []
        replay = []
        original = backend._forward_with_explicit_state
        with mock.patch.object(
            backend,
            "_forward_with_explicit_state",
            wraps=original,
        ) as shared_forward:
            for index, frame in enumerate(frames):
                normal.append(backend.step(frame, timestamp=index))
                value, replay_k, replay_v = backend.replay_step(
                    frame,
                    k_cache=replay_k,
                    v_cache=replay_v,
                    previous_frame=replay_previous,
                )
                replay.append(value)
                replay_previous = frame
        self.assertEqual(shared_forward.call_count, len(frames) * 2)
        for left, right in zip(normal, replay):
            self.assertTrue(torch.equal(left, right))

    def test_history_evaluator_replay_is_exact_and_controls_are_subset_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bsd-history-cpu-") as directory:
            root = Path(directory)
            blurry = []
            sharp = []
            for index in range(6):
                raw_path = root / f"raw_{index:08d}.png"
                gt_path = root / f"gt_{index:08d}.png"
                Image.fromarray(
                    np.full((8, 8, 3), index * 5, dtype=np.uint8), mode="RGB"
                ).save(raw_path)
                Image.fromarray(
                    np.full((8, 8, 3), index * 5 + 1, dtype=np.uint8), mode="RGB"
                ).save(gt_path)
                blurry.append(raw_path)
                sharp.append(gt_path)
            record = SequenceRecord(
                name="synthetic", blurry=tuple(blurry), sharp=tuple(sharp), teacher=None
            )
            backend = TurtleStreamingBackend(
                _ReplayModel(), device="cpu", inference_precision="fp32"
            )
            report = evaluate_sequences(
                [record],
                backend,
                device=torch.device("cpu"),
                output_dir=root / "output",
                max_visuals=0,
                history_controls=True,
            )
            self.assertEqual(report["history_ablation"]["ordered_replay_max_abs"], 0.0)
            self.assertEqual(report["history_ablation"]["ordered_replay_frame_count"], 6)
            self.assertEqual(report["history_ablation"]["control_frame_count"], 1)
            self.assertEqual(
                report["history_ablation"]["forward_accounting_excluding_warmup"][
                    "total"
                ],
                21,
            )
            self.assertEqual(
                report["history_ablation"]["forward_accounting_excluding_warmup"][
                    "dedicated_timing_normal_stream"
                ],
                6,
            )
            self.assertEqual(
                report["history_ablation"]["forward_accounting_excluding_warmup"][
                    "total_including_dedicated_timing_pass"
                ],
                27,
            )
            self.assertEqual(
                report["history_ablation"]["control_per_sequence"]["synthetic"][
                    "frame_indices"
                ],
                [3],
            )
            self.assertIs(report["performance"]["history_controls_timed"], False)
            self.assertEqual(report["performance"]["warmup"]["unmeasured_calls"], 1)
            self.assertEqual(
                report["performance"]["pass_separation"]["timing_pass"][
                    "normal_stream_model_steps"
                ],
                6,
            )
            self.assertIs(
                report["performance"]["pass_separation"]["timing_pass"][
                    "sharp_target_images_opened"
                ],
                False,
            )

    def test_direct_evssm_bsd_evaluator_keeps_float_and_warmup_out_of_latency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bsd-evssm-float-cpu-") as directory:
            root = Path(directory)
            blurry = []
            sharp = []
            for index in range(6):
                raw_path = root / f"raw_{index:08d}.png"
                gt_path = root / f"gt_{index:08d}.png"
                Image.fromarray(
                    np.full((8, 8, 3), 64 + index, dtype=np.uint8), mode="RGB"
                ).save(raw_path)
                Image.fromarray(
                    np.full((8, 8, 3), 65 + index, dtype=np.uint8), mode="RGB"
                ).save(gt_path)
                blurry.append(raw_path)
                sharp.append(gt_path)
            record = SequenceRecord(
                name="synthetic", blurry=tuple(blurry), sharp=tuple(sharp), teacher=None
            )
            report = evaluate_evssm_bsd(
                [record], _FloatEvssmBackend(), device=torch.device("cpu")
            )
            self.assertEqual(report["frame_count"], 6)
            self.assertEqual(report["performance"]["warmup"]["unmeasured_calls"], 1)
            self.assertEqual(report["performance"]["evssm_latency_ms"]["frames"], 6)
            self.assertEqual(
                report["performance"]["pass_separation"]["timing_pass"][
                    "stateless_model_steps"
                ],
                6,
            )
            self.assertEqual(report["raw_baseline"]["registration"]["per_frame_rows_present"], True)
            self.assertIn("all_frames", report["model_minus_raw"])
            self.assertIn("SLAM", report["performance"]["latency_scope"])

    def test_reference_plan_has_direct_E_smoke_real_gate_report_and_no_training(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        contract["sealed_outputs"]["output_root"] = "/srv/szha0669/unblur-slam/experiments/synthetic_reference_plan"
        plan = build_reference_plan(
            CONTRACT,
            contract,
            "a" * 64,
            preflight_report=Path("/srv/synthetic-preflight.json"),
            preflight_sha="b" * 64,
        )
        ids = [action["id"] for action in plan["actions"]]
        self.assertEqual(plan["training_actions"], 0)
        self.assertIs(plan["training_blocked"], True)
        self.assertIn("smoke_E_bsd_first2", ids)
        self.assertIn("reference_E_bsd_full", ids)
        self.assertIn("cpu_reference_qualification_gate", ids)
        self.assertIn("cpu_reference_validation_report", ids)
        self.assertFalse(any(action_id.startswith("train_") for action_id in ids))
        e_smoke = next(action for action in plan["actions"] if action["id"] == "smoke_E_bsd_first2")
        self.assertIn("evaluate_evssm_bsd_validation.py", " ".join(e_smoke["argv"]))
        self.assertIn("2", e_smoke["argv"])

    def test_future_full_plan_is_all_real_argv_and_uses_current_evaluator_clis(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        contract["sealed_outputs"]["output_root"] = (
            "/srv/szha0669/unblur-slam/experiments/synthetic_full_plan_v4"
        )
        plan = build_plan(
            CONTRACT,
            contract,
            "a" * 64,
            preflight_report=Path("/srv/synthetic-preflight.json"),
            preflight_sha="b" * 64,
        )
        self.assertEqual(plan["mode"], "full")
        self.assertEqual(plan["training_actions"], 6)
        self.assertFalse(plan["training_blocked"])
        self.assertFalse(plan["reference_gate_uses_quality"])
        self.assertEqual(len(plan["actions"]), 29)
        self.assertTrue(all("argv" in action for action in plan["actions"]))
        self.assertTrue(
            all(
                action["env"]
                == {
                    "CUDA_VISIBLE_DEVICES": "1",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                }
                for action in plan["actions"]
            )
        )
        flattened = [value for action in plan["actions"] for value in action["argv"]]
        self.assertFalse(any("precompute_video_deblur_evssm.py" in value for value in flattened))
        self.assertNotIn("--evssm-precompute-report", flattened)
        self.assertIn("evaluate_evssm_bsd_validation.py", " ".join(flattened))
        gate = next(
            action
            for action in plan["actions"]
            if action["id"] == "cpu_reference_qualification_gate"
        )
        self.assertIn("training-gate", gate["argv"])
        self.assertNotIn("reference-gate", gate["argv"])
        self.assertNotIn("--reference-only", gate["argv"])
        final = plan["actions"][-1]
        self.assertEqual(final["id"], "cpu_full_validation_report")
        self.assertIn("full-report", final["argv"])

    def test_reference_gate_does_not_use_bad_O_quality_and_report_blocks_training(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        contract_sha = "c" * 64
        with tempfile.TemporaryDirectory(prefix="bsd-reference-report-cpu-") as directory:
            root = Path(directory)
            e_path = write_new_json(
                root / "E.json", _v4_evssm_payload(contract, contract_sha)
            )
            g_path = write_new_json(
                root / "G.json", _v4_turtle_payload(contract, contract_sha, "G", 30.0)
            )
            # Intentionally terrible O quality: operational gate must still pass.
            o_path = write_new_json(
                root / "O.json", _v4_turtle_payload(contract, contract_sha, "O", 1.0)
            )
            gate = build_reference_gate_receipt(
                contract,
                contract_sha,
                e_report=e_path,
                g_report=g_path,
                o_report=o_path,
            )
            self.assertEqual(gate["status"], "pass")
            self.assertIs(
                gate["quality_selection"]["O_quality_used_to_authorize_training"],
                False,
            )
            gate_path = write_new_json(root / "gate.json", gate)
            report = build_reference_validation_report(
                contract,
                contract_sha,
                e_report=e_path,
                g_report=g_path,
                o_report=o_path,
                gate_receipt=gate_path,
            )
            self.assertEqual(
                report["status"],
                "reference_validation_complete_training_still_blocked",
            )
            self.assertLess(report["descriptive_deltas"]["O_minus_G_steady"]["psnr"], 0)
            self.assertIs(report["slam_quality_or_speed_claim"], False)
            self.assertIs(report["bsd_test_authorized"], False)

    def test_template_preregistration_is_frozen_but_launch_blocked(self) -> None:
        _, contract, _ = load_contract(CONTRACT)
        unbound = validate_protocol(contract, allow_template=True)
        self.assertIs(contract["launch_authorized"], False)
        self.assertEqual(set(contract["arms"]), {"E", "G", "O", "B", "BD"})
        self.assertEqual(contract["models"]["turtle_G"]["architecture_variant"], "t1")
        self.assertEqual(contract["models"]["turtle_O"]["architecture_variant"], "t0")
        self.assertEqual(contract["data"]["bsd"]["exposure_setting"], "3ms24ms")
        self.assertIs(contract["data"]["bsd"]["synthetic_high_fps_average"], False)
        self.assertTrue(any("train_manifest" in path for path in unbound))
        self.assertTrue(any("validation_manifest" in path for path in unbound))
        self.assertTrue(any("output_root" in path for path in unbound))

    def test_official_bsd_checkpoint_strictly_loads_only_t0(self) -> None:
        artifacts = validate_official_bsd_artifacts(load_weights=True)
        self.assertEqual(artifacts.checkpoint_sha256, PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256)
        self.assertEqual(artifacts.checkpoint_metadata["kind"], "official_bsd_3ms24ms_t0")
        model, metadata = load_official_bsd_turtle_model(device="cpu")
        self.assertEqual(metadata["turtle_arch_variant"], "t0")
        self.assertEqual(len(model.state_dict()), PINNED_OFFICIAL_BSD_STATE_TENSORS)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            PINNED_OFFICIAL_BSD_PARAMETERS,
        )
        with torch.no_grad():
            output, k_cache, v_cache = model(torch.zeros(1, 2, 3, 64, 64), None, None)
        self.assertEqual(tuple(output.shape), (1, 3, 64, 64))
        expected_mask = [False, False, False, True, True, True, True, True]
        self.assertEqual([value is not None for value in k_cache], expected_mask)
        self.assertEqual([value is not None for value in v_cache], expected_mask)
        del model

        with self.assertRaises((ValueError, RuntimeError)):
            load_turtle_model(
                "/srv/szha0669/unblur-slam/external/TURTLE",
                PINNED_OFFICIAL_BSD_CHECKPOINT,
                config="/srv/szha0669/unblur-slam/external/TURTLE/options/Turtle_Deblur_Gopro.yml",
                checkpoint_sha256=PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
                device="cpu",
            )

    def test_formal_B_BD_schedules_are_matched_and_budget_exact(self) -> None:
        names = [f"sequence_{index:02d}" for index in range(60)]
        schedule = deterministic_bsd_schedule(names, [100] * 60, seed=17)
        self.assertEqual(len(schedule), 300)
        for pass_index in range(5):
            rows = [row for row in schedule if row[0] == pass_index]
            self.assertEqual(len(rows), 60)
            self.assertEqual({row[1] for row in rows}, set(range(60)))
            self.assertTrue(all(0 <= row[2] <= 95 for row in rows))
        self.assertEqual(schedule, deterministic_bsd_schedule(names, [100] * 60, seed=17))
        self.assertNotEqual(schedule, deterministic_bsd_schedule(names, [100] * 60, seed=42))

        positions = approximately_even_positions(total_steps=300, selected_steps=70)
        self.assertEqual(len(positions), len(set(positions)))
        self.assertEqual(len(positions), 70)
        self.assertGreaterEqual(positions[0], 0)
        self.assertLess(positions[-1], 300)
        all_schedules = build_formal_schedules(
            names,
            [100] * 60,
            [f"dpdd_{index:03d}" for index in range(350)],
            seed=17,
        )
        self.assertEqual(tuple(all_schedules["dpdd_positions"]), positions)
        self.assertEqual(len(all_schedules["dpdd"]), 70)
        self.assertEqual(
            len({index for batch in all_schedules["dpdd"] for index in batch}),
            350,
        )
        b_only = build_formal_schedules(names, [100] * 60, None, seed=17)
        self.assertEqual(b_only["bsd"], all_schedules["bsd"])
        self.assertEqual(b_only["dpdd"], [])
        self.assertEqual(b_only["dpdd_positions"], ())
        self.assertEqual(optimizer_step_mode("B", has_dpdd=False), "V")
        self.assertEqual(optimizer_step_mode("BD", has_dpdd=False), "V")
        self.assertEqual(optimizer_step_mode("BD", has_dpdd=True), "M")
        with self.assertRaisesRegex(ValueError, "never receive"):
            optimizer_step_mode("B", has_dpdd=True)

    def test_bsd_test_row_is_rejected_before_asset_resolution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bsd-sealed-test-") as directory:
            root = Path(directory)
            manifest = root / "validation.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "unblur_slam.bsd_paired_video_sequence.v1",
                        "split": "test",
                        "exposure": "3ms24ms",
                        "capture_id": "sealed",
                        "sequence": "sealed",
                        "temporal_order": "gap_free_capture_order",
                        "paired_target_alignment": "center_aligned_synchronized",
                        "blurry": ["does/not/exist.png"],
                        "sharp": ["also/missing.png"],
                        "blurry_sha256": ["0" * 64],
                        "sharp_sha256": ["1" * 64],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "test rows are sealed"):
                inspect_bsd_sequence_manifest(
                    manifest,
                    dataset_root=root,
                    expected_sha256=_sha(manifest),
                    expected_split="validation",
                    expected_sequences=1,
                    expected_frames=1,
                    expected_per_exposure_sequences=1,
                    require_assets=True,
                )

    def test_plan_runner_writes_blocked_template_O_EXCL(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bsd-plan-test-") as directory:
            output = Path(directory) / "plan.json"
            command = [
                sys.executable,
                str(ROOT / "scripts/run_turtle_bsd_dpdd_v1.py"),
                "--contract",
                str(CONTRACT),
                "--expected-contract-sha256",
                _sha(CONTRACT),
                "--output",
                str(output),
                "--template",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "blocked_unbound_template")
            self.assertEqual(payload["actions"], [])
            self.assertIs(payload["gpu_queries_or_kernels_launched"], False)
            self.assertIs(payload["test_pixels_or_metrics_authorized"], False)
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(command, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
