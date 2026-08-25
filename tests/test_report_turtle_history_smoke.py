#!/usr/bin/env python3
"""Standard-library tests for the official TURTLE history smoke reporter."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_turtle_history_smoke import (  # noqa: E402
    EVALUATION_SCHEMA,
    FINETUNED_CHECKPOINT_FORMAT,
    HistorySmokeContractError,
    IMPLEMENTATION_PIN_PATHS,
    NORMAL,
    OFFICIAL_CACHE_CONTRACT,
    ORDERED,
    REPORT_SCHEMA,
    REPEAT,
    REQUIRED_METRIC_SOURCES,
    RESET,
    SHUFFLED,
    build_report,
    sha256_file,
    write_report,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


class HistorySmokeFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_root = root / "data"
        self.output_root = root / "output"
        self.output_root.mkdir(parents=True)
        self.base_checkpoint = root / "GoPro_Deblur.pth"
        self.base_checkpoint.write_bytes(b"synthetic official GoPro")
        self.fine_checkpoint = self.output_root / "finetuned_final.pth"
        self.fine_checkpoint.write_bytes(b"synthetic fixed-terminal fine tune")
        self.base_sha = sha256_file(self.base_checkpoint)
        self.fine_sha = sha256_file(self.fine_checkpoint)

        self.validation_records = [
            {
                "sequence": sequence,
                "blurry": [f"{sequence}/blur_{index}.png" for index in range(4)],
                "sharp": [f"{sequence}/sharp_{index}.png" for index in range(4)],
            }
            for sequence in ("validation_a", "validation_b")
        ]
        self.validation_manifest = _write_jsonl(
            self.root / "val_temporal.jsonl", self.validation_records
        )
        self.train_manifest = _write_jsonl(
            self.root / "train.jsonl",
            [{"sequence": "train", "blurry": ["a", "b"], "sharp": ["c", "d"]}],
        )
        self.contract = self._contract()
        self.contract_path = _write_json(self.root / "contract.json", self.contract)
        self.base_payload = self._evaluation("base")
        self.fine_payload = self._evaluation("finetuned")
        self.base_path = _write_json(self.root / "base_metrics.json", self.base_payload)
        self.fine_path = _write_json(self.root / "finetuned_metrics.json", self.fine_payload)

    def _contract(self) -> dict:
        pins = {
            key: sha256_file(ROOT / relative)
            for key, relative in IMPLEMENTATION_PIN_PATHS.items()
        }
        return {
            "schema": "unblur_slam.turtle_gopro_replica424_history_smoke.v1",
            "status": "preregistered_exploratory_smoke_not_paper_metric",
            "model": {
                "repo_commit": "repo-commit",
                "architecture_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "base_checkpoint": str(self.base_checkpoint),
                "base_checkpoint_sha256": self.base_sha,
                "causal_state": {
                    "cache_slots_per_kind": 8,
                    "populated_slots_per_kind": 5,
                    "direct_cache_capacity_frames": [3, 3, 3, 3, 2],
                    "effective_history": "recurrent_full_prefix",
                    "use_both_input": False,
                },
            },
            "data": {
                "root": str(self.data_root),
                "train_manifest": str(self.train_manifest),
                "train_manifest_sha256": sha256_file(self.train_manifest),
                "validation_manifest": str(self.validation_manifest),
                "validation_manifest_sha256": sha256_file(self.validation_manifest),
                "validation_inventory": {
                    "sequences": 2,
                    "frames": 8,
                    "real_transitions": 6,
                    "steady_frame_index_min": 3,
                    "steady_frames": 2,
                },
            },
            "training": {
                "seed": 42,
                "fixed_terminal_optimizer_steps": 126,
                "trainable_parameter_tensors": 56,
                "trainable_parameters": 3475994,
                "minimum_sequence_length": 2,
                "loss_start_frame": 1,
                "crop_size": 128,
                "augmentation": "one_shared_random_crop_flip_rotation_per_sequence",
                "objective": {
                    "l1_weight": 1.0,
                    "fft_l1_weight": 0.1,
                },
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": 0.00001,
                    "weight_decay": 0.001,
                    "betas": [0.9, 0.9],
                    "scheduler": "CosineAnnealingLR",
                    "eta_min": 1e-7,
                    "gradient_clip": 1.0,
                    "amp": True,
                },
            },
            "evaluation": {
                "arms": [
                    "normal_persistent_kv",
                    "reset_cache_current_only",
                    "repeat_current_same_recurrent_step_count",
                    "replayed_ordered_complete_prefix_contract",
                    "shuffled_complete_past_prefix_only",
                ],
                "same_checkpoint_same_frames": True,
                "future_frames_used": False,
                "eligibility_gates": {
                    "ordered_replay_max_abs_max": 0.000001,
                    "finetuned_normal_minus_base_normal_psnr_db_min": -0.05,
                    "finetuned_normal_minus_base_normal_ssim_min": -0.001,
                    "finetuned_normal_minus_reset_cache_psnr_db_min": 0.05,
                    "finetuned_normal_minus_repeat_current_psnr_db_min": 0.05,
                    "finetuned_normal_minus_shuffled_history_psnr_db_min": 0.05,
                    "history_gain_interaction_vs_base_reset_psnr_db_min": 0.03,
                    "finetuned_normal_vs_reset_temporal_error_relative_max": -0.01,
                    "both_validation_sequences_psnr_direction_positive": True,
                },
                "on_failure": "do not launch SLAM",
                "on_pass": "launch fixed SLAM smoke",
            },
            "implementation_pins": pins,
            "output_root": str(self.output_root),
        }

    def _metadata(self, role: str) -> dict:
        common = {
            "base_checkpoint_sha256": self.base_sha,
            "turtle_repo_commit": "repo-commit",
            "turtle_arch_sha256": "a" * 64,
            "turtle_config_sha256": "b" * 64,
            "input_domain": "raw",
            "cache_contract": OFFICIAL_CACHE_CONTRACT,
        }
        if role == "base":
            return {
                **common,
                "checkpoint_sha256": self.base_sha,
                "kind": "official_gopro",
                "format": "official_turtle.params",
            }
        return {
            **common,
            "checkpoint_sha256": self.fine_sha,
            "kind": "finetuned",
            "format": FINETUNED_CHECKPOINT_FORMAT,
            "uses_gt": False,
            "uses_gt_pose": False,
            "uses_gt_depth": False,
            "uses_sharp_rgb_supervision": True,
            "manifests": {
                "train": str(self.train_manifest),
                "train_sha256": sha256_file(self.train_manifest),
                "validation": None,
                "validation_sha256": None,
            },
            "loss": {"name": "l1_plus_fft_l1", "l1_weight": 1.0, "fft_weight": 0.1},
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 0.00001,
                "weight_decay": 0.001,
                "betas": [0.9, 0.9],
                "scheduler": "CosineAnnealingLR",
                "scheduler_eta_min": 1e-7,
                "scheduler_t_max_optimizer_steps": 126,
                "gradient_clip_norm": 1.0,
            },
            "history": {
                "mode": "official_incremental_kv",
                "sequence_boundary": "hard_reset",
                "backpropagation": "full_sequence",
            },
            "trainable_scope": {
                "scope": "history_attention",
                "parameter_tensors": 56,
                "parameter_count": 3475994,
            },
            "sequence_filter": {"minimum_length": 2, "loss_start_frame": 1},
            "augmentation": {
                "enabled": True,
                "shared_across_sequence_and_modalities": True,
                "crop_size": 128,
                "horizontal_flip": True,
                "vertical_flip": True,
                "quarter_turn_rotation": True,
                "sampling_policy": "shared_per_record; hflip_p=0.5; vflip_p=0.5; quarter_turn_uniform_0_1_2_3",
            },
            "training": {"seed": 42, "steps": 126, "amp": True},
        }

    def _source_metrics(self, role: str, sequence: str) -> dict[str, dict[str, float]]:
        del sequence
        if role == "base":
            values = {
                "raw": (28.0, 0.80, 0.040),
                NORMAL: (30.00, 0.900, 0.0200),
                RESET: (29.98, 0.899, 0.0202),
                REPEAT: (29.97, 0.898, 0.0203),
                ORDERED: (30.00, 0.900, 0.0200),
                SHUFFLED: (29.96, 0.897, 0.0204),
            }
        else:
            values = {
                "raw": (28.0, 0.80, 0.040),
                NORMAL: (30.10, 0.902, 0.0190),
                RESET: (30.00, 0.900, 0.0200),
                REPEAT: (30.00, 0.900, 0.0200),
                ORDERED: (30.10, 0.902, 0.0190),
                SHUFFLED: (30.00, 0.900, 0.0200),
            }
        return {
            source: {"psnr": value[0], "ssim": value[1], "l1": value[2]}
            for source, value in values.items()
        }

    def _temporal(self, role: str) -> dict[str, dict[str, float]]:
        error = {
            "raw": 0.030,
            NORMAL: 0.020 if role == "base" else 0.018,
            RESET: 0.0205 if role == "base" else 0.020,
            REPEAT: 0.0205 if role == "base" else 0.020,
            ORDERED: 0.020 if role == "base" else 0.018,
            SHUFFLED: 0.0205 if role == "base" else 0.020,
        }
        return {
            source: {
                "adjacent_change_l1": 0.025,
                "gt_temporal_difference_error_l1": value,
            }
            for source, value in error.items()
        }

    def _evaluation(self, role: str) -> dict:
        frames = []
        global_index = 0
        for record in self.validation_records:
            sequence = record["sequence"]
            for frame_index in range(4):
                frames.append(
                    {
                        "sequence": sequence,
                        "frame_index": frame_index,
                        "global_index": global_index,
                        "raw_path": str((self.data_root / record["blurry"][frame_index]).resolve()),
                        "gt_path": str((self.data_root / record["sharp"][frame_index]).resolve()),
                        "metrics": self._source_metrics(role, sequence),
                        "temporal": None if frame_index == 0 else self._temporal(role),
                    }
                )
                global_index += 1
        sources = list(REQUIRED_METRIC_SOURCES)
        means = {
            source: {
                metric: _mean([frame["metrics"][source][metric] for frame in frames])
                for metric in ("psnr", "ssim", "l1")
            }
            for source in sources
        }
        temporal_means = {
            source: {
                metric: _mean(
                    [
                        frame["temporal"][source][metric]
                        for frame in frames
                        if frame["temporal"] is not None
                    ]
                )
                for metric in (
                    "adjacent_change_l1",
                    "gt_temporal_difference_error_l1",
                )
            }
            for source in sources
        }
        steady = [frame for frame in frames if frame["frame_index"] >= 3]
        steady_means = {
            source: {
                metric: _mean([frame["metrics"][source][metric] for frame in steady])
                for metric in ("psnr", "ssim", "l1")
            }
            for source in (NORMAL, RESET, REPEAT, ORDERED, SHUFFLED)
        }
        steady_deltas = {
            control: {
                metric: steady_means[NORMAL][metric] - steady_means[control][metric]
                for metric in ("psnr", "ssim", "l1")
            }
            for control in (RESET, REPEAT, SHUFFLED)
        }
        checkpoint = self.base_checkpoint if role == "base" else self.fine_checkpoint
        checkpoint_sha = self.base_sha if role == "base" else self.fine_sha
        return {
            "schema": EVALUATION_SCHEMA,
            "frame_count": len(frames),
            "sequence_count": 2,
            "temporal_pair_count": 6,
            "sources": sources + ["gt"],
            "mean": means,
            "temporal": {"mean": temporal_means},
            "checkpoint_metadata": self._metadata(role),
            "provenance": {
                "manifest": str(self.validation_manifest),
                "manifest_sha256": sha256_file(self.validation_manifest),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "device": "cuda:0",
            },
            "history_ablation": {
                "protocol": {
                    "cache_source": "official TURTLE K/V only; use_both_input=false",
                    "cache_slots_per_kind": 8,
                    "populated_cache_slots_per_kind": 5,
                    "direct_cache_capacity_frames": [3, 3, 3, 3, 2],
                    "effective_history": "recurrent_full_prefix",
                    "ordered_control": "reset_then_replay_complete_past_prefix",
                    "repeat_current_control": "reset_then_replay_current_once_per_complete_past_frame",
                    "shuffled_control": "cyclic_left_shift_of_complete_past_prefix",
                    "steady_frame_index_min": 3,
                    "future_frames_used": False,
                },
                "ordered_replay_max_abs": 0.0,
                "ordered_replay_matches_stream": True,
                "steady_frame_count": 2,
                "steady_mean": steady_means,
                "steady_normal_minus_control": steady_deltas,
            },
            "frames": frames,
        }

    def rewrite(self) -> None:
        _write_json(self.base_path, self.base_payload)
        _write_json(self.fine_path, self.fine_payload)


class TurtleHistorySmokeReportTest(unittest.TestCase):
    def test_passing_report_computes_steady_interaction_temporal_and_per_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HistorySmokeFixture(Path(directory))
            report = build_report(fixture.contract_path, fixture.base_path, fixture.fine_path)
            self.assertEqual(report["schema"], REPORT_SCHEMA)
            self.assertTrue(report["eligible"])
            self.assertEqual(report["failed_gates"], [])
            self.assertEqual(report["metrics"]["base"]["all"]["frame_count"], 8)
            self.assertEqual(report["metrics"]["base"]["steady"]["frame_count"], 2)
            steady = report["metrics"]["comparisons"]["steady"]
            self.assertAlmostEqual(
                steady["history_gain_interaction_vs_base_reset_psnr_db"], 0.08
            )
            self.assertAlmostEqual(
                steady["finetuned_normal_vs_reset_temporal_difference_error_l1"]["relative_difference"],
                -0.1,
            )
            self.assertEqual(
                set(report["metrics"]["per_sequence_comparisons"]),
                {"validation_a", "validation_b"},
            )
            self.assertTrue(report["protocol"]["room2_read_or_evaluated"] is False)

    def test_valid_metric_failure_is_recorded_without_contract_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HistorySmokeFixture(Path(directory))
            for frame in fixture.fine_payload["frames"]:
                frame["metrics"][NORMAL]["psnr"] = frame["metrics"][RESET]["psnr"]
            # Keep the evaluator aggregate internally consistent after the
            # deliberately bad history result.
            fixture.fine_payload["mean"][NORMAL]["psnr"] = 30.0
            fixture.fine_payload["history_ablation"]["steady_mean"][NORMAL]["psnr"] = 30.0
            fixture.fine_payload["history_ablation"]["steady_normal_minus_control"][RESET]["psnr"] = 0.0
            fixture.fine_payload["history_ablation"]["steady_normal_minus_control"][REPEAT]["psnr"] = 0.0
            fixture.fine_payload["history_ablation"]["steady_normal_minus_control"][SHUFFLED]["psnr"] = 0.0
            fixture.rewrite()
            report = build_report(fixture.contract_path, fixture.base_path, fixture.fine_path)
            self.assertFalse(report["eligible"])
            self.assertIn(
                "finetuned_normal_minus_reset_cache_psnr_db_min",
                report["failed_gates"],
            )
            self.assertIn(
                "both_validation_sequences_psnr_direction_positive",
                report["failed_gates"],
            )

    def test_mismatched_frame_order_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HistorySmokeFixture(Path(directory))
            fixture.fine_payload["frames"][0], fixture.fine_payload["frames"][1] = (
                fixture.fine_payload["frames"][1],
                fixture.fine_payload["frames"][0],
            )
            fixture.rewrite()
            with self.assertRaisesRegex(HistorySmokeContractError, "frame_index"):
                build_report(fixture.contract_path, fixture.base_path, fixture.fine_path)

    def test_tampered_evaluator_aggregate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HistorySmokeFixture(Path(directory))
            fixture.base_payload["mean"][NORMAL]["psnr"] += 1.0
            fixture.rewrite()
            with self.assertRaisesRegex(HistorySmokeContractError, "aggregate mismatch"):
                build_report(fixture.contract_path, fixture.base_path, fixture.fine_path)

    def test_checkpoint_content_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HistorySmokeFixture(Path(directory))
            fixture.fine_checkpoint.write_bytes(b"changed after evaluation")
            with self.assertRaisesRegex(HistorySmokeContractError, "checkpoint bytes"):
                build_report(fixture.contract_path, fixture.base_path, fixture.fine_path)

    def test_ordered_replay_tolerance_is_checked_for_both_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HistorySmokeFixture(Path(directory))
            fixture.base_payload["history_ablation"]["ordered_replay_max_abs"] = 2.0e-5
            fixture.base_payload["history_ablation"]["ordered_replay_matches_stream"] = False
            fixture.rewrite()
            report = build_report(fixture.contract_path, fixture.base_path, fixture.fine_path)
            gate = report["gates"]["ordered_replay_max_abs_max"]
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["by_checkpoint"]["base"], 2.0e-5)

    def test_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            digest = write_report(output, {"schema": REPORT_SCHEMA})
            self.assertEqual(digest, hashlib.sha256(output.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                write_report(output, {"schema": REPORT_SCHEMA})


if __name__ == "__main__":
    unittest.main()
