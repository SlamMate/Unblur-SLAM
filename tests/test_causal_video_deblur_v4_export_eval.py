#!/usr/bin/env python3
"""CPU-only contract tests for v4 evaluation and export integration."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.export_causal_video_deblur as exporter
from scripts.export_causal_video_deblur import (
    ALIGNMENT_DIAGNOSTICS_SCHEMA_V4,
    CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1,
    CHECKPOINT_MIGRATION_KIND_V1,
    CHECKPOINT_MIGRATION_SCHEMA_V1,
    CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1,
    CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1,
    CHECKPOINT_FORMAT_V3,
    CHECKPOINT_FORMAT_V4,
    DEPLOYMENT_SELECTION_SCHEMA_V3,
    DEPLOYMENT_THRESHOLDS,
    EVALUATOR_SCHEMA_V4,
    OBJECTIVE_SCHEMA_V4,
    OPTIMIZATION_CONTRACT_SCHEMA_V4,
    REFINEMENT_SCHEMA_V4,
    REGISTERED_V4_CHECKPOINT_SEMANTIC_SHA256,
    REGISTERED_V4_CONTRACT_SCHEMA,
    REGISTERED_V4_CONTRACT_SHA256,
    REGISTERED_V4_DATA_IDENTITY,
    REGISTERED_V4_PRE_MIGRATION_CHECKPOINT_SHA256,
    REGISTERED_V4_SAFE_CHECKPOINT_SHA256,
    REGISTERED_V4_WARM_START_SHA256,
    RNG_STATE_SCHEMA_V4,
    TORCHSCRIPT_FORMAT_V4,
    TRAINING_CONTRACT_SCHEMA_V4,
    TRAINING_REQUIRED_SELECTOR_V4,
    WARM_START_SCHEMA_V4,
    _validate_v4_alignment_evidence,
    _validate_v4_layer_checkpoint_migration,
    checkpoint_semantic_digest,
    validate_registered_v4_checkpoint_migration,
    validate_v4_contracts,
)
from src.video_deblur import build_causal_video_deblur


SYNTHETIC_PRE_MIGRATION_SHA256 = "e" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _teacher_provenance(
    *, report: Path, manifest: Path, checkpoint: Path, frame_count: int
) -> dict[str, object]:
    return {
        "schema": "unblur_slam.video_deblur_teacher_provenance.v1",
        "storage": "precomputed_png_rgb8",
        "teacher_domain": "evssm_restored_rgb_0_1",
        "evssm_checkpoint_sha256": _sha256(checkpoint),
        "evssm_checkpoint": str(checkpoint.resolve()),
        "precompute_report": str(report.resolve()),
        "precompute_report_sha256": _sha256(report),
        "teacher_manifest": str(manifest.resolve()),
        "teacher_manifest_sha256": _sha256(manifest),
        "teacher_artifacts_verified": True,
        "sequence_count": 1,
        "frame_count": frame_count,
    }


def _make_v4_checkpoint(
    path: Path, teacher_provenance: dict[str, object]
) -> dict[str, object]:
    torch.manual_seed(41)
    config = {
        "channels": 32,
        "num_heads": 4,
        "num_blocks": 2,
        "max_history": 3,
        "use_teacher_input": False,
        "input_domain": "evssm",
        "max_residual": 8.0 / 255.0,
        "motion_alignment": {
            "mode": "coarse_local_correlation_v1",
            "match_channels": 16,
            "radius": 8,
            "temperature": 0.05,
        },
    }
    model = build_causal_video_deblur(config).eval()
    with torch.no_grad():
        model.motion_alignment_gate.fill_(0.2)
        model.output.weight.normal_(mean=0.0, std=0.001)
        model.output.bias.zero_()
    objective = {
        "schema": OBJECTIVE_SCHEMA_V4,
        "primary_reconstruction": {
            "frames": "latest_only",
            "l1_weight": 1.0,
            "fft_l1_weight": 0.1,
            "fft_normalization": "ortho",
            "phase": "joint_only",
        },
        "evssm_fidelity": {
            "frames": "all_causal_prefix_positions",
            "loss": "l1",
            "weight": 0.1,
            "phase": "joint_only",
        },
        "motion_alignment": {
            "flow_direction": "current_to_previous",
            "flow_units": "quarter_resolution_pixels",
            "real_transitions_only": True,
            "padding_transition_policy": "excluded_by_transition_valid",
            "photometric_weight": 1.0,
            "gradient_weight": 0.2,
            "smooth_weight": 0.01,
            "joint_phase_scale": 0.05,
            "confidence_weighted": False,
            "phase": "alignment_only_unscaled_and_joint_scaled",
        },
        "temporal_delta": {
            "frames": "two_shifted_full_history_windows",
            "reference": "motion_compensated_sharp_gt_difference",
            "loss": "l1",
            "weight": 0.05,
            "detached_flow": True,
            "flow_direction": "current_to_previous",
            "real_transitions_only": True,
            "phase": "joint_only",
        },
        "edge": {
            "frames": "latest_only",
            "operator": (
                "first_order_xy_plus_runtime_grayscale_zero_pad_laplacian"
            ),
            "loss": "sharp_gradient_and_laplacian_l1",
            "weight": 0.05,
            "phase": "joint_only",
        },
        "laplacian_gate": {
            "reference": "evssm_latest_frame",
            "loss": "relative_log_variance_hinge",
            "minimum_relative_gain": 0.0,
            "variance_floor": 1.0e-6,
            "runtime_gate_alignment": (
                "rgb_mean_then_four_neighbour_zero_pad_laplacian_unbiased_variance"
            ),
            "weight": 0.02,
            "phase": "joint_only",
        },
        "legacy_latest_evssm_distillation": {
            "loss": "l1",
            "weight": 0.0,
            "phase": "joint_only",
        },
    }
    phases = [
        {
            "name": "alignment_only",
            "start_step_inclusive": 0,
            "end_step_exclusive": 100,
            "optimizer_steps": 100,
            "base_trainable": False,
            "trainable_parameters": [
                "motion_aligner.match_projection.weight",
            ],
            "base_lr": 0.0,
            "alignment_lr": 2.0e-4,
        },
        {
            "name": "joint",
            "start_step_inclusive": 100,
            "end_step_exclusive": 600,
            "optimizer_steps": 500,
            "base_trainable": True,
            "trainable_parameters": [
                "base_parameters",
                "motion_aligner.match_projection.weight",
                "motion_alignment_gate",
            ],
            "base_lr": 2.0e-5,
            "alignment_lr": 2.0e-4,
        },
    ]
    payload = {
        "format": CHECKPOINT_FORMAT_V4,
        "model": model.state_dict(),
        "model_config": config,
        "registered_contract": {
            "schema": REGISTERED_V4_CONTRACT_SCHEMA,
            "path": str(
                ROOT
                / "configs/local/causal_evssm_v4_alignment_replica424_contract.json"
            ),
            "sha256": REGISTERED_V4_CONTRACT_SHA256,
        },
        "teacher_provenance": teacher_provenance,
        "warm_start_provenance": {
            "schema": WARM_START_SCHEMA_V4,
            "source_path": "/registered/v3/epoch_0020.pth",
            "source_sha256": REGISTERED_V4_WARM_START_SHA256,
            "source_format": CHECKPOINT_FORMAT_V3,
            "source_model_config": {
                key: value for key, value in config.items() if key != "motion_alignment"
            },
            "source_state_key_digest_sha256": "1" * 64,
            "copied_key_count": len(model.state_dict()) - 3,
            "allowed_missing_alignment_keys": [
                "motion_aligner.match_projection.weight",
                "motion_aligner.offsets",
                "motion_alignment_gate",
            ],
            "optimizer_state_loaded": False,
            "identity_probe": {
                "shape": [1, 3, 3, 12, 16],
                "atol": 1.0e-6,
                "rtol": 0.0,
                "max_abs_difference": 0.0,
                "passed": True,
            },
        },
        "training_contract": {
            "schema": TRAINING_CONTRACT_SCHEMA_V4,
            "supervised_output": "latest_frame_in_joint_phase",
            "temporal_output": "rolling_two_window_motion_diagnostics",
            "training_clip_length": 4,
            "rolling_window_length": 3,
            "stream_prefix_padding": "repeat_first_frame_on_left",
            "dropped_tail_policy": (
                "shuffle_then_drop_incomplete_microbatch_each_epoch"
            ),
            "terminal_checkpoint_policy": (
                "unconditional_atomic_save_at_exact_optimizer_step_600_before_exit"
            ),
            "resume_rng_policy": (
                "epoch_boundary_python_numpy_torch_cpu_and_loader_generators"
            ),
            "real_transition_slots": 2,
            "train_clips": 234,
            "train_sequences": 127,
            "unique_real_transitions": 107,
            "alignment_sampler_clips": 107,
            "alignment_sampler_policy": "clips_with_at_least_one_real_transition",
            "diagnostic_method": "forward_sequence_with_motion_diagnostics",
            "alignment_disabled_method": "forward_sequence_alignment_disabled",
        },
        "objective_contract": objective,
        "optimization_contract": {
            "schema": OPTIMIZATION_CONTRACT_SCHEMA_V4,
            "optimizer": "AdamW",
            "total_optimizer_steps": 600,
            "optimizer_steps_per_epoch": 1,
            "batch_size": 4,
            "num_workers": 0,
            "gradient_accumulation_micro_batches": 2,
            "effective_batch_size": 8,
            "drop_last": True,
            "alignment_loader_clips_per_epoch": 104,
            "alignment_loader_dropped_clips_per_epoch": 3,
            "alignment_micro_batches_per_epoch": 26,
            "joint_loader_clips_per_epoch": 232,
            "joint_loader_dropped_clips_per_epoch": 2,
            "joint_micro_batches_per_epoch": 58,
            "drop_incomplete_accumulation_group": False,
            "loader_generator_seeds": {
                "joint": 1042,
                "alignment_only": 2042,
            },
            "resume_boundary": "epoch_end_no_pending_accumulation",
            "resume_rng_state": (
                "python_numpy_torch_cpu_and_both_loader_generators"
            ),
            "schedule_unit": "optimizer_step",
            "lr_schedule": "fixed_by_phase",
            "optimizer_reset_at_phase_boundary": False,
            "optimizer_state_from_v3_loaded": False,
            "execution_device": "cpu",
            "amp_requested": False,
            "amp_effective": False,
            "weight_decay": 1.0e-3,
            "phases": phases,
        },
        "refinement_contract": {
            "schema": REFINEMENT_SCHEMA_V4,
            "base": "frozen_evssm_input",
            "formula": "output = input + max_residual * tanh(residual_logits)",
            "max_residual": 8.0 / 255.0,
            "bound_scope": "per_pixel_per_rgb_channel_normalized_0_1",
            "motion_alignment": config["motion_alignment"],
        },
        "checkpoint_selection": {
            "metric": "val_psnr",
            "mode": "max",
            "deployment_status": "not_deployment_selected",
            "required_deployment_selector": TRAINING_REQUIRED_SELECTOR_V4,
        },
        "validation_metrics": {"psnr": 30.0, "ssim": 0.9},
        "epoch": 25,
        "step": 600,
        "training_phase": "joint",
        "data_identity": dict(REGISTERED_V4_DATA_IDENTITY),
        "rng_state": {
            "schema": RNG_STATE_SCHEMA_V4,
            "checkpoint_boundary": "epoch_end_no_pending_accumulation",
            "python_random_state": ("synthetic",),
            "numpy_random_state": (
                "MT19937",
                torch.zeros(624, dtype=torch.int64),
                0,
                0,
                0.0,
            ),
            "numpy_random_state_encoding": (
                "numpy.random.RandomState.MT19937.keys_torch_int64.v1"
            ),
            "torch_cpu_rng_state": torch.arange(8, dtype=torch.uint8),
            "train_loader_generator_state": torch.arange(8, dtype=torch.uint8),
            "alignment_loader_generator_state": torch.arange(
                8, dtype=torch.uint8
            ),
        },
    }
    semantic_digest = checkpoint_semantic_digest(payload)
    payload["checkpoint_migration"] = {
        "schema": CHECKPOINT_MIGRATION_SCHEMA_V1,
        "kind": CHECKPOINT_MIGRATION_KIND_V1,
        "source_checkpoint_sha256": SYNTHETIC_PRE_MIGRATION_SHA256,
        "allowed_changes": list(CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1),
        "semantic_digest": {
            "schema": CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1,
            "algorithm": CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1,
            "sha256": semantic_digest,
            "source_and_target_equal": True,
        },
    }
    torch.save(payload, path)
    return payload


def _write_tiny_cache(root: Path) -> tuple[Path, dict[str, object]]:
    checkpoint = root / "tiny_official_evssm.pth"
    checkpoint.write_bytes(b"tiny deterministic EVSSM checkpoint")
    blurry_paths = []
    sharp_paths = []
    teacher_paths = []
    frames = []
    height, width = 16, 20
    grid = np.arange(height * width * 3, dtype=np.uint16).reshape(
        height, width, 3
    )
    for index in range(3):
        blurry = root / f"blurry_{index}.png"
        sharp = root / f"sharp_{index}.png"
        teacher = root / f"teacher_{index}.png"
        Image.fromarray(((grid + index * 5) % 256).astype(np.uint8)).save(blurry)
        Image.fromarray(((grid + index * 5 + 2) % 256).astype(np.uint8)).save(sharp)
        Image.fromarray(((grid + index * 5 + 1) % 256).astype(np.uint8)).save(teacher)
        blurry_paths.append(blurry.name)
        sharp_paths.append(sharp.name)
        teacher_paths.append(teacher.name)
        frames.append(
            {
                "sequence_index": 0,
                "frame_index": index,
                "blurry": str(blurry.resolve()),
                "blurry_sha256": _sha256(blurry),
                "sharp": str(sharp.resolve()),
                "sharp_sha256": _sha256(sharp),
                "teacher": str(teacher.resolve()),
                "teacher_sha256": _sha256(teacher),
            }
        )
    manifest = root / "tiny.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence": "tiny-v4",
                "blurry": blurry_paths,
                "sharp": sharp_paths,
                "teacher": teacher_paths,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = root / "precompute_report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "unblur_slam.video_deblur_evssm_precompute.v1",
                "input_manifest": str(manifest.resolve()),
                "input_manifest_sha256": _sha256(manifest),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "output_manifest": str(manifest.resolve()),
                "output_manifest_sha256": _sha256(manifest),
                "sequence_count": 1,
                "frame_count": 3,
                "frames": frames,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report, _teacher_provenance(
        report=report, manifest=manifest, checkpoint=checkpoint, frame_count=3
    )


def _synthetic_registered_pins(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return mock.patch.multiple(
        exporter,
        REGISTERED_V4_PRE_MIGRATION_CHECKPOINT_SHA256=(
            payload["checkpoint_migration"]["source_checkpoint_sha256"]
        ),
        REGISTERED_V4_SAFE_CHECKPOINT_SHA256=_sha256(checkpoint),
        REGISTERED_V4_CHECKPOINT_SEMANTIC_SHA256=(
            payload["checkpoint_migration"]["semantic_digest"]["sha256"]
        ),
    )


def _diagnostic_export(checkpoint: Path, output: Path) -> None:
    argv = [
            str(ROOT / "scripts/export_causal_video_deblur.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--diagnostic-output",
            "--verify-height",
            "12",
            "--verify-width",
            "16",
        ]
    with _synthetic_registered_pins(checkpoint), mock.patch.object(
        sys, "argv", argv
    ), contextlib.redirect_stdout(io.StringIO()):
        exporter.main()


def test_v4_preregistration_is_content_pinned() -> None:
    contract_path = (
        ROOT
        / "configs/local/causal_evssm_v4_alignment_replica424_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["schema"] == REGISTERED_V4_CONTRACT_SCHEMA
    assert contract["registered_before_training"] is True
    assert _sha256(contract_path) == REGISTERED_V4_CONTRACT_SHA256
    assert contract["warm_start"]["checkpoint_sha256"] == (
        REGISTERED_V4_WARM_START_SHA256
    )
    assert REGISTERED_V4_PRE_MIGRATION_CHECKPOINT_SHA256 == (
        "ad80e84f67f6c979de96ce2a65ceeb7201b2cf7f7159af64d8fe2c2face030e0"
    )
    assert REGISTERED_V4_SAFE_CHECKPOINT_SHA256 == (
        "92a1ab5301355e923fbd8c2059bbb0c5bdbe041cc00880b21591efdfd7de5bfd"
    )
    assert REGISTERED_V4_CHECKPOINT_SEMANTIC_SHA256 == (
        "a533ca551efc7543034ee73b64539c2056c913dcc4f9183df8ebbec4426c2c9d"
    )
    assert contract["warm_start"]["fresh_optimizer_required"] is True
    assert contract["data"]["train"]["pairs"] == 234
    assert contract["data"]["temporal_validation"]["pairs"] == 16
    assert contract["data"]["room2_test"]["pairs"] == 174
    assert contract["optimization"]["alignment_only_steps"] == 100
    assert contract["optimization"]["joint_steps"] == 500
    assert contract["optimization"]["weight_decay"] == 1.0e-3
    assert contract["optimization"]["execution_device"] == "cpu"
    assert contract["optimization"]["amp_requested"] is False
    assert contract["optimization"]["amp_effective"] is False
    assert contract["optimization"]["drop_last"] is True
    assert contract["optimization"]["num_workers"] == 0
    assert contract["optimization"]["alignment_loader_clips_per_epoch"] == 104
    assert contract["optimization"]["alignment_loader_dropped_clips_per_epoch"] == 3
    assert contract["optimization"]["alignment_micro_batches_per_epoch"] == 26
    assert contract["optimization"]["joint_loader_clips_per_epoch"] == 232
    assert contract["optimization"]["joint_loader_dropped_clips_per_epoch"] == 2
    assert contract["optimization"]["joint_micro_batches_per_epoch"] == 58
    assert contract["optimization"]["drop_incomplete_accumulation_group"] is False
    assert contract["training"]["dropped_tail_policy"] == (
        "shuffle_then_drop_incomplete_microbatch_each_epoch"
    )
    assert contract["training"]["terminal_checkpoint_policy"] == (
        "unconditional_atomic_save_at_exact_optimizer_step_600_before_exit"
    )
    assert contract["optimization"]["phases"][0]["trainable_parameters"] == [
        "motion_aligner.match_projection.weight"
    ]
    assert contract["optimization"]["phases"][1]["trainable_parameters"] == [
        "base_parameters",
        "motion_aligner.match_projection.weight",
        "motion_alignment_gate",
    ]
    assert contract["selection"]["temporal_validation"] == (
        DEPLOYMENT_THRESHOLDS["temporal_validation"]
    )
    assert contract["selection"][
        "room2_must_not_be_read_or_precomputed_before_layer1_pass"
    ] is True


def test_v4_registered_migration_and_layer_lineage_fail_closed(
    tmp_path: Path,
) -> None:
    _, provenance = _write_tiny_cache(tmp_path)
    checkpoint = tmp_path / "lineage_v4.pth"
    payload = _make_v4_checkpoint(checkpoint, provenance)
    target_sha256 = _sha256(checkpoint)

    with _synthetic_registered_pins(checkpoint):
        lineage = validate_registered_v4_checkpoint_migration(
            payload, target_sha256
        )
    assert lineage["source_checkpoint_sha256"] == (
        SYNTHETIC_PRE_MIGRATION_SHA256
    )
    assert lineage["target_checkpoint_sha256"] == target_sha256
    assert lineage["semantic_digest"]["sha256"] == checkpoint_semantic_digest(
        payload
    )

    layer = {
        "v4_provenance": {
            "source_checkpoint_sha256": target_sha256,
            "checkpoint_migration": copy.deepcopy(lineage),
        }
    }
    assert _validate_v4_layer_checkpoint_migration(layer, lineage) == lineage

    missing = copy.deepcopy(payload)
    del missing["checkpoint_migration"]
    with _synthetic_registered_pins(checkpoint):
        try:
            validate_registered_v4_checkpoint_migration(missing, target_sha256)
        except ValueError as error:
            assert "requires the audited checkpoint_migration" in str(error)
        else:
            raise AssertionError("missing migration lineage passed formal validation")

    wrong_source = copy.deepcopy(payload)
    wrong_source["checkpoint_migration"]["source_checkpoint_sha256"] = "d" * 64
    with _synthetic_registered_pins(checkpoint):
        try:
            validate_registered_v4_checkpoint_migration(
                wrong_source, target_sha256
            )
        except ValueError as error:
            assert "source is not the pinned formal terminal" in str(error)
        else:
            raise AssertionError("forged migration source passed formal validation")

    self_consistent_tamper = copy.deepcopy(payload)
    self_consistent_tamper["model"]["motion_alignment_gate"] = (
        self_consistent_tamper["model"]["motion_alignment_gate"] + 0.25
    )
    self_consistent_tamper["checkpoint_migration"]["semantic_digest"][
        "sha256"
    ] = checkpoint_semantic_digest(self_consistent_tamper)
    with _synthetic_registered_pins(checkpoint):
        try:
            validate_registered_v4_checkpoint_migration(
                self_consistent_tamper, target_sha256
            )
        except ValueError as error:
            assert "semantic digest is not the audited value" in str(error)
        else:
            raise AssertionError("self-reported tampered semantics passed the pin")

    for malformed in (
        {},
        {
            "v4_provenance": {
                "source_checkpoint_sha256": "0" * 64,
                "checkpoint_migration": copy.deepcopy(lineage),
            }
        },
        {
            "v4_provenance": {
                "source_checkpoint_sha256": target_sha256,
                "checkpoint_migration": {
                    **copy.deepcopy(lineage),
                    "target_checkpoint_sha256": "0" * 64,
                },
            }
        },
    ):
        try:
            _validate_v4_layer_checkpoint_migration(malformed, lineage)
        except ValueError:
            pass
        else:
            raise AssertionError("forged Layer1 checkpoint lineage was accepted")


def test_v4_diagnostic_export_and_v3_selection_rejection(tmp_path: Path) -> None:
    report, provenance = _write_tiny_cache(tmp_path)
    del report
    checkpoint = tmp_path / "tiny_v4.pth"
    payload = _make_v4_checkpoint(checkpoint, provenance)
    safe_loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert safe_loaded["rng_state"]["numpy_random_state"][1].dtype == torch.int64
    wrong_config = dict(payload["model_config"])
    wrong_config["channels"] = 16
    try:
        validate_v4_contracts(payload, wrong_config)
    except ValueError as error:
        assert "architecture" in str(error)
    else:
        raise AssertionError("wrong v4 architecture passed export validation")
    original_source_config = payload["warm_start_provenance"][
        "source_model_config"
    ]
    payload["warm_start_provenance"]["source_model_config"] = {
        **original_source_config,
        "num_blocks": 1,
    }
    try:
        validate_v4_contracts(payload, payload["model_config"])
    except ValueError as error:
        assert "source_model_config" in str(error)
    else:
        raise AssertionError("wrong warm-start architecture passed validation")
    payload["warm_start_provenance"]["source_model_config"] = original_source_config
    payload["step"] = 599
    try:
        validate_v4_contracts(payload, payload["model_config"])
    except ValueError as error:
        assert "terminal optimizer step 600" in str(error)
    else:
        raise AssertionError("nonterminal v4 checkpoint passed export validation")
    payload["step"] = 600
    original_data_identity = payload["data_identity"]
    payload["data_identity"] = {
        **original_data_identity,
        "val_manifest_sha256": "0" * 64,
    }
    try:
        validate_v4_contracts(payload, payload["model_config"])
    except ValueError as error:
        assert "data_identity" in str(error)
    else:
        raise AssertionError("wrong v4 data identity passed export validation")
    payload["data_identity"] = original_data_identity
    payload["rng_state"]["checkpoint_boundary"] = "mid_accumulation"
    try:
        validate_v4_contracts(payload, payload["model_config"])
    except ValueError as error:
        assert "RNG boundary" in str(error)
    else:
        raise AssertionError("invalid v4 RNG boundary passed export validation")
    payload["rng_state"]["checkpoint_boundary"] = (
        "epoch_end_no_pending_accumulation"
    )
    try:
        validate_v4_contracts(
            payload,
            payload["model_config"],
            checkpoint_sha256=_sha256(checkpoint),
        )
    except ValueError as error:
        assert "pinned tensor-safe checkpoint" in str(error)
    else:
        raise AssertionError("synthetic checkpoint bypassed the formal target pin")
    with _synthetic_registered_pins(checkpoint):
        validate_v4_contracts(
            payload,
            payload["model_config"],
            checkpoint_sha256=_sha256(checkpoint),
        )
    artifact = tmp_path / "tiny_v4.diagnostic.pt"
    _diagnostic_export(checkpoint, artifact)
    extra_files = {"metadata.json": ""}
    loaded = torch.jit.load(
        str(artifact), map_location="cpu", _extra_files=extra_files
    )
    metadata_value = extra_files["metadata.json"]
    if isinstance(metadata_value, bytes):
        metadata_value = metadata_value.decode("utf-8")
    metadata = json.loads(metadata_value)
    assert metadata["format"] == TORCHSCRIPT_FORMAT_V4
    assert metadata["deployment_eligible"] is False
    assert metadata["artifact_role"] == "diagnostic_evaluation_only"
    assert metadata["source_checkpoint_epoch"] == 25
    assert metadata["source_checkpoint_step"] == 600
    assert metadata["training_phase"] == "joint"
    assert metadata["data_identity"] == REGISTERED_V4_DATA_IDENTITY
    assert metadata["rng_state_provenance"] == {
        "schema": RNG_STATE_SCHEMA_V4,
        "checkpoint_boundary": "epoch_end_no_pending_accumulation",
        "captured": True,
    }
    assert set(metadata["exported_methods"]) == {
        "forward",
        "forward_sequence",
        "forward_sequence_with_motion_diagnostics",
        "forward_sequence_alignment_disabled",
    }
    frames = torch.rand(1, 3, 3, 12, 16)
    diagnostics = loaded.forward_sequence_with_motion_diagnostics(frames)
    disabled = loaded.forward_sequence_alignment_disabled(frames)
    assert diagnostics[0].shape == frames.shape
    assert diagnostics[1].shape[:3] == (1, 2, 2)
    assert disabled.shape == frames.shape

    v3_selection = tmp_path / "v3_selection.json"
    v3_selection.write_text(
        json.dumps({"schema": DEPLOYMENT_SELECTION_SCHEMA_V3}) + "\n",
        encoding="utf-8",
    )
    forbidden_output = tmp_path / "must_not_export.pt"
    argv = [
            str(ROOT / "scripts/export_causal_video_deblur.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(forbidden_output),
            "--selection-report",
            str(v3_selection),
            "--verify-height",
            "12",
            "--verify-width",
            "16",
        ]
    try:
        with _synthetic_registered_pins(checkpoint), mock.patch.object(
            sys, "argv", argv
        ), contextlib.redirect_stdout(io.StringIO()):
            exporter.main()
    except ValueError as error:
        assert "v3 reports are forbidden" in str(error)
    else:
        raise AssertionError("v3 selection report authorized v4 deployment export")
    assert not forbidden_output.exists()


def test_v4_evaluator_reports_motion_and_both_controls(tmp_path: Path) -> None:
    report, provenance = _write_tiny_cache(tmp_path)
    checkpoint = tmp_path / "tiny_v4.pth"
    _make_v4_checkpoint(checkpoint, provenance)
    artifact = tmp_path / "tiny_v4.diagnostic.pt"
    _diagnostic_export(checkpoint, artifact)
    output = tmp_path / "evaluation"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_causal_video_deblur.py"),
            "--precompute-report",
            str(report),
            "--checkpoint",
            str(artifact),
            "--output-dir",
            str(output),
            "--history",
            "3",
            "--device",
            "cpu",
            "--max-visuals",
            "1",
        ],
        check=True,
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema"] == EVALUATOR_SCHEMA_V4
    assert metrics["frame_count"] == 3
    assert metrics["temporal_pair_count"] == 2
    assert metrics["transition_count"] == 2
    assert set(metrics["mean"]) == {
        "blurry",
        "evssm",
        "causal",
        "causal_repeat_current",
    }
    assert "causal_minus_repeat_current" in metrics["history_ablation"]
    assert "causal_alignment_disabled" in metrics["frames"][0]
    assert metrics["frames"][0]["motion_alignment"] is None
    for index in (1, 2):
        transition = metrics["frames"][index]["motion_alignment"]
        assert transition["from_frame_index"] == index - 1
        assert transition["to_frame_index"] == index
        assert transition["finite_fraction"] == 1.0
        assert transition["flow_quarter_pixels"]["magnitude_p95"] <= (
            transition["flow_quarter_pixels"]["magnitude_max"]
        )
        assert transition["flow_input_pixels"]["magnitude_p95"] <= (
            transition["flow_input_pixels"]["magnitude_max"]
        )
    alignment = metrics["alignment_diagnostics"]
    assert alignment["schema"] == ALIGNMENT_DIAGNOSTICS_SCHEMA_V4
    assert alignment["transition_count"] == 2
    assert alignment["integrity"]["passed"] is True
    assert alignment["flow_quarter_pixels"]["component_abs_max"] <= 16.0
    assert 0.0 <= alignment["confidence"]["mean"] <= 1.0
    assert 0.0 <= alignment["warp_valid"]["mean"] <= 1.0
    assert set(alignment["controls"]) == {
        "repeat_current",
        "alignment_disabled",
    }
    assert metrics["alignment_disabled_control"] == alignment["controls"][
        "alignment_disabled"
    ]
    normalized = _validate_v4_alignment_evidence(
        metrics,
        label="tiny v4 temporal evaluator",
        expected_transition_count=2,
        require_lpips=False,
    )
    assert normalized["transition_count"] == 2
    assert normalized["integrity_passed"] is True


if __name__ == "__main__":
    test_v4_preregistration_is_content_pinned()
    with tempfile.TemporaryDirectory() as directory:
        test_v4_registered_migration_and_layer_lineage_fail_closed(
            Path(directory)
        )
    with tempfile.TemporaryDirectory() as directory:
        test_v4_diagnostic_export_and_v3_selection_rejection(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_v4_evaluator_reports_motion_and_both_controls(Path(directory))
    print("PASS test_causal_video_deblur_v4_export_eval")
