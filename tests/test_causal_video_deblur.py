#!/usr/bin/env python3
"""CPU smoke tests for the causal video-deblurring training/export contract."""

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_deblur import (
    CausalVideoDeblur,
    VideoDeblurJsonlDataset,
    build_causal_video_deblur,
)
from src.video_deblur.dataset import load_evssm_precompute_report
from src.deblur_backends import (
    CausalEVSSMBackend,
    CausalTorchScriptBackend,
    EVSSMBackend,
)
from scripts.train_causal_video_deblur import (
    evssm_fidelity_l1_loss,
    fft_l1_loss,
    rolling_window_temporal_delta_l1_loss,
    run_teacher,
    runtime_laplacian_logvar_hinge_loss,
    runtime_laplacian_variance,
    spatial_gradient_l1_loss,
    validate_teacher_options,
)
from scripts.export_causal_video_deblur import (
    DEPLOYMENT_LAYER_REPORT_SCHEMA_V1,
    DEPLOYMENT_SELECTION_POLICY_V1,
    DEPLOYMENT_SELECTION_SCHEMA_V3,
    DEPLOYMENT_THRESHOLDS,
    EVALUATOR_SCHEMA_V3,
    EXPECTED_LPIPS_PROTOCOL,
    ORACLE_GOOD_DEFINITION,
    REGISTERED_CONTRACT_SCHEMA,
    REGISTERED_CONTRACT_SHA256,
    REGISTERED_EVSSM_SHA256,
    ROOM2_ONE_SHOT_MANIFEST_SHA256,
    ROOM2_ONE_SHOT_FRAME_COUNT,
    ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256,
    TEMPORAL_VALIDATION_MANIFEST_SHA256,
    validate_deployment_selection,
)


def make_model(
    use_teacher_input: bool = False, input_domain: str = "raw"
) -> CausalVideoDeblur:
    torch.manual_seed(7)
    return CausalVideoDeblur(
        channels=4,
        num_heads=2,
        num_blocks=1,
        max_history=4,
        use_teacher_input=use_teacher_input,
        input_domain=input_domain,
    )


def test_shape_gradient_and_last_frame_contract() -> None:
    model = make_model()
    frames = torch.rand(2, 4, 3, 12, 16, requires_grad=True)
    sequence = model.forward_sequence(frames)
    newest = model(frames)
    assert sequence.shape == (2, 4, 3, 12, 16)
    assert newest.shape == (2, 3, 12, 16)
    assert torch.allclose(newest, sequence[:, -1], atol=0.0, rtol=0.0)
    newest.square().mean().backward()
    assert frames.grad is not None
    assert bool(torch.isfinite(frames.grad).all())
    learned_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert learned_gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in learned_gradients)
    assert model.output.weight.grad is not None
    assert float(model.output.weight.grad.abs().sum()) > 0.0


def test_v3_identity_initialization_and_hard_residual_bound() -> None:
    model = make_model(input_domain="evssm").eval()
    frames = torch.rand(2, 4, 3, 12, 16)
    with torch.no_grad():
        initial = model.forward_sequence(frames)
    assert torch.equal(initial, frames)
    assert model.config_dict()["max_residual"] == 8.0 / 255.0

    # Even adversarially large logits cannot escape the serialized bound.
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(100.0)
        refined = model.forward_sequence(frames)
    correction = refined - frames
    assert float(correction.abs().max()) <= model.max_residual + 1.0e-7
    assert float(correction.abs().min()) > 0.0


def test_legacy_config_without_bound_preserves_v1_formula() -> None:
    legacy_config = {
        "channels": 4,
        "num_heads": 2,
        "num_blocks": 1,
        "max_history": 4,
        "use_teacher_input": False,
        "input_domain": "evssm",
    }
    model = build_causal_video_deblur(legacy_config).eval()
    assert model.max_residual == 0.0
    frames = torch.rand(1, 4, 3, 12, 16)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(0.2)
        output = model.forward_sequence(frames)
    # v1 was input + raw output-head logits; it must not be silently rescaled
    # when loading a checkpoint whose config predates max_residual.
    assert torch.allclose(output, frames + 0.2, atol=1.0e-6, rtol=0.0)


def test_v3_objective_terms_are_prefix_safe_and_numerically_bound() -> None:
    torch.manual_seed(11)
    prediction = torch.rand(2, 4, 3, 8, 10, requires_grad=True)
    target = torch.rand_like(prediction)
    manual_prediction_fft = torch.fft.fft2(
        prediction[:, -1].float(), dim=(-2, -1), norm="ortho"
    )
    manual_target_fft = torch.fft.fft2(
        target[:, -1].float(), dim=(-2, -1), norm="ortho"
    )
    manual_fft = torch.nn.functional.l1_loss(
        torch.stack(
            (manual_prediction_fft.real, manual_prediction_fft.imag), dim=-1
        ),
        torch.stack((manual_target_fft.real, manual_target_fft.imag), dim=-1),
    )
    measured_fft = fft_l1_loss(prediction[:, -1], target[:, -1])
    assert torch.allclose(measured_fft, manual_fft, atol=0.0, rtol=0.0)

    repeated = torch.rand(2, 1, 3, 8, 10).repeat(1, 5, 1, 1, 1)
    previous_prediction = repeated[:, -2]
    current_prediction = repeated[:, -1]
    assert float(
        rolling_window_temporal_delta_l1_loss(
            current_prediction, previous_prediction, repeated
        )
    ) == 0.0
    changed_latest = current_prediction + 0.25
    assert float(
        rolling_window_temporal_delta_l1_loss(
            changed_latest, previous_prediction, repeated
        )
    ) > 0.0

    available = torch.ones(2, dtype=torch.bool)
    assert float(
        evssm_fidelity_l1_loss(repeated[:, 1:], repeated[:, 1:], available)
    ) == 0.0
    edge = spatial_gradient_l1_loss(prediction[:, -1], target[:, -1])
    total = measured_fft + edge + rolling_window_temporal_delta_l1_loss(
        prediction[:, -1], prediction[:, -2], target
    )
    total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())

    checkerboard = (
        torch.arange(8).view(1, 1, 8, 1)
        + torch.arange(10).view(1, 1, 1, 10)
    ) % 2
    evssm = checkerboard.float().repeat(1, 3, 1, 1)
    flat = torch.full_like(evssm, 0.5)
    assert float(runtime_laplacian_logvar_hinge_loss(evssm, evssm)) == 0.0
    assert float(runtime_laplacian_logvar_hinge_loss(flat, evssm)) > 0.0
    from thirdparty.monogs.utils.slam_utils import variance_of_laplacian

    runtime_variance, _ = variance_of_laplacian(evssm)
    training_variance = float(runtime_laplacian_variance(evssm)[0].item())
    assert abs(runtime_variance - training_variance) < 1.0e-6


def test_future_frames_do_not_change_past_outputs() -> None:
    model = make_model().eval()
    original = torch.rand(1, 4, 3, 12, 16)
    changed_future = original.clone()
    changed_future[:, 2:] = torch.rand_like(changed_future[:, 2:]) * 10.0 - 5.0
    with torch.no_grad():
        original_outputs = model.forward_sequence(original)
        changed_outputs = model.forward_sequence(changed_future)
    # Outputs 0 and 1 cannot see modified frames 2 and 3.
    assert torch.allclose(
        original_outputs[:, :2], changed_outputs[:, :2], atol=1.0e-6, rtol=1.0e-6
    )


def test_teacher_path_and_torchscript() -> None:
    model = make_model(use_teacher_input=True).eval()
    with torch.no_grad():
        model.teacher_gate.fill_(0.5)
        model.output.weight.normal_(std=0.01)
    frames = torch.rand(1, 4, 3, 12, 16)
    teacher = torch.rand_like(frames)
    with torch.no_grad():
        output_without_teacher = model(frames)
        output_with_teacher = model(frames, teacher)
    assert not torch.allclose(output_without_teacher, output_with_teacher)

    scripted = torch.jit.script(model)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "causal_video_deblur.pt"
        scripted.save(str(path))
        loaded = torch.jit.load(str(path), map_location="cpu")
        with torch.no_grad():
            scripted_output = loaded(frames, teacher)
    assert torch.allclose(output_with_teacher, scripted_output, atol=1.0e-6, rtol=1.0e-6)


def _save_image(path: Path, value: int) -> None:
    image = np.full((14, 18, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_deployment_selection(
    root: Path,
    checkpoint: Path,
    evssm_checkpoint_sha256: str,
) -> Path:
    checkpoint_sha256 = _sha256(checkpoint)
    temporal_manifest = root / "temporal_manifest.jsonl"
    temporal_manifest.write_text("{}\n", encoding="utf-8")
    h3_artifact = root / "h3_evaluated.pt"
    h3_artifact.write_bytes(b"evaluated-H3-artifact")
    h1_source_checkpoint = root / "history1_source.pth"
    h1_source_checkpoint.write_bytes(b"independent-H1-source-checkpoint")
    h1_checkpoint_sha256 = _sha256(h1_source_checkpoint)
    h1_artifact = root / "history1_evaluated.pt"
    h1_artifact.write_bytes(b"evaluated-H1-artifact")

    def evaluator_frame(index: int, *, history: int, psnr: float) -> dict:
        return {
            "sequence": "selector-smoke",
            "frame_index": index,
            "history_stage": (
                "prefix" if index < history - 1 else "steady_state"
            ),
            "blurry_path": f"/dataset/selector-smoke/blur/{index:06d}.png",
            "sharp_path": f"/dataset/selector-smoke/sharp/{index:06d}.png",
            "causal": {"psnr": psnr},
        }

    teacher_provenance = {
        "evssm_checkpoint_sha256": evssm_checkpoint_sha256,
    }
    h3_evaluator = root / "h3_temporal_evaluator.json"
    h3_evaluator.write_text(
        json.dumps(
            {
                "schema": EVALUATOR_SCHEMA_V3,
                "checkpoint": str(h3_artifact),
                "evaluated_artifact_sha256": _sha256(h3_artifact),
                "source_checkpoint_sha256": checkpoint_sha256,
                "manifest": str(temporal_manifest),
                "teacher_provenance": teacher_provenance,
                "input_domain": "evssm",
                "history": 3,
                "frame_count": 3,
                "frames": [
                    evaluator_frame(0, history=3, psnr=29.8),
                    evaluator_frame(1, history=3, psnr=29.9),
                    evaluator_frame(2, history=3, psnr=30.1),
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    h1_evaluator = root / "history1_temporal_evaluator.json"
    h1_evaluator.write_text(
        json.dumps(
            {
                "schema": EVALUATOR_SCHEMA_V3,
                "checkpoint": str(h1_artifact),
                "evaluated_artifact_sha256": _sha256(h1_artifact),
                "source_checkpoint_sha256": h1_checkpoint_sha256,
                "manifest": str(temporal_manifest),
                "teacher_provenance": teacher_provenance,
                "input_domain": "evssm",
                "history": 1,
                "frame_count": 3,
                "frames": [
                    evaluator_frame(0, history=1, psnr=29.7),
                    evaluator_frame(1, history=1, psnr=29.8),
                    evaluator_frame(2, history=1, psnr=30.0),
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    steady_frame_keys_sha256 = hashlib.sha256(
        json.dumps(
            [["selector-smoke", 2]],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    temporal_report = root / "temporal_validation_layer.json"
    temporal_report.write_text(
        json.dumps(
            {
                "schema": DEPLOYMENT_LAYER_REPORT_SCHEMA_V1,
                "layer": "temporal_validation",
                "role": "checkpoint_and_history_selection",
                "registered_contract_sha256": REGISTERED_CONTRACT_SHA256,
                "checkpoint_sha256": checkpoint_sha256,
                "evssm_checkpoint_sha256": evssm_checkpoint_sha256,
                "manifest_sha256": TEMPORAL_VALIDATION_MANIFEST_SHA256,
                "thresholds": DEPLOYMENT_THRESHOLDS["temporal_validation"],
                "eligible": True,
                "metrics": {
                    "steady_psnr_delta_db": 0.11,
                    "steady_ssim_delta": 0.001,
                    "steady_relative_l1_delta": -0.006,
                    "steady_gt_temporal_difference_relative_delta": -0.02,
                    "normal_vs_repeat_current_psnr_delta_db": 0.06,
                    "normal_vs_repeat_current_temporal_relative_delta": -0.02,
                    "normal_vs_history1_psnr_delta_db": 0.1,
                    "laplacian_gate_pass_ratio": 0.25,
                    "accepted_oracle_precision": 0.8,
                    "worst_run_psnr_delta_db": -0.1,
                    "prefix_psnr_delta_db": -0.02,
                },
                "evaluator_report": {
                    "path": str(h3_evaluator),
                    "sha256": _sha256(h3_evaluator),
                },
                "history1_control": {
                    "checkpoint_sha256": h1_checkpoint_sha256,
                    "evaluator_report": str(h1_evaluator),
                    "evaluator_report_sha256": _sha256(h1_evaluator),
                    "evaluated_artifact": str(h1_artifact),
                    "evaluated_artifact_sha256": _sha256(h1_artifact),
                    "manifest_sha256": TEMPORAL_VALIDATION_MANIFEST_SHA256,
                    "evssm_checkpoint_sha256": evssm_checkpoint_sha256,
                    "history": 1,
                    "h3_steady_psnr_db": 30.1,
                    "h1_steady_psnr_db": 30.0,
                    "steady_frame_keys_sha256": steady_frame_keys_sha256,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporal_report_sha256 = _sha256(temporal_report)

    room2_manifest = Path(
        "/srv/szha0669/unblur-slam/causal_video_data/replica424_v1/"
        "manifests/test_room2.jsonl"
    )
    assert room2_manifest.is_file()
    assert _sha256(room2_manifest) == ROOM2_ONE_SHOT_MANIFEST_SHA256
    room2_source_root = Path(
        "/srv/szha0669/unblur-slam/causal_video_data"
    )
    room2_rows = []
    room2_sequence_lengths = {}
    for line in room2_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sequence = record["sequence"]
        room2_sequence_lengths[sequence] = len(record["blurry"])
        for frame_index, (blurry_value, sharp_value) in enumerate(
            zip(record["blurry"], record["sharp"])
        ):
            blurry_path = Path(blurry_value).expanduser()
            sharp_path = Path(sharp_value).expanduser()
            if not blurry_path.is_absolute():
                blurry_path = room2_source_root / blurry_path
            if not sharp_path.is_absolute():
                sharp_path = room2_source_root / sharp_path
            temporal = (
                None
                if frame_index == 0
                else {
                    "evssm": {"gt_difference_error_l1_not_warp": 0.1},
                    "causal": {"gt_difference_error_l1_not_warp": 0.09},
                    "causal_repeat_current": {
                        "gt_difference_error_l1_not_warp": 0.095
                    },
                }
            )
            room2_rows.append(
                {
                    "sequence": sequence,
                    "frame_index": frame_index,
                    "history_stage": (
                        "prefix" if frame_index < 2 else "steady_state"
                    ),
                    "blurry_path": str(blurry_path.resolve()),
                    "sharp_path": str(sharp_path.resolve()),
                    "evssm": {
                        "psnr": 30.0,
                        "ssim": 0.9,
                        "l1": 0.1,
                        "lpips": 0.2,
                    },
                    "causal": {
                        "psnr": 30.1,
                        "ssim": 0.901,
                        "l1": 0.099,
                        "lpips": 0.19,
                    },
                    "causal_repeat_current": {
                        "psnr": 30.0,
                        "ssim": 0.9,
                        "l1": 0.1,
                        "lpips": 0.2,
                    },
                    "runtime_gate_proxy": {
                        "blurry_laplacian_variance": 1.0,
                        "evssm_laplacian_variance": 1.0,
                        "causal_laplacian_variance": 1.02,
                        "causal_vs_evssm_gain": 0.020000000000000018,
                        "causal_vs_blurry_gain": 0.020000000000000018,
                        "passes_default_gate": True,
                    },
                    "temporal": temporal,
                }
            )
    assert len(room2_rows) == ROOM2_ONE_SHOT_FRAME_COUNT
    room2_evaluator = root / "room2_evaluator.json"
    room2_evaluator.write_text(
        json.dumps(
            {
                "schema": EVALUATOR_SCHEMA_V3,
                "checkpoint": str(h3_artifact),
                "evaluated_artifact_sha256": _sha256(h3_artifact),
                "source_checkpoint_sha256": checkpoint_sha256,
                "manifest": str(room2_manifest),
                "teacher_provenance": teacher_provenance,
                "input_domain": "evssm",
                "history": 3,
                "lpips_computed": True,
                "lpips_protocol": EXPECTED_LPIPS_PROTOCOL,
                "frame_count": len(room2_rows),
                "temporal_pair_count": sum(
                    row["temporal"] is not None for row in room2_rows
                ),
                "frames": room2_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    room2_report = root / "room2_one_shot_layer.json"
    room2_report.write_text(
        json.dumps(
            {
                "schema": DEPLOYMENT_LAYER_REPORT_SCHEMA_V1,
                "layer": "room2_one_shot",
                "role": "scene_disjoint_one_shot_test",
                "registered_contract_sha256": REGISTERED_CONTRACT_SHA256,
                "checkpoint_sha256": checkpoint_sha256,
                "evssm_checkpoint_sha256": evssm_checkpoint_sha256,
                "manifest_sha256": ROOM2_ONE_SHOT_MANIFEST_SHA256,
                "thresholds": DEPLOYMENT_THRESHOLDS["room2_one_shot"],
                "opened_after_temporal_validation_report_sha256": (
                    temporal_report_sha256
                ),
                "tuning_after_open": False,
                "oracle_good_definition": ORACLE_GOOD_DEFINITION,
                "eligible": True,
                "metrics": {
                    "psnr_delta_db": 0.1,
                    "ssim_delta": 0.001,
                    "relative_l1_delta": -0.01,
                    "lpips_delta": -0.01,
                    "gt_temporal_difference_delta": -0.01,
                    "steady_normal_vs_repeat_current_psnr_delta_db": 0.1,
                    "laplacian_gate_pass_ratio": 1.0,
                    "accepted_oracle_precision": 1.0,
                    "nondegraded_long_runs": 16,
                    "long_runs_total": 16,
                    "worst_run_psnr_delta_db": 0.1,
                },
                "evaluator_report": {
                    "path": str(room2_evaluator),
                    "sha256": _sha256(room2_evaluator),
                },
                "source_manifest": {
                    "path": str(room2_manifest),
                    "source_root": str(room2_source_root),
                    "sha256": ROOM2_ONE_SHOT_MANIFEST_SHA256,
                    "frame_count": ROOM2_ONE_SHOT_FRAME_COUNT,
                    "sequence_count": len(room2_sequence_lengths),
                    "sequence_lengths": dict(sorted(room2_sequence_lengths.items())),
                    "frame_identity_schema": (
                        "sorted_compact_json_sequence_index_blurry_sharp.v1"
                    ),
                    "frame_identity_sha256": (
                        ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256
                    ),
                },
                "history": 3,
                "input_domain": "evssm",
                "lpips_required": True,
                "lpips_computed": True,
                "missing_metrics": [],
                "details": {
                    "gate_and_oracle_all_frames": {
                        "oracle_definition": ORACLE_GOOD_DEFINITION,
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = root / "deployment_selection.json"
    report.write_text(
        json.dumps(
            {
                "schema": DEPLOYMENT_SELECTION_SCHEMA_V3,
                "policy": DEPLOYMENT_SELECTION_POLICY_V1,
                "thresholds": DEPLOYMENT_THRESHOLDS,
                "oracle_good_definition": ORACLE_GOOD_DEFINITION,
                "registered_contract": {
                    "schema": REGISTERED_CONTRACT_SCHEMA,
                    "sha256": REGISTERED_CONTRACT_SHA256,
                },
                "checkpoint_sha256": checkpoint_sha256,
                "evssm_checkpoint_sha256": evssm_checkpoint_sha256,
                "tum_used_for_selection": False,
                "layers": {
                    "temporal_validation": {
                        "report": temporal_report.name,
                        "report_sha256": temporal_report_sha256,
                        "manifest_sha256": TEMPORAL_VALIDATION_MANIFEST_SHA256,
                    },
                    "room2_one_shot": {
                        "report": room2_report.name,
                        "report_sha256": _sha256(room2_report),
                        "manifest_sha256": ROOM2_ONE_SHOT_MANIFEST_SHA256,
                    },
                },
                "eligible": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _write_precompute_report(root: Path, manifest: Path, label: str = "cache") -> Path:
    checkpoint = root / f"{label}_official_evssm.pth"
    checkpoint.write_bytes(b"official-unblur-slam-evssm-weights-" + label.encode())
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frames = []
    for sequence_index, record in enumerate(records):
        for frame_index, (blurry, sharp, teacher) in enumerate(
            zip(record["blurry"], record["sharp"], record["teacher"])
        ):
            blurry_path = (manifest.parent / blurry).resolve()
            sharp_path = (manifest.parent / sharp).resolve()
            teacher_path = (manifest.parent / teacher).resolve()
            frames.append(
                {
                    "sequence_index": sequence_index,
                    "frame_index": frame_index,
                    "blurry": str(blurry_path),
                    "blurry_sha256": _sha256(blurry_path),
                    "sharp": str(sharp_path),
                    "sharp_sha256": _sha256(sharp_path),
                    "teacher": str(teacher_path),
                    "teacher_sha256": _sha256(teacher_path),
                }
            )
    report = root / f"{label}_precompute_report.json"
    payload = {
        "schema": "unblur_slam.video_deblur_evssm_precompute.v1",
        "input_manifest": str(manifest.resolve()),
        "input_manifest_sha256": _sha256(manifest),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "output_manifest": str(manifest.resolve()),
        "output_manifest_sha256": _sha256(manifest),
        "sequence_count": len(records),
        "frame_count": len(frames),
        "frames": frames,
    }
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report


def test_jsonl_sequence_dataset() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        blurry_paths = []
        sharp_paths = []
        teacher_paths = []
        for index in range(3):
            blurry = root / f"blur_{index}.png"
            sharp = root / f"sharp_{index}.png"
            teacher = root / f"teacher_{index}.png"
            _save_image(blurry, 10 + index)
            _save_image(sharp, 20 + index)
            _save_image(teacher, 30 + index)
            blurry_paths.append(blurry.name)
            sharp_paths.append(sharp.name)
            teacher_paths.append(teacher.name)
        manifest = root / "sequences.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "sequence": "turtle",
                    "blurry": blurry_paths,
                    "sharp": sharp_paths,
                    "teacher": teacher_paths,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = VideoDeblurJsonlDataset(
            str(manifest), clip_length=4, crop_size=8, augment=False
        )
        assert [indices for _, indices in dataset.clips] == [
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 1, 2),
        ]
        assert [int(dataset[index]["target_index"]) for index in range(len(dataset))] == [
            0,
            1,
            2,
        ]
        sample = dataset[1]
        assert sample["sequence"] == "turtle"
        assert sample["blurry"].shape == (4, 3, 8, 8)
        assert sample["sharp"].shape == (4, 3, 8, 8)
        assert sample["teacher"].shape == (4, 3, 8, 8)
        assert bool(sample["has_teacher"].item())
        # A short sequence is left-padded by repeating its first frame.
        assert torch.equal(sample["blurry"][0], sample["blurry"][1])
        assert not torch.equal(sample["blurry"][-2], sample["blurry"][-1])


def test_precompute_report_binds_manifest_teacher_and_evssm_sha() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        blurry = root / "blur.png"
        sharp = root / "sharp.png"
        teacher = root / "teacher.png"
        _save_image(blurry, 10)
        _save_image(sharp, 20)
        _save_image(teacher, 30)
        manifest = root / "bound.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "sequence": "bound",
                    "blurry": [blurry.name],
                    "sharp": [sharp.name],
                    "teacher": [teacher.name],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = _write_precompute_report(root, manifest, "bound")
        provenance = load_evssm_precompute_report(str(report))
        assert provenance["storage"] == "precomputed_png_rgb8"
        assert provenance["teacher_domain"] == "evssm_restored_rgb_0_1"
        assert provenance["evssm_checkpoint_sha256"] == _sha256(
            root / "bound_official_evssm.pth"
        )
        dataset = VideoDeblurJsonlDataset(
            None, clip_length=2, precompute_report=str(report)
        )
        assert dataset.manifest == manifest.resolve()

        _save_image(teacher, 31)
        try:
            load_evssm_precompute_report(str(report))
        except ValueError as error:
            assert "teacher SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("mutated cached teacher was accepted")


def test_teacher_configuration_is_not_silently_unused() -> None:
    checkpoint = Path("teacher.pth")
    validate_teacher_options(checkpoint, False, 0.1)
    for teacher_checkpoint, teacher_input, distill_weight in (
        (checkpoint, True, 0.0),
        (None, True, 0.0),
        (checkpoint, False, 0.0),
        (None, False, -0.1),
    ):
        try:
            validate_teacher_options(
                teacher_checkpoint, teacher_input, distill_weight
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "non-deployable or ineffective teacher configuration must fail fast"
            )


def test_evssm_teacher_is_always_microbatched() -> None:
    class BatchOneTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, frames):
            self.batch_sizes.append(int(frames.shape[0]))
            assert frames.shape[0] == 1
            return frames

    teacher = BatchOneTeacher()
    frames = torch.rand(2, 3, 3, 9, 10)
    output = run_teacher(teacher, frames, chunk_size=1)
    assert torch.equal(output, frames)
    assert teacher.batch_sizes == [1] * 6
    try:
        run_teacher(teacher, frames, chunk_size=2)
    except ValueError:
        pass
    else:
        raise AssertionError("EVSSM teacher chunk >1 must fail before forward")


def test_teacher_input_train_and_export_are_rejected() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        training = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/train_causal_video_deblur.py"),
                "--train-manifest",
                str(root / "missing.jsonl"),
                "--output",
                str(root / "training"),
                "--teacher-input",
            ],
            check=False,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert training.returncode != 0
        assert "not deployable" in training.stderr
        # Contract validation occurs before output creation or dataset loading.
        assert not (root / "training").exists()

        model = make_model(use_teacher_input=True)
        checkpoint = root / "teacher_input.pth"
        torch.save(
            {
                "format": "unblur_slam.causal_video_deblur.v1",
                "model": model.state_dict(),
                "model_config": model.config_dict(),
            },
            checkpoint,
        )
        exported = root / "teacher_input.pt"
        exporting = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_causal_video_deblur.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(exported),
            ],
            check=False,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert exporting.returncode != 0
        assert "cannot be exported" in exporting.stderr
        assert not exported.exists()


def test_legacy_v1_checkpoint_config_still_exports_with_v1_semantics() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    legacy_config = {
        "channels": 4,
        "num_heads": 2,
        "num_blocks": 1,
        "max_history": 4,
        "use_teacher_input": False,
        "input_domain": "raw",
    }
    model = build_causal_video_deblur(legacy_config).eval()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(0.125)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "legacy_v1.pth"
        torch.save(
            {
                "format": "unblur_slam.causal_video_deblur.v1",
                "model": model.state_dict(),
                "model_config": legacy_config,
                "teacher_provenance": {
                    "schema": "unblur_slam.video_deblur_teacher_provenance.v1",
                    "storage": "none",
                    "teacher_domain": "none",
                    "evssm_checkpoint_sha256": None,
                },
                "training_contract": {
                    "supervised_output": "newest_frame_at_every_sequence_position",
                    "stream_prefix_padding": "repeat_first_frame_on_left",
                },
            },
            checkpoint,
        )
        exported = root / "legacy_v1.pt"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_causal_video_deblur.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(exported),
                "--verify-height",
                "12",
                "--verify-width",
                "16",
            ],
            check=True,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        extra_files = {"metadata.json": ""}
        loaded = torch.jit.load(
            str(exported), map_location="cpu", _extra_files=extra_files
        )
        frames = torch.rand(1, 4, 3, 12, 16)
        assert torch.allclose(loaded(frames), frames[:, -1] + 0.125, atol=1.0e-6)
        metadata_value = extra_files["metadata.json"]
        if isinstance(metadata_value, bytes):
            metadata_value = metadata_value.decode("utf-8")
        metadata = json.loads(metadata_value)
        assert metadata["format"] == (
            "unblur_slam.causal_video_deblur.torchscript.v1"
        )
        assert "max_residual" not in metadata["model_config"]


def test_v3_export_fails_closed_without_objective_contract() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    model = make_model(input_domain="raw")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "malformed_v3.pth"
        torch.save(
            {
                "format": "unblur_slam.causal_video_deblur.v3",
                "model": model.state_dict(),
                "model_config": model.config_dict(),
                "teacher_provenance": {
                    "schema": "unblur_slam.video_deblur_teacher_provenance.v1",
                    "storage": "none",
                    "teacher_domain": "none",
                    "evssm_checkpoint_sha256": None,
                },
                "refinement_contract": {
                    "schema": "unblur_slam.causal_video_deblur.refinement.v3",
                    "base": "raw_input",
                    "formula": (
                        "output = input + max_residual * tanh(residual_logits)"
                    ),
                    "max_residual": model.max_residual,
                    "identity_safe_initialization": (
                        "zero_weight_and_bias_output_head"
                    ),
                },
                "training_contract": {
                    "schema": "unblur_slam.causal_video_deblur.training.v3",
                    "temporal_output": "rolling_two_window_forward",
                    "training_clip_length": 5,
                    "rolling_window_length": 4,
                    "stream_prefix_padding": "repeat_first_frame_on_left",
                    "causality": (
                        "strict_upper_triangular_temporal_attention_mask"
                    ),
                    "fft_normalization": "ortho",
                },
                "checkpoint_selection": {"metric": "val_psnr", "mode": "max"},
            },
            checkpoint,
        )
        exported = root / "must_not_exist.pt"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_causal_video_deblur.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(exported),
            ],
            check=False,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "objective_contract" in result.stderr
        assert not exported.exists()


def test_deployment_selector_binds_checkpoint_and_all_gate_metrics() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "candidate.pth"
        checkpoint.write_bytes(b"candidate-v3")
        teacher_sha = REGISTERED_EVSSM_SHA256
        report = _write_deployment_selection(root, checkpoint, teacher_sha)
        accepted = validate_deployment_selection(
            report,
            checkpoint,
            {"evssm_checkpoint_sha256": teacher_sha},
        )
        assert accepted["eligible"] is True
        assert accepted["layers"]["temporal_validation"][
            "manifest_sha256"
        ] == TEMPORAL_VALIDATION_MANIFEST_SHA256
        assert accepted["layers"]["room2_one_shot"][
            "manifest_sha256"
        ] == ROOM2_ONE_SHOT_MANIFEST_SHA256
        assert accepted["layers"]["temporal_validation"]["metrics"][
            "laplacian_gate_pass_ratio"
        ] == 0.25
        assert accepted["layers"]["room2_one_shot"]["metrics"][
            "laplacian_gate_pass_ratio"
        ] == 1.0
        assert accepted["layers"]["temporal_validation"]["metrics"][
            "accepted_oracle_precision"
        ] == 0.8
        assert accepted["layers"]["temporal_validation"]["metrics"][
            "normal_vs_repeat_current_temporal_relative_delta"
        ] == -0.02
        assert abs(
            accepted["layers"]["temporal_validation"]["metrics"][
                "normal_vs_history1_psnr_delta_db"
            ]
            - 0.1
        ) < 1.0e-9
        assert accepted["layers"]["temporal_validation"]["history1_control"][
            "history"
        ] == 1
        assert accepted["layers"]["temporal_validation"]["history1_control"][
            "manifest_sha256"
        ] == TEMPORAL_VALIDATION_MANIFEST_SHA256
        assert accepted["layers"]["temporal_validation"]["metrics"][
            "worst_run_psnr_delta_db"
        ] == -0.1
        assert accepted["layers"]["room2_one_shot"]["metrics"][
            "nondegraded_long_runs"
        ] == 16.0
        assert accepted["layers"]["room2_one_shot"]["metrics"][
            "worst_run_psnr_delta_db"
        ] == 0.1

        selection = json.loads(report.read_text(encoding="utf-8"))
        temporal_path = root / selection["layers"]["temporal_validation"]["report"]
        temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
        temporal["metrics"]["laplacian_gate_pass_ratio"] = 0.24
        temporal_path.write_text(json.dumps(temporal) + "\n", encoding="utf-8")
        selection["layers"]["temporal_validation"]["report_sha256"] = _sha256(
            temporal_path
        )
        report.write_text(json.dumps(selection) + "\n", encoding="utf-8")
        try:
            validate_deployment_selection(
                report,
                checkpoint,
                {"evssm_checkpoint_sha256": teacher_sha},
            )
        except ValueError as error:
            assert "Layer1" in str(error)
        else:
            raise AssertionError("temporal Layer1 Laplacian 24% was accepted")

        report = _write_deployment_selection(root, checkpoint, teacher_sha)
        selection = json.loads(report.read_text(encoding="utf-8"))
        room2_path = root / selection["layers"]["room2_one_shot"]["report"]
        room2 = json.loads(room2_path.read_text(encoding="utf-8"))
        room2["metrics"]["laplacian_gate_pass_ratio"] = 0.19
        room2_path.write_text(json.dumps(room2) + "\n", encoding="utf-8")
        selection["layers"]["room2_one_shot"]["report_sha256"] = _sha256(
            room2_path
        )
        report.write_text(json.dumps(selection) + "\n", encoding="utf-8")
        try:
            validate_deployment_selection(
                report,
                checkpoint,
                {"evssm_checkpoint_sha256": teacher_sha},
            )
        except ValueError as error:
            assert "Layer2" in str(error)
        else:
            raise AssertionError("room2 Layer2 Laplacian 19% was accepted")


def test_layered_selector_rejects_unbound_or_incomplete_reports() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "candidate.pth"
        checkpoint.write_bytes(b"candidate-layer-binding")
        provenance = {"evssm_checkpoint_sha256": REGISTERED_EVSSM_SHA256}

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection = json.loads(report.read_text(encoding="utf-8"))
        temporal_path = root / selection["layers"]["temporal_validation"]["report"]
        temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
        del temporal["metrics"]["accepted_oracle_precision"]
        temporal_path.write_text(json.dumps(temporal) + "\n", encoding="utf-8")
        selection["layers"]["temporal_validation"]["report_sha256"] = _sha256(
            temporal_path
        )
        report.write_text(json.dumps(selection) + "\n", encoding="utf-8")
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "accepted_oracle_precision" in str(error)
        else:
            raise AssertionError("Layer1 report without oracle precision was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection = json.loads(report.read_text(encoding="utf-8"))
        room2_path = root / selection["layers"]["room2_one_shot"]["report"]
        room2_path.write_text(
            room2_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "report SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("mutated room2 report was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection = json.loads(report.read_text(encoding="utf-8"))
        selection["layers"]["room2_one_shot"]["manifest_sha256"] = "0" * 64
        report.write_text(json.dumps(selection) + "\n", encoding="utf-8")
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "manifest SHA-256 is not preregistered" in str(error)
        else:
            raise AssertionError("unregistered room2 manifest was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection = json.loads(report.read_text(encoding="utf-8"))
        temporal_path = root / selection["layers"]["temporal_validation"]["report"]
        temporal_path.write_text(
            temporal_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "report SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("mutated temporal-validation report was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection = json.loads(report.read_text(encoding="utf-8"))
        temporal_path = root / selection["layers"]["temporal_validation"]["report"]
        temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
        temporal["manifest_sha256"] = "0" * 64
        temporal_path.write_text(json.dumps(temporal) + "\n", encoding="utf-8")
        selection["layers"]["temporal_validation"]["report_sha256"] = _sha256(
            temporal_path
        )
        report.write_text(json.dumps(selection) + "\n", encoding="utf-8")
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "uses a different manifest" in str(error)
        else:
            raise AssertionError("Layer1 report bound to another manifest was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        checkpoint.write_bytes(b"candidate-layer-binding-mutated")
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "different checkpoint" in str(error)
        else:
            raise AssertionError("selection for another checkpoint was accepted")

        checkpoint.write_bytes(b"candidate-layer-binding")
        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        try:
            validate_deployment_selection(
                report,
                checkpoint,
                {"evssm_checkpoint_sha256": "0" * 64},
            )
        except ValueError as error:
            assert "registered Unblur-SLAM EVSSM" in str(error)
        else:
            raise AssertionError("selection for another EVSSM baseline was accepted")


def test_history1_control_is_content_bound_and_recomputed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "candidate.pth"
        checkpoint.write_bytes(b"candidate-with-history1-control")
        provenance = {"evssm_checkpoint_sha256": REGISTERED_EVSSM_SHA256}

        def temporal_evidence(report: Path) -> tuple[dict, Path, dict, Path]:
            selection = json.loads(report.read_text(encoding="utf-8"))
            temporal_path = (
                root / selection["layers"]["temporal_validation"]["report"]
            )
            temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
            h1_path = Path(temporal["history1_control"]["evaluator_report"])
            return selection, temporal_path, temporal, h1_path

        def rewrite_temporal(
            report: Path, selection: dict, temporal_path: Path, temporal: dict
        ) -> None:
            temporal_path.write_text(
                json.dumps(temporal) + "\n", encoding="utf-8"
            )
            selection["layers"]["temporal_validation"]["report_sha256"] = (
                _sha256(temporal_path)
            )
            report.write_text(json.dumps(selection) + "\n", encoding="utf-8")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        _, _, _, h1_path = temporal_evidence(report)
        h1_path.write_text(
            h1_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "H1 evaluator report SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("tampered H1 evaluator evidence was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, temporal_path, temporal, _ = temporal_evidence(report)
        temporal["history1_control"]["checkpoint_sha256"] = "0" * 64
        rewrite_temporal(report, selection, temporal_path, temporal)
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "different source checkpoint" in str(error)
        else:
            raise AssertionError("H1 evidence for another checkpoint was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, temporal_path, temporal, h1_path = temporal_evidence(report)
        h1 = json.loads(h1_path.read_text(encoding="utf-8"))
        h1["manifest"] = str(root / "different_manifest.jsonl")
        h1_path.write_text(json.dumps(h1) + "\n", encoding="utf-8")
        temporal["history1_control"]["evaluator_report_sha256"] = _sha256(
            h1_path
        )
        rewrite_temporal(report, selection, temporal_path, temporal)
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "different manifests" in str(error)
        else:
            raise AssertionError("H1 evidence for another manifest was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, temporal_path, temporal, _ = temporal_evidence(report)
        temporal["history1_control"]["evssm_checkpoint_sha256"] = "0" * 64
        rewrite_temporal(report, selection, temporal_path, temporal)
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "registered Unblur-SLAM EVSSM" in str(error)
        else:
            raise AssertionError("H1 evidence for another EVSSM was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, temporal_path, temporal, _ = temporal_evidence(report)
        temporal["history1_control"]["history"] = 3
        rewrite_temporal(report, selection, temporal_path, temporal)
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "history must be the integer 1" in str(error)
        else:
            raise AssertionError("non-H1 spatial control was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, temporal_path, temporal, h1_path = temporal_evidence(report)
        h1 = json.loads(h1_path.read_text(encoding="utf-8"))
        h1["frames"][-1]["causal"]["psnr"] = 30.06
        h1_path.write_text(json.dumps(h1) + "\n", encoding="utf-8")
        temporal["history1_control"]["evaluator_report_sha256"] = _sha256(
            h1_path
        )
        rewrite_temporal(report, selection, temporal_path, temporal)
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "does not match evaluator rows" in str(error)
        else:
            raise AssertionError("forged H3-minus-H1 PSNR delta was accepted")


def test_room2_evaluator_evidence_is_content_bound_and_recomputed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "candidate.pth"
        checkpoint.write_bytes(b"candidate-with-room2-evidence")
        provenance = {"evssm_checkpoint_sha256": REGISTERED_EVSSM_SHA256}

        def room2_evidence(report: Path) -> tuple[dict, Path, dict, Path, dict]:
            selection = json.loads(report.read_text(encoding="utf-8"))
            layer_path = root / selection["layers"]["room2_one_shot"]["report"]
            layer = json.loads(layer_path.read_text(encoding="utf-8"))
            evaluator_path = Path(layer["evaluator_report"]["path"])
            evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
            return selection, layer_path, layer, evaluator_path, evaluator

        def rewrite_evidence(
            report: Path,
            selection: dict,
            layer_path: Path,
            layer: dict,
            evaluator_path: Path,
            evaluator: dict,
        ) -> None:
            evaluator_path.write_text(
                json.dumps(evaluator) + "\n", encoding="utf-8"
            )
            layer["evaluator_report"]["sha256"] = _sha256(evaluator_path)
            layer_path.write_text(json.dumps(layer) + "\n", encoding="utf-8")
            selection["layers"]["room2_one_shot"]["report_sha256"] = _sha256(
                layer_path
            )
            report.write_text(json.dumps(selection) + "\n", encoding="utf-8")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        _, _, _, evaluator_path, _ = room2_evidence(report)
        evaluator_path.write_text(
            evaluator_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "room2 evaluator report SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("tampered room2 evaluator report was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, layer_path, layer, evaluator_path, evaluator = room2_evidence(
            report
        )
        evaluator["frames"][0]["causal"]["lpips"] = 0.25
        rewrite_evidence(
            report, selection, layer_path, layer, evaluator_path, evaluator
        )
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "room2 metric lpips_delta" in str(error)
        else:
            raise AssertionError("forged room2 LPIPS summary was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, layer_path, layer, evaluator_path, evaluator = room2_evidence(
            report
        )
        evaluator["frames"][0]["blurry_path"] = "/forged/room2/frame.png"
        rewrite_evidence(
            report, selection, layer_path, layer, evaluator_path, evaluator
        )
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "keys/paths are not canonical" in str(error)
        else:
            raise AssertionError("noncanonical room2 frame identity was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, layer_path, layer, evaluator_path, evaluator = room2_evidence(
            report
        )
        gate = evaluator["frames"][0]["runtime_gate_proxy"]
        gate["causal_laplacian_variance"] = 0.5
        gate["causal_vs_evssm_gain"] = -0.5
        gate["causal_vs_blurry_gain"] = -0.5
        gate["passes_default_gate"] = False
        rewrite_evidence(
            report, selection, layer_path, layer, evaluator_path, evaluator
        )
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "laplacian_gate_pass_ratio" in str(error)
        else:
            raise AssertionError("forged room2 gate pass ratio was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, layer_path, layer, evaluator_path, evaluator = room2_evidence(
            report
        )
        evaluator["lpips_computed"] = False
        rewrite_evidence(
            report, selection, layer_path, layer, evaluator_path, evaluator
        )
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "did not compute LPIPS" in str(error)
        else:
            raise AssertionError("room2 evidence without LPIPS was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, layer_path, layer, evaluator_path, evaluator = room2_evidence(
            report
        )
        evaluator["frames"][0]["causal"]["l1"] = 0.1006
        layer["metrics"]["relative_l1_delta"] = (
            ((173 * 0.099 + 0.1006) / 174) / 0.1 - 1.0
        )
        rewrite_evidence(
            report, selection, layer_path, layer, evaluator_path, evaluator
        )
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "accepted_oracle_precision" in str(error)
        else:
            raise AssertionError("forged room2 oracle precision was accepted")

        report = _write_deployment_selection(
            root, checkpoint, REGISTERED_EVSSM_SHA256
        )
        selection, layer_path, layer, evaluator_path, evaluator = room2_evidence(
            report
        )
        lengths = {}
        for row in evaluator["frames"]:
            lengths[row["sequence"]] = lengths.get(row["sequence"], 0) + 1
        long_sequence = next(
            sequence for sequence, length in lengths.items() if length == 5
        )
        changed = 0
        for row in evaluator["frames"]:
            if (
                row["sequence"] == long_sequence
                and row["history_stage"] == "steady_state"
            ):
                row["causal"]["psnr"] = 29.8
                changed += 1
        assert changed == 3
        steady_count = sum(
            row["history_stage"] == "steady_state"
            for row in evaluator["frames"]
        )
        layer["metrics"]["psnr_delta_db"] = 0.1 - 0.3 * changed / 174
        layer["metrics"][
            "steady_normal_vs_repeat_current_psnr_delta_db"
        ] = 0.1 - 0.3 * changed / steady_count
        layer["metrics"]["accepted_oracle_precision"] = (174 - changed) / 174
        layer["metrics"]["worst_run_psnr_delta_db"] = -0.08
        rewrite_evidence(
            report, selection, layer_path, layer, evaluator_path, evaluator
        )
        try:
            validate_deployment_selection(report, checkpoint, provenance)
        except ValueError as error:
            assert "nondegraded_long_runs" in str(error)
        else:
            raise AssertionError("forged room2 per-run count was accepted")


def test_train_and_export_cli_smoke() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        blurry_paths = []
        sharp_paths = []
        teacher_paths = []
        for index in range(4):
            blurry = root / f"cli_blur_{index}.png"
            sharp = root / f"cli_sharp_{index}.png"
            teacher = root / f"cli_teacher_{index}.png"
            _save_image(blurry, 20 + index)
            _save_image(sharp, 24 + index)
            _save_image(teacher, 23 + index)
            blurry_paths.append(blurry.name)
            sharp_paths.append(sharp.name)
            teacher_paths.append(teacher.name)
        manifest = root / "cli.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "sequence": "smoke",
                    "blurry": blurry_paths,
                    "sharp": sharp_paths,
                    "teacher": teacher_paths,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = _write_precompute_report(root, manifest, "raw_distill")
        output = root / "training"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/train_causal_video_deblur.py"),
                "--train-manifest",
                str(manifest),
                "--precompute-report",
                str(report),
                "--val-precompute-report",
                str(report),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--history",
                "4",
                "--channels",
                "4",
                "--heads",
                "2",
                "--blocks",
                "1",
                "--crop-size",
                "8",
                "--batch-size",
                "1",
                "--grad-accumulation",
                "2",
                "--warmup-steps",
                "1",
                "--workers",
                "0",
                "--epochs",
                "1",
                "--distill-weight",
                "0.1",
                "--dry-run",
            ],
            check=True,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        checkpoint = output / "latest.pth"
        exported = root / "causal_video_deblur.pt"
        assert checkpoint.is_file()
        payload = torch.load(checkpoint, map_location="cpu")
        assert payload["format"] == "unblur_slam.causal_video_deblur.v3"
        assert payload["model_config"]["max_residual"] == 8.0 / 255.0
        assert payload["objective_contract"]["schema"] == (
            "unblur_slam.causal_video_deblur.objective.v3"
        )
        assert payload["objective_contract"]["primary_reconstruction"][
            "fft_normalization"
        ] == "ortho"
        assert payload["objective_contract"]["evssm_fidelity"]["weight"] > 0.0
        assert payload["objective_contract"]["temporal_delta"]["weight"] > 0.0
        assert payload["objective_contract"]["edge"]["weight"] > 0.0
        assert payload["objective_contract"]["laplacian_gate"]["weight"] > 0.0
        assert payload["objective_contract"]["edge"]["weight"] != payload[
            "objective_contract"
        ]["laplacian_gate"]["weight"]
        assert payload["refinement_contract"]["max_residual"] == 8.0 / 255.0
        assert payload["training_contract"]["temporal_output"] == (
            "rolling_two_window_forward"
        )
        assert payload["training_contract"]["training_clip_length"] == 5
        assert payload["training_contract"]["rolling_window_length"] == 4
        assert payload["training_contract"]["fft_normalization"] == "ortho"
        assert payload["checkpoint_selection"]["metric"] == "val_psnr"
        assert payload["optimization_contract"][
            "gradient_accumulation_micro_batches"
        ] == 2
        assert payload["optimization_contract"]["warmup_steps"] == 1
        assert payload["optimization_contract"]["schedule_unit"] == (
            "optimizer_step"
        )
        assert payload["step"] == 1
        assert np.isfinite(payload["validation_metrics"]["psnr"])
        assert np.isfinite(payload["validation_metrics"]["ssim"])
        assert payload["best_ssim_at_best_psnr"] == payload[
            "validation_metrics"
        ]["ssim"]
        assert payload["checkpoint_selection"]["deployment_status"] == (
            "not_deployment_selected"
        )
        without_selection = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_causal_video_deblur.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(exported),
            ],
            check=False,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert without_selection.returncode != 0
        assert "--diagnostic-output" in without_selection.stderr
        assert not exported.exists()
        diagnostic = root / "causal_video_deblur_diagnostic.pt"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_causal_video_deblur.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(diagnostic),
                "--diagnostic-output",
                "--verify-height",
                "12",
                "--verify-width",
                "16",
            ],
            check=True,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        diagnostic_extra = {"metadata.json": ""}
        torch.jit.load(
            str(diagnostic), map_location="cpu", _extra_files=diagnostic_extra
        )
        diagnostic_metadata_value = diagnostic_extra["metadata.json"]
        if isinstance(diagnostic_metadata_value, bytes):
            diagnostic_metadata_value = diagnostic_metadata_value.decode("utf-8")
        diagnostic_metadata = json.loads(diagnostic_metadata_value)
        assert diagnostic_metadata["artifact_role"] == (
            "diagnostic_evaluation_only"
        )
        assert diagnostic_metadata["deployment_eligible"] is False
        assert diagnostic_metadata["deployment_selection"] is None
        assert diagnostic_metadata["source_checkpoint_sha256"] == _sha256(checkpoint)
        # Formal Layer1/Layer2 deployment is locked to the preregistered
        # official EVSSM SHA.  This synthetic fixture rewrites only provenance
        # in a copy; the diagnostic artifact above retains the cache's real SHA.
        deployment_checkpoint = root / "deployment_candidate.pth"
        deployment_payload = dict(payload)
        deployment_payload["teacher_provenance"] = dict(
            payload["teacher_provenance"]
        )
        deployment_payload["teacher_provenance"][
            "evssm_checkpoint_sha256"
        ] = REGISTERED_EVSSM_SHA256
        torch.save(deployment_payload, deployment_checkpoint)
        selection_report = _write_deployment_selection(
            root,
            deployment_checkpoint,
            REGISTERED_EVSSM_SHA256,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_causal_video_deblur.py"),
                "--checkpoint",
                str(deployment_checkpoint),
                "--output",
                str(exported),
                "--selection-report",
                str(selection_report),
                "--verify-height",
                "12",
                "--verify-width",
                "16",
            ],
            check=True,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        loaded = torch.jit.load(str(exported), map_location="cpu")
        assert loaded(torch.rand(1, 4, 3, 12, 16)).shape == (1, 3, 12, 16)
        sequence = loaded.forward_sequence(torch.rand(1, 4, 3, 12, 16))
        assert sequence.shape == (1, 4, 3, 12, 16)
        extra_files = {"metadata.json": ""}
        torch.jit.load(str(exported), map_location="cpu", _extra_files=extra_files)
        metadata_value = extra_files["metadata.json"]
        if isinstance(metadata_value, bytes):
            metadata_value = metadata_value.decode("utf-8")
        metadata = json.loads(metadata_value)
        assert metadata["format"] == (
            "unblur_slam.causal_video_deblur.torchscript.v3"
        )
        assert metadata["checkpoint_format"] == (
            "unblur_slam.causal_video_deblur.v3"
        )
        assert metadata["artifact_role"] == "deployment_selected"
        assert metadata["deployment_eligible"] is True
        assert metadata["source_checkpoint_sha256"] == _sha256(
            deployment_checkpoint
        )
        assert metadata["model_config"]["max_residual"] == 8.0 / 255.0
        assert metadata["teacher_provenance"]["evssm_checkpoint_sha256"] == (
            REGISTERED_EVSSM_SHA256
        )
        assert metadata["objective_contract"] == payload["objective_contract"]
        assert metadata["optimization_contract"] == payload[
            "optimization_contract"
        ]
        assert metadata["refinement_contract"] == payload["refinement_contract"]
        assert metadata["checkpoint_selection"] == payload["checkpoint_selection"]
        assert metadata["deployment_selection"]["eligible"] is True
        assert metadata["deployment_selection"][
            "selection_report_sha256"
        ] == _sha256(selection_report)
        assert metadata["deployment_selection"]["tum_used_for_selection"] is False
        assert metadata["deployment_selection"]["layers"][
            "temporal_validation"
        ]["manifest_sha256"] == TEMPORAL_VALIDATION_MANIFEST_SHA256
        assert metadata["deployment_selection"]["layers"]["room2_one_shot"][
            "manifest_sha256"
        ] == ROOM2_ONE_SHOT_MANIFEST_SHA256
        assert metadata["export_verification"]["max_observed_abs_residual"] <= (
            8.0 / 255.0 + 1.0e-6
        )
        assert "forward_sequence" in metadata["exported_methods"]
        backend = CausalTorchScriptBackend(
            checkpoint=str(exported), history=4, device="cpu"
        )
        for timestamp in range(4):
            streamed = backend(
                torch.rand(1, 3, 12, 16), timestamp=timestamp
            )
            assert streamed.shape == (1, 3, 12, 16)
        assert len(backend.frames) == 4

        evaluation = root / "evaluation"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluate_causal_video_deblur.py"),
                "--manifest",
                str(manifest),
                "--precompute-report",
                str(report),
                "--checkpoint",
                str(diagnostic),
                "--output-dir",
                str(evaluation),
                "--history",
                "4",
                "--device",
                "cpu",
                "--max-visuals",
                "2",
            ],
            check=True,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        metrics = json.loads((evaluation / "metrics.json").read_text())
        assert metrics["temporal_pair_count"] == 3
        assert metrics["temporal"]["protocol"]["optical_flow_warp_used"] is False
        assert "not GT/flow warp" in metrics["temporal"]["protocol"][
            "gt_difference_error_l1_not_warp"
        ]
        assert "gt_difference_error_l1_not_warp" in metrics["temporal"]["mean"][
            "causal"
        ]


def test_evssm_domain_training_and_composite_streaming() -> None:
    class IdentityEVSSM(torch.nn.Module):
        def forward(self, image):
            return image

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        blurry_paths = []
        sharp_paths = []
        teacher_paths = []
        for index in range(4):
            blurry = root / f"evssm_blur_{index}.png"
            sharp = root / f"evssm_sharp_{index}.png"
            teacher = root / f"evssm_teacher_{index}.png"
            _save_image(blurry, 30 + index)
            _save_image(sharp, 40 + index)
            _save_image(teacher, 35 + index)
            blurry_paths.append(blurry.name)
            sharp_paths.append(sharp.name)
            teacher_paths.append(teacher.name)
        manifest = root / "evssm_domain.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "sequence": "evssm-domain-smoke",
                    "blurry": blurry_paths,
                    "sharp": sharp_paths,
                    "teacher": teacher_paths,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = _write_precompute_report(root, manifest, "evssm_domain")
        output = root / "training"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/train_causal_video_deblur.py"),
                "--precompute-report",
                str(report),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--input-domain",
                "evssm",
                "--history",
                "4",
                "--channels",
                "4",
                "--heads",
                "2",
                "--blocks",
                "1",
                "--crop-size",
                "8",
                "--batch-size",
                "1",
                "--workers",
                "0",
                "--epochs",
                "1",
                "--dry-run",
            ],
            check=True,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        checkpoint = output / "latest.pth"
        payload = torch.load(checkpoint, map_location="cpu")
        assert payload["format"] == "unblur_slam.causal_video_deblur.v3"
        assert payload["model_config"]["input_domain"] == "evssm"
        assert payload["model_config"]["max_residual"] == 8.0 / 255.0
        assert payload["teacher_provenance"]["storage"] == "precomputed_png_rgb8"
        assert payload["teacher_provenance"]["evssm_checkpoint_sha256"] == _sha256(
            root / "evssm_domain_official_evssm.pth"
        )
        exported = root / "causal_evssm.pt"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_causal_video_deblur.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(exported),
                "--diagnostic-output",
                "--verify-height",
                "12",
                "--verify-width",
                "16",
            ],
            check=True,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        extra_files = {"metadata.json": ""}
        torch.jit.load(str(exported), map_location="cpu", _extra_files=extra_files)
        metadata_value = extra_files["metadata.json"]
        if isinstance(metadata_value, bytes):
            metadata_value = metadata_value.decode("utf-8")
        metadata = json.loads(metadata_value)
        assert metadata["format"] == (
            "unblur_slam.causal_video_deblur.torchscript.v3"
        )
        assert metadata["refinement_contract"]["base"] == "frozen_evssm_input"
        backend = CausalEVSSMBackend(
            evssm_model=IdentityEVSSM(),
            checkpoint=str(exported),
            history=4,
            device="cpu",
        )
        for timestamp in range(4):
            frame = torch.rand(1, 3, 12, 16)
            output_frame = backend(frame, timestamp=timestamp)
            assert output_frame.shape == frame.shape
            assert torch.equal(backend.last_evssm_output, frame)
            assert torch.equal(
                backend.last_temporal_input,
                torch.round(frame * 255.0) / 255.0,
            )
        assert len(backend.frames) == 4


def test_causal_evssm_gate_falls_back_to_single_frame_selection() -> None:
    from thirdparty.glorie_slam.motion_filter import MotionFilter

    motion_filter = MotionFilter.__new__(MotionFilter)
    motion_filter.cfg = {
        "deblur": {
            "stream_every_frame": True,
            "stream_apply_to_tracking": True,
            "stream_min_laplacian_gain": 0.0,
            "stream_min_vs_evssm_gain": 0.0,
            "stream_replace_sharp": True,
        },
        "verbose": False,
    }
    raw = torch.full((1, 3, 8, 8), 0.10)
    raw_before = raw.clone()
    candidate = torch.full_like(raw, 0.20)
    sharper_evssm = torch.full_like(raw, 0.30)
    evssm_before = sharper_evssm.clone()
    motion_filter.deblur_backend_name = "causal_evssm"
    motion_filter.deblur_backend = SimpleNamespace(
        last_evssm_output=sharper_evssm
    )
    history_updates = []

    def apply_and_advance(image, stream=None, timestamp=None):
        del image, stream
        history_updates.append(timestamp)
        return candidate

    motion_filter.apply_evssm_deblur = apply_and_advance
    motion_filter._laplacian_value = lambda image: float(image.mean().item())

    selected, effective_candidate, replaced, effective_gain = (
        motion_filter._streaming_deblur(raw, 0, None)
    )
    assert history_updates == [0]
    assert replaced is False
    assert torch.equal(selected, raw_before)
    assert torch.equal(effective_candidate, sharper_evssm)
    assert torch.equal(raw, raw_before)
    assert torch.equal(sharper_evssm, evssm_before)
    assert effective_candidate.data_ptr() != sharper_evssm.data_ptr()
    assert float(effective_candidate.min()) >= 0.0
    assert float(effective_candidate.max()) <= 1.0
    assert effective_gain > 0.0
    assert motion_filter.last_streaming_evssm_fallback is True
    assert motion_filter.last_streaming_selection == "raw_evssm_candidate"
    assert motion_filter.last_streaming_candidate_safe is False
    assert motion_filter.last_streaming_vs_evssm_gain < 0.0
    # The temporal candidate was rejected.  Its independent EVSSM replacement
    # is only eligible if the later legacy keyframe/blur path is reached.
    assert (
        motion_filter._streaming_candidate_rejected(
            effective_candidate, effective_gain
        )
        is False
    )

    motion_filter.deblur_backend.last_evssm_output = candidate
    selected, _, replaced, _ = motion_filter._streaming_deblur(raw, 1, None)
    assert history_updates == [0, 1]
    assert replaced is True
    assert torch.equal(selected, candidate)
    assert motion_filter.last_streaming_evssm_fallback is False
    assert motion_filter.last_streaming_candidate_safe is True


def test_droid_normalization_never_aliases_rgb_observations() -> None:
    from thirdparty.glorie_slam.motion_filter import MotionFilter

    motion_filter = MotionFilter.__new__(MotionFilter)
    motion_filter.device = "cpu"
    motion_filter.MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    motion_filter.STDV = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    observation = torch.linspace(0.0, 1.0, 3 * 6 * 8).reshape(1, 3, 6, 8)
    before = observation.clone()
    normalized = motion_filter._normalize_droid_input(observation)
    expected = (before[None] - motion_filter.MEAN) / motion_filter.STDV

    assert torch.equal(observation, before)
    assert normalized.data_ptr() != observation.data_ptr()
    assert torch.allclose(normalized, expected, atol=0.0, rtol=0.0)
    assert float(observation.min()) == 0.0
    assert float(observation.max()) == 1.0


def test_first_track_frame_advances_frontend_but_stays_legacy_raw() -> None:
    from thirdparty.glorie_slam.motion_filter import MotionFilter

    motion_filter = MotionFilter.__new__(MotionFilter)
    motion_filter.device = "cpu"
    motion_filter.MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    motion_filter.STDV = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    motion_filter.cfg = {"exam_blur_score": False}
    motion_filter.tracking_anchor_indices = None
    motion_filter.last_streaming_evssm_fallback = False
    motion_filter.last_streaming_selection = "raw"

    appended = []
    feature_inputs = []
    video = SimpleNamespace(
        counter=SimpleNamespace(value=0),
        down_scale=1,
        append=lambda *args: appended.append(args),
    )
    motion_filter.video = video
    raw = torch.linspace(0.05, 0.95, 3 * 4 * 6).reshape(1, 3, 4, 6)
    raw_before = raw.clone()
    causal_candidate = (1.0 - raw).clone()
    candidate_before = causal_candidate.clone()
    frontend_inputs = []

    def streaming_deblur(image, timestamp, stream):
        del stream
        frontend_inputs.append((timestamp, image.clone()))
        motion_filter.last_streaming_evssm_fallback = False
        motion_filter.last_streaming_selection = "causal_evssm"
        return causal_candidate, causal_candidate, True, 1.0

    def feature_encoder(inputs):
        feature_inputs.append(inputs.clone())
        return torch.zeros(1, 1, 1, 1)

    motion_filter._streaming_deblur = streaming_deblur
    motion_filter._MotionFilter__feature_encoder = feature_encoder
    motion_filter._MotionFilter__context_encoder = lambda inputs: (
        torch.zeros(1, 1, 1, 1),
        torch.zeros(1, 1, 1, 1),
    )
    motion_filter._mono_depth_for_frame = (
        lambda timestamp, image, stream: torch.ones(4, 6)
    )

    result = motion_filter.track(
        0,
        raw,
        intrinsics=torch.tensor([1.0, 1.0, 0.5, 0.5]),
        fake_sharp=True,
        stream=None,
    )

    assert result is None
    assert len(frontend_inputs) == 1
    assert torch.equal(frontend_inputs[0][1], raw_before)
    assert len(feature_inputs) == 1
    expected = (raw_before[None] - motion_filter.MEAN) / motion_filter.STDV
    assert torch.allclose(feature_inputs[0], expected, atol=0.0, rtol=0.0)
    assert len(appended) == 1
    assert torch.equal(appended[0][1], raw_before[0])
    assert torch.equal(raw, raw_before)
    assert torch.equal(causal_candidate, candidate_before)
    assert float(causal_candidate.min()) >= 0.0
    assert float(causal_candidate.max()) <= 1.0
    assert motion_filter.last_streaming_selection == "raw_first_frame"
    assert motion_filter.last_track_info["streaming_replaced"] is False
    assert motion_filter.last_track_info["appended"] is True


def test_rejected_streaming_candidate_appends_raw_with_new_video_index() -> None:
    from unittest.mock import patch

    import thirdparty.glorie_slam.motion_filter as motion_filter_module

    MotionFilter = motion_filter_module.MotionFilter
    motion_filter = MotionFilter.__new__(MotionFilter)
    motion_filter.device = "cpu"
    motion_filter.MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    motion_filter.STDV = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    motion_filter.cfg = {
        "deblur": {"stream_min_laplacian_gain": 0.0},
        "mono_prior": {"predict_online": False},
        "exam_blur_score": False,
        "only_tracking": False,
        "verbose": False,
    }
    motion_filter.count = 0
    motion_filter.thresh = 2.5
    motion_filter.tracking_anchor_indices = None
    motion_filter.last_streaming_evssm_fallback = False
    motion_filter.last_streaming_candidate_safe = True
    motion_filter.last_streaming_selection = "raw"
    motion_filter.deblur_backend_name = "turtle_bsd_streaming"
    motion_filter.fmap = torch.zeros(1, 1, 1, 1)
    motion_filter.net = torch.zeros(1, 1, 1)
    motion_filter.inp = torch.zeros(1, 1, 1)

    raw = torch.linspace(0.05, 0.95, 3 * 4 * 6).reshape(1, 3, 4, 6)
    raw_before = raw.clone()
    rejected = torch.zeros_like(raw)
    appended = []

    def append(*args):
        appended.append(args)
        motion_filter.video.counter.value += 1

    motion_filter.video = SimpleNamespace(
        counter=SimpleNamespace(value=1),
        down_scale=1,
        append=append,
    )
    motion_filter._streaming_deblur = (
        lambda image, timestamp, stream: (image, rejected, False, -0.5)
    )
    motion_filter._MotionFilter__feature_encoder = (
        lambda inputs: torch.zeros(1, 1, 1, 1)
    )
    motion_filter._MotionFilter__context_encoder = lambda inputs: (
        torch.zeros(1, 1, 1, 1),
        torch.zeros(1, 1, 1, 1),
    )
    motion_filter.update = lambda *args: (
        None,
        torch.full((1, 1, 1, 2), 3.0),
        torch.ones(1, 1, 1, 1),
    )
    motion_filter.deblur_degree_detect = lambda original, candidate: 2.0
    motion_filter._mono_depth_for_frame = (
        lambda timestamp, image, stream: torch.ones(4, 6)
    )

    class FakeCorrBlock:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __call__(self, coords):
            return torch.zeros_like(coords[..., :1])

    with patch.object(motion_filter_module, "CorrBlock", FakeCorrBlock), patch.object(
        motion_filter_module.pops,
        "coords_grid",
        lambda ht, wd, device=None: torch.zeros(ht, wd, 2, device=device),
    ):
        result = motion_filter.track(
            72,
            raw,
            intrinsics=torch.tensor([1.0, 1.0, 0.5, 0.5]),
            fake_sharp=True,
            sharp_judge=False,
            stream=None,
            init=True,
        )

    assert result == (None, 1.0, True, True)
    assert motion_filter.video.counter.value == 2
    assert len(appended) == 1
    assert torch.equal(appended[0][1], raw_before[0])
    assert torch.equal(raw, raw_before)
    assert motion_filter.last_track_info["appended"] is True
    assert motion_filter.count == 0


def test_fallback_is_consumed_only_by_legacy_keyframe_deblur_path() -> None:
    from unittest.mock import patch

    import thirdparty.glorie_slam.motion_filter as motion_filter_module
    from src.tracker import _preserve_latest_keyframe

    MotionFilter = motion_filter_module.MotionFilter
    motion_filter = MotionFilter.__new__(MotionFilter)
    motion_filter.device = "cpu"
    motion_filter.MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    motion_filter.STDV = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    motion_filter.cfg = {
        "deblur": {
            "stream_every_frame": True,
            "stream_apply_to_tracking": True,
            "stream_min_laplacian_gain": 0.0,
            "stream_min_vs_evssm_gain": 0.0,
            "stream_replace_sharp": True,
        },
        "mono_prior": {"predict_online": False},
        "exam_blur_score": False,
        "only_tracking": True,
        "verbose": False,
    }
    motion_filter.count = 0
    motion_filter.thresh = 2.5
    motion_filter.tracking_anchor_indices = {7}
    motion_filter.net = torch.zeros(1, 1, 1)
    motion_filter.inp = torch.zeros(1, 1, 1)
    motion_filter.fmap = torch.zeros(1, 1, 1, 1)

    appended = []
    feature_inputs = []
    motion_filter.video = SimpleNamespace(
        counter=SimpleNamespace(value=1),
        down_scale=1,
        append=lambda *args: appended.append(args),
    )
    raw = torch.full((1, 3, 4, 6), 0.10)
    raw_before = raw.clone()
    temporal_candidate = torch.full_like(raw, 0.20)
    evssm_output = torch.full_like(raw, 0.30)
    evssm_before = evssm_output.clone()
    motion_filter.deblur_backend_name = "causal_evssm"
    motion_filter.deblur_backend = SimpleNamespace(last_evssm_output=evssm_output)
    backend_inputs = []

    def apply_and_advance(image, stream=None, timestamp=None):
        del stream
        backend_inputs.append((timestamp, image.clone()))
        return temporal_candidate

    def feature_encoder(inputs):
        feature_inputs.append(inputs.clone())
        return torch.zeros(1, 1, 1, 1)

    motion_filter.apply_evssm_deblur = apply_and_advance
    motion_filter._laplacian_value = lambda image: float(image.mean().item())
    motion_filter._MotionFilter__feature_encoder = feature_encoder
    motion_filter._MotionFilter__context_encoder = lambda inputs: (
        torch.zeros(1, 1, 1, 1),
        torch.zeros(1, 1, 1, 1),
    )
    motion_filter.update = lambda *args: (
        None,
        torch.zeros(1, 1, 1, 2),
        torch.ones(1, 1, 1, 1),
    )
    motion_filter.deblur_degree_detect = lambda original, deblurred: 0.5
    motion_filter._mono_depth_for_frame = (
        lambda timestamp, image, stream: torch.ones(4, 6)
    )

    class FakeCorrBlock:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __call__(self, coords):
            return torch.zeros_like(coords[..., :1])

    with patch.object(motion_filter_module, "CorrBlock", FakeCorrBlock), patch.object(
        motion_filter_module.pops,
        "coords_grid",
        lambda ht, wd, device=None: torch.zeros(ht, wd, 2, device=device),
    ):
        result = motion_filter.track(
            7,
            raw,
            intrinsics=torch.tensor([1.0, 1.0, 0.5, 0.5]),
            fake_sharp=True,
            sharp_judge=False,
            stream=None,
            init=True,
        )

    assert len(backend_inputs) == 1
    assert torch.equal(backend_inputs[0][1], raw_before)
    # DROID motion selection saw the raw observation; only after the legacy
    # anchor/blur/fake-sharp decision did it encode the EVSSM fallback.
    assert len(feature_inputs) == 2
    expected_raw = (raw_before[None] - motion_filter.MEAN) / motion_filter.STDV
    expected_evssm = (evssm_before[None] - motion_filter.MEAN) / motion_filter.STDV
    assert torch.allclose(feature_inputs[0], expected_raw, atol=0.0, rtol=0.0)
    assert torch.allclose(feature_inputs[1], expected_evssm, atol=0.0, rtol=0.0)
    assert len(appended) == 1
    assert torch.equal(appended[0][1], evssm_before[0])
    assert appended[0][1].data_ptr() != evssm_output.data_ptr()
    assert torch.equal(raw, raw_before)
    assert torch.equal(evssm_output, evssm_before)
    assert float(appended[0][1].min()) >= 0.0
    assert float(appended[0][1].max()) <= 1.0
    assert torch.equal(result[0], evssm_before)
    assert result[1:] == (0.5, True, True)
    assert motion_filter.last_track_info["streaming_replaced"] is False
    assert motion_filter.last_track_info["streaming_evssm_fallback"] is True
    assert motion_filter.last_track_info["motion_keyframe"] is False
    assert motion_filter.last_track_info["tracking_anchor"] is True
    assert motion_filter.last_track_info["appended"] is True
    # The protocol anchor may be preserved, but fallback status by itself can
    # never request causal motion recovery/preservation.
    no_anchor = {**motion_filter.last_track_info, "tracking_anchor": False}
    assert _preserve_latest_keyframe(no_anchor) is False


def test_failed_gate_advances_real_history_and_matches_evssm_output() -> None:
    from thirdparty.glorie_slam.motion_filter import MotionFilter

    class IdentityEVSSM(torch.nn.Module):
        def forward(self, image):
            return image

    class FlatTemporal(torch.nn.Module):
        def forward(self, frames):
            return torch.zeros_like(frames[:, -1]) + 0.5

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "flat_temporal.pt"
        history = 4
        traced = torch.jit.trace(
            FlatTemporal().eval(), torch.zeros(1, history, 3, 8, 8)
        )
        metadata = {
            "format": "unblur_slam.causal_video_deblur.torchscript.v1",
            "model_config": {
                "max_history": history,
                "input_domain": "evssm",
                "use_teacher_input": False,
            },
            "teacher_provenance": {
                "storage": "runtime_evssm_float_tensor",
            },
        }
        torch.jit.save(
            traced,
            str(checkpoint),
            _extra_files={"metadata.json": json.dumps(metadata)},
        )
        backend = CausalEVSSMBackend(
            evssm_model=IdentityEVSSM(),
            checkpoint=checkpoint,
            history=history,
            device="cpu",
        )

        motion_filter = MotionFilter.__new__(MotionFilter)
        motion_filter.cfg = {
            "apply_inverse_gamma": False,
            "deblur": {
                "stream_every_frame": True,
                "stream_apply_to_tracking": True,
                "stream_min_laplacian_gain": 0.0,
                "stream_min_vs_evssm_gain": 0.0,
                "stream_replace_sharp": True,
            },
            "verbose": False,
        }
        motion_filter.device = "cpu"
        motion_filter.MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        motion_filter.STDV = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        motion_filter.deblur_backend_name = "causal_evssm"
        motion_filter.deblur_backend = backend

        checkerboard = (
            torch.arange(8).view(1, 1, 8, 1)
            + torch.arange(8).view(1, 1, 1, 8)
        ) % 2
        raw = checkerboard.float().repeat(1, 3, 1, 1)
        raw_before = raw.clone()
        standalone_evssm = EVSSMBackend(IdentityEVSSM(), "cpu")(raw)

        selected, effective_candidate, replaced, effective_gain = (
            motion_filter._streaming_deblur(raw, 0, None)
        )
        assert len(backend.frames) == 1
        assert motion_filter.last_streaming_evssm_fallback is True
        assert replaced is False
        assert torch.equal(selected, raw_before)
        assert torch.equal(effective_candidate, standalone_evssm)
        assert torch.equal(raw, raw_before)
        assert effective_candidate.data_ptr() != raw.data_ptr()
        assert effective_candidate.data_ptr() != backend.last_evssm_output.data_ptr()
        assert float(effective_candidate.min()) >= 0.0
        assert float(effective_candidate.max()) <= 1.0
        assert not torch.equal(effective_candidate, torch.full_like(selected, 0.5))
        assert not motion_filter._streaming_candidate_rejected(
            effective_candidate, effective_gain
        )

        fallback_before = effective_candidate.clone()
        backend_evssm_before = backend.last_evssm_output.clone()
        history_before = backend.frames[-1].clone()
        normalized = motion_filter._normalize_droid_input(effective_candidate)
        expected = (
            fallback_before[None] - motion_filter.MEAN
        ) / motion_filter.STDV
        assert torch.allclose(normalized, expected, atol=0.0, rtol=0.0)
        assert normalized.data_ptr() != effective_candidate.data_ptr()
        assert torch.equal(effective_candidate, fallback_before)
        assert torch.equal(backend.last_evssm_output, backend_evssm_before)
        assert torch.equal(backend.frames[-1], history_before)

        motion_filter._streaming_deblur(1.0 - raw, 1, None)
        assert len(backend.frames) == 2


def main() -> None:
    test_shape_gradient_and_last_frame_contract()
    test_v3_identity_initialization_and_hard_residual_bound()
    test_legacy_config_without_bound_preserves_v1_formula()
    test_v3_objective_terms_are_prefix_safe_and_numerically_bound()
    test_future_frames_do_not_change_past_outputs()
    test_teacher_path_and_torchscript()
    test_jsonl_sequence_dataset()
    test_precompute_report_binds_manifest_teacher_and_evssm_sha()
    test_teacher_configuration_is_not_silently_unused()
    test_evssm_teacher_is_always_microbatched()
    test_teacher_input_train_and_export_are_rejected()
    test_legacy_v1_checkpoint_config_still_exports_with_v1_semantics()
    test_v3_export_fails_closed_without_objective_contract()
    test_deployment_selector_binds_checkpoint_and_all_gate_metrics()
    test_layered_selector_rejects_unbound_or_incomplete_reports()
    test_history1_control_is_content_bound_and_recomputed()
    test_room2_evaluator_evidence_is_content_bound_and_recomputed()
    test_train_and_export_cli_smoke()
    test_evssm_domain_training_and_composite_streaming()
    test_causal_evssm_gate_falls_back_to_single_frame_selection()
    test_droid_normalization_never_aliases_rgb_observations()
    test_first_track_frame_advances_frontend_but_stays_legacy_raw()
    test_rejected_streaming_candidate_appends_raw_with_new_video_index()
    test_fallback_is_consumed_only_by_legacy_keyframe_deblur_path()
    test_failed_gate_advances_real_history_and_matches_evssm_output()
    print("causal_video_deblur=PASS")


if __name__ == "__main__":
    main()
