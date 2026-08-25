"""CPU contracts for motion-aligned v4 data and training integration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_deblur import VideoDeblurJsonlDataset, build_causal_video_deblur
from scripts.train_causal_video_deblur import (
    CHECKPOINT_FORMAT_V3,
    CHECKPOINT_FORMAT_V4,
    OBJECTIVE_SCHEMA_V4,
    NUMPY_RNG_ENCODING_V4,
    OPTIMIZATION_SCHEMA_V4,
    RNG_STATE_SCHEMA_V4,
    TRAINING_CONTRACT_SCHEMA_V4,
    V4_ALIGNMENT_LR,
    V4_ALIGNMENT_ONLY_STEPS,
    V4_BASE_LR,
    V4_EXPECTED_EVSSM_SHA256,
    V4_EXPECTED_TRAIN_MANIFEST_SHA256,
    V4_EXPECTED_TRAIN_PRECOMPUTE_SHA256,
    V4_EXPECTED_TRAIN_TEACHER_MANIFEST_SHA256,
    V4_EXPECTED_VAL_MANIFEST_SHA256,
    V4_EXPECTED_VAL_PRECOMPUTE_SHA256,
    V4_EXPECTED_VAL_TEACHER_MANIFEST_SHA256,
    V4_MOTION_ALIGNMENT_CONFIG,
    V4_REGISTERED_CONTRACT_SCHEMA,
    V4_REGISTERED_CONTRACT_SHA256,
    V4_TOTAL_STEPS,
    V4_WARM_START_SHA256,
    WARM_START_SCHEMA_V4,
    build_objective_contract,
    build_optimization_contract,
    build_training_contract,
    build_v4_optimizer,
    capture_v4_rng_state,
    compute_v4_batch_losses,
    configure_v4_phase,
    dataset_transition_summary,
    load_v4_resume,
    load_v4_warm_start,
    model_payload,
    motion_alignment_auxiliary_losses,
    motion_compensated_temporal_delta_l1_loss,
    real_transition_clip_indices,
    save_training_checkpoint,
    stitch_rolling_pairwise_diagnostics,
    validate_v4_options,
    validate_v4_registered_contract,
    validate_v4_training_inventory,
    validate_v4_warm_start_provenance,
    v4_epoch_end_step,
    v4_phase_for_step,
)
from scripts.export_causal_video_deblur import validate_v4_contracts


def _touch_marker_during_unsafe_unpickle(path: str) -> dict[str, object]:
    Path(path).write_text("unsafe loader executed", encoding="utf-8")
    return {}


class _MaliciousCheckpointValue:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _touch_marker_during_unsafe_unpickle, (str(self.marker),)


def _base_config() -> dict[str, object]:
    return {
        "channels": 4,
        "num_heads": 2,
        "num_blocks": 1,
        "max_history": 3,
        "use_teacher_input": False,
        "input_domain": "evssm",
        "max_residual": 8.0 / 255.0,
    }


def _formal_v4_config() -> dict[str, object]:
    return {**_base_config(), "motion_alignment": dict(V4_MOTION_ALIGNMENT_CONFIG)}


def _registered_base_config() -> dict[str, object]:
    """Exact architecture pinned by the formal v4 experiment contract."""

    return {
        "channels": 32,
        "num_heads": 4,
        "num_blocks": 2,
        "max_history": 3,
        "use_teacher_input": False,
        "input_domain": "evssm",
        "max_residual": 8.0 / 255.0,
    }


def _registered_v4_config() -> dict[str, object]:
    return {
        **_registered_base_config(),
        "motion_alignment": dict(V4_MOTION_ALIGNMENT_CONFIG),
    }


def _tiny_v4_config() -> dict[str, object]:
    return {
        **_base_config(),
        "motion_alignment": {
            "mode": "coarse_local_correlation_v1",
            "match_channels": 2,
            "radius": 1,
            "temperature": 0.1,
        },
    }


def _args(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "motion_alignment_v4": True,
        "warm_start_v3": Path("source_v3.pth"),
        "resume": None,
        "history": 3,
        "input_domain": "evssm",
        "teacher_input": False,
        "max_steps": V4_TOTAL_STEPS,
        "v4_alignment_only_steps": V4_ALIGNMENT_ONLY_STEPS,
        "warmup_steps": 0,
        "weight_decay": 1.0e-3,
        "v4_base_lr": V4_BASE_LR,
        "v4_alignment_lr": V4_ALIGNMENT_LR,
        "v4_alignment_photo_weight": 1.0,
        "v4_alignment_gradient_weight": 0.2,
        "v4_alignment_smooth_weight": 0.01,
        "v4_joint_alignment_weight": 0.05,
        "channels": 32,
        "heads": 4,
        "blocks": 2,
        "crop_size": 192,
        "batch_size": 4,
        "grad_accumulation": 2,
        "workers": 0,
        "clip_stride": 1,
        "seed": 42,
        "max_residual": 8.0 / 255.0,
        "fft_weight": 0.1,
        "distill_weight": 0.0,
        "evssm_fidelity_weight": 0.1,
        "temporal_delta_weight": 0.05,
        "edge_weight": 0.05,
        "laplacian_gate_weight": 0.02,
        "grad_clip": 1.0,
        "amp": False,
        "device": "cpu",
        "dry_run": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _summary() -> dict[str, int]:
    return {
        "train_clips": 234,
        "train_sequences": 127,
        "real_transition_slots": 169,
        "unique_real_transitions": 107,
        "alignment_sampler_clips": 107,
    }


def _data_identity() -> dict[str, str]:
    return {
        "train_manifest_sha256": V4_EXPECTED_TRAIN_MANIFEST_SHA256,
        "train_precompute_report_sha256": V4_EXPECTED_TRAIN_PRECOMPUTE_SHA256,
        "train_teacher_manifest_sha256": (
            V4_EXPECTED_TRAIN_TEACHER_MANIFEST_SHA256
        ),
        "val_manifest_sha256": V4_EXPECTED_VAL_MANIFEST_SHA256,
        "val_precompute_report_sha256": V4_EXPECTED_VAL_PRECOMPUTE_SHA256,
        "val_teacher_manifest_sha256": V4_EXPECTED_VAL_TEACHER_MANIFEST_SHA256,
        "evssm_checkpoint_sha256": V4_EXPECTED_EVSSM_SHA256,
    }


def _warm_provenance(base_config: dict[str, object]) -> dict[str, object]:
    return {
        "schema": WARM_START_SCHEMA_V4,
        "source_path": "/registered/source_v3.pth",
        "source_sha256": V4_WARM_START_SHA256,
        "source_format": CHECKPOINT_FORMAT_V3,
        "source_model_config": dict(base_config),
        "source_state_key_digest_sha256": "a" * 64,
        "copied_key_count": 68,
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
    }


def _write_rgb(path: Path, value: int) -> None:
    array = np.full((8, 10, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def test_dataset_frame_indices_and_exact_prefix_transition_masks(tmp_path: Path) -> None:
    blurry: list[str] = []
    sharp: list[str] = []
    for index in range(3):
        blurry_path = tmp_path / f"b{index}.png"
        sharp_path = tmp_path / f"s{index}.png"
        _write_rgb(blurry_path, 20 + index)
        _write_rgb(sharp_path, 30 + index)
        blurry.append(blurry_path.name)
        sharp.append(sharp_path.name)
    manifest = tmp_path / "sequence.jsonl"
    manifest.write_text(
        json.dumps({"sequence": "one", "blurry": blurry, "sharp": sharp}) + "\n",
        encoding="utf-8",
    )

    dataset = VideoDeblurJsonlDataset(str(manifest), clip_length=4)
    expected_indices = ([0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2])
    expected_masks = ([False, False, False], [False, False, True], [False, True, True])
    for index, (indices, mask) in enumerate(zip(expected_indices, expected_masks)):
        sample = dataset[index]
        assert sample["frame_indices"].tolist() == indices
        assert sample["transition_valid"].dtype == torch.bool
        assert sample["transition_valid"].tolist() == mask

    summary = dataset_transition_summary(dataset)
    assert summary == {
        "train_clips": 3,
        "train_sequences": 1,
        "real_transition_slots": 3,
        "unique_real_transitions": 2,
        "alignment_sampler_clips": 2,
    }
    assert real_transition_clip_indices(dataset) == [1, 2]

    history_one = VideoDeblurJsonlDataset(str(manifest), clip_length=1)
    for index in range(len(history_one)):
        transition = history_one[index]["transition_valid"]
        assert transition.dtype == torch.bool
        assert tuple(transition.shape) == (0,)


def test_formal_options_inventory_phase_and_drop_last_contracts() -> None:
    validate_v4_options(_args())
    validate_v4_options(_args(warm_start_v3=None, resume=Path("v4.pth")))
    for updates in (
        {"amp": True},
        {"device": "auto"},
        {"workers": 1},
        {"dry_run": True},
        {"history": 4},
        {"warm_start_v3": None, "resume": None},
        {"warm_start_v3": Path("v3.pth"), "resume": Path("v4.pth")},
    ):
        with pytest.raises(ValueError):
            validate_v4_options(_args(**updates))

    validate_v4_training_inventory(_summary())
    bad_summary = _summary()
    bad_summary["real_transition_slots"] = 168
    with pytest.raises(ValueError):
        validate_v4_training_inventory(bad_summary)

    assert v4_phase_for_step(99) == "alignment_only"
    assert v4_phase_for_step(100) == "joint"
    assert v4_phase_for_step(599) == "joint"
    assert v4_phase_for_step(600) == "joint"
    assert [v4_epoch_end_step(epoch) for epoch in (0, 6, 7, 8, 24, 25)] == [
        13,
        91,
        100,
        129,
        593,
        600,
    ]

    joint = DataLoader(TensorDataset(torch.arange(234)), batch_size=4, drop_last=True)
    alignment = DataLoader(
        TensorDataset(torch.arange(107)), batch_size=4, drop_last=True
    )
    assert len(joint) == 58
    assert len(alignment) == 26
    assert len(joint) // 2 == 29
    assert len(alignment) // 2 == 13

    objective = build_objective_contract(_args())
    optimization = build_optimization_contract(
        _args(),
        max_steps=600,
        optimizer_steps_per_epoch=29,
        execution_device="cpu",
        amp_effective=False,
    )
    training = build_training_contract(
        {
            **_base_config(),
            "motion_alignment": dict(V4_MOTION_ALIGNMENT_CONFIG),
        },
        motion_alignment_v4=True,
        transition_summary=_summary(),
    )
    assert objective["schema"] == OBJECTIVE_SCHEMA_V4
    assert optimization["schema"] == OPTIMIZATION_SCHEMA_V4
    assert optimization["effective_batch_size"] == 8
    assert optimization["execution_device"] == "cpu"
    assert optimization["amp_effective"] is False
    assert optimization["alignment_micro_batches_per_epoch"] == 26
    assert optimization["joint_micro_batches_per_epoch"] == 58
    assert training["schema"] == TRAINING_CONTRACT_SCHEMA_V4
    assert training["dropped_tail_policy"] == (
        "shuffle_then_drop_incomplete_microbatch_each_epoch"
    )
    assert training["phases"][0]["trainable_parameters"] == [
        "motion_aligner.match_projection.weight"
    ]


def test_warm_start_is_exact_and_fails_closed(tmp_path: Path) -> None:
    torch.manual_seed(7)
    source = build_causal_video_deblur(_base_config()).eval()
    with torch.no_grad():
        source.output.weight.normal_(std=0.01)
        source.output.bias.normal_(std=0.01)
    checkpoint_path = tmp_path / "source.pth"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT_V3,
            "model_config": source.config_dict(),
            "model": source.state_dict(),
            "optimizer": {"must_not_be_loaded": True},
        },
        checkpoint_path,
    )

    torch.manual_seed(8)
    target = build_causal_video_deblur(_formal_v4_config()).eval()
    alignment_before = {
        key: value.clone()
        for key, value in target.state_dict().items()
        if key.startswith("motion_")
    }
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    provenance = load_v4_warm_start(
        target,
        checkpoint_path,
        expected_sha256=checkpoint_sha256,
    )
    assert provenance["source_sha256"] == checkpoint_sha256
    assert provenance["optimizer_state_loaded"] is False
    assert set(provenance["allowed_missing_alignment_keys"]) == {
        "motion_alignment_gate",
        "motion_aligner.match_projection.weight",
        "motion_aligner.offsets",
    }
    assert provenance["identity_probe"]["passed"] is True
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value), key
    for key, value in alignment_before.items():
        assert torch.equal(target.state_dict()[key], value), key

    frames = torch.rand(1, 3, 3, 12, 16)
    with torch.no_grad():
        assert torch.equal(
            source.forward_sequence(frames), target.forward_sequence(frames)
        )

    malformed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    malformed["format"] = CHECKPOINT_FORMAT_V4
    malformed_path = tmp_path / "bad_format.pth"
    torch.save(malformed, malformed_path)
    untouched = build_causal_video_deblur(_formal_v4_config())
    state_before = {key: value.clone() for key, value in untouched.state_dict().items()}
    with pytest.raises(ValueError):
        load_v4_warm_start(untouched, malformed_path)
    for key, value in state_before.items():
        assert torch.equal(untouched.state_dict()[key], value), key

    wrong_sha_target = build_causal_video_deblur(_formal_v4_config())
    wrong_sha_before = {
        key: value.clone() for key, value in wrong_sha_target.state_dict().items()
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_v4_warm_start(
            wrong_sha_target,
            checkpoint_path,
            expected_sha256="0" * 64,
        )
    for key, value in wrong_sha_before.items():
        assert torch.equal(wrong_sha_target.state_dict()[key], value), key

    marker = tmp_path / "unsafe_unpickle_executed"
    malicious_path = tmp_path / "malicious.pth"
    torch.save({"payload": _MaliciousCheckpointValue(marker)}, malicious_path)
    malicious_target = build_causal_video_deblur(_formal_v4_config())
    malicious_before = {
        key: value.clone() for key, value in malicious_target.state_dict().items()
    }
    with pytest.raises(ValueError, match=r"weights_only=True"):
        load_v4_warm_start(malicious_target, malicious_path)
    assert not marker.exists()
    for key, value in malicious_before.items():
        assert torch.equal(malicious_target.state_dict()[key], value), key


def test_phase_parameter_groups_and_one_real_optimizer_step() -> None:
    torch.manual_seed(11)
    model = build_causal_video_deblur(_tiny_v4_config()).train()
    optimizer = build_v4_optimizer(model)
    named = dict(model.named_parameters())
    assert named["motion_aligner.match_projection.weight"].requires_grad
    assert not named["motion_alignment_gate"].requires_grad
    assert all(
        not parameter.requires_grad
        for name, parameter in named.items()
        if not name.startswith("motion_")
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["base"]["lr"] == 0.0
    assert groups["alignment"]["lr"] == V4_ALIGNMENT_LR

    model_input = torch.rand(1, 4, 3, 16, 16)
    sharp = torch.roll(model_input, shifts=1, dims=-1)
    transition_valid = torch.tensor([[False, True, True]])
    available = torch.ones(1, dtype=torch.bool)
    phase1_args = _args()
    projection_before = named["motion_aligner.match_projection.weight"].detach().clone()
    losses = compute_v4_batch_losses(
        model,
        model_input,
        sharp,
        transition_valid,
        model_input,
        available,
        phase1_args,
        "alignment_only",
    )
    assert torch.isfinite(losses["loss"])
    assert int(losses["real_transition_count"].item()) == 2
    losses["loss"].backward()
    projection_grad = named["motion_aligner.match_projection.weight"].grad
    assert projection_grad is not None
    assert bool(torch.isfinite(projection_grad).all())
    assert float(projection_grad.abs().sum()) > 0.0
    assert named["motion_alignment_gate"].grad is None
    assert all(
        parameter.grad is None
        for name, parameter in named.items()
        if not name.startswith("motion_")
    )
    optimizer.step()
    assert not torch.equal(
        named["motion_aligner.match_projection.weight"], projection_before
    )

    optimizer_identity = id(optimizer)
    configure_v4_phase(model, optimizer, "joint")
    assert id(optimizer) == optimizer_identity
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert groups["base"]["lr"] == V4_BASE_LR
    assert groups["alignment"]["lr"] == V4_ALIGNMENT_LR
    with torch.no_grad():
        model.output.weight.normal_(std=0.01)
    optimizer.zero_grad(set_to_none=True)
    joint_losses = compute_v4_batch_losses(
        model,
        model_input,
        sharp,
        transition_valid,
        model_input,
        available,
        phase1_args,
        "joint",
    )
    joint_losses["loss"].backward()
    assert model.output.weight.grad is not None
    assert float(model.output.weight.grad.abs().sum()) > 0.0
    assert named["motion_alignment_gate"].grad is not None
    assert bool(torch.isfinite(named["motion_alignment_gate"].grad))


def test_masks_stitch_zero_valid_and_detached_motion_delta() -> None:
    previous = torch.tensor([[[1.0], [2.0]]])
    current = torch.tensor([[[2.0], [3.0]]])
    stitched = stitch_rolling_pairwise_diagnostics(previous, current)
    assert stitched.flatten().tolist() == [1.0, 2.0, 3.0]
    empty = torch.empty(1, 0, 2, 3, 4)
    assert stitch_rolling_pairwise_diagnostics(empty, empty).shape == empty.shape

    sharp = torch.rand(1, 4, 3, 12, 16)
    flow = torch.randn(1, 3, 2, 3, 4, requires_grad=True)
    photo, gradient, smooth, count = motion_alignment_auxiliary_losses(
        sharp,
        flow,
        torch.zeros(1, 3, dtype=torch.bool),
        diagnostic_valid=torch.ones(1, 3, 1, 3, 4),
    )
    total = photo + gradient + smooth
    assert float(total) == 0.0
    assert float(count) == 0.0
    total.backward()
    assert flow.grad is not None
    assert torch.equal(flow.grad, torch.zeros_like(flow.grad))

    current_prediction = torch.rand(1, 3, 12, 16, requires_grad=True)
    previous_prediction = torch.rand(1, 3, 12, 16, requires_grad=True)
    current_sharp = torch.rand_like(current_prediction)
    previous_sharp = torch.rand_like(previous_prediction)
    detached_flow = torch.zeros(1, 2, 3, 4, requires_grad=True)
    delta = motion_compensated_temporal_delta_l1_loss(
        current_prediction,
        previous_prediction,
        current_sharp,
        previous_sharp,
        detached_flow,
        torch.ones(1, dtype=torch.bool),
        diagnostic_valid=torch.ones(1, 1, 3, 4),
    )
    delta.backward()
    assert current_prediction.grad is not None
    assert previous_prediction.grad is not None
    assert detached_flow.grad is None


def test_strict_v4_resume_restores_state_phase_and_rng(tmp_path: Path) -> None:
    args = _args()
    config = _formal_v4_config()
    model = build_causal_video_deblur(config)
    with torch.no_grad():
        model.output.weight.normal_(std=0.01)
    optimizer = build_v4_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda _: 1.0, lambda _: 1.0]
    )
    configure_v4_phase(model, optimizer, "alignment_only")
    objective = build_objective_contract(args)
    optimization = build_optimization_contract(
        args,
        max_steps=600,
        optimizer_steps_per_epoch=29,
        execution_device="cpu",
        amp_effective=False,
    )
    training = build_training_contract(
        config, motion_alignment_v4=True, transition_summary=_summary()
    )
    registered = validate_v4_registered_contract()
    assert registered["schema"] == V4_REGISTERED_CONTRACT_SCHEMA
    assert registered["sha256"] == V4_REGISTERED_CONTRACT_SHA256
    teacher = {"schema": "test.teacher", "evssm_checkpoint_sha256": "b" * 64}
    train_generator = torch.Generator().manual_seed(1042)
    alignment_generator = torch.Generator().manual_seed(2042)
    rng_state = capture_v4_rng_state(train_generator, alignment_generator)
    assert rng_state["schema"] == RNG_STATE_SCHEMA_V4
    numpy_random_state = rng_state["numpy_random_state"]
    assert isinstance(numpy_random_state, tuple)
    assert numpy_random_state[0] == "MT19937"
    assert isinstance(numpy_random_state[1], torch.Tensor)
    assert numpy_random_state[1].dtype == torch.int64
    assert tuple(numpy_random_state[1].shape) == (624,)
    assert rng_state["numpy_random_state_encoding"] == NUMPY_RNG_ENCODING_V4

    expected_python = random.random()
    expected_numpy = np.random.random_sample(4)
    expected_torch = torch.rand(4)
    expected_train_generator = torch.Generator()
    expected_train_generator.set_state(rng_state["train_loader_generator_state"])
    expected_train = torch.rand(4, generator=expected_train_generator)
    expected_alignment_generator = torch.Generator()
    expected_alignment_generator.set_state(
        rng_state["alignment_loader_generator_state"]
    )
    expected_alignment = torch.rand(4, generator=expected_alignment_generator)
    payload = model_payload(
        model,
        optimizer,
        scheduler,
        epoch=0,
        step=13,
        best_psnr=20.0,
        best_ssim_at_best_psnr=0.8,
        input_domain="evssm",
        teacher_provenance=teacher,
        objective_contract=objective,
        optimization_contract=optimization,
        validation_metrics={"psnr": 20.0, "ssim": 0.8},
        motion_alignment_v4=True,
        warm_start_provenance=_warm_provenance(_base_config()),
        training_contract=training,
        registered_contract=registered,
        data_identity=_data_identity(),
        rng_state=rng_state,
    )
    assert payload["format"] == CHECKPOINT_FORMAT_V4
    checkpoint_path = tmp_path / "resume.pth"
    save_training_checkpoint(payload, checkpoint_path, atomic=True)
    assert checkpoint_path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    safe_loaded = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    assert safe_loaded["format"] == CHECKPOINT_FORMAT_V4
    assert safe_loaded["rng_state"]["numpy_random_state"][1].dtype == torch.int64

    resumed_model = build_causal_video_deblur(config)
    resumed_optimizer = build_v4_optimizer(resumed_model)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer, lr_lambda=[lambda _: 1.0, lambda _: 1.0]
    )
    resumed_train_generator = torch.Generator().manual_seed(1)
    resumed_alignment_generator = torch.Generator().manual_seed(2)
    result = load_v4_resume(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        model_config=config,
        objective_contract=objective,
        optimization_contract=optimization,
        training_contract=training,
        teacher_provenance=teacher,
        registered_contract=registered,
        data_identity=_data_identity(),
        train_loader_generator=resumed_train_generator,
        alignment_loader_generator=resumed_alignment_generator,
    )
    assert result[:4] == (1, 13, 20.0, 0.8)
    for key, value in model.state_dict().items():
        assert torch.equal(resumed_model.state_dict()[key], value), key
    assert torch.equal(
        resumed_train_generator.get_state(), rng_state["train_loader_generator_state"]
    )
    assert torch.equal(
        resumed_alignment_generator.get_state(),
        rng_state["alignment_loader_generator_state"],
    )
    assert random.random() == expected_python
    assert np.array_equal(np.random.random_sample(4), expected_numpy)
    assert torch.equal(torch.rand(4), expected_torch)
    assert torch.equal(
        torch.rand(4, generator=resumed_train_generator), expected_train
    )
    assert torch.equal(
        torch.rand(4, generator=resumed_alignment_generator), expected_alignment
    )
    assert not resumed_model.motion_alignment_gate.requires_grad
    assert resumed_model.motion_aligner.match_projection.weight.requires_grad

    legacy = copy.deepcopy(payload)
    safe_numpy_state = legacy["rng_state"]["numpy_random_state"]
    legacy_numpy_state = (
        safe_numpy_state[0],
        safe_numpy_state[1].numpy().astype(np.uint32, copy=True),
        safe_numpy_state[2],
        safe_numpy_state[3],
        safe_numpy_state[4],
    )
    legacy["rng_state"]["numpy_random_state"] = legacy_numpy_state
    del legacy["rng_state"]["numpy_random_state_encoding"]
    legacy_path = tmp_path / "legacy_numpy_rng.pth"
    torch.save(legacy, legacy_path)
    legacy_model = build_causal_video_deblur(config)
    legacy_optimizer = build_v4_optimizer(legacy_model)
    legacy_scheduler = torch.optim.lr_scheduler.LambdaLR(
        legacy_optimizer, lr_lambda=[lambda _: 1.0, lambda _: 1.0]
    )
    legacy_before = {
        key: value.clone() for key, value in legacy_model.state_dict().items()
    }
    with pytest.raises(ValueError, match=r"Migrate legacy v4"):
        load_v4_resume(
            legacy_path,
            model=legacy_model,
            optimizer=legacy_optimizer,
            scheduler=legacy_scheduler,
            model_config=config,
            objective_contract=objective,
            optimization_contract=optimization,
            training_contract=training,
            teacher_provenance=teacher,
            registered_contract=registered,
            data_identity=_data_identity(),
            train_loader_generator=torch.Generator(),
            alignment_loader_generator=torch.Generator(),
        )
    for key, value in legacy_before.items():
        assert torch.equal(legacy_model.state_dict()[key], value), key

    malformed = copy.deepcopy(payload)
    malformed["objective_contract"]["primary_reconstruction"]["l1_weight"] = 2.0
    malformed_path = tmp_path / "tampered.pth"
    torch.save(malformed, malformed_path)
    untouched = build_causal_video_deblur(config)
    untouched_optimizer = build_v4_optimizer(untouched)
    untouched_scheduler = torch.optim.lr_scheduler.LambdaLR(
        untouched_optimizer, lr_lambda=[lambda _: 1.0, lambda _: 1.0]
    )
    before = {key: value.clone() for key, value in untouched.state_dict().items()}
    with pytest.raises(ValueError):
        load_v4_resume(
            malformed_path,
            model=untouched,
            optimizer=untouched_optimizer,
            scheduler=untouched_scheduler,
            model_config=config,
            objective_contract=objective,
            optimization_contract=optimization,
            training_contract=training,
            teacher_provenance=teacher,
            registered_contract=registered,
            data_identity=_data_identity(),
            train_loader_generator=torch.Generator(),
            alignment_loader_generator=torch.Generator(),
        )
    for key, value in before.items():
        assert torch.equal(untouched.state_dict()[key], value), key

    illegal = copy.deepcopy(payload)
    illegal["step"] = 12
    illegal_path = tmp_path / "illegal_boundary.pth"
    torch.save(illegal, illegal_path)
    with pytest.raises(ValueError, match="legal phase boundary"):
        load_v4_resume(
            illegal_path,
            model=untouched,
            optimizer=untouched_optimizer,
            scheduler=untouched_scheduler,
            model_config=config,
            objective_contract=objective,
            optimization_contract=optimization,
            training_contract=training,
            teacher_provenance=teacher,
            registered_contract=registered,
            data_identity=_data_identity(),
            train_loader_generator=torch.Generator(),
            alignment_loader_generator=torch.Generator(),
        )


def test_pinned_warm_provenance_and_terminal_payload_contract() -> None:
    provenance = _warm_provenance(_base_config())
    validated = validate_v4_warm_start_provenance(
        provenance, expected_base_config=_base_config()
    )
    assert validated["source_sha256"] == V4_WARM_START_SHA256
    bad = copy.deepcopy(provenance)
    bad["source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_v4_warm_start_provenance(
            bad, expected_base_config=_base_config()
        )

    assert v4_epoch_end_step(25) == V4_TOTAL_STEPS
    with pytest.raises(ValueError):
        v4_epoch_end_step(26)

    args = _args()
    config = _registered_v4_config()
    registered_provenance = _warm_provenance(_registered_base_config())
    model = build_causal_video_deblur(config)
    optimizer = build_v4_optimizer(model)
    configure_v4_phase(model, optimizer, "joint")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda _: 1.0, lambda _: 1.0]
    )
    # LambdaLR construction does not define the phase; restore exact joint LRs.
    configure_v4_phase(model, optimizer, "joint")
    training = build_training_contract(
        config, motion_alignment_v4=True, transition_summary=_summary()
    )
    optimization = build_optimization_contract(
        args,
        max_steps=600,
        optimizer_steps_per_epoch=29,
        execution_device="cpu",
        amp_effective=False,
    )
    joint_generator = torch.Generator().manual_seed(1042)
    alignment_generator = torch.Generator().manual_seed(2042)
    terminal = model_payload(
        model,
        optimizer,
        scheduler,
        epoch=25,
        step=600,
        best_psnr=21.0,
        best_ssim_at_best_psnr=0.81,
        input_domain="evssm",
        teacher_provenance={"schema": "test.teacher"},
        objective_contract=build_objective_contract(args),
        optimization_contract=optimization,
        validation_metrics={"psnr": 21.0, "ssim": 0.81},
        motion_alignment_v4=True,
        warm_start_provenance=registered_provenance,
        training_contract=training,
        registered_contract=validate_v4_registered_contract(),
        data_identity=_data_identity(),
        rng_state=capture_v4_rng_state(joint_generator, alignment_generator),
    )
    assert terminal["format"] == CHECKPOINT_FORMAT_V4
    assert terminal["step"] == 600
    assert terminal["training_phase"] == "joint"
    # This is a newly captured/native-safe synthetic payload, not the sole
    # audited migration of the already-completed 511dbc formal run.  Reusing
    # that registered contract without its exact checkpoint lineage must fail;
    # a future native-safe run requires a new contract.
    with pytest.raises(ValueError, match="checkpoint file SHA-256"):
        validate_v4_contracts(terminal, config)
