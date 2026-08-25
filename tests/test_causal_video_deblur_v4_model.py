"""CPU-only contracts for the optional v4 motion-aligned model branch."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_deblur import (
    CausalVideoDeblur,
    MotionAlignedCausalVideoDeblurV4,
    build_causal_video_deblur,
)


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


def _v4_config() -> dict[str, object]:
    return {
        **_base_config(),
        "motion_alignment": {
            "mode": "coarse_local_correlation_v1",
            "match_channels": 2,
            "radius": 1,
            "temperature": 0.1,
        },
    }


def _randomize_residual_head(model: CausalVideoDeblur) -> None:
    with torch.no_grad():
        model.output.weight.normal_(std=0.02)
        model.output.bias.normal_(std=0.01)


def test_old_config_builds_exact_v3_class_and_state_contract() -> None:
    torch.manual_seed(10)
    model = build_causal_video_deblur(_base_config())
    assert type(model) is CausalVideoDeblur
    assert model.config_dict() == _base_config()
    assert len(model.state_dict()) == 68
    assert not any("motion" in key for key in model.state_dict())

    # Preserve the legacy supplied-config rule: absence of max_residual means
    # the old unbounded v1 output formula, not the fresh-model v3 default.
    legacy = _base_config()
    del legacy["max_residual"]
    assert build_causal_video_deblur(legacy).max_residual == 0.0


def test_v3_warm_start_has_only_declared_v4_keys_and_zero_gate_identity() -> None:
    torch.manual_seed(20)
    v3 = build_causal_video_deblur(_base_config()).eval()
    _randomize_residual_head(v3)
    torch.manual_seed(21)
    v4 = build_causal_video_deblur(_v4_config()).eval()
    assert type(v4) is MotionAlignedCausalVideoDeblurV4

    result = v4.load_state_dict(v3.state_dict(), strict=False)
    expected_new_keys = {
        "motion_alignment_gate",
        "motion_aligner.match_projection.weight",
        "motion_aligner.offsets",
    }
    assert set(result.missing_keys) == expected_new_keys
    assert result.unexpected_keys == []
    assert set(v4.state_dict()) - set(v3.state_dict()) == expected_new_keys
    for key, value in v3.state_dict().items():
        assert torch.equal(v4.state_dict()[key], value), key
    assert v4.config_dict() == _v4_config()

    # 13x17 produces an odd 3x4 bottleneck and exercises model-owned padding.
    frames = torch.rand(1, 3, 3, 13, 17)
    with torch.no_grad():
        expected = v3.forward_sequence(frames)
        actual = v4.forward_sequence(frames)
        disabled = v4.forward_sequence_alignment_disabled(frames)
    assert torch.equal(actual, expected)
    assert torch.equal(disabled, expected)

    with torch.no_grad():
        v4.motion_alignment_gate.fill_(0.7)
        enabled = v4.forward_sequence(frames)
        disabled_after_gate = v4.forward_sequence_alignment_disabled(frames)
    assert torch.equal(disabled_after_gate, expected)
    assert not torch.equal(enabled, expected)


def test_v4_is_causal_and_handles_minimal_bottlenecks_and_bound() -> None:
    torch.manual_seed(30)
    model = build_causal_video_deblur(_v4_config()).eval()
    assert isinstance(model, MotionAlignedCausalVideoDeblurV4)
    _randomize_residual_head(model)
    with torch.no_grad():
        model.motion_alignment_gate.fill_(0.7)

    original = torch.rand(1, 3, 3, 13, 17)
    changed_future = original.clone()
    changed_future[:, 2] = torch.rand_like(changed_future[:, 2]) * 4.0 - 2.0
    with torch.no_grad():
        original_outputs = model.forward_sequence(original)
        changed_outputs = model.forward_sequence(changed_future)
    assert torch.allclose(
        original_outputs[:, :2], changed_outputs[:, :2], atol=1.0e-6, rtol=1.0e-6
    )

    # A 4x4 image yields a 1x1 bottleneck; v4 pads it to 2x2 internally.
    one_frame = torch.rand(2, 1, 3, 4, 4)
    with torch.no_grad():
        prediction, flow, confidence, valid = (
            model.forward_sequence_with_motion_diagnostics(one_frame)
        )
    assert prediction.shape == one_frame.shape
    assert flow.shape == (2, 0, 2, 1, 1)
    assert confidence.shape == (2, 0, 1, 1, 1)
    assert valid.shape == (2, 0, 1, 1, 1)

    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(100.0)
        bounded = model.forward_sequence(original)
    assert float((bounded - original).abs().max()) <= model.max_residual + 1.0e-7


def test_v4_diagnostic_loss_reaches_gate_and_alignment_projection() -> None:
    torch.manual_seed(40)
    model = build_causal_video_deblur(_v4_config()).train()
    assert isinstance(model, MotionAlignedCausalVideoDeblurV4)
    _randomize_residual_head(model)
    frames = torch.rand(1, 3, 3, 12, 16, requires_grad=True)
    target = torch.rand_like(frames)
    prediction, flow, confidence, valid = (
        model.forward_sequence_with_motion_diagnostics(frames)
    )
    assert flow.shape == (1, 2, 2, 3, 4)
    assert confidence.shape == (1, 2, 1, 3, 4)
    assert valid.shape == (1, 2, 1, 3, 4)
    loss = torch.nn.functional.l1_loss(prediction, target)
    loss = loss + 0.05 * flow.square().mean() + 0.01 * confidence.mean()
    loss.backward()

    gate_gradient = model.motion_alignment_gate.grad
    projection_gradient = model.motion_aligner.match_projection.weight.grad
    assert gate_gradient is not None and bool(torch.isfinite(gate_gradient))
    assert float(gate_gradient.abs()) > 0.0
    assert projection_gradient is not None
    assert bool(torch.isfinite(projection_gradient).all())
    assert float(projection_gradient.abs().sum()) > 0.0
    assert frames.grad is not None and bool(torch.isfinite(frames.grad).all())


def test_v4_torchscript_preserves_public_and_diagnostic_methods() -> None:
    torch.manual_seed(50)
    model = build_causal_video_deblur(_v4_config()).eval()
    assert isinstance(model, MotionAlignedCausalVideoDeblurV4)
    _randomize_residual_head(model)
    with torch.no_grad():
        model.motion_alignment_gate.fill_(0.4)
    frames = torch.rand(1, 3, 3, 13, 17)

    scripted = torch.jit.script(model)
    frozen = torch.jit.freeze(
        scripted,
        preserved_attrs=[
            "forward_sequence",
            "forward_sequence_with_motion_diagnostics",
            "forward_sequence_alignment_disabled",
        ],
    )
    with torch.no_grad():
        eager_sequence = model.forward_sequence(frames)
        scripted_sequence = frozen.forward_sequence(frames)
        eager_diagnostics = model.forward_sequence_with_motion_diagnostics(frames)
        scripted_diagnostics = frozen.forward_sequence_with_motion_diagnostics(frames)
        eager_disabled = model.forward_sequence_alignment_disabled(frames)
        scripted_disabled = frozen.forward_sequence_alignment_disabled(frames)
    assert torch.allclose(
        eager_sequence, scripted_sequence, atol=1.0e-6, rtol=1.0e-6
    )
    assert torch.equal(frozen(frames), scripted_sequence[:, -1])
    for eager, compiled in zip(eager_diagnostics, scripted_diagnostics):
        assert torch.allclose(eager, compiled, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(
        eager_disabled, scripted_disabled, atol=1.0e-6, rtol=1.0e-6
    )
