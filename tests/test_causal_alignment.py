#!/usr/bin/env python3
"""CPU tests for the isolated causal feature-alignment primitive."""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_deblur.alignment import CoarseLocalCorrelationAligner


def _identity_projection(aligner: CoarseLocalCorrelationAligner) -> None:
    if aligner.input_channels != aligner.match_channels:
        raise ValueError("identity projection requires equal channel counts")
    with torch.no_grad():
        aligner.match_projection.weight.zero_()
        identity = torch.eye(aligner.input_channels)
        aligner.match_projection.weight[:, :, 0, 0].copy_(identity)


def _translated_quarter_features(
    *, channels: int = 16, coarse_height: int = 24, coarse_width: int = 24
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Make exact one-eighth features and expand them to one-quarter scale."""

    torch.manual_seed(4)
    dx_coarse, dy_coarse = -5, 3
    previous_coarse = F.normalize(
        torch.randn(1, channels, coarse_height, coarse_width), dim=1
    )
    current_coarse = torch.zeros_like(previous_coarse)
    valid_coarse = torch.zeros(
        1, 1, coarse_height, coarse_width, dtype=torch.bool
    )

    y0 = max(0, -dy_coarse)
    y1 = min(coarse_height, coarse_height - dy_coarse)
    x0 = max(0, -dx_coarse)
    x1 = min(coarse_width, coarse_width - dx_coarse)
    current_coarse[:, :, y0:y1, x0:x1] = previous_coarse[
        :, :, y0 + dy_coarse : y1 + dy_coarse, x0 + dx_coarse : x1 + dx_coarse
    ]
    valid_coarse[:, :, y0:y1, x0:x1] = True

    previous = previous_coarse.repeat_interleave(2, dim=2).repeat_interleave(
        2, dim=3
    )
    current = current_coarse.repeat_interleave(2, dim=2).repeat_interleave(
        2, dim=3
    )
    valid = valid_coarse.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    return previous, current, valid, dx_coarse * 2, dy_coarse * 2


def test_known_shift_sign_range_confidence_and_warp_improvement() -> None:
    aligner = CoarseLocalCorrelationAligner(
        input_channels=16, match_channels=16, radius=8, temperature=0.05
    ).eval()
    _identity_projection(aligner)
    previous, current, valid, expected_dx, expected_dy = (
        _translated_quarter_features()
    )

    with torch.no_grad():
        aligned, flow, confidence = aligner(previous, current)

    # Exclude one upsample-transition pixel around both the image boundary and
    # the translated support boundary.  Those pixels intentionally mix valid
    # and occluded coarse cells under bilinear flow upsampling.
    interior = F.max_pool2d(
        (~valid).float(), kernel_size=3, stride=1, padding=1
    ) == 0.0
    interior[:, :, :1] = False
    interior[:, :, -1:] = False
    interior[:, :, :, :1] = False
    interior[:, :, :, -1:] = False
    mask = interior[:, 0]
    measured_dx = float(flow[:, 0][mask].median().item())
    measured_dy = float(flow[:, 1][mask].median().item())
    endpoint_error = torch.sqrt(
        (flow[:, 0][mask] - float(expected_dx)).square()
        + (flow[:, 1][mask] - float(expected_dy)).square()
    )

    assert abs(measured_dx - expected_dx) < 0.10
    assert abs(measured_dy - expected_dy) < 0.10
    assert float(endpoint_error.mean()) < 0.10
    assert float(torch.quantile(endpoint_error, 0.90)) < 0.10
    assert float(flow.abs().max()) <= aligner.max_flow + 1.0e-6
    assert float(confidence[interior].mean()) > 0.90

    identity_mae = (previous - current).abs()[interior.expand_as(previous)].mean()
    aligned_mae = (aligned - current).abs()[interior.expand_as(previous)].mean()
    assert float(aligned_mae) < 0.05 * float(identity_mae)


def test_border_mask_border_padding_and_no_nan() -> None:
    torch.manual_seed(5)
    aligner = CoarseLocalCorrelationAligner(
        input_channels=4, match_channels=4, radius=8
    ).eval()
    previous = torch.rand(2, 4, 8, 10)
    current = torch.rand_like(previous)
    with torch.no_grad():
        aligned, flow, confidence = aligner(previous, current)

    assert bool(torch.isfinite(aligned).all())
    assert bool(torch.isfinite(flow).all())
    assert bool(torch.isfinite(confidence).all())
    assert float(confidence.min()) >= 0.0
    assert float(confidence.max()) <= 1.0
    assert float(flow.abs().max()) <= aligner.max_flow + 1.0e-6

    # A one-pixel coarse map has exactly one valid candidate.  It must produce
    # zero flow/confidence rather than an entropy division NaN.
    tiny_previous = torch.rand(1, 4, 2, 2)
    tiny_current = torch.rand_like(tiny_previous)
    tiny_aligned, tiny_flow, tiny_confidence = aligner(
        tiny_previous, tiny_current
    )
    assert bool(torch.isfinite(tiny_aligned).all())
    assert torch.equal(tiny_flow, torch.zeros_like(tiny_flow))
    assert torch.equal(tiny_confidence, torch.zeros_like(tiny_confidence))

    # Border padding must not manufacture black wedges.  The validity mask
    # separately identifies which samples were extrapolated.
    ones = torch.ones(1, 3, 8, 10)
    extreme_flow = torch.full((1, 2, 8, 10), aligner.max_flow)
    warped, valid = aligner.warp_source(ones, extreme_flow)
    assert torch.equal(warped, ones)
    assert valid.shape == (1, 1, 8, 10)
    assert float(valid.sum()) == 0.0
    assert bool(torch.isfinite(warped).all())


def test_pair_contract_is_stateless_and_direction_is_current_to_previous() -> None:
    aligner = CoarseLocalCorrelationAligner(
        input_channels=16, match_channels=16, radius=8, temperature=0.05
    ).eval()
    _identity_projection(aligner)
    previous, current, valid, expected_dx, expected_dy = (
        _translated_quarter_features()
    )

    with torch.no_grad():
        first = aligner(previous, current)
        # An unrelated later pair cannot alter the result: the primitive owns
        # no history and its API contains only previous/current tensors.
        aligner(torch.randn_like(previous), torch.randn_like(current))
        repeated = aligner(previous, current)
        reverse_flow, _ = aligner.estimate_flow(current, previous)

    for original, rerun in zip(first, repeated):
        assert torch.equal(original, rerun)

    mask = valid[:, 0]
    forward_dx = float(first[1][:, 0][mask].median().item())
    forward_dy = float(first[1][:, 1][mask].median().item())
    # On the common interior, reversing pair order reverses sampling direction.
    reverse_dx = float(reverse_flow[:, 0][mask].median().item())
    reverse_dy = float(reverse_flow[:, 1][mask].median().item())
    assert abs(forward_dx - expected_dx) < 0.10
    assert abs(forward_dy - expected_dy) < 0.10
    assert abs(reverse_dx + expected_dx) < 0.30
    assert abs(reverse_dy + expected_dy) < 0.30


def test_gradients_reach_features_projection_and_flow_warp() -> None:
    torch.manual_seed(8)
    aligner = CoarseLocalCorrelationAligner(
        input_channels=6, match_channels=4, radius=3, temperature=0.08
    )
    previous = torch.rand(2, 6, 12, 16, requires_grad=True)
    current = torch.rand(2, 6, 12, 16, requires_grad=True)

    aligned, flow, confidence = aligner(previous, current)
    loss = (
        (aligned - current).square().mean()
        + 0.01 * flow.square().mean()
        + 0.01 * confidence.mean()
    )
    loss.backward()

    assert previous.grad is not None and bool(torch.isfinite(previous.grad).all())
    assert current.grad is not None and bool(torch.isfinite(current.grad).all())
    assert float(previous.grad.abs().sum()) > 0.0
    assert float(current.grad.abs().sum()) > 0.0
    projection_gradient = aligner.match_projection.weight.grad
    assert projection_gradient is not None
    assert bool(torch.isfinite(projection_gradient).all())
    assert float(projection_gradient.abs().sum()) > 0.0

    independent_source = torch.rand(2, 5, 12, 16, requires_grad=True)
    independent_flow = torch.zeros(2, 2, 12, 16, requires_grad=True)
    warped, _ = aligner.warp_source(independent_source, independent_flow)
    warped.square().mean().backward()
    assert independent_source.grad is not None
    assert independent_flow.grad is not None
    assert float(independent_flow.grad.abs().sum()) > 0.0


def test_torchscript_forward_and_exported_methods_match_eager() -> None:
    torch.manual_seed(9)
    aligner = CoarseLocalCorrelationAligner(
        input_channels=4, match_channels=3, radius=3, temperature=0.07
    ).eval()
    previous = torch.rand(1, 4, 12, 16)
    current = torch.rand_like(previous)

    scripted = torch.jit.script(aligner)
    with torch.no_grad():
        eager_forward = aligner(previous, current)
        scripted_forward = scripted(previous, current)
        eager_flow = aligner.estimate_flow(previous, current)
        scripted_flow = scripted.estimate_flow(previous, current)
        eager_warp = aligner.warp_source(previous, eager_flow[0])
        scripted_warp = scripted.warp_source(previous, eager_flow[0])

    for eager, compiled in zip(eager_forward, scripted_forward):
        assert torch.allclose(eager, compiled, atol=1.0e-6, rtol=1.0e-6)
    for eager, compiled in zip(eager_flow, scripted_flow):
        assert torch.allclose(eager, compiled, atol=1.0e-6, rtol=1.0e-6)
    for eager, compiled in zip(eager_warp, scripted_warp):
        assert torch.allclose(eager, compiled, atol=1.0e-6, rtol=1.0e-6)


if __name__ == "__main__":
    tests = [
        test_known_shift_sign_range_confidence_and_warp_improvement,
        test_border_mask_border_padding_and_no_nan,
        test_pair_contract_is_stateless_and_direction_is_current_to_previous,
        test_gradients_reach_features_projection_and_flow_warp,
        test_torchscript_forward_and_exported_methods_match_eager,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
