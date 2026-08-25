#!/usr/bin/env python3
"""Evaluate a causal deblur checkpoint against blurry, EVSSM and sharp frames."""

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_deblur import VideoDeblurJsonlDataset
from scripts.export_causal_video_deblur import (
    ALIGNMENT_DIAGNOSTICS_SCHEMA_V4,
    REGISTERED_V4_CONTRACT_SCHEMA,
    REGISTERED_V4_CONTRACT_SHA256,
    TORCHSCRIPT_FORMAT_V4,
    validate_teacher_provenance,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_digest(value: object, label: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def _load_lpips_metric(device: torch.device):
    """Load LPIPS only for an explicitly requested Layer-2 evaluation.

    Keeping the import and construction here makes the default temporal-val
    path independent of the optional torchmetrics/LPIPS weights.  A requested
    metric is fail-closed: dependency or weight-loading errors are never
    silently converted into a missing/NaN score.
    """

    try:
        from torchmetrics.image.lpip import (
            LearnedPerceptualImagePatchSimilarity,
        )
    except Exception as error:
        raise RuntimeError(
            "--compute-lpips requires torchmetrics with LPIPS support"
        ) from error
    try:
        metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
    except Exception as error:
        raise RuntimeError(
            "--compute-lpips could not load the AlexNet LPIPS model/weights"
        ) from error
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    return metric


def _lpips_value(metric, prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Evaluate one image pair without leaking torchmetrics running state."""

    try:
        with torch.no_grad():
            value = metric(
                prediction.detach().float().clamp(0.0, 1.0).unsqueeze(0),
                target.detach().float().clamp(0.0, 1.0).unsqueeze(0),
            )
        result = float(torch.as_tensor(value).detach().float().mean().item())
        if hasattr(metric, "reset"):
            metric.reset()
    except Exception as error:
        raise RuntimeError("LPIPS evaluation failed") from error
    if not math.isfinite(result):
        raise RuntimeError("LPIPS evaluation returned a non-finite value")
    return result


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_metric=None,
) -> dict[str, float]:
    lpips_value = (
        _lpips_value(lpips_metric, prediction, target)
        if lpips_metric is not None
        else None
    )
    prediction = prediction.detach().float().clamp(0.0, 1.0).cpu()
    target = target.detach().float().clamp(0.0, 1.0).cpu()
    mse = float(torch.mean((prediction - target) ** 2).item())
    pred_np = prediction.permute(1, 2, 0).numpy()
    target_np = target.permute(1, 2, 0).numpy()
    metrics = {
        "psnr": -10.0 * math.log10(max(mse, 1.0e-12)),
        "ssim": float(
            structural_similarity(
                pred_np, target_np, data_range=1.0, channel_axis=2
            )
        ),
        "l1": float(torch.mean(torch.abs(prediction - target)).item()),
    }
    if lpips_value is not None:
        metrics["lpips"] = lpips_value
    return metrics


def _laplacian_variance(image: torch.Tensor) -> float:
    """Match the runtime sharpness gate on a single CHW RGB tensor."""

    image = image.detach().float().clamp(0.0, 1.0)
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError("Laplacian input must have shape [3,H,W]")
    grayscale = image.mean(dim=0, keepdim=True).unsqueeze(0)
    kernel = image.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    response = torch.nn.functional.conv2d(grayscale, kernel, padding=1)
    return float(torch.var(response).item())


def _mean_rows(rows: list[dict], source: str, metric: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([row[source][metric] for row in rows]))


def _quality_breakdown(rows: list[dict]) -> dict[str, object]:
    sources = ("blurry", "evssm", "causal", "causal_repeat_current")
    metrics = ("psnr", "ssim", "l1")
    mean = {
        source: {metric: _mean_rows(rows, source, metric) for metric in metrics}
        for source in sources
    }
    lpips_sources = ("evssm", "causal", "causal_repeat_current")
    lpips_computed = bool(
        rows
        and all("lpips" in row[source] for row in rows for source in lpips_sources)
    )
    if lpips_computed:
        for source in lpips_sources:
            mean[source]["lpips"] = _mean_rows(rows, source, "lpips")
    comparison_metrics = metrics + (("lpips",) if lpips_computed else ())
    return {
        "frame_count": len(rows),
        "mean": mean,
        "causal_minus_evssm": {
            metric: mean["causal"][metric] - mean["evssm"][metric]
            for metric in comparison_metrics
        },
        "causal_minus_repeat_current": {
            metric: mean["causal"][metric] - mean["causal_repeat_current"][metric]
            for metric in comparison_metrics
        },
    }


def _mean(rows: list[dict], key: str, metric: str) -> float:
    return float(np.mean([row[key][metric] for row in rows]))


def _temporal_metrics(
    current: torch.Tensor,
    previous: torch.Tensor,
    current_gt: torch.Tensor,
    previous_gt: torch.Tensor,
) -> dict[str, float]:
    """Adjacent-frame stability without pretending to use optical-flow warp."""

    current = current.detach().float().clamp(0.0, 1.0)
    previous = previous.detach().float().clamp(0.0, 1.0)
    current_gt = current_gt.detach().float().clamp(0.0, 1.0)
    previous_gt = previous_gt.detach().float().clamp(0.0, 1.0)
    predicted_difference = current - previous
    gt_difference = current_gt - previous_gt
    return {
        # This raw change contains real camera/object motion as well as flicker.
        "adjacent_change_l1": float(predicted_difference.abs().mean().item()),
        # Difference-of-differences discounts the GT's observed temporal
        # change, but is not a flow-warped temporal metric.
        "gt_difference_error_l1_not_warp": float(
            (predicted_difference - gt_difference).abs().mean().item()
        ),
    }


def _mean_temporal(rows: list[dict], source: str, metric: str) -> float:
    values = [
        row["temporal"][source][metric]
        for row in rows
        if row.get("temporal") is not None
    ]
    return float(np.mean(values)) if values else float("nan")


def _finite_flat(tensor: torch.Tensor, label: str) -> torch.Tensor:
    values = tensor.detach().float().cpu().reshape(-1)
    if values.numel() == 0:
        raise RuntimeError(f"{label} is empty")
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"{label} contains NaN or infinity")
    return values


def _quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values, probability).item())


def _scalar_distribution(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean().item()),
        "p05": _quantile(values, 0.05),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _alignment_transition(
    flow: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
    from_frame_index: int,
    to_frame_index: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Summarize one real adjacent v4 transition without serializing maps."""

    if flow.dim() != 3 or flow.shape[0] != 2:
        raise RuntimeError("v4 motion flow must have shape [2,h4,w4]")
    if confidence.shape != (1, flow.shape[1], flow.shape[2]):
        raise RuntimeError("v4 motion confidence must have shape [1,h4,w4]")
    if valid.shape != confidence.shape:
        raise RuntimeError("v4 motion valid mask must match confidence")
    quarter_components = _finite_flat(flow, "v4 quarter-resolution flow")
    confidence_values = _finite_flat(confidence, "v4 alignment confidence")
    valid_values = _finite_flat(valid, "v4 warp-valid mask")
    if float(confidence_values.min()) < 0.0 or float(confidence_values.max()) > 1.0:
        raise RuntimeError("v4 alignment confidence is outside [0,1]")
    if float(valid_values.min()) < 0.0 or float(valid_values.max()) > 1.0:
        raise RuntimeError("v4 warp-valid mask is outside [0,1]")

    quarter_dx = flow[0].detach().float().cpu()
    quarter_dy = flow[1].detach().float().cpu()
    scale_x = float(input_width) / float(flow.shape[2])
    scale_y = float(input_height) / float(flow.shape[1])
    quarter_magnitude = torch.sqrt(quarter_dx.square() + quarter_dy.square()).reshape(-1)
    input_dx = quarter_dx * scale_x
    input_dy = quarter_dy * scale_y
    input_magnitude = torch.sqrt(input_dx.square() + input_dy.square()).reshape(-1)
    row = {
        "schema": ALIGNMENT_DIAGNOSTICS_SCHEMA_V4,
        "from_frame_index": int(from_frame_index),
        "to_frame_index": int(to_frame_index),
        "flow_shape": [2, int(flow.shape[1]), int(flow.shape[2])],
        "input_shape": [3, int(input_height), int(input_width)],
        "quarter_to_input_scale": {"x": scale_x, "y": scale_y},
        "flow_quarter_pixels": {
            "magnitude_p95": _quantile(quarter_magnitude, 0.95),
            "magnitude_max": float(quarter_magnitude.max().item()),
            "component_abs_max": float(quarter_components.abs().max().item()),
        },
        "flow_input_pixels": {
            "magnitude_p95": _quantile(input_magnitude, 0.95),
            "magnitude_max": float(input_magnitude.max().item()),
            "dx_abs_max": float(input_dx.abs().max().item()),
            "dy_abs_max": float(input_dy.abs().max().item()),
        },
        "confidence": _scalar_distribution(confidence_values),
        "warp_valid": {
            "mean": float(valid_values.mean().item()),
            "min": float(valid_values.min().item()),
            "max": float(valid_values.max().item()),
        },
        "finite_fraction": 1.0,
    }
    samples = {
        "quarter_components": quarter_components,
        "quarter_magnitude": quarter_magnitude,
        "input_magnitude": input_magnitude,
        "confidence": confidence_values,
        "valid": valid_values,
    }
    return row, samples


def _alignment_summary(
    transitions: list[dict[str, torch.Tensor]],
    *,
    expected_transition_count: int,
    max_component_quarter_pixels: float,
) -> dict[str, object]:
    if len(transitions) != expected_transition_count:
        raise RuntimeError(
            "v4 alignment transition count mismatch: "
            f"{len(transitions)} != {expected_transition_count}"
        )
    if not transitions:
        raise RuntimeError("v4 alignment evaluation requires real transitions")
    combined = {
        key: torch.cat([transition[key] for transition in transitions])
        for key in transitions[0]
    }
    quarter_component_max = float(combined["quarter_components"].abs().max().item())
    if quarter_component_max > max_component_quarter_pixels + 1.0e-5:
        raise RuntimeError("v4 quarter-resolution flow exceeds its configured bound")
    confidence = _scalar_distribution(combined["confidence"])
    valid = combined["valid"]
    return {
        "schema": ALIGNMENT_DIAGNOSTICS_SCHEMA_V4,
        "protocol": {
            "transition_identity": (
                "one newest real adjacent transition per noninitial source frame; "
                "repeated prefix padding is excluded"
            ),
            "flow_statistic": "L2 vector magnitude over all finite transition pixels",
            "flow_quantile": "torch.quantile linear interpolation",
            "input_scale_rule": (
                "dx_input=dx_quarter*input_width/flow_width and "
                "dy_input=dy_quarter*input_height/flow_height"
            ),
        },
        "transition_count": len(transitions),
        "expected_transition_count": int(expected_transition_count),
        "flow_quarter_pixels": {
            "magnitude_p95": _quantile(combined["quarter_magnitude"], 0.95),
            "magnitude_max": float(combined["quarter_magnitude"].max().item()),
            "component_abs_max": quarter_component_max,
            "configured_component_abs_max": float(max_component_quarter_pixels),
            "finite_fraction": 1.0,
        },
        "flow_input_pixels": {
            "magnitude_p95": _quantile(combined["input_magnitude"], 0.95),
            "magnitude_max": float(combined["input_magnitude"].max().item()),
            "finite_fraction": 1.0,
        },
        "confidence": {**confidence, "finite_fraction": 1.0},
        "warp_valid": {
            "mean": float(valid.mean().item()),
            "min": float(valid.min().item()),
            "max": float(valid.max().item()),
            "finite_fraction": 1.0,
        },
        "integrity": {
            "transition_count_matches": True,
            "all_finite": True,
            "flow_within_configured_bound": True,
            "confidence_in_0_1": True,
            "warp_valid_in_0_1": True,
            "passed": True,
        },
    }


def _alignment_disabled_quality(rows: list[dict]) -> dict[str, object]:
    metrics = ("psnr", "ssim", "l1")
    mean = {
        source: {metric: _mean_rows(rows, source, metric) for metric in metrics}
        for source in ("causal", "causal_alignment_disabled")
    }
    if rows and all(
        "lpips" in row[source]
        for row in rows
        for source in ("causal", "causal_alignment_disabled")
    ):
        for source in mean:
            mean[source]["lpips"] = _mean_rows(rows, source, "lpips")
    return {
        "frame_count": len(rows),
        "mean": mean,
        "causal_minus_alignment_disabled": {
            metric: mean["causal"][metric] - mean["causal_alignment_disabled"][metric]
            for metric in mean["causal"]
        },
    }


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach().float().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    )
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8))


def _montage(path: Path, items: list[tuple[str, torch.Tensor]], subtitle: str) -> None:
    images = [(label, _to_pil(tensor)) for label, tensor in items]
    width = sum(image.width for _, image in images)
    height = max(image.height for _, image in images) + 54
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in images:
        canvas.paste(image, (x, 34))
        draw.text((x + 8, 8), label, fill=(255, 255, 255))
        x += image.width
    draw.text((8, height - 18), subtitle, fill=(100, 220, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--precompute-report",
        type=Path,
        help="validate and use the report's content-bound teacher manifest",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visuals", type=int, default=12)
    parser.add_argument(
        "--compute-lpips",
        action="store_true",
        help=(
            "compute AlexNet LPIPS for the locked Layer-2 test; disabled by "
            "default so temporal validation does not load optional weights"
        ),
    )
    args = parser.parse_args()
    if args.manifest is None and args.precompute_report is None:
        raise ValueError("--manifest or --precompute-report is required")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    device = torch.device(args.device)
    lpips_metric = _load_lpips_metric(device) if args.compute_lpips else None
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_files = {"metadata.json": ""}
    model = torch.jit.load(
        str(args.checkpoint.expanduser().resolve()),
        map_location=device,
        _extra_files=extra_files,
    ).eval()
    metadata_value = extra_files["metadata.json"]
    if isinstance(metadata_value, bytes):
        metadata_value = metadata_value.decode("utf-8")
    metadata = json.loads(metadata_value)
    artifact_format = str(metadata.get("format", ""))
    checkpoint_format = str(metadata.get("checkpoint_format", ""))
    is_v4 = artifact_format == TORCHSCRIPT_FORMAT_V4
    if checkpoint_format == "unblur_slam.causal_video_deblur.v4" and not is_v4:
        raise ValueError("a v4 checkpoint must use the v4 TorchScript format")
    source_checkpoint_sha256 = _sha256_digest(
        metadata.get("source_checkpoint_sha256"),
        "TorchScript metadata source_checkpoint_sha256",
    )
    config = metadata.get("model_config", {})
    input_domain = str(config.get("input_domain", "raw"))
    checkpoint_teacher = validate_teacher_provenance(
        metadata.get("teacher_provenance"), input_domain=input_domain
    )
    max_history = int(config.get("max_history", args.history))
    if args.history > max_history:
        raise ValueError("history exceeds checkpoint max_history")
    if is_v4:
        if args.history != 3 or max_history != 3:
            raise ValueError("v4 evaluation is preregistered for history=3")
        expected_alignment = {
            "mode": "coarse_local_correlation_v1",
            "match_channels": 16,
            "radius": 8,
            "temperature": 0.05,
        }
        if config.get("motion_alignment") != expected_alignment:
            raise ValueError("v4 TorchScript motion_alignment is not preregistered")
        registered_contract = metadata.get("registered_contract")
        if not isinstance(registered_contract, dict) or {
            "schema": registered_contract.get("schema"),
            "sha256": registered_contract.get("sha256"),
        } != {
            "schema": REGISTERED_V4_CONTRACT_SCHEMA,
            "sha256": REGISTERED_V4_CONTRACT_SHA256,
        }:
            raise ValueError("v4 TorchScript is not bound to the preregistered contract")
        for method in (
            "forward_sequence_with_motion_diagnostics",
            "forward_sequence_alignment_disabled",
        ):
            if not hasattr(model, method):
                raise ValueError(f"v4 TorchScript is missing exported method {method}")

    dataset = VideoDeblurJsonlDataset(
        str(args.manifest) if args.manifest is not None else None,
        clip_length=1,
        crop_size=0,
        augment=False,
        precompute_report=(
            str(args.precompute_report)
            if args.precompute_report is not None
            else None
        ),
    )
    if input_domain == "evssm" or checkpoint_teacher["storage"] != "none":
        if dataset.teacher_provenance is None:
            raise ValueError(
                "an EVSSM-trained/distilled checkpoint requires --precompute-report; "
                "unverified teacher paths cannot establish the training EVSSM checkpoint"
            )
        if (
            dataset.teacher_provenance["evssm_checkpoint_sha256"]
            != checkpoint_teacher["evssm_checkpoint_sha256"]
        ):
            raise ValueError(
                "evaluation teacher EVSSM SHA-256 differs from training provenance"
            )
    rows = []
    visual_records = []
    alignment_samples: list[dict[str, torch.Tensor]] = []
    expected_transition_count = sum(
        max(0, len(sequence.blurry) - 1) for sequence in dataset.sequences
    )
    for sequence in dataset.sequences:
        history: deque[torch.Tensor] = deque(maxlen=args.history)
        previous = None
        for frame_index, (blurry_path, sharp_path) in enumerate(
            zip(sequence.blurry, sequence.sharp)
        ):
            blurry = _tensor(blurry_path, device)
            sharp = _tensor(sharp_path, device)
            if sequence.teacher is None:
                if input_domain == "evssm":
                    raise ValueError("EVSSM-domain evaluation requires teacher paths")
                evssm = blurry
            else:
                evssm = _tensor(sequence.teacher[frame_index], device)
            model_frame = evssm if input_domain == "evssm" else blurry
            history.append(model_frame)
            padded = list(history)
            while len(padded) < args.history:
                padded.insert(0, padded[0])
            sequence_tensor = torch.stack(padded).unsqueeze(0)
            repeat_current_tensor = model_frame.unsqueeze(0).repeat(
                args.history, 1, 1, 1
            ).unsqueeze(0)
            alignment_row = None
            causal_alignment_disabled = None
            with torch.no_grad():
                if is_v4:
                    diagnostics = model.forward_sequence_with_motion_diagnostics(
                        sequence_tensor
                    )
                    if not isinstance(diagnostics, tuple) or len(diagnostics) != 4:
                        raise RuntimeError(
                            "v4 motion diagnostic method must return four tensors"
                        )
                    (
                        diagnostic_sequence,
                        adjacent_flow,
                        adjacent_confidence,
                        adjacent_valid,
                    ) = diagnostics
                    if diagnostic_sequence.shape != sequence_tensor.shape:
                        raise RuntimeError(
                            "v4 diagnostic prediction must match the input BTCHW shape"
                        )
                    expected_transitions = args.history - 1
                    if (
                        adjacent_flow.dim() != 5
                        or adjacent_flow.shape[0] != 1
                        or adjacent_flow.shape[1] != expected_transitions
                        or adjacent_flow.shape[2] != 2
                        or adjacent_confidence.shape[:3]
                        != (1, expected_transitions, 1)
                        or adjacent_valid.shape != adjacent_confidence.shape
                    ):
                        raise RuntimeError(
                            "v4 diagnostic adjacent tensors violate the B,T-1,C,h4,w4 contract"
                        )
                    disabled_sequence = model.forward_sequence_alignment_disabled(
                        sequence_tensor
                    )
                    if disabled_sequence.shape != diagnostic_sequence.shape:
                        raise RuntimeError(
                            "v4 alignment-disabled output must match prediction_sequence"
                        )
                    causal = diagnostic_sequence[0, -1].clamp(0.0, 1.0)
                    causal_alignment_disabled = disabled_sequence[0, -1].clamp(
                        0.0, 1.0
                    )
                    if frame_index > 0:
                        alignment_row, transition_samples = _alignment_transition(
                            adjacent_flow[0, -1],
                            adjacent_confidence[0, -1],
                            adjacent_valid[0, -1],
                            input_height=int(model_frame.shape[-2]),
                            input_width=int(model_frame.shape[-1]),
                            from_frame_index=frame_index - 1,
                            to_frame_index=frame_index,
                        )
                        alignment_samples.append(transition_samples)
                else:
                    causal = model(sequence_tensor)[0].clamp(0.0, 1.0)
                causal_repeat_current = model(repeat_current_tensor)[0].clamp(
                    0.0, 1.0
                )
            blurry_laplacian = _laplacian_variance(blurry)
            evssm_laplacian = _laplacian_variance(evssm)
            causal_laplacian = _laplacian_variance(causal)
            causal_vs_evssm_gain = (
                causal_laplacian - evssm_laplacian
            ) / max(evssm_laplacian, 1.0e-12)
            causal_vs_blurry_gain = (
                causal_laplacian - blurry_laplacian
            ) / max(blurry_laplacian, 1.0e-12)
            row = {
                "sequence": sequence.name,
                "frame_index": frame_index,
                "history_stage": (
                    "prefix" if frame_index < args.history - 1 else "steady_state"
                ),
                "blurry_path": str(blurry_path),
                "sharp_path": str(sharp_path),
                "blurry": _metrics(blurry, sharp),
                "evssm": _metrics(evssm, sharp, lpips_metric),
                "causal": _metrics(causal, sharp, lpips_metric),
                "causal_repeat_current": _metrics(
                    causal_repeat_current, sharp, lpips_metric
                ),
                "runtime_gate_proxy": {
                    "blurry_laplacian_variance": blurry_laplacian,
                    "evssm_laplacian_variance": evssm_laplacian,
                    "causal_laplacian_variance": causal_laplacian,
                    "causal_vs_evssm_gain": causal_vs_evssm_gain,
                    "causal_vs_blurry_gain": causal_vs_blurry_gain,
                    "passes_default_gate": bool(
                        causal_vs_evssm_gain >= 0.0
                        and causal_vs_blurry_gain >= 0.02
                    ),
                },
            }
            if is_v4:
                if causal_alignment_disabled is None:
                    raise RuntimeError("v4 alignment-disabled control was not produced")
                row["causal_alignment_disabled"] = _metrics(
                    causal_alignment_disabled, sharp, lpips_metric
                )
                row["motion_alignment"] = alignment_row
            sources = {
                "blurry": blurry,
                "evssm": evssm,
                "causal": causal,
                "causal_repeat_current": causal_repeat_current,
            }
            if is_v4:
                sources["causal_alignment_disabled"] = causal_alignment_disabled
            row["temporal"] = (
                {
                    source: _temporal_metrics(
                        image,
                        previous[source],
                        sharp,
                        previous["sharp"],
                    )
                    for source, image in sources.items()
                }
                if previous is not None
                else None
            )
            rows.append(row)
            visual_records.append((row, blurry, evssm, causal, sharp))
            previous = {**sources, "sharp": sharp}

    all_quality = _quality_breakdown(rows)
    prefix_quality = _quality_breakdown(
        [row for row in rows if row["history_stage"] == "prefix"]
    )
    steady_quality = _quality_breakdown(
        [row for row in rows if row["history_stage"] == "steady_state"]
    )
    gate_rows = [row["runtime_gate_proxy"] for row in rows]
    summary = {
        "schema": (
            "unblur_slam.causal_video_deblur_smoke_eval.v4"
            if is_v4
            else "unblur_slam.causal_video_deblur_smoke_eval.v3"
        ),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "evaluated_artifact_sha256": _sha256_file(
            args.checkpoint.expanduser().resolve()
        ),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "manifest": str(dataset.manifest),
        "teacher_provenance": checkpoint_teacher,
        "input_domain": input_domain,
        "history": args.history,
        "lpips_computed": bool(args.compute_lpips),
        "lpips_protocol": {
            "implementation": "torchmetrics.image.lpip",
            "network": "alex",
            "normalize_input_0_1": True,
            "per_frame_state_reset": True,
        },
        "frame_count": len(rows),
        "temporal_pair_count": sum(row["temporal"] is not None for row in rows),
        "mean": all_quality["mean"],
        "causal_minus_evssm": all_quality["causal_minus_evssm"],
        "history_ablation": {
            "protocol": (
                "causal_repeat_current supplies the current EVSSM frame at every "
                "history position; a positive full-minus-repeat PSNR isolates "
                "useful past-frame context rather than spatial-only correction"
            ),
            "causal_minus_repeat_current": all_quality[
                "causal_minus_repeat_current"
            ],
        },
        "history_stage_breakdown": {
            "prefix": prefix_quality,
            "steady_state": steady_quality,
        },
        "steady_psnr_db": steady_quality["mean"]["causal"]["psnr"],
        "steady_frame_count": steady_quality["frame_count"],
        "runtime_gate_proxy": {
            "protocol": (
                "exact runtime Laplacian variance proxy; default gate requires "
                "causal-vs-EVSSM >= 0 and causal-vs-blurry >= 0.02"
            ),
            "pass_count": sum(bool(row["passes_default_gate"]) for row in gate_rows),
            "pass_ratio": float(
                np.mean([bool(row["passes_default_gate"]) for row in gate_rows])
            ) if gate_rows else float("nan"),
            "mean_causal_vs_evssm_gain": float(
                np.mean([row["causal_vs_evssm_gain"] for row in gate_rows])
            ) if gate_rows else float("nan"),
            "mean_causal_vs_blurry_gain": float(
                np.mean([row["causal_vs_blurry_gain"] for row in gate_rows])
            ) if gate_rows else float("nan"),
        },
        "temporal": {
            "protocol": {
                "optical_flow_warp_used": False,
                "adjacent_change_l1": (
                    "mean absolute adjacent-frame change; includes true scene motion"
                ),
                "gt_difference_error_l1_not_warp": (
                    "L1 between predicted and GT adjacent-frame differences; "
                    "this is difference-based stability, not GT/flow warp error"
                ),
            },
            "mean": {
                source: {
                    metric: _mean_temporal(rows, source, metric)
                    for metric in (
                        "adjacent_change_l1",
                        "gt_difference_error_l1_not_warp",
                    )
                }
                for source in (
                    "blurry",
                    "evssm",
                    "causal",
                    "causal_repeat_current",
                )
            },
        },
        "frames": rows,
    }
    summary["temporal"]["causal_minus_evssm"] = {
        metric: (
            summary["temporal"]["mean"]["causal"][metric]
            - summary["temporal"]["mean"]["evssm"][metric]
        )
        for metric in (
            "adjacent_change_l1",
            "gt_difference_error_l1_not_warp",
        )
    }
    if is_v4:
        max_component_quarter_pixels = 2.0 * float(
            config["motion_alignment"]["radius"]
        )
        motion_summary = _alignment_summary(
            alignment_samples,
            expected_transition_count=expected_transition_count,
            max_component_quarter_pixels=max_component_quarter_pixels,
        )
        prefix_rows = [row for row in rows if row["history_stage"] == "prefix"]
        steady_rows = [
            row for row in rows if row["history_stage"] == "steady_state"
        ]
        disabled_temporal_mean = {
            metric: _mean_temporal(
                rows, "causal_alignment_disabled", metric
            )
            for metric in (
                "adjacent_change_l1",
                "gt_difference_error_l1_not_warp",
            )
        }
        causal_minus_disabled_temporal = {
            metric: (
                summary["temporal"]["mean"]["causal"][metric]
                - disabled_temporal_mean[metric]
            )
            for metric in disabled_temporal_mean
        }
        disabled_control = {
            "protocol": (
                "the identical v4 weights and history are evaluated through "
                "forward_sequence_alignment_disabled, which bypasses motion alignment"
            ),
            "quality": {
                "all": _alignment_disabled_quality(rows),
                "prefix": _alignment_disabled_quality(prefix_rows),
                "steady_state": _alignment_disabled_quality(steady_rows),
            },
            "temporal": {
                "pair_count": summary["temporal_pair_count"],
                "alignment_disabled_mean": disabled_temporal_mean,
                "causal_minus_alignment_disabled": (
                    causal_minus_disabled_temporal
                ),
            },
        }
        repeat_control = {
            "protocol": summary["history_ablation"]["protocol"],
            "quality_all_causal_minus_repeat_current": summary[
                "history_ablation"
            ]["causal_minus_repeat_current"],
            "quality_steady_causal_minus_repeat_current": (
                summary["history_stage_breakdown"]["steady_state"][
                    "causal_minus_repeat_current"
                ]
            ),
            "temporal_causal_minus_repeat_current": {
                metric: (
                    summary["temporal"]["mean"]["causal"][metric]
                    - summary["temporal"]["mean"]["causal_repeat_current"][
                        metric
                    ]
                )
                for metric in (
                    "adjacent_change_l1",
                    "gt_difference_error_l1_not_warp",
                )
            },
        }
        motion_summary["controls"] = {
            "repeat_current": repeat_control,
            "alignment_disabled": disabled_control,
        }
        summary["transition_count"] = len(alignment_samples)
        summary["alignment_disabled_control"] = disabled_control
        summary["alignment_diagnostics"] = motion_summary
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    ranked = sorted(
        visual_records,
        key=lambda item: item[0]["causal"]["psnr"] - item[0]["evssm"]["psnr"],
    )
    selected = ranked[: max(1, args.max_visuals // 2)] + ranked[-max(1, args.max_visuals // 2) :]
    for rank, (row, blurry, evssm, causal, sharp) in enumerate(selected):
        delta = row["causal"]["psnr"] - row["evssm"]["psnr"]
        _montage(
            output_dir / "visuals" / f"{rank:03d}_{row['sequence']}_{row['frame_index']:06d}.png",
            [("Blurry", blurry), ("EVSSM", evssm), ("Causal EVSSM", causal), ("Sharp GT", sharp)],
            f"PSNR: {row['blurry']['psnr']:.3f} | {row['evssm']['psnr']:.3f} | {row['causal']['psnr']:.3f}; causal-EVSSM {delta:+.3f} dB",
        )
    print(
        json.dumps(
            {
                "mean": summary["mean"],
                "causal_minus_evssm": summary["causal_minus_evssm"],
                "history_ablation": summary["history_ablation"],
                "runtime_gate_proxy": summary["runtime_gate_proxy"],
                "output": str(output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
