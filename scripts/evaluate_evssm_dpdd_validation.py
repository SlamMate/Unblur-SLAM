#!/usr/bin/env python3
"""Evaluate the pinned official Unblur-SLAM EVSSM on DPDD validation pairs.

The formal entry point accepts only the canonical, flat DPDD validation
manifest.  Every input remains 16-bit RGB until normalization, every image is
content-addressed, and the sealed test split is rejected before a path is
resolved or opened.  Outputs are published through a fresh staging directory;
an existing destination is never reused or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from skimage.metrics import structural_similarity
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.precompute_framecrafter_evssm import build_evssm_inference  # noqa: E402
from scripts.bsd_dpdd_contract import (  # noqa: E402
    load_contract,
    validate_protocol,
)
from scripts.bsd_dpdd_runtime import require_gpu1_a6000  # noqa: E402


SCHEMA = "unblur_slam.official_evssm_dpdd_validation.v1"
PAIR_SCHEMA = "unblur_slam.dpdd_hf_canonical_pair.v1"
DATASET_MANIFEST_SCHEMA = "unblur_slam.dpdd_hf_png16_materialization.v1"
DPDD_REPOSITORY = "JacobLinCool/DPDD"
DPDD_REVISION = "52e4035a045ea1763313b9ce2b27cf2e620cfc30"
DPDD_CONFIG = "combined"
OFFICIAL_EVSSM_CHECKPOINT = Path(
    "/srv/szha0669/unblur-slam/pretrained/net_g_latest_batch_8_no_NYU.pth"
)
OFFICIAL_EVSSM_SHA256 = (
    "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
)
FORMAL_VALIDATION_COUNT = 74
FORMAL_WARMUP_STEPS = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LPIPS_PROTOCOL = {
    "implementation": "torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity",
    "torchmetrics_version": "1.9.0",
    "network": "alex",
    "normalize_input_0_1": True,
    "reduction": "one_value_per_image_then_arithmetic_mean",
    "alexnet_backbone": {
        "path": "/home/szha0669/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
        "sha256": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    },
    "lpips_linear_weights": {
        "path": "/srv/szha0669/unblur-slam/env/lib/python3.10/site-packages/torchmetrics/functional/image/lpips_models/alex.pth",
        "sha256": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
    },
}


class EvaluationError(RuntimeError):
    """Raised when the DPDD/EVSSM evaluation contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise EvaluationError(f"{label} must be 64 lowercase hexadecimal characters")
    return normalized


def _relative_file(value: Any, root: Path, *, label: str) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        raise EvaluationError(f"{label} must be relative to the dataset root")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationError(f"{label} escapes the dataset root") from error
    if not resolved.is_file():
        raise EvaluationError(f"{label} does not exist: {resolved}")
    return resolved


def read_rgb16_png(path: Path) -> np.ndarray:
    """Decode an RGB16 PNG to HWC float32 without an 8-bit round trip."""

    try:
        import cv2
    except Exception as error:
        raise EvaluationError("formal RGB16 decoding requires OpenCV") from error
    array_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array_bgr is None:
        raise EvaluationError(f"OpenCV could not decode PNG: {path}")
    if array_bgr.dtype != np.uint16 or array_bgr.ndim != 3 or array_bgr.shape[2] != 3:
        raise EvaluationError(
            "DPDD canonical image must be a three-channel uint16 PNG; "
            f"got dtype={array_bgr.dtype}, shape={array_bgr.shape}: {path}"
        )
    array_rgb = np.ascontiguousarray(array_bgr[:, :, ::-1])
    return array_rgb.astype(np.float32) / 65535.0


def load_validation_dataset_contract(
    dataset_manifest: Path | str,
    *,
    expected_dataset_manifest_sha256: str,
    validation_manifest: Path | str,
    expected_validation_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind validation JSONL to the pinned PNG16 materialization provenance."""

    dataset_path = Path(dataset_manifest).expanduser().resolve()
    if not dataset_path.is_file():
        raise EvaluationError(f"dataset manifest does not exist: {dataset_path}")
    expected_dataset_sha = _sha256(
        expected_dataset_manifest_sha256, label="dataset manifest SHA-256"
    )
    if sha256_file(dataset_path) != expected_dataset_sha:
        raise EvaluationError("dataset manifest SHA-256 mismatch")
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationError("dataset manifest is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise EvaluationError("dataset manifest must be a JSON object")
    fixed = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "repository": DPDD_REPOSITORY,
        "revision": DPDD_REVISION,
        "config": DPDD_CONFIG,
        "splits": {"train": 350, "validation": 74},
    }
    mismatches = {
        key: (payload.get(key), wanted)
        for key, wanted in fixed.items()
        if payload.get(key) != wanted
    }
    if mismatches:
        raise EvaluationError(f"dataset provenance mismatch: {mismatches}")

    distribution = payload.get("distribution")
    if not isinstance(distribution, Mapping):
        raise EvaluationError("dataset distribution provenance is missing")
    if str(distribution.get("dataset_card_declared_license", "")).lower() != "mit":
        raise EvaluationError("dataset-card license disclosure changed")
    license_warning = distribution.get("license_scope_warning")
    if not isinstance(license_warning, str) or not license_warning.strip():
        raise EvaluationError("dataset mirror license-scope warning is missing")

    disclosure = payload.get("test_disclosure")
    if not isinstance(disclosure, Mapping):
        raise EvaluationError("sealed-test disclosure is missing")
    expected_disclosure = {
        "metadata_pristine": False,
        "metadata_exposure": (
            "filenames_lfs_oids_sizes_split_aggregate_row0_url_and_manifest_text_"
            "seen_before_freeze"
        ),
        "images_decoded": False,
        "pixels_opened": False,
        "metrics_opened": False,
        "requests_made_by_this_materializer": 0,
        "split_supported_by_this_materializer": False,
    }
    disclosure_mismatches = {
        key: (disclosure.get(key), wanted)
        for key, wanted in expected_disclosure.items()
        if disclosure.get(key) != wanted
    }
    if disclosure_mismatches:
        raise EvaluationError(f"sealed-test disclosure mismatch: {disclosure_mismatches}")

    canonical = payload.get("canonical_manifests")
    if not isinstance(canonical, Mapping):
        raise EvaluationError("canonical manifest index is missing")
    validation = canonical.get("validation")
    if not isinstance(validation, Mapping):
        raise EvaluationError("canonical validation manifest binding is missing")
    expected_validation_sha = _sha256(
        expected_validation_manifest_sha256, label="validation manifest SHA-256"
    )
    relative = Path(str(validation.get("path", "")))
    if relative.is_absolute():
        raise EvaluationError("canonical validation path must be relative")
    dataset_root = dataset_path.parent.resolve()
    resolved = (dataset_root / relative).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as error:
        raise EvaluationError("canonical validation path escapes dataset root") from error
    wanted_validation = {
        "schema": PAIR_SCHEMA,
        "rows": FORMAL_VALIDATION_COUNT,
        "paths_relative_to": "dataset_root",
        "sha256": expected_validation_sha,
    }
    validation_mismatches = {
        key: (validation.get(key), wanted)
        for key, wanted in wanted_validation.items()
        if validation.get(key) != wanted
    }
    if validation_mismatches:
        raise EvaluationError(
            f"canonical validation manifest contract mismatch: {validation_mismatches}"
        )
    supplied_validation = Path(validation_manifest).expanduser().resolve()
    if resolved != supplied_validation:
        raise EvaluationError("--manifest is not the materializer validation manifest")
    if not resolved.is_file() or sha256_file(resolved) != expected_validation_sha:
        raise EvaluationError("canonical validation manifest content mismatch")
    return {
        "path": str(dataset_path),
        "sha256": expected_dataset_sha,
        "schema": DATASET_MANIFEST_SCHEMA,
        "repository": DPDD_REPOSITORY,
        "revision": DPDD_REVISION,
        "config": DPDD_CONFIG,
        "dataset_card_declared_license": "mit",
        "license_scope_warning": license_warning,
        "test_disclosure": dict(disclosure),
        "canonical_validation_manifest": {
            "path": str(resolved),
            "sha256": expected_validation_sha,
            "rows": FORMAL_VALIDATION_COUNT,
        },
    }


def load_validation_manifest(
    manifest: Path | str,
    *,
    data_root: Path | str | None = None,
    expected_count: int = FORMAL_VALIDATION_COUNT,
) -> list[dict[str, Any]]:
    """Load one validation-only manifest and verify every referenced byte."""

    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise EvaluationError(f"manifest does not exist: {manifest_path}")
    root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else manifest_path.parent
    )
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    source_paths: set[Path] = set()
    target_paths: set[Path] = set()
    source_hashes: set[str] = set()
    target_hashes: set[str] = set()
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"invalid JSON at line {line_number}") from error
        if not isinstance(payload, Mapping):
            raise EvaluationError(f"manifest line {line_number} is not an object")
        if payload.get("schema") != PAIR_SCHEMA:
            raise EvaluationError(f"manifest line {line_number} has the wrong schema")
        split = str(payload.get("split", "")).strip().lower()
        if split == "test":
            raise EvaluationError("DPDD test split is sealed and cannot be evaluated")
        if split not in {"validation", "val"}:
            raise EvaluationError("formal EVSSM evaluation accepts validation only")
        required = {"name", "defocus", "sharp", "source_sha256", "target_sha256"}
        missing = sorted(required - set(payload))
        if missing:
            raise EvaluationError(f"manifest line {line_number} is missing {missing}")
        name = str(payload["name"])
        if not name or name in names:
            raise EvaluationError(f"invalid or duplicate pair name {name!r}")
        names.add(name)
        source_sha = _sha256(payload["source_sha256"], label="source_sha256")
        target_sha = _sha256(payload["target_sha256"], label="target_sha256")
        if source_sha == target_sha:
            raise EvaluationError(f"pair {name!r} has identical defocus/sharp content")
        source = _relative_file(payload["defocus"], root, label="defocus path")
        target = _relative_file(payload["sharp"], root, label="sharp path")
        if source == target:
            raise EvaluationError(f"pair {name!r} reuses one path for input and target")
        if sha256_file(source) != source_sha or sha256_file(target) != target_sha:
            raise EvaluationError(f"pair {name!r} file SHA-256 does not match manifest")
        if source in source_paths or target in target_paths:
            raise EvaluationError("duplicate path within DPDD validation manifest")
        if source_sha in source_hashes or target_sha in target_hashes:
            raise EvaluationError("duplicate content within DPDD validation manifest")
        source_paths.add(source)
        target_paths.add(target)
        source_hashes.add(source_sha)
        target_hashes.add(target_sha)
        rows.append(
            {
                "name": name,
                "source": source,
                "target": target,
                "source_sha256": source_sha,
                "target_sha256": target_sha,
            }
        )
    if len(rows) != int(expected_count):
        raise EvaluationError(
            f"validation pair count mismatch: expected {expected_count}, found {len(rows)}"
        )
    if source_paths & target_paths or source_hashes & target_hashes:
        raise EvaluationError("defocus and sharp validation inventories overlap")
    return rows


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 3:
        raise EvaluationError("metrics require matching HWC RGB arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise EvaluationError("metrics received NaN or Inf")
    prediction = np.clip(prediction, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    mse = float(np.mean((prediction - target) ** 2, dtype=np.float64))
    minimum_side = min(prediction.shape[:2])
    if minimum_side < 3:
        raise EvaluationError("SSIM requires image sides of at least three pixels")
    window = min(7, minimum_side if minimum_side % 2 else minimum_side - 1)
    return {
        "psnr": -10.0 * math.log10(max(mse, 1.0e-12)),
        "ssim": float(
            structural_similarity(
                prediction,
                target,
                data_range=1.0,
                channel_axis=2,
                win_size=window,
            )
        ),
        "l1": float(np.mean(np.abs(prediction - target), dtype=np.float64)),
    }


def verify_lpips_protocol(
    protocol: Mapping[str, Any] = LPIPS_PROTOCOL,
    *,
    installed_version: str | None = None,
) -> None:
    """Fail closed unless the exact shared TURTLE/EVSSM LPIPS stack is present."""

    if installed_version is None:
        try:
            import torchmetrics
        except Exception as error:
            raise EvaluationError("formal evaluation requires torchmetrics LPIPS") from error
        installed_version = str(torchmetrics.__version__)
    if installed_version != protocol.get("torchmetrics_version"):
        raise EvaluationError("formal LPIPS torchmetrics version changed")
    for label in ("alexnet_backbone", "lpips_linear_weights"):
        artifact = protocol.get(label)
        if not isinstance(artifact, Mapping):
            raise EvaluationError(f"formal LPIPS {label} contract is missing")
        path = Path(str(artifact.get("path", ""))).expanduser().resolve()
        expected = _sha256(artifact.get("sha256"), label=f"LPIPS {label} SHA-256")
        if not path.is_file() or sha256_file(path) != expected:
            raise EvaluationError(f"formal LPIPS {label} artifact changed")


def build_lpips(device: torch.device) -> Callable[[np.ndarray, np.ndarray], float]:
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except Exception as error:
        raise EvaluationError("formal evaluation requires torchmetrics LPIPS") from error
    verify_lpips_protocol()
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    metric.eval().requires_grad_(False)

    @torch.no_grad()
    def evaluate(prediction: np.ndarray, target: np.ndarray) -> float:
        tensors = []
        for value in (prediction, target):
            tensor = torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))
            tensors.append(tensor.unsqueeze(0).float().clamp(0.0, 1.0).to(device))
        result = float(metric(tensors[0], tensors[1]).detach().float().mean().item())
        metric.reset()
        if not math.isfinite(result):
            raise EvaluationError("LPIPS returned a non-finite value")
        return result

    return evaluate


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    infer: Callable[[np.ndarray, float], np.ndarray],
    device: torch.device,
    lpips: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, Any]:
    if not rows:
        raise EvaluationError("cannot evaluate zero pairs")
    # One fixed, target-independent call initializes CUDA kernels and allocator
    # state. Its output and latency are discarded, and it never sees sharp RGB.
    warmup_raw = read_rgb16_png(Path(rows[0]["source"]))
    _sync(device)
    warmup_output = np.asarray(infer(warmup_raw, 0.0), dtype=np.float32)
    _sync(device)
    if warmup_output.shape != warmup_raw.shape or not np.isfinite(warmup_output).all():
        raise EvaluationError("EVSSM warm-up returned an invalid image")
    # Pass 1: a contiguous timing-only traversal.  Only defocus inputs are
    # opened; targets, LPIPS and all quality metrics remain absent.
    latencies: list[float] = []
    for index, row in enumerate(rows):
        raw = read_rgb16_png(Path(row["source"]))
        _sync(device)
        started = time.perf_counter()
        restored = np.asarray(infer(raw, float(index)), dtype=np.float32)
        _sync(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if restored.shape != raw.shape or not np.isfinite(restored).all():
            raise EvaluationError("EVSSM timing pass returned an invalid image")
        latencies.append(float(latency_ms))

    # Pass 2: independent, wholly unmeasured raw/model quality computation.
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        raw = read_rgb16_png(Path(row["source"]))
        sharp = read_rgb16_png(Path(row["target"]))
        if raw.shape != sharp.shape:
            raise EvaluationError(f"pair {row['name']!r} is not pixel aligned")
        restored = np.asarray(infer(raw, float(index)), dtype=np.float32)
        if restored.shape != raw.shape or not np.isfinite(restored).all():
            raise EvaluationError("EVSSM quality pass returned an invalid image")
        raw_metric = image_metrics(raw, sharp)
        evssm_metric = image_metrics(restored, sharp)
        raw_metric["lpips"] = float(lpips(raw, sharp))
        evssm_metric["lpips"] = float(lpips(restored, sharp))
        results.append(
            {
                "index": index,
                "name": row["name"],
                "defocus_path": str(row["source"]),
                "sharp_path": str(row["target"]),
                "source_sha256": row["source_sha256"],
                "target_sha256": row["target_sha256"],
                "height": int(raw.shape[0]),
                "width": int(raw.shape[1]),
                "raw": raw_metric,
                "evssm": evssm_metric,
                "evssm_latency_ms": latencies[index],
            }
        )
    metrics = ("psnr", "ssim", "lpips", "l1")
    mean = {
        arm: {
            metric: float(np.mean([item[arm][metric] for item in results]))
            for metric in metrics
        }
        for arm in ("raw", "evssm")
    }
    latency_array = np.asarray(latencies, dtype=np.float64)
    return {
        "pair_count": len(results),
        "warmup": {
            "steps": FORMAL_WARMUP_STEPS,
            "input": "first_validation_defocus_only",
            "target_or_metric_used": False,
            "output_and_latency_discarded": True,
            "pre_and_post_cuda_synchronization": True,
        },
        "mean": mean,
        "raw_baseline": {
            "registration": {
                "per_image_rows_present": True,
                "identity_fields": [
                    "name",
                    "defocus_path",
                    "sharp_path",
                    "source_sha256",
                    "target_sha256",
                ],
                "decode": "direct_RGB16_float32_no_quantized_prediction_roundtrip",
            },
            "summary": mean["raw"],
        },
        "evssm_minus_raw": {
            metric: mean["evssm"][metric] - mean["raw"][metric] for metric in metrics
        },
        "latency_ms": {
            "mean": float(np.mean(latency_array)),
            "median": float(np.percentile(latency_array, 50)),
            "p95": float(np.percentile(latency_array, 95)),
            "max": float(np.max(latency_array)),
            "frames": len(latencies),
        },
        "pass_separation": {
            "timing_pass": {
                "stateless_model_steps": len(results),
                "sharp_target_images_opened": False,
                "metrics_or_lpips_computed": False,
            },
            "quality_pass": {
                "stateless_model_steps": len(results),
                "timed_model_steps": 0,
            },
            "passes_are_distinct_complete_dataset_traversals": True,
            "forward_accounting_excluding_warmup": {
                "timing_only_model_steps": len(results),
                "quality_model_steps": len(results),
                "combined_model_steps": 2 * len(results),
            },
        },
        "pairs": results,
    }


def verify_official_checkpoint(path: Path | str) -> tuple[Path, str]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise EvaluationError(f"EVSSM checkpoint does not exist: {checkpoint}")
    digest = sha256_file(checkpoint)
    if digest != OFFICIAL_EVSSM_SHA256:
        raise EvaluationError("EVSSM checkpoint is not the pinned official Unblur-SLAM artifact")
    return checkpoint, digest


def write_report_new(output_dir: Path | str, payload: Mapping[str, Any]) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise EvaluationError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    completed = False
    try:
        report = stage / "metrics.json"
        with report.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            raise EvaluationError(f"output appeared before publication: {output}")
        os.rename(stage, output)
        completed = True
        return output / "metrics.json"
    finally:
        if not completed:
            shutil.rmtree(stage, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL_EVSSM_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lpips-device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract_path, contract, contract_sha = load_contract(
        args.contract, expected_sha256=args.expected_contract_sha256
    )
    validate_protocol(contract, allow_template=False)
    runtime = require_gpu1_a6000(args.device)
    dpdd_contract = contract["data"]["dpdd"]
    expected_arguments = {
        "manifest": (args.manifest.expanduser().resolve(), Path(dpdd_contract["validation_manifest"]).expanduser().resolve()),
        "manifest_sha256": (args.manifest_sha256.lower(), str(dpdd_contract["validation_manifest_sha256"])),
        "dataset_manifest": (args.dataset_manifest.expanduser().resolve(), Path(dpdd_contract["dataset_manifest"]).expanduser().resolve()),
        "dataset_manifest_sha256": (args.dataset_manifest_sha256.lower(), str(dpdd_contract["dataset_manifest_sha256"])),
        "data_root": (
            None if args.data_root is None else args.data_root.expanduser().resolve(),
            Path(dpdd_contract["root"]).expanduser().resolve(),
        ),
        "checkpoint": (
            args.checkpoint.expanduser().resolve(),
            Path(contract["models"]["evssm_E"]["checkpoint"]).expanduser().resolve(),
        ),
    }
    mismatches = {
        key: (str(actual), str(wanted))
        for key, (actual, wanted) in expected_arguments.items()
        if actual != wanted
    }
    if mismatches:
        raise EvaluationError(f"CLI inputs differ from bound contract: {mismatches}")
    manifest = args.manifest.expanduser().resolve()
    expected_manifest_sha = _sha256(args.manifest_sha256, label="manifest SHA-256")
    if not manifest.is_file() or sha256_file(manifest) != expected_manifest_sha:
        raise EvaluationError("validation manifest SHA-256 mismatch")
    dataset_provenance = load_validation_dataset_contract(
        args.dataset_manifest,
        expected_dataset_manifest_sha256=args.dataset_manifest_sha256,
        validation_manifest=manifest,
        expected_validation_manifest_sha256=expected_manifest_sha,
    )
    checkpoint, checkpoint_sha = verify_official_checkpoint(args.checkpoint)
    if checkpoint_sha != contract["models"]["evssm_E"]["checkpoint_sha256"]:
        raise EvaluationError("EVSSM checkpoint SHA-256 differs from bound contract")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise EvaluationError("formal official EVSSM evaluation requires CUDA")
    rows = load_validation_manifest(manifest, data_root=args.data_root)
    infer = build_evssm_inference(checkpoint, str(device))
    lpips = build_lpips(torch.device(args.lpips_device))
    evaluated = evaluate_rows(rows, infer=infer, device=device, lpips=lpips)
    if (
        evaluated.get("pair_count") != FORMAL_VALIDATION_COUNT
        or evaluated.get("latency_ms", {}).get("frames") != FORMAL_VALIDATION_COUNT
        or evaluated.get("pass_separation", {}).get(
            "forward_accounting_excluding_warmup"
        )
        != {
            "timing_only_model_steps": 74,
            "quality_model_steps": 74,
            "combined_model_steps": 148,
        }
    ):
        raise EvaluationError("EVSSM DPDD two-pass coverage/accounting changed")
    payload = {
        "schema": SCHEMA,
        "formal": True,
        "arm": "E",
        "interpretation": (
            "external stateless paired single-image defocus restoration reference; "
            "not a same-method arm and no temporal/history or SLAM claim"
        ),
        "protocol": {
            "contract": str(contract_path),
            "contract_sha256": contract_sha,
            "split": "validation",
            "expected_pair_count": FORMAL_VALIDATION_COUNT,
            "manifest": str(manifest),
            "manifest_sha256": expected_manifest_sha,
            "dataset_materialization": dataset_provenance,
            "decode": "cv2.IMREAD_UNCHANGED uint16 BGR->RGB then divide by 65535",
            "metrics": "per-image RGB PSNR/SSIM/AlexNet-LPIPS/L1 then arithmetic mean",
            "lpips": LPIPS_PROTOCOL,
            "latency": (
                "EVSSM model/backend call in a dedicated timing-only full pass with "
                "pre/post CUDA synchronization; excludes I/O, target access, quality "
                "forwards, metrics, LPIPS, reporting, and SLAM"
            ),
            "latency_warmup": (
                "one unmeasured call on the first validation defocus image; "
                "no sharp target or metric; output discarded"
            ),
            "test_pixels_opened": False,
            "test_metrics_computed": False,
            "raw_common_baseline_registered": True,
            "reference_role": "external_stateless_single_frame_EVSSM_reference",
            "claim_scope": "restoration_module_only",
            "slam_quality_or_speed_claim": False,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "device": str(device),
        "runtime_identity": dict(runtime),
        "compute_precision": {
            "model_and_input": "CUDA_FP32",
            "autocast": "disabled",
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "comparison_to_turtle_fp16": (
                "descriptive architecture-specific model-step latency only; "
                "precision is not matched"
            ),
        },
        "lpips_device": str(torch.device(args.lpips_device)),
        "results": evaluated,
    }
    report = write_report_new(args.output_dir, payload)
    print(json.dumps({"report": str(report), "mean": evaluated["mean"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as error:
        raise SystemExit(f"error: {error}") from error
