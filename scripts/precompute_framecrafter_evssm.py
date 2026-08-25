#!/usr/bin/env python3
"""Precompute audited EVSSM images for FrameCrafter conditioning.

The command deliberately runs EVSSM before the much larger FrameCrafter model
is loaded.  Every output is content-addressed in one metadata file so a later
FrameCrafter run can distinguish real EVSSM inference from the explicit
CPU-only identity backend used by contract tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framecrafter_pipeline import (  # noqa: E402
    laplacian_sharpness,
    read_rgb,
    validate_pose_source,
)


METADATA_SCHEMA = "unblur_slam.framecrafter_evssm_precompute.v1"
_FALSE_VALUES = {"0", "false", "no"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceFrame:
    source_index: int
    frame_id: str
    timestamp: float
    raw_path: Path
    pose_source: str


def sha256_file(path: Path | str) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with open(source, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_rgb_path(csv_path: Path, image_root: Path | None, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (image_root if image_root is not None else csv_path.parent) / path
    return path.resolve()


def load_source_frames(
    frames_csv: Path | str, image_root: Path | str | None = None
) -> tuple[list[SourceFrame], str]:
    """Load only the image/provenance fields needed for EVSSM inference.

    Unlike a generic image-list loader, this is intentionally strict about the
    pose provenance carried by the FrameCrafter CSV.  It never consumes a pose,
    but retaining and validating the provenance prevents deblur artifacts from
    later being associated with a GT-derived planning stream.
    """

    csv_path = Path(frames_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"frames CSV does not exist: {csv_path}")
    root = None if image_root is None else Path(image_root).expanduser().resolve()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or ())
    required = {"frame", "pose_source", "uses_ground_truth_pose"}
    missing = required - columns
    if missing:
        raise ValueError(
            "frames CSV is missing EVSSM provenance columns: "
            f"{sorted(missing)}"
        )
    if not rows:
        raise ValueError(f"frames CSV is empty: {csv_path}")

    frames: list[SourceFrame] = []
    observed_pose_source: str | None = None
    for position, row in enumerate(rows):
        uses_gt = str(row.get("uses_ground_truth_pose", "")).strip().lower()
        if uses_gt not in _FALSE_VALUES:
            raise ValueError(
                "every source row must explicitly declare "
                "uses_ground_truth_pose=false"
            )
        pose_source = validate_pose_source(row.get("pose_source", ""))
        if observed_pose_source is None:
            observed_pose_source = pose_source
        elif pose_source != observed_pose_source:
            raise ValueError("pose_source must be identical on every source row")

        source_index = int(str(row.get("index", "")).strip() or position)
        frame_id = str(row["frame"]).strip()
        if not frame_id:
            raise ValueError(f"empty frame id at CSV row {position + 2}")
        rgb_value = str(row.get("rgb_path", "")).strip() or frame_id
        raw_path = _resolve_rgb_path(csv_path, root, rgb_value)
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"source RGB for source_index={source_index} does not exist: {raw_path}"
            )
        timestamp_value = str(row.get("timestamp", "")).strip()
        timestamp = float(timestamp_value) if timestamp_value else float(source_index)
        if not math.isfinite(timestamp):
            raise ValueError(f"non-finite timestamp at source_index={source_index}")
        frames.append(
            SourceFrame(
                source_index=source_index,
                frame_id=frame_id,
                timestamp=timestamp,
                raw_path=raw_path,
                pose_source=pose_source,
            )
        )

    indices = [frame.source_index for frame in frames]
    if len(set(indices)) != len(indices):
        raise ValueError("source_index values in frames CSV must be unique")
    assert observed_pose_source is not None
    return frames, observed_pose_source


def parse_source_indices(values: Sequence[str], available: Sequence[int]) -> list[int]:
    """Parse ``all`` or a comma/space-separated ordered source-index subset."""

    tokens = [piece.strip() for value in values for piece in str(value).split(",")]
    tokens = [token for token in tokens if token]
    if not tokens or tokens == ["all"]:
        return list(available)
    if "all" in tokens:
        raise ValueError("--source-indices all cannot be combined with indices")
    requested = [int(token) for token in tokens]
    if len(set(requested)) != len(requested):
        raise ValueError("--source-indices contains duplicates")
    available_set = set(available)
    missing = [index for index in requested if index not in available_set]
    if missing:
        raise ValueError(f"requested source indices are absent from CSV: {missing}")
    requested_set = set(requested)
    # Preserve source-stream order, not arbitrary CLI order.
    return [index for index in available if index in requested_set]


def _atomic_save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"EVSSM output must be HWC RGB, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("EVSSM output contains non-finite values")
    encoded = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.png"
    try:
        Image.fromarray(encoded, mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _unwrap_checkpoint(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("EVSSM checkpoint must contain a state-dict mapping")
    for key in ("params_ema", "params", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            payload = candidate
            break
    if not payload or not all(isinstance(key, str) for key in payload):
        raise ValueError("EVSSM checkpoint has an unsupported state-dict structure")
    state = dict(payload)
    if all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    return state


def build_evssm_inference(
    checkpoint: Path, device: str
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Load the repository EVSSM checkpoint and return a one-frame callable."""

    try:
        import torch

        from src.deblur_backends import EVSSMBackend
        from thirdparty.EVSSM.models.EVSSM import EVSSM
    except Exception as error:
        raise RuntimeError(
            "EVSSM inference requires torch plus thirdparty/EVSSM dependencies"
        ) from error

    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise RuntimeError(
            "production EVSSM inference requires CUDA selective_scan kernels; "
            "use --test-only-identity only for CPU contract testing"
        )
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA EVSSM device is unavailable: {device}")
    model = EVSSM().to(target_device)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = _unwrap_checkpoint(payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    backend = EVSSMBackend(model, target_device)

    def infer(image: np.ndarray, timestamp: float) -> np.ndarray:
        contiguous = np.ascontiguousarray(image.transpose(2, 0, 1))
        tensor = torch.from_numpy(contiguous).unsqueeze(0)
        output = backend(tensor, timestamp=timestamp)
        result = output[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        return np.asarray(result, dtype=np.float32)

    return infer


def _identity_inference(image: np.ndarray, timestamp: float) -> np.ndarray:
    del timestamp
    return np.asarray(image, dtype=np.float32).copy()


def _validate_thresholds(
    min_sharpness_gain: float, min_consistency: float, min_confidence: float
) -> None:
    if not math.isfinite(min_sharpness_gain) or min_sharpness_gain < 0.0:
        raise ValueError("min_sharpness_gain must be finite and non-negative")
    for name, value in (
        ("min_consistency", min_consistency),
        ("min_confidence", min_confidence),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")


def precompute(
    *,
    frames_csv: Path | str,
    checkpoint: Path | str,
    output_dir: Path | str,
    source_indices: Sequence[str] = ("all",),
    image_root: Path | str | None = None,
    device: str = "cuda:0",
    min_sharpness_gain: float = 1.0,
    min_consistency: float = 0.70,
    min_confidence: float = 0.50,
    test_only_identity: bool = False,
    overwrite: bool = False,
) -> Path:
    """Run precomputation and return the absolute metadata JSON path."""

    _validate_thresholds(min_sharpness_gain, min_consistency, min_confidence)
    csv_path = Path(frames_csv).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    metadata_path = destination / "metadata.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"EVSSM checkpoint does not exist: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if not _SHA256_PATTERN.fullmatch(checkpoint_sha256):
        raise RuntimeError("internal checkpoint SHA-256 computation failed")

    frames, pose_source = load_source_frames(csv_path, image_root=image_root)
    selected_indices = parse_source_indices(
        source_indices, [frame.source_index for frame in frames]
    )
    selected_set = set(selected_indices)
    selected = [frame for frame in frames if frame.source_index in selected_set]
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"metadata already exists (pass --overwrite to replace): {metadata_path}"
        )
    output_paths = {
        frame.source_index: destination / "images" / f"source_{frame.source_index:06d}.png"
        for frame in selected
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "EVSSM output already exists (pass --overwrite to replace): "
            f"{existing[0]}"
        )

    infer = (
        _identity_inference
        if test_only_identity
        else build_evssm_inference(checkpoint_path, device)
    )
    thresholds = {
        "min_sharpness_gain": float(min_sharpness_gain),
        "min_image_consistency": float(min_consistency),
        "min_confidence": float(min_confidence),
    }
    implementation_path = Path(__file__).resolve()
    implementation_sha256 = sha256_file(implementation_path)
    cache_options = {
        "schema": METADATA_SCHEMA,
        "backend": "test_only_identity" if test_only_identity else "evssm",
        "device": "cpu_identity" if test_only_identity else str(device),
        "thresholds": thresholds,
        "output_encoding": "rgb_uint8_png",
        "implementation_sha256": implementation_sha256,
    }
    records: list[dict[str, object]] = []
    for frame in selected:
        raw = read_rgb(frame.raw_path)
        output = np.asarray(infer(raw, frame.timestamp), dtype=np.float32)
        if output.shape != raw.shape:
            raise ValueError(
                "EVSSM must preserve input resolution: "
                f"source_index={frame.source_index}, {output.shape} != {raw.shape}"
            )
        output_path = output_paths[frame.source_index]
        _atomic_save_png(output_path, output)
        # Measure the exact 8-bit PNG that FrameCrafter will consume.
        stored = read_rgb(output_path)
        raw_sharpness = laplacian_sharpness(raw)
        output_sharpness = laplacian_sharpness(stored)
        if raw_sharpness <= 1.0e-12 and output_sharpness <= 1.0e-12:
            sharpness_gain = 1.0
        else:
            sharpness_gain = output_sharpness / max(raw_sharpness, 1.0e-12)
        image_consistency = float(
            np.clip(1.0 - np.mean(np.abs(stored - raw)), 0.0, 1.0)
        )
        metric_confidence = float(
            math.sqrt(image_consistency * min(1.0, max(0.0, sharpness_gain)))
        )
        failures: list[str] = []
        if sharpness_gain < min_sharpness_gain:
            failures.append("sharpness_gain")
        if image_consistency < min_consistency:
            failures.append("image_consistency")
        if metric_confidence < min_confidence:
            failures.append("confidence")
        metric_gates_passed = not failures
        # The identity backend is plumbing-only.  Force confidence to zero and
        # acceptance false so its files cannot satisfy production EVSSM gates.
        confidence = 0.0 if test_only_identity else metric_confidence
        if test_only_identity:
            failures.append("test_only_identity")
        raw_path = frame.raw_path.resolve()
        output_path = output_path.resolve()
        raw_sha256 = sha256_file(raw_path)
        output_sha256 = sha256_file(output_path)
        cache_key = _sha256_json(
            {
                "raw_sha256": raw_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "options": cache_options,
            }
        )
        records.append(
            {
                "schema": METADATA_SCHEMA,
                "source_index": int(frame.source_index),
                "frame_id": frame.frame_id,
                "timestamp": float(frame.timestamp),
                "raw": {"path": str(raw_path), "sha256": raw_sha256},
                "output": {"path": str(output_path), "sha256": output_sha256},
                "raw_path": str(raw_path),
                "raw_sha256": raw_sha256,
                "output_path": str(output_path),
                "output_sha256": output_sha256,
                "raw_laplacian_sharpness": float(raw_sharpness),
                "output_laplacian_sharpness": float(output_sharpness),
                "sharpness_gain": float(sharpness_gain),
                "image_consistency": image_consistency,
                "confidence": confidence,
                "metric_confidence": metric_confidence,
                "metric_gates_passed": metric_gates_passed,
                "accepted": bool(metric_gates_passed and not test_only_identity),
                "failures": failures,
                "thresholds": thresholds,
                "checkpoint_sha256": checkpoint_sha256,
                "implementation_sha256": implementation_sha256,
                "cache_key": cache_key,
                "uses_ground_truth_pose": False,
                "pose_source": pose_source,
                "provider": "test_only_identity" if test_only_identity else "evssm",
                "test_only": bool(test_only_identity),
                "production_eligible": not test_only_identity,
            }
        )

    payload: dict[str, object] = {
        "schema": METADATA_SCHEMA,
        "artifact_class": (
            "test_only_identity" if test_only_identity else "evssm_precompute"
        ),
        "test_only": bool(test_only_identity),
        "production_eligible": not test_only_identity,
        "backend": "test_only_identity" if test_only_identity else "evssm",
        "device": "cpu_identity" if test_only_identity else str(device),
        "frames_csv": {
            "path": str(csv_path),
            "sha256": sha256_file(csv_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "implementation": {
            "path": str(implementation_path),
            "sha256": implementation_sha256,
        },
        "cache_options": cache_options,
        "uses_ground_truth_pose": False,
        "pose_source": pose_source,
        "selection": {
            "request": list(source_indices),
            "source_indices": selected_indices,
            "count": len(selected_indices),
        },
        "thresholds": thresholds,
        "metric_definitions": {
            "laplacian_sharpness": "mean_absolute_4_neighbour_laplacian_on_rgb_luma_0_1",
            "sharpness_gain": "output_laplacian_sharpness/raw_laplacian_sharpness",
            "image_consistency": "clip(1-mean_absolute_rgb_difference,0,1)",
            "confidence": "sqrt(image_consistency*min(1,sharpness_gain))",
        },
        "output_template": str((destination / "images" / "source_{source_index:06d}.png").resolve()),
        "accepted_count": sum(bool(record["accepted"]) for record in records),
        "frames": records,
    }
    _atomic_write_json(metadata_path, payload)
    return metadata_path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--source-indices",
        nargs="+",
        default=["all"],
        help="'all' (default), or comma/space-separated CSV source indices",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-sharpness-gain", type=float, default=1.0)
    parser.add_argument("--min-consistency", type=float, default=0.70)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument(
        "--test-only-identity",
        action="store_true",
        help=(
            "CPU plumbing backend; metadata is forcibly marked test_only, "
            "confidence=0, accepted=false"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata_path = precompute(
        frames_csv=args.frames_csv,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        source_indices=args.source_indices,
        image_root=args.image_root,
        device=args.device,
        min_sharpness_gain=args.min_sharpness_gain,
        min_consistency=args.min_consistency,
        min_confidence=args.min_confidence,
        test_only_identity=args.test_only_identity,
        overwrite=args.overwrite,
    )
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
