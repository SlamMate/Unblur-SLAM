#!/usr/bin/env python3
"""Paired single-image defocus evaluation for pinned TURTLE checkpoints.

Every manifest record is an independent image pair.  The official K/V state
is hard-reset before every image, so this evaluator measures current-frame
spatial restoration only.  It must not be used as evidence of useful history.

The formal protocol runs every checkpoint on the same ordered manifest with
CUDA FP16, deterministic right/bottom padding to a multiple of eight, and
reports per-image PSNR, RGB SSIM, AlexNet LPIPS, L1, and synchronized model-step
latency.  ``--skip-lpips`` and CPU FP32 exist only for contract tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_turtle_streaming import image_metrics, prepare_output_directory  # noqa: E402
from scripts.train_turtle_streaming import (  # noqa: E402
    DEFAULT_TURTLE_CHECKPOINT,
    DEFAULT_TURTLE_CONFIG,
    DEFAULT_TURTLE_REPO,
    choose_device,
    read_rgb_tensor,
)
from src.turtle_backend import (  # noqa: E402
    FINETUNED_CHECKPOINT_FORMAT,
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    TURTLE_CACHE_CONTRACT,
    TurtleStreamingBackend,
    load_turtle_model,
    sha256_file,
)


EVALUATION_SCHEMA = "unblur_slam.turtle_single_image_defocus_evaluation.v1"
CANONICAL_PAIR_SCHEMA = "unblur_slam.dpdd_hf_canonical_pair.v1"
MIXED_TRAINING_SCHEMA = "unblur_slam.turtle_replica424_dpdd_hf_mixed_training.v3"
FORMAL_TRAINING_STEPS = 78
FORMAL_TRAINING_SEEDS = frozenset({17, 42, 73})
FORMAL_VIDEO_TRAIN_MANIFEST_SHA256 = (
    "bd7caa189374683c8ffd7e8fce83cb62e5f69b73f6048808c4808dc2b4ecd2ba"
)
DPDD_DATASET_MANIFEST_SCHEMA = "unblur_slam.dpdd_hf_png16_materialization.v1"
DPDD_REPOSITORY = "JacobLinCool/DPDD"
DPDD_REVISION = "52e4035a045ea1763313b9ce2b27cf2e620cfc30"
DPDD_CONFIG = "combined"
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


@dataclass(frozen=True)
class SingleImagePair:
    name: str
    blurry: Path
    sharp: Path
    split: Optional[str] = None
    source_sha256: Optional[str] = None
    target_sha256: Optional[str] = None


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError("missing one of: " + ", ".join(keys))


def _scalar_path(value: Any, *, label: str) -> Any:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"single-image {label} list must contain exactly one path")
        return value[0]
    return value


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _canonical_relative_path(value: Any, root: Path, *, label: str) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        raise ValueError(f"canonical {label} path must be relative to the data root")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"canonical {label} path escapes the data root: {value!r}") from error
    return resolved


def _sha256_text(value: Any, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"canonical {label} must be one lowercase SHA256")
    return normalized


def _normalize_split(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return "validation" if normalized == "val" else normalized


def reject_sealed_test_split(value: Any) -> None:
    """Reject DPDD test before touching paths, devices, outputs, or pixels."""

    if _normalize_split(value) == "test":
        raise ValueError(
            "DPDD test pixels/decodes/metrics remain sealed in this evaluator; "
            "test metadata is not pristine"
        )


def load_single_image_manifest(
    manifest: Path | str,
    *,
    root: Optional[Path | str] = None,
    expected_split: Optional[str] = None,
    canonical_contract: bool = False,
    verify_content: bool = False,
) -> List[SingleImagePair]:
    """Load paired JSONL while rejecting any accidental temporal sequence."""

    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    data_root = (
        Path(root).expanduser().resolve() if root is not None else manifest_path.parent
    )
    records: List[SingleImagePair] = []
    names = set()
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON on {manifest_path}:{line_number}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError(f"manifest line {line_number} must be an object")
        if canonical_contract and payload.get("schema") != CANONICAL_PAIR_SCHEMA:
            raise ValueError(
                f"manifest line {line_number} schema must be {CANONICAL_PAIR_SCHEMA!r}"
            )
        payload_split = payload.get("split")
        normalized_split = _normalize_split(payload_split)
        if normalized_split == "test":
            raise ValueError(
                "DPDD test pixels/decodes/metrics remain sealed in this evaluator; "
                "test metadata is not pristine"
            )
        if expected_split is not None:
            wanted_split = _normalize_split(expected_split)
            if normalized_split is None:
                raise ValueError(
                    f"manifest line {line_number} has no split field; cannot select "
                    f"the locked {wanted_split!r} split"
                )
            if normalized_split != wanted_split:
                # Do not resolve or open pixels belonging to an unselected split.
                continue
        if canonical_contract:
            required = {
                "name",
                "split",
                "defocus",
                "sharp",
                "source_sha256",
                "target_sha256",
            }
            missing = sorted(required - set(payload))
            if missing:
                raise ValueError(
                    f"canonical manifest line {line_number} is missing {missing}"
                )
            source = payload
        elif "frames" in payload:
            frames = payload["frames"]
            if not isinstance(frames, list) or len(frames) != 1:
                raise ValueError(
                    f"single-image manifest line {line_number} must contain one frame"
                )
            frame = frames[0]
            if not isinstance(frame, dict):
                raise ValueError(f"manifest line {line_number} frame must be an object")
            source = frame
        elif isinstance(payload.get("pair"), dict):
            source = payload["pair"]
        elif isinstance(payload.get("paths"), dict):
            source = payload["paths"]
        else:
            source = payload
        name = str(
            payload.get(
                "name",
                payload.get(
                    "pair_id",
                    payload.get("id", payload.get("sequence", f"line_{line_number}")),
                ),
            )
        )
        if name in names:
            raise ValueError(f"duplicate pair name: {name!r}")
        names.add(name)
        blurry_value = _scalar_path(
            _first(
                source,
                (
                    "combined",
                    "combined_blur",
                    "input_combined",
                    "blur_combined",
                    "defocus",
                    "blurry",
                    "blur",
                    "input",
                    "lq",
                ),
            ),
            label="blurry",
        )
        sharp_value = _scalar_path(
            _first(source, ("sharp", "target", "gt", "ground_truth")),
            label="sharp",
        )
        blurry = (
            _canonical_relative_path(blurry_value, data_root, label="defocus")
            if canonical_contract
            else _resolve_path(blurry_value, data_root)
        )
        sharp = (
            _canonical_relative_path(sharp_value, data_root, label="sharp")
            if canonical_contract
            else _resolve_path(sharp_value, data_root)
        )
        if blurry == sharp:
            raise ValueError(f"pair {name!r} points blurry and sharp to the same file")
        for label, path in (("blurry", blurry), ("sharp", sharp)):
            if not path.is_file():
                raise FileNotFoundError(f"pair {name!r} {label} file is missing: {path}")
        source_hash = (
            _sha256_text(payload.get("source_sha256"), label="source_sha256")
            if canonical_contract
            else None
        )
        target_hash = (
            _sha256_text(payload.get("target_sha256"), label="target_sha256")
            if canonical_contract
            else None
        )
        if verify_content:
            if source_hash is None or target_hash is None:
                raise ValueError("content verification requires canonical file hashes")
            if sha256_file(blurry) != source_hash:
                raise ValueError(f"pair {name!r} defocus SHA256 mismatch")
            if sha256_file(sharp) != target_hash:
                raise ValueError(f"pair {name!r} sharp SHA256 mismatch")
        with Image.open(blurry) as blurry_image, Image.open(sharp) as sharp_image:
            if blurry_image.size != sharp_image.size:
                raise ValueError(
                    f"pair {name!r} is not pixel-aligned: "
                    f"blurry={blurry_image.size}, sharp={sharp_image.size}"
                )
            if canonical_contract and (
                blurry_image.format != "PNG" or sharp_image.format != "PNG"
            ):
                raise ValueError(f"pair {name!r} canonical direct-HF assets must be PNG")
        records.append(
            SingleImagePair(
                name=name,
                blurry=blurry,
                sharp=sharp,
                split=normalized_split,
                source_sha256=source_hash,
                target_sha256=target_hash,
            )
        )
    if not records:
        raise ValueError(f"manifest contains no image pairs: {manifest_path}")
    return records


def pad_to_multiple(
    image: torch.Tensor, multiple: int = 8
) -> Tuple[torch.Tensor, int, int, int, int]:
    """Deterministically reflect-pad BCHW on the right/bottom only."""

    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("single-image TURTLE input must be 1x3xHxW")
    if multiple < 1:
        raise ValueError("padding multiple must be positive")
    height, width = (int(value) for value in image.shape[-2:])
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if not pad_height and not pad_width:
        return image, height, width, 0, 0
    mode = (
        "reflect"
        if height > pad_height and width > pad_width and height > 1 and width > 1
        else "replicate"
    )
    padded = F.pad(image, (0, pad_width, 0, pad_height), mode=mode)
    return padded, height, width, pad_height, pad_width


def read_pair_tensor(
    path: Path,
    *,
    device: torch.device,
    require_png_rgb16: bool,
) -> torch.Tensor:
    """Read canonical DPDD PNG without quantizing its 16-bit RGB samples."""

    if not require_png_rgb16:
        return read_rgb_tensor(path, device=device)
    try:
        import cv2
    except Exception as error:
        raise RuntimeError("canonical DPDD RGB16 decoding requires OpenCV") from error
    array_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array_bgr is None:
        raise ValueError(f"OpenCV could not decode canonical DPDD image: {path}")
    if array_bgr.dtype != np.uint16 or array_bgr.ndim != 3 or array_bgr.shape[2] != 3:
        raise ValueError(
            "canonical DPDD image must decode as 16-bit three-channel PNG, "
            f"got dtype={array_bgr.dtype}, shape={array_bgr.shape}: {path}"
        )
    # OpenCV returns BGR. Keep all 16 bits before normalization to [0,1].
    array_rgb = np.ascontiguousarray(array_bgr[:, :, ::-1])
    tensor = torch.from_numpy(array_rgb.astype(np.float32) / 65535.0)
    return tensor.permute(2, 0, 1).to(device)


def parse_assignment(value: str, *, kind: str) -> Tuple[str, str]:
    """Parse one ``name=value`` CLI assignment without path punctuation traps."""

    if "=" not in value:
        raise ValueError(f"{kind} must use name=value syntax")
    name, assigned = value.split("=", 1)
    name = name.strip()
    assigned = assigned.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"invalid {kind} name: {name!r}")
    if not assigned:
        raise ValueError(f"empty {kind} value for {name!r}")
    return name, assigned


def validate_distinct_formal_checkpoint_hashes(
    checkpoint_hashes: Mapping[str, str],
) -> None:
    """Reject duplicated weights hidden behind different formal arm labels."""

    if set(checkpoint_hashes) != {"G", "V", "S", "M"}:
        raise ValueError("formal checkpoint hashes must contain exactly G,V,S,M")
    normalized = {
        arm: _sha256_text(value, label=f"{arm} checkpoint_sha256")
        for arm, value in checkpoint_hashes.items()
    }
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("formal G/V/S/M arms must use four distinct checkpoints")


def validate_formal_arm_metadata(
    arm_name: str,
    checkpoint_sha256: str,
    metadata: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_video_manifest_sha256: str,
    expected_dpdd_train_sha256: str,
    expected_dpdd_dataset_manifest_sha256: str,
) -> None:
    """Bind a formal arm label to the exact checkpoint training provenance."""

    arm = str(arm_name).strip().upper()
    if arm not in {"G", "V", "S", "M"}:
        raise ValueError(f"unknown formal checkpoint arm: {arm_name!r}")
    actual_hash = _sha256_text(checkpoint_sha256, label=f"{arm} checkpoint_sha256")
    expected_video_hash = _sha256_text(
        expected_video_manifest_sha256,
        label="expected_video_manifest_sha256",
    )
    expected_dpdd_hash = _sha256_text(
        expected_dpdd_train_sha256,
        label="expected_dpdd_train_sha256",
    )
    expected_dpdd_dataset_hash = _sha256_text(
        expected_dpdd_dataset_manifest_sha256,
        label="expected_dpdd_dataset_manifest_sha256",
    )
    if expected_seed not in FORMAL_TRAINING_SEEDS:
        raise ValueError("formal checkpoint seed must be 17, 42, or 73")

    common = {
        "base_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
        "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
        "input_domain": "raw",
        "cache_contract": TURTLE_CACHE_CONTRACT,
    }
    mismatches = {
        key: (metadata.get(key), wanted)
        for key, wanted in common.items()
        if metadata.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"formal arm {arm} model/base pin mismatch: {mismatches}")

    if arm == "G":
        if actual_hash != PINNED_TURTLE_CHECKPOINT_SHA256:
            raise ValueError("formal G must be the byte-pinned official GoPro checkpoint")
        if metadata.get("kind") != "official_gopro":
            raise ValueError("formal G metadata kind must be official_gopro")
        if metadata.get("checkpoint_sha256") != actual_hash:
            raise ValueError("formal G normalized metadata hash mismatch")
        return

    if metadata.get("kind") != "finetuned":
        raise ValueError(f"formal arm {arm} must be a validated fine-tuned checkpoint")
    if metadata.get("format") != FINETUNED_CHECKPOINT_FORMAT:
        raise ValueError(f"formal arm {arm} checkpoint format mismatch")
    if metadata.get("schema") != MIXED_TRAINING_SCHEMA:
        raise ValueError(f"formal arm {arm} training schema mismatch")
    if metadata.get("checkpoint_sha256") != actual_hash:
        raise ValueError(f"formal arm {arm} normalized metadata hash mismatch")
    if metadata.get("mode") != arm:
        raise ValueError(
            f"formal arm {arm} contains mode={metadata.get('mode')!r} checkpoint"
        )
    if metadata.get("uses_paired_sharp_ground_truth_rgb") is not True:
        raise ValueError(f"formal arm {arm} paired-sharp supervision flag mismatch")
    if metadata.get("uses_gt_pose") is not False or metadata.get("uses_gt_depth") is not False:
        raise ValueError(f"formal arm {arm} pose/depth supervision flags mismatch")

    training = metadata.get("training")
    if not isinstance(training, Mapping):
        raise ValueError(f"formal arm {arm} has no training metadata mapping")
    if type(training.get("seed")) is not int or training.get("seed") != expected_seed:
        raise ValueError(f"formal arm {arm} seed does not match --expected-seed")
    if training.get("optimizer_steps") != FORMAL_TRAINING_STEPS:
        raise ValueError(f"formal arm {arm} must contain exactly 78 optimizer steps")
    if (
        training.get("attempted_optimizer_steps") != FORMAL_TRAINING_STEPS
        or training.get("executed_optimizer_steps") != FORMAL_TRAINING_STEPS
        or training.get("amp_skipped_optimizer_steps") != 0
    ):
        raise ValueError(f"formal arm {arm} AMP execution budget mismatch")
    if training.get("amp") is not True:
        raise ValueError(f"formal arm {arm} must have been trained with AMP")
    expected_mixed_step = "two_backward_one_joint_step" if arm == "M" else None
    if training.get("mixed_step") != expected_mixed_step:
        raise ValueError(f"formal arm {arm} mixed-step provenance mismatch")
    if training.get("grad_scaler") != {
        "init_scale": 1024.0,
        "growth_interval": 2000,
        "growth_disabled_within_78_steps": True,
        "overflow_policy": "fail_closed_no_checkpoint",
    }:
        raise ValueError(f"formal arm {arm} GradScaler contract mismatch")

    manifests = metadata.get("manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError(f"formal arm {arm} has no manifests metadata mapping")
    wanted_video_hash = expected_video_hash if arm in {"V", "M"} else None
    wanted_dpdd_hash = expected_dpdd_hash if arm in {"S", "M"} else None
    if manifests.get("video_sha256") != wanted_video_hash:
        raise ValueError(f"formal arm {arm} Replica training-manifest hash mismatch")
    if manifests.get("dpdd_pairs_sha256") != wanted_dpdd_hash:
        raise ValueError(f"formal arm {arm} DPDD training-manifest hash mismatch")
    wanted_split = "train" if arm in {"S", "M"} else None
    if manifests.get("dpdd_selected_split") != wanted_split:
        raise ValueError(f"formal arm {arm} DPDD split provenance mismatch")
    dpdd_dataset = manifests.get("dpdd_dataset")
    if arm in {"S", "M"}:
        if not isinstance(dpdd_dataset, Mapping):
            raise ValueError(f"formal arm {arm} has no DPDD dataset provenance")
        required_dataset = {
            "sha256": expected_dpdd_dataset_hash,
            "schema": DPDD_DATASET_MANIFEST_SCHEMA,
            "repository": DPDD_REPOSITORY,
            "revision": DPDD_REVISION,
            "config": DPDD_CONFIG,
            "dataset_card_declared_license": "mit",
            "test_metadata_pristine": False,
            "test_pixels_opened": False,
            "test_metrics_opened": False,
            "canonical_train_manifest_sha256": expected_dpdd_hash,
        }
        dataset_mismatches = {
            key: (dpdd_dataset.get(key), wanted)
            for key, wanted in required_dataset.items()
            if dpdd_dataset.get(key) != wanted
        }
        if dataset_mismatches:
            raise ValueError(
                f"formal arm {arm} DPDD dataset provenance mismatch: "
                f"{dataset_mismatches}"
            )
        if not str(dpdd_dataset.get("license_scope_warning", "")).strip():
            raise ValueError(f"formal arm {arm} DPDD license warning is missing")
    elif dpdd_dataset is not None:
        raise ValueError("formal V must not contain DPDD dataset provenance")
    if manifests.get("test_pixels_or_metrics_read") is not False:
        raise ValueError(f"formal arm {arm} does not attest unopened DPDD test pixels")


def load_dpdd_evaluation_dataset_contract(
    dataset_manifest: Path,
    *,
    expected_dataset_manifest_sha256: str,
    validation_manifest: Path,
    expected_validation_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Three-way bind dataset metadata, validation index, and selected JSONL."""

    path = Path(dataset_manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DPDD dataset manifest does not exist: {path}")
    dataset_hash = sha256_file(path)
    expected_dataset_hash = _sha256_text(
        expected_dataset_manifest_sha256,
        label="DPDD dataset-manifest SHA256",
    )
    if dataset_hash != expected_dataset_hash:
        raise ValueError("DPDD dataset-manifest SHA256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("DPDD dataset manifest is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("DPDD dataset manifest must be a JSON object")
    fixed = {
        "schema": DPDD_DATASET_MANIFEST_SCHEMA,
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
        raise ValueError(f"DPDD evaluation dataset provenance mismatch: {mismatches}")

    canonical = payload.get("canonical_manifests")
    entry = canonical.get("validation") if isinstance(canonical, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ValueError("DPDD canonical validation manifest metadata is missing")
    relative = Path(str(entry.get("path", "")))
    if relative.is_absolute():
        raise ValueError("DPDD canonical validation path must be relative")
    dataset_root = path.parent.resolve()
    resolved_validation = (dataset_root / relative).resolve()
    try:
        resolved_validation.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError("DPDD canonical validation path escapes dataset root") from error
    expected_validation_hash = _sha256_text(
        expected_validation_manifest_sha256,
        label="DPDD validation-manifest SHA256",
    )
    if (
        entry.get("schema") != CANONICAL_PAIR_SCHEMA
        or entry.get("rows") != 74
        or entry.get("paths_relative_to") != "dataset_root"
        or entry.get("sha256") != expected_validation_hash
        or resolved_validation != Path(validation_manifest).expanduser().resolve()
        or not resolved_validation.is_file()
        or sha256_file(resolved_validation) != expected_validation_hash
    ):
        raise ValueError("DPDD canonical validation three-way binding mismatch")

    distribution = payload.get("distribution")
    disclosure = payload.get("test_disclosure")
    if not isinstance(distribution, Mapping) or not isinstance(disclosure, Mapping):
        raise ValueError("DPDD distribution/test provenance is missing")
    license_claim = distribution.get("dataset_card_declared_license")
    license_warning = distribution.get("license_scope_warning")
    if str(license_claim).strip().lower() != "mit" or not str(license_warning).strip():
        raise ValueError("DPDD dataset license claim/warning is incomplete")
    required_disclosure = {
        "metadata_pristine": False,
        "images_decoded": False,
        "pixels_opened": False,
        "metrics_opened": False,
        "split_supported_by_this_materializer": False,
    }
    if any(disclosure.get(key) is not wanted for key, wanted in required_disclosure.items()):
        raise ValueError("DPDD test disclosure mismatch")
    return {
        "path": str(path),
        "sha256": dataset_hash,
        "schema": payload["schema"],
        "repository": payload["repository"],
        "revision": payload["revision"],
        "config": payload["config"],
        "dataset_card_declared_license": license_claim,
        "license_scope_warning": license_warning,
        "test_disclosure": {
            key: disclosure[key] for key in required_disclosure
        },
        "canonical_validation_manifest": {
            "path": str(resolved_validation),
            "sha256": expected_validation_hash,
            "rows": 74,
            "schema": CANONICAL_PAIR_SCHEMA,
        },
    }


def _mean(values: Iterable[float]) -> float:
    normalized = [float(value) for value in values]
    if not normalized or not all(math.isfinite(value) for value in normalized):
        raise ValueError("metric aggregation requires finite non-empty values")
    return float(np.mean(normalized))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("latency aggregation requires at least one value")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _load_lpips_metric(device: torch.device):
    try:
        import torchmetrics
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except Exception as error:
        raise RuntimeError("formal evaluation requires torchmetrics LPIPS") from error
    if str(torchmetrics.__version__) != LPIPS_PROTOCOL["torchmetrics_version"]:
        raise RuntimeError("formal LPIPS torchmetrics version changed")
    for label in ("alexnet_backbone", "lpips_linear_weights"):
        artifact = LPIPS_PROTOCOL[label]
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"formal LPIPS {label} artifact changed")
    try:
        metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
    except Exception as error:
        raise RuntimeError("could not load AlexNet LPIPS weights") from error
    metric.eval().requires_grad_(False)
    return metric


@torch.no_grad()
def _lpips_value(metric: Any, prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.detach().float().clamp(0.0, 1.0).unsqueeze(0)
    target = target.detach().float().clamp(0.0, 1.0).unsqueeze(0)
    metric_device = next(metric.parameters()).device
    value = metric(prediction.to(metric_device), target.to(metric_device))
    result = float(torch.as_tensor(value).detach().float().mean().item())
    if hasattr(metric, "reset"):
        metric.reset()
    if not math.isfinite(result):
        raise RuntimeError("LPIPS returned a non-finite value")
    return result


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_metric: Optional[Any],
) -> Dict[str, float]:
    values = image_metrics(prediction, target)
    if lpips_metric is not None:
        values["lpips"] = _lpips_value(lpips_metric, prediction, target)
    return values


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_arm(
    records: Sequence[SingleImagePair],
    backend: Any,
    *,
    device: torch.device,
    lpips_metric: Optional[Any],
    padding_multiple: int = 8,
    warmup_steps: int = 1,
    require_png_rgb16: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate one checkpoint, proving empty K/V before every image call."""

    if not records:
        raise ValueError("cannot evaluate an empty record list")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    first = read_pair_tensor(
        records[0].blurry,
        device=device,
        require_png_rgb16=require_png_rgb16,
    ).unsqueeze(0)
    first, _, _, _, _ = pad_to_multiple(first, padding_multiple)
    for _ in range(warmup_steps):
        backend.reset()
        backend.step(first, timestamp=0)
    backend.reset()

    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        blurry = read_pair_tensor(
            record.blurry, device=device, require_png_rgb16=require_png_rgb16
        )
        sharp = read_pair_tensor(
            record.sharp, device=device, require_png_rgb16=require_png_rgb16
        )
        if blurry.shape != sharp.shape:
            raise ValueError(f"pair {record.name!r} changed shape after RGB decoding")
        padded, height, width, pad_height, pad_width = pad_to_multiple(
            blurry.unsqueeze(0), padding_multiple
        )
        backend.reset()
        before = dict(backend.state_info())
        if bool(before.get("has_cache")):
            raise RuntimeError(f"pair {record.name!r} began with non-empty K/V")
        _synchronize(device)
        started = time.perf_counter()
        restored_padded = backend.step(padded, timestamp=0)
        _synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        after = dict(backend.state_info())
        if not bool(after.get("has_cache")):
            raise RuntimeError(f"pair {record.name!r} did not produce official K/V")
        restored = restored_padded[0, :, :height, :width]
        rows.append(
            {
                "index": index,
                "name": record.name,
                "blurry_path": str(record.blurry),
                "sharp_path": str(record.sharp),
                "source_height": height,
                "source_width": width,
                "pad_bottom": pad_height,
                "pad_right": pad_width,
                "cache_empty_before_call": True,
                "cache_populated_after_call": True,
                "metrics": _metrics(restored, sharp, lpips_metric),
                "latency_ms": float(latency_ms),
            }
        )
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    metric_names = tuple(rows[0]["metrics"].keys())
    for row in rows:
        if tuple(row["metrics"].keys()) != metric_names:
            raise ValueError("per-image metric keys changed within one arm")
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "image_count": len(rows),
        "mean": {
            metric: _mean(float(row["metrics"][metric]) for row in rows)
            for metric in metric_names
        },
        "latency_ms": {
            "mean": _mean(latencies),
            "median": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": max(latencies),
        },
    }


@torch.no_grad()
def evaluate_raw_pairs(
    records: Sequence[SingleImagePair],
    *,
    device: torch.device,
    lpips_metric: Optional[Any],
    require_png_rgb16: bool,
) -> List[Dict[str, Any]]:
    """Compute the absolute defocus-input baseline once on the locked pairs."""

    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        blurry = read_pair_tensor(
            record.blurry, device=device, require_png_rgb16=require_png_rgb16
        )
        sharp = read_pair_tensor(
            record.sharp, device=device, require_png_rgb16=require_png_rgb16
        )
        rows.append(
            {
                "index": index,
                "name": record.name,
                "blurry_path": str(record.blurry),
                "sharp_path": str(record.sharp),
                "metrics": _metrics(blurry, sharp, lpips_metric),
                "latency_ms": 0.0,
            }
        )
    return rows


def paired_arm_delta(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("paired arm comparison requires matching non-empty rows")
    metric_names = tuple(candidate[0]["metrics"].keys())
    per_image: List[Dict[str, Any]] = []
    for candidate_row, reference_row in zip(candidate, reference):
        if candidate_row["name"] != reference_row["name"]:
            raise ValueError("checkpoint arms are not aligned to the same image order")
        if tuple(candidate_row["metrics"].keys()) != metric_names or tuple(
            reference_row["metrics"].keys()
        ) != metric_names:
            raise ValueError("checkpoint arms do not contain identical metrics")
        per_image.append(
            {
                "name": candidate_row["name"],
                "candidate_minus_reference": {
                    metric: float(candidate_row["metrics"][metric])
                    - float(reference_row["metrics"][metric])
                    for metric in metric_names
                },
            }
        )
    return {
        "interpretation": (
            "paired current-frame spatial-restoration delta with K/V reset; "
            "not a history benefit"
        ),
        "mean_candidate_minus_reference": {
            metric: _mean(
                row["candidate_minus_reference"][metric] for row in per_image
            )
            for metric in metric_names
        },
        "per_image": per_image,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "val", "test"),
        help="select a split from a materializer pairs.jsonl before opening pixels",
    )
    parser.add_argument(
        "--expected-pair-count",
        type=int,
        default=0,
        help="fail closed when the selected split count differs; 0 disables",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        help="required by the formal protocol and checked before any pair pixels",
    )
    parser.add_argument(
        "--expected-seed",
        type=int,
        help="formal V/S/M training seed; one independent report per seed",
    )
    parser.add_argument(
        "--expected-video-manifest-sha",
        help="formal content hash of the Replica training manifest used by V/M",
    )
    parser.add_argument(
        "--expected-dpdd-train-sha",
        help="formal content hash of the canonical DPDD train manifest used by S/M",
    )
    parser.add_argument(
        "--dpdd-dataset-manifest",
        type=Path,
        help="formal materializer dataset_manifest.json",
    )
    parser.add_argument(
        "--dpdd-dataset-manifest-sha256",
        help="formal content hash of --dpdd-dataset-manifest",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "repeat for base/replica_only/mixed; when omitted, evaluate the "
            "pinned official GoPro checkpoint as base"
        ),
    )
    parser.add_argument(
        "--checkpoint-sha256",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="required for every non-official fine-tuned checkpoint",
    )
    parser.add_argument("--reference-arm", default="base")
    parser.add_argument("--turtle-repo", type=Path, default=DEFAULT_TURTLE_REPO)
    parser.add_argument("--turtle-config", type=Path, default=DEFAULT_TURTLE_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--padding-multiple", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--lpips-device", default="cpu")
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="CPU contract test only; output is marked non-formal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_sealed_test_split(args.split)
    if args.padding_multiple != 8:
        raise ValueError("formal TURTLE single-image protocol fixes padding multiple to 8")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    device = choose_device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 TURTLE evaluation requires CUDA")
    output_dir = prepare_output_directory(args.output_dir)
    manifest = args.manifest.expanduser().resolve()
    requested_formal = bool(
        not args.skip_lpips
        and args.precision == "fp16"
        and device.type == "cuda"
        and args.padding_multiple == 8
    )
    manifest_hash = sha256_file(manifest)
    dpdd_dataset_provenance = None
    if requested_formal:
        if args.warmup_steps != 1:
            raise ValueError("formal evaluation fixes --warmup-steps 1")
        if args.expected_manifest_sha256 is None:
            raise ValueError("formal evaluation requires --expected-manifest-sha256")
        if args.expected_manifest_sha256.lower() != manifest_hash:
            raise ValueError("formal dataset manifest SHA256 mismatch")
        normalized_split = _normalize_split(args.split)
        if normalized_split not in {"validation", "test"}:
            raise ValueError("formal evaluation is locked to validation or authorized test")
        expected_count = 74 if normalized_split == "validation" else 76
        if args.expected_pair_count != expected_count:
            raise ValueError(
                f"formal {normalized_split} evaluation requires "
                f"--expected-pair-count {expected_count}"
            )
        if (
            args.expected_seed is None
            or args.expected_video_manifest_sha is None
            or args.expected_dpdd_train_sha is None
            or args.dpdd_dataset_manifest is None
            or args.dpdd_dataset_manifest_sha256 is None
        ):
            raise ValueError(
                "formal evaluation requires --expected-seed, "
                "--expected-video-manifest-sha, --expected-dpdd-train-sha, "
                "--dpdd-dataset-manifest, and --dpdd-dataset-manifest-sha256"
            )
        if args.expected_seed not in FORMAL_TRAINING_SEEDS:
            raise ValueError("formal --expected-seed must be 17, 42, or 73")
        expected_video_hash = _sha256_text(
            args.expected_video_manifest_sha,
            label="expected_video_manifest_sha",
        )
        if expected_video_hash != FORMAL_VIDEO_TRAIN_MANIFEST_SHA256:
            raise ValueError("formal Replica training-manifest SHA256 mismatch")
        _sha256_text(args.expected_dpdd_train_sha, label="expected_dpdd_train_sha")
        dpdd_dataset_provenance = load_dpdd_evaluation_dataset_contract(
            args.dpdd_dataset_manifest,
            expected_dataset_manifest_sha256=args.dpdd_dataset_manifest_sha256,
            validation_manifest=manifest,
            expected_validation_manifest_sha256=manifest_hash,
        )
    records = load_single_image_manifest(
        manifest,
        root=args.data_root,
        expected_split=args.split,
        canonical_contract=requested_formal,
        verify_content=requested_formal,
    )
    if args.expected_pair_count < 0:
        raise ValueError("--expected-pair-count cannot be negative")
    if args.expected_pair_count and len(records) != args.expected_pair_count:
        raise ValueError(
            f"selected pair count mismatch: expected {args.expected_pair_count}, "
            f"got {len(records)}"
        )

    checkpoint_items = [
        parse_assignment(value, kind="checkpoint") for value in args.checkpoint
    ]
    if not checkpoint_items:
        checkpoint_items = [("base", str(DEFAULT_TURTLE_CHECKPOINT))]
    checkpoints = dict(checkpoint_items)
    if len(checkpoints) != len(checkpoint_items):
        raise ValueError("checkpoint arm names must be unique")
    sha_items = [
        parse_assignment(value, kind="checkpoint SHA256")
        for value in args.checkpoint_sha256
    ]
    configured_hashes = {
        name: _sha256_text(value, label=f"{name} checkpoint SHA256")
        for name, value in sha_items
    }
    if len(configured_hashes) != len(sha_items):
        raise ValueError("checkpoint SHA256 arm names must be unique")
    if not set(configured_hashes).issubset(checkpoints):
        raise ValueError("checkpoint SHA256 supplied for an unknown arm")
    if args.reference_arm not in checkpoints:
        raise ValueError("--reference-arm is not one of the checkpoint arms")
    if requested_formal:
        if set(checkpoints) != {"G", "V", "S", "M"}:
            raise ValueError("formal validation requires exactly checkpoint arms G,V,S,M")
        if args.reference_arm != "G":
            raise ValueError("formal validation fixes G as the reference arm")

    resolved_checkpoints: List[Tuple[str, Path, str]] = []
    checkpoint_hashes: Dict[str, str] = {}
    for arm_name, checkpoint_value in checkpoint_items:
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        checkpoint_hash = sha256_file(checkpoint)
        supplied_hash = configured_hashes.get(arm_name)
        if checkpoint_hash != PINNED_TURTLE_CHECKPOINT_SHA256:
            if supplied_hash is None:
                raise ValueError(
                    f"fine-tuned arm {arm_name!r} requires --checkpoint-sha256 "
                    f"{arm_name}={checkpoint_hash}"
                )
            if supplied_hash != checkpoint_hash:
                raise ValueError(f"arm {arm_name!r} checkpoint SHA256 mismatch")
        elif supplied_hash is not None and supplied_hash != checkpoint_hash:
            raise ValueError(f"arm {arm_name!r} official checkpoint SHA256 mismatch")
        resolved_checkpoints.append((arm_name, checkpoint, checkpoint_hash))
        checkpoint_hashes[arm_name] = checkpoint_hash
    if requested_formal:
        validate_distinct_formal_checkpoint_hashes(checkpoint_hashes)

    lpips_metric = None
    lpips_device = torch.device(args.lpips_device)
    if not args.skip_lpips:
        lpips_metric = _load_lpips_metric(lpips_device)

    raw_rows = evaluate_raw_pairs(
        records,
        device=device,
        lpips_metric=lpips_metric,
        require_png_rgb16=requested_formal,
    )

    arms: Dict[str, Any] = {}
    arm_rows: Dict[str, List[Dict[str, Any]]] = {}
    for arm_name, checkpoint, checkpoint_hash in resolved_checkpoints:
        supplied_hash = configured_hashes.get(arm_name)
        model, metadata = load_turtle_model(
            args.turtle_repo,
            checkpoint,
            config=args.turtle_config,
            device=device,
            checkpoint_sha256=(supplied_hash if supplied_hash is not None else None),
        )
        if requested_formal:
            validate_formal_arm_metadata(
                arm_name,
                checkpoint_hash,
                metadata,
                expected_seed=args.expected_seed,
                expected_video_manifest_sha256=args.expected_video_manifest_sha,
                expected_dpdd_train_sha256=args.expected_dpdd_train_sha,
                expected_dpdd_dataset_manifest_sha256=(
                    args.dpdd_dataset_manifest_sha256
                ),
            )
        if bool(getattr(model, "use_both_input", True)):
            raise ValueError("single-image reset protocol requires use_both_input=false")
        backend = TurtleStreamingBackend(
            model, device=device, inference_precision=args.precision
        )
        rows = evaluate_arm(
            records,
            backend,
            device=device,
            lpips_metric=lpips_metric,
            padding_multiple=args.padding_multiple,
            warmup_steps=args.warmup_steps,
            require_png_rgb16=requested_formal,
        )
        arm_rows[arm_name] = rows
        arms[arm_name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_metadata": dict(metadata),
            "summary": summarize_rows(rows),
            "images": rows,
        }
        del backend, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reference_rows = arm_rows[args.reference_arm]
    comparisons = {
        f"{arm_name}_minus_{args.reference_arm}": paired_arm_delta(rows, reference_rows)
        for arm_name, rows in arm_rows.items()
        if arm_name != args.reference_arm
    }
    arm_minus_raw = {
        arm_name: paired_arm_delta(rows, raw_rows)
        for arm_name, rows in arm_rows.items()
    }
    payload = {
        "schema": EVALUATION_SCHEMA,
        "formal": requested_formal,
        "interpretation": (
            "single-image current-frame spatial restoration only; every pair "
            "starts with empty K/V, so no value may be called a history gain"
        ),
        "protocol": {
            "manifest": str(manifest),
            "manifest_sha256": manifest_hash,
            "checkpoint_training_seed": (
                args.expected_seed if requested_formal else None
            ),
            "replica_train_manifest_sha256": (
                args.expected_video_manifest_sha if requested_formal else None
            ),
            "dpdd_train_manifest_sha256": (
                args.expected_dpdd_train_sha if requested_formal else None
            ),
            "dpdd_dataset": dpdd_dataset_provenance,
            "selected_split": args.split,
            "ordered_pair_names": [record.name for record in records],
            "pair_count": len(records),
            "cache_boundary": "hard_reset_before_every_image",
            "two_frame_wrapper": "pair=(current,current)",
            "use_both_input": False,
            "inference_precision": args.precision,
            "padding": {
                "multiple": args.padding_multiple,
                "sides": "right_and_bottom_only",
                "mode": "reflect_if_valid_else_replicate",
                "crop_output_back_to_source_shape": True,
            },
            "input_decode": (
                "cv2.IMREAD_UNCHANGED; uint16 BGR to RGB; float32 divide by 65535"
                if requested_formal
                else "non-formal generic RGB decode"
            ),
            "latency_scope": (
                "one TURTLE model step; input resident on target device; "
                "pre/post CUDA synchronization; excludes image I/O, reset, metrics"
            ),
            "warmup_independent_reset_calls_per_arm": args.warmup_steps,
            "metrics": {
                "psnr": "RGB [0,1], per-image then arithmetic mean",
                "ssim": "skimage RGB channel_axis=2, data_range=1",
                "l1": "RGB [0,1], per-image mean absolute error",
                "lpips": (LPIPS_PROTOCOL if lpips_metric is not None else None),
            },
            "device": str(device),
            "lpips_device": str(lpips_device) if lpips_metric is not None else None,
        },
        "reference_arm": args.reference_arm,
        "raw_defocus_baseline": {
            "summary": summarize_rows(raw_rows),
            "images": raw_rows,
        },
        "arms": arms,
        "paired_comparisons": comparisons,
        "arm_minus_raw_defocus": arm_minus_raw,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "formal": payload["formal"],
                "pair_count": len(records),
                "arm_summaries": {
                    name: value["summary"] for name, value in arms.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
