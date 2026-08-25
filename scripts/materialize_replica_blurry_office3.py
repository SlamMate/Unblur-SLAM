#!/usr/bin/env python3
"""Materialize causal ReplicaBlurry pairs from pinned Office3 subframes.

Each non-overlapping 36-frame exposure in the numerically sorted published
sequence is averaged in linear-light RGB and
encoded back to sRGB PNG.  Its paired target is the exposure midpoint at
offset 18, matching Unblur-SLAM's ReplicaBlurry indexing contract.  The
source must contain unique, strictly increasing ``rgb_<integer>.png`` indices;
the published Office3 subset is intentionally sparse in the original render
index, so numeric gaps are recorded rather than fabricated.  Trailing frames
shorter than one exposure are declared and ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Dict, List

import numpy as np
from PIL import Image


SCHEMA = "unblur_slam.replica_blurry_office3_materialization.v1"
PAIR_SCHEMA = "unblur_slam.paired_video_train.v1"
EXPOSURE_FRAMES = 36
MIDPOINT_OFFSET = 18
PINNED_REPO = "qizhangslam/Unblur_slam_traning_dataset"
PINNED_REVISION = "1f9d98158c3f27f6ec6de45ee2874c9caf2a2c59"
NAME = re.compile(r"rgb_(\d+)\.png")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def srgb_to_linear(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    return np.where(
        array <= 0.04045,
        array / 12.92,
        ((array + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    return np.where(
        array <= 0.0031308,
        array * 12.92,
        1.055 * np.power(array, 1.0 / 2.4) - 0.055,
    )


def _write_json_exclusive(path: Path, payload: Dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def materialize(source: Path, destination: Path) -> Dict[str, Any]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"source RGB directory is invalid: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    indexed: Dict[int, Path] = {}
    for path in source.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"unexpected non-regular source entry: {path}")
        match = NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected Office3 source filename: {path.name}")
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"duplicate Office3 frame index: {index}")
        indexed[index] = path
    if not indexed:
        raise ValueError("Office3 source is empty")
    ordered_indices = sorted(indexed)
    ordered_paths = [indexed[index] for index in ordered_indices]
    usable_count = len(indexed) - (len(indexed) % EXPOSURE_FRAMES)
    if usable_count < EXPOSURE_FRAMES:
        raise ValueError("Office3 has no complete 36-frame exposure")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        blur_dir = staging / "blur"
        sharp_dir = staging / "sharp"
        manifest_dir = staging / "manifests"
        blur_dir.mkdir()
        sharp_dir.mkdir()
        manifest_dir.mkdir()
        blurry_paths: List[str] = []
        sharp_paths: List[str] = []
        blurry_hashes: List[str] = []
        sharp_hashes: List[str] = []
        source_index_digest = hashlib.sha256()
        expected_shape = None
        for index, source_path in zip(ordered_indices, ordered_paths):
            source_hash = sha256_file(source_path)
            source_index_digest.update(
                f"{index}\0{source_path.name}\0{source_hash}\n".encode("utf-8")
            )
        for start in range(0, usable_count, EXPOSURE_FRAMES):
            accumulator = None
            for index in range(start, start + EXPOSURE_FRAMES):
                source_path = ordered_paths[index]
                with Image.open(source_path) as image:
                    if image.mode != "RGB":
                        raise ValueError(f"Office3 frame is not RGB: {source_path}")
                    array = np.asarray(image, dtype=np.float32) / 255.0
                if expected_shape is None:
                    expected_shape = tuple(int(value) for value in array.shape)
                if tuple(array.shape) != expected_shape:
                    raise ValueError("Office3 source resolution changed")
                linear = srgb_to_linear(array)
                accumulator = linear if accumulator is None else accumulator + linear
            assert accumulator is not None
            blurred = linear_to_srgb(accumulator / float(EXPOSURE_FRAMES))
            encoded = np.clip(np.rint(blurred * 255.0), 0, 255).astype(np.uint8)
            blur_name = f"rgb_{start}.png"
            sharp_position = start + MIDPOINT_OFFSET
            sharp_source_index = ordered_indices[sharp_position]
            sharp_name = f"rgb_{sharp_source_index}.png"
            blur_path = blur_dir / blur_name
            sharp_path = sharp_dir / sharp_name
            Image.fromarray(encoded, mode="RGB").save(blur_path, format="PNG")
            shutil.copyfile(ordered_paths[sharp_position], sharp_path)
            os.chmod(blur_path, 0o444)
            os.chmod(sharp_path, 0o444)
            blurry_paths.append(f"blur/{blur_name}")
            sharp_paths.append(f"sharp/{sharp_name}")
            blurry_hashes.append(sha256_file(blur_path))
            sharp_hashes.append(sha256_file(sharp_path))

        row = {
            "schema": PAIR_SCHEMA,
            "dataset": "ReplicaBlurry",
            "split": "train",
            "sequence": "replica_blurry_office3_pinned_unblur_slam",
            "frame_count": len(blurry_paths),
            "temporal_order": "strict_numeric_source_order_nonoverlapping_36_subframe_exposures",
            "blurry": blurry_paths,
            "sharp": sharp_paths,
            "blurry_sha256": blurry_hashes,
            "sharp_sha256": sharp_hashes,
        }
        manifest_path = manifest_dir / "train.jsonl"
        fd = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest_sha = sha256_file(manifest_path)
        audit = {
            "schema": SCHEMA,
            "status": "pass",
            "source": {
                "repo": PINNED_REPO,
                "revision": PINNED_REVISION,
                "path": str(source),
                "frame_count": len(indexed),
                "source_index_sha256": source_index_digest.hexdigest(),
                "minimum_index": ordered_indices[0],
                "maximum_index": ordered_indices[-1],
                "numeric_gap_count": sum(
                    right != left + 1
                    for left, right in zip(ordered_indices, ordered_indices[1:])
                ),
                "maximum_numeric_step": max(
                    (right - left for left, right in zip(ordered_indices, ordered_indices[1:])),
                    default=0,
                ),
                "shape_hwc": list(expected_shape or ()),
            },
            "formation": {
                "exposure_frames": EXPOSURE_FRAMES,
                "midpoint_offset": MIDPOINT_OFFSET,
                "linear_light_average": True,
                "transfer": "exact_iec_61966_2_1_srgb",
                "output_quantization": "round_to_nearest_rgb8_png",
                "complete_exposures": len(blurry_paths),
                "trailing_frames_ignored": len(indexed) - usable_count,
            },
            "manifest": {
                "path": "manifests/train.jsonl",
                "sha256": manifest_sha,
                "records": 1,
                "paired_frames": len(blurry_paths),
            },
            "test_members_opened_or_decoded": 0,
        }
        _write_json_exclusive(staging / "materialization_audit.json", audit)
        os.rename(staging, destination)
        staging = None
        return audit
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
