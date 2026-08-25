#!/usr/bin/env python3
"""Build the upload-only preprocessing bundle without third-party REDS/GoPro pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


SCHEMA = "unblur_slam.turtle_unblur_preprocessing_bundle.v1"
CARD = """---
license: other
task_categories:
- image-to-image
pretty_name: Unblur-SLAM TURTLE preprocessing bundle
---

# Unblur-SLAM TURTLE preprocessing bundle

This directory contains content-addressed **training manifests** and the
derived ReplicaBlurry Office3 stream used by the strict three-stage TURTLE
experiment.  It does not republish REDS, GoPro, BSD, DPDD, TUM, or held-out
test pixels.

REDS and GoPro must be downloaded from their pinned owner/mirror repositories;
the cloud preparation script verifies the archive hashes and extracts only the
training split.  The Unblur-SLAM defocus pixels already live in the parent
owner dataset repository and are not duplicated here.

The exact protocol and commands are in
`docs/TURTLE_UNBLUR_CLOUD_REPRODUCTION.md` in the matching GitHub commit.
The parent repository's licensing notice and each upstream dataset's terms
continue to apply; this bundle does not grant additional redistribution rights.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build(manifest_root: Path, replica_root: Path, output: Path) -> Mapping[str, Any]:
    manifest_root = manifest_root.expanduser().resolve()
    replica_root = replica_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_exclusive(output / "README.md", CARD.encode("utf-8"))
        manifest_names = (
            "reds_train.content_addressed.v1.jsonl",
            "gopro_large_blur_gamma_train.content_addressed.v1.jsonl",
            "unblur_hf_defocus_train.paired_image.v1.jsonl",
        )
        for name in manifest_names:
            copy_file(manifest_root / name, output / "manifests" / name)
        for source in sorted(replica_root.rglob("*")):
            if source.is_symlink():
                raise ValueError(f"replica preprocessing contains symlink: {source}")
            if source.is_file():
                relative = source.relative_to(replica_root)
                copy_file(source, output / "replica_blurry_office3" / relative)
        records = []
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "BUNDLE_MANIFEST.json":
                records.append({
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        payload = {
            "schema": SCHEMA,
            "status": "complete",
            "file_count_excluding_manifest": len(records),
            "total_bytes_excluding_manifest": sum(row["bytes"] for row in records),
            "records": records,
            "contains_third_party_reds_or_gopro_pixels": False,
            "contains_held_out_test_pixels": False,
            "replica_preprocessing_included": True,
        }
        write_exclusive(
            output / "BUNDLE_MANIFEST.json",
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return payload
    except Exception:
        shutil.rmtree(output)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--replica-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = build(arguments.manifest_root, arguments.replica_root, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
