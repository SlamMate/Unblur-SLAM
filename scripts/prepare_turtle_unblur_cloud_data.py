#!/usr/bin/env python3
"""Materialize the pinned cloud inputs for strict TURTLE three-stage training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from huggingface_hub import snapshot_download


SCHEMA = "unblur_slam.turtle_unblur_cloud_data.v1"
REDS_REPO = "snah/REDS"
REDS_REVISION = "62dc25d16e6f43d2214f1b365023abda86f7a0ae"
GOPRO_REPO = "snah/GOPRO_Large"
GOPRO_REVISION = "592978466ae510d2734b199cad2fc79a346bda1c"
UNBLUR_REPO = "qizhangslam/Unblur_slam_traning_dataset"
UNBLUR_REVISION = "1f9d98158c3f27f6ec6de45ee2874c9caf2a2c59"
ARCHIVE_SHA256 = {
    "reds/train_blur.zip": "415c360c0d71d2d89af099b3c64e76bca7e1d8317750f8e056e02ebaab957bb8",
    "reds/train_sharp.zip": "620294c1c3f23ed26c5ea228633770469c0b28e57d31eb41dc77deb401c6681b",
    "gopro/GOPRO_Large.zip": "24532c036712515cf33803704421311c91f6a080e76049d7e5e7fd22389f128e",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def download(repo: str, revision: str, destination: Path,
             patterns: Sequence[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=list(patterns),
        local_dir=destination,
    )


def extract_zip(archive: Path, destination: Path,
                members: Sequence[str] | None = None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = ["unzip", "-q", "-n", str(archive)]
    if members:
        command.extend(members)
    command.extend(["-d", str(destination)])
    subprocess.run(command, check=True)


def run(root: Path, bundle_revision: str) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if root == Path("/"):
        raise ValueError("artifact root must not be filesystem root")
    training = root / "training_data"
    reds = training / "reds_hf_snah"
    gopro = training / "gopro_large_hf_snah"
    unblur = training / "unblur_slam_official_rev1f9d981"
    bundle = root / "turtle_unblur_cloud_v1"

    download(REDS_REPO, REDS_REVISION, reds, ("train_blur.zip", "train_sharp.zip"))
    download(GOPRO_REPO, GOPRO_REVISION, gopro, ("GOPRO_Large.zip",))
    download(
        UNBLUR_REPO, UNBLUR_REVISION, unblur,
        ("Go_pro_defocus_train/**", ".gitattributes"),
    )
    download(
        UNBLUR_REPO, bundle_revision, root,
        ("turtle_unblur_cloud_v1/**",),
    )

    observed_archives = {
        "reds/train_blur.zip": sha256_file(reds / "train_blur.zip"),
        "reds/train_sharp.zip": sha256_file(reds / "train_sharp.zip"),
        "gopro/GOPRO_Large.zip": sha256_file(gopro / "GOPRO_Large.zip"),
    }
    if observed_archives != ARCHIVE_SHA256:
        raise ValueError(f"downloaded archive identity mismatch: {observed_archives}")

    extract_zip(reds / "train_blur.zip", reds / "materialized")
    extract_zip(reds / "train_sharp.zip", reds / "materialized")
    # The GoPro archive also contains test.  Only train members are extracted;
    # no GoPro test image is decoded, hashed or used by this protocol.
    extract_zip(gopro / "GOPRO_Large.zip", gopro / "materialized", ("train/*",))

    required = (
        bundle / "manifests/reds_train.content_addressed.v1.jsonl",
        bundle / "manifests/gopro_large_blur_gamma_train.content_addressed.v1.jsonl",
        bundle / "manifests/unblur_hf_defocus_train.paired_image.v1.jsonl",
        bundle / "replica_blurry_office3/manifests/train.jsonl",
        bundle / "replica_blurry_office3/materialization_audit.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cloud preprocessing bundle is incomplete: {missing}")
    payload = {
        "schema": SCHEMA,
        "status": "materialized_pending_training_preflight",
        "artifact_root": str(root),
        "sources": {
            "reds": {"repo": REDS_REPO, "revision": REDS_REVISION},
            "gopro": {"repo": GOPRO_REPO, "revision": GOPRO_REVISION,
                      "test_archive_bytes_downloaded_opaque": True,
                      "test_members_extracted_or_used": False},
            "unblur": {"repo": UNBLUR_REPO, "revision": UNBLUR_REVISION},
            "preprocessing_bundle": {"repo": UNBLUR_REPO,
                                     "revision": bundle_revision},
        },
        "archive_sha256": observed_archives,
        "bundle_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in required
        ],
        "held_out_data_used": False,
    }
    publish_json(root / "receipts/turtle_unblur_cloud_data.v1.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--bundle-revision", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments.artifact_root, arguments.bundle_revision),
                     indent=2, sort_keys=True))
