#!/usr/bin/env python3
"""Adopt a content-addressed HF BSD archive without opening ZIP members."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import acquire_bsd_3ms24ms as acquisition  # noqa: E402


HF_REPOSITORY = "RuixuanJiang/Video_Deblurring_Datasets"
HF_REVISION = "62a6c6985c0c72caeaf372f821060ac442bdfa4e"
HF_PATH = "BSD/BSD_3ms24ms.zip"
HF_LFS_SHA256 = "db14705307b4bcd75871c5efa832725a985dcdf398a63a737186ef0338a36ad2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=acquisition.DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol, protocol_sha = acquisition.load_protocol(args.protocol)
    partial = args.partial.expanduser().resolve()
    archive = args.archive.expanduser().resolve()
    receipt_path = archive.with_name(f"{archive.name}.acquisition.json")
    if not partial.is_file() or partial.is_symlink():
        raise FileNotFoundError(partial)
    if archive.exists() or receipt_path.exists():
        raise FileExistsError("archive/receipt destination already exists")
    expected_bytes = int(protocol["official_source"]["content_length_bytes"])
    if partial.stat().st_size != expected_bytes:
        raise ValueError("HF archive byte count differs from frozen official object")
    digest = _sha256(partial)
    if digest != HF_LFS_SHA256:
        raise ValueError("HF LFS SHA256 mismatch")
    identity = protocol["remote_zip_identity"]
    with partial.open("rb") as handle:
        handle.seek(int(identity["central_directory_offset"]))
        central = handle.read(int(identity["central_directory_bytes"]))
        if hashlib.sha256(central).hexdigest() != identity["central_directory_sha256"]:
            raise ValueError("HF archive central directory differs from frozen official object")
        for label in ("zip64_eocd", "zip64_locator", "eocd"):
            record = identity[label]
            handle.seek(int(record["offset"]))
            payload = handle.read(int(record["bytes"]))
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"HF archive {label} differs from frozen official object")
    directory = acquisition.audit_central_directory(
        acquisition.parse_central_directory(central), protocol
    )
    receipt = {
        "schema": acquisition.RECEIPT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_protocol": {"path": str(args.protocol.resolve()), "sha256": protocol_sha},
        "official_source": {
            "google_drive_file_id": protocol["official_source"]["google_drive_file_id"],
            "filename": protocol["official_source"]["filename"],
            "content_length_bytes": expected_bytes,
            "remote_zip_central_directory_sha256": identity["central_directory_sha256"],
        },
        "transport": {
            "kind": "third_party_huggingface_content_addressed_mirror",
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "path": HF_PATH,
            "lfs_sha256": HF_LFS_SHA256,
            "claim": "byte-identical archive identity, not an independent license grant",
        },
        "archive": {
            "path": str(archive), "bytes": expected_bytes, "sha256": digest,
            "mode_octal": protocol["storage_policy"]["archive_mode_octal_after_acquisition"],
        },
        "central_directory_audit": directory,
        "exposure_attestation": {
            "opaque_archive_payload_transferred": True,
            "opaque_test_payload_bytes_present_in_archive": True,
            "test_member_payload_opened": False,
            "test_member_payload_decompressed": False,
            "test_image_decoded": False,
            "whole_archive_sha256_is_not_an_individual_member_hash": True,
        },
    }
    os.chmod(partial, int(protocol["storage_policy"]["archive_mode_octal_after_acquisition"], 8))
    os.replace(partial, archive)
    try:
        _write_new(receipt_path, receipt)
    except BaseException:
        os.replace(archive, partial)
        os.chmod(partial, stat.S_IRUSR | stat.S_IWUSR)
        raise
    print(json.dumps({"archive": str(archive), "receipt": str(receipt_path), "sha256": digest}))


if __name__ == "__main__":
    main()
