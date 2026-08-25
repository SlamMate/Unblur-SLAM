#!/usr/bin/env python3
"""Selectively materialize BSD 3ms--24ms train/valid while sealing test.

The official archive bundles all three splits.  This tool verifies the frozen
acquisition receipt and local ZIP metadata, then opens only exact train/valid
PNG members.  It never opens, decompresses, hashes, copies, or decodes a test
member payload.  Output is staged on /srv, audited, and atomically published.
"""

from __future__ import annotations

import argparse
import binascii
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import acquire_bsd_3ms24ms as acquisition  # noqa: E402


SCHEMA = "unblur_slam.bsd_materialization.v1"
SEQUENCE_SCHEMA = "unblur_slam.bsd_paired_video_sequence.v1"
AUDIT_SCHEMA = "unblur_slam.bsd_materialization_audit.v1"
TEST_SEAL_SCHEMA = "unblur_slam.bsd_3ms24ms_test_seal.v1"
AUTHORIZED_SPLITS = ("train", "valid")
CANONICAL_SPLIT = {"train": "train", "valid": "validation"}
MEMBER_PATTERN = re.compile(
    r"^BSD_3ms24ms/(train|valid|test)/(\d{3})/"
    r"(Blur|Sharp)/RGB/(\d{8})\.png$"
)
COPY_BUFFER_BYTES = 4 * 1024 * 1024


class MaterializationError(RuntimeError):
    """Raised when provenance, content, or split isolation fails."""


@dataclasses.dataclass(frozen=True)
class ImageAudit:
    split: str
    sequence: str
    frame_index: int
    role: str
    member: str
    relative_path: str
    bytes: int
    sha256: str
    crc32: str
    width: int
    height: int
    bit_depth: int
    color_type: int
    interlace_method: int


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path, *, progress: bool = False) -> str:
    return acquisition.sha256_file(path, progress=progress)


def _write_new(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise MaterializationError(f"refusing to overwrite {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, mode)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise


def _under_required_root(
    path: Path,
    protocol: Mapping[str, Any],
    *,
    label: str,
    allow_required_root: bool = False,
) -> Path:
    required = Path(
        str(protocol["storage_policy"]["required_filesystem_root"])
    ).resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(required)
    except ValueError as error:
        raise MaterializationError(
            f"{label} must remain under {required}: {resolved}"
        ) from error
    if resolved == required and not allow_required_root:
        raise MaterializationError(f"{label} may not equal broad root {required}")
    return resolved


def _load_receipt(
    receipt_path: Path,
    archive_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> Mapping[str, Any]:
    receipt_path = _under_required_root(
        receipt_path, protocol, label="archive receipt"
    )
    archive_path = _under_required_root(archive_path, protocol, label="archive")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise MaterializationError(f"archive receipt missing/unsafe: {receipt_path}")
    if archive_path.is_symlink() or not archive_path.is_file():
        raise MaterializationError(f"archive missing/unsafe: {archive_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError("archive receipt is invalid JSON") from error
    if not isinstance(receipt, Mapping) or receipt.get("schema") != acquisition.RECEIPT_SCHEMA:
        raise MaterializationError("archive receipt schema drifted")
    if receipt.get("frozen_protocol", {}).get("sha256") != protocol_sha256:
        raise MaterializationError("archive receipt protocol binding drifted")
    observed = receipt.get("archive", {})
    if Path(str(observed.get("path", ""))).resolve() != archive_path:
        raise MaterializationError("archive receipt path does not match requested archive")
    expected_bytes = int(protocol["official_source"]["content_length_bytes"])
    if observed.get("bytes") != expected_bytes or archive_path.stat().st_size != expected_bytes:
        raise MaterializationError("archive/receipt byte count drifted")
    digest = str(observed.get("sha256", "")).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MaterializationError("archive receipt has no valid SHA-256")
    expected_mode = int(
        str(protocol["storage_policy"]["archive_mode_octal_after_acquisition"]), 8
    )
    actual_mode = stat.S_IMODE(archive_path.stat().st_mode)
    if actual_mode != expected_mode:
        raise MaterializationError(
            f"archive mode drifted: {actual_mode:04o} != {expected_mode:04o}"
        )
    return receipt


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise MaterializationError(f"short read for {label}: {len(payload)} != {size}")
    return payload


def audit_local_archive(
    archive_path: Path,
    receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[list[acquisition.CentralEntry], Mapping[str, Any], str]:
    expected_digest = str(receipt["archive"]["sha256"])
    actual_digest = _sha256_file(archive_path, progress=True)
    if actual_digest != expected_digest:
        raise MaterializationError("whole quarantined archive SHA-256 drifted")

    identity = protocol["remote_zip_identity"]
    central_offset = int(identity["central_directory_offset"])
    central_bytes = int(identity["central_directory_bytes"])
    with archive_path.open("rb") as stream:
        stream.seek(central_offset)
        central = _read_exact(stream, central_bytes, "local central directory")
        if hashlib.sha256(central).hexdigest() != identity["central_directory_sha256"]:
            raise MaterializationError("local ZIP central-directory identity drifted")
        for label in ("zip64_eocd", "zip64_locator", "eocd"):
            record = identity[label]
            stream.seek(int(record["offset"]))
            payload = _read_exact(stream, int(record["bytes"]), label)
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise MaterializationError(f"local ZIP {label} identity drifted")
    try:
        entries = acquisition.parse_central_directory(central)
        central_audit = acquisition.audit_central_directory(entries, protocol)
    except acquisition.AcquisitionError as error:
        raise MaterializationError(str(error)) from error
    return entries, central_audit, actual_digest


def _zip_info_map(
    archive: zipfile.ZipFile,
    entries: Sequence[acquisition.CentralEntry],
) -> Mapping[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) != len(entries):
        raise MaterializationError("zipfile and pinned central-directory counts differ")
    result: dict[str, zipfile.ZipInfo] = {}
    central = {entry.name: entry for entry in entries}
    for info in infos:
        if info.filename in result:
            raise MaterializationError(f"duplicate ZipInfo: {info.filename}")
        if info.filename not in central:
            raise MaterializationError(f"ZipInfo missing from pinned index: {info.filename}")
        expected = central[info.filename]
        if (
            info.file_size != expected.uncompressed_bytes
            or info.compress_size != expected.compressed_bytes
            or info.CRC != expected.crc32
        ):
            raise MaterializationError(f"ZipInfo metadata drifted: {info.filename}")
        if info.flag_bits & 0x1:
            raise MaterializationError(f"encrypted ZIP member is forbidden: {info.filename}")
        result[info.filename] = info
    return result


def _png_contract(header: bytes, expected: Mapping[str, Any], member: str) -> Mapping[str, int]:
    if len(header) < 33 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise MaterializationError(f"invalid PNG signature/header: {member}")
    if struct.unpack(">I", header[8:12])[0] != 13 or header[12:16] != b"IHDR":
        raise MaterializationError(f"invalid PNG IHDR: {member}")
    if binascii.crc32(header[12:29]) & 0xFFFFFFFF != struct.unpack(
        ">I", header[29:33]
    )[0]:
        raise MaterializationError(f"PNG IHDR CRC mismatch: {member}")
    width, height = struct.unpack(">II", header[16:24])
    bit_depth, color_type, compression, filtering, interlace = header[24:29]
    observed = {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "compression_method": compression,
        "filter_method": filtering,
        "interlace_method": interlace,
    }
    required = {
        "width": int(expected["width"]),
        "height": int(expected["height"]),
        "bit_depth": int(expected["bit_depth"]),
        "color_type": int(expected["color_type"]),
        "compression_method": 0,
        "filter_method": 0,
        "interlace_method": int(expected["interlace_method"]),
    }
    if observed != required:
        raise MaterializationError(
            f"PNG IHDR contract drifted for {member}: {observed} != {required}"
        )
    return observed


def _extract_image(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
    *,
    split: str,
    sequence: str,
    frame_index: int,
    role: str,
    relative_path: str,
    image_contract: Mapping[str, Any],
) -> ImageAudit:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    digest = hashlib.sha256()
    crc = 0
    count = 0
    header = bytearray()
    try:
        with archive.open(info, "r") as source, os.fdopen(descriptor, "wb") as output:
            while True:
                block = source.read(COPY_BUFFER_BYTES)
                if not block:
                    break
                if len(header) < 33:
                    header.extend(block[: 33 - len(header)])
                output.write(block)
                digest.update(block)
                crc = binascii.crc32(block, crc)
                count += len(block)
            output.flush()
            os.fsync(output.fileno())
        if count != info.file_size:
            raise MaterializationError(
                f"uncompressed byte count drifted for {info.filename}"
            )
        if crc & 0xFFFFFFFF != info.CRC:
            raise MaterializationError(f"ZIP CRC mismatch for {info.filename}")
        png = _png_contract(bytes(header), image_contract, info.filename)
        # The independent materialization audit claims a full authorized-split
        # decode, not merely an IHDR parse.  Pillow is invoked only after the
        # exact member has already passed the train/valid allow-list; test
        # members can never reach this function.
        with Image.open(destination) as decoded:
            decoded.load()
            if decoded.mode != "RGB" or decoded.size != (png["width"], png["height"]):
                raise MaterializationError(
                    f"decoded RGB image contract drifted for {info.filename}: "
                    f"mode={decoded.mode!r}, size={decoded.size!r}"
                )
        os.chmod(destination, 0o444)
        return ImageAudit(
            split=split,
            sequence=sequence,
            frame_index=frame_index,
            role=role,
            member=info.filename,
            relative_path=relative_path,
            bytes=count,
            sha256=digest.hexdigest(),
            crc32=f"{info.CRC:08x}",
            width=png["width"],
            height=png["height"],
            bit_depth=png["bit_depth"],
            color_type=png["color_type"],
            interlace_method=png["interlace_method"],
        )
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            destination.unlink()
        raise


def _sequence_manifest_payloads(
    audits: Sequence[ImageAudit], split: str, *, frames_per_sequence: int
) -> list[Mapping[str, Any]]:
    grouped: dict[str, dict[int, dict[str, ImageAudit]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for audit in audits:
        if audit.split == split:
            grouped[audit.sequence][audit.frame_index][audit.role] = audit
    payloads: list[Mapping[str, Any]] = []
    for sequence in sorted(grouped, key=int):
        expected_indices = list(range(int(frames_per_sequence)))
        if sorted(grouped[sequence]) != expected_indices:
            raise MaterializationError(f"{split}/{sequence} frame indices are not gap-free")
        blurry: list[str] = []
        sharp: list[str] = []
        blurry_sha256: list[str] = []
        sharp_sha256: list[str] = []
        for frame_index in expected_indices:
            roles = grouped[sequence][frame_index]
            if set(roles) != {"Blur", "Sharp"}:
                raise MaterializationError(f"{split}/{sequence}/{frame_index} is unpaired")
            blurry_item = roles["Blur"]
            sharp_item = roles["Sharp"]
            blurry.append(blurry_item.relative_path)
            sharp.append(sharp_item.relative_path)
            blurry_sha256.append(blurry_item.sha256)
            sharp_sha256.append(sharp_item.sha256)
        capture_id = f"bsd_3ms24ms_{split}_{sequence}"
        payloads.append(
            {
                "schema": SEQUENCE_SCHEMA,
                "dataset": "BSD",
                "exposure": "3ms24ms",
                "split": split,
                "capture_id": capture_id,
                "sequence": capture_id,
                "source_sequence": sequence,
                "temporal_order": "gap_free_capture_order",
                "paired_target_alignment": "center_aligned_synchronized",
                "frame_count": len(expected_indices),
                "frame_indices": expected_indices,
                "blurry": blurry,
                "sharp": sharp,
                "blurry_sha256": blurry_sha256,
                "sharp_sha256": sharp_sha256,
            }
        )
    return payloads


def _audit_index_sha256(audits: Sequence[ImageAudit]) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        audits,
        key=lambda value: (
            value.split,
            int(value.sequence),
            value.frame_index,
            value.role,
        ),
    ):
        digest.update(
            (
                f"{item.split}\0{item.sequence}\0{item.frame_index}\0{item.role}\0"
                f"{item.relative_path}\0{item.bytes}\0{item.sha256}\0{item.crc32}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _readonly_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise MaterializationError(f"symlink appeared in staged output: {path}")
        if path.is_file():
            os.chmod(path, 0o444)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        os.chmod(path, 0o555)
    os.chmod(root, 0o555)


def materialize(
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    archive_path: Path,
    receipt_path: Path,
    output_path: Path,
) -> Path:
    output = _under_required_root(output_path, protocol, label="materialized output")
    if output.exists() or output.is_symlink():
        raise MaterializationError(f"refusing to overwrite materialized output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt = _load_receipt(
        receipt_path, archive_path, protocol, protocol_sha256
    )
    archive = Path(str(receipt["archive"]["path"])).resolve()
    entries, central_audit, archive_sha256 = audit_local_archive(
        archive, receipt, protocol
    )

    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    if stage.exists() or stage.is_symlink():
        raise MaterializationError(f"staging path already exists: {stage}")
    stage.mkdir(mode=0o700)
    started = time.monotonic()
    test_member_open_calls = 0
    authorized_open_calls = 0
    audits: list[ImageAudit] = []
    try:
        with zipfile.ZipFile(archive, "r") as source:
            info_by_name = _zip_info_map(source, entries)
            selected: list[tuple[str, str, str, int, str]] = []
            for entry in entries:
                match = MEMBER_PATTERN.fullmatch(entry.name)
                if not match:
                    continue
                split, sequence, role, frame = match.groups()
                if split in AUTHORIZED_SPLITS:
                    selected.append((split, sequence, role, int(frame), entry.name))
            expected_selected = sum(
                int(protocol["remote_zip_identity"]["splits"][split]["png_members"])
                for split in AUTHORIZED_SPLITS
            )
            if len(selected) != expected_selected:
                raise MaterializationError("authorized member selection count drifted")
            selected.sort(key=lambda row: (AUTHORIZED_SPLITS.index(row[0]), int(row[1]), row[3], row[2]))
            image_contract = protocol["remote_zip_identity"]["image_contract"]
            for index, (split, sequence, role, frame_index, member) in enumerate(
                selected, start=1
            ):
                if split == "test" or "/test/" in member:
                    test_member_open_calls += 1
                    raise MaterializationError("internal error: test member selected")
                canonical_split = CANONICAL_SPLIT[split]
                relative = (
                    Path(canonical_split)
                    / sequence
                    / role
                    / "RGB"
                    / f"{frame_index:08d}.png"
                )
                relative_string = relative.as_posix()
                audit = _extract_image(
                    source,
                    info_by_name[member],
                    stage / relative,
                    split=canonical_split,
                    sequence=sequence,
                    frame_index=frame_index,
                    role=role,
                    relative_path=relative_string,
                    image_contract=image_contract,
                )
                authorized_open_calls += 1
                audits.append(audit)
                if index % 500 == 0:
                    print(
                        f"materialize: {index}/{len(selected)} authorized PNGs",
                        file=sys.stderr,
                        flush=True,
                    )

        if test_member_open_calls != 0:
            raise MaterializationError("test member payload was opened")
        if authorized_open_calls != len(audits):
            raise MaterializationError("authorized member open-call accounting drifted")
        audit_paths = {item.relative_path for item in audits}
        if len(audit_paths) != len(audits):
            raise MaterializationError("duplicate materialized path")
        if any(path.startswith("test/") or "/test/" in path for path in audit_paths):
            raise MaterializationError("test path appeared in materialized output")

        hashes_by_split = {
            split: {item.sha256 for item in audits if item.split == split}
            for split in ("train", "validation")
        }
        overlap = hashes_by_split["train"] & hashes_by_split["validation"]
        if overlap:
            raise MaterializationError(
                f"train/valid content overlap detected ({len(overlap)} SHA-256 values)"
            )

        manifests: dict[str, Mapping[str, Any]] = {}
        for source_split in AUTHORIZED_SPLITS:
            split = CANONICAL_SPLIT[source_split]
            frames_per_sequence = int(
                protocol["remote_zip_identity"]["splits"][source_split][
                    "frames_per_sequence"
                ]
            )
            payloads = _sequence_manifest_payloads(
                audits, split, frames_per_sequence=frames_per_sequence
            )
            manifest_path = stage / "manifests" / f"{split}.jsonl"
            manifest_bytes = b"".join(_canonical_bytes(payload) for payload in payloads)
            _write_new(manifest_path, manifest_bytes)
            split_audits = [item for item in audits if item.split == split]
            manifests[split] = {
                "path": f"manifests/{split}.jsonl",
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "sequence_records": len(payloads),
                "pairs": sum(payload["frame_count"] for payload in payloads),
                "png_files": len(split_audits),
                "materialized_bytes": sum(item.bytes for item in split_audits),
            }

        identical_pairs = 0
        pair_roles: dict[tuple[str, str, int], dict[str, str]] = defaultdict(dict)
        for item in audits:
            pair_roles[(item.split, item.sequence, item.frame_index)][item.role] = item.sha256
        for roles in pair_roles.values():
            if roles.get("Blur") == roles.get("Sharp"):
                identical_pairs += 1
        if identical_pairs:
            raise MaterializationError(
                f"identical blurry/sharp pairs detected: {identical_pairs}"
            )

        seal = {
            "schema": TEST_SEAL_SCHEMA,
            "frozen_protocol_sha256": protocol_sha256,
            "sealed_split": "test",
            "test_png_member_count_from_central_directory": protocol[
                "remote_zip_identity"
            ]["splits"]["test"]["png_members"],
            "test_png_index_sha256_from_central_directory": protocol[
                "remote_zip_identity"
            ]["splits"]["test"]["png_index_sha256"],
            "test_member_payload_open_calls": 0,
            "test_member_payload_bytes_read": 0,
            "test_member_payload_hashes_computed": 0,
            "test_images_decoded_or_displayed": 0,
            "test_paths_materialized": 0,
            "unlock_condition": (
                "A separately frozen evaluation protocol is required before any "
                "test member payload access."
            ),
        }
        _write_new(stage / "TEST_SEALED.json", _canonical_bytes(seal))

        png_files = [path for path in stage.rglob("*.png") if path.is_file()]
        if len(png_files) != len(audits):
            raise MaterializationError("published PNG inventory count drifted")
        if (stage / "test").exists() or any("test" in path.parts for path in png_files):
            raise MaterializationError("test output path exists")
        if any(path.is_symlink() for path in stage.rglob("*")):
            raise MaterializationError("staged output contains a symlink")

        materialization_audit = {
            "schema": AUDIT_SCHEMA,
            "status": "pass",
            "train_validation_only": True,
            "image_contract": {
                "resolution_width_height": [
                    int(protocol["remote_zip_identity"]["image_contract"]["width"]),
                    int(protocol["remote_zip_identity"]["image_contract"]["height"]),
                ],
                "rgb_channels": 3,
                "bit_depth": int(
                    protocol["remote_zip_identity"]["image_contract"]["bit_depth"]
                ),
                "all_train_validation_assets_hashed": True,
                "all_train_validation_assets_decoded": True,
                "all_zip_crc32_verified": True,
            },
            "pairing_contract": {
                "frame_indices_gap_free_from_zero": True,
                "blurry_sharp_basenames_equal": True,
                "identical_blur_sharp_pair_count": identical_pairs,
            },
            "disjoint_audit": {
                "capture_ids": True,
                "paths": True,
                "content_hashes": True,
            },
            "counts": {
                "train": manifests["train"],
                "validation": manifests["validation"],
                "materialized_png_files": len(audits),
                "authorized_member_open_calls": authorized_open_calls,
            },
            "test_audit": {
                "local_pixel_paths": 0,
                "pixels_opened": False,
                "images_decoded": False,
                "model_outputs_computed": False,
                "metrics_computed": False,
                "member_payload_open_calls": 0,
                "member_payload_bytes_read": 0,
            },
            "image_audit_index_sha256": _audit_index_sha256(audits),
        }
        audit_path = stage / "materialization_audit.json"
        _write_new(audit_path, _canonical_bytes(materialization_audit))

        dataset_manifest = {
            "schema": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "BSD",
            "exposure_setting": "3ms24ms",
            "blur_origin": "real_camera_long_exposure",
            "synthetic_high_fps_average": False,
            "frozen_protocol": {
                "path": str(acquisition.DEFAULT_PROTOCOL),
                "sha256": protocol_sha256,
            },
            "official_archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": archive_sha256,
                "receipt": str(receipt_path.resolve()),
                "receipt_sha256": _sha256_file(receipt_path.resolve()),
                "central_directory_sha256": protocol["remote_zip_identity"]
                ["central_directory_sha256"],
            },
            "authorized_splits": ["train", "validation"],
            "splits": {
                split: {
                    "sequences": int(manifests[split]["sequence_records"]),
                    "frames": int(manifests[split]["pairs"]),
                }
                for split in ("train", "validation")
            },
            "canonical_manifests": {
                split: {
                    "path": manifests[split]["path"],
                    "sha256": manifests[split]["sha256"],
                    "schema": SEQUENCE_SCHEMA,
                    "sequence_records": manifests[split]["sequence_records"],
                    "pairs": manifests[split]["pairs"],
                    "paths_relative_to": "dataset_root",
                }
                for split in ("train", "validation")
            },
            "materialization_audit_artifact": {
                "path": "materialization_audit.json",
                "sha256": hashlib.sha256(
                    _canonical_bytes(materialization_audit)
                ).hexdigest(),
                "schema": AUDIT_SCHEMA,
            },
            "image_contract": {
                "png_crc_verified_per_file": True,
                "png_sha256_computed_per_file": True,
                "width": protocol["remote_zip_identity"]["image_contract"]["width"],
                "height": protocol["remote_zip_identity"]["image_contract"]["height"],
                "bit_depth": protocol["remote_zip_identity"]["image_contract"]
                ["bit_depth"],
                "color_model": "RGB",
                "interlace_method": 0,
            },
            "materialization_audit": {
                "authorized_member_open_calls": authorized_open_calls,
                "materialized_png_files": len(audits),
                "materialized_bytes": sum(item.bytes for item in audits),
                "image_audit_index_sha256": _audit_index_sha256(audits),
                "identical_blur_sharp_pair_count": identical_pairs,
                "train_valid_path_sets_disjoint": True,
                "train_valid_sequence_ids_disjoint": True,
                "train_valid_content_sha256_sets_disjoint": True,
                "train_valid_content_sha256_overlap_count": 0,
                "test_content_disjointness_not_claimed_without_payload_access": True,
                "central_directory_audit": central_audit,
            },
            "test_seal": seal,
            "test_disclosure": {
                "pixel_paths_materialized": False,
                "pixels_opened": False,
                "images_decoded": False,
                "model_outputs_computed": False,
                "metrics_computed": False,
            },
            "test_split_available_to_trainer": False,
            "macos_metadata_materialized": False,
            "raw_members_materialized": False,
            "license_audit": protocol["license_audit"],
            "elapsed_seconds": time.monotonic() - started,
        }
        manifest_path = stage / "dataset_manifest.json"
        _write_new(manifest_path, _canonical_bytes(dataset_manifest))
        _readonly_tree(stage)
        os.replace(stage, output)
        return output / "dataset_manifest.json"
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            # Staging remains owner-only until the final success path, so this
            # exact PID-scoped directory is safe to remove on failure.
            shutil.rmtree(stage)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=acquisition.DEFAULT_PROTOCOL
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="defaults to <frozen quarantine>/BSD_3ms24ms.zip",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="defaults to <archive>.acquisition.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to the frozen materialized /srv directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol, protocol_sha256 = acquisition.load_protocol(args.protocol)
    quarantine = Path(protocol["storage_policy"]["default_quarantine_directory"])
    archive = args.archive or quarantine / protocol["official_source"]["filename"]
    receipt = args.receipt or archive.with_name(f"{archive.name}.acquisition.json")
    output = args.output or Path(
        protocol["storage_policy"]["default_materialized_directory"]
    )
    manifest = materialize(
        protocol,
        protocol_sha256,
        archive.resolve(),
        receipt.resolve(),
        output.resolve(),
    )
    print(
        json.dumps(
            {
                "dataset_manifest": str(manifest),
                "dataset_manifest_sha256": _sha256_file(manifest),
                "test_member_payload_open_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MaterializationError, acquisition.AcquisitionError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
