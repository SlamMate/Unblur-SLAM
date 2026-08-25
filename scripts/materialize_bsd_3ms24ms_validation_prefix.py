#!/usr/bin/env python3
"""Materialize only the complete BSD validation split in a frozen ZIP prefix.

This is deliberately not a full BSD materializer and cannot authorize
training.  The publisher ordered validation before the first test payload in
the currently quarantined partial ZIP.  Local headers and signed data
descriptors are parsed only up to a frozen byte boundary; no test member
payload is opened, decompressed, copied, decoded, or individually hashed.
"""

from __future__ import annotations

import argparse
import binascii
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import mmap
import os
import re
import shutil
import stat
import struct
import sys
import time
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import acquire_bsd_3ms24ms as acquisition  # noqa: E402


SEQUENCE_SCHEMA = "unblur_slam.bsd_paired_video_sequence.v1"
DATASET_SCHEMA = "unblur_slam.bsd_materialization.v1"
AUDIT_SCHEMA = "unblur_slam.bsd_materialization_audit.v1"
RECEIPT_SCHEMA = "unblur_slam.bsd_validation_prefix_receipt.v1"
DEFAULT_OUTPUT = Path(
    "/srv/szha0669/unblur-slam/bsd_3ms24ms_validation_prefix_v1"
)
LOCAL_HEADER = struct.Struct("<4s5H3L2H")
DATA_DESCRIPTOR = struct.Struct("<4sLLL")
LOCAL_SIGNATURE = b"PK\x03\x04"
DESCRIPTOR_SIGNATURE = b"PK\x07\x08"
COPY_BYTES = 4 * 1024 * 1024
VALID_PATTERN = re.compile(
    r"^BSD_3ms24ms/valid/(\d{3})/(Blur|Sharp)/RGB/(\d{8})\.png$"
)
TEST_PATTERN = re.compile(
    r"^BSD_3ms24ms/test/(\d{3})/(Blur|Sharp)/RGB/(\d{8})\.png$"
)


class PrefixMaterializationError(RuntimeError):
    """Raised when a frozen prefix, split, image, or seal invariant fails."""


@dataclasses.dataclass(frozen=True)
class PrefixContract:
    protocol_sha256: str
    partial_path: Path
    prefix_bytes: int
    opaque_prefix_sha256: str
    scan_limit_exclusive: int
    complete_entries_before_test: int
    first_test_member: str
    first_test_header_start: int
    first_test_payload_start: int
    valid_png_members: int
    valid_sequences: int
    valid_pairs: int
    valid_frames_per_sequence: int
    valid_compressed_bytes: int
    valid_uncompressed_bytes: int
    valid_index_sha256: str
    first_valid_member: str
    first_valid_header_start: int
    terminal_valid_member: str
    terminal_valid_header_start: int
    terminal_valid_compressed_end_exclusive: int
    terminal_valid_record_end_exclusive: int


FROZEN_PREFIX = PrefixContract(
    protocol_sha256=acquisition.EXPECTED_PROTOCOL_SHA256,
    partial_path=Path(
        "/srv/szha0669/unblur-slam/quarantine/bsd_3ms24ms/"
        "BSD_3ms24ms.zip.partial"
    ),
    prefix_bytes=5_832_241_152,
    opaque_prefix_sha256=(
        "274bb7bb75b389e28938531a1a4b3b91"
        "d42151a601d680b54321577f8ecb1775"
    ),
    scan_limit_exclusive=1_552_633_294,
    complete_entries_before_test=9_010,
    first_test_member="BSD_3ms24ms/test/014/Blur/RGB/00000038.png",
    first_test_header_start=1_552_633_190,
    first_test_payload_start=1_552_633_294,
    valid_png_members=4_000,
    valid_sequences=20,
    valid_pairs=2_000,
    valid_frames_per_sequence=100,
    valid_compressed_bytes=1_550_905_034,
    valid_uncompressed_bytes=1_550_567_865,
    valid_index_sha256=(
        "8ed9bc17a5d17937fa747a3ddc2ef858"
        "e9bf6811734b60be18bcc04116f3ff7a"
    ),
    first_valid_member="BSD_3ms24ms/valid/015/Blur/RGB/00000038.png",
    first_valid_header_start=172_156,
    terminal_valid_member="BSD_3ms24ms/valid/122/Sharp/RGB/00000021.png",
    terminal_valid_header_start=1_552_286_824,
    terminal_valid_compressed_end_exclusive=1_552_632_906,
    terminal_valid_record_end_exclusive=1_552_632_922,
)


@dataclasses.dataclass(frozen=True)
class LocalEntry:
    name: str
    flags: int
    method: int
    crc32: int
    compressed_bytes: int
    uncompressed_bytes: int
    header_start: int
    compressed_start: int
    compressed_end_exclusive: int
    record_end_exclusive: int


@dataclasses.dataclass(frozen=True)
class ImageRecord:
    sequence: str
    role: str
    frame_index: int
    relative_path: str
    sha256: str
    bytes: int
    crc32: str


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_fd(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(COPY_BYTES, size - offset), offset)
        if not block:
            raise PrefixMaterializationError("opaque prefix SHA read ended early")
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(COPY_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
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


def _required_path(path: Path, protocol: Mapping[str, Any], label: str) -> Path:
    root = Path(protocol["storage_policy"]["required_filesystem_root"]).resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PrefixMaterializationError(
            f"{label} must remain under {root}: {resolved}"
        ) from error
    if resolved == root:
        raise PrefixMaterializationError(f"{label} may not equal broad root {root}")
    return resolved


def _local_header(
    source: mmap.mmap, offset: int, read_limit: int
) -> tuple[tuple[Any, ...], str, int]:
    if offset < 0 or offset + LOCAL_HEADER.size > read_limit:
        raise PrefixMaterializationError(f"truncated local header at {offset}")
    values = LOCAL_HEADER.unpack(source[offset : offset + LOCAL_HEADER.size])
    if values[0] != LOCAL_SIGNATURE:
        raise PrefixMaterializationError(f"invalid local-header signature at {offset}")
    flags = int(values[2])
    name_bytes = int(values[9])
    extra_bytes = int(values[10])
    metadata_end = offset + LOCAL_HEADER.size + name_bytes + extra_bytes
    if name_bytes <= 0 or name_bytes > 4096 or metadata_end > read_limit:
        raise PrefixMaterializationError(f"unsafe local-header lengths at {offset}")
    raw_name = bytes(
        source[offset + LOCAL_HEADER.size : offset + LOCAL_HEADER.size + name_bytes]
    )
    try:
        name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
    except UnicodeDecodeError as error:
        raise PrefixMaterializationError("invalid local filename encoding") from error
    parts = [part for part in name.split("/") if part]
    if (
        not name.startswith(("BSD_3ms24ms/", "__MACOSX/"))
        or "\x00" in name
        or "\\" in name
        or ".." in parts
    ):
        raise PrefixMaterializationError(f"unsafe local member name: {name!r}")
    return values, name, metadata_end


def _descriptor_before_next_header(
    source: mmap.mmap,
    compressed_start: int,
    local_uncompressed_bytes: int,
    read_limit: int,
) -> tuple[int, int, int, int]:
    cursor = compressed_start
    while True:
        descriptor_start = source.find(DESCRIPTOR_SIGNATURE, cursor, read_limit)
        if descriptor_start < 0 or descriptor_start + DATA_DESCRIPTOR.size > read_limit:
            raise PrefixMaterializationError(
                f"no complete signed data descriptor before seal at {compressed_start}"
            )
        signature, crc32, compressed_bytes, uncompressed_bytes = DATA_DESCRIPTOR.unpack(
            source[descriptor_start : descriptor_start + DATA_DESCRIPTOR.size]
        )
        next_header = descriptor_start + DATA_DESCRIPTOR.size
        if (
            signature == DESCRIPTOR_SIGNATURE
            and compressed_bytes == descriptor_start - compressed_start
            and local_uncompressed_bytes in (0, uncompressed_bytes)
        ):
            try:
                _local_header(source, next_header, read_limit)
            except PrefixMaterializationError:
                pass
            else:
                return next_header, crc32, compressed_bytes, uncompressed_bytes
        cursor = descriptor_start + 1


def _valid_index_sha256(entries: Sequence[LocalEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.name):
        digest.update(
            (
                f"{entry.name}\0{entry.compressed_bytes}\0"
                f"{entry.uncompressed_bytes}\0{entry.crc32:08x}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def scan_validation_prefix(
    descriptor: int,
    contract: PrefixContract,
    protocol: Mapping[str, Any],
) -> tuple[list[LocalEntry], Mapping[str, Any]]:
    if contract.scan_limit_exclusive != contract.first_test_payload_start:
        raise PrefixMaterializationError("scan limit is not the frozen test seal")
    source = mmap.mmap(descriptor, contract.prefix_bytes, access=mmap.ACCESS_READ)
    entries: list[LocalEntry] = []
    valid: list[LocalEntry] = []
    cursor = 0
    test_header_reads = 0
    try:
        while cursor < contract.scan_limit_exclusive:
            values, name, compressed_start = _local_header(
                source, cursor, contract.scan_limit_exclusive
            )
            flags = int(values[2])
            method = int(values[3])
            if flags & 0x1:
                raise PrefixMaterializationError(f"encrypted member: {name}")
            if method not in (0, 8):
                raise PrefixMaterializationError(f"unsupported ZIP method {method}: {name}")
            if TEST_PATTERN.fullmatch(name):
                test_header_reads += 1
                if (
                    name != contract.first_test_member
                    or cursor != contract.first_test_header_start
                    or compressed_start != contract.first_test_payload_start
                ):
                    raise PrefixMaterializationError("first test seal identity drifted")
                break

            if flags & 0x8:
                next_header, crc32, compressed_bytes, uncompressed_bytes = (
                    _descriptor_before_next_header(
                        source,
                        compressed_start,
                        int(values[8]),
                        contract.scan_limit_exclusive,
                    )
                )
                compressed_end = compressed_start + compressed_bytes
                record_end = next_header
            else:
                crc32 = int(values[6])
                compressed_bytes = int(values[7])
                uncompressed_bytes = int(values[8])
                compressed_end = compressed_start + compressed_bytes
                record_end = compressed_end
                if record_end > contract.scan_limit_exclusive:
                    raise PrefixMaterializationError(
                        f"member crosses frozen test seal: {name}"
                    )
            entry = LocalEntry(
                name=name,
                flags=flags,
                method=method,
                crc32=crc32,
                compressed_bytes=compressed_bytes,
                uncompressed_bytes=uncompressed_bytes,
                header_start=cursor,
                compressed_start=compressed_start,
                compressed_end_exclusive=compressed_end,
                record_end_exclusive=record_end,
            )
            entries.append(entry)
            if VALID_PATTERN.fullmatch(name):
                valid.append(entry)
            elif "/train/" in name and name.endswith(".png"):
                raise PrefixMaterializationError("train PNG appeared in validation prefix")
            cursor = record_end
        else:
            raise PrefixMaterializationError("first sealed test header was not reached")
    finally:
        source.close()

    if test_header_reads != 1 or cursor != contract.first_test_header_start:
        raise PrefixMaterializationError("test-header accounting drifted")
    if len(entries) != contract.complete_entries_before_test:
        raise PrefixMaterializationError("complete local-entry count drifted")
    if len(valid) != contract.valid_png_members:
        raise PrefixMaterializationError("validation PNG count drifted")
    if sum(item.compressed_bytes for item in valid) != contract.valid_compressed_bytes:
        raise PrefixMaterializationError("validation compressed-byte count drifted")
    if sum(item.uncompressed_bytes for item in valid) != contract.valid_uncompressed_bytes:
        raise PrefixMaterializationError("validation uncompressed-byte count drifted")
    index_sha256 = _valid_index_sha256(valid)
    if index_sha256 != contract.valid_index_sha256:
        raise PrefixMaterializationError("validation local index SHA-256 drifted")
    if not valid:
        raise PrefixMaterializationError("validation selection is empty")
    first, terminal = valid[0], valid[-1]
    if (
        first.name != contract.first_valid_member
        or first.header_start != contract.first_valid_header_start
        or terminal.name != contract.terminal_valid_member
        or terminal.header_start != contract.terminal_valid_header_start
        or terminal.compressed_end_exclusive
        != contract.terminal_valid_compressed_end_exclusive
        or terminal.record_end_exclusive
        != contract.terminal_valid_record_end_exclusive
    ):
        raise PrefixMaterializationError("validation terminal-offset identity drifted")

    paired: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for entry in valid:
        match = VALID_PATTERN.fullmatch(entry.name)
        assert match is not None
        sequence, role, frame = match.groups()
        paired[sequence][int(frame)].add(role)
    if len(paired) != contract.valid_sequences:
        raise PrefixMaterializationError("validation sequence count drifted")
    expected_frames = set(range(contract.valid_frames_per_sequence))
    for sequence, frames in paired.items():
        if set(frames) != expected_frames or any(
            roles != {"Blur", "Sharp"} for roles in frames.values()
        ):
            raise PrefixMaterializationError(f"validation pairing drifted: {sequence}")
    if sum(len(frames) for frames in paired.values()) != contract.valid_pairs:
        raise PrefixMaterializationError("validation pair count drifted")
    return valid, {
        "complete_local_entries_before_test": len(entries),
        "validation_local_index_sha256": index_sha256,
        "first_test_member_header_reads": test_header_reads,
        "test_member_payload_bytes_read": 0,
    }


def _png_header_contract(
    header: bytes, expected: Mapping[str, Any], member: str
) -> None:
    if len(header) < 33 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise PrefixMaterializationError(f"invalid PNG signature: {member}")
    if struct.unpack(">I", header[8:12])[0] != 13 or header[12:16] != b"IHDR":
        raise PrefixMaterializationError(f"invalid PNG IHDR: {member}")
    if binascii.crc32(header[12:29]) & 0xFFFFFFFF != struct.unpack(
        ">I", header[29:33]
    )[0]:
        raise PrefixMaterializationError(f"PNG IHDR CRC mismatch: {member}")
    width, height = struct.unpack(">II", header[16:24])
    bit_depth, color_type, compression, filtering, interlace = header[24:29]
    observed = (width, height, bit_depth, color_type, compression, filtering, interlace)
    required = (
        int(expected["width"]),
        int(expected["height"]),
        int(expected["bit_depth"]),
        int(expected["color_type"]),
        0,
        0,
        int(expected["interlace_method"]),
    )
    if observed != required:
        raise PrefixMaterializationError(f"PNG contract drifted: {member}")


def _extract_validation_image(
    source_fd: int,
    entry: LocalEntry,
    destination: Path,
    image_contract: Mapping[str, Any],
) -> ImageRecord:
    match = VALID_PATTERN.fullmatch(entry.name)
    if match is None:
        raise PrefixMaterializationError("non-validation entry selected")
    sequence, role, frame = match.groups()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    output_fd = os.open(destination, flags, 0o600)
    digest = hashlib.sha256()
    crc32 = 0
    decoded_bytes = 0
    png_header = bytearray()
    decompressor = zlib.decompressobj(-15) if entry.method == 8 else None
    source_offset = entry.compressed_start
    remaining = entry.compressed_bytes
    try:
        with os.fdopen(output_fd, "wb") as output:
            while remaining:
                block = os.pread(source_fd, min(COPY_BYTES, remaining), source_offset)
                if not block:
                    raise PrefixMaterializationError(
                        f"short compressed read: {entry.name}"
                    )
                source_offset += len(block)
                remaining -= len(block)
                payload = decompressor.decompress(block) if decompressor else block
                if decompressor and (decompressor.unused_data or decompressor.unconsumed_tail):
                    raise PrefixMaterializationError(
                        f"deflate stream exceeded pinned size: {entry.name}"
                    )
                if len(png_header) < 33:
                    png_header.extend(payload[: 33 - len(png_header)])
                output.write(payload)
                digest.update(payload)
                crc32 = binascii.crc32(payload, crc32)
                decoded_bytes += len(payload)
            if decompressor:
                tail = decompressor.flush()
                if tail:
                    if len(png_header) < 33:
                        png_header.extend(tail[: 33 - len(png_header)])
                    output.write(tail)
                    digest.update(tail)
                    crc32 = binascii.crc32(tail, crc32)
                    decoded_bytes += len(tail)
                if not decompressor.eof:
                    raise PrefixMaterializationError(
                        f"truncated deflate stream: {entry.name}"
                    )
            output.flush()
            os.fsync(output.fileno())
        if decoded_bytes != entry.uncompressed_bytes:
            raise PrefixMaterializationError(
                f"uncompressed size drifted: {entry.name}"
            )
        if crc32 & 0xFFFFFFFF != entry.crc32:
            raise PrefixMaterializationError(f"ZIP CRC mismatch: {entry.name}")
        _png_header_contract(bytes(png_header), image_contract, entry.name)
        with Image.open(destination) as image:
            image_format = image.format
            image.load()
            if (
                image_format != "PNG"
                or image.mode != "RGB"
                or image.size
                != (int(image_contract["width"]), int(image_contract["height"]))
            ):
                raise PrefixMaterializationError(
                    f"decoded PNG/RGB contract drifted: {entry.name}"
                )
        os.chmod(destination, 0o444)
        return ImageRecord(
            sequence=sequence,
            role=role,
            frame_index=int(frame),
            relative_path=destination.relative_to(destination.parents[4]).as_posix(),
            sha256=digest.hexdigest(),
            bytes=decoded_bytes,
            crc32=f"{entry.crc32:08x}",
        )
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            destination.unlink()
        raise


def _sequence_manifest(records: Sequence[ImageRecord], contract: PrefixContract) -> bytes:
    grouped: dict[str, dict[int, dict[str, ImageRecord]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        grouped[record.sequence][record.frame_index][record.role] = record
    lines: list[bytes] = []
    for sequence in sorted(grouped, key=int):
        frames = grouped[sequence]
        indices = list(range(contract.valid_frames_per_sequence))
        if sorted(frames) != indices:
            raise PrefixMaterializationError(f"manifest frames drifted: {sequence}")
        blurry: list[str] = []
        sharp: list[str] = []
        blurry_sha256: list[str] = []
        sharp_sha256: list[str] = []
        for frame in indices:
            if set(frames[frame]) != {"Blur", "Sharp"}:
                raise PrefixMaterializationError(f"manifest pair drifted: {sequence}/{frame}")
            blur = frames[frame]["Blur"]
            target = frames[frame]["Sharp"]
            if blur.sha256 == target.sha256:
                raise PrefixMaterializationError(
                    f"identical blurry/sharp content: {sequence}/{frame}"
                )
            blurry.append(blur.relative_path)
            sharp.append(target.relative_path)
            blurry_sha256.append(blur.sha256)
            sharp_sha256.append(target.sha256)
        identity = f"bsd_3ms24ms_validation_{sequence}"
        lines.append(
            _canonical_bytes(
                {
                    "schema": SEQUENCE_SCHEMA,
                    "dataset": "BSD",
                    "exposure": "3ms24ms",
                    "split": "validation",
                    "capture_id": identity,
                    "sequence": identity,
                    "source_sequence": sequence,
                    "temporal_order": "gap_free_capture_order",
                    "paired_target_alignment": "center_aligned_synchronized",
                    "frame_count": len(indices),
                    "frame_indices": indices,
                    "blurry": blurry,
                    "sharp": sharp,
                    "blurry_sha256": blurry_sha256,
                    "sharp_sha256": sharp_sha256,
                }
            )
        )
    if len(lines) != contract.valid_sequences:
        raise PrefixMaterializationError("manifest sequence count drifted")
    return b"".join(lines)


def _readonly_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PrefixMaterializationError(f"symlink in staged tree: {path}")
        if path.is_file():
            os.chmod(path, 0o444)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o555)
    os.chmod(root, 0o555)


def materialize_validation_prefix(
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    contract: PrefixContract,
    partial_path: Path,
    output_path: Path,
) -> Path:
    partial = _required_path(partial_path, protocol, "partial ZIP")
    output = _required_path(output_path, protocol, "validation-prefix output")
    if partial != contract.partial_path.expanduser().resolve():
        raise PrefixMaterializationError("partial path differs from frozen prefix")
    if protocol_sha256 != contract.protocol_sha256:
        raise PrefixMaterializationError("frozen protocol binding drifted")
    expected = protocol["remote_zip_identity"]["splits"]["valid"]
    if (
        int(expected["png_members"]) != contract.valid_png_members
        or int(expected["sequences"]) != contract.valid_sequences
        or int(expected["pairs"]) != contract.valid_pairs
        or int(expected["frames_per_sequence"]) != contract.valid_frames_per_sequence
        or int(expected["compressed_png_bytes"]) != contract.valid_compressed_bytes
        or int(expected["uncompressed_png_bytes"]) != contract.valid_uncompressed_bytes
        or expected["png_index_sha256"] != contract.valid_index_sha256
    ):
        raise PrefixMaterializationError("prefix contract disagrees with frozen protocol")
    if output.exists() or output.is_symlink():
        raise PrefixMaterializationError(f"refusing to overwrite output: {output}")
    if partial.is_symlink() or not partial.is_file():
        raise PrefixMaterializationError(f"partial missing/unsafe: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    lock_path = partial.parent / ".acquisition.lock"
    lock_fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o600)
    source_fd = -1
    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PrefixMaterializationError("acquisition is currently writing partial") from error
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        source_fd = os.open(partial, flags)
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != contract.prefix_bytes:
            raise PrefixMaterializationError("partial byte length/type drifted")
        opaque_sha256 = _sha256_fd(source_fd, contract.prefix_bytes)
        if opaque_sha256 != contract.opaque_prefix_sha256:
            raise PrefixMaterializationError("opaque partial-prefix SHA-256 drifted")
        valid_entries, scan_audit = scan_validation_prefix(
            source_fd, contract, protocol
        )

        if stage.exists() or stage.is_symlink():
            raise PrefixMaterializationError(f"staging path exists: {stage}")
        stage.mkdir(mode=0o700)
        started = time.monotonic()
        image_contract = protocol["remote_zip_identity"]["image_contract"]
        records: list[ImageRecord] = []
        selected = sorted(
            valid_entries,
            key=lambda item: (
                int(VALID_PATTERN.fullmatch(item.name).group(1)),
                int(VALID_PATTERN.fullmatch(item.name).group(3)),
                VALID_PATTERN.fullmatch(item.name).group(2),
            ),
        )
        for index, entry in enumerate(selected, start=1):
            match = VALID_PATTERN.fullmatch(entry.name)
            assert match is not None
            sequence, role, frame = match.groups()
            destination = (
                stage
                / "validation"
                / sequence
                / role
                / "RGB"
                / f"{int(frame):08d}.png"
            )
            records.append(
                _extract_validation_image(
                    source_fd, entry, destination, image_contract
                )
            )
            if index % 500 == 0:
                print(
                    f"validation-prefix: {index}/{len(selected)} PNGs",
                    file=sys.stderr,
                    flush=True,
                )
        if len(records) != contract.valid_png_members:
            raise PrefixMaterializationError("extracted validation count drifted")
        all_hashes = [record.sha256 for record in records]
        if len(set(all_hashes)) != len(all_hashes):
            raise PrefixMaterializationError("duplicate validation content SHA-256")

        manifest_bytes = _sequence_manifest(records, contract)
        manifest_relative = Path("manifests/validation.jsonl")
        _write_new(stage / manifest_relative, manifest_bytes)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

        audit = {
            "schema": AUDIT_SCHEMA,
            "status": "pass",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "validation_only_from_frozen_partial_prefix",
            "validation_only": True,
            "train_materialized": False,
            "full_dataset_ready": False,
            "image_contract": {
                "resolution_width_height": [
                    int(image_contract["width"]),
                    int(image_contract["height"]),
                ],
                "rgb_channels": 3,
                "bit_depth": int(image_contract["bit_depth"]),
                "all_validation_assets_zip_crc_verified": True,
                "all_validation_assets_sha256_recomputed": True,
                "all_validation_assets_decoded": True,
            },
            "counts": {
                "validation_sequences": contract.valid_sequences,
                "validation_pairs": contract.valid_pairs,
                "validation_png_files": len(records),
                "validation_member_payload_open_calls": len(records),
                "validation_compressed_payload_bytes_read": sum(
                    item.compressed_bytes for item in valid_entries
                ),
            },
            "local_prefix_scan": dict(scan_audit),
            "test_audit": {
                "local_pixel_paths": 0,
                "member_payload_open_calls": 0,
                "member_payload_bytes_read": 0,
                "member_payload_hashes_computed": 0,
                "pixels_opened": False,
                "images_decoded": False,
                "model_outputs_computed": False,
                "metrics_computed": False,
            },
            "opaque_transport_identity": {
                "partial_path": str(partial),
                "partial_prefix_bytes": contract.prefix_bytes,
                "partial_prefix_sha256": opaque_sha256,
                "whole_prefix_hash_is_not_an_individual_test_member_hash": True,
            },
            "test_payload_seal": {
                "first_test_member": contract.first_test_member,
                "first_test_header_start": contract.first_test_header_start,
                "first_test_payload_start": contract.first_test_payload_start,
                "local_header_scan_limit_exclusive": contract.scan_limit_exclusive,
            },
            "validation_terminal_offsets": {
                "member": contract.terminal_valid_member,
                "header_start": contract.terminal_valid_header_start,
                "compressed_end_exclusive": (
                    contract.terminal_valid_compressed_end_exclusive
                ),
                "record_end_exclusive": contract.terminal_valid_record_end_exclusive,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        audit_bytes = _canonical_bytes(audit)
        _write_new(stage / "materialization_audit.json", audit_bytes)
        audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()

        dataset = {
            "schema": DATASET_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "BSD",
            "exposure_setting": "3ms24ms",
            "blur_origin": "real_camera_long_exposure",
            "real_camera_long_exposure": True,
            "synthetic": False,
            "synthetic_high_fps_average": False,
            "scope": "validation_only_partial_prefix",
            "train_materialized": False,
            "full_dataset_ready": False,
            "authorized_splits": ["validation"],
            "splits": {
                "validation": {
                    "sequences": contract.valid_sequences,
                    "frames": contract.valid_pairs,
                }
            },
            "canonical_manifests": {
                "validation": {
                    "path": manifest_relative.as_posix(),
                    "sha256": manifest_sha256,
                    "schema": SEQUENCE_SCHEMA,
                    "sequence_records": contract.valid_sequences,
                    "pairs": contract.valid_pairs,
                    "paths_relative_to": "dataset_root",
                }
            },
            "materialization_audit_artifact": {
                "path": "materialization_audit.json",
                "sha256": audit_sha256,
                "schema": AUDIT_SCHEMA,
            },
            "frozen_protocol": {
                "path": str(acquisition.DEFAULT_PROTOCOL),
                "sha256": protocol_sha256,
            },
            "source_partial_prefix": {
                "path": str(partial),
                "bytes": contract.prefix_bytes,
                "sha256": opaque_sha256,
            },
            "test_disclosure": {
                "pixel_paths_materialized": False,
                "pixels_opened": False,
                "images_decoded": False,
                "model_outputs_computed": False,
                "metrics_computed": False,
            },
        }
        dataset_bytes = _canonical_bytes(dataset)
        _write_new(stage / "dataset_manifest.json", dataset_bytes)
        dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()

        receipt = {
            "schema": RECEIPT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass_validation_only_not_full_dataset_ready",
            "frozen_protocol_sha256": protocol_sha256,
            "source_partial_prefix": {
                "path": str(partial),
                "bytes": contract.prefix_bytes,
                "sha256": opaque_sha256,
                "stat_device": before.st_dev,
                "stat_inode": before.st_ino,
            },
            "artifacts": {
                "dataset_manifest": {
                    "path": "dataset_manifest.json",
                    "sha256": dataset_sha256,
                },
                "materialization_audit": {
                    "path": "materialization_audit.json",
                    "sha256": audit_sha256,
                },
                "validation_manifest": {
                    "path": manifest_relative.as_posix(),
                    "sha256": manifest_sha256,
                },
            },
            "validation_local_index_sha256": contract.valid_index_sha256,
            "validation_terminal_offsets": audit["validation_terminal_offsets"],
            "test_audit": audit["test_audit"],
            "train_materialized": False,
            "full_dataset_ready": False,
        }
        _write_new(stage / "validation_prefix_receipt.json", _canonical_bytes(receipt))

        after = os.fstat(source_fd)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_after != identity_before:
            raise PrefixMaterializationError("partial changed during materialization")
        pngs = list(stage.rglob("*.png"))
        if len(pngs) != contract.valid_png_members:
            raise PrefixMaterializationError("staged PNG inventory drifted")
        if any("test" in path.parts or "train" in path.parts for path in pngs):
            raise PrefixMaterializationError("unauthorized split path appeared")
        _readonly_tree(stage)
        os.replace(stage, output)
        return output / "validation_prefix_receipt.json"
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            for path in stage.rglob("*"):
                with contextlib.suppress(OSError):
                    os.chmod(path, 0o700 if path.is_dir() else 0o600)
            with contextlib.suppress(OSError):
                os.chmod(stage, 0o700)
            shutil.rmtree(stage)
        raise
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        os.close(lock_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=acquisition.DEFAULT_PROTOCOL)
    parser.add_argument("--partial", type=Path, default=FROZEN_PREFIX.partial_path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol, protocol_sha256 = acquisition.load_protocol(args.protocol)
    receipt = materialize_validation_prefix(
        protocol,
        protocol_sha256,
        FROZEN_PREFIX,
        args.partial.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "validation_prefix_receipt": str(receipt),
                "receipt_sha256": _sha256_file(receipt),
                "train_materialized": False,
                "full_dataset_ready": False,
                "test_member_payload_bytes_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PrefixMaterializationError, acquisition.AcquisitionError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
