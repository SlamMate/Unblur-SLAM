"""CPU-only tests for validation materialization from a truncated ZIP prefix."""

from __future__ import annotations

import binascii
import dataclasses
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import materialize_bsd_3ms24ms_validation_prefix as module  # noqa: E402
from scripts.bsd_dpdd_contract import inspect_bsd_sequence_manifest  # noqa: E402


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _png(value: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes([value, value + 1, value + 2]) * 2
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(row))
        + _chunk(b"IEND", b"")
    )


class _NonSeekable(io.BytesIO):
    def seekable(self) -> bool:
        return False

    def seek(self, *args, **kwargs):
        del args, kwargs
        raise io.UnsupportedOperation("fixture forces signed data descriptors")


def _data_start(payload: bytes, header_start: int) -> int:
    values = module.LOCAL_HEADER.unpack(
        payload[header_start : header_start + module.LOCAL_HEADER.size]
    )
    return header_start + module.LOCAL_HEADER.size + int(values[9]) + int(values[10])


def _make_fixture(root: Path):
    stream = _NonSeekable()
    names = [
        "BSD_3ms24ms/valid/001/Blur/RGB/00000000.png",
        "BSD_3ms24ms/valid/001/Sharp/RGB/00000000.png",
        "BSD_3ms24ms/test/002/Blur/RGB/00000000.png",
    ]
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(names[0], _png(10))
        archive.writestr(names[1], _png(30))
        archive.writestr(names[2], _png(50))
    full = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(full), "r") as archive:
        infos = {info.filename: info for info in archive.infolist()}
    test_info = infos[names[2]]
    test_header = test_info.header_offset
    test_payload = _data_start(full, test_header)
    partial_bytes = full[: test_payload + 3]
    partial = root / "BSD_3ms24ms.zip.partial"
    partial.write_bytes(partial_bytes)
    os.chmod(partial, 0o600)

    valid_entries: list[module.LocalEntry] = []
    for name in names[:2]:
        info = infos[name]
        compressed_start = _data_start(full, info.header_offset)
        compressed_end = compressed_start + info.compress_size
        self_header = module.LOCAL_HEADER.unpack(
            full[info.header_offset : info.header_offset + module.LOCAL_HEADER.size]
        )
        self_flags = int(self_header[2])
        descriptor_bytes = 16 if self_flags & 0x8 else 0
        descriptor_signature = full[compressed_end : compressed_end + 4]
        assert not descriptor_bytes or descriptor_signature == module.DESCRIPTOR_SIGNATURE
        valid_entries.append(
            module.LocalEntry(
                name=name,
                flags=self_flags,
                method=info.compress_type,
                crc32=info.CRC,
                compressed_bytes=info.compress_size,
                uncompressed_bytes=info.file_size,
                header_start=info.header_offset,
                compressed_start=compressed_start,
                compressed_end_exclusive=compressed_end,
                record_end_exclusive=compressed_end + descriptor_bytes,
            )
        )
    contract = module.PrefixContract(
        protocol_sha256="a" * 64,
        partial_path=partial.resolve(),
        prefix_bytes=len(partial_bytes),
        opaque_prefix_sha256=hashlib.sha256(partial_bytes).hexdigest(),
        scan_limit_exclusive=test_payload,
        complete_entries_before_test=2,
        first_test_member=names[2],
        first_test_header_start=test_header,
        first_test_payload_start=test_payload,
        valid_png_members=2,
        valid_sequences=1,
        valid_pairs=1,
        valid_frames_per_sequence=1,
        valid_compressed_bytes=sum(item.compressed_bytes for item in valid_entries),
        valid_uncompressed_bytes=sum(
            item.uncompressed_bytes for item in valid_entries
        ),
        valid_index_sha256=module._valid_index_sha256(valid_entries),
        first_valid_member=valid_entries[0].name,
        first_valid_header_start=valid_entries[0].header_start,
        terminal_valid_member=valid_entries[-1].name,
        terminal_valid_header_start=valid_entries[-1].header_start,
        terminal_valid_compressed_end_exclusive=(
            valid_entries[-1].compressed_end_exclusive
        ),
        terminal_valid_record_end_exclusive=valid_entries[-1].record_end_exclusive,
    )
    protocol = {
        "storage_policy": {"required_filesystem_root": "/srv/szha0669"},
        "remote_zip_identity": {
            "image_contract": {
                "width": 2,
                "height": 1,
                "bit_depth": 8,
                "color_type": 2,
                "interlace_method": 0,
            },
            "splits": {
                "valid": {
                    "png_members": 2,
                    "sequences": 1,
                    "pairs": 1,
                    "frames_per_sequence": 1,
                    "compressed_png_bytes": contract.valid_compressed_bytes,
                    "uncompressed_png_bytes": contract.valid_uncompressed_bytes,
                    "png_index_sha256": contract.valid_index_sha256,
                }
            },
        },
    }
    return protocol, contract, partial, valid_entries


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        with __import__("contextlib").suppress(OSError):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
    os.chmod(root, 0o700)


class ValidationPrefixTests(unittest.TestCase):
    def test_truncated_test_payload_materializes_only_canonical_validation(self) -> None:
        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as directory:
            root = Path(directory)
            protocol, contract, partial, _ = _make_fixture(root)
            output = root / "validation_prefix"
            try:
                receipt_path = module.materialize_validation_prefix(
                    protocol, contract.protocol_sha256, contract, partial, output
                )
                self.assertTrue(receipt_path.is_file())
                self.assertEqual(len(list((output / "validation").rglob("*.png"))), 2)
                self.assertFalse((output / "test").exists())
                self.assertFalse((output / "train").exists())
                manifest = output / "manifests/validation.jsonl"
                inventory = inspect_bsd_sequence_manifest(
                    manifest,
                    dataset_root=output,
                    expected_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    expected_split="validation",
                    expected_sequences=1,
                    expected_frames=1,
                    expected_per_exposure_sequences=1,
                )
                self.assertEqual(inventory.frame_count, 1)
                dataset = json.loads(
                    (output / "dataset_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(set(dataset["canonical_manifests"]), {"validation"})
                self.assertFalse(dataset["train_materialized"])
                self.assertFalse(dataset["full_dataset_ready"])
                audit = json.loads(
                    (output / "materialization_audit.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(audit["status"], "pass")
                self.assertEqual(audit["test_audit"]["member_payload_bytes_read"], 0)
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                self.assertFalse(receipt["train_materialized"])
            finally:
                _make_writable(output)

    def test_valid_payload_tamper_fails_and_never_publishes(self) -> None:
        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as directory:
            root = Path(directory)
            protocol, contract, partial, entries = _make_fixture(root)
            payload = bytearray(partial.read_bytes())
            payload[entries[0].compressed_start + 1] ^= 1
            partial.write_bytes(payload)
            contract = dataclasses.replace(
                contract, opaque_prefix_sha256=hashlib.sha256(payload).hexdigest()
            )
            output = root / "validation_prefix"
            with self.assertRaises(Exception):
                module.materialize_validation_prefix(
                    protocol, contract.protocol_sha256, contract, partial, output
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".validation_prefix.staging-*")), [])

    def test_seal_boundary_drift_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as directory:
            root = Path(directory)
            protocol, contract, partial, _ = _make_fixture(root)
            contract = dataclasses.replace(
                contract, scan_limit_exclusive=contract.scan_limit_exclusive + 1
            )
            with self.assertRaisesRegex(
                module.PrefixMaterializationError, "scan limit"
            ):
                module.materialize_validation_prefix(
                    protocol,
                    contract.protocol_sha256,
                    contract,
                    partial,
                    root / "output",
                )


if __name__ == "__main__":
    unittest.main()
