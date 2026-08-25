"""CPU-only synthetic ZIP tests for sealed BSD selective materialization."""

from __future__ import annotations

import binascii
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import acquire_bsd_3ms24ms as acquisition  # noqa: E402
from scripts import materialize_bsd_3ms24ms as module  # noqa: E402
from scripts.bsd_dpdd_contract import (  # noqa: E402
    assert_train_validation_disjoint,
    inspect_bsd_sequence_manifest,
)
from scripts.train_turtle_streaming import load_sequence_manifest  # noqa: E402


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _png(value: int, *, width: int = 2, height: int = 1) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes([value, value + 1, value + 2]) * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(row * height))
        + _chunk(b"IEND", b"")
    )


def _index_sha(entries):
    return acquisition._entry_index_sha256(list(entries))


def _make_fixture(
    root: Path,
    *,
    cross_split_duplicate: bool = False,
    invalid_train_png: bool = False,
) -> tuple[dict, str, Path, Path]:
    archive_path = root / "BSD_3ms24ms.zip"
    values = {
        ("train", "Blur"): 10,
        ("train", "Sharp"): 20,
        ("valid", "Blur"): 30,
        ("valid", "Sharp"): 40,
        ("test", "Blur"): 50,
        ("test", "Sharp"): 60,
    }
    if cross_split_duplicate:
        values[("valid", "Blur")] = values[("train", "Blur")]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for split, sequence in (("train", "001"), ("valid", "002"), ("test", "003")):
            for role in ("Blur", "Sharp"):
                payload = _png(values[(split, role)])
                if invalid_train_png and (split, role) == ("train", "Blur"):
                    payload = b"not-a-png"
                archive.writestr(
                    f"BSD_3ms24ms/{split}/{sequence}/{role}/RGB/00000000.png",
                    payload,
                )
    archive_bytes = archive_path.read_bytes()
    eocd_offset = archive_bytes.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    central_size = int.from_bytes(
        archive_bytes[eocd_offset + 12 : eocd_offset + 16], "little"
    )
    central_offset = int.from_bytes(
        archive_bytes[eocd_offset + 16 : eocd_offset + 20], "little"
    )
    central = archive_bytes[central_offset : central_offset + central_size]
    eocd = archive_bytes[eocd_offset:]
    entries = acquisition.parse_central_directory(central)
    splits: dict[str, dict] = {}
    for split in ("train", "valid", "test"):
        selected = [
            entry
            for entry in entries
            if entry.name.startswith(f"BSD_3ms24ms/{split}/")
        ]
        splits[split] = {
            "sequences": 1,
            "frames_per_sequence": 1,
            "pairs": 1,
            "png_members": 2,
            "compressed_png_bytes": sum(item.compressed_bytes for item in selected),
            "uncompressed_png_bytes": sum(
                item.uncompressed_bytes for item in selected
            ),
            "png_index_sha256": _index_sha(selected),
        }
    protocol_sha256 = "a" * 64
    protocol = {
        "official_source": {
            "filename": archive_path.name,
            "content_length_bytes": len(archive_bytes),
        },
        "remote_zip_identity": {
            "entries_total": len(entries),
            "central_directory_offset": central_offset,
            "central_directory_bytes": central_size,
            "central_directory_sha256": hashlib.sha256(central).hexdigest(),
            "all_entry_index_sha256": _index_sha(entries),
            "zip64_eocd": {
                "offset": eocd_offset,
                "bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
            "zip64_locator": {
                "offset": eocd_offset,
                "bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
            "eocd": {
                "offset": eocd_offset,
                "bytes": len(eocd),
                "sha256": hashlib.sha256(eocd).hexdigest(),
            },
            "payload_root": "BSD_3ms24ms/",
            "payload_tree_entries": len(entries),
            "macos_metadata_root": "__MACOSX/",
            "macos_metadata_entries": 0,
            "real_png_members": 6,
            "raw_members": 0,
            "image_contract": {
                "width": 2,
                "height": 1,
                "bit_depth": 8,
                "color_type": 2,
                "interlace_method": 0,
            },
            "splits": splits,
        },
        "storage_policy": {
            "required_filesystem_root": "/srv/szha0669",
            "archive_mode_octal_after_acquisition": "0400",
        },
        "license_audit": {"dataset_license": "fixture"},
    }
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    os.chmod(archive_path, 0o400)
    receipt_path = root / "BSD_3ms24ms.zip.acquisition.json"
    receipt = {
        "schema": acquisition.RECEIPT_SCHEMA,
        "frozen_protocol": {"sha256": protocol_sha256},
        "archive": {
            "path": str(archive_path.resolve()),
            "bytes": len(archive_bytes),
            "sha256": archive_sha256,
        },
    }
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    os.chmod(receipt_path, 0o444)
    return protocol, protocol_sha256, archive_path, receipt_path


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o700)
        elif path.is_file():
            os.chmod(path, 0o600)
    os.chmod(root, 0o700)


class BsdMaterializerTests(unittest.TestCase):
    def test_selects_train_valid_and_never_opens_test_member(self) -> None:
        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as directory:
            root = Path(directory)
            protocol, digest, archive, receipt = _make_fixture(root)
            output = root / "materialized"
            calls: list[str] = []
            original = zipfile.ZipFile.open

            def guarded(instance, name, *args, **kwargs):
                filename = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
                if "/test/" in filename:
                    raise AssertionError(f"test payload opened: {filename}")
                calls.append(filename)
                return original(instance, name, *args, **kwargs)

            try:
                with mock.patch.object(zipfile.ZipFile, "open", guarded):
                    manifest_path = module.materialize(
                        protocol, digest, archive, receipt, output
                    )
                self.assertEqual(len(calls), 4)
                self.assertTrue(all("/train/" in name or "/valid/" in name for name in calls))
                self.assertFalse((output / "test").exists())
                self.assertEqual(len(list((output / "train").rglob("*.png"))), 2)
                self.assertEqual(len(list((output / "validation").rglob("*.png"))), 2)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["schema"], "unblur_slam.bsd_materialization.v1")
                self.assertEqual(set(manifest["canonical_manifests"]), {"train", "validation"})
                self.assertEqual(manifest["test_seal"]["test_member_payload_open_calls"], 0)
                self.assertEqual(
                    manifest["materialization_audit"][
                        "train_valid_content_sha256_overlap_count"
                    ],
                    0,
                )
                for split in ("train", "validation"):
                    sequence_manifest = output / "manifests" / f"{split}.jsonl"
                    record = json.loads(sequence_manifest.read_text(encoding="utf-8"))
                    self.assertEqual(record["frame_count"], 1)
                    self.assertEqual(record["frame_indices"], [0])
                    for key in ("blurry", "sharp"):
                        path = (output / record[key][0]).resolve()
                        self.assertTrue(path.is_file())
                        self.assertEqual(path.stat().st_mode & 0o777, 0o444)

                # Synthetic end-to-end compatibility: materializer output is
                # accepted without translation by the formal contract audit
                # and by the exact sequence loader used by train/evaluation.
                train_path = output / "manifests/train.jsonl"
                validation_path = output / "manifests/validation.jsonl"
                train = inspect_bsd_sequence_manifest(
                    train_path,
                    dataset_root=output,
                    expected_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
                    expected_split="train",
                    expected_sequences=1,
                    expected_frames=1,
                    expected_per_exposure_sequences=1,
                )
                validation = inspect_bsd_sequence_manifest(
                    validation_path,
                    dataset_root=output,
                    expected_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest(),
                    expected_split="validation",
                    expected_sequences=1,
                    expected_frames=1,
                    expected_per_exposure_sequences=1,
                )
                assert_train_validation_disjoint(train, validation)
                loaded = load_sequence_manifest(validation_path, root=output)
                self.assertEqual(len(loaded), 1)
                self.assertEqual(len(loaded[0].blurry), 1)
                audit = json.loads(
                    (output / "materialization_audit.json").read_text(encoding="utf-8")
                )
                self.assertEqual(audit["schema"], "unblur_slam.bsd_materialization_audit.v1")
                self.assertEqual(audit["status"], "pass")
            finally:
                _make_writable(output)

    def test_cross_split_content_duplicate_fails_and_cleans_stage(self) -> None:
        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as directory:
            root = Path(directory)
            protocol, digest, archive, receipt = _make_fixture(
                root, cross_split_duplicate=True
            )
            output = root / "materialized"
            with self.assertRaisesRegex(module.MaterializationError, "content overlap"):
                module.materialize(protocol, digest, archive, receipt, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".materialized.staging-*")), [])

    def test_invalid_authorized_png_contract_fails_before_publish(self) -> None:
        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as directory:
            root = Path(directory)
            protocol, digest, archive, receipt = _make_fixture(
                root, invalid_train_png=True
            )
            output = root / "materialized"
            with self.assertRaisesRegex(module.MaterializationError, "PNG"):
                module.materialize(protocol, digest, archive, receipt, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".materialized.staging-*")), [])

    def test_receipt_hash_mismatch_fails_before_member_open(self) -> None:
        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as directory:
            root = Path(directory)
            protocol, digest, archive, receipt = _make_fixture(root)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["archive"]["sha256"] = "0" * 64
            os.chmod(receipt, 0o600)
            receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            os.chmod(receipt, 0o444)
            calls: list[str] = []
            with mock.patch.object(
                zipfile.ZipFile,
                "open",
                side_effect=lambda *args, **kwargs: calls.append("open"),
            ):
                with self.assertRaisesRegex(module.MaterializationError, "archive SHA-256"):
                    module.materialize(protocol, digest, archive, receipt, root / "out")
            self.assertEqual(calls, [])

    def test_output_outside_srv_is_rejected_before_archive_read(self) -> None:
        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as directory:
            root = Path(directory)
            protocol, digest, archive, receipt = _make_fixture(root)
            with self.assertRaisesRegex(module.MaterializationError, "under"):
                module.materialize(
                    protocol,
                    digest,
                    archive,
                    receipt,
                    Path("/home/szha0669/forbidden-bsd-output"),
                )


if __name__ == "__main__":
    unittest.main()
