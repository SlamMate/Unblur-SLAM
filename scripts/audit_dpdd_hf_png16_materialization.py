#!/usr/bin/env python3
"""Offline, content-addressed audit of a materialized DPDD PNG16 train/val tree."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import json
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY = "JacobLinCool/DPDD"
REVISION = "52e4035a045ea1763313b9ce2b27cf2e620cfc30"
CONFIG = "combined"
MATERIALIZATION_SCHEMA = "unblur_slam.dpdd_hf_png16_materialization.v1"
CANONICAL_PAIR_SCHEMA = "unblur_slam.dpdd_hf_canonical_pair.v1"
AUDIT_SCHEMA = "unblur_slam.dpdd_hf_png16_materialization_audit.v1"
EXPECTED_SPLITS = {"train": 350, "validation": 74}
EXPECTED_SIZE = (1680, 1120)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, Mapping):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def _relative_file(root: Path, value: Any, *, label: str) -> Path:
    raw = Path(str(value))
    if raw.is_absolute() or ".." in raw.parts or raw.as_posix() != str(value):
        raise AuditError(f"{label} is not a canonical relative path: {value!r}")
    candidate = root / raw
    if candidate.is_symlink():
        raise AuditError(f"{label} must not be a symlink: {value!r}")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AuditError(f"{label} escapes dataset root: {value!r}") from error
    if not path.is_file():
        raise AuditError(f"{label} is missing or non-file: {value!r}")
    return path


def _sha(value: Any, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise AuditError(f"{label} is not a lowercase SHA-256")
    return normalized


def _decode_png16(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(33)
    if len(header) < 33 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise AuditError(f"asset is not PNG: {path}")
    if struct.unpack(">I", header[8:12])[0] != 13 or header[12:16] != b"IHDR":
        raise AuditError(f"asset has invalid PNG IHDR: {path}")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
    if bit_depth != 16 or color_type != 2:
        raise AuditError(
            f"asset is not 16-bit RGB PNG: depth={bit_depth}, color={color_type}: {path}"
        )
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise AuditError("OpenCV and NumPy are required for lossless audit") from error
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        decoded is None
        or decoded.dtype != np.uint16
        or decoded.ndim != 3
        or decoded.shape != (height, width, 3)
    ):
        actual = None if decoded is None else (str(decoded.dtype), decoded.shape)
        raise AuditError(f"uint16 HWC3 decode contract failed for {path}: {actual}")
    return width, height


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            raise AuditError(f"blank canonical row at {path}:{line_number}")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(record, Mapping):
            raise AuditError(f"non-object row at {path}:{line_number}")
        records.append(record)
    return records


def audit_materialization(
    dataset_root: Path,
    *,
    expected_splits: Mapping[str, int] = EXPECTED_SPLITS,
    expected_size: tuple[int, int] = EXPECTED_SIZE,
) -> Mapping[str, Any]:
    """Audit only local train/validation bytes; this function has no network code."""

    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise AuditError(f"dataset root does not exist: {root}")
    manifest_path = root / "dataset_manifest.json"
    manifest = _load_json(manifest_path)
    required_identity = {
        "schema": MATERIALIZATION_SCHEMA,
        "repository": REPOSITORY,
        "revision": REVISION,
        "config": CONFIG,
    }
    for key, expected in required_identity.items():
        if manifest.get(key) != expected:
            raise AuditError(f"dataset manifest {key} mismatch")
    if manifest.get("splits") != dict(expected_splits):
        raise AuditError("dataset manifest split counts mismatch")
    if manifest.get("pair_count") != sum(expected_splits.values()):
        raise AuditError("dataset manifest pair_count mismatch")
    expected_asset_count = 2 * sum(expected_splits.values())
    if manifest.get("asset_count") != expected_asset_count:
        raise AuditError("dataset manifest asset_count mismatch")

    disclosure = manifest.get("test_disclosure")
    if not isinstance(disclosure, Mapping):
        raise AuditError("dataset manifest has no test disclosure")
    if (
        disclosure.get("requests_made_by_this_materializer") != 0
        or disclosure.get("split_supported_by_this_materializer") is not False
        or disclosure.get("images_decoded") is not False
        or disclosure.get("pixels_opened") is not False
        or disclosure.get("metrics_opened") is not False
    ):
        raise AuditError(
            "dataset manifest does not preserve the sealed-test disclosure"
        )

    canonical = manifest.get("canonical_manifests")
    if not isinstance(canonical, Mapping) or set(canonical) != set(expected_splits):
        raise AuditError("dataset manifest canonical manifest inventory mismatch")

    names: set[str] = set()
    paths_by_role = {"source": set(), "target": set()}
    hashes_by_role = {"source": set(), "target": set()}
    paths_by_split: dict[str, set[Path]] = {split: set() for split in expected_splits}
    hashes_by_split: dict[str, set[str]] = {split: set() for split in expected_splits}
    dimensions: set[tuple[int, int]] = set()
    canonical_report: dict[str, Any] = {}
    asset_report: list[Mapping[str, Any]] = []

    for split, expected_rows in expected_splits.items():
        descriptor = canonical[split]
        if not isinstance(descriptor, Mapping):
            raise AuditError(f"invalid canonical descriptor for {split}")
        if (
            descriptor.get("rows") != expected_rows
            or descriptor.get("schema") != CANONICAL_PAIR_SCHEMA
            or descriptor.get("paths_relative_to") != "dataset_root"
        ):
            raise AuditError(f"canonical descriptor contract mismatch for {split}")
        path = _relative_file(root, descriptor.get("path"), label=f"{split} manifest")
        actual_manifest_sha = sha256_file(path)
        if actual_manifest_sha != _sha(
            descriptor.get("sha256"), label=f"{split} manifest sha256"
        ):
            raise AuditError(f"canonical manifest SHA mismatch for {split}")
        records = _read_jsonl(path)
        if len(records) != expected_rows:
            raise AuditError(f"canonical manifest row mismatch for {split}")
        canonical_report[split] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": actual_manifest_sha,
            "rows": len(records),
        }

        for row_idx, record in enumerate(records):
            required = {
                "schema",
                "name",
                "split",
                "defocus",
                "sharp",
                "source_sha256",
                "target_sha256",
            }
            if not required.issubset(record):
                raise AuditError(f"canonical {split} row {row_idx} is incomplete")
            if record["schema"] != CANONICAL_PAIR_SCHEMA or record["split"] != split:
                raise AuditError(f"canonical {split} row {row_idx} identity mismatch")
            name = str(record["name"])
            if not name or name in names:
                raise AuditError(f"duplicate/empty canonical name: {name!r}")
            names.add(name)
            pair_dimensions: list[tuple[int, int]] = []
            for role, key, hash_key in (
                ("source", "defocus", "source_sha256"),
                ("target", "sharp", "target_sha256"),
            ):
                asset = _relative_file(root, record[key], label=f"{name} {role}")
                relative = asset.relative_to(root).as_posix()
                if not relative.startswith(f"{split}/{role}/"):
                    raise AuditError(f"{name} {role} path has wrong split/role prefix")
                declared_sha = _sha(record[hash_key], label=f"{name} {hash_key}")
                actual_sha = sha256_file(asset)
                if actual_sha != declared_sha:
                    raise AuditError(f"{name} {role} content SHA mismatch")
                size = _decode_png16(asset)
                if size != expected_size:
                    raise AuditError(
                        f"{name} {role} size {size} != expected {expected_size}"
                    )
                if asset in paths_by_role[role] or actual_sha in hashes_by_role[role]:
                    raise AuditError(f"duplicate {role} path/content: {name}")
                paths_by_role[role].add(asset)
                hashes_by_role[role].add(actual_sha)
                paths_by_split[split].add(asset)
                hashes_by_split[split].add(actual_sha)
                dimensions.add(size)
                pair_dimensions.append(size)
                asset_report.append(
                    {
                        "split": split,
                        "row_idx": row_idx,
                        "role": role,
                        "path": relative,
                        "bytes": asset.stat().st_size,
                        "sha256": actual_sha,
                        "width": size[0],
                        "height": size[1],
                        "dtype": "uint16",
                        "channels": 3,
                    }
                )
            if pair_dimensions[0] != pair_dimensions[1]:
                raise AuditError(f"source/target dimensions differ for {name}")
            if record["source_sha256"] == record["target_sha256"]:
                raise AuditError(f"source/target content identical for {name}")

    if len(asset_report) != expected_asset_count:
        raise AuditError("audited asset count mismatch")
    if paths_by_role["source"] & paths_by_role["target"]:
        raise AuditError("global source/target paths overlap")
    if hashes_by_role["source"] & hashes_by_role["target"]:
        raise AuditError("global source/target content overlaps")
    split_names = list(expected_splits)
    for left_idx, left in enumerate(split_names):
        for right in split_names[left_idx + 1 :]:
            if paths_by_split[left] & paths_by_split[right]:
                raise AuditError(f"{left}/{right} paths overlap")
            if hashes_by_split[left] & hashes_by_split[right]:
                raise AuditError(f"{left}/{right} content overlaps")

    actual_pngs = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    }
    audited_paths = paths_by_role["source"] | paths_by_role["target"]
    if actual_pngs != audited_paths:
        raise AuditError("unreferenced or missing PNG assets exist in dataset root")
    local_test_paths = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if any(
            part.lower() in {"test", "test_c"}
            or part.lower().startswith("test_")
            or part.lower().startswith("test.")
            for part in path.relative_to(root).parts
        )
    ]
    if local_test_paths:
        raise AuditError("local test paths exist in train/validation materialization")

    pairs_descriptor = manifest.get("pairs_jsonl")
    if not isinstance(pairs_descriptor, Mapping):
        raise AuditError("dataset manifest has no pairs_jsonl binding")
    pairs_path = _relative_file(root, pairs_descriptor.get("path"), label="pairs_jsonl")
    pairs_sha = sha256_file(pairs_path)
    if pairs_sha != _sha(pairs_descriptor.get("sha256"), label="pairs_jsonl sha256"):
        raise AuditError("pairs_jsonl SHA mismatch")

    return {
        "schema": AUDIT_SCHEMA,
        "status": "pass",
        "dataset_root": str(root),
        "dataset_manifest": {
            "path": "dataset_manifest.json",
            "sha256": sha256_file(manifest_path),
        },
        "repository": REPOSITORY,
        "revision": REVISION,
        "config": CONFIG,
        "canonical_manifests": canonical_report,
        "pairs_jsonl": {
            "path": pairs_path.relative_to(root).as_posix(),
            "sha256": pairs_sha,
        },
        "pair_count": sum(expected_splits.values()),
        "asset_count": len(asset_report),
        "asset_bytes": sum(int(item["bytes"]) for item in asset_report),
        "image_contract": {
            "unique_sizes": [list(size) for size in sorted(dimensions)],
            "all_expected_size": list(expected_size),
            "all_png_ihdr_16bit_rgb": True,
            "all_opencv_uint16_hwc3": True,
        },
        "disjoint_audit": {
            "all_asset_paths_unique": True,
            "all_asset_hashes_unique": True,
            "source_target_paths_and_hashes_disjoint": True,
            "train_validation_paths_and_hashes_disjoint": True,
        },
        "test_audit": {
            "local_test_paths": 0,
            "network_requests_by_auditor": 0,
            "network_capability_in_auditor": False,
            "materializer_disclosure_verified": True,
        },
        "assets": asset_report,
    }


def _write_atomic_new(path: Path, payload: bytes) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise AuditError(f"refusing to overwrite audit: {destination}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise AuditError("atomic no-overwrite audit publication needs renameat2")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1):
            number = ctypes.get_errno()
            if number == errno.EEXIST:
                raise AuditError(f"refusing to overwrite audit: {destination}")
            raise AuditError(f"audit publication failed: {os.strerror(number)}")
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_materialization(args.dataset_root)
    payload = _canonical_bytes(report)
    if args.output is None:
        print(payload.decode("utf-8"), end="")
    else:
        _write_atomic_new(args.output, payload)
        print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        raise SystemExit(f"error: {error}") from error
