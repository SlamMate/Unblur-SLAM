#!/usr/bin/env python3
"""Safely materialize the third-party Hugging Face DPDD JPEG mirror.

This tool is deliberately fail-closed.  Normal use is restricted to the
``JacobLinCool/DPDD`` ``combined`` train and validation splits.  The test
split cannot even be enumerated unless an explicit, byte-pinned access
contract is supplied together with ``--allow-test-after-contract``.

``--preflight-only`` reads datasets-server row metadata but never opens an
image URL.  Materialization downloads into a sibling staging directory and
renames it into place only after every pair, digest, image dimension, and
manifest has been validated.  An existing output path is never reused or
overwritten.

The Hugging Face dataset is a third-party 8-bit JPEG mirror.  It is not the
official DPDD 16-bit distribution, and the generated manifest says so
explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import io
import json
import os
import re
import shutil
import struct
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence


REPOSITORY = "JacobLinCool/DPDD"
CONFIG = "combined"
DATASETS_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"
SCHEMA = "unblur_slam.dpdd_hf_materialization.v1"
PREFLIGHT_SCHEMA = "unblur_slam.dpdd_hf_preflight.v1"
TEST_CONTRACT_SCHEMA = "unblur_slam.dpdd_test_open_contract.v1"
EXPECTED_ROWS = {"train": 350, "validation": 74, "test": 76}
SERVER_SPLITS = {"train": "train", "validation": "val", "test": "test"}
DEFAULT_SPLITS = ("train", "validation")
DEFAULT_PAGE_SIZE = 100
SOURCE_FIELD = "source"
TARGET_FIELD = "target"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MaterializationError(RuntimeError):
    """Raised when a provenance or materialization contract is violated."""


@dataclass(frozen=True)
class AssetReference:
    """An in-memory asset locator plus query-free manifest provenance."""

    download_url: str
    provenance: Mapping[str, Any]
    declared_width: int | None
    declared_height: int | None


@dataclass(frozen=True)
class PairReference:
    split: str
    server_split: str
    row_idx: int
    source: AssetReference
    target: AssetReference


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _write_new_bytes(path: Path, payload: bytes) -> None:
    """Create one file without ever replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise MaterializationError(
            f"refusing to overwrite existing file: {path}"
        ) from error
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise


def _normalize_splits(splits: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in splits:
        name = str(raw).strip().lower()
        if name == "val":
            name = "validation"
        if name not in EXPECTED_ROWS:
            raise MaterializationError(
                f"unsupported DPDD split {raw!r}; expected train, validation, or test"
            )
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise MaterializationError("at least one split is required")
    return tuple(normalized)


def _validate_test_contract(
    *,
    splits: Sequence[str],
    allow_test_after_contract: bool,
    frozen_contract: Path | None,
    frozen_contract_sha256: str | None,
) -> Mapping[str, Any] | None:
    """Authorize test enumeration only after an exact frozen contract match."""

    if "test" not in splits:
        if allow_test_after_contract or frozen_contract or frozen_contract_sha256:
            raise MaterializationError(
                "test-access flags are forbidden when the test split was not requested"
            )
        return None

    if not allow_test_after_contract:
        raise MaterializationError(
            "DPDD test is sealed; pass --allow-test-after-contract only after the "
            "evaluation contract has been frozen"
        )
    if frozen_contract is None or frozen_contract_sha256 is None:
        raise MaterializationError(
            "DPDD test access requires both a frozen contract path and its SHA-256"
        )
    expected = str(frozen_contract_sha256).strip().lower()
    if not SHA256_RE.fullmatch(expected):
        raise MaterializationError(
            "frozen contract SHA-256 must be 64 lowercase hex chars"
        )
    contract_path = frozen_contract.expanduser().resolve()
    if not contract_path.is_file():
        raise MaterializationError(
            f"frozen test contract does not exist: {contract_path}"
        )
    try:
        contract_bytes = contract_path.read_bytes()
    except OSError as error:
        raise MaterializationError("frozen test contract could not be read") from error
    actual = hashlib.sha256(contract_bytes).hexdigest()
    if actual != expected:
        raise MaterializationError(
            f"frozen test contract SHA mismatch: expected {expected}, found {actual}"
        )
    try:
        contract = json.loads(contract_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError(
            "frozen test contract is not valid UTF-8 JSON"
        ) from error
    required = {
        "schema": TEST_CONTRACT_SCHEMA,
        "status": "frozen",
        "repository": REPOSITORY,
        "config": CONFIG,
        "split": "test",
        "expected_rows": EXPECTED_ROWS["test"],
        "allow_test_pixels": True,
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in required.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise MaterializationError(
            "frozen test contract does not authorize this exact test split: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "path": str(contract_path),
        "sha256": actual,
        "schema": TEST_CONTRACT_SCHEMA,
    }


def _query_free_provenance(locator: str) -> Mapping[str, Any]:
    """Return a stable URL path, never query parameters or signed tokens."""

    parsed = urllib.parse.urlsplit(str(locator))
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc and parsed.path:
        scheme = parsed.scheme.lower()
        return {
            "kind": "query_free_url_path",
            "origin": f"{scheme}://{parsed.netloc}",
            "path": parsed.path,
            "url_no_query": urllib.parse.urlunsplit(
                (scheme, parsed.netloc, parsed.path, "", "")
            ),
        }
    return {
        "kind": "opaque_locator_sha256",
        "sha256": hashlib.sha256(str(locator).encode("utf-8")).hexdigest(),
    }


def _positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _extract_asset(
    cell: Any, *, split: str, row_idx: int, field: str
) -> AssetReference:
    if not isinstance(cell, Mapping):
        raise MaterializationError(
            f"{split} row {row_idx} field {field!r} is not an image mapping"
        )
    locator = cell.get("src") or cell.get("url")
    if not isinstance(locator, str) or not locator.strip():
        raise MaterializationError(
            f"{split} row {row_idx} field {field!r} has no downloadable src/url"
        )
    parsed = urllib.parse.urlsplit(locator)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise MaterializationError(
            f"{split} row {row_idx} field {field!r} must use an HTTPS asset URL"
        )
    return AssetReference(
        download_url=locator,
        provenance=_query_free_provenance(locator),
        declared_width=_positive_int(cell.get("width")),
        declared_height=_positive_int(cell.get("height")),
    )


def _default_fetch_json(url: str, timeout: float) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Unblur-SLAM-DPDD-materializer/1"},
        method="GET",
    )
    payload: bytes | None = None
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            break
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    if payload is None:
        raise MaterializationError(
            "datasets-server row metadata request failed after 3 attempts"
        ) from last_error
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError("datasets-server returned invalid JSON") from error
    if not isinstance(result, Mapping):
        raise MaterializationError("datasets-server response must be a JSON object")
    return result


def _rows_url(api_base: str, server_split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": REPOSITORY,
            "config": CONFIG,
            "split": server_split,
            "offset": int(offset),
            "length": int(length),
        }
    )
    return f"{api_base}?{query}"


def enumerate_pairs(
    split: str,
    *,
    api_base: str = DATASETS_SERVER_ROWS,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = 30.0,
    fetch_json: Callable[[str, float], Mapping[str, Any]] = _default_fetch_json,
    expected_rows: Mapping[str, int] = EXPECTED_ROWS,
) -> list[PairReference]:
    """Enumerate one authorized split using row metadata only."""

    if split not in expected_rows:
        raise MaterializationError(
            f"no expected row count registered for split {split!r}"
        )
    if page_size < 1 or page_size > 100:
        raise MaterializationError("datasets-server page size must be in [1, 100]")
    expected_total = int(expected_rows[split])
    server_split = SERVER_SPLITS[split]
    pairs: list[PairReference] = []
    seen: set[int] = set()
    offset = 0
    while offset < expected_total:
        length = min(page_size, expected_total - offset)
        payload = fetch_json(_rows_url(api_base, server_split, offset, length), timeout)
        reported_total = payload.get("num_rows_total")
        if payload.get("partial") is not False:
            raise MaterializationError(
                f"{split} datasets-server response is partial; it cannot prove the "
                f"official {expected_total}-row split (reported rows: {reported_total})"
            )
        if reported_total is not None and int(reported_total) != expected_total:
            raise MaterializationError(
                f"{split} row-count drift: expected {expected_total}, "
                f"datasets-server reports {reported_total}"
            )
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != length:
            raise MaterializationError(
                f"{split} page offset {offset} expected {length} rows, "
                f"received {len(rows) if isinstance(rows, list) else 'non-list'}"
            )
        for expected_idx, wrapper in enumerate(rows, start=offset):
            if not isinstance(wrapper, Mapping):
                raise MaterializationError(f"{split} row wrapper is not an object")
            row_idx = wrapper.get("row_idx")
            if isinstance(row_idx, bool) or not isinstance(row_idx, int):
                raise MaterializationError(
                    f"{split} row has invalid row_idx {row_idx!r}"
                )
            if row_idx != expected_idx:
                raise MaterializationError(
                    f"{split} row order mismatch: expected {expected_idx}, found {row_idx}"
                )
            if row_idx in seen:
                raise MaterializationError(f"duplicate {split} row_idx {row_idx}")
            row = wrapper.get("row")
            if not isinstance(row, Mapping):
                raise MaterializationError(
                    f"{split} row {row_idx} payload is not an object"
                )
            if SOURCE_FIELD not in row or TARGET_FIELD not in row:
                raise MaterializationError(
                    f"{split} row {row_idx} must contain source and target image fields"
                )
            pairs.append(
                PairReference(
                    split=split,
                    server_split=server_split,
                    row_idx=row_idx,
                    source=_extract_asset(
                        row[SOURCE_FIELD],
                        split=split,
                        row_idx=row_idx,
                        field=SOURCE_FIELD,
                    ),
                    target=_extract_asset(
                        row[TARGET_FIELD],
                        split=split,
                        row_idx=row_idx,
                        field=TARGET_FIELD,
                    ),
                )
            )
            seen.add(row_idx)
        offset += length
    if len(pairs) != expected_total:
        raise MaterializationError(
            f"{split} enumeration incomplete: expected {expected_total}, found {len(pairs)}"
        )
    return pairs


def prepare_preflight(
    splits: Sequence[str],
    *,
    api_base: str = DATASETS_SERVER_ROWS,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = 30.0,
    fetch_json: Callable[[str, float], Mapping[str, Any]] = _default_fetch_json,
    expected_rows: Mapping[str, int] = EXPECTED_ROWS,
    allow_test_after_contract: bool = False,
    frozen_contract: Path | None = None,
    frozen_contract_sha256: str | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, list[PairReference]]]:
    """Validate split metadata without opening a source or target image URL."""

    normalized = _normalize_splits(splits)
    contract = _validate_test_contract(
        splits=normalized,
        allow_test_after_contract=allow_test_after_contract,
        frozen_contract=frozen_contract,
        frozen_contract_sha256=frozen_contract_sha256,
    )
    pairs_by_split = {
        split: enumerate_pairs(
            split,
            api_base=api_base,
            page_size=page_size,
            timeout=timeout,
            fetch_json=fetch_json,
            expected_rows=expected_rows,
        )
        for split in normalized
    }
    report: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "repository": REPOSITORY,
        "config": CONFIG,
        "datasets_server_endpoint": api_base,
        "splits": {split: len(pairs_by_split[split]) for split in normalized},
        "server_splits": {split: SERVER_SPLITS[split] for split in normalized},
        "expected_splits": {split: int(expected_rows[split]) for split in normalized},
        "image_urls_opened": False,
        "pixel_bytes_downloaded": 0,
        "test_access_contract": contract,
        "distribution": {
            "kind": "third_party_huggingface_jpeg_mirror",
            "mirror_declared_encoding": "8-bit JPEG",
            "observed_encoding": "unknown_preflight_no_pixel_bytes_opened",
            "official_dpdd_16bit_original": False,
            "license_status": "not_verified_by_materializer",
            "redistribution_status": "requires_separate_terms_review",
            "warning": (
                "This mirror is not the official DPDD 16-bit distribution; "
                "verify provenance and terms before publication or redistribution."
            ),
        },
    }
    return report, pairs_by_split


def _jpeg_dimensions(stream: BinaryIO) -> tuple[int, int]:
    if stream.read(2) != b"\xff\xd8":
        raise MaterializationError("asset is not a JPEG (missing SOI marker)")
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while True:
        byte = stream.read(1)
        if not byte:
            break
        if byte != b"\xff":
            continue
        marker_byte = stream.read(1)
        while marker_byte == b"\xff":
            marker_byte = stream.read(1)
        if not marker_byte:
            break
        marker = marker_byte[0]
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        raw_length = stream.read(2)
        if len(raw_length) != 2:
            break
        segment_length = struct.unpack(">H", raw_length)[0]
        if segment_length < 2:
            raise MaterializationError("JPEG contains an invalid segment length")
        if marker in sof_markers:
            payload = stream.read(segment_length - 2)
            if len(payload) < 5:
                break
            height, width = struct.unpack(">HH", payload[1:5])
            if width < 1 or height < 1:
                raise MaterializationError("JPEG reports non-positive dimensions")
            return width, height
        stream.seek(segment_length - 2, io.SEEK_CUR)
    raise MaterializationError("JPEG dimensions could not be resolved")


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        return _jpeg_dimensions(stream)


def _default_open_asset(url: str, timeout: float) -> BinaryIO:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Unblur-SLAM-DPDD-materializer/1"},
        method="GET",
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except (OSError, urllib.error.URLError) as error:
        provenance = _query_free_provenance(url)
        raise MaterializationError(
            "DPDD asset request failed for query-free provenance "
            + json.dumps(provenance, sort_keys=True)
        ) from error


def _download_asset(
    reference: AssetReference,
    destination: Path,
    *,
    timeout: float,
    open_asset: Callable[[str, float], BinaryIO],
) -> Mapping[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise MaterializationError(f"refusing to overwrite staged asset: {destination}")
    digest = hashlib.sha256()
    size = 0
    try:
        response = open_asset(reference.download_url, timeout)
        with contextlib.closing(response), destination.open("xb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                digest.update(block)
                size += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        if size <= 0:
            raise MaterializationError("downloaded DPDD asset is empty")
        width, height = jpeg_dimensions(destination)
        if reference.declared_width is not None and reference.declared_width != width:
            raise MaterializationError(
                f"datasets-server width mismatch: declared {reference.declared_width}, "
                f"decoded {width}"
            )
        if (
            reference.declared_height is not None
            and reference.declared_height != height
        ):
            raise MaterializationError(
                f"datasets-server height mismatch: declared {reference.declared_height}, "
                f"decoded {height}"
            )
        return {
            "relative_path": destination.as_posix(),
            "bytes": size,
            "sha256": digest.hexdigest(),
            "pixel_size": {"width": width, "height": height},
            "encoding": "JPEG",
            "asset_provenance": dict(reference.provenance),
        }
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            destination.unlink()
        raise


def _record_relative_paths(record: Mapping[str, Any], stage: Path) -> Mapping[str, Any]:
    result = dict(record)
    for role in (SOURCE_FIELD, TARGET_FIELD):
        item = dict(result[role])
        item["relative_path"] = str(Path(item["relative_path"]).relative_to(stage))
        result[role] = item
    return result


def _atomic_publish_directory(stage: Path, output: Path) -> None:
    """Publish a directory atomically without replacing a racing destination.

    A POSIX ``rename`` can replace an existing empty directory, so an
    existence check followed by ``os.rename`` does not meet the no-overwrite
    contract.  Linux ``renameat2(RENAME_NOREPLACE)`` provides both guarantees
    in one syscall.  This tool fails closed if that syscall is unavailable.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise MaterializationError(
            "atomic no-overwrite publication requires Linux renameat2"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(stage),
        -100,
        os.fsencode(output),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise MaterializationError(
            f"refusing to overwrite output path created during publication: {output}"
        )
    raise MaterializationError(
        f"atomic output publication failed: {os.strerror(error_number)}"
    )


def materialize(
    output: Path,
    splits: Sequence[str] = DEFAULT_SPLITS,
    *,
    api_base: str = DATASETS_SERVER_ROWS,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = 30.0,
    fetch_json: Callable[[str, float], Mapping[str, Any]] = _default_fetch_json,
    open_asset: Callable[[str, float], BinaryIO] = _default_open_asset,
    expected_rows: Mapping[str, int] = EXPECTED_ROWS,
    allow_test_after_contract: bool = False,
    frozen_contract: Path | None = None,
    frozen_contract_sha256: str | None = None,
) -> Path:
    """Download all requested pairs atomically and return dataset_manifest.json."""

    output = output.expanduser().resolve()
    if output.exists():
        raise MaterializationError(
            f"refusing to overwrite existing output path: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    preflight, pairs_by_split = prepare_preflight(
        splits,
        api_base=api_base,
        page_size=page_size,
        timeout=timeout,
        fetch_json=fetch_json,
        expected_rows=expected_rows,
        allow_test_after_contract=allow_test_after_contract,
        frozen_contract=frozen_contract,
        frozen_contract_sha256=frozen_contract_sha256,
    )
    if output.exists():
        raise MaterializationError(
            f"output path appeared during preflight; refusing to overwrite: {output}"
        )

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    completed = False
    try:
        records: list[Mapping[str, Any]] = []
        total_asset_bytes = 0
        for split in preflight["splits"]:
            for pair in pairs_by_split[split]:
                stem = f"{pair.row_idx:06d}.jpg"
                source_path = stage / split / SOURCE_FIELD / stem
                target_path = stage / split / TARGET_FIELD / stem
                source = _download_asset(
                    pair.source, source_path, timeout=timeout, open_asset=open_asset
                )
                target = _download_asset(
                    pair.target, target_path, timeout=timeout, open_asset=open_asset
                )
                if source["pixel_size"] != target["pixel_size"]:
                    raise MaterializationError(
                        f"{split} row {pair.row_idx} source/target dimensions differ"
                    )
                total_asset_bytes += int(source["bytes"]) + int(target["bytes"])
                records.append(
                    _record_relative_paths(
                        {
                            "schema": "unblur_slam.dpdd_hf_pair.v1",
                            "repository": REPOSITORY,
                            "config": CONFIG,
                            "split": split,
                            "server_split": pair.server_split,
                            "row_idx": pair.row_idx,
                            SOURCE_FIELD: source,
                            TARGET_FIELD: target,
                            "pairing": "same-scene defocused JPEG to all-in-focus JPEG",
                        },
                        stage,
                    )
                )

        pairs_path = stage / "pairs.jsonl"
        _write_new_bytes(
            pairs_path,
            b"".join(_canonical_json_bytes(record) for record in records),
        )
        pairs_sha = sha256_file(pairs_path)
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "repository": REPOSITORY,
            "config": CONFIG,
            "datasets_server_endpoint": api_base,
            "splits": dict(preflight["splits"]),
            "server_splits": dict(preflight["server_splits"]),
            "pair_count": len(records),
            "asset_count": len(records) * 2,
            "asset_bytes": total_asset_bytes,
            "pairs_jsonl": {"path": "pairs.jsonl", "sha256": pairs_sha},
            "test_access_contract": preflight["test_access_contract"],
            "distribution": {
                **dict(preflight["distribution"]),
                "observed_encoding": "JPEG_verified_from_every_downloaded_asset",
            },
            "materialization": {
                "atomic_directory_publication": True,
                "overwrite_allowed": False,
                "source_and_target_sha256_recorded": True,
                "source_and_target_pixel_size_recorded": True,
                "signed_url_queries_recorded": False,
            },
        }
        manifest_path = stage / "dataset_manifest.json"
        _write_new_bytes(manifest_path, _canonical_json_bytes(manifest))

        checksummed = sorted(
            path
            for path in stage.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        checksum_lines = [
            f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n"
            for path in checksummed
        ]
        _write_new_bytes(stage / "SHA256SUMS", "".join(checksum_lines).encode("utf-8"))
        if output.exists():
            raise MaterializationError(
                f"output path appeared before publication; refusing overwrite: {output}"
            )
        _atomic_publish_directory(stage, output)
        completed = True
        return output / "dataset_manifest.json"
    finally:
        if not completed:
            shutil.rmtree(stage, ignore_errors=True)


def _write_preflight_report(path: Path, report: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    _write_new_bytes(destination, _canonical_json_bytes(report))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "New output directory; an existing path is always rejected. "
            "Not required for --preflight-only."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Default: train validation. 'val' is normalized to validation.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--preflight-report",
        type=Path,
        help="Optional new JSON path for metadata-only preflight output.",
    )
    parser.add_argument("--api-base", default=DATASETS_SERVER_ROWS)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--allow-test-after-contract", action="store_true")
    parser.add_argument("--frozen-contract", type=Path)
    parser.add_argument("--frozen-contract-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        report, _ = prepare_preflight(
            args.splits,
            api_base=args.api_base,
            page_size=args.page_size,
            timeout=args.timeout,
            allow_test_after_contract=args.allow_test_after_contract,
            frozen_contract=args.frozen_contract,
            frozen_contract_sha256=args.frozen_contract_sha256,
        )
        if args.preflight_report is not None:
            _write_preflight_report(args.preflight_report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.preflight_report is not None:
        raise MaterializationError("--preflight-report requires --preflight-only")
    if args.output is None:
        raise MaterializationError(
            "--output is required unless --preflight-only is used"
        )
    manifest_path = materialize(
        args.output,
        args.splits,
        api_base=args.api_base,
        page_size=args.page_size,
        timeout=args.timeout,
        allow_test_after_contract=args.allow_test_after_contract,
        frozen_contract=args.frozen_contract,
        frozen_contract_sha256=args.frozen_contract_sha256,
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MaterializationError as error:
        raise SystemExit(f"error: {error}") from error
