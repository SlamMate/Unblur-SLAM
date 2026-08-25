#!/usr/bin/env python3
"""Fail-closed acquisition for the official BSD 3ms--24ms archive.

The default command performs metadata-only preflight.  A download requires two
explicit switches and writes the publisher's single ZIP as an opaque object to
an owner-only quarantine directory.  This module never opens a ZIP member and
therefore cannot expose an individual test image payload.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping


SCHEMA = "unblur_slam.bsd_3ms24ms_acquisition_protocol.v1"
RECEIPT_SCHEMA = "unblur_slam.bsd_3ms24ms_acquisition_receipt.v1"
CHECKPOINT_RECEIPT_SCHEMA = "unblur_slam.turtle_bsd_checkpoint_receipt.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPOSITORY_ROOT / "configs/bsd_3ms24ms_acquisition.v1.json"
EXPECTED_PROTOCOL_SHA256 = "0aaacf7166b6bbf1e50f9835340486010a25148a01bc8257f4355cc0dd1a8e40"
BUFFER_BYTES = 8 * 1024 * 1024
# Google Drive may downgrade a multi-gigabyte Range request to its quota HTML
# response even while bounded Range requests for the same pinned object remain
# available.  Keep every payload request bounded and let the outer acquisition
# loop advance monotonically across chunks.
# Google Drive currently replaces ranges above 1 MiB with a quota HTML page
# for this public object while serving 1 MiB ranges as the pinned byte stream.
# Keep chunks at that observed ceiling; every response remains independently
# identity-checked before its body reaches the partial archive.
DOWNLOAD_RANGE_BYTES = 1 * 1024 * 1024
RANGE_USER_AGENT = "Unblur-SLAM-BSD-metadata-audit/1"
RANGE_CONTENT_TYPE = "application/octet-stream"
RANGE_FILENAME = "BSD_3ms24ms.zip"
CURL_BINARY = "/usr/bin/curl"
MAX_CURL_HEADER_BYTES = 64 * 1024


class AcquisitionError(RuntimeError):
    """Raised when any pinned identity or safety invariant fails."""


@dataclasses.dataclass(frozen=True)
class CentralEntry:
    name: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int


@dataclasses.dataclass(frozen=True)
class RangeRead:
    payload: bytes
    headers: Mapping[str, str]


def _read_exact_range_body(stream: BinaryIO, expected: int, label: str) -> bytes:
    """Read a bounded response fully even when its pipe yields short reads."""
    payload = bytearray()
    while len(payload) < expected:
        block = stream.read(min(BUFFER_BYTES, expected - len(payload)))
        if not block:
            raise AcquisitionError(
                f"{label} body ended early: {len(payload)} != {expected}"
            )
        payload.extend(block)
    if stream.read(1):
        raise AcquisitionError(f"{label} body exceeded Content-Length")
    return bytes(payload)


def _range_headers(
    headers: Mapping[str, str],
    *,
    start: int,
    end: int,
    total: int,
    label: str,
) -> Mapping[str, str]:
    normalized = {key.lower(): value.strip() for key, value in headers.items()}
    expected_length = end - start + 1
    if normalized.get("content-range") != f"bytes {start}-{end}/{total}":
        raise AcquisitionError(
            f"{label} Content-Range mismatch: "
            f"{normalized.get('content-range', '')!r}"
        )
    try:
        content_length = int(normalized.get("content-length", "-1"))
    except ValueError as error:
        raise AcquisitionError(f"{label} Content-Length is invalid") from error
    if content_length != expected_length:
        raise AcquisitionError(
            f"{label} Content-Length mismatch: {content_length} != {expected_length}"
        )
    if normalized.get("content-type", "").split(";", 1)[0] != RANGE_CONTENT_TYPE:
        raise AcquisitionError(
            f"{label} Content-Type mismatch: "
            f"{normalized.get('content-type', '')!r}"
        )
    if RANGE_FILENAME not in normalized.get("content-disposition", ""):
        raise AcquisitionError(f"{label} Content-Disposition filename mismatch")
    return normalized


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


class _CurlRangeStream:
    """Expose only a header-validated curl response body as a binary stream."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        headers: Mapping[str, str],
        expected_bytes: int,
        timeout: float,
    ) -> None:
        if process.stdout is None or process.stderr is None:
            _stop_process(process)
            raise AcquisitionError("curl pipes were not created")
        self._process = process
        self._stdout = process.stdout
        self._stderr = process.stderr
        self._expected = expected_bytes
        self._seen = 0
        self._timeout = max(float(timeout), 1.0)
        self._finished = False
        self._closed = False
        self.status = 206
        self.headers = dict(headers)

    def _finish(self) -> None:
        if self._finished:
            return
        try:
            return_code = self._process.wait(timeout=self._timeout)
        except subprocess.TimeoutExpired as error:
            _stop_process(self._process)
            raise AcquisitionError("curl did not exit after the bounded Range") from error
        stderr = self._stderr.read(4096).decode("utf-8", "replace").strip()
        if return_code != 0:
            raise AcquisitionError(
                f"curl Range exited {return_code}: {stderr or 'no diagnostic'}"
            )
        self._finished = True

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise ValueError("read of closed curl Range stream")
        remaining = self._expected - self._seen
        if remaining == 0:
            self._finish()
            return b""
        requested = remaining if size is None or size < 0 else min(size, remaining)
        payload = self._stdout.read(requested)
        if not payload:
            _stop_process(self._process)
            stderr = self._stderr.read(4096).decode("utf-8", "replace").strip()
            raise AcquisitionError(
                f"curl Range body ended early at {self._seen}/{self._expected}: "
                f"{stderr or 'no diagnostic'}"
            )
        self._seen += len(payload)
        if self._seen > self._expected:
            _stop_process(self._process)
            raise AcquisitionError("curl Range body exceeded Content-Length")
        if self._seen == self._expected:
            self._finish()
        return payload

    def close(self) -> None:
        if self._closed:
            return
        if not self._finished:
            _stop_process(self._process)
        self._stdout.close()
        self._stderr.close()
        self._closed = True


def _read_curl_header_line(stream: BinaryIO, consumed: list[int]) -> bytes:
    line = stream.readline(MAX_CURL_HEADER_BYTES + 1)
    consumed[0] += len(line)
    if not line or consumed[0] > MAX_CURL_HEADER_BYTES:
        raise AcquisitionError("curl response headers are missing or oversized")
    return line


def _open_curl_range(
    url: str, start: int, end: int, total: int, timeout: float
) -> BinaryIO:
    if not Path(CURL_BINARY).is_file() or not os.access(CURL_BINARY, os.X_OK):
        raise AcquisitionError(f"curl HTTP/2 fallback unavailable: {CURL_BINARY}")
    command = [
        CURL_BINARY,
        "--silent",
        "--show-error",
        "--http2",
        "--location",
        "--max-redirs",
        "5",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--suppress-connect-headers",
        "--include",
        "--no-buffer",
        "--connect-timeout",
        str(max(1, min(int(timeout), 60))),
        "--max-time",
        str(max(1, int(timeout))),
        "--user-agent",
        RANGE_USER_AGENT,
        "--range",
        f"{start}-{end}",
        url,
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as error:
        raise AcquisitionError("failed to start curl HTTP/2 fallback") from error
    assert process.stdout is not None
    consumed = [0]
    try:
        while True:
            status_line = _read_curl_header_line(process.stdout, consumed)
            while status_line in (b"\r\n", b"\n"):
                status_line = _read_curl_header_line(process.stdout, consumed)
            match = re.fullmatch(rb"HTTP/(?:1\.[01]|2|3) (\d{3})(?: .*)?\r?\n", status_line)
            if match is None:
                raise AcquisitionError(
                    f"curl returned an invalid HTTP status line: {status_line[:120]!r}"
                )
            status_code = int(match.group(1))
            headers: dict[str, str] = {}
            while True:
                line = _read_curl_header_line(process.stdout, consumed)
                if line in (b"\r\n", b"\n"):
                    break
                if b":" not in line:
                    raise AcquisitionError("curl returned a malformed HTTP header")
                key, value = line.split(b":", 1)
                name = key.decode("ascii", "strict").strip().lower()
                decoded = value.decode("latin-1", "strict").strip()
                headers[name] = (
                    f"{headers[name]}, {decoded}" if name in headers else decoded
                )
            if 100 <= status_code < 200 or 300 <= status_code < 400:
                continue
            if status_code != 206:
                content_type = headers.get("content-type", "")
                raise AcquisitionError(
                    f"curl Range returned HTTP {status_code} ({content_type or 'no type'}), "
                    "refusing its body"
                )
            normalized = _range_headers(
                headers,
                start=start,
                end=end,
                total=total,
                label="curl Range",
            )
            return _CurlRangeStream(
                process, normalized, end - start + 1, timeout
            )
    except BaseException:
        _stop_process(process)
        if process.stderr is not None:
            process.stderr.close()
        if process.stdout is not None:
            process.stdout.close()
        raise


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, progress: bool = False) -> str:
    digest = hashlib.sha256()
    processed = 0
    last_report = time.monotonic()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER_BYTES), b""):
            digest.update(block)
            processed += len(block)
            if progress and time.monotonic() - last_report >= 30:
                print(
                    f"sha256: {processed / (1024**3):.2f} GiB read",
                    file=sys.stderr,
                    flush=True,
                )
                last_report = time.monotonic()
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> tuple[Mapping[str, Any], str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise AcquisitionError(f"protocol must be a regular file: {resolved}")
    payload = resolved.read_bytes()
    digest = sha256_bytes(payload)
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise AcquisitionError(
            f"frozen protocol SHA-256 mismatch: {digest} != "
            f"{EXPECTED_PROTOCOL_SHA256}"
        )
    try:
        protocol = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcquisitionError("frozen protocol is not valid JSON") from error
    if not isinstance(protocol, Mapping):
        raise AcquisitionError("frozen protocol must be an object")
    if protocol.get("schema") != SCHEMA or protocol.get("status") != "frozen":
        raise AcquisitionError("protocol schema/status drifted")
    return protocol, digest


def _http_head(url: str, timeout: float) -> Mapping[str, str]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "Unblur-SLAM-BSD-acquisition/1"},
    )
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise AcquisitionError(f"HEAD returned HTTP {response.status}")
                return {key.lower(): value for key, value in response.headers.items()}
        except (OSError, urllib.error.URLError, AcquisitionError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise AcquisitionError("official object HEAD failed") from last_error


def _http_range_chunk(
    url: str, start: int, end: int, total: int, timeout: float
) -> RangeRead:
    """Fetch and validate exactly one bounded metadata byte range."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": RANGE_USER_AGENT,
            "Range": f"bytes={start}-{end}",
        },
    )
    expected = end - start + 1
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
                if response.status != 206:
                    raise AcquisitionError(
                        f"metadata Range returned HTTP {response.status}, not 206"
                    )
                headers = _range_headers(
                    headers,
                    start=start,
                    end=end,
                    total=total,
                    label="urllib metadata Range",
                )
                payload = _read_exact_range_body(
                    response, expected, "urllib metadata Range"
                )
                return RangeRead(payload, headers)
        except (OSError, urllib.error.URLError, AcquisitionError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    try:
        stream = _open_curl_range(url, start, end, total, timeout)
        try:
            payload = _read_exact_range_body(
                stream, expected, "curl metadata Range"
            )
            return RangeRead(payload, dict(getattr(stream, "headers")))
        finally:
            stream.close()
    except AcquisitionError as curl_error:
        raise AcquisitionError(
            f"official ZIP metadata Range failed via urllib ({last_error}) "
            f"and curl HTTP/2 ({curl_error})"
        ) from curl_error


def _http_range(url: str, start: int, end: int, timeout: float) -> RangeRead:
    """Fetch a metadata range in independently verified bounded chunks."""
    if start < 0 or end < start:
        raise AcquisitionError(f"invalid metadata Range: {start}-{end}")
    total = end + 1
    expected = end - start + 1
    cursor = start
    payload = bytearray()
    first_headers: dict[str, str] | None = None
    while cursor <= end:
        chunk_end = min(end, cursor + DOWNLOAD_RANGE_BYTES - 1)
        chunk = _http_range_chunk(url, cursor, chunk_end, total, timeout)
        chunk_headers = {
            key.lower(): value for key, value in chunk.headers.items()
        }
        if first_headers is None:
            first_headers = dict(chunk_headers)
        elif chunk_headers.get("last-modified") != first_headers.get("last-modified"):
            raise AcquisitionError("metadata Range Last-Modified changed across chunks")
        payload.extend(chunk.payload)
        cursor = chunk_end + 1
    if len(payload) != expected or first_headers is None:
        raise AcquisitionError(
            f"metadata Range aggregate length mismatch: {len(payload)} != {expected}"
        )
    # Present the caller with the logical aggregate Range while preserving the
    # first chunk's publisher identity headers.  Every physical chunk has
    # already passed the exact 206/Content-Range/Length/type/name contract.
    aggregate_headers = dict(first_headers)
    aggregate_headers["content-range"] = f"bytes {start}-{end}/{total}"
    aggregate_headers["content-length"] = str(expected)
    return RangeRead(bytes(payload), aggregate_headers)


def _zip64_sizes(
    compressed: int, uncompressed: int, extra: bytes, name: str
) -> tuple[int, int]:
    needs_compressed = compressed == 0xFFFFFFFF
    needs_uncompressed = uncompressed == 0xFFFFFFFF
    if not needs_compressed and not needs_uncompressed:
        return compressed, uncompressed
    offset = 0
    while offset + 4 <= len(extra):
        field_id, field_len = struct.unpack("<HH", extra[offset : offset + 4])
        value = extra[offset + 4 : offset + 4 + field_len]
        offset += 4 + field_len
        if field_id != 1:
            continue
        cursor = 0
        if needs_uncompressed:
            if cursor + 8 > len(value):
                raise AcquisitionError(f"truncated ZIP64 size for {name}")
            uncompressed = struct.unpack("<Q", value[cursor : cursor + 8])[0]
            cursor += 8
        if needs_compressed:
            if cursor + 8 > len(value):
                raise AcquisitionError(f"truncated ZIP64 size for {name}")
            compressed = struct.unpack("<Q", value[cursor : cursor + 8])[0]
        return compressed, uncompressed
    raise AcquisitionError(f"missing ZIP64 size extra field for {name}")


def parse_central_directory(payload: bytes) -> list[CentralEntry]:
    header = struct.Struct("<4s6H3L5H2L")
    entries: list[CentralEntry] = []
    seen: set[str] = set()
    offset = 0
    while offset < len(payload):
        if offset + header.size > len(payload):
            raise AcquisitionError("truncated central-directory header")
        fields = header.unpack(payload[offset : offset + header.size])
        if fields[0] != b"PK\x01\x02":
            raise AcquisitionError(f"invalid central-directory signature at {offset}")
        flags = fields[3]
        crc32 = fields[7]
        compressed = fields[8]
        uncompressed = fields[9]
        name_len, extra_len, comment_len = fields[10:13]
        entry_end = offset + header.size + name_len + extra_len + comment_len
        if entry_end > len(payload):
            raise AcquisitionError("truncated central-directory entry")
        name_bytes = payload[offset + header.size : offset + header.size + name_len]
        extra_start = offset + header.size + name_len
        extra = payload[extra_start : extra_start + extra_len]
        try:
            name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        except UnicodeDecodeError as error:
            raise AcquisitionError("invalid central-directory filename encoding") from error
        parts = [part for part in name.split("/") if part]
        if (
            not name
            or "\x00" in name
            or "\\" in name
            or name.startswith("/")
            or ".." in parts
        ):
            raise AcquisitionError(f"unsafe ZIP member path: {name!r}")
        if name in seen:
            raise AcquisitionError(f"duplicate ZIP member path: {name!r}")
        seen.add(name)
        compressed, uncompressed = _zip64_sizes(
            compressed, uncompressed, extra, name
        )
        entries.append(CentralEntry(name, compressed, uncompressed, crc32))
        offset = entry_end
    return entries


def _entry_index_sha256(entries: list[CentralEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.name):
        digest.update(
            (
                f"{entry.name}\0{entry.compressed_bytes}\0"
                f"{entry.uncompressed_bytes}\0{entry.crc32:08x}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def audit_central_directory(
    entries: list[CentralEntry], protocol: Mapping[str, Any]
) -> Mapping[str, Any]:
    identity = protocol["remote_zip_identity"]
    if len(entries) != int(identity["entries_total"]):
        raise AcquisitionError("ZIP entry count drifted")
    if _entry_index_sha256(entries) != identity["all_entry_index_sha256"]:
        raise AcquisitionError("ZIP all-entry index identity drifted")

    payload_root = str(identity["payload_root"])
    macos_root = str(identity["macos_metadata_root"])
    real = [entry for entry in entries if entry.name.startswith(payload_root)]
    macos = [entry for entry in entries if entry.name.startswith(macos_root)]
    if len(real) != int(identity["payload_tree_entries"]):
        raise AcquisitionError("BSD payload-tree entry count drifted")
    if len(macos) != int(identity["macos_metadata_entries"]):
        raise AcquisitionError("__MACOSX metadata entry count drifted")
    if len(real) + len(macos) != len(entries):
        raise AcquisitionError("unexpected ZIP root is present")

    pattern = re.compile(
        r"^BSD_3ms24ms/(train|valid|test)/(\d{3})/"
        r"(Blur|Sharp)/RGB/(\d{8})\.png$"
    )
    parsed: dict[str, list[tuple[CentralEntry, str, str, int]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    for entry in real:
        match = pattern.fullmatch(entry.name)
        if match:
            split, sequence, role, frame = match.groups()
            parsed[split].append((entry, sequence, role, int(frame)))
        elif entry.name.endswith(".png"):
            raise AcquisitionError(f"unexpected PNG layout: {entry.name}")
        elif not entry.name.endswith("/") and entry.name != "BSD_3ms24ms/.DS_Store":
            raise AcquisitionError(f"unexpected non-PNG payload member: {entry.name}")

    sequence_sets: dict[str, set[str]] = {}
    split_report: dict[str, Any] = {}
    for split, expected in identity["splits"].items():
        rows = parsed[split]
        png_entries = [row[0] for row in rows]
        if len(rows) != int(expected["png_members"]):
            raise AcquisitionError(f"{split} PNG-member count drifted")
        if _entry_index_sha256(png_entries) != expected["png_index_sha256"]:
            raise AcquisitionError(f"{split} PNG index identity drifted")
        sequences = {row[1] for row in rows}
        sequence_sets[split] = sequences
        if len(sequences) != int(expected["sequences"]):
            raise AcquisitionError(f"{split} sequence count drifted")
        expected_frames = set(range(int(expected["frames_per_sequence"])))
        pairs: dict[tuple[str, int], set[str]] = {}
        for _, sequence, role, frame in rows:
            pairs.setdefault((sequence, frame), set()).add(role)
        if len(pairs) != int(expected["pairs"]):
            raise AcquisitionError(f"{split} pair count drifted")
        for sequence in sequences:
            frames = {frame for seq, frame in pairs if seq == sequence}
            if frames != expected_frames:
                raise AcquisitionError(f"{split}/{sequence} frame index set drifted")
        if any(roles != {"Blur", "Sharp"} for roles in pairs.values()):
            raise AcquisitionError(f"{split} Blur/Sharp pairing drifted")
        compressed = sum(entry.compressed_bytes for entry in png_entries)
        uncompressed = sum(entry.uncompressed_bytes for entry in png_entries)
        if compressed != int(expected["compressed_png_bytes"]):
            raise AcquisitionError(f"{split} compressed byte count drifted")
        if uncompressed != int(expected["uncompressed_png_bytes"]):
            raise AcquisitionError(f"{split} uncompressed byte count drifted")
        split_report[split] = {
            "sequences": len(sequences),
            "pairs": len(pairs),
            "png_members": len(rows),
            "png_index_sha256": expected["png_index_sha256"],
        }
    split_names = list(sequence_sets)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            if sequence_sets[first] & sequence_sets[second]:
                raise AcquisitionError(f"sequence IDs overlap: {first}/{second}")
    if sum(len(value) for value in parsed.values()) != int(identity["real_png_members"]):
        raise AcquisitionError("total real PNG count drifted")
    return {
        "entries_total": len(entries),
        "payload_tree_entries": len(real),
        "macos_metadata_entries": len(macos),
        "real_png_members": sum(len(value) for value in parsed.values()),
        "splits": split_report,
        "test_member_payload_bytes_read": 0,
    }


def preflight_remote(
    protocol: Mapping[str, Any],
    *,
    timeout: float = 60.0,
    head: Callable[[str, float], Mapping[str, str]] = _http_head,
    fetch_range: Callable[[str, int, int, float], bytes | RangeRead] = _http_range,
) -> Mapping[str, Any]:
    source = protocol["official_source"]
    url = str(source["download_url"])
    head_headers: Mapping[str, str] = {}
    head_error: str | None = None
    try:
        head_headers = {
            key.lower(): value for key, value in head(url, timeout).items()
        }
    except AcquisitionError as error:
        # Google Drive occasionally rejects HEAD while continuing to serve a
        # byte range.  The pinned tail Range below is authoritative because it
        # binds the total through Content-Range and the complete CD/trailer.
        head_error = str(error)
    expected_bytes = int(source["content_length_bytes"])
    identity = protocol["remote_zip_identity"]
    start = int(identity["central_directory_offset"])
    end = expected_bytes - 1
    range_value = fetch_range(url, start, end, timeout)
    if isinstance(range_value, RangeRead):
        tail = range_value.payload
        range_headers = {
            key.lower(): value for key, value in range_value.headers.items()
        }
        # For a resume/range response Content-Length is only the remaining
        # range length; total object identity comes from Content-Range.
        if range_headers.get("content-range") != f"bytes {start}-{end}/{expected_bytes}":
            raise AcquisitionError("official object Content-Range total drifted")
        if int(range_headers.get("content-length", "-1")) != end - start + 1:
            raise AcquisitionError("official Range Content-Length drifted")
        if (
            range_headers.get("content-type", "").split(";", 1)[0]
            != source["content_type"]
        ):
            raise AcquisitionError("official Range Content-Type drifted")
        if str(source["filename"]) not in range_headers.get(
            "content-disposition", ""
        ):
            raise AcquisitionError("official Range filename drifted")
        if range_headers.get("last-modified") != source["last_modified_http"]:
            raise AcquisitionError("official Range Last-Modified drifted")
    else:
        # Test transports may supply only bytes.  In that case the HEAD must
        # carry a full-object identity; production never takes this branch.
        tail = range_value
        if int(head_headers.get("content-length", "-1")) != expected_bytes:
            raise AcquisitionError("official object Content-Length drifted")
        if (
            head_headers.get("content-type", "").split(";", 1)[0]
            != source["content_type"]
        ):
            raise AcquisitionError("official object Content-Type drifted")
        if str(source["filename"]) not in head_headers.get(
            "content-disposition", ""
        ):
            raise AcquisitionError("official object filename drifted")
        if head_headers.get("last-modified") != source["last_modified_http"]:
            raise AcquisitionError("official object Last-Modified drifted")
    central_size = int(identity["central_directory_bytes"])
    central = tail[:central_size]
    if sha256_bytes(central) != identity["central_directory_sha256"]:
        raise AcquisitionError("remote ZIP central-directory SHA-256 drifted")
    for label in ("zip64_eocd", "zip64_locator", "eocd"):
        record = identity[label]
        relative = int(record["offset"]) - start
        payload = tail[relative : relative + int(record["bytes"])]
        if sha256_bytes(payload) != record["sha256"]:
            raise AcquisitionError(f"remote ZIP {label} identity drifted")
    directory_report = audit_central_directory(
        parse_central_directory(central), protocol
    )
    return {
        "source_identity_verified": True,
        "head_identity": {
            "available": bool(head_headers),
            "error": head_error,
            "content_length": head_headers.get("content-length"),
            "content_range": head_headers.get("content-range"),
        },
        "content_length_bytes": expected_bytes,
        "metadata_range": [start, end],
        "metadata_bytes_read": len(tail),
        "central_directory_sha256": identity["central_directory_sha256"],
        "directory": directory_report,
        "opaque_archive_payload_bytes_read": 0,
        "test_member_payload_bytes_read": 0,
    }


def audit_official_turtle_checkpoint(
    protocol: Mapping[str, Any], protocol_sha256: str
) -> Mapping[str, Any]:
    expected = protocol["official_turtle_checkpoint"]
    path = Path(str(expected["quarantine_path"]))
    if path.is_symlink() or not path.is_file():
        raise AcquisitionError(f"official TURTLE checkpoint missing/unsafe: {path}")
    info = path.stat()
    if info.st_size != int(expected["content_length_bytes"]):
        raise AcquisitionError("official TURTLE checkpoint byte count drifted")
    mode = stat.S_IMODE(info.st_mode)
    expected_mode = int(str(expected["observed_mode_octal"]), 8)
    if mode != expected_mode:
        raise AcquisitionError(
            f"official TURTLE checkpoint mode drifted: {mode:o} != {expected_mode:o}"
        )
    digest = sha256_file(path)
    if digest != expected["sha256"]:
        raise AcquisitionError("official TURTLE checkpoint SHA-256 drifted")
    return {
        "path": str(path),
        "bytes": info.st_size,
        "sha256": digest,
        "mode_octal": f"{mode:04o}",
        "protocol_sha256": protocol_sha256,
        "model_type": expected["model_type"],
        "architecture_implementation": expected["architecture_implementation"],
        "state_dict_tensor_keys": expected["state_dict_tensor_keys"],
        "parameters": expected["parameters"],
        "verified": True,
    }


def _atomic_write_new(path: Path, payload: bytes, mode: int) -> None:
    if path.exists() or path.is_symlink():
        raise AcquisitionError(f"refusing to overwrite receipt: {path}")
    temporary = path.with_name(f".{path.name}.staging-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(temporary, flags, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def write_checkpoint_receipt(
    protocol: Mapping[str, Any], protocol_sha256: str
) -> Path:
    audit = audit_official_turtle_checkpoint(protocol, protocol_sha256)
    expected = protocol["official_turtle_checkpoint"]
    path = Path(str(expected["quarantine_path"]))
    receipt_path = path.with_name(f"{path.name}.acquisition.json")
    receipt = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_protocol": {
            "path": str(DEFAULT_PROTOCOL),
            "sha256": protocol_sha256,
        },
        "official_source": {
            "repository_url": expected["repository_url"],
            "repository_commit": expected["repository_commit"],
            "google_drive_folder_id": expected["google_drive_folder_id"],
            "google_drive_file_id": expected["google_drive_file_id"],
            "filename": expected["filename"],
        },
        "artifact": audit,
        "architecture_contract": {
            "model_type": "t0",
            "implementation": "basicsr/models/archs/turtle_arch.py",
            "not_compatible_with_current_gopro_t1_backend": True,
        },
    }
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            existing.get("frozen_protocol", {}).get("sha256") != protocol_sha256
            or existing.get("artifact", {}).get("sha256") != expected["sha256"]
        ):
            raise AcquisitionError("existing checkpoint receipt conflicts")
        return receipt_path
    _atomic_write_new(receipt_path, _canonical_bytes(receipt), 0o444)
    return receipt_path


def _validate_quarantine_root(path: Path, protocol: Mapping[str, Any]) -> Path:
    required = Path(protocol["storage_policy"]["required_filesystem_root"]).resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(required)
    except ValueError as error:
        raise AcquisitionError(
            f"quarantine must remain under {required}, got {resolved}"
        ) from error
    if resolved == required:
        raise AcquisitionError("quarantine may not equal the broad filesystem root")
    return resolved


def _open_download_range(
    url: str, start: int, end: int, timeout: float
) -> BinaryIO:
    request_end = min(end, start + DOWNLOAD_RANGE_BYTES - 1)
    request = urllib.request.Request(
        url,
        headers={
            # Google Drive currently serves the pinned byte-range object to
            # the metadata-audit identity while returning a quota HTML page
            # to a distinct acquisition identity.  Keep one content-addressed
            # client identity for metadata and opaque payload ranges.
            "User-Agent": RANGE_USER_AGENT,
            "Range": f"bytes={start}-{request_end}",
        },
    )
    response: BinaryIO | None = None
    urllib_error: BaseException | None = None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
        if getattr(response, "status", None) != 206:
            raise AcquisitionError(
                f"urllib download Range returned HTTP "
                f"{getattr(response, 'status', None)}, not 206"
            )
        _range_headers(
            {key.lower(): value for key, value in response.headers.items()},
            start=start,
            end=request_end,
            total=end + 1,
            label="urllib download Range",
        )
        return response
    except (OSError, urllib.error.URLError, AcquisitionError) as error:
        urllib_error = error
        if response is not None:
            response.close()
    try:
        return _open_curl_range(url, start, request_end, end + 1, timeout)
    except AcquisitionError as curl_error:
        raise AcquisitionError(
            f"download Range failed via urllib ({urllib_error}) and "
            f"curl HTTP/2 ({curl_error})"
        ) from curl_error


def _validate_existing_receipt(
    receipt_path: Path,
    archive_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> Mapping[str, Any]:
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise AcquisitionError("archive exists without a regular acquisition receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise AcquisitionError("acquisition receipt schema drifted")
    if receipt.get("frozen_protocol", {}).get("sha256") != protocol_sha256:
        raise AcquisitionError("acquisition receipt protocol binding drifted")
    expected = protocol["official_source"]
    observed = receipt.get("archive", {})
    if observed.get("bytes") != expected["content_length_bytes"]:
        raise AcquisitionError("acquisition receipt byte count drifted")
    if archive_path.stat().st_size != expected["content_length_bytes"]:
        raise AcquisitionError("quarantined archive byte count drifted")
    digest = sha256_file(archive_path, progress=True)
    if digest != observed.get("sha256"):
        raise AcquisitionError("quarantined archive SHA-256 drifted")
    return receipt


def acquire_archive(
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    quarantine: Path,
    *,
    acknowledge_opaque_test_payload: bool,
    timeout: float = 120.0,
    open_range: Callable[[str, int, int, float], BinaryIO] = _open_download_range,
) -> tuple[Path, Path]:
    if not acknowledge_opaque_test_payload:
        raise AcquisitionError(
            "download requires --acknowledge-opaque-test-payload"
        )
    root = _validate_quarantine_root(quarantine, protocol)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, int(protocol["storage_policy"]["quarantine_directory_mode_octal"], 8))
    free = shutil.disk_usage(root).free
    minimum = int(protocol["storage_policy"]["minimum_free_bytes_before_new_download"])
    if free < minimum:
        raise AcquisitionError(f"insufficient free space: {free} < {minimum}")

    source = protocol["official_source"]
    total = int(source["content_length_bytes"])
    archive = root / str(source["filename"])
    # Deliberately non-hidden: this is also the recovery name used by the
    # initial metadata-gated acquisition, so interrupted transfers resume in
    # place instead of allocating a second multi-gigabyte partial.
    partial = root / f"{archive.name}.partial"
    receipt_path = root / f"{archive.name}.acquisition.json"
    lock_path = root / ".acquisition.lock"
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise AcquisitionError("another BSD acquisition holds the lock") from error
    try:
        if archive.exists() or archive.is_symlink():
            if archive.is_symlink() or not archive.is_file():
                raise AcquisitionError("quarantined archive path is unsafe")
            _validate_existing_receipt(
                receipt_path, archive, protocol, protocol_sha256
            )
            return archive, receipt_path
        if receipt_path.exists():
            raise AcquisitionError("receipt exists without the archive")
        if partial.is_symlink() or (partial.exists() and not partial.is_file()):
            raise AcquisitionError("partial archive path is unsafe")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > total:
            raise AcquisitionError("partial archive is larger than pinned object")

        attempts_without_progress = 0
        last_report = time.monotonic()
        while offset < total:
            before = offset
            try:
                response = open_range(str(source["download_url"]), offset, total - 1, timeout)
                flags = os.O_WRONLY | os.O_CREAT
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(partial, flags, 0o600)
                try:
                    os.lseek(fd, offset, os.SEEK_SET)
                    with os.fdopen(fd, "wb", closefd=True) as output:
                        while offset < total:
                            block = response.read(min(BUFFER_BYTES, total - offset))
                            if not block:
                                break
                            output.write(block)
                            offset += len(block)
                            if time.monotonic() - last_report >= 30:
                                print(
                                    f"download: {offset / total:.1%} "
                                    f"({offset / (1024**3):.2f} GiB)",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                last_report = time.monotonic()
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    response.close()
            except (OSError, urllib.error.URLError) as error:
                print(f"download retry after: {error}", file=sys.stderr, flush=True)
            if offset == before:
                attempts_without_progress += 1
                if attempts_without_progress >= 3:
                    raise AcquisitionError("download made no progress after three attempts")
                time.sleep(attempts_without_progress)
            else:
                attempts_without_progress = 0

        if partial.stat().st_size != total:
            raise AcquisitionError("completed partial has the wrong byte count")
        digest = sha256_file(partial, progress=True)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_protocol": {
                "path": str(DEFAULT_PROTOCOL),
                "sha256": protocol_sha256,
            },
            "official_source": {
                "google_drive_file_id": source["google_drive_file_id"],
                "filename": source["filename"],
                "content_length_bytes": total,
                "remote_zip_central_directory_sha256": protocol[
                    "remote_zip_identity"
                ]["central_directory_sha256"],
            },
            "archive": {
                "path": str(archive),
                "bytes": total,
                "sha256": digest,
                "mode_octal": protocol["storage_policy"]
                ["archive_mode_octal_after_acquisition"],
            },
            "exposure_attestation": {
                "opaque_archive_payload_transferred": True,
                "opaque_test_payload_bytes_present_in_archive": True,
                "test_member_payload_opened": False,
                "test_member_payload_decompressed": False,
                "test_image_decoded": False,
                "whole_archive_sha256_is_not_an_individual_member_hash": True,
            },
        }
        staged_receipt = root / f".{receipt_path.name}.ready-{os.getpid()}"
        _atomic_write_new(staged_receipt, _canonical_bytes(receipt), 0o400)
        os.chmod(partial, int(protocol["storage_policy"]["archive_mode_octal_after_acquisition"], 8))
        os.replace(partial, archive)
        os.replace(staged_receipt, receipt_path)
        os.chmod(receipt_path, 0o444)
        return archive, receipt_path
    finally:
        os.close(lock_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--quarantine",
        type=Path,
        help="owner-only archive directory; defaults to the frozen protocol value",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="after remote preflight, acquire/resume the opaque archive",
    )
    parser.add_argument(
        "--acknowledge-opaque-test-payload",
        action="store_true",
        help="required with --download because the publisher ships all splits together",
    )
    parser.add_argument(
        "--write-checkpoint-receipt",
        action="store_true",
        help="verify the official BSD t0 checkpoint and write its immutable receipt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.acknowledge_opaque_test_payload and not args.download:
        raise AcquisitionError(
            "--acknowledge-opaque-test-payload is meaningful only with --download"
        )
    protocol, protocol_sha256 = load_protocol(args.protocol)
    remote = preflight_remote(protocol, timeout=args.timeout)
    checkpoint = audit_official_turtle_checkpoint(protocol, protocol_sha256)
    report: dict[str, Any] = {
        "schema": "unblur_slam.bsd_3ms24ms_acquisition_preflight.v1",
        "protocol": {
            "path": str(args.protocol.expanduser().resolve()),
            "sha256": protocol_sha256,
            "status": "frozen",
        },
        "remote": remote,
        "official_turtle_checkpoint": checkpoint,
        "download_started": False,
    }
    if args.write_checkpoint_receipt:
        report["official_turtle_checkpoint"]["receipt"] = str(
            write_checkpoint_receipt(protocol, protocol_sha256)
        )
    if args.download:
        quarantine = args.quarantine or Path(
            protocol["storage_policy"]["default_quarantine_directory"]
        )
        archive, receipt = acquire_archive(
            protocol,
            protocol_sha256,
            quarantine,
            acknowledge_opaque_test_payload=args.acknowledge_opaque_test_payload,
            timeout=args.timeout,
        )
        report["download_started"] = True
        report["archive"] = str(archive)
        report["archive_receipt"] = str(receipt)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcquisitionError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
