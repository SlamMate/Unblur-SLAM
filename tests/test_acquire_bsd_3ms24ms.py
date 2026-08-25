"""CPU-only, no-network tests for the frozen BSD acquisition contract."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.acquire_bsd_3ms24ms as module  # noqa: E402


def _central_directory(*names: str) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for index, name in enumerate(names):
            archive.writestr(name, f"fixture-{index}".encode())
    payload = stream.getvalue()
    eocd = payload.rfind(b"PK\x05\x06")
    assert eocd >= 0
    size = int.from_bytes(payload[eocd + 12 : eocd + 16], "little")
    offset = int.from_bytes(payload[eocd + 16 : eocd + 20], "little")
    return payload[offset : offset + size]


class BsdAcquisitionTests(unittest.TestCase):
    def test_frozen_protocol_hash_split_policy_and_t0_checkpoint(self) -> None:
        protocol, digest = module.load_protocol()
        self.assertEqual(digest, module.EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(protocol["status"], "frozen")
        self.assertEqual(
            protocol["split_policy"]["authorized_for_materialization"],
            ["train", "valid"],
        )
        self.assertEqual(protocol["split_policy"]["sealed_split"], "test")
        checkpoint = protocol["official_turtle_checkpoint"]
        self.assertEqual(checkpoint["model_type"], "t0")
        self.assertEqual(checkpoint["state_dict_tensor_keys"], 621)
        self.assertEqual(checkpoint["parameters"], 58_620_800)
        self.assertIn("GoPro t1", checkpoint["incompatible_backend_warning"])

    def test_protocol_byte_tamper_is_rejected(self) -> None:
        payload = module.DEFAULT_PROTOCOL.read_bytes().replace(
            b'"status": "frozen"', b'"status": "changed"'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_bytes(payload)
            with self.assertRaisesRegex(module.AcquisitionError, "SHA-256"):
                module.load_protocol(path)

    def test_central_parser_accepts_safe_paths_and_rejects_traversal(self) -> None:
        central = _central_directory(
            "BSD_3ms24ms/train/001/Blur/RGB/00000000.png",
            "BSD_3ms24ms/train/001/Sharp/RGB/00000000.png",
        )
        entries = module.parse_central_directory(central)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].uncompressed_bytes, len(b"fixture-0"))

        unsafe = _central_directory("../test/secret.png")
        with self.assertRaisesRegex(module.AcquisitionError, "unsafe"):
            module.parse_central_directory(unsafe)

    def test_duplicate_member_names_are_rejected(self) -> None:
        with self.assertWarns(UserWarning):
            central = _central_directory("same.png", "same.png")
        with self.assertRaisesRegex(module.AcquisitionError, "duplicate"):
            module.parse_central_directory(central)

    def test_resume_range_content_length_is_remaining_not_total(self) -> None:
        frozen, _ = module.load_protocol()
        protocol = copy.deepcopy(frozen)
        central = _central_directory(
            "BSD_3ms24ms/train/001/Blur/RGB/00000000.png"
        )
        trailer_parts = [b"z" * 56, b"l" * 20, b"e" * 22]
        start = 100
        cursor = start + len(central)
        identity = protocol["remote_zip_identity"]
        identity["central_directory_offset"] = start
        identity["central_directory_bytes"] = len(central)
        identity["central_directory_sha256"] = hashlib.sha256(central).hexdigest()
        for label, payload in zip(
            ("zip64_eocd", "zip64_locator", "eocd"), trailer_parts
        ):
            identity[label] = {
                "offset": cursor,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            cursor += len(payload)
        tail = central + b"".join(trailer_parts)
        source = protocol["official_source"]
        source["content_length_bytes"] = cursor
        calls: list[tuple[int, int]] = []

        def head(url: str, timeout: float):
            del url, timeout
            resume_start = 50
            return {
                # A resumed response describes only the remaining bytes.
                "content-length": str(cursor - resume_start),
                "content-range": f"bytes {resume_start}-{cursor - 1}/{cursor}",
                "content-type": source["content_type"],
                "content-disposition": f'attachment; filename="{source["filename"]}"',
                "last-modified": source["last_modified_http"],
                "accept-ranges": "bytes",
            }

        def fetch(url: str, first: int, end: int, timeout: float):
            del url, timeout
            calls.append((first, end))
            return module.RangeRead(
                tail,
                {
                    "content-length": str(len(tail)),
                    "content-range": f"bytes {first}-{end}/{cursor}",
                    "content-type": source["content_type"],
                    "content-disposition": (
                        f'attachment; filename="{source["filename"]}"'
                    ),
                    "last-modified": source["last_modified_http"],
                },
            )

        with mock.patch.object(
            module,
            "audit_central_directory",
            return_value={"test_member_payload_bytes_read": 0},
        ):
            report = module.preflight_remote(
                protocol, head=head, fetch_range=fetch
            )
        self.assertEqual(calls, [(start, cursor - 1)])
        self.assertEqual(report["content_length_bytes"], cursor)
        self.assertEqual(report["head_identity"]["content_length"], str(cursor - 50))

    def test_download_requires_explicit_test_payload_acknowledgement(self) -> None:
        protocol, digest = module.load_protocol()
        calls: list[str] = []
        with self.assertRaisesRegex(module.AcquisitionError, "acknowledge"):
            module.acquire_archive(
                protocol,
                digest,
                Path("/srv/szha0669/should-not-be-created"),
                acknowledge_opaque_test_payload=False,
                open_range=lambda *args: calls.append("network") or io.BytesIO(),
            )
        self.assertEqual(calls, [])

    def test_download_range_uses_the_verified_metadata_user_agent(self) -> None:
        observed: list[tuple[str | None, str | None]] = []

        class Response(io.BytesIO):
            status = 206
            headers = {
                "Content-Range": "bytes 50-99/100",
                "Content-Length": "50",
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="BSD_3ms24ms.zip"',
            }

        def open_url(request, timeout):
            del timeout
            observed.append(
                (request.get_header("User-agent"), request.get_header("Range"))
            )
            return Response(b"x" * 50)

        with mock.patch.object(module.urllib.request, "urlopen", open_url):
            response = module._open_download_range("https://example.invalid", 50, 99, 1)
            self.assertEqual(response.read(), b"x" * 50)
            response.close()
        self.assertEqual(
            observed,
            [("Unblur-SLAM-BSD-metadata-audit/1", "bytes=50-99")],
        )

    def test_curl_http2_fallback_validates_headers_before_streaming_body(self) -> None:
        command: list[str] = []

        class BadUrllib(io.BytesIO):
            status = 200
            headers = {
                "Content-Type": "text/html; charset=utf-8",
                "Content-Length": "12",
            }

        class FakeProcess:
            def __init__(self, payload: bytes):
                self.stdout = io.BytesIO(payload)
                self.stderr = io.BytesIO()
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                del timeout
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        body = b"verified-curl-body"
        headers = (
            b"HTTP/2 206\r\n"
            b"content-range: bytes 50-67/100\r\n"
            b"content-length: 18\r\n"
            b"content-type: application/octet-stream\r\n"
            b'content-disposition: attachment; filename="BSD_3ms24ms.zip"\r\n'
            b"\r\n"
        )

        def popen(argv, **kwargs):
            del kwargs
            command.extend(argv)
            return FakeProcess(headers + body)

        with mock.patch.object(
            module.urllib.request,
            "urlopen",
            return_value=BadUrllib(b"quota-html"),
        ), mock.patch.object(module.subprocess, "Popen", popen), mock.patch.object(
            module, "DOWNLOAD_RANGE_BYTES", 18
        ):
            response = module._open_download_range(
                "https://example.invalid", 50, 99, 1
            )
            self.assertEqual(response.read(), body)
            self.assertEqual(response.read(), b"")
            response.close()
        self.assertIn("--http2", command)
        self.assertEqual(command[command.index("--range") + 1], "50-67")
        self.assertEqual(
            command[command.index("--user-agent") + 1],
            "Unblur-SLAM-BSD-metadata-audit/1",
        )

    def test_curl_quota_html_is_rejected_before_body_read(self) -> None:
        class CountingPipe(io.BytesIO):
            def __init__(self, payload: bytes):
                super().__init__(payload)
                self.body_read_calls = 0

            def read(self, size=-1):
                self.body_read_calls += 1
                return super().read(size)

        class BadUrllib(io.BytesIO):
            status = 200
            headers = {"Content-Type": "text/html", "Content-Length": "4"}

        class FakeProcess:
            def __init__(self):
                self.stdout = CountingPipe(
                    b"HTTP/2 200\r\ncontent-type: text/html\r\n"
                    b"content-length: 18\r\n\r\nquota exceeded html"
                )
                self.stderr = io.BytesIO()
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                del timeout
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        with mock.patch.object(
            module.urllib.request,
            "urlopen",
            side_effect=lambda *args, **kwargs: BadUrllib(b"html"),
        ), mock.patch.object(module.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(module.AcquisitionError, "HTTP 200"):
                module._open_download_range("https://example.invalid", 50, 99, 1)
        self.assertEqual(process.stdout.body_read_calls, 0)

    def test_metadata_range_also_uses_curl_http2_fallback(self) -> None:
        class BadUrllib(io.BytesIO):
            status = 200
            headers = {"Content-Type": "text/html", "Content-Length": "4"}

        class FragmentedPipe(io.BytesIO):
            def read(self, size=-1):
                if size is None or size < 0:
                    size = 3
                return super().read(min(size, 3))

        class FakeProcess:
            def __init__(self):
                self.stdout = FragmentedPipe(
                    b"HTTP/2 206\r\n"
                    b"content-range: bytes 50-59/60\r\n"
                    b"content-length: 10\r\n"
                    b"content-type: application/octet-stream\r\n"
                    b'content-disposition: attachment; filename="BSD_3ms24ms.zip"\r\n'
                    b"\r\n0123456789"
                )
                self.stderr = io.BytesIO()
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                del timeout
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        with mock.patch.object(
            module.urllib.request,
            "urlopen",
            side_effect=lambda *args, **kwargs: BadUrllib(b"html"),
        ), mock.patch.object(
            module.subprocess, "Popen", return_value=FakeProcess()
        ), mock.patch.object(module.time, "sleep", return_value=None):
            result = module._http_range("https://example.invalid", 50, 59, 1)
        self.assertEqual(result.payload, b"0123456789")
        self.assertEqual(result.headers["content-range"], "bytes 50-59/60")

    def test_metadata_range_is_bounded_and_reassembled(self) -> None:
        payload = bytes(range(50, 60))
        calls: list[tuple[int, int]] = []

        class Response(io.BytesIO):
            status = 206

            def __init__(self, first: int, last: int):
                super().__init__(payload[first - 50 : last - 49])
                self.headers = {
                    "Content-Range": f"bytes {first}-{last}/60",
                    "Content-Length": str(last - first + 1),
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": (
                        'attachment; filename="BSD_3ms24ms.zip"'
                    ),
                    "Last-Modified": "pinned-date",
                }

        def open_url(request, timeout):
            del timeout
            first, last = map(
                int, request.get_header("Range").removeprefix("bytes=").split("-")
            )
            calls.append((first, last))
            return Response(first, last)

        with mock.patch.object(module, "DOWNLOAD_RANGE_BYTES", 4), mock.patch.object(
            module.urllib.request, "urlopen", open_url
        ):
            result = module._http_range("https://example.invalid", 50, 59, 1)
        self.assertEqual(calls, [(50, 53), (54, 57), (58, 59)])
        self.assertEqual(result.payload, payload)
        self.assertEqual(result.headers["content-range"], "bytes 50-59/60")
        self.assertEqual(result.headers["content-length"], "10")

    def test_metadata_range_rejects_identity_change_between_chunks(self) -> None:
        call = 0

        class Response(io.BytesIO):
            status = 206

            def __init__(self, first: int, last: int, modified: str):
                super().__init__(b"x" * (last - first + 1))
                self.headers = {
                    "Content-Range": f"bytes {first}-{last}/60",
                    "Content-Length": str(last - first + 1),
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": (
                        'attachment; filename="BSD_3ms24ms.zip"'
                    ),
                    "Last-Modified": modified,
                }

        def open_url(request, timeout):
            nonlocal call
            del timeout
            call += 1
            first, last = map(
                int, request.get_header("Range").removeprefix("bytes=").split("-")
            )
            return Response(first, last, "first" if call == 1 else "changed")

        with mock.patch.object(module, "DOWNLOAD_RANGE_BYTES", 5), mock.patch.object(
            module.urllib.request, "urlopen", open_url
        ):
            with self.assertRaisesRegex(module.AcquisitionError, "Last-Modified"):
                module._http_range("https://example.invalid", 50, 59, 1)

    def test_acquire_resumes_partial_across_bounded_ranges(self) -> None:
        import copy

        frozen, _ = module.load_protocol()
        protocol = copy.deepcopy(frozen)
        payload = bytes(range(200))
        source = protocol["official_source"]
        source["content_length_bytes"] = len(payload)
        protocol["storage_policy"]["minimum_free_bytes_before_new_download"] = 0
        calls: list[tuple[int, int]] = []

        with tempfile.TemporaryDirectory(dir="/srv/szha0669") as directory:
            quarantine = Path(directory) / "quarantine"
            quarantine.mkdir(mode=0o700)
            partial = quarantine / f'{source["filename"]}.partial'
            partial.write_bytes(payload[:50])

            def open_range(url: str, start: int, end: int, timeout: float):
                del url, timeout
                request_end = min(end, start + 64 - 1)
                calls.append((start, request_end))
                return io.BytesIO(payload[start : request_end + 1])

            archive, receipt = module.acquire_archive(
                protocol,
                "b" * 64,
                quarantine,
                acknowledge_opaque_test_payload=True,
                open_range=open_range,
            )
            self.assertEqual(archive.read_bytes(), payload)
            self.assertTrue(receipt.is_file())
            observed = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(
                observed["archive"]["sha256"], hashlib.sha256(payload).hexdigest()
            )
        self.assertEqual(calls, [(50, 113), (114, 177), (178, 199)])

    def test_quarantine_outside_srv_is_rejected(self) -> None:
        protocol, _ = module.load_protocol()
        with self.assertRaisesRegex(module.AcquisitionError, "under"):
            module._validate_quarantine_root(Path("/home/szha0669/bsd"), protocol)


if __name__ == "__main__":
    unittest.main()
