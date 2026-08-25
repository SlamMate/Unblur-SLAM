"""Standard-library, CPU-only contracts for the DPDD HF materializer.

All HTTP and asset reads are in-memory fakes.  These tests never contact
Hugging Face and never open the sealed DPDD test split.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from scripts.materialize_dpdd_hf import (
    CONFIG,
    EXPECTED_ROWS,
    REPOSITORY,
    TEST_CONTRACT_SCHEMA,
    MaterializationError,
    materialize,
    parse_args,
    prepare_preflight,
)


def _jpeg(width: int = 7, height: int = 5, marker_payload: bytes = b"") -> bytes:
    # A minimal SOF0-bearing stream is enough for the materializer's strict
    # header parser; no image codec is invoked in these CPU contract tests.
    sof_payload = (
        b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    )
    return (
        b"\xff\xd8"
        + b"\xff\xe0"
        + struct.pack(">H", len(marker_payload) + 2)
        + marker_payload
        + b"\xff\xc0"
        + struct.pack(">H", len(sof_payload) + 2)
        + sof_payload
        + b"\xff\xd9"
    )


def _metadata_fetcher(counts: dict[str, int], calls: list[str]):
    """Return a datasets-server fake keyed by its real server split names."""

    def fetch(url: str, timeout: float):
        del timeout
        calls.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        assert query["dataset"] == [REPOSITORY]
        assert query["config"] == [CONFIG]
        split = query["split"][0]
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        total = counts[split]
        rows = []
        for row_idx in range(offset, min(offset + length, total)):
            rows.append(
                {
                    "row_idx": row_idx,
                    "row": {
                        "source": {
                            "src": (
                                "https://datasets-server.huggingface.co/assets/"
                                f"dpdd/{split}/source/{row_idx}.jpg?token=do-not-record"
                            ),
                            "width": 7,
                            "height": 5,
                        },
                        "target": {
                            "src": (
                                "https://datasets-server.huggingface.co/assets/"
                                f"dpdd/{split}/target/{row_idx}.jpg?token=do-not-record"
                            ),
                            "width": 7,
                            "height": 5,
                        },
                    },
                    "truncated_cells": [],
                }
            )
        return {"num_rows_total": total, "partial": False, "rows": rows}

    return fetch


class DpddHfMaterializerTests(unittest.TestCase):
    def test_train_validation_preflight_uses_server_val_and_opens_no_assets(self):
        calls: list[str] = []
        report, pairs = prepare_preflight(
            ["train", "val"],
            fetch_json=_metadata_fetcher({"train": 3, "val": 2}, calls),
            expected_rows={"train": 3, "validation": 2, "test": 1},
            page_size=2,
        )

        queried_splits = [
            urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["split"][0]
            for url in calls
        ]
        self.assertIn("train", queried_splits)
        self.assertIn("val", queried_splits)
        self.assertNotIn("validation", queried_splits)
        self.assertNotIn("test", queried_splits)
        self.assertEqual(report["splits"], {"train": 3, "validation": 2})
        self.assertEqual(
            report["server_splits"], {"train": "train", "validation": "val"}
        )
        self.assertFalse(report["image_urls_opened"])
        self.assertEqual(report["pixel_bytes_downloaded"], 0)
        self.assertEqual(
            report["distribution"]["observed_encoding"],
            "unknown_preflight_no_pixel_bytes_opened",
        )
        self.assertEqual(pairs["validation"][0].server_split, "val")
        provenance = pairs["validation"][0].source.provenance
        self.assertNotIn("?", provenance["url_no_query"])
        self.assertNotIn("token", json.dumps(provenance, sort_keys=True))

    def test_test_is_rejected_before_any_metadata_request(self):
        calls: list[str] = []
        with self.assertRaisesRegex(MaterializationError, "test is sealed"):
            prepare_preflight(
                ["test"],
                fetch_json=_metadata_fetcher({"test": EXPECTED_ROWS["test"]}, calls),
            )
        self.assertEqual(calls, [])

    def test_partial_rows_cache_is_rejected_as_incomplete(self):
        calls: list[str] = []

        def partial_fetch(url: str, timeout: float):
            del timeout
            calls.append(url)
            return {"num_rows_total": 321, "partial": True, "rows": []}

        with self.assertRaisesRegex(MaterializationError, "response is partial"):
            prepare_preflight(["train"], fetch_json=partial_fetch)
        self.assertEqual(len(calls), 1)
        queried = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0]).query)
        self.assertEqual(queried["split"], ["train"])

    def test_test_requires_exact_frozen_contract_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "schema": TEST_CONTRACT_SCHEMA,
                        "status": "frozen",
                        "repository": REPOSITORY,
                        "config": CONFIG,
                        "split": "test",
                        "expected_rows": EXPECTED_ROWS["test"],
                        "allow_test_pixels": True,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            digest = hashlib.sha256(contract.read_bytes()).hexdigest()
            calls: list[str] = []
            with self.assertRaisesRegex(MaterializationError, "SHA mismatch"):
                prepare_preflight(
                    ["test"],
                    allow_test_after_contract=True,
                    frozen_contract=contract,
                    frozen_contract_sha256="0" * 64,
                    fetch_json=_metadata_fetcher(
                        {"test": EXPECTED_ROWS["test"]}, calls
                    ),
                )
            self.assertEqual(calls, [])

            report, _ = prepare_preflight(
                ["test"],
                allow_test_after_contract=True,
                frozen_contract=contract,
                frozen_contract_sha256=digest,
                fetch_json=_metadata_fetcher({"test": EXPECTED_ROWS["test"]}, calls),
            )
            self.assertEqual(report["test_access_contract"]["sha256"], digest)
            self.assertEqual(report["server_splits"], {"test": "test"})

    def test_materialization_is_atomic_query_free_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "dpdd"
            calls: list[str] = []
            opened: list[str] = []

            def open_asset(url: str, timeout: float):
                del timeout
                opened.append(url)
                role = b"target" if "/target/" in url else b"source"
                return io.BytesIO(_jpeg(marker_payload=role))

            manifest_path = materialize(
                output,
                ["train"],
                fetch_json=_metadata_fetcher({"train": 1}, calls),
                open_asset=open_asset,
                expected_rows={"train": 1, "validation": 1, "test": 1},
            )
            self.assertEqual(manifest_path, output / "dataset_manifest.json")
            self.assertTrue((output / "train/source/000000.jpg").is_file())
            self.assertTrue((output / "train/target/000000.jpg").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
            self.assertEqual(len(opened), 2)
            self.assertTrue(all("token=do-not-record" in url for url in opened))

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["splits"], {"train": 1})
            self.assertEqual(
                manifest["distribution"]["observed_encoding"],
                "JPEG_verified_from_every_downloaded_asset",
            )
            pair = json.loads((output / "pairs.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(pair["server_split"], "train")
            self.assertEqual(pair["source"]["pixel_size"], {"width": 7, "height": 5})
            self.assertRegex(pair["source"]["sha256"], r"^[0-9a-f]{64}$")
            persisted_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (
                    output / "dataset_manifest.json",
                    output / "pairs.jsonl",
                    output / "SHA256SUMS",
                )
            )
            self.assertNotIn("?", persisted_text)
            self.assertNotIn("token", persisted_text)

            calls_before = list(calls)
            with self.assertRaisesRegex(MaterializationError, "refusing to overwrite"):
                materialize(
                    output,
                    ["train"],
                    fetch_json=_metadata_fetcher({"train": 1}, calls),
                    open_asset=open_asset,
                    expected_rows={"train": 1, "validation": 1, "test": 1},
                )
            self.assertEqual(calls, calls_before)

    def test_failed_asset_read_leaves_no_output_or_staging_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "dpdd"
            calls: list[str] = []

            def fail_asset(url: str, timeout: float):
                del url, timeout
                raise MaterializationError("synthetic asset failure")

            with self.assertRaisesRegex(
                MaterializationError, "synthetic asset failure"
            ):
                materialize(
                    output,
                    ["validation"],
                    fetch_json=_metadata_fetcher({"val": 1}, calls),
                    open_asset=fail_asset,
                    expected_rows={"train": 1, "validation": 1, "test": 1},
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".dpdd.staging-*")), [])

    def test_preflight_cli_does_not_require_output(self):
        args = parse_args(["--preflight-only", "--splits", "train", "val"])
        self.assertIsNone(args.output)
        self.assertTrue(args.preflight_only)


if __name__ == "__main__":
    unittest.main()
