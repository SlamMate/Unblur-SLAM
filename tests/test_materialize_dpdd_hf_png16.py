"""CPU-only fake-network tests for the pinned DPDD PNG16 materializer."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.materialize_dpdd_hf_png16 as module  # noqa: E402
from scripts.evaluate_turtle_single_image_defocus import (  # noqa: E402
    load_single_image_manifest,
)


def _png(value: int, width: int = 7, height: int = 5) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint16)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _fixture_specs():
    metadata = {
        "train": (
            b'{"source_file_name":"../../../dd_dp_dataset_png/train_c/source/a.png",'
            b'"target_file_name":"../../../dd_dp_dataset_png/train_c/target/b.png"}\n'
        ),
        "validation": (
            b'{"source_file_name":"../../../dd_dp_dataset_png/val_c/source/c.png",'
            b'"target_file_name":"../../../dd_dp_dataset_png/val_c/target/d.png"}\n'
        ),
    }
    specs = {
        "train": module.SplitSpec(
            "train",
            "train",
            "config/combined/train/metadata.jsonl",
            "dd_dp_dataset_png/train_c",
            1,
            len(metadata["train"]),
            module._sha256(metadata["train"]),
        ),
        "validation": module.SplitSpec(
            "validation",
            "val",
            "config/combined/val/metadata.jsonl",
            "dd_dp_dataset_png/val_c",
            1,
            len(metadata["validation"]),
            module._sha256(metadata["validation"]),
        ),
    }
    return metadata, specs


class DpddPng16Tests(unittest.TestCase):
    def _fetcher(self, metadata, calls):
        def fetch(url: str, timeout: float) -> bytes:
            del timeout
            calls.append(url)
            if url.endswith("/README.md"):
                # Production constants are intentionally patched only inside this test.
                return b"---\nlicense: mit\n---\n"
            if "/train/metadata.jsonl" in url:
                return metadata["train"]
            if "/val/metadata.jsonl" in url:
                return metadata["validation"]
            raise AssertionError(f"unexpected request: {url}")

        return fetch

    def _patched_card(self):
        class Patch:
            def __enter__(inner):
                inner.old = (module.README_BYTES, module.README_SHA256)
                payload = b"---\nlicense: mit\n---\n"
                module.README_BYTES = len(payload)
                module.README_SHA256 = module._sha256(payload)

            def __exit__(inner, *_):
                module.README_BYTES, module.README_SHA256 = inner.old

        return Patch()

    def test_preflight_requests_only_pinned_card_train_val_metadata(self):
        metadata, specs = _fixture_specs()
        calls: list[str] = []
        with self._patched_card():
            report, pairs, _ = module.prepare_preflight(
                ["train", "val"],
                fetch=self._fetcher(metadata, calls),
                split_specs=specs,
            )
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(module.REVISION in url for url in calls))
        self.assertFalse(any("/test/" in url for url in calls))
        self.assertEqual(len(pairs), 2)
        self.assertEqual(report["asset_requests_made"], 0)
        self.assertFalse(report["test_disclosure"]["metadata_pristine"])
        self.assertFalse(report["distribution"]["official_dpdd_download"])

    def test_materializes_png16_atomically_and_audits_disjointness(self):
        metadata, specs = _fixture_specs()
        calls: list[str] = []
        assets = {
            "/source/a.png": _png(101),
            "/target/b.png": _png(102),
            "/source/c.png": _png(103),
            "/target/d.png": _png(104),
        }

        def open_asset(url: str, timeout: float):
            del timeout
            for suffix, payload in assets.items():
                if url.endswith(suffix):
                    return io.BytesIO(payload)
            raise AssertionError(url)

        with tempfile.TemporaryDirectory() as directory, self._patched_card():
            output = Path(directory) / "dpdd"
            manifest_path = module.materialize(
                output,
                fetch=self._fetcher(metadata, calls),
                open_asset=open_asset,
                split_specs=specs,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["pair_count"], 2)
            self.assertEqual(manifest["workers"], 8)
            self.assertTrue(manifest["content_hash_audit"]["globally_unique"])
            self.assertEqual(manifest["image_contract"]["decoded_dtype"], "uint16")
            self.assertTrue((output / "train/source/000000.png").is_file())
            pair = json.loads(
                (output / "pairs.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(pair["source"]["png_ihdr_bit_depth"], 16)
            self.assertEqual(pair["source"]["decoded_dtype"], "uint16")
            canonical = manifest["canonical_manifests"]["train"]
            canonical_path = output / canonical["path"]
            self.assertEqual(module.sha256_file(canonical_path), canonical["sha256"])
            loaded = load_single_image_manifest(
                canonical_path,
                root=output,
                expected_split="train",
                canonical_contract=True,
                verify_content=True,
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].name, "dpdd_train_000000")
            self.assertEqual(loaded[0].blurry, output / "train/source/000000.png")

            before = len(calls)
            with self.assertRaisesRegex(module.MaterializationError, "overwrite"):
                module.materialize(
                    output,
                    fetch=self._fetcher(metadata, calls),
                    open_asset=open_asset,
                    split_specs=specs,
                )
            self.assertEqual(len(calls), before)

    def test_duplicate_source_target_content_fails_and_cleans_stage(self):
        metadata, specs = _fixture_specs()
        calls: list[str] = []
        same = _png(999)

        with tempfile.TemporaryDirectory() as directory, self._patched_card():
            root = Path(directory)
            output = root / "dpdd"
            with self.assertRaisesRegex(
                module.MaterializationError, "identical|duplicate"
            ):
                module.materialize(
                    output,
                    fetch=self._fetcher(metadata, calls),
                    open_asset=lambda url, timeout: io.BytesIO(same),
                    split_specs=specs,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".dpdd.staging-*")), [])

    def test_test_split_is_rejected_before_network(self):
        calls: list[str] = []
        with self.assertRaisesRegex(module.MaterializationError, "test is sealed"):
            module.prepare_preflight(
                ["test"], fetch=lambda url, timeout: calls.append(url) or b""
            )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
