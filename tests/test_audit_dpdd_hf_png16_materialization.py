"""CPU-only offline tests for the DPDD materialization auditor."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.audit_dpdd_hf_png16_materialization as audit  # noqa: E402


def _write_png(path: Path, value: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((5, 7, 3), value, dtype=np.uint16)
    assert cv2.imwrite(str(path), image)
    return audit.sha256_file(path)


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fixture(root: Path) -> None:
    canonical_descriptors = {}
    nested = []
    for split, base in (("train", 100), ("validation", 200)):
        source = root / split / "source/000000.png"
        target = root / split / "target/000000.png"
        source_sha = _write_png(source, base + 1)
        target_sha = _write_png(target, base + 2)
        record = {
            "schema": audit.CANONICAL_PAIR_SCHEMA,
            "name": f"dpdd_{split}_000000",
            "split": split,
            "defocus": source.relative_to(root).as_posix(),
            "sharp": target.relative_to(root).as_posix(),
            "source_sha256": source_sha,
            "target_sha256": target_sha,
        }
        manifest_path = root / "manifests" / f"{split}.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(_canonical(record))
        canonical_descriptors[split] = {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": audit.sha256_file(manifest_path),
            "rows": 1,
            "schema": audit.CANONICAL_PAIR_SCHEMA,
            "paths_relative_to": "dataset_root",
        }
        nested.append({"split": split})
    pairs = root / "pairs.jsonl"
    pairs.write_bytes(b"".join(_canonical(row) for row in nested))
    dataset_manifest = {
        "schema": audit.MATERIALIZATION_SCHEMA,
        "repository": audit.REPOSITORY,
        "revision": audit.REVISION,
        "config": audit.CONFIG,
        "splits": {"train": 1, "validation": 1},
        "pair_count": 2,
        "asset_count": 4,
        "canonical_manifests": canonical_descriptors,
        "pairs_jsonl": {"path": "pairs.jsonl", "sha256": audit.sha256_file(pairs)},
        "test_disclosure": {
            "requests_made_by_this_materializer": 0,
            "split_supported_by_this_materializer": False,
            "images_decoded": False,
            "pixels_opened": False,
            "metrics_opened": False,
        },
    }
    (root / "dataset_manifest.json").write_bytes(_canonical(dataset_manifest))


def test_offline_audit_and_atomic_output() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "dataset"
        root.mkdir()
        _fixture(root)
        report = audit.audit_materialization(
            root,
            expected_splits={"train": 1, "validation": 1},
            expected_size=(7, 5),
        )
        assert report["status"] == "pass"
        assert report["asset_count"] == 4
        assert report["image_contract"]["unique_sizes"] == [[7, 5]]
        assert report["test_audit"]["network_requests_by_auditor"] == 0
        output = Path(directory) / "audit.json"
        audit._write_atomic_new(output, audit._canonical_bytes(report))
        assert output.is_file()
        try:
            audit._write_atomic_new(output, b"replacement")
        except audit.AuditError as error:
            assert "overwrite" in str(error)
        else:
            raise AssertionError("audit output overwrite was accepted")


def test_content_mutation_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _fixture(root)
        _write_png(root / "train/source/000000.png", 999)
        try:
            audit.audit_materialization(
                root,
                expected_splits={"train": 1, "validation": 1},
                expected_size=(7, 5),
            )
        except audit.AuditError as error:
            assert "SHA mismatch" in str(error)
        else:
            raise AssertionError("mutated asset was accepted")


if __name__ == "__main__":
    test_offline_audit_and_atomic_output()
    test_content_mutation_fails_closed()
    print("DPDD PNG16 materialization audit CPU tests: PASS")
