#!/usr/bin/env python3
"""CPU-only contracts for the official-EVSSM DPDD validation evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_evssm_dpdd_validation import (  # noqa: E402
    DATASET_MANIFEST_SCHEMA,
    DPDD_CONFIG,
    DPDD_REPOSITORY,
    DPDD_REVISION,
    EvaluationError,
    LPIPS_PROTOCOL,
    PAIR_SCHEMA,
    evaluate_rows,
    load_validation_dataset_contract,
    load_validation_manifest,
    read_rgb16_png,
    verify_official_checkpoint,
    verify_lpips_protocol,
    write_report_new,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rgb16(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.empty((9, 11, 3), dtype=np.uint16)
    rgb[:, :, 0] = value
    rgb[:, :, 1] = value + 7
    rgb[:, :, 2] = value + 13
    assert cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1]))


def _write_manifest(root: Path, count: int = 2, *, split: str = "validation") -> Path:
    rows = []
    for index in range(count):
        source = root / "validation" / "defocus" / f"{index:03d}.png"
        target = root / "validation" / "sharp" / f"{index:03d}.png"
        _rgb16(source, 10_000 + index * 100)
        _rgb16(target, 11_000 + index * 100)
        rows.append(
            {
                "schema": PAIR_SCHEMA,
                "name": f"pair_{index}",
                "split": split,
                "defocus": source.relative_to(root).as_posix(),
                "sharp": target.relative_to(root).as_posix(),
                "source_sha256": _sha(source),
                "target_sha256": _sha(target),
            }
        )
    manifest = root / "validation.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def _raises(error_type, function, *args, **kwargs) -> str:
    try:
        function(*args, **kwargs)
    except error_type as error:
        return str(error)
    raise AssertionError(f"expected {error_type.__name__}")


def test_rgb16_manifest_and_raw_evssm_metrics() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _write_manifest(root)
        rows = load_validation_manifest(manifest, data_root=root, expected_count=2)
        assert len(rows) == 2
        decoded = read_rgb16_png(rows[0]["source"])
        assert decoded.dtype == np.float32
        assert decoded.shape == (9, 11, 3)
        assert abs(float(decoded[0, 0, 0]) - 10_000 / 65_535) < 1e-7

        calls = []

        def infer(image: np.ndarray, timestamp: float) -> np.ndarray:
            assert timestamp in {0.0, 1.0}
            calls.append(timestamp)
            return np.clip(image + 1_000 / 65_535, 0.0, 1.0)

        def lpips(prediction: np.ndarray, target: np.ndarray) -> float:
            return float(np.mean(np.abs(prediction - target)))

        result = evaluate_rows(
            rows,
            infer=infer,
            device=torch.device("cpu"),
            lpips=lpips,
        )
        assert result["pair_count"] == 2
        assert set(result["mean"]) == {"raw", "evssm"}
        assert set(result["mean"]["evssm"]) == {"psnr", "ssim", "lpips", "l1"}
        assert result["mean"]["evssm"]["psnr"] > result["mean"]["raw"]["psnr"]
        assert result["latency_ms"]["mean"] >= 0.0
        assert result["warmup"] == {
            "steps": 1,
            "input": "first_validation_defocus_only",
            "target_or_metric_used": False,
            "output_and_latency_discarded": True,
            "pre_and_post_cuda_synchronization": True,
        }
        # Warm-up, dedicated timing-only pass, then an independent untimed
        # quality pass.  Targets and metrics must never contaminate latency.
        assert calls == [0.0, 0.0, 1.0, 0.0, 1.0]
        assert result["pass_separation"] == {
            "timing_pass": {
                "stateless_model_steps": 2,
                "sharp_target_images_opened": False,
                "metrics_or_lpips_computed": False,
            },
            "quality_pass": {
                "stateless_model_steps": 2,
                "timed_model_steps": 0,
            },
            "passes_are_distinct_complete_dataset_traversals": True,
            "forward_accounting_excluding_warmup": {
                "timing_only_model_steps": 2,
                "quality_model_steps": 2,
                "combined_model_steps": 4,
            },
        }


def test_lpips_protocol_rejects_version_and_weight_drift() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        alexnet = root / "alexnet.pth"
        linear = root / "alex.pth"
        alexnet.write_bytes(b"pinned alexnet")
        linear.write_bytes(b"pinned lpips linear")
        protocol = {
            **LPIPS_PROTOCOL,
            "alexnet_backbone": {"path": str(alexnet), "sha256": _sha(alexnet)},
            "lpips_linear_weights": {"path": str(linear), "sha256": _sha(linear)},
        }
        verify_lpips_protocol(protocol, installed_version="1.9.0")
        assert "version changed" in _raises(
            EvaluationError,
            verify_lpips_protocol,
            protocol,
            installed_version="1.9.1",
        )
        tampered = {
            **protocol,
            "lpips_linear_weights": {"path": str(linear), "sha256": "0" * 64},
        }
        assert "artifact changed" in _raises(
            EvaluationError,
            verify_lpips_protocol,
            tampered,
            installed_version="1.9.0",
        )


def test_test_split_is_rejected_before_pixel_access() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "test.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "schema": PAIR_SCHEMA,
                    "name": "sealed",
                    "split": "test",
                    "defocus": "does/not/exist.png",
                    "sharp": "also/missing.png",
                    "source_sha256": "1" * 64,
                    "target_sha256": "2" * 64,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        message = _raises(
            EvaluationError,
            load_validation_manifest,
            manifest,
            data_root=root,
            expected_count=1,
        )
        assert "test split is sealed" in message
        assert "does not exist" not in message


def test_validation_requires_materializer_lineage_not_a_standalone_self_signed_list() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _write_manifest(root, count=2)
        dataset_manifest = root / "dataset_manifest.json"
        disclosure = {
            "metadata_pristine": False,
            "metadata_exposure": (
                "filenames_lfs_oids_sizes_split_aggregate_row0_url_and_manifest_text_"
                "seen_before_freeze"
            ),
            "images_decoded": False,
            "pixels_opened": False,
            "metrics_opened": False,
            "requests_made_by_this_materializer": 0,
            "split_supported_by_this_materializer": False,
        }
        payload = {
            "schema": DATASET_MANIFEST_SCHEMA,
            "repository": DPDD_REPOSITORY,
            "revision": DPDD_REVISION,
            "config": DPDD_CONFIG,
            "splits": {"train": 350, "validation": 74},
            "distribution": {
                "dataset_card_declared_license": "mit",
                "license_scope_warning": "mirror card is not original DPDD rights",
            },
            "test_disclosure": disclosure,
            "canonical_manifests": {
                "validation": {
                    "path": manifest.relative_to(root).as_posix(),
                    "sha256": _sha(manifest),
                    "rows": 74,
                    "schema": PAIR_SCHEMA,
                    "paths_relative_to": "dataset_root",
                }
            },
        }
        dataset_manifest.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        provenance = load_validation_dataset_contract(
            dataset_manifest,
            expected_dataset_manifest_sha256=_sha(dataset_manifest),
            validation_manifest=manifest,
            expected_validation_manifest_sha256=_sha(manifest),
        )
        assert provenance["repository"] == DPDD_REPOSITORY
        assert provenance["test_disclosure"]["metadata_pristine"] is False

        forged = dict(payload)
        forged["repository"] = "arbitrary/self-signed-74"
        dataset_manifest.write_text(
            json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8"
        )
        assert "provenance mismatch" in _raises(
            EvaluationError,
            load_validation_dataset_contract,
            dataset_manifest,
            expected_dataset_manifest_sha256=_sha(dataset_manifest),
            validation_manifest=manifest,
            expected_validation_manifest_sha256=_sha(manifest),
        )


def test_content_tamper_and_uint8_decode_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _write_manifest(root, count=1)
        row = json.loads(manifest.read_text(encoding="utf-8"))
        source = root / row["defocus"]
        _rgb16(source, 22_000)
        assert "SHA-256" in _raises(
            EvaluationError,
            load_validation_manifest,
            manifest,
            data_root=root,
            expected_count=1,
        )

        uint8_path = root / "uint8.png"
        assert cv2.imwrite(str(uint8_path), np.zeros((9, 11, 3), dtype=np.uint8))
        assert "uint16" in _raises(EvaluationError, read_rgb16_png, uint8_path)


def test_output_and_checkpoint_are_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "report"
        report = write_report_new(output, {"schema": "unit.test", "value": 1})
        assert report.is_file()
        assert "overwrite" in _raises(
            EvaluationError,
            write_report_new,
            output,
            {"schema": "unit.test", "value": 2},
        )
        fake_checkpoint = root / "fake.pth"
        fake_checkpoint.write_bytes(b"not the official EVSSM")
        assert "pinned official" in _raises(
            EvaluationError, verify_official_checkpoint, fake_checkpoint
        )


def main() -> None:
    test_rgb16_manifest_and_raw_evssm_metrics()
    test_lpips_protocol_rejects_version_and_weight_drift()
    test_test_split_is_rejected_before_pixel_access()
    test_validation_requires_materializer_lineage_not_a_standalone_self_signed_list()
    test_content_tamper_and_uint8_decode_fail_closed()
    test_output_and_checkpoint_are_fail_closed()
    print("EVSSM DPDD validation CPU contracts: PASS")


if __name__ == "__main__":
    main()
