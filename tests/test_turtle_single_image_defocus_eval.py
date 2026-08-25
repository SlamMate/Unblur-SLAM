#!/usr/bin/env python3
"""CPU-only contracts for paired single-image TURTLE evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_turtle_single_image_defocus import (  # noqa: E402
    CANONICAL_PAIR_SCHEMA,
    FORMAL_VIDEO_TRAIN_MANIFEST_SHA256,
    MIXED_TRAINING_SCHEMA,
    evaluate_arm,
    load_dpdd_evaluation_dataset_contract,
    load_single_image_manifest,
    main as evaluator_main,
    pad_to_multiple,
    paired_arm_delta,
    read_pair_tensor,
    summarize_rows,
    validate_distinct_formal_checkpoint_hashes,
    validate_formal_arm_metadata,
)
from src.turtle_backend import (  # noqa: E402
    FINETUNED_CHECKPOINT_FORMAT,
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    TURTLE_CACHE_CONTRACT,
    sha256_file,
)


def _image(path: Path, value: int, *, size=(11, 9)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


class _FakeIndependentBackend:
    def __init__(self):
        self.has_cache = False
        self.reset_count = 0
        self.calls = []

    def reset(self):
        self.has_cache = False
        self.reset_count += 1

    def state_info(self):
        return {"has_cache": self.has_cache}

    def step(self, image, timestamp=None):
        self.calls.append(
            {
                "had_cache": self.has_cache,
                "timestamp": timestamp,
                "shape": tuple(image.shape),
            }
        )
        self.has_cache = True
        return (image + 1.0 / 255.0).clamp(0.0, 1.0)


def test_manifest_requires_one_pixel_aligned_pair_per_record() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _image(root / "blur/a.jpg", 20)
        _image(root / "sharp/a.jpg", 25)
        manifest = root / "pairs.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "name": "a",
                    "defocus": "blur/a.jpg",
                    "sharp": "sharp/a.jpg",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        records = load_single_image_manifest(manifest, root=root)
        assert len(records) == 1
        assert records[0].name == "a"

        bad = root / "bad.jsonl"
        bad.write_text(
            json.dumps(
                {
                    "name": "sequence_not_single",
                    "blurry": ["blur/a.jpg", "blur/a.jpg"],
                    "sharp": ["sharp/a.jpg", "sharp/a.jpg"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            load_single_image_manifest(bad, root=root)
        except ValueError as error:
            assert "exactly one" in str(error)
        else:
            raise AssertionError("temporal record was accepted as one image")


def test_padding_is_deterministic_right_bottom_and_cropped_by_caller() -> None:
    source = torch.arange(3 * 9 * 11, dtype=torch.float32).reshape(1, 3, 9, 11)
    padded, height, width, pad_height, pad_width = pad_to_multiple(source, 8)
    assert (height, width) == (9, 11)
    assert (pad_height, pad_width) == (7, 5)
    assert tuple(padded.shape) == (1, 3, 16, 16)
    assert torch.equal(padded[:, :, :height, :width], source)


def test_every_image_is_evaluated_from_empty_cache() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        lines = []
        for index in range(2):
            _image(root / f"blur/{index}.png", 20 + index)
            _image(root / f"sharp/{index}.png", 25 + index)
            lines.append(
                json.dumps(
                    {
                        "name": f"pair_{index}",
                        "defocus": f"blur/{index}.png",
                        "sharp": f"sharp/{index}.png",
                    }
                )
            )
        manifest = root / "pairs.jsonl"
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        records = load_single_image_manifest(manifest, root=root)
        backend = _FakeIndependentBackend()
        rows = evaluate_arm(
            records,
            backend,
            device=torch.device("cpu"),
            lpips_metric=None,
            padding_multiple=8,
            warmup_steps=1,
        )
        assert len(rows) == 2
        assert all(row["cache_empty_before_call"] for row in rows)
        assert all(not call["had_cache"] for call in backend.calls)
        assert all(call["timestamp"] == 0 for call in backend.calls)
        # One independent warmup plus two independent evaluated pairs.
        assert len(backend.calls) == 3
        assert backend.reset_count >= 3
        summary = summarize_rows(rows)
        assert summary["image_count"] == 2
        assert set(summary["mean"]) == {"psnr", "ssim", "l1"}


def test_paired_delta_is_explicitly_spatial_not_history() -> None:
    reference = [
        {
            "name": "a",
            "metrics": {"psnr": 30.0, "ssim": 0.9, "l1": 0.02},
            "latency_ms": 1.0,
        }
    ]
    candidate = [
        {
            "name": "a",
            "metrics": {"psnr": 30.2, "ssim": 0.91, "l1": 0.019},
            "latency_ms": 1.1,
        }
    ]
    delta = paired_arm_delta(candidate, reference)
    assert "not a history benefit" in delta["interpretation"]
    assert abs(delta["mean_candidate_minus_reference"]["psnr"] - 0.2) < 1e-12


def test_canonical_dpdd_rgb16_is_content_verified_without_rgb8_quantization() -> None:
    import cv2

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "train/source/a.png"
        target = root / "train/target/a.png"
        source.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True)
        # OpenCV writes BGR; channel reversal in read_pair_tensor must preserve
        # all 16 bits before division by 65535.
        source_bgr = np.zeros((8, 8, 3), dtype=np.uint16)
        source_bgr[..., 0] = 12345
        source_bgr[..., 1] = 23456
        source_bgr[..., 2] = 34567
        target_bgr = source_bgr + 100
        assert cv2.imwrite(str(source), source_bgr)
        assert cv2.imwrite(str(target), target_bgr)
        manifest = root / "pairs.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "schema": CANONICAL_PAIR_SCHEMA,
                    "name": "a",
                    "split": "train",
                    "defocus": str(source.relative_to(root)),
                    "sharp": str(target.relative_to(root)),
                    "source_sha256": sha256_file(source),
                    "target_sha256": sha256_file(target),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        records = load_single_image_manifest(
            manifest,
            root=root,
            expected_split="train",
            canonical_contract=True,
            verify_content=True,
        )
        tensor = read_pair_tensor(
            records[0].blurry,
            device=torch.device("cpu"),
            require_png_rgb16=True,
        )
        assert abs(float(tensor[0, 0, 0]) - 34567.0 / 65535.0) < 1e-7
        assert abs(float(tensor[2, 0, 0]) - 12345.0 / 65535.0) < 1e-7


def _formal_metadata(arm: str, checkpoint_hash: str, *, seed: int = 17):
    video_hash = FORMAL_VIDEO_TRAIN_MANIFEST_SHA256
    dpdd_hash = "d" * 64
    dataset_hash = "e" * 64
    return {
        "format": FINETUNED_CHECKPOINT_FORMAT,
        "kind": "finetuned",
        "checkpoint_sha256": checkpoint_hash,
        "schema": MIXED_TRAINING_SCHEMA,
        "base_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
        "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
        "input_domain": "raw",
        "cache_contract": TURTLE_CACHE_CONTRACT,
        "mode": arm,
        "uses_paired_sharp_ground_truth_rgb": True,
        "uses_gt_pose": False,
        "uses_gt_depth": False,
        "training": {
            "seed": seed,
            "optimizer_steps": 78,
            "attempted_optimizer_steps": 78,
            "executed_optimizer_steps": 78,
            "amp_skipped_optimizer_steps": 0,
            "amp": True,
            "mixed_step": "two_backward_one_joint_step" if arm == "M" else None,
            "grad_scaler": {
                "init_scale": 1024.0,
                "growth_interval": 2000,
                "growth_disabled_within_78_steps": True,
                "overflow_policy": "fail_closed_no_checkpoint",
            },
        },
        "manifests": {
            "video_sha256": video_hash if arm in {"V", "M"} else None,
            "dpdd_pairs_sha256": dpdd_hash if arm in {"S", "M"} else None,
            "dpdd_selected_split": "train" if arm in {"S", "M"} else None,
            "dpdd_dataset": (
                {
                    "sha256": dataset_hash,
                    "schema": "unblur_slam.dpdd_hf_png16_materialization.v1",
                    "repository": "JacobLinCool/DPDD",
                    "revision": "52e4035a045ea1763313b9ce2b27cf2e620cfc30",
                    "config": "combined",
                    "dataset_card_declared_license": "mit",
                    "license_scope_warning": "mirror claim is not original rights",
                    "test_metadata_pristine": False,
                    "test_pixels_opened": False,
                    "test_metrics_opened": False,
                    "canonical_train_manifest_sha256": dpdd_hash,
                }
                if arm in {"S", "M"}
                else None
            ),
            "test_pixels_or_metrics_read": False,
        },
    }


def test_formal_checkpoint_identity_rejects_mislabeled_seed_and_duplicates() -> None:
    hashes = {"G": "1" * 64, "V": "2" * 64, "S": "3" * 64, "M": "4" * 64}
    validate_distinct_formal_checkpoint_hashes(hashes)
    duplicate_hashes = dict(hashes)
    duplicate_hashes["M"] = duplicate_hashes["V"]
    try:
        validate_distinct_formal_checkpoint_hashes(duplicate_hashes)
    except ValueError as error:
        assert "distinct" in str(error)
    else:
        raise AssertionError("duplicate checkpoint bytes passed as distinct formal arms")

    for arm in ("V", "S", "M"):
        metadata = _formal_metadata(arm, hashes[arm])
        validate_formal_arm_metadata(
            arm,
            hashes[arm],
            metadata,
            expected_seed=17,
            expected_video_manifest_sha256=FORMAL_VIDEO_TRAIN_MANIFEST_SHA256,
            expected_dpdd_train_sha256="d" * 64,
            expected_dpdd_dataset_manifest_sha256="e" * 64,
        )

    official_metadata = {
        "kind": "official_gopro",
        "checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        "base_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_arch_sha256": PINNED_TURTLE_ARCH_SHA256,
        "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
        "input_domain": "raw",
        "cache_contract": TURTLE_CACHE_CONTRACT,
    }
    validate_formal_arm_metadata(
        "G",
        PINNED_TURTLE_CHECKPOINT_SHA256,
        official_metadata,
        expected_seed=17,
        expected_video_manifest_sha256=FORMAL_VIDEO_TRAIN_MANIFEST_SHA256,
        expected_dpdd_train_sha256="d" * 64,
        expected_dpdd_dataset_manifest_sha256="e" * 64,
    )

    mislabeled = _formal_metadata("M", hashes["V"])
    try:
        validate_formal_arm_metadata(
            "V",
            hashes["V"],
            mislabeled,
            expected_seed=17,
            expected_video_manifest_sha256=FORMAL_VIDEO_TRAIN_MANIFEST_SHA256,
            expected_dpdd_train_sha256="d" * 64,
            expected_dpdd_dataset_manifest_sha256="e" * 64,
        )
    except ValueError as error:
        assert "mode=" in str(error)
    else:
        raise AssertionError("an M checkpoint was accepted under the V label")

    wrong_seed = _formal_metadata("S", hashes["S"], seed=42)
    try:
        validate_formal_arm_metadata(
            "S",
            hashes["S"],
            wrong_seed,
            expected_seed=17,
            expected_video_manifest_sha256=FORMAL_VIDEO_TRAIN_MANIFEST_SHA256,
            expected_dpdd_train_sha256="d" * 64,
            expected_dpdd_dataset_manifest_sha256="e" * 64,
        )
    except ValueError as error:
        assert "seed" in str(error)
    else:
        raise AssertionError("a checkpoint from another seed was accepted")

    skipped = copy.deepcopy(_formal_metadata("M", hashes["M"]))
    skipped["training"]["executed_optimizer_steps"] = 77
    skipped["training"]["amp_skipped_optimizer_steps"] = 1
    try:
        validate_formal_arm_metadata(
            "M",
            hashes["M"],
            skipped,
            expected_seed=17,
            expected_video_manifest_sha256=FORMAL_VIDEO_TRAIN_MANIFEST_SHA256,
            expected_dpdd_train_sha256="d" * 64,
            expected_dpdd_dataset_manifest_sha256="e" * 64,
        )
    except ValueError as error:
        assert "execution budget" in str(error)
    else:
        raise AssertionError("an AMP-skipped checkpoint passed formal provenance")


def test_test_split_is_rejected_before_any_manifest_or_output_access() -> None:
    original_argv = list(sys.argv)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "must_not_exist"
        sys.argv = [
            "evaluate_turtle_single_image_defocus.py",
            "--manifest",
            str(root / "missing_test_manifest.jsonl"),
            "--split",
            "test",
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--skip-lpips",
        ]
        try:
            evaluator_main()
        except ValueError as error:
            assert "remain sealed" in str(error)
        else:
            raise AssertionError("non-formal CLI could access the sealed test split")
        finally:
            sys.argv = original_argv
        assert not output.exists()


def test_manifest_embedded_test_split_is_rejected_before_path_resolution() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "test.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "name": "sealed",
                    "split": "test",
                    "defocus": "missing/source.png",
                    "sharp": "missing/target.png",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            load_single_image_manifest(manifest, root=root)
        except ValueError as error:
            assert "remain sealed" in str(error)
        else:
            raise AssertionError("manifest-embedded test split reached pixel paths")


def test_formal_dataset_manifest_three_way_binds_validation_jsonl() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        validation = root / "manifests/validation.jsonl"
        validation.parent.mkdir(parents=True)
        validation.write_text('{"fixture":"validation"}\n', encoding="utf-8")
        dataset_manifest = root / "dataset_manifest.json"
        dataset_manifest.write_text(
            json.dumps(
                {
                    "schema": "unblur_slam.dpdd_hf_png16_materialization.v1",
                    "repository": "JacobLinCool/DPDD",
                    "revision": "52e4035a045ea1763313b9ce2b27cf2e620cfc30",
                    "config": "combined",
                    "splits": {"train": 350, "validation": 74},
                    "canonical_manifests": {
                        "validation": {
                            "path": "manifests/validation.jsonl",
                            "sha256": sha256_file(validation),
                            "rows": 74,
                            "schema": CANONICAL_PAIR_SCHEMA,
                            "paths_relative_to": "dataset_root",
                        }
                    },
                    "distribution": {
                        "dataset_card_declared_license": "mit",
                        "license_scope_warning": "mirror claim is not original rights",
                    },
                    "test_disclosure": {
                        "metadata_pristine": False,
                        "images_decoded": False,
                        "pixels_opened": False,
                        "metrics_opened": False,
                        "split_supported_by_this_materializer": False,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        provenance = load_dpdd_evaluation_dataset_contract(
            dataset_manifest,
            expected_dataset_manifest_sha256=sha256_file(dataset_manifest),
            validation_manifest=validation,
            expected_validation_manifest_sha256=sha256_file(validation),
        )
        assert provenance["repository"] == "JacobLinCool/DPDD"
        assert provenance["test_disclosure"]["metadata_pristine"] is False
        assert provenance["canonical_validation_manifest"]["rows"] == 74


if __name__ == "__main__":
    test_manifest_requires_one_pixel_aligned_pair_per_record()
    test_padding_is_deterministic_right_bottom_and_cropped_by_caller()
    test_every_image_is_evaluated_from_empty_cache()
    test_paired_delta_is_explicitly_spatial_not_history()
    test_canonical_dpdd_rgb16_is_content_verified_without_rgb8_quantization()
    test_formal_checkpoint_identity_rejects_mislabeled_seed_and_duplicates()
    test_test_split_is_rejected_before_any_manifest_or_output_access()
    test_manifest_embedded_test_split_is_rejected_before_path_resolution()
    test_formal_dataset_manifest_three_way_binds_validation_jsonl()
    print("9 single-image defocus CPU contracts passed")
