"""CPU-only contracts for formal FrameCrafter -> EVSSM post-processing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "postprocess_framecrafter_evssm.py"
SPEC = importlib.util.spec_from_file_location(
    "postprocess_framecrafter_evssm", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from src.framecrafter_pipeline import (  # noqa: E402
    source_input_digest,
    synthetic_output_digest,
    validate_manifest_payload,
)


PostprocessConfig = MODULE.PostprocessConfig
postprocess = MODULE.postprocess
sha256_file = MODULE.sha256_file


def _rgb(path: Path, image: np.ndarray) -> None:
    encoded = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(path)


def _depth(path: Path, value_metres: float = 1.0) -> None:
    Image.fromarray(
        np.full((64, 64), round(value_metres * 5000.0), dtype=np.uint16)
    ).save(path)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, np.ndarray]:
    height = width = 64
    yy, xx = np.mgrid[:height, :width]
    checker = ((xx // 8 + yy // 8) % 2).astype(np.float32)
    pre = np.repeat((0.42 + 0.16 * checker)[..., None], 3, axis=2)
    support_left = pre.copy()
    support_right = pre.copy()
    left_rgb, right_rgb, synthetic_rgb = (
        tmp_path / "left.png",
        tmp_path / "right.png",
        tmp_path / "synthetic.png",
    )
    left_depth, right_depth, synthetic_depth = (
        tmp_path / "left_depth.png",
        tmp_path / "right_depth.png",
        tmp_path / "synthetic_depth.png",
    )
    _rgb(left_rgb, support_left)
    _rgb(right_rgb, support_right)
    _rgb(synthetic_rgb, pre)
    for path in (left_depth, right_depth, synthetic_depth):
        _depth(path)
    checkpoint = tmp_path / "official-shaped-test-checkpoint.pth"
    checkpoint.write_bytes(b"fake checkpoint bytes; inference is injected")

    left_pose = np.eye(4, dtype=np.float64)
    right_pose = np.eye(4, dtype=np.float64)
    right_pose[0, 3] = 0.01
    target_pose = np.eye(4, dtype=np.float64)
    target_pose[0, 3] = 0.005
    k = np.array([[60.0, 0.0, 31.5], [0.0, 60.0, 31.5], [0.0, 0.0, 1.0]])
    originals = [
        {
            "kind": "original",
            "source_index": 0,
            "rgb_path": str(left_rgb),
            "depth_path": str(left_depth),
            "rgb_sha256": sha256_file(left_rgb),
            "depth_sha256": sha256_file(left_depth),
            "c2w": left_pose.tolist(),
            "intrinsics": k.tolist(),
            "confidence": 1.0,
            "eval": True,
            "fixed_pose": False,
            "reasons": [],
            "left_index": None,
            "right_index": None,
            "alpha": None,
            "timestamp": 0.0,
        },
        {
            "kind": "original",
            "source_index": 1,
            "rgb_path": str(right_rgb),
            "depth_path": str(right_depth),
            "rgb_sha256": sha256_file(right_rgb),
            "depth_sha256": sha256_file(right_depth),
            "c2w": right_pose.tolist(),
            "intrinsics": k.tolist(),
            "confidence": 1.0,
            "eval": True,
            "fixed_pose": False,
            "reasons": [],
            "left_index": None,
            "right_index": None,
            "alpha": None,
            "timestamp": 1.0,
        },
    ]
    synthetic = {
        "kind": "synthetic",
        "target_id": "target_000",
        "source_index": None,
        "rgb_path": str(synthetic_rgb),
        "depth_path": str(synthetic_depth),
        "rgb_sha256": sha256_file(synthetic_rgb),
        "depth_sha256": sha256_file(synthetic_depth),
        "c2w": target_pose.tolist(),
        "intrinsics": k.tolist(),
        "confidence": 0.8,
        "eval": False,
        "fixed_pose": True,
        "reasons": ["sparse_gap"],
        "left_index": 0,
        "right_index": 1,
        "alpha": 0.5,
        "timestamp": 0.5,
        "source_ids": [],
        "gate_metrics": {
            "generated_sharpness": 0.01,
            "reference_sharpness": 0.01,
            "sharpness_gain": 1.0,
        },
        "acceptance_class": "geometry_only",
    }
    frames = [originals[0], synthetic, originals[1]]
    signature = "a" * 64
    generation_id = "b" * 32
    manifest_path = tmp_path / "upstream_manifest.json"
    report_path = tmp_path / "upstream_report.json"
    report = {
        "schema": "unblur_slam.framecrafter_preprocess_report.v1",
        "backend": "python_api",
        "backend_test_only": False,
        "uses_ground_truth_pose": False,
        "pose_source": "droid_traj_est_not_align",
        "preprocess_signature": signature,
        "generation_id": generation_id,
        "source_frame_count": 2,
        "planned_total_before_cap": 1,
        "planned_target_count": 1,
        "selected_target_count": 1,
        "generation_batch_count": 0,
        "backend_generate_call_count": 0,
        "accepted_target_count": 1,
        "rejected_target_count": 0,
        "sharp_accepted_target_count": 0,
        "geometry_only_target_count": 1,
        "geometry_rejected_target_count": 0,
        "accepted_output_sha256": synthetic_output_digest(frames),
        "source_input_sha256": source_input_digest(frames),
        "manifest": str(manifest_path),
        "generation_batches": [],
        "planned": [{"target_id": "target_000", "left_position": 0, "right_position": 1}],
        "accepted": [
            {
                "target_id": "target_000",
                "gate_support_source_indices": [0, 1],
                "context_ids": [],
                "acceptance_class": "geometry_only",
                "confidence": 0.8,
                "raw_gate_confidence": 0.8,
                "geometry_failures": [],
                "sharp_failures": ["sharpness_gain"],
                "metrics": synthetic["gate_metrics"],
                "rgb_path": str(synthetic_rgb),
                "depth_path": str(synthetic_depth),
            }
        ],
        "rejected": [],
        "quality_partition": {
            "sharp_accepted": [],
            "geometry_only": [],
            "rejected": [],
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest = {
        "schema": "unblur_slam.framecrafter_manifest.v1",
        "source_frame_count": 2,
        "generated_frame_count": 1,
        "pose_source": "droid_traj_est_not_align",
        "uses_ground_truth_pose": False,
        "frames": frames,
        "preprocess_signature": signature,
        "generation_id": generation_id,
        "backend": "python_api",
        "backend_test_only": False,
        "accepted_output_sha256": synthetic_output_digest(frames),
        "source_input_sha256": source_input_digest(frames),
        "preprocess_report_path": str(report_path),
        "preprocess_report_sha256": sha256_file(report_path),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validate_manifest_payload(
        manifest, manifest_path=manifest_path, require_provenance=True
    )
    return manifest_path, report_path, checkpoint, pre


def _two_synthetic_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, np.ndarray]:
    manifest_path, report_path, checkpoint, pre = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    first = next(item for item in manifest["frames"] if item["kind"] == "synthetic")
    second = json.loads(json.dumps(first))
    second["target_id"] = "target_001"
    second["timestamp"] = 0.75
    second["alpha"] = 0.75
    second["c2w"][0][3] = 0.0075
    originals = [item for item in manifest["frames"] if item["kind"] == "original"]
    manifest["frames"] = [originals[0], first, second, originals[1]]
    manifest["generated_frame_count"] = 2
    manifest["accepted_output_sha256"] = synthetic_output_digest(
        manifest["frames"]
    )
    manifest["source_input_sha256"] = source_input_digest(manifest["frames"])

    second_record = json.loads(json.dumps(report["accepted"][0]))
    second_record["target_id"] = second["target_id"]
    report["accepted"].append(second_record)
    report["planned"].append(
        {
            "target_id": second["target_id"],
            "left_position": 0,
            "right_position": 1,
        }
    )
    report.update(
        planned_total_before_cap=2,
        planned_target_count=2,
        selected_target_count=2,
        accepted_target_count=2,
        accepted_output_sha256=manifest["accepted_output_sha256"],
        source_input_sha256=manifest["source_input_sha256"],
    )
    report["quality_partition"]["geometry_only"] = json.loads(
        json.dumps(report["accepted"])
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest["preprocess_report_sha256"] = sha256_file(report_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validate_manifest_payload(
        manifest, manifest_path=manifest_path, require_provenance=True
    )
    return manifest_path, report_path, checkpoint, pre


def _run(
    tmp_path: Path,
    fake,
    *,
    config: PostprocessConfig = PostprocessConfig(),
):
    manifest, report, checkpoint, pre = _fixture(tmp_path)
    summary = postprocess(
        manifest_path=manifest,
        report_path=report,
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "postprocessed",
        infer=fake,
        test_only=True,
        config=config,
    )
    return summary, pre


def test_phone_like_local_collapse_falls_back(tmp_path: Path) -> None:
    def fake(image: np.ndarray, timestamp: float) -> np.ndarray:
        del timestamp
        result = np.clip(0.5 + 1.10 * (image - 0.5), 0.0, 1.0)
        # Globally this adds strong edges, but locally it collapses a textured
        # phone/keypad-sized object into a black block.
        result[16:48, 16:48] = 0.0
        return result

    summary, _ = _run(tmp_path, fake)
    assert summary["replace_count"] == 0
    assert summary["fallback_count"] == 1
    decision = summary["decisions"][0]
    assert decision["action"] == "fallback_to_framecrafter"
    assert decision["post_laplacian_sharpness"] > decision["pre_laplacian_sharpness"]

    report = json.loads(Path(summary["report"]).read_text(encoding="utf-8"))
    detail = report["postprocess"]["decisions"][0]
    assert detail["quality"]["global_sharpness_improved"] is True
    assert detail["quality"]["local_gate"]["passed"] is False
    assert any(
        reason.startswith("local_")
        for reason in detail["quality"]["replacement_failures"]
    )
    assert detail["candidate"]["content_addressed"] is True
    assert Path(detail["candidate"]["rgb_path"]).stem == detail["candidate"][
        "rgb_sha256"
    ]


def test_clean_improvement_replaces_and_binds_hashes(tmp_path: Path) -> None:
    def fake(image: np.ndarray, timestamp: float) -> np.ndarray:
        del timestamp
        return np.clip(0.5 + 1.12 * (image - 0.5), 0.0, 1.0)

    summary, _ = _run(tmp_path, fake)
    assert summary["replace_count"] == 1
    assert summary["fallback_count"] == 0
    decision = summary["decisions"][0]
    assert decision["action"] == "replace_with_evssm"
    assert decision["post_vs_pre_sharpness_gain"] > 1.0

    manifest_path = Path(summary["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    synthetic = next(item for item in manifest["frames"] if item["kind"] == "synthetic")
    assert synthetic["rgb_sha256"] == sha256_file(synthetic["rgb_path"])
    assert Path(synthetic["rgb_path"]).stem == synthetic["rgb_sha256"]
    assert synthetic["eval"] is False and synthetic["fixed_pose"] is True
    originals = [item for item in manifest["frames"] if item["kind"] == "original"]
    assert [item["source_index"] for item in originals] == [0, 1]
    assert all(item["eval"] is True and item["fixed_pose"] is False for item in originals)

    report = json.loads(Path(summary["report"]).read_text(encoding="utf-8"))
    detail = report["postprocess"]["decisions"][0]
    assert detail["evssm_checkpoint"]["sha256"] == summary["checkpoint_sha256"]
    assert detail["quality"]["local_gate"]["passed"] is True
    assert detail["quality"]["post_evaluate_candidate"]["geometry_passed"] is True
    assert manifest["preprocess_report_sha256"] == sha256_file(summary["report"])

    # An injected backend is visibly non-production and cannot enter SLAM.
    with pytest.raises(ValueError, match="real SLAM accepts only"):
        validate_manifest_payload(
            manifest, manifest_path=manifest_path, require_provenance=True
        )


def test_failed_candidate_can_be_rejected_without_touching_originals(
    tmp_path: Path,
) -> None:
    def fake(image: np.ndarray, timestamp: float) -> np.ndarray:
        del timestamp
        return image.copy()

    summary, _ = _run(
        tmp_path,
        fake,
        config=PostprocessConfig(failure_policy="reject"),
    )
    assert summary["reject_count"] == 1
    assert summary["output_synthetic_count"] == 0
    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    assert [item["kind"] for item in manifest["frames"]] == ["original", "original"]
    assert all(item["eval"] is True for item in manifest["frames"])


def test_upstream_artifact_hash_guard_runs_before_inference(tmp_path: Path) -> None:
    manifest, report, checkpoint, _ = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    original = next(item for item in payload["frames"] if item["kind"] == "original")
    Path(original["rgb_path"]).write_bytes(b"tampered")
    called = False

    def fake(image: np.ndarray, timestamp: float) -> np.ndarray:
        nonlocal called
        called = True
        return image

    with pytest.raises(ValueError, match="original artifact hash mismatch"):
        postprocess(
            manifest_path=manifest,
            report_path=report,
            checkpoint_path=checkpoint,
            output_dir=tmp_path / "postprocessed",
            infer=fake,
            test_only=True,
        )
    assert called is False


def test_ground_truth_provenance_is_rejected_before_inference(tmp_path: Path) -> None:
    manifest, report, checkpoint, _ = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["uses_ground_truth_pose"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    called = False

    def fake(image: np.ndarray, timestamp: float) -> np.ndarray:
        nonlocal called
        called = True
        return image

    with pytest.raises(ValueError, match="uses_ground_truth_pose=false"):
        postprocess(
            manifest_path=manifest,
            report_path=report,
            checkpoint_path=checkpoint,
            output_dir=tmp_path / "postprocessed",
            infer=fake,
            test_only=True,
        )
    assert called is False


def test_checkpoint_identity_guard_runs_before_inference(tmp_path: Path) -> None:
    manifest, report, checkpoint, _ = _fixture(tmp_path)
    called = False

    def fake(image: np.ndarray, timestamp: float) -> np.ndarray:
        nonlocal called
        called = True
        return image

    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        postprocess(
            manifest_path=manifest,
            report_path=report,
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            output_dir=tmp_path / "postprocessed",
            infer=fake,
            test_only=True,
        )
    assert called is False


def test_malformed_last_support_fails_before_any_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, report, checkpoint, _ = _two_synthetic_fixture(tmp_path)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_payload["accepted"][-1]["gate_support_source_indices"] = [-1, 1]
    report.write_text(json.dumps(report_payload), encoding="utf-8")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["preprocess_report_sha256"] = sha256_file(report)
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    # The shared upstream validator does not consume postprocess-specific gate
    # supports; this stage must still reject all jobs before the first GPU call.
    validate_manifest_payload(
        manifest_payload, manifest_path=manifest, require_provenance=True
    )
    model_loads = 0
    calls = 0

    def fake(image: np.ndarray, timestamp: float) -> np.ndarray:
        nonlocal calls
        del timestamp
        calls += 1
        return image.copy()

    def fake_builder(checkpoint_path: Path, device: str):
        nonlocal model_loads
        del checkpoint_path, device
        model_loads += 1
        return fake

    monkeypatch.setattr(MODULE, "build_evssm_inference", fake_builder)

    with pytest.raises(ValueError, match="gate supports.*not original frames"):
        postprocess(
            manifest_path=manifest,
            report_path=report,
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256=sha256_file(checkpoint),
            output_dir=tmp_path / "postprocessed",
        )
    assert model_loads == 0
    assert calls == 0
