#!/usr/bin/env python3
"""CPU contract test for FrameCrafter manifest application."""

import json
import hashlib
import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.augmented_dataset import SCHEMA, apply_framecrafter_manifest
from src.framecrafter_pipeline import source_input_digest, synthetic_output_digest


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    with tempfile.TemporaryDirectory() as root_string:
        root = Path(root_string)
        rgb_paths = []
        depth_paths = []
        for index, value in enumerate((32, 224)):
            rgb = root / f"rgb_{index}.png"
            depth = root / f"depth_{index}.png"
            cv2.imwrite(str(rgb), np.full((8, 10, 3), value, np.uint8))
            cv2.imwrite(str(depth), np.full((8, 10), 1000 + index, np.uint16))
            rgb_paths.append(str(rgb))
            depth_paths.append(str(depth))

        synthetic_rgb = root / "generated.png"
        synthetic_depth = root / "generated_depth.png"
        cv2.imwrite(str(synthetic_rgb), np.full((8, 10, 3), 128, np.uint8))
        cv2.imwrite(str(synthetic_depth), np.full((8, 10), 1000, np.uint16))

        pose0 = np.eye(4)
        pose1 = np.eye(4)
        pose1[0, 3] = 1.0
        pose_mid = np.eye(4)
        pose_mid[0, 3] = 0.5
        dataset = SimpleNamespace(
            color_paths=rgb_paths,
            depth_paths=depth_paths,
            poses=np.stack([
                np.repeat(pose0[None], 2, axis=0),
                np.repeat(pose1[None], 2, axis=0),
            ]),
            gt_paths=rgb_paths.copy(),
            image_timestamps=np.asarray([0.0, 1.0]),
            n_img=2,
        )
        manifest = root / "manifest.json"
        report = root / "preprocess_report.json"
        signature = "a" * 64
        frames = [
                {
                    "kind": "original", "source_index": 0, "eval": True,
                    "fixed_pose": False, "rgb_path": rgb_paths[0],
                    "depth_path": depth_paths[0], "c2w": pose0.tolist(),
                    "confidence": 1.0, "timestamp": 0.0,
                    "rgb_sha256": sha256(rgb_paths[0]),
                    "depth_sha256": sha256(depth_paths[0]),
                },
                {
                    "kind": "synthetic",
                    "target_id": "syn_0_1_00_0.500000",
                    "rgb_path": str(synthetic_rgb),
                    "depth_path": str(synthetic_depth),
                    "c2w": pose_mid.tolist(),
                    "confidence": 0.4,
                    "eval": False,
                    "fixed_pose": True,
                    "left_index": 0,
                    "right_index": 1,
                    "alpha": 0.5,
                    "source_index": None,
                    "timestamp": 0.5,
                    "reasons": ["large_pose_delta"],
                    "source_ids": ["0", "1"],
                    "gate_metrics": {"sharpness_gain": 1.1},
                    "rgb_sha256": sha256(synthetic_rgb),
                    "depth_sha256": sha256(synthetic_depth),
                },
                {
                    "kind": "original", "source_index": 1, "eval": True,
                    "fixed_pose": False, "rgb_path": rgb_paths[1],
                    "depth_path": depth_paths[1], "c2w": pose1.tolist(),
                    "confidence": 1.0, "timestamp": 1.0,
                    "rgb_sha256": sha256(rgb_paths[1]),
                    "depth_sha256": sha256(depth_paths[1]),
                },
            ]
        accepted_digest = synthetic_output_digest(frames)
        source_digest = source_input_digest(frames)
        report_payload = {
            "schema": "unblur_slam.framecrafter_preprocess_report.v1",
            "backend": "python_api",
            "backend_test_only": False,
            "uses_ground_truth_pose": False,
            "pose_source": "droid_traj_est_not_align",
            "preprocess_signature": signature,
            "generation_id": "1" * 32,
            "source_frame_count": 2,
            "accepted_target_count": 1,
            "accepted_output_sha256": accepted_digest,
            "source_input_sha256": source_digest,
            "manifest": str(manifest),
        }
        report.write_text(json.dumps(report_payload), encoding="utf-8")
        manifest_payload = {
            "schema": SCHEMA,
            "uses_ground_truth_pose": False,
            "pose_source": "droid_traj_est_not_align",
            "source_frame_count": 2,
            "generated_frame_count": 1,
            "preprocess_signature": signature,
            "generation_id": "1" * 32,
            "backend": "python_api",
            "backend_test_only": False,
            "accepted_output_sha256": accepted_digest,
            "source_input_sha256": source_digest,
            "preprocess_report_path": str(report),
            "preprocess_report_sha256": sha256(report),
            "frames": frames,
        }
        manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

        apply_framecrafter_manifest(dataset, manifest, expected_signature=signature)
        assert dataset.n_img == 3
        assert dataset.frame_metadata[1]["synthetic"]
        assert not dataset.frame_metadata[1]["eval"]
        assert dataset.frame_metadata[1]["confidence"] == 0.4
        assert dataset.poses.shape == (3, 2, 4, 4)
        assert np.allclose(dataset.poses[1, 0], pose_mid)
        assert np.allclose(dataset.image_timestamps, [0.0, 0.5, 1.0])

        scene_b_rgb = root / "scene_b_rgb.png"
        cv2.imwrite(str(scene_b_rgb), np.full((8, 10, 3), 99, np.uint8))
        wrong_scene_dataset = SimpleNamespace(
            color_paths=[str(scene_b_rgb), rgb_paths[1]],
            depth_paths=depth_paths,
            poses=np.stack([pose0, pose1]),
            gt_paths=[str(scene_b_rgb), rgb_paths[1]],
            image_timestamps=np.asarray([0.0, 1.0]),
            n_img=2,
        )
        expect_value_error(
            lambda: apply_framecrafter_manifest(wrong_scene_dataset, manifest)
        )
        expect_value_error(
            lambda: apply_framecrafter_manifest(
                wrong_scene_dataset, manifest, expected_signature="b" * 64
            )
        )
        missing_timestamps_tum = SimpleNamespace(
            name="tumrgbd",
            color_paths=rgb_paths,
            depth_paths=depth_paths,
            poses=np.stack([pose0, pose1]),
            gt_paths=rgb_paths.copy(),
            image_timestamps=None,
            n_img=2,
        )
        expect_value_error(
            lambda: apply_framecrafter_manifest(
                missing_timestamps_tum, manifest, expected_signature=signature
            )
        )

        invalid_dataset = SimpleNamespace(
            color_paths=rgb_paths,
            depth_paths=depth_paths,
            poses=np.stack([pose0, pose1]),
            gt_paths=rgb_paths.copy(),
            image_timestamps=np.asarray([0.0, 1.0]),
            n_img=2,
        )
        invalid_manifest = root / "invalid_manifest.json"
        invalid_payload = copy.deepcopy(manifest_payload)
        invalid_payload["frames"][2]["source_index"] = 0
        invalid_manifest.write_text(json.dumps(invalid_payload), encoding="utf-8")
        expect_value_error(
            lambda: apply_framecrafter_manifest(invalid_dataset, invalid_manifest)
        )

    print("augmented_dataset_contract=PASS")


if __name__ == "__main__":
    main()
