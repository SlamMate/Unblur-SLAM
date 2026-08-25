#!/usr/bin/env python3
"""CPU contracts for the bounded fr2_xyz causal smoke launcher."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_fr2_causal_smoke as smoke


def expect(exception, function) -> None:
    try:
        function()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TinyCausal(torch.nn.Module):
    def forward(self, frames):
        return frames[:, -1]


class FakeDataset:
    def __init__(self, image: Path):
        repeated = [str(image)] * 221
        self.color_paths = repeated
        self.depth_paths = repeated
        self.gt_paths = repeated
        self.poses = np.repeat(np.eye(4)[None, None], 221 * 2, axis=0).reshape(
            221, 2, 4, 4
        )
        self.image_timestamps = np.arange(221, dtype=np.float64)
        self.intrinsic = np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
        self.distortion = np.zeros(5, dtype=np.float64)

    def frame_info(self, index):
        return {
            "source_index": int(index),
            "synthetic": False,
            "eval": True,
        }


def save_tiny_export(
    checkpoint: Path,
    evssm: Path,
    report: Path,
    manifest: Path,
) -> None:
    module = torch.jit.trace(TinyCausal(), torch.zeros(1, 5, 3, 16, 16))
    metadata = {
        "format": "unblur_slam.causal_video_deblur.torchscript.v1",
        "model_config": {
            "max_history": 5,
            "input_domain": "evssm",
            "use_teacher_input": False,
        },
        "training_contract": {
            "stream_prefix_padding": "repeat_first_frame_on_left",
            "supervised_output": "newest_frame_at_every_sequence_position",
        },
        "teacher_provenance": {
            "schema": "unblur_slam.video_deblur_teacher_provenance.v1",
            "storage": "precomputed_png_rgb8",
            "teacher_domain": "evssm_restored_rgb_0_1",
            "teacher_artifacts_verified": True,
            "evssm_checkpoint": str(evssm),
            "evssm_checkpoint_sha256": sha256(evssm),
            "precompute_report": str(report),
            "precompute_report_sha256": sha256(report),
            "teacher_manifest": str(manifest),
            "teacher_manifest_sha256": sha256(manifest),
        },
    }
    torch.jit.save(
        module,
        str(checkpoint),
        _extra_files={"metadata.json": json.dumps(metadata)},
    )


def main() -> None:
    configs = {arm: smoke._load(arm) for arm in smoke.ARM_CONFIGS}
    smoke._assert_static_matrix_contract(configs)
    assert tuple(
        configs["baseline"]["evaluation"]["expected_clear_gt_source_indices"]
    ) == smoke.EXPECTED_PREFIX

    drifted = copy.deepcopy(configs)
    drifted["replay"]["max_frames"] = 222
    expect(ValueError, lambda: smoke._assert_static_matrix_contract(drifted))
    drifted = copy.deepcopy(configs)
    drifted["causal"]["mapping"]["resplat"]["extra_iters"] = 24
    expect(ValueError, lambda: smoke._assert_static_matrix_contract(drifted))
    drifted = copy.deepcopy(configs)
    drifted["causal"]["cam"]["fx"] += 1.0
    expect(ValueError, lambda: smoke._assert_static_matrix_contract(drifted))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        image = root / "asset.bin"
        image.write_bytes(b"same-reader-asset")
        dataset = FakeDataset(image)
        first_identity = smoke._dataset_identity(dataset)
        assert len(first_identity) == 64
        dataset.poses[220, 1, 0, 3] = 1.0
        assert smoke._dataset_identity(dataset) != first_identity

        output_cfg = {"data": {"output": str(root / "arm")}, "scene": "scene"}
        smoke._assert_output_available(output_cfg)
        (root / "arm").mkdir()
        (root / "arm" / "launch.log").write_text("claimed\n")
        expect(FileExistsError, lambda: smoke._assert_output_available(output_cfg))

        evssm = root / "evssm.pth"
        report = root / "precompute.json"
        manifest = root / "teacher.jsonl"
        checkpoint = root / "causal.pt"
        evssm.write_bytes(b"official-test-evssm")
        report.write_text("{}\n")
        manifest.write_text("{}\n")
        save_tiny_export(checkpoint, evssm, report, manifest)

        original_hashes = dict(smoke.EXPECTED_HASHES)
        try:
            smoke.EXPECTED_HASHES["evssm"] = sha256(evssm)
            smoke.EXPECTED_HASHES["causal"] = sha256(checkpoint)
            cfg = {
                "evssm_checkpoint": str(evssm),
                "evssm_checkpoint_sha256": sha256(evssm),
                "deblur": {
                    "frontend": "causal_evssm",
                    "causal_checkpoint": str(checkpoint),
                    "causal_checkpoint_sha256": sha256(checkpoint),
                    "causal_history": 5,
                    "stream_every_frame": True,
                    "stream_apply_to_tracking": True,
                    "stream_replace_sharp": False,
                    "stream_min_laplacian_gain": 0.02,
                    "stream_min_vs_evssm_gain": 0.0,
                },
            }
            audit = smoke._validate_causal_export(cfg)
            assert audit["history"] == 5
            assert audit["teacher_storage"] == "precomputed_png_rgb8"

            wrong_history = copy.deepcopy(cfg)
            wrong_history["deblur"]["causal_history"] = 4
            expect(ValueError, lambda: smoke._validate_causal_export(wrong_history))
            evssm.write_bytes(b"mutated")
            expect(ValueError, lambda: smoke._validate_causal_export(cfg))
        finally:
            smoke.EXPECTED_HASHES.clear()
            smoke.EXPECTED_HASHES.update(original_hashes)

    print("fr2_causal_smoke=PASS")


if __name__ == "__main__":
    main()
