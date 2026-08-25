from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.framecrafter_lora_dataset import (
    PairedFrameCrafterDataset,
    build_paired_dataset,
    official_dataset_repeat_is_broken,
    sha256_file,
    validate_cpu_contract,
)
from scripts.launch_framecrafter_lora import main as launch_main


def _paired_manifest(tmp_path: Path, *, pose_kind: str = "estimated_unaligned") -> Path:
    blurry_dir = tmp_path / "source/blur"
    sharp_dir = tmp_path / "source/sharp"
    evssm_dir = tmp_path / "source/evssm"
    for path in (blurry_dir, sharp_dir, evssm_dir):
        path.mkdir(parents=True)
    blurry, sharp, evssm = [], [], []
    poses = []
    for index in range(8):
        blur_path = blurry_dir / f"{index:03d}.png"
        sharp_path = sharp_dir / f"{index:03d}.png"
        evssm_path = evssm_dir / f"{index:03d}.png"
        Image.new("RGB", (32, 24), (10 + index, 20, 30)).save(blur_path)
        Image.new("RGB", (32, 24), (200 + index, 210, 220)).save(sharp_path)
        Image.new("RGB", (32, 24), (90 + index, 100, 110)).save(evssm_path)
        blurry.append(str(blur_path.relative_to(tmp_path)))
        sharp.append(str(sharp_path.relative_to(tmp_path)))
        evssm.append(str(evssm_path.relative_to(tmp_path)))
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = 0.05 * index
        poses.append(pose.tolist())
    manifest = tmp_path / "paired.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence": "turtle",
                "blurry": blurry,
                "sharp": sharp,
                "evssm": evssm,
                "c2w": poses,
                "pose_provenance": {"kind": pose_kind, "source": "first-pass DROID estimate"},
                "K": [[28.0, 0, 16.0], [0, 28.0, 12.0], [0, 0, 1]],
                "camera_convention": "opencv",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_build_and_load_role_aware_dataset_with_repeat(tmp_path: Path) -> None:
    manifest = _paired_manifest(tmp_path)
    root = tmp_path / "built"
    summary = build_paired_dataset(
        manifest,
        root,
        num_input_frames=3,
        num_output_frames=1,
        sample_stride=2,
        context_mode="hybrid",
        hybrid_evssm_fraction=1 / 3,
    )
    assert summary.sample_count == 4
    assert sha256_file(root / "samples.jsonl") == summary.samples_sha256
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    assert all(len(row["contexts"]) == 3 and len(row["targets"]) == 1 for row in rows)
    assert all(row["targets"][0]["image_mode"] == "sharp" for row in rows)
    assert all(row["pose_provenance"]["kind"] == "estimated_unaligned" for row in rows)
    assert {frame["image_mode"] for frame in rows[0]["contexts"]} == {"raw_blurry", "evssm"}
    for row in rows:
        for frame in row["contexts"] + row["targets"]:
            assert sha256_file(root / frame["path"]) == frame["sha256"]
            assert len(frame["c2w_sha256"]) == 64
            assert len(frame["K_sha256"]) == 64

    dataset = PairedFrameCrafterDataset(
        base_path=str(root),
        metadata_path=str(root / "samples.jsonl"),
        repeat=3,
        num_frames=4,
        height=16,
        width=32,
        num_input_frames=3,
        num_output_frames=1,
    )
    assert len(dataset) == 12
    assert dataset[0]["sample_id"] == dataset[4]["sample_id"]
    contract = validate_cpu_contract(dataset)
    assert contract["status"] == "passed"
    assert contract["raymap_shape"] == [4, 384, 2, 4]
    sample = dataset[0]
    # Prefix is the selected blur/EVSSM context; the final supervision image is sharp.
    assert sample["target_images"][:3] == sample["input_images"]
    assert np.asarray(sample["target_images"][-1])[0, 0, 0] >= 200


def test_builder_rejects_gt_or_aligned_pose_provenance(tmp_path: Path) -> None:
    manifest = _paired_manifest(tmp_path, pose_kind="ground_truth")
    with pytest.raises(ValueError, match="pose_provenance.kind"):
        build_paired_dataset(manifest, tmp_path / "built", num_input_frames=3)


def test_builder_requires_evssm_for_evssm_mode(tmp_path: Path) -> None:
    manifest = _paired_manifest(tmp_path)
    row = json.loads(manifest.read_text())
    row["evssm"][2] = None
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="requires an EVSSM image"):
        build_paired_dataset(
            manifest,
            tmp_path / "built",
            num_input_frames=3,
            context_mode="evssm",
        )


def test_repeat_bug_detector_distinguishes_fixed_len(tmp_path: Path) -> None:
    source = tmp_path / "diffsynth/core/data/dataset.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class WanNVSDataset:\n"
        "    def __len__(self):\n"
        "        return len(self.video_list)\n"
    )
    assert official_dataset_repeat_is_broken(tmp_path)
    source.write_text(
        "class WanNVSDataset:\n"
        "    def __len__(self):\n"
        "        return len(self.video_list) * self.repeat\n"
    )
    assert not official_dataset_repeat_is_broken(tmp_path)


def test_launcher_defaults_to_cpu_contract_and_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = _paired_manifest(tmp_path)
    dataset_root = tmp_path / "built"
    build_paired_dataset(manifest, dataset_root, num_input_frames=3, num_output_frames=1)

    framecrafter_root = tmp_path / "FrameCrafter"
    (framecrafter_root / "model_training").mkdir(parents=True)
    (framecrafter_root / "model_training/train.py").write_text("raise SystemExit('must not run in dry-run')\n")
    dataset_source = framecrafter_root / "diffsynth/core/data/dataset.py"
    dataset_source.parent.mkdir(parents=True)
    dataset_source.write_text(
        "class WanNVSDataset:\n"
        "    def __len__(self):\n"
        "        return len(self.video_list)\n"
    )

    model_root = tmp_path / "models"
    wan = model_root / "Wan-AI/Wan2.1-I2V-14B-480P"
    tokenizer = model_root / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    wan.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    for name in (
        "diffusion_pytorch_model-00001-of-00001.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ):
        (wan / name).touch()
    checkpoint = model_root / "framecrafter.safetensors"
    checkpoint.touch()
    accelerate = tmp_path / "accelerate"
    accelerate.touch()
    output = tmp_path / "launch"

    assert launch_main(
        [
            "--dataset-root", str(dataset_root),
            "--framecrafter-root", str(framecrafter_root),
            "--base-model-root", str(model_root),
            "--framecrafter-checkpoint", str(checkpoint),
            "--output-path", str(output),
            "--accelerate", str(accelerate),
            "--height", "16",
            "--width", "32",
            "--num-input-frames", "3",
            "--num-output-frames", "1",
            "--dataset-repeat", "2",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "dry_run_only_no_training_started"
    assert report["official_repeat_bug_detected"] is True
    assert report["adapter_repeat_length"] == 16
    assert report["defaults"] == {
        "learning_rate": 5e-6,
        "lora_rank": 32,
        "lora_targets": "q,k,v,o,ffn.0,ffn.2",
    }
    spec = json.loads((output / "framecrafter_lora_launch.json").read_text())
    assert spec["official_train_py"].endswith("model_training/train.py")
    assert spec["profile"]["zero_stage"] == 3
    assert spec["profile"]["offload_parameters"] == "cpu"
