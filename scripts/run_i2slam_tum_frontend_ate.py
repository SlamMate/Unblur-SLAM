#!/usr/bin/env python3
"""Four-arm native online frontend ATE on the I2-SLAM real TUM protocol.

This is deliberately a tracking-only experiment.  It compares untouched RGB,
the repository's native keyframe EVSSM path, and the two official streaming
TURTLE checkpoints without running Gaussian mapping.  The published I2-SLAM
sharp-frame membership may be validated by the common entry point, but it is
never used for tracking, frontend gating, or trajectory selection.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_evdeblurnerf_cdavis_frontend_ate import (  # noqa: E402
    EVSSM_SHA256,
    PINNED_BSD_CONFIG_SHA256,
    PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    sha256_file,
)
from src.utils.datasets import TUM_RGB  # noqa: E402
from thirdparty.glorie_slam import config as config_loader  # noqa: E402


ARMS = ("raw", "unblur_slam_evssm", "turtle_gopro_online", "turtle_bsd_online")
SCENES = {
    "freiburg1_desk": {
        "config": ROOT / "configs/I2slam/freiburg1_desk.yaml",
        "input_folder": "rgbd_dataset_freiburg1_desk",
        "expected_frames": 592,
    },
    "freiburg2_xyz": {
        "config": ROOT / "configs/I2slam/freiburg2_xyz.yaml",
        "input_folder": "rgbd_dataset_freiburg2_xyz",
        "expected_frames": 3397,
    },
    "freiburg3_office": {
        "config": ROOT / "configs/I2slam/freiburg3_office.yaml",
        "input_folder": "rgbd_dataset_freiburg3_long_office_household",
        "expected_frames": 2515,
    },
}
DATASET_ROOT = Path("/srv/szha0669/unblur-slam/datasets/TUM_RGBD")
OUTPUT = Path("/srv/szha0669/unblur-slam/experiments/i2slam_tum_frontend_ate_v1")
PYTHON = Path("/srv/szha0669/unblur-slam/env/bin/python")
DEFAULT_CONFIG = ROOT / "configs/unblur_slam.yaml"


def _arm_deblur(arm: str) -> dict[str, Any]:
    if arm == "raw":
        return {
            "frontend": "evssm",
            "open": False,
            "causal_checkpoint": "",
            "stream_every_frame": False,
        }
    if arm == "unblur_slam_evssm":
        return {
            "frontend": "evssm",
            "causal_checkpoint": "",
            "stream_every_frame": False,
        }
    common = {
        "causal_checkpoint": "",
        "stream_every_frame": True,
        "stream_apply_to_tracking": True,
        "stream_min_laplacian_gain": 0.02,
        "stream_replace_sharp": False,
        "turtle_inference_precision": "fp16",
        "turtle_repo": "/srv/szha0669/unblur-slam/external/TURTLE",
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
    }
    if arm == "turtle_gopro_online":
        return {
            **common,
            "frontend": "turtle_streaming",
            "turtle_config": "/srv/szha0669/unblur-slam/external/TURTLE/options/Turtle_Deblur_Gopro.yml",
            "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
            "turtle_checkpoint": "/srv/szha0669/unblur-slam/pretrained/turtle/GoPro_Deblur.pth",
            "turtle_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        }
    if arm == "turtle_bsd_online":
        return {
            **common,
            "frontend": "turtle_bsd_streaming",
            "turtle_config": "/srv/szha0669/unblur-slam/external/TURTLE/options/Turtle_Derain_VRDS.yml",
            "turtle_config_sha256": PINNED_BSD_CONFIG_SHA256,
            "turtle_checkpoint": "/srv/szha0669/real_video_data/bsd_3ms24ms_official_quarantine/BSD_Deblur.pth",
            "turtle_checkpoint_sha256": PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
        }
    raise ValueError(f"unknown arm {arm!r}")


def _arm_config(arm: str, scene: str) -> dict[str, Any]:
    scene_spec = SCENES[scene]
    return {
        "inherit_from": str(scene_spec["config"]),
        "scene": scene,
        "dataset": "tumrgbd",
        "only_tracking": True,
        "max_frames": -1,
        "stride": 1,
        "setup_seed": 43,
        "device": "cuda:0",
        "fake_sharp": arm != "raw",
        "evssm_checkpoint": (
            "/srv/szha0669/unblur-slam/pretrained/net_g_latest_batch_8_no_NYU.pth"
            if arm == "unblur_slam_evssm"
            else ""
        ),
        "evssm_checkpoint_sha256": (
            EVSSM_SHA256 if arm == "unblur_slam_evssm" else ""
        ),
        "framecrafter": {"enabled": False, "auto_prepare": False, "manifest": ""},
        "data": {
            "dataset_root": str(DATASET_ROOT),
            "input_folder": str(scene_spec["input_folder"]),
            "output": str(OUTPUT / "tracking" / arm),
        },
        "tracking": {
            "pretrained": "/srv/szha0669/unblur-slam/pretrained/droid.pth",
            "pretrained_sha256": "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526",
            "backend": {"final_ba": True},
        },
        "mono_prior": {
            "predict_online": True,
            "depth_pretrained": "/srv/szha0669/unblur-slam/pretrained/omnidata_dpt_depth_v2.ckpt",
            "depth_pretrained_sha256": "a0fab23fee64aa9e4bbe0b520b18b196ea7594a7f719c1d8c10cf11dcb6e4a1e",
        },
        "deblur": _arm_deblur(arm),
        "mapping": {"online_plotting": False, "resplat": {"enabled": False, "online_enabled": False}},
        "submaps": {"enabled": False, "loop_backend": "none", "official_resplat_sidecar": {"enabled": False}},
        "i2slam_tum_frontend_ate": {
            "schema": "unblur_slam.i2slam_tum_frontend_ate.v1",
            "scope": "native_online_frontend_plus_droid_tracking_only",
            "sharp_membership_used_for_tracking_or_gating": False,
            "full_sequence": True,
            "arm": arm,
        },
    }


def _dataset_inventory() -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for scene, spec in SCENES.items():
        folder = DATASET_ROOT / str(spec["input_folder"])
        files = {}
        for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            path = folder / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files[name] = sha256_file(path)
        cfg = config_loader.load_config(spec["config"], DEFAULT_CONFIG)
        cfg["data"]["dataset_root"] = str(DATASET_ROOT)
        cfg["data"]["input_folder"] = str(spec["input_folder"])
        dataset = TUM_RGB(cfg, device="cpu")
        count = len(dataset)
        if count != int(spec["expected_frames"]):
            raise ValueError(f"{scene}: expected {spec['expected_frames']} frames, got {count}")
        inventory[scene] = {
            "input_folder": str(folder),
            "associated_frame_count": count,
            "first_timestamp": float(dataset.image_timestamps[0]),
            "last_timestamp": float(dataset.image_timestamps[-1]),
            "protocol_file_sha256": files,
        }
    return inventory


def _preflight() -> dict[str, Any]:
    checkpoints = {
        "unblur_slam_evssm": (
            Path("/srv/szha0669/unblur-slam/pretrained/net_g_latest_batch_8_no_NYU.pth"),
            EVSSM_SHA256,
        ),
        "turtle_gopro_online": (
            Path("/srv/szha0669/unblur-slam/pretrained/turtle/GoPro_Deblur.pth"),
            PINNED_TURTLE_CHECKPOINT_SHA256,
        ),
        "turtle_bsd_online": (
            Path("/srv/szha0669/real_video_data/bsd_3ms24ms_official_quarantine/BSD_Deblur.pth"),
            PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
        ),
    }
    model_sha = {"raw": None}
    for arm, (path, expected) in checkpoints.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"checkpoint SHA mismatch for {arm}: {actual}")
        model_sha[arm] = actual
    return {
        "schema": "unblur_slam.i2slam_tum_frontend_ate_preflight.v1",
        "status": "pass_cpu_only",
        "dataset_identity": "I2-SLAM_real_protocol_on_TUM_RGBD_not_I2-SLAM_synthetic",
        "dataset_inventory": _dataset_inventory(),
        "arms": list(ARMS),
        "model_sha256": model_sha,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "output_root": str(OUTPUT),
        "output_absent": not OUTPUT.exists(),
        "sharp_membership_may_be_validated_but_not_used_for_tracking_gating_or_trajectory_selection": True,
        "gpu_started": False,
    }


def main() -> int:
    preflight = _preflight()
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if "--run" not in sys.argv:
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("I2-SLAM TUM ATE launch requires CUDA_VISIBLE_DEVICES=1")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("I2-SLAM TUM ATE launch requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "preflight.json").open("x", encoding="utf-8") as stream:
        json.dump(preflight, stream, indent=2, sort_keys=True)
    configs = OUTPUT / "configs"
    logs = OUTPUT / "logs"
    configs.mkdir()
    logs.mkdir()
    receipts = []
    for arm in ARMS:
        for scene in SCENES:
            cfg_path = configs / f"{arm}_{scene}.yaml"
            with cfg_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(_arm_config(arm, scene), stream, sort_keys=False)
            log_path = logs / f"{arm}_{scene}.log"
            command = [str(PYTHON), "-B", str(ROOT / "run.py"), str(cfg_path), "--only_tracking"]
            started = time.perf_counter()
            with log_path.open("x", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            receipt = {
                "arm": arm,
                "scene": scene,
                "exit_code": result.returncode,
                "wall_seconds": time.perf_counter() - started,
                "command": command,
                "config_sha256": sha256_file(cfg_path),
                "log_sha256": sha256_file(log_path),
            }
            receipts.append(receipt)
            if result.returncode:
                with (OUTPUT / "failed_receipts.json").open("x", encoding="utf-8") as stream:
                    json.dump(receipts, stream, indent=2, sort_keys=True)
                raise RuntimeError(f"failed {arm}/{scene}: {log_path}")
    with (OUTPUT / "receipts.json").open("x", encoding="utf-8") as stream:
        json.dump(receipts, stream, indent=2, sort_keys=True)

    per_scene: dict[str, Any] = {}
    values = {arm: [] for arm in ARMS}
    for scene in SCENES:
        per_scene[scene] = {}
        expected_frames = int(SCENES[scene]["expected_frames"])
        for arm in ARMS:
            path = OUTPUT / "tracking" / arm / scene / "traj" / "traj_full_full_traj.npz"
            payload = np.load(path)
            if bool(payload["uses_ground_truth_pose"]):
                raise RuntimeError(f"{arm}/{scene}: tracker trajectory used GT pose")
            timestamps = np.asarray(payload["timestamps"])
            if len(timestamps) != expected_frames:
                raise RuntimeError(
                    f"{arm}/{scene}: expected {expected_frames} trajectory rows, got {len(timestamps)}"
                )
            ate = float(payload["ate_rmse"])
            if not np.isfinite(ate):
                raise RuntimeError(f"{arm}/{scene}: non-finite ATE")
            per_scene[scene][arm] = {
                "ate_rmse_m": ate,
                "trajectory_rows": len(timestamps),
                "trajectory_sha256": sha256_file(path),
            }
            values[arm].append(ate)
    means = {arm: float(np.mean(items)) for arm, items in values.items()}
    bsd = means["turtle_bsd_online"]
    report = {
        "schema": "unblur_slam.i2slam_tum_frontend_ate_report.v1",
        "status": "complete",
        "scope": "native_online_frontend_plus_droid_tracking_only_not_gaussian_mapping",
        "dataset_identity": "I2-SLAM_real_protocol_on_three_TUM_RGBD_sequences",
        "per_scene": per_scene,
        "scene_mean_ate_rmse_m": means,
        "turtle_bsd_delta_m": {
            arm: bsd - means[arm] for arm in ("raw", "unblur_slam_evssm", "turtle_gopro_online")
        },
        "turtle_bsd_relative_percent": {
            arm: (bsd / means[arm] - 1.0) * 100.0
            for arm in ("raw", "unblur_slam_evssm", "turtle_gopro_online")
        },
        "sharp_membership_used_for_tracking_gating_or_trajectory_selection": False,
        "single_run_no_statistical_significance_claim": True,
    }
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
