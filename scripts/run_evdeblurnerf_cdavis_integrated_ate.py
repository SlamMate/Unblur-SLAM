#!/usr/bin/env python3
"""Integrated Unblur-SLAM frontend ATE on real Ev-DeblurNeRF CDAVIS."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

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
    SCENES,
    sha256_file,
    tracking_config,
)

ARMS = ("unblur_slam_evssm", "turtle_gopro_online", "turtle_bsd_online")
SOURCE = Path("/srv/szha0669/unblur-slam/experiments/evdeblurnerf_cdavis_frontend_ate_v4")
OUTPUT = Path("/srv/szha0669/unblur-slam/experiments/evdeblurnerf_cdavis_integrated_ate_v1")
PYTHON = Path("/srv/szha0669/unblur-slam/env/bin/python")


def arm_deblur(arm: str) -> dict:
    if arm == "unblur_slam_evssm":
        return {"frontend": "evssm", "causal_checkpoint": ""}
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
    return {
        **common,
        "frontend": "turtle_bsd_streaming",
        "turtle_config": "/srv/szha0669/unblur-slam/external/TURTLE/options/Turtle_Derain_VRDS.yml",
        "turtle_config_sha256": PINNED_BSD_CONFIG_SHA256,
        "turtle_checkpoint": "/srv/szha0669/real_video_data/bsd_3ms24ms_official_quarantine/BSD_Deblur.pth",
        "turtle_checkpoint_sha256": PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
    }


def config(arm: str, scene: str) -> dict:
    cfg = tracking_config(OUTPUT, arm, scene)
    cfg["fake_sharp"] = True
    cfg["data"]["dataset_root"] = str(SOURCE / "derived")
    cfg["data"]["input_folder"] = "raw"
    cfg["data"]["output"] = str(OUTPUT / "tracking" / arm)
    cfg["deblur"] = arm_deblur(arm)
    cfg["evssm_checkpoint"] = (
        "/srv/szha0669/unblur-slam/pretrained/net_g_latest_batch_8_no_NYU.pth"
        if arm == "unblur_slam_evssm" else ""
    )
    cfg["third_party_ate_contract"]["input_arm"] = arm
    cfg["third_party_ate_contract"]["integration"] = (
        "native_unblur_slam_motion_filter_online_frontend"
    )
    return cfg


def main() -> int:
    preflight = {
        "schema": "unblur_slam.evdeblurnerf_cdavis_integrated_ate_preflight.v1",
        "source_materialization": str(SOURCE / "materialization.json"),
        "source_materialization_sha256": sha256_file(SOURCE / "materialization.json"),
        "source_report_sha256": sha256_file(SOURCE / "report.json"),
        "arms": list(ARMS),
        "scenes": list(SCENES),
        "model_sha256": {
            "unblur_slam_evssm": EVSSM_SHA256,
            "turtle_gopro_online": PINNED_TURTLE_CHECKPOINT_SHA256,
            "turtle_bsd_online": PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
        },
        "output_absent": not OUTPUT.exists(),
    }
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if "--run" not in sys.argv:
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("integrated ATE launch requires CUDA_VISIBLE_DEVICES=1")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "preflight.json").open("x", encoding="utf-8") as stream:
        json.dump(preflight, stream, indent=2, sort_keys=True)
    configs = OUTPUT / "configs"; configs.mkdir()
    logs = OUTPUT / "logs"; logs.mkdir()
    receipts = []
    for arm in ARMS:
        for scene in SCENES:
            cfg_path = configs / f"{arm}_{scene}.yaml"
            with cfg_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(config(arm, scene), stream, sort_keys=False)
            log_path = logs / f"{arm}_{scene}.log"
            command = [str(PYTHON), "-B", str(ROOT / "run.py"), str(cfg_path), "--only_tracking"]
            started = time.perf_counter()
            with log_path.open("x", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            receipt = {"arm": arm, "scene": scene, "exit_code": result.returncode,
                       "wall_seconds": time.perf_counter()-started,
                       "config_sha256": sha256_file(cfg_path), "log_sha256": sha256_file(log_path)}
            receipts.append(receipt)
            if result.returncode:
                raise RuntimeError(f"failed {arm}/{scene}: {log_path}")
    with (OUTPUT / "receipts.json").open("x", encoding="utf-8") as stream:
        json.dump(receipts, stream, indent=2, sort_keys=True)
    per_scene = {}; values = {arm: [] for arm in ARMS}
    for scene in SCENES:
        per_scene[scene] = {}
        for arm in ARMS:
            path = OUTPUT / "tracking" / arm / scene / "traj" / "traj_full_full_traj.npz"
            payload = np.load(path); ate = float(payload["ate_rmse"])
            per_scene[scene][arm] = {"ate_rmse": ate, "trajectory_sha256": sha256_file(path)}
            values[arm].append(ate)
    means = {arm: float(np.mean(value)) for arm, value in values.items()}
    report = {
        "schema": "unblur_slam.evdeblurnerf_cdavis_integrated_ate_report.v1",
        "status": "complete",
        "scope": "native_online_frontend_plus_droid_tracking_only_not_gaussian_mapping",
        "per_scene": per_scene,
        "scene_mean_ate_rmse": means,
        "bsd_delta_vs_unblur_slam_evssm": means["turtle_bsd_online"]-means["unblur_slam_evssm"],
        "bsd_delta_vs_turtle_gopro": means["turtle_bsd_online"]-means["turtle_gopro_online"],
    }
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
