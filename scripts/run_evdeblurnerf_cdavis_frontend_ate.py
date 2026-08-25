#!/usr/bin/env python3
"""Third-party real-blur tracking ablation on Ev-DeblurNeRF CDAVIS.

The experiment intentionally separates restoration from tracking.  Every
100-ms exposure is restored once by E/G/B, quantized with the same rule, and
then passed to an otherwise identical DROID tracking-only run.  Short-exposure
images are never inputs.  Reference poses come from the dataset's per-image
motor-trajectory pose rows, not from COLMAP run on the restored images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.precompute_framecrafter_evssm import build_evssm_inference  # noqa: E402
from src.turtle_backend import (  # noqa: E402
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    TurtleStreamingBackend,
)
from src.turtle_official_bsd_backend import (  # noqa: E402
    PINNED_BSD_ARCH_SHA256,
    PINNED_BSD_CONFIG_SHA256,
    PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
    PINNED_UPSTREAM_INFERENCE_SHA256,
    OfficialBsdTurtleStreamingBackend,
)


SCENES = (
    "blurbatteries",
    "blurdrones",
    "blurfigures",
    "blurlabequipment",
    "blurpowersupplies",
)
ARMS = ("raw", "evssm", "turtle_gopro", "turtle_bsd")
DATASET_ZIP_SHA256 = "1117dd16caef0cc4e05db830119edf00860177bc7415f6778cc2bfb65e7c200a"
EVSSM_SHA256 = "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
EXPECTED = {
    "blurbatteries": (23, 18),
    "blurdrones": (16, 11),
    "blurfigures": (20, 15),
    "blurlabequipment": (22, 17),
    "blurpowersupplies": (23, 18),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, expected_sha: str | None = None) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha is not None:
        actual = sha256_file(path)
        if actual != expected_sha:
            raise ValueError(f"SHA256 mismatch for {path}: {actual}")
    return path


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_inventory(dataset_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scene in SCENES:
        folder = dataset_root / scene
        timestamps_path = require_file(folder / "images_1" / "timestamps.npz")
        poses_path = require_file(folder / "poses_bounds.npy")
        all_poses_path = require_file(folder / "all_poses_bounds.npy")
        all_timestamps_path = require_file(folder / "all_timestamps.npy")
        z = np.load(timestamps_path)
        timestamps = np.asarray(z["timestamps"], dtype=np.int64)
        starts = np.asarray(z["timestamps_start"], dtype=np.int64)
        ends = np.asarray(z["timestamps_end"], dtype=np.int64)
        poses = np.load(poses_path)
        all_poses = np.load(all_poses_path)
        all_timestamps = np.load(all_timestamps_path)
        images = sorted((folder / "images_1").glob("[0-9][0-9].png"))
        expected_total, expected_blur = EXPECTED[scene]
        if not (
            len(images) == len(timestamps) == len(starts) == len(ends) == len(poses) == expected_total
        ):
            raise ValueError(f"{scene}: image/timestamp/pose count mismatch")
        blur_indices = np.flatnonzero(starts != ends).astype(int).tolist()
        sharp_indices = np.flatnonzero(starts == ends).astype(int).tolist()
        if len(blur_indices) != expected_blur or len(sharp_indices) != 5:
            raise ValueError(f"{scene}: unexpected blur/sharp split")
        midpoint_error = np.abs(timestamps - ((starts + ends) // 2))
        exposure = ends[blur_indices] - starts[blur_indices]
        if int(midpoint_error.max(initial=0)) > 1:
            raise ValueError(f"{scene}: image timestamps are not exposure midpoints")
        if int(exposure.min()) < 99_998 or int(exposure.max()) > 100_000:
            raise ValueError(f"{scene}: expected approximately 100-ms blur exposures")
        if poses.shape != (expected_total, 17) or all_poses.shape[1:] != (17,):
            raise ValueError(f"{scene}: unexpected LLFF pose shape")
        if len(all_timestamps) != len(all_poses):
            raise ValueError(f"{scene}: encoder timestamp/pose count mismatch")
        hwf = poses[0, :15].reshape(3, 5)[:, 4]
        if not np.allclose(hwf, [260.0, 346.0, 450.0020724445828], atol=1e-8):
            raise ValueError(f"{scene}: camera calibration drifted: {hwf}")
        result[scene] = {
            "image_count": len(images),
            "blur_indices": blur_indices,
            "sharp_indices_excluded": sharp_indices,
            "exposure_us_min": int(exposure.min()),
            "exposure_us_max": int(exposure.max()),
            "timestamp_midpoint_max_error_us": int(midpoint_error.max(initial=0)),
            "timestamps_sha256": sha256_file(timestamps_path),
            "poses_bounds_sha256": sha256_file(poses_path),
            "all_poses_bounds_sha256": sha256_file(all_poses_path),
            "all_timestamps_sha256": sha256_file(all_timestamps_path),
            "blur_image_sha256": [sha256_file(images[index]) for index in blur_indices],
            "sharp_image_bytes_opened": False,
            "sharp_image_pixels_decoded": False,
        }
    return result


def _tensor_from_rgb(image: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    ).unsqueeze(0).to(device)
    height, width = tensor.shape[-2:]
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8
    return F.pad(tensor, (0, pad_width, 0, pad_height), mode="reflect")


def _rgb8(output: np.ndarray) -> np.ndarray:
    if output.shape != (260, 346, 3) or output.dtype not in (np.float32, np.float64):
        raise ValueError(f"unexpected restored image {output.shape}/{output.dtype}")
    if not np.isfinite(output).all():
        raise ValueError("restored image contains NaN or Inf")
    return np.rint(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)


def materialize(args: argparse.Namespace, output_root: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    derived = output_root / "derived"
    derived.mkdir(parents=True, exist_ok=False)
    device = str(args.device)

    evssm_infer = build_evssm_inference(args.evssm_checkpoint, device)
    gopro_cfg = {
        "turtle_repo": str(args.turtle_repo),
        "turtle_config": str(args.gopro_config),
        "turtle_checkpoint": str(args.gopro_checkpoint),
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
        "turtle_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
        "turtle_inference_precision": "fp16",
    }
    gopro = TurtleStreamingBackend.from_config(gopro_cfg, device=device)
    bsd_cfg = {
        "turtle_repo": str(args.turtle_repo),
        "turtle_config": str(args.bsd_config),
        "turtle_checkpoint": str(args.bsd_checkpoint),
        "turtle_checkpoint_sha256": PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
        "turtle_inference_precision": "fp16",
    }
    bsd = OfficialBsdTurtleStreamingBackend.from_config(bsd_cfg, device=device)

    manifest: dict[str, Any] = {
        "schema": "unblur_slam.evdeblurnerf_cdavis_frontend_materialization.v1",
        "arms": {},
        "scene_order": list(SCENES),
        "quantization": "clip_0_1_round_times_255_rgb8",
        "short_exposure_input_count": 0,
        "uses_pose_or_sharp_pixels_for_restoration": False,
    }
    for arm in ARMS:
        manifest["arms"][arm] = {}

    for scene in SCENES:
        source_scene = args.dataset_root / scene
        timestamps = np.load(source_scene / "images_1" / "timestamps.npz")
        poses = np.load(source_scene / "poses_bounds.npy")
        indices = inventory[scene]["blur_indices"]
        selected_poses = poses[indices]
        selected_ts = {}
        for key in timestamps.files:
            value = np.asarray(timestamps[key])
            selected_ts[key] = (
                value[indices]
                if value.ndim == 1 and len(value) == len(poses)
                else value
            )
        source_images = [source_scene / "images_1" / f"{index:02d}.png" for index in indices]
        gopro.reset()
        bsd.reset()

        for arm in ARMS:
            scene_out = derived / arm / scene
            image_out = scene_out / "images_4"
            image_out.mkdir(parents=True, exist_ok=False)
            np.save(scene_out / "poses_bounds.npy", selected_poses)
            np.savez(scene_out / "timestamps_selected.npz", **selected_ts)

        arm_hashes = {arm: [] for arm in ARMS}
        arm_seconds = {arm: 0.0 for arm in ARMS}
        for local_index, source_path in enumerate(source_images):
            bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (260, 346, 3):
                raise ValueError(f"failed to decode {source_path}")
            rgb8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb8.astype(np.float32) / 255.0
            outputs: dict[str, np.ndarray] = {"raw": rgb8}

            start = time.perf_counter()
            outputs["evssm"] = _rgb8(evssm_infer(rgb, float(local_index)))
            torch.cuda.synchronize()
            arm_seconds["evssm"] += time.perf_counter() - start

            tensor = _tensor_from_rgb(rgb, device)
            start = time.perf_counter()
            outputs["turtle_gopro"] = _rgb8(
                gopro.step(tensor, timestamp=local_index)[0, :, :260, :346]
                .detach().cpu().numpy().transpose(1, 2, 0)
            )
            torch.cuda.synchronize()
            arm_seconds["turtle_gopro"] += time.perf_counter() - start

            start = time.perf_counter()
            outputs["turtle_bsd"] = _rgb8(
                bsd.step(tensor, timestamp=local_index)[0, :, :260, :346]
                .detach().cpu().numpy().transpose(1, 2, 0)
            )
            torch.cuda.synchronize()
            arm_seconds["turtle_bsd"] += time.perf_counter() - start

            for arm, image in outputs.items():
                path = derived / arm / scene / "images_4" / f"{local_index:02d}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"failed to write {path}")
                arm_hashes[arm].append(sha256_file(path))
        for arm in ARMS:
            manifest["arms"][arm][scene] = {
                "frames": len(indices),
                "png_sha256": arm_hashes[arm],
                "model_step_seconds": arm_seconds[arm],
                "poses_bounds_sha256": sha256_file(derived / arm / scene / "poses_bounds.npy"),
            }
    manifest["manifest_payload_sha256"] = canonical_sha(manifest)
    path = output_root / "materialization.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    del evssm_infer, gopro, bsd
    torch.cuda.empty_cache()
    return manifest


def tracking_config(output_root: Path, arm: str, scene: str) -> dict[str, Any]:
    return {
        "inherit_from": str(ROOT / "configs/deblur_nerf_motion/deblur_nerf_motion.yaml"),
        "scene": scene,
        "dataset": "evdeblurnerf_cdavis",
        "only_tracking": True,
        "max_frames": -1,
        "stride": 1,
        "device": "cuda:0",
        "fake_sharp": False,
        "clear_init": False,
        "sharp_judge": False,
        "adjust_cam": False,
        "backend_compensation": False,
        "exam_blur_score": False,
        "composite_blur": False,
        "deblur": {"frontend": "evssm", "open": False},
        "cam": {
            "H": 260,
            "W": 346,
            "H_out": 256,
            "W_out": 344,
            "H_edge": 0,
            "W_edge": 0,
            "fx": 450.0020724445828,
            "fy": 450.0020724445828,
            "cx": 173.0,
            "cy": 130.0,
            "png_depth_scale": 1.0,
        },
        "tracking": {
            "pretrained": "/srv/szha0669/unblur-slam/pretrained/droid.pth",
            "buffer": 100,
            "warmup": 4,
            "backend": {"final_ba": True},
        },
        "mono_prior": {
            "predict_online": True,
            "depth_pretrained": "/srv/szha0669/unblur-slam/pretrained/omnidata_dpt_depth_v2.ckpt",
        },
        "data": {
            "dataset_root": str(output_root / "derived"),
            "input_folder": arm,
            "output": str(output_root / "tracking" / arm),
        },
        "third_party_ate_contract": {
            "schema": "unblur_slam.evdeblurnerf_cdavis_frontend_ate.v1",
            "input_arm": arm,
            "scene": scene,
            "reference": "evdeblurnerf_cdavis_motor_encoder_pose_at_image_timestamp",
            "blur_exposures_only": True,
            "short_exposure_inputs": 0,
            "restoration_uses_reference_pose": False,
        },
    }


def run_tracking(args: argparse.Namespace, output_root: Path) -> list[dict[str, Any]]:
    config_root = output_root / "configs"
    log_root = output_root / "logs"
    config_root.mkdir(exist_ok=False)
    log_root.mkdir(exist_ok=False)
    records = []
    for arm in ARMS:
        for scene in SCENES:
            cfg = tracking_config(output_root, arm, scene)
            cfg_path = config_root / f"{arm}_{scene}.yaml"
            with cfg_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(cfg, stream, sort_keys=False)
            log_path = log_root / f"{arm}_{scene}.log"
            command = [str(args.python), "-B", str(ROOT / "run.py"), str(cfg_path), "--only_tracking"]
            start = time.perf_counter()
            with log_path.open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.physical_gpu)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record = {
                "arm": arm,
                "scene": scene,
                "command": command,
                "exit_code": completed.returncode,
                "wall_seconds": time.perf_counter() - start,
                "config_sha256": sha256_file(cfg_path),
                "log_sha256": sha256_file(log_path),
            }
            records.append(record)
            if completed.returncode != 0:
                raise RuntimeError(f"tracking failed for {arm}/{scene}; see {log_path}")
    with (output_root / "tracking_receipts.json").open("x", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2, sort_keys=True)
    return records


def report(output_root: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    per_scene: dict[str, Any] = {}
    values = {arm: [] for arm in ARMS}
    for scene in SCENES:
        per_scene[scene] = {}
        for arm in ARMS:
            path = output_root / "tracking" / arm / scene / "traj" / "traj_full_full_traj.npz"
            require_file(path)
            payload = np.load(path)
            ate = float(np.asarray(payload["ate_rmse"]).reshape(()))
            frame_count = int(len(payload["timestamps"]))
            if frame_count != len(inventory[scene]["blur_indices"]):
                raise ValueError(f"{arm}/{scene}: full trajectory frame count mismatch")
            per_scene[scene][arm] = {"ate_rmse": ate, "frame_count": frame_count, "trajectory_sha256": sha256_file(path)}
            values[arm].append(ate)
    aggregate = {
        arm: {
            "scene_mean_ate_rmse": float(np.mean(values[arm])),
            "scene_median_ate_rmse": float(np.median(values[arm])),
            "scene_values": values[arm],
        }
        for arm in ARMS
    }
    for arm in ARMS[1:]:
        aggregate[arm]["delta_vs_raw_scene_mean"] = aggregate[arm]["scene_mean_ate_rmse"] - aggregate["raw"]["scene_mean_ate_rmse"]
        aggregate[arm]["scenes_better_than_raw"] = int(sum(a < b for a, b in zip(values[arm], values["raw"])))
    aggregate["turtle_bsd"]["delta_vs_evssm_scene_mean"] = aggregate["turtle_bsd"]["scene_mean_ate_rmse"] - aggregate["evssm"]["scene_mean_ate_rmse"]
    aggregate["turtle_bsd"]["delta_vs_turtle_gopro_scene_mean"] = aggregate["turtle_bsd"]["scene_mean_ate_rmse"] - aggregate["turtle_gopro"]["scene_mean_ate_rmse"]
    result = {
        "schema": "unblur_slam.evdeblurnerf_cdavis_frontend_ate_report.v1",
        "status": "complete",
        "dataset": "Ev-DeblurNeRF CDAVIS",
        "third_party_real_capture": True,
        "blur_exposure_ms": 100,
        "reference_pose_source": "camera_motor_encoder_trajectory_published_by_dataset",
        "colmap_pose_used_as_ate_reference": False,
        "short_exposure_images_used": False,
        "claim_scope": "tracking_only_restoration_frontend_ablation_not_full_unblur_slam_mapping",
        "per_scene": per_scene,
        "aggregate": aggregate,
    }
    result["canonical_payload_sha256"] = canonical_sha(result)
    path = output_root / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", type=Path, default=Path("/srv/szha0669/unblur-slam/env/bin/python"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/srv/szha0669/unblur-slam/datasets/evdeblurnerf_cdavis/ev-deblurnerf_cdavis"))
    parser.add_argument("--dataset-zip", type=Path, default=Path("/srv/szha0669/unblur-slam/datasets/evdeblurnerf_cdavis_download/evdeblurnerf_cdavis.zip"))
    parser.add_argument("--output-root", type=Path, default=Path("/srv/szha0669/unblur-slam/experiments/evdeblurnerf_cdavis_frontend_ate_v4"))
    parser.add_argument("--evssm-checkpoint", type=Path, default=Path("/srv/szha0669/unblur-slam/pretrained/net_g_latest_batch_8_no_NYU.pth"))
    parser.add_argument("--turtle-repo", type=Path, default=Path("/srv/szha0669/unblur-slam/external/TURTLE"))
    parser.add_argument("--gopro-config", type=Path, default=Path("/srv/szha0669/unblur-slam/external/TURTLE/options/Turtle_Deblur_Gopro.yml"))
    parser.add_argument("--gopro-checkpoint", type=Path, default=Path("/srv/szha0669/unblur-slam/pretrained/turtle/GoPro_Deblur.pth"))
    parser.add_argument("--bsd-config", type=Path, default=Path("/srv/szha0669/unblur-slam/external/TURTLE/options/Turtle_Derain_VRDS.yml"))
    parser.add_argument("--bsd-checkpoint", type=Path, default=Path("/srv/szha0669/real_video_data/bsd_3ms24ms_official_quarantine/BSD_Deblur.pth"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    args.python = require_file(args.python)
    require_file(args.dataset_zip, DATASET_ZIP_SHA256)
    require_file(args.evssm_checkpoint, EVSSM_SHA256)
    require_file(args.gopro_checkpoint, PINNED_TURTLE_CHECKPOINT_SHA256)
    require_file(args.gopro_config, PINNED_TURTLE_CONFIG_SHA256)
    require_file(args.bsd_checkpoint, PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256)
    require_file(args.bsd_config, PINNED_BSD_CONFIG_SHA256)
    if not args.turtle_repo.is_dir():
        raise FileNotFoundError(args.turtle_repo)
    inventory = source_inventory(args.dataset_root)
    preflight = {
        "schema": "unblur_slam.evdeblurnerf_cdavis_frontend_ate_preflight.v1",
        "dataset_zip_sha256": DATASET_ZIP_SHA256,
        "dataset_inventory": inventory,
        "total_blur_frames": sum(len(value["blur_indices"]) for value in inventory.values()),
        "total_short_exposure_frames_excluded": sum(len(value["sharp_indices_excluded"]) for value in inventory.values()),
        "model_pins": {
            "evssm": EVSSM_SHA256,
            "turtle_gopro": PINNED_TURTLE_CHECKPOINT_SHA256,
            "turtle_bsd": PINNED_OFFICIAL_BSD_CHECKPOINT_SHA256,
            "turtle_commit": PINNED_TURTLE_COMMIT,
            "gopro_arch": PINNED_TURTLE_ARCH_SHA256,
            "bsd_arch": PINNED_BSD_ARCH_SHA256,
            "bsd_upstream_inference": PINNED_UPSTREAM_INFERENCE_SHA256,
        },
        "output_root_absent": not args.output_root.exists(),
        "gpu_started": False,
    }
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if not args.run:
        return 0
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("launch requires CUDA_VISIBLE_DEVICES to equal --physical-gpu")
    args.output_root.mkdir(parents=True, exist_ok=False)
    with (args.output_root / "preflight.json").open("x", encoding="utf-8") as stream:
        json.dump(preflight, stream, indent=2, sort_keys=True)
    materialize(args, args.output_root, inventory)
    run_tracking(args, args.output_root)
    result = report(args.output_root, inventory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
