#!/usr/bin/env python3
"""Independent fail-closed verifier for the I2-SLAM/TUM frontend ATE run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path("/srv/szha0669/unblur-slam/experiments/i2slam_tum_frontend_ate_v1")
ARMS = ("raw", "unblur_slam_evssm", "turtle_gopro_online", "turtle_bsd_online")
SCENES = {
    "freiburg1_desk": 592,
    "freiburg2_xyz": 3397,
    "freiburg3_office": 2515,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    report_path = ROOT / "report.json"
    receipts_path = ROOT / "receipts.json"
    if not report_path.is_file() or not receipts_path.is_file():
        print("formal output is incomplete", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    require(report.get("status") == "complete", "report is not complete")
    require(
        report.get("schema") == "unblur_slam.i2slam_tum_frontend_ate_report.v1",
        "unexpected report schema",
    )
    require(isinstance(receipts, list) and len(receipts) == 12, "expected 12 receipts")
    expected_pairs = {(arm, scene) for arm in ARMS for scene in SCENES}
    observed_pairs = set()
    for receipt in receipts:
        pair = (receipt.get("arm"), receipt.get("scene"))
        require(pair in expected_pairs, f"unexpected receipt {pair}")
        require(pair not in observed_pairs, f"duplicate receipt {pair}")
        observed_pairs.add(pair)
        require(receipt.get("exit_code") == 0, f"failed receipt {pair}")
        config_path = ROOT / "configs" / f"{pair[0]}_{pair[1]}.yaml"
        log_path = ROOT / "logs" / f"{pair[0]}_{pair[1]}.log"
        require(sha256_file(config_path) == receipt.get("config_sha256"), f"config SHA mismatch {pair}")
        require(sha256_file(log_path) == receipt.get("log_sha256"), f"log SHA mismatch {pair}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        require("Traceback" not in log_text, f"traceback in {pair}")
        require("CUDA out of memory" not in log_text, f"OOM in {pair}")
    require(observed_pairs == expected_pairs, "receipt set mismatch")

    values = {arm: [] for arm in ARMS}
    verified = {}
    for scene, expected_rows in SCENES.items():
        verified[scene] = {}
        for arm in ARMS:
            path = ROOT / "tracking" / arm / scene / "traj" / "traj_full_full_traj.npz"
            require(path.is_file(), f"missing trajectory {arm}/{scene}")
            with np.load(path) as payload:
                required = {
                    "traj_est_poses",
                    "traj_ref_poses",
                    "timestamps",
                    "ate_rmse",
                    "uses_ground_truth_pose",
                }
                require(required.issubset(payload.files), f"trajectory fields missing {arm}/{scene}")
                est = np.asarray(payload["traj_est_poses"], dtype=np.float64)
                ref = np.asarray(payload["traj_ref_poses"], dtype=np.float64)
                timestamps = np.asarray(payload["timestamps"])
                require(est.shape == ref.shape == (expected_rows, 4, 4), f"pose shape mismatch {arm}/{scene}")
                require(len(timestamps) == expected_rows, f"timestamp count mismatch {arm}/{scene}")
                require(not bool(payload["uses_ground_truth_pose"]), f"GT pose used by tracker {arm}/{scene}")
                error = est[:, :3, 3] - ref[:, :3, 3]
                recomputed = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
                stored = float(payload["ate_rmse"])
                require(np.isfinite(recomputed), f"non-finite ATE {arm}/{scene}")
                require(abs(recomputed - stored) <= 1e-12, f"NPZ ATE mismatch {arm}/{scene}")
            declared = report["per_scene"][scene][arm]
            trajectory_sha = sha256_file(path)
            require(abs(recomputed - float(declared["ate_rmse_m"])) <= 1e-12, f"report ATE mismatch {arm}/{scene}")
            require(trajectory_sha == declared["trajectory_sha256"], f"trajectory SHA mismatch {arm}/{scene}")
            require(int(declared["trajectory_rows"]) == expected_rows, f"reported rows mismatch {arm}/{scene}")
            values[arm].append(recomputed)
            verified[scene][arm] = recomputed

    means = {arm: float(np.mean(items)) for arm, items in values.items()}
    for arm, mean in means.items():
        require(abs(mean - float(report["scene_mean_ate_rmse_m"][arm])) <= 1e-12, f"mean mismatch {arm}")
    bsd = means["turtle_bsd_online"]
    relative = {}
    for arm in ("raw", "unblur_slam_evssm", "turtle_gopro_online"):
        delta = bsd - means[arm]
        percent = (bsd / means[arm] - 1.0) * 100.0
        require(abs(delta - float(report["turtle_bsd_delta_m"][arm])) <= 1e-12, f"delta mismatch {arm}")
        require(abs(percent - float(report["turtle_bsd_relative_percent"][arm])) <= 1e-10, f"relative mismatch {arm}")
        relative[arm] = percent
    result = {
        "schema": "unblur_slam.i2slam_tum_frontend_ate_independent_verification.v1",
        "status": "pass",
        "report_sha256": sha256_file(report_path),
        "receipts_sha256": sha256_file(receipts_path),
        "verified_trajectory_count": 12,
        "scene_mean_ate_rmse_m": means,
        "turtle_bsd_relative_percent": relative,
        "per_scene_recomputed": verified,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
