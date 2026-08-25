#!/usr/bin/env python3
"""Run a fresh EVSSM/GoPro-TURTLE/BSD-TURTLE TUM ATE comparison.

This is a bounded 221-frame fr2_xyz diagnostic using identical SLAM budgets.
It is not the full three-sequence or 26K paper protocol.  The BSD checkpoint
is an external t0 architecture/training reference, so its comparison to the
GoPro t1 checkpoint is descriptive rather than an architecture-only effect.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/unblur_slam.yaml"
CONFIG_DIR = ROOT / "configs/local/fr2_xyz_bsd_turtle_ate_221"
OUTPUT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_paired/fr2_xyz_bsd_turtle_ate_221_v2"
)
CONFIGS = {
    "evssm": CONFIG_DIR / "evssm.yaml",
    "turtle_gopro": CONFIG_DIR / "turtle_gopro.yaml",
    "turtle_bsd": CONFIG_DIR / "turtle_bsd.yaml",
}
OUTPUTS = {arm: OUTPUT_ROOT / arm for arm in CONFIGS}
ARMS = tuple(CONFIGS)
EXPECTED_FRONTENDS = {
    "evssm": "evssm",
    "turtle_gopro": "turtle_streaming",
    "turtle_bsd": "turtle_bsd_streaming",
}
EXPECTED_PREFIX = (0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220)
EXPECTED_FULL_PROTOCOL = (
    0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220, 319, 374, 407,
    435, 470, 483, 523, 568, 704, 750, 789, 827, 926, 1004, 1160,
    1251, 1342, 1409, 1460, 1553, 1692, 1795, 1889, 1978, 2055,
    2206, 2282, 2358, 2425, 2590, 2764,
)
EXPECTED = {
    "evssm": "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41",
    "droid": "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526",
    "omnidata": "a0fab23fee64aa9e4bbe0b520b18b196ea7594a7f719c1d8c10cf11dcb6e4a1e",
    "gopro": "10334b3e81d0416bcde5ccaca960dc81dbfb5b6d23e53fadaf7896d72b580c82",
    "bsd": "183d5e344488382a39c32aad86559e6fd568954134ddfee8793f5851a2cdf809",
}
GPU_UUID = "GPU-3501b285-78cd-1494-87f1-ccac2136866e"
GPU_SERIAL = "1711224002341"
GPU_LOCK = Path("/srv/szha0669/unblur-slam/locks/physical_gpu1.lock")
ALLOWED_DIFFS = {
    "data.output",
    "deblur.frontend",
    "deblur.turtle_checkpoint",
    "deblur.turtle_checkpoint_sha256",
    "deblur.turtle_config",
    "deblur.turtle_config_sha256",
    "deblur.turtle_inference_precision",
    "deblur.turtle_repo",
    "deblur.turtle_repo_commit",
    "deblur.stream_apply_to_tracking",
    "deblur.stream_every_frame",
    "deblur.stream_min_laplacian_gain",
    "deblur.stream_replace_sharp",
    "evssm_checkpoint",
    "evssm_checkpoint_sha256",
}
ATE_RE = re.compile(r"statistics:\s*(\{[^\n]+\})")
STREAM_RE = re.compile(
    r"TRACKING: streaming deblur frame=(\d+).*?replace=(True|False).*?time_ms=([-+0-9.eE]+)"
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(flatten(value[key], path))
    return result


def load_configs() -> dict[str, dict[str, Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from thirdparty.glorie_slam.config import load_config

    return {arm: load_config(str(path), str(DEFAULT_CONFIG)) for arm, path in CONFIGS.items()}


def require_hash(path: Any, configured: Any, expected: str, label: str) -> Path:
    candidate = Path(str(path or "")).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"missing {label}: {candidate}")
    if str(configured or "").lower() != expected:
        raise ValueError(f"{label} configured SHA mismatch")
    actual = sha256_file(candidate)
    if actual != expected:
        raise ValueError(f"{label} byte SHA mismatch: {actual}")
    return candidate


def validate_configs(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = configs["evssm"]
    expected_common = (
        str(baseline.get("dataset", "")).lower(), baseline.get("scene"),
        int(baseline.get("max_frames", -1)), int(baseline.get("stride", -1)),
        int(baseline.get("setup_seed", -1)), baseline.get("device"),
        int(baseline["cam"]["W_out"]), int(baseline["cam"]["H_out"]),
        int(baseline["mapping"]["Training"]["init_itr_num"]),
        int(baseline["mapping"]["Training"]["mapping_itr_num"]),
        int(baseline["mapping"]["Training"]["tracking_itr_num"]),
        int(baseline["mapping"]["final_refine_iters"]),
    )
    wanted = ("tumrgbd", "freiburg2_xyz", 221, 1, 43, "cuda:0", 512, 384, 1050, 100, 100, 100)
    if expected_common != wanted:
        raise ValueError(f"common TUM/SLAM budget drifted: {expected_common}")
    for arm, cfg in configs.items():
        if Path(cfg["data"]["output"]).resolve() != OUTPUTS[arm]:
            raise ValueError(f"{arm} output drifted")
        if cfg["deblur"]["frontend"] != EXPECTED_FRONTENDS[arm]:
            raise ValueError(f"{arm} frontend drifted")
        if cfg["mapping"]["resplat"]["enabled"] or cfg["submaps"]["enabled"]:
            raise ValueError(f"{arm} map extensions must be disabled")
        disclosure = cfg["paired_official_online_budget_221"]
        if disclosure.get("schema") != "unblur_slam.fr2_xyz_bsd_turtle_ate_221.v2":
            raise ValueError(f"{arm} disclosure schema drifted")
        if tuple(cfg["evaluation"]["expected_clear_gt_source_indices"]) != EXPECTED_PREFIX:
            raise ValueError(f"{arm} clear-GT prefix drifted")
        left, right = flatten(baseline), flatten(cfg)
        diffs = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
        unexpected = sorted(diffs - ALLOWED_DIFFS)
        if unexpected:
            raise ValueError(f"{arm} differs outside frontend/artifacts/output: {unexpected}")
        if arm != "evssm":
            deblur = cfg["deblur"]
            if any((
                deblur.get("stream_every_frame") is not True,
                deblur.get("stream_apply_to_tracking") is not True,
                deblur.get("stream_replace_sharp") is not False,
                float(deblur.get("stream_min_laplacian_gain", -1)) != 0.02,
                deblur.get("turtle_inference_precision") != "fp16",
            )):
                raise ValueError(f"{arm} streaming policy drifted")
    return {
        arm: canonical_sha(cfg) for arm, cfg in configs.items()
    }


def preflight(*, require_fresh: bool = True) -> dict[str, Any]:
    configs = load_configs()
    resolved = validate_configs(configs)
    if require_fresh and (OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink()):
        raise FileExistsError(f"refusing to overwrite output root: {OUTPUT_ROOT}")

    evssm = require_hash(configs["evssm"]["evssm_checkpoint"], configs["evssm"]["evssm_checkpoint_sha256"], EXPECTED["evssm"], "EVSSM")
    droid = require_hash(configs["evssm"]["tracking"]["pretrained"], configs["evssm"]["tracking"]["pretrained_sha256"], EXPECTED["droid"], "DROID")
    omnidata = require_hash(configs["evssm"]["mono_prior"]["depth_pretrained"], configs["evssm"]["mono_prior"]["depth_pretrained_sha256"], EXPECTED["omnidata"], "Omnidata")

    from src.turtle_backend import validate_turtle_artifacts
    from src.turtle_official_bsd_backend import validate_official_bsd_artifacts

    gopro = validate_turtle_artifacts(configs["turtle_gopro"]["deblur"], load_weights=True)
    bsd = validate_official_bsd_artifacts(
        repo=configs["turtle_bsd"]["deblur"]["turtle_repo"],
        config=configs["turtle_bsd"]["deblur"]["turtle_config"],
        checkpoint=configs["turtle_bsd"]["deblur"]["turtle_checkpoint"],
        checkpoint_sha256=configs["turtle_bsd"]["deblur"]["turtle_checkpoint_sha256"],
        load_weights=True,
    )
    if gopro.checkpoint_sha256 != EXPECTED["gopro"] or bsd.checkpoint_sha256 != EXPECTED["bsd"]:
        raise ValueError("TURTLE checkpoint identity drifted")

    from src.utils.datasets import get_dataset
    from src.utils.eval_frames import clear_gt_source_indices, validate_clear_gt_protocol_scope

    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        dataset = get_dataset(configs["evssm"], device="cpu")
    finally:
        os.chdir(previous)
    if len(dataset) != 221:
        raise ValueError("TUM prefix length drifted")
    if tuple(sorted(clear_gt_source_indices(configs["evssm"], dataset) or ())) != EXPECTED_FULL_PROTOCOL:
        raise ValueError("full clear-GT protocol drifted")
    if tuple(sorted(validate_clear_gt_protocol_scope(configs["evssm"], dataset) or ())) != EXPECTED_PREFIX:
        raise ValueError("bounded clear-GT protocol drifted")

    code_paths = [
        ROOT / "run.py", ROOT / "src/deblur_backends.py", ROOT / "src/turtle_backend.py",
        ROOT / "src/turtle_official_bsd_backend.py", ROOT / "src/tracker.py",
        ROOT / "thirdparty/glorie_slam/motion_filter.py", Path(__file__).resolve(),
    ]
    return {
        "schema": "unblur_slam.fr2_xyz_bsd_turtle_ate_221_preflight.v2",
        "status": "pass_cpu_only",
        "scope": {
            "scene": "freiburg2_xyz", "source_frames": 221,
            "dynamic_actual_slam_primary": True, "fixed_keyframes": False,
            "paper_three_scene": False, "paper_26k": False,
        },
        "resolved_config_sha256": resolved,
        "artifacts": {
            "evssm": {"path": str(evssm), "sha256": EXPECTED["evssm"]},
            "droid": {"path": str(droid), "sha256": EXPECTED["droid"]},
            "omnidata": {"path": str(omnidata), "sha256": EXPECTED["omnidata"]},
            "turtle_gopro": {"path": str(gopro.checkpoint), "sha256": gopro.checkpoint_sha256, "architecture": "t1"},
            "turtle_bsd": {"path": str(bsd.checkpoint), "sha256": bsd.checkpoint_sha256, "architecture": "t0"},
        },
        "implementation": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths},
        "execution": {"physical_gpu": 1, "logical_device": "cuda:0", "gpu_uuid": GPU_UUID, "gpu_serial": GPU_SERIAL, "lock": str(GPU_LOCK), "output_root": str(OUTPUT_ROOT)},
        "claims": {"bsd_restoration_gain_does_not_imply_ate_gain": True, "cross_architecture_training_comparison_descriptive": True},
    }


def gpu_identity_and_idle() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,serial,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    fields = [part.strip() for part in query[1].split(",")]
    if fields[0] != "1" or fields[2] != GPU_UUID or fields[3] != GPU_SERIAL or fields[1] != "NVIDIA RTX A6000":
        raise RuntimeError(f"physical GPU1 identity drifted: {fields}")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    foreign = [line for line in processes if GPU_UUID in line]
    if foreign:
        raise RuntimeError(f"physical GPU1 has existing compute processes: {foreign}")
    return {"index": 1, "name": fields[1], "uuid": fields[2], "serial": fields[3], "memory_total_mib": int(fields[4])}


def run_arm(arm: str, audit: Mapping[str, Any]) -> int:
    output = OUTPUTS[arm]
    output.mkdir(parents=True, exist_ok=False)
    (output / "preflight.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    command = [sys.executable, str(ROOT / "run.py"), str(CONFIGS[arm])]
    environment = os.environ.copy()
    environment.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1", "PYTHONUNBUFFERED": "1", "UNBLUR_SKIP_NR_IQA": "1"})
    started = time.monotonic()
    code = -1
    with (output / "launch.log").open("x", buffering=1) as log:
        log.write(f"[launcher] fresh three-arm BSD-ATE diagnostic arm={arm}\n")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line); sys.stdout.flush(); log.write(line)
            code = int(process.wait())
        except KeyboardInterrupt:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            code = int(process.wait())
        log.write(f"[launcher] exit_code={code}\n")
    (output / "launcher_runtime.json").write_text(json.dumps({
        "schema": "unblur_slam.external_wall_runtime.v1", "arm": arm,
        "wall_runtime_seconds": time.monotonic() - started, "exit_code": code,
        "physical_gpu": 1, "gpu_uuid": GPU_UUID, "process_device": "cuda:0",
    }, indent=2, sort_keys=True) + "\n")
    return code


def parse_ate(path: Path) -> float:
    match = ATE_RE.search(path.read_text(errors="replace"))
    if match is None:
        raise RuntimeError(f"cannot parse ATE: {path}")
    value = float(ast.literal_eval(match.group(1))["rmse"])
    if not math.isfinite(value):
        raise RuntimeError("ATE is not finite")
    return value


def build_report(audit: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for arm in ARMS:
        root = OUTPUTS[arm]
        scene = root / "freiburg2_xyz"
        launcher = json.loads((root / "launcher_runtime.json").read_text())
        if launcher.get("exit_code") != 0:
            raise RuntimeError(f"{arm} did not complete")
        metric = json.loads((scene / "psnr/after_refine/final_result.json").read_text())
        if tuple(metric.get("evaluated_source_indices", ())) != EXPECTED_PREFIX:
            raise RuntimeError(f"{arm} evaluated frame set drifted")
        runtime = json.loads((scene / "runtime_stats.json").read_text())
        with np.load(scene / "traj/traj_full_full_traj.npz", allow_pickle=False) as archive:
            timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
            if not np.array_equal(timestamps, np.arange(221, dtype=np.float64)):
                raise RuntimeError(f"{arm} trajectory is not full prefix")
            if bool(np.asarray(archive["uses_ground_truth_pose"]).item()):
                raise RuntimeError(f"{arm} trajectory used GT poses")
        with np.load(scene / "video.npz", allow_pickle=False) as archive:
            keyframes = [int(round(value)) for value in np.asarray(archive["timestamps"]).tolist()]
        log_text = (root / "launch.log").read_text(errors="replace")
        stream = STREAM_RE.findall(log_text)
        if arm != "evssm" and [int(item[0]) for item in stream] != list(range(221)):
            raise RuntimeError(f"{arm} did not stream every frame")
        online = float(runtime["online_inference_time"])
        results[arm] = {
            "frontend": EXPECTED_FRONTENDS[arm],
            "full_trajectory_ate_rmse_m": parse_ate(scene / "traj/metrics_full_traj.txt"),
            "keyframe_trajectory_ate_rmse_m": parse_ate(scene / "traj/metrics_kf_traj.txt"),
            "keyframe_count": len(keyframes), "keyframe_source_indices": keyframes,
            "rendering_clear_gt_prefix": {"psnr_db": float(metric["mean_psnr"]), "ssim": float(metric["mean_ssim"]), "lpips": float(metric["mean_lpips"]), "depth_l1": float(metric["mean_depthl1"])},
            "official_online_seconds": online, "derived_prefix_fps": 221.0 / online,
            "external_wall_seconds": float(launcher["wall_runtime_seconds"]),
            "streamed_frame_count": len(stream),
            "tracking_replacement_count": sum(item[1] == "True" for item in stream),
        }
    baseline = results["evssm"]
    deltas = {}
    for arm in ("turtle_gopro", "turtle_bsd"):
        deltas[f"{arm}_minus_evssm"] = {
            "full_ate_m": results[arm]["full_trajectory_ate_rmse_m"] - baseline["full_trajectory_ate_rmse_m"],
            "psnr_db": results[arm]["rendering_clear_gt_prefix"]["psnr_db"] - baseline["rendering_clear_gt_prefix"]["psnr_db"],
            "online_seconds": results[arm]["official_online_seconds"] - baseline["official_online_seconds"],
        }
    return {
        "schema": "unblur_slam.fr2_xyz_bsd_turtle_ate_221_report.v2",
        "status": "complete", "preflight_sha256": canonical_sha(audit),
        "results": results, "deltas": deltas,
        "interpretation": {
            "primary_metric": "dynamic full-trajectory ATE RMSE",
            "lower_ate_is_better": True,
            "not_full_paper_benchmark": True,
            "rendering_membership_clear_gt_conditioned": True,
            "bsd_t0_vs_gopro_t1_is_not_architecture_only": True,
            "no_test_metric_used_for_model_or_checkpoint_selection": True,
        },
    }


def run_all() -> int:
    audit = preflight(require_fresh=True)
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        audit = dict(audit)
        audit["runtime_gpu_identity"] = gpu_identity_and_idle()
        for arm in ARMS:
            print(f"[launch] {arm} on physical GPU1")
            code = run_arm(arm, audit)
            if code != 0:
                return code
    report = build_report(audit)
    audit_dir = OUTPUT_ROOT / "_audit"
    audit_dir.mkdir(exist_ok=False)
    report_path = audit_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.run:
            return run_all()
        print(json.dumps(preflight(require_fresh=True), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
