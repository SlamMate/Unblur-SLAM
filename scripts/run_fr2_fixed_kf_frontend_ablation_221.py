#!/usr/bin/env python3
"""Preflight or run the 221-frame fixed-keyframe EVSSM/TURTLE ablation.

Both arms use the exact source-index schedule
``[0,9,15,49,58,72,89,109,125,166,220]`` and the same published online
optimization budgets.  Poses/depths are estimated independently: sharing the
EVSSM arm's poses with TURTLE would leak a baseline outcome and would no longer
measure each frontend's effect on tracking.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "scripts/run_fr2_official_online_budget_paired_221.py"
_SPEC = importlib.util.spec_from_file_location("paired_221_base", BASE_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = BASE
_SPEC.loader.exec_module(BASE)

EXPECTED_FIXED_SOURCE_KEYFRAMES = (0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220)
CONFIGS = {
    "baseline": REPO_ROOT
    / "configs/local/fr2_xyz_fixed_kf_frontend_ablation_221/evssm_fixed_11kf.yaml",
    "turtle": REPO_ROOT
    / "configs/local/fr2_xyz_fixed_kf_frontend_ablation_221/turtle_gopro_fp16_fixed_11kf.yaml",
}
OUTPUT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_paired/fr2_xyz_fixed_kf_frontend_ablation_221"
).resolve()
OUTPUTS = {
    "baseline": (OUTPUT_ROOT / "evssm_fixed_11kf").resolve(),
    "turtle": (OUTPUT_ROOT / "turtle_gopro_fp16_fixed_11kf").resolve(),
}
PHYSICAL_GPU = BASE.PHYSICAL_GPU
FROZEN_BASELINE_VIDEO = Path(
    "/srv/szha0669/unblur-slam/slam_paired/fr2_xyz_official_online_budget_221/"
    "evssm_baseline/freiburg2_xyz/video.npz"
).resolve()
FROZEN_BASELINE_VIDEO_SHA256 = (
    "39afbe2135480ae77530719f7bab2a5facc1b5fe0e4f2c62f045fafe25802ed8"
)

# Reuse the already-audited artifact/dataset/compute validation, pointed at the
# new configs and outputs.  The base module reads these globals at call time.
BASE.CONFIGS = CONFIGS
BASE.OUTPUTS = OUTPUTS


def _validate_fixed_contract(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from src.utils.fixed_keyframes import parse_fixed_source_keyframe_contract

    normalized = {}
    for arm in ("baseline", "turtle"):
        cfg = configs[arm]
        contract = parse_fixed_source_keyframe_contract(cfg)
        if contract is None:
            raise ValueError(f"{arm}: fixed-source-keyframe contract is disabled")
        if tuple(contract["source_indices"]) != EXPECTED_FIXED_SOURCE_KEYFRAMES:
            raise ValueError(f"{arm}: fixed source-keyframe schedule drifted")
        disclosure = cfg.get("fixed_kf_frontend_ablation_221", {}) or {}
        expected_disclosure = (
            "unblur_slam.fr2_xyz_fixed_kf_frontend_ablation_221.v1",
            True,
            False,
            False,
            False,
        )
        observed_disclosure = (
            disclosure.get("schema"),
            disclosure.get("conditional_on_frozen_evssm_baseline_schedule"),
            disclosure.get("shares_pose_estimates_between_arms"),
            disclosure.get("uses_ground_truth_poses"),
            disclosure.get("runtime_baseline_artifact_dependency"),
        )
        if observed_disclosure != expected_disclosure:
            raise ValueError(f"{arm}: fixed-keyframe scope disclosure drifted")
        configured_provenance = (
            disclosure.get("provenance_verified_during_cpu_preflight"),
            Path(str(disclosure.get("frozen_baseline_video_npz", "")))
            .expanduser()
            .resolve(),
            str(disclosure.get("frozen_baseline_video_npz_sha256", "")),
        )
        expected_provenance = (
            True,
            FROZEN_BASELINE_VIDEO,
            FROZEN_BASELINE_VIDEO_SHA256,
        )
        if configured_provenance != expected_provenance:
            raise ValueError(f"{arm}: frozen-baseline provenance declaration drifted")
        normalized[arm] = contract
    if normalized["baseline"] != normalized["turtle"]:
        raise ValueError("arms do not share the exact fixed-keyframe contract")
    return normalized["baseline"]


def _validate_frozen_baseline_provenance() -> dict[str, Any]:
    """Verify where the frozen list came from without importing its geometry."""

    if not FROZEN_BASELINE_VIDEO.is_file():
        raise FileNotFoundError(
            f"frozen baseline provenance is missing: {FROZEN_BASELINE_VIDEO}"
        )
    actual_sha = BASE.sha256_file(FROZEN_BASELINE_VIDEO)
    if actual_sha != FROZEN_BASELINE_VIDEO_SHA256:
        raise ValueError(
            "frozen baseline video bytes drifted: "
            f"expected {FROZEN_BASELINE_VIDEO_SHA256}, got {actual_sha}"
        )
    # Read timestamps only. Poses, depths, scale, metrics, and Gaussian state
    # are deliberately neither returned nor passed to either experiment arm.
    import numpy as np

    with np.load(FROZEN_BASELINE_VIDEO, allow_pickle=False) as archive:
        if "timestamps" not in archive.files:
            raise ValueError("frozen baseline video has no timestamps array")
        timestamps = tuple(int(value) for value in archive["timestamps"].tolist())
    if timestamps != EXPECTED_FIXED_SOURCE_KEYFRAMES:
        raise ValueError(
            "configured fixed schedule does not match pinned baseline timestamps"
        )
    return {
        "path": str(FROZEN_BASELINE_VIDEO),
        "sha256": actual_sha,
        "timestamps_only_read": True,
        "source_indices": list(timestamps),
        "poses_read_or_shared": False,
        "depths_read_or_shared": False,
        "arm_runtime_dependency": False,
    }


def _load_and_validate_configs() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    configs = BASE._load_configs()
    # Also enforces that configs differ only by frontend artifacts/output and
    # retain seed, resolution, mapper budgets, and all disabled extensions.
    BASE._validate_pair_contract(configs["baseline"], configs["turtle"])
    return configs, _validate_fixed_contract(configs)


def preflight(
    *, arms: Iterable[str] = ("baseline", "turtle"), check_output_available: bool = True
) -> dict[str, Any]:
    selected = tuple(arms)
    _configs, fixed = _load_and_validate_configs()
    audit = BASE.preflight(
        arms=selected, check_output_available=check_output_available
    )
    provenance = _validate_frozen_baseline_provenance()
    audit["schema"] = "unblur_slam.fr2_xyz_fixed_kf_frontend_ablation_221_preflight.v1"
    audit["fixed_keyframe_ablation"] = {
        "conditional_on_frozen_evssm_baseline_schedule": True,
        "coordinate_domain": fixed["coordinate_domain"],
        "source_indices": list(fixed["source_indices"]),
        "strict_exact_runtime_check": True,
        "runtime_baseline_artifact_dependency": False,
        "uses_ground_truth_poses": False,
        "poses_and_depths_estimated_independently_per_arm": True,
        "turtle_history_updates_on_all_source_frames": True,
        "frozen_schedule_provenance": provenance,
    }
    return audit


def _interrupt(process: subprocess.Popen[str], log: Any) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
    try:
        return int(process.wait(timeout=30))
    except subprocess.TimeoutExpired:
        log.write("[launcher] SIGINT timeout; forwarding SIGTERM\n")
        log.flush()
        os.killpg(process.pid, signal.SIGTERM)
        return int(process.wait())


def _assert_physical_gpu_free() -> dict[str, Any]:
    """Fail before creating an arm output when physical GPU 1 is occupied."""

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("cannot inspect physical GPUs with nvidia-smi") from error
    rows = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"cannot parse nvidia-smi GPU row: {line!r}")
        rows.append(fields)
    matches = [row for row in rows if row[0] == str(PHYSICAL_GPU)]
    if len(matches) != 1:
        raise RuntimeError(f"physical GPU {PHYSICAL_GPU} was not resolved exactly once")
    index, uuid, name, total_mib, used_mib, utilization = matches[0]

    try:
        processes_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("cannot inspect GPU compute processes with nvidia-smi") from error
    processes = []
    for line in processes_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[0] == uuid:
            processes.append(
                {
                    "pid": int(fields[1]),
                    "process_name": fields[2],
                    "used_memory_mib": int(fields[3]),
                }
            )
    used_mib_int = int(used_mib)
    # A few MiB of driver bookkeeping is normal on an otherwise idle GPU.
    if processes or used_mib_int > 64:
        raise RuntimeError(
            f"physical GPU {PHYSICAL_GPU} is busy: "
            f"used_memory_mib={used_mib_int}, compute_processes={processes}"
        )
    return {
        "physical_gpu": int(index),
        "uuid": uuid,
        "name": name,
        "memory_total_mib": int(total_mib),
        "memory_used_mib": used_mib_int,
        "utilization_percent_snapshot": int(utilization),
        "compute_processes": processes,
        "max_idle_memory_mib": 64,
        "passed": True,
    }


def _run_arm(arm: str, audit: Mapping[str, Any]) -> int:
    output = OUTPUTS[arm]
    output.mkdir(parents=True, exist_ok=False)
    (output / "preflight.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = [sys.executable, str(REPO_ROOT / "run.py"), str(CONFIGS[arm])]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": PHYSICAL_GPU,
            "PYTHONUNBUFFERED": "1",
            "UNBLUR_SKIP_NR_IQA": "1",
        }
    )
    log_path = output / "launch.log"
    started = time.monotonic()
    code = -1
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        log.write(f"[launcher] fixed-11KF frontend ablation arm={arm}\n")
        log.write(
            "[launcher] source_keyframes="
            + json.dumps(list(EXPECTED_FIXED_SOURCE_KEYFRAMES))
            + "\n"
        )
        log.write("[launcher] poses=independent gt_poses=false runtime_baseline_read=false\n")
        log.write("[launcher] init=1050 mapping=100 tracking=100 final=100\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            code = int(process.wait())
        except KeyboardInterrupt:
            log.write("[launcher] KeyboardInterrupt; forwarding SIGINT\n")
            code = _interrupt(process, log)
        log.write(f"[launcher] exit_code={code}\n")
    (output / "launcher_runtime.json").write_text(
        json.dumps(
            {
                "schema": "unblur_slam.external_wall_runtime.v1",
                "arm": arm,
                "wall_runtime_seconds": time.monotonic() - started,
                "exit_code": code,
                "physical_gpu": int(PHYSICAL_GPU),
                "process_device": "cuda:0",
                "fixed_source_keyframes": list(EXPECTED_FIXED_SOURCE_KEYFRAMES),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return code


def _selected_arms(value: str) -> tuple[str, ...]:
    return ("baseline", "turtle") if value == "all" else (value,)


def run_pair(selection: str) -> int:
    arms = _selected_arms(selection)
    audit = preflight(arms=arms, check_output_available=True)
    for arm in arms:
        gpu_guard = _assert_physical_gpu_free()
        arm_audit = json.loads(json.dumps(audit))
        arm_audit["execution"]["gpu_free_guard_before_launch"] = gpu_guard
        print(f"[launch] {arm}: physical GPU {PHYSICAL_GPU} -> process cuda:0")
        code = _run_arm(arm, arm_audit)
        if code != 0:
            return code
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true", help="CPU-only validation (default)")
    action.add_argument("--run", action="store_true", help="launch the selected arm(s)")
    parser.add_argument("--arm", choices=("baseline", "turtle", "all"), default="all")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.run:
            return run_pair(args.arm)
        print(json.dumps(preflight(arms=_selected_arms(args.arm)), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
