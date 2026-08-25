#!/usr/bin/env python3
"""Fail-closed CPU audit for the fixed-11KF EVSSM/TURTLE experiment.

No model, CUDA context, image metric, or trajectory optimization is run here.
The script verifies completed immutable records and reuses the audited paired
221-frame parser, adding strict fixed-schedule and provenance requirements.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "scripts/report_fr2_official_online_budget_paired_221.py"
_SPEC = importlib.util.spec_from_file_location("paired_221_report_base", BASE_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = BASE
_SPEC.loader.exec_module(BASE)

ContractError = BASE.ContractError
DEFAULT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_paired/fr2_xyz_fixed_kf_frontend_ablation_221"
).resolve()
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "_audit"
EXPECTED_FIXED_SOURCE_KEYFRAMES = (0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220)
FROZEN_BASELINE_VIDEO = Path(
    "/srv/szha0669/unblur-slam/slam_paired/fr2_xyz_official_online_budget_221/"
    "evssm_baseline/freiburg2_xyz/video.npz"
).resolve()
FROZEN_BASELINE_VIDEO_SHA256 = (
    "39afbe2135480ae77530719f7bab2a5facc1b5fe0e4f2c62f045fafe25802ed8"
)
BASE.ARMS = {
    "baseline": "evssm_fixed_11kf",
    "turtle": "turtle_gopro_fp16_fixed_11kf",
}

_ORIGINAL_VALIDATE_CONFIG = BASE._validate_config
_ORIGINAL_VALIDATE_PREFLIGHT = BASE._validate_preflight
_ORIGINAL_READ_RUNTIME = BASE._read_runtime
_ORIGINAL_READ_KEYFRAMES = BASE._read_keyframes
_ORIGINAL_VALIDATE_PROVENANCE = BASE._validate_provenance


def _fixed_contract_from_config(cfg: Mapping[str, Any], arm: str) -> dict[str, Any]:
    try:
        fixed = cfg["tracking"]["fixed_source_keyframes"]
        disclosure = cfg["fixed_kf_frontend_ablation_221"]
    except (KeyError, TypeError) as error:
        raise ContractError(f"{arm} resolved config lacks fixed-keyframe contract") from error
    expected_fixed = {
        "enabled": True,
        "schema": "unblur_slam.fixed_source_keyframes.v1",
        "coordinate_domain": "dataset_source_index",
        "strict_exact": True,
        "selection_source": "frozen_prior_evssm_baseline_schedule",
        "runtime_baseline_artifact_dependency": False,
        "uses_ground_truth_poses": False,
    }
    for key, expected in expected_fixed.items():
        BASE._expect(fixed.get(key), expected, f"{arm} fixed contract {key}")
    indices = tuple(int(value) for value in fixed.get("source_indices", ()))
    BASE._expect(indices, EXPECTED_FIXED_SOURCE_KEYFRAMES, f"{arm} fixed schedule")

    expected_disclosure = {
        "schema": "unblur_slam.fr2_xyz_fixed_kf_frontend_ablation_221.v1",
        "conditional_on_frozen_evssm_baseline_schedule": True,
        "shares_pose_estimates_between_arms": False,
        "uses_ground_truth_poses": False,
        "runtime_baseline_artifact_dependency": False,
        "provenance_verified_during_cpu_preflight": True,
        "frozen_baseline_video_npz": str(FROZEN_BASELINE_VIDEO),
        "frozen_baseline_video_npz_sha256": FROZEN_BASELINE_VIDEO_SHA256,
    }
    for key, expected in expected_disclosure.items():
        observed = disclosure.get(key)
        if key == "frozen_baseline_video_npz":
            observed = str(Path(str(observed)).expanduser().resolve())
        BASE._expect(observed, expected, f"{arm} fixed disclosure {key}")
    return {"fixed": dict(fixed), "disclosure": dict(disclosure)}


def _validate_config(cfg: Mapping[str, Any], arm: str, arm_root: Path) -> None:
    _ORIGINAL_VALIDATE_CONFIG(cfg, arm, arm_root)
    _fixed_contract_from_config(cfg, arm)


def _gpu_guard(preflight: Mapping[str, Any], arm: str) -> dict[str, Any]:
    guard = ((preflight.get("execution") or {}).get("gpu_free_guard_before_launch") or {})
    expected = {
        "physical_gpu": 1,
        "name": "NVIDIA RTX A6000",
        "compute_processes": [],
        "max_idle_memory_mib": 64,
        "passed": True,
    }
    for key, wanted in expected.items():
        BASE._expect(guard.get(key), wanted, f"{arm} GPU-free guard {key}")
    used = int(guard.get("memory_used_mib", -1))
    if used < 0 or used > 64:
        raise ContractError(f"{arm} GPU-free guard memory_used_mib={used}")
    return dict(guard)


def _validate_preflight(
    preflight: Mapping[str, Any], cfg: Mapping[str, Any], arm: str
) -> None:
    BASE._expect(
        preflight.get("schema"),
        "unblur_slam.fr2_xyz_fixed_kf_frontend_ablation_221_preflight.v1",
        f"{arm} fixed preflight schema",
    )
    compatibility = copy.deepcopy(dict(preflight))
    compatibility["schema"] = (
        "unblur_slam.fr2_xyz_paired_official_online_budget_221_preflight.v1"
    )
    _ORIGINAL_VALIDATE_PREFLIGHT(compatibility, cfg, arm)
    fixed = preflight.get("fixed_keyframe_ablation") or {}
    expected = {
        "conditional_on_frozen_evssm_baseline_schedule": True,
        "coordinate_domain": "dataset_source_index",
        "strict_exact_runtime_check": True,
        "runtime_baseline_artifact_dependency": False,
        "uses_ground_truth_poses": False,
        "poses_and_depths_estimated_independently_per_arm": True,
        "turtle_history_updates_on_all_source_frames": True,
    }
    for key, wanted in expected.items():
        BASE._expect(fixed.get(key), wanted, f"{arm} fixed preflight {key}")
    BASE._expect(
        tuple(int(value) for value in fixed.get("source_indices", ())),
        EXPECTED_FIXED_SOURCE_KEYFRAMES,
        f"{arm} preflight fixed schedule",
    )
    provenance = fixed.get("frozen_schedule_provenance") or {}
    BASE._expect(
        provenance.get("sha256"),
        FROZEN_BASELINE_VIDEO_SHA256,
        f"{arm} frozen schedule SHA-256",
    )
    BASE._expect(provenance.get("timestamps_only_read"), True, f"{arm} timestamps-only provenance")
    BASE._expect(provenance.get("poses_read_or_shared"), False, f"{arm} pose sharing")
    BASE._expect(provenance.get("depths_read_or_shared"), False, f"{arm} depth sharing")
    _gpu_guard(preflight, arm)


def _read_runtime(
    arm_root: Path, scene_root: Path, arm: str
) -> dict[str, float]:
    result = _ORIGINAL_READ_RUNTIME(arm_root, scene_root, arm)
    launcher = BASE._load_json(
        arm_root / "launcher_runtime.json", f"{arm} launcher runtime"
    )
    BASE._expect(
        tuple(int(value) for value in launcher.get("fixed_source_keyframes", ())),
        EXPECTED_FIXED_SOURCE_KEYFRAMES,
        f"{arm} launcher fixed schedule",
    )
    return result


def _read_keyframes(
    scene_root: Path, final_metrics: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    result = _ORIGINAL_READ_KEYFRAMES(scene_root, final_metrics, arm)
    BASE._expect(
        tuple(result["source_indices"]),
        EXPECTED_FIXED_SOURCE_KEYFRAMES,
        f"{arm} exact fixed keyframes",
    )
    result["matches_preregistered_fixed_schedule"] = True
    arm_root = scene_root.parent
    log = BASE._require_file(arm_root / "launch.log", f"{arm} launch log").read_text(
        encoding="utf-8", errors="replace"
    )
    markers = re.findall(
        r"TRACKING: fixed source-keyframe contract passed: (\[[^\n]+\])", log
    )
    BASE._expect(len(markers), 1, f"{arm} runtime exact-schedule marker count")
    try:
        logged = tuple(int(value) for value in json.loads(markers[0]))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ContractError(f"{arm} runtime fixed-schedule marker is invalid") from error
    BASE._expect(logged, EXPECTED_FIXED_SOURCE_KEYFRAMES, f"{arm} runtime schedule marker")
    return result


def _normalized_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(preflight))
    execution = normalized.get("execution")
    if isinstance(execution, dict):
        execution.pop("gpu_free_guard_before_launch", None)
    return normalized


def _validate_frozen_baseline_file() -> dict[str, Any]:
    actual_sha = BASE._sha256_file(FROZEN_BASELINE_VIDEO)
    BASE._expect(actual_sha, FROZEN_BASELINE_VIDEO_SHA256, "frozen baseline video SHA-256")
    try:
        with np.load(FROZEN_BASELINE_VIDEO, allow_pickle=False) as archive:
            timestamps = tuple(int(value) for value in archive["timestamps"].tolist())
    except (KeyError, OSError, ValueError) as error:
        raise ContractError("cannot verify frozen baseline timestamp provenance") from error
    BASE._expect(timestamps, EXPECTED_FIXED_SOURCE_KEYFRAMES, "frozen baseline timestamps")
    return {
        "path": str(FROZEN_BASELINE_VIDEO),
        "sha256": actual_sha,
        "timestamps_only_read": True,
        "source_indices": list(timestamps),
        "poses_read_or_shared": False,
        "depths_read_or_shared": False,
        "arm_runtime_dependency": False,
    }


def _validate_provenance(
    root: Path,
    configs: Mapping[str, Mapping[str, Any]],
    preflights: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = {
        arm: _normalized_preflight(preflights[arm])
        for arm in ("baseline", "turtle")
    }
    provenance = _ORIGINAL_VALIDATE_PROVENANCE(root, configs, normalized)
    provenance["fixed_schedule"] = _validate_frozen_baseline_file()
    provenance["gpu_free_guards_before_each_arm"] = {
        arm: _gpu_guard(preflights[arm], arm) for arm in ("baseline", "turtle")
    }
    provenance["baseline_outcome_data_shared_with_turtle"] = False
    provenance["poses_or_depths_shared_between_arms"] = False
    return provenance


# Install the strict extensions before delegating to the mature CPU parser.
BASE._validate_config = _validate_config
BASE._validate_preflight = _validate_preflight
BASE._read_runtime = _read_runtime
BASE._read_keyframes = _read_keyframes
BASE._validate_provenance = _validate_provenance


def build_report(root: Path) -> dict[str, Any]:
    report = BASE.build_report(root)
    report["schema"] = "unblur_slam.fr2_xyz_fixed_kf_frontend_ablation_221_report.v1"
    report["scope"].update(
        {
            "fixed_keyframe_schedule": list(EXPECTED_FIXED_SOURCE_KEYFRAMES),
            "fixed_keyframe_count": len(EXPECTED_FIXED_SOURCE_KEYFRAMES),
            "conditional_on_frozen_evssm_baseline_schedule": True,
            "selection_independent_comparison": False,
            "poses_and_depths_estimated_independently_per_arm": True,
        }
    )
    report["pair_contract"].update(
        {
            "fixed_keyframe_sets_exact_and_identical": True,
            "same_mapper_keyframe_count_and_iteration_budget": True,
            "turtle_motion_recovery_keyframes_disabled_by_fixed_policy": True,
        }
    )
    comparison = report["comparison"]
    if not comparison.get("keyframe_sets_identical"):
        raise ContractError("fixed-keyframe arms unexpectedly differ in keyframe set")
    comparison["fixed_schedule_exact_in_both_arms"] = True
    report["interpretation_notes"].extend(
        [
            "This ablation removes keyframe-count and per-keyframe mapper-budget differences: both arms map exactly the same eleven frozen source indices.",
            "The schedule is conditioned on a prior EVSSM baseline and is therefore not selection-independent; no prior pose, depth, metric, or Gaussian state is shared.",
            "Each arm estimates its own DROID poses/depths. Sharing EVSSM poses would leak a baseline outcome and would not measure the frontend's end-to-end tracking effect.",
            "TURTLE still advances official causal history on all 221 frames; fixed scheduling changes admission only, not its required streaming inference scope.",
        ]
    )
    return report


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = BASE._csv_rows(report)
    for row in rows:
        row["fixed_keyframe_schedule_exact"] = True
        row["baseline_conditioned_schedule"] = True
        row["poses_depths_shared_between_arms"] = False
    return rows


def write_report(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fixed_kf_paired_221_audit.json"
    csv_path = output_dir / "fixed_kf_paired_221_metrics.csv"
    for path in (json_path, csv_path):
        if path.exists() or path.is_symlink():
            raise ContractError(f"refusing to overwrite report output: {path}")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = _csv_rows(report)
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args.root)
        json_path, csv_path = write_report(report, args.output_dir)
        print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))
        return 0
    except ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
