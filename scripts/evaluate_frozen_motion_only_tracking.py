#!/usr/bin/env python3
"""Evaluate a motion-only DROID estimate strictly after its freeze gate.

This is the only selection-independent script allowed to open TUM
``groundtruth.txt``.  It first verifies ``FROZEN.json`` and every estimated
artifact hash, then computes full and keyframe ATE into a new immutable output
directory.  It never opens the clear-frame membership files and never mutates
the frozen selection artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fr2_turtle_motion_only_tracking import (  # noqa: E402
    FREEZE_SCHEMA,
    MANIFEST_SCHEMA,
    NON_PROTOCOL_SCENE,
    SOURCE_COUNT,
    load_config,
    sha256_file,
    validate_config,
)
from src.utils.datasets import TUM_RGB  # noqa: E402


EVAL_SCHEMA = "unblur_slam.postfreeze_motion_only_tracking_eval.v1"
DEFAULT_OUTPUT = Path(
    "/srv/szha0669/unblur-slam/selection_independent/"
    "fr2_xyz_turtle_motion_only_tracking_221_v2_postfreeze_eval"
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def verify_freeze(cfg: Mapping[str, Any]) -> dict[str, Any]:
    save_dir = (
        Path(str(cfg["data"]["output"])).expanduser().resolve()
        / NON_PROTOCOL_SCENE
    )
    freeze_path = save_dir / "FROZEN.json"
    freeze = _load_json(freeze_path, "freeze marker")
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("selection_frozen_before_evaluation") is not True
    ):
        raise ValueError("estimate was not frozen before evaluation")
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("freeze marker has no artifact records")
    verified: dict[str, Any] = {}
    for label in ("selection_manifest", "trajectory_npz", "video_npz"):
        record = artifacts.get(label)
        if not isinstance(record, Mapping):
            raise ValueError(f"freeze artifact {label} is missing")
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        expected = str(record.get("sha256", ""))
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen {label} SHA-256 mismatch")
        verified[label] = {"path": path, "sha256": actual}
    selection = _load_json(
        verified["selection_manifest"]["path"], "selection manifest"
    )
    if selection.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("selection manifest schema drifted")
    safety = selection.get("safety") or {}
    if any(
        safety.get(key) is not False
        for key in (
            "ground_truth_pose_file_opened",
            "clear_gt_membership_file_opened",
            "image_metric_computed",
            "trajectory_metric_computed",
        )
    ):
        raise ValueError("selection phase already accessed evaluation data")
    return {
        "save_dir": save_dir,
        "freeze_path": freeze_path,
        "freeze_sha256": sha256_file(freeze_path),
        "artifacts": verified,
        "selection": selection,
    }


def _pose_array(value: object, label: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"{label} must be finite Nx4x4, got {poses.shape}")
    return poses


def _ate(estimates: np.ndarray, references: np.ndarray) -> dict[str, Any]:
    from evo.core import metrics
    from evo.core.trajectory import PoseTrajectory3D

    estimates = _pose_array(estimates, "estimated poses")
    references = _pose_array(references, "reference poses")
    if len(estimates) != len(references) or len(estimates) < 2:
        raise ValueError("ATE requires paired estimate/reference trajectories")
    timestamps = np.arange(len(estimates), dtype=np.float64)
    estimate_path = PoseTrajectory3D(poses_se3=estimates.copy(), timestamps=timestamps)
    reference_path = PoseTrajectory3D(poses_se3=references.copy(), timestamps=timestamps)
    rotation, translation, scale = estimate_path.align(
        reference_path, correct_scale=True
    )
    metric = metrics.APE(metrics.PoseRelation.translation_part)
    metric.process_data((reference_path, estimate_path))
    statistics = {
        key: float(value) for key, value in metric.get_all_statistics().items()
    }
    return {
        "count": len(estimates),
        "alignment": {
            "correct_scale": True,
            "scale": float(scale),
            "rotation": np.asarray(rotation).tolist(),
            "translation": np.asarray(translation).tolist(),
        },
        "ape_translation": statistics,
    }


def evaluate(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    cfg = load_config()
    validate_config(cfg, verify_turtle_weights=False)
    frozen = verify_freeze(cfg)
    destination = output_dir.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluation: {destination}")

    # The reference file is resolved only after all checks above succeed.
    eval_cfg = dict(cfg)
    eval_cfg["scene"] = "freiburg2_xyz"
    eval_cfg["device"] = "cpu"
    eval_cfg["only_tracking"] = True
    stream = TUM_RGB(eval_cfg, device="cpu")
    if len(stream) != SOURCE_COUNT:
        raise ValueError(f"post-freeze TUM reference length drifted: {len(stream)}")
    references = np.asarray(stream.poses, dtype=np.float64)
    if references.ndim == 4:
        references = references[:, 0]
    references = _pose_array(references, "TUM reference poses")

    trajectory_path = frozen["artifacts"]["trajectory_npz"]["path"]
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        full_estimates = _pose_array(
            trajectory["traj_est_not_align"], "full estimated trajectory"
        )
        if bool(
            np.asarray(trajectory["uses_ground_truth_pose"]).reshape(()).item()
        ):
            raise ValueError("frozen estimate unexpectedly declares GT pose use")
    video_path = frozen["artifacts"]["video_npz"]["path"]
    with np.load(video_path, allow_pickle=False) as video:
        # DROID's internal ``video.poses`` are world-to-camera SE(3) vectors.
        # Use the already-frozen, final-BA camera-to-world trajectory for ATE;
        # the video archive supplies only the immutable membership/timestamps.
        video_pose_count = len(_pose_array(video["poses"], "DROID video poses"))
        timestamps = np.asarray(video["timestamps"], dtype=np.float64).reshape(-1)
    keyframe_indices = np.rint(timestamps).astype(np.int64)
    if (
        not np.allclose(timestamps, keyframe_indices, atol=1e-5)
        or video_pose_count != len(keyframe_indices)
        or len(keyframe_indices) < 2
        or np.any(keyframe_indices < 0)
        or np.any(keyframe_indices >= len(full_estimates))
        or np.any(np.diff(keyframe_indices) <= 0)
    ):
        raise ValueError("keyframe timestamps are not valid source indices")
    keyframe_estimates = full_estimates[keyframe_indices]

    groundtruth_path = (
        Path(str(cfg["data"]["dataset_root"])).expanduser().resolve()
        / str(cfg["data"]["input_folder"])
        / "groundtruth.txt"
    )
    if not groundtruth_path.is_file():
        raise FileNotFoundError(groundtruth_path)
    metrics_payload = {
        "full": _ate(full_estimates, references),
        "keyframes": _ate(keyframe_estimates, references[keyframe_indices]),
    }

    # Prove evaluation did not mutate the frozen bytes.
    for label, record in frozen["artifacts"].items():
        if sha256_file(record["path"]) != record["sha256"]:
            raise RuntimeError(f"evaluation mutated frozen artifact {label}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    installed = False
    try:
        report = {
            "schema": EVAL_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "evaluation_after_frozen_selection",
            "freeze": {
                "path": str(frozen["freeze_path"]),
                "sha256": frozen["freeze_sha256"],
                "verified_before_reference_open": True,
            },
            "reference": {
                "path": str(groundtruth_path),
                "sha256": sha256_file(groundtruth_path),
                "opened_after_freeze_verification": True,
                "clear_gt_membership_file_opened": False,
            },
            "selection": frozen["selection"]["keyframe_selection"],
            "metrics": metrics_payload,
            "safety": {
                "frozen_estimates_mutated": False,
                "ground_truth_pose_opened_before_freeze": False,
                "ground_truth_pose_opened_after_freeze": True,
                "clear_gt_membership_used_for_selection": False,
            },
        }
        report_path = staging / "evaluation.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        digest = sha256_file(report_path)
        (staging / "evaluation.sha256").write_text(
            f"{digest}  evaluation.json\n", encoding="utf-8"
        )
        os.rename(staging, destination)
        installed = True
    finally:
        if not installed and staging.exists():
            import shutil

            shutil.rmtree(staging)
    return destination / "evaluation.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-freeze-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    if args.verify_freeze_only:
        frozen = verify_freeze(cfg)
        print(
            json.dumps(
                {
                    "status": "frozen",
                    "freeze": str(frozen["freeze_path"]),
                    "sha256": frozen["freeze_sha256"],
                },
                indent=2,
            )
        )
        return 0
    report = evaluate(args.output_dir)
    print(json.dumps({"status": "evaluated", "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
