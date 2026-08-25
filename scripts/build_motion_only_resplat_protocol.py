#!/usr/bin/env python3
"""Freeze DROID-motion keyframes and official ReSplat FPS contexts.

This is a CPU-only stage.  It accepts only a hash-verified ``FROZEN.json``
created before evaluation by ``run_fr2_turtle_motion_only_tracking.py``.  It
never opens TUM ground truth, clear-frame membership, depth, or image metrics.
The output directory is immutable and contains plain index files plus a
content-addressed protocol manifest for the later TURTLE/ReSplat CUDA stages.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/local/selection_independent/fr2_xyz_turtle_motion_only_resplat_221.json"
CONFIG_SCHEMA = "unblur_slam.motion_only_resplat_protocol_config.v1"
TRACKING_SCHEMA = "unblur_slam.frozen_motion_only_tracking.v1"
FREEZE_SCHEMA = "unblur_slam.frozen_estimate_gate.v1"
OUTPUT_SCHEMA = "unblur_slam.motion_only_resplat_protocol.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OFFICIAL_RESPLAT_ORIGIN = "https://github.com/cvg/resplat"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path | str, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return source, value


def _sha(value: object, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _verified_record(value: object, label: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain path and sha256")
    path = Path(str(value.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    expected = _sha(value.get("sha256"), f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return path, actual


def _git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot inspect Git repository: {repo}") from error
    return result.stdout.strip()


def _normalize_url(value: str) -> str:
    normalized = str(value).strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized[len("git@github.com:") :]
    return normalized.lower()


def _rigid_poses(value: object, count: int, label: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"{label} must be finite {count}x4x4, got {poses.shape}")
    if not np.allclose(poses[:, 3], np.asarray([0, 0, 0, 1]), atol=1e-5):
        raise ValueError(f"{label} contains invalid homogeneous rows")
    rotations = poses[:, :3, :3]
    if not np.allclose(
        np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=5e-4
    ) or np.any(np.linalg.det(rotations) < 0.999):
        raise ValueError(f"{label} rotations are not proper")
    return poses


def official_fps_indices(c2w: np.ndarray, num_context: int) -> list[int]:
    """Match official infer_colmap.py FPS using float32 camera positions."""

    c2w = np.asarray(c2w)
    count = len(c2w)
    if num_context <= 0:
        raise ValueError("num_context must be positive")
    if num_context >= count:
        return list(range(count))
    positions = np.asarray(c2w[:, :3, 3], dtype=np.float32)
    distance = np.ones(count, dtype=np.float32) * np.float32(1.0e10)
    barycenter = np.sum(positions, axis=0) / np.float32(count)
    farthest = int(np.argmax(np.sum((positions - barycenter) ** 2, axis=1)))
    centroids: list[int] = []
    for _ in range(num_context):
        centroids.append(farthest)
        squared = np.sum((positions - positions[farthest]) ** 2, axis=1)
        mask = squared < distance
        distance[mask] = squared[mask]
        farthest = int(np.argmax(distance))
    return sorted(centroids)


def _write_text(path: Path, values: Sequence[int]) -> None:
    path.write_text("".join(f"{int(value)}\n" for value in values), encoding="utf-8")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA!r}")
    source = config.get("source") or {}
    if (
        int(source.get("first", -1)) != 0
        or int(source.get("last", -1)) != 220
        or int(source.get("count", -1)) != 221
    ):
        raise ValueError("source interval must remain contiguous 0..220")
    selection = config.get("selection") or {}
    expected_selection = {
        "keyframe_provider": "droid_motion_filter_only",
        "context_provider": "official_cvg_resplat_fps",
        "num_context": 8,
        "target_provider": "remaining",
        "minimum_target_count": 1,
        "clear_gt_membership_used": False,
        "ground_truth_evaluation_deferred": True,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"selection.{key} must equal {expected!r}")
    excluded = config.get("excluded") or {}
    for key in (
        "legacy_mapping_resplat",
        "residual_replay",
        "causal_evssm",
        "clear_conditioned_42_frame_artifact_reuse",
        "26k_comparison",
    ):
        if excluded.get(key) is not False:
            raise ValueError(f"excluded.{key} must be false")


def load_frozen_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    tracking = config["tracking"]
    freeze_path, freeze = _load_json(tracking["freeze_marker"], "freeze marker")
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get(
        "selection_frozen_before_evaluation"
    ) is not True:
        raise ValueError("tracking output was not frozen before evaluation")
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("freeze marker has no artifact records")
    selection_path, selection_sha = _verified_record(
        artifacts.get("selection_manifest"), "selection manifest"
    )
    trajectory_path, trajectory_sha = _verified_record(
        artifacts.get("trajectory_npz"), "estimated trajectory"
    )
    video_path, video_sha = _verified_record(artifacts.get("video_npz"), "DROID video")
    _, selection = _load_json(selection_path, "selection manifest")
    if selection.get("schema") != TRACKING_SCHEMA:
        raise ValueError("selection manifest has the wrong schema")
    safety = selection.get("safety") or {}
    for key in (
        "ground_truth_pose_file_opened",
        "clear_gt_membership_file_opened",
        "image_metric_computed",
        "trajectory_metric_computed",
        "legacy_replay_used",
        "official_resplat_used",
    ):
        if safety.get(key) is not False:
            raise ValueError(f"pre-freeze selection safety flag {key} is not false")

    declared_keyframes = [
        int(value)
        for value in selection["keyframe_selection"]["source_indices"]
    ]
    with np.load(video_path, allow_pickle=False) as video:
        timestamps = np.asarray(video["timestamps"], dtype=np.float64).reshape(-1)
    rounded = np.rint(timestamps).astype(np.int64)
    if not np.allclose(timestamps, rounded, atol=1e-5):
        raise ValueError("DROID video timestamps are not source indices")
    keyframes = rounded.tolist()
    if keyframes != declared_keyframes:
        raise ValueError("DROID video and frozen selection keyframes disagree")
    if (
        len(keyframes) < int(config["selection"]["num_context"])
        + int(config["selection"]["minimum_target_count"])
        or keyframes != sorted(set(keyframes))
        or keyframes[0] != int(config["source"]["first"])
        or keyframes[-1] > int(config["source"]["last"])
    ):
        raise ValueError(f"invalid DROID motion-keyframe pool: {keyframes}")

    pose_key = str(tracking["pose_key"])
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        keys = set(trajectory.files)
        forbidden = {
            key
            for key in keys
            if key in {"traj_ref_poses", "traj_est_poses", "ground_truth"}
            or (
                key != "uses_ground_truth_pose"
                and (
                    "ground_truth" in key.lower()
                    or key.lower().startswith("gt_")
                )
            )
        }
        if forbidden:
            raise ValueError(
                "frozen selection trajectory contains forbidden reference arrays: "
                f"{sorted(forbidden)}"
            )
        pose_source = str(np.asarray(trajectory["pose_source"]).reshape(()).item())
        uses_gt = bool(
            np.asarray(trajectory["uses_ground_truth_pose"]).reshape(()).item()
        )
        reference_present = bool(
            np.asarray(trajectory["reference_pose_arrays_present"]).reshape(()).item()
        )
        source_indices = np.asarray(
            trajectory["traj_est_not_align_timestamps"], dtype=np.float64
        ).reshape(-1)
        poses = _rigid_poses(
            trajectory[pose_key], int(config["source"]["count"]), pose_key
        )
    if pose_source != tracking["pose_source"] or uses_gt or reference_present:
        raise ValueError("frozen trajectory does not prove non-GT estimated-pose provenance")
    if not np.array_equal(
        source_indices,
        np.arange(int(config["source"]["count"]), dtype=np.float64),
    ):
        raise ValueError("frozen trajectory/source-index binding is not exact")

    frames_csv = Path(str(config["source"]["frames_csv"])).expanduser().resolve()
    frames_sha = _sha(config["source"]["frames_csv_sha256"], "frames CSV hash")
    if not frames_csv.is_file() or sha256_file(frames_csv) != frames_sha:
        raise ValueError("frames CSV bytes changed")
    return {
        "freeze_path": freeze_path,
        "freeze_sha256": sha256_file(freeze_path),
        "selection_path": selection_path,
        "selection_sha256": selection_sha,
        "trajectory_path": trajectory_path,
        "trajectory_sha256": trajectory_sha,
        "video_path": video_path,
        "video_sha256": video_sha,
        "frames_csv": frames_csv,
        "frames_csv_sha256": frames_sha,
        "keyframes": keyframes,
        "poses": poses,
        "trajectory_keys": sorted(keys),
    }


def inspect_official_resplat(config: Mapping[str, Any]) -> dict[str, Any]:
    official = config["official_resplat"]
    repo = Path(str(official["repository"])).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"official ReSplat repository is missing: {repo}")
    origin = _git(repo, "remote", "get-url", "origin")
    commit = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if _normalize_url(origin) != _normalize_url(OFFICIAL_RESPLAT_ORIGIN):
        raise ValueError(f"ReSplat origin is not official: {origin}")
    if commit != official["commit"] or dirty:
        raise ValueError("official ReSplat checkout commit/cleanliness drifted")
    infer_path = repo / "scripts/infer_colmap.py"
    infer_sha = sha256_file(infer_path)
    if infer_sha != official["infer_colmap_sha256"]:
        raise ValueError("official ReSplat inference helper bytes changed")
    checkpoint = Path(str(official["checkpoint"])).expanduser().resolve()
    expected_checkpoint = _sha(
        official["checkpoint_sha256"], "official ReSplat checkpoint hash"
    )
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_checkpoint:
        raise ValueError("official ReSplat checkpoint bytes changed")
    return {
        "repository": str(repo),
        "origin": origin,
        "commit": commit,
        "tracked_worktree_clean": True,
        "infer_colmap": {"path": str(infer_path), "sha256": infer_sha},
        "checkpoint": {"path": str(checkpoint), "sha256": expected_checkpoint},
        "model_preset": official["model_preset"],
        "num_refine": int(official["num_refine"]),
    }


def build_protocol(config_path: Path | str = CONFIG) -> Path:
    config_source, config = _load_json(config_path, "protocol config")
    _validate_config(config)
    frozen = load_frozen_inputs(config)
    official = inspect_official_resplat(config)

    keyframes = frozen["keyframes"]
    keyframe_poses = frozen["poses"][keyframes]
    context_local = official_fps_indices(
        keyframe_poses, int(config["selection"]["num_context"])
    )
    context_set = set(context_local)
    target_local = [index for index in range(len(keyframes)) if index not in context_set]
    context_sources = [keyframes[index] for index in context_local]
    target_sources = [keyframes[index] for index in target_local]
    if len(target_sources) < int(config["selection"]["minimum_target_count"]):
        raise ValueError("motion-keyframe pool leaves too few ReSplat target views")

    destination = Path(str(config["outputs"]["protocol_dir"])).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite protocol output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    installed = False
    try:
        keyframe_file = staging / "motion_keyframes.txt"
        context_file = staging / "resplat_context_source_indices.txt"
        target_file = staging / "resplat_target_source_indices.txt"
        _write_text(keyframe_file, keyframes)
        _write_text(context_file, context_sources)
        _write_text(target_file, target_sources)
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": config["experiment_id"],
            "config": {"path": str(config_source), "sha256": sha256_file(config_source)},
            "frozen_tracking": {
                "freeze_marker": {
                    "path": str(frozen["freeze_path"]),
                    "sha256": frozen["freeze_sha256"],
                },
                "selection_manifest": {
                    "path": str(frozen["selection_path"]),
                    "sha256": frozen["selection_sha256"],
                },
                "trajectory": {
                    "path": str(frozen["trajectory_path"]),
                    "sha256": frozen["trajectory_sha256"],
                    "key": config["tracking"]["pose_key"],
                    "pose_source": config["tracking"]["pose_source"],
                    "reference_arrays_present": False,
                },
                "video": {
                    "path": str(frozen["video_path"]),
                    "sha256": frozen["video_sha256"],
                },
            },
            "source": {
                "frames_csv": str(frozen["frames_csv"]),
                "frames_csv_sha256": frozen["frames_csv_sha256"],
                "processed_source_indices": list(
                    range(int(config["source"]["first"]), int(config["source"]["last"]) + 1)
                ),
                "contiguous": True,
            },
            "keyframe_selection": {
                "provider": "DROID MotionFilter",
                "policy": "motion_filter_only",
                "source_indices": keyframes,
                "count": len(keyframes),
                "indices_file": {
                    "path": str(destination / keyframe_file.name),
                    "sha256": sha256_file(keyframe_file),
                },
            },
            "resplat_selection": {
                "provider": "official cvg/ReSplat infer_colmap.py",
                "strategy": "fps",
                "coordinate": "unaligned_DROID_c2w_translation_float32",
                "context_local_indices": context_local,
                "context_source_indices": context_sources,
                "context_count": len(context_sources),
                "context_indices_file": {
                    "path": str(destination / context_file.name),
                    "sha256": sha256_file(context_file),
                },
                "target_strategy": "remaining",
                "target_local_indices": target_local,
                "target_source_indices": target_sources,
                "target_count": len(target_sources),
                "target_indices_file": {
                    "path": str(destination / target_file.name),
                    "sha256": sha256_file(target_file),
                },
            },
            "official_resplat": official,
            "official_turtle": dict(config["official_turtle"]),
            "execution": dict(config["execution"]),
            "future_outputs": {
                key: str(Path(str(value)).expanduser().resolve())
                for key, value in config["outputs"].items()
                if key != "protocol_dir"
            },
            "excluded": dict(config["excluded"]),
            "safety": {
                "selection_frozen_before_evaluation": True,
                "clear_gt_membership_file_opened": False,
                "ground_truth_pose_file_opened": False,
                "ground_truth_image_opened": False,
                "depth_file_opened": False,
                "metric_computed": False,
                "old_clear_conditioned_artifact_read": False,
                "old_clear_conditioned_artifact_overwritten": False,
                "same_frozen_run_pose_source_for_keyframes_and_contexts": True,
                "official_fps_must_match_postflight": True,
            },
        }
        manifest_path = staging / "protocol_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = sha256_file(manifest_path)
        (staging / "protocol_manifest.sha256").write_text(
            f"{digest}  protocol_manifest.json\n", encoding="utf-8"
        )
        os.rename(staging, destination)
        installed = True
    finally:
        if not installed and staging.exists():
            import shutil

            shutil.rmtree(staging)
    return destination / "protocol_manifest.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_source, config = _load_json(args.config, "protocol config")
    _validate_config(config)
    frozen = load_frozen_inputs(config)
    official = inspect_official_resplat(config)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "config": str(config_source),
                    "freeze_sha256": frozen["freeze_sha256"],
                    "motion_keyframe_count": len(frozen["keyframes"]),
                    "official_resplat": official,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output = build_protocol(args.config)
    print(json.dumps({"status": "frozen", "manifest": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
