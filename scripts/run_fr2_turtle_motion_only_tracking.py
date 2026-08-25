#!/usr/bin/env python3
"""Run a GT-deferred, selection-independent official-TURTLE/DROID smoke.

The normal repository TUM stream opens ``groundtruth.txt`` while it constructs
the dataset and ``MotionFilter`` normally loads a scene-specific tracking-anchor
list.  Both behaviours are inappropriate for a selection-independence audit.
This runner therefore:

* reads only RGB paths/timestamps from a content-pinned non-GT frames CSV;
* exposes identity placeholders where the tracking API requires a pose field;
* uses a scene alias that cannot resolve any predefined TUM anchor list;
* lets the DROID motion filter alone admit keyframes;
* performs final DROID BA and full trajectory filling without reference poses;
* freezes byte hashes before any separate evaluator may open ground truth.

It intentionally runs tracking only.  It does not run mapping, 26K refinement,
legacy replay, official ReSplat, or any image/trajectory metric.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thirdparty.glorie_slam import config as config_io  # noqa: E402
from src.slam import SLAM  # noqa: E402
from src.turtle_backend import (  # noqa: E402
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    validate_turtle_artifacts,
)
from src.utils.common import setup_seed  # noqa: E402
from src.utils.datasets import BaseDataset  # noqa: E402
from src.utils.Printer import FontColor  # noqa: E402


CONFIG = ROOT / "configs/local/selection_independent/fr2_xyz_turtle_motion_only_tracking_221.yaml"
DEFAULT_CONFIG = ROOT / "configs/unblur_slam.yaml"
SCHEMA = "unblur_slam.fr2_turtle_motion_only_tracking.v1"
MANIFEST_SCHEMA = "unblur_slam.frozen_motion_only_tracking.v1"
FREEZE_SCHEMA = "unblur_slam.frozen_estimate_gate.v1"
ALLOWED_POSE_SOURCE = "droid_traj_est_not_align"
NON_PROTOCOL_SCENE = "tum_f2_motion221_v1"
SOURCE_FIRST = 0
SOURCE_LAST = 220
SOURCE_COUNT = SOURCE_LAST - SOURCE_FIRST + 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_SCENE_TOKENS = (
    "freiburg1_desk",
    "fr1_desk",
    "freiburg2_xyz",
    "fr2_xyz",
    "freiburg3_office",
    "fr3_office",
)
MOTION_FILTER_SOURCE = ROOT / "thirdparty/glorie_slam/motion_filter.py"
TRACKER_SOURCE = ROOT / "src/tracker.py"
TURTLE_BACKEND_SOURCE = ROOT / "src/turtle_backend.py"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _scalar_false(value: object, label: str) -> None:
    if str(value).strip().lower() not in {"0", "false", "no"}:
        raise ValueError(f"{label} must explicitly be false")


def _pinned_file(path_value: object, hash_value: object, label: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    expected = str(hash_value or "").strip().lower()
    if not SHA256_RE.fullmatch(expected):
        raise ValueError(f"{label} configured SHA-256 is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return path


def load_config() -> dict[str, Any]:
    return config_io.load_config(str(CONFIG), str(DEFAULT_CONFIG))


def validate_config(cfg: Mapping[str, Any], *, verify_turtle_weights: bool) -> dict[str, Any]:
    metadata = cfg.get("selection_independent")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != SCHEMA:
        raise ValueError("selection_independent contract is missing or has the wrong schema")
    expected_top = (
        str(cfg.get("dataset", "")).lower(),
        str(cfg.get("scene", "")),
        int(cfg.get("stride", -1)),
        int(cfg.get("max_frames", -1)),
        int(cfg.get("setup_seed", -1)),
        str(cfg.get("device", "")),
        bool(cfg.get("only_tracking", False)),
    )
    lowered_scene = str(cfg.get("scene", "")).lower()
    matched = [token for token in FORBIDDEN_SCENE_TOKENS if token in lowered_scene]
    if matched:
        raise ValueError(
            "scene alias would activate a predefined MotionFilter anchor list: "
            f"{matched}"
        )
    if expected_top != (
        "tumrgbd", NON_PROTOCOL_SCENE, 1, SOURCE_COUNT, 43, "cuda:0", True
    ):
        raise ValueError(f"motion-only tracking contract drifted: {expected_top}")

    expected_metadata = {
        "source_first": SOURCE_FIRST,
        "source_last": SOURCE_LAST,
        "expected_source_count": SOURCE_COUNT,
        "keyframe_policy": "droid_motion_filter_only",
        "motion_filter_threshold_px": 2.5,
        "predefined_tracking_anchors": False,
        "clear_gt_membership_used": False,
        "ground_truth_pose_read_before_freeze": False,
        "evaluation_deferred_until_frozen": True,
        "source_scene_alias_is_non_protocol": True,
        "physical_gpu": 1,
        "cuda_visible_devices": "1",
        "process_device": "cuda:0",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"selection_independent.{key} must equal {expected!r}")

    source_hashes = {
        "motion_filter": (
            MOTION_FILTER_SOURCE,
            metadata.get("motion_filter_source_sha256"),
        ),
        "tracker": (TRACKER_SOURCE, metadata.get("tracker_source_sha256")),
        "turtle_backend": (
            TURTLE_BACKEND_SOURCE,
            metadata.get("turtle_backend_source_sha256"),
        ),
    }
    source_records: dict[str, Any] = {}
    for label, (path, expected) in source_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{label} source changed: expected {expected}, got {actual}"
            )
        source_records[label] = {"path": str(path), "sha256": actual}

    data = cfg.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("data config must be a mapping")
    frames_csv = _pinned_file(
        data.get("frames_csv"), data.get("frames_csv_sha256"), "frames CSV"
    )
    tracking = cfg.get("tracking")
    if not isinstance(tracking, Mapping):
        raise ValueError("tracking config must be a mapping")
    droid = _pinned_file(
        tracking.get("pretrained"), tracking.get("pretrained_sha256"), "DROID checkpoint"
    )
    if float((tracking.get("motion_filter") or {}).get("thresh", -1.0)) != 2.5:
        raise ValueError("tracking.motion_filter.thresh must remain 2.5 pixels")
    if not bool((tracking.get("backend") or {}).get("final_ba", False)):
        raise ValueError("selection run requires final DROID BA")
    mono = cfg.get("mono_prior")
    if not isinstance(mono, Mapping) or mono.get("predict_online") is not True:
        raise ValueError("selection run requires online mono depth")
    omnidata = _pinned_file(
        mono.get("depth_pretrained"),
        mono.get("depth_pretrained_sha256"),
        "Omnidata checkpoint",
    )

    deblur = cfg.get("deblur")
    if not isinstance(deblur, Mapping):
        raise ValueError("deblur config must be a mapping")
    expected_deblur = {
        "frontend": "turtle_streaming",
        "stream_every_frame": True,
        "stream_apply_to_tracking": True,
        "stream_replace_sharp": False,
        "turtle_inference_precision": "fp16",
        "turtle_repo_commit": PINNED_TURTLE_COMMIT,
        "turtle_config_sha256": PINNED_TURTLE_CONFIG_SHA256,
        "turtle_checkpoint_sha256": PINNED_TURTLE_CHECKPOINT_SHA256,
    }
    for key, expected in expected_deblur.items():
        if deblur.get(key) != expected:
            raise ValueError(f"deblur.{key} must equal {expected!r}")
    if float(deblur.get("stream_min_laplacian_gain", -1.0)) != 0.02:
        raise ValueError("TURTLE Laplacian gate must remain 0.02")
    turtle_artifacts = validate_turtle_artifacts(
        deblur, load_weights=verify_turtle_weights
    )
    if (
        turtle_artifacts.commit != PINNED_TURTLE_COMMIT
        or turtle_artifacts.architecture_sha256 != PINNED_TURTLE_ARCH_SHA256
        or turtle_artifacts.config_sha256 != PINNED_TURTLE_CONFIG_SHA256
        or turtle_artifacts.checkpoint_sha256 != PINNED_TURTLE_CHECKPOINT_SHA256
    ):
        raise ValueError("official TURTLE artifact pins drifted")

    mapping = cfg.get("mapping") or {}
    replay = mapping.get("resplat") or {}
    if (
        int(mapping.get("final_refine_iters", -1)) != 0
        or replay.get("enabled") is not False
        or replay.get("online_enabled") is not False
        or int(replay.get("extra_iters", -1)) != 0
    ):
        raise ValueError("mapping and legacy replay must remain disabled")
    if bool((cfg.get("framecrafter") or {}).get("enabled", False)):
        raise ValueError("FrameCrafter must be disabled")
    if bool((cfg.get("submaps") or {}).get("enabled", False)):
        raise ValueError("custom submaps must be disabled")
    evaluation = cfg.get("evaluation") or {}
    if evaluation.get("deferred_until_frozen") is not True:
        raise ValueError("evaluation must be explicitly deferred until freeze")

    output_root = Path(str(data.get("output", ""))).expanduser().resolve()
    if str(output_root).startswith(str(ROOT)):
        raise ValueError("large run outputs must live outside the repository filesystem")
    return {
        "frames_csv": frames_csv,
        "droid": droid,
        "omnidata": omnidata,
        "output_root": output_root,
        "save_dir": output_root / NON_PROTOCOL_SCENE,
        "sources": source_records,
        "turtle": {
            "repository": str(turtle_artifacts.repo),
            "commit": turtle_artifacts.commit,
            "architecture_sha256": turtle_artifacts.architecture_sha256,
            "config_sha256": turtle_artifacts.config_sha256,
            "checkpoint_sha256": turtle_artifacts.checkpoint_sha256,
            "checkpoint_kind": turtle_artifacts.checkpoint_metadata.get("kind"),
        },
    }


class SelectionOnlyRgbStream(BaseDataset):
    """RGB-only frame stream whose pose placeholders contain no reference data."""

    def __init__(self, cfg: Mapping[str, Any], frames_csv: Path):
        super().__init__(cfg, device=str(cfg["device"]))
        with frames_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            required = {
                "index",
                "timestamp",
                "rgb_path",
                "pose_source",
                "uses_ground_truth_pose",
            }
            missing = required - fields
            if missing:
                raise ValueError(f"frames CSV is missing columns: {sorted(missing)}")
            rows = list(reader)

        rows_by_index: dict[int, Mapping[str, str]] = {}
        for row_number, row in enumerate(rows, 2):
            try:
                source_index = int(str(row["index"]).strip())
            except ValueError as error:
                raise ValueError(f"invalid source index at CSV row {row_number}") from error
            if source_index < 0 or source_index in rows_by_index:
                raise ValueError(f"duplicate/invalid source index {source_index}")
            rows_by_index[source_index] = row

        wanted = list(range(SOURCE_FIRST, SOURCE_LAST + 1))
        missing_indices = [index for index in wanted if index not in rows_by_index]
        if missing_indices:
            raise ValueError(f"frames CSV is missing source indices {missing_indices[:16]}")

        color_paths: list[str] = []
        image_timestamps: list[float] = []
        prior_timestamp: Optional[float] = None
        for source_index in wanted:
            row = rows_by_index[source_index]
            _scalar_false(
                row["uses_ground_truth_pose"],
                f"frames CSV source {source_index} uses_ground_truth_pose",
            )
            pose_source = str(row["pose_source"]).strip()
            if pose_source != ALLOWED_POSE_SOURCE:
                raise ValueError(
                    f"frames CSV source {source_index} has unexpected pose_source "
                    f"{pose_source!r}"
                )
            timestamp = float(row["timestamp"])
            if not math.isfinite(timestamp) or (
                prior_timestamp is not None and timestamp <= prior_timestamp
            ):
                raise ValueError("frames CSV timestamps must be finite and increasing")
            prior_timestamp = timestamp
            image_path = Path(str(row["rgb_path"])).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"RGB source does not exist: {image_path}")
            color_paths.append(str(image_path))
            image_timestamps.append(timestamp)

        self.color_paths = color_paths
        self.gt_paths = color_paths  # API placeholder: the observed RGB itself, never clear GT.
        # ``BaseDataset.__getitem__`` expects concrete dataset subclasses to
        # define this field.  Pin it locally instead of inheriting any
        # clear-initialization behaviour from a TUM loader (which this stream
        # intentionally never constructs).
        self.clear_init = False
        self.depth_paths = None
        self.is_depth = False
        self.n_img = len(color_paths)
        self.original_frame_count = self.n_img
        self.image_timestamps = np.asarray(image_timestamps, dtype=np.float64)
        identity_knots = np.repeat(
            np.eye(4, dtype=np.float32)[None, None], self.n_img, axis=0
        )
        self.poses = np.repeat(identity_knots, self.num_control_knots, axis=1)
        self.w2c_first_pose = np.eye(4, dtype=np.float32)
        self.frame_metadata = [
            {
                "augmented_index": index,
                "source_index": index,
                "synthetic": False,
                "eval": False,
                "confidence": 1.0,
                "pose_placeholder": "identity_not_reference",
            }
            for index in wanted
        ]


def _rigid_poses(value: np.ndarray, label: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) != SOURCE_COUNT:
        raise ValueError(f"{label} must be {SOURCE_COUNT}x4x4, got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(poses[:, 3], np.asarray([0, 0, 0, 1]), atol=1e-5):
        raise ValueError(f"{label} has invalid homogeneous rows")
    rotations = poses[:, :3, :3]
    if not np.allclose(
        np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=5e-4
    ):
        raise ValueError(f"{label} has non-orthonormal rotations")
    if np.any(np.linalg.det(rotations) < 0.999):
        raise ValueError(f"{label} contains left-handed rotations")
    return poses


class SelectionOnlySLAM(SLAM):
    """SLAM termination that freezes estimates and performs no evaluation."""

    def terminate(self) -> None:
        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.backend()

        save_dir = Path(self.save_dir)
        video_path = save_dir / "video.npz"
        self.video.save_video(str(video_path))

        self.printer.print(
            "Filling unaligned full trajectory without reference poses...",
            FontColor.INFO,
        )
        trajectory_inverse = self.traj_filler(self.stream)
        full_c2w = trajectory_inverse.inv().matrix().data.cpu().numpy()
        keyframe_count = int(self.video.counter.value)
        keyframe_source_indices = (
            self.video.timestamp[:keyframe_count].detach().cpu().numpy()
        )
        rounded = np.rint(keyframe_source_indices).astype(np.int64)
        if not np.allclose(keyframe_source_indices, rounded, atol=1e-5):
            raise ValueError("DROID keyframe timestamps left source-index space")
        if (
            len(rounded) < 2
            or rounded[0] != SOURCE_FIRST
            or np.any(np.diff(rounded) <= 0)
            or rounded[-1] > SOURCE_LAST
        ):
            raise ValueError(f"invalid motion-only DROID keyframe sequence: {rounded.tolist()}")
        keyframe_c2w = (
            self.video.poses[:keyframe_count].clone().detach()
        )
        from lietorch import SE3

        keyframe_c2w = SE3(keyframe_c2w).inv().matrix().cpu().numpy()
        full_c2w[rounded] = keyframe_c2w
        full_c2w = _rigid_poses(full_c2w, "unaligned DROID trajectory")

        trajectory_path = save_dir / "trajectory_estimated_unaligned.npz"
        np.savez(
            trajectory_path,
            traj_est_not_align=full_c2w,
            traj_est_not_align_timestamps=np.arange(SOURCE_COUNT, dtype=np.float64),
            traj_est_not_align_eval_mask=np.zeros(SOURCE_COUNT, dtype=np.bool_),
            pose_source=np.asarray(ALLOWED_POSE_SOURCE),
            uses_ground_truth_pose=np.asarray(False),
            selection_phase=np.asarray(True),
            reference_pose_arrays_present=np.asarray(False),
        )

        selection_path = save_dir / "selection_manifest.json"
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "selection_before_evaluation",
            "source_interval": {
                "first": SOURCE_FIRST,
                "last": SOURCE_LAST,
                "count": SOURCE_COUNT,
                "contiguous": True,
            },
            "keyframe_selection": {
                "provider": "DROID MotionFilter",
                "policy": "motion_filter_only",
                "motion_filter_threshold_px": 2.5,
                "source_indices": rounded.tolist(),
                "count": len(rounded),
                "predefined_tracking_anchor_list_loaded": False,
                "scene_alias": NON_PROTOCOL_SCENE,
            },
            "estimated_outputs": {
                "video_npz": {
                    "path": str(video_path),
                    "sha256": sha256_file(video_path),
                },
                "trajectory_npz": {
                    "path": str(trajectory_path),
                    "sha256": sha256_file(trajectory_path),
                    "key": "traj_est_not_align",
                    "pose_source": ALLOWED_POSE_SOURCE,
                    "uses_ground_truth_pose": False,
                },
            },
            "deblur": {
                "provider": "official_ascend_research_turtle_gopro",
                "stream_every_source_frame": True,
                "persistent_kv": True,
                "inference_precision": self.cfg["deblur"][
                    "turtle_inference_precision"
                ],
                "checkpoint_sha256": self.cfg["deblur"][
                    "turtle_checkpoint_sha256"
                ],
            },
            "safety": {
                "ground_truth_pose_file_opened": False,
                "clear_gt_membership_file_opened": False,
                "reference_pose_array_created": False,
                "image_metric_computed": False,
                "trajectory_metric_computed": False,
                "legacy_replay_used": False,
                "official_resplat_used": False,
                "ready_to_freeze": True,
            },
        }
        _atomic_json(selection_path, manifest)


def _preflight_payload(cfg: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "preflight_only": True,
        "config": {"path": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "source": {
            "frames_csv": str(audit["frames_csv"]),
            "sha256": sha256_file(audit["frames_csv"]),
            "columns_used": [
                "index",
                "timestamp",
                "rgb_path",
                "pose_source",
                "uses_ground_truth_pose",
            ],
            "pose_columns_consumed": False,
            "depth_paths_consumed": False,
        },
        "selection": {
            "scene_alias": cfg["scene"],
            "policy": "droid_motion_filter_only",
            "motion_filter_threshold_px": 2.5,
            "predefined_tracking_anchors": False,
            "source_first": SOURCE_FIRST,
            "source_last": SOURCE_LAST,
        },
        "official_turtle": audit["turtle"],
        "checkpoints": {
            "droid": {"path": str(audit["droid"]), "sha256": sha256_file(audit["droid"])},
            "omnidata": {
                "path": str(audit["omnidata"]),
                "sha256": sha256_file(audit["omnidata"]),
            },
        },
        "sources": audit["sources"],
        "execution": {
            "physical_gpu": 1,
            "CUDA_VISIBLE_DEVICES": "1",
            "process_device": "cuda:0",
            "turtle_inference_precision": "fp16",
        },
        "outputs": {
            "root": str(audit["output_root"]),
            "save_dir": str(audit["save_dir"]),
            "overwrite_allowed": False,
        },
        "safety": {
            "repository_tum_loader_used": False,
            "groundtruth_txt_opened": False,
            "clear_gt_membership_opened": False,
            "evaluation_deferred_until_frozen": True,
            "mapping_used": False,
            "official_resplat_used": False,
        },
    }


def freeze_outputs(cfg: Mapping[str, Any], audit: Mapping[str, Any]) -> Path:
    save_dir = Path(audit["save_dir"])
    selection_path = save_dir / "selection_manifest.json"
    trajectory_path = save_dir / "trajectory_estimated_unaligned.npz"
    video_path = save_dir / "video.npz"
    for path in (selection_path, trajectory_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(f"required estimated output is missing: {path}")
    with selection_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("selection manifest schema drifted")
    safety = selection.get("safety") or {}
    if safety.get("ready_to_freeze") is not True or any(
        safety.get(key) is not False
        for key in (
            "ground_truth_pose_file_opened",
            "clear_gt_membership_file_opened",
            "image_metric_computed",
            "trajectory_metric_computed",
        )
    ):
        raise ValueError("selection manifest does not satisfy the pre-evaluation gate")
    freeze_path = save_dir / "FROZEN.json"
    if freeze_path.exists():
        raise FileExistsError(f"freeze marker already exists: {freeze_path}")
    payload = {
        "schema": FREEZE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_frozen_before_evaluation": True,
        "config": {"path": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "artifacts": {
            "selection_manifest": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
            },
            "trajectory_npz": {
                "path": str(trajectory_path),
                "sha256": sha256_file(trajectory_path),
            },
            "video_npz": {"path": str(video_path), "sha256": sha256_file(video_path)},
        },
        "next_phase_may_read_reference_data": True,
    }
    _atomic_json(freeze_path, payload)
    return freeze_path


def run_tracking(cfg: dict[str, Any], audit: Mapping[str, Any]) -> Path:
    output_root = Path(audit["output_root"])
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite selection output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    preflight_path = output_root / "preflight.json"
    _atomic_json(preflight_path, _preflight_payload(cfg, audit))

    stream = SelectionOnlyRgbStream(cfg, Path(audit["frames_csv"]))
    slam = SelectionOnlySLAM(cfg, stream)
    started = time.perf_counter()
    slam.run()
    elapsed = time.perf_counter() - started
    freeze_path = freeze_outputs(cfg, audit)
    _atomic_json(
        output_root / "run_complete.json",
        {
            "schema": SCHEMA,
            "wall_seconds": elapsed,
            "freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
            "exit_status": "success",
        },
    )
    return freeze_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="run the CUDA tracking smoke")
    mode.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify and freeze an already completed estimate directory",
    )
    parser.add_argument(
        "--verify-turtle-weights",
        action="store_true",
        help="also strict-load the 225 MiB TURTLE checkpoint during CPU preflight",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    audit = validate_config(cfg, verify_turtle_weights=args.verify_turtle_weights)
    if args.run:
        freeze = run_tracking(cfg, audit)
        print(json.dumps({"status": "complete", "freeze": str(freeze)}, indent=2))
        return 0
    if args.verify_existing:
        freeze = freeze_outputs(cfg, audit)
        print(json.dumps({"status": "frozen", "freeze": str(freeze)}, indent=2))
        return 0
    print(json.dumps(_preflight_payload(cfg, audit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    setup_seed(43)
    raise SystemExit(main())
