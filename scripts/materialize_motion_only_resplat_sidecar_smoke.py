#!/usr/bin/env python3
"""Freeze a real motion-only first-8-KF snapshot for queue-only sidecar smoke.

The selection is the first closed causal prefix: the first eight strictly
ordered DROID MotionFilter keyframes are captured when the ninth keyframe
arrives.  It never consumes a clear-frame list, GT pose, GT depth, or official
ReSplat FPS selection.  This prepares an *independent queue smoke*, not a claim
that the full SLAM process launched the child online.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.refinement.official_resplat_sidecar import (  # noqa: E402
    SidecarFrameInput,
    load_snapshot,
    materialize_closed_submap_snapshot,
    sha256_file,
)


SPEC_SCHEMA = "unblur_slam.motion_only_resplat_sidecar_smoke_input.v1"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _artifact(record: Mapping[str, Any], name: str) -> tuple[Path, str]:
    try:
        path = Path(str(record["path"])).expanduser().resolve()
        expected = str(record["sha256"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"FROZEN artifact {name} lacks path/SHA") from error
    if not path.is_file():
        raise FileNotFoundError(f"FROZEN artifact missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"FROZEN artifact SHA mismatch for {name}")
    return path, actual


def _scalar(archive: Any, key: str) -> Any:
    if key not in archive.files:
        raise ValueError(f"trajectory NPZ lacks required key {key!r}")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"trajectory NPZ key {key!r} must be scalar")
    return value.reshape(-1)[0].item()


def materialize(
    *, frozen_json: Path,
    turtle_manifest_path: Path,
    output_root: Path,
    expected_frozen_sha256: Optional[str],
    expected_turtle_sha256: Optional[str],
) -> Path:
    frozen_json = frozen_json.expanduser().resolve()
    turtle_manifest_path = turtle_manifest_path.expanduser().resolve()
    frozen_sha = sha256_file(frozen_json)
    turtle_sha = sha256_file(turtle_manifest_path)
    if expected_frozen_sha256 and frozen_sha != expected_frozen_sha256:
        raise ValueError("FROZEN.json SHA-256 mismatch")
    if expected_turtle_sha256 and turtle_sha != expected_turtle_sha256:
        raise ValueError("TURTLE manifest SHA-256 mismatch")

    frozen = _load_json(frozen_json)
    artifacts = frozen.get("artifacts") or {}
    selection_path, selection_sha = _artifact(
        artifacts.get("selection_manifest") or {}, "selection_manifest"
    )
    trajectory_path, trajectory_sha = _artifact(
        artifacts.get("trajectory_npz") or {}, "trajectory_npz"
    )
    selection = _load_json(selection_path)
    selection_safety = selection.get("safety") or {}
    if bool(selection_safety.get("clear_gt_membership_file_opened", True)):
        raise ValueError("motion-only selection opened clear-GT membership")
    if bool(selection_safety.get("ground_truth_pose_file_opened", True)):
        raise ValueError("motion-only selection opened a ground-truth pose file")
    if bool(selection_safety.get("reference_pose_array_created", True)):
        raise ValueError("motion-only selection created a reference pose array")
    if str((selection.get("keyframe_selection") or {}).get("policy", "")) != "motion_filter_only":
        raise ValueError("keyframe selection policy is not motion_filter_only")
    if bool(
        (selection.get("keyframe_selection") or {}).get(
            "predefined_tracking_anchor_list_loaded", True
        )
    ):
        raise ValueError("predefined tracking anchors entered motion-only selection")
    source_indices = (selection.get("keyframe_selection") or {}).get(
        "source_indices"
    )
    if not isinstance(source_indices, list) or len(source_indices) < 9:
        raise ValueError("motion-only selection needs at least 9 keyframes")
    source_indices = [int(value) for value in source_indices]
    if source_indices != sorted(set(source_indices)):
        raise ValueError("motion-only keyframe ids must be strictly increasing")
    selected = source_indices[:8]
    closure_trigger = source_indices[8]

    with np.load(trajectory_path, allow_pickle=False) as archive:
        if str(_scalar(archive, "pose_source")) != "droid_traj_est_not_align":
            raise ValueError("trajectory pose source is not unaligned DROID")
        if bool(_scalar(archive, "uses_ground_truth_pose")):
            raise ValueError("trajectory declares ground-truth pose use")
        if bool(_scalar(archive, "reference_pose_arrays_present")):
            raise ValueError("trajectory contains reference/GT pose arrays")
        if "traj_est_not_align" not in archive.files:
            raise ValueError("trajectory lacks traj_est_not_align")
        trajectory = np.asarray(archive["traj_est_not_align"], dtype=np.float64)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
        raise ValueError("traj_est_not_align must have shape Nx4x4")
    if max(selected) >= len(trajectory):
        raise ValueError("selected source index exceeds trajectory length")

    turtle = _load_json(turtle_manifest_path)
    if bool((turtle.get("source") or {}).get("uses_ground_truth_pose", True)):
        raise ValueError("TURTLE source declares ground-truth pose use")
    safety = turtle.get("safety") or {}
    if bool(safety.get("ground_truth_poses_used", True)):
        raise ValueError("TURTLE stream declares ground-truth pose use")
    emitted = [
        int(value)
        for value in (turtle.get("selection") or {}).get(
            "emitted_source_indices", []
        )
    ]
    if not set(selected).issubset(emitted):
        raise ValueError("TURTLE stream does not contain every first-8 motion keyframe")
    frame_by_source = {
        int(frame["source_index"]): frame for frame in turtle.get("frames", [])
    }
    if not set(selected).issubset(frame_by_source):
        raise ValueError("TURTLE frame records miss a first-8 motion keyframe")
    intrinsics = (turtle.get("camera") or {}).get("K")
    if intrinsics is None:
        raise ValueError("TURTLE manifest has no pixel-space K")

    frames = []
    input_records = []
    for ordinal, frame_id in enumerate(selected):
        output = frame_by_source[frame_id].get("output") or {}
        image_path = Path(str(output.get("path", ""))).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"TURTLE output missing: {image_path}")
        actual_image_sha = sha256_file(image_path)
        if actual_image_sha != str(output.get("sha256", "")):
            raise ValueError(f"TURTLE image SHA mismatch for frame {frame_id}")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        frames.append(
            SidecarFrameInput(
                frame_id=frame_id,
                sequence_ordinal=ordinal,
                c2w=trajectory[frame_id].tolist(),
                intrinsics_px=intrinsics,
                image=image,
            )
        )
        input_records.append(
            {
                "frame_id": frame_id,
                "sequence_ordinal": ordinal,
                "turtle_image_path": str(image_path),
                "turtle_image_sha256": actual_image_sha,
            }
        )

    provenance = {
        "schema": SPEC_SCHEMA,
        "independent_queue_smoke_not_full_slam_integration": True,
        "selection_rule": "first_8_motion_filter_keyframes_closed_on_9th",
        "selected_source_indices": selected,
        "closure_trigger_source_index_not_consumed": closure_trigger,
        "frozen_json": {"path": str(frozen_json), "sha256": frozen_sha},
        "selection_manifest": {
            "path": str(selection_path),
            "sha256": selection_sha,
        },
        "trajectory": {
            "path": str(trajectory_path),
            "sha256": trajectory_sha,
            "key": "traj_est_not_align",
            "pose_source": "droid_traj_est_not_align",
        },
        "turtle_manifest": {
            "path": str(turtle_manifest_path),
            "sha256": turtle_sha,
        },
        "frames": input_records,
        "uses_ground_truth_pose": False,
        "uses_clear_gt_membership": False,
        "uses_official_resplat_fps_selection": False,
    }
    snapshot_dir = materialize_closed_submap_snapshot(
        snapshots_root=output_root.expanduser().resolve() / "snapshots",
        submap_id=0,
        record_keyframe_ids=selected,
        frames=frames,
        closure_sequence_ordinal=8,
        pose_revision=8,
        integration_mode="independent_queue_smoke",
        selection_source="droid_motion_filter_first_closed_8kf_prefix",
        source_provenance=provenance,
    )
    snapshot = load_snapshot(snapshot_dir)
    print(
        json.dumps(
            {
                "snapshot_dir": str(snapshot_dir),
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "selected_source_indices": selected,
                "closure_trigger_source_index_not_consumed": closure_trigger,
                "integration_mode": "independent_queue_smoke",
            },
            sort_keys=True,
        )
    )
    return snapshot_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-json", type=Path, required=True)
    parser.add_argument("--turtle-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-frozen-sha256")
    parser.add_argument("--expected-turtle-sha256")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        materialize(
            frozen_json=args.frozen_json,
            turtle_manifest_path=args.turtle_manifest,
            output_root=args.output_root,
            expected_frozen_sha256=args.expected_frozen_sha256,
            expected_turtle_sha256=args.expected_turtle_sha256,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
