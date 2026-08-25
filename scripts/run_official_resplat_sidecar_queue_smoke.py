#!/usr/bin/env python3
"""Run and audit one independent-process official ReSplat sidecar queue smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.refinement.official_resplat_sidecar import (  # noqa: E402
    OfficialReSplatSidecarQueue,
    SidecarConfig,
    load_snapshot,
    sha256_file,
)


AUDIT_SCHEMA = "unblur_slam.official_resplat_sidecar_queue_smoke.v1"


def lexical_absolute_executable(path: Path | str) -> str:
    """Return an absolute executable path without resolving a venv symlink.

    CPython discovers ``pyvenv.cfg`` from the lexical ``.../bin/python`` path.
    ``Path.resolve()`` would turn it into the base interpreter and silently
    discard the official ReSplat environment's site-packages.
    """

    expanded = os.path.expanduser(os.fspath(path))
    return os.path.abspath(expanded)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(args: argparse.Namespace) -> Path:
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    snapshot = load_snapshot(snapshot_dir)
    if snapshot.get("integration_mode") != "independent_queue_smoke":
        raise ValueError("this launcher requires an independent_queue_smoke snapshot")
    expected_parent = (output_root / "snapshots").resolve()
    if snapshot_dir.parent != expected_parent:
        raise ValueError("snapshot must live in <output-root>/snapshots")
    config = SidecarConfig(
        enabled=True,
        output_root=str(output_root),
        python_executable=lexical_absolute_executable(args.resplat_python),
        runner_script=str(args.runner_script.expanduser().resolve()),
        resplat_repo=str(args.resplat_repo.expanduser().resolve()),
        checkpoint=str(args.checkpoint.expanduser().resolve()),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        cuda_visible_devices=str(args.cuda_visible_devices),
        process_device="cuda:0",
        max_runtime_seconds=float(args.max_runtime_seconds),
        final_drain_timeout_seconds=float(args.max_runtime_seconds),
        max_pose_revision_lag=0,
        max_pose_translation_drift=0.0,
        max_pose_rotation_drift_deg=0.0,
    )
    poses = {
        frame["frame_id"]: frame["c2w_opencv"] for frame in snapshot["frames"]
    }
    revision = int(snapshot["pose_revision"])
    queue = OfficialReSplatSidecarQueue(config)
    events = [queue.submit(snapshot_dir)]
    deadline = time.monotonic() + config.max_runtime_seconds + 5.0
    while (queue.active is not None or queue.pending) and time.monotonic() < deadline:
        events.extend(
            queue.poll(current_poses=poses, current_pose_revision=revision)
        )
        if queue.active is not None:
            time.sleep(0.10)
    if queue.active is not None or queue.pending:
        events.extend(
            queue.drain(current_pose_provider=lambda: (poses, revision))
        )
    published = [event for event in events if event.get("event") == "published"]
    if len(published) != 1:
        raise RuntimeError(f"sidecar queue smoke did not publish exactly once: {events}")
    published_root = Path(str(published[0]["path"])).resolve()
    result_path = published_root / "run_manifest.json"
    gate_path = published_root / "gate_decision.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not bool(gate.get("accepted", False)):
        raise RuntimeError("published sidecar has a rejected gate")
    if bool(result.get("active_map_merge_performed", True)):
        raise RuntimeError("queue smoke attempted active-map merge")
    if result.get("integration_mode") != "independent_queue_smoke":
        raise RuntimeError("queue smoke result lost its integration-mode disclosure")
    audit = {
        "schema": AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_class": "independent_queue_smoke_not_full_slam_integration",
        "snapshot": {
            "path": str(snapshot_dir),
            "id": snapshot["snapshot_id"],
            "sha256": snapshot["snapshot_sha256"],
            "selected_source_indices": [
                frame["frame_id"] for frame in snapshot["frames"]
            ],
            "past_only": True,
            "uses_ground_truth_pose": False,
            "uses_clear_gt_membership": False,
        },
        "queue": {
            "independent_process": True,
            "events": events,
            "cuda_visible_devices": str(args.cuda_visible_devices),
            "process_device": "cuda:0",
        },
        "frozen_gates": {
            "max_runtime_seconds": config.max_runtime_seconds,
            "max_pose_revision_lag": config.max_pose_revision_lag,
            "max_pose_translation_drift": config.max_pose_translation_drift,
            "max_pose_rotation_drift_deg": config.max_pose_rotation_drift_deg,
            "min_gaussian_count": config.min_gaussian_count,
            "max_gaussian_count": config.max_gaussian_count,
            "fixed_small8v_topology_count_also_required": True,
            "min_finite_fraction": config.min_finite_fraction,
            "max_p95_distance": config.max_p95_distance,
            "max_distance": config.max_distance,
            "max_p95_scale": config.max_p95_scale,
            "max_scale": config.max_scale,
            "max_quaternion_norm_deviation": config.max_quaternion_norm_deviation,
            "native_npz_geometry_recomputed_before_publish": True,
            "thresholds_changed_after_observing_result": False,
        },
        "published": {
            "path": str(published_root),
            "run_manifest_sha256": sha256_file(result_path),
            "gate_decision_sha256": sha256_file(gate_path),
            "gate_accepted": True,
            "active_map_merge_performed": False,
        },
        "limitations": {
            "full_slam_process_launched_sidecar": False,
            "native_sidecar_injected_into_active_map": False,
            "online_mapper_integration_covered_by_cpu_contract": True,
        },
        "launcher": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    audit_path = output_root / "queue_smoke_manifest.json"
    if audit_path.exists() or audit_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite smoke audit: {audit_path}")
    _atomic_json(audit_path, audit)
    print(json.dumps(audit["published"], sort_keys=True))
    return audit_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resplat-python", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--resplat-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--max-runtime-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        run(parse_args(argv))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
