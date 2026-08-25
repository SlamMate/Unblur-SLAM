#!/usr/bin/env python3
"""CPU-only contracts for the official closed-submap ReSplat sidecar."""

from __future__ import annotations

import json
import hashlib
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.refinement.official_resplat_sidecar import (
    OFFICIAL_PRESET,
    OfficialReSplatSidecarQueue,
    RESULT_SCHEMA,
    SidecarConfig,
    SidecarFrameInput,
    UnsupportedActiveMapMerge,
    active_map_merge_assessment,
    evaluate_result,
    evaluate_staleness,
    load_snapshot,
    materialize_closed_submap_snapshot,
    pose_hash,
    reject_active_map_merge,
)
from scripts.run_official_resplat_sidecar import run_official_refinement_core
from scripts.materialize_motion_only_resplat_sidecar_smoke import (
    materialize as materialize_motion_only_smoke,
)
from scripts.run_official_resplat_sidecar_queue_smoke import (
    lexical_absolute_executable,
)


def _pose(x: float = 0.0) -> list[list[float]]:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(x)
    return pose.tolist()


def _frame(frame_id: int, ordinal: int) -> SidecarFrameInput:
    image = np.full((12, 16, 3), (frame_id * 13) % 255, dtype=np.uint8)
    return SidecarFrameInput(
        frame_id=frame_id,
        sequence_ordinal=ordinal,
        c2w=_pose(0.01 * ordinal),
        intrinsics_px=((10.0, 0.0, 8.0), (0.0, 11.0, 6.0), (0.0, 0.0, 1.0)),
        image=image,
    )


def _snapshot(root: Path, *, count: int = 10) -> Path:
    frames = [_frame(index, index) for index in range(count)]
    return materialize_closed_submap_snapshot(
        snapshots_root=root,
        submap_id=3,
        record_keyframe_ids=list(range(count)),
        frames=frames,
        closure_sequence_ordinal=count - 1,
        pose_revision=17,
    )


def _config(root: Path, **overrides) -> SidecarConfig:
    values = {
        "enabled": False,
        "output_root": str(root),
        "max_runtime_seconds": 30.0,
    }
    values.update(overrides)
    return SidecarConfig(**values)


def _valid_result(snapshot: dict, checkpoint_sha256: str = "") -> dict:
    return {
        "schema": RESULT_SCHEMA,
        "artifact_class": "native_official_resplat_closed_submap_sidecar",
        "integration_mode": snapshot["integration_mode"],
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "source_pose_revision": snapshot["pose_revision"],
        "source_pose_hashes": [frame["pose_hash"] for frame in snapshot["frames"]],
        "past_only": True,
        "future_views_used": False,
        "ground_truth_used": False,
        "image_shape_hw": [320, 448],
        "official_resplat": {
            "model_preset": OFFICIAL_PRESET,
            "num_context": 8,
            "num_refine": 4,
            "repository": {
                "expected_origin": "https://github.com/cvg/resplat",
                "tracked_worktree_clean": True,
            },
            "checkpoint": {"sha256": checkpoint_sha256},
        },
        "execution_contract": {
            "encoder_forward_calls": 1,
            "forward_update_calls": 1,
            "init_object_passed_directly": True,
        },
        "geometry": {
            "gaussian_count": 71680,
            "finite_fraction": 1.0,
            "p95_distance_from_local_origin": 2.0,
            "max_distance_from_local_origin": 8.0,
            "p95_scale": 0.2,
            "max_scale": 0.9,
            "max_quaternion_norm_deviation": 1e-6,
        },
        "active_map_merge_performed": False,
        "native_to_unblur_conversion_performed": False,
        "local_coordinate_contract": {
            "safe_for_active_unblur_map_merge": False,
        },
    }


def test_snapshot_is_latest_eight_past_only_and_content_bound() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = _snapshot(root)
        snapshot = load_snapshot(output)
        assert [frame["frame_id"] for frame in snapshot["frames"]] == list(range(2, 10))
        assert snapshot["context_keyframes"] == 8
        assert snapshot["uses_ground_truth_pose"] is False
        assert snapshot["uses_clear_gt_membership"] is False
        assert snapshot["active_map_state_included"] is False
        image_path = output / snapshot["frames"][0]["image_path"]
        image_path.write_bytes(b"tampered")
        try:
            load_snapshot(output)
        except ValueError as error:
            assert "image hash" in str(error)
        else:
            raise AssertionError("tampered snapshot image was accepted")


def test_future_frame_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        try:
            materialize_closed_submap_snapshot(
                snapshots_root=directory,
                submap_id=0,
                record_keyframe_ids=list(range(8)),
                frames=[_frame(index, index) for index in range(8)],
                closure_sequence_ordinal=6,
                pose_revision=0,
            )
        except ValueError as error:
            assert "future keyframe" in str(error)
        else:
            raise AssertionError("future frame entered a past-only snapshot")


def test_snapshot_accepts_real_torch_batched_chw_image() -> None:
    with tempfile.TemporaryDirectory() as directory:
        frames = [_frame(index, index) for index in range(8)]
        batched = torch.linspace(0.0, 1.0, 3 * 12 * 16).reshape(1, 3, 12, 16)
        frames[0] = SidecarFrameInput(
            frame_id=0,
            sequence_ordinal=0,
            c2w=_pose(),
            intrinsics_px=((10.0, 0.0, 8.0), (0.0, 11.0, 6.0), (0.0, 0.0, 1.0)),
            image=batched,
        )
        output = materialize_closed_submap_snapshot(
            snapshots_root=directory,
            submap_id=0,
            record_keyframe_ids=list(range(8)),
            frames=frames,
            closure_sequence_ordinal=7,
            pose_revision=0,
        )
        snapshot = load_snapshot(output)
        assert snapshot["frames"][0]["image_size_wh"] == [16, 12]
        assert (output / snapshot["frames"][0]["image_path"]).is_file()


def test_configuration_refuses_nonofficial_topology_and_active_merge() -> None:
    try:
        SidecarConfig(context_keyframes=16)
    except ValueError as error:
        assert "exactly 8" in str(error)
    else:
        raise AssertionError("non-8v sidecar was accepted")
    try:
        SidecarConfig(active_map_merge=True)
    except UnsupportedActiveMapMerge:
        pass
    else:
        raise AssertionError("active map merge was accepted")
    assessment = active_map_merge_assessment()
    assert assessment["supported"] is False
    assert "no_unblur_optimizer_state" in assessment["reason_codes"]
    try:
        reject_active_map_merge()
    except UnsupportedActiveMapMerge as error:
        assert "no_unblur_optimizer_state" in str(error)
    else:
        raise AssertionError("merge rejection did not fail closed")


def test_result_and_pose_staleness_gates() -> None:
    with tempfile.TemporaryDirectory() as directory:
        snapshot = load_snapshot(_snapshot(Path(directory)))
        config = _config(Path(directory))
        result_gate = evaluate_result(
            _valid_result(snapshot), snapshot=snapshot, elapsed_seconds=1.2, config=config
        )
        assert result_gate.accepted
        current = {
            frame["frame_id"]: frame["c2w_opencv"] for frame in snapshot["frames"]
        }
        stale_gate = evaluate_staleness(
            snapshot,
            current_poses=current,
            current_pose_revision=18,
            config=config,
        )
        assert stale_gate.accepted
        changed = dict(current)
        changed[snapshot["frames"][0]["frame_id"]] = _pose(1.0)
        rejected = evaluate_staleness(
            snapshot,
            current_poses=changed,
            current_pose_revision=18,
            config=config,
        )
        assert not rejected.accepted
        assert "pose_translation_drift_exceeded" in rejected.reasons
        bad_result = _valid_result(snapshot)
        bad_result["geometry"]["max_distance_from_local_origin"] = 500.0
        rejected_geometry = evaluate_result(
            bad_result, snapshot=snapshot, elapsed_seconds=1.2, config=config
        )
        assert not rejected_geometry.accepted
        assert "max_distance_gate_exceeded" in rejected_geometry.reasons

        float32_like = copy.deepcopy(snapshot)
        approximate_rotation = float32_like["frames"][0]["c2w_opencv"]
        approximate_rotation[0][0] = 0.99999
        float32_like["frames"][0]["pose_hash"] = pose_hash(approximate_rotation)
        exact_same_current = {
            frame["frame_id"]: frame["c2w_opencv"]
            for frame in float32_like["frames"]
        }
        zero_gate = SidecarConfig(
            output_root=str(directory),
            max_pose_translation_drift=0.0,
            max_pose_rotation_drift_deg=0.0,
        )
        identical = evaluate_staleness(
            float32_like,
            current_poses=exact_same_current,
            current_pose_revision=float32_like["pose_revision"],
            config=zero_gate,
        )
        assert identical.accepted
        assert identical.measurements["max_rotation_drift_deg"] == 0.0


def test_process_queue_publishes_only_after_all_cpu_gates() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output_root = root / "sidecars"
        snapshot_dir = _snapshot(output_root / "snapshots")
        snapshot = load_snapshot(snapshot_dir)
        runner = ROOT / "scripts" / "run_official_resplat_sidecar.py"
        config = SidecarConfig(
            enabled=True,
            output_root=str(output_root),
            python_executable=sys.executable,
            runner_script=str(runner),
            resplat_repo=str(root / "official-resplat"),
            checkpoint=str(root / "checkpoint.pth"),
            expected_checkpoint_sha256="0" * 64,
            cuda_visible_devices="1",
        )

        class FakeProcess:
            def __init__(self, command, **_kwargs):
                output_index = command.index("--output-dir") + 1
                raw = Path(command[output_index])
                raw.mkdir(parents=True)
                artifact = raw / "native_gaussians_local.npz"
                gaussian_count = 8 * (64 // 4) * (64 // 4)
                np.savez_compressed(
                    artifact,
                    means=np.zeros((gaussian_count, 3), np.float32),
                    scales=np.full((gaussian_count, 3), 0.1, np.float32),
                    rotations=np.tile(
                        np.asarray([[0.0, 0.0, 0.0, 1.0]], np.float32),
                        (gaussian_count, 1),
                    ),
                    harmonics=np.zeros((gaussian_count, 3, 9), np.float32),
                    opacities=np.full((gaussian_count,), 0.5, np.float32),
                )
                result = _valid_result(snapshot, "0" * 64)
                result["image_shape_hw"] = [64, 64]
                result["geometry"] = {
                    "gaussian_count": gaussian_count,
                    "finite_fraction": 1.0,
                    "p95_distance_from_local_origin": 0.0,
                    "max_distance_from_local_origin": 0.0,
                    "p95_scale": float(np.float32(0.1)),
                    "max_scale": float(np.float32(0.1)),
                    "max_quaternion_norm_deviation": 0.0,
                }
                result["outputs"] = {
                    "native_gaussians_npz": artifact.name,
                    "native_gaussians_npz_sha256": hashlib.sha256(
                        artifact.read_bytes()
                    ).hexdigest(),
                    "npz_arrays": {
                        "means": {"shape": [gaussian_count, 3], "dtype": "float32"},
                        "scales": {"shape": [gaussian_count, 3], "dtype": "float32"},
                        "rotations": {"shape": [gaussian_count, 4], "dtype": "float32"},
                        "harmonics": {
                            "shape": [gaussian_count, 3, 9],
                            "dtype": "float32",
                        },
                        "opacities": {"shape": [gaussian_count], "dtype": "float32"},
                    },
                }
                (raw / "run_manifest.json").write_text(
                    json.dumps(result) + "\n", encoding="utf-8"
                )
                self.returncode = 0

            def poll(self):
                return 0

        current = {
            frame["frame_id"]: frame["c2w_opencv"] for frame in snapshot["frames"]
        }
        with mock.patch(
            "src.refinement.official_resplat_sidecar.subprocess.Popen", FakeProcess
        ):
            queue = OfficialReSplatSidecarQueue(config)
            submitted = queue.submit(snapshot_dir)
            assert submitted["event"] == "submitted"
            events = queue.poll(current_poses=current, current_pose_revision=18)
        assert [event["event"] for event in events] == ["published"]
        published = Path(events[0]["path"])
        assert published.parent.name == "published"
        gate = json.loads((published / "gate_decision.json").read_text(encoding="utf-8"))
        assert gate["accepted"] is True
        assert gate["active_map_merge"]["supported"] is False


def test_official_runner_core_uses_exact_init_object_and_four_updates() -> None:
    class VersionValue:
        _version = 0

    initial = SimpleNamespace(means=VersionValue())
    refined = [SimpleNamespace(means=VersionValue()) for _ in range(4)]

    class FakeEncoder:
        def __init__(self):
            self.forward_calls = 0
            self.update_calls = 0
            self.received_init = None

        def __call__(self, context, **kwargs):
            assert context == "past8"
            assert kwargs["global_step"] == 0
            assert kwargs["deterministic"] is False
            self.forward_calls += 1
            return {"gaussians": initial, "condition_features": "features"}

        def forward_update(
            self, context, target, features, init_gaussians, decoder, context_remain
        ):
            assert context == "past8"
            assert target == "same_past8"
            assert features == "features"
            assert decoder == "official_decoder"
            assert context_remain is None
            self.received_init = init_gaussians
            self.update_calls += 1
            return {"gaussian": refined}

    encoder = FakeEncoder()
    output, contract = run_official_refinement_core(
        encoder=encoder,
        decoder="official_decoder",
        batch={"context": "past8", "target": "same_past8"},
        stage_runner=lambda _name, operation: operation(),
    )
    assert output is refined[-1]
    assert encoder.forward_calls == 1
    assert encoder.update_calls == 1
    assert encoder.received_init is initial
    assert contract["encoder_forward_calls"] == 1
    assert contract["forward_update_calls"] == 1
    assert contract["init_object_passed_directly"] is True


def test_zero_drain_timeout_terminates_and_rejects_unfinished_child() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output_root = root / "sidecars"
        snapshot_dir = _snapshot(output_root / "snapshots")
        snapshot = load_snapshot(snapshot_dir)
        config = SidecarConfig(
            enabled=True,
            output_root=str(output_root),
            python_executable=sys.executable,
            runner_script=str(ROOT / "scripts" / "run_official_resplat_sidecar.py"),
            resplat_repo=str(root / "official-resplat"),
            checkpoint=str(root / "checkpoint.pth"),
            expected_checkpoint_sha256="0" * 64,
            cuda_visible_devices="1",
            final_drain_timeout_seconds=0.0,
        )

        class FakeRunningProcess:
            def __init__(self, _command, **_kwargs):
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                del timeout
                return self.returncode

        current = {
            frame["frame_id"]: frame["c2w_opencv"] for frame in snapshot["frames"]
        }
        with mock.patch(
            "src.refinement.official_resplat_sidecar.subprocess.Popen",
            FakeRunningProcess,
        ):
            queue = OfficialReSplatSidecarQueue(config)
            queue.submit(snapshot_dir)
            process = queue.active.process
            events = queue.drain(current_pose_provider=lambda: (current, 18))
        assert process.terminated is True
        assert queue.active is None
        assert not queue.pending
        assert any(event["event"] == "rejected" for event in events)
        assert any(
            event["event"] == "drain_timeout_all_unfinished_sidecars_rejected"
            for event in events
        )


def test_motion_only_smoke_materializer_binds_first_closed_causal_prefix() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        selection_path = root / "selection.json"
        selection_path.write_text(
            json.dumps(
                {
                    "keyframe_selection": {
                        "source_indices": list(range(9)),
                        "policy": "motion_filter_only",
                        "predefined_tracking_anchor_list_loaded": False,
                    },
                    "safety": {
                        "clear_gt_membership_file_opened": False,
                        "ground_truth_pose_file_opened": False,
                        "reference_pose_array_created": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        trajectory_path = root / "trajectory.npz"
        trajectory = np.stack([np.asarray(_pose(0.01 * index)) for index in range(9)])
        np.savez(
            trajectory_path,
            traj_est_not_align=trajectory,
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
            reference_pose_arrays_present=np.asarray(False),
        )
        frozen_path = root / "FROZEN.json"
        frozen_path.write_text(
            json.dumps(
                {
                    "artifacts": {
                        "selection_manifest": {
                            "path": str(selection_path),
                            "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                        },
                        "trajectory_npz": {
                            "path": str(trajectory_path),
                            "sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        image_root = root / "turtle_images"
        image_root.mkdir()
        turtle_frames = []
        for index in range(8):
            image_path = image_root / f"{index:06d}.png"
            from PIL import Image

            Image.fromarray(np.full((12, 16, 3), index, np.uint8)).save(image_path)
            turtle_frames.append(
                {
                    "source_index": index,
                    "output": {
                        "path": str(image_path),
                        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    },
                }
            )
        turtle_path = root / "turtle_manifest.json"
        turtle_path.write_text(
            json.dumps(
                {
                    "source": {"uses_ground_truth_pose": False},
                    "safety": {"ground_truth_poses_used": False},
                    "selection": {"emitted_source_indices": list(range(8))},
                    "camera": {
                        "K": [[10.0, 0.0, 8.0], [0.0, 11.0, 6.0], [0.0, 0.0, 1.0]]
                    },
                    "frames": turtle_frames,
                }
            ),
            encoding="utf-8",
        )
        snapshot_dir = materialize_motion_only_smoke(
            frozen_json=frozen_path,
            turtle_manifest_path=turtle_path,
            output_root=root / "sidecar_smoke",
            expected_frozen_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            expected_turtle_sha256=hashlib.sha256(turtle_path.read_bytes()).hexdigest(),
        )
        snapshot = load_snapshot(snapshot_dir)
        assert snapshot["integration_mode"] == "independent_queue_smoke"
        assert [frame["frame_id"] for frame in snapshot["frames"]] == list(range(8))
        provenance = snapshot["source_provenance"]
        assert provenance["closure_trigger_source_index_not_consumed"] == 8
        assert provenance["uses_official_resplat_fps_selection"] is False
        assert provenance["independent_queue_smoke_not_full_slam_integration"] is True


def test_queue_launcher_preserves_venv_python_symlink_lexically() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        lexical = root / "official-env" / "bin" / "python"
        lexical.parent.mkdir(parents=True)
        lexical.symlink_to(Path(sys.executable).resolve())
        selected = lexical_absolute_executable(lexical)
        assert selected == str(lexical.absolute())
        assert selected != str(lexical.resolve())


if __name__ == "__main__":
    test_snapshot_is_latest_eight_past_only_and_content_bound()
    test_future_frame_is_rejected()
    test_snapshot_accepts_real_torch_batched_chw_image()
    test_configuration_refuses_nonofficial_topology_and_active_merge()
    test_result_and_pose_staleness_gates()
    test_process_queue_publishes_only_after_all_cpu_gates()
    test_official_runner_core_uses_exact_init_object_and_four_updates()
    test_zero_drain_timeout_terminates_and_rejects_unfinished_child()
    test_motion_only_smoke_materializer_binds_first_closed_causal_prefix()
    test_queue_launcher_preserves_venv_python_symlink_lexically()
    print("official ReSplat sidecar CPU contracts passed")
