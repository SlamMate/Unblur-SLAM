#!/usr/bin/env python3
"""CPU checks for fail-fast pipeline configuration."""

import json
import hashlib
import copy
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run import (
    _framecrafter_export_id,
    _framecrafter_namespace,
    _validate_framecrafter_manifest,
    prepare_or_validate_inputs,
    validate_clear_gt_scope,
)
from src.utils.eval_frames import clear_gt_metric_scope
from thirdparty.glorie_slam.config import load_config
from src.framecrafter_pipeline import (
    load_frames_csv,
    source_input_digest,
    synthetic_output_digest,
)
from scripts.run_framecrafter_preprocess import compute_preprocess_signature
import scripts.run_framecrafter_preprocess as framecrafter_script


def base_config():
    return load_config(
        str(ROOT / "configs" / "TUM_RGBD" / "freiburg2_xyz.yaml"),
        str(ROOT / "configs" / "unblur_slam.yaml"),
    )


def expect(exception, function):
    try:
        function()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TinyCausalModule(torch.nn.Module):
    def forward(self, frames):
        return frames[:, -1]


class TwoInputCausalModule(torch.nn.Module):
    def forward(self, frames, teacher):
        return frames[:, -1] + 0.0 * teacher[:, -1]


def save_tiny_causal_checkpoint(
    path,
    max_history,
    include_metadata=True,
    use_teacher_input=False,
    input_domain="raw",
    teacher_provenance=None,
):
    model = torch.jit.trace(TinyCausalModule(), torch.zeros(1, 2, 3, 4, 4))
    extra_files = {}
    if include_metadata:
        if teacher_provenance is None:
            teacher_provenance = {
                "schema": "unblur_slam.video_deblur_teacher_provenance.v1",
                "storage": "none",
                "teacher_domain": "none",
                "evssm_checkpoint_sha256": None,
            }
        extra_files["metadata.json"] = json.dumps(
            {
                "format": "unblur_slam.causal_video_deblur.torchscript.v1",
                "model_config": {
                    "max_history": int(max_history),
                    "use_teacher_input": bool(use_teacher_input),
                    "input_domain": str(input_domain),
                },
                "teacher_provenance": teacher_provenance,
            }
        )
    torch.jit.save(model, str(path), _extra_files=extra_files)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cfg = base_config()
        cfg["mapping"]["resplat"].update(
            {"enabled": False, "online_enabled": True}
        )
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        cfg = base_config()
        cfg["mapping"]["resplat"].update(
            {
                "enabled": True,
                "backend": "official_resplat",
                "online_enabled": True,
            }
        )
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        for invalid_view_count in (0, 1, 3, 2.5, True):
            cfg = base_config()
            cfg["mapping"]["resplat"].update(
                {
                    "enabled": True,
                    "online_enabled": True,
                    "online_replay_views": invalid_view_count,
                }
            )
            expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        cfg = base_config()
        cfg["mapping"]["resplat"].update(
            {
                "enabled": True,
                "online_enabled": True,
                "online_replay_views": 2,
            }
        )
        prepare_or_validate_inputs(cfg, root)

        cfg = base_config()
        cfg["framecrafter"].update(
            {"enabled": True, "auto_prepare": False, "manifest": ""}
        )
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        image_root = root / "images"
        depth_root = root / "depth"
        repo = root / "FrameCrafter"
        base_model = root / "Wan2.1"
        for directory_path in (image_root, depth_root, repo, base_model):
            directory_path.mkdir()
        (repo / "model.py").write_text("class FrameCrafter: pass\n", encoding="utf-8")
        (base_model / "model_index.json").write_text("{}\n", encoding="utf-8")
        checkpoint = root / "framecrafter.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        original_paths = []
        depth_paths = []
        for index, value in enumerate((32, 224)):
            rgb = image_root / f"{index}.png"
            depth = depth_root / f"{index}.png"
            Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(rgb)
            Image.fromarray(np.full((8, 8), 1000, np.uint16)).save(depth)
            original_paths.append(rgb)
            depth_paths.append(depth)
        synthetic_rgb = root / "synthetic.png"
        synthetic_depth = root / "synthetic_depth.png"
        Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(synthetic_rgb)
        Image.fromarray(np.full((8, 8), 1000, np.uint16)).save(synthetic_depth)
        trajectory = root / "traj_est_not_align.npz"
        trajectory_poses = np.repeat(np.eye(4)[None], 2, axis=0)
        trajectory_poses[1, 0, 3] = 1.0
        np.savez(
            trajectory,
            traj_est_not_align=trajectory_poses,
            traj_est_not_align_timestamps=np.arange(2, dtype=np.float64),
            traj_est_not_align_eval_mask=np.ones(2, dtype=np.bool_),
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
        )
        trajectory_hash = sha256(trajectory)
        frames_csv = root / "estimated_frames.csv"
        frames_csv.write_text(
            "frame,timestamp,tx,ty,tz,qx,qy,qz,qw,depth_path,"
            "pose_source,uses_ground_truth_pose,trajectory_path,"
            "trajectory_sha256,trajectory_key\n"
            f"{original_paths[0]},0,0,0,0,0,0,0,1,{depth_paths[0]},"
            f"droid_traj_est_not_align,false,{trajectory},{trajectory_hash},"
            "traj_est_not_align\n"
            f"{original_paths[1]},1,1,0,0,0,0,0,1,{depth_paths[1]},"
            f"droid_traj_est_not_align,false,{trajectory},{trajectory_hash},"
            "traj_est_not_align\n",
            encoding="utf-8",
        )
        cfg = base_config()
        cfg["framecrafter"].update(
            {
                "enabled": True,
                "auto_prepare": False,
                "frames_csv": str(frames_csv),
                "image_root": str(image_root),
                "depth_root": str(depth_root),
                "repo_path": str(repo),
                "checkpoint": str(checkpoint),
                "base_model_dir": str(base_model),
                "output_dir": str(root / "generated"),
            }
        )
        namespace = _framecrafter_namespace(cfg, root)
        signature = compute_preprocess_signature(namespace)
        manifest = root / "manifest.json"
        report = root / "preprocess_report.json"
        identity = np.eye(4).tolist()
        midpoint = np.eye(4)
        midpoint[0, 3] = 0.5
        frames = [
            {
                "kind": "original", "source_index": 0,
                "rgb_path": str(original_paths[0]),
                "depth_path": str(depth_paths[0]),
                "rgb_sha256": sha256(original_paths[0]),
                "depth_sha256": sha256(depth_paths[0]),
                "c2w": identity, "timestamp": 0.0,
                "eval": True, "fixed_pose": False,
            },
            {
                "kind": "synthetic", "source_index": None,
                "target_id": "synthetic_test",
                "rgb_path": str(synthetic_rgb),
                "depth_path": str(synthetic_depth),
                "rgb_sha256": sha256(synthetic_rgb),
                "depth_sha256": sha256(synthetic_depth),
                "c2w": midpoint.tolist(), "timestamp": 0.5,
                "eval": False, "fixed_pose": True,
                "left_index": 0, "right_index": 1, "alpha": 0.5,
            },
            {
                "kind": "original", "source_index": 1,
                "rgb_path": str(original_paths[1]),
                "depth_path": str(depth_paths[1]),
                "rgb_sha256": sha256(original_paths[1]),
                "depth_sha256": sha256(depth_paths[1]),
                "c2w": identity, "timestamp": 1.0,
                "eval": True, "fixed_pose": False,
            },
        ]
        accepted_digest = synthetic_output_digest(frames)
        source_digest = source_input_digest(frames)
        report.write_text(
            json.dumps(
                {
                    "schema": "unblur_slam.framecrafter_preprocess_report.v1",
                    "pose_source": "droid_traj_est_not_align",
                    "uses_ground_truth_pose": False,
                    "backend": "python_api",
                    "backend_test_only": False,
                    "preprocess_signature": signature,
                    "generation_id": "1" * 32,
                    "source_frame_count": 2,
                    "accepted_target_count": 1,
                    "accepted_output_sha256": accepted_digest,
                    "source_input_sha256": source_digest,
                    "manifest": str(manifest),
                }
            ),
            encoding="utf-8",
        )
        manifest.write_text(
            json.dumps(
                {
                    "schema": "unblur_slam.framecrafter_manifest.v1",
                    "pose_source": "droid_traj_est_not_align",
                    "uses_ground_truth_pose": False,
                    "source_frame_count": 2,
                    "generated_frame_count": 1,
                    "preprocess_signature": signature,
                    "generation_id": "1" * 32,
                    "backend": "python_api",
                    "backend_test_only": False,
                    "preprocess_report_path": str(report),
                    "preprocess_report_sha256": sha256(report),
                    "accepted_output_sha256": accepted_digest,
                    "source_input_sha256": source_digest,
                    "frames": frames,
                }
            ),
            encoding="utf-8",
        )
        cfg["framecrafter"]["manifest"] = str(manifest)
        manual_cfg = copy.deepcopy(cfg)
        manual_cfg["framecrafter"].update(
            {
                "frames_csv": "",
                "trajectory_npz": "",
                "repo_path": "",
                "checkpoint": "",
                "base_model_dir": "",
            }
        )
        original_manual_preprocess = framecrafter_script.run_preprocess

        def manual_must_not_run(*_args, **_kwargs):
            raise AssertionError("manual manifest mode must not generate")

        try:
            framecrafter_script.run_preprocess = manual_must_not_run
            prepare_or_validate_inputs(manual_cfg, root)
        finally:
            framecrafter_script.run_preprocess = original_manual_preprocess
        assert Path(manual_cfg["framecrafter"]["manifest"]).is_absolute()
        assert manual_cfg["framecrafter"]["preprocess_signature"] == signature
        expect(
            ValueError,
            lambda: _validate_framecrafter_manifest(
                manifest, expected_signature="different-inputs"
            ),
        )
        valid_manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        for mutate in (
            lambda value: value.update(pose_source="aligned_to_gt"),
            lambda value: value.update(generated_frame_count=2),
            lambda value: value["frames"][0].update(eval=False),
            lambda value: value["frames"][1].update(fixed_pose=False),
            lambda value: value["frames"][1].update(rgb_sha256="0" * 64),
            lambda value: value.pop("preprocess_report_sha256", None),
            lambda value: value.update(
                backend="test_only_endpoint_blend", backend_test_only=True
            ),
        ):
            invalid_payload = json.loads(json.dumps(valid_manifest_payload))
            mutate(invalid_payload)
            manifest.write_text(json.dumps(invalid_payload), encoding="utf-8")
            expect(ValueError, lambda: _validate_framecrafter_manifest(manifest))
        manifest.write_text(json.dumps(valid_manifest_payload), encoding="utf-8")

        original_run_preprocess = framecrafter_script.run_preprocess
        try:
            cache_hit_cfg = copy.deepcopy(cfg)
            cache_hit_cfg["framecrafter"].update(
                auto_prepare=True, reuse_existing=True
            )

            def cache_must_not_run(*_args, **_kwargs):
                raise AssertionError("matching cache should be reused")

            framecrafter_script.run_preprocess = cache_must_not_run
            prepare_or_validate_inputs(cache_hit_cfg, root)

            snapshot_dir = root / "generated"
            snapshot_dir.mkdir(exist_ok=True)
            generation_id = valid_manifest_payload["generation_id"]
            snapshot_manifest = snapshot_dir / (
                f"manifest_{signature}_{generation_id}.json"
            )
            snapshot_report = snapshot_dir / (
                f"preprocess_report_{signature}_{generation_id}.json"
            )
            snapshot_report_payload = json.loads(report.read_text(encoding="utf-8"))
            snapshot_report_payload["manifest"] = str(snapshot_manifest)
            snapshot_report.write_text(
                json.dumps(snapshot_report_payload), encoding="utf-8"
            )
            snapshot_payload = json.loads(json.dumps(valid_manifest_payload))
            snapshot_payload["preprocess_report_path"] = str(snapshot_report)
            snapshot_payload["preprocess_report_sha256"] = sha256(snapshot_report)
            snapshot_manifest.write_text(
                json.dumps(snapshot_payload), encoding="utf-8"
            )
            discovery_cfg = copy.deepcopy(cfg)
            discovery_cfg["framecrafter"].update(
                auto_prepare=True, reuse_existing=True, manifest=""
            )
            prepare_or_validate_inputs(discovery_cfg, root)
            assert discovery_cfg["framecrafter"]["manifest"] == str(
                snapshot_manifest.resolve()
            )

            regeneration_calls = []

            def fake_regeneration(args, *, precomputed_signature=None):
                regeneration_calls.append(precomputed_signature)
                return {
                    "accepted_target_count": 1,
                    "preprocess_signature": precomputed_signature,
                    "manifest": str(manifest),
                }

            framecrafter_script.run_preprocess = fake_regeneration
            no_reuse_cfg = copy.deepcopy(cfg)
            no_reuse_cfg["framecrafter"].update(
                auto_prepare=True, reuse_existing=False
            )
            prepare_or_validate_inputs(no_reuse_cfg, root)
            assert regeneration_calls == [signature]
        finally:
            framecrafter_script.run_preprocess = original_run_preprocess

        missing_checkpoint_cfg = copy.deepcopy(cfg)
        missing_checkpoint_cfg["framecrafter"].update(
            auto_prepare=True,
            checkpoint=str(root / "missing.safetensors"),
        )
        expect(
            FileNotFoundError,
            lambda: prepare_or_validate_inputs(missing_checkpoint_cfg, root),
        )
        test_backend_cfg = copy.deepcopy(cfg)
        test_backend_cfg["framecrafter"].update(
            auto_prepare=True,
            backend="test_only_blend",
            allow_test_only_backend=True,
        )
        expect(ValueError, lambda: prepare_or_validate_inputs(test_backend_cfg, root))

        args = _framecrafter_namespace(cfg, root)
        assert args.frames_csv == frames_csv.resolve()
        assert args.translation_step == 0.08
        assert args.min_sharpness_gain == 1.05
        signature_before_checkpoint_change = compute_preprocess_signature(args)
        original_checkpoint_bytes = checkpoint.read_bytes()
        try:
            checkpoint.write_bytes(b"CHECKPOINT")
            expect(
                RuntimeError,
                lambda: framecrafter_script.run_preprocess(
                    args,
                    precomputed_signature=signature_before_checkpoint_change,
                ),
            )
        finally:
            checkpoint.write_bytes(original_checkpoint_bytes)

        tum_root = root / "tum_export"
        tum_root.mkdir()
        (tum_root / "rgb").mkdir()
        (tum_root / "depth").mkdir()
        tum_rgb_paths = []
        tum_depth_paths = []
        for index, (source_rgb, source_depth) in enumerate(
            zip(original_paths, depth_paths)
        ):
            target_rgb = tum_root / "rgb" / f"{index}.png"
            target_depth = tum_root / "depth" / f"{index}.png"
            target_rgb.write_bytes(source_rgb.read_bytes())
            target_depth.write_bytes(source_depth.read_bytes())
            tum_rgb_paths.append(target_rgb)
            tum_depth_paths.append(target_depth)
        (tum_root / "rgb.txt").write_text(
            "0.000 rgb/0.png\n0.040 rgb/1.png\n",
            encoding="utf-8",
        )
        (tum_root / "depth.txt").write_text(
            "0.003 depth/0.png\n0.043 depth/1.png\n",
            encoding="utf-8",
        )
        trajectory_a = root / "trajectory_a.npz"
        trajectory_b = root / "trajectory_b.npz"
        poses_a = np.repeat(np.eye(4)[None], 2, axis=0)
        poses_b = poses_a.copy()
        poses_a[1, 0, 3] = 1.0
        poses_b[1, 0, 3] = 9.0
        for path, poses in ((trajectory_a, poses_a), (trajectory_b, poses_b)):
            np.savez(
                path,
                traj_est_not_align=poses,
                traj_est_not_align_timestamps=np.arange(2, dtype=np.float64),
                traj_est_not_align_eval_mask=np.ones(2, dtype=np.bool_),
                pose_source=np.asarray("droid_traj_est_not_align"),
                uses_ground_truth_pose=np.asarray(False),
            )
        export_cfg_a = base_config()
        export_cfg_a["data"].update(
            dataset_root=str(root), input_folder=tum_root.name
        )
        export_cfg_a["max_frames"] = 2
        export_cfg_a["stride"] = 1
        export_cfg_a["framecrafter"].update(
            frames_csv="",
            trajectory_npz=str(trajectory_a),
            output_dir=str(root / "shared_exports"),
        )
        export_args_a = _framecrafter_namespace(export_cfg_a, root)
        export_a_bytes = export_args_a.frames_csv.read_bytes()
        export_cfg_b = copy.deepcopy(export_cfg_a)
        export_cfg_b["framecrafter"]["trajectory_npz"] = str(trajectory_b)
        export_args_b = _framecrafter_namespace(export_cfg_b, root)
        assert export_args_a.frames_csv != export_args_b.frames_csv
        assert export_args_a.frames_csv.read_bytes() == export_a_bytes
        loaded_a = load_frames_csv(
            export_args_a.frames_csv,
            pose_convention="c2w",
            compute_missing_sharpness=False,
            expected_pose_source="droid_traj_est_not_align",
            require_pose_provenance=True,
        )
        loaded_b = load_frames_csv(
            export_args_b.frames_csv,
            pose_convention="c2w",
            compute_missing_sharpness=False,
            expected_pose_source="droid_traj_est_not_align",
            require_pose_provenance=True,
        )
        assert loaded_a[1].c2w[0, 3] == 1.0
        assert loaded_b[1].c2w[0, 3] == 9.0
        second_tum_root = root / "tum_export_second_root"
        second_tum_root.mkdir()
        (second_tum_root / "rgb.txt").write_bytes(
            (tum_root / "rgb.txt").read_bytes()
        )
        (second_tum_root / "depth.txt").write_bytes(
            (tum_root / "depth.txt").read_bytes()
        )
        assert _framecrafter_export_id(
            trajectory_a,
            tum_root,
            export_cfg_a["framecrafter"],
            export_cfg_a,
        ) != _framecrafter_export_id(
            trajectory_a,
            second_tum_root,
            export_cfg_a["framecrafter"],
            export_cfg_a,
        )

        cfg = base_config()
        cfg["mapping"]["resplat"].update(
            {"enabled": True, "budget_mode": "replace_tail", "extra_iters": 26001}
        )
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        forged_two_input = root / "causal_two_input.pt"
        two_input = torch.jit.trace(
            TwoInputCausalModule(),
            (
                torch.zeros(1, 4, 3, 16, 16),
                torch.zeros(1, 4, 3, 16, 16),
            ),
        )
        torch.jit.save(
            two_input,
            str(forged_two_input),
            _extra_files={
                "metadata.json": json.dumps(
                    {
                        "format": "unblur_slam.causal_video_deblur.torchscript.v1",
                        "model_config": {
                            "max_history": 4,
                            "use_teacher_input": False,
                        },
                    }
                )
            },
        )
        cfg = base_config()
        cfg["deblur"].update(
            {
                "frontend": "causal_torchscript",
                "causal_checkpoint": str(forged_two_input),
                "causal_history": 4,
            }
        )
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        teacher_runtime = root / "causal_teacher_runtime.pt"
        save_tiny_causal_checkpoint(
            teacher_runtime, max_history=4, use_teacher_input=True
        )
        cfg = base_config()
        cfg["deblur"].update(
            {
                "frontend": "causal_torchscript",
                "causal_checkpoint": str(teacher_runtime),
                "causal_history": 4,
            }
        )
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        causal = root / "causal.pt"
        save_tiny_causal_checkpoint(causal, max_history=4)
        cfg = base_config()
        cfg["deblur"].update(
            {
                "frontend": "causal_torchscript",
                "causal_checkpoint": str(causal),
                "causal_history": 4,
            }
        )
        prepare_or_validate_inputs(cfg, root)
        assert Path(cfg["deblur"]["causal_checkpoint"]).is_absolute()

        # A raw-input adapter may have learned from an EVSSM distillation
        # target. It does not execute EVSSM at runtime, so that teacher SHA is
        # audited but must not be compared with cfg.evssm_checkpoint.
        raw_distilled = root / "causal_raw_distilled.pt"
        save_tiny_causal_checkpoint(
            raw_distilled,
            max_history=4,
            teacher_provenance={
                "schema": "unblur_slam.video_deblur_teacher_provenance.v1",
                "storage": "runtime_evssm_float_tensor",
                "teacher_domain": "evssm_restored_rgb_0_1",
                "evssm_checkpoint_sha256": "a" * 64,
            },
        )
        raw_distilled_cfg = base_config()
        raw_distilled_cfg["deblur"].update(
            {
                "frontend": "causal_torchscript",
                "causal_checkpoint": str(raw_distilled),
                "causal_history": 4,
            }
        )
        prepare_or_validate_inputs(raw_distilled_cfg, root)
        assert (
            raw_distilled_cfg["deblur"]["causal_teacher_provenance"][
                "evssm_checkpoint_sha256"
            ]
            == "a" * 64
        )

        evssm_checkpoint = root / "evssm.pth"
        evssm_checkpoint.write_bytes(b"preflight-only")
        causal_evssm = root / "causal_evssm.pt"
        save_tiny_causal_checkpoint(
            causal_evssm,
            max_history=4,
            input_domain="evssm",
            teacher_provenance={
                "schema": "unblur_slam.video_deblur_teacher_provenance.v1",
                "storage": "runtime_evssm_float_tensor",
                "teacher_domain": "evssm_restored_rgb_0_1",
                "evssm_checkpoint_sha256": sha256(evssm_checkpoint),
            },
        )
        cfg = base_config()
        cfg["evssm_checkpoint"] = str(evssm_checkpoint)
        cfg["deblur"].update(
            {
                "frontend": "causal_evssm",
                "causal_checkpoint": str(causal_evssm),
                "causal_history": 4,
            }
        )
        prepare_or_validate_inputs(cfg, root)
        assert cfg["deblur"]["causal_teacher_storage"] == "runtime_evssm_float_tensor"
        assert cfg["evssm_checkpoint_sha256"] == sha256(evssm_checkpoint)
        cfg["deblur"]["frontend"] = "causal_torchscript"
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))
        cfg["deblur"]["frontend"] = "causal_evssm"
        cfg["deblur"]["causal_checkpoint"] = str(causal)
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        cfg = base_config()
        cfg["deblur"].update(
            {
                "frontend": "causal_evssm",
                "causal_checkpoint": str(causal_evssm),
                "causal_history": 4,
            }
        )
        cfg["evssm_checkpoint"] = str(root / "missing_evssm.pth")
        expect(FileNotFoundError, lambda: prepare_or_validate_inputs(cfg, root))

        mismatched_evssm = root / "mismatched_evssm.pth"
        mismatched_evssm.write_bytes(b"different-evssm")
        cfg["evssm_checkpoint"] = str(mismatched_evssm)
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        cfg = base_config()
        cfg["deblur"].update(
            {
                "frontend": "causal_torchscript",
                "causal_checkpoint": str(causal),
                "causal_history": 4,
            }
        )
        cfg["deblur"]["causal_history"] = 5
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))
        cfg["deblur"]["causal_history"] = 3
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))
        cfg["deblur"]["causal_history"] = 4
        cfg["deblur"]["stream_every_frame"] = False
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        causal_without_metadata = root / "causal_without_metadata.pt"
        save_tiny_causal_checkpoint(
            causal_without_metadata, max_history=4, include_metadata=False
        )
        cfg = base_config()
        cfg["deblur"].update(
            {
                "frontend": "causal_torchscript",
                "causal_checkpoint": str(causal_without_metadata),
                "causal_history": 4,
            }
        )
        expect(ValueError, lambda: prepare_or_validate_inputs(cfg, root))

        reader = type(
            "Reader", (), {"original_frame_count": 2800, "__len__": lambda self: 2800}
        )()
        cfg = base_config()
        assert len(validate_clear_gt_scope(cfg, reader)) == 42
        prepare_or_validate_inputs(cfg, root)
        assert cfg["warmup_mapper"] is True
        smoke_cfg = load_config(
            ROOT / "configs" / "local" / "fr2_xyz_resplat_smoke" / "baseline.yaml",
            ROOT / "configs" / "unblur_slam.yaml",
        )
        assert smoke_cfg["clear_init"] is False
        assert smoke_cfg["warmup_mapper"] is True
        assert smoke_cfg["tracking"]["backend"]["final_ba"] is True
        assert smoke_cfg["mapping"]["hydrate_missing_droid_keyframes"] is True
        cfg["stride"] = 2
        expect(ValueError, lambda: validate_clear_gt_scope(cfg, reader))
        truncated_reader = type(
            "TruncatedReader",
            (),
            {"original_frame_count": 2700, "__len__": lambda self: 2700},
        )()
        cfg["stride"] = 1
        expect(ValueError, lambda: validate_clear_gt_scope(cfg, truncated_reader))

        prefix_indices = [0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220]
        prefix_reader = type(
            "PrefixReader",
            (),
            {"original_frame_count": 221, "__len__": lambda self: 221},
        )()
        prefix_cfg = base_config()
        prefix_cfg["evaluation"] = {
            "clear_gt_scope": "prefix_smoke",
            "expected_clear_gt_source_indices": prefix_indices,
        }
        assert validate_clear_gt_scope(prefix_cfg, prefix_reader) == set(prefix_indices)
        assert clear_gt_metric_scope(prefix_cfg) == "clear_gt_prefix_smoke"

        bad_prefix = copy.deepcopy(prefix_cfg)
        bad_prefix["evaluation"]["expected_clear_gt_source_indices"] = list(
            reversed(prefix_indices)
        )
        expect(ValueError, lambda: validate_clear_gt_scope(bad_prefix, prefix_reader))

        bad_prefix = copy.deepcopy(prefix_cfg)
        bad_prefix["evaluation"]["expected_clear_gt_source_indices"] = prefix_indices[:-1]
        expect(ValueError, lambda: validate_clear_gt_scope(bad_prefix, prefix_reader))

        one_frame_reader = type(
            "OneClearFrameReader",
            (),
            {"original_frame_count": 9, "__len__": lambda self: 9},
        )()
        one_frame_cfg = copy.deepcopy(prefix_cfg)
        one_frame_cfg["evaluation"]["expected_clear_gt_source_indices"] = [0]
        expect(ValueError, lambda: validate_clear_gt_scope(one_frame_cfg, one_frame_reader))

        full_prefix_cfg = copy.deepcopy(prefix_cfg)
        full_prefix_cfg["evaluation"]["expected_clear_gt_source_indices"] = sorted(
            validate_clear_gt_scope(base_config(), reader)
        )
        expect(ValueError, lambda: validate_clear_gt_scope(full_prefix_cfg, reader))

        stride_prefix_cfg = copy.deepcopy(prefix_cfg)
        stride_prefix_cfg["stride"] = 2
        expect(
            ValueError,
            lambda: validate_clear_gt_scope(stride_prefix_cfg, prefix_reader),
        )

        unknown_scope_cfg = copy.deepcopy(prefix_cfg)
        unknown_scope_cfg["evaluation"]["clear_gt_scope"] = "partial"
        expect(
            ValueError,
            lambda: validate_clear_gt_scope(unknown_scope_cfg, prefix_reader),
        )

        previous_cwd = Path.cwd()
        try:
            os.chdir(root)
            alternate_cwd_cfg = load_config(
                ROOT / "configs" / "I2slam" / "freiburg2_xyz.yaml",
                ROOT / "configs" / "unblur_slam.yaml",
            )
            assert alternate_cwd_cfg["scene"] == "freiburg2_xyz"
            assert alternate_cwd_cfg["tracking"]["warmup"] == 4
        finally:
            os.chdir(previous_cwd)

    print("pipeline_preflight=PASS")


if __name__ == "__main__":
    main()
