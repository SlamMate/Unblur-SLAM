#!/usr/bin/env python3
"""CPU/source contracts for the official TURTLE -> ReSplat orchestrator."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_official_turtle_resplat_pipeline.py"
SPEC = importlib.util.spec_from_file_location("official_turtle_resplat_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return PIPELINE.sha256_file(path)


def _pipeline_config(root: Path) -> dict:
    return {
        "schema": PIPELINE.SCHEMA,
        "source": {
            "frames_csv_sha256": "0" * 64,
            "pose_source": "droid_traj_est_not_align",
            "uses_ground_truth_pose": False,
        },
        "selection": {
            "source_indices": list(PIPELINE.KEYFRAME_INDICES),
            "num_context": 8,
            "context_strategy": "fps",
            "num_target": 34,
        },
        "official_turtle": {
            "origin": PIPELINE.OFFICIAL_TURTLE_ORIGIN,
            "commit": PIPELINE.OFFICIAL_TURTLE_COMMIT,
            "architecture_sha256": PIPELINE.OFFICIAL_TURTLE_ARCH_SHA256,
            "config_sha256": PIPELINE.OFFICIAL_TURTLE_CONFIG_SHA256,
            "checkpoint_kind": "official_gopro",
            "checkpoint_sha256": PIPELINE.OFFICIAL_TURTLE_GOPRO_SHA256,
            "cache_contract": PIPELINE.OFFICIAL_TURTLE_CACHE_CONTRACT,
            "required_processed_range": [0, 2764],
            "required_cache_updates": 2765,
            "preprocessing_contract": {
                "raw_size": [640, 480],
                "undistort": "opencv_same_K",
                "resize_before_crop": [528, 400],
                "crop_edges": [8, 8, 8, 8],
                "output_size": [512, 384],
                "K": [list(row) for row in PIPELINE.TRACKER_K],
                "distortion": list(PIPELINE.TRACKER_DISTORTION),
            },
        },
        "official_resplat": {
            "origin": PIPELINE.OFFICIAL_RESPLAT_ORIGIN,
            "commit": PIPELINE.OFFICIAL_RESPLAT_COMMIT,
            "model_preset": PIPELINE.OFFICIAL_RESPLAT_PRESET,
            "checkpoint_sha256": PIPELINE.OFFICIAL_RESPLAT_CHECKPOINT_SHA256,
            "num_refine": 4,
            "near": 0.01,
            "far": 200.0,
            "render_chunk_size": 4,
        },
        "excluded_legacy_components": {
            "mapping.resplat": False,
            "residual_replay": False,
            "causal_evssm": False,
            "custom_replay_sampler": False,
        },
        "execution": {
            "physical_gpu": 1,
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices": "1",
            "process_device": "cuda:0",
        },
        "paths": {
            "frames_csv": str(root / "frames.csv"),
            "turtle_manifest": str(root / "turtle" / "manifest.json"),
            "turtle_repo": str(root / "TURTLE"),
            "turtle_checkpoint": str(root / "turtle.pth"),
            "resplat_repo": str(root / "resplat"),
            "resplat_checkpoint": str(root / "resplat.pth"),
            "resplat_python": str(root / "python"),
            "scene_output_dir": str(root / "scene"),
            "paired_output_dir": str(root / "paired"),
            "audit_output_dir": str(root / "audit"),
        },
    }


def _stream_manifest(root: Path) -> tuple[Path, dict, dict[str, str]]:
    manifest_path = root / "turtle" / "manifest.json"
    csv_path = root / "frames.csv"
    architecture = root / "arch.py"
    config = root / "gopro.yml"
    checkpoint = root / "turtle.pth"
    image = root / "input.png"
    digests = {
        "csv": _write(csv_path, b"index,frame\n"),
        "architecture": _write(architecture, b"official architecture"),
        "config": _write(config, b"official config"),
        "checkpoint": _write(checkpoint, b"official checkpoint"),
        "image": _write(image, b"not decoded in this unit test"),
    }
    steps = [
        {
            "source_index": index,
            "step_index": index,
            "timestamp": 1000.0 + index * 0.1,
            "input_path": str(image),
            "input_file_sha256": digests["image"],
            "input_rgb_u8_pixel_sha256": "%064x" % (index + 1),
            "output_rgb_u8_pixel_sha256": "%064x" % (index + 2),
            "emitted_png": index in set(PIPELINE.KEYFRAME_INDICES),
            "emitted_png_sha256": (
                "%064x" % (index + 3)
                if index in set(PIPELINE.KEYFRAME_INDICES)
                else None
            ),
            "cache_present_before": index > 0,
            "cache_present_after": True,
            "k_cache_slots_after": 8,
            "v_cache_slots_after": 8,
            "k_cache_non_null_count_after": 5,
            "v_cache_non_null_count_after": 5,
            "k_cache_non_null_mask_after": list(
                PIPELINE.OFFICIAL_TURTLE_CACHE_NON_NULL_MASK
            ),
            "v_cache_non_null_mask_after": list(
                PIPELINE.OFFICIAL_TURTLE_CACHE_NON_NULL_MASK
            ),
            "cache_update_ordinal": index + 1,
            "reset_count": 1,
        }
        for index in range(PIPELINE.STREAM_COUNT)
    ]
    frames = [
        {
            "source_index": index,
            "step_index": index,
            "timestamp": 1000.0 + index * 0.1,
            "output": {
                "path": str(root / "turtle" / "images" / f"{index:06d}.png"),
                "sha256": "%064x" % (index + 3),
                "width": 512,
                "height": 384,
            },
            "stream_audit": {
                "cache_present_before": index > 0,
                "cache_present_after": True,
                "cache_update_ordinal": index + 1,
                "reset_count": 1,
            },
        }
        for index in PIPELINE.KEYFRAME_INDICES
    ]
    payload = {
        "schema": PIPELINE.TURTLE_MANIFEST_SCHEMA,
        "source": {
            "path": str(csv_path),
            "sha256": digests["csv"],
            "pose_source_declared_but_not_consumed": "droid_traj_est_not_align",
            "uses_ground_truth_pose": False,
            "poses_consumed_by_turtle": False,
            "depth_consumed_by_turtle": False,
            "ground_truth_images_consumed_by_turtle": False,
        },
        "camera": {
            "model": "PINHOLE",
            "width": 512,
            "height": 384,
            "K": [list(row) for row in PIPELINE.TRACKER_K],
            "raw_width": 640,
            "raw_height": 480,
            "resize_before_crop_width": 528,
            "resize_before_crop_height": 400,
            "crop_edges": {"left": 8, "right": 8, "top": 8, "bottom": 8},
            "distortion": {
                "model": "opencv_radial_tangential",
                "vector": list(PIPELINE.TRACKER_DISTORTION),
            },
            "preprocessing": list(PIPELINE.TRACKER_PREPROCESSING),
        },
        "turtle": {
            "repository": {
                "path": str(root / "TURTLE"),
                "origin": PIPELINE.OFFICIAL_TURTLE_ORIGIN,
                "commit": PIPELINE.OFFICIAL_TURTLE_COMMIT,
            },
            "architecture": {"path": str(architecture), "sha256": digests["architecture"]},
            "config": {"path": str(config), "sha256": digests["config"]},
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": digests["checkpoint"],
                "metadata": {"kind": "official_gopro"},
            },
            "cache_contract": PIPELINE.OFFICIAL_TURTLE_CACHE_CONTRACT,
        },
        "stream": {
            "processed_source_indices": list(range(PIPELINE.STREAM_COUNT)),
            "processed_range": {
                "start_source_index": 0,
                "end_source_index": 2764,
                "inclusive": True,
            },
            "processed_count": 2765,
            "step_count": 2765,
            "cache_updates": 2765,
            "reset_count": 1,
            "reset_events": [
                {
                    "before_source_index": 0,
                    "reset_ordinal": 1,
                    "reason": "explicit_sequence_start",
                }
            ],
            "strictly_increasing_source_indices": True,
            "strictly_increasing_timestamps": True,
            "gaps_skipped": False,
            "first_pair": "self",
            "one_step_per_source_frame": True,
            "persistent_kv": True,
            "k_cache_slots": 8,
            "v_cache_slots": 8,
            "k_cache_non_null_count": 5,
            "v_cache_non_null_count": 5,
            "official_gopro_cache_non_null_mask": list(
                PIPELINE.OFFICIAL_TURTLE_CACHE_NON_NULL_MASK
            ),
            "cache_contract": PIPELINE.OFFICIAL_TURTLE_CACHE_CONTRACT,
            "steps": steps,
        },
        "selection": {
            "emitted_source_indices": list(PIPELINE.KEYFRAME_INDICES),
            "emitted_count": 42,
        },
        "safety": {
            "ground_truth_images_used": False,
            "ground_truth_poses_used": False,
            "depth_used": False,
            "custom_causal_evssm_used": False,
            "sliding_window_recomputation_used": False,
        },
        "frames": frames,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path, payload, digests


class OfficialTurtleReSplatPipelineTests(unittest.TestCase):
    def test_accepts_only_full_consecutive_stream_then_42_emissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, payload, digests = _stream_manifest(root)
            with (
                mock.patch.object(PIPELINE, "OFFICIAL_TURTLE_ARCH_SHA256", digests["architecture"]),
                mock.patch.object(PIPELINE, "OFFICIAL_TURTLE_CONFIG_SHA256", digests["config"]),
                mock.patch.object(PIPELINE, "OFFICIAL_TURTLE_GOPRO_SHA256", digests["checkpoint"]),
            ):
                audit = PIPELINE.validate_turtle_stream_manifest(
                    payload,
                    manifest_path,
                    expected_frames_csv=root / "frames.csv",
                    expected_frames_csv_sha256=digests["csv"],
                    expected_turtle_repo=root / "TURTLE",
                    expected_checkpoint=root / "turtle.pth",
                    verify_frame_files=False,
                    verify_all_step_inputs=False,
                )
            self.assertEqual(audit["stream"]["processed_count"], 2765)
            self.assertEqual(audit["stream"]["cache_updates"], 2765)
            self.assertEqual(audit["selection"]["source_indices"], list(PIPELINE.KEYFRAME_INDICES))

    def test_rejects_sparse_42_frame_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, payload, digests = _stream_manifest(root)
            sparse = copy.deepcopy(payload)
            sparse["stream"]["processed_source_indices"] = list(PIPELINE.KEYFRAME_INDICES)
            with (
                mock.patch.object(PIPELINE, "OFFICIAL_TURTLE_ARCH_SHA256", digests["architecture"]),
                mock.patch.object(PIPELINE, "OFFICIAL_TURTLE_CONFIG_SHA256", digests["config"]),
                mock.patch.object(PIPELINE, "OFFICIAL_TURTLE_GOPRO_SHA256", digests["checkpoint"]),
                self.assertRaisesRegex(ValueError, "sparse 42-keyframe stream is invalid"),
            ):
                PIPELINE.validate_turtle_stream_manifest(
                    sparse,
                    manifest_path,
                    expected_frames_csv=root / "frames.csv",
                    expected_frames_csv_sha256=digests["csv"],
                    expected_turtle_repo=root / "TURTLE",
                    expected_checkpoint=root / "turtle.pth",
                    verify_frame_files=False,
                    verify_all_step_inputs=False,
                )

    def test_rejects_missing_cache_update(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, payload, digests = _stream_manifest(root)
            payload["stream"]["cache_updates"] = 2764
            with (
                mock.patch.object(PIPELINE, "OFFICIAL_TURTLE_ARCH_SHA256", digests["architecture"]),
                mock.patch.object(PIPELINE, "OFFICIAL_TURTLE_CONFIG_SHA256", digests["config"]),
                mock.patch.object(PIPELINE, "OFFICIAL_TURTLE_GOPRO_SHA256", digests["checkpoint"]),
                self.assertRaisesRegex(ValueError, "cache_updates must equal 2765"),
            ):
                PIPELINE.validate_turtle_stream_manifest(
                    payload,
                    manifest_path,
                    expected_frames_csv=root / "frames.csv",
                    expected_frames_csv_sha256=digests["csv"],
                    expected_turtle_repo=root / "TURTLE",
                    expected_checkpoint=root / "turtle.pth",
                    verify_frame_files=False,
                    verify_all_step_inputs=False,
                )

    def test_commands_are_fixed_official_bridges_and_gpu1_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _pipeline_config(root)
            real_python = root / "real" / "python3.12"
            real_python.parent.mkdir()
            real_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            real_python.chmod(0o755)
            (root / "python").symlink_to(real_python)
            export, paired, environment = PIPELINE.build_commands(config)
        self.assertEqual(Path(export[1]).name, "export_tum_official_resplat_scene.py")
        self.assertEqual(Path(paired[1]).name, "run_paired_official_resplat_smoke.py")
        self.assertIn("--formal-smoke", export)
        self.assertIn("--image-mode", export)
        self.assertEqual(export[export.index("--image-mode") + 1], "turtle")
        self.assertEqual(paired[paired.index("--context-selection") + 1], "fps")
        self.assertEqual(paired[paired.index("--expected-target-count") + 1], "34")
        self.assertEqual(paired[paired.index("--device") + 1], "cuda:0")
        self.assertEqual(paired[paired.index("--near") + 1], "0.01")
        self.assertEqual(paired[paired.index("--far") + 1], "200.0")
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1")
        combined = " ".join(export + paired).lower()
        self.assertNotIn("run_fr2_resplat_smoke", combined)
        self.assertNotIn("run_fr2_causal_smoke", combined)

    def test_resplat_python_symlink_preserves_lexical_environment_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _pipeline_config(root)
            lexical = root / "env" / "bin" / "python"
            real = root / "base" / "bin" / "python3.12"
            lexical.parent.mkdir(parents=True)
            real.parent.mkdir(parents=True)
            real.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            real.chmod(0o755)
            lexical.symlink_to(real)
            config["paths"]["resplat_python"] = str(lexical)

            configured, realpath = PIPELINE._required_executable_lexical(
                config, "resplat_python"
            )
            _, paired, _ = PIPELINE.build_commands(config)

            self.assertEqual(configured, lexical)
            self.assertEqual(realpath, real)
            self.assertNotEqual(configured, realpath)
            self.assertEqual(paired[0], str(lexical))
            self.assertNotEqual(paired[0], str(realpath))

    def test_existing_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _pipeline_config(root)
            (root / "scene").mkdir()
            with self.assertRaisesRegex(FileExistsError, "scene_output_dir"):
                PIPELINE._output_paths(config)

    def test_paired_only_requires_scene_but_refuses_existing_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _pipeline_config(root)
            with self.assertRaisesRegex(FileNotFoundError, "existing scene_output_dir"):
                PIPELINE._output_paths(config, require_existing_scene=True)
            (root / "scene").mkdir()
            outputs = PIPELINE._output_paths(config, require_existing_scene=True)
            self.assertEqual(outputs["scene_output_dir"], root / "scene")
            (root / "paired").mkdir()
            with self.assertRaisesRegex(FileExistsError, "paired_output_dir"):
                PIPELINE._output_paths(config, require_existing_scene=True)
            outputs = PIPELINE._output_paths(
                config,
                require_existing_scene=True,
                require_existing_paired=True,
            )
            self.assertEqual(outputs["paired_output_dir"], root / "paired")

    def test_paired_only_executes_no_export_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = {
                "outputs": {"audit_output_dir": str(root / "audit")},
                "existing_scene_validation": {"sha256": "1" * 64},
            }
            config = _pipeline_config(root)
            export_command = ["python", "export.py"]
            paired_command = ["env-python", "paired.py"]
            installed = {"scene_manifest": {}, "paired_manifest": {}}
            with (
                mock.patch.object(PIPELINE, "preflight", return_value=audit) as preflight,
                mock.patch.object(PIPELINE, "_load_json", return_value=(root / "c.json", config)),
                mock.patch.object(
                    PIPELINE,
                    "build_commands",
                    return_value=(
                        export_command,
                        paired_command,
                        {"CUDA_VISIBLE_DEVICES": "1"},
                    ),
                ),
                mock.patch.object(PIPELINE, "_run_logged") as run_logged,
                mock.patch.object(
                    PIPELINE, "_validate_installed_outputs", return_value=installed
                ),
                mock.patch.object(
                    PIPELINE,
                    "_install_pipeline_audit",
                    return_value=root / "audit",
                ) as install,
            ):
                output = PIPELINE.run_pipeline(root / "c.json", paired_only=True)

            self.assertEqual(output, root / "audit")
            preflight.assert_called_once_with(
                root / "c.json", require_existing_scene=True
            )
            self.assertEqual(run_logged.call_count, 1)
            self.assertEqual(run_logged.call_args.args[0], paired_command)
            self.assertNotEqual(run_logged.call_args.args[0], export_command)
            final_record = install.call_args.args[1]
            self.assertIsNone(final_record["commands"]["export"])
            self.assertTrue(final_record["resume"]["paired_only"])

    def test_source_does_not_import_legacy_refinement_modules(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any("resplat_replay" in name for name in imported))
        self.assertFalse(any("causal" in name for name in imported))
        self.assertFalse(any("mapper" in name for name in imported))

    def test_repository_formal_config_pins_contract(self) -> None:
        path = (
            ROOT
            / "configs"
            / "local"
            / "official_turtle_resplat"
            / "fr2_xyz_gopro_42kf_smoke.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        PIPELINE._assert_config_contract(payload)
        self.assertEqual(payload["selection"]["source_indices"], list(PIPELINE.KEYFRAME_INDICES))
        self.assertEqual(payload["official_turtle"]["required_cache_updates"], 2765)
        self.assertEqual(payload["execution"]["physical_gpu"], 1)


if __name__ == "__main__":
    unittest.main()
