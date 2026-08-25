"""CPU contracts for selection-independent DROID/TURTLE/ReSplat smoke."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


TRACKING = _module(
    "run_fr2_turtle_motion_only_tracking",
    ROOT / "scripts/run_fr2_turtle_motion_only_tracking.py",
)
PROTOCOL = _module(
    "build_motion_only_resplat_protocol",
    ROOT / "scripts/build_motion_only_resplat_protocol.py",
)
EVALUATE = _module(
    "evaluate_frozen_motion_only_tracking",
    ROOT / "scripts/evaluate_frozen_motion_only_tracking.py",
)
PIPELINE = _module(
    "run_motion_only_official_resplat_smoke",
    ROOT / "scripts/run_motion_only_official_resplat_smoke.py",
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class MotionOnlyTrackingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = TRACKING.load_config()

    def test_real_config_is_cpu_preflightable_and_pins_fp16(self) -> None:
        audit = TRACKING.validate_config(self.cfg, verify_turtle_weights=False)
        self.assertEqual(self.cfg["scene"], TRACKING.NON_PROTOCOL_SCENE)
        self.assertEqual(self.cfg["deblur"]["turtle_inference_precision"], "fp16")
        self.assertTrue(self.cfg["only_tracking"])
        self.assertEqual(audit["frames_csv"].name, "estimated_frames.csv")

    def test_selection_stream_first_item_satisfies_base_dataset_contract(self) -> None:
        audit = TRACKING.validate_config(self.cfg, verify_turtle_weights=False)
        stream = TRACKING.SelectionOnlyRgbStream(self.cfg, audit["frames_csv"])
        stream.device = "cpu"
        timestamp, image, depth, pose, observed_placeholders = stream[0]
        self.assertEqual(timestamp, 0)
        self.assertEqual(tuple(image.shape[-2:]), (384, 512))
        self.assertIsNone(depth)
        self.assertEqual(tuple(pose.shape), (2, 4, 4))
        self.assertEqual(len(observed_placeholders), 1)
        self.assertFalse(stream.clear_init)

    def test_scene_alias_disables_motion_filter_anchor_loader(self) -> None:
        from thirdparty.glorie_slam.motion_filter import MotionFilter

        motion_filter = MotionFilter.__new__(MotionFilter)
        motion_filter.cfg = {
            "dataset": "tumrgbd",
            "scene": TRACKING.NON_PROTOCOL_SCENE,
        }
        self.assertIsNone(motion_filter._load_tracking_anchor_indices())

    def test_every_known_anchor_alias_fails_preflight(self) -> None:
        for token in TRACKING.FORBIDDEN_SCENE_TOKENS:
            changed = copy.deepcopy(self.cfg)
            changed["scene"] = f"selection_{token}"
            with self.assertRaisesRegex(ValueError, "predefined MotionFilter"):
                TRACKING.validate_config(changed, verify_turtle_weights=False)

    def test_no_selection_script_imports_eval_membership_helpers(self) -> None:
        for name in (
            "run_fr2_turtle_motion_only_tracking.py",
            "build_motion_only_resplat_protocol.py",
            "run_motion_only_official_resplat_smoke.py",
        ):
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertNotIn("src.utils.eval_frames", source)
            self.assertNotIn("clear_gt_source_indices(", source)
            self.assertNotIn("validate_clear_gt_protocol_scope(", source)

    def test_tampering_precision_or_replay_fails(self) -> None:
        changed = copy.deepcopy(self.cfg)
        changed["deblur"]["turtle_inference_precision"] = "fp32"
        with self.assertRaisesRegex(ValueError, "turtle_inference_precision"):
            TRACKING.validate_config(changed, verify_turtle_weights=False)
        changed = copy.deepcopy(self.cfg)
        changed["mapping"]["resplat"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "legacy replay"):
            TRACKING.validate_config(changed, verify_turtle_weights=False)


class FrozenProtocolContracts(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, dict]:
        count = 221
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
        poses[:, 0, 3] = np.linspace(0.0, 2.2, count, dtype=np.float32)
        keyframes = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]

        trajectory = root / "trajectory_estimated_unaligned.npz"
        np.savez(
            trajectory,
            traj_est_not_align=poses,
            traj_est_not_align_timestamps=np.arange(count, dtype=np.float64),
            traj_est_not_align_eval_mask=np.zeros(count, dtype=np.bool_),
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
            selection_phase=np.asarray(True),
            reference_pose_arrays_present=np.asarray(False),
        )
        video = root / "video.npz"
        np.savez(
            video,
            poses=poses[keyframes],
            timestamps=np.asarray(keyframes, dtype=np.float32),
        )
        selection_path = root / "selection_manifest.json"
        selection = {
            "schema": "unblur_slam.frozen_motion_only_tracking.v1",
            "keyframe_selection": {"source_indices": keyframes},
            "safety": {
                "ground_truth_pose_file_opened": False,
                "clear_gt_membership_file_opened": False,
                "image_metric_computed": False,
                "trajectory_metric_computed": False,
                "legacy_replay_used": False,
                "official_resplat_used": False,
            },
        }
        _json(selection_path, selection)
        freeze = root / "FROZEN.json"
        _json(
            freeze,
            {
                "schema": "unblur_slam.frozen_estimate_gate.v1",
                "selection_frozen_before_evaluation": True,
                "artifacts": {
                    "selection_manifest": {
                        "path": str(selection_path),
                        "sha256": PROTOCOL.sha256_file(selection_path),
                    },
                    "trajectory_npz": {
                        "path": str(trajectory),
                        "sha256": PROTOCOL.sha256_file(trajectory),
                    },
                    "video_npz": {
                        "path": str(video),
                        "sha256": PROTOCOL.sha256_file(video),
                    },
                },
            },
        )
        frames_csv = root / "frames.csv"
        frames_csv.write_text("selection-phase bytes only\n", encoding="utf-8")
        config = json.loads(PROTOCOL.CONFIG.read_text(encoding="utf-8"))
        config["tracking"]["root"] = str(root)
        config["tracking"]["freeze_marker"] = str(freeze)
        config["source"]["frames_csv"] = str(frames_csv)
        config["source"]["frames_csv_sha256"] = PROTOCOL.sha256_file(frames_csv)
        config["outputs"]["protocol_dir"] = str(root / "protocol")
        return config, selection

    def test_load_frozen_inputs_and_official_fps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = self._fixture(Path(temporary))
            frozen = PROTOCOL.load_frozen_inputs(config)
            self.assertEqual(frozen["keyframes"][0], 0)
            self.assertEqual(len(frozen["keyframes"]), 10)
            local = PROTOCOL.official_fps_indices(
                frozen["poses"][frozen["keyframes"]], 8
            )
            self.assertEqual(len(local), 8)
            self.assertEqual(local, sorted(set(local)))

    def test_build_protocol_freezes_index_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._fixture(root)
            config_path = root / "protocol_config.json"
            _json(config_path, config)
            manifest_path = PROTOCOL.build_protocol(config_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            keyframes = manifest["keyframe_selection"]["source_indices"]
            contexts = manifest["resplat_selection"]["context_source_indices"]
            targets = manifest["resplat_selection"]["target_source_indices"]
            self.assertEqual(sorted(contexts + targets), keyframes)
            self.assertFalse(set(contexts) & set(targets))
            self.assertEqual(len(contexts), 8)
            self.assertTrue((manifest_path.parent / "protocol_manifest.sha256").is_file())

    def test_pipeline_preflight_preserves_resplat_environment_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._fixture(root)
            for key in ("turtle_dir", "scene_dir", "resplat_dir", "audit_dir"):
                config["outputs"][key] = str(root / key)
            config_path = root / "protocol_config.json"
            _json(config_path, config)
            PROTOCOL.build_protocol(config_path)
            audit = PIPELINE.preflight(config_path)
            lexical = audit["execution"]["resplat_python_lexical"]
            realpath = audit["execution"]["resplat_python_realpath"]
            self.assertNotEqual(lexical, realpath)
            self.assertTrue(audit["execution"]["lexical_environment_path_preserved"])

    def test_export_pose_bundle_is_minimal_and_identical_to_frozen_estimate(self) -> None:
        from scripts.export_tum_official_resplat_scene import load_pose_override

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._fixture(root)
            source = root / "trajectory_estimated_unaligned.npz"
            audit = {
                "frozen_tracking": {
                    "trajectory": str(source),
                    "trajectory_sha256": PROTOCOL.sha256_file(source),
                },
                "outputs": {"scene_dir": str(root / "scene")},
            }
            bundle = PIPELINE.materialize_export_pose_bundle(audit, config)
            with np.load(source, allow_pickle=False) as frozen, np.load(
                bundle, allow_pickle=False
            ) as exported:
                self.assertEqual(
                    set(exported.files),
                    {
                        "traj_est_not_align",
                        "pose_source",
                        "uses_ground_truth_pose",
                    },
                )
                self.assertTrue(
                    np.array_equal(
                        exported["traj_est_not_align"], frozen["traj_est_not_align"]
                    )
                )
            poses, metadata = load_pose_override(
                bundle, "traj_est_not_align", minimum_length=221
            )
            self.assertEqual(poses.shape, (221, 4, 4))
            self.assertFalse(metadata["contains_ground_truth_sidecar"])
            self.assertEqual(
                PIPELINE.materialize_export_pose_bundle(audit, config), bundle
            )

    def test_reference_array_in_frozen_trajectory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._fixture(root)
            trajectory = root / "trajectory_estimated_unaligned.npz"
            with np.load(trajectory, allow_pickle=False) as old:
                values = {key: old[key] for key in old.files}
            values["traj_ref_poses"] = values["traj_est_not_align"].copy()
            np.savez(trajectory, **values)
            freeze_path = Path(config["tracking"]["freeze_marker"])
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            freeze["artifacts"]["trajectory_npz"]["sha256"] = PROTOCOL.sha256_file(
                trajectory
            )
            _json(freeze_path, freeze)
            with self.assertRaisesRegex(ValueError, "forbidden reference arrays"):
                PROTOCOL.load_frozen_inputs(config)

    def test_freeze_hash_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._fixture(root)
            selection = root / "selection_manifest.json"
            selection.write_text(selection.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                PROTOCOL.load_frozen_inputs(config)

    def test_config_rejects_clear_conditioned_reuse_and_26k(self) -> None:
        config = json.loads(PROTOCOL.CONFIG.read_text(encoding="utf-8"))
        PROTOCOL._validate_config(config)
        for key in ("clear_conditioned_42_frame_artifact_reuse", "26k_comparison"):
            changed = copy.deepcopy(config)
            changed["excluded"][key] = True
            with self.assertRaisesRegex(ValueError, f"excluded.{key}"):
                PROTOCOL._validate_config(changed)

    def test_postfreeze_ate_recovers_similarity_transform(self) -> None:
        reference = np.repeat(np.eye(4)[None], 5, axis=0)
        reference[:, :3, 3] = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]],
            dtype=np.float64,
        )
        estimate = reference.copy()
        estimate[:, :3, 3] = 2.0 * estimate[:, :3, 3] + np.asarray([3, 1, -2])
        result = EVALUATE._ate(estimate, reference)
        self.assertLess(result["ape_translation"]["rmse"], 1.0e-8)
        self.assertAlmostEqual(result["alignment"]["scale"], 0.5, places=8)


if __name__ == "__main__":
    unittest.main()
