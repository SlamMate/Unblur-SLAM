from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


from scripts.run_framecrafter_preprocess import (
    compute_shard_runtime_identity,
    _validate_shard_cli,
    compute_preprocess_signature,
)
from src.framecrafter_sharding import (
    ASSIGNMENT_ALGORITHM,
    SHARD_CONTRACT_SCHEMA,
    assigned_batch_ids,
    build_assignment_table,
    build_shard_contract,
    build_shard_runtime_identity,
    canonical_sha256,
    file_sha256,
    merge_shard_envelopes,
    validate_runtime_identity_against_contract,
    validate_global_plan_against_contract,
    write_shard_envelope,
)


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def runtime_identity_from_contract(contract: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "unblur_slam.framecrafter_shard_runtime_identity.v1",
        **{
            field: contract[field]
            for field in (
                "source_identity_sha256",
                "model_artifact_identity_sha256",
                "semantic_config_sha256",
                "implementation_identity_sha256",
                "canonical_preprocess_signature",
            )
        },
    }


class FrameCrafterShardingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source_rgb = self.root / "source" / "rgb.png"
        self.source_depth = self.root / "source" / "depth.png"
        self.source_rgb.parent.mkdir(parents=True)
        self.source_rgb.write_bytes(b"real-rgb")
        self.source_depth.write_bytes(b"real-depth")
        self.original = {
            "kind": "original",
            "source_index": 0,
            "rgb_path": str(self.source_rgb),
            "depth_path": str(self.source_depth),
            "rgb_sha256": file_sha256(self.source_rgb),
            "depth_sha256": file_sha256(self.source_depth),
            "c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "confidence": 1.0,
            "eval": True,
            "fixed_pose": False,
            "reasons": [],
            "left_index": None,
            "right_index": None,
            "alpha": None,
            "timestamp": 0.0,
        }
        self.batches = []
        self.planned = []
        for index in range(7):
            batch_id = f"gap_batch_{index:02d}"
            target_id = f"gap_{index:02d}_a0500"
            conditioning = [
                {
                    "role": "sharp_before" if index % 2 == 0 else "local_blurry",
                    "image_path": str(self.source_rgb),
                    "image_sha256": file_sha256(self.source_rgb),
                    "input_mode": "raw" if index % 2 == 0 else "evssm",
                    "gate": {"passed": True, "confidence": 0.91},
                }
            ]
            self.batches.append(
                {
                    "batch_id": batch_id,
                    "target_ids": [target_id],
                    "target_count": 1,
                    "context_source_indices": [0],
                    "context_ids": ["000000"],
                    "context_count": 1,
                    "total_view_count": 2,
                    "conditioning": conditioning,
                }
            )
            self.planned.append(
                {
                    "target_id": target_id,
                    "batch_id": batch_id,
                    "batch_target_ids": [target_id],
                    "batch_target_position": 0,
                    "left_index": 10 * index,
                    "right_index": 10 * index + 9,
                    "left_position": 3 * index,
                    "right_position": 3 * index + 1,
                    "alpha": 0.5,
                    "conditioning": conditioning,
                }
            )
        self.contract = build_shard_contract(
            self.batches,
            self.planned,
            shard_count=2,
            runtime_identity=build_shard_runtime_identity(
                source_identity={
                    "rgb": file_sha256(self.source_rgb),
                    "depth": file_sha256(self.source_depth),
                },
                model_artifact_identity={
                    "framecrafter": "adapter-sha",
                    "wan": "base-sha",
                },
                semantic_config={
                    "steps": 20,
                    "seed": 43,
                    "height": 480,
                    "width": 832,
                },
                implementation_identity={"sharding": "implementation-sha"},
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_worker(
        self,
        shard_index: int,
        runtime_identity: dict[str, object] | None = None,
    ) -> Path:
        shard = self.root / f"worker_{shard_index}"
        artifacts = shard / "artifacts"
        artifacts.mkdir(parents=True)
        assigned = set(assigned_batch_ids(self.contract, shard_index))
        local_batches = []
        local_planned = []
        accepted = []
        rejected = []
        synthetic_frames = []
        planned_by_batch = {record["batch_id"]: record for record in self.planned}
        batch_by_id = {batch["batch_id"]: batch for batch in self.batches}
        for batch_id in assigned:
            batch = dict(batch_by_id[batch_id])
            plan = dict(planned_by_batch[batch_id])
            target_id = plan["target_id"]
            pose_npz = artifacts / f"{batch_id}.npz"
            pose_npz.write_bytes(f"poses:{batch_id}".encode())
            batch.update(
                poses_npz=str(pose_npz),
                poses_npz_sha256=file_sha256(pose_npz),
            )
            plan.update(
                poses_npz=str(pose_npz),
                poses_npz_sha256=file_sha256(pose_npz),
            )
            local_batches.append(batch)
            local_planned.append(plan)
            numeric = int(batch_id.rsplit("_", 1)[1])
            provenance = {
                "target_id": target_id,
                "batch_id": batch_id,
                "batch_target_ids": [target_id],
                "batch_target_position": 0,
                "context_source_indices": [0],
                "context_ids": ["000000"],
                "conditioning": batch["conditioning"],
            }
            candidate_rgb = artifacts / f"{target_id}.png"
            candidate_rgb.write_bytes(f"candidate:{target_id}".encode())
            if numeric % 2 == 0:
                candidate_depth = artifacts / f"{target_id}.depth.png"
                candidate_depth.write_bytes(f"depth:{target_id}".encode())
                accepted.append(
                    {
                        **provenance,
                        "confidence": 0.9,
                        "metrics": {"sharpness_gain": 1.2},
                        "rgb_path": str(candidate_rgb),
                        "depth_path": str(candidate_depth),
                    }
                )
                synthetic_frames.append(
                    {
                        "kind": "synthetic",
                        "target_id": target_id,
                        "source_index": None,
                        "rgb_path": str(candidate_rgb),
                        "depth_path": str(candidate_depth),
                        "rgb_sha256": file_sha256(candidate_rgb),
                        "depth_sha256": file_sha256(candidate_depth),
                        "c2w": [
                            [1, 0, 0, numeric],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        "confidence": 0.9,
                        "eval": False,
                        "fixed_pose": True,
                        "reasons": ["low_overlap"],
                        "left_index": plan["left_index"],
                        "right_index": plan["right_index"],
                        "alpha": 0.5,
                        "timestamp": numeric + 0.5,
                        "source_ids": ["000000"],
                        "gate_metrics": {"sharpness_gain": 1.2},
                        "batch_id": batch_id,
                        "batch_target_ids": [target_id],
                        "batch_target_position": 0,
                        "conditioning": batch["conditioning"],
                    }
                )
            else:
                rejected.append(
                    {
                        **provenance,
                        "failures": ["sharpness_gain"],
                        "metrics": {"sharpness_gain": 0.95},
                        "candidate_rgb_path": str(candidate_rgb),
                        "candidate_rgb_sha256": file_sha256(candidate_rgb),
                    }
                )
        local_batches.sort(key=lambda batch: batch["batch_id"])
        local_planned.sort(key=lambda record: record["batch_id"])
        accepted.sort(key=lambda record: record["batch_id"])
        rejected.sort(key=lambda record: record["batch_id"])
        manifest_path = shard / "manifest.json"
        manifest = {
            "schema": "unblur_slam.framecrafter_manifest.v1",
            "source_frame_count": 1,
            "generated_frame_count": len(synthetic_frames),
            "pose_source": "droid_traj_est_not_align",
            "uses_ground_truth_pose": False,
            "frames": [self.original, *synthetic_frames],
        }
        write_json(manifest_path, manifest)
        report = {
            "schema": "unblur_slam.framecrafter_preprocess_report.v1",
            "backend": "python_api",
            "backend_test_only": False,
            "uses_ground_truth_pose": False,
            "pose_source": "droid_traj_est_not_align",
            "shard_runtime_identity": (
                runtime_identity_from_contract(self.contract)
                if runtime_identity is None
                else runtime_identity
            ),
            "shard": {
                "schema": "unblur_slam.framecrafter_worker_shard.v1",
                "experiment_signature": self.contract["experiment_signature"],
                "canonical_preprocess_signature": self.contract[
                    "canonical_preprocess_signature"
                ],
                "runtime_identity_sha256": canonical_sha256(
                    runtime_identity_from_contract(self.contract)
                    if runtime_identity is None
                    else runtime_identity
                ),
                "assignment_table_sha256": self.contract[
                    "assignment_table_sha256"
                ],
                "global_plan_sha256": self.contract["global_plan_sha256"],
                "shard_index": shard_index,
                "shard_count": self.contract["shard_count"],
                "assigned_batch_ids": list(
                    assigned_batch_ids(self.contract, shard_index)
                ),
            },
            "source_frame_count": 1,
            "planned_total_before_cap": len(local_planned),
            "planned_target_count": len(local_planned),
            "selected_target_count": len(local_planned),
            "target_selection_policy": "overlap_deficit_v1",
            "max_targets": 256,
            "generation_batch_count": len(local_batches),
            "backend_generate_call_count": len(local_batches),
            "accepted_target_count": len(accepted),
            "rejected_target_count": len(rejected),
            "accepted_output_sha256": canonical_sha256([]),
            "source_input_sha256": canonical_sha256([]),
            "manifest": str(manifest_path),
            "generation_batches": local_batches,
            "planned": local_planned,
            "accepted": accepted,
            "rejected": rejected,
        }
        report_path = write_json(shard / "report.json", report)
        return write_shard_envelope(
            self.contract,
            shard_index=shard_index,
            report_path=report_path,
            manifest_path=manifest_path,
            output_path=shard / "shard_envelope.json",
        )

    def test_assignment_is_deterministic_gap_aware_and_complete(self) -> None:
        table = build_assignment_table(self.batches, self.planned, 2)
        reversed_table = build_assignment_table(
            list(reversed(self.batches)), list(reversed(self.planned)), 2
        )
        self.assertEqual(
            {row["batch_id"]: row["shard_index"] for row in table},
            {row["batch_id"]: row["shard_index"] for row in reversed_table},
        )
        self.assertEqual(self.contract["schema"], SHARD_CONTRACT_SCHEMA)
        self.assertEqual(self.contract["assignment_algorithm"], ASSIGNMENT_ALGORITHM)
        assigned = [
            batch_id
            for shard_index in range(2)
            for batch_id in assigned_batch_ids(self.contract, shard_index)
        ]
        self.assertCountEqual(assigned, [batch["batch_id"] for batch in self.batches])
        self.assertEqual(len(assigned), len(set(assigned)))

        changed = json.loads(json.dumps(self.planned))
        changed[0]["right_index"] += 1
        changed_table = build_assignment_table(self.batches, changed, 2)
        original_hash = next(row for row in table if row["batch_id"] == "gap_batch_00")[
            "assignment_key_sha256"
        ]
        changed_hash = next(
            row for row in changed_table if row["batch_id"] == "gap_batch_00"
        )["assignment_key_sha256"]
        self.assertNotEqual(original_hash, changed_hash)

    def test_worker_global_plan_must_match_contract_exactly(self) -> None:
        validate_global_plan_against_contract(self.contract, self.batches, self.planned)
        with_output_artifacts = json.loads(json.dumps(self.planned))
        with_output_artifacts[0].update(
            poses_npz=str(self.root / "worker-local.npz"),
            poses_npz_sha256="a" * 64,
        )
        validate_global_plan_against_contract(
            self.contract, self.batches, with_output_artifacts
        )

        changed = json.loads(json.dumps(self.planned))
        changed[0]["conditioning"][0]["role"] = "local_blurry"
        with self.assertRaisesRegex(ValueError, "global plan"):
            validate_global_plan_against_contract(self.contract, self.batches, changed)

    def test_signature_binds_contract_bytes_and_index_not_envelope_path(self) -> None:
        trajectory = self.root / "trajectory.npz"
        np.savez(
            trajectory,
            traj_est_not_align=np.eye(4, dtype=np.float64)[None],
            traj_est_not_align_timestamps=np.asarray([0.0]),
            traj_est_not_align_eval_mask=np.asarray([True]),
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
        )
        trajectory_digest = file_sha256(trajectory)
        frames_csv = self.root / "frames.csv"
        frames_csv.write_text(
            "frame,timestamp,tx,ty,tz,qx,qy,qz,qw,fx,fy,cx,cy,"
            "sharpness,depth_path,pose_source,uses_ground_truth_pose,"
            "trajectory_path,trajectory_sha256,trajectory_key\n"
            f"{self.source_rgb},0,0,0,0,0,0,0,1,20,20,7.5,5.5,1,"
            f"{self.source_depth},droid_traj_est_not_align,false,"
            f"{trajectory},{trajectory_digest},traj_est_not_align\n",
            encoding="utf-8",
        )
        contract_path = write_json(self.root / "contract.json", self.contract)
        model_repo = self.root / "framecrafter-repo"
        base_model = self.root / "wan-model"
        model_repo.mkdir()
        base_model.mkdir()
        (model_repo / "model.py").write_text("MODEL = 'fixture'\n", encoding="utf-8")
        (base_model / "weights.safetensors").write_bytes(b"wan-weights")
        checkpoint = self.root / "framecrafter.safetensors"
        checkpoint.write_bytes(b"checkpoint-v1")
        values = {
            "frames_csv": frames_csv,
            "pose_source": "droid_traj_est_not_align",
            "pose_convention": "c2w",
            "output_dir": self.root / "worker-output",
            "shard_contract": contract_path,
            "shard_index": 0,
            "shard_envelope": self.root / "envelope-a.json",
            "framecrafter_repo": model_repo,
            "checkpoint": checkpoint,
            "base_model_dir": base_model,
            "backend": "python_api",
            "num_inference_steps": 20,
            "seed": 43,
            "min_sharpness_gain": 1.05,
        }
        base_runtime_identity = compute_shard_runtime_identity(
            SimpleNamespace(**values)
        )
        runtime_contract = build_shard_contract(
            self.batches,
            self.planned,
            shard_count=2,
            runtime_identity=base_runtime_identity,
        )
        validate_runtime_identity_against_contract(
            runtime_contract, base_runtime_identity
        )

        for parameter, changed_value in (
            ("num_inference_steps", 50),
            ("seed", 99),
            ("min_sharpness_gain", 1.25),
        ):
            original = values[parameter]
            values[parameter] = changed_value
            changed = compute_shard_runtime_identity(SimpleNamespace(**values))
            self.assertNotEqual(
                changed["semantic_config_sha256"],
                base_runtime_identity["semantic_config_sha256"],
            )
            with self.assertRaisesRegex(ValueError, "semantic_config_sha256"):
                validate_runtime_identity_against_contract(runtime_contract, changed)
            values[parameter] = original

        checkpoint.write_bytes(b"checkpoint-v2")
        changed_model = compute_shard_runtime_identity(SimpleNamespace(**values))
        self.assertNotEqual(
            changed_model["model_artifact_identity_sha256"],
            base_runtime_identity["model_artifact_identity_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "model_artifact_identity_sha256"):
            validate_runtime_identity_against_contract(
                runtime_contract, changed_model
            )
        checkpoint.write_bytes(b"checkpoint-v1")

        signature = compute_preprocess_signature(SimpleNamespace(**values))
        canonical_signature = compute_shard_runtime_identity(
            SimpleNamespace(**values)
        )["canonical_preprocess_signature"]
        values["shard_envelope"] = self.root / "envelope-b.json"
        self.assertEqual(
            signature, compute_preprocess_signature(SimpleNamespace(**values))
        )
        values["shard_index"] = 1
        self.assertNotEqual(
            signature, compute_preprocess_signature(SimpleNamespace(**values))
        )
        self.assertEqual(
            canonical_signature,
            compute_shard_runtime_identity(SimpleNamespace(**values))[
                "canonical_preprocess_signature"
            ],
        )
        values["shard_index"] = 0
        contract_path.write_text(
            json.dumps(self.contract, separators=(",", ":")), encoding="utf-8"
        )
        self.assertNotEqual(
            signature, compute_preprocess_signature(SimpleNamespace(**values))
        )
        self.assertEqual(
            canonical_signature,
            compute_shard_runtime_identity(SimpleNamespace(**values))[
                "canonical_preprocess_signature"
            ],
        )

    def test_worker_sharding_fails_fast_for_partial_or_plan_only_cli(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be used together"):
            _validate_shard_cli(
                SimpleNamespace(
                    shard_contract=self.root / "contract.json",
                    shard_index=0,
                    shard_envelope=None,
                    plan_only=False,
                    backend="python_api",
                )
            )
        with self.assertRaisesRegex(ValueError, "plan-only"):
            _validate_shard_cli(
                SimpleNamespace(
                    shard_contract=self.root / "contract.json",
                    shard_index=0,
                    shard_envelope=self.root / "envelope.json",
                    plan_only=True,
                    backend="python_api",
                )
            )

    def test_envelope_rejects_worker_model_or_semantic_identity_mismatch(self) -> None:
        mismatched = build_shard_runtime_identity(
            source_identity={
                "rgb": file_sha256(self.source_rgb),
                "depth": file_sha256(self.source_depth),
            },
            model_artifact_identity={"framecrafter": "different-checkpoint"},
            semantic_config={
                "steps": 50,
                "seed": 99,
                "min_sharpness_gain": 1.25,
            },
            implementation_identity={"sharding": "implementation-sha"},
        )
        with self.assertRaisesRegex(ValueError, "runtime identity differs"):
            self._write_worker(0, runtime_identity=mismatched)

        extended = runtime_identity_from_contract(self.contract)
        extended["unbound_worker_value"] = "must-not-be-ignored"
        with self.assertRaisesRegex(ValueError, "fields differ from its schema"):
            self._write_worker(1, runtime_identity=extended)

    def test_merge_exact_partition_preserves_provenance_and_artifact_paths(
        self,
    ) -> None:
        envelopes = [self._write_worker(0), self._write_worker(1)]
        output = self.root / "merged"
        report_path, manifest_path = merge_shard_envelopes(envelopes, output)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [batch["batch_id"] for batch in report["generation_batches"]],
            [batch["batch_id"] for batch in self.batches],
        )
        self.assertEqual(report["planned_target_count"], len(self.planned))
        self.assertEqual(report["generation_batch_count"], len(self.batches))
        self.assertEqual(report["shard_merge"]["shard_count"], 2)
        self.assertEqual(
            report["planned"][0]["conditioning"][0]["role"], "sharp_before"
        )
        synthetic_paths = [
            frame["rgb_path"]
            for frame in manifest["frames"]
            if frame["kind"] == "synthetic"
        ]
        self.assertTrue(synthetic_paths)
        self.assertTrue(all("worker_" in path for path in synthetic_paths))
        self.assertFalse(any(str(output) in path for path in synthetic_paths))
        self.assertEqual(
            sorted(path.name for path in output.iterdir()),
            sorted((report_path.name, manifest_path.name)),
        )
        # Deterministic immutable publication is idempotent.
        self.assertEqual(
            (report_path, manifest_path), merge_shard_envelopes(envelopes, output)
        )

    def test_sharp_mode_quality_partition_does_not_double_count_geometry_only(
        self,
    ) -> None:
        batch = dict(self.batches[0])
        plan = dict(self.planned[0])
        contract = build_shard_contract(
            [batch],
            [plan],
            shard_count=1,
            runtime_identity=build_shard_runtime_identity(
                source_identity={"stream": "fixture"},
                model_artifact_identity={"framecrafter": "fixture"},
                semantic_config={"acceptance_mode": "sharp"},
                implementation_identity={"sharding": "fixture"},
            ),
        )
        worker = self.root / "double-gate-worker"
        worker.mkdir()
        poses = worker / "poses.npz"
        poses.write_bytes(b"poses")
        batch.update(poses_npz=str(poses), poses_npz_sha256=file_sha256(poses))
        plan.update(poses_npz=str(poses), poses_npz_sha256=file_sha256(poses))
        candidate = worker / "geometry-only.png"
        candidate.write_bytes(b"geometry-only")
        geometry_only = {
            "target_id": plan["target_id"],
            "batch_id": batch["batch_id"],
            "batch_target_ids": list(batch["target_ids"]),
            "batch_target_position": 0,
            "acceptance_class": "geometry_only",
            "failures": ["sharpness_gain"],
            "candidate_rgb_path": str(candidate),
            "candidate_rgb_sha256": file_sha256(candidate),
        }
        manifest_path = write_json(
            worker / "manifest.json",
            {
                "schema": "unblur_slam.framecrafter_manifest.v1",
                "source_frame_count": 1,
                "generated_frame_count": 0,
                "pose_source": "droid_traj_est_not_align",
                "uses_ground_truth_pose": False,
                "frames": [self.original],
            },
        )
        report_path = write_json(
            worker / "report.json",
            {
                "schema": "unblur_slam.framecrafter_preprocess_report.v1",
                "backend": "python_api",
                "backend_test_only": False,
                "uses_ground_truth_pose": False,
                "pose_source": "droid_traj_est_not_align",
                "shard_runtime_identity": runtime_identity_from_contract(contract),
                "shard": {
                    "schema": "unblur_slam.framecrafter_worker_shard.v1",
                    "experiment_signature": contract["experiment_signature"],
                    "canonical_preprocess_signature": contract[
                        "canonical_preprocess_signature"
                    ],
                    "runtime_identity_sha256": canonical_sha256(
                        runtime_identity_from_contract(contract)
                    ),
                    "assignment_table_sha256": contract[
                        "assignment_table_sha256"
                    ],
                    "global_plan_sha256": contract["global_plan_sha256"],
                    "shard_index": 0,
                    "shard_count": 1,
                    "assigned_batch_ids": list(assigned_batch_ids(contract, 0)),
                },
                "acceptance_mode": "sharp",
                "source_frame_count": 1,
                "planned_total_before_cap": 1,
                "planned_target_count": 1,
                "selected_target_count": 1,
                "generation_batch_count": 1,
                "backend_generate_call_count": 1,
                "accepted_target_count": 0,
                "rejected_target_count": 1,
                "sharp_accepted_target_count": 0,
                "geometry_only_target_count": 1,
                "geometry_rejected_target_count": 0,
                "accepted_output_sha256": canonical_sha256([]),
                "source_input_sha256": canonical_sha256([]),
                "manifest": str(manifest_path),
                "generation_batches": [batch],
                "planned": [plan],
                "accepted": [],
                # This is the injection-policy rejection view.
                "rejected": [geometry_only],
                # This is the disjoint scientific outcome partition.
                "quality_partition": {
                    "sharp_accepted": [],
                    "geometry_only": [geometry_only],
                    "rejected": [],
                },
            },
        )
        envelope = write_shard_envelope(
            contract,
            shard_index=0,
            report_path=report_path,
            manifest_path=manifest_path,
            output_path=worker / "envelope.json",
        )
        merged_report_path, _ = merge_shard_envelopes(
            [envelope], self.root / "double-gate-merged"
        )
        merged = json.loads(merged_report_path.read_text(encoding="utf-8"))
        self.assertEqual(merged["accepted_target_count"], 0)
        self.assertEqual(merged["rejected_target_count"], 1)
        self.assertEqual(merged["geometry_only_target_count"], 1)
        self.assertEqual(merged["geometry_rejected_target_count"], 0)
        self.assertEqual(len(merged["rejected"]), 1)
        self.assertEqual(len(merged["quality_partition"]["geometry_only"]), 1)
        self.assertEqual(len(merged["quality_partition"]["rejected"]), 0)

    def test_merger_rejects_raw_report_missing_shard_and_tampered_artifact(
        self,
    ) -> None:
        envelope0 = self._write_worker(0)
        report0 = self.root / "worker_0" / "report.json"
        with self.assertRaisesRegex(ValueError, "shard envelope"):
            merge_shard_envelopes([report0], self.root / "raw_rejected")
        with self.assertRaisesRegex(ValueError, "incomplete shard set"):
            merge_shard_envelopes([envelope0], self.root / "missing_rejected")

        envelope1 = self._write_worker(1)
        report = json.loads(report0.read_text(encoding="utf-8"))
        artifact = Path(report["generation_batches"][0]["poses_npz"])
        artifact.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            merge_shard_envelopes(
                [envelope0, envelope1], self.root / "tampered_rejected"
            )

    def test_merger_rejects_mismatched_experiment_contract(self) -> None:
        envelope0 = self._write_worker(0)
        other_contract = build_shard_contract(
            self.batches,
            self.planned,
            shard_count=2,
            runtime_identity=build_shard_runtime_identity(
                source_identity={
                    "rgb": file_sha256(self.source_rgb),
                    "depth": file_sha256(self.source_depth),
                },
                model_artifact_identity={"framecrafter": "different-model"},
                semantic_config={
                    "steps": 20,
                    "seed": 43,
                    "height": 480,
                    "width": 832,
                },
                implementation_identity={"sharding": "implementation-sha"},
            ),
        )
        original_contract = self.contract
        self.contract = other_contract
        try:
            envelope1 = self._write_worker(1)
        finally:
            self.contract = original_contract
        with self.assertRaisesRegex(ValueError, "contracts differ"):
            merge_shard_envelopes(
                [envelope0, envelope1], self.root / "mismatch_rejected"
            )


if __name__ == "__main__":
    unittest.main()
