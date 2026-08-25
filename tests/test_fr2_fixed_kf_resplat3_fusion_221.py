#!/usr/bin/env python3
"""CPU-only contracts for fixed-11KF EVSSM + active ReSplat-state3."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _module(
    "fusion221_cpu_test_runner", "scripts/run_fr2_fixed_kf_resplat3_fusion_221.py"
)
REPORT = _module(
    "fusion221_cpu_test_report", "scripts/report_fr2_fixed_kf_resplat3_fusion_221.py"
)

from src.refinement.official_resplat_active_fusion import (  # noqa: E402
    ActiveFusionConfig,
    context_reconstruction_gate,
    postmerge_reconstruction_gate,
)
from src.refinement.official_resplat_sidecar import (  # noqa: E402
    SidecarFrameInput,
    load_snapshot,
    materialize_closed_submap_snapshot,
)


def _fusion_config() -> ActiveFusionConfig:
    _configs, validation = RUNNER._load_and_validate_configs()
    return validation["contracts"]["fused"]["fusion"]


def _expect_rejected(raw: dict, fragment: str) -> None:
    try:
        ActiveFusionConfig.from_dict(raw, default_output_root="/tmp/fusion221-test")
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"active-fusion mutation was accepted: {fragment}")


def test_config_pair_and_every_numeric_gate_are_frozen() -> None:
    configs, validation = RUNNER._load_and_validate_configs()
    assert set(validation["differences"]) == RUNNER.ALLOWED_PAIR_DIFFS
    assert set(validation["differences"]["data.output"]) == {"baseline", "fused"}
    cfg = validation["contracts"]["fused"]["fusion"]
    assert tuple(cfg.expected_mapped_source_indices) == (
        15,
        49,
        58,
        72,
        89,
        109,
        125,
        166,
    )
    assert cfg.selection_membership_clear_gt_conditioned is True
    assert RUNNER.OUTPUT_ROOT.name.endswith("_v2")
    assert REPORT.DEFAULT_ROOT == RUNNER.OUTPUT_ROOT
    assert all("_v2" in str(path) for path in RUNNER.CONFIGS.values())
    for arm in ("baseline", "fused"):
        disclosure = configs[arm]["fixed_kf_resplat3_active_fusion_221"]
        assert disclosure["schema"].endswith(".v2")
        assert disclosure["mapped_viewpoint_binding_contract"] == (
            "captured_before_pose_window_loop"
        )
    assert configs["baseline"]["deblur"]["frontend"] == "evssm"
    assert configs["fused"]["deblur"]["frontend"] == "evssm"

    mutations = []
    for group, field, replacement in (
        ("geometry_gate", "max_p95_distance_from_pivot", 49.0),
        ("sidecar_quality_gate", "minimum_state3_mean_psnr_db", 11.9),
        ("sidecar_quality_gate", "minimum_state3_mean_ssim", 0.49),
        ("sidecar_quality_gate", "maximum_state3_mean_l1", 0.31),
        ("merge", "min_opacity", 0.49),
        ("merge", "min_scale", 2e-6),
        ("merge", "max_scale", 0.21),
        ("merge", "max_abs_position", 201.0),
        ("postmerge_quality_gate", "l1_weight", 0.79),
        (
            "postmerge_quality_gate",
            "maximum_relative_mean_composite_increase",
            0.006,
        ),
        (
            "postmerge_quality_gate",
            "maximum_absolute_per_view_composite_increase",
            0.021,
        ),
    ):
        raw = asdict(cfg)
        raw[group] = dict(raw[group])
        raw[group][field] = replacement
        mutations.append(raw)
    for raw in mutations:
        _expect_rejected(raw, "drifted")
    raw = asdict(cfg)
    raw["trigger_source_index"] = 220
    _expect_rejected(raw, "source index 166")
    raw = asdict(cfg)
    raw["expected_mapped_source_indices"] = [49, 58, 72, 89, 109, 125, 166, 220]
    _expect_rejected(raw, "first eight actually mapped")
    raw = asdict(cfg)
    raw["selection_membership_clear_gt_conditioned"] = False
    _expect_rejected(raw, "clear-GT-conditioned")


def _metrics(stage: str, *, psnr: float, ssim: float, l1: float) -> dict:
    del stage
    return {
        "mean_psnr_db": psnr,
        "mean_ssim": ssim,
        "mean_l1": l1,
        "mean_masked_l1": l1,
        "per_view": [
            {
                "frame_id": index + 10,
                "sequence_ordinal": index,
                "psnr_db": psnr,
                "ssim": ssim,
                "l1": l1,
                "masked_l1": l1,
            }
            for index in range(8)
        ],
    }


def _context_manifest() -> tuple[dict, dict]:
    snapshot = {
        "frames": [
            {"frame_id": index + 10, "sequence_ordinal": index}
            for index in range(8)
        ]
    }
    return (
        {
            "context_reconstruction": {
                "uses_clear_gt": False,
                "inputs": "eight_context_observations",
                "same_observations_for_init0_and_state3": True,
                "init0": _metrics("init0", psnr=20.0, ssim=0.80, l1=0.10),
                "state3": _metrics("state3", psnr=21.0, ssim=0.82, l1=0.09),
            }
        },
        snapshot,
    )


def test_context_gate_recomputes_means_and_binds_snapshot_identity() -> None:
    cfg = _fusion_config()
    manifest, snapshot = _context_manifest()
    accepted = context_reconstruction_gate(manifest, cfg, snapshot)
    assert accepted["accepted"] is True, accepted

    tampered = json.loads(json.dumps(manifest))
    tampered["context_reconstruction"]["state3"]["mean_psnr_db"] = 99.0
    result = context_reconstruction_gate(tampered, cfg, snapshot)
    assert result["accepted"] is False
    assert "state3_mean_psnr_db_does_not_match_per_view" in result["reasons"]

    reordered = json.loads(json.dumps(manifest))
    reordered["context_reconstruction"]["state3"]["per_view"].reverse()
    result = context_reconstruction_gate(reordered, cfg, snapshot)
    assert result["accepted"] is False
    assert "init0_state3_view_identity_or_order_mismatch" in result["reasons"]

    unbound = json.loads(json.dumps(manifest))
    unbound["context_reconstruction"]["same_observations_for_init0_and_state3"] = False
    result = context_reconstruction_gate(unbound, cfg, snapshot)
    assert result["accepted"] is False
    assert "init0_state3_observation_binding_missing" in result["reasons"]


def test_postmerge_gate_is_fail_closed() -> None:
    cfg = _fusion_config()
    before = {
        "mean_composite": 0.10,
        "per_view": [
            {"source_index": source, "composite": 0.10}
            for source in RUNNER.EXPECTED_FIRST_EIGHT_MAPPED_SOURCES
        ],
    }
    accepted_after = {
        "mean_composite": 0.1004,
        "per_view": [
            {"source_index": source, "composite": 0.1004}
            for source in RUNNER.EXPECTED_FIRST_EIGHT_MAPPED_SOURCES
        ],
    }
    assert postmerge_reconstruction_gate(before, accepted_after, cfg)["accepted"] is True
    rejected_after = json.loads(json.dumps(accepted_after))
    rejected_after["mean_composite"] = 0.101
    rejected_after["per_view"][0]["composite"] = 0.13
    decision = postmerge_reconstruction_gate(before, rejected_after, cfg)
    assert decision["accepted"] is False
    assert set(decision["reasons"]) == {
        "relative_mean_composite_increase_exceeded",
        "per_view_composite_increase_exceeded",
    }


def _frame(index: int) -> SidecarFrameInput:
    return SidecarFrameInput(
        frame_id=index + 100,
        sequence_ordinal=index,
        c2w=np.eye(4, dtype=np.float32),
        intrinsics_px=((10.0, 0.0, 4.0), (0.0, 10.0, 4.0), (0.0, 0.0, 1.0)),
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def test_conditioned_membership_is_disclosed_but_clear_inputs_are_forbidden() -> None:
    with tempfile.TemporaryDirectory() as directory:
        frames = [_frame(index) for index in range(8)]
        provenance = {
            "selection_membership_clear_gt_conditioned": True,
            "uses_ground_truth_pose_or_depth": False,
            "uses_independent_clear_pixels": False,
            "uses_clear_gt_metrics": False,
        }
        output = materialize_closed_submap_snapshot(
            snapshots_root=directory,
            submap_id=0,
            record_keyframe_ids=[frame.frame_id for frame in frames],
            frames=frames,
            closure_sequence_ordinal=7,
            pose_revision=1,
            source_provenance=provenance,
            uses_clear_gt_membership=True,
            uses_independent_clear_pixels=False,
        )
        snapshot = load_snapshot(output)
        assert snapshot["selection_membership_clear_gt_conditioned"] is True
        assert snapshot["uses_clear_gt_membership"] is True
        assert snapshot["uses_independent_clear_pixels"] is False

        try:
            materialize_closed_submap_snapshot(
                snapshots_root=Path(directory) / "bad",
                submap_id=0,
                record_keyframe_ids=[frame.frame_id for frame in frames],
                frames=frames,
                closure_sequence_ordinal=7,
                pose_revision=1,
                source_provenance={},
                uses_clear_gt_membership=True,
            )
        except ValueError as error:
            assert "conditioned-membership provenance" in str(error)
        else:
            raise AssertionError("conditioned membership without provenance was accepted")


def test_real_71680_bridge_probe_and_lexical_venv_contract() -> None:
    cfg = _fusion_config()
    if RUNNER.REAL_BRIDGE_PROBE_NATIVE.is_file():
        probe = RUNNER._real_artifact_bridge_probe(cfg)
        assert probe["accepted"] is True
        assert probe["gaussian_count"] == 71_680
        assert probe["world_harmonics_shape"] == [71_680, 3, 1]
        assert probe["capped_candidate_count_before_active_map_collision"] >= 1_024
    inspection = RUNNER._inspect_resplat(cfg)
    assert inspection["python_executable"] == cfg.python_executable
    assert inspection["python_executable_realpath"] != inspection["python_executable"]
    assert inspection["python_executable_symlink_preserved_lexically"] is True


def test_mapper_orchestration_contains_all_safety_hooks() -> None:
    source = (ROOT / "src/mapper.py").read_text(encoding="utf-8")
    for fragment in (
        "expected_sources = tuple(int(value) for value in cfg.expected_mapped_source_indices)",
        "active ReSplat fusion trigger drifted from mapped source-166",
        "torch.exp(viewpoint.exposure_a.detach()) * raw_prediction",
        "world artifact SH dimension must exactly match the active map",
        "official_resplat_commit_mismatch",
        '"xyz_gradient_accum"',
        '"denom"',
        '"max_radii2D"',
        '"byte_identical_to_premerge": True',
        "fusion_final_contract.json",
        "downstream_online_mapped_source_indices_after_fusion",
        "mapped_viewpoint is not self.cameras[video_idx]",
        "int(mapped_viewpoint.uid) != int(video_idx)",
        "int(mapped_viewpoint.timestamp) != int(idx)",
    ):
        assert fragment in source, fragment
    binding = RUNNER.validate_mapper_hook_binding()
    assert binding["accepted"] is True
    assert binding["hook_argument"] == "mapped_viewpoint"
    assert binding["capture_line"] < binding["pose_window_rebind_lines"][0]
    assert binding["pose_window_rebind_lines"][-1] < binding["prune_mapping_line"]
    assert binding["prune_mapping_line"] < binding["regular_fusion_hook_line"]
    regressed = source.replace(
        "self._active_fusion_after_mapped_keyframe(mapped_viewpoint)",
        "self._active_fusion_after_mapped_keyframe(viewpoint)",
        1,
    )
    try:
        RUNNER.validate_mapper_hook_binding(regressed)
    except ValueError as error:
        assert "must consume mapped_viewpoint exactly once" in str(error)
    else:
        raise AssertionError("AST contract accepted the v1 window-variable regression")
    ordinary = source.index("self.map(self.current_window, prune=True)")
    hook = source.index(
        "self._active_fusion_after_mapped_keyframe(mapped_viewpoint)", ordinary
    )
    timer = source.index("online_inference_time", hook)
    assert ordinary < hook < timer


def test_report_accepts_explicit_sidecar_rejection_but_not_internal_error() -> None:
    with tempfile.TemporaryDirectory() as directory:
        scene = Path(directory)
        fusion_root = scene / "official_resplat_active_fusion"
        fusion_root.mkdir()
        audit = {
            "schema": "unblur_slam.official_resplat_active_fusion_audit.v1",
            "status": "sidecar_rejected",
            "active_map_changed_final": False,
            "data_lineage": {
                "selection_membership_clear_gt_conditioned": True,
                "ground_truth_poses_or_depths_consumed_by_fusion": False,
                "independent_clear_pixels_consumed_by_fusion": False,
                "clear_gt_metrics_consumed_by_fusion": False,
            },
            "trigger": {
                "after_fully_mapped_keyframe_count": 8,
                "source_index": 166,
                "source_indices": list(RUNNER.EXPECTED_FIRST_EIGHT_MAPPED_SOURCES),
            },
            "official_state": {
                "requested_recurrent_updates": 3,
                "selected_state_index_zero_based": 2,
                "fourth_state_computed": False,
            },
            "timing": {
                "snapshot_seconds": 0.1,
                "subprocess_and_publication_seconds": 1.0,
                "premerge_active_render_seconds": 0.0,
                "merge_seconds": 0.0,
                "postmerge_active_render_seconds": 0.0,
                "rollback_seconds": 0.0,
                "total_wall_seconds": 1.1,
            },
            "rejection_reasons": ["official_state3_sidecar_not_published_exactly_once"],
        }
        audit_path = fusion_root / "fusion_audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        final = {
            "schema": "unblur_slam.official_resplat_active_fusion_final_contract.v1",
            "fusion_attempt_count": 1,
            "actually_mapped_source_indices": list(
                RUNNER.EXPECTED_FIRST_EIGHT_MAPPED_SOURCES
                + RUNNER.EXPECTED_DOWNSTREAM_MAPPED_SOURCES
            ),
            "trigger_context_source_indices": list(
                RUNNER.EXPECTED_FIRST_EIGHT_MAPPED_SOURCES
            ),
            "downstream_online_mapped_source_indices_after_fusion": list(
                RUNNER.EXPECTED_DOWNSTREAM_MAPPED_SOURCES
            ),
            "fusion_completed_before_downstream_online_mapping": True,
            "fusion_status": "sidecar_rejected",
            "fusion_committed_to_active_map_before_downstream_mapping": False,
            "imported_gaussian_survival_in_serialized_final_model_not_separately_tracked": True,
            "fusion_audit_sha256": REPORT.BASE._sha256_file(audit_path),
        }
        (fusion_root / "fusion_final_contract.json").write_text(
            json.dumps(final), encoding="utf-8"
        )
        result = REPORT._read_fusion(scene, "fused")
        assert result["status"] == "sidecar_rejected"
        assert (
            result["fusion_committed_to_active_map_before_downstream_mapping"]
            is False
        )

        audit["status"] = "error_rejected"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        try:
            REPORT._read_fusion(scene, "fused")
        except REPORT.ContractError as error:
            assert "valid fail-closed terminal status" in str(error)
        else:
            raise AssertionError("internal fusion error was accepted as comparable result")

        audit["status"] = "accepted"
        audit["active_map_changed_final"] = True
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        final["fusion_status"] = "accepted"
        final["fusion_committed_to_active_map_before_downstream_mapping"] = True
        final["fusion_audit_sha256"] = REPORT.BASE._sha256_file(audit_path)
        (fusion_root / "fusion_final_contract.json").write_text(
            json.dumps(final), encoding="utf-8"
        )
        try:
            REPORT._read_fusion(scene, "fused")
        except REPORT.ContractError as error:
            assert "lacks its bound published manifest" in str(error)
        else:
            raise AssertionError("accepted fusion without manifest/gates was reported")


if __name__ == "__main__":
    test_config_pair_and_every_numeric_gate_are_frozen()
    test_context_gate_recomputes_means_and_binds_snapshot_identity()
    test_postmerge_gate_is_fail_closed()
    test_conditioned_membership_is_disclosed_but_clear_inputs_are_forbidden()
    test_real_71680_bridge_probe_and_lexical_venv_contract()
    test_mapper_orchestration_contains_all_safety_hooks()
    test_report_accepts_explicit_sidecar_rejection_but_not_internal_error()
    print("fixed_kf_resplat3_active_fusion_221_cpu_contracts=PASS")
