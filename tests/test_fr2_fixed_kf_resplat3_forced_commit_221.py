#!/usr/bin/env python3
"""CPU-only contracts for the unsafe v3 forced-commit diagnostic."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load(
    "test_forced_commit_runner",
    ROOT / "scripts/run_fr2_fixed_kf_resplat3_forced_commit_221.py",
)
REPORT = _load(
    "test_forced_commit_report",
    ROOT / "scripts/report_fr2_fixed_kf_resplat3_forced_commit_221.py",
)


def _raises(error_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_v3_config_is_count_only_and_all_numeric_gates_equal_v2() -> None:
    v3, v2 = RUNNER._load_configs()
    fusion, differences = RUNNER._validate_v3_config(v3, v2)
    assert set(differences) == RUNNER.ALLOWED_V3_V2_CONFIG_DIFFERENCES
    assert fusion.expected_forced_commit_gaussian_count == 5554
    assert fusion.posthoc_after_v2_rejection is True
    assert fusion.unsafe_not_deployable is True
    assert fusion.gate_thresholds_unchanged is True
    assert fusion.force_commit_after_postmerge_rejection is True
    left = v2["mapping"]["official_resplat_active_fusion"]
    right = v3["mapping"]["official_resplat_active_fusion"]
    for field in (
        "geometry_gate",
        "sidecar_quality_gate",
        "merge",
        "postmerge_quality_gate",
    ):
        assert left[field] == right[field]
    disclosure = v3["fixed_kf_resplat3_active_fusion_221"]
    assert disclosure["fresh_append_required_to_reproduce_v2_accepted_count"] == 5554
    assert "candidate_identity" not in disclosure
    common = (
        ROOT
        / "configs/local/fr2_xyz_fixed_kf_resplat3_fusion_221_v3_forced_commit/common.yaml"
    ).read_text(encoding="utf-8")
    assert "fresh append with accepted count 5554" in common
    assert "Candidate identity equality" in common
    assert "same 5554-candidate" not in common


def test_gate_or_merge_numeric_drift_is_rejected() -> None:
    v3, v2 = RUNNER._load_configs()
    cases = (
        ("geometry_gate", "max_distance_from_pivot", 201.0),
        ("sidecar_quality_gate", "minimum_state3_mean_psnr_db", 11.0),
        ("merge", "min_opacity", 0.49),
        (
            "postmerge_quality_gate",
            "maximum_relative_mean_composite_increase",
            0.006,
        ),
    )
    for section, field, value in cases:
        mutated = copy.deepcopy(v3)
        mutated["mapping"]["official_resplat_active_fusion"][section][field] = value
        _raises(ValueError, RUNNER._validate_v3_config, mutated, v2)


def test_diagnostic_flags_are_all_or_nothing_and_v2_still_validates() -> None:
    v3, v2 = RUNNER._load_configs()
    for field in (
        "posthoc_after_v2_rejection",
        "unsafe_not_deployable",
        "gate_thresholds_unchanged",
        "force_commit_after_postmerge_rejection",
    ):
        mutated = copy.deepcopy(v3)
        mutated["mapping"]["official_resplat_active_fusion"][field] = False
        _raises(ValueError, RUNNER._validate_v3_config, mutated, v2)
    from src.refinement.official_resplat_active_fusion import ActiveFusionConfig

    ordinary = ActiveFusionConfig.from_dict(
        v2["mapping"]["official_resplat_active_fusion"],
        default_output_root=Path("/tmp/ordinary-v2-active-fusion"),
    )
    assert ordinary.posthoc_after_v2_rejection is False
    assert ordinary.expected_forced_commit_gaussian_count == 0


def test_canonical_event_hash_fails_closed_on_mutation() -> None:
    from src.refinement.official_resplat_active_fusion import (
        canonical_contract_sha256,
        stamp_contract_sha256,
    )

    event = stamp_contract_sha256(
        {
            "schema": "unblur_slam.forced_commit_chain_event.v1",
            "event_type": "postgate_rejected_forced_commit",
            "sequence": 0,
            "previous_event_sha256": None,
        }
    )
    assert event["event_sha256"] == canonical_contract_sha256(event)
    mutated = dict(event)
    mutated["sequence"] = 1
    assert mutated["event_sha256"] != canonical_contract_sha256(mutated)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "event.json"
        path.write_text(json.dumps(event), encoding="utf-8")
        assert REPORT._verify_event(
            path, 0, "postgate_rejected_forced_commit"
        ) == event
        path.write_text(json.dumps(mutated), encoding="utf-8")
        _raises(
            REPORT.ContractError,
            REPORT._verify_event,
            path,
            1,
            "postgate_rejected_forced_commit",
        )


def test_mapper_source220_and_final100_hook_order_is_ast_guarded() -> None:
    mapper_path = ROOT / "src/mapper.py"
    slam_path = ROOT / "src/slam.py"
    mapper = mapper_path.read_text(encoding="utf-8")
    slam = slam_path.read_text(encoding="utf-8")
    record = RUNNER.validate_forced_commit_instrumentation(mapper, slam)
    assert record["accepted"] is True
    assert record[
        "source220_entry_before_frame_metadata_pose_depth_deformation_mapping"
    ] is True

    call = "            self._forced_commit_source220_entry(int(idx), int(video_idx))\n"
    assert mapper.count(call) == 1
    moved_before_end_guard = mapper.replace(call, "", 1).replace(
        "            if is_finished:\n",
        call + "            if is_finished:\n",
        1,
    )
    _raises(
        ValueError,
        RUNNER.validate_forced_commit_instrumentation,
        moved_before_end_guard,
        slam,
    )

    entry = "        self._forced_commit_final100_entry(iters)\n"
    hydrate = "        self._hydrate_missing_droid_keyframes_for_final_refine()\n"
    assert mapper.count(entry) == 1 and mapper.count(hydrate) == 1
    moved_after_hydration = mapper.replace(entry, "", 1).replace(
        hydrate, hydrate + entry, 1
    )
    _raises(
        ValueError,
        RUNNER.validate_forced_commit_instrumentation,
        moved_after_hydration,
        slam,
    )

    save = "                self.mapper.gaussians.save_ply(ply_path)\n"
    terminal = "                self.mapper.finalize_forced_commit_terminal(ply_path)\n"
    assert slam.count(save) == 1 and slam.count(terminal) == 1
    reversed_terminal = slam.replace(save + terminal, terminal + save, 1)
    _raises(
        ValueError,
        RUNNER.validate_forced_commit_instrumentation,
        mapper,
        reversed_terminal,
    )


def test_force_status_is_nested_only_under_explicit_postgate_rejection() -> None:
    source = (ROOT / "src/mapper.py").read_text(encoding="utf-8")
    outer = source.index('            if not post_gate["accepted"]:')
    accepted_guard = source.index(
        "            if cfg.force_commit_after_postmerge_rejection:", outer + 1
    )
    forced_status = source.index(
        '"postmerge_gate_rejected_forced_commit_unsafe"', outer
    )
    ordinary_accept = source.index('            audit["status"] = "accepted"', outer)
    assert outer < forced_status < ordinary_accept
    assert outer < accepted_guard < ordinary_accept
    segment = source[outer:ordinary_accept]
    assert "posthoc_v2_rejection_not_reproduced_rolled_back" in segment
    assert "postmerge_gate_accepted_so_forced_commit_not_authorized" in segment
    assert '"rollback_performed": False' in segment
    assert '"byte_identical_to_trial": True' in segment
    assert segment.rindex("committed = True") > segment.index(
        'audit["active_state_final"]'
    )


def test_terminal_and_report_never_claim_individual_survival() -> None:
    mapper = (ROOT / "src/mapper.py").read_text(encoding="utf-8")
    report = (
        ROOT / "scripts/report_fr2_fixed_kf_resplat3_forced_commit_221.py"
    ).read_text(encoding="utf-8")
    assert '"world_bridge"' in mapper
    assert '"individual_imported_gaussian_survival_in_final_model_not_claimed": True' in mapper
    assert "individual_imported_gaussian_survival_not_tracked_or_claimed" in report
    assert "fresh_candidate_identity_equal_to_v2_not_claimed" in report
    assert "fair paired estimate" in report


def test_report_premerge_and_postmerge_tampering_fails_closed() -> None:
    v3, _ = RUNNER._load_configs()
    fusion_raw = v3["mapping"]["official_resplat_active_fusion"]
    from src.refinement.official_resplat_active_fusion import (
        ActiveFusionConfig,
        postmerge_reconstruction_gate,
    )

    fusion = ActiveFusionConfig.from_dict(
        fusion_raw, default_output_root=Path("/tmp/forced-report-test")
    )
    with tempfile.TemporaryDirectory() as temporary:
        result_root = Path(temporary).resolve()
        manifest_path = result_root / "run_manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        audit = {
            "official_state": {
                "requested_recurrent_updates": 3,
                "selected_state_index_zero_based": 2,
                "fourth_state_computed": False,
            },
            "premerge_gates": {
                "world_artifact": {"accepted": True},
                "context_reconstruction": {
                    "accepted": True,
                    "uses_clear_gt": False,
                },
                "repository_provenance": {
                    "accepted": True,
                    "expected_commit": RUNNER.V2.EXPECTED_RESPLAT_COMMIT,
                    "observed_commit": RUNNER.V2.EXPECTED_RESPLAT_COMMIT,
                },
                "data_lineage": {
                    "accepted": True,
                    "selection_membership_clear_gt_conditioned": True,
                    "ground_truth_pose_or_depth_used": False,
                    "independent_clear_pixels_used": False,
                    "clear_gt_metrics_used": False,
                },
            },
            "published_result": {
                "path": str(result_root),
                "manifest_path": str(manifest_path),
                "manifest_sha256": RUNNER._sha256_file(manifest_path),
            },
            "sh_import": {
                "native_harmonic_dimension": 16,
                "active_harmonic_dimension": 1,
                "dropped_higher_order_coefficients": 15,
                "mapper_side_truncation_performed": False,
                "dc_only": True,
            },
            "merge": {
                "config": copy.deepcopy(fusion_raw["merge"]),
                "optimizer_state_shapes_valid": True,
                "active_map_changed": True,
            },
        }
        valid = REPORT._validate_forced_premerge_contract(
            audit, v3, manifest_path
        )
        assert valid["manifest_path"] == manifest_path
        for gate in audit["premerge_gates"]:
            mutated = copy.deepcopy(audit)
            mutated["premerge_gates"][gate]["accepted"] = False
            _raises(
                REPORT.ContractError,
                REPORT._validate_forced_premerge_contract,
                mutated,
                v3,
                manifest_path,
            )
        mutated = copy.deepcopy(audit)
        mutated["published_result"]["manifest_sha256"] = "0" * 64
        _raises(
            REPORT.ContractError,
            REPORT._validate_forced_premerge_contract,
            mutated,
            v3,
            manifest_path,
        )
        mutated = copy.deepcopy(audit)
        mutated["sh_import"]["active_harmonic_dimension"] = 16
        _raises(
            REPORT.ContractError,
            REPORT._validate_forced_premerge_contract,
            mutated,
            v3,
            manifest_path,
        )
        mutated = copy.deepcopy(audit)
        mutated["merge"]["config"]["min_opacity"] = 0.49
        _raises(
            REPORT.ContractError,
            REPORT._validate_forced_premerge_contract,
            mutated,
            v3,
            manifest_path,
        )

    before_views = [
        {"source_index": source, "composite": 0.05}
        for source in RUNNER.EXPECTED_CONTEXT
    ]
    after_views = [
        {"source_index": source, "composite": 0.08}
        for source in RUNNER.EXPECTED_CONTEXT
    ]
    before = {"mean_composite": 0.05, "per_view": before_views}
    after = {"mean_composite": 0.08, "per_view": after_views}
    decision = postmerge_reconstruction_gate(before, after, fusion)
    assert decision["accepted"] is False
    postmerge = {"before": before, "after": after, "decision": decision}
    assert REPORT._validate_recorded_postmerge_decision(
        postmerge, fusion
    ) == decision
    for mutation in ("accepted", "reasons", "relative_mean_composite_change"):
        tampered = copy.deepcopy(postmerge)
        if mutation == "accepted":
            tampered["decision"][mutation] = True
        elif mutation == "reasons":
            tampered["decision"][mutation] = []
        else:
            tampered["decision"][mutation] = 0.0
        _raises(
            REPORT.ContractError,
            REPORT._validate_recorded_postmerge_decision,
            tampered,
            fusion,
        )

    _raises(
        REPORT.ContractError,
        REPORT._validate_terminal_code_bindings,
        {},
        {"implementation_provenance": {"files": {}}},
    )
    terminal = {
        "final_in_memory_state": {
            "gaussian_count": 77762,
            "full_active_state_sha256": "a" * 64,
        }
    }
    final100 = {
        "post_refinement_state": {
            "gaussian_count": 77762,
            "full_active_state_sha256": "a" * 64,
        }
    }
    REPORT._validate_terminal_final_state_link(terminal, final100)
    tampered_terminal = copy.deepcopy(terminal)
    tampered_terminal["final_in_memory_state"]["full_active_state_sha256"] = (
        "b" * 64
    )
    _raises(
        REPORT.ContractError,
        REPORT._validate_terminal_final_state_link,
        tampered_terminal,
        final100,
    )


def test_frozen_v2_baseline_and_rollback_tree_bindings() -> None:
    v3, _ = RUNNER._load_configs()
    record = RUNNER._validate_frozen_v2_references(v3)
    assert record["baseline"]["tree_sha256"] == RUNNER.V2_BASELINE_TREE_SHA256
    assert (
        record["rollback_arm"]["tree_sha256"]
        == RUNNER.V2_ROLLBACK_TREE_SHA256
    )
    assert record["v2_rollback_fusion_audit"]["accepted_count"] == 5554
    assert record["v2_rollback_fusion_audit"]["postmerge_gate_accepted"] is False
    assert record["reference_semantics"]["fresh_pair_rerun"] is False
    assert record["reference_semantics"]["consumed_by_v3_slam_or_resplat_runtime"] is False


def test_real_world_and_snapshot_three_way_lineage_tampering_fails() -> None:
    audit = json.loads(RUNNER.V2_ROLLBACK_AUDIT.read_text(encoding="utf-8"))
    manifest_path = Path(audit["published_result"]["manifest_path"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest["outputs"]
    artifact_paths = {
        "official_resplat_manifest": manifest_path,
        "official_resplat_native_gaussians": (
            manifest_path.parent / outputs["native_gaussians_npz"]
        ).resolve(),
        "official_resplat_world_gaussians": (
            manifest_path.parent / outputs["unblur_world_gaussians_npz"]
        ).resolve(),
        "snapshot_manifest": (
            Path(audit["snapshot"]["path"]) / "snapshot_manifest.json"
        ).resolve(),
    }
    lineage = REPORT._validate_resplat_artifact_lineage(
        audit, manifest, artifact_paths
    )
    assert lineage["world_sha256"] == outputs[
        "unblur_world_gaussians_npz_sha256"
    ]
    assert lineage["source_pose_revision"] == manifest["source_pose_revision"]

    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["outputs"]["unblur_world_gaussians_npz_sha256"] = "0" * 64
    _raises(
        REPORT.ContractError,
        REPORT._validate_resplat_artifact_lineage,
        audit,
        tampered_manifest,
        artifact_paths,
    )
    tampered_audit = copy.deepcopy(audit)
    tampered_audit["premerge_gates"]["world_artifact"]["measurements"][
        "world_artifact_sha256"
    ] = "0" * 64
    _raises(
        REPORT.ContractError,
        REPORT._validate_resplat_artifact_lineage,
        tampered_audit,
        manifest,
        artifact_paths,
    )
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["source_pose_revision"] += 1
    _raises(
        REPORT.ContractError,
        REPORT._validate_resplat_artifact_lineage,
        audit,
        tampered_manifest,
        artifact_paths,
    )


def test_full_v3_cpu_preflight_without_output_mutation() -> None:
    existed = RUNNER.OUTPUT_ROOT.exists()
    before = (
        tuple(
            (path.relative_to(RUNNER.OUTPUT_ROOT).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(RUNNER.OUTPUT_ROOT.rglob("*"))
            if path.is_file()
        )
        if existed
        else ()
    )
    record = RUNNER.preflight(check_output_available=not existed)
    assert record["scope"]["unsafe_not_deployable"] is True
    assert record["scientific_contract"]["v3_reruns_official_resplat"] is True
    assert (
        record["scientific_contract"]["v2_resplat_runtime_artifacts_reused"]
        is False
    )
    assert record["execution"][
        "terminal_validation_publication_after_internal_total_timer"
    ] is True
    assert RUNNER.OUTPUT_ROOT.exists() is existed
    after = (
        tuple(
            (path.relative_to(RUNNER.OUTPUT_ROOT).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(RUNNER.OUTPUT_ROOT.rglob("*"))
            if path.is_file()
        )
        if existed
        else ()
    )
    assert after == before


def main() -> None:
    test_v3_config_is_count_only_and_all_numeric_gates_equal_v2()
    test_gate_or_merge_numeric_drift_is_rejected()
    test_diagnostic_flags_are_all_or_nothing_and_v2_still_validates()
    test_canonical_event_hash_fails_closed_on_mutation()
    test_mapper_source220_and_final100_hook_order_is_ast_guarded()
    test_force_status_is_nested_only_under_explicit_postgate_rejection()
    test_terminal_and_report_never_claim_individual_survival()
    test_report_premerge_and_postmerge_tampering_fails_closed()
    test_frozen_v2_baseline_and_rollback_tree_bindings()
    test_real_world_and_snapshot_three_way_lineage_tampering_fails()
    test_full_v3_cpu_preflight_without_output_mutation()
    print("forced-commit v3 CPU contracts: PASS")


if __name__ == "__main__":
    main()
