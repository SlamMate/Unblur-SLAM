#!/usr/bin/env python3
"""CPU-only contracts for the unsafe v5 visibility-cache diagnostic."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types


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
    ROOT / "scripts/run_fr2_fixed_kf_resplat3_visibility_cache_refresh_forced_commit_221.py",
)
REPORT = _load(
    "test_forced_commit_report",
    ROOT / "scripts/report_fr2_fixed_kf_resplat3_visibility_cache_refresh_forced_commit_221.py",
)


def _raises(error_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_v5_directly_inherits_v4_and_changes_only_cache_contract() -> None:
    v5, v4, v2 = RUNNER._load_configs()
    fusion, differences = RUNNER._validate_v5_config(v5, v4, v2)
    assert set(differences) == RUNNER.ALLOWED_V5_V4_CONFIG_DIFFERENCES
    assert fusion.expected_forced_commit_gaussian_count == 0
    assert fusion.posthoc_after_v2_rejection is True
    assert fusion.posthoc_after_v3_count_mismatch is True
    assert fusion.count_agnostic_forced_commit is True
    assert fusion.v3_count_mismatch_audit_sha256 == RUNNER.V3_FAILED_AUDIT_SHA256
    assert fusion.posthoc_after_v4_visibility_cache_mismatch is True
    assert fusion.refresh_occ_aware_visibility_after_forced_commit is True
    assert (
        fusion.v4_visibility_cache_failure_audit_sha256
        == RUNNER.V4_FAILED_AUDIT_SHA256
    )
    assert fusion.unsafe_not_deployable is True
    assert fusion.gate_thresholds_unchanged is True
    assert fusion.force_commit_after_postmerge_rejection is True
    baseline = v2["mapping"]["official_resplat_active_fusion"]
    previous = v4["mapping"]["official_resplat_active_fusion"]
    current = v5["mapping"]["official_resplat_active_fusion"]
    for field in (
        "geometry_gate",
        "sidecar_quality_gate",
        "merge",
        "postmerge_quality_gate",
    ):
        assert baseline[field] == previous[field] == current[field]
    disclosure = v5["fixed_kf_resplat3_active_fusion_221_v5"]
    assert disclosure["cross_run_exact_candidate_count_required"] is False
    assert disclosure["fresh_accepted_count_minimum"] == 1024
    assert disclosure["fresh_accepted_count_maximum"] == 20000
    common_path = (
        ROOT
        / "configs/local/fr2_xyz_fixed_kf_resplat3_fusion_221_v5_visibility_cache_refresh_forced_commit/common.yaml"
    )
    common = common_path.read_text(encoding="utf-8")
    assert (
        "inherit_from: "
        "../fr2_xyz_fixed_kf_resplat3_fusion_221_v4_count_agnostic_forced_commit/common.yaml"
    ) in common
    assert disclosure["fresh_after_minus_before_must_equal_accepted_count"] is True
    assert disclosure["padding_or_truncation_forbidden"] is True
    assert current["expected_forced_commit_gaussian_count"] == 0
    assert current["merge"]["min_new_gaussians"] == 1024
    assert current["merge"]["max_new_gaussians"] == 20000
    assert "fresh_append_required_to_reproduce_v2_accepted_count" not in json.dumps(v5)


def test_gate_or_merge_numeric_drift_is_rejected() -> None:
    v5, v4, v2 = RUNNER._load_configs()
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
        mutated = copy.deepcopy(v5)
        mutated["mapping"]["official_resplat_active_fusion"][section][field] = value
        _raises(ValueError, RUNNER._validate_v5_config, mutated, v4, v2)


def test_v5_flags_are_all_or_nothing_and_v2_still_validates() -> None:
    v5, v4, v2 = RUNNER._load_configs()
    for field in (
        "posthoc_after_v2_rejection",
        "unsafe_not_deployable",
        "gate_thresholds_unchanged",
        "force_commit_after_postmerge_rejection",
        "posthoc_after_v3_count_mismatch",
        "count_agnostic_forced_commit",
        "posthoc_after_v4_visibility_cache_mismatch",
        "refresh_occ_aware_visibility_after_forced_commit",
    ):
        mutated = copy.deepcopy(v5)
        mutated["mapping"]["official_resplat_active_fusion"][field] = False
        _raises(ValueError, RUNNER._validate_v5_config, mutated, v4, v2)
    mutated = copy.deepcopy(v5)
    mutated["mapping"]["official_resplat_active_fusion"][
        "v3_count_mismatch_audit_sha256"
    ] = "0" * 64
    _raises(ValueError, RUNNER._validate_v5_config, mutated, v4, v2)
    mutated = copy.deepcopy(v5)
    mutated["mapping"]["official_resplat_active_fusion"][
        "expected_forced_commit_gaussian_count"
    ] = 5554
    _raises(ValueError, RUNNER._validate_v5_config, mutated, v4, v2)
    mutated = copy.deepcopy(v5)
    mutated["mapping"]["official_resplat_active_fusion"][
        "v4_visibility_cache_failure_audit_sha256"
    ] = "0" * 64
    _raises(ValueError, RUNNER._validate_v5_config, mutated, v4, v2)
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


def test_report_v5_diagnostic_protocol_tampering_fails_closed() -> None:
    protocol = {
        "schema": (
            "unblur_slam.posthoc_visibility_cache_refresh_"
            "forced_commit_diagnostic.v1"
        ),
        "posthoc_after_v2_rejection": True,
        "posthoc_after_v3_count_mismatch": True,
        "posthoc_after_v4_visibility_cache_mismatch": True,
        "refresh_occ_aware_visibility_after_forced_commit": True,
        "unsafe_not_deployable": True,
        "gate_thresholds_unchanged": True,
        "force_only_after_postmerge_gate_rejected": True,
    }
    REPORT._validate_v5_diagnostic_protocol(protocol)
    for field, bad_value in (
        ("schema", "unblur_slam.posthoc_count_agnostic_forced_commit_diagnostic.v1"),
        ("posthoc_after_v4_visibility_cache_mismatch", False),
        ("refresh_occ_aware_visibility_after_forced_commit", False),
    ):
        tampered = dict(protocol)
        tampered[field] = bad_value
        _raises(
            REPORT.ContractError,
            REPORT._validate_v5_diagnostic_protocol,
            tampered,
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
        ROOT / "scripts/report_fr2_fixed_kf_resplat3_visibility_cache_refresh_forced_commit_221.py"
    ).read_text(encoding="utf-8")
    assert '"world_bridge"' in mapper
    assert '"individual_imported_gaussian_survival_in_final_model_not_claimed": True' in mapper
    assert "individual_imported_gaussian_survival_not_tracked_or_claimed" in report
    assert "fresh_candidate_identity_equal_to_v2_or_v3_not_claimed" in report
    assert "fair paired estimate" in report


def test_v5_count_contract_is_bounded_and_internal_not_cross_run_exact() -> None:
    v5, _, _ = RUNNER._load_configs()
    from src.refinement.official_resplat_active_fusion import (
        ActiveFusionConfig,
        forced_commit_count_contract,
    )

    config = ActiveFusionConfig.from_dict(
        v5["mapping"]["official_resplat_active_fusion"],
        default_output_root=Path("/tmp/v5-count-contract"),
    )
    for accepted in (1024, 5450, 5554, 5716, 7777, 20000):
        record = forced_commit_count_contract(
            accepted_count=accepted,
            before_count=70000,
            after_count=70000 + accepted,
            config=config,
        )
        assert record["accepted"] is True
        assert record["cross_run_exact_count_required"] is False
        assert record["exact_cross_run_count_requirement_satisfied"] is None
    for accepted in (1023, 20001):
        record = forced_commit_count_contract(
            accepted_count=accepted,
            before_count=70000,
            after_count=70000 + accepted,
            config=config,
        )
        assert record["accepted"] is False
        assert "accepted_count_outside_preregistered_merge_bounds" in record["reasons"]
    mismatched = forced_commit_count_contract(
        accepted_count=5716,
        before_count=70000,
        after_count=75715,
        config=config,
    )
    assert mismatched["accepted"] is False
    assert "after_minus_before_does_not_equal_accepted_count" in mismatched["reasons"]
    for accepted in (1024, 5450, 5554, 5716, 7777, 20000):
        assert REPORT._validate_v5_fresh_count_algebra(
            {
                "accepted_count": accepted,
                "before_count": 70000,
                "after_count": 70000 + accepted,
            },
            {"gaussian_count": 70000},
            {"gaussian_count": 70000 + accepted},
        ) == accepted
    for accepted in (1023, 20001):
        _raises(
            REPORT.ContractError,
            REPORT._validate_v5_fresh_count_algebra,
            {
                "accepted_count": accepted,
                "before_count": 70000,
                "after_count": 70000 + accepted,
            },
            {"gaussian_count": 70000},
            {"gaussian_count": 70000 + accepted},
        )
    _raises(
        REPORT.ContractError,
        REPORT._validate_v5_fresh_count_algebra,
        {"accepted_count": 5716, "before_count": 70000, "after_count": 75715},
        {"gaussian_count": 70000},
        {"gaussian_count": 75715},
    )
    mapper = (ROOT / "src/mapper.py").read_text(encoding="utf-8")
    assert "forced_commit_count_contract(" in mapper
    assert '"prior_v3_observed_action": (' in mapper
    assert '"rollback_after_exact_count_mismatch"' in mapper


def test_report_premerge_and_postmerge_tampering_fails_closed() -> None:
    v5, _, _ = RUNNER._load_configs()
    fusion_raw = v5["mapping"]["official_resplat_active_fusion"]
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
            audit, v5, manifest_path
        )
        assert valid["manifest_path"] == manifest_path
        for gate in audit["premerge_gates"]:
            mutated = copy.deepcopy(audit)
            mutated["premerge_gates"][gate]["accepted"] = False
            _raises(
                REPORT.ContractError,
                REPORT._validate_forced_premerge_contract,
                mutated,
                v5,
                manifest_path,
            )
        mutated = copy.deepcopy(audit)
        mutated["published_result"]["manifest_sha256"] = "0" * 64
        _raises(
            REPORT.ContractError,
            REPORT._validate_forced_premerge_contract,
            mutated,
            v5,
            manifest_path,
        )
        mutated = copy.deepcopy(audit)
        mutated["sh_import"]["active_harmonic_dimension"] = 16
        _raises(
            REPORT.ContractError,
            REPORT._validate_forced_premerge_contract,
            mutated,
            v5,
            manifest_path,
        )
        mutated = copy.deepcopy(audit)
        mutated["merge"]["config"]["min_opacity"] = 0.49
        _raises(
            REPORT.ContractError,
            REPORT._validate_forced_premerge_contract,
            mutated,
            v5,
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
    v5, _, _ = RUNNER._load_configs()
    record = RUNNER._validate_frozen_v2_references(v5)
    assert record["baseline"]["tree_sha256"] == RUNNER.V2_BASELINE_TREE_SHA256
    assert (
        record["rollback_arm"]["tree_sha256"]
        == RUNNER.V2_ROLLBACK_TREE_SHA256
    )
    assert record["v2_rollback_fusion_audit"]["accepted_count"] == 5554
    assert record["v2_rollback_fusion_audit"]["postmerge_gate_accepted"] is False
    assert record["reference_semantics"]["fresh_pair_rerun"] is False
    assert record["reference_semantics"]["consumed_by_v5_slam_or_resplat_runtime"] is False


def test_frozen_v3_failed_attempt_tree_launcher_and_rollback_binding() -> None:
    record = RUNNER._v3_failure_binding()
    assert record["frozen_tree_sha256"] == RUNNER.V3_FAILED_TREE_SHA256
    assert record["launcher_runtime"]["sha256"] == RUNNER.V3_FAILED_LAUNCHER_SHA256
    assert record["launcher_runtime"]["exit_code"] == 1
    assert record["fusion_audit_sha256"] == RUNNER.V3_FAILED_AUDIT_SHA256
    assert record["status"] == "forced_commit_candidate_count_mismatch_rolled_back"
    assert record["accepted_count"] == 5716
    assert record["before_count"] == 72645
    assert record["trial_count"] == 78361
    assert record["rollback_byte_identical_to_before"] is True
    assert record["consumed_by_v5_runtime"] is False

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "broken-link").symlink_to(root / "missing-target")
        _raises(ValueError, RUNNER._tree_binding, root, "0" * 64)


def test_frozen_v4_failure_tree_audit_launcher_and_commit_binding() -> None:
    before = RUNNER._tree_binding(
        RUNNER.V4_FAILED_ROOT, RUNNER.V4_FAILED_TREE_SHA256
    )
    record = RUNNER._v4_failure_binding()
    after = RUNNER._tree_binding(
        RUNNER.V4_FAILED_ROOT, RUNNER.V4_FAILED_TREE_SHA256
    )
    assert before == after
    assert record["frozen_tree_sha256"] == RUNNER.V4_FAILED_TREE_SHA256
    assert record["fusion_audit"]["sha256"] == RUNNER.V4_FAILED_AUDIT_SHA256
    assert record["launcher_runtime"]["sha256"] == RUNNER.V4_FAILED_LAUNCHER_SHA256
    assert record["launcher_runtime"]["exit_code"] == 1
    assert record["launch_log"]["sha256"] == RUNNER.V4_FAILED_LAUNCH_LOG_SHA256
    assert (
        record["commit_event"]["file_sha256"]
        == RUNNER.V4_FAILED_COMMIT_EVENT_FILE_SHA256
    )
    assert record["fusion_audit"]["accepted_count"] == 5450
    assert record["fusion_audit"]["before_count"] == 72678
    assert record["fusion_audit"]["committed_count"] == 78128
    assert record["source220_complete_event_absent"] is True
    assert record["final100_and_terminal_events_absent"] is True
    assert record["read_only_lineage_only"] is True
    assert record["consumed_by_v5_runtime"] is False


def test_visibility_cache_refresh_uses_fresh_render_and_is_atomic() -> None:
    import torch
    import src.mapper as mapper_module
    from src.mapper import Mapper

    before_count = 3
    after_count = 5
    state_sha = "a" * 64
    regular_camera = types.SimpleNamespace(
        uid=11,
        timestamp=166,
        deblur_fail=False,
    )
    fallback_camera = types.SimpleNamespace(
        uid=12,
        timestamp=125,
        deblur_fail=True,
        n_virtual_cams=3,
        get_virtual_extrinsics=lambda: (
            [torch.tensor(0), torch.tensor(1), torch.tensor(2)],
            [torch.tensor(0), torch.tensor(1), torch.tensor(2)],
            [torch.tensor(0), torch.tensor(1), torch.tensor(2)],
            [torch.tensor(0), torch.tensor(1), torch.tensor(2)],
        ),
    )
    subject = object.__new__(Mapper)
    subject.gaussians = types.SimpleNamespace(
        get_xyz=torch.zeros((after_count, 3), dtype=torch.float32)
    )
    subject.current_window = [11, 12]
    subject.cameras = {11: regular_camera, 12: fallback_camera}
    subject.pipeline_params = object()
    subject.background = torch.zeros(3)
    subject.occ_aware_visibility = {
        11: torch.tensor([1, 0, 1], dtype=torch.long),
        12: torch.tensor([0, 1, 0], dtype=torch.long),
    }
    subject._capture_active_gaussian_transaction_state = types.MethodType(
        lambda _self: {"before_count": after_count, "sha256": state_sha},
        subject,
    )
    old_cache = subject.occ_aware_visibility
    old_ids = {key: id(value) for key, value in old_cache.items()}
    old_bytes = {key: value.clone() for key, value in old_cache.items()}
    render_calls = []
    virtual_calls = []
    virtual_outputs = iter(
        (
            torch.tensor([0, 0, 2, 0, 0]),
            torch.tensor([3, 0, 0, 0, 0]),
            torch.tensor([0, 0, 0, 4, 0]),
        )
    )

    def fake_render(viewpoint, *_args, **_kwargs):
        render_calls.append(viewpoint.uid)
        return {"n_touched": torch.tensor([0, 1, 0, 2, 0])}

    def fake_render_virtual(viewpoint, *_args, **_kwargs):
        virtual_calls.append(viewpoint.uid)
        return {"n_touched": next(virtual_outputs)}

    original_render = mapper_module.render
    original_render_virtual = mapper_module.render_virtual
    mapper_module.render = fake_render
    mapper_module.render_virtual = fake_render_virtual
    try:
        refreshed, report = subject._prepare_active_fusion_visibility_cache_refresh(
            before_count=before_count,
            after_count=after_count,
            expected_active_state_sha256=state_sha,
        )
        assert render_calls == [11]
        assert virtual_calls == [12, 12, 12]
        assert torch.equal(refreshed[11], torch.tensor([0, 1, 0, 1, 0]))
        assert torch.equal(refreshed[12], torch.tensor([1, 0, 1, 1, 0]))
        assert subject.occ_aware_visibility is old_cache
        assert tuple(subject.occ_aware_visibility) == (11, 12)
        for key in old_cache:
            assert id(subject.occ_aware_visibility[key]) == old_ids[key]
            assert torch.equal(subject.occ_aware_visibility[key], old_bytes[key])
        assert report["status"] == "validated_for_atomic_commit"
        assert report["atomic_commit_assignment_performed"] is False
        assert report["keys_before"] == report["keys_after"] == [11, 12]
        assert report["zero_padding_used"] is False
        assert report["padding_or_truncation_used"] is False
        assert report["uses_ground_truth"] is False
        assert report["elapsed_seconds"] >= 0.0

        subject.occ_aware_visibility[11] = torch.ones(
            before_count - 1, dtype=torch.long
        )
        calls_before_rejection = (len(render_calls), len(virtual_calls))
        _raises(
            RuntimeError,
            subject._prepare_active_fusion_visibility_cache_refresh,
            before_count=before_count,
            after_count=after_count,
            expected_active_state_sha256=state_sha,
        )
        assert (len(render_calls), len(virtual_calls)) == calls_before_rejection

        subject.occ_aware_visibility = old_cache
        subject.occ_aware_visibility[11] = old_bytes[11]
        mapper_module.render = lambda *_args, **_kwargs: {
            "n_touched": torch.ones(after_count - 1)
        }
        snapshot = {
            key: value.clone() for key, value in subject.occ_aware_visibility.items()
        }
        _raises(
            RuntimeError,
            subject._prepare_active_fusion_visibility_cache_refresh,
            before_count=before_count,
            after_count=after_count,
            expected_active_state_sha256=state_sha,
        )
        assert subject.occ_aware_visibility is old_cache
        for key, value in snapshot.items():
            assert torch.equal(subject.occ_aware_visibility[key], value)
    finally:
        mapper_module.render = original_render
        mapper_module.render_virtual = original_render_virtual


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


def test_full_v5_cpu_preflight_without_output_or_v4_mutation() -> None:
    assert not RUNNER.OUTPUT_ROOT.exists()
    v4_before = RUNNER._tree_binding(
        RUNNER.V4_FAILED_ROOT, RUNNER.V4_FAILED_TREE_SHA256
    )
    record = RUNNER.preflight(check_output_available=True)
    assert record["scope"]["unsafe_not_deployable"] is True
    assert record["scientific_contract"]["v5_reruns_official_resplat"] is True
    assert record["scope"]["posthoc_after_v3_count_mismatch"] is True
    assert record["scope"]["posthoc_after_v4_visibility_cache_mismatch"] is True
    assert record["scope"]["refresh_occ_aware_visibility_after_forced_commit"] is True
    assert record["scientific_contract"]["cross_run_exact_candidate_count_required"] is False
    assert record["scientific_contract"]["fresh_accepted_count_minimum"] == 1024
    assert record["scientific_contract"]["fresh_accepted_count_maximum"] == 20000
    assert (
        record["scientific_contract"]["v2_resplat_runtime_artifacts_reused"]
        is False
    )
    assert record["execution"][
        "terminal_validation_publication_after_internal_total_timer"
    ] is True
    assert record["execution"]["online_timer_includes_visibility_cache_refresh"] is True
    assert (
        record["frozen_v4_visibility_cache_failure_lineage"][
            "frozen_tree_sha256"
        ]
        == RUNNER.V4_FAILED_TREE_SHA256
    )
    v4_after = RUNNER._tree_binding(
        RUNNER.V4_FAILED_ROOT, RUNNER.V4_FAILED_TREE_SHA256
    )
    assert v4_before == v4_after
    assert not RUNNER.OUTPUT_ROOT.exists()


def main() -> None:
    test_v5_directly_inherits_v4_and_changes_only_cache_contract()
    test_gate_or_merge_numeric_drift_is_rejected()
    test_v5_flags_are_all_or_nothing_and_v2_still_validates()
    test_canonical_event_hash_fails_closed_on_mutation()
    test_report_v5_diagnostic_protocol_tampering_fails_closed()
    test_mapper_source220_and_final100_hook_order_is_ast_guarded()
    test_force_status_is_nested_only_under_explicit_postgate_rejection()
    test_terminal_and_report_never_claim_individual_survival()
    test_v5_count_contract_is_bounded_and_internal_not_cross_run_exact()
    test_report_premerge_and_postmerge_tampering_fails_closed()
    test_frozen_v2_baseline_and_rollback_tree_bindings()
    test_frozen_v3_failed_attempt_tree_launcher_and_rollback_binding()
    test_frozen_v4_failure_tree_audit_launcher_and_commit_binding()
    test_visibility_cache_refresh_uses_fresh_render_and_is_atomic()
    test_real_world_and_snapshot_three_way_lineage_tampering_fails()
    test_full_v5_cpu_preflight_without_output_or_v4_mutation()
    print("visibility-cache-refresh forced-commit v5 CPU contracts: PASS")


if __name__ == "__main__":
    main()
