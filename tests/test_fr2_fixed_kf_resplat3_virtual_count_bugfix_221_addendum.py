#!/usr/bin/env python3
"""CPU-only contracts for the immutable v6 post-run audit addendum."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
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


ADDENDUM = _load(
    "test_virtual_count_bugfix_v6_addendum",
    ROOT / "scripts/report_fr2_fixed_kf_resplat3_virtual_count_bugfix_221_addendum.py",
)
_CACHED_RECORD = None


def _record():
    global _CACHED_RECORD
    if _CACHED_RECORD is None:
        _CACHED_RECORD = ADDENDUM.build_addendum()
    return deepcopy(_CACHED_RECORD)


def _raises(error_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_build_and_validate_exact_postrun_addendum() -> None:
    primary_before = ADDENDUM._file_binding(
        ADDENDUM.DEFAULT_ROOT / "_report" / ADDENDUM.PRIMARY_REPORT_NAME
    )
    arm_before = ADDENDUM._tree_binding(ADDENDUM.RUNNER.OUTPUT)
    record = _record()
    ADDENDUM.validate_addendum(record)
    assert record["primary_report"]["sha256"] == ADDENDUM.PRIMARY_REPORT_SHA256
    assert record["v6_arm_frozen_tree"]["tree_sha256"] == ADDENDUM.V6_ARM_TREE_SHA256
    assert record["v6_arm_frozen_tree"]["file_count"] == 104
    assert record["v6_arm_frozen_tree"]["total_bytes"] == 95_380_540
    extension = record["post_report_audit_code_extension"]
    assert extension["persisted_preflight_git_dirty_entry_count"] == 198
    assert extension["current_git_dirty_entry_count"] == 200
    assert extension["exact_dirty_entry_count_delta"] == 2
    assert extension["exact_new_untracked_paths"] == list(
        ADDENDUM.ADDENDUM_AUDIT_CODE_PATHS
    )
    assert ADDENDUM._file_binding(
        ADDENDUM.DEFAULT_ROOT / "_report" / ADDENDUM.PRIMARY_REPORT_NAME
    ) == primary_before
    assert ADDENDUM._tree_binding(ADDENDUM.RUNNER.OUTPUT) == arm_before


def test_complete_v2_references_and_lineage_are_persisted() -> None:
    record = _record()
    v2 = record["reference_integrity"]["v2_dual_reference"]
    assert v2["baseline"]["tree_sha256"] == "54fa5800205dc980ce0cb8fc3f23798cdd07fe2b532fad1f6a3c723d9ee04ecc"
    assert v2["rollback_arm"]["tree_sha256"] == "5256a557e26f09a02438f241cae56ebad1bc654455d50d2a83a7951b62246eca"
    assert v2["v2_experiment_audit"]["sha256"] == "f95b9fdf74e0458e7e7103640c764f27c2b126e368db0b3508fa354deae53bc3"
    assert v2["v2_rollback_fusion_audit"]["sha256"] == "a8e65b6851c998c0038bf228a337de18af9cc182cfa4d711e5a92d9543053c3d"
    lineage = record["selection_and_gt_lineage"]
    assert lineage["selection_membership_clear_gt_conditioned"] is True
    assert lineage["fusion_consumes_ground_truth_pose_or_depth"] is False
    assert lineage["fusion_consumes_independent_clear_pixels"] is False
    assert lineage["fusion_consumes_clear_gt_metrics"] is False
    assert lineage["clear_gt_metrics_bound_posthoc_for_evaluation"] is True
    assert lineage["clear_gt_values_used_for_commit_or_checkpoint_selection"] is False


def test_speed_and_numeric_semantics() -> None:
    record = _record()
    speed = record["speed_semantics"]
    assert speed["derived_prefix_fps_formula"] == "221 / official_timer_online_seconds"
    assert speed["paper_table_6_fps_comparable"] is False
    assert speed["fresh_paired_comparison"] is False
    for arm in speed["arms"].values():
        assert math.isclose(
            arm["derived_prefix_online_fps"],
            221.0 / arm["official_timer_online_seconds"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    comparisons = record["numeric_results"][
        "diagnostic_comparisons_not_fair_paired_estimates"
    ]
    assert comparisons["v6_minus_v2_baseline_reference"]["psnr_db"] > 0.0
    assert comparisons["v6_minus_v2_baseline_reference"]["online_speed_ratio"] < 1.0
    assert comparisons["v6_minus_v2_safe_rollback_reference"]["psnr_db"] > 0.0


def test_chain_artifact_and_survival_bindings() -> None:
    record = _record()
    integrity = record["postrun_integrity"]
    assert integrity["fusion_status"] == "postmerge_gate_rejected_forced_commit_unsafe"
    assert integrity["postmerge_gate_accepted"] is False
    assert integrity["unsafe_forced_commit"] is True
    assert integrity["ordinary_policy_would_rollback"] is True
    assert integrity["source220_and_final100_participation_proven"] is True
    assert integrity["individual_imported_gaussian_survival_not_tracked_or_claimed"] is True
    chain = integrity["chain"]
    assert chain["commit"]["canonical_event_sha256"] == "d0e918ad43825dd88c109e38602a3d8b306afec0789fbb7bd4db52279a8c9df8"
    assert chain["source220"]["canonical_event_sha256"] == "a4ea331f99518929499a67c20293bf6bdc1cde173c22bb6288d5864f2f16f5c2"
    assert chain["final100"]["canonical_event_sha256"] == "20b7ca5cae913b5208c96364194283c139e0b2cdc2e4e135fa92431f7b746dab"
    assert chain["terminal"]["canonical_terminal_sha256"] == "17517a30d6e9906d4a328b44f6d54dcb8db8c5a1a2d5b6e74b5837b15e467e95"
    assert len(integrity["terminal_artifact_bindings"]) == 10
    assert len(integrity["terminal_code_bindings"]) == 8


def test_tampering_and_overwrite_fail_closed() -> None:
    record = _record()
    mutations = []
    tampered = deepcopy(record)
    tampered["primary_report"]["sha256"] = "0" * 64
    mutations.append(tampered)
    tampered = deepcopy(record)
    tampered["reference_integrity"]["v2_dual_reference"]["baseline"]["tree_sha256"] = "0" * 64
    mutations.append(tampered)
    tampered = deepcopy(record)
    tampered["selection_and_gt_lineage"]["selection_membership_clear_gt_conditioned"] = False
    mutations.append(tampered)
    tampered = deepcopy(record)
    tampered["postrun_integrity"]["chain"]["source220"]["canonical_event_sha256"] = "0" * 64
    mutations.append(tampered)
    tampered = deepcopy(record)
    tampered["speed_semantics"]["paper_table_6_fps_comparable"] = True
    mutations.append(tampered)
    expected_material = deepcopy(record)
    expected_material.pop("generated_at_utc")
    original_material = ADDENDUM._material
    ADDENDUM._material = lambda root: deepcopy(expected_material)
    try:
        for value in mutations:
            _raises(ADDENDUM.AddendumError, ADDENDUM.validate_addendum, value)
    finally:
        ADDENDUM._material = original_material
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        ADDENDUM.write_addendum(record, output)
        _raises(FileExistsError, ADDENDUM.write_addendum, record, output)


def main() -> None:
    test_build_and_validate_exact_postrun_addendum()
    test_complete_v2_references_and_lineage_are_persisted()
    test_speed_and_numeric_semantics()
    test_chain_artifact_and_survival_bindings()
    test_tampering_and_overwrite_fail_closed()
    print("virtual-count bugfix v6 post-run addendum contracts: PASS")


if __name__ == "__main__":
    main()
