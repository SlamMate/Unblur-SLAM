#!/usr/bin/env python3
"""Audit the preregistered three-seed TURTLE temporal-order v2 smoke.

The report deliberately opens only the Room1 temporal-validation manifest,
one pinned base evaluation, and the configured fine-tuned evaluations.  A
Room2 path is neither accepted by the CLI nor opened by this module.  Metric
gate failures are experimental results and produce ``eligible=false``;
provenance/contract failures raise before an output is written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

from scripts import report_turtle_history_smoke as v1  # noqa: E402


CONTRACT_SCHEMA = "unblur_slam.turtle_replica_temporal_order_multiseed.v2"
REPORT_SCHEMA = "unblur_slam.turtle_replica_temporal_order_multiseed_report.v2"
ADDENDUM_SCHEMA = "unblur_slam.turtle_temporal_v2_motion_only_addendum.v1"
REPORT_FIX_SCHEMA = "unblur_slam.turtle_temporal_v2_reporting_fix.v1"
EXPECTED_IMPLEMENTATION_PINS = {
    "train_script_sha256": "scripts/train_turtle_streaming.py",
    "evaluate_script_sha256": "scripts/evaluate_turtle_streaming.py",
    "report_script_sha256": "scripts/report_turtle_temporal_v2.py",
    "test_sha256": "tests/test_turtle_temporal_v2.py",
    "report_test_sha256": "tests/test_report_turtle_temporal_v2.py",
    "backend_sha256": "src/turtle_backend.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise v1.HistorySmokeContractError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _load_contract(path: Path | str) -> tuple[dict[str, Any], Path, str]:
    contract, source, digest = v1._load_json(path, "v2 preregistered contract")
    _require(contract.get("schema") == CONTRACT_SCHEMA, "v2 contract schema mismatch")
    _require(
        contract.get("status") == "preregistered_before_gpu_training",
        "v2 contract was not preregistered before GPU training",
    )
    protocol = _mapping(contract.get("protocol"), "contract.protocol")
    _require(protocol.get("future_frames_used") is False, "future frames must be disabled")
    _require(
        protocol.get("room2_frame_pixels_read_before_validation_pass") is False,
        "Room2 frame-pixel policy changed",
    )
    _require(
        protocol.get("room2_metrics_evaluated_before_validation_pass") is False,
        "Room2 metric policy changed",
    )
    _require(
        protocol.get("room2_manifest_bytes_already_read") is True,
        "known Room2 manifest-read disclosure was removed",
    )
    return contract, source, digest


def _load_addendum(
    path: Path | str, *, contract_source: Path, contract_sha: str
) -> tuple[dict[str, Any], Path, str]:
    addendum, source, digest = v1._load_json(path, "motion-only protocol addendum")
    _require(addendum.get("schema") == ADDENDUM_SCHEMA, "addendum schema mismatch")
    _require(
        addendum.get("status") == "frozen_before_any_v2_validation_evaluation",
        "addendum was not frozen before v2 validation",
    )
    binding = _mapping(addendum.get("binds"), "addendum.binds")
    _require(
        Path(str(binding.get("contract_path"))).expanduser().resolve()
        == contract_source,
        "addendum contract path mismatch",
    )
    _require(binding.get("contract_sha256") == contract_sha, "addendum contract SHA mismatch")
    policy = _mapping(addendum.get("selection_policy"), "addendum.selection_policy")
    required = {
        "tum_keyframe_selection": "motion_only_selection_independent",
        "clear_gt_membership_used": False,
        "gt_pose_or_depth_used_during_selection": False,
        "legacy_clear_conditioned_smoke_permitted": False,
        "open_gt_metrics_only_after_outputs_frozen": True,
    }
    for key, expected in required.items():
        _require(policy.get(key) == expected, f"addendum selection policy changed: {key}")
    return addendum, source, digest


def _load_reporting_fix(
    path: Path | str,
    *,
    contract_sha: str,
    motion_addendum_sha: str,
) -> tuple[dict[str, Any], Path, str]:
    fix, source, digest = v1._load_json(path, "post-validation reporting fix")
    _require(fix.get("schema") == REPORT_FIX_SCHEMA, "reporting-fix schema mismatch")
    _require(
        fix.get("status") == "frozen_after_seed17_validation_before_multiseed_report",
        "reporting-fix timing disclosure changed",
    )
    binding = _mapping(fix.get("binds"), "reporting_fix.binds")
    _require(binding.get("contract_sha256") == contract_sha, "reporting-fix contract SHA mismatch")
    _require(
        binding.get("motion_addendum_sha256") == motion_addendum_sha,
        "reporting-fix motion-addendum SHA mismatch",
    )
    _require(
        fix.get("metric_values_or_gates_changed") is False,
        "reporting fix may not change metric values or gates",
    )
    return fix, source, digest


def _verify_implementation_pins(
    contract: Mapping[str, Any], reporting_fix: Mapping[str, Any]
) -> dict[str, Any]:
    configured = _mapping(contract.get("implementation_pins"), "implementation_pins")
    _require(
        set(configured) == set(EXPECTED_IMPLEMENTATION_PINS),
        "v2 implementation pin set changed",
    )
    overrides = _mapping(
        reporting_fix.get("implementation_pin_overrides"),
        "reporting_fix.implementation_pin_overrides",
    )
    _require(
        set(overrides) == {"report_script_sha256", "report_test_sha256"},
        "addendum may override only the v2 reporter and its test",
    )
    expected_pins = dict(configured)
    expected_pins.update(overrides)
    result: dict[str, Any] = {}
    for name, relative in EXPECTED_IMPLEMENTATION_PINS.items():
        expected = str(expected_pins[name]).lower()
        actual = v1.sha256_file(ROOT / relative)
        _require(actual == expected, f"implementation pin mismatch for {relative}")
        result[name] = {"path": str((ROOT / relative).resolve()), "sha256": actual}
    return result


def _v1_compatible_contract(contract: Mapping[str, Any], seed: int) -> dict[str, Any]:
    training = dict(_mapping(contract.get("training"), "contract.training"))
    training["seed"] = int(seed)
    training.pop("seeds", None)
    runs = _mapping(contract.get("runs"), "contract.runs")
    run = _mapping(runs.get(str(seed)), f"contract.runs.{seed}")
    checkpoint_parent = Path(str(run.get("checkpoint"))).expanduser().resolve().parent
    return {
        "model": dict(_mapping(contract.get("model"), "contract.model")),
        "data": dict(_mapping(contract.get("data"), "contract.data")),
        "training": training,
        "evaluation": dict(_mapping(contract.get("evaluation"), "contract.evaluation")),
        "output_root": str(checkpoint_parent),
    }


def _verify_v2_loss_metadata(
    fine: Mapping[str, Any], contract: Mapping[str, Any], seed: int
) -> None:
    metadata = _mapping(
        _mapping(fine.get("checkpoint"), f"seed {seed} checkpoint").get("metadata"),
        f"seed {seed} checkpoint metadata",
    )
    actual = _mapping(
        _mapping(metadata.get("loss"), f"seed {seed} loss").get("temporal_order_v2"),
        f"seed {seed} temporal_order_v2",
    )
    objective = _mapping(
        _mapping(contract.get("training"), "contract.training").get("temporal_order_objective"),
        "contract.training.temporal_order_objective",
    )
    expected = {
        "temporal_delta_l1_weight": objective["temporal_delta_l1_weight"],
        "ordered_vs_shuffled_rank_weight": objective[
            "ordered_vs_shuffled_rank_weight"
        ],
        "ordered_vs_shuffled_l1_margin": objective[
            "ordered_vs_shuffled_l1_margin"
        ],
        "anchor_policy": "last_supervised_frame",
        "shuffle_policy": "cyclic_left_shift_complete_past_prefix",
        "counterfactual_past_multiset_preserved": True,
        "future_frames_used": False,
    }
    _require(dict(actual) == expected, f"seed {seed} temporal-order metadata changed")


def _seed_statistics(values: Sequence[float]) -> dict[str, float]:
    collected = [float(value) for value in values]
    _require(bool(collected), "seed statistic has no values")
    return {
        "mean": statistics.fmean(collected),
        "population_std": statistics.pstdev(collected),
        "min": min(collected),
        "max": max(collected),
    }


def _parse_seed_metrics(values: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError("--fine-metrics must use SEED=/absolute/metrics.json")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate fine metrics seed: {seed}")
        result[seed] = Path(path_text).expanduser().resolve()
    return result


def build_report(
    contract_path: Path | str,
    addendum_path: Path | str,
    reporting_fix_path: Path | str,
    base_metrics_path: Path | str,
    fine_metrics_by_seed: Mapping[int, Path | str],
) -> dict[str, Any]:
    contract, contract_source, contract_sha = _load_contract(contract_path)
    addendum, addendum_source, addendum_sha = _load_addendum(
        addendum_path,
        contract_source=contract_source,
        contract_sha=contract_sha,
    )
    reporting_fix, reporting_fix_source, reporting_fix_sha = _load_reporting_fix(
        reporting_fix_path,
        contract_sha=contract_sha,
        motion_addendum_sha=addendum_sha,
    )
    implementation = _verify_implementation_pins(contract, reporting_fix)
    expected_rows, manifest_metadata = v1._load_validation_manifest(contract)
    training = _mapping(contract.get("training"), "contract.training")
    seeds = [int(seed) for seed in training.get("seeds", [])]
    _require(len(seeds) >= 3 and len(seeds) == len(set(seeds)), "at least three unique seeds are required")
    _require(set(fine_metrics_by_seed) == set(seeds), "fine metrics seed set differs from preregistration")

    base_artifact = _mapping(
        _mapping(contract.get("artifacts"), "contract.artifacts").get("base_metrics"),
        "contract.artifacts.base_metrics",
    )
    base_path = Path(str(base_metrics_path)).expanduser().resolve()
    _require(
        base_path == Path(str(base_artifact.get("path"))).expanduser().resolve(),
        "base metrics path differs from preregistration",
    )
    base_payload, base_source, base_sha = v1._load_json(base_path, "base metrics")
    _require(base_sha == str(base_artifact.get("sha256")), "base metrics SHA-256 mismatch")
    base = v1._validate_evaluation(
        base_payload,
        role="base",
        expected_rows=expected_rows,
        manifest_metadata=manifest_metadata,
        contract=_v1_compatible_contract(contract, seeds[0]),
    )

    configured_runs = _mapping(contract.get("runs"), "contract.runs")
    per_seed: dict[str, Any] = {}
    metric_series: dict[str, list[float]] = {
        "normal_minus_base_psnr_db": [],
        "normal_minus_reset_psnr_db": [],
        "normal_minus_repeat_psnr_db": [],
        "normal_minus_shuffle_psnr_db": [],
        "history_gain_interaction_psnr_db": [],
    }
    for seed in seeds:
        run = _mapping(configured_runs.get(str(seed)), f"contract.runs.{seed}")
        fine_path = Path(str(fine_metrics_by_seed[seed])).expanduser().resolve()
        _require(
            fine_path == Path(str(run.get("metrics"))).expanduser().resolve(),
            f"seed {seed} fine metrics path differs from preregistration",
        )
        payload, source, digest = v1._load_json(fine_path, f"seed {seed} fine metrics")
        expected_digest = run.get("metrics_sha256")
        if expected_digest is not None:
            _require(digest == str(expected_digest), f"seed {seed} fine metrics SHA-256 mismatch")
        seed_contract = _v1_compatible_contract(contract, seed)
        fine = v1._validate_evaluation(
            payload,
            role="finetuned",
            expected_rows=expected_rows,
            manifest_metadata=manifest_metadata,
            contract=seed_contract,
        )
        _verify_v2_loss_metadata(fine, contract, seed)
        _require(
            Path(fine["checkpoint"]["path"]) == Path(str(run.get("checkpoint"))).resolve(),
            f"seed {seed} checkpoint path differs from preregistration",
        )
        for index, (base_row, fine_row) in enumerate(zip(base["rows"], fine["rows"])):
            _require(
                base_row["raw_path"] == fine_row["raw_path"]
                and base_row["gt_path"] == fine_row["gt_path"],
                f"seed {seed} base/fine frame identity mismatch at row {index}",
            )
        comparisons = {
            region: v1._comparison(base[region], fine[region])
            for region in ("all", "steady")
        }
        per_sequence = {
            sequence: {
                region: v1._comparison(
                    base["per_sequence"][sequence][region],
                    fine["per_sequence"][sequence][region],
                )
                for region in ("all", "steady")
            }
            for sequence in sorted(base["per_sequence"])
        }
        gates = v1._build_gates(
            seed_contract,
            base,
            fine,
            comparisons["steady"],
            per_sequence,
        )
        eligible = all(bool(gate["passed"]) for gate in gates.values())
        steady = comparisons["steady"]
        fine_controls = steady["finetuned_normal_minus_control"]
        series_row = {
            "normal_minus_base_psnr_db": steady[
                "finetuned_normal_minus_base_normal"
            ]["psnr"],
            "normal_minus_reset_psnr_db": fine_controls[v1.RESET]["psnr"],
            "normal_minus_repeat_psnr_db": fine_controls[v1.REPEAT]["psnr"],
            "normal_minus_shuffle_psnr_db": fine_controls[v1.SHUFFLED]["psnr"],
            "history_gain_interaction_psnr_db": steady[
                "history_gain_interaction_vs_base_reset_psnr_db"
            ],
        }
        for name, value in series_row.items():
            metric_series[name].append(float(value))
        per_seed[str(seed)] = {
            "eligible": eligible,
            "failed_gates": [name for name, gate in gates.items() if not gate["passed"]],
            "gates": gates,
            "steady_metrics": series_row,
            "artifacts": {
                "metrics": {"path": str(source), "sha256": digest},
                "checkpoint": fine["checkpoint"],
                "device": fine["device"],
            },
        }

    eligible = all(result["eligible"] for result in per_seed.values())
    evaluation = _mapping(contract.get("evaluation"), "contract.evaluation")
    return {
        "schema": REPORT_SCHEMA,
        "status": (
            "eligible_for_room2_one_shot_and_metric_bearing_slam_smoke"
            if eligible
            else "ineligible_room2_remains_unopened"
        ),
        "eligible": eligible,
        "decision": (
            addendum["on_pass"] if eligible else evaluation["on_failure"]
        ),
        "protocol": {
            "seed_eligibility": "every preregistered seed must pass every unchanged v1 history gate",
            "future_frames_used": False,
            "room2_frame_pixels_read_or_metrics_evaluated": False,
            "room2_manifest_bytes_already_read": True,
            "room2_is_pristine_sealed_holdout": False,
            "tum_metric_smoke_if_eligible": "motion_only_selection_independent",
            "legacy_clear_conditioned_tum_smoke_permitted": False,
            "validation_scene": "ReplicaBlurry Room1 only",
            "steady_frame_index_min": manifest_metadata["steady_frame_index_min"],
        },
        "artifacts": {
            "contract": {"path": str(contract_source), "sha256": contract_sha},
            "motion_only_protocol_addendum": {
                "path": str(addendum_source),
                "sha256": addendum_sha,
            },
            "post_seed17_reporting_fix": {
                "path": str(reporting_fix_source),
                "sha256": reporting_fix_sha,
                "metric_values_or_gates_changed": False,
            },
            "implementation_pins": implementation,
            "validation_manifest": manifest_metadata,
            "base_metrics": {"path": str(base_source), "sha256": base_sha},
            "base_checkpoint": base["checkpoint"],
        },
        "aggregate_across_seeds": {
            name: _seed_statistics(values) for name, values in metric_series.items()
        },
        "per_seed": per_seed,
    }


def write_report(path: Path | str, report: Mapping[str, Any]) -> str:
    output = Path(path).expanduser().resolve()
    digest_path = output.with_name(output.name + ".sha256")
    if output.exists() or digest_path.exists():
        raise FileExistsError(f"refusing to overwrite v2 report artifacts: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=output.name + ".",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = v1.sha256_file(output)
    digest_path.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--protocol-addendum", type=Path, required=True)
    parser.add_argument("--reporting-fix-addendum", type=Path, required=True)
    parser.add_argument("--base-metrics", type=Path, required=True)
    parser.add_argument(
        "--fine-metrics",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="repeat once for every preregistered seed",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.contract,
        args.protocol_addendum,
        args.reporting_fix_addendum,
        args.base_metrics,
        _parse_seed_metrics(args.fine_metrics),
    )
    digest = write_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "sha256": digest,
                "eligible": report["eligible"],
                "status": report["status"],
                "per_seed_failed_gates": {
                    seed: result["failed_gates"]
                    for seed, result in report["per_seed"].items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
