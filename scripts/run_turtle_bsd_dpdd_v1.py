#!/usr/bin/env python3
"""CPU-only plan generator for the staged BSD+DPDD experiment.

This program deliberately never executes a command.  It emits an immutable
argv/env/dependency plan after a bound CPU preflight, or a blocked template
receipt while BSD manifests remain unbound.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import (  # noqa: E402
    SEEDS,
    command_digest,
    load_contract,
    load_json_object,
    require,
    sha256_file,
    validate_protocol,
)


PYTHON = "/srv/szha0669/unblur-slam/env/bin/python"


def _action(
    action_id: str,
    argv: Sequence[str],
    *,
    depends_on: Sequence[str] = (),
    kind: str,
    output: str,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "kind": kind,
        "depends_on": list(depends_on),
        "env": {
            "CUDA_VISIBLE_DEVICES": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        },
        "argv": list(argv),
        "output": output,
        "overwrite": False,
    }


def _checkpoint(root: Path, arm: str, seed: int) -> Path:
    return root / "checkpoints" / arm / f"seed_{seed}.pth"


def build_reference_plan(
    contract_path: Path,
    contract: Mapping[str, Any],
    contract_sha: str,
    *,
    preflight_report: Path,
    preflight_sha: str,
) -> dict[str, Any]:
    """Build executable validation-prefix E/G/O actions only."""

    output_root = Path(contract["sealed_outputs"]["output_root"]).expanduser().resolve()
    common = [
        "--contract",
        str(contract_path),
        "--expected-contract-sha256",
        contract_sha,
        "--reference-only",
    ]
    actions: list[dict[str, Any]] = []
    for arm in ("E", "G", "O"):
        evaluator = (
            ROOT / "scripts/evaluate_evssm_bsd_validation.py"
            if arm == "E"
            else ROOT / "scripts/evaluate_turtle_bsd_streaming.py"
        )
        arm_args = [] if arm == "E" else ["--arm", arm]
        smoke_output = output_root / "smoke" / f"{arm}_bsd_first2"
        actions.append(
            _action(
                f"smoke_{arm}_bsd_first2",
                [
                    PYTHON,
                    str(evaluator),
                    *common,
                    *arm_args,
                    "--output-dir",
                    str(smoke_output),
                    "--device",
                    "cuda:0",
                    "--sequence-limit",
                    "2",
                ],
                kind="gpu_functional_smoke_identity_coverage_only",
                output=str(smoke_output),
            )
        )
        full_output = output_root / "references" / f"{arm}_bsd_validation"
        actions.append(
            _action(
                f"reference_{arm}_bsd_full",
                [
                    PYTHON,
                    str(evaluator),
                    *common,
                    *arm_args,
                    "--output-dir",
                    str(full_output),
                    "--device",
                    "cuda:0",
                ],
                depends_on=(f"smoke_{arm}_bsd_first2",),
                kind="gpu_reference_evaluation",
                output=str(full_output),
            )
        )

    e_metrics = output_root / "references/E_bsd_validation/metrics.json"
    g_metrics = output_root / "references/G_bsd_validation/metrics.json"
    o_metrics = output_root / "references/O_bsd_validation/metrics.json"
    gate = output_root / "receipts/reference_qualification.json"
    actions.append(
        _action(
            "cpu_reference_qualification_gate",
            [
                PYTHON,
                str(ROOT / "scripts/report_turtle_bsd_dpdd_v1.py"),
                "reference-gate",
                *common,
                "--E",
                str(e_metrics),
                "--G",
                str(g_metrics),
                "--O",
                str(o_metrics),
                "--output",
                str(gate),
            ],
            depends_on=(
                "reference_E_bsd_full",
                "reference_G_bsd_full",
                "reference_O_bsd_full",
            ),
            kind="cpu_operational_gate_no_quality_selection",
            output=str(gate),
        )
    )
    report = output_root / "validation_only_report.json"
    actions.append(
        _action(
            "cpu_reference_validation_report",
            [
                PYTHON,
                str(ROOT / "scripts/report_turtle_bsd_dpdd_v1.py"),
                "reference-report",
                *common,
                "--E",
                str(e_metrics),
                "--G",
                str(g_metrics),
                "--O",
                str(o_metrics),
                "--gate-receipt",
                str(gate),
                "--output",
                str(report),
            ],
            depends_on=("cpu_reference_qualification_gate",),
            kind="cpu_validation_only_report",
            output=str(report),
        )
    )
    argv_commands = [action["argv"] for action in actions]
    return {
        "schema": "unblur_slam.turtle_bsd_dpdd_execution_plan.v1",
        "status": "ready_reference_only_after_cpu_preflight",
        "mode": "reference_only",
        "contract": str(contract_path),
        "contract_sha256": contract_sha,
        "preflight_report": str(preflight_report),
        "preflight_report_sha256": preflight_sha,
        "output_root": str(output_root),
        "actions": actions,
        "argv_command_count": len(argv_commands),
        "argv_commands_sha256": command_digest(argv_commands),
        "runner_executes_commands": False,
        "executor_required": True,
        "gpu_queries_or_kernels_launched": False,
        "test_pixels_or_metrics_authorized": False,
        "training_actions": 0,
        "training_blocked": True,
        "code_bundle_sha256": contract["code_bundle"]["bundle_sha256"],
        "environment_fingerprint_sha256": contract["environment_fingerprint"][
            "fingerprint_sha256"
        ],
        "logical_runtime_mapping": {
            "physical_gpu": 1,
            "physical_gpu_uuid": contract["runtime"]["expected_gpu_uuid"],
            "physical_gpu_serial": contract["runtime"]["expected_gpu_serial"],
            "logical_device": "cuda:0",
        },
        "final_report_contract": {
            "schema": "unblur_slam.turtle_bsd_reference_validation_only_report.v1",
            "status": "reference_validation_complete_training_still_blocked",
            "contract_sha256": contract_sha,
            "arms": ["E", "G", "O"],
            "raw_baseline_required": True,
            "forbidden_claims": list(contract["claims_forbidden"]),
            "runtime_identity_required": True,
            "exact_top_level_keys": [
                "schema",
                "status",
                "contract_sha256",
                "arms",
                "raw_baseline",
                "quality",
                "model_minus_raw",
                "descriptive_deltas",
                "history_controls",
                "latency",
                "runtime_identity",
                "reference_semantics",
                "reference_gate_receipt",
                "input_reports",
                "decision_policy",
                "claim_scope",
                "slam_quality_or_speed_claim",
                "claims_policy",
                "bsd_test_pixels_opened",
                "bsd_test_authorized",
            ],
        },
    }


def build_plan(
    contract_path: Path,
    contract: Mapping[str, Any],
    contract_sha: str,
    *,
    preflight_report: Path,
    preflight_sha: str,
) -> dict[str, Any]:
    """Build the future full-data plan without touching any validation result.

    This builder is frozen while BSD train is still unbound.  It deliberately
    uses the same direct-float BSD E evaluator as reference-only execution, a
    real argv operational gate that never consults O quality, receipt-derived
    checkpoint hashes, and only evaluator arguments accepted by their parsers.
    """

    output_root = Path(contract["sealed_outputs"]["output_root"]).expanduser().resolve()
    dpdd = contract["data"]["dpdd"]
    models = contract["models"]
    common = ["--contract", str(contract_path), "--expected-contract-sha256", contract_sha]

    # Reuse the audited E/G/O action matrix and remove only the reference-only
    # flag.  The validation-only final report is replaced by the full three-seed
    # reporter below.
    reference = build_reference_plan(
        contract_path,
        contract,
        contract_sha,
        preflight_report=preflight_report,
        preflight_sha=preflight_sha,
    )
    actions: list[dict[str, Any]] = []
    for original in reference["actions"]:
        if original["id"] == "cpu_reference_validation_report":
            continue
        action = dict(original)
        action["argv"] = [value for value in original["argv"] if value != "--reference-only"]
        if action["id"] == "cpu_reference_qualification_gate":
            action["argv"][2] = "training-gate"
        actions.append(action)

    reference_gate_id = "cpu_reference_qualification_gate"
    bsd_reports: dict[tuple[str, int], Path] = {}
    training_ids: list[str] = []
    bsd_eval_ids: list[str] = []
    for seed in SEEDS:
        for arm in ("B", "BD"):
            checkpoint = _checkpoint(output_root, arm, seed)
            train_id = f"train_{arm}_seed{seed}"
            training_ids.append(train_id)
            actions.append(
                _action(
                    train_id,
                    [
                        PYTHON,
                        str(ROOT / "scripts/train_turtle_bsd_dpdd.py"),
                        *common,
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                        "--output",
                        str(checkpoint),
                        "--device",
                        "cuda:0",
                        "--amp",
                    ],
                    depends_on=(reference_gate_id,),
                    kind="gpu_terminal_only_training",
                    output=str(checkpoint),
                )
            )
            checkpoint_token = f"__FROM_{arm}_SEED{seed}_CHECKPOINT_SHA256_RECEIPT__"
            evaluation_output = output_root / "bsd_validation" / arm / f"seed_{seed}"
            evaluation_id = f"evaluate_{arm}_seed{seed}_bsd"
            bsd_eval_ids.append(evaluation_id)
            bsd_reports[(arm, seed)] = evaluation_output / "metrics.json"
            actions.append(
                _action(
                    evaluation_id,
                    [
                        PYTHON,
                        str(ROOT / "scripts/evaluate_turtle_bsd_streaming.py"),
                        *common,
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                        "--checkpoint",
                        str(checkpoint),
                        "--checkpoint-sha256",
                        checkpoint_token,
                        "--output-dir",
                        str(evaluation_output),
                        "--device",
                        "cuda:0",
                    ],
                    depends_on=(train_id,),
                    kind="gpu_validation_evaluation",
                    output=str(evaluation_output),
                )
            )

    dpdd_reports: dict[tuple[str, int], Path] = {}
    dpdd_reference_ids: list[str] = []
    evssm_dpdd_output = output_root / "references" / "E_dpdd_validation"
    actions.append(
        _action(
            "reference_E_dpdd_full",
            [
                PYTHON,
                str(ROOT / "scripts/evaluate_evssm_dpdd_validation.py"),
                *common,
                "--manifest",
                dpdd["validation_manifest"],
                "--manifest-sha256",
                dpdd["validation_manifest_sha256"],
                "--dataset-manifest",
                dpdd["dataset_manifest"],
                "--dataset-manifest-sha256",
                dpdd["dataset_manifest_sha256"],
                "--data-root",
                dpdd["root"],
                "--checkpoint",
                models["evssm_E"]["checkpoint"],
                "--device",
                "cuda:0",
                "--output-dir",
                str(evssm_dpdd_output),
            ],
            kind="gpu_reference_evaluation",
            output=str(evssm_dpdd_output),
        )
    )
    dpdd_reference_ids.append("reference_E_dpdd_full")
    for arm in ("G", "O"):
        evaluation_output = output_root / "references" / f"{arm}_dpdd_validation"
        action_id = f"reference_{arm}_dpdd_full"
        dpdd_reference_ids.append(action_id)
        actions.append(
            _action(
                action_id,
                [
                    PYTHON,
                    str(ROOT / "scripts/evaluate_turtle_bsd_dpdd.py"),
                    *common,
                    "--arm",
                    arm,
                    "--output-dir",
                    str(evaluation_output),
                    "--device",
                    "cuda:0",
                ],
                kind="gpu_reference_evaluation",
                output=str(evaluation_output),
            )
        )

    dpdd_eval_ids: list[str] = []
    for seed in SEEDS:
        for arm in ("B", "BD"):
            checkpoint = _checkpoint(output_root, arm, seed)
            checkpoint_token = f"__FROM_{arm}_SEED{seed}_CHECKPOINT_SHA256_RECEIPT__"
            evaluation_output = output_root / "dpdd_validation" / arm / f"seed_{seed}"
            action_id = f"evaluate_{arm}_seed{seed}_dpdd"
            dpdd_eval_ids.append(action_id)
            dpdd_reports[(arm, seed)] = evaluation_output / "metrics.json"
            actions.append(
                _action(
                    action_id,
                    [
                        PYTHON,
                        str(ROOT / "scripts/evaluate_turtle_bsd_dpdd.py"),
                        *common,
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                        "--checkpoint",
                        str(checkpoint),
                        "--checkpoint-sha256",
                        checkpoint_token,
                        "--output-dir",
                        str(evaluation_output),
                        "--device",
                        "cuda:0",
                    ],
                    depends_on=(f"train_{arm}_seed{seed}",),
                    kind="gpu_validation_evaluation",
                    output=str(evaluation_output),
                )
            )

    report_argv = [
        PYTHON,
        str(ROOT / "scripts/report_turtle_bsd_dpdd_v1.py"),
        "full-report",
        *common,
        "--gate-receipt",
        str(output_root / "receipts/reference_qualification.json"),
        "--E-bsd",
        str(output_root / "references/E_bsd_validation/metrics.json"),
        "--G-bsd",
        str(output_root / "references/G_bsd_validation/metrics.json"),
        "--O-bsd",
        str(output_root / "references/O_bsd_validation/metrics.json"),
        "--E-dpdd",
        str(evssm_dpdd_output / "metrics.json"),
        "--G-dpdd",
        str(output_root / "references/G_dpdd_validation/metrics.json"),
        "--O-dpdd",
        str(output_root / "references/O_dpdd_validation/metrics.json"),
    ]
    for seed in SEEDS:
        for arm in ("B", "BD"):
            report_argv.extend(
                [
                    f"--{arm}-bsd-seed{seed}",
                    str(bsd_reports[(arm, seed)]),
                    f"--{arm}-dpdd-seed{seed}",
                    str(dpdd_reports[(arm, seed)]),
                ]
            )
    final_report = output_root / "validation_only_report.json"
    report_argv.extend(["--output", str(final_report)])
    actions.append(
        _action(
            "cpu_full_validation_report",
            report_argv,
            depends_on=(
                reference_gate_id,
                *bsd_eval_ids,
                *dpdd_reference_ids,
                *dpdd_eval_ids,
            ),
            kind="cpu_full_validation_report_three_seed_conjunctive",
            output=str(final_report),
        )
    )

    argv_commands = [action["argv"] for action in actions]
    return {
        "schema": "unblur_slam.turtle_bsd_dpdd_execution_plan.v1",
        "status": "ready_after_bound_cpu_preflight_but_commands_not_executed",
        "mode": "full",
        "contract": str(contract_path),
        "contract_sha256": contract_sha,
        "preflight_report": str(preflight_report),
        "preflight_report_sha256": preflight_sha,
        "output_root": str(output_root),
        "actions": actions,
        "argv_command_count": len(argv_commands),
        "argv_commands_sha256": command_digest(argv_commands),
        "runner_executes_commands": False,
        "executor_required": True,
        "gpu_queries_or_kernels_launched": False,
        "test_pixels_or_metrics_authorized": False,
        "training_actions": len(training_ids),
        "training_blocked": False,
        "reference_gate_uses_quality": False,
        "code_bundle_sha256": contract["code_bundle"]["bundle_sha256"],
        "environment_fingerprint_sha256": contract["environment_fingerprint"][
            "fingerprint_sha256"
        ],
        "logical_runtime_mapping": {
            "physical_gpu": 1,
            "physical_gpu_uuid": contract["runtime"]["expected_gpu_uuid"],
            "physical_gpu_serial": contract["runtime"]["expected_gpu_serial"],
            "logical_device": "cuda:0",
        },
        "unresolved_post_training_tokens": "checkpoint SHA receipts must replace __FROM_* only after each O_EXCL checkpoint is hashed",
        "final_report_contract": {
            "schema": "unblur_slam.turtle_bsd_dpdd_full_validation_report.v1",
            "status": "full_validation_complete_three_seed_gate_reported",
            "contract_sha256": contract_sha,
            "arms": ["E", "G", "O", "B", "BD"],
            "raw_baseline_required": True,
            "forbidden_claims": list(contract["claims_forbidden"]),
            "runtime_identity_required": True,
            "exact_top_level_keys": [
                "schema",
                "status",
                "contract_sha256",
                "arms",
                "seeds",
                "raw_baseline",
                "quality",
                "model_minus_raw",
                "latency",
                "primary_gates",
                "reference_gate_receipt",
                "input_reports",
                "runtime_identity",
                "reference_semantics",
                "claim_scope",
                "slam_quality_or_speed_claim",
                "claims_policy",
                "bsd_test_pixels_opened",
                "dpdd_test_pixels_opened",
                "test_authorized",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path, contract, contract_sha = load_contract(
        args.contract,
        expected_sha256=args.expected_contract_sha256,
    )
    require(not (args.template and args.reference_only), "--template and --reference-only are exclusive")
    unbound = validate_protocol(
        contract,
        allow_template=args.template,
        reference_only=args.reference_only,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"plan output already exists: {output}")
    if args.template:
        payload = {
            "schema": "unblur_slam.turtle_bsd_dpdd_execution_plan.v1",
            "status": "blocked_unbound_template",
            "contract": str(contract_path),
            "contract_sha256": contract_sha,
            "unbound_fields": unbound,
            "actions": [],
            "mode": "template_blocked",
            "training_actions": 0,
            "training_blocked": True,
            "block_reason": "BSD train manifest/path/hash remain unbound",
            "future_full_bound_plan_builder_frozen": True,
            "code_bundle_sha256": contract["code_bundle"]["bundle_sha256"],
            "environment_fingerprint_sha256": contract[
                "environment_fingerprint"
            ]["fingerprint_sha256"],
            "runner_executes_commands": False,
            "gpu_queries_or_kernels_launched": False,
            "test_pixels_or_metrics_authorized": False,
        }
    else:
        require(args.preflight_report is not None, "bound planning requires --preflight-report")
        preflight_path = args.preflight_report.expanduser().resolve()
        preflight = load_json_object(preflight_path)
        expected_preflight_status = "pass_reference_only" if args.reference_only else "pass"
        require(preflight.get("status") == expected_preflight_status, "CPU preflight report did not pass in the requested mode")
        require(preflight.get("contract_sha256") == contract_sha, "CPU preflight contract mismatch")
        require(preflight.get("reference_only") is args.reference_only, "CPU preflight mode mismatch")
        require(preflight.get("gpu_queries_or_kernels_launched") is False, "preflight used GPU")
        require(preflight.get("bsd_test_pixels_opened") is False, "preflight touched BSD test")
        require(
            preflight.get("code_bundle_sha256")
            == contract["code_bundle"]["bundle_sha256"],
            "CPU preflight code-bundle fingerprint changed",
        )
        require(
            preflight.get("environment_fingerprint_sha256")
            == contract["environment_fingerprint"]["fingerprint_sha256"],
            "CPU preflight environment fingerprint changed",
        )
        require(
            preflight.get("logical_runtime_mapping")
            == {
                "physical_gpu": 1,
                "physical_gpu_uuid": contract["runtime"]["expected_gpu_uuid"],
                "physical_gpu_serial": contract["runtime"]["expected_gpu_serial"],
                "logical_device": "cuda:0",
            },
            "CPU preflight logical-device mapping changed",
        )
        builder = build_reference_plan if args.reference_only else build_plan
        payload = builder(
            contract_path,
            contract,
            contract_sha,
            preflight_report=preflight_path,
            preflight_sha=sha256_file(preflight_path),
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(output, 0o444)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(json.dumps({"plan": str(output), "status": payload["status"], "commands": len(payload["actions"])}))


if __name__ == "__main__":
    main()
