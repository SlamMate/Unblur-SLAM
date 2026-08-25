#!/usr/bin/env python3
"""Execute one frozen BSD+DPDD plan serially under the physical-GPU1 lock.

Unlike the plan generator, this command really runs actions.  It verifies the
content-addressed contract, preflight and plan, requires GPU1→logical cuda:0 on
an RTX A6000, holds one global lock for the entire run, creates immutable action
receipts, substitutes checkpoint hashes only from verified receipts, and stops
at the first failed dependency/gate.  It never supports a test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import (  # noqa: E402
    command_digest,
    load_contract,
    load_json_object,
    require,
    sha256_file,
    validate_protocol,
)
from scripts.bsd_dpdd_runtime import (  # noqa: E402
    CHILD_RUNTIME_ENVIRONMENT,
    CODE_BUNDLE_SCHEMA,
    DEFAULT_GLOBAL_LOCK,
    ENVIRONMENT_SCHEMA,
    EXPECTED_CUDA_DEVICE_ORDER,
    EXPECTED_GPU_NAME,
    EXPECTED_GPU_SERIAL,
    EXPECTED_GPU_UUID,
    EXPECTED_VISIBLE_DEVICE,
    RUNTIME_IDENTITY_SCHEMA,
    child_runtime_environment,
    exclusive_gpu1_lock,
    inspect_physical_gpu1,
    rehash_frozen_code_bundle,
    require_exact_cuda_environment,
    require_gpu1_a6000,
    require_only_known_compute_pids,
    verify_frozen_environment,
)


SCHEMA = "unblur_slam.turtle_bsd_dpdd_execution_receipt.v1"
PLAN_SCHEMA = "unblur_slam.turtle_bsd_dpdd_execution_plan.v1"
PYTHON = "/srv/szha0669/unblur-slam/env/bin/python"
CHECKPOINT_TOKEN = re.compile(
    r"^__FROM_(B|BD)_SEED(17|42|73)_CHECKPOINT_SHA256_RECEIPT__$"
)
ACTION_RECEIPT_SCHEMA = "unblur_slam.turtle_bsd_dpdd_action_receipt.v1"
E_REPORT_SCHEMA = "unblur_slam.evssm_bsd3ms24ms_direct_float_validation.v1"
TURTLE_REPORT_SCHEMA = "unblur_slam.turtle_streaming_evaluation.v1"
DPDD_E_REPORT_SCHEMA = "unblur_slam.official_evssm_dpdd_validation.v1"
DPDD_TURTLE_REPORT_SCHEMA = "unblur_slam.turtle_bsd_dpdd_validation_arm.v1"
REFERENCE_GATE_SCHEMA = "unblur_slam.turtle_bsd_reference_qualification_receipt.v1"
REFERENCE_REPORT_SCHEMA = "unblur_slam.turtle_bsd_reference_validation_only_report.v1"
REFERENCE_REPORT_STATUS = "reference_validation_complete_training_still_blocked"
GPU_MONITOR_INTERVAL_SECONDS = 1.0
GPU_CONTEXT_RELEASE_TIMEOUT_SECONDS = 30.0


class ExecutionError(RuntimeError):
    """Raised before downstream work when execution provenance is invalid."""


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _under_output(path: Path | str, output_root: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    try:
        candidate.relative_to(output_root)
    except ValueError as error:
        raise ExecutionError(f"{label} escapes frozen output root: {candidate}") from error
    if candidate == output_root:
        raise ExecutionError(f"{label} may not equal broad output root")
    return candidate


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_tree(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"non-finite report value at {path}")


def _validate_final_report_contract_spec(
    value: Any,
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    reference_only: bool,
) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), "plan final_report_contract is missing")
    expected_keys = {
        "schema",
        "status",
        "contract_sha256",
        "arms",
        "raw_baseline_required",
        "forbidden_claims",
        "runtime_identity_required",
        "exact_top_level_keys",
    }
    require(set(value) == expected_keys, "plan final_report_contract fields changed")
    require(value.get("contract_sha256") == contract_sha256, "final-report contract hash changed")
    require(value.get("raw_baseline_required") is True, "final report no longer requires raw baseline")
    require(value.get("runtime_identity_required") is True, "final report no longer requires runtime identity")
    require(
        value.get("forbidden_claims") == contract.get("claims_forbidden"),
        "final report forbidden-claim policy differs from the frozen contract",
    )
    arms = value.get("arms")
    require(isinstance(arms, list) and len(arms) == len(set(arms)), "final report arms are invalid")
    if reference_only:
        require(value.get("schema") == REFERENCE_REPORT_SCHEMA, "reference report schema changed")
        require(value.get("status") == REFERENCE_REPORT_STATUS, "reference report status changed")
        require(arms == ["E", "G", "O"], "reference report arms changed")
    else:
        require(
            value.get("schema") == "unblur_slam.turtle_bsd_dpdd_full_validation_report.v1",
            "full report schema changed",
        )
        require(
            value.get("status") == "full_validation_complete_three_seed_gate_reported",
            "full report status changed",
        )
        require(arms == ["E", "G", "O", "B", "BD"], "full report arms changed")
    top_level = value.get("exact_top_level_keys")
    require(
        isinstance(top_level, list)
        and top_level
        and len(top_level) == len(set(top_level))
        and all(isinstance(key, str) and key for key in top_level),
        "final report exact_top_level_keys are invalid",
    )
    for required_key in (
        "schema",
        "status",
        "contract_sha256",
        "arms",
        "raw_baseline",
        "claims_policy",
        "runtime_identity",
        "bsd_test_pixels_opened",
        "slam_quality_or_speed_claim",
    ):
        require(required_key in top_level, f"final report key is not sealed: {required_key}")
    if reference_only:
        require("bsd_test_authorized" in top_level, "reference report test authorization is not sealed")
    else:
        for required_key in ("dpdd_test_pixels_opened", "test_authorized"):
            require(required_key in top_level, f"full report key is not sealed: {required_key}")
    return value


def validate_execution_plan(
    plan_path: Path | str,
    *,
    expected_plan_sha256: str,
    contract_path: Path,
    contract_sha256: str,
    reference_only: bool,
) -> tuple[Path, Mapping[str, Any]]:
    path = Path(plan_path).expanduser().resolve()
    require(path.is_file(), f"execution plan does not exist: {path}")
    require(sha256_file(path) == expected_plan_sha256.lower(), "execution plan SHA256 mismatch")
    plan = load_json_object(path)
    contract_payload = load_json_object(contract_path)
    require(plan.get("schema") == PLAN_SCHEMA, "execution plan schema changed")
    wanted_status = (
        "ready_reference_only_after_cpu_preflight"
        if reference_only
        else "ready_after_bound_cpu_preflight_but_commands_not_executed"
    )
    require(plan.get("status") == wanted_status, "execution plan status/mode changed")
    require(
        plan.get("mode") == ("reference_only" if reference_only else "full"),
        "execution plan mode changed",
    )
    require(
        plan.get("training_actions") == (0 if reference_only else 6),
        "execution plan training-action count changed",
    )
    require(
        plan.get("training_blocked") is reference_only,
        "execution plan training-blocked state changed",
    )
    require(Path(plan.get("contract", "")).resolve() == contract_path, "plan contract path changed")
    require(plan.get("contract_sha256") == contract_sha256, "plan contract SHA changed")
    require(
        plan.get("code_bundle_sha256")
        == contract_payload.get("code_bundle", {}).get("bundle_sha256"),
        "plan code-bundle SHA differs from contract",
    )
    require(
        plan.get("environment_fingerprint_sha256")
        == contract_payload.get("environment_fingerprint", {}).get(
            "fingerprint_sha256"
        ),
        "plan environment fingerprint differs from contract",
    )
    logical_mapping = {
        "physical_gpu": 1,
        "physical_gpu_uuid": EXPECTED_GPU_UUID,
        "physical_gpu_serial": EXPECTED_GPU_SERIAL,
        "logical_device": "cuda:0",
    }
    require(
        plan.get("logical_runtime_mapping") == logical_mapping,
        "plan logical runtime mapping changed",
    )
    require(plan.get("runner_executes_commands") is False, "plan falsely claims prior execution")
    require(plan.get("test_pixels_or_metrics_authorized") is False, "plan authorized a test split")
    actions = plan.get("actions")
    require(isinstance(actions, list) and actions, "execution plan has no actions")
    ids = [action.get("id") for action in actions]
    require(all(isinstance(value, str) and value for value in ids), "plan action id is invalid")
    require(len(ids) == len(set(ids)), "plan action ids are duplicated")
    seen: set[str] = set()
    argv_commands = []
    for action in actions:
        dependencies = action.get("depends_on", [])
        require(isinstance(dependencies, list), "action dependencies must be a list")
        require(set(dependencies) <= seen, f"action {action['id']} has a forward/missing dependency")
        require(action.get("overwrite") is False, f"action {action['id']} permits overwrite")
        require(
            action.get("env")
            == {
                "CUDA_VISIBLE_DEVICES": EXPECTED_VISIBLE_DEVICE,
                "CUDA_DEVICE_ORDER": EXPECTED_CUDA_DEVICE_ORDER,
            },
            f"action {action['id']} environment changed",
        )
        argv = action.get("argv")
        require(isinstance(argv, list) and len(argv) >= 2, f"action {action['id']} has no argv")
        require(argv[0] == PYTHON, f"action {action['id']} Python environment changed")
        require("test" not in str(action.get("kind", "")).lower(), f"action {action['id']} is test-like")
        argv_commands.append(argv)
        seen.add(action["id"])
    require(plan.get("argv_command_count") == len(argv_commands), "plan argv count changed")
    require(plan.get("argv_commands_sha256") == command_digest(argv_commands), "plan argv digest changed")
    preflight_path = Path(plan.get("preflight_report", "")).expanduser().resolve()
    require(preflight_path.is_file(), "plan preflight receipt is missing")
    require(sha256_file(preflight_path) == plan.get("preflight_report_sha256"), "plan preflight receipt SHA changed")
    preflight = load_json_object(preflight_path)
    require(preflight.get("contract_sha256") == contract_sha256, "preflight contract changed")
    require(preflight.get("status") == ("pass_reference_only" if reference_only else "pass"), "preflight mode/status changed")
    require(preflight.get("gpu_queries_or_kernels_launched") is False, "CPU preflight used GPU")
    require(preflight.get("bsd_test_pixels_opened") is False, "CPU preflight touched BSD test")
    require(
        preflight.get("code_bundle_sha256") == plan.get("code_bundle_sha256"),
        "preflight/plan code-bundle SHA differs",
    )
    require(
        preflight.get("environment_fingerprint_sha256")
        == plan.get("environment_fingerprint_sha256"),
        "preflight/plan environment fingerprint differs",
    )
    require(
        preflight.get("logical_runtime_mapping") == logical_mapping,
        "preflight logical runtime mapping changed",
    )
    _validate_final_report_contract_spec(
        plan.get("final_report_contract"),
        contract=contract_payload,
        contract_sha256=contract_sha256,
        reference_only=reference_only,
    )
    return path, plan


def _resolve_checkpoint_tokens(
    argv: Sequence[str], checkpoint_receipts: Mapping[tuple[str, int], Mapping[str, Any]]
) -> list[str]:
    resolved: list[str] = []
    for value in argv:
        match = CHECKPOINT_TOKEN.fullmatch(str(value))
        if match is None:
            require(not str(value).startswith("__FROM_"), f"unknown unresolved receipt token: {value}")
            resolved.append(str(value))
            continue
        key = (match.group(1), int(match.group(2)))
        receipt = checkpoint_receipts.get(key)
        require(receipt is not None, f"checkpoint receipt is unavailable for {key}")
        digest = str(receipt.get("checkpoint_sha256", ""))
        require(len(digest) == 64, f"checkpoint receipt hash is invalid for {key}")
        resolved.append(digest)
    return resolved


def _checkpoint_identity(action_id: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"train_(B|BD)_seed(17|42|73)", action_id)
    return None if match is None else (match.group(1), int(match.group(2)))


def _extract_and_validate_runtime_identity(
    payload: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    candidates = []
    if isinstance(payload.get("runtime_identity"), Mapping):
        candidates.append(payload["runtime_identity"])
    if isinstance(payload.get("runtime"), Mapping):
        candidates.append(payload["runtime"])
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping) and isinstance(provenance.get("runtime"), Mapping):
        candidates.append(provenance["runtime"])
    protocol = payload.get("protocol")
    if isinstance(protocol, Mapping) and isinstance(protocol.get("runtime"), Mapping):
        candidates.append(protocol["runtime"])
    require(len(candidates) == 1, f"{label} must contain exactly one runtime identity")
    observed = candidates[0]
    require(observed.get("schema") == RUNTIME_IDENTITY_SCHEMA, f"{label} runtime schema changed")
    require(
        observed.get("identity_sha256") == expected.get("identity_sha256"),
        f"{label} logical cuda:0 identity digest changed",
    )
    require(dict(observed) == dict(expected), f"{label} runtime identity differs from executor")
    return observed


def _validate_metric_mapping(
    value: Any, *, metrics: set[str], label: str
) -> None:
    require(
        isinstance(value, Mapping) and set(value) == metrics,
        f"{label} metrics changed",
    )
    for metric, number in value.items():
        require(
            isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(float(number)),
            f"{label}.{metric} is not finite",
        )


def _validate_bsd_raw_baseline(raw: Any, *, full: bool) -> None:
    require(isinstance(raw, Mapping), "final report raw_baseline is missing")
    wanted = {
            "source",
            "identical_across_reported_arms",
            "all_frames",
            "steady",
            "per_sequence",
            "registration_sha256",
        }
    if not full:
        wanted.update(
            {
                "per_frame_E_G_O_identity_equal",
                "per_frame_E_G_O_metrics_abs_tolerance",
            }
        )
    require(
        set(raw) == wanted,
        "final report raw_baseline fields changed",
    )
    require(
        raw.get("source") == "common_blurry_input_vs_sharp_ground_truth",
        "final report raw baseline source changed",
    )
    require(
        raw.get("identical_across_reported_arms") is True,
        "raw baseline was not identical across reported arms",
    )
    for aggregation in ("all_frames", "steady"):
        _validate_metric_mapping(
            raw.get(aggregation), metrics={"psnr", "ssim", "l1"}, label=f"raw baseline {aggregation}"
        )
    sequences = raw.get("per_sequence")
    require(isinstance(sequences, Mapping) and len(sequences) == 20, "raw BSD sequence coverage changed")
    for name, row in sequences.items():
        require(
            isinstance(name, str) and isinstance(row, Mapping),
            "raw BSD per-sequence row changed",
        )
        require(
            set(row) == {"all_frames", "steady", "frame_count", "steady_frame_count"},
            f"raw BSD sequence fields changed: {name}",
        )
        require(row.get("frame_count") == 100 and row.get("steady_frame_count") == 97, f"raw BSD sequence coverage changed: {name}")
        _validate_metric_mapping(row.get("all_frames"), metrics={"psnr", "ssim", "l1"}, label=f"raw BSD {name} all")
        _validate_metric_mapping(row.get("steady"), metrics={"psnr", "ssim", "l1"}, label=f"raw BSD {name} steady")
    require(
        re.fullmatch(r"[0-9a-f]{64}", str(raw.get("registration_sha256", ""))) is not None,
        "raw BSD registration SHA256 changed",
    )
    if not full:
        require(raw.get("per_frame_E_G_O_identity_equal") is True, "raw E/G/O identities differ")
        require(
            raw.get("per_frame_E_G_O_metrics_abs_tolerance") == 1.0e-12,
            "raw E/G/O comparison tolerance changed",
        )


def _validate_raw_baseline(raw: Any, *, full: bool) -> None:
    if not full:
        _validate_bsd_raw_baseline(raw, full=False)
        return
    require(isinstance(raw, Mapping) and set(raw) == {"BSD", "DPDD"}, "full raw baseline datasets changed")
    _validate_bsd_raw_baseline(raw["BSD"], full=True)
    dpdd = raw["DPDD"]
    require(
        isinstance(dpdd, Mapping)
        and set(dpdd) == {"source", "identical_across_reported_arms", "all_images"},
        "full DPDD raw baseline fields changed",
    )
    require(dpdd.get("source") == "common_defocus_input_vs_sharp_ground_truth", "DPDD raw source changed")
    require(dpdd.get("identical_across_reported_arms") is True, "DPDD raw differs across arms")
    _validate_metric_mapping(
        dpdd.get("all_images"),
        metrics={"psnr", "ssim", "lpips", "l1"},
        label="raw DPDD all images",
    )


def _validate_final_report_payload(
    payload: Mapping[str, Any],
    *,
    specification: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> None:
    require(
        set(payload) == set(specification["exact_top_level_keys"]),
        "final report top-level fields differ from the sealed plan",
    )
    for key in ("schema", "status", "contract_sha256", "arms"):
        require(payload.get(key) == specification.get(key), f"final report {key} changed")
    require(payload.get("arms") == specification["arms"], "final report arm order changed")
    full = specification.get("arms") == ["E", "G", "O", "B", "BD"]
    _validate_raw_baseline(payload.get("raw_baseline"), full=full)
    policy = payload.get("claims_policy")
    require(isinstance(policy, Mapping), "final report claims_policy is missing")
    require(
        set(policy)
        == {
            "forbidden_claims",
            "forbidden_claims_made",
            "all_forbidden_claims_excluded",
        },
        "final report claims_policy fields changed",
    )
    require(
        policy.get("forbidden_claims") == specification["forbidden_claims"],
        "final report forbidden-claim list changed",
    )
    require(policy.get("forbidden_claims_made") == [], "final report made a forbidden claim")
    require(
        policy.get("all_forbidden_claims_excluded") is True,
        "final report did not exclude every forbidden claim",
    )
    require(payload.get("bsd_test_pixels_opened") is False, "final report touched BSD test pixels")
    if full:
        require(payload.get("dpdd_test_pixels_opened") is False, "final report touched DPDD test pixels")
        require(payload.get("test_authorized") is False, "final report authorized a test split")
    else:
        require(payload.get("bsd_test_authorized") is False, "final report authorized BSD test")
    require(
        payload.get("slam_quality_or_speed_claim") is False,
        "final report made a SLAM quality/speed claim",
    )
    _extract_and_validate_runtime_identity(payload, runtime_identity, label="final report")


def _verify_action_output(
    action: Mapping[str, Any],
    output_root: Path,
    *,
    contract_sha256: str,
    runtime_identity: Mapping[str, Any],
    final_report_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    output = _under_output(action["output"], output_root, label=f"{action['id']} output")
    kind = str(action["kind"])
    if kind == "gpu_terminal_only_training":
        require(output.is_file(), f"training checkpoint missing: {output}")
        sidecar = output.with_name(output.name + ".sha256")
        require(sidecar.is_file(), f"training checkpoint SHA sidecar missing: {sidecar}")
        digest = sha256_file(output)
        expected_line = f"{digest}  {output.name}\n"
        require(sidecar.read_text(encoding="utf-8") == expected_line, "checkpoint SHA sidecar changed")
        return {
            "kind": "checkpoint",
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": digest,
            "sha256_sidecar": str(sidecar),
            "sha256_sidecar_sha256": sha256_file(sidecar),
        }
    report = output / "metrics.json" if output.is_dir() else output
    require(report.is_file(), f"action report missing: {report}")
    payload = load_json_object(report)
    _finite_tree(payload)
    _extract_and_validate_runtime_identity(payload, runtime_identity, label=str(action["id"]))
    if action["id"].startswith("smoke_E_"):
        require(payload.get("schema") == E_REPORT_SCHEMA, "E smoke schema changed")
        require(payload.get("arm") == "E", "E smoke arm changed")
        require(payload.get("formal") is True, "E smoke is not formal")
        require(
            payload.get("protocol", {}).get("contract_sha256") == contract_sha256,
            "E smoke contract changed",
        )
        require(payload.get("results", {}).get("sequence_count") == 2, "E smoke sequence coverage changed")
        require(payload.get("results", {}).get("frame_count") == 200, "E smoke frame coverage changed")
    elif action["id"].startswith(("smoke_G_", "smoke_O_")):
        arm = str(action["id"]).split("_")[1]
        require(payload.get("schema") == TURTLE_REPORT_SCHEMA, f"{arm} smoke schema changed")
        require(payload.get("provenance", {}).get("arm") == arm, f"{arm} smoke arm changed")
        require(
            payload.get("provenance", {}).get("contract_sha256") == contract_sha256,
            f"{arm} smoke contract changed",
        )
        require(payload.get("sequence_count") == 2 and payload.get("frame_count") == 200, "TURTLE smoke coverage changed")
    elif action["id"] == "reference_E_bsd_full":
        require(
            payload.get("schema") == E_REPORT_SCHEMA
            and payload.get("formal") is True
            and payload.get("arm") == "E",
            "full E report identity changed",
        )
        require(
            payload.get("protocol", {}).get("contract_sha256") == contract_sha256,
            "full E report contract changed",
        )
        require(
            payload.get("results", {}).get("sequence_count") == 20
            and payload.get("results", {}).get("frame_count") == 2000,
            "full E BSD coverage changed",
        )
    elif action["id"] in {"reference_G_bsd_full", "reference_O_bsd_full"}:
        arm = str(action["id"]).split("_")[1]
        require(payload.get("schema") == TURTLE_REPORT_SCHEMA, f"full {arm} schema changed")
        require(payload.get("provenance", {}).get("arm") == arm, f"full {arm} arm changed")
        require(
            payload.get("provenance", {}).get("contract_sha256") == contract_sha256,
            f"full {arm} contract changed",
        )
        require(
            payload.get("sequence_count") == 20 and payload.get("frame_count") == 2000,
            f"full {arm} BSD coverage changed",
        )
    elif re.fullmatch(r"evaluate_(B|BD)_seed(17|42|73)_bsd", str(action["id"])):
        match = re.fullmatch(r"evaluate_(B|BD)_seed(17|42|73)_bsd", str(action["id"]))
        assert match is not None
        arm, seed = match.group(1), int(match.group(2))
        require(payload.get("schema") == TURTLE_REPORT_SCHEMA, f"{arm}/{seed} BSD schema changed")
        require(payload.get("provenance", {}).get("arm") == arm, f"{arm}/{seed} BSD arm changed")
        require(payload.get("provenance", {}).get("seed") == seed, f"{arm}/{seed} BSD seed changed")
        require(
            payload.get("provenance", {}).get("contract_sha256") == contract_sha256,
            f"{arm}/{seed} BSD contract changed",
        )
        require(
            payload.get("sequence_count") == 20 and payload.get("frame_count") == 2000,
            f"{arm}/{seed} BSD coverage changed",
        )
    elif action["id"] == "reference_E_dpdd_full":
        require(
            payload.get("schema") == DPDD_E_REPORT_SCHEMA
            and payload.get("formal") is True
            and payload.get("arm") == "E",
            "full E DPDD report identity changed",
        )
        require(
            payload.get("protocol", {}).get("contract_sha256") == contract_sha256,
            "full E DPDD contract changed",
        )
        require(
            payload.get("results", {}).get("pair_count") == 74,
            "full E DPDD coverage changed",
        )
        require(
            payload.get("protocol", {}).get("test_pixels_opened") is False
            and payload.get("protocol", {}).get("test_metrics_computed") is False,
            "full E DPDD report touched test",
        )
    elif action["id"] in {"reference_G_dpdd_full", "reference_O_dpdd_full"}:
        arm = str(action["id"]).split("_")[1]
        require(
            payload.get("schema") == DPDD_TURTLE_REPORT_SCHEMA
            and payload.get("formal") is True
            and payload.get("arm") == arm,
            f"full {arm} DPDD report identity changed",
        )
        require(
            payload.get("protocol", {}).get("contract_sha256") == contract_sha256,
            f"full {arm} DPDD contract changed",
        )
        require(payload.get("protocol", {}).get("pair_count") == 74, f"full {arm} DPDD coverage changed")
        require(
            payload.get("protocol", {}).get("dpdd_test_pixels_opened") is False,
            f"full {arm} DPDD report touched test",
        )
    elif re.fullmatch(r"evaluate_(B|BD)_seed(17|42|73)_dpdd", str(action["id"])):
        match = re.fullmatch(r"evaluate_(B|BD)_seed(17|42|73)_dpdd", str(action["id"]))
        assert match is not None
        arm, seed = match.group(1), int(match.group(2))
        require(
            payload.get("schema") == DPDD_TURTLE_REPORT_SCHEMA
            and payload.get("formal") is True,
            f"{arm}/{seed} DPDD schema/formal flag changed",
        )
        require(payload.get("arm") == arm and payload.get("seed") == seed, f"{arm}/{seed} DPDD identity changed")
        require(
            payload.get("protocol", {}).get("contract_sha256") == contract_sha256,
            f"{arm}/{seed} DPDD contract changed",
        )
        require(payload.get("protocol", {}).get("pair_count") == 74, f"{arm}/{seed} DPDD coverage changed")
        require(
            payload.get("protocol", {}).get("dpdd_test_pixels_opened") is False,
            f"{arm}/{seed} DPDD report touched test",
        )
    if action["id"] == "cpu_reference_qualification_gate":
        require(payload.get("schema") == REFERENCE_GATE_SCHEMA, "reference gate schema changed")
        require(payload.get("status") == "pass", "reference qualification gate failed")
        require(payload.get("contract_sha256") == contract_sha256, "reference gate contract changed")
        require(payload.get("quality_selection", {}).get("O_quality_used_to_authorize_training") is False, "O quality leaked into reference gate")
        require(payload.get("training_authorized_by_this_receipt") is False, "reference gate authorized training")
        require(payload.get("bsd_test_pixels_opened") is False, "reference gate touched BSD test")
    if action["id"] in {"cpu_reference_validation_report", "cpu_full_validation_report"}:
        _validate_final_report_payload(
            payload,
            specification=final_report_contract,
            runtime_identity=runtime_identity,
        )
    return {
        "kind": "report",
        "path": str(report),
        "bytes": report.stat().st_size,
        "sha256": sha256_file(report),
        "schema": payload.get("schema"),
        "reported_status": payload.get("status"),
        "runtime_identity_sha256": runtime_identity["identity_sha256"],
    }


def _terminate_owned_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_monitored_command(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    stdout: Any,
    stderr: Any,
    action_id: str,
) -> tuple[int, Mapping[str, Any]]:
    """Run one child while rejecting every unexpected physical-GPU1 PID."""

    # The parent may retain a zero-allocation CUDA context after its logical
    # identity check.  It and this one direct child are the only permitted PIDs.
    executor_pid = os.getpid()
    before = require_only_known_compute_pids(
        [executor_pid], phase=f"between_actions_before_{action_id}"
    )
    process = subprocess.Popen(
        list(argv),
        cwd=ROOT,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    samples = 0
    maximum_compute_processes = 0
    observed: dict[int, str] = {}
    digest = hashlib.sha256()

    def sample(phase: str) -> list[Mapping[str, Any]]:
        nonlocal samples, maximum_compute_processes
        rows = require_only_known_compute_pids(
            [executor_pid, process.pid], phase=phase
        )
        samples += 1
        maximum_compute_processes = max(maximum_compute_processes, len(rows))
        canonical_rows = [dict(row) for row in rows]
        digest.update(
            json.dumps(
                canonical_rows,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
        for row in rows:
            observed[int(row["pid"])] = str(row["process_name"])
        return rows

    try:
        while True:
            sample(f"monitor_{action_id}")
            returncode = process.poll()
            if returncode is not None:
                break
            time.sleep(GPU_MONITOR_INTERVAL_SECONDS)

        deadline = time.monotonic() + GPU_CONTEXT_RELEASE_TIMEOUT_SECONDS
        while True:
            rows = sample(f"context_release_{action_id}")
            if all(int(row["pid"]) != process.pid for row in rows):
                break
            require(
                time.monotonic() < deadline,
                f"action {action_id} CUDA context did not release within "
                f"{GPU_CONTEXT_RELEASE_TIMEOUT_SECONDS:g}s",
            )
            time.sleep(0.25)
        after = require_only_known_compute_pids(
            [executor_pid], phase=f"between_actions_after_{action_id}"
        )
    except BaseException:
        _terminate_owned_process(process)
        raise

    return int(returncode), {
        "schema": "unblur_slam.physical_gpu1_compute_process_monitor.v1",
        "status": "pass_no_unfamiliar_compute_pid",
        "physical_gpu_index": 1,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "executor_pid": executor_pid,
        "action_pid": process.pid,
        "allowed_pids": [executor_pid, process.pid],
        "before_action_compute_pids": [int(row["pid"]) for row in before],
        "after_action_compute_pids": [int(row["pid"]) for row in after],
        "sample_count": samples,
        "sample_interval_seconds": GPU_MONITOR_INTERVAL_SECONDS,
        "maximum_observed_compute_processes": maximum_compute_processes,
        "observed_known_processes": [
            {"pid": pid, "process_name": observed[pid]} for pid in sorted(observed)
        ],
        "unfamiliar_compute_pids": [],
        "samples_sha256": digest.hexdigest(),
    }


def execute(
    *,
    contract_path: Path,
    contract: Mapping[str, Any],
    contract_sha: str,
    plan_path: Path,
    plan: Mapping[str, Any],
    plan_sha: str,
    runtime_identity: Mapping[str, Any],
    lock_identity: Mapping[str, Any],
    physical_identity: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
    code_bundle_identity: Mapping[str, Any],
    implementation_path_map: Mapping[str, Path | str],
) -> Path:
    output_root = Path(plan["output_root"]).expanduser().resolve()
    require(str(output_root).startswith("/srv/szha0669/"), "formal output root must remain on /srv")
    require(output_root == Path(contract["sealed_outputs"]["output_root"]).resolve(), "plan output root changed")
    require(not output_root.exists(), f"formal output root already exists: {output_root}")
    output_root.mkdir(parents=True, mode=0o700)
    receipts_dir = output_root / "receipts/actions"
    logs_dir = output_root / "logs"
    receipts_dir.mkdir(parents=True, mode=0o700)
    logs_dir.mkdir(parents=True, mode=0o700)

    completed: set[str] = set()
    action_receipts: list[Mapping[str, Any]] = []
    checkpoint_receipts: MutableMapping[tuple[str, int], Mapping[str, Any]] = {}
    for action in plan["actions"]:
        action_id = str(action["id"])
        # Recheck before every process so a mid-run source/environment mutation
        # cannot silently enter a later arm.
        current_bundle = rehash_frozen_code_bundle(
            implementation_pins=contract["implementation_pins"],
            code_bundle=contract["code_bundle"],
            expected_path_map=implementation_path_map,
            base_root=ROOT,
        )
        require(
            current_bundle["bundle_sha256"] == code_bundle_identity["bundle_sha256"],
            f"code bundle changed before action {action_id}",
        )
        current_environment = verify_frozen_environment(
            contract["environment_fingerprint"]
        )
        require(
            current_environment["fingerprint_sha256"]
            == environment_identity["fingerprint_sha256"],
            f"Python/package environment changed before action {action_id}",
        )
        require(set(action["depends_on"]) <= completed, f"dependencies not complete for {action_id}")
        output = _under_output(action["output"], output_root, label=f"{action_id} output")
        require(not output.exists(), f"action output already exists before launch: {output}")
        argv = _resolve_checkpoint_tokens(action["argv"], checkpoint_receipts)
        require(argv[0] == PYTHON, f"action {action_id} interpreter changed")
        if str(action["kind"]).startswith("gpu_"):
            require("--device" in argv and argv[argv.index("--device") + 1] == "cuda:0", f"GPU action {action_id} device changed")

        stdout_path = logs_dir / f"{action_id}.stdout.log"
        stderr_path = logs_dir / f"{action_id}.stderr.log"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        stdout_fd = os.open(stdout_path, flags, 0o444)
        stderr_fd = os.open(stderr_path, flags, 0o444)
        environment = child_runtime_environment(
            os.environ,
            lock_identity=lock_identity,
            environment_fingerprint_sha256=environment_identity[
                "fingerprint_sha256"
            ],
            code_bundle_sha256=code_bundle_identity["bundle_sha256"],
        )
        environment["TMPDIR"] = "/srv/szha0669/tmp"
        try:
            with os.fdopen(stdout_fd, "w", encoding="utf-8") as stdout, os.fdopen(
                stderr_fd, "w", encoding="utf-8"
            ) as stderr:
                returncode, process_monitor = _run_monitored_command(
                    argv,
                    environment=environment,
                    stdout=stdout,
                    stderr=stderr,
                    action_id=action_id,
                )
        except BaseException:
            raise
        require(returncode == 0, f"action {action_id} failed; see {stderr_path}")
        artifact = _verify_action_output(
            action,
            output_root,
            contract_sha256=contract_sha,
            runtime_identity=runtime_identity,
            final_report_contract=plan["final_report_contract"],
        )
        receipt: dict[str, Any] = {
            "schema": ACTION_RECEIPT_SCHEMA,
            "status": "pass",
            "id": action_id,
            "kind": action["kind"],
            "depends_on": list(action["depends_on"]),
            "argv": argv,
            "env": {
                "CUDA_VISIBLE_DEVICES": EXPECTED_VISIBLE_DEVICE,
                "CUDA_DEVICE_ORDER": EXPECTED_CUDA_DEVICE_ORDER,
            },
            "executor_injected_runtime_environment": dict(
                sorted(CHILD_RUNTIME_ENVIRONMENT.items())
            ),
            "contract_sha256": contract_sha,
            "runtime_identity": dict(runtime_identity),
            "physical_gpu_lock_entry_identity": dict(physical_identity),
            "gpu_process_monitor": dict(process_monitor),
            "launch_integrity": {
                "code_bundle_schema": CODE_BUNDLE_SCHEMA,
                "code_bundle_sha256": code_bundle_identity["bundle_sha256"],
                "environment_schema": ENVIRONMENT_SCHEMA,
                "environment_fingerprint_sha256": environment_identity[
                    "fingerprint_sha256"
                ],
                "reverified_immediately_before_action": True,
            },
            "artifact": artifact,
            "stdout": {"path": str(stdout_path), "sha256": sha256_file(stdout_path)},
            "stderr": {"path": str(stderr_path), "sha256": sha256_file(stderr_path)},
            "bsd_test_pixels_opened": False,
        }
        checkpoint_key = _checkpoint_identity(action_id)
        if checkpoint_key is not None:
            receipt["arm"], receipt["seed"] = checkpoint_key
            receipt["checkpoint_sha256"] = artifact["sha256"]
            checkpoint_receipts[checkpoint_key] = receipt
        receipt_path = receipts_dir / f"{action_id}.json"
        _write_new_json(receipt_path, receipt)
        completed.add(action_id)
        action_receipts.append(
            {"id": action_id, "path": str(receipt_path), "sha256": sha256_file(receipt_path)}
        )

    final_actions = [
        action
        for action in plan["actions"]
        if action["id"]
        in {"cpu_reference_validation_report", "cpu_full_validation_report"}
    ]
    require(len(final_actions) == 1, "execution plan must contain exactly one final report action")
    final_report = _under_output(
        final_actions[0]["output"], output_root, label="final report output"
    )
    require(final_report.is_file(), f"executor finished without final report: {final_report}")
    final_payload = load_json_object(final_report)
    _finite_tree(final_payload)
    _validate_final_report_payload(
        final_payload,
        specification=plan["final_report_contract"],
        runtime_identity=runtime_identity,
    )
    execution_receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "mode": plan.get("mode", "full"),
        "contract": str(contract_path),
        "contract_sha256": contract_sha,
        "plan": str(plan_path),
        "plan_sha256": plan_sha,
        "runtime": dict(runtime_identity),
        "global_lock": dict(lock_identity),
        "physical_gpu_lock_entry_identity": dict(physical_identity),
        "code_bundle": dict(code_bundle_identity),
        "environment_fingerprint": dict(environment_identity),
        "actions_completed_in_order": action_receipts,
        "action_count": len(action_receipts),
        "final_report": {
            "path": str(final_report),
            "sha256": sha256_file(final_report),
            "schema": final_payload["schema"],
            "status": final_payload["status"],
            "arms": final_payload["arms"],
        },
        "checkpoint_receipts": {
            f"{arm}_seed{seed}": {
                "sha256": receipt["checkpoint_sha256"],
                "action": receipt["id"],
            }
            for (arm, seed), receipt in checkpoint_receipts.items()
        },
        "bsd_test_pixels_opened": False,
        "bsd_test_authorized": False,
        "slam_run_executed": False,
    }
    receipt_path = output_root / "execution_receipt.json"
    _write_new_json(receipt_path, execution_receipt)
    return receipt_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--reference-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # This is a pure environment-string check; it imports neither torch nor a
    # dataset loader and fails before taking any action on a malformed launch.
    require_exact_cuda_environment(formal=True)
    contract_path, contract, contract_sha = load_contract(
        args.contract, expected_sha256=args.expected_contract_sha256
    )
    validate_protocol(
        contract, allow_template=False, reference_only=args.reference_only
    )
    plan_path, plan = validate_execution_plan(
        args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        contract_path=contract_path,
        contract_sha256=contract_sha,
        reference_only=args.reference_only,
    )
    # No torch/CUDA API, nvidia-smi query, model, or pixel has been touched to
    # this point.  Every such operation below is enclosed by the one physical
    # GPU1 lock shared across studies.
    with exclusive_gpu1_lock(DEFAULT_GLOBAL_LOCK) as lock:
        require(
            sha256_file(contract_path) == contract_sha,
            "contract changed before global-lock entry",
        )
        require(
            sha256_file(plan_path) == args.expected_plan_sha256.lower(),
            "plan changed before global-lock entry",
        )
        import scripts.bsd_dpdd_contract as contract_module

        implementation_path_map = getattr(
            contract_module, "IMPLEMENTATION_PIN_PATHS", None
        )
        require(
            isinstance(implementation_path_map, Mapping),
            "central IMPLEMENTATION_PIN_PATHS mapping is missing",
        )
        implementation_path_map = {
            str(name): (
                Path(path).expanduser().resolve()
                if Path(path).expanduser().is_absolute()
                else (ROOT / Path(path).expanduser()).resolve()
            )
            for name, path in implementation_path_map.items()
        }
        code_bundle = rehash_frozen_code_bundle(
            implementation_pins=contract["implementation_pins"],
            code_bundle=contract["code_bundle"],
            expected_path_map=implementation_path_map,
            base_root=ROOT,
        )
        environment_identity = verify_frozen_environment(
            contract["environment_fingerprint"]
        )
        # This is the first hardware query.  It is inside the lease and must see
        # physical index 1 completely idle before this executor imports torch.
        physical = inspect_physical_gpu1(require_idle=True)
        os.environ.update(CHILD_RUNTIME_ENVIRONMENT)
        runtime = require_gpu1_a6000(
            "cuda:0",
            lock_identity=lock,
            physical_identity=physical,
            environment_fingerprint_sha256=environment_identity[
                "fingerprint_sha256"
            ],
            code_bundle_sha256=code_bundle["bundle_sha256"],
        )
        require_only_known_compute_pids(
            [os.getpid()], phase="after_executor_logical_cuda0_identity"
        )
        receipt = execute(
            contract_path=contract_path,
            contract=contract,
            contract_sha=contract_sha,
            plan_path=plan_path,
            plan=plan,
            plan_sha=sha256_file(plan_path),
            runtime_identity=runtime,
            lock_identity=lock,
            physical_identity=physical,
            environment_identity=environment_identity,
            code_bundle_identity=code_bundle,
            implementation_path_map=implementation_path_map,
        )
        require_only_known_compute_pids(
            [os.getpid()], phase="after_complete_execution_plan"
        )
    print(json.dumps({"execution_receipt": str(receipt), "sha256": sha256_file(receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
