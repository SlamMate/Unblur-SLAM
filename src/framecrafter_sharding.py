"""Deterministic, fail-closed sharding for offline FrameCrafter generation.

The generation workers intentionally write independent reports, manifests and
artifacts.  This module adds a small sidecar contract that binds those worker
outputs to one global plan.  The merger never copies or rewrites RGB-D/NPZ
artifacts: the merged manifest continues to reference the immutable files in
the shard directories.

This module is independent of the FrameCrafter model and can be tested on CPU.
The production preprocessor is expected to:

1. call :func:`build_shard_contract` once from the complete, pre-generation
   batch plan;
2. keep only batches assigned by :func:`assigned_batch_ids` on each worker;
3. call :func:`write_shard_envelope` after writing that worker's normal report
   and manifest; and
4. merge all envelopes with :func:`merge_shard_envelopes`.

An ordinary, unwrapped preprocess report is deliberately not accepted by the
merger.  It does not contain enough information to prove that two workers used
the same model/configuration or that the union is complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SHARD_CONTRACT_SCHEMA = "unblur_slam.framecrafter_shard_contract.v1"
SHARD_ENVELOPE_SCHEMA = "unblur_slam.framecrafter_shard_envelope.v1"
SHARD_RUNTIME_IDENTITY_SCHEMA = "unblur_slam.framecrafter_shard_runtime_identity.v1"
MERGED_REPORT_SCHEMA = "unblur_slam.framecrafter_preprocess_report.v1"
MERGED_MANIFEST_SCHEMA = "unblur_slam.framecrafter_manifest.v1"
ASSIGNMENT_ALGORITHM = "sha256_batch_id_and_target_gap_mod_v1"

_RUNTIME_IDENTITY_DIGEST_FIELDS = (
    "source_identity_sha256",
    "model_artifact_identity_sha256",
    "semantic_config_sha256",
    "implementation_identity_sha256",
)

_HEX = frozenset("0123456789abcdef")
_OUTPUT_PLAN_KEYS = {
    "poses_npz",
    "poses_npz_sha256",
    "candidate_rgb_path",
    "candidate_rgb_sha256",
    "rgb_path",
    "rgb_sha256",
    "depth_path",
    "depth_sha256",
}
_STANDALONE_DIGEST_KEYS = {
    "accepted_output_sha256",
    "source_input_sha256",
    "assignment_key_sha256",
    "assignment_table_sha256",
    "experiment_signature_sha256",
    "global_plan_sha256",
    "source_identity_sha256",
    "model_artifact_identity_sha256",
    "semantic_config_sha256",
    "implementation_identity_sha256",
    "runtime_identity_sha256",
    "canonical_preprocess_signature",
    "shard_merge_sha256",
}
_REPORT_OPERATIONAL_KEYS = {
    "schema",
    "preprocess_signature",
    "experiment_signature",
    "generation_id",
    "manifest",
    "generation_batches",
    "planned",
    "accepted",
    "sharp_accepted",
    "geometry_only",
    "quality_partition",
    "rejected",
    "source_frame_count",
    "planned_total_before_cap",
    "planned_target_count",
    "selected_target_count",
    "generation_batch_count",
    "backend_generate_call_count",
    "accepted_target_count",
    "sharp_accepted_target_count",
    "geometry_only_target_count",
    "geometry_rejected_target_count",
    "rejected_target_count",
    "accepted_output_sha256",
    "source_input_sha256",
    "shard",
    "shard_runtime_identity",
    "shard_merge",
}


def _jsonable(value: Any) -> Any:
    """Return a strict JSON-compatible value without importing NumPy."""

    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_shard_runtime_identity(
    *,
    source_identity: Any,
    model_artifact_identity: Any,
    semantic_config: Any,
    implementation_identity: Any,
) -> dict[str, str]:
    """Build the canonical identity every production worker must reproduce.

    The payloads are intentionally hashed independently.  This lets a failure
    name the scientific boundary that changed (source, model bytes, semantic
    parameters, or implementation) without embedding machine-local paths in
    the shard contract.  ``canonical_preprocess_signature`` binds all four
    digests and is independent of worker index, device, and output directory.
    """

    identity = {
        "schema": SHARD_RUNTIME_IDENTITY_SCHEMA,
        "source_identity_sha256": canonical_sha256(source_identity),
        "model_artifact_identity_sha256": canonical_sha256(
            model_artifact_identity
        ),
        "semantic_config_sha256": canonical_sha256(semantic_config),
        "implementation_identity_sha256": canonical_sha256(
            implementation_identity
        ),
    }
    identity["canonical_preprocess_signature"] = canonical_sha256(
        {field: identity[field] for field in _RUNTIME_IDENTITY_DIGEST_FIELDS}
    )
    return identity


def validate_shard_runtime_identity(value: Any) -> Mapping[str, str]:
    """Validate a worker identity and its canonical preprocessing signature."""

    if not isinstance(value, Mapping):
        raise ValueError("shard runtime identity must be an object")
    expected_fields = {
        "schema",
        *_RUNTIME_IDENTITY_DIGEST_FIELDS,
        "canonical_preprocess_signature",
    }
    if {str(field) for field in value} != expected_fields:
        missing = sorted(expected_fields - {str(field) for field in value})
        extra = sorted({str(field) for field in value} - expected_fields)
        raise ValueError(
            "shard runtime identity fields differ from its schema; "
            f"missing={missing}, extra={extra}"
        )
    if value.get("schema") != SHARD_RUNTIME_IDENTITY_SCHEMA:
        raise ValueError(
            f"unsupported shard runtime identity schema {value.get('schema')!r}"
        )
    for field in (*_RUNTIME_IDENTITY_DIGEST_FIELDS, "canonical_preprocess_signature"):
        _require_sha256(value.get(field), f"shard_runtime_identity.{field}")
    expected = canonical_sha256(
        {field: value[field] for field in _RUNTIME_IDENTITY_DIGEST_FIELDS}
    )
    if value.get("canonical_preprocess_signature") != expected:
        raise ValueError(
            "canonical_preprocess_signature does not bind source/model/config/implementation"
        )
    return value


def validate_runtime_identity_against_contract(
    contract: Mapping[str, Any], runtime_identity: Mapping[str, Any]
) -> None:
    """Fail closed unless actual worker inputs equal the immutable contract."""

    validate_shard_contract(contract)
    validate_shard_runtime_identity(runtime_identity)
    mismatches = [
        field
        for field in (
            *_RUNTIME_IDENTITY_DIGEST_FIELDS,
            "canonical_preprocess_signature",
        )
        if runtime_identity.get(field) != contract.get(field)
    ]
    if mismatches:
        raise ValueError(
            "worker runtime identity differs from shard contract: "
            + ", ".join(mismatches)
        )


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _read_json_object(path: Path | str, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object: {resolved}")
    return resolved, payload


def _resolve_artifact(path_value: Any, base_dir: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{label} must be a non-empty file path")
    artifact = Path(path_value).expanduser()
    if not artifact.is_absolute():
        artifact = base_dir / artifact
    artifact = artifact.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"{label} does not exist: {artifact}")
    return artifact


def _validate_path_hash_pairs(
    value: Any, base_dir: Path, location: str = "root"
) -> None:
    """Validate every ``*_path``/``*_sha256`` artifact pair recursively.

    A path without an adjacent hash is allowed because some fields are merely
    provenance paths.  A hash without its paired path is rejected.  This keeps
    the generic validator forward compatible with new conditioning records
    while still binding every artifact that claims a content hash.
    """

    if isinstance(value, Mapping):
        keys = {str(key) for key in value}
        for key in keys:
            if not key.endswith("_sha256"):
                continue
            if key in _STANDALONE_DIGEST_KEYS:
                _require_sha256(value[key], f"{location}.{key}")
                continue
            path_key = f"{key[:-7]}_path"
            # Existing reports call the file ``poses_npz`` rather than
            # ``poses_npz_path``.
            fallback_key = key[:-7]
            if path_key in value:
                artifact_key = path_key
            elif fallback_key in value:
                artifact_key = fallback_key
            else:
                raise ValueError(f"{location}.{key} has no paired artifact path")
            expected = _require_sha256(value[key], f"{location}.{key}")
            artifact = _resolve_artifact(
                value[artifact_key], base_dir, f"{location}.{artifact_key}"
            )
            if file_sha256(artifact) != expected:
                raise ValueError(f"artifact hash mismatch: {location}.{artifact_key}")
        for key, item in value.items():
            _validate_path_hash_pairs(item, base_dir, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_path_hash_pairs(item, base_dir, f"{location}[{index}]")


def _batch_id(batch: Mapping[str, Any], position: int) -> str:
    batch_id = str(batch.get("batch_id", "")).strip()
    if not batch_id:
        raise ValueError(f"generation_batches[{position}] has no batch_id")
    return batch_id


def _target_id(record: Mapping[str, Any], position: int) -> str:
    target_id = str(record.get("target_id", "")).strip()
    if not target_id:
        raise ValueError(f"planned[{position}] has no target_id")
    return target_id


def _target_gap(record: Mapping[str, Any]) -> dict[str, int]:
    try:
        left_index = int(record["left_index"])
        right_index = int(record["right_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "every planned target requires integer left/right_index"
        ) from error
    if left_index == right_index:
        raise ValueError("planned target endpoints must be distinct")
    gap = {
        "left_index": left_index,
        "right_index": right_index,
    }
    if "left_position" in record or "right_position" in record:
        try:
            gap.update(
                left_position=int(record["left_position"]),
                right_position=int(record["right_position"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "planned target must provide both integer endpoint positions"
            ) from error
    return gap


def _stable_plan_value(value: Any) -> Any:
    """Remove worker-local output artifacts from the global plan identity."""

    if isinstance(value, Mapping):
        paired_path_keys = {
            f"{str(key)[:-7]}_path" for key in value if str(key).endswith("_sha256")
        }
        return {
            str(key): _stable_plan_value(item)
            for key, item in value.items()
            if str(key) not in _OUTPUT_PLAN_KEYS and str(key) not in paired_path_keys
        }
    if isinstance(value, list):
        return [_stable_plan_value(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_plan_value(item) for item in value]
    return _jsonable(value)


def _content_identity_value(value: Any) -> Any:
    """Make provenance identities portable across machines with different roots."""

    if isinstance(value, Mapping):
        paired_path_keys = {
            f"{str(key)[:-7]}_path" for key in value if str(key).endswith("_sha256")
        }
        return {
            str(key): _content_identity_value(item)
            for key, item in value.items()
            if str(key) not in paired_path_keys
        }
    if isinstance(value, list):
        return [_content_identity_value(item) for item in value]
    if isinstance(value, tuple):
        return [_content_identity_value(item) for item in value]
    return _jsonable(value)


def build_assignment_table(
    generation_batches: Sequence[Mapping[str, Any]],
    planned: Sequence[Mapping[str, Any]],
    shard_count: int,
) -> list[dict[str, Any]]:
    """Assign complete generation batches with a stable gap-aware hash.

    Targets in one M-to-N diffusion call are never split between workers.
    Assignment depends only on the batch ID and endpoint gap(s), not list order,
    process hash seeds, output paths, or machine names.
    """

    shard_count = int(shard_count)
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    batches = [dict(batch) for batch in generation_batches]
    records = [dict(record) for record in planned]
    planned_by_batch: dict[str, list[Mapping[str, Any]]] = {}
    planned_ids: set[str] = set()
    for position, record in enumerate(records):
        target_id = _target_id(record, position)
        if target_id in planned_ids:
            raise ValueError(f"duplicate planned target_id {target_id!r}")
        planned_ids.add(target_id)
        batch_id = str(record.get("batch_id", "")).strip()
        if not batch_id:
            raise ValueError(f"planned target {target_id!r} has no batch_id")
        planned_by_batch.setdefault(batch_id, []).append(record)

    table: list[dict[str, Any]] = []
    seen_batches: set[str] = set()
    flattened_targets: set[str] = set()
    for position, batch in enumerate(batches):
        batch_id = _batch_id(batch, position)
        if batch_id in seen_batches:
            raise ValueError(f"duplicate generation batch_id {batch_id!r}")
        seen_batches.add(batch_id)
        target_ids_value = batch.get("target_ids")
        if not isinstance(target_ids_value, list) or not target_ids_value:
            raise ValueError(f"batch {batch_id!r} requires a non-empty target_ids list")
        target_ids = [str(value) for value in target_ids_value]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError(f"batch {batch_id!r} contains duplicate target IDs")
        batch_records = planned_by_batch.get(batch_id, [])
        records_by_id = {
            str(record.get("target_id")): record for record in batch_records
        }
        if set(records_by_id) != set(target_ids):
            raise ValueError(
                f"batch {batch_id!r} targets disagree with planned records"
            )
        flattened_targets.update(target_ids)
        gaps = [_target_gap(records_by_id[target_id]) for target_id in target_ids]
        assignment_identity = {
            "algorithm": ASSIGNMENT_ALGORITHM,
            "batch_id": batch_id,
            "target_gaps": gaps,
        }
        assignment_digest = canonical_sha256(assignment_identity)
        table.append(
            {
                "batch_id": batch_id,
                "target_ids": target_ids,
                "target_gaps": gaps,
                "assignment_key_sha256": assignment_digest,
                "shard_index": int(assignment_digest, 16) % shard_count,
            }
        )
    if seen_batches != set(planned_by_batch):
        missing = sorted(set(planned_by_batch) - seen_batches)
        raise ValueError(f"planned records name unknown generation batches: {missing}")
    if flattened_targets != planned_ids:
        raise ValueError(
            "generation batches do not cover every planned target exactly once"
        )
    return table


def build_shard_contract(
    generation_batches: Sequence[Mapping[str, Any]],
    planned: Sequence[Mapping[str, Any]],
    *,
    shard_count: int,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the immutable global contract shared by all workers."""

    validate_shard_runtime_identity(runtime_identity)
    assignments = build_assignment_table(generation_batches, planned, shard_count)
    stable_plan = {
        "generation_batches": _stable_plan_value(list(generation_batches)),
        "planned": _stable_plan_value(list(planned)),
    }
    identity_digests = {
        field: str(runtime_identity[field])
        for field in (
            *_RUNTIME_IDENTITY_DIGEST_FIELDS,
            "canonical_preprocess_signature",
        )
    }
    identity_digests.update(
        global_plan_sha256=canonical_sha256(stable_plan),
        assignment_table_sha256=canonical_sha256(assignments),
    )
    experiment_signature = canonical_sha256(identity_digests)
    return {
        "schema": SHARD_CONTRACT_SCHEMA,
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "experiment_signature": experiment_signature,
        "shard_count": int(shard_count),
        **identity_digests,
        "assignments": assignments,
    }


def validate_shard_contract(contract: Any) -> Mapping[str, Any]:
    if not isinstance(contract, Mapping):
        raise ValueError("shard contract root must be an object")
    if contract.get("schema") != SHARD_CONTRACT_SCHEMA:
        raise ValueError(
            f"unsupported shard contract schema {contract.get('schema')!r}"
        )
    if contract.get("assignment_algorithm") != ASSIGNMENT_ALGORITHM:
        raise ValueError("unsupported FrameCrafter shard assignment algorithm")
    signature_fields = (
        "experiment_signature",
        *_RUNTIME_IDENTITY_DIGEST_FIELDS,
        "canonical_preprocess_signature",
        "global_plan_sha256",
        "assignment_table_sha256",
    )
    for field in signature_fields:
        _require_sha256(contract.get(field), field)
    validate_shard_runtime_identity(
        {
            "schema": SHARD_RUNTIME_IDENTITY_SCHEMA,
            **{
                field: contract[field]
                for field in (
                    *_RUNTIME_IDENTITY_DIGEST_FIELDS,
                    "canonical_preprocess_signature",
                )
            },
        }
    )
    expected_experiment_signature = canonical_sha256(
        {
            field: contract[field]
            for field in (
                *_RUNTIME_IDENTITY_DIGEST_FIELDS,
                "canonical_preprocess_signature",
                "global_plan_sha256",
                "assignment_table_sha256",
            )
        }
    )
    if contract.get("experiment_signature") != expected_experiment_signature:
        raise ValueError("experiment_signature does not bind source/model/config/plan")
    try:
        shard_count = int(contract["shard_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("shard contract has invalid shard_count") from error
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    assignments = contract.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("shard contract assignments must be a list")
    batch_ids: set[str] = set()
    target_ids: set[str] = set()
    for position, assignment in enumerate(assignments):
        if not isinstance(assignment, Mapping):
            raise ValueError(f"assignments[{position}] must be an object")
        batch_id = str(assignment.get("batch_id", "")).strip()
        targets = assignment.get("target_ids")
        gaps = assignment.get("target_gaps")
        if (
            not batch_id
            or batch_id in batch_ids
            or not isinstance(targets, list)
            or not targets
            or not isinstance(gaps, list)
            or len(gaps) != len(targets)
        ):
            raise ValueError(f"invalid assignment at position {position}")
        targets = [str(value) for value in targets]
        if len(set(targets)) != len(targets) or target_ids.intersection(targets):
            raise ValueError("assignment targets must be globally unique")
        assignment_identity = {
            "algorithm": ASSIGNMENT_ALGORITHM,
            "batch_id": batch_id,
            "target_gaps": gaps,
        }
        digest = canonical_sha256(assignment_identity)
        if assignment.get("assignment_key_sha256") != digest:
            raise ValueError(f"assignment hash mismatch for batch {batch_id}")
        if int(assignment.get("shard_index", -1)) != int(digest, 16) % shard_count:
            raise ValueError(f"assignment shard mismatch for batch {batch_id}")
        batch_ids.add(batch_id)
        target_ids.update(targets)
    if canonical_sha256(assignments) != contract["assignment_table_sha256"]:
        raise ValueError("assignment table does not match its contract digest")
    return contract


def assigned_batch_ids(
    contract: Mapping[str, Any], shard_index: int
) -> tuple[str, ...]:
    validate_shard_contract(contract)
    shard_index = int(shard_index)
    shard_count = int(contract["shard_count"])
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count})")
    return tuple(
        str(assignment["batch_id"])
        for assignment in contract["assignments"]
        if int(assignment["shard_index"]) == shard_index
    )


def validate_global_plan_against_contract(
    contract: Mapping[str, Any],
    generation_batches: Sequence[Mapping[str, Any]],
    planned: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed unless a worker rebuilt the contract's exact global plan.

    Workers must run planning over the complete source stream before filtering
    their assigned batches.  Both the complete stable plan and the deterministic
    assignment table are checked: matching only a set of batch IDs would not
    detect changed target poses, conditioning inputs, or target order.
    """

    validate_shard_contract(contract)
    assignments = build_assignment_table(
        generation_batches,
        planned,
        int(contract["shard_count"]),
    )
    if canonical_sha256(assignments) != contract["assignment_table_sha256"]:
        raise ValueError(
            "worker global plan assignment table differs from shard contract"
        )
    if assignments != list(contract["assignments"]):
        raise ValueError("worker global plan assignments differ from shard contract")
    stable_plan = {
        "generation_batches": _stable_plan_value(list(generation_batches)),
        "planned": _stable_plan_value(list(planned)),
    }
    if canonical_sha256(stable_plan) != contract["global_plan_sha256"]:
        raise ValueError("worker global plan differs from shard contract")


def _report_partitions(report: Mapping[str, Any]) -> list[tuple[str, list[Any]]]:
    """Return injection and disjoint scientific gate partitions.

    Production double-gate reports keep ``accepted``/``rejected`` as the
    injection-policy view.  In ``acceptance_mode=sharp``, the top-level
    ``rejected`` list therefore contains both geometry-only and truly rejected
    targets.  Scientific outcomes live under ``quality_partition`` and must be
    used instead, otherwise geometry-only targets appear twice.
    """

    partitions: list[tuple[str, list[Any]]] = []
    accepted = report.get("accepted")
    if accepted is not None:
        if not isinstance(accepted, list):
            raise ValueError("report.accepted must be a list")
        partitions.append(("accepted", accepted))

    quality = report.get("quality_partition")
    if quality is not None:
        if not isinstance(quality, Mapping):
            raise ValueError("report.quality_partition must be an object")
        for name in ("sharp_accepted", "geometry_only", "rejected"):
            if name not in quality:
                raise ValueError(f"report.quality_partition.{name} is required")
            value = quality[name]
            if not isinstance(value, list):
                raise ValueError(f"report.quality_partition.{name} must be a list")
            partitions.append((name, value))
        return partitions

    for name in ("accepted", "sharp_accepted", "geometry_only", "rejected"):
        if name == "accepted":
            continue
        value = report.get(name)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"report.{name} must be a list")
        partitions.append((name, value))
    if not partitions:
        raise ValueError("shard report has no recognized per-target gate partitions")
    return partitions


def _outcome_records(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the disjoint scientific outcome classes.

    In a double-gate report, ``accepted`` is an injection-policy view and may
    overlap ``sharp_accepted``/``geometry_only``.  It therefore does not take
    part in the outcome partition when either explicit class exists.
    """

    partition_values = dict(_report_partitions(report))
    if "sharp_accepted" in partition_values or "geometry_only" in partition_values:
        names = ("sharp_accepted", "geometry_only", "rejected")
    else:
        names = ("accepted", "rejected")
    records: dict[str, Mapping[str, Any]] = {}
    for partition in names:
        values = partition_values.get(partition, [])
        for position, record in enumerate(values):
            if not isinstance(record, Mapping):
                raise ValueError(f"report.{partition}[{position}] must be an object")
            target_id = str(record.get("target_id", "")).strip()
            if not target_id:
                raise ValueError(f"report.{partition}[{position}] has no target_id")
            previous = records.get(target_id)
            if previous is not None:
                raise ValueError(
                    f"target {target_id!r} appears in multiple scientific gate classes"
                )
            records[target_id] = record
    return records


def _expected_manifest_target_ids(report: Mapping[str, Any]) -> set[str]:
    partitions = dict(_report_partitions(report))
    records = [
        record
        for values in partitions.values()
        for record in values
        if isinstance(record, Mapping)
    ]
    explicit_flags = [record for record in records if "injected" in record]
    if explicit_flags:
        return {
            str(record.get("target_id"))
            for record in explicit_flags
            if record.get("injected") is True
        }
    if "accepted" in partitions:
        # v1 and recommended v2 both use accepted as the canonical list of
        # observations actually injected into SLAM.
        return {
            str(record.get("target_id"))
            for record in partitions["accepted"]
            if isinstance(record, Mapping)
        }
    acceptance_mode = report.get("acceptance_mode")
    sharp_ids = {
        str(record.get("target_id"))
        for record in partitions.get("sharp_accepted", [])
        if isinstance(record, Mapping)
    }
    geometry_ids = {
        str(record.get("target_id"))
        for record in partitions.get("geometry_only", [])
        if isinstance(record, Mapping)
    }
    if acceptance_mode == "sharp":
        return sharp_ids
    if acceptance_mode == "geometry":
        return sharp_ids | geometry_ids
    raise ValueError(
        "double-gate report needs accepted, per-record injected flags, or "
        "acceptance_mode to identify manifest observations"
    )


def _validate_local_report(
    report: Mapping[str, Any], assigned_ids: Sequence[str]
) -> tuple[set[str], set[str]]:
    batches = report.get("generation_batches")
    planned = report.get("planned")
    if not isinstance(batches, list) or not isinstance(planned, list):
        raise ValueError("shard report requires generation_batches and planned lists")
    batch_ids: list[str] = []
    target_ids: list[str] = []
    for position, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise ValueError(f"generation_batches[{position}] must be an object")
        batch_ids.append(_batch_id(batch, position))
        values = batch.get("target_ids")
        if not isinstance(values, list) or not values:
            raise ValueError(f"batch {batch_ids[-1]!r} has invalid target_ids")
        target_ids.extend(str(value) for value in values)
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("shard report contains duplicate generation batch IDs")
    if int(report.get("generation_batch_count", -1)) != len(batches):
        raise ValueError("shard generation_batch_count is inconsistent")
    if int(report.get("backend_generate_call_count", -1)) != len(batches):
        raise ValueError("shard report must record one generation call per batch")
    if set(batch_ids) != set(assigned_ids):
        raise ValueError(
            "shard report generation batches do not equal deterministic assignment"
        )
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("shard report contains duplicate planned target IDs")
    planned_ids = []
    for position, record in enumerate(planned):
        if not isinstance(record, Mapping):
            raise ValueError(f"planned[{position}] must be an object")
        planned_ids.append(_target_id(record, position))
        if str(record.get("batch_id", "")) not in set(batch_ids):
            raise ValueError("planned target names a batch outside this shard")
    if set(planned_ids) != set(target_ids) or len(planned_ids) != len(target_ids):
        raise ValueError("shard planned records disagree with batch target IDs")
    if int(report.get("planned_target_count", -1)) != len(planned) or int(
        report.get("selected_target_count", -1)
    ) != len(planned):
        raise ValueError("shard planned target counts are inconsistent")
    outcomes = _outcome_records(report)
    if set(outcomes) != set(target_ids):
        raise ValueError("shard gate outcomes do not partition every planned target")
    partitions = dict(_report_partitions(report))
    expected_counts = {
        "accepted_target_count": len(partitions.get("accepted", [])),
        "sharp_accepted_target_count": len(partitions.get("sharp_accepted", [])),
        "geometry_only_target_count": len(partitions.get("geometry_only", [])),
    }
    if "quality_partition" in report:
        top_level_rejected = report.get("rejected")
        if not isinstance(top_level_rejected, list):
            raise ValueError("double-gate report.rejected must be a list")
        expected_counts.update(
            rejected_target_count=len(top_level_rejected),
            geometry_rejected_target_count=len(partitions.get("rejected", [])),
        )
    else:
        expected_counts["rejected_target_count"] = len(partitions.get("rejected", []))
    for field, count in expected_counts.items():
        if field in report and int(report[field]) != count:
            raise ValueError(f"shard {field} is inconsistent")
    return set(batch_ids), set(target_ids)


def _manifest_original_identity(manifest: Mapping[str, Any]) -> str:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("shard manifest frames must be a list")
    originals = [
        frame
        for frame in frames
        if isinstance(frame, Mapping) and frame.get("kind") == "original"
    ]
    canonical = sorted(
        (_content_identity_value(frame) for frame in originals),
        key=lambda frame: (
            int(frame.get("source_index", -1)),
            float(frame.get("timestamp", 0.0)),
        ),
    )
    if len(canonical) != int(manifest.get("source_frame_count", -1)):
        raise ValueError("shard manifest original count is inconsistent")
    return canonical_sha256(canonical)


def _manifest_synthetics(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("shard manifest frames must be a list")
    synthetics: dict[str, Mapping[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping) or frame.get("kind") != "synthetic":
            continue
        target_id = str(frame.get("target_id", "")).strip()
        if not target_id or target_id in synthetics:
            raise ValueError("shard manifest synthetic target IDs must be unique")
        synthetics[target_id] = frame
    if len(synthetics) != int(manifest.get("generated_frame_count", -1)):
        raise ValueError("shard manifest generated count is inconsistent")
    return synthetics


def _validate_report_shard_identity(
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    shard_index: int,
) -> None:
    """Bind the worker's compact shard record to its full runtime identity."""

    shard_record = report.get("shard")
    if not isinstance(shard_record, Mapping):
        raise ValueError("shard report identity record is missing")
    expected_runtime_digest = canonical_sha256(runtime_identity)
    try:
        recorded_index = int(shard_record.get("shard_index", -1))
        recorded_count = int(shard_record.get("shard_count", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("shard report identity record has invalid indices") from error
    if any(
        (
            shard_record.get("schema")
            != "unblur_slam.framecrafter_worker_shard.v1",
            shard_record.get("experiment_signature")
            != contract.get("experiment_signature"),
            shard_record.get("canonical_preprocess_signature")
            != contract.get("canonical_preprocess_signature"),
            shard_record.get("runtime_identity_sha256")
            != expected_runtime_digest,
            shard_record.get("assignment_table_sha256")
            != contract.get("assignment_table_sha256"),
            shard_record.get("global_plan_sha256")
            != contract.get("global_plan_sha256"),
            recorded_index != int(shard_index),
            recorded_count != int(contract.get("shard_count", -1)),
            shard_record.get("assigned_batch_ids")
            != list(assigned_batch_ids(contract, shard_index)),
        )
    ):
        raise ValueError("shard report identity record disagrees with contract")


def write_shard_envelope(
    contract: Mapping[str, Any],
    *,
    shard_index: int,
    report_path: Path | str,
    manifest_path: Path | str,
    output_path: Path | str,
) -> Path:
    """Validate one worker output and write its content-addressed sidecar."""

    validate_shard_contract(contract)
    assigned = assigned_batch_ids(contract, shard_index)
    report_file, report = _read_json_object(report_path, "shard report")
    manifest_file, manifest = _read_json_object(manifest_path, "shard manifest")
    if report.get("manifest") is not None:
        reported_manifest = Path(str(report["manifest"])).expanduser()
        if not reported_manifest.is_absolute():
            reported_manifest = report_file.parent / reported_manifest
        if reported_manifest.resolve() != manifest_file:
            raise ValueError("shard report points to a different manifest")
    if (
        bool(report.get("backend_test_only", True))
        or report.get("backend") != "python_api"
    ):
        raise ValueError(
            "only real python_api FrameCrafter shard outputs can be merged"
        )
    if (
        manifest.get("uses_ground_truth_pose") is not False
        or report.get("uses_ground_truth_pose") is not False
    ):
        raise ValueError("ground-truth pose inputs are forbidden in shard outputs")
    if manifest.get("pose_source") != report.get("pose_source"):
        raise ValueError("shard report/manifest pose_source mismatch")
    if int(manifest.get("source_frame_count", -1)) != int(
        report.get("source_frame_count", -2)
    ):
        raise ValueError("shard report/manifest source_frame_count mismatch")
    runtime_identity = report.get("shard_runtime_identity")
    validate_shard_runtime_identity(runtime_identity)
    validate_runtime_identity_against_contract(contract, runtime_identity)
    _validate_report_shard_identity(
        report, contract, runtime_identity, int(shard_index)
    )
    batch_ids, target_ids = _validate_local_report(report, assigned)
    synthetics = _manifest_synthetics(manifest)
    expected_synthetics = _expected_manifest_target_ids(report)
    if set(synthetics) != expected_synthetics:
        raise ValueError(
            "shard manifest synthetics disagree with accepted/injected outcomes"
        )
    if not set(synthetics).issubset(target_ids):
        raise ValueError("shard manifest includes a target outside this shard")
    _validate_path_hash_pairs(report, report_file.parent, "report")
    _validate_path_hash_pairs(manifest, manifest_file.parent, "manifest")
    envelope = {
        "schema": SHARD_ENVELOPE_SCHEMA,
        "contract": _jsonable(contract),
        "shard_index": int(shard_index),
        "assigned_batch_ids": list(assigned),
        "assigned_target_ids": sorted(target_ids),
        "shard_runtime_identity": _jsonable(runtime_identity),
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "report_path": str(report_file),
        "report_sha256": file_sha256(report_file),
        "manifest_path": str(manifest_file),
        "manifest_sha256": file_sha256(manifest_file),
        "original_frame_identity_sha256": _manifest_original_identity(manifest),
    }
    return _write_immutable_json(output_path, envelope)


def _canonical_frame_contract(
    entries: Sequence[Mapping[str, Any]], kind: str
) -> list[dict[str, Any]]:
    """Match framecrafter_pipeline's report-binding digest exactly."""

    records: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        if entry.get("kind") != kind:
            continue
        common = {
            "kind": kind,
            "source_index": entry.get("source_index"),
            "timestamp": entry.get("timestamp"),
            "rgb_path": entry.get("rgb_path"),
            "depth_path": entry.get("depth_path"),
            "rgb_sha256": entry.get("rgb_sha256"),
            "depth_sha256": entry.get("depth_sha256"),
            "c2w": entry.get("c2w"),
            "confidence": entry.get("confidence"),
            "eval": entry.get("eval"),
            "fixed_pose": entry.get("fixed_pose"),
        }
        if kind == "synthetic":
            common.update(
                target_id=entry.get("target_id"),
                left_index=entry.get("left_index"),
                right_index=entry.get("right_index"),
                alpha=entry.get("alpha"),
                reasons=entry.get("reasons"),
                source_ids=entry.get("source_ids"),
                gate_metrics=entry.get("gate_metrics"),
                batch_id=entry.get("batch_id"),
                batch_target_ids=entry.get("batch_target_ids"),
                batch_target_position=entry.get("batch_target_position"),
            )
            if "acceptance_class" in entry:
                common["acceptance_class"] = entry.get("acceptance_class")
        common["manifest_position"] = position
        records.append(common)
    return records


def _frames_digest(frames: Sequence[Mapping[str, Any]], kind: str) -> str:
    return canonical_sha256(_canonical_frame_contract(frames, kind))


def _sort_records_by_plan(
    records: Iterable[Mapping[str, Any]], target_order: Mapping[str, int]
) -> list[dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda record: target_order[str(record.get("target_id"))],
    )


def _shared_report_provenance(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Retain equal semantic top-level fields without merging worker counters."""

    if not reports:
        return {}
    shared: dict[str, Any] = {}
    common_keys = set(reports[0]).intersection(*(set(report) for report in reports[1:]))
    for key in sorted(common_keys - _REPORT_OPERATIONAL_KEYS):
        first = reports[0][key]
        try:
            digest = canonical_sha256(first)
            if all(canonical_sha256(report[key]) == digest for report in reports[1:]):
                shared[key] = _jsonable(first)
        except (TypeError, ValueError):
            # Non-JSON runtime handles are never valid scientific provenance.
            continue
    return shared


def _write_immutable_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise FileExistsError(
                f"refusing to overwrite immutable file: {destination}"
            )
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # os.link provides no-replace publication even if another merger wins
        # the race between the existence check and this point.
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if destination.read_bytes() != encoded:
                raise FileExistsError(
                    f"refusing to overwrite immutable file: {destination}"
                )
        return destination
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def write_shard_contract(contract: Mapping[str, Any], path: Path | str) -> Path:
    """Validate and atomically publish an immutable global shard contract."""

    validate_shard_contract(contract)
    return _write_immutable_json(path, contract)


def _load_envelope(
    path: Path | str,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    envelope_path, envelope = _read_json_object(path, "shard envelope")
    if envelope.get("schema") != SHARD_ENVELOPE_SCHEMA:
        raise ValueError(
            "expected a FrameCrafter shard envelope; ordinary preprocess reports "
            "cannot prove cross-worker completeness"
        )
    contract = envelope.get("contract")
    validate_shard_contract(contract)
    envelope_runtime_identity = envelope.get("shard_runtime_identity")
    validate_shard_runtime_identity(envelope_runtime_identity)
    validate_runtime_identity_against_contract(contract, envelope_runtime_identity)
    if envelope.get("runtime_identity_sha256") != canonical_sha256(
        envelope_runtime_identity
    ):
        raise ValueError("shard envelope runtime identity hash mismatch")
    shard_index = int(envelope.get("shard_index", -1))
    expected_ids = assigned_batch_ids(contract, shard_index)
    if envelope.get("assigned_batch_ids") != list(expected_ids):
        raise ValueError("shard envelope assigned_batch_ids were modified")
    report_path = _resolve_artifact(
        envelope.get("report_path"), envelope_path.parent, "report_path"
    )
    manifest_path = _resolve_artifact(
        envelope.get("manifest_path"), envelope_path.parent, "manifest_path"
    )
    if file_sha256(report_path) != _require_sha256(
        envelope.get("report_sha256"), "report_sha256"
    ):
        raise ValueError(f"shard report hash mismatch: {report_path}")
    if file_sha256(manifest_path) != _require_sha256(
        envelope.get("manifest_sha256"), "manifest_sha256"
    ):
        raise ValueError(f"shard manifest hash mismatch: {manifest_path}")
    _, report = _read_json_object(report_path, "shard report")
    _, manifest = _read_json_object(manifest_path, "shard manifest")
    report_runtime_identity = report.get("shard_runtime_identity")
    validate_shard_runtime_identity(report_runtime_identity)
    validate_runtime_identity_against_contract(contract, report_runtime_identity)
    if canonical_sha256(report_runtime_identity) != envelope.get(
        "runtime_identity_sha256"
    ):
        raise ValueError("shard report/envelope runtime identity mismatch")
    _validate_report_shard_identity(
        report, contract, report_runtime_identity, shard_index
    )
    _validate_local_report(report, expected_ids)
    if _manifest_original_identity(manifest) != envelope.get(
        "original_frame_identity_sha256"
    ):
        raise ValueError("shard original-frame identity mismatch")
    _validate_path_hash_pairs(report, report_path.parent, "report")
    _validate_path_hash_pairs(manifest, manifest_path.parent, "manifest")
    return envelope_path, envelope, report, manifest


def merge_shard_envelopes(
    envelope_paths: Sequence[Path | str], output_dir: Path | str
) -> tuple[Path, Path]:
    """Merge a complete shard set without copying any generated artifacts.

    Returns ``(merged_report_path, merged_manifest_path)``.  Output names are
    deterministic and immutable; repeating an identical merge is idempotent.
    """

    if not envelope_paths:
        raise ValueError("at least one shard envelope is required")
    loaded = [_load_envelope(path) for path in envelope_paths]
    contract = loaded[0][1]["contract"]
    contract_digest = canonical_sha256(contract)
    shard_count = int(contract["shard_count"])
    by_index: dict[
        int, tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = {}
    original_identity: str | None = None
    for item in loaded:
        envelope_path, envelope, report, manifest = item
        if canonical_sha256(envelope["contract"]) != contract_digest:
            raise ValueError(
                "shard contracts differ (source/model/config/plan mismatch)"
            )
        shard_index = int(envelope["shard_index"])
        if shard_index in by_index:
            raise ValueError(f"duplicate shard_index {shard_index}")
        by_index[shard_index] = item
        identity = str(envelope["original_frame_identity_sha256"])
        if original_identity is None:
            original_identity = identity
        elif identity != original_identity:
            raise ValueError("shard manifests contain different original RGB-D streams")
        if report.get("pose_source") != manifest.get("pose_source"):
            raise ValueError(f"pose_source mismatch in {envelope_path}")
    expected_indices = set(range(shard_count))
    if set(by_index) != expected_indices:
        missing = sorted(expected_indices - set(by_index))
        extra = sorted(set(by_index) - expected_indices)
        raise ValueError(f"incomplete shard set; missing={missing}, extra={extra}")

    expected_batch_ids = [str(item["batch_id"]) for item in contract["assignments"]]
    observed_batch_ids = [
        str(batch["batch_id"])
        for _, _, report, _ in by_index.values()
        for batch in report["generation_batches"]
    ]
    counts = Counter(observed_batch_ids)
    duplicates = sorted(batch_id for batch_id, count in counts.items() if count != 1)
    if duplicates or set(observed_batch_ids) != set(expected_batch_ids):
        missing = sorted(set(expected_batch_ids) - set(observed_batch_ids))
        extra = sorted(set(observed_batch_ids) - set(expected_batch_ids))
        raise ValueError(
            "shard batch union is not an exact partition; "
            f"duplicates={duplicates}, missing={missing}, extra={extra}"
        )

    report_by_batch: dict[str, Mapping[str, Any]] = {}
    planned_by_target: dict[str, Mapping[str, Any]] = {}
    partition_records: dict[str, dict[str, Mapping[str, Any]]] = {
        "accepted": {},
        "sharp_accepted": {},
        "geometry_only": {},
        "rejected": {},
    }
    injection_rejected_records: dict[str, Mapping[str, Any]] = {}
    manifest_synthetics: dict[str, Mapping[str, Any]] = {}
    first_report = by_index[0][2]
    first_manifest = by_index[0][3]
    runtime_identity = dict(first_report["shard_runtime_identity"])
    validate_runtime_identity_against_contract(contract, runtime_identity)
    runtime_identity_digest = canonical_sha256(runtime_identity)
    uses_quality_partition = "quality_partition" in first_report
    first_originals = [
        dict(frame)
        for frame in first_manifest["frames"]
        if isinstance(frame, Mapping) and frame.get("kind") == "original"
    ]
    for shard_index in range(shard_count):
        _, _, report, manifest = by_index[shard_index]
        if canonical_sha256(report.get("shard_runtime_identity")) != (
            runtime_identity_digest
        ):
            raise ValueError("shard workers have different runtime identities")
        if any(
            (
                report.get("backend") != first_report.get("backend"),
                report.get("backend_test_only")
                != first_report.get("backend_test_only"),
                report.get("pose_source") != first_report.get("pose_source"),
                manifest.get("pose_source") != first_manifest.get("pose_source"),
                ("quality_partition" in report) != uses_quality_partition,
            )
        ):
            raise ValueError("shard backend/pose/gate provenance differs")
        for batch in report["generation_batches"]:
            report_by_batch[str(batch["batch_id"])] = batch
        for record in report["planned"]:
            target_id = str(record["target_id"])
            if target_id in planned_by_target:
                raise ValueError(
                    f"duplicate planned target {target_id!r} across shards"
                )
            planned_by_target[target_id] = record
        for partition, records in _report_partitions(report):
            destination = partition_records[partition]
            for record in records:
                target_id = str(record["target_id"])
                previous = destination.get(target_id)
                if previous is not None and dict(previous) != dict(record):
                    raise ValueError(f"conflicting {partition} target {target_id!r}")
                destination[target_id] = record
        if uses_quality_partition:
            top_level_rejected = report.get("rejected")
            if not isinstance(top_level_rejected, list):
                raise ValueError("double-gate shard report.rejected must be a list")
            for record in top_level_rejected:
                if not isinstance(record, Mapping):
                    raise ValueError(
                        "double-gate shard rejected record must be an object"
                    )
                target_id = str(record.get("target_id", "")).strip()
                if not target_id or target_id in injection_rejected_records:
                    raise ValueError(
                        "double-gate top-level rejected target IDs must be unique"
                    )
                injection_rejected_records[target_id] = record
        for target_id, frame in _manifest_synthetics(manifest).items():
            if target_id in manifest_synthetics:
                raise ValueError(
                    f"duplicate synthetic target {target_id!r} across shards"
                )
            manifest_synthetics[target_id] = frame

    target_order_values = [
        str(target_id)
        for assignment in contract["assignments"]
        for target_id in assignment["target_ids"]
    ]
    if set(target_order_values) != set(planned_by_target):
        raise ValueError("merged planned targets disagree with global shard contract")
    target_order = {
        target_id: position for position, target_id in enumerate(target_order_values)
    }
    generation_batches = [report_by_batch[batch_id] for batch_id in expected_batch_ids]
    planned = [planned_by_target[target_id] for target_id in target_order_values]

    frames = [
        *first_originals,
        *(dict(frame) for frame in manifest_synthetics.values()),
    ]
    frames.sort(
        key=lambda frame: (
            float(frame.get("timestamp", 0.0)),
            0 if frame.get("kind") == "original" else 1,
            int(frame.get("source_index", -1))
            if frame.get("kind") == "original"
            else target_order[str(frame.get("target_id"))],
        )
    )
    source_digest = _frames_digest(frames, "original")
    synthetic_digest = _frames_digest(frames, "synthetic")
    envelope_hashes = [file_sha256(by_index[index][0]) for index in range(shard_count)]
    merge_identity = canonical_sha256(
        {
            "contract": contract_digest,
            "envelopes": envelope_hashes,
            "source": source_digest,
            "synthetic": synthetic_digest,
        }
    )
    generation_id = merge_identity[:32]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    signature = str(contract["experiment_signature"])
    report_path = output / f"preprocess_report_{signature}_{generation_id}.json"
    manifest_path = output / f"manifest_{signature}_{generation_id}.json"

    merged_partitions = {
        name: _sort_records_by_plan(records.values(), target_order)
        for name, records in partition_records.items()
        if records
        or any(
            name == existing
            for _, _, report, _ in loaded
            for existing, _ in _report_partitions(report)
        )
    }
    quality_partition: dict[str, list[dict[str, Any]]] | None = None
    if uses_quality_partition:
        quality_partition = {
            name: merged_partitions.get(name, [])
            for name in ("sharp_accepted", "geometry_only", "rejected")
        }
        merged_partitions["rejected"] = _sort_records_by_plan(
            injection_rejected_records.values(), target_order
        )
    # The old schema has only accepted/rejected.  Double-gate reports retain
    # both the injection-policy top-level view and the disjoint scientific
    # quality_partition.
    rejected_records = merged_partitions.get("rejected", [])
    ordered_reports = [by_index[index][2] for index in range(shard_count)]
    shared_provenance = _shared_report_provenance(ordered_reports)
    merged_report: dict[str, Any] = {
        "schema": MERGED_REPORT_SCHEMA,
        "backend": first_report.get("backend"),
        "backend_test_only": bool(first_report.get("backend_test_only", True)),
        "uses_ground_truth_pose": False,
        "pose_source": first_report.get("pose_source"),
        "preprocess_signature": signature,
        "experiment_signature": signature,
        "generation_id": generation_id,
        "source_frame_count": len(first_originals),
        "planned_total_before_cap": len(planned),
        "planned_target_count": len(planned),
        "selected_target_count": len(planned),
        "target_selection_policy": first_report.get("target_selection_policy"),
        "max_targets": first_report.get("max_targets"),
        "generation_batch_count": len(generation_batches),
        "backend_generate_call_count": sum(
            int(by_index[index][2].get("backend_generate_call_count", 0))
            for index in range(shard_count)
        ),
        "accepted_target_count": len(manifest_synthetics),
        "rejected_target_count": len(rejected_records),
        "accepted_output_sha256": synthetic_digest,
        "source_input_sha256": source_digest,
        "manifest": str(manifest_path),
        "generation_batches": generation_batches,
        "planned": planned,
        **merged_partitions,
        "shard_runtime_identity": runtime_identity,
        "shared_shard_provenance": shared_provenance,
        "shard_merge": {
            "schema": "unblur_slam.framecrafter_shard_merge.v1",
            "assignment_algorithm": ASSIGNMENT_ALGORITHM,
            "shard_count": shard_count,
            "contract_sha256": contract_digest,
            "source_identity_sha256": contract["source_identity_sha256"],
            "model_artifact_identity_sha256": contract[
                "model_artifact_identity_sha256"
            ],
            "semantic_config_sha256": contract["semantic_config_sha256"],
            "implementation_identity_sha256": contract[
                "implementation_identity_sha256"
            ],
            "canonical_preprocess_signature": contract[
                "canonical_preprocess_signature"
            ],
            "global_plan_sha256": contract["global_plan_sha256"],
            "assignment_table_sha256": contract["assignment_table_sha256"],
            "envelopes": [
                {
                    "shard_index": index,
                    "path": str(by_index[index][0]),
                    "sha256": envelope_hashes[index],
                }
                for index in range(shard_count)
            ],
        },
    }
    if quality_partition is not None:
        merged_report["quality_partition"] = quality_partition
        merged_report["geometry_rejected_target_count"] = len(
            quality_partition["rejected"]
        )
    if "sharp_accepted" in merged_partitions:
        merged_report["sharp_accepted_target_count"] = len(
            merged_partitions["sharp_accepted"]
        )
    if "geometry_only" in merged_partitions:
        merged_report["geometry_only_target_count"] = len(
            merged_partitions["geometry_only"]
        )
    if "acceptance_mode" in shared_provenance:
        merged_report["acceptance_mode"] = shared_provenance["acceptance_mode"]
    _write_immutable_json(report_path, merged_report)
    merged_manifest: dict[str, Any] = {
        "schema": MERGED_MANIFEST_SCHEMA,
        "source_frame_count": len(first_originals),
        "generated_frame_count": len(manifest_synthetics),
        "pose_source": first_manifest.get("pose_source"),
        "uses_ground_truth_pose": False,
        "frames": frames,
        "preprocess_signature": signature,
        "experiment_signature": signature,
        "shard_runtime_identity": runtime_identity,
        "generation_id": generation_id,
        "backend": first_report.get("backend"),
        "backend_test_only": bool(first_report.get("backend_test_only", True)),
        "accepted_output_sha256": synthetic_digest,
        "source_input_sha256": source_digest,
        "preprocess_report_path": str(report_path),
        "preprocess_report_sha256": file_sha256(report_path),
        "shard_merge_sha256": merge_identity,
    }
    _write_immutable_json(manifest_path, merged_manifest)
    return report_path, manifest_path


__all__ = [
    "ASSIGNMENT_ALGORITHM",
    "MERGED_MANIFEST_SCHEMA",
    "MERGED_REPORT_SCHEMA",
    "SHARD_CONTRACT_SCHEMA",
    "SHARD_ENVELOPE_SCHEMA",
    "SHARD_RUNTIME_IDENTITY_SCHEMA",
    "assigned_batch_ids",
    "build_assignment_table",
    "build_shard_contract",
    "build_shard_runtime_identity",
    "canonical_sha256",
    "file_sha256",
    "merge_shard_envelopes",
    "validate_global_plan_against_contract",
    "validate_runtime_identity_against_contract",
    "validate_shard_contract",
    "validate_shard_runtime_identity",
    "write_shard_envelope",
    "write_shard_contract",
]
