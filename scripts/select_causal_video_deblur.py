#!/usr/bin/env python3
"""Apply the pre-registered causal-EVSSM Layer-1/Layer-2 selection policy.

This tool deliberately does not export a model.  ``--stage temporal`` consumes
only H=3 temporal validation and the separately trained H=1 spatial control;
it cannot accept or open room2.  Only after that stage passes,
``--stage room2`` accepts its exact report SHA-256, validates it before opening
room2, and writes Layer 2 plus the exporter-compatible composite report.
Every metric is recomputed from per-frame evaluator rows and bound to pinned
source identities.

Layer 1 is the only model/history selection split.  Layer 2 is a locked,
one-shot cross-scene test; its thresholds are the registered constants shared
with the exporter and must never be adjusted after observing room2.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_causal_video_deblur import (
    ALIGNMENT_DIAGNOSTICS_SCHEMA_V4,
    CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1,
    CHECKPOINT_MIGRATION_KIND_V1,
    CHECKPOINT_MIGRATION_SCHEMA_V1,
    CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1,
    CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1,
    CHECKPOINT_FORMAT_V3,
    CHECKPOINT_FORMAT_V4,
    DEPLOYMENT_LAYER_REPORT_SCHEMA_V1,
    DEPLOYMENT_LAYER_REPORT_SCHEMA_V4,
    DEPLOYMENT_SELECTION_POLICY_V1,
    DEPLOYMENT_SELECTION_POLICY_V4,
    DEPLOYMENT_SELECTION_SCHEMA_V3,
    DEPLOYMENT_THRESHOLDS,
    EVALUATOR_SCHEMA_V4,
    ORACLE_GOOD_DEFINITION,
    REGISTERED_CONTRACT_SCHEMA,
    REGISTERED_CONTRACT_SHA256,
    REGISTERED_EVSSM_SHA256,
    REGISTERED_V4_CONTRACT_SCHEMA,
    REGISTERED_V4_CONTRACT_SHA256,
    REGISTERED_V4_BASE_MODEL_CONFIG,
    REGISTERED_V4_DATA_IDENTITY,
    REGISTERED_V4_MODEL_CONFIG,
    REGISTERED_V4_WARM_START_SHA256,
    RNG_STATE_SCHEMA_V4,
    ROOM2_ONE_SHOT_FRAME_COUNT,
    ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256,
    ROOM2_ONE_SHOT_MANIFEST_SHA256,
    TEMPORAL_VALIDATION_MANIFEST_SHA256,
    TORCHSCRIPT_FORMAT_V4,
    WARM_START_SCHEMA_V4,
    _resolve_layer_report,
    _validate_v4_alignment_evidence,
)

EVALUATOR_SCHEMA = "unblur_slam.causal_video_deblur_smoke_eval.v3"
EXPECTED_HISTORY = 3
EXPECTED_INPUT_DOMAIN = "evssm"
EXPECTED_EVSSM_SHA256 = REGISTERED_EVSSM_SHA256
PINNED_HISTORY1_EVALUATOR_REPORT_SHA256 = (
    "2b7531f37b85d854c2e24301e8c2ac3e2d24fd773f21fd6ef816cdebbc89b009"
)
PINNED_HISTORY1_ARTIFACT_SHA256 = (
    "3311e3adef39f551165c6b89490e99d5a351bc8e2d7f1bc6d1a1e48f100c9653"
)
PINNED_HISTORY1_CHECKPOINT_SHA256 = (
    "d373d72e0b719aa7464f303a98ffb13c8259f6fe6b0f819b7d5805421ef9405d"
)
PINNED_V4_PRE_MIGRATION_CHECKPOINT_SHA256 = (
    "ad80e84f67f6c979de96ce2a65ceeb7201b2cf7f7159af64d8fe2c2face030e0"
)
PINNED_V4_SAFE_MIGRATED_CHECKPOINT_SHA256 = (
    "92a1ab5301355e923fbd8c2059bbb0c5bdbe041cc00880b21591efdfd7de5bfd"
)
PINNED_V4_MIGRATION_SEMANTIC_DIGEST_SHA256 = (
    "a533ca551efc7543034ee73b64539c2056c913dcc4f9183df8ebbec4426c2c9d"
)

DEFAULT_TEMPORAL_MANIFEST = Path(
    "/srv/szha0669/unblur-slam/causal_video_data/replica424_v1/"
    "manifests/val_temporal.jsonl"
)
DEFAULT_ROOM2_MANIFEST = Path(
    "/srv/szha0669/unblur-slam/causal_video_data/replica424_v1/"
    "manifests/test_room2.jsonl"
)
DEFAULT_SOURCE_ROOT = Path("/srv/szha0669/unblur-slam/causal_video_data")
CANONICAL_SPLIT_SHA256 = {
    "temporal_val": TEMPORAL_VALIDATION_MANIFEST_SHA256,
    "room2_test": ROOM2_ONE_SHOT_MANIFEST_SHA256,
}

# Selection thresholds are imported from the exporter, so evaluator aggregation
# cannot silently drift from the deployment validator.  The oracle definition
# is an independent, locked interpretation of "accepted output remains good".
ORACLE_DEFINITION = dict(ORACLE_GOOD_DEFINITION)
EXPECTED_LPIPS_PROTOCOL = {
    "implementation": "torchmetrics.image.lpip",
    "network": "alex",
    "normalize_input_0_1": True,
    "per_frame_state_reset": True,
}
FRAME_IDENTITY_SCHEMA = "sorted_compact_json_sequence_index_blurry_sharp.v1"


class ReportContractError(ValueError):
    """The evaluator report is malformed or not bound to the locked split."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReportContractError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ReportContractError(f"{label} must be finite")
    return number


def _load_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReportContractError(f"{label} is not valid JSON: {resolved}") from error
    if not isinstance(payload, dict):
        raise ReportContractError(f"{label} must contain a JSON object")
    return payload


def _resolve_asset(value: Any, root: Path, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReportContractError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def _load_source_manifest(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    source_root: Path,
) -> tuple[dict[tuple[str, int], dict[str, str]], dict[str, int], dict[str, Any]]:
    manifest = path.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"{label} source manifest does not exist: {manifest}")
    digest = sha256_file(manifest)
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{label} source root does not exist: {root}")
    if digest != expected_sha256:
        raise ReportContractError(
            f"{label} source manifest SHA-256 mismatch: {digest} != {expected_sha256}"
        )
    expected: dict[tuple[str, int], dict[str, str]] = {}
    lengths: dict[str, int] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReportContractError(
                    f"{label} manifest line {line_number} is invalid JSON"
                ) from error
            sequence = record.get("sequence")
            blurry = record.get("blurry")
            sharp = record.get("sharp")
            if not isinstance(sequence, str) or not sequence:
                raise ReportContractError(
                    f"{label} manifest line {line_number} has no sequence"
                )
            if sequence in lengths:
                raise ReportContractError(f"{label} repeats sequence {sequence!r}")
            if (
                not isinstance(blurry, list)
                or not isinstance(sharp, list)
                or not blurry
                or len(blurry) != len(sharp)
            ):
                raise ReportContractError(
                    f"{label} sequence {sequence!r} has invalid blurry/sharp lists"
                )
            lengths[sequence] = len(blurry)
            for frame_index, (blurry_path, sharp_path) in enumerate(zip(blurry, sharp)):
                key = (sequence, frame_index)
                expected[key] = {
                    "blurry_path": _resolve_asset(
                        blurry_path, root, f"{label} blurry path"
                    ),
                    "sharp_path": _resolve_asset(
                        sharp_path, root, f"{label} sharp path"
                    ),
                }
    if not expected:
        raise ReportContractError(f"{label} source manifest is empty")
    identity_rows = [
        [
            sequence,
            frame_index,
            expected[(sequence, frame_index)]["blurry_path"],
            expected[(sequence, frame_index)]["sharp_path"],
        ]
        for sequence, frame_index in sorted(expected)
    ]
    frame_identity_sha256 = hashlib.sha256(
        json.dumps(
            identity_rows, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "path": str(manifest),
        "source_root": str(root),
        "sha256": digest,
        "frame_count": len(expected),
        "sequence_count": len(lengths),
        "sequence_lengths": dict(sorted(lengths.items())),
        "frame_identity_schema": FRAME_IDENTITY_SCHEMA,
        "frame_identity_sha256": frame_identity_sha256,
    }
    return expected, lengths, metadata


def _metric(row: dict[str, Any], source: str, metric: str, label: str) -> float:
    source_payload = row.get(source)
    if not isinstance(source_payload, dict):
        raise ReportContractError(f"{label}.{source} must be an object")
    if metric not in source_payload:
        raise KeyError(metric)
    return _finite(source_payload[metric], f"{label}.{source}.{metric}")


def _temporal_metric(
    row: dict[str, Any], source: str, label: str
) -> float:
    temporal = row.get("temporal")
    if not isinstance(temporal, dict):
        raise ReportContractError(f"{label}.temporal must be an object")
    source_payload = temporal.get(source)
    if not isinstance(source_payload, dict):
        raise ReportContractError(f"{label}.temporal.{source} must be an object")
    return _finite(
        source_payload.get("gt_difference_error_l1_not_warp"),
        f"{label}.temporal.{source}.gt_difference_error_l1_not_warp",
    )


def _validate_report(
    payload: dict[str, Any],
    *,
    report_path: Path,
    expected_rows: dict[tuple[str, int], dict[str, str]],
    sequence_lengths: dict[str, int],
    role: str,
    expected_history: int,
    evaluator_schema: str = EVALUATOR_SCHEMA,
) -> list[dict[str, Any]]:
    if payload.get("schema") != evaluator_schema:
        raise ReportContractError(
            f"{role} evaluator schema must be {evaluator_schema!r}"
        )
    if str(payload.get("input_domain", "")).lower() != EXPECTED_INPUT_DOMAIN:
        raise ReportContractError(f"{role} input_domain must be evssm")
    if not isinstance(payload.get("lpips_computed"), bool):
        raise ReportContractError(f"{role}.lpips_computed must be boolean")
    if payload["lpips_computed"] and payload.get("lpips_protocol") != EXPECTED_LPIPS_PROTOCOL:
        raise ReportContractError(f"{role}.lpips_protocol is unsupported")
    history_value = payload.get("history")
    if type(history_value) is not int:
        raise ReportContractError(f"{role} history must be an integer")
    history = history_value
    if history != expected_history:
        raise ReportContractError(
            f"{role} history={history} does not match required H={expected_history}"
        )
    provenance = payload.get("teacher_provenance")
    if not isinstance(provenance, dict):
        raise ReportContractError(f"{role} has no teacher_provenance")
    if provenance.get("evssm_checkpoint_sha256") != EXPECTED_EVSSM_SHA256:
        raise ReportContractError(f"{role} does not use the pinned Unblur-SLAM EVSSM")

    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ReportContractError(f"{role}.frames must be a list")
    if len(frames) != len(expected_rows):
        raise ReportContractError(
            f"{role} has {len(frames)} frames, expected {len(expected_rows)}"
        )
    if type(payload.get("frame_count")) is not int or payload["frame_count"] != len(frames):
        raise ReportContractError(f"{role}.frame_count disagrees with frames")

    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for row_number, row in enumerate(frames):
        label = f"{role}.frames[{row_number}]"
        if not isinstance(row, dict):
            raise ReportContractError(f"{label} must be an object")
        sequence = row.get("sequence")
        if not isinstance(sequence, str) or sequence not in sequence_lengths:
            raise ReportContractError(f"{label} has an unknown sequence")
        frame_index_value = row.get("frame_index")
        if type(frame_index_value) is not int:
            raise ReportContractError(f"{label}.frame_index must be an integer")
        frame_index = frame_index_value
        key = (sequence, frame_index)
        if key in observed:
            raise ReportContractError(f"{role} repeats frame {key!r}")
        if key not in expected_rows:
            raise ReportContractError(f"{role} contains unexpected frame {key!r}")
        expected_paths = expected_rows[key]
        for path_key in ("blurry_path", "sharp_path"):
            actual = _resolve_asset(
                row.get(path_key), report_path.parent, f"{label}.{path_key}"
            )
            if actual != expected_paths[path_key]:
                raise ReportContractError(
                    f"{label}.{path_key} does not match the pinned source manifest"
                )
        expected_stage = "prefix" if frame_index < expected_history - 1 else "steady_state"
        if row.get("history_stage") != expected_stage:
            raise ReportContractError(
                f"{label}.history_stage must be {expected_stage!r}"
            )
        for source in ("evssm", "causal", "causal_repeat_current"):
            for metric in ("psnr", "ssim", "l1"):
                _metric(row, source, metric, label)

        gate = row.get("runtime_gate_proxy")
        if not isinstance(gate, dict):
            raise ReportContractError(f"{label}.runtime_gate_proxy must be an object")
        laplacian = {
            source: _finite(
                gate.get(f"{source}_laplacian_variance"),
                f"{label}.runtime_gate_proxy.{source}_laplacian_variance",
            )
            for source in ("blurry", "evssm", "causal")
        }
        if any(value < 0.0 for value in laplacian.values()):
            raise ReportContractError(
                f"{label}.runtime_gate_proxy Laplacian variances must be non-negative"
            )
        vs_evssm = _finite(
            gate.get("causal_vs_evssm_gain"),
            f"{label}.runtime_gate_proxy.causal_vs_evssm_gain",
        )
        vs_blurry = _finite(
            gate.get("causal_vs_blurry_gain"),
            f"{label}.runtime_gate_proxy.causal_vs_blurry_gain",
        )
        recomputed_vs_evssm = (
            laplacian["causal"] - laplacian["evssm"]
        ) / max(laplacian["evssm"], 1.0e-12)
        recomputed_vs_blurry = (
            laplacian["causal"] - laplacian["blurry"]
        ) / max(laplacian["blurry"], 1.0e-12)
        if not math.isclose(
            vs_evssm, recomputed_vs_evssm, rel_tol=1.0e-7, abs_tol=1.0e-10
        ) or not math.isclose(
            vs_blurry, recomputed_vs_blurry, rel_tol=1.0e-7, abs_tol=1.0e-10
        ):
            raise ReportContractError(
                f"{label}.runtime_gate_proxy gains disagree with Laplacian variances"
            )
        expected_pass = recomputed_vs_evssm >= 0.0 and recomputed_vs_blurry >= 0.02
        if gate.get("passes_default_gate") is not expected_pass:
            raise ReportContractError(
                f"{label}.runtime_gate_proxy.passes_default_gate is inconsistent"
            )

        if frame_index == 0:
            if row.get("temporal") is not None:
                raise ReportContractError(
                    f"{label}.temporal must reset at the sequence boundary"
                )
        else:
            for source in ("evssm", "causal", "causal_repeat_current"):
                _temporal_metric(row, source, label)
        observed[key] = row

    if set(observed) != set(expected_rows):
        raise ReportContractError(f"{role} does not exactly cover its source manifest")
    return [observed[key] for key in sorted(observed)]


def _mean(values: Iterable[float], label: str) -> float:
    items = list(values)
    if not items:
        raise ReportContractError(f"cannot compute {label} from zero samples")
    result = sum(items) / len(items)
    if not math.isfinite(result):
        raise ReportContractError(f"{label} is non-finite")
    return result


def _relative(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0.0:
        raise ReportContractError(f"{label} denominator must be positive")
    value = numerator / denominator - 1.0
    if not math.isfinite(value):
        raise ReportContractError(f"{label} is non-finite")
    return value


def _quality(rows: list[dict[str, Any]], *, require_lpips: bool) -> dict[str, Any]:
    metrics = ("psnr", "ssim", "l1")
    means: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for source in ("evssm", "causal", "causal_repeat_current"):
        means[source] = {
            metric: _mean(
                (_metric(row, source, metric, "frame") for row in rows),
                f"{source}.{metric}",
            )
            for metric in metrics
        }
        lpips_values: list[float] = []
        for row in rows:
            try:
                lpips_values.append(_metric(row, source, "lpips", "frame"))
            except KeyError:
                missing.append(f"frames[].{source}.lpips")
                lpips_values = []
                break
        if lpips_values:
            means[source]["lpips"] = _mean(lpips_values, f"{source}.lpips")
    if require_lpips and missing:
        # Preserve one stable entry per missing field rather than one per frame.
        missing = sorted(set(missing))
    else:
        missing = []
    result = {
        "frame_count": len(rows),
        "mean": means,
        "delta_psnr_db": means["causal"]["psnr"] - means["evssm"]["psnr"],
        "delta_ssim": means["causal"]["ssim"] - means["evssm"]["ssim"],
        "relative_l1": _relative(
            means["causal"]["l1"], means["evssm"]["l1"], "relative_l1"
        ),
        "history_delta_psnr_db": (
            means["causal"]["psnr"] - means["causal_repeat_current"]["psnr"]
        ),
        "missing_metrics": missing,
    }
    if "lpips" in means["causal"] and "lpips" in means["evssm"]:
        result["delta_lpips"] = means["causal"]["lpips"] - means["evssm"]["lpips"]
    else:
        result["delta_lpips"] = None
    return result


def _temporal(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    temporal_rows = [row for row in rows if row.get("temporal") is not None]
    evssm = _mean(
        (_temporal_metric(row, "evssm", "frame") for row in temporal_rows),
        "EVSSM temporal error",
    )
    causal = _mean(
        (_temporal_metric(row, "causal", "frame") for row in temporal_rows),
        "causal temporal error",
    )
    repeat = _mean(
        (
            _temporal_metric(row, "causal_repeat_current", "frame")
            for row in temporal_rows
        ),
        "repeat-current temporal error",
    )
    return {
        "pair_count": len(temporal_rows),
        "evssm_gt_difference_error_l1": evssm,
        "causal_gt_difference_error_l1": causal,
        "repeat_current_gt_difference_error_l1": repeat,
        "causal_minus_evssm_gt_difference_error_l1": causal - evssm,
        "causal_relative_to_evssm": _relative(
            causal, evssm, "causal temporal relative to EVSSM"
        ),
        "causal_relative_to_repeat_current": _relative(
            causal, repeat, "causal temporal relative to repeat-current"
        ),
    }


def _gate_and_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        row
        for row in rows
        if bool(row["runtime_gate_proxy"]["passes_default_gate"])
    ]
    oracle = ORACLE_DEFINITION
    decisions = []
    for row in accepted:
        evssm_l1 = _metric(row, "evssm", "l1", "frame")
        relative_l1 = _relative(
            _metric(row, "causal", "l1", "frame"),
            evssm_l1,
            "oracle relative_l1",
        )
        delta_psnr = _metric(row, "causal", "psnr", "frame") - _metric(
            row, "evssm", "psnr", "frame"
        )
        delta_ssim = _metric(row, "causal", "ssim", "frame") - _metric(
            row, "evssm", "ssim", "frame"
        )
        good = (
            delta_psnr >= float(oracle["delta_psnr_db_min"])
            and delta_ssim >= float(oracle["delta_ssim_min"])
            and relative_l1 <= float(oracle["relative_l1_max"])
        )
        decisions.append(
            {
                "sequence": row["sequence"],
                "frame_index": int(row["frame_index"]),
                "delta_psnr_db": delta_psnr,
                "delta_ssim": delta_ssim,
                "relative_l1": relative_l1,
                "oracle_good": bool(good),
            }
        )
    good_count = sum(bool(item["oracle_good"]) for item in decisions)
    return {
        "frame_count": len(rows),
        "pass_count": len(accepted),
        "pass_ratio": len(accepted) / len(rows),
        "oracle_good_count": good_count,
        "oracle_precision": good_count / len(accepted) if accepted else 0.0,
        "oracle_definition": dict(oracle),
        "accepted_frames": decisions,
    }


def _check(value: Any, operator: str, threshold: Any) -> dict[str, Any]:
    if value is None:
        passed = False
    elif operator == ">=":
        passed = value >= threshold
    elif operator == "<=":
        passed = value <= threshold
    elif operator == "==":
        passed = value == threshold
    else:
        raise ValueError(f"unsupported operator: {operator}")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _per_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence"])].append(row)
    results = []
    for sequence, sequence_rows in sorted(grouped.items()):
        steady = [
            row for row in sequence_rows if row["history_stage"] == "steady_state"
        ]
        results.append(
            {
                "sequence": sequence,
                "frame_count": len(sequence_rows),
                "steady_count": len(steady),
                "delta_psnr_db": _mean(
                    (
                        _metric(row, "causal", "psnr", "frame")
                        - _metric(row, "evssm", "psnr", "frame")
                        for row in sequence_rows
                    ),
                    f"{sequence} delta PSNR",
                ),
                "steady_delta_psnr_db": (
                    _mean(
                        (
                            _metric(row, "causal", "psnr", "frame")
                            - _metric(row, "evssm", "psnr", "frame")
                            for row in steady
                        ),
                        f"{sequence} steady delta PSNR",
                    )
                    if steady
                    else None
                ),
            }
        )
    return results


def _layer1(
    rows: list[dict[str, Any]], history1_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    steady = [row for row in rows if row["history_stage"] == "steady_state"]
    prefix = [row for row in rows if row["history_stage"] == "prefix"]
    quality_steady = _quality(steady, require_lpips=False)
    quality_prefix = _quality(prefix, require_lpips=False)
    temporal_steady = _temporal(steady)
    history1_by_key = {
        (str(row["sequence"]), int(row["frame_index"])): row
        for row in history1_rows
    }
    steady_keys = sorted(
        (str(row["sequence"]), int(row["frame_index"])) for row in steady
    )
    if any(key not in history1_by_key for key in steady_keys):
        raise ReportContractError("H1 report does not cover every H3 steady frame")
    h3_steady_psnr_db = _mean(
        (_metric(row, "causal", "psnr", "H3 steady frame") for row in steady),
        "H3 steady causal PSNR",
    )
    h1_steady_psnr_db = _mean(
        (
            _metric(
                history1_by_key[key],
                "causal",
                "psnr",
                "H1 aligned steady frame",
            )
            for key in steady_keys
        ),
        "H1 PSNR on H3 steady keys",
    )
    steady_frame_keys_sha256 = hashlib.sha256(
        json.dumps(
            [[sequence, frame_index] for sequence, frame_index in steady_keys],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    gate = _gate_and_oracle(rows)
    runs = _per_run(rows)
    threshold = DEPLOYMENT_THRESHOLDS["temporal_validation"]
    metrics = {
        "steady_psnr_delta_db": quality_steady["delta_psnr_db"],
        "steady_ssim_delta": quality_steady["delta_ssim"],
        "steady_relative_l1_delta": quality_steady["relative_l1"],
        "steady_gt_temporal_difference_relative_delta": temporal_steady[
            "causal_relative_to_evssm"
        ],
        "normal_vs_repeat_current_psnr_delta_db": quality_steady[
            "history_delta_psnr_db"
        ],
        "normal_vs_history1_psnr_delta_db": (
            h3_steady_psnr_db - h1_steady_psnr_db
        ),
        "normal_vs_repeat_current_temporal_relative_delta": temporal_steady[
            "causal_relative_to_repeat_current"
        ],
        "laplacian_gate_pass_ratio": gate["pass_ratio"],
        "accepted_oracle_precision": gate["oracle_precision"],
        "worst_run_psnr_delta_db": min(
            float(run["delta_psnr_db"]) for run in runs
        ),
        "prefix_psnr_delta_db": quality_prefix["delta_psnr_db"],
    }
    checks = {
        "steady_psnr_delta_db": _check(
            metrics["steady_psnr_delta_db"],
            ">=",
            threshold["steady_psnr_delta_db_min"],
        ),
        "steady_ssim_delta": _check(
            metrics["steady_ssim_delta"], ">=", threshold["steady_ssim_delta_min"]
        ),
        "steady_relative_l1_delta": _check(
            metrics["steady_relative_l1_delta"],
            "<=",
            threshold["steady_relative_l1_delta_max"],
        ),
        "steady_gt_temporal_difference_relative_delta": _check(
            metrics["steady_gt_temporal_difference_relative_delta"],
            "<=",
            threshold["steady_gt_temporal_difference_relative_delta_max"],
        ),
        "normal_vs_repeat_current_psnr_delta_db": _check(
            metrics["normal_vs_repeat_current_psnr_delta_db"],
            ">=",
            threshold["normal_vs_repeat_current_psnr_delta_db_min"],
        ),
        "normal_vs_history1_psnr_delta_db": _check(
            metrics["normal_vs_history1_psnr_delta_db"],
            ">=",
            threshold["normal_vs_history1_psnr_delta_db_min"],
        ),
        "normal_vs_repeat_current_temporal_relative_delta": _check(
            metrics["normal_vs_repeat_current_temporal_relative_delta"],
            "<=",
            threshold["normal_vs_repeat_current_temporal_relative_delta_max"],
        ),
        "laplacian_gate_pass_ratio": _check(
            metrics["laplacian_gate_pass_ratio"],
            ">=",
            threshold["laplacian_gate_pass_ratio_min"],
        ),
        "accepted_oracle_precision": _check(
            metrics["accepted_oracle_precision"],
            ">=",
            threshold["accepted_oracle_precision_min"],
        ),
        "worst_run_psnr_delta_db": _check(
            metrics["worst_run_psnr_delta_db"],
            ">=",
            threshold["worst_run_psnr_delta_db_min"],
        ),
        "prefix_psnr_delta_db": _check(
            metrics["prefix_psnr_delta_db"],
            ">=",
            threshold["prefix_psnr_delta_db_min"],
        ),
    }
    return {
        "role": "checkpoint_and_history_selection",
        "metrics": metrics,
        "quality_steady_state": quality_steady,
        "quality_prefix": quality_prefix,
        "temporal_steady_state": temporal_steady,
        "gate_and_oracle_all_frames": gate,
        "per_run": runs,
        "history1_comparison": {
            "h3_steady_psnr_db": h3_steady_psnr_db,
            "h1_steady_psnr_db": h1_steady_psnr_db,
            "steady_frame_count": len(steady_keys),
            "steady_frame_keys_sha256": steady_frame_keys_sha256,
        },
        "checks": checks,
        "passed": all(bool(item["passed"]) for item in checks.values()),
    }


def _layer2(rows: list[dict[str, Any]], *, lpips_computed: bool) -> dict[str, Any]:
    steady = [row for row in rows if row["history_stage"] == "steady_state"]
    quality_all = _quality(rows, require_lpips=True)
    quality_steady = _quality(steady, require_lpips=False)
    temporal = _temporal(rows)
    gate = _gate_and_oracle(rows)
    runs = _per_run(rows)
    long_runs = [run for run in runs if int(run["frame_count"]) >= EXPECTED_HISTORY]
    nonregression_count = sum(
        float(run["steady_delta_psnr_db"]) >= 0.0 for run in long_runs
    )
    threshold = DEPLOYMENT_THRESHOLDS["room2_one_shot"]
    lpips_present = bool(
        lpips_computed
        and not quality_all["missing_metrics"]
        and quality_all["delta_lpips"] is not None
    )
    metrics = {
        "psnr_delta_db": quality_all["delta_psnr_db"],
        "ssim_delta": quality_all["delta_ssim"],
        "relative_l1_delta": quality_all["relative_l1"],
        "lpips_delta": quality_all["delta_lpips"],
        "gt_temporal_difference_delta": temporal[
            "causal_minus_evssm_gt_difference_error_l1"
        ],
        "steady_normal_vs_repeat_current_psnr_delta_db": quality_steady[
            "history_delta_psnr_db"
        ],
        "laplacian_gate_pass_ratio": gate["pass_ratio"],
        "accepted_oracle_precision": gate["oracle_precision"],
        "nondegraded_long_runs": nonregression_count,
        "long_runs_total": len(long_runs),
        "worst_run_psnr_delta_db": min(
            float(run["delta_psnr_db"]) for run in runs
        ),
    }
    checks = {
        "lpips_present": _check(lpips_present, "==", True),
        "psnr_delta_db": _check(
            metrics["psnr_delta_db"], ">=", threshold["psnr_delta_db_min"]
        ),
        "ssim_delta": _check(
            metrics["ssim_delta"], ">=", threshold["ssim_delta_min"]
        ),
        "relative_l1_delta": _check(
            metrics["relative_l1_delta"], "<=", threshold["relative_l1_delta_max"]
        ),
        "lpips_delta": _check(
            metrics["lpips_delta"], "<=", threshold["lpips_delta_max"]
        ),
        "gt_temporal_difference_delta": _check(
            metrics["gt_temporal_difference_delta"],
            "<=",
            threshold["gt_temporal_difference_delta_max"],
        ),
        "steady_normal_vs_repeat_current_psnr_delta_db": _check(
            metrics["steady_normal_vs_repeat_current_psnr_delta_db"],
            ">=",
            threshold["steady_normal_vs_repeat_current_psnr_delta_db_min"],
        ),
        "laplacian_gate_pass_ratio": _check(
            metrics["laplacian_gate_pass_ratio"],
            ">=",
            threshold["laplacian_gate_pass_ratio_min"],
        ),
        "accepted_oracle_precision": _check(
            metrics["accepted_oracle_precision"],
            ">=",
            threshold["accepted_oracle_precision_min"],
        ),
        "nondegraded_long_runs": _check(
            metrics["nondegraded_long_runs"],
            ">=",
            threshold["nondegraded_long_runs_min"],
        ),
        "long_runs_total": _check(
            metrics["long_runs_total"], "==", threshold["long_runs_total"]
        ),
        "worst_run_psnr_delta_db": _check(
            metrics["worst_run_psnr_delta_db"],
            ">=",
            threshold["worst_run_psnr_delta_db_min"],
        ),
    }
    missing = list(quality_all["missing_metrics"])
    if not lpips_computed:
        missing.append("summary.lpips_computed=true")
    return {
        "role": "scene_disjoint_one_shot_test",
        "metrics": metrics,
        "quality_all_frames": quality_all,
        "quality_steady_state": quality_steady,
        "temporal_all_pairs": temporal,
        "gate_and_oracle_all_frames": gate,
        "per_run": runs,
        "long_run_count": len(long_runs),
        "long_run_nonregression_count": nonregression_count,
        "missing_metrics": sorted(set(missing)),
        "checks": checks,
        "passed": all(bool(item["passed"]) for item in checks.values()),
    }


def _atomic_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite selection report: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The file itself is already fsynced and atomically replaced; some
            # filesystems do not permit fsync on a directory descriptor.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_digest(value: Any, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ReportContractError(f"{label} must be a SHA-256 digest")
    return digest


def _validated_evaluator_artifact(
    payload: dict[str, Any], report_path: Path, role: str
) -> tuple[Path, str, str]:
    artifact = Path(str(payload.get("checkpoint", ""))).expanduser()
    if not artifact.is_absolute():
        artifact = report_path.parent / artifact
    artifact = artifact.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"{role} evaluated artifact does not exist: {artifact}")
    artifact_sha256 = sha256_file(artifact)
    if _sha256_digest(
        payload.get("evaluated_artifact_sha256"),
        f"{role}.evaluated_artifact_sha256",
    ) != artifact_sha256:
        raise ReportContractError(
            f"{role} evaluated artifact SHA-256 does not match its checkpoint"
        )
    source_sha256 = _sha256_digest(
        payload.get("source_checkpoint_sha256"),
        f"{role}.source_checkpoint_sha256",
    )
    return artifact, artifact_sha256, source_sha256


def _load_torchscript_metadata(artifact: Path, role: str) -> dict[str, Any]:
    """Read the exporter's SHA-bound metadata without loading model code."""

    try:
        with zipfile.ZipFile(artifact) as archive:
            names = [
                name
                for name in archive.namelist()
                if name == "extra/metadata.json"
                or name.endswith("/extra/metadata.json")
            ]
            if len(names) != 1:
                raise ReportContractError(
                    f"{role} artifact must contain exactly one extra/metadata.json"
                )
            info = archive.getinfo(names[0])
            if info.file_size <= 0 or info.file_size > 4 * 1024 * 1024:
                raise ReportContractError(
                    f"{role} artifact metadata has an invalid size"
                )
            raw = archive.read(names[0])
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ReportContractError(
            f"{role} evaluated artifact is not a readable TorchScript archive"
        ) from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReportContractError(
            f"{role} TorchScript metadata is not valid UTF-8 JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ReportContractError(f"{role} TorchScript metadata must be an object")
    return payload


def _validate_v4_checkpoint_migration_metadata(
    metadata: dict[str, Any],
    *,
    target_checkpoint_sha256: str,
    role: str,
) -> dict[str, Any]:
    """Validate the required serialization lineage of the registered v4 run."""

    migration = metadata.get("checkpoint_migration")
    if migration is None:
        # The sole terminal artifact produced under the registered 511dbc
        # contract is the pinned legacy-ndarray checkpoint below.  Treating an
        # absent lineage as a future native-safe run would let metadata stripping
        # disguise a different artifact under this already-completed contract.
        raise ReportContractError(
            f"{role} registered v4 artifact requires checkpoint_migration lineage"
        )
    if not isinstance(migration, dict) or set(migration) != {
        "schema",
        "kind",
        "source_checkpoint_sha256",
        "allowed_changes",
        "semantic_digest",
    }:
        raise ReportContractError(
            f"{role} checkpoint_migration has an invalid field set"
        )
    if migration.get("schema") != CHECKPOINT_MIGRATION_SCHEMA_V1:
        raise ReportContractError(f"{role} checkpoint_migration schema mismatch")
    if migration.get("kind") != CHECKPOINT_MIGRATION_KIND_V1:
        raise ReportContractError(f"{role} checkpoint_migration kind mismatch")
    source_sha256 = _sha256_digest(
        migration.get("source_checkpoint_sha256"),
        f"{role} checkpoint_migration source_checkpoint_sha256",
    )
    if source_sha256 != PINNED_V4_PRE_MIGRATION_CHECKPOINT_SHA256:
        raise ReportContractError(
            f"{role} migration is not derived from the pinned formal v4 terminal"
        )
    if source_sha256 == target_checkpoint_sha256:
        raise ReportContractError(
            f"{role} migration source and tensor-safe target must have distinct SHA-256"
        )
    if target_checkpoint_sha256 != PINNED_V4_SAFE_MIGRATED_CHECKPOINT_SHA256:
        raise ReportContractError(
            f"{role} migration does not use the pinned tensor-safe v4 checkpoint"
        )
    if migration.get("allowed_changes") != CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1:
        raise ReportContractError(
            f"{role} checkpoint_migration allowed_changes mismatch"
        )
    semantic = migration.get("semantic_digest")
    if not isinstance(semantic, dict) or set(semantic) != {
        "schema",
        "algorithm",
        "sha256",
        "source_and_target_equal",
    }:
        raise ReportContractError(
            f"{role} checkpoint_migration semantic_digest has an invalid field set"
        )
    if semantic.get("schema") != CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1:
        raise ReportContractError(
            f"{role} checkpoint_migration semantic digest schema mismatch"
        )
    if semantic.get("algorithm") != CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1:
        raise ReportContractError(
            f"{role} checkpoint_migration semantic digest algorithm mismatch"
        )
    semantic_sha256 = _sha256_digest(
        semantic.get("sha256"),
        f"{role} checkpoint_migration semantic_digest.sha256",
    )
    if semantic_sha256 != PINNED_V4_MIGRATION_SEMANTIC_DIGEST_SHA256:
        raise ReportContractError(
            f"{role} checkpoint_migration does not use the pinned semantic digest"
        )
    if semantic.get("source_and_target_equal") is not True:
        raise ReportContractError(
            f"{role} checkpoint_migration does not prove semantic equality"
        )
    return {
        "schema": CHECKPOINT_MIGRATION_SCHEMA_V1,
        "kind": CHECKPOINT_MIGRATION_KIND_V1,
        "source_checkpoint_sha256": source_sha256,
        "target_checkpoint_sha256": target_checkpoint_sha256,
        "allowed_changes": list(CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1),
        "semantic_digest": {
            "schema": CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1,
            "algorithm": CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1,
            "sha256": semantic_sha256,
            "source_and_target_equal": True,
        },
    }


def _validate_v4_artifact_provenance(
    *,
    evaluator_payload: dict[str, Any],
    artifact: Path,
    artifact_sha256: str,
    source_checkpoint_sha256: str,
    role: str,
) -> dict[str, Any]:
    """Bind Layer 1 to the registered v4 diagnostic export and warm start."""

    metadata = _load_torchscript_metadata(artifact, role)
    if metadata.get("format") != TORCHSCRIPT_FORMAT_V4 or metadata.get(
        "checkpoint_format"
    ) != CHECKPOINT_FORMAT_V4:
        raise ReportContractError(f"{role} artifact is not a v4 TorchScript export")
    if metadata.get("artifact_role") != "diagnostic_evaluation_only" or metadata.get(
        "deployment_eligible"
    ) is not False:
        raise ReportContractError(
            f"{role} evaluator must consume a non-deployment diagnostic artifact"
        )
    if _sha256_digest(
        metadata.get("source_checkpoint_sha256"),
        f"{role} artifact source_checkpoint_sha256",
    ) != source_checkpoint_sha256:
        raise ReportContractError(
            f"{role} evaluator and artifact use different source checkpoints"
        )
    if _sha256_digest(
        evaluator_payload.get("evaluated_artifact_sha256"),
        f"{role}.evaluated_artifact_sha256",
    ) != artifact_sha256:
        raise ReportContractError(f"{role} artifact digest is inconsistent")
    checkpoint_migration = _validate_v4_checkpoint_migration_metadata(
        metadata,
        target_checkpoint_sha256=source_checkpoint_sha256,
        role=role,
    )

    model_config = metadata.get("model_config")
    if model_config != REGISTERED_V4_MODEL_CONFIG:
        raise ReportContractError(f"{role} artifact model_config is not preregistered")
    registered = metadata.get("registered_contract")
    if not isinstance(registered, dict) or {
        "schema": registered.get("schema"),
        "sha256": registered.get("sha256"),
    } != {
        "schema": REGISTERED_V4_CONTRACT_SCHEMA,
        "sha256": REGISTERED_V4_CONTRACT_SHA256,
    }:
        raise ReportContractError(
            f"{role} artifact is not bound to the registered v4 contract"
        )
    if {
        "epoch": metadata.get("source_checkpoint_epoch"),
        "step": metadata.get("source_checkpoint_step"),
        "phase": metadata.get("training_phase"),
    } != {"epoch": 25, "step": 600, "phase": "joint"}:
        raise ReportContractError(
            f"{role} artifact is not from the terminal epoch25/step600 joint checkpoint"
        )
    if metadata.get("data_identity") != REGISTERED_V4_DATA_IDENTITY:
        raise ReportContractError(f"{role} artifact data identity is not preregistered")
    if metadata.get("rng_state_provenance") != {
        "schema": RNG_STATE_SCHEMA_V4,
        "checkpoint_boundary": "epoch_end_no_pending_accumulation",
        "captured": True,
    }:
        raise ReportContractError(
            f"{role} artifact has no registered terminal RNG boundary"
        )

    warm_start = metadata.get("warm_start_provenance")
    if not isinstance(warm_start, dict) or warm_start.get("schema") != (
        WARM_START_SCHEMA_V4
    ):
        raise ReportContractError(f"{role} artifact has no v4 warm-start provenance")
    if _sha256_digest(
        warm_start.get("source_sha256"), f"{role} warm-start source_sha256"
    ) != REGISTERED_V4_WARM_START_SHA256 or warm_start.get(
        "source_format"
    ) != CHECKPOINT_FORMAT_V3:
        raise ReportContractError(
            f"{role} artifact does not use the registered H3 epoch20 warm start"
        )
    if warm_start.get("source_model_config") != REGISTERED_V4_BASE_MODEL_CONFIG:
        raise ReportContractError(
            f"{role} warm-start source_model_config is not preregistered"
        )
    if warm_start.get("optimizer_state_loaded") is not False:
        raise ReportContractError(f"{role} warm start must use a fresh optimizer")
    expected_missing = {
        "motion_alignment_gate",
        "motion_aligner.match_projection.weight",
        "motion_aligner.offsets",
    }
    allowed_missing = warm_start.get("allowed_missing_alignment_keys")
    if not isinstance(allowed_missing, list) or set(allowed_missing) != (
        expected_missing
    ):
        raise ReportContractError(f"{role} warm-start missing-key set is not registered")
    identity_probe = warm_start.get("identity_probe")
    if not isinstance(identity_probe, dict) or identity_probe.get("passed") is not True:
        raise ReportContractError(f"{role} warm-start identity probe did not pass")
    tolerance = _finite(identity_probe.get("atol"), f"{role} identity probe atol")
    difference = _finite(
        identity_probe.get("max_abs_difference"),
        f"{role} identity probe max_abs_difference",
    )
    if tolerance < 0.0 or difference < 0.0 or difference > tolerance:
        raise ReportContractError(f"{role} warm-start identity probe exceeds tolerance")

    teacher = metadata.get("teacher_provenance")
    if teacher != evaluator_payload.get("teacher_provenance") or not isinstance(
        teacher, dict
    ) or teacher.get("evssm_checkpoint_sha256") != EXPECTED_EVSSM_SHA256:
        raise ReportContractError(
            f"{role} evaluator/artifact teacher provenance is inconsistent"
        )
    expected_methods = {
        "forward",
        "forward_sequence",
        "forward_sequence_with_motion_diagnostics",
        "forward_sequence_alignment_disabled",
    }
    methods = metadata.get("exported_methods")
    if not isinstance(methods, list) or set(methods) != expected_methods:
        raise ReportContractError(f"{role} artifact diagnostic API is incomplete")
    optimization = metadata.get("optimization_contract")
    if not isinstance(optimization, dict) or {
        "execution_device": optimization.get("execution_device"),
        "amp_requested": optimization.get("amp_requested"),
        "amp_effective": optimization.get("amp_effective"),
        "num_workers": optimization.get("num_workers"),
    } != {
        "execution_device": "cpu",
        "amp_requested": False,
        "amp_effective": False,
        "num_workers": 0,
    }:
        raise ReportContractError(f"{role} execution provenance is not preregistered")
    training = metadata.get("training_contract")
    if not isinstance(training, dict) or training.get(
        "terminal_checkpoint_policy"
    ) != "unconditional_atomic_save_at_exact_optimizer_step_600_before_exit" or (
        training.get("resume_rng_policy")
        != "epoch_boundary_python_numpy_torch_cpu_and_loader_generators"
    ):
        raise ReportContractError(
            f"{role} terminal checkpoint/RNG policy is not preregistered"
        )
    return {
        "torchscript_format": TORCHSCRIPT_FORMAT_V4,
        "checkpoint_format": CHECKPOINT_FORMAT_V4,
        "evaluated_artifact_sha256": artifact_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "registered_contract": {
            "schema": REGISTERED_V4_CONTRACT_SCHEMA,
            "sha256": REGISTERED_V4_CONTRACT_SHA256,
        },
        "warm_start": {
            "schema": WARM_START_SCHEMA_V4,
            "source_sha256": REGISTERED_V4_WARM_START_SHA256,
            "source_format": CHECKPOINT_FORMAT_V3,
            "optimizer_state_loaded": False,
            "identity_probe_passed": True,
        },
        "execution_device": "cpu",
        "amp_effective": False,
        "num_workers": 0,
        "source_checkpoint_epoch": 25,
        "source_checkpoint_step": 600,
        "training_phase": "joint",
        "data_identity": dict(REGISTERED_V4_DATA_IDENTITY),
        "rng_state": {
            "schema": RNG_STATE_SCHEMA_V4,
            "checkpoint_boundary": "epoch_end_no_pending_accumulation",
        },
        "checkpoint_migration": checkpoint_migration,
    }


def build_temporal_layer_report(
    *,
    temporal_report_path: Path,
    history1_report_path: Path,
    temporal_manifest_path: Path,
    source_root_path: Path,
    expected_temporal_manifest_sha256: str,
    expected_history1_report_sha256: str | None = None,
    expected_history1_artifact_sha256: str | None = None,
    expected_history1_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Build Layer 1 without accepting, opening, or parsing any room2 input."""

    temporal_report_path = temporal_report_path.expanduser().resolve()
    history1_report_path = history1_report_path.expanduser().resolve()
    temporal_payload = _load_json(temporal_report_path, "temporal-val report")
    history1_payload = _load_json(history1_report_path, "H1 temporal-val report")
    temporal_schema = temporal_payload.get("schema")
    if temporal_schema not in {EVALUATOR_SCHEMA, EVALUATOR_SCHEMA_V4}:
        raise ReportContractError("temporal_val evaluator schema is unsupported")
    is_v4 = temporal_schema == EVALUATOR_SCHEMA_V4
    expected, lengths, manifest = _load_source_manifest(
        temporal_manifest_path,
        expected_sha256=expected_temporal_manifest_sha256,
        label="temporal_val",
        source_root=source_root_path,
    )
    temporal_rows = _validate_report(
        temporal_payload,
        report_path=temporal_report_path,
        expected_rows=expected,
        sequence_lengths=lengths,
        role="temporal_val",
        expected_history=EXPECTED_HISTORY,
        evaluator_schema=(EVALUATOR_SCHEMA_V4 if is_v4 else EVALUATOR_SCHEMA),
    )
    history1_rows = _validate_report(
        history1_payload,
        report_path=history1_report_path,
        expected_rows=expected,
        sequence_lengths=lengths,
        role="history1_temporal_val",
        expected_history=1,
    )
    temporal_evaluator_manifest = Path(
        str(temporal_payload.get("manifest", ""))
    ).expanduser().resolve()
    history1_evaluator_manifest = Path(
        str(history1_payload.get("manifest", ""))
    ).expanduser().resolve()
    if temporal_evaluator_manifest != history1_evaluator_manifest:
        raise ReportContractError(
            "H1 and H3 evaluator reports must use the identical manifest"
        )
    h3_artifact, h3_artifact_sha256, checkpoint_sha256 = (
        _validated_evaluator_artifact(
            temporal_payload, temporal_report_path, "temporal_val"
        )
    )
    h1_artifact, h1_artifact_sha256, h1_checkpoint_sha256 = (
        _validated_evaluator_artifact(
            history1_payload, history1_report_path, "history1_temporal_val"
        )
    )
    if h1_artifact == h3_artifact or h1_artifact_sha256 == h3_artifact_sha256:
        raise ReportContractError("H1 and H3 evaluated artifacts must differ")
    if h1_checkpoint_sha256 == checkpoint_sha256:
        raise ReportContractError(
            "H1 control and H3 candidate must use different source checkpoints"
        )
    if is_v4:
        pinned_h1_report = (
            expected_history1_report_sha256
            or PINNED_HISTORY1_EVALUATOR_REPORT_SHA256
        )
        pinned_h1_artifact = (
            expected_history1_artifact_sha256
            or PINNED_HISTORY1_ARTIFACT_SHA256
        )
        pinned_h1_checkpoint = (
            expected_history1_checkpoint_sha256
            or PINNED_HISTORY1_CHECKPOINT_SHA256
        )
        if sha256_file(history1_report_path) != _sha256_digest(
            pinned_h1_report, "pinned H1 evaluator report SHA-256"
        ):
            raise ReportContractError(
                "v4 Layer1 must use the pinned H1 best evaluator report"
            )
        if h1_artifact_sha256 != _sha256_digest(
            pinned_h1_artifact, "pinned H1 artifact SHA-256"
        ) or h1_checkpoint_sha256 != _sha256_digest(
            pinned_h1_checkpoint, "pinned H1 source checkpoint SHA-256"
        ):
            raise ReportContractError(
                "v4 Layer1 must use the pinned H1 best artifact/checkpoint"
            )
    if temporal_payload.get("teacher_provenance") != history1_payload.get(
        "teacher_provenance"
    ):
        raise ReportContractError("H1 and H3 reports have different teacher provenance")

    layer = _layer1(temporal_rows, history1_rows)
    v4_provenance = None
    alignment_evidence = None
    if is_v4:
        v4_provenance = _validate_v4_artifact_provenance(
            evaluator_payload=temporal_payload,
            artifact=h3_artifact,
            artifact_sha256=h3_artifact_sha256,
            source_checkpoint_sha256=checkpoint_sha256,
            role="temporal_val",
        )
        try:
            alignment_evidence = _validate_v4_alignment_evidence(
                temporal_payload,
                label="v4 temporal evaluator",
                expected_transition_count=14,
                require_lpips=False,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ReportContractError(
                f"v4 temporal alignment evidence is invalid: {error}"
            ) from error

    report = {
        "schema": (
            DEPLOYMENT_LAYER_REPORT_SCHEMA_V4
            if is_v4
            else DEPLOYMENT_LAYER_REPORT_SCHEMA_V1
        ),
        "layer": "temporal_validation",
        "role": "checkpoint_and_history_selection",
        "registered_contract_sha256": (
            REGISTERED_V4_CONTRACT_SHA256
            if is_v4
            else REGISTERED_CONTRACT_SHA256
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "evssm_checkpoint_sha256": EXPECTED_EVSSM_SHA256,
        "manifest_sha256": expected_temporal_manifest_sha256,
        "thresholds": DEPLOYMENT_THRESHOLDS["temporal_validation"],
        "oracle_good_definition": ORACLE_DEFINITION,
        "metrics": layer["metrics"],
        "eligible": bool(layer["passed"]),
        "evaluator_report": {
            "path": str(temporal_report_path),
            "sha256": sha256_file(temporal_report_path),
        },
        "evaluated_artifact": {
            "path": str(h3_artifact),
            "sha256": h3_artifact_sha256,
        },
        "history1_control": {
            "checkpoint_sha256": h1_checkpoint_sha256,
            "evaluator_report": str(history1_report_path),
            "evaluator_report_sha256": sha256_file(history1_report_path),
            "evaluated_artifact": str(h1_artifact),
            "evaluated_artifact_sha256": h1_artifact_sha256,
            "manifest_sha256": expected_temporal_manifest_sha256,
            "evssm_checkpoint_sha256": EXPECTED_EVSSM_SHA256,
            "history": 1,
            "h3_steady_psnr_db": layer["history1_comparison"][
                "h3_steady_psnr_db"
            ],
            "h1_steady_psnr_db": layer["history1_comparison"][
                "h1_steady_psnr_db"
            ],
            "steady_frame_keys_sha256": layer["history1_comparison"][
                "steady_frame_keys_sha256"
            ],
        },
        "source_manifest": manifest,
        "history": EXPECTED_HISTORY,
        "input_domain": EXPECTED_INPUT_DOMAIN,
        "checks": layer["checks"],
        "details": {
            "quality_steady_state": layer["quality_steady_state"],
            "quality_prefix": layer["quality_prefix"],
            "temporal_steady_state": layer["temporal_steady_state"],
            "gate_and_oracle_all_frames": layer["gate_and_oracle_all_frames"],
            "per_run": layer["per_run"],
        },
    }
    if is_v4:
        report.update(
            {
                "policy": DEPLOYMENT_SELECTION_POLICY_V4,
                "registered_contract": {
                    "schema": REGISTERED_V4_CONTRACT_SCHEMA,
                    "sha256": REGISTERED_V4_CONTRACT_SHA256,
                },
                "warm_start_checkpoint_sha256": (
                    REGISTERED_V4_WARM_START_SHA256
                ),
                "alignment_diagnostics_schema": (
                    ALIGNMENT_DIAGNOSTICS_SCHEMA_V4
                ),
                "alignment_integrity_passed": True,
                "v4_provenance": v4_provenance,
                "alignment_evidence": alignment_evidence,
                "room2_status": "not_opened_layer1_only",
                "tuning_after_room2_open": False,
            }
        )
    return report


def _validate_passed_temporal_layer(
    report_path: Path,
    expected_report_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    path = report_path.expanduser().resolve()
    digest = _sha256_digest(
        expected_report_sha256, "temporal layer report SHA-256"
    )
    if sha256_file(path) != digest:
        raise ReportContractError("temporal layer report SHA-256 mismatch")
    payload = _load_json(path, "temporal layer report")
    if payload.get("schema") == DEPLOYMENT_LAYER_REPORT_SCHEMA_V4:
        raise ReportContractError(
            "v4 room2 selection is not implemented; room2 was not opened"
        )
    checkpoint_sha256 = _sha256_digest(
        payload.get("checkpoint_sha256"), "temporal layer checkpoint_sha256"
    )
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        isinstance(item, dict) and item.get("passed") is True
        for item in checks.values()
    ):
        raise ReportContractError("temporal layer report has failed/incomplete checks")
    _resolve_layer_report(
        selection_report_path=path,
        layer_name="temporal_validation",
        layer_entry={
            "report": str(path),
            "report_sha256": digest,
            "manifest_sha256": expected_manifest_sha256,
        },
        expected_manifest_sha256=expected_manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        evssm_checkpoint_sha256=EXPECTED_EVSSM_SHA256,
    )
    return payload


def build_room2_selection_bundle(
    *,
    temporal_layer_report_path: Path,
    temporal_layer_report_sha256: str,
    room2_report_path: Path,
    room2_manifest_path: Path,
    source_root_path: Path,
    expected_room2_manifest_sha256: str,
    expected_temporal_manifest_sha256: str,
    expected_room2_frame_identity_sha256: str,
    expected_room2_frame_count: int,
) -> dict[str, dict[str, Any]]:
    """Open room2 only after a content-bound, exporter-validated Layer 1 pass."""

    temporal_layer_path = temporal_layer_report_path.expanduser().resolve()
    temporal_layer = _validate_passed_temporal_layer(
        temporal_layer_path,
        temporal_layer_report_sha256,
        expected_temporal_manifest_sha256,
    )
    temporal_layer_sha256 = _sha256_digest(
        temporal_layer_report_sha256, "temporal layer report SHA-256"
    )

    # No room2 path is opened above this line.
    room2_report_path = room2_report_path.expanduser().resolve()
    room2_payload = _load_json(room2_report_path, "room2 report")
    expected, lengths, manifest = _load_source_manifest(
        room2_manifest_path,
        expected_sha256=expected_room2_manifest_sha256,
        label="room2_test",
        source_root=source_root_path,
    )
    if manifest["frame_count"] != expected_room2_frame_count:
        raise ReportContractError("room2 source manifest frame count is not registered")
    if manifest["frame_identity_sha256"] != expected_room2_frame_identity_sha256:
        raise ReportContractError("room2 source frame identity is not registered")
    rows = _validate_report(
        room2_payload,
        report_path=room2_report_path,
        expected_rows=expected,
        sequence_lengths=lengths,
        role="room2_test",
        expected_history=EXPECTED_HISTORY,
    )
    artifact, artifact_sha256, checkpoint_sha256 = _validated_evaluator_artifact(
        room2_payload, room2_report_path, "room2_test"
    )
    if checkpoint_sha256 != temporal_layer["checkpoint_sha256"]:
        raise ReportContractError(
            "room2 and accepted temporal layer use different source checkpoints"
        )

    layer = _layer2(
        rows, lpips_computed=room2_payload.get("lpips_computed") is True
    )
    missing_metrics = list(layer.get("missing_metrics", []))
    eligible = bool(layer["passed"])
    if missing_metrics:
        status = "blocked_missing_metric"
    elif eligible:
        status = "eligible_for_export"
    else:
        status = "rejected_by_preregistered_thresholds"
    room2_layer_report = {
        "schema": DEPLOYMENT_LAYER_REPORT_SCHEMA_V1,
        "layer": "room2_one_shot",
        "role": "scene_disjoint_one_shot_test",
        "registered_contract_sha256": REGISTERED_CONTRACT_SHA256,
        "checkpoint_sha256": checkpoint_sha256,
        "evssm_checkpoint_sha256": EXPECTED_EVSSM_SHA256,
        "manifest_sha256": expected_room2_manifest_sha256,
        "thresholds": DEPLOYMENT_THRESHOLDS["room2_one_shot"],
        "oracle_good_definition": ORACLE_DEFINITION,
        "opened_after_temporal_validation_report_sha256": temporal_layer_sha256,
        "tuning_after_open": False,
        "metrics": layer["metrics"],
        "eligible": eligible,
        "evaluator_report": {
            "path": str(room2_report_path),
            "sha256": sha256_file(room2_report_path),
        },
        "source_manifest": manifest,
        "frame_identity": {
            "schema": FRAME_IDENTITY_SCHEMA,
            "sha256": manifest["frame_identity_sha256"],
            "frame_count": manifest["frame_count"],
        },
        "history": EXPECTED_HISTORY,
        "input_domain": EXPECTED_INPUT_DOMAIN,
        "lpips_required": True,
        "lpips_computed": room2_payload.get("lpips_computed") is True,
        "missing_metrics": missing_metrics,
        "checks": layer["checks"],
        "details": {
            "quality_all_frames": layer["quality_all_frames"],
            "quality_steady_state": layer["quality_steady_state"],
            "temporal_all_pairs": layer["temporal_all_pairs"],
            "gate_and_oracle_all_frames": layer["gate_and_oracle_all_frames"],
            "per_run": layer["per_run"],
        },
    }
    room2_layer_sha256 = hashlib.sha256(_json_bytes(room2_layer_report)).hexdigest()
    room2_layer_name = f"room2_one_shot_{room2_layer_sha256}.json"
    selection_report = {
        "schema": DEPLOYMENT_SELECTION_SCHEMA_V3,
        "policy": DEPLOYMENT_SELECTION_POLICY_V1,
        "thresholds": DEPLOYMENT_THRESHOLDS,
        "oracle_good_definition": ORACLE_DEFINITION,
        "registered_contract": {
            "schema": REGISTERED_CONTRACT_SCHEMA,
            "sha256": REGISTERED_CONTRACT_SHA256,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "evssm_checkpoint_sha256": EXPECTED_EVSSM_SHA256,
        "tum_used_for_selection": False,
        "layers": {
            "temporal_validation": {
                "report": str(temporal_layer_path),
                "report_sha256": temporal_layer_sha256,
                "manifest_sha256": expected_temporal_manifest_sha256,
            },
            "room2_one_shot": {
                "report": room2_layer_name,
                "report_sha256": room2_layer_sha256,
                "manifest_sha256": expected_room2_manifest_sha256,
            },
        },
        "eligible": eligible,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "room2_evaluated_artifact": {
            "path": str(artifact),
            "sha256": artifact_sha256,
        },
        "source_training_checkpoint_sha256": checkpoint_sha256,
        "teacher_provenance": room2_payload["teacher_provenance"],
        "missing_metrics": missing_metrics,
        "room2_use": "locked_one_shot_test_not_checkpoint_selection",
    }
    return {
        "selection_report": selection_report,
        "room2_layer_report": room2_layer_report,
    }


def write_temporal_layer_report(
    output: Path, report: dict[str, Any], *, overwrite: bool
) -> Path:
    destination = output.expanduser().resolve()
    protected = {
        Path(str(report["evaluator_report"]["path"])).expanduser().resolve(),
        Path(str(report["history1_control"]["evaluator_report"]))
        .expanduser()
        .resolve(),
        Path(str(report["source_manifest"]["path"])).expanduser().resolve(),
        Path(str(report["evaluated_artifact"]["path"])).expanduser().resolve(),
        Path(str(report["history1_control"]["evaluated_artifact"]))
        .expanduser()
        .resolve(),
    }
    if destination in protected:
        raise ReportContractError("refusing to overwrite a Layer1 evaluator/source input")
    _atomic_json(destination, report, overwrite=overwrite)
    return destination


def write_room2_selection_bundle(
    output: Path,
    bundle: dict[str, dict[str, Any]],
    *,
    overwrite: bool,
) -> dict[str, Path]:
    destination = output.expanduser().resolve()
    selection = bundle["selection_report"]
    room2_name = str(selection["layers"]["room2_one_shot"]["report"])
    room2_path = destination.parent / room2_name
    temporal_path = Path(
        str(selection["layers"]["temporal_validation"]["report"])
    ).expanduser().resolve()
    protected = {
        temporal_path,
        Path(str(bundle["room2_layer_report"]["evaluator_report"]["path"]))
        .expanduser()
        .resolve(),
        Path(str(bundle["room2_layer_report"]["source_manifest"]["path"]))
        .expanduser()
        .resolve(),
        Path(str(selection["room2_evaluated_artifact"]["path"]))
        .expanduser()
        .resolve(),
    }
    if destination == room2_path or {destination, room2_path} & protected:
        raise ReportContractError("refusing to overwrite a Layer2 evaluator/source input")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite selection report: {destination}")
    expected_room2_sha256 = hashlib.sha256(
        _json_bytes(bundle["room2_layer_report"])
    ).hexdigest()
    if expected_room2_sha256 != selection["layers"]["room2_one_shot"][
        "report_sha256"
    ]:
        raise ReportContractError("room2 layer digest changed after construction")
    if room2_path.exists():
        if sha256_file(room2_path) != expected_room2_sha256:
            raise FileExistsError(
                f"content-addressed room2 layer report collision: {room2_path}"
            )
    else:
        _atomic_json(room2_path, bundle["room2_layer_report"], overwrite=False)
    _atomic_json(destination, selection, overwrite=overwrite)
    return {"selection_report": destination, "room2_layer_report": room2_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("temporal", "room2"), required=True)
    parser.add_argument("--temporal-val-report", type=Path)
    parser.add_argument(
        "--history1-temporal-val-report",
        type=Path,
        help="H=1 spatial-control evaluator report on the identical temporal split",
    )
    parser.add_argument(
        "--temporal-layer-report",
        type=Path,
        help="accepted Layer1 report; required and opened before room2",
    )
    parser.add_argument(
        "--temporal-layer-report-sha256",
        help="explicit SHA-256 of the accepted Layer1 report",
    )
    parser.add_argument("--room2-report", type=Path)
    parser.add_argument(
        "--temporal-val-source-manifest",
        type=Path,
        default=DEFAULT_TEMPORAL_MANIFEST,
    )
    parser.add_argument(
        "--room2-source-manifest", type=Path, default=DEFAULT_ROOM2_MANIFEST
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="root used to resolve relative blurry/sharp paths in pinned manifests",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="exit 2 after writing the report when the candidate is not eligible",
    )
    return parser.parse_args()


def _cli_option_present(name: str) -> bool:
    return any(
        argument == name or argument.startswith(f"{name}=")
        for argument in sys.argv[1:]
    )


def main() -> None:
    args = parse_args()
    if args.stage == "temporal":
        if (
            args.room2_report is not None
            or args.temporal_layer_report is not None
            or args.temporal_layer_report_sha256 is not None
        ):
            raise ValueError("--stage temporal does not accept any room2-stage input")
        if args.temporal_val_report is None or args.history1_temporal_val_report is None:
            raise ValueError(
                "--stage temporal requires --temporal-val-report and "
                "--history1-temporal-val-report"
            )
        temporal_schema = _load_json(
            args.temporal_val_report, "temporal-val report"
        ).get("schema")
        if temporal_schema == EVALUATOR_SCHEMA_V4 and _cli_option_present(
            "--room2-source-manifest"
        ):
            raise ValueError(
                "v4 --stage temporal does not accept any room2 input"
            )
        report = build_temporal_layer_report(
            temporal_report_path=args.temporal_val_report,
            history1_report_path=args.history1_temporal_val_report,
            temporal_manifest_path=args.temporal_val_source_manifest,
            source_root_path=args.source_root,
            expected_temporal_manifest_sha256=CANONICAL_SPLIT_SHA256[
                "temporal_val"
            ],
        )
        output = write_temporal_layer_report(
            args.output, report, overwrite=args.overwrite
        )
        digest = sha256_file(output)
        print(
            json.dumps(
                {
                    "stage": "temporal",
                    "eligible": report["eligible"],
                    "output": str(output),
                    "report_sha256": digest,
                },
                sort_keys=True,
            )
        )
        if args.require_pass and not report["eligible"]:
            raise SystemExit(2)
        return

    if args.temporal_val_report is not None or args.history1_temporal_val_report is not None:
        raise ValueError("--stage room2 does not accept raw temporal evaluator reports")
    if (
        args.temporal_layer_report is None
        or args.temporal_layer_report_sha256 is None
        or args.room2_report is None
    ):
        raise ValueError(
            "--stage room2 requires --temporal-layer-report, "
            "--temporal-layer-report-sha256, and --room2-report"
        )
    bundle = build_room2_selection_bundle(
        temporal_layer_report_path=args.temporal_layer_report,
        temporal_layer_report_sha256=args.temporal_layer_report_sha256,
        room2_report_path=args.room2_report,
        room2_manifest_path=args.room2_source_manifest,
        source_root_path=args.source_root,
        expected_temporal_manifest_sha256=CANONICAL_SPLIT_SHA256[
            "temporal_val"
        ],
        expected_room2_manifest_sha256=CANONICAL_SPLIT_SHA256["room2_test"],
        expected_room2_frame_identity_sha256=(
            ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256
        ),
        expected_room2_frame_count=ROOM2_ONE_SHOT_FRAME_COUNT,
    )
    paths = write_room2_selection_bundle(
        args.output, bundle, overwrite=args.overwrite
    )
    report = bundle["selection_report"]
    print(
        json.dumps(
            {
                "stage": "room2",
                "status": report["status"],
                "eligible": report["eligible"],
                "output": str(paths["selection_report"]),
                "room2_layer_report": str(paths["room2_layer_report"]),
            },
            sort_keys=True,
        )
    )
    if args.require_pass and not report["eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
