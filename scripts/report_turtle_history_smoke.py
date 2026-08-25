#!/usr/bin/env python3
"""Audit the preregistered official-TURTLE Replica history smoke on CPU.

This script consumes the immutable JSON produced by
``evaluate_turtle_streaming.py`` for the official GoPro base checkpoint and
the fixed-terminal fine-tuned checkpoint.  It does not import or run TURTLE,
and it deliberately reads only the validation manifest named by the
preregistered contract (never the Room2 holdout).

Metric-gate failures are valid experimental outcomes: they produce an
``eligible=false`` report.  Missing, malformed, or provenance-mismatched
artifacts instead raise :class:`HistorySmokeContractError` and no report is
written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


CONTRACT_SCHEMA = "unblur_slam.turtle_gopro_replica424_history_smoke.v1"
EVALUATION_SCHEMA = "unblur_slam.turtle_streaming_evaluation.v1"
REPORT_SCHEMA = "unblur_slam.turtle_gopro_replica424_history_smoke_report.v1"
FINETUNED_CHECKPOINT_FORMAT = "unblur_slam.turtle_streaming.checkpoint.v1"
OFFICIAL_CACHE_CONTRACT = "official_kv_8_incremental"

NORMAL = "turtle"
RESET = "turtle_reset_cache"
REPEAT = "turtle_repeat_current"
ORDERED = "turtle_replayed_ordered"
SHUFFLED = "turtle_shuffled_history"
REQUIRED_METRIC_SOURCES = ("raw", NORMAL, RESET, REPEAT, ORDERED, SHUFFLED)
CONTROL_SOURCES = (RESET, REPEAT, SHUFFLED)
IMAGE_METRICS = ("psnr", "ssim", "l1")
TEMPORAL_METRICS = (
    "adjacent_change_l1",
    "gt_temporal_difference_error_l1",
)
EXPECTED_GATE_KEYS = {
    "ordered_replay_max_abs_max",
    "finetuned_normal_minus_base_normal_psnr_db_min",
    "finetuned_normal_minus_base_normal_ssim_min",
    "finetuned_normal_minus_reset_cache_psnr_db_min",
    "finetuned_normal_minus_repeat_current_psnr_db_min",
    "finetuned_normal_minus_shuffled_history_psnr_db_min",
    "history_gain_interaction_vs_base_reset_psnr_db_min",
    "finetuned_normal_vs_reset_temporal_error_relative_max",
    "both_validation_sequences_psnr_direction_positive",
}
IMPLEMENTATION_PIN_PATHS = {
    "train_script_sha256": "scripts/train_turtle_streaming.py",
    "evaluate_script_sha256": "scripts/evaluate_turtle_streaming.py",
    "test_sha256": "tests/test_turtle_training_contract.py",
    "backend_sha256": "src/turtle_backend.py",
    "report_script_sha256": "scripts/report_turtle_history_smoke.py",
    "report_test_sha256": "tests/test_report_turtle_history_smoke.py",
}
ROOT = Path(__file__).resolve().parents[1]


class HistorySmokeContractError(ValueError):
    """An input artifact cannot be trusted under the preregistered protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HistorySmokeContractError(message)


def sha256_file(path: Path | str) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path | str, label: str) -> tuple[dict[str, Any], Path, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} does not exist: {source}")
    digest = sha256_file(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HistorySmokeContractError(f"{label} is not valid JSON: {source}") from error
    _require(isinstance(payload, dict), f"{label} JSON root must be an object")
    return payload, source, digest


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be an array")
    return value


def _integer(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    return int(value)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    number = float(value)
    _require(math.isfinite(number), f"{label} must be finite")
    return number


def _sha(value: Any, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a SHA-256 string")
    normalized = value.lower()
    _require(
        len(normalized) == 64 and all(character in "0123456789abcdef" for character in normalized),
        f"{label} must be a lowercase-compatible 64-digit SHA-256",
    )
    return normalized


def _resolved(value: Any, label: str) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty path")
    return Path(value).expanduser().resolve()


def _close(actual: float, expected: float, label: str, tolerance: float = 1.0e-10) -> None:
    _require(
        math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
        f"{label} aggregate mismatch: {actual} != {expected}",
    )


def _mean(values: Iterable[float], label: str) -> float:
    collected = [float(value) for value in values]
    _require(bool(collected), f"{label} has no values")
    _require(all(math.isfinite(value) for value in collected), f"{label} contains a non-finite value")
    return math.fsum(collected) / len(collected)


def _resolve_manifest_path(value: Any, root: Path, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def _load_validation_manifest(
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = _mapping(contract.get("data"), "contract.data")
    root = _resolved(data.get("root"), "contract.data.root")
    manifest = _resolved(
        data.get("validation_manifest"), "contract.data.validation_manifest"
    )
    expected_sha = _sha(
        data.get("validation_manifest_sha256"),
        "contract.data.validation_manifest_sha256",
    )
    actual_sha = sha256_file(manifest)
    _require(actual_sha == expected_sha, "validation manifest SHA-256 differs from preregistration")

    expected: list[dict[str, Any]] = []
    names: set[str] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise HistorySmokeContractError(
                    f"validation manifest line {line_number} is invalid JSON"
                ) from error
            record = _mapping(record, f"validation manifest line {line_number}")
            name_value = record.get("sequence", record.get("name"))
            _require(isinstance(name_value, str) and bool(name_value), "validation sequence name is missing")
            name = str(name_value)
            _require(name not in names, f"duplicate validation sequence {name!r}")
            names.add(name)
            if "frames" in record:
                frame_records = _list(record["frames"], f"validation {name}.frames")
                blurry_values = []
                sharp_values = []
                for frame_index, frame_value in enumerate(frame_records):
                    frame = _mapping(frame_value, f"validation {name}.frames[{frame_index}]")
                    blurry_values.append(
                        next((frame[key] for key in ("blurry", "blur", "input", "lq") if key in frame), None)
                    )
                    sharp_values.append(
                        next((frame[key] for key in ("sharp", "target", "gt") if key in frame), None)
                    )
            else:
                blurry_values = record.get("blurry", record.get("blur", record.get("input", record.get("lq"))))
                sharp_values = record.get("sharp", record.get("target", record.get("gt")))
            _require(isinstance(blurry_values, list) and bool(blurry_values), f"validation {name} blurry frames are invalid")
            _require(
                isinstance(sharp_values, list) and len(sharp_values) == len(blurry_values),
                f"validation {name} sharp frames are invalid",
            )
            for frame_index, (blurry, sharp) in enumerate(zip(blurry_values, sharp_values)):
                expected.append(
                    {
                        "sequence": name,
                        "frame_index": frame_index,
                        "global_index": len(expected),
                        "raw_path": _resolve_manifest_path(blurry, root, f"validation {name} blurry"),
                        "gt_path": _resolve_manifest_path(sharp, root, f"validation {name} sharp"),
                    }
                )
    _require(bool(expected), "validation manifest is empty")

    inventory = _mapping(data.get("validation_inventory"), "contract.data.validation_inventory")
    lengths = {name: sum(row["sequence"] == name for row in expected) for name in names}
    temporal_pairs = sum(max(0, length - 1) for length in lengths.values())
    steady_min = _integer(
        inventory.get("steady_frame_index_min"),
        "contract.data.validation_inventory.steady_frame_index_min",
    )
    steady_count = sum(row["frame_index"] >= steady_min for row in expected)
    _require(_integer(inventory.get("sequences"), "validation_inventory.sequences") == len(names), "validation sequence inventory mismatch")
    _require(_integer(inventory.get("frames"), "validation_inventory.frames") == len(expected), "validation frame inventory mismatch")
    _require(_integer(inventory.get("real_transitions"), "validation_inventory.real_transitions") == temporal_pairs, "validation transition inventory mismatch")
    _require(_integer(inventory.get("steady_frames"), "validation_inventory.steady_frames") == steady_count, "validation steady-frame inventory mismatch")
    return expected, {
        "path": str(manifest),
        "sha256": actual_sha,
        "data_root": str(root),
        "sequence_lengths": dict(sorted(lengths.items())),
        "frame_count": len(expected),
        "sequence_count": len(names),
        "temporal_pair_count": temporal_pairs,
        "steady_frame_index_min": steady_min,
        "steady_frame_count": steady_count,
    }


def _verify_implementation_pins(contract: Mapping[str, Any]) -> dict[str, Any]:
    pins = _mapping(contract.get("implementation_pins"), "contract.implementation_pins")
    _require(set(pins) == set(IMPLEMENTATION_PIN_PATHS), "implementation pin names changed")
    verified: dict[str, Any] = {}
    for key, relative in IMPLEMENTATION_PIN_PATHS.items():
        expected = _sha(pins.get(key), f"contract.implementation_pins.{key}")
        path = (ROOT / relative).resolve()
        actual = sha256_file(path)
        _require(actual == expected, f"implementation pin mismatch for {relative}")
        verified[key] = {"path": str(path), "sha256": actual}
    return verified


def _expect_equal(mapping: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    _require(mapping.get(key) == expected, f"{label}.{key} != preregistered value")


def _verify_checkpoint_metadata(
    report: Mapping[str, Any], contract: Mapping[str, Any], role: str
) -> dict[str, Any]:
    model = _mapping(contract.get("model"), "contract.model")
    metadata = _mapping(report.get("checkpoint_metadata"), f"{role}.checkpoint_metadata")
    provenance = _mapping(report.get("provenance"), f"{role}.provenance")
    checkpoint_path = _resolved(provenance.get("checkpoint"), f"{role}.provenance.checkpoint")
    checkpoint_sha = _sha(provenance.get("checkpoint_sha256"), f"{role}.provenance.checkpoint_sha256")
    _require(sha256_file(checkpoint_path) == checkpoint_sha, f"{role} checkpoint bytes do not match provenance")
    _expect_equal(metadata, "checkpoint_sha256", checkpoint_sha, f"{role}.checkpoint_metadata")
    for metadata_key, model_key in (
        ("base_checkpoint_sha256", "base_checkpoint_sha256"),
        ("turtle_repo_commit", "repo_commit"),
        ("turtle_arch_sha256", "architecture_sha256"),
        ("turtle_config_sha256", "config_sha256"),
    ):
        _expect_equal(metadata, metadata_key, model.get(model_key), f"{role}.checkpoint_metadata")
    _expect_equal(metadata, "input_domain", "raw", f"{role}.checkpoint_metadata")
    _expect_equal(metadata, "cache_contract", OFFICIAL_CACHE_CONTRACT, f"{role}.checkpoint_metadata")

    base_sha = _sha(model.get("base_checkpoint_sha256"), "contract.model.base_checkpoint_sha256")
    if role == "base":
        _require(checkpoint_path == _resolved(model.get("base_checkpoint"), "contract.model.base_checkpoint"), "base checkpoint path differs from preregistration")
        _require(checkpoint_sha == base_sha, "base evaluation did not use official GoPro checkpoint")
        _expect_equal(metadata, "kind", "official_gopro", "base.checkpoint_metadata")
        _expect_equal(metadata, "format", "official_turtle.params", "base.checkpoint_metadata")
    else:
        _require(checkpoint_sha != base_sha, "fine-tuned evaluation reused the base checkpoint")
        _expect_equal(metadata, "kind", "finetuned", "finetuned.checkpoint_metadata")
        _expect_equal(metadata, "format", FINETUNED_CHECKPOINT_FORMAT, "finetuned.checkpoint_metadata")
        output_root = _resolved(contract.get("output_root"), "contract.output_root")
        _require(checkpoint_path.parent == output_root, "fine-tuned checkpoint is outside preregistered output root")
        _require(checkpoint_path.name == "finetuned_final.pth", "fine-tuned checkpoint is not the fixed-terminal artifact")
        _verify_training_metadata(metadata, contract)
    return {"path": str(checkpoint_path), "sha256": checkpoint_sha, "metadata": dict(metadata)}


def _verify_training_metadata(metadata: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    training_contract = _mapping(contract.get("training"), "contract.training")
    data = _mapping(contract.get("data"), "contract.data")
    for key, expected in (
        ("uses_gt", False),
        ("uses_gt_pose", False),
        ("uses_gt_depth", False),
        ("uses_sharp_rgb_supervision", True),
    ):
        _expect_equal(metadata, key, expected, "finetuned.metadata")
    manifests = _mapping(metadata.get("manifests"), "finetuned.metadata.manifests")
    _require(
        _resolved(manifests.get("train"), "finetuned.metadata.manifests.train")
        == _resolved(data.get("train_manifest"), "contract.data.train_manifest"),
        "fine-tuned train manifest path differs from preregistration",
    )
    _expect_equal(manifests, "train_sha256", data.get("train_manifest_sha256"), "finetuned.metadata.manifests")
    _expect_equal(manifests, "validation", None, "finetuned.metadata.manifests")
    _expect_equal(manifests, "validation_sha256", None, "finetuned.metadata.manifests")

    training = _mapping(metadata.get("training"), "finetuned.metadata.training")
    _expect_equal(training, "seed", training_contract.get("seed"), "finetuned.metadata.training")
    _expect_equal(training, "steps", training_contract.get("fixed_terminal_optimizer_steps"), "finetuned.metadata.training")
    optimizer = _mapping(metadata.get("optimizer"), "finetuned.metadata.optimizer")
    expected_optimizer = _mapping(training_contract.get("optimizer"), "contract.training.optimizer")
    for key in ("name", "learning_rate", "weight_decay", "betas", "scheduler"):
        _expect_equal(optimizer, key, expected_optimizer.get(key), "finetuned.metadata.optimizer")
    _expect_equal(
        optimizer,
        "scheduler_eta_min",
        expected_optimizer.get("eta_min"),
        "finetuned.metadata.optimizer",
    )
    _expect_equal(
        optimizer,
        "scheduler_t_max_optimizer_steps",
        training_contract.get("fixed_terminal_optimizer_steps"),
        "finetuned.metadata.optimizer",
    )
    _expect_equal(
        optimizer,
        "gradient_clip_norm",
        expected_optimizer.get("gradient_clip"),
        "finetuned.metadata.optimizer",
    )
    loss = _mapping(metadata.get("loss"), "finetuned.metadata.loss")
    objective = _mapping(training_contract.get("objective"), "contract.training.objective")
    _expect_equal(loss, "name", "l1_plus_fft_l1", "finetuned.metadata.loss")
    _expect_equal(loss, "l1_weight", objective.get("l1_weight"), "finetuned.metadata.loss")
    _expect_equal(loss, "fft_weight", objective.get("fft_l1_weight"), "finetuned.metadata.loss")
    history = _mapping(metadata.get("history"), "finetuned.metadata.history")
    _expect_equal(history, "mode", "official_incremental_kv", "finetuned.metadata.history")
    _expect_equal(history, "sequence_boundary", "hard_reset", "finetuned.metadata.history")
    _expect_equal(history, "backpropagation", "full_sequence", "finetuned.metadata.history")
    scope = _mapping(metadata.get("trainable_scope"), "finetuned.metadata.trainable_scope")
    _expect_equal(scope, "scope", "history_attention", "finetuned.metadata.trainable_scope")
    _expect_equal(scope, "parameter_tensors", training_contract.get("trainable_parameter_tensors"), "finetuned.metadata.trainable_scope")
    _expect_equal(scope, "parameter_count", training_contract.get("trainable_parameters"), "finetuned.metadata.trainable_scope")
    sequence_filter = _mapping(metadata.get("sequence_filter"), "finetuned.metadata.sequence_filter")
    _expect_equal(sequence_filter, "minimum_length", training_contract.get("minimum_sequence_length"), "finetuned.metadata.sequence_filter")
    _expect_equal(sequence_filter, "loss_start_frame", training_contract.get("loss_start_frame"), "finetuned.metadata.sequence_filter")
    augmentation = _mapping(metadata.get("augmentation"), "finetuned.metadata.augmentation")
    _expect_equal(augmentation, "enabled", True, "finetuned.metadata.augmentation")
    _expect_equal(augmentation, "shared_across_sequence_and_modalities", True, "finetuned.metadata.augmentation")
    _expect_equal(augmentation, "crop_size", training_contract.get("crop_size"), "finetuned.metadata.augmentation")
    for key in ("horizontal_flip", "vertical_flip", "quarter_turn_rotation"):
        _expect_equal(augmentation, key, True, "finetuned.metadata.augmentation")
    _expect_equal(
        training_contract,
        "augmentation",
        "one_shared_random_crop_flip_rotation_per_sequence",
        "contract.training",
    )
    _expect_equal(
        augmentation,
        "sampling_policy",
        "shared_per_record; hflip_p=0.5; vflip_p=0.5; quarter_turn_uniform_0_1_2_3",
        "finetuned.metadata.augmentation",
    )
    _expect_equal(training, "amp", expected_optimizer.get("amp"), "finetuned.metadata.training")


def _metric_value(row: Mapping[str, Any], source: str, metric: str, label: str) -> float:
    metrics = _mapping(row.get("metrics"), f"{label}.metrics")
    source_metrics = _mapping(metrics.get(source), f"{label}.metrics.{source}")
    return _finite(source_metrics.get(metric), f"{label}.metrics.{source}.{metric}")


def _temporal_value(row: Mapping[str, Any], source: str, metric: str, label: str) -> float:
    temporal = _mapping(row.get("temporal"), f"{label}.temporal")
    source_metrics = _mapping(temporal.get(source), f"{label}.temporal.{source}")
    return _finite(source_metrics.get(metric), f"{label}.temporal.{source}.{metric}")


def _aggregate(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    _require(bool(rows), f"{label} has no rows")
    source_names = tuple(sorted(_mapping(rows[0].get("metrics"), f"{label}[0].metrics")))
    for index, row in enumerate(rows):
        metrics = _mapping(row.get("metrics"), f"{label}[{index}].metrics")
        _require(tuple(sorted(metrics)) == source_names, f"{label} metric sources vary by frame")
    image = {
        source: {
            metric: _mean(
                (_metric_value(row, source, metric, f"{label}[{index}]") for index, row in enumerate(rows)),
                f"{label}.{source}.{metric}",
            )
            for metric in IMAGE_METRICS
        }
        for source in source_names
    }
    temporal_rows = [row for row in rows if row.get("temporal") is not None]
    temporal = {
        source: {
            metric: _mean(
                (_temporal_value(row, source, metric, f"{label}.temporal[{index}]") for index, row in enumerate(temporal_rows)),
                f"{label}.temporal.{source}.{metric}",
            )
            for metric in TEMPORAL_METRICS
        }
        for source in source_names
    }
    return {"frame_count": len(rows), "temporal_pair_count": len(temporal_rows), "mean": image, "temporal_mean": temporal}


def _validate_evaluation(
    payload: Mapping[str, Any],
    *,
    role: str,
    expected_rows: Sequence[Mapping[str, Any]],
    manifest_metadata: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    _require(payload.get("schema") == EVALUATION_SCHEMA, f"{role} evaluation schema mismatch")
    provenance = _mapping(payload.get("provenance"), f"{role}.provenance")
    _require(_resolved(provenance.get("manifest"), f"{role}.provenance.manifest") == Path(manifest_metadata["path"]), f"{role} manifest path mismatch")
    _expect_equal(provenance, "manifest_sha256", manifest_metadata["sha256"], f"{role}.provenance")
    _require(isinstance(provenance.get("device"), str) and bool(provenance.get("device")), f"{role} provenance device is missing")

    rows = _list(payload.get("frames"), f"{role}.frames")
    _require(len(rows) == len(expected_rows), f"{role} frame list length mismatch")
    _require(_integer(payload.get("frame_count"), f"{role}.frame_count") == len(rows), f"{role} frame_count mismatch")
    _require(_integer(payload.get("sequence_count"), f"{role}.sequence_count") == manifest_metadata["sequence_count"], f"{role} sequence_count mismatch")
    _require(_integer(payload.get("temporal_pair_count"), f"{role}.temporal_pair_count") == manifest_metadata["temporal_pair_count"], f"{role} temporal_pair_count mismatch")

    declared_sources = _list(payload.get("sources"), f"{role}.sources")
    _require(len(declared_sources) == len(set(declared_sources)), f"{role}.sources contains duplicates")
    _require(set(REQUIRED_METRIC_SOURCES).issubset(declared_sources), f"{role} history-control sources are incomplete")
    _require("gt" in declared_sources, f"{role}.sources omits GT")

    checked_rows: list[dict[str, Any]] = []
    for index, (raw_row, expected) in enumerate(zip(rows, expected_rows)):
        row = _mapping(raw_row, f"{role}.frames[{index}]")
        for key in ("sequence", "frame_index", "global_index"):
            _expect_equal(row, key, expected[key], f"{role}.frames[{index}]")
        _require(_resolved(row.get("raw_path"), f"{role}.frames[{index}].raw_path") == Path(expected["raw_path"]), f"{role} raw path mismatch at row {index}")
        _require(_resolved(row.get("gt_path"), f"{role}.frames[{index}].gt_path") == Path(expected["gt_path"]), f"{role} GT path mismatch at row {index}")
        metrics = _mapping(row.get("metrics"), f"{role}.frames[{index}].metrics")
        _require(set(metrics) == set(declared_sources) - {"gt"}, f"{role} row {index} metric sources differ from declaration")
        for source in metrics:
            for metric in IMAGE_METRICS:
                _metric_value(row, source, metric, f"{role}.frames[{index}]")
        frame_index = int(expected["frame_index"])
        if frame_index == 0:
            _require(row.get("temporal") is None, f"{role} first frame has cross-sequence temporal metrics")
        else:
            temporal = _mapping(row.get("temporal"), f"{role}.frames[{index}].temporal")
            _require(set(temporal) == set(metrics), f"{role} row {index} temporal sources differ from image sources")
            for source in temporal:
                for metric in TEMPORAL_METRICS:
                    _temporal_value(row, source, metric, f"{role}.frames[{index}]")
        checked_rows.append(dict(row))

    history = _mapping(payload.get("history_ablation"), f"{role}.history_ablation")
    protocol = _mapping(history.get("protocol"), f"{role}.history_ablation.protocol")
    expected_state = _mapping(_mapping(contract.get("model"), "contract.model").get("causal_state"), "contract.model.causal_state")
    _expect_equal(protocol, "cache_source", "official TURTLE K/V only; use_both_input=false", f"{role}.history_ablation.protocol")
    _expect_equal(protocol, "cache_slots_per_kind", expected_state.get("cache_slots_per_kind"), f"{role}.history_ablation.protocol")
    _expect_equal(protocol, "populated_cache_slots_per_kind", expected_state.get("populated_slots_per_kind"), f"{role}.history_ablation.protocol")
    _expect_equal(
        protocol,
        "direct_cache_capacity_frames",
        expected_state.get("direct_cache_capacity_frames"),
        f"{role}.history_ablation.protocol",
    )
    _expect_equal(protocol, "effective_history", "recurrent_full_prefix", f"{role}.history_ablation.protocol")
    _expect_equal(protocol, "ordered_control", "reset_then_replay_complete_past_prefix", f"{role}.history_ablation.protocol")
    _expect_equal(protocol, "repeat_current_control", "reset_then_replay_current_once_per_complete_past_frame", f"{role}.history_ablation.protocol")
    _expect_equal(protocol, "shuffled_control", "cyclic_left_shift_of_complete_past_prefix", f"{role}.history_ablation.protocol")
    _expect_equal(protocol, "steady_frame_index_min", manifest_metadata["steady_frame_index_min"], f"{role}.history_ablation.protocol")
    _expect_equal(protocol, "future_frames_used", False, f"{role}.history_ablation.protocol")
    ordered_max = _finite(history.get("ordered_replay_max_abs"), f"{role}.history_ablation.ordered_replay_max_abs")
    tolerance = _finite(
        _mapping(_mapping(contract.get("evaluation"), "contract.evaluation").get("eligibility_gates"), "contract.evaluation.eligibility_gates").get("ordered_replay_max_abs_max"),
        "ordered replay tolerance",
    )
    _expect_equal(history, "ordered_replay_matches_stream", ordered_max <= tolerance, f"{role}.history_ablation")
    steady_rows = [row for row in checked_rows if int(row["frame_index"]) >= manifest_metadata["steady_frame_index_min"]]
    _require(_integer(history.get("steady_frame_count"), f"{role}.history_ablation.steady_frame_count") == len(steady_rows), f"{role} steady frame count mismatch")

    all_aggregate = _aggregate(checked_rows, f"{role}.all")
    steady_aggregate = _aggregate(steady_rows, f"{role}.steady")
    reported_mean = _mapping(payload.get("mean"), f"{role}.mean")
    for source, metrics in all_aggregate["mean"].items():
        reported = _mapping(reported_mean.get(source), f"{role}.mean.{source}")
        for metric, value in metrics.items():
            _close(_finite(reported.get(metric), f"{role}.mean.{source}.{metric}"), value, f"{role}.mean.{source}.{metric}")
    reported_temporal = _mapping(_mapping(payload.get("temporal"), f"{role}.temporal").get("mean"), f"{role}.temporal.mean")
    for source, metrics in all_aggregate["temporal_mean"].items():
        reported = _mapping(reported_temporal.get(source), f"{role}.temporal.mean.{source}")
        for metric, value in metrics.items():
            _close(_finite(reported.get(metric), f"{role}.temporal.mean.{source}.{metric}"), value, f"{role}.temporal.mean.{source}.{metric}")
    reported_steady = _mapping(history.get("steady_mean"), f"{role}.history_ablation.steady_mean")
    for source in (NORMAL, RESET, REPEAT, ORDERED, SHUFFLED):
        reported = _mapping(reported_steady.get(source), f"{role}.history_ablation.steady_mean.{source}")
        for metric, value in steady_aggregate["mean"][source].items():
            _close(_finite(reported.get(metric), f"{role}.history_ablation.steady_mean.{source}.{metric}"), value, f"{role}.history_ablation.steady_mean.{source}.{metric}")
    reported_deltas = _mapping(
        history.get("steady_normal_minus_control"),
        f"{role}.history_ablation.steady_normal_minus_control",
    )
    computed_deltas = _normal_minus_control(steady_aggregate)
    for control in CONTROL_SOURCES:
        reported = _mapping(
            reported_deltas.get(control),
            f"{role}.history_ablation.steady_normal_minus_control.{control}",
        )
        for metric, value in computed_deltas[control].items():
            _close(
                _finite(
                    reported.get(metric),
                    f"{role}.history_ablation.steady_normal_minus_control.{control}.{metric}",
                ),
                value,
                f"{role}.history_ablation.steady_normal_minus_control.{control}.{metric}",
            )

    sequences = sorted({str(row["sequence"]) for row in checked_rows})
    per_sequence = {}
    for sequence in sequences:
        sequence_rows = [row for row in checked_rows if row["sequence"] == sequence]
        sequence_steady = [row for row in sequence_rows if int(row["frame_index"]) >= manifest_metadata["steady_frame_index_min"]]
        _require(bool(sequence_steady), f"{role} sequence {sequence!r} has no steady frames")
        per_sequence[sequence] = {
            "all": _aggregate(sequence_rows, f"{role}.{sequence}.all"),
            "steady": _aggregate(sequence_steady, f"{role}.{sequence}.steady"),
        }
    checkpoint = _verify_checkpoint_metadata(payload, contract, role)
    return {
        "rows": checked_rows,
        "ordered_replay_max_abs": ordered_max,
        "all": all_aggregate,
        "steady": steady_aggregate,
        "per_sequence": per_sequence,
        "checkpoint": checkpoint,
        "device": provenance["device"],
    }


def _normal_minus_control(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    means = _mapping(aggregate.get("mean"), "aggregate.mean")
    normal = _mapping(means.get(NORMAL), "aggregate.mean.turtle")
    result = {}
    for control in CONTROL_SOURCES:
        controlled = _mapping(means.get(control), f"aggregate.mean.{control}")
        result[control] = {
            metric: float(normal[metric]) - float(controlled[metric])
            for metric in IMAGE_METRICS
        }
    return result


def _comparison(base: Mapping[str, Any], fine: Mapping[str, Any]) -> dict[str, Any]:
    base_mean = _mapping(base.get("mean"), "base.mean")
    fine_mean = _mapping(fine.get("mean"), "finetuned.mean")
    base_control = _normal_minus_control(base)
    fine_control = _normal_minus_control(fine)
    model_delta = {
        metric: float(_mapping(fine_mean[NORMAL], "finetuned.mean.turtle")[metric])
        - float(_mapping(base_mean[NORMAL], "base.mean.turtle")[metric])
        for metric in IMAGE_METRICS
    }
    base_temporal = float(
        _mapping(_mapping(base.get("temporal_mean"), "base.temporal_mean")[NORMAL], "base.temporal_mean.turtle")["gt_temporal_difference_error_l1"]
    )
    fine_temporal = float(
        _mapping(_mapping(fine.get("temporal_mean"), "fine.temporal_mean")[NORMAL], "fine.temporal_mean.turtle")["gt_temporal_difference_error_l1"]
    )
    fine_reset_temporal = float(
        _mapping(_mapping(fine.get("temporal_mean"), "fine.temporal_mean")[RESET], "fine.temporal_mean.reset")["gt_temporal_difference_error_l1"]
    )
    _require(fine_reset_temporal > 0.0, "fine-tuned reset temporal error must be positive for relative comparison")
    return {
        "finetuned_normal_minus_base_normal": model_delta,
        "base_normal_minus_control": base_control,
        "finetuned_normal_minus_control": fine_control,
        "history_gain_interaction_vs_base_reset_psnr_db": fine_control[RESET]["psnr"] - base_control[RESET]["psnr"],
        "finetuned_normal_vs_reset_temporal_difference_error_l1": {
            "normal": fine_temporal,
            "reset": fine_reset_temporal,
            "absolute_difference": fine_temporal - fine_reset_temporal,
            "relative_difference": (fine_temporal - fine_reset_temporal) / fine_reset_temporal,
        },
        "base_to_finetuned_normal_temporal_difference_error_l1": fine_temporal - base_temporal,
    }


def _gate(value: Any, threshold: Any, operator: str, passed: bool) -> dict[str, Any]:
    return {"value": value, "threshold": threshold, "operator": operator, "passed": bool(passed)}


def _build_gates(
    contract: Mapping[str, Any],
    base: Mapping[str, Any],
    fine: Mapping[str, Any],
    steady_comparison: Mapping[str, Any],
    per_sequence: Mapping[str, Any],
) -> dict[str, Any]:
    configured = _mapping(_mapping(contract.get("evaluation"), "contract.evaluation").get("eligibility_gates"), "contract.evaluation.eligibility_gates")
    _require(set(configured) == EXPECTED_GATE_KEYS, "eligibility gate set differs from supported preregistration")
    gates: dict[str, Any] = {}
    ordered_threshold = _finite(configured["ordered_replay_max_abs_max"], "ordered replay threshold")
    ordered_values = {"base": float(base["ordered_replay_max_abs"]), "finetuned": float(fine["ordered_replay_max_abs"])}
    ordered_value = max(ordered_values.values())
    gates["ordered_replay_max_abs_max"] = {
        **_gate(ordered_value, ordered_threshold, "<=", ordered_value <= ordered_threshold),
        "by_checkpoint": ordered_values,
    }
    model_delta = _mapping(steady_comparison.get("finetuned_normal_minus_base_normal"), "steady model delta")
    for gate_name, metric in (
        ("finetuned_normal_minus_base_normal_psnr_db_min", "psnr"),
        ("finetuned_normal_minus_base_normal_ssim_min", "ssim"),
    ):
        threshold = _finite(configured[gate_name], gate_name)
        value = float(model_delta[metric])
        gates[gate_name] = _gate(value, threshold, ">=", value >= threshold)
    fine_controls = _mapping(steady_comparison.get("finetuned_normal_minus_control"), "steady fine controls")
    for gate_name, control in (
        ("finetuned_normal_minus_reset_cache_psnr_db_min", RESET),
        ("finetuned_normal_minus_repeat_current_psnr_db_min", REPEAT),
        ("finetuned_normal_minus_shuffled_history_psnr_db_min", SHUFFLED),
    ):
        threshold = _finite(configured[gate_name], gate_name)
        value = float(_mapping(fine_controls[control], f"fine control {control}")["psnr"])
        gates[gate_name] = _gate(value, threshold, ">=", value >= threshold)
    interaction_name = "history_gain_interaction_vs_base_reset_psnr_db_min"
    interaction_threshold = _finite(configured[interaction_name], interaction_name)
    interaction_value = _finite(steady_comparison.get("history_gain_interaction_vs_base_reset_psnr_db"), "steady interaction")
    gates[interaction_name] = _gate(interaction_value, interaction_threshold, ">=", interaction_value >= interaction_threshold)
    temporal_name = "finetuned_normal_vs_reset_temporal_error_relative_max"
    temporal_threshold = _finite(configured[temporal_name], temporal_name)
    temporal_value = _finite(
        _mapping(steady_comparison.get("finetuned_normal_vs_reset_temporal_difference_error_l1"), "steady temporal comparison").get("relative_difference"),
        "steady temporal relative difference",
    )
    gates[temporal_name] = _gate(temporal_value, temporal_threshold, "<=", temporal_value <= temporal_threshold)
    direction_name = "both_validation_sequences_psnr_direction_positive"
    _require(configured[direction_name] is True, "per-sequence direction gate must remain enabled")
    directions = {
        sequence: float(
            _mapping(
                _mapping(values, f"per_sequence.{sequence}").get("steady"),
                f"per_sequence.{sequence}.steady",
            )["finetuned_normal_minus_control"][RESET]["psnr"]
        )
        for sequence, values in per_sequence.items()
    }
    direction_pass = len(directions) == 2 and all(value > 0.0 for value in directions.values())
    gates[direction_name] = _gate(directions, "each > 0", ">", direction_pass)
    return gates


def build_report(
    contract_path: Path | str,
    base_metrics_path: Path | str,
    finetuned_metrics_path: Path | str,
) -> dict[str, Any]:
    contract, contract_source, contract_sha = _load_json(contract_path, "preregistered contract")
    _require(contract.get("schema") == CONTRACT_SCHEMA, "preregistered contract schema mismatch")
    _expect_equal(contract, "status", "preregistered_exploratory_smoke_not_paper_metric", "contract")
    evaluation_contract = _mapping(contract.get("evaluation"), "contract.evaluation")
    _expect_equal(evaluation_contract, "same_checkpoint_same_frames", True, "contract.evaluation")
    _expect_equal(evaluation_contract, "future_frames_used", False, "contract.evaluation")
    _expect_equal(
        evaluation_contract,
        "arms",
        [
            "normal_persistent_kv",
            "reset_cache_current_only",
            "repeat_current_same_recurrent_step_count",
            "replayed_ordered_complete_prefix_contract",
            "shuffled_complete_past_prefix_only",
        ],
        "contract.evaluation",
    )
    implementation = _verify_implementation_pins(contract)
    expected_rows, manifest_metadata = _load_validation_manifest(contract)
    base_payload, base_source, base_sha = _load_json(base_metrics_path, "base metrics")
    fine_payload, fine_source, fine_sha = _load_json(finetuned_metrics_path, "fine-tuned metrics")
    base = _validate_evaluation(
        base_payload,
        role="base",
        expected_rows=expected_rows,
        manifest_metadata=manifest_metadata,
        contract=contract,
    )
    fine = _validate_evaluation(
        fine_payload,
        role="finetuned",
        expected_rows=expected_rows,
        manifest_metadata=manifest_metadata,
        contract=contract,
    )
    for index, (base_row, fine_row) in enumerate(zip(base["rows"], fine["rows"])):
        for source in set(base_row["metrics"]) & set(fine_row["metrics"]):
            if source in {NORMAL, RESET, REPEAT, ORDERED, SHUFFLED}:
                continue
            _require(base_row["metrics"][source] == fine_row["metrics"][source], f"same-frame invariant source {source!r} changed at row {index}")
        if base_row.get("evssm_path") is not None or fine_row.get("evssm_path") is not None:
            _expect_equal(fine_row, "evssm_path", base_row.get("evssm_path"), f"finetuned.frames[{index}]")

    comparisons = {
        region: _comparison(base[region], fine[region])
        for region in ("all", "steady")
    }
    per_sequence: dict[str, Any] = {}
    _require(set(base["per_sequence"]) == set(fine["per_sequence"]), "base/fine-tuned sequence sets differ")
    for sequence in sorted(base["per_sequence"]):
        per_sequence[sequence] = {
            region: _comparison(
                base["per_sequence"][sequence][region],
                fine["per_sequence"][sequence][region],
            )
            for region in ("all", "steady")
        }
    gates = _build_gates(contract, base, fine, comparisons["steady"], per_sequence)
    eligible = all(bool(_mapping(gate, f"gate.{name}").get("passed")) for name, gate in gates.items())
    status = "eligible_for_metric_bearing_slam_smoke" if eligible else "ineligible_for_metric_bearing_slam_smoke"
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "eligible": eligible,
        "scope": "exploratory_smoke_not_paper_metric",
        "decision": evaluation_contract["on_pass"] if eligible else evaluation_contract["on_failure"],
        "protocol": {
            "metric_aggregation": "arithmetic mean of per-frame evaluator metrics",
            "steady_frame_index_min": manifest_metadata["steady_frame_index_min"],
            "eligibility_region": "steady",
            "temporal_metric": "non-flow-warped GT adjacent-difference L1",
            "temporal_relative_difference": "(finetuned_normal-finetuned_reset)/finetuned_reset",
            "per_sequence_direction": "finetuned normal-minus-reset PSNR must be strictly positive in each validation sequence",
            "ordered_replay_gate": "maximum image-domain absolute difference after reset and exact replay of the complete causal past prefix",
            "future_frames_used": False,
            "room2_read_or_evaluated": False,
        },
        "artifacts": {
            "contract": {"path": str(contract_source), "sha256": contract_sha},
            "validation_manifest": manifest_metadata,
            "implementation_pins": implementation,
            "base_metrics": {"path": str(base_source), "sha256": base_sha},
            "finetuned_metrics": {"path": str(fine_source), "sha256": fine_sha},
            "base_checkpoint": base["checkpoint"],
            "finetuned_checkpoint": fine["checkpoint"],
            "evaluation_devices": {"base": base["device"], "finetuned": fine["device"]},
        },
        "metrics": {
            "base": {region: base[region] for region in ("all", "steady")},
            "finetuned": {region: fine[region] for region in ("all", "steady")},
            "comparisons": comparisons,
            "per_sequence_comparisons": per_sequence,
        },
        "gates": gates,
        "failed_gates": [name for name, gate in gates.items() if not bool(gate["passed"])],
    }


def write_report(path: Path | str, report: Mapping[str, Any], *, overwrite: bool = False) -> str:
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite report: {output}")
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
    return sha256_file(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--base-metrics", type=Path, required=True)
    parser.add_argument("--finetuned-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.contract, args.base_metrics, args.finetuned_metrics)
    digest = write_report(args.output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "sha256": digest,
                "status": report["status"],
                "eligible": report["eligible"],
                "failed_gates": report["failed_gates"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
