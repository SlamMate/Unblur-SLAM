#!/usr/bin/env python3
"""Fail-closed reference gate and validation-only reporter for E/G/O.

The reference gate is deliberately operational, not a quality-selection gate:
O quality and O-minus-G quality never decide whether B/BD may train.  It checks
only frozen identity, validation coverage, finite outputs and replay/runtime
implementation integrity.  The report still discloses every preregistered
quality delta descriptively and never authorizes a test split or a SLAM claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import (  # noqa: E402
    load_contract,
    load_json_object,
    require,
    sha256_file,
    validate_protocol,
)
from scripts.evaluate_evssm_bsd_validation import SCHEMA as E_SCHEMA  # noqa: E402
from scripts.evaluate_turtle_streaming import (  # noqa: E402
    FORMAL_HISTORY_CONTROL_FRAME_INDICES,
    FORMAL_STEADY_FRAME_INDEX_MIN,
    FORMAL_WARMUP_STEPS,
)


GATE_SCHEMA = "unblur_slam.turtle_bsd_reference_qualification_receipt.v1"
REPORT_SCHEMA = "unblur_slam.turtle_bsd_reference_validation_only_report.v1"
FULL_REPORT_SCHEMA = "unblur_slam.turtle_bsd_dpdd_full_validation_report.v1"
TURTLE_SCHEMA = "unblur_slam.turtle_streaming_evaluation.v1"
TURTLE_DPDD_SCHEMA = "unblur_slam.turtle_bsd_dpdd_validation_arm.v1"
EVSSM_DPDD_SCHEMA = "unblur_slam.official_evssm_dpdd_validation.v1"
RGB_METRICS = ("psnr", "ssim", "l1")
DPDD_METRICS = ("psnr", "ssim", "lpips", "l1")
METRIC_ABS_TOLERANCE = 1.0e-12


def _finite_tree(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"non-finite report value at {path}")


def _load_report(path: Path | str) -> tuple[Path, Mapping[str, Any], str]:
    candidate = Path(path).expanduser().resolve()
    require(candidate.is_file(), f"missing validation report: {candidate}")
    payload = load_json_object(candidate)
    _finite_tree(payload)
    return candidate, payload, sha256_file(candidate)


def _frame_identity(payload: Mapping[str, Any], *, arm: str) -> list[tuple[str, int]]:
    rows = payload["results"]["frames"] if arm == "E" else payload["frames"]
    return [(str(row["sequence"]), int(row["frame_index"])) for row in rows]


def _quality(payload: Mapping[str, Any], *, arm: str) -> Mapping[str, Any]:
    if arm == "E":
        results = payload["results"]
        return {
            "all_frames": results["mean"],
            "steady": results["steady_mean"],
            "per_sequence": {
                name: {
                    "all_frames": values["mean"],
                    "steady": values["steady_mean"],
                }
                for name, values in results["per_sequence"].items()
            },
        }
    return {
        "all_frames": payload["mean"]["turtle"],
        "steady": payload["steady_mean"]["turtle"],
        "per_sequence": {
            name: {
                "all_frames": values["mean"]["turtle"],
                "steady": values["steady_mean"]["turtle"],
            }
            for name, values in payload["per_sequence"].items()
        },
    }


def _quality_minus_raw(
    quality: Mapping[str, Any], raw: Mapping[str, Any]
) -> Mapping[str, Any]:
    require(set(quality["per_sequence"]) == set(raw["per_sequence"]), "quality/raw sequence sets differ")
    return {
        "all_frames": _delta(quality["all_frames"], raw["all_frames"]),
        "steady": _delta(quality["steady"], raw["steady"]),
        "per_sequence": {
            name: {
                "all_frames": _delta(
                    quality["per_sequence"][name]["all_frames"],
                    raw["per_sequence"][name]["all_frames"],
                ),
                "steady": _delta(
                    quality["per_sequence"][name]["steady"],
                    raw["per_sequence"][name]["steady"],
                ),
            }
            for name in quality["per_sequence"]
        },
    }


def _delta(candidate: Mapping[str, float], reference: Mapping[str, float]) -> Mapping[str, float]:
    return {
        metric: float(candidate[metric]) - float(reference[metric])
        for metric in RGB_METRICS
    }


def _delta_named(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    metrics: Sequence[str],
) -> Mapping[str, float]:
    return {
        metric: float(candidate[metric]) - float(reference[metric])
        for metric in metrics
    }


def _metric_mapping(value: Any, *, label: str) -> dict[str, float]:
    require(isinstance(value, Mapping), f"{label} is not a metric mapping")
    result: dict[str, float] = {}
    for metric in RGB_METRICS:
        require(metric in value, f"{label} is missing {metric}")
        number = float(value[metric])
        require(math.isfinite(number), f"{label}.{metric} is non-finite")
        result[metric] = number
    return result


def _require_metrics_close(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label: str,
) -> None:
    normalized_left = _metric_mapping(left, label=f"{label}.left")
    normalized_right = _metric_mapping(right, label=f"{label}.right")
    for metric in RGB_METRICS:
        require(
            math.isclose(
                normalized_left[metric],
                normalized_right[metric],
                rel_tol=0.0,
                abs_tol=METRIC_ABS_TOLERANCE,
            ),
            f"{label}.{metric} differs",
        )


def _mean_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    require(bool(rows), f"cannot aggregate empty {key} rows")
    return {
        metric: sum(float(row[key][metric]) for row in rows) / len(rows)
        for metric in RGB_METRICS
    }


def _normalized_bsd_rows(
    payload: Mapping[str, Any], *, arm: str
) -> list[dict[str, Any]]:
    source_rows = payload["results"]["frames"] if arm == "E" else payload["frames"]
    require(isinstance(source_rows, list), f"{arm} frames are not a list")
    normalized: list[dict[str, Any]] = []
    for position, row in enumerate(source_rows):
        require(isinstance(row, Mapping), f"{arm} frame {position} is not an object")
        raw_path = row.get("raw_path", row.get("blurry_path"))
        gt_path = row.get("gt_path", row.get("sharp_path"))
        require(isinstance(raw_path, str) and raw_path, f"{arm} frame raw path missing")
        require(isinstance(gt_path, str) and gt_path, f"{arm} frame GT path missing")
        raw_metrics = (
            row.get("raw_metrics")
            if arm == "E"
            else row.get("metrics", {}).get("raw")
        )
        model_metrics = (
            row.get("metrics")
            if arm == "E"
            else row.get("metrics", {}).get("turtle")
        )
        normalized.append(
            {
                "sequence": str(row["sequence"]),
                "frame_index": int(row["frame_index"]),
                "raw_path": str(Path(raw_path).expanduser().resolve()),
                "gt_path": str(Path(gt_path).expanduser().resolve()),
                "raw_metrics": _metric_mapping(
                    raw_metrics, label=f"{arm} frame {position} raw"
                ),
                "model_metrics": _metric_mapping(
                    model_metrics, label=f"{arm} frame {position} model"
                ),
            }
        )
    identities = [
        (row["sequence"], row["frame_index"], row["raw_path"], row["gt_path"])
        for row in normalized
    ]
    require(len(set(identities)) == len(identities), f"{arm} raw identities are not unique")
    return normalized


def _aggregate_normalized_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    steady = [
        row
        for row in rows
        if int(row["frame_index"]) >= FORMAL_STEADY_FRAME_INDEX_MIN
    ]
    sequence_names = list(dict.fromkeys(str(row["sequence"]) for row in rows))
    return {
        "all_frames": _mean_metrics(rows, "raw_metrics"),
        "steady": _mean_metrics(steady, "raw_metrics"),
        "per_sequence": {
            name: {
                "all_frames": _mean_metrics(
                    [row for row in rows if row["sequence"] == name], "raw_metrics"
                ),
                "steady": _mean_metrics(
                    [
                        row
                        for row in steady
                        if row["sequence"] == name
                    ],
                    "raw_metrics",
                ),
                "frame_count": sum(row["sequence"] == name for row in rows),
                "steady_frame_count": sum(
                    row["sequence"] == name for row in steady
                ),
            }
            for name in sequence_names
        },
    }


def _reported_raw_baseline(payload: Mapping[str, Any], *, arm: str) -> Mapping[str, Any]:
    return payload["results"]["raw_baseline"] if arm == "E" else payload["raw_baseline"]


def _validate_raw_aggregation(
    reported: Mapping[str, Any],
    computed: Mapping[str, Any],
    *,
    arm: str,
) -> None:
    require(
        reported.get("registration", {}).get("per_frame_rows_present") is True,
        f"{arm} raw baseline is not registered per frame",
    )
    _require_metrics_close(
        reported.get("all_frames", {}), computed["all_frames"], label=f"{arm} raw all"
    )
    _require_metrics_close(
        reported.get("steady", {}), computed["steady"], label=f"{arm} raw steady"
    )
    reported_sequences = reported.get("per_sequence", {})
    require(set(reported_sequences) == set(computed["per_sequence"]), f"{arm} raw sequence set changed")
    for name, wanted in computed["per_sequence"].items():
        actual = reported_sequences[name]
        require(actual.get("frame_count") == wanted["frame_count"], f"{arm}/{name} raw frame count changed")
        require(actual.get("steady_frame_count") == wanted["steady_frame_count"], f"{arm}/{name} raw steady count changed")
        _require_metrics_close(actual.get("all_frames", {}), wanted["all_frames"], label=f"{arm}/{name} raw all")
        _require_metrics_close(actual.get("steady", {}), wanted["steady"], label=f"{arm}/{name} raw steady")


def _raw_registration_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "sequence": row["sequence"],
            "frame_index": row["frame_index"],
            "raw_path": row["raw_path"],
            "gt_path": row["gt_path"],
            "raw_metrics": row["raw_metrics"],
        }
        for row in rows
    ]
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_full_reference_reports(
    contract: Mapping[str, Any],
    contract_sha: str,
    *,
    e_report: Path | str,
    g_report: Path | str,
    o_report: Path | str,
) -> Mapping[str, Any]:
    reports: dict[str, Mapping[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    for arm, path in (("E", e_report), ("G", g_report), ("O", o_report)):
        candidate, payload, digest = _load_report(path)
        reports[arm] = payload
        artifacts[arm] = {"path": str(candidate), "sha256": digest}

    e = reports["E"]
    require(e.get("schema") == E_SCHEMA and e.get("arm") == "E", "E report identity changed")
    require(e.get("formal") is True, "E report is not formal")
    require(e.get("protocol", {}).get("contract_sha256") == contract_sha, "E contract lineage changed")
    require(e.get("protocol", {}).get("manifest_sha256") == contract["data"]["bsd"]["validation_manifest_sha256"], "E BSD manifest changed")
    require(e.get("protocol", {}).get("formal_full_validation") is True, "E is not full validation")
    require(e.get("results", {}).get("sequence_count") == 20, "E sequence coverage changed")
    require(e.get("results", {}).get("frame_count") == 2000, "E frame coverage changed")
    require(e.get("model", {}).get("checkpoint_sha256") == contract["models"]["evssm_E"]["checkpoint_sha256"], "E checkpoint identity changed")
    require(e.get("results", {}).get("performance", {}).get("warmup", {}).get("unmeasured_calls") == FORMAL_WARMUP_STEPS, "E warmup changed")
    require(e.get("protocol", {}).get("prediction_representation_for_metrics") == "direct_float32_tensor_no_png_roundtrip", "E metrics used a quantized cache")
    e_performance = e.get("results", {}).get("performance", {})
    e_passes = e_performance.get("pass_separation", {})
    require(e_passes.get("timing_pass", {}).get("stateless_model_steps") == 2000, "E timing coverage changed")
    require(e_passes.get("timing_pass", {}).get("sharp_target_images_opened") is False, "E timing opened targets")
    require(e_passes.get("timing_pass", {}).get("metrics_computed") is False, "E timing computed metrics")
    require(e_passes.get("timing_pass", {}).get("history_or_replay_control_forwards") == 0, "E timing executed controls")
    require(e_passes.get("quality_pass", {}).get("stateless_model_steps") == 2000, "E quality coverage changed")
    require(e_passes.get("quality_pass", {}).get("timed_model_steps") == 0, "E quality pass was timed")
    require(
        e_passes.get("forward_accounting_excluding_warmup")
        == {
            "timing_only_model_steps": 2000,
            "quality_model_steps": 2000,
            "combined_model_steps": 4000,
        },
        "E two-pass forward accounting changed",
    )
    require(e_performance.get("evssm_latency_ms", {}).get("frames") == 2000, "E latency frame count changed")
    require(e_performance.get("steady_evssm_latency_ms", {}).get("frames") == 1940, "E steady latency frame count changed")
    e_precision = e_performance.get("compute_precision", {})
    require(e_precision.get("model_and_input") == "CUDA_FP32", "E precision changed")
    require(e_precision.get("autocast") == "disabled", "E autocast changed")
    require(isinstance(e_precision.get("cuda_matmul_allow_tf32"), bool), "E matmul TF32 flag missing")
    require(isinstance(e_precision.get("cudnn_allow_tf32"), bool), "E cuDNN TF32 flag missing")

    for arm in ("G", "O"):
        payload = reports[arm]
        require(payload.get("schema") == TURTLE_SCHEMA, f"{arm} report schema changed")
        provenance = payload.get("provenance", {})
        require(provenance.get("arm") == arm, f"{arm} report label changed")
        require(provenance.get("contract_sha256") == contract_sha, f"{arm} contract lineage changed")
        require(provenance.get("manifest_sha256") == contract["data"]["bsd"]["validation_manifest_sha256"], f"{arm} BSD manifest changed")
        require(provenance.get("formal_full_validation") is True, f"{arm} is not full validation")
        require(payload.get("sequence_count") == 20 and payload.get("frame_count") == 2000, f"{arm} coverage changed")
        require(provenance.get("checkpoint_sha256") == contract["models"][f"turtle_{arm}"]["checkpoint_sha256"], f"{arm} checkpoint identity changed")
        history = payload.get("history_ablation", {})
        require(history.get("ordered_replay_frame_count") == 2000, f"{arm} replay coverage changed")
        require(history.get("ordered_replay_matches_stream") is True, f"{arm} ordered replay mismatch")
        require(float(history.get("ordered_replay_max_abs", math.inf)) <= 1.0e-6, f"{arm} replay maxabs failed")
        require(history.get("control_frame_count") == 20 * len(FORMAL_HISTORY_CONTROL_FRAME_INDICES), f"{arm} control subset coverage changed")
        require(
            history.get("forward_accounting_excluding_warmup", {}).get("total")
            == 16280,
            f"{arm} history forward budget changed",
        )
        protocol = history.get("protocol", {})
        require(protocol.get("expensive_control_frame_indices_per_sequence") == list(FORMAL_HISTORY_CONTROL_FRAME_INDICES), f"{arm} history subset changed")
        require(protocol.get("normal_and_all_controls_share_backend_autocast_path") is True, f"{arm} autocast path not shared")
        performance = payload.get("performance", {})
        require(performance.get("warmup", {}).get("unmeasured_calls") == FORMAL_WARMUP_STEPS, f"{arm} warmup changed")
        require(performance.get("history_controls_timed") is False, f"{arm} control forwards polluted latency")
        passes = performance.get("pass_separation", {})
        require(passes.get("timing_pass", {}).get("normal_stream_model_steps") == 2000, f"{arm} timing coverage changed")
        require(passes.get("timing_pass", {}).get("sharp_target_images_opened") is False, f"{arm} timing opened targets")
        require(passes.get("timing_pass", {}).get("metrics_computed") is False, f"{arm} timing computed metrics")
        require(passes.get("timing_pass", {}).get("history_or_replay_control_forwards") == 0, f"{arm} timing executed controls")
        require(passes.get("quality_history_pass", {}).get("normal_stream_model_steps") == 2000, f"{arm} quality coverage changed")
        require(passes.get("quality_history_pass", {}).get("timed_model_steps") == 0, f"{arm} quality/history pass was timed")
        require(passes.get("quality_history_pass", {}).get("total_model_steps_including_controls") == 16280, f"{arm} quality/history total changed")
        require(
            passes.get("forward_accounting_excluding_warmup")
            == {
                "timing_only_model_steps": 2000,
                "quality_history_model_steps": 16280,
                "combined_model_steps": 18280,
            },
            f"{arm} two-pass accounting changed",
        )
        require(performance.get("turtle_latency_ms", {}).get("frames") == 2000, f"{arm} latency frame count changed")
        require(performance.get("steady_turtle_latency_ms", {}).get("frames") == 1940, f"{arm} steady latency frame count changed")
        precision = performance.get("compute_precision", {})
        require(precision.get("backend_inference_precision") == "fp16", f"{arm} precision changed")
        require(precision.get("normal_and_control_forward_autocast") == "CUDA_FP16", f"{arm} autocast changed")
        accounting = history.get("forward_accounting_excluding_warmup", {})
        require(accounting.get("dedicated_timing_normal_stream") == 2000, f"{arm} dedicated timing forward count changed")
        require(accounting.get("quality_normal_stream") == 2000, f"{arm} quality normal count changed")
        require(accounting.get("timing_pass_history_or_replay_controls") == 0, f"{arm} controls leaked into timing")
        require(accounting.get("total_including_dedicated_timing_pass") == 18280, f"{arm} total two-pass forward budget changed")

    normalized = {
        arm: _normalized_bsd_rows(reports[arm], arm=arm)
        for arm in ("E", "G", "O")
    }
    identity = [
        (row["sequence"], row["frame_index"], row["raw_path"], row["gt_path"])
        for row in normalized["E"]
    ]
    require(len(identity) == 2000 and len(set(identity)) == 2000, "E frame identity is not unique")
    for arm in ("G", "O"):
        candidate_identity = [
            (row["sequence"], row["frame_index"], row["raw_path"], row["gt_path"])
            for row in normalized[arm]
        ]
        require(candidate_identity == identity, f"{arm}/E raw frame identity/order differs")
        for index, (reference_row, candidate_row) in enumerate(
            zip(normalized["E"], normalized[arm])
        ):
            _require_metrics_close(
                candidate_row["raw_metrics"],
                reference_row["raw_metrics"],
                label=f"{arm}/E frame {index} raw metrics",
            )

    computed_raw = _aggregate_normalized_rows(normalized["E"])
    for arm in ("E", "G", "O"):
        _validate_raw_aggregation(
            _reported_raw_baseline(reports[arm], arm=arm),
            computed_raw,
            arm=arm,
        )

    runtime = {
        "E": e.get("runtime"),
        "G": reports["G"].get("provenance", {}).get("runtime"),
        "O": reports["O"].get("provenance", {}).get("runtime"),
    }
    require(all(isinstance(value, Mapping) for value in runtime.values()), "runtime identity missing")
    require(dict(runtime["E"]) == dict(runtime["G"]) == dict(runtime["O"]), "E/G/O logical runtime identities differ")

    return {
        "reports": reports,
        "artifacts": artifacts,
        "frame_identity_count": len(identity),
        "raw_baseline": computed_raw,
        "raw_registration_sha256": _raw_registration_sha256(normalized["E"]),
        "runtime_identity": dict(runtime["E"]),
    }


def _validate_trained_bsd_report(
    contract: Mapping[str, Any],
    contract_sha: str,
    payload: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    common_rows: Sequence[Mapping[str, Any]],
    common_raw: Mapping[str, Any],
) -> Mapping[str, Any]:
    require(payload.get("schema") == TURTLE_SCHEMA, f"{arm}/{seed} BSD schema changed")
    require(payload.get("sequence_count") == 20 and payload.get("frame_count") == 2000, f"{arm}/{seed} BSD coverage changed")
    provenance = payload.get("provenance", {})
    require(provenance.get("arm") == arm and provenance.get("seed") == seed, f"{arm}/{seed} BSD label changed")
    require(provenance.get("contract_sha256") == contract_sha, f"{arm}/{seed} BSD contract changed")
    require(provenance.get("manifest_sha256") == contract["data"]["bsd"]["validation_manifest_sha256"], f"{arm}/{seed} BSD manifest changed")
    require(provenance.get("formal_full_validation") is True, f"{arm}/{seed} BSD is not full")
    checkpoint_sha = str(provenance.get("checkpoint_sha256", ""))
    require(len(checkpoint_sha) == 64 and all(character in "0123456789abcdef" for character in checkpoint_sha), f"{arm}/{seed} checkpoint SHA missing")
    metadata = payload.get("checkpoint_metadata", {})
    require(metadata.get("kind") == "finetuned" and metadata.get("mode") == arm, f"{arm}/{seed} checkpoint metadata changed")
    require(metadata.get("training", {}).get("seed") == seed, f"{arm}/{seed} training seed changed")

    performance = payload.get("performance", {})
    passes = performance.get("pass_separation", {})
    require(passes.get("forward_accounting_excluding_warmup") == {
        "timing_only_model_steps": 2000,
        "quality_history_model_steps": 16280,
        "combined_model_steps": 18280,
    }, f"{arm}/{seed} BSD two-pass accounting changed")
    require(passes.get("timing_pass", {}).get("history_or_replay_control_forwards") == 0, f"{arm}/{seed} controls leaked into timing")
    require(passes.get("quality_history_pass", {}).get("timed_model_steps") == 0, f"{arm}/{seed} quality/history was timed")
    require(performance.get("compute_precision", {}).get("backend_inference_precision") == "fp16", f"{arm}/{seed} precision changed")

    history = payload.get("history_ablation", {})
    require(history.get("ordered_replay_frame_count") == 2000, f"{arm}/{seed} replay coverage changed")
    replay_max_abs = float(history.get("ordered_replay_max_abs", math.inf))
    require(math.isfinite(replay_max_abs), f"{arm}/{seed} replay maxabs is non-finite")
    require(
        history.get("ordered_replay_matches_stream")
        is (replay_max_abs <= 1.0e-6),
        f"{arm}/{seed} replay boolean/maxabs are inconsistent",
    )
    require(history.get("control_frame_count") == 120, f"{arm}/{seed} history subset coverage changed")

    rows = _normalized_bsd_rows(payload, arm=arm)
    require(len(rows) == 2000, f"{arm}/{seed} BSD per-frame row coverage changed")
    for index, (reference_row, candidate_row) in enumerate(zip(common_rows, rows)):
        require(
            (
                candidate_row["sequence"], candidate_row["frame_index"],
                candidate_row["raw_path"], candidate_row["gt_path"],
            )
            == (
                reference_row["sequence"], reference_row["frame_index"],
                reference_row["raw_path"], reference_row["gt_path"],
            ),
            f"{arm}/{seed} BSD raw identity differs at frame {index}",
        )
        _require_metrics_close(candidate_row["raw_metrics"], reference_row["raw_metrics"], label=f"{arm}/{seed} BSD raw frame {index}")
    _validate_raw_aggregation(_reported_raw_baseline(payload, arm=arm), common_raw, arm=f"{arm}/{seed}")
    return {
        "quality": _quality(payload, arm=arm),
        "history": history,
        "runtime": provenance.get("runtime"),
        "checkpoint_sha256": checkpoint_sha,
    }


def _dpdd_metric_mapping(value: Any, *, label: str) -> dict[str, float]:
    require(isinstance(value, Mapping), f"{label} is not a metric mapping")
    result = {}
    for metric in DPDD_METRICS:
        require(metric in value, f"{label} missing {metric}")
        number = float(value[metric])
        require(math.isfinite(number), f"{label}.{metric} is non-finite")
        result[metric] = number
    return result


def _require_dpdd_metrics_close(left: Mapping[str, Any], right: Mapping[str, Any], *, label: str) -> None:
    left_values = _dpdd_metric_mapping(left, label=f"{label}.left")
    right_values = _dpdd_metric_mapping(right, label=f"{label}.right")
    for metric in DPDD_METRICS:
        require(math.isclose(left_values[metric], right_values[metric], rel_tol=0.0, abs_tol=METRIC_ABS_TOLERANCE), f"{label}.{metric} differs")


def _normalized_dpdd_rows(payload: Mapping[str, Any], *, arm: str) -> list[dict[str, Any]]:
    if arm == "E":
        pairs = payload["results"]["pairs"]
        return [
            {
                "name": str(row["name"]),
                "raw_path": str(Path(row["defocus_path"]).expanduser().resolve()),
                "gt_path": str(Path(row["sharp_path"]).expanduser().resolve()),
                "source_sha256": str(row["source_sha256"]),
                "target_sha256": str(row["target_sha256"]),
                "raw_metrics": _dpdd_metric_mapping(row["raw"], label=f"E DPDD {index} raw"),
                "model_metrics": _dpdd_metric_mapping(row["evssm"], label=f"E DPDD {index} model"),
            }
            for index, row in enumerate(pairs)
        ]
    raw_rows = payload["raw_baseline"]["images"]
    model_rows = payload["model"]["images"]
    require(len(raw_rows) == len(model_rows), f"{arm} DPDD raw/model coverage differs")
    result = []
    for index, (raw, model) in enumerate(zip(raw_rows, model_rows)):
        for key in ("name", "blurry_path", "sharp_path", "source_sha256", "target_sha256"):
            require(raw.get(key) == model.get(key), f"{arm} DPDD {index} {key} differs between raw/model")
        result.append({
            "name": str(raw["name"]),
            "raw_path": str(Path(raw["blurry_path"]).expanduser().resolve()),
            "gt_path": str(Path(raw["sharp_path"]).expanduser().resolve()),
            "source_sha256": str(raw["source_sha256"]),
            "target_sha256": str(raw["target_sha256"]),
            "raw_metrics": _dpdd_metric_mapping(raw["metrics"], label=f"{arm} DPDD {index} raw"),
            "model_metrics": _dpdd_metric_mapping(model["metrics"], label=f"{arm} DPDD {index} model"),
        })
    return result


def _mean_dpdd(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, float]:
    require(len(rows) == 74, "DPDD aggregation requires 74 rows")
    return {
        metric: sum(float(row[key][metric]) for row in rows) / len(rows)
        for metric in DPDD_METRICS
    }


def _validate_dpdd_report(
    contract: Mapping[str, Any],
    contract_sha: str,
    payload: Mapping[str, Any],
    *,
    arm: str,
    seed: int | None,
) -> Mapping[str, Any]:
    if arm == "E":
        require(payload.get("schema") == EVSSM_DPDD_SCHEMA and payload.get("arm") == "E", "E DPDD identity changed")
        require(payload.get("protocol", {}).get("contract_sha256") == contract_sha, "E DPDD contract changed")
        require(payload.get("results", {}).get("pair_count") == 74, "E DPDD coverage changed")
        require(payload.get("results", {}).get("pass_separation", {}).get("forward_accounting_excluding_warmup") == {
            "timing_only_model_steps": 74, "quality_model_steps": 74, "combined_model_steps": 148,
        }, "E DPDD two-pass accounting changed")
        require(payload.get("compute_precision", {}).get("model_and_input") == "CUDA_FP32", "E DPDD precision changed")
        require(payload.get("results", {}).get("latency_ms", {}).get("frames") == 74, "E DPDD latency coverage changed")
        runtime = payload.get("runtime_identity")
        latency = {
            "warmup": payload["results"]["warmup"],
            "latency_ms": payload["results"]["latency_ms"],
            "pass_separation": payload["results"]["pass_separation"],
            "compute_precision": payload["compute_precision"],
            "scope": payload["protocol"]["latency"],
        }
    else:
        require(payload.get("schema") == TURTLE_DPDD_SCHEMA and payload.get("arm") == arm, f"{arm} DPDD identity changed")
        require(payload.get("seed") == seed, f"{arm} DPDD seed changed")
        protocol = payload.get("protocol", {})
        require(protocol.get("contract_sha256") == contract_sha, f"{arm} DPDD contract changed")
        require(protocol.get("pair_count") == 74, f"{arm} DPDD coverage changed")
        require(protocol.get("pass_separation", {}).get("forward_accounting_excluding_warmup") == {
            "timing_only_model_steps": 74, "quality_model_steps": 74, "combined_model_steps": 148,
        }, f"{arm} DPDD two-pass accounting changed")
        require(protocol.get("compute_precision", {}).get("forward_autocast") == "CUDA_FP16", f"{arm} DPDD precision changed")
        require(payload.get("model", {}).get("summary", {}).get("image_count") == 74, f"{arm} DPDD latency coverage changed")
        if arm in {"G", "O"}:
            require(payload.get("checkpoint_sha256") == contract["models"][f"turtle_{arm}"]["checkpoint_sha256"], f"{arm} DPDD checkpoint changed")
        else:
            require(payload.get("checkpoint_metadata", {}).get("mode") == arm, f"{arm}/{seed} DPDD checkpoint mode changed")
            require(payload.get("checkpoint_metadata", {}).get("training", {}).get("seed") == seed, f"{arm}/{seed} DPDD training seed changed")
        runtime = payload.get("runtime_identity")
        latency = {
            "warmup_steps": protocol["warmup_steps"],
            "latency_ms": payload["model"]["summary"]["latency_ms"],
            "latency_frames": 74,
            "pass_separation": protocol["pass_separation"],
            "compute_precision": protocol["compute_precision"],
            "scope": protocol["latency_scope"],
            "comparability": protocol["latency_comparability"],
        }
    rows = _normalized_dpdd_rows(payload, arm=arm)
    require(len(rows) == 74 and len({row["name"] for row in rows}) == 74, f"{arm} DPDD identities changed")
    raw_mean = _mean_dpdd(rows, "raw_metrics")
    model_mean = _mean_dpdd(rows, "model_metrics")
    if arm == "E":
        _require_dpdd_metrics_close(payload["results"]["mean"]["raw"], raw_mean, label="E DPDD raw mean")
        _require_dpdd_metrics_close(payload["results"]["mean"]["evssm"], model_mean, label="E DPDD model mean")
    else:
        _require_dpdd_metrics_close(payload["raw_baseline"]["summary"]["mean"], raw_mean, label=f"{arm} DPDD raw mean")
        _require_dpdd_metrics_close(payload["model"]["summary"]["mean"], model_mean, label=f"{arm} DPDD model mean")
    return {
        "rows": rows,
        "raw": raw_mean,
        "quality": model_mean,
        "runtime": runtime,
        "latency": latency,
    }


def build_reference_gate_receipt(
    contract: Mapping[str, Any],
    contract_sha: str,
    *,
    e_report: Path | str,
    g_report: Path | str,
    o_report: Path | str,
    reference_only: bool = True,
) -> Mapping[str, Any]:
    validated = validate_full_reference_reports(
        contract,
        contract_sha,
        e_report=e_report,
        g_report=g_report,
        o_report=o_report,
    )
    return {
        "schema": GATE_SCHEMA,
        "status": "pass",
        "contract_sha256": contract_sha,
        "scope": "operational_reference_qualification_only",
        "checks": {
            "E_G_O_identity": True,
            "full_validation_coverage_20_sequences_2000_frames": True,
            "all_reported_numbers_finite": True,
            "G_O_incremental_ordered_replay_implementation": True,
            "same_autocast_path_for_G_O_controls": True,
            "warmup_and_model_step_latency_scope_present": True,
            "dedicated_timing_and_quality_passes_disjoint": True,
            "common_raw_frame_identity_and_metrics_consistent": True,
            "raw_all_steady_per_sequence_aggregates_recomputed": True,
        },
        "quality_selection": {
            "quality_thresholds_consulted": False,
            "O_quality_used_to_authorize_training": False,
            "O_minus_G_used_to_authorize_training": False,
            "history_quality_used_to_authorize_training": False,
            "reason": "official O quality is descriptive; only crash/identity/coverage/finite implementation integrity qualifies references",
        },
        "inputs": validated["artifacts"],
        "raw_registration_sha256": validated["raw_registration_sha256"],
        "runtime_identity": validated["runtime_identity"],
        "reference_only": bool(reference_only),
        "bound_contract_operational_reference_gate_passed": not reference_only,
        "bsd_test_pixels_opened": False,
        "training_authorized_by_this_receipt": False,
    }


def build_reference_validation_report(
    contract: Mapping[str, Any],
    contract_sha: str,
    *,
    e_report: Path | str,
    g_report: Path | str,
    o_report: Path | str,
    gate_receipt: Path | str,
) -> Mapping[str, Any]:
    validated = validate_full_reference_reports(
        contract,
        contract_sha,
        e_report=e_report,
        g_report=g_report,
        o_report=o_report,
    )
    gate_path, gate, gate_sha = _load_report(gate_receipt)
    require(gate.get("schema") == GATE_SCHEMA and gate.get("status") == "pass", "reference gate receipt did not pass")
    require(gate.get("contract_sha256") == contract_sha, "reference gate contract changed")
    require(gate.get("quality_selection", {}).get("O_quality_used_to_authorize_training") is False, "O quality leaked into launch gate")
    require(gate.get("inputs") == validated["artifacts"], "reference gate input reports changed")
    require(gate.get("raw_registration_sha256") == validated["raw_registration_sha256"], "reference gate raw registration changed")
    require(gate.get("runtime_identity") == validated["runtime_identity"], "reference gate runtime identity changed")
    require(gate.get("reference_only") is True, "reference report requires a reference-only gate")
    reports = validated["reports"]
    quality = {arm: _quality(reports[arm], arm=arm) for arm in ("E", "G", "O")}
    raw_baseline = validated["raw_baseline"]
    minus_raw = {
        arm: _quality_minus_raw(quality[arm], raw_baseline)
        for arm in ("E", "G", "O")
    }
    history = {}
    for arm in ("G", "O"):
        ablation = reports[arm]["history_ablation"]
        history[arm] = {
            "frozen_control_frame_indices_per_sequence": list(FORMAL_HISTORY_CONTROL_FRAME_INDICES),
            "control_frame_count": ablation["control_frame_count"],
            "normal_minus_reset": ablation["steady_normal_minus_control"]["turtle_reset_cache"],
            "normal_minus_repeat": ablation["steady_normal_minus_control"]["turtle_repeat_current"],
            "normal_minus_shuffled": ablation["steady_normal_minus_control"]["turtle_shuffled_history"],
            "ordered_replay_max_abs": ablation["ordered_replay_max_abs"],
        }
    return {
        "schema": REPORT_SCHEMA,
        "status": "reference_validation_complete_training_still_blocked",
        "contract_sha256": contract_sha,
        "arms": ["E", "G", "O"],
        "raw_baseline": {
            "source": "common_blurry_input_vs_sharp_ground_truth",
            "identical_across_reported_arms": True,
            **raw_baseline,
            "registration_sha256": validated["raw_registration_sha256"],
            "per_frame_E_G_O_identity_equal": True,
            "per_frame_E_G_O_metrics_abs_tolerance": METRIC_ABS_TOLERANCE,
        },
        "quality": {"raw": raw_baseline, **quality},
        "model_minus_raw": minus_raw,
        "descriptive_deltas": {
            "O_minus_G_all_frames": _delta(quality["O"]["all_frames"], quality["G"]["all_frames"]),
            "O_minus_G_steady": _delta(quality["O"]["steady"], quality["G"]["steady"]),
            "G_minus_E_all_frames": _delta(quality["G"]["all_frames"], quality["E"]["all_frames"]),
            "O_minus_E_all_frames": _delta(quality["O"]["all_frames"], quality["E"]["all_frames"]),
            "E_minus_raw": minus_raw["E"],
            "G_minus_raw": minus_raw["G"],
            "O_minus_raw": minus_raw["O"],
        },
        "history_controls": history,
        "latency": {
            "E": reports["E"]["results"]["performance"],
            "G": reports["G"]["performance"],
            "O": reports["O"]["performance"],
            "comparability": (
                "Each arm uses a dedicated timing-only full pass after one unmeasured "
                "warm-up and reports a synchronized model/backend step excluding I/O, "
                "targets, quality/history forwards, metrics, reporting, and SLAM. "
                "E is FP32 with runtime TF32 flags disclosed; G/O use CUDA FP16 "
                "autocast, so cross-arm latency remains descriptive, not a "
                "precision-controlled architecture benchmark."
            ),
        },
        "runtime_identity": validated["runtime_identity"],
        "reference_semantics": {
            "O_t0_vs_G_t1": {
                "O_architecture": "TURTLE_t0",
                "G_architecture": "TURTLE_t1",
                "same_architecture": False,
                "same_training_data": False,
                "training_budget_matched": False,
                "causal_effect_estimate": False,
                "interpretation": (
                    "descriptive cross-architecture, cross-training-data, "
                    "cross-training-budget system comparison only"
                ),
            },
            "E": {
                "architecture": "EVSSM",
                "external_reference": True,
                "stateless_single_frame": True,
                "same_method_arm": False,
                "causal_history_claim": False,
            },
            "precision": {
                "E": reports["E"]["results"]["performance"]["compute_precision"],
                "G": reports["G"]["performance"]["compute_precision"],
                "O": reports["O"]["performance"]["compute_precision"],
                "precision_matched_across_E_G_O": False,
                "summary": "E FP32/TF32-runtime-flags-disclosed versus G/O FP16 autocast",
            },
        },
        "reference_gate_receipt": {"path": str(gate_path), "sha256": gate_sha},
        "input_reports": validated["artifacts"],
        "decision_policy": {
            "O_quality_gated_training": False,
            "quality_thresholds_changed": False,
            "training_blocked_because_BSD_train_is_unbound": True,
        },
        "claim_scope": "image_restoration_module_quality_and_model_step_latency_only",
        "slam_quality_or_speed_claim": False,
        "claims_policy": {
            "forbidden_claims": list(contract["claims_forbidden"]),
            "forbidden_claims_made": [],
            "all_forbidden_claims_excluded": True,
        },
        "bsd_test_pixels_opened": False,
        "bsd_test_authorized": False,
    }


def _paired_dpdd_delta(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> Mapping[str, float]:
    require(len(candidate) == len(reference) == 74, "paired DPDD delta coverage changed")
    totals = {metric: 0.0 for metric in DPDD_METRICS}
    for index, (left, right) in enumerate(zip(candidate, reference)):
        identity_left = (
            left["name"], left["raw_path"], left["gt_path"],
            left["source_sha256"], left["target_sha256"],
        )
        identity_right = (
            right["name"], right["raw_path"], right["gt_path"],
            right["source_sha256"], right["target_sha256"],
        )
        require(identity_left == identity_right, f"paired DPDD identity differs at {index}")
        for metric in DPDD_METRICS:
            totals[metric] += float(left["model_metrics"][metric]) - float(
                right["model_metrics"][metric]
            )
    return {metric: totals[metric] / 74 for metric in DPDD_METRICS}


def _gate_check(value: float, threshold: float, operator: str) -> Mapping[str, Any]:
    if operator == ">=":
        passed = value >= threshold
    elif operator == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"unsupported gate operator {operator!r}")
    return {
        "value": float(value),
        "operator": operator,
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def build_full_validation_report(
    contract: Mapping[str, Any],
    contract_sha: str,
    *,
    e_bsd: Path | str,
    g_bsd: Path | str,
    o_bsd: Path | str,
    e_dpdd: Path | str,
    g_dpdd: Path | str,
    o_dpdd: Path | str,
    trained_bsd: Mapping[tuple[str, int], Path | str],
    trained_dpdd: Mapping[tuple[str, int], Path | str],
    gate_receipt: Path | str,
) -> Mapping[str, Any]:
    """Build the terminal three-seed validation report without selecting models."""

    references = validate_full_reference_reports(
        contract, contract_sha, e_report=e_bsd, g_report=g_bsd, o_report=o_bsd
    )
    gate_path, gate, gate_sha = _load_report(gate_receipt)
    require(gate.get("schema") == GATE_SCHEMA and gate.get("status") == "pass", "full-plan operational gate failed")
    require(gate.get("contract_sha256") == contract_sha, "full-plan gate contract changed")
    require(gate.get("reference_only") is False, "full-plan gate is reference-only")
    require(gate.get("bound_contract_operational_reference_gate_passed") is True, "bound operational reference gate missing")
    require(gate.get("quality_selection", {}).get("O_quality_used_to_authorize_training") is False, "O quality leaked into full launch gate")
    require(gate.get("inputs") == references["artifacts"], "full-plan gate inputs changed")

    common_bsd_rows = _normalized_bsd_rows(references["reports"]["E"], arm="E")
    bsd_artifacts: dict[str, Any] = dict(references["artifacts"])
    bsd_trained: dict[str, dict[int, Mapping[str, Any]]] = {"B": {}, "BD": {}}
    for arm in ("B", "BD"):
        for seed in (17, 42, 73):
            path, payload, digest = _load_report(trained_bsd[(arm, seed)])
            validated = _validate_trained_bsd_report(
                contract,
                contract_sha,
                payload,
                arm=arm,
                seed=seed,
                common_rows=common_bsd_rows,
                common_raw=references["raw_baseline"],
            )
            bsd_trained[arm][seed] = {"payload": payload, **validated}
            bsd_artifacts[f"{arm}_seed{seed}"] = {"path": str(path), "sha256": digest}

    dpdd_paths: dict[tuple[str, int | None], Path | str] = {
        ("E", None): e_dpdd,
        ("G", None): g_dpdd,
        ("O", None): o_dpdd,
        **{key: value for key, value in trained_dpdd.items()},
    }
    dpdd_validated: dict[tuple[str, int | None], Mapping[str, Any]] = {}
    dpdd_artifacts: dict[str, Any] = {}
    for (arm, seed), report_path in dpdd_paths.items():
        path, payload, digest = _load_report(report_path)
        dpdd_validated[(arm, seed)] = _validate_dpdd_report(
            contract, contract_sha, payload, arm=arm, seed=seed
        )
        label = arm if seed is None else f"{arm}_seed{seed}"
        dpdd_artifacts[label] = {"path": str(path), "sha256": digest}

    common_dpdd = dpdd_validated[("E", None)]["rows"]
    common_dpdd_raw = dpdd_validated[("E", None)]["raw"]
    for key, values in dpdd_validated.items():
        label = key[0] if key[1] is None else f"{key[0]}/{key[1]}"
        rows = values["rows"]
        require(len(rows) == len(common_dpdd), f"{label} DPDD coverage changed")
        for index, (reference_row, candidate_row) in enumerate(zip(common_dpdd, rows)):
            reference_identity = (
                reference_row["name"], reference_row["raw_path"], reference_row["gt_path"],
                reference_row["source_sha256"], reference_row["target_sha256"],
            )
            candidate_identity = (
                candidate_row["name"], candidate_row["raw_path"], candidate_row["gt_path"],
                candidate_row["source_sha256"], candidate_row["target_sha256"],
            )
            require(candidate_identity == reference_identity, f"{label} DPDD raw identity differs at {index}")
            _require_dpdd_metrics_close(candidate_row["raw_metrics"], reference_row["raw_metrics"], label=f"{label} DPDD raw {index}")
        _require_dpdd_metrics_close(values["raw"], common_dpdd_raw, label=f"{label} DPDD raw mean")

    runtime_identity = references["runtime_identity"]
    for arm in ("B", "BD"):
        for seed in (17, 42, 73):
            require(bsd_trained[arm][seed]["runtime"] == runtime_identity, f"{arm}/{seed} BSD runtime changed")
    for key, values in dpdd_validated.items():
        require(values["runtime"] == runtime_identity, f"{key} DPDD runtime changed")

    reference_quality = {
        arm: _quality(references["reports"][arm], arm=arm)
        for arm in ("E", "G", "O")
    }
    bsd_quality: dict[str, Any] = {
        "raw": references["raw_baseline"],
        **reference_quality,
        "B": {str(seed): bsd_trained["B"][seed]["quality"] for seed in (17, 42, 73)},
        "BD": {str(seed): bsd_trained["BD"][seed]["quality"] for seed in (17, 42, 73)},
    }
    dpdd_quality: dict[str, Any] = {
        "raw": common_dpdd_raw,
        "E": dpdd_validated[("E", None)]["quality"],
        "G": dpdd_validated[("G", None)]["quality"],
        "O": dpdd_validated[("O", None)]["quality"],
        "B": {str(seed): dpdd_validated[("B", seed)]["quality"] for seed in (17, 42, 73)},
        "BD": {str(seed): dpdd_validated[("BD", seed)]["quality"] for seed in (17, 42, 73)},
    }
    bsd_minus_raw: dict[str, Any] = {
        arm: _quality_minus_raw(reference_quality[arm], references["raw_baseline"])
        for arm in ("E", "G", "O")
    }
    for arm in ("B", "BD"):
        bsd_minus_raw[arm] = {
            str(seed): _quality_minus_raw(
                bsd_trained[arm][seed]["quality"], references["raw_baseline"]
            )
            for seed in (17, 42, 73)
        }
    dpdd_minus_raw: dict[str, Any] = {
        arm: _delta_named(
            dpdd_validated[(arm, None)]["quality"], common_dpdd_raw, DPDD_METRICS
        )
        for arm in ("E", "G", "O")
    }
    for arm in ("B", "BD"):
        dpdd_minus_raw[arm] = {
            str(seed): _delta_named(
                dpdd_validated[(arm, seed)]["quality"], common_dpdd_raw, DPDD_METRICS
            )
            for seed in (17, 42, 73)
        }

    thresholds = contract["preregistered_validation_gates"]
    seed_results: dict[str, Any] = {}
    all_checks: list[bool] = []
    for seed in (17, 42, 73):
        b_minus_g = _delta(
            bsd_trained["B"][seed]["quality"]["steady"],
            reference_quality["G"]["steady"],
        )
        bd_minus_b_bsd = _delta(
            bsd_trained["BD"][seed]["quality"]["steady"],
            bsd_trained["B"][seed]["quality"]["steady"],
        )
        bd_minus_b_dpdd = _paired_dpdd_delta(
            dpdd_validated[("BD", seed)]["rows"],
            dpdd_validated[("B", seed)]["rows"],
        )
        adaptation = thresholds["bsd_adaptation_B_minus_G"]
        preservation = thresholds["bsd_preservation_BD_minus_B"]
        dpdd_gate = thresholds["dpdd_value_BD_minus_B"]
        temporal = thresholds["temporal_preservation"]
        checks: dict[str, Any] = {
            "bsd_adaptation_B_minus_G": {
                "delta": b_minus_g,
                "psnr": _gate_check(b_minus_g["psnr"], adaptation["per_seed_steady_pooled_psnr_db_min"], ">="),
                "ssim": _gate_check(b_minus_g["ssim"], adaptation["per_seed_steady_pooled_ssim_min"], ">="),
                "l1": _gate_check(b_minus_g["l1"], adaptation["per_seed_steady_pooled_l1_max"], "<="),
            },
            "dpdd_value_BD_minus_B": {
                "delta": bd_minus_b_dpdd,
                "psnr": _gate_check(bd_minus_b_dpdd["psnr"], dpdd_gate["per_seed_psnr_db_min"], ">="),
                "ssim": _gate_check(bd_minus_b_dpdd["ssim"], dpdd_gate["per_seed_ssim_min"], ">="),
                "lpips": _gate_check(bd_minus_b_dpdd["lpips"], dpdd_gate["per_seed_lpips_max"], "<="),
                "l1": _gate_check(bd_minus_b_dpdd["l1"], dpdd_gate["per_seed_l1_max"], "<="),
            },
            "bsd_preservation_BD_minus_B": {
                "delta": bd_minus_b_bsd,
                "psnr": _gate_check(bd_minus_b_bsd["psnr"], preservation["per_seed_steady_pooled_psnr_db_min"], ">="),
                "ssim": _gate_check(bd_minus_b_bsd["ssim"], preservation["per_seed_steady_pooled_ssim_min"], ">="),
                "l1": _gate_check(bd_minus_b_bsd["l1"], preservation["per_seed_steady_pooled_l1_max"], "<="),
            },
            "temporal_preservation": {
                "normal_steady_BD_minus_B_psnr": _gate_check(
                    bd_minus_b_bsd["psnr"],
                    temporal["per_seed_BD_minus_B_normal_steady_pooled_psnr_db_min"],
                    ">=",
                )
            },
        }
        history_thresholds = thresholds["history_value"]
        checks["history_value"] = {}
        for arm in ("B", "BD"):
            controls = bsd_trained[arm][seed]["history"]["steady_normal_minus_control"]
            checks["history_value"][arm] = {
                name: _gate_check(
                    controls[control]["psnr"],
                    history_thresholds[
                        f"B_and_BD_per_seed_steady_pooled_normal_minus_{name}_psnr_db_min"
                    ],
                    ">=",
                )
                for name, control in (
                    ("reset", "turtle_reset_cache"),
                    ("repeat", "turtle_repeat_current"),
                    ("shuffled", "turtle_shuffled_history"),
                )
            }
        checks["ordered_replay"] = {
            arm: _gate_check(
                float(bsd_trained[arm][seed]["history"]["ordered_replay_max_abs"]),
                float(thresholds["ordered_replay"]["max_abs_over_every_pixel_frame_and_G_O_B_BD_checkpoint_max"]),
                "<=",
            )
            for arm in ("B", "BD")
        }
        seed_passes = [
            bool(value["passed"])
            for group in checks.values()
            for value in (
                group.values() if isinstance(group, Mapping) else ()
            )
            if isinstance(value, Mapping) and "passed" in value
        ]
        # Include one additional nesting level (history arm -> control).
        for arm_values in checks["history_value"].values():
            seed_passes.extend(bool(value["passed"]) for value in arm_values.values())
        seed_results[str(seed)] = {
            "checks": checks,
            "all_primary_gates_passed": all(seed_passes),
        }
        all_checks.extend(seed_passes)

    return {
        "schema": FULL_REPORT_SCHEMA,
        "status": "full_validation_complete_three_seed_gate_reported",
        "contract_sha256": contract_sha,
        "arms": ["E", "G", "O", "B", "BD"],
        "seeds": [17, 42, 73],
        "raw_baseline": {
            "BSD": {
                "source": "common_blurry_input_vs_sharp_ground_truth",
                "identical_across_reported_arms": True,
                **references["raw_baseline"],
                "registration_sha256": references["raw_registration_sha256"],
            },
            "DPDD": {
                "source": "common_defocus_input_vs_sharp_ground_truth",
                "identical_across_reported_arms": True,
                "all_images": common_dpdd_raw,
            },
        },
        "quality": {"BSD": bsd_quality, "DPDD": dpdd_quality},
        "model_minus_raw": {"BSD": bsd_minus_raw, "DPDD": dpdd_minus_raw},
        "latency": {
            "BSD": {
                "E": references["reports"]["E"]["results"]["performance"],
                "G": references["reports"]["G"]["performance"],
                "O": references["reports"]["O"]["performance"],
                "B": {
                    str(seed): bsd_trained["B"][seed]["payload"]["performance"]
                    for seed in (17, 42, 73)
                },
                "BD": {
                    str(seed): bsd_trained["BD"][seed]["payload"]["performance"]
                    for seed in (17, 42, 73)
                },
            },
            "DPDD": {
                "E": dpdd_validated[("E", None)]["latency"],
                "G": dpdd_validated[("G", None)]["latency"],
                "O": dpdd_validated[("O", None)]["latency"],
                "B": {
                    str(seed): dpdd_validated[("B", seed)]["latency"]
                    for seed in (17, 42, 73)
                },
                "BD": {
                    str(seed): dpdd_validated[("BD", seed)]["latency"]
                    for seed in (17, 42, 73)
                },
            },
            "comparability": {
                "timing_only_normal_model_step_passes": True,
                "quality_and_history_or_lpips_passes_excluded": True,
                "warmup_unmeasured": True,
                "E_precision": "CUDA_FP32_TF32_disabled",
                "TURTLE_precision": "CUDA_FP16_autocast_TF32_disabled",
                "precision_flops_and_architecture_matched": False,
                "scope": "restoration_module_model_step_only_not_SLAM",
            },
        },
        "primary_gates": {
            "all_three_seeds_and_all_primary_gates_conjunctive": True,
            "O_quality_or_history_consulted": False,
            "per_seed": seed_results,
            "all_primary_gates_passed": all(all_checks),
            "terminal_rule": thresholds["terminal_rule"],
        },
        "reference_gate_receipt": {"path": str(gate_path), "sha256": gate_sha},
        "input_reports": {"BSD": bsd_artifacts, "DPDD": dpdd_artifacts},
        "runtime_identity": runtime_identity,
        "reference_semantics": {
            "O_t0_vs_G_t1_cross_architecture_data_budget": True,
            "O_minus_G_causal_effect_estimate": False,
            "E_external_stateless_single_frame_reference": True,
            "E_same_method_arm": False,
            "precision": "E CUDA FP32 with TF32 flags disclosed; TURTLE CUDA FP16 autocast",
        },
        "claim_scope": "image_restoration_module_quality_and_model_step_latency_only",
        "slam_quality_or_speed_claim": False,
        "claims_policy": {
            "forbidden_claims": list(contract["claims_forbidden"]),
            "forbidden_claims_made": [],
            "all_forbidden_claims_excluded": True,
        },
        "bsd_test_pixels_opened": False,
        "dpdd_test_pixels_opened": False,
        "test_authorized": False,
    }


def write_new_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("reference-gate", "training-gate", "reference-report", "full-report"),
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--E", type=Path)
    parser.add_argument("--G", type=Path)
    parser.add_argument("--O", type=Path)
    parser.add_argument("--E-bsd", type=Path)
    parser.add_argument("--G-bsd", type=Path)
    parser.add_argument("--O-bsd", type=Path)
    parser.add_argument("--E-dpdd", type=Path)
    parser.add_argument("--G-dpdd", type=Path)
    parser.add_argument("--O-dpdd", type=Path)
    for arm in ("B", "BD"):
        for seed in (17, 42, 73):
            parser.add_argument(f"--{arm}-bsd-seed{seed}", type=Path)
            parser.add_argument(f"--{arm}-dpdd-seed{seed}", type=Path)
    parser.add_argument("--gate-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _, contract, contract_sha = load_contract(
        args.contract, expected_sha256=args.expected_contract_sha256
    )
    validate_protocol(
        contract, allow_template=False, reference_only=args.reference_only
    )
    if args.mode in {"reference-gate", "training-gate"}:
        require(args.E is not None and args.G is not None and args.O is not None, f"{args.mode} requires --E/--G/--O")
        require(args.gate_receipt is None, "reference-gate does not accept --gate-receipt")
        if args.mode == "training-gate":
            require(args.reference_only is False, "training-gate cannot use --reference-only")
        payload = build_reference_gate_receipt(
            contract,
            contract_sha,
            e_report=args.E,
            g_report=args.G,
            o_report=args.O,
            reference_only=args.reference_only,
        )
    elif args.mode == "reference-report":
        require(args.E is not None and args.G is not None and args.O is not None, "reference-report requires --E/--G/--O")
        require(args.gate_receipt is not None, "reference-report requires --gate-receipt")
        require(args.reference_only is True, "reference-report requires --reference-only")
        payload = build_reference_validation_report(
            contract,
            contract_sha,
            e_report=args.E,
            g_report=args.G,
            o_report=args.O,
            gate_receipt=args.gate_receipt,
        )
    else:
        require(args.reference_only is False, "full-report cannot use --reference-only")
        require(args.gate_receipt is not None, "full-report requires --gate-receipt")
        required_references = {
            "E_bsd": args.E_bsd,
            "G_bsd": args.G_bsd,
            "O_bsd": args.O_bsd,
            "E_dpdd": args.E_dpdd,
            "G_dpdd": args.G_dpdd,
            "O_dpdd": args.O_dpdd,
        }
        require(all(value is not None for value in required_references.values()), "full-report requires all explicit E/G/O BSD+DPDD reports")
        trained_bsd = {
            (arm, seed): getattr(args, f"{arm}_bsd_seed{seed}")
            for arm in ("B", "BD")
            for seed in (17, 42, 73)
        }
        trained_dpdd = {
            (arm, seed): getattr(args, f"{arm}_dpdd_seed{seed}")
            for arm in ("B", "BD")
            for seed in (17, 42, 73)
        }
        require(all(value is not None for value in trained_bsd.values()), "full-report requires all six B/BD BSD reports")
        require(all(value is not None for value in trained_dpdd.values()), "full-report requires all six B/BD DPDD reports")
        payload = build_full_validation_report(
            contract,
            contract_sha,
            e_bsd=args.E_bsd,
            g_bsd=args.G_bsd,
            o_bsd=args.O_bsd,
            e_dpdd=args.E_dpdd,
            g_dpdd=args.G_dpdd,
            o_dpdd=args.O_dpdd,
            trained_bsd=trained_bsd,
            trained_dpdd=trained_dpdd,
            gate_receipt=args.gate_receipt,
        )
    output = write_new_json(args.output, payload)
    print(json.dumps({"output": str(output), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
