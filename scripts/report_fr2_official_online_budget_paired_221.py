#!/usr/bin/env python3
"""Audit the bounded 221-frame official-online-budget EVSSM/TURTLE pair.

This is deliberately a result auditor, not an evaluator.  It performs no CUDA
work and recomputes no image metrics.  Instead it verifies that both completed
runs used the predeclared paired contract, that their saved metric records use
the exact eleven clear-GT frames, and that their saved full trajectories cover
source frames 0..220.  It then emits a compact JSON/CSV comparison.

The experiment uses the published *online optimization budgets* (1050/100/100)
but only a 221-frame fr2_xyz prefix and 100 final-refinement iterations.  It is
not the paper's three-sequence runtime benchmark and not its 26K refinement
protocol.  Derived prefix FPS must therefore not be presented as Table 6 FPS.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml


DEFAULT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_paired/"
    "fr2_xyz_official_online_budget_221"
)
SCENE = "freiburg2_xyz"
SOURCE_COUNT = 221
EXPECTED_CLEAR_GT = (0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220)
EXPECTED_REFINEMENT_CHECKPOINTS = (100,)
ARMS = {
    "baseline": "evssm_baseline",
    "turtle": "turtle_gopro_fp16",
}
EXPECTED_FRONTENDS = {
    "baseline": "evssm",
    "turtle": "turtle_streaming",
}
EXPECTED_EVSSM_SHA256 = (
    "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
)
EXPECTED_DROID_SHA256 = (
    "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526"
)
EXPECTED_OMNIDATA_SHA256 = (
    "a0fab23fee64aa9e4bbe0b520b18b196ea7594a7f719c1d8c10cf11dcb6e4a1e"
)
EXPECTED_TURTLE_CHECKPOINT_SHA256 = (
    "10334b3e81d0416bcde5ccaca960dc81dbfb5b6d23e53fadaf7896d72b580c82"
)
EXPECTED_TURTLE_CONFIG_SHA256 = (
    "123b07de8d3f329769562e2f943e08fdf86c576c405634bad199ced95b25aa23"
)
EXPECTED_TURTLE_ARCH_SHA256 = (
    "4d19c676f92574dbad493eb591312fdeaf2b3b519f57410af2ed95fdbef5f058"
)
EXPECTED_TURTLE_COMMIT = "7094f4221b64ad0962b4f27ff1b76d788836e804"
ALLOWED_PAIR_DIFFS = {
    "data.output",
    "deblur.frontend",
    "deblur.turtle_checkpoint",
    "deblur.turtle_config",
    "deblur.turtle_inference_precision",
    "deblur.turtle_repo",
    "evssm_checkpoint",
    "evssm_checkpoint_sha256",
}
FINAL_LOG_RE = re.compile(
    r"mean psnr: (?P<psnr>[-+0-9.eE]+),\s*"
    r"ssim: (?P<ssim>[-+0-9.eE]+),\s*"
    r"lpips: (?P<lpips>[-+0-9.eE]+),\s*"
    r"depth l1: (?P<depth_l1>[-+0-9.eE]+)"
)
STATISTICS_RE = re.compile(r"statistics:\s*(\{[^\n]+\})")
TRACKING_FRAME_RE = re.compile(r"TRACKING: Processing frame (\d+)/221")
EVSSM_LATENCY_RE = re.compile(r"evssm去模糊耗时: ([-+0-9.eE]+) ms")
TURTLE_CACHE_LATENCY_RE = re.compile(
    r"turtle_streaming去模糊耗时: ([-+0-9.eE]+) ms"
)
TURTLE_STREAM_RE = re.compile(
    r"TRACKING: streaming deblur frame=(?P<frame>\d+) "
    r"gain=(?P<gain>[-+0-9.eE]+) "
    r"vs_evssm=(?:None|[-+0-9.eE]+) "
    r"replace=(?P<replace>True|False) "
    r"time_ms=(?P<latency>[-+0-9.eE]+)"
)


class ContractError(RuntimeError):
    """Raised when an input is partial or violates the paired protocol."""


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ContractError(f"missing {label}: {path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} is not a JSON object: {path}")
    return value


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ContractError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} is not a YAML mapping: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _require_file(path, "SHA-256 input").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(value[key], path))
    return result


def _pair_differences(
    baseline: Mapping[str, Any], turtle: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    left = _flatten(baseline)
    right = _flatten(turtle)
    return {
        path: {"baseline": left.get(path), "turtle": right.get(path)}
        for path in sorted(set(left) | set(right))
        if left.get(path) != right.get(path)
    }


def _without_runtime_metadata(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the preflight-equivalent config saved before backend enrichment."""

    normalized = copy.deepcopy(dict(cfg))
    deblur = normalized.get("deblur")
    if isinstance(deblur, dict):
        deblur.pop("turtle_checkpoint_metadata", None)
    return normalized


def _nested(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ContractError(f"resolved config is missing {dotted}")
        current = current[part]
    return current


def _expect(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ContractError(f"{label}={value!r}, expected {expected!r}")


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ContractError(f"{label} is not finite: {number!r}")
    return number


def _validate_config(cfg: Mapping[str, Any], arm: str, arm_root: Path) -> None:
    expected = {
        "dataset": "tumrgbd",
        "scene": SCENE,
        "stride": 1,
        "max_frames": SOURCE_COUNT,
        "setup_seed": 43,
        "device": "cuda:0",
        "warmup_mapper": True,
        "clear_init": False,
        "cam.W_out": 512,
        "cam.H_out": 384,
        "mapping.Training.init_itr_num": 1050,
        "mapping.Training.mapping_itr_num": 100,
        "mapping.Training.tracking_itr_num": 100,
        "mapping.final_refine_iters": 100,
        "mapping.online_plotting": False,
        "mapping.eval_before_final_ba": False,
        "mapping.hydrate_missing_droid_keyframes": True,
        "mapping.resplat.enabled": False,
        "mapping.resplat.online_enabled": False,
        "mapping.resplat.extra_iters": 0,
        "framecrafter.enabled": False,
        "submaps.enabled": False,
        "submaps.official_resplat_sidecar.enabled": False,
        "evaluation.clear_gt_scope": "prefix_smoke",
        "deblur.frontend": EXPECTED_FRONTENDS[arm],
    }
    for path, wanted in expected.items():
        _expect(_nested(cfg, path), wanted, f"{arm} config {path}")
    _expect(
        tuple(int(value) for value in _nested(cfg, "evaluation.expected_clear_gt_source_indices")),
        EXPECTED_CLEAR_GT,
        f"{arm} config clear-GT indices",
    )
    _expect(
        Path(str(_nested(cfg, "data.output"))).expanduser().resolve(),
        arm_root.resolve(),
        f"{arm} config output",
    )
    disclosure = _nested(cfg, "paired_official_online_budget_221")
    if not isinstance(disclosure, Mapping):
        raise ContractError(f"{arm} scope disclosure is not a mapping")
    _expect(
        disclosure.get("schema"),
        "unblur_slam.fr2_xyz_paired_official_online_budget_221.v1",
        f"{arm} disclosure schema",
    )
    _expect(disclosure.get("official_online_optimization_budget"), True, f"{arm} online budget disclosure")
    _expect(disclosure.get("complete_three_sequence_paper_benchmark"), False, f"{arm} paper benchmark disclosure")
    _expect(disclosure.get("paper_26k_offline_refinement"), False, f"{arm} 26K disclosure")

    deblur = cfg.get("deblur") or {}
    if arm == "baseline":
        _expect(cfg.get("evssm_checkpoint_sha256"), EXPECTED_EVSSM_SHA256, "baseline EVSSM digest")
        _expect(str(deblur.get("causal_checkpoint", "")), "", "baseline causal checkpoint")
    else:
        _expect(str(cfg.get("evssm_checkpoint", "")), "", "TURTLE EVSSM checkpoint")
        _expect(str(cfg.get("evssm_checkpoint_sha256", "")), "", "TURTLE EVSSM digest")
        _expect(deblur.get("turtle_inference_precision"), "fp16", "TURTLE precision")
        _expect(deblur.get("stream_every_frame"), True, "TURTLE stream_every_frame")
        _expect(deblur.get("stream_apply_to_tracking"), True, "TURTLE tracking application")
        _expect(deblur.get("stream_replace_sharp"), False, "TURTLE sharp replacement")
        _expect(float(deblur.get("stream_min_laplacian_gain", -1)), 0.02, "TURTLE gate")
        _expect(
            deblur.get("turtle_checkpoint_sha256"),
            EXPECTED_TURTLE_CHECKPOINT_SHA256,
            "TURTLE checkpoint digest",
        )
        metadata = deblur.get("turtle_checkpoint_metadata") or {}
        _expect(metadata.get("kind"), "official_gopro", "TURTLE runtime checkpoint kind")
        _expect(metadata.get("input_domain"), "raw", "TURTLE runtime input domain")
        _expect(
            metadata.get("cache_contract"),
            "official_kv_8_incremental",
            "TURTLE runtime cache contract",
        )
        _expect(
            metadata.get("checkpoint_sha256"),
            EXPECTED_TURTLE_CHECKPOINT_SHA256,
            "TURTLE runtime checkpoint digest",
        )
        _expect(
            metadata.get("turtle_config_sha256"),
            EXPECTED_TURTLE_CONFIG_SHA256,
            "TURTLE runtime config digest",
        )
        _expect(
            metadata.get("turtle_arch_sha256"),
            EXPECTED_TURTLE_ARCH_SHA256,
            "TURTLE runtime architecture digest",
        )
        _expect(
            metadata.get("turtle_repo_commit"),
            EXPECTED_TURTLE_COMMIT,
            "TURTLE runtime repository commit",
        )


def _validate_preflight(preflight: Mapping[str, Any], cfg: Mapping[str, Any], arm: str) -> None:
    _expect(
        preflight.get("schema"),
        "unblur_slam.fr2_xyz_paired_official_online_budget_221_preflight.v1",
        f"{arm} preflight schema",
    )
    scope = preflight.get("scope") or {}
    _expect(scope.get("source_count"), SOURCE_COUNT, f"{arm} preflight source count")
    _expect(scope.get("source_first"), 0, f"{arm} preflight first source")
    _expect(scope.get("source_last"), 220, f"{arm} preflight last source")
    _expect(scope.get("official_online_optimization_budget"), True, f"{arm} preflight online budget")
    _expect(scope.get("complete_three_sequence_paper_benchmark"), False, f"{arm} preflight paper benchmark")
    _expect(scope.get("paper_26k_offline_refinement"), False, f"{arm} preflight 26K")
    _expect(
        tuple(int(value) for value in scope.get("clear_gt_source_indices", ())),
        EXPECTED_CLEAR_GT,
        f"{arm} preflight clear-GT indices",
    )
    paired = preflight.get("paired_contract") or {}
    digest_key = f"{arm}_resolved_sha256"
    _expect(
        paired.get(digest_key),
        _canonical_sha256(_without_runtime_metadata(cfg)),
        f"{arm} pre-backend resolved config digest",
    )


def _read_runtime(arm_root: Path, scene_root: Path, arm: str) -> dict[str, float]:
    launcher = _load_json(arm_root / "launcher_runtime.json", f"{arm} launcher runtime")
    _expect(launcher.get("exit_code"), 0, f"{arm} launcher exit code")
    _expect(launcher.get("arm"), arm, f"{arm} launcher arm")
    _expect(launcher.get("physical_gpu"), 1, f"{arm} physical GPU")
    _expect(launcher.get("process_device"), "cuda:0", f"{arm} process device")
    runtime = _load_json(scene_root / "runtime_stats.json", f"{arm} runtime stats")
    online = _finite_float(runtime.get("online_inference_time"), f"{arm} online time")
    total = _finite_float(runtime.get("total_inference_time"), f"{arm} total time")
    wall = _finite_float(launcher.get("wall_runtime_seconds"), f"{arm} external wall time")
    if not (0.0 < online <= total <= wall + 1.0):
        raise ContractError(
            f"{arm} timing order is invalid: online={online}, total={total}, wall={wall}"
        )
    return {
        "official_timer_online_seconds": online,
        "official_timer_total_seconds": total,
        "external_launcher_wall_seconds": wall,
        "derived_prefix_online_fps": SOURCE_COUNT / online,
        "mapper_torch_peak_gpu_gib": _finite_float(
            runtime.get("peak_gpu_memory"), f"{arm} mapper peak GPU memory"
        ),
        "peak_cpu_memory_gib": _finite_float(
            runtime.get("peak_cpu_memory"), f"{arm} peak CPU memory"
        ),
    }


def _read_metrics(arm_root: Path, scene_root: Path, arm: str) -> dict[str, Any]:
    result = _load_json(
        scene_root / "psnr" / "after_refine" / "final_result.json",
        f"{arm} final rendering metrics",
    )
    _expect(result.get("metric_scope"), "clear_gt_prefix_smoke", f"{arm} metric scope")
    _expect(result.get("num_evaluated_frames"), len(EXPECTED_CLEAR_GT), f"{arm} metric frame count")
    evaluated = tuple(int(value) for value in result.get("evaluated_source_indices", ()))
    _expect(evaluated, EXPECTED_CLEAR_GT, f"{arm} evaluated source indices")
    metrics = {
        "psnr_db": _finite_float(result.get("mean_psnr"), f"{arm} PSNR"),
        "ssim": _finite_float(result.get("mean_ssim"), f"{arm} SSIM"),
        "lpips": _finite_float(result.get("mean_lpips"), f"{arm} LPIPS"),
        "depth_l1": _finite_float(result.get("mean_depthl1"), f"{arm} depth L1"),
    }

    log_path = _require_file(arm_root / "launch.log", f"{arm} launch log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    exit_codes = re.findall(r"\[launcher\] exit_code=(-?\d+)", log_text)
    _expect(exit_codes, ["0"], f"{arm} persistent log exit marker")
    log_matches = list(FINAL_LOG_RE.finditer(log_text))
    _expect(len(log_matches), 1, f"{arm} final metric log-line count")
    logged = {key: float(value) for key, value in log_matches[0].groupdict().items()}
    log_keys = {
        "psnr_db": "psnr",
        "ssim": "ssim",
        "lpips": "lpips",
        "depth_l1": "depth_l1",
    }
    for key, value in metrics.items():
        if not math.isclose(
            value, logged[log_keys[key]], rel_tol=0.0, abs_tol=1e-10
        ):
            raise ContractError(f"{arm} {key} disagrees between JSON and launch.log")

    refinement = _load_json(
        scene_root / "refinement_checkpoint_metrics.json",
        f"{arm} refinement checkpoint metrics",
    )
    _expect(refinement.get("metric_scope"), "clear_gt_prefix_smoke", f"{arm} checkpoint metric scope")
    _expect(refinement.get("test_metric_used_for_selection"), False, f"{arm} checkpoint selection leakage")
    checkpoints = refinement.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ContractError(f"{arm} refinement checkpoints are not a list")
    iterations = tuple(int(entry.get("iteration", -1)) for entry in checkpoints)
    _expect(iterations, EXPECTED_REFINEMENT_CHECKPOINTS, f"{arm} refinement checkpoint iterations")
    for entry in checkpoints:
        _expect(int(entry.get("total_iterations", -1)), 100, f"{arm} checkpoint total iterations")
        _expect(int(entry.get("num_frames", -1)), len(EXPECTED_CLEAR_GT), f"{arm} checkpoint frame count")
        _expect(
            tuple(int(value) for value in entry.get("evaluated_source_indices", ())),
            EXPECTED_CLEAR_GT,
            f"{arm} checkpoint clear-GT sources",
        )
    selected = refinement.get("selected_checkpoint") or {}
    _expect(int(selected.get("iteration", -1)), 100, f"{arm} selected refinement iteration")
    if not math.isclose(
        metrics["psnr_db"],
        _finite_float(selected.get("mean_psnr_db"), f"{arm} selected-checkpoint PSNR"),
        rel_tol=0.0,
        abs_tol=1e-4,
    ):
        raise ContractError(f"{arm} final PSNR disagrees with selected checkpoint")

    rendered = tuple(
        sorted(
            int(path.stem.removeprefix("kf_"))
            for path in (scene_root / "after_refine" / "rendered_frames").glob("kf_*.png")
        )
    )
    _expect(rendered, EXPECTED_CLEAR_GT, f"{arm} rendered clear-GT sources")
    return {
        "metric_scope": "clear_gt_prefix_smoke",
        "evaluated_frame_count": len(evaluated),
        "evaluated_source_indices": list(evaluated),
        "selected_refinement_iteration": 100,
        **metrics,
    }


def _read_keyframes(scene_root: Path, final_metrics: Mapping[str, Any], arm: str) -> dict[str, Any]:
    path = _require_file(scene_root / "video.npz", f"{arm} saved keyframe video")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "timestamps" not in archive or "poses" not in archive:
                raise ContractError(f"{arm} video.npz lacks timestamps/poses")
            raw = np.asarray(archive["timestamps"], dtype=np.float64)
            pose_count = int(np.asarray(archive["poses"]).shape[0])
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot read {arm} keyframe video: {path}") from error
    if not np.all(np.isfinite(raw)) or not np.allclose(raw, np.rint(raw), atol=1e-5):
        raise ContractError(f"{arm} keyframe timestamps are not finite integers")
    indices = tuple(int(value) for value in np.rint(raw).tolist())
    if indices != tuple(sorted(set(indices))):
        raise ContractError(f"{arm} keyframe indices are not unique and ordered: {indices}")
    if not indices or indices[0] < 0 or indices[-1] >= SOURCE_COUNT:
        raise ContractError(f"{arm} keyframes fall outside source prefix: {indices}")
    if not set(EXPECTED_CLEAR_GT).issubset(indices):
        missing = sorted(set(EXPECTED_CLEAR_GT) - set(indices))
        raise ContractError(f"{arm} keyframes omit evaluated clear-GT sources: {missing}")
    _expect(pose_count, len(indices), f"{arm} keyframe pose count")
    # The final rendering record is an independent count written by evaluation.
    result = _load_json(
        scene_root / "psnr" / "after_refine" / "final_result.json",
        f"{arm} final rendering metrics",
    )
    _expect(int(result.get("total_keyframes", -1)), len(indices), f"{arm} reported total keyframes")
    return {
        "count": len(indices),
        "source_indices": list(indices),
        "evaluated_clear_gt_is_subset": True,
    }


def _latency_summary(values: Sequence[float], label: str) -> dict[str, float]:
    if not values:
        raise ContractError(f"{label} has no latency samples")
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(array).all()) or bool((array < 0.0).any()):
        raise ContractError(f"{label} has invalid latency samples")
    return {
        "total_ms": float(array.sum()),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _read_frontend(
    arm_root: Path, keyframe_indices: Sequence[int], arm: str
) -> dict[str, Any]:
    text = _require_file(arm_root / "launch.log", f"{arm} launch log").read_text(
        encoding="utf-8", errors="replace"
    )
    if arm == "baseline":
        current_frame: Optional[int] = None
        events: list[tuple[int, float]] = []
        for line in text.splitlines():
            frame_match = TRACKING_FRAME_RE.search(line)
            if frame_match is not None:
                current_frame = int(frame_match.group(1))
            latency_match = EVSSM_LATENCY_RE.search(line)
            if latency_match is not None:
                if current_frame is None:
                    raise ContractError("EVSSM latency precedes its source-frame marker")
                events.append((current_frame, float(latency_match.group(1))))
        if not events:
            raise ContractError("baseline log has no EVSSM inference records")
        frames = [frame for frame, _ in events]
        if frames != sorted(set(frames)) or not set(frames).issubset(keyframe_indices):
            raise ContractError(f"baseline EVSSM invocation frames are invalid: {frames}")
        return {
            "inference_scope": "blurry_selected_keyframes_only",
            "inference_count": len(events),
            "inference_source_indices": frames,
            "applied_deblur_count": len(events),
            "logged_inference_latency": _latency_summary(
                [latency for _, latency in events], "EVSSM inference"
            ),
        }

    stream_events = [
        {
            "frame": int(match.group("frame")),
            "gain": float(match.group("gain")),
            "replace": match.group("replace") == "True",
            "latency": float(match.group("latency")),
        }
        for match in TURTLE_STREAM_RE.finditer(text)
    ]
    frames = [event["frame"] for event in stream_events]
    _expect(frames, list(range(SOURCE_COUNT)), "TURTLE streaming decision frames")
    latencies = [event["latency"] for event in stream_events]
    replacements = [event for event in stream_events if event["replace"]]

    current_frame = None
    cache_events: list[tuple[int, float]] = []
    for line in text.splitlines():
        frame_match = TRACKING_FRAME_RE.search(line)
        if frame_match is not None:
            current_frame = int(frame_match.group(1))
        latency_match = TURTLE_CACHE_LATENCY_RE.search(line)
        if latency_match is not None:
            if current_frame is None:
                raise ContractError("TURTLE cache latency precedes its frame marker")
            cache_events.append((current_frame, float(latency_match.group(1))))
    cache_frames = [frame for frame, _ in cache_events]
    if cache_frames != sorted(set(cache_frames)) or not set(cache_frames).issubset(
        keyframe_indices
    ):
        raise ContractError(f"TURTLE keyframe cache frames are invalid: {cache_frames}")
    return {
        "inference_scope": "every_source_frame_streaming",
        "inference_count": len(stream_events),
        "inference_source_first": frames[0],
        "inference_source_last": frames[-1],
        "gated_replace_count": len(replacements),
        "gated_keep_raw_count": len(stream_events) - len(replacements),
        "logged_inference_latency_warmup_inclusive": _latency_summary(
            latencies, "TURTLE streaming inference"
        ),
        "logged_inference_latency_steady_frames_1_to_220": _latency_summary(
            latencies[1:], "TURTLE steady streaming inference"
        ),
        "cached_blurry_keyframe_lookup_count": len(cache_events),
        "cached_blurry_keyframe_source_indices": cache_frames,
        "logged_cached_keyframe_lookup_latency": _latency_summary(
            [latency for _, latency in cache_events], "TURTLE keyframe cache lookup"
        ),
    }


def _read_ate_file(path: Path, label: str) -> float:
    text = _require_file(path, label).read_text(encoding="utf-8", errors="replace")
    match = STATISTICS_RE.search(text)
    if match is None:
        raise ContractError(f"cannot parse {label}: {path}")
    try:
        statistics = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as error:
        raise ContractError(f"invalid statistics in {label}: {path}") from error
    if not isinstance(statistics, dict):
        raise ContractError(f"statistics are not a mapping in {label}: {path}")
    return _finite_float(statistics.get("rmse"), f"{label} RMSE")


def _read_trajectory(scene_root: Path, keyframe_count: int, arm: str) -> dict[str, Any]:
    trajectory_root = scene_root / "traj"
    full_ate = _read_ate_file(
        trajectory_root / "metrics_full_traj.txt", f"{arm} full-trajectory metrics"
    )
    keyframe_ate = _read_ate_file(
        trajectory_root / "metrics_kf_traj.txt", f"{arm} keyframe-trajectory metrics"
    )
    archive_path = _require_file(
        trajectory_root / "traj_full_full_traj.npz", f"{arm} full trajectory archive"
    )
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
            estimated = np.asarray(archive["traj_est_poses"])
            reference = np.asarray(archive["traj_ref_poses"])
            uses_gt = bool(np.asarray(archive["uses_ground_truth_pose"]).item())
            pose_source = str(np.asarray(archive["pose_source"]).item())
            archived_ate = float(np.asarray(archive["ate_rmse"]).item())
    except (KeyError, OSError, ValueError) as error:
        raise ContractError(f"cannot audit {arm} full trajectory archive") from error
    expected_timestamps = np.arange(SOURCE_COUNT, dtype=np.float64)
    if timestamps.shape != expected_timestamps.shape or not np.allclose(
        timestamps, expected_timestamps, atol=1e-8
    ):
        raise ContractError(f"{arm} full trajectory does not cover source frames 0..220")
    _expect(int(estimated.shape[0]), SOURCE_COUNT, f"{arm} estimated trajectory length")
    _expect(int(reference.shape[0]), SOURCE_COUNT, f"{arm} reference trajectory length")
    _expect(uses_gt, False, f"{arm} trajectory uses ground-truth poses")
    if "droid" not in pose_source.lower():
        raise ContractError(f"{arm} unexpected trajectory pose source: {pose_source!r}")
    if not math.isclose(full_ate, archived_ate, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(f"{arm} archived and text full-trajectory ATE disagree")
    return {
        "full_trajectory_frame_count": SOURCE_COUNT,
        "full_trajectory_ate_rmse_m": full_ate,
        "keyframe_trajectory_frame_count": keyframe_count,
        "keyframe_trajectory_ate_rmse_m": keyframe_ate,
        "pose_source": pose_source,
        "uses_ground_truth_pose": False,
    }


def _git(repo: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ContractError(f"cannot inspect TURTLE repository: {repo}") from error
    return completed.stdout.strip()


def _verified_file_record(path_value: Any, expected_sha: str, label: str) -> dict[str, Any]:
    path = Path(str(path_value or "")).expanduser().resolve()
    observed = _sha256_file(path)
    _expect(observed, expected_sha, f"{label} file SHA-256")
    return {"path": str(path), "sha256": observed}


def _validate_provenance(
    root: Path,
    configs: Mapping[str, Mapping[str, Any]],
    preflights: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _expect(preflights["baseline"], preflights["turtle"], "per-arm preflight records")
    preflight = preflights["baseline"]
    artifacts = preflight.get("artifacts") or {}
    droid = artifacts.get("droid") or {}
    omnidata = artifacts.get("omnidata") or {}
    evssm = artifacts.get("evssm") or {}
    _expect(droid.get("sha256"), EXPECTED_DROID_SHA256, "preflight DROID digest")
    _expect(omnidata.get("sha256"), EXPECTED_OMNIDATA_SHA256, "preflight Omnidata digest")
    _expect(evssm.get("sha256"), EXPECTED_EVSSM_SHA256, "preflight EVSSM digest")
    verified_artifacts = {
        "droid": _verified_file_record(
            droid.get("path"), EXPECTED_DROID_SHA256, "DROID checkpoint"
        ),
        "omnidata": _verified_file_record(
            omnidata.get("path"), EXPECTED_OMNIDATA_SHA256, "Omnidata checkpoint"
        ),
        "evssm": _verified_file_record(
            evssm.get("path"), EXPECTED_EVSSM_SHA256, "EVSSM checkpoint"
        ),
        "turtle_config": _verified_file_record(
            _nested(configs["turtle"], "deblur.turtle_config"),
            EXPECTED_TURTLE_CONFIG_SHA256,
            "official TURTLE config",
        ),
        "turtle_checkpoint": _verified_file_record(
            _nested(configs["turtle"], "deblur.turtle_checkpoint"),
            EXPECTED_TURTLE_CHECKPOINT_SHA256,
            "official TURTLE GoPro checkpoint",
        ),
    }
    turtle = artifacts.get("official_turtle") or {}
    expected_turtle = {
        "origin": "https://github.com/Ascend-Research/Turtle.git",
        "commit": EXPECTED_TURTLE_COMMIT,
        "architecture_sha256": EXPECTED_TURTLE_ARCH_SHA256,
        "config_sha256": EXPECTED_TURTLE_CONFIG_SHA256,
        "checkpoint_sha256": EXPECTED_TURTLE_CHECKPOINT_SHA256,
        "checkpoint_kind": "official_gopro",
        "strict_cpu_load": True,
        "cache_contract": "official_kv_8_incremental",
    }
    for field, expected in expected_turtle.items():
        _expect(turtle.get(field), expected, f"preflight official TURTLE {field}")
    turtle_repo = Path(
        str(_nested(configs["turtle"], "deblur.turtle_repo"))
    ).expanduser().resolve()
    _expect(_git(turtle_repo, "rev-parse", "HEAD"), EXPECTED_TURTLE_COMMIT, "TURTLE checkout HEAD")
    checkout_origin = _git(turtle_repo, "remote", "get-url", "origin").rstrip("/")
    normalized_origin = (
        checkout_origin[:-4] if checkout_origin.lower().endswith(".git") else checkout_origin
    )
    _expect(
        normalized_origin.lower(),
        "https://github.com/Ascend-Research/Turtle".lower(),
        "TURTLE checkout origin",
    )
    _expect(
        _git(turtle_repo, "status", "--porcelain", "--untracked-files=no"),
        "",
        "TURTLE tracked worktree status",
    )

    run_records: dict[str, Any] = {}
    for arm in ("baseline", "turtle"):
        arm_root = root / ARMS[arm]
        scene_root = arm_root / SCENE
        paths = {
            "preflight": arm_root / "preflight.json",
            "resolved_config": scene_root / "cfg.yaml",
            "launch_log": arm_root / "launch.log",
            "launcher_runtime": arm_root / "launcher_runtime.json",
            "runtime_stats": scene_root / "runtime_stats.json",
            "final_metrics": scene_root / "psnr" / "after_refine" / "final_result.json",
            "keyframe_video": scene_root / "video.npz",
            "full_trajectory": scene_root / "traj" / "traj_full_full_traj.npz",
        }
        run_records[arm] = {
            label: {"path": str(path), "sha256": _sha256_file(path)}
            for label, path in paths.items()
        }
    return {
        "artifact_file_bytes_sha256_verified": True,
        "artifacts": verified_artifacts,
        "official_turtle_repository": {
            **expected_turtle,
            "path": str(turtle_repo),
            "tracked_worktree_clean": True,
        },
        "preflight_resolved_config_sha256": {
            "baseline": preflight["paired_contract"]["baseline_resolved_sha256"],
            "turtle": preflight["paired_contract"]["turtle_resolved_sha256"],
        },
        "run_record_files": run_records,
    }


def _read_arm(root: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    arm_root = root / ARMS[arm]
    scene_root = arm_root / SCENE
    cfg = _load_yaml(scene_root / "cfg.yaml", f"{arm} resolved config")
    _validate_config(cfg, arm, arm_root)
    preflight = _load_json(arm_root / "preflight.json", f"{arm} preflight")
    _validate_preflight(preflight, cfg, arm)
    runtime = _read_runtime(arm_root, scene_root, arm)
    metrics = _read_metrics(arm_root, scene_root, arm)
    keyframes = _read_keyframes(scene_root, metrics, arm)
    frontend = _read_frontend(arm_root, keyframes["source_indices"], arm)
    trajectory = _read_trajectory(scene_root, keyframes["count"], arm)
    report = {
        "output_root": str(arm_root),
        "frontend": EXPECTED_FRONTENDS[arm],
        "runtime": runtime,
        "frontend_activity": frontend,
        "rendering_and_depth": metrics,
        "keyframes": keyframes,
        "trajectory": trajectory,
    }
    return report, cfg, preflight


def build_report(
    root: Path, observed_whole_device_mib: Optional[Mapping[str, int]] = None
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    arms: dict[str, Any] = {}
    configs: dict[str, dict[str, Any]] = {}
    preflights: dict[str, dict[str, Any]] = {}
    for arm in ("baseline", "turtle"):
        arms[arm], configs[arm], preflights[arm] = _read_arm(root, arm)

    comparison_configs = {
        arm: _without_runtime_metadata(configs[arm]) for arm in ("baseline", "turtle")
    }
    differences = _pair_differences(
        comparison_configs["baseline"], comparison_configs["turtle"]
    )
    unexpected = sorted(set(differences) - ALLOWED_PAIR_DIFFS)
    if unexpected:
        raise ContractError(
            "resolved paired configs differ outside frontend/artifact/output fields: "
            + ", ".join(unexpected)
        )
    preflight_differences = (
        preflights["baseline"].get("paired_contract", {})
        .get("allowed_resolved_config_differences", {})
    )
    _expect(differences, preflight_differences, "resolved config differences vs preflight")
    provenance = _validate_provenance(root, configs, preflights)

    baseline = arms["baseline"]
    turtle = arms["turtle"]
    br = baseline["runtime"]
    tr = turtle["runtime"]
    bm = baseline["rendering_and_depth"]
    tm = turtle["rendering_and_depth"]
    btraj = baseline["trajectory"]
    ttraj = turtle["trajectory"]
    keyframes_identical = (
        baseline["keyframes"]["source_indices"]
        == turtle["keyframes"]["source_indices"]
    )
    comparison = {
        "same_eleven_clear_gt_sources": (
            bm["evaluated_source_indices"] == tm["evaluated_source_indices"]
            == list(EXPECTED_CLEAR_GT)
        ),
        "keyframe_sets_identical": keyframes_identical,
        "baseline_only_keyframes": sorted(
            set(baseline["keyframes"]["source_indices"])
            - set(turtle["keyframes"]["source_indices"])
        ),
        "turtle_only_keyframes": sorted(
            set(turtle["keyframes"]["source_indices"])
            - set(baseline["keyframes"]["source_indices"])
        ),
        "runtime": {
            "online_speedup_baseline_over_turtle_x": br["official_timer_online_seconds"] / tr["official_timer_online_seconds"],
            "total_speedup_baseline_over_turtle_x": br["official_timer_total_seconds"] / tr["official_timer_total_seconds"],
            "external_wall_speedup_baseline_over_turtle_x": br["external_launcher_wall_seconds"] / tr["external_launcher_wall_seconds"],
            "online_seconds_delta_turtle_minus_baseline": tr["official_timer_online_seconds"] - br["official_timer_online_seconds"],
            "total_seconds_delta_turtle_minus_baseline": tr["official_timer_total_seconds"] - br["official_timer_total_seconds"],
            "external_wall_seconds_delta_turtle_minus_baseline": tr["external_launcher_wall_seconds"] - br["external_launcher_wall_seconds"],
        },
        "quality_delta_turtle_minus_baseline": {
            "psnr_db": tm["psnr_db"] - bm["psnr_db"],
            "ssim": tm["ssim"] - bm["ssim"],
            "lpips": tm["lpips"] - bm["lpips"],
            "depth_l1": tm["depth_l1"] - bm["depth_l1"],
            "full_trajectory_ate_rmse_m": ttraj["full_trajectory_ate_rmse_m"] - btraj["full_trajectory_ate_rmse_m"],
            "keyframe_trajectory_ate_rmse_m": ttraj["keyframe_trajectory_ate_rmse_m"] - btraj["keyframe_trajectory_ate_rmse_m"],
        },
    }
    report = {
        "schema": "unblur_slam.fr2_xyz_official_online_budget_paired_221_report.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(root),
        "scope": {
            "scene": SCENE,
            "source_first": 0,
            "source_last": 220,
            "source_count": SOURCE_COUNT,
            "clear_gt_metric_scope": "clear_gt_prefix_smoke",
            "clear_gt_frame_count": len(EXPECTED_CLEAR_GT),
            "clear_gt_source_indices": list(EXPECTED_CLEAR_GT),
            "published_online_optimization_budgets": {
                "initialization_iterations": 1050,
                "mapping_iterations_per_keyframe": 100,
                "tracking_iterations": 100,
            },
            "final_refinement_iterations": 100,
            "complete_three_sequence_paper_benchmark": False,
            "paper_26k_offline_refinement": False,
            "paper_table_6_fps_comparable": False,
        },
        "pair_contract": {
            "same_seed_resolution_dataset_and_nonfrontend_settings": True,
            "resolved_configs_identical_except_allowed_frontend_artifact_output_fields": True,
            "allowed_resolved_config_differences": differences,
        },
        "provenance": provenance,
        "arms": arms,
        "comparison": comparison,
        "interpretation_notes": [
            "Derived FPS is 221 divided by the repository online_inference_time field; it is a bounded-prefix diagnostic, not paper Table 6 FPS.",
            "online_inference_time ends after online mapping; total_inference_time also spans intervening termination work through final refinement; external launcher wall additionally includes setup and post-refinement output/evaluation.",
            "mapper_torch_peak_gpu_gib is the mapper process's torch peak-allocation statistic, not whole-device nvidia-smi occupancy.",
            "PSNR/SSIM/LPIPS/depth L1 use exactly eleven clear-GT frames; full-trajectory ATE uses all 221 source frames.",
            "LPIPS, depth L1, and ATE are lower-is-better; PSNR and SSIM are higher-is-better.",
            "This fr2_xyz clear-GT prefix contains sharp reference frames, so it primarily probes sharp-frame behavior and frontend overhead.",
            "Frontend log latencies have different invocation scopes: EVSSM is timed only on blurry selected keyframes, whereas TURTLE streaming inference is timed on every source frame.",
        ],
    }
    if observed_whole_device_mib is not None:
        values = {arm: int(observed_whole_device_mib[arm]) for arm in ARMS}
        if any(value <= 0 for value in values.values()):
            raise ContractError("observed whole-device GPU snapshots must be positive")
        report["external_gpu_monitoring"] = {
            "method": "intermittent nvidia-smi whole-device snapshots during the run",
            "continuous_sampling": False,
            "is_guaranteed_peak": False,
            "highest_observed_snapshot_mib": values,
            "caveat": "These are lower bounds on the true whole-device peak and are not comparable to mapper-only torch peak allocation.",
        }
    return report


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ("baseline", "turtle"):
        data = report["arms"][arm]
        runtime = data["runtime"]
        metrics = data["rendering_and_depth"]
        keyframes = data["keyframes"]
        trajectory = data["trajectory"]
        frontend = data["frontend_activity"]
        observed = (
            (report.get("external_gpu_monitoring") or {})
            .get("highest_observed_snapshot_mib", {})
            .get(arm, "")
        )
        if arm == "baseline":
            frontend_latency = frontend["logged_inference_latency"]
            frontend_apply_count = frontend["applied_deblur_count"]
        else:
            frontend_latency = frontend["logged_inference_latency_warmup_inclusive"]
            frontend_apply_count = frontend["gated_replace_count"]
        rows.append(
            {
                "arm": arm,
                "frontend": data["frontend"],
                "source_frames": SOURCE_COUNT,
                "clear_gt_frames": len(EXPECTED_CLEAR_GT),
                "clear_gt_source_indices": ";".join(map(str, EXPECTED_CLEAR_GT)),
                "keyframe_count": keyframes["count"],
                "keyframe_source_indices": ";".join(map(str, keyframes["source_indices"])),
                "online_seconds": runtime["official_timer_online_seconds"],
                "total_seconds": runtime["official_timer_total_seconds"],
                "external_wall_seconds": runtime["external_launcher_wall_seconds"],
                "derived_prefix_online_fps": runtime["derived_prefix_online_fps"],
                "mapper_torch_peak_gpu_gib": runtime["mapper_torch_peak_gpu_gib"],
                "peak_cpu_memory_gib": runtime["peak_cpu_memory_gib"],
                "highest_observed_intermittent_whole_device_mib": observed,
                "frontend_inference_count": frontend["inference_count"],
                "frontend_applied_or_replaced_count": frontend_apply_count,
                "frontend_logged_latency_mean_ms": frontend_latency["mean_ms"],
                "frontend_logged_latency_p95_ms": frontend_latency["p95_ms"],
                "frontend_logged_latency_total_ms": frontend_latency["total_ms"],
                "psnr_db": metrics["psnr_db"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "depth_l1": metrics["depth_l1"],
                "full_trajectory_ate_rmse_m": trajectory["full_trajectory_ate_rmse_m"],
                "keyframe_trajectory_ate_rmse_m": trajectory["keyframe_trajectory_ate_rmse_m"],
                "final_refinement_iterations": 100,
                "paper_table_6_comparable": False,
                "paper_26k_refinement": False,
            }
        )
    return rows


def write_report(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paired_221_audit.json"
    csv_path = output_dir / "paired_221_metrics.csv"
    for path in (json_path, csv_path):
        if path.exists() or path.is_symlink():
            raise ContractError(f"refusing to overwrite report output: {path}")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = _csv_rows(report)
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write paired_221_audit.json and paired_221_metrics.csv after validation",
    )
    parser.add_argument(
        "--baseline-observed-gpu-mib",
        type=int,
        help="highest intermittent nvidia-smi whole-device snapshot for baseline",
    )
    parser.add_argument(
        "--turtle-observed-gpu-mib",
        type=int,
        help="highest intermittent nvidia-smi whole-device snapshot for TURTLE",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        observed_values = (
            args.baseline_observed_gpu_mib,
            args.turtle_observed_gpu_mib,
        )
        if (observed_values[0] is None) != (observed_values[1] is None):
            raise ContractError("provide both observed-GPU snapshot values or neither")
        observed = None
        if observed_values[0] is not None:
            observed = {
                "baseline": int(observed_values[0]),
                "turtle": int(observed_values[1]),
            }
        report = build_report(args.root, observed_whole_device_mib=observed)
        if args.output_dir is None:
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        else:
            json_path, csv_path = write_report(report, args.output_dir)
            print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))
        return 0
    except ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
