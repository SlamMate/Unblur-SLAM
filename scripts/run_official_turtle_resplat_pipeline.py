#!/usr/bin/env python3
"""Preflight or run the audited official TURTLE -> cvg/ReSplat smoke.

The input to this runner is an immutable TURTLE stream-materialization
manifest.  The manifest must prove that the official recurrent frontend was
advanced on every TUM source frame 0..2764.  Only after those 2765 consecutive
cache updates may the fixed 42 DROID keyframes be selected for the standalone
official cvg/ReSplat experiment.

This runner deliberately has no integration point with Unblur-SLAM's legacy
``mapping.resplat`` residual-priority sampler or its historical
``causal_evssm`` adapter.  It invokes only these two audited bridge scripts:

* ``export_tum_official_resplat_scene.py``
* ``run_paired_official_resplat_smoke.py``

Preflight is CPU-only.  A real run exposes physical GPU 1 as the sole CUDA
device, so the official paired runner addresses it as ``cuda:0``.  Every
output directory is installed atomically by its owning stage and an existing
path is never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "unblur_slam.official_turtle_resplat_pipeline.v1"
TURTLE_MANIFEST_SCHEMA = "unblur_slam.turtle_stream_materialization.v1"
PIPELINE_RUN_SCHEMA = "unblur_slam.official_turtle_resplat_pipeline_run.v1"

OFFICIAL_TURTLE_ORIGIN = "https://github.com/Ascend-Research/Turtle"
OFFICIAL_TURTLE_COMMIT = "7094f4221b64ad0962b4f27ff1b76d788836e804"
OFFICIAL_TURTLE_ARCH_SHA256 = (
    "4d19c676f92574dbad493eb591312fdeaf2b3b519f57410af2ed95fdbef5f058"
)
OFFICIAL_TURTLE_CONFIG_SHA256 = (
    "123b07de8d3f329769562e2f943e08fdf86c576c405634bad199ced95b25aa23"
)
OFFICIAL_TURTLE_GOPRO_SHA256 = (
    "10334b3e81d0416bcde5ccaca960dc81dbfb5b6d23e53fadaf7896d72b580c82"
)
OFFICIAL_TURTLE_CACHE_CONTRACT = "official_kv_8_incremental"
OFFICIAL_TURTLE_CACHE_NON_NULL_MASK = (
    False, False, False, True, True, True, True, True,
)

OFFICIAL_RESPLAT_ORIGIN = "https://github.com/cvg/resplat"
OFFICIAL_RESPLAT_COMMIT = "cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1"
OFFICIAL_RESPLAT_PRESET = "dl3dv_8v_256x448_small"
OFFICIAL_RESPLAT_CHECKPOINT_SHA256 = (
    "548993fede0d9536d2d914cbe51e0ebea0ad6f88c898c909e02127d59bb2be9a"
)

SOURCE_FIRST = 0
SOURCE_LAST = 2764
STREAM_COUNT = SOURCE_LAST - SOURCE_FIRST + 1
TRACKER_RAW_SIZE = (640, 480)
TRACKER_RESIZE_BEFORE_CROP = (528, 400)
TRACKER_CROP_EDGES = (8, 8, 8, 8)
TRACKER_OUTPUT_SIZE = (512, 384)
TRACKER_K = (
    (429.7425, 0.0, 260.2075),
    (0.0, 434.1666666666667, 200.08333333333334),
    (0.0, 0.0, 1.0),
)
TRACKER_DISTORTION = (0.2312, -0.7849, -0.0033, -0.0001, 0.9172)
TRACKER_PREPROCESSING = (
    "cv2.imread_color",
    "cv2.undistort_same_K",
    "cv2.resize_inter_linear_528x400",
    "crop_l8_r8_t8_b8",
    "bgr_to_rgb",
    "uint8_to_float32_div_255",
)
KEYFRAME_INDICES = (
    0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220, 319, 374, 407,
    435, 470, 483, 523, 568, 704, 750, 789, 827, 926, 1004, 1160,
    1251, 1342, 1409, 1460, 1553, 1692, 1795, 1889, 1978, 2055,
    2206, 2282, 2358, 2425, 2590, 2764,
)
NUM_CONTEXT = 8
NUM_TARGET = len(KEYFRAME_INDICES) - NUM_CONTEXT
NUM_REFINE = 4
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_tum_official_resplat_scene.py"
PAIRED_SCRIPT = REPO_ROOT / "scripts" / "run_paired_official_resplat_smoke.py"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path | str, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return source, value


def _require_sha256(value: object, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a full lowercase SHA-256")
    return digest


def _require_false(value: object, label: str) -> None:
    if value is not False:
        raise ValueError(f"{label} must be the JSON boolean false")


def _normalize_git_url(value: str) -> str:
    normalized = str(value).strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized[len("git@github.com:"):]
    return normalized.lower()


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
        raise ValueError(f"cannot inspect Git repository: {repo}") from error
    return completed.stdout.strip()


def inspect_pinned_repo(
    path: Path | str,
    *,
    expected_origin: str,
    expected_commit: str,
    label: str,
) -> dict[str, Any]:
    repo = Path(path).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"{label} repository does not exist: {repo}")
    origin = _git(repo, "remote", "get-url", "origin")
    commit = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if _normalize_git_url(origin) != _normalize_git_url(expected_origin):
        raise ValueError(f"{label} origin is not official: {origin}")
    if commit != expected_commit:
        raise ValueError(
            f"{label} commit mismatch: expected {expected_commit}, got {commit}"
        )
    if dirty:
        raise ValueError(f"{label} checkout has tracked modifications")
    return {
        "path": str(repo),
        "origin": origin,
        "expected_origin": expected_origin,
        "commit": commit,
        "tracked_worktree_clean": True,
    }


def _artifact_record(value: object, label: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object with path and sha256")
    path = Path(str(value.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    expected = _require_sha256(value.get("sha256"), f"{label} declared sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return path, actual


def _record_path(value: object, manifest_root: Path, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    raw = str(value.get("path", "")).strip()
    if not raw:
        raise ValueError(f"{label}.path is empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_root / path
    return path.resolve()


def _verify_hashed_record(
    value: object,
    manifest_root: Path,
    label: str,
    *,
    verify_file: bool,
) -> tuple[Path, str]:
    path = _record_path(value, manifest_root, label)
    assert isinstance(value, Mapping)
    digest = _require_sha256(value.get("sha256"), f"{label}.sha256")
    if verify_file:
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"{label} SHA-256 mismatch: expected {digest}, got {actual}")
    return path, digest


def _stream_range(value: object) -> tuple[int, int]:
    if isinstance(value, Mapping):
        first = value.get("first", value.get("start_source_index", -1))
        last = value.get("last", value.get("end_source_index", -1))
        if "inclusive" in value and value.get("inclusive") is not True:
            raise ValueError("stream.processed_range must be inclusive")
        return int(first), int(last)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            return int(value[0]), int(value[1])
    raise ValueError("stream.processed_range must be {first,last} or [first,last]")


def _cache_count(stream: Mapping[str, Any], kind: str) -> int:
    for key in (
        f"{kind}_count",
        f"{kind}_cache_count",
        f"{kind}_tensor_count",
        f"{kind}_cache_slots",
    ):
        if key in stream:
            return int(stream[key])
    cache = stream.get("cache")
    if isinstance(cache, Mapping):
        for key in (f"{kind}_count", f"{kind}_tensor_count"):
            if key in cache:
                return int(cache[key])
    raise ValueError(f"stream must declare {kind} cache tensor count")


def _source_frames_csv(source: Mapping[str, Any], root: Path) -> tuple[Path, str]:
    value = source.get("frames_csv")
    if isinstance(value, Mapping):
        return _verify_hashed_record(value, root, "source.frames_csv", verify_file=True)
    if value is not None:
        record = {"path": value, "sha256": source.get("frames_csv_sha256")}
    else:
        # The materializer's v1 source record is itself the CSV path+hash
        # record; no alternate pose or GT artifact is opened.
        record = {"path": source.get("path"), "sha256": source.get("sha256")}
    return _verify_hashed_record(record, root, "source.frames_csv", verify_file=True)


def _manifest_repo_record(value: object, label: str) -> tuple[Path, str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    path = Path(str(value.get("path", ""))).expanduser().resolve()
    origin = str(value.get("origin", ""))
    commit = str(value.get("commit", ""))
    return path, origin, commit


def validate_turtle_stream_manifest(
    payload: Mapping[str, Any],
    manifest_path: Path | str,
    *,
    expected_frames_csv: Path | str,
    expected_frames_csv_sha256: str,
    expected_turtle_repo: Path | str,
    expected_checkpoint: Path | str,
    verify_frame_files: bool = True,
    verify_all_step_inputs: bool = True,
) -> dict[str, Any]:
    """Validate the consecutive official-TURTLE stream and sparse emission."""

    manifest = Path(manifest_path).expanduser().resolve()
    root = manifest.parent
    if payload.get("schema") != TURTLE_MANIFEST_SCHEMA:
        raise ValueError(
            f"TURTLE manifest schema must be {TURTLE_MANIFEST_SCHEMA!r}"
        )

    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("TURTLE manifest source must be an object")
    _require_false(source.get("uses_ground_truth_pose"), "source.uses_ground_truth_pose")
    _require_false(source.get("poses_consumed_by_turtle"), "source.poses_consumed_by_turtle")
    _require_false(source.get("depth_consumed_by_turtle"), "source.depth_consumed_by_turtle")
    _require_false(
        source.get("ground_truth_images_consumed_by_turtle"),
        "source.ground_truth_images_consumed_by_turtle",
    )
    if source.get("pose_source_declared_but_not_consumed") != "droid_traj_est_not_align":
        raise ValueError("TURTLE source must identify, but not consume, first-pass DROID poses")
    csv_path, csv_sha = _source_frames_csv(source, root)
    expected_csv = Path(expected_frames_csv).expanduser().resolve()
    if csv_path != expected_csv:
        raise ValueError(f"TURTLE source CSV mismatch: {csv_path} != {expected_csv}")
    if csv_sha != _require_sha256(
        expected_frames_csv_sha256, "configured frames CSV sha256"
    ):
        raise ValueError("TURTLE source CSV digest disagrees with pipeline config")

    turtle = payload.get("turtle")
    if not isinstance(turtle, Mapping):
        raise ValueError("TURTLE manifest turtle must be an object")
    repo_value = turtle.get("repository", turtle.get("repo"))
    repo, origin, commit = _manifest_repo_record(repo_value, "turtle.repository")
    if repo != Path(expected_turtle_repo).expanduser().resolve():
        raise ValueError("TURTLE manifest repository path disagrees with config")
    if _normalize_git_url(origin) != _normalize_git_url(OFFICIAL_TURTLE_ORIGIN):
        raise ValueError("TURTLE manifest origin is not Ascend-Research/Turtle")
    if commit != OFFICIAL_TURTLE_COMMIT:
        raise ValueError("TURTLE manifest commit is not the pinned official revision")
    if turtle.get("cache_contract") != OFFICIAL_TURTLE_CACHE_CONTRACT:
        raise ValueError("TURTLE manifest does not declare official incremental K/V cache")

    artifact_expectations = {
        "architecture": OFFICIAL_TURTLE_ARCH_SHA256,
        "config": OFFICIAL_TURTLE_CONFIG_SHA256,
        "checkpoint": OFFICIAL_TURTLE_GOPRO_SHA256,
    }
    artifact_audit: dict[str, Any] = {}
    for label, expected_hash in artifact_expectations.items():
        path, digest = _verify_hashed_record(
            turtle.get(label), root, f"turtle.{label}", verify_file=True
        )
        if digest != expected_hash:
            raise ValueError(f"TURTLE {label} is not the pinned official GoPro artifact")
        artifact_audit[label] = {"path": str(path), "sha256": digest}
    if artifact_audit["checkpoint"]["path"] != str(
        Path(expected_checkpoint).expanduser().resolve()
    ):
        raise ValueError("TURTLE manifest checkpoint path disagrees with config")
    checkpoint_record = turtle.get("checkpoint")
    assert isinstance(checkpoint_record, Mapping)
    checkpoint_metadata = checkpoint_record.get("metadata")
    if not isinstance(checkpoint_metadata, Mapping) or checkpoint_metadata.get("kind") != "official_gopro":
        raise ValueError("TURTLE manifest checkpoint must be the official GoPro release")

    stream = payload.get("stream")
    if not isinstance(stream, Mapping):
        raise ValueError("TURTLE manifest stream must be an object")
    expected_stream = list(range(SOURCE_FIRST, SOURCE_LAST + 1))
    try:
        processed = [int(value) for value in stream.get("processed_source_indices", ())]
    except (TypeError, ValueError) as error:
        raise ValueError("stream.processed_source_indices must be integer source indices") from error
    if processed != expected_stream:
        raise ValueError(
            "formal TURTLE stream must process every consecutive source index "
            "0..2764; a sparse 42-keyframe stream is invalid"
        )
    if _stream_range(stream.get("processed_range")) != (SOURCE_FIRST, SOURCE_LAST):
        raise ValueError("stream.processed_range must be exactly 0..2764")
    integer_contract = {
        "processed_count": STREAM_COUNT,
        "step_count": STREAM_COUNT,
        "cache_updates": STREAM_COUNT,
        "reset_count": 1,
    }
    for key, expected in integer_contract.items():
        if int(stream.get(key, -1)) != expected:
            raise ValueError(f"stream.{key} must equal {expected}")
    if stream.get("strictly_increasing_source_indices") is not True:
        raise ValueError("TURTLE stream must declare strictly increasing source indices")
    if stream.get("strictly_increasing_timestamps") is not True:
        raise ValueError("TURTLE stream must declare strictly increasing timestamps")
    if stream.get("gaps_skipped") is not False:
        raise ValueError("TURTLE stream must declare gaps_skipped=false")
    if stream.get("first_pair") != "self":
        raise ValueError("TURTLE stream first pair must repeat the first frame (self)")
    if stream.get("persistent_kv") is not True:
        raise ValueError("TURTLE stream must use persistent_kv=true")
    if stream.get("one_step_per_source_frame") is not True:
        raise ValueError("TURTLE stream must take exactly one step per source frame")
    if stream.get("cache_contract") != OFFICIAL_TURTLE_CACHE_CONTRACT:
        raise ValueError("stream cache contract disagrees with official TURTLE")
    if _cache_count(stream, "k") != 8 or _cache_count(stream, "v") != 8:
        raise ValueError("official TURTLE stream must carry eight K and eight V tensors")
    expected_mask = list(OFFICIAL_TURTLE_CACHE_NON_NULL_MASK)
    if (
        int(stream.get("k_cache_non_null_count", -1)) != 5
        or int(stream.get("v_cache_non_null_count", -1)) != 5
        or stream.get("official_gopro_cache_non_null_mask") != expected_mask
    ):
        raise ValueError(
            "official GoPro TURTLE cache must use the exact sparse 8-slot mask "
            f"{expected_mask}"
        )
    reset_events = stream.get("reset_events")
    if not isinstance(reset_events, list) or len(reset_events) != 1:
        raise ValueError("TURTLE stream must contain exactly one initial reset event")
    reset = reset_events[0]
    if (
        not isinstance(reset, Mapping)
        or int(reset.get("before_source_index", -1)) != SOURCE_FIRST
        or int(reset.get("reset_ordinal", -1)) != 1
    ):
        raise ValueError("the only TURTLE reset must occur before source_index=0")

    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("TURTLE manifest selection must be an object")
    emitted = tuple(int(value) for value in selection.get("emitted_source_indices", ()))
    count = selection.get("emitted_count", selection.get("count", -1))
    if emitted != KEYFRAME_INDICES or int(count) != len(KEYFRAME_INDICES):
        raise ValueError("TURTLE manifest must emit exactly the fixed 42 DROID keyframes")

    steps = stream.get("steps")
    if not isinstance(steps, list) or len(steps) != STREAM_COUNT:
        raise ValueError(f"stream.steps must contain exactly {STREAM_COUNT} records")
    emitted_set = set(KEYFRAME_INDICES)
    verified_input_paths: dict[Path, str] = {}
    step_by_source: dict[int, Mapping[str, Any]] = {}
    previous_timestamp: Optional[float] = None
    for expected_index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"stream.steps[{expected_index}] must be an object")
        source_index = int(step.get("source_index", -1))
        if source_index != expected_index or int(step.get("step_index", -1)) != expected_index:
            raise ValueError("TURTLE stream steps must be gap-free and position-aligned")
        emitted_now = step.get("emitted_png", step.get("emitted"))
        if emitted_now is not (source_index in emitted_set):
            raise ValueError(f"stream step {source_index} has incorrect emitted flag")
        try:
            timestamp = float(step.get("timestamp"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"stream step {source_index} timestamp is invalid") from error
        if not math.isfinite(timestamp) or (
            previous_timestamp is not None and timestamp <= previous_timestamp
        ):
            raise ValueError("TURTLE step timestamps must be finite and strictly increasing")
        previous_timestamp = timestamp
        input_value = step.get("input")
        if not isinstance(input_value, Mapping):
            input_value = {
                "path": step.get("input_path"),
                "sha256": step.get("input_file_sha256"),
            }
        input_path, input_sha = _verify_hashed_record(
            input_value,
            root,
            f"stream.steps[{source_index}].input",
            verify_file=False,
        )
        if verify_all_step_inputs:
            previous = verified_input_paths.get(input_path)
            if previous is None:
                if not input_path.is_file():
                    raise FileNotFoundError(f"stream input is missing: {input_path}")
                actual = sha256_file(input_path)
                if actual != input_sha:
                    raise ValueError(f"stream input SHA mismatch at source {source_index}")
                verified_input_paths[input_path] = actual
            elif previous != input_sha:
                raise ValueError("one stream input path is declared with conflicting hashes")
        input_pixel_sha = step.get(
            "input_rgb_u8_pixel_sha256", step.get("preprocessed_rgb_sha256")
        )
        output_pixel_sha = step.get(
            "output_rgb_u8_pixel_sha256", step.get("output_rgb_u8_sha256")
        )
        _require_sha256(input_pixel_sha, f"stream step {source_index} input pixel sha256")
        _require_sha256(output_pixel_sha, f"stream step {source_index} output pixel sha256")
        if (
            step.get("cache_present_before") is not (source_index > SOURCE_FIRST)
            or step.get("cache_present_after") is not True
            or int(step.get("k_cache_slots_after", -1)) != 8
            or int(step.get("v_cache_slots_after", -1)) != 8
            or int(step.get("k_cache_non_null_count_after", -1)) != 5
            or int(step.get("v_cache_non_null_count_after", -1)) != 5
            or step.get("k_cache_non_null_mask_after") != expected_mask
            or step.get("v_cache_non_null_mask_after") != expected_mask
            or int(step.get("cache_update_ordinal", -1)) != source_index + 1
            or int(step.get("reset_count", -1)) != 1
        ):
            raise ValueError(f"stream step {source_index} cache audit is inconsistent")
        step_by_source[source_index] = step

    camera = payload.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("TURTLE manifest camera must be an object")
    if (
        camera.get("model") != "PINHOLE"
        or int(camera.get("width", -1)) != 512
        or int(camera.get("height", -1)) != 384
    ):
        raise ValueError("formal TURTLE mapping camera must be PINHOLE 512x384")
    matrix = camera.get("K")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise ValueError("TURTLE mapping camera K must be a 3x3 list")
    for row_index, (actual_row, expected_row) in enumerate(zip(matrix, TRACKER_K)):
        for column_index, (actual, expected) in enumerate(zip(actual_row, expected_row)):
            try:
                difference = abs(float(actual) - float(expected))
            except (TypeError, ValueError) as error:
                raise ValueError("TURTLE mapping camera K must be numeric") from error
            if difference > 1.0e-6:
                raise ValueError(
                    "TURTLE mapping K does not match Unblur tracker resize/crop space: "
                    f"K[{row_index}][{column_index}]={actual} != {expected}"
                )
    if (
        int(camera.get("raw_width", -1)) != TRACKER_RAW_SIZE[0]
        or int(camera.get("raw_height", -1)) != TRACKER_RAW_SIZE[1]
        or int(camera.get("resize_before_crop_width", -1))
        != TRACKER_RESIZE_BEFORE_CROP[0]
        or int(camera.get("resize_before_crop_height", -1))
        != TRACKER_RESIZE_BEFORE_CROP[1]
    ):
        raise ValueError("TURTLE camera must encode raw640x480 -> resize528x400")
    crop = camera.get("crop_edges")
    if not isinstance(crop, Mapping) or tuple(
        int(crop.get(key, -1)) for key in ("left", "right", "top", "bottom")
    ) != TRACKER_CROP_EDGES:
        raise ValueError("TURTLE camera must crop exactly 8 pixels on every edge")
    distortion = camera.get("distortion")
    if not isinstance(distortion, Mapping) or tuple(
        float(value) for value in distortion.get("vector", ())
    ) != TRACKER_DISTORTION:
        raise ValueError("TURTLE manifest distortion vector drifted from fr2_xyz")
    if tuple(camera.get("preprocessing", ())) != TRACKER_PREPROCESSING:
        raise ValueError(
            "TURTLE preprocessing must match tracker: undistort -> resize 528x400 "
            "-> crop 8 pixels per edge -> 512x384"
        )

    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != len(KEYFRAME_INDICES):
        raise ValueError("TURTLE manifest frames must contain exactly 42 emitted records")
    frame_indices: list[int] = []
    output_records: list[dict[str, str]] = []
    for position, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"TURTLE frame {position} must be an object")
        source_index = int(frame.get("source_index", -1))
        frame_indices.append(source_index)
        if source_index != KEYFRAME_INDICES[position]:
            raise ValueError("TURTLE frame records are not the fixed sorted keyframe list")
        if int(frame.get("step_index", -1)) != source_index:
            raise ValueError("TURTLE emitted frame step_index must equal source_index")
        matching_step = step_by_source[source_index]
        try:
            frame_timestamp = float(frame.get("timestamp"))
            step_timestamp = float(matching_step.get("timestamp"))
        except (TypeError, ValueError) as error:
            raise ValueError("TURTLE emitted frame timestamp is invalid") from error
        if frame_timestamp != step_timestamp:
            raise ValueError("TURTLE emitted frame timestamp disagrees with its stream step")
        if matching_step.get("emitted_png", matching_step.get("emitted")) is not True:
            raise ValueError("TURTLE emitted frame has no matching emitted stream step")
        output_path, output_sha = _verify_hashed_record(
            frame.get("output"),
            root,
            f"frames[{position}].output",
            verify_file=verify_frame_files,
        )
        output_value = frame["output"]
        assert isinstance(output_value, Mapping)
        try:
            output_path.relative_to(root)
        except ValueError as error:
            raise ValueError("emitted TURTLE PNG must be inside its immutable bundle") from error
        if int(output_value.get("width", -1)) != 512 or int(output_value.get("height", -1)) != 384:
            raise ValueError("every emitted TURTLE PNG must declare 512x384")
        if matching_step.get("emitted_png_sha256") != output_sha:
            raise ValueError("emitted TURTLE frame hash disagrees with its stream step")
        stream_audit = frame.get("stream_audit")
        if (
            not isinstance(stream_audit, Mapping)
            or stream_audit.get("cache_present_after") is not True
            or int(stream_audit.get("cache_update_ordinal", -1)) != source_index + 1
            or int(stream_audit.get("reset_count", -1)) != 1
        ):
            raise ValueError("emitted TURTLE frame has an invalid cache audit")
        output_records.append({"path": str(output_path), "sha256": output_sha})

    safety = payload.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError("TURTLE manifest safety must be an object")
    for key in (
        "ground_truth_images_used",
        "ground_truth_poses_used",
        "depth_used",
        "custom_causal_evssm_used",
        "sliding_window_recomputation_used",
    ):
        _require_false(safety.get(key), f"safety.{key}")

    return {
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "schema": TURTLE_MANIFEST_SCHEMA,
        "frames_csv": str(csv_path),
        "frames_csv_sha256": csv_sha,
        "official_turtle": {
            "repository": str(repo),
            "origin": origin,
            "commit": commit,
            "artifacts": artifact_audit,
            "cache_contract": OFFICIAL_TURTLE_CACHE_CONTRACT,
        },
        "stream": {
            "source_first": SOURCE_FIRST,
            "source_last": SOURCE_LAST,
            "processed_count": STREAM_COUNT,
            "cache_updates": STREAM_COUNT,
            "reset_count": 1,
            "k_count": 8,
            "v_count": 8,
            "non_null_count": 5,
            "non_null_mask": expected_mask,
        },
        "selection": {
            "source_indices": frame_indices,
            "count": len(frame_indices),
        },
        "emitted_outputs": output_records,
    }


def _assert_config_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"pipeline config schema must be {SCHEMA!r}")
    selection = config.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("pipeline selection must be an object")
    if tuple(int(value) for value in selection.get("source_indices", ())) != KEYFRAME_INDICES:
        raise ValueError("pipeline config must pin the fixed 42 DROID keyframes")
    if (
        int(selection.get("num_context", -1)) != NUM_CONTEXT
        or int(selection.get("num_target", -1)) != NUM_TARGET
        or selection.get("context_strategy") != "fps"
    ):
        raise ValueError("pipeline config must pin 8 FPS contexts and 34 remaining targets")

    excluded = config.get("excluded_legacy_components")
    if not isinstance(excluded, Mapping):
        raise ValueError("pipeline config must explicitly exclude legacy components")
    for key in (
        "mapping.resplat",
        "residual_replay",
        "causal_evssm",
        "custom_replay_sampler",
    ):
        _require_false(excluded.get(key), f"excluded_legacy_components.{key}")

    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("pipeline source must be an object")
    _require_false(source.get("uses_ground_truth_pose"), "source.uses_ground_truth_pose")
    if source.get("pose_source") != "droid_traj_est_not_align":
        raise ValueError("pipeline must use the non-aligned first-pass DROID pose source")

    turtle = config.get("official_turtle")
    if not isinstance(turtle, Mapping):
        raise ValueError("official_turtle config must be an object")
    expected_turtle = {
        "origin": OFFICIAL_TURTLE_ORIGIN,
        "commit": OFFICIAL_TURTLE_COMMIT,
        "architecture_sha256": OFFICIAL_TURTLE_ARCH_SHA256,
        "config_sha256": OFFICIAL_TURTLE_CONFIG_SHA256,
        "checkpoint_kind": "official_gopro",
        "checkpoint_sha256": OFFICIAL_TURTLE_GOPRO_SHA256,
        "cache_contract": OFFICIAL_TURTLE_CACHE_CONTRACT,
        "required_cache_updates": STREAM_COUNT,
    }
    for key, expected in expected_turtle.items():
        if turtle.get(key) != expected:
            raise ValueError(f"official_turtle.{key} must equal {expected!r}")
    if tuple(int(value) for value in turtle.get("required_processed_range", ())) != (
        SOURCE_FIRST,
        SOURCE_LAST,
    ):
        raise ValueError("official_turtle.required_processed_range must be [0,2764]")
    preprocessing = turtle.get("preprocessing_contract")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("official_turtle.preprocessing_contract must be an object")
    expected_preprocessing = {
        "raw_size": list(TRACKER_RAW_SIZE),
        "undistort": "opencv_same_K",
        "resize_before_crop": list(TRACKER_RESIZE_BEFORE_CROP),
        "crop_edges": list(TRACKER_CROP_EDGES),
        "output_size": list(TRACKER_OUTPUT_SIZE),
        "K": [list(row) for row in TRACKER_K],
        "distortion": list(TRACKER_DISTORTION),
    }
    if dict(preprocessing) != expected_preprocessing:
        raise ValueError("official TURTLE preprocessing config drifted from tracker space")

    execution = config.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("pipeline execution must be an object")
    expected_execution = {
        "physical_gpu": 1,
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": "1",
        "process_device": "cuda:0",
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise ValueError(f"execution.{key} must equal {expected!r}")

    resplat = config.get("official_resplat")
    if not isinstance(resplat, Mapping):
        raise ValueError("official_resplat config must be an object")
    if (
        resplat.get("origin") != OFFICIAL_RESPLAT_ORIGIN
        or resplat.get("commit") != OFFICIAL_RESPLAT_COMMIT
        or resplat.get("model_preset") != OFFICIAL_RESPLAT_PRESET
        or resplat.get("checkpoint_sha256") != OFFICIAL_RESPLAT_CHECKPOINT_SHA256
        or int(resplat.get("num_refine", -1)) != NUM_REFINE
        or float(resplat.get("near", -1.0)) != 0.01
        or float(resplat.get("far", -1.0)) != 200.0
        or int(resplat.get("render_chunk_size", -1)) != 4
    ):
        raise ValueError("official ReSplat pins drifted from the formal smoke contract")


def _required_path(config: Mapping[str, Any], key: str, *, directory: bool = False) -> Path:
    value = config.get("paths")
    if not isinstance(value, Mapping):
        raise ValueError("pipeline paths must be an object")
    path = Path(str(value.get(key, ""))).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"paths.{key} {kind} does not exist: {path}")
    return path


def _required_executable_lexical(
    config: Mapping[str, Any], key: str
) -> tuple[Path, Path]:
    """Validate an interpreter without resolving away its environment prefix.

    Conda/venv launchers are commonly symlinks.  Executing their realpath can
    change ``sys.prefix`` and silently drop the environment's site-packages,
    so the child command must retain the configured lexical absolute path.
    """

    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("pipeline paths must be an object")
    raw = str(paths.get(key, "")).strip()
    if not raw:
        raise ValueError(f"paths.{key} is empty")
    lexical = Path(os.path.expanduser(raw))
    if not lexical.is_absolute():
        raise ValueError(f"paths.{key} must be a lexical absolute path")
    if not lexical.is_file():
        raise FileNotFoundError(f"paths.{key} executable does not exist: {lexical}")
    if not os.access(lexical, os.X_OK):
        raise ValueError(f"paths.{key} is not executable: {lexical}")
    realpath = Path(os.path.realpath(lexical))
    if not realpath.is_file() or not os.access(realpath, os.X_OK):
        raise ValueError(f"paths.{key} realpath is not an executable file: {realpath}")
    return lexical, realpath


def _output_paths(
    config: Mapping[str, Any],
    *,
    require_existing_scene: bool = False,
    require_existing_paired: bool = False,
) -> dict[str, Path]:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("pipeline paths must be an object")
    result = {
        key: Path(str(paths.get(key, ""))).expanduser().resolve()
        for key in ("scene_output_dir", "paired_output_dir", "audit_output_dir")
    }
    if any(str(path) == "/" for path in result.values()) or len(set(result.values())) != 3:
        raise ValueError("pipeline output directories must be three distinct non-root paths")
    if require_existing_paired and not require_existing_scene:
        raise ValueError("an existing paired output requires an existing audited scene")
    policies = {
        "scene_output_dir": "existing" if require_existing_scene else "absent",
        "paired_output_dir": "existing" if require_existing_paired else "absent",
        "audit_output_dir": "absent",
    }
    for label, path in result.items():
        if path.is_symlink():
            raise ValueError(f"pipeline output may not be a symlink: {path}")
        if policies[label] == "existing":
            if not path.is_dir():
                raise FileNotFoundError(f"required existing {label} is missing: {path}")
        elif path.exists():
            raise FileExistsError(f"refusing to overwrite {label}: {path}")
    return result


def preflight(
    config_path: Path | str,
    *,
    verify_all_step_inputs: bool = True,
    require_existing_scene: bool = False,
    require_existing_paired: bool = False,
) -> dict[str, Any]:
    """Run the complete CPU-only formal preflight and return its audit record."""

    source, config = _load_json(config_path, "pipeline config")
    _assert_config_contract(config)
    outputs = _output_paths(
        config,
        require_existing_scene=require_existing_scene,
        require_existing_paired=require_existing_paired,
    )

    paths = config["paths"]
    assert isinstance(paths, Mapping)
    frames_csv = _required_path(config, "frames_csv")
    frames_csv_sha = _require_sha256(
        config.get("source", {}).get("frames_csv_sha256")
        if isinstance(config.get("source"), Mapping)
        else None,
        "source.frames_csv_sha256",
    )
    actual_csv_sha = sha256_file(frames_csv)
    if actual_csv_sha != frames_csv_sha:
        raise ValueError("configured frames CSV SHA-256 does not match its bytes")

    turtle_repo = _required_path(config, "turtle_repo", directory=True)
    turtle_checkpoint = _required_path(config, "turtle_checkpoint")
    turtle_repo_record = inspect_pinned_repo(
        turtle_repo,
        expected_origin=OFFICIAL_TURTLE_ORIGIN,
        expected_commit=OFFICIAL_TURTLE_COMMIT,
        label="official TURTLE",
    )
    if sha256_file(turtle_checkpoint) != OFFICIAL_TURTLE_GOPRO_SHA256:
        raise ValueError("configured TURTLE checkpoint is not the pinned official GoPro weight")

    turtle_manifest_path, turtle_manifest = _load_json(
        paths.get("turtle_manifest"), "TURTLE stream manifest"
    )
    turtle_audit = validate_turtle_stream_manifest(
        turtle_manifest,
        turtle_manifest_path,
        expected_frames_csv=frames_csv,
        expected_frames_csv_sha256=frames_csv_sha,
        expected_turtle_repo=turtle_repo,
        expected_checkpoint=turtle_checkpoint,
        verify_frame_files=True,
        verify_all_step_inputs=verify_all_step_inputs,
    )

    resplat_repo = _required_path(config, "resplat_repo", directory=True)
    resplat_checkpoint = _required_path(config, "resplat_checkpoint")
    resplat_python, resplat_python_realpath = _required_executable_lexical(
        config, "resplat_python"
    )
    resplat_repo_record = inspect_pinned_repo(
        resplat_repo,
        expected_origin=OFFICIAL_RESPLAT_ORIGIN,
        expected_commit=OFFICIAL_RESPLAT_COMMIT,
        label="official cvg/ReSplat",
    )
    if sha256_file(resplat_checkpoint) != OFFICIAL_RESPLAT_CHECKPOINT_SHA256:
        raise ValueError("configured ReSplat checkpoint is not the pinned official weight")
    for script in (EXPORT_SCRIPT, PAIRED_SCRIPT):
        if not script.is_file():
            raise FileNotFoundError(f"required audited bridge script is missing: {script}")

    audit = {
        "schema": PIPELINE_RUN_SCHEMA,
        "preflight_only": True,
        "config": {"path": str(source), "sha256": sha256_file(source)},
        "source": {"frames_csv": str(frames_csv), "sha256": frames_csv_sha},
        "turtle_stream": turtle_audit,
        "official_turtle_repo": turtle_repo_record,
        "official_resplat": {
            "repository": resplat_repo_record,
            "checkpoint": {
                "path": str(resplat_checkpoint),
                "sha256": OFFICIAL_RESPLAT_CHECKPOINT_SHA256,
            },
            "python": str(resplat_python),
            "python_realpath": str(resplat_python_realpath),
            "python_command_preserves_lexical_environment_path": True,
            "model_preset": OFFICIAL_RESPLAT_PRESET,
            "num_context": NUM_CONTEXT,
            "context_strategy": "fps",
            "num_target": NUM_TARGET,
            "num_refine": NUM_REFINE,
        },
        "excluded_legacy_components": dict(config["excluded_legacy_components"]),
        "execution": dict(config["execution"]),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "scripts": {
            "export": {"path": str(EXPORT_SCRIPT), "sha256": sha256_file(EXPORT_SCRIPT)},
            "paired": {"path": str(PAIRED_SCRIPT), "sha256": sha256_file(PAIRED_SCRIPT)},
        },
    }
    if require_existing_scene:
        audit["existing_scene_validation"] = _validate_scene_output(audit)
    if require_existing_paired:
        audit["existing_paired_validation"] = _validate_paired_output(audit)
    return audit


def build_commands(config: Mapping[str, Any]) -> tuple[list[str], list[str], dict[str, str]]:
    """Build fixed argument arrays; no caller-provided shell fragment is accepted."""

    _assert_config_contract(config)
    paths = config["paths"]
    assert isinstance(paths, Mapping)
    source = config["source"]
    assert isinstance(source, Mapping)
    resplat = config["official_resplat"]
    assert isinstance(resplat, Mapping)
    execution = config["execution"]
    assert isinstance(execution, Mapping)
    indices = ",".join(str(value) for value in KEYFRAME_INDICES)
    resplat_python, _ = _required_executable_lexical(config, "resplat_python")

    export_command = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--frames-csv", str(Path(str(paths["frames_csv"])).expanduser().resolve()),
        "--output-dir", str(Path(str(paths["scene_output_dir"])).expanduser().resolve()),
        "--indices", indices,
        "--image-mode", "turtle",
        "--images-json", str(Path(str(paths["turtle_manifest"])).expanduser().resolve()),
        "--resplat-repo", str(Path(str(paths["resplat_repo"])).expanduser().resolve()),
        "--model-preset", OFFICIAL_RESPLAT_PRESET,
        "--checkpoint", str(Path(str(paths["resplat_checkpoint"])).expanduser().resolve()),
        "--expected-checkpoint-sha256", OFFICIAL_RESPLAT_CHECKPOINT_SHA256,
        "--formal-smoke",
    ]

    paired_command = [
        str(resplat_python),
        str(PAIRED_SCRIPT),
        "--scene-path", str(Path(str(paths["scene_output_dir"])).expanduser().resolve()),
        "--scene-manifest", str(
            Path(str(paths["scene_output_dir"])).expanduser().resolve() / "manifest.json"
        ),
        "--resplat-repo", str(Path(str(paths["resplat_repo"])).expanduser().resolve()),
        "--checkpoint", str(Path(str(paths["resplat_checkpoint"])).expanduser().resolve()),
        "--expected-checkpoint-sha256", OFFICIAL_RESPLAT_CHECKPOINT_SHA256,
        "--output-dir", str(Path(str(paths["paired_output_dir"])).expanduser().resolve()),
        "--model-preset", OFFICIAL_RESPLAT_PRESET,
        "--device", "cuda:0",
        "--context-selection", "fps",
        "--expected-target-count", str(NUM_TARGET),
        "--near", str(float(resplat.get("near", 0.01))),
        "--far", str(float(resplat.get("far", 200.0))),
        "--render-chunk-size", str(int(resplat.get("render_chunk_size", 4))),
        "--max-save-images", str(NUM_TARGET),
        "--save-ply",
    ]
    child_environment = {
        "CUDA_DEVICE_ORDER": str(execution["cuda_device_order"]),
        "CUDA_VISIBLE_DEVICES": str(execution["cuda_visible_devices"]),
        "PYTHONUNBUFFERED": "1",
    }
    return export_command, paired_command, child_environment


def _run_logged(command: Sequence[str], log_path: Path, environment: Mapping[str, str]) -> None:
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        log.write("command=" + json.dumps(list(command)) + "\n")
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return_code = int(process.wait())
        log.write(f"exit_code={return_code}\n")
        if return_code != 0:
            raise RuntimeError(f"pipeline stage failed with exit code {return_code}")


def _validate_scene_output(audit: Mapping[str, Any]) -> dict[str, Any]:
    outputs = audit["outputs"]
    assert isinstance(outputs, Mapping)
    scene = Path(str(outputs["scene_output_dir"]))
    scene_manifest_path, scene_manifest = _load_json(scene / "manifest.json", "scene manifest")
    if scene_manifest.get("schema") != "unblur_slam.official_resplat_colmap_scene.v1":
        raise ValueError("export stage did not create the audited official ReSplat scene")
    if scene_manifest.get("formal_smoke") is not True:
        raise ValueError("existing scene was not exported under the formal-smoke contract")
    if scene_manifest.get("selection", {}).get("source_indices") != list(KEYFRAME_INDICES):
        raise ValueError("exported scene keyframe selection drifted")
    source_csv = scene_manifest.get("source_csv")
    if (
        not isinstance(source_csv, Mapping)
        or source_csv.get("sha256") != audit["source"]["sha256"]
        or source_csv.get("pose_source") != "droid_traj_est_not_align"
        or source_csv.get("uses_ground_truth_pose") is not False
    ):
        raise ValueError("existing scene source/pose contract disagrees with preflight")
    ground_truth = scene_manifest.get("ground_truth_contract")
    if not isinstance(ground_truth, Mapping) or any(
        ground_truth.get(key) is not False
        for key in (
            "uses_ground_truth_pose",
            "ground_truth_pose_used",
            "ground_truth_file_read",
            "contains_ground_truth_sidecar",
            "ground_truth_sidecar_arrays_accessed",
        )
    ):
        raise ValueError("existing scene read or used forbidden ground-truth pose data")
    images = scene_manifest.get("images")
    if not isinstance(images, Mapping) or images.get("mode_label") != "turtle":
        raise ValueError("exported scene is not bound to the TURTLE image mapping")
    mapping_manifest = images.get("mapping_manifest")
    if not isinstance(mapping_manifest, Mapping) or mapping_manifest.get("sha256") != audit[
        "turtle_stream"
    ]["manifest_sha256"]:
        raise ValueError("scene does not bind the preflighted TURTLE manifest bytes")

    camera = scene_manifest.get("camera")
    if (
        not isinstance(camera, Mapping)
        or camera.get("model") != "PINHOLE"
        or int(camera.get("width", -1)) != TRACKER_OUTPUT_SIZE[0]
        or int(camera.get("height", -1)) != TRACKER_OUTPUT_SIZE[1]
    ):
        raise ValueError("existing scene camera is not the audited tracker output space")
    matrix = camera.get("K")
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError("existing scene camera K is invalid")
    for actual_row, expected_row in zip(matrix, TRACKER_K):
        if len(actual_row) != 3 or any(
            abs(float(actual) - expected) > 1.0e-6
            for actual, expected in zip(actual_row, expected_row)
        ):
            raise ValueError("existing scene K differs from the tracker preprocessing K")
    official = scene_manifest.get("official_resplat")
    if not isinstance(official, Mapping):
        raise ValueError("existing scene lacks official ReSplat provenance")
    repository = official.get("repository")
    checkpoint = official.get("checkpoint")
    actual_checkpoint = checkpoint.get("actual") if isinstance(checkpoint, Mapping) else None
    if (
        not isinstance(repository, Mapping)
        or repository.get("commit") != OFFICIAL_RESPLAT_COMMIT
        or repository.get("tracked_worktree_clean") is not True
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("model_preset") != OFFICIAL_RESPLAT_PRESET
        or int(checkpoint.get("num_context", -1)) != NUM_CONTEXT
        or int(checkpoint.get("num_refine", -1)) != NUM_REFINE
        or not isinstance(actual_checkpoint, Mapping)
        or actual_checkpoint.get("sha256") != OFFICIAL_RESPLAT_CHECKPOINT_SHA256
    ):
        raise ValueError("existing scene official ReSplat provenance drifted")
    return {
        "path": str(scene_manifest_path),
        "sha256": sha256_file(scene_manifest_path),
        "validated_for_safe_resume": True,
    }


def _validate_paired_output(audit: Mapping[str, Any]) -> dict[str, Any]:
    outputs = audit["outputs"]
    assert isinstance(outputs, Mapping)
    paired = Path(str(outputs["paired_output_dir"]))
    paired_manifest_path, paired_manifest = _load_json(
        paired / "run_manifest.json", "paired run manifest"
    )

    if paired_manifest.get("schema") != "unblur_slam.paired_official_resplat_smoke.v1":
        raise ValueError("paired stage did not create the audited official runner output")
    official = paired_manifest.get("official_resplat")
    selection = paired_manifest.get("selection")
    if not isinstance(official, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("paired run manifest is missing official selection provenance")
    paired_checkpoint = official.get("checkpoint")
    paired_repository = official.get("repository")
    if (
        official.get("model_preset") != OFFICIAL_RESPLAT_PRESET
        or int(official.get("num_context", -1)) != NUM_CONTEXT
        or int(official.get("num_refine", -1)) != NUM_REFINE
        or not isinstance(paired_checkpoint, Mapping)
        or paired_checkpoint.get("sha256") != OFFICIAL_RESPLAT_CHECKPOINT_SHA256
        or not isinstance(paired_repository, Mapping)
        or paired_repository.get("commit") != OFFICIAL_RESPLAT_COMMIT
        or paired_repository.get("tracked_worktree_clean") is not True
        or selection.get("context_strategy") != "fps"
        or int(selection.get("target_count", -1)) != NUM_TARGET
        or selection.get("same_targets_for_init0_and_refine4") is not True
    ):
        raise ValueError("paired official ReSplat selection/refinement contract drifted")
    if float(paired_manifest.get("near", -1.0)) != 0.01 or float(
        paired_manifest.get("far", -1.0)
    ) != 200.0:
        raise ValueError("paired output changed the registered near/far planes")
    expected_scene = audit.get("existing_scene_validation")
    scene_record = paired_manifest.get("scene")
    if not isinstance(scene_record, Mapping):
        raise ValueError("paired output lacks its input scene hash")
    if expected_scene is None:
        expected_scene = _validate_scene_output(audit)
    if scene_record.get("manifest_sha256") != expected_scene["sha256"]:
        raise ValueError("paired output does not consume the audited scene bytes")
    for relative in ("paired_init0/metrics.json", "paired_refine4/metrics.json"):
        if not (paired / relative).is_file():
            raise FileNotFoundError(f"paired output is incomplete: {paired / relative}")
    return {
        "path": str(paired_manifest_path),
        "sha256": sha256_file(paired_manifest_path),
        "metrics": paired_manifest.get("metrics"),
        "validated_for_safe_resume": True,
    }


def _validate_installed_outputs(audit: Mapping[str, Any]) -> dict[str, Any]:
    scene = _validate_scene_output(audit)
    paired = _validate_paired_output({**audit, "existing_scene_validation": scene})
    return {
        "scene_manifest": scene,
        "paired_manifest": paired,
        "metrics": paired.get("metrics"),
    }


def _install_pipeline_audit(
    audit: Mapping[str, Any],
    final_record: Mapping[str, Any],
    *,
    staged_logs: Optional[Mapping[str, Path]] = None,
) -> Path:
    """Atomically install the immutable pipeline audit and optional logs."""

    audit_output = Path(audit["outputs"]["audit_output_dir"])
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{audit_output.name}.staging-", dir=audit_output.parent)
    )
    installed = False
    try:
        copied_logs: list[Path] = []
        for name, source in (staged_logs or {}).items():
            destination = staging / name
            shutil.copyfile(source, destination)
            copied_logs.append(destination)
        manifest_path = staging / "pipeline_run_manifest.json"
        manifest_path.write_text(
            json.dumps(final_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in (*copied_logs, manifest_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        if audit_output.exists() or audit_output.is_symlink():
            raise FileExistsError(f"refusing concurrent audit overwrite: {audit_output}")
        os.rename(staging, audit_output)
        installed = True
        return audit_output
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)


def run_pipeline(config_path: Path | str, *, paired_only: bool = False) -> Path:
    """Run the full pipeline or resume only paired inference from an audited scene."""

    audit = preflight(config_path, require_existing_scene=paired_only)
    _, config = _load_json(config_path, "pipeline config")
    export_command, paired_command, child_updates = build_commands(config)
    environment = os.environ.copy()
    for key in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES"):
        environment.pop(key, None)
    environment.update(child_updates)

    # Logs live in a disposable sibling staging directory while GPU work runs;
    # only the final audit installer can publish them.
    audit_output = Path(audit["outputs"]["audit_output_dir"])
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    log_staging = Path(
        tempfile.mkdtemp(prefix=f".{audit_output.name}.logs-", dir=audit_output.parent)
    )
    try:
        logs: dict[str, Path] = {}
        if not paired_only:
            logs["export.log"] = log_staging / "export.log"
            _run_logged(export_command, logs["export.log"], environment)
        logs["paired.log"] = log_staging / "paired.log"
        _run_logged(paired_command, logs["paired.log"], environment)
        installed_outputs = _validate_installed_outputs(audit)
        final_record = {
            **audit,
            "preflight_only": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "resume": {
                "paired_only": bool(paired_only),
                "existing_scene_validated_before_gpu": bool(paired_only),
                "scene_reexported": not paired_only,
            },
            "commands": {
                "export": None if paired_only else export_command,
                "paired": paired_command,
                "environment": child_updates,
                "shell_used": False,
            },
            "installed_outputs": installed_outputs,
        }
        return _install_pipeline_audit(
            audit, final_record, staged_logs=logs
        )
    finally:
        if log_staging.exists():
            shutil.rmtree(log_staging)


def audit_existing_outputs(config_path: Path | str) -> Path:
    """CPU-only validation and atomic audit of already installed scene+paired outputs."""

    audit = preflight(
        config_path,
        require_existing_scene=True,
        require_existing_paired=True,
    )
    installed_outputs = _validate_installed_outputs(audit)
    final_record = {
        **audit,
        "preflight_only": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "resume": {
            "audit_existing_only": True,
            "gpu_started": False,
            "scene_reexported": False,
            "paired_rerun": False,
        },
        "commands": {"export": None, "paired": None, "shell_used": False},
        "installed_outputs": installed_outputs,
    }
    return _install_pipeline_audit(audit, final_record)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true", help="CPU-only validation (default)")
    action.add_argument("--run", action="store_true", help="run export and paired GPU smoke")
    action.add_argument(
        "--paired-only",
        action="store_true",
        help="resume from an existing audited scene; never re-export/overwrite it",
    )
    action.add_argument(
        "--preflight-paired-only",
        action="store_true",
        help="CPU-only validation of the safe paired-only resume boundary",
    )
    action.add_argument(
        "--audit-existing",
        action="store_true",
        help="CPU-only validation and audit of existing scene+paired outputs",
    )
    parser.add_argument(
        "--skip-all-step-file-rehash",
        action="store_true",
        help="development-only faster preflight; formal --run never uses this",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.run or args.paired_only:
            if args.skip_all_step_file_rehash:
                raise ValueError("GPU runs forbid --skip-all-step-file-rehash")
            output = run_pipeline(args.config, paired_only=args.paired_only)
            print(f"official TURTLE -> ReSplat pipeline audit installed at {output}")
        elif args.audit_existing:
            if args.skip_all_step_file_rehash:
                raise ValueError("--audit-existing forbids --skip-all-step-file-rehash")
            output = audit_existing_outputs(args.config)
            print(f"existing official pipeline outputs audited atomically at {output}")
        else:
            audit = preflight(
                args.config,
                verify_all_step_inputs=not args.skip_all_step_file_rehash,
                require_existing_scene=args.preflight_paired_only,
            )
            print(json.dumps(audit, indent=2, sort_keys=True))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
