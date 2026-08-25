#!/usr/bin/env python3
"""CPU-only, fail-closed preflight for the fair offline 26K/ReSplat protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from offline_fair_gpu_contract import validate_gpu_contract


SCHEMA = "unblur_slam.offline_resplat_fair_26k.v1"
REPORT_SCHEMA = "unblur_slam.offline_resplat_fair_26k_preflight.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GLOBAL_GPU_LOCK = "/srv/szha0669/unblur-slam/locks/physical_gpu1.lock"
REQUIRED_PROTOCOL_TOOLS = {
    "materialize_frozen_bundle",
    "scene_exporter",
    "common_rgb_evaluator",
    "sequential_wall_executor",
    "cpu_preflight",
    "gpu_contract",
    "pinned_gpu_wrapper",
    "final_reporter",
    "runbook",
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _git_blob_sha256(repo: Path, commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{relative_path}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def chronological_context_windows(
    source_indices: Sequence[int], context_count: int = 8
) -> list[list[int]]:
    ordered = [int(value) for value in source_indices]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ValueError("source indices must be strictly increasing and unique")
    if len(ordered) < context_count:
        raise ValueError("not enough keyframes for one official ReSplat context")
    windows = [
        ordered[start : start + context_count]
        for start in range(0, len(ordered), context_count)
    ]
    if len(windows[-1]) < context_count:
        windows[-1] = ordered[-context_count:]
    return windows


def _read_indices(path: Path) -> list[int]:
    values = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if values != sorted(set(values)):
        raise ValueError(f"indices must be sorted and unique: {path}")
    return values


def _normalize_origin(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.replace("git@github.com:", "https://github.com/").lower()


def validate_contract(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("wrong contract schema")
    boundary = payload.get("claim_boundary", {})
    for key in (
        "official_resplat_refines_existing_unblur_map",
        "official_resplat_reads_unblur_gaussian_or_optimizer_state",
        "active_map_merge",
        "single_global_resplat_map",
        "residual_replay_is_official_resplat",
    ):
        if boundary.get(key) is not False:
            errors.append(f"claim boundary {key} must be false")
    if boundary.get("resplat_is_independent_terminal_backend") is not True:
        errors.append("ReSplat must be declared an independent terminal backend")
    baseline = payload.get("unblur_baseline", {})
    if baseline.get("milestones_from_one_trajectory") != [8000, 12000, 26000]:
        errors.append("baseline milestones must be one [8K,12K,26K] trajectory")
    if baseline.get("legacy_bpn") is not False or baseline.get("residual_replay") is not False:
        errors.append("legacy BPN and residual replay must both be disabled")
    resplat = payload.get("official_resplat", {})
    if resplat.get("context_count") != 8 or resplat.get("recurrent_updates") != 4:
        errors.append("official small ReSplat must use 8 contexts and 4 updates")
    if payload.get("reporting", {}).get("combined_U_plus_R_arm") is not None:
        errors.append("a fused U+R arm is forbidden without a trained adapter")
    reporting = payload.get("reporting", {})
    if reporting.get("rgb_evaluation", {}).get("resolution_hw") != [320, 448]:
        errors.append("formal RGB comparison must use one common 320x448 resolution")
    if reporting.get("depth_l1", {}).get("formal_gate_available") is not False:
        errors.append("depth-L1 gate must remain unavailable without audited R4 raw depth")
    if "depth L1" in reporting.get("metrics", []):
        errors.append("unavailable depth L1 cannot be listed as a formal metric")
    query_scope = reporting.get("query_scope", {})
    if (
        query_scope.get("all_formal_queries_are_context") is not True
        or query_scope.get("mapped_training_views") is not True
        or query_scope.get("novel_views") is not False
        or query_scope.get("non_context_stratum_available") is not False
    ):
        errors.append("formal query scope must be 100% context mapped-training views")
    if reporting.get("png_quantization") != "official_resplat_mul255_astype_uint8_floor":
        errors.append("all arms must use the official floor PNG quantizer")
    if reporting.get("aggregate_eval_count") != 113:
        errors.append("formal pooled aggregate must contain exactly 113 views")
    if reporting.get("expected_total_resplat_submaps") != 38:
        errors.append("formal protocol must require exactly 38 ReSplat subprocesses")
    primary_excludes = reporting.get("primary_timing", {}).get("excludes", [])
    for excluded in ("metric-reference loading", "metric computation", "output PNG/PLY artifact I/O"):
        if excluded not in primary_excludes:
            errors.append(f"primary timing must exclude {excluded}")
    if (
        reporting.get("primary_timing", {}).get("resplat_wall_scope")
        != "terminal_backend_setup_plus_core_no_metrics_or_artifact_io"
        or reporting.get("primary_timing", {}).get("resplat_peak_scope")
        != "process_setup_plus_core_before_metrics_and_artifact_io"
    ):
        errors.append("R4 wall/peak scopes must be explicit and frozen")
    hardware = payload.get("execution_hardware", {})
    gpu = hardware.get("gpu", {})
    if (
        gpu.get("physical_index") != 1
        or gpu.get("visible_devices") != "1"
        or gpu.get("logical_device") != "cuda:0"
        or gpu.get("name") != "NVIDIA RTX A6000"
        or not str(gpu.get("uuid", "")).startswith("GPU-")
        or not str(gpu.get("serial", ""))
        or hardware.get("all_gpu_jobs_serial") is not True
    ):
        errors.append("execution hardware must pin physical GPU1 as sole logical cuda:0")
    if (
        hardware.get("exclusive_lock") != GLOBAL_GPU_LOCK
        or payload.get("assets", {}).get("gpu_lock") != GLOBAL_GPU_LOCK
    ):
        errors.append("all managed GPU1 pipelines must use the shared physical-GPU1 lock")
    tooling = payload.get("protocol_tooling", {})
    if set(tooling) != REQUIRED_PROTOCOL_TOOLS:
        errors.append("protocol tooling must contain the exact frozen transitive tool set")
    if len(payload.get("scenes", [])) != 3:
        errors.append("formal protocol requires all three TUM scenes")
    return errors


def run_preflight(contract_path: Path) -> dict[str, Any]:
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    failures = validate_contract(payload)
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            failures.append(name)

    baseline = payload["unblur_baseline"]
    worktree = Path(baseline["worktree"]).resolve()
    check("baseline_worktree_exists", worktree.is_dir(), str(worktree))
    check(
        "baseline_python",
        Path(baseline["python"]).is_file(),
        baseline["python"],
    )
    if worktree.is_dir():
        try:
            head = _git(worktree, "rev-parse", "HEAD")
            origin = _git(worktree, "remote", "get-url", "origin")
            status_lines = _git(
                worktree, "status", "--porcelain", "--untracked-files=all"
            ).splitlines()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            head, origin, status_lines = "", "", []
            failures.append(f"baseline git inspection failed: {error}")
        check("baseline_commit", head == baseline["commit"], head)
        check(
            "baseline_origin",
            _normalize_origin(origin) == _normalize_origin(baseline["official_origin"]),
            origin,
        )
        allowed_paths = set(baseline["measurement_hook"]["allowed_worktree_paths"])
        observed_paths = {line[2:].strip() for line in status_lines if len(line) >= 3}
        check(
            "measurement_only_worktree_diff",
            observed_paths == allowed_paths,
            {"observed": sorted(observed_paths), "allowed": sorted(allowed_paths)},
        )
        for relative, expected in baseline["measurement_hook"][
            "allowed_worktree_paths"
        ].items():
            path = worktree / relative
            actual = sha256_file(path) if path.is_file() else None
            check(f"measurement_file:{relative}", actual == expected, actual)
        for relative, expected in baseline["pristine_blob_sha256"].items():
            try:
                actual = _git_blob_sha256(worktree, baseline["commit"], relative)
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                actual = None
            check(f"pristine_blob:{relative}", actual == expected, actual)

    for label in ("evssm_checkpoint", "droid_checkpoint", "omnidata_checkpoint"):
        record = payload["assets"][label]
        path = Path(record["path"])
        actual = sha256_file(path) if path.is_file() else None
        check(f"asset:{label}", actual == record["sha256"], actual)

    resplat = payload["official_resplat"]
    resplat_repo = Path(resplat["repository"])
    try:
        resplat_head = _git(resplat_repo, "rev-parse", "HEAD")
        resplat_dirty = _git(
            resplat_repo, "status", "--porcelain", "--untracked-files=no"
        )
        resplat_origin = _git(resplat_repo, "remote", "get-url", "origin")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        resplat_head, resplat_dirty, resplat_origin = "", "inspection_failed", ""
    check("resplat_commit", resplat_head == resplat["commit"], resplat_head)
    check("resplat_clean", resplat_dirty == "", resplat_dirty)
    check(
        "resplat_origin",
        _normalize_origin(resplat_origin) == _normalize_origin(resplat["origin"]),
        resplat_origin,
    )
    checkpoint = Path(resplat["checkpoint"])
    actual_checkpoint = sha256_file(checkpoint) if checkpoint.is_file() else None
    check("resplat_checkpoint", actual_checkpoint == resplat["checkpoint_sha256"], actual_checkpoint)
    check("resplat_python", Path(resplat["python"]).is_file(), resplat["python"])
    runner = Path(resplat["runner"])
    actual_runner = sha256_file(runner) if runner.is_file() else None
    check("explicit_selection_runner", actual_runner == resplat["runner_sha256"], actual_runner)

    for label, record in payload.get("protocol_tooling", {}).items():
        tool = Path(record["path"])
        actual = sha256_file(tool) if tool.is_file() else None
        check(f"protocol_tool:{label}", actual == record["sha256"], actual)

    try:
        observed_gpu = validate_gpu_contract(
            payload["execution_hardware"]["gpu"],
            require_visible_mask=False,
            require_idle=False,
        )
    except (OSError, RuntimeError, ValueError) as error:
        observed_gpu = {"error": str(error)}
    check(
        "pinned_gpu_identity",
        observed_gpu.get("uuid") == payload["execution_hardware"]["gpu"]["uuid"],
        observed_gpu,
    )

    dataset_root = Path(payload["assets"]["dataset_root"])
    for scene in payload["scenes"]:
        dataset = dataset_root / scene["input_folder"]
        check(f"dataset:{scene['name']}", dataset.is_dir(), str(dataset))
        indices_path = worktree / scene["eval_indices_file"]
        actual_hash = sha256_file(indices_path) if indices_path.is_file() else None
        check(f"eval_hash:{scene['name']}", actual_hash == scene["eval_indices_sha256"], actual_hash)
        try:
            indices = _read_indices(indices_path)
        except (OSError, ValueError):
            indices = []
        check(f"eval_count:{scene['name']}", len(indices) == scene["eval_count"], len(indices))
        planned = list(range(scene["expected_keyframe_count"]))
        window_count = len(chronological_context_windows(planned))
        check(
            f"planning_submaps:{scene['name']}",
            window_count == scene["expected_resplat_submaps"],
            window_count,
        )

    usage = shutil.disk_usage("/srv")
    free_gib = usage.free / 1024**3
    minimum = float(payload["planning_estimate"]["minimum_free_srv_gib"])
    check("srv_free_space", free_gib >= minimum, {"free_gib": free_gib, "minimum_gib": minimum})
    output_root = Path(payload["assets"]["output_root"])
    check("output_not_started", not output_root.exists(), str(output_root))

    return {
        "schema": REPORT_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "gpu_started": False,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "planning": {
            "total_expected_resplat_submaps": sum(
                scene["expected_resplat_submaps"] for scene in payload["scenes"]
            ),
            "root_filesystem_is_not_an_output_target": True,
            "temporary_and_output_filesystems": "/srv",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_preflight(args.contract.expanduser().resolve())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        destination = args.report.expanduser().resolve()
        if not str(destination).startswith("/srv/"):
            print("error: preflight report must be written under /srv", file=sys.stderr)
            return 2
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
