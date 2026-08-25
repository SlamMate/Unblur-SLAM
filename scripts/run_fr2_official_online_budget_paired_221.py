#!/usr/bin/env python3
"""CPU-preflight or launch the direct 221-frame EVSSM/TURTLE pair.

Both arms use the published Unblur-SLAM online budgets (1050 initialization,
100 mapping, and 100 tracking iterations), seed 43, and 512x384 tensors.  They
use only frames 0..220 and 100 ordinary final-refinement iterations, so this is
not the complete three-sequence paper benchmark and not the 26K protocol.

Module import is standard-library-only.  Heavy project/Torch imports occur only
inside the explicit CPU preflight or GPU launch path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/unblur_slam.yaml"
CONFIGS = {
    "baseline": REPO_ROOT
    / "configs/local/fr2_xyz_official_online_budget_221/evssm_baseline.yaml",
    "turtle": REPO_ROOT
    / "configs/local/fr2_xyz_official_online_budget_221/turtle_gopro_fp16.yaml",
}
OUTPUTS = {
    "baseline": Path(
        "/srv/szha0669/unblur-slam/slam_paired/"
        "fr2_xyz_official_online_budget_221/evssm_baseline"
    ).resolve(),
    "turtle": Path(
        "/srv/szha0669/unblur-slam/slam_paired/"
        "fr2_xyz_official_online_budget_221/turtle_gopro_fp16"
    ).resolve(),
}
EXPECTED_PREFIX = (0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220)
EXPECTED_FULL_PROTOCOL = (
    0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220, 319, 374, 407,
    435, 470, 483, 523, 568, 704, 750, 789, 827, 926, 1004, 1160,
    1251, 1342, 1409, 1460, 1553, 1692, 1795, 1889, 1978, 2055,
    2206, 2282, 2358, 2425, 2590, 2764,
)
EXPECTED_SHA256 = {
    "evssm": "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41",
    "droid": "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526",
    "omnidata": "a0fab23fee64aa9e4bbe0b520b18b196ea7594a7f719c1d8c10cf11dcb6e4a1e",
    "turtle_config": "123b07de8d3f329769562e2f943e08fdf86c576c405634bad199ced95b25aa23",
    "turtle_checkpoint": "10334b3e81d0416bcde5ccaca960dc81dbfb5b6d23e53fadaf7896d72b580c82",
}
OFFICIAL_TURTLE_ORIGIN = "https://github.com/Ascend-Research/Turtle"
PHYSICAL_GPU = "1"

# These are the only scientifically intentional resolved-config differences.
ALLOWED_PAIR_DIFFS = {
    "data.output",
    "deblur.frontend",
    "deblur.turtle_checkpoint",
    "deblur.turtle_checkpoint_sha256",
    "deblur.turtle_config",
    "deblur.turtle_config_sha256",
    "deblur.turtle_inference_precision",
    "deblur.turtle_repo",
    "deblur.turtle_repo_commit",
    "evssm_checkpoint",
    "evssm_checkpoint_sha256",
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    differences: dict[str, dict[str, Any]] = {}
    for path in sorted(set(left) | set(right)):
        if left.get(path) != right.get(path):
            differences[path] = {
                "baseline": left.get(path),
                "turtle": right.get(path),
            }
    return differences


def _validate_disabled_extensions(cfg: Mapping[str, Any], arm: str) -> None:
    framecrafter = cfg.get("framecrafter", {}) or {}
    submaps = cfg.get("submaps", {}) or {}
    replay = (cfg.get("mapping", {}) or {}).get("resplat", {}) or {}
    deblur = cfg.get("deblur", {}) or {}
    if bool(framecrafter.get("enabled", False)):
        raise ValueError(f"{arm}: FrameCrafter must be disabled")
    if bool(submaps.get("enabled", False)):
        raise ValueError(f"{arm}: submaps must be disabled")
    if bool((submaps.get("official_resplat_sidecar", {}) or {}).get("enabled", False)):
        raise ValueError(f"{arm}: official ReSplat sidecar must be disabled")
    if (
        replay.get("enabled") is not False
        or replay.get("online_enabled") is not False
        or int(replay.get("extra_iters", -1)) != 0
    ):
        raise ValueError(f"{arm}: historical residual replay must be disabled")
    if str(deblur.get("causal_checkpoint", "")):
        raise ValueError(f"{arm}: causal frontend/checkpoint must be disabled")


def _validate_arm_contract(cfg: Mapping[str, Any], arm: str) -> None:
    mapping = cfg.get("mapping", {}) or {}
    training = mapping.get("Training", {}) or {}
    common = (
        str(cfg.get("dataset", "")).lower(),
        str(cfg.get("scene", "")),
        int(cfg.get("max_frames", -1)),
        int(cfg.get("stride", -1)),
        int(cfg.get("setup_seed", -1)),
        int((cfg.get("cam", {}) or {}).get("W_out", -1)),
        int((cfg.get("cam", {}) or {}).get("H_out", -1)),
        int(training.get("init_itr_num", -1)),
        int(training.get("mapping_itr_num", -1)),
        int(training.get("tracking_itr_num", -1)),
        int(mapping.get("final_refine_iters", -1)),
    )
    expected = (
        "tumrgbd", "freiburg2_xyz", 221, 1, 43, 512, 384,
        1050, 100, 100, 100,
    )
    if common != expected:
        raise ValueError(f"{arm}: paired compute contract drifted: {common}")
    if str(cfg.get("device", "")) != "cuda:0":
        raise ValueError(f"{arm}: process device must be cuda:0")
    if not bool(cfg.get("warmup_mapper", False)) or bool(cfg.get("clear_init", True)):
        raise ValueError(f"{arm}: warmup_mapper=true and clear_init=false are required")
    if mapping.get("online_plotting") is not False:
        raise ValueError(f"{arm}: online plotting must be disabled")
    if mapping.get("eval_before_final_ba") is not False:
        raise ValueError(f"{arm}: pre-final-BA evaluation must be disabled")
    if mapping.get("hydrate_missing_droid_keyframes") is not True:
        raise ValueError(f"{arm}: complete-prefix DROID hydration must be enabled")
    evaluation = cfg.get("evaluation", {}) or {}
    if (
        evaluation.get("clear_gt_scope") != "prefix_smoke"
        or tuple(evaluation.get("expected_clear_gt_source_indices", ()))
        != EXPECTED_PREFIX
    ):
        raise ValueError(f"{arm}: eleven-frame prefix metric contract drifted")
    disclosure = cfg.get("paired_official_online_budget_221", {}) or {}
    if (
        disclosure.get("schema")
        != "unblur_slam.fr2_xyz_paired_official_online_budget_221.v1"
        or disclosure.get("official_online_optimization_budget") is not True
        or disclosure.get("complete_three_sequence_paper_benchmark") is not False
        or disclosure.get("paper_26k_offline_refinement") is not False
    ):
        raise ValueError(f"{arm}: scope disclosure is missing or ambiguous")
    _validate_disabled_extensions(cfg, arm)

    deblur = cfg.get("deblur", {}) or {}
    frontend = str(deblur.get("frontend", ""))
    if arm == "baseline":
        if frontend != "evssm":
            raise ValueError("baseline: frontend must be official Unblur-SLAM EVSSM")
        if str(cfg.get("evssm_checkpoint_sha256", "")) != EXPECTED_SHA256["evssm"]:
            raise ValueError("baseline: EVSSM checkpoint digest drifted")
    elif arm == "turtle":
        if frontend != "turtle_streaming":
            raise ValueError("turtle: frontend must be turtle_streaming")
        if str(cfg.get("evssm_checkpoint", "")) or str(
            cfg.get("evssm_checkpoint_sha256", "")
        ):
            raise ValueError("turtle: EVSSM must be absent")
        if str(deblur.get("turtle_inference_precision", "")) != "fp16":
            raise ValueError("turtle: inference precision must be fp16")
        if (
            deblur.get("stream_every_frame") is not True
            or deblur.get("stream_apply_to_tracking") is not True
            or deblur.get("stream_replace_sharp") is not False
            or float(deblur.get("stream_min_laplacian_gain", -1.0)) != 0.02
        ):
            raise ValueError("turtle: streaming/gating contract drifted")
    else:
        raise ValueError(f"unknown arm {arm!r}")

    output = Path(str((cfg.get("data", {}) or {}).get("output", ""))).expanduser().resolve()
    if output != OUTPUTS[arm]:
        raise ValueError(f"{arm}: output path drifted: {output}")


def _validate_pair_contract(
    baseline: Mapping[str, Any], turtle: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Pure-Python paired contract used by the standard-library tests."""

    _validate_arm_contract(baseline, "baseline")
    _validate_arm_contract(turtle, "turtle")
    differences = _pair_differences(baseline, turtle)
    unexpected = sorted(set(differences) - ALLOWED_PAIR_DIFFS)
    if unexpected:
        raise ValueError(
            "paired configs differ outside frontend/artifact/output fields: "
            + ", ".join(unexpected)
        )
    missing = sorted(
        {"data.output", "deblur.frontend", "evssm_checkpoint"} - set(differences)
    )
    if missing:
        raise ValueError("paired configs lost required arm differences: " + ", ".join(missing))
    return differences


def _load_configs() -> dict[str, dict[str, Any]]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from thirdparty.glorie_slam import config as config_io

    return {
        arm: config_io.load_config(str(path), str(DEFAULT_CONFIG))
        for arm, path in CONFIGS.items()
    }


def _require_artifact(
    path_value: object, configured_sha: object, expected_sha: str, label: str
) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    configured = str(configured_sha or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", configured):
        raise ValueError(f"{label} configured SHA-256 is invalid")
    if configured != expected_sha:
        raise ValueError(f"{label} config is not pinned to the expected artifact")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError(f"{label} bytes changed: expected {expected_sha}, got {actual}")
    return path


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
        raise ValueError(f"cannot inspect official TURTLE checkout: {repo}") from error
    return completed.stdout.strip()


def _normalize_url(value: str) -> str:
    normalized = str(value).strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def _validate_turtle_artifacts(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.turtle_backend import (
        PINNED_TURTLE_ARCH_SHA256,
        PINNED_TURTLE_CHECKPOINT_SHA256,
        PINNED_TURTLE_COMMIT,
        PINNED_TURTLE_CONFIG_SHA256,
        TURTLE_CACHE_CONTRACT,
        validate_turtle_artifacts,
    )

    artifacts = validate_turtle_artifacts(cfg["deblur"], load_weights=True)
    origin = _git(artifacts.repo, "remote", "get-url", "origin")
    if _normalize_url(origin) != _normalize_url(OFFICIAL_TURTLE_ORIGIN):
        raise ValueError(f"TURTLE origin is not official: {origin}")
    if _git(artifacts.repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("official TURTLE checkout has tracked modifications")
    observed = (
        artifacts.commit,
        artifacts.architecture_sha256,
        artifacts.config_sha256,
        artifacts.checkpoint_sha256,
        artifacts.checkpoint_metadata.get("kind"),
    )
    expected = (
        PINNED_TURTLE_COMMIT,
        PINNED_TURTLE_ARCH_SHA256,
        PINNED_TURTLE_CONFIG_SHA256,
        PINNED_TURTLE_CHECKPOINT_SHA256,
        "official_gopro",
    )
    if observed != expected:
        raise ValueError(f"official TURTLE strict-load contract drifted: {observed}")
    return {
        "origin": origin,
        "commit": artifacts.commit,
        "architecture_sha256": artifacts.architecture_sha256,
        "config_sha256": artifacts.config_sha256,
        "checkpoint_sha256": artifacts.checkpoint_sha256,
        "checkpoint_kind": artifacts.checkpoint_metadata.get("kind"),
        "strict_cpu_load": True,
        "cache_contract": TURTLE_CACHE_CONTRACT,
    }


def _selected_arms(value: str) -> tuple[str, ...]:
    return ("baseline", "turtle") if value == "all" else (value,)


def preflight(
    *, arms: Iterable[str] = ("baseline", "turtle"), check_output_available: bool = True
) -> dict[str, Any]:
    """Validate configs, artifacts, and the bounded dataset without CUDA."""

    selected = tuple(arms)
    if not selected or any(arm not in CONFIGS for arm in selected):
        raise ValueError(f"invalid arm selection: {selected}")
    configs = _load_configs()
    differences = _validate_pair_contract(configs["baseline"], configs["turtle"])
    if check_output_available:
        for arm in selected:
            output = OUTPUTS[arm]
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"refusing to overwrite {arm} output: {output}")

    baseline = configs["baseline"]
    turtle_cfg = configs["turtle"]
    droid = _require_artifact(
        baseline["tracking"].get("pretrained"),
        baseline["tracking"].get("pretrained_sha256"),
        EXPECTED_SHA256["droid"],
        "DROID checkpoint",
    )
    omnidata = _require_artifact(
        baseline["mono_prior"].get("depth_pretrained"),
        baseline["mono_prior"].get("depth_pretrained_sha256"),
        EXPECTED_SHA256["omnidata"],
        "Omnidata checkpoint",
    )
    evssm = _require_artifact(
        baseline.get("evssm_checkpoint"),
        baseline.get("evssm_checkpoint_sha256"),
        EXPECTED_SHA256["evssm"],
        "EVSSM checkpoint",
    )
    turtle_artifacts = _validate_turtle_artifacts(turtle_cfg)

    from src.utils.datasets import get_dataset
    from src.utils.eval_frames import (
        PREFIX_SMOKE_METRIC_SCOPE,
        clear_gt_metric_scope,
        clear_gt_source_indices,
        validate_clear_gt_protocol_scope,
    )

    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        dataset = get_dataset(baseline, device="cpu")
    finally:
        os.chdir(previous_cwd)
    if len(dataset) != 221:
        raise ValueError(f"bounded dataset has {len(dataset)} frames, expected 221")
    protocol = clear_gt_source_indices(baseline, dataset)
    if protocol is None or tuple(sorted(protocol)) != EXPECTED_FULL_PROTOCOL:
        raise ValueError("published fr2_xyz clear-GT protocol drifted")
    available = validate_clear_gt_protocol_scope(baseline, dataset)
    if available is None or tuple(sorted(available)) != EXPECTED_PREFIX:
        raise ValueError("bounded dataset does not expose the exact eleven-frame prefix")
    if clear_gt_metric_scope(baseline) != PREFIX_SMOKE_METRIC_SCOPE:
        raise ValueError("bounded metric label drifted")

    return {
        "schema": "unblur_slam.fr2_xyz_paired_official_online_budget_221_preflight.v1",
        "scope": {
            "official_online_optimization_budget": True,
            "complete_three_sequence_paper_benchmark": False,
            "paper_26k_offline_refinement": False,
            "source_first": 0,
            "source_last": 220,
            "source_count": 221,
            "clear_gt_metric_scope": PREFIX_SMOKE_METRIC_SCOPE,
            "clear_gt_source_indices": list(EXPECTED_PREFIX),
        },
        "paired_contract": {
            "seed": 43,
            "resolution_wh": [512, 384],
            "init_iterations": 1050,
            "mapping_iterations_per_keyframe": 100,
            "tracking_iterations": 100,
            "final_refine_iterations": 100,
            "online_plotting": False,
            "allowed_resolved_config_differences": differences,
            "baseline_resolved_sha256": _canonical_sha256(baseline),
            "turtle_resolved_sha256": _canonical_sha256(turtle_cfg),
        },
        "artifacts": {
            "droid": {"path": str(droid), "sha256": EXPECTED_SHA256["droid"]},
            "omnidata": {
                "path": str(omnidata),
                "sha256": EXPECTED_SHA256["omnidata"],
            },
            "evssm": {"path": str(evssm), "sha256": EXPECTED_SHA256["evssm"]},
            "official_turtle": turtle_artifacts,
        },
        "execution": {
            "selected_arms": list(selected),
            "sequential_same_physical_gpu": int(PHYSICAL_GPU),
            "process_device": "cuda:0",
            "outputs": {arm: str(OUTPUTS[arm]) for arm in selected},
        },
    }


def _interrupt(process: subprocess.Popen[str], log: Any) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
    try:
        return int(process.wait(timeout=30))
    except subprocess.TimeoutExpired:
        log.write("[launcher] SIGINT timeout; forwarding SIGTERM\n")
        log.flush()
        os.killpg(process.pid, signal.SIGTERM)
        return int(process.wait())


def _run_arm(arm: str, audit: Mapping[str, Any]) -> int:
    output = OUTPUTS[arm]
    output.mkdir(parents=True, exist_ok=False)
    (output / "preflight.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = [sys.executable, str(REPO_ROOT / "run.py"), str(CONFIGS[arm])]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": PHYSICAL_GPU,
            "PYTHONUNBUFFERED": "1",
            "UNBLUR_SKIP_NR_IQA": "1",
        }
    )
    log_path = output / "launch.log"
    started = time.monotonic()
    code = -1
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        log.write("[launcher] direct paired fr2_xyz source 0..220 arm=" + arm + "\n")
        log.write("[launcher] official_online_budget=true final_refine=100\n")
        log.write("[launcher] complete_paper_benchmark=false paper_26k=false\n")
        log.write("[launcher] replay=false causal=false framecrafter=false submaps=false\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            code = int(process.wait())
        except KeyboardInterrupt:
            log.write("[launcher] KeyboardInterrupt; forwarding SIGINT\n")
            code = _interrupt(process, log)
        log.write(f"[launcher] exit_code={code}\n")
    runtime = {
        "schema": "unblur_slam.external_wall_runtime.v1",
        "arm": arm,
        "wall_runtime_seconds": time.monotonic() - started,
        "exit_code": code,
        "physical_gpu": int(PHYSICAL_GPU),
        "process_device": "cuda:0",
    }
    (output / "launcher_runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return code


def run_pair(selection: str) -> int:
    arms = _selected_arms(selection)
    audit = preflight(arms=arms, check_output_available=True)
    for arm in arms:
        print(f"[launch] {arm}: physical GPU {PHYSICAL_GPU} -> process cuda:0")
        code = _run_arm(arm, audit)
        if code != 0:
            return code
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--preflight", action="store_true", help="CPU-only validation (default)"
    )
    action.add_argument("--run", action="store_true", help="launch the selected arm(s)")
    parser.add_argument(
        "--arm", choices=("baseline", "turtle", "all"), default="all"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.run:
            return run_pair(args.arm)
        print(
            json.dumps(
                preflight(arms=_selected_arms(args.arm)), indent=2, sort_keys=True
            )
        )
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

