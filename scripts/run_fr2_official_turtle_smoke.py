#!/usr/bin/env python3
"""CPU-preflight or launch the 221-frame official TURTLE-only SLAM smoke.

This is a single treatment arm, not an extension of the historical causal or
residual-replay smoke matrices.  It uses the upstream Ascend-Research/Turtle
GoPro checkpoint with persistent K/V state on every source frame 0..220 and
then runs exactly 100 ordinary final-refinement iterations.

The eleven available clear-GT frames are an explicitly bounded smoke metric;
they are not the complete 42-frame TUM paper protocol.
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
from typing import Any, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thirdparty.glorie_slam import config as config_io  # noqa: E402
from src.turtle_backend import (  # noqa: E402
    PINNED_TURTLE_ARCH_SHA256,
    PINNED_TURTLE_CHECKPOINT_SHA256,
    PINNED_TURTLE_COMMIT,
    PINNED_TURTLE_CONFIG_SHA256,
    TURTLE_CACHE_CONTRACT,
    validate_turtle_artifacts,
)
from src.utils.datasets import get_dataset  # noqa: E402
from src.utils.eval_frames import (  # noqa: E402
    PREFIX_SMOKE_METRIC_SCOPE,
    clear_gt_metric_scope,
    clear_gt_source_indices,
    validate_clear_gt_protocol_scope,
)


CONFIG = REPO_ROOT / "configs/local/fr2_xyz_causal_smoke/turtle_official.yaml"
DEFAULT_CONFIG = REPO_ROOT / "configs/unblur_slam.yaml"
OUTPUT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_smoke/"
    "fr2_xyz_official_turtle_smoke/gopro_stream_221"
).resolve()
EXPECTED_FULL_PROTOCOL = (
    0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220, 319, 374, 407,
    435, 470, 483, 523, 568, 704, 750, 789, 827, 926, 1004, 1160,
    1251, 1342, 1409, 1460, 1553, 1692, 1795, 1889, 1978, 2055,
    2206, 2282, 2358, 2425, 2590, 2764,
)
EXPECTED_PREFIX = EXPECTED_FULL_PROTOCOL[:11]
EXPECTED_DROID_SHA256 = (
    "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526"
)
EXPECTED_OMNIDATA_SHA256 = (
    "a0fab23fee64aa9e4bbe0b520b18b196ea7594a7f719c1d8c10cf11dcb6e4a1e"
)
OFFICIAL_TURTLE_ORIGIN = "https://github.com/Ascend-Research/Turtle"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_artifact(
    path_value: object, sha_value: object, expected_sha256: str, label: str
) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    configured = str(sha_value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", configured):
        raise ValueError(f"{label} configured SHA-256 is invalid")
    if configured != expected_sha256:
        raise ValueError(f"{label} config is not pinned to the expected artifact")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} bytes changed: expected {expected_sha256}, got {actual}")
    return path


def _git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot inspect official TURTLE checkout: {repo}") from error
    return result.stdout.strip()


def _normalize_url(value: str) -> str:
    result = str(value).strip().rstrip("/")
    if result.endswith(".git"):
        result = result[:-4]
    return result.lower()


def _load() -> dict[str, Any]:
    return config_io.load_config(str(CONFIG), str(DEFAULT_CONFIG))


def _validate_static_contract(cfg: Mapping[str, Any]) -> None:
    expected_common = (
        str(cfg.get("dataset", "")).lower(),
        str(cfg.get("scene", "")),
        int(cfg.get("max_frames", -1)),
        int(cfg.get("stride", -1)),
        int(cfg.get("setup_seed", -1)),
        int(cfg.get("cam", {}).get("W_out", -1)),
        int(cfg.get("cam", {}).get("H_out", -1)),
        int(cfg.get("mapping", {}).get("final_refine_iters", -1)),
    )
    if expected_common != (
        "tumrgbd", "freiburg2_xyz", 221, 1, 43, 512, 384, 100
    ):
        raise ValueError(f"bounded official TURTLE smoke contract drifted: {expected_common}")
    if str(cfg.get("device")) != "cuda:0":
        raise ValueError("SLAM process device must be cuda:0 after GPU visibility mapping")
    if not bool(cfg.get("warmup_mapper", False)) or bool(cfg.get("clear_init", True)):
        raise ValueError("official TURTLE smoke requires warmup_mapper=true, clear_init=false")
    if bool((cfg.get("framecrafter", {}) or {}).get("enabled", False)):
        raise ValueError("FrameCrafter must be disabled in the official TURTLE smoke")
    if bool((cfg.get("submaps", {}) or {}).get("enabled", False)):
        raise ValueError("custom submaps must be disabled in the official TURTLE smoke")

    deblur = cfg.get("deblur")
    if not isinstance(deblur, Mapping) or deblur.get("frontend") != "turtle_streaming":
        raise ValueError("the only active deblur frontend must be turtle_streaming")
    if str(deblur.get("causal_checkpoint", "")):
        raise ValueError("custom causal checkpoint must be empty")
    if str(cfg.get("evssm_checkpoint", "")) or str(cfg.get("evssm_checkpoint_sha256", "")):
        raise ValueError("EVSSM artifacts must be absent from the official TURTLE-only arm")
    if (
        deblur.get("stream_every_frame") is not True
        or deblur.get("stream_apply_to_tracking") is not True
        or deblur.get("stream_replace_sharp") is not False
        or float(deblur.get("stream_min_laplacian_gain", -1.0)) != 0.02
    ):
        raise ValueError("official TURTLE stream/gating settings drifted")

    replay = cfg.get("mapping", {}).get("resplat")
    if not isinstance(replay, Mapping):
        raise ValueError("disabled legacy mapping.resplat config is missing")
    if (
        replay.get("enabled") is not False
        or replay.get("online_enabled") is not False
        or int(replay.get("extra_iters", -1)) != 0
        or replay.get("log_csv") is not False
        or replay.get("save_full_state") is not False
    ):
        raise ValueError("legacy residual replay must be fully disabled")

    if Path(str(cfg.get("data", {}).get("output", ""))).expanduser().resolve() != OUTPUT_ROOT:
        raise ValueError("official TURTLE arm output root drifted")
    evaluation = cfg.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("clear_gt_scope") != "prefix_smoke"
        or tuple(evaluation.get("expected_clear_gt_source_indices", ())) != EXPECTED_PREFIX
        or clear_gt_metric_scope(cfg) != PREFIX_SMOKE_METRIC_SCOPE
    ):
        raise ValueError("eleven-frame clear-GT prefix contract drifted")

    metadata = cfg.get("official_turtle_smoke")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != (
        "unblur_slam.fr2_xyz_official_turtle_smoke.v1"
    ):
        raise ValueError("official TURTLE smoke audit metadata is missing")
    expected_metadata = {
        "stream_source_first": 0,
        "stream_source_last": 220,
        "expected_stream_steps": 221,
        "old_mapping_resplat_used": False,
        "residual_replay_used": False,
        "causal_evssm_used": False,
        "evssm_used": False,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"official_turtle_smoke.{key} must equal {expected!r}")
    if tuple(metadata.get("expected_clear_gt_source_indices", ())) != EXPECTED_PREFIX:
        raise ValueError("official TURTLE smoke metadata must pin the eleven GT frames")


def _validate_turtle(cfg: Mapping[str, Any]) -> dict[str, Any]:
    deblur = cfg["deblur"]
    artifacts = validate_turtle_artifacts(deblur, load_weights=True)
    origin = _git(artifacts.repo, "remote", "get-url", "origin")
    if _normalize_url(origin) != _normalize_url(OFFICIAL_TURTLE_ORIGIN):
        raise ValueError(f"TURTLE origin is not official: {origin}")
    if _git(artifacts.repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("official TURTLE checkout has tracked modifications")
    expected = (
        artifacts.commit,
        artifacts.architecture_sha256,
        artifacts.config_sha256,
        artifacts.checkpoint_sha256,
        artifacts.checkpoint_metadata.get("kind"),
    )
    wanted = (
        PINNED_TURTLE_COMMIT,
        PINNED_TURTLE_ARCH_SHA256,
        PINNED_TURTLE_CONFIG_SHA256,
        PINNED_TURTLE_CHECKPOINT_SHA256,
        "official_gopro",
    )
    if expected != wanted:
        raise ValueError(f"official TURTLE strict-load contract drifted: {expected}")
    return {
        "repository": {
            "path": str(artifacts.repo),
            "origin": origin,
            "commit": artifacts.commit,
            "tracked_worktree_clean": True,
        },
        "architecture": {
            "path": str(artifacts.architecture),
            "sha256": artifacts.architecture_sha256,
        },
        "config": {"path": str(artifacts.config), "sha256": artifacts.config_sha256},
        "checkpoint": {
            "path": str(artifacts.checkpoint),
            "sha256": artifacts.checkpoint_sha256,
            "metadata": dict(artifacts.checkpoint_metadata),
            "strict_cpu_load": True,
        },
        "cache_contract": TURTLE_CACHE_CONTRACT,
    }


def preflight(*, check_output_available: bool = True) -> dict[str, Any]:
    """Validate the complete arm on CPU without constructing a CUDA worker."""

    cfg = _load()
    _validate_static_contract(cfg)
    if check_output_available and (OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink()):
        raise FileExistsError(f"refusing to overwrite official TURTLE arm: {OUTPUT_ROOT}")

    droid = _require_artifact(
        cfg["tracking"].get("pretrained"),
        cfg["tracking"].get("pretrained_sha256"),
        EXPECTED_DROID_SHA256,
        "DROID checkpoint",
    )
    omnidata = _require_artifact(
        cfg["mono_prior"].get("depth_pretrained"),
        cfg["mono_prior"].get("depth_pretrained_sha256"),
        EXPECTED_OMNIDATA_SHA256,
        "Omnidata checkpoint",
    )
    turtle = _validate_turtle(cfg)

    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        dataset = get_dataset(cfg, device="cpu")
    finally:
        os.chdir(previous_cwd)
    if len(dataset) != 221:
        raise ValueError(f"official TURTLE dataset has {len(dataset)} frames, expected 221")
    protocol = clear_gt_source_indices(cfg, dataset)
    if protocol is None or tuple(sorted(protocol)) != EXPECTED_FULL_PROTOCOL:
        raise ValueError("published fr2_xyz clear-GT protocol drifted")
    available = validate_clear_gt_protocol_scope(cfg, dataset)
    if available is None or tuple(sorted(available)) != EXPECTED_PREFIX:
        raise ValueError("bounded dataset does not expose exactly eleven clear-GT frames")
    for source_index in range(221):
        metadata = dataset.frame_info(source_index)
        if (
            int(metadata.get("source_index", -1)) != source_index
            or bool(metadata.get("synthetic", True))
            or not bool(metadata.get("eval", False))
        ):
            raise ValueError(f"dataset frame {source_index} is not the expected original frame")

    return {
        "schema": "unblur_slam.fr2_xyz_official_turtle_smoke_preflight.v1",
        "config": {"path": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "source": {
            "count": 221,
            "source_first": 0,
            "source_last": 220,
            "clear_gt_metric_scope": PREFIX_SMOKE_METRIC_SCOPE,
            "clear_gt_source_indices": list(EXPECTED_PREFIX),
            "clear_gt_count": len(EXPECTED_PREFIX),
            "complete_paper_metric": False,
        },
        "frontend": {
            "name": "turtle_streaming",
            "official_turtle": turtle,
            "every_source_frame_updates_cache": True,
            "expected_cache_updates": 221,
            "process_device": "cuda:0",
        },
        "slam": {
            "droid": {"path": str(droid), "sha256": EXPECTED_DROID_SHA256},
            "omnidata": {"path": str(omnidata), "sha256": EXPECTED_OMNIDATA_SHA256},
            "final_refine": {"uniform_iters": 100, "legacy_replay_iters": 0},
            "old_mapping_resplat_used": False,
            "causal_evssm_used": False,
            "evssm_used": False,
        },
        "execution": {
            "physical_gpu": 1,
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices": "1",
            "process_device": "cuda:0",
        },
        "output": str(OUTPUT_ROOT),
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


def run_smoke() -> int:
    audit = preflight(check_output_available=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    log_path = OUTPUT_ROOT / "launch.log"
    preflight_path = OUTPUT_ROOT / "preflight.json"
    preflight_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [sys.executable, str(REPO_ROOT / "run.py"), str(CONFIG)]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTHONUNBUFFERED": "1",
            "UNBLUR_SKIP_NR_IQA": "1",
        }
    )
    print("[launch] physical GPU 1 -> process cuda:0")
    print("[launch] " + " ".join(command))
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        log.write("[launcher] official TURTLE-only fr2_xyz 221-frame smoke\n")
        log.write("[launcher] physical_gpu=1 process_device=cuda:0\n")
        log.write("[launcher] residual_replay=false causal_evssm=false evssm=false\n")
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
        return code


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true", help="CPU-only validation (default)")
    action.add_argument("--run", action="store_true", help="launch on physical GPU 1")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.run:
            return run_smoke()
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
