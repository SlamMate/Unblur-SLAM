#!/usr/bin/env python3
"""Preflight or launch the historical residual-view-replay smoke matrix.

This is a legacy ablation launcher, not an official ReSplat entry point.  New
official-model experiments must use ``run_paired_official_resplat_smoke.py``.

Examples
--------
CPU-only contract check (the default)::

    /srv/szha0669/unblur-slam/env/bin/python \
        scripts/run_fr2_resplat_smoke.py --preflight

Launch exactly one arm on physical GPU 1 after preflight succeeds::

    /srv/szha0669/unblur-slam/env/bin/python \
        scripts/run_fr2_resplat_smoke.py --run baseline --gpu 1

This launcher does not resume or overwrite an existing scene output.  Delete or
archive a failed smoke output explicitly before relaunching it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thirdparty.glorie_slam import config as config_io  # noqa: E402
from src.utils.datasets import get_dataset  # noqa: E402
from src.utils.eval_frames import (  # noqa: E402
    available_clear_gt_source_indices,
    clear_gt_source_indices,
)


CONFIG_ROOT = REPO_ROOT / "configs" / "local" / "fr2_xyz_resplat_smoke"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "unblur_slam.yaml"
ARM_CONFIGS = {
    "baseline": CONFIG_ROOT / "baseline.yaml",
    "offline": CONFIG_ROOT / "offline_replay.yaml",
    "online": CONFIG_ROOT / "online_replay.yaml",
    "offline_online": CONFIG_ROOT / "offline_online_replay.yaml",
}
EXPECTED_CLEAR_GT = 42


def _load(arm: str) -> dict:
    return config_io.load_config(ARM_CONFIGS[arm], DEFAULT_CONFIG)


def _resolved_output(cfg: dict) -> Path:
    return (Path(cfg["data"]["output"]) / str(cfg["scene"])).resolve()


def _shared_contract(cfg: dict) -> tuple:
    training = cfg["mapping"]["Training"]
    return (
        int(cfg["max_frames"]),
        int(cfg["stride"]),
        int(cfg["setup_seed"]),
        int(cfg["cam"]["H_out"]),
        int(cfg["cam"]["W_out"]),
        int(cfg["mapping"]["final_refine_iters"]),
        int(training["init_itr_num"]),
        int(training["mapping_itr_num"]),
        int(training["tracking_itr_num"]),
        bool(cfg["mapping"].get("hydrate_missing_droid_keyframes", False)),
        str(cfg["deblur"]["frontend"]),
        str(cfg["evssm_checkpoint"]),
    )


def _assert_arm_switches(configs: dict[str, dict]) -> None:
    expected = {
        "baseline": (False, False, 0),
        "offline": (True, False, 100),
        "online": (True, True, 0),
        "offline_online": (True, True, 100),
    }
    for arm, values in expected.items():
        replay = configs[arm]["mapping"]["resplat"]
        actual = (
            bool(replay["enabled"]),
            bool(replay["online_enabled"]),
            int(replay["extra_iters"]),
        )
        if actual != values:
            raise ValueError(f"{arm}: replay switches {actual}, expected {values}")
        if str(replay["budget_mode"]) != "replace_tail":
            raise ValueError(f"{arm}: smoke must use equal-budget replace_tail")


def _validate_cpu_safe_runtime_contract(cfg: dict) -> None:
    """Validate this matrix without importing CUDA-only DROID extensions.

    Importing :mod:`run` imports ``droid_backends`` at module load time, which
    makes a nominal CPU preflight depend on a compiled CUDA extension.  These
    checks mirror the run.py contracts relevant to the replay-only configs and
    deliberately reject optional preprocessors/frontends outside this matrix.
    """

    if str(cfg.get("dataset", "")).lower() not in {"tumrgbd", "tumrgb"}:
        raise ValueError("fr2 replay smoke requires the TUM RGB-D dataset")
    if not bool(cfg.get("warmup_mapper", False)):
        raise ValueError("paper clear-GT coverage requires warmup_mapper=true")
    if not bool(
        cfg.get("mapping", {}).get("hydrate_missing_droid_keyframes", False)
    ):
        raise ValueError(
            "complete offline camera coverage requires "
            "mapping.hydrate_missing_droid_keyframes=true"
        )
    if bool((cfg.get("framecrafter", {}) or {}).get("enabled", False)):
        raise ValueError("FrameCrafter must remain disabled in the replay-only matrix")
    if str((cfg.get("deblur", {}) or {}).get("frontend", "evssm")).lower() != "evssm":
        raise ValueError("the four-arm matrix must hold the EVSSM frontend constant")

    input_root = (
        Path(cfg["data"]["dataset_root"]).expanduser()
        / str(cfg["data"]["input_folder"])
    ).resolve()
    required = {
        "TUM input": input_root,
        "DROID checkpoint": Path(cfg["tracking"]["pretrained"]).expanduser(),
        "Omnidata checkpoint": Path(cfg["mono_prior"]["depth_pretrained"]).expanduser(),
        "EVSSM checkpoint": Path(cfg["evssm_checkpoint"]).expanduser(),
    }
    for label, path in required.items():
        resolved = path.resolve()
        if label == "TUM input":
            valid = resolved.is_dir()
        else:
            valid = resolved.is_file()
        if not valid:
            raise FileNotFoundError(f"{label} does not exist: {resolved}")

    replay = cfg["mapping"]["resplat"]
    if str(replay.get("backend", "residual_replay")) != "residual_replay":
        raise ValueError("only residual_replay is implemented in this smoke")
    base_budget = int(cfg["mapping"]["final_refine_iters"])
    replay_iters = int(replay.get("extra_iters", 0))
    if replay_iters < 0 or replay_iters > base_budget:
        raise ValueError("invalid equal-budget replay tail")
    if bool(replay.get("enabled", False)) and not bool(
        cfg["tracking"]["backend"].get("final_ba", False)
    ):
        raise ValueError("offline replay requires tracking.backend.final_ba=true")
    if not bool(cfg["tracking"]["backend"].get("final_ba", False)):
        raise ValueError("DROID keyframe hydration requires final_ba=true")


def preflight(arms: Iterable[str]) -> dict[str, dict]:
    selected = list(arms)
    configs = {arm: _load(arm) for arm in selected}
    all_configs = {arm: _load(arm) for arm in ARM_CONFIGS}

    contracts = {_shared_contract(cfg) for cfg in all_configs.values()}
    if len(contracts) != 1:
        raise ValueError(f"smoke arms do not share one compute/data contract: {contracts}")
    _assert_arm_switches(all_configs)

    outputs = [_resolved_output(cfg) for cfg in all_configs.values()]
    if len(set(outputs)) != len(outputs):
        raise ValueError(f"smoke outputs are not isolated: {outputs}")
    if any(str(path).startswith(str(REPO_ROOT)) for path in outputs):
        raise ValueError("smoke outputs must stay on /srv, not the full /home filesystem")

    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        for arm, cfg in configs.items():
            _validate_cpu_safe_runtime_contract(cfg)
            dataset = get_dataset(cfg, device="cpu")
            protocol = clear_gt_source_indices(cfg, dataset)
            available = available_clear_gt_source_indices(cfg, dataset)
            if len(dataset) != int(cfg["max_frames"]):
                raise ValueError(
                    f"{arm}: dataset has {len(dataset)} frames, expected "
                    f"max_frames={cfg['max_frames']}"
                )
            if (
                protocol is None
                or available is None
                or set(protocol) != set(available)
                or len(available) != EXPECTED_CLEAR_GT
            ):
                raise ValueError(
                    f"{arm}: expected {EXPECTED_CLEAR_GT} legal clear-GT frames, "
                    f"got {None if available is None else len(available)}"
                )
            print(
                f"[preflight] {arm}: source={len(dataset)}, clear_gt={len(available)}, "
                f"resolution={cfg['cam']['W_out']}x{cfg['cam']['H_out']}, "
                f"final_iters={cfg['mapping']['final_refine_iters']}, "
                f"output={_resolved_output(cfg)}"
            )
    finally:
        os.chdir(previous_cwd)
    return configs


def run_arm(arm: str, gpu: str) -> int:
    configs = preflight([arm])
    cfg = configs[arm]
    scene_output = _resolved_output(cfg)
    if scene_output.exists() and any(scene_output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty smoke output: {scene_output}"
        )

    scene_output.parent.mkdir(parents=True, exist_ok=True)
    log_path = scene_output.parent / "launch.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [sys.executable, str(REPO_ROOT / "run.py"), str(ARM_CONFIGS[arm])]
    print(f"[launch] CUDA_VISIBLE_DEVICES={gpu} {' '.join(command)}")
    print(f"[launch] log={log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return int(process.wait())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--preflight",
        action="store_true",
        help="CPU-only validation (default; never starts SLAM/GPU workers)",
    )
    action.add_argument("--run", choices=sorted(ARM_CONFIGS), help="launch one arm")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=sorted(ARM_CONFIGS),
        default=sorted(ARM_CONFIGS),
        help="arms checked by --preflight",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="physical CUDA device exposed as cuda:0 for --run (for example 1)",
    )
    args = parser.parse_args()

    if args.run:
        return run_arm(args.run, args.gpu)
    preflight(args.arms)
    print("[preflight] PASS: no GPU process was started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
