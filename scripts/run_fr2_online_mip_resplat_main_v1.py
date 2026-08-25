#!/usr/bin/env python3
"""Fail-closed launcher for the three-arm online Mip/ReSplat main experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CFG = ROOT / "configs/unblur_slam.yaml"
CONFIG_ROOT = ROOT / "configs/local/fr2_xyz_online_mip_resplat_main_v1"
OUTPUT_ROOT = Path(
    "/srv/szha0669/unblur-slam/slam_paired/fr2_xyz_online_mip_resplat_main_v1"
)
LOCK = Path("/srv/szha0669/unblur-slam/locks/physical_gpu1.lock")
PYTHON = Path("/srv/szha0669/unblur-slam/env/bin/python")
WRAPPER = ROOT / "scripts/execute_pinned_gpu_command.py"
ARMS = (
    ("baseline", CONFIG_ROOT / "baseline.yaml"),
    ("mip_splatting", CONFIG_ROOT / "mip_splatting.yaml"),
    (
        "mip_splatting_plus_safe_resplat",
        CONFIG_ROOT / "mip_splatting_plus_safe_resplat.yaml",
    ),
)
EXPECTED_GPU = {
    "physical_index": "1",
    "visible_devices": "1",
    "logical_device": "cuda:0",
    "name": "NVIDIA RTX A6000",
    "uuid": "GPU-3501b285-78cd-1494-87f1-ccac2136866e",
    "serial": "1711224002341",
}
EXPECTED_DIFFS = {
    "mip_splatting": {
        "data.output",
        "mapping.mip_splatting.enabled",
    },
    "mip_splatting_plus_safe_resplat": {
        "data.output",
        "mapping.mip_splatting.enabled",
        "mapping.official_resplat_active_fusion.enabled",
    },
}
IMPLEMENTATION = {
    "launcher": Path(__file__).resolve(),
    "mapper": ROOT / "src/mapper.py",
    "gaussian_model": ROOT
    / "thirdparty/gaussian_splatting/scene/gaussian_model.py",
    "gaussian_renderer": ROOT
    / "thirdparty/gaussian_splatting/gaussian_renderer/__init__.py",
    "active_fusion": ROOT / "src/refinement/official_resplat_active_fusion.py",
    "active_merge": ROOT / "src/refinement/active_map_merge.py",
    "world_bridge": ROOT / "src/refinement/resplat_unblur_bridge.py",
    "sidecar_runner": ROOT / "scripts/run_official_resplat_sidecar.py",
    "gpu_wrapper": WRAPPER,
    "run_py": ROOT / "run.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, name))
        return result
    return {prefix: value}


def _resolved_configs() -> dict[str, Mapping[str, Any]]:
    from thirdparty.glorie_slam import config

    return {
        arm: config.load_config(path.resolve(), DEFAULT_CFG.resolve())
        for arm, path in ARMS
    }


def preflight() -> dict[str, Any]:
    for path in (*IMPLEMENTATION.values(), DEFAULT_CFG, *(path for _, path in ARMS)):
        if not path.is_file():
            raise FileNotFoundError(path)
    resolved = _resolved_configs()
    baseline = _flatten(resolved["baseline"])
    differences: dict[str, list[str]] = {}
    for arm in ("mip_splatting", "mip_splatting_plus_safe_resplat"):
        candidate = _flatten(resolved[arm])
        changed = {
            key
            for key in set(baseline) | set(candidate)
            if baseline.get(key) != candidate.get(key)
        }
        if changed != EXPECTED_DIFFS[arm]:
            raise ValueError(f"unexpected resolved config differences for {arm}: {changed}")
        differences[arm] = sorted(changed)
    for arm, cfg in resolved.items():
        protocol = cfg.get("online_mip_resplat_main", {})
        if (
            protocol.get("schema") != "unblur_slam.fr2_xyz_online_mip_resplat_main.v1"
            or protocol.get("unsafe_forced_commit") is not False
            or cfg["mapping"]["mip_splatting"]["filter_kernel_variance"] != 0.2
            or cfg["mapping"]["official_resplat_active_fusion"].get(
                "unsafe_force_commit_after_postmerge_rejection", False
            )
            is not False
        ):
            raise ValueError(f"scientific contract drifted for {arm}")
    renderer = IMPLEMENTATION["gaussian_renderer"].read_text(encoding="utf-8")
    model = IMPLEMENTATION["gaussian_model"].read_text(encoding="utf-8")
    mapper = IMPLEMENTATION["mapper"].read_text(encoding="utf-8")
    if renderer.count("antialiasing=_mip_splatting_enabled(pc)") != 2:
        raise ValueError("2D Mip-Splatting antialiasing wiring drifted")
    for fragment in (
        "def compute_3D_filter",
        "get_scaling_with_3D_filter",
        "get_opacity_with_3D_filter",
    ):
        if fragment not in model:
            raise ValueError(f"3D Mip-Splatting implementation is missing {fragment}")
    if mapper.count("_refresh_mip_splatting") < 4:
        raise ValueError("Mapper Mip-Splatting refresh wiring is incomplete")
    expected_outputs = {
        arm: str((OUTPUT_ROOT / arm).resolve()) for arm, _ in ARMS
    }
    observed_outputs = {
        arm: str(Path(str(cfg["data"]["output"])).resolve())
        for arm, cfg in resolved.items()
    }
    if observed_outputs != expected_outputs:
        raise ValueError("resolved output roots changed")
    return {
        "schema": "unblur_slam.fr2_xyz_online_mip_resplat_main_preflight.v1",
        "passed": True,
        "gpu_started": False,
        "output_root": str(OUTPUT_ROOT),
        "output_root_absent": not OUTPUT_ROOT.exists(),
        "gpu": EXPECTED_GPU,
        "resolved_differences": differences,
        "implementation": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in IMPLEMENTATION.items()
        },
        "configs": {
            arm: {"path": str(path), "sha256": sha256_file(path)}
            for arm, path in ARMS
        },
    }


def _wrapper_command(arm: str, cfg: Path) -> list[str]:
    audit = OUTPUT_ROOT / arm / "gpu_execution.json"
    return [
        str(PYTHON),
        str(WRAPPER),
        "--lock-file", str(LOCK),
        "--audit-report", str(audit),
        "--expected-physical-index", EXPECTED_GPU["physical_index"],
        "--expected-cuda-visible-devices", EXPECTED_GPU["visible_devices"],
        "--expected-gpu-name", EXPECTED_GPU["name"],
        "--expected-gpu-uuid", EXPECTED_GPU["uuid"],
        "--expected-gpu-serial", EXPECTED_GPU["serial"],
        "--",
        str(PYTHON),
        str(ROOT / "run.py"),
        str(cfg),
    ]


def run() -> dict[str, Any]:
    evidence = preflight()
    if not evidence["output_root_absent"]:
        raise FileExistsError(f"refusing to reuse experiment root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT_ROOT / "preflight.json", evidence)
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(OUTPUT_ROOT / ".pycache"),
            "TMPDIR": str(OUTPUT_ROOT / ".tmp"),
            "UNBLUR_SKIP_NR_IQA": "1",
        }
    )
    records = []
    for arm, cfg in ARMS:
        arm_root = OUTPUT_ROOT / arm
        arm_root.mkdir(parents=True, exist_ok=False)
        command = _wrapper_command(arm, cfg.resolve())
        started = time.monotonic()
        with (arm_root / "launch.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        record = {
            "arm": arm,
            "command": command,
            "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "exit_code": int(completed.returncode),
            "wall_seconds": time.monotonic() - started,
            "launch_log": str(arm_root / "launch.log"),
            "launch_log_sha256": sha256_file(arm_root / "launch.log"),
        }
        records.append(record)
        _atomic_json(arm_root / "launcher_result.json", record)
        if completed.returncode != 0:
            raise RuntimeError(f"formal arm {arm} failed with exit {completed.returncode}")
    terminal = {
        "schema": "unblur_slam.fr2_xyz_online_mip_resplat_main_launcher.v1",
        "status": "complete",
        "preflight_sha256": sha256_file(OUTPUT_ROOT / "preflight.json"),
        "arms": records,
    }
    _atomic_json(OUTPUT_ROOT / "launcher_terminal.json", terminal)
    return terminal


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run() if args.run else preflight()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
