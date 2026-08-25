#!/usr/bin/env python3
"""Fail-closed physical-GPU identity checks for the offline fair benchmark.

This module intentionally uses ``nvidia-smi`` rather than CUDA so a launcher can
verify and lock the requested physical board before its child creates a CUDA
context.  CUDA workers additionally require the exact one-device visibility
mask and therefore address the pinned board as logical ``cuda:0``.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Mapping


def _run_nvidia_smi(query: str) -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-{query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"cannot inspect NVIDIA GPU state ({query})") from error
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def inspect_physical_gpus() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in _run_nvidia_smi("gpu=index,name,uuid,serial,memory.total"):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        records.append(
            {
                "physical_index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "serial": parts[3],
                "memory_total_mib": int(parts[4]),
            }
        )
    if not records:
        raise RuntimeError("nvidia-smi reported no physical GPUs")
    return records


def compute_processes_for_uuid(uuid: str) -> list[int]:
    processes: list[int] = []
    for line in _run_nvidia_smi("compute-apps=gpu_uuid,pid"):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise RuntimeError(f"unexpected nvidia-smi compute row: {line!r}")
        if parts[0] == uuid:
            processes.append(int(parts[1]))
    return sorted(set(processes))


def validate_gpu_contract(
    expected: Mapping[str, Any],
    *,
    require_visible_mask: bool,
    require_idle: bool,
) -> dict[str, Any]:
    required = {
        "physical_index",
        "visible_devices",
        "logical_device",
        "name",
        "uuid",
        "serial",
    }
    missing = sorted(required - set(expected))
    if missing:
        raise ValueError(f"GPU contract lacks fields: {missing}")
    if str(expected["logical_device"]) != "cuda:0":
        raise ValueError("formal GPU contract requires logical_device=cuda:0")
    if str(expected["visible_devices"]) != str(int(expected["physical_index"])):
        raise ValueError("CUDA visibility must expose only the pinned physical index")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if require_visible_mask and visible != str(expected["visible_devices"]):
        raise RuntimeError(
            "wrong CUDA_VISIBLE_DEVICES: "
            f"expected {expected['visible_devices']!r}, got {visible!r}"
        )
    if require_visible_mask and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("formal CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    matches = [
        record
        for record in inspect_physical_gpus()
        if record["physical_index"] == int(expected["physical_index"])
    ]
    if len(matches) != 1:
        raise RuntimeError("pinned physical GPU index is absent or ambiguous")
    observed = matches[0]
    for key in ("name", "uuid", "serial"):
        if str(observed[key]) != str(expected[key]):
            raise RuntimeError(
                f"pinned GPU {key} mismatch: expected {expected[key]!r}, "
                f"got {observed[key]!r}"
            )
    active = compute_processes_for_uuid(str(expected["uuid"])) if require_idle else []
    if active:
        raise RuntimeError(
            f"pinned GPU is not idle; compute PIDs on {expected['uuid']}: {active}"
        )
    return {
        **observed,
        "visible_devices": str(expected["visible_devices"]),
        "logical_device": "cuda:0",
        "visibility_mask_verified": bool(require_visible_mask),
        "cuda_device_order": "PCI_BUS_ID",
        "idle_verified_before_launch": bool(require_idle),
        "active_compute_pids_before_launch": active,
    }
