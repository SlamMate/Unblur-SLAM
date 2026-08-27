#!/usr/bin/env python3
"""Run one argv-safe command while exclusively holding both pinned A6000 GPUs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from offline_fair_gpu_contract import compute_processes_for_uuid, inspect_physical_gpus


REPORT_SCHEMA = "unblur_slam.pinned_dual_gpu_command.v1"
LOCKS = (
    Path("/srv/szha0669/unblur-slam/locks/physical_gpu0.lock"),
    Path("/srv/szha0669/unblur-slam/locks/physical_gpu1.lock"),
)
EXPECTED = (
    {
        "physical_index": 0,
        "name": "NVIDIA RTX A6000",
        "uuid": "GPU-05ade78f-34d4-87ae-ba2f-3c3f42968c48",
        "serial": "1711224002323",
    },
    {
        "physical_index": 1,
        "name": "NVIDIA RTX A6000",
        "uuid": "GPU-3501b285-78cd-1494-87f1-ccac2136866e",
        "serial": "1711224002341",
    },
)

# These two A6000s sit behind a SYS PCIe/CPU path.  CUDA reports peer access,
# but a real two-rank NCCL all-reduce hangs indefinitely on NCCL's default P2P
# transport.  The pinned local launcher therefore uses NCCL shared memory and
# records the exact transport override in every runtime receipt.  This changes
# transport only; DDP's reduced gradients and global-batch semantics are
# unchanged.
NCCL_TRANSPORT_ENV = {
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_idle_pair() -> list[dict[str, Any]]:
    inventory = {row["physical_index"]: row for row in inspect_physical_gpus()}
    bindings: list[dict[str, Any]] = []
    for expected in EXPECTED:
        observed = inventory.get(expected["physical_index"])
        if observed is None:
            raise RuntimeError(f"physical GPU {expected['physical_index']} is absent")
        for key in ("name", "uuid", "serial"):
            if str(observed[key]) != str(expected[key]):
                raise RuntimeError(
                    f"physical GPU {expected['physical_index']} {key} mismatch"
                )
        active = compute_processes_for_uuid(str(expected["uuid"]))
        if active:
            raise RuntimeError(
                f"physical GPU {expected['physical_index']} is not idle: {active}"
            )
        bindings.append({
            **observed,
            "logical_device": f"cuda:{expected['physical_index']}",
            "active_compute_pids_before_launch": active,
            "idle_verified_before_launch": True,
        })
    return bindings


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    audit = args.audit_report.expanduser().resolve()
    if not str(audit).startswith("/srv/"):
        raise ValueError("audit report must be under /srv")
    if audit.exists() or audit.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit report: {audit}")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("a non-empty argv-safe command is required")

    handles = []
    try:
        # The stable physical_gpu0/1 lock namespace is shared with single-GPU
        # experiments.  Acquire in index order before any NVIDIA/CUDA query.
        for lock in LOCKS:
            lock.parent.mkdir(parents=True, exist_ok=True)
            handle = lock.open("a+", encoding="utf-8")
            handles.append(handle)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"GPU lock is already held: {lock}") from error
        bindings = _validate_idle_pair()
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = "0,1"
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment.update(NCCL_TRANSPORT_ENV)
        started = time.perf_counter()
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, env=environment, check=False
        )
        wall = time.perf_counter() - started
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete" if result.returncode == 0 else "failed",
        "wrapper": {"path": str(Path(__file__).resolve()),
                    "sha256": _sha256(Path(__file__).resolve())},
        "command": command,
        "command_sha256": _command_sha256(command),
        "working_directory": str(Path.cwd().resolve()),
        "returncode": result.returncode,
        "full_child_process_wall_seconds": wall,
        "exclusive_locks": [str(path) for path in LOCKS],
        "child_environment": {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            **NCCL_TRANSPORT_ENV,
        },
        "distributed": {
            "world_size": 2,
            "logical_devices": ["cuda:0", "cuda:1"],
            "physical_gpu_bindings": bindings,
        },
    }
    _publish_json(audit, report)
    return report, result.returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report, returncode = run(parse_args(argv))
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
