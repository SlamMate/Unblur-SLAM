#!/usr/bin/env python3
"""Run one command under an exclusive lock on the pinned physical GPU."""

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
from offline_fair_gpu_contract import validate_gpu_contract


REPORT_SCHEMA = "unblur_slam.pinned_gpu_command.v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A pre-check followed by rename(2) is not a no-overwrite operation for
        # regular files: another process could publish ``path`` between the two
        # calls and be silently replaced.  A same-directory hard link publishes
        # the already-fsynced inode atomically and fails with EEXIST on that race.
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing concurrent overwrite: {path}") from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _command_sha256(command: Sequence[str]) -> str:
    encoded = json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    audit_report = args.audit_report.expanduser().resolve()
    lock_file = args.lock_file.expanduser().resolve()
    if not str(audit_report).startswith("/srv/") or not str(lock_file).startswith("/srv/"):
        raise ValueError("GPU audit report and lock must both be under /srv")
    if audit_report.exists() or audit_report.is_symlink():
        raise FileExistsError(f"refusing to overwrite GPU audit report: {audit_report}")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command or not all(isinstance(value, str) and value for value in command):
        raise ValueError("a non-empty argv-safe child command is required")
    expected = {
        "physical_index": args.expected_physical_index,
        "visible_devices": args.expected_cuda_visible_devices,
        "logical_device": "cuda:0",
        "name": args.expected_gpu_name,
        "uuid": args.expected_gpu_uuid,
        "serial": args.expected_gpu_serial,
    }
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"pinned GPU lock is already held: {lock_file}") from error
        binding = validate_gpu_contract(
            expected, require_visible_mask=False, require_idle=True
        )
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(args.expected_cuda_visible_devices)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        started = time.perf_counter()
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            env=environment,
            check=False,
        )
        wall = time.perf_counter() - started
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete" if result.returncode == 0 else "failed",
        "wrapper": {
            "path": str(Path(__file__).resolve()),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "command": command,
        "command_sha256": _command_sha256(command),
        "working_directory": str(Path.cwd().resolve()),
        "returncode": result.returncode,
        "full_child_process_wall_seconds": wall,
        "exclusive_lock": str(lock_file),
        "sequential_execution": True,
        "child_environment": {
            "CUDA_VISIBLE_DEVICES": str(args.expected_cuda_visible_devices),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        },
        "gpu": binding,
    }
    _atomic_json(audit_report, report)
    return report, result.returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--expected-physical-index", type=int, required=True)
    parser.add_argument("--expected-cuda-visible-devices", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-gpu-serial", required=True)
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
