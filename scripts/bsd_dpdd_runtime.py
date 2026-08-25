#!/usr/bin/env python3
"""Fail-closed runtime, provenance, and locking for the BSD+DPDD study.

This module deliberately has no top-level torch import.  A formal executor must
first take the single physical-GPU1 lease, re-hash its frozen launch bundle,
verify the pinned Python environment, and inspect physical GPU1 with
``nvidia-smi``.  Only then may :func:`require_gpu1_a6000` import torch and map
the one visible device to logical ``cuda:0``.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping, Sequence


EXPECTED_VISIBLE_DEVICE = "1"
EXPECTED_CUDA_DEVICE_ORDER = "PCI_BUS_ID"
EXPECTED_SCRIPT_DEVICE = "cuda:0"
EXPECTED_GPU_NAME = "NVIDIA RTX A6000"
EXPECTED_GPU_UUID = "GPU-3501b285-78cd-1494-87f1-ccac2136866e"
EXPECTED_GPU_SERIAL = "1711224002341"
MINIMUM_TOTAL_MEMORY_BYTES = 47 * 1024**3
DEFAULT_GLOBAL_LOCK = Path(
    "/srv/szha0669/unblur-slam/locks/physical_gpu1.lock"
)
PINNED_PYTHON = Path("/srv/szha0669/unblur-slam/env/bin/python")
PINNED_PYTHON_REALPATH = Path(
    "/srv/szha0669/unblur-slam/env/bin/python3.10"
)
PINNED_PYTHON_SHA256 = (
    "3aecd4f06769d24b66d327d33f95b72b59759a77756de09ff07270cf4ef875d7"
)
EXPECTED_PYTHON_VERSION = "3.10.20"
EXPECTED_TORCH_VERSION = "2.3.1"
EXPECTED_TORCH_CUDA_BUILD = "12.1"
EXPECTED_CUDNN_VERSION = 8902
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
PINNED_NVIDIA_SMI_SHA256 = (
    "4859c3eb3c53f4dbedaaf5f22f144bf9f16acb823aec483f846679080c947e72"
)

REQUIRED_DISTRIBUTIONS = {
    "einops": "0.8.2",
    "lpips": "0.1.4",
    "mamba-ssm": "2.2.2",
    "numpy": "1.26.4",
    "opencv-python": "4.8.1.78",
    "pillow": "12.3.0",
    "pyyaml": "6.0.3",
    "scikit-image": "0.25.2",
    "torch": EXPECTED_TORCH_VERSION,
    "torchmetrics": "1.9.0",
    "torchvision": "0.18.1",
}

LOCK_SCHEMA = "unblur_slam.physical_gpu1_lock_lease.v1"
PHYSICAL_GPU_SCHEMA = "unblur_slam.physical_gpu1_identity.v1"
RUNTIME_IDENTITY_SCHEMA = "unblur_slam.bsd_dpdd_cuda_runtime_identity.v1"
ENVIRONMENT_SCHEMA = "unblur_slam.bsd_dpdd_environment_fingerprint.v1"
CODE_BUNDLE_SCHEMA = "unblur_slam.bsd_dpdd_code_bundle.v1"
LOCK_PATH_ENV = "UNBLUR_SLAM_PHYSICAL_GPU1_LOCK_PATH"
LOCK_OWNER_PID_ENV = "UNBLUR_SLAM_PHYSICAL_GPU1_LOCK_OWNER_PID"
LOCK_LEASE_ENV = "UNBLUR_SLAM_PHYSICAL_GPU1_LOCK_LEASE"
ENVIRONMENT_FINGERPRINT_ENV = "UNBLUR_SLAM_ENVIRONMENT_FINGERPRINT_SHA256"
CODE_BUNDLE_FINGERPRINT_ENV = "UNBLUR_SLAM_CODE_BUNDLE_SHA256"

# These are injected by the executor before a child interpreter starts.  They
# are intentionally separate from the plan's exact two-variable GPU mapping.
CHILD_RUNTIME_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


class RuntimeContractError(RuntimeError):
    """Raised before pixel/model work when the physical runtime is wrong."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def _sha256_file(path: Path | str) -> str:
    candidate = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def validate_static_runtime_contract(runtime: Mapping[str, Any]) -> None:
    expected = {
        "physical_gpu": 1,
        "visible_device_environment": "CUDA_VISIBLE_DEVICES=1",
        "cuda_device_order_environment": "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "script_device": EXPECTED_SCRIPT_DEVICE,
        "expected_gpu_model": EXPECTED_GPU_NAME,
        "expected_gpu_uuid": EXPECTED_GPU_UUID,
        "expected_gpu_serial": EXPECTED_GPU_SERIAL,
        "minimum_total_memory_bytes": MINIMUM_TOTAL_MEMORY_BYTES,
        "global_lock": str(DEFAULT_GLOBAL_LOCK),
        "lock_acquired_before_cuda_query_torch_import_model_or_pixels": True,
        "require_idle_no_compute_pid_on_lock_entry": True,
        "monitor_between_and_during_actions_fail_on_unfamiliar_pid": True,
        "logical_identity_required_in_every_report_and_receipt": True,
        "executor_runs_commands": True,
        "large_artifacts_on_srv_only": True,
    }
    if dict(runtime) != expected:
        raise RuntimeContractError(
            f"runtime contract changed: observed={dict(runtime)!r}, expected={expected!r}"
        )


def require_exact_cuda_environment(*, formal: bool = True) -> Mapping[str, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != EXPECTED_VISIBLE_DEVICE:
        raise RuntimeContractError(
            "formal BSD+DPDD work requires exactly CUDA_VISIBLE_DEVICES=1; "
            f"observed {visible!r}"
        )
    order = os.environ.get("CUDA_DEVICE_ORDER")
    if formal and order != EXPECTED_CUDA_DEVICE_ORDER:
        raise RuntimeContractError(
            "formal BSD+DPDD work requires exactly "
            f"CUDA_DEVICE_ORDER={EXPECTED_CUDA_DEVICE_ORDER}; observed {order!r}"
        )
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "CUDA_DEVICE_ORDER": order or "",
    }


def build_environment_fingerprint() -> Mapping[str, Any]:
    """Fingerprint the interpreter and installed distributions without torch import."""

    executable = Path(sys.executable).expanduser().absolute()
    realpath = executable.resolve()
    _require(executable == PINNED_PYTHON, f"Python executable changed: {executable}")
    _require(realpath == PINNED_PYTHON_REALPATH, f"Python realpath changed: {realpath}")
    _require(
        _sha256_file(executable) == PINNED_PYTHON_SHA256,
        "Python executable SHA256 changed",
    )
    _require(
        platform.python_implementation() == "CPython",
        "formal interpreter must be CPython",
    )
    _require(
        platform.python_version() == EXPECTED_PYTHON_VERSION,
        f"Python version changed: {platform.python_version()}",
    )
    _require(NVIDIA_SMI.is_file(), f"pinned nvidia-smi is missing: {NVIDIA_SMI}")
    _require(
        _sha256_file(NVIDIA_SMI) == PINNED_NVIDIA_SMI_SHA256,
        "nvidia-smi executable SHA256 changed",
    )

    installed_rows: list[tuple[str, str]] = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed_rows.append(
                (_normalized_distribution_name(str(name)), str(distribution.version))
            )
    installed_rows.sort()
    installed: dict[str, str] = {}
    duplicates: set[str] = set()
    for name, version in installed_rows:
        if name in installed and installed[name] != version:
            duplicates.add(name)
        installed[name] = version
    _require(not duplicates, f"ambiguous installed distributions: {sorted(duplicates)}")
    observed_required = {
        name: installed.get(name) for name in sorted(REQUIRED_DISTRIBUTIONS)
    }
    _require(
        observed_required == dict(sorted(REQUIRED_DISTRIBUTIONS.items())),
        "required package versions changed: "
        f"observed={observed_required!r}, expected={REQUIRED_DISTRIBUTIONS!r}",
    )

    torch_distribution = metadata.distribution("torch")
    torch_version_file = Path(
        torch_distribution.locate_file("torch/version.py")
    ).resolve()
    _require(torch_version_file.is_file(), "torch/version.py is missing")
    version_source = torch_version_file.read_text(encoding="utf-8")
    cuda_match = re.search(r"^cuda:.*= ['\"]([^'\"]+)['\"]$", version_source, re.M)
    git_match = re.search(r"^git_version = ['\"]([^'\"]+)['\"]$", version_source, re.M)
    _require(cuda_match is not None, "torch CUDA build could not be read without import")
    _require(git_match is not None, "torch git revision could not be read without import")
    _require(
        cuda_match.group(1) == EXPECTED_TORCH_CUDA_BUILD,
        f"torch CUDA build changed: {cuda_match.group(1)}",
    )

    payload: dict[str, Any] = {
        "schema": ENVIRONMENT_SCHEMA,
        "python": {
            "executable": str(executable),
            "realpath": str(realpath),
            "executable_sha256": PINNED_PYTHON_SHA256,
            "implementation": "CPython",
            "version": EXPECTED_PYTHON_VERSION,
            "prefix": str(Path(sys.prefix).resolve()),
        },
        "required_distributions": observed_required,
        "all_installed_distributions_count": len(installed_rows),
        "all_installed_distributions_sha256": _canonical_sha256(installed_rows),
        "torch_build": {
            "version": installed["torch"],
            "cuda": cuda_match.group(1),
            "git_version": git_match.group(1),
            "version_file": str(torch_version_file),
            "version_file_sha256": _sha256_file(torch_version_file),
        },
        "nvidia_smi": {
            "path": str(NVIDIA_SMI),
            "sha256": PINNED_NVIDIA_SMI_SHA256,
        },
    }
    payload["fingerprint_sha256"] = _canonical_sha256(payload)
    return payload


def verify_frozen_environment(expected: Mapping[str, Any]) -> Mapping[str, Any]:
    observed = build_environment_fingerprint()
    _require(
        dict(expected) == dict(observed),
        "frozen Python/package environment fingerprint changed",
    )
    return observed


def rehash_frozen_code_bundle(
    *,
    implementation_pins: Mapping[str, Any],
    code_bundle: Mapping[str, Any],
    expected_path_map: Mapping[str, Path | str],
    base_root: Path | str | None = None,
) -> Mapping[str, Any]:
    """Re-hash every frozen implementation file from an independently pinned map."""

    _require(code_bundle.get("schema") == CODE_BUNDLE_SCHEMA, "code-bundle schema changed")
    files = code_bundle.get("files")
    _require(isinstance(files, Mapping), "code-bundle files mapping is missing")
    root = Path.cwd().resolve() if base_root is None else Path(base_root).resolve()

    def resolve(value: Path | str) -> str:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return str(candidate.resolve())

    expected_paths = {str(name): resolve(path) for name, path in expected_path_map.items()}
    observed_paths = {str(name): resolve(path) for name, path in files.items()}
    _require(observed_paths == expected_paths, "code-bundle path mapping changed")
    _require(
        set(implementation_pins) == set(expected_paths),
        "implementation pin names and code-bundle files differ",
    )
    _require(
        len(set(expected_paths.values())) == len(expected_paths),
        "two implementation pins resolve to the same file",
    )
    rows: list[dict[str, str]] = []
    for name in sorted(expected_paths):
        path = Path(expected_paths[name])
        _require(path.is_file(), f"frozen implementation file is missing: {path}")
        digest = _sha256_file(path)
        expected_digest = str(implementation_pins[name]).lower()
        _require(
            re.fullmatch(r"[0-9a-f]{64}", expected_digest) is not None,
            f"invalid implementation pin for {name}",
        )
        _require(digest == expected_digest, f"implementation pin changed for {name}")
        rows.append({"pin": name, "path": str(path), "sha256": digest})
    bundle_sha256 = _canonical_sha256(rows)
    _require(
        code_bundle.get("bundle_sha256") == bundle_sha256,
        "code-bundle aggregate SHA256 changed",
    )
    return {
        "schema": CODE_BUNDLE_SCHEMA,
        "status": "verified_inside_global_lock_before_cuda_or_pixels",
        "files": rows,
        "file_count": len(rows),
        "bundle_sha256": bundle_sha256,
    }


def _run_nvidia_smi(
    arguments: Sequence[str],
    *,
    command_runner: Callable[..., Any] | None = None,
) -> str:
    runner = subprocess.run if command_runner is None else command_runner
    command = [str(NVIDIA_SMI), *map(str, arguments)]
    try:
        result = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeContractError(f"nvidia-smi query failed: {command!r}") from error
    if int(result.returncode) != 0:
        raise RuntimeContractError(
            f"nvidia-smi query failed ({result.returncode}): {str(result.stderr).strip()}"
        )
    return str(result.stdout)


def inspect_physical_gpu1(
    *,
    require_idle: bool,
    command_runner: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Inspect nvidia-smi physical index 1 and optionally require no compute PID."""

    output = _run_nvidia_smi(
        [
            "--id=1",
            "--query-gpu=index,name,uuid,serial,memory.total",
            "--format=csv,noheader,nounits",
        ],
        command_runner=command_runner,
    )
    rows = [row for row in csv.reader(output.splitlines()) if row]
    _require(len(rows) == 1 and len(rows[0]) == 5, "physical GPU1 query shape changed")
    index, name, uuid, serial, memory_mib = (item.strip() for item in rows[0])
    _require(index == "1", f"nvidia-smi did not return physical index 1: {index!r}")
    _require(name == EXPECTED_GPU_NAME, f"physical GPU1 name changed: {name!r}")
    _require(uuid == EXPECTED_GPU_UUID, f"physical GPU1 UUID changed: {uuid!r}")
    _require(serial == EXPECTED_GPU_SERIAL, f"physical GPU1 serial changed: {serial!r}")
    try:
        memory_bytes = int(memory_mib) * 1024**2
    except ValueError as error:
        raise RuntimeContractError(f"invalid physical GPU1 memory: {memory_mib!r}") from error
    _require(
        memory_bytes >= MINIMUM_TOTAL_MEMORY_BYTES,
        f"physical GPU1 memory contract failed: {memory_bytes} bytes",
    )
    processes = query_gpu1_compute_processes(command_runner=command_runner)
    if require_idle:
        _require(
            not processes,
            "physical GPU1 was not idle on global-lock entry; compute PIDs="
            f"{[row['pid'] for row in processes]}",
        )
    return {
        "schema": PHYSICAL_GPU_SCHEMA,
        "physical_index": 1,
        "name": name,
        "uuid": uuid,
        "serial": serial,
        "nvidia_smi_total_memory_mib": int(memory_mib),
        "nvidia_smi_total_memory_bytes": memory_bytes,
        "idle_compute_processes_required": bool(require_idle),
        "compute_processes": processes,
    }


def query_gpu1_compute_processes(
    *, command_runner: Callable[..., Any] | None = None
) -> list[Mapping[str, Any]]:
    output = _run_nvidia_smi(
        [
            "--id=1",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        command_runner=command_runner,
    )
    records: list[Mapping[str, Any]] = []
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        _require(len(row) == 4, f"compute-process query shape changed: {row!r}")
        uuid, pid_text, process_name, used_memory = (item.strip() for item in row)
        _require(uuid == EXPECTED_GPU_UUID, f"compute process reported wrong UUID: {uuid!r}")
        try:
            pid = int(pid_text)
        except ValueError as error:
            raise RuntimeContractError(f"invalid GPU compute PID: {pid_text!r}") from error
        _require(pid > 0, f"invalid GPU compute PID: {pid}")
        records.append(
            {
                "gpu_uuid": uuid,
                "pid": pid,
                "process_name": process_name,
                "used_gpu_memory_mib": used_memory,
            }
        )
    records.sort(key=lambda item: int(item["pid"]))
    return records


def require_only_known_compute_pids(
    allowed_pids: Sequence[int],
    *,
    phase: str,
    command_runner: Callable[..., Any] | None = None,
) -> list[Mapping[str, Any]]:
    allowed = {int(pid) for pid in allowed_pids}
    observed = query_gpu1_compute_processes(command_runner=command_runner)
    unfamiliar = [row for row in observed if int(row["pid"]) not in allowed]
    _require(
        not unfamiliar,
        f"unfamiliar physical-GPU1 compute PID during {phase}: {unfamiliar!r}",
    )
    return observed


def _read_lock_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeContractError(f"invalid physical-GPU1 lock lease: {path}") from error
    _require(isinstance(payload, Mapping), "physical-GPU1 lock payload is not an object")
    _require(payload.get("schema") == LOCK_SCHEMA, "physical-GPU1 lock schema changed")
    return payload


def require_active_gpu1_lock(
    identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Prove this process is a child of the executor holding the global lease."""

    if identity is not None:
        path = Path(str(identity.get("path", ""))).expanduser().resolve()
        owner_pid = int(identity.get("owner_pid", -1))
        lease = str(identity.get("lease", ""))
        _require(owner_pid == os.getpid(), "in-process lock identity owner changed")
        payload = _read_lock_payload(path)
        _require(payload.get("owner_pid") == owner_pid, "lock owner PID changed")
        _require(payload.get("lease") == lease, "lock lease token changed")
        return payload

    path_text = os.environ.get(LOCK_PATH_ENV)
    owner_text = os.environ.get(LOCK_OWNER_PID_ENV)
    lease = os.environ.get(LOCK_LEASE_ENV)
    _require(path_text == str(DEFAULT_GLOBAL_LOCK), "child global-lock path proof is missing")
    _require(owner_text is not None and owner_text.isdigit(), "child lock owner proof is missing")
    _require(bool(lease), "child lock lease proof is missing")
    path = Path(path_text).resolve()
    payload = _read_lock_payload(path)
    owner_pid = int(owner_text)
    _require(payload.get("owner_pid") == owner_pid, "child lock owner PID changed")
    _require(payload.get("lease") == lease, "child lock lease token changed")
    try:
        os.kill(owner_pid, 0)
    except OSError as error:
        raise RuntimeContractError("global-lock owner process is not alive") from error
    descriptor = os.open(path, os.O_RDWR)
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        _require(not acquired, "global physical-GPU1 lock is not actually held")
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    return payload


def _runtime_fingerprint_from_environment(name: str) -> str:
    value = os.environ.get(name, "")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"formal runtime fingerprint environment is missing: {name}",
    )
    return value


def _configure_torch_runtime(torch_module: Any) -> Mapping[str, Any]:
    _require(
        str(torch_module.__version__).split("+")[0] == EXPECTED_TORCH_VERSION,
        f"torch version changed: {torch_module.__version__}",
    )
    _require(
        str(torch_module.version.cuda) == EXPECTED_TORCH_CUDA_BUILD,
        f"torch CUDA build changed: {torch_module.version.cuda}",
    )
    cudnn_version = int(torch_module.backends.cudnn.version())
    _require(
        cudnn_version == EXPECTED_CUDNN_VERSION,
        f"cuDNN version changed: {cudnn_version}",
    )
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before torch runtime use",
    )
    _require(
        os.environ.get("NVIDIA_TF32_OVERRIDE") == "0",
        "NVIDIA_TF32_OVERRIDE must be 0 before torch runtime use",
    )
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.set_deterministic_debug_mode("error")
    torch_module.set_float32_matmul_precision("highest")
    settings = {
        "torch_version": EXPECTED_TORCH_VERSION,
        "torch_cuda_build": EXPECTED_TORCH_CUDA_BUILD,
        "cudnn_version": cudnn_version,
        "cuda_matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch_module.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch_module.is_deterministic_algorithms_warn_only_enabled()
        ),
        "deterministic_debug_mode": int(torch_module.get_deterministic_debug_mode()),
        "float32_matmul_precision": str(torch_module.get_float32_matmul_precision()),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "NVIDIA_TF32_OVERRIDE": os.environ["NVIDIA_TF32_OVERRIDE"],
    }
    expected = {
        "torch_version": EXPECTED_TORCH_VERSION,
        "torch_cuda_build": EXPECTED_TORCH_CUDA_BUILD,
        "cudnn_version": EXPECTED_CUDNN_VERSION,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "deterministic_debug_mode": 2,
        "float32_matmul_precision": "highest",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    _require(settings == expected, f"torch runtime settings changed: {settings!r}")
    return settings


def require_gpu1_a6000(
    device: str,
    *,
    query_hardware: bool = True,
    lock_identity: Mapping[str, Any] | None = None,
    physical_identity: Mapping[str, Any] | None = None,
    environment_fingerprint_sha256: str | None = None,
    code_bundle_sha256: str | None = None,
    command_runner: Callable[..., Any] | None = None,
    torch_module: Any | None = None,
) -> Mapping[str, Any]:
    """Require pinned physical GPU1 mapped alone to logical ``cuda:0``.

    ``query_hardware=False`` exists solely for CPU contract tests.  Formal
    launchers use the default, which additionally proves the parent lock lease,
    verifies physical identity with nvidia-smi, pins torch/CUDA/cuDNN, and
    enforces deterministic/TF32 settings before model or pixel work.
    """

    environment = require_exact_cuda_environment(formal=query_hardware)
    if str(device) != EXPECTED_SCRIPT_DEVICE:
        raise RuntimeContractError(
            f"formal script device must be {EXPECTED_SCRIPT_DEVICE}, got {device!r}"
        )
    result: dict[str, Any] = {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "physical_gpu": 1,
        "physical_index": 1,
        "visible_device": environment["CUDA_VISIBLE_DEVICES"],
        "cuda_device_order": environment["CUDA_DEVICE_ORDER"],
        "logical_device": EXPECTED_SCRIPT_DEVICE,
        "hardware_queried": bool(query_hardware),
    }
    if not query_hardware:
        return result

    lease_payload = require_active_gpu1_lock(lock_identity)
    physical = (
        inspect_physical_gpu1(require_idle=False, command_runner=command_runner)
        if physical_identity is None
        else dict(physical_identity)
    )
    for key, expected in {
        "physical_index": 1,
        "name": EXPECTED_GPU_NAME,
        "uuid": EXPECTED_GPU_UUID,
        "serial": EXPECTED_GPU_SERIAL,
    }.items():
        _require(physical.get(key) == expected, f"physical GPU identity changed: {key}")
    unfamiliar = [
        row
        for row in physical.get("compute_processes", [])
        if int(row["pid"]) not in {int(lease_payload["owner_pid"]), os.getpid()}
    ]
    _require(
        not unfamiliar,
        f"unfamiliar physical-GPU1 compute PID during logical mapping: {unfamiliar!r}",
    )

    if torch_module is None:
        import torch as torch_module  # imported only after lock + physical checks

    settings = _configure_torch_runtime(torch_module)
    _require(
        torch_module.cuda.is_available() and torch_module.cuda.device_count() == 1,
        "CUDA visibility must expose exactly one available logical device",
    )
    name = str(torch_module.cuda.get_device_name(0))
    properties = torch_module.cuda.get_device_properties(0)
    total_memory = int(properties.total_memory)
    _require(name == EXPECTED_GPU_NAME, f"logical cuda:0 name changed: {name!r}")
    _require(
        total_memory >= MINIMUM_TOTAL_MEMORY_BYTES,
        f"logical cuda:0 memory contract failed: {total_memory} bytes",
    )
    environment_sha = (
        _runtime_fingerprint_from_environment(ENVIRONMENT_FINGERPRINT_ENV)
        if environment_fingerprint_sha256 is None
        else str(environment_fingerprint_sha256)
    )
    bundle_sha = (
        _runtime_fingerprint_from_environment(CODE_BUNDLE_FINGERPRINT_ENV)
        if code_bundle_sha256 is None
        else str(code_bundle_sha256)
    )
    _require(
        re.fullmatch(r"[0-9a-f]{64}", environment_sha) is not None,
        "environment fingerprint SHA256 is invalid",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{64}", bundle_sha) is not None,
        "code-bundle SHA256 is invalid",
    )
    result.update(
        {
            "gpu_name": name,
            "gpu_uuid": physical["uuid"],
            "gpu_serial": physical["serial"],
            "nvidia_smi_total_memory_bytes": int(
                physical["nvidia_smi_total_memory_bytes"]
            ),
            "total_memory_bytes": total_memory,
            "logical_device_count": 1,
            "mapping_assertion": (
                "physical nvidia-smi index 1 and pinned UUID/serial mapped alone "
                "by CUDA_VISIBLE_DEVICES=1,CUDA_DEVICE_ORDER=PCI_BUS_ID to cuda:0"
            ),
            "environment_fingerprint_sha256": environment_sha,
            "code_bundle_sha256": bundle_sha,
            "torch_runtime": settings,
        }
    )
    result["identity_sha256"] = _canonical_sha256(result)
    return result


def child_runtime_environment(
    base: Mapping[str, str],
    *,
    lock_identity: Mapping[str, Any],
    environment_fingerprint_sha256: str,
    code_bundle_sha256: str,
) -> dict[str, str]:
    environment = dict(base)
    environment.update(CHILD_RUNTIME_ENVIRONMENT)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": EXPECTED_VISIBLE_DEVICE,
            "CUDA_DEVICE_ORDER": EXPECTED_CUDA_DEVICE_ORDER,
            LOCK_PATH_ENV: str(lock_identity["path"]),
            LOCK_OWNER_PID_ENV: str(lock_identity["owner_pid"]),
            LOCK_LEASE_ENV: str(lock_identity["lease"]),
            ENVIRONMENT_FINGERPRINT_ENV: environment_fingerprint_sha256,
            CODE_BUNDLE_FINGERPRINT_ENV: code_bundle_sha256,
        }
    )
    return environment


@contextmanager
def exclusive_gpu1_lock(
    path: Path | str = DEFAULT_GLOBAL_LOCK,
) -> Iterator[Mapping[str, Any]]:
    """Hold the one global non-blocking physical-GPU1 experiment lock."""

    lock_path = Path(path).expanduser().resolve()
    required_root = Path("/srv/szha0669/unblur-slam/locks").resolve()
    try:
        lock_path.relative_to(required_root)
    except ValueError as error:
        raise RuntimeContractError(
            f"global lock must remain below {required_root}: {lock_path}"
        ) from error
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeContractError(
                f"physical GPU1 global lock is already held: {lock_path}"
            ) from error
        identity = {
            "schema": LOCK_SCHEMA,
            "path": str(lock_path),
            "owner_pid": os.getpid(),
            "lease": secrets.token_hex(32),
            "exclusive": True,
            "nonblocking": True,
            "scope": "all_physical_gpu1_work_across_the_complete_execution_plan",
        }
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "ascii"
            ),
        )
        os.fsync(descriptor)
        yield identity
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
