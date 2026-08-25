#!/usr/bin/env python3
"""Safely migrate a legacy causal-EVSSM v4 NumPy RNG checkpoint.

Legacy v4 checkpoints stored the MT19937 key array as a NumPy ``uint32``
array.  That object requires pickle globals and therefore prevents
``torch.load(..., weights_only=True)``.  This one-shot migration replaces only
that array with a lossless ``torch.int64`` tensor.  ``torch.uint32`` is not
used because PyTorch 2.3.1 cannot serialize that dtype.

The source bytes are copied into a private temporary stream and hashed before
they are inspected.  The legacy payload is then loaded with
``weights_only=True`` under an exact NumPy-only allowlist; unrestricted pickle
loading is never used.  Outputs are new files, are never overwritten,
and become visible through no-replace hard links only after all content and
safe-loading checks pass.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from typing import Any, Optional, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_causal_video_deblur import (
    CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1,
    CHECKPOINT_MIGRATION_KIND_V1,
    CHECKPOINT_MIGRATION_SCHEMA_V1,
    CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1,
    CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1,
    NUMPY_RNG_ENCODING_V4,
    checkpoint_semantic_digest,
    validate_checkpoint_migration,
)


CHECKPOINT_FORMAT_V4 = "unblur_slam.causal_video_deblur.v4"
RNG_STATE_SCHEMA_V4 = "unblur_slam.causal_video_deblur.rng_state.v4"
RNG_BOUNDARY_V4 = "epoch_end_no_pending_accumulation"
MIGRATION_REPORT_SCHEMA = (
    "unblur_slam.causal_video_deblur.rng_checkpoint_migration.v1"
)
ALLOWED_PATH = "rng_state.numpy_random_state[1]"
MT19937_KEY_COUNT = 624
UINT32_MAX = (1 << 32) - 1
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_LEGACY_UNSAFE_GLOBALS = frozenset(
    {
        "numpy.core.multiarray._reconstruct",
        "numpy.dtype",
        "numpy.ndarray",
    }
)
DEFAULT_TORCH23_PYTHON = Path("/srv/szha0669/unblur-slam/env/bin/python")


class MigrationError(RuntimeError):
    """Raised when a checkpoint cannot be migrated without ambiguity."""


def _sha256_stream(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return _sha256_stream(handle)[0]


def _validate_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise MigrationError("--expected-source-sha256 must be 64 lowercase hex digits")
    return normalized


def _torch_version_tuple() -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", str(torch.__version__))
    if match is None:
        raise MigrationError(f"unrecognized PyTorch version: {torch.__version__}")
    return int(match.group(1)), int(match.group(2))


def _require_safe_source_capability() -> None:
    if _torch_version_tuple() < (2, 6):
        raise MigrationError(
            "migration must run under PyTorch >=2.6 so legacy globals can be "
            "audited before weights-only loading; use --torch23-python only "
            "for the separate compatibility verifier"
        )
    if not hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint"):
        raise MigrationError("PyTorch lacks get_unsafe_globals_in_checkpoint")
    if not hasattr(torch.serialization, "safe_globals"):
        raise MigrationError("PyTorch lacks the safe_globals context manager")


def _legacy_numpy_safe_globals() -> list[object]:
    try:
        reconstruct = np._core.multiarray._reconstruct
        uint32_dtype_class = np.dtypes.UInt32DType
    except AttributeError as error:
        raise MigrationError(
            "migration host requires NumPy >=1.25 dtype classes"
        ) from error
    return [
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        (np.dtype, "numpy.dtype"),
        (np.ndarray, "numpy.ndarray"),
        uint32_dtype_class,
    ]


def _load_hashed_source(
    source: Path, expected_sha256: str
) -> tuple[dict[str, Any], int]:
    """Hash trusted bytes first, then deserialize exactly those same bytes."""

    try:
        source_handle = source.open("rb")
    except OSError as error:
        raise MigrationError(f"cannot open source checkpoint: {source}") from error

    with source_handle, tempfile.TemporaryFile(mode="w+b") as trusted_copy:
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = source_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            trusted_copy.write(chunk)
            size += len(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise MigrationError(
                "source SHA-256 mismatch; refusing any deserialization "
                f"(expected {expected_sha256}, got {actual_sha256})"
            )
        trusted_copy.flush()
        trusted_copy.seek(0)
        try:
            unsafe_globals = set(
                torch.serialization.get_unsafe_globals_in_checkpoint(trusted_copy)
            )
        except Exception as error:
            raise MigrationError("trusted source global scan failed") from error
        if unsafe_globals != EXPECTED_LEGACY_UNSAFE_GLOBALS:
            missing = sorted(EXPECTED_LEGACY_UNSAFE_GLOBALS - unsafe_globals)
            extra = sorted(unsafe_globals - EXPECTED_LEGACY_UNSAFE_GLOBALS)
            raise MigrationError(
                "legacy unsafe-global set is not exact; "
                f"missing={missing}, extra={extra}"
            )
        trusted_copy.seek(0)
        try:
            with torch.serialization.safe_globals(_legacy_numpy_safe_globals()):
                payload = torch.load(
                    trusted_copy, map_location="cpu", weights_only=True
                )
        except Exception as error:
            raise MigrationError(
                "trusted source failed exact-allowlist weights-only load"
            ) from error

    if type(payload) is not dict:
        raise MigrationError("source checkpoint payload must be a plain dict")
    return payload, size


def _validate_numpy_rng_tuple(value: object) -> tuple[Any, ...]:
    if not isinstance(value, tuple) or len(value) != 5:
        raise MigrationError("legacy numpy_random_state must be a five-item tuple")
    algorithm, keys, position, has_gauss, cached_gaussian = value
    if algorithm != "MT19937" or type(algorithm) is not str:
        raise MigrationError("legacy NumPy RNG algorithm must be MT19937")
    if not isinstance(keys, np.ndarray):
        raise MigrationError(
            "source is not a legacy checkpoint: MT19937 keys must be a NumPy ndarray"
        )
    if keys.dtype != np.dtype(np.uint32):
        raise MigrationError("legacy MT19937 keys must have NumPy uint32 dtype")
    if keys.ndim != 1 or keys.shape != (MT19937_KEY_COUNT,):
        raise MigrationError("legacy MT19937 keys must have shape [624]")
    if type(position) is not int or not 0 <= position <= MT19937_KEY_COUNT:
        raise MigrationError("legacy MT19937 position must be an int in [0,624]")
    if type(has_gauss) is not int or has_gauss not in (0, 1):
        raise MigrationError("legacy NumPy has_gauss must be integer 0 or 1")
    if type(cached_gaussian) is not float or not math.isfinite(cached_gaussian):
        raise MigrationError("legacy NumPy cached_gaussian must be a finite float")
    return value


def _build_migrated_payload(
    source_payload: dict[str, Any], source_sha256: str
) -> tuple[dict[str, Any], np.ndarray, torch.Tensor, str]:
    if source_payload.get("format") != CHECKPOINT_FORMAT_V4:
        raise MigrationError(f"source format must be exactly {CHECKPOINT_FORMAT_V4}")
    if "checkpoint_migration" in source_payload:
        raise MigrationError("source already contains checkpoint_migration lineage")
    for key in ("model", "optimizer", "scheduler"):
        if key not in source_payload:
            raise MigrationError(f"v4 checkpoint is missing required field: {key}")
    rng_state = source_payload.get("rng_state")
    if type(rng_state) is not dict:
        raise MigrationError("v4 checkpoint rng_state must be a plain dict")
    if rng_state.get("schema") != RNG_STATE_SCHEMA_V4:
        raise MigrationError("v4 checkpoint RNG schema mismatch")
    if rng_state.get("checkpoint_boundary") != RNG_BOUNDARY_V4:
        raise MigrationError("v4 checkpoint is not at the registered RNG boundary")
    if "numpy_random_state_encoding" in rng_state:
        raise MigrationError("source already contains a NumPy RNG encoding tag")

    numpy_state = _validate_numpy_rng_tuple(rng_state.get("numpy_random_state"))
    source_keys = numpy_state[1]
    safe_keys = torch.from_numpy(source_keys.astype(np.int64, copy=True))
    if safe_keys.dtype != torch.int64 or tuple(safe_keys.shape) != (
        MT19937_KEY_COUNT,
    ):
        raise MigrationError("internal MT19937 conversion failed")
    if int(safe_keys.min().item()) < 0 or int(safe_keys.max().item()) > UINT32_MAX:
        raise MigrationError("converted MT19937 values are outside uint32 range")
    round_trip = safe_keys.numpy().astype(np.uint32, copy=True)
    if not np.array_equal(round_trip, source_keys):
        raise MigrationError("MT19937 uint32 -> int64 conversion was not lossless")

    migrated_numpy_state = (
        numpy_state[0],
        safe_keys,
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    migrated_rng_state = rng_state.copy()
    migrated_rng_state["numpy_random_state"] = migrated_numpy_state
    migrated_rng_state["numpy_random_state_encoding"] = NUMPY_RNG_ENCODING_V4
    migrated_payload = source_payload.copy()
    migrated_payload["rng_state"] = migrated_rng_state
    try:
        source_semantic_digest = checkpoint_semantic_digest(source_payload)
        target_semantic_digest = checkpoint_semantic_digest(migrated_payload)
    except (TypeError, ValueError, RuntimeError) as error:
        raise MigrationError("checkpoint semantic digest failed") from error
    if source_semantic_digest != target_semantic_digest:
        raise MigrationError("source/target semantic checkpoint digests differ")
    migrated_payload["checkpoint_migration"] = {
        "schema": CHECKPOINT_MIGRATION_SCHEMA_V1,
        "kind": CHECKPOINT_MIGRATION_KIND_V1,
        "source_checkpoint_sha256": source_sha256,
        "allowed_changes": list(CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1),
        "semantic_digest": {
            "schema": CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1,
            "algorithm": CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1,
            "sha256": source_semantic_digest,
            "source_and_target_equal": True,
        },
    }
    try:
        validate_checkpoint_migration(migrated_payload)
    except (TypeError, ValueError, RuntimeError) as error:
        raise MigrationError("checkpoint migration lineage self-check failed") from error
    return migrated_payload, source_keys, safe_keys, source_semantic_digest


def _format_path(path: tuple[object, ...]) -> str:
    result = "$"
    for component in path:
        if isinstance(component, int):
            result += f"[{component}]"
        else:
            result += f".{component}"
    return result


def _tensor_exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    if (
        left.dtype != right.dtype
        or left.device != right.device
        or left.layout != right.layout
        or tuple(left.shape) != tuple(right.shape)
        or left.requires_grad != right.requires_grad
    ):
        return False
    if left.layout == torch.strided and tuple(left.stride()) != tuple(right.stride()):
        return False
    if torch.equal(left, right):
        return True
    if left.layout != torch.strided or not (left.is_floating_point() or left.is_complex()):
        return False
    try:
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0, equal_nan=True)
    except AssertionError:
        return False
    if left.is_floating_point():
        zero = left == 0
        if bool(zero.any()) and not torch.equal(
            torch.signbit(left[zero]), torch.signbit(right[zero])
        ):
            return False
    return True


def _deep_equal(left: object, right: object, path: tuple[object, ...] = ()) -> None:
    location = _format_path(path)
    if type(left) is not type(right):
        raise MigrationError(
            f"invariant mismatch at {location}: {type(left).__name__} != "
            f"{type(right).__name__}"
        )
    if isinstance(left, torch.Tensor):
        if not _tensor_exact(left, right):
            raise MigrationError(f"tensor invariant mismatch at {location}")
        return
    if isinstance(left, np.ndarray):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or left.strides != right.strides
            or not np.array_equal(left, right, equal_nan=True)
        ):
            raise MigrationError(f"NumPy invariant mismatch at {location}")
        return
    if isinstance(left, Mapping):
        if list(left.keys()) != list(right.keys()):
            raise MigrationError(f"mapping-key invariant mismatch at {location}")
        for key in left:
            _deep_equal(left[key], right[key], path + (key,))
        return
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise MigrationError(f"sequence-length invariant mismatch at {location}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _deep_equal(left_item, right_item, path + (index,))
        return
    if isinstance(left, float):
        if struct.pack("!d", left) != struct.pack("!d", right):
            raise MigrationError(f"float invariant mismatch at {location}")
        return
    if isinstance(left, (str, bytes, int, bool, type(None))):
        if left != right:
            raise MigrationError(f"value invariant mismatch at {location}")
        return
    try:
        equal = left == right
    except Exception as error:
        raise MigrationError(f"unsupported invariant type at {location}") from error
    if not isinstance(equal, (bool, np.bool_)) or not bool(equal):
        raise MigrationError(f"value invariant mismatch at {location}")


def _assert_migration_relation(source: object, target: object) -> None:
    """Prove that only the three registered serialization changes occurred."""

    if type(source) is not dict or type(target) is not dict:
        raise MigrationError("migration relation requires plain dict payloads")
    if "checkpoint_migration" in source:
        raise MigrationError("legacy source unexpectedly has checkpoint_migration")
    if list(target.keys()) != [*source.keys(), "checkpoint_migration"]:
        raise MigrationError("target must append only checkpoint_migration at top level")
    for key in source:
        if key != "rng_state":
            _deep_equal(source[key], target[key], (key,))

    source_rng = source.get("rng_state")
    target_rng = target.get("rng_state")
    if type(source_rng) is not dict or type(target_rng) is not dict:
        raise MigrationError("rng_state container changed")
    if "numpy_random_state_encoding" in source_rng:
        raise MigrationError("legacy source unexpectedly has an RNG encoding tag")
    expected_target_rng_keys = [
        *source_rng.keys(),
        "numpy_random_state_encoding",
    ]
    if list(target_rng.keys()) != expected_target_rng_keys:
        raise MigrationError(
            "target rng_state must append only numpy_random_state_encoding"
        )
    for key in source_rng:
        if key != "numpy_random_state":
            _deep_equal(source_rng[key], target_rng[key], ("rng_state", key))
    if target_rng.get("numpy_random_state_encoding") != NUMPY_RNG_ENCODING_V4:
        raise MigrationError("target NumPy RNG encoding tag mismatch")

    source_numpy = _validate_numpy_rng_tuple(source_rng.get("numpy_random_state"))
    target_numpy = target_rng.get("numpy_random_state")
    if not isinstance(target_numpy, tuple) or len(target_numpy) != 5:
        raise MigrationError("target numpy_random_state is malformed")
    for index in (0, 2, 3, 4):
        _deep_equal(
            source_numpy[index],
            target_numpy[index],
            ("rng_state", "numpy_random_state", index),
        )
    target_keys = target_numpy[1]
    if (
        not isinstance(target_keys, torch.Tensor)
        or target_keys.dtype != torch.int64
        or target_keys.device.type != "cpu"
        or target_keys.ndim != 1
        or tuple(target_keys.shape) != (MT19937_KEY_COUNT,)
    ):
        raise MigrationError("target MT19937 keys must be a CPU int64 tensor [624]")
    if int(target_keys.min().item()) < 0 or int(target_keys.max().item()) > UINT32_MAX:
        raise MigrationError("target MT19937 tensor is outside uint32 range")
    restored = target_keys.numpy().astype(np.uint32, copy=True)
    if not np.array_equal(source_numpy[1], restored):
        raise MigrationError("target MT19937 tensor is not lossless")
    try:
        migration = validate_checkpoint_migration(target)
        source_digest = checkpoint_semantic_digest(source)
        target_digest = checkpoint_semantic_digest(target)
    except (TypeError, ValueError, RuntimeError) as error:
        raise MigrationError("target checkpoint_migration lineage is invalid") from error
    if migration is None or source_digest != target_digest:
        raise MigrationError("source/target semantic checkpoint digest mismatch")
    semantic = migration["semantic_digest"]
    if semantic["sha256"] != source_digest:
        raise MigrationError("lineage semantic digest does not bind the source")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_path(parent: Path, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=parent)
    os.close(descriptor)
    return Path(raw_path)


def _same_inode(left: Path, right: Path) -> bool:
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except FileNotFoundError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    )


def _unlink_owned_link(path: Path, temporary: Path) -> None:
    if _same_inode(path, temporary):
        path.unlink()


def _safe_load_target(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise MigrationError(
            "migrated target failed torch.load(weights_only=True)"
        ) from error
    if type(payload) is not dict:
        raise MigrationError("weights-only target payload must be a plain dict")
    return payload


def _verify_target_with_torch23(
    path: Path, python_executable: Path
) -> dict[str, str]:
    """Reload the committed target in the formal PyTorch 2.3 environment."""

    python_executable = Path(python_executable).expanduser().resolve()
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise MigrationError(
            f"PyTorch 2.3 verifier is not executable: {python_executable}"
        )
    verifier = r'''
import json
import sys
import numpy as np
import torch

version = str(torch.__version__)
if not version.startswith("2.3."):
    raise RuntimeError(f"expected PyTorch 2.3.x, got {version}")
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if type(payload) is not dict:
    raise RuntimeError("target is not a plain dict")
if payload.get("format") != "unblur_slam.causal_video_deblur.v4":
    raise RuntimeError("target format is not v4")
rng = payload.get("rng_state")
if type(rng) is not dict:
    raise RuntimeError("target rng_state is malformed")
state = rng.get("numpy_random_state")
if not isinstance(state, tuple) or len(state) != 5:
    raise RuntimeError("target NumPy state is malformed")
keys = state[1]
if not isinstance(keys, torch.Tensor) or keys.device.type != "cpu":
    raise RuntimeError("target keys are not a CPU tensor")
if keys.dtype != torch.int64 or tuple(keys.shape) != (624,):
    raise RuntimeError("target keys are not int64[624]")
if int(keys.min().item()) < 0 or int(keys.max().item()) > (1 << 32) - 1:
    raise RuntimeError("target keys exceed uint32 range")
restored = keys.numpy().astype(np.uint32, copy=True)
if restored.dtype != np.uint32 or restored.shape != (624,):
    raise RuntimeError("target keys cannot be restored to uint32[624]")
print(json.dumps({
    "status": "PASS",
    "torch_version": version,
    "numpy_version": str(np.__version__),
}))
'''
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [str(python_executable), "-c", verifier, str(path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise MigrationError(
            "formal PyTorch 2.3 weights-only target reload failed: " + detail
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise MigrationError("PyTorch 2.3 verifier returned invalid JSON") from error
    if result.get("status") != "PASS" or not str(
        result.get("torch_version", "")
    ).startswith("2.3."):
        raise MigrationError("PyTorch 2.3 verifier did not return PASS")
    return {
        "python": str(python_executable),
        "torch_version": str(result["torch_version"]),
        "numpy_version": str(result["numpy_version"]),
    }


def migrate_v4_rng_checkpoint(
    *,
    source: Path,
    target: Path,
    expected_source_sha256: str,
    report: Optional[Path] = None,
    torch23_python: Path = DEFAULT_TORCH23_PYTHON,
) -> dict[str, Any]:
    """Migrate one legacy v4 checkpoint without overwriting any output."""

    _require_safe_source_capability()
    expected_source_sha256 = _validate_sha256(expected_source_sha256)
    source = Path(source).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    report = (
        target.parent / "migration_report.json"
        if report is None
        else Path(report).expanduser().resolve()
    )
    if report.name != "migration_report.json":
        raise MigrationError("the report filename must be migration_report.json")
    if source in (target, report) or target == report:
        raise MigrationError("source, target, and report paths must be distinct")
    if not target.parent.is_dir() or not report.parent.is_dir():
        raise MigrationError("target and report parent directories must already exist")
    if target.exists() or target.is_symlink():
        raise MigrationError(f"target already exists; refusing overwrite: {target}")
    if report.exists() or report.is_symlink():
        raise MigrationError(f"report already exists; refusing overwrite: {report}")

    source_payload, source_size = _load_hashed_source(
        source, expected_source_sha256
    )
    migrated_payload, source_keys, safe_keys, semantic_digest = (
        _build_migrated_payload(
            source_payload, expected_source_sha256
        )
    )
    _assert_migration_relation(source_payload, migrated_payload)

    target_temporary = _temporary_path(target.parent, f".{target.name}.")
    report_temporary: Optional[Path] = None
    target_linked = False
    report_linked = False
    try:
        try:
            torch.save(migrated_payload, target_temporary)
        except Exception as error:
            raise MigrationError("failed to serialize migrated checkpoint") from error
        _fsync_file(target_temporary)
        target_sha256 = sha256_file(target_temporary)
        target_size = target_temporary.stat().st_size

        temporary_reload = _safe_load_target(target_temporary)
        _assert_migration_relation(source_payload, temporary_reload)
        _deep_equal(migrated_payload, temporary_reload)

        try:
            target_unsafe_globals = list(
                torch.serialization.get_unsafe_globals_in_checkpoint(
                    target_temporary
                )
            )
        except Exception as error:
            raise MigrationError("migrated target global scan failed") from error
        if target_unsafe_globals:
            raise MigrationError(
                "migrated target still contains unsafe globals: "
                f"{sorted(target_unsafe_globals)}"
            )

        report_payload: dict[str, Any] = {
            "schema": MIGRATION_REPORT_SCHEMA,
            "status": "PASS",
            "source": {
                "path": str(source),
                "sha256": expected_source_sha256,
                "size_bytes": source_size,
                "format": CHECKPOINT_FORMAT_V4,
            },
            "target": {
                "path": str(target),
                "sha256": target_sha256,
                "size_bytes": target_size,
                "format": CHECKPOINT_FORMAT_V4,
            },
            "conversion": {
                "allowed_changes": list(CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1),
                "rng_keys_path": ALLOWED_PATH,
                "source_type": "numpy.ndarray",
                "source_dtype": "uint32",
                "target_type": "torch.Tensor",
                "target_dtype": "int64",
                "element_count": int(source_keys.size),
                "minimum": int(safe_keys.min().item()),
                "maximum": int(safe_keys.max().item()),
                "allowed_value_range": [0, UINT32_MAX],
                "lossless": True,
                "torch_uint32_not_used": (
                    "PyTorch 2.3.1 torch.save does not recognize torch.uint32"
                ),
                "rng_encoding_added": NUMPY_RNG_ENCODING_V4,
                "checkpoint_lineage_added": CHECKPOINT_MIGRATION_SCHEMA_V1,
            },
            "invariants": {
                "only_allowed_changes": list(
                    CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1
                ),
                "original_checkpoint_key_order_unchanged": True,
                "lineage_appended_without_target_sha_self_reference": True,
                "all_other_fields_deep_content_equal": True,
                "model_unchanged": True,
                "optimizer_unchanged": True,
                "scheduler_unchanged": True,
                "rng_round_trip_to_numpy_uint32_equal": True,
                "target_weights_only_reload": "PASS",
                "weights_only_torch_version": str(torch.__version__),
                "deserialized_source_only_after_expected_sha256_match": True,
                "source_unsafe_globals_exact_allowlist": sorted(
                    EXPECTED_LEGACY_UNSAFE_GLOBALS
                ),
                "source_load_weights_only": True,
                "target_unsafe_globals": [],
                "outputs_overwritten": False,
                "semantic_digest": {
                    "schema": CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1,
                    "algorithm": CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1,
                    "sha256": semantic_digest,
                    "source_and_target_equal": True,
                },
                "section_semantic_sha256": {
                    key: checkpoint_semantic_digest(
                        {key: source_payload[key]}
                    )
                    for key in ("model", "optimizer", "scheduler")
                },
            },
        }
        report_temporary = _temporary_path(report.parent, f".{report.name}.")
        with report_temporary.open("w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with report_temporary.open("r", encoding="utf-8") as handle:
            if json.load(handle) != report_payload:
                raise MigrationError("migration report JSON self-check failed")

        # Hard-link commits are atomic and fail if a destination appeared after
        # preflight; unlike os.replace, they can never overwrite user data.
        try:
            os.link(target_temporary, target)
        except FileExistsError as error:
            raise MigrationError(f"target appeared during migration: {target}") from error
        target_linked = True
        _fsync_directory(target.parent)

        final_reload = _safe_load_target(target)
        _assert_migration_relation(source_payload, final_reload)
        _deep_equal(migrated_payload, final_reload)
        if sha256_file(target) != target_sha256:
            raise MigrationError("committed target SHA-256 changed unexpectedly")
        torch23_verification = _verify_target_with_torch23(
            target, torch23_python
        )
        report_payload["invariants"]["torch23_subprocess_weights_only_reload"] = {
            "status": "PASS",
            **torch23_verification,
        }

        # The report was staged before the committed-path subprocess check.
        # Rewrite only its private temporary file with the final verifier fact.
        with report_temporary.open("w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with report_temporary.open("r", encoding="utf-8") as handle:
            if json.load(handle) != report_payload:
                raise MigrationError("final migration report JSON self-check failed")

        try:
            os.link(report_temporary, report)
        except FileExistsError as error:
            raise MigrationError(f"report appeared during migration: {report}") from error
        report_linked = True
        _fsync_directory(report.parent)
        return report_payload
    except Exception:
        if report_linked and report_temporary is not None:
            _unlink_owned_link(report, report_temporary)
        if target_linked:
            _unlink_owned_link(target, target_temporary)
        raise
    finally:
        if report_temporary is not None:
            report_temporary.unlink(missing_ok=True)
        target_temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate exactly one legacy v4 NumPy RNG checkpoint to a "
            "PyTorch-2.3 weights-only-safe checkpoint without overwrite."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--expected-source-sha256",
        required=True,
        help="Required pre-authorized SHA-256; mismatch prevents deserialization.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path named migration_report.json (default: target directory).",
    )
    parser.add_argument(
        "--torch23-python",
        type=Path,
        default=DEFAULT_TORCH23_PYTHON,
        help=(
            "Trusted PyTorch 2.3.x interpreter used for the mandatory final "
            "weights-only reload."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = migrate_v4_rng_checkpoint(
            source=args.source,
            target=args.target,
            expected_source_sha256=args.expected_source_sha256,
            report=args.report,
            torch23_python=args.torch23_python,
        )
    except MigrationError as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
