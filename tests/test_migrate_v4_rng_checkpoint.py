"""CPU safety tests for the legacy v4 RNG checkpoint migrator.

Run this file with the repository's PyTorch >=2.6 host interpreter.  The test
also invokes the formal Unblur-SLAM PyTorch 2.3.1 environment for both legacy
fixture creation and mandatory target reload verification.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_causal_video_deblur import (
    CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1,
    NUMPY_RNG_ENCODING_V4,
    checkpoint_semantic_digest,
    validate_checkpoint_migration,
)
from scripts.migrate_v4_rng_checkpoint import (
    CHECKPOINT_FORMAT_V4,
    DEFAULT_TORCH23_PYTHON,
    EXPECTED_LEGACY_UNSAFE_GLOBALS,
    MigrationError,
    _assert_migration_relation,
    _deep_equal,
    _load_hashed_source,
    sha256_file,
)


SCRIPT = ROOT / "scripts/migrate_v4_rng_checkpoint.py"


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def _make_legacy_checkpoint(
    path: Path,
    *,
    checkpoint_format: str = CHECKPOINT_FORMAT_V4,
    malicious_marker: Path | None = None,
) -> None:
    code = r'''
import os
import random
import shlex
import sys
import numpy as np
import torch

class DeferredCommand:
    def __init__(self, marker):
        self.marker = marker
    def __reduce__(self):
        return (os.system, ("touch -- " + shlex.quote(self.marker),))

path, checkpoint_format, marker = sys.argv[1:]
generator_a = torch.Generator().manual_seed(101)
generator_b = torch.Generator().manual_seed(202)
payload = {
    "format": checkpoint_format,
    "model": {
        "weight": torch.tensor([[1.0, -0.0], [float("nan"), 4.0]]),
        "counter": torch.tensor(9, dtype=torch.int64),
    },
    "optimizer": {
        "state": {0: {"step": torch.tensor(7), "exp_avg": torch.arange(4.0)}},
        "param_groups": [{"lr": 2.0e-5, "params": [0], "name": "base"}],
    },
    "scheduler": {"last_epoch": 25, "_last_lr": [2.0e-5]},
    "epoch": 25,
    "step": 600,
    "rng_state": {
        "schema": "unblur_slam.causal_video_deblur.rng_state.v4",
        "checkpoint_boundary": "epoch_end_no_pending_accumulation",
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "train_loader_generator_state": generator_a.get_state().clone(),
        "alignment_loader_generator_state": generator_b.get_state().clone(),
    },
    "diagnostic": {"text": "unchanged", "none": None, "flag": True},
}
if marker:
    payload["forbidden_extra_global"] = DeferredCommand(marker)
torch.save(payload, path)
'''
    completed = subprocess.run(
        [
            str(DEFAULT_TORCH23_PYTHON),
            "-c",
            code,
            str(path),
            checkpoint_format,
            "" if malicious_marker is None else str(malicious_marker),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode == 0, completed.stderr


def _run_cli(
    source: Path,
    target: Path,
    expected_sha256: str,
    *,
    include_expected_sha: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        "--target",
        str(target),
        "--torch23-python",
        str(DEFAULT_TORCH23_PYTHON),
    ]
    if include_expected_sha:
        command.extend(["--expected-source-sha256", expected_sha256])
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_environment(),
    )


def _assert_no_outputs(directory: Path, target_name: str = "target.pth") -> None:
    assert not (directory / target_name).exists()
    assert not (directory / "migration_report.json").exists()
    assert not list(directory.glob(".*.tmp"))


def _test_success_and_torch23_compatibility(root: Path) -> tuple[Path, Path, dict]:
    data = root / "success"
    data.mkdir()
    source = data / "legacy.pth"
    target = data / "target.pth"
    _make_legacy_checkpoint(source)
    source_sha256 = sha256_file(source)
    source_bytes = source.read_bytes()

    completed = _run_cli(source, target, source_sha256)
    assert completed.returncode == 0, completed.stderr
    stdout_report = json.loads(completed.stdout)
    report_path = data / "migration_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == stdout_report
    assert source.read_bytes() == source_bytes
    assert sha256_file(source) == source_sha256
    assert report["status"] == "PASS"
    assert report["source"]["sha256"] == source_sha256
    assert report["target"]["sha256"] == sha256_file(target)
    assert report["conversion"]["lossless"] is True
    assert report["conversion"]["allowed_changes"] == (
        CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1
    )
    assert report["invariants"]["source_load_weights_only"] is True
    assert report["invariants"]["source_unsafe_globals_exact_allowlist"] == (
        sorted(EXPECTED_LEGACY_UNSAFE_GLOBALS)
    )
    torch23 = report["invariants"]["torch23_subprocess_weights_only_reload"]
    assert torch23["status"] == "PASS"
    assert torch23["torch_version"].startswith("2.3.")
    assert report["invariants"]["target_unsafe_globals"] == []

    target_payload = torch.load(target, map_location="cpu", weights_only=True)
    source_payload, _ = _load_hashed_source(source, source_sha256)
    _assert_migration_relation(source_payload, target_payload)
    for section in ("model", "optimizer", "scheduler"):
        _deep_equal(source_payload[section], target_payload[section], (section,))
    rng = target_payload["rng_state"]
    keys = rng["numpy_random_state"][1]
    assert keys.dtype == torch.int64
    assert keys.device.type == "cpu"
    assert keys.shape == (624,)
    assert rng["numpy_random_state_encoding"] == NUMPY_RNG_ENCODING_V4
    restored = keys.numpy().astype(np.uint32, copy=True)
    assert np.array_equal(
        restored, source_payload["rng_state"]["numpy_random_state"][1]
    )
    lineage = validate_checkpoint_migration(target_payload)
    assert lineage is not None
    assert lineage["source_checkpoint_sha256"] == source_sha256
    assert lineage["allowed_changes"] == CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1
    assert checkpoint_semantic_digest(source_payload) == checkpoint_semantic_digest(
        target_payload
    )
    assert torch.serialization.get_unsafe_globals_in_checkpoint(target) == []
    return source, target, target_payload


def _test_hash_gate_precedes_deserialization(root: Path) -> None:
    data = root / "hash_gate"
    data.mkdir()
    source = data / "malicious.pth"
    target = data / "target.pth"
    marker = data / "must_not_exist"
    _make_legacy_checkpoint(source, malicious_marker=marker)

    wrong_hash = "0" * 64
    assert wrong_hash != sha256_file(source)
    completed = _run_cli(source, target, wrong_hash)
    assert completed.returncode == 2
    assert "source SHA-256 mismatch" in completed.stderr
    assert not marker.exists()
    _assert_no_outputs(data)

    # Even with an authorized hash, an additional pickle global must be
    # rejected by the scanner before the weights-only unpickler can call it.
    completed = _run_cli(source, target, sha256_file(source))
    assert completed.returncode == 2
    assert "unsafe-global set is not exact" in completed.stderr
    assert "extra=" in completed.stderr
    assert not marker.exists()
    _assert_no_outputs(data)


def _test_strict_source_and_no_overwrite(
    root: Path, source: Path, target: Path
) -> None:
    wrong_format = root / "wrong_format"
    wrong_format.mkdir()
    invalid_source = wrong_format / "legacy.pth"
    _make_legacy_checkpoint(
        invalid_source,
        checkpoint_format="unblur_slam.causal_video_deblur.v3",
    )
    completed = _run_cli(
        invalid_source,
        wrong_format / "target.pth",
        sha256_file(invalid_source),
    )
    assert completed.returncode == 2
    assert "source format must be exactly" in completed.stderr
    _assert_no_outputs(wrong_format)

    already_safe = root / "already_safe"
    already_safe.mkdir()
    completed = _run_cli(target, already_safe / "target.pth", sha256_file(target))
    assert completed.returncode == 2
    assert "unsafe-global set is not exact" in completed.stderr
    _assert_no_outputs(already_safe)

    target_exists = root / "target_exists"
    target_exists.mkdir()
    sentinel_target = target_exists / "target.pth"
    sentinel_target.write_bytes(b"do-not-overwrite-target")
    completed = _run_cli(source, sentinel_target, sha256_file(source))
    assert completed.returncode == 2
    assert "target already exists" in completed.stderr
    assert sentinel_target.read_bytes() == b"do-not-overwrite-target"
    assert not (target_exists / "migration_report.json").exists()

    report_exists = root / "report_exists"
    report_exists.mkdir()
    sentinel_report = report_exists / "migration_report.json"
    sentinel_report.write_text("do-not-overwrite-report", encoding="utf-8")
    completed = _run_cli(
        source, report_exists / "target.pth", sha256_file(source)
    )
    assert completed.returncode == 2
    assert "report already exists" in completed.stderr
    assert sentinel_report.read_text(encoding="utf-8") == "do-not-overwrite-report"
    assert not (report_exists / "target.pth").exists()

    missing_hash = root / "missing_hash"
    missing_hash.mkdir()
    completed = _run_cli(
        source,
        missing_hash / "target.pth",
        sha256_file(source),
        include_expected_sha=False,
    )
    assert completed.returncode == 2
    assert "--expected-source-sha256" in completed.stderr
    _assert_no_outputs(missing_hash)


def _test_deep_invariant_rejects_unregistered_change(
    source: Path, target_payload: dict
) -> None:
    source_payload, _ = _load_hashed_source(source, sha256_file(source))
    tampered = copy.deepcopy(target_payload)
    tampered["optimizer"]["param_groups"][0]["lr"] = 3.0e-5
    try:
        _assert_migration_relation(source_payload, tampered)
    except MigrationError as error:
        assert "invariant mismatch" in str(error)
    else:
        raise AssertionError("unregistered optimizer mutation was accepted")


def test_migrate_v4_rng_checkpoint(tmp_path: Path) -> None:
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    assert version >= (2, 6), "run migration safety test with PyTorch >=2.6"
    assert DEFAULT_TORCH23_PYTHON.exists()
    source, target, target_payload = _test_success_and_torch23_compatibility(
        tmp_path
    )
    _test_hash_gate_precedes_deserialization(tmp_path)
    _test_strict_source_and_no_overwrite(tmp_path, source, target)
    _test_deep_invariant_rejects_unregistered_change(source, target_payload)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="v4_rng_migration_test_") as raw:
        test_migrate_v4_rng_checkpoint(Path(raw))
    print("PASS test_migrate_v4_rng_checkpoint")


if __name__ == "__main__":
    main()
