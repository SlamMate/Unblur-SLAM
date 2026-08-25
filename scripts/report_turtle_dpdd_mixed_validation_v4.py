#!/usr/bin/env python3
"""Fail-closed CPU-only acceptance report for the preregistered TURTLE v4 ablation.

This reporter imports only the Python standard library.  It never initializes a
model or a CUDA runtime.  All metric aggregates are recomputed from the persisted
per-image/per-frame rows, and the final report is created with exclusive-create
semantics so an existing report can never be overwritten.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT_DEFAULT = Path(
    "/srv/szha0669/unblur-slam/turtle_finetune/"
    "replica424_dpdd_png16_mixed_v4_validation_only"
)
OUTPUT_NAME = "validation_only_report.json"
CONTRACT_SCHEMA = "unblur_slam.turtle_replica424_dpdd_png16_mixed_validation_only.v4"
CONTRACT_SHA256 = "93aa22e33bfafe3acbaab7a858e494eea2207d9bef6806cc95bbe5524ceb4a1f"
REPORT_SCHEMA = "unblur_slam.turtle_dpdd_mixed_v4_validation_only_report.v1"
SEEDS = (17, 42, 73)
TRAINED_ARMS = ("V", "S", "M")
DPDD_ARMS = ("G", "V", "S", "M")
METRICS = ("psnr", "ssim", "l1", "lpips")
TEMPORAL_METRICS = ("psnr", "ssim", "l1")
TEMPORAL_SOURCES = (
    "raw",
    "turtle",
    "turtle_reset_cache",
    "turtle_repeat_current",
    "turtle_replayed_ordered",
    "turtle_shuffled_history",
)
CONTROL_LABELS = {
    "normal": "turtle",
    "reset_cache": "turtle_reset_cache",
    "repeat_current": "turtle_repeat_current",
    "ordered_replay": "turtle_replayed_ordered",
    "cyclic_shuffled_strict_past": "turtle_shuffled_history",
}
FLOAT_ATOL = 1e-9
T_CRITICAL_TWO_SIDED_95_DF73 = 1.992997125889855


class AcceptanceError(RuntimeError):
    """Raised whenever an acceptance invariant is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def require_equal(actual: Any, expected: Any, message: str) -> None:
    require(actual == expected, f"{message}: expected {expected!r}, got {actual!r}")


def require_close(actual: float, expected: float, message: str, atol: float = FLOAT_ATOL) -> None:
    require(math.isfinite(actual) and math.isfinite(expected), f"{message}: non-finite value")
    require(
        abs(actual - expected) <= atol,
        f"{message}: expected {expected!r}, got {actual!r}, |delta|={abs(actual - expected)!r}",
    )


def as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def as_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def finite_float(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    return result


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_snapshot(path: Path) -> tuple[Mapping[str, Any], str]:
    require(path.is_file(), f"missing JSON input: {path}")
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"invalid JSON input {path}: {error}") from error
    return as_mapping(value, str(path)), digest


def mean(values: Iterable[float], label: str) -> float:
    rows = [finite_float(value, label) for value in values]
    require(bool(rows), f"cannot average an empty collection: {label}")
    return math.fsum(rows) / len(rows)


def metric_mean(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str], label: str) -> dict[str, float]:
    require(bool(rows), f"empty metric rows: {label}")
    return {
        metric: mean(
            (as_mapping(row.get("metrics"), f"{label}.metrics").get(metric) for row in rows),
            f"{label}.{metric}",
        )
        for metric in metrics
    }


def nested_metric_mean(
    frames: Sequence[Mapping[str, Any]], source: str, metrics: Sequence[str], label: str
) -> dict[str, float]:
    require(bool(frames), f"empty temporal frames: {label}")
    return {
        metric: mean(
            (
                as_mapping(
                    as_mapping(frame.get("metrics"), f"{label}.frame.metrics").get(source),
                    f"{label}.{source}",
                ).get(metric)
                for frame in frames
            ),
            f"{label}.{source}.{metric}",
        )
        for metric in metrics
    }


def require_metric_mapping_close(
    actual: Mapping[str, Any], expected: Mapping[str, float], metrics: Sequence[str], label: str
) -> None:
    for metric in metrics:
        require_close(
            finite_float(actual.get(metric), f"{label}.{metric}"),
            expected[metric],
            f"{label}.{metric}",
        )


def metric_delta(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    return {metric: candidate[metric] - reference[metric] for metric in candidate}


def paired_metric_delta(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    label: str,
) -> dict[str, float]:
    require_equal(len(candidate), len(reference), f"{label} row count")
    deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
    for index, (candidate_row, reference_row) in enumerate(zip(candidate, reference)):
        require_equal(candidate_row.get("name"), reference_row.get("name"), f"{label} name[{index}]")
        candidate_metrics = as_mapping(candidate_row.get("metrics"), f"{label}.candidate[{index}].metrics")
        reference_metrics = as_mapping(reference_row.get("metrics"), f"{label}.reference[{index}].metrics")
        for metric in metrics:
            deltas[metric].append(
                finite_float(candidate_metrics.get(metric), f"{label}.candidate.{metric}")
                - finite_float(reference_metrics.get(metric), f"{label}.reference.{metric}")
            )
    return {metric: mean(values, f"{label}.{metric}") for metric, values in deltas.items()}


def gate(gate_id: str, observed: float, operator: str, threshold: float) -> dict[str, Any]:
    observed = finite_float(observed, f"{gate_id}.observed")
    threshold = finite_float(threshold, f"{gate_id}.threshold")
    operations = {
        ">=": observed >= threshold,
        "<=": observed <= threshold,
        ">": observed > threshold,
        "<": observed < threshold,
    }
    require(operator in operations, f"unsupported gate operator: {operator}")
    return {
        "id": gate_id,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "pass": operations[operator],
    }


def gate_family(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"pass": all(check.get("pass") is True for check in checks), "checks": list(checks)}


def flatten_gate_checks(families: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [check for family in families.values() for check in as_list(family.get("checks"), "gate checks")]


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Create *path* exactly once; never truncate or replace an existing entry."""

    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    created = False
    try:
        fd = os.open(path, flags, 0o644)
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            require(written > 0, f"zero-byte write while creating {path}")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
    except FileExistsError as error:
        raise AcceptanceError(f"refusing to overwrite existing report: {path}") from error
    except Exception:
        if fd is not None:
            os.close(fd)
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def required_result_paths(root: Path) -> list[Path]:
    paths = [
        root / "preregistered_contract.json",
        root / "preflight.json",
        root / "evssm_dpdd_validation/metrics.json",
        root / "replica_temporal/G/metrics.json",
    ]
    for seed in SEEDS:
        paths.append(root / f"seed{seed}/dpdd_validation/metrics.json")
        for arm in TRAINED_ARMS:
            paths.append(root / f"replica_temporal/seed{seed}_{arm}/metrics.json")
    return paths


def require_complete_result_set(root: Path) -> None:
    missing = [str(path) for path in required_result_paths(root) if not path.is_file()]
    require(not missing, "incomplete validation result set; missing: " + ", ".join(missing))


def record_snapshot(
    snapshots: dict[str, dict[str, Any]], root: Path, path: Path, digest: str, kind: str
) -> None:
    try:
        label = str(path.resolve().relative_to(root))
    except ValueError:
        label = str(path.resolve())
    snapshots[label] = {"kind": kind, "path": str(path.resolve()), "sha256": digest}


def check_path_sha(
    snapshots: dict[str, dict[str, Any]], root: Path, path: Path, expected: str, kind: str
) -> str:
    actual = sha256_file(path.resolve())
    require_equal(actual, expected, f"{kind} SHA256 for {path}")
    record_snapshot(snapshots, root, path, actual, kind)
    return actual


def validate_training_metadata(
    metadata_value: Any,
    arm: str,
    seed: int,
    checkpoint_sha256: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = as_mapping(metadata_value, f"seed{seed}.{arm}.checkpoint_metadata")
    require_equal(metadata.get("format"), "unblur_slam.turtle_streaming.checkpoint.v1", "checkpoint format")
    require_equal(metadata.get("kind"), "finetuned", "checkpoint kind") if "kind" in metadata else None
    require_equal(metadata.get("mode"), arm, "checkpoint mode")
    require_equal(metadata.get("checkpoint_sha256"), checkpoint_sha256, "metadata checkpoint hash")
    model = as_mapping(contract.get("model"), "contract.model")
    require_equal(metadata.get("base_checkpoint_sha256"), model.get("official_gopro_checkpoint_sha256"), "base hash")
    require_equal(metadata.get("turtle_repo_commit"), model.get("repo_commit"), "TURTLE commit")
    require_equal(metadata.get("turtle_arch_sha256"), model.get("architecture_sha256"), "architecture hash")
    require_equal(metadata.get("turtle_config_sha256"), model.get("config_sha256"), "config hash")

    training = as_mapping(metadata.get("training"), f"seed{seed}.{arm}.training")
    prereg_training = as_mapping(contract.get("training"), "contract.training")
    expected_steps = prereg_training.get("optimizer_steps_per_trained_arm")
    require_equal(training.get("seed"), seed, "training seed")
    require_equal(training.get("optimizer_steps"), expected_steps, "optimizer steps")
    require_equal(training.get("attempted_optimizer_steps"), expected_steps, "attempted steps")
    require_equal(training.get("executed_optimizer_steps"), expected_steps, "executed steps")
    require_equal(training.get("amp_skipped_optimizer_steps"), 0, "AMP skipped steps")

    manifests = as_mapping(metadata.get("manifests"), f"seed{seed}.{arm}.manifests")
    replica = as_mapping(as_mapping(contract.get("data"), "contract.data").get("replica_train"), "replica_train")
    dpdd = as_mapping(as_mapping(contract.get("data"), "contract.data").get("dpdd"), "dpdd")
    expected_video = arm in {"V", "M"}
    expected_dpdd = arm in {"S", "M"}
    require_equal(manifests.get("video"), replica.get("manifest") if expected_video else None, "video manifest")
    require_equal(
        manifests.get("video_sha256"), replica.get("sha256") if expected_video else None, "video manifest hash"
    )
    require_equal(manifests.get("dpdd_pairs"), dpdd.get("train_manifest") if expected_dpdd else None, "DPDD manifest")
    require_equal(
        manifests.get("dpdd_pairs_sha256"),
        dpdd.get("train_manifest_sha256") if expected_dpdd else None,
        "DPDD manifest hash",
    )
    require_equal(manifests.get("test_pixels_or_metrics_read"), False, "checkpoint test disclosure")
    return {
        "arm": arm,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "base_checkpoint_sha256": metadata.get("base_checkpoint_sha256"),
        "turtle_repo_commit": metadata.get("turtle_repo_commit"),
        "optimizer_steps": training.get("optimizer_steps"),
        "attempted_optimizer_steps": training.get("attempted_optimizer_steps"),
        "executed_optimizer_steps": training.get("executed_optimizer_steps"),
        "amp_skipped_optimizer_steps": training.get("amp_skipped_optimizer_steps"),
        "video_manifest_sha256": manifests.get("video_sha256"),
        "dpdd_train_manifest_sha256": manifests.get("dpdd_pairs_sha256"),
        "test_pixels_or_metrics_read": False,
    }


def validate_official_g_metadata(metadata_value: Any, checkpoint_sha256: str, contract: Mapping[str, Any]) -> None:
    metadata = as_mapping(metadata_value, "official G checkpoint metadata")
    model = as_mapping(contract.get("model"), "contract.model")
    require_equal(metadata.get("format"), "official_turtle.params", "official G format")
    require_equal(metadata.get("kind"), "official_gopro", "official G kind")
    require_equal(metadata.get("checkpoint_sha256"), checkpoint_sha256, "official G metadata hash")
    require_equal(metadata.get("base_checkpoint_sha256"), checkpoint_sha256, "official G base hash")
    require_equal(metadata.get("turtle_repo_commit"), model.get("repo_commit"), "official G commit")
    require_equal(metadata.get("turtle_arch_sha256"), model.get("architecture_sha256"), "official G architecture")
    require_equal(metadata.get("turtle_config_sha256"), model.get("config_sha256"), "official G config")


def validate_checkpoint_files(
    root: Path, contract: Mapping[str, Any], snapshots: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    model = as_mapping(contract.get("model"), "contract.model")
    evaluation = as_mapping(contract.get("evaluation"), "contract.evaluation")
    official_g_path = Path(str(model.get("official_gopro_checkpoint"))).resolve()
    official_g_sha = str(model.get("official_gopro_checkpoint_sha256"))
    check_path_sha(snapshots, root, official_g_path, official_g_sha, "official TURTLE checkpoint")
    evssm_contract = as_mapping(evaluation.get("official_evssm_reference"), "official EVSSM reference")
    evssm_path = Path(str(evssm_contract.get("checkpoint"))).resolve()
    evssm_sha = str(evssm_contract.get("checkpoint_sha256"))
    check_path_sha(snapshots, root, evssm_path, evssm_sha, "official EVSSM checkpoint")

    trained: dict[str, dict[str, Any]] = {}
    seen_hashes: set[str] = set()
    for seed in SEEDS:
        trained[str(seed)] = {}
        for arm in TRAINED_ARMS:
            checkpoint = (root / f"seed{seed}/{arm}.pth").resolve()
            sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
            require(sidecar.is_file(), f"missing checkpoint SHA sidecar: {sidecar}")
            fields = sidecar.read_text(encoding="utf-8").strip().split()
            require_equal(len(fields), 2, f"sidecar field count {sidecar}")
            expected_sha, filename = fields
            require_equal(filename, checkpoint.name, f"sidecar filename {sidecar}")
            actual_sha = check_path_sha(snapshots, root, checkpoint, expected_sha, "trained checkpoint")
            require(actual_sha not in seen_hashes, f"duplicate trained checkpoint content: {actual_sha}")
            seen_hashes.add(actual_sha)
            sidecar_sha = sha256_file(sidecar)
            record_snapshot(snapshots, root, sidecar, sidecar_sha, "checkpoint SHA sidecar")
            trained[str(seed)][arm] = {
                "path": str(checkpoint),
                "sha256": actual_sha,
                "sidecar_path": str(sidecar),
                "sidecar_sha256": sidecar_sha,
            }
    require_equal(len(seen_hashes), len(SEEDS) * len(TRAINED_ARMS), "unique trained checkpoint hashes")
    return {
        "official_G": {"path": str(official_g_path), "sha256": official_g_sha},
        "official_EVSSM": {"path": str(evssm_path), "sha256": evssm_sha},
        "trained": trained,
        "all_nine_trained_checkpoints_content_unique": True,
    }


def validate_pinned_inputs(
    root: Path, contract: Mapping[str, Any], snapshots: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    data = as_mapping(contract.get("data"), "contract.data")
    dpdd = as_mapping(data.get("dpdd"), "contract.data.dpdd")
    replica_train = as_mapping(data.get("replica_train"), "contract.data.replica_train")
    replica_validation = as_mapping(data.get("replica_validation"), "contract.data.replica_validation")
    paths = [
        (Path(str(dpdd.get("dataset_manifest"))), str(dpdd.get("dataset_manifest_sha256")), "DPDD dataset manifest"),
        (Path(str(dpdd.get("train_manifest"))), str(dpdd.get("train_manifest_sha256")), "DPDD train manifest"),
        (
            Path(str(dpdd.get("validation_manifest"))),
            str(dpdd.get("validation_manifest_sha256")),
            "DPDD validation manifest",
        ),
        (
            Path(str(dpdd.get("materialization_audit"))),
            str(dpdd.get("materialization_audit_sha256")),
            "DPDD materialization audit",
        ),
        (Path(str(replica_train.get("manifest"))), str(replica_train.get("sha256")), "Replica train manifest"),
        (
            Path(str(replica_validation.get("manifest"))),
            str(replica_validation.get("sha256")),
            "Replica validation manifest",
        ),
    ]
    for path, expected, kind in paths:
        check_path_sha(snapshots, root, path, expected, kind)

    dataset_manifest, dataset_sha = load_json_snapshot(Path(str(dpdd.get("dataset_manifest"))))
    require_equal(dataset_sha, dpdd.get("dataset_manifest_sha256"), "dataset manifest hash")
    require_equal(dataset_manifest.get("repository"), "JacobLinCool/DPDD", "DPDD mirror repository")
    require_equal(dataset_manifest.get("revision"), dpdd.get("revision"), "DPDD mirror revision")
    distribution = as_mapping(dataset_manifest.get("distribution"), "DPDD distribution")
    require_equal(distribution.get("kind"), "third_party_huggingface_mirror_of_dpdd_png16", "DPDD source kind")
    require_equal(distribution.get("official_dpdd_download"), False, "official DPDD download flag")
    image_contract = as_mapping(dataset_manifest.get("image_contract"), "DPDD image contract")
    require_equal(image_contract.get("files_preserved_byte_exact"), True, "DPDD byte preservation")
    require_equal(image_contract.get("ihdr_bit_depth"), 16, "DPDD PNG bit depth")

    audit, audit_sha = load_json_snapshot(Path(str(dpdd.get("materialization_audit"))))
    require_equal(audit_sha, dpdd.get("materialization_audit_sha256"), "materialization audit hash")
    require_equal(audit.get("status"), "pass", "materialization audit status")
    require_equal(audit.get("pair_count"), 424, "materialization pair count")
    require_equal(audit.get("asset_count"), 848, "materialization asset count")
    require_equal(audit.get("asset_bytes"), 7243127232, "materialization bytes")
    audited_image_contract = as_mapping(audit.get("image_contract"), "audited image contract")
    require_equal(audited_image_contract.get("unique_sizes"), [[1680, 1120]], "DPDD image dimensions")
    require_equal(audited_image_contract.get("all_png_ihdr_16bit_rgb"), True, "DPDD RGB16 contract")
    require_equal(audited_image_contract.get("all_opencv_uint16_hwc3"), True, "DPDD decode contract")
    test_audit = as_mapping(audit.get("test_audit"), "materialization test audit")
    require_equal(test_audit.get("local_test_paths"), 0, "materialization local test paths")
    require_equal(test_audit.get("network_requests_by_auditor"), 0, "materialization audit network requests")

    pins = as_mapping(contract.get("implementation_pins"), "implementation pins")
    pin_paths = {
        "launch_preflight_sha256": repo_root / "scripts/preflight_turtle_dpdd_mixed_v4.py",
        "mixed_trainer_sha256": repo_root / "scripts/train_turtle_mixed_defocus.py",
        "turtle_single_image_evaluator_sha256": repo_root / "scripts/evaluate_turtle_single_image_defocus.py",
        "evssm_validation_evaluator_sha256": repo_root / "scripts/evaluate_evssm_dpdd_validation.py",
        "turtle_streaming_evaluator_sha256": repo_root / "scripts/evaluate_turtle_streaming.py",
        "turtle_backend_sha256": repo_root / "src/turtle_backend.py",
        "evssm_backend_sha256": repo_root / "src/deblur_backends.py",
        "materializer_sha256": repo_root / "scripts/materialize_dpdd_hf_png16.py",
        "materialization_auditor_sha256": repo_root / "scripts/audit_dpdd_hf_png16_materialization.py",
        "mixed_trainer_test_sha256": repo_root / "tests/test_turtle_mixed_defocus_training.py",
        "turtle_single_image_evaluator_test_sha256": repo_root / "tests/test_turtle_single_image_defocus_eval.py",
        "evssm_evaluator_test_sha256": repo_root / "tests/test_evssm_dpdd_validation.py",
        "materializer_test_sha256": repo_root / "tests/test_materialize_dpdd_hf_png16.py",
        "materialization_auditor_test_sha256": repo_root / "tests/test_audit_dpdd_hf_png16_materialization.py",
    }
    for pin_name, path in pin_paths.items():
        check_path_sha(snapshots, root, path, str(pins.get(pin_name)), f"implementation pin {pin_name}")
    return {
        "dpdd_pair_count": audit.get("pair_count"),
        "dpdd_asset_count": audit.get("asset_count"),
        "dpdd_asset_bytes": audit.get("asset_bytes"),
        "dpdd_distribution": {
            "repository": dataset_manifest.get("repository"),
            "revision": dataset_manifest.get("revision"),
            "source_kind": distribution.get("kind"),
            "official_dpdd_download": False,
            "materialized_files_preserved_byte_exact": True,
            "audited_dimensions_width_height": [1680, 1120],
            "local_resize_or_downsample_applied": False,
            "upstream_mirror_downsample_equivalence_to_official_dpdd": "not_established_by_pinned_metadata",
            "paper_comparable": False,
            "license_scope_warning": distribution.get("license_scope_warning"),
        },
        "implementation_pin_count": len(pin_paths),
        "all_pinned_hashes_match": True,
    }


def validate_contract_and_preflight(
    root: Path,
    contract: Mapping[str, Any],
    contract_sha: str,
    preflight: Mapping[str, Any],
    preflight_sha: str,
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    require_equal(contract_sha, CONTRACT_SHA256, "frozen preregistration hash")
    require_equal(contract.get("schema"), CONTRACT_SCHEMA, "contract schema")
    require_equal(
        contract.get("status"),
        "preregistered_before_any_mixed_gpu_training_or_dpdd_model_evaluation",
        "contract status",
    )
    require_equal(contract.get("seeds"), list(SEEDS), "contract seeds")
    require_equal(set(as_mapping(contract.get("arms"), "contract arms")), set(DPDD_ARMS), "contract arms")
    runtime = as_mapping(contract.get("runtime"), "contract runtime")
    require_equal(Path(str(runtime.get("output_root"))).resolve(), root, "contract output root")
    require_equal(runtime.get("overwrite"), False, "contract overwrite policy")

    require_equal(preflight.get("schema"), "unblur_slam.turtle_dpdd_mixed_v4_preflight.v1", "preflight schema")
    require_equal(preflight.get("status"), "pass", "preflight status")
    require_equal(preflight.get("contract_sha256"), contract_sha, "preflight contract hash")
    require_equal(preflight.get("output_root_absent_at_preflight"), True, "fresh output root preflight")
    require_equal(preflight.get("dpdd_test_requests"), 0, "preflight DPDD test requests")
    require_equal(preflight.get("dpdd_test_pixels_opened"), False, "preflight DPDD test pixels")
    require_equal(preflight.get("independent_launch_audit"), "FULL LAUNCH PASS; no P0/P1", "launch audit")
    original_contract = Path(str(preflight.get("contract"))).resolve()
    check_path_sha(snapshots, root, original_contract, contract_sha, "original frozen preregistration")

    disclosure = as_mapping(contract.get("sealed_data_disclosure"), "sealed data disclosure")
    require_equal(disclosure.get("dpdd_test_pixel_bytes_opened"), False, "DPDD test bytes disclosure")
    require_equal(disclosure.get("dpdd_test_images_decoded"), False, "DPDD test decode disclosure")
    require_equal(disclosure.get("dpdd_test_metrics_computed"), False, "DPDD test metric disclosure")
    require_equal(
        disclosure.get("dpdd_test_policy"),
        "unconditionally rejected for this entire validation-only experiment regardless of validation outcome",
        "DPDD test policy",
    )
    require_equal(
        disclosure.get("replica_room2_pixels_or_metrics_opened_in_this_experiment"),
        False,
        "Replica Room2 disclosure",
    )
    return {
        "contract_path": str((root / "preregistered_contract.json").resolve()),
        "contract_sha256": contract_sha,
        "preflight_path": str((root / "preflight.json").resolve()),
        "preflight_sha256": preflight_sha,
        "preflight_status": "pass",
        "independent_launch_audit": preflight.get("independent_launch_audit"),
        "scope": contract.get("scope"),
    }


def _validate_dpdd_rows(
    rows_value: Any,
    expected_names: Sequence[str],
    metrics: Sequence[str],
    label: str,
    require_cache_contract: bool,
) -> list[Mapping[str, Any]]:
    rows = [as_mapping(row, f"{label}[{index}]") for index, row in enumerate(as_list(rows_value, label))]
    require_equal(len(rows), len(expected_names), f"{label} row count")
    for index, (row, expected_name) in enumerate(zip(rows, expected_names)):
        require_equal(row.get("index"), index, f"{label}[{index}].index")
        require_equal(row.get("name"), expected_name, f"{label}[{index}].name")
        require("/validation/source/" in str(row.get("blurry_path", row.get("defocus_path", ""))), f"{label} source is not validation")
        require("/validation/target/" in str(row.get("sharp_path", "")), f"{label} target is not validation")
        metric_values = as_mapping(row.get("metrics"), f"{label}[{index}].metrics")
        for metric in metrics:
            finite_float(metric_values.get(metric), f"{label}[{index}].{metric}")
        if require_cache_contract:
            require_equal(row.get("cache_empty_before_call"), True, f"{label}[{index}] cache empty")
            require_equal(row.get("cache_populated_after_call"), True, f"{label}[{index}] cache populated")
            require(finite_float(row.get("latency_ms"), f"{label}[{index}].latency") >= 0.0, "negative latency")
            require_equal(row.get("source_width"), 1680, f"{label}[{index}] source width")
            require_equal(row.get("source_height"), 1120, f"{label}[{index}] source height")
    return rows


def validate_dpdd_reports(
    root: Path,
    contract: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    snapshots: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, list[Mapping[str, Any]]]], list[dict[str, Any]]]:
    data = as_mapping(contract.get("data"), "contract.data")
    dpdd_contract = as_mapping(data.get("dpdd"), "contract.data.dpdd")
    model = as_mapping(contract.get("model"), "contract.model")
    evaluation = as_mapping(contract.get("evaluation"), "contract.evaluation")
    expected_names = [f"dpdd_validation_{index:06d}" for index in range(int(dpdd_contract.get("validation_pairs")))]
    reports: dict[int, Mapping[str, Any]] = {}
    rows_by_seed: dict[int, dict[str, list[Mapping[str, Any]]]] = {}
    per_seed: dict[str, Any] = {}
    metadata_summaries: list[dict[str, Any]] = []

    for seed in SEEDS:
        path = root / f"seed{seed}/dpdd_validation/metrics.json"
        report, digest = load_json_snapshot(path)
        record_snapshot(snapshots, root, path, digest, "DPDD validation metrics")
        reports[seed] = report
        require_equal(report.get("schema"), "unblur_slam.turtle_single_image_defocus_evaluation.v1", "DPDD schema")
        require_equal(report.get("formal"), True, "DPDD formal flag")
        require_equal(report.get("reference_arm"), "G", "DPDD reference arm")
        protocol = as_mapping(report.get("protocol"), f"seed{seed}.DPDD protocol")
        require_equal(protocol.get("manifest"), dpdd_contract.get("validation_manifest"), "DPDD validation manifest")
        require_equal(protocol.get("manifest_sha256"), dpdd_contract.get("validation_manifest_sha256"), "DPDD validation hash")
        require_equal(protocol.get("checkpoint_training_seed"), seed, "DPDD checkpoint seed")
        require_equal(protocol.get("selected_split"), "validation", "DPDD selected split")
        require_equal(protocol.get("pair_count"), len(expected_names), "DPDD pair count")
        require_equal(protocol.get("ordered_pair_names"), expected_names, "DPDD ordered names")
        require_equal(protocol.get("cache_boundary"), "hard_reset_before_every_image", "DPDD cache boundary")
        require_equal(protocol.get("warmup_independent_reset_calls_per_arm"), 1, "DPDD warmup")
        test_disclosure = as_mapping(
            as_mapping(protocol.get("dpdd_dataset"), "DPDD dataset protocol").get("test_disclosure"),
            "DPDD report test disclosure",
        )
        require_equal(test_disclosure.get("pixels_opened"), False, "DPDD report test pixels")
        require_equal(test_disclosure.get("images_decoded"), False, "DPDD report test images")
        require_equal(test_disclosure.get("metrics_opened"), False, "DPDD report test metrics")

        raw = as_mapping(report.get("raw_defocus_baseline"), f"seed{seed}.raw")
        raw_rows = _validate_dpdd_rows(raw.get("images"), expected_names, METRICS, f"seed{seed}.raw", False)
        raw_mean = metric_mean(raw_rows, METRICS, f"seed{seed}.raw")
        require_metric_mapping_close(
            as_mapping(as_mapping(raw.get("summary"), "raw summary").get("mean"), "raw summary mean"),
            raw_mean,
            METRICS,
            f"seed{seed}.raw.summary",
        )

        arms = as_mapping(report.get("arms"), f"seed{seed}.arms")
        require_equal(set(arms), set(DPDD_ARMS), f"seed{seed} DPDD arms")
        arm_rows: dict[str, list[Mapping[str, Any]]] = {"raw": raw_rows}
        seed_result: dict[str, Any] = {"raw": {"mean": raw_mean}}
        for arm in DPDD_ARMS:
            arm_value = as_mapping(arms.get(arm), f"seed{seed}.{arm}")
            if arm == "G":
                expected_path = str(Path(str(model.get("official_gopro_checkpoint"))).resolve())
                expected_sha = str(model.get("official_gopro_checkpoint_sha256"))
                validate_official_g_metadata(arm_value.get("checkpoint_metadata"), expected_sha, contract)
            else:
                checkpoint_info = as_mapping(
                    as_mapping(as_mapping(checkpoints.get("trained"), "trained checkpoints").get(str(seed)), "seed checkpoints").get(arm),
                    f"seed{seed}.{arm}.checkpoint",
                )
                expected_path = str(checkpoint_info.get("path"))
                expected_sha = str(checkpoint_info.get("sha256"))
                metadata_summaries.append(
                    validate_training_metadata(arm_value.get("checkpoint_metadata"), arm, seed, expected_sha, contract)
                )
            require_equal(str(Path(str(arm_value.get("checkpoint"))).resolve()), expected_path, f"seed{seed}.{arm} path")
            require_equal(arm_value.get("checkpoint_sha256"), expected_sha, f"seed{seed}.{arm} hash")
            rows = _validate_dpdd_rows(
                arm_value.get("images"), expected_names, METRICS, f"seed{seed}.{arm}", True
            )
            arm_rows[arm] = rows
            calculated = metric_mean(rows, METRICS, f"seed{seed}.{arm}")
            summary = as_mapping(arm_value.get("summary"), f"seed{seed}.{arm}.summary")
            require_equal(summary.get("image_count"), len(expected_names), f"seed{seed}.{arm} summary count")
            require_metric_mapping_close(
                as_mapping(summary.get("mean"), f"seed{seed}.{arm}.summary.mean"),
                calculated,
                METRICS,
                f"seed{seed}.{arm}.summary",
            )
            latency = as_mapping(summary.get("latency_ms"), f"seed{seed}.{arm}.latency")
            latency_mean = mean((row.get("latency_ms") for row in rows), f"seed{seed}.{arm}.latency")
            require_close(finite_float(latency.get("mean"), "latency mean"), latency_mean, "DPDD latency mean")
            seed_result[arm] = {"mean": calculated, "latency_ms": dict(latency)}

        comparisons = {
            "M_minus_V": paired_metric_delta(arm_rows["M"], arm_rows["V"], METRICS, f"seed{seed}.M-V"),
            "M_minus_S": paired_metric_delta(arm_rows["M"], arm_rows["S"], METRICS, f"seed{seed}.M-S"),
            "M_minus_G": paired_metric_delta(arm_rows["M"], arm_rows["G"], METRICS, f"seed{seed}.M-G"),
            "M_minus_raw": paired_metric_delta(arm_rows["M"], arm_rows["raw"], METRICS, f"seed{seed}.M-raw"),
        }
        m_minus_v_psnr_per_image = [
            finite_float(as_mapping(m_row.get("metrics"), "M metrics").get("psnr"), "M PSNR")
            - finite_float(as_mapping(v_row.get("metrics"), "V metrics").get("psnr"), "V PSNR")
            for m_row, v_row in zip(arm_rows["M"], arm_rows["V"])
        ]
        paired_count = len(m_minus_v_psnr_per_image)
        paired_mean = mean(m_minus_v_psnr_per_image, f"seed{seed}.M-minus-V PSNR")
        paired_sample_sd = statistics.stdev(m_minus_v_psnr_per_image)
        paired_margin = T_CRITICAL_TWO_SIDED_95_DF73 * paired_sample_sd / math.sqrt(paired_count)
        seed_result["paired_comparisons"] = comparisons
        seed_result["posthoc_descriptive_non_gating"] = {
            "M_minus_V_psnr": {
                "positive_image_count": sum(delta > 0.0 for delta in m_minus_v_psnr_per_image),
                "negative_image_count": sum(delta < 0.0 for delta in m_minus_v_psnr_per_image),
                "zero_image_count": sum(delta == 0.0 for delta in m_minus_v_psnr_per_image),
                "positive_image_fraction": sum(delta > 0.0 for delta in m_minus_v_psnr_per_image) / paired_count,
                "pair_count": paired_count,
                "mean_db": paired_mean,
                "median_db": statistics.median(m_minus_v_psnr_per_image),
                "minimum_db": min(m_minus_v_psnr_per_image),
                "maximum_db": max(m_minus_v_psnr_per_image),
                "sample_standard_deviation_db": paired_sample_sd,
                "paired_student_t_two_sided_95ci_db": [paired_mean - paired_margin, paired_mean + paired_margin],
                "t_critical": T_CRITICAL_TWO_SIDED_95_DF73,
                "degrees_of_freedom": paired_count - 1,
                "preregistered": False,
                "gate_effect": "none",
            }
        }
        per_seed[str(seed)] = seed_result
        rows_by_seed[seed] = arm_rows

    g_repeat: dict[str, float] = {}
    for metric in METRICS:
        maximum = 0.0
        for index in range(len(expected_names)):
            values = [
                finite_float(rows_by_seed[seed]["G"][index]["metrics"][metric], f"G repeat {metric}")
                for seed in SEEDS
            ]
            maximum = max(maximum, max(values) - min(values))
        g_repeat[metric] = maximum
        require(maximum <= 1e-7, f"official G cross-seed repeat sanity failed for {metric}: {maximum}")

    raw_repeat: dict[str, float] = {}
    for metric in METRICS:
        maximum = 0.0
        for index in range(len(expected_names)):
            values = [
                finite_float(rows_by_seed[seed]["raw"][index]["metrics"][metric], f"raw repeat {metric}")
                for seed in SEEDS
            ]
            maximum = max(maximum, max(values) - min(values))
        raw_repeat[metric] = maximum
        require(maximum <= 1e-12, f"raw DPDD cross-seed repeat failed for {metric}: {maximum}")

    across_seed: dict[str, Any] = {}
    for arm in ("raw",) + DPDD_ARMS:
        metric_values = {
            metric: [per_seed[str(seed)][arm]["mean"][metric] for seed in SEEDS] for metric in METRICS
        }
        across_seed[arm] = {
            "mean_of_seed_means": {metric: mean(values, f"{arm}.{metric}") for metric, values in metric_values.items()},
            "population_stddev_of_seed_means": {
                metric: statistics.pstdev(values) for metric, values in metric_values.items()
            },
        }

    return (
        {
            "dataset": "DPDD combined validation from pinned third-party mirror",
            "pair_count": len(expected_names),
            "interpretation": "real-camera paired single-image defocus restoration only; K/V reset per image",
            "latency_interpretation": (
                "descriptive only; no preregistered latency gate. G/V/S/M use the same TURTLE inference graph, "
                "so small process-timing differences must not be called a training-induced speed gain; EVSSM is "
                "a different architecture/evaluator and these values are not online-SLAM FPS."
            ),
            "per_seed": per_seed,
            "across_seed": across_seed,
            "official_G_cross_seed_per_image_max_abs": g_repeat,
            "official_G_repeat_threshold": 1e-7,
            "raw_cross_seed_per_image_max_abs": raw_repeat,
        },
        rows_by_seed,
        metadata_summaries,
    )


def validate_evssm_report(
    root: Path,
    contract: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    dpdd_rows: Mapping[int, Mapping[str, list[Mapping[str, Any]]]],
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = root / "evssm_dpdd_validation/metrics.json"
    report, digest = load_json_snapshot(path)
    record_snapshot(snapshots, root, path, digest, "EVSSM DPDD validation metrics")
    require_equal(report.get("schema"), "unblur_slam.official_evssm_dpdd_validation.v1", "EVSSM schema")
    require_equal(report.get("formal"), True, "EVSSM formal flag")
    dpdd_contract = as_mapping(as_mapping(contract.get("data"), "contract.data").get("dpdd"), "DPDD contract")
    protocol = as_mapping(report.get("protocol"), "EVSSM protocol")
    require_equal(protocol.get("split"), "validation", "EVSSM split")
    require_equal(protocol.get("expected_pair_count"), 74, "EVSSM expected count")
    require_equal(protocol.get("manifest"), dpdd_contract.get("validation_manifest"), "EVSSM manifest")
    require_equal(protocol.get("manifest_sha256"), dpdd_contract.get("validation_manifest_sha256"), "EVSSM manifest hash")
    require_equal(protocol.get("test_pixels_opened"), False, "EVSSM test pixels")
    require_equal(protocol.get("test_metrics_computed"), False, "EVSSM test metrics")
    disclosure = as_mapping(
        as_mapping(protocol.get("dataset_materialization"), "EVSSM dataset materialization").get("test_disclosure"),
        "EVSSM test disclosure",
    )
    require_equal(disclosure.get("pixels_opened"), False, "EVSSM materializer test pixels")
    require_equal(disclosure.get("images_decoded"), False, "EVSSM materializer test images")
    require_equal(disclosure.get("metrics_opened"), False, "EVSSM materializer test metrics")

    checkpoint = as_mapping(report.get("checkpoint"), "EVSSM checkpoint")
    expected_checkpoint = as_mapping(checkpoints.get("official_EVSSM"), "official EVSSM checkpoint")
    require_equal(str(Path(str(checkpoint.get("path"))).resolve()), expected_checkpoint.get("path"), "EVSSM path")
    require_equal(checkpoint.get("sha256"), expected_checkpoint.get("sha256"), "EVSSM hash")

    results = as_mapping(report.get("results"), "EVSSM results")
    require_equal(results.get("pair_count"), 74, "EVSSM result count")
    pairs = [as_mapping(row, f"EVSSM pair[{index}]") for index, row in enumerate(as_list(results.get("pairs"), "EVSSM pairs"))]
    require_equal(len(pairs), 74, "EVSSM pair rows")
    reference_rows = dpdd_rows[SEEDS[0]]["raw"]
    raw_agreement = {metric: 0.0 for metric in METRICS}
    for index, (row, reference) in enumerate(zip(pairs, reference_rows)):
        require_equal(row.get("index"), index, f"EVSSM pair[{index}].index")
        require_equal(row.get("name"), reference.get("name"), f"EVSSM pair[{index}].name")
        require("/validation/source/" in str(row.get("defocus_path")), "EVSSM source is not validation")
        require("/validation/target/" in str(row.get("sharp_path")), "EVSSM target is not validation")
        for source in ("raw", "evssm"):
            source_metrics = as_mapping(row.get(source), f"EVSSM pair[{index}].{source}")
            for metric in METRICS:
                finite_float(source_metrics.get(metric), f"EVSSM pair[{index}].{source}.{metric}")
        require_equal(row.get("width"), 1680, f"EVSSM pair[{index}] width")
        require_equal(row.get("height"), 1120, f"EVSSM pair[{index}] height")
        for metric in METRICS:
            delta = abs(
                finite_float(as_mapping(row.get("raw"), "EVSSM raw").get(metric), f"EVSSM raw {metric}")
                - finite_float(as_mapping(reference.get("metrics"), "TURTLE raw").get(metric), f"TURTLE raw {metric}")
            )
            raw_agreement[metric] = max(raw_agreement[metric], delta)
    for metric, maximum in raw_agreement.items():
        require(maximum <= 1e-6, f"EVSSM/TURTLE raw baseline mismatch for {metric}: {maximum}")

    calculated = {
        source: {
            metric: mean(
                (as_mapping(row.get(source), f"EVSSM {source}").get(metric) for row in pairs),
                f"EVSSM.{source}.{metric}",
            )
            for metric in METRICS
        }
        for source in ("raw", "evssm")
    }
    stored_mean = as_mapping(results.get("mean"), "EVSSM stored mean")
    for source in calculated:
        require_metric_mapping_close(
            as_mapping(stored_mean.get(source), f"EVSSM stored {source}"),
            calculated[source],
            METRICS,
            f"EVSSM.{source}.mean",
        )
    calculated_delta = metric_delta(calculated["evssm"], calculated["raw"])
    require_metric_mapping_close(
        as_mapping(results.get("evssm_minus_raw"), "EVSSM minus raw"),
        calculated_delta,
        METRICS,
        "EVSSM-minus-raw",
    )
    latency = as_mapping(results.get("latency_ms"), "EVSSM latency")
    calculated_latency = mean((row.get("evssm_latency_ms") for row in pairs), "EVSSM latency")
    require_close(finite_float(latency.get("mean"), "EVSSM latency mean"), calculated_latency, "EVSSM latency mean")
    return {
        "interpretation": report.get("interpretation"),
        "pair_count": len(pairs),
        "mean": calculated,
        "evssm_minus_raw": calculated_delta,
        "latency_ms": dict(latency),
        "checkpoint": dict(expected_checkpoint),
        "raw_baseline_max_abs_difference_vs_turtle_evaluator": raw_agreement,
        "raw_baseline_agreement_threshold": 1e-6,
    }


def temporal_frame_identity(frame: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        frame.get("sequence"),
        frame.get("frame_index"),
        frame.get("global_index"),
        frame.get("raw_path"),
        frame.get("gt_path"),
    )


def aggregate_temporal_frames(frames: Sequence[Mapping[str, Any]], steady_index_min: int = 3) -> dict[str, Any]:
    steady = [frame for frame in frames if int(frame.get("frame_index", -1)) >= steady_index_min]
    require(bool(steady), "no steady temporal frames")
    all_mean = {
        source: nested_metric_mean(frames, source, TEMPORAL_METRICS, f"all.{source}")
        for source in TEMPORAL_SOURCES
    }
    steady_mean = {
        source: nested_metric_mean(steady, source, TEMPORAL_METRICS, f"steady.{source}")
        for source in TEMPORAL_SOURCES
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for frame in steady:
        grouped[str(frame.get("sequence"))].append(frame)
    per_sequence = {
        sequence: {
            "steady_frame_count": len(rows),
            "mean": {
                source: nested_metric_mean(rows, source, TEMPORAL_METRICS, f"{sequence}.{source}")
                for source in TEMPORAL_SOURCES
            },
        }
        for sequence, rows in sorted(grouped.items())
    }
    return {
        "all_frame_count": len(frames),
        "steady_frame_count": len(steady),
        "all_frame_mean": all_mean,
        "steady_pooled_mean": steady_mean,
        "per_sequence": per_sequence,
    }


def validate_temporal_report(
    path: Path,
    root: Path,
    contract: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    arm: str,
    seed: int | None,
    snapshots: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[Mapping[str, Any]], dict[str, Any] | None]:
    report, digest = load_json_snapshot(path)
    record_snapshot(snapshots, root, path, digest, "Replica temporal validation metrics")
    require_equal(report.get("schema"), "unblur_slam.turtle_streaming_evaluation.v1", "temporal schema")
    require_equal(report.get("frame_count"), 16, "temporal frame count")
    require_equal(report.get("sequence_count"), 2, "temporal sequence count")
    require_equal(report.get("sources"), list(TEMPORAL_SOURCES) + ["gt"], "temporal sources")
    replica = as_mapping(as_mapping(contract.get("data"), "contract.data").get("replica_validation"), "replica validation")
    provenance = as_mapping(report.get("provenance"), "temporal provenance")
    require_equal(provenance.get("manifest"), replica.get("manifest"), "temporal manifest")
    require_equal(provenance.get("manifest_sha256"), replica.get("sha256"), "temporal manifest hash")
    require_equal(provenance.get("device"), "cuda:0", "temporal logical device")

    if arm == "G":
        expected_checkpoint = as_mapping(checkpoints.get("official_G"), "official G checkpoint")
        validate_official_g_metadata(report.get("checkpoint_metadata"), str(expected_checkpoint.get("sha256")), contract)
        metadata_summary = None
    else:
        require(seed is not None, "trained temporal arm requires a seed")
        expected_checkpoint = as_mapping(
            as_mapping(as_mapping(checkpoints.get("trained"), "trained checkpoints").get(str(seed)), "seed checkpoint").get(arm),
            f"seed{seed}.{arm} checkpoint",
        )
        metadata_summary = validate_training_metadata(
            report.get("checkpoint_metadata"), arm, seed, str(expected_checkpoint.get("sha256")), contract
        )
    require_equal(str(Path(str(provenance.get("checkpoint"))).resolve()), expected_checkpoint.get("path"), "temporal checkpoint path")
    require_equal(provenance.get("checkpoint_sha256"), expected_checkpoint.get("sha256"), "temporal checkpoint hash")

    frames = [as_mapping(row, f"{path}.frames[{index}]") for index, row in enumerate(as_list(report.get("frames"), "temporal frames"))]
    require_equal(len(frames), 16, "temporal frame row count")
    identities = [temporal_frame_identity(frame) for frame in frames]
    require_equal(len(set(identities)), len(identities), "unique temporal frame identities")
    for index, frame in enumerate(frames):
        require_equal(frame.get("global_index"), index, f"temporal global index {index}")
        sequence = str(frame.get("sequence"))
        raw_path = str(frame.get("raw_path"))
        gt_path = str(frame.get("gt_path"))
        lowered = " ".join((sequence, raw_path, gt_path)).lower()
        require("room2" not in lowered and "room_2" not in lowered, f"Room2 provenance appeared: {lowered}")
        require("room1" in sequence and "/room_1/" in raw_path and "/room_1/" in gt_path, "non-Room1 temporal input")
        frame_metrics = as_mapping(frame.get("metrics"), f"temporal frame[{index}].metrics")
        require_equal(set(frame_metrics), set(TEMPORAL_SOURCES), f"temporal metric sources frame {index}")
        for source in TEMPORAL_SOURCES:
            values = as_mapping(frame_metrics.get(source), f"temporal frame[{index}].{source}")
            for metric in TEMPORAL_METRICS:
                finite_float(values.get(metric), f"temporal frame[{index}].{source}.{metric}")

    aggregation = aggregate_temporal_frames(frames)
    require_equal(aggregation["steady_frame_count"], replica.get("steady_frames_from_index_3"), "steady count")
    require_equal(set(aggregation["per_sequence"]), set(frame["sequence"] for frame in frames), "sequence set")
    require_equal(len(aggregation["per_sequence"]), replica.get("sequences"), "sequence count")
    for sequence, sequence_value in aggregation["per_sequence"].items():
        require_equal(sequence_value["steady_frame_count"], 5, f"{sequence} steady count")

    stored_all = as_mapping(report.get("mean"), "stored temporal all-frame mean")
    for source in TEMPORAL_SOURCES:
        require_metric_mapping_close(
            as_mapping(stored_all.get(source), f"stored all {source}"),
            aggregation["all_frame_mean"][source],
            TEMPORAL_METRICS,
            f"stored all {source}",
        )
    history = as_mapping(report.get("history_ablation"), "history ablation")
    require_equal(history.get("steady_frame_count"), aggregation["steady_frame_count"], "stored steady count")
    stored_steady = as_mapping(history.get("steady_mean"), "stored steady mean")
    for source in TEMPORAL_SOURCES[1:]:
        require_metric_mapping_close(
            as_mapping(stored_steady.get(source), f"stored steady {source}"),
            aggregation["steady_pooled_mean"][source],
            TEMPORAL_METRICS,
            f"stored steady {source}",
        )

    gaps = {
        control: metric_delta(
            aggregation["steady_pooled_mean"]["turtle"],
            aggregation["steady_pooled_mean"][source],
        )
        for control, source in CONTROL_LABELS.items()
        if control not in {"normal", "ordered_replay"}
    }
    stored_gaps = as_mapping(history.get("steady_normal_minus_control"), "stored history gaps")
    for control in ("reset_cache", "repeat_current", "cyclic_shuffled_strict_past"):
        source = CONTROL_LABELS[control]
        require_metric_mapping_close(
            as_mapping(stored_gaps.get(source), f"stored gap {source}"),
            gaps[control],
            TEMPORAL_METRICS,
            f"stored gap {source}",
        )

    replay_metric_max = 0.0
    for frame in frames:
        metrics = as_mapping(frame.get("metrics"), "frame metrics")
        normal = as_mapping(metrics.get("turtle"), "normal metrics")
        replay = as_mapping(metrics.get("turtle_replayed_ordered"), "replay metrics")
        for metric in TEMPORAL_METRICS:
            replay_metric_max = max(
                replay_metric_max,
                abs(finite_float(normal.get(metric), "normal") - finite_float(replay.get(metric), "replay")),
            )
    ordered_replay_max_abs = finite_float(history.get("ordered_replay_max_abs"), "ordered replay max abs")
    require_equal(history.get("ordered_replay_matches_stream"), True, "ordered replay match flag")
    require(replay_metric_max <= 1e-12, f"ordered replay metric mismatch: {replay_metric_max}")
    aggregation["steady_normal_minus_control"] = gaps
    aggregation["ordered_replay_max_abs_over_pixels"] = ordered_replay_max_abs
    aggregation["ordered_replay_metric_max_abs"] = replay_metric_max
    aggregation["performance"] = dict(as_mapping(report.get("performance"), "temporal performance"))
    aggregation["checkpoint"] = dict(expected_checkpoint)
    aggregation["provenance"] = {
        "manifest": provenance.get("manifest"),
        "manifest_sha256": provenance.get("manifest_sha256"),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "device": provenance.get("device"),
    }
    return aggregation, frames, metadata_summary


def validate_temporal_reports(
    root: Path,
    contract: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    snapshots: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    g, g_frames, _ = validate_temporal_report(
        root / "replica_temporal/G/metrics.json", root, contract, checkpoints, "G", None, snapshots
    )
    per_seed: dict[int, dict[str, dict[str, Any]]] = {}
    metadata_summaries: list[dict[str, Any]] = []
    reference_identities = [temporal_frame_identity(frame) for frame in g_frames]
    reference_raw = [as_mapping(as_mapping(frame.get("metrics"), "G frame metrics").get("raw"), "G raw") for frame in g_frames]
    report_seed: dict[str, Any] = {}
    for seed in SEEDS:
        per_seed[seed] = {}
        report_seed[str(seed)] = {}
        for arm in TRAINED_ARMS:
            result, frames, metadata = validate_temporal_report(
                root / f"replica_temporal/seed{seed}_{arm}/metrics.json",
                root,
                contract,
                checkpoints,
                arm,
                seed,
                snapshots,
            )
            require_equal([temporal_frame_identity(frame) for frame in frames], reference_identities, f"seed{seed}.{arm} frame alignment")
            for index, frame in enumerate(frames):
                raw = as_mapping(as_mapping(frame.get("metrics"), "frame metrics").get("raw"), "raw metrics")
                for metric in TEMPORAL_METRICS:
                    require_close(
                        finite_float(raw.get(metric), f"seed{seed}.{arm}.raw.{metric}"),
                        finite_float(reference_raw[index].get(metric), f"G.raw.{metric}"),
                        f"seed{seed}.{arm} raw alignment {metric}",
                        atol=1e-12,
                    )
            per_seed[seed][arm] = result
            report_seed[str(seed)][arm] = result
            require(metadata is not None, "trained temporal metadata summary missing")
            metadata_summaries.append(metadata)
    return {"G": g, "per_seed": report_seed}, per_seed, metadata_summaries


def build_preregistered_gates(
    contract: Mapping[str, Any],
    dpdd_report: Mapping[str, Any],
    temporal_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    prereg = as_mapping(contract.get("preregistered_validation_gates"), "preregistered validation gates")
    spatial = as_mapping(prereg.get("spatial_mixed_value"), "spatial gates")
    temporal = as_mapping(prereg.get("temporal_preservation"), "temporal preservation")
    history = as_mapping(prereg.get("useful_history"), "useful history")
    replay = as_mapping(prereg.get("replay_contract"), "replay gate")
    per_seed_dpdd = as_mapping(dpdd_report.get("per_seed"), "DPDD per-seed report")

    spatial_checks: list[dict[str, Any]] = []
    preservation_checks: list[dict[str, Any]] = []
    history_checks: list[dict[str, Any]] = []
    sequence_checks: list[dict[str, Any]] = []
    replay_checks: list[dict[str, Any]] = []
    for seed in SEEDS:
        comparisons = as_mapping(as_mapping(per_seed_dpdd.get(str(seed)), "DPDD seed").get("paired_comparisons"), "paired comparisons")
        m_minus_v = as_mapping(comparisons.get("M_minus_V"), "M-minus-V")
        m_minus_s = as_mapping(comparisons.get("M_minus_S"), "M-minus-S")
        spatial_checks.extend(
            [
                gate(
                    f"spatial.seed{seed}.M_minus_V.psnr",
                    m_minus_v["psnr"],
                    ">=",
                    spatial["per_seed_M_minus_V_dpdd_psnr_db_min"],
                ),
                gate(
                    f"spatial.seed{seed}.M_minus_V.ssim",
                    m_minus_v["ssim"],
                    ">=",
                    spatial["per_seed_M_minus_V_dpdd_ssim_min"],
                ),
                gate(
                    f"spatial.seed{seed}.M_minus_V.lpips",
                    m_minus_v["lpips"],
                    "<=",
                    spatial["per_seed_M_minus_V_dpdd_lpips_max"],
                ),
                gate(
                    f"spatial.seed{seed}.M_minus_V.l1",
                    m_minus_v["l1"],
                    "<=",
                    spatial["per_seed_M_minus_V_dpdd_l1_max"],
                ),
                gate(
                    f"spatial.seed{seed}.M_minus_S.psnr",
                    m_minus_s["psnr"],
                    ">=",
                    spatial["per_seed_M_minus_S_dpdd_psnr_db_min"],
                ),
            ]
        )

        arm_results = temporal_by_seed[seed]
        m = arm_results["M"]
        v = arm_results["V"]
        s = arm_results["S"]
        m_normal = m["steady_pooled_mean"]["turtle"]["psnr"]
        v_normal = v["steady_pooled_mean"]["turtle"]["psnr"]
        preservation_checks.append(
            gate(
                f"temporal_preservation.seed{seed}.M_minus_V.normal_psnr",
                m_normal - v_normal,
                ">=",
                temporal["per_seed_M_minus_V_normal_psnr_db_min"],
            )
        )
        m_reset_gap = m["steady_normal_minus_control"]["reset_cache"]["psnr"]
        m_repeat_gap = m["steady_normal_minus_control"]["repeat_current"]["psnr"]
        m_shuffle_gap = m["steady_normal_minus_control"]["cyclic_shuffled_strict_past"]["psnr"]
        s_reset_gap = s["steady_normal_minus_control"]["reset_cache"]["psnr"]
        interaction = m_reset_gap - s_reset_gap
        history_checks.extend(
            [
                gate(
                    f"useful_history.seed{seed}.M.normal_minus_reset",
                    m_reset_gap,
                    ">=",
                    history["per_seed_M_normal_minus_reset_psnr_db_min"],
                ),
                gate(
                    f"useful_history.seed{seed}.M.normal_minus_repeat_current",
                    m_repeat_gap,
                    ">=",
                    history["per_seed_M_normal_minus_repeat_current_psnr_db_min"],
                ),
                gate(
                    f"useful_history.seed{seed}.M.normal_minus_shuffled",
                    m_shuffle_gap,
                    ">=",
                    history["per_seed_M_normal_minus_shuffled_psnr_db_min"],
                ),
                gate(
                    f"useful_history.seed{seed}.interaction_M_vs_S",
                    interaction,
                    ">=",
                    history["per_seed_interaction_M_vs_S_psnr_db_min"],
                ),
            ]
        )

        require_equal(set(m["per_sequence"]), set(s["per_sequence"]), f"seed{seed} M/S sequence set")
        for sequence in sorted(m["per_sequence"]):
            m_means = m["per_sequence"][sequence]["mean"]
            s_means = s["per_sequence"][sequence]["mean"]
            m_seq_reset = m_means["turtle"]["psnr"] - m_means["turtle_reset_cache"]["psnr"]
            m_seq_repeat = m_means["turtle"]["psnr"] - m_means["turtle_repeat_current"]["psnr"]
            m_seq_shuffle = m_means["turtle"]["psnr"] - m_means["turtle_shuffled_history"]["psnr"]
            s_seq_reset = s_means["turtle"]["psnr"] - s_means["turtle_reset_cache"]["psnr"]
            seq_interaction = m_seq_reset - s_seq_reset
            short_sequence = sequence.replace("replica_blur_", "")
            sequence_checks.extend(
                [
                    gate(f"per_sequence.seed{seed}.{short_sequence}.M.normal_minus_reset", m_seq_reset, ">", 0.0),
                    gate(
                        f"per_sequence.seed{seed}.{short_sequence}.M.normal_minus_repeat_current",
                        m_seq_repeat,
                        ">",
                        0.0,
                    ),
                    gate(
                        f"per_sequence.seed{seed}.{short_sequence}.M.normal_minus_shuffled",
                        m_seq_shuffle,
                        ">",
                        0.0,
                    ),
                    gate(
                        f"per_sequence.seed{seed}.{short_sequence}.interaction_M_vs_S",
                        seq_interaction,
                        ">",
                        0.0,
                    ),
                ]
            )

    replay_threshold = replay["ordered_replay_max_abs_over_every_pixel_frame_arm_and_seed_max"]
    temporal_report = temporal_by_seed
    for seed in SEEDS:
        for arm in TRAINED_ARMS:
            replay_checks.append(
                gate(
                    f"replay.seed{seed}.{arm}.ordered_replay_max_abs",
                    temporal_report[seed][arm]["ordered_replay_max_abs_over_pixels"],
                    "<=",
                    replay_threshold,
                )
            )
    families = {
        "spatial_mixed_value": gate_family(spatial_checks),
        "temporal_preservation": gate_family(preservation_checks),
        "useful_history_pooled": gate_family(history_checks),
        "useful_history_per_sequence_same_direction": gate_family(sequence_checks),
        "replay_trained_arms": gate_family(replay_checks),
    }
    return families


def add_g_replay_gate(
    contract: Mapping[str, Any], families: dict[str, Any], temporal_report: Mapping[str, Any]
) -> None:
    threshold = as_mapping(
        as_mapping(contract.get("preregistered_validation_gates"), "preregistered gates").get("replay_contract"),
        "replay contract",
    )["ordered_replay_max_abs_over_every_pixel_frame_arm_and_seed_max"]
    checks = list(as_list(as_mapping(families.get("replay_trained_arms"), "replay family").get("checks"), "replay checks"))
    checks.insert(
        0,
        gate(
            "replay.official_G.ordered_replay_max_abs",
            temporal_report["G"]["ordered_replay_max_abs_over_pixels"],
            "<=",
            threshold,
        ),
    )
    families["replay_contract"] = gate_family(checks)
    del families["replay_trained_arms"]


def deduplicate_metadata_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for summary in summaries:
        key = (int(summary["seed"]), str(summary["arm"]))
        if key in by_key:
            require_equal(summary, by_key[key], f"checkpoint metadata agreement seed{key[0]}.{key[1]}")
        else:
            by_key[key] = summary
    require_equal(len(by_key), len(SEEDS) * len(TRAINED_ARMS), "checkpoint metadata coverage")
    return {
        str(seed): {arm: dict(by_key[(seed, arm)]) for arm in TRAINED_ARMS} for seed in SEEDS
    }


def build_dpdd_comparison_table(
    dpdd_report: Mapping[str, Any], evssm_report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build a compact, explicitly non-paper-comparable real-validation table."""

    across_seed = as_mapping(dpdd_report.get("across_seed"), "DPDD across-seed")
    per_seed = as_mapping(dpdd_report.get("per_seed"), "DPDD per-seed")
    rows: list[dict[str, Any]] = [
        {
            "method": "raw_defocus_input",
            "mean": dict(as_mapping(as_mapping(across_seed.get("raw"), "raw aggregate").get("mean_of_seed_means"), "raw mean")),
            "latency_ms": 0.0,
            "aggregation": "same 74 images; exact repeat across the three TURTLE evaluation processes",
        },
        {
            "method": "official_EVSSM_reference",
            "mean": dict(as_mapping(as_mapping(evssm_report.get("mean"), "EVSSM mean").get("evssm"), "EVSSM mean values")),
            "latency_ms": as_mapping(evssm_report.get("latency_ms"), "EVSSM latency").get("mean"),
            "aggregation": "one formal evaluation over the same 74 validation images",
        },
    ]
    for arm in DPDD_ARMS:
        latency_values = [
            finite_float(
                as_mapping(as_mapping(per_seed.get(str(seed)), "DPDD seed").get(arm), "DPDD arm")["latency_ms"]["mean"],
                f"DPDD {arm} latency",
            )
            for seed in SEEDS
        ]
        rows.append(
            {
                "method": f"TURTLE_{arm}",
                "mean": dict(
                    as_mapping(
                        as_mapping(across_seed.get(arm), f"{arm} aggregate").get("mean_of_seed_means"),
                        f"{arm} mean",
                    )
                ),
                "latency_ms": mean(latency_values, f"DPDD {arm} latency across processes"),
                "aggregation": (
                    "quality is an exact repeated official-G sanity mean; latency averages three fresh process runs"
                    if arm == "G"
                    else "unweighted mean of the three seed-specific 74-image arithmetic means"
                ),
            }
        )
    for row in rows:
        row["split"] = "DPDD mirror validation"
        row["paper_comparable"] = False
    return rows


def build_report(root: Path) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir(), f"validation root is missing: {root}")
    require_complete_result_set(root)
    snapshots: dict[str, dict[str, Any]] = {}

    contract_path = root / "preregistered_contract.json"
    preflight_path = root / "preflight.json"
    contract, contract_sha = load_json_snapshot(contract_path)
    preflight, preflight_sha = load_json_snapshot(preflight_path)
    record_snapshot(snapshots, root, contract_path, contract_sha, "preregistered contract")
    record_snapshot(snapshots, root, preflight_path, preflight_sha, "launch preflight")
    contract_summary = validate_contract_and_preflight(
        root, contract, contract_sha, preflight, preflight_sha, snapshots
    )
    pinned_summary = validate_pinned_inputs(root, contract, snapshots)
    checkpoints = validate_checkpoint_files(root, contract, snapshots)
    dpdd_report, dpdd_rows, dpdd_metadata = validate_dpdd_reports(
        root, contract, checkpoints, snapshots
    )
    evssm_report = validate_evssm_report(root, contract, checkpoints, dpdd_rows, snapshots)
    dpdd_report["comparison_table"] = build_dpdd_comparison_table(dpdd_report, evssm_report)
    temporal_report, temporal_by_seed, temporal_metadata = validate_temporal_reports(
        root, contract, checkpoints, snapshots
    )
    families = build_preregistered_gates(contract, dpdd_report, temporal_by_seed)
    add_g_replay_gate(contract, families, temporal_report)
    all_checks = flatten_gate_checks(families)
    failed = [str(check["id"]) for check in all_checks if check.get("pass") is not True]
    overall_pass = not failed and all(as_mapping(family, "gate family").get("pass") is True for family in families.values())
    metadata = deduplicate_metadata_summaries(dpdd_metadata + temporal_metadata)

    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    record_snapshot(snapshots, root, script_path, script_sha, "post-validation CPU-only report script")
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "reporter": {
            "path": str(script_path),
            "sha256": script_sha,
            "runtime_contract": "Python standard library only; no torch/cv2/CUDA import; persisted metrics only",
            "output_policy": "exclusive create; existing validation_only_report.json is never overwritten",
            "posthoc_disclosure": "reporter created after validation; it does not alter preregistered gates",
        },
        "contract_and_preflight": contract_summary,
        "integrity": {
            "status": "pass",
            "result_set_complete": True,
            "all_input_hashes_and_provenance_match": True,
            "pinned_inputs": pinned_summary,
            "checkpoints": checkpoints,
            "evaluator_copied_checkpoint_metadata": metadata,
            "checkpoint_metadata_evidence_scope": (
                "This standard-library reporter hashes each .pth byte stream and checks the metadata copied into "
                "both evaluator JSONs; it does not deserialize .pth files and therefore does not itself claim an "
                "in-checkpoint metadata audit."
            ),
            "input_snapshot_count": len(snapshots),
            "input_snapshots": dict(sorted(snapshots.items())),
        },
        "scope_and_sealing": {
            "validation_only": True,
            "paper_benchmark": False,
            "paper_comparability_reason": (
                "DPDD data came from a pinned third-party Hugging Face mirror, not the official DPDD download; "
                "local materialization preserved mirror PNG16 bytes at 1680x1120 without resize, while upstream "
                "mirror downsample/equivalence to official paper assets is not established."
            ),
            "dpdd_split_evaluated": "validation",
            "replica_split_evaluated": "val_temporal Room1 only",
            "dpdd_test_metadata_pristine": False,
            "dpdd_test_pixel_bytes_opened": False,
            "dpdd_test_images_decoded": False,
            "dpdd_test_metrics_computed": False,
            "replica_room2_manifest_bytes_known": True,
            "replica_room2_pixels_or_metrics_opened_in_this_experiment": False,
            "evidence_basis": "frozen preregistration + persisted preflight + checkpoint disclosures + validation metric provenance",
            "test_or_room2_unlock_allowed": False,
            "test_remains_sealed_regardless_of_gate_result": True,
        },
        "dpdd_validation": dpdd_report,
        "official_evssm_dpdd_validation_reference": evssm_report,
        "replica_temporal_validation": temporal_report,
        "preregistered_validation_gates": {
            "all_gates_conjunctive": True,
            "families": families,
            "total_check_count": len(all_checks),
            "failed_check_count": len(failed),
            "failed_gate_ids": failed,
            "pass": overall_pass,
        },
        "terminal_conclusion": {
            "preregistered_hypothesis_supported": overall_pass,
            "result": "pass" if overall_pass else "fail",
            "failed_gate_ids": failed,
            "test_or_room2_unlocked": False,
            "allowed_claim": (
                "All preregistered validation gates passed. This remains exploratory validation-only evidence."
                if overall_pass
                else "The preregistered mixed-training hypothesis is not supported because one or more conjunctive validation gates failed."
            ),
            "forbidden_claims": as_list(
                as_mapping(contract.get("interpretation"), "contract interpretation").get("claims_forbidden"),
                "forbidden claims",
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else root / OUTPUT_NAME)
    try:
        require_equal(output, root / OUTPUT_NAME, "report output path")
        require(not os.path.lexists(output), f"refusing to overwrite existing report: {output}")
        report = build_report(root)
        write_json_exclusive(output, report)
    except (AcceptanceError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "created",
                "output": str(output),
                "sha256": sha256_file(output),
                "preregistered_gate_pass": report["preregistered_validation_gates"]["pass"],
                "failed_gate_ids": report["preregistered_validation_gates"]["failed_gate_ids"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
