#!/usr/bin/env python3
"""CPU-only fail-closed preflight for the BSD 3ms24ms TURTLE study.

Template mode verifies frozen rules and official checkpoint identities, then
reports the exact unbound fields.  Bound mode additionally verifies canonical
BSD train/validation metadata and filesystem presence without decoding images.
Neither mode accepts, resolves, or opens a BSD/DPDD test path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import (  # noqa: E402
    BSD_AUDIT_SCHEMA,
    BSD_DATASET_SCHEMA,
    IMPLEMENTATION_PIN_PATHS,
    assert_train_validation_disjoint,
    inspect_bsd_sequence_manifest,
    load_contract,
    load_json_object,
    normalized_sha256,
    require,
    sha256_file,
    summarize_inventory,
    validate_protocol,
)
from scripts.bsd_dpdd_runtime import (  # noqa: E402
    validate_static_runtime_contract,
    verify_frozen_environment,
)


def _check_sha(path: Path | str, expected: Any, *, label: str) -> str:
    candidate = Path(path).expanduser().resolve()
    require(candidate.is_file(), f"missing {label}: {candidate}")
    digest = sha256_file(candidate)
    require(digest == normalized_sha256(expected, label=f"{label} SHA256"), f"{label} SHA256 mismatch")
    return digest


def _check_models(contract: Mapping[str, Any], *, strict_load: bool) -> Mapping[str, Any]:
    models = contract["models"]
    shared = models["turtle_shared_repo"]
    repo = Path(shared["repo"]).expanduser().resolve()
    require(repo.is_dir(), "TURTLE repository is missing")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == shared["repo_commit"], "TURTLE repository commit changed")
    tracked = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(not tracked, "TURTLE tracked tree is dirty")
    _check_sha(shared["upstream_inference"], shared["upstream_inference_sha256"], label="upstream inference recipe")
    for label in ("turtle_G", "turtle_O"):
        arm = models[label]
        _check_sha(arm["architecture"], arm["architecture_sha256"], label=f"{label} architecture")
        _check_sha(arm["config"], arm["config_sha256"], label=f"{label} config")
        checkpoint = Path(arm["checkpoint"]).expanduser().resolve()
        _check_sha(checkpoint, arm["checkpoint_sha256"], label=f"{label} checkpoint")
        require(checkpoint.stat().st_size == arm.get("checkpoint_bytes", checkpoint.stat().st_size), f"{label} byte size changed")
    official_o = Path(models["turtle_O"]["checkpoint"]).expanduser().resolve()
    require(stat.S_IMODE(official_o.stat().st_mode) == 0o444, "official O checkpoint is not mode 0444")
    evssm = models["evssm_E"]
    _check_sha(evssm["checkpoint"], evssm["checkpoint_sha256"], label="official EVSSM")
    _check_sha(evssm["architecture"], evssm["architecture_sha256"], label="EVSSM architecture")
    _check_sha(evssm["backend_source"], evssm["backend_source_sha256"], label="EVSSM backend")
    _check_sha(evssm["builder_source"], evssm["builder_source_sha256"], label="EVSSM builder")

    strict_summary: dict[str, Any] = {"requested": bool(strict_load)}
    if strict_load:
        # These imports construct/load CPU models only.  No forward pass and no
        # CUDA API is used.  The t0 loader uses direct-file import to avoid the
        # upstream package's optional LMDB dependency.
        from src.turtle_backend import validate_turtle_artifacts
        from src.turtle_official_bsd_backend import validate_official_bsd_artifacts

        g = models["turtle_G"]
        validate_turtle_artifacts(
            {
                "turtle_repo": shared["repo"],
                "turtle_repo_commit": shared["repo_commit"],
                "turtle_config": g["config"],
                "turtle_checkpoint": g["checkpoint"],
                "turtle_checkpoint_sha256": g["checkpoint_sha256"],
            },
            load_weights=True,
        )
        o = models["turtle_O"]
        artifacts = validate_official_bsd_artifacts(
            repo=shared["repo"],
            config=o["config"],
            checkpoint=o["checkpoint"],
            checkpoint_sha256=o["checkpoint_sha256"],
            load_weights=True,
        )
        strict_summary.update(
            {
                "G_t1": "strict_load_pass",
                "O_t0": "strict_load_pass",
                "O_kind": artifacts.checkpoint_metadata["kind"],
                "O_state_tensors": o["state_tensors"],
                "O_parameters": o["parameters"],
            }
        )
        import torch
        from thirdparty.EVSSM.models.EVSSM import EVSSM

        evssm_model = EVSSM()
        try:
            payload = torch.load(evssm["checkpoint"], map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(evssm["checkpoint"], map_location="cpu")
        state = payload.get("params", payload) if isinstance(payload, Mapping) else payload
        require(isinstance(state, Mapping), "EVSSM checkpoint has no state mapping")
        incompatible = evssm_model.load_state_dict(dict(state), strict=True)
        require(
            not incompatible.missing_keys and not incompatible.unexpected_keys,
            "EVSSM strict state load failed",
        )
        strict_summary.update(
            {
                "E_EVSSM": "strict_load_pass",
                "E_state_tensors": len(evssm_model.state_dict()),
                "E_parameters": sum(
                    parameter.numel() for parameter in evssm_model.parameters()
                ),
            }
        )
        del evssm_model
    return {"repo_commit": head, "strict_load": strict_summary}


def _check_dpdd(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    dpdd = contract["data"]["dpdd"]
    _check_sha(dpdd["dataset_manifest"], dpdd["dataset_manifest_sha256"], label="DPDD dataset manifest")
    _check_sha(dpdd["train_manifest"], dpdd["train_manifest_sha256"], label="DPDD train manifest")
    _check_sha(dpdd["validation_manifest"], dpdd["validation_manifest_sha256"], label="DPDD validation manifest")
    dataset = load_json_object(dpdd["dataset_manifest"])
    require(dataset.get("repository") == dpdd["repository"], "DPDD repository identity changed")
    require(dataset.get("revision") == dpdd["revision"], "DPDD revision changed")
    require(dataset.get("splits") == {"train": 350, "validation": 74}, "DPDD split counts changed")
    disclosure = dataset.get("test_disclosure", {})
    for key in ("pixels_opened", "images_decoded", "metrics_opened", "split_supported_by_this_materializer"):
        require(disclosure.get(key) is False, f"DPDD sealed-test invariant failed: {key}")
    return {
        "train_pairs": 350,
        "validation_pairs": 74,
        "test_pixels_opened": False,
        "pixels_decoded_by_this_preflight": False,
    }


def _check_bsd_bound(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    bsd = contract["data"]["bsd"]
    root = Path(bsd["dataset_root"]).expanduser().resolve()
    dataset_path = Path(bsd["dataset_manifest"]).expanduser().resolve()
    audit_path = Path(bsd["materialization_audit"]).expanduser().resolve()
    _check_sha(dataset_path, bsd["dataset_manifest_sha256"], label="BSD dataset manifest")
    _check_sha(audit_path, bsd["materialization_audit_sha256"], label="BSD materialization audit")
    dataset = load_json_object(dataset_path)
    require(dataset.get("schema") == BSD_DATASET_SCHEMA, "BSD dataset manifest schema changed")
    require(dataset.get("dataset") == "BSD", "BSD dataset identity changed")
    require(dataset.get("exposure_setting") == "3ms24ms", "BSD exposure setting changed")
    require(dataset.get("blur_origin") == "real_camera_long_exposure", "BSD blur origin changed")
    require(dataset.get("synthetic_high_fps_average") is False, "BSD cannot be labeled synthetic")
    require(dataset.get("splits") == {"train": {"sequences": 60, "frames": 6000}, "validation": {"sequences": 20, "frames": 2000}}, "BSD train/validation counts changed")
    disclosure = dataset.get("test_disclosure", {})
    for key in ("pixel_paths_materialized", "pixels_opened", "images_decoded", "model_outputs_computed", "metrics_computed"):
        require(disclosure.get(key) is False, f"BSD sealed-test invariant failed: {key}")
    canonical = dataset.get("canonical_manifests")
    require(isinstance(canonical, Mapping) and set(canonical) == {"train", "validation"}, "BSD canonical manifest index changed")
    for split in ("train", "validation"):
        entry = canonical[split]
        require(isinstance(entry, Mapping), f"BSD {split} canonical entry missing")
        require(Path(entry["path"]).expanduser().resolve() == Path(bsd[f"{split}_manifest"]).expanduser().resolve(), f"BSD {split} canonical path changed")
        require(entry.get("sha256") == bsd[f"{split}_manifest_sha256"], f"BSD {split} canonical hash changed")

    audit = load_json_object(audit_path)
    require(audit.get("schema") == BSD_AUDIT_SCHEMA and audit.get("status") == "pass", "BSD materialization audit did not pass")
    require(audit.get("train_validation_only") is True, "BSD audit scope is not train/validation only")
    image = audit.get("image_contract", {})
    require(image.get("resolution_width_height") == [640, 480], "BSD image resolution changed")
    require(image.get("rgb_channels") == 3 and image.get("bit_depth") == 8, "BSD RGB8 contract changed")
    require(image.get("all_train_validation_assets_hashed") is True, "BSD asset hashing audit failed")
    require(image.get("all_train_validation_assets_decoded") is True, "BSD train/validation decode audit failed")
    disjoint = audit.get("disjoint_audit", {})
    require(disjoint.get("capture_ids") is True, "BSD capture split overlap")
    require(disjoint.get("paths") is True, "BSD path split overlap")
    require(disjoint.get("content_hashes") is True, "BSD content split overlap")
    test_audit = audit.get("test_audit", {})
    for key in ("local_pixel_paths", "pixels_opened", "images_decoded", "model_outputs_computed", "metrics_computed"):
        expected = 0 if key == "local_pixel_paths" else False
        require(test_audit.get(key) == expected, f"BSD test audit invariant failed: {key}")

    train = inspect_bsd_sequence_manifest(
        bsd["train_manifest"],
        dataset_root=root,
        expected_sha256=bsd["train_manifest_sha256"],
        expected_split="train",
        expected_sequences=60,
        expected_frames=6000,
        expected_per_exposure_sequences=60,
        require_assets=True,
    )
    validation = inspect_bsd_sequence_manifest(
        bsd["validation_manifest"],
        dataset_root=root,
        expected_sha256=bsd["validation_manifest_sha256"],
        expected_split="validation",
        expected_sequences=20,
        expected_frames=2000,
        expected_per_exposure_sequences=20,
        require_assets=True,
    )
    assert_train_validation_disjoint(train, validation)
    return {
        "train": summarize_inventory(train),
        "validation": summarize_inventory(validation),
        "test_pixels_opened": False,
        "pixels_decoded_by_this_preflight": False,
    }


def _check_bsd_reference_bound(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a validation-prefix binding without resolving BSD train."""

    bsd = contract["data"]["bsd"]
    root = Path(bsd["dataset_root"]).expanduser().resolve()
    dataset_path = Path(bsd["dataset_manifest"]).expanduser().resolve()
    audit_path = Path(bsd["materialization_audit"]).expanduser().resolve()
    _check_sha(dataset_path, bsd["dataset_manifest_sha256"], label="BSD reference dataset manifest")
    _check_sha(audit_path, bsd["materialization_audit_sha256"], label="BSD reference materialization audit")
    dataset = load_json_object(dataset_path)
    require(dataset.get("schema") == BSD_DATASET_SCHEMA, "BSD reference dataset schema changed")
    require(dataset.get("dataset") == "BSD", "BSD reference dataset identity changed")
    require(dataset.get("exposure_setting") == "3ms24ms", "BSD reference exposure changed")
    require(dataset.get("blur_origin") == "real_camera_long_exposure", "BSD reference blur origin changed")
    require(dataset.get("synthetic_high_fps_average") is False, "BSD reference cannot be synthetic")
    require(
        dataset.get("splits")
        == {"validation": {"sequences": 20, "frames": 2000}},
        "BSD validation-prefix split/count contract changed",
    )
    require(dataset.get("train_materialized") is False, "reference prefix must not claim BSD train")
    canonical = dataset.get("canonical_manifests")
    require(
        isinstance(canonical, Mapping) and set(canonical) == {"validation"},
        "BSD reference canonical index must contain validation only",
    )
    entry = canonical["validation"]
    require(
        Path(entry["path"]).is_absolute() is False,
        "BSD reference canonical validation path must be relative",
    )
    resolved_manifest = (dataset_path.parent / entry["path"]).resolve()
    require(
        resolved_manifest == Path(bsd["validation_manifest"]).expanduser().resolve(),
        "BSD reference validation path binding changed",
    )
    require(
        entry.get("sha256") == bsd["validation_manifest_sha256"],
        "BSD reference validation hash binding changed",
    )
    disclosure = dataset.get("test_disclosure", {})
    for key in (
        "pixel_paths_materialized",
        "pixels_opened",
        "images_decoded",
        "model_outputs_computed",
        "metrics_computed",
    ):
        require(disclosure.get(key) is False, f"BSD reference sealed-test invariant failed: {key}")

    audit = load_json_object(audit_path)
    require(
        audit.get("schema") == BSD_AUDIT_SCHEMA and audit.get("status") == "pass",
        "BSD reference materialization audit did not pass",
    )
    require(audit.get("validation_only") is True, "BSD reference audit scope changed")
    image = audit.get("image_contract", {})
    require(image.get("resolution_width_height") == [640, 480], "BSD reference resolution changed")
    require(image.get("rgb_channels") == 3 and image.get("bit_depth") == 8, "BSD reference RGB8 changed")
    require(
        image.get("all_validation_assets_sha256_recomputed") is True,
        "BSD reference SHA audit failed",
    )
    require(image.get("all_validation_assets_decoded") is True, "BSD reference decode audit failed")
    test_audit = audit.get("test_audit", {})
    for key in ("local_pixel_paths", "pixels_opened", "images_decoded", "model_outputs_computed", "metrics_computed"):
        expected = 0 if key == "local_pixel_paths" else False
        require(test_audit.get(key) == expected, f"BSD reference test invariant failed: {key}")

    validation = inspect_bsd_sequence_manifest(
        bsd["validation_manifest"],
        dataset_root=root,
        expected_sha256=bsd["validation_manifest_sha256"],
        expected_split="validation",
        expected_sequences=20,
        expected_frames=2000,
        expected_per_exposure_sequences=20,
        require_assets=True,
        verify_content=True,
    )
    return {
        "mode": "validation_only_reference_prefix",
        "train_manifest_resolved_or_read": False,
        "validation": summarize_inventory(validation),
        "test_pixels_opened": False,
        "pixels_decoded_by_this_preflight": False,
    }


def _check_implementation_pins(contract: Mapping[str, Any]) -> Mapping[str, str]:
    pins = contract["implementation_pins"]
    result: dict[str, str] = {}
    for name, relative_path in IMPLEMENTATION_PIN_PATHS.items():
        path = ROOT / relative_path
        result[name] = _check_sha(path, pins[name], label=name)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--template", action="store_true", help="validate the intentionally unbound preregistration")
    parser.add_argument("--reference-only", action="store_true", help="validate BSD validation binding while keeping train blocked")
    parser.add_argument("--skip-strict-model-load", action="store_true", help="unit-test convenience; formal review must not use this")
    parser.add_argument("--output", type=Path, help="optional immutable JSON receipt path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path, contract, contract_sha = load_contract(
        args.contract,
        expected_sha256=args.expected_contract_sha256,
    )
    require(not (args.template and args.reference_only), "--template and --reference-only are exclusive")
    unbound = validate_protocol(
        contract,
        allow_template=args.template,
        reference_only=args.reference_only,
    )
    # Freeze and re-hash the complete launch implementation and Python package
    # environment before importing torch or opening any model/data artifact.
    implementation = _check_implementation_pins(contract)
    validate_static_runtime_contract(contract["runtime"])
    environment_fingerprint = verify_frozen_environment(
        contract["environment_fingerprint"]
    )
    model_summary = _check_models(contract, strict_load=not args.skip_strict_model_load)
    dpdd_summary = (
        {
            "status": "not_inspected_in_bsd_reference_only_preflight",
            "test_pixels_opened": False,
        }
        if args.reference_only
        else _check_dpdd(contract)
    )
    bsd_summary = None
    output_absent = None
    if not args.template:
        bsd_summary = (
            _check_bsd_reference_bound(contract)
            if args.reference_only
            else _check_bsd_bound(contract)
        )
        output = Path(contract["sealed_outputs"]["output_root"]).expanduser().resolve()
        require(not output.exists(), f"formal output root must be absent: {output}")
        output_absent = True
    payload = {
        "schema": "unblur_slam.turtle_bsd_dpdd_cpu_preflight.v1",
        "status": (
            "blocked_unbound_template"
            if args.template
            else "pass_reference_only"
            if args.reference_only
            else "pass"
        ),
        "contract": str(contract_path),
        "contract_sha256": contract_sha,
        "template": bool(args.template),
        "reference_only": bool(args.reference_only),
        "unbound_fields": unbound,
        "models": model_summary,
        "dpdd": dpdd_summary,
        "bsd": bsd_summary,
        "implementation_pins": implementation,
        "code_bundle": {
            "schema": contract["code_bundle"]["schema"],
            "bundle_sha256": contract["code_bundle"]["bundle_sha256"],
            "file_count": contract["code_bundle"]["file_count"],
            "all_files_rehashed": True,
        },
        "code_bundle_sha256": contract["code_bundle"]["bundle_sha256"],
        "environment_fingerprint": environment_fingerprint,
        "environment_fingerprint_sha256": environment_fingerprint[
            "fingerprint_sha256"
        ],
        "logical_runtime_mapping": {
            "physical_gpu": 1,
            "physical_gpu_uuid": contract["runtime"]["expected_gpu_uuid"],
            "physical_gpu_serial": contract["runtime"]["expected_gpu_serial"],
            "logical_device": "cuda:0",
        },
        "expected_runtime_identity": {
            "physical_gpu": contract["runtime"]["physical_gpu"],
            "physical_gpu_uuid": contract["runtime"]["expected_gpu_uuid"],
            "physical_gpu_serial": contract["runtime"]["expected_gpu_serial"],
            "logical_device": contract["runtime"]["script_device"],
            "mapping_not_queried_by_cpu_preflight": True,
        },
        "torch_runtime_policy": dict(contract["torch_runtime_policy"]),
        "fresh_output_root_absent": output_absent,
        "cuda_visible_devices_observed_not_used": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_queries_or_kernels_launched": False,
        "bsd_test_pixels_opened": False,
        "dpdd_test_pixels_opened": False,
    }
    encoded = json.dumps(payload, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(output, flags, 0o444)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    print(encoded, end="")


if __name__ == "__main__":
    main()
