#!/usr/bin/env python3
"""Bind a canonical BSD validation-prefix to a new E/G/O-only contract."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bsd_dpdd_contract import (  # noqa: E402
    BSD_AUDIT_SCHEMA,
    BSD_DATASET_SCHEMA,
    BSD_SEQUENCE_SCHEMA,
    REFERENCE_BOUND_STATUS,
    load_contract,
    load_json_object,
    require,
    sha256_file,
    validate_protocol,
)


def build_reference_contract(
    template: Mapping[str, Any],
    *,
    template_path: Path,
    template_sha256: str,
    dataset_root: Path,
    output_root: Path,
) -> Mapping[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    require(dataset_root.is_dir(), f"BSD validation-prefix root is missing: {dataset_root}")
    require(str(output_root).startswith("/srv/szha0669/"), "formal output must remain on /srv")
    require(not output_root.exists(), f"formal output root must be fresh: {output_root}")
    dataset_path = dataset_root / "dataset_manifest.json"
    audit_path = dataset_root / "materialization_audit.json"
    dataset = load_json_object(dataset_path)
    audit = load_json_object(audit_path)
    require(dataset.get("schema") == BSD_DATASET_SCHEMA, "BSD prefix dataset schema changed")
    require(dataset.get("scope") == "validation_only_partial_prefix", "BSD prefix scope changed")
    require(dataset.get("train_materialized") is False, "BSD prefix unexpectedly contains train")
    require(dataset.get("full_dataset_ready") is False, "BSD prefix falsely claims full data")
    require(dataset.get("splits") == {"validation": {"sequences": 20, "frames": 2000}}, "BSD prefix counts changed")
    canonical = dataset.get("canonical_manifests", {})
    require(set(canonical) == {"validation"}, "BSD prefix must expose validation only")
    validation_entry = canonical["validation"]
    require(validation_entry.get("schema") == BSD_SEQUENCE_SCHEMA, "BSD validation sequence schema changed")
    validation_path = (dataset_root / validation_entry["path"]).resolve()
    require(validation_path.is_file(), "BSD canonical validation manifest is missing")
    require(sha256_file(validation_path) == validation_entry.get("sha256"), "BSD validation manifest hash changed")
    require(audit.get("schema") == BSD_AUDIT_SCHEMA and audit.get("status") == "pass", "BSD prefix audit did not pass")
    require(audit.get("validation_only") is True, "BSD prefix audit is not validation-only")
    require(audit.get("train_materialized") is False, "BSD prefix audit unexpectedly includes train")
    test_audit = audit.get("test_audit", {})
    require(test_audit.get("pixels_opened") is False, "BSD prefix audit touched test pixels")
    require(test_audit.get("member_payload_bytes_read") == 0, "BSD prefix audit read test payload")

    bound = copy.deepcopy(dict(template))
    bound["status"] = REFERENCE_BOUND_STATUS
    bound["launch_authorized"] = False
    bound["reference_launch_authorized"] = True
    bsd = bound["data"]["bsd"]
    bsd["dataset_root"] = str(dataset_root)
    bsd["dataset_manifest"] = str(dataset_path)
    bsd["dataset_manifest_sha256"] = sha256_file(dataset_path)
    bsd["materialization_audit"] = str(audit_path)
    bsd["materialization_audit_sha256"] = sha256_file(audit_path)
    bsd["validation_manifest"] = str(validation_path)
    bsd["validation_manifest_sha256"] = sha256_file(validation_path)
    bound["sealed_outputs"]["output_root"] = str(output_root)
    bound["reference_binding"] = {
        "mode": "validation_only_partial_prefix",
        "template": str(template_path),
        "template_sha256": template_sha256,
        "dataset_manifest_sha256": sha256_file(dataset_path),
        "materialization_audit_sha256": sha256_file(audit_path),
        "validation_manifest_sha256": sha256_file(validation_path),
        "code_bundle_sha256": bound["code_bundle"]["bundle_sha256"],
        "environment_fingerprint_sha256": bound["environment_fingerprint"][
            "fingerprint_sha256"
        ],
        "logical_cuda0_mapping": {
            "physical_gpu": bound["runtime"]["physical_gpu"],
            "physical_gpu_uuid": bound["runtime"]["expected_gpu_uuid"],
            "physical_gpu_serial": bound["runtime"]["expected_gpu_serial"],
            "logical_device": bound["runtime"]["script_device"],
            "hardware_not_queried_by_binder": True,
        },
        "validation_prefix_receipt": str(dataset_root / "validation_prefix_receipt.json"),
        "validation_prefix_receipt_sha256": sha256_file(
            dataset_root / "validation_prefix_receipt.json"
        ),
        "BSD_train_resolved_or_read": False,
        "BSD_test_pixels_opened": False,
        "training_authorized": False,
    }
    validate_protocol(bound, allow_template=False, reference_only=True)
    return bound


def write_new(path: Path, payload: Mapping[str, Any]) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    sidecar_fd = os.open(sidecar, sidecar_flags, 0o444)
    with os.fdopen(sidecar_fd, "w", encoding="utf-8") as stream:
        stream.write(f"{digest}  {path.name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--expected-template-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-contract", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    template_path, template, template_sha = load_contract(
        args.template, expected_sha256=args.expected_template_sha256
    )
    validate_protocol(template, allow_template=True)
    bound = build_reference_contract(
        template,
        template_path=template_path,
        template_sha256=template_sha,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
    )
    digest = write_new(args.output_contract, bound)
    print(json.dumps({"contract": str(args.output_contract.resolve()), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
