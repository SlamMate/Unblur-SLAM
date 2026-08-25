#!/usr/bin/env python3
"""CPU-only, fail-closed acceptance audit for Replica424 EVSSM caches.

This tool does not run EVSSM.  It verifies the three independently generated
``precompute_report.json`` files against the pre-registered experiment
contract, source manifests/inventory, the exact official Unblur-SLAM EVSSM
checkpoint, and every source/teacher artifact.  A cache is marked production
eligible only for its declared role after the entire three-way audit passes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from PIL import Image


PRECOMPUTE_SCHEMA = "unblur_slam.video_deblur_evssm_precompute.v1"
AUDIT_SCHEMA = "unblur_slam.replica424_evssm_cache_acceptance.v1"
INVENTORY_SCHEMA = "unblur_slam.replica_blurry_strict_pairs.v1"
SPLIT_SCHEMA = "unblur_slam.replica_blurry_split_ranges.v1"
TEACHER_KIND = "frozen_unblur_slam_evssm"
OFFICIAL_EVSSM_SHA256 = (
    "4c18cd0c783b88e0c4efb8afb60642afed6bdc682cb076bcdd8c358e3c565b41"
)
OFFICIAL_REPLICA424_INVENTORY_SHA256 = (
    "3a1171deb9a034df81aa5adffb78b3de1dcc012feb0076149668007e3edcaadb"
)
OFFICIAL_REPLICA424_SOURCE_REVISION = (
    "1f9d98158c3f27f6ec6de45ee2874c9caf2a2c59"
)

ROLE_CONFIG = {
    "train": ("train", "train", "optimization"),
    "val_temporal": (
        "temporal_validation",
        "val_temporal",
        "checkpoint_selection_only",
    ),
    "test_room2": ("room2_test", "test_room2", "one_shot_test_only"),
}


class AuditError(ValueError):
    """A provenance or artifact failed a production acceptance contract."""


@dataclass(frozen=True)
class AuditInputs:
    contract: Path
    source_inventory: Path
    split_ranges: Path
    data_root: Path
    reports: Mapping[str, Path]
    output: Path


def sha256_file(path: Path, cache: MutableMapping[Path, str] | None = None) -> str:
    path = path.expanduser().resolve()
    if cache is not None and path in cache:
        return cache[path]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if cache is not None:
        cache[path] = value
    return value


def _digest(value: object, label: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise AuditError(f"{label} is not a SHA-256 digest")
    return value


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AuditError(f"{label} must be a JSON object: {path}")
    return payload


def _load_jsonl(path: Path, label: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(
                    f"invalid JSON in {label} at line {line_number}: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise AuditError(f"{label} line {line_number} is not an object")
            records.append(payload)
    if not records:
        raise AuditError(f"{label} contains no records: {path}")
    return records


def _arrays(record: Mapping[str, Any], label: str) -> Tuple[List[str], List[str]]:
    blurry = record.get("blurry")
    sharp = record.get("sharp")
    if not isinstance(blurry, list) or not isinstance(sharp, list):
        raise AuditError(f"{label} requires blurry/sharp arrays")
    if not blurry or len(blurry) != len(sharp):
        raise AuditError(f"{label} has empty or mismatched blurry/sharp arrays")
    if any(not isinstance(value, str) or not value for value in blurry + sharp):
        raise AuditError(f"{label} contains an invalid frame path")
    return list(blurry), list(sharp)


def _resolve(value: object, base: Path, label: str) -> Path:
    if not value:
        raise AuditError(f"missing {label}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    return path


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _assert_cache_roots_disjoint(roots: Mapping[str, Path]) -> None:
    labels = list(roots)
    for index, left_name in enumerate(labels):
        left = roots[left_name].resolve()
        for right_name in labels[index + 1 :]:
            right = roots[right_name].resolve()
            if left == right or _within(left, right) or _within(right, left):
                raise AuditError(
                    f"cache roots overlap: {left_name}={left}, {right_name}={right}"
                )


def _expected_split_records(
    split_payload: Mapping[str, Any], split_name: str
) -> List[Mapping[str, Any]]:
    splits = split_payload.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get(split_name), list):
        raise AuditError(f"split_ranges is missing {split_name}")
    return splits[split_name]


def _source_sequences(
    manifest: Path,
    data_root: Path,
    expected_ranges: Sequence[Mapping[str, Any]],
    label: str,
) -> Tuple[List[Dict[str, Any]], set[Tuple[Path, Path]]]:
    records = _load_jsonl(manifest, f"{label} source manifest")
    if len(records) != len(expected_ranges):
        raise AuditError(f"{label} sequence count does not match split_ranges")
    normalized: List[Dict[str, Any]] = []
    pairs: set[Tuple[Path, Path]] = set()
    for sequence_index, (record, range_record) in enumerate(zip(records, expected_ranges)):
        sequence = str(record.get("sequence", ""))
        if sequence != str(range_record.get("sequence", "")):
            raise AuditError(f"{label} sequence order/name mismatch at {sequence_index}")
        blurry_values, sharp_values = _arrays(record, f"{label}[{sequence_index}]")
        if len(blurry_values) != int(range_record.get("length", -1)):
            raise AuditError(f"{label} range length mismatch at {sequence_index}")
        blurry = [_resolve(value, data_root, "source blurry frame") for value in blurry_values]
        sharp = [_resolve(value, data_root, "source sharp frame") for value in sharp_values]
        sequence_pairs = list(zip(blurry, sharp))
        if len(set(sequence_pairs)) != len(sequence_pairs):
            raise AuditError(f"{label} contains a duplicate pair inside {sequence}")
        overlap = pairs.intersection(sequence_pairs)
        if overlap:
            raise AuditError(f"{label} repeats source pairs across sequences")
        pairs.update(sequence_pairs)
        normalized.append(
            {"sequence": sequence, "blurry": blurry, "sharp": sharp}
        )
    return normalized, pairs


def _inventory_hashes(
    inventory: Mapping[str, Any], data_root: Path
) -> Dict[Path, Tuple[str, int]]:
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise AuditError("unsupported source inventory schema")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise AuditError("source inventory files must be an array")
    output: Dict[Path, Tuple[str, int]] = {}
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise AuditError(f"inventory file {index} is not an object")
        path = (data_root / str(record.get("local", ""))).resolve()
        digest = _digest(record.get("sha256"), f"inventory[{index}].sha256")
        size = int(record.get("size", -1))
        if path in output:
            raise AuditError(f"duplicate inventory path: {path}")
        output[path] = (digest, size)
    return output


def _verify_source_files(
    sequences: Iterable[Mapping[str, Any]],
    inventory: Mapping[Path, Tuple[str, int]],
    digest_cache: MutableMapping[Path, str],
) -> int:
    paths = {
        path
        for sequence in sequences
        for key in ("blurry", "sharp")
        for path in sequence[key]
    }
    for path in sorted(paths):
        if path not in inventory:
            raise AuditError(f"source manifest path is absent from pinned inventory: {path}")
        expected_digest, expected_size = inventory[path]
        if not path.is_file() or path.stat().st_size != expected_size:
            raise AuditError(f"source file size mismatch: {path}")
        if sha256_file(path, digest_cache) != expected_digest:
            raise AuditError(f"source file SHA-256 mismatch: {path}")
    return len(paths)


def _verify_png_matches_source(teacher: Path, source: Path) -> Dict[str, Any]:
    try:
        with Image.open(source) as source_image:
            source_size = source_image.size
        with Image.open(teacher) as image:
            metadata = {
                "format": image.format,
                "mode": image.mode,
                "size": list(image.size),
            }
            if image.format != "PNG" or image.mode != "RGB":
                raise AuditError(f"teacher must be RGB PNG: {teacher}")
            if image.size != source_size:
                raise AuditError(f"teacher/source dimensions differ: {teacher}")
            image.verify()
    except OSError as error:
        raise AuditError(f"invalid teacher image {teacher}: {error}") from error
    return metadata


def _audit_one_report(
    *,
    role: str,
    report_path: Path,
    source_manifest: Path,
    source_manifest_sha256: str,
    source_sequences: Sequence[Mapping[str, Any]],
    expected_checkpoint: Path,
    expected_checkpoint_sha256: str,
    digest_cache: MutableMapping[Path, str],
) -> Dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    report = _load_json(report_path, f"{role} precompute report")
    if report.get("schema") != PRECOMPUTE_SCHEMA:
        raise AuditError(f"{role} has unsupported precompute schema")

    reported_input = _resolve(
        report.get("input_manifest"), report_path.parent, f"{role} input_manifest"
    )
    if reported_input != source_manifest.resolve():
        raise AuditError(f"{role} report points to the wrong source manifest")
    reported_source_digest = _digest(
        report.get("input_manifest_sha256"), f"{role}.input_manifest_sha256"
    )
    if reported_source_digest != source_manifest_sha256:
        raise AuditError(f"{role} source-manifest SHA does not match the contract")
    if sha256_file(reported_input, digest_cache) != source_manifest_sha256:
        raise AuditError(f"{role} source manifest changed after registration")

    reported_checkpoint = _resolve(
        report.get("checkpoint"), report_path.parent, f"{role} checkpoint"
    )
    reported_checkpoint_digest = _digest(
        report.get("checkpoint_sha256"), f"{role}.checkpoint_sha256"
    )
    if reported_checkpoint_digest != expected_checkpoint_sha256:
        raise AuditError(f"{role} did not use the exact official Unblur EVSSM SHA")
    if sha256_file(reported_checkpoint, digest_cache) != expected_checkpoint_sha256:
        raise AuditError(f"{role} checkpoint bytes do not match the official EVSSM")
    if sha256_file(expected_checkpoint, digest_cache) != expected_checkpoint_sha256:
        raise AuditError("registered official EVSSM checkpoint changed on disk")

    output_manifest = _resolve(
        report.get("output_manifest"), report_path.parent, f"{role} output_manifest"
    )
    if output_manifest.parent != report_path.parent:
        raise AuditError(f"{role} output manifest is outside its cache root")
    output_manifest_digest = _digest(
        report.get("output_manifest_sha256"), f"{role}.output_manifest_sha256"
    )
    if sha256_file(output_manifest, digest_cache) != output_manifest_digest:
        raise AuditError(f"{role} output-manifest SHA-256 mismatch")

    output_records = _load_jsonl(output_manifest, f"{role} teacher manifest")
    if len(output_records) != len(source_sequences):
        raise AuditError(f"{role} output/source sequence counts differ")
    report_frames = report.get("frames")
    if not isinstance(report_frames, list):
        raise AuditError(f"{role} report frames must be an array")
    expected_frame_count = sum(len(sequence["blurry"]) for sequence in source_sequences)
    if int(report.get("sequence_count", -1)) != len(source_sequences):
        raise AuditError(f"{role} report sequence_count mismatch")
    if int(report.get("frame_count", -1)) != expected_frame_count:
        raise AuditError(f"{role} report frame_count mismatch")
    if len(report_frames) != expected_frame_count:
        raise AuditError(f"{role} report frames length mismatch")

    teacher_paths: List[Path] = []
    flat_index = 0
    image_contract: Dict[str, Any] | None = None
    teacher_root = (report_path.parent / "teacher").resolve()
    for sequence_index, (source, output) in enumerate(
        zip(source_sequences, output_records)
    ):
        if str(output.get("sequence", "")) != source["sequence"]:
            raise AuditError(f"{role} output sequence order/name mismatch")
        if output.get("teacher_kind") != TEACHER_KIND:
            raise AuditError(f"{role} teacher_kind is not official frozen EVSSM")
        output_blurry, output_sharp = _arrays(
            output, f"{role} output sequence {sequence_index}"
        )
        resolved_blurry = [
            _resolve(value, output_manifest.parent, "output blurry frame")
            for value in output_blurry
        ]
        resolved_sharp = [
            _resolve(value, output_manifest.parent, "output sharp frame")
            for value in output_sharp
        ]
        if resolved_blurry != source["blurry"] or resolved_sharp != source["sharp"]:
            raise AuditError(f"{role} changed source order/content in output manifest")
        teachers = output.get("teacher")
        if not isinstance(teachers, list) or len(teachers) != len(resolved_blurry):
            raise AuditError(f"{role} has incomplete teacher paths")
        for frame_index, (teacher_value, blurry, sharp) in enumerate(
            zip(teachers, resolved_blurry, resolved_sharp)
        ):
            teacher = _resolve(
                teacher_value, output_manifest.parent, f"{role} teacher frame"
            )
            if not _within(teacher, teacher_root):
                raise AuditError(f"{role} teacher escapes cache-local teacher root")
            frame = report_frames[flat_index]
            if not isinstance(frame, dict):
                raise AuditError(f"{role} report frame {flat_index} is not an object")
            if int(frame.get("sequence_index", -1)) != sequence_index:
                raise AuditError(f"{role} sequence_index mismatch at frame {flat_index}")
            if int(frame.get("frame_index", -1)) != frame_index:
                raise AuditError(f"{role} frame_index mismatch at frame {flat_index}")
            report_blurry = _resolve(frame.get("blurry"), report_path.parent, "report blurry")
            report_sharp = _resolve(frame.get("sharp"), report_path.parent, "report sharp")
            report_teacher = _resolve(
                frame.get("teacher"), report_path.parent, "report teacher"
            )
            if (report_blurry, report_sharp, report_teacher) != (blurry, sharp, teacher):
                raise AuditError(f"{role} report path/order mismatch at frame {flat_index}")
            for key, path in (
                ("blurry_sha256", blurry),
                ("sharp_sha256", sharp),
                ("teacher_sha256", teacher),
            ):
                expected = _digest(frame.get(key), f"{role}.frames[{flat_index}].{key}")
                if sha256_file(path, digest_cache) != expected:
                    raise AuditError(f"{role} {key} mismatch at frame {flat_index}")
            current_contract = _verify_png_matches_source(teacher, blurry)
            if image_contract is None:
                image_contract = current_contract
            elif current_contract != image_contract:
                raise AuditError(f"{role} teacher image contracts are inconsistent")
            teacher_paths.append(teacher)
            flat_index += 1

    if len(set(teacher_paths)) != len(teacher_paths):
        raise AuditError(f"{role} reuses a teacher output for different source frames")
    return {
        "role": role,
        "declared_use": ROLE_CONFIG[role][2],
        "production_eligible_for_declared_role": True,
        "precompute_report": str(report_path),
        "precompute_report_sha256": sha256_file(report_path, digest_cache),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": output_manifest_digest,
        "cache_root": str(report_path.parent),
        "sequence_count": len(source_sequences),
        "frame_count": expected_frame_count,
        "teacher_artifact_count": len(teacher_paths),
        "teacher_paths": [str(path) for path in teacher_paths],
        "teacher_image_contract": image_contract,
        "checkpoint": str(reported_checkpoint),
        "checkpoint_sha256": reported_checkpoint_digest,
    }


def audit(inputs: AuditInputs) -> Dict[str, Any]:
    digest_cache: Dict[Path, str] = {}
    contract = _load_json(inputs.contract, "experiment contract")
    inventory_payload = _load_json(inputs.source_inventory, "source inventory")
    split_payload = _load_json(inputs.split_ranges, "split ranges")
    if split_payload.get("schema") != SPLIT_SCHEMA:
        raise AuditError("unsupported split_ranges schema")
    if contract.get("registered_before_training") is not True:
        raise AuditError("experiment contract was not registered before training")
    actual_inventory_sha = sha256_file(inputs.source_inventory, digest_cache)
    registered_inventory_sha = data_inventory_sha = None

    teacher_contract = contract.get("teacher")
    if not isinstance(teacher_contract, dict):
        raise AuditError("experiment contract has no teacher section")
    if teacher_contract.get("kind") != "official_unblur_slam_evssm":
        raise AuditError("contract teacher is not official Unblur-SLAM EVSSM")
    if teacher_contract.get("gopro_or_turtle_allowed") is not False:
        raise AuditError("contract does not explicitly forbid GoPro/TURTLE teachers")
    expected_checkpoint = _resolve(
        teacher_contract.get("checkpoint"), inputs.contract.parent, "official checkpoint"
    )
    expected_checkpoint_sha256 = _digest(
        teacher_contract.get("checkpoint_sha256"), "contract checkpoint_sha256"
    )
    if expected_checkpoint_sha256 != OFFICIAL_EVSSM_SHA256:
        raise AuditError("contract does not register the exact official Unblur EVSSM SHA")

    data_contract = contract.get("data")
    if not isinstance(data_contract, dict):
        raise AuditError("experiment contract has no data section")
    if data_contract.get("fresh_initialization_required") is not True:
        raise AuditError("fresh initialization is not required by the contract")
    if data_contract.get("legacy_replica40_checkpoint_forbidden") is not True:
        raise AuditError("legacy replica40 checkpoint is not explicitly forbidden")
    if data_contract.get("source_inventory_sha256") is not None:
        data_inventory_sha = _digest(
            data_contract.get("source_inventory_sha256"),
            "contract source_inventory_sha256",
        )
    registered_inventory_sha = (
        data_inventory_sha or OFFICIAL_REPLICA424_INVENTORY_SHA256
    )
    if registered_inventory_sha != OFFICIAL_REPLICA424_INVENTORY_SHA256:
        raise AuditError("contract source inventory is not the pinned Replica424 inventory")
    if actual_inventory_sha != registered_inventory_sha:
        raise AuditError("source inventory changed after registration")
    if (
        str(inventory_payload.get("source_revision", ""))
        != OFFICIAL_REPLICA424_SOURCE_REVISION
    ):
        raise AuditError("source inventory revision is not the pinned Hugging Face revision")
    registered_root = Path(str(data_contract.get("root", ""))).expanduser().resolve()
    if registered_root != inputs.data_root.expanduser().resolve():
        raise AuditError("data root does not match the registered experiment")
    if sha256_file(inputs.split_ranges, digest_cache) != _digest(
        data_contract.get("split_ranges_sha256"), "contract split_ranges_sha256"
    ):
        raise AuditError("split_ranges changed after experiment registration")

    validation_report = inputs.split_ranges.parent / "validation_report.json"
    if sha256_file(validation_report, digest_cache) != _digest(
        data_contract.get("validation_report_sha256"),
        "contract validation_report_sha256",
    ):
        raise AuditError("source validation report changed after registration")

    inventory = _inventory_hashes(inventory_payload, inputs.data_root)
    source_by_role: Dict[str, List[Dict[str, Any]]] = {}
    pair_sets: Dict[str, set[Tuple[Path, Path]]] = {}
    manifest_by_role: Dict[str, Path] = {}
    manifest_sha_by_role: Dict[str, str] = {}
    for role, (contract_key, split_key, _declared_use) in ROLE_CONFIG.items():
        role_contract = data_contract.get(contract_key)
        if not isinstance(role_contract, dict):
            raise AuditError(f"contract is missing data.{contract_key}")
        manifest = _resolve(
            role_contract.get("manifest"), inputs.contract.parent, f"{role} manifest"
        )
        manifest_sha = _digest(role_contract.get("sha256"), f"{role} manifest SHA")
        if sha256_file(manifest, digest_cache) != manifest_sha:
            raise AuditError(f"{role} source manifest changed after registration")
        sequences, pairs = _source_sequences(
            manifest,
            inputs.data_root,
            _expected_split_records(split_payload, split_key),
            role,
        )
        expected_pairs = int(role_contract.get("pairs", -1))
        if len(pairs) != expected_pairs:
            raise AuditError(f"{role} pair count does not match the contract")
        source_by_role[role] = sequences
        pair_sets[role] = pairs
        manifest_by_role[role] = manifest
        manifest_sha_by_role[role] = manifest_sha

    role_names = list(ROLE_CONFIG)
    for index, left in enumerate(role_names):
        for right in role_names[index + 1 :]:
            if pair_sets[left].intersection(pair_sets[right]):
                raise AuditError(f"source split leakage between {left} and {right}")
    if {path.parts[-3] for pair in pair_sets["test_room2"] for path in pair} != {
        "room_2"
    }:
        raise AuditError("test_room2 is not exclusively room_2")
    if any(
        path.parts[-3] != "room_1"
        for role in ("train", "val_temporal")
        for pair in pair_sets[role]
        for path in pair
    ):
        raise AuditError("train/val_temporal are not exclusively room_1")

    unique_source_files = _verify_source_files(
        [sequence for sequences in source_by_role.values() for sequence in sequences],
        inventory,
        digest_cache,
    )
    report_paths = {role: inputs.reports[role].expanduser().resolve() for role in ROLE_CONFIG}
    cache_roots = {role: path.parent for role, path in report_paths.items()}
    _assert_cache_roots_disjoint(cache_roots)
    split_results: Dict[str, Dict[str, Any]] = {}
    for role in ROLE_CONFIG:
        split_results[role] = _audit_one_report(
            role=role,
            report_path=report_paths[role],
            source_manifest=manifest_by_role[role],
            source_manifest_sha256=manifest_sha_by_role[role],
            source_sequences=source_by_role[role],
            expected_checkpoint=expected_checkpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            digest_cache=digest_cache,
        )

    teacher_sets = {
        role: set(result["teacher_paths"]) for role, result in split_results.items()
    }
    output_manifests = {
        role: result["output_manifest"] for role, result in split_results.items()
    }
    for index, left in enumerate(role_names):
        for right in role_names[index + 1 :]:
            if teacher_sets[left].intersection(teacher_sets[right]):
                raise AuditError(f"teacher artifact overlap between {left} and {right}")
            if output_manifests[left] == output_manifests[right]:
                raise AuditError(f"output manifest overlap between {left} and {right}")

    # Keep the compact acceptance report useful.  Full ordered teacher paths
    # remain cryptographically bound by each precompute report.
    for result in split_results.values():
        result.pop("teacher_paths", None)
    return {
        "schema": AUDIT_SCHEMA,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "production_eligible": True,
        "eligibility_scope": (
            "frozen official Unblur-SLAM EVSSM teacher caches for their "
            "declared train/selection/one-shot-test roles; not model deployment"
        ),
        "gpu_used": False,
        "contract": str(inputs.contract.resolve()),
        "contract_sha256": sha256_file(inputs.contract, digest_cache),
        "source_inventory": str(inputs.source_inventory.resolve()),
        "source_inventory_sha256": actual_inventory_sha,
        "source_revision": OFFICIAL_REPLICA424_SOURCE_REVISION,
        "split_ranges": str(inputs.split_ranges.resolve()),
        "split_ranges_sha256": sha256_file(inputs.split_ranges, digest_cache),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "source_split_pairwise_disjoint": True,
        "cache_roots_pairwise_disjoint": True,
        "teacher_artifacts_pairwise_disjoint": True,
        "unique_source_file_count": unique_source_files,
        "splits": split_results,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--split-ranges", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--val-temporal-report", type=Path, required=True)
    parser.add_argument("--test-room2-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = AuditInputs(
        contract=args.contract,
        source_inventory=args.source_inventory,
        split_ranges=args.split_ranges,
        data_root=args.data_root,
        reports={
            "train": args.train_report,
            "val_temporal": args.val_temporal_report,
            "test_room2": args.test_room2_report,
        },
        output=args.output,
    )
    try:
        result = audit(inputs)
    except Exception as error:
        failure = {
            "schema": AUDIT_SCHEMA,
            "audited_utc": datetime.now(timezone.utc).isoformat(),
            "production_eligible": False,
            "gpu_used": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(inputs.output, failure)
        print(f"EVSSM cache acceptance failed: {error}", file=sys.stderr)
        return 1
    _atomic_json(inputs.output, result)
    print(
        json.dumps(
            {
                "production_eligible": True,
                "output": str(inputs.output.expanduser().resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "frames": {
                    role: split["frame_count"]
                    for role, split in result["splits"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
