import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_replica424_evssm_cache as cache_audit


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((12, 16, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _source_record(root: Path, scene: str, indices: list[int], name: str) -> dict:
    blurry = []
    sharp = []
    for offset, index in enumerate(indices):
        blurry_relative = f"replica_blur/{scene}/blur/rgb_{index}.png"
        sharp_relative = f"replica_blur/{scene}/sharp/rgb_{index + 18}.png"
        _image(root / blurry_relative, 20 + offset + index % 31)
        _image(root / sharp_relative, 80 + offset + index % 31)
        blurry.append(blurry_relative)
        sharp.append(sharp_relative)
    return {"sequence": name, "blurry": blurry, "sharp": sharp}


def _range(scene: str, indices: list[int], name: str) -> dict:
    return {
        "sequence": name,
        "scene": scene,
        "blur_start": indices[0],
        "blur_end": indices[-1],
        "sharp_start": indices[0] + 18,
        "sharp_end": indices[-1] + 18,
        "length": len(indices),
        "step": 36,
        "sharp_offset": 18,
    }


def _precompute(
    *,
    root: Path,
    source_manifest: Path,
    checkpoint: Path,
    cache_root: Path,
) -> Path:
    source_records = [
        json.loads(line)
        for line in source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_records = []
    frames = []
    for sequence_index, source in enumerate(source_records):
        teachers = []
        blurry = [(root / value).resolve() for value in source["blurry"]]
        sharp = [(root / value).resolve() for value in source["sharp"]]
        for frame_index, (blurry_path, sharp_path) in enumerate(zip(blurry, sharp)):
            teacher = (
                cache_root
                / "teacher"
                / f"seq_{sequence_index:04d}"
                / f"{frame_index:06d}.png"
            ).resolve()
            _image(teacher, 140 + sequence_index + frame_index)
            teachers.append(str(teacher))
            frames.append(
                {
                    "sequence_index": sequence_index,
                    "frame_index": frame_index,
                    "blurry": str(blurry_path),
                    "blurry_sha256": _sha(blurry_path),
                    "sharp": str(sharp_path),
                    "sharp_sha256": _sha(sharp_path),
                    "teacher": str(teacher),
                    "teacher_sha256": _sha(teacher),
                }
            )
        output_records.append(
            {
                "sequence": source["sequence"],
                "blurry": [str(path) for path in blurry],
                "sharp": [str(path) for path in sharp],
                "teacher": teachers,
                "teacher_kind": cache_audit.TEACHER_KIND,
            }
        )
    output_manifest = cache_root / "sequences_with_evssm.jsonl"
    _write_jsonl(output_manifest, output_records)
    report = cache_root / "precompute_report.json"
    _write_json(
        report,
        {
            "schema": cache_audit.PRECOMPUTE_SCHEMA,
            "input_manifest": str(source_manifest.resolve()),
            "input_manifest_sha256": _sha(source_manifest),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha(checkpoint),
            "output_manifest": str(output_manifest.resolve()),
            "output_manifest_sha256": _sha(output_manifest),
            "sequence_count": len(source_records),
            "frame_count": len(frames),
            "frames": frames,
        },
    )
    return report


def _fixture(tmp_path: Path) -> dict:
    root = tmp_path / "data"
    package = root / "replica424_v1"
    manifests = package / "manifests"
    sources = {
        "train": [
            _source_record(root, "room_1", [0, 36], "train_room1_00000_00036")
        ],
        "val_temporal": [
            _source_record(root, "room_1", [72], "val_room1_00072_00072")
        ],
        "test_room2": [
            _source_record(root, "room_2", [0], "test_room2_00000_00000")
        ],
    }
    manifest_paths = {}
    for role, records in sources.items():
        manifest_paths[role] = manifests / f"{role}.jsonl"
        _write_jsonl(manifest_paths[role], records)

    split_ranges = package / "split_ranges.json"
    _write_json(
        split_ranges,
        {
            "schema": cache_audit.SPLIT_SCHEMA,
            "splits": {
                "train": [_range("room_1", [0, 36], sources["train"][0]["sequence"])],
                "val_temporal": [
                    _range("room_1", [72], sources["val_temporal"][0]["sequence"])
                ],
                "test_room2": [
                    _range("room_2", [0], sources["test_room2"][0]["sequence"])
                ],
            },
        },
    )
    validation_report = package / "validation_report.json"
    _write_json(validation_report, {"validated": True})

    source_paths = {
        (root / value).resolve()
        for records in sources.values()
        for record in records
        for key in ("blurry", "sharp")
        for value in record[key]
    }
    inventory = root / "replica424_source_inventory.json"
    _write_json(
        inventory,
        {
            "schema": cache_audit.INVENTORY_SCHEMA,
            "source_revision": "fixture-revision",
            "files": [
                {
                    "local": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                    "sha256": _sha(path),
                }
                for path in sorted(source_paths)
            ],
        },
    )

    checkpoint = tmp_path / "net_g_latest_batch_8_no_NYU.pth"
    checkpoint.write_bytes(b"official-unblur-evssm-test-checkpoint")
    checkpoint_sha = _sha(checkpoint)

    contract = tmp_path / "experiment_contract.json"
    data_contract = {
        "root": str(root.resolve()),
        "fresh_initialization_required": True,
        "legacy_replica40_checkpoint_forbidden": True,
        "train": {
            "manifest": str(manifest_paths["train"].resolve()),
            "sha256": _sha(manifest_paths["train"]),
            "pairs": 2,
        },
        "temporal_validation": {
            "manifest": str(manifest_paths["val_temporal"].resolve()),
            "sha256": _sha(manifest_paths["val_temporal"]),
            "pairs": 1,
        },
        "room2_test": {
            "manifest": str(manifest_paths["test_room2"].resolve()),
            "sha256": _sha(manifest_paths["test_room2"]),
            "pairs": 1,
        },
        "split_ranges_sha256": _sha(split_ranges),
        "validation_report_sha256": _sha(validation_report),
    }
    _write_json(
        contract,
        {
            "registered_before_training": True,
            "teacher": {
                "kind": "official_unblur_slam_evssm",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "gopro_or_turtle_allowed": False,
            },
            "data": data_contract,
        },
    )

    reports = {
        role: _precompute(
            root=root,
            source_manifest=manifest_paths[role],
            checkpoint=checkpoint,
            cache_root=package / f"evssm_{role}",
        )
        for role in sources
    }
    inputs = cache_audit.AuditInputs(
        contract=contract,
        source_inventory=inventory,
        split_ranges=split_ranges,
        data_root=root,
        reports=reports,
        output=package / "cache_acceptance.json",
    )
    return {
        "inputs": inputs,
        "reports": reports,
        "contract": contract,
        "split_ranges": split_ranges,
        "checkpoint_sha": checkpoint_sha,
        "inventory_sha": _sha(inventory),
        "source_revision": "fixture-revision",
    }


class Replica424EvssmCacheAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.case = _fixture(Path(self.temporary.name))
        self.patchers = [
            mock.patch.object(
                cache_audit,
                "OFFICIAL_EVSSM_SHA256",
                self.case["checkpoint_sha"],
            ),
            mock.patch.object(
                cache_audit,
                "OFFICIAL_REPLICA424_INVENTORY_SHA256",
                self.case["inventory_sha"],
            ),
            mock.patch.object(
                cache_audit,
                "OFFICIAL_REPLICA424_SOURCE_REVISION",
                self.case["source_revision"],
            ),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def test_accepts_exact_three_way_cache_contract(self) -> None:
        result = cache_audit.audit(self.case["inputs"])
        self.assertIs(result["production_eligible"], True)
        self.assertIs(result["gpu_used"], False)
        self.assertEqual(result["checkpoint_sha256"], self.case["checkpoint_sha"])
        self.assertIs(result["source_split_pairwise_disjoint"], True)
        self.assertIs(result["teacher_artifacts_pairwise_disjoint"], True)
        self.assertEqual(
            {
                role: split["frame_count"]
                for role, split in result["splits"].items()
            },
            {"train": 2, "val_temporal": 1, "test_room2": 1},
        )

    def test_rejects_teacher_image_sha_tamper(self) -> None:
        report = json.loads(
            self.case["reports"]["train"].read_text(encoding="utf-8")
        )
        _image(Path(report["frames"][0]["teacher"]), 255)
        with self.assertRaisesRegex(
            cache_audit.AuditError, "teacher_sha256 mismatch"
        ):
            cache_audit.audit(self.case["inputs"])

    def test_rejects_output_source_reordering_with_updated_manifest_sha(self) -> None:
        report_path = self.case["reports"]["train"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        output = Path(report["output_manifest"])
        record = json.loads(output.read_text(encoding="utf-8"))
        record["blurry"] = list(reversed(record["blurry"]))
        _write_jsonl(output, [record])
        report["output_manifest_sha256"] = _sha(output)
        _write_json(report_path, report)
        with self.assertRaisesRegex(
            cache_audit.AuditError, "changed source order/content"
        ):
            cache_audit.audit(self.case["inputs"])

    def test_rejects_nonofficial_checkpoint_sha(self) -> None:
        report_path = self.case["reports"]["val_temporal"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["checkpoint_sha256"] = "0" * 64
        _write_json(report_path, report)
        with self.assertRaisesRegex(
            cache_audit.AuditError, "exact official Unblur EVSSM SHA"
        ):
            cache_audit.audit(self.case["inputs"])

    def test_rejects_registered_split_ranges_sha_change(self) -> None:
        with self.case["split_ranges"].open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(cache_audit.AuditError, "split_ranges changed"):
            cache_audit.audit(self.case["inputs"])

    def test_rejects_overlapping_cache_roots(self) -> None:
        reports = dict(self.case["inputs"].reports)
        reports["test_room2"] = reports["val_temporal"]
        inputs = cache_audit.AuditInputs(
            contract=self.case["inputs"].contract,
            source_inventory=self.case["inputs"].source_inventory,
            split_ranges=self.case["inputs"].split_ranges,
            data_root=self.case["inputs"].data_root,
            reports=reports,
            output=self.case["inputs"].output,
        )
        with self.assertRaisesRegex(cache_audit.AuditError, "cache roots overlap"):
            cache_audit.audit(inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
