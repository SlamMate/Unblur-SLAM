#!/usr/bin/env python3
"""CPU-only v4 temporal Layer-1 selector and exporter round-trip tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.export_causal_video_deblur as exporter
from scripts.select_causal_video_deblur import (
    EVALUATOR_SCHEMA,
    EXPECTED_EVSSM_SHA256,
    PINNED_V4_MIGRATION_SEMANTIC_DIGEST_SHA256,
    PINNED_V4_PRE_MIGRATION_CHECKPOINT_SHA256,
    PINNED_V4_SAFE_MIGRATED_CHECKPOINT_SHA256,
    ReportContractError,
    build_room2_selection_bundle,
    build_temporal_layer_report,
    write_temporal_layer_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path) -> list[dict[str, object]]:
    records = []
    for sequence_index in range(2):
        sequence = f"temporal_{sequence_index}"
        records.append(
            {
                "sequence": sequence,
                "blurry": [f"assets/{sequence}_blur_{i}.png" for i in range(8)],
                "sharp": [f"assets/{sequence}_sharp_{i}.png" for i in range(8)],
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def _quality(*, psnr: float, ssim: float, l1: float) -> dict[str, float]:
    return {"psnr": psnr, "ssim": ssim, "l1": l1}


def _temporal() -> dict[str, dict[str, float]]:
    return {
        "evssm": {
            "adjacent_change_l1": 0.03,
            "gt_difference_error_l1_not_warp": 0.020,
        },
        "causal": {
            "adjacent_change_l1": 0.029,
            "gt_difference_error_l1_not_warp": 0.019,
        },
        "causal_repeat_current": {
            "adjacent_change_l1": 0.030,
            "gt_difference_error_l1_not_warp": 0.020,
        },
        "causal_alignment_disabled": {
            "adjacent_change_l1": 0.030,
            "gt_difference_error_l1_not_warp": 0.020,
        },
    }


def _motion(frame_index: int) -> dict[str, object]:
    return {
        "schema": exporter.ALIGNMENT_DIAGNOSTICS_SCHEMA_V4,
        "from_frame_index": frame_index - 1,
        "to_frame_index": frame_index,
        "flow_shape": [2, 3, 4],
        "input_shape": [3, 12, 16],
        "quarter_to_input_scale": {"x": 4.0, "y": 4.0},
        "flow_quarter_pixels": {
            "magnitude_p95": 0.5,
            "magnitude_max": 1.0,
            "component_abs_max": 1.0,
        },
        "flow_input_pixels": {
            "magnitude_p95": 2.0,
            "magnitude_max": 4.0,
            "dx_abs_max": 4.0,
            "dy_abs_max": 4.0,
        },
        "confidence": {
            "mean": 0.8,
            "p05": 0.7,
            "p50": 0.8,
            "p95": 0.9,
            "min": 0.6,
            "max": 0.95,
        },
        "warp_valid": {"mean": 0.9, "min": 0.0, "max": 1.0},
        "finite_fraction": 1.0,
    }


def _frame(
    *,
    root: Path,
    record: dict[str, object],
    frame_index: int,
    history: int,
    v4: bool,
    causal_psnr: float,
) -> dict[str, object]:
    sequence = str(record["sequence"])
    blurry = list(record["blurry"])
    sharp = list(record["sharp"])
    row: dict[str, object] = {
        "sequence": sequence,
        "frame_index": frame_index,
        "history_stage": (
            "prefix" if frame_index < history - 1 else "steady_state"
        ),
        "blurry_path": str((root / str(blurry[frame_index])).resolve()),
        "sharp_path": str((root / str(sharp[frame_index])).resolve()),
        "evssm": _quality(psnr=30.0, ssim=0.900, l1=0.0200),
        "causal": _quality(psnr=causal_psnr, ssim=0.901, l1=0.0196),
        "causal_repeat_current": _quality(
            psnr=causal_psnr - 0.1, ssim=0.9005, l1=0.0198
        ),
        "runtime_gate_proxy": {
            "blurry_laplacian_variance": 1.0,
            "evssm_laplacian_variance": 1.0,
            "causal_laplacian_variance": 1.2,
            "causal_vs_evssm_gain": 0.2,
            "causal_vs_blurry_gain": 0.2,
            "passes_default_gate": True,
        },
        "temporal": None if frame_index == 0 else _temporal(),
    }
    if v4:
        row["causal_alignment_disabled"] = _quality(
            psnr=30.05, ssim=0.9002, l1=0.0199
        )
        row["motion_alignment"] = (
            None if frame_index == 0 else _motion(frame_index)
        )
    return row


def _alignment_summary() -> dict[str, object]:
    disabled = {
        "protocol": "same artifact with alignment bypassed",
        "quality": {},
        "temporal": {},
    }
    return {
        "schema": exporter.ALIGNMENT_DIAGNOSTICS_SCHEMA_V4,
        "transition_count": 14,
        "expected_transition_count": 14,
        "flow_quarter_pixels": {
            "magnitude_p95": 0.5,
            "magnitude_max": 1.0,
            "component_abs_max": 1.0,
            "configured_component_abs_max": 16.0,
            "finite_fraction": 1.0,
        },
        "flow_input_pixels": {
            "magnitude_p95": 2.0,
            "magnitude_max": 4.0,
            "finite_fraction": 1.0,
        },
        "confidence": {
            "mean": 0.8,
            "p05": 0.7,
            "p50": 0.8,
            "p95": 0.9,
            "min": 0.6,
            "max": 0.95,
            "finite_fraction": 1.0,
        },
        "warp_valid": {
            "mean": 0.9,
            "min": 0.0,
            "max": 1.0,
            "finite_fraction": 1.0,
        },
        "integrity": {
            "transition_count_matches": True,
            "all_finite": True,
            "flow_within_configured_bound": True,
            "confidence_in_0_1": True,
            "warp_valid_in_0_1": True,
            "passed": True,
        },
        "controls": {
            "repeat_current": {"protocol": "repeat current"},
            "alignment_disabled": disabled,
        },
    }


def _artifact_metadata(
    teacher: dict[str, object], *, include_migration: bool = True
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "format": exporter.TORCHSCRIPT_FORMAT_V4,
        "checkpoint_format": exporter.CHECKPOINT_FORMAT_V4,
        "source_checkpoint_sha256": PINNED_V4_SAFE_MIGRATED_CHECKPOINT_SHA256,
        "artifact_role": "diagnostic_evaluation_only",
        "deployment_eligible": False,
        "model_config": dict(exporter.REGISTERED_V4_MODEL_CONFIG),
        "registered_contract": {
            "schema": exporter.REGISTERED_V4_CONTRACT_SCHEMA,
            "sha256": exporter.REGISTERED_V4_CONTRACT_SHA256,
        },
        "warm_start_provenance": {
            "schema": exporter.WARM_START_SCHEMA_V4,
            "source_sha256": exporter.REGISTERED_V4_WARM_START_SHA256,
            "source_format": exporter.CHECKPOINT_FORMAT_V3,
            "source_model_config": dict(
                exporter.REGISTERED_V4_BASE_MODEL_CONFIG
            ),
            "allowed_missing_alignment_keys": [
                "motion_alignment_gate",
                "motion_aligner.match_projection.weight",
                "motion_aligner.offsets",
            ],
            "optimizer_state_loaded": False,
            "identity_probe": {
                "passed": True,
                "atol": 1.0e-6,
                "max_abs_difference": 0.0,
            },
        },
        "teacher_provenance": teacher,
        "exported_methods": [
            "forward",
            "forward_sequence",
            "forward_sequence_with_motion_diagnostics",
            "forward_sequence_alignment_disabled",
        ],
        "optimization_contract": {
            "execution_device": "cpu",
            "amp_requested": False,
            "amp_effective": False,
            "num_workers": 0,
        },
        "training_contract": {
            "terminal_checkpoint_policy": (
                "unconditional_atomic_save_at_exact_optimizer_step_600_before_exit"
            ),
            "resume_rng_policy": (
                "epoch_boundary_python_numpy_torch_cpu_and_loader_generators"
            ),
        },
        "source_checkpoint_epoch": 25,
        "source_checkpoint_step": 600,
        "training_phase": "joint",
        "data_identity": dict(exporter.REGISTERED_V4_DATA_IDENTITY),
        "rng_state_provenance": {
            "schema": exporter.RNG_STATE_SCHEMA_V4,
            "checkpoint_boundary": "epoch_end_no_pending_accumulation",
            "captured": True,
        },
    }
    if include_migration:
        metadata["checkpoint_migration"] = _checkpoint_migration()
    return metadata


def _checkpoint_migration() -> dict[str, object]:
    return {
        "schema": exporter.CHECKPOINT_MIGRATION_SCHEMA_V1,
        "kind": exporter.CHECKPOINT_MIGRATION_KIND_V1,
        "source_checkpoint_sha256": (
            PINNED_V4_PRE_MIGRATION_CHECKPOINT_SHA256
        ),
        "allowed_changes": list(
            exporter.CHECKPOINT_MIGRATION_ALLOWED_CHANGES_V1
        ),
        "semantic_digest": {
            "schema": exporter.CHECKPOINT_SEMANTIC_DIGEST_SCHEMA_V1,
            "algorithm": exporter.CHECKPOINT_SEMANTIC_DIGEST_ALGORITHM_V1,
            "sha256": PINNED_V4_MIGRATION_SEMANTIC_DIGEST_SHA256,
            "source_and_target_equal": True,
        },
    }


def _write_artifact(path: Path, metadata: dict[str, object]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "synthetic/extra/metadata.json",
            json.dumps(metadata, sort_keys=True),
        )


def _replace_candidate_metadata(
    fixture: dict[str, object], metadata: dict[str, object]
) -> None:
    candidate = Path(fixture["candidate"])
    _write_artifact(candidate, metadata)
    report_path = Path(fixture["h3_report"])
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["evaluated_artifact_sha256"] = _sha256(candidate)
    report_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _write_report(
    path: Path,
    *,
    manifest: Path,
    records: list[dict[str, object]],
    artifact: Path,
    teacher: dict[str, object],
    history: int,
    v4: bool,
    causal_psnr: float,
    source_checkpoint_sha256: str,
) -> dict[str, object]:
    frames = [
        _frame(
            root=manifest.parent,
            record=record,
            frame_index=frame_index,
            history=history,
            v4=v4,
            causal_psnr=causal_psnr,
        )
        for record in records
        for frame_index in range(8)
    ]
    payload: dict[str, object] = {
        "schema": exporter.EVALUATOR_SCHEMA_V4 if v4 else EVALUATOR_SCHEMA,
        "checkpoint": str(artifact.resolve()),
        "evaluated_artifact_sha256": _sha256(artifact),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "manifest": str(manifest.resolve()),
        "teacher_provenance": teacher,
        "input_domain": "evssm",
        "history": history,
        "lpips_computed": False,
        "lpips_protocol": {
            "implementation": "torchmetrics.image.lpip",
            "network": "alex",
            "normalize_input_0_1": True,
            "per_frame_state_reset": True,
        },
        "frame_count": 16,
        "temporal_pair_count": 14,
        "frames": frames,
    }
    if v4:
        alignment = _alignment_summary()
        payload["transition_count"] = 14
        payload["alignment_diagnostics"] = alignment
        payload["alignment_disabled_control"] = alignment["controls"][
            "alignment_disabled"
        ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _fixture(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "val_temporal.jsonl"
    records = _write_manifest(manifest)
    teacher = {
        "schema": "unblur_slam.video_deblur_teacher_provenance.v1",
        "storage": "precomputed_png_rgb8",
        "teacher_domain": "evssm_restored_rgb_0_1",
        "evssm_checkpoint_sha256": EXPECTED_EVSSM_SHA256,
    }
    candidate = root / "candidate_v4.pt"
    _write_artifact(candidate, _artifact_metadata(teacher))
    history1 = root / "history1.pt"
    history1.write_bytes(b"pinned synthetic H1 artifact")
    h3_report = root / "h3_v4_metrics.json"
    h1_report = root / "h1_v3_metrics.json"
    _write_report(
        h3_report,
        manifest=manifest,
        records=records,
        artifact=candidate,
        teacher=teacher,
        history=3,
        v4=True,
        causal_psnr=30.2,
        source_checkpoint_sha256=PINNED_V4_SAFE_MIGRATED_CHECKPOINT_SHA256,
    )
    _write_report(
        h1_report,
        manifest=manifest,
        records=records,
        artifact=history1,
        teacher=teacher,
        history=1,
        v4=False,
        causal_psnr=30.1,
        source_checkpoint_sha256="b" * 64,
    )
    return {
        "manifest": manifest,
        "candidate": candidate,
        "history1": history1,
        "h3_report": h3_report,
        "h1_report": h1_report,
    }


def _select(fixture: dict[str, object]) -> dict[str, object]:
    manifest = Path(fixture["manifest"])
    h1_report = Path(fixture["h1_report"])
    history1 = Path(fixture["history1"])
    return build_temporal_layer_report(
        temporal_report_path=Path(fixture["h3_report"]),
        history1_report_path=h1_report,
        temporal_manifest_path=manifest,
        source_root_path=manifest.parent,
        expected_temporal_manifest_sha256=_sha256(manifest),
        expected_history1_report_sha256=_sha256(h1_report),
        expected_history1_artifact_sha256=_sha256(history1),
        expected_history1_checkpoint_sha256="b" * 64,
    )


def test_v4_layer1_roundtrips_exporter_and_pins_h1() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        try:
            build_temporal_layer_report(
                temporal_report_path=Path(fixture["h3_report"]),
                history1_report_path=Path(fixture["h1_report"]),
                temporal_manifest_path=Path(fixture["manifest"]),
                source_root_path=root,
                expected_temporal_manifest_sha256=_sha256(
                    Path(fixture["manifest"])
                ),
            )
        except ReportContractError as error:
            assert "pinned H1 best evaluator" in str(error)
        else:
            raise AssertionError("synthetic H1 bypassed the production pin")

        layer = _select(fixture)
        assert layer["schema"] == exporter.DEPLOYMENT_LAYER_REPORT_SCHEMA_V4
        assert layer["policy"] == exporter.DEPLOYMENT_SELECTION_POLICY_V4
        assert layer["registered_contract_sha256"] == (
            exporter.REGISTERED_V4_CONTRACT_SHA256
        )
        assert layer["alignment_integrity_passed"] is True
        assert layer["alignment_evidence"]["transition_count"] == 14
        assert layer["eligible"] is True
        assert layer["v4_provenance"]["checkpoint_migration"][
            "source_checkpoint_sha256"
        ] == PINNED_V4_PRE_MIGRATION_CHECKPOINT_SHA256
        output = root / "v4_layer1.json"
        write_temporal_layer_report(output, layer, overwrite=False)
        normalized = exporter._resolve_layer_report(
            selection_report_path=output,
            layer_name="temporal_validation",
            layer_entry={
                "report": str(output),
                "report_sha256": _sha256(output),
                "manifest_sha256": _sha256(Path(fixture["manifest"])),
            },
            expected_manifest_sha256=_sha256(Path(fixture["manifest"])),
            checkpoint_sha256=PINNED_V4_SAFE_MIGRATED_CHECKPOINT_SHA256,
            evssm_checkpoint_sha256=EXPECTED_EVSSM_SHA256,
            layer_report_schema=exporter.DEPLOYMENT_LAYER_REPORT_SCHEMA_V4,
            registered_contract_sha256=exporter.REGISTERED_V4_CONTRACT_SHA256,
            h3_evaluator_schema=exporter.EVALUATOR_SCHEMA_V4,
            require_v4_alignment_diagnostics=True,
        )
        assert normalized["eligible"] is True
        assert normalized["alignment_evidence"]["transition_count"] == 14
        try:
            exporter.validate_v4_deployment_selection(
                output,
                Path(fixture["candidate"]),
                {"evssm_checkpoint_sha256": EXPECTED_EVSSM_SHA256},
            )
        except ValueError as error:
            assert "v4 layered report" in str(error)
        else:
            raise AssertionError(
                "Layer1-only evidence incorrectly authorized formal v4 export"
            )


def test_v4_checkpoint_migration_lineage_is_validated_and_reported() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        teacher = json.loads(Path(fixture["h3_report"]).read_text())[
            "teacher_provenance"
        ]
        metadata = _artifact_metadata(teacher)
        metadata["checkpoint_migration"] = _checkpoint_migration()
        _replace_candidate_metadata(fixture, metadata)
        layer = _select(fixture)
        lineage = layer["v4_provenance"]["checkpoint_migration"]
        assert lineage["source_checkpoint_sha256"] == (
            PINNED_V4_PRE_MIGRATION_CHECKPOINT_SHA256
        )
        assert lineage["target_checkpoint_sha256"] == (
            PINNED_V4_SAFE_MIGRATED_CHECKPOINT_SHA256
        )
        assert lineage["semantic_digest"]["sha256"] == (
            PINNED_V4_MIGRATION_SEMANTIC_DIGEST_SHA256
        )
        assert lineage["semantic_digest"]["source_and_target_equal"] is True

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        teacher = json.loads(Path(fixture["h3_report"]).read_text())[
            "teacher_provenance"
        ]
        metadata = _artifact_metadata(teacher, include_migration=False)
        _replace_candidate_metadata(fixture, metadata)
        try:
            _select(fixture)
        except ReportContractError as error:
            assert "requires checkpoint_migration lineage" in str(error)
        else:
            raise AssertionError(
                "registered formal v4 artifact accepted stripped migration lineage"
            )

    mutations = (
        (
            lambda value: value.__setitem__(
                "source_checkpoint_sha256", "d" * 64
            ),
            "pinned formal v4 terminal",
        ),
        (
            lambda value: value.__setitem__(
                "allowed_changes", list(reversed(value["allowed_changes"]))
            ),
            "allowed_changes",
        ),
        (
            lambda value: value["semantic_digest"].__setitem__(
                "source_and_target_equal", False
            ),
            "semantic equality",
        ),
        (
            lambda value: value["semantic_digest"].__setitem__(
                "sha256", "d" * 64
            ),
            "pinned semantic digest",
        ),
        (
            lambda value: value["semantic_digest"].__setitem__(
                "sha256", "not-a-digest"
            ),
            "must be a SHA-256 digest",
        ),
    )
    for mutate, message in mutations:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _fixture(root)
            teacher = json.loads(Path(fixture["h3_report"]).read_text())[
                "teacher_provenance"
            ]
            migration = _checkpoint_migration()
            mutate(migration)
            metadata = _artifact_metadata(teacher)
            metadata["checkpoint_migration"] = migration
            _replace_candidate_metadata(fixture, metadata)
            try:
                _select(fixture)
            except ReportContractError as error:
                assert message in str(error)
            else:
                raise AssertionError(
                    f"malformed checkpoint migration was accepted: {message}"
                )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        teacher = json.loads(Path(fixture["h3_report"]).read_text())[
            "teacher_provenance"
        ]
        metadata = _artifact_metadata(teacher)
        metadata["source_checkpoint_sha256"] = "e" * 64
        _replace_candidate_metadata(fixture, metadata)
        report_path = Path(fixture["h3_report"])
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["source_checkpoint_sha256"] = "e" * 64
        report_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        try:
            _select(fixture)
        except ReportContractError as error:
            assert "pinned tensor-safe v4 checkpoint" in str(error)
        else:
            raise AssertionError("unregistered tensor-safe target SHA was accepted")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        teacher = json.loads(Path(fixture["h3_report"]).read_text())[
            "teacher_provenance"
        ]
        metadata = _artifact_metadata(teacher)
        metadata["source_checkpoint_sha256"] = (
            PINNED_V4_PRE_MIGRATION_CHECKPOINT_SHA256
        )
        metadata["checkpoint_migration"] = _checkpoint_migration()
        _replace_candidate_metadata(fixture, metadata)
        report_path = Path(fixture["h3_report"])
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["source_checkpoint_sha256"] = (
            PINNED_V4_PRE_MIGRATION_CHECKPOINT_SHA256
        )
        report_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        try:
            _select(fixture)
        except ReportContractError as error:
            assert "distinct SHA-256" in str(error)
        else:
            raise AssertionError("self-referential migration SHA was accepted")


def test_v4_alignment_and_artifact_provenance_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        payload = json.loads(Path(fixture["h3_report"]).read_text())
        payload["frames"][-1]["motion_alignment"]["finite_fraction"] = 0.99
        Path(fixture["h3_report"]).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        try:
            _select(fixture)
        except ReportContractError as error:
            assert "alignment evidence" in str(error)
        else:
            raise AssertionError("non-finite alignment evidence was accepted")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        teacher = json.loads(Path(fixture["h3_report"]).read_text())[
            "teacher_provenance"
        ]
        metadata = _artifact_metadata(teacher)
        metadata["registered_contract"]["sha256"] = "0" * 64
        _write_artifact(Path(fixture["candidate"]), metadata)
        payload = json.loads(Path(fixture["h3_report"]).read_text())
        payload["evaluated_artifact_sha256"] = _sha256(
            Path(fixture["candidate"])
        )
        Path(fixture["h3_report"]).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        try:
            _select(fixture)
        except ReportContractError as error:
            assert "registered v4 contract" in str(error)
        else:
            raise AssertionError("forged v4 artifact provenance was accepted")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        teacher = json.loads(Path(fixture["h3_report"]).read_text())[
            "teacher_provenance"
        ]
        metadata = _artifact_metadata(teacher)
        metadata["model_config"] = {
            **metadata["model_config"],
            "channels": 16,
        }
        _write_artifact(Path(fixture["candidate"]), metadata)
        payload = json.loads(Path(fixture["h3_report"]).read_text())
        payload["evaluated_artifact_sha256"] = _sha256(
            Path(fixture["candidate"])
        )
        Path(fixture["h3_report"]).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        try:
            _select(fixture)
        except ReportContractError as error:
            assert "model_config" in str(error)
        else:
            raise AssertionError("wrong v4 artifact architecture was accepted")


def test_v4_room2_is_fail_closed_before_any_room2_read() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        layer = _select(fixture)
        layer_path = root / "v4_layer1.json"
        write_temporal_layer_report(layer_path, layer, overwrite=False)
        try:
            build_room2_selection_bundle(
                temporal_layer_report_path=layer_path,
                temporal_layer_report_sha256=_sha256(layer_path),
                room2_report_path=root / "must_not_be_opened.json",
                room2_manifest_path=root / "must_not_be_opened.jsonl",
                source_root_path=root,
                expected_room2_manifest_sha256="0" * 64,
                expected_temporal_manifest_sha256=_sha256(
                    Path(fixture["manifest"])
                ),
                expected_room2_frame_identity_sha256="0" * 64,
                expected_room2_frame_count=174,
            )
        except ReportContractError as error:
            assert "room2 was not opened" in str(error)
        else:
            raise AssertionError("v4 unexpectedly entered the room2 stage")

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/select_causal_video_deblur.py"),
                "--stage",
                "temporal",
                "--temporal-val-report",
                str(fixture["h3_report"]),
                "--history1-temporal-val-report",
                str(fixture["h1_report"]),
                "--temporal-val-source-manifest",
                str(fixture["manifest"]),
                "--source-root",
                str(root),
                "--room2-source-manifest",
                str(root / "must_not_be_opened.jsonl"),
                "--output",
                str(root / "must_not_write.json"),
            ],
            check=False,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "does not accept any room2 input" in result.stderr


if __name__ == "__main__":
    test_v4_layer1_roundtrips_exporter_and_pins_h1()
    test_v4_checkpoint_migration_lineage_is_validated_and_reported()
    test_v4_alignment_and_artifact_provenance_fail_closed()
    test_v4_room2_is_fail_closed_before_any_room2_read()
    print("select_causal_video_deblur_v4=PASS")
