#!/usr/bin/env python3
"""CPU-only contracts for causal-EVSSM evaluator selection and LPIPS wiring."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_causal_video_deblur import (
    _lpips_value,
    _metrics,
    _quality_breakdown,
)
from scripts.select_causal_video_deblur import (
    EVALUATOR_SCHEMA,
    EXPECTED_EVSSM_SHA256,
    ReportContractError,
    _load_source_manifest,
    build_room2_selection_bundle,
    build_temporal_layer_report,
    write_room2_selection_bundle,
    write_temporal_layer_report,
)
import scripts.export_causal_video_deblur as exporter


class FakeLPIPS:
    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return (prediction - target).abs().mean()

    def reset(self) -> None:
        self.resets += 1


class FailingLPIPS:
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise OSError("mock missing LPIPS weights")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, lengths: list[int], prefix: str) -> list[dict]:
    records = []
    for sequence_index, length in enumerate(lengths):
        sequence = f"{prefix}_run_{sequence_index:03d}"
        blurry = [f"assets/{sequence}_blur_{index:03d}.png" for index in range(length)]
        sharp = [f"assets/{sequence}_sharp_{index:03d}.png" for index in range(length)]
        records.append({"sequence": sequence, "blurry": blurry, "sharp": sharp})
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records


def _frame(
    *,
    manifest: Path,
    record: dict,
    frame_index: int,
    include_lpips: bool,
    history: int,
    causal_psnr: float,
) -> dict:
    sequence = record["sequence"]
    evssm = {"psnr": 30.0, "ssim": 0.9000, "l1": 0.0200}
    causal = {"psnr": causal_psnr, "ssim": 0.9010, "l1": 0.0196}
    repeat = {"psnr": causal_psnr - 0.1, "ssim": 0.9005, "l1": 0.0198}
    if include_lpips:
        evssm["lpips"] = 0.100
        causal["lpips"] = 0.090
        repeat["lpips"] = 0.095
    temporal = None
    if frame_index > 0:
        temporal = {
            "evssm": {
                "adjacent_change_l1": 0.030,
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
        }
    return {
        "sequence": sequence,
        "frame_index": frame_index,
        "history_stage": (
            "prefix" if frame_index < history - 1 else "steady_state"
        ),
        "blurry_path": str((manifest.parent / record["blurry"][frame_index]).resolve()),
        "sharp_path": str((manifest.parent / record["sharp"][frame_index]).resolve()),
        "evssm": evssm,
        "causal": causal,
        "causal_repeat_current": repeat,
        "runtime_gate_proxy": {
            "blurry_laplacian_variance": 1.0,
            "evssm_laplacian_variance": 1.1,
            "causal_laplacian_variance": 1.12,
            "causal_vs_evssm_gain": (1.12 - 1.1) / 1.1,
            "causal_vs_blurry_gain": (1.12 - 1.0) / 1.0,
            "passes_default_gate": True,
        },
        "temporal": temporal,
    }


def _write_report(
    path: Path,
    *,
    manifest: Path,
    records: list[dict],
    checkpoint: Path,
    include_lpips: bool,
    history: int = 3,
    causal_psnr: float = 30.2,
) -> dict:
    frames = [
        _frame(
            manifest=manifest,
            record=record,
            frame_index=frame_index,
            include_lpips=include_lpips,
            history=history,
            causal_psnr=causal_psnr,
        )
        for record in records
        for frame_index in range(len(record["blurry"]))
    ]
    payload = {
        "schema": EVALUATOR_SCHEMA,
        "checkpoint": str(checkpoint.resolve()),
        "evaluated_artifact_sha256": _sha256(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "manifest": str(manifest.resolve()),
        "teacher_provenance": {
            "storage": "precomputed_png_rgb8",
            "teacher_domain": "evssm_restored_rgb_0_1",
            "evssm_checkpoint_sha256": EXPECTED_EVSSM_SHA256,
        },
        "input_domain": "evssm",
        "history": history,
        "lpips_computed": include_lpips,
        "lpips_protocol": {
            "implementation": "torchmetrics.image.lpip",
            "network": "alex",
            "normalize_input_0_1": True,
            "per_frame_state_reset": True,
        },
        "frame_count": len(frames),
        "temporal_pair_count": sum(frame["temporal"] is not None for frame in frames),
        "steady_psnr_db": sum(
            frame["causal"]["psnr"]
            for frame in frames
            if frame["history_stage"] == "steady_state"
        )
        / sum(frame["history_stage"] == "steady_state" for frame in frames),
        "steady_frame_count": sum(
            frame["history_stage"] == "steady_state" for frame in frames
        ),
        "frames": frames,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _fixture(root: Path, *, include_lpips: bool = True):
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "candidate.pt"
    checkpoint.write_bytes(b"content-bound-causal-checkpoint")
    history1_checkpoint = root / "history1_control.pt"
    history1_checkpoint.write_bytes(b"independent-content-bound-H1-control")
    temporal_manifest = root / "val_temporal.jsonl"
    room2_manifest = root / "test_room2.jsonl"
    temporal_records = _write_manifest(temporal_manifest, [4, 4], "room1")
    # Sixteen H=3 runs exercise the exact registered long-run total; all are
    # nondegraded, safely above the required ten.
    room2_records = _write_manifest(room2_manifest, [3] * 16, "room2")
    temporal_report = root / "temporal_metrics.json"
    history1_report = root / "history1_temporal_metrics.json"
    room2_report = root / "room2_metrics.json"
    temporal_payload = _write_report(
        temporal_report,
        manifest=temporal_manifest,
        records=temporal_records,
        checkpoint=checkpoint,
        include_lpips=False,
    )
    history1_payload = _write_report(
        history1_report,
        manifest=temporal_manifest,
        records=temporal_records,
        checkpoint=history1_checkpoint,
        include_lpips=False,
        history=1,
        causal_psnr=30.1,
    )
    room2_payload = _write_report(
        room2_report,
        manifest=room2_manifest,
        records=room2_records,
        checkpoint=checkpoint,
        include_lpips=include_lpips,
    )
    return {
        "checkpoint": checkpoint,
        "history1_checkpoint": history1_checkpoint,
        "temporal_manifest": temporal_manifest,
        "room2_manifest": room2_manifest,
        "temporal_report": temporal_report,
        "history1_report": history1_report,
        "room2_report": room2_report,
        "temporal_payload": temporal_payload,
        "history1_payload": history1_payload,
        "room2_payload": room2_payload,
    }


def _temporal_layer(fixture: dict) -> dict:
    return build_temporal_layer_report(
        temporal_report_path=fixture["temporal_report"],
        history1_report_path=fixture["history1_report"],
        temporal_manifest_path=fixture["temporal_manifest"],
        source_root_path=fixture["temporal_manifest"].parent,
        expected_temporal_manifest_sha256=_sha256(fixture["temporal_manifest"]),
    )


def _select(fixture: dict) -> dict:
    temporal = _temporal_layer(fixture)
    temporal_path = fixture["temporal_manifest"].parent / "accepted_layer1.json"
    write_temporal_layer_report(temporal_path, temporal, overwrite=True)
    _, _, room2_metadata = _load_source_manifest(
        fixture["room2_manifest"],
        expected_sha256=_sha256(fixture["room2_manifest"]),
        label="room2_test",
        source_root=fixture["room2_manifest"].parent,
    )
    bundle = build_room2_selection_bundle(
        temporal_layer_report_path=temporal_path,
        temporal_layer_report_sha256=_sha256(temporal_path),
        room2_report_path=fixture["room2_report"],
        room2_manifest_path=fixture["room2_manifest"],
        source_root_path=fixture["room2_manifest"].parent,
        expected_temporal_manifest_sha256=_sha256(
            fixture["temporal_manifest"]
        ),
        expected_room2_manifest_sha256=_sha256(fixture["room2_manifest"]),
        expected_room2_frame_identity_sha256=room2_metadata[
            "frame_identity_sha256"
        ],
        expected_room2_frame_count=room2_metadata["frame_count"],
    )
    bundle["temporal_layer_report"] = temporal
    return bundle


def test_optional_lpips_metric_uses_explicit_mock() -> None:
    prediction = torch.full((3, 12, 12), 0.75)
    target = torch.full((3, 12, 12), 0.50)
    without = _metrics(prediction, target)
    assert "lpips" not in without
    metric = FakeLPIPS()
    with_lpips = _metrics(prediction, target, metric)
    assert with_lpips["lpips"] == 0.25
    assert metric.calls == 1
    assert metric.resets == 1
    try:
        _lpips_value(FailingLPIPS(), prediction, target)
    except RuntimeError as error:
        assert "LPIPS evaluation failed" in str(error)
    else:
        raise AssertionError("LPIPS failure was silently accepted")

    row = {
        source: {"psnr": 30.0, "ssim": 0.9, "l1": 0.02}
        for source in ("blurry", "evssm", "causal", "causal_repeat_current")
    }
    row["evssm"]["lpips"] = 0.10
    row["causal"]["lpips"] = 0.09
    row["causal_repeat_current"]["lpips"] = 0.095
    breakdown = _quality_breakdown([row])
    assert "lpips" not in breakdown["mean"]["blurry"]
    assert abs(breakdown["causal_minus_evssm"]["lpips"] + 0.01) < 1.0e-12


def test_passing_reports_are_content_bound_and_atomically_written() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        bundle = _select(fixture)
        report = bundle["selection_report"]
        assert report["status"] == "eligible_for_export"
        assert report["eligible"] is True
        temporal = bundle["temporal_layer_report"]
        room2 = bundle["room2_layer_report"]
        assert temporal["eligible"] is True
        assert room2["eligible"] is True
        assert temporal["metrics"]["normal_vs_history1_psnr_delta_db"] > 0.09
        assert temporal["history1_control"]["history"] == 1
        assert temporal["history1_control"]["checkpoint_sha256"] == _sha256(
            fixture["history1_checkpoint"]
        )
        assert temporal["history1_control"]["h3_steady_psnr_db"] > temporal[
            "history1_control"
        ]["h1_steady_psnr_db"]
        assert room2["metrics"]["nondegraded_long_runs"] == 16
        assert report["checkpoint_sha256"] == _sha256(fixture["checkpoint"])
        assert all(
            item["passed"] for item in temporal["checks"].values()
        )
        assert all(
            item["passed"] for item in room2["checks"].values()
        )

        output = root / "selection" / "selection_report.json"
        paths = write_room2_selection_bundle(output, bundle, overwrite=False)
        assert json.loads(output.read_text())["eligible"] is True
        assert not list(output.parent.glob(f".{output.name}.*.tmp"))
        original_temporal_sha = exporter.TEMPORAL_VALIDATION_MANIFEST_SHA256
        original_room2_sha = exporter.ROOM2_ONE_SHOT_MANIFEST_SHA256
        original_room2_identity = exporter.ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256
        original_room2_count = exporter.ROOM2_ONE_SHOT_FRAME_COUNT
        original_room2_root = exporter.ROOM2_ONE_SHOT_SOURCE_ROOT
        exporter.TEMPORAL_VALIDATION_MANIFEST_SHA256 = _sha256(
            fixture["temporal_manifest"]
        )
        exporter.ROOM2_ONE_SHOT_MANIFEST_SHA256 = _sha256(
            fixture["room2_manifest"]
        )
        exporter.ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256 = room2[
            "source_manifest"
        ]["frame_identity_sha256"]
        exporter.ROOM2_ONE_SHOT_FRAME_COUNT = room2["source_manifest"][
            "frame_count"
        ]
        exporter.ROOM2_ONE_SHOT_SOURCE_ROOT = root
        try:
            accepted = exporter.validate_deployment_selection(
                output,
                fixture["checkpoint"],
                {"evssm_checkpoint_sha256": EXPECTED_EVSSM_SHA256},
            )
            assert accepted["eligible"] is True
        finally:
            exporter.TEMPORAL_VALIDATION_MANIFEST_SHA256 = original_temporal_sha
            exporter.ROOM2_ONE_SHOT_MANIFEST_SHA256 = original_room2_sha
            exporter.ROOM2_ONE_SHOT_FRAME_IDENTITY_SHA256 = original_room2_identity
            exporter.ROOM2_ONE_SHOT_FRAME_COUNT = original_room2_count
            exporter.ROOM2_ONE_SHOT_SOURCE_ROOT = original_room2_root
        assert (root / "accepted_layer1.json").is_file()
        assert paths["room2_layer_report"].is_file()
        try:
            write_room2_selection_bundle(output, bundle, overwrite=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("atomic writer overwrote without explicit permission")
        write_room2_selection_bundle(output, bundle, overwrite=True)


def test_layer2_missing_lpips_is_fail_closed_but_reportable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = _fixture(Path(directory), include_lpips=False)
        bundle = _select(fixture)
        report = bundle["selection_report"]
        assert report["status"] == "blocked_missing_metric"
        assert report["eligible"] is False
        layer2 = bundle["room2_layer_report"]
        assert layer2["checks"]["lpips_present"]["passed"] is False
        assert layer2["checks"]["lpips_delta"]["passed"] is False
        assert "summary.lpips_computed=true" in report["missing_metrics"]


def test_temporal_stage_never_opens_room2_and_room2_requires_layer1_sha() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        fixture["room2_report"].write_text("not JSON\n", encoding="utf-8")
        temporal = _temporal_layer(fixture)
        assert temporal["eligible"] is True
        temporal_path = root / "layer1.json"
        write_temporal_layer_report(temporal_path, temporal, overwrite=False)
        try:
            build_room2_selection_bundle(
                temporal_layer_report_path=temporal_path,
                temporal_layer_report_sha256="0" * 64,
                room2_report_path=root / "does_not_exist.json",
                room2_manifest_path=fixture["room2_manifest"],
                source_root_path=root,
                expected_temporal_manifest_sha256=_sha256(
                    fixture["temporal_manifest"]
                ),
                expected_room2_manifest_sha256=_sha256(
                    fixture["room2_manifest"]
                ),
                expected_room2_frame_identity_sha256="0" * 64,
                expected_room2_frame_count=48,
            )
        except ReportContractError as error:
            assert "temporal layer report SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("room2 stage opened room2 before validating Layer1")


def test_oracle_and_per_run_gates_are_recomputed_from_frames() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        payload = copy.deepcopy(fixture["temporal_payload"])
        first_sequence = payload["frames"][0]["sequence"]
        for frame in payload["frames"]:
            if frame["sequence"] == first_sequence:
                frame["causal"]["psnr"] = 29.0
                frame["causal"]["ssim"] = 0.88
                frame["causal"]["l1"] = 0.03
        fixture["temporal_report"].write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        layer1 = _temporal_layer(fixture)
        assert layer1["eligible"] is False
        assert layer1["checks"]["accepted_oracle_precision"]["passed"] is False
        assert layer1["checks"]["worst_run_psnr_delta_db"]["passed"] is False


def test_mismatched_checkpoint_and_manifest_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        other = root / "other.pt"
        other.write_bytes(b"other")
        payload = copy.deepcopy(fixture["room2_payload"])
        payload["checkpoint"] = str(other)
        payload["evaluated_artifact_sha256"] = _sha256(other)
        payload["source_checkpoint_sha256"] = _sha256(other)
        fixture["room2_report"].write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        try:
            _select(fixture)
        except ReportContractError as error:
            assert "different source checkpoints" in str(error)
        else:
            raise AssertionError("selector accepted reports for different checkpoints")

        try:
            build_temporal_layer_report(
                temporal_report_path=fixture["temporal_report"],
                history1_report_path=fixture["history1_report"],
                temporal_manifest_path=fixture["temporal_manifest"],
                source_root_path=fixture["temporal_manifest"].parent,
                expected_temporal_manifest_sha256="0" * 64,
            )
        except ReportContractError as error:
            assert "SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("selector accepted a drifted source manifest hash")


def test_lpips_flag_and_laplacian_gains_cannot_be_forged() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        payload = copy.deepcopy(fixture["room2_payload"])
        payload["lpips_computed"] = "true"
        fixture["room2_report"].write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        try:
            _select(fixture)
        except ReportContractError as error:
            assert "lpips_computed must be boolean" in str(error)
        else:
            raise AssertionError("selector accepted a truthy non-boolean LPIPS flag")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        payload = copy.deepcopy(fixture["temporal_payload"])
        payload["frames"][0]["runtime_gate_proxy"]["causal_vs_evssm_gain"] = 1.0
        fixture["temporal_report"].write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        try:
            _select(fixture)
        except ReportContractError as error:
            assert "disagree with Laplacian variances" in str(error)
        else:
            raise AssertionError("selector trusted a forged Laplacian gain")


def test_history1_is_independent_aligned_and_performance_gated() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        payload = copy.deepcopy(fixture["history1_payload"])
        payload["source_checkpoint_sha256"] = _sha256(fixture["checkpoint"])
        fixture["history1_report"].write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        try:
            _select(fixture)
        except ReportContractError as error:
            assert "different source checkpoints" in str(error)
        else:
            raise AssertionError("selector accepted H3 itself as the H1 control")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _fixture(root)
        payload = copy.deepcopy(fixture["history1_payload"])
        for frame in payload["frames"]:
            frame["causal"]["psnr"] = 30.19
        fixture["history1_report"].write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        temporal = _temporal_layer(fixture)
        assert temporal["eligible"] is False
        assert temporal["checks"]["normal_vs_history1_psnr_delta_db"][
            "passed"
        ] is False


def main() -> None:
    test_optional_lpips_metric_uses_explicit_mock()
    test_passing_reports_are_content_bound_and_atomically_written()
    test_layer2_missing_lpips_is_fail_closed_but_reportable()
    test_temporal_stage_never_opens_room2_and_room2_requires_layer1_sha()
    test_oracle_and_per_run_gates_are_recomputed_from_frames()
    test_mismatched_checkpoint_and_manifest_are_rejected()
    test_lpips_flag_and_laplacian_gains_cannot_be_forged()
    test_history1_is_independent_aligned_and_performance_gated()
    print("select_causal_video_deblur=PASS")


if __name__ == "__main__":
    main()
