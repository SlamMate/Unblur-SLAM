#!/usr/bin/env python3
"""CPU-only tests for role-aware FrameCrafter context construction."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framecrafter_context import (  # noqa: E402
    ContextFrameMetadata,
    ContextSelectionConfig,
    EVSSMImageCandidate,
    EVSSMResolver,
    audit_evssm_local_degradation,
    select_framecrafter_contexts,
)
from src.framecrafter_pipeline import (  # noqa: E402
    FrameRecord,
    TargetView,
    interpolate_c2w,
)


def _pose(position: int) -> np.ndarray:
    angle = math.radians(1.5 * position)
    cosine, sine = math.cos(angle), math.sin(angle)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ]
    pose[0, 3] = 0.025 * position
    return pose


def _make_problem(tmp_path: Path):
    intrinsics = np.array(
        [[40.0, 0.0, 7.5], [0.0, 40.0, 5.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    frames = []
    for index in range(10):
        path = tmp_path / f"rgb_{index:02d}.png"
        Image.new("RGB", (16, 12), color=(10 * index, 20, 30)).save(path)
        frames.append(
            FrameRecord(
                source_index=index,
                frame_id=path.name,
                timestamp=float(index),
                rgb_path=path,
                c2w=_pose(index),
                intrinsics=intrinsics,
                sharpness=float(index + 1),
            )
        )
    target = TargetView(
        target_id="between_04_05",
        left_index=4,
        right_index=5,
        left_position=4,
        right_position=5,
        timestamp=4.5,
        alpha=0.5,
        c2w=interpolate_c2w(frames[4].c2w, frames[5].c2w, 0.5),
        intrinsics=intrinsics,
        reasons=("low_frustum_overlap",),
    )
    # Explicit blur labels make the intended role contract independent of raw
    # Laplacian scale.  Frames 1/8 are the best overlapping sharp guides.
    blur = {2, 3, 4, 5, 6, 7}
    overlap = {0: 0.10, 1: 0.82, 2: 0.75, 3: 0.91, 4: 1.0,
               5: 1.0, 6: 0.90, 7: 0.74, 8: 0.85, 9: 0.12}
    reliability = {index: 0.95 for index in range(10)}
    reliability[9] = 0.20
    metadata = [
        ContextFrameMetadata(
            frame=frame,
            position=index,
            overlap=overlap[index],
            reliability=reliability[index],
            sharpness=100.0 if index in (0, 1, 8, 9) else 10.0,
            is_blurry=index in blur,
            pairwise_overlap={4: 0.25 + 0.02 * abs(index - 4)},
        )
        for index, frame in enumerate(frames)
    ]
    return frames, target, metadata


def test_role_balancing_budget_dedup_and_determinism(tmp_path: Path) -> None:
    _, target, metadata = _make_problem(tmp_path)
    config = ContextSelectionConfig(
        context_budget=6,
        local_blurry_count=2,
        sharp_context_count=2,
        local_radius=3,
        min_sharp_overlap=0.40,
        seed=1729,
    )
    result = select_framecrafter_contexts(target, metadata, config)
    assert result.source_indices == (1, 3, 4, 5, 6, 8)
    assert [item.role for item in result.contexts] == [
        "sharp_before",
        "local_blurry_before",
        "endpoint_left",
        "endpoint_right",
        "local_blurry_after",
        "sharp_after",
    ]
    assert len(set(result.source_indices)) == config.context_budget
    assert all(item.provenance.resolved_mode == "raw" for item in result.contexts)
    required_components = {
        "overlap",
        "sharpness",
        "reliability",
        "view_diversity",
        "redundancy",
        "locality",
        "total",
    }
    assert all(set(item.score.as_dict()) == required_components for item in result.contexts)

    # Input ordering and an inferior duplicate record cannot change selection.
    duplicate = ContextFrameMetadata(
        frame=metadata[3].frame,
        position=3,
        overlap=0.20,
        reliability=0.10,
        sharpness=1.0,
        is_blurry=True,
    )
    reordered = select_framecrafter_contexts(
        target, list(reversed(metadata)) + [duplicate], config
    )
    assert reordered.source_indices == result.source_indices
    assert [item.role for item in reordered.contexts] == [
        item.role for item in result.contexts
    ]
    assert [item.score.as_dict() for item in reordered.contexts] == [
        item.score.as_dict() for item in result.contexts
    ]


def test_precomputed_evssm_gate_and_explicit_raw_fallback(tmp_path: Path) -> None:
    _, target, metadata = _make_problem(tmp_path)
    evssm_root = tmp_path / "evssm"
    evssm_root.mkdir()
    for item in metadata:
        Image.new("RGB", (16, 12), color=(200, item.position, 10)).save(
            evssm_root / item.frame.rgb_path.name
        )
        item.evssm_confidence = 0.95
        item.evssm_sharpness_gain = 1.20
        item.evssm_consistency = 0.94
    # The right endpoint exists but fails only the sharpness-improvement gate.
    metadata[5].evssm_sharpness_gain = 0.92
    resolver = EVSSMResolver(
        precomputed_root=evssm_root,
        path_template="{name}",
    )
    config = ContextSelectionConfig(
        image_mode="evssm",
        seed=4,
        evssm_min_confidence=0.8,
        evssm_min_sharpness_gain=1.05,
        evssm_min_consistency=0.9,
        evssm_local_gate_enabled=False,
    )
    result = select_framecrafter_contexts(target, metadata, config, resolver)
    by_index = {item.frame.source_index: item for item in result.contexts}
    assert by_index[4].image_path == (evssm_root / "rgb_04.png").resolve()
    assert by_index[4].provenance.resolved_mode == "evssm"
    assert by_index[4].provenance.provider == "precomputed_root"
    assert by_index[5].image_path == metadata[5].frame.rgb_path
    assert by_index[5].provenance.resolved_mode == "raw"
    assert by_index[5].provenance.provider == "raw_fallback"
    assert by_index[5].provenance.fallback_reason == "evssm_sharpness_gain"
    assert by_index[5].provenance.evssm_sharpness_gain == 0.92
    assert all(path.rgb_path == selected.image_path for path, selected in zip(
        result.frame_records, result.contexts
    ))


def test_hybrid_roles_keep_sharp_guides_raw(tmp_path: Path) -> None:
    _, target, metadata = _make_problem(tmp_path)
    callback_root = tmp_path / "callback_outputs"
    callback_root.mkdir()
    calls: list[int] = []

    def callback(item: ContextFrameMetadata):
        calls.append(item.frame.source_index)
        output = callback_root / item.frame.rgb_path.name
        Image.new("RGB", (16, 12), color=(240, 240, 240)).save(output)
        return EVSSMImageCandidate(
            path=output,
            confidence=0.96,
            sharpness_gain=1.3,
            consistency=0.92,
            provider="test_callback",
        )

    result = select_framecrafter_contexts(
        target,
        metadata,
        ContextSelectionConfig(
            image_mode="hybrid",
            seed=99,
            hybrid_evssm_roles=("local_blurry_before", "local_blurry_after"),
            evssm_local_gate_enabled=False,
        ),
        EVSSMResolver(callback=callback),
    )
    assert calls == [3, 6]
    by_role = {item.role: item for item in result.contexts}
    assert by_role["sharp_before"].provenance.resolved_mode == "raw"
    assert by_role["sharp_after"].provenance.resolved_mode == "raw"
    assert by_role["sharp_before"].provenance.provider == "hybrid_raw_role"
    assert by_role["endpoint_left"].provenance.resolved_mode == "raw"
    assert by_role["endpoint_left"].provenance.provider == "hybrid_raw_role"
    assert by_role["local_blurry_before"].provenance.resolved_mode == "evssm"
    assert by_role["local_blurry_after"].provenance.resolved_mode == "evssm"
    assert by_role["local_blurry_after"].provenance.provider == "test_callback"
    assert ContextSelectionConfig().hybrid_evssm_roles == ("local_blurry_inside",)

    # Raw mode must not even consult a configured callback.
    calls.clear()
    raw = select_framecrafter_contexts(
        target,
        metadata,
        ContextSelectionConfig(image_mode="raw", seed=99),
        EVSSMResolver(callback=callback),
    )
    assert calls == []
    assert all(item.image_path == item.frame.rgb_path for item in raw.contexts)


def test_local_evssm_tile_gate_falls_back_entire_frame(tmp_path: Path) -> None:
    _, target, metadata = _make_problem(tmp_path)
    yy, xx = np.mgrid[:64, :64]
    raw_array = np.full((64, 64, 3), 210, dtype=np.uint8)
    raw_array[((xx // 4 + yy // 4) % 2) == 0] = 150
    raw_path = metadata[4].frame.rgb_path
    Image.fromarray(raw_array, mode="RGB").save(raw_path)
    good_path = tmp_path / "evssm_good.png"
    Image.fromarray(raw_array, mode="RGB").save(good_path)
    good = audit_evssm_local_degradation(
        raw_path, good_path, ContextSelectionConfig()
    )
    assert good.passed
    assert good.tile_count == 9

    bad_array = raw_array.copy()
    bad_array[16:48, 16:48] = 20
    bad_path = tmp_path / "evssm_bad.png"
    Image.fromarray(bad_array, mode="RGB").save(bad_path)
    resolver = EVSSMResolver(
        precomputed={
            4: EVSSMImageCandidate(
                path=bad_path,
                confidence=0.99,
                sharpness_gain=1.20,
                consistency=0.99,
            )
        }
    )
    result = select_framecrafter_contexts(
        target,
        metadata,
        ContextSelectionConfig(
            image_mode="hybrid",
            hybrid_evssm_roles=("endpoint_left",),
        ),
        resolver,
    )
    endpoint = next(item for item in result.contexts if item.role == "endpoint_left")
    assert endpoint.image_path == raw_path
    assert endpoint.provenance.provider == "raw_fallback"
    assert "evssm_local_brightness_drop" in endpoint.provenance.fallback_reason
    audit = endpoint.provenance.evssm_local_gate
    assert audit is not None and not audit.passed
    assert audit.metrics["tile_mae"].failed
    payload = audit.as_dict()
    assert payload["schema"] == "unblur_slam.evssm_local_gate.v1"
    assert payload["metrics"]["dark_expansion"]["failed"] is True


def test_real_phone_fixture_local_gate_regression(tmp_path: Path) -> None:
    fixture_path = Path(
        os.environ.get(
            "UNBLUR_SLAM_PHONE_FIXTURE_METADATA",
            "/srv/szha0669/unblur-slam/framecrafter_evssm_context/"
            "fr2_2282_2358_final_pnp_sharp3/metadata.json",
        )
    )
    if not fixture_path.is_file():
        print("framecrafter_context_real_phone_fixture=SKIP")
        return
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    records = {int(record["source_index"]): record for record in payload["frames"]}
    decisions = {}
    for source_index in (2250, 2282, 2320, 2352, 2358, 2387):
        record = records[source_index]
        decision = audit_evssm_local_degradation(
            record["raw_path"], record["output_path"], ContextSelectionConfig()
        )
        decisions[source_index] = decision.passed
    assert decisions == {
        2250: False,
        2282: False,
        2320: False,
        2352: True,
        2358: True,
        2387: True,
    }


def test_missing_or_bad_evssm_is_never_silent(tmp_path: Path) -> None:
    _, target, metadata = _make_problem(tmp_path)
    no_resolver = select_framecrafter_contexts(
        target,
        metadata,
        ContextSelectionConfig(image_mode="evssm"),
    )
    assert all(item.provenance.resolved_mode == "raw" for item in no_resolver.contexts)
    assert all(
        item.provenance.fallback_reason == "evssm_resolver_not_configured"
        for item in no_resolver.contexts
    )

    missing = tmp_path / "does_not_exist.png"
    metadata[4].evssm_path = missing
    metadata[4].evssm_confidence = 1.0
    metadata[4].evssm_sharpness_gain = 1.2
    metadata[4].evssm_consistency = 1.0
    with_missing = select_framecrafter_contexts(
        target,
        metadata,
        ContextSelectionConfig(image_mode="evssm"),
        EVSSMResolver(),
    )
    endpoint = next(item for item in with_missing.contexts if item.role == "endpoint_left")
    assert endpoint.provenance.resolved_mode == "raw"
    assert endpoint.provenance.fallback_reason == "missing_evssm_file"


def main() -> None:
    tests = (
        test_role_balancing_budget_dedup_and_determinism,
        test_precomputed_evssm_gate_and_explicit_raw_fallback,
        test_hybrid_roles_keep_sharp_guides_raw,
        test_local_evssm_tile_gate_falls_back_entire_frame,
        test_real_phone_fixture_local_gate_regression,
        test_missing_or_bad_evssm_is_never_silent,
    )
    for test in tests:
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
    print("framecrafter_context=PASS")


if __name__ == "__main__":
    main()
