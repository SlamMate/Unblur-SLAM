#!/usr/bin/env python3
"""CPU contract tests for ReSplat-inspired replay and submap interfaces."""

import csv
import math
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.refinement.resplat_replay import ReplayConfig, ResidualReplaySampler
from src.submaps import (
    IDENTITY4,
    RigidCorrection,
    SubmapBoundaryPolicy,
    SubmapRecord,
    apply_rigid_correction,
)


def translated_pose(x=0.0, y=0.0, z=0.0, yaw_deg=0.0):
    angle = math.radians(yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        (cosine, -sine, 0.0, x),
        (sine, cosine, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _check_replay_sampler(root):
    log_path = root / "replay.csv"
    state_path = root / "replay.json"
    config = ReplayConfig(ema_alpha=0.5, uniform_probability=0.10)
    sampler = ResidualReplaySampler(range(4000), config=config, seed=7, log_path=log_path)

    first = sampler.observe(
        17, residual=0.2, laplacian_gap=0.4, coverage=0.1, novelty=0.8, step=10
    )
    second = sampler.observe(
        17, residual=0.6, laplacian_gap=0.2, coverage=0.5, novelty=0.4, step=11
    )
    state = sampler.state_for(17)
    assert abs(state.residual - 0.4) < 1e-12
    assert abs(state.laplacian_gap - 0.3) < 1e-12
    assert abs(state.coverage - 0.3) < 1e-12
    assert abs(state.novelty - 0.6) < 1e-12
    assert first > 0.0 and second > 0.0

    selected = sampler.sample_many(128, step=12)
    assert len(selected) == 128
    assert len(set(selected)) == 128

    # A pure uniform floor ignores priority but remains deterministic and can
    # revisit every member; a pure priority path strongly favors hard frames.
    uniform = ResidualReplaySampler(
        ["easy", "hard"],
        config=ReplayConfig(uniform_probability=1.0),
        seed=2,
    )
    uniform_counts = {"easy": 0, "hard": 0}
    for _ in range(200):
        uniform_counts[uniform.sample()] += 1
    assert uniform_counts["easy"] > 50 and uniform_counts["hard"] > 50

    prioritized = ResidualReplaySampler(
        ["easy", "hard"],
        config=ReplayConfig(
            uniform_probability=0.0,
            coverage_gap_weight=0.0,
            novelty_weight=0.0,
        ),
        seed=3,
    )
    prioritized.observe("easy", residual=0.001, laplacian_gap=0.0, coverage=1.0, novelty=0.0)
    prioritized.observe("hard", residual=10.0, laplacian_gap=1.0, coverage=0.0, novelty=1.0)
    hard_count = sum(prioritized.sample() == "hard" for _ in range(500))
    assert hard_count > 450

    # Online mapping keeps statistics for views that can move in and out of
    # the current window.  A currently ineligible, very hard view must never
    # leak into the two background replacements.
    active = ResidualReplaySampler(
        [10, 20, 30, 99],
        config=ReplayConfig(uniform_probability=0.0),
        seed=5,
    )
    active.observe(10, residual=0.2)
    active.observe(20, residual=0.3)
    active.observe(30, residual=0.4)
    active.observe(99, residual=100.0)
    selected_active = active.sample_many_from([10, 20, 30], 2, step=4)
    assert len(selected_active) == 2
    assert len(set(selected_active)) == 2
    assert set(selected_active) <= {10, 20, 30}
    assert active.state_for(99).visits == 0
    assert active.sample_many_from([20], 2, step=5) == [20]
    try:
        active.sample_many_from([404], 1)
    except KeyError:
        pass
    else:
        raise AssertionError("unregistered active replay frame was accepted")

    sampler.save_state(state_path)
    expected_continuation = sampler.sample_many(40, step=20)
    restored = ResidualReplaySampler.load_state(state_path)
    actual_continuation = restored.sample_many(40, step=20)
    assert actual_continuation == expected_continuation
    assert restored.state_for(17).residual == sampler.state_for(17).residual

    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["event"] for row in rows} == {"observe", "sample"}
    assert rows[-1]["priority"]


def _check_submap_contract(root):
    policy = SubmapBoundaryPolicy(
        min_keyframes=3,
        max_keyframes=8,
        translation_threshold=0.5,
        rotation_threshold_deg=30.0,
    )
    held = policy.decide(IDENTITY4, translated_pose(x=0.8), keyframe_count=2)
    assert not held.start_new_submap
    moved = policy.decide(IDENTITY4, translated_pose(x=0.8), keyframe_count=3)
    assert moved.start_new_submap and "translation" in moved.reasons
    rotated = policy.decide(IDENTITY4, translated_pose(yaw_deg=40.0), keyframe_count=3)
    assert rotated.start_new_submap and "rotation" in rotated.reasons
    full = policy.decide(IDENTITY4, IDENTITY4, keyframe_count=8)
    assert full.start_new_submap and full.reasons == ("max_keyframes",)

    record = SubmapRecord(2, 100, IDENTITY4)
    record.add_frame(101)
    record.add_frame(102, is_keyframe=True)
    record.close("submaps/000002.ckpt", gaussian_count=1234, metadata={"note": "contract-only"})

    called = []

    def apply_native_gaussians(current_record, transform):
        called.append((current_record.submap_id, transform))

    correction = RigidCorrection(2, translated_pose(x=1.25), source="verified_external_edge", confidence=0.9)
    updated = apply_rigid_correction(record, correction, gaussian_applier=apply_native_gaussians)
    assert called and called[0][0] == 2
    assert abs(updated[0][3] - 1.25) < 1e-12

    metadata_path = root / "submap.json"
    record.save_checkpoint_metadata(metadata_path)
    payload = metadata_path.read_text(encoding="utf-8")
    assert '"registration_implemented": false' in payload
    assert '"gaussian_count": 1234' in payload

    try:
        # Scale is not silently accepted as a rigid LoopSplat correction.
        RigidCorrection(2, ((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 1)))
    except ValueError:
        pass
    else:
        raise AssertionError("non-rigid correction was accepted")


def test_replay_sampler(tmp_path):
    _check_replay_sampler(tmp_path)


def test_submap_contract(tmp_path):
    _check_submap_contract(tmp_path)


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _check_replay_sampler(root)
        _check_submap_contract(root)
    print("resplat_inspired_replay_and_submaps=PASS")


if __name__ == "__main__":
    main()
