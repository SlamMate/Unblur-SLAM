#!/usr/bin/env python3
"""CPU contracts for integration helpers that otherwise run inside Mapper."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mapper import (
    Mapper,
    _device_slerp,
    _motion_knots_are_optimizable,
    _match_virtual_render_resolution,
    _offline_pose_is_frozen,
    _select_online_replay_background,
)
from src.refinement.resplat_replay import ReplayConfig, ResidualReplaySampler
from src.refinement.official_resplat_sidecar import SidecarConfig, load_snapshot
from src.submaps import SubmapRecord
from src.utils.eval_frames import (
    available_clear_gt_source_indices,
    clear_gt_source_indices,
)
import thirdparty.glorie_slam.motion_filter as motion_filter_module
from thirdparty.glorie_slam.motion_filter import MotionFilter
from thirdparty.monogs.utils.camera_utils import Camera


class FakeFrameReader:
    def __init__(self, metadata):
        self.metadata = metadata

    def frame_info(self, index):
        return self.metadata[int(index)]


class FakeAugmentedStream(FakeFrameReader):
    def __getitem__(self, index):
        metadata = self.metadata[int(index)]
        depth = metadata.get("depth")
        return int(index), None, depth, None, None


class FakeHydrationVideo:
    def __init__(self, timestamps, invalid_ids=()):
        self.timestamp = torch.tensor(timestamps, dtype=torch.float32)
        self.counter = SimpleNamespace(value=len(timestamps))
        self.invalid_ids = {int(value) for value in invalid_ids}

    def get_depth_and_pose(self, video_idx, device):
        video_idx = int(video_idx)
        depth = torch.full((12, 12), float(video_idx + 1), device=device)
        valid = torch.ones((12, 12), dtype=torch.bool, device=device)
        if video_idx in self.invalid_ids:
            valid.zero_()
        c2w = torch.eye(4, device=device)
        c2w[0, 3] = float(video_idx)
        return depth, valid, c2w


class FakeHydrationReader:
    fx = 5.0
    fy = 5.0
    cx = 4.0
    cy = 3.0
    W_out = 10
    H_out = 8
    fovx = 1.0
    fovy = 1.0
    device = "cpu"

    def __init__(self, length=1000, synthetic_ids=()):
        self.length = int(length)
        self.synthetic_ids = {int(value) for value in synthetic_ids}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # The sentinel objects model TUM GT depth/pose. Hydration may unpack
        # them but must never tensorize, invert, or pass either to Camera.
        forbidden_gt_depth = object()
        forbidden_gt_pose = object()
        color = torch.full((1, 3, self.H_out, self.W_out), 0.25)
        return int(index), color, forbidden_gt_depth, forbidden_gt_pose, None

    def frame_info(self, index):
        index = int(index)
        synthetic = index in self.synthetic_ids
        return {
            "synthetic": synthetic,
            "fixed_pose": synthetic,
            "source_index": None if synthetic else index,
            # Timestamp 777 deliberately is not an evaluation frame. It must
            # still be promoted because selection is from DepthVideo, not GT.
            "eval": index != 777,
            "confidence": 1.0,
        }


class FakePrinter:
    def __init__(self):
        self.messages = []

    def print(self, message, *_args, **_kwargs):
        self.messages.append(str(message))


def _hydration_mapper(tmpdir, *, invalid_ids=(), synthetic_ids=()):
    mapper = Mapper.__new__(Mapper)
    mapper.config = {
        "mapping": {"hydrate_missing_droid_keyframes": True},
        "tracking": {"backend": {"final_ba": True}},
        "framecrafter": {
            "supervision_weight_min": 0.10,
            "supervision_weight_max": 0.30,
        },
        "composite_blur": False,
        "n_virtual_cams": 5,
        "interpolation": "linear",
    }
    mapper.device = "cpu"
    mapper.save_dir = str(tmpdir)
    mapper.video = FakeHydrationVideo(
        [0, 9, 15, 49, 777], invalid_ids=invalid_ids
    )
    mapper.frame_reader = FakeHydrationReader(synthetic_ids=synthetic_ids)
    camera_15 = SimpleNamespace(uid=2, timestamp=15)
    camera_49 = SimpleNamespace(uid=3, timestamp=49)
    mapper.cameras = {2: camera_15, 3: camera_49}
    mapper.video_idxs = [2, 3]
    mapper.keyframe_idxs = [15, 49]
    mapper.is_kf = {2: True, 3: True}
    mapper.kf2mapper_idx = {15: 2, 49: 3}
    mapper.initial_frame_uid = 2
    mapper.printer = FakePrinter()
    mapper._load_mono_depth_for_timestamp = lambda timestamp: torch.full(
        (12, 12), float(int(timestamp) + 1)
    )
    get_calls = []

    def get_w2c_and_depth(video_idx, timestamp, mono_depth, init=False):
        del mono_depth, init
        get_calls.append((int(video_idx), int(timestamp)))
        depth, valid, c2w = mapper.video.get_depth_and_pose(video_idx, "cpu")
        return depth, torch.linalg.inv(c2w), int(valid.sum()) < 100

    mapper.get_w2c_and_depth = get_w2c_and_depth
    return mapper, (camera_15, camera_49), get_calls


def _fake_hydrated_camera(_dataset, data, _projection_matrix):
    assert "gt_pose" not in data
    pose = data["glorie_pose"]
    viewpoint = SimpleNamespace(
        uid=int(data["idx"]),
        timestamp=None,
        R_gt=pose[:3, :3].clone(),
        T_gt=pose[:3, 3].clone(),
        original_image=data["gt_color"],
        depth=data["glorie_depth"],
        deblur_fail=False,
    )

    def update_rt(rotation, translation):
        viewpoint.R = rotation.clone()
        viewpoint.T = translation.clone()

    viewpoint.update_RT = update_rt
    viewpoint.compute_grad_mask = lambda _config: None
    return viewpoint


def _fake_hydrated_motion_camera(
    _dataset, data, _projection_matrix, deblur_fail
):
    assert deblur_fail is True
    assert "gt_pose" not in data
    pose = data["glorie_pose"]
    pose_sequence = data["estimated_pose_sequence"]
    assert pose_sequence.ndim == 3
    assert bool(
        torch.allclose(
            pose_sequence,
            pose.unsqueeze(0).expand_as(pose_sequence),
        )
    )
    knot_count = int(pose_sequence.shape[0])
    viewpoint = SimpleNamespace(
        uid=int(data["idx"]),
        timestamp=None,
        R_gt=pose[:3, :3].clone(),
        T_gt=pose[:3, 3].clone(),
        original_image=data["gt_color"],
        depth=data["glorie_depth"],
        deblur_fail=True,
        n_virtual_cams=int(data["n_virtual_cams"]),
        interpolation=str(data["interpolation"]),
        num_control_knots=knot_count,
        R_i=torch.zeros(knot_count, 3, 3),
        t_i=torch.zeros(knot_count, 3),
        T_i_rot_delta=[
            torch.nn.Parameter(torch.zeros(3)) for _ in range(knot_count)
        ],
        T_i_trans_delta=[
            torch.nn.Parameter(torch.zeros(3)) for _ in range(knot_count)
        ],
        cam_rot_delta=torch.nn.Parameter(torch.zeros(3)),
        cam_trans_delta=torch.nn.Parameter(torch.zeros(3)),
    )

    def update_rt(rotation, translation):
        viewpoint.R = rotation.clone()
        viewpoint.T = translation.clone()

    def update_rt_motion(rotation, translation, knot):
        viewpoint.R_i[int(knot)] = rotation.clone()
        viewpoint.t_i[int(knot)] = translation.clone()

    viewpoint.update_RT = update_rt
    viewpoint.update_RT_motion = update_rt_motion
    viewpoint.compute_grad_mask = lambda _config: None
    return viewpoint


def test_exact_26k_budget() -> None:
    total, replay_start, mode = Mapper._resolve_refinement_budget(
        26000,
        True,
        {"budget_mode": "replace_tail", "extra_iters": 4000},
    )
    assert (total, replay_start, mode) == (26000, 22000, "replace_tail")
    assert Mapper._resolve_refinement_budget(
        26000, True, {"budget_mode": "extend", "extra_iters": 4000}
    )[:2] == (30000, 26000)


def test_non_module_gaussian_checkpoint() -> None:
    mapper = Mapper.__new__(Mapper)
    gaussian = SimpleNamespace(
        _xyz=torch.randn(2, 3, requires_grad=True),
        _features_dc=torch.randn(2, 1, 3, requires_grad=True),
        _features_rest=torch.empty(2, 0, 3, requires_grad=True),
        _scaling=torch.randn(2, 3, requires_grad=True),
        _rotation=torch.randn(2, 4, requires_grad=True),
        _opacity=torch.randn(2, 1, requires_grad=True),
        max_radii2D=torch.ones(2),
        unique_kfIDs=torch.tensor([0, 1]),
        n_obs=torch.tensor([2, 3]),
        active_sh_degree=0,
        max_sh_degree=0,
        spatial_lr_scale=6.0,
        mlp_rgb_ms=torch.nn.Conv2d(3, 3, 1),
        mlp_rgb_ss=torch.nn.Conv2d(3, 3, 1),
    )
    mapper.gaussians = gaussian
    state = mapper._gaussian_inference_state()
    assert state["_xyz"].device.type == "cpu"
    assert state["resume_exact"] is False
    assert "weight" in state["mlp_rgb_ms"]
    assert not hasattr(gaussian, "state_dict")


def test_generated_pose_reanchoring() -> None:
    mapper = Mapper.__new__(Mapper)
    mapper.config = {"framecrafter": {"align_manifest_pose_online": True}}
    anchor = SimpleNamespace(
        synthetic=False,
        source_frame_index=0,
        timestamp=0,
        R=torch.eye(3),
        # w2c translation -2 means current c2w camera centre +2 on x.
        T=torch.tensor([-2.0, 0.0, 0.0]),
    )
    mapper.cameras = {0: anchor}
    mapper.frame_reader = FakeFrameReader(
        {0: {"c2w": np.eye(4).tolist(), "source_index": 0}}
    )
    target = torch.eye(4)
    target[0, 3] = 1.0
    aligned = mapper._align_generated_c2w(target, {"left_index": 0})
    assert torch.allclose(aligned[:3, 3], torch.tensor([3.0, 0.0, 0.0]))


def test_paper_clear_gt_indices_are_not_tracking_anchors() -> None:
    mapper = Mapper.__new__(Mapper)
    expected_counts = {
        "freiburg1_desk": 14,
        "freiburg2_xyz": 42,
        "freiburg3_office": 57,
    }
    for scene, expected in expected_counts.items():
        mapper.config = {"dataset": "tumrgbd", "scene": scene}
        assert len(mapper._clear_gt_source_indices()) == expected


def test_mono_depth_cache_uses_original_source_indices() -> None:
    stream = FakeAugmentedStream(
        {
            0: {"synthetic": False, "source_index": 0},
            1: {
                "synthetic": True,
                "source_index": None,
                "depth": np.full((2, 3), 7.0, dtype=np.float32),
            },
            2: {"synthetic": False, "source_index": 1},
        }
    )
    motion_filter = MotionFilter.__new__(MotionFilter)
    motion_filter.cfg = {
        "mono_prior": {"predict_online": True},
        "data": {"output": "/unused"},
        "scene": "unused",
    }
    motion_filter.mono_depth_estimator = object()
    motion_filter.device = "cpu"

    calls = []
    original_predict = motion_filter_module.predict_mono_depth

    def fake_predict(estimator, cache_index, image, cfg, device):
        calls.append(int(cache_index))
        return torch.full((2, 3), float(cache_index))

    motion_filter_module.predict_mono_depth = fake_predict
    try:
        image = torch.zeros(1, 3, 2, 3)
        first = motion_filter._mono_depth_for_frame(0, image, stream)
        generated = motion_filter._mono_depth_for_frame(1, image, stream)
        second = motion_filter._mono_depth_for_frame(2, image, stream)
    finally:
        motion_filter_module.predict_mono_depth = original_predict

    assert calls == [0, 1]
    assert torch.all(first == 0)
    assert torch.all(generated == 7)
    assert torch.all(second == 1)


def test_hold_protocol_uses_dataset_root_and_never_falls_back_to_all(tmp_root=None) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene = root / "dataset" / "motion" / "blurball"
        scene.mkdir(parents=True)
        reader = SimpleNamespace(original_frame_count=15)
        config = {
            "dataset": "deblur_nerf_motion",
            "scene": "blurball",
            "data": {"dataset_root": str(root / "dataset"), "input_folder": "motion"},
        }
        try:
            clear_gt_source_indices(config, reader)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing hold=X must not evaluate every frame")
        (scene / "hold=7").touch()
        assert clear_gt_source_indices(config, reader) == {0, 7, 14}
        assert available_clear_gt_source_indices(config, reader) == {0, 7, 14}


def test_pose_extrapolation_slerp_is_device_independent() -> None:
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    half_angle = torch.tensor(torch.pi / 8.0)
    q1 = torch.tensor(
        [torch.cos(half_angle), 0.0, 0.0, torch.sin(half_angle)]
    )
    extrapolated = _device_slerp(2.0, q0, q1)
    expected = torch.tensor(
        [torch.cos(2.0 * half_angle), 0.0, 0.0, torch.sin(2.0 * half_angle)]
    )
    assert extrapolated.device.type == "cpu"
    assert torch.allclose(extrapolated, expected, atol=1e-5)


def test_online_replay_replaces_two_background_views_and_observes_on_cpu() -> None:
    candidates = [
        SimpleNamespace(uid=10),
        SimpleNamespace(uid=20),
        SimpleNamespace(uid=30),
    ]
    sampler = ResidualReplaySampler(
        [99], config=ReplayConfig(uniform_probability=0.0), seed=11
    )
    # UID 99 represents a high-priority view that has moved into the current
    # window; it must not leak into the active background selection.
    sampler.observe(99, residual=100.0)
    selected = _select_online_replay_background(
        candidates, sampler, step=7, view_count=2
    )
    assert len(selected) == 2
    assert len({viewpoint.uid for _, viewpoint in selected}) == 2
    assert all(candidates[index] is viewpoint for index, viewpoint in selected)
    assert all(viewpoint.uid != 99 for _, viewpoint in selected)
    assert sampler.state_for(99).visits == 0

    mapper = Mapper.__new__(Mapper)
    observed = SimpleNamespace(
        uid=selected[0][1].uid,
        original_image=torch.zeros(3, 4, 4),
        supervision_weight=0.25,
    )
    prediction = torch.full((3, 4, 4), 0.5)
    opacity = torch.tensor([[[0.25, 0.75], [0.25, 0.75]]])
    mapper._observe_replay(sampler, observed, prediction, opacity, step=8)
    state = sampler.state_for(observed.uid)
    assert state.observations == 1
    assert abs(state.residual - 0.5) < 1e-6
    assert abs(state.coverage - 0.5) < 1e-6
    assert abs(state.novelty - 0.5) < 1e-6
    assert abs(state.reliability - 0.25) < 1e-6

    try:
        _select_online_replay_background(
            candidates, sampler, step=9, view_count=3
        )
    except ValueError:
        pass
    else:
        raise AssertionError("online replay changed the two-view mapping budget")


def test_virtual_render_resolution_contract_preserves_gradients() -> None:
    rgb = torch.randn(3, 384, 512, requires_grad=True)
    depth = torch.randn(1, 384, 512, requires_grad=True)
    scaled_rgb, scaled_depth = _match_virtual_render_resolution(
        rgb, depth, (96, 128)
    )
    assert scaled_rgb.shape == (3, 96, 128)
    assert scaled_depth.shape == (1, 96, 128)
    (scaled_rgb.square().mean() + scaled_depth.square().mean()).backward()
    assert rgb.grad is not None and bool(torch.isfinite(rgb.grad).all())
    assert depth.grad is not None and bool(torch.isfinite(depth.grad).all())
    assert bool((rgb.grad != 0).any())
    assert bool((depth.grad != 0).any())

    # Match the real mapper boundary: differentiable resized tensors are
    # written into preallocated multi-view accumulators before averaging.
    rgb_for_buffer = torch.randn(3, 384, 512, requires_grad=True)
    depth_for_buffer = torch.randn(1, 384, 512, requires_grad=True)
    buffered_rgb, buffered_depth = _match_virtual_render_resolution(
        rgb_for_buffer, depth_for_buffer, (96, 128)
    )
    rgb_accumulator = torch.empty(1, 3, 96, 128)
    depth_accumulator = torch.empty(1, 1, 96, 128)
    rgb_accumulator[0] = buffered_rgb
    depth_accumulator[0] = buffered_depth
    (rgb_accumulator.mean() + depth_accumulator.mean()).backward()
    assert rgb_for_buffer.grad is not None
    assert depth_for_buffer.grad is not None

    matched_rgb = torch.randn(3, 96, 128, requires_grad=True)
    matched_depth = torch.randn(1, 96, 128, requires_grad=True)
    same_rgb, same_depth = _match_virtual_render_resolution(
        matched_rgb, matched_depth, (96, 128)
    )
    assert same_rgb is matched_rgb
    assert same_depth is matched_depth
    (same_rgb.mean() + same_depth.mean()).backward()
    assert matched_rgb.grad is not None
    assert matched_depth.grad is not None


def test_final_ba_hydration_promotes_all_missing_droid_keyframes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        mapper, existing_cameras, get_calls = _hydration_mapper(directory)
        with mock.patch.object(
            Camera, "init_from_dataset", side_effect=_fake_hydrated_camera
        ) as camera_factory:
            hydrated = mapper._hydrate_missing_droid_keyframes_for_final_refine()

    # IDs 0/1 are the invalid-at-warmup frames from fr2. ID 4/timestamp 777
    # proves hydration is not secretly driven by the clear-GT protocol.
    assert hydrated == [0, 1, 4]
    assert get_calls == [(0, 0), (1, 9), (4, 777)]
    assert camera_factory.call_count == 3
    assert mapper.video_idxs == [0, 1, 2, 3, 4]
    assert mapper.keyframe_idxs == [0, 9, 15, 49, 777]
    assert list(mapper.cameras) == [0, 1, 2, 3, 4]
    assert mapper.cameras[2] is existing_cameras[0]
    assert mapper.cameras[3] is existing_cameras[1]
    assert mapper.initial_frame_uid == 0
    for video_idx in hydrated:
        viewpoint = mapper.cameras[video_idx]
        assert viewpoint.hydrated_after_final_ba is True
        assert viewpoint.pose_provenance == "droid_final_ba"
        assert viewpoint.depth_provenance == "droid_final_ba_with_mono_completion"
        assert mapper.is_kf[video_idx] is False
        expected_w2c = torch.eye(4)
        expected_w2c[0, 3] = -float(video_idx)
        actual_w2c = torch.eye(4)
        actual_w2c[:3, :3] = viewpoint.R
        actual_w2c[:3, 3] = viewpoint.T
        assert torch.allclose(actual_w2c, expected_w2c)
    assert any("online map unchanged" in item for item in mapper.printer.messages)


def test_final_ba_hydration_fails_closed_before_camera_mutation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        mapper, existing_cameras, _ = _hydration_mapper(
            directory, invalid_ids={1}
        )
        try:
            with mock.patch.object(
                Camera, "init_from_dataset", side_effect=_fake_hydrated_camera
            ) as camera_factory:
                mapper._hydrate_missing_droid_keyframes_for_final_refine()
        except RuntimeError as error:
            assert "valid_depth_pixels=0" in str(error)
        else:
            raise AssertionError("invalid final DROID depth was promoted")

    assert camera_factory.call_count == 0
    assert mapper.video_idxs == [2, 3]
    assert mapper.keyframe_idxs == [15, 49]
    assert mapper.cameras == {2: existing_cameras[0], 3: existing_cameras[1]}


def test_composite_hydration_uses_estimated_motion_camera_and_freezes_anchor() -> None:
    with tempfile.TemporaryDirectory() as directory:
        mapper, _, _ = _hydration_mapper(directory)
        mapper.config["composite_blur"] = True
        with mock.patch.object(
            Camera,
            "init_from_dataset_motion",
            side_effect=_fake_hydrated_motion_camera,
        ) as motion_factory, mock.patch.object(
            Camera,
            "init_from_dataset",
            side_effect=AssertionError("composite hydration used a plain Camera"),
        ):
            hydrated = mapper._hydrate_missing_droid_keyframes_for_final_refine()

    assert hydrated == [0, 1, 4]
    assert motion_factory.call_count == 3
    for call in motion_factory.call_args_list:
        data = call.args[1]
        assert "gt_pose" not in data
        assert data["estimated_pose_sequence"].shape == (2, 4, 4)
        assert data["n_virtual_cams"] == 5
        assert data["interpolation"] == "linear"

    anchor = mapper.cameras[0]
    assert anchor.deblur_fail is True
    assert anchor.offline_gauge_anchor is True
    assert anchor.motion_pose_seed_provenance == "repeated_droid_final_ba_w2c"
    assert _offline_pose_is_frozen(anchor)
    assert not _motion_knots_are_optimizable(anchor)
    for parameter in (
        [anchor.cam_rot_delta, anchor.cam_trans_delta]
        + anchor.T_i_rot_delta
        + anchor.T_i_trans_delta
    ):
        assert parameter.requires_grad is False

    non_anchor = mapper.cameras[1]
    assert non_anchor.deblur_fail is True
    assert non_anchor.offline_gauge_anchor is False
    assert not _offline_pose_is_frozen(non_anchor)
    assert _motion_knots_are_optimizable(non_anchor)
    expected_w2c = torch.eye(4)
    expected_w2c[0, 3] = -1.0
    assert torch.allclose(non_anchor.R, expected_w2c[:3, :3])
    assert torch.allclose(non_anchor.T, expected_w2c[:3, 3])
    for knot in range(non_anchor.num_control_knots):
        assert torch.allclose(non_anchor.R_i[knot], expected_w2c[:3, :3])
        assert torch.allclose(non_anchor.t_i[knot], expected_w2c[:3, 3])


def test_motion_camera_factory_needs_no_gt_pose_field() -> None:
    reader = FakeHydrationReader()
    w2c = torch.eye(4)
    w2c[0, 3] = -2.0
    estimated_sequence = w2c.unsqueeze(0).repeat(2, 1, 1)
    data = {
        "idx": 7,
        "gt_color": torch.zeros(3, reader.H_out, reader.W_out),
        "glorie_depth": np.ones((reader.H_out, reader.W_out), np.float32),
        "glorie_pose": w2c,
        "n_virtual_cams": 5,
        "interpolation": "linear",
        "estimated_pose_sequence": estimated_sequence,
    }
    assert "gt_pose" not in data
    viewpoint = Camera.init_from_dataset_motion(
        reader, data, torch.eye(4), deblur_fail=True
    )
    assert isinstance(viewpoint, Camera)
    assert viewpoint.deblur_fail is True
    assert viewpoint.num_control_knots == 2
    assert viewpoint.n_virtual_cams == 5
    for knot in range(viewpoint.num_control_knots):
        assert torch.allclose(viewpoint.R_gt_motion[knot], w2c[:3, :3])
        assert torch.allclose(viewpoint.T_gt_motion[knot], w2c[:3, 3])


def test_hydration_rejects_missing_synthetic_manifest_camera() -> None:
    with tempfile.TemporaryDirectory() as directory:
        mapper, existing_cameras, _ = _hydration_mapper(
            directory, synthetic_ids={9}
        )
        try:
            with mock.patch.object(
                Camera, "init_from_dataset", side_effect=_fake_hydrated_camera
            ) as camera_factory:
                mapper._hydrate_missing_droid_keyframes_for_final_refine()
        except RuntimeError as error:
            assert "synthetic/fixed-manifest" in str(error)
        else:
            raise AssertionError("synthetic manifest camera used a DROID pose")

    assert camera_factory.call_count == 0
    assert mapper.video_idxs == [2, 3]
    assert mapper.cameras == {2: existing_cameras[0], 3: existing_cameras[1]}


def test_mapper_closed_submap_enqueues_native_sidecar_without_map_mutation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        class FakeQueue:
            def __init__(self):
                self.root = root / "official_sidecars"
                self.submitted = []

            def submit(self, snapshot_dir):
                self.submitted.append(Path(snapshot_dir))
                snapshot = load_snapshot(snapshot_dir)
                return {"event": "submitted", "snapshot_id": snapshot["snapshot_id"]}

        mapper = Mapper.__new__(Mapper)
        mapper.save_dir = str(root / "scene")
        mapper.iteration_count = 123
        mapper.official_resplat_sidecar_cfg = SidecarConfig(
            output_root=str(root / "official_sidecars")
        )
        mapper.official_resplat_sidecar = FakeQueue()
        mapper.official_resplat_sidecar_events = []
        mapper.submap_keyframe_ordinals = {index: index for index in range(8)}
        mapper.cameras = {}
        for index in range(8):
            mapper.cameras[index] = SimpleNamespace(
                uid=index,
                R=torch.eye(3),
                T=torch.tensor([-0.01 * index, 0.0, 0.0]),
                original_image=torch.full((3, 12, 16), index / 10.0),
                fx=10.0,
                fy=11.0,
                cx=8.0,
                cy=6.0,
            )
        record = SubmapRecord(
            submap_id=0,
            anchor_frame_id=0,
            anchor_c2w=np.eye(4).tolist(),
        )
        for index in range(1, 8):
            record.add_frame(index, is_keyframe=True)
        record.closed = True
        state = mapper._enqueue_official_resplat_sidecar(record, closure_ordinal=7)

        assert state["status"] == "submitted"
        assert state["active_map_merge_performed"] is False
        assert len(mapper.official_resplat_sidecar.submitted) == 1
        snapshot = load_snapshot(mapper.official_resplat_sidecar.submitted[0])
        assert [frame["frame_id"] for frame in snapshot["frames"]] == list(range(8))
        assert snapshot["pose_revision"] == 123
        assert snapshot["active_map_state_included"] is False


def main() -> None:
    test_exact_26k_budget()
    test_non_module_gaussian_checkpoint()
    test_generated_pose_reanchoring()
    test_paper_clear_gt_indices_are_not_tracking_anchors()
    test_mono_depth_cache_uses_original_source_indices()
    test_hold_protocol_uses_dataset_root_and_never_falls_back_to_all()
    test_pose_extrapolation_slerp_is_device_independent()
    test_online_replay_replaces_two_background_views_and_observes_on_cpu()
    test_virtual_render_resolution_contract_preserves_gradients()
    test_final_ba_hydration_promotes_all_missing_droid_keyframes()
    test_final_ba_hydration_fails_closed_before_camera_mutation()
    test_composite_hydration_uses_estimated_motion_camera_and_freezes_anchor()
    test_motion_camera_factory_needs_no_gt_pose_field()
    test_hydration_rejects_missing_synthetic_manifest_camera()
    test_mapper_closed_submap_enqueues_native_sidecar_without_map_mutation()
    print("mapper_extension_contracts=PASS")


if __name__ == "__main__":
    main()
