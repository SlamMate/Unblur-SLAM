# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import hashlib
import json
import math
from contextlib import contextmanager
from functools import lru_cache
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import random
import psutil
import time
from tqdm import tqdm
from thirdparty.monogs.utils.pose_utils import get_new_RT, slerp

from colorama import Fore, Style
from multiprocessing.connection import Connection
from munch import munchify
import numpy as np

from src.utils.datasets import get_dataset, load_mono_depth
from src.utils.eval_frames import (
    available_clear_gt_source_indices,
    clear_gt_metric_scope,
    clear_gt_source_indices,
)
from src.utils.common import as_intrinsics_matrix, setup_seed

from src.utils.Printer import Printer, FontColor

from thirdparty.glorie_slam.depth_video import DepthVideo
from thirdparty.gaussian_splatting.gaussian_renderer import render, render_virtual
from thirdparty.gaussian_splatting.utils.general_utils import rotation_matrix_to_quaternion, quaternion_multiply
from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss, ssim
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from thirdparty.monogs.utils.pose_utils import update_pose
from thirdparty.monogs.utils.slam_utils import get_loss_mapping, get_median_depth, BAD_mapping_loss, BAD_tracking_loss, render_video
from thirdparty.monogs.utils.camera_utils import Camera
from src.utils.motion_and_defocus_blur_mlp import CompositeBlurModel
from src.utils.common import load_tensor
from pathlib import Path
from thirdparty.monogs.utils.pose_utils import update_pose, update_pose_knot, get_next_traj, slerp, compute_pose_error, get_next_traj_from_dspo

from thirdparty.monogs.utils.rotation_conv import quaternion_to_matrix, matrix_to_quaternion
from thirdparty.gaussian_splatting.gaussian_renderer import render_virtual
from thirdparty.monogs.utils.pose_utils import slerp

from src.utils.vis_kernel import visualize_kernel_weights
from src.refinement.resplat_replay import (
    ONLINE_REPLAY_VIEW_COUNT,
    ReplayConfig,
    ResidualReplaySampler,
    validate_resplat_config,
)
from src.submaps import SubmapBoundaryPolicy, SubmapRecord
from src.refinement.official_resplat_sidecar import (
    OfficialReSplatSidecarQueue,
    SidecarConfig as OfficialReSplatSidecarConfig,
    SidecarFrameInput as OfficialReSplatSidecarFrame,
    load_snapshot as load_official_resplat_snapshot,
    materialize_closed_submap_snapshot,
    verify_unblur_world_artifact,
)
from src.refinement.official_resplat_active_fusion import (
    ACTIVE_FUSION_AUDIT_SCHEMA,
    ActiveFusionConfig as OfficialReSplatActiveFusionConfig,
    atomic_write_json as atomic_write_active_fusion_json,
    stamp_contract_sha256,
    forced_commit_count_contract,
    context_reconstruction_gate as evaluate_resplat_context_reconstruction,
    postmerge_reconstruction_gate as evaluate_postmerge_reconstruction,
)
from copy import deepcopy


def _select_online_replay_background(
    viewpoints,
    sampler,
    *,
    step,
    view_count=ONLINE_REPLAY_VIEW_COUNT,
):
    """Replace the baseline's two background draws with replay-priority draws.

    The returned integer is the viewpoint's index in the original background
    stack.  Preserving it matters because legacy mapping uses ``cam_idx == 0``
    when deciding whether a view has already been seen.
    """

    validate_resplat_config(
        {
            "enabled": True,
            "backend": "residual_replay",
            "online_enabled": True,
            "online_replay_views": view_count,
        }
    )
    by_uid = {}
    for stack_index, viewpoint in enumerate(viewpoints):
        uid = int(viewpoint.uid)
        if uid in by_uid:
            raise ValueError(f"duplicate online replay viewpoint uid {uid}")
        by_uid[uid] = (stack_index, viewpoint)
        sampler.register(uid)
    count = min(ONLINE_REPLAY_VIEW_COUNT, len(by_uid))
    selected_uids = sampler.sample_many_from(
        by_uid.keys(), count, step=int(step)
    )
    return [by_uid[int(uid)] for uid in selected_uids]


def _match_virtual_render_resolution(image, depth, target_size):
    """Match one virtual RGB-D render to its multi-view accumulation scale.

    ``CompositeBlurModel`` returns the current multi-scale resolution for
    blurry views, while a sharp/deblur-fallback view bypasses that model and
    remains at the renderer's full resolution.  BAD accumulation is defined at
    the current multi-scale resolution in both cases.  Resize RGB with the same
    bilinear convention used by the blur model and depth with nearest-neighbor;
    both operations remain differentiable with respect to their inputs.
    """

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("virtual RGB render must have shape [3,H,W]")
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError("virtual depth render must have shape [1,H,W]")
    if len(target_size) != 2:
        raise ValueError("virtual render target_size must be (height, width)")
    target_height, target_width = (int(target_size[0]), int(target_size[1]))
    if target_height < 1 or target_width < 1:
        raise ValueError("virtual render target dimensions must be positive")
    target = (target_height, target_width)
    if tuple(image.shape[-2:]) != target:
        image = F.interpolate(
            image.unsqueeze(0),
            size=target,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if tuple(depth.shape[-2:]) != target:
        depth = F.interpolate(
            depth.unsqueeze(0), size=target, mode="nearest"
        ).squeeze(0)
    return image, depth


def _offline_pose_is_frozen(viewpoint):
    """Whether offline refinement must leave a camera pose untouched."""

    return bool(getattr(viewpoint, "fixed_pose", False)) or bool(
        getattr(viewpoint, "offline_gauge_anchor", False)
    )


def _motion_knots_are_optimizable(viewpoint):
    """Select motion-pose parameters without moving the offline gauge anchor."""

    return bool(getattr(viewpoint, "deblur_fail", False)) and not (
        _offline_pose_is_frozen(viewpoint)
    )


def _freeze_offline_gauge_anchor_pose(viewpoint):
    """Disable gradients for every main/sub-frame pose delta on the anchor."""

    if not bool(getattr(viewpoint, "offline_gauge_anchor", False)):
        return viewpoint
    for name in ("cam_rot_delta", "cam_trans_delta"):
        parameter = getattr(viewpoint, name, None)
        if parameter is not None:
            parameter.requires_grad_(False)
    for name in ("T_i_rot_delta", "T_i_trans_delta"):
        for parameter in getattr(viewpoint, name, ()):
            parameter.requires_grad_(False)
    return viewpoint


def _motion_control_knot_count(interpolation):
    interpolation = str(interpolation).lower()
    if interpolation == "linear":
        return 2
    if interpolation == "cubic":
        return 4
    raise ValueError(
        "hydrated motion camera interpolation must be 'linear' or 'cubic'"
    )


def _device_slerp(t, q0, q1, dot_threshold=0.9995):
    """Shortest-path quaternion SLERP/extrapolation on the input device."""

    q0 = F.normalize(q0.float(), dim=-1)
    q1 = F.normalize(q1.float(), dim=-1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    t = torch.as_tensor(t, dtype=q0.dtype, device=q0.device)
    while t.ndim < q0.ndim:
        t = t.unsqueeze(-1)
    linear = F.normalize(torch.lerp(q0, q1, t), dim=-1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    safe_denominator = sin_theta.clamp_min(torch.finfo(q0.dtype).eps)
    spherical = (
        torch.sin((1.0 - t) * theta) / safe_denominator * q0
        + torch.sin(t * theta) / safe_denominator * q1
    )
    result = torch.where(dot.abs() > dot_threshold, linear, spherical)
    return F.normalize(result, dim=-1)


class Mapper(object):
    """
    Mapper thread.

    """
    def __init__(self, slam, pipe:Connection):
        # setup seed
        setup_seed(slam.cfg["setup_seed"])
        torch.autograd.set_detect_anomaly(True)
        self.slam = slam

        self.config = slam.cfg
        self.resplat_cfg = validate_resplat_config(
            self.config.get("mapping", {}).get("resplat", {}) or {}
        )
        self.printer:Printer = slam.printer
        if self.config['only_tracking']:
            return
        self.pipe = pipe
        self.verbose = slam.verbose

        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None

        self.dtype = torch.float32
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.cameras = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None
      
        self.video:DepthVideo = slam.video

        model_params = munchify(self.config["mapping"]["model_params"])
        opt_params = munchify(self.config["mapping"]["opt_params"])
        pipeline_params = munchify(self.config["mapping"]["pipeline_params"])
        self.use_spherical_harmonics = self.config["mapping"]["Training"]["spherical_harmonics"]
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)

        mip_cfg = self.config.get("mapping", {}).get("mip_splatting", {}) or {}
        if not isinstance(mip_cfg, dict):
            raise TypeError("mapping.mip_splatting must be a mapping")
        self.mip_splatting_enabled = bool(mip_cfg.get("enabled", False))
        self.mip_filter_kernel_variance = float(
            mip_cfg.get("filter_kernel_variance", 0.2)
        )
        if self.mip_splatting_enabled and (
            not math.isfinite(self.mip_filter_kernel_variance)
            or self.mip_filter_kernel_variance <= 0.0
        ):
            raise ValueError(
                "mapping.mip_splatting.filter_kernel_variance must be positive"
            )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.cameras_extent = 6.0

        self.set_hyperparams()

        self.device = torch.device(self.config['device'])
       
        self.frame_reader = get_dataset(
            self.config, device=self.device)

        self.global_optimiz_video = []
        self.global_panoramic = []
        self.global_frame_idx = []

        self.deblur_fail_kfs = set()  # 只在mapper中存在的关键帧

        self.initial_frame_uid = None
        self.mapper_idx = 0
        self.kf2mapper_idx = {}
        self.fake_sharp = self.config.get("fake_sharp", False)
        self.deblur_kf2video = {}  # {mapper_frame_id: original_video_idx}
        self.refinement_checkpoint_metrics = []
        self.online_replay_sampler = None
        self.submap_cfg = self.config.get("submaps", {}) or {}
        self.submap_policy = None
        self.submap_records = []
        self.active_submap = None
        self.submap_keyframe_ordinals = {}
        self.submap_keyframe_ordinal = -1
        self.official_resplat_sidecar_events = []
        sidecar_raw_cfg = self.submap_cfg.get("official_resplat_sidecar", {}) or {}
        self.official_resplat_sidecar_cfg = OfficialReSplatSidecarConfig.from_dict(
            sidecar_raw_cfg,
            default_output_root=Path(self.save_dir)
            / "submaps"
            / "official_resplat_sidecars",
        )
        self.official_resplat_sidecar = None
        if self.official_resplat_sidecar_cfg.enabled:
            if not self.submap_cfg.get("enabled", False):
                raise ValueError(
                    "submaps.official_resplat_sidecar.enabled=true requires "
                    "submaps.enabled=true"
                )
            self.official_resplat_sidecar = OfficialReSplatSidecarQueue(
                self.official_resplat_sidecar_cfg
            )

        # One-shot synchronous official-ReSplat state3 experiment.  This is
        # deliberately independent of asynchronous partition-only sidecars:
        # it blocks after the eighth *fully mapped* fixed keyframe so the
        # subprocess, gates, merge, and possible rollback are all charged to
        # Unblur-SLAM's online timer.
        active_fusion_raw = (
            self.config.get("mapping", {}).get(
                "official_resplat_active_fusion", {}
            )
            or {}
        )
        self.official_resplat_active_fusion_cfg = (
            OfficialReSplatActiveFusionConfig.from_dict(
                active_fusion_raw,
                default_output_root=Path(self.save_dir)
                / "official_resplat_active_fusion",
            )
        )
        self.official_resplat_active_fusion_queue = None
        self.official_resplat_active_fusion_mapped_keyframes = []
        self.official_resplat_active_fusion_attempts = 0
        self.official_resplat_active_fusion_audit = None
        self.official_resplat_forced_commit_chain = None
        if self.official_resplat_active_fusion_cfg.enabled:
            fusion = self.official_resplat_active_fusion_cfg
            geometry = dict(fusion.geometry_gate)
            self.official_resplat_active_fusion_queue = OfficialReSplatSidecarQueue(
                OfficialReSplatSidecarConfig(
                    enabled=True,
                    mode="sidecar_only",
                    context_keyframes=fusion.trigger_keyframe_count,
                    queue_capacity=1,
                    output_root=fusion.output_root,
                    python_executable=fusion.python_executable,
                    runner_script=fusion.runner_script,
                    resplat_repo=fusion.resplat_repo,
                    checkpoint=fusion.checkpoint,
                    expected_checkpoint_sha256=fusion.expected_checkpoint_sha256,
                    model_preset=fusion.model_preset,
                    refinement_updates=fusion.refinement_updates,
                    cuda_visible_devices=fusion.cuda_visible_devices,
                    process_device=fusion.process_device,
                    near=fusion.near,
                    far=fusion.far,
                    max_runtime_seconds=fusion.max_runtime_seconds,
                    final_drain_timeout_seconds=min(
                        60.0, fusion.max_runtime_seconds
                    ),
                    max_pose_revision_lag=fusion.max_pose_revision_lag,
                    max_pose_translation_drift=fusion.max_pose_translation_drift,
                    max_pose_rotation_drift_deg=fusion.max_pose_rotation_drift_deg,
                    min_gaussian_count=int(
                        geometry["expected_gaussian_count"]
                    ),
                    max_gaussian_count=int(
                        geometry["expected_gaussian_count"]
                    ),
                    min_finite_fraction=float(
                        geometry["min_finite_fraction"]
                    ),
                    max_p95_distance=float(
                        geometry["max_p95_distance_from_pivot"]
                    ),
                    max_distance=float(
                        geometry["max_distance_from_pivot"]
                    ),
                    max_p95_scale=float(geometry["max_p95_scale"]),
                    max_scale=float(geometry["max_scale"]),
                    max_quaternion_norm_deviation=float(
                        geometry["max_quaternion_norm_deviation"]
                    ),
                    active_map_merge=False,
                )
            )
        if self.submap_cfg.get("enabled", False):
            if str(self.submap_cfg.get("loop_backend", "none")) != "none":
                raise ValueError(
                    "Submap partitioning is implemented, but loop registration/PGO "
                    "is not. Set submaps.loop_backend=none until a verified backend "
                    "supplies corrections."
                )
            self.submap_policy = SubmapBoundaryPolicy(
                min_keyframes=int(self.submap_cfg.get("min_keyframes", 8)),
                max_keyframes=int(self.submap_cfg.get("max_keyframes", 80)),
                translation_threshold=float(
                    self.submap_cfg.get("translation_threshold", 1.5)
                ),
                rotation_threshold_deg=float(
                    self.submap_cfg.get("rotation_threshold_deg", 45.0)
                ),
            )
        
    def set_pipe(self, pipe):
        self.pipe = pipe

    def _observation_weight(self, viewpoint):
        """Return the confidence gate for an original or generated view."""
        return float(getattr(viewpoint, "supervision_weight", 1.0))

    def _refresh_mip_splatting(self, cameras):
        """Refresh the non-optimizable 3D sampling filter for live cameras."""

        if not self.mip_splatting_enabled:
            return
        cameras = tuple(cameras)
        if not cameras:
            raise RuntimeError("Mip-Splatting cannot run without mapping cameras")
        self.gaussians.configure_mip_splatting(
            cameras,
            enabled=True,
            kernel_variance=self.mip_filter_kernel_variance,
            refresh=True,
        )

    def _annotate_augmented_viewpoint(self, viewpoint, dataset_index):
        if not hasattr(self.frame_reader, "frame_info"):
            viewpoint.synthetic = False
            viewpoint.eval_frame = True
            viewpoint.fixed_pose = False
            viewpoint.source_frame_index = int(dataset_index)
            viewpoint.framecrafter_acceptance_class = "original"
            viewpoint.supervision_weight = 1.0
            return viewpoint

        metadata = self.frame_reader.frame_info(int(dataset_index))
        synthetic = bool(metadata.get("synthetic", False))
        confidence = float(metadata.get("confidence", 1.0))
        framecrafter_cfg = self.config.get("framecrafter", {}) or {}
        minimum = float(framecrafter_cfg.get("supervision_weight_min", 0.10))
        maximum = float(framecrafter_cfg.get("supervision_weight_max", 0.30))
        if not 0.0 <= minimum <= maximum <= 1.0:
            raise ValueError(
                "FrameCrafter supervision weights must satisfy "
                "0 <= min <= max <= 1"
            )
        viewpoint.synthetic = synthetic
        viewpoint.eval_frame = bool(metadata.get("eval", not synthetic))
        viewpoint.fixed_pose = bool(metadata.get("fixed_pose", False))
        viewpoint.source_frame_index = int(metadata.get("source_index", dataset_index))
        viewpoint.framecrafter_confidence = confidence
        acceptance_class = str(metadata.get("acceptance_class", "sharp_accepted"))
        if acceptance_class not in {"sharp_accepted", "geometry_only"}:
            raise ValueError(
                f"unknown FrameCrafter acceptance class {acceptance_class!r}"
            )
        viewpoint.framecrafter_acceptance_class = acceptance_class
        supervision_weight = (
            minimum + confidence * (maximum - minimum) if synthetic else 1.0
        )
        if synthetic and acceptance_class == "geometry_only":
            geometry_scale = float(
                framecrafter_cfg.get("geometry_only_weight_scale", 0.50)
            )
            if not 0.0 <= geometry_scale <= 1.0:
                raise ValueError(
                    "framecrafter.geometry_only_weight_scale must be in [0,1]"
                )
            supervision_weight *= geometry_scale
        viewpoint.supervision_weight = supervision_weight
        if synthetic:
            # A generated observation is already the sharp target and must not
            # enter the motion-blur image-formation branch.
            viewpoint.is_blurry = False
        return viewpoint

    def _make_replay_sampler(self, frame_ids, scope):
        cfg = self.resplat_cfg
        ema_decay = float(cfg.get("ema_decay", 0.90))
        replay_config = ReplayConfig(
            ema_alpha=1.0 - ema_decay,
            uniform_probability=float(cfg.get("uniform_floor", 0.20)),
            residual_weight=float(cfg.get("residual_weight", 1.0)),
            laplacian_gap_weight=float(cfg.get("laplacian_weight", 0.25)),
            coverage_gap_weight=float(cfg.get("coverage_weight", 0.20)),
            novelty_weight=float(cfg.get("novelty_weight", 0.10)),
            residual_scale=float(cfg.get("residual_scale", 0.10)),
            laplacian_gap_scale=float(cfg.get("laplacian_gap_scale", 0.10)),
            min_priority=float(cfg.get("min_priority", 1e-6)),
        )
        log_path = None
        if cfg.get("log_csv", False):
            log_path = Path(self.save_dir) / f"resplat_replay_{scope}.csv"
        return ResidualReplaySampler(
            frame_ids,
            config=replay_config,
            seed=int(self.config.get("setup_seed", 43)),
            log_path=log_path,
        )

    def _load_mono_depth_for_timestamp(self, dataset_index):
        """Load cached depth by original index after manifest insertion."""
        metadata = self.frame_reader.frame_info(int(dataset_index))
        if bool(metadata.get("synthetic", False)):
            _, _, depth, _, _ = self.frame_reader[int(dataset_index)]
            if depth is None:
                raise ValueError(
                    f"Synthetic frame {dataset_index} has no gated RGB-D depth"
                )
            return torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        source_index = int(metadata.get("source_index", dataset_index))
        return load_mono_depth(source_index, self.save_dir).to(self.device)

    def _hydrate_missing_droid_keyframes_for_final_refine(self):
        """Promote every missing DROID keyframe after final global BA.

        DROID's warm-up frames can have an empty two-view-valid depth mask when
        Mapper first receives them.  Mapper must not seed the online Gaussian
        map from an unscaled relative monocular prior, so those frames retain
        the legacy online skip behaviour.  Final global BA can subsequently
        make their DROID depth valid.  Immediately before offline refinement,
        enumerate the *entire* DepthVideo (not an evaluation-frame list) and
        build training-only cameras for every DROID keyframe that was not
        mapped online.

        ``get_w2c_and_depth`` supplies the final DROID pose and uses the normal
        DROID-valid-mask/Omnidata completion path shared by already-mapped
        cameras.  A still-invalid DROID keyframe fails closed; GT RGB-D and GT
        pose are never passed to the camera constructor.
        """

        if not bool(
            self.config.get("mapping", {}).get(
                "hydrate_missing_droid_keyframes", False
            )
        ):
            self.final_refine_hydrated_droid_ids = []
            return []
        if not bool(
            self.config.get("tracking", {})
            .get("backend", {})
            .get("final_ba", False)
        ):
            raise RuntimeError(
                "mapping.hydrate_missing_droid_keyframes requires final global BA"
            )

        video_count = int(self.video.counter.value)
        existing_pairs = list(zip(self.video_idxs, self.keyframe_idxs))
        if len(existing_pairs) != len(self.video_idxs) or len(existing_pairs) != len(
            self.keyframe_idxs
        ):
            raise RuntimeError("Mapper keyframe/video index lists are inconsistent")

        mapped_droid_ids = []
        for video_idx, _ in existing_pairs:
            video_idx = int(video_idx)
            if video_idx < 0:
                continue
            if video_idx >= video_count:
                raise RuntimeError(
                    f"mapped DROID keyframe id {video_idx} exceeds video count "
                    f"{video_count}"
                )
            mapped_droid_ids.append(video_idx)
        if len(mapped_droid_ids) != len(set(mapped_droid_ids)):
            raise RuntimeError("duplicate mapped DROID keyframe ids before hydration")

        missing_ids = sorted(set(range(video_count)) - set(mapped_droid_ids))
        records = []
        timestamps = {}
        for video_idx in range(video_count):
            timestamp = int(self.video.timestamp[video_idx].item())
            if timestamp in timestamps:
                raise RuntimeError(
                    "duplicate DROID timestamps cannot be promoted safely: "
                    f"timestamp={timestamp}, video_ids="
                    f"{timestamps[timestamp]},{video_idx}"
                )
            timestamps[timestamp] = video_idx
            if video_idx not in missing_ids:
                continue
            if timestamp < 0 or timestamp >= len(self.frame_reader):
                raise RuntimeError(
                    f"DROID keyframe {video_idx} has out-of-range timestamp {timestamp}"
                )
            metadata = self.frame_reader.frame_info(timestamp)
            if bool(metadata.get("synthetic", False)) or bool(
                metadata.get("fixed_pose", False)
            ):
                raise RuntimeError(
                    "refusing to hydrate a synthetic/fixed-manifest DROID "
                    "observation with a tracking pose: "
                    f"video_idx={video_idx}, timestamp={timestamp}"
                )

            # Check the final DROID validity explicitly before the standard
            # mono-completion helper.  This keeps promotion fail-closed and
            # makes the >=100-pixel contract independent of clear-GT labels.
            _, valid_depth_mask, _ = self.video.get_depth_and_pose(
                video_idx, self.device
            )
            valid_count = int(valid_depth_mask.sum().item())
            if valid_count < 100:
                raise RuntimeError(
                    "final DROID keyframe remains invalid after global BA; "
                    "refusing relative-mono-only promotion: "
                    f"video_idx={video_idx}, timestamp={timestamp}, "
                    f"valid_depth_pixels={valid_count}"
                )

            mono_depth = self._load_mono_depth_for_timestamp(timestamp)
            depth, w2c, invalid = self.get_w2c_and_depth(
                video_idx, timestamp, mono_depth, init=False
            )
            if invalid:
                raise RuntimeError(
                    "DROID hydration validity changed unexpectedly: "
                    f"video_idx={video_idx}, timestamp={timestamp}"
                )
            if not bool(torch.isfinite(w2c).all()):
                raise RuntimeError(
                    f"non-finite final DROID pose for video_idx={video_idx}"
                )
            valid_completed_depth = torch.isfinite(depth) & (depth > 0)
            if int(valid_completed_depth.sum().item()) < 100:
                raise RuntimeError(
                    f"non-finite/empty final DROID depth for video_idx={video_idx}"
                )
            records.append((video_idx, timestamp, depth, w2c))

        if len(records) != len(missing_ids):
            raise AssertionError("not every missing DROID keyframe was collected")
        if not records:
            self.final_refine_hydrated_droid_ids = []
            return []

        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.frame_reader.fx,
            fy=self.frame_reader.fy,
            cx=self.frame_reader.cx,
            cy=self.frame_reader.cy,
            W=self.frame_reader.W_out,
            H=self.frame_reader.H_out,
        ).transpose(0, 1).to(device=self.device)

        hydrated_pairs = []
        for video_idx, timestamp, depth, w2c in records:
            # Deliberately ignore dataset depth and pose. They are evaluation
            # data only; the promoted training camera is built exclusively
            # from the final DROID estimate plus the cached RGB observation.
            _, color, _, _, _ = self.frame_reader[timestamp]
            color = color.to(self.device)
            sharp_dir = Path(self.save_dir) / "sharp"
            sharp_file = sharp_dir / f"{timestamp}.pt"
            if sharp_file.is_file():
                color = load_tensor(timestamp, str(sharp_dir)).to(self.device)

            camera_data = {
                "gt_color": color.squeeze(),
                "glorie_depth": depth.detach().cpu().numpy(),
                "glorie_pose": w2c,
                "idx": video_idx,
            }
            if bool(self.config.get("composite_blur", False)):
                interpolation = str(self.config.get("interpolation", "linear"))
                knot_count = _motion_control_knot_count(interpolation)
                n_virtual_cams = int(self.config.get("n_virtual_cams", 5))
                if n_virtual_cams < 1:
                    raise ValueError("n_virtual_cams must be positive")
                # The third-party factory historically called this argument
                # ``realgt_pose``.  Hydration must never expose GT, so pass an
                # explicitly named sequence made only by repeating final DROID
                # w2c at every motion-control knot.
                estimated_pose_sequence = (
                    w2c.detach()
                    .clone()
                    .unsqueeze(0)
                    .repeat(knot_count, 1, 1)
                )
                camera_data.update(
                    {
                        "n_virtual_cams": n_virtual_cams,
                        "interpolation": interpolation,
                        "estimated_pose_sequence": estimated_pose_sequence,
                    }
                )
                viewpoint = Camera.init_from_dataset_motion(
                    self.frame_reader,
                    camera_data,
                    projection_matrix,
                    deblur_fail=True,
                )
                for knot in range(viewpoint.num_control_knots):
                    viewpoint.update_RT_motion(
                        w2c[:3, :3], w2c[:3, 3], knot
                    )
                viewpoint.motion_pose_seed_provenance = (
                    "repeated_droid_final_ba_w2c"
                )
            else:
                viewpoint = Camera.init_from_dataset(
                    self.frame_reader,
                    camera_data,
                    projection_matrix,
                )
            viewpoint.timestamp = timestamp
            viewpoint.is_blurry = not sharp_file.is_file()
            self._annotate_augmented_viewpoint(viewpoint, timestamp)
            viewpoint.update_RT(w2c[:3, :3], w2c[:3, 3])
            viewpoint.compute_grad_mask(self.config)
            viewpoint.is_valid = True
            viewpoint.hydrated_after_final_ba = True
            viewpoint.pose_provenance = "droid_final_ba"
            viewpoint.depth_provenance = "droid_final_ba_with_mono_completion"
            viewpoint.offline_gauge_anchor = video_idx == 0
            _freeze_offline_gauge_anchor_pose(viewpoint)

            self.cameras[video_idx] = viewpoint
            self.is_kf[video_idx] = False  # no online Gaussian seed/deformation
            self.kf2mapper_idx.setdefault(timestamp, video_idx)
            hydrated_pairs.append((video_idx, timestamp))

        all_pairs = existing_pairs + hydrated_pairs
        all_video_ids = [int(video_idx) for video_idx, _ in all_pairs]
        if len(all_video_ids) != len(set(all_video_ids)):
            raise RuntimeError("duplicate Mapper camera ids after DROID hydration")
        all_pairs.sort(key=lambda item: (int(item[1]), int(item[0])))
        self.video_idxs = [int(video_idx) for video_idx, _ in all_pairs]
        self.keyframe_idxs = [int(timestamp) for _, timestamp in all_pairs]

        # Make camera iteration deterministic and chronological as well. This
        # prevents the deferred frames from receiving an arbitrary tail order
        # in the uniform final-refinement sampler.
        self.cameras = dict(
            sorted(
                self.cameras.items(),
                key=lambda item: (int(item[1].timestamp), int(item[0])),
            )
        )
        hydrated_ids = [int(video_idx) for video_idx, _ in hydrated_pairs]
        remaining = set(range(video_count)) - {
            int(video_idx) for video_idx in self.video_idxs if int(video_idx) >= 0
        }
        if remaining:
            raise AssertionError(
                f"DROID hydration left missing keyframes: {sorted(remaining)}"
            )

        # Camera zero is DROID's gauge anchor and is now present in the
        # offline camera pool. Online initialization/Gaussians remain exactly
        # as they were; only final-refinement predecessor/anchor semantics are
        # made consistent with the complete chronological DROID sequence.
        if video_count > 0:
            self.initial_frame_uid = 0
        self.final_refine_hydrated_droid_ids = hydrated_ids
        self.printer.print(
            "Deferred final-BA DROID promotion: "
            f"hydrated {len(hydrated_ids)} training cameras "
            f"{hydrated_ids}; online map unchanged",
            FontColor.MAPPER,
        )
        return hydrated_ids

    def _clear_gt_source_indices(self):
        """Return the paper's clear evaluation indices, never generated views."""
        return clear_gt_source_indices(
            self.config, getattr(self, "frame_reader", None)
        )

    @staticmethod
    def _cpu_state_dict(module):
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in module.state_dict().items()
        }

    def _gaussian_inference_state(self):
        """Serialize GaussianModel, which intentionally is not an nn.Module."""
        tensor_names = (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
            "max_radii2D",
            "unique_kfIDs",
            "n_obs",
        )
        state = {
            name: getattr(self.gaussians, name).detach().cpu()
            for name in tensor_names
        }
        state.update(
            {
                "active_sh_degree": int(self.gaussians.active_sh_degree),
                "max_sh_degree": int(self.gaussians.max_sh_degree),
                "spatial_lr_scale": float(self.gaussians.spatial_lr_scale),
                # These checkpoints are for predeclared milestone diagnostics.
                # Optimizer moments are deliberately omitted, so they are not
                # advertised as bit-exact training-resume checkpoints.
                "resume_exact": False,
            }
        )
        if hasattr(self.gaussians, "mlp_rgb_ms"):
            state["mlp_rgb_ms"] = self._cpu_state_dict(
                self.gaussians.mlp_rgb_ms
            )
        if hasattr(self.gaussians, "mlp_rgb_ss"):
            state["mlp_rgb_ss"] = self._cpu_state_dict(
                self.gaussians.mlp_rgb_ss
            )
        return state

    @staticmethod
    def _resolve_refinement_budget(configured_budget, replay_enabled, replay_cfg):
        configured_budget = int(configured_budget)
        replay_iters = (
            max(0, int(replay_cfg.get("extra_iters", 0)))
            if replay_enabled
            else 0
        )
        budget_mode = str(replay_cfg.get("budget_mode", "replace_tail"))
        if replay_enabled and budget_mode == "replace_tail":
            if replay_iters > configured_budget:
                raise ValueError(
                    "ReSplat replay tail cannot exceed final_refine_iters in "
                    "replace_tail mode"
                )
            return configured_budget, configured_budget - replay_iters, budget_mode
        if replay_enabled and budget_mode == "extend":
            return configured_budget + replay_iters, configured_budget, budget_mode
        if replay_enabled:
            raise ValueError(
                "mapping.resplat.budget_mode must be replace_tail or extend"
            )
        return configured_budget, configured_budget, budget_mode

    def _save_refinement_checkpoint(self, step, total_iters, replay_sampler):
        """Save an inference checkpoint plus clear-GT PSNR/render artifacts."""
        checkpoint_dir = (
            Path(self.save_dir) / "refinement_checkpoints" / f"iter_{step:06d}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gaussians.save_ply(str(checkpoint_dir / "point_cloud.ply"))

        camera_state = {}
        for uid, viewpoint in self.cameras.items():
            camera_state[int(uid)] = {
                "R": viewpoint.R.detach().cpu(),
                "T": viewpoint.T.detach().cpu(),
                "exposure_a": viewpoint.exposure_a.detach().cpu(),
                "exposure_b": viewpoint.exposure_b.detach().cpu(),
                "timestamp": int(viewpoint.timestamp),
                "source_index": int(
                    getattr(viewpoint, "source_frame_index", viewpoint.timestamp)
                ),
                "synthetic": bool(getattr(viewpoint, "synthetic", False)),
                "fixed_pose": bool(getattr(viewpoint, "fixed_pose", False)),
                "hydrated_after_final_ba": bool(
                    getattr(viewpoint, "hydrated_after_final_ba", False)
                ),
                "pose_provenance": getattr(viewpoint, "pose_provenance", None),
                "depth_provenance": getattr(viewpoint, "depth_provenance", None),
                "motion_pose_seed_provenance": getattr(
                    viewpoint, "motion_pose_seed_provenance", None
                ),
                "offline_gauge_anchor": bool(
                    getattr(viewpoint, "offline_gauge_anchor", False)
                ),
            }
        if bool(self.resplat_cfg.get("save_full_state", True)):
            torch.save(
                {
                    "schema": "unblur_slam.refinement_checkpoint.v1",
                    "iteration": int(step),
                    "total_iterations": int(total_iters),
                    "gaussians": self._gaussian_inference_state(),
                    "cameras": camera_state,
                },
                checkpoint_dir / "model.pth",
            )
        else:
            torch.save(camera_state, checkpoint_dir / "cameras.pth")
        if replay_sampler is not None:
            replay_sampler.save_state(checkpoint_dir / "replay_state.json")

        metric_scope = clear_gt_metric_scope(self.config)
        metrics = {
            "iteration": int(step),
            "total_iterations": int(total_iters),
            "metric_scope": metric_scope,
            "mean_psnr_db": None,
            "num_frames": 0,
            "evaluated_source_indices": [],
            "checkpoint": str(checkpoint_dir),
        }
        clear_indices = (
            self._clear_gt_source_indices()
            if bool(self.resplat_cfg.get("checkpoint_clear_gt_psnr", True))
            else None
        )
        if clear_indices:
            render_dir = checkpoint_dir / "clear_gt_renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            scores = []
            evaluated_sources = set()
            with torch.no_grad():
                for viewpoint in self.cameras.values():
                    timestamp = int(viewpoint.timestamp)
                    metadata = self.frame_reader.frame_info(timestamp)
                    if (
                        bool(metadata.get("synthetic", False))
                        or not bool(metadata.get("eval", True))
                    ):
                        continue
                    source_index = int(metadata.get("source_index", timestamp))
                    if source_index not in clear_indices:
                        continue
                    rendering = render(
                        viewpoint,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                    )["render"].detach()
                    image = torch.clamp(
                        torch.exp(viewpoint.exposure_a.detach()) * rendering
                        + viewpoint.exposure_b.detach(),
                        0.0,
                        1.0,
                    )
                    _, gt_image, _, _, _ = self.frame_reader[timestamp]
                    gt_image = gt_image.squeeze().to(image.device)
                    mask = gt_image > 0
                    if not bool(mask.any()):
                        continue
                    mse = torch.mean((image[mask] - gt_image[mask]) ** 2)
                    score = float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item())
                    scores.append(score)
                    evaluated_sources.add(source_index)
                    pred = (
                        image.detach().cpu().permute(1, 2, 0).numpy() * 255.0
                    ).round().clip(0, 255).astype(np.uint8)
                    gt = (
                        gt_image.detach().cpu().permute(1, 2, 0).numpy() * 255.0
                    ).round().clip(0, 255).astype(np.uint8)
                    comparison = np.concatenate((gt, pred), axis=1)
                    cv2.imwrite(
                        str(render_dir / f"source_{source_index:06d}_gt_render.png"),
                        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
                    )
            expected_sources = available_clear_gt_source_indices(
                self.config, self.frame_reader
            )
            if evaluated_sources != expected_sources:
                missing = sorted(expected_sources - evaluated_sources)
                extra = sorted(evaluated_sources - expected_sources)
                raise RuntimeError(
                    "refinement checkpoint did not cover the complete configured "
                    "clear-GT scope; refusing an incomplete metric: "
                    f"missing={missing}, extra={extra}"
                )
            if len(scores) != len(evaluated_sources):
                raise RuntimeError(
                    "duplicate clear-GT source frames reached refinement "
                    "checkpoint evaluation; refusing a reweighted metric"
                )
            metrics["mean_psnr_db"] = float(np.mean(scores))
            metrics["num_frames"] = len(scores)
            metrics["evaluated_source_indices"] = sorted(evaluated_sources)

        import json

        with (checkpoint_dir / "clear_gt_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metrics, handle, indent=2)
            handle.write("\n")
        self.refinement_checkpoint_metrics.append(metrics)
        selected = next(
            (
                item
                for item in self.refinement_checkpoint_metrics
                if int(item["iteration"]) == int(total_iters)
            ),
            None,
        )
        summary = {
            "schema": "unblur_slam.refinement_checkpoint_metrics.v2",
            "metric_scope": metric_scope,
            "checkpoints": self.refinement_checkpoint_metrics,
            "selection_policy": "predeclared_final_iteration",
            "test_metric_used_for_selection": False,
            "selected_checkpoint": selected,
        }
        with (Path(self.save_dir) / "refinement_checkpoint_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        self.printer.print(
            "Refinement checkpoint "
            f"{step}/{total_iters}: {metric_scope} PSNR="
            f"{metrics['mean_psnr_db']} (diagnostic only; selected step="
            f"{total_iters})",
            FontColor.MAPPER,
        )

    @staticmethod
    def _laplacian_energy(image):
        gray = image.detach().float().mean(dim=0, keepdim=True).unsqueeze(0)
        kernel = image.new_tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        return F.conv2d(gray, kernel, padding=1).abs().mean()

    def _observe_replay(self, sampler, viewpoint, image, opacity, step):
        if sampler is None:
            return
        with torch.no_grad():
            prediction = image.detach().float().clamp(0.0, 1.0)
            target = viewpoint.original_image.detach().float().to(prediction.device)
            if target.shape[-2:] != prediction.shape[-2:]:
                target = F.interpolate(
                    target.unsqueeze(0),
                    size=prediction.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            residual = float((prediction - target).abs().mean().item())
            target_laplacian = self._laplacian_energy(target)
            prediction_laplacian = self._laplacian_energy(prediction)
            laplacian_gap = float(
                (prediction_laplacian - target_laplacian).abs().div(
                    target_laplacian.abs() + 1e-6
                ).item()
            )
            opacity_value = opacity.detach().float().clamp(0.0, 1.0)
            coverage = float(opacity_value.mean().item())
            novelty = float((opacity_value < 0.5).float().mean().item())
        sampler.observe(
            int(viewpoint.uid),
            residual=residual,
            laplacian_gap=laplacian_gap,
            coverage=coverage,
            novelty=novelty,
            reliability=self._observation_weight(viewpoint),
            step=int(step),
        )

    @staticmethod
    def _viewpoint_c2w(viewpoint):
        w2c = torch.eye(4, dtype=torch.float64)
        w2c[:3, :3] = viewpoint.R.detach().double().cpu()
        w2c[:3, 3] = viewpoint.T.detach().double().cpu()
        return torch.linalg.inv(w2c).numpy().tolist()

    def _align_generated_c2w(self, target_c2w, target_metadata):
        """Put a first-pass generated pose into the current map's gauge.

        FrameCrafter is necessarily a two-pass preprocessor because it needs an
        estimated trajectory.  The second SLAM pass can drift to a slightly
        different gauge.  We therefore anchor the target's first-pass pose to
        the closest already-mapped original frame, using estimated poses only.
        """
        framecrafter = self.config.get("framecrafter", {}) or {}
        if not framecrafter.get("align_manifest_pose_online", True):
            return target_c2w
        left_source = int(target_metadata.get("left_index", -1))
        candidates = []
        for camera in self.cameras.values():
            if bool(getattr(camera, "synthetic", False)):
                continue
            camera_source = int(
                getattr(camera, "source_frame_index", camera.timestamp)
            )
            if camera_source <= left_source:
                candidates.append((camera_source, camera))
        if not candidates:
            return target_c2w
        _, anchor = max(candidates, key=lambda item: item[0])
        anchor_metadata = self.frame_reader.frame_info(int(anchor.timestamp))
        first_pass_anchor = anchor_metadata.get("c2w")
        if first_pass_anchor is None:
            return target_c2w

        dtype, device = target_c2w.dtype, target_c2w.device
        reference_c2w = torch.as_tensor(
            first_pass_anchor, dtype=dtype, device=device
        )
        current_w2c = torch.eye(4, dtype=dtype, device=device)
        current_w2c[:3, :3] = anchor.R.detach().to(device=device, dtype=dtype)
        current_w2c[:3, 3] = anchor.T.detach().to(device=device, dtype=dtype)
        current_c2w = torch.linalg.inv(current_w2c)
        gauge = current_c2w @ torch.linalg.inv(reference_c2w)
        return gauge @ target_c2w

    def _save_submap_metadata(self, record):
        directory = Path(self.save_dir) / "submaps"
        record.save_checkpoint_metadata(
            directory / f"submap_{record.submap_id:04d}.json"
        )

    def _official_resplat_current_pose_state(self):
        poses = {
            int(frame_id): self._viewpoint_c2w(camera)
            for frame_id, camera in self.cameras.items()
            if getattr(camera, "R", None) is not None
            and getattr(camera, "T", None) is not None
        }
        return poses, int(self.iteration_count)

    def _poll_official_resplat_sidecar(self):
        if self.official_resplat_sidecar is None:
            return []
        poses, revision = self._official_resplat_current_pose_state()
        events = self.official_resplat_sidecar.poll(
            current_poses=poses,
            current_pose_revision=revision,
        )
        self.official_resplat_sidecar_events.extend(events)
        return events

    def _enqueue_official_resplat_sidecar(self, record, closure_ordinal):
        """Snapshot one closed map without touching the active Gaussian state."""
        queue = self.official_resplat_sidecar
        if queue is None:
            return None
        if not record.closed:
            raise RuntimeError("official ReSplat sidecars require a closed submap")
        existing = record.metadata.get("official_resplat_snapshot")
        if existing is not None:
            return existing
        unique_ids = list(dict.fromkeys(record.keyframe_ids))
        if len(unique_ids) < self.official_resplat_sidecar_cfg.context_keyframes:
            record.metadata["official_resplat_snapshot"] = {
                "status": "not_submitted",
                "reason": "fewer_than_8_closed_submap_keyframes",
                "available_keyframes": len(unique_ids),
            }
            self._save_submap_metadata(record)
            return record.metadata["official_resplat_snapshot"]

        ordered_ids = sorted(
            unique_ids,
            key=lambda value: self.submap_keyframe_ordinals.get(int(value), -1),
        )
        selected_ids = ordered_ids[
            -self.official_resplat_sidecar_cfg.context_keyframes :
        ]
        frames = []
        missing = []
        for frame_id in selected_ids:
            camera = self.cameras.get(int(frame_id))
            ordinal = self.submap_keyframe_ordinals.get(int(frame_id))
            image = None if camera is None else getattr(camera, "original_image", None)
            if camera is None or ordinal is None or image is None:
                missing.append(int(frame_id))
                continue
            intrinsics = (
                (float(camera.fx), 0.0, float(camera.cx)),
                (0.0, float(camera.fy), float(camera.cy)),
                (0.0, 0.0, 1.0),
            )
            frames.append(
                OfficialReSplatSidecarFrame(
                    frame_id=int(frame_id),
                    sequence_ordinal=int(ordinal),
                    c2w=self._viewpoint_c2w(camera),
                    intrinsics_px=intrinsics,
                    image=image,
                )
            )
        if missing:
            record.metadata["official_resplat_snapshot"] = {
                "status": "not_submitted",
                "reason": "closed_keyframe_payload_unavailable",
                "missing_frame_ids": missing,
            }
            self._save_submap_metadata(record)
            return record.metadata["official_resplat_snapshot"]

        snapshot_dir = materialize_closed_submap_snapshot(
            snapshots_root=queue.root / "snapshots",
            submap_id=record.submap_id,
            record_keyframe_ids=unique_ids,
            frames=frames,
            closure_sequence_ordinal=int(closure_ordinal),
            pose_revision=int(self.iteration_count),
        )
        event = queue.submit(snapshot_dir)
        record.metadata["official_resplat_snapshot"] = {
            "status": event["event"],
            "path": str(snapshot_dir),
            "queue_event": event,
            "active_map_merge_performed": False,
        }
        self.official_resplat_sidecar_events.append(event)
        self._save_submap_metadata(record)
        return record.metadata["official_resplat_snapshot"]

    def _register_submap_keyframe(self, viewpoint):
        """Record LoopSplat-style boundaries without claiming registration."""
        if self.submap_policy is None:
            return
        frame_id = int(viewpoint.uid)
        if frame_id not in self.submap_keyframe_ordinals:
            self.submap_keyframe_ordinal += 1
            self.submap_keyframe_ordinals[frame_id] = self.submap_keyframe_ordinal
        current_ordinal = self.submap_keyframe_ordinals[frame_id]
        current_c2w = self._viewpoint_c2w(viewpoint)
        if self.active_submap is None:
            self.active_submap = SubmapRecord(
                submap_id=0,
                anchor_frame_id=frame_id,
                anchor_c2w=current_c2w,
                metadata={
                    "partition_only": True,
                    "loop_registration_implemented": False,
                },
            )
            self.submap_records.append(self.active_submap)
            viewpoint.submap_id = self.active_submap.submap_id
            self._save_submap_metadata(self.active_submap)
            return

        decision = self.submap_policy.decide(
            self.active_submap.anchor_c2w,
            current_c2w,
            len(self.active_submap.keyframe_ids),
        )
        if decision.start_new_submap:
            previous = self.active_submap
            previous.closed = True
            previous.metadata["boundary"] = {
                "reasons": decision.reasons,
                "translation_delta": decision.translation_delta,
                "rotation_delta_deg": decision.rotation_delta_deg,
            }
            self._save_submap_metadata(previous)
            self._enqueue_official_resplat_sidecar(previous, current_ordinal)
            overlap = max(0, int(self.submap_cfg.get("overlap_keyframes", 5)))
            overlap_ids = previous.keyframe_ids[-overlap:] if overlap else []
            self.active_submap = SubmapRecord(
                submap_id=len(self.submap_records),
                anchor_frame_id=frame_id,
                anchor_c2w=current_c2w,
                metadata={
                    "partition_only": True,
                    "loop_registration_implemented": False,
                    "overlap_from_submap": previous.submap_id,
                },
            )
            for overlap_id in overlap_ids:
                self.active_submap.add_frame(overlap_id, is_keyframe=True)
            self.submap_records.append(self.active_submap)
        self.active_submap.add_frame(frame_id, is_keyframe=True)
        viewpoint.submap_id = self.active_submap.submap_id
        self._save_submap_metadata(self.active_submap)
        self._poll_official_resplat_sidecar()

    def _finalize_submaps(self):
        if self.active_submap is None:
            return
        self.active_submap.closed = True
        self._save_submap_metadata(self.active_submap)
        closure_ordinal = max(
            (
                self.submap_keyframe_ordinals[int(frame_id)]
                for frame_id in self.active_submap.keyframe_ids
                if int(frame_id) in self.submap_keyframe_ordinals
            ),
            default=self.submap_keyframe_ordinal,
        )
        self._enqueue_official_resplat_sidecar(
            self.active_submap, closure_ordinal
        )
        if self.official_resplat_sidecar is not None:
            events = self.official_resplat_sidecar.drain(
                current_pose_provider=self._official_resplat_current_pose_state
            )
            self.official_resplat_sidecar_events.extend(events)
        summary = {
            "schema": "unblur_slam.submap_partition_summary.v1",
            "partition_only": True,
            "registration_implemented": False,
            "official_resplat_sidecar": {
                "enabled": self.official_resplat_sidecar is not None,
                "mode": "sidecar_only",
                "active_map_merge_implemented": False,
                "events": self.official_resplat_sidecar_events,
            },
            "submaps": [record.checkpoint_metadata() for record in self.submap_records],
        }
        path = Path(self.save_dir) / "submaps" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

    @staticmethod
    def _fusion_hash_tensor(digest, name, tensor):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))

    def _capture_active_gaussian_transaction_state(self):
        """Clone every live-map value that an append/rollback may touch."""

        parameter_attributes = {
            "xyz": "_xyz",
            "f_dc": "_features_dc",
            "f_rest": "_features_rest",
            "opacity": "_opacity",
            "scaling": "_scaling",
            "rotation": "_rotation",
        }
        parameters = {
            name: getattr(self.gaussians, attribute).detach().clone()
            for name, attribute in parameter_attributes.items()
        }
        statistics = {
            name: getattr(self.gaussians, name).detach().clone()
            for name in ("xyz_gradient_accum", "denom", "max_radii2D")
        }
        metadata = {
            "unique_kfIDs": self.gaussians.unique_kfIDs.detach().clone(),
            "n_obs": self.gaussians.n_obs.detach().clone(),
        }
        optimizer = {}
        for group in self.gaussians.optimizer.param_groups:
            name = group.get("name")
            if name not in parameter_attributes:
                continue
            parameter = group["params"][0]
            state = self.gaussians.optimizer.state.get(parameter)
            optimizer[name] = None if state is None else {
                key: value.detach().clone()
                for key, value in state.items()
                if isinstance(value, torch.Tensor)
            }
        snapshot = {
            "before_count": int(self.gaussians.get_xyz.shape[0]),
            "parameter_attributes": parameter_attributes,
            "parameters": parameters,
            "statistics": statistics,
            "metadata": metadata,
            "optimizer": optimizer,
        }
        snapshot["sha256"] = self._active_gaussian_state_sha256(snapshot)
        return snapshot

    def _active_gaussian_state_sha256(self, snapshot=None):
        digest = hashlib.sha256()
        if snapshot is None:
            snapshot = self._capture_active_gaussian_transaction_state()
            return snapshot["sha256"]
        for section in ("parameters", "statistics", "metadata"):
            for name in sorted(snapshot[section]):
                self._fusion_hash_tensor(
                    digest, f"{section}.{name}", snapshot[section][name]
                )
        for group_name in sorted(snapshot["optimizer"]):
            state = snapshot["optimizer"][group_name]
            digest.update(f"optimizer.{group_name}".encode("utf-8"))
            if state is None:
                digest.update(b"none")
                continue
            for name in sorted(state):
                self._fusion_hash_tensor(
                    digest, f"optimizer.{group_name}.{name}", state[name]
                )
        return digest.hexdigest()

    def _assert_active_state_matches_transaction(self, snapshot):
        current = self._capture_active_gaussian_transaction_state()
        if int(current["before_count"]) != int(snapshot["before_count"]):
            raise RuntimeError("active-fusion rollback did not restore Gaussian count")
        for section in ("parameters", "statistics", "metadata"):
            if set(current[section]) != set(snapshot[section]):
                raise RuntimeError(f"active-fusion rollback changed {section} keys")
            for name in snapshot[section]:
                if not torch.equal(current[section][name], snapshot[section][name]):
                    raise RuntimeError(
                        f"active-fusion rollback changed {section}.{name} bytes"
                    )
        if set(current["optimizer"]) != set(snapshot["optimizer"]):
            raise RuntimeError("active-fusion rollback changed optimizer group keys")
        for group_name, expected in snapshot["optimizer"].items():
            observed = current["optimizer"][group_name]
            if (expected is None) != (observed is None):
                raise RuntimeError(
                    f"active-fusion rollback changed optimizer state {group_name}"
                )
            if expected is None:
                continue
            if set(expected) != set(observed):
                raise RuntimeError(
                    f"active-fusion rollback changed optimizer keys {group_name}"
                )
            for name in expected:
                if not torch.equal(observed[name], expected[name]):
                    raise RuntimeError(
                        f"active-fusion rollback changed optimizer {group_name}.{name}"
                    )
        if current["sha256"] != snapshot["sha256"]:
            raise RuntimeError("active-fusion rollback state SHA-256 mismatch")
        return current["sha256"]

    def _rollback_active_gaussian_append(self, snapshot):
        """Remove an appended tail and restore reset densification statistics."""

        before = int(snapshot["before_count"])
        current = int(self.gaussians.get_xyz.shape[0])
        if current < before:
            raise RuntimeError("append-mode fusion unexpectedly removed active Gaussians")
        if current > before:
            prune = torch.arange(current, device=self.gaussians.get_xyz.device) >= before
            self.gaussians.prune_points(prune)
        for name, value in snapshot["statistics"].items():
            setattr(self.gaussians, name, value.detach().clone())
        return self._assert_active_state_matches_transaction(snapshot)

    def _active_fusion_context_metrics(self, frame_ids):
        """Render the same eight online observations; never load clear GT."""

        cfg = self.official_resplat_active_fusion_cfg
        gate = dict(cfg.postmerge_quality_gate)
        l1_weight = float(gate["l1_weight"])
        ssim_weight = float(gate["one_minus_ssim_weight"])
        records = []
        with torch.no_grad():
            for frame_id in frame_ids:
                viewpoint = self.cameras[int(frame_id)]
                if bool(getattr(viewpoint, "deblur_fail", False)):
                    virtual_count = int(viewpoint.n_virtual_cams)
                    if virtual_count <= 0:
                        raise RuntimeError(
                            "deblur-fail context requires at least one virtual camera"
                        )
                    R, t, theta, rho = viewpoint.get_virtual_extrinsics()
                    if any(
                        len(values) != virtual_count
                        for values in (R, t, theta, rho)
                    ):
                        raise RuntimeError(
                            "virtual-extrinsics count does not match n_virtual_cams"
                        )
                    middle = int(viewpoint.n_virtual_cams) // 2
                    package = render_virtual(
                        viewpoint,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                        R=R[middle],
                        t=t[middle],
                        theta=theta[middle],
                        rho=rho[middle],
                    )
                else:
                    package = render(
                        viewpoint,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                    )
                raw_prediction = package["render"].detach().float()
                prediction = (
                    torch.exp(viewpoint.exposure_a.detach()) * raw_prediction
                    + viewpoint.exposure_b.detach()
                ).clamp(0.0, 1.0)
                observation = (
                    viewpoint.original_image.detach()
                    .float()
                    .to(prediction.device)
                    .clamp(0.0, 1.0)
                )
                if observation.shape[-2:] != prediction.shape[-2:]:
                    observation = F.interpolate(
                        observation.unsqueeze(0),
                        size=prediction.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                l1_value = float((prediction - observation).abs().mean().item())
                ssim_value = float(ssim(prediction, observation).item())
                composite = l1_weight * l1_value + ssim_weight * (
                    1.0 - ssim_value
                )
                records.append(
                    {
                        "frame_id": int(frame_id),
                        "source_index": int(viewpoint.timestamp),
                        "l1": l1_value,
                        "ssim": ssim_value,
                        "composite": composite,
                    }
                )
        return {
            "uses_clear_gt": False,
            "inputs": "same_eight_context_observations",
            "renderer_proxy": "exposure_compensated_midpoint_without_learned_blur",
            "purpose": "isolate_active_gaussian_map_change",
            "per_view": records,
            "mean_l1": float(sum(item["l1"] for item in records) / len(records)),
            "mean_ssim": float(
                sum(item["ssim"] for item in records) / len(records)
            ),
            "mean_composite": float(
                sum(item["composite"] for item in records) / len(records)
            ),
        }

    def _prepare_active_fusion_visibility_cache_refresh(
        self,
        *,
        before_count,
        after_count,
        expected_active_state_sha256,
    ):
        """Re-render every cached camera against the appended active map.

        This method deliberately does not mutate ``occ_aware_visibility``.  It
        constructs and validates a complete replacement first, so the caller
        can publish it with one dictionary assignment immediately before the
        map transaction is marked committed.  A render or validation failure
        therefore leaves the old cache byte-for-byte reachable for rollback.
        """

        started = time.monotonic()
        before_count = int(before_count)
        after_count = int(after_count)
        if before_count <= 0 or after_count <= before_count:
            raise RuntimeError(
                "visibility-cache refresh requires a non-empty appended map"
            )
        active_count = int(self.gaussians.get_xyz.shape[0])
        if active_count != after_count:
            raise RuntimeError(
                "visibility-cache refresh active-map count does not match after_count"
            )
        active_device = self.gaussians.get_xyz.device
        old_cache = self.occ_aware_visibility
        if not isinstance(old_cache, dict) or not old_cache:
            raise RuntimeError(
                "visibility-cache refresh requires the live non-empty cache"
            )
        old_keys = tuple(old_cache.keys())
        if old_keys != tuple(self.current_window):
            raise RuntimeError(
                "visibility-cache keys must exactly match the live mapping window"
            )
        refreshed = {}
        per_camera = []
        old_cache_object_id = id(old_cache)
        old_tensor_object_ids = {
            frame_id: id(old_cache[frame_id]) for frame_id in old_keys
        }
        old_cache_digest_before = hashlib.sha256()
        old_cache_digest_before.update(
            json.dumps([int(value) for value in old_keys]).encode("ascii")
        )

        for frame_id in old_keys:
            if isinstance(frame_id, bool) or not isinstance(frame_id, int):
                raise RuntimeError("visibility-cache frame keys must be exact integers")
            old_visibility = old_cache[frame_id]
            if (
                not isinstance(old_visibility, torch.Tensor)
                or old_visibility.ndim != 1
                or int(old_visibility.numel()) != before_count
                or old_visibility.device != active_device
                or old_visibility.dtype != torch.long
                or not old_visibility.is_contiguous()
            ):
                raise RuntimeError(
                    "visibility-cache premerge vector must be a live-device "
                    f"binary-long 1-D tensor of length {before_count} for frame "
                    f"{frame_id}"
                )
            if not bool(
                torch.logical_or(old_visibility == 0, old_visibility == 1)
                .all()
                .item()
            ):
                raise RuntimeError(
                    f"visibility-cache premerge vector is non-binary for frame {frame_id}"
                )
            if frame_id not in self.cameras:
                raise RuntimeError(
                    f"visibility-cache camera {frame_id} is not live"
                )
            if int(self.cameras[frame_id].uid) != frame_id:
                raise RuntimeError(
                    "visibility-cache dictionary key does not equal camera uid"
                )
            self._fusion_hash_tensor(
                old_cache_digest_before,
                f"occ_aware_visibility.{frame_id}",
                old_visibility,
            )
        old_cache_sha256_before = old_cache_digest_before.hexdigest()

        with torch.no_grad():
            for frame_id in old_keys:
                old_visibility = old_cache[frame_id]
                viewpoint = self.cameras[frame_id]
                deblur_fail = bool(getattr(viewpoint, "deblur_fail", False))
                if deblur_fail:
                    virtual_count = int(viewpoint.n_virtual_cams)
                    if virtual_count <= 0:
                        raise RuntimeError(
                            "deblur-fallback cache camera has no virtual views"
                        )
                    R, t, theta, rho = viewpoint.get_virtual_extrinsics()
                    touched_per_virtual = []
                    for virtual_cam in range(virtual_count):
                        package = render_virtual(
                            viewpoint,
                            self.gaussians,
                            self.pipeline_params,
                            self.background,
                            R=R[virtual_cam],
                            t=t[virtual_cam],
                            theta=theta[virtual_cam],
                            rho=rho[virtual_cam],
                        )
                        n_touched = package.get("n_touched")
                        if not isinstance(n_touched, torch.Tensor):
                            raise RuntimeError(
                                "virtual render did not return tensor n_touched"
                            )
                        if (
                            n_touched.ndim != 1
                            or int(n_touched.numel()) != after_count
                            or n_touched.device != active_device
                        ):
                            raise RuntimeError(
                                "virtual-render n_touched must match the live "
                                f"postmerge map for frame {frame_id}"
                            )
                        touched_per_virtual.append(n_touched)
                    max_n_touched = torch.stack(
                        touched_per_virtual, dim=0
                    ).max(dim=0).values
                    new_visibility = (max_n_touched > 0).long()
                    renderer_semantics = "all_virtual_views_max_n_touched"
                else:
                    package = render(
                        viewpoint,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                    )
                    n_touched = package.get("n_touched")
                    if not isinstance(n_touched, torch.Tensor):
                        raise RuntimeError(
                            "regular render did not return tensor n_touched"
                        )
                    if (
                        n_touched.ndim != 1
                        or int(n_touched.numel()) != after_count
                        or n_touched.device != active_device
                    ):
                        raise RuntimeError(
                            "regular-render n_touched must match the live postmerge "
                            f"map for frame {frame_id}"
                        )
                    new_visibility = (n_touched > 0).long()
                    virtual_count = 0
                    renderer_semantics = "single_regular_render_n_touched"

                if (
                    new_visibility.ndim != 1
                    or int(new_visibility.numel()) != after_count
                    or new_visibility.device != active_device
                ):
                    raise RuntimeError(
                        "visibility-cache postmerge vector must be a live-device "
                        f"1-D tensor of length {after_count} for frame {frame_id}"
                    )
                new_visibility = new_visibility.detach().contiguous()
                if (
                    new_visibility.dtype != torch.long
                    or not new_visibility.is_contiguous()
                    or not bool(
                        torch.logical_or(new_visibility == 0, new_visibility == 1)
                        .all()
                        .item()
                    )
                ):
                    raise RuntimeError(
                        "fresh visibility-cache vector must be contiguous binary long"
                    )
                refreshed[frame_id] = new_visibility
                per_camera.append(
                    {
                        "frame_id": int(frame_id),
                        "source_index": int(viewpoint.timestamp),
                        "deblur_fail": deblur_fail,
                        "renderer_semantics": renderer_semantics,
                        "virtual_views_rendered": virtual_count,
                        "old_vector_length": int(old_visibility.numel()),
                        "new_vector_length": int(new_visibility.numel()),
                        "old_visible_gaussians": int(
                            old_visibility.count_nonzero().item()
                        ),
                        "new_visible_gaussians": int(
                            new_visibility.count_nonzero().item()
                        ),
                    }
                )

        if tuple(refreshed.keys()) != old_keys:
            raise RuntimeError("visibility-cache key order or membership changed")
        old_cache_digest_after = hashlib.sha256()
        old_cache_digest_after.update(
            json.dumps([int(value) for value in old_keys]).encode("ascii")
        )
        if (
            self.occ_aware_visibility is not old_cache
            or id(self.occ_aware_visibility) != old_cache_object_id
            or tuple(self.occ_aware_visibility.keys()) != old_keys
        ):
            raise RuntimeError(
                "visibility cache object or membership changed during preparation"
            )
        for frame_id in old_keys:
            if id(self.occ_aware_visibility[frame_id]) != old_tensor_object_ids[frame_id]:
                raise RuntimeError(
                    "visibility cache tensor object changed during preparation"
                )
            self._fusion_hash_tensor(
                old_cache_digest_after,
                f"occ_aware_visibility.{frame_id}",
                self.occ_aware_visibility[frame_id],
            )
        old_cache_sha256_after = old_cache_digest_after.hexdigest()
        if old_cache_sha256_after != old_cache_sha256_before:
            raise RuntimeError(
                "visibility cache tensor bytes changed during preparation"
            )
        post_render_state = self._capture_active_gaussian_transaction_state()
        if (
            int(post_render_state["before_count"]) != after_count
            or str(post_render_state["sha256"])
            != str(expected_active_state_sha256)
        ):
            raise RuntimeError(
                "visibility-cache rendering mutated the active Gaussian state"
            )
        elapsed = time.monotonic() - started
        report = {
            "schema": "unblur_slam.active_fusion_visibility_cache_refresh.v1",
            "status": "validated_for_atomic_commit",
            "accepted": True,
            "atomic_replacement": True,
            "atomic_commit_assignment_performed": False,
            "old_cache_mutated_during_preparation": False,
            "old_cache_object_and_tensor_identities_preserved": True,
            "old_cache_bytes_preserved": True,
            "old_cache_sha256_before": old_cache_sha256_before,
            "old_cache_sha256_after": old_cache_sha256_after,
            "zero_padding_used": False,
            "padding_or_truncation_used": False,
            "all_values_derived_from_fresh_active_map_renders": True,
            "deblur_fallback_uses_all_virtual_views_max_n_touched": True,
            "keys_before": [int(value) for value in old_keys],
            "keys_after": [int(value) for value in refreshed.keys()],
            "key_membership_and_order_unchanged": True,
            "before_gaussian_count": before_count,
            "after_gaussian_count": after_count,
            "active_map_state_sha256_before_and_after_render": str(
                expected_active_state_sha256
            ),
            "active_map_unchanged_by_refresh_rendering": True,
            "uses_ground_truth": False,
            "uses_clear_gt_metrics": False,
            "inputs": "active_map_and_cached_online_cameras_only",
            "per_camera": per_camera,
            "elapsed_seconds": elapsed,
        }
        return refreshed, report

    def _active_fusion_pose_state(self, frame_ids):
        return {
            int(frame_id): self._viewpoint_c2w(self.cameras[int(frame_id)])
            for frame_id in frame_ids
        }

    def _run_synchronous_active_resplat_fusion(self, frame_ids, source_indices):
        cfg = self.official_resplat_active_fusion_cfg
        queue = self.official_resplat_active_fusion_queue
        if queue is None:
            raise RuntimeError("enabled active fusion has no official sidecar queue")
        started = time.monotonic()
        timings = {
            "snapshot_seconds": 0.0,
            "subprocess_and_publication_seconds": 0.0,
            "premerge_active_render_seconds": 0.0,
            "merge_seconds": 0.0,
            "postmerge_active_render_seconds": 0.0,
            "visibility_cache_refresh_seconds": 0.0,
            "rollback_seconds": 0.0,
        }
        audit = {
            "schema": ACTIVE_FUSION_AUDIT_SCHEMA,
            "status": "started",
            "active_map_changed_final": False,
            "uses_ground_truth": False,
            "uses_clear_gt_metrics": False,
            "data_lineage": {
                "selection_membership_clear_gt_conditioned": True,
                "ground_truth_poses_or_depths_consumed_by_fusion": False,
                "independent_clear_pixels_consumed_by_fusion": False,
                "clear_gt_metrics_consumed_by_fusion": False,
            },
            "trigger": {
                "after_fully_mapped_keyframe_count": len(frame_ids),
                "source_index": int(source_indices[-1]),
                "frame_ids": [int(value) for value in frame_ids],
                "source_indices": [int(value) for value in source_indices],
                "iteration_count": int(self.iteration_count),
            },
            "official_state": {
                "requested_recurrent_updates": 3,
                "selected_state_index_zero_based": 2,
                "fourth_state_computed": False,
            },
            "timing": timings,
        }
        if cfg.posthoc_after_v2_rejection:
            audit["diagnostic_protocol"] = {
                "schema": (
                    "unblur_slam.posthoc_visibility_cache_refresh_forced_commit_"
                    "diagnostic.v1"
                    if cfg.refresh_occ_aware_visibility_after_forced_commit
                    else (
                        "unblur_slam.posthoc_count_agnostic_forced_commit_"
                        "diagnostic.v1"
                        if cfg.count_agnostic_forced_commit
                        else "unblur_slam.posthoc_forced_commit_diagnostic.v1"
                    )
                ),
                "posthoc_after_v2_rejection": True,
                "posthoc_after_v3_count_mismatch": bool(
                    cfg.posthoc_after_v3_count_mismatch
                ),
                "unsafe_not_deployable": True,
                "gate_thresholds_unchanged": True,
                "force_only_after_postmerge_gate_rejected": True,
                "ordinary_policy_would_rollback": True,
                "expected_forced_commit_gaussian_count": int(
                    cfg.expected_forced_commit_gaussian_count
                ),
                "count_contract": {
                    "kind": (
                        "fresh_run_merge_bounds_and_internal_count_algebra"
                        if cfg.count_agnostic_forced_commit
                        else "cross_run_exact_count"
                    ),
                    "cross_run_exact_count_required": not bool(
                        cfg.count_agnostic_forced_commit
                    ),
                    "minimum_accepted_count": int(
                        cfg.merge["min_new_gaussians"]
                    ),
                    "maximum_accepted_count": int(
                        cfg.merge["max_new_gaussians"]
                    ),
                },
                "v2_rejection_audit_sha256": str(
                    cfg.v2_rejection_audit_sha256
                ),
                "v3_count_mismatch_audit_sha256": str(
                    cfg.v3_count_mismatch_audit_sha256
                ),
                "posthoc_after_v4_visibility_cache_mismatch": bool(
                    cfg.posthoc_after_v4_visibility_cache_mismatch
                ),
                "refresh_occ_aware_visibility_after_forced_commit": bool(
                    cfg.refresh_occ_aware_visibility_after_forced_commit
                ),
                "v4_visibility_cache_failure_audit_sha256": str(
                    cfg.v4_visibility_cache_failure_audit_sha256
                ),
                "uses_ground_truth": False,
                "uses_clear_gt_metrics": False,
            }
        transaction = None
        committed = False
        visibility_cache_before_commit = None
        visibility_cache_replaced = False
        try:
            stage = time.monotonic()
            frames = []
            for ordinal, frame_id in enumerate(frame_ids):
                camera = self.cameras[int(frame_id)]
                frames.append(
                    OfficialReSplatSidecarFrame(
                        frame_id=int(frame_id),
                        sequence_ordinal=int(ordinal),
                        c2w=self._viewpoint_c2w(camera),
                        intrinsics_px=(
                            (float(camera.fx), 0.0, float(camera.cx)),
                            (0.0, float(camera.fy), float(camera.cy)),
                            (0.0, 0.0, 1.0),
                        ),
                        image=camera.original_image,
                    )
                )
            snapshot_dir = materialize_closed_submap_snapshot(
                snapshots_root=queue.root / "snapshots",
                submap_id=0,
                record_keyframe_ids=list(frame_ids),
                frames=frames,
                closure_sequence_ordinal=len(frame_ids) - 1,
                pose_revision=int(self.iteration_count),
                integration_mode="online_mapper",
                selection_source="online_mapper_closed_submap_membership",
                source_provenance={
                    "schema": "unblur_slam.fixed_kf_resplat3_source_provenance.v1",
                    "coordinate_domain": "dataset_source_index",
                    "source_indices": [int(value) for value in source_indices],
                    "frame_id_to_source_index": {
                        str(frame_id): int(source)
                        for frame_id, source in zip(frame_ids, source_indices)
                    },
                    "uses_ground_truth": False,
                    "selection_membership_clear_gt_conditioned": True,
                    "uses_ground_truth_pose_or_depth": False,
                    "uses_independent_clear_pixels": False,
                    "uses_clear_gt_metrics": False,
                },
                uses_clear_gt_membership=True,
                uses_independent_clear_pixels=False,
            )
            timings["snapshot_seconds"] = time.monotonic() - stage
            snapshot = load_official_resplat_snapshot(snapshot_dir)
            audit["snapshot"] = {
                "path": str(snapshot_dir),
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "pose_revision": int(snapshot["pose_revision"]),
            }

            stage = time.monotonic()
            queue_events = [queue.submit(snapshot_dir)]
            deadline = time.monotonic() + float(cfg.max_runtime_seconds) + 10.0
            while (queue.active is not None or queue.pending) and time.monotonic() <= deadline:
                queue_events.extend(
                    queue.poll(
                        current_poses=self._active_fusion_pose_state(frame_ids),
                        current_pose_revision=int(self.iteration_count),
                    )
                )
                if queue.active is not None:
                    time.sleep(0.05)
            if queue.active is not None or queue.pending:
                queue_events.extend(
                    queue.drain(
                        current_pose_provider=lambda: (
                            self._active_fusion_pose_state(frame_ids),
                            int(self.iteration_count),
                        )
                    )
                )
            timings["subprocess_and_publication_seconds"] = time.monotonic() - stage
            audit["queue_events"] = queue_events
            published = [
                event for event in queue_events if event.get("event") == "published"
            ]
            if len(published) != 1:
                audit["status"] = "sidecar_rejected"
                audit["rejection_reasons"] = [
                    "official_state3_sidecar_not_published_exactly_once"
                ]
                return audit
            result_root = Path(str(published[0]["path"])).resolve()
            manifest_path = result_root / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit["published_result"] = {
                "path": str(result_root),
                "manifest_path": str(manifest_path),
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
            world_gate = verify_unblur_world_artifact(
                manifest,
                result_root,
                snapshot,
                expected_refinement_updates=3,
            )
            context_gate = evaluate_resplat_context_reconstruction(
                manifest, cfg, snapshot
            )
            repository = (
                (manifest.get("official_resplat") or {}).get("repository") or {}
            )
            observed_commit = str(repository.get("commit", ""))
            repository_gate = {
                "accepted": observed_commit == str(cfg.resplat_repo_commit),
                "expected_commit": str(cfg.resplat_repo_commit),
                "observed_commit": observed_commit,
                "exact_match_required": True,
            }
            manifest_lineage = {
                "selection_membership_clear_gt_conditioned": manifest.get(
                    "selection_membership_clear_gt_conditioned"
                ),
                "ground_truth_pose_or_depth_used": manifest.get(
                    "ground_truth_pose_or_depth_used"
                ),
                "independent_clear_pixels_used": manifest.get(
                    "independent_clear_pixels_used"
                ),
                "clear_gt_metrics_used": manifest.get("clear_gt_metrics_used"),
            }
            lineage_ok = manifest_lineage == {
                "selection_membership_clear_gt_conditioned": True,
                "ground_truth_pose_or_depth_used": False,
                "independent_clear_pixels_used": False,
                "clear_gt_metrics_used": False,
            }
            lineage_gate = {
                "accepted": lineage_ok,
                **manifest_lineage,
            }
            audit["premerge_gates"] = {
                "world_artifact": world_gate.to_dict(),
                "context_reconstruction": context_gate,
                "repository_provenance": repository_gate,
                "data_lineage": lineage_gate,
            }
            premerge_reasons = [
                *world_gate.reasons,
                *context_gate["reasons"],
            ]
            if not repository_gate["accepted"]:
                premerge_reasons.append("official_resplat_commit_mismatch")
            if not lineage_gate["accepted"]:
                premerge_reasons.append("fusion_data_lineage_disclosure_mismatch")
            if premerge_reasons:
                audit["status"] = "premerge_gate_rejected"
                audit["rejection_reasons"] = list(premerge_reasons)
                return audit

            outputs = manifest.get("outputs") or {}
            relative = Path(str(outputs.get("unblur_world_gaussians_npz", "")))
            if relative.is_absolute() or ".." in relative.parts or not str(relative):
                raise ValueError("invalid bound Unblur-world artifact path")
            world_path = result_root / relative
            with np.load(world_path, allow_pickle=False) as archive:
                required = {
                    "means_world",
                    "scales_world",
                    "rotations_world_wxyz",
                    "harmonics_world",
                    "opacities",
                    "owner_frame_ids",
                }
                if not required.issubset(archive.files):
                    raise ValueError("Unblur-world artifact lacks active merge arrays")
                arrays = {name: np.asarray(archive[name]) for name in required}
            expected_sh = 1 + int(self.gaussians._features_rest.shape[1])
            if arrays["harmonics_world"].shape != (
                int(arrays["means_world"].shape[0]),
                3,
                expected_sh,
            ):
                raise ValueError(
                    "world artifact SH dimension must exactly match the active map; "
                    "mapper-side SH truncation is forbidden"
                )
            bridge = manifest.get("unblur_world_artifact") or {}
            if (
                int(bridge.get("source_harmonic_dimension", -1)) != 16
                or int(bridge.get("imported_harmonic_dimension", -1)) != expected_sh
                or int(bridge.get("dropped_higher_order_harmonics", -1)) != 15
                or bridge.get("official_no_rotate_sh") is not True
            ):
                raise ValueError("DC-only official no_rotate_sh bridge contract drifted")
            audit["sh_import"] = {
                "native_harmonic_dimension": 16,
                "active_harmonic_dimension": expected_sh,
                "dropped_higher_order_coefficients": 15,
                "mapper_side_truncation_performed": False,
                "dc_only": True,
            }

            stage = time.monotonic()
            before_metrics = self._active_fusion_context_metrics(frame_ids)
            timings["premerge_active_render_seconds"] = time.monotonic() - stage
            transaction = self._capture_active_gaussian_transaction_state()
            audit["active_state_before"] = {
                "gaussian_count": int(transaction["before_count"]),
                "sha256": transaction["sha256"],
            }

            stage = time.monotonic()
            merge_report = self.gaussians.merge_resplat_submap(
                means_world=arrays["means_world"],
                scales_linear=arrays["scales_world"],
                rotations_wxyz_world=arrays["rotations_world_wxyz"],
                harmonics_world=arrays["harmonics_world"],
                opacities_probability=arrays["opacities"],
                owner_kf_ids=arrays["owner_frame_ids"],
                replace_kf_ids=[int(value) for value in frame_ids],
                merge_config=dict(cfg.merge),
            )
            timings["merge_seconds"] = time.monotonic() - stage
            audit["merge"] = merge_report
            if merge_report.get("mode") != "append" or int(
                merge_report.get("removed_owned_count", -1)
            ) != 0:
                raise RuntimeError("first active-fusion experiment must be append-only")
            if not bool(merge_report.get("active_map_changed", False)):
                final_hash = self._assert_active_state_matches_transaction(transaction)
                audit["status"] = "merge_gate_rejected"
                audit["rejection_reasons"] = [str(merge_report.get("status"))]
                audit["active_state_final"] = {
                    "gaussian_count": int(self.gaussians.get_xyz.shape[0]),
                    "sha256": final_hash,
                    "byte_identical_to_premerge": True,
                }
                return audit

            trial_state = self._capture_active_gaussian_transaction_state()
            stage = time.monotonic()
            after_metrics = self._active_fusion_context_metrics(frame_ids)
            timings["postmerge_active_render_seconds"] = time.monotonic() - stage
            post_gate = evaluate_postmerge_reconstruction(
                before_metrics, after_metrics, cfg
            )
            audit["postmerge_gate"] = {
                "decision": post_gate,
                "before": before_metrics,
                "after": after_metrics,
            }
            audit["active_state_trial"] = {
                "gaussian_count": int(trial_state["before_count"]),
                "sha256": trial_state["sha256"],
            }
            if not post_gate["accepted"]:
                if cfg.force_commit_after_postmerge_rejection:
                    accepted_count = int(merge_report.get("accepted_count", -1))
                    expected_count = int(cfg.expected_forced_commit_gaussian_count)
                    minimum_count = int(cfg.merge["min_new_gaussians"])
                    maximum_count = int(cfg.merge["max_new_gaussians"])
                    before_count = int(transaction["before_count"])
                    trial_count = int(trial_state["before_count"])
                    count_contract = forced_commit_count_contract(
                        accepted_count=accepted_count,
                        before_count=before_count,
                        after_count=trial_count,
                        config=cfg,
                    )
                    count_rule_failed = (
                        not count_contract["within_preregistered_merge_bounds"]
                        if cfg.count_agnostic_forced_commit
                        else not count_contract[
                            "exact_cross_run_count_requirement_satisfied"
                        ]
                    )
                    if count_rule_failed:
                        stage = time.monotonic()
                        final_hash = self._rollback_active_gaussian_append(transaction)
                        timings["rollback_seconds"] = time.monotonic() - stage
                        audit["status"] = (
                            "forced_commit_candidate_count_out_of_bounds_rolled_back"
                            if cfg.count_agnostic_forced_commit
                            else "forced_commit_candidate_count_mismatch_rolled_back"
                        )
                        audit["rejection_reasons"] = ([
                            f"accepted_count={accepted_count},"
                            f"bounds=[{minimum_count},{maximum_count}]"
                        ] if cfg.count_agnostic_forced_commit else [
                            f"accepted_count={accepted_count},expected={expected_count}"
                        ])
                        audit["active_state_final"] = {
                            "gaussian_count": int(self.gaussians.get_xyz.shape[0]),
                            "sha256": final_hash,
                            "byte_identical_to_premerge": True,
                        }
                        return audit
                    if (
                        int(merge_report.get("before_count", -1)) != before_count
                        or int(merge_report.get("after_count", -1)) != trial_count
                        or not count_contract[
                            "after_minus_before_equals_accepted_count"
                        ]
                    ):
                        raise RuntimeError(
                            "forced-commit count lineage does not equal "
                            "before_count + accepted_count"
                        )
                    committed_state = (
                        self._capture_active_gaussian_transaction_state()
                    )
                    if (
                        int(committed_state["before_count"]) != trial_count
                        or committed_state["sha256"] != trial_state["sha256"]
                        or committed_state["sha256"] == transaction["sha256"]
                    ):
                        raise RuntimeError(
                            "forced-commit state must equal the trial state and "
                            "differ from the premerge state"
                        )
                    audit["status"] = (
                        "postmerge_gate_rejected_forced_commit_unsafe"
                    )
                    audit["active_map_changed_final"] = True
                    audit["rejection_reasons"] = list(post_gate["reasons"])
                    audit["unsafe_forced_commit"] = {
                        "posthoc_after_v2_rejection": True,
                        "posthoc_after_v3_count_mismatch": bool(
                            cfg.posthoc_after_v3_count_mismatch
                        ),
                        "posthoc_after_v4_visibility_cache_mismatch": bool(
                            cfg.posthoc_after_v4_visibility_cache_mismatch
                        ),
                        "unsafe_not_deployable": True,
                        "gate_thresholds_unchanged": True,
                        "postmerge_gate_decision_recorded": True,
                        "postmerge_gate_accepted": False,
                        "ordinary_action": "rollback",
                        "diagnostic_override_action": "commit_without_rollback",
                        "committed_gaussian_count": accepted_count,
                        "count_contract": count_contract,
                        "rollback_performed": False,
                        "source220_and_final100_execution_planned": True,
                        "source220_and_final100_completion_not_yet_asserted": True,
                        "uses_ground_truth": False,
                        "uses_clear_gt_metrics": False,
                    }
                    audit["active_state_final"] = {
                        "gaussian_count": int(committed_state["before_count"]),
                        "sha256": committed_state["sha256"],
                        "byte_identical_to_premerge": False,
                        "byte_identical_to_trial": True,
                    }
                    prepared_visibility_cache = None
                    if cfg.refresh_occ_aware_visibility_after_forced_commit:
                        (
                            prepared_visibility_cache,
                            visibility_cache_refresh,
                        ) = self._prepare_active_fusion_visibility_cache_refresh(
                            before_count=before_count,
                            after_count=trial_count,
                            expected_active_state_sha256=committed_state["sha256"],
                        )
                        timings["visibility_cache_refresh_seconds"] = float(
                            visibility_cache_refresh["elapsed_seconds"]
                        )
                        audit["visibility_cache_refresh"] = visibility_cache_refresh
                        audit["unsafe_forced_commit"][
                            "visibility_cache_refresh_committed_atomically"
                        ] = True
                    # Set only after the complete immutable audit payload has
                    # been constructed.  Any exception above must enter the
                    # generic rollback path.
                    if prepared_visibility_cache is not None:
                        visibility_cache_before_commit = self.occ_aware_visibility
                        self.occ_aware_visibility = prepared_visibility_cache
                        visibility_cache_replaced = True
                        visibility_cache_refresh["status"] = "committed_atomically"
                        visibility_cache_refresh[
                            "atomic_commit_assignment_performed"
                        ] = True
                    committed = True
                    return audit
                stage = time.monotonic()
                final_hash = self._rollback_active_gaussian_append(transaction)
                timings["rollback_seconds"] = time.monotonic() - stage
                audit["status"] = "postmerge_gate_rejected_rolled_back"
                audit["rejection_reasons"] = list(post_gate["reasons"])
                audit["active_state_final"] = {
                    "gaussian_count": int(self.gaussians.get_xyz.shape[0]),
                    "sha256": final_hash,
                    "byte_identical_to_premerge": True,
                }
                return audit

            if cfg.force_commit_after_postmerge_rejection:
                stage = time.monotonic()
                final_hash = self._rollback_active_gaussian_append(transaction)
                timings["rollback_seconds"] = time.monotonic() - stage
                audit["status"] = (
                    "posthoc_v2_rejection_not_reproduced_rolled_back"
                )
                audit["rejection_reasons"] = [
                    "postmerge_gate_accepted_so_forced_commit_not_authorized"
                ]
                audit["active_state_final"] = {
                    "gaussian_count": int(self.gaussians.get_xyz.shape[0]),
                    "sha256": final_hash,
                    "byte_identical_to_premerge": True,
                }
                return audit

            audit["status"] = "accepted"
            audit["active_map_changed_final"] = True
            audit["accepted_map_semantics"] = {
                "mode": "append",
                "original_gaussian_prefix_preserved": True,
                "densification_statistics_reset_after_acceptance": True,
                "optimizer_extended_for_appended_gaussians": True,
            }
            audit["active_state_final"] = {
                "gaussian_count": int(self.gaussians.get_xyz.shape[0]),
                "sha256": trial_state["sha256"],
                "byte_identical_to_premerge": False,
            }
            (
                prepared_visibility_cache,
                visibility_cache_refresh,
            ) = self._prepare_active_fusion_visibility_cache_refresh(
                before_count=int(transaction["before_count"]),
                after_count=int(trial_state["before_count"]),
                expected_active_state_sha256=trial_state["sha256"],
            )
            timings["visibility_cache_refresh_seconds"] = float(
                visibility_cache_refresh["elapsed_seconds"]
            )
            audit["visibility_cache_refresh"] = visibility_cache_refresh
            audit["accepted_map_semantics"][
                "visibility_cache_refresh_committed_atomically"
            ] = True
            visibility_cache_before_commit = self.occ_aware_visibility
            self.occ_aware_visibility = prepared_visibility_cache
            visibility_cache_replaced = True
            visibility_cache_refresh["status"] = "committed_atomically"
            visibility_cache_refresh["atomic_commit_assignment_performed"] = True
            committed = True
            return audit
        except Exception as error:
            audit["status"] = "error_rejected"
            audit["rejection_reasons"] = [
                f"{type(error).__name__}: {error}"
            ]
            if transaction is not None and not committed:
                if visibility_cache_replaced:
                    self.occ_aware_visibility = visibility_cache_before_commit
                    visibility_cache_replaced = False
                audit["active_map_changed_final"] = False
                stage = time.monotonic()
                audit["active_state_final"] = {
                    "gaussian_count": int(transaction["before_count"]),
                    "sha256": self._rollback_active_gaussian_append(transaction),
                    "byte_identical_to_premerge": True,
                }
                timings["rollback_seconds"] += time.monotonic() - stage
            return audit
        finally:
            timings["total_wall_seconds"] = time.monotonic() - started

    @staticmethod
    def _forced_commit_file_sha256(path):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"required forced-commit artifact is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _forced_commit_ply_vertex_count(path):
        path = Path(path).expanduser().resolve()
        with path.open("rb") as handle:
            consumed = 0
            while consumed <= 1024 * 1024:
                line = handle.readline()
                if not line:
                    break
                consumed += len(line)
                try:
                    text = line.decode("ascii").strip()
                except UnicodeDecodeError as error:
                    raise RuntimeError("PLY header is not ASCII") from error
                if text.startswith("element vertex "):
                    value = int(text.split()[-1])
                    if value < 0:
                        raise RuntimeError("PLY vertex count is negative")
                    vertex_count = value
                if text == "end_header":
                    if "vertex_count" not in locals():
                        raise RuntimeError("PLY header lacks element vertex")
                    return vertex_count
        raise RuntimeError("PLY header is missing or exceeds 1 MiB")

    def _forced_commit_chain_root(self):
        return (
            Path(self.official_resplat_active_fusion_cfg.output_root)
            .expanduser()
            .resolve()
            / "forced_commit_chain"
        )

    def _write_forced_commit_event(self, filename, payload):
        event = stamp_contract_sha256(payload)
        path = self._forced_commit_chain_root() / str(filename)
        atomic_write_active_fusion_json(path, event)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        if persisted != event:
            raise RuntimeError("forced-commit event changed during publication")
        return event, path

    @staticmethod
    def _forced_commit_state_record(snapshot):
        return {
            "gaussian_count": int(snapshot["before_count"]),
            "full_active_state_sha256": str(snapshot["sha256"]),
        }

    def _forced_commit_visibility_cache_state_record(self, expected_count):
        expected_count = int(expected_count)
        active_device = self.gaussians.get_xyz.device
        keys = tuple(self.occ_aware_visibility.keys())
        if keys != tuple(self.current_window) or not keys:
            raise RuntimeError(
                "forced-commit visibility cache does not match the live window"
            )
        digest = hashlib.sha256()
        digest.update(json.dumps([int(value) for value in keys]).encode("ascii"))
        records = []
        for frame_id in keys:
            visibility = self.occ_aware_visibility[frame_id]
            if (
                not isinstance(visibility, torch.Tensor)
                or visibility.ndim != 1
                or int(visibility.numel()) != expected_count
                or visibility.device != active_device
                or visibility.dtype != torch.long
                or not visibility.is_contiguous()
            ):
                raise RuntimeError(
                    "forced-commit visibility cache has a stale shape/device/dtype"
                )
            if not bool(
                torch.logical_or(visibility == 0, visibility == 1).all().item()
            ):
                raise RuntimeError("forced-commit visibility cache is non-binary")
            self._fusion_hash_tensor(
                digest, f"occ_aware_visibility.{int(frame_id)}", visibility
            )
            records.append(
                {
                    "frame_id": int(frame_id),
                    "source_index": int(self.cameras[frame_id].timestamp),
                    "vector_length": int(visibility.numel()),
                    "visible_gaussians": int(visibility.count_nonzero().item()),
                    "dtype": str(visibility.dtype),
                    "device": str(visibility.device),
                }
            )
        return {
            "schema": "unblur_slam.occ_aware_visibility_cache_state.v1",
            "keys": [int(value) for value in keys],
            "matches_current_window": True,
            "gaussian_count": expected_count,
            "all_vectors_binary_long_live_device": True,
            "per_camera": records,
            "sha256": digest.hexdigest(),
        }

    def _initialize_forced_commit_chain(self, audit, audit_path):
        """Publish the commit link only after the unsafe override is complete."""

        diagnostic_started = time.monotonic()
        cfg = self.official_resplat_active_fusion_cfg
        if not cfg.force_commit_after_postmerge_rejection:
            return
        if self.official_resplat_forced_commit_chain is not None:
            raise RuntimeError("forced-commit chain was initialized twice")
        if audit.get("status") != "postmerge_gate_rejected_forced_commit_unsafe":
            raise RuntimeError(
                "forced diagnostic did not reproduce the complete post-gate rejection"
            )
        protocol = audit.get("diagnostic_protocol") or {}
        required_flags = (
            "posthoc_after_v2_rejection",
            "unsafe_not_deployable",
            "gate_thresholds_unchanged",
            "force_only_after_postmerge_gate_rejected",
        )
        if any(protocol.get(name) is not True for name in required_flags):
            raise RuntimeError("forced-commit audit disclosure flags are incomplete")
        if cfg.count_agnostic_forced_commit:
            if (
                protocol.get("posthoc_after_v3_count_mismatch") is not True
                or (protocol.get("count_contract") or {}).get("kind")
                != "fresh_run_merge_bounds_and_internal_count_algebra"
                or (protocol.get("count_contract") or {}).get(
                    "cross_run_exact_count_required"
                )
                is not False
            ):
                raise RuntimeError(
                    "v4 count-agnostic diagnostic disclosure is incomplete"
                )
        if cfg.refresh_occ_aware_visibility_after_forced_commit:
            if (
                protocol.get("posthoc_after_v4_visibility_cache_mismatch") is not True
                or protocol.get(
                    "refresh_occ_aware_visibility_after_forced_commit"
                )
                is not True
                or str(protocol.get("v4_visibility_cache_failure_audit_sha256", ""))
                != str(cfg.v4_visibility_cache_failure_audit_sha256)
            ):
                raise RuntimeError(
                    "v5 visibility-cache refresh diagnostic disclosure is incomplete"
                )
        decision = ((audit.get("postmerge_gate") or {}).get("decision") or {})
        if decision.get("accepted") is not False:
            raise RuntimeError("forced commit requires an explicit post-gate rejection")
        before = audit.get("active_state_before") or {}
        trial = audit.get("active_state_trial") or {}
        committed = audit.get("active_state_final") or {}
        merge = audit.get("merge") or {}
        before_count = int(before.get("gaussian_count", -1))
        trial_count = int(trial.get("gaussian_count", -1))
        committed_count = int(committed.get("gaussian_count", -1))
        accepted_count = int(merge.get("accepted_count", -1))
        count_contract = forced_commit_count_contract(
            accepted_count=accepted_count,
            before_count=before_count,
            after_count=trial_count,
            config=cfg,
        )
        if (
            not count_contract["accepted"]
            or int(merge.get("before_count", -1)) != before_count
            or int(merge.get("after_count", -1)) != trial_count
            or trial_count != before_count + accepted_count
            or committed_count != trial_count
        ):
            raise RuntimeError("forced-commit count chain is inconsistent")
        before_sha = str(before.get("sha256", ""))
        trial_sha = str(trial.get("sha256", ""))
        committed_sha = str(committed.get("sha256", ""))
        if trial_sha != committed_sha or trial_sha == before_sha:
            raise RuntimeError(
                "forced-commit full state must equal trial and differ from before"
            )
        if float((audit.get("timing") or {}).get("rollback_seconds", -1.0)) != 0.0:
            raise RuntimeError("forced commit unexpectedly recorded rollback work")
        override = audit.get("unsafe_forced_commit") or {}
        if (
            override.get("rollback_performed") is not False
            or override.get("diagnostic_override_action")
            != "commit_without_rollback"
        ):
            raise RuntimeError("forced-commit override semantics drifted")
        visibility_cache_refresh = audit.get("visibility_cache_refresh")
        committed_visibility_cache = None
        if cfg.refresh_occ_aware_visibility_after_forced_commit:
            if not isinstance(visibility_cache_refresh, dict) or (
                visibility_cache_refresh.get("schema")
                != "unblur_slam.active_fusion_visibility_cache_refresh.v1"
                or visibility_cache_refresh.get("status") != "committed_atomically"
                or visibility_cache_refresh.get("accepted") is not True
                or visibility_cache_refresh.get("atomic_replacement") is not True
                or visibility_cache_refresh.get(
                    "atomic_commit_assignment_performed"
                )
                is not True
                or visibility_cache_refresh.get("zero_padding_used") is not False
                or visibility_cache_refresh.get("padding_or_truncation_used")
                is not False
                or visibility_cache_refresh.get(
                    "all_values_derived_from_fresh_active_map_renders"
                )
                is not True
                or visibility_cache_refresh.get(
                    "deblur_fallback_uses_all_virtual_views_max_n_touched"
                )
                is not True
                or visibility_cache_refresh.get(
                    "key_membership_and_order_unchanged"
                )
                is not True
                or int(
                    visibility_cache_refresh.get("before_gaussian_count", -1)
                )
                != before_count
                or int(visibility_cache_refresh.get("after_gaussian_count", -1))
                != committed_count
            ):
                raise RuntimeError(
                    "forced-commit visibility-cache refresh audit is incomplete"
                )
            keys_before = list(visibility_cache_refresh.get("keys_before") or [])
            keys_after = list(visibility_cache_refresh.get("keys_after") or [])
            per_camera = list(visibility_cache_refresh.get("per_camera") or [])
            if (
                keys_before != keys_after
                or keys_after != [int(value) for value in self.current_window]
                or [int(item.get("frame_id", -1)) for item in per_camera]
                != keys_after
                or any(
                    int(item.get("old_vector_length", -1)) != before_count
                    or int(item.get("new_vector_length", -1)) != committed_count
                    for item in per_camera
                )
            ):
                raise RuntimeError(
                    "forced-commit visibility-cache per-camera lineage drifted"
                )
            committed_visibility_cache = (
                self._forced_commit_visibility_cache_state_record(committed_count)
            )
            if (
                committed_visibility_cache["keys"] != keys_after
                or [
                    int(item["visible_gaussians"])
                    for item in committed_visibility_cache["per_camera"]
                ]
                != [
                    int(item.get("new_visible_gaussians", -1))
                    for item in per_camera
                ]
            ):
                raise RuntimeError(
                    "live committed visibility cache differs from the refresh audit"
                )
        audit_path = Path(audit_path).expanduser().resolve()
        audit_sha = self._forced_commit_file_sha256(audit_path)
        event_payload = {
            "schema": "unblur_slam.forced_commit_chain_event.v1",
            "event_type": "postgate_rejected_forced_commit",
            "sequence": 0,
            "previous_event_sha256": None,
            "source_index": 166,
            "posthoc_after_v2_rejection": True,
            "posthoc_after_v3_count_mismatch": bool(
                cfg.posthoc_after_v3_count_mismatch
            ),
            "posthoc_after_v4_visibility_cache_mismatch": bool(
                cfg.posthoc_after_v4_visibility_cache_mismatch
            ),
            "unsafe_not_deployable": True,
            "gate_thresholds_unchanged": True,
            "ordinary_v2_action": "rollback",
            "diagnostic_v3_action": (
                None
                if cfg.count_agnostic_forced_commit
                else "commit_without_rollback"
            ),
            "prior_v3_observed_action": (
                "rollback_after_exact_count_mismatch"
                if cfg.count_agnostic_forced_commit
                else None
            ),
            "diagnostic_v4_action": (
                "commit_without_rollback"
                if cfg.count_agnostic_forced_commit
                else None
            ),
            "prior_v4_observed_action": (
                "commit_then_source220_visibility_cache_length_mismatch"
                if cfg.refresh_occ_aware_visibility_after_forced_commit
                else None
            ),
            "diagnostic_v5_action": (
                "fresh_render_visibility_cache_then_commit_without_rollback"
                if cfg.refresh_occ_aware_visibility_after_forced_commit
                else None
            ),
            "diagnostic_revision": (
                "v5_visibility_cache_refresh_forced_commit"
                if cfg.refresh_occ_aware_visibility_after_forced_commit
                else (
                    "v4_count_agnostic_forced_commit"
                    if cfg.count_agnostic_forced_commit
                    else "v3_forced_commit"
                )
            ),
            "postmerge_gate_accepted": False,
            "postmerge_gate_reasons": list(decision.get("reasons") or []),
            "rollback_performed": False,
            "rollback_seconds": 0.0,
            "accepted_gaussian_count": accepted_count,
            "count_contract": count_contract,
            "before_state": {
                "gaussian_count": before_count,
                "full_active_state_sha256": before_sha,
            },
            "trial_state": {
                "gaussian_count": trial_count,
                "full_active_state_sha256": trial_sha,
            },
            "committed_state": {
                "gaussian_count": committed_count,
                "full_active_state_sha256": committed_sha,
            },
            "trial_equals_committed": True,
            "committed_differs_from_before": True,
            "fusion_audit": {
                "path": str(audit_path),
                "sha256": audit_sha,
            },
            "v2_rejection_audit_sha256_preregistered": str(
                cfg.v2_rejection_audit_sha256
            ),
            "v3_count_mismatch_audit_sha256_preregistered": str(
                cfg.v3_count_mismatch_audit_sha256
            ),
            "v4_visibility_cache_failure_audit_sha256_preregistered": str(
                cfg.v4_visibility_cache_failure_audit_sha256
            ),
            "visibility_cache_refresh": visibility_cache_refresh,
            "committed_visibility_cache_state": committed_visibility_cache,
            "ground_truth_pose_depth_or_clear_pixels_used_for_decision": False,
            "individual_imported_gaussian_survival_not_claimed": True,
        }
        event, event_path = self._write_forced_commit_event(
            "00_forced_commit.json", event_payload
        )
        self.official_resplat_forced_commit_chain = {
            "commit_event": event,
            "commit_event_path": str(event_path),
            "source220_entry": None,
            "source220_event": None,
            "source220_event_path": None,
            "final100_entry": None,
            "final100_event": None,
            "final100_event_path": None,
            "terminal_contract": None,
            "diagnostic_overhead_seconds": {},
        }
        elapsed = time.monotonic() - diagnostic_started
        self.official_resplat_forced_commit_chain[
            "diagnostic_overhead_seconds"
        ]["commit_validation_and_event_publication"] = elapsed
        self.slam.timing_stats[
            "forced_commit_chain_commit_event_seconds"
        ] = elapsed

    def _forced_commit_source220_entry(self, source_index, frame_id):
        """Capture state immediately after receiving source220's message."""

        diagnostic_started = time.monotonic()
        cfg = self.official_resplat_active_fusion_cfg
        if not cfg.force_commit_after_postmerge_rejection:
            return
        chain = self.official_resplat_forced_commit_chain
        if chain is None:
            # The commit happens at source166, so earlier mapper messages do not
            # have a chain yet.  After that point absence is a hard failure.
            if int(source_index) > int(cfg.trigger_source_index):
                raise RuntimeError("source220 arrived without a forced commit chain")
            return
        if chain["source220_entry"] is not None:
            raise RuntimeError("received another mapping frame after source220 entry")
        if int(source_index) != 220 or int(frame_id) != 10:
            raise RuntimeError(
                "the first mapper message after forced commit must be source220/frame10"
            )
        snapshot = self._capture_active_gaussian_transaction_state()
        observed = self._forced_commit_state_record(snapshot)
        expected = chain["commit_event"]["committed_state"]
        if observed != expected:
            raise RuntimeError(
                "source220 entry state does not equal the forced committed state"
            )
        observed_visibility_cache = None
        if cfg.refresh_occ_aware_visibility_after_forced_commit:
            expected_visibility_cache = chain["commit_event"].get(
                "committed_visibility_cache_state"
            )
            if not isinstance(expected_visibility_cache, dict):
                raise RuntimeError(
                    "v5 commit event lacks its visibility-cache state binding"
                )
            observed_visibility_cache = (
                self._forced_commit_visibility_cache_state_record(
                    expected["gaussian_count"]
                )
            )
            if observed_visibility_cache != expected_visibility_cache:
                raise RuntimeError(
                    "source220 entry visibility cache differs from the committed "
                    "v5 refresh state"
                )
        chain["source220_entry"] = {
            "source_index": 220,
            "frame_id": 10,
            "iteration_count": int(self.iteration_count),
            "active_state": observed,
            "capture_point": (
                "immediately_after_pipe_recv_and_end_check_before_frame_metadata_"
                "pose_depth_deformation_or_mapping"
            ),
            "captured_before_any_source220_map_mutation": True,
            "visibility_cache_state": observed_visibility_cache,
            "visibility_cache_equals_committed_refresh_state": (
                True
                if cfg.refresh_occ_aware_visibility_after_forced_commit
                else None
            ),
        }
        elapsed = time.monotonic() - diagnostic_started
        chain["source220_entry"]["diagnostic_state_capture_seconds"] = elapsed
        chain["diagnostic_overhead_seconds"]["source220_entry_capture"] = elapsed
        self.slam.timing_stats[
            "forced_commit_chain_source220_entry_seconds"
        ] = elapsed

    def _forced_commit_source220_complete(self, viewpoint):
        diagnostic_started = time.monotonic()
        cfg = self.official_resplat_active_fusion_cfg
        if not cfg.force_commit_after_postmerge_rejection:
            return
        chain = self.official_resplat_forced_commit_chain
        if chain is None or chain["source220_entry"] is None:
            raise RuntimeError("source220 completed without an entry-state capture")
        if chain["source220_event"] is not None:
            raise RuntimeError("source220 completion was recorded twice")
        if int(viewpoint.timestamp) != 220 or int(viewpoint.uid) != 10:
            raise RuntimeError("forced-commit downstream mapping must be source220/frame10")
        if int(self.mapping_itr_num) != 100:
            raise RuntimeError("forced-commit diagnostic requires 100 online map steps")
        entry = chain["source220_entry"]
        iteration_delta = int(self.iteration_count) - int(entry["iteration_count"])
        if iteration_delta != 101:
            raise RuntimeError(
                "source220 must complete 100 mapping iterations plus one prune pass"
            )
        post = self._capture_active_gaussian_transaction_state()
        post_record = self._forced_commit_state_record(post)
        payload = {
            "schema": "unblur_slam.forced_commit_chain_event.v1",
            "event_type": "source220_mapping_complete",
            "sequence": 1,
            "previous_event_sha256": chain["commit_event"]["event_sha256"],
            "source_index": 220,
            "frame_id": 10,
            "entry_capture_point": entry["capture_point"],
            "entry_captured_before_any_source220_map_mutation": True,
            "entry_state_capture_seconds": float(
                entry["diagnostic_state_capture_seconds"]
            ),
            "entry_state": entry["active_state"],
            "entry_state_equals_committed_state": True,
            "entry_visibility_cache_state": entry.get("visibility_cache_state"),
            "entry_visibility_cache_equals_committed_refresh_state": entry.get(
                "visibility_cache_equals_committed_refresh_state"
            ),
            "mapping_iterations_completed": 100,
            "prune_passes_completed": 1,
            "iteration_count_before": int(entry["iteration_count"]),
            "iteration_count_after": int(self.iteration_count),
            "iteration_count_delta": iteration_delta,
            "post_mapping_state": post_record,
            "committed_import_batch_was_live_at_source220_entry": True,
            "committed_import_batch_participated_in_downstream_map_computation": True,
            "individual_imported_gaussian_survival_after_mapping_not_tracked": True,
            "posthoc_after_v2_rejection": True,
            "posthoc_after_v3_count_mismatch": bool(
                cfg.posthoc_after_v3_count_mismatch
            ),
            "posthoc_after_v4_visibility_cache_mismatch": bool(
                cfg.posthoc_after_v4_visibility_cache_mismatch
            ),
            "diagnostic_revision": (
                "v5_visibility_cache_refresh_forced_commit"
                if cfg.refresh_occ_aware_visibility_after_forced_commit
                else (
                    "v4_count_agnostic_forced_commit"
                    if cfg.count_agnostic_forced_commit
                    else "v3_forced_commit"
                )
            ),
            "unsafe_not_deployable": True,
            "uses_ground_truth_for_chain_acceptance": False,
        }
        event, path = self._write_forced_commit_event(
            "01_source220_complete.json", payload
        )
        chain["source220_event"] = event
        chain["source220_event_path"] = str(path)
        elapsed = time.monotonic() - diagnostic_started
        chain["diagnostic_overhead_seconds"][
            "source220_complete_capture_and_event_publication"
        ] = elapsed
        self.slam.timing_stats[
            "forced_commit_chain_source220_complete_seconds"
        ] = elapsed

    def _forced_commit_final100_entry(self, configured_iters):
        diagnostic_started = time.monotonic()
        cfg = self.official_resplat_active_fusion_cfg
        if not cfg.force_commit_after_postmerge_rejection:
            return
        chain = self.official_resplat_forced_commit_chain
        if chain is None or chain["source220_event"] is None:
            raise RuntimeError("final100 began before source220 completion was proven")
        if chain["final100_entry"] is not None:
            raise RuntimeError("final100 entry was recorded twice")
        if int(configured_iters) != 100:
            raise RuntimeError("forced-commit diagnostic requires final_refine_iters=100")
        snapshot = self._capture_active_gaussian_transaction_state()
        observed = self._forced_commit_state_record(snapshot)
        expected = chain["source220_event"]["post_mapping_state"]
        if observed != expected:
            raise RuntimeError("final100 entry state does not equal source220 post-state")
        chain["final100_entry"] = {
            "configured_iterations": 100,
            "iteration_count": int(self.iteration_count),
            "active_state": observed,
            "capture_point": (
                "before_final_refine_hydration_pose_depth_deformation_or_optimizer_work"
            ),
        }
        elapsed = time.monotonic() - diagnostic_started
        chain["final100_entry"]["diagnostic_state_capture_seconds"] = elapsed
        chain["diagnostic_overhead_seconds"]["final100_entry_capture"] = elapsed
        self.slam.timing_stats[
            "forced_commit_chain_final100_entry_seconds"
        ] = elapsed

    def _forced_commit_final100_complete(self, configured_iters, total_iters):
        diagnostic_started = time.monotonic()
        cfg = self.official_resplat_active_fusion_cfg
        if not cfg.force_commit_after_postmerge_rejection:
            return
        chain = self.official_resplat_forced_commit_chain
        if chain is None or chain["final100_entry"] is None:
            raise RuntimeError("final100 completed without an entry-state capture")
        if chain["final100_event"] is not None:
            raise RuntimeError("final100 completion was recorded twice")
        if int(configured_iters) != 100 or int(total_iters) != 100:
            raise RuntimeError("forced-commit diagnostic must execute exactly final100")
        entry = chain["final100_entry"]
        iteration_delta = int(self.iteration_count) - int(entry["iteration_count"])
        if iteration_delta != 100:
            raise RuntimeError("final100 iteration counter did not advance by exactly 100")
        post = self._capture_active_gaussian_transaction_state()
        post_record = self._forced_commit_state_record(post)
        checkpoint = (
            Path(self.save_dir)
            / "refinement_checkpoints"
            / "iter_000100"
            / "point_cloud.ply"
        ).resolve()
        checkpoint_count = self._forced_commit_ply_vertex_count(checkpoint)
        if checkpoint_count != int(post_record["gaussian_count"]):
            raise RuntimeError("final100 checkpoint PLY count differs from live map")
        payload = {
            "schema": "unblur_slam.forced_commit_chain_event.v1",
            "event_type": "final100_complete",
            "sequence": 2,
            "previous_event_sha256": chain["source220_event"]["event_sha256"],
            "entry_capture_point": entry["capture_point"],
            "entry_state": entry["active_state"],
            "entry_state_capture_seconds": float(
                entry["diagnostic_state_capture_seconds"]
            ),
            "entry_state_equals_source220_post_state": True,
            "configured_iterations": 100,
            "executed_iterations": 100,
            "iteration_count_before": int(entry["iteration_count"]),
            "iteration_count_after": int(self.iteration_count),
            "iteration_count_delta": iteration_delta,
            "post_refinement_state": post_record,
            "checkpoint_ply": {
                "path": str(checkpoint),
                "sha256": self._forced_commit_file_sha256(checkpoint),
                "vertex_count": checkpoint_count,
            },
            "committed_import_batch_participated_in_final100_computation": True,
            "individual_imported_gaussian_survival_after_final100_not_tracked": True,
            "posthoc_after_v2_rejection": True,
            "posthoc_after_v3_count_mismatch": bool(
                cfg.posthoc_after_v3_count_mismatch
            ),
            "posthoc_after_v4_visibility_cache_mismatch": bool(
                cfg.posthoc_after_v4_visibility_cache_mismatch
            ),
            "diagnostic_revision": (
                "v5_visibility_cache_refresh_forced_commit"
                if cfg.refresh_occ_aware_visibility_after_forced_commit
                else (
                    "v4_count_agnostic_forced_commit"
                    if cfg.count_agnostic_forced_commit
                    else "v3_forced_commit"
                )
            ),
            "unsafe_not_deployable": True,
            "uses_ground_truth_for_chain_acceptance": False,
        }
        event, path = self._write_forced_commit_event(
            "02_final100_complete.json", payload
        )
        chain["final100_event"] = event
        chain["final100_event_path"] = str(path)
        elapsed = time.monotonic() - diagnostic_started
        chain["diagnostic_overhead_seconds"][
            "final100_complete_capture_and_event_publication"
        ] = elapsed
        self.slam.timing_stats[
            "forced_commit_chain_final100_complete_seconds"
        ] = elapsed

    def finalize_forced_commit_terminal(self, final_ply_path):
        """Bind final serialization to the completed unsafe diagnostic chain."""

        terminal_started = time.monotonic()
        cfg = self.official_resplat_active_fusion_cfg
        if not cfg.force_commit_after_postmerge_rejection:
            return
        chain = self.official_resplat_forced_commit_chain
        if chain is None or chain["final100_event"] is None:
            raise RuntimeError("cannot serialize forced diagnostic without final100 proof")
        if chain["terminal_contract"] is not None:
            raise RuntimeError("forced-commit terminal contract was written twice")
        final_state = self._capture_active_gaussian_transaction_state()
        final_state_record = self._forced_commit_state_record(final_state)
        expected_state = chain["final100_event"]["post_refinement_state"]
        if final_state_record != expected_state:
            raise RuntimeError("serialized final-map state differs from final100 state")
        final_ply = Path(final_ply_path).expanduser().resolve()
        final_ply_sha = self._forced_commit_file_sha256(final_ply)
        final_vertex_count = self._forced_commit_ply_vertex_count(final_ply)
        if final_vertex_count != int(final_state_record["gaussian_count"]):
            raise RuntimeError("final_model.ply vertex count differs from live map")
        checkpoint = chain["final100_event"]["checkpoint_ply"]
        if (
            str(checkpoint["sha256"]) != final_ply_sha
            or int(checkpoint["vertex_count"]) != final_vertex_count
        ):
            raise RuntimeError(
                "iter100 checkpoint PLY is not byte-identical to final_model.ply"
            )

        audit = self.official_resplat_active_fusion_audit or {}
        published = audit.get("published_result") or {}
        manifest_path = Path(str(published.get("manifest_path", ""))).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative_world = Path(
            str((manifest.get("outputs") or {}).get("unblur_world_gaussians_npz", ""))
        )
        if relative_world.is_absolute() or ".." in relative_world.parts:
            raise RuntimeError("terminal world artifact path escaped result root")
        world_path = Path(str(published.get("path", ""))).resolve() / relative_world
        relative_native = Path(
            str((manifest.get("outputs") or {}).get("native_gaussians_npz", ""))
        )
        if relative_native.is_absolute() or ".." in relative_native.parts:
            raise RuntimeError("terminal native artifact path escaped result root")
        native_path = (
            Path(str(published.get("path", ""))).resolve() / relative_native
        )
        snapshot_manifest = (
            Path(str((audit.get("snapshot") or {}).get("path", ""))).resolve()
            / "snapshot_manifest.json"
        )
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = (Path(self.save_dir) / "cfg.yaml").resolve()
        metrics_path = (
            Path(self.save_dir) / "psnr" / "after_refine" / "final_result.json"
        ).resolve()
        fusion_audit_path = (
            Path(cfg.output_root).expanduser().resolve() / "fusion_audit.json"
        )
        final_contract_path = (
            Path(cfg.output_root).expanduser().resolve()
            / "fusion_final_contract.json"
        )
        artifact_paths = {
            "resolved_config": cfg_path,
            "fusion_audit": fusion_audit_path,
            "fusion_final_contract": final_contract_path,
            "snapshot_manifest": snapshot_manifest,
            "official_resplat_manifest": manifest_path,
            "official_resplat_native_gaussians": native_path,
            "official_resplat_world_gaussians": world_path,
            "final_metrics": metrics_path,
            "runtime_stats": (Path(self.save_dir) / "runtime_stats.json").resolve(),
            "final_model_ply": final_ply,
        }
        artifact_bindings = {
            name: {
                "path": str(path),
                "sha256": self._forced_commit_file_sha256(path),
            }
            for name, path in artifact_paths.items()
        }
        code_paths = {
            "mapper": repo_root / "src" / "mapper.py",
            "active_fusion_helper": (
                repo_root / "src" / "refinement" / "official_resplat_active_fusion.py"
            ),
            "active_map_merge": (
                repo_root / "src" / "refinement" / "active_map_merge.py"
            ),
            "sidecar_bridge": (
                repo_root / "src" / "refinement" / "official_resplat_sidecar.py"
            ),
            "world_bridge": (
                repo_root / "src" / "refinement" / "resplat_unblur_bridge.py"
            ),
            "official_sidecar_runner": (
                repo_root / "scripts" / "run_official_resplat_sidecar.py"
            ),
            "gaussian_model": (
                repo_root
                / "thirdparty"
                / "gaussian_splatting"
                / "scene"
                / "gaussian_model.py"
            ),
            "slam_terminal_writer": repo_root / "src" / "slam.py",
        }
        code_bindings = {
            name: {
                "path": str(path.resolve()),
                "sha256": self._forced_commit_file_sha256(path),
            }
            for name, path in code_paths.items()
        }
        terminal_payload = {
            "schema": "unblur_slam.forced_commit_terminal_contract.v1",
            "status": "complete_unsafe_posthoc_diagnostic",
            "previous_event_sha256": chain["final100_event"]["event_sha256"],
            "chain": {
                "commit_event_sha256": chain["commit_event"]["event_sha256"],
                "source220_event_sha256": chain["source220_event"]["event_sha256"],
                "final100_event_sha256": chain["final100_event"]["event_sha256"],
                "tip_sha256": chain["final100_event"]["event_sha256"],
            },
            "posthoc_after_v2_rejection": True,
            "posthoc_after_v3_count_mismatch": bool(
                cfg.posthoc_after_v3_count_mismatch
            ),
            "posthoc_after_v4_visibility_cache_mismatch": bool(
                cfg.posthoc_after_v4_visibility_cache_mismatch
            ),
            "diagnostic_revision": (
                "v5_visibility_cache_refresh_forced_commit"
                if cfg.refresh_occ_aware_visibility_after_forced_commit
                else (
                    "v4_count_agnostic_forced_commit"
                    if cfg.count_agnostic_forced_commit
                    else "v3_forced_commit"
                )
            ),
            "unsafe_not_deployable": True,
            "gate_thresholds_unchanged": True,
            "merge_filters_unchanged": True,
            "v3_reran_official_resplat": not bool(
                cfg.count_agnostic_forced_commit
            ),
            "v2_resplat_runtime_artifacts_reused": False,
            "v4_reran_official_resplat": bool(
                cfg.count_agnostic_forced_commit
                and not cfg.refresh_occ_aware_visibility_after_forced_commit
            ),
            "v3_resplat_runtime_artifacts_reused": False,
            "v5_reran_official_resplat": bool(
                cfg.refresh_occ_aware_visibility_after_forced_commit
            ),
            "v4_resplat_runtime_artifacts_reused": False,
            "count_contract": dict(chain["commit_event"]["count_contract"]),
            "v3_count_mismatch_audit_sha256_preregistered": str(
                cfg.v3_count_mismatch_audit_sha256
            ),
            "v4_visibility_cache_failure_audit_sha256_preregistered": str(
                cfg.v4_visibility_cache_failure_audit_sha256
            ),
            "visibility_cache_refresh": chain["commit_event"].get(
                "visibility_cache_refresh"
            ),
            "committed_visibility_cache_state": chain["commit_event"].get(
                "committed_visibility_cache_state"
            ),
            "source220_entry_visibility_cache_equals_committed_refresh_state": (
                chain["source220_event"].get(
                    "entry_visibility_cache_equals_committed_refresh_state"
                )
            ),
            "source220_mapping_completed": True,
            "final100_completed": True,
            "final_serialization_completed": True,
            "final_in_memory_state": final_state_record,
            "final_model_ply": {
                "path": str(final_ply),
                "sha256": final_ply_sha,
                "vertex_count": final_vertex_count,
                "byte_identical_to_iter100_checkpoint": True,
            },
            "artifact_bindings": artifact_bindings,
            "code_bindings": code_bindings,
            "committed_import_batch_proven_live_at_source220_entry": True,
            "committed_import_batch_participated_in_source220_and_final100": True,
            "individual_imported_gaussian_survival_in_final_model_not_tracked": True,
            "individual_imported_gaussian_survival_in_final_model_not_claimed": True,
            "uses_gt_for_forced_commit_decision": False,
            "clear_gt_metrics_bound_posthoc_for_evaluation": True,
            "clear_gt_values_used_for_commit_or_checkpoint_selection": False,
            "timing_disclosure": {
                "online_time_includes_commit_and_source220_chain_overhead": True,
                "online_time_includes_visibility_cache_refresh": bool(
                    cfg.refresh_occ_aware_visibility_after_forced_commit
                ),
                "total_time_includes_commit_source220_and_final100_chain_overhead": True,
                "diagnostic_overhead_not_subtracted_from_reported_times": True,
                "recorded_stage_seconds": dict(
                    chain["diagnostic_overhead_seconds"]
                ),
            },
        }
        terminal = stamp_contract_sha256(
            terminal_payload, digest_field="terminal_sha256"
        )
        terminal_path = (
            self._forced_commit_chain_root() / "terminal_contract.json"
        )
        atomic_write_active_fusion_json(terminal_path, terminal)
        chain["diagnostic_overhead_seconds"][
            "terminal_validation_and_publication"
        ] = time.monotonic() - terminal_started
        chain["terminal_contract"] = terminal

    def _active_fusion_after_mapped_keyframe(self, viewpoint):
        cfg = self.official_resplat_active_fusion_cfg
        if not cfg.enabled:
            return
        frame_id = int(viewpoint.uid)
        source_index = int(viewpoint.timestamp)
        existing = [item[0] for item in self.official_resplat_active_fusion_mapped_keyframes]
        if frame_id in existing:
            return
        self.official_resplat_active_fusion_mapped_keyframes.append(
            (frame_id, source_index)
        )
        count = len(self.official_resplat_active_fusion_mapped_keyframes)
        if count < cfg.trigger_keyframe_count:
            return
        if self.official_resplat_active_fusion_attempts:
            self._forced_commit_source220_complete(viewpoint)
            return
        if count != cfg.trigger_keyframe_count or source_index != cfg.trigger_source_index:
            raise RuntimeError(
                "active ReSplat fusion trigger drifted from mapped source-166"
            )
        expected_sources = tuple(int(value) for value in cfg.expected_mapped_source_indices)
        observed_sources = tuple(
            source for _, source in self.official_resplat_active_fusion_mapped_keyframes
        )
        if observed_sources != expected_sources:
            raise RuntimeError(
                "active ReSplat first-eight actually-mapped source schedule drifted: "
                f"{observed_sources}"
            )
        self.official_resplat_active_fusion_attempts += 1
        frame_ids = [
            frame for frame, _ in self.official_resplat_active_fusion_mapped_keyframes
        ]
        source_indices = [
            source for _, source in self.official_resplat_active_fusion_mapped_keyframes
        ]
        audit = self._run_synchronous_active_resplat_fusion(
            frame_ids, source_indices
        )
        self.official_resplat_active_fusion_audit = audit
        audit_path = (
            Path(cfg.output_root).expanduser().resolve() / "fusion_audit.json"
        )
        atomic_write_active_fusion_json(audit_path, audit)
        self._initialize_forced_commit_chain(audit, audit_path)
        timing = audit.get("timing") or {}
        self.slam.timing_stats["official_resplat_active_fusion_enabled"] = True
        self.slam.timing_stats["official_resplat_active_fusion_status"] = audit.get(
            "status"
        )
        self.slam.timing_stats[
            "official_resplat_active_fusion_total_wall_seconds"
        ] = float(timing.get("total_wall_seconds", 0.0))
        self.slam.timing_stats[
            "official_resplat_active_fusion_subprocess_seconds"
        ] = float(timing.get("subprocess_and_publication_seconds", 0.0))
        self.slam.timing_stats[
            "official_resplat_active_fusion_merge_seconds"
        ] = float(timing.get("merge_seconds", 0.0))
        self.slam.timing_stats[
            "official_resplat_active_fusion_visibility_cache_refresh_seconds"
        ] = float(timing.get("visibility_cache_refresh_seconds", 0.0))
        self.slam.timing_stats[
            "official_resplat_active_fusion_rollback_seconds"
        ] = float(timing.get("rollback_seconds", 0.0))
        self.printer.print(
            "official ReSplat state3 active fusion: "
            f"status={audit.get('status')} wall={timing.get('total_wall_seconds', 0.0):.3f}s",
            FontColor.MAPPER,
        )

    def _finalize_active_fusion_contract(self):
        cfg = self.official_resplat_active_fusion_cfg
        if not cfg.enabled:
            return
        if self.official_resplat_active_fusion_attempts != 1:
            raise RuntimeError(
                "enabled fixed-11KF active fusion did not attempt exactly once"
            )
        if self.official_resplat_active_fusion_audit is None:
            raise RuntimeError("enabled active fusion produced no immutable audit")
        if len(self.official_resplat_active_fusion_mapped_keyframes) < 8:
            raise RuntimeError("active fusion ran before eight mapped keyframes")
        observed_sources = tuple(
            int(source)
            for _, source in self.official_resplat_active_fusion_mapped_keyframes
        )
        expected_all_mapped = (
            15,
            49,
            58,
            72,
            89,
            109,
            125,
            166,
            220,
        )
        if observed_sources != expected_all_mapped:
            raise RuntimeError(
                "fixed-11KF actually-mapped sequence drifted: "
                f"{observed_sources}"
            )
        cfg_root = Path(cfg.output_root).expanduser().resolve()
        audit_path = cfg_root / "fusion_audit.json"
        if not audit_path.is_file():
            raise RuntimeError("active fusion audit disappeared before finalization")
        forced_chain = self.official_resplat_forced_commit_chain
        if cfg.force_commit_after_postmerge_rejection:
            if (
                self.official_resplat_active_fusion_audit.get("status")
                != "postmerge_gate_rejected_forced_commit_unsafe"
                or forced_chain is None
                or forced_chain.get("source220_event") is None
            ):
                raise RuntimeError(
                    "forced diagnostic online phase lacks a commit/source220 "
                    "hash-chain proof"
                )
            if cfg.refresh_occ_aware_visibility_after_forced_commit and (
                forced_chain["source220_event"].get(
                    "entry_visibility_cache_equals_committed_refresh_state"
                )
                is not True
            ):
                raise RuntimeError(
                    "v5 online phase lacks the source220 visibility-cache proof"
                )
        if forced_chain is not None and forced_chain.get("source220_event") is not None:
            contract_note = (
                "the source220 chain event proves a committed batch was live at "
                "source220 entry and participated in that mapping computation; "
                "final100 is still pending here and individual imported-Gaussian "
                "survival after prune/training is not tracked"
            )
        else:
            contract_note = (
                "this online-phase contract records the immediate fusion decision; "
                "individual imported-Gaussian survival in the serialized final model "
                "is not separately tracked"
            )
        final_contract = {
            "schema": "unblur_slam.official_resplat_active_fusion_final_contract.v1",
            "fusion_attempt_count": int(self.official_resplat_active_fusion_attempts),
            "actually_mapped_source_indices": list(observed_sources),
            "trigger_context_source_indices": list(observed_sources[:8]),
            "trigger_source_index": int(observed_sources[7]),
            "downstream_online_mapped_source_indices_after_fusion": list(
                observed_sources[8:]
            ),
            "fusion_completed_before_downstream_online_mapping": True,
            "final_refinement_not_started": True,
            "fusion_status": self.official_resplat_active_fusion_audit.get("status"),
            "fusion_committed_to_active_map_before_downstream_mapping": (
                self.official_resplat_active_fusion_audit.get("status")
                in {
                    "accepted",
                    "postmerge_gate_rejected_forced_commit_unsafe",
                }
            ),
            "posthoc_after_v2_rejection": bool(cfg.posthoc_after_v2_rejection),
            "posthoc_after_v3_count_mismatch": bool(
                cfg.posthoc_after_v3_count_mismatch
            ),
            "posthoc_after_v4_visibility_cache_mismatch": bool(
                cfg.posthoc_after_v4_visibility_cache_mismatch
            ),
            "count_agnostic_forced_commit": bool(
                cfg.count_agnostic_forced_commit
            ),
            "refresh_occ_aware_visibility_after_forced_commit": bool(
                cfg.refresh_occ_aware_visibility_after_forced_commit
            ),
            "diagnostic_revision": (
                "v5_visibility_cache_refresh_forced_commit"
                if cfg.refresh_occ_aware_visibility_after_forced_commit
                else (
                    "v4_count_agnostic_forced_commit"
                    if cfg.count_agnostic_forced_commit
                    else "v3_forced_commit"
                )
            ),
            "v4_visibility_cache_failure_audit_sha256_preregistered": str(
                cfg.v4_visibility_cache_failure_audit_sha256
            ),
            "unsafe_not_deployable": bool(cfg.unsafe_not_deployable),
            "gate_thresholds_unchanged": bool(cfg.gate_thresholds_unchanged),
            "source220_downstream_mapping_completed": bool(
                forced_chain is not None
                and forced_chain.get("source220_event") is not None
            ),
            "source220_chain_event_sha256": (
                forced_chain["source220_event"]["event_sha256"]
                if forced_chain is not None
                and forced_chain.get("source220_event") is not None
                else None
            ),
            "source220_entry_visibility_cache_equals_committed_refresh_state": (
                forced_chain["source220_event"].get(
                    "entry_visibility_cache_equals_committed_refresh_state"
                )
                if forced_chain is not None
                and forced_chain.get("source220_event") is not None
                else None
            ),
            "final_refinement_completion_pending_at_contract_write": True,
            "imported_gaussian_survival_in_serialized_final_model_not_separately_tracked": True,
            "note": contract_note,
            "fusion_audit_path": str(audit_path),
            "fusion_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "selection_membership_clear_gt_conditioned": True,
            "ground_truth_poses_or_depths_consumed_by_fusion": False,
            "independent_clear_pixels_consumed_by_fusion": False,
            "clear_gt_metrics_consumed_by_fusion": False,
        }
        atomic_write_active_fusion_json(
            cfg_root / "fusion_final_contract.json", final_contract
        )
        
    def scale_judge(self, mode, iteration):
        if mode == "final":
            stage1_end = 5000
            stage2_end = 10000
        elif mode == "mapping":
            stage1_end = 30
            stage2_end = 50
        elif mode == "init":
            stage1_end = 500
            stage2_end = 800

        if mode == "tracking":
            current_scale = 3
        else:
            if iteration < stage1_end:
                # Stage 1: Small kernel (5x5)
                current_scale = 3
                
            elif iteration < stage2_end:
                # Stage 2: Medium kernel (9x9)
                current_scale = 2
            else:
                # Stage 3: Large kernel (17x17)
                current_scale = 1
        return current_scale

    def set_hyperparams(self):
        mapping_config = self.config["mapping"]

        self.gt_camera = mapping_config["Training"]["gt_camera"]

        self.init_itr_num = mapping_config["Training"]["init_itr_num"]
        self.init_gaussian_update = mapping_config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = mapping_config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = mapping_config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * mapping_config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = mapping_config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = mapping_config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = mapping_config["Training"]["gaussian_update_offset"]
        self.gaussian_th = mapping_config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * mapping_config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = mapping_config["Training"]["gaussian_reset"]
        self.size_threshold = mapping_config["Training"]["size_threshold"]
        self.window_size = mapping_config["Training"]["window_size"]

        self.save_dir = self.config['data']['output'] + '/' + self.config['scene']

        self.move_points = self.config['mapping']['move_points']
        self.online_plotting = self.config['mapping']['online_plotting']
        self.render_videos = self.config['mapping']['online_plotting']

        

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        # This function computes the new Gaussians to be added given a new keyframe
        if (
            getattr(viewpoint, "synthetic", False)
            and not self.config.get("framecrafter", {}).get(
                "inject_gaussians", False
            )
        ):
            # Generated views supervise the existing map by default. Their
            # gated depth is useful for RGB-D loss, but injecting a full PCD at
            # weak confidence can duplicate or hallucinate geometry.
            return
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )


    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.cameras = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
    
    def get_adaptive_lr_multiplier(self, cam_idx, window_size, mode='exponential'):
        """
        根据帧在滑动窗口中的位置计算学习率乘数
        cam_idx: 0是最新的帧，window_size-1是最老的帧
        """
        if window_size == 1:
            return 1.0  # 只有一帧时正常学习率
        
        # 计算帧的"年龄"（0是最新，1是最老）
        age_ratio = cam_idx / (window_size - 1)
        
        if mode == 'exponential':
            # 指数衰减：新帧lr=1.0，老帧接近0
            # 使用更陡峭的衰减曲线
            return np.exp(-50 * age_ratio)  # e^(-5) ≈ 0.0067 for oldest frame
        
        elif mode == 'stepped':
            # 分级衰减（您建议的方案）
            if cam_idx == window_size - 1:  # 最老的帧
                return 0.0  # 完全冻结
            elif cam_idx == window_size - 2:  # 第二老
                return 1e-5  
            elif cam_idx == window_size - 3:  # 第三老
                return 1e-3
            elif cam_idx == window_size - 4:  # 第四老
                return 1e-1
            else:
                return 1.0  # 较新的帧保持正常学习率
        
        elif mode == 'smooth_decay':
            # 平滑的多项式衰减
            # 保证最新的几帧有较高学习率，老帧快速衰减
            if cam_idx < 2:  # 最新的2帧
                return 1.0
            else:
                # 使用三次函数实现平滑过渡
                return (1 - age_ratio) ** 10

        
    def update_mapping_points(self, frame_idx, w2c, w2c_old, depth, depth_old, intrinsics, method=None):
        if method == "rigid":
            # just move the points according to their SE(3) transformation without updating depth
            frame_idxs = self.gaussians.unique_kfIDs # idx which anchored the set of points
            frame_mask = (frame_idxs==frame_idx) # global variable
            if frame_mask.sum() == 0:
                return
            # Retrieve current set of points to be deformed
            # But first we need to retrieve all mean locations and clone them
            means = self.gaussians.get_xyz.detach()
            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means[frame_mask], pix_ones), dim=1)
            means[frame_mask] = (transformation @ pts4.T).T[:, :3]
            # put the new means back to the optimizer
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(means, "xyz")["xyz"]
            # transform the corresponding rotation matrices
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(transformation.expand_as(rots[frame_mask]), rots[frame_mask])
           
            with torch.no_grad():
                self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(rots, "rotation")["rotation"]
        else:
            # Update pose and depth by projecting points into the pixel space to find updated correspondences.
            # This strategy also adjusts the scale of the gaussians to account for the distance change from the camera
           
            depth = depth.to(self.device)
            frame_idxs = self.gaussians.unique_kfIDs # idx which anchored the set of points
            frame_mask = (frame_idxs==frame_idx) # global variable
            if frame_mask.sum() == 0:
                return

            # Retrieve current set of points to be deformed
            means = self.gaussians.get_xyz.detach()[frame_mask]

            # Project the current means into the old camera to get the pixel locations
            pix_ones = torch.ones(means.shape[0], 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            pixel_locations = (intrinsics @ (w2c_old @ pts4.T)[:3, :]).T
            pixel_locations[:, 0] /= pixel_locations[:, 2]
            pixel_locations[:, 1] /= pixel_locations[:, 2]
            pixel_locations = pixel_locations[:, :2].long()
            height, width = depth.shape
            # Some pixels may project outside the viewing frustum.
            # Assign these pixels the depth of the closest border pixel
            pixel_locations[:, 0] = torch.clamp(pixel_locations[:, 0], min=0, max=width - 1)
            pixel_locations[:, 1] = torch.clamp(pixel_locations[:, 1], min=0, max=height - 1)

            # Extract the depth at those pixel locations from the new depth 
            depth = depth[pixel_locations[:, 1], pixel_locations[:, 0]]
            depth_old = depth_old[pixel_locations[:, 1], pixel_locations[:, 0]]
            # Next, we can either move the points to the new pose and then adjust the 
            # depth or the other way around.
            # Lets adjust the depth per point first
            # First we need to transform the global means into the old camera frame
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            means_cam = (w2c_old @ pts4.T).T[:, :3]

            rescale_scale = (1 + 1/(means_cam[:, 2])*(depth - depth_old)).unsqueeze(-1) # shift
            # account for 0 depth values - then just do rigid deformation
            rigid_mask = torch.logical_or(depth == 0, depth_old == 0)
            rescale_scale[rigid_mask] = 1
            if (rescale_scale <= 0.0).sum() > 0:
                rescale_scale[rescale_scale <= 0.0] = 1
        
            rescale_mean = rescale_scale.repeat(1, 3)
            means_cam = rescale_mean*means_cam

            # Transform back means_cam to the world space
            pts4 = torch.cat((means_cam, pix_ones), dim=1)
            means = (torch.linalg.inv(w2c_old) @ pts4.T).T[:, :3]

            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pts4 = torch.cat((means, pix_ones), dim=1)
            means = (transformation @ pts4.T).T[:, :3]

            # reassign the new means of the frame mask to the self.gaussian object
            global_means = self.gaussians.get_xyz.detach()
            global_means[frame_mask] = means
            # print("mean nans: ", global_means.isnan().sum()/global_means.numel())
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(global_means, "xyz")["xyz"]

            # update the rotation of the gaussians
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(transformation.expand_as(rots[frame_mask]), rots[frame_mask])
            self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(rots, "rotation")["rotation"]

            # Update the scale of the Gaussians
            scales = self.gaussians._scaling.detach()
            scales[frame_mask] = scales[frame_mask] + torch.log(rescale_scale)
            self.gaussians._scaling = self.gaussians.replace_tensor_to_optimizer(scales, "scaling")["scaling"]
            self.gaussians.invalidate_mip_filter()


    def get_w2c_and_depth(self, video_idx, idx, mono_depth, print_info=False, init=False):
        est_droid_depth, valid_depth_mask, c2w = self.video.get_depth_and_pose(video_idx,self.device)
        c2w = c2w.to(self.device)
        w2c = torch.linalg.inv(c2w)
        if print_info:
            print(f"valid depth number: {valid_depth_mask.sum().item()}, " 
                    f"valid depth ratio: {(valid_depth_mask.sum()/(valid_depth_mask.shape[0]*valid_depth_mask.shape[1])).item()}")
        if valid_depth_mask.sum() < 100:
            invalid = True
            print(f"Skip mapping frame {idx} at video idx {video_idx} because of not enough valid depth ({valid_depth_mask.sum()}).")  
        else:
            invalid = False

        est_droid_depth[~valid_depth_mask] = 0
        if not invalid:
            mono_valid_mask = mono_depth < (mono_depth.mean()*3)
            mono_depth[mono_depth > 4*mono_depth.mean()] = 0
            from scipy.ndimage import binary_erosion
            mono_depth = mono_depth.cpu().numpy()
            binary_image = (mono_depth > 0).astype(int)
            # Add padding around the binary_image to protect the borders
            iterations = 5
            padded_binary_image = np.pad(binary_image, pad_width=iterations, mode='constant', constant_values=1)
            structure = np.ones((3, 3), dtype=int)
            # Apply binary erosion with padding
            eroded_padded_image = binary_erosion(padded_binary_image, structure=structure, iterations=iterations)
            # Remove padding after erosion
            eroded_image = eroded_padded_image[iterations:-iterations, iterations:-iterations]
            # set mono depth to zero at mask
            mono_depth[eroded_image == 0] = 0

            if (mono_depth == 0).sum() > 0:
                mono_depth = torch.from_numpy(cv2.inpaint(mono_depth, (mono_depth == 0).astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_NS)).to(self.device)
            else:
                mono_depth = torch.from_numpy(mono_depth).to(self.device)

            valid_mask = torch.from_numpy(eroded_image).to(self.device)*valid_depth_mask # new

            cur_wq = self.video.get_depth_scale_and_shift(video_idx, mono_depth, est_droid_depth, valid_mask)
            mono_depth_wq = mono_depth * cur_wq[0] + cur_wq[1]

            est_droid_depth[~valid_depth_mask] = mono_depth_wq[~valid_depth_mask]

        return est_droid_depth, w2c, invalid

    def initialize_map(self, cur_frame_idx, viewpoint):
        # Initialize blur model if enabled
        if self.config["deblur"]["open"]:
            from src.utils.motion_and_defocus_blur_mlp import CompositeBlurModel
            self.blur_model = CompositeBlurModel(self.gaussians, self.config)
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            self._refresh_mip_splatting((viewpoint,))
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config["mapping"], image, depth, viewpoint, opacity, initialization=True
            )
            # loss_init = get_loss_mapping(
            #     self.config["mapping"], image, depth, viewpoint, opacity, initialization=True
            # )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        self.printer.print("Initialized map", FontColor.MAPPER)
        torch.cuda.empty_cache()  # 重要：释放显存

        # online plotting
        if self.online_plotting:
            from thirdparty.gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            import cv2
            import numpy as np
            cur_idx = self.current_window[np.array(self.current_window).argmax()]
            viewpoint = self.cameras[cur_idx]
            render_pkg = render(
                                viewpoint, self.gaussians, self.pipeline_params, self.background
                            )
            (
                image,
                depth,
            ) = (
                render_pkg["render"].detach(),
                render_pkg["depth"].detach(),
            )
            gt_image = viewpoint.original_image
            gt_depth = viewpoint.depth

            original_dir = self.save_dir + "/online_plots/original_images"
            os.makedirs(original_dir, exist_ok=True)
            original_img = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{original_dir}/{viewpoint.timestamp}.png", original_img)

            image = torch.clamp(image, 0.0, 1.0)
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            diff_depth_l1 = torch.abs(depth.detach().cpu() - gt_depth)
            diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
            depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()

            # Add plotting 2x3 grid here
            plot_dir = self.save_dir + "/online_plots"
            plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1,
                                    psnr_score.item(), depth_l1, plot_dir=plot_dir, idx=str(cur_idx),
                                    diff_rgb=np.abs(gt - pred))

        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        initial_poses = {}
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.cameras[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.cameras.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for i in range(iters):
            self.iteration_count += 1
            self.last_sent += 1
            self._refresh_mip_splatting(viewpoint_stack)

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                observation_weight = self._observation_weight(viewpoint)
                loss_weight = observation_weight
                # 加强锐利帧的损失，使得系统更专注于锐利帧
                if self.config["sharp_loss_weight"]:
                    loss_weight *= 1.0 if viewpoint.is_blurry else self.config.get("sharp_loss_weight_value", 2.0)
                    # 调试信息：验证权重分配是否正确
                    if i == 0 and cam_idx == 0 and self.verbose:
                        self.printer.print(f"Frame {viewpoint.timestamp}: is_blurry={viewpoint.is_blurry}, loss_weight={loss_weight}", FontColor.MAPPER)
                dataset = self.config["dataset"]
                if viewpoint.deblur_fail and not self.config["composite_blur"]:
                    # 如果deblur fail的话，权重应该进一步下降
                    loss_weight = 0.1 * observation_weight
                    if dataset == 'replica_blurry':
                        if viewpoint.uid!= 0 and viewpoint.uid!= 8:
                            prev_video_idx = self.deblur_kf2video[viewpoint.uid]
                            prev = self.cameras[prev_video_idx]
                        else:
                            prev = viewpoint
                    else:
                        if viewpoint.uid!=self.initial_frame_uid:
                            prev_video_idx = self.deblur_kf2video[viewpoint.uid]
                            prev = self.cameras[prev_video_idx]
                        else:
                            prev = viewpoint
                else:
                    if dataset == 'replica_blurry':
                        if viewpoint.uid!= 0 and viewpoint.uid!= 8:
                            prev = self.cameras[viewpoint.uid-1]
                        else:
                            prev = viewpoint
                    else:
                        if viewpoint.uid!=self.initial_frame_uid:
                            prev = self.cameras[viewpoint.uid-1]
                        else:
                            prev = viewpoint
                keyframes_opt.append(viewpoint)
                if viewpoint.deblur_fail:
                    if self.config["deblur"]["open"]:
                        mode = 'mapping'
                        scale = self.scale_judge(mode, i)
                        scale_factor = 2 ** (scale - 1)
                    else:
                        scale_factor = 1
                    images_tensor = torch.empty((viewpoint.n_virtual_cams), 3, viewpoint.image_height//scale_factor, viewpoint.image_width//scale_factor, device="cuda:0")
                    depths_tensor = torch.empty((viewpoint.n_virtual_cams), 1, viewpoint.image_height//scale_factor, viewpoint.image_width//scale_factor, device="cuda:0")
                    mid_cam_idx = viewpoint.n_virtual_cams // 2
                    touched_cam = []
                    gt_image = viewpoint.original_image
                    R, t, theta, rho = viewpoint.get_virtual_extrinsics()
                    for virtual_cam in range(viewpoint.n_virtual_cams):
                        render_pkg = render_virtual(
                            viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam],rho = rho[virtual_cam] 
                        )
                        (
                            image,
                            viewspace_point_tensor,
                            visibility_filter,
                            radii,
                            depth,
                            opacity,
                            n_touched,
                        ) = (
                            render_pkg["render"],
                            render_pkg["viewspace_points"],
                            render_pkg["visibility_filter"],
                            render_pkg["radii"],
                            render_pkg["depth"],
                            render_pkg["opacity"],
                            render_pkg["n_touched"],
                        )

                        
                        # 对每个渲染帧应用复合模糊模型
                        if self.config["deblur"]["open"] and viewpoint.is_blurry:
                            # 应用复合模糊模型到单帧

                            blur_output = self.blur_model(
                                image, depth, viewpoint, i, opacity = opacity, mode = mode, kf = self.kf2mapper_idx
                            )
                            
                            # 使用复合模糊后的图像
                            image = blur_output['composite_blurred']
                            depth = blur_output['depth']

                            loss_blur = self.blur_model.compute_losses(
                                opacity,
                                blur_output, 
                                viewpoint.original_image.cuda(),
                                viewpoint,
                                i,
                                mode=mode
                            )

                            if self.config["sharp_loss_weight"]:
                                loss_mapping += loss_weight * loss_blur['total']
                            else:
                                loss_mapping += observation_weight * loss_blur['total']

                        image_ab, depth = _match_virtual_render_resolution(
                            image, depth, images_tensor.shape[-2:]
                        )
                        images_tensor[virtual_cam] = image_ab
                        depths_tensor[virtual_cam] = depth
                        
                        viewspace_point_tensor_acm.append(viewspace_point_tensor)
                        visibility_filter_acm.append(visibility_filter)
                        radii_acm.append(radii)
                        touched_cam.append(n_touched)
                    avg_image = images_tensor.mean(0)
                    avg_depth = depths_tensor.mean(0)

                    seen = not (cam_idx == 0)
                    if not self.config["deblur"]["open"]:
                        if self.config["sharp_loss_weight"]:
                            loss_mapping += loss_weight * BAD_mapping_loss(
                                self.config, avg_image, gt_image, images_tensor, depths_tensor, viewpoint, seen
                            )
                        else:
                            loss_mapping += observation_weight * BAD_mapping_loss(
                                self.config, avg_image, gt_image, images_tensor, depths_tensor, viewpoint, seen
                            )
                    else:
                        if self.config["sharp_loss_weight"]:
                            loss_mapping += loss_weight * self.blur_model.compute_BAD_losses(
                                avg_image, gt_image, images_tensor, depths_tensor, viewpoint, scale, seen, mode = "mapping", prev = prev
                            )
                        else:
                            loss_mapping += observation_weight * self.blur_model.compute_BAD_losses(
                                avg_image, gt_image, images_tensor, depths_tensor, viewpoint, scale, seen, mode = "mapping", prev = prev
                            )

                    touched_cam = torch.stack(touched_cam)
                    n_touched_acm.append(touched_cam.max(dim=0).values)
                else:
                    mode = 'mapping'
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background
                    )
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )

                    # Apply blur model if enabled and after warm-up
                    # if self.config["deblur"]["open"] and viewpoint.is_blurry and i >10:
                    if self.config["deblur"]["open"] and viewpoint.is_blurry:
                        blur_output = self.blur_model(
                            image, depth, viewpoint, i, opacity=opacity, mode='mapping', kf = self.kf2mapper_idx
                        )
                        if i == iters - 1 and cam_idx == 0 and iters > 1 and self.render_videos:
                            kernel_dir = os.path.join(self.save_dir, "kernel")
                            os.makedirs(kernel_dir, exist_ok=True)
                            save_path = os.path.join(kernel_dir, f"kernel_idx_{viewpoint.timestamp}.png")
                            visualize_kernel_weights(
                                blur_output['kernel_weights'], 
                                blur_output['mask'], 
                                blur_output['kernel_size'], 
                                num_kernels=3,
                                save_path=save_path
                            )
                                                
                        # Use blurred image for loss computation
                        image = blur_output['composite_blurred']
                        
                        # Add regularization losses
                        loss_blur = self.blur_model.compute_losses(
                            opacity,
                            blur_output, 
						    viewpoint.original_image.cuda(),
                            viewpoint,
                            i,
                            mode=mode,
                        )
                        if self.config["sharp_loss_weight"]:
                            loss_mapping += loss_weight * loss_blur['total']
                        else:
                            loss_mapping += observation_weight * loss_blur['total']
                        # loss_mapping += get_loss_mapping(
                        #     self.config["mapping"], image, depth, viewpoint, opacity
                        # )
                    else:
                        if self.config["sharp_loss_weight"]:
                            # Standard loss computation
                            loss_mapping += loss_weight * get_loss_mapping(
                                self.config["mapping"], image, depth, viewpoint, opacity
                            )
                        else:
                            loss_mapping += observation_weight * get_loss_mapping(
                                self.config["mapping"], image, depth, viewpoint, opacity
                            )
                    # loss_mapping += get_loss_mapping(
                    #     self.config["mapping"], image, depth, viewpoint, opacity
                    # )
                    image_ab = image
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)
                    n_touched_acm.append(n_touched)
                # 这个输出的是细化的渲染过程，但是我不想要，，只要渲染图像即可
                # if i%10==0 and cam_idx == 0 and iters > 1 and self.render_videos and i != 0:
                if i == iters - 1 and cam_idx == 0 and iters > 1 and self.render_videos:
                    with torch.no_grad():
                        _, color, _, _, _ = self.frame_reader[viewpoint.timestamp]
                        gt_image = color.to(self.device).squeeze()
                        deblur_image = viewpoint.original_image
                        # 创建全景相机（后移视角）
                        panoramic_cam = deepcopy(viewpoint)
                        R_mat = viewpoint.R.cpu().numpy()
                        T_vec = viewpoint.T.cpu().numpy()
                        cam_pos = -R_mat.T @ T_vec
                        view_direction = R_mat[2, :]  # z轴方向
                        move_distance = 0.6  # 后移距离
                        new_cam_pos = cam_pos - view_direction * move_distance
                        new_T = -R_mat @ new_cam_pos
                        panoramic_cam.T = torch.tensor(new_T, dtype=torch.float32, device=viewpoint.T.device)
                        
                        panoramic_pkg = render(
                            panoramic_cam, self.gaussians, self.pipeline_params, self.background
                        )

                        panoramic_image = torch.clamp(panoramic_pkg["render"], 0.0, 1.0)

                        # 获取两个图像的尺寸
                        gt_h, gt_w = gt_image.shape[1], gt_image.shape[2]  # [C, H, W]
                        ab_h, ab_w = image_ab.shape[1], image_ab.shape[2]  # [C, H, W]
                        
                        # 如果尺寸不同，使用插值调整image_ab到gt_image的尺寸
                        if ab_h != gt_h or ab_w != gt_w:
                            import torch.nn.functional as F
                            # 插值image_ab到gt_image的尺寸
                            image_ab = F.interpolate(
                                image_ab.unsqueeze(0),  # 添加batch维度 [1, C, H, W]
                                size=(gt_h, gt_w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)  # 移除batch维度 [C, H, W]
                        
                        # 确保全景图像尺寸一致
                        if panoramic_image.shape[1] != gt_h or panoramic_image.shape[2] != gt_w:
                            panoramic_image = F.interpolate(
                                panoramic_image.unsqueeze(0),
                                size=(gt_h, gt_w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)

                        panoramic_video_frame = (torch.clamp(panoramic_image.detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                        if self.fake_sharp:
                            # 直接使用插值后的image_ab
                            video_frame = (torch.clamp(torch.cat((gt_image, deblur_image, image_ab), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                        else:
                            video_frame = (torch.clamp(torch.cat((gt_image, image_ab), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                    self.global_panoramic.append(panoramic_video_frame)
                    self.global_optimiz_video.append(video_frame)
                    self.global_frame_idx.append(current_window[0])
                # 输出渲染过程
                """
                if i == iters - 1 and cam_idx == 0 and iters > 1 and self.render_videos:
                    with torch.no_grad():
                        _, color, _, _, _ = self.frame_reader[viewpoint.timestamp]
                        gt_image = color.to(self.device).squeeze()
                        video_frame = (torch.clamp(torch.cat((image_ab, gt_image), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                    self.global_optimiz_video.append(video_frame)
                    self.global_frame_idx.append(current_window[0])
                """
                    

            if self.resplat_cfg.get("online_enabled", False) and random_viewpoint_stack:
                if self.online_replay_sampler is None:
                    self.online_replay_sampler = self._make_replay_sampler(
                        (), scope="online"
                    )
                background_viewpoints = _select_online_replay_background(
                    random_viewpoint_stack,
                    self.online_replay_sampler,
                    step=self.iteration_count,
                    view_count=self.resplat_cfg.get(
                        "online_replay_views", ONLINE_REPLAY_VIEW_COUNT
                    ),
                )
            else:
                background_viewpoints = [
                    (int(cam_idx), random_viewpoint_stack[int(cam_idx)])
                    for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]
                ]

            for cam_idx, viewpoint in background_viewpoints:
                observation_weight = self._observation_weight(viewpoint)
                loss_weight = observation_weight
                if self.config["sharp_loss_weight"]:
                    loss_weight *= 1.0 if viewpoint.is_blurry else self.config.get("sharp_loss_weight_value", 2.0)
                dataset = self.config["dataset"]
                if viewpoint.deblur_fail and not self.config["composite_blur"]:
                    # 如果deblur fail的话，权重应该进一步下降
                    loss_weight = 0.1 * observation_weight
                    if dataset == 'replica_blurry':
                        if viewpoint.uid!= 0 and viewpoint.uid!= 8:
                            prev_video_idx = self.deblur_kf2video[viewpoint.uid]
                            prev = self.cameras[prev_video_idx]
                        else:
                            prev = viewpoint
                    else:
                        if viewpoint.uid!=self.initial_frame_uid:
                            prev_video_idx = self.deblur_kf2video[viewpoint.uid]
                            prev = self.cameras[prev_video_idx]
                        else:
                            prev = viewpoint
                else:
                    if dataset == 'replica_blurry':
                        if viewpoint.uid!= 0 and viewpoint.uid!= 8:
                            prev = self.cameras[viewpoint.uid-1]
                        else:
                            prev = viewpoint
                    else:
                        if viewpoint.uid!=self.initial_frame_uid:
                            prev = self.cameras[viewpoint.uid-1]
                        else:
                            prev = viewpoint
                if viewpoint.deblur_fail:
                    if self.config["deblur"]["open"]:
                        mode = 'mapping'
                        scale = self.scale_judge(mode, i)
                        scale_factor = 2 ** (scale - 1)
                    else:
                        scale_factor = 1
                    images_tensor = torch.empty((viewpoint.n_virtual_cams), 3, viewpoint.image_height// scale_factor, viewpoint.image_width// scale_factor, device="cuda:0")
                    depths_tensor = torch.empty((viewpoint.n_virtual_cams), 1, viewpoint.image_height// scale_factor, viewpoint.image_width// scale_factor, device="cuda:0")
                    mid_cam_idx = viewpoint.n_virtual_cams // 2
                    touched_cam = []
                    gt_image = viewpoint.original_image
                    R, t, theta, rho = viewpoint.get_virtual_extrinsics()
                    for virtual_cam in range(viewpoint.n_virtual_cams):
                        render_pkg = render_virtual(
                            viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam],rho = rho[virtual_cam] 
                        )
                        (
                            image,
                            viewspace_point_tensor,
                            visibility_filter,
                            radii,
                            depth,
                            opacity,
                            n_touched,
                        ) = (
                            render_pkg["render"],
                            render_pkg["viewspace_points"],
                            render_pkg["visibility_filter"],
                            render_pkg["radii"],
                            render_pkg["depth"],
                            render_pkg["opacity"],
                            render_pkg["n_touched"],
                        )

                        # 对每个渲染帧应用复合模糊模型
                        if self.config["deblur"]["open"] and viewpoint.is_blurry:
                            # 应用复合模糊模型到单帧
                            blur_output = self.blur_model(
                                image, depth, viewpoint, i, opacity=opacity, mode=mode, kf = self.kf2mapper_idx
                            )
                            
                            # 使用复合模糊后的图像
                            image = blur_output['composite_blurred']
                            depth = blur_output['depth']

                            loss_blur = self.blur_model.compute_losses(
                                opacity,
                                blur_output, 
                                viewpoint.original_image.cuda(),
                                viewpoint,
                                i,
                                mode=mode,
                            )

                            if self.config["sharp_loss_weight"]:
                                loss_mapping += loss_weight * loss_blur['total']
                            else:
                                loss_mapping += observation_weight * loss_blur['total']

                        image_ab, depth = _match_virtual_render_resolution(
                            image, depth, images_tensor.shape[-2:]
                        )
                        images_tensor[virtual_cam] = image_ab
                        depths_tensor[virtual_cam] = depth
                        
                        viewspace_point_tensor_acm.append(viewspace_point_tensor)
                        visibility_filter_acm.append(visibility_filter)
                        radii_acm.append(radii)
                        touched_cam.append(n_touched)
                    avg_image = images_tensor.mean(0)
                    avg_depth = depths_tensor.mean(0)

                    seen = not (cam_idx == 0)

                    if not self.config["deblur"]["open"]:
                        if self.config["sharp_loss_weight"]:
                            loss_mapping += loss_weight * BAD_mapping_loss(
                                self.config, avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint, seen
                            )
                        else:
                            loss_mapping += observation_weight * BAD_mapping_loss(
                                self.config, avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint, seen
                            )
                    else:
                        if self.config["sharp_loss_weight"]:
                            loss_mapping += loss_weight * self.blur_model.compute_BAD_losses(
                                avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint, scale, seen, mode = "mapping", prev = prev
                            )
                        else:
                            loss_mapping += observation_weight * self.blur_model.compute_BAD_losses(
                                avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint, scale, seen, mode = "mapping", prev = prev
                            )

                    touched_cam = torch.stack(touched_cam)
                    n_touched_acm.append(touched_cam.max(dim=0).values)
                else:
                    mode = 'mapping'
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background
                    )
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )
                    if self.config["deblur"]["open"] and viewpoint.is_blurry:
                        # 应用复合模糊模型到单帧
                        blur_output = self.blur_model(
                            image, depth, viewpoint, i, opacity=opacity, mode=mode, kf = self.kf2mapper_idx
                        )
                        
                        # 使用复合模糊后的图像
                        image = blur_output['composite_blurred']
                        depth = blur_output['depth']

                        # Add regularization losses
                        loss_blur = self.blur_model.compute_losses(
                            opacity, 
                            blur_output, 
                            viewpoint.original_image.cuda(),
                            viewpoint,
                            i,
                        )
                        if self.config["sharp_loss_weight"]:
                            loss_mapping += loss_weight * loss_blur['total']
                        else:
                            loss_mapping += observation_weight * loss_blur['total']
                        # loss_mapping += get_loss_mapping(
                        #    self.config["mapping"], image, depth, viewpoint, opacity
                        #)
                    else:
                        # Standard loss computation
                        if self.config["sharp_loss_weight"]:
                            loss_mapping += loss_weight * get_loss_mapping(
                                self.config["mapping"], image, depth, viewpoint, opacity
                            )
                        else:
                            loss_mapping += observation_weight * get_loss_mapping(
                                self.config["mapping"], image, depth, viewpoint, opacity
                            )
                    #loss_mapping += get_loss_mapping(
                    #    self.config["mapping"], image, depth, viewpoint, opacity
                    #)
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)

                replay_image = avg_image if viewpoint.deblur_fail else image
                self._observe_replay(
                    self.online_replay_sampler,
                    viewpoint,
                    replay_image,
                    opacity,
                    step=self.iteration_count,
                )

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))

            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()

            gaussian_split = False
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # compute the visibility of the gaussians
                # Only prune on the last iteration and when we have a full window
                if prune:
                    if len(current_window) == self.window_size:
                        prune_mode = self.config["mapping"]["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True # not used it seems

                ## Opacity reset
                # self.iteration_count is a global parameter. We use gaussian reset
                # every 2001 iterations meaning if we use 60 per mapping frame
                # and there are 160 keyframes in the sequence, we do resetting
                # 4 times. Using more mapping iterations leads to more resetting
                # which can prune away more gaussians.
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    self.printer.print("Resetting the opacity of non-visible Gaussians", FontColor.MAPPER)
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                # 执行优化器更新
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                

                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    # 0是replica的初始帧，11是tum的
                    if viewpoint.uid == self.initial_frame_uid or getattr(
                        viewpoint, "fixed_pose", False
                    ):
                        continue
                    if viewpoint.deblur_fail:
                        converged = True
                        for i in range(viewpoint.num_control_knots):
                            converged = update_pose_knot(viewpoint, i) and converged
                    if viewpoint.deblur_fail:
                        # 获取优化后的中间帧位姿
                        R_mid, t_mid, _, _ = viewpoint.get_mid_extrinsic()
                        # 同步到Camera的主位姿属性
                        viewpoint.update_RT(R_mid, t_mid)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == self.initial_frame_uid or getattr(
                        viewpoint, "fixed_pose", False
                    ):
                        continue
                    update_pose(viewpoint)
                del viewspace_point_tensor_acm
                del visibility_filter_acm
                del radii_acm
                del n_touched_acm
                del keyframes_opt
                torch.cuda.empty_cache()
        if self.render_videos:
            with torch.no_grad():
                render_video(os.path.join(self.save_dir,"global_mapping.mp4"), self.global_optimiz_video, 5, self.global_frame_idx)
                render_video(os.path.join(self.save_dir,"global_panoramic.mp4"), self.global_panoramic, 5, self.global_frame_idx)
        # online plotting
        if self.online_plotting:
            from thirdparty.gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            import cv2
            import numpy as np
            
            # 遍历当前窗口所有帧
            for cur_idx in current_window:
                viewpoint = self.cameras[cur_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    depth,
                ) = (
                    render_pkg["render"].detach(),
                    render_pkg["depth"].detach(),
                )
                _, color, _, _, _ = self.frame_reader[viewpoint.timestamp]
                gt_image = color.to(self.device).squeeze()
                gt_depth = viewpoint.depth 
                original_dir = self.save_dir + "/online_plots/original_images"
                os.makedirs(original_dir, exist_ok=True)
                original_img = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"{original_dir}/{viewpoint.timestamp}.png", original_img)

                if viewpoint.uid != self.video_idxs[0]:
                    image = (torch.exp(viewpoint.exposure_a.detach())) * image + viewpoint.exposure_b.detach()
                
                image = torch.clamp(image, 0.0, 1.0)
                gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
                pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
                mask = gt_image > 0
                psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
                diff_depth_l1 = torch.abs(depth.detach().cpu() - gt_depth)
                diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
                depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()

                plot_dir = self.save_dir + "/online_plots"
                plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1,
                                    psnr_score.item(), depth_l1, plot_dir=plot_dir, 
                                    idx=str(cur_idx), diff_rgb=np.abs(gt - pred))
                
                # 释放单帧显存
                del render_pkg, image, depth, gt_image, diff_depth_l1
            
            # 最后统一清理显存
            torch.cuda.empty_cache()
        return gaussian_split

    def get_warmup_exp_lr(self, current_iter, warmup_iters=10000, decay_rate=0.9998):
        """先warm-up再指数衰减，适合大尺寸模糊核的精细调整"""
        if current_iter < warmup_iters:
            # Linear warm-up
            return 1.0
        else:
            # Exponential decay
            decay_steps = current_iter - warmup_iters
            return max(1e-2, decay_rate ** decay_steps)


    def final_refine(self, prune=False, iters=26000):
        self._forced_commit_final100_entry(iters)
        self._hydrate_missing_droid_keyframes_for_final_refine()
        # self.gaussians.set_mlp_learning_rate(lr_multiplier=1000.0)
        self.gaussians.set_mlp_learning_rate(lr_multiplier=10000.0)
        self.printer.print("Starting final refinement", FontColor.MAPPER)
        use_sharp_weight = self.config.get("sharp_weight_final", self.config.get("sharp_loss_weight", False))
        sharp_weight_value = self.config.get("sharp_weight_final_value", self.config.get("sharp_loss_weight_value", 2.0))
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]
        lr_multiplier = 1.0

        final_panoramic_video = []
        final_frame_idx = []

        # Do final update of depths and poses
        for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
            camera = self.cameras[keyframe_idx]
            if getattr(camera, "fixed_pose", False):
                # The generated RGB-D observation was rendered at its manifest
                # pose. DROID has no measurement at this inserted timestamp and
                # must not overwrite or deform geometry around that pose.
                continue
            intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)
            mono_depth = self._load_mono_depth_for_timestamp(frame_idx)
            
            # 对于 deblur_fail 帧 (负数索引)，使用SLERP外推位姿
            if keyframe_idx in self.deblur_fail_kfs:
                # deblur_fail 帧不经过 Droid-SLAM，需要特殊处理位姿
                w2c_temp = self.estimate_pose_with_slerp(keyframe_idx, mono_depth)
                depth_temp = mono_depth
                invalid = False
                print(f"[final_refine] Updating deblur_fail frame {keyframe_idx} (timestamp={frame_idx}) with SLERP pose")
            else:
                depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth, init=False)
            
            # Update tracking parameters
            w2c_old = torch.cat((self.cameras[keyframe_idx].R, self.cameras[keyframe_idx].T.unsqueeze(-1)), dim=1)
            w2c_old = torch.cat((w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0)
            
            if self.cameras[keyframe_idx].deblur_fail:
                if self.config["composite_blur"]:
                    # update the estimated pose to be the glorie pose
                    for knot in range(self.cameras[keyframe_idx].num_control_knots):
                        self.cameras[keyframe_idx].update_RT_motion(w2c_temp[:3, :3], w2c_temp[:3, 3], knot)
                # 对于 deblur_fail 且非 composite_blur 的情况，也要更新位姿
                self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
                self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy() if hasattr(depth_temp, 'cpu') else depth_temp
                self.cameras[keyframe_idx].is_valid = True  # 标记为有效，使其参与后续优化
            else:
                self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
                self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()
                self.cameras[keyframe_idx].is_valid = ~invalid

            if keyframe_idx in self.cameras:
                # Update tracking parameters
                self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
                self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy() if hasattr(depth_temp, 'cpu') else depth_temp

            # Update mapping parameters - 对于 deblur_fail 帧不做点云变形
            if (
                self.move_points
                and self.is_kf.get(keyframe_idx, False)
                and keyframe_idx not in self.deblur_fail_kfs
                and not getattr(self.cameras[keyframe_idx], "fixed_pose", False)
            ):
                if invalid:
                    self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics, method="rigid")
                else:
                    self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics)
                    self.depth_dict[keyframe_idx] = depth_temp # not needed since it is the last deformation but keeping for clarity.

        # Initialize optimizer with all keyframe parameters including sub-frame poses
        opt_params = []
        # Convert lr values to float to avoid type errors
        lr_cam_rot = float(self.config["mapping"]["Training"]["lr"]["cam_rot_delta"])
        lr_cam_trans = float(self.config["mapping"]["Training"]["lr"]["cam_trans_delta"])

        total = len(self.cameras)
        for idx, (cam_idx, viewpoint) in enumerate(self.cameras.items(), 1):
            if _motion_knots_are_optimizable(viewpoint):
                for knot in range(viewpoint.num_control_knots):
                    opt_params.append({
                        "params": [viewpoint.T_i_rot_delta[knot]],
                        "lr": lr_cam_rot * 0.25,
                        "name": "rot_{}_{}".format(viewpoint.uid, knot),
                    })
                    opt_params.append({
                        "params": [viewpoint.T_i_trans_delta[knot]],
                        "lr": lr_cam_trans * 0.25,
                        "name": "trans_{}_{}".format(viewpoint.uid, knot),
                    })          
                opt_params.append({
                    "params": [viewpoint.exposure_a],
                    "lr": 0.001,
                    "name": "exposure_a_{}".format(viewpoint.uid),
                })
                opt_params.append({
                    "params": [viewpoint.exposure_b],
                    "lr": 0.001,
                    "name": "exposure_b_{}".format(viewpoint.uid),
                })
            
            if viewpoint.uid == self.initial_frame_uid:
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_a],
                        "lr": 0.001,
                        "name": "exposure_a_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_b],
                        "lr": 0.001,
                        "name": "exposure_b_{}".format(viewpoint.uid),
                    }
                )
                for cam_idx in range(len(self.current_window)):
                    viewpoint = self.cameras[self.current_window[cam_idx]]
                    if not self.gt_camera and self.config["mapping"]["BA"]:
                        if (
                            cam_idx < frames_to_optimize
                            and not viewpoint.deblur_fail
                            and not _offline_pose_is_frozen(viewpoint)
                        ):
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": lr_cam_rot
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": lr_cam_trans
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        # elif cam_idx < frames_to_optimize and viewpoint.deblur_fail:
                        elif _motion_knots_are_optimizable(viewpoint):
                            for knot in range(viewpoint.num_control_knots):
                                opt_params.append(
                                    {
                                        "params": [viewpoint.T_i_rot_delta[knot]],
                                        "lr": lr_cam_rot * lr_multiplier,
                                        "name": "rot_{}_{}".format(viewpoint.uid, knot),
                                    }
                                )
                                opt_params.append(
                                    {
                                        "params": [viewpoint.T_i_trans_delta[knot]],
                                        "lr": lr_cam_trans
                                        * 0.5 * lr_multiplier,
                                        "name": "trans_{}_{}".format(viewpoint.uid, knot),
                                    }
                                )
                        if viewpoint.deblur_fail:
                            opt_params.append(
                                {
                                    "params": [viewpoint.exposure_a],
                                    "lr": 0.001,
                                    "name": "exposure_a_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.exposure_b],
                                    "lr": 0.001,
                                    "name": "exposure_b_{}".format(viewpoint.uid),
                                }
                            )
                        else:
                            opt_params.append(
                                {
                                    "params": [viewpoint.exposure_a],
                                    "lr": 0.01,
                                    "name": "exposure_a_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.exposure_b],
                                    "lr": 0.01,
                                    "name": "exposure_b_{}".format(viewpoint.uid),
                                }
                            )
            if idx == total:
                self.keyframe_optimizers = torch.optim.Adam(opt_params)


        random_viewpoint_stack = []
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]

        for cam_idx, viewpoint in self.cameras.items():
            random_viewpoint_stack.append(viewpoint)

        if not random_viewpoint_stack:
            raise RuntimeError("final refinement has no training viewpoints")
        replay_enabled = bool(self.resplat_cfg.get("enabled", False))
        configured_budget = int(iters)
        total_iters, replay_start, budget_mode = self._resolve_refinement_budget(
            configured_budget, replay_enabled, self.resplat_cfg
        )
        replay_sampler = None
        viewpoints_by_uid = {int(viewpoint.uid): viewpoint for viewpoint in random_viewpoint_stack}
        stack_index_by_uid = {
            int(viewpoint.uid): index
            for index, viewpoint in enumerate(random_viewpoint_stack)
        }
        if replay_enabled:
            replay_sampler = self._make_replay_sampler(
                viewpoints_by_uid.keys(), scope="final"
            )
            self.printer.print(
                "ReSplat-inspired residual replay: "
                f"budget_mode={budget_mode}, uniform={replay_start}, "
                f"replay={total_iters - replay_start}, total={total_iters}, "
                f"views={len(viewpoints_by_uid)}",
                FontColor.MAPPER,
            )

        checkpoint_steps = {
            int(step)
            for step in self.resplat_cfg.get("checkpoint_steps", [])
            if 0 < int(step) <= total_iters
        }
        checkpoint_steps.add(total_iters)

        for i in tqdm(range(total_iters)):
            # Final refinement has no Gaussian topology changes, but camera
            # poses continue to move.  Refresh at the cadence used by the
            # public Mip-Splatting implementation rather than charging an
            # all-camera O(N) pass to every optimizer step.
            if i % 100 == 0:
                self._refresh_mip_splatting(random_viewpoint_stack)
            # 调整学习率衰减策略：更长的warmup和更慢的衰减，避免过早过度拟合
            # warmup_iters从2增加到2000，让学习率保持较高更长时间
            # decay_rate从0.5改为0.9995，使衰减更平缓
            lr_mult = self.get_warmup_exp_lr(i, warmup_iters=2000, decay_rate=0.9995)

            self.gaussians.set_mlp_learning_rate(lr_multiplier=lr_mult)
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []
           
            if replay_sampler is not None and i >= replay_start:
                replay_uid = int(replay_sampler.sample(step=i))
                viewpoint = viewpoints_by_uid[replay_uid]
                rand_idx = stack_index_by_uid[replay_uid]
            else:
                rand_idx = np.random.randint(0, len(random_viewpoint_stack))
                viewpoint = random_viewpoint_stack[rand_idx]
            
            observation_weight = self._observation_weight(viewpoint)
            loss_weight = observation_weight
            if use_sharp_weight:
                loss_weight *= 1.0 if viewpoint.is_blurry else sharp_weight_value
            dataset = self.config["dataset"]
            if viewpoint.deblur_fail and not self.config["composite_blur"]:
                # 如果deblur fail的话，权重应该进一步下降
                loss_weight = 0.1 * observation_weight
                if dataset == 'replica_blurry':
                    if viewpoint.uid!= 0 and viewpoint.uid!= 8:
                        prev_video_idx = self.deblur_kf2video[viewpoint.uid]
                        prev = self.cameras[prev_video_idx]
                    else:
                        prev = viewpoint
                else:
                    if viewpoint.uid!=self.initial_frame_uid:
                        prev_video_idx = self.deblur_kf2video[viewpoint.uid]
                        prev = self.cameras[prev_video_idx]
                    else:
                        prev = viewpoint
            else:
                if dataset == 'replica_blurry':
                    if viewpoint.uid!= 0 and viewpoint.uid!= 8:
                        prev = self.cameras[viewpoint.uid-1]
                    else:
                        prev = viewpoint
                else:
                    if viewpoint.uid!=self.initial_frame_uid:
                        prev = self.cameras[viewpoint.uid-1]
                    else:
                        prev = viewpoint
            if viewpoint.deblur_fail:
                if self.config["deblur"]["open"]:
                    mode = 'mapping'
                    scale = self.scale_judge(mode, i)
                    scale_factor = 2 ** (scale - 1)
                else:
                    scale_factor = 1
                images_tensor = torch.empty((viewpoint.n_virtual_cams), 3, viewpoint.image_height//scale_factor, viewpoint.image_width//scale_factor, device="cuda:0")
                depths_tensor = torch.empty((viewpoint.n_virtual_cams), 1, viewpoint.image_height//scale_factor, viewpoint.image_width//scale_factor, device="cuda:0")

                # 找到中间虚拟相机索引
                mid_cam_idx = viewpoint.n_virtual_cams // 2
                R, t, theta, rho = viewpoint.get_virtual_extrinsics()
                for virtual_cam in range(viewpoint.n_virtual_cams):
                    render_pkg = render_virtual(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam], rho = rho[virtual_cam] 
                    )
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )
                    # 对每个渲染帧应用复合模糊模型
                    if self.config["deblur"]["open"] and viewpoint.is_blurry:
                        # 应用复合模糊模型到单帧
                        blur_output = self.blur_model(
                            image,
                            depth,
                            viewpoint, 
                            i,
                            opacity = opacity,
                            mode=mode,
                            kf = self.kf2mapper_idx
                        )
                        
                        # 使用复合模糊后的图像
                        image = blur_output['composite_blurred']
                        depth = blur_output['depth']

                        loss_blur = self.blur_model.compute_losses(
                            opacity,
                            blur_output, 
                            viewpoint.original_image.cuda(),
                            viewpoint,
                            i,
                            mode=mode
                        )

                        if use_sharp_weight:
                            loss_mapping += loss_weight * loss_blur['total']
                        else:
                            loss_mapping += observation_weight * loss_blur['total']
                    image_ab, depth = _match_virtual_render_resolution(
                        image, depth, images_tensor.shape[-2:]
                    )
                    images_tensor[virtual_cam] = image_ab
                    depths_tensor[virtual_cam] = depth
                    
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)
                
                avg_image = images_tensor.mean(0)
                avg_depth = depths_tensor.mean(0)

                if not self.config["deblur"]["open"]:
                    if use_sharp_weight:
                        loss_mapping += loss_weight * BAD_mapping_loss(
                            self.config, avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint, seen = True
                        )
                    else:
                        loss_mapping += observation_weight * BAD_mapping_loss(
                            self.config, avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint, seen = True
                        )
                else:
                    if use_sharp_weight:
                        loss_mapping += loss_weight * self.blur_model.compute_BAD_losses(
                            avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint,scale, seen = True, mode = "mapping", prev = prev
                        )
                    else:
                        loss_mapping += observation_weight * self.blur_model.compute_BAD_losses(
                            avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint,scale, seen = True, mode = "mapping", prev = prev
                        )

                n_touched_acm.append(n_touched)

                scaling = self.gaussians.get_scaling
                loss_mapping.backward()
                gaussian_split = False
                ## Deinsifying / Pruning Gaussians
                with torch.no_grad():
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                    
                    # Update poses for all viewpoints
                    for cam_idx, vp in self.cameras.items():
                        if vp.uid == 0 or _offline_pose_is_frozen(vp):
                            continue
                            
                        if vp.deblur_fail:
                            # Update sub-frame poses for blurry frames
                            converged = True
                            for knot_idx in range(vp.num_control_knots):
                                converged = update_pose_knot(vp, knot_idx) and converged
                                
                        if vp.deblur_fail:
                            # Sync middle frame pose to main pose attributes
                            R_mid, t_mid, _, _ = vp.get_mid_extrinsic()
                            vp.update_RT(R_mid, t_mid)
            else:
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                ) 
                if self.config["deblur"]["open"] and viewpoint.is_blurry:
                    blur_output = self.blur_model(
                        image, depth, viewpoint, i, opacity = opacity, mode = "final", kf = self.kf2mapper_idx
                    )
                    
                    # Use blurred image for loss computation
                    image = blur_output['composite_blurred']
                    
                    # Add regularization losses
                    loss_blur = self.blur_model.compute_losses(
                        opacity,
                        blur_output, 
                        viewpoint.original_image.cuda(),
                        viewpoint,
                        i,
                        mode="final"
                    )
                    if use_sharp_weight:
                        loss_mapping += loss_weight * loss_blur['total']
                    else:
                        loss_mapping += observation_weight * loss_blur['total']
                else:
                    if use_sharp_weight:
                        # Standard loss computation
                        loss_mapping += loss_weight * get_loss_mapping(
                            self.config["mapping"], image, depth, viewpoint, opacity
                        )
                    else:
                        loss_mapping += observation_weight * get_loss_mapping(
                            self.config["mapping"], image, depth, viewpoint, opacity
                        )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

                scaling = self.gaussians.get_scaling
                loss_mapping.backward()
                gaussian_split = False
                ## Deinsifying / Pruning Gaussians
                with torch.no_grad():
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)
                    # optimize the exposure compensation
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)

            replay_image = avg_image if viewpoint.deblur_fail else image
            self._observe_replay(
                replay_sampler, viewpoint, replay_image, opacity, step=i
            )
            if (
                replay_sampler is not None
                and i > 0
                and i % max(
                    1, int(self.resplat_cfg.get("checkpoint_interval", 1000))
                ) == 0
            ):
                replay_sampler.save_state(
                    Path(self.save_dir) / "resplat_replay_final_state.json"
                )
            completed_step = i + 1
            if completed_step in checkpoint_steps:
                self._save_refinement_checkpoint(
                    completed_step, total_iters, replay_sampler
                )

            del viewspace_point_tensor_acm
            del visibility_filter_acm
            del radii_acm
            del n_touched_acm
            del keyframes_opt
            torch.cuda.empty_cache()
            if i % 2000 == 0 and i != 0 and self.render_videos:
                with torch.no_grad():
                    # 选择一个代表性的视点
                    sample_viewpoint = random_viewpoint_stack[rand_idx]
                    
                    # 创建全景相机（后移视角）
                    from copy import deepcopy
                    panoramic_cam = deepcopy(sample_viewpoint)
                    R_mat = sample_viewpoint.R.cpu().numpy()
                    T_vec = sample_viewpoint.T.cpu().numpy()
                    cam_pos = -R_mat.T @ T_vec
                    view_direction = R_mat[2, :]  # z轴方向
                    move_distance = 0.6  # 后移距离
                    new_cam_pos = cam_pos - view_direction * move_distance
                    new_T = -R_mat @ new_cam_pos
                    panoramic_cam.T = torch.tensor(new_T, dtype=torch.float32, device=sample_viewpoint.T.device)
                    
                    panoramic_pkg = render(
                        panoramic_cam, self.gaussians, self.pipeline_params, self.background
                    )
                    panoramic_image = torch.clamp(panoramic_pkg["render"], 0.0, 1.0)
                    
                    # 保存帧
                    panoramic_video_frame = (panoramic_image.detach().clone().cpu().permute(1, 2, 0) * 255).type(torch.uint8)
                    final_panoramic_video.append(panoramic_video_frame)
                    final_frame_idx.append(i)
            
            # 在最后1000次迭代中，每500次保存一次模糊核可视化
            if i >= total_iters - 1000 and i % 500 == 0 and self.config["deblur"]["open"]:
                # 为模糊的视点保存kernel可视化
                pass
                """
                for vp in random_viewpoint_stack:
                    if vp.is_blurry and not vp.deblur_fail and self.config["deblur"]["open"]:
                        render_pkg_kernel = render(
                            vp, self.gaussians, self.pipeline_params, self.background
                        )
                        blur_output = self.blur_model(
                            render_pkg_kernel["render"],
                            render_pkg_kernel["depth"],
                            vp,
                            i,
                            opacity=render_pkg_kernel["opacity"],
                            mode="final",
                            kf=self.kf2mapper_idx
                        )
                        
                        # 保存kernel可视化
                        kernel_dir = os.path.join(self.save_dir, "kernel_final")
                        os.makedirs(kernel_dir, exist_ok=True)
                        save_path = os.path.join(kernel_dir, f"kernel_iter_{i}_frame_{vp.timestamp}.png")
                        visualize_kernel_weights(
                            blur_output['kernel_weights'],
                            blur_output['mask'],
                            blur_output['kernel_size'],
                            num_kernels=3,
                            save_path=save_path
                        )
                        break  # 只为一个模糊帧保存
            """

            
        
        if replay_sampler is not None:
            replay_sampler.save_state(
                Path(self.save_dir) / "resplat_replay_final_state.json"
            )

        self._forced_commit_final100_complete(configured_budget, total_iters)

        # 在final_refine结束时保存视频
        if self.render_videos and len(final_panoramic_video) > 0:
            with torch.no_grad():
                render_video(
                    os.path.join(self.save_dir, "final_panoramic.mp4"),
                    final_panoramic_video,
                    5,
                    final_frame_idx
                )
        self.printer.print("Final refinement done", FontColor.MAPPER)
        # 记录包含final_refine的总推理时间
        self.slam.timing_stats['total_inference_time'] = time.time() - self.slam.start_time_total


    def initialize(self, cur_frame_idx, viewpoint):
        # self.initialized only False at beginning for monocular MonoGS
        # in the slam_frontend.py it is used in the monocular setting
        # for some minor things for bootstrapping, but it is not relevant
        # in out "with proxy depth" setting.
        self.initialized = True
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        self.mapped_video_idxs = []
        self.mapped_kf_idxs = []

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)


    def add_new_keyframe(self, cur_frame_idx, idx, depth=None, opacity=None):
        rgb_boundary_threshold = self.config["mapping"]["Training"]["rgb_boundary_threshold"]
        self.mapped_video_idxs.append(cur_frame_idx)
        self.mapped_kf_idxs.append(idx)
        viewpoint = self.cameras[cur_frame_idx]
        
        gt_img = viewpoint.original_image.cuda()
        # Filter out RGB pixels where the R + G + B values < 0.01
        # valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        valid_rgb = (gt_img.sum(dim=0) > -1)[None]

        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels. THIS LINE OVERWRITES THE self.viewpoints[cur_frame_idx].depth with "initial_depth"
        return initial_depth[0].cpu().numpy()

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["mapping"]["Training"]["kf_translation"]
        kf_min_translation = self.config["mapping"]["Training"]["kf_min_translation"]
        kf_overlap = self.config["mapping"]["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        # multiply by median depth in rgb-only setting to account for scale ambiguity
        dist_check = dist > kf_translation * self.median_depth 
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["mapping"]["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["mapping"]["Training"]
                else 0.4
            )
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.window_size:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame
    
    def estimate_pose_with_slerp(self, video_idx, mono_depth):
        """使用SLERP的匀速运动估计器 (修复了负数索引导致的查找失败问题)"""
        # print("video_idx is", video_idx)

        # 1. 获取所有已存在的相机，并按时间戳排序
        # 这样无论Key是正数还是deblur_fail产生的负数，都能找到最近的帧
        sorted_cameras = sorted(self.cameras.values(), key=lambda x: x.timestamp)
        
        # 2. 如果历史帧超过2帧，使用最后两帧进行 SLERP 外推
        """
        if len(sorted_cameras) >= 2:
            # 获取最近的两帧
            cam_prev = sorted_cameras[-1]       # 上一帧 (t-1)
            cam_prev_prev = sorted_cameras[-2]  # 上上帧 (t-2)

            # 检查这两帧位姿是否相同（避免零运动外推）
            R_same = torch.norm(cam_prev.R - cam_prev_prev.R) < 1e-5
            T_same = torch.norm(cam_prev.T - cam_prev_prev.T) < 1e-5
            
            if R_same and T_same and len(sorted_cameras) >= 3:
                # 如果最近两帧位姿相同，尝试使用更早的帧
                cam_prev_prev = sorted_cameras[-3]
                
            # 构建w2c矩阵
            w2c_prev = getWorld2View2(cam_prev.R, cam_prev.T)
            w2c_prev_prev = getWorld2View2(cam_prev_prev.R, cam_prev_prev.T)
            
            # 分解为旋转和平移
            R_prev, t_prev = w2c_prev[:3, :3], w2c_prev[:3, 3]
            _, t_prev_prev = w2c_prev_prev[:3, :3], w2c_prev_prev[:3, 3] # R_prev_prev 仅用于四元数计算
            R_prev_prev = w2c_prev_prev[:3, :3] # 显式获取

            dt1 = cam_prev.timestamp - cam_prev_prev.timestamp
            if dt1 > 0:
                # 平移：线性外推 (当前位置 = 上一帧 + (上一帧 - 上上帧))
                t_new = t_prev + (t_prev - t_prev_prev)
                
                # 旋转：SLERP外推
                q_prev = matrix_to_quaternion(R_prev)
                q_prev_prev = matrix_to_quaternion(R_prev_prev)
                # t=2.0 意味着基于 q_prev_prev 和 q_prev 的差值，向前推一步
                q_new = _device_slerp(2.0, q_prev_prev, q_prev)
                R_new = quaternion_to_matrix(q_new)
            else:
                # 时间间隔为0，直接复用上一帧
                R_new, t_new = R_prev, t_prev
            
            # 组合新的 W2C
            w2c = torch.eye(4, device=R_new.device)
            w2c[:3, :3] = R_new
            w2c[:3, 3] = t_new
            return w2c
        """
            
        # 3. 如果有2帧以上历史，使用SLERP外推
        if len(sorted_cameras) >= 2:
            cam_prev = sorted_cameras[-1]       # 上一帧 (t-1)
            cam_prev_prev = sorted_cameras[-2]  # 上上帧 (t-2)
            
            # 检查这两帧位姿是否相同（避免零运动外推）
            R_same = torch.norm(cam_prev.R - cam_prev_prev.R) < 1e-5
            T_same = torch.norm(cam_prev.T - cam_prev_prev.T) < 1e-5
            
            if R_same and T_same and len(sorted_cameras) >= 3:
                # 如果最近两帧位姿相同，尝试使用更早的帧
                cam_prev_prev = sorted_cameras[-3]
                
            # 构建w2c矩阵
            w2c_prev = getWorld2View2(cam_prev.R, cam_prev.T)
            w2c_prev_prev = getWorld2View2(cam_prev_prev.R, cam_prev_prev.T)
            
            # 分解为旋转和平移
            R_prev, t_prev = w2c_prev[:3, :3], w2c_prev[:3, 3]
            R_prev_prev, t_prev_prev = w2c_prev_prev[:3, :3], w2c_prev_prev[:3, 3]

            dt1 = cam_prev.timestamp - cam_prev_prev.timestamp
            if dt1 > 0:
                # 平移：线性外推 (当前位置 = 上一帧 + (上一帧 - 上上帧))
                t_new = t_prev + (t_prev - t_prev_prev)
                
                # 旋转：SLERP外推
                q_prev = matrix_to_quaternion(R_prev)
                q_prev_prev = matrix_to_quaternion(R_prev_prev)
                # t=2.0 意味着基于 q_prev_prev 和 q_prev 的差值，向前推一步
                q_new = _device_slerp(2.0, q_prev_prev, q_prev)
                R_new = quaternion_to_matrix(q_new)
            else:
                # 时间间隔为0，直接复用上一帧
                R_new, t_new = R_prev, t_prev
            
            # 组合新的 W2C
            w2c = torch.eye(4, device=R_new.device)
            w2c[:3, :3] = R_new
            w2c[:3, 3] = t_new
            print(f"[Info] SLERP extrapolation: used frames {cam_prev_prev.uid} -> {cam_prev.uid} to estimate pose for deblur_fail frame")
            return w2c
        
        # 4. 如果只有1帧历史，直接复用该帧位姿
        elif len(sorted_cameras) == 1:
            print("[Info] Only 1 history frame, assuming constant pose for deblur_fail.")
            cam_prev = sorted_cameras[-1]  # 只有一帧时使用[-1]
            print(f"Using pose from camera uid: {cam_prev.uid}")
            return getWorld2View2(cam_prev.R.detach(), cam_prev.T.detach())
            
        # 5. 没有任何历史帧 (初始化阶段的 fallback)
        else:
            print(f"[Warning] No history frames available, using mono_depth-based fallback for video_idx={video_idx}")
            # 使用单位矩阵作为初始位姿
            w2c = torch.eye(4, device=mono_depth.device)
            return w2c

    def run(self):
        """
        Trigger mapping process, get estimated pose and depth from tracking process,
        send continue signal to tracking process when the mapping of the current frame finishes.  
        """
        config = self.config

        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.frame_reader.fx,
            fy=self.frame_reader.fy,
            cx=self.frame_reader.cx,
            cy=self.frame_reader.cy,
            W=self.frame_reader.W_out,
            H=self.frame_reader.H_out,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
    
        num_frames = len(self.frame_reader)

        # Initialize list to keep track of Keyframes
        self.keyframe_idxs = [] # 
        self.video_idxs = [] # keyframe numbering (note first
        # keyframe for mapping is the 7th keyframe in total)
        self.is_kf = dict() # keys are video_idx and value is boolean. This prevents trying to deform frames that were never mapped.
        # this is only a problem when the last keyframe is not mapped as this would otherwise be handled by the code.
        
        # Init Variables to keep track of ground truth poses and runtimes
        self.gt_w2c_all_frames = []

        init = True

        # Define first frame pose
        _, color, _, first_frame_c2w, _ = self.frame_reader[0]
        intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)

        # global camera dictionary - updated during mapping.
        self.cameras = dict()
        self.depth_dict = dict()
        self.mapper_only_frame_counter = 0


        while (1):
            deblur_fail = False
            is_blurry = True
            frame_info = self.pipe.recv()
            idx = frame_info['timestamp'] # frame index
            video_idx = frame_info['video_idx'] # keyframe index
            is_finished = frame_info['end']
            if is_finished:
                print("Done with Mapping and Tracking")
                break
            # Forced-diagnostic provenance: this is intentionally the first call after the
            # received message has been identified as a live frame.  It must
            # precede metadata access, pose/depth work, deformation, PCD
            # extension, mapping, and pruning for source220.
            self._forced_commit_source220_entry(int(idx), int(video_idx))
            input_metadata = self.frame_reader.frame_info(int(idx))
            is_synthetic_input = bool(input_metadata.get("synthetic", False))
            if 'blur_degree' in frame_info:
                blur_degree = frame_info['blur_degree']
                if blur_degree == 0.0:
                    is_blurry = False
            else:
                blur_degree = None
            print(idx)
            
            if 'check_score' in frame_info:
                check_score = frame_info['check_score']
            else:
                check_score = None
            self.kf2mapper_idx[idx] = self.mapper_idx
            self.mapper_idx += 1
            jump_droid = False
            should_skip_frame = False
            if blur_degree is not None and check_score is not None and blur_degree > 0.99 and check_score < 0.45:
                deblur_fail = True
                jump_droid = True
                # Check if skip_fail is enabled - if so, skip this frame entirely
                if self.config.get('skip_fail', False):
                    should_skip_frame = True
                    deblur_fail = False
                    jump_droid = False

            if should_skip_frame:
                self.printer.print(f"Skipping deblur-failed frame {idx} (skip_fail=True)", FontColor.MAPPER)
                self.pipe.send("continue")
                continue

            # 为mapper独有的帧创建独立索引
            if jump_droid:
                # 使用负数索引来区分mapper独有的帧
                mapper_video_idx = -1000000 - self.mapper_only_frame_counter  # 使用大负数避免冲突
                self.deblur_kf2video[mapper_video_idx] = self.video_idxs[-1]
                self.mapper_only_frame_counter += 1
                actual_video_idx = mapper_video_idx  # 用于存储到self.cameras
            else:
                actual_video_idx = video_idx  # 正常帧使用原始video_idx
            video_idx = actual_video_idx

            if self.config["composite_blur"] and not is_synthetic_input:
                deblur_fail = True
            
            if init:
                deblur_fail = False

            if self.verbose:
                self.printer.print(f"\nMapping Frame {idx} ...", FontColor.MAPPER)
            
            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx)
                print(Style.RESET_ALL)

            self.keyframe_idxs.append(idx)
            # 这个会搞一个负索引出来，但是camera也是这个负索引嘛？
            self.video_idxs.append(video_idx)
            
            if deblur_fail and jump_droid:
                self.deblur_fail_kfs.add(video_idx)  # 标记video_idx为mapper独有


            _, color, depth_gt, c2w_gt, _ = self.frame_reader[idx]
            mono_depth = (
                None
                if is_synthetic_input
                else self._load_mono_depth_for_timestamp(idx)
            )
            color = color.to(self.device)
            c2w_gt = c2w_gt.to(self.device)
            if not deblur_fail or self.config["composite_blur"]:
                dataset = self.config["dataset"]
                sequence = self.config["scene"]
                # 加载中间锐利图像
                # 默认viewpoint有输入时间戳
                # Run-local cache: never consume a deblurred tensor left by a
                # different baseline/ablation sharing the same scene name.
                sharp_dir = str(Path(self.save_dir) / "sharp")
                sharp_file = Path(sharp_dir) / f"{idx}.pt"
                mid_sharp = None
                if os.path.exists(sharp_file):
                    mid_sharp = load_tensor(idx, sharp_dir)
                    color = mid_sharp.to(self.device) 

            if is_synthetic_input:
                if depth_gt is None:
                    raise ValueError(
                        "A gated FrameCrafter RGB-D observation requires synthetic depth"
                    )
                depth = torch.as_tensor(
                    depth_gt, dtype=torch.float32, device=self.device
                )
                manifest_c2w = (
                    c2w_gt[c2w_gt.shape[0] // 2]
                    if c2w_gt.ndim == 3
                    else c2w_gt
                )
                if self.config.get("framecrafter", {}).get(
                    "use_manifest_pose", True
                ):
                    manifest_c2w = self._align_generated_c2w(
                        manifest_c2w, input_metadata
                    )
                    w2c = torch.linalg.inv(manifest_c2w)
                else:
                    _, w2c, _ = self.get_w2c_and_depth(
                        video_idx, idx, depth, init=False
                    )
                invalid = False
                deblur_fail = False
                jump_droid = False
            elif deblur_fail and not self.config["composite_blur"] and jump_droid:
                # depth, w2c, invalid = self.get_w2c_and_depth(video_idx, idx, mono_depth, init=False)
                mono_depth = self._load_mono_depth_for_timestamp(idx)
                depth = mono_depth
                # 获取上一个关键帧的video_idx
                if len(self.video_idxs) > 0:
                    print("get consistant speed model")
                    # 因为现在已经加进去现在运行的帧的，所以应该是之前第二帧
                    # 为了防止索引越界，做个保护
                    if len(self.video_idxs) >= 2:
                        last_kf_video_idx = self.video_idxs[-2]
                    else:
                        last_kf_video_idx = self.video_idxs[-1]
                    w2c = self.estimate_pose_with_slerp(last_kf_video_idx, mono_depth)
                else:
                    print("get identity speed model")
                    # 初始情况，使用单位矩阵或其他初始化
                    w2c = torch.eye(4, device=self.device)
            else:
                depth, w2c, invalid = self.get_w2c_and_depth(video_idx, idx, mono_depth, init=False)
                if invalid:
                    w2c_gt = torch.linalg.inv(c2w_gt)
                    self.gt_w2c_all_frames.append(w2c_gt)
                    print("WARNING: Too few valid pixels from droid depth")
                    # online glorieslam pose and depth
                    data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose": w2c, \
                            "gt_pose": w2c_gt, "idx": video_idx}
                    self.is_kf[video_idx] = False
                    viewpoint = Camera.init_from_dataset(
                            self.frame_reader, data, projection_matrix, 
                        )
                    viewpoint.timestamp = idx
                    viewpoint.is_valid = False
                    if is_blurry:
                        viewpoint.is_blurry = True
                    else:
                        viewpoint.is_blurry = False
                    self._annotate_augmented_viewpoint(viewpoint, idx)
                    # update the estimated pose to be the glorie pose
                    viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
                    viewpoint.compute_grad_mask(self.config)
                    self.cameras[video_idx] = viewpoint
                    # Dictionary of Camera objects at the frame index
                    # self.cameras contains all cameras.
                    self.pipe.send("continue")
                    # 对于无效帧,不要添加到video_idxs/keyframe_idxs
                    self.keyframe_idxs.pop()  # 移除刚添加的索引
                    self.video_idxs.pop()
                    continue # too few valid pixels from droid depth
            
            if deblur_fail:
                # 这里缺一个w2c_gt的匀速运动估计器，因为Droid-SLAM的位姿估计并不准
                n_virtual_cams = self.config.get("n_virtual_cams", 5)
                w2c_gt = torch.linalg.inv(c2w_gt)
                self.gt_w2c_all_frames.append(w2c_gt)
                data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose": w2c, \
                    "gt_pose": w2c_gt, "idx": video_idx, "n_virtual_cams": n_virtual_cams,
                    "interpolation": self.config["interpolation"]
                    }
                
                # 别忘问这里property对吗？
                viewpoint = Camera.init_from_dataset_motion(
                    self.frame_reader, data, projection_matrix, deblur_fail
                )
                viewpoint.timestamp = idx
            else:
                w2c_gt = torch.linalg.inv(c2w_gt)
                self.gt_w2c_all_frames.append(w2c_gt)
                # online glorieslam pose and depth
                data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose": w2c, \
                        "gt_pose": w2c_gt, "idx": video_idx}

                viewpoint = Camera.init_from_dataset(
                        self.frame_reader, data, projection_matrix, 
                    )
                viewpoint.timestamp = idx
            viewpoint.timestamp = idx
            if is_blurry:
                viewpoint.is_blurry = True
            else:
                viewpoint.is_blurry = False
            self._annotate_augmented_viewpoint(viewpoint, idx)
            # update the estimated pose to be the glorie pose
            # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

            if deblur_fail:
                # update the estimated pose to be the glorie pose
                for knot in range(viewpoint.num_control_knots):
                    viewpoint.update_RT_motion(viewpoint.R_gt_motion[knot], viewpoint.T_gt_motion[knot], knot)
            else:
                viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            viewpoint.compute_grad_mask(self.config)
            # Dictionary of Camera objects at the frame index
            # self.cameras contains all cameras.
            self.cameras[video_idx] = viewpoint

            if init:
                self.initialize(video_idx, viewpoint)

                self.printer.print("Resetting the system", FontColor.MAPPER)
                self.reset()
                self.current_window.append(video_idx)
                # Add first depth map to depth dictionary - important for the first deformation
                # of the first frame
                self.depth_dict[video_idx] = depth
                self.is_kf[video_idx] = True # we map the first keyframe (after warmup)

                self.cameras[video_idx] = viewpoint
                depth = self.add_new_keyframe(video_idx, idx)
                self.add_next_kf(
                    video_idx, viewpoint, depth_map=depth, init=True
                )
                self.initialize_map(video_idx, viewpoint)
                self.initial_frame_uid = video_idx
                self._register_submap_keyframe(viewpoint)
                self._active_fusion_after_mapped_keyframe(viewpoint)
                init = False
                self.pipe.send("continue")
                continue
            
            if deblur_fail:
                R, t, theta, rho = viewpoint.get_virtual_extrinsics()
                virtual_cam = viewpoint.n_virtual_cams//2
                render_pkg = render_virtual(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, R = R[virtual_cam], t = t[virtual_cam], theta = theta[virtual_cam],rho = rho[virtual_cam] 
                )
            else: 
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
            self.median_depth = get_median_depth(render_pkg["depth"], render_pkg["opacity"])


            # keyframe selection
            last_keyframe_idx = self.current_window[0]
            
            curr_visibility = (render_pkg["n_touched"] > 0).long()
            create_kf = self.is_keyframe(
                video_idx,
                last_keyframe_idx,
                curr_visibility,
                self.occ_aware_visibility,
            )
            if len(self.current_window) < self.window_size:
                # When we have not filled up the keyframe window size
                # we rely on just the covisibility thresholding, not the 
                # translation thresholds.
                union = torch.logical_or(
                    curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                ).count_nonzero()
                intersection = torch.logical_and(
                    curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                ).count_nonzero()
                point_ratio = intersection / union
                create_kf = (
                    point_ratio < self.config["mapping"]["Training"]["kf_overlap"]
                )
            
            dataset_type = self.config.get('dataset', '').lower()
        
            # Check if it's a TUM RGB-D dataset
            if dataset_type not in ['replica_blurry','archviz']:
                create_kf = True
            
            if create_kf:
                self.current_window, removed = self.add_to_window(
                    video_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                    self.current_window,
                )
                self.is_kf[video_idx] = True
                self._register_submap_keyframe(viewpoint)
            else:
                self.is_kf[video_idx] = False
                self.pipe.send("continue")
                continue

            last_idx = self.keyframe_idxs[-1]

            # 这个是更新历史所有的位姿，但是这样会覆盖原来优化的多帧位姿
            for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
                if (
                    keyframe_idx in self.deblur_fail_kfs
                    or getattr(self.cameras[keyframe_idx], "fixed_pose", False)
                ):
                    continue
                else:
                    # need to update depth_dict even if the last idx since this is important
                    # for the first deformation of the keyframe
                    mono_depth = self._load_mono_depth_for_timestamp(frame_idx)
                    depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth, init=False)

                    if keyframe_idx not in self.depth_dict and self.is_kf[keyframe_idx]:
                        self.depth_dict[keyframe_idx] = depth_temp

                    # No need to move the latest pose and depth
                    if frame_idx != last_idx: 
                        # Update tracking parameters
                        w2c_old = torch.cat((self.cameras[keyframe_idx].R, self.cameras[keyframe_idx].T.unsqueeze(-1)), dim=1)
                        w2c_old = torch.cat((w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0)
                        self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
                        if not invalid:
                            # Update depth for viewpoint
                            self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()

                        # 只更新当前窗口里的关键帧子帧位姿
                        """
                        if keyframe_idx in self.cameras and keyframe_idx in self.current_window:
                            # Update tracking parameters
                            if self.cameras[keyframe_idx].deblur_fail:
                                # update the estimated pose to be the glorie pose
                                for knot in range(viewpoint.num_control_knots):
                                    viewpoint.update_RT_motion(w2c_temp[:3, :3], w2c_temp[:3, 3], knot)
                        """
                        # 更新所有不模糊的关键帧的单独位姿
                        self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
                        # Update depth for viewpoint
                        if keyframe_idx in self.cameras:
                            self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()
                            self.cameras[keyframe_idx].is_valid = ~invalid

                        # Update mapping parameters
                        if self.move_points and self.is_kf[keyframe_idx]:
                            if invalid:
                                # if the frame was invalid, we don't update the depth old and just do a rigid correction for this frame
                                self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics, method="rigid")
                            else:
                                self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics)
                                self.depth_dict[keyframe_idx] = depth_temp # line does not matter since it is the last deformation anyway
 
            # Do mapping
            # self.viewpoints contains the subset of self.cameras where we did mapping
            self.cameras[video_idx] = viewpoint
            depth = self.add_new_keyframe(video_idx, idx)
            self.add_next_kf(video_idx, viewpoint, depth_map=depth, init=False) # set init to True for debugging

            # Keep the camera that triggered this mapping update distinct from
            # the pose-window loop variable below.  That loop intentionally
            # visits and rebinds ``viewpoint`` to historical cameras; active
            # fusion must be notified about the newly mapped camera instead.
            mapped_viewpoint = viewpoint

            self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

            opt_params = []
            frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]
            iter_per_kf = self.mapping_itr_num

            for cam_idx in range(len(self.current_window)):
                if self.current_window[cam_idx] == 0:
                    # Do not add GT frame pose for optimization
                    continue
                print(self.current_window)
                viewpoint = self.cameras[self.current_window[cam_idx]]
                # 计算当前帧的学习率乘数
                #lr_multiplier = self.get_adaptive_lr_multiplier(
                #    cam_idx, 
                #    len(self.current_window),
                #    mode='smooth_decay'
                #)
                # 既然所有的窗口内的关键帧每次都要更新一遍位姿，那为什么不直接同一个学习率了呢？
                lr_multiplier = 1.0
                # 如果学习率乘数为0，跳过这个帧的优化
                # if lr_multiplier == 0.0:
                #    continue
                # Convert lr values to float to avoid type errors
                lr_cam_rot = float(self.config["mapping"]["Training"]["lr"]["cam_rot_delta"])
                lr_cam_trans = float(self.config["mapping"]["Training"]["lr"]["cam_trans_delta"])
                if not self.gt_camera and self.config["mapping"]["BA"]:
                    if (
                        cam_idx < frames_to_optimize
                        and not viewpoint.deblur_fail
                        and not getattr(viewpoint, "fixed_pose", False)
                    ):
                        opt_params.append(
                            {
                                "params": [viewpoint.cam_rot_delta],
                                "lr": lr_cam_rot
                                * 0.5,
                                "name": "rot_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.cam_trans_delta],
                                "lr": lr_cam_trans
                                * 0.5,
                                "name": "trans_{}".format(viewpoint.uid),
                            }
                        )
                    # elif cam_idx < frames_to_optimize and viewpoint.deblur_fail:
                elif viewpoint.deblur_fail and not getattr(
                    viewpoint, "fixed_pose", False
                ):
                    print(viewpoint.uid)
                    for knot in range(viewpoint.num_control_knots):
                        opt_params.append(
                            {
                                "params": [viewpoint.T_i_rot_delta[knot]],
                                "lr": lr_cam_rot * lr_multiplier,
                                "name": "rot_{}_{}".format(viewpoint.uid, knot),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.T_i_trans_delta[knot]],
                                "lr": lr_cam_trans
                                * 0.5 * lr_multiplier,
                                "name": "trans_{}_{}".format(viewpoint.uid, knot),
                            }
                        )

                if viewpoint.deblur_fail:
                    opt_params.append(
                        {
                            "params": [viewpoint.exposure_a],
                            "lr": 0.001,
                            "name": "exposure_a_{}".format(viewpoint.uid),
                        }
                    )
                    opt_params.append(
                        {
                            "params": [viewpoint.exposure_b],
                            "lr": 0.001,
                            "name": "exposure_b_{}".format(viewpoint.uid),
                        }
                    )
                else:
                    opt_params.append(
                        {
                            "params": [viewpoint.exposure_a],
                            "lr": 0.01,
                            "name": "exposure_a_{}".format(viewpoint.uid),
                        }
                    )
                    opt_params.append(
                        {
                            "params": [viewpoint.exposure_b],
                            "lr": 0.01,
                            "name": "exposure_b_{}".format(viewpoint.uid),
                        }
                    )
            self.keyframe_optimizers = torch.optim.Adam(opt_params)
            
            self.map(self.current_window, iters=iter_per_kf)
            self.map(self.current_window, prune=True)
            if self.official_resplat_active_fusion_cfg.enabled:
                if mapped_viewpoint is not self.cameras[video_idx]:
                    raise RuntimeError(
                        "active-fusion mapped_viewpoint identity changed inside pose window"
                    )
                if (
                    int(mapped_viewpoint.uid) != int(video_idx)
                    or int(mapped_viewpoint.timestamp) != int(idx)
                ):
                    raise RuntimeError(
                        "active-fusion mapped_viewpoint no longer matches current frame"
                    )
            self._active_fusion_after_mapped_keyframe(mapped_viewpoint)

            self.pipe.send("continue")
        self._finalize_active_fusion_contract()
        self._finalize_submaps()
        # 记录在线推理结束时间
        self.slam.timing_stats['online_inference_time'] = time.time() - self.slam.start_time_online
        
        # 记录峰值GPU显存
        if torch.cuda.is_available():
            self.slam.timing_stats['peak_gpu_memory'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # 记录峰值CPU内存
        process = psutil.Process()
        self.slam.timing_stats['peak_cpu_memory'] = process.memory_info().rss / 1024**3  # GB
        if self.online_plotting and hasattr(self, 'fig_trajectory'):
            plt.close(self.fig_trajectory)
