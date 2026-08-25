# Copyright 2024 The GlORIE-SLAM Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
import lietorch
import time
import cv2
import os
from pathlib import Path
from thirdparty.monogs.utils.slam_utils import variance_of_laplacian
from src.utils.datasets import BaseDataset
from src.utils.blur_detector_metrics import BlurDetector

import thirdparty.glorie_slam.geom.projective_ops as pops
from thirdparty.glorie_slam.modules.droid_net import CorrBlock
from src.mono_estimators import get_mono_depth_estimator,predict_mono_depth
from src.utils.datasets import load_mono_depth
from src.utils.fixed_keyframes import parse_fixed_source_keyframe_contract

from src.utils.common import save_tensor
from src.utils.tum_inverse_processor import CompleteInverseProcessor
from src.deblur_backends import build_deblur_backend

class MotionFilter:
    """ This class is used to filter incoming frames and extract features 
        mainly inherited from DROID-SLAM
    """

    def __init__(self, net, video, cfg, thresh=2.5, device="cuda:0"):
        self.cfg = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0
        self.deblur_backend_name = None
        self.deblur_backend = None
        self.last_streaming_vs_evssm_gain = None
        self.last_streaming_candidate_safe = True
        self.last_streaming_causal_gain = None
        self.last_streaming_evssm_fallback = False
        self.last_streaming_selection = "raw"
        self.last_track_info = {
            "synthetic": False,
            "streaming_replaced": False,
            "streaming_evssm_fallback": False,
            "motion_keyframe": False,
            "tracking_anchor": False,
            "appended": False,
        }

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
        if cfg["mono_prior"]["predict_online"]:
            self.mono_depth_estimator = get_mono_depth_estimator(cfg)
        deblur_cfg = self.cfg.get("deblur", {}) or {}
        needs_blur_detector = (
            (self.cfg["adjust_cam"] and not self.cfg["sharp_judge"])
            or self.cfg.get("sharp_judge", False)
            or not bool(deblur_cfg.get("stream_replace_sharp", True))
        )
        if needs_blur_detector:
            self.detector = BlurDetector(cfg)
        frontend_name = str(
            (self.cfg.get("deblur", {}) or {}).get("frontend", "evssm")
        ).lower()
        if frontend_name not in {
            "evssm",
            "causal_torchscript",
            "causal_evssm",
            "turtle_streaming",
            "turtle_bsd_streaming",
            "precomputed",
        }:
            raise ValueError(f"Unsupported deblur.frontend={frontend_name!r}")
        frontend_requested = self.cfg.get("fake_sharp", False) or frontend_name in {
            "causal_torchscript",
            "causal_evssm",
            "turtle_streaming",
            "turtle_bsd_streaming",
            "precomputed",
        }
        if frontend_requested:
            self.evssm_model = None
            if frontend_name in {"evssm", "causal_evssm"}:
                # Keep alternative frontends independent from EVSSM's optional
                # model/package dependencies.
                from thirdparty.EVSSM.models.EVSSM import EVSSM

                self.evssm_model = EVSSM()
                if torch.cuda.is_available():
                    self.evssm_model = self.evssm_model.to(self.device)
                evssm_checkpoint_path = self.cfg.get(
                    "evssm_checkpoint", "./pretrained/evssm/net_g_latest.pth"
                )
                state_dict = torch.load(evssm_checkpoint_path, map_location=self.device)['params']
                self.evssm_model.load_state_dict(state_dict, strict=True)
                self.evssm_model.eval()
                print(f"TRACKING: EVSSM model loaded from {evssm_checkpoint_path}")
            self.deblur_backend_name, self.deblur_backend = build_deblur_backend(
                self.cfg, evssm_model=self.evssm_model, device=self.device
            )
            print(f"TRACKING: deblur frontend={self.deblur_backend_name}")
        
        if self.cfg.get("apply_inverse_gamma", False):
            self.processor = CompleteInverseProcessor()
        
        self.fixed_source_keyframe_contract = (
            parse_fixed_source_keyframe_contract(self.cfg)
        )
        self.fixed_source_keyframes = (
            None
            if self.fixed_source_keyframe_contract is None
            else self.fixed_source_keyframe_contract["source_index_set"]
        )
        if self.fixed_source_keyframe_contract is not None:
            # Do not even open the repository tracking-anchor file in a fixed
            # ablation. The only observable schedule is the preregistered list
            # already embedded identically in both resolved configs.
            self.tracking_anchor_indices = None
            print(
                "TRACKING: fixed source-keyframe ablation enabled: "
                f"{list(self.fixed_source_keyframe_contract['source_indices'])}"
            )
        else:
            # These are the original TUM tracking-anchor lists, not a runtime
            # result produced by the current sequence.
            self.tracking_anchor_indices = self._load_tracking_anchor_indices()

    @staticmethod
    def _stream_frame_info(stream, timestamp):
        """Return augmentation metadata without making GT poses observable."""
        if stream is None or not hasattr(stream, "frame_info"):
            return {
                "synthetic": False,
                "source_index": int(timestamp),
                "confidence": 1.0,
            }
        return stream.frame_info(int(timestamp))

    @staticmethod
    def _laplacian_value(image):
        value, _ = variance_of_laplacian(image)
        if torch.is_tensor(value):
            value = value.detach().float().mean().item()
        return float(value)

    def _streaming_deblur(self, image, timestamp, stream):
        """Update a causal/precomputed frontend before DROID motion filtering.

        EVSSM remains on the legacy keyframe-only path because it is a
        single-image model.  A causal model, by contrast, must see every input
        frame, including frames later rejected by the motion filter.  The
        candidate only replaces the tracking image after a sharpness-gain
        gate; a synthetic FrameCrafter observation updates temporal state but
        is never overwritten.
        """
        deblur_cfg = self.cfg.get("deblur", {}) or {}
        self.last_streaming_causal_gain = None
        self.last_streaming_vs_evssm_gain = None
        self.last_streaming_candidate_safe = True
        self.last_streaming_evssm_fallback = False
        self.last_streaming_selection = "raw"
        enabled = (
            self.deblur_backend is not None
            and bool(deblur_cfg.get("stream_every_frame", False))
            and self.deblur_backend_name in {
                "causal_torchscript",
                "causal_evssm",
                "turtle_streaming",
                "turtle_bsd_streaming",
                "precomputed",
            }
        )
        if not enabled:
            return image, None, False, None

        metadata = self._stream_frame_info(stream, timestamp)
        if self.deblur_backend_name == "precomputed" and bool(
            metadata.get("synthetic", False)
        ):
            # There is no original-frame file for an inserted view, and this
            # stateless backend has no temporal history that needs updating.
            return image, None, False, None
        backend_timestamp = (
            int(metadata.get("source_index", timestamp))
            if self.deblur_backend_name == "precomputed"
            else timestamp
        )

        started = time.perf_counter()
        candidate = self.apply_evssm_deblur(
            image.clone(), stream=stream, timestamp=backend_timestamp
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        original_laplacian = self._laplacian_value(image)
        candidate_laplacian = self._laplacian_value(candidate)
        relative_gain = (candidate_laplacian - original_laplacian) / max(
            original_laplacian, 1e-6
        )
        self.last_streaming_causal_gain = relative_gain

        if self.deblur_backend_name == "causal_evssm":
            evssm_baseline = getattr(
                self,
                "_last_evssm_output_for_gate",
                getattr(self.deblur_backend, "last_evssm_output", None),
            )
            if evssm_baseline is None:
                raise RuntimeError(
                    "causal_evssm backend did not expose its single-frame EVSSM baseline"
                )
            evssm_laplacian = self._laplacian_value(evssm_baseline)
            self.last_streaming_vs_evssm_gain = (
                candidate_laplacian - evssm_laplacian
            ) / max(evssm_laplacian, 1e-6)
            min_vs_evssm = max(
                0.0, float(deblur_cfg.get("stream_min_vs_evssm_gain", 0.0))
            )
            self.last_streaming_candidate_safe = (
                self.last_streaming_vs_evssm_gain >= min_vs_evssm
            )
            if not self.last_streaming_candidate_safe:
                # The temporal model has already consumed this frame, so its
                # causal history remains advanced.  Cache an independent copy
                # of the exact single-frame EVSSM result, but keep DROID on the
                # raw observation.  The cached result may only be consumed
                # later by the legacy keyframe + blur + fake-sharp branch.
                # In particular, a failed causal gate must not create a motion
                # recovery keyframe or a preserve-latest request.
                fallback = evssm_baseline.detach().clone()
                fallback_laplacian = self._laplacian_value(fallback)
                fallback_gain = (
                    fallback_laplacian - original_laplacian
                ) / max(original_laplacian, 1e-6)
                self.last_streaming_evssm_fallback = True
                self.last_streaming_selection = "raw_evssm_candidate"
                if self.cfg.get("verbose", False):
                    print(
                        "TRACKING: causal candidate failed EVSSM gate; "
                        f"frame={timestamp} vs_evssm="
                        f"{self.last_streaming_vs_evssm_gain:.4f} "
                        f"fallback_gain={fallback_gain:.4f} "
                        "tracking=raw cache=evssm"
                    )
                return image, fallback, False, fallback_gain

        replace = (
            bool(deblur_cfg.get("stream_apply_to_tracking", True))
            and not bool(metadata.get("synthetic", False))
            and self.last_streaming_candidate_safe
            and relative_gain >= float(
                deblur_cfg.get("stream_min_laplacian_gain", 0.0)
            )
        )
        if not bool(deblur_cfg.get("stream_replace_sharp", True)):
            # This inexpensive gate deliberately uses the existing blur
            # detector. It does not inspect clear-GT frame labels.
            detected = self.classic_blur_check(image.clone())
            replace = replace and bool(detected[0] if isinstance(detected, tuple) else detected)

        self.last_streaming_selection = (
            self.deblur_backend_name if replace else "raw"
        )

        if self.cfg.get("verbose", False):
            print(
                "TRACKING: streaming deblur "
                f"frame={timestamp} gain={relative_gain:.4f} "
                f"vs_evssm={self.last_streaming_vs_evssm_gain} "
                f"replace={replace} time_ms={elapsed_ms:.2f}"
            )
        return candidate if replace else image, candidate, replace, relative_gain

    def _streaming_candidate_rejected(self, candidate, relative_gain):
        """Apply causal gates, but admit cached EVSSM in the legacy branch."""
        if candidate is None or self.last_streaming_evssm_fallback:
            return False
        return (
            not self.last_streaming_candidate_safe
            or relative_gain is None
            or relative_gain
            < float(
                (self.cfg.get("deblur", {}) or {}).get(
                    "stream_min_laplacian_gain", 0.0
                )
            )
        )

    def _normalize_droid_input(self, image):
        """Normalize a private DROID copy without mutating RGB observations.

        ``Tensor.to`` can return the input unchanged when it is already on the
        requested device.  The subsequent in-place normalization would then
        corrupt the raw image, a streaming candidate, the mapper output, and
        (for runtime-EVSSM storage) the causal backend history.  Clone after
        device transfer so the in-place arithmetic is confined to DROID's
        private feature-extractor input.
        """
        inputs = image[None, :, :].to(self.device).clone()
        return inputs.sub_(self.MEAN).div_(self.STDV)

    def _mono_depth_for_frame(self, timestamp, image, stream):
        metadata = self._stream_frame_info(stream, timestamp)
        if bool(metadata.get("synthetic", False)):
            if stream is None:
                raise ValueError("synthetic FrameCrafter depth requires the dataset stream")
            _, _, generated_depth, _, _ = stream[int(timestamp)]
            if generated_depth is None:
                raise ValueError(
                    "accepted FrameCrafter RGB-D frame is missing gated depth"
                )
            return torch.as_tensor(generated_depth, dtype=torch.float32)
        source_index = int(metadata.get("source_index", timestamp))
        if self.cfg["mono_prior"]["predict_online"]:
            return predict_mono_depth(
                self.mono_depth_estimator,
                source_index,
                image.clone(),
                self.cfg,
                self.device,
            )
        save_dir = os.path.join(
            self.cfg["data"]["output"], self.cfg["scene"]
        )
        return load_mono_depth(source_index, save_dir)
    
    def _load_tracking_anchor_indices(self):
        """Load predefined TUM tracking anchors (never an eval-frame source)."""
        dataset_type = self.cfg.get('dataset', '').lower()
        
        # Check if it's a TUM RGB-D dataset
        if dataset_type not in ['tumrgbd', 'tumrgb']:
            return None
        
        # Get the scene name
        scene = self.cfg.get('scene', '').lower()
        
        # Map scene name to file path
        indices_file_map = {
            'freiburg1_desk': 'fr1_desk_indices.txt',
            'fr1_desk': 'fr1_desk_indices.txt',
            'freiburg2_xyz': 'fr2_xyz_indices.txt',
            'fr2_xyz': 'fr2_xyz_indices.txt',
            'freiburg3_office': 'fr3_office_indices.txt',
            'fr3_office': 'fr3_office_indices.txt',
        }
        
        indices_file = None
        for key in indices_file_map:
            if key in scene:
                indices_file = indices_file_map[key]
                break
        
        if indices_file is None:
            print(f"Warning: Unknown TUM scene '{scene}', using motion-based keyframe selection")
            return None
        indices_file = (
            Path(__file__).resolve().parents[2] / "scripts" / indices_file
        )
        
        try:
            with open(indices_file, 'r') as f:
                # Read all lines and convert to integers (removing leading zeros)
                indices = set(int(line.strip()) for line in f if line.strip())
            print(f"Loaded {len(indices)} tracking anchors from {indices_file}")
            return indices
        except Exception as e:
            print(f"Warning: Failed to load tracking anchors from {indices_file}: {e}")
            print("Falling back to motion-based keyframe selection")
            return None


    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)
    
    # 假设图片质量就是图片模糊的判断方法之一
    def classic_blur_check(self, image_tensor: torch.Tensor, sensitivity: str = 'low') -> bool:
        """
        快速检查图片是否模糊
        
        Args:
            image_tensor: 图片tensor
            sensitivity: 敏感度设置
                - 'high': 高敏感度，更多图片被判断为模糊
                - 'medium': 中等敏感度
                - 'low': 低敏感度
        
        Returns:
            是否可能模糊
        """
        with torch.cuda.amp.autocast(enabled=False):
            detector = self.detector
            is_blurry, blur_score_check = detector.detect_blur(image_tensor, True)
            if self.cfg.get("disable_blur_detection", False):
                is_blurry = True
            if is_blurry:
                print("This is the blur frame")

            # timing = detector.benchmark_speed(image_tensor, n_iterations=100)
            # print(f"直方图熵计算时间：{timing['histogram_entropy_ms']:.2f}毫秒")
                        
            if self.cfg["exam_blur_score"]:
                return is_blurry, blur_score_check
            else:
                return is_blurry

    def apply_evssm_deblur(self, image_tensor: torch.Tensor, stream:BaseDataset = None,
                           timestamp=None) -> torch.Tensor:
        """
        Apply the configured deblurring frontend to the input image tensor.
        
        Args:
            image_tensor: Input blurry image tensor [H, W, 3] or [3, H, W]
            
        Returns:
            Sharp image tensor in the same format as input
        """
        # Store original shape and format
        if self.cfg.get("apply_inverse_gamma", False):
            image_tensor = self.processor.process_tum_image(
                image_tensor, 
                correct_gamma=True,
                correct_white_balance=False,  # 可以根据需要调整
                output_format='float32'  # 使用uint16以支持16-bit PNG
            )
            print("转换")

        original_shape = image_tensor.shape
        if len(original_shape) == 3:
            if original_shape[-1] == 3:  # [H, W, 3]
                is_hwc = True
                image_tensor = image_tensor.permute(2, 0, 1)  # Convert to [3, H, W]
            else:  # Already [3, H, W]
                is_hwc = False
        
        # Add batch dimension and move to device
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]
        
        # Ensure tensor is on the correct device and normalized to [0, 1]
        # 转换为float32以确保全精度
        image_tensor = image_tensor.to(dtype=torch.float32, device=self.device)
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # Apply the configured frontend. The method name is retained for
        # backward compatibility with older experiment scripts.
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                sharp_image = self.deblur_backend(image_tensor, timestamp=timestamp)
        evssm_baseline = getattr(
            self.deblur_backend, "last_evssm_output", None
        )
        
        # Convert back to original format
        if len(original_shape) == 3:
            sharp_image = sharp_image[0]
            if is_hwc:
                sharp_image = sharp_image.permute(1, 2, 0)  # Convert back to [H, W, 3]
            if evssm_baseline is not None:
                evssm_baseline = evssm_baseline[0]
                if is_hwc:
                    evssm_baseline = evssm_baseline.permute(1, 2, 0)
        

        """
        sharp_dir = f"./output/sharp_image"
        save_tensor(sharp_image, 1, sharp_dir, True)
        """
        if self.cfg.get("apply_inverse_gamma", False):
            sharp_image = self.processor.encode_srgb(sharp_image)
            if evssm_baseline is not None:
                evssm_baseline = self.processor.encode_srgb(evssm_baseline)
        self._last_evssm_output_for_gate = evssm_baseline
        return sharp_image

    def deblur_degree_detect(self, original_image, deblur_image):
        """
        检测输入图像的模糊程度，并动态调整knot数量
        返回: (blur_ratio, rendered_image, rendered_depth, opacity, n_virtual_cams)
        """
        with torch.cuda.amp.autocast(enabled=False):
            # 计算输入图像的拉普拉斯方差
            input_var, _ = variance_of_laplacian(original_image)

            # 计算中间锐利图像的拉普拉斯方差
            mid_sharp_var, _ = variance_of_laplacian(deblur_image)

            blur_degree = input_var/mid_sharp_var
            print(blur_degree)

            return blur_degree

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, intrinsics=None, fake_sharp=False, sharp_judge=False, stream=None, init = True):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // self.video.down_scale
        wd = image.shape[-1] // self.video.down_scale

        # Keep the raw observation for blur classification and failure gates,
        # while allowing a causal frontend to improve what DROID sees.  This
        # call occurs before feature extraction and therefore also updates the
        # temporal state for frames that will not become keyframes.
        first_frame = self.video.counter.value == 0
        raw_image = image.clone()
        image, streaming_candidate, streaming_replaced, streaming_gain = (
            self._streaming_deblur(image, tstamp, stream)
        )
        if first_frame:
            # Advance a causal backend's history on frame zero, but reproduce
            # the legacy SLAM initialization exactly: DROID, mono depth, and
            # the video buffer all receive the raw first observation.
            image = raw_image
            streaming_replaced = False
            self.last_streaming_selection = "raw_first_frame"
        frame_metadata = self._stream_frame_info(stream, tstamp)
        is_synthetic = bool(frame_metadata.get("synthetic", False))
        source_frame_index = int(frame_metadata.get("source_index", tstamp))
        fixed_source_keyframes = getattr(self, "fixed_source_keyframes", None)
        is_tracking_anchor = bool(
            not is_synthetic
            and (
                (
                    fixed_source_keyframes is not None
                    and source_frame_index in fixed_source_keyframes
                )
                or (
                    fixed_source_keyframes is None
                    and self.tracking_anchor_indices is not None
                    and source_frame_index in self.tracking_anchor_indices
                )
            )
        )
        # Tracker consumes this explicit per-call contract to decide whether
        # Frontend may cull the just-appended observation. Keep it independent
        # from the legacy image/blur return tuple, whose ``image`` field has
        # several unrelated meanings.
        self.last_track_info = {
            "synthetic": is_synthetic,
            "streaming_replaced": bool(streaming_replaced),
            "streaming_evssm_fallback": bool(
                self.last_streaming_evssm_fallback
            ),
            "motion_keyframe": False,
            "tracking_anchor": is_tracking_anchor,
            "appended": False,
        }

        # normalize images
        inputs = self._normalize_droid_input(image)

        # extract features
        gmap = self.__feature_encoder(inputs)

        ### always add first frame to the depth video ###
        if first_frame:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            mono_depth = self._mono_depth_for_frame(tstamp, image, stream)
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0,0], inp[0,0])
            self.last_track_info["appended"] = True
            if streaming_replaced:
                is_blurry = True
                check_score = 0.0
                if sharp_judge:
                    detected = self.classic_blur_check(raw_image.clone())
                    if isinstance(detected, tuple):
                        is_blurry, check_score = detected
                    else:
                        is_blurry = bool(detected)
                elif self.cfg.get("exam_blur_score", False):
                    _, check_score = self.classic_blur_check(raw_image.clone())
                blur_degree = self.deblur_degree_detect(
                    raw_image.clone(), image.clone()
                )
                if self.cfg.get("exam_blur_score", False):
                    return image, blur_degree, True, is_blurry, check_score
                return image, blur_degree, True, is_blurry
        ### only add new frame if there is enough motion ###
        else:
            # In a fixed-keyframe ablation, the preregistered source-index list
            # is the entire admission policy.  TURTLE still advances its causal
            # cache on every frame above, but a streaming/motion recovery may
            # not create an extra keyframe and confound mapper compute.
            fixed_keyframe_policy = fixed_source_keyframes is not None
            use_tracking_anchors = (
                fixed_keyframe_policy
                or self.tracking_anchor_indices is not None
            )
            
            if is_synthetic:
                # Accepted FrameCrafter frames were generated specifically at
                # a blur/motion gap and are weak training observations.  They
                # must not disappear because the original TUM sharp-index list
                # has no entry for an inserted frame.
                is_keyframe = True
            else:
                # index correlation volume
                coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
                corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

                # approximate flow magnitude using 1 update iteration
                _, delta, weight = self.update(self.net[None], self.inp[None], corr)

                motion_keyframe = delta.norm(dim=-1).mean().item() > self.thresh
                self.last_track_info["motion_keyframe"] = bool(motion_keyframe)
                if fixed_keyframe_policy:
                    is_keyframe = is_tracking_anchor
                elif use_tracking_anchors:
                    # Preserve the repository's original tracking anchors and
                    # additionally retain a motion-significant frame when the
                    # streaming frontend actually improved tracking. A cached
                    # EVSSM fallback has ``streaming_replaced=False`` and can
                    # never enter this causal recovery path. Clear-GT
                    # evaluation indices are kept separate in eval_utils.
                    is_keyframe = is_tracking_anchor or (
                        streaming_replaced
                        and not self.last_streaming_evssm_fallback
                        and motion_keyframe
                    )
                else:
                    is_keyframe = motion_keyframe
            check_score = 0.0
            # check motion magnitue / add new frame to video
            if is_keyframe:
                # 跳掉模糊帧，或者将模糊帧初始化为匀速，再用后端细化，且将深度也初始化为单位矩阵
                is_blurry = not is_synthetic
                if sharp_judge and not is_synthetic:
                    if self.cfg["exam_blur_score"]:
                        # img_numpy = image.clone().cpu().numpy()
                        # img_numpy = img_numpy[0]
                        # is_blurry, check_score = self.is_image_blurry_fft(img_numpy)
                        is_blurry, check_score = self.classic_blur_check(raw_image.clone())
                    else:
                        # img_numpy = image.clone().cpu().numpy()
                        # img_numpy = img_numpy[0]
                        # is_blurry, _ = self.is_image_blurry_fft(img_numpy)
                        is_blurry = self.classic_blur_check(raw_image.clone())
                        
                if self.cfg["exam_blur_score"] and not is_synthetic:
                    _, check_score = self.classic_blur_check(raw_image.clone())
                elif is_synthetic:
                    check_score = 1.0
                if fake_sharp and is_blurry: 
                    original_image = raw_image  # Preserve the real observation for gates.
                    sharp_dir = f"./output/sharp_image"
                    # save_tensor(original_image, tstamp, sharp_dir, True)
                    # 在autocast上下文外进行去模糊处理（保持全精度）
                    with torch.cuda.amp.autocast(enabled=False):
                        # 同步CUDA操作，确保准确计时
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        # 开始计时
                        start_time = time.perf_counter()
                        deblurred_image = (
                            streaming_candidate
                            if streaming_candidate is not None
                            else self.apply_evssm_deblur(
                                original_image, stream=stream, timestamp=tstamp
                            )
                        )
                        # 同步CUDA操作
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        # 结束计时
                        end_time = time.perf_counter()
                        
                        # 计算耗时（毫秒）
                        deblur_time_ms = (end_time - start_time) * 1000
                        print(f"{self.deblur_backend_name}去模糊耗时: {deblur_time_ms:.2f} ms")
                        blur_degree = self.deblur_degree_detect(original_image.clone(), deblurred_image.clone())
                    # 初始化的话就算了吧，mapping初始化的时候不要跳过,然后用多点的帧来学锐利结果即可
                    if not init and not self.cfg["only_tracking"]:
                        blur_degree = 0.6
                    streaming_rejected = self._streaming_candidate_rejected(
                        streaming_candidate, streaming_gain
                    )
                    if streaming_rejected or (blur_degree > 0.99 and check_score < 0.45):
                        if streaming_rejected:
                            if not self.last_streaming_candidate_safe:
                                print(
                                    "causal streaming candidate rejected by "
                                    "single-frame EVSSM baseline gate "
                                    f"(gain={self.last_streaming_vs_evssm_gain:.4f})"
                                )
                            else:
                                print(
                                    "streaming deblur candidate rejected by "
                                    f"Laplacian gain gate (gain={streaming_gain:.4f})"
                                )
                            # The causal backend has already consumed this
                            # frame, but its candidate failed the replacement
                            # gate.  Admit the untouched observation as the
                            # new DROID keyframe.  Returning a finite blur
                            # degree defers Mapper notification until Tracker
                            # has recomputed ``curr_kf_idx`` from the appended
                            # video entry; the legacy ``None`` failure path
                            # notified Mapper immediately with the previous
                            # index and could create duplicate camera
                            # parameters in its optimizer window.
                            net, inp = self.__context_encoder(inputs[:, [0]])
                            self.net, self.inp, self.fmap = net, inp, gmap
                            mono_depth = self._mono_depth_for_frame(
                                tstamp, raw_image, stream
                            )
                            self.video.append(
                                tstamp,
                                raw_image[0],
                                None,
                                None,
                                mono_depth,
                                intrinsics / float(self.video.down_scale),
                                gmap,
                                net[0],
                                inp[0],
                            )
                            self.last_track_info["appended"] = True
                            self.count = 0
                            if self.cfg["exam_blur_score"]:
                                return None, 1.0, True, is_blurry, check_score
                            return None, 1.0, True, is_blurry
                        print("deblur失败我直接跳掉")
                        self.count += 1
                        if self.cfg["mono_prior"]["predict_online"] and not self.cfg['only_tracking']:
                            # Keep the cache key in the original source-index
                            # domain.  Augmented timestamps shift after every
                            # inserted FrameCrafter view and would otherwise
                            # make Mapper read the wrong (or a missing) file.
                            self._mono_depth_for_frame(tstamp, image, stream)
                        if self.cfg["exam_blur_score"]:
                            return None, None, True, is_blurry, check_score
                        else:
                            return None, None, True, is_blurry
                    else:
                        self.count = 0
                        # 重新提取去模糊图像的特征
                        # 这部分会使其色域改变，所以直接clone掉
                        deblurred_inputs = self._normalize_droid_input(
                            deblurred_image
                        )
                        gmap_deblurred = self.__feature_encoder(deblurred_inputs)
                        
                        # 重新计算运动
                        coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
                        corr_deblurred = CorrBlock(self.fmap[None,[0]], gmap_deblurred[None,[0]])(coords0)
                        print("this is keyframe")
                        net, inp = self.__context_encoder(deblurred_inputs[:,[0]])
                        self.net, self.inp, self.fmap = net, inp, gmap_deblurred
                        
                        mono_depth = self._mono_depth_for_frame(
                            tstamp, deblurred_image, stream
                        )
                        
                        # 使用去模糊后的图像添加到video
                        self.video.append(tstamp, deblurred_image[0], None, None, mono_depth, 
                                        intrinsics / float(self.video.down_scale), 
                                        gmap_deblurred, net[0], inp[0])
                        self.last_track_info["appended"] = True
                        if self.cfg["exam_blur_score"]:
                            return deblurred_image, blur_degree, True, is_blurry, check_score
                        else:
                            return deblurred_image, blur_degree, True, is_blurry
                        # else:
                            # self.count += 1
                            # return None, None, False, is_blurry
                else:
                    # 如果不是fake sharp的话就退回原来的逻辑，直接加关键帧，不用额外的模糊判断。
                    self.count = 0
                    print("this is keyframe")
                    net, inp = self.__context_encoder(inputs[:,[0]])
                    self.net, self.inp, self.fmap = net, inp, gmap
                    mono_depth = self._mono_depth_for_frame(tstamp, image, stream)
                    self.video.append(tstamp, image[0], None, None, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0], inp[0])
                    self.last_track_info["appended"] = True
                    output_for_mapper = image if streaming_replaced else None
                    if self.cfg["exam_blur_score"]:
                        return output_for_mapper, None, True, is_blurry, check_score
                    else:
                        return output_for_mapper, None, True, is_blurry
            else:
                self.count += 1
                if self.cfg["exam_blur_score"]:
                    return None, None, False, True, None
                else:
                    return None, None, False, True
