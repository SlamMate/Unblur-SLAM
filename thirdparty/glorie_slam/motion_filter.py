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
from thirdparty.monogs.utils.slam_utils import variance_of_laplacian
from src.utils.datasets import BaseDataset
from src.utils.blur_detector_metrics import BlurDetector

import thirdparty.glorie_slam.geom.projective_ops as pops
from thirdparty.glorie_slam.modules.droid_net import CorrBlock
from src.mono_estimators import get_mono_depth_estimator,predict_mono_depth
from src.utils.datasets import load_mono_depth

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

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
        if cfg["mono_prior"]["predict_online"]:
            self.mono_depth_estimator = get_mono_depth_estimator(cfg)
        if self.cfg["adjust_cam"] and not self.cfg["sharp_judge"]:
            self.detector = BlurDetector(cfg)
        if self.cfg.get("sharp_judge", False):
            self.detector = BlurDetector(cfg)
        if self.cfg.get("fake_sharp", False):
            frontend_name = str((self.cfg.get("deblur", {}) or {}).get("frontend", "evssm")).lower()
            self.evssm_model = None
            if frontend_name == "evssm":
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
        
        # Load sharp frame indices for TUM datasets
        self.sharp_indices = self._load_sharp_indices()
    
    def _load_sharp_indices(self):
        """Load sharp frame indices from predefined files for TUM datasets"""
        dataset_type = self.cfg.get('dataset', '').lower()
        
        # Check if it's a TUM RGB-D dataset
        if dataset_type not in ['tumrgbd', 'tumrgb']:
            return None
        
        # Get the scene name
        scene = self.cfg.get('scene', '').lower()
        
        # Map scene name to file path
        indices_file_map = {
            'freiburg1_desk': './scripts/fr1_desk_indices.txt',
            'fr1_desk': './scripts/fr1_desk_indices.txt',
            'freiburg2_xyz': './scripts/fr2_xyz_indices.txt',
            'fr2_xyz': './scripts/fr2_xyz_indices.txt',
            'freiburg3_office': './scripts/fr3_office_indices.txt',
            'fr3_office': './scripts/fr3_office_indices.txt',
        }
        
        indices_file = None
        for key in indices_file_map:
            if key in scene:
                indices_file = indices_file_map[key]
                break
        
        if indices_file is None:
            print(f"Warning: Unknown TUM scene '{scene}', using motion-based keyframe selection")
            return None
        
        try:
            with open(indices_file, 'r') as f:
                # Read all lines and convert to integers (removing leading zeros)
                indices = set(int(line.strip()) for line in f if line.strip())
            print(f"Loaded {len(indices)} sharp frame indices from {indices_file}")
            return indices
        except Exception as e:
            print(f"Warning: Failed to load sharp indices from {indices_file}: {e}")
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
        
        # Convert back to original format
        if len(original_shape) == 3:
            sharp_image = sharp_image[0]
            if is_hwc:
                sharp_image = sharp_image.permute(1, 2, 0)  # Convert back to [H, W, 3]
        

        """
        sharp_dir = f"./output/sharp_image"
        save_tensor(sharp_image, 1, sharp_dir, True)
        """
        if self.cfg.get("apply_inverse_gamma", False):
            sharp_image = self.processor.encode_srgb(sharp_image)
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

        # normalize images
        inputs = image[None, :, :].to(self.device)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            if self.cfg["mono_prior"]["predict_online"]:
                mono_depth = predict_mono_depth(self.mono_depth_estimator,tstamp,image.clone(),self.cfg,self.device)
            else:
                mono_depth = load_mono_depth(tstamp,self.cfg)
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0,0], inp[0,0])
        ### only add new frame if there is enough motion ###
        else:
            # Check if we should use predefined sharp indices for TUM datasets
            use_sharp_indices = self.sharp_indices is not None
            
            if use_sharp_indices:
                # Use predefined sharp frame indices
                is_keyframe = tstamp in self.sharp_indices
            else:                
                # index correlation volume
                coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
                corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

                # approximate flow magnitude using 1 update iteration
                _, delta, weight = self.update(self.net[None], self.inp[None], corr)

                is_keyframe = delta.norm(dim=-1).mean().item() > self.thresh
            check_score = 0.0
            # check motion magnitue / add new frame to video
            if is_keyframe:
                # 跳掉模糊帧，或者将模糊帧初始化为匀速，再用后端细化，且将深度也初始化为单位矩阵
                is_blurry = True
                if sharp_judge:
                    if self.cfg["exam_blur_score"]:
                        # img_numpy = image.clone().cpu().numpy()
                        # img_numpy = img_numpy[0]
                        # is_blurry, check_score = self.is_image_blurry_fft(img_numpy)
                        is_blurry, check_score = self.classic_blur_check(image.clone())
                    else:
                        # img_numpy = image.clone().cpu().numpy()
                        # img_numpy = img_numpy[0]
                        # is_blurry, _ = self.is_image_blurry_fft(img_numpy)
                        is_blurry = self.classic_blur_check(image.clone())
                        
                if self.cfg["exam_blur_score"]:
                    _, check_score = self.classic_blur_check(image.clone())
                if fake_sharp and is_blurry: 
                    original_image = image.clone()  # Keep original for comparison if needed
                    sharp_dir = f"./output/sharp_image"
                    # save_tensor(original_image, tstamp, sharp_dir, True)
                    # 在autocast上下文外进行去模糊处理（保持全精度）
                    with torch.cuda.amp.autocast(enabled=False):
                        # 同步CUDA操作，确保准确计时
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        # 开始计时
                        start_time = time.perf_counter()
                        deblurred_image = self.apply_evssm_deblur(
                            original_image, stream=stream, timestamp=tstamp
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
                    if blur_degree > 0.99 and check_score < 0.45:
                        print("deblur失败我直接跳掉")
                        self.count += 1
                        if self.cfg["mono_prior"]["predict_online"] and not self.cfg['only_tracking']:
                            # 计算monodepth并进行保存，多帧去拟合monodepth
                            mono_depth = predict_mono_depth(self.mono_depth_estimator,tstamp,image.clone(),self.cfg,self.device)
                        if self.cfg["exam_blur_score"]:
                            return None, None, True, is_blurry, check_score
                        else:
                            return None, None, True, is_blurry
                    else:
                        self.count = 0
                        # 重新提取去模糊图像的特征
                        # 这部分会使其色域改变，所以直接clone掉
                        deblurred_inputs = deblurred_image[None, :, :].to(self.device).clone()
                        deblurred_inputs = deblurred_inputs.sub_(self.MEAN).div_(self.STDV)
                        gmap_deblurred = self.__feature_encoder(deblurred_inputs)
                        
                        # 重新计算运动
                        coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
                        corr_deblurred = CorrBlock(self.fmap[None,[0]], gmap_deblurred[None,[0]])(coords0)
                        print("this is keyframe")
                        net, inp = self.__context_encoder(deblurred_inputs[:,[0]])
                        self.net, self.inp, self.fmap = net, inp, gmap_deblurred
                        
                        if self.cfg["mono_prior"]["predict_online"]:
                            mono_depth = predict_mono_depth(self.mono_depth_estimator, tstamp, 
                                                            deblurred_image.clone(), self.cfg, self.device)
                        else:
                            mono_depth = load_mono_depth(tstamp, self.cfg)
                        
                        # 使用去模糊后的图像添加到video
                        self.video.append(tstamp, deblurred_image[0], None, None, mono_depth, 
                                        intrinsics / float(self.video.down_scale), 
                                        gmap_deblurred, net[0], inp[0])
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
                    if self.cfg["mono_prior"]["predict_online"]:
                        mono_depth = predict_mono_depth(self.mono_depth_estimator,tstamp,image.clone(),self.cfg,self.device)
                    else:
                        mono_depth = load_mono_depth(tstamp,self.cfg)
                    self.video.append(tstamp, image[0], None, None, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0], inp[0])
                    if self.cfg["exam_blur_score"]:
                        return None, None, True, is_blurry, check_score
                    else:
                        return None, None, True, is_blurry
            else:
                self.count += 1
                if self.cfg["exam_blur_score"]:
                    return None, None, False, True, None
                else:
                    return None, None, False, True
