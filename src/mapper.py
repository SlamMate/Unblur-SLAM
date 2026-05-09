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
from contextlib import contextmanager
from functools import lru_cache
import cv2
import numpy as np
import open3d as o3d
import torch
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
from copy import deepcopy


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
        
    def set_pipe(self, pipe):
        self.pipe = pipe
        
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

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                # 加强锐利帧的损失，使得系统更专注于锐利帧
                if self.config["sharp_loss_weight"]:
                    loss_weight = 1.0 if viewpoint.is_blurry else self.config.get("sharp_loss_weight_value", 2.0)
                    # 调试信息：验证权重分配是否正确
                    if i == 0 and cam_idx == 0 and self.verbose:
                        self.printer.print(f"Frame {viewpoint.timestamp}: is_blurry={viewpoint.is_blurry}, loss_weight={loss_weight}", FontColor.MAPPER)
                dataset = self.config["dataset"]
                if viewpoint.deblur_fail and not self.config["composite_blur"]:
                    # 如果deblur fail的话，权重应该进一步下降
                    loss_weight = 0.1
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
                                loss_mapping += loss_blur['total']

                        image_ab = image
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
                            loss_mapping += BAD_mapping_loss(
                                self.config, avg_image, gt_image, images_tensor, depths_tensor, viewpoint, seen
                            )
                    else:
                        if self.config["sharp_loss_weight"]:
                            loss_mapping += loss_weight * self.blur_model.compute_BAD_losses(
                                avg_image, gt_image, images_tensor, depths_tensor, viewpoint, scale, seen, mode = "mapping", prev = prev
                            )
                        else:
                            loss_mapping += self.blur_model.compute_BAD_losses(
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
                            loss_mapping += loss_blur['total']
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
                            loss_mapping += get_loss_mapping(
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
                    

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                if self.config["sharp_loss_weight"]:
                    loss_weight = 1.0 if viewpoint.is_blurry else self.config.get("sharp_loss_weight_value", 2.0)
                dataset = self.config["dataset"]
                if viewpoint.deblur_fail and not self.config["composite_blur"]:
                    # 如果deblur fail的话，权重应该进一步下降
                    loss_weight = 0.1
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
                                loss_mapping += loss_blur['total']

                        image_ab = image
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
                            loss_mapping += BAD_mapping_loss(
                                self.config, avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint, seen
                            )
                    else:
                        if self.config["sharp_loss_weight"]:
                            loss_mapping += loss_weight * self.blur_model.compute_BAD_losses(
                                avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint, scale, seen, mode = "mapping", prev = prev
                            )
                        else:
                            loss_mapping += self.blur_model.compute_BAD_losses(
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
                            loss_mapping += loss_blur['total']
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
                            loss_mapping += get_loss_mapping(
                                self.config["mapping"], image, depth, viewpoint, opacity
                            )
                    #loss_mapping += get_loss_mapping(
                    #    self.config["mapping"], image, depth, viewpoint, opacity
                    #)
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)

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
                    if viewpoint.uid == self.initial_frame_uid:
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
                    if viewpoint.uid == self.initial_frame_uid:
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
            intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)
            mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)
            
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
            if self.move_points and self.is_kf.get(keyframe_idx, False) and keyframe_idx not in self.deblur_fail_kfs:
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
            if viewpoint.deblur_fail:
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
                        if cam_idx < frames_to_optimize and not viewpoint.deblur_fail:
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
                        elif viewpoint.deblur_fail:
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

        for i in tqdm(range(iters)):
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
           
            rand_idx = np.random.randint(0, len(random_viewpoint_stack))
            viewpoint = random_viewpoint_stack[rand_idx]
            
            if use_sharp_weight:
                loss_weight = 1.0 if viewpoint.is_blurry else sharp_weight_value
            dataset = self.config["dataset"]
            if viewpoint.deblur_fail and not self.config["composite_blur"]:
                # 如果deblur fail的话，权重应该进一步下降
                loss_weight = 0.1
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
                            loss_mapping += loss_blur['total']
                    image_ab = image
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
                        loss_mapping += BAD_mapping_loss(
                            self.config, avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint, seen = True
                        )
                else:
                    if use_sharp_weight:
                        loss_mapping += loss_weight * self.blur_model.compute_BAD_losses(
                            avg_image, viewpoint.original_image, images_tensor, depths_tensor, viewpoint,scale, seen = True, mode = "mapping", prev = prev
                        )
                    else:
                        loss_mapping += self.blur_model.compute_BAD_losses(
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
                        if vp.uid == 0:  # Skip initial frame
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
                        loss_mapping += loss_blur['total']
                else:
                    if use_sharp_weight:
                        # Standard loss computation
                        loss_mapping += loss_weight * get_loss_mapping(
                            self.config["mapping"], image, depth, viewpoint, opacity
                        )
                    else:
                        loss_mapping += get_loss_mapping(
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
            if i >= iters - 1000 and i % 500 == 0 and self.config["deblur"]["open"]:
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
                q_new = slerp(q_prev_prev, q_prev, 2.0)  
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
                q_new = slerp(q_prev_prev, q_prev, 2.0)  
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

            if self.config["composite_blur"]:
                deblur_fail = True
            
            if init:
                deblur_fail = False

            if self.verbose:
                self.printer.print(f"\nMapping Frame {idx} ...", FontColor.MAPPER)
            
            if is_finished:
                print("Done with Mapping and Tracking")
                break

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
            mono_depth = load_mono_depth(idx, self.save_dir).to(self.device)
            color = color.to(self.device)
            c2w_gt = c2w_gt.to(self.device)
            if not deblur_fail or self.config["composite_blur"]:
                dataset = self.config["dataset"]
                sequence = self.config["scene"]
                # 加载中间锐利图像
                # 默认viewpoint有输入时间戳
                sharp_dir = f"./output/sharp/{dataset}/{sequence}"
                sharp_file = Path(sharp_dir) / f"{idx}.pt"
                mid_sharp = None
                if os.path.exists(sharp_file):
                    mid_sharp = load_tensor(idx, sharp_dir)
                    color = mid_sharp.to(self.device) 

            if deblur_fail and not self.config["composite_blur"] and jump_droid:
                # depth, w2c, invalid = self.get_w2c_and_depth(video_idx, idx, mono_depth, init=False)
                mono_depth = load_mono_depth(idx, self.save_dir).to(self.device)
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
            else:
                self.is_kf[video_idx] = False
                self.pipe.send("continue")
                continue

            last_idx = self.keyframe_idxs[-1]

            # 这个是更新历史所有的位姿，但是这样会覆盖原来优化的多帧位姿
            for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
                if keyframe_idx in self.deblur_fail_kfs:
                    continue
                else:
                    # need to update depth_dict even if the last idx since this is important
                    # for the first deformation of the keyframe
                    mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)
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
                    if cam_idx < frames_to_optimize and not viewpoint.deblur_fail:
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
                elif viewpoint.deblur_fail:
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

            self.pipe.send("continue")
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