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

import json
import os

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import open3d as o3d
import trimesh
import glob
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from thirdparty.monogs.utils.slam_utils import render_video

from thirdparty.monogs.utils.camera_utils import Camera
from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.gaussian_splatting.utils.system_utils import mkdir_p
from src.utils.datasets import load_mono_depth
import pyiqa 

import traceback
from evaluate_3d_reconstruction import run_evaluation

# 添加插值相关的导入
from scipy.spatial.transform import Rotation, Slerp

"""
# Define sharp frame indices for TUM dataset scenes
SHARP_FRAME_INDICES = {
    'TUM/fr1_desk': [13, 20, 21, 35, 44, 60, 61, 65, 67, 72, 82, 85, 87, 88],
    'TUM/fr2_xyz': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
                    38, 39, 40, 41],
    'TUM/fr3_office': [0, 1, 3, 11, 14, 16, 18, 22, 23, 24, 25, 31, 33, 34, 44, 46, 47, 50, 
                       51, 52, 66, 69, 70, 73, 75, 77, 82, 83, 84, 85, 86, 87, 88, 94, 95, 96, 
                       97, 100, 101, 102, 103, 104, 105, 109, 110, 111, 112, 113, 123, 124, 125, 
                       137, 145, 147, 149, 151, 152]
"""

# Define sharp frame indices for TUM dataset scenes
SHARP_FRAME_INDICES = {
    'TUM/fr1_desk': [42, 92, 100, 170, 231, 308, 316, 337, 351, 399, 465, 480, 503, 525],
    'TUM/fr2_xyz': [0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220, 319, 374, 407, 435, 470, 483, 523, 568, 704, 750, 789, 827, 926, 1004, 1160, 1251, 1342, 1409, 1460, 1553, 1692, 1795, 1889, 1978, 2055, 2206, 2282, 2358, 2425, 2590, 2764],
    'TUM/fr3_office': [0, 45, 70, 166, 207, 240, 278, 383, 399, 425, 454, 554, 589, 605, 724, 771, 788, 851, 868, 888, 1030, 1072, 1091, 1111, 1132, 1162, 1228, 1241, 1258, 1275, 1294, 1308, 1316, 1380, 1405, 1427, 1452, 1547, 1567, 1596, 1604, 1628, 1661, 1719, 1741, 1754, 1780, 1801, 1912, 1922, 1935, 2093, 2225, 2345, 2390, 2435, 2455]
}

# ============ 添加的视频插值相关函数 ============
def interpolate_SE(R1, T1, R2, T2, num_points=40):
    """
    在两个相机位姿之间进行球面线性插值
    
    Parameters:
    -----------
    R1, R2: 旋转矩阵 (3x3)
    T1, T2: 平移向量 (3,)
    num_points: 插值点的数量
    
    Returns:
    --------
    Rs: 插值后的旋转矩阵列表
    Ts: 插值后的平移向量列表
    """
    # 将numpy数组转换为正确的格式
    if torch.is_tensor(R1):
        R1 = R1.cpu().numpy()
    if torch.is_tensor(R2):
        R2 = R2.cpu().numpy()
    if torch.is_tensor(T1):
        T1 = T1.cpu().numpy()
    if torch.is_tensor(T2):
        T2 = T2.cpu().numpy()
    
    # 使用Slerp进行旋转插值
    slerp = Slerp([0, 1], Rotation.from_matrix([R1, R2]))
    
    interp_points = np.linspace(0, 1, num_points)
    
    Rs, Ts = [], []
    for alpha in interp_points:
        # 线性插值平移向量
        t = (1 - alpha) * T1 + alpha * T2
        # 球面线性插值旋转矩阵
        R_interp = slerp(alpha).as_matrix()
        Rs.append(R_interp)
        Ts.append(t)
    
    return Rs, Ts

def get_video_cams(frames, num_interp_points=40):
    """
    在关键帧之间插值生成视频相机
    
    Parameters:
    -----------
    frames: 原始相机帧列表
    num_interp_points: 每两帧之间的插值点数量
    
    Returns:
    --------
    video_frames: 插值后的相机帧列表
    """
    if len(frames) < 2:
        return frames
    
    video_frames = []
    
    for idx in range(len(frames) - 1):
        frame_current = frames[idx]
        frame_next = frames[idx + 1]
        
        # 获取当前帧和下一帧的旋转和平移
        Rs, Ts = interpolate_SE(
            frame_current.R, 
            frame_current.T,
            frame_next.R,
            frame_next.T,
            num_points=num_interp_points
        )
        
        # 为每个插值位置创建新的相机
        for k in range(len(Rs)):
            # 创建新的相机对象，复制原始相机的内参和其他属性
            new_cam = Camera(
                uid=frame_current.uid,
                color=frame_current.original_image,
                depth=frame_current.depth,
                gt_T=torch.eye(4, device=frame_current.device),
                projection_matrix=frame_current.projection_matrix,
                fx=frame_current.fx,
                fy=frame_current.fy,
                cx=frame_current.cx,
                cy=frame_current.cy,
                fovx=frame_current.FoVx,
                fovy=frame_current.FoVy,
                image_height=frame_current.image_height,
                image_width=frame_current.image_width,
                device=frame_current.device
            )
            
            # 更新位姿为插值后的值
            new_cam.update_RT(
                torch.tensor(Rs[k], dtype=torch.float32, device=frame_current.device),
                torch.tensor(Ts[k], dtype=torch.float32, device=frame_current.device)
            )
            
            video_frames.append(new_cam)
    
    return video_frames

def render_interpolated_video(
    mapper,
    save_dir,
    iteration="after_refine",
    num_interp_points=40,
    fps=60,
    keyframe_indices=None
):
    """
    渲染插值后的平滑视频
    
    Parameters:
    -----------
    mapper: SLAM mapper对象
    save_dir: 保存目录
    iteration: 迭代名称
    num_interp_points: 每两帧之间的插值点数量
    fps: 视频帧率
    keyframe_indices: 要使用的关键帧索引，None表示使用所有关键帧
    """
    gaussians = mapper.gaussians
    background = mapper.background
    pipe = mapper.pipeline_params
    
    # 创建保存目录
    video_dir = os.path.join(save_dir, iteration, "interpolated_video")
    mkdir_p(video_dir)
    
    # 获取要用于插值的帧
    if keyframe_indices is not None:
        frames_to_interpolate = [mapper.cameras[idx] for idx in keyframe_indices if idx < len(mapper.cameras)]
    else:
        # 使用关键帧
        frames_to_interpolate = [mapper.cameras[idx] for idx in mapper.keyframe_idxs if idx < len(mapper.cameras)]
    
    print(f"Creating interpolated video with {len(frames_to_interpolate)} keyframes...")
    
    # 生成插值相机
    video_frames = get_video_cams(frames_to_interpolate, num_interp_points=num_interp_points)
    
    print(f"Generated {len(video_frames)} interpolated frames")
    
    # 渲染所有帧
    rgb_frames = []
    depth_frames = []
    
    for idx, frame in enumerate(video_frames):
        if idx % 100 == 0:
            print(f"Rendering frame {idx}/{len(video_frames)}")
        
        with torch.no_grad():
            render_pkg = render(gaussians, frame, pipe, background)
            rgb = render_pkg["render"]
            depth = render_pkg["depth"]
            
            # 转换为numpy数组
            rgb_np = rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            depth_np = depth[0].cpu().numpy()
            
            # 归一化深度
            depth_np = depth_np - depth_np.min()
            if depth_np.max() > 0:
                depth_np = depth_np / depth_np.max()
            
            rgb_frames.append((rgb_np * 255).astype(np.uint8))
            depth_frames.append((depth_np * 255).astype(np.uint8))
    
    # 保存视频
    rgb_video_path = os.path.join(video_dir, "interpolated_rgb.mp4")
    depth_video_path = os.path.join(video_dir, "interpolated_depth.mp4")
    
    render_video(rgb_video_path, rgb_frames, fps=fps)
    render_video(depth_video_path, depth_frames, fps=fps)
    
    print(f"Saved interpolated RGB video to: {rgb_video_path}")
    print(f"Saved interpolated depth video to: {depth_video_path}")
    
    return rgb_video_path, depth_video_path

# ============ 原始函数保持不变 ============
def get_sharp_frame_indices(mapper):
    """
    Get sharp frame indices based on mapper configuration for TUM dataset
    """
    # Check if this is a TUM dataset
    if not hasattr(mapper, 'config'):
        return None
    
    dataset_type = mapper.config.get('dataset', '').lower()
    
    # Get the scene name
    scene = mapper.config.get('scene', '').lower()

    # Check if it's a TUM RGB-D dataset
    if dataset_type in ['tumrgbd', 'tumrgb']:
        # Get the scene name
        scene = mapper.config.get('scene', '').lower()
        
        # Map scene name to sharp frame indices
        if 'freiburg1_desk' in scene or 'fr1_desk' in scene:
            return SHARP_FRAME_INDICES['TUM/fr1_desk']
        elif 'freiburg2_xyz' in scene or 'fr2_xyz' in scene:
            return SHARP_FRAME_INDICES['TUM/fr2_xyz']
        elif 'freiburg3_office' in scene or 'fr3_office' in scene:
            return SHARP_FRAME_INDICES['TUM/fr3_office']
        else:
            print(f"Warning: Unknown TUM scene '{scene}', evaluating all frames")
            return None
    
    # Check for deblur/exblur datasets with hold=X files
    elif dataset_type in ['deblur_nerf_motion', 'deblur_nerf_defocus', 'exblurf_motion', 'real_camera_motion_blur', 'deblur_nerf_motion_whole', 'deblur_nerf_motion_no_deblur_no_refine']:
        
        # Get the data path from config
        data_path = mapper.config["data"]["input_folder"]
        scene = mapper.config.get('scene', 'blurball')  # 可以在config中指定scene
        data_path = os.path.join(data_path, scene)
        
        print(data_path)
        # Look for hold=X file in the dataset directory
        hold_files = glob.glob(os.path.join(data_path, 'hold=*'))
        
        if not hold_files:
            print(f"No hold=X file found in {data_path}, evaluating all frames")
            return None
        
        # Extract the hold value from the filename
        hold_file = hold_files[0]  # Take first if multiple exist
        hold_value = int(os.path.basename(hold_file).split('=')[1])
        
        # Get total number of frames
        total_frames = len(mapper.cameras)
        
        # Generate sharp frame indices (multiples of hold_value)
        # Start from hold_value instead of 0 to match the naming convention (e.g., 007.jpg, 014.jpg)
        sharp_indices = [i for i in range(total_frames) if i % hold_value == 0]
        
        print(f"Found hold={hold_value} for dataset {dataset_type}")
        print(f"Sharp frame indices (multiples of {hold_value}): {sharp_indices}")
        
        return sharp_indices
    return None


def eval_rendering(
    mapper,
    save_dir,
    iteration="after_refine",
    monocular=False,
    mesh=False,
    traj_est_aligned=None,
    global_scale=None,
    eval_mesh=True,
    scene=None,
    gt_mesh_path=None,
    render_smooth_video=False,  # 新增参数
    video_interp_points=40,      # 新增参数
    video_fps=60                 # 新增参数
):  
    dataset = mapper.frame_reader
    frames = mapper.cameras
    gaussians = mapper.gaussians
    background = mapper.background
    pipe = mapper.pipeline_params
    video_idxs = mapper.video_idxs
    dataset_type = mapper.config.get('dataset', '').lower()

    mkdir_p(os.path.join(save_dir, iteration))
    render_frames_dir = os.path.join(save_dir, iteration, "rendered_frames")
    mkdir_p(render_frames_dir)

    keyframe_idxs = mapper.keyframe_idxs
    end_idx = len(frames) - 1

    # Get sharp frame indices for TUM dataset (Deblur-SLAM)
    sharp_frame_indices = get_sharp_frame_indices(mapper)
    has_sharp_frames = sharp_frame_indices is not None
    
    if has_sharp_frames:
        dataset_type = mapper.config.get('dataset', '').lower()
        print(f"{dataset_type} dataset detected. Evaluating {len(sharp_frame_indices)} sharp frames only for PSNR.")
        print(f"Sharp frame indices: {sharp_frame_indices}")

    img_pred, img_gt, saved_frame_idx = [], [], []
    
    psnr_array, ssim_array, lpips_array, depth_l1_array = [], [], [], []

    psnr_sharp_array, ssim_sharp_array, lpips_sharp_array, depth_l1_sharp_array = [], [], [], []

    psnr_sharp_only_array = []
    ssim_sharp_only_array = []
    lpips_sharp_only_array = []

    psnr_mid_sharp_array, ssim_mid_sharp_array, lpips_mid_sharp_array = [], [], []

    # 初始化无参考指标数组
    qalign_input_array, qalign_render_array, qalign_ratio_array = [], [], []
    niqe_input_array, niqe_render_array, niqe_ratio_array = [], [], []
    # 添加这些行 - 用于存储评估过程的视频帧
    eval_video_frames = []  # 存储渲染和GT并列的帧

    # 添加这些行 - 用于存储评估过程的视频帧
    eval_video_frames_gt = []  # 存储渲染和GT并列的帧

    global_optimiz_video = []
    global_optimiz_video_sharp = []

    global_frame_idx = []

    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    # 初始化无参考图像质量评估模型
    qalign_model = pyiqa.create_metric('qalign', device='cuda')
    niqe_model = pyiqa.create_metric('niqe', device='cuda')

    if mesh:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    if traj_est_aligned is not None:
        traj_est = traj_est_aligned
    else:
        traj_est = []
        for idx in keyframe_idxs:
            if idx > end_idx:
                break

            frame = frames[idx]
            R = frame.R.detach().cpu().numpy()
            t = frame.T.detach().cpu().numpy()
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            traj_est.append(w2c)
        traj_est = np.stack(traj_est, axis=0)

    traj_est_inv = np.linalg.inv(traj_est)

    plot_dir = os.path.join(save_dir, iteration, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    psnr_max = 0
    psnr_min = 100

    for iteration_idx, idx in enumerate(keyframe_idxs):
        if idx > end_idx:
            break

        frame = frames[idx]
        #color_gt = frame.original_image
        # if monocular:
        #     depth_gt = frame.depth_mono
        # else:
        # depth_gt = frame.depth

        gaussian_orig = gaussians._opacity.clone()

        render_pkg = render(gaussians, frame, pipe, background)
        rasterized_color, rasterized_depth = render_pkg["render"], render_pkg["depth"]

        # 将opacity恢复为原始值
        gaussians._opacity = gaussian_orig

        rasterized_depth = rasterized_depth[0, :, :]

        color_gt, depth_gt = frame.original_image, frame.depth

        color_gt_cpu = color_gt.clone().detach().cpu()
        depth_gt_cpu = depth_gt.clone().detach().cpu()
        
        #convert to numpy
        color_gt_cpu_save = color_gt_cpu.permute(1, 2, 0).numpy()
        depth_gt_cpu_save = depth_gt_cpu.numpy()
        #save for debugging
        #cv2.imwrite(os.path.join(plot_dir, "debug_color_gt_{}.png".format(idx)), (color_gt_cpu_save * 255).astype(np.uint8))

        # 原始图像质量评分
        color_gt_save = torch.clamp(color_gt, 0.0, 1.0)
        rasterized_color_save = torch.clamp(rasterized_color, 0.0, 1.0)

        # 计算无参考图像质量
        # qalign_gt = qalign_model(color_gt_save.unsqueeze(0)).item()
        # qalign_pred = qalign_model(rasterized_color_save.unsqueeze(0)).item()
        # qalign_ratio = qalign_pred / qalign_gt if qalign_gt > 0 else 0
        
        # niqe_gt = niqe_model(color_gt_save.unsqueeze(0)).item()
        # niqe_pred = niqe_model(rasterized_color_save.unsqueeze(0)).item()
        # niqe_ratio = niqe_gt / niqe_pred if niqe_pred > 0 else 0

        # qalign_input_array.append(qalign_gt)
        # qalign_render_array.append(qalign_pred)
        # qalign_ratio_array.append(qalign_ratio)
        
        # niqe_input_array.append(niqe_gt)
        # niqe_render_array.append(niqe_pred)
        # niqe_ratio_array.append(niqe_ratio)

        # RGB Metric Computation
        psnr_value = psnr(rasterized_color.unsqueeze(0), color_gt.unsqueeze(0)).item()
        ssim_value = ssim(rasterized_color.unsqueeze(0), color_gt.unsqueeze(0)).item()

        # LPIPS expects a different dimension order
        lpips_value = cal_lpips(
            rasterized_color_save.unsqueeze(0), color_gt_save.unsqueeze(0), normalize=False
        ).item()

        # Median Depth L1
        mask = depth_gt > 0
        if mask.sum() > 0:
            if global_scale is not None:
                median_depth_gt = torch.median(depth_gt[mask]) * global_scale
                median_depth_rasterized = torch.median(rasterized_depth[mask])
                depth_l1_render = torch.abs(rasterized_depth - depth_gt * global_scale).mean().item()
            else:
                median_depth_gt = torch.median(depth_gt[mask])
                median_depth_rasterized = torch.median(rasterized_depth[mask])
                scale_ratio = median_depth_rasterized / median_depth_gt
                depth_l1_render = torch.abs(rasterized_depth[mask] - depth_gt[mask] * scale_ratio).mean().item()
        else:
            depth_l1_render = 0.0
            median_depth_gt = 0.0
            median_depth_rasterized = 0.0

        psnr_array.append(psnr_value)
        ssim_array.append(ssim_value)
        lpips_array.append(lpips_value)
        depth_l1_array.append(depth_l1_render)

        # 检查是否为sharp frame并相应地添加到数组
        if has_sharp_frames and idx in sharp_frame_indices:
            psnr_sharp_array.append(psnr_value)
            ssim_sharp_array.append(ssim_value)
            lpips_sharp_array.append(lpips_value)
            depth_l1_sharp_array.append(depth_l1_render)
            
            psnr_sharp_only_array.append(psnr_value)
            ssim_sharp_only_array.append(ssim_value)
            lpips_sharp_only_array.append(lpips_value)

        # 保存评估过程的帧（用于视频生成）
        eval_frame = np.hstack([
            color_gt_cpu.permute(1, 2, 0).numpy(),
            rasterized_color_save.cpu().permute(1, 2, 0).numpy()
        ])
        eval_video_frames.append(eval_frame)
        if idx in sharp_frame_indices:
            #print('append')
            eval_video_frames_gt.append(eval_frame)

        # 检查中间帧渲染 (针对motion blur数据集)
        if hasattr(frame, 'deblur_fail') and frame.deblur_fail == True:
            # 获取中间帧的外参
            R_mid, t_mid, _, _ = frame.get_mid_extrinsic()
            
            # 创建中间帧相机用于渲染
            mid_cam = Camera(
                uid=frame.uid,
                color=frame.original_image,
                depth=frame.depth,
                gt_T=torch.eye(4, device=frame.device),
                projection_matrix=frame.projection_matrix,
                fx=frame.fx,
                fy=frame.fy,
                cx=frame.cx,
                cy=frame.cy,
                fovx=frame.FoVx,
                fovy=frame.FoVy,
                image_height=frame.image_height,
                image_width=frame.image_width,
                device=frame.device
            )
            mid_cam.update_RT(R_mid, t_mid)
            
            # 渲染中间帧
            render_pkg_mid = render(gaussians, mid_cam, pipe, background)
            rasterized_color_mid = render_pkg_mid["render"]
            
            # 计算中间帧的指标
            psnr_mid = psnr(rasterized_color_mid.unsqueeze(0), color_gt.unsqueeze(0)).item()
            ssim_mid = ssim(rasterized_color_mid.unsqueeze(0), color_gt.unsqueeze(0)).item()
            lpips_mid = cal_lpips(
                torch.clamp(rasterized_color_mid, 0, 1).unsqueeze(0), 
                color_gt_save.unsqueeze(0), 
                normalize=False
            ).item()
            
            psnr_mid_sharp_array.append(psnr_mid)
            ssim_mid_sharp_array.append(ssim_mid)
            lpips_mid_sharp_array.append(lpips_mid)
            
            print(f"Frame {idx} (Motion blur) - Mid-frame PSNR: {psnr_mid:.2f}, SSIM: {ssim_mid:.4f}, LPIPS: {lpips_mid:.4f}")

        saved_frame_idx.append(idx)

        # Median Depth L1
        mask = depth_gt > 0
        diff_depth_l1 = torch.abs(rasterized_depth - median_depth_gt)
        diff_rgb_l1 = torch.abs(rasterized_color.detach() - color_gt).mean(dim=0).cpu()

        diff_depth_l1_no_scale = torch.abs(rasterized_depth - depth_gt)
        diff_depth_l1_no_scale = diff_depth_l1_no_scale.detach()

        if psnr_value > psnr_max:
            psnr_max = psnr_value
        if psnr_value < psnr_min:
            psnr_min = psnr_value

        plot_rgbd_silhouette(
            color_gt_cpu,
            depth_gt_cpu.squeeze(0),
            rasterized_color_save.cpu(),
            rasterized_depth.cpu().unsqueeze(0),
            diff_depth_l1_no_scale.cpu(),
            psnr_value,
            depth_l1_render,
            plot_dir=plot_dir,
            idx=idx,
            save_plot=True,
            diff_rgb=diff_rgb_l1,
        )

        img_pred.append(rasterized_color.detach())
        img_gt.append(color_gt)

        # For mesh
        if mesh:
            w2c = np.eye(4)
            w2c[:3, :3] = frame.R.detach().cpu().numpy()
            w2c[:3, 3] = frame.T.detach().cpu().numpy()
            im = o3d.geometry.Image(rasterized_color.detach().permute(1, 2, 0).cpu().numpy())
            depth = o3d.geometry.Image(rasterized_depth.detach().cpu().numpy())
            cam = o3d.camera.PinholeCameraIntrinsic(frame.image_width, frame.image_height,
                                                     frame.fx, frame.fy, frame.cx, frame.cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                im, depth, depth_scale=1.0, depth_trunc=100.0, convert_rgb_to_intensity=False)
            volume.integrate(rgbd, cam, traj_est_inv[iteration_idx])

    img_pred_all = torch.stack(img_pred, dim=0)
    img_gt_all = torch.stack(img_gt, dim=0)

    # Save the rendered images
    os.makedirs(os.path.join(
        save_dir, iteration, "rgb_renderings"), exist_ok=True)
    render_images_dir = os.path.join(save_dir, iteration, "rgb_renderings")
    for i, idx in enumerate(saved_frame_idx):
        torchvision_save_image(
            img_pred_all[i],
            os.path.join(render_images_dir, f"{idx:05d}.png"),
        )
        
        # 也保存depth如果需要
        # depth_path = os.path.join(render_images_dir, f"{idx:05d}_depth.png")
        # cv2.imwrite(depth_path, (rasterized_depth.cpu().numpy() * 255).astype(np.uint8))

    # Extract mesh if required
    if mesh:
        o3dmesh = volume.extract_triangle_mesh()
        o3dmesh = clean_mesh(o3dmesh)
        mesh_path = os.path.join(save_dir, iteration, "mesh_online.ply")
        o3d.io.write_triangle_mesh(mesh_path, o3dmesh)
        
        # Mesh evaluation
        if gt_mesh_path is not None and eval_mesh:
            try:
                from mesh_eval import compute_mesh_metrics
                print("Computing mesh metrics...")
                mesh_metrics = compute_mesh_metrics(mesh_path, gt_mesh_path)
                print(f"Mesh metrics: {mesh_metrics}")
            except Exception as e:
                print(f"Failed to compute mesh metrics: {e}")
                mesh_metrics = {}

    avg_psnr_all_frames = torch.tensor(psnr_array).mean()
    avg_ssim_all_frames = torch.tensor(ssim_array).mean()
    avg_lpips_all_frames = torch.tensor(lpips_array).mean()
    if depth_l1_array:
        avg_l1_all_frames = torch.tensor(depth_l1_array).mean()
    else:
        avg_l1_all_frames = torch.tensor(0.0)
    
    output = {
        "mean_psnr_all_frames": float(avg_psnr_all_frames),
        "mean_ssim_all_frames": float(avg_ssim_all_frames),
        "mean_lpips_all_frames": float(avg_lpips_all_frames),
        "mean_depth_l1_all_frames": float(avg_l1_all_frames),
        "num_frames_evaluated": len(psnr_array),
        "total_keyframes": len(keyframe_idxs),
    }
    
    # Sharp frames评估结果(如果有)
    if has_sharp_frames and psnr_sharp_only_array:
        avg_psnr_sharp = torch.tensor(psnr_sharp_only_array).mean()
        avg_ssim_sharp = torch.tensor(ssim_sharp_only_array).mean()
        avg_lpips_sharp = torch.tensor(lpips_sharp_only_array).mean()
        
        output.update({
            "mean_psnr_sharp_frames_only": float(avg_psnr_sharp),
            "mean_ssim_sharp_frames_only": float(avg_ssim_sharp),
            "mean_lpips_sharp_frames_only": float(avg_lpips_sharp),
            "num_sharp_frames_evaluated": len(psnr_sharp_only_array),
        })
    
    # 如果有中间帧评估结果
    if psnr_mid_sharp_array:
        output.update({
            "mean_psnr_mid_frame": float(torch.tensor(psnr_mid_sharp_array).mean()),
            "mean_ssim_mid_frame": float(torch.tensor(ssim_mid_sharp_array).mean()),
            "mean_lpips_mid_frame": float(torch.tensor(lpips_mid_sharp_array).mean()),
            "num_mid_frames_evaluated": len(psnr_mid_sharp_array),
        })
    
    print(f"{'='*60}")
    print(f"Evaluation Results for {iteration}:")
    print(f"{'='*60}")
    print(f'All keyframes - PSNR: {output["mean_psnr_all_frames"]:.2f}, '
          f'SSIM: {output["mean_ssim_all_frames"]:.4f}, '
          f'LPIPS: {output["mean_lpips_all_frames"]:.4f} '
          f'({output["num_frames_evaluated"]}/{output["total_keyframes"]} frames evaluated)')
    
    print(f"{'='*60}")
    if has_sharp_frames and psnr_sharp_only_array:
        print(f'Sharp frames only - PSNR: {output["mean_psnr_sharp_frames_only"]:.2f}, '
              f'SSIM: {output["mean_ssim_sharp_frames_only"]:.4f}, '
              f'LPIPS: {output["mean_lpips_sharp_frames_only"]:.4f} '
              f'({output["num_sharp_frames_evaluated"]}/{output["total_keyframes"]} frames evaluated)')

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )

    # Create gif
    create_gif_from_directory(plot_dir, plot_dir + '/output.gif', online=True)

    # 在返回output之前，保存评估视频
    if len(eval_video_frames) > 0:
        # 保存简单并列版本
        eval_video_path = os.path.join(save_dir, iteration, "eval_comparison_sharp.mp4")
        render_video(eval_video_path, eval_video_frames)
        print(f"Saved evaluation comparison video to: {eval_video_path}")
        
        eval_video_depth_path = os.path.join(save_dir, iteration, "eval_comparison_gt.mp4")
        render_video(eval_video_depth_path, eval_video_frames_gt)
        print(f"Saved evaluation comparison video gt to: {eval_video_depth_path}")

    # 如果启用了插值视频渲染
    if render_smooth_video:
        print("\nRendering interpolated smooth video...")
        render_interpolated_video(
            mapper=mapper,
            save_dir=save_dir,
            iteration=iteration,
            num_interp_points=video_interp_points,
            fps=video_fps,
            keyframe_indices=sharp_frame_indices if has_sharp_frames else None
        )

    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

def plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth, diff_depth_l1,
                         psnr, depth_l1, plot_dir=None, idx=None, 
                         save_plot=True, diff_rgb=None, depth_max=5, opacities=None,
                         scales=None):

    os.makedirs(plot_dir, exist_ok=True)
    # Determine Plot Aspect Ratio
    aspect_ratio = color.shape[2] / color.shape[1]
    fig_height = 8
    fig_width = 14/1.55
    fig_width = fig_width * aspect_ratio
    # Plot the Ground Truth and Rasterized RGB & Depth, along with Diff Depth & Silhouette
    if opacities is not None or scales is not None:
        fig, axs = plt.subplots(2, 4, figsize=(fig_width, fig_height))
    else:
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
    axs[0, 0].imshow(color.cpu().permute(1, 2, 0))
    axs[0, 0].set_title("Ground Truth RGB")
    axs[0, 1].imshow(depth, cmap='jet', vmin=0, vmax=depth_max)
    axs[0, 1].set_title("Input Depth")
    rastered_color = torch.clamp(rastered_color, 0, 1)
    axs[1, 0].imshow(rastered_color.cpu().permute(1, 2, 0))
    axs[1, 0].set_title("Rasterized RGB, PSNR: {:.2f}".format(psnr))
    axs[1, 1].imshow(rastered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=depth_max)
    axs[1, 1].set_title("Rasterized Depth, L1: {:.2f}".format(depth_l1))
    if diff_rgb is not None:
        axs[0, 2].imshow(diff_rgb, cmap='jet', vmin=0, vmax=diff_rgb.max())
        axs[0, 2].set_title("Diff RGB L1")
    diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
    axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=diff_depth_l1.max())
    axs[1, 2].set_title("Diff Depth L1")

    if opacities is not None:
        axs[0, 3].hist(opacities, bins=50, range=(0,1))
        axs[0, 3].set_title('Histogram of Opacities')
        axs[0, 3].set_xlabel('Opacity')
        axs[0, 3].set_ylabel('Frequency')
    if scales is not None:
        axs[1, 3].hist(scales, bins=50, range=(0, scales.max()))
        axs[1, 3].set_title('Histogram of Scales')
        axs[1, 3].set_xlabel('Scale')
        axs[1, 3].set_ylabel('Frequency')
        axs[1, 3].locator_params(axis='x', nbins=6)

    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[0, 2].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    axs[1, 2].axis('off')
    fig.suptitle("frame: " + str(idx), y=0.95, fontsize=16)
    fig.tight_layout()
    if save_plot:
        save_path = os.path.join(plot_dir, f"{idx}.png")
        plt.savefig(save_path, bbox_inches='tight')

    plt.close()


def create_gif_from_directory(directory_path, output_filename, duration=100, online=True):
    """
    Creates a GIF from all PNG images in a given directory.

    :param directory_path: Path to the directory containing PNG images.
    :param output_filename: Output filename for the GIF.
    :param duration: Duration of each frame in the GIF (in milliseconds).
    """

    from PIL import Image
    import re
    # Function to extract the number from the filename
    def extract_number(filename):
        # Pattern to find a number followed by '.png'
        match = re.search(r'(\d+)\.png$', filename)
        if match:
            return int(match.group(1))
        else:
            return None


    if online:
        # Get all PNG files in the directory
        image_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.png')]

        # Sort the files based on the number in the filename
        image_files.sort(key=extract_number)
    else:
        # Get all PNG files in the directory
        image_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.png')]

        # Sort the files based on the number in the filename
        image_files.sort()

    # Load images
    images = [Image.open(file) for file in image_files]

    # Convert images to the same mode and size for consistency
    images = [img.convert('RGBA') for img in images]
    base_size = images[0].size
    resized_images = [img.resize(base_size, Image.LANCZOS) for img in images]

    # Save as GIF
    resized_images[0].save(output_filename, save_all=True, append_images=resized_images[1:], optimize=False, duration=duration, loop=0)


def clean_mesh(mesh):
    mesh_tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(
        mesh.triangles), vertex_colors=np.asarray(mesh.vertex_colors))
    components = trimesh.graph.connected_components(
        edges=mesh_tri.edges_sorted)

    min_len = 100
    components_to_keep = [c for c in components if len(c) >= min_len]

    new_vertices = []
    new_faces = []
    new_colors = []
    vertex_count = 0
    for component in components_to_keep:
        vertices = mesh_tri.vertices[component]
        colors = mesh_tri.visual.vertex_colors[component]

        # Create a mapping from old vertex indices to new vertex indices
        index_mapping = {old_idx: vertex_count +
                         new_idx for new_idx, old_idx in enumerate(component)}
        vertex_count += len(vertices)

        # Select faces that are part of the current connected component and update vertex indices
        faces_in_component = mesh_tri.faces[np.any(
            np.isin(mesh_tri.faces, component), axis=1)]
        reindexed_faces = np.vectorize(index_mapping.get)(faces_in_component)

        new_vertices.extend(vertices)
        new_faces.extend(reindexed_faces)
        new_colors.extend(colors)

    cleaned_mesh_tri = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    cleaned_mesh_tri.visual.vertex_colors = np.array(new_colors)

    cleaned_mesh_tri.remove_degenerate_faces()
    cleaned_mesh_tri.remove_duplicate_faces()
    print(
        f'Mesh cleaning (before/after), vertices: {len(mesh_tri.vertices)}/{len(cleaned_mesh_tri.vertices)}, faces: {len(mesh_tri.faces)}/{len(cleaned_mesh_tri.faces)}')

    cleaned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned_mesh_tri.vertices),
        o3d.utility.Vector3iVector(cleaned_mesh_tri.faces)
    )
    vertex_colors = np.asarray(cleaned_mesh_tri.visual.vertex_colors)[
        :, :3] / 255.0
    cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors.astype(np.float64))

    return cleaned_mesh

# 添加辅助函数用于导入torchvision
def torchvision_save_image(tensor, path):
    """
    保存tensor图像到文件
    """
    import torchvision
    # 确保tensor在[0,1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    torchvision.utils.save_image(tensor, path)