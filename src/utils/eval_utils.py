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
from src.utils.eval_frames import (
    available_clear_gt_source_indices,
    clear_gt_metric_scope,
)
import pyiqa 
import traceback
from evaluate_3d_reconstruction import run_evaluation
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

def get_sharp_frame_indices(mapper):
    """Return the paper-aligned clear-GT source indices for this run."""
    if not hasattr(mapper, 'config'):
        return None
    return available_clear_gt_source_indices(
        mapper.config, getattr(mapper, "frame_reader", None)
    )


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
            render_pkg = render(frame, gaussians, pipe, background)
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
    render_video(rgb_video_path, rgb_frames, fps)
    print(f"Saved interpolated RGB video to: {rgb_video_path}")
    return rgb_video_path


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
    gt_mesh_path=None
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
    metric_scope = (
        clear_gt_metric_scope(mapper.config)
        if has_sharp_frames
        else "all_eval_frames"
    )
    if has_sharp_frames:
        dataset_type = mapper.config.get('dataset', '').lower()
        print(
            f"{dataset_type} dataset detected. Evaluating "
            f"{len(sharp_frame_indices)} sharp frames for {metric_scope}."
        )
        if metric_scope == "clear_gt_prefix_smoke":
            print(
                "WARNING: clear_gt_prefix_smoke is a bounded functionality "
                "check, not a complete paper metric."
            )
        print(f"Sharp frame indices: {sharp_frame_indices}")

    # ==================== 新增: 模糊检测评估初始化 ====================
    blur_detection_stats = {
        'true_positive': 0,   # 真实模糊，预测模糊 (正确)
        'true_negative': 0,   # 真实锐利，预测锐利 (正确)
        'false_positive': 0,  # 真实锐利，预测模糊 (错误)
        'false_negative': 0,  # 真实模糊，预测锐利 (错误)
        'total_frames': 0,
        'details': []  # 存储每帧的详细信息
    }
    # 检查是否是支持模糊检测评估的数据集
    blur_eval_datasets = ['deblur_nerf_motion', 'deblur_nerf_defocus', 'exblurf_motion', 
                          'real_camera_motion_blur', 'deblur_nerf_motion_whole', 
                          'deblur_nerf_motion_no_deblur_no_refine']
    # This run's paper metric scope is clear-GT only.  A blur-detector
    # confusion matrix needs the complementary blurry labels and therefore
    # must not be fabricated from this filtered evaluation pass.
    evaluate_blur_detection = False
    
    if evaluate_blur_detection:
        print(f"\nBlur detection evaluation enabled for dataset: {dataset_type}")
    # ==================== 新增结束 ====================

    img_pred, img_gt, saved_frame_idx = [], [], []
    psnr_array, ssim_array, lpips_array, depth_l1_array = [], [], [], []
    psnr_sharp_array, ssim_sharp_array, lpips_sharp_array, depth_l1_sharp_array = [], [], [], []
    psnr_sharp_only_array = []
    ssim_sharp_only_array = []
    lpips_sharp_only_array = []
    psnr_mid_sharp_array, ssim_mid_sharp_array, lpips_mid_sharp_array = [], [], []
    evaluated_clear_sources = set()
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
    # No-reference IQA metrics. Disable with UNBLUR_SKIP_NR_IQA=1 (e.g. on a 24
    # GB GPU where the QAlign LLaMA backbone OOMs alongside the SLAM state).
    skip_nr_iqa = os.environ.get("UNBLUR_SKIP_NR_IQA", "").lower() in ("1", "true", "yes")
    if skip_nr_iqa:
        print("[eval_rendering] UNBLUR_SKIP_NR_IQA=1 -> skipping QAlign+NIQE")
        qalign_model = None
        niqe_model = None
    else:
        qalign_model = pyiqa.create_metric('qalign', device='cuda')
        niqe_model = pyiqa.create_metric('niqe', device='cuda')
    if mesh:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for k, (kf_idx, video_idx) in enumerate(zip(keyframe_idxs, video_idxs)):
        # FrameCrafter views are weak training observations, never evaluation
        # targets.  Keeping this guard here prevents a sharp-looking generated
        # frame from inflating PSNR/SSIM/LPIPS.
        if hasattr(dataset, "is_eval_frame") and not dataset.is_eval_frame(kf_idx):
            print(f"[eval_rendering] skipping synthetic/non-eval frame {kf_idx}")
            continue
        source_kf_idx = (
            dataset.source_frame_index(kf_idx)
            if hasattr(dataset, "source_frame_index")
            else kf_idx
        )
        # 定期清理GPU缓存
        if k % 10 == 0:
            torch.cuda.empty_cache()
        # 检查GPU内存
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            if memory_reserved > 45:  # 如果超过10GB，警告
                print(f"Warning: High GPU memory usage: {memory_reserved:.2f} GB")
        # Check if current frame is a sharp frame
        is_sharp_frame = False
        if has_sharp_frames and source_kf_idx in sharp_frame_indices:
            print("indice is", k)
            print("kf_idx is", kf_idx)
            is_sharp_frame = True
        if has_sharp_frames and not is_sharp_frame:
            print(
                f"[eval_rendering] skipping non-clear-GT frame {kf_idx} "
                f"(source {source_kf_idx})"
            )
            continue

        # ==================== 新增: 模糊检测评估 ====================
        if evaluate_blur_detection:
            frame = frames[video_idx]
            # 获取系统预测的is_blurry属性
            predicted_blurry = getattr(frame, 'is_blurry', None)
            
            if predicted_blurry is not None:
                # Ground truth: 如果kf_idx在sharp_frame_indices中，则是锐利帧(gt_blurry=False)
                gt_is_sharp = source_kf_idx in sharp_frame_indices
                gt_blurry = not gt_is_sharp
                
                blur_detection_stats['total_frames'] += 1
                
                if gt_blurry and predicted_blurry:
                    blur_detection_stats['true_positive'] += 1
                    result = 'TP'
                elif not gt_blurry and not predicted_blurry:
                    blur_detection_stats['true_negative'] += 1
                    result = 'TN'
                elif not gt_blurry and predicted_blurry:
                    blur_detection_stats['false_positive'] += 1
                    result = 'FP'
                else:  # gt_blurry and not predicted_blurry
                    blur_detection_stats['false_negative'] += 1
                    result = 'FN'
                
                blur_detection_stats['details'].append({
                    'kf_idx': kf_idx,
                    'video_idx': video_idx,
                    'gt_is_sharp': gt_is_sharp,
                    'predicted_blurry': predicted_blurry,
                    'result': result
                })
        # ==================== 新增结束 ====================

        saved_frame_idx.append(video_idx)
        frame = frames[video_idx]
        _, gt_image, gt_depth, _, gt_images = dataset[kf_idx]
        # retrieve mono depth
        mono_depth = load_mono_depth(source_kf_idx, save_dir).to("cuda:0")
        
        # 对于 deblur_fail 帧 (负数 video_idx)，跳过需要 Droid-SLAM 深度的操作，但仍然渲染
        is_deblur_fail_frame = video_idx < 0
        if is_deblur_fail_frame:
            # 对于 deblur_fail 帧，直接使用 mono_depth 作为 sensor_depth
            sensor_depth = mono_depth.cpu()
            invalid = False
            print(f"[eval_rendering] Processing deblur_fail frame: kf_idx={kf_idx}, video_idx={video_idx}")
        else:
            # retrieve sensor 
            sensor_depth, _, invalid = mapper.get_w2c_and_depth(video_idx, kf_idx, mono_depth, init=False)
            sensor_depth = sensor_depth.cpu()
        if gt_depth is not None:
            gt_depth = gt_depth.cpu().numpy()
        else:
            gt_depth = sensor_depth
        gt_image = gt_image.squeeze().to("cuda:0")
        rendering_pkg = render(frame, gaussians, pipe, background)
        rendering = rendering_pkg["render"].detach()
        depth = rendering_pkg["depth"].detach()
        gt = (gt_image.squeeze().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        # include optimized exposure compensation
        image = (torch.exp(frame.exposure_a.detach())) * rendering + frame.exposure_b.detach()
        image = torch.clamp(image, 0.0, 1.0)
        # 保存渲染帧到单独文件夹
        render_frame_path = os.path.join(render_frames_dir, f"kf_{kf_idx:06d}.png")
        render_image_save = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        render_image_save = cv2.cvtColor(render_image_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(render_frame_path, render_image_save)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        # 计算无参考指标
        with torch.no_grad():
            # 将图像转换为适合pyiqa的格式 [B, C, H, W]
            gt_img_batch = gt_image.unsqueeze(0)
            render_img_batch = image.unsqueeze(0)
            if qalign_model is not None:
                qalign_input = qalign_model(gt_img_batch).item()
                qalign_render = qalign_model(render_img_batch).item()
                qalign_ratio = qalign_input / (qalign_render + 1e-8)
            else:
                qalign_input = qalign_render = qalign_ratio = float('nan')
            if niqe_model is not None:
                niqe_input = niqe_model(gt_img_batch).item()
                niqe_render = niqe_model(render_img_batch).item()
                niqe_ratio = niqe_input / (niqe_render + 1e-8)
            else:
                niqe_input = niqe_render = niqe_ratio = float('nan')
            qalign_input_array.append(qalign_input)
            qalign_render_array.append(qalign_render)
            qalign_ratio_array.append(qalign_ratio)
            niqe_input_array.append(niqe_input)
            niqe_render_array.append(niqe_render)
            niqe_ratio_array.append(niqe_ratio)
        video_frame = (torch.clamp(torch.cat((image, gt_image), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
        if gt_images is not None and len(gt_images) > 0:
            try:
                # 对于合成视频，采用中间帧作为gt
                gt_sharp = gt_images[int(0.5*len(gt_images))].squeeze().to("cuda:0")
                mask = gt_sharp > 0
                video_frame_sharp = (torch.clamp(torch.cat((image, gt_sharp), dim=2).detach().clone().cpu().permute(1, 2, 0), 0, 1) * 255).type(torch.uint8)
                psnr_score_sharp = psnr((image[mask]).unsqueeze(0), (gt_sharp[mask]).unsqueeze(0))
                ssim_score_sharp = ssim((image).unsqueeze(0), (gt_sharp).unsqueeze(0))
                lpips_score_sharp = cal_lpips((image).unsqueeze(0), (gt_sharp).unsqueeze(0))
                psnr_sharp_array.append(psnr_score_sharp.item())
                ssim_sharp_array.append(ssim_score_sharp.item())
                lpips_sharp_array.append(lpips_score_sharp.item())
                # 计算中间帧的sharp指标
                mid_index = int(0.5 * len(gt_images))
                gt_mid_sharp = gt_images[mid_index].squeeze().to("cuda:0")
                mask_mid = gt_mid_sharp > 0
                psnr_score_mid_sharp = psnr((image[mask_mid]).unsqueeze(0), (gt_mid_sharp[mask_mid]).unsqueeze(0))
                ssim_score_mid_sharp = ssim((image).unsqueeze(0), (gt_mid_sharp).unsqueeze(0))
                lpips_score_mid_sharp = cal_lpips((image).unsqueeze(0), (gt_mid_sharp).unsqueeze(0))
                psnr_mid_sharp_array.append(psnr_score_mid_sharp.item())
                ssim_mid_sharp_array.append(ssim_score_mid_sharp.item())
                lpips_mid_sharp_array.append(lpips_score_mid_sharp.item())
                global_optimiz_video_sharp.append(video_frame_sharp)
            except Exception as e:
                print(f"Warning: Failed to process gt_images for frame {k}: {e}")
                print(f"  gt_images type: {type(gt_images)}, length: {len(gt_images) if gt_images else 'N/A'}")
                # Fallback to using gt_image
                mask = gt_image > 0
                psnr_score_sharp = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
                ssim_score_sharp = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
                lpips_score_sharp = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))
                psnr_sharp_array.append(psnr_score_sharp.item())
                ssim_sharp_array.append(ssim_score_sharp.item())
                lpips_sharp_array.append(lpips_score_sharp.item())
                psnr_mid_sharp_array.append(psnr_score_sharp.item())
                ssim_mid_sharp_array.append(ssim_score_sharp.item())
                lpips_mid_sharp_array.append(lpips_score_sharp.item())
                global_optimiz_video_sharp.append(video_frame)
        else:
            # Fallback: use gt_image as sharp reference if gt_images is not available
            print(f"Warning: gt_images is None or empty for frame {k} (kf_idx={kf_idx}), using gt_image as fallback")
            mask = gt_image > 0
            psnr_score_sharp = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            ssim_score_sharp = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
            lpips_score_sharp = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))
            psnr_sharp_array.append(psnr_score_sharp.item())
            ssim_sharp_array.append(ssim_score_sharp.item())
            lpips_sharp_array.append(lpips_score_sharp.item())
            psnr_mid_sharp_array.append(psnr_score_sharp.item())
            ssim_mid_sharp_array.append(ssim_score_sharp.item())
            lpips_mid_sharp_array.append(lpips_score_sharp.item())
            global_optimiz_video_sharp.append(video_frame)
        global_optimiz_video.append(video_frame)
        global_frame_idx.append(kf_idx)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)
        mask = gt_image > 0 
        gt_depth = torch.tensor(gt_depth)
        depth = depth.detach().cpu()
        # compute depth errors
        depth_mask = (depth > 0) * (gt_depth > 0)
        # depth = global_scale*depth
        diff_depth_l1 = torch.abs(depth - gt_depth)
        diff_depth_l1_gt = diff_depth_l1 * depth_mask
        depth_l1_gt = diff_depth_l1_gt.sum() / depth_mask.sum()
        if not has_sharp_frames or is_sharp_frame:
            depth_l1_array.append(depth_l1_gt)
        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))
        eval_video_frames.append((torch.clamp(
            image.detach().clone().cpu().permute(1, 2, 0), 
            0, 1
        ) * 255).type(torch.uint8))
        eval_video_frames_gt.append((torch.clamp(
            gt_image.detach().clone().cpu().permute(1, 2, 0), 
            0, 1
        ) * 255).type(torch.uint8))
        # For datasets with sharp frames, only add to psnr_array if it's a sharp frame
        # For datasets with sharp frames, only add to psnr_array if it's a sharp frame
        if has_sharp_frames:
            if is_sharp_frame:
                psnr_array.append(psnr_score.item())
                evaluated_clear_sources.add(int(source_kf_idx))
                psnr_sharp_only_array.append(psnr_score.item())
                ssim_array.append(ssim_score.item())
                lpips_array.append(lpips_score.item())
                ssim_sharp_only_array.append(ssim_score.item())
                lpips_sharp_only_array.append(lpips_score.item())
                print(f"Frame {k} (kf_idx: {kf_idx}) is a sharp frame, PSNR: {psnr_score.item():.2f}")
            else:
                print(f"Frame {k} (kf_idx: {kf_idx}) is not a sharp frame, skipping PSNR evaluation")
        else:
            psnr_array.append(psnr_score.item())
            ssim_array.append(ssim_score.item())
            lpips_array.append(lpips_score.item())
        # Add plotting 2x3 grid here
        plot_dir = save_dir + "/plots_" + iteration
        plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1_gt,
                                 psnr_score.item(), depth_l1_gt, plot_dir=plot_dir, idx='video_idx_' + str(video_idx) + "_kf_idx_" + str(kf_idx),
                                 diff_rgb=np.abs(gt - pred))
        has_gt_poses = traj_est_aligned is not None
        # do volumetric TSDF fusion from which the mesh will be extracted later
        if mesh and has_gt_poses:
            # mask out the pixels where the GT mesh is non-existent. Do this with the gt depth mask
            depth[gt_depth.unsqueeze(0) == 0] = 0
            depth_o3d = np.ascontiguousarray(depth.permute(1, 2, 0).numpy().astype(np.float32))
            depth_o3d = o3d.geometry.Image(depth_o3d)
            color_o3d = np.ascontiguousarray((np.clip(image.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)*255.0).astype(np.uint8))
            color_o3d = o3d.geometry.Image(color_o3d)
            w2c_o3d = np.linalg.inv(traj_est_aligned[k]) # convert from c2w to w2c
            fx = frame.fx
            fy = frame.fy
            cx = frame.cx
            cy = frame.cy
            W =  depth.shape[-1]
            H = depth.shape[1]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=30,
                convert_rgb_to_intensity=False)
            # use gt pose for debugging
            # w2c_o3d = torch.linalg.inv(pose).cpu().numpy() @ dataset.w2c_first_pose
            volume.integrate(rgbd, intrinsic, w2c_o3d)
    if has_sharp_frames and evaluated_clear_sources != set(sharp_frame_indices):
        missing = sorted(set(sharp_frame_indices) - evaluated_clear_sources)
        extra = sorted(evaluated_clear_sources - set(sharp_frame_indices))
        raise RuntimeError(
            "rendering evaluation did not cover the complete configured "
            "clear-GT scope; refusing an incomplete metric: "
            f"missing={missing}, extra={extra}"
        )
    if has_sharp_frames and len(psnr_array) != len(evaluated_clear_sources):
        raise RuntimeError(
            "duplicate clear-GT source frames reached rendering evaluation; "
            "refusing a reweighted metric"
        )

    print("\n=== Rendering 3 overview perspectives ===")
    overview_dir = os.path.join(save_dir, iteration, "overview_renders")
    mkdir_p(overview_dir)
    # 随机选择3个不同的关键帧作为基础视角
    num_overview_views = min(3, len(keyframe_idxs))  # 确保不超过关键帧数量
    selected_indices = np.random.choice(len(keyframe_idxs), num_overview_views, replace=False)
    for view_idx, sel_idx in enumerate(selected_indices):
        kf_idx = keyframe_idxs[sel_idx]
        video_idx = video_idxs[sel_idx]
        if video_idx < 0:
            continue
        base_frame = frames[video_idx]
        # 创建一个新的相机，将其向后移动
        # 计算当前相机的位置（C = -R^T * T）
        R_mat = base_frame.R.cpu().numpy()
        T_vec = base_frame.T.cpu().numpy()
        cam_pos = -R_mat.T @ T_vec
        # 相机的观察方向是R矩阵的第三行（z轴）
        view_direction = R_mat[2, :]
        # 将相机向后移动（沿着负z方向）
        # 移动距离可以根据场景大小调整
        move_distance = 0.6  # 可以根据需要调整这个值
        new_cam_pos = cam_pos - view_direction * move_distance
        # 计算新的T向量
        new_T = -R_mat @ new_cam_pos
        # 创建新的Camera对象（复制原相机但使用新的位置）
        from copy import deepcopy
        overview_cam = deepcopy(base_frame)
        overview_cam.T = torch.tensor(new_T, dtype=torch.float32, device=base_frame.T.device)
        # 渲染全景视图
        rendering_pkg = render(overview_cam, gaussians, pipe, background)
        rendering = rendering_pkg["render"].detach()
        depth = rendering_pkg["depth"].detach()
        # 应用曝光补偿
        image = (torch.exp(base_frame.exposure_a.detach())) * rendering + base_frame.exposure_b.detach()
        image = torch.clamp(image, 0.0, 1.0)
        # 保存渲染结果
        overview_image = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        overview_depth = depth.detach().cpu().numpy()
        # 保存RGB图像
        overview_rgb_path = os.path.join(overview_dir, f"overview_{view_idx:02d}_rgb.png")
        cv2.imwrite(overview_rgb_path, cv2.cvtColor(overview_image, cv2.COLOR_RGB2BGR))
        # 保存深度图可视化
        overview_depth_vis = (overview_depth[0] / overview_depth[0].max() * 255).astype(np.uint8)
        overview_depth_vis_colored = cv2.applyColorMap(overview_depth_vis, cv2.COLORMAP_JET)
        overview_depth_path = os.path.join(overview_dir, f"overview_{view_idx:02d}_depth.png")
        cv2.imwrite(overview_depth_path, overview_depth_vis_colored)
        # 创建并保存组合图像（RGB + Depth并排）
        combined = np.hstack([overview_image, cv2.cvtColor(overview_depth_vis_colored, cv2.COLOR_BGR2RGB)])
        combined_path = os.path.join(overview_dir, f"overview_{view_idx:02d}_combined.png")
        cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"  Saved overview view {view_idx} (based on keyframe {kf_idx})")
    if mesh:
        # Mesh the final volumetric model
        mesh_out_file = os.path.join(save_dir, iteration, "mesh.ply")
        o3d_mesh = volume.extract_triangle_mesh()
        o3d_mesh = clean_mesh(o3d_mesh)
        o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
        print('Meshing finished.')
        # evaluate the mesh
        if eval_mesh:
            try:
                pred_ply = mesh_out_file.split('/')[-1]
                last_slash_index = mesh_out_file.rindex('/')
                path_to_pred_ply = mesh_out_file[:last_slash_index]
                gt_mesh = gt_mesh_path
                result_3d = run_evaluation(pred_ply, path_to_pred_ply, "mesh",
                                           distance_thresh=0.05, full_path_to_gt_ply=gt_mesh, icp_align=True)
                print(f"3D Mesh evaluation: {result_3d}")
            except Exception as e:
                traceback.print_exception(e)
    if has_sharp_frames and not psnr_array:
        raise RuntimeError(
            "no paper clear-GT frame reached rendering evaluation; refusing "
            "to write NaN/all-frame fallback metrics"
        )
    output = dict()
    output["metric_scope"] = metric_scope
    output["num_evaluated_frames"] = len(psnr_array)
    output["evaluated_source_indices"] = (
        sorted(evaluated_clear_sources) if has_sharp_frames else None
    )
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_depthl1"] = float(np.mean(depth_l1_array))  # 添加这行
    output["mean_psnr_sharp"] = float(np.mean(psnr_sharp_array))
    output["mean_ssim_sharp"] = float(np.mean(ssim_sharp_array))
    output["mean_lpips_sharp"] = float(np.mean(lpips_sharp_array))
    output["mean_psnr_mid_sharp"] = float(np.mean(psnr_mid_sharp_array))
    output["mean_ssim_mid_sharp"] = float(np.mean(ssim_mid_sharp_array))
    output["mean_lpips_mid_sharp"] = float(np.mean(lpips_mid_sharp_array))
    # 添加无参考指标统计
    output["mean_qalign_input"] = float(np.mean(qalign_input_array))
    output["mean_qalign_render"] = float(np.mean(qalign_render_array))
    output["mean_qalign_ratio"] = float(np.mean(qalign_ratio_array))
    output["std_qalign_ratio"] = float(np.std(qalign_ratio_array))
    output["min_qalign_ratio"] = float(np.min(qalign_ratio_array))
    output["max_qalign_ratio"] = float(np.max(qalign_ratio_array))
    output["mean_niqe_input"] = float(np.mean(niqe_input_array))
    output["mean_niqe_render"] = float(np.mean(niqe_render_array))
    output["mean_niqe_ratio"] = float(np.mean(niqe_ratio_array))
    output["std_niqe_ratio"] = float(np.std(niqe_ratio_array))
    output["min_niqe_ratio"] = float(np.min(niqe_ratio_array))
    output["max_niqe_ratio"] = float(np.max(niqe_ratio_array))
    if has_sharp_frames and psnr_sharp_only_array:
        output["mean_psnr_sharp_frames_only"] = float(np.mean(psnr_sharp_only_array))
        output["mean_ssim_sharp_frames_only"] = float(np.mean(ssim_sharp_only_array))
        output["mean_lpips_sharp_frames_only"] = float(np.mean(lpips_sharp_only_array))
        output["num_sharp_frames_evaluated"] = len(psnr_sharp_only_array)
        output["total_keyframes"] = len(keyframe_idxs)

    # ==================== 新增: 添加模糊检测评估结果到output ====================
    if evaluate_blur_detection and blur_detection_stats['total_frames'] > 0:
        tp = blur_detection_stats['true_positive']
        tn = blur_detection_stats['true_negative']
        fp = blur_detection_stats['false_positive']
        fn = blur_detection_stats['false_negative']
        total = blur_detection_stats['total_frames']
        
        # 计算各种指标
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        output["blur_detection"] = {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "total_frames": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "correctly_identified_blurry": tp,
            "incorrectly_identified_as_blurry": fp,
            "correctly_identified_sharp": tn,
            "missed_blurry_frames": fn
        }
        
        # 打印模糊检测评估结果
        print(f"\n{'='*60}")
        print(f"Blur Detection Evaluation Results:")
        print(f"  Total frames evaluated: {total}")
        print(f"  True Positive (correctly identified blurry): {tp}")
        print(f"  True Negative (correctly identified sharp): {tn}")
        print(f"  False Positive (incorrectly marked as blurry): {fp}")
        print(f"  False Negative (missed blurry frames): {fn}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
        print(f"{'='*60}")
    # ==================== 新增结束 ====================

    print(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, depth l1: {output["mean_depthl1"]}', #, depth l1 sensor: {output["mean_depthl1_sensor"]}, depth l1 to sensor: {output["mean_depthl1_to_sensor"]}', 
        f'psnr sharp: {output["mean_psnr_sharp"]}, ssim sharp: {output["mean_ssim_sharp"]}, lpips sharp: {output["mean_lpips_sharp"]},',
        f'psnr mid sharp: {output["mean_psnr_mid_sharp"]}, ssim mid sharp: {output["mean_ssim_mid_sharp"]}, lpips mid sharp: {output["mean_lpips_mid_sharp"]}', 
    )
    # 打印无参考指标结果
    print(f"\n{'='*60}")
    print(f"QAlign Metrics:")
    print(f"  Mean Input QAlign: {output['mean_qalign_input']:.4f}")
    print(f"  Mean Render QAlign: {output['mean_qalign_render']:.4f}")
    print(f"  Mean Ratio (Input/Render): {output['mean_qalign_ratio']:.4f} ± {output['std_qalign_ratio']:.4f}")
    print(f"  Best Ratio (lowest): {output['min_qalign_ratio']:.4f}")
    print(f"  Worst Ratio (highest): {output['max_qalign_ratio']:.4f}")
    print(f"\nNIQE Metrics:")
    print(f"  Mean Input NIQE: {output['mean_niqe_input']:.4f}")
    print(f"  Mean Render NIQE: {output['mean_niqe_render']:.4f}")
    print(f"  Mean Ratio (Input/Render): {output['mean_niqe_ratio']:.4f} ± {output['std_niqe_ratio']:.4f}")
    print(f"  Best Ratio (lowest): {output['min_niqe_ratio']:.4f}")
    print(f"  Worst Ratio (highest): {output['max_niqe_ratio']:.4f}")
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

    # ==================== 新增: 保存模糊检测详细信息 ====================
    if evaluate_blur_detection and blur_detection_stats['total_frames'] > 0:
        blur_details_path = os.path.join(psnr_save_dir, "blur_detection_details.json")
        json.dump(blur_detection_stats, open(blur_details_path, "w", encoding="utf-8"), indent=4)
        print(f"Blur detection details saved to: {blur_details_path}")
    # ==================== 新增结束 ====================

    # Create gif
    create_gif_from_directory(plot_dir, plot_dir + '/output.gif', online=True)
    # 在返回output之前，保存评估视频
    if len(eval_video_frames) > 0:
        # 保存简单并列版本
        eval_video_path = os.path.join(save_dir, iteration, "eval_comparison_sharp.mp4")
        render_video(eval_video_path, eval_video_frames, 5)
        print(f"Saved evaluation comparison video to: {eval_video_path}")
        eval_video_depth_path = os.path.join(save_dir, iteration, "eval_comparison_gt.mp4")
        render_video(eval_video_depth_path, eval_video_frames_gt, 5)
        print(f"Saved evaluation comparison video gt to: {eval_video_depth_path}")
    print("\nRendering interpolated smooth video...")
    render_interpolated_video(
        mapper=mapper,
        save_dir=save_dir,
        iteration=iteration,
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
