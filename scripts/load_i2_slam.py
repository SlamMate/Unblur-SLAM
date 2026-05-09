#!/usr/bin/env python3
"""
读取I2-SLAM数据集的npy文件并进行可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_and_analyze_data(data_dir, frame_idx="0003"):
    """
    加载指定帧的所有数据文件
    
    Args:
        data_dir: 数据目录路径
        frame_idx: 帧索引（例如 "0000"）
    """
    data_path = Path(data_dir)
    
    # 定义文件路径
    files = {
        'depth_cov': data_path / f"{frame_idx}_depth_cov.npy",
        'gt_depth': data_path / f"{frame_idx}_gt_depth.npy",
        'gt_pose': data_path / f"{frame_idx}_gt_pose.npy",
        'input_depth': data_path / f"{frame_idx}_input_depth.npy",
        'input_image': data_path / f"{frame_idx}_input_image.npy",
        'input_pose': data_path / f"{frame_idx}_input_pose.npy",
        'pose_cov': data_path / f"{frame_idx}_pose_cov.npy"
    }
    
    # 加载所有数据
    print("=" * 60)
    print(f"加载帧 {frame_idx} 的数据...")
    print("=" * 60)
    
    data = {}
    for key, filepath in files.items():
        if filepath.exists():
            data[key] = np.load(filepath)
            print(f"✓ 已加载: {filepath.name}")
        else:
            print(f"✗ 文件不存在: {filepath.name}")
    
    print("\n" + "=" * 60)
    print("数据形状信息:")
    print("=" * 60)
    for key, array in data.items():
        print(f"{key:15s}: shape={array.shape}, dtype={array.dtype}")
    
    # 输出GT pose的详细信息
    if 'gt_pose' in data:
        print("\n" + "=" * 60)
        print("Ground Truth Pose 详细信息:")
        print("=" * 60)
        gt_pose = data['gt_pose']
        print(f"形状 (Shape): {gt_pose.shape}")
        print(f"数据类型 (Dtype): {gt_pose.dtype}")
        print(f"\nGT Pose 数值:\n{gt_pose}")
        
        # 如果是4x4变换矩阵，分解显示
        if gt_pose.shape == (4, 4):
            print("\n旋转矩阵 (Rotation Matrix):")
            print(gt_pose[:3, :3])
            print("\n平移向量 (Translation Vector):")
            print(gt_pose[:3, 3])
    
    # 保存输入图像
    if 'input_image' in data:
        print("\n" + "=" * 60)
        print("保存输入图像:")
        print("=" * 60)
        
        input_image = data['input_image']
        
        # 归一化图像到 [0, 1] 或 [0, 255]
        if input_image.max() <= 1.0:
            # 已经是 [0, 1] 范围
            image_to_save = input_image
        else:
            # 归一化到 [0, 1]
            image_to_save = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        
        # 保存为PNG图片
        output_filename = f"{frame_idx}_input_image.png"
        plt.figure(figsize=(10, 8))
        
        # 根据图像维度处理
        if len(image_to_save.shape) == 2:
            # 灰度图
            im = plt.imshow(image_to_save, cmap='gray')
            plt.title(f'Input Image - Frame {frame_idx}')
            plt.colorbar(im)
        elif len(image_to_save.shape) == 3:
            # 多通道图像
            num_channels = image_to_save.shape[2]
            
            if num_channels == 1:
                # 单通道图像
                im = plt.imshow(image_to_save[:, :, 0], cmap='gray')
                plt.title(f'Input Image - Frame {frame_idx}')
                plt.colorbar(im)
            elif num_channels == 3:
                # RGB图像，不添加colorbar
                plt.imshow(image_to_save)
                plt.title(f'Input Image (RGB) - Frame {frame_idx}')
            elif num_channels == 4:
                # RGBA图像，只显示RGB通道（忽略alpha）
                plt.imshow(image_to_save[:, :, :3])
                plt.title(f'Input Image (RGBA, showing RGB) - Frame {frame_idx}')
                print(f"  注意: 检测到4通道图像，只显示前3个通道(RGB)")
            else:
                # 其他通道数，显示第一个通道
                im = plt.imshow(image_to_save[:, :, 0], cmap='gray')
                plt.title(f'Input Image (Channel 0/{num_channels}) - Frame {frame_idx}')
                plt.colorbar(im)
                print(f"  注意: 检测到{num_channels}通道图像，只显示第一个通道")
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 图像已保存为: {output_filename}")
        print(f"  图像值范围: [{input_image.min():.4f}, {input_image.max():.4f}]")
        
        # 如果是4通道图像，额外保存各通道的可视化
        if len(image_to_save.shape) == 3 and image_to_save.shape[2] == 4:
            print(f"\n  保存4通道图像的各通道分析...")
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            channel_names = ['Channel 0 (R)', 'Channel 1 (G)', 'Channel 2 (B)', 'Channel 3 (Alpha/Extra)']
            
            for i in range(4):
                ax = axes[i // 2, i % 2]
                im = ax.imshow(image_to_save[:, :, i], cmap='gray')
                ax.set_title(channel_names[i])
                ax.axis('off')
                plt.colorbar(im, ax=ax)
                
                # 显示通道统计信息
                ch_min = image_to_save[:, :, i].min()
                ch_max = image_to_save[:, :, i].max()
                ch_mean = image_to_save[:, :, i].mean()
                ax.text(0.02, 0.98, f'Range: [{ch_min:.3f}, {ch_max:.3f}]\nMean: {ch_mean:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontsize=8)
            
            plt.tight_layout()
            channels_filename = f"{frame_idx}_input_image_channels.png"
            plt.savefig(channels_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 各通道分析已保存为: {channels_filename}")
    
    # 可选：保存深度图可视化
    if 'input_depth' in data:
        print("\n保存输入深度图可视化:")
        input_depth = data['input_depth']
        output_filename = f"{frame_idx}_input_depth_vis.png"
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(input_depth, cmap='jet')
        plt.title(f'Input Depth - Frame {frame_idx}')
        plt.colorbar(im, label='Depth (m)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 深度图已保存为: {output_filename}")
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    
    return data


if __name__ == "__main__":
    # 默认数据目录
    default_data_dir = "./datasets/I2-SLAM/fr1_desk"
    
    # 如果提供了命令行参数，使用参数指定的目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = default_data_dir
    
    # 如果提供了第二个参数作为帧索引
    if len(sys.argv) > 2:
        frame_idx = sys.argv[2]
    else:
        frame_idx = "0000"
    
    print(f"\n使用数据目录: {data_dir}")
    print(f"读取帧索引: {frame_idx}\n")
    
    # 执行数据加载和分析
    data = load_and_analyze_data(data_dir, frame_idx)