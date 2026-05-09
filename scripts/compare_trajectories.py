#!/usr/bin/env python3
"""
独立的轨迹对比工具
用法: python compare_trajectories.py traj1.npz traj2.npz --output comparison.png
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 尝试导入evo库
try:
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core import metrics
    from evo.tools import plot
except ImportError:
    print("请安装evo库: pip install evo --upgrade --no-binary evo")
    sys.exit(1)


def load_trajectory(npz_path):
    """加载轨迹数据"""
    data = np.load(npz_path)
    
    # 提取信息
    info = {
        'filename': Path(npz_path).name,
        'ate_rmse': float(data.get('ate_rmse', -1)),
        'scale': float(data.get('scale', 1.0))
    }
    
    # 创建轨迹对象
    traj_est = PoseTrajectory3D(
        poses_se3=data['traj_est_poses'],
        timestamps=data['timestamps']
    )
    
    traj_ref = None
    if 'traj_ref_poses' in data:
        traj_ref = PoseTrajectory3D(
            poses_se3=data['traj_ref_poses'],
            timestamps=data['timestamps']
        )
    
    return traj_est, traj_ref, info


def calculate_ate(traj_est, traj_ref):
    """计算ATE (Absolute Trajectory Error)"""
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))
    stats = ape_metric.get_all_statistics()
    return stats['rmse'], ape_metric


def plot_comparison(trajectories, output_path, plot_3d=False):
    """绘制轨迹对比图"""
    
    if plot_3d:
        fig = plt.figure(figsize=(15, 10))
        
        # 2D视图
        ax_2d = fig.add_subplot(121)
        plot_mode_2d = plot.PlotMode.xy
        
        # 3D视图
        ax_3d = fig.add_subplot(122, projection='3d')
        plot_mode_3d = plot.PlotMode.xyz
    else:
        fig = plt.figure(figsize=(12, 10))
        ax_2d = fig.add_subplot(111)
        plot_mode_2d = plot.PlotMode.xy
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'olive']
    linestyles = ['-', '-.', ':', '--']
    
    # 绘制地面真实轨迹（如果存在）
    gt_plotted = False
    for traj_est, traj_ref, info in trajectories:
        if traj_ref is not None and not gt_plotted:
            plot.traj(ax_2d, plot_mode_2d, traj_ref, '--', 'black', 'Ground Truth', alpha=0.7, linewidth=2)
            if plot_3d:
                plot.traj(ax_3d, plot_mode_3d, traj_ref, '--', 'black', 'Ground Truth', alpha=0.7, linewidth=2)
            gt_plotted = True
            break
    
    # 绘制估计轨迹
    for i, (traj_est, traj_ref, info) in enumerate(trajectories):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        label = f"{info['filename'].replace('.npz', '')}"
        if info['ate_rmse'] > 0:
            label += f" (RMSE: {info['ate_rmse']:.4f}m)"
        
        plot.traj(ax_2d, plot_mode_2d, traj_est, linestyle, color, label, linewidth=1.5)
        if plot_3d:
            plot.traj(ax_3d, plot_mode_3d, traj_est, linestyle, color, label, linewidth=1.5)
    
    # 设置图表属性
    ax_2d.set_title('Trajectory Comparison (Top View)', fontsize=14)
    ax_2d.set_xlabel('X [m]', fontsize=12)
    ax_2d.set_ylabel('Y [m]', fontsize=12)
    ax_2d.legend(loc='best', fontsize=10)
    ax_2d.grid(True, alpha=0.3)
    ax_2d.axis('equal')
    
    if plot_3d:
        ax_3d.set_title('Trajectory Comparison (3D View)', fontsize=14)
        ax_3d.set_xlabel('X [m]', fontsize=12)
        ax_3d.set_ylabel('Y [m]', fontsize=12)
        ax_3d.set_zlabel('Z [m]', fontsize=12)
        ax_3d.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存至: {output_path}")
    
    return fig


def plot_error_distribution(trajectories, output_path):
    """绘制误差分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (traj_est, traj_ref, info) in enumerate(trajectories):
        if traj_ref is None:
            continue
            
        color = colors[i % len(colors)]
        label = info['filename'].replace('.npz', '')
        
        # 计算误差
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data((traj_ref, traj_est))
        errors = ape_metric.error
        
        # 误差时序图
        axes[0, 0].plot(errors, color=color, label=label, alpha=0.7)
        
        # 误差直方图
        axes[0, 1].hist(errors, bins=50, alpha=0.5, color=color, label=label, edgecolor='black')
        
        # 误差累积分布
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 0].plot(sorted_errors, cumulative, color=color, label=label, linewidth=2)
        
        # 统计信息
        stats = ape_metric.get_all_statistics()
        stats_text = f"{label}:\n"
        stats_text += f"RMSE: {stats['rmse']:.4f}m\n"
        stats_text += f"Mean: {stats['mean']:.4f}m\n"
        stats_text += f"Median: {stats['median']:.4f}m\n"
        stats_text += f"Std: {stats['std']:.4f}m\n"
        stats_text += f"Max: {stats['max']:.4f}m\n"
        
        axes[1, 1].text(0.1 + i * 0.3, 0.5, stats_text, fontsize=10, 
                       verticalalignment='center', transform=axes[1, 1].transAxes)
    
    # 设置子图标题和标签
    axes[0, 0].set_title('Error Over Time', fontsize=12)
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('Error [m]')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Error Distribution', fontsize=12)
    axes[0, 1].set_xlabel('Error [m]')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Cumulative Error Distribution', fontsize=12)
    axes[1, 0].set_xlabel('Error [m]')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Statistics Summary', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    error_output = output_path.replace('.png', '_errors.png')
    plt.savefig(error_output, dpi=300, bbox_inches='tight')
    print(f"✓ 误差分析图已保存至: {error_output}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='比较多个轨迹文件')
    parser.add_argument('trajectories', nargs='+', help='轨迹NPZ文件路径列表')
    parser.add_argument('--output', '-o', default='trajectory_comparison.png', 
                       help='输出图片路径 (默认: trajectory_comparison.png)')
    parser.add_argument('--plot-3d', action='store_true', 
                       help='同时绘制3D视图')
    parser.add_argument('--plot-errors', action='store_true',
                       help='绘制误差分布分析图')
    parser.add_argument('--recalculate-ate', action='store_true',
                       help='重新计算ATE而不是使用文件中保存的值')
    
    args = parser.parse_args()
    
    # 加载所有轨迹
    print(f"正在加载 {len(args.trajectories)} 个轨迹文件...")
    trajectories = []
    
    for traj_path in args.trajectories:
        if not Path(traj_path).exists():
            print(f"警告: 文件不存在 - {traj_path}")
            continue
            
        traj_est, traj_ref, info = load_trajectory(traj_path)
        
        # 如果需要重新计算ATE
        if args.recalculate_ate and traj_ref is not None:
            ate_rmse, _ = calculate_ate(traj_est, traj_ref)
            info['ate_rmse'] = ate_rmse
            print(f"  ✓ {info['filename']}: RMSE = {ate_rmse:.4f}m")
        else:
            print(f"  ✓ {info['filename']}: RMSE = {info['ate_rmse']:.4f}m")
        
        trajectories.append((traj_est, traj_ref, info))
    
    if not trajectories:
        print("错误: 没有有效的轨迹文件")
        return
    
    # 绘制对比图
    print(f"\n正在生成对比图...")
    plot_comparison(trajectories, args.output, args.plot_3d)
    
    # 绘制误差分析图
    if args.plot_errors:
        print(f"\n正在生成误差分析图...")
        plot_error_distribution(trajectories, args.output)
    
    # 显示图表（可选）
    plt.show()


if __name__ == '__main__':
    main()