#!/usr/bin/env python3
"""
轨迹3D对比工具 - 适用于集群环境(无需显示)
用法: python compare_trajectories_3d.py traj1.npz traj2.npz -o output.png
"""

import numpy as np
import argparse
import sys
from pathlib import Path

# 设置matplotlib后端为非交互式（适用于集群）
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 尝试导入evo库
try:
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core import metrics
    from evo.tools import plot
except ImportError:
    print("错误: 请安装evo库")
    print("运行: pip install evo --upgrade --no-binary evo")
    sys.exit(1)


class TrajectoryComparator:
    """轨迹对比器类"""
    
    def __init__(self):
        self.trajectories = []
        self.gt_trajectory = None
        
    def load_trajectory(self, npz_path):
        """
        加载轨迹数据
        
        Args:
            npz_path: npz文件路径
        Returns:
            traj_est: 估计轨迹
            traj_ref: 参考轨迹（GT）
            info: 轨迹信息字典
        """
        print(f"加载轨迹: {npz_path}")
        
        if not Path(npz_path).exists():
            raise FileNotFoundError(f"文件不存在: {npz_path}")
        
        data = np.load(npz_path)
        
        # 提取基本信息
        info = {
            'filename': Path(npz_path).stem,  # 不包含扩展名
            'full_path': str(npz_path),
            'ate_rmse': float(data.get('ate_rmse', -1)),
            'scale': float(data.get('scale', 1.0))
        }
        
        # 加载估计轨迹
        if 'traj_est_poses' not in data:
            raise ValueError(f"文件中缺少 'traj_est_poses': {npz_path}")
        
        traj_est = PoseTrajectory3D(
            poses_se3=data['traj_est_poses'],
            timestamps=data['timestamps']
        )
        
        # 加载参考轨迹（如果存在）
        traj_ref = None
        if 'traj_ref_poses' in data:
            traj_ref = PoseTrajectory3D(
                poses_se3=data['traj_ref_poses'],
                timestamps=data['timestamps']
            )
            print(f"  ✓ 找到GT轨迹")
        
        print(f"  ✓ 轨迹点数: {len(traj_est.poses_se3)}")
        if info['ate_rmse'] > 0:
            print(f"  ✓ ATE RMSE: {info['ate_rmse']:.4f} m")
        
        return traj_est, traj_ref, info
    
    def calculate_ate(self, traj_est, traj_ref):
        """
        计算ATE误差
        
        Args:
            traj_est: 估计轨迹
            traj_ref: 参考轨迹
        Returns:
            rmse: RMSE误差值
            stats: 完整统计信息
        """
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data((traj_ref, traj_est))
        stats = ape_metric.get_all_statistics()
        return stats['rmse'], stats
    
    def plot_3d_comparison(self, traj1_path, traj2_path, output_path, 
                          figsize=(12, 12), dpi=150, 
                          elev=30, azim=45, show_grid=True):
        """
        创建单视角3D轨迹对比图
        
        Args:
            traj1_path: 第一个轨迹文件路径
            traj2_path: 第二个轨迹文件路径
            output_path: 输出图片路径
            figsize: 图片尺寸
            dpi: 分辨率
            elev: 仰角
            azim: 方位角
            show_grid: 是否显示网格
        """
        # 加载轨迹
        traj1_est, traj1_ref, info1 = self.load_trajectory(traj1_path)
        traj2_est, traj2_ref, info2 = self.load_trajectory(traj2_path)
        
        # 确定GT轨迹（假设两个文件的GT相同）
        gt_traj = traj1_ref if traj1_ref is not None else traj2_ref
        if gt_traj is None:
            print("警告: 未找到GT轨迹")
        
        # 创建图形 - 单个大图
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置标题
        title = 'Trajectory Comparison'
        if info1['ate_rmse'] > 0 and info2['ate_rmse'] > 0:
            title += f'\n{info1["filename"]}: RMSE={info1["ate_rmse"]:.4f}m | {info2["filename"]}: RMSE={info2["ate_rmse"]:.4f}m'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 提取轨迹坐标
        traj1_positions = np.array([pose[:3, 3] for pose in traj1_est.poses_se3])
        traj2_positions = np.array([pose[:3, 3] for pose in traj2_est.poses_se3])
        
        # 绘制GT轨迹（深色）
        if gt_traj is not None:
            gt_positions = np.array([pose[:3, 3] for pose in gt_traj.poses_se3])
            ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
                   color='#2c3e50',  # 深灰蓝色
                   linestyle='--', 
                   label='Ground Truth', 
                   linewidth=4.5, 
                   alpha=0.9)
        
        # 绘制第一个轨迹（淡蓝色）
        label1 = f"{info1['filename']}"
        ax.plot(traj1_positions[:, 0], traj1_positions[:, 1], traj1_positions[:, 2],
               color='#74b9ff',  # 淡蓝色
               linestyle='-', 
               label=label1, 
               linewidth=4.0, 
               alpha=0.7)
        
        # 绘制第二个轨迹（淡红色）
        label2 = f"{info2['filename']}"
        ax.plot(traj2_positions[:, 0], traj2_positions[:, 1], traj2_positions[:, 2],
               color='#ff9999',  # 淡红色
               linestyle='-', 
               label=label2, 
               linewidth=4.0, 
               alpha=0.7)
        
        # 设置视角
        ax.view_init(elev=elev, azim=azim)
        
        # 设置轴标签
        ax.set_xlabel('X [m]', fontsize=48, labelpad=30)
        ax.set_ylabel('Y [m]', fontsize=48, labelpad=30)
        ax.set_zlabel('Z [m]', fontsize=48, labelpad=30)
        
        # 设置图例 - 字体大小改为120
        ax.legend(loc='upper right', fontsize=36, framealpha=0.9)
        
        # 强制不显示网格
        ax.grid(False)
        
        # 设置背景色
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        
        # 设置相等的轴比例
        self._set_axes_equal(ax, gt_positions if gt_traj is not None else traj1_positions)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # 关闭图形，释放内存
        
        print(f"\n✓ 3D对比图已保存至: {output_path}")
        print(f"  图片尺寸: {figsize}, DPI: {dpi}")
    
    def plot_3d_with_error(self, traj1_path, traj2_path, output_path, 
                          figsize=(12, 12), dpi=150):
        """
        创建带误差可视化的3D轨迹对比图
        
        Args:
            traj1_path: 第一个轨迹文件路径
            traj2_path: 第二个轨迹文件路径  
            output_path: 输出图片路径
            figsize: 图片尺寸
            dpi: 分辨率
        """
        # 加载轨迹
        traj1_est, traj1_ref, info1 = self.load_trajectory(traj1_path)
        traj2_est, traj2_ref, info2 = self.load_trajectory(traj2_path)
        
        # 确定GT轨迹
        gt_traj = traj1_ref if traj1_ref is not None else traj2_ref
        if gt_traj is None:
            print("错误: 未找到GT轨迹，无法计算误差")
            return
        
        # 计算误差
        ape_metric1 = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric1.process_data((gt_traj, traj1_est))
        errors1 = ape_metric1.error
        
        ape_metric2 = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric2.process_data((gt_traj, traj2_est))
        errors2 = ape_metric2.error
        
        # 创建图形
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置标题
        title = 'Trajectory Comparison with Error Visualization'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 提取轨迹坐标
        gt_positions = np.array([pose[:3, 3] for pose in gt_traj.poses_se3])
        traj1_positions = np.array([pose[:3, 3] for pose in traj1_est.poses_se3])
        traj2_positions = np.array([pose[:3, 3] for pose in traj2_est.poses_se3])
        
        # 绘制GT轨迹（深色）
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
                color='#2c3e50', linestyle='--', label='Ground Truth', 
                linewidth=4.5, alpha=0.9)
        
        # 使用颜色映射显示误差
        self._plot_colored_trajectory(ax, traj1_positions, errors1, 
                                     label=f"{info1['filename']} (RMSE: {info1['ate_rmse']:.3f}m)", 
                                     cmap='Blues_r')  # 使用反向色图，误差越大颜色越深
        self._plot_colored_trajectory(ax, traj2_positions, errors2,
                                     label=f"{info2['filename']} (RMSE: {info2['ate_rmse']:.3f}m)", 
                                     cmap='Reds_r')  # 使用反向色图，误差越大颜色越深
        
        ax.set_xlabel('X [m]', fontsize=48, labelpad=30)
        ax.set_ylabel('Y [m]', fontsize=48, labelpad=30)
        ax.set_zlabel('Z [m]', fontsize=48, labelpad=30)
        ax.legend(loc='upper right', fontsize=36, framealpha=0.9)
        ax.grid(False)
        
        # 设置背景色
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        self._set_axes_equal(ax, gt_positions)
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"\n✓ 带误差的3D对比图已保存至: {output_path}")
    
    def _plot_colored_trajectory(self, ax, positions, errors, label, cmap='viridis'):
        """根据误差值给轨迹着色"""
        from matplotlib import cm
        from matplotlib.colors import Normalize
        
        norm = Normalize(vmin=errors.min(), vmax=errors.max())
        cmap = cm.get_cmap(cmap)
        
        # 分段绘制轨迹，每段使用不同颜色
        for i in range(len(positions) - 1):
            color = cmap(norm(errors[i]))
            ax.plot(positions[i:i+2, 0], 
                   positions[i:i+2, 1], 
                   positions[i:i+2, 2],
                   color=color, linewidth=4.0, alpha=0.8)
        
        # 添加一个虚拟线条用于图例
        ax.plot([], [], '-', color=cmap(0.5), label=label, linewidth=4)
    
    def _set_axes_equal(self, ax, positions):
        """设置3D轴的等比例"""
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 5.0
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def generate_report(self, traj1_path, traj2_path, output_dir):
        """
        生成完整的对比报告
        
        Args:
            traj1_path: 第一个轨迹文件路径
            traj2_path: 第二个轨迹文件路径
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("生成轨迹对比报告")
        print("="*60)
        
        # 生成3D对比图
        output_3d = output_dir / "comparison_3d.png"
        self.plot_3d_comparison(traj1_path, traj2_path, output_3d, figsize=(12, 12))
        
        # 生成带误差的3D对比图
        output_error = output_dir / "comparison_3d_with_error.png"
        self.plot_3d_with_error(traj1_path, traj2_path, output_error, figsize=(12, 12))
        
        # 生成文本报告
        self._generate_text_report(traj1_path, traj2_path, output_dir / "report.txt")
        
        print("\n" + "="*60)
        print(f"报告已生成至: {output_dir}")
        print("="*60)
    
    def _generate_text_report(self, traj1_path, traj2_path, output_path):
        """生成文本格式的对比报告"""
        traj1_est, traj1_ref, info1 = self.load_trajectory(traj1_path)
        traj2_est, traj2_ref, info2 = self.load_trajectory(traj2_path)
        
        gt_traj = traj1_ref if traj1_ref is not None else traj2_ref
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TRAJECTORY COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Trajectory 1: {info1['full_path']}\n")
            f.write(f"Trajectory 2: {info2['full_path']}\n\n")
            
            if gt_traj is not None:
                # 计算统计信息
                _, stats1 = self.calculate_ate(traj1_est, gt_traj)
                _, stats2 = self.calculate_ate(traj2_est, gt_traj)
                
                f.write("TRAJECTORY 1 STATISTICS:\n")
                f.write("-"*30 + "\n")
                for key, value in stats1.items():
                    f.write(f"  {key:10s}: {value:.4f} m\n")
                
                f.write("\nTRAJECTORY 2 STATISTICS:\n")
                f.write("-"*30 + "\n")
                for key, value in stats2.items():
                    f.write(f"  {key:10s}: {value:.4f} m\n")
                
                f.write("\nCOMPARISON:\n")
                f.write("-"*30 + "\n")
                rmse_diff = stats1['rmse'] - stats2['rmse']
                if rmse_diff < 0:
                    f.write(f"Trajectory 1 is better by {abs(rmse_diff):.4f} m (RMSE)\n")
                else:
                    f.write(f"Trajectory 2 is better by {abs(rmse_diff):.4f} m (RMSE)\n")
        
        print(f"✓ 文本报告已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='3D轨迹对比工具 - 适用于集群环境',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  基础3D对比:
    python compare_trajectories_3d.py traj1.npz traj2.npz -o output.png
  
  自定义视角:
    python compare_trajectories_3d.py traj1.npz traj2.npz -o output.png --elev 30 --azim 45
  
  高分辨率输出:
    python compare_trajectories_3d.py traj1.npz traj2.npz -o output.png --dpi 300
  
  大尺寸输出:
    python compare_trajectories_3d.py traj1.npz traj2.npz -o output.png --size 1200
  
  生成完整报告:
    python compare_trajectories_3d.py traj1.npz traj2.npz --report-dir ./report
        """
    )
    
    parser.add_argument('traj1', type=str, help='第一个轨迹NPZ文件路径')
    parser.add_argument('traj2', type=str, help='第二个轨迹NPZ文件路径')
    parser.add_argument('-o', '--output', type=str, default='trajectory_comparison_3d.png',
                       help='输出图片路径 (默认: trajectory_comparison_3d.png)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='输出图片DPI (默认: 150)')
    parser.add_argument('--size', type=int, default=1200,
                       help='图片尺寸（正方形边长，单位：像素，默认: 1200）')
    parser.add_argument('--elev', type=float, default=30,
                       help='3D视角仰角 (默认: 30)')
    parser.add_argument('--azim', type=float, default=45,
                       help='3D视角方位角 (默认: 45)')
    parser.add_argument('--with-error', action='store_true',
                       help='生成带误差可视化的对比图')
    parser.add_argument('--report-dir', type=str,
                       help='生成完整报告的目录')
    parser.add_argument('--no-grid', action='store_true',
                       help='不显示网格')
    
    args = parser.parse_args()
    
    # 计算图片尺寸（以英寸为单位）
    # matplotlib中figsize是以英寸为单位的，需要从像素转换
    # 假设标准DPI为100，则尺寸 = 像素 / 100
    figsize_inches = args.size / 100.0
    figsize = (figsize_inches, figsize_inches)
    
    # 创建对比器
    comparator = TrajectoryComparator()
    
    try:
        # 生成完整报告
        if args.report_dir:
            comparator.generate_report(args.traj1, args.traj2, args.report_dir)
        else:
            # 生成3D对比图
            comparator.plot_3d_comparison(
                args.traj1, args.traj2, args.output,
                figsize=figsize, dpi=args.dpi,
                elev=args.elev, azim=args.azim,
                show_grid=not args.no_grid
            )
            
            # 生成带误差的对比图（如果指定）
            if True:
                error_output = args.output.replace('.png', '_with_error.png')
                comparator.plot_3d_with_error(
                    args.traj1, args.traj2, error_output,
                    figsize=figsize, dpi=args.dpi
                )
                
        print("\n完成！")
        
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()