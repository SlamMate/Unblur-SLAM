import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.ndimage import zoom

def visualize_kernel_weights(kernel_weights, mask, kernel_size, num_kernels=3, save_path=None):
    """
    可视化kernel weights，展示权重最大的kernel位置
    
    Args:
        kernel_weights: [1, 1, k^2, H, W] kernel权重
        mask: [1, H, W] or [H, W] blur mask
        kernel_size: int, kernel的尺寸 (k x k)
        num_kernels: int, 要可视化的kernel数量
        save_path: str, 保存路径
    """
    # 确保输入在CPU上并转换为numpy
    if torch.is_tensor(kernel_weights):
        kernel_weights = kernel_weights.detach().cpu()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu()
    
    # 获取维度
    if len(kernel_weights.shape) == 5:
        kernel_weights = kernel_weights[0, 0]  # [k^2, H, W]
    H, W = kernel_weights.shape[-2:]
    
    # 如果mask维度不对，调整它
    if len(mask.shape) == 3:
        mask = mask[0]  # [H, W]
    elif len(mask.shape) == 1:
        mask = mask.reshape(H, W)
    
    # 转换为numpy
    kernel_weights_np = kernel_weights.numpy()
    mask_np = mask.numpy()
    
    # 找到每个位置的最大kernel权重
    max_weights = np.max(kernel_weights_np, axis=0)  # [H, W]
    
    # ===== 修改部分开始 =====
    # 找到中位数附近的kernel位置，且空间分布均匀
    median_weight = np.median(max_weights[max_weights > 0])  # 忽略零值计算中位数
    
    # 找到接近中位数的所有位置
    tolerance = np.std(max_weights[max_weights > 0]) * 0.3  # 容差范围为标准差的30%
    median_mask = np.abs(max_weights - median_weight) < tolerance
    median_positions = np.where(median_mask)
    median_positions = list(zip(median_positions[0], median_positions[1]))
    
    # 如果中位数附近的点太少，扩大容差
    while len(median_positions) < num_kernels * 3:
        tolerance *= 1.5
        median_mask = np.abs(max_weights - median_weight) < tolerance
        median_positions = np.where(median_mask)
        median_positions = list(zip(median_positions[0], median_positions[1]))
    
    # 使用贪心算法选择空间分布均匀的点
    selected_positions = []
    if len(median_positions) > 0:
        # 选择第一个点（随机或最接近中位数的）
        first_idx = np.argmin([abs(max_weights[y, x] - median_weight) 
                               for y, x in median_positions])
        selected_positions.append(median_positions[first_idx])
        
        # 选择剩余的点，每次选择离已选点最远的
        while len(selected_positions) < num_kernels and len(selected_positions) < len(median_positions):
            max_min_dist = -1
            best_candidate = None
            
            for y, x in median_positions:
                if (y, x) not in selected_positions:
                    # 计算到所有已选点的最小距离
                    min_dist = min([np.sqrt((y-sy)**2 + (x-sx)**2) 
                                   for sy, sx in selected_positions])
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_candidate = (y, x)
            
            if best_candidate:
                selected_positions.append(best_candidate)
            else:
                break
    
    top_positions = selected_positions
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)
    
    # 显示mask作为背景
    im = ax.imshow(mask_np, cmap='gray', alpha=0.7)
    ax.set_title(f'Kernel Weights Visualization (kernel_size={kernel_size}x{kernel_size})')
    
    # 为每个top kernel创建可视化
    colors = ['blue']
    scale_factor = 6  # kernel放大倍数
    scaled_kernel_size = kernel_size * scale_factor  # 放大后的kernel尺寸
    kernel_offset = scaled_kernel_size + 10  # 增大偏移量，避免重叠
    
    for idx, (y, x) in enumerate(top_positions):
        color = colors[idx % len(colors)]
        
        # 获取该位置的kernel权重
        kernel = kernel_weights_np[:, y, x].reshape(kernel_size, kernel_size)
        
        # 归一化kernel用于显示
        kernel_norm = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
        
        # 使用插值放大kernel
        kernel_scaled = zoom(kernel_norm, scale_factor, order=1)  # 双线性插值
        
        # 计算kernel块的放置位置（使用角度分布更均匀）
        angle = idx * 2 * np.pi / max(num_kernels, 3)  # 均匀分布角度
        kernel_x = x + kernel_offset * np.cos(angle - np.pi/2)
        kernel_y = y + kernel_offset * np.sin(angle - np.pi/2)
        
        # 确保kernel块在图像边界内
        kernel_x = np.clip(kernel_x, 0, W - scaled_kernel_size)
        kernel_y = np.clip(kernel_y, 0, H - scaled_kernel_size)
        
        # 绘制放大后的kernel块
        for ky in range(scaled_kernel_size):
            for kx in range(scaled_kernel_size):
                val = kernel_scaled[ky, kx]
                val_enhanced = np.power(val, 2)  # 平方变换
                rect = patches.Rectangle(
                    (kernel_x + kx, kernel_y + ky),
                    1, 1,
                    linewidth=0.2,
                    edgecolor=color,
                    facecolor=plt.cm.plasma(val_enhanced),
                    alpha=0.8
                )
                ax.add_patch(rect)
        
        # 添加kernel块的边框
        kernel_rect = patches.Rectangle(
            (kernel_x, kernel_y),
            scaled_kernel_size, scaled_kernel_size,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(kernel_rect)
        
        corners = [
            (kernel_x, kernel_y),  # 左上
            (kernel_x + scaled_kernel_size, kernel_y),  # 右上
            (kernel_x, kernel_y + scaled_kernel_size),  # 左下
            (kernel_x + scaled_kernel_size, kernel_y + scaled_kernel_size)  # 右下
        ]

        # 计算每个顶点到像素位置的距离
        pixel_x, pixel_y = x + 0.5, y + 0.5
        distances = [(np.sqrt((cx - pixel_x)**2 + (cy - pixel_y)**2), (cx, cy)) 
                    for cx, cy in corners]
        distances.sort(key=lambda x: x[0])

        # 连接到两个最近的顶点
        for i in range(2):
            _, (corner_x, corner_y) = distances[i]
            line = Line2D(
                [pixel_x, corner_x],
                [pixel_y, corner_y],
                color=color,
                linewidth=2,
                alpha=0.7,
                linestyle='--'  # 虚线
            )
            ax.add_line(line)
        
        # 在像素位置添加标记
        circle = patches.Circle(
            (x + 0.5, y + 0.5),
            radius=1,
            edgecolor=color,
            facecolor=color,
            alpha=0.8
        )
        ax.add_patch(circle)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('Mask Value', rotation=270, labelpad=15)
    
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close(fig)
    return fig, ax


def visualize_kernel_weights_advanced(kernel_weights, mask, kernel_size, image=None, num_kernels=3, save_path=None):
    """
    高级可视化：在原始图像上叠加kernel weights可视化
    
    Args:
        kernel_weights: [1, 1, k^2, H, W] kernel权重
        mask: [1, H, W] or [H, W] blur mask  
        kernel_size: int, kernel的尺寸 (k x k)
        image: [3, H, W] or [H, W, 3] 原始图像（可选）
        num_kernels: int, 要可视化的kernel数量
        save_path: str, 保存路径
    """
    # 确保输入在CPU上并转换为numpy
    if torch.is_tensor(kernel_weights):
        kernel_weights = kernel_weights.detach().cpu()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu()
    if image is not None and torch.is_tensor(image):
        image = image.detach().cpu()
    
    # 获取维度
    if len(kernel_weights.shape) == 5:
        kernel_weights = kernel_weights[0, 0]  # [k^2, H, W]
    H, W = kernel_weights.shape[-2:]
    
    # 处理mask维度
    if len(mask.shape) == 3:
        mask = mask[0]
    elif len(mask.shape) == 1:
        mask = mask.reshape(H, W)
    
    # 处理image维度
    if image is not None:
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)  # [H, W, 3]
        image = image.numpy()
        # 归一化到[0, 1]
        if image.max() > 1:
            image = image / image.max()
    
    # 转换为numpy
    kernel_weights_np = kernel_weights.numpy()
    mask_np = mask.numpy()
    
    # 找到每个位置的最大kernel权重和对应的kernel索引
    max_weights = np.max(kernel_weights_np, axis=0)  # [H, W]
    max_kernel_idx = np.argmax(kernel_weights_np, axis=0)  # [H, W]
    
    # 找到前num_kernels个最大权重的位置
    flat_indices = np.argsort(max_weights.flatten())[-num_kernels:][::-1]
    top_positions = [(idx // W, idx % W) for idx in flat_indices]
    
    # 创建图形
    fig = plt.figure(figsize=(16, 8))
    
    # 子图1：mask和kernel可视化
    ax1 = plt.subplot(1, 2, 1)
    if image is not None:
        ax1.imshow(image, alpha=0.5)
        ax1.imshow(mask_np, cmap='jet', alpha=0.3)
    else:
        ax1.imshow(mask_np, cmap='gray', alpha=0.7)
    ax1.set_title(f'Mask with Top {num_kernels} Kernel Locations')
    
    # 子图2：kernel权重热图
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(max_weights, cmap='hot')
    ax2.set_title('Max Kernel Weights Heatmap')
    plt.colorbar(im2, ax=ax2)
    
    # 在子图1上添加kernel可视化
    colors = ['red', 'lime', 'cyan']
    scale_factor = 3  # kernel放大倍数
    scaled_kernel_size = kernel_size * scale_factor
    kernel_offset = scaled_kernel_size + 15  # 更大的偏移量
    
    for idx, (y, x) in enumerate(top_positions):
        color = colors[idx % len(colors)]
        
        # 获取该位置的kernel权重
        kernel = kernel_weights_np[:, y, x].reshape(kernel_size, kernel_size)
        
        # 归一化kernel用于显示
        kernel_norm = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
        
        # 使用插值放大kernel
        kernel_scaled = zoom(kernel_norm, scale_factor, order=1)
        
        # 计算kernel块的放置位置（围绕像素位置均匀排列）
        angle = idx * 2 * np.pi / num_kernels + np.pi/4  # 添加初始角度偏移
        kernel_x = x + kernel_offset * np.cos(angle)
        kernel_y = y + kernel_offset * np.sin(angle)
        
        # 确保kernel块在图像边界内
        kernel_x = np.clip(kernel_x, 0, W - scaled_kernel_size)
        kernel_y = np.clip(kernel_y, 0, H - scaled_kernel_size)
        
        # 在两个子图上都标记位置
        for ax in [ax1, ax2]:
            # 绘制kernel块（仅在ax1上）
            if ax == ax1:
                for ky in range(scaled_kernel_size):
                    for kx in range(scaled_kernel_size):
                        rect = patches.Rectangle(
                            (kernel_x + kx, kernel_y + ky),
                            1, 1,
                            linewidth=0.1,
                            edgecolor=color,
                            facecolor=plt.cm.hot(kernel_scaled[ky, kx]),
                            alpha=0.9
                        )
                        ax.add_patch(rect)
                
                # 添加kernel块的边框
                kernel_rect = patches.Rectangle(
                    (kernel_x, kernel_y),
                    scaled_kernel_size, scaled_kernel_size,
                    linewidth=2.5,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(kernel_rect)
                
                # 绘制连线
                kernel_center_x = kernel_x + scaled_kernel_size / 2
                kernel_center_y = kernel_y + scaled_kernel_size / 2
                
                line = Line2D(
                    [x + 0.5, kernel_center_x],
                    [y + 0.5, kernel_center_y],
                    color=color,
                    linewidth=2.5,
                    alpha=0.8,
                    linestyle='--'
                )
                ax.add_line(line)
            
            # 在像素位置添加标记
            circle = patches.Circle(
                (x + 0.5, y + 0.5),
                radius=1.5,
                edgecolor=color,
                facecolor='none' if ax == ax2 else color,
                linewidth=2.5,
                alpha=0.9
            )
            ax.add_patch(circle)
            
            # 添加标签
            if ax == ax1:
                ax.text(
                    kernel_center_x,
                    kernel_y - 2,
                    f'K{idx+1}: {max_weights[y, x]:.3f}',
                    color='white',
                    fontsize=9,
                    ha='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
                )
    
    # 设置坐标轴
    for ax in [ax1, ax2]:
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.grid(True, alpha=0.2)
    
    plt.suptitle(f'Kernel Weights Visualization (size={kernel_size}x{kernel_size})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Advanced visualization saved to {save_path}")
    
    plt.show()
    return fig


# 使用示例
def demo_usage():
    """
    演示如何使用可视化函数
    """
    # 模拟数据
    H, W = 128, 128
    kernel_size = 5
    k_squared = kernel_size * kernel_size
    
    # 创建模拟的kernel_weights
    kernel_weights = torch.randn(1, 1, k_squared, H, W)
    # 添加一些峰值
    kernel_weights[0, 0, :, 30, 40] = torch.randn(k_squared) * 2
    kernel_weights[0, 0, :, 60, 80] = torch.randn(k_squared) * 2.5
    kernel_weights[0, 0, :, 90, 50] = torch.randn(k_squared) * 3
    
    # 创建模拟的mask
    mask = torch.sigmoid(torch.randn(1, H, W) * 2)
    
    # 创建模拟的图像（可选）
    image = torch.rand(3, H, W)
    
    # 基础可视化
    print("Running basic visualization...")
    visualize_kernel_weights(
        kernel_weights, 
        mask, 
        kernel_size, 
        num_kernels=3,
        save_path="kernel_weights_basic.png"
    )
    
    # 高级可视化（带图像背景）
    print("Running advanced visualization...")
    visualize_kernel_weights_advanced(
        kernel_weights,
        mask,
        kernel_size,
        image=image,
        num_kernels=3,
        save_path="kernel_weights_advanced.png"
    )


if __name__ == "__main__":
    demo_usage()