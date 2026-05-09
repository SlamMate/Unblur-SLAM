import numpy as np
import cv2
from typing import Union, Tuple
import matplotlib.pyplot as plt
import torch

class TUMRGBDInverseProcessor:
    """TUM RGBD数据集RGB图像逆处理器"""
    
    def __init__(self, gamma: float = 2.2):
        """
        初始化处理器
        
        Args:
            gamma: gamma值，默认2.2（标准sRGB）
        """
        self.gamma = gamma
        self.create_lut()
    
    def create_lut(self):
        """创建查找表以加速处理"""
        # 创建8-bit到线性的查找表
        self.inverse_gamma_lut = np.zeros(256, dtype=np.float32)
        
        for i in range(256):
            normalized = i / 255.0
            
            # sRGB标准分段函数
            if normalized <= 0.04045:
                linear = normalized / 12.92
            else:
                linear = np.power((normalized + 0.055) / 1.055, 2.4)
            
            self.inverse_gamma_lut[i] = linear
    
    def linearize_srgb(self, image: np.ndarray) -> np.ndarray:
        """
        将sRGB图像转换为线性RGB（使用与encode_srgb相同的标准方法）
        
        Args:
            image: 输入图像 (H, W, 3) uint8格式 [0, 255]
            
        Returns:
            线性RGB图像 (H, W, 3) float32格式，范围[0, 1]
        """
        # 归一化到[0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # 使用精确的sRGB解码公式（逆向encode_srgb的过程）
        linear_image = np.zeros_like(normalized)
        
        # 使用向量化操作
        # sRGB解码的分段函数
        mask = normalized <= 0.04045
        linear_image[mask] = normalized[mask] / 12.92
        linear_image[~mask] = np.power((normalized[~mask] + 0.055) / 1.055, 2.4)
        
        # 确保输出在有效范围内
        linear_image = np.clip(linear_image, 0, 1)
        
        return linear_image
    
    def linearize_simple(self, image: np.ndarray) -> np.ndarray:
        """
        简单幂函数逆gamma校正
        
        Args:
            image: 输入图像
            
        Returns:
            线性化图像
        """
        # 归一化到[0, 1]
        if image.dtype == np.uint8:
            normalized = image.astype(np.float32) / 255.0
        else:
            normalized = image.astype(np.float32)
        
        # 应用逆gamma
        linear = np.power(normalized, self.gamma)
        return linear

class CompleteInverseProcessor:
    """完整的相机处理逆转管道"""
    
    def __init__(self):
        self.gamma_processor = TUMRGBDInverseProcessor()
        
    def process_tum_image(self,
        image,
        correct_gamma: bool = True,
        correct_white_balance: bool = False,
        output_format: str = 'float32') -> torch.Tensor:  # 修改返回类型注释
        """
        处理TUM RGBD数据集的RGB图像
        Args:
            image_path: 图像路径
            correct_gamma: 是否进行gamma校正
            correct_white_balance: 是否校正白平衡
            output_format: 输出格式 ('float32', 'float16', 'uint16')
        Returns:
            处理后的线性RGB图像 (GPU tensor)
        """
        image = image.cpu().numpy()

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # 检查并转换数据类型
        if image.dtype == np.float32 or image.dtype == np.float64:
            # 如果是浮点数，假设范围是 [0, 1]
            if image.max() > 1.0:
                # 如果最大值大于1，可能是 [0, 255] 范围的浮点数
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                # 转换 [0, 1] 范围到 [0, 255] uint8
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
                # 如果不是uint8，尝试转换
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        
        # Step 1: 逆Gamma校正
        if correct_gamma:
            linear_image = self.gamma_processor.linearize_srgb(image)
        else:
            linear_image = image.astype(np.float32) / 255.0
        
        # Step 2: 可选的白平衡校正
        if correct_white_balance:
            linear_image = self.correct_white_balance(linear_image)
        
        # 转换输出格式并转为GPU tensor
        if output_format == 'float16':
            linear_image = linear_image.astype(np.float16)
            return torch.from_numpy(linear_image).cuda()
        elif output_format == 'uint16':
            # 转换为16-bit整数（更高精度）
            linear_image = (linear_image * 65535).astype(np.uint16)
            return torch.from_numpy(linear_image).cuda()
        else:  # float32
            return torch.from_numpy(linear_image).cuda()
        
    def correct_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        简单的灰世界白平衡校正
        
        Args:
            image: 线性RGB图像
            
        Returns:
            白平衡校正后的图像
        """
        # 计算每个通道的平均值
        avg_r = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_b = np.mean(image[:, :, 2])
        
        # 计算灰度平均
        avg_gray = (avg_r + avg_g + avg_b) / 3.0
        
        # 计算缩放因子
        scale_r = avg_gray / (avg_r + 1e-10)
        scale_g = avg_gray / (avg_g + 1e-10)
        scale_b = avg_gray / (avg_b + 1e-10)
        
        # 应用缩放
        corrected = image.copy()
        corrected[:, :, 0] *= scale_r
        corrected[:, :, 1] *= scale_g
        corrected[:, :, 2] *= scale_b
        
        # 裁剪到有效范围
        corrected = np.clip(corrected, 0, 1)
        
        return corrected
    # 处理torch tensor的便捷函数
    def encode_srgb(self, linear_image: np.ndarray, output_uint8: bool = False) -> np.ndarray:
        """
        将线性RGB图像转换回sRGB（gamma编码）
        
        Args:
            linear_image: 线性RGB图像，范围[0, 1]，float格式
            output_uint8: 如果为True，输出uint8格式[0,255]；否则输出float[0,1]
            
        Returns:
            sRGB编码的图像
        """
        # 确保在[0, 1]范围内
        linear_image = torch.clamp(linear_image, 0, 1)
        
        # 使用标准sRGB编码
        srgb_tensor = torch.zeros_like(linear_image)
        
        # 低值部分：线性段
        mask = linear_image <= 0.0031308
        srgb_tensor[mask] = 12.92 * linear_image[mask]
        
        # 高值部分：幂函数段
        srgb_tensor[~mask] = 1.055 * torch.pow(linear_image[~mask], 1.0/2.4) - 0.055
        
        return srgb_tensor

def validate_linearization(processor: CompleteInverseProcessor, 
                          image_path: str):
    """
    验证线性化效果
    
    Args:
        processor: 处理器实例
        image_path: 测试图像路径
    """
    # 读取原始图像
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # 处理图像
    linear = processor.process_tum_image(image_path, correct_gamma=True)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('原始Gamma编码图像')
    axes[0, 0].axis('off')
    
    # 线性化图像
    axes[0, 1].imshow(linear)
    axes[0, 1].set_title('线性化RGB图像')
    axes[0, 1].axis('off')
    
    # 差异图
    diff = np.abs(original_rgb/255.0 - linear)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('差异热图')
    axes[0, 2].axis('off')
    
    # 直方图对比
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        axes[1, 0].hist(original_rgb[:, :, i].ravel(), 50, 
                       color=color, alpha=0.5, label=f'{color.upper()}-gamma')
        axes[1, 1].hist((linear[:, :, i] * 255).ravel(), 50,
                       color=color, alpha=0.5, label=f'{color.upper()}-linear')
    
    axes[1, 0].set_title('原始直方图')
    axes[1, 0].legend()
    axes[1, 1].set_title('线性化直方图')
    axes[1, 1].legend()
    
    # Gamma曲线
    x = np.linspace(0, 1, 256)
    y_srgb = np.zeros_like(x)
    for i, val in enumerate(x):
        if val <= 0.04045:
            y_srgb[i] = val / 12.92
        else:
            y_srgb[i] = np.power((val + 0.055) / 1.055, 2.4)
    
    axes[1, 2].plot(x, y_srgb, 'b-', label='sRGB逆变换')
    axes[1, 2].plot(x, np.power(x, 2.2), 'r--', label='简单γ=2.2')
    axes[1, 2].plot(x, x, 'k:', label='线性')
    axes[1, 2].set_xlabel('输入（gamma编码）')
    axes[1, 2].set_ylabel('输出（线性）')
    axes[1, 2].set_title('Gamma校正曲线')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"原始图像范围: [{original_rgb.min()}, {original_rgb.max()}]")
    print(f"线性图像范围: [{linear.min():.4f}, {linear.max():.4f}]")
    print(f"平均差异: {np.mean(diff):.4f}")
    print(f"最大差异: {np.max(diff):.4f}")

import os
from glob import glob
from tqdm import tqdm

def batch_process_tum_dataset(dataset_path: str, 
                             output_path: str,
                             sequence: str = 'fr1/xyz'):
    """
    批量处理TUM RGBD数据集
    
    Args:
        dataset_path: TUM数据集根目录
        output_path: 输出目录
        sequence: 序列名称
    """
    processor = CompleteInverseProcessor()
    
    # 构建路径
    rgb_path = os.path.join(dataset_path, sequence, 'rgb')
    output_rgb_path = os.path.join(output_path, sequence, 'rgb_linear')
    
    # 创建输出目录
    os.makedirs(output_rgb_path, exist_ok=True)
    
    # 获取所有RGB图像
    image_files = sorted(glob(os.path.join(rgb_path, '*.png')))
    
    print(f"处理 {len(image_files)} 张图像...")
    
    for img_path in tqdm(image_files):
        # 处理图像
        linear_img = processor.process_tum_image(
            img_path, 
            correct_gamma=True,
            output_format='float16'  # 使用float16节省空间
        )
        
        # 保存为EXR格式（保留线性值）
        filename = os.path.basename(img_path).replace('.png', '.exr')
        output_file = os.path.join(output_rgb_path, filename)
        
        # 使用OpenCV保存EXR
        cv2.imwrite(output_file, cv2.cvtColor(
            linear_img, cv2.COLOR_RGB2BGR))
    
    print("处理完成！")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    processor = CompleteInverseProcessor()
    
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图片文件（假设png或jpg）
    image_files = sorted(glob(os.path.join(input_folder, '*.png')) + glob(os.path.join(input_folder, '*.jpg')))
    
    print(f"处理 {len(image_files)} 张图像...")
    
    for img_path in tqdm(image_files):
        # 处理图像
        linear_img = processor.process_tum_image(
            img_path, 
            correct_gamma=True,
            correct_white_balance=False,  # 可以根据需要调整
            output_format='uint16'  # 使用uint16以支持16-bit PNG
        )
        
        # 保存为PNG格式（16-bit）
        filename = os.path.basename(img_path).rsplit('.', 1)[0] + '.png'
        output_file = os.path.join(output_folder, filename)
        
        cv2.imwrite(output_file, cv2.cvtColor(linear_img, cv2.COLOR_RGB2BGR))
    
    print("处理完成！")