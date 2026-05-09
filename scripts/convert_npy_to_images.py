#!/usr/bin/env python3
import numpy as np
from PIL import Image
import os
from pathlib import Path
import glob

def convert_npy_to_image(npy_path, output_path):
    """
    将.npy文件转换为图片
    """
    # 加载npy文件
    data = np.load(npy_path)
    
    # 确保数据在0-255范围内
    if data.dtype == np.float32 or data.dtype == np.float64:
        # 如果是浮点数，假设范围是0-1，转换为0-255
        if data.max() <= 1.0:
            data = (data * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    elif data.dtype != np.uint8:
        data = data.astype(np.uint8)
    
    # 转换为PIL图像
    img = Image.fromarray(data)
    
    # 保存图片
    img.save(output_path)
    print(f"已转换: {npy_path} -> {output_path}")

def process_directory(base_dir):
    """
    处理单个目录下的所有_input_image.npy文件
    """
    base_path = Path(base_dir)
    
    # 创建rgb子文件夹
    rgb_dir = base_path / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    print(f"\n处理目录: {base_dir}")
    print(f"RGB输出目录: {rgb_dir}")
    
    # 查找所有匹配的npy文件
    npy_files = sorted(glob.glob(str(base_path / "*_input_image.npy")))
    
    if not npy_files:
        print(f"  警告: 没有找到匹配的文件")
        return
    
    print(f"  找到 {len(npy_files)} 个文件")
    
    # 转换每个文件
    for npy_file in npy_files:
        # 提取文件名（不含扩展名）
        filename = Path(npy_file).stem
        # 移除_input_image后缀，只保留编号
        file_number = filename.replace("_input_image", "")
        
        # 生成输出文件名
        output_filename = f"{file_number}.png"
        output_path = rgb_dir / output_filename
        
        try:
            convert_npy_to_image(npy_file, str(output_path))
        except Exception as e:
            print(f"  错误: 转换 {npy_file} 时出错: {e}")

def main():
    # 定义要处理的目录列表
    directories = [
        "./datasets/I2-SLAM/fr1_desk",
        "./datasets/I2-SLAM/fr2_xyz",
        "./datasets/I2-SLAM/fr3_office"
    ]
    
    print("=" * 60)
    print("开始批量转换 .npy 文件为图片")
    print("=" * 60)
    
    # 处理每个目录
    for directory in directories:
        if os.path.exists(directory):
            process_directory(directory)
        else:
            print(f"\n警告: 目录不存在: {directory}")
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()