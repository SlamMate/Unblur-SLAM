#!/usr/bin/env python3
import numpy as np
import sys
import os

def read_npy_file(file_path):
    """读取并打印numpy文件内容"""
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return
    
    try:
        # 读取npy文件
        data = np.load(file_path)
        
        # 打印文件信息
        print("=" * 50)
        print(f"文件路径: {file_path}")
        print("=" * 50)
        print(f"\n数据类型: {type(data)}")
        print(f"数据shape: {data.shape}")
        print(f"数据dtype: {data.dtype}")
        print(f"\n文件内容:")
        print(data)
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    # 默认文件路径
    default_path = "./datasets/I2-SLAM/fr1_desk/resolution.npy"
    
    # 如果提供了命令行参数，使用参数中的路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_path
    
    read_npy_file(file_path)