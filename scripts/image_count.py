#!/usr/bin/env python3
"""
图片计数脚本 - 统计文件夹中的图片数量
"""
import os
import sys
from pathlib import Path

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg', '.ico'}

def count_images(folder_path, recursive=False):
    """
    计数文件夹中的图片数量
    
    Args:
        folder_path: 文件夹路径
        recursive: 是否递归搜索子文件夹
    
    Returns:
        图片总数
    """
    if not os.path.exists(folder_path):
        print(f"错误: 路径不存在 - {folder_path}")
        return 0
    
    if not os.path.isdir(folder_path):
        print(f"错误: 不是文件夹 - {folder_path}")
        return 0
    
    image_count = 0
    images_by_type = {}
    
    if recursive:
        # 递归搜索所有子文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    image_count += 1
                    images_by_type[ext] = images_by_type.get(ext, 0) + 1
    else:
        # 只搜索当前文件夹
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                ext = Path(item).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    image_count += 1
                    images_by_type[ext] = images_by_type.get(ext, 0) + 1
    
    # 打印结果
    print(f"\n文件夹: {folder_path}")
    print(f"{'递归搜索' if recursive else '仅当前文件夹'}")
    print(f"\n图片总数: {image_count}")
    
    if images_by_type:
        print("\n按格式分类:")
        for ext, count in sorted(images_by_type.items()):
            print(f"  {ext}: {count}")
    
    return image_count

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法:")
        print(f"  python {sys.argv[0]} <文件夹路径> [--recursive]")
        print("\n选项:")
        print("  --recursive, -r  递归搜索所有子文件夹")
        print("\n示例:")
        print(f"  python {sys.argv[0]} ./images")
        print(f"  python {sys.argv[0]} /path/to/folder --recursive")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    recursive = '--recursive' in sys.argv or '-r' in sys.argv
    
    count_images(folder_path, recursive)

if __name__ == "__main__":
    main()