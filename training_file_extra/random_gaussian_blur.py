#!/usr/bin/env python3
"""
对指定文件夹中的图片进行随机高斯模糊处理（默认覆盖原图）
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import argparse
import shutil
from datetime import datetime

def apply_random_gaussian_blur(image, min_kernel=0, max_kernel=21):
    """
    对图片应用随机高斯模糊
    
    Args:
        image: 输入图片
        min_kernel: 最小核大小（0表示不模糊）
        max_kernel: 最大核大小（必须是奇数）
    
    Returns:
        处理后的图片
    """
    if min_kernel == 0:
        kernel_size = random.choice([0] + list(range(3, max_kernel + 1, 2)))
    else:
        kernel_size = random.randrange(min_kernel, max_kernel + 1, 2)
    
    if kernel_size == 0:
        return image, 0, 0
    
    sigma = random.uniform(kernel_size / 6, kernel_size / 2)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return blurred, kernel_size, sigma

def backup_folder(folder_path, backup_suffix="_backup"):
    """
    备份整个文件夹
    
    Args:
        folder_path: 要备份的文件夹路径
        backup_suffix: 备份文件夹的后缀
    
    Returns:
        备份文件夹的路径
    """
    folder_path = Path(folder_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = folder_path.parent / f"{folder_path.name}{backup_suffix}_{timestamp}"
    
    print(f"正在备份文件夹到: {backup_path}")
    shutil.copytree(folder_path, backup_path)
    print("备份完成！")
    
    return backup_path

def process_images_in_folder(input_folder, min_kernel=0, max_kernel=21,
                           create_backup=False, save_to_new=False, output_suffix="_blurred"):
    """
    处理文件夹中的所有图片
    
    Args:
        input_folder: 输入文件夹路径
        min_kernel: 最小核大小
        max_kernel: 最大核大小
        create_backup: 是否创建整个文件夹的备份
        save_to_new: 是否保存到新文件夹而不是覆盖
        output_suffix: 新文件夹的后缀（仅在save_to_new=True时使用）
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"错误：文件夹 {input_folder} 不存在")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in input_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"文件夹 {input_folder} 中没有找到图片")
        return
    
    print(f"\n处理文件夹: {input_folder}")
    print(f"找到 {len(image_files)} 张图片")
    
    backup_path = None
    if create_backup and not save_to_new:
        backup_path = backup_folder(input_path)
    
    if save_to_new:
        output_path = input_path.parent / f"{input_path.name}{output_suffix}"
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"将保存到新文件夹: {output_path}")
    else:
        output_path = input_path
        print("将直接覆盖原图片")
    
    blur_info = []
    success_count = 0
    fail_count = 0
    
    for img_file in tqdm(image_files, desc="处理进度"):
        try:
            img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"警告：无法读取图片 {img_file}")
                fail_count += 1
                continue
            
            blurred_img, kernel_size, sigma = apply_random_gaussian_blur(img, min_kernel, max_kernel)
            
            blur_info.append({
                'file': img_file.name,
                'kernel_size': kernel_size,
                'sigma': sigma,
                'status': '已模糊' if kernel_size > 0 else '未处理'
            })
            
            if save_to_new:
                output_file = output_path / img_file.name
            else:
                output_file = img_file
            
            if img_file.suffix.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(str(output_file), blurred_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif img_file.suffix.lower() == '.png':
                cv2.imwrite(str(output_file), blurred_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
                cv2.imwrite(str(output_file), blurred_img)
            
            success_count += 1
            
        except Exception as e:
            print(f"错误：处理图片 {img_file} 时出错: {e}")
            fail_count += 1
            blur_info.append({
                'file': img_file.name,
                'kernel_size': -1,
                'sigma': -1,
                'status': f'错误: {str(e)}'
            })
    
    log_file = output_path / "blur_info.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理文件夹: {input_folder}\n")
        f.write(f"成功处理: {success_count} 张\n")
        f.write(f"处理失败: {fail_count} 张\n")
        if backup_path:
            f.write(f"备份位置: {backup_path}\n")
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("文件名\t核大小\tSigma值\t状态\n")
        f.write("-" * 60 + "\n")
        for info in blur_info:
            if info['kernel_size'] >= 0:
                f.write(f"{info['file']}\t{info['kernel_size']}\t{info['sigma']:.2f}\t{info['status']}\n")
            else:
                f.write(f"{info['file']}\t-\t-\t{info['status']}\n")
    
    print("\n处理完成！")
    print(f"  成功: {success_count} 张")
    print(f"  失败: {fail_count} 张")
    print(f"  模糊信息已保存到: {log_file}")
    if backup_path:
        print(f"  原始文件备份在: {backup_path}")

def main():
    parser = argparse.ArgumentParser(
        description='对图片进行随机高斯模糊处理（默认直接覆盖原图）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 直接覆盖原图（默认行为）
  python blur_images.py
  
  # 覆盖前先备份整个文件夹
  python blur_images.py --backup
  
  # 保存到新文件夹，不覆盖原图
  python blur_images.py --save-to-new
  
  # 自定义模糊参数
  python blur_images.py --min-kernel 3 --max-kernel 15
  
  # 处理指定文件夹
  python blur_images.py --folders /path/to/folder1 /path/to/folder2
        """
    )
    
    parser.add_argument('--folders', nargs='+', type=str,
                       help='要处理的文件夹路径（可以指定多个）')
    parser.add_argument('--min-kernel', type=int, default=0,
                       help='最小核大小（0表示可能不模糊，默认: 0）')
    parser.add_argument('--max-kernel', type=int, default=21,
                       help='最大核大小（必须是奇数，默认: 21）')
    parser.add_argument('--backup', action='store_true',
                       help='处理前备份整个文件夹')
    parser.add_argument('--save-to-new', action='store_true',
                       help='保存到新文件夹而不是覆盖原图')
    parser.add_argument('--output-suffix', type=str, default='_blurred',
                       help='新文件夹的后缀（仅在--save-to-new时使用，默认: _blurred）')
    parser.add_argument('--no-confirm', action='store_true',
                       help='跳过确认提示，直接处理')
    
    args = parser.parse_args()
    
    if args.max_kernel % 2 == 0:
        args.max_kernel += 1
        print(f"注意：最大核大小必须是奇数，已调整为 {args.max_kernel}")
    
    if args.folders:
        folders = args.folders
    else:
        folders = [
            '/var/scratch/qzhang2/Deblur-SLAM/thirdparty/EVSSM/download/GoPro_gaussian_blur/train/input'
        ]
    
    print("=" * 60)
    print("随机高斯模糊处理工具")
    print("=" * 60)
    print(f"处理模式: {'保存到新文件夹' if args.save_to_new else '覆盖原图'}")
    print(f"备份选项: {'创建备份' if args.backup else '不备份'}")
    print(f"核大小范围: {args.min_kernel} - {args.max_kernel}")
    print(f"待处理文件夹数: {len(folders)}")
    print("=" * 60)
    
    if not args.save_to_new and not args.backup and not args.no_confirm:
        print("\n⚠️  警告：将直接覆盖原图片文件，且没有创建备份！")
        print("如果需要保留原图，请使用 --backup 或 --save-to-new 选项")
        confirm = input("\n是否继续？(y/n): ")
        if confirm.lower() != 'y':
            print("已取消操作")
            return
    
    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] 开始处理...")
        process_images_in_folder(
            folder,
            min_kernel=args.min_kernel,
            max_kernel=args.max_kernel,
            create_backup=args.backup,
            save_to_new=args.save_to_new,
            output_suffix=args.output_suffix
        )
    
    print("\n" + "=" * 60)
    print("所有文件夹处理完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
