#!/usr/bin/env python3
"""
查找每个时间戳最接近的图片并确定其在文件夹中的位置 - 灵活版本
可以通过配置文件或命令行参数指定路径
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

def get_sorted_image_timestamps(rgb_dir):
    """
    获取RGB文件夹中所有图片的时间戳并排序
    
    Args:
        rgb_dir: RGB图片文件夹路径
        
    Returns:
        sorted_timestamps: 排序后的时间戳数组
        timestamp_to_index: 时间戳到索引的映射字典
        timestamp_to_filename: 时间戳到文件名的映射字典
    """
    print(f"  扫描图片目录: {rgb_dir}")
    
    # 获取所有PNG文件
    rgb_path = Path(rgb_dir)
    if not rgb_path.exists():
        raise FileNotFoundError(f"目录不存在: {rgb_dir}")
    
    png_files = list(rgb_path.glob("*.png"))
    if not png_files:
        raise ValueError(f"目录中没有PNG文件: {rgb_dir}")
    
    # 提取时间戳（去掉.png后缀）
    timestamps = []
    timestamp_to_filename = {}
    
    for png_file in png_files:
        timestamp_str = png_file.stem  # 获取不带扩展名的文件名
        try:
            timestamp = float(timestamp_str)
            timestamps.append(timestamp)
            timestamp_to_filename[timestamp] = png_file.name
        except ValueError:
            print(f"    警告: 无法解析时间戳 {timestamp_str}")
            continue
    
    # 排序时间戳
    sorted_timestamps = np.array(sorted(timestamps))
    
    # 创建时间戳到索引的映射
    timestamp_to_index = {ts: idx for idx, ts in enumerate(sorted_timestamps)}
    
    print(f"  找到 {len(sorted_timestamps)} 张图片")
    print(f"  时间戳范围: {sorted_timestamps[0]:.6f} ~ {sorted_timestamps[-1]:.6f}")
    
    return sorted_timestamps, timestamp_to_index, timestamp_to_filename

def find_closest_timestamp_index(query_timestamp, sorted_timestamps):
    """
    找到最接近查询时间戳的图片索引
    
    Args:
        query_timestamp: 查询的时间戳
        sorted_timestamps: 排序后的时间戳数组
        
    Returns:
        closest_index: 最接近的图片索引
        closest_timestamp: 最接近的时间戳值
        difference: 时间差
    """
    # 使用二分查找找到最接近的时间戳
    idx = np.searchsorted(sorted_timestamps, query_timestamp)
    
    # 检查边界情况
    if idx == 0:
        closest_idx = 0
    elif idx == len(sorted_timestamps):
        closest_idx = len(sorted_timestamps) - 1
    else:
        # 比较左右两个时间戳，选择更接近的
        left_diff = abs(query_timestamp - sorted_timestamps[idx - 1])
        right_diff = abs(query_timestamp - sorted_timestamps[idx])
        closest_idx = idx - 1 if left_diff < right_diff else idx
    
    closest_timestamp = sorted_timestamps[closest_idx]
    difference = abs(query_timestamp - closest_timestamp)
    
    return closest_idx, closest_timestamp, difference

def process_single_file(input_file, rgb_dir, output_file, include_filename=False):
    """
    处理单个文件
    
    Args:
        input_file: 输入文件路径
        rgb_dir: RGB图片目录
        output_file: 输出文件路径
        include_filename: 是否包含文件名
        
    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*70}")
    print(f"处理文件: {os.path.basename(input_file)}")
    print(f"{'='*70}")
    
    # 1. 获取所有图片的时间戳并排序
    try:
        sorted_timestamps, timestamp_to_index, timestamp_to_filename = get_sorted_image_timestamps(rgb_dir)
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False
    
    # 2. 读取输入文件
    print(f"\n  读取输入文件: {input_file}")
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  ❌ 错误: 文件不存在")
        return False
    
    # 跳过注释行
    data_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]
    print(f"  共 {len(data_lines)} 行数据")
    
    # 3. 处理每一行，找到最接近的图片
    results = []
    if include_filename:
        results.append("# frame_id timestamp difference closest_image_index closest_timestamp time_diff closest_filename\n")
    else:
        results.append("# frame_id timestamp difference closest_image_index closest_timestamp time_diff\n")
    
    print(f"\n  查找最接近的图片...")
    max_time_diff = 0.0
    max_time_diff_line = ""
    
    for i, line in enumerate(data_lines):
        parts = line.strip().split()
        if len(parts) < 3:
            print(f"    警告: 第 {i+1} 行格式不正确")
            continue
        
        frame_id = parts[0]
        timestamp = float(parts[1])
        difference = float(parts[2])
        
        # 找到最接近的图片
        closest_idx, closest_ts, time_diff = find_closest_timestamp_index(timestamp, sorted_timestamps)
        
        # 跟踪最大时间差
        if time_diff > max_time_diff:
            max_time_diff = time_diff
            max_time_diff_line = f"{frame_id} {timestamp:.6f}"
        
        # 格式化输出
        if include_filename:
            closest_filename = timestamp_to_filename[closest_ts]
            result_line = f"{frame_id} {timestamp:.6f} {difference:.6e} {closest_idx:04d} {closest_ts:.6f} {time_diff:.6e} {closest_filename}\n"
        else:
            result_line = f"{frame_id} {timestamp:.6f} {difference:.6e} {closest_idx:04d} {closest_ts:.6f} {time_diff:.6e}\n"
        
        results.append(result_line)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(data_lines):
            print(f"    已处理: {i+1}/{len(data_lines)}", end='\r')
    
    print()  # 换行
    
    # 4. 保存结果
    print(f"\n  保存结果到: {output_file}")
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.writelines(results)
        
        print(f"  ✓ 成功保存 {len(results)-1} 行数据")
        
        # 显示统计信息
        time_diffs = []
        for line in results[1:]:  # 跳过头行
            parts = line.strip().split()
            if len(parts) >= 6:
                time_diffs.append(float(parts[5]))
        
        if time_diffs:
            print(f"\n  统计信息:")
            print(f"    平均时间差: {np.mean(time_diffs):.6e} 秒")
            print(f"    最大时间差: {np.max(time_diffs):.6e} 秒 (frame {max_time_diff_line})")
            print(f"    最小时间差: {np.min(time_diffs):.6e} 秒")
            print(f"    中位数时间差: {np.median(time_diffs):.6e} 秒")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 保存失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='查找每个时间戳最接近的图片并确定其索引位置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单个文件
  python3 %(prog)s -i input.txt -r /path/to/rgb -o output.txt
  
  # 处理所有三个数据集（使用默认路径）
  python3 %(prog)s --all
  
  # 包含文件名
  python3 %(prog)s -i input.txt -r /path/to/rgb -o output.txt --filename
        """
    )
    
    parser.add_argument('-i', '--input', help='输入文件路径')
    parser.add_argument('-r', '--rgb-dir', help='RGB图片目录')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--filename', action='store_true', help='在输出中包含图片文件名')
    parser.add_argument('--all', action='store_true', help='处理所有三个数据集（使用默认路径）')
    
    args = parser.parse_args()
    
    print("="*70)
    print("查找最近时间戳图片并确定索引位置")
    print("="*70)
    
    if args.all:
        # 处理所有数据集
        datasets = {
            'fr1_desk': {
                'input': './scripts/matched_timestamps_fr1_desk_sharp.txt',
                'rgb_dir': './datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb',
                'output': './scripts/matched_timestamps_fr1_desk_sharp_with_index.txt'
            },
            'fr2_xyz': {
                'input': './scripts/matched_timestamps_fr2_xyz_sharp.txt',
                'rgb_dir': './datasets/TUM_RGBD/rgbd_dataset_freiburg2_xyz/rgb',
                'output': './scripts/matched_timestamps_fr2_xyz_sharp_with_index.txt'
            },
            'fr3_office': {
                'input': './scripts/matched_timestamps_fr3_office_sharp.txt',
                'rgb_dir': './datasets/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household/rgb',
                'output': './scripts/matched_timestamps_fr3_office_sharp_with_index.txt'
            }
        }
        
        success_count = 0
        for name, config in datasets.items():
            if process_single_file(config['input'], config['rgb_dir'], config['output'], args.filename):
                success_count += 1
        
        print(f"\n{'='*70}")
        print(f"处理完成! 成功: {success_count}/{len(datasets)}")
        print(f"{'='*70}")
        
    elif args.input and args.rgb_dir and args.output:
        # 处理单个文件
        success = process_single_file(args.input, args.rgb_dir, args.output, args.filename)
        
        print(f"\n{'='*70}")
        if success:
            print("处理完成!")
        else:
            print("处理失败!")
        print(f"{'='*70}")
        
    else:
        parser.print_help()
        print("\n错误: 请指定 --all 或者提供 -i, -r, -o 参数")
        sys.exit(1)
    
    print("\n输出文件格式说明:")
    print("  列1: frame_id - 原始帧ID")
    print("  列2: timestamp - 原始时间戳")
    print("  列3: difference - 原始差异值")
    print("  列4: closest_image_index - 最接近图片的索引(0开始)")
    print("  列5: closest_timestamp - 最接近图片的时间戳")
    print("  列6: time_diff - 时间差(秒)")
    if args.filename:
        print("  列7: closest_filename - 最接近图片的文件名")

if __name__ == '__main__':
    main()