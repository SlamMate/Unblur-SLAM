#!/usr/bin/env python3
import re
import os
from pathlib import Path

# 定义序列名称
sequences = [
    'blurball',
    'blurbasket',
    'blurbuick',
    'blurcoffee',
    'blurdecoration',
    'blurgirl',
    'blurheron',
    'blurparterre'
]

# 定义两个文件夹路径
folder1 = './output/tracking'
folder2 = './sharp_weight2_5e-5'

def extract_rmse(file_path):
    """从metrics_kf_traj.txt文件中提取rmse值"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # 使用正则表达式匹配 'rmse': 数值
            match = re.search(r"'rmse':\s*([\d.]+)", content)
            if match:
                return float(match.group(1))
            else:
                print(f"Warning: Could not find rmse in {file_path}")
                return None
    except FileNotFoundError:
        print(f"Warning: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# 存储结果
results = []

print("=" * 80)
print("Comparing RMSE values between two folders")
print("=" * 80)
print(f"\nFolder 1: {folder1}")
print(f"Folder 2: {folder2}")
print("\n" + "-" * 80)

for seq in sequences:
    # 构建文件路径
    file1 = os.path.join(folder1, seq, 'traj', 'metrics_kf_traj.txt')
    file2 = os.path.join(folder2, seq, 'traj', 'metrics_kf_traj.txt')
    
    # 提取RMSE值
    rmse1 = extract_rmse(file1)
    rmse2 = extract_rmse(file2)
    
    if rmse1 is not None and rmse2 is not None:
        # 计算差异（绝对值）
        diff = abs(rmse1 - rmse2)
        # 计算相对差异（百分比）
        relative_diff = (abs(rmse1 - rmse2) / rmse1) * 100 if rmse1 != 0 else 0
        
        results.append({
            'sequence': seq,
            'rmse_tracking': rmse1,
            'rmse_sharp': rmse2,
            'abs_diff': diff,
            'relative_diff': relative_diff
        })
        
        print(f"\n{seq}:")
        print(f"  Tracking RMSE:        {rmse1:.6f}")
        print(f"  Sharp_weight RMSE:    {rmse2:.6f}")
        print(f"  Absolute Difference:  {diff:.6f}")
        print(f"  Relative Difference:  {relative_diff:.2f}%")

# 按绝对差异排序
if results:
    print("\n" + "=" * 80)
    print("RANKING BY ABSOLUTE DIFFERENCE")
    print("=" * 80)
    
    sorted_results = sorted(results, key=lambda x: x['abs_diff'], reverse=True)
    
    for i, res in enumerate(sorted_results, 1):
        print(f"\n{i}. {res['sequence']}")
        print(f"   Tracking:     {res['rmse_tracking']:.6f}")
        print(f"   Sharp_weight: {res['rmse_sharp']:.6f}")
        print(f"   Difference:   {res['abs_diff']:.6f} ({res['relative_diff']:.2f}%)")
    
    # 找出差异最大的序列
    max_diff_seq = sorted_results[0]
    print("\n" + "=" * 80)
    print("SEQUENCE WITH MAXIMUM RMSE DIFFERENCE")
    print("=" * 80)
    print(f"\nSequence: {max_diff_seq['sequence']}")
    print(f"Tracking RMSE:        {max_diff_seq['rmse_tracking']:.6f}")
    print(f"Sharp_weight RMSE:    {max_diff_seq['rmse_sharp']:.6f}")
    print(f"Absolute Difference:  {max_diff_seq['abs_diff']:.6f}")
    print(f"Relative Difference:  {max_diff_seq['relative_diff']:.2f}%")
    
    # 按相对差异排序
    print("\n" + "=" * 80)
    print("RANKING BY RELATIVE DIFFERENCE (%)")
    print("=" * 80)
    
    sorted_by_relative = sorted(results, key=lambda x: x['relative_diff'], reverse=True)
    
    for i, res in enumerate(sorted_by_relative, 1):
        print(f"\n{i}. {res['sequence']}")
        print(f"   Tracking:     {res['rmse_tracking']:.6f}")
        print(f"   Sharp_weight: {res['rmse_sharp']:.6f}")
        print(f"   Difference:   {res['abs_diff']:.6f} ({res['relative_diff']:.2f}%)")
    
    print("\n" + "=" * 80)
else:
    print("\nNo valid results found!")