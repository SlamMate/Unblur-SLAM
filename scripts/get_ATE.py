#!/usr/bin/env python3
"""
提取所有序列的ATE RMSE值并计算均值
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

# 定义基础路径
BASE_PATH = "./history_other/replica_few_mcd"

# 定义所有序列名称
sequence_names = [
    "mcd_hcd_nosync_s1r00",
    "mcd_hcd_nosync_s1r01",
    "mcd_hcd_nosync_s1r02",
    "mcd_hcd_nosync_s1r03",
    "mcd_hcd_nosync_s1r04",
    "mcd_hcd_nosync_s1r05",
    "mcd_hcd_nosync_s1r06",
    "mcd_hcd_nosync_s1r07",
    "mcd_hcd_nosync_s1r08",
    "mcd_hcd_nosync_s1r09",
    "mcd_hcd_nosync_s1r10",
    "mcd_hcd_nosync_s1r11",
    "mcd_hcd_nosync_s1r15",
    "mcd_hcd_nosync_s1r16",
    "mcd_hcd_nosync_s1r25",
    "mcd_hcd_nosync_s1r26",
    "mcd_hcd_nosync_s1r27",
    "mcd_hcd_nosync_s1r28",
    "mcd_hcd_sync_s0r00",
    "mcd_hcd_sync_s0r01",
    "mcd_hcd_sync_s0r02",
    "mcd_hcd_sync_s2r00",
    "mcd_hcd_sync_s2r01",
    "mcd_hcd_sync_s2r02",
    "mcd_hcd_sync_s2r03",
    "mcd_hcd_sync_s2r04",
    "mcd_hcd_sync_s2r05",
    "mcd_hcd_sync_s2r06",
    "mcd_hcd_sync_s2r07",
    "mcd_hcd_sync_s2r08",
    "mcd_hcd_sync_s2r09",
    "mcd_hcd_sync_s2r10",
    "mcd_hcd_sync_s2r11",
    "mcd_hcd_sync_s2r12",
    "mcd_hcd_sync_s2r16",
    "mcd_hcd_sync_s2r22",
    "mcd_hcd_sync_s3r00",
    "mcd_hcd_sync_s3r08",
    "mcd_hcd_sync_s3r09",
    "mcd_hcd_sync_s3r10",
    "mcd_hcd_sync_s3r12",
    "mcd_hcd_sync_s3r13",
    "mcd_hcd_sync_s3r14",
    "mcd_hcd_sync_s3r15",
    "mcd_marvin_sync_s4r00",
    "mcd_marvin_sync_s4r01",
    "mcd_marvin_sync_s4r02",
    "mcd_marvin_sync_s4r03",
    "mcd_marvin_sync_s4r04",
    "mcd_marvin_sync_s4r05",
    "mcd_marvin_sync_s4r06",
    "mcd_marvin_sync_s4r07",
    "mcd_marvin_sync_s4r08",
    "mcd_marvin_sync_s4r09",
    "mcd_marvin_sync_s4r10",
    "mcd_marvin_sync_s4r11",
    "mcd_marvin_sync_s4r12",
]

# 构建完整路径
sequences = [os.path.join(BASE_PATH, name) for name in sequence_names]


def extract_rmse_from_file(file_path: str) -> float:
    """
    从metrics文件中提取RMSE值
    
    Args:
        file_path: 文件路径
    
    Returns:
        RMSE值，如果失败返回None
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # 使用正则表达式提取statistics字典
        # 查找 'rmse': 后面的数字
        rmse_pattern = r"'rmse':\s*([\d.]+)"
        match = re.search(rmse_pattern, content)
        
        if match:
            return float(match.group(1))
        else:
            print(f"警告: 未能在 {file_path} 中找到RMSE值")
            return None
            
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {file_path}")
        return None
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时出错: {e}")
        return None


def main():
    """主函数"""
    
    # 存储所有的RMSE值
    rmse_results = []
    failed_sequences = []
    
    print("=" * 80)
    print("ATE RMSE 提取脚本")
    print("=" * 80)
    print(f"\n总共需要处理 {len(sequences)} 个序列\n")
    
    # 遍历所有序列
    for i, seq_path in enumerate(sequences, 1):
        # 构建metrics文件的完整路径
        metrics_file = os.path.join(seq_path, "traj", "metrics_full_traj.txt")
        
        # 提取序列名称（路径的最后部分）
        seq_name = os.path.basename(seq_path)
        
        # 提取RMSE值
        rmse_value = extract_rmse_from_file(metrics_file)
        
        if rmse_value is not None:
            rmse_results.append({
                'sequence': seq_name,
                'rmse': rmse_value
            })
            print(f"[{i:2d}/{len(sequences)}] {seq_name:<30} RMSE: {rmse_value:.6f}")
        else:
            failed_sequences.append(seq_name)
            print(f"[{i:2d}/{len(sequences)}] {seq_name:<30} 失败")
    
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    if rmse_results:
        # 计算统计信息
        rmse_values = [r['rmse'] for r in rmse_results]
        mean_rmse = sum(rmse_values) / len(rmse_values)
        min_rmse = min(rmse_values)
        max_rmse = max(rmse_values)
        
        # 计算标准差
        import math
        variance = sum((x - mean_rmse) ** 2 for x in rmse_values) / len(rmse_values)
        std_rmse = math.sqrt(variance)
        
        print(f"\n成功处理的序列数: {len(rmse_results)}")
        print(f"失败的序列数: {len(failed_sequences)}")
        print(f"\nATE RMSE 统计:")
        print(f"  均值 (Mean):     {mean_rmse:.6f}")
        print(f"  标准差 (Std):    {std_rmse:.6f}")
        print(f"  最小值 (Min):    {min_rmse:.6f}")
        print(f"  最大值 (Max):    {max_rmse:.6f}")
        
        # 保存结果到文件
        output_file = "ate_rmse_results.txt"
        with open(output_file, 'w') as f:
            f.write("ATE RMSE 结果报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"处理时间: {__import__('datetime').datetime.now()}\n")
            f.write(f"总序列数: {len(sequences)}\n")
            f.write(f"成功处理: {len(rmse_results)}\n")
            f.write(f"处理失败: {len(failed_sequences)}\n")
            f.write("\n统计信息:\n")
            f.write(f"  均值 (Mean):     {mean_rmse:.6f}\n")
            f.write(f"  标准差 (Std):    {std_rmse:.6f}\n")
            f.write(f"  最小值 (Min):    {min_rmse:.6f}\n")
            f.write(f"  最大值 (Max):    {max_rmse:.6f}\n")
            f.write("\n详细结果:\n")
            f.write("-" * 50 + "\n")
            
            # 按RMSE值排序
            sorted_results = sorted(rmse_results, key=lambda x: x['rmse'])
            for i, result in enumerate(sorted_results, 1):
                f.write(f"{i:3d}. {result['sequence']:<30} RMSE: {result['rmse']:.6f}\n")
            
            if failed_sequences:
                f.write("\n失败的序列:\n")
                f.write("-" * 50 + "\n")
                for seq in failed_sequences:
                    f.write(f"  - {seq}\n")
        
        print(f"\n结果已保存到: {output_file}")
        
        # 同时保存为CSV格式，方便后续处理
        csv_file = "ate_rmse_results.csv"
        with open(csv_file, 'w') as f:
            f.write("Sequence,RMSE\n")
            for result in sorted_results:
                f.write(f"{result['sequence']},{result['rmse']:.6f}\n")
            # 添加统计行
            f.write(f"\nMean,{mean_rmse:.6f}\n")
            f.write(f"Std,{std_rmse:.6f}\n")
            f.write(f"Min,{min_rmse:.6f}\n")
            f.write(f"Max,{max_rmse:.6f}\n")
        
        print(f"CSV结果已保存到: {csv_file}")
        
    else:
        print("\n错误: 没有成功提取任何RMSE值")
    
    if failed_sequences:
        print(f"\n警告: {len(failed_sequences)} 个序列处理失败:")
        for seq in failed_sequences:
            print(f"  - {seq}")


if __name__ == "__main__":
    main()