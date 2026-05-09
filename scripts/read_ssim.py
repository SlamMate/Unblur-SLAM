#!/usr/bin/env python3
import json
from pathlib import Path

# 基础路径
base_path = "./just_nn"

# 序列列表
sequences = [
    "blurball",
    "blurbasket",
    "blurbuick",
    "blurcoffee",
    "blurdecoration",
    "blurgirl",
    "blurheron",
    "blurparterre",
    "blurpuppet",
    "blurstair"
]

# 存储结果
ssim_values = []
lpips_values = []

print("序列结果:")
print("-" * 80)

for seq in sequences:
    json_path = Path(base_path) / seq / "psnr" / "after_refine" / "final_result.json"
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        ssim = data.get("mean_ssim_sharp_frames_only")
        lpips = data.get("mean_lpips_sharp_frames_only")
        
        if ssim is not None and lpips is not None:
            ssim_values.append(ssim)
            lpips_values.append(lpips)
            print(f"{seq:20s} | SSIM: {ssim:.6f} | LPIPS: {lpips:.6f}")
        else:
            print(f"{seq:20s} | 缺失字段")
    
    except FileNotFoundError:
        print(f"{seq:20s} | 文件不存在")
    except json.JSONDecodeError:
        print(f"{seq:20s} | JSON解析错误")

print("-" * 80)

# 计算平均值
if ssim_values and lpips_values:
    mean_ssim = sum(ssim_values) / len(ssim_values)
    mean_lpips = sum(lpips_values) / len(lpips_values)
    
    print(f"\n{'平均值':20s} | SSIM: {mean_ssim:.6f} | LPIPS: {mean_lpips:.6f}")
    print(f"有效序列数: {len(ssim_values)}/{len(sequences)}")
else:
    print("\n没有找到有效数据")