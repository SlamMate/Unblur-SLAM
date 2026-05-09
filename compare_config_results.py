#!/usr/bin/env python3
"""
对比 config 测试结果与历史基线 (ate_rmse_results.csv, sharp_weight2_5e-5)
"""
import os
import re
import glob
import csv
from pathlib import Path

BAGS_ROOT = Path(".")
ATE_CSV = BAGS_ROOT / "ate_rmse_results.csv"
ATE_TXT = BAGS_ROOT / "ate_rmse_results.txt"
SHARP_WEIGHT_DIR = BAGS_ROOT / "sharp_weight2_5e-5"
MCD_OUTPUT = BAGS_ROOT / "MCD"
MCD_MAPPING = BAGS_ROOT / "MCD_mapping"

def parse_rmse_from_log(log_path):
    """从日志中解析 rmse (Keyframes traj statistics)"""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        content = f.read()
    # 匹配 statistics: {'rmse': 0.xxx, ...}
    m = re.search(r"'rmse':\s*([\d.]+)", content)
    return float(m.group(1)) if m else None

def parse_rmse_from_cfg(cfg_path):
    """从保存的 cfg.yaml 中查找 (如果有评估结果)"""
    return None  # cfg 通常不存 rmse

def load_baseline_csv():
    """加载 ate_rmse_results.csv 基线"""
    baseline = {}
    if not ATE_CSV.exists():
        return baseline
    with open(ATE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("Sequence", "").strip()
            rmse_str = row.get("RMSE", "").strip()
            if seq and rmse_str and seq not in ("Mean", "Std", "Min", "Max"):
                try:
                    baseline[seq] = float(rmse_str)
                except ValueError:
                    pass
    return baseline

def main():
    baseline = load_baseline_csv()
    print("=" * 60)
    print("Unblur-SLAM config benchmark comparison")
    print("=" * 60)
    print(f"\n基线来源: {ATE_CSV}")
    print(f"基线序列数: {len(baseline)}")
    if baseline:
        print(f"示例: mcd_hcd_nosync_s1r00 -> RMSE {baseline.get('mcd_hcd_nosync_s1r00', 'N/A')}")

    # 检查 MCD 输出
    print("\n--- MCD 输出检查 ---")
    mcd_scenes = ["mcd_hcd_nosync_s1r00", "mcd_hcd_sync_s0r00"]  # 示例
    for scene in mcd_scenes:
        out_dir = MCD_OUTPUT / scene
        mapping_dir = MCD_MAPPING / scene
        cfg_path = out_dir / "cfg.yaml" if out_dir.exists() else (mapping_dir / "cfg.yaml" if mapping_dir.exists() else None)
        traj_path = out_dir / "traj.txt" if out_dir.exists() else (mapping_dir / "traj.txt" if mapping_dir.exists() else None)
        if out_dir.exists() or mapping_dir.exists():
            print(f"  {scene}: 输出目录存在")
            if (out_dir / "cfg.yaml").exists() or (mapping_dir / "cfg.yaml").exists():
                print(f"    - cfg.yaml 存在")
            if (out_dir / "traj.txt").exists() or (mapping_dir / "traj.txt").exists():
                print(f"    - traj.txt 存在")
        else:
            print(f"  {scene}: 输出目录不存在 (尚未运行)")

    # 检查 sharp_weight2_5e-5
    print("\n--- sharp_weight2_5e-5 结果检查 ---")
    if SHARP_WEIGHT_DIR.exists():
        bench_cfg = SHARP_WEIGHT_DIR / "bench" / "cfg.yaml"
        if bench_cfg.exists():
            print(f"  bench/cfg.yaml 存在 (inherit_from: configs/exblurf_motion/exblurf_motion.yaml)")
        scenes = [d.name for d in SHARP_WEIGHT_DIR.iterdir() if d.is_dir()]
        print(f"  场景数: {len(scenes)}")
        for s in scenes[:5]:
            print(f"    - {s}")
        if len(scenes) > 5:
            print(f"    ... 等 {len(scenes)} 个")
    else:
        print("  sharp_weight2_5e-5 目录不存在")

    # 检查 configs 完整性
    print("\n--- Config 完整性检查 ---")
    configs_dir = BAGS_ROOT / "configs"
    required = [
        "unblur_slam.yaml",
        "MCD/mcd.yaml",
        "MCD/mcd_hcd_nosync_s1r00.yaml",
        "exblurf_motion/exblurf_motion.yaml",
        "exblurf_motion/bench.yaml",
    ]
    for r in required:
        p = configs_dir / r
        status = "✓" if p.exists() else "✗"
        print(f"  {status} {r}")

    splat = configs_dir / "unblur_slam.yaml"
    if splat.exists():
        with open(splat) as f:
            c = f.read()
        checks = [
            ("sharp_loss_weight", "sharp_loss_weight" in c),
            ("sharp_loss_weight_value", "sharp_loss_weight_value" in c),
        ]
        print("\n  unblur_slam.yaml additions:")
        for name, ok in checks:
            print(f"    {'✓' if ok else '✗'} {name}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
