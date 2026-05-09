#!/bin/bash
#SBATCH --job-name=deblur_all
#SBATCH --output=logs/deblur_%A_%a.out
#SBATCH --error=logs/deblur_%A_%a.err
#SBATCH --array=0-10
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# 创建日志目录
mkdir -p logs

# 配置文件数组
configs=(
    "configs/deblur_nerf_motion_ori/blurball.yaml"
    "configs/deblur_nerf_motion_ori/blurbasket.yaml"
    "configs/deblur_nerf_motion_ori/blurbuick.yaml"
    "configs/deblur_nerf_motion_ori/blurcoffee.yaml"
    "configs/deblur_nerf_motion_ori/blurdecoration.yaml"
    "configs/deblur_nerf_motion_ori/blurgirl.yaml"
    "configs/deblur_nerf_motion_ori/blurheron.yaml"
    "configs/deblur_nerf_motion_ori/blurparterre.yaml"
)

# 遍历所有配置文件并运行
for config in "${configs[@]}"; do
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="
    
    srun python run.py "$config"
    
    # 检查退出状态
    if [ $? -ne 0 ]; then
        echo "Error: $config failed!"
        # 可选：取消注释下面这行来在出错时停止
        # exit 1
    else
        echo "Success: $config completed!"
    fi
    echo ""
done