#!/bin/bash
#SBATCH --job-name=deblur_all
#SBATCH --output=logs/deblur.out
#SBATCH --error=logs/deblur.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# 创建日志目录
mkdir -p logs

# 配置文件数组
configs=(
    "configs/deblur_nerf_motion/blurgirl.yaml"
    "configs/deblur_nerf_motion/blurstair.yaml"
    "configs/deblur_nerf_motion/blurball.yaml"
    "configs/deblur_nerf_motion/blurcoffee.yaml"
    "configs/deblur_nerf_motion/blurbuick.yaml"
    "configs/deblur_nerf_motion/blurdecoration.yaml"
    "configs/deblur_nerf_motion/blurgirl.yaml"
    "configs/deblur_nerf_motion/blurheron.yaml"
    "configs/deblur_nerf_motion/blurparterre.yaml"
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