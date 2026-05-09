#!/bin/bash
#SBATCH --job-name=deblur_all_no_deblur_no_refine
#SBATCH --output=logs/deblur_no_deblur_no_refine.out
#SBATCH --error=logs/deblur_no_deblur_no_refine.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# 创建日志目录
mkdir -p logs

# 配置文件数组
configs=(
    "configs/deblur_nerf_motion_no_deblur_no_refine/blurgirl.yaml"
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