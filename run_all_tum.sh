#!/bin/bash
#SBATCH --job-name=tum_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=48:00:00

# 创建日志目录
mkdir -p logs

# 配置文件数组
configs=(
    "configs/TUM_RGBD/freiburg1_desk.yaml"
    "configs/TUM_RGBD/freiburg2_xyz.yaml"
    "configs/TUM_RGBD/freiburg3_office.yaml"
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