#!/bin/bash
#!/bin/bash
#SBATCH --job-name=exblurf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=24:00:00

# 运行所有 exblurf_motion 序列的脚本

# 配置文件数组
configs=(
    "configs/exblurf_motion/bench.yaml"
    "configs/exblurf_motion/camellia.yaml"
    "configs/exblurf_motion/dragon.yaml"
    "configs/exblurf_motion/jars.yaml"
    "configs/exblurf_motion/jars2.yaml"
    "configs/exblurf_motion/postbox.yaml"
    "configs/exblurf_motion/stone_lantern.yaml"
    "configs/exblurf_motion/sunflowers.yaml"
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

echo "All exblurf sequences completed!"