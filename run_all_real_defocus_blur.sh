#!/bin/bash
#SBATCH --job-name=deblur_defocus
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# 配置文件数组
configs=(
    "configs/deblur_nerf_defocus/defocusbush.yaml"
    "configs/deblur_nerf_defocus/defocuscake.yaml"
    "configs/deblur_nerf_defocus/defocuscaps.yaml"
    "configs/deblur_nerf_defocus/defocuscisco.yaml"
    "configs/deblur_nerf_defocus/defocuscoral.yaml"
    "configs/deblur_nerf_defocus/defocuscupcake.yaml"
    "configs/deblur_nerf_defocus/defocuscups.yaml"
    "configs/deblur_nerf_defocus/defocusdaisy.yaml"
    "configs/deblur_nerf_defocus/defocussausage.yaml"
    "configs/deblur_nerf_defocus/defocusseal.yaml"
    "configs/deblur_nerf_defocus/defocustools.yaml"
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

echo "All defocus sequences completed!"