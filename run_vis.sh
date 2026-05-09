#!/bin/bash
#SBATCH --job-name=tum_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=10:00:00

# 创建日志目录
mkdir -p logs

srun python run.py configs/deblur_nerf_motion/blurpuppet.yaml
srun python run.py configs/deblur_nerf_motion/blurstair.yaml