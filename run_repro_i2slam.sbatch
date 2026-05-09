#!/bin/bash
#SBATCH --job-name=unblur_repro
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/unblur_repro_%j.out
#SBATCH --error=logs/unblur_repro_%j.err

set -eo pipefail

mkdir -p logs

cd /ivi/zfs/s0/original_homes/qzhang/BAGS-SLAM
source /home/qzhang/anaconda3/etc/profile.d/conda.sh
conda activate /ivi/zfs/s0/original_homes/qzhang/conda_env/splat-slam
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
export LD_LIBRARY_PATH="/ivi/zfs/s0/original_homes/qzhang/conda_env/splat-slam/lib:${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/ivi/zfs/s0/original_homes/qzhang/BAGS-SLAM/.cache/transformers}"
export HF_HOME="${HF_HOME:-/ivi/zfs/s0/original_homes/qzhang/BAGS-SLAM/.cache/huggingface}"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_HOME"
# 24 GB cards (3090/A5000) need a friendlier allocator vs paper's A6000 (48 GB).
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

echo "[$(date)] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[$(date)] python=$(which python)"
python -c "import torch; print(f'cuda available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}')"

CONFIG="${1:-configs/I2slam/freiburg1_desk.yaml}"
echo "[$(date)] running: $CONFIG"
python run.py "$CONFIG"

echo "[$(date)] done"
