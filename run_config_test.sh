#!/bin/bash
#SBATCH --job-name=config_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/config_test_%j.out
#SBATCH --error=logs/config_test_%j.err

# Config 恢复测试脚本 - 使用 deblur-slam 环境测试 MCD 配置
# 参考 ate_rmse_results.csv 验证结果

set -e
mkdir -p logs

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deblur-slam

cd .

# 测试单个 MCD 序列 (mcd_hcd_nosync_s1r00, 历史 RMSE: 0.162353)
TEST_CONFIG="configs/MCD/mcd_hcd_nosync_s1r00.yaml"
OUTPUT_DIR="MCD"
LOG_FILE="logs/config_test_mcd_s1r00.log"

echo "=========================================="
echo "BAGS-SLAM Config 恢复测试"
echo "Config: $TEST_CONFIG"
echo "参考基线: ate_rmse_results.csv 中 mcd_hcd_nosync_s1r00 RMSE=0.162353"
echo "=========================================="

srun python run.py "$TEST_CONFIG" --only_tracking 2>&1 | tee "$LOG_FILE"

echo ""
echo "测试完成! 检查输出目录: $OUTPUT_DIR/mcd_hcd_nosync_s1r00/"
echo "完整日志: $LOG_FILE"
