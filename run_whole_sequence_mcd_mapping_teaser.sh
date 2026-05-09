#!/bin/bash
#SBATCH --job-name=tum_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=48:00:00


# 设置输出目录
OUTPUT_DIR="./output"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 日志文件
MAIN_LOG="$LOG_DIR/execution_$(date +%Y%m%d_%H%M%S).log"

/MCD_mapping/mcd_hcd_ns1r15.yaml
# 定义配置文件列表
# 定义配置文件列表
configs=(

"configs/MCD_mapping/mcd_hcd_sync_s2r22.yaml"

)

# 记录开始时间
start_time=$(date +%s)
echo "Starting execution at $(date)" | tee -a "$MAIN_LOG"

# 循环执行每个配置
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    
    # 提取序列名称
    sequence_name=$(basename "$config" .yaml)
    output_file="$OUTPUT_DIR/${sequence_name}.txt"
    full_log="$LOG_DIR/${sequence_name}_full.log"
    
    echo "========================================"  | tee -a "$MAIN_LOG"
    echo "[$((i+1))/${#configs[@]}] Processing: $sequence_name" | tee -a "$MAIN_LOG"
    echo "Config: $config" | tee -a "$MAIN_LOG"
    echo "========================================"  | tee -a "$MAIN_LOG"
    
    # 执行命令
    if srun python run.py "$config" --only_tracking > "$full_log" 2>&1; then
        # 成功执行，提取最后10行
        tail -n 20 "$full_log" > "$output_file"
        echo "✓ Success: Last 20 lines saved to $output_file" | tee -a "$MAIN_LOG"
    else
        # 执行失败
        echo "✗ Error: Command failed for $sequence_name" | tee -a "$MAIN_LOG"
        echo "Check full log: $full_log" | tee -a "$MAIN_LOG"
        # 仍然保存最后10行以便查看错误
        tail -n 20 "$full_log" > "$output_file"
    fi
    
    echo "" | tee -a "$MAIN_LOG"
done

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "========================================"  | tee -a "$MAIN_LOG"
echo "All tasks completed at $(date)" | tee -a "$MAIN_LOG"
echo "Total duration: $((duration / 60)) minutes $((duration % 60)) seconds" | tee -a "$MAIN_LOG"
echo "Results saved in: $OUTPUT_DIR" | tee -a "$MAIN_LOG"
echo "Full logs saved in: $LOG_DIR" | tee -a "$MAIN_LOG"