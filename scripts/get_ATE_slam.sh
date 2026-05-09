#!/bin/bash

# Base directory
BASE_DIR="./no_detector_mcd_recent"

# Output files
OUTPUT_FILE="ate_rmse_results.txt"
SUMMARY_FILE="ate_rmse_summary.txt"

# Clear output files
> "$OUTPUT_FILE"
> "$SUMMARY_FILE"

echo "======================================"
echo "Extracting ATE RMSE values..."
echo "======================================"

# Counter for successful extractions
count=0
sum=0

# Process each sequence from s1r00 to s1r56
for i in $(seq -w 0 56); do
    SEQ_DIR="mcd_hcd_nosync_s1r${i}"
    FILE_PATH="${BASE_DIR}/${SEQ_DIR}/traj/metrics_full_traj.txt"
    
    if [ -f "$FILE_PATH" ]; then
        # Extract RMSE value using grep and sed
        RMSE=$(grep "'rmse':" "$FILE_PATH" 2>/dev/null | sed -E "s/.*'rmse': ([0-9.]+).*/\1/")
        
        if [ ! -z "$RMSE" ]; then
            echo "${SEQ_DIR}: ${RMSE}" | tee -a "$OUTPUT_FILE"
            sum=$(echo "$sum + $RMSE" | bc -l)
            ((count++))
        else
            echo "${SEQ_DIR}: Failed to extract RMSE" | tee -a "$OUTPUT_FILE"
        fi
    else
        echo "${SEQ_DIR}: File not found" | tee -a "$OUTPUT_FILE"
    fi
done

echo ""
echo "======================================"
echo "SUMMARY"
echo "======================================"

if [ $count -gt 0 ]; then
    # Calculate mean
    MEAN=$(echo "scale=6; $sum / $count" | bc -l)
    
    echo "Successfully processed: $count/57 sequences" | tee -a "$SUMMARY_FILE"
    echo "Mean ATE RMSE: $MEAN" | tee -a "$SUMMARY_FILE"
    echo ""
    echo "Individual results saved to: $OUTPUT_FILE"
    echo "Summary saved to: $SUMMARY_FILE"
else
    echo "ERROR: No RMSE values could be extracted!"
fi