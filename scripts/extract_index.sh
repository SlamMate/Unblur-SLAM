#!/bin/bash
# 提取I2-SLAM数据集文件序号的bash脚本

BASE_PATH="./datasets/I2-SLAM"

# 定义数据集文件夹
DATASETS=("fr2_xyz" "fr1_desk" "fr3_office")

echo "开始提取数据集序号..."
echo "================================"

for dataset in "${DATASETS[@]}"; do
    folder_path="${BASE_PATH}/${dataset}"
    output_file="./scripts/${dataset}_indices.txt"
    
    echo ""
    echo "处理数据集: ${dataset}"
    echo "路径: ${folder_path}"
    
    if [ ! -d "$folder_path" ]; then
        echo "警告: 文件夹不存在: ${folder_path}"
        continue
    fi
    
    # 提取序号：查找所有4位数字开头的文件，提取数字部分，去重并排序
    ls "${folder_path}" | grep -oE '^[0-9]{4}' | sort -u > "${output_file}"
    
    # 统计序号数量
    count=$(wc -l < "${output_file}")
    
    if [ $count -gt 0 ]; then
        echo "找到 ${count} 个唯一序号"
        echo "前5个序号:"
        head -5 "${output_file}"
        echo "已保存到: ${output_file}"
    else
        echo "未找到任何序号"
    fi
done

echo ""
echo "================================"
echo "所有数据集处理完成！"