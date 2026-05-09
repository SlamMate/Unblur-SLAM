#!/usr/bin/env python3
"""
提取锐利帧对应的时间戳
"""

SHARP_FRAME_INDICES = {
    'TUM/fr1_desk': [13, 20, 21, 35, 44, 60, 61, 65, 67, 72, 82, 85, 87, 88],
    'TUM/fr2_xyz': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
                    38, 39, 40, 41],
    'TUM/fr3_office': [0, 1, 3, 11, 14, 16, 18, 22, 23, 24, 25, 31, 33, 34, 44, 46, 47, 50, 
                       51, 52, 66, 69, 70, 73, 75, 77, 82, 83, 84, 85, 86, 87, 88, 94, 95, 96, 
                       97, 100, 101, 102, 103, 104, 105, 109, 110, 111, 112, 113, 123, 124, 125, 
                       137, 145, 147, 149, 151, 152]
}

# 文件路径配置
FILES = {
    'TUM/fr1_desk': './scripts/matched_timestamps_fr1_desk.txt',
    'TUM/fr2_xyz': './scripts/matched_timestamps_fr2_xyz.txt',
    'TUM/fr3_office': './scripts/matched_timestamps_fr3_office.txt'
}

def extract_sharp_timestamps(input_file, sharp_indices, output_file):
    """
    从输入文件中提取锐利帧对应的时间戳行
    
    Args:
        input_file: 输入的时间戳文件路径
        sharp_indices: 锐利帧的索引列表
        output_file: 输出文件路径
    """
    try:
        # 读取所有行
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        print(f"读取文件: {input_file}")
        print(f"总行数: {len(lines)}")
        print(f"锐利帧数量: {len(sharp_indices)}")
        
        # 提取锐利帧对应的行
        sharp_lines = []
        for idx in sharp_indices:
            if idx < len(lines):
                sharp_lines.append(lines[idx])
            else:
                print(f"警告: 索引 {idx} 超出范围 (总行数: {len(lines)})")
        
        # 保存到输出文件
        with open(output_file, 'w') as f:
            f.writelines(sharp_lines)
        
        print(f"成功保存 {len(sharp_lines)} 行到: {output_file}")
        print()
        
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {input_file}")
        print()
    except Exception as e:
        print(f"错误: {e}")
        print()

def main():
    print("=" * 60)
    print("提取锐利帧时间戳")
    print("=" * 60)
    print()
    
    # 处理每个数据集
    for dataset_name, input_file in FILES.items():
        sharp_indices = SHARP_FRAME_INDICES[dataset_name]
        
        # 生成输出文件名
        output_file = input_file.replace('.txt', '_sharp.txt')
        
        extract_sharp_timestamps(input_file, sharp_indices, output_file)
    
    print("=" * 60)
    print("处理完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()