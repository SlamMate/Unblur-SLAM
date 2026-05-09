import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def inverse_gamma_srgb(image):
    """Apply precise sRGB inverse gamma correction."""
    normalized = image.astype(np.float32) / 255.0
    linear = np.zeros_like(normalized)
    threshold = 0.04045
    mask_linear = normalized <= threshold
    linear[mask_linear] = normalized[mask_linear] / 12.92
    mask_power = normalized > threshold
    linear[mask_power] = np.power((normalized[mask_power] + 0.055) / 1.055, 2.4)
    return (linear * 255.0).astype(np.uint8)

def process_directory_overwrite(input_dir):
    """Process and overwrite all images in a directory."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"⚠️  {input_dir} 不存在，跳过...")
        return
    
    image_files = list(input_path.glob('**/*.png')) + list(input_path.glob('**/*.jpg')) + list(input_path.glob('**/*.jpeg'))
    
    if not image_files:
        print(f"未找到图像文件：{input_dir}")
        return
    
    print(f"\n处理 {len(image_files)} 张图像：{input_dir}")
    
    success_count = 0
    for img_path in tqdm(image_files, desc="线性化处理"):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_linear = inverse_gamma_srgb(img)
            cv2.imwrite(str(img_path), img_linear)
            success_count += 1
        except Exception as e:
            print(f"处理错误 {img_path}: {e}")
    
    print(f"✅ 成功处理 {success_count}/{len(image_files)} 张图像")

base_dirs = [
    '/var/scratch/qzhang2/Deblur-SLAM/thirdparty/EVSSM/download/GoPro_scannet_expand/train/input',
    '/var/scratch/qzhang2/Deblur-SLAM/thirdparty/EVSSM/download/GoPro_scannet_expand/train/target'
]

print("\n" + "=" * 60)
print("反Gamma校正（线性化）- 覆盖模式（无备份）")
print("⚠️  警告：原始图像将被永久修改！")
print("=" * 60)

for input_dir in base_dirs:
    process_directory_overwrite(input_dir)

print("\n" + "=" * 60)
print("✅ 所有处理完成！原始图像已被永久替换")
print("⚠️  无法恢复原始图像（未创建备份）")
print("=" * 60)
