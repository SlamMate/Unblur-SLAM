# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import random
import torch
from pathlib import Path
from PIL import Image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = torch.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def update_cam(cfg):
    """
    Update the camera intrinsics according to the pre-processing config,
    such as resize or edge crop
    """
    # resize the input images to crop_size(variable name used in lietorch)
    H, W = cfg['cam']['H'], cfg['cam']['W']
    fx, fy = cfg['cam']['fx'], cfg['cam']['fy']
    cx, cy = cfg['cam']['cx'], cfg['cam']['cy']

    h_edge, w_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']
    H_out, W_out = cfg['cam']['H_out'], cfg['cam']['W_out']

    fx = fx * (W_out + w_edge * 2) / W
    fy = fy * (H_out + h_edge * 2) / H
    cx = cx * (W_out + w_edge * 2) / W
    cy = cy * (H_out + h_edge * 2) / H
    H, W = H_out, W_out

    cx = cx - w_edge
    cy = cy - h_edge
    return H,W,fx,fy,cx,cy    


@torch.no_grad()
def align_scale_and_shift(prediction, target, weights):

    '''
    weighted least squares problem to solve scale and shift: 
        min sum{ 
                  weight[i,j] * 
                  (prediction[i,j] * scale + shift - target[i,j])^2 
               }

    prediction: [B,H,W]
    target: [B,H,W]
    weights: [B,H,W]
    '''

    if weights is None:
        weights = torch.ones_like(prediction).to(prediction.device)
    if len(prediction.shape)<3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        weights = weights.unsqueeze(0)  
    a_00 = torch.sum(weights * prediction * prediction, dim=[1,2])
    a_01 = torch.sum(weights * prediction, dim=[1,2])
    a_11 = torch.sum(weights, dim=[1,2])
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(weights * prediction * target, dim=[1,2])
    b_1 = torch.sum(weights * target, dim=[1,2])
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b            
    det = a_00 * a_11 - a_01 * a_01
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    error = (scale[:,None,None]*prediction+shift[:,None,None]-target).abs()
    masked_error = error*weights
    error_sum = masked_error.sum(dim=[1,2])
    error_num = weights.sum(dim=[1,2])
    avg_error = error_sum/error_num

    return scale,shift,avg_error


def save_tensor(tensor: torch.Tensor, timestamp: str, save_dir: str = "./output/sharp", save_image: bool = False) -> str:
    """
    保存GPU/CPU tensor到本地（同时保存.pt和.png格式）
    
    Args:
        tensor: 要保存的tensor，shape应为 (C, H, W) 或 (B, C, H, W)
        timestamp: 时间戳字符串，作为文件名
        save_dir: 保存目录
        save_image: 是否同时保存为可查看的图片
        
    Returns:
        保存的.pt文件完整路径
    """
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存.pt格式（完整信息）
    pt_filepath = save_path / f"{timestamp}.pt"
    tensor_to_save = tensor.cpu() if tensor.is_cuda else tensor
    torch.save({
        'tensor': tensor_to_save,
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'timestamp': timestamp
    }, pt_filepath)
    
    # 同时保存为图片格式（可查看）
    if save_image:
        png_filepath = save_path / "visualization" / f"{timestamp}.png"
        png_filepath.parent.mkdir(parents=True, exist_ok=True)

        # 处理tensor维度
        img_tensor = tensor
        if tensor.dim() == 4:  # (B, C, H, W)
            img_tensor = tensor[0]  # 取第一张
        
        # 确保在CPU上
        if img_tensor.is_cuda:
            img_tensor = img_tensor.cpu()
        
        # 转换为numpy并调整维度
        if img_tensor.shape[0] == 1:  # 灰度图
            img_array = img_tensor.squeeze(0).numpy()
        elif img_tensor.shape[0] == 3:  # RGB
            img_array = img_tensor.permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
        else:
            # 如果通道数不是1或3，只保存前3个通道
            img_array = img_tensor[:3].permute(1, 2, 0).numpy()
        
        # 归一化到0-255
        if img_array.min() < 0 or img_array.max() <= 1.0:
            # 假设数据在[-1,1]或[0,1]范围
            img_array = np.clip(img_array, -1, 1)
            img_array = ((img_array + 1) * 127.5).astype(np.uint8) if img_array.min() < 0 else (img_array * 255).astype(np.uint8)
        else:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # 保存图片
        Image.fromarray(img_array).save(png_filepath)
        print(f"Saved: {pt_filepath} (tensor) and {png_filepath} (image)")
    else:
        print(f"Saved: {pt_filepath}")
    
    return str(pt_filepath)

def load_tensor(timestamp: str, save_dir: str = "./output/sharp", device: str = 'cuda') -> torch.Tensor:
    """
    根据时间戳从本地加载tensor
    
    Args:
        timestamp: 时间戳字符串，对应保存时的文件名
        save_dir: 保存目录
        device: 加载到的设备 ('cuda' 或 'cpu')
        
    Returns:
        加载的tensor
    """
    # 构建文件路径
    filepath = Path(save_dir) / f"{timestamp}.pt"
    
    # 加载数据
    checkpoint = torch.load(filepath, map_location='cpu')
    tensor = checkpoint['tensor']
    
    # 移动到指定设备
    if device == 'cuda' and torch.cuda.is_available():
        tensor = tensor.cuda()
    
    return tensor