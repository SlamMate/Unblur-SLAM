import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

def get_2d_emb(batch_size, x, y, out_ch, device):
    out_ch = int(np.ceil(out_ch / 4) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, out_ch, 2).float() / out_ch))
    pos_x = torch.arange(x, device=device).type(inv_freq.type())*2*np.pi/x
    pos_y = torch.arange(y, device=device).type(inv_freq.type())*2*np.pi/y
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    emb_x = get_emb(sin_inp_x).unsqueeze(1)
    emb_y = get_emb(sin_inp_y)
    emb = torch.zeros((x, y, out_ch * 2), device=device)
    emb[:, :, : out_ch] = emb_x
    emb[:, :, out_ch : 2 * out_ch] = emb_y
    return emb[None, :, :, :].repeat(batch_size, 1, 1, 1)

class CompositeBlurModel_small(nn.Module):
    """复合模糊模型：运动模糊 + 失焦模糊（单kernel版本）"""
    def __init__(self, gaussians, config):
        super().__init__()
        self.gaussians = gaussians
        self.config = config
        
        # 使用单个kernel size = 5的BAGS网络
        if hasattr(gaussians, 'mlp_rgb_ss'):
            self.mlp_rgb_ss = gaussians.mlp_rgb_ss
        
        self.kernel_size = 5
        
        # 只需要kernel_size=5的unfold操作器
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.kernel_size//2).cuda()
    
    def forward(self, image, depth, viewpoint, iteration, mode='train'):
        """
        前向传播：运动模糊 -> 失焦模糊
        Args:
            images_tensor: [n_virtual_cams, 3, H, W] 多个虚拟相机的渲染
            depths_tensor: [n_virtual_cams, 1, H, W] 对应的深度
            viewpoint: 当前视点
            iteration: 当前迭代次数
            mode: 'train' 或 'inference'
        Returns:
            dict: 包含运动模糊、失焦模糊和复合模糊图像
        """
        # Step 2: 失焦模糊 - 通过BAGS单kernel网络

        blur_input = image.unsqueeze(0) # [1, 3, H, W]
        depth_input = depth.unsqueeze(0) # [1, 1, H, W]
        
        # 生成位置编码
        H, W = blur_input.shape[-2:]
        pos_enc = get_2d_emb(1, H, W, 16, blur_input.device)
        
        # 使用单个kernel_size=5的网络
        kernel_weights, mask = self.mlp_rgb_ss(
            viewpoint.uid,
            pos_enc,
            torch.cat([blur_input, depth_input], 1),
            iteration
        )
        
        # 应用模糊核
        patches = self.unfold(blur_input)  # [1, 3*25, H*W]
        patches = patches.view(1, 3, self.kernel_size**2, H, W)
        kernel_weights = kernel_weights.unsqueeze(1)  # [1, 1, 25, H, W]
        
        # 模糊图像
        blurred = torch.sum(patches * kernel_weights, 2)[0]  # [3, H, W]
        
        # 复合模糊：使用mask控制模糊程度
        composite_blurred = mask[0] * blurred + (1 - mask[0]) * image
        
        return {
            'blurred': blurred,
            'composite_blurred': composite_blurred,
            'mask': mask[0],
            'kernel_weights': kernel_weights,
            'depth': depth
        }
