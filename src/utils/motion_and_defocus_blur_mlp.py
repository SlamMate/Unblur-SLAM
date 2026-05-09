import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.monogs.utils.pose_utils import get_new_RT_prev, get_new_RT, slerp
from thirdparty.monogs.utils.rotation_conv import quaternion_angle_difference, quaternion_to_matrix, matrix_to_quaternion, quaternion_multiply, quaternion_invert, quaternion_angle_difference_dot
def get_emb(sin_inp):
    """Gets a base embedding for one dimension with sin and cos intertwined"""
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

def get_2d_emb(batch_size, x, y, out_ch, device):
    """Generate 2D positional embeddings"""
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


def compute_local_gradients(image, kernel_size=3, stride=1):
    """计算图像的局部梯度"""
    # 保证输入的tensor是float
    image = image.float()
    # Sobel算子计算水平和垂直梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)

    # 使用卷积计算梯度
    # 用权重来调整注意点,weight*(grad_x+grad_y)
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)

    # 计算局部块的梯度均值
    local_grad_x = F.avg_pool2d(grad_x, kernel_size, stride=stride, padding=kernel_size//2)
    local_grad_y = F.avg_pool2d(grad_y, kernel_size, stride=stride, padding=kernel_size//2)

    return local_grad_x, local_grad_y

def local_gradient_loss(pred, target, kernel_size=3, stride=1):
    """计算局部梯度损失"""

    """示例使用
    pred_depth = torch.randn(1, 1, 256, 256)  # 模拟预测的深度图
    target_depth = torch.randn(1, 1, 256, 256)  # 模拟目标的深度图

    loss = local_gradient_loss(pred_depth, target_depth, kernel_size=5, stride=2)
    print(f"Local Gradient Loss: {loss.item()}")"""
    # 计算预测和目标的局部梯度
    pred_grad_x, pred_grad_y = compute_local_gradients(pred, kernel_size, stride)
    target_grad_x, target_grad_y = compute_local_gradients(target, kernel_size, stride)

    # 使用L1损失计算局部梯度损失
    loss_x = F.l1_loss(pred_grad_x, target_grad_x)
    loss_y = F.l1_loss(pred_grad_y, target_grad_y)

    # 将水平方向和垂直方向的局部梯度损失相加
    loss = loss_x + loss_y

    return loss

class CompositeBlurModel(nn.Module):
    """Composite blur model for SLAM: Motion blur + Defocus blur with multi-resolution training"""
    
    def __init__(self, gaussians, config):
        super().__init__()
        self.gaussians = gaussians
        self.config = config
        
        # Multi-resolution kernel sizes from config
        self.ks1 = self.config["deblur"].get("ks1", 5)
        self.ks2 = self.config["deblur"].get("ks2", 7) 
        self.ks3 = self.config["deblur"].get("ks3", 15)
        self.ks_ss = self.config["deblur"].get("ks_ss", 5)
        
        # Use networks from gaussians (already initialized)
        if hasattr(gaussians, 'mlp_rgb_ss'):
            self.mlp_rgb_ss = gaussians.mlp_rgb_ss
        if hasattr(gaussians, 'mlp_rgb_ms'):
            self.mlp_rgb_ms = gaussians.mlp_rgb_ms
            
        # Create unfold operators for different kernel sizes
        self.unfold1 = nn.Unfold(kernel_size=self.ks1, padding=self.ks1//2).cuda()
        self.unfold2 = nn.Unfold(kernel_size=self.ks2, padding=self.ks2//2).cuda()
        self.unfold3 = nn.Unfold(kernel_size=self.ks3, padding=self.ks3//2).cuda()
        self.unfold_ss = nn.Unfold(kernel_size=self.ks_ss, padding=self.ks_ss//2).cuda()
        
        # Training milestones
        self.stage1_end = 30
        self.stage2_end = 60
        self.ms_steps = self.config["mapping"]["Training"].get("ms_steps", 150)

    def forward(self, image, depth, viewpoint, iteration, opacity, mode='mapping', kf = None):
        """
        Forward pass with multi-resolution kernel estimation
        
        Args:
            image: [3, H, W] rendered image
            depth: [1, H, W] rendered depth
            viewpoint: camera viewpoint with uid
            iteration: current training iteration
            mode: 'train' or 'inference'
            
        Returns:
            dict containing blurred image, mask, and kernel weights
        """
        if mode == "final":
            self.stage1_end = 5000
            self.stage2_end = 10000
        elif mode == "mapping":
            self.stage1_end = 30
            self.stage2_end = 50
        elif mode == "init":
            self.stage1_end = 500
            self.stage2_end = 800

        if mode == "tracking":
            self.current_scale = 3
        else:
            if iteration < self.stage1_end:
                # Stage 1: Small kernel (5x5)
                self.current_scale = 3
                
            elif iteration < self.stage2_end:
                # Stage 2: Medium kernel (9x9)
                self.current_scale = 2
            else:
                # Stage 3: Large kernel (17x17)
                self.current_scale = 1



        # Early return for initial iterations (warm-up)
        """
        if iteration <= 250 and mode == 'init':
            return {
                'blurred': image,
                'composite_blurred': image,
                'mask': torch.zeros_like(depth),
                'kernel_weights': None,
                'depth': depth
            }
        """

        # Apply downsampling based on current scale
        scale_factor = 2 ** (self.current_scale - 1)
        
        if scale_factor > 1 and mode!="tracking":
            # Downsample inputs
            H_orig, W_orig = image.shape[-2:]
            H_scaled = H_orig // scale_factor
            W_scaled = W_orig // scale_factor
            
            opacity = F.interpolate(
                opacity.unsqueeze(0),
                size=(H_scaled, W_scaled),
                mode='bilinear',
                align_corners=False
            )[0]

            # Downsample image and depth
            image_scaled = F.interpolate(
                image.unsqueeze(0), 
                size=(H_scaled, W_scaled), 
                mode='bilinear', 
                align_corners=False
            )[0]
            depth_scaled = F.interpolate(
                depth.unsqueeze(0), 
                size=(H_scaled, W_scaled), 
                mode='nearest'
            )[0]
        else:
            H_orig, W_orig = image.shape[-2:]
            image_scaled = image
            depth_scaled = depth
            H_scaled, W_scaled = image.shape[-2:]
        
        # Prepare inputs
        blur_input = image_scaled.unsqueeze(0)
        depth_input = depth_scaled.unsqueeze(0)
        
        # Normalize depth for network input
        depth_normalized = depth_input.clone()
        if depth_normalized.max() > 0:
            depth_normalized = (depth_normalized - depth_normalized.min()) / (depth_normalized.max() - depth_normalized.min())
        
        # Generate positional encoding
        H, W = blur_input.shape[-2:]
        pos_enc = get_2d_emb(1, H_scaled, W_scaled, 16, blur_input.device)

        # Multi-resolution training strategy
        if iteration < self.stage1_end:
            mapping_idx = kf[viewpoint.timestamp]
            # Stage 1: Small kernel (5x5)
            kernel_weights, mask = self.mlp_rgb_ms(
                mapping_idx,
                pos_enc,
                torch.cat([blur_input, depth_normalized], 1),
                ks = 1
            )
            patches = self.unfold1(blur_input)
            patches = patches.view(1, 3, self.ks1**2, H, W)
            kernel_size = self.ks1
            
        elif iteration < self.stage2_end:
            mapping_idx = kf[viewpoint.timestamp]
            # Stage 2: Medium kernel (9x9)
            kernel_weights, mask = self.mlp_rgb_ms(
                mapping_idx,
                pos_enc,
                torch.cat([blur_input, depth_normalized], 1),
                ks = 2
            )
            patches = self.unfold2(blur_input)
            patches = patches.view(1, 3, self.ks2**2, H, W)
            kernel_size = self.ks2
        else:
            mapping_idx = kf[viewpoint.timestamp]
            # Stage 3: Large kernel (17x17)
            kernel_weights, mask = self.mlp_rgb_ms(
                mapping_idx,
                pos_enc,
                torch.cat([blur_input, depth_normalized], 1),
                ks = 3
            )
            patches = self.unfold3(blur_input)
            patches = patches.view(1, 3, self.ks3**2, H, W)
            kernel_size = self.ks3
        
        # Apply blur kernel
        kernel_weights = kernel_weights.unsqueeze(1)  # [1, 1, k^2, H, W]
        blurred = torch.sum(patches * kernel_weights, 2)[0]  # [3, H, W]
        
        opacity_mask = (opacity > 0.95).view(*depth_scaled.shape)
        # Apply mask
        mask = mask[0]
        # mask = mask * opacity_mask
        composite_blurred = mask * blurred + (1 - mask) * image_scaled

        # Upsample back to original resolution if needed
        """
        if scale_factor > 1:
            composite_blurred = F.interpolate(
                composite_blurred.unsqueeze(0),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )[0]
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )[0]
            # Use original resolution depth for output
            depth_output = depth
        else:
            depth_output = depth_scaled
        """
        
        return {
            'opacity': opacity,
            'blurred': blurred,
            'composite_blurred': composite_blurred,
            'mask': mask,
            'kernel_weights': kernel_weights,
            'depth': depth_scaled,
            'image': image_scaled,
            'kernel_size': kernel_size
        }
    
    def compute_losses(self, opacity, output, gt_image, viewpoint, iteration, mode='mapping'):
        """
        Compute training losses
        
        Args:
            output: dict from forward pass
            gt_image: ground truth image
            iteration: current training iteration
            
        Returns:
            dict of individual losses and total loss
        """
        from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss, ssim

        # Get current scale
        scale_factor = 2 ** (self.current_scale - 1)
        
        depth = output['depth']
        rgb = output['composite_blurred']
        if mode != 'init':
            rgb = (torch.exp(viewpoint.exposure_a)) * rgb + viewpoint.exposure_b
        gt_depth = torch.from_numpy(viewpoint.depth).to(
            dtype=torch.float32, device=rgb.device
        )[None]

        # Downsample ground truth if needed
        if scale_factor > 1 and mode!="tracking":
            # Downsample inputs
            H_orig, W_orig = gt_image.shape[-2:]
            H_scaled = H_orig // scale_factor
            W_scaled = W_orig // scale_factor
            
            opacity = F.interpolate(
                opacity.unsqueeze(0),
                size=(H_scaled, W_scaled),
                mode='bilinear',
                align_corners=False
            )[0]

            gt_image = F.interpolate(
                gt_image.unsqueeze(0),
                size=(H_scaled, W_scaled),
                mode='bilinear',
                align_corners=False
            )[0]
            
            # Downsample image and depth
            """
            rgb = F.interpolate(
                rgb.unsqueeze(0), 
                size=(H_scaled, W_scaled), 
                mode='bilinear', 
                align_corners=False
            )[0]

            depth = F.interpolate(
                depth.unsqueeze(0), 
                size=(H_scaled, W_scaled), 
                mode='nearest'
            )[0]
            """

            gt_depth = F.interpolate(
                gt_depth.unsqueeze(0), 
                size=(H_scaled, W_scaled), 
                mode='nearest'
            )[0]
        else:
            H_scaled, W_scaled = gt_image.shape[-2:]

        opacity_mask = (opacity > 0.95).view(*depth.shape)
        # final 不需要不透明度mask
        # if mode == 'final':
        # opacity_mask = torch.ones_like(opacity_mask)

        depth_mask = opacity_mask

        losses = {}

        # TV loss on depth (if enabled)
        if self.config["deblur"].get("use_depth_loss", False):
            depth_tv = self.tv_loss((depth * depth_mask).unsqueeze(0))
            losses['depth_tv'] = self.config["deblur"].get("depth_loss_alpha", 1e-30) * depth_tv
        
        # Mask sparsity loss (if enabled)
        if self.config["deblur"].get("use_mask_loss", True):
            mask_loss = (output['mask'] * opacity_mask).sum() / (opacity_mask.sum() + 1e-8)
            losses['mask'] = self.config["deblur"].get("mask_loss_alpha", 0.01) * mask_loss
        
        # RGB TV loss (if enabled)
        if self.config["deblur"].get("use_rgbtv_loss", False):
            rgb_tv = self.tv_loss((rgb * depth_mask).unsqueeze(0))
            losses['rgb_tv'] = self.config["deblur"].get("rgbtv_loss_alpha", 1e-30) * rgb_tv
        

        total = 0
        if mode!="tracking" and not self.config["composite_blur"]:
            loss = 0
            loss_l1 = 0
            alpha = self.config["mapping"]["Training"]["alpha"] if "alpha" in self.config["mapping"]["Training"] else 0.95
            rgb_boundary_threshold = self.config["mapping"]["Training"]["rgb_boundary_threshold"]
            _, h, w = gt_image.shape
            mask_shape = (1, h, w)

            rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
            l1_rgb = torch.abs(rgb * rgb_pixel_mask - gt_image * rgb_pixel_mask)

            depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
            l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

            ssim_loss = 1.0 - ssim(rgb, gt_image)
            lambda_dssim = self.config["mapping"]["opt_params"]["lambda_dssim"]
            loss = (1.0 - lambda_dssim) * (
                l1_rgb
            ) + lambda_dssim * (ssim_loss)

            total += alpha * loss.mean() + (1 - alpha) * l1_depth.mean()
        
        # Add regularization terms
        for key in ['depth_tv', 'mask', 'rgb_tv']:
            if key in losses:
                total = total + losses[key]
        
        losses['total'] = total
        return losses
    
    def compute_BAD_losses(self, image, gt_image, images, depths, viewpoint, scale, seen=True, mode="mapping", opacities=None, prev = None):

        # Get current scale
        scale_factor = 2 ** (scale - 1)

        gt_depth = torch.from_numpy(viewpoint.depth).to(
            dtype=torch.float32, device=image.device
        )[None]

        grad_mask = viewpoint.grad_mask

        # Downsample ground truth if needed
        if scale_factor > 1:
            # Downsample inputs
            H_orig, W_orig = gt_image.shape[-2:]
            H_scaled = H_orig // scale_factor
            W_scaled = W_orig // scale_factor

            gt_image = F.interpolate(
                gt_image.unsqueeze(0),
                size=(H_scaled, W_scaled),
                mode='bilinear',
                align_corners=False
            )[0]

            gt_depth = F.interpolate(
                gt_depth.unsqueeze(0), 
                size=(H_scaled, W_scaled), 
                mode='nearest'
            )[0]

            grad_mask = F.interpolate(
                grad_mask.unsqueeze(0), 
                size=(H_scaled, W_scaled), 
                mode='bilinear'
            )[0]
        else:
            H_scaled, W_scaled = gt_image.shape[-2:]

        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
        if mode == "tracking":
            lambda_dssim = self.config['mapping']["opt_params"]["lambda_dssim"]
            lambda_total_variation = self.config['mapping']["opt_params"]["lambda_total_variation"]
            _, h, w = gt_image.shape
            mask_shape = (1, h, w)
            rgb_boundary_threshold = self.config['mapping']["Training"]["rgb_boundary_threshold"]
            rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
            rgb_pixel_mask = rgb_pixel_mask * grad_mask
            
            # l1 = (opacities.mean(0) * torch.abs(image_ab * rgb_pixel_mask - gt_image * rgb_pixel_mask)).mean()
            l1 = (opacities.mean(0) * torch.abs(image_ab * rgb_pixel_mask - gt_image * rgb_pixel_mask))
            # trans_dir_loss  + rot_dir_loss可能是一些正则化项
            # loss = l1
            loss = (1.0 - lambda_dssim) * (
                l1
            ) + lambda_dssim * opacities.mean(0) *(1.0 - ssim(image, gt_image))
            #print(f"Loss: {loss.shape}")
            depth = depths.mean(0)
            # l1_gradient = local_gradient_loss(depth, gt_depth)
            lambda_rot_smooth = self.config["mapping"]["opt_params"]["lambda_rot_smooth"]    
            lambda_trans_smooth = self.config["mapping"]["opt_params"]["lambda_trans_smooth"]

            q_cur_0, t_cur_0 = get_new_RT(viewpoint, 0)
            q_cur_1, t_cur_1  = get_new_RT(viewpoint, 1)

            if prev.uid != viewpoint.uid:

                with torch.no_grad():
                    # q_prev_0, t_prev_0 = get_new_RT(prev, 0)  # Start of previous frame
                    q_prev_1, t_prev_1 = get_new_RT_prev(prev, 1)  # End of previous frame

                rot_dir_loss = lambda_rot_smooth * torch.norm( quaternion_angle_difference(q_cur_0, q_cur_1) ** 2  + quaternion_angle_difference(q_prev_1, q_cur_0) ** 2)
                trans_dir_loss = lambda_trans_smooth * ( torch.norm(t_cur_1 - t_cur_0) ** 2 + torch.norm(t_cur_0 - t_prev_1) ** 2 )

            else:
                rot_dir_loss = 0.0
                trans_dir_loss = 0.0

            # rot_dir_loss = 0.0
            # trans_dir_loss = 0.0

            # trans_dir_loss  + rot_dir_loss可能是一些正则化项
            return loss.mean() + trans_dir_loss  + rot_dir_loss
        if mode == "mapping":
            alpha = self.config["mapping"]["Training"]["alpha"] if "alpha" in self.config["mapping"]["Training"] else 0.95
            lambda_dssim = self.config["mapping"]["opt_params"]["lambda_dssim"]
            lambda_total_variation = 0.0 #config["opt_params"]["lambda_total_variation"]
            _, h, w = gt_image.shape
            mask_shape = (1, h, w)
            rgb_boundary_threshold = self.config["mapping"]["Training"]["rgb_boundary_threshold"]
            rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
            
            l1 = (torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)).mean()


            loss_edges = torch.tensor(0.0).to(image.device)

            loss = (1.0 - lambda_dssim) * (
                l1
            ) + lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_edges #+ total_variation_loss
            #print("l1: ", l1.item(), "loss_edges", loss_edges.item())

            l1_depth = None

            depth = depths.mean(dim=0)
            depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
            l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
            
            lambda_rot_smooth = self.config["mapping"]["opt_params"]["lambda_rot_smooth"]    
            lambda_trans_smooth = self.config["mapping"]["opt_params"]["lambda_trans_smooth"]

            q_cur_0, t_cur_0 = get_new_RT(viewpoint, 0)
            q_cur_1, t_cur_1  = get_new_RT(viewpoint, 1)

            
            """
            if prev.uid != viewpoint.uid:
                with torch.no_grad():
                    q_prev_0, t_prev_0 = get_new_RT_prev(prev, 0)  # Start of previous frame
                    q_prev_1, t_prev_1 = get_new_RT_prev(prev, 1)  # End of previous frame

                rot_dir_loss = lambda_rot_smooth * torch.norm( quaternion_angle_difference(q_cur_0, q_cur_1) ** 2  + quaternion_angle_difference(q_prev_1, q_cur_0) ** 2)
                trans_dir_loss = lambda_trans_smooth * ( torch.norm(t_cur_1 - t_cur_0) ** 2 + torch.norm(t_cur_0 - t_prev_1) ** 2 )
            else:
                rot_dir_loss = 0.0
                trans_dir_loss = 0.0
            """
            rot_dir_loss = 0.0
            trans_dir_loss = 0.0

            if l1_depth is None:
                return alpha * loss.mean() + rot_dir_loss + trans_dir_loss
            else:
                return alpha * loss.mean() + (1 - alpha) * l1_depth.mean() + rot_dir_loss + trans_dir_loss
        
    def tv_loss(self, grids):
        """Total variation loss for smoothness regularization"""
        h_tv = torch.pow((grids[:, :, 1:, :] - grids[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((grids[:, :, :, 1:] - grids[:, :, :, :-1]), 2).sum()
        h_count = grids[:, :, 1:, :].numel()
        w_count = grids[:, :, :, 1:].numel()
        return 2 * (h_tv / h_count + w_tv / w_count) / grids.shape[0]


class BlurAwareSLAMMapper:
    """Helper class to integrate blur-aware mapping into SLAM pipeline"""
    
    def __init__(self, gaussians, config):
        self.gaussians = gaussians
        self.config = self.config
        self.blur_model = CompositeBlurModel(gaussians, config)
        self.blur_start_iter = self.config["deblur"].get("start_iter", 250)
    def apply_blur_aware_rendering(self, render_pkg, viewpoint, iteration):
        """
        Apply blur-aware rendering to a standard render package
        
        Args:
            render_pkg: dict from standard Gaussian rendering
            viewpoint: camera viewpoint
            iteration: current iteration
            
        Returns:
            Modified render_pkg with blur applied
        """
        if not self.config["deblur"]["open"] or iteration < self.blur_start_iter:
            return render_pkg
        
        # Extract rendered image and depth
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        
        # Apply blur model
        blur_output = self.blur_model(image, depth, viewpoint, iteration)
        
        # Update render package
        render_pkg["render"] = blur_output["composite_blurred"]
        render_pkg["render_sharp"] = image  # Keep original sharp render
        render_pkg["blur_mask"] = blur_output["mask"]
        render_pkg["kernel_weights"] = blur_output["kernel_weights"]
        
        return render_pkg