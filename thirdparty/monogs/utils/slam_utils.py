# Copyright 2024 The MonoGS Authors.

# Licensed under the License issued by the MonoGS Authors
# available here: https://github.com/muskie82/MonoGS/blob/main/LICENSE.md

import torch
import torch.nn.functional as F
import cv2
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.monogs.utils.pose_utils import get_new_RT, slerp


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)

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


# Not used, but kept for reference
def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)

def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = (opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask))
    return l1.mean()

# Not used, but kept for reference
def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)

    if viewpoint.is_valid:
        try:
            if isinstance(viewpoint.depth, torch.Tensor):
                gt_depth = viewpoint.depth
            else:
                gt_depth = torch.from_numpy(viewpoint.depth)
            gt_depth = gt_depth.to(dtype=torch.float32, device=image.device)[None]
        except RuntimeError as e:
            print(f"Error processing depth: {e}")
            print(f"viewpoint.depth type: {type(viewpoint.depth)}")
            if isinstance(viewpoint.depth, torch.Tensor):
                print(f"Is CUDA: {viewpoint.depth.is_cuda}")
            raise
    loss = 0
    if config["Training"]["ssim_loss"]:
        ssim_loss = 1.0 - ssim(image, gt_image)
        
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    if config["Training"]["ssim_loss"]:
        hyperparameter = config["opt_params"]["lambda_dssim"]
        loss += (1.0 - hyperparameter) * l1_rgb + hyperparameter * ssim_loss
    else:
        loss += l1_rgb

    if viewpoint.is_valid:
        depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
        l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    if viewpoint.is_valid:
        return alpha * loss.mean() + (1 - alpha) * l1_depth.mean()
    else:
        return loss.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def variance_of_laplacian(image):
    """
    Pure PyTorch implementation of variance of Laplacian
    """
    # Define Laplacian kernel
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    if image.device.type == 'cuda':
        laplacian_kernel = laplacian_kernel.cuda()
    
    # Apply convolution
    if len(image.shape) == 3:  # Add batch dimension if needed
        image = image.unsqueeze(0)
    
    # Ensure single channel
    if image.shape[1] > 1:
        image = torch.mean(image, dim=1, keepdim=True)
    
    # Apply Laplacian filter
    laplacian = torch.nn.functional.conv2d(
        image, laplacian_kernel, padding=1
    )
    
    # Compute variance
    variance = torch.var(laplacian).item()
    
    return variance, laplacian.squeeze()

def BAD_tracking_loss(config, image, gt_image, images, opacities, viewpoint, depths, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
        
    lambda_dssim = config['mapping']["opt_params"]["lambda_dssim"]
    lambda_total_variation = config['mapping']["opt_params"]["lambda_total_variation"]
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config['mapping']["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    
    # l1 = (opacities.mean(0) * torch.abs(image_ab * rgb_pixel_mask - gt_image * rgb_pixel_mask)).mean()
    l1 = (opacities.mean(0) * torch.abs(image_ab * rgb_pixel_mask - gt_image * rgb_pixel_mask))

    #loss = (1.0 - lambda_dssim) * (
    #    l1
    #) + lambda_dssim * opacities.mean(0) * (1.0 - ssim(image, gt_image))
    # trans_dir_loss  + rot_dir_loss可能是一些正则化项
    loss = l1
    #print(f"Loss: {loss.shape}")
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth = depths.mean(0)
    l1_gradient = local_gradient_loss(depth, gt_depth)
    return loss.mean()

def BAD_mapping_loss(config, image, gt_image, images, depths, viewpoint, seen=True, initialization=False):
    
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    
    alpha = config["mapping"]["Training"]["alpha"] if "alpha" in config["mapping"]["Training"] else 0.95
    lambda_dssim = config["mapping"]["opt_params"]["lambda_dssim"]
    lambda_total_variation = 0.0 #config["opt_params"]["lambda_total_variation"]
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["mapping"]["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    
    l1 = (torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)).mean()


    loss_edges = torch.tensor(0.0).to(image.device)

    loss = (1.0 - lambda_dssim) * (
        l1
    ) + lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_edges #+ total_variation_loss
    #print("l1: ", l1.item(), "loss_edges", loss_edges.item())
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]

    l1_depth = None

    depth = depths.mean(dim=0)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    
    if l1_depth is None:
        return alpha * loss.mean()
    else:
        return alpha * loss.mean() + (1 - alpha) * l1_depth.mean()
    
def render_video(path, frames, framerate=5, frames_idx = None):
    if len(frames) == 0:
        return
    
    print("the shape is :", frames[0].shape)
    # Handle both grayscale (H, W) and color (H, W, C) images
    if len(frames[0].shape) == 3:
        height, width, _ = frames[0].shape
    else:
        height, width = frames[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, framerate, (width, height))

    for i, frame in enumerate(frames):
        # 兼容 tensor 和 numpy array
        if hasattr(frame, 'numpy'):
            frame = frame.numpy()
        frame = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_BGR2RGB)
        video.write(frame)
    video.release()

def plot_tensor(tensor, path):
    import torchvision.transforms.functional as TF

    tensor = tensor.float()
    tensor = TF.to_pil_image(tensor)

    tensor.save(path)