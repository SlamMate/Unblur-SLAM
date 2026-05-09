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

"""
Monocular depth estimation priors for Deblur-SLAM.

Supports:
- Omnidata
- Depth Anything V2
- Depth Anything V3 (NEW)
- Depth Anything V3 Metric (NEW)
"""

import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2
import numpy as np

from thirdparty.monogs.utils.slam_utils import plot_tensor
from torchvision.transforms import Compose
from src.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
import sys
sys.path.insert(0, './src/depth-anything_v3/src')
from depth_anything_3.api import DepthAnything3


def get_mono_depth_estimator(cfg):
    """
    Get monocular depth estimator based on configuration.
    
    Args:
        cfg: Configuration dictionary containing:
            - device: torch device
            - mono_prior.depth: model type ("omnidata", "anydepth_v2", "anydepth_v3", "anydepth_v3_metric")
            - mono_prior.depth_pretrained: pretrained weights path or HF model name
    
    Returns:
        Depth estimation model
    """
    device = cfg["device"]
    depth_model = cfg["mono_prior"]["depth"]
    depth_pretrained = cfg["mono_prior"]["depth_pretrained"]
    
    if depth_model == "omnidata":
        model = get_omnidata_model(depth_pretrained, device, 1)
    elif depth_model == "anydepth_v2":
        model = get_anydepth_model(depth_pretrained, device)
    elif depth_model == "anydepth_v3":
        model = get_anydepth_v3_model(depth_pretrained, device)
    elif depth_model == "anydepth_v3_metric":
        model = get_anydepth_v3_metric_model(depth_pretrained, device)
    else:
        raise NotImplementedError(f"Depth model '{depth_model}' not implemented. "
                                  f"Available: omnidata, anydepth_v2, anydepth_v3, anydepth_v3_metric")
    
    return model


def get_omnidata_model(pretrained_path, device, num_channels):
    """Load Omnidata depth model."""
    from thirdparty.mono_priors.omnidata.modules.midas.dpt_depth import DPTDepthModel
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=num_channels)
    checkpoint = torch.load(pretrained_path)
    
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def get_anydepth_model(pretrained_path, device):
    """Load Depth Anything V2 model."""
    from src.depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitl'
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(pretrained_path, map_location=device))
    return depth_anything.to(device).eval()


def get_anydepth_v3_model(pretrained_path, device):
    """
    Load Depth Anything V3 model for relative depth estimation.
    
    Args:
        pretrained_path: HuggingFace model name or alias
            Aliases: "da3-small", "da3-base", "da3-large", "da3-giant", "da3mono-large"
            Full names: "depth-anything/DA3-LARGE", "depth-anything/DA3MONO-LARGE", etc.
        device: torch device
    
    Returns:
        DA3 model (DepthAnything3 instance)
    """
    from src.depth_anything_v3.dpt import DepthAnything3
    
    # Model name mapping for convenience
    model_name_map = {
        'da3-small': 'depth-anything/DA3-SMALL',
        'da3-base': 'depth-anything/DA3-BASE',
        'da3-large': 'depth-anything/DA3-LARGE',
        'da3-giant': 'depth-anything/DA3-GIANT',
        'da3mono-large': 'depth-anything/DA3MONO-LARGE',
    }
    
    # Check if pretrained_path is a known alias
    model_name = model_name_map.get(pretrained_path.lower(), pretrained_path)
    
    print(f"[DA3] Loading model: {model_name}")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    
    return model


def get_anydepth_v3_metric_model(pretrained_path, device):
    """
    Load Depth Anything V3 Metric model for metric depth estimation.
    
    Returns depth in meters when using predict_mono_depth.
    
    Args:
        pretrained_path: HuggingFace model name or alias
            Aliases: "da3metric-large", "da3nested-giant-large"
            Full names: "depth-anything/DA3METRIC-LARGE", "depth-anything/DA3NESTED-GIANT-LARGE"
        device: torch device
    
    Returns:
        DA3 Metric model (DepthAnything3 instance)
    """
    
    model_name_map = {
        'da3metric-large': 'depth-anything/DA3METRIC-LARGE',
        'da3nested-giant-large': 'depth-anything/DA3NESTED-GIANT-LARGE',
    }
    
    model_name = model_name_map.get(pretrained_path.lower(), pretrained_path)
    
    print(f"[DA3-Metric] Loading model: {model_name}")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    
    return model


@torch.no_grad()
def predict_mono_depth(model, idx, input, cfg, device):
    """
    Predict monocular depth from a single image.
    
    Args:
        model: Depth estimation model
        idx: Frame index for saving
        input: tensor (1, 3, H, W) - RGB image (normalized to [0, 1] or ImageNet norm)
        cfg: Configuration dictionary
        device: torch device
    
    Returns:
        depth: tensor (H, W) - normalized depth map [0, 1]
    """
    depth_model = cfg["mono_prior"]["depth"]
    output_dir = f"{cfg['data']['output']}/{cfg['scene']}"
    
    if depth_model == "omnidata":
        output = _predict_omnidata(model, input, device)
        
    elif depth_model == "anydepth_v2":
        output = _predict_anydepth_v2(model, input, device)
        
    elif depth_model in ["anydepth_v3", "anydepth_v3_metric"]:
        output = _predict_anydepth_v3(model, input, cfg, device)
        
    else:
        raise NotImplementedError(f"Depth model '{depth_model}' not implemented")
    
    # Save depth map
    output_path_np = f"{output_dir}/mono_priors/depths/{idx:05d}.npy"
    final_depth = output.detach().cpu().float().numpy()
    np.save(output_path_np, final_depth)
    
    return output


def _predict_omnidata(model, input, device):
    """Omnidata depth prediction."""
    image_size = (512, 512)
    input_size = input.shape[-2:]
    trans_totensor = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    img_tensor = trans_totensor(input).to(device)
    output = model(img_tensor).clamp(min=0, max=1)
    output = F.interpolate(output.unsqueeze(0), input_size, mode='bicubic').squeeze(0)
    output = output.clamp(0, 1).squeeze()  # [H, W]
    return output


def _predict_anydepth_v2(model, input, device):
    """Depth Anything V2 depth prediction."""
    input_size = input.shape[-2:]
    image_size = 518
    trans_totensor = transforms.Compose([
        transforms.Resize(
            size=(image_size, image_size),
            interpolation=cv2.INTER_CUBIC,
        ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = trans_totensor(input).to(device)
    output = model(img_tensor)
    output = F.interpolate(output[:, None], input_size, mode="bilinear", align_corners=True)[0, 0]
    
    # Convert disparity to depth and normalize
    output = (1 / output)
    output = (output - output.min()) / (output.max() - output.min())
    return output


def _predict_anydepth_v3(model, input, cfg, device):
    """
    Depth Anything V3 depth prediction.
    
    DA3 outputs depth directly (not disparity like DA2).
    """
    input_size = input.shape[-2:]  # (H, W)
    
    # Convert tensor to numpy array for DA3 API
    # DA3 expects RGB images in range [0, 255] as uint8
    img_np = input.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    
    # Handle different input normalizations
    if img_np.min() < 0:
        # ImageNet normalized: denormalize first
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
    
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    else:
        img_np = img_np.clip(0, 255).astype(np.uint8)
    
    # Run DA3 inference
    prediction = model.inference(
        image=[img_np],  # DA3 expects a list of images
        export_format=None,  # Don't export, just get prediction
    )
    
    # Get depth map: shape (1, H_out, W_out)
    depth = prediction.depth[0]  # (H_out, W_out)
    
    # Convert to tensor and resize to original input size
    depth_tensor = torch.from_numpy(depth).float().to(device)
    
    if depth_tensor.shape != input_size:
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0).unsqueeze(0),
            size=input_size,
            mode='bilinear',
            align_corners=True
        ).squeeze()
    
    # Normalize to [0, 1] for consistency with other models
    depth_min = depth_tensor.min()
    depth_max = depth_tensor.max()
    if depth_max > depth_min:
        output = (depth_tensor - depth_min) / (depth_max - depth_min)
    else:
        output = depth_tensor
    
    return output


# ============================================================================
# ADVANCED: Multi-view depth prediction for SLAM
# ============================================================================

@torch.no_grad()
def predict_multiview_depth_v3(model, images, cfg, device, extrinsics=None, intrinsics=None):
    """
    Predict spatially consistent depth maps from multiple views using DA3.
    
    This leverages DA3's multi-view capability for consistent depth across frames,
    which is particularly useful for SLAM applications.
    
    Args:
        model: DA3 model (should be DA3-LARGE or DA3-GIANT for best results)
        images: list of tensors [(1, 3, H, W), ...] or tensor (N, 3, H, W)
        cfg: Configuration dictionary
        device: torch device
        extrinsics: optional (N, 4, 4) camera extrinsics (world-to-camera)
        intrinsics: optional (N, 3, 3) camera intrinsics
    
    Returns:
        depths: tensor (N, H, W) - depth maps
        est_poses: tensor (N, 3, 4) - estimated/refined camera poses (w2c)
        est_intrinsics: tensor (N, 3, 3) - estimated/refined intrinsics
    """
    # Convert images to numpy format for DA3
    imgs_np = _convert_images_to_numpy(images)
    
    # Prepare optional camera parameters
    ex_np = extrinsics.cpu().numpy() if extrinsics is not None else None
    in_np = intrinsics.cpu().numpy() if intrinsics is not None else None
    
    # Run DA3 multi-view inference
    prediction = model.inference(
        image=imgs_np,
        extrinsics=ex_np,
        intrinsics=in_np,
        export_format=None,
    )
    
    # Extract results
    depths = torch.from_numpy(prediction.depth).float().to(device)  # (N, H, W)
    est_poses = torch.from_numpy(prediction.extrinsics).float().to(device)  # (N, 3, 4)
    est_intrinsics = torch.from_numpy(prediction.intrinsics).float().to(device)  # (N, 3, 3)
    
    return depths, est_poses, est_intrinsics


def _convert_images_to_numpy(images):
    """Convert various image formats to numpy arrays for DA3."""
    if isinstance(images, torch.Tensor):
        if images.dim() == 4:  # (N, C, H, W)
            imgs_np = []
            for i in range(images.shape[0]):
                img = images[i].permute(1, 2, 0).cpu().numpy()
                # Handle normalization
                if img.min() < 0:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = img * std + mean
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
                imgs_np.append(img)
            return imgs_np
        else:
            raise ValueError(f"Expected 4D tensor, got {images.dim()}D")
    
    elif isinstance(images, list):
        imgs_np = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                if img.min() < 0:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = img * std + mean
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
            imgs_np.append(img)
        return imgs_np
    
    else:
        raise ValueError(f"Unsupported image type: {type(images)}")


# ============================================================================
# Utility: Get metric depth from DA3METRIC-LARGE
# ============================================================================

def get_metric_depth_from_prediction(prediction, focal_length):
    """
    Convert DA3METRIC-LARGE output to metric depth in meters.
    
    For DA3METRIC-LARGE: metric_depth = focal * net_output / 300.
    For DA3NESTED-GIANT-LARGE: output is already in meters.
    
    Args:
        prediction: DA3 prediction object
        focal_length: focal length in pixels (average of fx and fy)
    
    Returns:
        metric_depth: depth in meters
    """
    raw_depth = prediction.depth
    metric_depth = focal_length * raw_depth / 300.0
    return metric_depth