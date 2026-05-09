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

import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov
from PIL import Image
from scipy.spatial.transform import Rotation
from src.utils.tum_inverse_processor import CompleteInverseProcessor

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y

def load_mono_depth(idx,path):
    # omnidata depth
    mono_depth_path = f"{path}/mono_priors/depths/{idx:05d}.npy"
    mono_depth = np.load(mono_depth_path)
    mono_depth_tensor = torch.from_numpy(mono_depth)
    
    return mono_depth_tensor  


def get_dataset(cfg, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, device='cuda:0'):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.n_img = -1
        self.depth_paths = None
        self.color_paths = None
        self.poses = None
        self.image_timestamps = None
        # self.full_size_image = None
        # self.full_size_depth = None
        self.config = cfg

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig = self.fx, self.fy, self.cx, self.cy
        self.H_out, self.W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
        self.H_edge, self.W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']
        
        # 读取skew参数，如果不存在则默认为0
        # self.skew = cfg['cam'].get('skew', 0.0)
        # self.skew_orig = self.skew

        self.H_out_with_edge, self.W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        self.intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()
        # self.intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy, self.skew]).float()
        self.intrinsic[0] *= self.W_out_with_edge / self.W
        self.intrinsic[1] *= self.H_out_with_edge / self.H
        self.intrinsic[2] *= self.W_out_with_edge / self.W
        self.intrinsic[3] *= self.H_out_with_edge / self.H
        # self.intrinsic[4] *= self.W_out_with_edge / self.W  # skew (也需要随x方向缩放)
        self.intrinsic[2] -= self.W_edge
        self.intrinsic[3] -= self.H_edge
        self.fx = self.intrinsic[0].item()
        self.fy = self.intrinsic[1].item()
        self.cx = self.intrinsic[2].item()
        self.cy = self.intrinsic[3].item()
        # self.skew = self.intrinsic[4].item()

        self.fovx = focal2fov(self.fx, self.W_out)
        self.fovy = focal2fov(self.fy, self.H_out)

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None

        # self.interpolation = cfg['interpolation']

        self.num_control_knots = 2
        
        """
        if self.interpolation == "linear":
            self.num_control_knots = 2
        elif self.interpolation == "cubic":
            self.num_control_knots = 4
        """


        # retrieve input folder as temporary folder
        self.input_folder = os.path.join(cfg['data']['dataset_root'], cfg['data']['input_folder'])
        self.only_tracking = self.config['only_tracking']
        # if self.config.get("apply_inverse_gamma", False):
            # self.processor = CompleteInverseProcessor()
        self.is_depth = False
        

    def __len__(self):
        return self.n_img

    def depthloader(self, index, depth_paths, depth_scale):
        if depth_paths is None:
            return None
        depth_path = depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        else:
            raise TypeError(depth_path)
        depth_data = depth_data.astype(np.float32) / depth_scale

        return depth_data

    def get_color(self,index):
        # not used now
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx_orig, self.cx_orig, self.fy_orig, self.cy_orig
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)

        color_data = cv2.resize(color_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]
        self.is_depth 

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
        return color_data

    def get_intrinsic(self):
        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        # tracker不支持多余的内参参数
        # intrinsic = torch.as_tensor([self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig, self.skew_orig]).float()
        intrinsic = torch.as_tensor([self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H
        # intrinsic[4] *= W_out_with_edge / self.W  # skew   
        if self.W_edge > 0:
            intrinsic[2] -= self.W_edge
        if self.H_edge > 0:
            intrinsic[3] -= self.H_edge   
        return intrinsic 

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)

        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx_orig, self.cx_orig, self.fy_orig, self.cy_orig
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)

        
        outsize = (self.H_out_with_edge, self.W_out_with_edge)

        color_data = cv2.resize(color_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        # color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)
        
        
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]
        if not self.is_depth:
            depth_data = None
        else:
            depth_data_fullsize = self.depthloader(index,self.depth_paths,self.png_depth_scale)
            depth_data_fullsize = torch.from_numpy(depth_data_fullsize).float()
            depth_data = F.interpolate(
                depth_data_fullsize[None, None], outsize, mode='nearest')[0, 0]
        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
            if self.is_depth:
                depth_data = depth_data[:, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
            if self.is_depth:
                depth_data = depth_data[edge:-edge, :]


        if self.poses is not None:
            pose = torch.from_numpy(self.poses[index]).float()
        else:
            # Return identity matrix if no poses available
            pose = torch.eye(4).unsqueeze(0).repeat(self.num_control_knots, 1, 1).float()
            print(f"Warning: Using identity pose for frame {index}")

        color_data_fullsize = cv2.cvtColor(color_data_fullsize,cv2.COLOR_BGR2RGB)
        color_data_fullsize = color_data_fullsize / 255.
        color_data_fullsize = torch.from_numpy(color_data_fullsize)

        gt_paths_mid = None

        # Determine GT paths based on dataset configuration
        if self.config["dataset"] == "replica_blurry":
            # middle_index = index * self.averaged_frames + 1
            if self.clear_init and index != 0:
                gt_paths = self.gt_paths[(index - 1) * self.averaged_frames:(index) * self.averaged_frames][::int(self.averaged_frames / self.n_virtual_cams)]
                middle_index = (index - 1) * self.averaged_frames + self.averaged_frames // 2
                gt_paths_mid = self.gt_paths[middle_index]
            else:
                gt_paths = self.gt_paths[index * self.averaged_frames:(index + 1) * self.averaged_frames][::int(self.averaged_frames / self.n_virtual_cams)]
        elif self.config["dataset"] == "tumrgb_ext":
            # middle_index = index * self.averaged_frames + self.averaged_frames // 2
            middle_index = index * self.averaged_frames
            if self.clear_init and index != 0:
                middle_index = (index - 1) * self.averaged_frames + self.averaged_frames // 2
                gt_paths_mid = self.gt_paths[middle_index]
                gt_paths = self.gt_paths[(index - 1) * self.averaged_frames:(index) * self.averaged_frames]
            else:
                gt_paths = self.gt_paths[index * self.averaged_frames:(index + 1) * self.averaged_frames]
        else:
            gt_paths = [self.gt_paths[index]]
        gt_images = []
        gt_img_mid = None
        #print(gt_paths)
        # Handle initial frame if clear_init is True
        # 初始帧是清晰帧的话就没必要用中间帧了
        if index == 0 and self.clear_init:
            init_path = gt_paths[0]
            for i in range(len(gt_paths)):
                gt_img = np.array(Image.open(init_path))
                
                # **Resize GT image to match color image dimensions**
                gt_img = cv2.resize(gt_img, (color_data.shape[3], color_data.shape[2]), interpolation=cv2.INTER_LINEAR)
                
                gt_img = (
                    torch.from_numpy(gt_img / 255.0)
                    .clamp(0.0, 1.0)
                    .permute(2, 0, 1)
                    .to(device=self.device, dtype = torch.float32)
                )
                gt_images.insert(0, gt_img)
        else:            
            #print(gt_paths)
            if gt_paths_mid is not None and self.config["fake_sharp_gt"]:
                gt_img_mid_fullsize = cv2.imread(gt_paths_mid)
                if self.distortion is not None:
                    K = np.eye(3)
                    K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx_orig, self.cx_orig, self.fy_orig, self.cy_orig
                    # undistortion is only applied on color image, not depth!
                    gt_img_mid_fullsize = cv2.undistort(gt_img_mid_fullsize, K, self.distortion)
                outsize = (self.H_out_with_edge, self.W_out_with_edge)
                gt_img_mid = cv2.resize(gt_img_mid_fullsize, (self.W_out_with_edge, self.H_out_with_edge))
                gt_img_mid = torch.from_numpy(gt_img_mid).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
                gt_img_mid = gt_img_mid.unsqueeze(dim=0)
                
                if self.W_edge > 0:
                    edge = self.W_edge
                    gt_img_mid = gt_img_mid[:, :, :, edge:-edge]
                    if self.is_depth:
                        depth_data = depth_data[:, edge:-edge]

                if self.H_edge > 0:
                    edge = self.H_edge
                    gt_img_mid = gt_img_mid[:, :, edge:-edge, :]
                    if self.is_depth:
                        depth_data = depth_data[edge:-edge, :]
            for i in range(len(gt_paths)):
                if len(gt_paths) == 1:
                    gt_img = np.array(Image.open(gt_paths[0]))
                else:
                    gt_img = np.array(Image.open(gt_paths[i]))
                
                # **Resize GT image to match color image dimensions**
                gt_img = cv2.resize(gt_img, (color_data.shape[3], color_data.shape[2]), interpolation=cv2.INTER_LINEAR)
                gt_img = (
                    torch.from_numpy(gt_img / 255.0)
                    .clamp(0.0, 1.0)
                    .permute(2, 0, 1)
                    .to(device=self.device, dtype=torch.float32)
                )
                gt_images.append(gt_img)
            if gt_paths_mid is not None and self.config["fake_sharp_gt"]:
                return index, color_data, depth_data, pose, gt_images, gt_img_mid
        return index, color_data, depth_data, pose, gt_images


class Replica(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(Replica, self).__init__(cfg, device)
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)

        self.load_poses(f'{self.input_folder}/traj.txt')
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        self.n_img = len(self.color_paths)


    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            self.poses.append(c2w)

class ReplicaBlurry(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(ReplicaBlurry, self).__init__(cfg, device)
        stride = cfg['stride']
        self.config = cfg
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)

        self.averaged_frames = cfg['averaged_frames']
        self.clear_init = cfg['clear_init']
        added_frames = cfg['added_frames']
        self.n_virtual_cams = cfg['n_virtual_cams']
        
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/blur_{self.averaged_frames}/rgb_*.png"),
            key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.gt_paths = sorted(glob.glob(f"{self.input_folder}/results/rgb/rgb_*.png"), 
            key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.depth_paths = None
        self.n_img = len(self.color_paths)
        """
        if self.only_tracking:
            self.num_control_knots = 1
            self.load_poses(f"{self.input_folder}/blur_{self.averaged_frames}/mid_traj.txt")
        else:
        """
        self.load_poses(f"{self.input_folder}/blur_{self.averaged_frames}/traj.txt")
        self.color_paths = self.color_paths[:max_frames][::stride]
        #self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        #self.w2c_first_pose = np.linalg.inv(self.poses[0])

        self.n_img = len(self.color_paths)

        # 这是初始化pose的逻辑代码
        # clear_init在原有的数据集序列上加了一个锐利的视频帧作为初始化，所以一切的index都必须是index-1才能还原数据集中的序列
        if self.clear_init and self.averaged_frames:
            print("Adding clear frame")
            first_sharp = sorted(glob.glob(f"{self.input_folder}/results/rgb/rgb_*.png"), key=lambda x: int(x.split("_")[-1].split(".")[0]))[0]
            print(first_sharp)
            self.color_paths.insert(0,first_sharp)
            self.n_img = self.n_img + 1
            with open(f"{self.input_folder}/traj_{added_frames}.txt", "r") as f:
                line = f.readline()
                pose = np.array(list(map(float, line.split()))).reshape(4, 4)
                #one for each knot
                current_poses = np.zeros((self.num_control_knots,4,4))
                for i in range(self.num_control_knots):
                    current_poses[i] = pose
                self.poses.insert(0, current_poses)
        
        print(len(self.color_paths), len(self.poses))
        #check loading all gt_paths
        #for i in range(len(self.gt_paths)):
        #    print(self.gt_paths[i])
        #    gt_img = np.array(Image.open(self.gt_paths[i]))


    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            """
            if self.only_tracking:
                # 使用 mid_traj 时，只有一个控制点
                line = lines[i]
                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                current_poses = np.array([c2w])  # 形状为 (1, 4, 4)
            else:
            """
            current_poses = np.zeros((self.num_control_knots,4,4))
            #take only start and finish
            for j in range(self.num_control_knots):
                line = lines[self.num_control_knots*i+j]
                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                # c2w[:3, 1] *= -1
                # c2w[:3, 2] *= -1
                current_poses[j] = c2w
            self.poses.append(current_poses)


class ScanNet(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(ScanNet, self).__init__(cfg, device)
        self.config = cfg
        self.averaged_frames = cfg['averaged_frames']
        self.clear_init = cfg['clear_init']
        added_frames = cfg['added_frames']
        self.n_virtual_cams = cfg['n_virtual_cams']
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))[:max_frames][::stride]
        
        self.gt_paths = self.color_paths

        #self.depth_paths = sorted(glob.glob(os.path.join(
        #    self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))[:max_frames][::stride]
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.poses = self.poses[:max_frames][::stride]

        self.n_img = len(self.color_paths)
        print("INFO: {} images got!".format(self.n_img))

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w = np.expand_dims(c2w, axis=0)
            #make it knotsx4x4
            c2w = np.repeat(c2w, self.num_control_knots, axis=0)
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)

        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            # timestamp tx ty tz qx qy qz qw
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w

            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

class TUM_RGB_EXT(BaseDataset):
    def __init__(self, cfg, device='cuda:0', clear_init=False):
        super(TUM_RGB_EXT, self).__init__(cfg, device)
        self.clear_init = clear_init
        self.gt_input_folder = cfg['data']['gt_dataset_root']
        self.averaged_frames = cfg['averaged_frames']
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        self.gt_paths = sorted(glob.glob(f"{self.gt_input_folder}/rgb/*.png"))
        self.n_img = len(self.color_paths)

        self.load_poses(f"{self.input_folder}/traj.txt")
        self.n_virtual_cams = cfg['n_virtual_cams']
        self.config = cfg
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)

        self.color_paths = self.color_paths[:max_frames][::stride] 
        self.poses = self.poses[:max_frames][::stride]

        print("INFO: {} images got!".format(len(self.color_paths)))
        print("INFO: {} poses got!".format(len(self.poses)))
        print("INFO: {} gt images got!".format(len(self.gt_paths)))
    
    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            current_poses = np.zeros((2,4,4))
            for j in range(2):
                line = lines[2*i+j]
                pose = np.array(list(map(float, line.split()))).reshape(4, 4)
                pose = np.linalg.inv(pose)
                current_poses[j] = pose

            self.poses.append(current_poses)
            frame = {
                "file_path": self.color_paths[i],
                #"depth_path": self.depth_paths[i],
                "transform_matrix": current_poses.tolist(),
            }
            frames.append(frame)
        self.frames = frames


class TUM_RGB(BaseDataset):
    def __init__(self, cfg, device='cuda:0', clear_init=False):
        super(TUM_RGB, self).__init__(cfg, device)
        self.clear_init = clear_init
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_virtual_cams = cfg['n_virtual_cams']
        self.config = cfg
        
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)

        self.color_paths = self.color_paths[:max_frames][::stride] 
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.is_depth = True
        self.poses = self.poses[:max_frames][::stride]

        self.w2c_first_pose = np.linalg.inv(self.poses[0][0])

        self.n_img = len(self.color_paths)
        self.gt_paths = self.color_paths
        self.gamma_processor = CompleteInverseProcessor()
    
    def parse_list(self, filepath, skiprows=0):
        """Read list data from a file."""
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data
    
    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """Read video data in TUM-RGBD format, associating images with poses."""
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')
        else:
            raise FileNotFoundError("Pose file not found.")

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        # Apply frame rate constraints
        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies.append(i)

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            if tstamp_pose is None:
                (i,j) = associations[ix]
                k = None
            else:
                (i, j, k) = associations[ix]
            image_path = os.path.join(datapath, image_data[i, 1])
            depth_path = os.path.join(datapath, depth_data[j, 1])
            images.append(image_path)
            depths.append(depth_path)
            
            num_poses = len(pose_vecs)

            # Get start and end poses
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])

            # Compute relative poses
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w

            #make it (self.num_control_knots)x4x4
            final_pose = np.repeat(c2w[None, :, :], self.num_control_knots, axis=0)
            poses.append(final_pose)

        # Convert list of poses to a NumPy array
        poses = np.array(poses)
        print(f"Loaded {len(images)} images and {len(poses)} poses.")
        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """Convert a pose vector (translation + quaternion) to a 4x4 pose matrix."""
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

class ETH3D(BaseDataset):
    def __init__(self, cfg, device='cuda:0'
                 ):
        super(ETH3D, self).__init__(cfg, device)
        stride = cfg['stride']
        self.color_paths, self.depth_paths, self.poses, self.image_timestamps = self.loadtum(
            self.input_folder, frame_rate=-1)
        self.color_paths = self.color_paths[::stride]
        self.depth_paths = self.depth_paths[::stride]
        self.poses = None if self.poses is None else self.poses[::stride]
        self.image_timestamps = self.image_timestamps[::stride]
        self.clear_init = cfg['clear_init']
        self.gt_paths = self.color_paths
        self.config = cfg

        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                # we need all images for benchmark, no max_dt checking here
                # if (np.abs(tstamp_depth[j] - t) < max_dt):
                associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')
        else:
            pose_list = None

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)

        if pose_list is not None:
            pose_data = self.parse_list(pose_list, skiprows=1)
            pose_vecs = pose_data[:, 1:].astype(np.float64)

            tstamp_pose = pose_data[:, 0].astype(np.float64)
        else:
            tstamp_pose = None
            pose_vecs = None

        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        images, poses, depths, timestamps = [], [], [], tstamp_image
        if pose_list is not None:
            inv_pose = None
            for ix in range(len(associations)):
                (i, j, k) = associations[ix]
                images += [os.path.join(datapath, image_data[i, 1])]
                depths += [os.path.join(datapath, depth_data[j, 1])]
                # timestamp tx ty tz qx qy qz qw
                c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
                if inv_pose is None:
                    inv_pose = np.linalg.inv(c2w)
                    c2w = np.eye(4)
                else:
                    c2w = inv_pose@c2w
                c2w = np.expand_dims(c2w, axis=0)
                c2w = np.repeat(c2w, 2, axis=0)
                poses += [c2w]
        else:
            assert len(associations) == len(tstamp_image), 'Not all images are loaded. While benchmark need all images\' pose!'
            print('\nDataset: no gt pose avaliable, {} images found\n'.format(len(tstamp_image)))
            for ix in range(len(associations)):
                (i, j) = associations[ix]
                images += [os.path.join(datapath, image_data[i, 1])]
                depths += [os.path.join(datapath, depth_data[j, 1])]

            poses = None

        return images, depths, poses, timestamps, gt_images

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

class ETH3D_EXT(BaseDataset):

    def associate_frames(self, tstamp_image, tstamp_pose, max_dt=0.02):
        associations = []
        for i, t in enumerate(tstamp_image):
            k = np.argmin(np.abs(tstamp_pose - t))
            if abs(tstamp_pose[k] - t) < max_dt:
                associations.append((i, k))
        return associations

    def __init__(self, cfg, device='cuda:0', clear_init=False):
        super(ETH3D_EXT, self).__init__(cfg, device)
        self.clear_init = clear_init

        # Root where ground-truth images live
        self.gt_input_folder = cfg['data']['gt_dataset_root']

        # The factor by which frames were averaged
        self.averaged_frames = cfg['averaged_frames']

        # Paths to the images used for this dataset
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/*.png"))

        # Ground-truth images
        #self.gt_paths = sorted(glob.glob(f"{self.gt_input_folder}/*.png"))
        timestamps_file = os.path.join(self.gt_input_folder, "rgb.txt")
        groundtruth_file = os.path.join(self.gt_input_folder, "groundtruth.txt")
        
        # Load timestamps
        timestamps_data = np.genfromtxt(timestamps_file, delimiter=" ", dtype=str)
        frame_stamps = timestamps_data[:, 0].astype(np.float64)
        frame_files = timestamps_data[:, 1]

        # Load ground truth for poses: [timestamp, px, py, pz, qx, qy, qz, qw]
        gt_data = np.loadtxt(groundtruth_file, delimiter=" ", dtype=np.unicode_, skiprows=0)
        gt_stamps = gt_data[:, 0].astype(np.float64)
        associations = self.associate_frames(frame_stamps, gt_stamps, 0.05)

        self.gt_paths = []
        for i in range(len(associations)):
            index, _ = associations[i]
            self.gt_paths.append(os.path.join(self.gt_input_folder, frame_files[index]))


        self.n_img = len(self.color_paths)

        print(self.input_folder, self.gt_input_folder)

        # Load the poses from a text file that presumably has 2 lines per image
        traj_path = os.path.join(self.input_folder, "traj_mat.txt")
        print("Loading poses from: ", traj_path)
        self.load_poses(traj_path)

        # Additional config
        self.n_virtual_cams = cfg['n_virtual_cams']
        self.config = cfg
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)

        # Subsample the color_paths and poses if needed
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        print("INFO: {} images got!".format(len(self.color_paths)))
        print("INFO: {} poses got!".format(len(self.poses)))
        print("INFO: {} gt images got!".format(len(self.gt_paths)))

    def load_poses(self, path):
        """
        Reads a trajectory file with 2 lines per image:
        e.g., for i-th image, lines [2*i, 2*i+1].
        Each line is a flattened 4x4 transform, which we invert
        if we want camera->world or world->camera, etc.
        """

        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        # We'll parse 2 lines for each of the self.n_img images
        # (start pose and end pose, for the same blurred chunk).
        for i in range(self.n_img):
            current_poses = np.zeros((2, 4, 4), dtype=np.float64)

            # For each image i, we have lines [2*i, 2*i+1]
            for j in range(2):
                line_idx = 2 * i + j
                line = lines[2*i+j]
                pose = np.array(list(map(float, line.split()[1:]))).reshape(4, 4)

                # We invert if we want camera->world; remove if not needed
                pose = np.linalg.inv(pose)

                current_poses[j] = pose

            self.poses.append(current_poses)

            # Book-keeping if you need it
            frame_dict = {
                "file_path": self.color_paths[i],
                "transform_matrix": current_poses.tolist(),
            }
            frames.append(frame_dict)

        self.frames = frames

class IndoorMCD(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(IndoorMCD, self).__init__(cfg, device)
        
        self.color_paths, self.poses = self.load_indoormcd(
            self.input_folder, frame_rate=32)
        
        self.n_virtual_cams = cfg['n_virtual_cams']
        self.config = cfg
        self.clear_init = cfg.get('clear_init', False)
        
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)

        self.color_paths = self.color_paths[:max_frames][::stride] 
        self.poses = self.poses[:max_frames][::stride]

        self.w2c_first_pose = np.linalg.inv(self.poses[0][0])

        self.n_img = len(self.color_paths)
        self.gt_paths = self.color_paths
        
        print(f"INFO: Loaded {self.n_img} images from IndoorMCD dataset")
    
    def parse_list(self, filepath, skiprows=0):
        """Read list data from a file."""
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_pose, max_dt=0.05):
        """Associate images with poses based on timestamps."""
        associations = []
        for i, t in enumerate(tstamp_image):
            k = np.argmin(np.abs(tstamp_pose - t))
            if np.abs(tstamp_pose[k] - t) < max_dt:
                associations.append((i, k))
            else:
                print(f"Warning: Image {i} timestamp {t} has no close pose match (closest: {tstamp_pose[k]}, diff: {abs(tstamp_pose[k] - t)})")
        return associations

    def load_indoormcd(self, datapath, frame_rate=-1):
        """Load IndoorMCD dataset with nanosecond timestamp images and gt.txt poses."""
        
        # Load ground truth poses (same format as TUM: timestamp tx ty tz qx qy qz qw)
        gt_file = os.path.join(datapath, 'gt.txt')
        if not os.path.isfile(gt_file):
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        
        # Parse gt.txt (skip comment lines starting with #)
        pose_data = self.parse_list(gt_file, skiprows=2)  # Skip the header comments
        pose_vecs = pose_data[:, 1:].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)  # Already in seconds
        
        # Load RGB images from front/rgb/
        rgb_folder = os.path.join(datapath, 'front', 'rgb')
        if not os.path.isdir(rgb_folder):
            raise FileNotFoundError(f"RGB folder not found: {rgb_folder}")
        
        image_files = sorted(glob.glob(os.path.join(rgb_folder, '*.png')))
        
        # Extract timestamps from image filenames (nanoseconds -> seconds)
        tstamp_image = []
        for img_path in image_files:
            filename = os.path.basename(img_path)
            timestamp_ns = int(filename.split('.')[0])  # e.g., 1643272782212801489
            timestamp_s = timestamp_ns / 1e9  # Convert to seconds
            tstamp_image.append(timestamp_s)
        
        tstamp_image = np.array(tstamp_image)
        
        # Associate images with poses
        associations = self.associate_frames(tstamp_image, tstamp_pose)
        
        # Apply frame rate constraints
        indices = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indices[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indices.append(i)

        images, poses = [], []
        inv_pose = None
        
        for ix in indices:
            (i, k) = associations[ix]
            image_path = image_files[i]
            images.append(image_path)
            
            # Convert pose vector to 4x4 matrix
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])

            # Compute relative poses (relative to first frame)
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w

            # Repeat for num_control_knots (matching TUM_RGB structure)
            final_pose = np.repeat(c2w[None, :, :], self.num_control_knots, axis=0)
            poses.append(final_pose)

        poses = np.array(poses)
        print(f"Loaded {len(images)} images and {len(poses)} poses from IndoorMCD.")
        return images, poses

    def pose_matrix_from_quaternion(self, pvec):
        """Convert a pose vector (translation + quaternion) to a 4x4 pose matrix."""
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

class Archviz(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(Archviz, self).__init__(cfg, device)
        
        stride = cfg['stride']
        self.config = cfg
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)
            
        self.averaged_frames = cfg['averaged_frames']
        self.clear_init = cfg['clear_init']
        added_frames = cfg['added_frames']
        self.n_virtual_cams = cfg['n_virtual_cams']
        
        # 确定使用哪个场景 (archviz1, archviz2, 或 archviz3)
        scene = cfg.get('scene', 'archviz1')  # 可以在config中指定scene
        scene_folder = os.path.join(self.input_folder, scene)
        
        # 模糊图像路径 (results文件夹)
        self.color_paths = sorted(
            glob.glob(f"{scene_folder}/results/frame*.jpg"),
            key=lambda x: int(x.split("frame")[-1].split(".")[0])
        )
        
        # 锐利图像路径作为GT (results_sharp文件夹)
        self.gt_paths = sorted(
            glob.glob(f"{scene_folder}/results_sharp/frame*.jpg"),
            key=lambda x: int(x.split("frame")[-1].split(".")[0])
        )
        
        self.depth_paths = None  # 不需要深度图
        self.n_img = len(self.color_paths)
        
        # 加载poses
        self.load_poses(f"{scene_folder}/groundtruth_synced.txt")
        
        # 应用stride和max_frames
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.n_img = len(self.color_paths)
        
        # 处理clear_init逻辑（添加第一个锐利帧作为初始化）
        if self.clear_init and self.averaged_frames:
            print("Adding clear frame for Archviz dataset")
            first_sharp = sorted(
                glob.glob(f"{scene_folder}/results_sharp/frame*.jpg"),
                key=lambda x: int(x.split("frame")[-1].split(".")[0])
            )[0]
            print(f"First sharp frame: {first_sharp}")
            self.color_paths.insert(0, first_sharp)
            self.n_img = self.n_img + 1
            
            # 为初始帧添加pose（与replica_blurry保持一致）
            with open(f"{scene_folder}/groundtruth_synced.txt", "r") as f:
                lines = [l for l in f.readlines() if not l.startswith('#')]
                line = lines[0]  # 使用第一个pose
                pose = self.parse_pose_line(line)
                current_poses = np.zeros((self.num_control_knots, 4, 4))
                for i in range(self.num_control_knots):
                    current_poses[i] = pose
                self.poses.insert(0, current_poses)
            
        print(f"Archviz dataset - Scene: {scene}")
        print(f"Loaded {len(self.color_paths)} images, {len(self.poses)} poses")
        
    def parse_pose_line(self, line):
        """
        解析单行pose数据
        格式: timestamp tx ty tz qx qy qz qw
        """
        parts = line.strip().split()
        timestamp = float(parts[0])
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        
        # 从四元数创建4x4变换矩阵
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_quat([qx, qy, qz, qw])
        c2w = np.eye(4)
        c2w[:3, :3] = rotation.as_matrix()
        c2w[:3, 3] = [tx, ty, tz]
        
        return c2w
        
    def load_poses(self, path):
        """
        加载groundtruth_synced.txt文件中的poses
        与replica_blurry保持一致的风格
        """
        self.poses = []
        
        with open(path, "r") as f:
            lines = f.readlines()
            
        # 过滤注释行
        lines = [line for line in lines if not line.startswith('#')]
        
        for i in range(self.n_img):
            if i < len(lines):
                # 为了保持与replica_blurry的一致性，每个图像创建num_control_knots个控制点
                # 如果groundtruth只有一个pose序列，我们复制相同的pose
                current_poses = np.zeros((self.num_control_knots, 4, 4))
                
                # 读取基础pose
                c2w = self.parse_pose_line(lines[i])
                
                # 为每个控制点设置相同的pose（或根据需要进行插值）
                for j in range(self.num_control_knots):
                    current_poses[j] = c2w
                    
                self.poses.append(current_poses)
            else:
                # 如果poses不够，使用最后一个pose
                if self.poses:
                    self.poses.append(self.poses[-1].copy())
                else:
                    # 创建单位矩阵作为默认pose
                    current_poses = np.zeros((self.num_control_knots, 4, 4))
                    for j in range(self.num_control_knots):
                        current_poses[j] = np.eye(4)
                    self.poses.append(current_poses)
class Deblur_nerf(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(Deblur_nerf, self).__init__(cfg, device)
        
        stride = cfg['stride']
        self.config = cfg
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = int(1e5)
            
        self.averaged_frames = cfg['averaged_frames']
        self.clear_init = cfg['clear_init']
        added_frames = cfg['added_frames']
        self.n_virtual_cams = cfg['n_virtual_cams']
        
        # 确定使用哪个场景 (blurball, blurbasket, 或 blurbuick)
        scene = cfg.get('scene', 'blurball')  # 可以在config中指定scene
        scene_folder = os.path.join(self.input_folder, scene)
        
        # 图像路径 (images_4文件夹)
        if cfg['dataset'] == 'exblurf_motion':
            image_folder = os.path.join(scene_folder, 'images_1')
        else:
            image_folder = os.path.join(scene_folder, 'images_4')
        self.color_paths = sorted(
            glob.glob(f"{image_folder}/*.png") + glob.glob(f"{image_folder}/*.jpg"),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        
        # 这个数据集没有锐利的真实值，GT也指向相同的模糊图像
        self.gt_paths = self.color_paths.copy()
        
        self.depth_paths = None  # 不需要深度图
        self.n_img = len(self.color_paths)
        
        # 加载LLFF格式的poses
        poses_path = os.path.join(scene_folder, 'poses_bounds.npy')
        self.load_poses(poses_path)
        
        # 应用stride和max_frames
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.gt_paths = self.gt_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]
        self.n_img = len(self.color_paths)
        
        # 处理clear_init逻辑（如果需要的话，添加第一帧作为初始化）
        if self.clear_init and self.averaged_frames:
            print("Adding clear frame for Deblur_nerf dataset")
            first_frame = self.color_paths[0]
            print(f"First frame: {first_frame}")
            self.color_paths.insert(0, first_frame)
            self.gt_paths.insert(0, first_frame)
            self.n_img = self.n_img + 1
            
            # 为初始帧添加pose（复制第一个pose）
            current_poses = np.zeros((self.num_control_knots, 4, 4))
            for i in range(self.num_control_knots):
                current_poses[i] = self.poses[0][0].copy()
            self.poses.insert(0, current_poses)
            
        print(f"Deblur_nerf dataset - Scene: {scene}")
        print(f"Loaded {len(self.color_paths)} images, {len(self.poses)} poses")
        
    def parse_llff_pose(self, pose_data):
        """
        解析LLFF格式的pose数据
        输入: (17,) 数组
        - 前15个元素: 3x5矩阵 [R|t|hwf] (flatten)
        - 最后2个元素: [near, far]
        
        输出: 4x4 c2w变换矩阵
        """
        # 提取前15个元素并reshape为3x5
        pose_matrix = pose_data[:15].reshape(3, 5)
        
        # 提取旋转矩阵R (3x3)
        R = pose_matrix[:, :3]
        
        # 提取平移向量t (3x1)
        t = pose_matrix[:, 3]
        
        # 提取相机内参 hwf (height, width, focal)
        hwf = pose_matrix[:, 4]
        
        # 构建4x4 c2w矩阵
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        
        return c2w
        
    def load_poses(self, path):
        """
        加载poses_bounds.npy文件中的poses
        格式: (N, 17) - LLFF/NeRF格式
        """
        self.poses = []

        # Check if poses file exists
        if not os.path.exists(path):
            print(f"Warning: Poses file not found at {path}")
            print("Creating identity poses for all images (no ground truth available)")
            # Create identity poses for all images
            for i in range(self.n_img):
                current_poses = np.zeros((self.num_control_knots, 4, 4))
                for j in range(self.num_control_knots):
                    current_poses[j] = np.eye(4)
                self.poses.append(current_poses)
            return
        
        # 加载npy文件
        poses_bounds = np.load(path)  # Shape: (N, 17)
        n_poses = poses_bounds.shape[0]
        
        print(f"Loaded poses_bounds.npy with shape: {poses_bounds.shape}")
        
        # 检查pose数量是否匹配图像数量
        if n_poses != self.n_img:
            print(f"Warning: Number of poses ({n_poses}) doesn't match number of images ({self.n_img})")
            # 使用最小值
            n_poses = min(n_poses, self.n_img)
        
        for i in range(n_poses):
            # 为了保持与Archviz的一致性，每个图像创建num_control_knots个控制点
            current_poses = np.zeros((self.num_control_knots, 4, 4))
            
            # 解析LLFF格式的pose
            c2w = self.parse_llff_pose(poses_bounds[i])
            
            # 为每个控制点设置相同的pose
            # 如果需要运动模糊建模，这里可以添加轻微的插值
            for j in range(self.num_control_knots):
                current_poses[j] = c2w.copy()
                
            self.poses.append(current_poses)
        
        # 如果pose数量不够，用最后一个pose填充
        while len(self.poses) < self.n_img:
            if self.poses:
                self.poses.append(self.poses[-1].copy())
            else:
                # 创建单位矩阵作为默认pose
                current_poses = np.zeros((self.num_control_knots, 4, 4))
                for j in range(self.num_control_knots):
                    current_poses[j] = np.eye(4)
                self.poses.append(current_poses)

dataset_dict = {
    "replica": Replica,
    "replica_blurry": ReplicaBlurry,
    "scannet": ScanNet,
    "tumrgbd": TUM_RGB,
    "tumrgb": TUM_RGB,
    "tumrgb_ext": TUM_RGB_EXT,
    "eth3d": ETH3D,
    "eth3d_ext": ETH3D_EXT,
    "mcd": IndoorMCD,
    "archviz": Archviz,
    "deblur_nerf_motion": Deblur_nerf,
    "deblur_nerf_motion_no_deblur_no_refine": Deblur_nerf,
    "deblur_nerf_motion_whole": Deblur_nerf,
    "deblur_nerf_defocus": Deblur_nerf,
    "exblurf_motion": Deblur_nerf,
}
