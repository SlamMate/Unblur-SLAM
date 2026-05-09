# Copyright 2024 The MonoGS Authors.

# Licensed under the License issued by the MonoGS Authors
# available here: https://github.com/muskie82/MonoGS/blob/main/LICENSE.md
from thirdparty.monogs.utils.pose_utils import slerp, SO3_exp
import torch
from torch import nn

from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from thirdparty.monogs.utils.slam_utils import image_gradient, image_gradient_mask
from thirdparty.monogs.utils.rotation_conv import *
from thirdparty.monogs.utils.Spline import *

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        n_virtual_cams = 9,
        interpolation = "linear",
        device="cuda:0",
        realgt_pose=None,
        deblur_fail = False
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        # Absolute pose as W2C
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        # Note that we only optimize the delta pose to the 
        # previous pose. We keep track of the absolute pose
        # in the self.T and self.R variables.
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        
        if deblur_fail:
            self.n_virtual_cams = n_virtual_cams
            self.interpolation = interpolation

            if self.interpolation == "linear":
                self.num_control_knots = 2 
            elif self.interpolation == "cubic":
                self.num_control_knots = 4
            else:
                raise NotImplementedError
            T = torch.eye(4, device=device)
            # Absolute pose as W2C
            self.R_i = torch.zeros((self.num_control_knots, 3, 3), device=device)
            self.t_i = torch.zeros((self.num_control_knots, 3), device=device)
            self.R_gt_motion = torch.zeros((self.num_control_knots, 3, 3), device=device)
            self.T_gt_motion = torch.zeros((self.num_control_knots, 3), device=device)
            # self.R_realgt = torch.zeros((self.num_control_knots, 3, 3), device=device)
            # self.T_realgt = torch.zeros((self.num_control_knots, 3), device=device)

            for i in range(self.num_control_knots):
                self.R_i[i] = T[:3, :3]
                self.t_i[i] = T[:3, 3]
                
                self.R_gt_motion[i] = gt_T[:3, :3]
                self.T_gt_motion[i] = gt_T[:3, 3]

                #check out dimensions of realgt_pose
                if realgt_pose.shape[0] != self.num_control_knots:
                    #expand
                    realgt_pose = realgt_pose.expand(self.num_control_knots, -1, -1)

                # add actual GT poses for replicaglorieslam
                #if realgt_pose is not None:
                    #self.R_realgt[i] = realgt_pose[i, :3, :3]
                    #self.T_realgt[i] = realgt_pose[i, :3, 3]

            # Note that we only optimize the delta pose to the 
            # previous pose. We keep track of the absolute pose
            # in the self.T and self.R variables.
            #sub-frame trajectory
            self.T_i_rot_delta = []
            self.T_i_trans_delta = []

            for i in range(self.num_control_knots):
                self.T_i_rot_delta.append(nn.Parameter(
                    torch.zeros(3, requires_grad=True, device=device)
                ))
                self.T_i_trans_delta.append(nn.Parameter(
                    torch.zeros(3, requires_grad=True, device=device)
                ))

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        # normal variable
        self.projection_matrix = projection_matrix.to(device=device)

        self.is_blurry = True
        # 添加新属性
        self.deblur_fail = deblur_fail
        # 存序号来随时随地加载图片，而不将其存在GPU内存里，节省空间
        self.timestamp = None
        self.is_valid = True


    @staticmethod
    def init_from_dataset(dataset, data, projection_matrix):
        
        return Camera(
            data["idx"],
            data["gt_color"],
            data["glorie_depth"], # depth as in GlORIE-SLAM
            data["glorie_pose"], # pose as in GlORIE-SLAM
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.H_out,
            dataset.W_out,
            device=dataset.device,
        )
        
    @staticmethod
    def init_from_dataset_motion(dataset, data, projection_matrix, deblur_fail):
        
        return Camera(
            data["idx"],
            data["gt_color"],
            data["glorie_depth"], # depth as in GlORIE-SLAM
            data["glorie_pose"], # pose as in GlORIE-SLAM
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.H_out,
            dataset.W_out,
            data["n_virtual_cams"],
            data["interpolation"],
            device=dataset.device,
            realgt_pose=data["gt_pose"],
            deblur_fail = deblur_fail,
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)
    
    @property
    def world_view_transform_motion(self):
        R, t, _, _ = self.get_mid_extrinsic()
        return getWorld2View2(R, t).transpose(0, 1)
    
    def world_view_transform_custom(self, R, t):
        return getWorld2View2(R, t).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def full_proj_transform_motion(self):
        return (
            self.world_view_transform_motion.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
    
    def full_proj_transform_custom(self, R, t):
        return (
            self.world_view_transform_custom(R,t).unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]
    
    @property
    def camera_center_motion(self):
        return  self.world_view_transform_motion().inverse()[3, :3]

    def camera_center_custom(self, R, t):
        return  self.world_view_transform_custom(R,t).inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def update_RT_motion(self, R, t, index):
        self.R_i[index] = R.to(device=self.device)
        self.t_i[index] = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["mapping"]["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        row, col = 32, 32
        multiplier = edge_threshold
        _, h, w = self.original_image.shape
        for r in range(row):
            for c in range(col):
                block = img_grad_intensity[
                    :,
                    r * int(h / row) : (r + 1) * int(h / row),
                    c * int(w / col) : (c + 1) * int(w / col),
                ]
                th_median = block.median()
                block[block > (th_median * multiplier)] = 1
                block[block <= (th_median * multiplier)] = 0
        self.grad_mask = img_grad_intensity

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None

    def interpolate_linear(
        self,
        u: torch.Tensor,
        R_i,
        t_i,
        T_i_rot_delta,
        T_i_trans_delta,
        device,
        return_gradients: bool = True
    ):
        """
        Perform linear interpolation on rotation and translation.

        Parameters
        ----------
        u : torch.Tensor
            A scalar or 1D array of interpolation parameters in [0, 1].
        R_i : list of torch.Tensor
            A list (or tuple) of base rotation matrices. For linear we expect something like [R0, R1].
        t_i : list of torch.Tensor
            A list (or tuple) of base translations [t0, t1].
        T_i_rot_delta : list of torch.Tensor
            The rotation deltas [delta_R0, delta_R1]. If you need these for advanced usage.
        T_i_trans_delta : list of torch.Tensor
            The translation deltas [delta_t0, delta_t1].
        device : torch.device
            Where computations should happen.
        return_gradients : bool
            If False, detach the final outputs from the computation graph.
        
        Returns
        -------
        R : torch.Tensor
            Interpolated rotation(s) as a batch or single matrix.
        t : torch.Tensor
            Interpolated translation(s).
        theta : torch.Tensor or None
            Axis-angle representation of the rotation (if return_gradients=True).
        rho : torch.Tensor or None
            Interpolated translation delta (if return_gradients=True).
        """

        # Enforce tensor shape. For a single index, you might pass a scalar u;
        # for multiple indices, you might pass a 1D tensor of shape [N, 1].
        if u.dim() == 0:
            u = u.unsqueeze(-1)  # [1,]

        if R_i is None and t_i is None and T_i_rot_delta is None and T_i_trans_delta is None:
            R_i = self.R_i
            t_i = self.t_i
            T_i_rot_delta = self.T_i_rot_delta
            T_i_trans_delta = self.T_i_trans_delta

        # Grab base rotations from R_i plus exponential map if needed
        # For linear, you might do an SO3_exp on T_i_rot_delta if that’s your pipeline:
        R0 = SO3_exp(T_i_rot_delta[0]) @ R_i[0]
        R1 = SO3_exp(T_i_rot_delta[1]) @ R_i[1]
        # Convert to quaternions
        q0 = matrix_to_quaternion(R0).to(device)
        q1 = matrix_to_quaternion(R1).to(device)

        # Interpolate rotation via slerp
        # shape of q_sl will match u: if u has shape [N, 1], slerp can return [N, 4].
        q_sl = slerp(u, q0, q1)
        #print("q0", q0.shape, "q1", q1.shape, "u", u.shape, "q_sl", q_sl.shape)

        # Convert slerped quaternions back to rotation matrices
        R_out = quaternion_to_matrix(q_sl)
        if not return_gradients:
            R_out = R_out.detach()

        # Interpolate translation linearly - 包含 T_i_trans_delta 以支持梯度传播
        # 基础位姿插值 + delta插值
        t_base = t_i[0] * (1 - u) + t_i[1] * u
        t_delta = T_i_trans_delta[0] * (1 - u) + T_i_trans_delta[1] * u
        t_out = t_base + t_delta  # 将 delta 加到基础位姿上

        # If needed, also compute the axis-angle and the rho translation deltas
        if return_gradients:
            theta = quaternion_to_axis_angle(q_sl)
            rho = t_delta  # rho 就是 t_delta
        else:
            theta = None
            rho = None

        #print("theta", theta.shape, "rho", rho.shape)

        return R_out, t_out, theta, rho


    def interpolate_cubic_spline(
        self,
        u: torch.Tensor,
        R_i,
        t_i,
        T_i_rot_delta,
        T_i_trans_delta,
        device,
        return_gradients: bool = True
    ):
        """
        Perform cubic spline interpolation on rotation and translation.

        Parameters
        ----------
        u : torch.Tensor
            A scalar or 1D array of interpolation parameters in [0, 1].
        R_i : list of torch.Tensor
            A list or tuple of base rotation matrices [R0, R1, R2, R3].
        t_i : list of torch.Tensor
            A list or tuple of base translations [t0, t1, t2, t3].
        T_i_rot_delta : list of torch.Tensor
            The rotation deltas [delta_R0, delta_R1, delta_R2, delta_R3].
        T_i_trans_delta : list of torch.Tensor
            The translation deltas [delta_t0, delta_t1, delta_t2, delta_t3].
        device : torch.device
            Where computations should happen.
        return_gradients : bool
            If False, detach the final outputs from the computation graph.

        Returns
        -------
        R : torch.Tensor
            Interpolated rotation(s).
        t : torch.Tensor
            Interpolated translation(s).
        theta : torch.Tensor or None
            Axis-angle representation of the rotation (if return_gradients=True).
        rho : torch.Tensor or None
            Interpolated translation delta (if return_gradients=True).
        """

        if R_i is None and t_i is None and T_i_rot_delta is None and T_i_trans_delta is None:
            R_i = self.R_i
            t_i = self.t_i
            T_i_rot_delta = self.T_i_rot_delta
            T_i_trans_delta = self.T_i_trans_delta


        if u.dim() == 0:
            u = u.unsqueeze(0)

        #u = u.unsqueeze(-1)

        # Precompute powers of u
        uu = u**2
        uuu = u**3
        one_over_six = 1.0 / 6.0
        half_one = 0.5

        # ---------------------
        # Compute interpolation coefficients
        # t-coeffs
        coeff0 = one_over_six - half_one * u + half_one * uu - one_over_six * uuu
        coeff1 = 4 * one_over_six - uu + half_one * uuu
        coeff2 = one_over_six + half_one * u + half_one * uu - half_one * uuu
        coeff3 = one_over_six * uuu

        # r-coeffs (a slightly different scheme for rotations)
        coeff1_r = 5 * one_over_six + half_one * u - half_one * uu + one_over_six * uuu
        coeff2_r = one_over_six + half_one * u + half_one * uu - 2 * one_over_six * uuu
        coeff3_r = one_over_six * uuu

        # ---------------------
        # Translation Spline - 包含 T_i_trans_delta 以支持梯度传播
        t_0 = t_i[0] 
        t_1 = t_i[1]       
        t_2 = t_i[2]
        t_3 = t_i[3]

        # 基础位姿插值
        t_base = coeff0 * t_0 + coeff1 * t_1 + coeff2 * t_2 + coeff3 * t_3
        # delta 插值
        t_delta = (
            T_i_trans_delta[0] * coeff0
            + T_i_trans_delta[1] * coeff1
            + T_i_trans_delta[2] * coeff2
            + T_i_trans_delta[3] * coeff3
        )
        # 将 delta 加到基础位姿上
        t_out = t_base + t_delta

        # Expand last dim to match e.g. [N,3] vs. [N,3,1] if needed
        # t_out = t_out.unsqueeze(-1)  # only if your pipeline expects [N,3,1]

        # Rotation Spline
        # Convert to rotation matrices, apply deltas, then to quaternions
        R0 = SO3_exp(T_i_rot_delta[0]) @ R_i[0]
        R1 = SO3_exp(T_i_rot_delta[1]) @ R_i[1]
        R2 = SO3_exp(T_i_rot_delta[2]) @ R_i[2]
        R3 = SO3_exp(T_i_rot_delta[3]) @ R_i[3]

        q0 = matrix_to_quaternion(R0).to(device)
        q1 = matrix_to_quaternion(R1).to(device)
        q2 = matrix_to_quaternion(R2).to(device)
        q3 = matrix_to_quaternion(R3).to(device)

        q_01 = quaternion_multiply(quaternion_invert(q0), q1)  # shape (...,4)
        q_12 = quaternion_multiply(quaternion_invert(q1), q2)
        q_23 = quaternion_multiply(quaternion_invert(q2), q3)


        # Use your log-exp approach to do the cubic blend
        r_01 = quaternion_to_axis_angle(q_01)  # shape (...,3)
        r_12 = quaternion_to_axis_angle(q_12)
        r_23 = quaternion_to_axis_angle(q_23)
        
        # Scale the axis-angles
        r_01_scaled = r_01 * coeff1_r
        r_12_scaled = r_12 * coeff2_r
        r_23_scaled = r_23 * coeff3_r
        
        # "Exponentiate" each scaled axis-angle => quaternions
        q_t_0 = axis_angle_to_quaternion(r_01_scaled)  # shape (N,4)
        q_t_1 = axis_angle_to_quaternion(r_12_scaled)  # shape (N,4)
        q_t_2 = axis_angle_to_quaternion(r_23_scaled)  # shape (N,4)

        q_product1 = quaternion_multiply(q_t_1, q_t_2)       # (N,4)
        q_product2 = quaternion_multiply(q_t_0, q_product1)  # (N,4)
        q_out = quaternion_multiply(q0, q_product2)          # (N,4)

        #q_out = q_out.squeeze(-1).squeeze(0)

        #print("q0", q0.shape, "q1", q1.shape, "q2", q2.shape, "q3", q3.shape, "q_out", q_out.shape)

        R_out = quaternion_to_matrix(q_out)
        if not return_gradients:
            R_out = R_out.detach()

        if return_gradients:
            theta = quaternion_to_axis_angle(q_out)
            rho = t_delta  # rho 就是 t_delta，已经在上面计算过了
        else:
            theta = None
            rho = None
        #should be n_virtual_cams x n_knots x 3
        #print("theta", theta.shape, "rho", rho.shape)

        return R_out, t_out, theta, rho

    def get_mid_extrinsic(self):
        return self.get_virtual_extrinsic(self.n_virtual_cams//2)

    def get_virtual_extrinsic(self, index):
        u = torch.linspace(0, 1, steps=self.n_virtual_cams, device=self.device)[index]

        if self.interpolation == "linear":

            R, t, theta, rho = self.interpolate_linear(
                u = u,
                R_i = None,
                t_i = None,
                T_i_rot_delta = None,
                T_i_trans_delta = None,
                device = self.device,
                return_gradients = True
            )
            return R, t, theta, rho

        elif self.interpolation == "cubic":
            R, t, theta, rho = self.interpolate_cubic_spline(
                u = u,
                R_i = None,
                t_i = None,
                T_i_rot_delta = None,
                T_i_trans_delta = None,
                device = self.device,
                return_gradients = True
            )
            return R, t, theta, rho

        elif self.interpolation == "bezier":
            raise NotImplementedError
        
        else:
            raise NotImplementedError
        
    def get_virtual_extrinsics(self, return_gradients=True):
        u_all = torch.linspace(0, 1, steps=self.n_virtual_cams, device=self.device).unsqueeze(1)

        if self.interpolation == "linear":
            R, t, theta, rho = self.interpolate_linear(
                u = u_all, 
                R_i = None,
                t_i = None,
                T_i_rot_delta = None,
                T_i_trans_delta = None,
                device = self.device,
                return_gradients = return_gradients
            )
            return R, t, theta, rho

        elif self.interpolation == "cubic":
            R, t, theta, rho = self.interpolate_cubic_spline(
                u = u_all,
                R_i = None,
                t_i = None,
                T_i_rot_delta = None,
                T_i_trans_delta = None,
                device = self.device,
                return_gradients = return_gradients
            )
            return R, t, theta, rho

        elif self.interpolation == "bezier":
            raise NotImplementedError
        else:
            raise NotImplementedError
        
    """
    def get_gt_virtual_extrinsics(self, realgt_pose=True):
        # For example: linear version
        u_all = torch.linspace(0, 1, steps=self.n_virtual_cams, device=self.device).unsqueeze(1)

        if realgt_pose:
            # R_realgt and T_realgt are, say, [R0, R1], [t0, t1] for linear
            R_i = [self.R_realgt[0], self.R_realgt[1]]
            t_i = [self.T_realgt[0], self.T_realgt[1]]
        else:
            R_i = [self.R_gt[0], self.R_gt[1]]
            t_i = [self.T_gt[0], self.T_gt[1]]

        R, t, _, _ = self.interpolate_linear(
            u = u_all,
            R_i = R_i,
            t_i = t_i,
            T_i_rot_delta = self.T_i_rot_delta,
            T_i_trans_delta = self.T_i_trans_delta,
            device = self.device,
            return_gradients = False
        )
        return R, t
    """



