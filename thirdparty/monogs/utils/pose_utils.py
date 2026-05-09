# Copyright 2024 The MonoGS Authors.

# Licensed under the License issued by the MonoGS Authors
# available here: https://github.com/muskie82/MonoGS/blob/main/LICENSE.md


import numpy as np
import torch
from torch import lerp, zeros_like
from torch.linalg import norm
from thirdparty.monogs.utils.rotation_conv import matrix_to_quaternion, quaternion_to_matrix



def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged




def get_new_RT(camera, knot):
    tau = torch.cat([camera.T_i_trans_delta[knot], camera.T_i_rot_delta[knot]], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R_i[knot]
    T_w2c[0:3, 3] = camera.t_i[knot]

    new_w2c = SE3_exp(tau) @ T_w2c

    if not torch.isfinite(new_w2c).all():
        raise ValueError(f"Invalid new_w2c matrix at knot {knot}")

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    q = matrix_to_quaternion(new_R)

    q = q / (torch.norm(q) + 1e-6)

    return q, new_T

def get_new_RT_prev(camera, knot):

    T_w2c = torch.eye(4, device=camera.R.device)
    if camera.deblur_fail:
        T_w2c[0:3, 0:3] = camera.R_i[knot]
        T_w2c[0:3, 3] = camera.t_i[knot]
    else:
        T_w2c[0:3, 0:3] = camera.R
        T_w2c[0:3, 3] = camera.T

    new_w2c = T_w2c

    if not torch.isfinite(new_w2c).all():
        raise ValueError(f"Invalid new_w2c matrix at knot {knot}")

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    q = matrix_to_quaternion(new_R)

    q = q / (torch.norm(q) + 1e-6)

    return q, new_T

def update_pose_knot(camera, knot, converged_threshold=1e-4):
    tau = torch.cat([camera.T_i_trans_delta[knot], camera.T_i_rot_delta[knot]], axis=0)

    #print("tau: ", tau, "camera: ", camera.uid, "knot: ", knot)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R_i[knot]
    T_w2c[0:3, 3] = camera.t_i[knot]

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT_motion(new_R, new_T, knot)
    camera.T_i_rot_delta[knot].data.fill_(0)
    camera.T_i_trans_delta[knot].data.fill_(0)
    return converged

def get_next_traj(prev_camera):
    if prev_camera.interpolation =='linear':
        cur_frame_R_i = []
        cur_frame_t_i = []
        cur_frame_R_i.append(prev_camera.R_i[1])
        cur_frame_t_i.append(prev_camera.t_i[1])
        q_0 = matrix_to_quaternion(prev_camera.R_i[0])
        q_1 = matrix_to_quaternion(prev_camera.R_i[1])
        
        t_1 = (prev_camera.t_i[1] - prev_camera.t_i[0]) * 2 + prev_camera.t_i[0]
        R_1 = quaternion_to_matrix(slerp(torch.tensor(2), q_0, q_1))

        cur_frame_R_i.append(R_1)
        cur_frame_t_i.append(t_1)

        return cur_frame_R_i, cur_frame_t_i



def get_next_traj_from_dspo(w2c, w2c_gt, cameras, keyframe_idxs, video_idxs, idx):
        
    t = w2c[0:3, 3]
    R = w2c[0:3, 0:3]
    mid_q_cur = matrix_to_quaternion(R)
    if len(keyframe_idxs) > 1:
        prev_keyframe_idx = keyframe_idxs[- 2]
        prev_frame_idx = video_idxs[-2]
    else:
        prev_keyframe_idx = None
        prev_frame_idx = None

    if prev_keyframe_idx is not None:
        R_prev, t_prev, _, _ = cameras[prev_frame_idx].get_mid_extrinsic()
        mid_q_prev = matrix_to_quaternion(R_prev)
        delta_prev = idx - prev_keyframe_idx
        fraction_start = (idx - 0.5 - prev_keyframe_idx) / delta_prev

        # Print details for previous frame interpolation
        print(f"delta_prev: {delta_prev}")
        print(f"fraction_start: {fraction_start}")

        t_expected_start = torch.lerp(t_prev, t, fraction_start)
        q_expected_start = slerp(torch.tensor(fraction_start), mid_q_prev, mid_q_cur)
        R_start = quaternion_to_matrix(q_expected_start)

        fraction_end = 1 + 1 - fraction_start
        print(f"fraction_end: {fraction_end}")
        t_expected_end = torch.lerp(t, t_prev, fraction_end)
        q_expected_end = slerp(torch.tensor(fraction_end), mid_q_prev, mid_q_cur)
        R_end = quaternion_to_matrix(q_expected_end)

        w2c_test = torch.zeros((2, 4, 4)).to(w2c_gt.device)
        w2c_test[0] = torch.eye(4)
        w2c_test[1] = torch.eye(4)
        w2c_test[0, 0:3, 0:3] = R_start
        w2c_test[0, 0:3, 3] = t_expected_start
        w2c_test[1, 0:3, 0:3] = R_end
        w2c_test[1, 0:3, 3] = t_expected_end
        
        #print("new init error", compute_pose_error(w2c_test, w2c_gt))

        #with torch.no_grad():
        #    self.cameras[video_idx].update_RT(R_start, t_expected_start, 0)
        #    self.cameras[video_idx].update_RT(R_end, t_expected_end, 1)

    return w2c_test

def slerp(t, v0, v1,  DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        v0: Starting vector
        v1: Final vector
        t: Float value between 0.0 and 1.0
        DOT_THRESHOLD: Threshold for considering the two vectors as
                                colinear. Not recommended to alter this.
    Returns:
        Interpolation vector between v0 and v1
    '''
    #assert v0.shape == v1.shape, "shapes of v0 and v1 must match"
    
        # 确保所有输入都是 tensors
    if not isinstance(v0, torch.Tensor):
        v0 = torch.tensor(v0, dtype=torch.float32, device="cuda")
    if not isinstance(v1, torch.Tensor):
        v1 = torch.tensor(v1, dtype=torch.float32, device="cuda")
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32, device="cuda")

    v0 = v0.type(torch.float32).to("cuda")
    v1 = v1.type(torch.float32).to("cuda")
    #print("v0", v0, "v1", v1)
    t = t.type(torch.float32).to("cuda")

    # Normalize the vectors to get the directions and angles
    v0_norm = norm(v0, dim=-1)
    v1_norm = norm(v1, dim=-1)
    v0_normed = (v0 / v0_norm.unsqueeze(-1))
    v1_normed = (v1 / v1_norm.unsqueeze(-1)) 
    # Dot product with the normalized vectors
    dot = ((v0_normed * v1_normed).sum(-1))
    dot_mag = dot.abs()

    if dot<0:
        v0, dot = -v0, -dot

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    DOT_THRESHOLD = torch.tensor(DOT_THRESHOLD, device="cuda")
    gotta_lerp = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
    can_slerp = ~gotta_lerp

    t_batch_dim_count = max(0, t.dim()-v0.dim())
    t_batch_dims = t.shape[:t_batch_dim_count] 
    #print("t_batch_dims", t_batch_dims)
    out = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        #print("gotta_lerp", gotta_lerp)
        #print type of v0, v1, t
        #print("v0", v0.dtype)
        lerped = lerp(v0, v1, t)

        out = lerped.where(gotta_lerp.unsqueeze(-1), out)

    # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
    if can_slerp.any():
        #print("can_slerp", can_slerp)
        # Calculate initial angle between v0 and v1
        theta_0 = dot.arccos().unsqueeze(-1)
        sin_theta_0 = theta_0.sin()
        # Angle at timestep t
        theta_t = theta_0 * t
        sin_theta_t = theta_t.sin()
        # Finish the slerp algorithm
        s0 = (theta_0 - theta_t).sin() / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        slerped = s0 * v0 + s1 * v1

        out = slerped.where(can_slerp.unsqueeze(-1), out)
    #print("out", out)
    return out

def compute_pose_error(T1, T2):

    # Pose difference
    delta_T = torch.linalg.inv(T1) @ T2

    # Translation error (batch)
    t = delta_T[:, :3, 3]  # Extract batch translation vectors
    e_translation = torch.norm(t, dim=1)  # Compute norm for each batch

    # Rotational error (batch)
    R = delta_T[:, :3, :3]  # Extract batch rotation matrices
    trace_R = torch.diagonal(R, dim1=1, dim2=2).sum(dim=1)
    e_rotation = torch.acos(torch.clamp((trace_R - 1) / 2, -1, 1))

    return torch.norm(e_translation).item() + torch.norm(e_rotation).item()

