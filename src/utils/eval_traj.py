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
from lietorch import SE3
from src.utils.Printer import FontColor
from scipy.spatial.transform import Rotation, Slerp


def _align_sim3_with_collinear_fallback(traj_est, traj_ref):
    """Align trajectories, allowing a non-constant rank-1 reference path.

    EVO deliberately rejects rank-1 covariance because the rotation about the
    reference line is not identifiable.  Motorized linear-rail datasets such
    as Ev-DeblurNeRF CDAVIS are nevertheless valid translation-ATE datasets:
    that free rotation does not change the least-squares positional residual.
    We use the same Umeyama equations/SVD and permit exactly rank 1, while
    retaining EVO's failure for a constant reference or estimate.
    """

    try:
        return traj_est.align(traj_ref, correct_scale=True)
    except Exception as error:
        from evo.core.geometry import GeometryException

        if not isinstance(error, GeometryException) or "Degenerate covariance rank" not in str(error):
            raise
        x = np.asarray(traj_est.positions_xyz, dtype=np.float64).T
        y = np.asarray(traj_ref.positions_xyz, dtype=np.float64).T
        if x.shape != y.shape or x.shape[0] != 3 or x.shape[1] < 3:
            raise
        mean_x = x.mean(axis=1)
        mean_y = y.mean(axis=1)
        x_centered = x - mean_x[:, None]
        y_centered = y - mean_y[:, None]
        reference_rank = int(np.linalg.matrix_rank(y_centered))
        estimate_rank = int(np.linalg.matrix_rank(x_centered))
        if reference_rank != 1 or estimate_rank < 1:
            raise
        sigma_x = float(np.linalg.norm(x_centered) ** 2 / x.shape[1])
        if not np.isfinite(sigma_x) or sigma_x <= np.finfo(np.float64).eps:
            raise
        covariance = y_centered @ x_centered.T / x.shape[1]
        u, singular, v = np.linalg.svd(covariance)
        if int(np.count_nonzero(singular > np.finfo(singular.dtype).eps)) < 1:
            raise
        sign = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(v) < 0.0:
            sign[-1, -1] = -1.0
        rotation = u @ sign @ v
        scale = float(np.trace(np.diag(singular) @ sign) / sigma_x)
        if not np.isfinite(scale) or scale <= 0.0:
            raise
        translation = mean_y - scale * rotation @ mean_x
        from evo.core import lie_algebra as lie

        traj_est.scale(scale)
        traj_est.transform(lie.se3(rotation, translation))
        return rotation, translation, scale


def _frame_is_evaluation_target(stream, dataset_index):
    """Return whether a dataset frame may contribute a reference trajectory.

    FrameCrafter inserts carry estimated poses used to generate their images.
    Those poses are training metadata, not ground truth.  The augmented dataset
    marks them ``eval=false``; checking that bit before touching ``stream.poses``
    prevents the estimate from leaking back into ATE as its own reference.
    """

    dataset_index = int(dataset_index)
    metadata = getattr(stream, "frame_metadata", None)
    if metadata is not None:
        return bool(metadata[dataset_index].get("eval", True))
    if hasattr(stream, "is_eval_frame"):
        return bool(stream.is_eval_frame(dataset_index))
    return True


def _dataset_index(value, stream_length):
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"trajectory timestamp must be a dataset index, got {value!r}")
    index = int(round(numeric))
    if not np.isclose(numeric, index, atol=1.0e-5):
        raise ValueError(f"trajectory timestamp must be a dataset index, got {value!r}")
    if not 0 <= index < int(stream_length):
        raise IndexError(
            f"trajectory dataset index {index} outside stream of {stream_length} frames"
        )
    return index


def _midpoint_reference_pose(gt_pose):
    """Convert a dataset pose/control-knot array to one C2W reference pose."""

    gt_pose = np.asarray(gt_pose)
    if gt_pose.shape == (4, 4):
        return gt_pose.astype(np.float64, copy=True)
    if gt_pose.ndim != 3 or gt_pose.shape[1:] != (4, 4) or len(gt_pose) == 0:
        raise ValueError(f"reference pose must be 4x4 or Kx4x4, got {gt_pose.shape}")
    if len(gt_pose) == 1:
        return gt_pose[0].astype(np.float64, copy=True)

    first = np.asarray(gt_pose[0], dtype=np.float64)
    last = np.asarray(gt_pose[-1], dtype=np.float64)
    translation = 0.5 * (first[:3, 3] + last[:3, 3])
    key_rotations = Rotation.from_matrix(
        np.stack((first[:3, :3], last[:3, :3]))
    )
    rotation = Slerp([0.0, 1.0], key_rotations)([0.5]).as_matrix()[0]

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def build_evaluation_trajectory_pairs(estimates, dataset_indices, stream, printer=None):
    """Build strictly paired estimate/reference arrays for ATE.

    ``estimates[position]`` is paired only with the dataset frame named by
    ``dataset_indices[position]``.  Frames whose augmentation metadata says
    ``eval=false`` are removed from both arrays at the same position.
    """

    estimates = np.asarray(estimates)
    dataset_indices = np.asarray(dataset_indices).reshape(-1)
    if estimates.ndim != 3 or estimates.shape[1:] != (4, 4):
        raise ValueError(f"estimated trajectory must be Nx4x4, got {estimates.shape}")
    if len(estimates) != len(dataset_indices):
        raise ValueError(
            "estimated trajectory/timestamp length mismatch: "
            f"{len(estimates)} != {len(dataset_indices)}"
        )

    stream_length = len(stream.poses)
    paired_estimates = []
    paired_references = []
    paired_timestamps = []
    kept_dataset_indices = []
    seen_dataset_indices = set()
    for position, value in enumerate(dataset_indices):
        index = _dataset_index(value, stream_length)
        if index in seen_dataset_indices:
            raise ValueError(f"duplicate trajectory dataset index {index}")
        seen_dataset_indices.add(index)
        if not _frame_is_evaluation_target(stream, index):
            continue
        gt_pose = np.asarray(stream.poses[index])
        if not np.isfinite(gt_pose).all():
            if printer is not None:
                printer.print(
                    f"Nan or Inf found in gt poses, skipping dataset frame {index}!",
                    FontColor.INFO,
                )
            continue
        estimate = np.asarray(estimates[position], dtype=np.float64)
        if not np.isfinite(estimate).all():
            if printer is not None:
                printer.print(
                    f"Nan or Inf found in estimated pose, skipping dataset frame {index}!",
                    FontColor.INFO,
                )
            continue
        paired_estimates.append(estimate)
        paired_references.append(_midpoint_reference_pose(gt_pose))
        paired_timestamps.append(float(value))
        kept_dataset_indices.append(index)

    if not paired_estimates:
        raise ValueError("no evaluation frames remain after filtering eval=false metadata")
    return (
        np.asarray(paired_estimates),
        np.asarray(paired_references),
        np.asarray(paired_timestamps, dtype=np.float64),
        np.asarray(kept_dataset_indices, dtype=np.int64),
    )


def align_kf_traj(npz_path,stream,return_full_est_traj=False,printer=None):
    offline_video = dict(np.load(npz_path))
    video_traj = offline_video['poses']
    # 这个timestamp是图片的index
    video_timestamps = offline_video['timestamps']
    traj_est, traj_ref, timestamps, _ = build_evaluation_trajectory_pairs(
        video_traj, video_timestamps, stream, printer=printer
    )

    from evo.core.trajectory import PoseTrajectory3D

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    from evo.core import sync

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = _align_sim3_with_collinear_fallback(traj_est, traj_ref)

    if return_full_est_traj:
        from evo.core import lie_algebra as lie
        traj_est_full = PoseTrajectory3D(poses_se3=video_traj,timestamps=video_timestamps)
        traj_est_full.scale(s)
        traj_est_full.transform(lie.se3(r_a, t_a))
        traj_est = traj_est_full

    return r_a, t_a, s, traj_est, traj_ref    

def align_full_traj(traj_est_full,stream,printer):
    traj_est_full = np.asarray(traj_est_full)
    if len(traj_est_full) != len(stream.poses):
        raise ValueError(
            "full estimated trajectory must match the augmented stream length: "
            f"{len(traj_est_full)} != {len(stream.poses)}"
        )
    traj_est, traj_ref, timestamps, _ = build_evaluation_trajectory_pairs(
        traj_est_full,
        np.arange(len(traj_est_full), dtype=np.float64),
        stream,
        printer=printer,
    )
    
    from evo.core.trajectory import PoseTrajectory3D

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    from evo.core import sync

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = _align_sim3_with_collinear_fallback(traj_est, traj_ref)
    return r_a, t_a, s, traj_est, traj_ref    


def traj_eval_and_plot(traj_est, traj_ref, plot_parent_dir, plot_name,printer):
    import os
    from evo.core import metrics
    from evo.tools import plot
    import matplotlib.pyplot as plt
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)
    printer.print("Calculating APE ...",FontColor.EVAL)
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()

    printer.print("Plotting ...",FontColor.EVAL)
    print("ATE RMSE: ",ape_statistics['rmse'])
    plot_collection = plot.PlotCollection("kf factor graph")
    # metric values
    fig_1 = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig_1, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
    plot.traj_colormap(
    ax, traj_est, ape_metric.error, plot_mode, min_map=ape_statistics["min"],
    max_map=ape_statistics["max"], title="APE mapped onto trajectory")
    plot_collection.add_figure("2d", fig_1)
    plot_collection.export(f"{plot_parent_dir}/{plot_name}.png", False)

    return ape_statistics


def kf_traj_eval(npz_path, plot_parent_dir, plot_name, stream, logger,printer):
    r_a, t_a, s, traj_est, traj_ref = align_kf_traj(npz_path, stream, printer=printer)

    offline_video = dict(np.load(npz_path))
    
    import os
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    ape_statistics = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name,printer)

    output_str = "#"*10+"Keyframes traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    output_str += f"statistics:\n{ape_statistics}"
    printer.print(output_str,FontColor.EVAL)
    printer.print("#"*34,FontColor.EVAL)
    out_path=f'{plot_parent_dir}/metrics_kf_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)
    if logger is not None:
        logger.log({'kf_ate_rmse':ape_statistics['rmse'],'pose_scale':s})

    offline_video["scale"]=np.array(s)
    np.savez(npz_path,**offline_video)

    return ape_statistics, s, r_a, t_a


def full_traj_eval(traj_filler, plot_parent_dir, plot_name, stream,logger,printer):

    traj_est_inv = traj_filler(stream)
    traj_est_lietorch = traj_est_inv.inv()
    traj_est = traj_est_lietorch.matrix().data.cpu().numpy()
    kf_num = traj_filler.video.counter.value
    kf_timestamps = traj_filler.video.timestamp[:kf_num].cpu().int().numpy()
    kf_poses = SE3(traj_filler.video.poses[:kf_num].clone()).inv().matrix().data.cpu().numpy()
    traj_est[kf_timestamps] = kf_poses
    traj_est_not_align = traj_est.copy()
    traj_est_not_align_timestamps = np.arange(
        len(traj_est_not_align), dtype=np.float64
    )
    traj_est_not_align_eval_mask = np.asarray(
        [
            _frame_is_evaluation_target(stream, index)
            for index in range(len(traj_est_not_align))
        ],
        dtype=np.bool_,
    )

    r_a, t_a, s, traj_est, traj_ref = align_full_traj(traj_est, stream, printer)    

    import os
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    ape_statistics = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name,printer)

    traj_save_path = f'{plot_parent_dir}/traj_full_{plot_name}.npz'
    np.savez(traj_save_path,
            traj_est_poses=np.array([pose for pose in traj_est.poses_se3]),
            traj_ref_poses=np.array([pose for pose in traj_ref.poses_se3]),
            traj_est_not_align=traj_est_not_align,
            traj_est_not_align_timestamps=traj_est_not_align_timestamps,
            traj_est_not_align_eval_mask=traj_est_not_align_eval_mask,
            pose_source=np.asarray("droid_traj_est_not_align"),
            uses_ground_truth_pose=np.asarray(False),
            timestamps=traj_est.timestamps,
            scale=s,
            rotation=r_a,
            translation=t_a,
            ate_rmse=ape_statistics['rmse'])
    printer.print(f"Saved full trajectory to {traj_save_path}", FontColor.INFO)
    output_str = "#"*10+"Full traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    output_str += f"statistics:\n{ape_statistics}"
    printer.print(output_str,FontColor.EVAL)
    printer.print("#"*29,FontColor.EVAL)

    
    out_path=f'{plot_parent_dir}/metrics_full_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)
    if logger is not None:
        logger.log({'full_ate_rmse':ape_statistics['rmse']})
    
    return traj_est_not_align, traj_est, traj_ref
