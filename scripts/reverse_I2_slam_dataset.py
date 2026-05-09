import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import os

def quaternion_to_matrix(qx, qy, qz, qw):
    """将四元数转换为旋转矩阵"""
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_matrix()

def pose_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    """将位姿转换为4x4变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T

def compute_pose_difference(pose1, pose2):
    """计算两个位姿矩阵之间的差异
    使用Frobenius范数"""
    return np.linalg.norm(pose1 - pose2, 'fro')

def load_groundtruth(gt_path):
    """加载groundtruth.txt文件
    返回: timestamps列表和poses列表(4x4矩阵)"""
    timestamps = []
    poses = []
    
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) != 8:
                continue
            
            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            pose = pose_to_matrix(tx, ty, tz, qx, qy, qz, qw)
            timestamps.append(timestamp)
            poses.append(pose)
    
    return timestamps, poses

def match_npy_to_timestamp(npy_dir, gt_path, output_path):
    """匹配npy文件到对应的timestamp"""
    print(f"\n处理: {npy_dir}")
    print(f"Ground truth: {gt_path}")
    
    # 加载ground truth
    timestamps, gt_poses = load_groundtruth(gt_path)
    print(f"加载了 {len(timestamps)} 个ground truth位姿")
    
    # 获取所有_gt_pose.npy文件
    npy_files = sorted(Path(npy_dir).glob("*_gt_pose.npy"))
    print(f"找到 {len(npy_files)} 个npy位姿文件")
    
    results = []
    
    for npy_file in npy_files:
        # 提取帧号
        frame_id = npy_file.stem.split('_')[0]
        
        # 加载npy位姿
        npy_pose = np.load(npy_file)
        
        # 计算与所有ground truth位姿的差异
        min_diff = float('inf')
        best_idx = -1
        
        for idx, gt_pose in enumerate(gt_poses):
            diff = compute_pose_difference(npy_pose, gt_pose)
            if diff < min_diff:
                min_diff = diff
                best_idx = idx
        
        matched_timestamp = timestamps[best_idx]
        results.append((frame_id, matched_timestamp, min_diff))
        print(f"  {frame_id}: timestamp={matched_timestamp:.6f}, diff={min_diff:.6e}")
    
    # 保存结果
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# frame_id timestamp difference\n")
        for frame_id, timestamp, diff in results:
            f.write(f"{frame_id} {timestamp:.6f} {diff:.6e}\n")
    
    print(f"结果已保存到: {output_file}")
    return results

if __name__ == "__main__":
    # 数据集配置
    datasets = [
        {
            'name': 'fr1_desk',
            'npy_dir': './datasets/I2-SLAM/fr1_desk',
            'gt_path': './datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/groundtruth.txt',
            'output': './scripts/matched_timestamps_fr1_desk.txt'
        },
        {
            'name': 'fr2_xyz',
            'npy_dir': './datasets/I2-SLAM/fr2_xyz',
            'gt_path': './datasets/TUM_RGBD/rgbd_dataset_freiburg2_xyz/groundtruth.txt',
            'output': './scripts/matched_timestamps_fr2_xyz.txt'
        },
        {
            'name': 'fr3_office',
            'npy_dir': './datasets/I2-SLAM/fr3_office',
            'gt_path': './datasets/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt',
            'output': './scripts/matched_timestamps_fr3_office.txt'
        }
    ]
    
    # 处理所有数据集
    for dataset in datasets:
        try:
            match_npy_to_timestamp(
                dataset['npy_dir'],
                dataset['gt_path'],
                dataset['output']
            )
        except Exception as e:
            print(f"处理 {dataset['name']} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n所有数据集处理完成!")