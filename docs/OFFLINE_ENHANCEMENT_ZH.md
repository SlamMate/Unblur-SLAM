# Unblur-SLAM 离线增强流水线

这条实验流水线在不改变 Triangle Splatting 场景表示的前提下，把
FrameCrafter 稀疏视角生成、EVSSM 条件图和 residual replay 接到
Unblur-SLAM。当前已经实现并测试的是**两遍式生成与数据适配**；本页末尾的
fr2 smoke 不是完整 SLAM 收益实验。

当前边界必须先说明：

- FrameCrafter 是 pose-conditioned sparse-view NVS，不是传统视频插帧器，也
  不能因为输出通过几何门控就称为“锐帧”。
- 规划只允许使用第一遍 DROID 的未对齐估计轨迹、相机内参、原始 RGB-D 和
  DROID tracking anchors。`groundtruth.txt`、参考轨迹、对齐轨迹和第二遍含
  synthetic 的轨迹都禁止作为生成位姿来源。
- 已有子图边界与外部 SE(3) correction 接口，但尚未实现 LoopSplat 的
  place recognition、3DGS registration、PGO 或真实子图融合。
- 因果视频去模糊与 FrameCrafter LoRA 都有训练代码入口，但仓库中没有声称
  已完成训练、验证或可发布的对应权重。

## 1. 第一遍跟踪与 DROID anchors

先运行只跟踪，得到每个原始视频帧的未对齐估计位姿和 DROID 实际选中的
tracking anchors：

```bash
python run.py configs/I2slam/freiburg2_xyz.yaml --only_tracking
```

完整轨迹输出中必须包含 `traj_est_not_align`。可显式导出逐帧 CSV：

```bash
python scripts/export_framecrafter_trajectory.py \
  --trajectory-npz /ABS/first_pass/traj_full_full_traj.npz \
  --config configs/I2slam/freiburg2_xyz.yaml \
  --output /ABS/framecrafter_fr2_xyz/estimated_frames.csv
```

`framecrafter.anchor_indices` 接收第一遍的 `video.npz`（读取其中 DROID
timestamps 对应的 source indices）或严格递增的 anchor-index 文本文件。这里
的 anchor 与论文清晰 GT eval 帧是两套列表，不得混用。

导出器只读取 TUM 的 `rgb.txt` 和 `depth.txt`，不会读取
`groundtruth.txt`，也不会回退到 `traj_ref_poses` 或已对齐轨迹。生产 CSV
绑定 NPZ c2w、trajectory key、pose provenance、源 RGB-D 及代码内容的
SHA-256；source-index 必须为 `0..N-1`，eval mask 必须全是 original/eval。

## 2. 在相邻 anchors 间规划稀疏视角

推荐使用 `planner_mode: overlap_blur_feature`。相邻 DROID anchors 按下面的
顺序处理：

1. 由相机内外参与深度范围计算粗视锥重合，覆盖“相机中心接近但朝向相反”
   这类仅靠平移/旋转阈值容易漏掉的情况。
2. 把两端深度分别投影到对端，进行遮挡、深度一致性与目标覆盖检查，得到
   保守的**双向 RGB-D 可见重合度**。
3. 在配置的模糊区间内，可选 ORB 或 SIFT 匹配，并用
   homography/fundamental/essential RANSAC 给出图像覆盖证据。该分支用于融合
   overlap；essential 本身没有平移尺度，默认不据此改完整 SE(3)。
4. 可选 RGB-D `solvePnPRansac`：用左端 RAW RGB-D 反投影出有尺度 3D 点，
   与右端 RAW 图像匹配点求解相对位姿，再把完整 rotation + translation
   correction 接到右相机。只有内点数、内点率、重投影 RMSE、最大旋转改正量
   和最大平移改正量全部通过安全门，才采用精化位姿并重算 RGB-D/视锥 overlap；
   否则保留第一遍位姿。

当测得 overlap 低于 `target_pair_overlap` 时，按 overlap deficit 在同一 gap
内产生多个均匀 alpha。多个 alpha 共用一次 FrameCrafter batch，目标位姿沿
该区间的完整逐帧估计轨迹局部插值，不把长 gap 简化成单条直线 SE(3) chord。
如果预算无法跨过 `hard_submap_overlap`，只记录 submap boundary，不冒充已完成
子图融合。连续模糊区域的候选与稀疏视角候选分别保留，最后由 scene-wide cap
优先保留低重合 gap，再对其余时序候选做均匀采样。

## 3. 上下文：局部模糊帧、锐帧与 EVSSM

默认 `M=6` 个条件视角由角色约束选择：

- 2 个不可替换的真实 gap endpoints；
- 2 个目标附近的连续模糊观测，用来保留局部运动上下文；
- 2 个目标前后（或同区间内）的高重合锐帧，以目标视锥 overlap、锐度、
  可靠度、视角多样性、冗余和时序局部性共同排序。

`raw`、`evssm`、`hybrid` 三种输入均已接通。`evssm` 要求每个选中条件帧都
通过预计算元数据中的置信度、锐度增益和一致性门控；`hybrid` 默认仅让
`sharp_before`、`sharp_after`、`sharp_context` 使用合格 EVSSM 图，endpoints
和 local blurry 仍用 RAW。每个条件帧都会在 report 中记录角色、source
index、raw/resolved path、实际 resolved mode、provider、fallback reason、
EVSSM 指标和 resolved SHA-256，所以 EVSSM 实验不能静默退化成 RAW。

先独立预计算 EVSSM，以避免 EVSSM 与 14B FrameCrafter 同时占显存：

```bash
python scripts/precompute_framecrafter_evssm.py \
  --frames-csv /ABS/framecrafter_fr2_xyz/estimated_frames.csv \
  --checkpoint /ABS/pretrained/evssm/net_g_latest.pth \
  --source-indices 2250 2282 2320 2352 2358 2387 \
  --device cuda:0 \
  --output-dir /ABS/framecrafter_fr2_xyz/evssm_context
```

随后把 `/ABS/framecrafter_fr2_xyz/evssm_context/metadata.json` 写入
`framecrafter.evssm_metadata`。纯 `evssm` 推荐 `evssm_fallback: error`；做
hybrid ablation 时可显式设置 `raw`，fallback 仍会进入 provenance。

## 4. 生成后的三分区与第二遍注入

每个候选同时经过锐度、双侧 RGB-D、深度一致性、光度和重投影门控，report
固定分成：

- `sharp_accepted`：几何门和 `min_sharpness_gain` 都通过；
- `geometry_only`：所有几何/光度/重投影门通过，但未过锐度门；
- `rejected`：至少一个几何类门控失败。

`acceptance_mode: sharp` 只注入 `sharp_accepted`；`geometry` 还允许注入
`geometry_only`，但 Mapper 会再乘
`framecrafter.geometry_only_weight_scale`（默认 `0.5`），并保持合成帧只监督
既有几何、不创建 Gaussian、不参与 ATE，也不进入清晰 GT
PSNR/SSIM/LPIPS。这个低权重不能把 `geometry_only` 改称“锐帧”。

复制并填写示例后运行第二遍：

```bash
python run.py /ABS/freiburg2_xyz_offline_enhanced.yaml
```

`reuse_existing: true` 只复用内容指纹完全一致的结果。指纹覆盖轨迹、anchors、
源 RGB-D、EVSSM metadata、模型全部 shard、规划/门控/adapter 源码和所有生成
参数；生成结果与 manifest 是按 signature/generation-id 写出的不可变快照。
worker 会再次核对 RGB-D/timestamp、report 和每张合成 RGB-D 的哈希。
`test_only_blend` 仅供 CPU 合约测试，`run.py` 禁止它进入真实 SLAM。

## 5. 两台机器的不可变 shard contract

不能让两台机器各自重新挑 targets。先用完整同参数 `--plan-only` 产生唯一全局
plan report，然后创建包含模型身份、语义配置身份、batch→worker 分配和所有
target IDs 的不可变 contract：

```bash
python scripts/run_framecrafter_preprocess.py \
  --frames-csv /ABS/framecrafter_fr2_xyz/estimated_frames.csv \
  --anchor-indices /ABS/first_pass/video.npz \
  --depth-scale 5000 \
  --planner-mode overlap_blur_feature \
  --feature-refinement --pnp-refinement \
  --target-pair-overlap 0.80 \
  --context-count 6 --local-blurry-contexts 2 --sharp-contexts 2 \
  --context-image-mode hybrid \
  --evssm-metadata /ABS/framecrafter_fr2_xyz/evssm_context/metadata.json \
  --plan-only --output-dir /ABS/framecrafter_fr2_xyz/global_plan

python scripts/build_framecrafter_shard_contract.py \
  --plan-report /ABS/framecrafter_fr2_xyz/global_plan/preprocess_report_SIGNATURE_GENERATION.json \
  --shard-count 2 \
  --model-identity /ABS/framecrafter_fr2_xyz/model_identity.json \
  --config-identity /ABS/framecrafter_fr2_xyz/config_identity.json \
  --output /ABS/framecrafter_fr2_xyz/shared/shard_contract.json
```

两台机器必须拿到同一个 contract 文件，并用与 global plan 完全相同的规划、
上下文、模型和门控参数运行；这里只省略了前面已经列出的共同参数：

```bash
# machine 0
python scripts/run_framecrafter_preprocess.py COMMON_IDENTICAL_ARGS \
  --shard-contract /ABS/framecrafter_fr2_xyz/shared/shard_contract.json \
  --shard-index 0 \
  --shard-envelope /ABS/framecrafter_fr2_xyz/shard0/envelope.json \
  --output-dir /ABS/framecrafter_fr2_xyz/shard0

# machine 1
python scripts/run_framecrafter_preprocess.py COMMON_IDENTICAL_ARGS \
  --shard-contract /ABS/framecrafter_fr2_xyz/shared/shard_contract.json \
  --shard-index 1 \
  --shard-envelope /ABS/framecrafter_fr2_xyz/shard1/envelope.json \
  --output-dir /ABS/framecrafter_fr2_xyz/shard1

python scripts/merge_framecrafter_shards.py \
  /ABS/framecrafter_fr2_xyz/shard0/envelope.json \
  /ABS/framecrafter_fr2_xyz/shard1/envelope.json \
  --output-dir /ABS/framecrafter_fr2_xyz/merged
```

worker 会先重建并核对完整 global plan，再只运行 contract 分配的完整 batch；
同 gap 多 alpha 不会被拆到两台机器。merge 缺一个 shard、重复 target、模型或
配置身份不一致、report/manifest 哈希变化都会 fail closed。不要直接合并裸
report。

## 6. FrameCrafter LoRA 与因果去模糊状态

面向 Replica Blurry/Unblur-SLAM 成对数据的 LoRA 数据 adapter 和 launcher
已经提供，支持 `M` 个 raw/EVSSM/hybrid 模糊上下文到 `N` 个锐目标，并拒绝
GT/对齐轨迹作为相机条件：

```bash
python scripts/build_framecrafter_lora_dataset.py \
  --manifest /ABS/paired_sequences.jsonl \
  --output-root /ABS/framecrafter_lora_data \
  --num-input-frames 6 --num-output-frames 1 \
  --context-mode hybrid

# 默认只打印/保存启动规范，不会训练
python scripts/launch_framecrafter_lora.py \
  --dataset-root /ABS/framecrafter_lora_data \
  --framecrafter-root /ABS/FrameCrafter \
  --base-model-root /ABS/Wan2.1-I2V-14B \
  --framecrafter-checkpoint /ABS/framecrafter.safetensors \
  --output-path /ABS/framecrafter_lora_run
```

默认规范是小学习率 `5e-6`、rank 32、2×A6000、`<=192×336`、bf16、ZeRO-3
及 CPU parameter/optimizer offload；真正启动还需要 `--execute` 和显式风险确认。
这只是训练代码入口，**本项目尚未执行或验证该 LoRA 训练**。

`scripts/train_causal_video_deblur.py` 和
`scripts/export_causal_video_deblur.py` 同样是把 EVSSM 训练分布迁移到只看当前/
历史帧的流式模型的实验入口。当前没有可声称已训练完成的 causal checkpoint，
也不能把现有单帧 EVSSM 权重写成“已训练的视频因果模型”。

## 7. 历史 26K residual-view-replay 消融（非官方 ReSplat）

`budget_mode: replace_tail` 保持总预算 26K：22K 均匀采样 + 4K 基于渲染残差
的优先重放。它只是本项目早期自研的视角采样消融，不是官方 ReSplat
网络，也没有改变 Triangle Splatting 表示。该路径默认关闭，既有结果不得作为
“ReSplat”结果引用；官方 cvg/ReSplat 只通过独立、哈希绑定的官方模型桥接运行。

系统在 10K/20K/26K 保存 Gaussian、推理状态、
`clear_gt_metrics.json` 与清晰 GT renders。正式比较只统计论文规定的完整清晰
GT 集合，基线与增强版必须使用相同 eval mask；测试集 PSNR 不可用于反向挑选
checkpoint，若需 best checkpoint 必须另设 validation 序列。

## 8. fr2_xyz 单 gap 真实生成 smoke

下面仅是 `2282 → 2358` 单 gap 的 FrameCrafter 生成/adapter smoke，不是完整
SLAM 训练，也没有证明 ATE、PSNR、SSIM 或 LPIPS 提升：

- PnP 前双向 RGB-D 可见 overlap 为约 `0.228`；RGB-D
  `solvePnPRansac` 得到 `483` 个内点并通过全部安全门，精化后为约 `0.663`。
  与图像 RANSAC 覆盖证据融合后用于规划的 overlap 为约 `0.572`。
- `target_pair_overlap=0.80`，因此同 gap 插入 alpha=`1/3, 2/3` 两个目标，并
  共用一次生成调用。
- 测试组合中的最佳结果是原生 `640×480`、20 inference steps；两个目标的
  Laplacian sharpness gain 分别是 `1.0003`、`1.0035`。它们都低于锐度门
  `1.05`，所以均为 `geometry_only`，**不能称为生成锐帧**。
- 作为消融，`832` 宽、20 steps 的 gains 是 `0.8178/0.7927`；
  `640×480`、50 steps 是 `0.9825/0.9885`。更多 diffusion steps 在这个 gap
  上没有提高锐度。
- 此 smoke 在 3397 个原始帧中注入 2 个候选，即 `2/3397 = 0.0589%`；这是
  单 gap smoke 的比例，不是完整 fr2 场景最终增帧率。

因此当前证据支持“重合规划、PnP、安全门控、多 alpha、角色上下文、EVSSM
provenance 和第二遍 adapter 已跑通”，不支持“FrameCrafter 已生成可靠锐帧”或
“SLAM 指标已经提升”。下一步必须在固定清晰 GT eval mask 上跑原版与增强版
相同预算的 SLAM 对照，才能报告系统收益。

## 当前外部依赖

- FrameCrafter-compatible checkout、FrameCrafter checkpoint 和
  Wan2.1-I2V-14B backbone；本仓库不会自动下载 14B 模型。
- EVSSM checkpoint（只可称单帧去模糊权重）。
- 若启用 `deblur.frontend: causal_torchscript`，需要另行真实训练并导出的
  causal TorchScript checkpoint。
