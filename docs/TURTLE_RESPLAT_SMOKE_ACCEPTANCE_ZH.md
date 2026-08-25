# 官方 TURTLE + 官方 ReSplat smoke 预注册验收方案

本文件在读取新的 TURTLE/ReSplat 结果前固定评估集合、指标和解释边界。它的
目的不是设置一个容易通过的质量阈值，而是防止在看到结果后挑帧、换指标或把
不同预算的实验混在一起。

## 1. 实验身份

- 视频前端：官方 `Ascend-Research/Turtle` 架构，官方 GoPro deblur checkpoint，
  zero-shot 用于 TUM。它不是 ReplicaBlurry 微调模型，报告中必须写明 domain
  shift。
- 重建后端：官方 `cvg/resplat` small 8-view checkpoint；一次 initializer，随后
  对同一初始化 Gaussian 调用一次 `forward_update`，其中包含四次 recurrent
  update。
- smoke 输入：`freiburg2_xyz` 从源帧 0 到 2764 连续、按时间顺序运行 TURTLE；
  只在固定的 42 个 DROID keyframe 保存前端 PNG。流开始时 reset 一次，中间不
  reset。pose、clear-GT 和 ReSplat 结果均不得进入 TURTLE。
- 这不是 ReplicaBlurry fine-tune，不是完整 SLAM ATE 实验，也不是 26K mapping
  实验。

## 2. 预先固定的帧集合

42 个 TUM clear-frame protocol source indices：

```text
0, 9, 15, 49, 58, 72, 89, 109, 125, 166, 220, 319, 374, 407,
435, 470, 483, 523, 568, 704, 750, 789, 827, 926, 1004, 1160,
1251, 1342, 1409, 1460, 1553, 1692, 1795, 1889, 1978, 2055,
2206, 2282, 2358, 2425, 2590, 2764
```

其中只有以下 26 帧存在 baseline arm 保存的、由官方 Unblur-SLAM EVSSM
checkpoint 产生的张量：

```text
49, 72, 166, 319, 435, 470, 483, 750, 827, 1004, 1160, 1251,
1342, 1409, 1460, 1692, 1795, 1889, 1978, 2055, 2206, 2282,
2358, 2425, 2590, 2764
```

因此：

- raw 与 TURTLE 的 clear-frame preservation 报告覆盖 42 帧；
- raw、official EVSSM、official TURTLE 的公平三方报告只覆盖上述 26 帧；
- 旧 ReSplat 输入中另外 16 帧的 `raw_undistorted` fallback 不能写成 EVSSM；
- 固定可视化五帧为 `49, 483, 1342, 2055, 2764`。它们是在 26 帧交集上按
  时间分位点选定，且均为 ReSplat held-out target。

## 3. 前端指标

reference 使用历史 `iter_000400/clear_gt_renders` 的左 512x384 半幅；右半幅
400-step Gaussian render 完全不进入前端指标。已核验左半幅与 TUM 原始 RGB 经
运行配置中的 undistort、528x400 resize、四边各裁 8 像素后逐像素一致。

这 42 帧本来就是 TUM clear-frame protocol；raw 与 reference 因而完全相同。
所以该部分叫 **清晰帧保真 smoke**，不能称为去模糊质量评测。raw 的
PSNR 为正无穷、SSIM 为 1、LPIPS 为 0 是 reference-control，不是算法结果。

对 512x384 RGB PNG 计算：

- PSNR：RGB full-frame MSE，像素范围 `[0,1]`；
- SSIM：`win_size=11, gaussian_weights=True, channel_axis=0`；
- LPIPS：VGG LPIPS，`normalize=True`，与官方 ReSplat metric 实现一致；
- 每个方法给出 mean、median 和逐帧值，不用单个最好帧支撑结论。

时序诊断同时报告：

- adjacent change L1；
- 与 reference 相邻差分的 L1 误差；
- 可行时使用 reference-only Farneback backward flow 对上一帧做 warp，再计算
  temporal residual error；forward/backward consistency 阈值固定为 1 pixel，
  有效像素少于 25% 的 pair 标为 unavailable。

这些 pair 是稀疏 keyframe，相邻 source index 间隔不规则，光流结果只能作为
diagnostic。不能把较小的 adjacent change 单独解释成更稳定，因为它也可能来自
过度平滑。

TURTLE 延迟从完整 0..2764 的 manifest 读取：首帧单独列出，steady-state 统计
排除首帧，报告 mean/median/p95/max、吞吐率和 peak memory（若 producer 提供）。
`p95 <= 33.33 ms` 只作为 30 FPS feasibility flag；不满足它不否定“因果流式”
合同，只说明当前分辨率/硬件没有实时达到 30 FPS。

## 4. ReSplat 指标

对 TURTLE-stream 与既有 EVSSM/raw-mix 两个 paired run，先硬性核验：

- 相同 42 source indices、相同非 GT DROID C2W、相同 FPS context names 和
  held-out target names；
- 相同官方 ReSplat repository commit、checkpoint SHA、image shape；
- 每个 run 都是一次 encoder init、同一初始化对象、四次 official recurrent
  update；
- GT 未用于 pose、context/target selection 或模型输入。

预审计还发现一个必须保留在报告里的历史限制：旧 EVSSM/raw-mix scene 的 26 个
tracker EVSSM tensor 来自 `528x400 resize -> crop 8`，而它的 16 个 raw fallback
来自 direct 512x384 resize；scene 却只声明了一套 direct-resize K。因此旧 scene
内部不是单一投影合同。它只可作为历史 interface probe，并不是相机一致的公平
baseline。新的 TURTLE stream 必须统一使用真实 tracker 的 resize/crop/K 合同。

Primary 指标分别以各自前端 stream 为 target，使用官方保存的 PSNR/SSIM/LPIPS。
由于 target 像素不同，两个 run 的绝对 primary 值只并列展示；只在同一 run 内
解释 `refine4-init0`。

Post-hoc clear-GT 指标在共同的 34 个 held-out targets 上计算。reference 左半幅
用 LANCZOS 直接 resize 到 448x320，再与保存的 init0/refine4 PNG 比较。这个共同
target 允许并列展示；由于旧 mix scene 的相机不一致，不能把两个 run 的差值归因
于 TURTLE。它们仍只是 first-pass-pose、zero-shot interface smoke。

同时对每个 paired run 的 init0/refine4 native PLY 做同索引位置诊断：顶点数、
有限性、非零更新数，以及位移 mean/median/p95/max、`>1 m` 和 `>5 m` 计数。
官方 fixed-grid topology 使同索引比较可作为 update-path 诊断；它不是跨迭代点对应
真值，也没有 GT geometry，因此不能把位移更大或更小直接解释为几何更准确。

## 5. 硬验收与结论边界

硬验收只判断 provenance/执行完整性：官方 repo/checkpoint、连续 2765 步、仅一次
reset、cache 连续更新、42 个输出及 SHA、无 pose/GT 输入、ReSplat paired contract
和共同 selection。质量数值不设事后阈值，如实报告改善或退化。

本 smoke 可以证明：官方 TURTLE causal cache path 和官方 ReSplat recurrent path
按预定接口运行。它不能证明：

- GoPro zero-shot 等价于 ReplicaBlurry fine-tune；
- TURTLE 提升了模糊帧去模糊质量（本 TUM 子集是 clear-frame preservation）；
- SLAM ATE、跟踪成功率或整体稳定性改善；
- 官方 ReSplat 直接细化 Unblur-SLAM 任意 26K Gaussian state；
- 400-step smoke 等价于正式 26K baseline。

历史 `iter_000400` 在本审计中只提供左半幅 clear reference；不得引用其右半幅或
400-step PSNR 作为 26K 结果。本轮没有正式 26K artifact 时，报告必须明确写
`formal_26k_result_present=false`。
