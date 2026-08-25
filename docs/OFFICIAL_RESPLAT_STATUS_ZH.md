# 官方 cvg/ReSplat 接入边界与 smoke 状态

本项目的正式 ReSplat 名称只指 `cvg/resplat` 的官方模型、官方代码与官方
checkpoint。`src/refinement/resplat_replay.py` 是历史 residual-view-replay
采样消融，不属于 ReSplat 方法，默认关闭，也不得用其结果支持 ReSplat 声明。

## 已完成

- 官方仓库与 checkpoint 以 commit/SHA-256 锁定，并在独立环境 strict-load。
- `materialize_official_resplat_inputs.py` 将 official Unblur-SLAM EVSSM tracker
  张量或经过同一相机预处理的 raw fallback 原子物化为审计输入。
- `export_tum_official_resplat_scene.py` 将非 GT DROID OpenCV C2W 轨迹导出为
  官方推理脚本可读的 COLMAP PINHOLE scene。
- `run_paired_official_resplat_smoke.py` 只调用一次官方 encoder，先渲染其初始
  Gaussian，再把同一对象交给官方 `forward_update` 的四次 recurrent update，
  在完全相同的 held-out target 上配对比较。

fr2_xyz 的 small 8-view zero-shot paired smoke 使用 42 个 DROID keyframe（8 个
context、34 个 held-out target）。对 materialized frontend stream，init0 到
refine4 为 PSNR `17.873 -> 17.903 dB`、SSIM `0.656 -> 0.658`、LPIPS
`0.337 -> 0.314`。这是接口与 recurrent path smoke，不是正式 26K 论文结果。

上面这组早期结果使用了 26 个 tracker-space EVSSM tensor 与 16 个 direct-resize
raw fallback，却为整个 scene 声明了一套相机内参。它现在只保留为历史接口 probe，
不能作为正式相机一致的对照。

## 当前官方方案（2026-08-22）

当前实验路线已经删除名称歧义，且只使用两个上游官方组件：

1. 在线前端使用 `Ascend-Research/Turtle` 的官方 GoPro 权重。所有源帧按时间顺序
   调用一次官方增量 K/V `step()`；TUM fr2 的像素合同固定为
   `undistort -> resize 528x400 -> crop 8 -> 512x384`。它不调用 EVSSM、
   自研 causal-EVSSM 或 residual replay。
2. 在线 Unblur-SLAM 仍使用原有 DROID/Mapper。TURTLE 只在模糊门控通过时替换
   tracking/mapping 图像；现阶段不把 ReSplat 放进每帧 tracking 热路径。
3. 每个固定 8-view 子图关闭后，异步运行官方 `cvg/ReSplat` initializer，并把同一
   初始化 Gaussian 对象交给官方四次 recurrent update。结果先保留为 native
   sidecar map；只有将来通过几何门控、坐标转换、去重和 optimizer 重建后，才讨论
   融回 Unblur 的活跃 map。
4. Unblur 的 26K 优化仍是正式基线。官方 ReSplat 当前既不冒充 26K tail sampler，
   也不声称直接细化 26K 后的任意拓扑 Gaussian state。

可复现入口为：

- `scripts/materialize_tum_turtle_stream.py`
- `scripts/run_official_turtle_resplat_pipeline.py`
- `configs/local/official_turtle_resplat/fr2_xyz_gopro_42kf_smoke.json`
- `scripts/run_fr2_official_turtle_smoke.py`
- `configs/local/fr2_xyz_causal_smoke/turtle_official.yaml`

## 当前 smoke 结果

官方 GoPro TURTLE 在 fr2 `0..2764` 上连续更新 2765 次 cache，只输出固定 42 个
DROID keyframe。manifest 与 42 个 PNG 全量哈希通过；steady-state 平均延迟
`232.51 ms`（约 `4.30 FPS`），所以因果流式接口成立，但当前实现不是 30 FPS
实时前端。42 个 clear-frame 上的保真为 PSNR `42.426 dB`、SSIM `0.99673`、
LPIPS `0.003892`。这是清晰帧保真，不是模糊去模糊质量。

在相机一致的 TURTLE stream 上，官方 small 8-view ReSplat 的同对象配对结果为：

- frontend-stream target：`22.976/0.824/0.233 -> 24.184/0.858/0.187`；
- post-hoc clear-GT：`23.007/0.8253/0.2299 -> 24.225/0.8588/0.1845`；
- 34 个 held-out target 的 PSNR、SSIM、LPIPS 都是 34/34 同向改善。

但 native Gaussian 更新仍有几何离群点：同索引位置位移 median `1.81 cm`、
p95 `20.91 cm`、max `12.17 m`。因此这只能支持“官方 recurrent 更新有重建信号”，
还不能支持“几何可靠”或“可以安全融回在线 map”。

另一个 221-frame、100-uniform-step 的实际 Unblur-SLAM smoke 中，TURTLE 输出被
采用 `81/221` 帧，并增加两个 tracking keyframe（总数 `11 -> 13`）。相对相同
预算 baseline，clear-GT 结果为：

- PSNR `22.076 -> 22.016 dB`；
- SSIM `0.8477 -> 0.8251`；
- LPIPS `0.1790 -> 0.1967`；
- full-trajectory ATE RMSE `0.001859 -> 0.001874 m`。

因此官方 GoPro zero-shot 权重已经正确接入并真实生效，但本段 TUM smoke 没有证明
SLAM 更稳。下一步应在不读取 TUM 测试结果调参的前提下，使用 ReplicaBlurry 训练集
微调官方 TURTLE，并增加固定关键帧、相同在线计算量的纯前端消融；不能把当前负结果
包装成提升。

## 官方接口不支持的声明

官方 `forward_update` 依赖同一次 initializer 产生的 condition features，以及与
context latent 网格一一对应的 Gaussian 拓扑。Unblur-SLAM 经过 densify/prune
后的任意点数 Gaussian map 不满足该合同。因此，官方模型不能原样执行
“先完成 Unblur-SLAM 26K，再直接细化这份现有 Gaussian state”。强行转换或训练
adapter 将成为新的自研方法，不能继续称为官方模型原样使用。

官方代码也没有持久化的 streaming-SLAM/add-view map API。可忠实实现的在线方向
是：每关闭一个固定 context 的 submap，异步运行一次完整官方 ReSplat sidecar；
把其 native map 融回活跃 Unblur map 仍需要新的坐标、拓扑、去重和 optimizer
研究，当前没有完成。

## 解释限制

早期相机不一致的 mixed-input probe 只保留作接口记录，不再进入质量结论。当前
TURTLE 相机一致 smoke 在 frontend stream 与 post-hoc clear-GT 上都观察到
refine4 相比 init0 的 RGB 指标改善，但 native PLY 仍有大位移离群点。因此当前可以
说官方 recurrent path 已正确接通、且固定子图上存在渲染质量信号；仍不能声称 TUM
几何可靠、能够安全融回活跃 map，或已经证明正式 26K 加速成立。
