# <img src="https://github.com/user-attachments/assets/e508ba2c-59e4-4f43-a640-d80c9bd0102c" width="50" alt="logo" align="center" /> Unblur-SLAM: Dense Neural SLAM for Blurry Inputs

<img width="532" height="270" alt="blur_teaser_3D" src="https://github.com/user-attachments/assets/f8d12d59-20fb-4645-a79a-f0ef62d7dd42" />

Welcome to the official repository for **Unblur-SLAM**, a novel RGB SLAM pipeline designed for sharp 3D reconstruction from blurred image inputs.

📄 **Paper:** [Unblur-SLAM (arXiv)](https://arxiv.org/pdf/2603.26810)

> Qi Zhang, Denis Rozumny, Francesco Girlanda, Sezer Karaoglu, Marc Pollefeys, Theo Gevers, Martin R. Oswald.
> *Unblur-SLAM: Dense Neural SLAM for Blurry Inputs.* CVPR 2026.

## 📖 Overview
<img width="700" height="266" alt="unblur-slam-overview" src="https://github.com/user-attachments/assets/663d569b-5269-4e7d-b488-709e16c2b130" />

In contrast to previous work, Unblur-SLAM is capable of handling different types of blur and demonstrates state-of-the-art performance in the presence of both motion blur and defocus blur.

Our system intelligently adjusts its computational effort based on the amount of blur detected in the input image. By treating sharp and blurry frames separately and skipping costly refinements for sharp frames, it avoids the significant slowdowns typical of previous blur-aware SLAM approaches.

## 🚀 Release Plan

### TODO List
- [x] **Phase 1:** Open-source the pre-trained model weights and the curated datasets.
  - 🏋️ **Pre-trained Models:** Available on [Hugging Face](https://huggingface.co/qizhangslam/Unblur-SLAM-checkpoints)
  - 🗄️ **Curated Datasets:** Available on [Hugging Face](https://huggingface.co/datasets/qizhangslam/Unblur_slam_traning_dataset)
- [ ] **Phase 2:** Open-source the training code for the deblurring model.
- [x] **Phase 3:** Open-source the inference code of the whole system. *(this commit)*

Please star or watch this repository to stay updated on our progress!

## 🛠️ Installation

1. Clone the repository.
   ```bash
   git clone https://github.com/SlamMate/Unblur-SLAM.git
   cd Unblur-SLAM
   ```

2. Create a conda environment.
   ```bash
   conda create --name unblur-slam python=3.10 -y
   conda activate unblur-slam
   ```

3. Install CUDA toolkit and PyTorch.
   ```bash
   conda install conda-forge::cudatoolkit-dev=11.7.0 -y
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
   python -c "import torch; print('cuda:', torch.cuda.is_available())"
   ```

4. **Patch the Gaussian rasterizer near plane.** In our monocular setting the global scale is ambiguous, so we lower the rasterizer's near plane from `0.2` to `0.001`. Edit `thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/auxiliary.h:154` so the line reads:
   ```c
   if (p_view.z <= 0.001f)
   ```

5. Install the in-tree extensions.
   ```bash
   python -m pip install -e thirdparty/lietorch/
   python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose/
   python -m pip install -e thirdparty/simple-knn/
   python -m pip install -e thirdparty/evaluate_3d_reconstruction_lib/
   ```

6. Build the DROID backends and install Python requirements.
   ```bash
   python -m pip install -e .
   python -m pip install -r requirements.txt
   python -m pip install pytorch-lightning==1.9 --no-deps
   ```

7. Sanity check.
   ```bash
   python -c "import torch, lietorch, simple_knn, diff_gaussian_rasterization; print(torch.cuda.is_available())"
   ```

8. Download pretrained weights into `./pretrained/`.
   ```text
   pretrained/
   ├── droid.pth                       # DROID-SLAM tracker (Splat-SLAM Drive bundle)
   ├── omnidata_dpt_depth_v2.ckpt      # Omnidata monocular depth (same bundle)
   └── evssm/
       ├── net_g_latest.pth            # EVSSM deblurring weights (motion + defocus)
       └── net_g_realblur_j.pth        # optional: RealBlur_J variant
   ```
   The `droid.pth` and `omnidata_dpt_depth_v2.ckpt` come from the original Splat-SLAM [Google Drive bundle](https://drive.google.com/file/d/1oZbVPrubtaIUjRRuT8F-YjjHBW-1spKT/view?usp=drive_link); the EVSSM checkpoints come from our [Hugging Face checkpoints repo](https://huggingface.co/qizhangslam/Unblur-SLAM-checkpoints).

## 🗄️ Datasets

Place every dataset under `./datasets/` (or symlink). Defaults in the config files assume this layout — change `data.dataset_root` / `data.input_folder` if you store data elsewhere.

### Deblur-NeRF (motion blur and defocus blur)
We follow the layout of Ma et al. (2022) — see the [Deblur-NeRF release](https://github.com/limacv/Deblur-NeRF). Place the two subsets at:
```
datasets/real_camera_motion_blur/<scene>/    # motion blur, 10 scenes (blurball, blurbasket, ...)
datasets/real_defocus_blur/<scene>/          # defocus blur, 11 scenes (defocusbush, defocuscake, ...)
```

### I2-SLAM evaluation on TUM-RGBD
The I2-SLAM rendering benchmark uses TUM-RGBD `fr1_desk`, `fr2_xyz`, and `fr3_office` with the I2-SLAM keyframe annotations. Download the TUM sequences with:
```bash
bash scripts/download_tum.sh
```
Then place them at `datasets/tum/rgbd_dataset_freiburg{1_desk,2_xyz,3_long_office_household}/`.

### Replica, ScanNet, ReplicaBlurry, MCD, ArchViz, exblurf
Configs for these auxiliary benchmarks live under `configs/Replica/`, `configs/Scannet/`, `configs/ReplicaBlurry/`, `configs/MCD/`, and `configs/exblurf_motion/`. Helper download scripts are in `scripts/` (Replica, TUM, ScanNet).

## ▶️ Run

Each scene has its own config; the inference entry point is `run.py <config>`.

### Deblur-NeRF — motion blur (paper Tab. 5)
```bash
python run.py configs/deblur_nerf_motion/blurball.yaml
python run.py configs/deblur_nerf_motion/blurbasket.yaml
python run.py configs/deblur_nerf_motion/blurbuick.yaml
python run.py configs/deblur_nerf_motion/blurcoffee.yaml
python run.py configs/deblur_nerf_motion/blurdecoration.yaml
python run.py configs/deblur_nerf_motion/blurgirl.yaml
python run.py configs/deblur_nerf_motion/blurheron.yaml
python run.py configs/deblur_nerf_motion/blurparterre.yaml
python run.py configs/deblur_nerf_motion/blurpuppet.yaml
python run.py configs/deblur_nerf_motion/blurstair.yaml
```
Or sweep all 10 with `bash run_all_deblur_nerf_motion.sh`.

### Deblur-NeRF — defocus blur (paper Tab. 4)
```bash
python run.py configs/deblur_nerf_defocus/defocusbush.yaml
# ...same pattern for defocuscake, defocuscaps, defocuscisco, defocuscoral,
# defocuscupcake, defocuscups, defocusdaisy, defocussausage, defocusseal, defocustools
```

### I2-SLAM rendering benchmark on TUM (paper Tab. 6)
```bash
python run.py configs/I2slam/freiburg1_desk.yaml
python run.py configs/I2slam/freiburg2_xyz.yaml
python run.py configs/I2slam/freiburg3_office.yaml
```

### Tracking-only mode
Append `--only_tracking` to skip mapping/rendering and only produce the camera trajectory:
```bash
python run.py configs/I2slam/freiburg3_office.yaml --only_tracking
```

## 🎯 Reproducing paper numbers

Reference numbers from the camera-ready Unblur-SLAM paper:

| Benchmark | Metric | Target |
|---|---|---|
| Deblur-NeRF motion blur (Tab. 5) | PSNR / SSIM / LPIPS | **29.49** / **0.9213** / **0.0728** |
| Deblur-NeRF defocus blur (Tab. 4) | PSNR | **27.45** |
| TUM I2-SLAM (Tab. 6) — fr1_desk / fr2_xyz / fr3_office | PSNR | 28.03 / 31.14 / 29.22 |
| TUM tracking (Tab. 3, mean over 19 sequences) | ATE RMSE [m] | **0.336** |
| MCD tracking (Tab. 3, mean over 57 sequences) | ATE RMSE [m] | **0.128** |

The configs in this repository ship with the exact hyperparameters used to produce those numbers (kernel sizes `(3, 5, 9, 3)`, 7 virtual sub-frames for motion blur on Deblur-NeRF, sharp-loss weight `2.0` on Deblur-NeRF and `1.1` on the I2-SLAM TUM split, `mlp_lr=5e-5` for motion / `5e-6` for defocus, DSPO bundle adjustment with loop closure for Deblur-NeRF and DBA for TUM). Hardware in the paper: AMD EPYC-2 7282 + RTX A6000 (48 GB).

### Verified end-to-end run (RTX 3090, 24 GB)

We re-ran `configs/deblur_nerf_motion/blurball.yaml` from this exact commit on an RTX 3090 (`ivi-cn002`, 65 min wall-clock, peak 11.7 GB GPU memory):

| Stage | PSNR | SSIM | LPIPS |
|---|---|---|---|
| before final refine | 28.39 | 0.851 | 0.161 |
| **after 26 000-iter refine** | **29.85** | **0.892** | **0.114** |

`blurball` is one of the easier scenes in the Deblur-NeRF motion-blur subset, so its PSNR is slightly above the 10-scene average reported in the paper (29.49). The full ATE/PSNR sweep across all 10 motion-blur scenes (and the 11 defocus scenes) needs the rest of the Deblur-NeRF data placed under `./datasets/`.

### Hardware notes — running on 24 GB GPUs

The paper used a 48 GB A6000. On 24 GB cards (RTX 3090 / A5000 / Quadro RTX 6000) two extra knobs are required:

- Set `UNBLUR_SKIP_NR_IQA=1` (pre-set in `run_repro_i2slam.sbatch`) — skips the QAlign LLaMA-based IQA model in `eval_utils.py`. PSNR/SSIM/LPIPS are still computed.
- For the longer TUM/I2-SLAM sequences (`fr1_desk`, `fr2_xyz`, `fr3_office`), the multi-resolution BPN kernel state grows linearly in the number of keyframes and exceeds 24 GB. Use `configs/I2slam/freiburg1_desk_24gb.yaml` (smaller mapping `window_size`) as a starting point, or run with reduced `n_virtual_cams` / `final_refine_iters`. With these knobs you can complete the run; the numbers will not match the paper exactly because the multi-scale refinement budget changes.

### Reproducing the cluster job

```bash
sbatch --gres=gpu:rtx_3090:1 run_repro_i2slam.sbatch configs/deblur_nerf_motion/blurball.yaml
```

> **Note on hardware variance:** As with most CUDA-based SLAM systems, exact metrics can drift slightly across GPU generations even with a fixed seed. If your numbers differ at the second decimal, that is expected.

## 🔬 Pipeline at a glance
- **Blur quantification** with ARNIQA classifies each frame as sharp / blurry-success / blurry-fail.
- **Sharp & blurry-success frames** are tracked with DROID-SLAM and then refined through deformable 3DGS, multi-scale BPN kernels (sizes 3 / 5 / 9 / 3), and exposure compensation (Sec. 3.6 of the paper).
- **Blurry-fail frames** are modeled with `n_virtual_cams` sub-frame poses inside the rasterizer to invert motion-blur formation (Eq. 1 of the paper).
- **Global consistency** comes from DSPO/DBA local bundle adjustment, loop closure detection, and a final-stage global BA + multi-scale refinement (`mapping.final_refine_iters: 26000`).

## ⚠️ Pre-trained Model Loading & Limitations
The pre-trained models can be loaded directly by referring to the [EVSSM repository](https://github.com/kkkls/EVSSM).

For RGB images that have been processed and enhanced by smartphone AI algorithms (computational photography), our algorithm cannot invert these non-linear enhancements to recover the linear RGB values required for fixed-timestamp deblurring.

## 🙏 Acknowledgements
Our codebase builds on [Splat-SLAM](https://github.com/google-research/Splat-SLAM), [GlORIE-SLAM](https://github.com/zhangganlin/GlORIE-SLAM), [GO-SLAM](https://github.com/youmi-zym/GO-SLAM), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [MonoGS](https://github.com/muskie82/MonoGS), [BAGS](https://github.com/peng-zhou/BAGS), and [EVSSM](https://github.com/kkkls/EVSSM). We thank the authors of [I2-SLAM](https://github.com/Bae-Jiseong/I2-SLAM) for sharing manually annotated TUM keyframes for fair PSNR comparison, and the authors of [MBA-SLAM](https://github.com/WU-CVGL/MBA-SLAM) for sharing reconstructions on `fr1_desk`. None of this work would have been possible without those efforts.

## 📝 Citation
If you find our work or datasets helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{unblur_slam_2026,
  title={Unblur-SLAM: Dense Neural SLAM for Blurry Inputs},
  author={Zhang, Qi and Rozumny, Denis and Girlanda, Francesco and Karaoglu, Sezer and Pollefeys, Marc and Gevers, Theo and Oswald, Martin R.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## 📬 Contact
Open an issue on this repository, or reach Qi Zhang at <q.zhang@uva.nl> for questions and bug reports.
