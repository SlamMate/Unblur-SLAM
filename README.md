# <img src="https://github.com/user-attachments/assets/e508ba2c-59e4-4f43-a640-d80c9bd0102c" width="50" alt="logo" align="center" /> Unblur-SLAM: Dense Neural SLAM for Blurry Inputs
<img width="532" height="270" alt="blur_teaser_3D" src="https://github.com/user-attachments/assets/f8d12d59-20fb-4645-a79a-f0ef62d7dd42" />



Welcome to the official repository for **Unblur-SLAM**, a novel RGB SLAM pipeline designed for sharp 3D reconstruction from blurred image inputs. 

## 📖 Overview
<img width="700" height="266" alt="unblur-slam-overview" src="https://github.com/user-attachments/assets/663d569b-5269-4e7d-b488-709e16c2b130" />


In contrast to previous work, Unblur-SLAM is capable of handling different types of blur and demonstrates state-of-the-art performance in the presence of both motion blur and defocus blur. 

Our system intelligently adjusts its computational effort based on the amount of blur detected in the input image. By treating sharp and blurry frames separately and skipping costly refinements for sharp frames, it avoids the significant slowdowns typical of previous blur-aware SLAM approaches.

## 🚀 Release Plan
The codebase is currently undergoing further refinement and cleanup to ensure it is robust and easy to use. We are fully committed to open-sourcing our work and will release the assets according to the following schedule:

### TODO List
- [x] **Phase 1:** Open-source the pre-trained model weights and the curated datasets.
  - 🏋️ **Pre-trained Models:** Available on [Hugging Face](https://huggingface.co/qizhangslam/Unblur-SLAM-checkpoints)
  - 🗄️ **Curated Datasets:** Available on [Hugging Face](https://huggingface.co/datasets/qizhangslam/Unblur_slam_traning_dataset)
- [ ] **Phase 2:** Open-source the complete Unblur-SLAM codebase.

Please star or watch this repository to stay updated on our progress!
