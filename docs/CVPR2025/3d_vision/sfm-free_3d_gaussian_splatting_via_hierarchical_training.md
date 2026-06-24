---
title: >-
  [Paper Note] SfM-Free 3D Gaussian Splatting via Hierarchical Training
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] Proposes an SfM-free 3DGS method (SFGS) that merges multiple local 3DGS models into a unified scene representation through a hierarchical training strategy, and utilizes video frame interpolation to improve camera pose estimation, achieving a 2.25dB PSNR improvement on Tanks and Temples.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "SfM-free reconstruction"
  - "hierarchical training"
  - "video frame interpolation"
  - "novel view synthesis"
date: 2026-05-08
content_hash: 2e12251fc365ec63
---

# SfM-Free 3D Gaussian Splatting via Hierarchical Training

**Conference**: CVPR 2025  
**arXiv**: [2412.01553](https://arxiv.org/abs/2412.01553)  
**Code**: [GitHub](https://github.com/jibo27/3DGS_Hierarchical_Training)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, SfM-free reconstruction, hierarchical training, video frame interpolation, novel view synthesis

## TL;DR

Proposes an SfM-free 3DGS method (SFGS) that merges multiple local 3DGS models into a unified scene representation through a hierarchical training strategy, and utilizes video frame interpolation to improve camera pose estimation, achieving a 2.25dB PSNR improvement on Tanks and Temples.

## Background & Motivation

Standard 3DGS relies on SfM preprocessing to obtain camera poses and sparse point clouds for initialization, but SfM faces the following issues:
- **Time-consuming**: Large-scale scenes require substantial calculation time for feature matching and triangulation.
- **Insufficient robustness**: It easily fails in regions with repetitive patterns, textureless areas, or incorrect feature extraction.
- **Non-differentiable**: It limits the potential for end-to-end learning.

Existing SfM-free methods, such as CF-3DGS, estimate relative poses via affine transformation, but suffer from two core limitations:
1. **Insufficient pose estimation accuracy**: During large camera motions, reduced inter-frame overlap leads to accumulated pose errors.
2. **Sparse Gaussian coverage**: Initializing point clouds solely with the first frame's depth map results in a lack of Gaussians in uncovered areas of the scene. Standard adaptive density control struggles to effectively densify these sparse regions.

The core insight of this work is to train the complex scene in segments, merge them hierarchically, and smooth large motion estimations using video frame interpolation.

## Method

### Overall Architecture

The input is a video sequence with small camera motion $\mathcal{I}=\{I_i\}_{i=1}^N$, and the method consists of three stages:
1. **Camera Pose Estimation**: Obtain global poses by stacking relative poses of consecutive frame pairs, and leverage a VFI model to interpolate intermediate frames, reducing the estimation error of large motions.
2. **Hierarchical Training**: Segment the video into multiple overlapping clips, train a base 3DGS model for each clip, and then iteratively merge them into a unified model using an importance-score-based pruning and merging strategy.
3. **Multi-Source Supervision**: After merging, perform multi-source training using the original frames, pseudo-views from base models, and VFI-interpolated frames to alleviate overfitting.

### Key Design 1: Video Frame Interpolation Assisted Pose Estimation

- **Function**: Address the problem of inaccurate relative pose estimation under large camera motions.
- **Mechanism**: Utilize a pre-trained video frame interpolation (VFI) model (e.g., EMA-VFI) to generate an intermediate frame $I_{i+0.5}$ between consecutive frames $I_i$ and $I_{i+1}$, decomposing a single step of large motion into two steps of small motion. The relative pose becomes $T_{i \to i+1} = T_{i \to i+0.5} \odot T_{i+0.5 \to i+1}$.
- **Design Motivation**: Large camera motions lead to low overlap between frames, which causes severe rendering artifacts when a single-frame 3DGS model renders targets. Interpolated frames reduce the single-step motion scale, significantly mitigating rendering artifacts and improving pose estimation accuracy. This strategy yields a 0.35dB PSNR improvement on Tanks and Temples.

### Key Design 2: Hierarchical Training and Merging Strategy

- **Function**: Address the core problem of insufficient Gaussian coverage in distant regions of the scene caused by single-frame initialization.
- **Mechanism**: Set the hierarchy level to $L$ and divide the video evenly into $2^L$ overlapping segments. Each segment is trained independently as a base 3DGS model, and then iteratively merged in pairs until a unified model is obtained. Prior to merging, an importance score (based on rendering gradient sensitivity) is computed for each model's Gaussians. The top $\gamma$% of Gaussians are retained, followed by a union operation.
- **Design Motivation**: Standard adaptive density control relies on accumulated gradients to clone or split Gaussians, but in sparse regions, the gradients are too small to trigger densification. The proposed method reinterprets the merging process itself as a densification process: pruning unimportant Gaussians while introducing key Gaussians from other models to fill the gaps. Globally, $L=2$ and $\gamma=50\%$ yield the best performance, bringing an improvement of 1.19-1.58dB on Tanks and Temples.

### Key Design 3: Multi-Source Supervised Training

- **Function**: Prevent the merged model from overfitting to a limited set of training frames.
- **Mechanism**: After merging, train with three sources of data: (1) original training frames; (2) pseudo-views rendered by base 3DGS models at virtual intermediate views (with camera poses acquired via SE(3) spatial interpolation); (3) VFI-generated interpolated frames. During training, there is a 50% probability of selecting either pseudo-views or interpolated frames.
- **Design Motivation**: The merged 3DGS requires fine-tuning to fuse Gaussians from different models, but training strictly on original frames triggers overfitting. Pseudo-views and interpolated frames provide denser novel viewpoint supervision, boosting generalization capacity.

### Loss & Training

Standard 3DGS photometric loss: $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$, applied to all training, pseudo-rendered, and interpolated frames.

## Key Experimental Results

### Main Results: Tanks and Temples Novel View Synthesis

| Method | Church | Barn | Museum | Family | Horse | Ballroom | Francis | Ignatius | **Mean PSNR** |
|------|--------|------|--------|--------|-------|----------|---------|----------|---------------|
| BARF | 23.17 | 25.28 | 23.58 | 23.04 | 24.09 | 20.66 | 25.85 | 21.78 | 23.42 |
| Nope-NeRF | 25.17 | 26.35 | 26.77 | 26.01 | 27.64 | 25.33 | 29.48 | 23.96 | 26.34 |
| CF-3DGS | 30.23 | 31.23 | 29.91 | 31.27 | 33.94 | 32.47 | 32.72 | 28.43 | 31.28 |
| **Ours** | **31.34** | **34.95** | **31.59** | **34.71** | **35.82** | **34.12** | **34.09** | **31.64** | **33.53** |

Average PSNR increases by **+2.25dB**, with a maximum gain of 3.72dB on the Barn scene.

### Ablation Study: Component Contributions

| Settings | PSNR | Gain |
|------|------|------|
| Baseline (w/o hierarchy / w/o VFI) | ~31.28 | - |
| + Video Frame Interpolation (VFI) | +0.35dB | VFI improves pose estimation |
| + Hierarchical Training (L=2) | +1.19~1.58dB | Solves insufficient Gaussian coverage |
| + Multi-source Supervision | Extra Gain | Reduces overfitting |
| Progressive vs. Hierarchical | Both achieve ≥1.32dB | Hierarchical is slightly better |

### Key Findings

- The average PSNR improves by 1.74dB on the CO3D-V2 dataset, with the best-performing scene gaining 3.90dB.
- Even when camera intrinsics are unknown, the proposed method still outperforms the state-of-the-art by 0.89dB PSNR.
- Hierarchical training yields more evenly distributed Gaussians, effectively covering all regions of the scene.

## Highlights & Insights

1. **Merging as Densification**: Reinterprets model merging as a high-level Gaussian densification strategy, resolving the issue where standard density control fails in sparse regions.
2. **Innovative Application of VFI**: Leverages 2D video generation models to assist pose estimation in 3D reconstruction, presenting a valuable cross-domain methodology transfer.
3. **Simple yet Effective Importance Pruning**: The gradient-based importance scoring is simple and intuitive. The pruning and union-based merging strategy circumvents complex calculations for 3D Gaussian correspondences.

## Limitations & Future Work

- Relies on video inputs assuming small camera motions between adjacent frames, making it inapplicable to unstructured image collections.
- Hierarchical training increases overall training time, as multiple base models must be trained before merging.
- Pose estimation is still approximated via affine transformation, which lacks theoretical rigor.
- Future work could integrate stronger monocular depth estimators or methods like DUSt3R to achieve further performance gains.

## Related Work & Insights

- **CF-3DGS**: Direct baseline of this work, which performs SfM-free 3DGS training through affine transformations.
- **InstantSplat / COGS**: SfM-free methods designed for sparse views, which are complementary to the video-based setting of this work.
- **3DGS Compression**: The importance scoring borrows parameter sensitivity analysis concepts from compression methods like LightGaussian.

## Rating

⭐⭐⭐⭐ — The methodology is highly intuitive. Hierarchical training serves as an elegant solution to the lack of Gaussian coverage in SfM-free settings, leading to significant experimental improvements (+2.25dB). The main limitations lie in the increased training overhead and the strict requirement for video inputs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Hash3D: Training-free Acceleration for 3D Generation](hash3d_training-free_acceleration_for_3d_generation.md)
- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)

</div>

<!-- RELATED:END -->
