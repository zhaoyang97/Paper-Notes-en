---
title: >-
  [Paper Note] Shape of Motion: 4D Reconstruction from a Single Video
description: >-
  [ICCV 2025][3D Vision][4D reconstruction] This paper proposes a dynamic 3D Gaussian representation based on $\mathbb{SE}(3)$ motion bases…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "4D reconstruction"
  - "dynamic scenes"
  - "3D Gaussian splatting"
  - "SE(3) motion basis"
  - "monocular video"
date: 2026-05-08
content_hash: 0a88a64809ebee64
---

# Shape of Motion: 4D Reconstruction from a Single Video

**Conference**: ICCV 2025  
**arXiv**: [2407.13764](https://arxiv.org/abs/2407.13764)  
**Code**: [Project Page](https://shape-of-motion.github.io/)  
**Area**: 3D Vision  
**Keywords**: 4D reconstruction, dynamic scenes, 3D Gaussian splatting, SE(3) motion basis, monocular video

## TL;DR

This paper proposes a dynamic 3D Gaussian representation based on $\mathbb{SE}(3)$ motion bases, recovering globally consistent 3D motion trajectories from monocular video while simultaneously enabling real-time novel view synthesis and long-range 3D tracking, outperforming prior methods comprehensively on the iPhone and Kubric datasets.

## Background & Motivation

Reconstructing dynamic 3D scenes from monocular video is a long-standing challenge in vision, severely underconstrained since moving objects are observed from only a single viewpoint at each moment.

Limitations of existing methods:

**Multi-view methods**: Require synchronized multi-camera rigs or LiDAR sensors, limiting practical applicability.

**Short-range scene flow**: Only computes 3D motion between consecutive frames (e.g., DynIBaR), failing to capture long-range 3D trajectories across the video.

**Deformation field methods**: Such as HyperNeRF and Deformable-3DGS, model motion via canonical-to-observation-space mappings, but struggle with large-motion scenarios.

**Frame-space 3D tracking**: Such as DELTA and SpatialTracker, predict motion in per-frame coordinate systems, entangling object and camera motion.

**Two core insights**:
- **Low-dimensional structure of motion**: Image-space dynamics may be complex and discontinuous, but the underlying 3D motion is a composition of continuous, simple rigid motions.
- **Complementarity of data-driven priors**: Monocular depth estimation and long-range 2D tracking provide complementary yet noisy cues that can be fused into a globally consistent representation.

## Method

### Overall Architecture

The input consists of an RGB video, camera parameters, monocular depth maps, and long-range 2D tracked points. The method optimizes a dynamic 3D Gaussian scene representation, decomposing the scene into **static Gaussians** (standard 3DGS) and **dynamic Gaussians** (with motion trajectories), jointly optimized and rendered.

### Key Designs

1. **$\mathbb{SE}(3)$ Motion Basis Representation**:

    - Defines $B \ll N$ globally shared basis trajectories $\{\mathbf{T}_{0 \to t}^{(b)}\}_{b=1}^B$ (with $B=10$ in experiments).
    - The pose transformation of each Gaussian is expressed as a weighted combination:
    $\mathbf{T}_{0 \to t} = \sum_{b=0}^{B} \mathbf{w}^{(b)} \mathbf{T}_{0 \to t}^{(b)}, \quad \|\mathbf{w}^{(b)}\|=1$
    - Motion is parameterized via 6D rotation and translation, each combined with the same weights.
    - **Design Motivation**: The compact motion basis explicitly regularizes trajectories into a low-dimensional structure, encouraging Gaussians with similar motion to share similar coefficients—equivalent to a **soft rigid motion decomposition** of the scene.

2. **Trajectory Rasterization**:

    - World-space 3D positions at time $t'$ are rendered per pixel via alpha compositing:
    ${}^{w}\hat{\mathbf{X}}_{t \to t'}(\mathbf{p}) = \sum_{i \in H(\mathbf{p})} T_i \alpha_i \boldsymbol{\mu}_{i,t'}$
    - Projection to 2D yields 2D correspondences; extracting the third component yields reprojection depth.
    - This enables complete 3D trajectories from any pixel in any query frame to any target frame.

3. **Initialization Strategy**:

    - The canonical frame $t_0$ is selected as the frame with the most visible 3D tracked points.
    - K-means clustering is applied to velocity vectors of noisy 3D tracks to initialize $B$ motion bases from $B$ clusters.
    - Within each cluster, weighted Procrustes alignment initializes the basis transformations.
    - The model is first optimized for 1000 steps to fit 3D tracking observations before entering the main training phase.

4. **Data-Driven Prior Fusion**:

    - **Depth prior**: Depth Anything estimates relative depth, aligned to metric scale for supervision.
    - **2D tracking prior**: TAPIR provides long-range 2D tracks for foreground regions.
    - **Motion segmentation**: Track-Anything generates foreground masks with minimal user clicks.
    - **Camera poses**: MegaSaM estimates initial camera parameters, jointly optimized during training.

### Loss & Training

**Reconstruction loss (per-frame)**:
$$L_{recon} = \|\hat{\mathbf{I}} - \mathbf{I}\|_1 + \lambda_{depth}\|\hat{\mathbf{D}} - \mathbf{D}\|_1 + \lambda_{mask}\|\hat{\mathbf{M}} - \mathbf{M}\|_1$$

**Correspondence loss (cross-frame)**:
- 2D tracking loss: $L_{track-2d} = \|\mathbf{U}_{t \to t'} - \hat{\mathbf{U}}_{t \to t'}\|_1$
- Tracking depth loss: $L_{track-depth} = \|\hat{\mathbf{d}}_{t \to t'} - \hat{\mathbf{D}}(\mathbf{U}_{t \to t'})\|_1$

**Physical prior**:
$$L_{rigidity} = \|dist(\hat{\mathbf{X}}_t, \mathcal{C}_k(\hat{\mathbf{X}}_t)) - dist(\hat{\mathbf{X}}_{t'}, \mathcal{C}_k(\hat{\mathbf{X}}_{t'}))\|_2^2$$
A distance-preservation loss that constrains k-nearest-neighbor distances to remain stable across time, reinforcing the rigid motion prior.

Training runs for 500 epochs with the Adam optimizer, initialized with 40k dynamic and 100k static Gaussians. A 300-frame video requires approximately 2 hours on an A100; rendering speed is ~140 fps.

## Key Experimental Results

### Main Results

Full-task evaluation on the iPhone dataset (14 sequences):

| Method | 3D EPE↓ | $\delta_{3D}^{.05}$↑ | AJ↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|---------|----------------------|-----|-------|-------|--------|
| HyperNeRF | 0.182 | 28.4 | 10.1 | 15.99 | 0.59 | 0.51 |
| D-3DGS | 0.151 | 33.4 | 14.0 | 11.92 | 0.49 | 0.66 |
| TAPIR+DA | 0.114 | 38.1 | 27.8 | - | - | - |
| SpatialTracker | 0.125 | 37.7 | 24.9 | - | - | - |
| **Ours** | **0.082** | **43.0** | **34.4** | **16.72** | **0.63** | **0.45** |

- **The only method to achieve state-of-the-art performance simultaneously on 3D tracking, 2D tracking, and novel view synthesis.**
- 3D EPE is reduced by 28% relative to the strongest tracking baseline (TAPIR+DA); AJ improves by +6.6.
- NVS PSNR surpasses all dynamic reconstruction baselines.

3D tracking on Kubric dataset:

| Method | EPE↓ | $\delta_{3D}^{.05}$↑ | $\delta_{3D}^{.10}$↑ |
|------|------|----------------------|----------------------|
| CoTracker+DA | 0.19 | 34.4 | 56.5 |
| TAPIR+DA | 0.20 | 34.0 | 56.2 |
| **Ours** | **0.16** | **39.8** | **62.2** |

### Ablation Study

Ablation of key components (iPhone dataset, 3D tracking):

| Configuration | EPE↓ | $\delta_{3D}^{.05}$↑ | Note |
|------|------|----------------------|------|
| w/o motion basis (independent trajectories) | ~0.12 | ~35 | Missing low-dimensional regularization |
| w/o rigidity loss | degraded | degraded | Physical prior is critical |
| w/o depth supervision | degraded | degraded | Monocular constraints insufficient |
| **Full model** | **0.082** | **43.0** | - |

### Key Findings

- PCA visualization of motion bases correlates strongly with rigid motion groups (e.g., rotating blades appear as a uniform color).
- Lifting noisy 2D tracks and depth to 3D tracks performs substantially worse than the proposed global fusion, highlighting the importance of joint optimization.
- As few as 10 motion bases suffice to represent complex scene motion.

## Highlights & Insights

- **Unified framework**: The first approach to simultaneously achieve long-range 3D tracking and high-quality NVS within a single representation.
- **"Shape" of motion**: The geometric patterns of 3D trajectories themselves convey rich motion semantic information.
- **Power of low-dimensional motion assumption**: Just $B=10$ SE(3) bases suffice to express complex scene motion, naturally providing motion segmentation capability.
- **Real-time rendering**: A rendering speed of 140 fps makes the method suitable for interactive applications.

## Limitations & Future Work

- Foreground mask annotation requires manual input, though only a few clicks are needed.
- The initialization of noisy 3D tracks depends on the quality of TAPIR and Depth Anything.
- Training time is approximately 2 hours per sequence, precluding real-time processing.
- The number of motion bases is fixed at 10; more complex scenes may require adaptive selection.

## Related Work & Insights

- **DynMF**: Also employs motion bases but parameterizes them with neural networks; the explicit SE(3) formulation in this paper is more interpretable.
- **TAPIR/CoTracker**: Powerful 2D tracking priors, but lack 3D understanding.
- **Depth Anything**: A key source of monocular depth priors.
- **MegaSaM**: Estimates camera parameters from monocular video.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (SE(3) motion basis + multi-prior fusion)
- Technical Depth: ⭐⭐⭐⭐⭐ (complete initialization–optimization pipeline, multi-task unification)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (comprehensive evaluation across three datasets and three tasks)
- Value: ⭐⭐⭐⭐ (real-time rendering, but training is slow)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](../../CVPR2026/3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[ICCV 2025\] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)
- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](../../NeurIPS2025/3d_vision/orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)
- [\[ICCV 2025\] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)

</div>

<!-- RELATED:END -->
