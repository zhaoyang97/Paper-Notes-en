---
title: >-
  [Paper Note] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving
description: >-
  [ICCV 2025][Autonomous Driving][Autonomous driving scene rendering] This paper proposes AD-GS, a self-supervised autonomous driving scene rendering framework that models dynamic object motion by combining locally-aware l…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Autonomous driving scene rendering"
  - "Gaussian splatting"
  - "B-spline curves"
  - "self-supervised learning"
  - "dynamic scene reconstruction"
date: 2026-05-08
content_hash: f47efa1bc27692bf
---

# AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving

**Conference**: ICCV 2025
**arXiv**: N/A (CVF Open Access)
**Code**: [https://jiaweixu8.github.io/AD-GS-web/](https://jiaweixu8.github.io/AD-GS-web/)
**Area**: Autonomous Driving
**Keywords**: Autonomous driving scene rendering, Gaussian splatting, B-spline curves, self-supervised learning, dynamic scene reconstruction

## TL;DR

This paper proposes AD-GS, a self-supervised autonomous driving scene rendering framework that models dynamic object motion by combining locally-aware learnable B-spline curves with globally-aware trigonometric functions. It employs simplified pseudo 2D segmentation for scene decomposition, significantly outperforming existing self-supervised methods and approaching the performance of annotation-dependent approaches without relying on manual 3D annotations.

## Background & Motivation

1. **Background**: Autonomous driving scene rendering aims to reconstruct dynamic driving environments from LiDAR point clouds and multi-camera images, supporting novel-view and novel-time rendering. Mainstream methods fall into two categories: (1) annotation-dependent methods such as 4DGF and StreetGS, which rely on manual 3D annotations (object bounding boxes and trajectories) and achieve high quality at a significant labeling cost; (2) self-supervised methods, which require no annotations but suffer from a notable performance gap.

2. **Limitations of Prior Work**: Self-supervised methods are deficient in both motion modeling and scene decomposition. In motion modeling, neural network-based approaches incur high computational overhead and struggle to capture local motion details, while trigonometric function-based approaches are fast but their global fitting nature limits their expressiveness for local motion variations. In scene decomposition, methods relying on instance segmentation or feature-based supervision are prone to reconstruction artifacts under noisy pseudo-labels.

3. **Key Challenge**: Self-supervised methods must balance **global motion fitting stability** against **local motion detail accuracy**. Global trigonometric functions remain stable under noisy supervision but lack precision, whereas local fitting methods are accurate but prone to overfitting noise. Scene decomposition granularity similarly requires a trade-off between robustness and accuracy.

4. **Goal**: Design a motion representation that achieves both global stability and local accuracy, and propose a robust scene decomposition strategy that allows self-supervised methods to approach the rendering quality of annotation-dependent methods without relying on 3D labels.

5. **Key Insight**: B-spline curves inherently possess local control properties (each point on the curve is influenced only by nearby control points), while trigonometric functions have global fitting characteristics. Combining the two can simultaneously achieve local detail fitting and global motion capture. Scene decomposition requires only a simple object/background binary classification, avoiding the noise issues of fine-grained segmentation.

6. **Core Idea**: Dynamic Gaussian motion is modeled jointly by B-spline position curves, B-spline quaternion curves, and trigonometric functions, with bidirectional temporal visibility masks introduced to handle the sudden appearance and disappearance of objects.

## Method

### Overall Architecture

AD-GS is built upon 3D Gaussian Splatting and decomposes the scene into object and background components. Background Gaussians remain static, while object Gaussians undergo time-dependent positional and rotational deformation via B-spline curves and trigonometric functions. The inputs are LiDAR point clouds and multi-view image sequences; SAM is used to generate simplified binary object/background segmentation as pseudo-labels. Self-supervised training employs multiple pseudo-supervision signals including optical flow, monocular depth, segmentation, and sky masks.

### Key Designs

1. **Learnable B-Spline Motion Curves**:

    - **Function**: Accurately model local motion details of dynamic objects.
    - **Mechanism**: Uniform B-spline curves represent the temporal deformation of Gaussian positions: $\mu' = \mu + p(t) + \sum_{l=1}^L a_l \sin(t \cdot l\vartheta) + b_l \cos(t \cdot l\vartheta)$, where $p(t)$ is the B-spline curve. The key property of B-splines is local control — the curve value at each time point is determined by only the nearest $k$ control points, so that noise in a single frame affects only a local region rather than the global trajectory. A matrix formulation $p(t) = [1, u, u^2, ..., u^{k-1}] M_k [p_{i-k+1}, ..., p_i]^T$ replaces recursive computation for efficiency. Rotation is modeled directly via a B-spline quaternion curve $q(t)$.
    - **Design Motivation**: The global optimization nature of trigonometric functions provides stability under noisy self-supervision but fails to capture local motion variations such as hard braking or sharp turns. The local fitting property of B-splines provides a natural complement.

2. **Scene Decomposition via Simplified Pseudo 2D Segmentation**:

    - **Function**: Robustly decompose the scene into object and background components.
    - **Mechanism**: Rather than employing complex instance segmentation, all categories are simplified into "object" (potentially moving categories such as vehicles) and "background." SAM generates a binary segmentation $M_{obj}$, and LiDAR projection initializes Gaussian class assignments. During training, an object mask is rendered as $\hat{M}_{obj} = \sum_i \mathbb{I}\{G_i \in \Omega_{obj}\} \alpha_i T_i$, and a BCE loss $L_{obj}$ enforces that the two classes of Gaussians remain in their respective regions.
    - **Design Motivation**: Fine-grained instance segmentation (e.g., individual vehicle segmentation) is overly fragile under noisy pseudo-labels; reducing to binary classification substantially improves robustness.

3. **Bidirectional Temporal Visibility Mask**:

    - **Function**: Handle the sudden appearance and disappearance of dynamic objects in the sequence.
    - **Mechanism**: A temporal decay is applied to the opacity of each object Gaussian $G \in \Omega_{obj}$: $\omega'(t) = \omega \cdot e^{-(t-\mu_t)^2 / 2s^2}$, where $\mu_t$ is fixed to the LiDAR acquisition timestamp (serving as a prior for when the object is visible), and $s_0$ ($t < \mu_t$) and $s_1$ ($t \geq \mu_t$) are learnable forward and backward scale parameters. An expansion regularization $L_s = \|2\bar{f} / (s_0 + s_1)\|_1$ is introduced to prevent the mask from becoming overly narrow.
    - **Design Motivation**: In driving scenarios, vehicles enter and leave the field of view abruptly. Without temporal masking, Gaussians associated with departed vehicles continue to produce "ghost" artifacts in subsequent frames.

### Loss & Training

Total loss: $L = (1-\varsigma_c)L_1 + \varsigma_c L_{D-SSIM} + \varsigma_d L_d + \varsigma_f L_f + \varsigma_{obj} L_{obj} + \varsigma_{sky} L_{sky} + \varsigma_r L_r + \varsigma_s L_s$

- Image reconstruction loss: L1 + D-SSIM
- Inverse depth supervision $L_d$: pseudo-labels from DPTv2, aligned via scale-and-shift
- Optical flow supervision $L_f$: pseudo-labels from CoTracker3, applied only to object regions
- Physical rigid-body regularization $L_r$: variance constraint on deformation parameters within KNN neighborhoods

## Key Experimental Results

### Main Results

| Dataset | Method | Annotation | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|------|-------|-------|--------|
| KITTI-75% | 4DGF | Yes | 31.34 | 0.945 | 0.026 |
| KITTI-75% | PVG | No | 27.13 | 0.895 | 0.049 |
| KITTI-75% | **AD-GS** | **No** | **29.16** | **0.920** | **0.033** |
| Waymo | StreetGS | Yes | 33.97 | 0.926 | 0.227 |
| Waymo | EmerNeRF | No | 31.32 | 0.881 | 0.301 |
| Waymo | **AD-GS** | **No** | **33.91** | **0.927** | **0.228** |
| nuScenes | Grid4D | No | 30.29 | 0.920 | 0.172 |
| nuScenes | **AD-GS** | **No** | **31.06** | **0.925** | **0.164** |

### Ablation Study

| Configuration | PSNR↑ (Dynamic) | PSNR↑ (Full Scene) | Note |
|------|------------|-------------|------|
| sin&cos only | 24.28 | 32.61 | Trigonometric functions only |
| + B-spline | 26.65 | 33.65 | Adding B-splines +2.37 |
| + t-mask (full) | **27.41** | **33.91** | Adding temporal mask +0.76 |

| Loss Configuration | PSNR↑ | Note |
|---------|-------|------|
| Base | 26.52 | Image loss only |
| + obj&sky | 26.98 | + Segmentation supervision +0.46 |
| + flow&depth | 28.03 | + Motion and 3D information +1.05 |
| + reg (full) | **29.16** | + Regularization +1.13 |

### Key Findings

- AD-GS substantially outperforms all self-supervised methods and even approaches the annotation-dependent StreetGS on Waymo (33.91 vs. 33.97 PSNR).
- B-spline curves contribute the largest PSNR gain for dynamic objects (+2.37), validating the importance of local fitting.
- Physical rigid-body regularization is critical for preventing degenerate behavior (+1.13 PSNR).
- Under extremely sparse viewpoints (KITTI-25%), a notable gap between self-supervised and annotation-dependent methods remains.

## Highlights & Insights

- **Complementary combination of B-splines and trigonometric functions**: B-splines provide locally precise fitting while trigonometric functions provide global stability; the combination perfectly balances noise robustness and motion accuracy in self-supervised settings. This local–global combination paradigm is transferable to other time-series fitting tasks.
- **The wisdom of simplified segmentation**: Pursuing coarse binary classification rather than fine-grained segmentation proves more robust under noisy pseudo-label environments — a "less is more" design philosophy.
- **LiDAR timestamps as visibility priors**: Existing information (LiDAR acquisition timestamps) is cleverly leveraged as an anchor for object appearance timing, avoiding the instability of learning temporal positions from scratch.

## Limitations & Future Work

- Performance under extremely sparse viewpoints remains limited, as self-supervised motion fitting is insufficiently constrained without adequate observations.
- Non-rigid moving objects such as pedestrians are not specifically modeled.
- Binary scene decomposition may be insufficiently fine-grained for complex interaction scenarios.

## Related Work & Insights

- **vs. PVG**: PVG models motion with trigonometric functions alone, lacking local fitting capacity; AD-GS complements local detail via B-splines.
- **vs. EmerNeRF**: EmerNeRF uses detector feature supervision for decomposition; AD-GS achieves superior results with simpler binary classification segmentation.
- **vs. 4DGF**: 4DGF relies on manual 3D annotations; AD-GS approaches its performance without any such labels.

## Rating

- Novelty: ⭐⭐⭐⭐ The B-spline + trigonometric function motion modeling scheme is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, comprehensive comparisons against both annotated and annotation-free methods, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with complete formulations.
- Value: ⭐⭐⭐⭐⭐ Self-supervised performance approaching annotation-dependent methods carries significant implications for reducing the cost of autonomous driving simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)

</div>

<!-- RELATED:END -->
