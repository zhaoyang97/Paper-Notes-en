---
title: >-
  [Paper Note] SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM
description: >-
  [CVPR 2026][3D Vision][3DGS] This paper proposes SGAD-SLAM, which adopts a pixel-aligned simplified Gaussian representation and allows Gaussians to adjust their depth offset along the ray to improve rendering quality and scalability. A geometry-similarity-based GICP tracking strategy is introduced to accelerate camera pose estimation. The method comprehensively outperforms state-of-the-art approaches on Replica, TUM, ScanNet, and ScanNet++.
tags:
  - CVPR 2026
  - 3D Vision
  - 3DGS
  - RGBD SLAM
  - pixel-aligned Gaussians
  - depth offset
  - generalized ICP
date: 2026-05-08
content_hash: 17f5a65634629628
---

# SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM

**Conference**: CVPR 2026  
**arXiv**: [2603.21055](https://arxiv.org/abs/2603.21055)  
**Code**: [https://machineperceptionlab.github.io/SGAD-SLAM-Project](https://machineperceptionlab.github.io/SGAD-SLAM-Project)  
**Area**: 3D Vision  
**Keywords**: 3DGS, RGBD SLAM, pixel-aligned Gaussians, depth offset, generalized ICP

## TL;DR
This paper proposes SGAD-SLAM, which adopts a pixel-aligned simplified Gaussian representation and allows Gaussians to adjust their depth offset along the ray to improve rendering quality and scalability. A geometry-similarity-based GICP tracking strategy is introduced to accelerate camera pose estimation. The method comprehensively outperforms state-of-the-art approaches on Replica, TUM, ScanNet, and ScanNet++.

## Background & Motivation

1. **State of the Field**: 3DGS has become the dominant radiance field representation in RGBD SLAM as an alternative to NeRF, significantly improving rendering efficiency through differentiable splatting. Current 3DGS-SLAM methods fall into two categories based on Gaussian mobility: (a) globally free 3D Gaussians—flexible but requiring all Gaussians to be maintained in GPU memory, making scalability to large scenes difficult; (b) view-tied Gaussians (e.g., VTGS-SLAM)—scalable but strictly anchored at fixed depth points, which limits rendering quality.
2. **Limitations of Prior Work**: Global Gaussian methods suffer from insufficient GPU memory in large scenes; view-tied Gaussian methods are constrained because Gaussian positions are fully fixed to observed depths, making them unable to adapt to depth noise and geometric inaccuracies, thereby limiting rendering quality. On the tracking side, rendering-based tracking (optimizing rendering error) is computationally inefficient.
3. **Root Cause**: A fundamental trade-off exists between scalability and rendering quality—pixel alignment saves memory but fixed positions hurt quality; free movement improves quality but incurs high memory overhead.
4. **Paper Goals**: (1) Design a Gaussian representation that is both scalable and high-quality; (2) achieve camera pose estimation faster than rendering-based tracking.
5. **Starting Point**: Allowing pixel-aligned Gaussians to make limited adjustments along the ray direction (depth offset), thereby combining the scalability of pixel alignment with the rendering flexibility of position adjustment. Tracking replaces 2D rendering optimization with 3D geometric alignment.
6. **Core Idea**: Pixel-aligned Gaussians + learnable depth offset = scalable and high-quality radiance fields; GICP geometric tracking = fast and accurate localization.

## Method

### Overall Architecture
The system consists of two parallel branches: (1) Mapping branch: initializes pixel-aligned Gaussians for each incoming depth frame, learns Gaussian attributes (color, radius, opacity, depth offset), and builds the map by minimizing rendering error; (2) Tracking branch: maintains a global 3D geometric Gaussian point set representing scene structure, and estimates the camera pose each frame by aligning the local depth Gaussian distribution with the global geometric distribution. The input is an RGBD image sequence; the output is a set of pixel-aligned Gaussians and camera poses for each frame.

### Key Designs

1. **Simplified Spherical Gaussians + Depth Offset**:

    - **Function**: Represent the radiance field with minimal parameters while maintaining rendering quality.
    - **Mechanism**: Each Gaussian retains only color ($\mathbb{R}^3$), a single variance as radius ($\mathbb{R}^1$), opacity ($\mathbb{R}^1$), and depth offset ($\mathbb{R}^1$), totaling 6 parameters. The 4D rotation, 3D position, and 2 additional variances of standard 3DGS are discarded. Each Gaussian is aligned to one pixel; its 3D position is determined by the ray from the camera center through that pixel plus a learnable depth offset $\tilde{D}_i = |D_i + \delta_i|$. No local densification is performed.
    - **Design Motivation**: Parameters are drastically reduced from 59 in standard 3DGS to 6, substantially lowering storage overhead. The introduction of depth offset enables high-quality rendering through adaptive per-pixel position adjustment, even under the constraints of a simplified model, restricted motion, and no densification.

2. **Pixel-Aligned Mapping**:

    - **Function**: Learn Gaussian attributes for each frame to represent scene geometry and appearance.
    - **Mechanism**: For each frame, a set of pixel-aligned Gaussians $G_i = \{g_i^j\}_{j=1}^J$ is initialized and rendered onto the current frame and its neighboring frames via differentiable splatting. The mapping minimizes the rendering error $\min_{G_i, \delta_i} \sum_k (\rho \|V_k - V_k'\|_1 + \tau L_S + \sigma U_k \|D_k - D_k'\|_1)$. Only the Gaussians and depth offsets of the current frame are optimized; neighboring frame Gaussians are fixed to ensure cross-frame consistency. Missing depth regions are filled via interpolation or by rendering Gaussians from neighboring frames.
    - **Design Motivation**: Each frame's Gaussians only need to fit the current frame and its neighbors, eliminating the need to maintain all scene Gaussians on the GPU and substantially improving scalability for large scenes.

3. **Geometry-Similarity-Based GICP Tracking**:

    - **Function**: Efficiently and accurately estimate the camera pose for each frame.
    - **Mechanism**: (1) Uniformly sample 3D points from the current frame's depth map; compute the covariance matrix of each point's neighborhood using KNN to construct a local Gaussian distribution set $T_i$. (2) Maintain a global Gaussian point set $T$ representing the geometric structure of the scanned scene. (3) Apply Generalized ICP (GICP) to align $T_i$ and $T$, estimating the pose by maximizing the overlap between the two Gaussian distributions. Point-to-plane rather than point-to-point distances are used to establish correspondences, with normal vectors obtained via SVD decomposition. Scale normalization eliminates depth-range differences across frames.
    - **Design Motivation**: 3D geometric alignment is considerably faster than 2D rendering optimization (GICP supports efficient parallelization) and does not rely on pretrained priors (e.g., NetVLAD), offering a simpler approach than loop-closure-based methods. Modeling local geometry with Gaussian distributions is more robust than naive point-to-point alignment.

### Loss & Training
- Mapping loss: L1 RGB loss + SSIM loss + masked L1 depth loss
- Tracking initialization: constant-velocity assumption by default; optional rendering-based initialization (coarsely estimating pose using rendering error from previous-frame Gaussians) for texture-less or large-motion scenarios
- Global geometric point set $T$ is updated incrementally: non-overlapping Gaussians are added after each frame's tracking

## Key Experimental Results

### Main Results — Rendering Quality

| Dataset | Metric | SGAD-SLAM | VTGS-SLAM | Gaussian-SLAM | SplaTAM |
|---------|--------|-----------|-----------|---------------|---------|
| Replica | PSNR↑ | **44.87** | 43.34 | 42.08 | 34.11 |
| Replica | SSIM↑ | **0.998** | 0.996 | 0.996 | 0.970 |
| TUM | PSNR↑ | **38.60** | 30.20 | 25.05 | 22.80 |
| TUM | SSIM↑ | **0.997** | 0.972 | 0.929 | 0.893 |
| ScanNet | PSNR↑ | **42.31** | 31.10 | 27.70 | 19.14 |

### Main Results — Tracking Accuracy (ATE RMSE [cm]↓)

| Dataset | SGAD-SLAM | GS-ICP SLAM | VTGS-SLAM | LoopSplat* | CG-SLAM* |
|---------|-----------|-------------|-----------|------------|----------|
| Replica Avg | **0.16** | **0.16** | 0.28 | 0.26 | 0.27 |
| TUM Avg | **2.0** | 2.4 | 2.6 | 2.3 | **2.0** |
| ScanNet Avg | 7.9 | - | 11.3 | **7.7** | 8.1 |
| ScanNet++ Avg | **0.59** | - | 1.6 | 2.05 | - |

### Reconstruction Quality (Replica)

| Metric | SGAD-SLAM | VTGS-SLAM | Point-SLAM | Loopy-SLAM* |
|--------|-----------|-----------|------------|-------------|
| Depth L1 [cm]↓ | **0.30** | 0.51 | 0.44 | 0.35 |
| F1 [%]↑ | **90.9** | 90.4 | 89.8 | 90.8 |

### Key Findings
- **Critical role of depth offset**: Pixel-aligned Gaussians with depth offset surpass both globally free Gaussians and fixed-depth Gaussians in rendering quality, even under constrained conditions (simplified Gaussians + restricted motion + no densification).
- **Substantial rendering gains on TUM and ScanNet**: PSNR improves from 30.20 to 38.60 on TUM (+8.4 dB) and from 31.10 to 42.31 on ScanNet (+11.2 dB), demonstrating that depth offset is especially effective for real-world scenes.
- **Tracking without pretrained priors**: Without relying on pretrained models such as NetVLAD for loop closure, GICP geometric alignment alone achieves accuracy comparable to or better than methods that use pretrained priors.
- **Rendering-based initialization is critical for ScanNet++**: Removing this initialization raises the ScanNet++ ATE from 0.59 to 6.5, indicating that rendering-based initialization is necessary for large-motion or texture-less scenes.

## Highlights & Insights
- The **"pixel-aligned + depth offset"** compromise is elegant: it simultaneously achieves the memory scalability of pixel alignment (no global Gaussian maintenance required) and the rendering flexibility of free Gaussians (limited adjustment along the ray), adding only a single scalar parameter. This "constrained yet adjustable" design philosophy has broad applicability.
- **Geometric tracking as a replacement for rendering-based tracking**: Modeling local geometry with 3D Gaussian distributions and aligning them via GICP is not only faster but also more robust on noisy real-world data, offering a technically distinct alternative to the mainstream rendering-based tracking paradigm.
- **Extreme compression of simplified Gaussians**: Reducing parameters from 59 to 6 (approximately 10×) demonstrates that carefully designed constraints in SLAM scenarios allow a small number of parameters to match or exceed full-parameter performance.

## Limitations & Future Work
- The depth offset is a 1D adjustment along the ray and cannot handle scenarios requiring lateral displacement (e.g., systematic bias in depth sensors).
- The continuously growing global geometric point set $T$ may still pose memory challenges in extremely large scenes, necessitating point set pruning strategies.
- Current experiments focus primarily on indoor environments; performance in outdoor or open-world settings has not been validated.
- GICP tracking may lack discriminative geometric features when sampling depth maps in texture-uniform regions.

## Related Work & Insights
- **vs. VTGS-SLAM**: VTGS-SLAM strictly anchors Gaussians at depth points with no freedom to move; this paper introduces depth offsets to break this constraint. Both use simplified Gaussians, but rendering quality is significantly better here (TUM: +8.4 dB).
- **vs. GS-ICP SLAM**: Both employ ICP-style tracking, but this paper models local geometry with Gaussian distributions rather than using raw points directly. Tracking accuracy is comparable on Replica, while rendering quality is substantially superior.
- **vs. SplaTAM/Gaussian-SLAM**: These methods use globally free Gaussians with high storage overhead and slow convergence. The proposed pixel-aligned strategy achieves better rendering quality while maintaining scalability.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of depth offset and pixel alignment is simple yet effective; the geometric tracking strategy is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, multiple metrics (rendering/tracking/reconstruction), and extensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive figures, and detailed method descriptions.
- Value: ⭐⭐⭐⭐ Advances the state of the art in 3DGS SLAM with substantial gains on real-world data.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Unblur-SLAM: Dense Neural SLAM for Blurry Inputs](unblur-slam_dense_neural_slam_for_blurry_inputs.md)
- [\[CVPR 2026\] VGGT-SLAM++: Visual SLAM with DEM-Based Covisibility and Local Bundle Adjustment](vggt-slam.md)
- [\[CVPR 2026\] DROID-W: DROID-SLAM in the Wild](droid-slam_in_the_wild.md)
- [\[CVPR 2026\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[ICCV 2025\] 4D Gaussian Splatting SLAM](../../ICCV2025/3d_vision/4d_gaussian_splatting_slam.md)

<!-- RELATED:END -->
