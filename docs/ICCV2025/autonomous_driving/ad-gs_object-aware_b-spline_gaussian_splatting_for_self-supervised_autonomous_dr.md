---
title: >-
  [Paper Note] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving
description: >-
  [ICCV 2025][Autonomous Driving][Gaussian Splatting] This paper proposes AD-GS, a self-supervised autonomous driving scene rendering framework based on 3D Gaussian Splatting. The core innovation is combining learnable B-spline curves with trigonometric functions for local-global motion modeling, coupled with a simplified binary pseudo-segmentation for robust scene decomposition. Without relying on manual 3D annotations, AD-GS substantially outperforms existing self-supervised methods.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Gaussian Splatting
  - Self-Supervised Learning
  - B-Spline Motion Modeling
  - Dynamic Scene Rendering
date: 2026-05-08
content_hash: 852517afcea32df6
---

# AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving

**Conference**: ICCV 2025
**arXiv**: [2507.12137](https://arxiv.org/abs/2507.12137)
**Code**: [Project Page](https://jiaweixu8.github.io/AD-GS-web/)
**Area**: Autonomous Driving Scene Reconstruction
**Keywords**: Gaussian Splatting, Autonomous Driving, Self-Supervised Learning, B-Spline Motion Modeling, Dynamic Scene Rendering

## TL;DR
This paper proposes AD-GS, a self-supervised autonomous driving scene rendering framework based on 3D Gaussian Splatting. The core innovation is combining learnable B-spline curves with trigonometric functions for local-global motion modeling, coupled with a simplified binary pseudo-segmentation for robust scene decomposition. Without relying on manual 3D annotations, AD-GS substantially outperforms existing self-supervised methods.

## Background & Motivation
High-quality rendering and reconstruction of autonomous driving scenes is critical for simulation. Existing high-quality methods (e.g., 4DGF, StreetGS, ML-NSG) rely on expensive manual 3D annotations (bounding boxes and poses of objects), which, while effective, incur high annotation costs that limit large-scale deployment.

Self-supervised methods attempt to reconstruct dynamic scenes solely from images and LiDAR data, but face two major technical challenges. First, the local-global dilemma in motion modeling: MLP-based motion modeling is computationally expensive and struggles to capture local motion details, whereas trigonometric function-based modeling (e.g., PVG) is fast and globally smooth (since all parameters participate in optimization at every frame), but lacks local motion fidelity. Second, noise in scene decomposition: using instance segmentation pseudo-labels for scene decomposition introduces substantial noise, leading to reconstruction artifacts.

The root cause lies in: how to simultaneously achieve precise motion modeling and robust scene decomposition under self-supervised (noisy pseudo-label) conditions. The paper's starting point is the introduction of B-spline curves — their local control property ensures that the position at any given moment is influenced only by nearby control points, enabling precise fitting of local motion details, while combination with trigonometric functions preserves global smoothness. For scene decomposition, the problem is simplified to only two categories (objects vs. background), greatly reducing the impact of pseudo-label noise.

Core Idea: Local fitting via B-spline + Global fitting via trigonometric functions = Precise and robust motion modeling.

## Method

### Overall Architecture
AD-GS decomposes the scene into object Gaussians $\boldsymbol{\Omega}_{obj}$ and background Gaussians $\boldsymbol{\Omega}_{bkg}$. Background Gaussians remain static, with only color varying over time. Object Gaussians undergo position and rotation deformation via B-spline + trigonometric functions, augmented with bidirectional temporal visibility masks to handle sudden object appearance/disappearance. Distant regions (e.g., sky) are represented by a learnable spherical environment map. The entire system is trained self-supervisedly using multiple pseudo-labels and regularization terms derived from image reconstruction, optical flow, depth, and segmentation.

### Key Designs
1. **Learnable B-Spline Motion Curves**:

    - Function: Deforms the position of each object Gaussian over time using a combination of B-spline curves and trigonometric functions.
    - Mechanism: Given $n+1$ learnable control points $\mathbf{p}_i$, a $k$-th order B-spline curve is constructed as $\mathbf{p}(t) = \sum_{i=0}^{n} \mathbf{p}_i B_{i,k}(t)$. The basis functions $B_{i,k}$ are nonzero only over the local interval $[t_i, t_{i+k}]$, so the position at each moment is influenced by only $k$ nearby control points. To avoid the inefficiency of the de Boor-Cox recursion, the matrix form $\mathbf{p}(t) = [1, u, u^2, ..., u^{k-1}] M_k [\mathbf{p}_{i-k+1}, ..., \mathbf{p}_i]^T$ is used for efficient computation. The final position deformation is:
    $$\boldsymbol{\mu}' = \boldsymbol{\mu} + \mathbf{p}(t) + \sum_{l=1}^{L} \mathbf{a}_l \sin(t \cdot l\pi) + \mathbf{b}_l \cos(t \cdot l\pi)$$
    - Design Motivation: Using B-splines alone is prone to overfitting local noise under noisy self-supervision, while trigonometric functions smooth out noise through global parameter optimization. The two are complementary: B-splines capture local details (e.g., lane changes), while trigonometric functions ensure global trends (e.g., uniform forward motion).

2. **B-Spline Quaternion Rotation Curves**:

    - Function: Models the rotational deformation of object Gaussians using B-spline quaternion curves.
    - Mechanism: Standard B-splines cannot handle the non-uniform interpolation of unit quaternions. A dedicated quaternion B-spline curve is employed: $\mathbf{q}(t) = \mathbf{q}_{i-k+1} \prod_{j=i-k+2}^{i} \exp(\mathbf{w}_j \tilde{B}_{j,k}(t))$, where $\mathbf{w}_i = \log(\mathbf{q}_{i-1}^{-1} \mathbf{q}_i)$.
    - Design Motivation: Rotation parameters in quaternion space require special handling to maintain the unit constraint and uniform interpolation. The quaternion B-spline curve $R = \mathbf{q}(t)$ is sufficient to cover rigid-body rotation in autonomous driving scenarios.

3. **Scene Decomposition via Simplified 2D Pseudo-Segmentation**:

    - Function: Simplifies the scene into two categories — objects (potentially moving, e.g., vehicles) and background (everything else).
    - Mechanism: SAM is used to generate binary segmentation masks $\mathcal{M}_{obj}$. Gaussians are assigned to $\boldsymbol{\Omega}_{obj}$ or $\boldsymbol{\Omega}_{bkg}$ based on the projection of LiDAR points onto the binary mask. The object mask $\hat{\mathcal{M}}_{obj}$ is rendered via $\alpha$-blending and supervised with a BCE loss.
    - Design Motivation: Prior work employs instance segmentation (fine-grained, multi-class) for scene decomposition, but instance-level pseudo-labels suffer from significant noise. Simplifying to binary segmentation substantially improves robustness — for autonomous driving, distinguishing "dynamic objects" from "static background" is sufficient.

4. **Bidirectional Temporal Visibility Mask**:

    - Function: Handles the sudden appearance and disappearance of objects in a temporal sequence.
    - Mechanism: A temporal Gaussian window is applied to the opacity of each object Gaussian: $\sigma'(t) = \sigma \cdot e^{-\frac{(t - \mu_t)^2}{2s^2}}$, where $\mu_t$ is fixed to the LiDAR acquisition timestamp (non-learnable), and $s_0, s_1$ are bidirectional learnable scaling factors. An extension regularization loss $\mathcal{L}_s = \| \frac{2\Delta_f}{s_0 + s_1} \|_1$ is used to prevent the window from becoming too narrow.
    - Design Motivation: Prior work also treats $\mu_t$ as a learnable parameter, but this paper argues that the LiDAR acquisition timestamp already provides an effective cue for the object's visible moment. The bidirectional design (different $s$ for forward and backward directions) allows objects to enter and exit the field of view from different sides.

### Loss & Training
The total loss consists of 8 terms:
$$\mathcal{L} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c \mathcal{L}_{D-SSIM} + \lambda_d \mathcal{L}_d + \lambda_f \mathcal{L}_f + \lambda_{obj} \mathcal{L}_{obj} + \lambda_{sky} \mathcal{L}_{sky} + \lambda_r \mathcal{L}_r + \lambda_s \mathcal{L}_s$$

- Image reconstruction losses ($\mathcal{L}_1$, $\mathcal{L}_{D-SSIM}$)
- Inverse depth loss ($\mathcal{L}_d$): Uses DPTv2-generated monocular depth pseudo-labels; directly renders the expected inverse depth value to avoid numerical instability.
- Optical flow loss ($\mathcal{L}_f$): Uses CoTracker3-generated pseudo-labels; supervision is applied only to the object regions.
- Object/sky segmentation losses ($\mathcal{L}_{obj}$, $\mathcal{L}_{sky}$): BCE losses.
- Physical rigidity regularization ($\mathcal{L}_r$): Constrains the variance of deformation parameters among the 8 nearest-neighbor Gaussians (KNN updated every 10 steps).
- Temporal mask extension regularization ($\mathcal{L}_s$).
- B-spline order $k=6$; number of control points is 1/3 of the total frame count; maximum trigonometric frequency $K=L=6$.

## Key Experimental Results

### Main Results

| Dataset/Setting | Metric | AD-GS (Self-sup.) | PVG (Self-sup.) | 4DGF (Supervised) |
|----------------|--------|-------------------|-----------------|-------------------|
| KITTI-75% | PSNR↑ | **29.16** | 27.13 | 31.34 |
| KITTI-75% | SSIM↑ | **0.920** | 0.895 | 0.945 |
| KITTI-75% | LPIPS↓ | **0.033** | 0.049 | 0.026 |
| Waymo | PSNR↑ | **33.91** | 29.54 | 34.64 |
| Waymo | PSNR* (Dynamic)↑ | **27.41** | 21.56 | 29.77 |
| nuScenes | PSNR↑ | **31.06** | 29.49 | - |
| nuScenes | LPIPS↓ | **0.164** | 0.211 | - |

AD-GS significantly outperforms all self-supervised methods across three datasets and closely approaches methods that rely on manual 3D annotations.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Note |
|--------------|-------|-------|--------|------|
| Full AD-GS | **29.16** | **0.920** | **0.033** | Complete model |
| w/o reg | 28.03 | 0.910 | 0.042 | Without regularization; structural artifacts |
| w/o flow & depth | 26.98 | 0.902 | 0.048 | Without motion and 3D information |
| w/o obj & sky | 26.52 | 0.896 | 0.053 | Without scene decomposition |
| sin/cos only | PSNR*=24.28 | - | - | Global fitting only |
| B-spline only | PSNR*=25.70 | - | - | Local fitting only |
| B-spline + sin/cos | PSNR*=26.65 | - | - | Without temporal mask |
| Full (+t-mask) | PSNR*=**27.41** | - | - | All components |

### Key Findings
- The combination of B-spline and trigonometric functions outperforms either component alone, validating the core hypothesis of local-global complementarity.
- The bidirectional temporal visibility mask yields significant gains in regions where objects enter or leave the scene (PSNR* improves from 26.65 to 27.41).
- Simplified binary segmentation is more robust than fine-grained instance segmentation due to reduced pseudo-label noise.
- When training views are extremely sparse (KITTI-25%), self-supervised motion estimation becomes more difficult, widening the performance gap.
- Physical rigidity regularization ($\mathcal{L}_r$) prevents artifacts caused by inconsistent deformation among neighboring Gaussians.

## Highlights & Insights
- **Introduction of B-splines**: This is among the first works to employ B-splines for motion modeling in dynamic Gaussian Splatting; their local control property is well-suited for handling noisy self-supervised signals.
- **Matrix-form acceleration**: Replacing the de Boor-Cox recursion with a matrix formulation enables efficient B-spline computation.
- **Inverse depth rendering trick**: Directly rendering the expected value of inverse depth avoids numerical instability when no Gaussians lie along a pixel ray.
- **"Less is more" segmentation strategy**: Reducing multi-class instance segmentation to binary segmentation achieves better results precisely because of the improved robustness.

## Limitations & Future Work
- Under extremely sparse training views (25%), the method still falls noticeably short of annotation-based methods, indicating that the ceiling of self-supervised motion estimation remains to be pushed further.
- Only rigid-body motion is modeled; non-rigid dynamic objects such as pedestrians are not handled.
- The number of B-spline control points is fixed at 1/3 of the frame count, which may not be optimal for scenes with varying speeds and complexity.
- The method depends on multiple external pseudo-labels (SAM segmentation, DPTv2 depth, CoTracker3 optical flow); poor quality in any one source may cascade into degraded performance.

## Related Work & Insights
- **vs. PVG**: PVG models motion using trigonometric functions and periodic vibrations, which is globally smooth but lacks local detail. AD-GS builds upon PVG by incorporating B-splines to achieve local-global complementarity.
- **vs. EmerNeRF**: EmerNeRF uses detection model features to supervise static-dynamic decomposition, but is slow due to its NeRF backbone. AD-GS is more efficient owing to its Gaussian Splatting foundation.
- **vs. 4DGF/StreetGS**: These methods rely on manual 3D annotations; AD-GS approaches their performance in a fully self-supervised manner, demonstrating the substantial potential of annotation-free approaches.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined B-spline + trigonometric motion modeling design is novel; the simplified segmentation strategy is counter-intuitive yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, comprehensive comparisons against both supervised and self-supervised methods, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear, mathematical derivations are complete, and both qualitative and quantitative results are richly presented.
- Value: ⭐⭐⭐⭐⭐ Significantly narrows the gap between self-supervised and supervised methods, with direct engineering value for autonomous driving simulation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving](coda-4dgs_dynamic_gaussian_splatting_with_context_and_deformation_awareness_for_.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)

<!-- RELATED:END -->
