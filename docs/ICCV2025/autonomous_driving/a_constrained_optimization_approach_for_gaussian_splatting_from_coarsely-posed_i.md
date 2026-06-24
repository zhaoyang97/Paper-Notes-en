---
title: >-
  [Paper Note] A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds
description: >-
  [ICCV 2025][Autonomous Driving][3D Gaussian Splatting] This paper proposes an SfM-free constrained optimization framework that jointly optimizes camera parameters and 3DGS scene reconstruction from coarse poses and noisy point clouds produced by multi-camera SLAM systems, via camera pose decomposition, sensitivity-based pre-conditioning, log-barrier constraints, and geometric constraints.
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D Gaussian Splatting"
  - "camera pose optimization"
  - "constrained optimization"
  - "multi-camera SLAM"
  - "LiDAR point cloud"
date: 2026-05-08
content_hash: 3b1dc651eb11f56c
---

# A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds

**Conference**: ICCV 2025
**arXiv**: [2504.09129](https://arxiv.org/abs/2504.09129)  
**Code**: Unavailable (dataset release planned)  
**Area**: Autonomous Driving / 3D Reconstruction / 3D Gaussian Splatting
**Keywords**: 3D Gaussian Splatting, camera pose optimization, constrained optimization, multi-camera SLAM, LiDAR point cloud

## TL;DR

This paper proposes an SfM-free constrained optimization framework that jointly optimizes camera parameters and 3DGS scene reconstruction from coarse poses and noisy point clouds produced by multi-camera SLAM systems, via camera pose decomposition, sensitivity-based pre-conditioning, log-barrier constraints, and geometric constraints.

## Background & Motivation

3D Gaussian Splatting (3DGS) is an efficient 3D reconstruction technique, but it relies heavily on accurate camera poses and high-quality sparse point clouds for initialization, typically obtained from time-consuming Structure-from-Motion (SfM) pipelines such as COLMAP. This dependency limits the applicability of 3DGS in real-world and large-scale scenarios.

In practical robotics/AR/VR settings, multi-camera SLAM systems can rapidly acquire camera poses and point clouds, but suffer from the following issues:

**Pose noise**: Device poses output by SLAM are inaccurate due to sensor noise and LiDAR odometry drift.

**Temporal asynchrony**: RGB image acquisition and device pose estimation are temporally misaligned (up to 50 ms), causing discrepancies between estimated and true poses.

**Calibration error**: Imperfect calibration of camera intrinsics and LiDAR-camera extrinsics introduces additional errors.

**COLMAP latency**: COLMAP processing requires 4–12 hours and may fail in scenes with repetitive textures.

Directly using these noisy inputs results in blurry reconstructions and degraded geometry. The goal of this paper is to achieve high-quality 3DGS reconstruction from imprecise multi-camera SLAM outputs without relying on SfM.

## Method

### Overall Architecture

The core of the approach is to decompose the camera pose optimization problem and impose multiple constraints. Given $N$ RGB images, coarse camera poses, and noisy point clouds, the method jointly optimizes camera intrinsics and extrinsics alongside the 3DGS scene representation. The overall pipeline consists of: pose decomposition → sensitivity-based pre-conditioning → log-barrier constraints → geometric constraints (epipolar + reprojection) → test-time adaptation.

### Key Designs

#### 1. Camera Pose Decomposition

Each camera pose is decomposed into a composition of two transforms:

$$\mathcal{P}^{(j,t)} = \hat{\mathcal{P}}^t \times \mathcal{E}^j$$

where $\hat{\mathcal{P}}^t$ is the device-to-world pose at time $t$, and $\mathcal{E}^j$ is the extrinsic transform from camera $j$ to the device.

- For a 4-camera system with 10k images, the naive degrees of freedom amount to 60k.
- After decomposition, only 2,500 independent device poses and 4 shared extrinsics need to be optimized, yielding 15,024 degrees of freedom.
- Small offset vectors $\vec{\phi}^t$ and $\vec{\rho}^j$ (each 6-dimensional: 3 rotation + 3 translation) are learned to correct the poses.

A key design choice is the use of **right-multiplication** for the error matrix ($f(\hat{\mathcal{P}}^t, \vec{\phi}^t) = \hat{\mathcal{P}}^t \times \Phi^t$) rather than left-multiplication. Left-multiplication causes camera positions to rotate around the world origin (typically far from the initialization), leading to unstable optimization, whereas right-multiplication applies local rotations around the initial camera position, yielding greater stability.

#### 2. Intrinsic Refinement

Existing methods generally assume known intrinsics; however, in multi-camera systems, intrinsic errors cannot be compensated by adjusting extrinsics as in single-camera setups, since all cameras share the same device pose and modifying one camera's extrinsics affects all others. The 3DGS rasterizer is modified to derive analytic gradients, enabling end-to-end optimization of focal lengths and principal points:
- $\partial u/\partial f_x = \vec{u}^x_{\text{cam}} / \vec{u}^z_{\text{cam}}$, $\partial u/\partial c_x = 1$

#### 3. Sensitivity-based Pre-conditioning

Inspired by the Levenberg–Marquardt algorithm, the Jacobian matrix $\mathcal{J}$ of the projection function with respect to each parameter is computed. The diagonal elements of $(\mathcal{J}^\top \mathcal{J})^{-1/2}$ (an approximation of the inverse square root of the Hessian) are used to adaptively scale the learning rate for each parameter group. Even a 1% change in certain parameters can produce drastically different rendering outcomes, necessitating different step sizes across parameters.

#### 4. Log-barrier Constraint

To prevent sensitive parameters from leaving the feasible domain, a log-barrier function is introduced:

$$\mathcal{L}_{\text{barrier}} = \frac{1}{\mathcal{T}} \sum_{i=1}^m \log(-h_i(x))$$

- The temperature $\mathcal{T}$ is gradually increased: strong constraints are enforced early (the gradient $-1/(\tau h_i(x))$ becomes extremely large near the boundary), while constraints are relaxed later to allow fuller exploration.
- Intrinsic constraints: focal length and principal point deviations are bounded within ±2%.
- Extrinsic constraints: device pose rotation ±0.625°, translation ±0.125 m; camera extrinsic rotation ±2.5°, translation ±0.5 m.

#### 5. Geometric Constraints

Semi-dense matching point pairs between adjacent frames are obtained using LoFTR (hundreds of pairs per image pair), and two complementary constraints are proposed:

**Soft epipolar constraint**: The Sampson distance is computed via the fundamental matrix $\mathbb{F}$ to constrain relative poses to satisfy epipolar geometry. This constraint does not account for depth but provides a strong prior.

**Reprojection error regularization**: Traditional bundle adjustment from SfM is extended as a geometric regularizer, utilizing matched point pairs and depth through bidirectional projection. Depth is computed accurately via ray intersection rather than through the unstable alpha-blending approach.

### Loss & Training

The total loss function is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pixel}} + 0.2 \cdot \mathcal{L}_{\text{ssim}} + 0.1 \cdot \mathcal{L}_{\text{barrier}} + 10^{-3} \cdot \mathcal{L}_{\text{epipolar}} + 5 \times 10^{-4} \cdot \mathcal{L}_{\text{reproj}}$$

- Intrinsic learning rate: $8\times10^{-4}$; extrinsic base learning rate: $5\times10^{-3}$ (scaled by Jacobian).
- Cosine learning rate decay with 3 restarts (at iterations 1, max_iter/6, and max_iter/2).
- Training runs for 48k iterations; Gaussian pruning is disabled; densification is enabled after 67% of training.
- **Test-Time Adaptation (TTA)**: The 3DGS model is frozen, and poses along with exposure compensation (low-frequency luminance shifts in YCbCr space) are optimized for 500 steps at a learning rate of $5\times10^{-4}$.

## Key Experimental Results

### Main Results

**In-house dataset**: Four scenes (Cafeteria / Office / Laboratory / Town), captured using a custom device with 4 fisheye cameras, IMU, and LiDAR.

| Method | Preprocessing Time | Cafeteria PSNR/SSIM | Office PSNR/SSIM | Lab PSNR/SSIM | Town PSNR/SSIM |
|---|---|---|---|---|---|
| Direct reconst. | 3 min | 19.23/0.789 | 17.49/0.758 | 18.35/0.798 | 16.12/0.615 |
| Pose optimize | 5 min | 26.89/0.872 | 23.96/0.837 | 26.11/0.867 | 20.18/0.685 |
| 3DGS-COLMAP | 4–12 hrs | 17.03/0.768 | 25.82/0.883 | 28.30/0.908 | 24.07/0.830 |
| 3DGS-COLMAP△ | 2–3 hrs | 26.51/0.838 | 23.91/0.839 | 23.76/0.816 | 23.51/0.809 |
| CF-3DGS | 1 min | 15.44/0.541 | 16.53/0.756 | 16.44/0.756 | 15.45/0.541 |
| MonoGS | 1 min | 8.27/0.468 | 9.56/0.496 | 13.08/0.601 | 12.74/0.309 |
| InstantSplat | 50 min | 19.86/0.774 | 23.30/0.872 | 20.89/0.862 | 21.48/0.738 |
| **Ours** | **5 min** | **29.05/0.917** | **26.07/0.885** | **28.64/0.910** | **24.52/0.826** |

**Public dataset comparison (multi-modal methods)**:

| Method | GarageWorld G0 PSNR | GarageWorld G6 PSNR | Waymo S002 PSNR | Waymo S031 PSNR |
|---|---|---|---|---|
| 3DGS | 25.43 | 21.23 | 25.84 | 24.42 |
| LetsGo | 25.29 | 21.72 | 26.11 | 24.79 |
| Street-GS | 24.20 | 20.52 | 27.96 | 25.04 |
| **Ours** | **26.06** | **23.76** | **29.75** | **28.48** |

### Ablation Study

**Camera decomposition + pre-conditioning** (CVG% denotes the training fraction at which SSIM reaches 95% of its peak; lower is faster convergence):

| C.D. | P.C. | Cafeteria PSNR/SSIM | CVG% | Lab PSNR/SSIM | CVG% |
|---|---|---|---|---|---|
| ✗ | ✗ | 26.91/0.866 | 34.38 | 27.00/0.881 | 31.25 |
| ✗ | ✓ | 26.45/0.858 | 22.92 | 26.07/0.865 | 18.76 |
| ✓ | ✗ | 28.87/0.915 | 43.10 | 28.52/0.909 | 39.58 |
| ✓ | ✓ | **29.05/0.917** | **15.65** | **28.64/0.910** | **16.67** |

**Geometric constraint ablation (under varying noise levels)**:

| Noise | E.P. | R.P. | PSNR/SSIM | Ep-e↓ | RP-e↓ |
|---|---|---|---|---|---|
| - | ✗ | ✗ | 27.05/0.895 | 1.14 | 2.52 |
| - | ✓ | ✓ | 27.31/0.915 | 1.08 | 1.88 |
| 0.2° | ✗ | ✗ | 26.04/0.890 | 1.23 | 2.56 |
| 0.2° | ✓ | ✓ | 26.84/0.905 | 1.11 | 2.00 |
| 0.5° | ✗ | ✗ | 24.80/0.858 | 1.72 | 3.92 |
| 0.5° | ✓ | ✓ | 25.20/0.867 | 1.21 | 2.32 |

**Other key ablations**:
- Intrinsic optimization: PSNR improves from 27.40 to 29.05 on Cafeteria, with notably sharper text and fine details.
- Log-barrier: constraining within ±2% alone improves SSIM by 6.8%.
- Camera count: improvements of +2.30 / +2.24 / +3.07 dB PSNR for 1 / 2 / 4 cameras respectively.
- TTA ablation: joint use of pose optimization and exposure compensation yields the best result (Cafeteria: 28.58 → pose only: 23.04, exposure only: 22.65, neither: 19.80).

### Key Findings

1. **COLMAP is not universally reliable**: It fails on the Cafeteria scene due to repetitive textures (PSNR 17.03), whereas the proposed method achieves 29.05.
2. **Preprocessing time advantage is substantial**: The proposed method requires only 5 minutes versus 4–12 hours for COLMAP.
3. **Camera decomposition contributes most**: It yields approximately 2 dB PSNR gain, and when combined with pre-conditioning, CVG% drops from 43% to 16%, yielding more stable convergence.
4. **Geometric constraints show greater improvement under higher noise levels** (epipolar error decreases from 1.72 to 1.21 pixels under 0.5° noise).
5. Incremental SLAM-based methods (CF-3DGS, MonoGS) degrade severely in low-overlap scenes (SSIM 0.3–0.75).

## Highlights & Insights

1. **Precise problem formulation**: This is the first work to systematically address the use of multi-camera SLAM outputs for 3DGS, with a coherent chain from analysis to constraint design.
2. **Elegant decomposition**: Degrees of freedom are reduced from 60k to 15k; shared extrinsics provide global constraints.
3. **Well-grounded constraint design**: Sensitivity pre-conditioning is derived from the LM algorithm, log-barrier from convex optimization theory, and geometric constraints from SfM.
4. **Right- vs. left-multiplication subtlety**: This seemingly minor technical choice critically determines optimization stability.
5. **Exposure compensation module**: Only the low-frequency component of the luminance channel in YCbCr space is modified, implemented via tinycudann — a concise and efficient design.
6. **Depth computation**: Ray intersection is used for accurate depth estimation rather than the unstable alpha-blending approach.

## Limitations & Future Work

1. Only static scenes are handled; YOLOv8 is used to detect and exclude pedestrian regions.
2. The log-barrier bounds (±2%, etc.) are empirically determined and may require adjustment for different scenes.
3. Keypoint matching quality depends on LoFTR performance.
4. The dataset covers only four scenes; generalization at larger scale remains to be validated.
5. Fisheye undistortion is performed as a preprocessing step; joint optimization of distortion parameters warrants exploration.
6. Semantic information is not incorporated to assist optimization.

## Related Work & Insights

- **InstantSplat**: Leverages a 3D foundation model to provide relative poses, but GPU memory limits processing to at most 30 images.
- **CF-3DGS / MonoGS**: Incremental SLAM + 3DGS approaches that degrade severely under low co-visibility.
- **BARF**: Coarse-to-fine positional encoding for joint pose and NeRF optimization, but limited to NeRF.
- **Street-GS**: A multi-modal autonomous driving method that independently optimizes per-camera poses without intrinsic refinement.
- **LetsGo**: Also exploits LiDAR + cameras but assumes accurate poses.

The constrained optimization framework is generalizable to other 3D tasks requiring recovery from noisy initialization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Constrained optimization theory is systematically introduced into 3DGS pose optimization; the pose decomposition idea is elegant.
- **Practicality**: ⭐⭐⭐⭐⭐ — Replacing 12-hour COLMAP with a 5-minute pipeline carries extremely high value for industrial deployment.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — In-house dataset plus two public benchmarks; comprehensive ablations; well-designed noise robustness experiments on GarageWorld.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-motivated, complete mathematical derivations, thorough supplementary material.
- **Overall**: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [\[ICCV 2025\] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)

</div>

<!-- RELATED:END -->
