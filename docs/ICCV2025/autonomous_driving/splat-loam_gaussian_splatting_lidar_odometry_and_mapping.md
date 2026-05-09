---
title: >-
  [Paper Note] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping
description: >-
  [ICCV 2025][Autonomous Driving][Gaussian Splatting] The first LiDAR odometry and mapping pipeline built entirely on 2D Gaussian primitives, simultaneously achieving high-accuracy pose estimation and lightweight scene reconstruction via a spherical-projection-driven differentiable rasterizer.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Gaussian Splatting
  - LiDAR SLAM
  - Odometry
  - Mapping
  - Spherical Projection
date: 2026-05-08
content_hash: a3685610da989138
---

# Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping

**Conference**: ICCV 2025
**arXiv**: [2503.17491](https://arxiv.org/abs/2503.17491)
**Code**: [GitHub](https://github.com/rvp-group/Splat-LOAM)
**Area**: Autonomous Driving
**Keywords**: Gaussian Splatting, LiDAR SLAM, Odometry, Mapping, Spherical Projection

## TL;DR

The first LiDAR odometry and mapping pipeline built entirely on 2D Gaussian primitives, simultaneously achieving high-accuracy pose estimation and lightweight scene reconstruction via a spherical-projection-driven differentiable rasterizer.

## Background & Motivation

LiDAR sensors provide precise geometric measurements and are widely used for ego-motion estimation and environment reconstruction in autonomous driving. However, classical approaches face a fundamental trilemma: **the trade-off among accuracy, memory, and runtime**.

- **Classical methods** (point cloud stacking, surfels, meshes) can incrementally build global maps but typically result in massive point clouds or require compromises between accuracy and memory.
- **NeRF-based implicit methods** (e.g., SHINE-Mapping, N³-Mapping, PIN-SLAM) introduce neural implicit SDF representations, improving accuracy and storage efficiency, but require complex sampling strategies for SDF estimation and suffer from speed bottlenecks in online execution.
- **3D Gaussian Splatting (3DGS)** has achieved impressive results in visual SLAM (e.g., SplatAM, Gaussian-SLAM), yet these methods rely on camera images and color information and cannot be directly applied to pure LiDAR data.

The paper's starting point is that **LiDAR naturally provides precise 3D structural information that can directly initialize Gaussian primitives**, and the efficiency of Gaussian Splatting—together with its freedom from modeling empty space—is particularly well-suited to LiDAR scenarios. The core idea is to combine 2D Gaussian primitives with a spherical projection model and design a dedicated differentiable rasterizer, enabling an efficient LOAM pipeline driven purely by LiDAR data.

## Method

### Overall Architecture

The system employs a keyframe-driven local map strategy. A new Gaussian local model is initialized whenever visibility conditions are satisfied. Incoming frames estimate poses via frame-to-model registration, while the local model is iteratively optimized. The entire pipeline uses only 2D Gaussian primitives as the scene representation.

### Key Designs

1. **Spherical Projection Model**: Since LiDAR provides 360° panoramic input, the pinhole camera projection model is unsuitable. The paper adopts spherical projection $\phi(\mathbf{p}) = \mathbf{K}\psi(\mathbf{p})$, mapping 3D points to image coordinates defined by azimuth and elevation angles. The camera intrinsic matrix $\mathbf{K}$ is adaptively computed from the actual angular range of each frame's point cloud, avoiding blank regions caused by hard-coded FoV values.

2. **2D Gaussian Primitive Definition and Rasterization**: Each 2D Gaussian is defined by opacity $o$, center $\boldsymbol{\mu}$, two tangent vectors $\mathbf{t}_\alpha, \mathbf{t}_\beta$, and scales $(s_\alpha, s_\beta)$. Rendering uses tile-based α-blending: the image is divided into 16×16 tiles, and within each tile, Gaussians sorted by distance are integrated front-to-back to yield depth $d$, normal $\mathbf{n}$, and opacity $o$.

3. **Explicit Ray–Splat Intersection Computation**: The paper abandons the local affine approximation used in the original 3DGS (which introduces numerical instability and distortion under spherical projection), instead computing explicit ray–splat tri-plane intersections by solving a homogeneous linear system to obtain intersection coordinates $(\alpha, \beta)$.

4. **Spherical Bounding Box Handling**: Spherical images exhibit coordinate singularities at the horizontal boundaries $\{0, W\}$. The paper proposes a method that first shifts vertices to the image center, computes the extent, and then maps back, ensuring correct bounding box computation even when a splat lies behind the sensor.

5. **Frame-to-Model Registration**: Both geometric and photometric registration are combined. Geometric registration uses PCA-based kd-tree point-to-plane ICP, while photometric registration minimizes projection range errors on the spherical depth map. Pose updates are parameterized locally via the Lie algebra $\mathfrak{se}(3)$ and solved using second-order Gauss-Newton optimization.

### Loss & Training

The total mapping loss is:

$$\mathcal{L}_{\text{map}} = \mathcal{L}_d + \lambda_o \mathcal{L}_o + \lambda_n \mathcal{L}_n + \lambda_s \sum_{i=1}^{N} \mathcal{L}_{s_i}$$

- $\mathcal{L}_d$: L1 depth map error computed only on valid pixels, with distance weighting
- $\mathcal{L}_o$: Opacity coverage loss, encouraging splats to cover valid measurement regions
- $\mathcal{L}_n$: Normal self-regularization, aligning splat normals with surface normals estimated from rendered depth map gradients
- $\mathcal{L}_s$: Scale regularization, penalizing splat extent only beyond threshold $\tau_s$, supporting anisotropic splats

The odometry loss is the sum of geometric and photometric terms: $\mathcal{L}_{\text{odom}} = \mathcal{L}_{\text{geo}} + \mathcal{L}_{\text{photo}}$.

Keyframes are sampled according to a geometric distribution, ensuring the most recent keyframe has a ≥40% probability of being selected. Opacity reset is not performed to avoid catastrophic forgetting.

## Key Experimental Results

### Main Results

Mapping quality is evaluated on the Newer College and Oxford Spires datasets using ground-truth poses:

| Method | Acc↓ | Com↓ | C-l1↓ | F-score↑ |
|--------|------|------|-------|----------|
| OpenVDB | 11.45 | 4.38 | 7.92 | 88.85 |
| VoxBlox | 20.36 | 12.64 | 16.50 | 64.63 |
| N³-Mapping | 6.32 | 9.75 | 8.04 | 94.54 |
| PIN-SLAM | 15.28 | 10.50 | 12.89 | 88.05 |
| **Splat-LOAM** | **6.64** | **4.09** | **5.37** | **96.74** |

For odometry evaluation across four datasets (NC, VBR, Oxford Spires, Mai City), Splat-LOAM with combined geometric and photometric registration achieves competitive performance compared to state-of-the-art methods such as MAD-ICP.

### Ablation Study

| Mesh Extraction | Acc↓ | Com↓ | C-l1↓ | F-score↑ |
|----------------|------|------|-------|----------|
| Marching Cubes | 16.76 | 5.53 | 11.14 | 76.76 |
| Poisson (centers) | 10.15 | 6.70 | 8.43 | 92.33 |
| **Ours (rendered)** | **6.64** | **4.09** | **5.37** | **96.74** |

Registration ablation shows that jointly using geometric and photometric factors outperforms any single method (Point-to-Point / Point-to-Plane / photometric only), validating the hybrid registration design.

### Key Findings

- Mapping FPS stabilizes when the number of active primitives is in the range of 200K–300K (limited by rasterizer saturation).
- The system is sensitive to LiDAR motion distortion; joint velocity and pose estimation is an important direction for improvement.
- Compared to N³-Mapping, Splat-LOAM achieves superior detail fidelity, as the latter tends to over-smooth surfaces.

## Highlights & Insights

- The first pure-LiDAR Gaussian Splatting LOAM system, filling a gap in this research direction.
- The bounding box computation scheme for handling coordinate singularities under spherical projection is particularly elegant.
- The threshold design of the scale regularization loss $\mathcal{L}_s$ permits anisotropic splats and is more flexible than directly minimizing mean deviation.
- Extremely low GPU memory requirements make the system suitable for real-time robotic perception.

## Limitations & Future Work

- Sensitivity to LiDAR motion distortion necessitates simultaneous velocity estimation.
- No loop closure module is included, leading to accumulated drift over long trajectories.
- LiDAR intensity/reflectance information is not exploited for appearance modeling.
- The compact bounding box computation for the spherical projection rasterizer is an approximation, leaving room for improvement.

## Related Work & Insights

- PIN-SLAM's neural point cloud approach offers good global consistency but is unsatisfactory in terms of memory and speed.
- The paradigm of visual Gaussian SLAM (SplatAM, Splat-SLAM) can be adapted in reverse for LiDAR.
- GS-LiDAR's periodic oscillation Gaussian design targets dynamic objects and is complementary to this paper's static scene mapping approach.

## Rating

- Novelty: ⭐⭐⭐⭐ First pure-LiDAR GS LOAM
- Technical Depth: ⭐⭐⭐⭐ Complete spherical rasterizer design
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-metric evaluation
- Value: ⭐⭐⭐⭐ Low GPU requirements, suitable for robotics
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[ICCV 2025\] A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](a_constrained_optimization_approach_for_gaussian_splatting_from_coarsely-posed_i.md)

</div>

<!-- RELATED:END -->
