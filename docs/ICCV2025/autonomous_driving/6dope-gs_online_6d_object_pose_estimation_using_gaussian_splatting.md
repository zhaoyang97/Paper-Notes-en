---
title: >-
  [Paper Note] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting
description: >-
  [ICCV 2025][Autonomous Driving][6D pose estimation] This paper proposes 6DOPE-GS, a model-free online tracking method that jointly optimizes 6D object pose and 3D reconstruction using 2D Gaussian Splatting (2DGS). Through dynamic keyframe selection and opacity-percentile-based density control, it achieves a 5× speedup while maintaining state-of-the-art accuracy.
tags:
  - ICCV 2025
  - Autonomous Driving
  - 6D pose estimation
  - Gaussian splatting
  - real-time tracking
  - 3D reconstruction
  - RGB-D
date: 2026-05-08
content_hash: 85b79451a5e13a64
---

# 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2412.01543](https://arxiv.org/abs/2412.01543)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: 6D pose estimation, Gaussian splatting, real-time tracking, 3D reconstruction, RGB-D

## TL;DR

This paper proposes 6DOPE-GS, a model-free online tracking method that jointly optimizes 6D object pose and 3D reconstruction using 2D Gaussian Splatting (2DGS). Through dynamic keyframe selection and opacity-percentile-based density control, it achieves a 5× speedup while maintaining state-of-the-art accuracy.

## Background & Motivation

6D object pose estimation is a fundamental task in AR, autonomous driving, and robotic manipulation. Existing methods fall into two main categories:

**Model-based methods**: Rely on CAD models or reference images, with poor generalization to unseen objects.

**Model-free methods**: Such as BundleSDF, which jointly optimizes pose and reconstruction via a Neural Object Field, but incurs prohibitive computational cost (~2.1 s per frame), making real-time deployment infeasible.

Although BundleSDF reports near real-time pose optimization (~10 Hz), its neural field training is far from real-time (~6.7 s per round), yielding an overall tracking frequency of only ~0.4 Hz. This severely limits applicability in dynamic scenarios such as hand-held object manipulation.

**Core Motivation**: Leverage the efficient differentiable rendering capability of Gaussian Splatting to replace slow neural implicit field training, enabling truly online joint pose estimation and 3D reconstruction.

## Method

### Overall Architecture

The 6DOPE-GS pipeline consists of the following stages:

1. **Object Segmentation**: SAM2 is applied to continuously segment the target object in the video stream.
2. **Feature Matching**: LoFTR establishes inter-frame point correspondences.
3. **Coarse Pose Initialization**: RANSAC computes an initial pose and constructs a keyframe pool.
4. **Gaussian Object Field Joint Optimization**: 2DGS jointly optimizes keyframe poses and object reconstruction.
5. **Online Pose Graph Optimization**: Poses of incoming frames are continuously updated using the optimized keyframe poses.

### Key Designs

**1. Gaussian Object Field**

2D Gaussian Splatting (2DGS) is adopted over 3DGS for object modeling:
- 2DGS compresses each Gaussian into a 2D planar disk (surfel) by setting the z-axis scale to zero.
- It provides more accurate surface normals and depth estimates.
- Gradients are back-propagated to keyframe pose parameters through differentiable rendering, enabling joint optimization.

**2. Dynamic Keyframe Selection**

- The vertices and face centers of an icosahedron serve as anchor points, approximating a uniform spherical distribution of viewing directions.
- Initial keyframes are clustered by pose to each anchor, and the frame with the largest object mask in each cluster is selected.
- During joint optimization, outlier keyframes are filtered using the Median Absolute Deviation (MAD) of reconstruction error.
- Viewpoints exceeding 3× MAD are identified as outliers and discarded.

**3. Adaptive Density Control via Opacity Percentile**

- At fixed optimization intervals, Gaussians whose opacity falls in the bottom 5th percentile are pruned.
- Pruning continues until the 95th-percentile opacity exceeds a threshold.
- Compared to the absolute-threshold pruning in vanilla 3DGS, this strategy is more stable and avoids drastic fluctuations in the number of Gaussians.

### Loss & Training

- Rendering loss: RGB reconstruction loss + depth reconstruction loss + normal consistency loss.
- Keyframe poses receive gradients from the rendering loss via automatic differentiation (PyTorch) and are updated accordingly.
- Once the Gaussian Object Field converges, Gaussian parameters are frozen and all keyframe poses are jointly refined.
- Online pose graph optimization uses dense pixel-wise re-projection error.

## Key Experimental Results

### Main Results

YCBInEOAT dataset:

| Method | ADD-S (%) | ADD (%) | CD (cm) | Time/frame (s) |
|--------|-----------|---------|---------|----------------|
| BundleTrack | 92.54 | 84.91 | - | 0.21 |
| BundleSDF | 92.82 | 84.28 | 0.53 | 0.82 |
| MonoGS (RGB-D) | 20.16 | 15.32 | 2.43 | 0.29 |
| **6DOPE-GS** | **93.79** | **87.82** | **0.15** | **0.22** |

HO3D dataset:

| Method | ADD-S (%) | ADD (%) | CD (cm) | Time/frame (s) |
|--------|-----------|---------|---------|----------------|
| BundleSDF | 94.86 | **89.56** | 0.58 | 2.10 |
| BundleTrack | 93.96 | 77.75 | - | 0.29 |
| **6DOPE-GS** | **95.07** | 84.33 | **0.41** | **0.24** |

### Ablation Study

HO3D ablation:

| Configuration | ADD-S (%) | ADD (%) | CD (cm) |
|---------------|-----------|---------|---------|
| Ours (basic) | 93.52 | 80.25 | 0.44 |
| w/o KF Selection | 94.44 | 82.40 | 0.42 |
| w/o Pruning | 92.48 | 80.87 | 0.44 |
| Ours (3DGS) | 92.51 | 79.49 | 0.47 |
| **Ours (final, 2DGS)** | **95.07** | **84.33** | **0.41** |

### Key Findings

1. Compared to BundleSDF, 6DOPE-GS achieves approximately 5× speedup (0.22 s vs. 0.82 s) while achieving higher ADD-S.
2. 2DGS substantially outperforms 3DGS (ADD-S 95.07 vs. 92.51), attributed to the normal and depth regularization in 2DGS that encourages Gaussians to conform to object surfaces.
3. Both dynamic keyframe selection and percentile-based pruning are indispensable: removing either component degrades performance.
4. MonoGS performs poorly on object-level tracking, demonstrating that scene-level SLAM methods cannot be directly applied to object tracking.
5. In real-time demonstrations, the tracking frequency reaches 3.5–5 Hz, with the Gaussian field updated every 8 seconds.

## Highlights & Insights

- **First method to apply Gaussian Splatting to model-free 6D object tracking and reconstruction**, demonstrating the significant potential of GS in object-level SLAM.
- The **icosahedron anchor strategy** in dynamic keyframe selection is elegant, ensuring spatial coverage of viewing directions.
- MAD-based outlier keyframe filtering is more robust than simple thresholding and represents good engineering practice.
- The overall approach transfers the paradigm of scene-level GS-SLAM to the object level, opening a new research direction.

## Limitations & Future Work

1. The ADD score on HO3D remains lower than that of BundleSDF; insufficient supervision signal due to hand occlusion is the primary cause.
2. Gaussian rasterization is less accurate than differentiable ray casting for gradient computation under large or out-of-plane rotations.
3. The optimized 2D Gaussians are not directly used in online pose graph optimization (only the optimized poses are), resulting in insufficient coupling.
4. The method depends on the segmentation quality of SAM2 and may fail under severe occlusion.
5. Only single-object tracking is currently supported; extension to multi-object scenarios requires further work.

## Related Work & Insights

- The keyframe pose graph optimization framework is inherited from BundleTrack/BundleSDF, with GS replacing the slow Neural Field.
- The distinction from GS-SLAM methods such as MonoGS lies in the object-centric rather than scene-level formulation of 6DOPE-GS.
- The advantage of the 2DGS surfel representation for accurate depth and normal modeling is validated in this work.

## Rating

- Novelty: ⭐⭐⭐⭐ — First application of 2DGS to model-free 6D object tracking, with well-designed keyframe selection and density control.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, detailed ablations, and real-time demonstrations, though evaluation on larger-scale benchmarks is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with complete method descriptions.
- Value: ⭐⭐⭐⭐ — Real-time model-free tracking offers high practical value and advances the application of GS in robotics.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[ICCV 2025\] 3DRealCar: An In-the-wild RGB-D Car Dataset with 360-degree Views](3drealcar_an_in-the-wild_rgb-d_car_dataset_with_360-degree_views.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)

<!-- RELATED:END -->
