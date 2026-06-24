---
title: >-
  [Paper Note] LiDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-Simulation
description: >-
  [CVPR 2025][Autonomous Driving][LiDAR simulation] This paper proposes LiDAR-RT, which integrates 3D Gaussian primitives with NVIDIA OptiX hardware-accelerated ray tracing to achieve real-time and physically accurate LiDAR re-simulation in dynamic driving scenes for the first time. It achieves a rendering speed of 30 FPS and requires only 2 hours of training, significantly surpassing NeRF-based approaches (0.2 FPS and 15 hours).
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "LiDAR simulation"
  - "3D Gaussian"
  - "Ray Tracing"
  - "Dynamic Scene"
  - "Novel View Synthesis"
  - "OptiX"
date: 2026-05-08
content_hash: f00fb03a9dac3926
---

# LiDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-Simulation

**Conference**: CVPR 2025  
**arXiv**: [2412.15199](https://arxiv.org/abs/2412.15199)  
**Code**: [zju3dv/LiDAR-RT](https://github.com/zju3dv/LiDAR-RT)  
**Area**: Autonomous Driving  
**Keywords**: LiDAR simulation, 3D Gaussian, Ray Tracing, Dynamic Scene, Novel View Synthesis, OptiX

## TL;DR

This paper proposes LiDAR-RT, which integrates 3D Gaussian primitives with NVIDIA OptiX hardware-accelerated ray tracing to achieve real-time and physically accurate LiDAR re-simulation in dynamic driving scenes for the first time. It achieves a rendering speed of 30 FPS and requires only 2 hours of training, significantly surpassing NeRF-based approaches (0.2 FPS and 15 hours).

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: LiDAR sensors are core components for 3D perception in autonomous driving, and LiDAR simulation is critical for scaling training data and validating perception algorithms. Existing methods face the following limitations:

1. **Traditional Simulators** (CARLA, AirSim): Suffer from severe sim-to-real gaps and require substantial manual creation of virtual assets.
2. **Explicit Reconstruction Methods** (LiDARsim, PCGen): Rely on explicit representations like surfels/meshes, are sensitive to geometric quality, and only support static scenes.
3. **NeRF-based Methods** (NFL, LiDAR4D, DyNFL): Although offering good rendering quality, they incur prohibitive training costs (15+ hours) and extremely slow rendering speeds (0.2 FPS), making it difficult to handle complex dynamic scenes.

Core Motivation: Can the efficiency of 3D Gaussian Splatting be combined with the physical accuracy of ray tracing to realize real-time LiDAR simulation?

## Method

### Overall Architecture

LiDAR-RT consists of four components:
1. **Dynamic Scene Representation**: Decomposes the scene into a static background and multiple dynamic objects, each represented by Gaussian primitives.
2. **Gaussian Ray Tracing**: Forward rendering based on BVH acceleration structures and proxy geometries.
3. **Differentiable Rendering**: A forward-order backpropagation strategy supporting end-to-end optimization.
4. **Ray Drop Optimization**: A UNet-based network to refine sensor-level ray drop effects.

### Key Designs

**1. Enhanced Gaussian Primitives**

Building on standard 3DGS parameters (position $\mu$, covariance $\Sigma$, opacity $\sigma$), LiDAR physical parameters are introduced:
- **Intensity $\zeta$**: View-dependent intensity is modeled using SH coefficients.
- **Ray Drop Probability $\beta$**: Modeled via two logit values $(\beta_{drop}, \beta_{hit})$ and a softmax function, similarly represented by SH coefficients for view dependency.

Dynamic objects are managed via a scene graph: Gaussian parameters are defined in local coordinate systems and transformed to the world coordinate system using tracked rotation matrices and translation vectors.

**2. Ray Tracing based on Proxy Geometry**

- Employs 2D Gaussian discs as primitives, represented by a pair of coplanar triangles serving as proxy geometries.
- Compared to AABB bounding boxes, coplanar triangles tightly wrap Gaussian primitives, reducing the number of meshes.
- The sampling position equals the ray intersection directly, eliminating the need for approximation.
- Utilizes the NVIDIA OptiX framework for BVH construction and hardware-accelerated ray casting.

**3. Chunk Rendering Strategy**

Divides each ray into multiple chunks:
- Each chunk contains a fixed number of intersections, sorted only within the chunk.
- The Gaussian response and LiDAR properties ($\zeta$, $\beta$) are calculated for each intersection and accumulated using the volume rendering formulation.
- Traversal stops when all Gaussians are traversed or the accumulated transmittance falls below a threshold.

### Loss & Training

$$\mathcal{L} = \lambda_d \mathcal{L}_d + \lambda_i \mathcal{L}_i + \lambda_r \mathcal{L}_r + \lambda_{CD} \mathcal{L}_{CD}$$

- $\mathcal{L}_d$: Depth L1 loss
- $\mathcal{L}_i$: Intensity L1 loss
- $\mathcal{L}_r$: Ray drop BCE loss
- $\mathcal{L}_{CD}$: Chamfer Distance loss for joint supervision of scene geometry.

Ray drop is divided into scene-level (environmental factors like highly reflective materials) and sensor-level (hardware noise); the latter is refined via UNet post-processing.

## Key Experimental Results

### Waymo Open Dataset (64×2650 Resolution)

### Main Results

| Method | FPS | Storage | Depth RMSE↓ | Depth MedAE↓ | SSIM↑ | CD↓ | F-score↑ |
|------|-----|------|-------------|--------------|-------|-----|----------|
| LiDAR-NeRF | 0.98 | 1.6GB | 7.726 | 0.052 | 0.682 | 0.182 | 0.918 |
| DyNFL | 0.21 | 14.9GB | 6.979 | 0.039 | 0.708 | 0.118 | 0.779 |
| LiDAR4D | 0.17 | 7.7GB | 6.623 | 0.038 | 0.701 | 0.106 | 0.944 |
| **Ours** | **20.1** | **1.37GB** | **6.458** | **0.034** | **0.733** | **0.100** | **0.946** |

### Key Findings

- **Speed**: LiDAR-RT (20.1 FPS) vs LiDAR4D (0.17 FPS) — **118× acceleration**
- **Storage**: 1.37 GB vs 14.9 GB (DyNFL) — **10× compression**
- **Training**: ~2 hours vs 15 hours (LiDAR4D) — **7.5× speedup**
- **Rendering Quality**: Outperforms or matches state-of-the-art methods in depth and point cloud metrics.

### KITTI-360 Dataset

Achieves superior depth and point cloud rendering quality on KITTI-360 as well, while supporting flexible scene editing (object removal, addition, and sensor configuration changes).

## Highlights & Insights

1. **Innovative Technical Route**: Combines the highly efficient representation of 3DGS with physical-level ray tracing for LiDAR simulation for the first time, resolving the inherent limitation of rasterization in handling cylindrical range image projections.
2. **Hardware-Accelerated Engineering Implementation**: Based on OptiX BVH construction and any-hit program design, fully unleashing the hardware capabilities of GPU RT cores for LiDAR rendering tasks.
3. **Forward-Order Backpropagation**: Ingeniously solves the challenge in ray tracing where maintaining a global sorting buffer is not possible, unlike in tile-based rasterizers.
4. **High Practicality**: Supports scene editing (adding/removing objects, modifying sensor parameters), enabling direct application to simulation data augmentation.

## Limitations & Future Work

1. Modeling of dynamic objects relies on accurate tracking bounding boxes; tracking quality directly affects reconstruction results.
2. Density control strategies of Gaussian primitives (splitting/pruning) are directly inherited from 3DGS, without optimization for the sparse nature of LiDAR.
3. UNet post-processing adds inference overhead, compromising the elegance of an end-to-end framework.
4. Validation is limited to Waymo and KITTI-360 datasets, without evaluating cross-dataset generalization.

## Related Work & Insights

- **LiDAR Simulation**: LiDARsim → PCGen → NFL → LiDAR4D → DyNFL
- **Dynamic Scene Reconstruction**: 3DGS → S3Gaussian → OmniRe → PVG
- **Gaussian Ray Tracing**: 3DGRT, Gaussian Ray Tracing (GRT) — but these are only used for camera sensors.
- **LiDAR Physics Modeling**: NFL first modeled the physical characteristics of LiDAR sensors in detail (intensity, ray drop, etc.).

## Rating

- **Novelty**: 4/5 — The first to apply Gaussian + ray tracing to LiDAR simulation, featuring an innovative technical route.
- **Effectiveness**: 5/5 — Speed is increased by a hundredfold with no decline in quality, proving immense practical value.
- **Clarity**: 4/5 — Rendering pipeline description is detailed, and proxy geometry design is clearly illustrated.
- **Significance**: 5/5 — Real-time LiDAR simulation is a critical requirement for autonomous driving simulation.

## Related Papers

- [\[CVPR 2025\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)
- [\[AAAI 2026\] LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors](../../AAAI2026/autonomous_driving/lidar-gsimproving_lidar_gaussian_reconstruction_via_diffusion_priors.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](../../ICCV2025/autonomous_driving/splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[CVPR 2025\] Zero-Shot 4D Lidar Panoptic Segmentation](zero-shot_4d_lidar_panoptic_segmentation.md)
- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](../../ICCV2025/autonomous_driving/gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)
- [\[AAAI 2026\] LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors](../../AAAI2026/autonomous_driving/lidar-gsimproving_lidar_gaussian_reconstruction_via_diffusion_priors.md)
- [\[ICLR 2026\] SimULi: Real-Time LiDAR and Camera Simulation with Unscented Transforms](../../ICLR2026/autonomous_driving/simuli_real-time_lidar_and_camera_simulation_with_unscented_transforms.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](../../ICCV2025/autonomous_driving/splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[CVPR 2025\] Zero-Shot 4D Lidar Panoptic Segmentation](zero-shot_4d_lidar_panoptic_segmentation.md)

</div>

<!-- RELATED:END -->
