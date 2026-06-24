---
title: >-
  [Paper Note] VTGaussian-SLAM: RGBD SLAM for Large Scale Scenes with Splatting View-Tied 3D Gaussians
description: >-
  [ICML2025][3D Vision][SLAM] This paper proposes View-Tied 3D Gaussians, which bind Gaussians to depth pixels and simplify them to spherical shapes to significantly reduce storage footprint. Combined with a tracking/mapping strategy that only optimizes Gaussians associated with adjacent views, a scalable RGBD SLAM system is realized for large-scale scenes.
tags:
  - "ICML2025"
  - "3D Vision"
  - "SLAM"
  - "3D Gaussian Splatting"
  - "View-Tied Gaussians"
  - "Large-scale Scenes"
  - "RGBD"
  - "Camera Tracking"
  - "Scene Reconstruction"
date: 2026-05-08
content_hash: ed101248af9fc6d3
---

# VTGaussian-SLAM: RGBD SLAM for Large Scale Scenes with Splatting View-Tied 3D Gaussians

**Conference**: ICML2025  
**arXiv**: [2506.02741](https://arxiv.org/abs/2506.02741)  
**Code**: [Project Page](https://machineperceptionlab.github.io/VTGaussian-SLAM-Project)  
**Area**: 3D Vision  
**Keywords**: SLAM, 3D Gaussian Splatting, View-Tied Gaussians, Large-scale Scenes, RGBD, Camera Tracking, Scene Reconstruction

## TL;DR

This paper proposes View-Tied 3D Gaussians, which bind Gaussians to depth pixels and simplify them to spherical shapes to significantly reduce storage footprint. Combined with a tracking/mapping strategy that only optimizes Gaussians associated with adjacent views, a scalable RGBD SLAM system is realized for large-scale scenes.

## Background & Motivation

- **Core Task of SLAM**: Simultaneously estimating camera poses and reconstructing maps from RGBD images is a fundamental problem in computer vision, 3D reconstruction, and robotics.
- **Limitations of Prior Work in NeRF**: NeRF-based SLAM methods perform continuous scene representation through volume rendering, but ray tracing is computationally expensive and struggles to achieve real-time speed.
- **Introduction & Advantages of 3DGS**: 3D Gaussian Splatting (3DGS) achieves high-quality real-time rendering through differentiable rasterization, offering a new paradigm for SLAM scene representation.
- **Limitations of Prior Work in 3DGS-SLAM**: Current approaches must maintain all 3D Gaussians spanning the entire scene within restricted GPU memory, and repeatedly optimize all Gaussians throughout training to preserve color and geometric consistency with previous frames. This prevents scalability to extremely large-scale scenes.
- **Memory Efficiency Issues**: The original 3DGS requires storing numerous parameters per Gaussian, such as position (3), rotation (4), scaling (6), and color (spherical harmonics), limiting the maximum number of Gaussians.
- **Key Challenge**: Larger scenes require more Gaussians while GPU memory is limited; at the same time, maintaining global consistency further increases the optimization burden, establishing a fundamental conflict.

## Method

### Overall Architecture

VTGaussian-SLAM comprises two core components: **View-Tied 3D Gaussians representation** and a **novel tracking/mapping strategy** based on this representation. Inputting an RGBD stream, the system binds each frame's depth pixels with simplified spherical Gaussians, loading and optimizing only the subset of Gaussians relevant to the current viewpoint as needed to process large-scale scenes.

### Key Designs

#### Key Design 1: View-Tied 3D Gaussians

| Property | Function | Mechanism | Design Motivation |
|------|--------|----------|----------|
| Position Binding | Binds Gaussians to depth pixels, with position uniquely determined by depth values and camera poses | Position = R * unproj(u,v,d) + t | Eliminates the need to learn and store position parameters, avoiding density control |
| Spherical Simplification | Simplifies ellipsoidal Gaussians into spherical ones, removing rotation and multi-dimensional variance parameters | Isotropic Gaussians, requiring only a single radius parameter | Further reduces memory usage; spherical shapes are sufficient to describe local geometry in SLAM scenarios |
| Attribute Pruning | Each Gaussian retains only color, opacity, and radius | Removes position (3), rotation (4), and covariance (6), saving 13 parameters in total | Accommodates more Gaussians within limited GPU memory to describe finer local details |

#### Key Design 2: Novel Tracking Strategy

| Property | Function | Mechanism | Design Motivation |
|------|--------|----------|----------|
| Local Optimization | Renders and optimizes only the Gaussians related to the most recent views | Avoids maintaining the global consistency of Gaussians across the entire scene | Prevents out-of-memory errors and efficiency issues caused by optimizing all Gaussians in large scenes |
| Pose Estimation | Optimizes camera poses by minimizing the difference between the rendered and observed images of the current frame | Utilizes differentiable rendering gradients of view-tied Gaussians backpropagated to pose parameters | Gaussian positions are determined by poses; pose updates automatically prompt Gaussian position updates |
| Keyframe Selection | Selects a representative subset of keyframes for tracking reference | Based on changes in viewpoint and overlap criteria | Reduces the number of historical keyframe Gaussians that need to be loaded into memory |

#### Key Design 3: Novel Mapping Strategy

- **Function**: Optimizes only the color, opacity, and radius parameters of the view-tied Gaussians associated with each new keyframe.
- **Mechanism**: The mapping process does not need to maintain consistency constraints across all historical keyframes, but naturally ensures local consistency through the view-tied binding relationships.
- **Design Motivation**: Traditional methods require keeping all Gaussians learnable during mapping to maintain global consistency, which is the main source of memory consumption. The view-tied design binds Gaussians to specific views, meaning Gaussians moved out of memory do not affect current frame optimization.

### Loss & Training

- **Rendering Loss**: Combines photographic loss of RGB images and depth maps to supervise the optimization of Gaussian attributes.
- **Incremental Processing**: Processes the RGBD stream frame-by-frame, performing tracking first to estimate poses, followed by mapping to optimize local Gaussians.
- **Memory Management**: Loads only the currently relevant subset of Gaussians into the GPU, which can be released after processing, decoupling memory usage from the scene scale.

## Key Experimental Results

### Main Results: Rendering Quality & Tracking Accuracy Comparison

| Method | Scene Representation | Scalability | Rendering Quality | Tracking Accuracy | Requires Global Gaussian Maintenance |
|------|----------|----------|----------|----------|----------------|
| SplaTAM | 3DGS | Poor | Moderate | Moderate | Yes |
| MonoGS | 3DGS | Poor | Good | Good | Yes |
| Gaussian-SLAM | 3DGS | Moderate | Good | Good | Yes |
| Photo-SLAM | 3DGS+Implicit | Moderate | Excellent | Good | Yes |
| **VTGaussian-SLAM** | **View-Tied 3DGS** | **Strong** | **Best** | **Best** | **No** |

### Ablation Study: Component Contributions

| Configuration | Rendering Quality Change | Tracking Accuracy Change | Description |
|------|-------------|-------------|------|
| Full Model | Baseline | Baseline | All components of VTGaussian-SLAM |
| Without Spherical Simplification (Ellipsoid instead) | No significant improvement | Comparable | Illustrates that spherical simplification is sufficient while saving a large number of parameters |
| Without View-Tied Binding (Learning Positions) | Degraded | Degraded | Reverts to traditional 3DGS, losing the scalability advantage |
| Using Global Optimization Strategy | Slight improvement | Comparable | However, memory consumption increases significantly, making it unscalable |

### Key Findings

- The view-tied strategy enables the use of several times more Gaussians than traditional methods within the same GPU memory, compensating for representation capability lost through simplification.
- In large-scale scenes (multi-room/long sequences), traditional 3DGS-SLAM methods fail to run due to out-of-memory errors, whereas VTGaussian-SLAM continues to function normally.
- Spherical simplification incurs virtually no loss in rendering quality in SLAM scenarios, indicating that Gaussians in SLAM primarily serve as local color/geometric samples rather than precise ellipsoidal models.

## Highlights & Insights

1. **Co-design of Representation and Algorithm**: View-tied Gaussians represent more than just micro-level storage optimization; they fundamentally shift the tracking/mapping paradigm from global consistency optimization to local incremental optimization.
2. **Trading Constraints for Degrees of Freedom**: Bounding Gaussian positions to depth pixels (a strong constraint) yields immense freedom in memory usage, serving as a classic example of design by subtraction.
3. **Simplification is not Degradation**: Spherical Gaussians may seem like a compromise on representation capacity, but allowing more Gaussians under GPU memory constraints actually boosts overall quality, demonstrating a trade-off between quantity and quality.

## Limitations & Future Work

- **Dependence on Depth Input**: The method heavily relies on depth maps provided by RGBD sensors and cannot be directly applied to monocular RGB SLAM scenarios.
- **No Loop Closure**: The current system does not integrate a loop closure module, which may cause drift accumulation in long sequence trajectories.
- **Dynamic Scenes**: The paper does not address dynamic objects; the view-tied strategy may produce artifacts in dynamic environments.
- **Boundary of the Spherical Assumption**: For thin structures (e.g., railings, wires), spherical Gaussians may require a higher count to approximate, indicating an efficiency limit.
- **Scene Editing Capability**: The view-tied binding makes it difficult to edit or reuse Gaussians independently of their original views.

## Related Work & Insights

- **SplaTAM / MonoGS**: Represents the mainstream paradigm of current 3DGS-SLAM methods, which require maintaining global Gaussians. This study specifically targets and improves their scalability bottlenecks.
- **NICE-SLAM / Co-SLAM**: NeRF-based SLAM methods that use hierarchical feature grids for local updates, inspiring the local optimization strategy in this work.
- **Dynamic 3DGS**: The concept of binding Gaussians to physical entities is also applied in dynamic scene reconstruction, but the binding strategy proposed for SLAM in this paper (binding to depth pixels) is more concise.
- **Insight**: The core philosophy of being view-tied—binding representation to observation—can be transferred to other 3DGS applications requiring large-scale scene processing, such as autonomous driving scene reconstruction.

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | View-tied Gaussians are a fundamental improvement to the 3DGS-SLAM paradigm |
| Technical Depth | ⭐⭐⭐⭐ | Co-design of representation and optimization strategy, with contributions at both theoretical and system levels |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Outperforms state-of-the-art methods on standard benchmarks, with ablation studies thoroughly validating each component |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured paper with intuitive diagrams and clear motivation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](../../CVPR2025/3d_vision/wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)
- [\[CVPR 2026\] SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM](../../CVPR2026/3d_vision/sgad-slam_splatting_gaussians_at_adjusted_depth_for_better_radiance_fields_in_rg.md)
- [\[CVPR 2026\] ODGS-SLAM: Omnidirectional Gaussian Splatting SLAM](../../CVPR2026/3d_vision/odgs-slam_omnidirectional_gaussian_splatting_slam.md)
- [\[ECCV 2024\] SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](../../ECCV2024/3d_vision/sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](../../CVPR2025/3d_vision/varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)

</div>

<!-- RELATED:END -->
