---
title: >-
  [Paper Note] CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field
description: >-
  [ECCV 2024][3D Vision][Dense Visual SLAM] This paper proposes CG-SLAM, an efficient dense RGB-D SLAM framework based on a consistency- and geometric-stability-optimized uncertainty-aware 3D Gaussian field, achieving state-of-the-art (SOTA) performance in both localization accuracy and reconstruction quality with a tracking speed of up to 15Hz.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Dense Visual SLAM"
  - "3D Gaussian Splatting"
  - "Uncertainty Modeling"
  - "Real-time Localization and Mapping"
  - "Isotropic Regularization"
date: 2026-05-08
content_hash: e2a02c040fcca6f9
---

# CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field

**Conference**: ECCV 2024  
**arXiv**: [2403.16095](https://arxiv.org/abs/2403.16095)  
**Code**: [Available](https://zju3dv.github.io/cg-slam)  
**Area**: 3D Vision  
**Keywords**: Dense Visual SLAM, 3D Gaussian Splatting, Uncertainty Modeling, Real-time Localization and Mapping, Isotropic Regularization

## TL;DR

This paper proposes CG-SLAM, an efficient dense RGB-D SLAM framework based on a consistency- and geometric-stability-optimized uncertainty-aware 3D Gaussian field, achieving state-of-the-art (SOTA) performance in both localization accuracy and reconstruction quality with a tracking speed of up to 15Hz.

## Background & Motivation

NeRF-based SLAM methods (e.g., NICE-SLAM, Co-SLAM, Point-SLAM) achieve impressive results, but their **volume rendering pipelines are computationally intensive and time-consuming**, only allowing a limited number of camera rays to be sampled. While 3D Gaussian Splatting (3DGS) rasterization rendering is natively more efficient, directly applying it to SLAM encounters three challenges: overfitting, inaccurate geometry, and efficiency bottlenecks. Concurrent works such as GS-SLAM and SplaTAM lack targeted designs for these issues.

## Method

### Overall Architecture

The framework consists of four core modules: a GPU-accelerated rasterizer (based on full pose derivative analysis), uncertainty modeling, consistent mapping, and efficient tracking (sequential tracking + sliding BA).

### Key Designs

#### 1. Multi-modal Rendering

The rasterizer simultaneously renders color maps, alpha-blended depth, median depth (the point where cumulative transmittance first falls below 0.5), and cumulative opacity maps (to detect unobserved regions).

#### 2. Uncertainty Modeling (Core Innovation)

**Uncertainty Map**: Measured by the variance of depth rendering, where a geometric variance loss constrains the Gaussian primitives to lie close to the ground-truth depth.

**Alignment Loss**: Aligns the alpha-blended depth with the median depth, forcing the Gaussian primitive with the largest weight on each pixel ("dominant Gaussian") to appear at the median depth location.

**Gaussian Primitive Uncertainty**: Defined as the weighted average of depth deviations across all dominant pixels within the keyframe window. Primitives exceeding a threshold (0.025) have their opacity reduced before further optimization, and truly irrecoverable ones are removed—an adaptive progressive pruning strategy.

#### 3. Isotropic Regularization

A soft constraint is introduced to encourage Gaussian ellipsoids to remain spherical, keeping the ratio of maximum to minimum scaling factors below a threshold (1.0). This prevents spike-like artifacts and balances tracking accuracy with rendering photorealism.

#### 4. Tracking Module

This work presents the first complete mathematical derivation of pose derivatives in EWA splatting. The **Lie algebra representation** is more suitable for camera tracking within Gaussian fields. Sequential tracking: Initialization based on constant velocity assumption + optimization via re-rendering loss. Sliding BA: NetVLAD is used to determine co-visibility (which is more efficient than view-frustum overlap), enabling joint optimization of camera poses and the scene.

### Loss & Training

Total mapping loss = Color L1 + SSIM + Depth L1 + Alignment Loss + Isotropic Regularization + Geometric Variance. Tracking loss = Color L1 + Depth L1. Gaussian Management: Dense initialization on the first frame, with subsequent insertion of new Gaussians for pixels with low opacity. Hardware: i9-14900K + RTX 4090.

## Key Experimental Results

### Main Results: Replica Tracking Accuracy (ATE RMSE [cm]↓)

| Method | rm-0 | rm-1 | rm-2 | off-0 | off-1 | Avg. |
|------|------|------|------|-------|-------|------|
| NICE-SLAM | 0.97 | 1.31 | 1.07 | 0.88 | 1.00 | 1.06 |
| Co-SLAM | 0.77 | 1.04 | 1.09 | 0.58 | 0.53 | 0.99 |
| Point-SLAM | 0.56 | 0.47 | 0.30 | 0.35 | 0.62 | 0.54 |
| SplaTAM | 0.31 | 0.40 | 0.29 | 0.47 | 0.27 | 0.36 |
| **CG-SLAM** | **0.29** | **0.27** | **0.25** | **0.33** | **0.14** | **0.27** |

The average ATE is 0.27 cm, outperforming SplaTAM by approximately 25%.

### TUM-RGBD Tracking Accuracy (ATE RMSE [cm]↓)

| Method | fr1/desk | fr2/xyz | fr3/office | Avg. |
|------|----------|---------|-----------|------|
| Co-SLAM | 2.7 | 1.9 | 2.6 | 8.38 |
| SplaTAM | 3.35 | 1.24 | 5.16 | 5.48 |
| **CG-SLAM** | **2.43** | **1.20** | **2.45** | **4.0** |

### Replica Mapping Accuracy

| Method | Acc.[cm]↓ | Comp.[cm]↓ | Comp.Ratio↑ |
|------|-----------|------------|-------------|
| Point-SLAM | 1.26 | 3.00 | 88.73% |
| Co-SLAM | 2.10 | **2.08** | **93.44%** |
| **CG-SLAM** | **1.01** | 2.84 | 88.51% |

Reconstruction accuracy is state-of-the-art (1.01 cm), while the completeness is slightly lower than Co-SLAM (due to the lack of global MLP-based hole-filling capabilities).

### Execution Efficiency

| Method | Tracking [ms x it]| System FPS↑ |
|------|--------------|----------|
| NICE-SLAM | 6.19x10 | 0.98 |
| Co-SLAM | 4.45x10 | 14.2 |
| SplaTAM | 41.7x40 | 0.21 |
| **CG-SLAM** | **7.89x15** | **8.5** |
| **CG-SLAM-light** | **3.80x15** | **15.4** |

The lightweight version runs at 15.4 FPS, which is over 70 times sharper/faster than SplaTAM.

### Ablation Study

**Isotropic Loss**: Without this component, tracking fails completely in certain scenes (e.g., off-3), and the error on rm-2 doubles.

**Alignment and Variance Losses**:

| Configuration | ATE↓ | Chamfer Distance↓ |
|------|------|-------------|
| w/o Alignment + Variance | 0.33 | 4.79 |
| **Full** | **0.26** | **3.85** |

The synergistic effect of these two losses reduces the Chamfer distance by 20%.

### Key Findings

1. The uncertainty model effectively eliminates extreme tracking errors and reduces tracking variance.
2. Isotropic regularization is critical for tracking stability; without it, tracking fails in certain scenes.
3. The alignment loss is core to establishing a consistent Gaussian field.
4. The Lie algebra representation outperforms other pose parameterization methods.

## Highlights & Insights

1. **Efficiency and Quality Simultaneously**: First to simultaneously achieve SOTA accuracy and near-real-time speed in 3DGS-SLAM.
2. **Two-Level Uncertainty Modeling**: Pixel-level (variance map) and primitive-level (dominant pixel deviation) modeling complement each other.
3. **Depth Alignment Strategy**: Resolves the fundamental issue of unconstrained Gaussian primitive positioning.
4. **NetVLAD Co-visibility**: Replaces traditional view-frustum overlap detection, offering superior efficiency.
5. **Complete Pose Derivative Derivation**: Provides the first complete theoretical derivation of pose derivatives in EWA splatting.

## Limitations & Future Work

1. **Memory Consumption**: 231.66MB compared to 6.37MB for Co-SLAM.
2. **Weak Hole-filling**: Lacking a global MLP, the completeness for unobserved regions is slightly degraded.
3. **No Loop Closure Detection**: Accumulated drifting in large-scale scenes remains a challenge.
4. **RGB-D Only**: Cannot be directly applied to monocular settings.
5. Future: Adaptive Gaussian management to reduce memory consumption and expansion to RGB-only settings.

## Related Work & Insights

- Inherits the NeRF-SLAM paradigm from NICE-SLAM, Co-SLAM, and Point-SLAM, while leveraging 3DGS to break through efficiency bottlenecks.
- Concurrent works like SplaTAM and GS-SLAM lack designs for maintaining consistency.
- This is the first time uncertainty modeling has been adapted to Gaussian-field SLAM.

## Rating

- Novelty: ⭐⭐⭐⭐ — Depth in uncertainty modeling and consistent Gaussian field designs.
- Value: ⭐⭐⭐⭐⭐ — 15Hz real-time performance with SOTA accuracy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated across 3 datasets, compared against 6+ baselines, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ — Well-structured, mathematically rigorous, and comprehensive experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [\[ECCV 2024\] I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM](i2-slam_inverting_imaging_process_for_robust_photorealistic_dense_slam.md)
- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](../../CVPR2025/3d_vision/varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[ECCV 2024\] Deep Patch Visual SLAM](deep_patch_visual_slam.md)
- [\[CVPR 2025\] MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](../../CVPR2025/3d_vision/magic-slam_multi-agent_gaussian_globally_consistent_slam.md)

</div>

<!-- RELATED:END -->
