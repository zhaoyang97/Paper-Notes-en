---
title: >-
  [Paper Note] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] By incorporating full 3D evaluation (rather than 2D splat approximations) into every stage of the 3DGS rendering pipeline, this work proposes an adaptive 3D smoothing filter, view-space bounding computation, and frustum-based tile culling to jointly address aliasing, projection artifacts, and popping artifacts in 3DGS. The method substantially outperforms existing approaches under out-of-distribution (OOD) viewpoints while maintaining real-time rendering (>100 FPS).
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Anti-Aliasing
  - Artifact-Free Rendering
  - Frustum Culling
  - View Consistency
date: 2026-05-08
content_hash: e98ed5af91a55cf4
---

# AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering

**Conference**: ICCV 2025
**arXiv**: [2504.12811](https://arxiv.org/abs/2504.12811)
**Code**: [GitHub](https://github.com/DerThomy/AAA-Gaussians)
**Area**: 3D Vision / Novel View Synthesis / Real-Time Rendering
**Keywords**: 3D Gaussian Splatting, Anti-Aliasing, Artifact-Free Rendering, Frustum Culling, View Consistency

## TL;DR
By incorporating full 3D evaluation (rather than 2D splat approximations) into every stage of the 3DGS rendering pipeline, this work proposes an adaptive 3D smoothing filter, view-space bounding computation, and frustum-based tile culling to jointly address aliasing, projection artifacts, and popping artifacts in 3DGS. The method substantially outperforms existing approaches under out-of-distribution (OOD) viewpoints while maintaining real-time rendering (>100 FPS).

## Background & Motivation
3DGS achieves efficient rendering by projecting 3D Gaussians onto 2D splats, but this approximation introduces multiple types of artifacts: (1) distortion artifacts at wide angles and image boundaries due to affine projection approximation; (2) aliasing artifacts caused by the absence of anti-aliasing under scale changes; (3) popping artifacts when Gaussians extend behind the view frustum; and (4) view inconsistency caused by global sorting. Existing methods each address only one or two of these issues and still fail under OOD viewpoints.

## Core Problem
How to unify the elimination of all rendering artifact types through full 3D Gaussian evaluation while preserving rasterization efficiency? The central challenge is that 2D anti-aliasing approaches (e.g., Mip-Splatting's screen-space Mip filter) cannot be directly applied to 3D evaluation.

## Method

### Overall Architecture
Built upon StopThePop's hierarchical rasterizer, the method replaces bounding, culling, depth evaluation, contribution estimation, and anti-aliasing with fully 3D-aware implementations. MCMC densification is adopted. The rendering pipeline proceeds as: 3D bounding (view-space angles) → 3D frustum culling → per-pixel 3D Gaussian evaluation → hierarchical sort-and-blend.

### Key Designs
1. **Adaptive 3D Smoothing Filter**: Replaces Mip-Splatting's 2D screen-space Mip filter. A key finding is that directly using volumetric change as amplitude scaling causes highly anisotropic Gaussians to become excessively transparent. The proposed approach instead scales the amplitude solely based on **area variation perpendicular to the viewing ray direction** (rather than volumetric change). An efficient closed-form solution (Eq. 10–12) is derived via Schur complement and Hartley analysis, avoiding matrix inversion. An adaptive switching mechanism is applied between the maximum sampling frequency at training time $\hat{v}_{train}$ and the current rendering frequency $\hat{v}$.

2. **View-Space Perspective-Correct Bounding**: Gaussian bounding is computed in view space rather than screen space. Plane fitting is performed on the ellipsoid to solve for angles $\theta_{1,2}$ and $\phi_{1,2}$ (Eq. 14–15). Screen-space methods fail when a Gaussian extends behind the image plane (causing popping), whereas view-space angular computation is inherently stable. The resulting angles are then converted to screen coordinates for tile assignment.

3. **Frustum-Based 3D Tile Culling**: Extends StopThePop's 2D tile culling to 3D. A four-plane frustum $\mathcal{F}$ is constructed per tile; the maximum contribution point within the frustum is located in the Gaussian's canonical space and checked against $\rho(\mathbf{x})^2 < \tau_\rho$. An implementation optimization projects only the 2 nearest planes and 3 edges (rather than all 4 planes and 4 edges), yielding significant performance gains.

### Loss & Training
- Standard 3DGS training objective (L1 + SSIM) with MCMC densification
- Filter parameter $k=0.3$ (consistent with Mip-Splatting)
- Truncation threshold $\tau_\rho$ used for bounding and culling computations

## Key Experimental Results

### Standard Datasets (In-Distribution Viewpoints)

| Method | Mip-NeRF360 PSNR | T&T PSNR | Deep Blending PSNR |
|--------|-------------------|----------|-------------------|
| 3DGS | 27.443 | 23.734 | 29.510 |
| StopThePop | 27.304 | 23.226 | 29.929 |
| Mip-Splatting | 27.540 | 23.821 | 29.660 |
| MCMC | 28.027 | **24.642** | 29.727 |
| **Ours** | **27.835** | 23.582 | **30.485** |

### Large-FOV Evaluation (3× FOV, OOD)

| Method | Mip-NeRF360 PSNR | T&T PSNR |
|--------|-------------------|----------|
| 3DGS | 17.112 | - |
| MCMC | 14.369 | - |
| Taming 3DGS | 11.545 | - |
| **Ours** | **23.583** | - |

**OOD quality improvement is substantial**: at 3× FOV, PSNR improves from 14.369 (MCMC) to 23.583 (ours), a gain of +9.2 dB.

### Multi-Resolution Evaluation (Bicycle Scene)

| Resolution | MCMC PSNR | Ours PSNR | Ours w/o AA PSNR |
|------------|-----------|-----------|-----------------|
| 1/2× | 21.15 | **26.73** | 20.99 |
| 1× | 28.98 | 32.12 | 28.54 |
| 2× | 25.69 | **25.74** | 25.65 |

### Rendering Performance (RTX 4090, ms/frame)

| Method | Mip-NeRF360 | T&T |
|--------|-------------|-----|
| Ours | 7.72 | 5.81 |
| MCMC | 6.79 | 4.43 |
| Ours w/o culling | 14.40 | 8.88 |

### Ablation Study
- Removing hierarchical sorting: marginally improves in-distribution PSNR (overfitting) but introduces severe popping artifacts.
- Removing AA: minor impact in-distribution, but significant quality degradation at low/high resolutions.
- Removing 3D evaluation: produces projection distortions at increased FOV.
- Removing frustum culling: substantial performance degradation (7.72→14.40 ms) with no quality impact.
- Components are complementary: artifact-free rendering requires all components in combination.

## Highlights & Insights
- **Perpendicular area vs. volume for amplitude scaling**: A key insight for the 3D filter — scale changes along the viewing ray should not affect amplitude; only area variation perpendicular to the ray is relevant. The mathematical derivation via Schur complement is elegant.
- **View-space angles vs. screen-space coordinates**: Fundamentally eliminates instabilities when Gaussians cross the near plane, representing the principled approach to bounding computation.
- **Unified resolution of all artifact types**: The first rasterization method to address aliasing, projection distortion, popping, and culling within a single framework.
- **Value of OOD evaluation**: In-distribution metric differences are modest, but OOD scenarios (large FOV, resolution variation) reveal the true robustness of the proposed approach.

## Limitations & Future Work
- Limited improvement on in-distribution metrics (comparable to MCMC on standard benchmarks).
- View-space bounding remains tied to the pinhole camera model; fisheye and other camera types are not supported.
- Hierarchical sorting introduces additional performance overhead (nearly 2× slower than no sorting, which however produces popping).
- Stronger view consistency reduces overfitting capacity, potentially necessitating richer view-dependent encoding (e.g., higher-order spherical harmonics).

## Related Work & Insights
- **vs. Mip-Splatting**: Both target anti-aliasing, but Mip-Splatting operates in 2D screen space and cannot support 3D evaluation; the proposed 3D filter provides substantially greater robustness under OOD viewpoints.
- **vs. StopThePop**: Shares the hierarchical sorting framework, but StopThePop still employs 2D bounding and 2D culling; this work elevates all operations to 3D.
- **vs. Hybrid Transparency (Hahlbohm et al.)**: Both perform 3D evaluation and precise bounding, but Hybrid Transparency still exhibits pop-in at boundaries (by discarding out-of-bounds Gaussians), whereas this work fully resolves the issue through view-space bounding.

The view-space operation paradigm is particularly relevant for VR and large-FOV applications. A promising future direction is to transfer the "orthogonal decomposition perpendicular to the ray" concept from the adaptive 3D filter to LOD/anti-aliasing in other 3D representations.

## Rating
- Novelty: ⭐⭐⭐⭐ — Novel amplitude scaling derivation for the 3D AA filter; elegant view-space bounding computation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive in-distribution and OOD evaluation, detailed ablations, multiple datasets and scenes, and runtime profiling.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous mathematical derivations, thorough problem analysis, and clear illustrations.
- Value: ⭐⭐⭐⭐ — Substantial improvement in 3DGS rendering quality and robustness; open-source code; an important reference implementation for the 3DGS rendering pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
