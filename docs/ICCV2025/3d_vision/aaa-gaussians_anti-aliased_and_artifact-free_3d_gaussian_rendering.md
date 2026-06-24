---
title: >-
  [Paper Note] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] AAA-Gaussians proposes a unified 3D Gaussian rasterization framework that simultaneously addresses the three persistent problems of 3DGS—aliasing, projection distortion, and popping artifacts—through an adaptive 3D smoothing filter, view-space perspective-correct bounding computation, and frustum-based 3D culling, all within a single framework. The method substantially outperforms competing approaches under out-of-distribution vie…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "anti-aliasing"
  - "artifact removal"
  - "frustum culling"
  - "real-time rendering"
date: 2026-05-08
content_hash: 4600386c08c2ffb4
---

# AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering

**Conference**: ICCV 2025
**arXiv**: [2504.12811](https://arxiv.org/abs/2504.12811)  
**Code**: [https://github.com/DerThomy/AAA-Gaussians](https://github.com/DerThomy/AAA-Gaussians)  
**Area**: 3D Vision / 3D Gaussian Splatting
**Keywords**: 3D Gaussian Splatting, anti-aliasing, artifact removal, frustum culling, real-time rendering

## TL;DR
AAA-Gaussians proposes a unified 3D Gaussian rasterization framework that simultaneously addresses the three persistent problems of 3DGS—aliasing, projection distortion, and popping artifacts—through an adaptive 3D smoothing filter, view-space perspective-correct bounding computation, and frustum-based 3D culling, all within a single framework. The method substantially outperforms competing approaches under out-of-distribution viewpoint evaluation while maintaining real-time rendering performance.

## Background & Motivation
- **Background**: 3D Gaussian Splatting (3DGS) has fundamentally transformed inverse rendering by enabling real-time rendering through efficient rasterization. However, the original 3DGS employs an affine approximation when projecting 3D Gaussians to 2D splats, which introduces several categories of artifacts.
- **Three Core Artifact Categories**:
    - (1) **Projection Distortion**: Approximating 3D Gaussians as 2D splats produces cloud-like distortions at large FOVs and near image borders.
    - (2) **Aliasing Artifacts**: Discrepancies between training and test resolutions cause flickering and undersampling.
    - (3) **Popping/Flickering Artifacts**: Imprecision in global depth sorting causes abrupt changes in blending order during camera rotation.
- **Fragmentation of Existing Solutions**: StopThePop addresses sorting, Mip-Splatting addresses aliasing, and 3D evaluation methods address distortion—yet no single method handles all three simultaneously. Some approaches introduce new problems while solving one (e.g., 3D evaluation is incompatible with 2D anti-aliasing filters).
- **Key Challenge**: 3D evaluation (computing Gaussian contributions along rays) avoids projection distortion, but existing 3D anti-aliasing schemes and 2D bounding computation methods conflict with this approach, incur performance penalties, or introduce new artifacts.
- **Key Insight**: This work advocates treating the 3D nature of Gaussians comprehensively throughout the entire 3DGS rendering pipeline—performing anti-aliasing, bounding, culling, and sorting entirely in 3D.

## Method

### Overall Architecture
The method builds upon the hierarchical-sort rasterizer (StopThePop) and screen-space planar 3D evaluation (Hybrid Transparency), replacing their bounding computation, culling, depth evaluation, contribution estimation, and anti-aliasing with fully 3D implementations. MCMC is used for point cloud densification.

### Key Designs

1. **Adaptive 3D Anti-Aliasing Filter**:

    - **Function**: Resolves aliasing artifacts under 3D evaluation, replacing the 2D Mip filter that cannot be directly applied in this setting.
    - **Core Problem**: Naively recomputing the smoothing filter in 3D causes Gaussians to become excessively transparent—because the amplitude scales with volumetric changes (the product of all three axes in Eq. (8)), whereas 3D evaluation only samples the maximum contribution point along the ray rather than integrating over it.
    - **Solution**: The amplitude is adjusted based solely on area changes perpendicular to the viewing ray direction $\mathbf{d}$:
    $\hat{\mathcal{G}}_\perp(\mathbf{x}) = \sqrt{\frac{|\mathbf{\Sigma}_\perp|}{|\hat{\mathbf{\Sigma}}_\perp|}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \hat{\mathbf{\Sigma}}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$
    - Closed-form perpendicular scaling factor: $\sqrt{\frac{|\mathbf{\Sigma}| \cdot \mathbf{d}^\top\mathbf{\Sigma}^{-1}\mathbf{d}}{|\hat{\mathbf{\Sigma}}| \cdot \mathbf{d}^\top\hat{\mathbf{\Sigma}}^{-1}\mathbf{d}}}$
    - Adaptive mechanism: $\hat{v}' = \min(\hat{v}_{\text{train}}, \hat{v})$, where $\hat{v} = f/d$, ensuring that approaching views do not shrink excessively while distant views are filtered effectively.
    - **Design Motivation**: Excluding scaling variation along the ray direction from the amplitude computation prevents highly anisotropic Gaussians from becoming overly transparent.

2. **View-Space Perspective-Correct Bounding**:

    - **Function**: Resolves popping artifacts caused by Gaussians extending behind the near plane.
    - **Core Problem**: Hahlbohm et al. solve for bounding planes in screen space and discard any Gaussian whose z-extent exceeds the near/far planes, resulting in popping at image borders.
    - **Solution**: Bounding planes are instead fitted in view space using an angular parameterization:
    $\theta_{1,2} = \tan^{-1}\left(\frac{s_{1,3} \pm \sqrt{s_{1,3}^2 - s_{1,1}s_{3,3}}}{s_{3,3}}\right)$
    - Angular ranges are clipped to $[-\pi/2+\epsilon, \pi/2-\epsilon]$, ensuring correct handling even when Gaussians extend far outside the view frustum.
    - **Degenerate Cases**: When the ellipsoid intersects a coordinate axis, the bounding region is conservatively set to full-screen; when the camera is inside the ellipsoid, the Gaussian is discarded.
    - **Design Motivation**: The angular parameterization in view space naturally handles Gaussians that penetrate the near plane, a case where screen-space approaches fail—critical for practical VR rendering scenarios.

3. **Frustum-Based 3D Culling**:

    - **Function**: Elevates 2D tile-based culling to full 3D frustum culling, accelerating rendering and reducing sorting overhead.
    - **Mechanism**: A 3D frustum $\mathcal{F}$ is constructed for each tile, defined by 4 screen-space planes. The maximum contribution point within the frustum is then found in the normalized Gaussian space.
    - **Culling Criterion**: $\min_{\mathbf{x}\in\mathcal{F}} \rho(\mathbf{x})^2 < \tau_\rho$; a Gaussian is discarded if its maximum contribution falls below the threshold.
    - **Optimization**: Only the nearest x/y planes (2 planes + 3 edges) are projected, rather than all 4 planes and 4 edges.
    - **Additional Use**: The same frustum culling is applied during global preprocessing to discard Gaussians that do not intersect the full view frustum.
    - **Design Motivation**: 2D tile culling performs poorly for elongated, non-axis-aligned Gaussians; 3D culling substantially reduces invalid Gaussian–tile pairs and significantly decreases sorting overhead in hierarchical sorting.

### Loss & Training
- Standard 3DGS training procedure is followed with MCMC densification.
- The 3D anti-aliasing filter kernel size is set to $k = 0.3$, consistent with Mip-Splatting.
- All components are enabled during training to produce view-consistent representations.

## Key Experimental Results

### Main Results (Standard In-Distribution Evaluation)

| Dataset | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Artifact-Free |
|--------|------|--------|--------|---------|--------|
| Mip-NeRF 360 | 3DGS | 27.44 | 0.814 | 0.215 | ✗ |
| Mip-NeRF 360 | MCMC | 28.03 | 0.836 | 0.187 | ✗ |
| Mip-NeRF 360 | Mip-Splatting | 27.54 | 0.817 | 0.216 | Partial |
| Mip-NeRF 360 | **AAA-Gaussians** | **27.84** | **0.836** | **0.188** | **✓** |
| Deep Blending | **AAA-Gaussians** | **30.49** | **0.913** | **0.222** | **✓** |

### Ablation Study (Out-of-Distribution Evaluation — Large FOV 3×)

| Method | Mip-NeRF 360 PSNR | T&T PSNR | Deep Blending PSNR |
|------|-------------------|----------|-------------------|
| 3DGS | 26.82 | 17.11 | 26.19 |
| MCMC | 23.35 | 14.37 | 18.32 |
| Mip-Splatting | 26.05 | 17.31 | 25.59 |
| StopThePop | 27.04 | 20.24 | 27.55 |
| **AAA-Gaussians** | **27.84** | **23.58** | **30.49** |

### Key Findings
- Under out-of-distribution viewpoints (large FOV 3×), MCMC and Taming 3DGS suffer severe quality collapse (T&T drops from 24.6 to 14.4), whereas the proposed method is entirely unaffected.
- In multi-resolution evaluation, half-resolution PSNR on the Bonsai scene improves from 28.98 (MCMC) to 32.12; significant advantages are also observed at double resolution.
- 3D frustum culling has no impact on quality but markedly improves performance (removing culling increases render time from 7.72 ms to 14.40 ms).
- The proposed method is only marginally slower than MCMC (7.72 ms vs. 6.79 ms) and is even faster on certain scenes.
- 3D evaluation achieves speed comparable to or faster than the 2D splat approximation (7.03 ms vs. 7.52 ms).

## Highlights & Insights
- **Unified Framework**: To the best of the authors' knowledge, this is the only rasterization method that simultaneously eliminates all three artifact categories—aliasing, distortion, and popping—within a single framework.
- **Perpendicular Scaling Factor Derivation**: Restricting amplitude adjustment to area changes perpendicular to the viewing direction is a subtle but critical insight that prevents 3D anti-aliasing from causing Gaussians to become excessively transparent.
- **View-Space Angular Bounding**: The angular parameterization correctly handles Gaussians penetrating the near plane, a key requirement for practical VR rendering.
- **Engineering Quality**: The method is open-sourced, achieves over 100 FPS on consumer-grade hardware in practice, and demonstrates strong potential for real-world deployment.
- **Honest Evaluation**: The authors candidly acknowledge that standard in-distribution metrics show limited improvement, as stronger view consistency prevents the model from "cheating" through overfitting.

## Limitations & Future Work
- Improvements on standard in-distribution metrics are modest, as strong view consistency constrains the available overfitting headroom.
- The view-space bounding approach still assumes a pinhole camera model and has limited adaptability to other camera models such as fisheye lenses.
- Stronger view consistency implies a need for more expressive view-dependent encoding to compensate.
- The current method relies on MCMC densification; integration with alternative densification strategies remains to be explored.

## Related Work & Insights
- **vs. Mip-Splatting**: Mip-Splatting's 2D Mip filter is incompatible with 3D evaluation; the proposed 3D adaptive filter serves as a complete replacement for the 2D scheme.
- **vs. StopThePop**: The proposed method inherits its hierarchical sorting for popping suppression but replaces 2D tile culling with 3D frustum culling.
- **vs. Hybrid Transparency**: The proposed method inherits screen-space planar 3D evaluation but corrects its bounding computation, which was responsible for popping and aliasing artifacts.
- **vs. Ray Tracing Methods**: While ray tracing is inherently 3D, it is computationally expensive and requires acceleration structures; this work demonstrates that rasterization can achieve equivalent artifact-free quality.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The perpendicular scaling factor for 3D anti-aliasing and view-space bounding computation exhibit genuine technical novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers in- and out-of-distribution evaluation, multi-resolution, large FOV, performance timing, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Technical derivations are clear and problem analysis is thorough.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical; represents a critical step toward the engineering deployment of 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [\[NeurIPS 2025\] Anti-Aliased 2D Gaussian Splatting](../../NeurIPS2025/3d_vision/anti-aliased_2d_gaussian_splatting.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)

</div>

<!-- RELATED:END -->
