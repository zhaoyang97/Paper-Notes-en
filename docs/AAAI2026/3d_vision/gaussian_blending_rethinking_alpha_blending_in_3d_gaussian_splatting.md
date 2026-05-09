---
title: >-
  [Paper Note] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This paper revisits scalar alpha blending in 3DGS and identifies its neglect of intra-pixel spatial variation as the root cause of multi-scale rendering artifacts (enlargement erosion / downscaling dilation). The proposed Gaussian Blending models alpha and transmittance as spatial distributions within a pixel (2D uniform window), achieving real-time anti-aliasing without retraining. PSNR on multi-scale Blender improves from 31.59 to 35.80.
tags:
  - AAAI 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - Alpha Blending
  - Anti-aliasing
  - Multi-scale Rendering
  - Intra-pixel Spatial Distribution
date: 2026-05-08
content_hash: 4624bb2682c65a5b
---

# Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting

**Conference**: AAAI 2026  
**arXiv**: [2511.15102](https://arxiv.org/abs/2511.15102)  
**Code**: To be released  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Alpha Blending, Anti-aliasing, Multi-scale Rendering, Intra-pixel Spatial Distribution

## TL;DR
This paper revisits scalar alpha blending in 3DGS and identifies its neglect of intra-pixel spatial variation as the root cause of multi-scale rendering artifacts (enlargement erosion / downscaling dilation). The proposed Gaussian Blending models alpha and transmittance as spatial distributions within a pixel (2D uniform window), achieving real-time anti-aliasing without retraining. PSNR on multi-scale Blender improves from 31.59 to 35.80.

## Background & Motivation
**Background**: 3DGS explicitly represents 3D scenes via Gaussian splats, achieving rendering speeds orders of magnitude faster than NeRF, and has become the dominant approach for Novel View Synthesis (NVS). Methods such as Mip-Splatting and Analytic-Splatting improve multi-scale anti-aliasing through prefiltering.

**Limitations of Prior Work**: All existing NVS methods still exhibit pronounced artifacts at sampling rates unseen during training — enlargement produces edge erosion (blurring) while downscaling produces dilation (staircase artifacts). These issues persist even in Analytic-Splatting, which performs analytic integration.

**Key Challenge**: All methods employ scalar alpha blending — computing alpha and transmittance as scalars (one value per pixel). This causes foreground splats to fully occlude background splats that should not be occluded, because intra-pixel spatial occlusion relationships are ignored. This error is amplified when the sampling rate changes.

**Goal**: Incorporate intra-pixel spatial variation into the alpha blending process without sacrificing real-time performance, thereby eliminating erosion and dilation artifacts.

**Key Insight**: Gaussian splats form continuous surfaces in 2D screen space, and their combined transmittance can be approximated by a simple 2D uniform distribution. By dynamically tracking the window extent of this distribution, spatial occlusion can be modeled efficiently.

**Core Idea**: Replace scalar alpha blending with spatially distributed alpha blending — transmittance is no longer a scalar but a spatial window within the pixel.

## Method

### Overall Architecture
Gaussian Blending replaces the rendering kernel in the original 3DGS pipeline:
- **Input**: Same Gaussian splat scene representation as 3DGS
- **Modification**: In the alpha blending stage, the scalar transmittance $T_i$ is replaced by a 2D uniform distribution representation (center $x_i$, size $l_i$, value $t_i$)
- **Output**: More accurate pixel colors, especially at sampling rates unseen during training

### Key Designs

1. **Spatial Transmittance Distribution**:

    - Function: Tracks the spatial distribution of transmittance within a pixel using a 2D uniform window.
    - Mechanism: In conventional methods, $T_i = \prod_{j=1}^{i-1}(1-\alpha_j(p))$ is a scalar. Gaussian Blending represents it as a window $(x_i, l_i, t_i)$, initialized to the full pixel region ($x_1=p, l_1=[1,1]^\top, t_1=1$). After each splat is rendered, the window shrinks according to the splat's spatial coverage — transmittance in occluded regions decreases while unoccluded regions retain high transmittance.
    - Design Motivation: Physically correct rendering requires integrating over the pixel area $C_p^p = \int_p \sum_i T_i^p(x)\alpha_i(x)c_i dx$, but direct computation is exponentially complex. Since Gaussian splats tend to cluster into continuous surfaces, the combined transmittance approximates a uniform distribution, making the window approximation sufficient.

2. **Weight Computation (Splat Response Integration)**:

    - Function: Computes the integrated response of the current splat within the transmittance window.
    - Mechanism: The 2D Gaussian is decomposed via eigenvalue decomposition to identify principal axes; the window is rotated to align with these axes, reducing the problem to two independent 1D Gaussian integrals: $\int w_i(x)dx = t_i \cdot o_i \cdot I^0_{\sigma_1}(u_1,u_2) \cdot I^0_{\sigma_2}(v_1,v_2)$, where $I^k_\sigma(a,b)$ denotes the $k$-th order moment of a 1D Gaussian.
    - Design Motivation: Direct 2D integration has no closed form; eigenvalue decomposition combined with rotation alignment decomposes the problem into analytically tractable 1D integrals.

3. **Window Update (Transmittance Distribution Evolution)**:

    - Function: Updates the spatial window after each splat is rendered.
    - Mechanism: First- and second-order moments are used to match the updated transmittance distribution. The new window center and size are computed via moment matching, ensuring accurate tracking of the remaining transmittance's spatial distribution. The window progressively shrinks toward regions that have not yet been occluded.
    - Design Motivation: Regions with high transmittance should remain visible to background splats, while regions with low transmittance should suppress redundant rendering contributions.

### Loss & Training
- **Training-free**: Gaussian Blending is a pure rendering method requiring no additional training.
- **Drop-in replacement**: Can directly replace the rendering kernel of existing 3DGS methods.
- Real-time rendering speed is maintained through optimized CUDA implementation with no additional memory overhead.
- A variant $\text{GB}_\text{test}$ is also provided, applying Gaussian Blending only at test time.

## Key Experimental Results

### Main Results
PSNR on the Multi-scale Blender dataset (trained at ×1, tested at ×1/2–×1/8):

| Method | ×1 | ×1/2 | ×1/4 | ×1/8 | Avg. |
|------|-----|------|------|------|------|
| 3DGS | 33.57 | 27.04 | 21.43 | 17.74 | 24.95 |
| Mip-Splatting | 33.54 | 34.09 | 31.50 | 27.80 | 31.73 |
| Analytic-Splatting | 33.78 | 34.20 | 31.16 | 27.22 | 31.59 |
| **Gaussian Blending** | **33.92** | **35.80** | **36.82** | **35.79** | **35.58** |
| **Analytic+GB_test** | 33.62 | 35.72 | 37.36 | 36.51 | **35.80** |

At ×1/8 downscaling: 3DGS's 17.74 → Gaussian Blending's 35.79, a gain of **18 dB**!

### Ablation Study

| Configuration | Performance | Notes |
|------|------|------|
| Scalar alpha blending | Baseline | Erosion + dilation artifacts |
| Structured pruning (without window) | Marginal improvement | No spatial distribution tracking |
| Gaussian Blending (full) | Best | Dynamic window tracking |
| GB applied at test time only | Close to full | Effective without retraining |

### Key Findings
- **Scalar alpha blending is the root cause of multi-scale artifacts** — insufficient prefiltering is not the primary issue; the blending mechanism itself is flawed. Even with perfect pixel-area integration in Analytic-Splatting, scalar transmittance still induces boundary artifacts.
- **Substantial gains at unseen scales (×1/4: +5.7 dB, ×1/8: +8.6 dB vs. Analytic-Splatting)**, with marginal improvement even at the training scale (×1).
- **Orthogonal and complementary to existing methods**: can be stacked on top of Mip-Splatting or Analytic-Splatting for further gains.
- **Real-time rendering speed is maintained** with no additional memory overhead.

## Highlights & Insights
- **Precise problem diagnosis**: Rather than proposing a new anti-aliasing filter, this work identifies that the blending mechanism itself is fundamentally incorrect, revealing a limitation shared by all NVS methods.
- The **uniform distribution approximation of transmittance** is coarse yet effective — Gaussian splats do tend to cluster into continuous surfaces, and their combined alpha distribution approaches uniformity. This observation is highly insightful.
- The **drop-in replacement design** is highly practical: multi-scale anti-aliasing can be obtained on any 3DGS method by simply replacing the rendering kernel, without retraining.

## Limitations & Future Work
- **The uniform distribution approximation may be inaccurate for translucent objects**: the alpha distribution in scenes with smoke, glass, or similar materials is far from uniform.
- **The simplification of the window as axis-aligned (rotation ≤45°)** may introduce errors in certain extreme splat configurations.
- **Evaluation is limited to Blender and Mip-NeRF 360**: validation on larger-scale real-world scenes (e.g., city-scale reconstruction) is absent.
- **No direct comparison with supersampling** to validate the accuracy of the approximation.

## Related Work & Insights
- **vs. Mip-Splatting**: Applies 3D+2D prefiltering to address frequency aliasing, but alpha blending remains scalar. Gaussian Blending targets the blending mechanism itself; the two approaches are complementary (combined gain: +3.8 dB avg).
- **vs. Analytic-Splatting**: Replaces point sampling with analytic integration over the pixel area, but transmittance remains scalar. Spatial transmittance from GB can be stacked on top.
- **vs. Supersampling**: Physically correct but computationally expensive. GB approximates the effect of supersampling at negligible cost via the uniform distribution assumption.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Approaches the problem from the perspective that "alpha blending itself is incorrect," representing a fundamental rethinking of the rendering mechanism rather than an incremental improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-scale Blender and Mip-NeRF 360, comparisons against multiple baselines, and stacking tests with various methods.
- Writing Quality: ⭐⭐⭐⭐⭐ — Figure 2 intuitively illustrates the essential difference between scalar blending and Gaussian Blending.
- Value: ⭐⭐⭐⭐⭐ — Drop-in replacement + real-time + no retraining = extremely high practical value; a strong candidate to become the new standard for 3DGS rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](../../CVPR2026/3d_vision/rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[AAAI 2026\] Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning](rethinking_rainy_3d_scene_reconstruction_via_perspective_transforming_and_bright.md)
- [\[AAAI 2026\] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](mobgs_motion_deblurring_dynamic_3d_gaussian_splatting_for_blurry_monocular_video.md)

</div>

<!-- RELATED:END -->
