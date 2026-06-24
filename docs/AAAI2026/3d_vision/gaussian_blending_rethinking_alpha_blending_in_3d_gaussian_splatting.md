---
title: >-
  [Paper Note] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] Revisits scalar alpha blending in 3DGS, pointing out that ignoring intra-pixel spatial variation is the root cause of multi-scale rendering artifacts (erosion when zoomed in and dilation when zoomed out). It proposes Gaussian Blending, which models alpha and transmittance as an intra-pixel spatial distribution (a 2D uniform window) to achieve real-time anti-aliasing without retraining, improving the PSNR from 31.59 to 35.80 on Mul…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Alpha Blending"
  - "Anti-aliasing"
  - "Multi-scale Rendering"
  - "Intra-pixel Spatial Distribution"
date: 2026-05-08
content_hash: 14d8e88d4cbc5531
---

# Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting

**Conference**: AAAI 2026  
**arXiv**: [2511.15102](https://arxiv.org/abs/2511.15102)  
**Code**: To be released  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Alpha Blending, Anti-aliasing, Multi-scale Rendering, Intra-pixel Spatial Distribution

## TL;DR
Revisits scalar alpha blending in 3DGS, pointing out that ignoring intra-pixel spatial variation is the root cause of multi-scale rendering artifacts (erosion when zoomed in and dilation when zoomed out). It proposes Gaussian Blending, which models alpha and transmittance as an intra-pixel spatial distribution (a 2D uniform window) to achieve real-time anti-aliasing without retraining, improving the PSNR from 31.59 to 35.80 on Multi-scale Blender.

## Background & Motivation
**Background**: 3DGS explicitly represents 3D scenes via Gaussian splats, achieving rendering speeds several orders of magnitude faster than NeRF, and has become the mainstream method for Novel View Synthesis (NVS). Methods like Mip-Splatting and Analytic-Splatting have improved multi-scale anti-aliasing through pre-filtering.

**Limitations of Prior Work**: All existing NVS methods still exhibit significant artifacts at unseen sampling rates during training—showing edge erosion (blurring) when zoomed in and dilation (staircase artifacts) when zoomed out. Even though Analytic-Splatting performs analytical integration, the problem persists.

**Key Challenge**: All methods use scalar alpha blending, calculating alpha and transmittance as scalars (a single value per pixel). This causes foreground splats to fully obscure background splats that should not be fully occluded, because intra-pixel spatial occlusion relations are ignored. This error is amplified when the sampling rate changes.

**Goal**: Incorporate intra-pixel spatial variations into the alpha blending process to eliminate erosion and dilation artifacts without sacrificing real-time performance.

**Key Insight**: It is observed that Gaussian splats form a continuous surface in 2D screen space, and their merged transmittance can be approximated by a simple 2D uniform distribution. By dynamically tracking the window range of this distribution, spatial occlusion can be modeled efficiently.

**Core Idea**: Replace scalar alpha blending with spatial distribution alpha blending, where transmittance is no longer a single number but a spatial window within the pixel.

## Method

### Overall Architecture
Gaussian Blending replaces the rendering kernel in the original 3DGS pipeline:
- **Input**: The same Gaussian splat scene representation as in 3DGS.
- **Improvement**: In the alpha blending stage, the scalar transmittance $T_i$ is replaced with a 2D uniform distribution representation (center $x_i$, size $l_i$, value $t_i$).
- **Output**: More accurate pixel colors, especially under sampling rates unseen during training.

### Key Designs

1. **Spatial Transmittance Distribution**:

    - **Function**: Tracks the intra-pixel spatial distribution of transmittance using a 2D uniform window.
    - **Mechanism**: In traditional methods, $T_i = \prod_{j=1}^{i-1}(1-\alpha_j(p))$ is a scalar. Gaussian Blending represents it as a window $(x_i, l_i, t_i)$, initialized as the entire pixel area ($x_1=p, l_1=[1,1]^\top, t_1=1$). After rendering each splat, the window shrinks according to the spatial coverage of the splat—transmittance decreases in occluded areas and remains high in unoccluded areas.
    - **Design Motivation**: Physically correct rendering requires integrating over the pixel area $C_p^p = \int_p \sum_i T_i^p(x)\alpha_i(x)c_i dx$, but direct computation has exponential complexity. Observing that Gaussian splats aggregate to form continuous surfaces, the merged transmittance approximates a uniform distribution, which can therefore be approximated using a window.

2. **Weight Calculation (Splat Response Integration)**:

    - **Function**: Calculates the integrated response of the current splat within the transmittance window.
    - **Mechanism**: Performs eigenvalue decomposition on the 2D Gaussian to find the principal axes, rotates the window to align with these axes, and then factorizes it into two independent 1D Gaussian integrations: $\int w_i(x)dx = t_i \cdot o_i \cdot I^0_{\sigma_1}(u_1,u_2) \cdot I^0_{\sigma_2}(v_1,v_2)$, where $I^k_\sigma(a,b)$ represents the $k$-th order moment of the 1D Gaussian.
    - **Design Motivation**: Direct 2D integration has no closed-form solution; using eigenvalue decomposition and rotation alignment allows it to be factorized into analytically computable 1D integrations.

3. **Window Update (Transmittance Distribution Evolution)**:

    - **Function**: Updates the spatial window after rendering each splat.
    - **Mechanism**: Employs the first and second moments to match the updated transmittance distribution. The center and size of the new window are calculated through moment matching, ensuring that the spatial distribution of the remaining transmittance is accurately tracked. The window gradually shrinks to areas that have not yet been occluded.
    - **Design Motivation**: High transmittance regions should retain visibility for background splats, while low transmittance regions should suppress redundant rendering.

### Loss & Training
- **Training-free**: Gaussian Blending is a pure rendering method and does not require additional training.
- **Drop-in replacement**: It can directly replace the rendering kernels of existing 3DGS methods.
- The real-time rendering speed is preserved through optimized CUDA implementation with no extra memory overhead.
- It also provides the $\text{GB}_\text{test}$ variant, which applies Gaussian Blending only during testing.

## Key Experimental Results

### Main Results
PSNR results on the Multi-scale Blender dataset (trained at $\times 1$, tested at $\times 1/2$ to $\times 1/8$):

| Method | ×1 | ×1/2 | ×1/4 | ×1/8 | Avg. |
|------|-----|------|------|------|------|
| 3DGS | 33.57 | 27.04 | 21.43 | 17.74 | 24.95 |
| Mip-Splatting | 33.54 | 34.09 | 31.50 | 27.80 | 31.73 |
| Analytic-Splatting | 33.78 | 34.20 | 31.16 | 27.22 | 31.59 |
| **Gaussian Blending** | **33.92** | **35.80** | **36.82** | **35.79** | **35.58** |
| **Analytic+GB_test** | 33.62 | 35.72 | 37.36 | 36.51 | **35.80** |

At $\times 1/8$ downscaling: 3DGS improves from 17.74 to 35.79 with Gaussian Blending, yielding a **Gain** of **18dB**!

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| Scalar alpha blending | Baseline | erosion+dilation artifacts |
| Structured pruning (without window) | Slight improvement | Does not track spatial distribution |
| Gaussian Blending (full) | Optimal | Dynamic window tracking |
| GB only during testing | Close to full | Effective without retraining |

### Key Findings
- **Scalar alpha blending is the root cause of multi-scale artifacts**—the issue is not insufficient pre-filtering, but rather the blending process itself. Even though Analytic-Splatting performs perfect pixel integration, scalar transmittance still leads to edge artifacts.
- **Significant gains on unseen scales ($\times 1/4$: +5.7dB, $\times 1/8$: +8.6dB vs. Analytic-Splatting)**, while also yielding slight improvements on the training scale ($\times 1$).
- **Orthogonal and complementary to existing methods**: It can be combined with Mip-Splatting or Analytic-Splatting for further improvements.
- **Preserves real-time rendering speed** with no additional memory overhead.

## Highlights & Insights
- **Precise Problem Diagnosis**: Instead of proposing a new anti-aliasing filter, this work tackles the problem from the perspective of "blending itself is flawed," identifying a fundamental limitation shared by all NVS methods.
- **Effectiveness of the Uniform Distribution Approximation**: Although the assumption of approximating transmittance with a uniform distribution is coarse, it is effective because Gaussian splats tend to cluster and form continuous surfaces, approaching a uniform alpha distribution. This observation is highly insightful.
- **Practical Drop-in Replacement Design**: It is highly practical as it requires no model retraining; simply replacing the rendering kernel enables multi-scale anti-aliasing in any 3DGS method.

## Limitations & Future Work
- **The uniform distribution approximation may be inaccurate for semi-transparent objects**: The alpha distribution in scenes containing smoke, glass, etc., is far from being uniform.
- **The simplification of the window as a square (axis-aligned rotation $\le 45^\circ$)** may introduce errors under certain extreme splat distributions.
- **Evaluation is limited to Blender and Mip-NeRF 360**: Validation on larger-scale real-world scenes (such as city-scale reconstruction) is missing.
- **Lacks a direct comparison with supersampling** to validate the accuracy of the approximation.

## Related Work & Insights
- **vs Mip-Splatting**: Mip-Splatting performs 3D+2D pre-filtering to handle frequency aliasing, but its alpha blending remains scalar. GB addresses the blending mechanism itself, making the two complementary (yielding an average of +3.8dB when combined).
- **vs Analytic-Splatting**: Analytic-Splatting performs analytical integration over the pixel area instead of point sampling, but the transmittance is still scalar. GB's spatial transmittance can be layered on top of it.
- **vs Supersampling**: Supersampling is physically correct but computationally expensive. GB uses a uniform approximation to achieve a performance close to supersampling at a negligible cost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Tackling the problem from the perspective of "alpha blending itself is flawed" represents a fundamental rethinking of the underlying rendering mechanism, rather than an incremental improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on Multi-scale Blender and Mip-NeRF 360, compared with multiple baselines, and tested in combination with various methods.
- Writing Quality: ⭐⭐⭐⭐⭐ The comparison in Figure 2 extremely intuitively demonstrates the fundamental difference between scalar blending and Gaussian Blending.
- Value: ⭐⭐⭐⭐⭐ Drop-in replacement + real-time + training-free = extremely high practical value; it has the potential to become a new standard in 3DGS rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](../../CVPR2026/3d_vision/rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[CVPR 2026\] MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification](../../CVPR2026/3d_vision/morel_long-range_flicker-free_4d_motion.md)
- [\[AAAI 2026\] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)

</div>

<!-- RELATED:END -->
