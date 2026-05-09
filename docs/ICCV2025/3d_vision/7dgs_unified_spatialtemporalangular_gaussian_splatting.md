---
title: >-
  [Paper Note] 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting
description: >-
  [ICCV 2025][3D Vision][7D Gaussian] This paper proposes 7DGS, which models scene elements as **7-dimensional Gaussian distributions** (3D spatial + 1D temporal + 3D view direction). A conditional slicing mechanism converts 7D Gaussians into time- and view-conditioned 3D Gaussians, unifying dynamic scene rendering with view-dependent appearance. On the proposed 7DGS-PBR dataset, 7DGS achieves up to 7.36 dB PSNR gain over 4DGS while using only 15.3% of the Gaussian primitives, with real-time rendering at 401 FPS.
tags:
  - ICCV 2025
  - 3D Vision
  - 7D Gaussian
  - Dynamic Scene
  - View-Dependent
  - Conditional Slicing
  - Real-Time Rendering
date: 2026-05-08
content_hash: 115ecc1313295e4d
---

# 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2503.07946](https://arxiv.org/abs/2503.07946)
**Code**: [https://gaozhongpai.github.io/7dgs/](https://gaozhongpai.github.io/7dgs/)
**Area**: 3D Vision / Dynamic Scene Rendering / 3D Gaussian Splatting
**Keywords**: 7D Gaussian, Dynamic Scene, View-Dependent, Conditional Slicing, Real-Time Rendering

## TL;DR
This paper proposes 7DGS, which models scene elements as **7-dimensional Gaussian distributions** (3D spatial + 1D temporal + 3D view direction). A conditional slicing mechanism converts 7D Gaussians into time- and view-conditioned 3D Gaussians, unifying dynamic scene rendering with view-dependent appearance. On the proposed 7DGS-PBR dataset, 7DGS achieves up to 7.36 dB PSNR gain over 4DGS while using only 15.3% of the Gaussian primitives, with real-time rendering at 401 FPS.

## Background & Motivation
High-quality real-time rendering of dynamic scenes requires simultaneous modeling along three dimensions: (1) spatial geometry, (2) temporal dynamics, and (3) view-dependent appearance. Existing methods address only subsets of these: 4DGS handles dynamics (space + time) but ignores view-dependent effects; 6DGS handles view dependence (space + direction) but is limited to static scenes. In the real world, all three dimensions are mutually coupled — for instance, specular highlights on a moving object vary simultaneously with object position and viewing direction. No prior method addresses all three in a unified framework.

## Core Problem
How to simultaneously model spatial, temporal, and view-directional dependencies within a unified framework, enabling real-time rendering of dynamic scenes with view-dependent appearance?

## Method

### Overall Architecture
Each scene element is represented as a 7D Gaussian $\mathcal{N}(\mu, \Sigma)$, where $\mu = [\mu_p, \mu_t, \mu_d]$ (3D position + 1D time + 3D direction) and $\Sigma$ is a $7 \times 7$ covariance matrix. At render time, given the current time $t$ and viewing direction $d$, conditional slicing produces a 3D Gaussian, which is then passed to the standard 3DGS rasterization pipeline.

### Key Designs
1. **7D Gaussian Representation**: The $7 \times 7$ covariance matrix is parameterized via Cholesky decomposition $\Sigma = LL^T$ to ensure positive definiteness. Cross-covariance blocks $\Sigma_{pt}$, $\Sigma_{pd}$, and $\Sigma_{td}$ encode spatial-temporal-directional coupling — a critical design that enables a single Gaussian to simultaneously capture motion-induced positional changes and view-dependent appearance variations.

2. **Conditional Slicing Mechanism**: For a given $(t, d)$, the conditional distribution formula for multivariate Gaussians is applied to slice the 7D Gaussian into a conditioned 3D Gaussian:
$$\mu_{cond} = \mu_p + \Sigma_{p,(t,d)} \Sigma_{(t,d)}^{-1} \begin{pmatrix} t - \mu_t \\ d - \mu_d \end{pmatrix}$$
$$\Sigma_{cond} = \Sigma_p - \Sigma_{p,(t,d)} \Sigma_{(t,d)}^{-1} \Sigma_{p,(t,d)}^T$$
The conditional opacity is modulated by temporal and directional attenuation factors: $\alpha_{cond} = \alpha \cdot f_{temp} \cdot f_{dir}$.

3. **Adaptive Gaussian Refinement (AGR)**: Conditional slicing adjusts position and opacity but keeps the covariance shape static. AGR employs a lightweight MLP (2 layers × 64 units) to predict residual corrections for key parameters (position, time, direction, covariance), dynamically adjusting Gaussian shape based on temporal encoding $\gamma(t)$ to capture non-rigid deformations.

4. **Compatibility with 3DGS Pipeline**: The sliced conditional 3D Gaussians are directly fed into the standard 3DGS projection and rasterization pipeline, leveraging existing adaptive density control and efficient rendering, with only a minimal opacity threshold ($\tau_{min}=0.01$) added.

### Loss & Training
- Same loss function as 3DGS (L1 + SSIM)
- Temporal densification: splitting triggered when spatial-temporal covariance $\Sigma_{pt}$ magnitude $> 0.05$ and temporal scale $> 0.25$
- Single V100 (16 GB), Adam optimizer
- AGR training begins after 3,000 steps; $\lambda_t$ and $\lambda_d$ become trainable after 15,000 steps

## Key Experimental Results

### 7DGS-PBR Dataset (Dynamic + View-Dependent)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | #Points | FPS |
|--------|-------|-------|--------|---------|-----|
| 4DGS | 27.79 | 0.934 | 0.079 | 641K | 193 |
| **7DGS** | **32.50** | **0.958** | **0.051** | **98K** | 175 |
| 7DGS (w/o AGR) | 31.77 | 0.955 | 0.055 | 88K | **376** |

+4.71 dB PSNR with only 15.3% of the Gaussian primitives. The *heart1* scene achieves +8.18 dB.

### D-NeRF Dataset (Synthetic Dynamic)

| Method | PSNR↑ |
|--------|-------|
| 4DGaussians | 33.30 |
| 4DGS | 33.21 |
| **7DGS** | **34.34** |

### Technicolor Dataset (Real Multi-View)

| Method | PSNR↑ | SSIM↑ |
|--------|-------|-------|
| Ex4DGS | 33.49 | **0.917** |
| STG | 33.23 | 0.912 |
| **7DGS** | **33.58** | 0.912 |

### Ablation Study: w/o AGR vs. w/ AGR
- 7DGS-PBR: 31.77 vs. 32.50 (+0.73 dB), with FPS dropping from 376 to 175
- D-NeRF: 33.26 vs. 34.34 (+1.08 dB)
- AGR contributes substantially to quality at the cost of rendering speed

## Highlights & Insights
- **Elegant unification**: The 7D Gaussian naturally encodes spatial-temporal-directional cross-covariances, yielding a mathematically complete and intuitively grounded representation
- **Elegance of conditional slicing**: Reduces the high-dimensional problem to a standard 3D problem, remaining fully compatible with existing pipelines
- **Remarkable parameter efficiency**: Surpassing 4DGS by 4.7 dB with only 15% of the Gaussian count — a testament to the compressive power of high-dimensional representations
- **View-dependent effect breakthrough**: On scenes such as hearts, clouds, and flames in 7DGS-PBR, view-dependent effects that 4DGS entirely fails to handle are effectively captured by 7DGS
- **Flexible degraded mode**: Removing AGR yields 400+ FPS, suitable for speed-critical applications

## Limitations & Future Work
- The AGR MLP introduces additional computational overhead (FPS drops from 376 to 175)
- Advantages on real-world data such as Technicolor are relatively modest
- The 7D covariance has 28 independent parameters (Cholesky lower triangle), making training less stable than 3DGS
- Explicit motion modeling strategies such as keyframe interpolation have not been incorporated

## Related Work & Insights
- **4DGS**: 4D only (space + time); cannot handle view-dependent effects. 7DGS outperforms it across all datasets
- **6DGS**: 6D (space + direction) but static only. 7DGS adds the temporal dimension to unify dynamics and view dependence
- **Ex4DGS**: Models motion explicitly via keyframe interpolation. Results on Technicolor are comparable, suggesting complementary strengths
- **SSS / 3D-HGS**: Improve kernel functions. 7DGS pursues dimensional extension — an orthogonal direction that could be combined with these approaches

The conceptual progression 3DGS (3D) → 4DGS (4D) → 6DGS (6D) → 7DGS (7D) exemplifies an elegant "unification through dimensionality" paradigm. Conditional slicing is grounded in elementary multivariate Gaussian conditioning, yet proves remarkably powerful. This framework paves the way for even higher-dimensional representations — e.g., 8D incorporating wavelength for spectral rendering.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The 7D unified representation combined with conditional slicing is elegant and natural, filling the gap between 4DGS and 6DGS
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets, SOTA comparisons, and ablations; additional real-world validation would strengthen the work
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous and complete; the logical progression from 3DGS to 6DGS to 7DGS is clearly articulated
- Value: ⭐⭐⭐⭐⭐ — Unifies dynamic and view-dependent rendering in a single framework; the 15% point count with +5 dB gain demonstrates high practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Spatial-Temporal Aware Visuomotor Diffusion Policy Learning](spatial-temporal_aware_visuomotor_diffusion_policy_learning.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [\[ICCV 2025\] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)
- [\[ICCV 2025\] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](aaa-gaussians_anti-aliased_and_artifact-free_3d_gaussian_rendering.md)
- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)

</div>

<!-- RELATED:END -->
