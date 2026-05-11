---
title: >-
  [Paper Note] 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting
description: >-
  [ICCV 2025][3D Vision][Gaussian Splatting] This work extends 3DGS to seven dimensions (spatial 3D + temporal 1D + directional 3D). A conditional slicing mechanism projects 7D Gaussians into 3D Gaussians compatible with the standard 3DGS pipeline, achieving up to 7.36 dB PSNR improvement on dynamic scenes with view-dependent effects while maintaining 401 FPS real-time rendering.
tags:
  - ICCV 2025
  - 3D Vision
  - Gaussian Splatting
  - Dynamic Scene Rendering
  - View-Dependent Effects
  - Real-Time Rendering
  - Novel View Synthesis
date: 2026-05-08
content_hash: 445d3dd6112d4ef5
---

# 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2503.07946](https://arxiv.org/abs/2503.07946)
**Code**: [https://gaozhongpai.github.io/7dgs/](https://gaozhongpai.github.io/7dgs/) (project page)
**Area**: 3D Vision
**Keywords**: Gaussian Splatting, Dynamic Scene Rendering, View-Dependent Effects, Real-Time Rendering, Novel View Synthesis

## TL;DR
This work extends 3DGS to seven dimensions (spatial 3D + temporal 1D + directional 3D). A conditional slicing mechanism projects 7D Gaussians into 3D Gaussians compatible with the standard 3DGS pipeline, achieving up to 7.36 dB PSNR improvement on dynamic scenes with view-dependent effects while maintaining 401 FPS real-time rendering.

## Background & Motivation
Photorealistic rendering of dynamic scenes requires simultaneous modeling of three dimensions: **spatial geometry**, **temporal dynamics**, and **view-dependent appearance**. These three dimensions exhibit complex interdependencies — for instance, specular highlights on moving objects vary simultaneously with both viewing direction and position.

Existing methods address individual sub-problems:
- **4DGS** (spatial + temporal): handles dynamic scenes but ignores view-dependent effects
- **6DGS** (spatial + directional): captures view dependence but is limited to static scenes
- No prior method handles all three dimensions within a **unified framework** while maintaining real-time performance

**Key Challenge**: High-dimensional representations offer stronger modeling capacity but introduce substantial computational overhead. Operating directly in 7D space precludes real-time rendering.

**Core Idea**: Scene elements are represented as 7D Gaussian distributions. Leveraging the conditional distribution property of multivariate Gaussians, a mathematically rigorous conditional slicing mechanism "slices" each 7D Gaussian into a 3D Gaussian conditioned on time and viewing direction, which seamlessly integrates into the existing 3DGS rendering pipeline.

## Method

### Overall Architecture
In 7DGS, each scene element is modeled as a 7D Gaussian $X = (X_p, X_t, X_d) \sim \mathcal{N}(\mu, \Sigma)$, where $X_p \in \mathbb{R}^3$ denotes spatial coordinates, $X_t \in \mathbb{R}$ denotes time, and $X_d \in \mathbb{R}^3$ denotes viewing direction. The cross-block terms $\Sigma_{pt}, \Sigma_{pd}, \Sigma_{td}$ of the $7\times7$ covariance matrix encode inter-dimensional correlations. At render time, the 7D Gaussian is reduced to 3D via conditional slicing and fed into the standard 3DGS rasterization pipeline.

### Key Designs

1. **7D Gaussian Representation**: The full 7D covariance matrix of each scene element is parameterized via Cholesky decomposition $\Sigma = LL^\top$ to guarantee positive definiteness. The cross-covariance blocks $\Sigma_{pt}$ (spatial–temporal), $\Sigma_{pd}$ (spatial–directional), and $\Sigma_{td}$ (temporal–directional) naturally model the coupling among all three dimensions.

    - **Design Motivation**: Moving specular highlights must respond simultaneously to changes in position, time, and direction — a coupling that cannot be captured by modeling each dimension independently.

2. **Conditional Slicing Mechanism**: Given observation time $t$ and viewing direction $d$, the conditional distribution of the spatial component is derived using the multivariate Gaussian conditioning formula:
    $$\mu_{cond} = \mu_p + \Sigma_{p,(t,d)} \Sigma_{(t,d)}^{-1} \begin{pmatrix} t - \mu_t \\ d - \mu_d \end{pmatrix}$$
    $$\Sigma_{cond} = \Sigma_p - \Sigma_{p,(t,d)} \Sigma_{(t,d)}^{-1} \Sigma_{p,(t,d)}^\top$$
   Opacity is simultaneously modulated by temporal and directional factors: $\alpha_{cond} = \alpha \cdot f_{temp} \cdot f_{dir}$.
    - **Design Motivation**: This operation is mathematically exact with no approximation error, and the resulting 3D Gaussians can directly reuse the efficient 3DGS rasterizer.

3. **Adaptive Gaussian Refinement (AGR)**: A lightweight MLP ($C_{in} \times 64 \times C_{out}$) predicts residual corrections $\Delta\mu_p, \Delta\mu_t, \Delta\mu_d, \Delta l$ from a feature vector $f = \mu_p \oplus \mu_t \oplus \mu_d \oplus \gamma(t)$, dynamically adjusting Gaussian parameters prior to conditional slicing to model complex motions such as non-rigid deformations.

    - **Design Motivation**: Conditional slicing preserves Gaussian shape invariant over time. AGR compensates for this limitation, enabling the same Gaussian to exhibit different spatial shapes at different time steps.

### Loss & Training
- The same loss function (L1 + SSIM), optimizer, and hyperparameters as 3DGS are adopted
- The minimum opacity threshold is modified to $\tau_{min}=0.01$ to compensate for conditional opacity modulation
- The Gaussian splitting strategy adds a temporal criterion: splitting is triggered when $\|\Sigma_{pt}\|$ exceeds a threshold and the temporal scale $\Sigma_t$ is large
- The AGR network begins training after 3,000 iterations; modulation parameters $\lambda_t, \lambda_d$ become learnable after 15,000 iterations

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | # Points↓ |
|--------|------|-------|-------|--------|------|-------|
| 7DGS-PBR (avg) | 4DGS | 27.79 | 0.934 | 0.079 | 192.6 | 641,960 |
| 7DGS-PBR (avg) | **7DGS** | **32.50** | **0.958** | **0.051** | 174.7 | **98,440** |
| D-NeRF (avg) | 4DGS | 33.21 | 0.969 | 0.036 | 296.4 | 255,319 |
| D-NeRF (avg) | **7DGS** | **34.34** | **0.972** | **0.032** | 194.2 | **47,378** |
| Technicolor (avg) | 4DGS | 33.25 | 0.905 | 0.216 | 84.2 | 838,892 |
| Technicolor (avg) | **7DGS** | **33.58** | **0.912** | **0.198** | 79.2 | **416,390** |

The largest gain is observed on the *heart1* scene: PSNR improves from 27.30 → 35.48 (+8.18 dB), while the point count drops from 694K to 83K (only 11.9% retained).

### Ablation Study

| Variant | PSNR (7DGS-PBR) | FPS | # Points |
|------|-----------------|-----|------|
| 4DGS (baseline) | 27.79 | 192.6 | 641,960 |
| 7DGS w/o AGR | 31.77 | **376.0** | 88,393 |
| 7DGS (full) | **32.50** | 174.7 | 98,440 |

Even without AGR, 7DGS surpasses 4DGS by +3.98 dB while doubling FPS to 376 (377.8 FPS on D-NeRF), demonstrating the effectiveness of the core 7D representation. AGR contributes an additional +0.73 dB at the cost of reduced rendering speed.

### Key Findings
- 7DGS achieves the largest gains on scenes with pronounced view-dependent effects (*heart*, *cloud*, *suzanne*), with PSNR improvements of 4–8 dB
- 7DGS requires only 15–50% as many points as 4DGS, as the unified representation eliminates redundant Gaussians
- 7DGS w/o AGR offers a favorable speed–quality trade-off (401 FPS with quality far exceeding 4DGS)

## Highlights & Insights
- **Mathematically elegant unified representation**: The closed-form solution of multivariate Gaussian conditioning reduces 7D to 3D without auxiliary networks, yielding a theoretically clean formulation
- **Backward compatibility with the 3DGS ecosystem**: The sliced 3D Gaussians directly reuse 3DGS rasterization and densification modules, facilitating practical deployment
- **Efficiency**: Higher quality is achieved with fewer points and rendering speeds up to 401 FPS
- A custom **7DGS-PBR dataset** (real CT heart + volumetric cloud + volumetric flame) is introduced, filling the evaluation gap for dynamic scenes with view-dependent effects

## Limitations & Future Work
- The 7D covariance matrix has 28 independent parameters per Gaussian, substantially more than 3DGS (6) and 4DGS, increasing memory overhead
- Gains on the Technicolor dataset are modest (+0.33 dB), suggesting limited generalization to complex real-world scenes
- The AGR MLP introduces additional computation, causing the full 7DGS to achieve lower FPS than 4DGS
- Color is still represented with spherical harmonics; time-dependent color modeling is not incorporated
- Conditional slicing requires inverting $\Sigma_{(t,d)}$ (a $4\times4$ matrix), which, while small, incurs overhead when applied to a large number of Gaussians

## Related Work & Insights
- **3DGS / 4DGS / 6DGS**: This work is a natural extension of the dimensionality-expansion line within the Gaussian Splatting family, unifying temporal and directional components
- **D-NeRF / HexPlane**: NeRF-based dynamic scene methods; 7DGS outperforms them comprehensively in both speed and quality
- **Insight**: The conditional slicing paradigm is generalizable to higher-dimensional Gaussians (e.g., incorporating lighting or material dimensions) and to other scene representations requiring dimensionality reduction
- Compared to Ex4DGS (which models motion via keyframe interpolation), 7DGS implicitly encodes motion through the covariance matrix without requiring explicit keyframes

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First unified 7D Gaussian representation for spatial–temporal–directional modeling; the conditional slicing mechanism is elegantly designed
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on three datasets including a custom benchmark; additional validation on real-world scenes would strengthen the work
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous and algorithmic pseudocode is clear
- Value: ⭐⭐⭐⭐⭐ Unified framework with real-time performance represents a significant advance for dynamic rendering

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgslm_faster_gaussiansplatting_optimization_with_levenbergm.md)
- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)

</div>

<!-- RELATED:END -->
