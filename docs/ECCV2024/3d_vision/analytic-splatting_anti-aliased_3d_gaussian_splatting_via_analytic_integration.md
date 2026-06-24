---
title: >-
  [Paper Note] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] By analytically approximating the integration of Gaussian signals over pixel windows using a conditional logistic function, instead of pixel-center point sampling in 3DGS, alias-free 3D Gaussian Splatting is achieved, outperforming Mip-Splatting in multi-scale rendering.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Anti-Aliasing"
  - "Analytic Integration"
  - "novel view synthesis"
  - "Multi-Scale"
date: 2026-05-08
content_hash: 1fe7b9e7893e656b
---

# Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration

**Conference**: ECCV 2024  
**arXiv**: [2403.11056](https://arxiv.org/abs/2403.11056)  
**Code**: [https://lzhnb.github.io/project-pages/analytic-splatting/](https://lzhnb.github.io/project-pages/analytic-splatting/)  
**Area**: 3D Vision / Novel View Synthesis  
**Keywords**: 3D Gaussian Splatting, Anti-Aliasing, Analytic Integration, novel view synthesis, Multi-Scale

## TL;DR
By analytically approximating the integration of Gaussian signals over pixel windows using a conditional logistic function, instead of pixel-center point sampling in 3DGS, alias-free 3D Gaussian Splatting is achieved, outperforming Mip-Splatting in multi-scale rendering.

## Background & Motivation
**Background**: 3D Gaussian Splatting (3DGS) represents scenes as 3D Gaussians and rasterizes them to pixels, achieving real-time high-quality novel view synthesis. However, standard 3DGS only evaluates the Gaussian values at the center point of each pixel, treating pixels as isolated points rather than area regions.

**Limitations of Prior Work**:
   - **Severe Aliasing**: When training and testing resolutions differ, or when camera distances change in multi-view images, changes in the pixel footprint cause insufficient sampling bandwidth (violating Nyquist), leading to aliasing and blurriness.
   - **Over-smoothing of Mip-Splatting**: Mip-Splatting uses 3D Gaussian filtering as pre-filtering for anti-aliasing, but this over-smoothes high-frequency details and drops sharpness.
   - **High Cost of Super-Sampling**: Sampling each pixel multiple times and averaging yields good quality but increases computational cost linearly.

**Key Challenge**: Anti-aliasing requires considering the area effect of pixels (integration instead of point sampling), but accurately computing the integration of 2D Gaussians over a rectangular window is difficult (the 2D Gaussian CDF has no analytical closed-form).

**Goal**: How to efficiently integrate the Gaussian signal over the pixel window to eliminate aliasing without introducing over-smoothing?

**Key Insight**: Accurately approximate the 1D Gaussian CDF with a logistic function, then decompose the 2D integration into two independent 1D integrations via covariance matrix diagonalization.

**Core Idea**: Use the conditional logistic function $S(x) = 1/(1+\exp(-1.6x - 0.07x^3))$ to analytically approximate the Gaussian CDF, replacing point sampling with pixel window integration with an error of only $10^{-4}$ order of magnitude.

## Method

### Overall Architecture
In the rendering pipeline of 3DGS, the evaluation of the Gaussian value for each pixel is changed from "center point sampling" to "window integration". Technically, the conditional logistic function is used to approximate the Gaussian CDF, and a separable integration of the 2D window is achieved through the eigendecomposition of the covariance matrix.

### Key Designs

1. **Conditional Logistic Function Approximation for Gaussian CDF**:

    - **Function**: Approximate the CDF of the standard normal distribution using $S(x) = \frac{1}{1+\exp(-1.6x - 0.07x^3)}$.
    - **Mechanism**: Window integration = difference of CDFs: $\mathcal{I}_g(u) = S_\sigma(u+1/2) - S_\sigma(u-1/2)$, where $S_\sigma(x) = S(x/\sigma)$.
    - **Design Motivation**: The approximation error is only of $10^{-4}$ order of magnitude, differentiable-friendly (suitable for backpropagation), and unlike Gaussian filtering, it does not over-smooth high-frequency components.

2. **Decomposition of 2D Window Integration**:

    - **Function**: Transform the integration of a 2D Gaussian over a rectangular pixel window into the product of two independent 1D integrations.
    - **Mechanism**: Perform eigendecomposition on the 2D covariance matrix $\hat{\Sigma}$ to obtain eigenvalues $\{\lambda_1, \lambda_2\}$ and eigenvectors $\{v_1, v_2\}$, rotate the integration domain to the direction of the eigenvectors to diagonalize the covariance, and then decompose: $\mathcal{I}_g^{2D}(\mathbf{u}) \approx 2\pi\sigma_1\sigma_2 [S_{\sigma_1}(\tilde{u}_x+1/2) - S_{\sigma_1}(\tilde{u}_x-1/2)][S_{\sigma_2}(\tilde{u}_y+1/2) - S_{\sigma_2}(\tilde{u}_y-1/2)]$.
    - **Design Motivation**: Directly performing 2D Gaussian integration has no analytical closed-form. Rotation and decomposition introduce minor errors but allow for analytical computation.

3. **Replacing the Volume Rendering Formulation**:

    - **Function**: Replace $g_i^{2D}(\mathbf{u})$ in the volume rendering of 3DGS with $\mathcal{I}_{g_i}^{2D}(\mathbf{u})$.
    - **Formulation**: $\mathbf{C}(\mathbf{u}) = \sum_{i} T_i \mathcal{I}_{g_i}^{2D}(\mathbf{u}|\hat{\mu}_i, \hat{\Sigma}_i) \alpha_i \mathbf{c}_i$.
    - **Design Motivation**: Fully compatible with the CUDA rasterization pipeline of 3DGS, requiring only modifications to the shader.

### Loss & Training
- Uses the same training strategy and loss functions ($L_1$ + SSIM) as 3DGS.
- Custom CUDA shader is implemented for the analytic shading module.
- Supports backpropagation (gradients are derived in the appendix).

## Key Experimental Results

### Main Results (Multi-Scale Blender Synthetic)

| Method | Full PSNR | 1/2 PSNR | 1/4 PSNR | 1/8 PSNR | Avg PSNR |
|------|-----------|----------|----------|----------|----------|
| 3DGS | 28.79 | 30.66 | 31.64 | 27.98 | 29.77 |
| 3DGS-SS | 32.05 | 33.78 | 33.92 | 31.12 | 32.71 |
| Mip-Splatting | 32.81 | 34.49 | 35.45 | 35.50 | 34.56 |
| **Ours** | **33.22** | **34.92** | **35.98** | **36.00** | **35.03** |

### Multi-Scale Mip-NeRF 360

| Method | Avg PSNR | Avg SSIM | Avg LPIPS |
|------|----------|----------|-----------|
| 3DGS | 27.63 | 0.853 | 0.156 |
| Mip-Splatting | 29.12 | 0.883 | 0.134 |
| **Ours** | **29.51** | **0.887** | **0.123** |

### Ablation Study

| Approximation Scheme | Characteristics | Error Order of Magnitude |
|---------|------|-------------|
| 3DGS (Point Sampling) | Center point sampling only | Largest |
| Super-Sampling | Multi-point sampling average | Medium |
| Mip-Splatting (Gaussian Filtering) | Pre-filtering $\rightarrow$ Over-smoothing | Medium |
| **Ours (Logistic CDF)** | Analytic integration | **~10⁻⁴** |

### Key Findings
- **Most Advantageous at Small $\sigma$ (High-Frequency Details)**: Mip-Splatting's Gaussian filtering over-smoothes at small $\sigma$, while the proposed method preserves high frequencies.
- **Most Significant Improvement at 1/8 Resolution**: 36.00 vs 35.50 PSNR, as anti-aliasing is more crucial at lower resolutions.
- **Advantage on Super-Resolution ($2\times$)**: 26.90 vs 26.46 PSNR (vs. Mip-Splatting).
- **Frame Rate Decreased by ~10%**: Extra square root and exponential operations introduce minor computational overhead.

## Highlights & Insights
- **Mathematical Elegance**: The idea of using a logistic function to approximate the Gaussian CDF is very clever—the specific parameterization of $S(x) = 1/(1+\exp(-1.6x-0.07x^3))$ achieves an accuracy of $10^{-4}$ order of magnitude.
- **Superior Anti-Aliasing Strategy over "Pre-filtering"**: Mip-Splatting applies low-pass filtering at the signal end (effectively blurring), whereas this method integrates at the sampling end (without altering the signal), thereby preserving more high-frequency details.
- **Separable Integration via Covariance Diagonalization**: Rotating the 2D window to the direction of the principal axes of covariance. Although the rectangular window becomes a rotated rectangle (which does not perfectly match the pixel boundaries), the error is well-controlled.

## Limitations & Future Work
- **~10% Frame Rate Reduction**: More exponential and square root calculations. This could potentially be alleviated through CUDA optimization or utilizing Tensor Cores.
- **Approximation Error from Rotated Pixel Windows**: The integrated domain after diagonalization is no longer perfectly aligned with the pixels, yielding larger errors when the rotation angle is large.
- **Applicable Range of the Logistic Function $\sigma \in [0.3, 6.6]$**: Accuracy drops outside this range, though it is rarely exceeded in practical 3DGS settings.
- **Lack of Comparison with Recent Methods (e.g., 2DGS, GS-Surfel)**: 3DGS variants are evolving rapidly, necessitating broader comparisons.

## Related Work & Insights
- **vs Mip-Splatting**: Both works tackle the aliasing problem of 3DGS. Mip-Splatting performs Gaussian pre-filtering in 3D space (modifying the signal), while this method performs window integration in 2D (modifying the sampling). The integration strategy preserves more details.
- **vs Mip-NeRF**: Mip-NeRF uses conical pixel frustum projection + IPE for anti-aliasing. The concept is similar but implemented within the NeRF framework. This work brings a similar insight to 3DGS.

## Rating
- Novelty: ⭐⭐⭐⭐ The technical solution involving logistic CDF approximation and covariance diagonalization is novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-scale, super-resolution, and comprehensive error analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations, illustrative figures, and meticulous error analysis.
- Value: ⭐⭐⭐⭐ Provides fundamental improvements to the core rendering pipeline of 3DGS, which can be directly applied to all 3DGS-based methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Anti-Aliased 2D Gaussian Splatting](../../NeurIPS2025/3d_vision/anti-aliased_2d_gaussian_splatting.md)
- [\[ECCV 2024\] SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [\[ECCV 2024\] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy](on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)

</div>

<!-- RELATED:END -->
