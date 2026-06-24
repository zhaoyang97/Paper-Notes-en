---
title: >-
  [Paper Note] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] Analysis of the projection error introduced by local affine approximation in 3D-GS from a mathematical standpoint, deriving that the error is minimized when the direction connecting the Gaussian mean and camera center is selected as the projection direction, and proposing the Optimal Gaussian Splatting projection strategy.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Projection Error Analysis"
  - "Optimal Projection"
  - "Real-time Rendering"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 39163511052e5323
---

# On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy

**Conference**: ECCV 2024  
**arXiv**: [2402.00752](https://arxiv.org/abs/2402.00752)  
**Code**: [https://letianhuang.github.io/op43dgs/](https://letianhuang.github.io/op43dgs/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Projection Error Analysis, Optimal Projection, Real-time Rendering, Novel View Synthesis

## TL;DR

This work systematically analyzes the projection error introduced by local affine approximation in 3D Gaussian Splatting (3D-GS) mathematically. It proves that the error function reaches its minimum when the Gaussian mean direction aligns with the projection plane normal. Based on this, a projection-to-tangent-plane strategy for each Gaussian is proposed (Optimal Gaussian Splatting), which significantly reduces rendering artifacts without sacrificing real-time performance.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3D-GS) uses Gaussian functions as an explicit scene representation and achieves real-time rendering through differentiable rasterization, becoming the most prominent novel view synthesis method post-NeRF.

**Limitations of Prior Work**: Gaussian functions are closed under affine transformations but not under projective transformations. 3D-GS utilizes a first-order Taylor expansion for local affine approximation to handle projection, but the omitted high-order remainders introduce errors. For Gaussians far from the projection plane center, the error is particularly significant, leading to stretched and cloudy artifacts.

**Key Challenge**: While various improvements for 3D-GS (such as sparse views, storage optimization, mesh reconstruction, etc.) have emerged continuously, the fundamental projection error has been largely neglected, which is a foundational "bedrock" issue.

**Goal**: To quantitatively analyze the relationship between projection error and the positions of Gaussian means, and to identify a projection strategy that minimizes this error.

**Key Insight**: The analysis initiates from the Taylor remainder of the projection, formulation of the error as a function of the Gaussian mean in spherical coordinates, and derivation of the extremum using function optimization theory.

**Core Idea**: Projecting all Gaussians onto a single $z=1$ plane is sub-optimal; projecting each Gaussian onto its respective tangent plane along the direction from the camera center to its mean minimizes the projection error.

## Method

### Overall Architecture

Optimal Gaussian Splatting (OGS) modifies the projection process of 3D-GS: instead of projecting all 3D Gaussians uniformly onto the camera's $z=1$ plane, it computes a unit sphere tangent plane along the direction of the line connecting the camera center to each Gaussian's mean, and performs projection on this tangent plane. Finally, a Unit Sphere Based Rasterizer maps the 2D Gaussians from the tangent planes back to image pixels.

### Key Designs

1. **Derivation of the Projection Error Function**: The first-order Taylor expansion remainder of the projection function $\varphi(\mathbf{x}') = \mathbf{x}'(\mathbf{x_0}^\top \mathbf{x}')^{-1}$ in 3D-GS is given by:

$$R_1(\mathbf{x}') = \varphi(\mathbf{x}') - \varphi(\boldsymbol{\mu}') - \frac{\partial \varphi}{\partial \mathbf{x}'}(\boldsymbol{\mu}')(\mathbf{x}' - \boldsymbol{\mu}')$$

   Taking the mathematical expectation of the squared Frobenius norm of this remainder yields an error function dependent solely on the Gaussian mean:

$$\epsilon(\boldsymbol{\mu}') = \int_{\mathbf{x}' \in \mathcal{X}'} \|R_1(\mathbf{x}')\|_F^2 \, d\mathbf{x}'$$

   After representing the mean in spherical coordinates $(\theta_\mu, \phi_\mu)$, the error function has a closed-form solution. **Design Motivation**: Rigorously mathematicalizing the intuitive observation that "errors are larger further from the center".

2. **Extreme Value Analysis of the Error Function**: Taking partial derivatives of the error function yields:

$$\frac{\partial \epsilon}{\partial \theta_\mu}(0,0) = 0, \quad \frac{\partial \epsilon}{\partial \phi_\mu}(0,0) = 0$$

   The minimum is reached at $(\theta_\mu, \phi_\mu) = (0, 0)$. This implies that the error is minimized when the projection direction of the Gaussian mean aligns with the projection plane normal (i.e., the direction of $\mathbf{x_0}$). The function is flat near the origin but increases sharply near the boundaries. **Core Finding**: The minimum error is greater than zero (since Gaussians are not closed under projective transformation), but it can be minimized by choosing an optimal projection plane.

3. **Optimal Projection Strategy**: An individual projection tangent plane is selected for each Gaussian. The equation of the tangent plane is $\mathbf{x_p}^\top \cdot (\mathbf{x}' - \mathbf{x_p}) = 0$, where:

$$\mathbf{x_p} = \varpi(\boldsymbol{\mu}') = \boldsymbol{\mu}'(\boldsymbol{\mu}'^\top \boldsymbol{\mu}')^{-1/2}$$

   is the projection of the Gaussian mean onto the unit sphere. The optimal projection function is $\varphi_p(\mathbf{x}') = \mathbf{x}'(\mathbf{x_p}^\top \mathbf{x}')^{-1}$. The corresponding Jacobian matrix $\mathbf{J_p}$ is determined by the Gaussian mean coordinates $(\mu_x, \mu_y, \mu_z)$, which can be implemented simply by modifying the Jacobian computation in the forward pass.

4. **Unit Sphere Based Rasterizer**: For an image pixel $(u,v)$, it is first converted to camera space and then projected onto the unit sphere:

$$\mathbf{x}_{2D} = \varphi_p\left(\begin{bmatrix}(u-c_x)/f_x \\ (v-c_y)/f_y \\ 1\end{bmatrix}\right)$$

   Then, the function values of the tangent plane Gaussians at this point are queried to perform alpha blending. Since the projection does not depend on the $z=1$ plane, it naturally adapts to different camera models such as fisheye cameras and panoramic images.

5. **Rotation Matrix $\mathbf{Q}$**: Since the third row/column of $\mathbf{J_p}\boldsymbol{\Sigma}'\mathbf{J_p}^\top$ is not entirely zero (unlike the original 3D-GS), an invertible matrix $\mathbf{Q}$ is introduced to left-multiply $\mathbf{x}_{2D}$, $\varphi_p(\boldsymbol{\mu}')$ and $\mathbf{J_p}$ so that the inverse of the 2D covariance matrix can be correctly extracted. $\mathbf{Q}$ is constructed from the coordinates of the Gaussian mean.

### Loss & Training

The identical default parameters and training strategy as the original 3D-GS (L1 + D-SSIM loss) are employed. Only the Jacobian computation in the projection process and the rasterization pipeline are modified to maintain experimental control. The rasterization is implemented via custom CUDA kernels.

## Key Experimental Results

### Main Results: 13 Real-world Scenes (Mip-NeRF360 + Tanks & Temples + Deep Blending)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Plenoxels | 22.77 | 0.666 | 0.457 |
| INGP-Big | 24.93 | 0.724 | 0.336 |
| Mip-NeRF360 | 27.11 | 0.803 | 0.241 |
| 3D-GS | 26.92 | 0.832 | 0.214 |
| **Ours (OGS)** | **27.17** | **0.836** | **0.210** |

### Robustness to Short Focal Lengths (×0.2 / ×0.3 Focal Length, Mean of 6 Scenes)

| Metric | Focal Length | 3D-GS | OGS (Ours) | Gain |
|------|------|-------|------------|------|
| PSNR ↑ | ×0.2 | 15.58 | 20.46 | **+4.88** |
| PSNR ↑ | ×0.3 | 19.49 | 22.10 | **+2.61** |
| SSIM ↑ | ×0.2 | 0.499 | 0.628 | +0.129 |
| SSIM ↑ | ×0.3 | 0.630 | 0.684 | +0.054 |
| LPIPS ↓ | ×0.2 | 0.397 | 0.251 | **-0.146** |
| LPIPS ↓ | ×0.3 | 0.263 | 0.231 | -0.032 |

### Key Findings

- Under standard focal lengths, OGS improves by approximately 0.25 PSNR over 3D-GS (27.17 vs 26.92), outperforming Mip-NeRF360 while maintaining real-time rendering performance.
- **Tremendous Advantage in Short Focal Length Scenes**: When the focal length is reduced to ×0.2, PSNR improves by 4.88 dB, and LPIPS improves by 0.146, confirming that the projection error has a highly prominent impact in wide-angle settings.
- The projection error of 3D-GS increases sharply as the focal length decreases (the expanded field-of-view cause more Gaussians to deviate from the projection center), whereas OGS is unaffected by this due to its central radial projection.
- OGS naturally adapts to fisheye and panoramic camera models, which are unsupported by the planar projection of the original 3D-GS.

## Highlights & Insights

- **Theory-Driven Improvement**: Distinct from most empirical or tuning-based improvements, this work derives the optimal strategy from rigorous mathematical analysis, with a closed-form proof showing that the error is minimized during tangent plane projection.
- **Minimalist Modifications, Significant Effects**: The core modification only involves altering the Jacobian matrix computation and the rasterization pipeline, without changing training parameters or increasing model complexity.
- **Adaptability to Wide-Angle/Fisheye**: The original 3D-GS severely degrades in wide-angle scenes. OGS's solution not only improves quality but also extends the scope of application, which is of great significance for wide-angle application scenarios such as VR/AR.
- **Error Visualization**: Visualizing the error function $\epsilon(\theta_\mu, \phi_\mu)$ as a 3D surface intuitively demonstrates the explosive growth of error in outer regions.

## Limitations & Future Work

- The current analysis assumes a constant covariance, focusing solely on the impact of the mean on the error. The impact of covariance (Gaussian shape/size) on projection error requires further investigation.
- The additional transformation to the image plane after tangent plane projection leads to a slight increase in training time (inference is unaffected).
- In degenerate cases where the Gaussian mean is extremely close to the camera center ($\|\boldsymbol{\mu}'\| \to 0$ ), the projection may become unstable.
- Optimizing the CUDA implementation could further reduce training time overhead.

## Related Work & Insights

- **3D-GS Ecosystem**: This work rigorously analyzes a neglected fundamental issue within the 3D-GS pipeline, which is orthogonally complementary to various high-level improvements (such as compression, sparse views, dynamic scenes, etc.).
- **Generality of Affine Approximation**: Classical methods such as EWA Splatting also utilize local affine approximations; the analytical framework of this paper may be extended to other splatting techniques.
- **2D GS / Gaussian Surfels**: Subsequent works like 2DGS have shifted towards surface-aligned Gaussians; the projection strategy concept in this paper may also provide insights for such methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Analyzes the projection error of 3D-GS from a fundamental mathematical perspective, offering a novel entry point and theoretical depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive validation across standard scenes, short focal lengths, and multiple camera models, though the number of scenes could be further expanded.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous and clear mathematical derivations, with a complete and fluent logical chain from problem definition to solution.
- **Value**: ⭐⭐⭐⭐ — Minimal modifications yet significant effects, possessing high practical value especially for wide-angle/VR scenarios.

**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Projection Error Analysis, Optimal Projection, Real-time Rendering, Novel View Synthesis

## TL;DR

Analysis of the projection error introduced by local affine approximation in 3D-GS from a mathematical standpoint, deriving that the error is minimized when the direction connecting the Gaussian mean and camera center is selected as the projection direction, and proposing the Optimal Gaussian Splatting projection strategy.

## Background & Motivation

- 3D-GS utilizes a local affine approximation (first-order Taylor expansion) to project 3D Gaussians onto the image plane.
- Gaussian functions are not closed under projective transformations, making projection error inevitable with affine approximations.
- Gaussians deviating from the image center incur larger projection errors, leading to elongated artifacts particularly in wide-angle/short focal length settings.
- There is currently a lack of systematic analysis and optimization solutions for this projection error.

## Method

### Overall Architecture

1. Analyze the projection error function $\epsilon(\theta_\mu, \phi_\mu)$ , establishing the relationship between the error and the position of the Gaussian mean.
2. Identify the minimum error points using function optimization theory.
3. Propose an optimal strategy for projecting each Gaussian onto its tangent plane along the direction from the camera center to its mean.

### Key Designs

- **Derivation of the Error Function**: Starting from the Taylor remainder $R_1(\mathbf{x}')$, the mathematical expectation of the Frobenius norm is calculated to formulate the error function $\epsilon(\theta_\mu, \phi_\mu)$.
- **Extremum Analysis**: Proof that the error reaches its minimum when the spherical coordinates $(\theta_\mu, \phi_\mu)$ of the Gaussian mean equal the projection plane's normal direction $(0,0)$.
- **Optimal Projection**: Instead of projecting onto a unified $z=1$ plane, each Gaussian is projected onto the tangent plane of the unit sphere along the direction from the camera origin to the Gaussian mean.
- **Unit Sphere Rasterizer**: For each pixel, a ray is projected onto the unit sphere, and the intersecting tangent-plane Gaussians are queried for alpha blending.
- **Adaptability to Multiple Camera Models**: Since it does not rely on the $z=1$ plane, it natively supports camera models such as fisheye and panoramic views.

### Loss & Training

The original photometric loss (L1 + D-SSIM) of 3D-GS is utilized, without introducing additional losses.

## Key Experimental Results

### Main Results

Quantitative evaluation on 13 real-world scenes (Mip-NeRF360 + Tanks&Temples + Deep Blending):

| Method | Avg PSNR↑ | Avg SSIM↑ | Avg LPIPS↓ |
|------|-----------|-----------|------------|
| Plenoxels | 22.77 | 0.666 | 0.457 |
| INGP-Big | 24.93 | 0.724 | 0.336 |
| M-NeRF360 | 27.11 | 0.803 | 0.241 |
| 3D-GS | 26.92 | 0.832 | 0.214 |
| **Ours** | **27.17** | **0.836** | **0.210** |

### Ablation Study

Comparison of robustness under short focal length settings (average of 6 scenes):

| Metric | Focal Length Ratio | 3D-GS | Ours |
|------|--------|-------|------|
| PSNR↑ | ×0.2 | 15.58 | 20.46 |
| PSNR↑ | ×0.3 | 19.49 | 22.10 |
| SSIM↑ | ×0.2 | 0.499 | 0.628 |
| LPIPS↓ | ×0.2 | 0.397 | 0.251 |

When the focal length is shortened to ×0.2, the proposed method yields an improvement of approximately 5dB in PSNR compared to the original 3D-GS.

### Key Findings

- Projection error increases sharply at the image boundaries / large view angles.
- Shorter focal lengths lead to larger fields of view (FOV), exacerbating the error and causing spiky and cloudy artifacts.
- The optimal projection only requires modifying the Jacobian computation and does not affect real-time rendering performance.
- It natively supports fisheye cameras and panorama generation.

## Highlights & Insights

- **Theory-Driven**: The optimal projection is derived from rigorous mathematical analysis rather than heuristic design.
- Only a small amount of code modification is required for implementation, making it highly engineering-friendly.
- It addresses the systematic artifact issues of 3D-GS under wide-angle lenses.
- It provides the first closed-form expression of projection error for 3D-GS.

## Limitations & Future Work

- Post-projection coordinates on the tangent plane need to be transformed back to image space, leading to a slight increase in training time.
- The current analysis assumes a constant covariance and does not consider the impact of covariance on the error.
- The practical improvement under standard focal length settings is relatively limited.

## Related Work & Insights

- **3D-GS**: Kerbl et al. 2023 laid the foundation for Gaussian Splatting.
- **NeRF Series**: Methods such as Mip-NeRF360 do not employ approximations throughout the pipeline, yielding slightly higher rendering quality.
- **Insight**: Mathematical analysis of fundamental operations can bring low-cost yet high-reward improvements.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — In-depth theoretical analysis with an elegant derivation of the optimal projection.
- **Value**: ⭐⭐⭐⭐ — Highly practical to implement, though benefits are primarily observed in wide-angle scenes.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Verified across multiple datasets, with highly convincing short focal length experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear mathematical derivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)
- [\[ECCV 2024\] SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [\[ECCV 2024\] Binomial Self-compensation for Motion Error in Dynamic 3D Scanning](binomial_self-compensation_for_motion_error_in_dynamic_3d_scanning.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[ECCV 2024\] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)

</div>

<!-- RELATED:END -->
