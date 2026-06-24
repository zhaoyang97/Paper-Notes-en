---
title: >-
  [Paper Note] Volumetrically Consistent 3D Gaussian Rasterization
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] This paper points out unnecessary physical approximations in 3DGS splatting rendering and proposes to directly analytically integrate the transmittance of 3D Gaussians within the rasterization framework to compute more accurate alpha values. This maintains the speed advantage of rasterization while achieving physical accuracy close to ray tracing.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Volumetric Consistency"
  - "Analytical Transmittance"
  - "Novel View Synthesis"
  - "Computed Tomography"
date: 2026-05-08
content_hash: f76f7f57362964b0
---

# Volumetrically Consistent 3D Gaussian Rasterization

**Conference**: CVPR 2025  
**arXiv**: [2412.03378](https://arxiv.org/abs/2412.03378)  
**Code**: [https://github.com/chinmay0301ucsd/Vol3DGS](https://github.com/chinmay0301ucsd/Vol3DGS)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Volumetric Consistency, Analytical Transmittance, Novel View Synthesis, Computed Tomography

## TL;DR

This paper points out unnecessary physical approximations in 3DGS splatting rendering and proposes to directly analytically integrate the transmittance of 3D Gaussians within the rasterization framework to compute more accurate alpha values. This maintains the speed advantage of rasterization while achieving physical accuracy close to ray tracing.

## Background & Motivation

**Background**: 3DGS achieves fast rendering by projecting (splatting) 3D Gaussians to 2D Gaussians in screen space, achieving remarkable results in novel view synthesis. Ray tracing methods (such as NeRF) are physically more accurate but slow, while 3DGS rasterization is fast but sacrifices physical accuracy.

**Limitations of Prior Work**: The EWA splatting in 3DGS introduces three additional approximations: (1) linearization of the exponential transmittance ($e^{-x} \approx 1-x$); (2) assumption of no self-occlusion; (3) affine approximation of the covariance matrix (Jacobian approximation). These approximations make it difficult for 3DGS to represent opaque surfaces—splatted 2D Gaussians can only reach $\alpha=1$ at the center, remaining semi-transparent in other regions.

**Key Challenge**: The speed advantage of rasterization and the physical accuracy of volumetric rendering seem irreconcilable. However, the authors point out that splatting is not a necessary step for rasterization, and rasterization itself can directly perform volumetric integration.

**Goal**: To eliminate physical approximations introduced by splatting while preserving the rendering speed of rasterization.

**Key Insight**: Leveraging the mathematical property that the integration of a 3D Gaussian along a ray can be solved analytically.

**Core Idea**: Directly integrate the density of each Gaussian analytically along the ray direction in 3D space to calculate the transmittance $\alpha_i = 1 - \exp(-\int \kappa_i G_i(\mathbf{r}(t)) dt)$, replacing the approximated alpha values obtained from splatting in 3DGS. This acts as a drop-in replacement that can be directly embedded into the 3DGS framework.

## Method

### Overall Architecture

The input consists of the 3D Gaussian scene representation (density $\kappa_i$, mean $\boldsymbol{\mu}_i$, covariance $\boldsymbol{\Sigma}_i$, color $c_i$). It shares the same alpha blending framework $C(p) = \sum_i c_i \alpha_i \prod_{j<i}(1-\alpha_j)$ with 3DGS, but replaces the splatted alpha of 3DGS with analytically computed, volumetrically consistent alpha values. The entire pipeline is still executed within the rasterizer.

### Key Designs

1. **无 splatting 的 Alpha Blending 推导**:

    - **Function**: Prove that the volumetric rendering equation can be expressed as alpha blending without using splatting
    - **Mechanism**: Substitute the density field $\sigma(\mathbf{x}) = \sum_i \kappa_i G_i(\mathbf{x})$ into the volume rendering equation, assuming the Gaussians do not overlap and are sorted from front to back to separate each Gaussian's contribution. Utilizing the differential property of exponential transmittance $dT(t_{in}, t) = -T(t_{in}, t) \sigma(\mathbf{r}(t))$, it is proved that the inner integral is exactly equal to $\alpha_i = 1 - \bar{T}_i$. This yields an alpha blending equation structurally identical to 3DGS, but with a different definition for $\alpha_i$.
    - **Design Motivation**: Shows that the difference lies solely in how alpha values are calculated. Thus, replacing the alpha computation is sufficient to bring volumetric consistency into any 3DGS framework, enabling a drop-in replacement with minimal modifications.

2. **解析透射率计算**:

    - **Function**: Accurately calculate the accumulated opacity of each 3D Gaussian along the ray
    - **Mechanism**: The integration of a 3D Gaussian along the ray direction can be simplified to a 1D Gaussian integration. First, project the 3D Gaussian along the ray direction to obtain the 1D parameters: mean $\gamma_j = (\boldsymbol{\mu}_j - \mathbf{o})^T \Sigma_j^{-1} \mathbf{d} / (\mathbf{d}^T \Sigma_j^{-1} \mathbf{d})$ and variance $\beta_j = 1/\sqrt{\mathbf{d}^T \Sigma_j^{-1} \mathbf{d}}$. Then, integrating the 1D Gaussian yields an error function, which simplifies to $\bar{T}_j = \exp(-\kappa_j G_j(\gamma_j \mathbf{d}) \sqrt{2\pi} \beta_j)$ under the assumption of infinite support.
    - **Design Motivation**: 3DGS splatting discards scale information along the z-direction (along the ray), leading to identical rendering results for Gaussians of different z-scales, which is physically unrealistic. Analytical integration preserves the impact of the z-scale, where a larger $\beta_j$ (wider in the z-direction) results in a larger alpha.

3. **密度重参数化**:

    - **Function**: Address the gradient vanishing problem in optimizing high-density Gaussians
    - **Mechanism**: Reparameterize the density $\kappa$ as $\kappa = -\log(1 - 0.99\theta) \cdot \frac{1}{3}(\frac{1}{s_x} + \frac{1}{s_y} + \frac{1}{s_z})$, where $\theta \in [0,1]$. This naturally assigns higher density (more opaque) to smaller Gaussians and lower density to larger Gaussians, preventing large Gaussians from creating blurry occlusions. Meanwhile, adaptive density refinement strategies are modified: halve the density during split/clone, and prune low-density points.
    - **Design Motivation**: Volumetrically consistent alpha computation allows density $\kappa$ to approach infinity to represent fully opaque surfaces, which can lead to vanishing gradients. Reparameterization maintains the gradient flow while preserving the representation capability.

### Loss & Training

Uses the same loss function as 3DGS (L1 + SSIM), implemented based on the SLANG.D rasterizer. Opacity reset is disabled (as experiments show no benefit), and strategies for splitting based on high density and pruning based on low density are added.

## Key Experimental Results

### Main Results

Novel view synthesis comparison (Mip-NeRF360 / Tanks&Temples / DeepBlending):

| Method | MN360 SSIM↑ | MN360 LPIPS↓ | T&T SSIM↑ | T&T LPIPS↓ | DB SSIM↑ | DB LPIPS↓ |
|---|---|---|---|---|---|---|
| 3DGS-Slang | 0.813 | 0.222 | 0.850 | 0.176 | 0.906 | 0.248 |
| GES | 0.794 | 0.250 | 0.836 | 0.198 | 0.901 | 0.252 |
| EVER | 0.825 | 0.194 | 0.870 | 0.160 | 0.908 | 0.308 |
| **Ours** | **0.813** | **0.209** | **0.854** | **0.167** | **0.908** | **0.247** |

### Ablation Study

Fitting experiments on opaque objects:

| Method | Piecewise Constant Texture LPIPS↓ | Description |
|---|---|---|
| 3DGS | 0.027 | 2D Gaussians cannot fit sharp boundaries |
| Ours | 0.005 | High density makes Gaussians close to opaque, fitting perfectly |

### Key Findings

- Consistently matches or outperforms 3DGS on SSIM and LPIPS metrics, particularly on Tanks&Temples where LPIPS drops from 0.176 to 0.167.
- Volumetrically consistent alpha enables the same number of Gaussians to better represent opaque surfaces, reducing artifacts and blurriness at edges.
- As a side benefit, the method can be directly applied to computed tomography (CT) reconstruction, matching the quality of the SOTA method R2-Gaussian while using fewer points.
- Slightly lower on PSNR compared to 3DGS (27.30 vs 27.52 on MN360), but superior in perceptual metrics (SSIM/LPIPS).

## Highlights & Insights

1. **Elegant "drop-in replacement" design**: Volumetric consistency is achieved simply by replacing the alpha calculation. It is compatible with all subsequent 3DGS works, demonstrating a highly referenceable philosophy of minimal modification.
2. **In-depth analysis of splatting approximations**: Clearly deconstructs the individual impact of the three levels of approximation in 3DGS, providing a theoretical foundation for understanding the limitations of 3DGS.
3. **Zero-shot transfer from view synthesis to CT**: A physically accurate rendering model is naturally suited for tasks requiring integrated density, showcasing the practical value of physical consistency.

## Limitations & Future Work

- Underperforms 3DGS in terms of PSNR (as PSNR is less sensitive to high-frequency details, and the model prefers a more compact representation).
- Due to the computational overhead of the error function (erf), FPS is slightly lower than the original 3DGS (136 vs 159 on MN360).
- Still retains the per-tile sorting and non-overlapping assumption, not completely solving all approximations of volume rendering.
- Future work can integrate more complete sorting and overlap handling to further improve physical accuracy.

## Related Work & Insights

- **vs 3DGS**: 3DGS uses splatting approximation, which is replaced by analytical integration in this paper. This is a direct improvement on the core rendering engine of 3DGS.
- **vs 2DGS**: 2DGS uses 2D Gaussians to eliminate the affine approximation, but still does not calculate exact transmittance. Ours is physically more accurate.
- **vs EVER**: EVER uses ray tracing for precise calculations (including sorting and overlaps), yielding the highest quality but slower speed. Ours achieves comparable quality within a rasterization framework.
- Insight: Replacing the core physical model without changing the overall framework can yield significant improvements.

## Rating

- Novelty: 7/10 — The analytical integration idea is theoretically straightforward, but the derivation is elegant and holds high practical value.
- Experimental Thoroughness: 7/10 — Comprehensive coverage of mainstream datasets, but lacks comparisons with more recent methods.
- Writing Quality: 9/10 — Mathematical derivations are clear and rigorous, illustrations are intuitive, and physical intuition is well-explained.
- Value: 8/10 — As a plug-and-play improvement to 3DGS, it features good compatibility and potential for CT applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](magic-slam_multi-agent_gaussian_globally_consistent_slam.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](../../ICCV2025/3d_vision/stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)

</div>

<!-- RELATED:END -->
