---
title: >-
  [Paper Note] ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery
description: >-
  [3D Vision] This paper proposes a residual split operation to replace the binary split/clone mechanism in 3D-GS, combined with image pyramid progressive supervision and a variable gradient threshold selection strategy, to adaptively address both over-reconstruction and under-reconstruction simultaneously, achieving state-of-the-art rendering quality while reducing the number of Gaussians.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 35252b6359a0c9f6
---

# ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2412.07494](https://arxiv.org/abs/2412.07494)
- **Code**: [Project Page](https://yanzhelyu.github.io/resgs.github.io/)
- **Area**: 3D Vision · Novel View Synthesis
- **Keywords**: 3D Gaussian Splatting, Residual Densification, Progressive Training, Novel View Synthesis, Detail Recovery

## TL;DR
This paper proposes a residual split operation to replace the binary split/clone mechanism in 3D-GS, combined with image pyramid progressive supervision and a variable gradient threshold selection strategy, to adaptively address both over-reconstruction and under-reconstruction simultaneously, achieving state-of-the-art rendering quality while reducing the number of Gaussians.

## Background & Motivation

3D Gaussian Splatting (3D-GS) has achieved high fidelity and real-time rendering speed in novel view synthesis, but its densification mechanism suffers from fundamental limitations:

**The split/clone binary dilemma**: Existing methods use a fixed threshold $\tau_s$ to determine the densification operation — when scale $s > \tau_s$, a split is applied (decomposing large Gaussians); when $s < \tau_s$, a clone is applied (duplicating small Gaussians). However, the two operations target over-reconstruction and under-reconstruction respectively and cannot address both simultaneously.

**Threshold selection contradiction**:
   - High threshold → insufficient splitting → blurry details (over-reconstruction unresolved)
   - Low threshold → suppressed cloning → missing geometry (under-reconstruction uncompensated)

**Redundancy issue**: 3D-GS tends to split first and then clone, causing Gaussians in the scene to converge to similar scales, with textureless regions redundantly covered by large numbers of small Gaussians.

**Core Goal**: Design an adaptive densification operation that eliminates the binary split/clone dilemma.

## Method

### Residual Split

The core idea is that for any Gaussian $G_i$ requiring densification, a **scaled-down copy** is generated as a residual supplement while the opacity of the original Gaussian is reduced:

**Step 1: Generate a scaled-down copy $G_j$**

$$\mathbf{S}_j = \frac{1}{\lambda_s} \mathbf{S}_i$$

$$\mathbf{R}_j = \mathbf{R}_i, \quad SH_j = SH_i, \quad o_j = o_i$$

$$\boldsymbol{\mu}_j \sim \mathcal{N}(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$$

The new Gaussian inherits the rotation, spherical harmonic coefficients, and opacity of the original Gaussian, with its scale reduced by a factor of $\lambda_s$ and its position sampled randomly from the distribution of the original Gaussian.

**Step 2: Reduce the opacity of the original Gaussian**

$$o'_i = \beta \cdot o_i$$

where $\beta$ is a predefined factor (default 0.3), preventing excessive density in the overlapping region of the two Gaussians.

**Adaptivity Analysis**:
- **Under-reconstructed regions**: The original Gaussian scale remains unchanged while the added small Gaussian expands coverage → compensates for missing geometry.
- **Over-reconstructed regions**: The small Gaussian provides finer-grained fitting → recovers detail.
- **Textureless regions**: A small number of large Gaussians suffices for coverage, avoiding redundant accumulation of same-scale small Gaussians.

### Image Pyramid Progressive Supervision

The training process is divided into $L$ stages, constructing an $L$-level image pyramid $\{\mathcal{I}_i\}_{i=1}^L$:

$$(H_i^v, W_i^v) = (H_L^v / 2^{L-i}, W_L^v / 2^{L-i})$$

- Stage $i$ uses images at resolution level $i$ for supervision.
- Early stages focus on overall structure (low frequency); later stages focus on fine details (high frequency).
- This decouples coverage optimization from detail optimization, reducing optimization difficulty.

### Variable Gradient Threshold Selection Strategy

Each Gaussian $G_i$ is assigned a **fineness level** $l_i$ (initial Gaussians have $l_i = 0$; newly densified Gaussians satisfy $l_j = l_i + 1$).

Each stage is further divided into $K$ sub-stages, yielding $L \times K$ sub-stages in total. At the $k$-th sub-stage:

$$\tau_{k,i} = \begin{cases} \tau, & l_i \geq k \\ \frac{\tau}{\alpha^{k-l_i}}, & l_i < k \end{cases}$$

where $\alpha > 1$. **Effect**: As training progresses, the densification threshold for coarse-grained Gaussians (small $l_i$) gradually decreases, encouraging further refinement and introducing finer structures in later stages.

### Implementation Details

- $L=3$, $K=3$ (9 sub-stages in total)
- Stage 1: 2500 steps; Stage 2: 3500 steps; remaining steps form Stage 3
- $\alpha=2^{1/3}$, $\lambda_s=1.6$, $\beta=0.3$
- Total training: 30K steps; densification stops at 12,000 steps
- Loss function: $\mathcal{L}_1$ + D-SSIM (consistent with original 3D-GS)

## Key Experimental Results

### Main Results: Quantitative Comparison on Three Datasets

| Method | Mip-NeRF360 PSNR↑ | SSIM↑ | LPIPS↓ | Memory | T&T PSNR↑ | SSIM↑ | LPIPS↓ | DB PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------------|-------|--------|------|----------|-------|--------|---------|-------|--------|
| 3D-GS | 27.21 | 0.815 | 0.214 | 734MB | 23.14 | 0.841 | 0.183 | 29.41 | 0.903 | 0.243 |
| Scaffold-GS | 27.69 | 0.812 | 0.225 | 176MB | 23.96 | 0.853 | 0.177 | 30.21 | 0.906 | 0.254 |
| AbsGS | 27.49 | 0.820 | 0.191 | 728MB | 23.73 | 0.853 | 0.162 | 29.67 | 0.902 | 0.236 |
| FreGS | 27.85 | 0.826 | 0.209 | - | 23.96 | 0.849 | 0.178 | 29.93 | 0.904 | 0.240 |
| Mini-Splatting-D | 27.51 | 0.831 | 0.176 | 1.11GB | 23.23 | 0.853 | 0.140 | 29.88 | 0.906 | 0.211 |
| **Ours (AbsGS)** | **28.00** | **0.833** | **0.174** | 698MB | **24.38** | **0.867** | **0.132** | 29.91 | 0.902 | 0.227 |
| Ours-Small | 27.94 | 0.830 | 0.191 | **342MB** | 24.33 | 0.862 | 0.150 | 30.01 | 0.906 | 0.234 |

The proposed method achieves state-of-the-art performance across all metrics on Mip-NeRF360; achieves the best SSIM and LPIPS on Tanks & Temples; and the compact model variant maintains high quality while significantly reducing memory consumption.

### Ablation Study

| Configuration | Mip-NeRF360 PSNR↑ | SSIM↑ | LPIPS↓ | Deep Blending PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------------|-------|--------|---------------------|-------|--------|
| Base | 27.41 | 0.817 | 0.189 | 29.60 | 0.899 | 0.240 |
| Base + RS | 27.66 | 0.825 | 0.183 | 29.68 | 0.900 | 0.233 |
| Base + IP | 27.54 | 0.823 | 0.182 | 29.05 | 0.896 | 0.245 |
| Base + RS + IP | 27.88 | 0.831 | 0.178 | 29.82 | 0.902 | 0.230 |
| **Base + RS + IP + VT (full)** | **28.00** | **0.833** | **0.174** | **29.91** | **0.902** | **0.227** |

(RS = Residual Split, IP = Image Pyramid, VT = Variable Gradient Threshold)

### Cross-Method Compatibility of Residual Split

| Method | PSNR↑ (MipNeRF360) | Memory | Improvement |
|------|-------------------|------|------|
| 3D-GS | 27.21 | 734MB | — |
| 3D-GS + residual split | 27.44 | 586MB | +0.23, −20% memory |
| AbsGS | 27.49 | 728MB | — |
| AbsGS + residual split | 27.71 | 712MB | +0.22 |
| Pixel-GS | 27.52 | 1.32GB | — |
| Pixel-GS + residual split | 27.62 | 1.00GB | +0.10, −24% memory |
| Mini-Splatting-D | 27.51 | 1.11GB | — |
| Mini-Splatting-D + residual split | 27.64 | 1.04GB | +0.13 |

Residual split consistently yields PSNR improvements and memory reductions across all four 3D-GS variants.

### Key Findings

1. **RS and IP are complementary**: Using the image pyramid alone degrades performance in textureless regions (Deep Blending) due to overfitting by small Gaussians, but combining it with RS yields benefits.
2. **Hyperparameter robustness**: Within the ranges $\beta \in [0.05, 0.4)$ and $\lambda_s \in [1.55, 2.0]$, the LPIPS variation is only 0.006.
3. **Efficiency advantage**: The compact model variant achieves the highest FPS (141–206) and shortest training time (11–20 minutes) while maintaining near state-of-the-art quality.

## Highlights & Insights

1. **Precise problem diagnosis**: The paper accurately identifies the binary split/clone selection as the fundamental bottleneck of 3D-GS, rather than technical details such as gradient computation or learning rate.
2. **Elegant simplicity**: The core operation of residual split — scaled-down copy plus opacity reduction — is extremely simple yet simultaneously addresses both over-reconstruction and under-reconstruction.
3. **Plug-and-play**: The operation can directly replace the split/clone step in any 3D-GS variant without modifying other components, yielding consistent improvements.
4. **Synergistic coarse-to-fine design**: The pyramid supervision, variable threshold, and residual split form an organic whole — any single component is less effective than their combination.

## Limitations & Future Work

- The method does not achieve best performance across all metrics on textureless datasets such as Deep Blending, lacking explicit regularization for textureless regions.
- The variable gradient threshold introduces additional hyperparameters ($\alpha$, $K$), increasing tuning complexity.
- Validation is limited to static scenes; extension to dynamic scenes or downstream tasks such as SLAM has not been explored.
- The multi-stage split ratio for progressive training (2500/3500/remaining) relies on manual specification.

## Related Work & Insights

- **3D-GS Improvements**: AbsGS (absolute gradient accumulation), Pixel-GS (pixel-level analysis), Mini-Splatting (spatial position reorganization), GaussianPro (progressive propagation)
- **Progressive Training**: FreGS (frequency-domain progressive regularization), Octree-GS (octree depth progression), Pyramid NeRF (image pyramid + implicit field subdivision)
- **Efficient 3D-GS**: Scaffold-GS (anchor + MLP), Mip-Splatting (anti-aliasing filtering)

**Insights**: The densification strategy is the core bottleneck for 3D-GS quality, rather than network architecture or loss functions. The successful application of the "residual" concept to 3D Gaussians suggests that the philosophy underlying ResNet may find broader utility in explicit representations.

## Rating
- **Novelty**: ★★★★☆ — The residual split concept is novel and intuitively clear, though the overall framework remains within the scope of 3D-GS refinement.
- **Technical Depth**: ★★★★☆ — The synergistic design of three components is sophisticated, and ablation studies thoroughly reveal the role of each.
- **Practicality**: ★★★★★ — Plug-and-play, significant improvements, and training efficiency make this work highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)

</div>

<!-- RELATED:END -->
