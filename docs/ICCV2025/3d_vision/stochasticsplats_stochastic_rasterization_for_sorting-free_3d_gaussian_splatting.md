---
title: >-
  [Paper Note] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting
description: >-
  [3D Vision] StochasticSplats introduces Stochastic Transparency into 3DGS, replacing depth-sorted alpha blending with an unbiased Monte Carlo estimator to achieve sorting-free, popping-free rendering. At 1 SPP, it is 4× faster than standard CUDA 3DGS, and the number of samples provides a flexible quality–speed trade-off.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: ed71a9d0b4c3fe65
---

# StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2503.24366](https://arxiv.org/abs/2503.24366)
- **Code**: Not released (OpenGL + CUDA dual implementation)
- **Area**: 3D Vision
- **Keywords**: 3D Gaussian Splatting, Stochastic Transparency, Sorting-Free Rendering, Monte Carlo Estimation, Volume Rendering, OpenGL Hardware Acceleration

## TL;DR
StochasticSplats introduces Stochastic Transparency into 3DGS, replacing depth-sorted alpha blending with an unbiased Monte Carlo estimator to achieve sorting-free, popping-free rendering. At 1 SPP, it is 4× faster than standard CUDA 3DGS, and the number of samples provides a flexible quality–speed trade-off.

## Background & Motivation

3D Gaussian Splatting has become a dominant radiance field method, yet its core rendering algorithm—depth-sorted sequential rasterization—exhibits several fundamental limitations:

**Costly sorting**: The global sort operation incurs significant overhead in the rendering pipeline.

**Popping artifacts**: Sorting is based on Gaussian centroid depth, so minor camera movements can cause abrupt order changes and temporal discontinuities.

**No flexible trade-off**: For a fixed representation, reducing render resolution does not necessarily improve speed, as each pixel covers more Gaussians.

**Inaccurate billboard approximation**: Projecting 3D Gaussians as camera-aligned billboards cannot correctly handle volumetric blending.

**Poor portability**: Most implementations rely on custom CUDA rasterizers, making them difficult to port to non-NVIDIA platforms and standard graphics pipelines.

Existing methods each address part of these issues with their own trade-offs: StopThePop resolves popping but requires complex CUDA kernels; EVER achieves accurate volumetric blending via ray tracing but is slow; Splatapult uses OpenGL but underperforms CUDA.

## Method

### Core Idea: Stochastic Transparency

Standard alpha blending requires all primitives to be depth-sorted:

$$C = \sum_{i=1}^{L} c_i \alpha_i \prod_{z_k < z_i}(1-\alpha_k)$$

Stochastic Transparency reformulates this as a Monte Carlo estimate—randomly sampling a single primitive $i$:

$$C \approx c_i \quad \text{for} \quad i \sim P$$

where the sampling probability is $P(i) = \alpha_i \prod_{z_k < z_i}(1-\alpha_k)$.

The sampling probability exactly matches the alpha blending weight; after sampling, the pixel color is simply set to $c_i$, and the denominator cancels, yielding an extremely concise estimator.

### Algorithm

For each pixel, all Gaussians are traversed without sorting:
1. Compute the opacity $\alpha_i$ and depth $z_i$ of the current Gaussian.
2. Draw $u \sim \mathcal{U}(0,1)$ uniformly.
3. If $u < \alpha_i$ and $z_i$ is closer than the currently selected Gaussian, select it.
4. The final pixel color equals the color of the selected Gaussian.

Sampling according to $P(i)$ is achieved automatically via standard depth testing (Z-buffer), **requiring no sorting**. Multiple independent samples are averaged to reduce estimation variance.

### Differentiable Stochastic Transparency

Backpropagating gradients through 3DGS training uses a detached gradient estimator:

**Color gradient** (Eq. 7): Only the sampled Gaussian $i$ receives a color gradient; all others receive zero.

$$\frac{\partial \mathcal{L}}{\partial c_i} = \frac{\partial \mathcal{L}}{\partial C}$$

**Opacity gradient** (Eq. 8): The sampled Gaussian and all Gaussians in front of it receive opacity gradients.

$$\frac{\partial \mathcal{L}}{\partial \alpha_i} = \frac{\partial \mathcal{L}}{\partial C} \frac{c_i}{\alpha_i}, \quad \frac{\partial \mathcal{L}}{\partial \alpha_{z_k<z_i}} = \frac{\partial \mathcal{L}}{\partial C} \frac{-c_i}{1-\alpha_{z_k<z_i}}$$

**Decorrelated loss gradient**: $\partial\mathcal{L}/\partial C$ and $\partial C/\partial\theta$ are computed with different random seeds to avoid gradient bias arising from the correlation of two noisy terms.

In practice, training follows path replay backpropagation in three stages:
1. Render an image and compute $\partial\mathcal{L}/\partial C$.
2. Render a second image with a different random seed to obtain $c_i$.
3. Replay the second render with the same seed to compute parameter gradients.

### Eliminating Popping Artifacts

**Simplified approach**: Rather than modifying depth per pixel, the billboard orientation is adjusted to linearly approximate the surface of maximum Gaussian density:

$$\mathbf{n}^\top(\mathbf{x} - \boldsymbol{\mu}) = 0, \quad \text{where} \quad \mathbf{n} = \Sigma^{-1}(\boldsymbol{\mu} - \mathbf{o})$$

This planar approximation of the curved surface preserves hardware-rasterized depth interpolation at no additional time cost.

**Full volumetric blending** (optional high-quality mode): The fixed depth is replaced by a sampled free-path distance $z_i \sim p(t)$, where $p(t)$ is derived from the Beer–Lambert law and the analytically integrated density of 3D Gaussians. For overlapping Gaussians, decomposition tracking selects the shortest free path.

### OpenGL Implementation

- Sampling is naturally realized via standard depth testing in the fragment shader.
- The transparency test is implemented by discarding fragments with `discard` when $u \geq \alpha_i$.
- Multiple samples are obtained through supersampling (high-resolution rendering followed by downsampling).
- Temporal Anti-Aliasing (TAA) can be combined to accumulate and denoise samples across frames.

## Key Experimental Results

### Main Results: Rendering Quality (MipNeRF-360)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3DGS (alpha blending) | 28.99 | 0.869 | 0.185 |
| StopThePop | 28.79 | 0.870 | 0.181 |
| Ours (1 SPP) | 17.95 | 0.285 | 0.611 |
| Ours (4 SPP) | 22.81 | 0.512 | 0.493 |
| Ours (16 SPP) | 26.25 | 0.714 | 0.351 |
| Ours (64 SPP) | 27.93 | 0.819 | 0.235 |
| Ours (256 SPP) | 28.50 | 0.856 | 0.178 |
| Ours (1024 SPP) | 28.66 | 0.867 | 0.168 |

Quality progressively approaches alpha blending as SPP increases. At 1024 SPP, LPIPS even surpasses the original 3DGS.

### Ablation Study: Rendering Speed (ms)

| Method | T1000 | RTX3090 | RTX4090 |
|------|-------|---------|---------|
| 3DGS-CUDA | 65.08 | 8.14 | 5.60 |
| 3DGS-OpenGL | 100.28 | 32.15 | 20.70 |
| StopThePop | 75.91 | 9.48 | - |
| **Ours (1 SPP)** | **16.23** | **3.25** | **1.85** |
| Ours (2 SPP) | 18.95 | 4.18 | 2.05 |
| Ours (4 SPP) | 24.71 | 6.42 | 2.86 |
| Ours (8 SPP) | 37.51 | 15.31 | 6.71 |
| Ours (16 SPP) | 61.25 | 18.48 | 8.00 |

At 1 SPP, the method achieves 1.85 ms on an RTX 4090, **3.0×** faster than 3DGS-CUDA, and **4.0×** faster on the low-end T1000.

### Semantic Localization Application (LERF Dataset)

| Method | Teatime | Ramen | Waldo-Kitchen | Figurines | Mean Accuracy |
|------|---------|-------|---------------|-----------|------------|
| Alpha Blending | 88.1 | 73.2 | 95.5 | 80.4 | 84.3 |
| Ours (1 SPP) | 88.1 | 73.2 | 86.4 | 80.4 | 82.0 |

Even noisy 1 SPP renderings effectively support downstream perception tasks, with only a 2.3% drop in localization accuracy.

### Key Findings

1. **Lower resolution ≠ faster rendering**: In 3DGS, reducing resolution increases the number of Gaussians per pixel/tile, slowing rendering; the sample count in StochasticSplats is the correct quality–speed knob.
2. **TAA is effective**: Temporal Anti-Aliasing substantially reduces rendering noise at 1 SPP with negligible additional latency.
3. **i.i.d. noise vs. structured artifacts**: Uniform noise is perceptually less disruptive than structured artifacts such as popping.
4. **Hardware acceleration is critical**: The 1 SPP OpenGL implementation leverages hardware features including early-out and Z-buffering.
5. **Gradient estimation is valid**: The decorrelated gradient estimator accurately approximates alpha blending gradients at 128 SPP.

## Highlights & Insights

1. **Elegant perspective shift**: A well-established computer graphics technique (Stochastic Transparency) is gracefully introduced into radiance field rendering, bridging two communities.
2. **Quality–speed continuum**: SPP provides a smooth transition from rapid preview (1 SPP) to high-quality rendering (1024 SPP), analogous to physically based rendering workflows.
3. **Portability**: The OpenGL implementation is natively cross-platform and cross-hardware, with significant implications for VR/AR and mobile deployment.
4. **General gradient estimator**: The backpropagation derivation is not specific to 3DGS and can be applied to any semi-transparent representation.
5. **Correct direction for volumetric blending**: Free-path sampling naturally handles volumetric blending of overlapping Gaussians without ray tracing.

## Limitations & Future Work

1. **Visible noise at low SPP**: PSNR is only 17.95 at 1 SPP, which, while not affecting perception, degrades quantitative quality metrics.
2. **Gradient bias**: For non-L2 loss functions, the decorrelated estimator reduces but does not eliminate gradient bias.
3. **Fine-tuning rather than training from scratch**: Results are demonstrated only through 1,000-step fine-tuning of pre-trained 3DGS; convergence from scratch remains unverified.
4. **Per-frame variance**: Multiple frames combined with TAA or higher sample counts are needed to match the rendering quality of alpha blending.
5. **Full volumetric blending is slower**: Complete volumetric blending requires disabling hardware early-out, reducing efficiency.

## Related Work & Insights

- **Stochastic Transparency (Enderton 2010)**: A mature order-independent transparency method in real-time graphics; the direct inspiration for this work.
- **StopThePop**: Resolves popping via per-pixel sorting, but incurs large computational overhead and is difficult to port.
- **EVER**: Achieves accurate volumetric blending using constant-density ellipsoid ray tracing, but is far slower than rasterization.
- **Path Replay Backpropagation (Vicini 2021)**: A paradigm for gradient computation in differentiable Monte Carlo rendering.
- **Insight**: Combining mature graphics techniques with emerging representations often yields unexpectedly elegant solutions.

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical derivation is rigorous, and the work elegantly transplants a classical graphics technique into 3DGS. The portability afforded by the OpenGL implementation and the continuous quality–speed trade-off have practical value. Weaknesses include a substantial quality gap at low SPP and training results limited to fine-tuning. Overall, this is a valuable contribution to the 3DGS rendering efficiency literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)

</div>

<!-- RELATED:END -->
