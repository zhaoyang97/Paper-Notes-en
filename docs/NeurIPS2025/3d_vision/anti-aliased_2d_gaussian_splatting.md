---
title: >-
  [Paper Note] Anti-Aliased 2D Gaussian Splatting
description: >-
  [NeurIPS 2025][3D Vision][2D Gaussian Splatting] This paper proposes AA-2DGS, which addresses severe aliasing artifacts in 2D Gaussian Splatting under varying sampling rates through two complementary mechanisms: a world-…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "2D Gaussian Splatting"
  - "Anti-Aliasing"
  - "Novel View Synthesis"
  - "Surface Reconstruction"
  - "Mip Filtering"
date: 2026-05-08
content_hash: bf317825e46032f6
---

# Anti-Aliased 2D Gaussian Splatting

**Conference**: NeurIPS 2025
**arXiv**: [2506.11252](https://arxiv.org/abs/2506.11252)  
**Authors**: Mae Younes, Adnane Boukhayma (INRIA France, University of Rennes, CNRS, IRISA)
**Code**: [AA-2DGS](https://github.com/maeyounes/AA-2DGS)  
**Area**: 3D Vision
**Keywords**: 2D Gaussian Splatting, Anti-Aliasing, Novel View Synthesis, Surface Reconstruction, Mip Filtering

## TL;DR

This paper proposes AA-2DGS, which addresses severe aliasing artifacts in 2D Gaussian Splatting under varying sampling rates through two complementary mechanisms: a world-space flat smoothing kernel and an object-space Mip filter. The method significantly improves multi-scale rendering quality while preserving the geometric accuracy advantages of 2DGS.

## Background & Motivation

**Geometric advantages of 2DGS**: 2D Gaussian Splatting represents scenes using planar Gaussian disks embedded in 3D space rather than the volumetric Gaussians of 3DGS. By evaluating Gaussian values directly at ray-splat intersections, it achieves superior depth and normal reconstruction accuracy, making it suitable for tasks requiring precise geometry such as mesh recovery, physically-based rendering, and reflection modeling.

**Severe aliasing problem**: 2DGS produces severe aliasing artifacts when the rendering resolution differs from the training resolution (e.g., due to camera zoom or field-of-view changes), greatly limiting its practical applicability.

**Harmful clamping in prior work**: The original 2DGS implementation adopts screen-space clamping (lower-bound constraints) to handle degenerate cases, but the authors find that this approach actually **exacerbates** aliasing artifacts rather than mitigating them. Clamping introduces gradient discontinuities, CUDA warp thread divergence, and cross-domain distance comparisons.

**Non-transferability of Mip-Splatting**: Although Mip-Splatting has resolved anti-aliasing for 3DGS (via 3D smoothing filters and screen-space Mip filters), these solutions cannot be directly applied to 2DGS due to its fundamentally different planar primitive representation and rendering pipeline.

**Sampling theory perspective**: Aliasing is fundamentally a violation of the Nyquist-Shannon sampling theorem—high-frequency signals are incorrectly mapped to low frequencies at insufficient sampling rates—requiring low-pass filtering prior to sampling.

**Dual sources of aliasing**: Mip-Splatting identifies two sources of aliasing: (a) the representation itself lacks 3D frequency constraints (causing high-frequency artifacts when zooming in), and (b) insufficient screen-space filtering (causing aliasing when zooming out). AA-2DGS must address both issues within the 2DGS framework.

## Method

### Overall Architecture

AA-2DGS introduces two complementary anti-aliasing mechanisms on top of 2DGS:

- **World-Space Flat Smoothing Kernel**: Constrains the maximum frequency of 2D Gaussian primitives to prevent high-frequency artifacts during zoom-in.
- **Object-Space Mip Filter**: Leverages an affine approximation of the ray-splat intersection mapping to perform screen-space anti-aliasing in the splat's local space, preventing aliasing during zoom-out.

Together, these two mechanisms cover aliasing in both magnification and minification scenarios.

### Key Design 1: World-Space Flat Smoothing Kernel

**Objective**: Constrain the maximum frequency of the representation based on the Nyquist limit of the training views.

**Multi-view frequency estimation**: For each Gaussian primitive $k$, the maximum sampling rate across all training views is computed as:

$$\hat{\nu}_k = \max_{n=1\dots N} \left\{ \mathbb{1}_n(\mathbf{p}_k) \cdot \frac{f_n}{d_n} \right\}$$

where $f_n$ is the focal length in pixels, $d_n$ is the depth, and $\mathbb{1}_n$ indicates whether the primitive center lies within the frustum of the $n$-th camera.

**Flat projection**: The 3D isotropic smoothing kernel from Mip-Splatting is projected onto the plane of the 2D Gaussian, yielding a 2D smoothing filter with equal variance. After convolution with the intrinsic 2D Gaussian of the primitive, the effective covariance becomes:

$$\mathbf{V}_k^{\text{eff}} = \mathbf{V}_k + \sigma_{\text{smooth},k}^2 \mathbf{I}_2$$

Opacity is modulated accordingly to preserve energy conservation: $\alpha_k^{\text{smooth}} = \alpha_k \frac{s_u s_v}{\sqrt{s_u^2 + \sigma^2} \cdot \sqrt{s_v^2 + \sigma^2}}$.

Sampling rates are computed during training and updated every 100 iterations, then fixed at test time.

### Key Design 2: Object-Space Mip Filter

**Core Idea**: Using a first-order Taylor expansion (affine approximation) of the ray-splat intersection mapping $m: \mathbf{x} \to \mathbf{u}$, the screen-space Mip filter is mapped into the splat's local space.

**Derivation**:
1. The affine approximation $m(\mathbf{x}) \approx \mathbf{u}_0 + \mathbf{J}(\mathbf{x} - \mathbf{x}_0)$ maps the local-coordinate Gaussian to screen space.
2. Convolution with the Mip filter (variance $\sigma\mathbf{I}$) is performed in screen space.
3. The Gaussian affine transformation property is applied to map back to object space, yielding the Mip-filtered Gaussian in local space:

$$\mathcal{G}_{\text{obj-mip},k}(\mathbf{x}) = \sqrt{\frac{1}{|\boldsymbol{\Sigma}'_{\text{local},k}|}} \exp\left(-\frac{1}{2}\mathbf{u}_k^\top (\boldsymbol{\Sigma}'_{\text{local},k})^{-1} \mathbf{u}_k\right)$$

where $\boldsymbol{\Sigma}'_{\text{local},k} = \mathbf{I} + \sigma \mathbf{J}\mathbf{J}^\top$.

**Distinction from classical EWA**: Classical object-space EWA applies the affine approximation at the primitive center, whereas AA-2DGS computes the approximation at each pixel, yielding greater accuracy for large primitives and extreme viewing angles.

### Key Design 3: Custom CUDA Implementation

Both forward and backward passes of the Mip filter are implemented via custom CUDA kernels, increasing rendering time by 15–30% compared to the original 2DGS. The Jacobian $\mathbf{J}$ is computed analytically from the ray-splat intersection formula (Eq. 6).

### Loss & Training

The loss function follows the design of 2DGS. Depth and normal regularization are disabled for novel view synthesis experiments and enabled for surface reconstruction experiments. Training runs for 30K iterations; Gaussian densification strategies and hyperparameters follow Mip-Splatting. The Mip filter variance is set to 0.1 (approximating one pixel), and the flat smoothing filter variance is set to 0.2.

## Key Experimental Results

### Blender Dataset — Multi-Scale Training + Multi-Scale Testing

| Method | Full PSNR | 1/2 PSNR | 1/4 PSNR | 1/8 PSNR | Avg PSNR |
|------|-----------|----------|----------|----------|----------|
| 2DGS | 28.58 | 30.24 | 31.42 | 27.35 | 29.40 |
| 2DGS w/o Clamping | 31.64 | 33.33 | 31.61 | 27.62 | 31.05 |
| Mip-Splatting | 32.81 | 34.49 | 35.45 | 35.50 | 34.56 |
| **AA-2DGS** | **32.68** | **34.53** | **35.65** | **35.53** | **34.60** |

### Blender Dataset — Single-Scale Training + Multi-Scale Testing (Zoom-Out)

| Method | Full PSNR | 1/2 PSNR | 1/4 PSNR | 1/8 PSNR | Avg PSNR |
|------|-----------|----------|----------|----------|----------|
| 2DGS | 33.05 | 27.64 | 20.61 | 16.59 | 24.47 |
| Mip-Splatting | 33.36 | 34.00 | 31.85 | 28.67 | 31.97 |
| **AA-2DGS** | **33.24** | **34.10** | **32.11** | **29.00** | **32.11** |

### Mip-NeRF 360 Dataset — Single-Scale Training + Multi-Scale Testing (Zoom-In)

| Method | 1× PSNR | 2× PSNR | 4× PSNR | 8× PSNR | Avg PSNR |
|------|---------|---------|---------|---------|----------|
| 2DGS | 28.82 | 24.97 | 23.79 | 23.55 | 25.28 |
| Mip-Splatting | 29.39 | 27.39 | 26.47 | 26.22 | 27.37 |
| **AA-2DGS** | **29.30** | **27.16** | **26.10** | **25.77** | **27.08** |

### DTU Surface Reconstruction (Chamfer Distance ↓)

| Method | Mean Chamfer |
|------|-------------|
| 2DGS | 0.80 |
| 2DGS* (retrained) | 0.76 |
| 2DGS w/o Clamping | 0.75 |
| **AA-2DGS** | **0.74** |

### Key Findings

1. **Clamping is harmful**: Removing clamping improves 2DGS performance (Blender multi-scale avg PSNR: 29.40→31.05), revealing that the original clamping design is counterproductive.
2. **AA-2DGS surpasses Mip-Splatting**: On multi-scale Blender training and testing, avg PSNR is 34.60 vs. 34.56; in zoom-out settings, 32.11 vs. 31.97.
3. **Geometric accuracy preserved**: Chamfer distance of 0.74 on DTU, outperforming the original 2DGS (0.80), demonstrating that anti-aliasing does not compromise geometry.
4. **Slight degradation at same scale**: PSNR on MipNeRF360 at the same scale is 27.38 vs. 2DGS's 27.56, as band-limiting constraints attenuate some high frequencies—a fundamental trade-off between anti-aliasing and sharpness.

## Highlights & Insights

1. **Theoretical elegance**: The adaptation of Mip-Splatting's 3D framework to 2DGS's planar primitives is accomplished through mathematically clean derivations of the flat projection and object-space Mip filter.
2. **Fundamental insight**: The paper reveals a critical flaw in 2DGS's original clamping mechanism—rather than providing anti-aliasing, it introduces additional artifacts.
3. **Per-pixel affine approximation**: Unlike classical EWA, which applies the affine approximation only at the primitive center, AA-2DGS computes the Jacobian at each pixel, yielding greater accuracy for large primitives.
4. **Anti-aliasing extended to normals**: Anti-aliasing benefits extend beyond RGB rendering to geometric attributes such as normals, which is valuable for surface reconstruction and reflection modeling.
5. **First anti-aliasing work for 2DGS**: This work fills an important gap in the 2DGS research landscape.

## Limitations & Future Work

1. **"Needle-like" artifacts**: Planar 2D primitives still produce needle-like artifacts under extreme magnification or grazing angles—the flat smoothing kernel mitigates but cannot fundamentally resolve the zero-thickness issue.
2. **Sharpness–anti-aliasing trade-off**: Fixed filter parameters introduce a slight loss of sharpness at the training resolution (approximately 0.2 dB PSNR reduction on MipNeRF360 at the same scale).
3. **Rendering overhead**: The custom CUDA kernels increase rendering time by 15–30%.
4. **Issues with low-resolution training**: When trained at low resolution, 2D primitives tend to become extremely thin, producing needle-like artifacts at higher rendering resolutions—an inherent limitation of 2D primitives.
5. **Fixed filter parameters**: The variances of the smoothing kernel and Mip filter (0.2 and 0.1) are hyperparameters that may not be optimal for all scenes.

## Related Work & Insights

- **Mip-Splatting**: The direct inspiration for AA-2DGS; resolves anti-aliasing for 3DGS but is not directly applicable to 2DGS.
- **3DGS**: Volumetric Gaussian primitives with strong rendering quality but inferior geometric accuracy compared to 2DGS.
- **2DGS**: Planar Gaussian primitives with excellent geometric accuracy but lacking effective anti-aliasing.
- **EWA Splatting**: Classical elliptical weighted average filtering; AA-2DGS adapts its core ideas to the 2DGS ray-splat intersection framework.
- **MipNeRF / Tri-MipRF**: Multi-scale anti-aliasing methods in the NeRF family, achieved via cone tracing and pre-filtered positional encodings.
- **Analytic Splatting / Multi-Scale GS / HDGS**: Alternative anti-aliasing approaches for 3DGS, each with respective computational or memory trade-offs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Adapting Mip-Splatting ideas to 2DGS involves clear technical challenges; the derivations of the object-space Mip filter and flat projection are original, though the overall direction follows Mip-Splatting.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers three major benchmarks: Blender (multi-scale and single-scale), MipNeRF360 (zoom-in, zoom-out, same scale), and DTU (surface reconstruction), with thorough ablation studies and insightful clamping analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear, problem formulation is precise, and experiments are well organized.
- **Value**: ⭐⭐⭐⭐ — Fills the gap in anti-aliasing for 2DGS and has practical significance for applications requiring geometric accuracy combined with multi-scale rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](../../ICCV2025/3d_vision/aaa_gaussians_anti_aliased_artifact_free_3d_gaussian_rendering.md)
- [\[NeurIPS 2025\] IBGS: Image-Based Gaussian Splatting](ibgs_image-based_gaussian_splatting.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
