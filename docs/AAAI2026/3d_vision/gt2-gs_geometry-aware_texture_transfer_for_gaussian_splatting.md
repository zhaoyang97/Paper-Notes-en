---
title: >-
  [Paper Note] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This paper proposes GT2-GS, a framework that achieves high-quality, view-consistent texture transfer for 3DGS via a geometry-aware texture transfer loss (GT2 Loss)…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "texture transfer"
  - "geometry-aware"
  - "style transfer"
  - "3DGS appearance editing"
date: 2026-05-08
content_hash: 3acc554ae561c8db
---

# GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting

**Conference**: AAAI 2026
**arXiv**: [2505.15208](https://arxiv.org/abs/2505.15208)  
**Code**: [https://vpx-ecnu.github.io/GT2-GS-website](https://vpx-ecnu.github.io/GT2-GS-website)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, texture transfer, geometry-aware, style transfer, 3DGS appearance editing

## TL;DR

This paper proposes GT2-GS, a framework that achieves high-quality, view-consistent texture transfer for 3DGS via a geometry-aware texture transfer loss (GT2 Loss), an adaptive fine-grained control module (AFCM), and a geometry-preserving branch (GPB), outperforming existing 3D style transfer methods in both texture fidelity and scene content preservation.

## Background & Motivation

3D style transfer aims to transfer style elements from 2D reference images onto 3D scenes, with broad demand in virtual reality, gaming, and related domains. Existing methods (e.g., ARF, ABC-GS, SGSST) primarily focus on abstract artistic style transfer but perform poorly on **texture** transfer. The authors analyze three core issues from the perspective of the optimization process:

**Lack of geometric consistency**: Existing methods rely on the NNFM loss, where optimization targets are constructed independently per view, ignoring the rich geometric structure within the scene and cross-view geometric consistency. Texture and geometry are intrinsically coupled — the same texture region exhibits different texture orientations (e.g., scaling and rotation) across viewpoints, yet the NNFM loss is entirely geometry-agnostic.

**Granularity mismatch between features and pixels**: After multiple convolutional layers, VGG feature maps have spatial resolutions far lower than the original image pixels. In regions with high pixel information density (e.g., distant regions, fine structures such as stair railings), coarse-grained texture feature learning overwrites and destroys these important details.

**Coupling of Gaussian geometry and color parameters**: In 3DGS, geometry and color parameters are jointly encoded. During texture transfer, the lack of ground-truth supervision means that the densification strategy may introduce erroneous floating Gaussians, which depth regularization alone cannot resolve.

## Method

### Overall Architecture

GT2-GS takes as input scene Gaussians, a content image, and a texture reference image. The framework consists of three core components: (1) a Geometry-aware Texture Transfer Loss (GT2 Loss); (2) an Adaptive Fine-grained Control Module (AFCM); and (3) a Geometry-Preserving Branch (GPB). An additional color parameter $c^g$ is bound to each Gaussian to decouple appearance optimization from geometry optimization.

### Key Designs

#### 1. **Geometry-aware Texture Transfer Loss (GT2 Loss)**

The core idea of GT2 Loss is to incorporate geometric information into the texture feature matching process to achieve view-consistent texture transfer.

**Texture feature set construction**: The scene depth map is used to sort and discretize depth values into $K$ groups (default $K=4$), with a scaling factor of $Z_1/Z_k$ computed for each group. VGG features are extracted from correspondingly scaled and rotated versions of the texture image, forming the feature set $\{f_{k,\theta}\}$, where $k$ denotes the scaling parameter and $\theta$ the rotation angle.

**Cross-view geometric prior**: When constructing the target feature map $F_t^v$ for the current view, the feature map $F_t^{v-1}$ from the previous view is used as a prior. Cross-view correspondences are established via the homography matrix $M_p^{v,v-1} = K_{v-1}[R_{v-1}|T_{v-1}][R_v|T_v]^{-1}K_v^{-1}$.

**Viewpoint transformation awareness**: The orientation of the same texture region varies across viewpoints. The pixel set $\{p_v\}$ and its corresponding set in the previous view $\{p_{v-1}\}$ are obtained via upsampling, a linear transformation matrix $M_L$ is estimated by least squares, and the rotation angle $\beta$ is extracted via SVD decomposition. The target feature map is constructed as:

$$F_t(i,j) = \arg\min_{f_{k,\theta}} dist(F_r(i,j), f_{k,\theta}) + \lambda_p |\theta' + \beta - \theta|$$

The final GT2 Loss is the cosine distance between the rendered feature map and the target feature map:

$$L_{gt} = \frac{1}{N}\sum_{i,j} dist(F_r^v(i,j), F_t^v(i,j))$$

#### 2. **Adaptive Fine-grained Control Module (AFCM)**

AFCM addresses the granularity mismatch between VGG features and pixel space. It adaptively adjusts texture learning intensity using three information sources:

- **Depth map $I_d$**: Regions at greater depth concentrate more scene information and require reduced texture learning intensity.
- **Frequency density map $I_f$**: Extracted from the content image; high-frequency regions (e.g., stairs, railings) must be preserved.
- **Geometric distortion map $\Phi$**: The angular discrepancy between texture features obtained with and without the geometric prior.

The adaptive weight matrix is:

$$W^v = \lambda_d(1-I_d^v) + \lambda_f(1-I_f^v) + \lambda_\Phi(1-\Phi)$$

The weighted GT2 Loss is: $L_{wgt} = \frac{1}{N}\sum_{i,j} W^v(i,j) \cdot dist(F_r^v(i,j), F_t^v(i,j))$

The total loss is: $L_{tot} = \lambda_{wgt}L_{wgt} + \lambda_c L_{content} + \lambda_{tv}L_{tv}$

#### 3. **Geometry-Preserving Branch (GPB)**

GPB addresses geometric degradation caused by the coupling of geometry and color parameters in 3DGS. The core insight is to introduce an additional geometry optimization objective to balance appearance optimization with geometric integrity.

Specifically, an additional color parameter $c^g$ (initialized with the original color) is bound to each Gaussian. An image $I_g$ is rendered using $c^g$, and a 3DGS reconstruction loss is optimized with the content image $I_c$ as ground truth:

$$\mathcal{L}_{rec} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$$

Through ground-truth-supervised optimization, Gaussians are relocated to geometrically correct positions.

### Loss & Training

- A view-consistent color transfer is applied prior to texture transfer.
- Features are extracted from the conv3 block of VGG-16.
- Depth grouping uses $K=4$; rotation angles $\theta$ span 360°.
- AFCM weights: $\{\lambda_d, \lambda_f, \lambda_\Phi\} = \{0.8, 0.8, 0.25\}$.
- Texture transfer optimization weights: $\{\lambda_{wgt}, \lambda_c, \lambda_{tv}\} = \{2, 0.005, 0.02\}$.
- Hardware: single NVIDIA RTX 4090.

## Key Experimental Results

### Main Results

Quantitative evaluation on 100 scene–reference image pairs (multi-view consistency + content preservation):

| Method | SSIM↑ | CLIP-score↑ | ST-LPIPS↓ | ST-RMSE↓ | LT-LPIPS↓ | LT-RMSE↓ |
|--------|-------|-------------|-----------|----------|-----------|----------|
| **GT2-GS (Ours)** | **0.51** | **0.47** | 0.054 | 0.048 | 0.087 | 0.077 |
| SGSST | 0.45 | 0.44 | 0.075 | 0.072 | 0.119 | 0.108 |
| ABC-GS | 0.56 | 0.46 | **0.049** | **0.041** | **0.080** | **0.068** |
| StyleGaussian | 0.41 | 0.40 | 0.058 | 0.052 | 0.097 | 0.082 |
| ARF | 0.37 | 0.45 | 0.109 | 0.072 | 0.152 | 0.108 |
| Ref-NPR | 0.35 | 0.42 | 0.092 | 0.069 | 0.137 | 0.102 |
| SNeRF | 0.48 | 0.36 | 0.075 | 0.057 | 0.127 | 0.090 |

GT2-GS leads significantly on SSIM and CLIP-score, indicating that the transferred results preserve semantic content while achieving high-quality texture transfer. ABC-GS performs better on multi-view consistency metrics, but achieves this by disabling the densification strategy; GT2-GS maintains multi-view consistency while keeping densification enabled.

### Ablation Study

Experiments on 25 randomly selected LLFF scenes:

| Configuration | SSIM↑ | CLIP-score↑ | Notes |
|---------------|-------|-------------|-------|
| Full model | 0.41 | 0.39 | Complete model |
| w/o GT2 Loss | 0.38 | 0.36 | Obvious texture discontinuity and blurring |
| w/o AFCM | 0.45 | 0.38 | Foreground low-texture regions fail to capture style |
| w/o GPB | 0.31 | 0.37 | Significant artifacts appear in the scene |

### Key Findings

- Removing GT2 Loss substantially increases texture discontinuity and blurring, confirming the critical role of geometric information in texture transfer.
- Removing AFCM causes foreground low-texture regions to fail at learning texture patterns; in 360° scenes (e.g., truck), geometric fidelity degrades severely in regions with large depth variation.
- Removing GPB causes the largest drop in SSIM (0.41→0.31) and introduces conspicuous artifacts, demonstrating that geometry preservation is essential for content fidelity.
- Simply adding depth regularization cannot replace GPB, particularly when the number of Gaussians increases.

## Highlights & Insights

1. **Texture ≠ Style**: This paper is the first to systematically distinguish texture transfer from artistic style transfer, highlighting the intrinsic coupling between texture and geometry.
2. **Cross-view geometric prior**: Homography matrices and SVD decomposition are used to elegantly handle orientation changes of texture across viewpoints.
3. **Additive design of AFCM**: Depth and frequency information jointly satisfy the requirements of shallow depth and high frequency via additive rather than multiplicative fusion.
4. **Decoupling rationale of GPB**: Introducing an additional color parameter to decouple appearance and geometry optimization is more effective than depth regularization.

## Limitations & Future Work

- Because the method simultaneously minimizes texture cosine distance and content loss, the resulting texture represents an interpolation between the scene geometry and the reference texture geometry.
- Scalability to high-resolution scenes has not been explored.
- The computational overhead of constructing the texture feature set (involving multiple scaling and rotation combinations) may affect efficiency in large-scale scenes.

## Related Work & Insights

- Building on ARF (ECCV 2022), which first applied NNFM loss to 3D style transfer, this work further incorporates geometric consistency.
- ABC-GS preserves geometry by disabling densification, whereas GPB allows geometry preservation while keeping densification enabled.
- StyleGaussian's zero-shot approach is fast but insufficient in texture transfer quality.
- Inspiration: geometry-aware approaches can be generalized to other 3DGS editing tasks, such as relighting and material editing.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic introduction of geometric information into 3DGS texture transfer.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Sufficient qualitative and quantitative evaluation with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, though some formulations are relatively complex.
- Value: ⭐⭐⭐⭐ — Makes a tangible contribution to 3DGS appearance editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting](../../CVPR2026/3d_vision/fact-gs_frequency-aligned_complexity-aware_texture_reparameterization_for_2d_gau.md)
- [\[AAAI 2026\] TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction](tg-field_geometry-aware_radiative_gaussian_fields_for_tomographic_reconstruction.md)
- [\[AAAI 2026\] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[CVPR 2026\] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment](../../CVPR2026/3d_vision/cross-instance_gaussian_splatting_registration_via_geometry-aware_feature-guided.md)

</div>

<!-- RELATED:END -->
