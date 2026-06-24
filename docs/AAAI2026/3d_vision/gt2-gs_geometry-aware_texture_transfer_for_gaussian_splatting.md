---
title: >-
  [Paper Note] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This paper proposes the GT2-GS framework, which achieves high-quality and view-consistent 3DGS texture transfer through a geometry-aware texture transfer loss, an adaptive fine-grained control module, and a geometry-preserving branch. It outperforms existing 3D style transfer methods in both texture fidelity and scene content preservation.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Texture Transfer"
  - "Geometry-aware"
  - "Style Transfer"
  - "3DGS Appearance Editing"
date: 2026-05-08
content_hash: 31cf7e8e91510655
---

# GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting

**Conference**: AAAI 2026  
**arXiv**: [2505.15208](https://arxiv.org/abs/2505.15208)  
**Code**: [https://vpx-ecnu.github.io/GT2-GS-website](https://vpx-ecnu.github.io/GT2-GS-website)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Texture Transfer, Geometry-aware, Style Transfer, 3DGS Appearance Editing

## TL;DR

This paper proposes the GT2-GS framework, which achieves high-quality and view-consistent 3DGS texture transfer through a geometry-aware texture transfer loss, an adaptive fine-grained control module, and a geometry-preserving branch. It outperforms existing 3D style transfer methods in both texture fidelity and scene content preservation.

## Background & Motivation

3D style transfer aims to transfer style elements from 2D reference images to 3D scenes, which is highly demanded in virtual reality, gaming, and other fields. Existing methods (such as ARF, ABC-GS, SGSST, etc.) mainly focus on abstract artistic style transfer, but perform poorly when handling **texture** transfer. The authors analyze three core problems from the perspective of the optimization process:

**Lack of geometric consistency**: Existing methods are based on the NNFM loss, where the optimization targets for each view are constructed independently, ignoring the rich internal geometry of the scene and cross-view geometric consistency. Texture and geometry are inherently related — the same texture region displays different texture directions (such as scaling and rotation) from different perspective views, but the NNFM loss is completely unaware of this.

**Mismatch between feature and pixel granularity**: After passing through multiple convolutional layers, the spatial resolution of VGG feature maps is far lower than that of the original image pixels. In areas with high pixel information density (such as distant regions, fine structures like stair handrails, etc.), coarse-grained texture feature learning tends to overwrite and destroy these important details.

**Coupling of Gaussian geometry and color parameters**: In 3DGS, geometric and color parameters are jointly encoded. Due to the lack of ground truth supervision during the texture transfer process, the densification strategy may introduce floating erroneous Gaussians, which cannot be resolved solely by depth regularization.

## Method

### Overall Architecture

The inputs to GT2-GS consist of scene Gaussians, a content image, and a texture reference image. The framework incorporates three core components: (1) Geometry-aware Texture Transfer Loss (GT2 Loss); (2) Adaptive Fine-grained Control Module (AFCM); and (3) Geometry-preserving Branch (GPB). By additionally binding a color parameter $c^g$ to the Gaussians, decoupling of appearance and geometric optimization is achieved.

### Key Designs

#### 1. **Geometry-aware Texture Transfer Loss (GT2 Loss)**

The core idea of GT2 Loss is to incorporate geometric information into the texture feature matching process to achieve view-consistent texture transfer.

**Texture Feature Set Construction**: First, using the scene depth map, depth values are sorted and discretized into $K$ groups (default $K=4$), and the scaling factor for each group is calculated as $Z_1/Z_k$. The corresponding scaling and rotation operations are applied to the texture image, followed by VGG feature extraction to form the feature set $\{f_{k,\theta}\}$, where $k$ represents the scaling parameter and $\theta$ represents the rotation angle.

**Cross-view Geometric Prior**: When constructing the target feature map $F_t^v$ of the current view, the feature map of the previous view $F_t^{v-1}$ is used as a prior. The cross-view correspondence is established using the homography matrix $M_p^{v,v-1} = K_{v-1}[R_{v-1}|T_{v-1}][R_v|T_v]^{-1}K_v^{-1}$.

**Viewpoint Transformation Awareness**: The orientation of the same texture region varies under different viewpoints. Up-sampling is performed to obtain the pixel set $\{p_v\}$ and its corresponding pixel set from the previous view $\{p_{v-1}\}$. The linear transformation matrix $M_L$ is calculated via least squares, and the rotation angle $\beta$ is extracted through SVD decomposition. The target feature map is constructed as follows:

$$F_t(i,j) = \arg\min_{f_{k,\theta}} dist(F_r(i,j), f_{k,\theta}) + \lambda_p |\theta' + \beta - \theta|$$

The final GT2 Loss is the cosine distance between the rendered feature map and the target feature map:

$$L_{gt} = \frac{1}{N}\sum_{i,j} dist(F_r^v(i,j), F_t^v(i,j))$$

#### 2. **Adaptive Fine-grained Control Module (AFCM)**

AFCM addresses the granularity mismatch between VGG features and pixel space. It adaptively adjusts the intensity of texture learning using three information sources:

- **Depth Map $I_d$**: Regions that are further away aggregate more scene information, necessitating a reduction in the intensity of texture learning.
- **Frequency Density Map $I_f$**: Extracted from the content image; high-frequency regions (such as stairs, handrails, and other fine structures) need to be protected.
- **Geometric Distortion Map $\Phi$**: The angular difference between texture features obtained with and without prior information.

The adaptive weight matrix is formulated as:

$$W^v = \lambda_d(1-I_d^v) + \lambda_f(1-I_f^v) + \lambda_\Phi(1-\Phi)$$

The weighted GT2 Loss is: $L_{wgt} = \frac{1}{N}\sum_{i,j} W^v(i,j) \cdot dist(F_r^v(i,j), F_t^v(i,j))$

The total loss is: $L_{tot} = \lambda_{wgt}L_{wgt} + \lambda_c L_{content} + \lambda_{tv}L_{tv}$

#### 3. **Geometry-preserving Branch (GPB)**

GPB addresses the geometric degradation issue caused by the coupling of geometry and color parameters in 3DGS. The key insight is to introduce an additional geometric optimization target to balance appearance optimization and geometric integrity.

Specifically: each Gaussian is bound with an additional color parameter $c^g$ (initialized with the original color). Image $I_g$ is rendered using $c^g$, and the 3DGS reconstruction loss is optimized by taking the content image $I_c$ as the ground truth:

$$\mathcal{L}_{rec} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$$

Through optimization with the ground truth, Gaussians are moved to the correct geometric positions.

### Loss & Training

- Perform view-consistent color transfer prior to texture transfer.
- Extract features using the conv3 block of VGG-16.
- Set depth grouping to $K=4$, with rotation angle $\theta$ covering 360 degrees.
- AFCM weights: $\{\lambda_d, \lambda_f, \lambda_\Phi\} = \{0.8, 0.8, 0.25\}$.
- Texture transfer optimization weights: $\{\lambda_{wgt}, \lambda_c, \lambda_{tv}\} = \{2, 0.005, 0.02\}$.
- Trained on a single NVIDIA RTX 4090 GPU.

## Key Experimental Results

### Main Results

Quantitative evaluation on 100 scene-reference image pairs (assessing multi-view consistency and content preservation):

| Method | SSIM↑ | CLIP-score↑ | ST-LPIPS↓ | ST-RMSE↓ | LT-LPIPS↓ | LT-RMSE↓ |
|------|-------|-------------|-----------|----------|-----------|----------|
| **GT2-GS (Ours)** | **0.51** | **0.47** | 0.054 | 0.048 | 0.087 | 0.077 |
| SGSST | 0.45 | 0.44 | 0.075 | 0.072 | 0.119 | 0.108 |
| ABC-GS | 0.56 | 0.46 | **0.049** | **0.041** | **0.080** | **0.068** |
| StyleGaussian | 0.41 | 0.40 | 0.058 | 0.052 | 0.097 | 0.082 |
| ARF | 0.37 | 0.45 | 0.109 | 0.072 | 0.152 | 0.108 |
| Ref-NPR | 0.35 | 0.42 | 0.092 | 0.069 | 0.137 | 0.102 |
| SNeRF | 0.48 | 0.36 | 0.075 | 0.057 | 0.127 | 0.090 |

GT2-GS significantly leads in both SSIM and CLIP-score, indicating that the texture transfer results achieve high-quality texture transfer while preserving semantic content. ABC-GS performs well on multi-view consistency metrics, but it disables the densification strategy, whereas GT2-GS maintains multi-view consistency even when the densification strategy is enabled.

### Ablation Study

Ablation study on 25 random LLFF scenes:

| Configuration | SSIM↑ | CLIP-score↑ | Description |
|------|-------|-------------|------|
| Full model | 0.41 | 0.39 | Full model |
| w/o GT2 Loss | 0.38 | 0.36 | Texture shows obvious discontinuity and blurriness |
| w/o AFCM | 0.45 | 0.38 | Foreground regions with low texture fail to capture style |
| w/o GPB | 0.31 | 0.37 | Obvious artifacts appear in the scene |

### Key Findings

- Removing GT2 Loss leads to significantly aggravated texture discontinuity and blurriness, confirming the key role of geometric information in texture transfer.
- Removing AFCM prevents foreground low-texture regions from learning texture patterns; 360° scenes (e.g., truck) exhibit a severe drop in geometric fidelity in regions with large depth variations.
- Removing GPB causes the most severe drop in SSIM (0.41 to 0.31) and introduces obvious artifacts into the scene, showing that geometry preservation is crucial for content fidelity.
- Simply adding depth regularization cannot resolve the issues addressed by GPB, especially as the number of Gaussians increases.

## Highlights & Insights

1. **Texture $\neq$ Style**: This work is the first to systematically distinguish texture transfer from artistic style transfer, highlighting the inherent correlation between texture and geometry.
2. **Cross-view Geometric Prior**: Viewpoint-dependent orientation variations of texture under different perspectives are elegantly handled via the homography matrix and SVD decomposition.
3. **Additive Design of AFCM**: Depth and frequency information simultaneously satisfy the requirement of shallow depth + high frequency, utilizing additive instead of multiplicative fusion.
4. **Decoupled Strategy of GPB**: Achieving decoupling of appearance and geometric optimization using additional color parameters is more effective than depth regularization.

## Limitations & Future Work

- Since it concurrently minimizes the texture cosine distance and preserves content loss, the resulting texture is an interpolation between the scene geometry and the reference texture geometry.
- Scalability to ultra-high-resolution scenes has not yet been explored.
- Computational overhead (building the texture feature set involves multiple scaling and rotation combinations) might limit efficiency on large-scale scenes.

## Related Work & Insights

- Compared to ARF (ECCV 2022), which first utilized NNFM loss for 3D style transfer, this paper further incorporates geometric consistency.
- ABC-GS disables the densification strategy to maintain geometry, whereas GPB allows maintaining geometry while enabling densification.
- Although the zero-shot approach of StyleGaussian is fast, its texture transfer quality remains insufficient.
- Insight: Geometry-aware methods can be extended to other 3DGS editing tasks (such as relighting, material editing, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ — First to systematically introduce geometric information into 3DGS texture transfer.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Sufficient qualitative and quantitative evaluations, with a complete ablation study.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, though some formula descriptions are relatively complex.
- Value: ⭐⭐⭐⭐ — Practically promotes the advancement of 3DGS appearance editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](../../ECCV2024/3d_vision/texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [\[AAAI 2026\] TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction](tg-field_geometry-aware_radiative_gaussian_fields_for_tomographic_reconstruction.md)
- [\[AAAI 2026\] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
