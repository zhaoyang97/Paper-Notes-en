---
title: >-
  [Paper Note] "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting"
description: >-
  [CVPR2025][3D Vision][Paper Note] Academic paper note for "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting".
tags:
  - CVPR2025
  - 3D Vision
date: 2024-12-17
content_hash: 02e01bffb558ba30
---
# 3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting

## Background & Motivation

3D Gaussian Splatting (3DGS) has achieved remarkable success in novel view synthesis, but its core EWA splatting pipeline has fundamental limitations. EWA splatting requires projecting 3D Gaussians onto a 2D plane, a process that relies on **affine approximation** (first-order Taylor expansion), which assumes the projection is locally linear within the Gaussian's support. This assumption is severely violated in the following scenarios:

**Distorted Camera Models**: Non-pinhole camera models such as fisheye lenses and rolling shutters introduce non-linear distortions, where the affine approximation leads to significant artifacts.

**Secondary Rays**: Effects like reflection and refraction require tracing secondary rays. The directions of these rays differ from the original projection direction, making them impossible to describe with a single affine transformation.

**Wide-angle Lenses**: When the field of view exceeds 120°, perspective distortion in the peripheral areas causes the linear approximation error to increase drastically.

Existing methods attempt to address these issues through post-processing or scene-specific patching, but they lack a unified mathematical framework. This paper proposes using the **Unscented Transform (UT)** to replace the affine approximation in EWA, fundamentally resolving this limitation.

## Method

### Unscented Transform Principle

The Unscented Transform (UT) is a technique for propagating probability distributions through non-linear transformations. Its core idea is to capture the statistical properties of a distribution using a set of carefully selected **sigma points**, rather than linearizing the transformation function itself.

For an $n$-dimensional Gaussian distribution $\mathcal{N}(oldsymbol{\mu}, oldsymbol{\Sigma})$, UT selects $2n+1$ sigma points:

$$oldsymbol{\chi}_0 = oldsymbol{\mu}, \quad oldsymbol{\chi}_i = oldsymbol{\mu} + \sqrt{(n+\lambda)} \cdot oldsymbol{L}_i, \quad oldsymbol{\chi}_{n+i} = oldsymbol{\mu} - \sqrt{(n+\lambda)} \cdot oldsymbol{L}_i$$

where $oldsymbol{L}$ is the Cholesky decomposition of $oldsymbol{\Sigma}$, and $\lambda = lpha^2(n+\kappa) - n$ is the scaling parameter.

### 3DGUT Splatting Pipeline

This paper integrates UT into the splatting pipeline of 3DGS, forming **3DGUT**:

| Step | EWA Splatting | 3DGUT (Ours) |
|------|--------------|-------------|
| Projection Method | Affine Approximation (Jacobian Matrix) | Unscented Transform (sigma points) |
| Camera Model | Pinhole Only | Any Differentiable Camera Model |
| Secondary Rays | Unsupported | Native Support |
| Computational Overhead | Low | Slightly Higher (7 sigma points vs 1 matrix multiplication) |
| Accuracy | First-order Approximation | Second-order Accuracy |

#### Sigma Points Generation

For each 3D Gaussian $\mathcal{G}(oldsymbol{\mu}_{3D}, oldsymbol{\Sigma}_{3D})$, 7 sigma points are generated ($n=3$ in 3D space):
- Center point $oldsymbol{\chi}_0 = oldsymbol{\mu}_{3D}$
- 6 sample points along the directions of the principal axes of the covariance matrix

#### Non-linear Projection

All sigma points are food through the complete non-linear camera projection function $\mathbf{h}(\cdot)$:

$$oldsymbol{\mathcal{Y}}_i = \mathbf{h}(oldsymbol{\chi}_i)$$

Here, $\mathbf{h}$ can be any differentiable projection function, including fisheye models with distortion coefficients or rolling shutter models.

#### 2D Gaussian Recovery

The 2D Gaussian parameters are recovered from the projected sigma points:

$$oldsymbol{\mu}_{2D} = \sum_i w_i^{(m)} oldsymbol{\mathcal{Y}}_i, \quad oldsymbol{\Sigma}_{2D} = \sum_i w_i^{(c)} (oldsymbol{\mathcal{Y}}_i - oldsymbol{\mu}_{2D})(oldsymbol{\mathcal{Y}}_i - oldsymbol{\mu}_{2D})^T$$

### Rolling Shutter Support

Each row of pixels in a rolling shutter camera is exposed at different times, causing the jelly effect in moving objects. 3DGUT incorporates the temporal dimension into the projection function:

$$\mathbf{h}_{RS}(oldsymbol{x}, t) = \pi(oldsymbol{T}(t) \cdot oldsymbol{x})$$

where $oldsymbol{T}(t)$ is the time-varying camera pose. UT can naturally handle this space-time coupled non-linear projection.

### Secondary Ray Tracing

For reflection and refraction effects, this method attaches normal information to each Gaussian, computes the reflection/refraction direction via ray-Gaussian intersection, and generates secondary rays. The secondary rays interact with the Gaussians in the scene again, a process to which UT is equally applicable.

## Experimental Results

### Standard Scene Comparison

| Method | Mip-NeRF360 PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------------|-------|--------|
| 3DGS (Original) | 27.21 | 0.815 | 0.214 |
| Mip-Splatting | 27.79 | 0.827 | 0.203 |
| 3DGUT (Pinhole) | 27.83 | 0.829 | 0.199 |
| 3DGUT (Distorted) | **28.14** | **0.838** | **0.188** |

### Rolling Shutter Scenes

On the TUM-RS and Unreal-RS datasets, 3DGUT significantly outperforms baseline methods that ignore the shutter effect when processing rolling shutter scenes, achieving a PSNR improvement of 2-4 dB.

### Reflection/Refraction Scenes

In synthetic scenes containing specular reflection and glass refraction, 3DGUT is the first method capable of directly processing secondary ray effects within the 3DGS framework, avoiding complex designs of the past that required additional environment maps or multi-layer representations.

### Ablation Study

| Component | PSNR | Description |
|------|------|------|
| Full 3DGUT | 28.14 | Full Model |
| w/o UT (Fallback to EWA) | 27.21 | Degrades to standard 3DGS |
| UT + Pinhole | 27.83 | UT only, without distortion |
| UT + Distorted | 28.14 | Full distortion model |

## Summary & Future Work

By introducing the Unscented Transform to replace the traditional affine approximation in EWA splatting, 3DGUT provides a unified and elegant framework to handle distorted camera models and secondary ray effects. The core advantage of this method is that it **requires no linearization assumptions about the projection function**, as long as the projection function itself is differentiable. Although the computational overhead is slightly increased (approximately 1.3×), the resulting improvements in flexibility and accuracy offer significant value in practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images](cadcrafter_generating_computer-aided_design_models_from_unconstrained_images.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](../../ICCV2025/3d_vision/splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](../../ICCV2025/3d_vision/stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](../../ICCV2025/3d_vision/ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)

</div>

<!-- RELATED:END -->
