---
title: >-
  [Paper Note] Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)
description: >-
  [CVPR 2025][3D Vision][Gaussian Splatting] This paper proposes EOGS, the first Earth observation framework based on 3D Gaussian Splatting. Through affine camera approximation, shadow mapping, and three regularization strategies, it achieves reconstruction accuracy comparable to EO-NeRF on satellite image 3D reconstruction tasks, while speeding up training by 300x (3 minutes vs. 15 hours).
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Gaussian Splatting"
  - "Satellite Photogrammetry"
  - "Digital Elevation Model"
  - "Shadow Modeling"
  - "Remote Sensing"
date: 2026-05-08
content_hash: 875260a2206d63df
---

# Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)

**Conference**: CVPR 2025  
**arXiv**: [2412.13047](https://arxiv.org/abs/2412.13047)  
**Code**: [https://mezzelfo.github.io/EOGS/](https://mezzelfo.github.io/EOGS/)  
**Area**: 3D Vision  
**Keywords**: Gaussian Splatting, Satellite Photogrammetry, Digital Elevation Model, Shadow Modeling, Remote Sensing

## TL;DR
This paper proposes EOGS, the first Earth observation framework based on 3D Gaussian Splatting. Through affine camera approximation, shadow mapping, and three regularization strategies, it achieves reconstruction accuracy comparable to EO-NeRF on satellite image 3D reconstruction tasks, while speeding up training by 300x (3 minutes vs. 15 hours).

## Background & Motivation

**Background**: Satellite remote sensing photogrammetry aims to recover the 3D surface geometry (Digital Surface Model, DSM) and appearance from multi-view satellite images. Traditional methods rely on binocular or trinocular stereo vision, which requires images captured almost simultaneously and at specific satellite positions. Recently, NeRF-based methods (e.g., EO-NeRF) have achieved high-accuracy reconstruction across multiple dates and arbitrary satellite positions using differentiable volume rendering, becoming the current SOTA.

**Limitations of Prior Work**: Although EO-NeRF achieves the highest accuracy, its training time exceeds 15 hours, making it difficult to cope with the exponential growth of future satellite data. While SAT-NGP reduces the training time to 25 minutes, its accuracy decreases significantly.

**Key Challenge**: Remote sensing scenarios have unique characteristics—such as pushbroom sensor camera models, complex illumination/shadow variations, and radiometric inconsistencies caused by multi-date acquisition—which standard 3DGS cannot directly handle. In addition, 3DGS lacks the implicit regularization properties of NeRF, leading to unstable optimization under sparse views.

**Goal**: (1) How to adapt the highly efficient 3DGS framework to the unique camera models and illumination conditions of remote sensing scenes; (2) how to ensure reconstruction quality under sparse views in 3DGS, which lacks implicit regularization.

**Key Insight**: Pushbroom satellite sensors can be accurately approximated locally by an affine camera (with an error of only 0.012 pixels), which is perfectly compatible with the mathematical formulation of Gaussian Splatting. Shadows can be modeled using Shadow Mapping from graphics, determining occlusions via rendering from an extra "sun camera".

**Core Idea**: Using affine camera approximation for satellite sensors + Shadow Mapping for shadow modeling + a triple regularization strategy (sparsity, view consistency, and opaqueness) to enable 3DGS to balance both efficiency and accuracy in remote sensing scenarios.

## Method

### Overall Architecture
Based on standard 3DGS, EOGS takes $N$ non-orthorectified satellite images and their RPC camera parameters as input, optimizing a set of Gaussian primitives to reconstruct the 3D geometry and appearance of the scene. The overall pipeline includes: (1) approximate conversion from RPC to affine camera; (2) differentiable rendering with shadow mapping; (3) joint optimization with three regularization constraints. The final outputs are the rendered elevation map and albedo map.

### Key Designs

1. **Affine Camera Approximation**:

    - **Function**: Transforms the complex RPC satellite camera model into a linear projection compatible with 3DGS.
    - **Core Idea**: The complete transformation chain from world coordinates to NDC space (world $\rightarrow$ UTM $\rightarrow$ Lon-Lat-Alt $\rightarrow$ RPC $\rightarrow$ row-col $\rightarrow$ NDC) can be approximated by a scene-specific affine transformation $\mathcal{A}(\mathbf{x}) = A\mathbf{x} + \mathbf{a}$ with an average error of only about 0.012 pixels. Under the affine model, the Gaussian projection formulas simplify to $\boldsymbol{\mu}^{\mathcal{A}}_k = A\boldsymbol{\mu}_k + \mathbf{a}$ and $\Sigma^{\mathcal{A}}_k = A\Sigma A'$, eliminating the need for local first-order approximations as in standard 3DGS.
    - **Design Motivation**: Eliminates the non-linear approximation errors of perspective projection, while unifying the processing of both satellite and sun cameras.

2. **Shadow Mapping**:

    - **Function**: Physically and accurately models building shadows in satellite images within the Gaussian Splatting framework.
    - **Core Idea**: Constructs a "sun camera" $\mathcal{S}$ (modeling directional light sources using an affine camera) to render elevation maps from both the sun and satellite views. For each pixel $\mathbf{u}$ in the satellite view, the elevation difference $\Delta h$ with its corresponding point in the sun view is computed. If $\Delta h > 0$ (meaning the sun sees a point that is higher), then the point is in shadow. An exponential decay function $s = \min\{\exp(-\rho \Delta h), 1\}$ is used to calculate the darkening coefficient, which is combined with per-camera ambient light parameters $\psi^{\mathcal{A}}$ to obtain the final illumination coefficient.
    - **Design Motivation**: EO-NeRF uses ray tracing to compute shadows, but Gaussian Splatting lacks the concept of ray tracing. Shadow Mapping only requires rendering the scene from different perspectives—which is precisely what 3DGS excels at, perfectly aligning with the locality assumption of Gaussian Splatting.

3. **Triple Regularization Strategy**:

    - **Function**: Compensates for the lack of implicit regularization in 3DGS compared to NeRF, improving reconstruction quality under sparse views.
    - **Core Idea**: (a) **Sparsity Regularization** $\mathcal{L}_o = \frac{1}{K}\sum\alpha_k$: Applies an L1 penalty on opacities, encouraging useless Gaussians to diminish naturally, which speeds up training by 2x when combined with threshold pruning; (b) **View Consistency Regularization**: Randomly perturbs the camera to generate virtual views, requiring the color and elevation of the same 3D point to remain consistent between real and virtual views, utilizing an occlusion mask to avoid interference from occluded parts; (c) **Opaqueness Regularization** $\mathcal{L}_s = \sum H(s)$ : Applies a binary entropy penalty to the shadow map, forcing shadows to be binary (black or white), preventing semi-transparent objects from "falsely leveraging" shadows to encode texture.
    - **Design Motivation**: Remote sensing scenarios have few and sparse views, resulting in 3DGS primitives being optimized almost independently with insufficient constraints. The three regularizations impose priors from different angles: reducing primitive count, ensuring multi-view geometric consistency, and securing physically reasonable shadows.

### Loss & Training
The total loss is: $\min \sum_{i=1}^N \ell(\hat{I}_i, I_i) + 0.1\mathcal{L}_o + 0.1\mathcal{L}_{cc} + 0.01\mathcal{L}_{ac} + 0.01\mathcal{L}_s$, where $\ell$ is the standard 3DGS photometric loss. The regularization coefficients are determined experimentally on a single scene, taken as the nearest power of 10, and directly applied to all scenes. Training only requires 5000 iterations, with shadow mapping and regularization enabled at the 1000th iteration. Gaussians are initialized as white with low opacity (1%) and a density of approximately $0.13 \text{ per } m^3$.

## Key Experimental Results

### Main Results
The datasets are from DFC2019 (JAX 4 scenes) and IARPA2016 (3 scenes), using LiDAR scans as elevation ground truths.

| Method | JAX MAE↓(m) | IARPA MAE↓(m) | Training Time |
|------|-------------|---------------|----------|
| EO-NeRF | 1.35 | 1.51 | 15 hours |
| SAT-NGP | 1.72 | 1.78 | 25 mins |
| S2P | 1.53 | 1.78 | 20 mins |
| **EOGS** | **1.46** | **1.62** | **3 mins** |

After applying vegetation masking:

| Method | JAX MAE↓(m) | IARPA MAE↓(m) |
|------|-------------|---------------|
| EO-NeRF | 1.21 | 1.38 |
| **EOGS** | **1.19** | **1.37** |

### Ablation Study

| Configuration | MAE↓(m) | Training Time (min) |
|------|---------|--------------|
| Base (Affine 3DGS) | 5.03 | 4.18 |
| + Shadow Mapping | 1.86 | - |
| + Sparsity | 1.83 | - |
| + Consistency | 1.69 | - |
| + Opaqueness | 1.79 | - |
| Full EOGS | 1.54 | 2.85 |

Linear regression analysis shows the independent contribution of each component: Shadow Mapping 3.16m > Consistency 0.20m > Opaqueness 0.09m > Sparsity 0.04m.

### Key Findings
- Shadow Mapping is the most critical component, contributing the vast majority of the accuracy gain (3.16m), which highlights shadow modeling as the core of remote sensing scenarios.
- Although sparsity regularization has the smallest contribution to accuracy (0.04m), it significantly accelerates training (by 2x) and is key to efficiency.
- After removing vegetation areas, EOGS's accuracy is fully on par with EO-NeRF, indicating that EOGS achieves extremely high reconstruction quality for structured objects (such as buildings).
- EOGS performs better than EO-NeRF in high-coverage areas (regions observed by multiple cameras) but is slightly weaker in low-coverage areas.

## Highlights & Insights
- **The elegance of affine approximation**: Although pushbroom satellite sensors seem complex, because the satellites are extremely far away and the scene scale is relatively small, the affine approximation error is only 0.012 pixels. This observation eliminates the largest obstacle to adapting 3DGS and enables the unified processing of both the sun camera and satellite cameras.
- **The ingenious migration of Shadow Mapping**: Introducing the classic Shadow Mapping technique from computer graphics into a differentiable rendering framework perfectly bypasses the limitation that "3DGS does not have ray tracing". This approach—borrowing mature techniques from graphics to solve differentiable rendering problems—is highly worth promoting.
- **The practical value of 300x speedup**: The difference between 3 minutes and 15 hours changes large-scale satellite data processing from infeasible to feasible, marking a genuine engineering breakthrough.

## Limitations & Future Work
- Poor performance in vegetation areas: Non-rigid, semi-transparent objects like trees remain difficult to model precisely with Gaussian primitives.
- The reconstruction quality in low-coverage areas (observed by only a few cameras) is weaker than EO-NeRF, requiring stronger priors or better initialization.
- Currently, only optical satellite images are supported, without considering SAR or multi-spectral data.
- Although the regularization coefficients are robust, they are still manually set (rounded to the nearest power of 10).

## Related Work & Insights
- **vs EO-NeRF**: EO-NeRF uses ray tracing to compute shadows and has implicit regularization, which achieves slightly higher accuracy but extremely slow training. EOGS replaces these with Shadow Mapping and explicit regularization, achieving a qualitative leap in efficiency.
- **vs SAT-NGP**: An acceleration scheme based on Instant-NGP, taking 25 minutes but with a significant drop in accuracy. EOGS is both faster and more accurate, representing a Pareto-optimal solution.
- **vs Standard 3DGS**: The core contribution of EOGS is demonstrating that "adapting 3DGS for remote sensing" only requires a few adjustments (affine camera + shadow modeling + regularization) rather than redesigning the entire framework from scratch.

## Rating
- Novelty: ⭐⭐⭐⭐ The first to apply 3DGS to satellite photogrammetry; the combination of affine approximation and Shadow Mapping is natural yet genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on 7 scenes + detailed ablation studies + parameter sensitivity analysis + visibility analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations, intuitive illustrations, and a logically complete methodology section.
- Value: ⭐⭐⭐⭐⭐ The 300x acceleration turns large-scale remote sensing into a reality, possessing extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Matrix3D: Large Photogrammetry Model All-in-One](matrix3d_large_photogrammetry_model_all-in-one.md)
- [\[CVPR 2025\] RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting](rainygs_efficient_rain_synthesis_with_physically-based_gaussian_splatting.md)
- [\[CVPR 2025\] GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](../../ICCV2025/3d_vision/sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)

</div>

<!-- RELATED:END -->
