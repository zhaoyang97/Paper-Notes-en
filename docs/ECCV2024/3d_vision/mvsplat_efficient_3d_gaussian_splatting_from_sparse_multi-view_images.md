---
title: >-
  [Paper Note] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] MVSplat is proposed, which constructs a cost volume via plane-sweep to accurately locate Gaussian centers. It achieves state-of-the-art sparse-view feed-forward 3D Gaussian prediction with significantly fewer parameters (1/10 of pixelSplat) and the fastest inference speed (22 fps).
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "feed-forward reconstruction"
  - "cost volume"
  - "sparse views"
  - "novel view synthesis"
date: 2026-05-08
content_hash: ffaf62ae68bd17db
---

# MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images

**Conference**: ECCV 2024  
**arXiv**: [2403.14627](https://arxiv.org/abs/2403.14627)  
**Code**: [GitHub](https://github.com/donydchen/mvsplat)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, feed-forward reconstruction, cost volume, sparse views, novel view synthesis

## TL;DR

MVSplat is proposed, which constructs a cost volume via plane-sweep to accurately locate Gaussian centers. It achieves state-of-the-art sparse-view feed-forward 3D Gaussian prediction with significantly fewer parameters (1/10 of pixelSplat) and the fastest inference speed (22 fps).

## Background & Motivation

**Background**: Feed-forward reconstruction of 3D scenes from sparse views (as few as 2 images) is a highly active research area. NeRF-based methods (e.g., pixelNeRF, MuRF) require expensive volume rendering sampling, resulting in slow speeds. Benefiting from rasterized rendering, 3DGS naturally bypasses the volume sampling overhead. Recent works like pixelSplat have begun exploring feed-forward 3DGS.

**Limitations of Prior Work**: Although pixelSplat introduces an epipolar Transformer to learn cross-view features, it still directly regresses a probabilistic depth distribution from image features. This mapping from features to depth is fundamentally ambiguous and unreliable, leading to poor geometric quality and a large number of floater Gaussians. To obtain reasonable geometry, an additional 50K steps of depth-regularized fine-tuning are required. Moreover, pixelSplat contains 125M parameters, which is quite heavy.

**Key Challenge**: Accurately localizing 3D Gaussian centers is crucial for high-quality rendering, but the data-driven regression approach (from-feature-to-depth) struggles to provide reliable geometric awareness. A more geometrically aware depth estimation approach is required.

**Goal**: Design a lightweight and efficient feed-forward model to directly predict high-quality 3D Gaussians from sparse multi-view images.

**Key Insight**: Introduce the classic **cost volume** from Multi-View Stereo (MVS). Estimate depth through feature matching instead of feature regression, shifting the paradigm from "predicting depth from features" to "finding depth from matching."

**Core Idea**: Utilizing cross-view feature matching information encoded by a plane-sweep cost volume to locate Gaussian centers is more reliable and lightweight than directly regressing probabilistic depth.

## Method

### Overall Architecture

The pipeline of MVSplat is: (1) Extract cross-view aware features $\{\boldsymbol{F}^i\}$ using a CNN + Transformer; (2) Uniformly sample $D$ depth candidates in the inverse depth space via plane-sweep to construct a cost volume $\boldsymbol{C}^i \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times D}$ for each view; (3) Refine the cost volume and predict depth maps using a U-Net; (4) Back-project the depth maps to obtain Gaussian centers while concurrently predicting opacity, covariance, and spherical harmonics (SH) colors; (5) Render novel views via 3DGS rasterization, trained end-to-end using only an RGB photometric loss.

### Key Designs

1. **Cost Volume Construction**: For view $i$, the features of another view $j$ are warped via homography according to the depth candidate $d_m$:

$$\boldsymbol{F}_{d_m}^{j \to i} = \mathcal{W}(\boldsymbol{F}^j, \boldsymbol{P}^i, \boldsymbol{P}^j, d_m)$$

The correlation is then computed using dot-product similarity:

$$\boldsymbol{C}_{d_m}^i = \frac{\boldsymbol{F}^i \cdot \boldsymbol{F}_{d_m}^{j \to i}}{\sqrt{C}}$$

Stacking $D$ correlations yields the cost volume $\boldsymbol{C}^i = [\boldsymbol{C}_{d_1}^i, \ldots, \boldsymbol{C}_{d_D}^i]$. When there are more than two views, pixel-wise averaging is performed on the correlations, enabling the model to accept an arbitrary number of inputs. **Design Motivation**: The cost volume captures the **relative similarity** between features, decoupling it from the absolute scale of features, which naturally provides strong cross-dataset generalization.

2. **Cost Volume Refinement**: A lightweight 2D U-Net takes the concatenation of Transformer features and the cost volume as input to output the residual $\Delta\boldsymbol{C}^i$:

$$\tilde{\boldsymbol{C}}^i = \boldsymbol{C}^i + \Delta\boldsymbol{C}^i$$

To exchange information between different views' cost volumes, a 3-layer cross-view attention is injected at the lowest resolution layer of the U-Net. This cross-view attention is independent of the number of views. Finally, the cost volume is upsampled to the full resolution $\hat{\boldsymbol{C}}^i \in \mathbb{R}^{H \times W \times D}$ using a CNN upsampler.

3. **Depth Estimation**: The refined cost volume is normalized via softmax along the depth dimension, and the depth candidates are then weighted averaged:

$$\boldsymbol{V}^i = \text{softmax}(\hat{\boldsymbol{C}}^i) \boldsymbol{G}$$

where $\boldsymbol{G} = [d_1, \ldots, d_D]$ represent the depth candidates. A lightweight U-Net is additionally employed for depth residual refinement.

4. **Gaussian Parameter Prediction**: (a) **Center $\mu$**: The depth map is directly back-projected to 3D world coordinates, with a simple union applied to multi-view point clouds; (b) **Opacity $\alpha$**: The maximum value of the softmax matching distribution indicates the matching confidence, which is mapped to opacity via two convolutional layers; (c) **Covariance and Color**: Predicted via two convolutional layers from the concatenated image features, cost volume, and original image. Only 1 Gaussian is predicted per pixel (compared to 3 in pixelSplat), resulting in a total of $H \times W \times K$ Gaussians.

### Loss & Training

A linear combination of $\ell_2$ + 0.05 × LPIPS. Does not require any ground-truth depth supervision. Trained for 300K iterations on a single A100. The cost volume samples 128 depth candidates. Swin Transformer's local window attention is utilized to improve efficiency.

## Key Experimental Results

### Main Results

**RealEstate10K + ACID Novel View Synthesis**:

| Method | Params (M) | Time (s) | RE10K PSNR↑ | RE10K LPIPS↓ | ACID PSNR↑ | ACID LPIPS↓ |
|------|---------|---------|-------------|-------------|------------|-------------|
| pixelNeRF | 28.2 | 5.299 | 20.43 | 0.550 | 20.97 | 0.533 |
| MuRF | 5.3 | 0.186 | 26.10 | 0.143 | 28.09 | 0.155 |
| pixelSplat | 125.4 | 0.104 | 25.89 | 0.142 | 28.14 | 0.150 |
| **MVSplat** | **12.0** | **0.044** | **26.39** | **0.128** | **28.25** | **0.144** |

MVSplat outperforms pixelSplat while using only **1/10 of the parameters** and being over **2x faster**.

### Ablation Study

**Cross-Dataset Generalization (Trained on RE10K → Tested on ACID/DTU)**:

| Method | ACID PSNR↑ | ACID LPIPS↓ | DTU PSNR↑ | DTU LPIPS↓ |
|------|------------|-------------|-----------|-----------|
| pixelSplat | 27.64 | 0.160 | 12.89 | 0.560 |
| **MVSplat** | **28.15** | **0.147** | **13.94** | **0.385** |

In high domain-gap scenarios such as transferring from the source domain to DTU, MVSplat improves LPIPS by 31%, demonstrating the generalization advantage brought by the feature-invariant nature of cost volumes.

### Key Findings

- The underlying 3D structure of pixelSplat contains a large number of floater Gaussians, although its 2D rendering appears reasonable; MVSplat's predicted center quality is far superior to pixelSplat's.
- Utilizing the cost volume yields high-quality geometry without requiring extra depth-regularized fine-tuning.
- The cost volume captures relative correlation, making it effective even when the feature distribution changes (across datasets).
- With only 1 Gaussian per pixel (vs. 3 in pixelSplat), rendering is also faster.

## Highlights & Insights

- **Paradigm Shift from Regression to Matching**: Shifting depth estimation from "data-driven regression" to "matching-based inference" intrinsically simplifies the learning difficulty.
- **Return of Classic MVS Wisdom**: Re-demonstrates the value of the classic stereo vision tool—the cost volume—in the deep learning era, without requiring ground-truth depth supervision.
- **Extremely Efficient**: Achieves state-of-the-art performance with only 12M parameters and 22 fps inference speed, showcasing high practicality.
- **Design Consistency**: Opacity is derived from matching confidence, and Gaussian centers are derived from matching depth. All key quantities originate from the same cost volume.

## Limitations & Future Work

- The cost volume requires known camera intrinsic and extrinsic parameters, making it unable to handle unknown camera settings.
- Restricted by the $256 \times 256$ resolution limit, as the memory footprint of the cost volume scales up heavily at higher resolutions.
- Only evaluated on 2–3 input views; the scalability of cost volume computation for more views needs further assessment.
- Ambiguities in textureless regions remain a challenge for the cost volume; although the U-Net can partially correct them, there is an upper bound on performance.
- Temporal consistency has not been explored, which hinders direct application to video scenarios.

## Related Work & Insights

- **pixelSplat**: The most direct baseline method, which highlights the limitations of data-driven depth regression.
- **MVSNet Series**: The concept of cost volume construction originates from classic MVS, though MVSplat does not require ground-truth depth supervision.
- **UniMatch/GMFlow**: The success of combining Transformers and cost volumes in optical flow and stereo matching inspired the design of this work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introduces the MVS cost volume into feed-forward 3DGS, presenting a clear and effective approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Solid evaluation across three datasets, cross-dataset generalization, geometric visualization, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The paper is well-structured and the comparative analyses are explicitly clear.
- **Value**: ⭐⭐⭐⭐⭐ — Significantly advances the practicality of feed-forward 3DGS, with the code being open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[ECCV 2024\] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [\[ECCV 2024\] MVDiffusion++: A Dense High-Resolution Multi-View Diffusion Model for Single or Sparse-View 3D Object Reconstruction](mvdiffusion_a_dense_high-resolution_multi-view_diffusion_model_for_single_or_spa.md)
- [\[ECCV 2024\] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)
- [\[ECCV 2024\] SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
