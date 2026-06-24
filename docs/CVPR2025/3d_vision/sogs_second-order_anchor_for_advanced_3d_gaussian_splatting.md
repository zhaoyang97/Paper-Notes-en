---
title: >-
  [Paper Note] SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] This work proposes SOGS, which introduces second-order anchors (utilizing the covariance matrix to capture correlations across feature dimensions for feature enhancement) and selective gradient loss into anchor-based 3D-GS. It achieves superior rendering quality compared to Scaffold-GS while reducing anchor feature dimensions from 32 down to 12-16.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Anchor Feature Enhancement"
  - "Second-Order Statistics"
  - "Model Compression"
  - "Selective Gradient Loss"
date: 2026-05-08
content_hash: 5d9545ce87f3ff87
---

# SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2503.07476](https://arxiv.org/abs/2503.07476)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Anchor Feature Enhancement, Second-Order Statistics, Model Compression, Selective Gradient Loss

## TL;DR

This work proposes SOGS, which introduces second-order anchors (utilizing the covariance matrix to capture correlations across feature dimensions for feature enhancement) and selective gradient loss into anchor-based 3D-GS. It achieves superior rendering quality compared to Scaffold-GS while reducing anchor feature dimensions from 32 down to 12-16.

## Background & Motivation

Anchor-based 3D-GS (such as Scaffold-GS) utilizes anchor features to predict Gaussian attributes via MLPs, effectively reducing Gaussian redundancy. However, it faces a core dilemma:

1. **Conflict between Feature Dimension and Model Size**: Large anchor features (e.g., 32 dimensions) improve rendering quality but significantly increase model size (due to the massive number of anchors). Conversely, reducing the feature dimension degrades the prediction quality of Gaussian attributes, leading to texture and geometric artifacts.
2. **Limitations of Compression Methods like HAC**: These methods compress the storage size rather than the actual model size during training and rendering.
3. **Insensitivity of Pixel-Level Loss to Textures**: L1 loss struggles to guide the model to focus on detailed textures and structures.

The core idea of this paper: correlations between feature dimensions (second-order statistics) can enhance the representation capability of each anchor, enabling higher-quality rendering with fewer feature dimensions.

## Method

### Overall Architecture

SOGS is built upon Scaffold-GS, where each anchor stores a feature $\mathbf{f}^a \in \mathbb{R}^D$ ($D$ is reduced from 32 to 12-16). By computing the covariance and correlation matrices of all anchor features, SOGS extracts the top-$M$ principal components (feature covariance patterns). These are combined with the original feature of each anchor to produce enhanced second-order features for Gaussian attribute prediction. A selective gradient loss is also added to focus on hard-to-render regions.

### Key Designs

**1. Second-Order Anchor**

- **Function**: To enhance anchor representation through the covariance relationship across feature dimensions, compensating for the information loss caused by reducing the feature dimension.
- **Mechanism**: The $D$-dimensional features of all $N$ anchors are treated as $N$ samples of $D$ variables to compute the covariance matrix $\Sigma \in \mathbb{R}^{D \times D}$, which is then normalized into a correlation matrix $\mathbf{R}$ (eliminating scale effects). Eigenvalue decomposition is performed on $\mathbf{R}$ to select the top-$M$ ($M=2$) eigenvectors $\mathbf{P}$ as the globally shared main covariance patterns. For each anchor: $\mathbf{f}^t_i = F_i([\mathbf{P}_i, \mathbf{f}^a])$, the enhanced features are concatenated with the original ones to predict Gaussian attributes.
- **Design Motivation**: Textures and structures are defined not only by individual features but also by the dependencies among them. Second-order statistics capture these dependencies, achieving feature enhancement without increasing the storage dimension of the anchors.

**2. Selective Gradient Loss**

- **Function**: To guide the model to adaptively focus on hard-to-render texture and structural regions.
- **Mechanism**: The Sobel operator is used to extract gradient maps of the rendered image and the ground truth, computing the gradient differences $l_x, l_y$ as loss. The key innovation is the introduction of dynamic region selection—using the absolute difference of gradients $w_x = |G'_x - G_x|$ as a weight map, which focuses the loss $\mathcal{L}_s = w_x \cdot l_x + w_y \cdot l_y$ on the regions with the largest rendering errors.
- **Design Motivation**: Most regions in a gradient map are flat with low gradients, meaning a naive gradient loss would be dominated by these non-informative regions. The weighting mechanism ensures the model continuously focuses on critical texture regions and dynamically adjusts the focus areas during training.

**3. Covariance Calculation and Correlation Matrix Construction**

- **Function**: To normalize the raw covariance, eliminating the impact of scale differences across different feature dimensions.
- **Mechanism**: $R_{uv} = \frac{\Sigma_{uv}}{\sigma_u \sigma_v}$ ensures that the principal components reflect true correlations between features rather than the magnitude of variances.
- **Design Motivation**: Two variables with large variances can produce a large covariance even if they are weakly related. The normalized correlation matrix more accurately captures the covariance patterns among features.

### Loss & Training

$$\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_{SSIM} \mathcal{L}_{SSIM} + \lambda_{vol} \mathcal{L}_{vol} + \lambda_s \mathcal{L}_s$$

where $\lambda_1 = 0.8$, $\lambda_{SSIM} = 0.2$, $\lambda_{vol} = 0.01$, and $\lambda_s = 0.01$.

## Key Experimental Results

### Main Results: Comparison with Scaffold-GS

| Dataset | Method | SSIM↑ | PSNR↑ | LPIPS↓ | Anchor Dimension |
|--------|------|-------|-------|--------|---------|
| Mip-NeRF360 | Scaffold-GS | 0.806 | 27.50 | 0.252 | 32 dim |
| Mip-NeRF360 | **SOGS** | **0.815** | **27.85** | **0.221** | **16 dim** |
| T&T | Scaffold-GS | 0.853 | 23.96 | 0.177 | 32 dim |
| T&T | **SOGS** | **0.855** | **24.14** | **0.176** | **12 dim** |
| Deep Blending | Scaffold-GS | 0.906 | 30.21 | 0.254 | 32 dim |
| Deep Blending | **SOGS** | **0.907** | **30.29** | **0.252** | **12 dim** |
| BungeeNeRF | Scaffold-GS | 0.865 | 26.62 | 0.241 | 32 dim |
| BungeeNeRF | **SOGS** | **0.880** | **27.06** | **0.171** | **16 dim** |

### Ablation Study (BungeeNeRF, Anchor Dimension 32)

| Model | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Base (Scaffold-GS) | 26.62 | 0.865 | 0.241 |
| Base + SOA | 27.25 | 0.879 | 0.208 |
| Base + SOA + SGL (SOGS) | **27.39** | **0.887** | **0.161** |

### Key Findings

- Reducing anchor dimensions from 32 to 12-16 actually improves rendering quality (PSNR +0.35, LPIPS -0.031).
- The second-order anchor contributes the most (PSNR +0.63, LPIPS -0.033), while the selective gradient loss further improves LPIPS.
- Diminishing returns are observed when the feature dimension exceeds 16; 12-16 serves as the optimal trade-off between model size and performance.
- The improvement is even more significant in large-scale scenes like BungeeNeRF (LPIPS drops from 0.241 to 0.171).

## Highlights & Insights

1. **Introduction of Statistical Concepts to 3D-GS**: Classical statistical methods (PCA/covariance) are used to enhance neural field features, constituting a novel concept with a solid theoretical foundation.
2. **Smaller Model with Better Quality**: Breaks the conventional assumption that "larger features equal higher quality", proving that correlation among features can compensate for dimensionality reduction.
3. **High Practicality of Selective Gradient Loss**: The simple Sobel-based scheme effectively improves texture rendering quality.

## Limitations & Future Work

- PCA decomposition introduces additional computational overhead; though real-time rendering is maintained, training time increases slightly.
- The choice of top-$M$ requires parameter tuning ($M=2$ is used in the paper).
- The MLP used for feature enhancement introduces extra parameters.
- Future work could explore adaptively determining $M$ and the feature dimension $D$.

## Related Work & Insights

- **Scaffold-GS**: The baseline method; SOGS performs second-order enhancement on its anchor features.
- **HAC/ContextGS**: Methods that compress the storage size of Scaffold-GS, but do not reduce the actual runtime model size.
- **3D-GS**: The original method, which suffers from severe Gaussian redundancy.

## Rating

⭐⭐⭐⭐ — Addresses the feature efficiency issue of 3D-GS using classical statistical methods, with a clear and elegant approach. It achieves superior performance across multiple benchmarks with a smaller model, and the ablation studies thoroughly validate the contributions of each component.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Mitigating Ambiguities in 3D Classification with Gaussian Splatting](mitigating_ambiguities_in_3d_classification_with_gaussian_splatting.md)
- [\[CVPR 2025\] DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)

</div>

<!-- RELATED:END -->
