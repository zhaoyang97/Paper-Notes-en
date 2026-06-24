---
title: >-
  [Paper Note] CF³: Compact and Fast 3D Feature Fields
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes the CF³ pipeline, which constructs a compact and high-speed 3D feature field using only 5% of the original Gaussian count via top-down feature lifting, per-Gaussian autoencoder compression, and adaptive sparsification, achieving 121–245× storage compression with real-time rendering.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Feature Field"
  - "Sparsification"
  - "Feature Compression"
  - "Open-Vocabulary Segmentation"
date: 2026-05-08
content_hash: 03e4a0080d999b57
---

# CF³: Compact and Fast 3D Feature Fields

**Conference**: ICCV 2025
**arXiv**: [2508.05254](https://arxiv.org/abs/2508.05254)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Feature Field, Sparsification, Feature Compression, Open-Vocabulary Segmentation

## TL;DR

This paper proposes the CF³ pipeline, which constructs a compact and high-speed 3D feature field using only 5% of the original Gaussian count via top-down feature lifting, per-Gaussian autoencoder compression, and adaptive sparsification, achieving 121–245× storage compression with real-time rendering.

## Background & Motivation

Embedding semantic features from 2D foundation models (e.g., CLIP, SAM, LSeg) into 3DGS is the dominant approach for open-vocabulary 3D understanding. However, existing methods (Feature-3DGS, LangSplat) suffer from two major issues:

**Heavy bottom-up optimization overhead**: Treating raw 2D features as ground truth and jointly optimizing color and features leads to long training times, and produces excessive redundant Gaussians to recover color detail.

**Storage explosion from high-dimensional features**: Directly embedding 512-dimensional features into each Gaussian imposes enormous storage and computational burdens. Existing compression techniques (2D autoencoders, hash grids, vector quantization) do not explicitly address the key insight that Gaussians optimized for color rendering are redundant for the feature field.

Furthermore, features from 2D foundation models lack multi-view consistency. CF³ adopts a top-down perspective: it first uses a pretrained 3DGS to perform feature lifting to obtain view-consistent features, then applies compression and sparsification.

## Method

### Overall Architecture

The CF³ pipeline consists of three stages:
1. **Feature Lifting**: Weighted aggregation of multi-view 2D features using alpha blending weights from a pretrained 3DGS, assigning each Gaussian a view-consistent reference feature.
2. **Feature Compression**: A per-Gaussian autoencoder trained on the lifted features, compressing high-dimensional features to a 3-dimensional latent space (equivalent to RGB channels).
3. **Adaptive Sparsification**: Optimization of Gaussian attributes on the feature field, drastically reducing the number of Gaussians through pruning and merging of redundant ones.

### Key Designs

1. **Feature Lifting**: The rendering weights $w_{i,m,p}$ from 3DGS are used to compute a weighted average of 2D features across all views:

    $\boldsymbol{f}_i \approx \frac{\sum_{m=1}^{M}\sum_{p \in \mathcal{P}_{i,m}} w_{i,m,p} \boldsymbol{F}_{m,p}}{\sum_{m=1}^{M}\sum_{p \in \mathcal{P}_{i,m}} w_{i,m,p}}$

   Feature variance is also computed to filter out the top 0.01% highest-variance Gaussians (typically located at geometrically inaccurate regions or object boundaries), eliminating noise from multi-view inconsistency. **Design Motivation**: This avoids joint optimization from scratch and directly reuses the geometric information of the pretrained 3DGS, enabling fast and view-consistent feature assignment.

2. **Per-Gaussian Autoencoder (Feature Compression)**: Unlike LangSplat, which trains a 2D autoencoder before lifting, CF³ lifts first and then compresses. The autoencoder is a 5-layer MLP (128→64→32→16→3) that compresses $D$-dimensional features to 3 dimensions. The loss function includes:

    - MSE reconstruction loss
    - Cosine similarity loss $\mathcal{L}_{cos}$
    - Structure-preserving regularization $\mathcal{L}_{struc}$: preserves inter-Gaussian feature similarity relationships

   The key advantage of 3D compression is that it enables direct reuse of existing 3DGS rasterizers by treating the compressed features as RGB channels during rendering.

3. **Adaptive Sparsification**: Two alternating operations are performed:

    - **Pruning**: Low-contribution Gaussians are removed based on global contribution score $C(g_i) = \sum_{m,p} w_{i,m,p}$
    - **Merging**: For Gaussians with small gradients (converged regions), $k$-nearest neighbors are identified; if the feature cosine similarity is high $\langle \boldsymbol{c}_i, \boldsymbol{c}_j \rangle > \tau_{sim}$ and the Mahalanobis distance satisfies a chi-squared test $d_M < \chi^2_\beta$, moment matching is applied to merge them. The attributes (position, covariance, opacity, features) of the merged Gaussian are computed via weighted averaging.

### Loss & Training

- Autoencoder training: $\mathcal{L} = \mathcal{L}_{MSE} + \lambda_{cos} \mathcal{L}_{cos} + \lambda_{struc} \mathcal{L}_{struc}$
- Sparsification stage (optimizing Gaussian attributes): $\mathcal{L} = \|\boldsymbol{F}_{ref} - \boldsymbol{F}\|_1 + \lambda_{depth} \|D_{ref} - D\|_1$, where the reference feature field is frozen and the new feature field is optimized via L1 feature loss and depth regularization
- Built upon a pretrained 3DGS scene (30k iterations); the full pipeline takes approximately 30 minutes per scene

## Key Experimental Results

### Main Results (Tables)

**Replica Dataset (LSeg Features)**

| Method | Storage ↓ | FPS ↑ | mIoU ↑ | Acc ↑ | #Gaussian ↓ |
|--------|-----------|-------|--------|-------|-------------|
| Feature-3DGS (512d) | 1393.9M | 7.2 | 73.0 | 91.9 | 636k |
| Feature-3DGS (128d) | 463.9M | 113.8 | 73.4 | 92.9 | 640k |
| CF³ (Ours) | **3.6M** | **328.3** | 70.8 | 91.6 | **47k** |
| CF³ + VQ | **1.7M** | 327.3 | 70.1 | 90.9 | 47k |

**LERF Dataset (CLIP+SAM Features)**

| Method | Storage ↓ | FPS ↑ | mIoU ↑ | Acc ↑ | #Gaussian ↓ |
|--------|-----------|-------|--------|-------|-------------|
| LangSplat | 314.9M | 33.4 | 44.7 | 72.3 | 1270k |
| Feature-3DGS (128d) | 1031.7M | 55.6 | 53.8 | 75.8 | 1423k |
| CF³ (Ours) | **4.2M** | **145.0** | 52.4 | 76.8 | **55k** |

### Ablation Study (Tables)

**Component-wise Ablation (Replica + LERF)**

| VF | Pruning | Merging | LSeg mIoU | LSeg #G | CLIP+SAM mIoU | CLIP+SAM #G |
|----|---------|---------|-----------|---------|---------------|-------------|
| ✗ | ✗ | ✗ | 61.0 | 600k | 29.7 | 1289k |
| ✓ | ✓ | ✗ | 71.0 | 165k | 53.4 | 324k |
| ✗ | ✓ | ✓ | 69.8 | 43k | 54.5 | 56k |
| ✓ | ✓ | ✓ | **70.8** | **47k** | 52.2 | **55k** |

The merging step contributes an additional ~70% storage reduction; variance filtering is particularly effective for low-resolution MaskCLIP features.

### Key Findings

- CF³ is **121×** more compact than Feature-3DGS (128d) and **74×** more compact than LangSplat, while remaining competitive in performance
- In MaskCLIP experiments, CF³ surpasses Feature-3DGS in mIoU by **over 30%** (46.9 vs. 35.9), indicating that adaptive sparsification compensates for low-resolution features
- On the 3D-OVS dataset, CF³ achieves 84.5 mIoU (vs. LangSplat's 81.9) using only 21k Gaussians
- On the large-scale outdoor KITTI-360 dataset, CF³ requires only 6.2M storage (vs. 3810.2M) and achieves real-time rendering at 141.6 FPS

## Highlights & Insights

1. **Top-down is superior to bottom-up**: Lifting features to 3D before compression better preserves feature distribution and reduces inconsistency compared to compressing 2D features before optimization.
2. **3D compression = RGB channels**: This minimalist design ensures full compatibility with existing 3DGS rendering pipelines without requiring additional decoders.
3. **Explicit treatment of redundancy**: Densely packed Gaussians produced by color optimization are redundant for the feature field; merging semantically similar Gaussians via moment matching is a natural and efficient strategy.
4. **Variance filtering**: A simple yet effective mechanism that leverages multi-view consistency to detect and filter noisy features.

## Limitations & Future Work

- The full pipeline takes approximately 30 minutes per scene (primarily in autoencoder training and sparsification), leaving room for further acceleration.
- Whether 3 dimensions is the optimal compression target remains an open question; higher dimensionality may preserve more semantic detail.
- The robustness of merging thresholds $\tau_{sim}$ and $\chi^2_\beta$ across diverse scenes warrants further investigation.
- Dynamic scenes and large-scale real-time feature field updates are not addressed.

## Related Work & Insights

- **Feature-3DGS / LangSplat**: Representative feature embedding methods; CF³ addresses their redundancy and compression limitations.
- **LightGaussian**: Provides the foundation for global contribution-based pruning.
- **FiT3D / CONDENSE**: Pioneers of 3D-aware training paradigms.
- Insight: *Compactness* and *speed* of 3D feature fields may matter more than accuracy, especially in real-time application scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of a top-down pipeline, per-Gaussian AE, and moment-matching merging constitutes systematic innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across multiple datasets, feature types, with full ablations, 3D segmentation, and large-scale scene validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear pipeline description, rigorous mathematical derivations, and intuitive figures.
- **Value**: ⭐⭐⭐⭐⭐ 100×+ compression with real-time rendering carries significant practical importance for deploying 3D feature fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)
- [\[CVPR 2026\] Learning Convex Decomposition via Feature Fields](../../CVPR2026/3d_vision/learning_convex_decomposition_via_feature_fields.md)
- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)
- [\[CVPR 2025\] Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](../../CVPR2025/3d_vision/gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)
- [\[ICCV 2025\] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)

</div>

<!-- RELATED:END -->
