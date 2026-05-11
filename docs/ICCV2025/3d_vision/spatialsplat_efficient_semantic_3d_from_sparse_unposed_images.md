---
title: >-
  [Paper Note] SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images
description: >-
  [ICCV 2025][3D Vision][Semantic 3DGS] SpatialSplat is proposed to generate compact semantic 3D Gaussians from sparse unposed images via feed-forward inference…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Semantic 3DGS"
  - "Feed-forward Reconstruction"
  - "Unposed Images"
  - "Dual-field Architecture"
  - "Gaussian Selection"
date: 2026-05-08
content_hash: 020d5dd08c763ebb
---

# SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images

**Conference**: ICCV 2025  
**arXiv**: [2505.23044](https://arxiv.org/abs/2505.23044)  
**Code**: [GitHub](https://github.com/shengyu724/SpatialSplat)  
**Area**: 3D Vision  
**Keywords**: Semantic 3DGS, Feed-forward Reconstruction, Unposed Images, Dual-field Architecture, Gaussian Selection

## TL;DR

SpatialSplat is proposed to generate compact semantic 3D Gaussians from sparse unposed images via feed-forward inference, leveraging a dual-field semantic representation and a selective Gaussian mechanism that reduces representation parameters by 60% while surpassing state-of-the-art methods.

## Background & Motivation

Semantics-aware 3D reconstruction, which recovers semantic 3D structure from 2D images, is a foundational technology for robotics, autonomous driving, and VR/AR. Existing feed-forward 3DGS methods face two core challenges when incorporating semantics:

**Redundancy in per-pixel Gaussian prediction** — overlapping regions produce a large number of redundant primitives, incurring unnecessary memory overhead.

**Compression loss of high-dimensional semantic features** — 512-dimensional language features must be compressed to 64–128 dimensions to be attached to each primitive, leading to **irreversible information loss**.

Existing methods (e.g., LSM) naively append compressed features to every pixel-level Gaussian, which is neither efficient nor accurate.

### Key Findings

1. Redundant primitives share similar geometry and appearance and can be identified directly from image features (without geometric priors).

**Per-primitive semantics are unnecessary** — Gaussians within the same instance exhibit high semantic consistency; coarse-grained semantics combined with fine-grained instance information are sufficient.

## Method

### Dual-field Semantic Representation

The dense semantic feature field is decomposed into two components:

**Fine-grained instance-aware radiance field** $\mathcal{F}_I$:
- Each Gaussian carries a low-dimensional instance feature $\boldsymbol{f}_I \in \mathbb{R}^N$ and an importance score $\boldsymbol{\beta}$
- Guided by 2D foundation models (e.g., SAM)

**Coarse-grained semantic feature field** $\mathcal{F}_S$:
- Predicted at a resolution downsampled by a factor of $S$, substantially reducing the number of primitives
- Retains **uncompressed** semantic features $\boldsymbol{f}_S \in \mathbb{R}^M$
- A small number of primitives suffices to encode complete semantics (due to intra-instance semantic consistency)

### Selective Gaussian Mechanism (SGM)

An importance score $\beta_i$ is predicted for each primitive and multiplied into the opacity to modify alpha blending:

$$\boldsymbol{c} = \sum_{i=1}^n \boldsymbol{c}_i \boldsymbol{\alpha}_i \boldsymbol{\beta}_i \prod_{j=1}^{i-1}(1 - \boldsymbol{\alpha}_j \boldsymbol{\beta}_j)$$

A Leaky ReLU-style thresholding is applied:
$$\beta_i = \begin{cases} \beta_i & \text{if } \beta_i > \tau \\ \beta_i \times 10^{-3} & \text{if } \beta_i < \tau \end{cases}$$

A BCE loss combined with L1 regularization drives $\beta_i$ toward binary values of 0 or 1:
$$\mathcal{L}_I = \mathcal{L}_{BCE}(\boldsymbol{S}, \hat{\boldsymbol{S}}) + \frac{1}{\|\boldsymbol{S}\|}\sum_{\beta_i \in \boldsymbol{S}} \beta_i$$

### 3D Geometry Prediction

A pure ViT encoder-decoder is adopted without geometric priors. Scale ambiguity is resolved by injecting camera intrinsics (without depth supervision).

## Key Experimental Results

### ScanNet Semantic 3D Reconstruction

| Method | Feed-forward | Source mIoU↑ | Target mIoU↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------------|-------------|-------------|-------|-------|--------|
| L-Seg | ✗ | 0.5541 | 0.5558 | N/A | N/A | N/A |
| NeRF-DFF | ✗ | 0.5381 | 0.5137 | 22.49 | 0.765 | 0.283 |
| Feature-3DGS | ✗ | 0.4992 | 0.3223 | 17.96 | 0.581 | 0.489 |
| NoPoSplat | ✔ | N/A | N/A | 25.70 | 0.816 | 0.188 |
| LSM | ✔ | 0.5141 | 0.5104 | 24.12 | 0.796 | 0.253 |
| SpatialSplat-Lite | ✔ | 0.5272 | 0.5265 | 25.45 | 0.803 | 0.204 |
| **SpatialSplat** | ✔ | **0.5593** | **0.5587** | **25.46** | **0.805** | **0.205** |

### Parameter Efficiency

SpatialSplat uses only 40% of the representation parameters of the baseline while outperforming it across all metrics.

### Key Findings

1. The dual-field architecture achieves superior semantic segmentation and rendering quality with only 40% of the parameters.
2. The selective Gaussian mechanism effectively identifies and removes redundant primitives without requiring geometric priors.
3. Coarse-grained uncompressed semantics outperform fine-grained compressed semantics, validating the "uncompressed but sparse" strategy over "compressed but dense."
4. This constitutes the first feed-forward 3DGS framework that jointly learns semantic and instance priors.

## Highlights & Insights

1. **Decoupled semantic representation** — The decomposition into coarse semantics and fine-grained instance features is both novel and efficient.
2. **Counterintuitive choice of no compression** — Retaining full semantic features on a small number of primitives proves more effective than compressing and distributing them across all primitives.
3. **Redundancy identification from images** — This bypasses the need for precise camera extrinsics to detect overlapping regions.
4. **No 3D supervision** — Learning is entirely guided by 2D foundation models.

## Limitations & Future Work

- The downsampling rate $S$ for the coarse semantic field must be set in advance.
- Segmentation accuracy at instance boundaries is constrained by the quality of the 2D foundation model.
- The importance score threshold $\tau$ requires manual tuning.

## Related Work & Insights

- **Feed-forward 3DGS**: pixelSplat, MVSplat, NoPoSplat
- **Feature field distillation**: LERF, LangSplat, Feature-3DGS, LSM
- **Compact 3DGS**: Scaffold-GS, HAC, LightGaussian

## Rating

- Novelty: ⭐⭐⭐⭐ (dual-field architecture + selective Gaussians)
- Technical Depth: ⭐⭐⭐⭐ (complete SGM design + loss formulation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (comprehensive comparison with diverse baselines)
- Practical Value: ⭐⭐⭐⭐⭐ (60% parameter reduction with high deployment utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sparfels: Fast Reconstruction from Sparse Unposed Imagery](sparfels_fast_reconstruction_from_sparse_unposed_imagery.md)
- [\[ICCV 2025\] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](../../ICLR2026/3d_vision/ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)

</div>

<!-- RELATED:END -->
