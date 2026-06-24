---
title: >-
  [Paper Note] HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] This work utilizes structured binary hash-grids to establish spatial context relationships for unordered 3DGS anchors. Through conditional probability modeling and adaptive quantization, it achieves efficient entropy coding, reaching a **75×** compression rate compared to vanilla 3DGS while maintaining or even improving rendering quality.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Model Compression"
  - "Context Modeling"
  - "Entropy Coding"
  - "Hash-grid"
date: 2026-05-08
content_hash: 6b9655be3eed4499
---

# HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression

**Conference**: ECCV 2024  
**arXiv**: [2403.14530](https://arxiv.org/abs/2403.14530)  
**Code**: [https://github.com/YihangChen-ee/HAC](https://github.com/YihangChen-ee/HAC)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Model Compression, Context Modeling, Entropy Coding, Hash-grid

## TL;DR

This work utilizes structured binary hash-grids to establish spatial context relationships for unordered 3DGS anchors. Through conditional probability modeling and adaptive quantization, it achieves efficient entropy coding, reaching a **75×** compression rate compared to vanilla 3DGS while maintaining or even improving rendering quality.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the mainstream method for novel view synthesis due to its high fidelity and real-time rendering speed. However, it requires a large number of Gaussian primitives (in the millions) to represent a scene, leading to storage overheads of up to several gigabytes.

**Limitations of Prior Work**: Existing 3DGS compression methods primarily focus on the parameter "values" themselves (e.g., pruning, vector quantization), ignoring the **structural relationship redundancy** among Gaussian primitives. Scaffold-GS introduces anchor-based clustered Gaussians but still processes each anchor independently.

**Key Challenge**: The point-cloud nature of 3DGS makes the Gaussian primitives sparse and unorganized, making it difficult to directly exploit spatial structural relationships for compression, unlike NeRF feature grids.

**Goal**: How to mine the spatial consistency among unordered anchors and establish an effective context model to significantly compress the 3DGS representation.

**Key Insight**: Inspired by NeRF series using feature grids to represent 3D space, this work explores the mutual information relationship between unordered anchor attributes and structured hash-grids.

**Core Idea**: Jointly learn a binarized hash-grid as the context for anchor attributes, achieving efficient entropy coding compression through conditional probability estimation.

## Method

### Overall Architecture

HAC is built upon Scaffold-GS and consists of three levels overall:
- **Bottom level**: The anchor-Gaussian structure of Scaffold-GS, where anchor attributes $\mathcal{A} = \{\mathbf{f}^a, \mathbf{l}, \mathbf{o}\}$ predict Gaussian attributes via an MLP.
- **Middle level**: A jointly learned binary hash-grid $\mathcal{H}$, which is interpolated at any anchor position $\mathbf{x}^a$ to obtain the hash feature $\mathbf{f}^h$.
- **Top level**: A context model (MLP) taking $\mathbf{f}^h$ as input to predict the distribution parameters of anchor attributes for arithmetic coding.

The core formula is the conditional probability decomposition:

$$p(\mathcal{A}, \mathbf{x}^a, \mathcal{H}) = p(\mathcal{A}|\mathbf{x}^a, \mathcal{H}) \times p(\mathbf{x}^a, \mathcal{H}) \sim p(\mathcal{A}|\mathbf{f}^h) \times p(\mathcal{H})$$

Key insight: Instead of directly replacing anchor features with hash features (which would lead to quality degradation), they are used as **context** to estimate the probability distribution of anchor attributes.

### Key Designs

1. **Adaptive Quantization Module (AQM)**: Anchor attributes must be quantized into a finite set for entropy coding. Since the numerical ranges of different attributes (feature $\mathbf{f}^a$, scale $\mathbf{l}$, offset $\mathbf{o}$) differ significantly, a fixed step size is not applicable. AQM predicts the quantization step size adjustment factor $\mathbf{r}$ from $\mathbf{f}^h$ via a context MLP:

$$\mathbf{q}_i = Q_0 \times (1 + \text{Tanh}(\mathbf{r}_i)), \quad \mathbf{r}_i = \text{MLP}_q(\mathbf{f}^h_i)$$

The quantization step size is constrained within the range $(0, 2Q_0)$, where $Q_0$ is set to 1, 0.001, and 0.2 for $\mathbf{f}^a$, $\mathbf{l}$, and $\mathbf{o}$, respectively. Noise-adding approximation is used during training, and rounding is used during testing.

2. **Gaussian Distribution Probability Modeling**: Statistical analysis shows that anchor attributes approximately follow a Gaussian distribution. The context MLP independently predicts $\boldsymbol{\mu}_i$ and $\boldsymbol{\sigma}_i$ for each anchor from $\mathbf{f}^h$ to calculate the probability that the quantized attribute $\hat{\mathbf{f}}_i$ falls into the quantization interval:

$$p(\hat{\mathbf{f}}_i) = \Phi_{\boldsymbol{\mu}_i, \boldsymbol{\sigma}_i}\left(\hat{\mathbf{f}}_i + \frac{1}{2}\mathbf{q}_i\right) - \Phi_{\boldsymbol{\mu}_i, \boldsymbol{\sigma}_i}\left(\hat{\mathbf{f}}_i - \frac{1}{2}\mathbf{q}_i\right)$$

where $\Phi$ is the Gaussian CDF. High probability implies low entropy, i.e., fewer coding bits.

3. **Adaptive Offset Mask**: Statistical analysis shows that the offset $\mathbf{o}$ has an impulse distribution at zero, indicating a large number of redundant Gaussians. A binary mask with Straight-Through Estimator (STE) is used to prune ineffective offsets. If all offsets of an anchor are pruned, the entire anchor is removed.

4. **Hash-grid Compression**: Hash table parameters are binarized to $\{-1, +1\}$ and encoded with Arithmetic Encoding (AE) by calculating the occurrence frequency $h_f$ of "+1". A hybrid 3D-2D structure is adopted: 12-layer 3D embedding (resolution 16~512) + 4-layer 2D embedding (resolution 128~1024) with feature dimension $D^h=4$.

### Loss & Training

The total loss is a weighted sum of three parts:

$$\mathcal{L} = L_{\text{Scaffold}} + \lambda_e \frac{1}{N(D^a+6+3K)}(L_{\text{entropy}} + L_{\text{hash}}) + \lambda_m L_m$$

- $L_{\text{Scaffold}}$: Original rendering loss of Scaffold-GS
- $L_{\text{entropy}}$: Entropy loss of anchor attributes $\sum -\log_2 p(\hat{f}_{i,j})$
- $L_{\text{hash}}$: Estimated bit consumption of the hash-grid
- $L_m$: Regularization term for the offset mask

**Multi-stage Training**: Iterations 0-3K are for vanilla Scaffold-GS training; 3K-10K introduces noise-adding quantization adaptation; after 10K, the hash-grid and entropy constraint are fully integrated. $\lambda_e$ is adjusted from $5\times10^{-4}$ to $4\times10^{-3}$ to control the compression rate.

## Key Experimental Results

### Main Results

| Method | Synthetic-NeRF PSNR↑ | Size (MB)↓ | Mip-NeRF360 PSNR↑ | Size (MB)↓ | BungeeNeRF PSNR↑ | Size (MB)↓ |
|------|------|------|------|------|------|------|
| 3DGS | 33.80 | 68.46 | 27.49 | 744.7 | 24.87 | 1616 |
| Scaffold-GS | 33.41 | 19.36 | 27.50 | 253.9 | 26.62 | 183.0 |
| Lee et al. | 33.33 | 5.54 | 27.08 | 48.80 | 23.36 | 82.60 |
| Compressed3D | 32.94 | 3.68 | 26.98 | 28.80 | 24.13 | 55.79 |
| **HAC-low** | **33.24** | **1.18** | **27.53** | **15.26** | **26.48** | **18.49** |
| **HAC-high** | **33.71** | **1.86** | **27.77** | **21.87** | **27.08** | **29.72** |

### Ablation Study

| Component | BungeeNeRF Results | Synthetic-NeRF Results |
|------|------|------|
| Full HAC | Optimal RD curve | Optimal RD curve |
| Remove hash-grid mutual information (set to zero) | Bit consumption surges, probability degenerates to unconditional $p(\mathcal{A})$ | Same as left |
| Remove AQM | Quality drops significantly in high bitrate / complex scenes | Fidelity loss |
| Remove offset mask | Reduced bitrate savings in simple scenes / low bitrates | Removes a large amount of positionally redundant space |

### Key Findings

- The high-fidelity mode of HAC even exceeds the Scaffold-GS baseline in PSNR, which is attributed to the regularization effect of the entropy loss and the increased anchor feature dimension.
- The hash-grid can be completely removed during inference without affecting rendering FPS (HAC achieves 283 FPS on BungeeNeRF vs. 232 FPS for Scaffold-GS).
- Bit allocation visualization shows that complex texture regions are allocated more total bits, but the average bit consumption per anchor is actually smoother, validating the spatial consistency assumption.

## Highlights & Insights

- **Pioneering Context Modeling for 3DGS Compression**: Transferred the mature concept of context modeling from image/video compression to 3D Gaussian compression.
- **Compression without Modifying the Original Structure**: Context modeling is only utilized for probability estimation during the encoding and decoding stages and is entirely removed during rendering, ensuring that speed and quality upper bounds are unaffected.
- **Dual Role of the Binary Hash-grid**: It acts as both the source of context signals and is itself extremely easy to compress (requiring only storing 0/1 frequencies).

## Limitations & Future Work

- Training time increases by approximately 0.9× compared to Scaffold-GS (BungeeNeRF: 15.1 min vs. 27.6 min).
- Single-threaded CPU execution of AE during encoding/decoding is a bottleneck (taking 26.7 seconds on BungeeNeRF).
- Anchor positions $\mathbf{x}^a$ are directly stored in 32-bit format and are not included in the entropy constraint.
- Exploring stronger probability models (e.g., Gaussian mixture models) instead of a single Gaussian distribution could be a direction for future work.

## Related Work & Insights

- **Scaffold-GS**: Provides the hierarchical anchor-Gaussian structure framework, making context modeling possible.
- **Instant-NGP / CNC**: The compression schemes for hash-grids can be directly reused.
- **Context Models in Image Compression**: The core idea of conditional probability modeling is directly derived from learned image compression.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introduces context coding to 3DGS compression for the first time; the idea of bridging unordered anchors using a hash-grid is clever.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated on 5 datasets with complete ablation studies, bit allocation visualizations, and RD curves.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic, progressing step-by-step from mutual information validation to conditional probability modeling.
- **Value**: ⭐⭐⭐⭐⭐ — The 75× compression rate makes large-scale deployment of 3DGS highly feasible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [\[ECCV 2024\] GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](../../CVPR2026/3d_vision/off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[ICCV 2025\] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](../../ICCV2025/3d_vision/evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)
- [\[ECCV 2024\] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)

</div>

<!-- RELATED:END -->
