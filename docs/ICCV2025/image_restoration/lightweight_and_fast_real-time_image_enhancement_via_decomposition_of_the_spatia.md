---
title: >-
  [Paper Note] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables
description: >-
  [ICCV 2025][Image Restoration][3D LUT] By decomposing 3D LUTs into linear combinations of 2D LUTs followed by SVD, and adopting a cache-efficient spatial feature fusion structure, the proposed method achieves spatially-aware image enhancement while reducing model parameters by 84% and accelerating 4K inference by 2.8×.
tags:
  - ICCV 2025
  - Image Restoration
  - 3D LUT
  - Singular Value Decomposition
  - Spatial Awareness
  - Cache Efficiency
  - Image Enhancement
date: 2026-05-08
content_hash: b8491ad9bb2ca1e9
---

# Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables

**Conference**: ICCV 2025
**arXiv**: [2508.16121](https://arxiv.org/abs/2508.16121)
**Code**: [https://github.com/WontaeaeKim/SVDLUT](https://github.com/WontaeaeKim/SVDLUT)
**Area**: Image Restoration / Image Enhancement
**Keywords**: 3D LUT, Singular Value Decomposition, Spatial Awareness, Cache Efficiency, Image Enhancement

## TL;DR

By decomposing 3D LUTs into linear combinations of 2D LUTs followed by SVD, and adopting a cache-efficient spatial feature fusion structure, the proposed method achieves spatially-aware image enhancement while reducing model parameters by 84% and accelerating 4K inference by 2.8×.

## Background & Motivation

3D lookup table (3D LUT)-based image enhancement has attracted wide attention due to its efficient interpolation operations, yet suffers from a fundamental tension:

**Lack of spatial awareness**: 3D LUTs transform color values point-wise without spatial context.

**Cost of spatial-aware methods**: SA-3DLUT and SABLUT introduce additional modules for spatial information, but significantly increase parameter count and inference time, especially at high resolutions (480p: 1.2 ms → 4K: 3.6 ms).

The authors draw on two key observations:
- 3D LUT utilization is extremely low (<10% of vertices are actually referenced), with high-frequency accesses concentrated near the diagonal.
- The inference bottleneck of existing spatial-aware methods lies in cache-unfriendly reads and writes of intermediate outputs during slicing.

## Method

### Overall Architecture

The SVDLUT framework consists of:
1. A backbone network that extracts context features from a downscaled image: $\rho = B(\hat{X})$
2. A generator that produces LUT components, a bilateral grid, and corresponding weights
3. Spatially-aware enhancement via decomposed 2D LUT transforms and 2D bilateral grid slicing

### Key Designs

1. **3D→2D Dimensionality Reduction**

   By analyzing 3D LUT utilization on the FiveK dataset (fewer than 10% of vertices are referenced, with high-frequency accesses along the diagonal), the method replaces 3D LUTs with a linear combination of 2D LUTs:

   $t_{rgb}^c \to w_{rg}^c \cdot t_{rg}^c + w_{rb}^c \cdot t_{rb}^c + w_{gb}^c \cdot t_{gb}^c + b^c$

   Color enhancement becomes:
   $Y_{(c,x,y)} = w_{rg}^c \cdot I_{bi}(\bar{X}_{(x,y)}, t_{rg}^c) + w_{rb}^c \cdot I_{bi}(\bar{X}_{(x,y)}, t_{rb}^c) + w_{gb}^c \cdot I_{bi}(\bar{X}_{(x,y)}, t_{gb}^c) + b^c$

   where $I_{bi}(\cdot)$ denotes bilinear interpolation. The 3D bilateral grid is similarly decomposed into 2D. Experiments confirm that the 3D→2D transition incurs negligible PSNR loss (25.68 vs. 25.67) while reducing model size by 84%.

2. **SVD-based Further Compression**

   The 2D LUT is further decomposed via singular value decomposition:
   $T^{2D} = U \cdot S \cdot V^T$

   The generator directly predicts LUT components $(S, U, V^T)$ rather than the full 2D LUT. A toy experiment shows that retaining 8 singular values causes almost no performance drop, owing to the monotonic and simple structure of LUTs. In contrast, the bilateral grid encodes spatially complex information and is not amenable to SVD; therefore, SVD is applied only to the LUT.

   SVD reduces LUT parameters by a further ~88% relative to the 3D LUT baseline.

3. **Cache-efficient Spatial Feature Fusion**

   The inference bottleneck of prior methods stems from slicing producing high-resolution intermediate outputs $f_s$, which are then fused via $1\times1$ convolution, causing frequent high-to-low memory traffic at 4K resolution.

   The proposed solution merges slicing and LUT transform into a single step, eliminating intermediate outputs $f_s$ and $\bar{X}$, reusing LUT index computations, and removing the $1\times1$ convolution (replaced by the decomposed weighted sum):

   $Y_{(c,x,y)} = \text{Transform}_{2D}^c(X_{(c,x,y)}, T^{2D}) + \sum_{k=0}^{K/3-1} \text{Slicing}_{2D}^{c'_{c+3k}}(X_{(c,x,y)}, G^{2D})$

   This reduces 4K inference time from 3.84 ms to 1.38 ms while also improving PSNR by 0.08 dB.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{mse} + \lambda_c \cdot \mathcal{L}_c + \lambda_p \cdot \mathcal{L}_p$$

- $\mathcal{L}_{mse}$: mean squared error for reconstruction fidelity
- $\mathcal{L}_c$: CIE94 LAB-space color difference loss ($\lambda_c = 0.005$)
- $\mathcal{L}_p$: AlexNet-based LPIPS perceptual loss ($\lambda_p = 0.05$)

Training uses the Adam optimizer for 400 epochs with an initial learning rate of $1 \times 10^{-4}$, decayed by a factor of 0.1 every 100 epochs.

## Key Experimental Results

### Main Results

FiveK dataset — Photo Retouch task:

| Method | Params | PSNR(480p) | SSIM(480p) | Time 480p (ms) | PSNR(4K) | SSIM(4K) | Time 4K (ms) |
|--------|--------|------------|------------|----------------|----------|----------|--------------|
| 3D LUT | 593.5K | 25.29 | 0.923 | 1.02 | 25.25 | 0.932 | 1.04 |
| SA-3DLUT | 4.5M | 25.50 | — | 2.27 | — | — | 4.39 |
| AdaInt | 619.7K | 25.49 | 0.926 | 1.29 | 25.48 | 0.934 | 1.59 |
| SABLUT | 463.7K | 25.66 | 0.930 | 1.20 | 25.66 | 0.937 | 3.64 |
| **SVDLUT** | **160.5K** | **25.76** | **0.931** | 1.37 | **25.69** | **0.938** | **1.38** |

SVDLUT has only 160.5K parameters (65% fewer than SABLUT), achieves 4K inference in 1.38 ms (2.6× faster than SABLUT), and attains the highest PSNR overall. Top or near-top PSNR is also maintained on the PPR10K dataset.

### Ablation Study

**LUT and bilateral grid dimensionality combinations** (FiveK dataset):

| | 3D Grid | 2D Grid | 1D Grid |
|---|---------|---------|---------|
| 3D LUT | 25.68 (1.3M) | 25.67 (1.1M) | 25.54 (1.0M) |
| 2D LUT | 25.67 (421.5K) | **25.68 (205.3K)** | 25.53 (161.2K) |
| 1D LUT | 25.37 (335.9K) | 25.53 (119.8K) | 25.22 (75.7K) |

The 2D LUT + 2D Grid combination matches 3D+3D performance while reducing parameters by 84%. Reducing to 1D incurs a notable performance drop.

**Component ablation** (FiveK dataset):

| Configuration | PSNR | Params |
|---------------|------|--------|
| Grid only | 25.21 | 112.9K |
| LUT only | 25.49 | 109.3K |
| $1\times1$ conv fusion | 25.68 | 160.5K |
| **Cache-efficient structure** | **25.76** | 160.5K |

### Key Findings

- **3D LUTs are highly redundant**: utilization is below 10%, with accesses concentrated along the diagonal.
- **2D is the optimal trade-off**: 1D has higher utilization but insufficient capacity (saturation); 2D balances utilization and expressiveness.
- **SVD is effective for LUTs but not grids**: LUTs have monotonic and simple structure, whereas grids encode spatially complex information.
- **Cache efficiency is critical at high resolution**: merging operations to eliminate intermediate outputs makes 4K inference time nearly identical to 480p.
- Slicing handles global/local adjustments; LUT Transform handles color correlation adjustments — each with a distinct role.

## Highlights & Insights

1. **Motivation grounded in utilization analysis**: the decomposition is not a naive compression but is derived from the empirical finding that the majority of 3D LUT vertices remain unused.
2. **Three-stage progressive compression**: 3D→2D reduction → SVD decomposition → cache-efficient fusion, each step experimentally validated.
3. **Near-zero latency growth from 480p to 4K**: inference times are nearly identical (1.37 ms vs. 1.38 ms), directly addressing the core bottleneck of real-time high-resolution enhancement.
4. **Strong visualization analysis**: clearly demonstrates the distinct roles of slicing (local adjustments) and LUT transform (color adjustments).

## Limitations & Future Work

- Validation is limited to image enhancement; extension to other LUT-based tasks (e.g., super-resolution) remains unexplored.
- The SVD rank is selected empirically; adaptive rank selection may yield further gains.
- The backbone network is not optimized, leaving room for additional lightweighting of the overall framework.

## Related Work & Insights

- The 3D→2D→SVD decomposition paradigm may inspire compression of other high-dimensional table structures.
- Cache efficiency is an underappreciated but critical factor in high-resolution image processing.
- The CUDA-based merged slicing and LUT transform implementation is a noteworthy engineering contribution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The decomposition scheme derived from LUT utilization analysis is both novel and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two mainstream benchmarks (FiveK and PPR10K), detailed ablations, and visualization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic with highly informative figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — Real-time 4K enhancement, open-source code, and high deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices](mobileie_an_extremely_lightweight_and_effective_convnet_for_real-time_image_enha.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[ICCV 2025\] IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution](im-lut_interpolation_mixing_look-up_tables_for_image_super-resolution.md)
- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ICCV 2025\] CWNet: Causal Wavelet Network for Low-Light Image Enhancement](cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)

</div>

<!-- RELATED:END -->
