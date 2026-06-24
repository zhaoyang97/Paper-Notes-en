---
title: >-
  [Paper Note] GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][2D Gaussian Splatting] This paper proposes GaussianImage, which represents the first attempt to apply 2D Gaussian Splatting to image representation and compression. By utilizing compact 8-parameter 2D Gaussians and an accumulative summation rasterization algorithm, it achieves a decoding speed of over 2000 FPS, while matching the representation quality and compression performance of INR-based methods.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "2D Gaussian Splatting"
  - "Image Representation"
  - "Image Compression"
  - "Vector Quantization"
  - "Fast Rendering"
date: 2026-05-08
content_hash: 8cb204c77594803a
---

# GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting

**Conference**: ECCV 2024  
**arXiv**: [2403.08551](https://arxiv.org/abs/2403.08551)  
**Code**: [https://github.com/Xinjie-Q/GaussianImage](https://github.com/Xinjie-Q/GaussianImage)  
**Area**: 3D Vision  
**Keywords**: 2D Gaussian Splatting, Image Representation, Image Compression, Vector Quantization, Fast Rendering

## TL;DR

This paper proposes GaussianImage, which represents the first attempt to apply 2D Gaussian Splatting to image representation and compression. By utilizing compact 8-parameter 2D Gaussians and an accumulative summation rasterization algorithm, it achieves a decoding speed of over 2000 FPS, while matching the representation quality and compression performance of INR-based methods.

## Background & Motivation

**Background**: Implicit Neural Representations (INRs) have achieved great success in image representation and compression, enabling high-quality image reconstruction with compact networks. However, two main types of methods exist, each with its limitations: MLP-based INRs (such as SIREN, WIRE) exhibit slow training and slow rendering, whereas Feature grid-based INRs (such as I-NGP, NeuRBF) accelerate training and inference but require substantial GPU memory.

**Limitations of Prior Work**: INR methods are difficult to deploy on low-end devices due to either high memory requirements (feature grid methods require 1500–2900 MiB) or slow rendering speeds (WIRE runs at only 11 FPS). This severely limits the practical application of neural image codecs.

**Key Challenge**: High-quality image representation requires a large number of parameters or computation, whereas practical deployment demands low memory footprint and fast decoding. How can explicit representations be leveraged to break the implicit bottleneck of INRs?

**Goal**: To develop an image representation and compression technique that is highly training-efficient, memory-friendly, and ultra-fast in decoding.

**Key Insight**: 3D Gaussian Splatting has demonstrated speed advantages brought by explicit representation and parallel rasterization in 3D scene reconstruction. Can this be adapted to the 2D image representation task? Direct adaptation faces three major challenges: 3D Gaussians contain too many parameters (59 per point), $\alpha$-blending requires depth sorting (which is absent in 2D images), and early termination leads to underutilization of Gaussians.

**Core Idea**: Replace 59-parameter 3D Gaussians with compact 2D Gaussians possessing only 8 parameters, and replace depth-sorted $\alpha$-blending with order-independent accumulative summation to achieve ultra-fast image representation.

## Method

### Overall Architecture

GaussianImage consists of two stages:
1. **Image Representation**: Fitting the image with a set of 2D Gaussian points, where each Gaussian has only 8 parameters (position 2 + covariance 3 + weighted color 3).
2. **Image Compression**: Performing quantization-aware fine-tuning and coding on the fitted Gaussian attributes, with optional bits-back coding to further reduce the bitrate.

### Key Designs

1. **Compact 2D Gaussian Representation**: Each 2D Gaussian is represented by a position $\boldsymbol{\mu} \in \mathbb{R}^2$, a covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{2 \times 2}$, a color $\boldsymbol{c} \in \mathbb{R}^3$, and an opacity $o \in \mathbb{R}$. The covariance matrix is decomposed via Cholesky decomposition to ensure positive-definiteness:

    $\boldsymbol{\Sigma} = \boldsymbol{L}\boldsymbol{L}^T$

   The Cholesky vector $\boldsymbol{l} = \{l_1, l_2, l_3\}$ is used to represent the lower triangular elements. The basic 2D Gaussian requires only 9 parameters in total, achieving a $6.5\times$ compression compared to the 59 parameters of a 3D Gaussian.

   Alternatively, rotation-scaling decomposition can be used: $\boldsymbol{\Sigma} = (\boldsymbol{RS})(\boldsymbol{RS})^T$, with rotation angle $\theta$ and scaling factors $s_1, s_2$, also requiring 3 parameters.

2. **Accumulative Summation Rasterization**: The $\alpha$-blending in 3D GS requires sorting Gaussians by depth and computing the cumulative transmittance $T_n$, which is not applicable to 2D images without depth. This paper proposes a direct weighted summation:

    $\boldsymbol{C}_i = \sum_{n \in \mathcal{N}} \boldsymbol{c}_n \cdot o_n \cdot \exp(-\sigma_n), \quad \sigma_n = \frac{1}{2}\boldsymbol{d}_n^T \boldsymbol{\Sigma}^{-1} \boldsymbol{d}_n$

   Furthermore, the color $\boldsymbol{c}_n$ and opacity $o_n$ are merged into a weighted color coefficient $\boldsymbol{c}_n' \in \mathbb{R}^3$ (no longer constrained to the range $[0, 1]$):

    $\boldsymbol{C}_i = \sum_{n \in \mathcal{N}} \boldsymbol{c}_n' \cdot \exp(-\sigma_n)$

   As a result, each 2D Gaussian requires only **8 parameters** (position 2 + covariance 3 + weighted color 3), achieving a compression ratio of $7.375\times$.

   Three advantages: (a) Insensitive to Gaussian ordering, eliminating the need for sorting; (b) Sidesteps the sequential computation of cumulative transmittance $T_n$, accelerating training and inference; (c) All Gaussians covering the pixel participate in rendering, utilizing the information fully.

3. **Image Compression Pipeline**: Different quantization strategies are applied to different attributes:

    - **Positions**: 16-bit float precision (highly sensitive to quantization).
    - **Covariance parameters**: $b$-bit (default 6-bit) asymmetric quantization with learned scaling factors $\gamma_i$ and offsets $\beta_i$:
   
    $\hat{l}_i^n = \lfloor \text{clamp}(\frac{l_i^n - \beta_i}{\gamma_i}, 0, 2^b - 1) \rceil$
   
    - **Weighted color coefficients**: Residual Vector Quantization (RVQ) with $M=2$ stages and a codebook size of $B=8$:
   
    $\hat{\boldsymbol{c}}_n^{\prime m} = \sum_{k=1}^{m} \mathcal{C}^k[i^k]$
   
    - **Optional Partial Bits-back Coding**: Taking advantage of the permutation invariance of the Gaussian point set, the first $K$ Gaussians are encoded using ordinary entropy coding to serve as initial bits. The remaining $N-K$ Gaussians are encoded using bits-back coding, saving $\log(N-K)! - \log(N-K)$ bits.

### Loss & Training

- **Image Representation**: L2 loss is used to optimize Gaussian parameters with the Adan optimizer. The initial learning rate is $1 \times 10^{-3}$, halved every 20,000 steps, with a total of 50,000 steps.
- **Image Compression**: $\mathcal{L} = \mathcal{L}_{rec} + \lambda \mathcal{L}_c$, where $\mathcal{L}_c$ is the commitment loss of the RVQ codebooks.
- Adaptive density control from 3D GS is omitted as there are no empty regions in the 2D image space.
- The codebook is initialized via K-means (5 iterations) and updated using exponential moving average (EMA) during training.
- Implemented on top of the `gsplat` library with custom CUDA kernels.

## Key Experimental Results

### Main Results — Image Representation (Kodak Dataset)

| Method | PSNR↑ | MS-SSIM↑ | Training Time (s)↓ | FPS↑ | VRAM (MiB)↓ | Parameters (K)↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| WIRE | 41.47 | 0.9939 | 14338 | 11 | 2619 | 137 |
| SIREN | 40.83 | 0.9960 | 6582 | 29 | 1809 | 273 |
| I-NGP| 43.88 | 0.9976 | 491  | 1297| 1525 | 300 |
| NeuRBF| 43.78| 0.9964 | 992  | 663 | 2091 | 337 |
| 3D GS| 43.69 | 0.9991 | 340  | 859 | 557  | 3540|
| **Ours** | **44.08** | **0.9985** | **107** | **2092** | **419** | 560 |

Key metrics: Training speed is $4.6\times$ faster than I-NGP, rendering speed reaches $2092$ FPS ($188\times$ that of WIRE), and VRAM is only $419$ MiB (the lowest).

### Main Results — Image Compression Complexity Comparison (DIV2K)

| Method | Low bpp PSNR↑ | Encoding FPS | **Decoding FPS↑** |
|------|:---:|:---:|:---:|
| JPEG | 25.29 | 609 | 615 |
| JPEG2000 | 27.28 | 3.5 | 4.3 |
| Ballé17 | 27.72 | 21 | 19 |
| Ballé18 | 28.75 | 17 | 16 |
| COIN | 25.80 | 5.3e-4 | 166 |
| **Ours** | 25.66 | 4.1e-3 | **1971** |

Decoding speed is $12\times$ faster than COIN and $3\times$ faster than JPEG, reaching approximately **2000 FPS**.

### Ablation Study

| Configuration | PSNR↑ | Training Time (s)↓ | FPS↑ | Parameters (K)↓ | Description |
|------|:---:|:---:|:---:|:---:|------|
| 3D GS (L1+SSIM) | 37.75 | 285 | 1067 | 1770 | Baseline |
| 3D GS (L2) | 37.41 | 198 | 1190 | 1770 | Swapping to L2 for acceleration |
| 2D GS (w/o AR, w/o M) | 37.89 | 105 | 2340 | 270 | 2D Gaussian + $\alpha$-blending |
| + Accumulative Summation (AR) | **38.69** | 99 | 2555 | 270 | +0.8dB! |
| + Merge Color & Opacity (M) | 38.57 | 91 | 2565 | **240** | 10% reduction in parameters |

### Quantization Strategy Ablation (BD Metrics, Anchor is Final Scheme)

| Variant | BD-PSNR (dB) | BD-rate (%) |
|------|:---:|:---:|
| Final Scheme (Anchor) | 0 | 0 |
| w/o $\mathcal{L}_c$ + RVQ + 6-bit | -3.145 | +333% |
| w/o $\mathcal{L}_c$ + w/o RVQ + 6-bit | -0.159 | +7.02% |
| w/o $\mathcal{L}_c$ + w/o RVQ + 8-bit | -0.195 | +11.69% |

### Key Findings

- Accumulative summation instead of $\alpha$-blending contributes the most substantial performance improvement (+0.8 dB PSNR) while accelerating both training and inference.
- 2D Gaussians reduce the number of parameters by $6.5\times$ compared to 3D Gaussians (270K vs 1770K) but deliver better representation quality.
- L2 loss is optimal for this method (outperforming L1, SSIM, and their combinations), which differs from 3D GS where L1+SSIM is optimal.
- RVQ is crucial for compressing color attributes, as different Gaussians exhibit similar color vectors, making them highly suitable for codebook-based encoding.
- Cholesky decomposition and rotation-scaling decomposition are equivalent in representation capability, but differ in quantization robustness (Cholesky is used by default).

## Highlights & Insights

- **Elegant Adaptation from 3D to 2D**: Instead of simply fixing camera parameters, the author radically redesigns the Gaussian representation (2D projection, parameter merging) and the rasterization algorithm (accumulative summation), with clear motivations behind each step.
- **Clever Use of Permutation Invariance**: Since accumulative summation is order-independent, partial bits-back coding can leverage the $N!$ equivalent permutations of the Gaussian set to save bitrate, which is theoretically impossible under $\alpha$-blending.
- **Pareto-Optimal Performance-Efficiency**: Concurrently achieves optimal or sub-optimal performance across four key dimensions: representation quality (44.08 dB), training speed (107s), rendering speed (2092 FPS), and VRAM (419 MiB).
- **Insight to Discard Adaptive Density Control**: Since there are no empty regions in 2D image spaces, the split/clone strategies used in 3D GS are rendered unnecessary.

## Limitations & Future Work

- Compression performance in high-bitrate regions falls behind VAE-based methods (e.g., Ballé17/18) due to the lack of an autoregressive context model.
- Although bits-back coding offers good theoretical performance, its processing latency is high, which contradicts the goal of "ultra-fast codec".
- Currently, Gaussians are fitted independently for each image, making it impossible to exploit codes across image sequences (such as temporal redundancy in videos).
- As the image resolution scales up, the required number of Gaussian points increases; thus, scalability remains to be validated.
- Exploring adaptive numbers of Gaussians (dynamically adjusted based on image complexity) stands as a promising direction.

## Related Work & Insights

- The success of 3D Gaussian Splatting demonstrates the powerful capability of explicit representation joined with differentiable rasterization; this paper introduces this paradigm to the 2D image domain as pioneering work.
- Complimentary to INR methods: while INRs utilize continuous implicit functions, GaussianImage utilizes discrete explicit Gaussians, rendering the latter highly suitable for high-speed decoding scenarios.
- RVQ is widely used in audio coding (e.g., SoundStream), and this paper demonstrates its effectiveness for compressing the color attributes of 2D Gaussians.
- Future directions could combine this paradigm with autoregressive entropy models to improve compression performance, or extend it to video representation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Pioneered the 2D Gaussian Splatting image representation paradigm, featuring exquisitely designed accumulative summation and parameter merging.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on Kodak and DIV2K standard datasets with comprehensive representation/compression benchmarks and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, complete technical details, and a logical progression of adapting from 3D GS to 2D.
- **Value**: ⭐⭐⭐⭐⭐ The decoding speed of over 2000 FPS marks a new milestone in neural image codecs, indicating high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](../../AAAI2026/3d_vision/gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)
- [\[ECCV 2024\] HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression](hac_hash-grid_assisted_context_for_3d_gaussian_splatting_compression.md)

</div>

<!-- RELATED:END -->
