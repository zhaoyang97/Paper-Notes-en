---
title: >-
  [Paper Note] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][2D Gaussian Splatting] GaussianImage++ is proposed to achieve high-quality image representation and compression under limited 2D Gaussian primitives through a distortion-driven densification mechanism and content-aware Gaussian filters, while maintaining real-time decoding speed.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "2D Gaussian Splatting"
  - "Image Compression"
  - "Implicit Neural Representation"
  - "Densification Mechanism"
  - "Quantization-Aware Training"
date: 2026-05-08
content_hash: d3098792caddf610
---

# GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting

**Conference**: AAAI 2026  
**arXiv**: [2512.19108](https://arxiv.org/abs/2512.19108)  
**Code**: [https://github.com/Sweethyh/GaussianImage_plus](https://github.com/Sweethyh/GaussianImage_plus)  
**Area**: 3D Vision / Image Representation and Compression  
**Keywords**: 2D Gaussian Splatting, Image Compression, Implicit Neural Representation, Densification Mechanism, Quantization-Aware Training

## TL;DR

GaussianImage++ is proposed to achieve high-quality image representation and compression under limited 2D Gaussian primitives through a distortion-driven densification mechanism and content-aware Gaussian filters, while maintaining real-time decoding speed.

## Background & Motivation

### Background

Image representation and compression are core problems of visual data storage and transmission. Current mainstream solutions include:
- **Autoencoder-based neural compression** (e.g., Ballé18, ELIC): Excellent rate-distortion performance, but high decoding latency.
- **Implicit Neural Representation (INR)** (e.g., SIREN, COIN): Uses MLPs to fit pixel coordinate-to-color mapping, but has slow training speed and high memory footprints.
- **2D Gaussian Splatting (GS)**: GaussianImage pioneered the use of GS for 2D images, significantly reducing training time and memory.

### Limitations of Prior Work

1. **GaussianImage lacks a densification mechanism**: It cannot adaptively allocate Gaussian primitives based on image content, leading to a large number of under-reconstructed regions.
2. **Mirage utilizes the ADC of 3D GS**: This easily leads to uncontrollable growth of Gaussian numbers, resulting in OOM errors.
3. **LIG lacks compression**: It focuses on fitting large images but does not explore attribute compression, resulting in high storage overhead.
4. **3D GS compression methods cannot be directly migrated**: HAC and ContextGS are based on neural Gaussian (Scaffold), and their architectures do not match explicit 2D GS.

### Key Challenge

How to simultaneously achieve high visual fidelity and efficient compression under a **limited number of 2D Gaussian primitives**?

### Key Insight

Enhance 2D GS from three dimensions: (1) progressive distortion-driven densification to control Gaussian distribution; (2) content-aware filters to optimize Gaussian rendering quality; (3) attribute-separated learnable scalar quantization for efficient compression.

## Method

### Overall Architecture

The pipeline of GaussianImage++ is divided into two major stages:
1. **Image Representation**: Sparse Initialization → Periodic Distortion-Driven Densification → Content-Aware Filtering → Accumulation & Rasterization.
2. **Image Compression**: Overfit Gaussian attributes first → Quantization-Aware Training (QAT) fine-tuning → Encode into a compact bitstream.

Each 2D Gaussian is parameterized by position $\boldsymbol{\mu} \in \mathbb{R}^2$, covariance $\boldsymbol{\Sigma} \in \mathbb{R}^{2 \times 2}$, and color $\mathbf{c} \in \mathbb{R}^3$. The rendering formulation is:

$$G_i(\mathbf{x}) = \exp\left(-\frac{(\mathbf{x}-\boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}_i)}{2}\right)$$

$$\mathbf{C} = \sum_{i \in N} \mathbf{c}_i G_i(\mathbf{x})$$

### Key Designs

#### 1. Distortion-Driven Densification (D³)

**Function**: Progressively allocate Gaussian primitives to under-reconstructed regions.

**Mechanism**: A three-stage mechanism:

- **Sparse Initialization**: The initial number of Gaussians is set to $N_0 = M/2$ (with $M$ being the maximum number of Gaussians). Positions are uniformly randomly sampled within image coordinates, and colors are initialized to zero.
- **Gaussian Growth**: Every 5,000 iterations, new Gaussians are added at the top-k pixel locations with the largest reconstruction distortion. The quantity of new Gaussians is determined by a scheduler $\tau(t, N_t, M) = (M - N_t)/2$.
- **Gaussian Pruning**: Every 100 iterations, the positive semi-definiteness of the covariance matrix is checked, and invalid Gaussians are pruned.

**Design Motivation**: 3D GS's ADC relies on positional gradients, but in 2D scenarios, gradient changes are too minor to trigger effectively. This paper directly utilizes pixel-level distortion ($L_1$ loss) to determine densification positions, which is more direct and image-quality-oriented. The positions and colors of new Gaussians are obtained directly from high-distortion pixels of the original image:

$$\boldsymbol{\mu}_\Psi = \xi(\text{Top}_k(D(X, \hat{X})))$$
$$\mathbf{c}_\Psi = X(\xi(\text{Top}_k(D(X, \hat{X}))))$$

#### 2. Content-Aware Gaussian Filter (CAF)

**Function**: Apply adaptive-intensity low-pass filtering to each Gaussian primitive to reduce rendering holes and artifacts.

**Mechanism**: Apply a zero-mean Gaussian low-pass filter $h(x)$ to the original Gaussian kernel, where the variance vector $\mathbf{s} \in \mathbb{R}^{N_t}$ controls the filtering intensity of each Gaussian:

$$G_i'(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T(\boldsymbol{\Sigma}_i + s_i I)^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)}$$

Variance formula:
$$s_i = \frac{HW}{\alpha N_t} \quad (\text{newly added Gaussians})$$

**Design Motivation**: In the early stage of training when Gaussians are sparse ($N_t \ll HW$), filters with large variances expand the coverage area and reduce holes, yielding a coarse but recognizable image to guide optimization. As densification progresses, the filtering intensity of new Gaussians gradually decreases, focusing on details. Crucially, $\mathbf{s}$ does not increase storage overhead—the filtered covariance $\boldsymbol{\Sigma} + sI$ is stored directly.

#### 3. Compression Framework (Attribute-Separated Quantization)

**Function**: Apply quantization with different bit-depths to different attributes using a learnable scalar quantizer (LSQ+).

**Mechanism**:
- Position $\boldsymbol{\mu}$: 12-bit (highly sensitive to geometry, requiring high precision)
- Covariance $\boldsymbol{\Sigma}$: 10-bit
- Color $\mathbf{c}$: 6-bit

Quantization formula:
$$\bar{\mathbf{v}} = \lfloor \text{clip}(\frac{\mathbf{v} - \beta}{s}, 0, 2^b - 1) \rfloor, \quad \hat{\mathbf{v}} = \bar{\mathbf{v}} \cdot s + \beta$$

**Design Motivation**: Quantization-Aware Training (QAT) enables Gaussians to actively adapt their attributes to cope with quantization errors. Compared with FP16 or RVQ, LSQ+'s learnable offsets and scales achieve a better rate-distortion trade-off.

### Loss & Training

- Representation stage: $L_2$ loss, Adam optimizer, 50,000 iterations, learning rate of 0.18 (halved after 20,000 iterations).
- Compression stage: Quantization-aware fine-tuning is performed after 6,000 warm-up steps, with a quantizer learning rate of 0.001.

## Key Experimental Results

### Main Results

#### Image Representation (Kodak, 10k Gaussians)

| Method | PSNR↑ | MS-SSIM↑ | Params (M) | GPU Memory (MiB) | Rendering FPS |
|------|-------|----------|-----------|-------------|---------|
| Siren (INR) | 26.50 | 0.875 | 3.74 | 2044 | 977 |
| GaussianImage | 32.48 | 0.982 | 0.08 | 814 | 2009 |
| LIG | 31.00 | 0.975 | 0.08 | 832 | 1331 |
| **Ours** | **35.41** | **0.983** | 0.08 | 876 | **2216** |

#### Image Compression (Kodak, Low/High bpp)

| Method | Bpp | PSNR | Decoding FPS |
|------|-----|------|---------|
| JPEG | 0.22/1.03 | 23.8/32.8 | 377/148 |
| COIN | 0.17/0.98 | 24.9/27.4 | 769/344 |
| GaussianImage | 0.15/1.00 | 25.0/29.7 | 1827/1822 |
| **Ours** | 0.15/1.08 | **25.3/31.1** | **1839/1666** |

### Ablation Study

#### Component Ablation (Kodak)

| Configuration | PSNR Gain (vs GS Cholesky) | Description |
|------|--------------------------|------|
| + D³ alone | ~2dB | Densification alone contributes the most |
| + D³ + CAF | ~3dB | Synergy between the two brings further gains |
| vs LIG | ~4dB | Significant overall improvement |

#### Quantization Strategy Ablation

| Configuration (Position/Color) | BD-PSNR (dB) | BD-Rate (%) |
|------------------|-------------|------------|
| LSQ+/LSQ+ (Ours) | 0 | 0 |
| FP16/LSQ+ | -0.761 | +25.11% |
| FP16/RVQ | -2.471 | +138.88% |
| LSQ+/RVQ | -2.491 | +147.24% |

### Key Findings

1. $D^3$ densification is particularly effective when the number of Gaussians is small, as sparse Gaussians require more precise allocation.
2. CAF plays a critical role in the early training stages—generating a recognizable coarse image at $t=500$ (whereas the baseline exhibits massive holes).
3. Both components are effective and versatile across three different covariance parameterizations (Cholesky, RS, direct parameterization).
4. The decoding speed of the GS method far surpasses traditional and learned codecs ($>1800$ FPS vs. $\sim 150$ FPS of JPEG).

## Highlights & Insights

1. **Distortion-driven densification is highly intuitive**: Directly placing new Gaussians at the "worst" pixel locations is simple yet effective.
2. **The progressive decay strategy of CAF is elegant**: Expanding coverage in the early stage $\rightarrow$ refining in the later stage, which naturally synergizes with densification.
3. **General enhancement technology**: $D^3$ and CAF can act as plug-and-play modules applicable to other 2D GS methods.
4. **Distinct real-time decoding advantage**: Compared with the decoding latency of VAEs and INRs, the simple accumulative summation of GS exhibits a fundamental speed advantage.

## Limitations & Future Work

1. **Lagging behind SOTA learned codecs at high bitrates**: This is a common issue in current 2D GS compression, which requires more advanced entropy models.
2. **Encoding time is far from real-time**: The training and quantization processes are time-consuming, hindering physical deployment.
3. **Lack of adaptive bit allocation**: Currently, the same quantization configuration is applied to all images without adapting to image complexity.
4. Future work could explore extending $D^3$ and CAF to video GS scenarios.

## Related Work & Insights

- **GaussianImage** (Zhang et al., 2024): The first 2D GS image representation, serving as the direct baseline for ours.
- **3D GS ADC** (Kerbl et al., 2023): Gradient-based density control, which inspired $D^3$ but utilizes a different mechanism.
- **LSQ+** (Bhalgat et al., 2020): Low-bit quantization with learnable offset/scale, serving as the core tool for our compression.
- **COOL-CHIC** (Ladune et al., 2023): A hybrid INR compression method, which incurs decoding overhead due to the autoregressive entropy model.

## Rating

- Novelty: ⭐⭐⭐⭐ — The designs of $D^3$ and CAF are simple and effective, but the core idea (adding Gaussians at high-distortion locations) is somewhat intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covered dual datasets, multiple baselines, cross-method ablation, and quantization strategy ablation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and well-motivated arguments.
- Value: ⭐⭐⭐⭐ — Holds practical value as a general enhancement technology, but the gap with SOTA codecs limits its application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](../../ECCV2024/3d_vision/gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)
- [\[AAAI 2026\] SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images](smartsplat_feature-smart_gaussians_for_scalable_compression_of_ultra-high-resolu.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[AAAI 2026\] Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space](split-layer_enhancing_implicit_neural_representation_by_maximizing_the_dimension.md)
- [\[CVPR 2026\] SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation](../../CVPR2026/3d_vision/sgi_structured_2d_gaussians_for_efficient_and_compact_large_image_representation.md)

</div>

<!-- RELATED:END -->
