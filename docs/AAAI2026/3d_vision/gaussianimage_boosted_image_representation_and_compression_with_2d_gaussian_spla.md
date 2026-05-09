---
title: >-
  [Paper Note] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][2D Gaussian Splatting] This paper proposes GaussianImage++, which achieves high-quality image representation and compression with a limited number of 2D Gaussian primitives through a distortion-driven densification mechanism and content-aware Gaussian filters, while maintaining real-time decoding speed.
tags:
  - AAAI 2026
  - 3D Vision
  - 2D Gaussian Splatting
  - Image Compression
  - Implicit Neural Representation
  - Densification Mechanism
  - Quantization-Aware Training
date: 2026-05-08
content_hash: 42d320ab315b3df1
---

# GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting

**Conference**: AAAI 2026
**arXiv**: [2512.19108](https://arxiv.org/abs/2512.19108)
**Code**: [GitHub](https://github.com/Sweethyh/GaussianImage_plus.git)
**Area**: 3D Vision
**Keywords**: 2D Gaussian Splatting, Image Representation, Image Compression, Density Control, Quantization-Aware Training

## TL;DR

This paper proposes GaussianImage++, which achieves high-quality image representation and compression with a limited number of 2D Gaussian primitives via a distortion-driven densification mechanism and content-aware Gaussian filters, combined with an attribute-separated learnable scalar quantizer for efficient compression.

## Background & Motivation

### State of the Field

Neural image representation and compression are critical technologies for visual data storage, transmission, and rendering. Implicit neural representation (INR) methods such as SIREN and COIN achieve reasonable visual fidelity via lightweight MLPs, but suffer from long training times and high memory overhead. In recent years, 3D Gaussian Splatting (3DGS) has attracted broad attention for its efficiency in explicit primitive representation. GaussianImage first applied Gaussian Splatting to 2D image representation and compression, significantly reducing memory usage and training time.

### Limitations of Prior Work

**GaussianImage lacks a densification mechanism**: it cannot adaptively control the number of 2D Gaussian primitives according to image content, leaving representational capacity underutilized and limiting fitting quality.

**Mirage adopts the ADC from 3D GS for density control**: this tends to cause uncontrolled growth in the number of Gaussians, leading to out-of-memory (OOM) errors during training.

**LIG focuses on large-image fitting**: it employs a large number of 2D Gaussians but does not explore compact compression, resulting in high storage overhead.

**3D GS compression methods (HAC, ContextGS)** are built on neural Gaussians (Scaffold), creating an architectural mismatch that prevents direct application to 2D GS.

### Root Cause

How can **high-quality image representation** be achieved with a **limited number of 2D Gaussian primitives**, while simultaneously enabling **efficient compression** and **real-time decoding**?

### Starting Point

The paper approaches the problem from two complementary directions: (1) progressive distortion-driven densification, which allocates Gaussian primitives **on demand** to regions with poor reconstruction quality; and (2) content-aware Gaussian low-pass filters, which use strong filtering in early training to fill "holes" between sparse Gaussians and gradually reduce filtering strength in later stages to preserve fine details.

## Method

### Overall Architecture

GaussianImage++ builds upon 2D Gaussian Splatting, where each Gaussian primitive is parameterized by position $\boldsymbol{\mu} \in \mathbb{R}^2$, covariance $\boldsymbol{\Sigma} \in \mathbb{R}^{2 \times 2}$, and color $\mathbf{c} \in \mathbb{R}^3$. The rendering formula is an accumulative summation rather than alpha blending:

$$\mathbf{C} = \sum_{i \in N} \mathbf{c}_i G_i(\mathbf{x})$$

The framework consists of two pipelines: a **representation pipeline** (densification + filtering + rasterization) and a **compression pipeline** (overfitting initialization + quantization-aware training).

### Key Designs

#### 1. Distortion-Driven Densification (D³)

**Function**: Progressively allocates Gaussian primitives to regions of the image with poor reconstruction quality.

**Mechanism**: Three stages are employed:

- **Sparse initialization**: Training begins with $N_0 = M/2$ Gaussians, with positions uniformly sampled at random within the image coordinate space, accelerating early-stage training.
- **Gaussian growth**: Every 5,000 iterations, the per-pixel distortion $D(X, \hat{X})$ between the rendered image and the ground truth is computed, and new Gaussian primitives are inserted at the top-$k$ highest-distortion pixel locations:

$$\boldsymbol{\mu}_\Psi = \xi(\text{Top}_k(D(X, \hat{X})))$$

$$\mathbf{c}_\Psi = X(\xi(\text{Top}_k(D(X, \hat{X}))))$$

where $k = \tau(t, N_t, M) = (M - N_t)/2$, allocating half of the remaining budget at each step.

- **Gaussian pruning**: Every 100 iterations, the positive semi-definiteness of $\Sigma$ is checked and invalid Gaussians are removed.

**Design Motivation**: The ADC in 3D GS relies on positional gradients, which are typically too small in 2D scenes to trigger densification. The proposed method directly determines densification locations based on pixel-level distortion, making it more intuitive and image-quality-oriented.

#### 2. Content-Aware Gaussian Filter (CAF)

**Function**: Assigns an adaptively varying low-pass filter to each Gaussian primitive to regulate its coverage area.

**Mechanism**: A zero-mean Gaussian low-pass filter $h(x)$ is applied to each Gaussian footprint function:

$$G_i'(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T(\boldsymbol{\Sigma}_i + s_i I)^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)}$$

The filter variance $s_i$ for each Gaussian decreases according to a temporal schedule:

$$s_i = \frac{HW}{\alpha N_t}$$

Early-stage Gaussians have large $s_i$ (strong filtering, expanding coverage to reduce holes), while newly added Gaussians in later stages have small $s_i$ (weak filtering, focusing on fine details).

**Design Motivation**: In early training, $N_t \ll HW$, leaving many pixels uncovered by any Gaussian and producing visible artifacts. A large $s_i$ allows even a small number of Gaussians to cover the entire image, yielding a coarse but recognizable reconstruction that provides a favorable initialization for subsequent optimization. The filtered covariance $\Sigma + sI$ is stored directly, introducing no additional storage overhead.

#### 3. Compression Framework: Attribute-Separated LSQ+ Quantization

**Function**: Applies quantization with different bit depths to different overfitted Gaussian attributes.

**Mechanism**: LSQ+ (learnable scalar quantizer) is adopted, with different bit depths for different attributes:

- Position $\mu$: 12-bit (geometrically sensitive)
- Covariance $\Sigma$: 10-bit
- Color $c$: 6-bit

The quantization formula (with straight-through gradient estimator):

$$\bar{\mathbf{v}} = \lfloor \text{clip}(\frac{\mathbf{v}-\beta}{s}, 0, 2^b-1) \rfloor, \quad \hat{\mathbf{v}} = \bar{\mathbf{v}} \cdot s + \beta$$

A 6,000-step warm-up is performed first, followed by quantization-aware fine-tuning to adapt Gaussian attributes to quantization error.

### Loss & Training

- Representation task: L2 loss, Adam optimizer, learning rate 0.18 (halved after 20k iterations), 50k iterations total.
- Compression task: 6,000-step warm-up followed by quantization-aware training with a quantizer learning rate of 0.001.
- Densification is performed every 5,000 steps; covariance validity is checked and pruning is applied every 100 steps.

## Key Experimental Results

### Main Results (Image Representation)

| Dataset | Method | PSNR (5k GS) | PSNR (10k GS) | Rendering FPS |
|--------|------|-------------|---------------|---------|
| Kodak | GaussianImage | 29.85 | 32.48 | 2009 |
| Kodak | GaussianImage++ | **31.83** | **35.41** | 2216 |
| DIV2K | GaussianImage | 26.54 | 31.45 | 662 |
| DIV2K | GaussianImage++ | **28.14** | **33.75** | 765 |
| Kodak | Siren (INR) | 26.50 | - | 977 |

Key finding: On Kodak with 10k Gaussians, GaussianImage++ outperforms GaussianImage by approximately 3 dB in PSNR and LIG by approximately 4 dB, while rendering speed is not reduced but improved.

### Ablation Study

| Configuration | BD-PSNR (dB)↑ | BD-Rate (%)↓ | Notes |
|------|-------------|-------------|------|
| LSQ+/LSQ+ (Ours) | 0 | 0 | Final strategy |
| FP16/LSQ+ | -0.761 | +25.11 | FP16 for position |
| FP16/RVQ | -2.471 | +138.88 | RVQ for color |
| LSQ+/RVQ | -2.491 | +147.24 | RVQ for color, worst |

D³ alone yields approximately 2 dB PSNR improvement; the combination of D³ and CAF achieves approximately 3 dB gain over GaussianImage and 4 dB over LIG. Both components are effective across three different covariance parameterization schemes (Cholesky, RS, and direct parameterization).

### Key Findings

1. D³ is the most critical component, with the largest gains when the number of Gaussians is small.
2. CAF further improves performance in combination with D³, particularly stabilizing optimization in early training.
3. RVQ for color quantization performs poorly, indicating insufficient codebook expressiveness.
4. Compression performance surpasses JPEG at low bit rates (0.1–0.7 bpp) but falls behind learned codecs at high bit rates.
5. Decoding speed (>1,600 FPS) substantially outperforms all VAE-based and INR-based methods.

## Highlights & Insights

1. **Progressive "sparse-to-dense" training strategy**: Rather than initializing all Gaussians at once, the method starts from $M/2$ and grows progressively, reducing early training cost while directing resources to the most deficient regions via distortion guidance.
2. **Adaptive scheduling of filter variance**: The method elegantly exploits the "coverage expansion" effect of low-pass filters to compensate for sparse Gaussians in early training, without incurring additional storage overhead.
3. **Generality of the two components**: D³ and CAF can serve as plug-and-play modules to enhance other 2D GS methods.
4. **Real-time decoding advantage**: The simple accumulative summation in 2D GS is inherently parallelizable, yielding decoding speeds far superior to methods requiring autoregressive entropy decoding.

## Limitations & Future Work

1. Performance at high bit rates still lags behind state-of-the-art learned codecs (e.g., Ballé18), necessitating more advanced attribute coding and entropy models.
2. Encoding time is far from real-time; the training and quantization processes require substantial optimization.
3. Integration with more advanced entropy coding schemes (e.g., arithmetic coding) has not been explored.
4. Evaluation is limited to Kodak and DIV2K; validation on a broader range of resolutions and content types is lacking.

## Related Work & Insights

- **GaussianImage**: The pioneering work on 2D GS image representation and the direct predecessor of the proposed method.
- **LIG**: Focuses on large-image fitting without compression; D³ offers a superior alternative to its hierarchical scheme.
- **3DGS ADC**: Adaptive density control in 3D scenes based on positional gradients; the paper analyzes why this approach is inapplicable in 2D and proposes an alternative.
- **Insight**: Significant opportunities remain for 2D GS in image compression, particularly in integrating improved entropy models and multi-resolution representations.

## Rating

- Novelty: ⭐⭐⭐⭐ — D³ and CAF are clearly motivated and well-designed, though the core ideas are not highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset, multi-configuration ablations are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Logically structured with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ — Provides general-purpose enhancement modules for 2D GS image representation and compression.

---

# GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting

**Conference**: AAAI 2026
**arXiv**: [2512.19108](https://arxiv.org/abs/2512.19108)
**Code**: [https://github.com/Sweethyh/GaussianImage_plus](https://github.com/Sweethyh/GaussianImage_plus)
**Area**: 3D Vision / Image Representation and Compression
**Keywords**: 2D Gaussian Splatting, Image Compression, Implicit Neural Representation, Densification Mechanism, Quantization-Aware Training

## TL;DR

This paper proposes GaussianImage++, which achieves high-quality image representation and compression with a limited number of 2D Gaussian primitives through a distortion-driven densification mechanism and content-aware Gaussian filters, while maintaining real-time decoding speed.

## Background & Motivation

### State of the Field

Image representation and compression are fundamental problems in visual data storage and transmission. Current mainstream approaches include:
- **Autoencoder-based neural compression** (e.g., Ballé18, ELIC): excellent rate-distortion performance but high decoding latency.
- **Implicit neural representations (INR)** (e.g., SIREN, COIN): map pixel coordinates to colors via MLPs, but are slow to train and memory-intensive.
- **2D Gaussian Splatting (GS)**: GaussianImage first applied GS to 2D images, significantly reducing training time and memory usage.

### Limitations of Prior Work

1. **GaussianImage lacks a densification mechanism**: Gaussian primitives cannot be adaptively allocated based on image content, leaving many under-reconstructed regions.
2. **Mirage uses the ADC from 3D GS**: prone to uncontrolled growth in the number of Gaussians, causing OOM errors.
3. **LIG does not perform compression**: focused on fitting large images without exploring attribute compression, incurring high storage overhead.
4. **3D GS compression methods are not directly transferable**: HAC and ContextGS are based on neural Gaussians (Scaffold), which are architecturally incompatible with explicit 2D GS.

### Root Cause

How can high visual fidelity and efficient compression be achieved simultaneously with a **limited number of 2D Gaussian primitives**?

### Starting Point

The paper enhances 2D GS along three dimensions: (1) progressive distortion-driven densification to control Gaussian distribution; (2) content-aware filters to improve Gaussian rendering quality; and (3) attribute-separated learnable scalar quantization for efficient compression.

## Method

### Overall Architecture

GaussianImage++ operates in two main stages:
1. **Image representation**: sparse initialization → periodic distortion-driven densification → content-aware filtering → accumulative summation rasterization.
2. **Image compression**: overfitting of Gaussian attributes → quantization-aware training fine-tuning → encoding into a compact bitstream.

Each 2D Gaussian is parameterized by position $\boldsymbol{\mu} \in \mathbb{R}^2$, covariance $\boldsymbol{\Sigma} \in \mathbb{R}^{2 \times 2}$, and color $\mathbf{c} \in \mathbb{R}^3$. The rendering formula is:

$$G_i(\mathbf{x}) = \exp\left(-\frac{(\mathbf{x}-\boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}_i)}{2}\right)$$

$$\mathbf{C} = \sum_{i \in N} \mathbf{c}_i G_i(\mathbf{x})$$

### Key Designs

#### 1. Distortion-Driven Densification (D³)

**Function**: Progressively allocates Gaussian primitives to under-reconstructed regions.

**Mechanism**: A three-stage process:

- **Sparse initialization**: Initial count $N_0 = M/2$ (where $M$ is the maximum number of Gaussians), with positions uniformly sampled at random within the image coordinate space and colors initialized to zero.
- **Gaussian growth**: Every 5,000 iterations, new Gaussians are added at the top-$k$ highest-distortion pixel locations, with the count determined by the scheduler $\tau(t, N_t, M) = (M - N_t)/2$.
- **Gaussian pruning**: Every 100 iterations, the positive semi-definiteness of the covariance matrix is checked and invalid Gaussians are removed.

**Design Motivation**: The ADC in 3D GS relies on positional gradients, which are too small in 2D scenes to trigger densification effectively. The proposed method directly uses pixel-level distortion (L1 loss) to determine densification locations, which is more straightforward and quality-oriented. The position and color of new Gaussians are initialized directly from the high-distortion pixels of the original image:

$$\boldsymbol{\mu}_\Psi = \xi(\text{Top}_k(D(X, \hat{X})))$$
$$\mathbf{c}_\Psi = X(\xi(\text{Top}_k(D(X, \hat{X}))))$$

#### 2. Content-Aware Gaussian Filter (CAF)

**Function**: Applies adaptively scaled low-pass filtering to each Gaussian primitive to reduce rendering holes and artifacts.

**Mechanism**: A zero-mean Gaussian low-pass filter $h(x)$ is applied to the original Gaussian kernel. A variance vector $\mathbf{s} \in \mathbb{R}^{N_t}$ controls the filtering strength for each Gaussian:

$$G_i'(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T(\boldsymbol{\Sigma}_i + s_i I)^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)}$$

Variance formula:
$$s_i = \frac{HW}{\alpha N_t} \quad (\text{for newly added Gaussians})$$

**Design Motivation**: When Gaussians are sparse in early training ($N_t \ll HW$), large-variance filters expand coverage and reduce holes, producing a coarse but recognizable image to guide optimization. As densification progresses, the filtering strength for new Gaussians decreases, enabling focus on fine details. Crucially, $\mathbf{s}$ introduces no storage overhead, as the filtered covariance $\boldsymbol{\Sigma} + sI$ is stored directly.

#### 3. Compression Framework (Attribute-Separated Quantization)

**Function**: Applies learnable scalar quantization (LSQ+) with different bit depths to different Gaussian attributes.

**Mechanism**:
- Position $\boldsymbol{\mu}$: 12-bit (geometrically sensitive, requires high precision)
- Covariance $\boldsymbol{\Sigma}$: 10-bit
- Color $\mathbf{c}$: 6-bit

Quantization formula:
$$\bar{\mathbf{v}} = \lfloor \text{clip}(\frac{\mathbf{v} - \beta}{s}, 0, 2^b - 1) \rfloor, \quad \hat{\mathbf{v}} = \bar{\mathbf{v}} \cdot s + \beta$$

**Design Motivation**: Quantization-aware training (QAT) enables Gaussians to actively adjust their attributes to accommodate quantization error. Compared to FP16 or RVQ, the learnable offset and scale in LSQ+ achieve a more favorable rate-distortion trade-off.

### Loss & Training

- Representation stage: L2 loss, Adam optimizer, 50,000 iterations, learning rate 0.18 (halved after 20,000 iterations).
- Compression stage: 6,000-step warm-up followed by quantization-aware fine-tuning with a quantizer learning rate of 0.001.

## Key Experimental Results

### Main Results

#### Image Representation (Kodak, 10k Gaussians)

| Method | PSNR↑ | MS-SSIM↑ | Params (M) | GPU Mem (MiB) | Rendering FPS |
|------|-------|----------|-----------|-------------|---------|
| Siren (INR) | 26.50 | 0.875 | 3.74 | 2044 | 977 |
| GaussianImage | 32.48 | 0.982 | 0.08 | 814 | 2009 |
| LIG | 31.00 | 0.975 | 0.08 | 832 | 1331 |
| **Ours** | **35.41** | **0.983** | 0.08 | 876 | **2216** |

#### Image Compression (Kodak, low/high bpp)

| Method | Bpp | PSNR | Decoding FPS |
|------|-----|------|---------|
| JPEG | 0.22/1.03 | 23.8/32.8 | 377/148 |
| COIN | 0.17/0.98 | 24.9/27.4 | 769/344 |
| GaussianImage | 0.15/1.00 | 25.0/29.7 | 1827/1822 |
| **Ours** | 0.15/1.08 | **25.3/31.1** | **1839/1666** |

### Ablation Study

#### Component Ablation (Kodak)

| Configuration | PSNR Gain (vs GS Cholesky) | Notes |
|------|--------------------------|------|
| + D³ alone | ~2 dB | Densification contributes the most individually |
| + D³ + CAF | ~3 dB | Synergistic improvement from both components |
| vs LIG | ~4 dB | Substantial overall improvement |

#### Quantization Strategy Ablation

| Configuration (Position/Color) | BD-PSNR (dB) | BD-Rate (%) |
|------------------|-------------|------------|
| LSQ+/LSQ+ (Ours) | 0 | 0 |
| FP16/LSQ+ | -0.761 | +25.11% |
| FP16/RVQ | -2.471 | +138.88% |
| LSQ+/RVQ | -2.491 | +147.24% |

### Key Findings

1. D³ densification is especially effective when the number of Gaussians is small, as sparse Gaussians benefit most from precise allocation.
2. CAF plays a crucial role in early training—at $t=500$, it already produces a recognizable coarse image, whereas the baseline exhibits numerous holes.
3. Both components are effective across three different covariance parameterization schemes (Cholesky, RS, direct parameterization), demonstrating generality.
4. GS-based methods substantially outperform traditional and learned codecs in decoding speed (>1,800 FPS vs. JPEG at ~150 FPS).

## Highlights & Insights

1. **Intuitively motivated distortion-driven densification**: placing new Gaussians directly at the worst-reconstructed pixels is simple yet highly effective.
2. **Elegant progressive attenuation of CAF**: large coverage in early stages transitions to fine-grained detail in later stages, forming a natural synergy with densification.
3. **Universally applicable enhancement technique**: D³ and CAF can be applied as plug-and-play modules to other 2D GS methods.
4. **Clear real-time decoding advantage**: the simple accumulative summation in GS offers an inherent speed advantage over the decoding latency of VAE- and INR-based methods.

## Limitations & Future Work

1. **Performance at high bit rates still lags behind state-of-the-art neural codecs**: this is a common limitation of current 2D GS compression methods, requiring more advanced entropy models.
2. **Encoding time is far from real-time**: the training and quantization processes are time-consuming, limiting practical deployment.
3. **Lack of adaptive bit allocation**: the same quantization configuration is applied to all images without adjustment for image complexity.
4. Extending D³ and CAF to video GS scenarios is a promising direction for future work.

## Related Work & Insights

- **GaussianImage** (Zhang et al., 2024): the first 2D GS image representation method and the direct foundation of this work.
- **3D GS ADC** (Kerbl et al., 2023): density control based on positional gradients; inspired D³ but operates through a fundamentally different mechanism.
- **LSQ+** (Bhalgat et al., 2020): low-bit quantization with learnable offset and scale; the core tool for compression in this work.
- **COOL-CHIC** (Ladune et al., 2023): a hybrid INR compression method that requires an autoregressive entropy model, increasing decoding overhead.

## Rating

- Novelty: ⭐⭐⭐⭐ — D³ and CAF are concise and effective, though the core idea of placing Gaussians at high-distortion locations is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Dual-dataset evaluation, multiple baselines, cross-method ablations, and quantization strategy ablations are all covered.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with thorough motivation.
- Value: ⭐⭐⭐⭐ — Practical value as a general-purpose enhancement technique, though the gap with state-of-the-art codecs limits application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation](../../CVPR2026/3d_vision/sgi_structured_2d_gaussians_large_image_representation.md)
- [\[AAAI 2026\] SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images](smartsplat_feature-smart_gaussians_for_scalable_compression_of_ultra-high-resolu.md)
- [\[NeurIPS 2025\] Anti-Aliased 2D Gaussian Splatting](../../NeurIPS2025/3d_vision/anti-aliased_2d_gaussian_splatting.md)
- [\[ICLR 2026\] SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors](../../ICLR2026/3d_vision/surfsplat_conquering_feedforward_2d_gaussian_splatting_with_surface_continuity_p.md)
- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](point-sra_self-representation_alignment_for_3d_representation_learning.md)

</div>

<!-- RELATED:END -->
