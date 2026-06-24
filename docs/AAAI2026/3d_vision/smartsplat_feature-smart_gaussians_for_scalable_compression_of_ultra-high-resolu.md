---
title: >-
  [Paper Note] SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images
description: >-
  [AAAI2026][3D Vision][2D Gaussian Splatting] This paper proposes SmartSplat, a feature-aware 2D Gaussian Splatting image compression framework. By adopting three coordinate strategies—gradient-color-guided variational sampling, repulsive uniform sampling, and scale-adaptive color initialization—it achieves high-quality reconstruction of 8K/16K ultra-high-resolution images under extreme compression ratios (up to 5000×) for the first time.
tags:
  - "AAAI2026"
  - "3D Vision"
  - "2D Gaussian Splatting"
  - "Image Compression"
  - "Ultra-High-Resolution"
  - "Feature-Guided Sampling"
  - "High Compression Ratio"
date: 2026-05-08
content_hash: b127de3ae06e61ad
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images

**Conference**: AAAI2026  
**arXiv**: [2512.20377](https://arxiv.org/abs/2512.20377)  
**Code**: [lif314/SmartSplat](https://github.com/lif314/SmartSplat)  
**Authors**: Linfei Li, Lin Zhang, Zhong Wang, Ying Shen  
**Area**: 3D Vision  
**Keywords**: 2D Gaussian Splatting, Image Compression, Ultra-High-Resolution, Feature-Guided Sampling, High Compression Ratio  

## TL;DR

This paper proposes SmartSplat, a feature-aware 2D Gaussian Splatting image compression framework. By adopting three coordinate strategies—gradient-color-guided variational sampling, repulsive uniform sampling, and scale-adaptive color initialization—it achieves high-quality reconstruction of 8K/16K ultra-high-resolution images under extreme compression ratios (up to 5000×) for the first time.

## Background & Motivation

### Compression Bottlenecks of Ultra-High-Resolution Images
With the rapid development of generative AI, ultra-high-resolution (UHR) visual content has become increasingly common, with file sizes of 8K and even 16K images reaching tens to hundreds of megabytes. Traditional formats like JPEG can only achieve a compression ratio of around 50×, which falls far short of meeting the demands for efficient transmission and real-time rendering. Although Implicit Neural Representations (INR) offer powerful compression capabilities, they rely on fixed architectures and full-image training, leading to massive computational overhead. Moreover, their neural inference causes slow decoding speeds, making them unsuitable for real-time scenarios.

### Opportunities and Limitations of 2D Gaussian Splatting
3D Gaussian Splatting (3DGS) achieves an excellent balance between rendering quality and real-time performance through explicit modeling of Gaussian primitives and a differentiable tile-based rasterization pipeline. Extending it to 2D image representations (e.g., GaussianImage, LIG, ImageGS) significantly improves training and decoding efficiency. However, existing methods either rely on a large number of Gaussian primitives to guarantee reconstruction accuracy, or only achieve limited compression ratios on low-resolution images below 2K, underperforming in UHR scenarios.

### Key Challenge: Efficient Representation with Limited Gaussians
Under highly constrained compression ratios, the allowable number of Gaussians $N_g = \frac{3HW}{7 \cdot \mathrm{CR}}$ decreases drastically. How to simultaneously capture the high-frequency structures and low-frequency textures of an image using extremely limited Gaussian primitives stands as a key technical challenge. Existing methods often suffer from optimization crashes due to NaN values during rasterization caused by sparse distributions when the number of Gaussians is highly constrained. The motivation behind SmartSplat is precisely to fill this gap—designing a feature-driven adaptive Gaussian distribution strategy that enables efficient image compression across arbitrary resolutions and compression ratios.

## Core Problem

How to efficiently represent ultra-high-resolution images using limited 2D Gaussian primitives under extreme compression ratio constraints (200× to 5000×) while maintaining high-quality reconstruction? The key lies in jointly optimizing the spatial position, scale, and color initialization of Gaussians to make them adaptively cover different frequency components of the image.

## Method

### Overall Architecture

SmartSplat starts with the input image, initializes the Gaussian primitives via a three-stage feature-aware sampling process, and then iteratively optimizes them through differentiable rasterization. The overall pipeline is as follows:

1. **Gradient-Color-Guided Variational Sampling (VS)**: Jointly generates sampling probability maps based on image gradients and color variance, sampling densely in high-frequency regions and sparsely in low-frequency regions, while initializing spatial positions and scales.
2. **Repulsive Uniform Sampling (US)**: Complements variational sampling with uniform sampling in regions with low structural complexity that are not covered by VS, avoiding overlaps through a repulsion radius constraint.
3. **Scale-Adaptive Color Initialization**: Estimates the color of each primitive based on Gaussian-weighted median filtering, enhancing robustness.
4. **Joint Optimization**: Performs end-to-end optimization of all Gaussian parameters using a compound L1 + SSIM loss.

### Key Designs

### Key Design 1: Gradient-Color-Guided Variational Sampling

The image is divided into multiple tiles to be processed independently. Within each tile $\mathbf{I}_{i,j}$, the pixel gradient magnitude and color variance are calculated:

$$m_{i,j}(\mathbf{x}) = \frac{1}{C}\sum_{c=1}^{C}\|\nabla \mathbf{I}_{i,j,c}(\mathbf{x})\|_2, \quad v_{i,j}(\mathbf{x}) = \frac{1}{C}\sum_{c=1}^{C}\mathrm{Var}(\mathbf{I}_{i,j,c}(\mathcal{N}_\mathbf{x}))$$

After normalization, they are combined via a weighted sum to obtain the sampling weight $w_{i,j}(\mathbf{x}) = \lambda_m \cdot \tilde{m}_{i,j}(\mathbf{x}) + (1 - \lambda_m) \cdot \tilde{v}_{i,j}(\mathbf{x})$, where $\lambda_m = 0.9$. The sampling probability is given by $\mathbb{P}_{i,j}(\mathbf{x}) = w_{i,j}(\mathbf{x}) / \sum_\mathbf{y} w_{i,j}(\mathbf{y})$, from which points are selected using multinomial sampling.

The scale initialization employs exponential decay: $s_{i,j}(\mathbf{x}) = s_{base} \cdot \exp(-\frac{1}{2} w_{i,j}(\mathbf{x}))$, where the base scale is derived from the maximum non-overlapping coverage:

$$s_{base} = \frac{1}{3}\sqrt{\frac{HW}{\pi N_g}}$$

### Key Design 2: Repulsive Uniform Sampling

To cover low-frequency regions, uniform sampling is performed based on the variational sampling set $\mathcal{X}_{vs}$, forcing the new sample points to satisfy a repulsion constraint:

$$\forall j, \quad \min_i \|\mathbf{x}_j^{us} - \mathbf{x}_i^{vs}\| \geq r_{excl}, \quad r_{excl} = \max(s_{base}, \mathrm{median}(\{s_i^{vs}\}))$$

The scale of uniform sample points is estimated via Query-to-Reference KNN: $s_j^{us} = \sqrt{\frac{1}{K}\sum_{\mathbf{q} \in \mathcal{N}_K(\mathbf{x}_j^{us}, \mathcal{X})}\|\mathbf{x}_j^{us} - \mathbf{q}\|^2}$, where $K=3$.

### Key Design 3: Scale-Adaptive Color Sampling

For each sample point $\mathbf{x}_i$, a neighborhood with a radius of its scale $s_i$ is defined. The color is estimated using Gaussian-weighted median filtering:

$$\mathbf{c}_i^{(d)} = \arg\min_{z \in \mathbb{R}} \sum_{\mathbf{u} \in \mathcal{N}_{\mathbf{x}_i}} w_i(\mathbf{u}) \cdot |z - \mathbf{I}^{(d)}(\mathbf{u})|$$

Compared to random initialization or pixel center estimation, the weighted median is more robust against noise and outliers.

### Optimization Objective

The ratio of variational sampling to uniform sampling is set to $\lambda_g = 0.7$ (i.e., 70% variational, 30% uniform), with the loss function defined as:

$$L = \lambda_l \|\hat{\mathbf{I}} - \mathbf{I}\|_1 + (1 - \lambda_l)(1 - \mathrm{SSIM}(\hat{\mathbf{I}}, \mathbf{I})), \quad \lambda_l = 0.9$$

## Key Experimental Results

### Main Results on DIV8K (Average Resolution 5736×6120, Average Size 53.56MB)

| CR | 3DGS | LIG | GI (RS) | GI (Cholesky) | ImageGS | **SmartSplat** |
|------|------|------|---------|---------------|---------|----------------|
| 20× | 30.99/0.9636 | 28.05/0.9362 | 30.45/0.9707 | 30.33/0.9698 | 32.00/0.8680 | **33.26/0.9752** |
| 50× | 28.56/0.9340 | 24.90/0.8402 | 26.99/0.9291 | 26.87/0.9271 | 29.47/0.8052 | **29.65/0.9482** |
| 100× | 26.84/0.8990 | 22.91/0.7230 | 25.00/0.8827 | 24.90/0.8790 | 26.65/0.7449 | **27.49/0.9164** |
| 200× | 24.92/0.8556 | 21.06/0.5792 | 23.45/0.8223 | 23.35/0.8176 | 26.80/0.7181 | **25.75/0.8745** |
| 500× | 22.38/0.7874 | 17.68/0.3633 | Fail | Fail | 24.88/0.6544 | **23.82/0.8055** |
| 1000× | 20.38/0.7068 | 12.49/0.2083 | Fail | Fail | 23.50/0.6165 | **22.66/0.7469** |

Metric format: PSNR (dB) / MS-SSIM. SmartSplat leads the runner-up method by 1.26dB in PSNR at 20×. At 500× and 1000×, GI completely fails while SmartSplat still works stably.

### Results on DIV16K (Average Resolution 12684×15898, Average Size 235.52MB)

| CR | 3DGS | GI (RS) | **SmartSplat** |
|------|------|---------|----------------|
| 50× | OOM | 29.24/0.7917 | **34.34/0.9267** |
| 100× | OOM | 27.39/0.7648 | **33.00/0.9117** |
| 200× | OOM | 25.63/0.7394 | **31.85/0.8897** |
| 500× | 28.61/0.8117 | Fail | **29.40/0.8524** |
| 1000× | 27.06/0.7854 | Fail | **27.49/0.8226** |
| 2000× | 25.54/0.7642 | Fail | **25.70/0.7966** |
| 3000× | Fail | Fail | **24.72/0.7844** |

SmartSplat is the only method capable of completing training under a 3000× compression ratio, leading GI by approximately 5.64dB in average PSNR on DIV16K.

### Efficiency Comparison (10848×16320 Image, CR=200)

| Method | Iterration Speed | Training Time (s) | VRAM (GB) | FPS | PSNR |
|------|---------|-----------|---------|-----|------|
| 3DGS (10K) | 1.32 it/s | 7841.80 | 50.19 | 10.98 | 24.42 |
| GI (10K) | 7.44 it/s | 1334.73 | 16.29 | 62.33 | 19.86 |
| SmartSplat (10K) | 5.01 it/s | 2237.52 | 19.59 | 32.35 | **31.87** |
| SmartSplat (1K) | 5.03 it/s | 336.12 | 19.38 | 33.12 | 30.52 |

SmartSplat requires only 1K iterations to reach 30.52dB, surpassing the results of 3DGS and GI at 10K iterations. The VRAM footprint is only 39% of 3DGS.

### Ablation Study (4416×6720 Image, CR=200, 10K Iterations)

| Configuration | PSNR (dB) | MS-SSIM |
|------|----------|---------|
| Full Random | 22.34 | 0.8435 |
| +VS/US Position Init | 22.18 | 0.8270 |
| +VS/US Scale Init | 23.12 | 0.8647 |
| +Scale-Adaptive Color (Full SmartSplat) | **24.38** | **0.8972** |

Scale initialization contributes the most (+0.94dB), and color initialization further improves performance by 1.26dB, proving that all three components are indispensable.

## Highlights & Insights

- **First UHR GS Compression Framework**: Translates GS-based image compression into the 8K/16K tier for the first time, supporting extreme compression ratios up to 5000×.
- **Hyperparameter-Free Scale Initialization**: $s_{base}$ is fully derived from image resolution and compression ratio without manual tuning or heuristic clamping.
- **Three-Stage Joint Initialization**: The collaborative initialization strategy of position, scale, and color allows SmartSplat to outperform baselines at 10K iterations with just 1K iterations.
- **Robust Color Estimation**: Gaussian-weighted median filtering performs significantly better than random initialization and pixel center estimation in high-frequency texture regions.
- **Excellent Scalability**: SmartSplat operates stably even when other methods fail due to OOM or NaN errors.

## Limitations & Future Work

- **Focus on Spatial Distribution Only**: The current framework focuses on optimizing the spatial distribution of Gaussians and does not address further compression (such as quantization and encoding) of Gaussian attributes (color, opacity), which remains a key direction for boosting compression efficiency.
- **DIV16K Dataset Construction**: The dataset is upsampled from DIV2K via super-resolution tools, which may introduce distribution discrepancies compared to genuine 16K-captured images in terms of texture details.
- **Limited Evaluation Scale**: DIV8K includes only 16 images and DIV16K only 8 images for evaluation, which may raise questions regarding statistical significance.
- **Moderate Decoding Speed**: The decoding speed is around 32 FPS. Although it is far superior to INR methods, it is slower than GI's 62 FPS, leaving room for further optimization in real-time applications.
- **No Comparison with Neural Codecs**: Lacks comparisons against learning-based end-to-end image codecs (e.g., Hyperprior, ELIC).

## Related Work & Insights

- **GaussianImage (GI)**: Utilizes a two-stage optimization with vector quantization but easily fails due to NaN values at high compression ratios when Gaussians are scarce. SmartSplat circumvents this issue via adaptive initialization, surpassing GI by 2.57dB in PSNR under the same CR.
- **LIG**: Prioritizes fitting accuracy over compression performance with a hierarchical Gaussian scheme, requiring a large number of parameters; its performance drops sharply under high CR.
- **ImageGS**: Content-aware initialization combined with progressive training is unstable under extreme compression scenarios; ImageGS completely fails due to OOM on DIV16K, whereas SmartSplat runs stably.
- **3DGS**: Direct 2D extension is somewhat effective but leads to slow training and high VRAM usage (3DGS: 50GB vs. SmartSplat: 20GB) due to identity matrix mapping.

## Insights & Connections

- **General Paradigm of Feature-Guided Initialization**: Leveraging image gradients and color variance to guide the spatial distribution of discrete primitives is a paradigm that can be extended to other primitive-based tasks (e.g., point cloud compression, NeRF initialization).
- **Explicit Relationship Between Compression Ratio and Primitive Count**: The formula $N_g = 3HW / (7 \cdot \mathrm{CR})$ clearly defines the relationship between compression ratio and representations capacity, providing an analytical framework for future work.
- **Concept of Repulsive Sampling**: Avoiding overlapping sample points via a repulsion radius limit is conceptually similar to Poisson Disk Sampling in computer graphics, which can be adapted to other scenarios requiring uniform coverage.

## Rating

- Novelty: ⭐⭐⭐⭐ — Pushes GS compression into the UHR domain. The three-stage initialization strategy is well-designed, although the core concept (feature-guided sampling) is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐ — The experimental design is complete (main results + ablations + efficiency), but the number of evaluation images is small (16+8) and it lacks comparisons with neural codecs.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations, detailed methodology descriptions, and rich diagrams, though some notation is a bit verbose.
- Value: ⭐⭐⭐⭐ — Makes significant progress in the highly practical direction of UHR image compression, and the open-source code enhances reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LoG3D: Ultra-High-Resolution 3D Shape Modeling via Local-to-Global Partitioning](../../CVPR2026/3d_vision/log3d_ultra-high-resolution_3d_shape_modeling_via_local-to-global_partitioning.md)
- [\[AAAI 2026\] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)
- [\[ICLR 2026\] Hyden: A Hybrid Dual-Path Encoder for Monocular Geometry of High-resolution Images](../../ICLR2026/3d_vision/hyden_a_hybrid_dual-path_encoder_for_monocular_geometry_of_high-resolution_image.md)
- [\[CVPR 2026\] Scalable Feature Matching via State Space Modeling and Sparse Correlation](../../CVPR2026/3d_vision/scalable_feature_matching_via_state_space_modeling_and_sparse_correlation.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](../../ICCV2025/3d_vision/one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)

</div>

<!-- RELATED:END -->
