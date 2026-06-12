---
title: >-
  [Paper Note] SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images
description: >-
  [AAAI2026][3D Vision][2D Gaussian Splatting] This paper proposes SmartSplat, a feature-aware 2D Gaussian Splatting framework for image compression. By introducing three key strategies—gradient-color-guided variational sa…
tags:
  - "AAAI2026"
  - "3D Vision"
  - "2D Gaussian Splatting"
  - "image compression"
  - "ultra-high-resolution"
  - "feature-guided sampling"
  - "high compression ratio"
date: 2026-05-08
content_hash: a80cefcb62efa47d
---

# SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images

**Conference**: AAAI2026
**arXiv**: [2512.20377](https://arxiv.org/abs/2512.20377)  
**Code**: [lif314/SmartSplat](https://github.com/lif314/SmartSplat)  
**Authors**: Linfei Li, Lin Zhang, Zhong Wang, Ying Shen
**Area**: 3D Vision
**Keywords**: 2D Gaussian Splatting, image compression, ultra-high-resolution, feature-guided sampling, high compression ratio

## TL;DR

This paper proposes SmartSplat, a feature-aware 2D Gaussian Splatting framework for image compression. By introducing three key strategies—gradient-color-guided variational sampling, repulsion-based uniform sampling, and scale-adaptive color initialization—SmartSplat achieves, for the first time, high-quality reconstruction of 8K/16K ultra-high-resolution (UHR) images at extreme compression ratios (up to 5000×).

## Background & Motivation

### Compression Bottleneck for Ultra-High-Resolution Images

With the rapid advancement of generative AI, UHR visual content has become increasingly prevalent, with 8K and 16K images reaching tens to hundreds of megabytes. Traditional formats such as JPEG achieve at most approximately 50× compression ratios, which is far from sufficient for efficient transmission and real-time rendering. Although Implicit Neural Representations (INR) offer strong compression capabilities, they rely on fixed architectures and full-image training, incurring prohibitive computational costs, while their neural decoding leads to slow decompression that is unsuitable for real-time applications.

### Opportunities and Limitations of 2D Gaussian Splatting

3D Gaussian Splatting (3DGS) achieves an excellent balance between rendering quality and real-time performance through explicit Gaussian primitive modeling and a differentiable tile-based rasterization pipeline. Its extension to 2D image representation (e.g., GaussianImage, LIG, ImageGS) significantly improves training and decoding efficiency. However, existing methods either rely on a large number of Gaussian primitives to ensure reconstruction quality, or achieve only limited compression ratios on low-resolution images below 2K, performing poorly in UHR scenarios.

### Core Challenge: Efficient Representation with Limited Gaussians

Under high compression ratio constraints, the allowable number of Gaussians $N_g = \frac{3HW}{7 \cdot \mathrm{CR}}$ decreases drastically. How to simultaneously capture high-frequency structures and low-frequency textures with an extremely limited number of Gaussian primitives constitutes the core technical challenge. Existing methods frequently encounter NaN values during rasterization due to sparse distributions when the Gaussian count is constrained, causing optimization to collapse. SmartSplat is motivated to fill this gap by designing a feature-driven adaptive Gaussian distribution strategy that enables efficient image compression at arbitrary resolutions and compression ratios.

## Core Problem

How to efficiently represent ultra-high-resolution images using a limited number of 2D Gaussian primitives under extreme compression ratio constraints (200×–5000×) while maintaining high reconstruction quality? The key lies in jointly optimizing the spatial positions, scales, and color initialization of Gaussians so that they adaptively cover different frequency components of the image.

## Method

### Overall Architecture

SmartSplat takes an input image and initializes Gaussian primitives through a three-stage feature-aware sampling process, followed by iterative optimization via differentiable rasterization. The overall pipeline is as follows:

1. **Gradient-color-guided variational sampling (VS)**: Jointly computes sampling probabilities from image gradients and color variance, densely sampling in high-frequency regions and sparsely in low-frequency regions, while initializing positions and scales.
2. **Repulsion-based uniform sampling (US)**: Supplements uniform sampling in low-structural-complexity regions not covered by variational sampling, with repulsion radius constraints to avoid overlap.
3. **Scale-adaptive color initialization**: Estimates the color of each primitive using Gaussian-weighted median filtering for improved robustness.
4. **Joint optimization**: Performs end-to-end optimization of all Gaussian parameters using a composite L1 + SSIM loss.

### Key Design 1: Gradient-Color-Guided Variational Sampling

The image is divided into multiple tiles and processed independently. Within each tile $\mathbf{I}_{i,j}$, the gradient magnitude and color variance of each pixel are computed:

$$m_{i,j}(\mathbf{x}) = \frac{1}{C}\sum_{c=1}^{C}\|\nabla \mathbf{I}_{i,j,c}(\mathbf{x})\|_2, \quad v_{i,j}(\mathbf{x}) = \frac{1}{C}\sum_{c=1}^{C}\mathrm{Var}(\mathbf{I}_{i,j,c}(\mathcal{N}_\mathbf{x}))$$

After normalization, the sampling weight is obtained via a weighted combination: $w_{i,j}(\mathbf{x}) = \lambda_m \cdot \tilde{m}_{i,j}(\mathbf{x}) + (1 - \lambda_m) \cdot \tilde{v}_{i,j}(\mathbf{x})$, where $\lambda_m = 0.9$. The sampling probability is $\mathbb{P}_{i,j}(\mathbf{x}) = w_{i,j}(\mathbf{x}) / \sum_\mathbf{y} w_{i,j}(\mathbf{y})$, and points are selected via multinomial sampling.

Scale initialization follows an exponential decay: $s_{i,j}(\mathbf{x}) = s_{base} \cdot \exp(-\frac{1}{2} w_{i,j}(\mathbf{x}))$, where the base scale is derived from the maximum non-overlapping coverage:

$$s_{base} = \frac{1}{3}\sqrt{\frac{HW}{\pi N_g}}$$

### Key Design 2: Repulsion-Based Uniform Sampling

To cover low-frequency regions, uniform sampling is performed on top of the variational sampling set $\mathcal{X}_{vs}$, requiring new sample points to satisfy a repulsion constraint:

$$\forall j, \quad \min_i \|\mathbf{x}_j^{us} - \mathbf{x}_i^{vs}\| \geq r_{excl}, \quad r_{excl} = \max(s_{base}, \mathrm{median}(\{s_i^{vs}\}))$$

The scale of uniformly sampled points is estimated via Query-to-Reference KNN: $s_j^{us} = \sqrt{\frac{1}{K}\sum_{\mathbf{q} \in \mathcal{N}_K(\mathbf{x}_j^{us}, \mathcal{X})}\|\mathbf{x}_j^{us} - \mathbf{q}\|^2}$, with $K=3$.

### Key Design 3: Scale-Adaptive Color Initialization

For each sample point $\mathbf{x}_i$, a neighborhood of radius $s_i$ is defined, and the color is estimated using Gaussian-weighted median filtering:

$$\mathbf{c}_i^{(d)} = \arg\min_{z \in \mathbb{R}} \sum_{\mathbf{u} \in \mathcal{N}_{\mathbf{x}_i}} w_i(\mathbf{u}) \cdot |z - \mathbf{I}^{(d)}(\mathbf{u})|$$

Compared to random initialization or pixel-center estimation, the weighted median is more robust to noise and outliers.

### Loss & Training

The ratio of variational to uniform sampling is $\lambda_g = 0.7$ (70% variational, 30% uniform). The loss function is:

$$L = \lambda_l \|\hat{\mathbf{I}} - \mathbf{I}\|_1 + (1 - \lambda_l)(1 - \mathrm{SSIM}(\hat{\mathbf{I}}, \mathbf{I})), \quad \lambda_l = 0.9$$

## Key Experimental Results

### Main Results on DIV8K (average resolution 5736×6120, average size 53.56 MB)

| CR | 3DGS | LIG | GI (RS) | GI (Cholesky) | ImageGS | **SmartSplat** |
|------|------|------|---------|---------------|---------|----------------|
| 20× | 30.99/0.9636 | 28.05/0.9362 | 30.45/0.9707 | 30.33/0.9698 | 32.00/0.8680 | **33.26/0.9752** |
| 50× | 28.56/0.9340 | 24.90/0.8402 | 26.99/0.9291 | 26.87/0.9271 | 29.47/0.8052 | **29.65/0.9482** |
| 100× | 26.84/0.8990 | 22.91/0.7230 | 25.00/0.8827 | 24.90/0.8790 | 26.65/0.7449 | **27.49/0.9164** |
| 200× | 24.92/0.8556 | 21.06/0.5792 | 23.45/0.8223 | 23.35/0.8176 | 26.80/0.7181 | **25.75/0.8745** |
| 500× | 22.38/0.7874 | 17.68/0.3633 | Fail | Fail | 24.88/0.6544 | **23.82/0.8055** |
| 1000× | 20.38/0.7068 | 12.49/0.2083 | Fail | Fail | 23.50/0.6165 | **22.66/0.7469** |

Metrics are reported as PSNR (dB) / MS-SSIM. At 20×, SmartSplat outperforms the second-best method by 1.26 dB in PSNR. At 500× and 1000×, GI fails completely while SmartSplat remains stable.

### Results on DIV16K (average resolution 12684×15898, average size 235.52 MB)

| CR | 3DGS | GI (RS) | **SmartSplat** |
|------|------|---------|----------------|
| 50× | OOM | 29.24/0.7917 | **34.34/0.9267** |
| 100× | OOM | 27.39/0.7648 | **33.00/0.9117** |
| 200× | OOM | 25.63/0.7394 | **31.85/0.8897** |
| 500× | 28.61/0.8117 | Fail | **29.40/0.8524** |
| 1000× | 27.06/0.7854 | Fail | **27.49/0.8226** |
| 2000× | 25.54/0.7642 | Fail | **25.70/0.7966** |
| 3000× | Fail | Fail | **24.72/0.7844** |

SmartSplat is the only method capable of completing training at a 3000× compression ratio, achieving an average PSNR gain of approximately 5.64 dB over GI on DIV16K.

### Efficiency Comparison (10848×16320 image, CR=200)

| Method | Iteration Speed | Training Time (s) | Memory (GB) | FPS | PSNR |
|------|---------|-----------|---------|-----|------|
| 3DGS (10K) | 1.32 it/s | 7841.80 | 50.19 | 10.98 | 24.42 |
| GI (10K) | 7.44 it/s | 1334.73 | 16.29 | 62.33 | 19.86 |
| SmartSplat (10K) | 5.01 it/s | 2237.52 | 19.59 | 32.35 | **31.87** |
| SmartSplat (1K) | 5.03 it/s | 336.12 | 19.38 | 33.12 | 30.52 |

With only 1K iterations, SmartSplat achieves 30.52 dB, surpassing both 3DGS and GI at 10K iterations. GPU memory usage is only 39% of that required by 3DGS.

### Ablation Study (4416×6720 image, CR=200, 10K iterations)

| Configuration | PSNR (dB) | MS-SSIM |
|------|----------|---------|
| Full Random | 22.34 | 0.8435 |
| +VS/US position initialization | 22.18 | 0.8270 |
| +VS/US scale initialization | 23.12 | 0.8647 |
| +Scale-adaptive color (full SmartSplat) | **24.38** | **0.8972** |

Scale initialization contributes the most (+0.94 dB), and color initialization provides a further gain of 1.26 dB; all three components are indispensable.

## Highlights & Insights

- **First UHR GS compression framework**: SmartSplat is the first to extend GS-based image compression to the 8K/16K scale, supporting extreme compression ratios up to 5000×.
- **Parameter-free scale initialization**: $s_{base}$ is derived entirely from image resolution and compression ratio, requiring no manual tuning or heuristic clamping.
- **Three-stage joint initialization**: The coordinated initialization of position, scale, and color enables SmartSplat to surpass baseline results at 10K iterations using only 1K iterations.
- **Robust color estimation**: Gaussian-weighted median filtering outperforms random initialization and pixel-center estimation, particularly in high-frequency texture regions.
- **Exceptional scalability**: SmartSplat operates stably in scenarios where competing methods fail due to OOM errors or NaN values.

## Limitations & Future Work

- **Spatial distribution only**: The current framework focuses on optimizing the spatial distribution of Gaussians without addressing further compression of Gaussian attributes (color, opacity) via quantization or entropy coding, which represents an important direction for improving compression efficiency.
- **DIV16K dataset construction**: The dataset is generated from DIV2K via super-resolution tools, which may introduce distributional discrepancies in texture details compared to natively captured 16K images.
- **Limited evaluation scale**: Only 16 images from DIV8K and 8 images from DIV16K are evaluated, raising concerns about statistical significance.
- **Moderate decoding speed**: The FPS of approximately 32 is substantially better than INR methods but falls short of GI's 62 FPS, leaving room for improvement in real-time applications.
- **No comparison with neural codecs**: Comparisons with end-to-end learned image codecs (e.g., Hyperprior, ELIC) are absent.

## Related Work & Insights

- **GaussianImage (GI)**: Employs two-stage optimization with vector quantization; at high compression ratios, insufficient Gaussians frequently cause NaN failures. SmartSplat avoids this through adaptive initialization and achieves a PSNR gain of 2.57 dB at the same CR.
- **LIG**: The hierarchical Gaussian strategy prioritizes fitting accuracy over compression performance and requires a large number of Gaussian components; performance degrades sharply at high CRs.
- **ImageGS**: Content-aware initialization combined with progressive training becomes unstable under extreme compression; ImageGS fails to run on DIV16K due to OOM.
- **3DGS**: Direct extension to 2D yields reasonable results but suffers from slow training and high memory consumption due to identity matrix mapping (50 GB vs. SmartSplat's 20 GB).
- **General paradigm for feature-guided initialization**: Using image gradients and color variance to guide the spatial distribution of discrete primitives is a principle generalizable to other primitive-based representations, such as point cloud compression and NeRF initialization.
- **Explicit relationship between compression ratio and primitive count**: The formula $N_g = 3HW / (7 \cdot \mathrm{CR})$ clearly establishes the connection between compression ratio and representational capacity, providing an analytical framework for future work.
- **Repulsion sampling**: Constraining sample point overlap via repulsion radii is analogous to Poisson Disk Sampling in computer graphics and can be applied to other scenarios requiring uniform spatial coverage.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Extends GS compression to the UHR regime with a well-designed three-stage initialization strategy; however, the core idea of feature-guided sampling is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐ — Experimental design is comprehensive (main results + ablation + efficiency comparison), but the evaluation set is small (16 + 8 images) and comparisons with neural codecs are missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear, method descriptions are detailed, and figures are informative; some notation is somewhat verbose.
- **Value**: ⭐⭐⭐⭐ — Achieves significant progress in UHR image compression, an area with strong practical demand; open-sourced code enhances reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](../../ICCV2025/3d_vision/one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)
- [\[NeurIPS 2025\] High Resolution UDF Meshing via Iterative Networks](../../NeurIPS2025/3d_vision/high_resolution_udf_meshing_via_iterative_networks.md)
- [\[NeurIPS 2025\] ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS](../../NeurIPS2025/3d_vision/zpressor_bottleneck-aware_compression_for_scalable_feed-forward_3dgs.md)
- [\[AAAI 2026\] IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution](ie-srgs_an_internal-external_knowledge_fusion_framework_for_high-fidelity_3d_gau.md)

</div>

<!-- RELATED:END -->
