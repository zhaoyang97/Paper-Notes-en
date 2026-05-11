---
title: >-
  [Paper Note] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement
description: >-
  [ICCV 2025][Image Restoration][Image Enhancement] This paper proposes the BPAM framework, which combines the spatial modeling capability of bilateral grids with the nonlinear mapping power of MLPs by dynamically generati…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Image Enhancement"
  - "Bilateral Grid"
  - "Pixel-adaptive MLP"
  - "Real-time Processing"
  - "Color Mapping"
date: 2026-05-08
content_hash: cad1e81b4ba85160
---

# Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement

**Conference**: ICCV 2025
**arXiv**: [2507.12135](https://arxiv.org/abs/2507.12135)
**Code**: [GitHub](https://github.com/LabShuHangGU/BPAM)
**Area**: Image Restoration
**Keywords**: Image Enhancement, Bilateral Grid, Pixel-adaptive MLP, Real-time Processing, Color Mapping

## TL;DR

This paper proposes the BPAM framework, which combines the spatial modeling capability of bilateral grids with the nonlinear mapping power of MLPs by dynamically generating unique micro-MLP parameters for each pixel, enabling high-quality, real-time image enhancement.

## Background & Motivation

Image enhancement requires balancing quality and speed. Existing methods fall into two camps, each with notable limitations:

**End-to-end deep learning methods**: Represented by Restormer and NAFNet, these achieve high enhancement quality but incur large computational costs, making real-time processing of high-resolution images impractical.

**Hybrid methods**: Combining deep learning with efficient physical models:
   - **3D LUT methods**: Enable fast color mapping via lookup tables, achieving very high throughput (>1000 FPS), but 3D LUTs are purely image-level operators—they perform color transformations based solely on RGB values and completely ignore spatial context. Although methods such as SA-3DLUT attempt to introduce spatial awareness, the color mapping stage remains RGB-exclusive.
   - **Bilateral grid methods**: Represented by HDRNet, these naturally encode both spatial and intensity domain information and can efficiently map from low-resolution representations to full resolution via slicing. However, two bottlenecks exist: (a) only affine transformations are supported, precluding complex nonlinear color relationships; (b) the design inherits grayscale image conventions, fusing RGB channels into a single-channel guidance map for coefficient extraction, leading to insufficient utilization of color information.

**Global MLP methods**: Methods such as CSRNet replace traditional color transformations with MLPs, but parameters are shared across the entire image, limiting their ability to handle spatially varying color and illumination; deep hidden layers are required to achieve cross-region generalization.

The core insight of this paper is that if each pixel possesses its own unique MLP parameters, a very small MLP (only 3-8-3) can achieve high-quality spatially adaptive color transformation. Bilateral grids provide exactly the mechanism needed to efficiently generate spatially varying parameters.

## Method

### Overall Architecture

The BPAM pipeline proceeds as follows: (1) a three-level U-Net-style NAFNet backbone extracts features from a downsampled image; (2) Pixel Unshuffle further reduces the grid spatial size while increasing channel count; (3) $1\times1$ convolutions generate two bilateral grids storing the parameters of the first and second MLP layers, respectively; (4) a multi-channel guidance map is generated, and two slicing operations extract the complete MLP parameters for each pixel; (5) each pixel's color is transformed through its own micro-MLP.

### Key Designs

1. **Pixel-Adaptive MLP Learning**:

    - Function: Generates a unique three-layer MLP (3-8-3) for each pixel, converting color transformation from globally shared parameters to pixel-level adaptation.
    - Mechanism: The backbone network generates two bilateral grids. The first grid stores 32 parameters per cell ($\mathbf{W}_1 \in \mathbb{R}^{8\times3}$: 24 weights, plus $\mathbf{b}_1 \in \mathbb{R}^8$: 8 biases); the second grid stores 27 parameters per cell ($\mathbf{W}_2 \in \mathbb{R}^{3\times8}$: 24 weights, plus $\mathbf{b}_2 \in \mathbb{R}^3$: 3 biases). The color transformation proceeds in two stages:
        - Hidden layer: $\mathbf{z}(x,y) = \sigma(\mathbf{W}_1(x,y,s) \cdot \mathbf{I}(x,y) + \mathbf{b}_1(x,y,s))$
        - Output: $O(x,y) = \mathbf{W}_2(x,y,s) \cdot \mathbf{z}(x,y) + \mathbf{b}_2(x,y,s)$
    - Design Motivation: MLPs possess strong nonlinear modeling capacity, but global parameter sharing limits their ability to handle local variations. Pixel-adaptive parameters endow each pixel with a dedicated nonlinear transformation function, achieving high performance with an extremely compact MLP.

2. **Grid Decomposition**:

    - Function: Decomposes the bilateral grid into multiple sub-grids, each paired with an independent guidance channel.
    - Mechanism: MLP parameters are naturally grouped by category. For the first grid, the 32 parameters are divided into 4 sub-grids (one per RGB channel for weights, plus shared biases). For the second grid, the 27 parameters are divided into 9 sub-grids (one per hidden unit for weights, plus shared biases). Corresponding 4-channel and 9-channel guidance maps are generated, with each channel extracting parameters from its associated sub-grid.
    - Design Motivation: Conventional bilateral grids fuse RGB channels into a single-channel guidance map to extract all coefficients, causing color information loss at the slicing stage. The decomposition strategy grants each color channel an independent guidance and extraction path, fully exploiting color information—analogous to how 3D LUTs leverage the complete color space for mapping.

3. **Two-stage Guidance Map Generation**:

    - Function: Sequentially generates two sets of guidance maps for the two slicing operations.
    - Mechanism: A first convolutional network takes the original image as input and produces the first guidance map for extracting hidden-layer parameters → the hidden-layer vector is computed → a second convolutional network takes the hidden-layer vector as input and produces the second guidance map for extracting output-layer parameters.
    - Design Motivation: The two groups of MLP parameters are sequentially dependent; generating guidance maps in order allows the second stage to exploit intermediate results, yielding more accurate parameter estimation.

### Loss & Training

- **Total Loss**: $\mathcal{L} = \mathcal{L}_2 + 0.5 \times \mathcal{L}_{ssim} + 0.005 \times \mathcal{L}_{per}$
    - $\mathcal{L}_2$: MSE pixel loss
    - $\mathcal{L}_{ssim}$: SSIM structural loss
    - $\mathcal{L}_{per}$: Perceptual loss based on pretrained VGG19
- **Optimizer**: Adam with cosine annealing learning rate and an additional decay factor of 0.1
- **Adaptive Grid Size**: 1/4 resolution for PPR10K, 1/32 for FiveK full-resolution, 1/8 for others; grid depth fixed at 8
- **Efficient Implementation**: Slicing operations and MLP parameter application are accelerated via CUDA extensions

## Key Experimental Results

### Main Results

**FiveK Dataset Tone Mapping (Full Resolution)**

| Method | Params | PSNR↑ | SSIM↑ | ΔE↓ |
|--------|--------|-------|-------|-----|
| HDRNet | 482K | 24.17 | 0.919 | 8.91 |
| CSRNet | 37K | 24.23 | 0.920 | 8.75 |
| 3DLUT | 592K | 24.39 | 0.923 | 8.33 |
| LutBGrid | 464K | 24.57 | 0.931 | 8.03 |
| **BPAM (Ours)** | 624K | **25.12** | **0.934** | **7.73** |

**LCDP Dataset Exposure Correction**

| Method | Params | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|--------|-------|-------|--------|
| LCDPNet | 960K | 23.24 | 0.842 | 0.137 |
| CoTF | 310K | 23.89 | 0.858 | 0.104 |
| LutBGrid | 464K | 22.71 | 0.803 | 0.154 |
| **BPAM (Ours)** | 624K | **24.22** | **0.872** | **0.097** |

### Ablation Study

**Component Contribution Analysis (FiveK 480p Tone Mapping)**

| Config | Affine | MLP | Grid Decomp. | PSNR | SSIM |
|--------|--------|-----|-------------|------|------|
| Setting 1 | ✓ | | | 25.53 | 0.935 |
| Setting 2 | | ✓ | | 25.70 | 0.939 |
| Setting 3 | ✓ | | ✓ | 25.63 | 0.937 |
| **Setting 4** | | ✓ | ✓ | **25.83** | **0.941** |

**Inference Speed Comparison**

| Method | 1080p Latency (ms) | 4K FPS |
|--------|--------------------|--------|
| 3DLUT | 0.89 | 862 |
| LutBGrid | 1.66 | 319 |
| HDRNet | 11.8 | 22.8 |
| **BPAM** | 10.2 | **36.0** |
| CSRNet | 16.2 | 15.3 |

### Key Findings

1. Replacing affine transformation with MLP yields a +0.17 dB improvement; the grid decomposition strategy contributes an additional +0.13 dB, and the two components are complementary.
2. On full-resolution FiveK, BPAM outperforms LutBGrid by 0.55 dB, with the advantage becoming more pronounced at higher resolutions.
3. BPAM processes 4K content at over 30 FPS, meeting real-time requirements (3D LUT is faster but lacks spatial awareness).
4. The performance advantage is most prominent on exposure correction (+0.33 dB over CoTF), as extreme illumination conditions demand spatially adaptive nonlinear transformations.
5. A pixel-adaptive 3-8-3 micro-MLP suffices without requiring deep hidden layers or large channel counts.

## Highlights & Insights

- **Clever innovation in the parameter dimension**: Rather than making MLPs larger and deeper, the method assigns each pixel its own micro-MLP—trading spatial dimensions for depth, a genuinely novel perspective.
- **New use of bilateral grids**: Extending bilateral grids from storing affine coefficients to storing MLP parameters breaks the longstanding limitation of linear-only bilateral grid transformations.
- **Grid decomposition bridges two paradigms**: Bilateral grids gain the multi-channel color exploitation capability of 3D LUTs while retaining their spatial awareness advantage.
- **Strong practicality**: With only 624K parameters, 4K 30+ FPS throughput, and leading PSNR, this is a rare solution that achieves both quality and efficiency.
- **Engineering awareness in CUDA acceleration**: Slicing operations and MLP parameter application are implemented via custom CUDA kernels, ensuring that theoretical advantages translate into practical speed.

## Limitations & Future Work

1. Compared to 3D LUT methods (>800 FPS), BPAM's speed remains lower—a common trade-off for spatially aware methods.
2. The 3D bilateral grid requires manually specified grid-size ratios for different resolutions, lacking full automation.
3. The MLP architecture is fixed at 3-8-3; adaptive determination of the optimal structure has not been explored.
4. Validation is limited to enhancement, tone mapping, and exposure correction tasks; applicability to other restoration tasks such as denoising and deblurring remains unknown.
5. The perceptual loss weight (0.005) appears empirically chosen; a systematic hyperparameter sensitivity analysis is absent.

## Related Work & Insights

- **HDRNet**: Pioneering work on bilateral grids for image enhancement, but restricted to affine transformations.
- **CSRNet**: An early approach using global MLPs for color transformation, but with globally shared parameters.
- **3D LUT series** (SA-3DLUT, LutBGrid): Mainstream solutions for efficient color mapping, but with limited spatial awareness.
- Insight: Combining a strong representational structure (bilateral grid) with a powerful functional form (MLP) can surpass the limitations of each in a lightweight manner.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of storing MLP parameters in bilateral grids is original; the grid decomposition strategy is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, three tasks, clear ablations, and comprehensive speed comparisons.
- Writing Quality: ⭐⭐⭐⭐ Method motivation is clear, mathematical derivations are complete, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Provides a competitive new solution for real-time image enhancement with strong prospects for industrial application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)
- [\[ICCV 2025\] MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices](mobileie_an_extremely_lightweight_and_effective_convnet_for_real-time_image_enha.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[ICCV 2025\] Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions](metric_convolutions_a_unifying_theory_to_adaptive_image_convolutions.md)
- [\[ICCV 2025\] CWNet: Causal Wavelet Network for Low-Light Image Enhancement](cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)

</div>

<!-- RELATED:END -->
