---
title: >-
  [Paper Note] Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks (cmKAN)
description: >-
  [ICCV 2025][Model Compression][color matching] This paper proposes cmKAN, a hypernetwork-driven Kolmogorov-Arnold Network for color matching. A generator predicts spatially varying KAN spline parameters, supporting three scenarios (supervised / unsupervised / pairwise optimization) and three tasks (raw-to-raw / raw-to-sRGB / sRGB-to-sRGB). cmKAN outperforms existing methods by an average of 37.3% across all tasks while remaining extremely lightweight (76.4K parameters).
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "color matching"
  - "KAN"
  - "Kolmogorov-Arnold Network"
  - "hypernetwork"
  - "camera ISP"
  - "raw-to-sRGB"
  - "color transfer"
  - "lightweight"
date: 2026-05-08
content_hash: c3fe5b310da7d936
---

# Color Matching Using Hypernetwork-Based Kolmogorov-Arnold Networks (cmKAN)

**Conference**: ICCV 2025
**arXiv**: [2503.11781](https://arxiv.org/abs/2503.11781)  
**Code**: [https://github.com/gosha20777/cmKAN](https://github.com/gosha20777/cmKAN)  
**Institution**: Samara National Research University, University of Wurzburg, York University
**Area**: Model Compression / Color Matching / Lightweight Networks
**Keywords**: color matching, KAN, Kolmogorov-Arnold Network, hypernetwork, camera ISP, raw-to-sRGB, color transfer, lightweight

## TL;DR
This paper proposes cmKAN, a hypernetwork-driven Kolmogorov-Arnold Network for color matching. A generator predicts spatially varying KAN spline parameters, supporting three scenarios (supervised / unsupervised / pairwise optimization) and three tasks (raw-to-raw / raw-to-sRGB / sRGB-to-sRGB). cmKAN outperforms existing methods by an average of 37.3% across all tasks while remaining extremely lightweight (76.4K parameters).

## Background & Motivation
Different camera ISPs produce different color renderings; color matching aims to map source image colors to be consistent with a target. Limitations of prior work:
1. Polynomial methods fail to capture complex nonlinear transformations accurately.
2. Deep CNN/MLP methods are computationally expensive and parameter-heavy.
3. Most methods assume both source and target images are available.
4. No unified framework covers multiple scenarios.

## Core Problem
How to design a lightweight yet accurate general-purpose color matching framework?

## Method

### Core Idea: KAN Is Naturally Suited for Color Matching
The color matching formulation is $\hat{y} = F(x) \cdot L$. KAN models nonlinear mappings directly via trainable B-spline basis functions:
$$\hat{y}_j = \sum_i \left( u_{ij} \cdot \text{SiLU}(x_i) + v_{ij} \sum_m c_{ijm} B_{ijm}(x_i) \right)$$
- Only 90 parameters are needed to express a $3 \to 3$ color transformation.
- B-splines are smoother and more accurate than polynomials.

### Spatially Varying Hypernetwork KAN
Standard KAN operates globally and cannot handle spatially non-uniform chromatic aberration. The solution:
- Generator $G(X, \theta)$ produces 2D spatial parameter maps $W = (W_u, W_v, W_c)$.
- Each spatial location has its own independent set of 90 KAN parameters.
- $\hat{Y} = \text{KAN}(G(X, \theta), X)$

### Generator Network Architecture

1. **Illumination Estimator (IE)**:
    - A small CNN processes the input and channel-wise mean.
    - $1\times1$ conv $\to$ $3\times3$ dilated depthwise conv $\to$ $1\times1$ conv, outputting illumination features and an illumination map.
    - Prevents color distortion in overexposed/underexposed regions.

2. **Color Transformer (CT)**:
    - ViT-inspired architecture operating on DWT-downsampled inputs.
    - Multi-Scale Color Attention (MCA): operates along the channel dimension, introduces anchors as intermediate bridges, and applies spatial compression to Q and K to reduce computation; V is modulated by illumination features.

3. **Color Feature Modulator (CFM)**:
    - Processes concatenated features from IE and CT.
    - Modulates via linear projection with trainable biases: $X_m = B_i \cdot \text{ReLU}(X'' \cdot B_j)$.
    - Output is passed through an FFN to generate the final KAN parameter maps.

### Three Training Scenarios
1. **Supervised**: $\mathcal{L} = \mathcal{L}_1 + 0.15 \cdot (1 - \text{SSIM})$
2. **Unsupervised**: Pretraining (reconstruction under random color perturbation) + CycleGAN-like unpaired training.
3. **Pairwise Optimization**: Simplified cmKAN-Light variant; pretrained model is fine-tuned on a specific pair for only 10 steps with $\mathcal{L}_1$.

## Key Experimental Results

### Raw-to-Raw Unsupervised Mapping

| Method | PSNR | SSIM | ΔE |
|--------|------|------|-----|
| UVCGANv2 | 36.32 | 0.94 | 4.21 |
| RawFormer | 40.98 | 0.97 | 2.09 |
| **cmKAN** | **41.01** | **0.97** | **1.23** |

### sRGB-to-sRGB Supervised (Proprietary Dataset)

| Method | PSNR | SSIM | #Params | Time |
|--------|------|------|---------|------|
| MW-ISP | 23.31 | 0.76 | 29.2M | 8.8s |
| SIRLUT | 24.12 | 0.78 | 113.3K | 2.1s |
| **cmKAN** | **25.94** | **0.89** | **76.4K** | **1.1s** |

### MIT-Adobe FiveK

| Method | PSNR | ΔE |
|--------|------|-----|
| SIRLUT | 27.25 | 6.19 |
| **cmKAN** | **31.74** | **2.83** |

- PSNR +4.49 dB; ΔE reduced by 54%.

### Ablation Study
- MLP → KAN: +2.29 dB
- 1D global → 2D spatial: further improvement
- Incrementally adding IE / MCA / CFM: consistent gains, final +3.33 dB vs. baseline

## Highlights & Insights
- **KAN is theoretically natural for color matching**: well-grounded formulation.
- **Hypernetwork design enables spatial adaptivity**: elegantly handles spatially non-uniform chromatic aberration.
- **Extreme lightweight**: 76.4K parameters + 1.1s inference, 100–400× smaller than methods of comparable accuracy.
- **Unified framework across three scenarios**: supervised, unsupervised, and pairwise optimization all share the same architecture.
- **New dataset**: 2.5K aligned dual-camera image pairs.
- **User study**: MOS scores are approximately 2× those of competing methods.

## Limitations & Future Work
- A single-layer $3 \to 3$ mapping may be insufficient for extreme chromatic differences.
- The two-stage unsupervised pipeline adds complexity.
- Inference time of 1.1s is still too slow for real-time applications.
- Generalization across brands and illumination conditions is insufficiently validated.

## Related Work & Insights
- **vs. RawFormer**: 26.1M parameters vs. 76.4K (343× smaller), with comparable or superior performance.
- **vs. SIRLUT/SepLUT**: LUT-based methods lack precision; cmKAN's continuous splines are more accurate.
- **vs. NeuralPreset**: 5.15M + 20.4s vs. cmKAN-Light at 7.8K + 1.5s.

KAN outperforms MLP in low-dimensional signal processing, providing inspiration for other low-dimensional regression tasks. Proper inductive bias (splines) proves more effective than simply scaling parameters. The approach has direct engineering value for dual-camera color consistency and ISP development.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Theoretical naturality of KAN for color matching combined with hypernetwork-based spatial adaptivity.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three tasks × three scenarios + ablation + user study + new dataset.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear, but architectural descriptions are slightly verbose.
- **Value**: ⭐⭐⭐⭐⭐ — Extremely lightweight, high accuracy, and universally applicable across scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lookup multivariate Kolmogorov-Arnold Networks](../../ICLR2026/model_compression/lookup_multivariate_kolmogorov-arnold_networks.md)
- [\[ICCV 2025\] Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement](colors_see_colors_ignore_clothes_changing_reid_with_color_disentanglement.md)
- [\[ICCV 2025\] ARGMatch: Adaptive Refinement Gathering for Efficient Dense Matching](argmatch_adaptive_refinement_gathering_for_efficient_dense_matching.md)
- [\[ICCV 2025\] Variance-Based Pruning for Accelerating and Compressing Trained Networks](variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)
- [\[CVPR 2025\] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](../../CVPR2025/model_compression/jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)

</div>

<!-- RELATED:END -->
