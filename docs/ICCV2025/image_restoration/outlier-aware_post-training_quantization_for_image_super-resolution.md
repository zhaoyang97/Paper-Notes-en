---
title: >-
  [Paper Note] Outlier-Aware Post-Training Quantization for Image Super-Resolution
description: >-
  [ICCV 2025][Image Restoration][Post-training quantization] This paper proposes an outlier-aware post-training quantization method for image super-resolution. It introduces a dual-region piecewise linear quantizer to bala…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Post-training quantization"
  - "image super-resolution"
  - "activation outliers"
  - "piecewise linear quantization"
  - "sensitivity-aware"
date: 2026-05-08
content_hash: b60e1297bcadf01b
---

# Outlier-Aware Post-Training Quantization for Image Super-Resolution

**Conference**: ICCV 2025
**arXiv**: [2511.00682](https://arxiv.org/abs/2511.00682)  
**Code**: N/A  
**Area**: Image Super-Resolution / Model Quantization
**Keywords**: Post-training quantization, image super-resolution, activation outliers, piecewise linear quantization, sensitivity-aware

## TL;DR

This paper proposes an outlier-aware post-training quantization method for image super-resolution. It introduces a dual-region piecewise linear quantizer to balance outlier preservation with normal activation fidelity, and incorporates a sensitivity-aware finetuning strategy that directs attention to quantization-sensitive layers. Under the W4A4 setting, the method substantially outperforms existing PTQ approaches and approaches QAT-level performance.

## Background & Motivation

Quantizing SR models presents unique challenges. Existing PTQ methods perform poorly on SR tasks, primarily because they neglect the outlier problem in activations. The authors' empirical analysis reveals two key observations:

**Observation 1: Outliers are strongly correlated with color information.** Outliers are prevalent in SR network activation distributions. Most activations concentrate in a narrow range (e.g., $[-50, 50]$), yet extreme values far exceeding this range appear occasionally. Clipping just 1% of outliers leads to noticeable color distortion (e.g., color fading in flowers), indicating that outliers encode critical color information.

**Observation 2: Different layers exhibit drastically different sensitivity to quantization.** In SRResNet, quantizing the `head.0` layer causes PSNR to drop from 32.06 dB to 18.26 dB, while quantizing `body.4.conv1` still maintains 31.20 dB. This heterogeneity demands differentiated quantization strategies across layers.

**Key Challenge**: Preserving outliers consumes a disproportionate share of bit capacity, compressing the representational space for normal activations; clipping outliers, on the other hand, leads to severe performance degradation.

## Method

### Overall Architecture

A two-stage pipeline: (1) **Calibration stage** — quantization parameters are estimated using 100 DIV2K low-resolution images; (2) **Finetuning stage** — quantization parameters are refined via a sensitivity-aware loss without requiring labeled data.

### Key Designs

1. **Piecewise Linear Quantizer (PLQ)**:

    - **Core Idea**: The activation distribution is partitioned into two non-overlapping regions, each quantized uniformly.
    - A learnable breakpoint $bp$ divides the activation range $[l_a, u_a]$ into:
        - Dense region $R_1 = [-bp, bp]$: contains the majority of normal activations.
        - Outlier region $R_2 = [l_a, -bp) \cup (bp, u_a]$: contains extreme values.
    - For $b$-bit quantization, $2^{b-1}-1$ quantization points are allocated to the dense region, and $2^{b-2}-1$ points each to the positive and negative sides of the outlier region.
    - **Initialization**: $l_a$ = minimum activation value, $u_a$ = maximum activation value, $bp$ = 99th percentile value.
    - Subsequent batches update parameters via exponential moving average (EMA, $\beta=0.9$).

2. **Sensitivity-Aware Finetuning (SAFT)**:

    - Layer quantization sensitivity is computed as:
      $$s_k = \frac{\exp\!\left(\frac{1}{N}\sum_{x \in D_{cal}} \sigma(x_k)\right)}{\sum_{j=1}^K \exp\!\left(\frac{1}{N}\sum_{x \in D_{cal}} \sigma(x_j)\right)}$$
    - Here $\sigma(x_k)$ denotes the standard deviation of the feature map at layer $k$, normalized via softmax.
    - Layers with higher variance are assigned higher sensitivity weights and receive greater attention during quantization.

3. **Staged Parameter Optimization**:

    - epoch mod 3 = 1: update weight upper bound $u_w$.
    - epoch mod 3 = 2: update activation bounds $l_a, u_a$.
    - epoch mod 3 = 0: update breakpoint $bp$.
    - Parameters are iteratively refined in a cyclic fashion.

### Loss & Training

Total loss $L_{all} = \mathcal{L}_{sen} + \lambda \mathcal{L}_{rec}$ ($\lambda = 5$):
- **Reconstruction loss**: $\mathcal{L}_{rec} = \frac{1}{N}\sum_{i=1}^N \|\mathcal{K}(I_{lr}^i) - \mathcal{Q}(I_{lr}^i)\|_1$, the L1 distance between the full-precision and quantized network outputs.
- **Sensitivity-aware loss**: $\mathcal{L}_{sen} = \frac{s_k}{K}\sum_{k=1}^K \left\|\frac{F_\mathcal{K}^k}{\|F_\mathcal{K}^k\|_2} - \frac{F_\mathcal{Q}^k}{\|F_\mathcal{Q}^k\|_2}\right\|_2$

Key advantage: **only low-resolution images are required for training** — no high-resolution ground truth is needed.

## Key Experimental Results

### Main Results

PSNR comparison of EDSR and RDN under the W4A4 setting:

| Method | Finetune | Set5 | Set14 | BSD100 | Urban100 |
|--------|----------|------|-------|--------|----------|
| EDSR FP32 | - | 32.10 | 28.58 | 27.56 | 26.04 |
| EDSR-MinMax | ✗ | 26.83 | 25.04 | 24.57 | 23.12 |
| EDSR-PTQ4SR | ✓ | 30.51 | 27.62 | 26.88 | 24.92 |
| EDSR-AdaBM | ✓ | 31.02 | 27.87 | 26.91 | 25.11 |
| **EDSR-Ours** | **✓** | **31.54** | **28.26** | **27.36** | **25.61** |
| RDN FP32 | - | 32.24 | 28.67 | 27.63 | 26.29 |
| RDN-PTQ4SR | ✓ | 28.32 | 26.11 | 25.82 | 23.31 |
| RDN-AdaBM | ✓ | 28.71 | 26.30 | 26.10 | 23.38 |
| **RDN-Ours** | **✓** | **31.80** | **28.39** | **27.47** | **25.93** |

On RDN W4A4 Urban100, the proposed method surpasses the second-best by **2.55 dB**.

Comparison with QAT methods (EDSR W4A4):

| Method | Requires GT | Processing Time | Set5 | Set14 | BSD100 | Urban100 |
|--------|-------------|-----------------|------|-------|--------|----------|
| PAMS (QAT) | ✓ | 75× | 31.59 | 28.20 | 27.32 | 25.32 |
| ODM (QAT) | ✓ | 120× | 32.00 | 28.47 | 27.51 | 25.80 |
| **Ours (PTQ)** | **✗** | **1×** | **31.79** | **28.40** | **27.45** | **25.75** |

The proposed method achieves QAT-comparable performance at **over 75× speedup**.

### Ablation Study

Component ablation on EDSR W4A4:

| PLQ | SAFT | VFT | Set5 PSNR | Set14 PSNR | BSD100 PSNR | Urban100 PSNR |
|-----|------|-----|-----------|------------|-------------|---------------|
| ✗ | ✗ | ✗ | 26.83 | 25.04 | 24.57 | 23.12 |
| ✓ | ✗ | ✗ | 30.50 | 27.71 | 27.03 | 25.12 |
| ✗ | - | ✓ | 29.45 | 26.95 | 26.27 | 24.40 |
| ✗ | ✓ | - | 29.87 | 27.24 | 26.55 | 24.57 |
| ✓ | ✓ | - | **31.54** | **28.26** | **27.36** | **25.61** |

PLQ alone contributes the most (Set5 +3.67 dB); SAFT outperforms vanilla finetuning (VFT).

### Key Findings

- **Outlier preservation is critical**: Both MinMax (preserving outliers but squeezing normal values) and Percentile (clipping outliers) are suboptimal; the dual-region strategy yields the best results.
- **PLQ is the primary contributing module**: PLQ alone improves Set5 PSNR from 26.83 to 30.50 dB.
- **SAFT outperforms vanilla finetuning**: Sensitivity-weighted training is more effective than treating all layers equally (0.42 dB gap on Set5).
- **Highly efficient**: The entire PTQ process requires only 73 seconds, with inference latency on par with the baseline.

## Highlights & Insights

- The discovery of the **outlier–color information correlation** is insightful and offers a new perspective on outlier handling in quantization.
- The **piecewise quantization** approach is simple yet effective, decomposing a difficult single-interval quantization problem into two simpler sub-problems.
- **No GT data required**: PTQ relies solely on low-resolution calibration images, substantially lowering the barrier to deployment.
- The **75× speedup** over QAT methods offers significant practical value.

## Limitations & Future Work

- The breakpoint position $bp$ (fixed at the 99th percentile) may not be globally optimal; adaptive selection warrants investigation.
- Validation is limited to classic SR architectures (EDSR, RDN, SRResNet); experiments on Transformer-based SR models (e.g., SwinIR) are absent.
- The dual-region partition is fixed; finer-grained multi-region strategies could be explored.
- Hardware compatibility of piecewise quantization in practical deployment requires further verification.

## Related Work & Insights

- The dual-region quantization idea is transferable to other quantization tasks involving outliers (e.g., KV-cache quantization in LLMs).
- The sensitivity-aware strategy can be generalized to bit-width allocation decisions in mixed-precision quantization.
- The outlier–color correlation finding may have reference value for quantization in other image generation and restoration tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The outlier–color correlation finding and the dual-region strategy are genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-model, multi-dataset validation with comprehensive PTQ/QAT comparisons, ablation studies, and visual analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The observation→motivation→method narrative flows naturally, with clear figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — An efficient PTQ solution of high practical value for SR model deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [\[NeurIPS 2025\] Spend Wisely: Maximizing Post-Training Gains in Iterative Synthetic Data Bootstrapping](../../NeurIPS2025/image_restoration/spend_wisely_maximizing_post-training_gains_in_iterative_synthetic_data_bootstra.md)
- [\[NeurIPS 2025\] Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement](../../NeurIPS2025/image_restoration/luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)
- [\[ICCV 2025\] IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution](im-lut_interpolation_mixing_look-up_tables_for_image_super-resolution.md)
- [\[ICCV 2025\] FoundIR: Unleashing Million-scale Training Data to Advance Foundation Models for Image Restoration](foundir_unleashing_million-scale_training_data_to_advance_foundation_models_for_.md)

</div>

<!-- RELATED:END -->
