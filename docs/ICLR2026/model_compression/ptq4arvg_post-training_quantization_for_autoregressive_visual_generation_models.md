---
title: >-
  [Paper Note] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models
description: >-
  [ICLR 2026][Model Compression][Visual Generation] PTQ4ARVG is proposed as the first systematic PTQ framework for autoregressive visual generation (ARVG) models. It addresses three ARVG-specific quantization challenges vi…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Visual Generation"
  - "Autoregressive Models"
  - "Post-Training Quantization"
  - "Activation Quantization"
  - "Outlier Suppression"
date: 2026-05-08
content_hash: 3f98144cb6368582
---

# PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models

**Conference**: ICLR 2026
**arXiv**: [2601.21238](https://arxiv.org/abs/2601.21238)  
**Code**: [GitHub](http://github.com/BienLuky/PTQ4ARVG)  
**Area**: Model Compression
**Keywords**: Visual Generation, Autoregressive Models, Post-Training Quantization, Activation Quantization, Outlier Suppression

## TL;DR

PTQ4ARVG is proposed as the first systematic PTQ framework for autoregressive visual generation (ARVG) models. It addresses three ARVG-specific quantization challenges via Gain-Projected Scaling (GPS), Static Token-Wise Quantization (STWQ), and Distribution-Guided Calibration (DGC).

## Background & Motivation

### Limitations of Prior Work

**Background**: Autoregressive visual generation models (VAR, RAR, PAR, MAR) have surpassed diffusion models in image generation quality, yet suffer from large model sizes (2–3B parameters) and slow inference (PAR-3B takes >3 seconds per image). Quantization is an effective acceleration technique, but applying existing methods to ARVG introduces three unique challenges:

**Channel-wise severe outliers**: Activations modulated by AdaLN exhibit extreme inter-channel range disparities.

**Token-wise highly dynamic activations**: Positional encodings cause drastic distribution shifts along the token dimension, and conditional tokens form sink tokens.

**Sample-level distribution mismatch**: Network activations are highly similar across different samples (especially unconditional ones), leading to redundancy in the calibration set.

## Method

### Overall Architecture

PTQ4ARVG comprises three targeted components addressing channel-level, token-level, and sample-level quantization challenges respectively, all in a training-free manner.

### Key Designs

1. **Gain-Projected Scaling (GPS)**:

    - Applies Taylor expansion to the quantization loss, separately quantifying activation and weight losses.
    - Defines the scaling gain as: $g(s_2) = g_{\bm{x}} - g_{\bm{W}_{:,1}}$ (reduction in activation loss minus increase in weight loss).
    - Derives a closed-form optimal scaling factor via differentiation: $s_2 = s_1 \frac{\sqrt{\sum|{\Delta W_{2,i} x_2}|}}{\sqrt{\sum|{W_{2,i} \Delta x_2}|}}$
    - Represents the first mathematically optimized quantization scaling strategy, outperforming empirically designed alternatives.

2. **Static Token-Wise Quantization (STWQ)**:

    - Exploits two distinctive properties of ARVG: **fixed token sequence length** and **position-invariant cross-sample distributions**.
    - Assigns static quantization parameters along the token sequence for AdaLN modules.
    - Handles sink tokens and regular tokens separately for linear layers.
    - Quantization parameters are set offline with no online calibration overhead, remaining compatible with standard CUDA kernels.

3. **Distribution-Guided Calibration (DGC)**:

    - Measures the distributional entropy of each sample via Mahalanobis distance: $\rho(x) = \sqrt{(x-u)^T S^{-1} (x-u)}$
    - Selects the top-50% samples with the highest distributional entropy as the calibration set.
    - Eliminates redundant samples to ensure that the calibration distribution matches the true data distribution.

### Loss & Training

- Entirely training-free PTQ.
- GPS derivation is based on Taylor expansion combined with convex optimization.
- STWQ employs percentile-based calibration (rather than min-max) to ensure high precision.
- Evaluation is conducted by generating 50K images on ImageNet, measuring FID, sFID, IS, and Precision.

## Key Experimental Results

### Main Results (VAR-d16 / VAR-d24 — W8A8 Quantization)

| Method | VAR-d16 FID ↓ | VAR-d16 IS ↑ | VAR-d24 FID ↓ | VAR-d24 IS ↑ |
|--------|-------------|-------------|-------------|-------------|
| FP | 3.60 | 283.21 | 2.33 | 317.16 |
| SmoothQuant | 4.29 | 229.87 | 4.42 | 246.68 |
| OS+ | 4.11 | 230.41 | 4.14 | 250.61 |
| OmniQuant | 4.19 | 226.92 | - | - |
| **PTQ4ARVG** | **3.82** | **268.19** | **2.69** | **304.82** |

### 6-bit Quantization Results (VAR-d24)

| Method | FID ↓ | IS ↑ | Precision ↑ |
|--------|------|------|------------|
| SmoothQuant W6A6 | >10 | <200 | Severe degradation |
| **PTQ4ARVG W6A6** | **~4.5** | **~280** | Competitive |

### Key Findings

- PTQ4ARVG substantially outperforms existing PTQ methods under both 8-bit and 6-bit settings.
- The mathematically optimized scaling of GPS consistently surpasses empirical approaches (SmoothQuant, RepQ-ViT).
- STWQ handles token-level variance without additional inference overhead; dynamic alternatives incur a 0.5× speed penalty.
- DGC significantly improves calibration quality by removing redundant samples.
- The framework is effective across all four ARVG model families: VAR, RAR, PAR, and MAR.

## Highlights & Insights

- Precise problem formulation: the paper is the first to systematically identify three distinct quantization challenges in ARVG, providing a dedicated solution for each.
- GPS is the first scaling strategy grounded in rigorous mathematical derivation, establishing a theoretical foundation for quantization scaling.
- STWQ cleverly exploits the fixed token length of ARVG — a property unavailable in LLMs due to variable-length sequences.
- Experiments cover four ARVG architectures (VAR/RAR/PAR/MAR), demonstrating strong generalizability.

## Limitations & Future Work

- 4-bit quantization results are not presented, likely due to severe accuracy degradation of ARVG models at 4-bit.
- Remark 1 in the GPS derivation is based on empirical observation rather than formal proof.
- Comparisons with recent methods such as SVDQuant are absent, though the latter relies on custom CUDA kernels.
- Given the relatively smaller scale of ARVG models compared to LLMs, the practical demand for quantization compression may be less urgent.

## Related Work & Insights

- **vs. SmoothQuant**: GPS replaces empirical scaling with mathematically optimized scaling.
- **vs. Dynamic quantization in LLMs**: STWQ leverages ARVG's fixed token length to achieve overhead-free static quantization.
- **vs. Diffusion model PTQ**: ARVG lacks timestep conditioning but exhibits token-level dynamics, necessitating a fundamentally different approach.
- **Insight**: Architecture-specific properties can be fully exploited to design more effective quantization methods.

## Rating

- Novelty: ⭐⭐⭐⭐ — First ARVG PTQ framework; GPS theoretical derivation is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive validation across four models with deployment verification.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is thorough, though formula derivations occupy substantial space.
- Value: ⭐⭐⭐⭐ — Establishes a foundation for efficient deployment of ARVG models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)
- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](../../ICCV2025/model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/model_compression/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](../../CVPR2026/model_compression/markovian_scale_prediction_a_new_era_of_visual_autoregressive_generation.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)

</div>

<!-- RELATED:END -->
