---
title: >-
  [Paper Note] Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization
description: >-
  [ICML 2025][Image Generation][Diffusion model acceleration] MoDiff proposes a framework combining modulated quantization and error compensation to accelerate diffusion models. It reduces activation quantization from 8-bit to 3-bit without performance loss, while inheriting the dual advantages of both caching and quantization methods.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "quantization"
  - "caching"
  - "error compensation"
  - "post-training quantization"
date: 2026-05-08
content_hash: d7f93117a733b2d2
---

# Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

**Conference**: ICML 2025  
**arXiv**: [2506.22463](https://arxiv.org/abs/2506.22463)  
**Code**: [https://github.com/WeizhiGao/MoDiff](https://github.com/WeizhiGao/MoDiff)  
**Area**: Image Generation  
**Keywords**: Diffusion model acceleration, quantization, caching, error compensation, post-training quantization

## TL;DR
MoDiff proposes a framework combining modulated quantization and error compensation to accelerate diffusion models. It reduces activation quantization from 8-bit to 3-bit without performance loss, while inheriting the dual advantages of both caching and quantization methods.

## Background & Motivation

### Background

**Background**: Diffusion models exhibit outstanding performance in generative tasks, but iterative sampling incurs massive computational overhead. Acceleration techniques primarily include step reduction (distillation), caching (reusing intermediate computations), and quantization (low-precision computation).

**Limitations of Prior Work**: Although caching methods avoid redundant computation, the approximation errors they introduce accumulate over time steps. Quantization methods suffer from significant degradation in generation quality at low bitwidths (<8 bits). Combining the two approaches is highly prone to overlapping errors.

**Key Challenge**: The fundamental trade-off between aggressive acceleration (lower bitwidths/more caching) and generative quality. How can quantization and caching/approximation errors be mathematically understood and systematically controlled?

**Goal**: To provide a unified framework that deeply analyzes the error sources of caching and quantization, and designs effective error compensation mechanisms.

**Key Insight**: Starting from theoretical analysis, this work reveals the intrinsic connection between quantization and caching, designing "modulated quantization" to dynamically adjust quantization parameters to compensate for errors.

**Core Idea**: Actively compensate for quantization errors during the diffusion process by modulating quantization parameters, maintaining generative quality even at extremely low bitwidths.

## Method

### Overall Architecture

- **Input**: Pre-trained diffusion model + calibration data
- **Process**: (1) Analyze the activation distribution characteristics at each time step; (2) design modulation parameters for each time step; (3) dynamically adjust the quantization range and step size using modulation parameters during quantization.
- **Output**: Quantized diffusion model ready for low-bit inference.
- The entire process is Post-Training Quantization (PTQ), requiring no retraining.

### Key Designs

1. **Unified Analysis of Caching and Quantization**:

    - Theoretically proves that caching methods are essentially a special form of quantization—replacing the current step's value with the previous step is equivalent to infinitely coarse quantization.
    - Reveals the shared error source of both methods: the variation of activations across time steps.
    - **Design Motivation**: A unified perspective enables the design of a framework that simultaneously addresses both types of errors.

2. **Modulated Quantization**:

    - Learns a set of modulation parameters $\gamma_t, \beta_t$ for each time step $t$.
    - Applies a modulation transform to the activation: $\hat{x} = \gamma_t \cdot Q(x) + \beta_t$.
    - Modulation parameters are learned by minimizing the gap between the quantized output and the full-precision output.
    - **Design Motivation**: Activation distributions vary significantly across different time steps (due to differing noise scales), making fixed quantization parameters unsuitable.

3. **Error Compensation Mechanism**:

    - Tracks the residual errors introduced by quantization and compensates for them in subsequent steps.
    - Theoretically derives the upper bound of the error to ensure that the total error after compensation remains controllable.
    - Supports integration with caching methods.
    - **Design Motivation**: Quantization errors accumulate during iterative sampling, necessitating explicit error control.

### Loss & Training

- Modulation parameters are learned by minimizing the MSE loss: $\min_{\gamma, \beta} \|f(x) - (\gamma \cdot Q(x) + \beta)\|^2$
- Requires only a small amount of calibration data (forward propagation of a few hundred images) without full retraining.
- Supports both layer-wise and step-wise parameter optimization.

## Key Experimental Results

### Main Results

| Dataset | Model | Bitwidth | FID↓ | Gap to Full Precision |
|--------|------|--------|------|-------------|
| CIFAR-10 | DDPM | 8-bit | Virtually lossless | <0.5 |
| CIFAR-10 | DDPM | 4-bit | Slight degradation | ~1-2 |
| CIFAR-10 | DDPM | **3-bit** | **No significant degradation** | **<1** |
| LSUN | LDM | 8-bit | Virtually lossless | <0.5 |
| LSUN | LDM | **3-bit** | **No significant degradation** | **<1** |

### Ablation Study

| Configuration | FID | Description |
|------|-----|------|
| Standard PTQ (3-bit) | Significant degradation | Direct quantization without modulation |
| MoDiff (3-bit) | Virtually lossless | Modulated quantization is effective |
| Caching only | Slight degradation | Approximation error of caching |
| MoDiff + Caching | Optimal | Complementary to each other |
| Without error compensation | Moderate | Compensation is crucial for low bitwidths |

### Key Findings

- **Lossless 3-bit Quantization**: On CIFAR-10 and LSUN, 3-bit is sufficient to maintain full-precision performance.
- Criticality of Modulation Parameters: Different time steps require different quantization strategies.
- Error compensation is vital for scenarios with <4-bit.
- MoDiff, as a general framework, can accelerate all diffusion models.

## Highlights & Insights

1. **Theoretical Innovation**: A unified error analysis framework for caching and quantization.
2. **Extreme Compression**: 3-bit activation quantization (theoretically 8x speedup vs. FP32) without performance loss.
3. **Versatility**: As a PTQ method, it is plug-and-play and requires no modification to the pre-trained model.
4. **Theoretical Guarantees**: The derivation of the error upper bound provides reliability guarantees.

## Limitations & Future Work

1. Weight quantization is not fully explored (this paper focuses on activation quantization).
2. Actual hardware speedup in deployment requires support for low-bit operations.
3. Insufficient validation on larger models (such as SDXL, DiT-XL).
4. The storage overhead of modulation parameters requires a trade-off.

## Related Work & Insights

- Q-Diffusion, PTQD, etc., are primary baselines for diffusion model quantization.
- Caching methods like DeepCache provide complementary acceleration strategies.
- Inspiration: The idea of dynamic quantization parameters may be applicable to other iterative inference models (such as ARMs, MCTS).

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical framework of modulated quantization is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid validation on CIFAR-10 and LSUN.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations.
- Value: ⭐⭐⭐⭐ Highly practical value for diffusion model deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)
- [\[ICML 2025\] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [\[ICCV 2025\] DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization](../../ICCV2025/image_generation/dmq_dissecting_outliers_of_diffusion_models_for_post-training_quantization.md)
- [\[ICML 2025\] Compositional Scene Understanding through Inverse Generative Modeling](compositional_scene_understanding_through_inverse_generative_modeling.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](../../ICCV2025/image_generation/summdiff_generative_modeling_of_video_summarization_with_diffusion.md)

</div>

<!-- RELATED:END -->
