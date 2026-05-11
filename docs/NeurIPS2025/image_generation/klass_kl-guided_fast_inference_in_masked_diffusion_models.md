---
title: >-
  [Paper Note] KLASS: KL-Guided Fast Inference in Masked Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Masked Diffusion Models] This paper proposes KLASS (KL-Adaptive Stability Sampling), a training-free sampling method that leverages token-level KL divergence and confidence scores to iden…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Masked Diffusion Models"
  - "KL Divergence"
  - "Accelerated Sampling"
  - "Token Stability"
  - "Parallel Decoding"
date: 2026-05-08
content_hash: 7c0b4270df323404
---

# KLASS: KL-Guided Fast Inference in Masked Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.05664](https://arxiv.org/abs/2511.05664)
**Code**: [GitHub](https://github.com/shkim0116/KLASS)
**Area**: Image Generation / Discrete Diffusion Model Sampling
**Keywords**: Masked Diffusion Models, KL Divergence, Accelerated Sampling, Token Stability, Parallel Decoding

## TL;DR

This paper proposes KLASS (KL-Adaptive Stability Sampling), a training-free sampling method that leverages token-level KL divergence and confidence scores to identify stable tokens for parallel decoding, achieving up to 2.78× speedup on masked diffusion models without sacrificing—and in many cases improving—generation quality.

## Background & Motivation

**Background**: Masked diffusion models (MDMs) have demonstrated competitive performance across language generation, image generation, and molecular generation tasks, with large-scale models such as LLaDA and DREAM having acquired reasoning capabilities.

**Limitations of Prior Work**: The sampling process of MDMs relies on iterative unmasking, typically employing fixed Top-k or random sampling strategies that unmask only a small number of tokens per step, resulting in slow inference.

**Key Challenge**: A fundamental trade-off exists between accelerated sampling (unmasking more tokens per step) and generation quality—prematurely unmasking unstable tokens leads to degraded accuracy.

**Goal**: To enable safe parallel unmasking of multiple tokens to accelerate generation, without requiring additional training or external planners.

**Key Insight**: The model's own internal signals—KL divergence and confidence scores—are used to determine whether a token is "stable" and thus safe to unmask early.

**Core Idea**: Tokens exhibiting low KL divergence and high confidence are considered "stable" and can be unmasked in parallel; incorrectly predicted tokens cannot maintain dynamic stability throughout the diffusion process.

## Method

### Overall Architecture

At each step of the reverse sampling process in a standard masked diffusion model, KLASS computes two metrics for all masked tokens—a confidence score and a KL score—and then adaptively selects tokens satisfying the stability criteria for batch unmasking.

### Key Designs

1. **Confidence Score**: Defined as the maximum probability value in the model's predicted distribution:
    $\text{conf}_t^i = \max_v p_t^i(v)$
   High confidence indicates greater certainty in the model's estimate for that token.

2. **KL Score**: Defined as the KL divergence between predicted distributions at adjacent time steps:
    $d_t^i = D_{\text{KL}}(p_t^i \| p_{t+1}^i)$
   A low KL score indicates that the model's estimate for the token is temporally consistent and stable. Empirical findings show that correctly predicted tokens consistently exhibit lower KL scores.

3. **Stable Token Selection**: Given history length $n$, KL threshold $\epsilon_{\text{KL}}$, and confidence threshold $\tau$, the stable token set is defined as:
    $S_t = \{i \mid \forall k \in \{1,...,n\}, D_{\text{KL}}(p_{t+k-1}^i \| p_{t+k}^i) < \epsilon_{\text{KL}} \wedge \text{conf}_t^i > \tau\}$

4. **Unmasking Rule**: If stable tokens exist ($S_t \neq \emptyset$), all are unmasked simultaneously; otherwise, the method falls back to Top-$u$ confidence-based unmasking.

### Theoretical Support

**Proposition 5.3**: For a well-trained model, if a token's current prediction is incorrect, its predicted distribution will necessarily undergo significant change as context is progressively revealed (the average KL divergence is lower-bounded). This implies that **incorrectly predicted tokens cannot maintain dynamic stability**, and KLASS therefore avoids errors by deferring the unmasking of unstable tokens.

### Computational Overhead

KL computation is a lightweight post-processing operation requiring no additional forward passes. Experiments show memory overhead of less than 1.57% and latency overhead of less than 0.21%.

## Key Experimental Results

### Main Results: Reasoning Tasks

| Method | Model | MATH Acc↑ | Steps↓ | GSM8K Acc↑ | Steps↓ | HumanEval Acc↑ | Steps↓ |
|--------|-------|-----------|--------|------------|--------|----------------|--------|
| Top-1 | LLaDA | 31.4 | 256 | 75.13 | 256 | 39.63 | 256 |
| Confidence | LLaDA | 31.6 | 96.5 | 75.21 | 74.4 | 37.80 | 54.4 |
| **KLASS** | **LLaDA** | **33.8** | **128.6** | **76.50** | **98.6** | **40.85** | **92.0** |
| Top-1 | DREAM | 37.97 | 256 | 79.55 | 256 | 58.53 | 256 |
| Confidence | DREAM | 41.80 | 95.1 | 73.67 | 74.8 | 50.00 | 52.5 |
| **KLASS** | **DREAM** | **43.20** | **149.7** | **79.43** | **155.7** | **59.35** | **74.9** |

KLASS outperforms standard Top-1 decoding on nearly all tasks while reducing the number of steps by 40–70%, achieving a wall-clock speedup of up to 2.78×.

### Text Generation

| Method | MAUVE↑ | LLaMA2 PPL↓ | LLaMA3 PPL↓ | GPT-2 PPL↓ |
|--------|--------|-------------|-------------|------------|
| MDLM | 0.115 | 30.88 | 54.15 | 51.78 |
| **KLASS** | **0.179** | **26.94** | **49.19** | **45.50** |

### Image Generation (MMaDA)

| Method | Steps | FID↓ | IS↑ |
|--------|-------|------|-----|
| Confidence | 16 | 34.48 | 75.72 |
| **KLASS** | **16** | **30.48** | **93.07** |

### Ablation Study

| Unmasking Strategy | MATH Acc↑ | Steps↓ |
|--------------------|-----------|--------|
| Single (conf) | 31.2 | 256 |
| Single (KL) | 29.0 | 256 |
| **Parallel (KLASS)** | **33.8** | **128.6** |

Parallel unmasking of stable tokens outperforms single-token selection strategies while requiring fewer steps.

### Key Findings

- The combination of KL score and confidence is essential—neither criterion alone is sufficient.
- Correctly predicted tokens consistently exhibit lower KL scores (empirically verified in Figure 1b).
- KLASS proves effective across text, image, and molecular modalities, demonstrating its generality.

## Highlights & Insights

- **Training-free universal sampler**: KLASS does not modify model parameters and relies purely on statistical signals derived from the model's own inference process.
- **Theory-practice alignment**: Proposition 5.3 provides a theoretical justification for why KL divergence can discriminate between correct and incorrect tokens.
- **Cross-modal generality**: A single method applies to language, image, and molecular generation—a rare property among diffusion model samplers.
- **Simultaneous acceleration and quality improvement**: KLASS not only speeds up inference but also improves accuracy on multiple benchmarks, challenging the conventional speed–quality trade-off.

## Limitations & Future Work

- Hyperparameters (KL threshold, confidence threshold) require tuning across different models and tasks; although the authors claim low sensitivity, grid search is still necessary.
- Validation is currently limited to masked diffusion models and has not been extended to other discrete diffusion formulations (e.g., uniform noise schedules).
- The choice of history length $n$ affects performance, with longer histories incurring greater caching overhead.
- Performance on larger-scale models remains to be verified.

## Related Work & Insights

- Compared to concurrent work such as Fast-dLLM, Dimple, and EB-Sampler, KLASS distinguishes itself by incorporating KL divergence as a measure of "dynamic stability" in addition to confidence scores.
- The fine-grained, token-level monitoring of inference quality can inspire sampling strategy design in other discrete generative models.
- The KL-based "prior verification" mechanism has potential for extension to speculative decoding in autoregressive models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual-criterion approach of combining KL divergence and confidence for stable token selection is novel, though the core idea is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers four modalities—text reasoning, text generation, image generation, and molecular generation—with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-structured with concise and compelling theoretical sections and intuitive figures.
- **Value**: ⭐⭐⭐⭐ A plug-and-play acceleration method with high practical utility, though applicability is currently limited to the masked diffusion model ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [\[NeurIPS 2025\] Fast Data Attribution for Text-to-Image Models](fast_data_attribution_for_text-to-image_models.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)
- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)

</div>

<!-- RELATED:END -->
