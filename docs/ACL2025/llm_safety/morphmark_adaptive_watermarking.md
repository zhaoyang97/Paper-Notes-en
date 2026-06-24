---
title: >-
  [Paper Note] MorphMark: Flexible Adaptive Watermarking for Large Language Models
description: >-
  [ACL 2025][LLM Safety][LLM watermarking] Through a multi-objective trade-off analysis framework, MorphMark reveals the critical role of the greenlist probability $P_G$ in the trade-off between watermark effectiveness and text quality. Based on this, it proposes a method to adaptively adjust the watermark strength $r$—strengthening the watermark when $P_G$ is high and weakening it when $P_G$ is low, thereby simultaneously improving watermark detectability and text quality with…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "LLM watermarking"
  - "red-green list"
  - "adaptive watermark strength"
  - "multi-objective optimization"
  - "text quality"
  - "model-agnostic"
date: 2026-05-08
content_hash: fd55d7191900e91a
---

# MorphMark: Flexible Adaptive Watermarking for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.11541](https://arxiv.org/abs/2505.11541)  
**Code**: None  
**Institution**: Tsinghua University & The Chinese University of Hong Kong, Shenzhen  
**Area**: AI Safety  
**Keywords**: LLM watermarking, red-green list, adaptive watermark strength, multi-objective optimization, text quality, model-agnostic

## TL;DR

Through a multi-objective trade-off analysis framework, MorphMark reveals the critical role of the greenlist probability $P_G$ in the trade-off between watermark effectiveness and text quality. Based on this, it proposes a method to adaptively adjust the watermark strength $r$—strengthening the watermark when $P_G$ is high and weakening it when $P_G$ is low, thereby simultaneously improving watermark detectability and text quality without relying on additional model training.

## Background & Motivation

### Background

- Sourcing and copyright protection of LLM-generated text are increasingly important requirements.
- Watermarking methods based on red-green lists are the mainstream approach: dividing the vocabulary into green and red lists and increasing the sampling probability of green list tokens.
- KGW (Kirchenbauer et al., 2023) is a pioneering method that uses a fixed hyperparameter $\delta$ to control watermark strength.

### Limitations of Prior Work

- **Fundamental Dilemma**: There is an inherent conflict between watermark effectiveness (detectability, robustness) and text quality.
- Stronger watermark $\rightarrow$ better detection, but decreased text quality; weaker watermark $\rightarrow$ high text quality, but hard to detect and vulnerable to attacks.
- Unbiased watermarking keeps the expected distribution unchanged but has poor robustness.
- Low-entropy watermarking avoids watermarking low-entropy tokens but requires the original model for detection.
- Methods training auxiliary models lack flexibility (model-specific, increasing deployment complexity and inference latency).

### Key Insight

- Existing methods treat watermark strength as a fixed hyperparameter, but the optimal watermark strength should vary across different token locations.
- The cumulative greenlist probability $P_G$ is the key factor determining the watermark effectiveness-quality trade-off.
- When $P_G$ is high, the overall benefit of increasing watermark strength is large; when $P_G$ is low, the marginal benefit of increasing watermark strength diminishes or even becomes negative.

## Method

### Overall Architecture

The trade-off between watermark effectiveness and text quality is modeled as a multi-objective trade-off analysis function $F(r) = T(r) + \omega\cdot W(r)$, where $T(r)$ uses the Bhattacharyya coefficient to measure the similarity between the original distribution and the watermarked distribution, and $W(r)$ measures the increment of the difference between the green and red list probabilities. Through theoretical derivation, a positive correlation between the optimal watermark strength $r^*$ and $P_G$ is identified, based on which an adaptive watermarking algorithm is designed.

### Key Designs

#### Key Design 1: Multi-Objective Trade-off Framework

- Text quality $T(r) = \text{BC}(P, \hat{P}) = P_G \cdot \sqrt{1 + r(1-P_G)/P_G} + (1-P_G) \cdot \sqrt{1-r}$
- Watermark effectiveness $W(r) = 2r(1-P_G)$
- Objective function $F(r) = T(r) + \omega\cdot W(r)$, where $\omega > 0$ is the weight.
- **Theorem 1**: For any $\omega > 0$, there exists an optimal $r^* \in (0,1)$ that maximizes $F$, and $\partial r^*/\partial P_G > 0$.
- The conclusion holds for any weight $\omega$, demonstrating its universality.

#### Key Design 2: Adaptive Watermark Strength Function

- $r = \varphi(P_G)$, modeled as a piecewise linear function:
    - When $P_G \le p_0$: $r = \epsilon$ (minimal watermarking to protect text quality at low-entropy positions)
    - When $P_G > p_0$: $r = \min(k_{\text{linear}} \cdot P_G, 1-\epsilon)$ (linear growth)
- $p_0$ is the watermark threshold, controlling when to initiate watermarking.
- Variants supporting exponential growth $z(x) = e^{k_{\text{exp}}\cdot x} - 1$ and logarithmic growth $z(x) = \ln(k_{\text{log}}\cdot x + 1)$ are also supported.

#### Key Design 3: Model-agnostic and Model-free

- No training of auxiliary models is required.
- No access to the original model is required for detection (uses z-score statistical detection).
- Supports end-to-end inference without increasing deployment complexity.
- Can be directly applied to any red-green list-based watermarking framework.

### Detection Method

- Uses standard z-score detection: $z = (|S|_G - \gamma|T|) / \sqrt{|T|\gamma(1-\gamma)}$
- If it exceeds the threshold, the text is determined to contain a watermark.
- Fully compatible with the KGW detection pipeline.

## Key Experimental Results

### Main Results: Watermarking Performance Comparison on OPT-1.3B

| Method | TPR@1%↑ | TPR@1%(Word-S/30%)↑ | Best F1↑ | PPL↓ |
|------|---------|---------------------|----------|------|
| KGW | 0.9900 | 0.8050 | 0.9268 | 11.50 |
| UW (Unbiased) | 1.0000 | 0.7425 | 0.9221 | 11.59 |
| DiPmark | 0.9975 | 0.7250 | 0.9138 | 11.50 |
| SWEET | 0.9975 | 0.8225 | 0.9501 | 11.51 |
| EWD | 1.0000 | 0.8450 | 0.9549 | 11.48 |
| **MorphMark_exp** | **1.0000** | **0.9600** | **0.9778** | 11.36 |
| **MorphMark_linear** | **1.0000** | **0.9275** | **0.9727** | 11.24 |

### OPT-2.7B Results

| Method | TPR@1%↑ | TPR@1%(Word-S/30%)↑ | Best F1↑ | PPL↓ |
|------|---------|---------------------|----------|------|
| KGW | 0.9950 | 0.8275 | 0.9098 | 10.93 |
| **MorphMark_exp** | **1.0000** | **0.9625** | **0.9686** | 10.51 |

### Efficiency Comparison

| Method | Generation Time (s) | Detection Time (ms) | Extra Memory (B) |
|------|-----------|-------------|------------|
| KGW | 11.50 | 33.81 | 0 |
| SWEET | 11.51 | 44.27 | 1.3 |
| MorphMark_exp | **11.36** | **34.17** | 0 |

### Key Findings

1. MorphMark significantly leads in robustness (after a 30% Word Substitution attack), with TPR@1% increasing from 0.8050 (KGW) to 0.9600.
2. Text quality (PPL) is even lower (better), breaking the conventional perception of a trade-off between effectiveness and quality.
3. It requires no extra memory or computational overhead, and the generation time is even slightly faster than the baseline.
4. The exponential growth function (exp) performs the best overall, while the linear and logarithmic variants also remain competitive.

## Highlights & Insights

- **Theory-Driven Design**: The adaptive strategy is derived through rigorous multi-objective optimization theory rather than heuristic parameter tuning.
- **Breaking the Effectiveness-Quality Trade-off**: Proves theoretically for the first time that an optimal dynamic watermark strength exists, which can simultaneously improve both objectives.
- **Critical Role of $P_G$**: Reveals the core impact of the cumulative green list probability on the watermarking trade-off for the first time, providing a theoretical foundation for subsequent work.
- The completely model-free design grants it extremely high value for practical deployment.

## Limitations & Future Work

- The specific form of the adaptive function $\varphi(P_G)$ (linear/exponential/logarithmic) and the parameters $p_0, k$ need to be tuned for different models.
- The theoretical analysis is based on single-token generation and does not consider cumulative effects in long sequences.
- The experimented model sizes are relatively small (OPT-1.3B/2.7B) and have not been validated on larger models (e.g., 70B+).
- The performance differences across different text domains (code, mathematics, creative writing, etc.) have not been thoroughly analyzed.

## Related Work & Insights

- KGW (Kirchenbauer et al., 2023) serves as the theoretical foundation and unified baseline for this work.
- SWEET and EWD are the closest competing methods, which also adopt entropy-aware strategies but require auxiliary models.
- Insight: The adaptive approach can be extended to other scenarios requiring a quality-effectiveness trade-off (e.g., noise injection in differential privacy).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Theory-driven adaptive watermarking is a meaningful contribution.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Complete derivation and proof of multi-objective optimization theory.
- **Practicality**: ⭐⭐⭐⭐⭐ — Model-free, zero extra overhead, plug-and-play.
- **Experimental Thoroughness**: ⭐⭐⭐ — Model sizes are relatively small, could be extended to larger models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/llm_safety/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[ACL 2025\] From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models](from_tradeoff_to_synergy_a_versatile.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](../../NeurIPS2025/llm_safety/steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)
- [\[ACL 2025\] Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)
- [\[ACL 2025\] SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models](saferoute_adaptive_model_selection_for_efficient_and_accurate_safety_guardrails_.md)

</div>

<!-- RELATED:END -->
