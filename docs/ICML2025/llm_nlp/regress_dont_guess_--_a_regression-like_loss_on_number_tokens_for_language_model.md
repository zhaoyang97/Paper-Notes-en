---
title: >-
  [Paper Note] Regress, Don't Guess — A Regression-like Loss on Number Tokens for Language Models
description: >-
  [ICML 2025][LLM (Other)][number token loss] Proposes Number Token Loss (NTL), a pure token-level regression-like loss function that injects a numerical proximity inductive bias into LLMs by minimizing the $L_p$ norm or Wasserstein distance between target and predicted numerical tokens.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "number token loss"
  - "Wasserstein distance"
  - "numerical reasoning"
  - "regression loss"
  - "LLM"
date: 2026-05-08
content_hash: b1eb7fae9954e52f
---

# Regress, Don't Guess — A Regression-like Loss on Number Tokens for Language Models

**Conference**: ICML 2025  
**arXiv**: [2411.02083](https://arxiv.org/abs/2411.02083)  
**Code**: [GitHub (ntloss)](https://github.com/ai4sd/number-token-loss)  
**Area**: Language Modeling / Numerical Reasoning  
**Keywords**: number token loss, Wasserstein distance, numerical reasoning, regression loss, LLM  

## TL;DR

Proposes Number Token Loss (NTL), a pure token-level regression-like loss function that injects a numerical proximity inductive bias into LLMs by minimizing the $L_p$ norm or Wasserstein distance between target and predicted numerical tokens.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Four fundamental issues in how LLMs handle numbers:

**Tokenization**: Standard subword tokenization splits numbers into arbitrary tokens

**Embeddings**: Number token embeddings are learned like ordinary words, failing to preserve numerical relationships

**Sequential Prediction**: Token-by-token decoding ignores the greater importance of high-order digits

**Training Objective**: Cross-Entropy (CE) loss assumes a nominal scale, punishing a prediction of 3 and 9 equally when the ground truth is 2

## Method

### NTL-MSE ($L_p$-norm Family)

Maps predicted probabilities to a real-valued output, and then calculates the MSE against the ground truth:

$$\mathcal{L}_{\text{NTL-MSE}} = \frac{1}{N}\sum_i^N \left(y_i - \hat{\mathbf{y}}_i^{s:t} \circ \mathcal{V}^{s:t}\right)^2$$

where $\mathcal{V}: V \to \mathbb{R}$ maps tokens to numerical values, and $s...t$ represents the index range of the number tokens.

**Problem**: NTL-MSE suffers from **non-unique minima** — for example, when the label is 4, predicting 0 with 50% probability and 8 with 50% probability also yields zero loss.

### NTL-WAS (Wasserstein Distance)

Uses the discrete Wasserstein-1 distance to measure the discrepancy between the predicted and true distributions:

$$\mathcal{L}_{\text{NTL-WAS}} = \frac{1}{N}\sum_{i=1}^N \sum_{j=s}^t \hat{\mathbf{y}}_i^j |y_i - \mathcal{V}^j|$$

This simplifies to a sum of weighted absolute differences when the label is one-hot, thereby resolving the non-unique minima problem.

### Joint Training

Both NTL variants are jointly optimized with the standard CE loss:

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{NTL}$$

The default is $\lambda = 0.3$. For non-numerical tokens, the NTL loss is zero, avoiding any impact on general text tasks.

### Key Properties

- **Model-Agnostic**: Applicable to any LM architecture (Transformer, Mamba, etc.)
- **Plug-and-Play**: Only requires a token-to-numeric mapping, compatible with both digit-level and multi-digit tokenization
- **Zero Overhead**: NTL-WAS only slows down loss computation by 1%, which is negligible over the entire training step

## Key Experimental Results

### Mathematical Dataset (DeepMind, T5-Base, 220M)

| Model | Loss | Accuracy | MAE | R² |
|------|------|--------|-----|-----|
| T5 | CE | 0.64 | 0.13 | 0.97 |
| T5 | NTL-MSE | 0.72 | 0.11 | 0.97 |
| T5 | **NTL-WAS** | **0.75** | **0.10** | **0.98** |
| Regression Transformer | CE | 0.71 | 0.11 | 0.97 |
| xVal | MSE | 0.10 | 0.26 | 0.97 |

- NTL-WAS performs best on both interpolation and extrapolation tests
- On regression tasks, NTL matches the performance of dedicated regression heads, achieving a 10% improvement over standard CE

### Large-scale Scaling (T5-3B, 33B Parameters)

NTL shows consistent performance gains on larger models without degrading general text task performance.

### Ablation: Tokenization Support

| Tokenization Method | CE Accuracy | NTL-WAS Accuracy |
|----------|----------|---------------|
| Standard Subword | 0.43 | 0.51 |
| Digit-level | 0.64 | **0.75** |

The combination of NTL and digit-level tokenization yields the best results.

## Highlights & Insights

- Elegantly addresses the nominal scale issue of CE loss on numerical tokens
- NTL-WAS perfectly overcomes the non-unique minima limitation of naive regression loss via the Wasserstein distance
- True plug-and-play: released as a PyPI package `ntloss`, enabling single-line integration
- Flexible cost function design supports non-Euclidean spaces (e.g., modular arithmetic)
- No impact on text tasks + zero runtime overhead → no reason not to adopt it in pre-training

## Limitations & Future Work

- Only validated on mathematics-related tasks, without covering fields like scientific computing or finance
- Supporting multi-digit tokenizers requires a non-Euclidean cost matrix, which increases deployment complexity
- Selection of the $\lambda$ hyperparameter lacks theoretical guidance
- Not jointly validated with recent reasoning enhancement methods like chain-of-thought
- Primarily evaluated on T5 and GPT-2, lacking validation on mainstream decoder-only LLMs

## Rating

⭐⭐⭐⭐⭐ — Elegant and simple method, theoretically sound, plug-and-play, with zero overhead; it should become a standard component in LLM pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning](rulebreakers_challenging_llms_at_the_crossroads_between_formal_logic_and_human-l.md)
- [\[ACL 2025\] ReCall: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles](../../ACL2025/llm_nlp/library-like_behavior_in_language_models_is_enhanced_by_self-referencing_causal_.md)
- [\[ICCV 2025\] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](../../ICCV2025/llm_nlp/va_gpt_aligning_effective_tokens_video_anomaly.md)
- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models](../../ACL2025/llm_nlp/analyzing_and_mitigating_inconsistency_in_discrete_speech_tokens_for_neural_code.md)
- [\[ACL 2025\] HumT DumT: Measuring and Controlling Human-like Language in LLMs](../../ACL2025/llm_nlp/humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)

</div>

<!-- RELATED:END -->
