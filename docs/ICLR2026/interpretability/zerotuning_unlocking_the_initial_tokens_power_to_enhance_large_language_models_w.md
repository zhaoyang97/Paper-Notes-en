---
title: >-
  [Paper Note] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training
description: >-
  [Interpretability] This paper proposes ZeroTuning, a training-free method that improves LLM performance across 15 datasets by applying head-specific scaling to the attention scores of the initial token (e.g., `<BOS>`)…
tags:
  - "Interpretability"
date: 2026-05-08
content_hash: 744ff3ddb8de706d
---

# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2505.11739](https://arxiv.org/abs/2505.11739)
- **Code**: [https://anonymous.4open.science/r/ZeroTuning](https://anonymous.4open.science/r/ZeroTuning)
- **Area**: Interpretability
- **Keywords**: Attention tuning, initial token, attention sink, training-free enhancement, head-specificity

## TL;DR
This paper proposes ZeroTuning, a training-free method that improves LLM performance across 15 datasets by applying head-specific scaling to the attention scores of the initial token (e.g., `<BOS>`), requiring only 4 lines of code modification.

## Background & Motivation

### Core Problem
Token-level attention tuning methods (e.g., PASTA, ACT) are effective but rely on **external heuristics** to identify task-specific "important" tokens, introducing bias and limiting applicability. The question is whether a **universal, task-agnostic control point** can be identified.

### Attention Sink Phenomenon
The initial token tends to become an attention sink, yet its potential for performance enhancement has not been exploited.

### Key Findings
- Adjusting the attention of the initial token consistently yields the largest and most stable gains.
- The direction of the gain is task-dependent: classification tasks require upscaling ($\gamma > 1$), while QA tasks require downscaling ($\gamma < 1$).
- Different attention heads respond heterogeneously to initial token scaling.

## Method

### Formalization of Attention Scaling

A scaling factor $\gamma > 0$ is introduced to adjust the attention weight of the initial token:

$$a_0' = \frac{\gamma a_0}{D}, \quad a_i' = \frac{a_i}{D}, \quad D = (\gamma-1)a_0 + 1$$

**Key properties**:
- Preserves the relative proportions among non-initial tokens.
- $\gamma > 1$: amplifies the initial token and flattens the remaining distribution.
- $\gamma < 1$: suppresses the initial token and sharpens the remaining distribution.

### Analysis of Modulation Effects

Change in attention disparity:
$$E_{\text{diff},i,j} = |a_i - a_j| \cdot \frac{|\gamma-1| a_0}{(\gamma-1) a_0 + 1}$$

Derivative with respect to $a_0$:
$$\frac{\partial E_{\text{diff},i,j}}{\partial a_0} = |a_i - a_j| |\gamma-1| \cdot \frac{1}{((\gamma-1)a_0+1)^2} \geq 0$$

**Core Insight**: The larger the attention weight of the initial token (i.e., the stronger the attention sink), the greater its leverage as a modulation mechanism.

### Layer-wise Analysis

Adjustments in shallow (layers 1–10) and middle layers (layers 11–21) are generally more effective than in deep layers (layers 22–31), because:
- Early and middle layers primarily support representation learning and knowledge integration.
- Deep layers focus on task-specific reasoning.

### Head-Specificity

Different heads respond heterogeneously to initial token scaling:
- **Up-effective heads**: amplifying attention improves performance.
- **Down-effective heads**: suppressing attention improves performance.
- Functional differences arise from head-level specialization during pretraining.

### ZeroTuning Procedure

Three-step pipeline:
1. **Head behavior analysis**: assess each head's sensitivity to initial token scaling.
2. **Selective scaling**: apply scaling factor $\gamma$ only to the dominant head type.
3. **Re-normalization**: softmax re-normalization maintains a valid attention distribution.

**Two calibration modes**:
- **Supervised mode**: search for $\gamma$ by maximizing accuracy on a labeled validation set.
- **Unsupervised mode**: minimize output entropy — $\gamma$ that minimizes entropy is strongly correlated with that which maximizes accuracy.

**Compatibility**: Supports both SDPA and FlashAttention (via scaling of query/key states).

## Experiments

### Classification Tasks

| Model | Vanilla | ACT | Auto-PASTA | **ZeroTuning** |
|-------|---------|-----|------------|----------------|
| Llama-3.1-8B Avg | 59.59 | 60.11 | 63.73 | **71.44** |
| Qwen-2-7B Avg | 55.10 | - | 65.57 | **68.19** |
| Deepseek-R1-14B Avg | 67.67 | - | 69.04 | **71.87** |

Maximum single-dataset improvement: SST-2 73.20 → 91.60 (+18.4%); SUBJ 44.60 → 66.60 (+22.0%).

### Multiple-Choice QA Tasks

| Model | Vanilla | Auto-PASTA | **ZeroTuning** |
|-------|---------|------------|----------------|
| Llama-3.1-8B Avg | 58.84 | 60.18 | **61.48** |
| Qwen-2-7B Avg | 63.10 | 64.01 | **64.84** |
| Deepseek-R1-14B Avg | 60.05 | 60.31 | **62.20** |

LogiQA with Deepseek-R1-14B: 27.80 → 35.60 (+7.80%).

### MT-Bench Dialogue

| Model | Vanilla | ZeroTuning |
|-------|---------|-----------|
| Llama-3.1-8B | 7.804 | **7.966** |
| Llama-2-13B | 6.650 | **6.916** |

### Key Findings

1. **Tuning a single token consistently outperforms multi-token tuning methods.**
2. **Strong inverse correlation between accuracy and output entropy**, validating the feasibility of the unsupervised mode.
3. **Head-specific tuning substantially outperforms uniform tuning.**
4. **Robust performance under quantized inference, long-context, and few-shot settings.**
5. **Only 4 lines of code modification required.**

## Highlights & Insights

1. **Minimalist design**: one token, one scaling factor, 4 lines of code.
2. **Rigorous theoretical analysis**: complete derivation from attention reshaping to bias correction.
3. **Unsupervised mode**: based on entropy minimization, requiring no labeled data.
4. **Kernel-agnostic**: compatible with both SDPA and FlashAttention.
5. **Consistently effective across models and tasks.**

## Limitations & Future Work

1. The optimal scaling direction is task-dependent (classification vs. QA), requiring preliminary experiments or heuristic judgment.
2. Head behavior analysis still incurs a non-trivial computational cost.
3. Gains are relatively limited on already strong large models (e.g., Deepseek-R1-14B).
4. The combined Up+Down strategy does not outperform a single strategy; joint optimization remains to be explored.
5. Improvements on generative tasks (open-ended dialogue) are limited.

## Related Work & Insights
- **Attention tuning**: PASTA, Auto-PASTA, ACT — require identification of important tokens.
- **Attention sink research**: StreamingLLM, Barbero et al. — explain the phenomenon without exploiting it.
- **Inference-time optimization**: self-consistency, CoT — prompt engineering approaches.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — A minimalist yet effective idea that transforms attention sinks from passive observations into active levers.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 15 datasets, 4 models, multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Logically progressive, presenting a complete narrative from theory to method to experiments.
- **Value**: ⭐⭐⭐⭐⭐ — Deployable with 4 lines of code; no training, no additional memory overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)
- [\[ICLR 2026\] Universal Properties of Activation Sparsity in Modern Large Language Models](universal_properties_of_activation_sparsity_in_modern_large_language_models.md)
- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](../../NeurIPS2025/interpretability/probabilistic_token_alignment_for_large_language_model_fusion.md)

</div>

<!-- RELATED:END -->
