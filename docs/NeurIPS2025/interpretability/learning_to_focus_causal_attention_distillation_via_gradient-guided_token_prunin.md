---
title: >-
  [Paper Note] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning
description: >-
  [NeurIPS 2025][Knowledge Distillation] This paper proposes the Learning to Focus (LeaF) framework, which leverages gradient-guided detection to identify "confounding tokens" in training data. During knowledge distillation, these tokens are pruned to construct counterfactual samples, aligning the student model's attention to the key contextual tokens attended by the teacher model, thereby improving accuracy on mathematical reasoning and code generation tasks.
tags:
  - NeurIPS 2025
  - Knowledge Distillation
  - Causal Inference
  - Attention Alignment
  - Token Pruning
  - Confounding Factors
date: 2026-05-08
content_hash: 82325aa54f9238b0
---

# Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning

**Conference**: NeurIPS 2025

**arXiv**: [2506.07851](https://arxiv.org/abs/2506.07851)

**Code**: None

**Area**: Interpretability

**Keywords**: Knowledge Distillation, Causal Inference, Attention Alignment, Token Pruning, Confounding Factors

## TL;DR

This paper proposes the Learning to Focus (LeaF) framework, which leverages gradient-guided detection to identify "confounding tokens" in training data. During knowledge distillation, these tokens are pruned to construct counterfactual samples, aligning the student model's attention to the key contextual tokens attended by the teacher model, thereby improving accuracy on mathematical reasoning and code generation tasks.

## Background & Motivation

Large language models are frequently misled by "distracting patterns" in long-context reasoning and complex tasks, resulting in erroneous outputs. Preliminary experiments reveal a critical phenomenon:

- On mathematical training corpora, directly removing distracting patterns improves small model accuracy by more than **20%**
- On code training corpora, removing distracting patterns improves accuracy by more than **10%**
- Complex reasoning tasks (e.g., AMC_AIME) are more susceptible to distracting patterns than simpler tasks (e.g., GSM8K)

The authors attribute this phenomenon to **spurious correlations** present in training data, which hinder the model from learning genuine causal instruction-response relationships. Conventional knowledge distillation focuses solely on output imitation and fails to address this fundamental issue.

## Method

### Overall Architecture

LeaF is a two-stage framework grounded in Pearl's Structural Causal Model (SCM), which models the reasoning process as a causal graph:

- **Input tokens** $X = [x_1, x_2, \ldots, x_n]$
- **Confounding tokens** $A \subset X$: a subset of tokens that introduce spurious correlations
- **Output** $Y$: the model's reasoning result

Confounding tokens $A$ simultaneously influence $X$ and $Y$, causing the observed distribution to deviate from the interventional distribution:

$$P(Y|X_i=x) = \sum_A P(Y|X_i=x, A) P(A|X_i=x)$$

### Key Designs

#### Stage 1: Confounding Token Detection

A gradient-sensitivity method is employed to compare the teacher and student models:

1. For each token $x_i$, compute the gradient sensitivity for both models:

$$g_i^{(T)} = \left|\frac{\partial \ell(x_i|X; \theta_T)}{\partial x_i}\right|, \quad g_i^{(S)} = \left|\frac{\partial \ell(x_i|X; \theta_S)}{\partial x_i}\right|$$

2. After min-max normalization, compute the gradient difference:

$$\Delta\hat{g}_i = \hat{g}_i^{(T)} - \hat{g}_i^{(S)}$$

3. Tokens whose normalized difference falls below threshold $\tau_{\text{confounder}}$ — i.e., tokens heavily attended by the student but largely ignored by the teacher — are marked as confounding tokens.

4. An additional validation step ensures that removing the token leads to correct predictions by both models.

#### Pruning Strategies

- **Collective Pruning**: removes all confounding tokens simultaneously → may disrupt sentence integrity
- **Span Pruning**: removes one contiguous confounding span $A_i$ at a time, generating multiple counterfactual samples → yields better performance

$$\mathcal{D}_{\text{pruned}} = \{(X \setminus A_i, y)\}_{i=1}^k$$

#### Stage 2: Causal Attention Distillation

Two complementary distillation objectives are employed:

- **Standard distillation loss**: alignment on original instructions

$$\mathcal{L}_{kd} = D_{\text{KL}}(p_T(y|X) \| p_S(y|X))$$

- **Counterfactual distillation loss**: alignment on pruned instructions

$$\mathcal{L}_{cd} = D_{\text{KL}}(p_T(y|X \setminus A) \| p_S(y|X \setminus A))$$

### Loss & Training

The combined loss function is defined as:

$$\mathcal{L} = \lambda \mathcal{L}_{kd} + (1-\lambda) \mathcal{L}_{cd}$$

where $\lambda \in [0,1]$ controls the trade-off between standard and counterfactual distillation.

**Response segmentation strategy**:
- Instruction-level pruning: confounding token detection and pruning applied only to the instruction portion
- Response-level pruning: previously generated tokens are also treated as context, with confounding tokens that mislead subsequent generation detected and pruned (2-segment / 3-segment splits)

**Training hyperparameters**: Alpaca-LoRA framework with full-parameter logits distillation, cosine learning rate schedule, peak learning rate $10^{-5}$, trained for 3 epochs.

## Key Experimental Results

### Main Results

| Model | GSM8K | MATH | OlympiadBench | Avg. | HumanEval+ | LeetCode | LivecodeBench | Avg. |
|------|-------|------|---------------|------|------------|----------|---------------|------|
| **Teacher: LLaMA3.3-70B** | 95.60 | 70.40 | 36.50 | 67.50 | 78.05 | 53.90 | 45.02 | 58.99 |
| **LLaMA3.2-1B (Base)** | 44.88 | 24.20 | 5.79 | 24.96 | 29.27 | 7.22 | 9.68 | 15.39 |
| KD w/o Mask | 56.79 | 33.40 | 8.90 | 33.03 | 32.32 | 6.11 | 13.74 | 17.39 |
| LeaF (Instr Mask) | 57.70 | 35.40 | 10.09 | 34.40 | - | - | - | - |
| **LLaMA3.2-3B (Base)** | 77.56 | 42.80 | 14.83 | 45.06 | 56.71 | 20.00 | 21.58 | 32.76 |
| KD w/o Mask | 80.59 | 50.00 | 18.99 | 49.86 | 59.76 | 24.44 | 23.87 | 36.02 |
| LeaF (Resp Mask) | **82.26** | **54.40** | **20.03** | **52.23** | - | - | - | - |

**Key finding**: LeaF achieves an average improvement of **2.41%** (math) and **2.48%** (code) over standard KD on LLaMA-1B/3B.

### Ablation Study

| Pruning Strategy | MATH-500 (1B) | MATH-500 (3B) |
|----------|---------------|---------------|
| Standard KD (no pruning) | 34.00 | 50.00 |
| Collective Pruning | 34.20 | 49.20 (↓) |
| **Span Pruning** | **37.40** | **54.40** |

**Masking strategy comparison**:
- Random masking: performance degrades on GSM8K and Olympiad
- PPL masking: marginal gains on simple tasks but comparable to random masking on complex tasks
- Gradient masking (Ours): consistently outperforms both baselines across all tasks

**Threshold sensitivity analysis**:
- The 1B model performs best with higher thresholds (0.10 for instruction-level, 0.15 for response-level)
- The 3B model performs best with lower thresholds (0.05 for instruction-level, 0.10 for response-level)
- Smaller models are more susceptible to confounding tokens and require higher thresholds for effective filtering

### Key Findings

1. Response-level pruning (2-segment split) significantly outperforms instruction-level pruning, indicating that distracting patterns within responses also substantially affect subsequent generation.
2. The 3-segment split performs comparably to the 2-segment split, with diminishing returns from further segmentation.
3. The gradient-based method is indispensable in complex reasoning scenarios that require teacher guidance.
4. Attention visualizations confirm that LeaF directs the model's focus toward critical information such as constraint terms like "real numbers," "all," and "are all real."

## Highlights & Insights

1. **Causal perspective**: Framing attention bias in knowledge distillation as a confounding factor problem in causal inference provides a theoretically grounded explanation.
2. **Compelling empirical evidence**: Removing confounding tokens alone — without any additional training — yields accuracy gains exceeding 20%, strongly supporting the core hypothesis.
3. **Interpretability**: Attention heatmap visualizations clearly demonstrate how LeaF guides the model to focus on key information.
4. **Cross-domain effectiveness**: Consistent improvements are observed across both mathematical reasoning and code generation.

## Limitations & Future Work

1. **Dependence on a capable teacher model**: Confounding token detection requires teacher-student gradient comparison and cannot be applied in a self-improvement setting.
2. **Limited generalization to long-form text**: Validation is currently restricted to math and code tasks; domains such as long-document understanding remain unexplored.
3. **Computational overhead**: Computing gradients for both teacher and student models increases preprocessing cost.
4. **Future direction**: Exploring self-improvement mechanisms that enable a model to identify its own confounded attention patterns without a teacher.

## Related Work & Insights

- **Reasoning consistency**: Methods such as Self-Consistency focus on decoding-stage consistency, whereas LeaF addresses the problem at the training stage.
- **CoT knowledge distillation**: Works such as CD and SCORE emphasize data quality and diversity; LeaF focuses on the causal structure within training data.
- **Key token identification**: Works such as RHO-1 and TokenSkip identify important tokens from different perspectives; LeaF performs cross-model comparison via teacher-student gradient differences.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of causal perspective and gradient-guided confounding token detection is a novel contribution.
- **Technical Depth**: ⭐⭐⭐⭐ — Causal modeling is rigorous and experiments are comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extensive evaluation across multiple models, tasks, and ablation settings.
- **Practicality**: ⭐⭐⭐⭐ — A plug-and-play enhancement to existing distillation frameworks.
- **Overall**: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[NeurIPS 2025\] Efficient Vision-Language Reasoning via Adaptive Token Pruning](efficient_vision-language_reasoning_via_adaptive_token_pruning.md)
- [\[NeurIPS 2025\] Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework](discovering_transformer_circuits_via_a_hybrid_attribution_and_pruning_framework.md)
- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)

<!-- RELATED:END -->
