---
title: >-
  [Paper Note] LASER: Attention with Exponential Transformation
description: >-
  [ICML 2025][LLM (Other)][attention mechanism] By analyzing the gradient backpropagation bottleneck of softmax in the attention mechanism, this paper proposes LASER attention—performing attention computation in the exponentially transformed Value space (i.e., applying attention to $\exp(V)$ and then taking the logarithm), thereby obtaining larger Jacobian signals and improving parameter learning efficiency.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "attention mechanism"
  - "vanishing gradient"
  - "Log-Sum-Exp"
  - "Softmax"
  - "Transformer"
date: 2026-05-08
content_hash: 1245aa7b18b2d212
---

# LASER: Attention with Exponential Transformation

**Conference**: ICML 2025  
**arXiv**: [2411.03493](https://arxiv.org/abs/2411.03493)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: attention mechanism, vanishing gradient, Log-Sum-Exp, Softmax, Transformer

## TL;DR

By analyzing the gradient backpropagation bottleneck of softmax in the attention mechanism, this paper proposes LASER attention—performing attention computation in the exponentially transformed Value space (i.e., applying attention to $\exp(V)$ and then taking the logarithm), thereby obtaining larger Jacobian signals and improving parameter learning efficiency.

## Background & Motivation

The core of the Transformer is the softmax dot-product attention. However, the authors observe that during backpropagation, gradients passing through softmax are scaled by its Jacobian matrix. The Jacobian of softmax is proportional to the attention probabilities/weights. In large language models, approximately 80% of the attention probabilities are smaller than $10^{-3}$, and about 20% are smaller than $10^{-7}$. This implies that the gradient signal is severely attenuated when passing through softmax, leading to low learning efficiency for parameters prior to the attention layer (such as $W_Q$, $W_K$, and $W_V$).

Although residual connections can "bypass" gradient attenuation between layers, gradients must still traverse the softmax within the attention block. This issue becomes more severe as the sequence length increases (since attention is more dispersed and probabilities are smaller). The authors aim to find a method that allows the attention mechanism itself to propagate larger gradients.

## Method

### Overall Architecture

The core idea of LASER (**L**og**A**rithm of **S**ummed **E**xponentials of **R**epresentations) is to replace the direct weighted sum of the Value matrix $V$ in standard attention with a weighted sum in the exponential space $\exp(V)$, and then apply a logarithm to recover the scale:

$$\text{LASER}(X) = \log\!\Big(\text{softmax}(QK^\top) \cdot \exp(V)\Big)$$

Here, $\log(\cdot)$ and $\exp(\cdot)$ are both element-wise operations. While standard attention outputs $\text{attn}(X) = \text{softmax}(QK^\top) V$, LASER only introduces exp and log transformations at the input and output sides of $V$, **without modifying the intermediate attention computation block** (making it directly compatible with efficient implementations like FlashAttention).

### Key Designs

1. **Gradient Analysis and Derivation of Motivation**

   The authors derive the formulation starting from the simplest case of $N=2, d=1$. The Jacobian element of standard attention is:
    $\frac{\partial o_1}{\partial \tilde{a}_{11}} = (v_1 - v_2) \cdot \sigma(\tilde{a}_{11} - \tilde{a}_{12})(1 - \sigma(\tilde{a}_{11} - \tilde{a}_{12}))$
   where $\sigma$ is the sigmoid function. When the absolute value of $\tilde{a}_{11} - \tilde{a}_{12}$ is large (i.e., attention is highly focused on a specific token), $\sigma(1-\sigma)$ approaches 0, causing gradients to vanish.

   Generalizing to an arbitrary sequence length $N$, the Jacobian of softmax is $\text{diag}(a) - aa^\top$, with elements given by $a_j(\mathbb{1}\{i=j\} - a_i)$. When the attention probabilities $a_i, a_j$ are small, the overall Jacobian becomes very small.

   For LASER, under identical conditions, the Jacobian in the limiting case where $\exp(v_1) \gg \exp(v_2)$ simplifies to $1 - \alpha_1 = 1 - \sigma(\tilde{a}_{11} - \tilde{a}_{12})$, which contains only a single sigmoid factor instead of a product of two, **significantly alleviating saturation**.

2. **Log-Weighted-Sum-Exp Numerical Stabilization Trick**

   Direct computation of $\exp(V)$ may cause numerical overflow (especially in large models). Inspired by the Log-Sum-Exp technique, the authors propose the Log-Weighted-Sum-Exp trick:

    - Compute the maximum value along each column of $V$: $m_j = \max_i V_{ij}$
    - Construct a shifted matrix $\hat{V}_{ij} = V_{ij} - m_j$ (ensuring $\exp(\hat{V})$ does not overflow)
    - Compute standard attention using $\hat{V}$ instead of $V$: $O_{ij} = \log\!\big(\text{softmax}(QK^\top) \exp(\hat{V})\big)_{ij} + m_j$

   Key Advantage: It only requires modifying the **input** ($V \to \exp(\hat{V})$) and the **output** ($\log(\cdot) + m$) of the attention block, without altering the intermediate attention function itself.

3. **Theoretical Connection with the Max Function**

   The output of LASER possesses a log-sum-exp structure, which can be viewed as a differentiable approximation of the max function. Based on the classical result of Boyd & Vandenberghe:
    $\max(x_1, \ldots, x_n) \leq \text{LSE}(x_1, \ldots, x_n) \leq \max(x_1, \ldots, x_n) + \log n$
   Thus, in a sense, LASER implements a **differentiable "max attention"**, preserving selectivity for key information while facilitating gradient propagation.

### Loss & Training

The training follows standard configurations without custom loss function designs:

- Autoregressive LLM: Employs AdamW with a cosine learning rate schedule on the C4 dataset, with a size of 1024 × 1024 tokens per batch, trained for 160K steps (approximately 168B tokens).
- ViT: Employs NAdamW, searching for the optimal hyperparameter setting across 50 configurations.
- All experiments directly replace standard attention with LASER, where hyperparameters are searched on a small model (16 layers) and directly migrated to larger models.

## Key Experimental Results

### Main Results

| Model/Task | Metric | LASER | Standard Attention | Gain |
|-----------|------|-------|---------------|------|
| LLM 301M (32 layers, C4) | Test Loss | 2.595 | 2.641 | 1.74% relative |
| LLM 2.2B (C4) | 16-task average Acc | 63.39% | 62.34% | +1.05% |
| LLM 7.7B (C4, 44B tokens) | 11-task average Acc | 53.97% | 52.53% | +1.44% |
| ViT-S/16 (ImageNet) | Valid Error | 24.17% | 25.32% | -1.15% (absolute) |
| Conformer (Librispeech) | Valid WER | 8.08% | 8.32% | -0.24% |
| BERT 2.2B (MLM) | MLM Error Rate | 0.2125 | 0.2145 | 0.93% relative |
| SuperGLUE Finetuning (2.2B) | Average Acc | 44.01% | 42.35% | +1.65% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| 16 layers vs 32 layers (301M) | 2.673 vs 2.595 | LASER consistently outperforms standard attention across different depths |
| AdamW vs LAMB Optimizer | Test Loss 2.741 vs 2.758 | Improvement remains even with LAMB normalization, indicating gains do not solely stem from gradient magnitude |
| Without Log-Weighted-Sum-Exp | 2.2B training collapse | The numerical stabilization technique is indispensable for large models |
| LASER + QK-Norm (ViT) | Valid Error 23.72% | Orthogonally complementary to QK-Normalization, yielding the best joint performance |
| LASER + per-dim temp | LLM average Acc 63.52% | Per-dimension temperature scaling yields a further marginal improvement |
| Diff+LASER (2.2B) | Average Acc 51.52% vs 50.59% | LASER can be combined on top of the Differential Transformer |

### Key Findings

- **Gradient Magnitude**: LASER maintains a higher gradient norm (`grad_norm`) throughout the training process, though the LAMB experiment verifies that the improvement does not stem solely from larger gradients.
- **Scaling Law**: Through power-law fitting, to achieve the same loss value, LASER requires approximately 15.65% fewer parameters.
- **Training Efficiency**: The time required for the 2.2B model to reach the optimal loss of standard attention is reduced by 9.4% (27.22h $\to$ 24.88h).
- **Cross-Modal Generalization**: Demonstrates effectiveness across text, vision, and speech modalities.
- **Orthogonality with Other Techniques**: Compatible with QK-Normalization, temperature scaling, and DiffTransformer, among others.

## Highlights & Insights

1. **Precise Problem Identification**: Quantitatively analyzes the gradient bottleneck departing from the Jacobian of the attention mechanism, rather than relying solely on empirical observations.
2. **Exceedingly Simple Modification**: Only requires adding `exp` at the attention input and `log` at the output, without modifying the core attention implementation (remaining compatible with FlashAttention).
3. **Ingenious Log-Weighted-Sum-Exp Trick**: Resolves numerical overflow through column-wise max shifting, keeping the implementation simple.
4. **LASER as Differentiable Max Attention**: Establishes a theoretical connection with the max operation, providing intuition for understanding its behavior.
5. **Comprehensive Experimental Evaluation**: Spans sizes from 234M to 7.7B and covers decoder-only, encoder-only, ViT, and Conformer, demonstrating strong generalization capability.

## Limitations & Future Work

1. **Theoretical Analysis Based on Extreme Cases**: The Jacobian improvement is most prominent under the condition $\exp(v_1) \gg \exp(v_2)$; a tighter bound for the improvement in general cases is lacking.
2. **Moderate Improvements**: In most experiments, the performance gain lies within the 1-2% range, which, while consistent, is not highly substantial.
3. **Unanalyzed Computational Overhead**: The additional computational cost of the exp/log operations and their training impact at scale are not thoroughly quantified.
4. **Validation Limited to Pre-training**: Has not explored the effectiveness in downstream phases such as RLHF and instruction tuning.
5. **Compatibility with Linear Attention**: Since LASER relies on standard softmax attention, it cannot be directly applied to linear approximation attention mechanisms.

## Related Work & Insights

- **Linear Attention / Performer**: Reduces complexity via kernel approximations, whereas LASER focuses on gradient flow rather than efficiency.
- **QK-Normalization**: Directs attention to preventing training instability by controlling Q/K norms via LayerNorm, which is orthogonally complementary to LASER.
- **Differential Transformer**: Achieves denoising through the difference of two softmax maps, upon which LASER can be integrated.
- **Insights**: This line of thinking suggests that similar exponential space transformations can be explored in other components with probability normalization (such as the softmax output layer).

## Rating

- Novelty: ⭐⭐⭐⭐ — Analyzing the softmax bottleneck from the perspective of the gradient Jacobian is novel, though the log-sum-exp structure itself is a classical tool.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Highly comprehensive, spanning cross-modality, multiple scales, various baselines, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Highly readable, with clear derivations starting from simple cases and gradually generalizing.
- Value: ⭐⭐⭐⭐ — Practical as a plug-and-play attention modification, though the performance gain is modest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reparameterized LLM Training via Orthogonal Equivalence Transformation](../../NeurIPS2025/llm_nlp/reparameterized_llm_training_via_orthogonal_equivalence_transformation.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](../../NeurIPS2025/llm_nlp/spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[ACL 2025\] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge](../../ACL2025/llm_nlp/enhancing_transformation_from_natural_language_to_signal_temporal_logic_using_ll.md)
- [\[ICML 2025\] TabFlex: Scaling Tabular Learning to Millions with Linear Attention](tabflex_scaling_tabular_learning_to_millions_with_linear_attention.md)
- [\[ICML 2025\] Star Attention: Efficient LLM Inference over Long Sequences](star_attention_efficient_llm_inference_over_long_sequences.md)

</div>

<!-- RELATED:END -->
