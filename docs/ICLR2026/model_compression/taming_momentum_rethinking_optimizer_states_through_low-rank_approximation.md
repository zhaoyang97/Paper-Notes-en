---
title: >-
  [Paper Note] Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation
description: >-
  [ICLR 2026][Model Compression][Low-rank optimizer] This work reveals that EMA-based momentum updates are equivalent to gradient descent on an online linear regression objective…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Low-rank optimizer"
  - "momentum compression"
  - "pretraining efficiency"
  - "LoRA"
  - "Adam"
  - "Muon"
date: 2026-05-08
content_hash: e90c0726c24ef223
---

# Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation

**Conference**: ICLR 2026 Oral  
**arXiv**: [2602.24283](https://arxiv.org/abs/2602.24283)  
**Code**: [github.com/mrflogs/LoRA-Pre](https://github.com/mrflogs/LoRA-Pre)  
**Area**: Model Compression / Efficient Optimizers
**Keywords**: Low-rank optimizer, momentum compression, pretraining efficiency, LoRA, Adam, Muon

## TL;DR

This work reveals that EMA-based momentum updates are equivalent to gradient descent on an online linear regression objective, and builds upon this insight to propose LoRA-Pre — a method that compresses optimizer momentum via low-rank factorization for memory-efficient LLM pretraining and fine-tuning. LoRA-Pre achieves state-of-the-art performance across all model scales using only 1/8 the rank required by baseline methods.

## Background & Motivation

- Optimizers such as Adam maintain first- and second-order momentum, causing memory consumption to reach **three times that of the model weights**.
- Existing low-rank optimization methods (GaLore, Flora, Fira, etc.) compress optimizer states by projecting gradients into lower-dimensional subspaces.
    - Periodic subspace updates introduce **optimization discontinuities and error accumulation**.
    - These methods cannot adapt instantaneously to shifting gradient subspaces.
- There is a need for an efficient momentum compression approach that **continuously adapts to the gradient subspace**.

## Method

### Core Insight: Momentum as a Covert Online Linear Regressor

The EMA momentum update can be rewritten as:

$$m_{t+1} = \underbrace{m_t}_{weight} - \underbrace{(1-\beta)}_{lr} \cdot \underbrace{(m_t - g)}_{gradient}$$

This is equivalent to online gradient descent on the objective:

$$\min_m L(m; g) = \frac{1}{2} \|m - g\|_F^2$$

with learning rate $1-\beta$ and loss gradient $m_t - g$.

### LoRA-Pre: Low-Rank Online Linear Regression

#### First-Order Momentum Compression

The full-rank momentum $m \in \mathbb{R}^{p \times q}$ is factorized as $m = m_B \cdot m_A$, where $m_B \in \mathbb{R}^{p \times r}$, $m_A \in \mathbb{R}^{r \times q}$, and $r \ll \min(p,q)$:

$$\min_{m_B, m_A} L(m_B, m_A; g) = \frac{1}{2} \|m_B m_A - g\|_F^2$$

Memory cost is reduced from $p \times q$ to $(p+q) \times r$.

Closed-form update rules are derived via the Newton method (Theorem 3.1):

$$m_B \leftarrow (1-\gamma_1) m_B + \gamma_1 g m_A^T (m_A m_A^T)^{-1}$$
$$m_A \leftarrow (1-\gamma_1) m_A + \gamma_1 (m_B^T m_B)^{-1} m_B^T g$$

These updates take the form of EMA and require no backpropagation.

#### Second-Order Momentum Compression

Challenge: Adam's parameter update requires $\sqrt{v}$, which demands element-wise non-negativity of $v$.

Solution: Reparameterize as $v = (v_B v_A)^{\circ 2}$ (Hadamard square) and optimize:

$$\min_{v_B, v_A} L(v_B, v_A; g) = \frac{1}{2} \|v_B v_A - |g|\|_F^2$$

This guarantees element-wise positivity while preserving the low-rank structure.

### Generality

LoRA-Pre can be applied to any momentum-based optimizer:
- **LoRA-Pre (Adam)**: compresses both $m$ and $v$
- **LoRA-Pre (Muon)**: compresses the momentum of the Muon optimizer

## Key Experimental Results

### Pretraining: Validation Perplexity of Llama Models on C4 (↓)

| Model | Full-rank Adam | GaLore | Flora | Fira | **LoRA-Pre** |
|-------|---------------|--------|-------|------|-------------|
| 60M | baseline | second-best | — | — | **best** |
| 130M | baseline | second-best | — | — | **best** |
| 350M | baseline | second-best | — | — | **best** |
| 1B | baseline | second-best | — | — | **best** |

### Rank Efficiency Comparison

| Method | Rank Required (to achieve comparable performance) |
|--------|--------------------------------------------------|
| GaLore | baseline rank $r$ |
| Flora | baseline rank $r$ |
| **LoRA-Pre** | **$r/8$** |

### Fine-tuning: MetaMathQA → GSM8K / MATH-500

| Method | Llama-3.1-8B | Llama-2-7B |
|--------|-------------|------------|
| Standard LoRA | baseline | baseline |
| **LoRA-Pre** | **+3.14** | **+6.17** |

### Ablation Study

| Component | Effect |
|-----------|--------|
| First-order compression only | effective but inferior to both orders |
| First-order + second-order compression | **best** |
| Varying rank $r$ | robust to rank variation; $r/8$ suffices |
| Adam vs. Muon variants | both optimizers benefit |

### Key Findings

1. LoRA-Pre achieves the lowest validation perplexity **across all model scales**.
2. Only **1/8 the rank** of baseline methods is needed to match or surpass their performance.
3. The method remains effective in fine-tuning settings, yielding a +6.17 improvement on Llama-2-7B.
4. Closed-form update rules require no backpropagation, ensuring computational efficiency.
5. The Hadamard square reparameterization for second-order momentum resolves the positivity constraint.

## Highlights & Insights

- **Elegant theoretical contribution**: The equivalence between EMA and online linear regression reveals a fundamentally new perspective on momentum.
- **From model compression to optimizer compression**: The core idea of LoRA is transferred from model weights to optimizer states.
- **Continuous subspace adaptation**: Unlike periodic-update methods such as GaLore, LoRA-Pre adapts to the gradient subspace at every step.
- **Exceptional rank efficiency**: 1/8 rank = lower memory footprint + better performance.
- **Unified framework**: The same framework applies to both Adam and Muon, for both pretraining and fine-tuning.

## Limitations & Future Work

- Computing $(m_A m_A^T)^{-1}$ or $(m_B^T m_B)^{-1}$ introduces additional overhead when $r$ is large.
- The Hadamard reparameterization for second-order momentum introduces approximation error.
- Validation is limited to the Llama architecture; cross-architecture generalization remains to be confirmed.
- Communication efficiency in distributed training settings is insufficiently analyzed.

## Related Work & Insights

- Low-rank pretraining: GaLore (SVD projection), Flora (random projection), Fira (SGD complementary subspace)
- Online momentum compression: MLorc, MoFaSGD, ADAPM
- Parameter-efficient fine-tuning: LoRA, LoRA+, DoRA, LoFT, LoRA-Pro

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The EMA = online regression insight is remarkably elegant.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations with clean closed-form solutions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage from 60M–1B pretraining to 7B–8B fine-tuning.
- **Practical Value**: ⭐⭐⭐⭐⭐ — Directly reduces LLM training memory; highly deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[CVPR 2026\] UniComp: Rethinking Video Compression Through Informational Uniqueness](../../CVPR2026/model_compression/unicomp_rethinking_video_compression_through_informational_uniqueness.md)
- [\[ICML 2026\] From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers](../../ICML2026/model_compression/from_per-image_low-rank_to_encoding_mismatch_rethinking_feature_distillation_in_.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[NeurIPS 2025\] QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression](../../NeurIPS2025/model_compression/qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)

</div>

<!-- RELATED:END -->
