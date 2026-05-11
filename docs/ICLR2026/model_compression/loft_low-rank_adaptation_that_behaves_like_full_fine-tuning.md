---
title: >-
  [Paper Note] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning
description: >-
  [ICLR 2026][Model Compression][LoRA] This paper proposes LoFT, a low-rank adaptation method composed of six building blocks that aligns the internal optimizer dynamics (momentum and second-order moments) with those of fu…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "LoRA"
  - "Low-Rank Adaptation"
  - "Full Fine-Tuning"
  - "Optimizer State Alignment"
  - "AdamW"
date: 2026-05-08
content_hash: 934f2bf2aed0864d
---

# LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning

**Conference**: ICLR 2026
**arXiv**: [2505.21289](https://arxiv.org/abs/2505.21289)
**Code**: None
**Area**: Parameter-Efficient Fine-Tuning / Model Compression
**Keywords**: LoRA, Low-Rank Adaptation, Full Fine-Tuning, Optimizer State Alignment, AdamW

## TL;DR

This paper proposes LoFT, a low-rank adaptation method composed of six building blocks that aligns the internal optimizer dynamics (momentum and second-order moments) with those of full fine-tuning. In the full-rank limit, LoFT exactly recovers AdamW, and it substantially closes the performance gap between LoRA and full fine-tuning across multiple benchmarks.

## Background & Motivation

Adapting large-scale pretrained models to downstream tasks has become the standard paradigm in NLP and beyond. However, as model sizes scale to billions of parameters, full fine-tuning becomes computationally expensive and impractical, particularly in multi-task or multi-user settings. Parameter-efficient fine-tuning (PEFT) techniques address this challenge by training only a small subset of parameters, with LoRA (Low-Rank Adaptation) being the most popular approach.

**Successes and Limitations of LoRA:**

LoRA freezes the original weights and injects trainable low-rank matrices $W = W_0 + UV^\top$ into selected layers, where $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$, $r \ll \min\{m, n\}$. This significantly reduces the number of trainable parameters without adding inference latency. Nevertheless, LoRA still lags behind full fine-tuning in certain settings:

**Persistent performance gap**: Empirical studies consistently show a non-trivial gap between LoRA and full fine-tuning.

**Slower convergence**: The optimization dynamics of LoRA differ fundamentally from those of full fine-tuning.

**Hyperparameter sensitivity**: The choice of the scaling factor $\alpha$ significantly affects performance, incurring non-trivial tuning costs.

**Key Insight of This Paper:**

Prior work (e.g., DoRA, LoRA-Pro) primarily focused on more accurate gradient approximation within the low-rank subspace. This paper reveals a neglected yet critical factor: **optimizer state misalignment** — specifically, the first-order moment (momentum) and second-order moment (variance) in AdamW. When these internal statistics are not properly aligned with the low-rank constraint, adaptation quality degrades.

## Method

### Overall Architecture

LoFT can be understood as "the closest approximation to full fine-tuning under the constraint that weight updates are restricted to a low-rank subspace." It consists of six core building blocks that systematically address the discrepancies between LoRA and full fine-tuning.

### Key Designs

1. **Alternating Updates (Building Block 1)**:

    - Function: Updates $U$ and $V$ alternately rather than simultaneously.
    - Mechanism: When standard LoRA updates $U$ and $V$ jointly, a cross term proportional to $\eta^2$ appears in the weight update, which is inconsistent with full fine-tuning. Alternating updates eliminate this term: when only $U$ is updated, $W^+ = W - \eta \nabla_W f(W) VV^\top$.
    - **Design Motivation**: Eliminates the second-order cross term in LoRA updates, bringing the update form closer to full fine-tuning.

2. **Gradient Scaling (Building Block 2)**:

    - Function: Uses the scaled gradient $\tilde{\nabla}_U f(W) = \nabla_U f(W)(V^\top V)^{-1}$.
    - Mechanism: Using LoRA gradients directly introduces a scale-dependency issue — $UV^\top = (cU)(V/c)^\top$ holds for any $c \neq 0$, yet gradient updates scale with $c$. The projection matrix $\mathcal{P}_V = V(V^\top V)^{-1}V^\top$ is used to normalize the update: $W^+ = W - \eta \nabla_W f(W) \mathcal{P}_V$. This ensures the update direction is always the orthogonal projection of the full gradient onto the current low-rank subspace — the optimal low-rank approximation.
    - **Design Motivation**: Eliminates scale ambiguity, makes updates independent of the specific parameterization of $U$ and $V$, and removes the need for the $\alpha$ hyperparameter.

3. **First-Order Moment (Momentum) Calibration (Building Block 3)**:

    - Function: Introduces a calibration matrix $C_V^k = (V_{k-1}^\top V_k)(V_k^\top V_k)^{-1}$ into momentum accumulation.
    - Mechanism: The standard LoRA momentum $m_U^k V^\top$ involves $V_i$ from different time steps and the current $V_k$, leading to inconsistent implicit projections. The calibration matrix "rotates" historical momentum into the current subspace:
    $m_U^k = \beta_1 m_U^{k-1} C_V^k + (1-\beta_1)\tilde{\nabla}_U f(W_i)$
    The calibrated momentum is equivalent to accumulating the full gradient projected onto the intersection of all historical and current subspaces.
    - **Design Motivation**: As the low-rank subspace evolves during training, historical momentum must be "transformed" into the current subspace to remain meaningful.

4. **Second-Order Moment Alignment (Building Blocks 4 & 5)**:

    - Function: Efficiently maintains cross terms of the second-order moment via Khatri-Rao and Kronecker product identities.
    - Mechanism: Adam's second-order moment $v_k$ involves element-wise squared gradients. Under low-rank parameterization, maintaining an $r^2$-sized cross-term matrix $p_U^k$ is required:
    $p_U^k = \beta_2 p_U^{k-1}(C_V^k \otimes C_V^k) + (1-\beta_2)(\tilde{\nabla}_U f \bullet \tilde{\nabla}_U f)$
    The full second-order moment estimate is then reconstructed via $\tilde{v}_U^k = p_U^k(V_k * V_k)$. Building Block 5 uses the calibrated first- and second-order moments to construct the final update:
    $U_{k+1} = U_k - \eta_k \frac{m_U^k V_k / (1-\beta_1^k)}{p_U^k(V_k * V_k)/(1-\beta_2^k) + \varepsilon} V_k(V_k^\top V_k)^{-1}$
    - **Design Motivation**: Precisely aligns Adam's adaptive learning rate mechanism so that each step of low-rank optimization is as close as possible to full fine-tuning.

5. **Gradient Clipping (Building Block 6)**:

    - Function: Uses the projected effective gradient $\nabla_W f(W) \mathcal{P}_V$ as the layer representative during gradient clipping.
    - **Design Motivation**: Ensures clipping behavior is consistent with full fine-tuning.

### Loss & Training

- LoFT is an optimizer-level improvement and does not modify the loss function; it is compatible with any standard training objective.
- Weight decay requires no special modification: alternating updates ensure that decay $UV^\top \to (1-\lambda\eta_k)UV^\top$ is consistent with full fine-tuning.
- Additional memory overhead: $\mathcal{O}((m+n)r)$ for first-order moment calibration; $\mathcal{O}((m+n)r^2)$ for second-order moment cross terms.
- Computational overhead: primarily from $r \times r$ matrix inversions and Khatri-Rao products, at $\mathcal{O}(r^3)$ complexity.

**Core Theoretical Guarantee**: When $r = \min\{m, n\}$ and $U_k, V_k$ are full-rank, LoFT-AdamW **exactly recovers** the full AdamW update. This is the first low-rank adaptation method with this property.

## Key Experimental Results

### Main Results

**Commonsense Reasoning (LLaMA series)**:

| Model | Method | BoolQ | PIQA | SIQA | HS | WG | ARC-C | ARC-E | OBQA | Avg. |
|-------|--------|-------|------|------|----|----|-------|-------|------|------|
| LLaMA-7B | LoRA | - | - | - | - | - | - | - | - | Baseline |
| LLaMA-7B | DoRA | - | - | - | - | - | 64.68 | - | - | Baseline+ |
| LLaMA-7B | LoFT | - | 80.96 | 78.27 | 80.50 | 76.40 | - | 80.26 | 78.40 | 74.95 |
| LLaMA2-7B | DoRA | - | 82.92 | 79.22 | 88.90 | - | - | - | - | 79.71 |
| LLaMA2-7B | LoFT | 71.80 | - | - | - | 82.72 | 69.11 | 84.43 | 81.00 | Best |

**Image Classification (ViT-Base)**:
- Evaluated on medical imaging datasets and highly imbalanced datasets such as DomainNet.
- LoFT matches or exceeds full fine-tuning performance on most datasets.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| LoFT (full) | Best convergence | All components working together |
| w/o alternating updates | Significantly worse | Second-order cross terms harm convergence |
| w/o optimizer state calibration | Noticeably worse | Misaligned momentum and variance lead to suboptimal results |
| Gradient scaling only | Limited improvement | Gradient alignment is only a partial solution |
| Standard LoRA | Slowest convergence | All issues compounded |

A synthetic experiment ($f(W) = \|W - A\|_F^2$, $m=1024, n=512, r=8$) clearly demonstrates the importance of each component: the convergence curve of LoFT nearly overlaps with that of full fine-tuning.

### Key Findings

1. LoFT addresses not only gradient alignment but also the long-neglected problem of optimizer state alignment.
2. LoFT maintains robust performance even under extreme low-rank constraints ($r=1$).
3. LoFT naturally eliminates the need for the LoRA scaling factor $\alpha$.
4. Significant training quality improvements are achieved without any additional inference cost.

## Highlights & Insights

1. **Optimizer state misalignment is a neglected critical issue**: All prior LoRA improvement work focused on gradient approximation; this paper is the first to systematically identify and address the misalignment of momentum and second-order moments — a genuine blind spot in the field.
2. **The six-block decomposition is clear and elegant**: Each building block has a well-defined corresponding problem and theoretical motivation, and together they form a complete solution.
3. **Mathematically provable equivalence**: Exact recovery of AdamW in the full-rank limit constitutes a very strong theoretical guarantee.
4. **Elimination of the $\alpha$ hyperparameter**: Gradient scaling naturally resolves norm ambiguity, reducing tuning burden.
5. **Clever use of Khatri-Rao / Kronecker products**: Matrix decomposition theory is leveraged to maintain second-order moments efficiently, keeping computational complexity tractable.

## Limitations & Future Work

1. **Additional memory overhead**: The second-order moment cross terms require $\mathcal{O}((m+n)r^2)$ extra memory, which is unfriendly for large values of $r$. The authors plan to explore LLM-specific optimizers (e.g., Muon) to mitigate this.
2. **Increased computational cost**: While inference is unaffected, training requires additional matrix operations (calibration, projection, etc.).
3. **Incomplete experimental tables**: Conversion errors in ar5iv caused some numerical results to be missing, affecting the completeness of the presented results.
4. **Larger-scale models not explored**: Experiments are primarily conducted at the 7B–8B scale; effectiveness on models of 70B parameters and above remains to be verified.
5. **Integration with other PEFT methods**: Whether the ideas underlying LoFT can transfer to other PEFT approaches such as Adapter and Prefix Tuning is an open question.

## Related Work & Insights

- **LoRA and variants**: LoRA → DoRA (decoupled direction and magnitude) → LoRA-Pro (improved gradient approximation) → LoFT (comprehensive alignment of optimizer dynamics).
- **Riemannian optimization perspective**: Zhang et al. derive similar gradient scaling results from a Riemannian geometry viewpoint.
- **Inspiration**: The idea of optimizer state alignment may be applicable to other constrained optimization settings — any optimization method operating within a subspace may benefit from analogous calibration strategies.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First work to systematically reveal and address optimizer state misalignment; solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers language and vision tasks across multiple model scales, though some data are missing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Building-block organization is clear; mathematical derivations are rigorous.
- **Value**: ⭐⭐⭐⭐⭐ — Provides important guidance for LoRA improvements and has the potential to become a new standard practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions](../../ACL2026/model_compression/polynomial_expansion_rank_adaptation_enhancing_low-rank_fine-tuning_with_high-or.md)
- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](../../NeurIPS2025/model_compression/data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[ICLR 2026\] Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation](taming_momentum_rethinking_optimizer_states_through_low-rank_approximation.md)

</div>

<!-- RELATED:END -->
