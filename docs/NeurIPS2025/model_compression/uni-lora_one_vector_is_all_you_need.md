---
title: >-
  [Paper Note] Uni-LoRA: One Vector is All You Need
description: >-
  [NeurIPS 2025][Model Compression][Parameter-Efficient Fine-Tuning] This paper proposes Uni-LoRA, a unified framework demonstrating that the parameter reduction strategies of various LoRA variants (Tied-LoRA, VeRA, VB-LoRA, etc.) are fundamentally distinguished by the choice of projection matrix mapping the full parameter space $\mathbb{R}^D$ to a low-dimensional subspace $\mathbb{R}^d$. An isometric random grouping projection matrix is designed such that training a single vector suffices to reconstruct all LoRA parameters of an LLM, achieving extreme parameter efficiency.
tags:
  - NeurIPS 2025
  - Model Compression
  - Parameter-Efficient Fine-Tuning
  - LoRA
  - Projection Matrix
  - Isometry
  - Parameter Sharing
date: 2026-05-08
content_hash: b1243c551d6b6fa1
---

# Uni-LoRA: One Vector is All You Need

**Conference**: NeurIPS 2025
**arXiv**: [2506.00799](https://arxiv.org/abs/2506.00799)
**Code**: [GitHub](https://github.com/KaiyangLi1992/Uni-LoRA)
**Area**: Model Compression
**Keywords**: Parameter-Efficient Fine-Tuning, LoRA, Projection Matrix, Isometry, Parameter Sharing

## TL;DR

This paper proposes Uni-LoRA, a unified framework demonstrating that the parameter reduction strategies of various LoRA variants (Tied-LoRA, VeRA, VB-LoRA, etc.) are fundamentally distinguished by the choice of projection matrix mapping the full parameter space $\mathbb{R}^D$ to a low-dimensional subspace $\mathbb{R}^d$. An isometric random grouping projection matrix is designed such that training a single vector suffices to reconstruct all LoRA parameters of an LLM, achieving extreme parameter efficiency.

## Background & Motivation

LoRA enables parameter-efficient fine-tuning via low-rank decomposition $\Delta W = BA$, and subsequent works (Tied-LoRA, VeRA, LoRA-XS, VB-LoRA) further reduce the number of trainable parameters. However, these methods each introduce distinct architectural modifications without a unified perspective. Three common structural limitations are identified:

**Local Projection**: Most methods (Tied-LoRA, VeRA, LoRA-XS) project each LoRA module's parameters independently per layer, precluding cross-layer parameter sharing.

**Non-Uniform Projection**: The $B$ and $A$ matrices in Tied-LoRA and VeRA are projected onto subspaces of different dimensionalities ($m$ vs. $r$), leading to uneven information allocation.

**Non-Isometric Projection**: The implicit projection matrices in Tied-LoRA, VeRA, and VB-LoRA do not preserve distances, distorting the geometry of the optimization landscape.

**Core Insight**: Drawing on research into intrinsic dimensionality — which shows that fine-tuning neural networks effectively operates within a subspace far smaller than the nominal parameter space — if all LoRA parameters across all layers and modules are flattened into a single $D$-dimensional vector $\theta_D$, the essential distinction among different LoRA methods reduces to the choice of projection matrix $P \in \mathbb{R}^{D \times d}$ such that $\theta_D = P \theta_d$.

**Starting Point**: Design an optimal projection matrix satisfying globality, uniformity, and isometry.

## Method

### Overall Architecture

The $B^\ell$ and $A^\ell$ matrices from all $L$ LoRA modules are flattened and concatenated into a global parameter vector:

$$\theta_D = \text{Concat}(\text{vec}(B^1), \text{vec}(A^1), \cdots, \text{vec}(B^L), \text{vec}(A^L))$$

This vector is then mapped to a low-dimensional subspace via $\theta_D = P \theta_d$, with only $\theta_d \in \mathbb{R}^d$ ($d \ll D$) being trained.

### Key Designs

1. **Unified Framework Representation**: Existing methods are shown to be expressible within the $\theta_D = P \theta_d$ framework:

    - **LoRA**: $P = I_D$ (identity matrix), $d = D$
    - **Tied-LoRA/VeRA**: $P$ is a block-diagonal sparse matrix, repeated $L$ times — local, non-uniform, and non-isometric
    - **VB-LoRA**: $P$ is a learned vector bank — global but non-isometric

   Analyzing each method's projection matrix in terms of globality, uniformity, and isometry reveals their structural deficiencies.

2. **Isometric Random Grouping Projection Matrix**: The construction of $P \in \mathbb{R}^{D \times d}$ is remarkably simple — each row is a one-hot vector whose "1" position is sampled uniformly at random from $d$ slots, followed by column-wise normalization: each nonzero entry in column $j$ is set to $1/\sqrt{n_j}$, where $n_j$ is the number of nonzero entries in that column.

   **Intuition**: The $D$ LoRA parameters are randomly partitioned into $d$ groups; parameters within the same group share a single value throughout training.

   **Theorem 1** (Isometry): $P^\top P = I_d$, hence $\|P(x-y)\| = \|x-y\|$, preserving distances. The proof hinges on the fact that each row contains exactly one nonzero entry, ensuring off-diagonal elements of $P^\top P$ vanish and diagonal elements equal one after normalization.

3. **Analysis of Three Projection Properties**:

    - **Globality**: Parameters are shared across layers and matrix types ($B$ and $A$), breaking physical layer boundaries.
    - **Uniformity (Load Balancing)**: Each subspace dimension maps to approximately the same number of original parameters, ensuring balanced information allocation.
    - **Isometry**: The geometric structure of the original parameter space is preserved, leaving the optimization landscape undistorted.

### Loss & Training

- The projection matrix $P$ is generated from a random seed and frozen; only $\theta_d$ is trained.
- Storage requires only $d + 1$ values ($\theta_d$ plus the random seed), realizing the "one vector is all you need" principle.
- Both time and space complexity of the projection are $\mathcal{O}(D)$, substantially more efficient than Fastfood's $\mathcal{O}(D \log d)$ and Gaussian projection's $\mathcal{O}(Dd)$.
- In practice, $P$ is never explicitly constructed; only the index array and normalization factors are stored.

## Key Experimental Results

### Main Results: GLUE (RoBERTa-large)

| Method | Trainable Params | SST-2 | MRPC | CoLA | QNLI | RTE | STS-B | Avg. |
|---|---|---|---|---|---|---|---|---|
| LoRA | 786K | 96.2 | 90.2 | 68.2 | 94.8 | 85.2 | 92.3 | 87.8 |
| VeRA | 61K | 96.1 | 90.9 | 68.0 | 94.4 | 85.9 | 91.7 | 87.8 |
| VB-LoRA | 162K† | 96.1 | 91.4 | 68.3 | 94.7 | 86.6 | 91.8 | 88.2 |
| LoRA-XS | 25K | 95.9 | 90.7 | 67.0 | 93.9 | 88.1 | 92.0 | 87.9 |
| **Uni-LoRA** | **23K** | **96.3** | 91.3 | **68.5** | 94.6 | 86.6 | **92.1** | **88.3** |

### Mathematical Reasoning (Gemma-7B on GSM8K/MATH)

| Method | Trainable Params | GSM8K | MATH |
|---|---|---|---|
| LoRA | 200M | 74.90 | 31.28 |
| VeRA | 1.90M | 74.98 | 28.84 |
| VB-LoRA | 113M† | 74.86 | 28.90 |
| FourierFT | 0.59M | 72.97 | 25.14 |
| **Uni-LoRA** | **0.52M** | **75.59** | **28.94** |

### Instruction Tuning (Llama2-13B, MT-Bench)

| Method | Trainable Params | Score1 | Score2 |
|---|---|---|---|
| LoRA | 250.3M | 6.20 | 4.13 |
| VB-LoRA | 256M† | 5.96 | 4.33 |
| **Uni-LoRA** | **1.0M** | **6.34** | **4.43** |

### Key Findings

- Uni-LoRA ranks in the top two on 11 of 12 GLUE experiments while using the fewest trainable parameters.
- On Gemma-7B, only 0.52M parameters (0.0061% of the base model; 0.26% of LoRA) are required to match or surpass LoRA.
- Ablation experiments comparing uniform vs. non-uniform projection confirm the importance of uniformity.
- Isometric random projection matches the performance of Fastfood projection while reducing computational complexity from $\mathcal{O}(D \log d)$ to $\mathcal{O}(D)$.
- The approach generalizes effectively to computer vision tasks (ViT-Base/Large).

## Highlights & Insights

- The unified framework is itself a significant contribution, subsuming seemingly disparate LoRA variants under the single lens of projection matrix design.
- The "one vector is all you need" minimalist design is notable — random grouping with shared values, despite its simplicity, rivals carefully engineered alternatives.
- The isometry proof is concise and elegant: $P^\top P = I_d$ directly guarantees distance preservation.
- Parameter efficiency reaches a new extreme: 0.0061% of base model parameters suffice to attain LoRA-level performance.

## Limitations & Future Work

- Random grouping treats all parameters equally regardless of their importance; adaptive grouping strategies may yield further improvements.
- Isometry guarantees that the optimization landscape geometry is not distorted, but does not ensure the subspace itself is optimal.
- The choice of $d$ requires grid search; no principled method for automatic dimensionality selection is proposed.
- Performance under extremely low ranks $r$ and at very large scales (>13B parameters) remains unexplored.

## Related Work & Insights

- The work connects to intrinsic dimensionality research (Li et al. 2018; Aghajanyan et al. 2021): the effective degrees of freedom in LoRA parameter space are far below its nominal dimensionality.
- FourierFT performs local projection in the frequency domain, whereas Uni-LoRA performs global projection in the original parameter space.
- Implication: parameter-efficient fine-tuning may be approaching a "limit of parameter sharing" — further compression likely requires more intelligent grouping rather than additional constraints.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Both the unified framework perspective and the isometric random projection are highly original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four task categories — NLU, mathematical reasoning, instruction tuning, and CV — with comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Framework diagrams are intuitive, theoretical proofs are concise, and pseudocode is directly implementable.
- Value: ⭐⭐⭐⭐ Extreme parameter efficiency has practical deployment value, though LoRA itself is already sufficiently lightweight.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] All You Need is One: Capsule Prompt Tuning with a Single Vector](all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)

<!-- RELATED:END -->
