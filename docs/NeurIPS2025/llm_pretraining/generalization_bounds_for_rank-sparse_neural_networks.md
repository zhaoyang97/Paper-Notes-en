---
title: >-
  [Paper Note] Generalization Bounds for Rank-sparse Neural Networks
description: >-
  [NeurIPS 2025][LLM Pretraining][generalization bounds] This paper establishes generalization bounds that exploit the approximate low-rank structure of neural network weight matrices. When the Schatten $p$ quasi-norm is s…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "generalization bounds"
  - "low-rank structure"
  - "Schatten norm"
  - "bottleneck rank"
  - "sample complexity"
date: 2026-05-08
content_hash: 97fba39ebabfc333
---

# Generalization Bounds for Rank-sparse Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2510.21945](https://arxiv.org/abs/2510.21945)  
**Code**: None  
**Area**: Learning Theory / Generalization Theory
**Keywords**: generalization bounds, low-rank structure, Schatten norm, bottleneck rank, sample complexity

## TL;DR

This paper establishes generalization bounds that exploit the approximate low-rank structure of neural network weight matrices. When the Schatten $p$ quasi-norm is small, the sample complexity reduces to $\widetilde{O}(WrL^2)$, where $W$, $L$, and $r$ denote the width, depth, and rank of the weight matrices, respectively.

## Background & Motivation

### The Bottleneck Rank Phenomenon

A growing body of literature has observed that neural networks trained with gradient-based methods exhibit a **bottleneck rank** property: as network depth increases, the activations and weights at each layer tend to become approximately low-rank. Specifically, the rank of each layer's activations converges to a fixed value—the bottleneck rank—corresponding to the minimum rank required to represent the training data.

### Limitations of Existing Generalization Bounds

Classical generalization bounds typically rely on the following strategies:
- **Norm-based bounds**: Depend on spectral or Frobenius norms of weight matrices, generally yielding sample complexity of order $\widetilde{O}(W^2 L)$.
- **Compression-based bounds**: Although tighter, they do not directly leverage low-rank structure.
- **PAC-Bayes bounds**: Sensitive to the choice of prior distribution.

None of these approaches systematically exploit the low-rank structure of weight matrices. This paper addresses that gap.

### Connection to Weight Decay

For linear networks (without activation functions), weight decay regularization is equivalent to minimizing the Schatten $p$ quasi-norm of the end-to-end network. This makes the Schatten norm a natural tool for characterizing the generalization ability of low-rank networks.

## Method

### Overall Architecture

The central approach is to develop a generalization theory that leverages Schatten $p$ quasi-norms to capture the approximate low-rank structure of weight matrices. The key insight is that the Schatten quasi-norm interpolates between **rank sparsity** and **norm constraints**, with the parameter $p$ controlling the balance between these two effects.

### Key Designs

#### Schatten $p$ Quasi-norm

For a matrix $A$, the Schatten $p$ quasi-norm is defined as:

$$\|A\|_{S_p}^p = \sum_{i=1}^{\min(m,n)} \sigma_i(A)^p$$

where $\sigma_i(A)$ denotes the $i$-th singular value of $A$. Smaller values of $p$ make the norm more sensitive to rank, while larger values recover the behavior of conventional norm constraints.

#### Main Theoretical Results

The core theorem establishes generalization bounds of the following form:

| Parameter $p$ | Sample Complexity | Characteristics |
|:---|:---|:---|
| $p \to 0$ | $\widetilde{O}(WrL^2)$ | Fully exploits low-rank structure; linear in rank $r$ |
| $p = 1$ | Intermediate | Balances rank and norm |
| $p \to \infty$ | $\widetilde{O}(W^2L)$ | Recovers classical norm-based bound |

Here $W$ is the network width, $L$ is the depth, and $r$ is the rank of the weight matrices.

#### Covering Number Analysis

The central technical tool is an estimate of the covering numbers of sets constrained by the Schatten quasi-norm. For the Schatten $p$ ball:

$$\mathcal{B}_{S_p}(R) = \{A : \|A\|_{S_p} \leq R\}$$

the authors derive upper bounds on the $\epsilon$-covering number that are substantially tighter than those based on the Frobenius norm in the low-rank regime.

### Proof Strategy

The main proof proceeds in the following steps:

1. **Layer-wise decomposition**: The generalization error is decomposed into contributions from each layer's weight matrix.
2. **Covering number factorization**: The covering numbers of Schatten quasi-norm balls at each layer are combined.
3. **Lipschitz continuity of composed functions**: The network's Lipschitz constant is controlled via the product of spectral norms across layers.
4. **Union bound**: Coverings across all layers are combined to obtain the final generalization bound.

## Key Experimental Results

### Comparison of Theoretical Results

This is a purely theoretical work; its core contribution lies in the mathematical proofs. The following table compares different generalization bound approaches:

| Method | Sample Complexity | Exploits Low-Rank | Depth Dependence |
|:---|:---|:---|:---|
| Spectral norm (Bartlett et al.) | $\widetilde{O}(W^2 L)$ | No | $O(L)$ |
| Frobenius norm | $\widetilde{O}(W^2 L)$ | Indirectly | $O(L)$ |
| PAC-Bayes (Neyshabur et al.) | $\widetilde{O}(W^2 L^2)$ | No | $O(L^2)$ |
| **Ours (small $p$)** | $\widetilde{O}(WrL^2)$ | **Yes** | $O(L^2)$ |
| **Ours (large $p$)** | $\widetilde{O}(W^2 L)$ | Partially | $O(L)$ |

### Key Findings

| Setting | Rank $r$ | Width $W$ | Improvement Factor |
|:---|:---|:---|:---|
| Full rank ($r = W$) | $W$ | $W$ | 1× (matches classical methods) |
| Moderate rank ($r = \sqrt{W}$) | $\sqrt{W}$ | $W$ | $\sqrt{W}$ improvement |
| Very low rank ($r = O(1)$) | $O(1)$ | $W$ | $W$ improvement |

### Key Theoretical Findings

1. **Low-rank networks generalize more easily**: When weight matrices are low-rank ($r \ll W$), the proposed bounds are substantially tighter than classical norm-based bounds, providing a theoretical explanation for the superior generalization observed in low-rank networks in practice.

2. **Continuous interpolation**: The parameter $p$ provides a continuous interpolation between a pure rank-constraint bound ($p \to 0$) and a pure norm-constraint bound ($p \to \infty$), enabling selection of the optimal $p$ based on the actual rank structure of the network.

3. **Connection to the bottleneck rank phenomenon**: The theoretical results are consistent with the empirically observed bottleneck rank phenomenon—whereby deep networks trained with gradient descent tend to learn low-rank weight matrices—and provide a formal justification that this implicit bias is beneficial for generalization.

## Highlights & Insights

- **Theoretical unification**: By varying $p$, rank constraints and norm constraints are unified within a single framework, with different values of $p$ corresponding to different generalization regimes.
- **Practical relevance**: The results provide theoretical justification for model compression techniques such as low-rank matrix factorization and knowledge distillation.
- **New perspective on weight decay**: In linear networks, weight decay implicitly minimizes the Schatten norm; analogous effects may exist in nonlinear networks.

## Limitations & Future Work

- Although tighter in the low-rank regime, the bounds may still be loose; empirical generalization error is typically far below the theoretical upper bound.
- The depth dependence is $O(L^2)$ for small $p$, which is worse than the $O(L)$ dependence of some classical methods.
- The analysis does not address the approximately low-rank case in which ranks are not exactly small but singular values decay rapidly.
- The results are derived for fully connected networks and have not been extended to convolutional or attention-based architectures.

## Related Work & Insights

- **Bartlett et al. (2017)**: Spectral norm-based generalization bounds; seminal contribution to the field.
- **Neyshabur et al. (2018)**: Generalization analysis under the PAC-Bayes framework.
- **Arora et al. (2018)**: Compression-based bounds via matrix factorization.
- **Bottleneck rank literature**: Empirical studies such as Huh et al. (2021) document the convergence of layer ranks during training.
- This paper provides theoretical justification from a generalization-theoretic perspective for the use of low-rank network structures.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Schatten quasi-norm generalization bounds constitute a novel theoretical direction.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Rigorous mathematical proofs with careful covering number analysis.
- **Practicality**: ⭐⭐⭐ — Purely theoretical, but offers meaningful guidance for low-rank compression.
- **Clarity**: ⭐⭐⭐⭐ — Theoretical results are clearly stated; the continuous interpolation intuition is well motivated.
- **Overall Score**: 8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)
- [\[ICLR 2026\] Block-Sample MAC-Bayes Generalization Bounds](../../ICLR2026/llm_pretraining/block-sample_mac-bayes_generalization_bounds.md)
- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)

</div>

<!-- RELATED:END -->
