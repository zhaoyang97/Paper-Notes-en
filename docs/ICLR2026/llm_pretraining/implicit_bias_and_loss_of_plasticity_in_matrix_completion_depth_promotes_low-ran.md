---
title: >-
  [Paper Note] Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank
description: >-
  [ICLR 2026][LLM Pretraining][matrix completion] By analyzing the gradient flow dynamics of deep matrix factorization (deep linear networks) in matrix completion…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "matrix completion"
  - "deep matrix factorization"
  - "implicit bias"
  - "low-rank preference"
  - "loss of plasticity"
date: 2026-05-08
content_hash: 6d0bc6727dbe2d7c
---

# Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank

**Conference**: ICLR 2026
**arXiv**: [2603.04703](https://arxiv.org/abs/2603.04703)  
**Code**: None  
**Area**: LLM Pretraining
**Keywords**: matrix completion, deep matrix factorization, implicit bias, low-rank preference, loss of plasticity

## TL;DR

By analyzing the gradient flow dynamics of deep matrix factorization (deep linear networks) in matrix completion, this paper proves that coupled dynamics is the key mechanism underlying the low-rank implicit bias of deep networks, and that networks of depth $\geq 3$ inevitably exhibit coupling except under diagonal initialization. This provides a theoretical explanation for why deep models are able to avoid loss of plasticity.

## Background & Motivation

Matrix completion is a fundamental and important problem: given partial observations of a matrix, recover the full matrix. Deep matrix factorization — representing the target matrix as a product of multiple matrices $W = W_L \cdot W_{L-1} \cdots W_1$ — is equivalent to a deep linear neural network and serves as an ideal simplified testbed for studying how depth affects learning dynamics.

Despite extensive prior study of deep matrix factorization, existing theory has two critical gaps:

**Most theory focuses on shallow (depth-2) models**: Such work cannot fully explain the stronger low-rank preference observed in deeper networks. Why does greater depth cause models to converge more strongly to low-rank solutions? Prior work (e.g., Menon, 2024) identified this as an open problem without resolving it.

**Loss of plasticity lacks a theoretical explanation**: Kleinman et al. (2024) found that pretraining on a small set of observations and then continuing training on a larger set can actually degrade performance — a phenomenon known as loss of plasticity. Deep networks are immune to this effect while shallow networks suffer severely, yet the underlying mechanism remains unclear.

The central contribution of this paper is the identification of **coupled dynamics** as the key mechanism that unifies depth, low-rank bias, and loss of plasticity within a single theoretical framework.

## Method

### Overall Architecture

The paper adopts a purely theoretical approach, with gradient flow (the limiting case of gradient descent as the learning rate approaches zero) as the primary analytical tool. The analysis is conducted under the **block-diagonal observation** setting — a special yet representative case that admits exact theoretical results.

### Key Designs

1. **Formal Definition of Coupled Dynamics**:

   In deep matrix factorization $W = W_L \cdots W_1$, consider the dynamics of the product matrix. When the gradient flow of different components (e.g., different diagonal blocks) of the factor matrices $W_l$ mutually influence one another, the dynamics are termed **coupled**; otherwise they are **decoupled**.

   Coupling is central to the low-rank bias: coupled dynamics cause different components of the system to "compete" for resources, naturally suppressing certain directions and producing low-rank solutions.

2. **Inevitability of Coupling for Depth $\geq 3$**:

   The first core theorem establishes that for networks of depth $L \geq 3$, **unless the initialization is exactly diagonal**, the gradient flow dynamics are necessarily coupled. This is a measure-zero result with respect to initial conditions — almost all initializations lead to coupling.

   By contrast, depth-2 networks can maintain decoupled dynamics under a much broader range of initializations, explaining why shallow networks exhibit weaker low-rank preference.

3. **Equivalence Between Coupling and Rank-1 Convergence**:

   The second core theorem proves that under the block-diagonal observation setting, **convergence to rank-1 holds if and only if the dynamics are coupled**. This resolves the open problem posed by Menon (2024): what is the necessary and sufficient condition for depth to promote low rank?

   Intuitively, coupled dynamics create a "winner-takes-all" competition among components — energy concentrates along the dominant direction while all others are suppressed to zero.

4. **Mechanism of Loss of Plasticity**:

   The paper analyzes the following scenario: pretraining on a small set of observations followed by continued training on a larger set.

   - **Deep networks ($L \geq 3$)**: Coupled dynamics during pretraining drive the model toward a low-rank solution. Since the coupling condition continues to hold during subsequent training, the model can effectively adapt from the low-rank initialization to new data — the implicit bias conferred by depth is adaptive.

   - **Depth-2 networks**: If pretraining proceeds under decoupled dynamics, the model converges to a high-rank solution. Although continued training (with more observations) may satisfy the coupling condition, **gradient flow initialized from a high-rank solution cannot converge to a low-rank solution**. The "imprint" of pretraining prevents the model from adapting — this is the root cause of loss of plasticity.

### Theoretical Tools

- **Gradient flow analysis**: ODE systems governing parameter evolution in the continuous-time limit
- **Block-diagonal structure**: Enables decomposition of high-dimensional problems into interacting low-dimensional components
- **Invariants and conservation laws**: Identification of invariants in gradient flow to constrain long-time behavior
- **Łojasiewicz inequality**: Used to establish convergence of gradient flow

## Key Experimental Results

### Main Results: Numerical Validation of Coupling and Low Rank

The paper validates its theoretical results through numerical simulation on synthetic matrix completion tasks:

| Depth $L$ | Initialization Type | Coupled? | Convergence Rank | Loss of Plasticity |
|-----------|--------------------|---------|-----------------|--------------------|
| 2 | General | Decoupled | High rank | ✓ Present |
| 2 | Special | Coupled | Rank-1 | ✗ Absent |
| 3 | General | Coupled | Rank-1 | ✗ Absent |
| 3 | Diagonal | Decoupled | High rank | ✓ Present |
| 5 | General | Strongly coupled | Rank-1 | ✗ Absent |

### Ablation Study: Effect of Depth on Coupling Strength

| Depth $L$ | Coupling Strength | Speed of Rank-1 Convergence | Notes |
|-----------|------------------|-----------------------------|-------|
| 2 | None / Weak | Slow / Does not converge | Initialization-dependent |
| 3 | Moderate | Moderate | Coupled under almost all initializations |
| 5 | Strong | Fast | Coupling effect strengthens with depth |
| 10 | Very strong | Very fast | Stronger low-rank preference at greater depth |

### Loss of Plasticity Experiment

| Configuration | Pretraining Phase | Continued Training Phase | Final Performance |
|--------------|------------------|--------------------------|-------------------|
| Depth-2, pretrained on few observations | Decoupled → high rank | More observations added | Performance degrades (loss of plasticity) |
| Depth-3, pretrained on few observations | Coupled → low rank | More observations added | Performance improves (no loss of plasticity) |
| Depth-2, no pretraining | — | All observations directly | Normal convergence |

### Key Findings

1. **Coupling is a necessary and sufficient condition**: Rank-1 convergence $\Leftrightarrow$ coupled dynamics; this equivalence is remarkably clean.
2. **Depth $\geq 3$ marks a qualitative transition**: The gap between depth-2 and depth-3 is fundamental, not merely a matter of degree.
3. **Loss of plasticity is a residual effect of decoupling**: High-rank representations learned by shallow networks under decoupled conditions "lock" the model's capacity for subsequent learning.
4. **Depth provides implicit regularization**: Without any explicit rank constraint or regularization term, depth alone induces a preference for low-rank solutions.

## Highlights & Insights

- **Clean theoretical contributions**: The two main theorems each characterize a key aspect of the problem, and together provide a complete answer to how depth promotes low rank.
- **A new perspective on loss of plasticity**: The paper connects the widely discussed phenomenon of loss of plasticity in neural networks to the concrete mathematical structure of linear networks, providing a precisely analyzable test case.
- **Resolution of an open problem**: The paper explicitly answers the open question posed by Menon (2024) regarding the conditions under which depth leads to low-rank convergence.
- **Practical implications**: Although the analysis is restricted to linear networks, the insight that "depth induces implicit low-rank preference" is broadly applicable to nonlinear settings and offers guidance for understanding overparameterization and generalization in deep learning.

## Limitations & Future Work

1. **Restricted to linear networks**: Deep matrix factorization is a useful theoretical tool, but the nonlinear dynamics of real deep networks may exhibit qualitatively different behavior.
2. **Specificity of block-diagonal observations**: The theoretical results rely on this structural assumption; generalization to arbitrary observation patterns remains to be established.
3. **Gradient flow vs. discrete gradient descent**: The analysis is conducted in the continuous-time limit; additional effects may arise at finite learning rates.
4. **Interaction between depth and width not addressed**: In practice, width also influences implicit bias; the joint effect of depth and width is not considered.
5. **Limited discussion of initialization scale**: The paper primarily addresses the structure of initialization (diagonal vs. non-diagonal); the effect of initialization scale receives limited treatment.

## Related Work & Insights

- **Connection to Arora et al. (2019) and Razin & Cohen (2020)**: These works revealed the low-rank preference of deep matrix factorization; the present paper provides more precise necessary and sufficient conditions.
- **Complementary to loss-of-plasticity research by Lyle et al. (2023) and Kumar et al. (2024)**: The latter empirically observe loss of plasticity in nonlinear networks; this paper provides exact theory in the linear setting.
- **Implications for continual learning and domain adaptation**: Loss of plasticity is a central challenge in continual learning; the depth–coupling–low-rank mechanism uncovered here offers a theoretical basis for designing new methods.
- **Impact on matrix recovery algorithms**: Demonstrates that deep parameterization can serve as implicit nuclear norm regularization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The formal analysis of coupled dynamics constitutes an entirely new perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Numerical validation is thorough, but limited to synthetic data)
- Writing Quality: ⭐⭐⭐⭐⭐ (A model theoretical paper with clearly stated theorems)
- Value: ⭐⭐⭐⭐ (Excellent theoretical contributions, though practical applicability remains distant)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ETA: Energy-based Test-time Adaptation for Depth Completion](../../ICCV2025/llm_pretraining/eta_energy-based_test-time_adaptation_for_depth_completion.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](../../NeurIPS2025/llm_pretraining/breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[NeurIPS 2025\] Global Minimizers of Sigmoid Contrastive Loss](../../NeurIPS2025/llm_pretraining/global_minimizers_of_sigmoid_contrastive_loss.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](../../NeurIPS2025/llm_pretraining/the_curse_of_depth_in_large_language_models.md)

</div>

<!-- RELATED:END -->
