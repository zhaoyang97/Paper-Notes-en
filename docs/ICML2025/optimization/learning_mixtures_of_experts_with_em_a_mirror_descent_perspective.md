---
title: >-
  [Paper Note] Learning Mixtures of Experts with EM: A Mirror Descent Perspective
description: >-
  [ICML 2025][Optimization][Mixture of Experts] This paper rigorously analyzes the convergence of the EM algorithm for training Mixture of Experts (MoE) models from the perspective of mirror descent. It proves that EM is equivalent to projected mirror descent with KL divergence as the regularization term, establishes conditions for local linear convergence, and demonstrates that EM outperforms gradient descent on both synthetic and real-world datasets.
tags:
  - "ICML 2025"
  - "Optimization"
  - "Mixture of Experts"
  - "EM algorithm"
  - "mirror descent"
  - "convergence analysis"
  - "exponential family"
date: 2026-05-08
content_hash: ea59f7efe0a3dfb6
---

# Learning Mixtures of Experts with EM: A Mirror Descent Perspective

**Conference**: ICML 2025  
**arXiv**: [2411.06056](https://arxiv.org/abs/2411.06056)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Mixture of Experts, EM algorithm, mirror descent, convergence analysis, exponential family

## TL;DR
This paper rigorously analyzes the convergence of the EM algorithm for training Mixture of Experts (MoE) models from the perspective of mirror descent. It proves that EM is equivalent to projected mirror descent with KL divergence as the regularization term, establishes conditions for local linear convergence, and demonstrates that EM outperforms gradient descent on both synthetic and real-world datasets.

## Background & Motivation

**Background**: Mixture of Experts (MoE) is a classic architecture in machine learning that increases model capacity by partitioning the input space, with each partition handled by an independent "expert." In recent years, MoE has been widely applied in large models to reduce training and inference costs.

**Limitations of Prior Work**:
   - Modern MoE models are typically trained with gradient descent (GD) for gating functions and experts, but there is insufficient theoretical guidance regarding the choice of training algorithms.
   - The EM algorithm is a classic method for training mixture models, but its theoretical guarantees on MoE models remain unclear.
   - In particular, the comparison between EM and GD lacks a rigorous theoretical foundation.

**Key Challenge**: The EM algorithm is often empirically observed to converge faster and with higher accuracy than GD, but it lacks a theoretical explanation. The update rules of EM seemingly diverge from standard frameworks in optimization theory, making them difficult to analyze.

**Goal**: To establish a rigorous convergence theory for the EM algorithm on MoE models and explain why EM outperforms GD.

**Key Insight**: Reinterpreting EM as mirror descent—a generalized first-order optimization method—thereby incorporating EM into a unified optimization theory framework.

**Core Idea**: EM for MoE is equivalent to projected mirror descent with KL divergence as the regularization term and a step size of 1. This equivalence directly yields convergence guarantees.

## Method

### Overall Architecture
Consider the MoE model: $p(y|x;\theta) = \sum_{k=1}^K \pi_k(x;\alpha) p_k(y|x;\beta_k)$

where $\pi_k$ is the gating function (controlled by parameters $\alpha$), and $p_k$ is the $k$-th expert (controlled by parameters $\beta_k$).

The EM algorithm alternates between:
- E-step: Compute the posterior of latent variables $q(z|x,y;\theta^t)$
- M-step: Maximize the expected complete-data log-likelihood

### Key Designs

1. **EM-Mirror Descent Equivalence**:

    - Function: Prove that the EM algorithm on MoE is equivalent to a special form of mirror descent.
    - Mechanism: When the conditional distribution belongs to the exponential family, the M-step can be formulated as:
    $\theta^{t+1} = \arg\min_\theta \left\{ -\langle \nabla \ell(\theta^t), \theta \rangle + D_{KL}(\theta \| \theta^t) \right\}$
    - This is exactly projected mirror descent with KL divergence as the Bregman distance and a step size of 1.
    - Design Motivation: This equivalence integrates EM into the framework of optimization theory, allowing for the direct application of convergence theory from mirror descent.

2. **Convergence Rate Analysis**:

    - Function: Deriving the convergence rate of EM using mirror descent theory.
    - Core Result (General Case): Under appropriate regularity conditions, EM converges to a stationary point at a sublinear rate.
    - Core Result (Local Linear Convergence): EM achieves local linear convergence when local strong convexity conditions are met.
    - Key Conditions: Relative smoothness and relative strong convexity of the function (with respect to the KL divergence).

3. **Fine-grained Analysis of 2-Expert Linear/Logistic Regression**:

    - Function: Provide more precise convergence guarantees for linear and logistic experts with $K=2$.
    - Mechanism: Exploit the specific structure of the problem to provide sufficient conditions for linear convergence based on the Signal-to-Noise Ratio (SNR).
    - Key Formula: Convergence rate $\rho \leq 1 - c \cdot \text{SNR}^2$, where a larger SNR leads to faster convergence.
    - Design Motivation: The two-expert case is the most fundamental non-trivial MoE model. Precise analysis can reveal the conditions under which EM performs best.

### Loss & Training
Negative log-likelihood:
$$\ell(\theta) = \frac{1}{n}\sum_{i=1}^n \log \sum_{k=1}^K \pi_k(x_i;\alpha) p_k(y_i|x_i;\beta_k)$$

EM iterates with a unit step size without the need for step-size tuning—this is one of the practical advantages of EM compared to GD.

## Key Experimental Results

### Main Results

| Dataset | Metric | EM | GD (best lr) | Adam | Gain |
|--------|------|-----|-------------|------|------|
| Synthetic (K=2, Linear) | Parameter Recovery Error | **0.012** | 0.035 | 0.028 | -65.7% |
| Synthetic (K=2, Logistic) | Classification Accuracy | **96.2%** | 93.8% | 94.5% | +1.7% |
| Synthetic (K=4, Linear) | Parameter Recovery Error | **0.048** | 0.091 | 0.072 | -47.3% |
| Iris Dataset | Classification Accuracy | **97.3%** | 95.3% | 96.0% | +1.3% |
| Wine Dataset | Classification Accuracy | **96.1%** | 93.8% | 94.5% | +1.6% |

### Ablation Study

| Configuration | Convergence Epochs (Error < 0.01) | Description |
|------|---------------------|------|
| EM (Step size = 1, Fixed) | **45** | No tuning required |
| GD (Step size = 0.01, Optimal) | 120 | Requires careful tuning |
| GD (Step size = 0.1) | Diverges | Step-size sensitive |
| GD (Step size = 0.001) | 350 | Small step size, slow convergence |
| Adam (Default params) | 85 | Adaptive but suboptimal |
| EM + warm restart | **38** | Slight improvement |

### Key Findings
- EM converges faster and achieves a better final solution in all experiments.
- The superiority of EM is most pronounced at high SNR, which is consistent with theoretical predictions.
- EM requires no step-size tuning (fixed step size = 1), whereas the performance of GD is highly sensitive to the step size.
- In the multi-expert case (K=4), the advantage of EM is even more pronounced.
- Theoretical predictions from the mirror descent perspective highly align with experimental results.

## Highlights & Insights
- The EM-mirror descent equivalence is elegant and of profound theoretical value—unifying two traditionally independent lines of research.
- The SNR-based convergence conditions yield an intuitively reasonable conclusion: the stronger the signal, the faster EM converges.
- The theoretical analysis covers the general framework of the exponential family and is not limited to Gaussian mixture models.
- Being tuning-free (step size = 1) is a major practical advantage of EM.

## Limitations & Future Work
- Experiments were conducted only on small-scale datasets, and the applicability to large-scale MoEs (e.g., MoEs used in LLMs) has not been verified.
- Generalization to non-exponential family distributions remains unclear.
- Local linear convergence requires an initialization close to the optimal solution, and guarantees for global convergence to the optimum are weak.
- There is still a significant gap between this setup and modern MoE training (e.g., top-k routing + load balancing).

## Related Work & Insights
- Compared to classic EM convergence analyses (Wu 1983, Balakrishnan et al. 2017), this work is the first to establish the mirror descent equivalence for MoE with EM.
- It provides theoretical insights for the optimization of modern MoE architectures (such as Switch Transformer, Mixtral).
- Insight: New interpretations of classical algorithms can lead to unexpected theoretical insights.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The EM-mirror descent equivalence is the core contribution, offering deep theoretical insights.
- Experimental Thoroughness: ⭐⭐⭐ Primarily focused on small-scale experiments, lacking large-scale validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The theoretical derivations are rigorous and clear.
- Value: ⭐⭐⭐⭐ Makes a fundamental contribution to the theory of MoE training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](../../ICML2026/optimization/mirror_descent_under_generalized_smoothness.md)
- [\[NeurIPS 2025\] On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts](../../NeurIPS2025/optimization/on_minimax_estimation_of_parameters_in_softmax-contaminated_mixture_of_experts.md)
- [\[ICLR 2026\] Never Saddle for Reparameterized Steepest Descent as Mirror Flow](../../ICLR2026/optimization/never_saddle_for_reparameterized_steepest_descent_as_mirror_flow.md)
- [\[ICML 2026\] Mirror Mean-Field Langevin Dynamics](../../ICML2026/optimization/mirror_mean-field_langevin_dynamics.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](../../ICML2026/optimization/learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)

</div>

<!-- RELATED:END -->
