---
title: >-
  [Paper Note] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity
description: >-
  [NeurIPS 2025][Optimization][second-order optimization] This paper provides the first systematic theoretical treatment of second-order stochastic optimization under heavy-tailed noise. It establishes tight minimax sample complexity lower bounds, proposes a normalized SGD algorithm with gradient and Hessian clipping (Clip NSGDHess), and proves that the proposed algorithm nearly achieves the information-theoretic limit.
tags:
  - NeurIPS 2025
  - Optimization
  - second-order optimization
  - heavy-tailed noise
  - Hessian clipping
  - sample complexity
  - high-probability convergence
date: 2026-05-08
content_hash: 35b4ea0cec6563e8
---

# Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity

**Conference**: NeurIPS 2025
**arXiv**: [2510.10690](https://arxiv.org/abs/2510.10690)
**Code**: None
**Area**: Optimization
**Keywords**: second-order optimization, heavy-tailed noise, Hessian clipping, sample complexity, high-probability convergence

## TL;DR

This paper provides the first systematic theoretical treatment of second-order stochastic optimization under heavy-tailed noise. It establishes tight minimax sample complexity lower bounds, proposes a normalized SGD algorithm with gradient and Hessian clipping (Clip NSGDHess), and proves that the proposed algorithm nearly achieves the information-theoretic limit.

## Background & Motivation

Heavy-tailed noise is prevalent in modern machine learning, arising from data heterogeneity, outliers, and non-stationary environments. Existing theoretical and algorithmic frameworks suffer from fundamental deficiencies:

**First-order methods are mature**: SGD and its variants already enjoy optimal convergence guarantees and complete complexity theory under the $p$-BCM (bounded $p$-th central moment) noise model, with gradient clipping established as a standard tool.

**Second-order methods lag significantly**: (1) Existing second-order methods (stochastic cubic regularization SCN, trust-region methods, etc.) require bounded noise or bounded variance assumptions, offering no guarantees under heavy-tailed conditions; (2) No existing second-order method can operate reliably under unbounded variance noise; (3) Hessian estimation is more sensitive to noise than gradient estimation, yet the corresponding robustness tools (e.g., Hessian clipping) have never been studied.

**Core Problem**: Can one design second-order algorithms that retain high-probability convergence guarantees when both gradients and Hessians are corrupted by heavy-tailed noise?

## Method

### Overall Architecture

The paper is organized around three levels: (1) establishing minimax sample complexity lower bounds for second-order methods under $p$-BCM noise; (2) designing the optimal algorithm NSGDHess that matches the lower bound; (3) achieving high-probability convergence guarantees via Hessian clipping.

### Key Designs

1. **Minimax Lower Bound (Theorem 1)**: For any $q$-th-order zero-respecting algorithm, the minimum number of oracle queries required to find an $\varepsilon$-stationary point under $p$-BCM noise ($p \in (1,2]$) is:

    $$\Omega\left(\frac{\Delta \sigma_h}{\varepsilon^2} \left(\frac{\sigma}{\varepsilon}\right)^{\frac{1}{p-1}}\right)$$

    This improves upon the optimal complexity of first-order methods by a factor of $1/\varepsilon$ (uniformly for all $p \in (1,2]$), demonstrating that second-order information provides provable acceleration even under heavy-tailed noise. The key technique is a worst-case function construction exploiting the "zero-chain" property, extending the $p$-BCM noise model to higher-order derivatives.

2. **Normalized SGD with Hessian Correction (NSGDHess, Algorithm 1)**: The core update rule is:

    - Uniformly random interpolation between consecutive iterates: $\hat{x}_t = q_t x_t + (1-q_t) x_{t-1}$
    - Recursive momentum with Hessian correction: $g_t = (1-\alpha)(g_{t-1} + \nabla^2 f(\hat{x}_t, \hat{\xi}_t)(x_t - x_{t-1})) + \alpha \nabla f(x_t, \xi_t)$
    - Normalized step: $x_{t+1} = x_t - \gamma \frac{g_t}{\|g_t\|}$

    **Design Motivation**: The random interpolation point ensures $\mathbb{E}[\nabla^2 f(\hat{x}_t)(x_t - x_{t-1})] = \nabla F(x_t) - \nabla F(x_{t-1})$, i.e., the exact gradient difference, which forms the theoretical basis for the effectiveness of Hessian-corrected momentum. The normalized step guarantees bounded update magnitude ($\|x_{t+1} - x_t\| = \gamma$), which is critical under heavy-tailed noise.

3. **Hessian Clipping (Clip NSGDHess, Algorithm 2)**: Algorithm 1 is augmented with dual clipping of gradients and Hessian-vector products:

    $$g_t = (1-\alpha)(g_{t-1} + \gamma \cdot \text{clip}(\gamma^{-1}\nabla^2 f \cdot (x_t - x_{t-1}), \bar{\lambda}_h)) + \alpha \cdot \text{clip}(\nabla f, \lambda)$$

    where $\text{clip}(v, \lambda) = \min\{1, \lambda/\|v\|\} \cdot v$. **Key Insight**: Clipping the Hessian matrix directly requires $O(d^2)$ computation (operator norm estimation), whereas clipping the Hessian-vector product requires only $O(d)$ operations while remaining compatible with backpropagation.

### Loss & Training

- **Convergence target**: Finding an $\varepsilon$-first-order stationary point, i.e., $\|\nabla F(x)\| \leq \varepsilon$
- **NSGDHess sample complexity (Theorem 2)**: $O\left(\frac{\Delta(L+\sigma_h)}{\varepsilon^2}\left(\frac{\sigma}{\varepsilon}\right)^{\frac{1}{p-1}}\right)$, exactly matching the lower bound when $L \leq \sigma_h$
- **Clip NSGDHess high-probability guarantee (Theorem 3)**: Achieves near-optimal complexity with probability $\geq 1-\delta$, with only $\text{polylog}(T/\delta)$ additional overhead

## Key Experimental Results

### Main Results

Synthetic experiment: $F(x) = 0.5\|x\|^2$, $d=10$, with heavy-tailed noise generated from a two-sided Pareto distribution (tail index $p=1.1$).

| Algorithm | Noise Robustness | Convergence | Clipping Required |
|-----------|-----------------|-------------|-------------------|
| NSGDM (first-order normalized momentum) | Poor | Severe oscillation | No |
| NSGDHess (without clipping) | Moderate | Improved but unstable | No |
| **Clip NSGDHess** | **Strong** | **Stable and fast** | **Yes** |

### Ablation Study

| Configuration | Complexity (in $\varepsilon$) | High-Probability? | Remarks |
|---------------|------------------------------|-------------------|---------|
| First-order FOSO (optimal) | $O(\varepsilon^{-3} \cdot (\sigma/\varepsilon)^{1/(p-1)})$ | Yes | Known lower bound |
| Second-order NSGDHess (expected) | $O(\varepsilon^{-2} \cdot (\sigma/\varepsilon)^{1/(p-1)})$ | No | This paper, Thm 2 |
| **Second-order Clip NSGDHess** | $O(\varepsilon^{-2} \cdot (\sigma/\varepsilon)^{1/(p-1)})$ | **Yes** | **This paper, Thm 3** |
| SCN (Tripuraneni 2018) | $O(\varepsilon^{-7/2})$ | No | Requires bounded noise |
| SGDHess (Tran 2021) | $O(\varepsilon^{-3})$ | No | Requires bounded variance |

### Key Findings

- Second-order methods retain a $1/\varepsilon$ sample complexity advantage over first-order methods even under heavy-tailed noise
- The unclipped variant suffers noticeably from noise in synthetic experiments, validating the necessity of Hessian clipping
- The complexity result does not depend on the second-order smoothness constant $L_h$ (which may be infinite), an important practical feature
- For the special case $p=2$ (bounded variance), the results are also novel and unify previously scattered conclusions

## Highlights & Insights

- **First complete theory of second-order optimization under heavy-tailed noise**: Provides a full minimax characterization from lower to matching upper bounds
- **Hessian clipping is an entirely new concept**: Generalizes the success of gradient clipping to the Hessian, maintaining computational feasibility via Hessian-vector product clipping
- **Theoretically proves that the advantage of second-order methods does not vanish under heavy-tailed noise**: The $1/\varepsilon$ improvement is fundamental
- The Hessian-vector product clipping formula $\gamma \cdot \text{clip}(\gamma^{-1} \nabla^2 f \cdot v, \bar{\lambda}_h)$ is elegantly designed so that the clipping thresholds $\lambda$ and $\bar{\lambda}_h$ are on the same scale, facilitating practical hyperparameter tuning

## Limitations & Future Work

- Experiments are conducted only on synthetic problems; empirical validation on real machine learning tasks is absent
- The algorithm requires multiple hyperparameters (step size $\gamma$, momentum $\alpha$, two clipping thresholds), and further simplification is needed for practical deployment
- The convergence target is limited to first-order stationary points; extending to second-order stationary points (saddle point escape) remains an important open problem
- Whether normalized steps are necessary in the clipped variant, and whether pure clipping can replace normalization, merits further investigation

## Related Work & Insights

- **Relation to gradient clipping**: This paper is a natural extension of clipping techniques from first- to second-order methods, continuing the line of work by Gorbunov (2020), Sadiev (2023), and others
- **Connection to reinforcement learning**: NSGDHess is inspired by the SHARP/HAR-PG algorithms from RL (Fatkhullin et al., 2023), reflecting the deepening intersection between optimization theory and RL
- **Insights**: The concept of Hessian clipping may be valuable for robustness design in distributed optimization and federated learning

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First complete theory of second-order optimization under heavy-tailed noise; Hessian clipping is entirely new)
- Experimental Thoroughness: ⭐⭐⭐ (Strong theoretical contributions but limited to synthetic experiments)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, well-structured presentation, highly informative figures and tables)
- Value: ⭐⭐⭐⭐ (Provides a solid theoretical foundation for optimization under heavy-tailed noise)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[NeurIPS 2025\] perturbation bounds for low-rank inverse approximations under noise](perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)
- [\[AAAI 2026\] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](../../AAAI2026/optimization/fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)

</div>

<!-- RELATED:END -->
