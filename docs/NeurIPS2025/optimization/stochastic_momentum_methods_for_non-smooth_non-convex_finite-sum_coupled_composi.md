---
title: >-
  [Paper Note] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization
description: >-
  [NeurIPS2025][Optimization][FCCO] For non-smooth non-convex finite-sum coupled compositional optimization (FCCO), this paper proposes two stochastic momentum methods — SONEX (single-loop) and ALEXR2 (double-loop) — that improve the iteration complexity from $O(1/\epsilon^6)$ to $O(1/\epsilon^5)$ via outer Moreau envelope smoothing and nested smoothing techniques, and achieve the same optimal complexity for non-convex inequality-constrained optimization.
tags:
  - "NeurIPS2025"
  - "Optimization"
  - "FCCO"
  - "non-smooth non-convex optimization"
  - "stochastic momentum methods"
  - "Moreau envelope"
  - "constrained optimization"
  - "compositional optimization"
date: 2026-05-08
content_hash: 9db673a61b4d4f11
---

# Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization

**Conference**: NeurIPS2025
**arXiv**: [2506.02504](https://arxiv.org/abs/2506.02504)  
**Authors**: Xingyu Chen, Bokun Wang, Ming Yang, Qihang Lin, Tianbao Yang (Texas A&M, Iowa)
**Code**: To be confirmed  
**Area**: Optimization
**Keywords**: FCCO, non-smooth non-convex optimization, stochastic momentum methods, Moreau envelope, constrained optimization, compositional optimization

## TL;DR
For non-smooth non-convex finite-sum coupled compositional optimization (FCCO), this paper proposes two stochastic momentum methods — SONEX (single-loop) and ALEXR2 (double-loop) — that improve the iteration complexity from $O(1/\epsilon^6)$ to $O(1/\epsilon^5)$ via outer Moreau envelope smoothing and nested smoothing techniques, and achieve the same optimal complexity for non-convex inequality-constrained optimization.

## Background & Motivation

The finite-sum coupled compositional optimization (FCCO) problem takes the form:
$$\min_{\mathbf{w}} F(\mathbf{w}) := \frac{1}{n}\sum_{i=1}^n f_i(g_i(\mathbf{w}))$$

where $f_i$ are outer functions (possibly non-smooth) and $g_i$ are inner functions (stochastically estimated). This formulation arises broadly in X-risk optimization (AUC, contrastive losses), group distributionally robust optimization (GDRO), and non-convex constrained optimization.

Prior methods suffer from two key limitations:
- **High complexity**: The previous state-of-the-art method SONX requires $O(1/\epsilon^6)$ iterations under the mean Lipschitz continuity assumption on the stochastic inner function mean.
- **Incompatibility with deep learning**: SONX relies on vanilla SGD updates and lacks momentum mechanisms, making it unsuitable for training deep neural networks.

This paper aims to address both issues simultaneously: designing stochastic momentum methods with provable convergence guarantees while reducing complexity by an order of magnitude.

## Method

### Mechanism: Outer Smoothing + Nested Smoothing

**Outer Smoothing**: The non-smooth outer function $f_i$ is smoothed via the Moreau envelope:
$$f_{i,\lambda}(\mathbf{u}) := \min_{\mathbf{v}} f_i(\mathbf{v}) + \frac{1}{2\lambda}\|\mathbf{u} - \mathbf{v}\|^2$$

This transforms the original problem into a smooth compositional objective $\min_{\mathbf{w}} F_\lambda(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n f_{i,\lambda}(g_i(\mathbf{w}))$. When $\lambda = \epsilon / C_f$, an $\epsilon$-stationary point of $F_\lambda$ is an approximate $\epsilon$-stationary solution of the original problem (Theorem 4.2).

**Nested Smoothing**: When the inner function $g_i$ is also non-smooth (weakly convex), a second Moreau envelope is applied to $F_\lambda$, yielding the doubly smoothed objective $\min_{\mathbf{w}} F_{\lambda,\nu}(\mathbf{w})$, which is amenable to momentum-based methods.

### Algorithm 1: SONEX (Single-Loop, Smooth Inner Functions)

**Applicable when**: $f_i$ is weakly convex and non-smooth or convex; $g_i$ is smooth.

Key steps:
1. **MSVR tracking**: Maintain $n$ auxiliary sequences $u_{i,t}$ to track $g_i(\mathbf{w}_t)$, updated in a coordinate-wise fashion (only sampled indices $i \in \mathcal{B}_1^t$ are updated).
2. **Gradient estimation**: $G_t = \frac{1}{|\mathcal{B}_1^t|}\sum_{i \in \mathcal{B}_1^t} \nabla f_{i,\lambda}(u_{t,i}) \nabla g_i(\mathbf{w}_t, \mathcal{B}_{i,2}^t)$
3. **Momentum update**: $\mathbf{v}_{t+1} = (1-\beta)\mathbf{v}_t + \beta G_t$, $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}$ (or Adam update)

Complexity: $O\left(\frac{n}{B_1\sqrt{B_2}} \epsilon^{-5}\right)$

### Algorithm 2: ALEXR2 (Double-Loop, Smooth or Weakly Convex Inner Functions)

**Applicable when**: $f_i$ is convex; $g_i$ is smooth or weakly convex (the latter requiring $f_i$ to be monotonically non-decreasing).

$F_\lambda$ is reformulated as a minimax problem, then doubly smoothed via nested Moreau envelopes to obtain $F_{\lambda,\nu}$:

Key steps:
1. **Outer loop**: Momentum update of $\mathbf{w}$ using the approximate gradient $G_t = \frac{\beta}{\nu}(\mathbf{w}_t - \mathbf{z}_{t,K_t})$.
2. **Inner loop**: Run ALEXR to solve a strongly convex–strongly concave minimax subproblem, yielding an approximate proximal mapping solution $\hat{\mathbf{z}}_t$.
3. Supports both Adam and momentum updates.

Complexity: $\tilde{O}\left(\frac{n}{B_1 B_2 \epsilon^5} + \frac{1}{\epsilon^5}\right)$

### Application to Constrained Optimization

For inequality-constrained optimization $\min g_0(\mathbf{w})$ s.t. $g_i(\mathbf{w}) \leq 0$, a smooth hinge penalty function is employed:
$$\Phi_\lambda(\mathbf{w}) = g_0(\mathbf{w}) + \frac{1}{m}\sum_{i=1}^m f_\lambda(g_i(\mathbf{w}))$$

where $f_\lambda$ is the Moreau-smoothed version of the hinge function $\rho[\cdot]_+$. Optimizing via SONEX/ALEXR2 achieves $O(1/\epsilon^5)$ complexity for finding an approximate $\epsilon$-KKT solution, improving the dependence on $\delta$ from $O(\delta^{-6})$ to $O(\delta^{-4})$.

## Key Experimental Results

### Experiment 1: Group Distributionally Robust Optimization (GDRO with CVaR)

Evaluated on Camelyon17 (30 groups, DenseNet121), Amazon (1252 groups, DistilBERT), and CelebA (16 groups, ResNet50) with CVaR ratio $r=0.15$:

| Method | Update Type | Camelyon17 | Amazon | CelebA Loss | CelebA Test Acc. |
|--------|-------------|------------|--------|-------------|-----------------|
| OOA | SGD | Slower convergence | Slower convergence | Slower convergence | Lower |
| SONX | SGD | Moderate | Moderate | Moderate | Moderate |
| **SONEX** | **Momentum/Adam** | **Fastest convergence** | **Fastest convergence** | **Fastest convergence** | **Highest** |

SONEX achieves the fastest training loss reduction across all datasets, with the best test accuracy on CelebA. The advantage is most pronounced on Amazon (1252 groups) with Adam updates.

### Experiment 2: AUC Maximization + ROC Fairness Constraints

Evaluated on Adult and COMPAS datasets with 14 fairness constraints (FPR/TPR gap tolerance $\kappa=0.005$, thresholds $\Gamma=\{-3,...,3\}$) using a 2-hidden-layer network:

| Method | Penalty | Adult AUC | COMPAS AUC | Constraint Satisfaction |
|--------|---------|-----------|------------|------------------------|
| SOX | Squared hinge | Lower | Lower | Acceptable |
| SONX | Hinge | Moderate | Moderate | Acceptable |
| ICPPAC | Double-loop | Moderate | Moderate | Acceptable |
| **ALEXR2** | **Smooth hinge** | **Highest** | **Highest** | **Comparable** |

ALEXR2 achieves superior AUC maximization while maintaining comparable constraint satisfaction.

### Experiment 3: Continual Learning + Non-forgetting Constraints

Fine-tuning CLIP on BDD100K (target tasks: foggy/cloudy classification; constraints: no forgetting on other weather categories):
- SONEX achieves significantly higher Target $\Delta$ACC than SOX (squared hinge).
- Constraint violations are comparable across methods.
- SONX (SGD) completely fails on the Transformer architecture and is unable to train.

## Highlights & Insights

- **Theoretical contribution**: Improves the optimal complexity for non-smooth non-convex FCCO from $O(\epsilon^{-6})$ to $O(\epsilon^{-5})$; this is the first stochastic momentum method for this problem class.
- **Dual smoothing technique**: Outer Moreau envelope smoothing combined with nested smoothing elegantly handles the double non-smoothness of both outer and inner functions.
- **Practical applicability**: Supports Adam updates and is directly applicable to deep learning; effective on Transformer/CLIP architectures where the SGD-based SONX fails.
- **New results for constrained optimization**: Improvements in both complexity and dependence on the regularization constant $\delta$.
- **Unified framework**: SONEX and ALEXR2 respectively cover smooth and weakly convex inner function settings, applicable to GDRO, AUC fairness, continual learning, and beyond.

## Limitations & Future Work

- The single-loop SONEX has a weaker dependence on the inner batch size $B_2$ (i.e., $1/\sqrt{B_2}$) compared to the double-loop ALEXR2 (i.e., $1/B_2$); the authors explicitly identify this as a direction for improvement.
- SONEX requires the mean Lipschitz continuity (MLC0) assumption, while ALEXR2 requires convexity of the outer function, each with its own restrictions.
- Assumption 4.3 (lower bound on the minimum singular value of the inner function Jacobian) is empirically verified but does not hold universally.
- Experimental scale is limited: validation is confined to relatively small classification tasks, without large-scale deep learning benchmark experiments.
- Hyperparameter settings rely on theoretically derived constants, which may require additional tuning in practice.

## Related Work & Insights

- **FCCO methods**: SOX, MSVR/STORM, and SONX either require smooth $f_i$ or are restricted to SGD updates; this paper is the first to handle non-smoothness with momentum.
- **Smoothing techniques**: Nesterov smoothing and Moreau envelopes are well-established for weakly convex problems; this paper extends them to the coupled structure of FCCO.
- **Non-convex constrained optimization**: OSS/ICPPAC achieve $O(\epsilon^{-6})$, SSG achieves $O(\epsilon^{-8})$, and Li et al. achieve $O(\epsilon^{-7})$; the proposed $O(\epsilon^{-5})$ is optimal in both smooth and weakly convex settings.
- **X-risk optimization**: Objectives such as AUROC and contrastive losses can be modeled as FCCO, and the proposed methods are directly applicable.

## Rating
- Novelty: ⭐⭐⭐⭐ — First stochastic momentum method for non-smooth FCCO; the outer smoothing + nested smoothing design is clear and effective.
- Experimental Thoroughness: ⭐⭐⭐ — Covers GDRO, fairness constraints, and continual learning, but experiments are limited in scale and baseline coverage.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous theoretical derivations with a well-organized notation system; however, the high content density poses a significant barrier for readers outside the optimization community.
- Value: ⭐⭐⭐⭐ — The complexity improvement carries solid theoretical significance, and the introduction of momentum methods directly benefits practical deep learning applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization](../../ICML2025/optimization/a_near-optimal_single-loop_stochastic_algorithm_for_convex_finite-sum_coupled_co.md)
- [\[NeurIPS 2025\] Non-Stationary Bandit Convex Optimization: A Comprehensive Study](non-stationary_bandit_convex_optimization_a_comprehensive_study.md)
- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[NeurIPS 2025\] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)

</div>

<!-- RELATED:END -->
