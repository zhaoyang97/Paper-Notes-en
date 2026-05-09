---
title: >-
  [Paper Note] Problem-Parameter-Free Decentralized Bilevel Optimization
description: >-
  [NeurIPS 2025][Optimization][Decentralized optimization] This paper proposes AdaSDBO, a fully parameter-free decentralized bilevel optimization algorithm that requires no prior knowledge of problem parameters. By employing adaptive step sizes based on cumulative gradient norms, it achieves a convergence rate of $\tilde{O}(1/\sqrt{T})$.
tags:
  - NeurIPS 2025
  - Optimization
  - Decentralized optimization
  - bilevel optimization
  - parameter-free
  - adaptive step size
  - single-loop
date: 2026-05-08
content_hash: 583cb735c9c45a9a
---

# Problem-Parameter-Free Decentralized Bilevel Optimization

**Conference**: NeurIPS 2025

**arXiv**: [2510.24288](https://arxiv.org/abs/2510.24288)

**Code**: None

**Area**: Optimization

**Keywords**: Decentralized optimization, bilevel optimization, parameter-free, adaptive step size, single-loop

## TL;DR

This paper proposes AdaSDBO, a fully parameter-free decentralized bilevel optimization algorithm that requires no prior knowledge of problem parameters. By employing adaptive step sizes based on cumulative gradient norms, it achieves a convergence rate of $\tilde{O}(1/\sqrt{T})$.

## Background & Motivation

Decentralized bilevel optimization (DBO) has important applications in large-scale machine learning, including meta-learning, hyperparameter optimization, and federated learning. The problem is formulated as:

$$\min_x f(x, y^*(x)) \quad \text{s.t.} \quad y^*(x) = \arg\min_y g(x, y)$$

where multiple nodes collaboratively optimize without a central server.

Key limitations of existing methods:

**Dependence on problem parameters**: Step sizes require knowledge of Lipschitz constants, strong convexity parameters, and communication graph spectral information.

**Difficulty in tuning**: These parameters are typically unavailable in practice, necessitating extensive manual tuning.

**Double-loop structure**: Many methods require inner and outer loops, leading to complex implementations.

## Method

### Overall Architecture

AdaSDBO is a **single-loop**, **fully parameter-free** algorithm that simultaneously updates upper-level variables, lower-level variables, and communication variables using adaptive step sizes.

### Key Designs

**1. Adaptive Step Size Mechanism**

Step sizes are adjusted automatically based on cumulative gradient norms, without requiring any problem parameters:
$$\eta_t^x = \frac{c}{\sqrt{\sum_{s=1}^{t} \|G_s^x\|^2}}$$
$$\eta_t^y = \frac{c}{\sqrt{\sum_{s=1}^{t} \|G_s^y\|^2}}$$

This follows the spirit of AdaGrad, adapted to the specific structure of bilevel optimization.

**2. Single-Loop Structure**

- Simultaneously updates the upper-level variable $x$, lower-level variable $y$, and consensus variables.
- Employs hypergradient approximation: $\hat{\nabla} F(x) = \nabla_x f - \nabla_{xy}^2 g \cdot (\nabla_{yy}^2 g)^{-1} \cdot \nabla_y f$
- Approximates the Hessian inverse via Neumann series or finite differences.

**3. Decentralized Communication**

- Nodes exchange information via a communication graph.
- Gradient tracking is used to achieve consensus.
- Adaptive step sizes are also applied to the communication steps.

### Loss & Training

- Upper-level update: $x_{t+1} = x_t - \eta_t^x \cdot \hat{\nabla}_x F(x_t, y_t)$
- Lower-level update: $y_{t+1} = y_t - \eta_t^y \cdot \nabla_y g(x_t, y_t)$
- Consensus update: Information aggregation across nodes via mixing matrix $W$.

## Key Experimental Results

### Main Results

Data hyper-cleaning task (test accuracy):

| Method | MNIST | CIFAR-10 | FashionMNIST |
|--------|-------|----------|--------------|
| DSBO (manually tuned) | 95.2% | 68.5% | 85.3% |
| MA-DSBO | 94.8% | 67.1% | 84.5% |
| BSMD | 94.5% | 66.8% | 83.9% |
| AdaSDBO (Ours) | **95.4%** | **68.3%** | **85.1%** |

Hyperparameter search task (final loss value):

| Method | Best Tuning | Random LR 1 | Random LR 2 | Random LR 3 |
|--------|-------------|-------------|-------------|-------------|
| DSBO | 0.032 | 0.156 | 0.089 | Diverged |
| MA-DSBO | 0.038 | 0.125 | 0.076 | 0.345 |
| AdaSDBO | **0.035** | **0.037** | **0.036** | **0.038** |

### Ablation Study

Robustness under different step size initializations (MNIST hyper-cleaning):

| Step Size Multiplier | DSBO | MA-DSBO | AdaSDBO |
|----------------------|------|---------|---------|
| 0.01x | 88.2% | 90.1% | 94.8% |
| 0.1x | 93.5% | 93.8% | 95.1% |
| 1x (optimal) | 95.2% | 94.8% | 95.4% |
| 10x | 91.1% | 92.5% | 95.2% |
| 100x | Diverged | 85.3% | 95.0% |

### Key Findings

1. AdaSDBO performs robustly across all step size configurations, whereas baseline methods are highly sensitive to step size selection.
2. The convergence rate $\tilde{O}(1/\sqrt{T})$ matches carefully tuned state-of-the-art methods, up to polylogarithmic factors.
3. Under varying network topologies (ring, random graph, complete graph), the adaptive step sizes adjust automatically.
4. The single-loop structure achieves 2–3× speedup in wall-clock time compared to double-loop methods.

## Highlights & Insights

- **Practical breakthrough**: Completely eliminates dependence on problem parameters, enabling true plug-and-play deployment.
- **Exceptional robustness**: Performance remains nearly unchanged across a 100× range of step size variation.
- **Complete theoretical guarantees**: $\tilde{O}(1/\sqrt{T})$ convergence rate matching well-tuned baselines.

## Limitations & Future Work

1. Theoretical analysis is restricted to the nonconvex-strongly-convex setting (nonconvex upper level, strongly convex lower level).
2. The polylogarithmic gap may be non-negligible in certain scenarios.
3. Advanced decentralized techniques such as communication compression have not been integrated.
4. Validation on large-scale neural network training remains insufficient.

## Related Work & Insights

- **AdaGrad**: A classical method for adaptive step sizes.
- **DSBO**: The standard algorithm for decentralized bilevel optimization.
- **D-SOBA, BSMD**: Other decentralized bilevel optimization methods.

## Rating

- ⭐ Novelty: 7/10 — Extending AdaGrad-style ideas to bilevel optimization is a natural yet valuable contribution.
- ⭐ Value: 9/10 — Eliminating the need for parameter tuning represents a significant practical convenience.
- ⭐ Writing Quality: 8/10 — Theoretical derivations are complete, and experiments are well-designed to highlight robustness.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization](set_smoothness_unlocks_clarke_hyper-stationarity_in_bilevel_optimization.md)
- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [\[NeurIPS 2025\] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)

<!-- RELATED:END -->
