---
title: >-
  [Paper Note] Automated Algorithm Design via Nevanlinna-Pick Interpolation
description: >-
  [NeurIPS 2025 (DynaFront Workshop)][Optimization][Automated algorithm design] This paper proposes an automated algorithm design framework based on Nevanlinna-Pick interpolation from frequency-domain robust control theory, targeting strongly convex optimization with equality constraints, and achieves an optimal trade-off between the number of matrix-vector multiplications and the convergence rate.
tags:
  - NeurIPS 2025 (DynaFront Workshop)
  - Optimization
  - Automated algorithm design
  - Nevanlinna-Pick interpolation
  - robust control
  - equality-constrained optimization
  - convergence rate
date: 2026-05-08
content_hash: ee2e232998f693ea
---

# Automated Algorithm Design via Nevanlinna-Pick Interpolation

**Conference**: NeurIPS 2025 (DynaFront Workshop)  
**arXiv**: [2509.21416](https://arxiv.org/abs/2509.21416)  
**Code**: None  
**Area**: Optimization & Control  
**Keywords**: Automated algorithm design, Nevanlinna-Pick interpolation, robust control, equality-constrained optimization, convergence rate

## TL;DR

This paper proposes an automated algorithm design framework based on Nevanlinna-Pick interpolation from frequency-domain robust control theory, targeting strongly convex optimization with equality constraints, and achieves an optimal trade-off between the number of matrix-vector multiplications and the convergence rate.

## Background & Motivation

Traditional optimization algorithm design typically follows a "design-then-analyze" paradigm: one first selects an appropriate algorithmic structure based on the optimality conditions of the problem, then performs convergence analysis. This approach has several fundamental drawbacks:

**Obscured performance limits**: The separation of design and analysis makes it difficult to systematically identify fundamental performance boundaries of algorithm classes.

**Non-systematic nature**: Deriving tight analytical inequalities is inherently difficult and highly problem-dependent.

**Difficulty in Lyapunov function design**: Understanding algorithmic behavior via Lyapunov stability theory requires deep problem-specific insight.

In recent years, frameworks based on frequency-domain techniques from robust control theory have emerged as powerful tools for automated algorithm synthesis. By integrating the design and analysis phases, such frameworks can identify fundamental performance limits. For instance, constructive proofs of complexity lower bounds for Nesterov's gradient method have been obtained via this framework.

The core problem considered in this paper is strongly convex optimization with equality constraints:
$$\min_x f(x) \quad \text{s.t.} \quad Ex = q$$
where $f$ is $m$-strongly convex with $L$-Lipschitz continuous gradient. Such problems arise broadly in federated learning, imaging, optimal transport, and fluid dynamics. A particularly important special case is decentralized optimization.

## Method

### Overall Architecture

The core idea is to reformulate optimization algorithm design as a controller synthesis problem, which is then solved via Nevanlinna-Pick interpolation theory. The framework proceeds through the following steps:

1. **Algorithm specification**: Define design objectives (linearity, explicitness, optimality, linear convergence).
2. **Z-transform representation**: Use the Z-transform to represent algorithmic iterations as transfer functions.
3. **Lur'e system modeling**: Model the algorithm as the feedback interconnection of a linear time-invariant system and a nonlinear block.
4. **Coordinate and loop transformations**: Simplify the system structure.
5. **Interpolation-based synthesis**: Construct transfer functions satisfying all design specifications via Nevanlinna-Pick interpolation.
6. **Algorithm implementation**: Convert the transfer function into an executable optimization iteration.

### Key Designs

**Algorithm representation**: Algorithms with linear updates can be expressed as:
$$z\mathcal{K}_0(z, W)\hat{x}(z) = \mathcal{K}_1(z, W)\hat{x}(z) + \mathcal{K}_2(z, W)\widehat{\nabla f}(z)$$

where $\hat{x}(z)$ and $\widehat{\nabla f}(z)$ are the Z-transforms of the iterate sequence and the gradient sequence, respectively. The three transfer functions govern:
- $\mathcal{K}_0$: the mapping between the optimization variable and the gradient input;
- $\mathcal{K}_1$: how past iterates are utilized;
- $\mathcal{K}_2$: how past gradients are utilized.

**Lur'e system perspective**: The algorithm is modeled as the feedback interconnection of a linear system defined by transfer function $\mathcal{H}(z, W)$ and a static nonlinear block $\Delta$:
$$\mathcal{H}(z, W) = (z\mathcal{K}_0 - \mathcal{K}_1)^{-1}\mathcal{K}_2, \quad \Delta(e) = \nabla f(e + x^\star) - \nabla f(x^\star)$$

**Analytic characterization of design specifications**:
- **Explicitness → Causality**: The transfer function is strictly proper, i.e., $\mathcal{H}(\infty, W) = 0$.
- **Linear convergence**: Convergence rate of at least $\rho$, i.e., $\|x_k - x^\star\| \leq M\|x_0 - x^\star\|\rho^k$.
- **Optimality**: The algorithm converges asymptotically to the unique solution of the problem.

### Core Method: Interpolated Gradient Method (I-GM)

Via the above framework, the authors synthesize the Interpolated Gradient Method (I-GM), characterized by:
- An explicit trade-off between the parameter $\ell$ (number of matrix-vector multiplications per iteration) and the convergence rate;
- An iteration complexity of $O(\max(\kappa_f, \kappa_E/\ell) \log(1/\epsilon))$.

Here $\kappa_f = L/m$ is the condition number of $f$, and $\kappa_E = \sigma_1/\sigma_r$ is the condition number of the constraint matrix.

## Key Experimental Results

### Main Results: Comparison with Existing Algorithms

| Algorithm | Matrix multiplications per iteration | Iteration complexity | Total matrix multiplication complexity |
|-----------|--------------------------------------|----------------------|----------------------------------------|
| GDA | 1 | $O(\kappa_f \kappa_E \log(1/\epsilon))$ | $O(\kappa_f \kappa_E \log(1/\epsilon))$ |
| PAPC | 1 | $O(\max(\kappa_f, \kappa_E) \log(1/\epsilon))$ | $O(\max(\kappa_f, \kappa_E) \log(1/\epsilon))$ |
| Optimal multi-loop method | $O(\kappa_E)$ | $O(\sqrt{\kappa_f} \log(1/\epsilon))$ | $O(\sqrt{\kappa_f} \kappa_E \log(1/\epsilon))$ |
| **I-GM (Ours)** | $\ell$ | $O(\max(\kappa_f, \kappa_E/\ell) \log(1/\epsilon))$ | $O(\ell \cdot \max(\kappa_f, \kappa_E/\ell) \log(1/\epsilon))$ |

### Ablation Study: Trade-offs of Parameter $\ell$

| Parameter $\ell$ | Per-iteration cost | Convergence rate | Applicable scenario |
|------------------|--------------------|------------------|---------------------|
| $\ell = 1$ | Lowest (single matrix multiplication) | $O(\max(\kappa_f, \kappa_E))$ | Communication-limited distributed optimization |
| $\ell = \sqrt{\kappa_E}$ | Moderate | $O(\max(\kappa_f, \sqrt{\kappa_E}))$ | Balanced computation and communication |
| $\ell = \kappa_E$ | Highest | $O(\kappa_f)$ | Computation-resource-abundant scenarios |

### Key Findings

1. I-GM achieves an optimal trade-off between iteration complexity and computational cost among single-loop algorithms.
2. When $\ell = 1$, I-GM reduces to a complexity comparable to PAPC.
3. Increasing $\ell$ effectively amortizes the impact of the constraint matrix condition number $\kappa_E$ on convergence.
4. In decentralized optimization settings, $\ell$ can be intuitively interpreted as the number of steps for local information propagation.

## Highlights & Insights

1. **Unified design and analysis**: By recasting algorithm design as controller synthesis, the framework naturally unifies the identification of performance limits with algorithm construction.
2. **Parameterized trade-off**: I-GM provides a tunable parameter $\ell$ that can be flexibly adjusted according to practical computational and communication constraints.
3. **Constructive proof**: The framework not only establishes the existence of performance lower bounds, but also constructively produces algorithms that attain them.
4. **Power of frequency-domain methods**: The work demonstrates a profound application of classical tools from robust control theory to algorithm design.

## Limitations & Future Work

1. The current framework is restricted to algorithms with linear updates and cannot handle nonlinear iteration formats such as proximal operators.
2. Only strongly convex objective functions are considered; extensions to general convex or nonconvex problems remain unclear.
3. Numerical experiments are limited, and the contribution is primarily theoretical.
4. The computation of Nevanlinna-Pick interpolation may face numerical stability challenges in high-dimensional settings.
5. The extension incorporating proximal operators is only briefly discussed and lacks a complete analysis.

## Related Work & Insights

- **Lur'e systems and optimization**: Modeling optimization algorithms as Lur'e systems is an important recent development; this paper builds on it to achieve automated design for constrained problems.
- **Performance Estimation Problem (PEP)**: Complementary to PEP methods based on semidefinite programming, the frequency-domain approach proposed here is advantageous when the algorithm structure is more explicit.
- **Distributed optimization**: The multi-loop structure of I-GM is naturally suited to decentralized computing settings.

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 4 |
| Theoretical Depth | 5 |
| Experimental Thoroughness | 2 |
| Writing Quality | 4 |
| Overall | 4 |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [\[AAAI 2026\] A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds](../../AAAI2026/optimization/a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)
- [\[ICLR 2026\] RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees](../../ICLR2026/optimization/rs-ort_a_reduced-space_branch-and-bound_algorithm_for_optimal_regression_trees.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](../../AAAI2026/optimization/pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)

<!-- RELATED:END -->
