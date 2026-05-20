---
title: >-
  [Paper Note] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds
description: >-
  [NeurIPS 2025][Optimization][Riemannian manifold] AdaRHD is the first adaptive algorithm for Riemannian bilevel optimization (RBO) that requires no prior knowledge of problem-specific parameters (strong convexity constan…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Riemannian manifold"
  - "bilevel optimization"
  - "adaptive step size"
  - "hypergradient"
  - "conjugate gradient"
date: 2026-05-08
content_hash: 774ef8948b2353ca
---

# An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds

**Conference**: NeurIPS 2025
**arXiv**: [2504.06042](https://arxiv.org/abs/2504.06042)  
**Code**: [https://github.com/RufengXiao/AdaRHD](https://github.com/RufengXiao/AdaRHD)  
**Area**: Riemannian Optimization / Bilevel Optimization
**Keywords**: Riemannian manifold, bilevel optimization, adaptive step size, hypergradient, conjugate gradient

## TL;DR
AdaRHD is the first adaptive algorithm for Riemannian bilevel optimization (RBO) that requires no prior knowledge of problem-specific parameters (strong convexity constants, Lipschitz bounds, manifold curvature). By adopting an inverse cumulative gradient norm strategy for adaptive step size selection and solving the lower-level problem, linear system, and upper-level update sequentially within a three-stage framework, AdaRHD achieves a convergence rate of $O(1/\epsilon)$ matching non-adaptive methods, while exhibiting substantially greater robustness to initial step size choices compared to RHGD.

## Background & Motivation

**Background**: Riemannian bilevel optimization (RBO) finds broad applications in machine learning, including hyperparameter optimization on SPD manifolds and robust PCA on Stiefel manifolds. Existing methods such as RHGD require prior knowledge of strong convexity constants, Lipschitz continuity bounds, and manifold curvature parameters to determine step sizes.

**Limitations of Prior Work**: These parameters are difficult to estimate in practice, particularly under non-Euclidean geometry. Incorrect step size choices cause RHGD to fail entirely—in experiments, step sizes of 5, 1, and 0.5 lead to divergence—whereas AdaRHD converges across a step size range of 0.2 to 20.

**Key Challenge**: While adaptive methods (e.g., AdaGrad) have been developed for bilevel optimization in Euclidean spaces, the geometric structures of Riemannian manifolds—geodesics, parallel transport, exponential maps—substantially complicate adaptive step size design.

**Goal**: Design a Riemannian bilevel optimization algorithm that requires no hyperparameter tuning while maintaining the same convergence rate as methods with known parameters.

**Key Insight**: Extend the AdaGrad-Norm strategy to three-level nested optimization on Riemannian manifolds.

**Core Idea**: Inverse cumulative gradient norm adaptive step sizes + three-stage pipeline (lower-level RGD → linear system CG → upper-level hypergradient descent) = parameter-free Riemannian bilevel optimization.

## Method

### Overall Architecture
Three stages: **Stage 1** solves the lower-level problem (geodesically strongly convex) via adaptive RGD; **Stage 2** solves the Hessian inverse-vector product (linear system) via CG/GD; **Stage 3** performs upper-level adaptive descent using the approximate hypergradient. Each stage adopts the inverse cumulative norm step size: $\eta_t = \eta_0 / \sqrt{\sum_{s=1}^t \|\nabla f_s\|^2}$

Two variants are proposed: AdaRHD-GD (gradient descent for the linear system, $O(1/\epsilon)$) and AdaRHD-CG (conjugate gradient, $O(\log(1/\epsilon))$ — superior Hessian-vector product complexity).

### Key Designs

1. **Adaptive Step Size Strategy (Inverse Cumulative Gradient Norm)**:

    - Function: Automatically adjusts step sizes based on historical gradients.
    - Mechanism: $\eta_t = \eta_0 / \sqrt{\sum_{s=1}^t \|g_s\|^2}$ — large gradients yield small steps (preventing oscillation); small gradients yield large steps (accelerating convergence). Stage 1 uses logarithmic scaling; Stage 2 uses exponential scaling.
    - Design Motivation: Eliminate dependence on the strong convexity constant $\mu$, Lipschitz constant $L$, and manifold curvature $\kappa$.

2. **Two Variants (AdaRHD-GD and AdaRHD-CG)**:

    - Function: Solve the linear system $H^{-1}v$ via different methods.
    - Mechanism: The GD variant solves the linear system via gradient descent with $O(1/\epsilon)$ complexity; the CG variant employs conjugate gradient with $O(\log(1/\epsilon))$ complexity (matching AdaGrad-Norm theory). The CG variant achieves $O(1/\epsilon)$ Hessian-vector product complexity, superior to the $O(1/\epsilon^2)$ of competing methods.
    - Design Motivation: The CG variant is superior both theoretically and empirically — converging faster with lower hypergradient estimation error.

3. **Retraction Map Support**:

    - Function: Replace the exponential map with retraction to reduce computational cost.
    - Mechanism: Algorithm 3 supports general retraction maps (not restricted to exponential maps) while maintaining the same $O(1/\epsilon)$ convergence rate.
    - Design Motivation: The exponential map is computationally expensive on certain manifolds (e.g., Stiefel manifold); retraction provides a practical alternative.

### Loss & Training
- Convergence rate: $O(1/\epsilon)$ to find an $\epsilon$-stationary point (matching non-adaptive methods).
- Assumption: The lower-level objective is geodesically strongly convex.

## Key Experimental Results

### Main Results

| Experiment | AdaRHD-CG | RHGD-50 | RHGD-20 |
|------|-----------|---------|---------|
| Matrix alignment (Stiefel-SPD) | Fastest convergence, lowest error | Slow | Slower |
| AFEW robustness (SPD network) | **85% validation accuracy**, shortest runtime | Step sizes 5/1/0.5 completely fail | Only step size 0.1 works |

### Ablation Study / Robustness Analysis

| Initial Step Size | AdaRHD | RHGD |
|----------|--------|------|
| 0.2 | ✓ Converges | — |
| 1.0 | ✓ Converges | ✗ Fails |
| 2.0 | ✓ Converges | — |
| 10.0 | ✓ Converges | — |
| 20.0 | ✓ Converges | — |
| RHGD 0.05 | — | ✓ Only working setting |

### Key Findings
- AdaRHD is highly robust to initial step size — converging across the range 0.2 to 20.0, while RHGD only works at 0.05.
- The CG variant exhibits lower hypergradient estimation error than the GD variant, yielding faster convergence.
- Lower standard deviation indicates that adaptive step sizes lead to more stable training.

## Highlights & Insights
- **First parameter-free Riemannian bilevel optimization algorithm**: Eliminating dependence on problem-specific parameters represents a significant practical advance.
- **Substantial robustness gap**: AdaRHD operates across a 100× range of step sizes, whereas RHGD requires precise tuning.
- **Superior second-order complexity of CG variant**: $O(1/\epsilon)$ vs. $O(1/\epsilon^2)$ Hessian-vector products.

## Limitations & Future Work
- Only addresses deterministic RBO with geodesically strongly convex lower-level objectives — non-strongly convex and PL-condition objectives are not covered.
- Incurs an additional $1/\epsilon$ factor in gradient/Hessian complexity compared to non-adaptive methods — the cost of adaptivity.
- Stochastic/mini-batch settings are not addressed, though large-scale practical problems typically rely on stochastic gradients.
- Validation is limited to small-to-medium scale problems — SPD and Stiefel manifold experiments involve dimensions $\leq 50$.
- Design of single-loop algorithms (more efficient) is left as future work.
- Hypergradient estimation error bounds may be looser on certain manifolds, necessitating manifold-specific analysis.
- No direct runtime comparison with adaptive bilevel methods in Euclidean space is provided.
- The impact of retraction map choice on convergence speed is not quantified.

## Related Work & Insights
- **vs. RHGD**: Requires precise problem-specific parameters; AdaRHD is fully adaptive.
- **vs. Euclidean AdaGrad**: AdaRHD extends the framework to manifolds and bilevel optimization.
- **vs. SUSTAIN/MA-SOBA**: Euclidean adaptive bilevel methods; AdaRHD generalizes these to Riemannian manifolds.
- **Transferability**: The inverse cumulative norm strategy can be applied to other Riemannian optimization problems beyond bilevel settings.
- **Practical significance**: Eliminates the tuning barrier — practitioners no longer need to estimate strong convexity constants or manifold curvature to run the algorithm.

## Rating
- Novelty: ⭐⭐⭐⭐ First adaptive Riemannian bilevel optimization algorithm with solid theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Two experiments (matrix alignment + AFEW); modest scale but robustness validation is thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with complete complexity analysis.
- Value: ⭐⭐⭐⭐ Makes Riemannian bilevel optimization more practically accessible by removing the primary barrier of parameter tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](problem-parameter-free_decentralized_bilevel_optimization.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization](set_smoothness_unlocks_clarke_hyper-stationarity_in_bilevel_optimization.md)
- [\[NeurIPS 2025\] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)

</div>

<!-- RELATED:END -->
