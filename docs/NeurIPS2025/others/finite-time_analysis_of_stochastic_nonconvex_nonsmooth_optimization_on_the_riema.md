---
title: >-
  [Paper Note] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds
description: >-
  [NeurIPS 2025][Riemannian optimization] This paper proposes the Riemannian Online to NonConvex (RO2NC) algorithm and its zeroth-order variant ZO-RO2NC, establishing for the first time a finite-time sample complexity guarantee of $O(\delta^{-1}\epsilon^{-3})$ for fully nonsmooth nonconvex stochastic optimization on Riemannian manifolds, matching the optimal result in Euclidean space.
tags:
  - NeurIPS 2025
  - Riemannian optimization
  - nonsmooth nonconvex
  - Goldstein stationarity
  - zeroth-order optimization
  - finite-time analysis
date: 2026-05-08
content_hash: 48c6833b0ccb325e
---

# Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds

**Conference**: NeurIPS 2025
**arXiv**: [2510.21468](https://arxiv.org/abs/2510.21468)
**Code**: None
**Area**: Other
**Keywords**: Riemannian optimization, nonsmooth nonconvex, Goldstein stationarity, zeroth-order optimization, finite-time analysis

## TL;DR
This paper proposes the Riemannian Online to NonConvex (RO2NC) algorithm and its zeroth-order variant ZO-RO2NC, establishing for the first time a finite-time sample complexity guarantee of $O(\delta^{-1}\epsilon^{-3})$ for fully nonsmooth nonconvex stochastic optimization on Riemannian manifolds, matching the optimal result in Euclidean space.

## Background & Motivation

**Background**: Riemannian optimization is widely applied in deep learning (orthogonality constraints), PCA, dictionary learning, low-rank matrix completion, and other problems whose search spaces possess manifold structure. Existing Riemannian optimization algorithms primarily target smooth objective functions, including gradient descent, projection-free methods, and accelerated methods.

**Limitations of Prior Work**:
   - $\epsilon$-stationary points of nonsmooth nonconvex functions are intractable (NP-hard), necessitating surrogate stationarity criteria.
   - Goldstein stationarity in Euclidean space has only been studied recently (2020–), where the optimal $O(\delta^{-1}\epsilon^{-3})$ rate has been achieved.
   - In the Riemannian setting, finite-time analysis of nonsmooth nonconvex optimization is entirely absent—only asymptotic convergence results exist.

**Key Challenge**: Manifold geometry (distinct tangent spaces at different points, distortion introduced by parallel transport, nonlinearity of retractions) prevents the direct transfer of Euclidean analysis tools.

**Key Insight**: Adapt the Euclidean O2NC algorithm (based on the online-to-nonconvex conversion) to Riemannian manifolds by using retractions to ensure feasibility, and employing parallel transport/projection to handle vector operations across different tangent spaces.

**Core Idea**: By carefully choosing the benchmark action $u_t$ (transporting gradients to a common tangent space via parallel transport) and analyzing error terms introduced by manifold curvature, the paper extends the optimal Euclidean complexity to the Riemannian setting.

## Method

### Overall Architecture
RO2NC adopts a double-loop structure: outer epochs $k=1,\ldots,K$ and inner iterations $t=0,\ldots,T-1$. Within each epoch, an online learning subroutine generates actions $\Delta_t$, and the iterate is updated as $x_{t+1} = \text{Retr}_{x_t}(\Delta_t)$. Gradients are computed at a stochastic intermediate point $w_t = \text{Retr}_{x_t}(s_t\Delta_t)$ (with $s_t \sim \text{unif}[0,1]$) and used as feedback. The output is an intermediate point from a randomly selected epoch.

### Key Designs

1. **Riemannian Generalization of Goldstein Stationarity**:

    - Function: Defines the notion of a $(\delta,\epsilon)$-stationary point on Riemannian manifolds.
    - Mechanism: The Riemannian $\delta$-subdifferential is defined as $\partial_\delta f(x) := \text{cl conv}\{P_{y,x}^g(\partial f(y)): y \in \text{cl } B(x,\delta)\}$, requiring parallel transport to map subdifferentials from different tangent spaces to a common one. A point $x$ is a $(\delta,\epsilon)$-stationary point if $\|\text{grad } f(x)\|_\delta \leq \epsilon$.
    - Design Motivation: Parallel transport is an isometry (preserving vector norms), making it suitable for defining distance-based notions; projection does not preserve norms and is therefore less appropriate.

2. **Parallel Transport Version of RO2NC**:

    - Function: Updates actions $\Delta_t$ and feedback gradients $g_t$ using parallel transport.
    - Mechanism: $\Delta_{t+1} = \text{clip}(P_{x_t,x_{t+1}}^g(\Delta_t) - \eta g_t')$, where $g_t' = P_{w_t,x_{t+1}}^g(g_t)$ is the parallel-transported gradient. The clip operator restricts actions to the tangent space ball $\mathbb{B}_{T_{x_{t+1}}\mathcal{M}}(D)$.
    - Design Motivation: Parallel transport enables reuse of regret bounds from online optimization. The key innovation lies in the benchmark action choice: $u_t = \mathcal{P}_{S_t}^s\big(-D \frac{\sum_\tau (\mathcal{P}_{S_{\tau+1}}^s)^{-1} \circ P_{w_\tau,x_{\tau+1}}^g(\nabla_\tau)}{\|\cdot\|}\big)$, transporting all gradients to the common tangent space $T_{x_0}\mathcal{M}$.

3. **Projection Version of RO2NC**:

    - Function: Replaces parallel transport with the computationally more efficient projection.
    - Mechanism: $\Delta_{t+1} = \text{Proj}_{T_{x_{t+1}}\mathcal{M}}(\Delta_t - \eta g_t)$, computed in the ambient space and then projected back to the tangent space.
    - Design Motivation: Projection is more efficient in practice (no geodesic computation required). Although isometry is lost, the distortion is controlled via Lemma 2.8: $\|P_{0,t}^\gamma(v) - \text{Proj}(v)\| \leq C\|v\| \cdot \text{length}(\gamma)$.

4. **Zeroth-Order ZO-RO2NC**:

    - Function: Estimates gradients via function value queries when gradients are unavailable.
    - Mechanism: The Riemannian gradient estimator is $g_\delta(x) = \frac{d}{2\delta}(F(\text{Exp}_x(\delta u), \nu) - F(\text{Exp}_x(-\delta u), \nu))u$, where $u$ is sampled uniformly from the unit ball in the tangent space. An auxiliary function $h_\delta(x) = \int f \circ \text{Exp}_x(u)\, dp_x(u)$ is introduced to relate the gradient estimator to the Goldstein subdifferential.
    - Design Motivation: Sampling in the tangent space (rather than on the manifold) avoids computing manifold volume measures. Lemma 4.1 is used to control the distance between the gradient estimator and the subdifferential set.

### Loss & Training
- Step size $\eta = D/G\sqrt{T}$; clipping parameter $D = \delta/T$.
- $K$ outer epochs, $T$ inner iterations per epoch, total samples $N = KT$.
- Output is the randomly selected intermediate point $\bar{w}_k = w_{k,\lfloor T/2 \rfloor}$.

## Key Experimental Results

### Main Results — Sparse PCA on the Sphere $\mathbb{S}^{n-1}$

Objective: $\min_{x \in \mathbb{S}^{n-1}} \{-x^\top A x + \mu \|x\|_1\}$

| Setting | Algorithm | Convergence |
|---------|-----------|-------------|
| First-order + parallel transport | RO2NC (PT) | Gradient norm decreases monotonically across epochs |
| First-order + projection | RO2NC (Proj) | Comparable performance to the parallel transport version |
| Zeroth-order | ZO-RO2NC | Converges but slower than first-order; more sensitive to hyperparameters |

### Theoretical Complexity Comparison

| Reference | Method | Setting | Objective | Convergence Rate |
|-----------|--------|---------|-----------|-----------------|
| grohs2016 | Subgradient | Deterministic | Nonsmooth | Asymptotic |
| chen2024 | Proximal gradient | Deterministic | Composite (Stiefel) | $O(\epsilon^{-2})$ |
| wang2022 | SPIDER | Stochastic | Composite (Stiefel) | $O(\epsilon^{-3})$ |
| **Ours** | **RO2NC** | **Stochastic** | **Fully nonsmooth** | $O(\delta^{-1}\epsilon^{-3})$ |

### Key Findings
- The projection version achieves the same complexity $O(\delta^{-1}\epsilon^{-3})$ as the parallel transport version, with comparable empirical performance.
- The zeroth-order version maintains the same complexity order but with larger constants and greater sensitivity to hyperparameters.
- Manifold curvature enters the convergence bound through associated constants but does not affect the rate order.

## Highlights & Insights
- **First finite-time guarantee for fully nonsmooth Riemannian optimization**: Prior work on nonsmooth nonconvex Riemannian optimization provided only asymptotic results; this paper fills an important gap.
- **Matches Euclidean optimality**: The $O(\delta^{-1}\epsilon^{-3})$ complexity matches the optimal Euclidean result of Cutkosky & Mehta.
- **Clever benchmark action construction**: Time-varying benchmarks $u_t$ are constructed by composing parallel transport maps, enabling reuse of online optimization regret analysis.
- **Tangent-space sampling for zeroth-order estimation**: Avoids manifold volume computation, substantially reducing implementation complexity.

## Limitations & Future Work
- The Goldstein stationarity definition relies on the choice of parallel transport; other transport maps may yield different structural properties.
- Experiments are conducted only on the sphere; validation on more complex manifolds such as Stiefel or Grassmann manifolds is lacking.
- Deeper analysis of curvature effects and the design of algorithms that adaptively exploit curvature information remain open problems.
- The theory requires retraction curves to satisfy first- and second-order derivative bounds, excluding certain non-standard retractions.

## Related Work & Insights
- **vs. O2NC (Cutkosky & Mehta)**: The Euclidean counterpart; this paper is its Riemannian extension, additionally addressing tangent space inconsistency and curvature-induced distortion.
- **vs. composite objective methods**: Chen, Wang, Li, Deng et al. handle $f(x)+h(x)$ with smooth $f$; this paper addresses the fully nonsmooth case.
- **vs. zeroth-order Riemannian methods**: The zeroth-order method of Li et al. handles only smooth objectives; this paper is the first to address the nonsmooth setting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First finite-time analysis for fully nonsmooth nonconvex Riemannian optimization; outstanding theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ Only a single experiment (sparse PCA on the sphere); limited empirical validation.
- Writing Quality: ⭐⭐⭐⭐ Rigorous technical analysis; core innovations (benchmark construction, curvature control) are clearly presented.
- Value: ⭐⭐⭐⭐⭐ Fills an important theoretical gap in nonsmooth Riemannian optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](semi-infinite_nonconvex_constrained_min-max_optimization.md)
- [\[NeurIPS 2025\] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[ICLR 2026\] Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry](../../ICLR2026/others/fast_and_stable_riemannian_metrics_on_spd_manifolds_via_cholesky_product_geometr.md)
- [\[NeurIPS 2025\] A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation](a_theoretical_framework_for_grokking_interpolation_followed_by_riemannian_norm_m.md)

</div>

<!-- RELATED:END -->
