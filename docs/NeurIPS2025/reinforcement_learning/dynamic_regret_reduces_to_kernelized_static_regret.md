---
title: >-
  [Paper Note] Dynamic Regret Reduces to Kernelized Static Regret
description: >-
  [NeurIPS 2025][Reinforcement Learning][dynamic regret] This paper reformulates dynamic regret minimization as a static regret problem in a reproducing kernel Hilbert space (RKHS), achieving the optimal path-length-dependent bound $\widetilde{\mathcal{O}}(\sqrt{MP_TT})$ via carefully designed shift-invariant kernels, without requiring prior knowledge of the time horizon.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - dynamic regret
  - online convex optimization
  - RKHS
  - kernel methods
  - path-length
date: 2026-05-08
content_hash: a8a168ec07320dcb
---

# Dynamic Regret Reduces to Kernelized Static Regret

**Conference**: NeurIPS 2025
**arXiv**: [2507.05478](https://arxiv.org/abs/2507.05478)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: dynamic regret, online convex optimization, RKHS, kernel methods, path-length

## TL;DR
This paper reformulates dynamic regret minimization as a static regret problem in a reproducing kernel Hilbert space (RKHS), achieving the optimal path-length-dependent bound $\widetilde{\mathcal{O}}(\sqrt{MP_TT})$ via carefully designed shift-invariant kernels, without requiring prior knowledge of the time horizon.

## Background & Motivation

**Background**: In online convex optimization (OCO), a learner selects $w_t \in \mathcal{W}$ at each round, and the environment reveals a convex loss $\ell_t$. The classical objective is to minimize **static regret** relative to a fixed comparator. Under non-stationary environments, the goal shifts to minimizing **dynamic regret** relative to a time-varying comparator sequence $u_1,\dots,u_T$.

**Limitations of Prior Work**: Existing dynamic regret algorithms face three challenges: (a) achieving the optimal $\sqrt{P_T}$ dependence requires prior knowledge of the time horizon $T$, or relies on the doubling trick, which performs poorly in practice; (b) existing dynamic-to-static reductions apply only to linear losses; (c) the curvature of the loss function cannot be exploited to obtain tighter bounds.

**Key Challenge**: The approach of stacking comparator sequences into high-dimensional vectors (Jacobsen & Orabona 2024) is limited by finite-dimensional feature spaces and the linear loss assumption, making it ill-suited for infinite-dimensional designs and nonlinear losses.

**Goal**: (a) Achieve optimal $\sqrt{P_T}$ dependence without prior knowledge of $T$; (b) generalize the reduction to arbitrary convex losses; (c) exploit loss curvature to obtain bounds of the form $\mathcal{O}(\|u\|_{\mathcal{H}}^2 + d_{\text{eff}}(\lambda)\ln T)$.

**Key Insight**: The key observation is that competing against an arbitrary comparator sequence is equivalent to competing against a fixed function $u(\cdot)$ satisfying $u(t) = u_t$. Embedding this function into an RKHS reduces dynamic regret to static regret in function space.

**Core Idea**: Dynamic regret equals kernelized static regret in an RKHS; the relationship between the RKHS norm and path length is controlled by designing an appropriate kernel function.

## Method

### Overall Architecture
The input is a sequence of online convex losses $\ell_1,\dots,\ell_T$, and the output is the learner's decision $w_t$ at each round. The core pipeline is: (1) select an RKHS $\mathcal{H}$ and its feature map $\phi$; (2) transform the original loss $\ell_t$ into an auxiliary loss $\tilde{\ell}_t(W) = \ell_t(W\phi(t))$ over the operator space $L(\mathcal{H}, \mathcal{W})$; (3) run any static regret algorithm $\mathcal{A}$ over the operator space; (4) recover the decision each round via $w_t = W_t\phi(t)$.

### Key Designs

1. **Dynamic-to-Static Equivalence Theorem (Theorem 1)**:

    - Function: Proves that dynamic regret and kernelized static regret are strictly equal.
    - Mechanism: For any operator $U$ satisfying $u_t = U\phi(t)$, the identity $R_T(u_1,\dots,u_T) = \sum_t (\tilde{\ell}_t(W_t) - \tilde{\ell}_t(U)) = \tilde{R}_T(U)$ holds exactly — as an identity rather than an upper bound — for arbitrary losses.
    - Design Motivation: Prior reductions apply only to linear losses (as they require decomposing gradients as $g_t \otimes \phi(t)$). This paper's reduction operates directly on the loss functions, preserving curvature information.

2. **Kernelized FTRL and the Kernel Trick (Example 1)**:

    - Function: Demonstrates efficient computation in infinite-dimensional spaces.
    - Mechanism: By the reproducing property, the FTRL decision admits the representation $w_t = -\frac{(\Psi_t^*)'}{\|\theta_t\|_{\text{HS}}} \sum_{s=1}^{t-1} k(s,t)g_s$, requiring only kernel evaluations $k(s,t)$ without explicitly computing $\phi(t)$.
    - Design Motivation: Although the RKHS is infinite-dimensional, the kernel trick renders the algorithm computationally tractable.

3. **Carefully Designed Shift-Invariant Kernel (Proposition 2)**:

    - Function: Constructs a spectral density $Q(\omega)$ such that the RKHS norm is proportional to path length.
    - Mechanism: The target is $Q(\omega) \approx 1/|\omega|$ (since by Parseval's identity $\|u\|_{\mathcal{H}}^2 \approx \sup_t|u(t)| \cdot \|\nabla u\|_{L^1}$), but $1/|\omega|$ is not integrable. A specific integrable spectral density $Q(\omega)$ is obtained by introducing a small regularization.
    - Key Property: $k(t,t) \leq 8\pi^2$ holds for all $t$ (independent of $T$), and $\|f\|_{\mathcal{H}}^2 = \widetilde{\mathcal{O}}(\|\nabla f\|_{L^1}\|f - \nabla^2 f\|_{L^\infty})$.
    - Design Motivation: Standard kernels (e.g., Gaussian, Matérn) yield suboptimal bounds because their RKHS norms contain a term $\|u\|_{L^2}^2 = \Omega(T)$; spline kernels satisfy $\|u\|_{\mathcal{H}}^2 = \|\nabla u\|_{L^2}^2$ but have $k(t,t) = t$, which is also suboptimal.

4. **Bridging Continuous and Discrete Path Lengths (Theorem 4)**:

    - Function: Proves the existence of an RKHS function $u$ satisfying $u(t)=u_t$ with $\|\nabla u\|_{L^1} = \mathcal{O}(P_T)$.
    - Mechanism: Constructs a compactly supported smooth interpolant, using the condition that the RKHS contains compactly supported functions with bounded derivatives.
    - Design Motivation: Continuous and discrete path lengths can differ substantially (the function may oscillate wildly between interpolation points); it is therefore necessary to establish the existence of a "well-behaved" interpolant.

### Loss & Training
For linear losses: a parameter-free algorithm is applied, and combined with the designed kernel, directly yields the optimal bound $\widetilde{\mathcal{O}}(\sqrt{(M^2+MP_T)\sum_t\|g_t\|^2})$.

For curved losses:
- **Exp-concave losses**: The reduction preserves exp-concavity, enabling direct application of Kernelized ONS (KONS), yielding $R_T = \mathcal{O}(\lambda\|U\|_{\text{HS}}^2 + d_{\text{eff}}(\lambda)\ln T / \beta)$.
- **Strongly convex losses**: A weighted norm based on the feature covariance matrix yields an analogous logarithmic bound.
- **Online linear regression**: Recovers the standard bound of the kernelized Vovk-Azoury-Warmuth predictor.

## Key Experimental Results

### Comparison of Main Theoretical Results

| Loss Type | Best Prior Bound | This Paper's Bound | Improvement |
|-----------|-----------------|-------------------|-------------|
| Linear (bounded domain) | $\mathcal{O}(\sqrt{DP_TT})$, requires $T$ | $\widetilde{\mathcal{O}}(\sqrt{(M^2+MP_T)T})$, no $T$ required | First optimal bound without prior $T$ |
| Linear (scale-free) | No prior result | $\widetilde{\mathcal{O}}(\sqrt{(M^2+MP_T)\sum_t\|g_t\|^2})$ | First scale-free dynamic regret bound |
| Exp-concave | No dynamic regret result | $\mathcal{O}(\lambda\|U\|_{\text{HS}}^2 + d_{\text{eff}}(\lambda)\ln T)$ | First logarithmic bound exploiting curvature |
| Strongly convex | No dynamic regret result | Analogous logarithmic bound | Same as above |

### Comparison with Prior Reductions

| Property | Jacobsen & Orabona (2024) | This Paper |
|----------|--------------------------|------------|
| Applicable losses | Linear only | Arbitrary convex |
| Feature dimensionality | Finite ($\mathbb{R}^{dT}$) | Infinite-dimensional RKHS |
| Requires prior $T$ | Yes | No |
| Scale-free | No | Yes |
| Exploits loss curvature | No | Yes |
| Direction-adaptive | No | Yes |

### Key Findings
- Selecting the Dirac kernel exactly recovers the reduction of Jacobsen & Orabona, showing their approach is a special case of the proposed framework.
- The design of $Q(\omega)$ is the technical crux: standard kernels (Gaussian, Matérn) yield suboptimal bounds, necessitating a non-standard spectral density that is "close to $1/|\omega|$ but integrable."
- Direction adaptivity guarantees $\widetilde{\mathcal{O}}(\sqrt{d_{\text{eff}}(\lambda)(\|u\|_{\mathcal{H}^d}^2 + \sum_t \langle g_t, u_t \rangle^2)})$, which can be tighter when gradients are small in certain directions.

## Highlights & Insights
- **Elegance of the Perspective Shift**: Viewing the comparator sequence as samples of a function is a natural observation that nevertheless unlocks the entire RKHS toolbox — a "lift-and-solve" strategy in which the problem becomes simpler in a higher-dimensional space.
- **Horizon Independence**: This is the first dynamic regret algorithm to achieve optimal $\sqrt{P_T}$ dependence without prior knowledge of $T$, avoiding the practical performance loss associated with the doubling trick.
- **Unified Framework**: A single framework handles linear, exp-concave, and strongly convex losses simultaneously, yielding (near-)optimal guarantees in each case — a unification that suggests a deep connection between dynamic regret and kernel methods.
- The specific construction of $Q(\omega)$ may inspire analogous trade-off designs in signal processing and nonparametric statistics.

## Limitations & Future Work
- **Computational Complexity**: The naive implementation requires $\mathcal{O}(t)$ time and memory; kernel approximation methods could accelerate this, but no empirical evaluation is provided.
- **Purely Theoretical Contribution**: The paper contains no experiments, leaving performance in practical non-stationary online learning settings unclear.
- **Logarithmic Factors**: All bounds hide polylogarithmic factors; it is an open question whether these can be removed.
- **Automated Kernel Selection**: The current kernel design relies on prior knowledge of problem structure; automating the selection of the optimal kernel remains an open problem.

## Related Work & Insights
- **vs. Zhang et al. (2023, signal processing approach)**: They stack comparator sequences into high-dimensional signals and apply dictionary decomposition, whereas this paper embeds them into an RKHS. The proposed approach is more general (not restricted to linear losses), though their method may be more computationally efficient.
- **vs. Jacobsen & Orabona (2024)**: Their method is a special case of the proposed framework (Dirac kernel), but handles only linear losses and finite-dimensional features.
- **vs. Online Newton Step**: The RKHS reduction is naturally compatible with ONS, yielding the first logarithmic dynamic regret bound.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The perspective shift is highly elegant, entirely recasting dynamic regret as a kernel methods problem.
- Experimental Thoroughness: ⭐⭐ — A purely theoretical paper with no empirical validation.
- Writing Quality: ⭐⭐⭐⭐ — Mathematically rigorous, clearly structured, with well-motivated intuition.
- Value: ⭐⭐⭐⭐ — Establishes a deep connection between dynamic regret and kernel methods, unifying optimal bounds across multiple loss types.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Simultaneous Swap Regret Minimization via KL-Calibration](simultaneous_swap_regret_minimization_via_kl-calibration.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[NeurIPS 2025\] Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality](improved_regret_and_contextual_linear_extension_for_pandoras_box_and_prophet_ine.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)
- [\[NeurIPS 2025\] Comparing Uniform Price and Discriminatory Multi-Unit Auctions through Regret Minimization](comparing_uniform_price_and_discriminatory_multi-unit_auctions_through_regret_mi.md)

</div>

<!-- RELATED:END -->
