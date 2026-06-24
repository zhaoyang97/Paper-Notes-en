---
title: >-
  [Paper Note] Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality
description: >-
  [NeurIPS 2025][Reinforcement Learning][Pandora's Box] This paper proposes new algorithms for the online Pandora's Box problem, improving regret from $\widetilde{O}(n\sqrt{T})$ to $\widetilde{O}(\sqrt{nT})$ (matching the lower bound), and introduces the first contextual linear extension achieving $\widetilde{O}(nd\sqrt{T})$ regret.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Pandora's Box"
  - "prophet inequality"
  - "online learning"
  - "regret bound"
  - "contextual linear bandits"
date: 2026-05-08
content_hash: 6d967464da5792de
---

# Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality

**Conference**: NeurIPS 2025
**arXiv**: [2505.18828](https://arxiv.org/abs/2505.18828)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Pandora's Box, prophet inequality, online learning, regret bound, contextual linear bandits

## TL;DR
This paper proposes new algorithms for the online Pandora's Box problem, improving regret from $\widetilde{O}(n\sqrt{T})$ to $\widetilde{O}(\sqrt{nT})$ (matching the lower bound), and introduces the first contextual linear extension achieving $\widetilde{O}(nd\sqrt{T})$ regret.

## Background & Motivation

**Background**: Pandora's Box — $n$ boxes each with a cost and an unknown reward distribution, requiring decisions on the order of opening and when to stop. Weitzman (1978) provides the optimal policy under known distributions.

**Limitations of Prior Work**: In the online setting, Agarwal et al. (2024) achieve $\widetilde{O}(Un\sqrt{T})$ (where $U$ can be as large as $n$), while the lower bound is $\Omega(\sqrt{nT})$, leaving an $n$-factor gap.

**Key Challenge**: Prior methods shift probability mass by a fixed amount, yielding a loose TV-distance bound; $\text{Term}_{t,i}$ is coarsely bounded by $U \cdot \text{TV}$.

**Goal**: Close the $n$-factor gap in the non-contextual setting; establish a contextual linear model achieving $\sqrt{T}$-type regret.

**Key Insight**: Bernstein-type DKW inequalities for adaptive probability mass adjustment, combined with fine-grained analysis of the utility function's derivative.

**Core Idea**: Adaptive optimistic distributions + two-region derivative analysis = minimax-optimal regret.

## Method

### Overall Architecture

Each round runs the Weitzman policy with threshold vector $\sigma_t$. The core step constructs an optimistic distribution $\hat{\mathcal{E}}_t$ from samples to compute $\sigma_t$.

### Key Designs

1. **Adaptive Optimistic Distribution (Bernstein-type)**:

    - The empirical CDF is shifted downward using a Bernstein confidence interval: $F_{\hat{\mathcal{E}}}(x) = \max\{0, F_\mathcal{E}(x) - \sqrt{2F_\mathcal{E}(x)(1-F_\mathcal{E}(x))L/m} - L/m\}$
    - Unlike the fixed shift of $\sim 1/\sqrt{m}$, this automatically shifts less in tail regions.

2. **Two-Region Derivative Analysis of the Utility Function**:

    - $\widetilde{R}_i(\sigma_t; z)$ is 1-Lipschitz and monotone in $z$
    - Large-$z$ region: $\sqrt{F(1-F)}$ automatically shrinks when $z$ is large
    - Small-$z$ region: derivative bound $\partial_z \widetilde{R}_i \leq \prod_{j<i} F_{D_j}(z)/Q_{t,i}$
    - Cauchy-Schwarz + telescoping sum yields per-round regret of $\widetilde{O}(\sqrt{\sum_i Q_{t,i}/m_{t,i}})$

3. **Contextual Linear Extension**:

    - Expected reward $\mu_{t,i} = \theta_i^\top x_{t,i}$, with fixed noise
    - Ridge regression + value-optimistic empirical distribution: $\hat{z} = \min\{1, v - \text{LCB} + \text{UCB}\}$
    - Regret is decomposed into three terms: optimistic reweighting, value shift, and empirical vs. true distribution

### Core Results

| Problem | Ours | Prev. SOTA | Lower Bound |
|---------|------|-----------|-------------|
| Non-contextual Pandora's Box | $\widetilde{O}(\sqrt{nT})$ | $\widetilde{O}(Un\sqrt{T})$ | $\Omega(\sqrt{nT})$ |
| Contextual Linear PB | $\widetilde{O}(nd\sqrt{T})$ | $\widetilde{O}(nT^{5/6})$ | Open |
| Non-contextual Prophet | $\widetilde{O}(\sqrt{nT})$ | $\widetilde{O}(n\sqrt{T})$ | $\Omega(\sqrt{T})$ |

## Key Experimental Results

### Technical Comparison

| Technique | Agarwal et al. | Ours |
|-----------|---------------|------|
| Optimistic distribution | Fixed mass shift $\sim 1/\sqrt{m}$ | Bernstein-type adaptive |
| Regret decomposition | $U \cdot \text{TV}$ | Two-region derivative analysis |
| Utility function | Lipschitz not exploited | 1-Lipschitz + monotone |
| Result | $\widetilde{O}(Un\sqrt{T})$ | $\widetilde{O}(\sqrt{nT})$ |

### Key Findings
- The Bernstein-type construction's reduced mass shift in regions where the CDF concentrates well is the core improvement.
- The 1-Lipschitz + monotonicity property eliminates the $U$ factor.
- Linear dependence on $n$ in the contextual setting may be unavoidable.
- Whether $\widetilde{O}(\sqrt{T})$ is achievable for Prophet Inequality remains open.

## Highlights & Insights
- The **adaptive optimistic distribution** is the most elegant contribution: automatic shrinkage in tail regions.
- The **two-region derivative analysis** exploits the telescoping sum structure inherent to Pandora's Box.
- **Value-optimistic debiasing** handles the contextual setting in a natural and elegant manner.
- The techniques are transferable to bandit problems requiring full distribution learning.

## Limitations & Future Work
- Can the linear dependence on $n$ in the Contextual Linear regret be improved? This may be unavoidable even when all boxes share $\theta$, since independent noise distributions must be learned per box.
- The minimax-optimal regret for Prophet Inequality remains unresolved — a $\sqrt{n}$ gap persists between the $\widetilde{O}(\sqrt{nT})$ upper bound and the $\Omega(\sqrt{T})$ lower bound.
- Jin et al. (2024) show that the optimal sample complexity for Prophet is $\widetilde{O}(1/\epsilon^2)$, independent of $n$, suggesting $\widetilde{O}(\sqrt{T})$ may be achievable.
- Empirical validation of the algorithms' finite-$T$ performance is absent.
- All results assume independent reward distributions; correlations present in real applications are not addressed.
- When $d$ is large ($d = \Omega(T^{1/3})$), the contextual results do not improve upon prior methods.

### Algorithm Pseudocode Overview

Core loop: at each round $t$, for each box $i$, construct optimistic distribution $\hat{\mathcal{E}}_{t,i}$ → compute threshold $\sigma_{t,i}$ → run Weitzman's algorithm → observe rewards from opened boxes → update counter $m_{t+1,i}$. The overall framework follows the optimism-in-the-face-of-uncertainty principle.

## Related Work & Insights
- **vs. Agarwal et al. (2024)**: Fixed mass shift introduces the $U$ factor; the adaptive construction in this work eliminates it.
- **vs. Atsidakou et al. (2024)**: Contextual regret of $T^{5/6}$ vs. $\sqrt{T}$ in this work.
- **vs. Gatmiry et al. (2024)**: Lower bound $\Omega(\sqrt{nT})$ is matched by this work.
- **vs. Guo et al. (2021)**: They use a similar Bernstein-type construction for PAC guarantees; this work adapts it to regret analysis via a fundamentally different analytical approach.
- **vs. Weitzman (1978)**: The classical optimal policy requires known distributions; this work learns distributions while maintaining optimality.

### Algorithm Complexity

Each round requires constructing an optimistic distribution and solving for thresholds for each box, with computational cost $O(n \cdot m_{\max})$ where $m_{\max}$ is the maximum number of samples. The computational cost is comparable to Agarwal et al.; the improvement is purely in statistical efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bernstein optimistic distribution and two-region analysis are entirely new; contextual extension is the first of its kind.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical; matching the lower bound serves as the strongest validation.
- Writing Quality: ⭐⭐⭐⭐ Technical overview is clear, though notation-heavy.
- Value: ⭐⭐⭐⭐⭐ Closes an important theoretical gap and opens a new direction for contextual Pandora's Box.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)
- [\[NeurIPS 2025\] Establishing Linear Surrogate Regret Bounds for Convex Smooth Losses via Convolutional Fenchel–Young Losses](establishing_linear_surrogate_regret_bounds_for_convex_smooth_losses_via_convolu.md)

</div>

<!-- RELATED:END -->
