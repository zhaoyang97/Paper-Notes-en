---
title: >-
  [Paper Note] On the $O(1/T)$ Convergence of Alternating Gradient Descent-Ascent in Bilinear Games
description: >-
  [ICLR 2026][Reinforcement Learning][Alternating gradient descent-ascent] This paper provides the first proof that alternating gradient descent-ascent (AltGDA) converges to a Nash equilibrium at an $O(1/T)$ rate in constr…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Alternating gradient descent-ascent"
  - "bilinear games"
  - "Nash equilibrium"
  - "convergence rate"
  - "performance estimation programming"
date: 2026-05-08
content_hash: 667ab7ac6b234d3c
---

# On the $O(1/T)$ Convergence of Alternating Gradient Descent-Ascent in Bilinear Games

**Conference**: ICLR 2026
**arXiv**: [2510.03855](https://arxiv.org/abs/2510.03855)  
**Code**: None  
**Area**: Reinforcement Learning / Game Theory / Optimization
**Keywords**: Alternating gradient descent-ascent, bilinear games, Nash equilibrium, convergence rate, performance estimation programming

## TL;DR
This paper provides the first proof that alternating gradient descent-ascent (AltGDA) converges to a Nash equilibrium at an $O(1/T)$ rate in constrained bilinear zero-sum games (when an interior NE exists), outperforming simultaneous GDA's $O(1/\sqrt{T})$ rate. The analysis characterizes the "friction" effect produced when trajectories collide with the boundary via an energy function decay argument, and further optimizes step sizes through performance estimation programming (PEP).

## Background & Motivation

**Background**: No-regret learning is the dominant approach for computing game equilibria, underpinning superhuman-level AI in poker, Stratego, and Diplomacy. Theoretically, optimistic methods (e.g., OGDA) achieve $O(1/T)$, yet in practice alternating update strategies such as CFR+ are more widely adopted. Alternating updates—where the two players update sequentially—serve as a numerical trick that substantially improves empirical performance.

**Limitations of Prior Work**: Although alternating updates empirically outperform simultaneous updates by a wide margin, their theoretical understanding remains severely limited. In the unconstrained setting, AltGDA has been shown to achieve $O(1/T)$, but **in the constrained setting (which corresponds to the standard definition of Nash equilibrium) no theoretical guarantees exist**—this is a longstanding open problem.

**Key Challenge**: Simultaneous GDA achieves only $O(1/\sqrt{T})$ in constrained bilinear games, while optimistic methods reach $O(1/T)$ but require additional structure. AltGDA empirically exhibits $O(1/T)$ behavior, yet no proof has been established.

**Goal**: To prove an $O(1/T)$ convergence rate for AltGDA in the constrained setting.

**Key Insight**: The authors identify a two-phase behavior in AltGDA trajectories—an initial phase in which iterates collide with the simplex boundary and are "reflected," followed by an interior cycling phase. These collisions cause the energy function to decay, and the decay precisely controls the additivity of residual regret terms.

**Core Idea**: The "energy dissipation" generated when AltGDA trajectories collide with the constraint boundary renders the residual terms summable, thereby extending the unconstrained $O(1/T)$ result to the constrained setting.

## Method

### Overall Architecture
The AltGDA algorithm proceeds as: $\mathbf{x}^{t+1} = \Pi_\mathcal{X}(\mathbf{x}^t - \eta A^\top \mathbf{y}^t)$, followed by $\mathbf{y}^{t+1} = \Pi_\mathcal{Y}(\mathbf{y}^t + \eta A \mathbf{x}^{t+1})$ (note that the $\mathbf{y}$ update uses the updated value of $\mathbf{x}$). The convergence metric is the duality gap of the ergodic average.

### Key Designs

1. **Energy Function and the "Collision Dissipation" Mechanism**

    - Function: Construct the energy function $\mathcal{E}(\mathbf{x}^t, \mathbf{y}^t) = \|\mathbf{x}^t - \mathbf{x}^*\|^2 + \|\mathbf{y}^t - \mathbf{y}^*\|^2 - \eta(\mathbf{y}^t)^\top A \mathbf{x}^t$
    - Core Finding: In the unconstrained case the residual $r_t \equiv 0$ (a known result); in the constrained case the first-order optimality conditions of the projection guarantee $r_t \geq 0$, and the key contribution is proving $r_t \leq \mathcal{E}^t - \mathcal{E}^{t+1}$
    - Physical Intuition: Trajectories collide with the simplex boundary → projection induces energy loss → energy decreases monotonically → $\sum r_t$ is bounded → $O(1/T)$ convergence

2. **Global $O(1/T)$ Convergence under an Interior NE (Theorem 1)**

    - Condition: The game possesses an interior Nash equilibrium (all strategy probabilities are strictly positive)
    - Step-size requirement: $\eta \leq \frac{1}{\|A\|_2} \min\{\min_i x_i^*, \min_j y_j^*\}$
    - Convergence bound: $\text{DualityGap}(\bar{\mathbf{x}}^T, \bar{\mathbf{y}}^T) \leq \frac{9 + 4\eta\|A\|_2}{\eta T}$
    - Significance: **The first result to prove that alternating updates outperform simultaneous updates in constrained minimax optimization**

3. **Local $O(1/T)$ Convergence (Theorem 2)**

    - Function: Establishes local convergence when no interior NE exists
    - Mechanism: A neighborhood $S_0$ of the NE is defined such that within this region trajectories no longer touch the boundary of non-support faces; the energy may increase locally but its cumulative increment is bounded (by $\leq \delta^2/128$)
    - Step size: $\eta \leq \frac{1}{2\|A\|_2}$, **independent** of game parameters
    - Convergence bound: $\frac{9 + 7\eta\|A\|_2 + \delta^2/128}{\eta T}$

4. **Performance Estimation Programming (PEP) Framework**

    - Function: Numerically compute the tightest convergence bound for AltGDA and optimize step sizes
    - Mechanism: The worst-case performance problem is reformulated as an SDP, and a grid search is performed over the step size
    - Key Findings: The optimized step size follows a periodically decaying pattern ($O(1/(\log T)^\alpha)$), and the corresponding duality gap confirms convergence toward $O(1/T)$; SimGDA retains $O(1/\sqrt{T})$ even under optimized step sizes

### Loss & Training
(This work concerns an optimization algorithm; there is no training loss. The step size $\eta$ is the central hyperparameter.)

## Key Experimental Results

### Main Results (10×20 random matrix games, 6 distributions)

| Algorithm | Convergence Behavior | Step Size | $T=10^6$ |
|-----------|---------------------|-----------|---------|
| AltGDA | $O(1/T)$ | $\eta=0.01$ (constant) | Gap→0 |
| SimGDA | Does not converge (constant step size) | $\eta=0.01$ (constant) | Gap oscillates |

### PEP Numerical Results ($T=5$ to $T=50$)

| Algorithm | Convergence Rate under Optimized Step Size |
|-----------|------------------------------------------|
| AltGDA | **$O(1/T)$** |
| SimGDA | $O(1/\sqrt{T})$ |

### Key Findings
- AltGDA **consistently** exhibits $O(1/T)$ convergence across 6 distributions (uniform/integer/binary/normal/lognormal/exponential) and 3 scales (10×20 / 30×60 / 60×120)
- The "slow convergence phase" observed at the beginning of trajectories corresponds to the energy dissipation stage, after which fast $O(1/T)$ convergence ensues
- SimGDA cannot converge under a constant step size—a $T$-dependent decreasing step size is necessary to obtain $O(1/\sqrt{T})$
- The empirical convergence rate scales linearly with $1/\eta$, consistent with the predictions of Theorems 1 and 2

## Highlights & Insights
- **The physical intuition of "collision dissipation" is remarkably elegant**: the convergence property of projection-based algorithms is likened to the energy loss of a particle colliding with a boundary. This analogy is not only intuitively transparent but directly motivates the proof technique. The two-phase behavior (boundary collision → interior cycling) was first identified through trajectory visualization.
- **Constant step size + alternation = explorative + implicit optimism**: The alternating structure of AltGDA inherently provides an implicit "optimistic" effect—the $\mathbf{y}$ update observes the most recent value of $\mathbf{x}$, producing an effect analogous to extrapolation. This explains why alternating updates are so effective in practice.
- **Methodological contribution of the PEP framework**: This is the first application of step-size-optimizing PEP to primal-dual algorithms involving linear operators, and the framework is readily extensible to a broader class of minimax algorithms.

## Limitations & Future Work
- Theorem 1 requires the existence of an interior NE (with step size depending on the minimum support probability), a condition not satisfied by most matrix games
- The local result of Theorem 2 requires the initial point to lie within the NE neighborhood, with no guarantee on how to reach that neighborhood
- **Coverage is limited to bilinear games**; more general convex-concave games are not addressed
- PEP numerical results suggest that global $O(1/T)$ convergence may hold in general, but an analytic proof has yet to be established

## Related Work & Insights
- **vs. Bailey et al. (2020)**: $O(1/T)$ convergence of AltGDA in the unconstrained setting is already known; this paper extends the result to the constrained case—the core difficulty being that projection breaks energy conservation
- **vs. Wibisono et al. (2022)**: They prove that alternating mirror descent (with Legendre regularization) achieves $O(1/T^{2/3})$, but their analysis does not cover Euclidean projection / GDA
- **vs. Optimistic methods**: OGDA/Mirror-Prox achieve $O(1/T)$ but require additional structure; AltGDA is a simpler, pure gradient method

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First proof of accelerated convergence for AltGDA in the constrained setting; the energy dissipation mechanism is an entirely new analytical tool
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple distributions, multiple scales of matrix games, and PEP numerical validation; large-scale game applications are absent
- Writing Quality: ⭐⭐⭐⭐⭐ — The three-pronged argument (physical intuition → mathematical proof → numerical validation) is highly persuasive
- Value: ⭐⭐⭐⭐⭐ — Resolves a longstanding open problem in the game theory/optimization community with important implications for large-scale game solving

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Last Iterate Convergence in Monotone Mean Field Games](../../NeurIPS2025/reinforcement_learning/last_iterate_convergence_in_monotone_mean_field_games.md)
- [\[ICLR 2026\] Solving Football by Exploiting Equilibrium Structure of 2p0s Differential Games with One-Sided Information](solving_football_by_exploiting_equilibrium_structure_of_2p0s_differential_games_.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)
- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](learning_to_play_multi-follower_bayesian_stackelberg_games.md)
- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)

</div>

<!-- RELATED:END -->
