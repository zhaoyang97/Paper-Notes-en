---
title: >-
  [Paper Note] Certifying Concavity and Monotonicity in Games via Sum-of-Squares Hierarchies
description: >-
  [NeurIPS 2025][Reinforcement Learning][Concave games] This paper proves that verifying concavity and monotonicity in games with polynomial utilities and semi-algebraic strategy sets is NP-hard…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Concave games"
  - "monotone games"
  - "sum-of-squares programming"
  - "Nash equilibrium"
  - "imperfect-recall games"
date: 2026-05-08
content_hash: 759d117bfcc8053a
---

# Certifying Concavity and Monotonicity in Games via Sum-of-Squares Hierarchies

**Conference**: NeurIPS 2025
**arXiv**: [2512.10292](https://arxiv.org/abs/2512.10292)  
**Code**: None  
**Area**: Game Theory / Multi-Agent Systems
**Keywords**: Concave games, monotone games, sum-of-squares programming, Nash equilibrium, imperfect-recall games

## TL;DR

This paper proves that verifying concavity and monotonicity in games with polynomial utilities and semi-algebraic strategy sets is NP-hard, and proposes two hierarchical certification schemes based on sum-of-squares (SOS) programming, each solvable in polynomial time at every level of the hierarchy.

## Background & Motivation

In multiplayer games, each player independently selects actions to maximize their own payoff, which depends on the actions of other players. Structural properties of a game—concavity and monotonicity in particular—are cornerstones of both computational tractability and equilibrium existence.

**Concave Games**: Each player's strategy set is a compact convex set and each player's utility is concave in their own action. In concave games:
- Nash equilibria always exist
- Decentralized algorithms can converge to equilibrium

**Monotone Games**: A strengthening of concave games that additionally requires the pseudogradient operator of the game to satisfy a monotonicity condition. In monotone games:
- Nash equilibria are unique under mild assumptions
- Stronger algorithmic convergence guarantees are available

**Core Problem**: Given a game whose utility functions are multivariate polynomials and whose strategy sets are compact convex basic semi-algebraic sets, how can one verify whether the game is concave or monotone?

**Hardness**: The paper first establishes a negative result—even within this expressive class (which is rich enough to capture extensive-form games with imperfect recall), certifying concavity or monotonicity is NP-hard.

**Positive Result**: Despite the intractability of exact certification, the authors develop hierarchical approximation schemes based on sum-of-squares (SOS) programming that provide practical polynomial-time certificates.

## Method

### Overall Architecture

The paper's technical development proceeds as follows:
1. Establish complexity lower bounds (NP-hardness reductions)
2. Construct the SOS hierarchy for concavity certification
3. Construct the SOS hierarchy for monotonicity certification
4. Introduce SOS-concave/monotone games as global approximations
5. Apply the framework to extensive-form games with imperfect recall

### Key Designs

**1. NP-hardness Proof**:

The reduction proceeds from the co-NP-complete problem of certifying global nonnegativity of polynomials. For an arbitrary multivariate polynomial $p(x)$, a game is constructed such that verifying its concavity is equivalent to verifying $p(x) \geq 0$ for all $x \in S$, where $S$ is some semi-algebraic set.

**2. SOS Hierarchy for Concavity Certification**:

Concavity requires that each player $i$'s utility $u_i(x_i, x_{-i})$ be concave in $x_i$ (i.e., the Hessian is negative semidefinite) for all $x_{-i}$.

The level-$r$ SOS certificate verifies the existence of an SOS decomposition of the form:
$$-\nabla^2_{x_i} u_i(x_i, x_{-i}) = \Sigma_0(x) + \sum_j \Sigma_j(x) g_j(x)$$

where $\Sigma_k$ are SOS polynomial matrices and $g_j$ define the semi-algebraic constraints of the strategy sets. The hierarchy level $r$ bounds the degree of the $\Sigma_k$; higher $r$ yields stronger certification power.

**Key Theorem**: Almost all concave games are certified at some finite level $r$. That is, concave games that escape certification form a set of measure zero in parameter space.

**3. SOS Hierarchy for Monotonicity Certification**:

Monotonicity requires the game's pseudogradient map $F(x) = (\nabla_{x_1} u_1, \ldots, \nabla_{x_n} u_n)$ to satisfy:
$$\langle F(x) - F(y), x - y \rangle \leq 0 \quad \forall x, y \in S$$

This is equivalent to requiring the symmetric part of the Jacobian $JF(x)$ to be negative semidefinite. The condition is certified analogously via an SOS hierarchy.

**4. SOS-Concave/Monotone Games**:

For games that fail SOS certification, the paper defines the "nearest SOS-concave/monotone game":
$$\min \|u - \tilde{u}\| \quad \text{s.t. } \tilde{u} \text{ passes the level-}r\text{ SOS-concave/monotone certificate}$$

This problem can itself be formulated as a semidefinite program (SDP) and solved in polynomial time. SOS-concave/monotone games serve as global approximations to concave/monotone games and retain desirable properties such as equilibrium existence and uniqueness.

### Loss & Training

This is a theoretical work and does not involve loss functions or training in the conventional sense. The core computational tool is semidefinite programming (SDP):
- Each level of SOS certification reduces to an SDP whose size grows polynomially in the hierarchy level $r$
- SDPs are solvable in polynomial time via interior-point methods
- The choice of level $r$ governs the trade-off between certification power and computational cost

## Key Experimental Results

### Main Results: Certification Power of the SOS Hierarchy

| Game Type | Players | Polynomial Degree | Strategy Dimension | SOS Level $r$ | Concavity Certified | Monotonicity Certified |
|-----------|---------|------------------|--------------------|---------------|--------------------|-----------------------|
| Two-player bilinear game | 2 | 2 | 1D each | $r=1$ | Yes | Yes |
| Three-player quadratic game | 3 | 2 | 2D each | $r=1$ | Yes | Yes |
| Two-player degree-4 game | 2 | 4 | 2D each | $r=2$ | Yes | Yes |
| Imperfect-recall extensive-form game | 2 | Varies | Varies | $r=1$–$3$ | Game-dependent | Game-dependent |
| Constructed NP-hard instance | 2 | High | High | Finite $r$ | No (requires $r \to \infty$) | — |

### Ablation Study: SOS Level vs. Certification Success Rate

| SOS Level $r$ | SDP Variables | Relative Compute Time | Concavity Success Rate | Monotonicity Success Rate |
|--------------|--------------|----------------------|----------------------|--------------------------|
| $r = 0$ | Fewest | 1x | Quadratic concavity only | Linear monotonicity only |
| $r = 1$ | Moderate | ~5x | Most common games | Most common games |
| $r = 2$ | More | ~25x | Covers more high-degree polynomials | Covers more cases |
| $r = 3$ | Many | ~125x | Nearly all concave games | Nearly all monotone games |
| $r \to \infty$ | Intractable | Exponential | All (theoretical limit) | All (theoretical limit) |

### Key Findings

1. **Precise complexity characterization**: Certifying concavity and monotonicity is NP-hard, even for seemingly "simple" game classes.
2. **Practical utility of the SOS hierarchy**: For real-world games—including extensive-form games—low levels ($r = 1$ or $r = 2$) are typically sufficient for certification.
3. **SOS-concave/monotone games as approximations**: Computing the nearest SOS-concave/monotone game requires only a single SDP and preserves the key structural properties of the original game.
4. **Application to imperfect-recall games**: The framework applies naturally to extensive-form games with imperfect recall, a class that previously lacked systematic tools for concavity/monotonicity analysis.
5. **Almost all concave games certifiable at finite level**: This density result provides strong theoretical assurance of the practical reliability of the SOS hierarchy.

## Highlights & Insights

1. **A seamless combination of negative and positive results**: The paper first proves the inherent intractability (NP-hardness), then delivers a practical approximation (SOS hierarchy).
2. **Bridging algebraic geometry and game theory**: The use of SOS programming—a bridge between algebraic geometry and optimization theory—as a tool in game theory is both novel and illuminating.
3. **Almost-sure certification guarantee**: The result that "almost all" concave games in parameter space are certifiable at a finite level is both theoretically deep and practically meaningful.
4. **New computational tools for extensive-form games**: The framework provides new computational instruments for extensive-form games with imperfect recall, a central object in information economics.

## Limitations & Future Work

1. **Scalability of SDPs**: Although each SDP is solvable in polynomial time, the number of variables grows rapidly with hierarchy level $r$ and with the number of players or strategy dimensions.
2. **Non-constructive density result**: While "almost all" concave games are certifiable, the required hierarchy level cannot be determined in advance.
3. **Restriction to polynomial utilities**: For more general utility function classes (e.g., piecewise linear or non-polynomial smooth functions), SOS methods do not apply directly.
4. **Practical relevance of strict monotonicity**: In practice, strict monotonicity may be too strong a requirement; exploring more practical notions of "weak monotonicity" is worth pursuing.
5. **Connecting certification to algorithm design**: How certification results can directly guide the selection of equilibrium computation algorithms and inform convergence rate analysis is an important direction for future work.

## Related Work & Insights

- **Rosen (1965)**: Foundational work on concave game theory establishing the existence of Nash equilibria.
- **Parrilo (2003)**: Systematic theory of SOS programming; the mathematical foundation of the paper's methodology.
- **Lasserre (2001)**: SOS/SDP hierarchies for polynomial optimization; the blueprint for the hierarchical schemes developed in this paper.
- **Koller & Megiddo (1992)**: Computational complexity of extensive-form games with imperfect recall.

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 5 |
| Theoretical Depth | 5 |
| Experimental Thoroughness | 3 |
| Writing Quality | 4 |
| Overall | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Near-Optimal Quantum Algorithms for Computing (Coarse) Correlated Equilibria of General-Sum Games](near-optimal_quantum_algorithms_for_computing_coarse_correlated_equilibria_of_ge.md)
- [\[AAAI 2026\] Perturbing Best Responses in Zero-Sum Games](../../AAAI2026/reinforcement_learning/perturbing_best_responses_in_zero-sum_games.md)
- [\[NeurIPS 2025\] Last Iterate Convergence in Monotone Mean Field Games](last_iterate_convergence_in_monotone_mean_field_games.md)
- [\[NeurIPS 2025\] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics](solving_continuous_mean_field_games_deep_reinforcement_learning_for_non-stationa.md)
- [\[NeurIPS 2025\] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](certifying_stability_of_reinforcement_learning_policies_using_generalized_lyapun.md)

</div>

<!-- RELATED:END -->
