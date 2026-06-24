---
title: >-
  [Paper Note] Solving Zero-Sum Convex Markov Games
description: >-
  [ICML 2025][Reinforcement Learning][convex Markov games] This work provides the first theoretical guarantee of global convergence to Nash equilibrium for independent policy gradient methods in two-player zero-sum convex Markov games (cMGs). By utilizing non-convex regularization, the problem is reduced to a nonconvex-pPL min-max optimization, and nested/alternate policy gradient algorithms are designed.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "convex Markov games"
  - "Nash equilibrium"
  - "policy gradient"
  - "hidden convexity"
  - "nonconvex-pPL"
date: 2026-05-08
content_hash: 43dde789fe3c3987
---

# Solving Zero-Sum Convex Markov Games

**Conference**: ICML 2025  
**arXiv**: [2506.16120](https://arxiv.org/abs/2506.16120)  
**Code**: None  
**Area**: Reinforcement Learning / Game Theory  
**Keywords**: convex Markov games, Nash equilibrium, policy gradient, hidden convexity, nonconvex-pPL

## TL;DR
This work provides the first theoretical guarantee of global convergence to Nash equilibrium for independent policy gradient methods in two-player zero-sum convex Markov games (cMGs). By utilizing non-convex regularization, the problem is reduced to a nonconvex-pPL min-max optimization, and nested/alternate policy gradient algorithms are designed.

## Background & Motivation
**Background**: Multi-agent reinforcement learning (MARL) is typically modeled as Markov games (MGs), but traditional MGs require utility functions to be additively decomposable over time steps. Convex Markov games (cMGs) allow utility functions to be convex functions of state-action occupancy measures, offering stronger modeling expressive power that covers scenarios such as finding creative game strategies, multi-step alignment for language models, and multi-agent exploration in robotics.

**Limitations of Prior Work**: cMGs violate Bellman consistency—preventing the definition of state-value and action-value functions. Consequently, traditional value-iteration or dynamic programming-based MARL algorithms (such as Q-learning variants) are completely inapplicable. Even proving the existence of a Nash equilibrium requires techniques beyond classic Brouwer/Kakutani fixed-point theorems.

**Key Challenge**: The utility function in the policy space is naturally non-convex. Even in the simplest normal-form games (a single-state special case of cMGs), gradient methods cycle and produce chaotic trajectories. Computing non-convex non-concave saddle points is generally intractable.

**Goal**: Can policy gradient methods converge in zero-sum cMGs?

**Key Insight**: The authors observe that utility functions in cMGs exhibit "hidden convexity"—they are convex functions of the occupancy measure composed with an invertible mapping. Leveraging this structure, combined with non-convex regularization, allows the problem to satisfy the proximal Polyak-Łojasiewicz (pPL) condition.

**Core Idea**: Transform the zero-sum cMG into a nonconvex-pPL min-max optimization problem over a constrained domain through regularization in the occupancy measure space, and design nested/alternate gradient iterations to guarantee convergence.

## Method

### Overall Architecture
Input: A two-player zero-sum cMG $\Gamma = (\mathcal{S}, \mathcal{A}, \mathcal{B}, \mathbb{P}, F, \gamma, \varrho)$, where the maximizer chooses policy $x$, the minimizer chooses policy $y$, and the utility $U(x,y) = F(\lambda_1(x,y), \lambda_2(x,y))$ is defined over occupancy measures. Output: An $\epsilon$-approximate Nash equilibrium policy pair.

The method consists of two steps: (1) reducing the cMG to a nonconvex-pPL (NC-pPL) min-max optimization problem through regularization; (2) designing convergent algorithms tailored for the NC-pPL problem.

### Key Designs

1. **Non-convex Regularization (Hidden-Strongly-Convex Regularization)**:

    - **Function**: Add a regularization term with respect to the maximizer's occupancy measure, $-\frac{\mu}{2}\|\lambda_2(x,y)\|^2$, to the original utility $U(x,y)$, yielding the perturbed utility $U^\mu(x,y)$.
    - **Mechanism**: While $U$ is concave with respect to each player's occupancy measure (hidden concavity), adding the $\mu$-strong convexity regularization turns it into hidden strongly concave, thereby satisfying the pPL conditions.
    - **Design Motivation**: The pPL condition ensures that the best-response mapping $y^\star(x)$ is Lipschitz continuous with respect to $x$ (rather than the typical $\frac{1}{2}$-Hölder continuous), which is key to stabilizing the iterations in policy gradient methods.

2. **Nested Policy Gradient (Nest-PG, Algorithm 1)**:

    - **Function**: The inner loop uses projected gradient ascent to approximate the maximizer's best response, and the outer loop uses projected gradient descent to update the minimizer.
    - **Mechanism**: Leveraging the NC-pPL structure, the inner loop converges to an approximate best response at a linear rate, while the outer loop guarantees descent using the gradient-dominant property of $\Phi(x) = \max_y U^\mu(x,y)$.
    - **Differences from prior work**: It does not require solving the inner problem exactly, allows for inexact gradients, and enables players to learn independently without sharing policies.

3. **Alternating Policy Gradient (Alt-PGDA, Algorithm 2)**:

    - **Function**: The two players alternately execute projected gradient steps but use asymmetric step sizes (the minimizer's step size is much smaller than the maximizer's).
    - **Mechanism**: Through two-time-scale separation (step size $\alpha_x \ll \alpha_y$), the maximizer's response "tracks" the slowly-changing minimizer. It similarly leverages the Lipschitz best-response property of the pPL condition to maintain stability.
    - **Design Motivation**: It is simpler and easier to implement than the nested method, while remaining robust to stochastic/inexact gradients.

### Loss & Training
The objective is to compute the saddle point of $U^\mu$: $\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} U^\mu(x,y)$. Policy gradients are obtained from sampled trajectories via a REINFORCE-style estimator (Definition 3), and an $\epsilon$-greedy policy is required to ensure sufficient exploration. The final approximation error of the Nash equilibrium is controlled by the regularization strength $\mu$.

## Key Experimental Results

### Main Results
This work is purely theoretical with no numerical experiments. The main results concern convergence complexity:

| Algorithm | Iteration/Sample Complexity to $\epsilon$-NE | Features |
|------|----------------------------------------|------|
| Nest-PG | $\text{poly}(1/\epsilon, |\mathcal{S}|, |\mathcal{A}|+|\mathcal{B}|, 1/(1-\gamma))$ | Nested loop, stronger guarantees |
| Alt-PGDA | $\text{poly}(1/\epsilon, |\mathcal{S}|, |\mathcal{A}|+|\mathcal{B}|, 1/(1-\gamma))$ | Single loop, easier to implement |

### Theoretical Contributions

| Contribution | Significance |
|------|------|
| Best-response Lipschitz continuity (Thm 4.1) | First proof of the Lipschitz continuity of the best-response mapping under hidden convexity/NC-pPL conditions. |
| NC-pPL min-max global convergence (Thm 4.3) | First global convergence guarantee of stochastic nested/alternated GDA for nonconvex-pPL functions over constrained domains. |
| cMG Nash equilibrium computation (Main Theorem) | First provably convergent independent policy gradient method for zero-sum cMGs. |

### Key Findings
- Hidden convexity + regularization = pPL condition, which is the key chain of the unified analysis.
- Both algorithms allow inexact gradients, which is crucial for independent learning (since the regularization term depends on both players' policies, exact gradients would require policy sharing).
- The step-size ratio of the alternating method requires careful design; an excessively large ratio leads to divergence.

## Highlights & Insights
- **Bridge from hidden convexity to pPL**: This reduction is highly elegant—while the cMG utility function appears nonconvex-nonconcave, its convex structure is revealed from the perspective of occupancy measures. Activating this structure via regularization then yields the gradient-dominant property. This provides a methodological template for other optimization problems with hidden convex structures.
- **Independent learning guarantee**: The two players do not need to exchange policy information; they only need to estimate their own gradients and update independently, aligning with the practical needs of decentralized learning in MARL.
- **Independent contributions to optimization theory**: The convergence guarantees for NC-pPL and bilateral pPL min-max optimization over constrained domains are themselves new results in optimization theory, which can be transferred to other nonconvex-nonconcave min-max optimization scenarios (e.g., theoretical analysis of GAN training).

## Limitations & Future Work
- Purely theoretical work, lacking experimental validation; empirical performance in practical cMG scenarios remains unknown.
- Requires knowledge of problem parameters (Lipschitz constants, strong convexity parameters, etc.) to configure step sizes, which may be difficult to obtain in practice.
- The complexity is polynomial with respect to the size of the state/action space; function approximation is not considered (for large-scale scenarios).
- Regularization introduces additional approximation error; convergence slows down as $\mu \to 0$.

## Related Work & Insights
- **vs Traditional MG algorithms (Jin et al., 2021; Wei et al., 2021)**: These rely on value iteration and Bellman consistency, thus being inapplicable to cMGs; this work completely bypasses these limitations using policy gradients.
- **vs cMDP Optimization (Kalogiannis et al., 2024)**: Exploiting hidden convexity has been studied in single-agent cMDPs, but the coupling of occupancy measures in the multi-agent setting degrades Hölder continuity. This work recovers Lipschitz continuity through the pPL condition over the policy space.
- **vs Nesterov Smoothing (Nesterov, 2005)**: The proposed method can be viewed as an extension of Nesterov's non-smooth minimization to hidden-convex and non-convex settings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ This work provides the first resolution to the convergence of policy gradients in cMGs; the reduction from hidden convexity to pPL is highly original.
- Experimental Thoroughness: ⭐⭐ Purely theoretical work, no experimental validation.
- Writing Quality: ⭐⭐⭐⭐ The technical overview is clear, and the technical roadmap in Section 1.1 is particularly helpful.
- Value: ⭐⭐⭐⭐ Significantly advances both MARL and min-max optimization theory, opening up new avenues for research into cMG algorithms.
- Overall: ⭐⭐⭐⭐ Solid theoretical contributions, laying the foundation for subsequent empirical studies of cMGs.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Bilevel Optimization over Saddle Points of Zero-Sum Markov Games](../../ICML2026/reinforcement_learning/bilevel_optimization_over_saddle_points_of_zero-sum_markov_games.md)
- [\[AAAI 2026\] Perturbing Best Responses in Zero-Sum Games](../../AAAI2026/reinforcement_learning/perturbing_best_responses_in_zero-sum_games.md)
- [\[NeurIPS 2025\] Certifying Concavity and Monotonicity in Games via Sum-of-Squares Hierarchies](../../NeurIPS2025/reinforcement_learning/certifying_concavity_and_monotonicity_in_games_via_sum-of-squares_hierarchies.md)
- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](../../ICML2026/reinforcement_learning/global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[NeurIPS 2025\] Solving Neural Min-Max Games: The Role of Architecture, Initialization & Dynamics](../../NeurIPS2025/reinforcement_learning/solving_neural_min-max_games_the_role_of_architecture_initialization_dynamics.md)

</div>

<!-- RELATED:END -->
