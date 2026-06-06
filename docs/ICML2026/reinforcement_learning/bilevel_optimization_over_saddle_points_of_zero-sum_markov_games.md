---
title: >-
  [Paper Note] Bilevel Optimization over Saddle Points of Zero-Sum Markov Games
description: >-
  [ICML2026][Reinforcement Learning][Bilevel optimization] The PANDA algorithm is proposed to solve bilevel RL problems where the lower level is a regularized zero-sum Markov game. By employing a penalty reformulation base…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Bilevel optimization"
  - "zero-sum Markov games"
  - "policy gradient"
  - "Nikaido-Isoda function"
  - "saddle-point equilibrium"
date: 2026-05-08
content_hash: cad9281b8c88a5e7
---

# Bilevel Optimization over Saddle Points of Zero-Sum Markov Games

**Conference**: ICML2026  
**arXiv**: [2605.26654](https://arxiv.org/abs/2605.26654)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Bilevel optimization, zero-sum Markov games, policy gradient, Nikaido-Isoda function, saddle-point equilibrium

## TL;DR
The PANDA algorithm is proposed to solve bilevel RL problems where the lower level is a regularized zero-sum Markov game. By employing a penalty reformulation based on the Nikaido-Isoda function and a pure first-order policy gradient method, the approach achieves an iteration complexity of $\tilde{O}(\epsilon^{-1})$ and a sample complexity of $\tilde{O}(\epsilon^{-3})$, matching the best-known rates for single-policy lower-level BRL.

## Background & Motivation

**Background**: Bilevel Reinforcement Learning (BRL) is a powerful paradigm for modeling hierarchical decision-making. The upper-level (UL) learner optimizes high-level variables (e.g., incentive parameters, reward design), while the lower-level (LL) solves an RL problem influenced by those variables. Recently, algorithms such as PARL, HPGD, SoBiRL, and First-Order BRL have been proposed with theoretical convergence guarantees.

**Limitations of Prior Work**: Existing BRL methods almost exclusively assume the lower level is a **single-policy** MDP (with only one agent performing max or min), making them unable to handle multi-agent adversarial structures. However, in scenarios like incentive design and RLHF preference learning, the lower level naturally involves the coupled gaming of two adversarial policies. Directly migrating single-policy BRL methods to min-max game settings fails; for instance, the hypergradient derivations in HPGD/SoBiRL rely on the closed-form optimal policy characteristics of single-policy MDPs, which do not hold under coupled dual-policy optimization.

**Key Challenge**: The strategic coupling of two adversarial policies in zero-sum Markov games makes algorithm design inherently more difficult. Existing methods for this setting are either heuristics without convergence guarantees (Meta-Gradient), rely on computationally expensive second-order information like Hessian inverses (DA), or can only converge to stationary points of a penalty proxy problem rather than the original problem (PBRL).

**Goal**: To design a stochastic first-order method to solve bilevel optimization problems over saddle points (BOSMG) where the lower level is a regularized min-max zero-sum Markov game (MMZSMG), while achieving provably efficient iteration and sample complexity.

**Key Insight**: The Nikaido-Isoda (NI) function is utilized to characterize the degree of deviation of lower-level policies from equilibrium—the NI function is non-negative and zero if and only if the policy pair is a Nash equilibrium. By adding the NI function as a penalty term to the upper-level objective, the bilevel constrained problem is transformed into an unconstrained penalty optimization, thereby avoiding hypergradient computation. Furthermore, the inherent structure of regularized MMZSMG (unique equilibrium, non-uniform PŁ property of the NI function) is leveraged to ensure convergence to an approximate stationary point of the original problem.

**Core Idea**: Combining NI function penalty reformulation with descent-ascent policy gradients enables a pure first-order solution for BOSMG, bypassing hypergradients and second-order information.

## Method

### Overall Architecture
PANDA solves problems of the form: $\min_{x,\phi,\psi} f(x,\phi,\psi)$ s.t. $(\phi,\psi) \in \arg\min_{\phi'}\max_{\psi'} J(x,\phi',\psi')$, where $x$ represents the upper-level variables, and $(\phi,\psi)$ parameterize the policies of the min-player and max-player, respectively, with $J$ as the regularized value function. The algorithm reformulates this into a penalty form using the NI function: $\min_{x,\phi,\psi} f(x,\phi,\psi) + \lambda \cdot g(x,\phi,\psi)$. It updates $x$ in an outer loop while the inner loop alternately approximates best responses and solves the penalty subproblem, requiring only first-order information.

### Key Designs

1. **Nikaido-Isoda Penalty Reformulation**:

    - Function: Transforms the bilevel constrained problem into a single-level penalty optimization solvable by first-order methods.
    - Mechanism: Defines the NI function as $g(x,\phi,\psi) = \max_{\psi'} J(x,\phi,\psi') - \min_{\phi'} J(x,\phi',\psi)$, which precisely measures the current policies' deviation from the Nash equilibrium. Adding this as a penalty term yields $L_\lambda(x,\phi,\psi) = f(x,\phi,\psi) + \lambda \cdot g(x,\phi,\psi)$. When $\lambda$ is sufficiently large, the gradient deviation between the stationary points of $L_\lambda^*(x)$ and the original hyper-objective $F(x)$ is $O(\lambda^{-1})$.
    - Design Motivation: Avoids the high computational cost of traditional bilevel RL, which requires computing hypergradients via the chain rule (involving Hessian inverses). The NI function naturally fits the saddle-point structure of min-max games and is more accurate than simple value function gaps.

2. **Three-step Inner Loop (Best Response + Penalty Subproblem + Hypergradient Update)**:

    - Function: Alternately completes lower-level equilibrium approximation and upper-level parameter updates within each outer loop iteration.
    - Mechanism: **Step 1** — Perform $K$ steps of policy gradient descent/ascent on $\tilde{\phi}$ and $\tilde{\psi}$ to solve the two best-response problems in the NI function; **Step 2** — Use the approximated NI function $\tilde{g}(x,\phi,\psi,\tilde{\phi},\tilde{\psi}) = J(x,\phi,\tilde{\psi}) - J(x,\tilde{\phi},\psi)$ to construct the penalty subproblem gradient and update $(\phi,\psi)$; **Step 3** — Estimate the hypergradient $\nabla_x \tilde{L}_\lambda$ using the updated policies and update the upper-level variable $x$ via stochastic gradient descent.
    - Design Motivation: The three-step decomposition exploits the nested structure—an inner loop of $K = O(\log\lambda)$ steps ensures $(\phi,\psi)$ are close enough to the penalty subproblem optimum for effective outer-loop hypergradient estimation.

3. **Non-uniform PŁ Property of the NI Function**:

    - Function: Provides gradient dominance conditions for the NI function in the policy space to support convergence analysis.
    - Mechanism: Proves that for any $x$ and $(\phi,\psi)$, the NI function satisfies $\frac{1}{2}\|\nabla_{(\phi,\psi)}g\|^2 \geq \mu(\phi,\psi) \cdot g(x,\phi,\psi)$, where $\mu(\phi,\psi)$ depends on the minimum probability of policies and the regularization coefficient. This generalizes the non-uniform PŁ results for single-agent soft value functions by Mei et al. (2020) to two-agent zero-sum games.
    - Design Motivation: This property is a critical structural tool for proving that PANDA converges without requiring strong convexity assumptions on the upper or lower-level objectives.

## Key Experimental Results

### Main Results

| Environment | Method | UL Objective | NE Gap | Note |
|------|------|---------|--------|------|
| Synthetic (Incentive Design) | **PANDA** | **Highest** (≈2.55) | **≈0** | Close to Oracle upper bound |
| Synthetic | META | ≈2.2 | ≈0 | Heuristic, insufficient UL optimization |
| Synthetic | DA | ≈2.3 | ≈0 | Requires second-order info |
| Synthetic | PBRL | ≈2.35 | ≈0 | Suboptimal UL |
| Sentinel-Intruder 5×5 | **PANDA** | **Lowest UL loss** | — | Effectively avoids forbidden zones |
| Sentinel-Intruder 20×20 | **PANDA** | **Lowest UL loss** | — | Superior at large scale |

### Ablation Study (Effect of Penalty Parameter $\lambda$)

| $\lambda$ | UL Reward | NE Gap | Note |
|-----------|---------|--------|------|
| 1 | High | **Large** | Weak lower-level equilibrium constraint |
| 4 | High | ≈0 | Good balance between equilibrium and UL goal |
| 10 | Slightly lower | ≈0 | Strong penalty slightly sacrifices UL goal |

### Complexity Comparison

| Algorithm | LL Problem | Stochastic/Det. | Iteration Complexity | Sample Complexity | Oracle |
|------|---------|-----------|------------|------------|--------|
| **PANDA** | **Min-Max** | **Stochastic** | $\tilde{O}(\epsilon^{-1})$ | $\tilde{O}(\epsilon^{-3})$ | First-order |
| First-Order BRL | Max | Stochastic | $\tilde{O}(\epsilon^{-1})$ | $\tilde{O}(\epsilon^{-3})$ | First-order |
| SoBiRL | Max | Stochastic | $\tilde{O}(\epsilon^{-1.5})$ | $\tilde{O}(\epsilon^{-3.5})$ | First-order |
| DA | Min-Max | Deterministic | $\tilde{O}(\epsilon^{-1})$ | — | 1st + 2nd order |
| META | Min-Max | Stochastic | N/A | N/A | First-order |

### Key Findings
- PANDA is the **first** first-order method to provide convergence guarantees for BOSMG in a stochastic setting, with iteration and sample complexities matching the optimal rates of single-policy BRL.
- $\lambda$ controls the trade-off between lower-level equilibrium accuracy and upper-level objective optimization: a $\lambda$ that is too small relaxes the equilibrium constraint, while a $\lambda$ that is too large results in excessive penalization.
- PANDA remains effective and outperforms baselines in 20×20 large grid environments, demonstrating its scalability to larger dimensions.

## Highlights & Insights
- **The combination of NI function + penalty method** is an elegant solution for min-max bilevel problems. The NI function naturally measures saddle-point deviation, and the penalty reformulation integrates it into the objective, avoiding hypergradient computation. This framework is transferable to other hierarchical optimization scenarios with adversarial lower-level structures.
- **Generalizing the non-uniform PŁ property to zero-sum games** is a significant theoretical contribution in its own right. It indicates that the NI function of regularized zero-sum games under softmax parameterization possesses a favorable optimization landscape, which can be applied to analyze other game-theoretic algorithms based on the NI function.
- **Requiring only $O(\log\lambda)$ inner loop steps** is a practical design improvement—logarithmic inner loops mean that the overall computational burden grows slowly.

## Limitations & Future Work
- Currently, Ours only handles **regularized** zero-sum Markov games (relying on the strong convexity-concavity of the regularizer to ensure equilibrium uniqueness); extending this to unregularized or general-sum games remains an open problem.
- Uses **tabular softmax parameterization**, and performance under function approximation (e.g., neural network policies) has not yet been verified.
- The scale of experimental environments is limited (max 20×20 grid), and scalability in real-world large-scale multi-agent scenarios needs further validation.
- The logarithmic factors and constants hidden in the sample complexity might be large, potentially creating a gap between actual efficiency and theoretical rates.

## Related Work & Insights
- **Bilevel RL**: PARL (Chakraborty+, ICLR'24), HPGD (Thoma+, '24), and SoBiRL (Yang+, '25) handle single-policy lower levels; First-Order BRL (Gaur+, NeurIPS'25) and SLAC (Zeng+, '25) are representative penalty methods.
- **Zero-Sum Markov Games**: Cen+ (JMLR'24) provided linear convergence algorithms for Nash equilibria in regularized zero-sum games; Munos+ ('24) applied the zero-sum game framework to RLHF.
- **Bilevel Optimization Theory**: Kwon+ ('24) and Chen+ ('25) established convergence theories for penalty methods in non-convex lower-level bilevel optimization.
- Insight: The NI function penalty approach could be applied to preference alignment in RLHF—when the preference model involves adversarial training, a similar framework may prove effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Perturbing Best Responses in Zero-Sum Games](../../AAAI2026/reinforcement_learning/perturbing_best_responses_in_zero-sum_games.md)
- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](../../ICLR2026/reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICML 2026\] FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs](fab_a_first-order_ab-based_gradient_algorithm_for_distributed_bilevel_optimizati.md)
- [\[ICML 2026\] MindZero: Learning Online Mental Reasoning with Zero Annotations](mindzero_learning_online_mental_reasoning_with_zero_annotations.md)

</div>

<!-- RELATED:END -->
