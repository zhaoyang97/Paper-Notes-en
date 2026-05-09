---
title: >-
  [Paper Note] Distributionally Robust Online Markov Game with Linear Function Approximation
description: >-
  [AAAI 2026][Reinforcement Learning][Markov Game] This paper studies online distributionally robust Markov games with linear function approximation. It is the first to identify the hardness of learning in this setting, and proposes the DR-CCE-LSI algorithm, which achieves minimax-optimal sample complexity with respect to the feature dimension $d$ under a specific feature mapping condition.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Markov Game
  - Distributional Robustness
  - Linear Function Approximation
  - Coarse Correlated Equilibrium
  - Sample Complexity
date: 2026-05-08
content_hash: 5418897578e2326e
---

# Distributionally Robust Online Markov Game with Linear Function Approximation

**Conference**: AAAI 2026
**arXiv**: [2511.07831](https://arxiv.org/abs/2511.07831)
**Code**: None
**Area**: Multi-Agent Reinforcement Learning / Robust Game Theory
**Keywords**: Markov Game, Distributional Robustness, Linear Function Approximation, Coarse Correlated Equilibrium, Sample Complexity

## TL;DR

This paper studies online distributionally robust Markov games with linear function approximation. It is the first to identify the hardness of learning in this setting, and proposes the DR-CCE-LSI algorithm, which achieves minimax-optimal sample complexity with respect to the feature dimension $d$ under a specific feature mapping condition.

## Background & Motivation

### State of the Field

Multi-agent reinforcement learning (MARL) has been extensively studied within game-theoretic frameworks, with Markov games serving as a central model. Linear function approximation has been introduced to handle large state spaces. Some progress has been made on single-agent robust RL and tabular robust games.

### Limitations of Prior Work

**The sim-to-real gap is more severe in MARL**: equilibrium solutions are highly sensitive to small environmental perturbations.

**Online settings are understudied**: existing robust Markov game algorithms primarily target generative model or offline settings.

**Linear approximation and robustness have not been combined**: the online robust general-sum game setting with linear function approximation remains entirely unexplored.

### Root Cause

**Support shift**: samples collected online may fail to cover all trajectories under worst-case environments, causing algorithms to perform no better than random guessing on unobserved states. This is the fundamental obstacle to robust online learning.

### Paper Goals

To design provably sample-efficient algorithms for finding approximate robust coarse correlated equilibria (CCE) in general-sum games with linear function approximation and $d$-rectangular uncertainty sets.

### Starting Point

1. First establish the impossibility of online learning via a constructed counterexample (lower bound $\Omega(\sigma \cdot HK)$).
2. Introduce the **vanishing minimum assumption** to circumvent the support shift problem.
3. Design a least-squares value iteration algorithm with agent-specific bonus terms.

### Core Idea

**Under the vanishing minimum assumption, exploit the $d$-rectangular structure to preserve linearity, and achieve sample-efficient robust equilibrium learning via coordinate-wise ridge regression and agent-specific exploration bonuses.**

## Method

### Overall Architecture

The DR-CCE-LSI (Distributionally Robust CCE Least Square Iteration) algorithm proceeds as follows in each episode $k$:
1. Each player $i$ estimates the robust action-value function $Q_{i,h}^k$ via ridge regression using historical data.
2. A CCE of the $n$-player matrix game induced by the estimated $Q$ matrices is computed via the Find-CCE subroutine.
3. The CCE policy is executed to interact with the environment and collect new data.
4. The feature covariance matrix $\Lambda_h^{k+1}$ is updated.

### Key Design 1: $d$-Rectangular Uncertainty Set

**Function**: Define an appropriate robust uncertainty set compatible with the linear MDP structure.

**Mechanism**: Assume the transition kernel $P_h(s'|s, \boldsymbol{a}) = \langle \phi_{s\boldsymbol{a}}, \mu_h^0(s') \rangle$ has a linear structure. The $d$-rectangular uncertainty set applies perturbations to each coordinate of $\mu_h$:
$$\mathcal{U}_{TV}^{\sigma_i}(\mu_h^0) : \bigotimes_{j \in [d]} \{\mu_{h,j} : D_{TV}(\mu_{h,j} \| \mu_{h,j}^0) \leq \sigma_i\}$$

**Design Motivation**: The $d$-rectangular structure ensures the robust action-value function retains a linear form $Q_{i,h}^{\pi,\sigma}(s,\boldsymbol{a}) = \langle \phi_{s\boldsymbol{a}}, w_{i,h} \rangle + \text{bonus}$, avoiding the completeness assumptions required in general function approximation.

### Key Design 2: Vanishing Minimum Assumption and Equivalent Optimization

**Function**: Avoid the support shift problem by assuming $\min_s V_{i,h}^{\pi,\sigma}(s) = 0$.

**Mechanism**: Under this assumption, the robust optimization problem simplifies to:
$$\inf_{P \in \mathcal{U}_{TV}^{\sigma_i}(P_{h,s,\boldsymbol{a}}^0)} \mathbb{E}_P[V] = \sigma_i \mathbb{E}_{\tilde{P}_{h,s,\boldsymbol{a}}}[V]$$

This guarantees that the support of the worst-case transition kernel does not exceed that of the nominal kernel (equation (12) in Proposition 4.3).

**Design Motivation**: This assumption is equivalent to adding an absorbing failure state $s_f$ (with zero reward) to the Markov game, which is natural in many practical scenarios (e.g., games where a player may fail at each step).

### Key Design 3: Agent-Specific Bonus Terms

**Function**: Design exploration bonuses to guide sufficient policy exploration.

**Mechanism**: The bonus for player $i$ is:
$$\Gamma_{h,k}^i(s,\boldsymbol{a}) = \beta_i \sum_{j=1}^d \sqrt{\phi_j(s,\boldsymbol{a}) \mathbf{1}_j^T (\Lambda_h^k)^{-1} \mathbf{1}_j \phi_j(s,\boldsymbol{a})}$$

where $\beta_i = \min\{H, 1/\sigma_i\} \sqrt{c_\beta n d \log(ndHK/\delta)}$.

**Design Motivation**: Unlike the single UCB bonus in non-robust settings, the robust setting requires $d$ coordinate-wise confidence upper bounds. Each coordinate corresponds to an independent ridge regression task targeting $[V_{i,h+1}^k(s_{h+1}^\tau)]_{\alpha_j}$. Furthermore, the factor $\min\{H, 1/\sigma_i\}$ in $\beta_i$ reflects each player's individual robustness preference.

### Key Design 4: Find-CCE Subroutine

**Function**: Address the instability (non-Lipschitz continuity) of CCE with respect to the $Q$-value matrix.

**Mechanism**: First locate the nearest surrogate game matrix $\tilde{Q}$ within an $\epsilon$-cover of the $Q$-value function class, then compute the exact CCE of $\tilde{Q}$. This ensures that for any two neighboring $Q$-values in the cover, the deviation in value functions when using the same CCE is bounded within $2\epsilon$.

**Design Motivation**: Directly computing CCE from $Q$ functions causes the covering number of the value function class to explode (Lemma 4.4 gives a counterexample showing that a small perturbation to $Q$ can cause a CCE value difference of 1). The "discretize first, then solve equilibrium" strategy resolves this technical difficulty.

### Loss & Training

The regret is defined as the sum over all players of the maximum unilateral deviation gain:
$$\text{Regret}(K) = \max_{i \in [n]} \sum_{k=1}^K [V_{i,1}^{\star, \pi_{-i}^k, \sigma}(s_1^k) - V_{i,1}^{\pi^k, \sigma}(s_1^k)]$$

## Key Experimental Results

### Main Results

In a simulated game with 5 states, 2 players, and $H=3$ (structured as in Figure 1 of the paper), DR-CCE-LSI is compared against the non-robust baseline NQOVI:

| Perturbation level $\rho$ | DR-CCE-LSI (Ours) | NQOVI |
|---|---|---|
| 0.0 (no shift) | Slightly lower (optimality sacrificed for robustness) | Better |
| 0.1 | Comparable | Begins to degrade |
| 0.2+ | **Significantly better** | Performance degrades sharply |

### Theoretical Comparison

| Setting | Method | Upper Bound | Lower Bound |
|---|---|---|---|
| Non-robust single-agent | he2023 | $\sqrt{d^2H^3K}$ | $\sqrt{d^2H^3K}$ |
| Non-robust multi-player | cisneros2023 | $\sqrt{d^3H^5K}$ | $\sqrt{d^2H^3K}$ |
| **Robust multi-player (Ours)** | DR-CCE-LSI | $dH\min\{H,1/\sigma\}\sqrt{K}$ | $dH^{1/2}\min\{H,1/\sigma\}\sqrt{K}$ |

### Key Findings

1. The proposed upper bound matches that of single-agent robust settings in $d$ and $K$, achieving **minimax optimality with respect to feature dimension $d$**.
2. The factor $\min\{H, 1/\min\{\sigma_i\}\}$ in the regret bound characterizes the cost of robustness.
3. The advantage of DR-CCE-LSI becomes more pronounced under higher environmental perturbation sensitivity.
4. The impossibility theorem (Theorem 4.1) establishes that without additional assumptions, the regret is $\Omega(\sigma \cdot HK)$ (linear growth, hence not learnable).

## Highlights & Insights

1. **First theoretical result**: The first sample-efficient algorithm for online robust linear Markov games.
2. **Complete picture of impossibility and feasibility**: The paper first proves a lower bound impossibility, then makes the problem tractable via the appropriate vanishing minimum assumption.
3. **Interesting observation on "risk preference consistency"**: Inconsistent robustness levels among players (large variance in $\sigma_i$) leads to reduced sample efficiency, suggesting that a shared risk awareness is beneficial in cooperative learning.
4. **Technical contribution**: The Find-CCE subroutine resolves the fundamental technical challenge of CCE instability with respect to $Q$-values.

## Limitations & Future Work

1. A gap of $H^{1/2}$ in the $H$-dependence between upper and lower bounds remains, potentially addressable via variance-weighted ridge regression.
2. The vanishing minimum assumption, while reasonable, restricts the scope of applicability.
3. Only TV-divergence-based $d$-rectangular uncertainty sets are considered; extensions to $\chi^2$ and KL divergences remain unexplored.
4. The feature mapping must satisfy non-degeneracy and positive definiteness conditions (Corollary 5.3), imposing additional requirements on feature design.
5. Experiments are conducted only in small-scale simulated environments, lacking empirical evaluation in more complex settings.
6. Robust games in the decentralized learning setting remain an open problem.

## Related Work & Insights

1. **Liu et al. (2024)**: Single-agent online robust linear MDP with upper bound $O(dH\min\{1/\sigma, H\}\sqrt{K})$; the present work generalizes this to the multi-player setting.
2. **Cisneros et al. (2023)**: The NQOVI algorithm for non-robust online linear games with upper bound $O(\sqrt{d^3H^5K})$.
3. **Jiao et al. (2024)**: Minimax-optimal algorithm for robust games in the generative model setting.
4. **Insight**: The additional challenges of robustness in game settings (equilibrium instability + support shift) provide a rich space for theoretical investigation.

## Rating

⭐⭐⭐⭐ (4/5)

**Strengths**: Novel problem formulation, complete theoretical analysis (impossibility + feasibility + minimax optimality), and solid technical contributions.

**Weaknesses**: The gap in $H$-dependence remains unresolved, experiments are overly simplistic, and empirical validation of the vanishing minimum assumption is insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?](../../ICLR2026/reinforcement_learning/is_pure_exploitation_sufficient_in_exogenous_mdps_with_linear_function_approxima.md)
- [\[NeurIPS 2025\] A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning](../../NeurIPS2025/reinforcement_learning/a_unifying_view_of_linear_function_approximation_in_offpolic.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](../../ICLR2026/reinforcement_learning/distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)
- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)
- [\[AAAI 2026\] TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents](towermind_a_tower_defence_game_learning_environment_and_benchmark_for_llm_as_age.md)

</div>

<!-- RELATED:END -->
