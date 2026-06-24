---
title: >-
  [Paper Note] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis
description: >-
  [NeurIPS 2025][Reinforcement Learning][Stackelberg games] This paper proposes AC-SMFG, the first single-loop Actor-Critic algorithm with non-asymptotic convergence guarantees for solving Stackelberg Mean Field Games (SMFGs), achieving a convergence rate of $\widetilde{\mathcal{O}}(k^{-1/2})$.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Stackelberg games"
  - "mean field games"
  - "Actor-Critic"
  - "non-asymptotic convergence"
  - "bilevel optimization"
date: 2026-05-08
content_hash: 58829b5dfc33b4cf
---

# Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis

**Conference**: NeurIPS 2025
**arXiv**: [2509.15392](https://arxiv.org/abs/2509.15392)  
**Code**: Available (provided in supplementary material)  
**Area**: Reinforcement Learning
**Keywords**: Stackelberg games, mean field games, Actor-Critic, non-asymptotic convergence, bilevel optimization

## TL;DR

This paper proposes AC-SMFG, the first single-loop Actor-Critic algorithm with non-asymptotic convergence guarantees for solving Stackelberg Mean Field Games (SMFGs), achieving a convergence rate of $\widetilde{\mathcal{O}}(k^{-1/2})$.

## Background & Motivation

**Background**: Mean field games (MFGs) provide a modeling framework for strategic interactions among infinitely many rational agents, with broad applications in resource allocation, telecommunications, and power system optimization. Stackelberg MFGs further introduce a hierarchical structure in which a single leader agent influences a follower population.

**Limitations of Prior Work**: Existing SMFG algorithms suffer from three key issues: (i) overly strong independence assumptions between the leader and follower objectives; (ii) nested-loop structures that lead to poor sample efficiency; and (iii) lack of finite-time convergence guarantees, with only asymptotic convergence established.

**Key Challenge**: SMFGs are inherently bilevel optimization problems — the leader's objective depends on the equilibrium response of the follower population, which is itself influenced by the leader's policy. This coupling makes algorithm design and analysis extremely challenging.

**Goal**: To design a simple, practical, single-loop algorithm with non-asymptotic convergence guarantees for solving SMFGs.

**Key Insight**: The SMFG is formulated as a bilevel optimization problem, and a multi-timescale analysis is employed to coordinate updates of the leader policy, follower policy, and mean field.

**Core Idea**: A "gradient alignment" condition is introduced to relax the leader-follower independence assumption; the Polyak-Łojasiewicz (PL) condition is leveraged to handle the non-convex lower-level problem, yielding a convergence rate superior to that of standard bilevel optimization.

## Method

### Overall Architecture

AC-SMFG is a single-loop Actor-Critic algorithm. The SMFG is modeled as a tuple $({\cal S},{\cal A},{\cal B},{\cal P},r_f,r_l,\gamma)$. The leader and follower maintain policy parameters $\omega$ and $\theta$, respectively, using softmax parameterization:

$$\phi_\omega(b|s) = \frac{\exp(\omega(s,b))}{\sum_{b'}\exp(\omega(s,b'))}, \quad \pi_\theta(a|s) = \frac{\exp(\theta(s,a))}{\sum_{a'}\exp(\theta(s,a'))}$$

### Key Designs

**Module 1: Leader Policy Update (Slowest Timescale)**
- Function: Updates the leader policy parameter $\omega$ to maximize the Stackelberg objective $\Phi(\phi) = J_l(\phi, \mu^*(\phi))$
- Core update: $\omega_{k+1} = \omega_k + \zeta_k \nabla_\omega \log\phi_{\omega_k}(b_k|s_k)(r_l(s_k,b_k,\hat{\mu}_k) + \gamma \hat{V}_{l,k}(s_{k+1}) - \hat{V}_{l,k}(s_k))$
- Design Motivation: A policy gradient approach is used with the smallest step size $\zeta_k$, ensuring the leader is updated only after the follower and mean field have converged.

**Module 2: Follower Policy Update (Intermediate Timescale)**
- Function: Updates the follower policy parameter $\theta$ to maximize the regularized cumulative reward $J_f(\pi,\phi,\mu)$
- Core update: A policy gradient update of analogous form, with step size $\alpha_k > \zeta_k$
- Design Motivation: The follower must converge faster than the leader to approximate the best response.

**Module 3: Mean Field Update (Fast Timescale)**
- Function: Tracks the aggregate behavior $\mu$ of the follower population
- Core update: $\hat{\mu}_{k+1} = \Pi_{\Delta_{\cal S}}(\hat{\mu}_k + \xi_k(e_{\bar{s}_k} - \hat{\mu}_k))$
- Design Motivation: The stationary distribution is estimated via a second sample trajectory.

**Module 4: Value Function (Critic) Update (Fastest Timescale)**
- Function: Estimates value functions for both the leader and follower via TD learning
- The step size $\beta_k$ is the largest, ensuring value function estimates converge first.

### Loss & Training

- A four-timescale design is adopted: $\zeta_k \leq \xi_k \leq \alpha_k \leq \beta_k$, with all step sizes set to $\frac{c}{(k+1)^{1/2}}$
- Entropy regularization with weight $\tau$ ensures that Assumption 4 (contraction condition) holds
- Each iteration requires only two samples (one for the occupancy measure, one for the stationary distribution)

## Key Experimental Results

### Main Results

| Environment | AC-SMFG | OneByOne | CuiKoeppl | ADAGE |
|------|---------|----------|-----------|-------|
| Market Entrance (sell skew) | **Fastest convergence** | Slow convergence | Oscillation / suboptimal | Moderate |
| Shop Positioning ($c=0.05$) | **Fastest leader convergence** | Slower | Slower | Unstable follower |
| Equilibrium Pricing ($i=5$) | **Significantly outperforms other SMFGs** | Poor convergence | Poor convergence | Slightly better (function approximation) |

### Ablation Study

| Analysis | Conclusion |
|--------|------|
| Different leader configurations | AC-SMFG converges to an appropriate MFE across all configurations |
| Continuous state/action spaces | Compatible with continuous spaces via function approximation (Gaussian mean field assumption) |
| Convergence rate comparison | $\widetilde{\mathcal{O}}(k^{-1/2})$ outperforms hong2023two's $\widetilde{\mathcal{O}}(k^{-2/5})$ |

### Key Findings

1. AC-SMFG demonstrates substantially faster leader convergence and smoother convergence trajectories across all environments.
2. The improvement of AC-SMFG is more pronounced in environments where follower behavior is more strongly influenced by the leader.
3. Continuous-space experiments confirm that AC-SMFG is compatible with function approximation (neural network policies and Gaussian mean fields).

## Highlights & Insights

1. **First Non-Asymptotic Guarantee**: This is the first algorithm in the SMFG literature to provide finite-time and finite-sample complexity guarantees.
2. **Beyond Bilevel Optimization**: A better convergence rate is achieved under a more difficult setting (non-convex lower level) than standard bilevel optimization (strongly convex lower level).
3. **Technical Innovation**: Novel analytical techniques are developed for the lower level under the PL condition (rather than strong convexity), which generalize to broader bilevel optimization settings.
4. **Gradient Alignment Condition**: A weaker assumption than the existing "leader-follower independence" assumption is proposed, accommodating a wider range of coupling scenarios.

## Limitations & Future Work

1. The approach relies on a sufficiently large regularization weight $\tau$ and therefore cannot solve the equilibrium of the original unregularized problem — a limitation shared across the regularized MFG literature.
2. Convergence is to a stationary point rather than a global optimum, since the composite function $\Phi$ does not satisfy the gradient dominance condition.
3. The homogeneity assumption on followers limits applicability to practical scenarios.
4. As an asymptotic approximation of $N$-agent Markov games, MFGs may incur non-negligible errors when the number of followers is small.

## Related Work & Insights

- **Comparison with cui2024learning**: That work employs a nested-loop virtual play approach with only asymptotic convergence; the present paper uses a single-loop approach with non-asymptotic guarantees.
- **Comparison with hong2023two**: A standard bilevel optimization method achieving $\widetilde{\mathcal{O}}(k^{-2/5})$ under strong convexity of the lower level; the present paper achieves $\widetilde{\mathcal{O}}(k^{-1/2})$ under weaker assumptions.
- **Insight**: The analytical techniques for bilevel optimization under the PL condition are transferable to more general hierarchical decision-making problems.

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical contributions are rigorous, providing the first non-asymptotic guarantees for SMFGs, with independently valuable technical innovations in handling the PL condition. The experimental settings adequately demonstrate the advantages of the proposed method. The primary limitations lie in the regularization and homogeneity assumptions, which restrict the direct applicability of the approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics](solving_continuous_mean_field_games_deep_reinforcement_learning_for_non-stationa.md)
- [\[NeurIPS 2025\] Last Iterate Convergence in Monotone Mean Field Games](last_iterate_convergence_in_monotone_mean_field_games.md)
- [\[NeurIPS 2025\] Non-convex Entropic Mean-Field Optimization via Best Response Flow](non-convex_entropic_mean-field_optimization_via_best_response_flow.md)
- [\[NeurIPS 2025\] Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
