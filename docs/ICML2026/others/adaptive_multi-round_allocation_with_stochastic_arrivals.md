---
title: >-
  [Paper Note] Adaptive Multi-Round Allocation with Stochastic Arrivals
description: >-
  [ICML 2026][Adaptive recruitment] This work formalizes network recruitment as a budget-constrained sequential control problem, proves that the single-round optimal allocation is greedy…
tags:
  - "ICML 2026"
  - "Adaptive recruitment"
  - "multi-round allocation"
  - "stochastic arrivals"
  - "dynamic programming"
  - "population surrogate value function"
date: 2026-05-08
content_hash: 7ae6d4c0b777772a
---

# Adaptive Multi-Round Allocation with Stochastic Arrivals

**Conference**: ICML 2026  
**arXiv**: [2605.12111](https://arxiv.org/abs/2605.12111)  
**Code**: Publicly available  
**Area**: Sequential Decision Making / Budget-Constrained Optimization / Stochastic Control  
**Keywords**: Adaptive recruitment, multi-round allocation, stochastic arrivals, dynamic programming, population surrogate value function

## TL;DR
This work formalizes network recruitment as a budget-constrained sequential control problem, proves that the single-round optimal allocation is greedy; reduces multi-round planning to $O(b^5\log b)$ complexity via a population-level surrogate value function, and provides robustness guarantees under model misspecification by decomposing errors into frontier, population, and approximation types.

## Background & Motivation

**Background**  
Adaptive network recruitment is widely used in public health (HIV 95-95-95, contact tracing, survey sampling), epidemiology, and social sciences. Under resource scarcity, the core problem is how to allocate limited incentives (referral coupons, test kits) to maximize coverage and early recruitment.

**Limitations of Prior Work**
1. **Endogenous Dynamics**: Unlike stochastic knapsack or bandit problems, allocation here both consumes budget for immediate reward and alters the distribution of future decision opportunities via new recruitments, leading to complex state evolution.
2. **High-Dimensional Intractability**: Individual features are high-dimensional (demographics, network position, etc.), and exact value functions must track the entire frontier distribution, making computation infeasible.
3. **Optimal Planning Difficulty**: Even with fully known distributions, Bellman recursion involves infinite-dimensional continuous state spaces, so traditional DP cannot be directly applied.

**Key Challenge**  
The trade-off between fine-grained adaptive allocation based on individual features and solving for optimal strategies within limited computational budgets—balancing exact planning and scalability.

**Goal**  
Design computationally feasible strategies to maximize recruitment under multi-round budget constraints, robust to model misspecification.

**Key Insight**  
Start from the combinatorial structure of a single round: for a fixed round budget, use marginal decomposition of survival probabilities to derive greedy optimality; separate "within-round allocation" and "inter-round budgeting" as two subproblems. For multiple rounds, introduce a population-level surrogate value function to fold individual heterogeneity into group statistics.

**Core Idea**  
Single-round optimality (greedy) + population surrogate value function (state dimension reduction) → precise yet tractable multi-round DP. The surrogate Bellman equation is computed exactly via probability generating functions, with complexity $O(b^5\log b)$; robustness error is decomposed into frontier, population, and approximation components.

## Method

### Overall Architecture
At time $t\geq 1$, the system is in state $(r_t,\mathcal D_{1:n_t}^{(t)})$ (remaining budget $r_t$, frontier of $n_t$ individuals with arrival distributions). Policy $\pi$ selects each round: (i) round budget $s_t\in\{0,\ldots,r_t\}$, (ii) allocation vector $\mathbf k_t=(k_1,\ldots,k_{n_t})$, $\sum_i k_i\leq s_t$. Each individual $i$ is limited by arrival capacity $X_i\sim\mathcal D_i$, with actual recruitment $\min\{k_i,X_i\}$; newly recruited individuals enter the next round's frontier, with distributions sampled from population $\mathcal P$. The objective is $\max_\pi\mathbb E[\sum_{t\geq 1}\gamma^{t-1}N_t]$.

### Key Designs

1. **Single-Round Greedy Optimal Allocation**:

    - **Function**: Precisely solves the optimal allocation for a fixed round budget and frontier.
    - **Mechanism**: Marginal decomposition $\mathbb E[\sum_i\min\{k_i,X_i\}]=\sum_i\sum_{\ell=1}^{k_i}p_i(\ell)$ rewrites the objective as a sum of survival probabilities—discrete concave, with diminishing marginal returns. Greedy: allocate in order of highest marginal gain per unit (Theorem 4.2 proves optimality).
    - **Design Motivation**: Leverages discrete concavity to avoid combinatorial explosion; marginal decomposition is intuitive and computable using only survival probabilities.

2. **Population-Level Surrogate Value Function**:

    - **Function**: Reduces high-dimensional individual state to one-dimensional population statistics, making multi-round DP tractable.
    - **Mechanism**: Defines $U_{\mathcal P}(r,n)$ as the optimal expected recruitment with remaining budget $r$ and size $n$ (individuals i.i.d. from $\mathcal P$), greatly reducing the state space compared to $V_{\mathcal P}(r,\mathcal D_{1:n})$. Recursion: $U_{\mathcal P}(r,n)=\max_{0\leq s\leq r}\mathbb E[N_s^e+\gamma U_{\mathcal P}(r-s,N_s^e)]$, where $N_s^e$ is the expected recruitment from uniformly allocating $s$ to $n$ individuals. Proposition 6.1 proves uniform allocation is optimal under the population model (exchangeability + diminishing marginal returns).
    - **Design Motivation**: Future new individuals are ex ante indistinguishable (all from $\mathcal P$), so treating them identically is optimal; this abstraction captures the essence of planning while discarding irrelevant details.

3. **Generating Function Computation + Improved Bellman Operator**:

    - **Function**: Precisely computes the recursion for $U_{\mathcal P}(r,n)$, plugging the surrogate value function back into the original Bellman equation while maintaining within-round optimality.
    - **Mechanism**: Uses the truncated probability generating function of population survival probability $\bar p(\ell)$ to describe the distribution of $N_s^e$, avoiding discrete convolution enumeration, with $O(b^2)$ space and $O(b^5\log b)$ time complexity (Theorem 6.2). For each actual frontier $\mathcal D_{1:n}$, greedy allocation $\mathbf k^{\text{greedy}}$ yields $N_s^g$ for this round, and future expectation uses $U_{\mathcal P}(r-s,N_s^g)$ in place of exact $V$, forming the modified Bellman operator $\widetilde V_{\mathcal P;U_{\mathcal P}}$.
    - **Design Motivation**: Generating functions use polynomial arithmetic for fast computation; surrogate insertion is a principled form of value function approximation, retaining round-level optimality and multi-round tractability.

### Loss & Training
Objective: $\sum_{t\geq 1}\gamma^{t-1} N_t$. Multi-round error decomposition (Theorem 7.2): under estimation noise, suboptimality $\leq 2(1+\gamma)r\sum_i\|\mathcal D_i-\hat{\mathcal D}_i\|_{\text{TV}}+c_{r,\gamma}\|\mathcal P-\hat{\mathcal P}\|_{\text{TV}}+c_{r,\gamma}r\mathbb E\|\mathcal D-\bar{\mathcal D}\|_{\text{TV}}$, $c_{r,\gamma}=2\gamma r/(1-\gamma)$. These correspond to frontier error, population distribution error, and surrogate approximation error, respectively.

## Key Experimental Results

### Main Results (ICPSR HIV Social Network, Simulated RDS)

| Initial Frontier | $\gamma$ | Total Budget $b$ | Const(k=3) | Greedy(α=0.5) | GreedyRem | TAP (Ours) |
|------------------|----------|------------------|------------|---------------|-----------|------------|
| n=5 | 0.5 | 200 | 32.1 | 35.4 | 36.2 | **39.8** |
| n=5 | 0.7 | 200 | 28.3 | 31.1 | 31.7 | **34.5** |
| n=5 | 0.9 | 200 | 24.1 | 26.8 | 27.3 | **29.1** |
| n=10 | 0.5 | 200 | 58.2 | 62.1 | 63.5 | **68.3** |
| n=10 | 0.7 | 200 | 51.4 | 55.3 | 56.4 | **61.2** |
| n=15 | 0.5 | 200 | 79.5 | 85.3 | 87.1 | **94.2** |
| n=15 | 0.9 | 200 | 42.7 | 46.5 | 47.2 | **51.8** |

Const(k): fixed allocation of $k$ per person (tuned post hoc); Greedy methods use fixed/remaining budget ratios without cross-round planning. TAP combines greedy within-round allocation and population-level multi-round planning.

### Simulation vs Real Network

| Setting | Method | HIV | Chlamydia | Gonorrhea |
|---------|--------|-----|-----------|-----------|
| Simulated Dist. | TAP | **68.3** | 72.1 | 65.4 |
| Simulated Dist. | Const(3) | 58.2 | 63.5 | 58.1 |
| Real Network | TAP | **67.5** | 71.2 | 64.8 |
| Real Network | Const(3) | 57.1 | 62.8 | 57.3 |

Simulation and real results are close, validating the effectiveness of learning $\mathcal P$; in some cases (Gonorrhea, $\gamma=0.9$), greedy is better, indicating robustness to model error remains a challenge.

### Ablation Study

| Component | Change | Avg. Recruitment | Note |
|-----------|--------|------------------|------|
| Full TAP | - | 68.3 | Baseline |
| Remove Multi-Round Planning | Fixed round budget 0.5x | 62.1 | No cross-round optimization |
| Remove Population Surrogate | Enumerate all frontier configs | 68.1 | Computationally expensive, not scalable |
| No Greedy Within-Round | Random within-round + population planning | 55.3 | Within-round optimality is key |
| Uniform Baseline | Equal allocation to all | 51.2 | Ignores heterogeneity |

### Key Findings
- **Both greedy within-round and multi-round planning are necessary**: Removing either component significantly underperforms full TAP.
- **Population surrogate is nearly lossless**: Compared to enumerating all frontiers (68.1), TAP (68.3) is even slightly better—surrogate avoids overfitting to specific configurations.
- **Robustness on real networks**: Simulation vs real gap < 2 recruits, validating model transferability to real data.
- **Baselines outperform in some settings**: For Gonorrhea + high discount, greedy wins, revealing model error is not fully eliminated.

## Highlights & Insights
- **Elegance of greedy single-round optimality**: Survival probability decomposition transforms complex stochastic constraints into a discrete concave objective, a refined improvement over the stochastic knapsack problem.
- **Creativity of population surrogate**: Converts the "new individuals ex ante from the same distribution" modeling assumption into a state reduction tool, both theoretically sound and practically relevant.
- **Transparent error decomposition**: Theorem 7.2 clearly separates three error types, giving practitioners clear guidance on which input precision is most critical.
- **Validation on real networks**: HIV network application demonstrates the practical value of this framework in public health.

## Limitations & Future Work
- **Scalability**: $O(b^5\log b)$ is still high for large budgets, requiring further approximation or heuristics.
- **Model error challenge**: In some disease + discount settings, greedy is better, suggesting that adaptive strategies for model learning may be more valuable.
- **Data availability**: Assumes access to arrival distributions or sufficient statistics; for emerging infectious diseases, historical data may be lacking.

## Related Work & Insights
- **vs Stochastic Knapsack / Bandit**: Classic problems have fixed action sets; here, the action space evolves endogenously, requiring cross-round dynamics.
- **vs Prophet Inequality**: Prophet inequalities assume independent candidates; here, recruitment generates correlated future candidates, with more complex dependency structures.
- **vs Heuristic RDS Methods**: In practice, RDS often uses fixed per-round allocation; this work provides a theoretical improvement via adaptive multi-round planning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of greedy single-round and population surrogate value function forms a novel, tractable multi-round planning framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Real HIV network + two other diseases + simulation/real comparison + multiple baselines + thorough ablation.
- Writing Quality: ⭐⭐⭐⭐ Problem formalization is clear, main algorithms and theorems are rigorously presented.
- Value: ⭐⭐⭐⭐ Directly valuable for adaptive network recruitment and public health, tightly integrating theory and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Generalist to Specialist Representation](from_generalist_to_specialist_representation.md)
- [\[ICML 2026\] Local and Mixing-Based Algorithms for Gaussian Graphical Model Selection from Glauber Dynamics](local_and_mixing-based_algorithms_for_gaussian_graphical_model_selection_from_gl.md)
- [\[ICML 2026\] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)
- [\[ICML 2026\] Complexity as Advantage: A Regret-Based Perspective on Emergent Structure](complexity_as_advantage_a_regret-based_perspective_on_emergent_structure.md)
- [\[ICML 2026\] Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)

</div>

<!-- RELATED:END -->
