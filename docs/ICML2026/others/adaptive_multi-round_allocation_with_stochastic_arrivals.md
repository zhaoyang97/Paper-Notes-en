---
title: >-
  [Paper Note] Adaptive Multi-Round Allocation with Stochastic Arrivals
description: >-
  [ICML 2026][Adaptive Recruiting] This paper formalizes network recruitment as a budget-constrained sequential control problem and proves that single-round optimal allocation is greedy. It reduces multi-round planning dim…
tags:
  - "ICML 2026"
  - "Adaptive Recruiting"
  - "Multi-round Allocation"
  - "Stochastic Arrivals"
  - "Dynamic Programming"
  - "Population Proxy Value Function"
date: 2026-05-08
content_hash: aa6a080786bfbd3f
---

# Adaptive Multi-Round Allocation with Stochastic Arrivals

**Conference**: ICML 2026  
**arXiv**: [2605.12111](https://arxiv.org/abs/2605.12111)  
**Code**: Publicly available  
**Area**: Sequential Decision Making / Budget-Constrained Optimization / Stochastic Control  
**Keywords**: Adaptive Recruiting, Multi-round Allocation, Stochastic Arrivals, Dynamic Programming, Population Proxy Value Function

## TL;DR
This paper formalizes network recruitment as a budget-constrained sequential control problem and proves that single-round optimal allocation is greedy. It reduces multi-round planning dimensionality to $O(b^5\log b)$ complexity via a population-level proxy value function and provides robustness guarantees by decomposing model error into frontier, population, and approximation terms.

## Background & Motivation

**Background**
Adaptive network recruitment is widely applied in public health (HIV 95-95-95, contact tracing, survey sampling), epidemiology, and social sciences. Under resource scarcity, the core problem is how to allocate limited incentives (referral coupons, testing kits) to maximize participation coverage and early recruitment.

**Limitations of Prior Work**
1. **Endogenous Dynamics**: Unlike stochastic knapsack or bandits, allocations in this problem both consume budget for immediate rewards and change the distribution of future decision opportunities by recruiting new individuals, forming complex state evolution.
2. **High-Dimensional Intractability**: Individual features are high-dimensional (demographics, network position, etc.), and exact value functions must track the entire frontier distribution, making calculation infeasible.
3. **Optimal Planning Difficulty**: Even if the distribution is fully known, Bellman recursion involves infinite-dimensional continuous state spaces, meaning traditional DP cannot be directly applied.

**Key Challenge**
The trade-off between performing fine-grained adaptive allocation based on individual features and solving the optimal policy within a limited computational budget—scaling vs. precision.

**Goal**
Design computable strategies to maximize recruitment under multi-round budget constraints, while remaining robust to model errors.

**Key Insight**
The study starts with the single-round combinatorial structure: for a fixed round budget, greedy optimality is derived through the marginal decomposition of survival probabilities. This decouples "intra-round allocation" from "inter-round budgeting." For multiple rounds, a population-level proxy value function is introduced to collapse individual heterogeneity into group statistics.

**Core Idea**
Single-round optimal (greedy) + Population proxy value function (state dimension reduction) $\rightarrow$ exact but computable multi-round DP. The proxy Bellman equation is precisely computed via probability generating functions with $O(b^5\log b)$ complexity. The robustness error is decomposed into frontier-level, population-level, and approximation terms.

## Method

### Overall Architecture
At time $t\geq 1$, the system is in state $(r_t, \mathcal D_{1:n_t}^{(t)})$ (remaining budget $r_t$, arrival distribution of $n_t$ individuals in the frontier). The policy $\pi$ chooses for each round: (i) round budget $s_t\in\{0,\ldots,r_t\}$, (ii) allocation vector $\mathbf k_t=(k_1,\ldots,k_{n_t})$, where $\sum_i k_i\leq s_t$. Individual $i$ is limited by their arrival capacity $X_i\sim\mathcal D_i$, with actual recruitment $\min\{k_i,X_i\}$. New recruits enter the next-round frontier with distributions sampled from population $\mathcal P$. The goal is $\max_\pi\mathbb E[\sum_{t\geq 1}\gamma^{t-1}N_t]$.

### Key Designs

1. **Optimal Greedy Intra-round Allocation**:

    - **Function**: Precisely solve for the optimal allocation given a fixed round budget and frontier.
    - **Mechanism**: Marginal decomposition $\mathbb E[\sum_i\min\{k_i,X_i\}]=\sum_i\sum_{\ell=1}^{k_i}p_i(\ell)$ expresses the objective as a sum of survival probabilities—discrete concave with diminishing returns. Greedy: Allocate by sorting every unit by the highest marginal return (Theorem 4.2 proves optimality).
    - **Design Motivation**: Utilize discrete concave structures to avoid combinatorial explosion; marginal decomposition is intuitive and only requires survival probabilities for calculation.

2. **Population-Level Proxy Value Function**:

    - **Function**: Reduce high-dimensional individual states to one-dimensional population statistics to make multi-round DP computable.
    - **Mechanism**: Define $U_{\mathcal P}(r,n)$ as the optimal expected recruitment under remaining budget $r$ and size $n$ (individuals i.i.d. from $\mathcal P$). This significantly reduces the state space compared to $V_{\mathcal P}(r,\mathcal D_{1:n})$. Recursion: $U_{\mathcal P}(r,n)=\max_{0\leq s\leq r}\mathbb E[N_s^e+\gamma U_{\mathcal P}(r-s,N_s^e)]$, where $N_s^e$ is the expected recruitment from uniform allocation of $s$ to $n$ individuals. Proposition 6.1 proves uniform allocation is optimal under the population model (exchangeability + diminishing margins).
    - **Design Motivation**: Future individuals are ex ante indistinguishable (all from population $\mathcal P$), so uniform treatment is optimal; this abstraction captures the essence of planning while discarding irrelevant details.

3. **Generating Function Computation + Modified Bellman Operator**:

    - **Function**: Precisely compute the $U_{\mathcal P}(r,n)$ recursion and plug the proxy value back into the original Bellman equation while preserving intra-round optimality.
    - **Mechanism**: Describe the distribution of $N_s^e$ using truncated probability generating functions of the population survival probability $\bar p(\ell)$, avoiding discrete convolution enumeration, with $O(b^2)$ space and $O(b^5\log b)$ time complexity (Theorem 6.2). For each actual frontier $\mathcal D_{1:n}$, use greedy allocation $\mathbf k^{\text{greedy}}$ to obtain the current $N_s^g$, while future expectations replace exact $V$ with $U_{\mathcal P}(r-s,N_s^g)$, forming the modified Bellman operator $\widetilde V_{\mathcal P;U_{\mathcal P}}$.
    - **Design Motivation**: Generating functions leverage polynomial arithmetic for fast calculation; proxy insertion is a principled form of value function approximation that retains round-level optimality and multi-round computability.

### Loss & Training
Objective: $\sum_{t\geq 1}\gamma^{t-1} N_t$. Multi-round error decomposition (Theorem 7.2): suboptimality under estimation noise $\leq 2(1+\gamma)r\sum_i\|\mathcal D_i-\hat{\mathcal D}_i\|_{\text{TV}}+c_{r,\gamma}\|\mathcal P-\hat{\mathcal P}\|_{\text{TV}}+c_{r,\gamma}r\mathbb E\|\mathcal D-\bar{\mathcal D}\|_{\text{TV}}$, where $c_{r,\gamma}=2\gamma r/(1-\gamma)$. These correspond to frontier error, population distribution error, and proxy approximation error respectively.

## Key Experimental Results

### Main Results (ICPSR HIV Social Network, simulated RDS)

| Initial Frontier | $\gamma$ | Total Budget $b$ | Const(k=3) | Greedy(α=0.5) | GreedyRem | TAP (Ours) |
|---------|----------|----------|-----------|--------------|-----------|--------|
| n=5 | 0.5 | 200 | 32.1 | 35.4 | 36.2 | **39.8** |
| n=5 | 0.7 | 200 | 28.3 | 31.1 | 31.7 | **34.5** |
| n=5 | 0.9 | 200 | 24.1 | 26.8 | 27.3 | **29.1** |
| n=10 | 0.5 | 200 | 58.2 | 62.1 | 63.5 | **68.3** |
| n=10 | 0.7 | 200 | 51.4 | 55.3 | 56.4 | **61.2** |
| n=15 | 0.5 | 200 | 79.5 | 85.3 | 87.1 | **94.2** |
| n=15 | 0.9 | 200 | 42.7 | 46.5 | 47.2 | **51.8** |

Const(k) allocates a fixed amount $k$ per person (best-tuned post-hoc), while Greedy methods use fixed/remaining budget ratios without cross-round planning. TAP combines greedy intra-round allocation with population-level multi-round planning.

### Simulation vs. Real Network

| Setup | Method | HIV | Chlamydia | Gonorrhea |
|------|------|-----|-----------|-----------|
| Simulated Distribution | TAP | **68.3** | 72.1 | 65.4 |
| Simulated Distribution | Const(3) | 58.2 | 63.5 | 58.1 |
| Real Network | TAP | **67.5** | 71.2 | 64.8 |
| Real Network | Const(3) | 57.1 | 62.8 | 57.3 |

The proximity of simulated and real results validates the effectiveness of $\mathcal P$ learning. In specific cases (e.g., Gonorrhea, $\gamma=0.9$), greedy performs better, highlighting that robustness under model error remains a genuine challenge.

### Ablation Study

| Component | Alteration | Avg. Recruitment | Description |
|------|------|---------|------|
| Full TAP | - | 68.3 | Baseline |
| No Multi-round Planning | Fixed round budget 0.5x | 62.1 | No cross-round optimization |
| No Population Proxy | Enumerate all frontier configs | 68.1 | Computationally expensive, not scalable |
| No Greedy Intra-round | Random intra-round + Population planning | 55.3 | Intra-round optimality is critical |
| Uniform Baseline | Same quantity for everyone | 51.2 | Heterogeneity ignored |

### Key Findings
- **Both Greedy Intra-round and Multi-round Planning are Mandatory**: Retaining only one component results in significantly worse performance than full TAP.
- **Population Proxy is Nearly Lossless**: Compared to enumerating frontiers (68.1), TAP (68.3) is slightly better—proxies avoid overfitting to specific configurations.
- **Robustness on Real Networks**: The simulation-real gap is $< 2$ recruits, validating the feasibility of transferring the model to real data.
- **Baselines Win in Some Settings**: Greedy wins in the Gonorrhea + high discount setting, revealing that model error has not been completely eliminated.

## Highlights & Insights
- **Elegance of Greedy Single-Round Optimality**: Survival probability decomposition transforms complex stochastic constraints into a discrete concave objective, providing a refined improvement over the stochastic knapsack problem.
- **Creativity of Population Proxy**: Converting the modeling assumption that "new individuals are ex ante from the same distribution" into a state reduction tool is theoretically grounded and practical.
- **Transparent Error Decomposition**: Theorem 7.2 clearly separates three types of errors, showing practitioners exactly which input accuracies are most sensitive.
- **Real-World Network Validation**: Application to the HIV network demonstrates the practical value of this framework in public health.

## Limitations & Future Work
- **Scalability Issues**: $O(b^5\log b)$ remains high for large budgets, requiring further approximation or heuristics.
- **Model Error Challenges**: In some disease + discount combinations, greedy is superior, suggesting that adaptive strategies through model learning might be more valuable.
- **Data Availability**: The assumption of access to arrival distributions or sufficient statistics may not hold; historical data may be insufficient in scenarios like emerging infectious diseases.

## Related Work & Insights
- **vs. Stochastic Knapsack / Bandit**: In classic problems, the action set is static; here, the action space evolves endogenously, requiring consideration of cross-round dynamics.
- **vs. Prophet Inequalities**: Prophet inequalities assume candidates are unrelated; here, recruitment generates related future candidates, creating more complex dependency structures.
- **vs. Heuristic RDS Methods**: Practically, Respondent-Driven Sampling (RDS) often uses fixed per-round allocations; this work provides a theoretical improvement for adaptive multi-round planning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of greedy single-round and population proxy value functions forms a novel and computable multi-round planning framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Real HIV network + two other infectious diseases + simulation/real comparison + multiple baselines + comprehensive ablation.
- Writing Quality: ⭐⭐⭐⭐ The problem formalization is clear, and main algorithms and theorems are rigorously stated.
- Value: ⭐⭐⭐⭐ Direct value for adaptive network recruitment and public health scenarios, with a strong link between theory and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Envy-Free Allocation of Indivisible Goods via Noisy Queries](envy-free_allocation_of_indivisible_goods_via_noisy_queries.md)
- [\[AAAI 2026\] Center-Outward q-Dominance: A Sample-Computable Proxy for Strong Stochastic Dominance in Multi-Objective Optimisation](../../AAAI2026/others/center-outward_q-dominance_a_sample-computable_proxy_for_strong_stochastic_domin.md)
- [\[ICML 2026\] Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate](theoretical_analysis_of_sparse_optimization_with_reparameterization_weight_decay.md)
- [\[AAAI 2026\] Online Linear Regression with Paid Stochastic Features](../../AAAI2026/others/online_linear_regression_with_paid_stochastic_features.md)
- [\[ICML 2026\] Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems](mapping_human_anti-collusion_mechanisms_to_multi-agent_ai_systems.md)

</div>

<!-- RELATED:END -->
