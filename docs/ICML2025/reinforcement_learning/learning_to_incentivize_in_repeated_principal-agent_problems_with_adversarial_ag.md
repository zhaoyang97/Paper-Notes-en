---
title: >-
  [Paper Note] Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals
description: >-
  [ICML2025][Reinforcement Learning][principal-agent] This paper is the first to study repeated principal-agent problems with adversarial agent arrivals. It provides tight upper and lower regret bounds under both greedy and smooth response models, with the core idea of reducing the incentive design problem to adversarial linear bandits.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "principal-agent"
  - "adversarial bandits"
  - "incentive design"
  - "regret bound"
  - "mechanism design"
date: 2026-05-08
content_hash: 771b52a129f1149b
---

# Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals

**Conference**: ICML2025  
**arXiv**: [2505.23124](https://arxiv.org/abs/2505.23124)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: principal-agent, adversarial bandits, incentive design, regret bound, mechanism design

## TL;DR

This paper is the first to study repeated principal-agent problems with adversarial agent arrivals. It provides tight upper and lower regret bounds under both greedy and smooth response models, with the core idea of reducing the incentive design problem to adversarial linear bandits.

## Background & Motivation

The repeated principal-agent model characterizes a class of sequential decision-making scenarios where a principal guides an agent's actions through incentives. This model is widely used in e-commerce discounts, insurance contracts, crowdsourcing pricing, etc. Existing works either assume that the principal repeatedly faces the same type of agent, or that agent types are randomly sampled from a fixed distribution. However, in reality (e.g., the response sequence of online shoppers, herding behavior of crowdsourced workers), the arrival sequence of agents is often non-random.

This paper is the first to study the **adversarial agent arrivals** setting: over $T$ interaction rounds, $K \geq 2$ types of agents arrive in an adversarial order, and the principal does not know the arriving agent's type in advance at each round. In each round, the principal selects an incentive vector $\pi_t \in \mathcal{D} \subseteq [0,1]^N$. The agent chooses one of the $N$ arms according to their preferences and the incentive, and the principal obtains utility $U(\pi_t, j_t) = v_{a(\pi_t,j_t)} - \pi_{t,a(\pi_t,j_t)}$ (reward minus incentive cost). The goal is to minimize the regret against the optimal fixed incentive:

$$R_T = \sup_{\pi \in \mathcal{D}} \mathbb{E}\left[\sum_{t=1}^T \big(U(\pi, j_t) - U(\pi_t, j_t)\big)\right]$$

## Method

### Two Agent Response Models

**Greedy Choice Model**: The agent deterministically chooses the arm that maximizes their utility $b(\pi,j) \in \arg\max_{i} (\mu_i^j + \pi_i)$. In this model, tiny changes in incentives can lead to drastic jumps in the agent's decisions.

**Smooth Choice Model**: The agent's choice probability is Lipschitz continuous with respect to the incentive:

$$\sum_{i=1}^N \big|\Pr[a(\pi,j)=i] - \Pr[a(\pi',j)=i]\big| \leq L \cdot \|\pi - \pi'\|_\infty$$

This can be viewed as the agent adding smoothing noise to the preference vector (e.g., Gumbel $\to$ Logit model, Gaussian $\to$ Lipschitz model).

### Greedy Model — Impossibility Results

**Theorem 2.1**: If the principal does not know the agent's best response function $b(\cdot,\cdot)$, any algorithm yields $R_T = \Omega(T)$ (linear regret) when $K=2, N=3$. The intuition is that precise learning of the optimal incentive value $\Delta$ is impossible.

### Greedy Model + Known Response — Discretization + Linear Bandit Reduction

**Mechanism**:

1. **Discretization**: For each arm $i$ and agent type $j$, construct the minimum incentive $\pi^{i,j}$ that exactly induces agent $j$ to choose arm $i$, yielding a finite incentive set of size $|\Pi| = O(NK)$. Equivalent incentives are further merged via a mapping $h: \Pi \to \{0,1\}^K$, reducing the size to $|\hat\Pi| \leq \min(2KN, 2^K) + 1$.

2. **Reduction to Adversarial Linear Bandits**: Map each incentive $\pi$ to a $K$-dimensional vector $z^\pi$ (whose $j$-th component is $U(\pi,j)$), with the reward vector in each round being $y_t = e_{j_t}$:

$$U(\pi, j_t) = \langle z^\pi, y_t \rangle$$

Then, directly solve it using the Exp3-for-linear-bandits algorithm.

### Greedy Model + General Incentives

When the principal can incentivize multiple arms simultaneously, the decision space is $\mathcal{D} = [0,1]^N$. By identifying performance-equivalent polytopes $\mathcal{P}_\sigma$ and their extreme points for discretization, and combining this with covering number techniques, the size of the arm set is reduced to $(6KT)^K$.

### Smooth Model — Uniform Grid + Tsallis-INF

Perform $\epsilon$-grid discretization on single-arm incentives to obtain $N/\epsilon$ arms, and run Tsallis-INF. The discretization error is controlled by the Lipschitz constant:

$$\mathbb{E}[U(\pi^*, j)] - \mathbb{E}[U(\hat\pi, j)] \leq (2L+1)\epsilon$$

The total regret is $O(\sqrt{N\epsilon^{-1}T} + T(2L+1)\epsilon)$. Setting $\epsilon = N^{1/3}(2L+1)^{-2/3}T^{-1/3}$ yields the optimal bound.

## Summary of Theoretical Results

| Agent Behavior | Incentive Type | Upper Bound | Lower Bound |
|:---|:---|:---|:---|
| Unknown Greedy | Single-arm | N/A | $\Omega(T)$ |
| Known Greedy | Single-arm | $\tilde{O}(\min\{\sqrt{KT\log N},\, K\sqrt{T}\})$ | $\tilde{\Omega}(\min\{\sqrt{KT\log N},\, K\sqrt{T}\})$ |
| Known Greedy | General | $\tilde{O}(K\sqrt{T\log(KT)})$ | — |
| Unknown Smooth | Single-arm | $\tilde{O}(L^{1/3}N^{1/3}T^{2/3})$ | $\Omega(L^{1/3}N^{1/3}T^{2/3})$ |
| Unknown Smooth | General | $\tilde{O}(L^{N/(N+2)}T^{(N+1)/(N+2)})$ | — |

**Key Findings**:

- Under single-arm incentives, the upper and lower bounds for the Greedy model match within a $\log K$ factor — **tight**.
- Under single-arm incentives, the upper and lower bounds for the Smooth model match within polylogarithmic factors — **tight**.
- The min structure from $K\sqrt{T}$ to $\sqrt{KT\log N}$ reflects the tradeoff between the type count of agents and the number of arms.

## Highlights & Insights

1. **New Problem Formulation**: This paper is the first to systematically study repeated principal-agent problems with adversarial agent arrivals, filling the gap in existing literature that only considers fixed or stochastic arrivals.
2. **Impossibility Characterization**: Proving that a greedy agent leads to linear regret without prior knowledge, which reveals the fundamental challenge of "non-smooth sensitivity" in incentive design.
3. **Unified Reduction Framework**: Dynamically reducing incentive design problems under different settings to adversarial linear bandits in a technically elegant manner.
4. **Tight Bounds**: Providing matching upper and lower bounds under both the greedy (known) and smooth settings, proving a strong theoretical completeness.
5. **Bridge to Practicality**: The smooth model can be naturally derived from the greedy model through Gumbel/Gaussian noise, providing a smooth transition from theory to practice.

## Limitations & Future Work

1. **Purely Theoretical Work**: Lacks any experimental or simulation verification, making its effectiveness in practical scenarios unknown.
2. **Non-strategic Agents**: Assumes that agents are myopic and does not consider potential strategic behaviors of agents in repeated games (e.g., misreporting preferences).
3. **Lack of Lower Bounds for General Incentives**: Both greedy + general and smooth + general settings lack lower bounds, leaving their tightness unknown.
4. **Smooth Model Requires Known $L$**: The algorithm requires the Lipschitz constant $L$ as input. When $L$ is unknown, the granularity of discretization cannot be determined (the paper notes this leads to a linear dependence on $L$).
5. **Offline Optimal Benchmark**: The regret definition compares against the optimal fixed incentive; stronger notions like dynamic regret or comparisons against optimal policy sequences are not discussed.
6. **Scalability**: The discretization size for the greedy + general setting is $(6KT)^K$, which is computationally intractable as $K$ grows.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Completely new problem setting, studying adversarial arrivals in the PA framework for the first time.
- Theoretical Depth: ⭐⭐⭐⭐⭐ — Matching upper and lower bounds for multiple settings with elegant analysis techniques.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and intuitive motivational examples, though notations are somewhat heavy.
- Value: ⭐⭐⭐ — Purely theoretical contribution, with a gap still remaining before practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Principal-Agent Bandit Games with Self-Interested and Exploratory Learning Agents](principal-agent_bandit_games_with_self-interested_and_exploratory_learning_agent.md)
- [\[ICLR 2026\] Optimal Robust Subsidy Policies for Irrational Agent in Principal-Agent MDPs](../../ICLR2026/reinforcement_learning/optimal_robust_subsidy_policies_for_irrational_agent_in_principal-agent_mdps.md)
- [\[ICML 2025\] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)
- [\[ICML 2025\] Learning Progress Driven Multi-Agent Curriculum](learning_progress_driven_multi-agent_curriculum.md)
- [\[ICML 2025\] Preference Optimization for Combinatorial Optimization Problems](preference_optimization_for_combinatorial_optimization_problems.md)

</div>

<!-- RELATED:END -->
