---
title: >-
  [Paper Note] Risk-Sensitive Agent Compositions
description: >-
  [ICLR 2026][AI Safety][Risk-Sensitive] This paper formalizes agent workflows as directed acyclic graphs (Agent Graphs), models safety/fairness/privacy requirements via a max loss function, and proposes the BucketedVaR algorithm, which combines union bounds with dynamic programming to find the optimal agent composition minimizing VaR/CVaR in polynomial time. The approach is proven to be asymptotically near-optimal under an independence assumption on agent losses.
tags:
  - ICLR 2026
  - AI Safety
  - Risk-Sensitive
  - Agent Composition
  - VaR
  - CVaR
  - Dynamic Programming
date: 2026-05-08
content_hash: ab5fb83612afb3c8
---

# Risk-Sensitive Agent Compositions

**Conference**: ICLR 2026
**arXiv**: [2506.04632](https://arxiv.org/abs/2506.04632)
**Code**: None
**Area**: AI Safety / Agent Systems
**Keywords**: Risk-Sensitive, Agent Composition, VaR, CVaR, Dynamic Programming

## TL;DR

This paper formalizes agent workflows as directed acyclic graphs (Agent Graphs), models safety/fairness/privacy requirements via a max loss function, and proposes the BucketedVaR algorithm, which combines union bounds with dynamic programming to find the optimal agent composition minimizing VaR/CVaR in polynomial time. The approach is proven to be asymptotically near-optimal under an independence assumption on agent losses.

## Background & Motivation

**Prevalence of Agent Compositions**: Modern agent systems decompose complex tasks into sequences of subtasks, delegating each to a specialized AI agent (LLM, VLM, RL policy). Representative applications include automated software development, information retrieval, and long-horizon robotic control.

**Necessity of Risk Minimization**: Real-world deployment demands not only maximizing task success but also minimizing violations of safety, fairness, and privacy requirements. A critical characteristic of such violations is their **tail behavior**—low-probability but high-consequence events.

**Max Loss vs. Cumulative Loss**: When losses measure requirement violations (rather than costs), the aggregate loss of a composed agent pipeline should be the **maximum** of individual agent losses—a single severe violation implies a systemic violation—rather than the **cumulative sum** as in conventional MDPs. This constitutes a fundamental departure from prior risk-sensitive planning literature.

**Combinatorial Explosion**: The number of feasible paths in an agent graph may grow exponentially with the number of agents, rendering naive per-path VaR estimation computationally intractable.

**Choice of Risk Measures**: VaR (Value-at-Risk) controls tail quantiles, while CVaR (Conditional Value-at-Risk) measures the conditional expected loss in the tail. Both capture extreme events more faithfully than the expected loss.

**Black-Box Agent Assumption**: The paper assumes only sampling access to each agent (no knowledge of internal structure), estimating risk measures via Monte Carlo sampling—applicable to arbitrary agent types including RL policies and LLMs.

## Method

### 1. Agent Graph Formalization

An agent workflow is represented as a DAG $G = (V, E, X, T, F, L, s, t, \mathcal{D}_s)$:
- Each edge $e \in E$ is associated with an agent $f_e$, a trajectory set $T_e$, and a loss function $L_e: T_e \to \mathbb{R}$
- The source node $s$ has an initial input distribution $\mathcal{D}_s$; the sink node $t$ is the goal
- A path $p = v_1 \xrightarrow{e_1} \cdots \xrightarrow{e_m} v_{m+1}$ corresponds to an agent composition
- The composition loss is defined as the maximum:

$$L_p(t_1, \ldots, t_m) = \max_i \{L_{e_i}(t_i)\}$$

### 2. Risk-Minimizing Agent Graph (RMAG) Objective

Given a risk level $\alpha \in (0,1)$, the optimization objective is:

$$\arg\min_{p \in \mathcal{P}} \rho[L_p(Z_p)], \quad \rho \in \{\text{VaR}_\alpha, \text{CVaR}_\alpha\}$$

where $\text{VaR}_\alpha$ denotes the $(1-\alpha)$-quantile:

$$\text{VaR}_\alpha[L(Z)] = \inf\{q \in \mathbb{R}: \Pr[L(Z) \leq q] \geq 1-\alpha\}$$

and $\text{CVaR}_\alpha$ is the tail conditional expectation:

$$\text{CVaR}_\alpha[L(Z)] = \frac{1}{\alpha}\int_0^\alpha \text{VaR}_\gamma[L(Z)]\,d\gamma$$

### 3. BucketedVaR Algorithm

**Key Designs**:
- **Union Bound Decomposition**: Exploits $\Pr[\max(R_1,...,R_m) > q] \leq \sum_i \Pr[R_i > q]$ to decompose the composition VaR into independent per-agent quantile estimates.
- **Risk Budget Discretization**: Discretizes the total budget $\alpha$ into $d+1$ buckets $B = \{0, \alpha/d, 2\alpha/d, \ldots, \alpha\}$.
- **Dynamic Programming**: Traverses the graph in topological order, maintaining the optimal partial path for each vertex–bucket pair $(v, \bar{\alpha})$.
- **Incremental Estimation**: Allocates budget $\bar{\alpha} - \alpha'$ to each edge, using the empirical $(1-(\bar{\alpha}-\alpha'))$-quantile as the edge VaR estimate.
- **Path VaR via Max**: $\text{pathVaR} = \max(\text{VaR}[v', \alpha'], \text{edgeVaR})$.
- **CVaR Recovery**: $\text{CVaR}_\alpha \approx \frac{1}{d}\sum_{k=1}^d \text{VaR}_{k\alpha/d}$, reusing already-computed VaR values without additional sampling.

**Theoretical Guarantee (Theorem 1)**: Time complexity $O(n(d+1)^2|V|^2)$; with probability $\geq 1-\delta$:

$$q \geq \text{quantile}(L_p(Z_p), 1-\alpha-\gamma), \quad \gamma = |V|\sqrt{\frac{1}{2n}\ln\frac{2(d+1)^2|V|^2}{\delta}}$$

**Near-Optimality (Theorem 2)**: Under the independence assumption, as $n, d \to \infty$:

$$q \leq \text{quantile}\left(L_{p^*}(Z_{p^*}), 1-\alpha+\frac{\alpha^2}{2}\right)$$

The suboptimality gap is at most $\alpha^2/2$ (only $0.005$ for $\alpha=0.1$).

## Key Experimental Results

### Table 1: BucketedVaR vs. Optimal Baseline — Approximation Accuracy

| Benchmark | Risk Level $\alpha$ | VaR Quantile Error (%) | CVaR Error (%) | Optimal Path Match |
|-----------|--------------------|-----------------------|---------------|-------------------|
| DroneNav | 0.1 | < 2 | < 2 | ✓ |
| 16-Rooms | 0.1 | < 3 | < 2 | ✓ |
| Fetch | 0.05 | < 2 | < 2 | ✓ |
| BoxRelay | 0.1 | < 2 | < 3 | ✓ |

### Table 2: Robustness — Effect of Loss Correlation on Approximation Quality

| Correlation $\rho$ | Path Length=4 Coverage | Path Length=8 Coverage | Path Length=16 Coverage |
|--------------------|----------------------|----------------------|------------------------|
| 0.0 (Independent) | ~0.90 | ~0.90 | ~0.90 |
| 0.25 | ~0.91 | ~0.92 | ~0.93 |
| 0.5 | ~0.92 | ~0.94 | ~0.95 |
| 0.75 | ~0.95 | ~0.97 | ~0.98 |
| 1.0 (Perfect Correlation) | ~0.99 | ~0.99 | ~0.99 |

## Key Findings

1. **Union Bound is Tight in Practice**: BucketedVaR estimates for both VaR and CVaR deviate from the exhaustive optimal baseline by only a few percentage points, confirming the practical effectiveness of the union bound approach.

2. **Non-Trivial Risk Budget Allocation**: The optimal allocation learned by the algorithm is non-uniform—in 16-Rooms, the VaR$_{0.1}$ budget across 8 agents is distributed as $16\bar\alpha, 0\bar\alpha, 10\bar\alpha, 23\bar\alpha, 19\bar\alpha, 11\bar\alpha, 7\bar\alpha, 14\bar\alpha$, reflecting heterogeneous risk levels across subtasks.

3. **Robustness to Moderate Correlation**: Even under moderate inter-agent loss correlation ($\rho \leq 0.5$), the algorithm produces reasonable risk estimates, degrading substantially only under perfect correlation ($\rho=1$).

4. **Scalability to Many Agents**: Scaling from 8 to 40 agents ($8 \times 5$), the VaR approximation accuracy consistently remains near the target quantile, validating the theoretical scalability.

5. **Convergence of Samples and Buckets**: As the sample size increases from 500 to $10^4$ and the number of buckets from 5 to 100, empirical quantiles converge stably to $\sim 0.91$ (target: $0.90$), with fast convergence rates.

## Highlights & Insights

- **Novel Max-Loss Modeling**: Safety/fairness/privacy violations are governed by the worst-offending agent; modeling with max rather than sum represents a fundamental departure from traditional cumulative-loss MDP risk optimization, requiring entirely different theoretical tools.
- **Surprising Power of Union Bounds**: The seemingly coarse union bound is asymptotically near-exact for quantile estimation (suboptimality only $\alpha^2/2$), illustrating the unexpected power of simple methods.
- **Generality of the Agent Graph Formalization**: The framework uniformly subsumes RL policy compositions, LLM-based information retrieval pipelines, and other agent workflows, elevating agent composition optimization from trial-and-error to theoretically grounded graph search.
- **CVaR as a Byproduct of VaR**: CVaR is recovered by averaging the already-computed discrete VaR values, requiring no additional sampling.

## Limitations & Future Work

- **Independence Assumption**: Theoretical guarantees rely on independence of agent losses. When agents share environmental state—e.g., a preceding agent's output influences a subsequent agent's safety—this assumption may be violated. While experiments demonstrate robustness to moderate correlation, no theoretical guarantee exists for the non-independent case.
- **Loss Function Design**: Loss functions are naturally defined for RL control environments (e.g., distance to obstacles), but for LLM agent properties such as hallucination and bias, loss definition relies on proxy approaches such as LLM-as-Judge, which can be costly and noisy.
- **Experimental Scale**: Validation is conducted only on RL control benchmarks with at most 40 agents; large-scale LLM agent systems remain untested.
- **Static Composition**: The current framework selects a fixed path and does not consider dynamically switching agents based on runtime observations.

## Related Work & Insights

| Dimension | Ours (BucketedVaR) | Risk-Sensitive MDP (Ahmadi 2021 et al.) | Hierarchical RL (Jothimurugan 2021) |
|-----------|-------------------|-----------------------------------------|-------------------------------------|
| Loss Type | Max (requirement violation) | Cumulative sum (cost) | Expected reward |
| Risk Measure | VaR / CVaR | CVaR / EVaR | None (expectation optimization) |
| Agent Model | Black-box sampling | MDP internal structure | Trainable policy |
| Optimization Target | Path selection | Policy optimization | Policy learning |
| Scalability | Polynomial (in number of agents) | Single agent | Single agent, hierarchical |

Compared to automated agent workflow generation methods such as **AFlow (Zhang 2025a)**, this work focuses not on designing workflow structures but on selecting the optimal agent composition within a given workflow structure to minimize risk—the two approaches are complementary.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The max-loss + Agent Graph formalization for risk minimization fills a genuine theoretical gap
- **Experimental Thoroughness**: ⭐⭐⭐ Theoretically rigorous but limited experimental scale; LLM agent experiments are absent
- **Writing Quality**: ⭐⭐⭐⭐ Formal definitions are clear; illustrative examples (DroneNav / information retrieval) are intuitive
- **Value**: ⭐⭐⭐⭐ Provides solid theoretical guidance for safety-critical agent deployment

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction](sample-efficient_distributionally_robust_multi-agent_reinforcement_learning_via_.md)
- [\[NeurIPS 2025\] It's Complicated: The Relationship of Algorithmic Fairness and Non-Discrimination Provisions for High-Risk Systems in the EU AI Act](../../NeurIPS2025/ai_safety/its_complicated_the_relationship_of_algorithmic_fairness_and_non-discrimination_.md)
- [\[ICLR 2026\] Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature](dataless_weight_disentanglement_in_task_arithmetic_via_kronecker-factored_approx.md)
- [\[ICLR 2026\] Action-Free Offline-to-Online RL via Discretised State Policies](action-free_offline-to-online_rl_via_discretised_state_policies.md)
- [\[ICLR 2026\] Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction](extending_sequence_length_is_not_all_you_need_effective_integration_of_multimoda.md)

<!-- RELATED:END -->
