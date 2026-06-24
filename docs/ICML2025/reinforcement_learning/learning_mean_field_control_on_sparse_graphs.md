---
title: >-
  [Paper Note] Learning Mean Field Control on Sparse Graphs
description: >-
  [ICML2025][Reinforcement Learning][mean field control] The paper proposes the Local Weak Mean Field Control (LWMFC) framework, which leverages local weak convergence theory to extend mean field control to extremely sparse graphs with a power-law exponent of $\gamma > 2$. Combined with a two-systems approximation and scalable RL algorithms, this method significantly outperforms Lp-graphon and graphex-based methods on both synthetic and real-world networks.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "mean field control"
  - "sparse graphs"
  - "multi-agent reinforcement learning"
  - "local weak convergence"
  - "graphon"
date: 2026-05-08
content_hash: 30739bf44592e041
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Learning Mean Field Control on Sparse Graphs

**Conference**: ICML2025  
**arXiv**: [2501.17079](https://arxiv.org/abs/2501.17079)  
**Authors**: Christian Fabian, Kai Cui, Heinz Koeppl (TU Darmstadt)
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: mean field control, sparse graphs, multi-agent reinforcement learning, local weak convergence, graphon

## TL;DR

The paper proposes the Local Weak Mean Field Control (LWMFC) framework, which leverages local weak convergence theory to extend mean field control to extremely sparse graphs with a power-law exponent of $\gamma > 2$. Combined with a two-systems approximation and scalable RL algorithms, this method significantly outperforms Lp-graphon and graphex-based methods on both synthetic and real-world networks.

## Background & Motivation

### Problem Definition

The cooperative optimization problem in large-scale multi-agent networks (Mean Field Control): $N$ agents are connected via a graph $G_N$ and jointly maximize a global objective $J^N(\pi) = \sum_{t=1}^{T} r(\mu_t^N)$, where $\mu_t^N$ represents the empirical mean field distribution.

### Limitations of Prior Work

**Graphon MFG/MFC**: Only applicable to dense graphs (complete graph limits); a vast majority of real-world networks do not satisfy this assumption.

**Lp Graphon MFG (LPGMFG)**: Extends to moderately sparse graphs, but still requires the average degree to diverge to infinity.

**Graphex MFG (GXMFG)**: Likewise limited to graph sequences where the average expected degree diverges to infinity.

None of these three classes of methods **cover** sparse networks with a power-law exponent of $\gamma > 2$ (such as the Internet, co-authorship networks, or biological networks), because the average expected degree of these networks remains finite.

### Core Motivation

Many highly valuable empirical networks (the Internet with $\gamma \approx 2.1-2.5$, co-authorship graphs, YouTube social networks, etc.) exhibit topologies that are significantly sparser than what the aforementioned methods can handle. In a power-law graph with $\gamma = 2.5$, approximately 96% of the nodes have a degree $\le 5$; however, these low-degree nodes account for only about 68% of the expected degree. The heavy tail of high-degree nodes makes simple truncation infeasible.

## Method

### 1. Local Weak Convergence

Core graph-theoretic tool: A graph sequence $(G_N)_N$ converges in the local weak sense to a random element $G$, meaning that for all continuous bounded functions $f$:

$$\lim_{N\to\infty} \frac{1}{N} \sum_{i \in [N]} f(C_{v_i}(G_N)) = \mathbb{E}[f(G)] \quad \text{in probability}$$

Graph models satisfying this convergence include:

- **Configuration Model (CM)**: Random multigraphs with arbitrary degree sequences.
- **Barabási-Albert Model**: Preferential attachment, generating networks with a power-law exponent of exactly 3.
- **Chung-Lu (CL) Model**: Node weights $w_i$ determine connection probabilities $w_i \cdot w_j / \bar{w}$, efficiently generating power-law networks with $\gamma > 2$.

Key Advantage: Local weak convergence covers graph sequences with a finite average expected degree, which is precisely the regime that Lp graphon and graphex methods fail to address.

### 2. LWMFC Finite Model and Limit System

**Finite Model**: $N$ agents are grouped by degree $k$ and share a policy $\pi_t^k$. The empirical mean field of degree $k$ is defined as:

$$\mu_t^{N,k} = \frac{1}{|V_N^k|} \sum_{i: v_i \in V_N^k} \delta_{X_{i,t}^N}$$

The transition dynamics of each agent depend on its own state, action, and the neighborhood state distribution $\mathbb{G}_{i,t}^N$.

**Limit System** ($N \to \infty$): The mean field of each degree $k$ evolves according to a deterministic equation:

$$\mu_{t+1}^k = \sum_{x \in \mathcal{X}} \mu_t^k(x) \sum_{G \in \mathcal{G}^k} P_\pi(\mathbb{G}_t^k = G | x_t = x) \cdot \sum_{u \in \mathcal{U}} \pi_t^k(u|x,G) P(\cdot|x,u,G)$$

### 3. Theoretical Guarantees

- **Theorem 3.1 (MF Convergence)**: Under Assumption 2.2, $\mu_t^{N,k} \to \mu_t^k$ in probability.
- **Proposition 3.3 (Objective Convergence)**: $J^N(\pi) \to J(\pi)$ in probability.
- **Corollary 3.4 (Optimal Policy Transfer)**: The optimal policy of the limit system is also optimal in all sufficiently large finite systems.

### 4. Two Systems Approximation

Directly solving the limit system is intractable—Lemma 4.1 proves that the $t$-hop neighborhood complexity is $\Omega(2^{\text{poly}(k)})$.

**Heuristic 1 (Neighbor Degree Distribution Approximation)**:

$$P(\deg(v)=k | (v',v) \in E) \approx \frac{k \cdot P(\deg(v)=k)}{\sum_{k''} k'' \cdot P(\deg(v)=k'')}$$

Based on this, agents are partitioned into a **low-degree system** (degree $\le k^*$) and a **high-degree system** (degree $> k^*$):

- High-degree agents share a unified neighborhood distribution $\hat{\mathbb{G}}_t^\infty$, simplifying their dynamics to:

$$\hat{\mu}_{t+1}^\infty = \sum_{x,u} \hat{\mu}_t^\infty(x) \pi_t^\infty(u|x, \hat{\mathbb{G}}_t^\infty) P(\cdot|x,u,\hat{\mathbb{G}}_t^\infty)$$

- Low-degree agents (degree $k \le k^*$) have neighborhoods sampled from a multinomial distribution: $\hat{\mathbb{G}}_t^k \sim \text{Mult}(k, \hat{\mathbb{G}}_t^\infty)$.

The paper also derives a more refined extensive approximation (LWMFC*), which uses state-degree neighborhood distributions to yield higher accuracy, albeit at the cost of significantly increased computational complexity.

### 5. Learning Algorithms

**Algorithm 1 (LWMFC Policy Gradient)**: Transforms the two-systems approximation into an MFC MDP, where the system state is $\boldsymbol{\mu}_t = (\mu_t^1, \ldots, \mu_t^{k^*}, \mu_t^\infty)$ and the action is the policy set $\boldsymbol{\pi}_t$. The single-agent MFC MDP is solved using PPO.

**Algorithm 2 (LWMFMARL Policy Gradient)**: Does not assume model knowledge and directly samples execution on the real network to substitute the MF equations. Each node executes actions based on its sampled policy, strictly guaranteed by single-agent theory. This approach avoids the inaccuracy of the two-systems approximation.

## Experimental Setup & Main Results

### Experimental Settings

- **Four Problems**: SIS (epidemic propagation), SIR (epidemic propagation with recovery), Graph Coloring, and Rumor (rumor spreading).
- **Eight Real-world Networks**: CAIDA (~26k nodes), Cities (~14k), Digg Friends (~280k), Enron (~87k), Flixster (~2.5M), Slashdot (~50k), Yahoo (~653k), and YouTube (~3.2M).
- **Synthetic Networks**: Chung-Lu power-law graphs with sizes $N \in \{167, 406, 860, 1598\}$.
- **Baselines**: LPGMFG, GXMFG, IPPO.
- **Training**: Approximately 80,000 CPU core hours, running about one day per training run (96 parallel CPU cores). PPO configuration: 2x256 tanh hidden layers, lr=5e-5, $\gamma=0.99$, GAE $\lambda=1.0$.

### Main Results — MF Approximation Accuracy (Table 1)

Average expected total variation $\Delta\mu = \frac{1}{2T}\mathbb{E}[\sum_t \|\hat{\mu}_t - \mu_t\|_1]$ ($\times 100\%$, 50 trials), lower is better:

| Problem | Model | CAIDA | Enron | Flixster | YouTube |
|------|------|-------|-------|----------|---------|
| SIS | LPGMFG | 24.02 | 24.77 | 22.48 | 22.94 |
| SIS | GXMFG | 9.07 | 4.73 | 3.78 | 6.43 |
| SIS | **LWMFC** | **2.59** | **3.39** | **1.60** | **3.53** |
| SIS | LWMFC* | 1.75 | 2.67 | 0.90 | 2.93 |
| SIR | LPGMFG | 9.11 | 9.51 | 8.99 | 8.90 |
| SIR | GXMFG | 2.81 | 0.99 | 0.99 | 1.79 |
| SIR | **LWMFC** | **1.31** | **0.91** | **0.58** | **1.07** |
| Color | LPGMFG | 38.73 | 39.83 | 39.55 | 38.52 |
| Color | GXMFG | 11.33 | 4.91 | 6.38 | 8.76 |
| Color | **LWMFC** | **0.70** | **0.36** | **0.39** | **0.19** |

**Key Findings**: LWMFC consistently and comprehensively outperforms LPGMFG and GXMFG across all networks and problems. The improvement is particularly prominent in the Color problem (e.g., reducing the error from 11.33 to 0.70 on CAIDA, a 94% reduction).

### Main Results — Learning Algorithm Performance (Table 2)

Optimal objective values after 24 hours of training on synthetic CL graphs:

| Problem | LWMFC | LWMFMARL | IPPO |
|------|-------|----------|------|
| SIS (N=860) | -19.70 | -12.42 | -9.11 |
| SIS (N=1598) | -22.42 | -13.51 | -11.13 |
| SIR (N=860) | -10.64 | -6.86 | -5.15 |
| Color (N=860) | -8.48 | -7.08 | -5.85 |
| Rumor (N=860) | 0.25 | 1.47 | 1.35 |

While LWMFMARL and IPPO perform well with direct network interaction, the LWMFC method family overall outperforms IPPO on larger networks ($N = 860, 1598$).

## Limitations & Future Work

1. **Inaccuracy of the Two-Systems Approximation**: The threshold $k^*$ separating low-degree and high-degree agents is manually configured, and the optimal $k^*$ varies across different problems.
2. **Evaluation of LWMFC on the Limit System Instead of the Real System**: The policy learned by Algorithm 1 is optimal for the limit system, which may differ from the true finite system.
3. **Applicability of Heuristic 1**: The neighbor degree distribution approximation only holds theoretical backing on Chung-Lu (CL) style graphs, which might be inaccurate for other topologies.
4. **High Computational Overhead of Extensive Approximation**: While LWMFC* achieves higher accuracy, it fails to finish within a reasonable timeframe on the Color and Rumor problems.
5. **Cooperative Scenario Only**: The paper focuses exclusively on MFC (cooperative) and does not extend to competitive MFG (Mean Field Game) settings.
6. **Simplified Policy Parameterization**: The experiments implement policies relying solely on node states (instead of complete neighborhood information), which may limit expressiveness.
7. **Lack of Comparison on Moderately Sparse Graphs**: There is no systematic exploration comparing LWMFC against LPGMFG/GXMFG in moderately sparse regimes where the latter two are also applicable.

## Related Work & Insights

- **Graphon MFG/MFC**: Dense-graph methods such as Caines et al. (2019, 2021) and Hu et al. (2023) represent the direct predecessors of this work.
- **Lp Graphon / Graphex MFG**: Fabian et al. (2023, 2024) broaden MFG to moderately sparse graphs.
- **Local Weak Convergence Theory**: Studies like van der Hofstad (2024) and Lacker & Soret (2023) supply the theoretical foundation for particle system convergence on sparse graphs.
- **Scalable MARL Methods**: Independent learning methods such as IPPO (Tan 1993) remain standard paradigms in large-scale MARL.

**Key Insight**: Combining graph-theoretic convergence concepts with mean-field methods is a powerful and generalizable approach, potentially extending to more complex MF models with partial observability or bounded rationality.

## Personal Commentary

The core contribution of this paper is filling the theoretical and algorithmic gaps of MFC on sparse graphs. Introducing local weak convergence to the mean-field framework is a natural and convincing idea. Although the two-systems approximation is heuristic, experiments demonstrate its overwhelming superiority over existing methods across various real-world networks. The order-of-magnitude improvement of LWMFC over GXMFG on the Color problem in Table 1 (e.g., from 8.76 to 0.19 on YouTube) provides exceptionally strong empirical evidence. However, Algorithm 1 (the MFC MDP route) underperforms compared to IPPO on small-scale graphs, indicating a gap in mean-field approximation when the node count is insufficiently large. Overall, this is a solid piece of work with both rigorous theory and extensive experiments.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Introduces local weak convergence to MFC for the first time, addressing extremely sparse graph classes that were completely unresolved previously.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 4 problems × 8 real-world networks + synthetic networks, and compares thoroughly with baselines, though lacking a detailed ablation (e.g., on the selection of $k^*$).
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous theoretical derivations; somewhat heavy on notation but has a clear structure.
- **Value**: ⭐⭐⭐⭐ — Clear positive impact on the networked MARL community, with real-world experiments enhancing its practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](../../NeurIPS2025/reinforcement_learning/learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [\[NeurIPS 2025\] Last Iterate Convergence in Monotone Mean Field Games](../../NeurIPS2025/reinforcement_learning/last_iterate_convergence_in_monotone_mean_field_games.md)
- [\[NeurIPS 2025\] Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](../../NeurIPS2025/reinforcement_learning/scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)
- [\[NeurIPS 2025\] Non-convex Entropic Mean-Field Optimization via Best Response Flow](../../NeurIPS2025/reinforcement_learning/non-convex_entropic_mean-field_optimization_via_best_response_flow.md)

</div>

<!-- RELATED:END -->
