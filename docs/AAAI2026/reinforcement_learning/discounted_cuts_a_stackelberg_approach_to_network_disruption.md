---
title: >-
  [Paper Note] Discounted Cuts: A Stackelberg Approach to Network Disruption
description: >-
  [AAAI 2026][Reinforcement Learning][Stackelberg Games] This paper introduces the Discounted Cuts mathematical framework, modeling the classical Most Vital Links problem as a Stackelberg game. It systematically establishes a computational complexity classification for eight variants of discounted cuts and proves that all variants are solvable in polynomial time on bounded-genus graphs.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Stackelberg Games
  - Network Cuts
  - Discounted Cost
  - Bounded-Genus Graphs
  - Most Vital Links
date: 2026-05-08
content_hash: 49709752cd546aee
---

# Discounted Cuts: A Stackelberg Approach to Network Disruption

**Conference**: AAAI 2026
**arXiv**: [2511.10804](https://arxiv.org/abs/2511.10804)
**Code**: None
**Area**: Algorithmic Game Theory / Network Optimization
**Keywords**: Stackelberg Games, Network Cuts, Discounted Cost, Bounded-Genus Graphs, Most Vital Links

## TL;DR

This paper introduces the Discounted Cuts mathematical framework, modeling the classical Most Vital Links problem as a Stackelberg game. It systematically establishes a computational complexity classification for eight variants of discounted cuts and proves that all variants are solvable in polynomial time on bounded-genus graphs.

## Background & Motivation

### Problem Setting

Consider a network security scenario: an attacker attempts to escalate privileges from server $s$ to a critical database $t$, while a security team has budget $\beta$ to deploy firewalls or sever connections. Certain links (e.g., backbone lines) are prohibitively expensive to disable, but a cyber-insurance policy allows the team to disable up to $k$ links for free. This motivates a Stackelberg game variant: the leader selects a cut, and the follower discounts the cost of $k$ edges.

A second scenario arises in conflict networks within multi-agent systems: agents are modeled as nodes, interaction intensities as weighted edges, and the goal is to maximize inter-group tension via MaxCut. However, low-trust or noisy edges should be discounted to focus on strongly strategic interactions.

### Motivation Analysis

**Non-additive cost functions**: Standard minimum cut assumes an additive cost function, but in adversarial settings the attacker discounts a subset of edge costs, rendering the function non-additive.

**Theoretical gap**: Despite the long history of the Most Vital Links problem (Wollmer, 1980s), the literature lacks a formal NP-hardness proof for general graphs.

**Cross-domain unification**: The work bridges AI (multi-agent systems), algorithmic game theory (Stackelberg games), and operations research (network optimization).

### Core Idea

Two discount cost functions are introduced:
- $\mathsf{Cost}^{\rm exp}_k$: excludes the $k$ most expensive edges (insurance covers the costliest assets).
- $\mathsf{Cost}^{\rm cheap}_k$: excludes the $k$ cheapest edges (ignores noisy links).

Combining these two discount mechanisms with min/max cuts and s-t/global cuts yields a unified framework of eight variants.

## Method

### Overall Architecture

The discounted cut problem is unified into a framework of eight variants. A series of reductions transforms each discounted cut variant into its classical counterpart, and a complexity classification is established for each variant across different graph classes.

### Key Designs

#### 1. **Discount Cost Function Definition**

Given a cut $(A,B)$ and edge cost function $c: E(G) \to \mathbb{Z}_{\geq 0}$, the discounted cost is defined as:

$$\mathsf{Cost}^{\rm exp}_k(A,B) = c(E(A,B)) - c(R)$$

where $R$ denotes the $k$ most expensive edges in the cut edge set. When $|E(A,B)| \leq k$, the cost is zero.

**Design Motivation**: The attacker may remove the $k$ most critical edges for free; the defender must find a minimum-cost cut under this discount.

#### 2. **General Reduction Technique (Theorem 3)**

**Mechanism**: Reduces the discounted problem to a classical one. If Minimum $\Pi$ is solvable in time $T$, then Minimum $\Pi$ with $k$ Free Cheap Elements can be solved by enumerating thresholds $w \in C(U)$ (at most $|U|$ values) and invoking the classical algorithm under modified costs:

$$O_{\min} = \min_{w \in C(U)} \left(\mathsf{Opt}_{\min}(U, c_w, \mathcal{F}) - kw\right)$$

where $c_w(x) = \max(c(x), w)$ raises costs below $w$ to $w$. Total time: $\mathcal{O}(|U| \cdot T)$.

**Design Motivation**: Avoids direct handling of non-additive cost functions by solving classical additive problems at different thresholds.

#### 3. **Dual Graph Algorithm for Planar Graphs (Theorem 4)**

For planar Min s-t-Cut$-k$ exp:
- Exploits duality: minimum cuts in planar graph $G$ correspond to cycles in the dual $G^*$.
- Discrete Jordan Curve Theorem: a cycle in $G^*$ corresponds to an s-t cut $\iff$ it crosses some s-t path an odd number of times.
- Dynamic programming is applied to find the minimum-cost closed walk in the dual graph.

Complexity: $\mathcal{O}(kn^2 \log n \log M)$

#### 4. **Polynomial Algorithm for Global Minimum Discounted Cut (Theorem 5)**

For general MinCut$-k$ exp:
- Reduces to Bicriteria Global Minimum Cut: given two weight functions $w_1, w_2$ and budgets $b_1, b_2$, find a cut satisfying both criteria.
- Leverages the randomized algorithm of Aissi et al.

Complexity: $\mathcal{O}(n^3 \cdot m \cdot \log^4 n \cdot \log\log n \cdot \log M)$

#### 5. **NP-completeness Proof (Theorem 1)**

Min s-t-Cut$-k$ exp is NP-complete even when edge costs are restricted to $\{1, 2\}$, and is W[1]-hard under integer edge costs (parameterized by $k$). This substantially strengthens previously known lower bounds.

### Complexity Classification Summary

The bounded-genus graph algorithm (Theorem 2) handles all eight variants uniformly:

$$\text{Time} = (4^g \cdot n^{1.5} + m \cdot M) \cdot m^2 \cdot M \cdot \log^{\mathcal{O}(1)}(n+M)$$

where $g$ is the genus of the graph and $M$ is the maximum edge cost. The approach relies on algebraic techniques applied to generating functions for cuts on bounded-genus graphs.

## Key Experimental Results

This paper is a purely theoretical contribution; its core results constitute a complete complexity classification.

### Main Results

| Problem Variant | General Graphs | Planar/Bounded-Genus Graphs |
|---|---|---|
| Min s-t-Cut exp | W[1]-hard | **P** |
| Max s-t-Cut cheap | paraNP-hard | **P** |
| Min s-t-Cut cheap | **P** | **P** |
| Max s-t-Cut exp | paraNP-hard | **P** |
| MinCut exp | **P** | **P** |
| MaxCut cheap | paraNP-hard | **P** |
| MinCut cheap | **P** | **P** |
| MaxCut exp | paraNP-hard | **P** |

### Ablation Study (Applicability of Each Reduction)

| Reduction Method | Applicable Setting | Complexity |
|---|---|---|
| Threshold Enumeration (Thm 3) | Min-cheap / Max-exp | $O(\|U\| \cdot T)$ |
| Dual Graph DP (Thm 4) | Planar Min s-t-Cut exp | $O(kn^2\log n\log M)$ |
| Bicriteria Reduction (Thm 5) | General MinCut exp | $O(n^3 m \log^4 n)$ |
| Generating Functions (Thm 2) | All variants on bounded-genus graphs | Depends on genus $g$ |

### Key Findings

1. **Surprising complexity gap**: Min s-t-Cut$-k$ exp is NP-complete on general graphs, yet its global counterpart MinCut$-k$ exp is polynomial-time solvable.
2. **Asymmetry of discount direction**: Discounting the most expensive edges (exp) versus the cheapest (cheap) yields fundamentally different computational complexities.
3. **Unified tractability on bounded-genus graphs**: All eight variants are polynomial-time solvable.
4. **Resolution of a historical open problem**: The first formal proof of NP-completeness for Most Vital Links on general graphs.

## Highlights & Insights

1. **Elegant unified framework**: 8 variants × 2 graph classes = 16 complexity entries, providing a comprehensive characterization of the computational landscape.
2. **Ingenious threshold enumeration**: Classical algorithms are invoked $O(|U|)$ times to resolve the discounted variant, yielding a clean and efficient reduction.
3. **Deep application of dual graph methods**: Planar graph duality is combined with dynamic programming to handle non-additive cost functions.
4. **Cross-domain bridge**: Methods from AI, game theory, and operations research are unified under the discounted cuts framework.

## Limitations & Future Work

1. **Bounded-genus restriction**: The core algorithm requires graphs of bounded genus; the complexity for more general graph classes remains open.
2. **Polynomial cost assumption**: Some results require edge costs to be polynomially bounded.
3. **Lack of empirical validation**: No experiments on real-world networks (e.g., transportation, infrastructure) are conducted.
4. **Extension to directed graphs**: The work is primarily limited to undirected graphs; discounted cuts in directed networks warrant further investigation.
5. **Approximation algorithms**: Approximation ratios for NP-hard variants are not explored.

## Related Work & Insights

- **Most Vital Links (Wollmer, 1980s)**: The direct predecessor, studying the removal of $k$ edges to minimize maximum flow.
- **Network Interdiction**: Network disruption problems in the Stackelberg framework.
- **Karger/Stoer-Wagner Algorithms**: Classical algorithms for global minimum cut, used as subroutines for the discounted variants.
- **MaxCut on Bounded-Genus Graphs**: Algebraic techniques for handling generating functions.
- The work offers insights for multi-agent security research: how to most effectively protect or attack a network under limited resources.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic study of all eight discounted cut variants, filling a theoretical gap.
- Experimental Thoroughness: ⭐⭐⭐ — A purely theoretical work; the complexity classification is complete but no experiments are included.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, precise theorems, and well-motivated intuitions.
- Value: ⭐⭐⭐⭐ — Establishes theoretical foundations for adversarial network optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Deep (Predictive) Discounted Counterfactual Regret Minimization](deep_predictive_discounted_counterfactual_regret_minimization.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[ICLR 2026\] Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](../../ICLR2026/reinforcement_learning/virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)
- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](../../ICLR2026/reinforcement_learning/learning_to_play_multi-follower_bayesian_stackelberg_games.md)
- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)

</div>

<!-- RELATED:END -->
