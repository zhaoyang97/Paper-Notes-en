---
title: >-
  [Paper Note] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach
description: >-
  [AAAI 2026][k-center clustering] This paper studies the k-center clustering problem with instance-level cannot-link (CL) and must-link (ML) constraints. It proposes a local search framework based on a dominating matching set (DMS) reduction, and, under the disjoint CL sets condition, is the first to achieve the optimal approximation ratio of 2 via local search—resolving an open problem in the field.
tags:
  - "AAAI 2026"
  - "k-center clustering"
  - "constrained clustering"
  - "local search"
  - "approximation ratio"
  - "cannot-link"
  - "must-link"
  - "dominating matching set"
date: 2026-05-08
content_hash: ec94d98aedf7e6a9
---

# Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach

**Conference**: AAAI 2026
**arXiv**: [2601.11883](https://arxiv.org/abs/2601.11883)  
**Code**: [https://github.com/ChaoqiJia/LSCKC](https://github.com/ChaoqiJia/LSCKC)  
**Area**: Theoretical Algorithms / Clustering / Approximation Algorithms
**Keywords**: k-center clustering, constrained clustering, local search, approximation ratio, cannot-link, must-link, dominating matching set

## TL;DR

This paper studies the k-center clustering problem with instance-level cannot-link (CL) and must-link (ML) constraints. It proposes a local search framework based on a dominating matching set (DMS) reduction, and, under the disjoint CL sets condition, is the first to achieve the optimal approximation ratio of 2 via local search—resolving an open problem in the field.

## Background & Motivation

k-center is a fundamental theoretical problem in clustering and data analysis: given $n$ data points and a positive integer $k$, select $k$ centers to minimize the maximum distance from any point to its nearest center. The optimal approximation ratio for standard k-center is 2 (Gonzalez 1985; Hochbaum & Shmoys 1985), and any $2-\epsilon$ approximation implies $\mathcal{P}=\mathcal{NP}$.

In semi-supervised learning and practical applications, background knowledge is often expressed as instance-level constraints:
- **Cannot-link (CL)**: two points cannot belong to the same cluster, e.g., modeling conflict relationships
- **Must-link (ML)**: two points must belong to the same cluster, e.g., label propagation

**Theoretical difficulty**: General CL constraints lead to inapproximability—finding a clustering satisfying all CL constraints is itself NP-hard (via reduction from k-coloring). However, Guo et al. (2024) showed that when CL sets are disjoint (i.e., no point is shared across CL sets), a constant-factor approximation ratio of 2 is still achievable. Their approach relies on a reverse dominating set (RDS) construction, but **whether local search can attain the optimal approximation ratio of 2 in this setting has remained an open problem**.

## Method

### Overall Architecture

A three-stage algorithm (Algorithm 3):
1. **Stage 1(a)**: Ignoring CL constraints, apply a threshold algorithm (Algorithm 1) to find a center set $C_1$ that handles ML constraints.
2. **Stage 1(b)**: Find candidate centers $C_2$ for CL sets such that $C_1 \cup C_2$ covers all CL-constrained points. For each CL set $Y_i$, construct a bipartite graph and find a maximum matching; unmatched CL points are added to the candidates.
3. **Stage 2**: Apply local search (Algorithm 2) to $C_2$, using an enhanced single swap operation to reduce the number of centers to $\leq k$.

### Key Designs

1. **Reduction to DMS**: CL constraints are represented via an auxiliary $l$-partite graph $G_\mathcal{Y}$, where vertices are CL-set points and edges connect pairs from different CL sets within distance $\leq \eta$. A dominating matching set (DMS) is defined on this graph—a center set $\Gamma$ must ensure that for each CL set $Y$, a threshold maximum matching exists between $\Gamma \setminus Y$ and $Y \setminus \Gamma$, with matching size exactly $|Y \setminus \Gamma|$.
2. **Swap-free DMS (SF-DMS)**: Computing a minimum DMS is NP-hard (even when $|Y_i|=1$ and points lie in the plane), but an SF-DMS can be computed in polynomial time. An SF-DMS satisfies: for any $p \in V$ and $\{u,v\} \subseteq \Gamma$, $\Gamma \cup \{p\} \setminus \{u,v\}$ is no longer a DMS—i.e., no improvement is possible by adding one point and removing two.
3. **Core Lemma (Lemma 8)**: The size of an SF-DMS satisfies $|\Gamma| \leq k_Y$, where $k_Y$ is the number of clusters in the optimal solution that contain CL points. This guarantees that the number of centers upon termination of local search is $\leq k$.
4. **No dependence on known opt**: By Lemma 11, the observation that opt must equal some pairwise distance allows enumeration of candidate distance values to find the minimum feasible threshold.

### Complexity

The algorithm runs in $O(n^2 k^{4.5})$ time: Stage 1(a) in $O(kn)$, Stage 1(b) in $O(nk^{3/2})$ (Hopcroft–Karp matching), and Stage 2 local search in $O(k)$ rounds $\times$ $O(nk^2)$ swap pairs $\times$ $O(nk^{1.5})$ feasibility checks.

## Key Experimental Results

### Main Results: Empirical Approximation Ratio (Synthetic Dataset, $k=50$)

| Algorithm | Mean Empirical Ratio | Max Ratio | Theoretical Guarantee |
|-----------|---------------------|-----------|----------------------|
| Greedy_H | Highest | >2 | None |
| Matching_H | High | >2 | None |
| Approx (Guo et al.) | Moderate | <2 | 2-approximation |
| **LSCKC (Ours)** | **Lowest** | **<2** | **2-approximation** |

### Clustering Cost on Real Datasets ($k=30$, Disjoint CL/ML Constraints)

| Dataset | Constraint Ratio | LSCKC Cost | Approx Cost | Advantage |
|---------|-----------------|------------|-------------|-----------|
| Cnae-9 | 2%–10% | Lowest | Higher | More pronounced with more constraints |
| Skin | 2%–10% | Lowest | Higher | Significantly outperforms all baselines |
| Covertype | 2%–10% | Lowest | Higher | Consistently leading |

### Key Findings

- LSCKC achieves the lowest clustering cost across all constraint ratios (2%–10%) and repetition ratios (10%–50%).
- As the number of constraints increases, the empirical approximation ratio tends to rise for all algorithms (due to a shrinking feasible region), but LSCKC exhibits the slowest rate of increase.
- The swap step in local search enables exploration of better centers from non-constrained points—something the baseline Approx cannot achieve.
- LSCKC performs well even under crossing CL constraints, despite the absence of theoretical guarantees in that setting.
- On Cnae-9, LSCKC's advantage grows with more constraints: additional constraints restrict the solution space of Approx, while local search can still select centers from non-constrained points.
- Experimental setup: Java 1.8.0, Apple M1 Max CPU, 32 GB RAM; each configuration repeated at least 20 times and averaged.

## Highlights & Insights

- **Resolving an open problem**: This work provides the first proof that local search can achieve the optimal approximation ratio of 2 for k-center with disjoint CL constraints, answering the open question posed by Guo et al. (2024).
- **Elegance of the DMS reduction**: Translating constrained clustering into a dominating matching set on a graph enables the use of matching-theoretic tools within local search. The key insight is that while a minimum DMS is NP-hard to compute, an SF-DMS is polynomial—this "just-enough structure" is the core theoretical contribution.
- **Enhanced single swap**: Unlike the standard 1-swap in local search, the proposed enhanced swap adds one point and removes two, yielding stronger search capability.
- **Ingenuity of the inductive proof**: The proof of Lemma 8 analyzes coverage of optimal clusters via Case (a) and Case (b), using proof by contradiction and sub-problem recursion to establish $|\Gamma| \leq k_Y$.

## Limitations & Future Work

- **Restricted to disjoint CL sets**: Theoretical guarantees do not hold when CL sets share points (crossing constraints), although empirical performance remains strong.
- **Runtime $O(n^2 k^{4.5})$**: May be slow for large-scale datasets; maximum matching computation is the bottleneck.
- **Inherent limitations of k-center**: The maximum-distance objective is sensitive to outliers; k-median or k-means may be more practical in applied settings.
- Experiments are conducted only on 3 UCI datasets and synthetic data, with relatively small scale.
- The possibility of extending the DMS framework to other constrained clustering variants (e.g., k-means with CL/ML) is not explored.

## Related Work & Insights

| Work | Constraint Type | Method | Approx. Ratio | Range of $k$ |
|------|----------------|--------|---------------|--------------|
| Davidson et al. 2010 | CL+ML | SAT-based | $(1+\epsilon)$ | $k=2$ |
| Brubach et al. 2021 | ML only | — | 2 | General $k$ |
| Guo et al. 2024 | Disjoint CL | Reverse dominating set (RDS) | 2 | General $k$ |
| **Ours** | **Disjoint CL + ML** | **Local search (DMS)** | **2** | **General $k$** |

## Rating

- Novelty: ⭐⭐⭐⭐ Resolves an open problem; the DMS reduction is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐ Primarily a theoretical work; experiments serve as auxiliary validation.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation and proof structure are clearly presented.
- Value: ⭐⭐⭐ Valuable to the theoretical algorithms community; application scope is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GDBA Revisited: Unleashing the Power of Guided Local Search for Distributed Constraint Optimization](gdba_revisited_unleashing_the_power_of_guided_local_search_for_distributed_const.md)
- [\[AAAI 2026\] A Fast Heuristic Search Approach for Energy-Optimal Profile Routing for Electric Vehicles](a_fast_heuristic_search_approach_for_energy-optimal_profile_.md)
- [\[ICML 2025\] Modified K-means Algorithm with Local Optimality Guarantees](../../ICML2025/others/modified_k-means_algorithm_with_local_optimality_guarantees.md)
- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](../../NeurIPS2025/others/improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](local_guidance_for_configuration-based_multi-agent_pathfinding.md)

</div>

<!-- RELATED:END -->
