---
title: >-
  [Paper Note] EFX and PO Allocation Exists for Two Types of Goods
description: >-
  [AAAI2026][AI Safety][fair division] It is proved that when there are only two types of goods and all valuations are positive, an allocation satisfying both EFX (Envy-Freeness up to any good) and Pareto optimality always exists, and a quasi-linear time algorithm is provided.
tags:
  - "AAAI2026"
  - "AI Safety"
  - "fair division"
  - "EFX"
  - "Pareto optimality"
  - "indivisible goods"
  - "additive valuations"
date: 2026-05-08
content_hash: 6ff0cc18c9a3b4c8
---

# EFX and PO Allocation Exists for Two Types of Goods

**Conference**: AAAI2026  
**arXiv**: [2601.03438](https://arxiv.org/abs/2601.03438)  
**Code**: None (theoretical work)  
**Area**: AI Safety  
**Keywords**: fair division, EFX, Pareto optimality, indivisible goods, additive valuations  

## TL;DR

It is proved that when there are only two types of goods and all valuations are positive, an allocation satisfying both EFX (Envy-Freeness up to any good) and Pareto optimality always exists, and a quasi-linear time algorithm is provided.

## Background & Motivation

The fair division of indivisible goods is a central research problem in algorithmic game theory and computational social choice. Classical Envy-Freeness (EF) requires that no agent prefers another's bundle. However, for indivisible goods, an EF allocation often does not exist, and even when it does, finding it is NP-complete.

EFX (Envy-Freeness up to Any good) is an important relaxation of EF: it allows an agent to envy another, but the envy disappears after removing **any single** good from the envied bundle. Whether EFX always exists for any number of agents remains a core open problem in the field of fair division.

Prior work has proved the existence of EFX in various restricted settings: identical valuations, two agents, ordered instances, at most three valuations, and two types of goods, among others. In particular, Garg & Murhekar proved that EFX exists when there are two types of goods, but without guaranteeing Pareto optimality. The motivation of this work is to fill this gap—simultaneously guaranteeing both **fairness** (EFX) and **efficiency** (PO).

## Core Problem

**Core Problem**: When there are only two types of goods (each with multiple copies) and agents have additive positive valuations, does there always exist an allocation that simultaneously satisfies EFX and Pareto optimality?

**Formal Settings**:

- $n$ agents, the set of goods $M$ contains $m_1$ goods of the first type and $m_2$ goods of the second type
- Each agent $i$ has a valuation $v_{i,j} > 0$ for goods of type $j$ (positive valuation assumption)
- An allocation $X = (X_1, \ldots, X_n)$ is a partition of $M$
- **EFX**: For any $i, j$ and $g \in X_j$, $v_i(X_i) + v_i(g) \geq v_i(X_j)$
- **PO**: There is no other allocation that makes at least one agent strictly better off without making any agent worse off

## Method

### 1. Input Preprocessing

The algorithm first standardizes the input:

- **Grouping**: Agents are split into two groups—those who prefer the first type of goods ($v_{i,2} \leq v_{i,1}$) and those who prefer the second type.
- **Normalization**: The valuations of each agent are normalized to $(1, v_{i,2}/v_{i,1})$, standardizing the valuation of the first type of goods to 1.
- **Sorting**: Agents are sorted in ascending order of $v_{i,2}$.
- **Constraint Guarantee**: Ensure that $m_1/n_1 \geq m_2/n_2$; otherwise, the two types of goods are swapped.

### 2. Proper Allocation and Pareto Optimality

The authors introduce the concept of a **proper allocation**: there exists a split point $t$ such that agents before $t$ receive only the first type of goods, while the number of the first type of goods received by agents after $t$ is upper-bounded. The core theorem proves that all proper allocations are Pareto optimal, providing an efficiency guarantee for subsequent constructions.

### 3. Split Allocation

The core allocation construction is a **$(t, k)$-split-allocation**:

- Agent $t$ acts as the "split point", receiving both types of goods (specifically, receiving $k$ goods of the second type).
- Agents $1, \ldots, t-1$ receive only the first type of goods (allocated equitably).
- Agents $t+1, \ldots, n$ receive only the second type of goods (allocated equitably).
- The remaining goods of the first type are allocated among agents $1, \ldots, t$.

The allocation utilizes the **Prioritized Equitable Allocation (PEA)** mechanism—allocating almost equitably among designated agents, with higher-priority agents receiving more. Since each split allocation is proper, it automatically satisfies PO.

### 4. Envy Direction Analysis

Key finding: If a split allocation is not EFX, the direction of envy is **unidirectional**. Definitions:

- **Left Envy (LE)**: There exists a higher-indexed agent envying a lower-indexed agent.
- **Right Envy (RE)**: There exists a lower-indexed agent envying a higher-indexed agent.

**Theorem 2** proves that no split allocation can simultaneously exhibit both left envy and right envy. **Theorem 3** further proves that the minimum split allocation is either EFX or exhibits only LE, while the maximum is either EFX or exhibits only RE.

### 5. Binary Search and Reallocation

If no split allocation is EFX, the envy direction must transition from LE to RE at some point. The algorithm employs **binary search** to find this "envy direction flip point" $(t, k)$.

At the flip point, a **$(t, k)$-reallocation** is constructed: based on the split allocation, some goods of the first type are reallocated from left agents to right agents (the exact reallocation quantity is controlled by $\lceil d \cdot v_{t,2} \rceil$). **Theorem 4** proves that such a reallocation satisfies both EFX and PO.

### 6. Time Complexity

The overall running time of the algorithm is $\mathcal{O}(n \log n + \log m)$; if the agents are already sorted by their relative valuations, it only requires $\mathcal{O}(\log n + \log m)$.

## Key Experimental Results

This work is purely theoretical and does not contain experimental data. The main theoretical results are summarized below:

| Results | Description |
|------|------|
| Existence of EFX+PO | Always exists under two types of goods and positive additive valuations |
| Time Complexity | $\mathcal{O}(n \log n + \log m)$ |
| Incompatibility of EFX+fPO | Proves that EFX and fractional PO (fPO) are incompatible through a counterexample |
| Special Case | A trivial solution exists when $m_1 + m_2 \leq n$ |

## Highlights & Insights

- **Filling a Theoretical Gap**: This work is the first to simultaneously guarantee EFX and PO in a two-commodity setting, strengthening prior results that only guaranteed EFX.
- **Elegant and Clean Algorithm**: The three-step construction based on split allocation, binary search for envy direction, and reallocation is highly natural.
- **Quasi-linear Efficiency**: The $\mathcal{O}(n \log n + \log m)$ complexity is extremely efficient, significantly lower than general fair division algorithms.
- **Proper Allocation Framework**: The introduced concept of proper allocation unifies the proof of PO, so that all intermediate and final allocations automatically satisfy PO.
- **Monotonicity of Envy Direction**: The discovery of the monotonic property of envy direction in the split allocation family is the key insight that enables binary search.

## Limitations & Future Work

- **Restricted Types of Goods**: It is only applicable to two types of goods, which is still far from the general case (arbitrary types or arbitrary goods).
- **Positive Valuations Assumption**: It requires $v_{i,j} > 0$ for all $i, j$, which is inapplicable to scenarios where agents have zero valuation for some goods (it is known that EFX+PO can be incompatible under zero valuations).
- **Unachievability of EFX+fPO**: Although discrete PO is guaranteed, the stronger fractional PO (fPO) remains incompatible with EFX.
- **Valuation Restrictions**: It only handles additive valuations and does not cover submodular or more general valuation functions.
- **No Experimental Validation**: It lacks empirical validation in real-world scenarios (e.g., resource allocation platforms).

## Related Work & Insights

| Work | Setting | Guarantees |
|------|------|------|
| Plaut & Roughgarden (2020) | 2 agents | EFX |
| Chaudhury et al. (2020) | 3 agents | EFX |
| Mahara (2023) | 2 valuation types | EFX |
| Garg & Murhekar (2024) | 2 types of goods | EFX (no PO guarantee) |
| Amanatidis et al. (2021) | bivalued instances | EFX+PO |
| **Ours** | **2 types of goods, positive valuations** | **EFX+PO, $\mathcal{O}(n\log n + \log m)$** |

The core improvement of this work compared to Garg & Murhekar is the additional guarantee of Pareto optimality under the same setting, along with a more efficient algorithm.

## Insights & Connections

- **Methodological Insight**: The analysis of the monotonicity of envy directions is worth generalizing to other fair division problems—when a certain monotonic structure exists within a family of allocations, binary search can efficiently locate an EFX allocation.
- The **proper allocation framework** might provide valuable insights for solving EFX+PO problems with more types of goods.
- **Future Directions of Interest**: The existence of EFX+PO with three or more types of goods, relaxed versions with zero valuations, and generalizations to non-additive valuations.

## Rating

- Novelty: ⭐⭐⭐⭐ (Naturally and meaningfully adding PO guarantees on top of existing EFX results)
- Experimental Thoroughness: N/A (Purely theoretical work)
- Writing Quality: ⭐⭐⭐⭐ (Clear construction, step-by-step progression, with sufficient preliminary definitions)
- Value: ⭐⭐⭐⭐ (Advances the core open problem of fair division, though a gap remains to the general solution)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation](matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)
- [\[ICML 2026\] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](../../ICML2026/ai_safety/position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)
- [\[ICLR 2026\] Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](../../ICLR2026/ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)
- [\[CVPR 2026\] DeepfakeImpact: A Two-Stage Benchmark with Real-World Impact in Deepfake Detection](../../CVPR2026/ai_safety/deepfakeimpact_a_two-stage_benchmark_with_real-world_impact_in_deepfake_detectio.md)
- [\[ICLR 2026\] Differentially Private Two-Stage Gradient Descent for Instrumental Variable Regression](../../ICLR2026/ai_safety/differentially_private_two-stage_gradient_descent_for_instrumental_variable_regr.md)

</div>

<!-- RELATED:END -->
