---
title: >-
  [Paper Note] EFX and PO Allocation Exists for Two Types of Goods
description: >-
  [AAAI2026][AI Safety][fair division] This paper proves that an allocation satisfying both EFX (envy-freeness up to any good) and Pareto optimality always exists when goods are of only two types and all valuations are positive, and provides a near-linear-time algorithm.
tags:
  - AAAI2026
  - AI Safety
  - fair division
  - EFX
  - Pareto optimality
  - indivisible goods
  - additive valuations
date: 2026-05-08
content_hash: 5dc32a3d88fb7188
---

# EFX and PO Allocation Exists for Two Types of Goods

**Conference**: AAAI2026
**arXiv**: [2601.03438](https://arxiv.org/abs/2601.03438)
**Code**: None (theoretical work)
**Area**: AI Safety
**Keywords**: fair division, EFX, Pareto optimality, indivisible goods, additive valuations

## TL;DR

This paper proves that an allocation satisfying both EFX (envy-freeness up to any good) and Pareto optimality always exists when goods are of only two types and all valuations are positive, and provides a near-linear-time algorithm.

## Background & Motivation

Fair allocation of indivisible goods is a central research problem in algorithmic game theory and computational social choice. Classical envy-freeness (EF) requires that no agent prefers another agent's bundle, but EF allocations for indivisible goods often do not exist and, when they do, are NP-hard to compute.

EFX (Envy-Freeness up to Any good) is an important relaxation of EF: an agent may envy another, but the envy disappears upon removing **any single** good from the envied bundle. Whether EFX allocations always exist for an arbitrary number of agents remains a central open problem in fair division.

Prior work has established EFX existence in several restricted settings: identical valuations, two agents, ordered instances, at most three distinct valuations, and two types of goods. In particular, Garg & Murhekar proved EFX existence for two types of goods but without guaranteeing Pareto optimality. This paper is motivated by closing that gap—simultaneously guaranteeing both **fairness** (EFX) and **efficiency** (PO).

## Core Problem

**Main Research Question**: When goods are of only two types (each with multiple copies) and agents have additive positive valuations, does an allocation satisfying both EFX and Pareto optimality always exist?

**Formal Setting**:

- $n$ agents; good set $M$ contains $m_1$ copies of type-1 goods and $m_2$ copies of type-2 goods
- Each agent $i$ has valuation $v_{i,j} > 0$ for type-$j$ goods (positivity assumption)
- An allocation $X = (X_1, \ldots, X_n)$ is a partition of $M$
- **EFX**: for all $i, j$ and $g \in X_j$, $v_i(X_i) + v_i(g) \geq v_i(X_j)$
- **PO**: no reallocation makes some agent better off without making any agent worse off

## Method

### 1. Input Preprocessing

The algorithm first normalizes the input:

- **Grouping**: agents are partitioned into those preferring type-1 goods ($v_{i,2} \leq v_{i,1}$) and those preferring type-2 goods
- **Normalization**: each agent's valuations are normalized to $(1, v_{i,2}/v_{i,1})$, setting the type-1 valuation to 1
- **Sorting**: agents are ordered by $v_{i,2}$ in non-decreasing order
- **Constraint enforcement**: $m_1/n_1 \geq m_2/n_2$ is ensured; otherwise good types are swapped

### 2. Proper Allocations and Pareto Optimality

The authors introduce the notion of a **proper allocation**: there exists a threshold $t$ such that agents before $t$ receive only type-1 goods, and agents after $t$ receive a bounded number of type-1 goods. A core theorem establishes that all proper allocations are Pareto optimal, providing an efficiency guarantee for subsequent constructions.

### 3. Split Allocations

The central allocation construction is the **$(t, k)$-split-allocation**:

- Agent $t$ is the "split point," receiving goods of both types ($k$ type-2 goods)
- Agents $1, \ldots, t-1$ receive only type-1 goods (distributed nearly equally)
- Agents $t+1, \ldots, n$ receive only type-2 goods (distributed nearly equally)
- Remaining type-1 goods are distributed among agents $1, \ldots, t$

Distribution follows the **Prioritized Equitable Allocation (PEA)** mechanism—near-equal allocation among designated agents, with higher-priority agents receiving slightly more. Every split allocation is proper and thus automatically satisfies PO.

### 4. Analysis of Envy Directions

A key observation is that if a split allocation is not EFX, envy is **unidirectional**. Define:

- **Left Envy (LE)**: some agent with a larger index envies an agent with a smaller index
- **Right Envy (RE)**: some agent with a smaller index envies an agent with a larger index

**Theorem 2** proves that no split allocation can exhibit both LE and RE simultaneously. **Theorem 3** further establishes that the minimal split allocation is either EFX or exhibits only LE, and the maximal split allocation is either EFX or exhibits only RE.

### 5. Binary Search and Reallocation

If no split allocation is EFX, the envy direction must transition from LE to RE at some point. The algorithm applies **binary search** to locate this "envy-direction flip point" $(t, k)$.

At the flip point, a **$(t, k)$-reallocation** is constructed: starting from the split allocation, a portion of type-1 goods is redistributed from left-side agents to right-side agents (the exact quantity governed by $\lceil d \cdot v_{t,2} \rceil$). **Theorem 4** proves that this reallocation satisfies both EFX and PO.

### 6. Time Complexity

The overall algorithm runs in $\mathcal{O}(n \log n + \log m)$ time; if agents are already sorted by relative valuations, this reduces to $\mathcal{O}(\log n + \log m)$.

## Key Experimental Results

This is a purely theoretical paper with no experimental data. Main theoretical results:

| Result | Description |
|--------|-------------|
| EFX+PO existence | Always exists for two good types with positive additive valuations |
| Time complexity | $\mathcal{O}(n \log n + \log m)$ |
| EFX+fPO incompatibility | A counterexample shows EFX and fractional PO are incompatible |
| Special case | Trivial solution exists when $m_1 + m_2 \leq n$ |

## Highlights & Insights

- **Closes a theoretical gap**: the first result simultaneously guaranteeing EFX and PO under the two-good-type setting, strengthening prior work that guaranteed only EFX
- **Elegant construction**: the three-step framework of split allocations + envy-direction binary search + reallocation is clean and natural
- **Near-linear efficiency**: the $\mathcal{O}(n \log n + \log m)$ complexity is highly efficient, far below that of general fair allocation algorithms
- **Proper allocation framework**: the introduced notion of proper allocations unifies the PO argument, ensuring all intermediate and final allocations satisfy PO automatically
- **Monotonicity of envy directions**: the monotonicity of envy directions within the family of split allocations is the key insight enabling binary search

## Limitations & Future Work

- **Restricted good types**: applicable only to two types of goods; a substantial gap remains to the general case of arbitrarily many types or arbitrary goods
- **Positivity assumption**: requires $v_{i,j} > 0$ for all $i, j$; inapplicable when agents assign zero value to some good type (EFX+PO is known to be incompatible under zero valuations)
- **EFX+fPO unachievable**: while discrete PO is guaranteed, the stronger fractional PO (fPO) remains incompatible with EFX
- **Additive valuations only**: restricted to additive valuations; submodular or more general valuation functions are not addressed
- **No empirical validation**: lacks empirical evaluation on practical scenarios such as resource allocation platforms

## Related Work & Insights

| Work | Setting | Guarantee |
|------|---------|-----------|
| Plaut & Roughgarden (2020) | 2 agents | EFX |
| Chaudhury et al. (2020) | 3 agents | EFX |
| Mahara (2023) | 2 valuation types | EFX |
| Garg & Murhekar (2024) | 2 good types | EFX (no PO guarantee) |
| Amanatidis et al. (2021) | Bivalued instances | EFX+PO |
| **Ours** | **2 good types, positive valuations** | **EFX+PO, $\mathcal{O}(n\log n + \log m)$** |

The core advance over Garg & Murhekar is the additional Pareto optimality guarantee under the same setting, along with a more efficient algorithm.

- **Methodological insight**: the envy-direction monotonicity analysis is worth generalizing to other fair division problems—when a family of allocations exhibits some monotone structure, binary search may efficiently locate an EFX allocation
- The **proper allocation framework** may offer a useful blueprint for EFX+PO problems with more good types
- **Promising future directions**: EFX+PO existence for three or more good types, relaxed versions with zero valuations, and extensions to non-additive valuations

## Rating

- Novelty: ⭐⭐⭐⭐ (naturally and meaningfully augments existing EFX results with a PO guarantee)
- Experimental Thoroughness: N/A (purely theoretical work)
- Writing Quality: ⭐⭐⭐⭐ (clear and progressive construction with well-prepared definitions)
- Value: ⭐⭐⭐⭐ (advances a core open problem in fair division, though a general solution remains distant)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation](matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)
- [\[ICLR 2026\] Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](../../ICLR2026/ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)
- [\[NeurIPS 2025\] PubSub-VFL: Towards Efficient Two-Party Split Learning in Heterogeneous Environments via Publisher/Subscriber Architecture](../../NeurIPS2025/ai_safety/pubsub-vfl_towards_efficient_two-party_split_learning_in_heterogeneous_environme.md)
- [\[AAAI 2026\] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias](easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)
- [\[AAAI 2026\] Enhancing DPSGD via Per-Sample Momentum and Low-Pass Filtering](enhancing_dpsgd_via_per-sample_momentum_and_low-pass_filtering.md)

</div>

<!-- RELATED:END -->
