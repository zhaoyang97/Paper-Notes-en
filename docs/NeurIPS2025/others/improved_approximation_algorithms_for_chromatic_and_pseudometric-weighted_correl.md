---
title: >-
  [Paper Note] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering
description: >-
  [NeurIPS 2025][Correlation Clustering] For two important generalizations of Correlation Clustering—Chromatic CC and pseudometric-weighted CC—this paper achieves a 2.15-approximation and a tight 10/3-approximation…
tags:
  - "NeurIPS 2025"
  - "Correlation Clustering"
  - "Approximation Algorithms"
  - "LP Rounding"
  - "Chromatic CC"
  - "Pseudometric"
date: 2026-05-08
content_hash: d6071bf12160f9f6
---

# Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering

**Conference**: NeurIPS 2025
**arXiv**: [2505.21939](https://arxiv.org/abs/2505.21939)  
**Code**: None  
**Area**: Theoretical Computer Science / Combinatorial Optimization
**Keywords**: Correlation Clustering, Approximation Algorithms, LP Rounding, Chromatic CC, Pseudometric

## TL;DR

For two important generalizations of Correlation Clustering—Chromatic CC and pseudometric-weighted CC—this paper achieves a 2.15-approximation and a tight 10/3-approximation, respectively, via LP relaxation and carefully designed rounding functions, significantly improving upon the previous best results of 2.5 and 6.

## Background & Motivation

### Limitations of Prior Work

**Background**: Classical Correlation Clustering (CC) assigns each edge a "+" or "−" label and seeks a node partition minimizing disagreements. However, real-world edge relationships are more complex:

1. **Chromatic CC (CCC)**: Edges carry multiple semantic colors (e.g., "colleague/classmate/family" in social networks), each cluster must be assigned a single color, and the goal is to minimize the number of color-mismatched edges. The previous best approximation ratio was **2.5** by Xiu et al.
2. **Pseudometric-weighted CC**: Edge weights satisfy the triangle inequality, and the goal is to minimize weighted disagreements. The previous best result was a **6-approximation**.

Both problems are APX-hard, and general weighted CC admits no constant-factor approximation under the Unique Games Conjecture. The core contribution of this paper lies in advancing the approximation frontier for both problems within the standard LP relaxation + rounding function framework, through refined rounding function design and triple-based analysis.

## Method

### Overall Architecture

A unified **triple-based LP rounding analysis framework** is employed: for each triple of vertices $(u, v, w)$, the algorithm's expected cost satisfies $\text{ALG}(uvw) \leq \alpha \cdot \text{LP}(uvw)$, and summing over all triples yields a global approximation guarantee. The key lies in designing an optimal rounding function for each problem.

### Key Designs

1. **Rounding Function for Pseudometric-Weighted CC**:
    - **Function**: Maps LP solutions $x \in [0,1]$ to clustering probabilities via a piecewise constant/linear function.
    - **Mechanism**: $f^+(x) = f^-(x) = \begin{cases} 0 & x < 0.4 \\ 5/3 \cdot x & 0.4 \leq x < 0.6 \\ 1 & x \geq 0.6 \end{cases}$, using a linear mapping on $[0.4, 0.6)$.
    - **Design Motivation**: Upper and lower bound constraints from configurations $(x, 1-x, 1)$ and $(x, x, 2x)$ uniquely determine the optimal piecewise function form.

2. **Color-Cluster Decoupling and Multi-Type Rounding for CCC**:
    - **Function**: Decouples color assignment from clustering decisions, introducing three rounding functions ($f^+$, $f^-$, $f^\circ$) to handle same-color, adversarial-color, and neutral edges, respectively.
    - **Mechanism**: A two-phase approach—first assign colors via LP solution (majority voting), then run LP-Pivot within each color class; $f^\circ(x) = 1.7x\ (x < 0.5)$ or $0.3x + 0.7\ (x \geq 0.5)$ is specifically designed to control the cost of neutral edges.
    - **Design Motivation**: Edge types in CCC are diverse (same-color, adversarial-color, neutral), requiring separate cost control for each type to ensure the overall ratio does not exceed 2.15.

3. **Tight Lower Bound Construction**:
    - **Function**: Establishes impossibility results for approximation ratios within the LP-Pivot framework.
    - **Mechanism**: A contradiction via upper and lower bounds on $f^\circ(1/2)$ proves the CCC lower bound of 2.11; analysis of extremal configurations proves the pseudometric-weighted CC lower bound of 10/3.
    - **Design Motivation**: Matching (or near-matching) upper and lower bounds demonstrate the optimality of the algorithms within this framework.

### Algorithm Pipeline

LP-Pivot Algorithm (Algorithm 1): (1) randomly select a pivot node $v$; (2) for each other node $u$, compute $p_{uv}$ using the rounding function corresponding to the edge type; (3) merge $u$ into the pivot's cluster with probability $1 - p_{uv}$; (4) recursively process remaining nodes.

## Key Experimental Results

### Main Results

| Problem | Upper Bound (Ours) | Lower Bound (Ours) | Prev. SOTA | Gain |
|---|---|---|---|---|
| Pseudometric-weighted CC | **10/3 ≈ 3.33** | **10/3** (tight) | 6 | **44.4%** |
| Chromatic CC (CCC) | **2.15** | **2.11** | 2.5 | **14%** |

### Key Findings

- **The pseudometric-weighted CC result is tight**: Upper and lower bounds match exactly at 10/3, indicating no room for improvement within the LP-Pivot framework.
- **CCC is near-optimal**: The gap between upper and lower bounds is only 0.04 (2.15 vs. 2.11); whether it can be closed remains open.
- **Lemma 4 (weight reduction)** is a key technique: pseudometric weight constraints under the triangle inequality form a convex cone, requiring verification of only three extremal configurations $(1,1,0)$, $(1,0,1)$, $(0,1,1)$.

## Highlights & Insights

- **Elegant tight lower bound construction**: The perfect matching at 10/3 showcases the full potential of LP rounding analysis.
- **Methodological value**: The triple-based analysis combined with exhaustive region-by-region verification (Regions I–VI) is transferable to other clustering problems.
- **The "art" of rounding function design**: Analytical bounds are first used to delineate forbidden regions, after which the optimal function is identified within the feasible region.

## Limitations & Future Work

- **Purely theoretical work**: No empirical validation is provided; the gap with heuristics such as Greedy Expansion on real data remains unclear.
- **Framework limitations**: Results are confined to the standard LP relaxation + rounding framework; stronger relaxations such as the Sherali-Adams hierarchy or cluster LP may yield further improvements.
- **Remaining gap for CCC**: A gap of approximately 0.04 remains between the lower bound 2.11 and upper bound 2.15.
- **LP solving overhead**: In practice, LP solving time may become a bottleneck for large-scale graphs.
- **Dynamic/streaming settings not covered**: Many practical applications require streaming or online formulations.

## Related Work & Insights

- **Evolution of classical CC**: $3 \to 2.06 \to 1.994 \to 1.73 \to 1.437$, primarily driven by stronger LP relaxations.
- **Chawla et al.**: Provided the classical CC LP-Pivot rounding functions ($f^+$, $f^-$), upon which this paper builds.
- **Charikar & Gao**: Previous best 6-approximation for pseudometric-weighted CC via LP-UMVD-Pivot.
- **Xiu et al.**: Previous best 2.5-approximation for CCC.
- **Implications for combinatorial optimization**: Whether CCC and pseudometric-weighted CC can follow a similar trajectory to classical CC via stronger relaxations is a promising direction.

## Rating

- Novelty: ⭐⭐⭐⭐ — tight lower bound + novel rounding function design
- Experimental Thoroughness: ⭐⭐ — purely theoretical, no experiments
- Writing Quality: ⭐⭐⭐⭐⭐ — proofs are clear and well-structured; suitable as a reference for LP rounding pedagogy
- Value: ⭐⭐⭐⭐ — advances the theoretical frontier for two important CC variants

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](../../AAAI2026/others/improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[NeurIPS 2025\] Stable Matching with Ties: Approximation Ratios and Learning](stable_matching_with_ties_approximation_ratios_and_learning.md)
- [\[ICML 2026\] Estimating Correlation Clustering Cost in Node-Arrival Stream](../../ICML2026/others/estimating_correlation_clustering_cost_in_node-arrival_stream.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)

</div>

<!-- RELATED:END -->
