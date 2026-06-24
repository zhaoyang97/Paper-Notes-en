---
title: >-
  [Paper Note] Learning-Augmented Hierarchical Clustering
description: >-
  [ICML 2025][Algorithms][hierarchical clustering] This paper investigates leveraging side information from a splitting oracle to bypass the approximation hardness barriers of hierarchical clustering, achieving an $O(1)$-approximation for the Dasgupta objective and a $(1-o(1))$-approximation for the Moseley-Wang objective, with extensions to streaming and parallel computing settings.
tags:
  - "ICML 2025"
  - "Algorithms"
  - "Clustering"
  - "hierarchical clustering"
  - "learning-augmented algorithms"
  - "splitting oracle"
  - "Dasgupta objective"
  - "approximation algorithms"
date: 2026-05-08
content_hash: 4d8beb2ed29de4fe
---

# Learning-Augmented Hierarchical Clustering

**Conference**: ICML 2025  
**arXiv**: [2506.05495](https://arxiv.org/abs/2506.05495)  
**Code**: None  
**Area**: Algorithms / Clustering  
**Keywords**: hierarchical clustering, learning-augmented algorithms, splitting oracle, Dasgupta objective, approximation algorithms

## TL;DR
This paper investigates leveraging side information from a splitting oracle to bypass the approximation hardness barriers of hierarchical clustering, achieving an $O(1)$-approximation for the Dasgupta objective and a $(1-o(1))$-approximation for the Moseley-Wang objective, with extensions to streaming and parallel computing settings.

## Background & Motivation
**Background**: Hierarchical clustering (HC) is a fundamental technique in data analysis that recursively partitions a dataset into a tree-like structure. Dasgupta (STOC'16) introduced a classic objective function for hierarchical clustering, and Moseley-Wang (NeurIPS'17) proposed a dual objective.

**Limitations of Prior Work**: Strong approximation hardness barriers exist for both objective functions. Under the Small Set Expansion Hypothesis (SSEH), there is no polynomial-time constant-approximation algorithm for the Dasgupta objective, and the Moseley-Wang objective admits no $(1-C)$-approximation.

**Key Challenge**: The gap between the theoretically impossible high-quality approximations and the urgent practical demand for high-quality clustering in applications.

**Goal**: Utilize natural side information (an oracle) to bypass the approximation hardness.

**Key Insight**: The Learning-Augmented Algorithms paradigm—allowing algorithms access to a (potentially noisy) oracle.

**Core Idea**: A splitting oracle provides triplet relation information, which can guide hierarchical clustering to break through traditional hardness limits.

## Method

### Overall Architecture
Input: Weighted graph $G = (V, E, w)$, splitting oracle  
Output: Hierarchical clustering tree $T$  

Splitting oracle definition: Given a triplet $(u, v, w)$, it returns which vertex is "split off" from the other two in the optimal tree (i.e., its lowest common ancestor contains all three vertices).

### Key Designs

1. **$O(1)$-Approximation Algorithm for the Dasgupta Objective**:

    - **Function**: Constructs a low-cost hierarchical clustering tree using a splitting oracle.
    - **Mechanism**: Recursively utilizes the oracle to determine the optimal split points. For a node subset $S$, a pivot is selected, and other nodes are partitioned into those on the same side as the pivot and those on the opposite side by querying the oracle, thereby recursively partitioning the set.
    - **Design Motivation**: The triplet relations provided by the oracle can recover the topology of the optimal tree. Even with errors (when the error rate is bounded), a constant approximation can still be guaranteed.

2. **$(1-o(1))$-Approximation for the Moseley-Wang Objective**:

    - **Function**: Constructs an almost-optimal hierarchical clustering tree in near-linear time.
    - **Mechanism**: Utilizes the dual relationship between the Moseley-Wang objective and the Dasgupta objective. The oracle is first used to determine a coarse-grained structure, followed by local fine-tuning.
    - **Design Motivation**: The Moseley-Wang objective focuses on "good splits" (similar points in the same subtree), which is directly provided by the splitting oracle.

3. **Extensions to Sublinear Settings**:

    - **Function**: Designs streaming and parallel (PRAM) algorithms.
    - **Mechanism**:
        - Streaming version: Uses limited space to maintain key triplet relations and constructs the clustering tree after a single pass.
        - PRAM version: Parallelizes the recursive splitting process, taking advantage of the independent query property of the oracle.
    - **Design Motivation**: Large-scale datasets cannot fit entirely in memory, necessitating solutions with sublinear space/time complexity.

### Loss & Training
This paper is a purely theoretical algorithmic work.
- Dasgupta objective: $\min_T \sum_{(i,j) \in E} w_{ij} |T[i \vee j]|$, where $T[i \vee j]$ is the number of leaves under the lowest common ancestor of $i, j$.
- Moseley-Wang objective: $\max_T \sum_{(i,j) \in E} w_{ij} (n - |T[i \vee j]|)$.

## Key Experimental Results

### Main Results

| Objective Function | Metric | Ours | SOTA without Oracle | Hardness Lower Bound |
|---|---|---|---|---|
| Dasgupta | Approximation Ratio | $O(1)$ | $O(\sqrt{\log n})$ | No constant approximation (SSEH) |
| Moseley-Wang | Approximation Ratio | $1-o(1)$ | $\frac{2}{3}+\delta$ | $1-C$ (SSEH) |
| Dasgupta (Streaming) | Space/Approximation | $\tilde{O}(n)$/O(1) | None known | — |
| Dasgupta (PRAM) | Depth/Approximation | $\text{polylog}(n)$/O(1) | None known | — |

### Ablation Study

| Oracle Error Rate | Dasgupta Approx. Ratio | Moseley-Wang Approx. Ratio | Description |
|---|---|---|---|
| 0% (Perfect Oracle) | $O(1)$ optimal | $1-o(1)$ | Ideal case |
| $< 1/3$ | $O(1)$ | $(1-o(1))$ | Within fault-tolerance limit |
| $\to 1/2$ (Random Guessing) | Degenerated | Degenerated | Close to useless information |

### Key Findings
- The splitting oracle is a natural and powerful form of side information that makes "impossible" approximations possible.
- The algorithms show robustness to oracle errors (retaining performance guarantees when the error rate is $< 1/3$).
- Extensions to streaming and parallel settings demonstrate the practical deployability of the methods.

## Highlights & Insights
- Outstanding theoretical contribution: Proves for the first time that a natural oracle can bypass the approximation hardness of hierarchical clustering.
- The definition of the splitting oracle is highly intuitive and can be provided by domain experts or pre-trained models.
- A paradigm example of learning-augmented algorithms: strong guarantees are obtained even when the side information is imperfect.

## Limitations & Future Work
- The practical construction of splitting oracles (e.g., learning them via machine learning models) requires further exploration.
- There may still be room for optimizing the approximation ratio constants.
- Whether the results on weighted graphs can be further improved remains an open question.

## Related Work & Insights
- Classic objective functions of Dasgupta [STOC'16] and Moseley-Wang [NeurIPS'17].
- Learning-augmented algorithms represent a highly popular direction in algorithm design in recent years.
- Exploring other types of oracles (e.g., pairwise comparison oracles) could be a fruitful direction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bypassing approximation hardness with an oracle is a significant theoretical breakthrough.
- Experimental Thoroughness: ⭐⭐⭐ Primarily a theoretical work, hence the experiments are relatively limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretically rigorous under complete proofs.
- Value: ⭐⭐⭐⭐ Offers inspiring implications for the paradigm of algorithm design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICML 2025\] Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](../../NeurIPS2025/learning_theory/learning-augmented_online_bipartite_fractional_matching.md)
- [\[ICML 2025\] Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)
- [\[ICML 2025\] Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions](sparse-pivot_dynamic_correlation_clustering_for_node_insertions.md)

</div>

<!-- RELATED:END -->
