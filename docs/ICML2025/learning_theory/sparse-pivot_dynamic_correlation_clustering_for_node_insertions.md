---
title: >-
  [Paper Note] Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions
description: >-
  [ICML2025][Miscellaneous/Graph Clustering][Correlation Clustering] This paper proposes the Sparse-Pivot algorithm, which achieves a $(20+\varepsilon)$-approximation for the correlation clustering problem under dynamic node insertions with amortized $O_\varepsilon(\log^{O(1)} n)$ database operations. It significantly improves the approximation factor of Cohen-Addad et al. (ICML 2024) and comprehensively outperforms benchmarks in experiments.
tags:
  - "ICML2025"
  - "Miscellaneous/Graph Clustering"
  - "Correlation Clustering"
  - "Dynamic Graph Algorithms"
  - "Node Insertion"
  - "Sublinear Update Time"
  - "Approximation Algorithms"
date: 2026-05-08
content_hash: ea4774c5f3b050e1
---

# Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions

**Conference**: ICML2025  
**arXiv**: [2507.01830](https://arxiv.org/abs/2507.01830)  
**Code**: Not released  
**Area**: Miscellaneous/Graph Clustering  
**Keywords**: Correlation Clustering, Dynamic Graph Algorithms, Node Insertion, Sublinear Update Time, Approximation Algorithms

## TL;DR

This paper proposes the Sparse-Pivot algorithm, which achieves a $(20+\varepsilon)$-approximation for the correlation clustering problem under dynamic node insertions with amortized $O_\varepsilon(\log^{O(1)} n)$ database operations. It significantly improves the approximation factor of Cohen-Addad et al. (ICML 2024) and comprehensively outperforms benchmarks in experiments.

## Background & Motivation

**Correlation Clustering** is a classic graph partitioning problem: given "similar/dissimilar" labels for vertex pairs in a complete graph, the goal is to find a clustering that minimizes the number of "disagreed" edges (dissimilar within the same cluster + similar across different clusters). Introduced by Bansal et al. (2004), it is NP-hard, and the current state-of-the-art static approximation ratio is 1.437 (Cao et al., 2024).

**Dynamic Scenario Demand**: Online stores add new merchandise daily and need to dynamically update item clustering—efficiently updating the clustering after inserting nodes one by one, rather than re-clustering the entire dataset. Cohen-Addad et al. (ICML 2024) first proposed a sublinear update time algorithm under the node-insertion model, but its approximation factor is an extremely large constant (not specified in their paper).

**Goal**: To reduce the approximation factor explicitly to $20+\varepsilon$ while maintaining the $\mathrm{polylog}(n)$ update time, and to validate the practical performance advantages through experiments.

## Method

### Overall Architecture

The algorithm is based on the **5-approximation Pivot variant** by Behnezhad et al. (2023). In the static version:
1. Randomly order all nodes to obtain a permutation $\pi$.
2. Each node $u$ selects its lowest-ranked neighbor in its neighborhood as its pivot: $p(u) = \arg\min_{w \in N[u]} \pi(w)$.
3. If $p(u) = u$, then $u$ becomes a pivot and forms a cluster; if $p(u)$ is a pivot, then $u$ joins its cluster; otherwise, $u$ becomes a singleton.

The issue with direct dynamization is that each insertion requires traversing all neighbors ($\Theta(n)$). This work resolves this via sparse sampling.

### Key Designs

**Design 1: Exact search for low-ranked nodes**

For nodes with $\pi(u) \leq L/d(u)$ ($L = O(\log n)$), the algorithm exactly scans all neighbors. If $u$ is a pivot, it runs an Explore process to update the pivot assignments of its neighbors. Key observation: all pivots in the Reference Clustering satisfy this condition with high probability:

$$\Pr\left[\pi(u) > \frac{L}{d(u)}\right] \leq \left(1 - \frac{L}{d(u)}\right)^{d(u)} \leq \mathrm{poly}(1/n)$$

Therefore, all pivots can be correctly detected.

**Design 2: Random sampling for high-ranked nodes**

For nodes with $\pi(u) > L/d(u)$, the algorithm samples $O(\log n)$ neighbors, inspects their pivots, and chooses the one with the minimum rank as $u$'s pivot. For "good nodes" whose clusters are already mostly correctly clustered, sampling hits the correct pivot with probability $1 - 1/n^{100}$.

**Design 3: Break-cluster denoising**

In the tentative clustering, nodes with in-cluster degrees lower than a threshold $t$ are moved out to become singletons. The algorithm tries $O(\log n)$ candidate thresholds ($1, (1+\epsilon), \ldots, (1+\epsilon)^{\lceil\log n\rceil+1}$), estimates the cost of each threshold via sampling, and selects the optimal one. This is triggered every $\epsilon \cdot |B_v|$ updates.

### Node Taxonomy

The paper defines a fine-grained node classification:
- **Light nodes**: In-cluster degree $\leq |C|/3$ (too many non-edges in the parent cluster).
- **Heavy nodes**: Total degree $\geq \beta|C|$ (too many cross-cluster edges).
- **Poor nodes**: Total degree $\leq \alpha \cdot d(\text{pivot})$ (degree much smaller than its pivot).
- **Bad nodes**: The union of light, heavy, and poor nodes.
- **Lost nodes**: Nodes with at least $\beta$ times as many bad neighbors as good neighbors.

Core Lemma (Lemma 3.2): Turning bad and lost nodes into singletons increases the expected cost by only a factor of $(1+7\varepsilon)$.

### Approximation Guarantee

$$\mathbb{E}[\mathrm{cost}(\text{Sparse-Pivot})] \leq 4(1+O(\varepsilon)) \cdot \mathbb{E}[\mathrm{cost}(\text{Reference Clustering})]$$

Reference Clustering itself is a 5-approximation, so the total approximation ratio is $4 \times 5 \times (1+O(\varepsilon)) = 20 + O(\varepsilon)$.

### Handling Deletions

A lazy deletion strategy is adopted: delete operations are ignored, and a global recomputation is triggered after accumulating $\varepsilon N/6$ updates. Since deletions are random, the clustering of each pair of nodes remains unchanged with probability $\geq 1 - 5\varepsilon/6$.

## Key Experimental Results

### Datasets

| Type | Dataset | Source |
|------|--------|------|
| Sparse Real-world Graphs | musae-facebook, email-Enron, ca-AstroPh, cit-HepTh | SNAP |
| Adjustable Density Graphs | drift (13910 nodes, 129 dimensions, threshold $c \in \{10,15,20,25,30\}$) | UCI ML Repository |

### Main Results: Clustering Objectives on the Drift Dataset (Lower is Better)

| Density | Dynamic Agreement | Reference Clustering | **Sparse-Pivot** |
|---------|-------------------|---------------------|-----------------|
| 235.36 | 0.69 | 0.59 | **0.60** |
| 114.87 | 0.59 | 0.64 | **0.49** |
| 69.74 | 0.50 | 0.50 | **0.41** |
| 52.17 | 0.39 | 0.42 | **0.32** |
| 42.25 | 0.35 | 0.35 | **0.29** |

Costs are normalized by the Singletons baseline. Sparse-Pivot shows significant advantages on low-to-medium density graphs, being 17% lower than DA at the lowest density (42.25).

### Key Findings

- On all datasets, the approximation quality of Sparse-Pivot consistently outperforms both Dynamic Agreement and the Singletons baseline.
- On SNAP graphs, the clustering objective value of Sparse-Pivot remains consistently lower than that of DA over all time steps.
- The actual running time of Sparse-Pivot is also faster than Dynamic Agreement.
- The threshold selection in Break-cluster is the bottleneck that inflates the approximation ratio from 5 to 20; in practice, it can be replaced with a more precise implementation.

## Highlights & Insights

1. **Explicit Approximation Factor**: For the first time, an explicit $(20+\varepsilon)$ approximation ratio is provided for the node-insertion CC problem, which is a qualitative leap compared to the unknown, massive constant in Cohen-Addad et al.
2. **Elegant Node Taxonomy**: The classification system of light/heavy/poor/bad/lost nodes allows the theoretical analysis to precisely track the cost contribution of each node class.
3. **Adaptive Switching between Sampling and Exact Search**: Automatically selecting a strategy based on the relationship between $\pi(u)$ and $L/d(u)$ beautifully balances time and accuracy.
4. **Theoretical-Practical Consistency**: Beyond theoretical improvements, experiments comprehensively outperform the benchmarks, demonstrating the practicality of the design.
5. **Simplicity of Lazy Deletions**: Directly ignoring deletions and periodically recalculating in random deletion scenarios makes the analysis clean and effective.

## Limitations & Future Work

1. **Ample Room for Approximation Ratio Improvement**: $(20+\varepsilon)$ vs. the static optimal of 1.437, with the main bottleneck being the $4\times$ amplification from Break-cluster.
2. **Only Support for Soft Deletions**: Unable to handle deterministic deletions, and deletions must be uniformly random.
3. **Complete Information Assumption**: Requires similarity labels for all node pairs, making it inapplicable to incomplete graphs.
4. **Non-adaptive Arrival Sequences**: Assumes that the node arrival order is independent of the algorithm's decisions.
5. **Limited Experimental Scale**: Scalability has not been validated on million-scale graphs.
6. **Sensitivity to Parameter $\varepsilon$**: The experiments fix $\varepsilon=0.1$ and lack a parameter sensitivity analysis.

## Related Work & Insights

- **Ailon et al. (2008)**: Classic Pivot algorithm with a 3-approximation; the Reference Clustering in this paper is based on its variant.
- **Behnezhad et al. (2023)**: A 5-approximation streaming Pivot variant, which forms the direct foundation of this work.
- **Cohen-Addad et al. (ICML 2024)**: The first sublinear dynamic algorithm (Dynamic Agreement) for node insertions, but with an extremely large approximation constant.
- **Dalirrooyfard et al. (2024)**: Dynamic CC with edge updates, achieving $(3+\varepsilon)$-approximation with constant update time.
- **Cao et al. (2024)**: State-of-the-art static CC with a 1.437-approximation.

Insight: The gap between "theoretical approximation ratio vs. practical quality" in dynamic algorithms deserves attention; the Break-cluster subroutine is a key target for improvement.

## Rating

- Novelty: ⭐⭐⭐⭐ (The first explicit constant approximation for dynamic CC under node insertions)
- Experimental Thoroughness: ⭐⭐⭐ (Small datasets, lacking parameter analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical analysis, well-structured node taxonomy)
- Value: ⭐⭐⭐⭐ (Fills the gap in approximation ratios for dynamic CC)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Estimating Correlation Clustering Cost in Node-Arrival Stream](../../ICML2026/learning_theory/estimating_correlation_clustering_cost_in_node-arrival_stream.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/learning_theory/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](../../ICML2026/learning_theory/simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)
- [\[ICML 2025\] Learning-Augmented Hierarchical Clustering](learning-augmented_hierarchical_clustering.md)

</div>

<!-- RELATED:END -->
