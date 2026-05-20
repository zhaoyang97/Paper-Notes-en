---
title: >-
  [Paper Note] Generalizing Fair Clustering to Multiple Groups: Algorithms and Applications
description: >-
  [AAAI 2026][AI Safety][Fair clustering] This paper generalizes the Closest Fair Clustering problem from two groups to arbitrarily many groups, proves NP-hardness for the equal-proportion case with three or more groups…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Fair clustering"
  - "multi-group fairness"
  - "approximation algorithms"
  - "correlation clustering"
  - "consensus clustering"
  - "NP-hard"
date: 2026-05-08
content_hash: fe411b84068f9660
---

# Generalizing Fair Clustering to Multiple Groups: Algorithms and Applications

**Conference**: AAAI 2026
**arXiv**: [2511.11539](https://arxiv.org/abs/2511.11539)  
**Code**: To be confirmed  
**Area**: AI Safety / Fairness
**Keywords**: Fair clustering, multi-group fairness, approximation algorithms, correlation clustering, consensus clustering, NP-hard

## TL;DR

This paper generalizes the Closest Fair Clustering problem from two groups to arbitrarily many groups, proves NP-hardness for the equal-proportion case with three or more groups, proposes near-linear-time approximation algorithms (equal-proportion $O(|\chi|^{1.6}\log^{2.81}|\chi|)$, arbitrary-proportion $O(|\chi|^{3.81})$), and extends the results to fair correlation clustering and fair consensus clustering.

## Background & Motivation

- **Background**: Clustering is a fundamental unsupervised learning task in machine learning, yet traditional algorithms tend to produce unfair outcomes with respect to protected attributes (gender, race, age, etc.). Since Chierichetti et al. (2017) introduced fair clustering, the community has advanced fairness constraints across numerous variants including k-center/median/means and correlation clustering.
- **Limitations of Prior Work**: Chakraborty et al. [COLT'25] first studied the "Closest Fair Clustering" problem—obtaining a fair clustering from an existing one with minimal modifications—but addressed only the **two-group** case, whereas real-world data often involves multiple protected attributes corresponding to multiple groups.
- **Key Challenge**: An exact near-linear-time algorithm exists for the two-group equal-proportion case; this paper proves that the multi-group ($\geq 3$) setting is NP-hard even under equal proportions, revealing a **computational complexity cliff** between two groups and multiple groups.
- **Key Insight**: The paper designs approximation algorithms via a hierarchical block merging strategy, answering open problems posed by Chakraborty et al.

## Method

### Overall Architecture

A hierarchical recursive strategy decomposes multi-group fair clustering into multiple rounds of pairwise merging. The framework comprises three algorithms: (1) **fairpower-of-two**—handles the equal-proportion case when the number of colors is a power of two; (2) **make-pdc-fair**—converts a p-divisible clustering into a fair clustering under arbitrary proportions; (3) **create-pdc**—transforms an arbitrary clustering into a p-divisible clustering. Their composition achieves approximate conversion from an arbitrary clustering to a multi-group fair clustering.

### Key Designs

1. **Algorithm fairpower-of-two (equal proportion + color count is a power of two)**

    - Function: Transforms an arbitrary clustering $\mathcal{D}$ into a fair clustering in which each cluster contains an equal number of points of each color.
    - Mechanism: Performs $\log|\chi|$ iterations; in each round, adjacent color blocks are merged pairwise. In round $i$, the color set is partitioned into $|\chi|/2^i$ blocks of size $2^i$, maintaining the invariant that each cluster contains an equal number of points from each color within a block.
    - Each round: For each pair of adjacent blocks, the "surplus" between the two blocks within each cluster is computed, removed from the clusters, collected into sets $S_j, S_{j+1}$, and re-paired via the **multi-GM** subroutine to form new fair subsets.
    - Approximation ratio: Each round introduces a factor of 2; after $\log|\chi|$ rounds, recursive application of the triangle inequality yields $3^{\log|\chi|} = O(|\chi|^{1.6})$.

2. **Algorithm make-pdc-fair (arbitrary proportion, p-divisible → fair clustering)**

    - Function: Given proportions $p_1:p_2:\cdots:p_r$ and a p-divisible clustering, outputs a fair clustering satisfying the global proportions.
    - Mechanism: Applies a hierarchical strategy similar to fairpower-of-two but handles unequal proportions. Performs $\lceil\log_2 r\rceil$ rounds, merging adjacent color blocks each round and using a "scaling factor" to balance mismatches between blocks.
    - Balancing rule: For sub-blocks $A$ and $B$, scaling factors $x$ and $y$ are computed. If $x > y$, points of color $B$ are merged into the cluster; if $x < y$, points of color $B$ are trimmed from the cluster.
    - Approximation ratio: Each round introduces a factor of 6, yielding a total of $7^{\log r} = O(r^{2.81})$.

3. **Algorithm create-pdc (arbitrary clustering → p-divisible clustering)**

    - Function: For each color $c_j$, makes the number of points of that color in each cluster a multiple of $p_j$.
    - Mechanism: Clusters are classified into CUT sets (surplus $\leq p_j/2$, trimming the surplus) and MERGE sets (surplus $> p_j/2$, receiving points from other clusters to fill deficits). Auxiliary clusters are created to accommodate excess surpluses.
    - Approximation ratio: $O(|\chi|)$-close p-divisible.

### Loss & Training

This paper addresses a combinatorial optimization problem. The objective is to minimize $\text{dist}(\mathcal{D}, \mathcal{F})$—the distance (number of discordant point pairs) between the original clustering and the output fair clustering. The theoretical analysis chains approximation ratios across steps via the triangle inequality:

$$\text{dist}(\mathcal{D}, \mathcal{F}) \leq O(|\chi|^{3.81}) \cdot \text{dist}(\mathcal{D}, \mathcal{F}^*)$$

## Key Experimental Results

This paper is a **theoretical work** containing no empirical experiments; complete approximation ratio and computational complexity analyses are provided.

### Main Results

| Problem Setting | Approximation Ratio | Time Complexity | Notes |
|:--|:--|:--|:--|
| Equal proportion + $|\chi|$ is a power of 2 | $O(|\chi|^{1.6})$ | $O(|V|\log|V|)$ | fairpower-of-two |
| Equal proportion + arbitrary $|\chi|$ | $O(|\chi|^{1.6}\log^{2.81}|\chi|)$ | $O(|V|\log|V|)$ | fair-equi |
| Arbitrary proportion | $O(|\chi|^{3.81})$ | $O(|V|\log|V|)$ | fair-general |
| Fair correlation clustering (equal proportion) | $O(|\chi|^{1.6}\log^{2.81}|\chi|)$ | — | Combined with $O(1)$ correlation clustering |
| Fair correlation clustering (arbitrary proportion) | $O(|\chi|^{3.81})$ | — | Eliminates dependence on group ratio $q$ |
| Fair consensus clustering | Same as above | $O(m^2|V|^2)$ | First multi-group result |

### Ablation Study

| Problem | Hardness Result | Reduction Source |
|:--|:--|:--|
| $k$-Closest EquiFair ($k \geq 3$) | NP-hard | 3-Partition (strongly NP-complete) |
| Arbitrary-proportion Closest Fair Clustering | NP-hard | Same (extended reduction) |

### Key Findings

- **Computational cliff between two and multiple groups**: An exact near-linear-time algorithm exists for two-group equal proportions, yet three-group equal proportions is NP-hard; this is the first work to rigorously prove this gap.
- **Elimination of dependence on group proportions**: The prior best approximation for fair correlation clustering was $O(q^2|\chi|^2)$ (where $q = \max(p_j)/\min(p_j)$ can be $\text{poly}(|V|)$); this paper achieves $O(|\chi|^{3.81})$, which is independent of $q$.
- **First multi-group fair consensus clustering algorithm**: Prior work only provided constant-factor approximations for the two-group case.

## Highlights & Insights

- ⭐ **Rigorous theoretical contributions**: Proves NP-hardness of multi-group fair clustering, revealing a fundamental computational complexity jump from two groups to multiple groups.
- ⭐ **Generality of the hierarchical merging strategy**: The hierarchical recursive framework of fairpower-of-two + make-pdc-fair naturally extends to correlation clustering and consensus clustering.
- ⭐ **Near-linear-time algorithms**: The $O(|V|\log|V|)$ time complexity endows the algorithms with good practical scalability.
- ⭐ Answers two open problems posed in COLT'25.

## Limitations & Future Work

- **Approximation ratio grows with the number of colors**: The approximation quality of $O(|\chi|^{3.81})$ may deteriorate when the number of groups is large; improving the approximation factor is an important direction for future work.
- **Purely theoretical work**: Empirical experiments on real-world datasets are absent, leaving the practical performance and runtime efficiency unverified.
- **Strict fairness definition**: The formulation requires the group proportions within each cluster to exactly match the global proportions, without considering relaxed fairness definitions that tolerate bounded deviations.
- **Single distance metric**: Only the distance based on the number of discordant point pairs is considered; fair clustering under other metric spaces is not addressed.

## Related Work & Insights

- **Foundational fair clustering work**: Chierichetti et al. (2017) first studied fair clustering for two groups; Rosner & Schmidt (2018) extended it to multi-group proportion constraints.
- **Closest Fair Clustering**: Chakraborty et al. [COLT'25] introduced the Closest Fair Clustering problem and provided exact and constant-factor approximation algorithms for the two-group case.
- **Fair correlation clustering**: Ahmadian et al. [AISTATS'20] and Ahmadi et al. (2020) achieved an $O(q^2|\chi|^2)$ approximation; this paper improves it to an approximation independent of $q$.
- **Correlation clustering**: Bansal et al. (2004) provided a comprehensive study; the current best approximation is 1.438 (Cao et al. 2024); the problem is APX-hard.
- **Fair consensus clustering**: Chakraborty et al. [COLT'25] introduced and resolved the two-group case.

## Rating

⭐⭐⭐⭐ — Solid theoretical contributions that resolve important open problems, but empirical validation is absent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Fair Model-Based Clustering](fair_model-based_clustering.md)
- [\[AAAI 2026\] Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection](towards_multiple_missing_values-resistant_unsupervised_graph_anomaly_detection.md)
- [\[NeurIPS 2025\] Unifying Proportional Fairness in Centroid and Non-Centroid Clustering](../../NeurIPS2025/ai_safety/unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)
- [\[CVPR 2026\] ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering](../../CVPR2026/ai_safety/clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with.md)
- [\[NeurIPS 2025\] Fair Minimum Labeling: Efficient Temporal Network Activations for Reachability and Equity](../../NeurIPS2025/ai_safety/fair_minimum_labeling_efficient_temporal_network_activations_for_reachability_an.md)

</div>

<!-- RELATED:END -->
