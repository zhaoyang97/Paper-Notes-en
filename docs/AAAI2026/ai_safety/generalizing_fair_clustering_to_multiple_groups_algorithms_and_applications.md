---
title: >-
  [Paper Note] Generalizing Fair Clustering to Multiple Groups: Algorithms and Applications
description: >-
  [AAAI 2026 Oral][AI Safety][Fair Clustering] Generalizes the Closest Fair Clustering problem from only two groups to arbitrarily many groups, proves that the equal-proportion case with three or more groups is already NP-hard, proposes near-linear time approximation algorithms ($O(|\chi|^{1.6}\log^{2.81}|\chi|)$ for equal proportions, and $O(|\chi|^{3.81})$ for arbitrary proportions), and extends the results to fair correlation clustering and fair consensus clustering problems…
tags:
  - "AAAI 2026 Oral"
  - "AI Safety"
  - "Fair Clustering"
  - "Multi-Group Fairness"
  - "Approximation Algorithms"
  - "Correlation Clustering"
  - "Consensus Clustering"
  - "NP-hard"
date: 2026-05-08
content_hash: 41c49308b63cfda8
---

# Generalizing Fair Clustering to Multiple Groups: Algorithms and Applications

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.11539](https://arxiv.org/abs/2511.11539)  
**Code**: To be confirmed  
**Area**: AI Safety/Fairness  
**Keywords**: Fair Clustering, Multi-Group Fairness, Approximation Algorithms, Correlation Clustering, Consensus Clustering, NP-hard

## TL;DR

Generalizes the Closest Fair Clustering problem from only two groups to arbitrarily many groups, proves that the equal-proportion case with three or more groups is already NP-hard, proposes near-linear time approximation algorithms ($O(|\chi|^{1.6}\log^{2.81}|\chi|)$ for equal proportions, and $O(|\chi|^{3.81})$ for arbitrary proportions), and extends the results to fair correlation clustering and fair consensus clustering problems.

## Background & Motivation

- **Background**: Clustering is a fundamental unsupervised learning task in machine learning, but traditional algorithms easily map to unfair outcomes when involving protected attributes (gender, race, age, etc.). Since Chierichetti et al. (2017) proposed fair clustering, the community has continuously advanced fairness constraints on various variants such as k-center/median/means and correlation clustering.
- **Limitations of Prior Work**: Chakraborty et al. [COLT'25] first studied the "Closest Fair Clustering" problem—given an existing clustering, obtain a fair clustering with minimal modifications. However, this prior work only handles the **two-group** case, whereas real-world data often has multiple protected attributes corresponding to multiple groups (such as age, race, and gender).
- **Key Challenge**: The equal-proportion case for two groups has exact algorithms (near-linear time). This paper proves that multiple groups ($\geq 3$) are NP-hard even under equal proportions, revealing a **computational complexity cliff** from two groups to multiple groups.
- **Key Insight**: Designs approximation algorithms via a hierarchical block merging strategy, answering the open questions posed by Chakraborty et al.

## Method

### Overall Architecture

Adopts a hierarchical recursive strategy to decompose multi-group fair clustering into multiple rounds of pairwise merging operations. The core consists of three algorithms: (1) **fairpower-of-two**—handles the equal-proportion case where the number of colors is a power of two; (2) **make-pdc-fair**—handles converting p-divisible clustering with arbitrary proportions into fair clustering; (3) **create-pdc**—converts arbitrary clustering into p-divisible clustering. The combination of these three achieves an approximate transformation from arbitrary clustering to multi-group fair clustering.

### Key Designs

1. **Algorithm fairpower-of-two (Equal Proportions + Power-of-Two Number of Colors)**

    - **Function**: Converts an arbitrary clustering $\mathcal{D}$ into a fair clustering where each cluster has an equal number of points of each color.
    - **Mechanism**: Executes $\log|\chi|$ iterations, merging adjacent color blocks pairwise in each round. In the $i$-th round, the color set is divided into $|\chi|/2^i$ blocks of size $2^i$, maintaining the invariant that the number of colors of the same block within each cluster is equal.
    - **Operations per round**: For each pair of adjacent blocks, computes the "surplus" between the two blocks in each cluster, removes the surplus from the cluster, collects them into sets $S_j, S_{j+1}$, and calls the **multi-GM** subroutine to re-pair the surplus to form new fair subsets.
    - **Approximation Ratio**: Each round introduces a factor of 2, recursively yielding $3^{\log|\chi|} = O(|\chi|^{1.6})$ after $\log|\chi|$ rounds using the triangle inequality.

2. **Algorithm make-pdc-fair (Arbitrary Proportions p-divisible → Fair Clustering)**

    - **Function**: Given proportions $p_1:p_2:\cdots:p_r$ and a p-divisible clustering, outputs a fair clustering that satisfies the global proportions.
    - **Mechanism**: Uses a hierarchical strategy similar to fairpower-of-two, but must handle unequal proportions. Executes $\lceil\log_2 r\rceil$ rounds to merge adjacent color blocks in each round, balancing the mismatch between two blocks through a "scaling factor".
    - **Balancing rules**: For sub-blocks $A$ and $B$, computes scaling factors $x$ and $y$. If $x > y$, merges points of type $B$ color into the cluster; if $x < y$, prunes points of type $B$ color from the cluster.
    - **Approximation Ratio**: Each round introduces a factor of 6, totaling $7^{\log r} = O(r^{2.81})$.

3. **Algorithm create-pdc (Arbitrary Clustering → p-divisible Clustering)**

    - **Function**: For each color $c_j$, makes the number of points of this color in each cluster a multiple of $p_j$.
    - **Mechanism**: Splits clusters into a CUT set (surplus $\leq p_j/2$, pruning surplus) and a MERGE set (surplus $> p_j/2$, receiving points from other clusters to fill the deficit). Creates auxiliary clusters to accommodate the excess surplus.
    - **Approximation Ratio**: $O(|\chi|)$-close p-divisible.

### Loss & Training

This paper deals with a combinatorial optimization problem, where the optimization objective is to minimize $\text{dist}(\mathcal{D}, \mathcal{F})$—the distance (number of inconsistent pairs) between the original clustering and the output fair clustering. The theoretical analysis combines the approximation ratios of each step through a chain of triangle inequalities:

$$\text{dist}(\mathcal{D}, \mathcal{F}) \leq O(|\chi|^{3.81}) \cdot \text{dist}(\mathcal{D}, \mathcal{F}^*)$$

## Key Experimental Results

This paper is a **theoretical work** and does not contain empirical experiments, but provides a complete analysis of approximation ratios and computational complexity.

### Main Results

| Problem Setting | Approximation Ratio | Time Complexity | Remarks |
|:--|:--|:--|:--|
| Equal proportions + $|\chi|$ is a power of 2 | $O(|\chi|^{1.6})$ | $O(|V|\log|V|)$ | fairpower-of-two |
| Equal proportions + arbitrary $|\chi|$ | $O(|\chi|^{1.6}\log^{2.81}|\chi|)$ | $O(|V|\log|V|)$ | fair-equi |
| Arbitrary proportions | $O(|\chi|^{3.81})$ | $O(|V|\log|V|)$ | fair-general |
| Fair correlation clustering (equal proportions) | $O(|\chi|^{1.6}\log^{2.81}|\chi|)$ | — | Combined with $O(1)$ correlation clustering |
| Fair correlation clustering (arbitrary proportions) | $O(|\chi|^{3.81})$ | — | Eliminating dependence on group proportion $q$ |
| Fair consensus clustering | Same as above | $O(m^2|V|^2)$ | First multi-group result |

### NP-hard Results

| Problem | Hardness Conclusion | Reduction Source |
|:--|:--|:--|
| $k$-Closest EquiFair ($k \geq 3$) | NP-hard | 3-Partition (strongly NP-complete) |
| Arbitrary proportion Closest Fair Clustering | NP-hard | Same as above (extended reduction) |

### Key Findings

- **Computational cliff of two groups vs. multiple groups**: The equal-proportion case for two groups has an exact near-linear time algorithm, but the equal-proportion case for three groups is already NP-hard. This is the first work to rigorously prove this gap.
- **Eliminating dependence on group proportions**: The prior state-of-the-art approximation for fair correlation clustering was $O(q^2|\chi|^2)$ (where $q = \max(p_j)/\min(p_j)$ can reach $\text{poly}(|V|)$), whereas the $O(|\chi|^{3.81})$ in this paper is independent of the proportions.
- **First multi-group fair consensus clustering algorithm**: Previously, only constant-factor approximation results for two groups existed.

## Highlights & Insights

- ⭐ **Rigorous theoretical contributions**: Proves the NP-hardness of multi-group fair clustering, revealing the fundamental computational complexity jump from two groups to multiple groups.
- ⭐ **Generality of the hierarchical merging strategy**: The hierarchical recursive framework of fairpower-of-two + make-pdc-fair can naturally generalize to correlation clustering and consensus clustering.
- ⭐ **Near-linear time complexity**: The time complexity of $O(|V|\log|V|)$ endows the algorithm with good practical scalability.
- ⭐ Answers two open questions posed by COLT'25.

## Limitations & Future Work

- **Approximation ratio grows with the number of colors**: $O(|\chi|^{3.81})$ may yield suboptimal approximation quality when the number of groups is large; improving the approximation factor is an important future direction.
- **Purely theoretical work**: Lacks empirical experiments to validate the actual performance and running efficiency of the algorithms on real-world datasets.
- **Very strict definition of fairness**: Requires the group proportions in each cluster to match the global proportions exactly, without considering relaxed fairness definitions that allow for a certain tolerance.
- **Single distance metric**: Only considers distances based on the number of inconsistent pairs, without involving fair clustering under other metric spaces.

## Related Work

- **Pioneering work in fair clustering**: Chierichetti et al. (2017) first studied fair clustering, handling two groups; Rosner & Schmidt (2018) generalized it to multi-group proportion constraints.
- **Closest Fair Clustering**: Chakraborty et al. [COLT'25] proposed the closest fair clustering problem, presenting exact and constant-factor approximation algorithms for two groups.
- **Fair correlation clustering**: Ahmadian et al. [AISTATS'20] and Ahmadi et al. (2020) provided $O(q^2|\chi|^2)$ approximations; this paper improves it to an approximation independent of $q$.
- **Correlation clustering**: Comprehensively studied by Bansal et al. (2004); the state-of-the-art 1.438-approximation is by Cao et al. (2024); it is APX-hard.
- **Fair consensus clustering**: Proposed and solved by Chakraborty et al. [COLT'25] for the two-group case.

## Rating

⭐⭐⭐⭐ — Solid theoretical contributions, solving important open problems, but lacks experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Fair Model-Based Clustering](fair_model-based_clustering.md)
- [\[ICLR 2026\] Fair Conformal Classification via Learning Representation-Based Groups](../../ICLR2026/ai_safety/fair_conformal_classification_via_learning_representation-based_groups.md)
- [\[ICLR 2026\] Private Rate-Constrained Optimization with Applications to Fair Learning](../../ICLR2026/ai_safety/private_rate-constrained_optimization_with_applications_to_fair_learning.md)
- [\[ICML 2025\] Relative Error Fair Clustering in the Weak-Strong Oracle Model](../../ICML2025/ai_safety/relative_error_fair_clustering_in_the_weak-strong_oracle_model.md)
- [\[NeurIPS 2025\] Unifying Proportional Fairness in Centroid and Non-Centroid Clustering](../../NeurIPS2025/ai_safety/unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)

</div>

<!-- RELATED:END -->
