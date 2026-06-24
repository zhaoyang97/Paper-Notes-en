---
title: >-
  [Paper Note] Relative Error Fair Clustering in the Weak-Strong Oracle Model
description: >-
  [ICML 2025][AI Safety][Fair Clustering] Proposes the first fair $k$-median clustering algorithm achieving $(1+\varepsilon)$ approximation under the weak-strong oracle model, requiring only $\text{poly}(k \log n / \varepsilon)$ expensive strong oracle queries, representing a fundamental improvement over previous constant-factor approximations greater than 10.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Fair Clustering"
  - "Weak-Strong Oracle"
  - "Coreset"
  - "Query Complexity"
  - "Algorithmic Fairness"
date: 2026-05-08
content_hash: 0195565e936c2fc0
---

# Relative Error Fair Clustering in the Weak-Strong Oracle Model

**Conference**: ICML 2025  
**arXiv**: [2506.12287](https://arxiv.org/abs/2506.12287)  
**Code**: To be confirmed  
**Area**: AI Safety  
**Keywords**: Fair Clustering, Weak-Strong Oracle, Coreset, Query Complexity, Algorithmic Fairness

## TL;DR

Proposes the first fair $k$-median clustering algorithm achieving $(1+\varepsilon)$ approximation under the weak-strong oracle model, requiring only $\text{poly}(k \log n / \varepsilon)$ expensive strong oracle queries, representing a fundamental improvement over previous constant-factor approximations greater than 10.

## Background & Motivation

### 1. Background

Clustering is a cornerstone of unsupervised learning, widely used for tasks such as data analysis, feature extraction, and data summarization. The classical $k$-median problem requires selecting $k$ centers from $n$ points to minimize the sum of distances from all points to their nearest centers. However, standard clustering can produce systematic biases against protected groups in high-stakes scenarios (such as hiring, lending, and healthcare resource allocation).

### 2. Fair Clustering Constraints

To address biases, Chierichetti et al. proposed $(\alpha, \beta)$-fair clustering: the proportion of group $j$ in each cluster is at least $\alpha_j$ and at most $\beta_j$. This ensures that protected groups receive fair representation in clustering decisions. Each data point can belong to at most $\Lambda$ disjoint groups.

### 3. Real-World Motivation for Weak-Strong Oracles

In modern ML systems, precise distance computations (using large-scale embedding models) are extremely expensive, while lightweight alternatives (simple features, compact networks) are cheap but noisy. Bateni et al. proposed the weak-strong oracle model to formalize this trade-off:
- **Weak Oracle WO**: returns the correct distance with probability $2/3$ and an arbitrary value with probability $1/3$, where the randomness is sampled once and cannot be corrected by majority voting
- **Strong Oracle SO (Point Query)**: deterministically reveals $d(x,y)$ after querying $\text{SO}(x)$ and $\text{SO}(y)$
- **Strong Oracle SO (Distance Query)**: directly queries $\text{SO}(x,y)$ to obtain the precise distance

### 4. Key Challenge & Key Insight

Previously, clustering algorithms under the weak-strong model (based on Meyerson sketch) could only achieve constant-factor $C > 10$ approximations and did not support fairness constraints. Since $k$-clustering itself is NP-hard, pursuing a $(1+\varepsilon)$ approximation in fixed time is unrealistic, but one can pursue efficiency in **query complexity**. The key insight of this paper: construct a query-efficient coreset, solve fair clustering on the small sample, and guarantee a $(1+\varepsilon)$ approximation to the original problem.

## Method

### Overall Architecture

1. Obtain a large amount of cheap (but not fully reliable) distance information through the weak oracle
2. Strategically invoke a small number of strong oracle queries to calibrate key distances
3. Construct a weighted coreset of size $\text{poly}(k \log n / \varepsilon)$
4. Solve fair $k$-median on the coreset to obtain a $(1+\varepsilon)$-approximate solution to the original problem
5. Further extend the method to general $(k,z)$-clustering ($z = O(1)$)

### Key Designs

#### 1. Coreset Construction

- **Function**: Compress the original $n$ points into a weighted small subset $\mathcal{S}$, satisfying $\text{fair}_k(\mathcal{S}) \leq (1+\varepsilon) \cdot \text{fair}_k(\mathcal{X})$
- **Mechanism**: Build rough distance estimates based on extensive weak oracle sampling, and then invoke the strong oracle at key points to correct bias, ensuring that fairness constraints are also maintained on the coreset
- **Design Motivation**: Directly performing fair clustering on the full dataset requires $O(n^2)$ or more strong queries; the coreset reduces the scale to $\text{poly}(k \log n / \varepsilon)$, and the queries for subsequent processing are correspondingly reduced

#### 2. Maintaining Fairness Constraints

- **Function**: Ensure that the composition of protected groups in the coreset is proportional to the original set
- **Mechanism**: During the sampling and weighting process, perform stratified sampling by group and adjust weights to ensure that fairness conditions are maintained in a weighted sense
- **Difference from Prior Work**: The Meyerson sketch of Bateni et al. completely ignores fairness; this paper "builds" the fairness constraints directly into the coreset construction

#### 3. Leap from Constant Factor to Relative Error

- **Function**: Improve the approximation quality from $C > 10$ to $1 + \varepsilon$
- **Mechanism**: The constant factor of the Meyerson sketch comes from the inherent relaxation of its greedy framework, which cannot be broken through parameter tuning. This paper adopts a brand-new coreset construction paradigm, controlling the accumulation of biases at each stage through a more refined sampling probability design and error propagation analysis
- **Key Theorem (Theorem 1)**: There exists an algorithm that constructs a $(1+\varepsilon)$-coreset of size $\text{poly}(k \log n / \varepsilon)$, requiring the same $\text{poly}(k \log n / \varepsilon)$ number of strong oracle queries

## Key Experimental Results

### Main Results Comparison

| Metric | Ours | Prev. SOTA (Bateni et al., 2024) | Gain |
|------|---------|---------------------|------|
| Approximation Ratio | $(1+\varepsilon)$ | $> 10$ (constant) | From coarse approximation to relative error |
| Number of Strong Point Queries | $\text{poly}(k \log n / \varepsilon)$ | $\tilde{O}(k)$ | Same order of magnitude but qualitative leap in accuracy |
| Number of Strong Distance Queries | $\text{poly}(k \log n / \varepsilon)$ | $\tilde{O}(k^2)$ | Comparable |
| Fairness Constraint | Fully supports $(\alpha,\beta)$ | Not supported | First implementation |
| Scope of Problem | $(k,z)$-clustering, $z=O(1)$ | $(k,z)$-clustering | Simultaneously improves accuracy |

### Ablation Study: Contributions of Each Mechanism

| Configuration | Approximation Quality | Strong Query Cost | Description |
|------|---------|----------|------|
| Weak oracle only, no strong correction | Uncontrollable | Lowest | Noise makes clustering quality unstable |
| No coreset, full strong queries | Theoretically optimal | $O(n)$ or $O(n^2)$ | Query cost is unacceptable |
| Old framework of Meyerson sketch | $> 10$ constant | $\tilde{O}(k)$ | Cannot break through the constant factor |
| Full framework of Ours | $(1+\varepsilon)$ | $\text{poly}(k \log n / \varepsilon)$ | Best balance between accuracy and query cost |

Note: This paper is a theoretical contribution. The cache is truncated after Theorem 1, and complete experimental values could not be retrieved. The above table is structured based on the conclusions stated in the paper.

### Key Findings

- Coreset is a key bridge connecting high accuracy with low query cost: first cover breadth with the weak oracle, then ensure depth with the strong oracle.
- Moving from over $10\times$ to $(1+\varepsilon)$ is not an incremental improvement but a fundamental change of framework.
- Fairness constraints are "naturally preserved" during the coreset construction process, requiring no post-hoc corrections.

## Highlights & Insights

- **Significant theoretical breakthrough**: Advances fair clustering in weak-strong oracle scenarios from coarse approximations to fine relative error guarantees, crossing methodological tiers.
- **Elegant unification of a trilemma**: Simultaneously solves the coupled challenges of "fairness + query constraints + high accuracy".
- **Methodological generalizability**: The core idea (broad coverage with weak signals + precise correction with strong signals) can be transferred to other ML scenarios requiring heterogeneous information sources.
- **Reusable techniques**: The practice of using stratified sampling to maintain fairness constraints can be directly applied to other fair optimization problems.

## Limitations & Future Work

- The cache is truncated at Theorem 1, and the complete chain of proofs, lower bound results, and experimental evaluations could not be reconstructed.
- The specific polynomial degree of $\text{poly}(k \log n / \varepsilon)$ is crucial in engineering practice and needs to be verified in the full version.
- Robustness to extreme long-tail group distributions (where some groups account for only one-thousandth) is not discussed.
- Heterogeneous cost modeling of strong queries (different query costs for different point pairs) and online update mechanisms can be further explored.
- Coreset designs combining privacy constraints, such as differential privacy, with fairness constraints can be investigated.

## Related Work & Insights

- **vs Bateni et al. (2024)**: Same weak-strong oracle model, but only achieved a constant factor and lacked fairness; this paper advances in both directions simultaneously.
- **vs Chierichetti et al. (2017)**: The origin of the formal definition of fair clustering, but assumes precise distances are available with no query limitations.
- **vs Classical coreset theory (Feldman & Langberg 2011)**: Generalizes coresets from standard clustering to a joint setting of query constraints + fairness constraints.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First relative error guarantee for fair clustering under weak-strong oracles.
- Experimental Thoroughness: ⭐⭐⭐ Theoretical paper; cache does not contain complete experimental values.
- Writing Quality: ⭐⭐⭐⭐⭐ Precise definitions, complete motivational chain, and clear conclusions.
- Value: ⭐⭐⭐⭐⭐ Long-term theoretical value for the directions of fair ML and cost-constrained learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Fair Model-Based Clustering](../../AAAI2026/ai_safety/fair_model-based_clustering.md)
- [\[ICML 2025\] Accelerating Spectral Clustering under Fairness Constraints](accelerating_spectral_clustering_under_fairness_constraints.md)
- [\[ICML 2025\] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification](breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [\[ICML 2025\] Improving the Variance of Differentially Private Randomized Experiments through Clustering](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)

</div>

<!-- RELATED:END -->
