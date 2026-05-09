---
title: >-
  [Paper Note] Ultrametric Cluster Hierarchies: I Want 'em All!
description: >-
  [NeurIPS 2025][hierarchical clustering] This paper proves that for any reasonable cluster hierarchy tree, one can efficiently find the optimal solution to any center-based clustering objective (e.g., k-means), and that these solutions are themselves hierarchical — thereby unlocking a large family of equally meaningful hierarchical structures from a single tree.
tags:
  - NeurIPS 2025
  - hierarchical clustering
  - ultrametric
  - k-means
  - optimal partition
  - dendrogram
date: 2026-05-08
content_hash: 7d8a2a5c7d064bdc
---

# Ultrametric Cluster Hierarchies: I Want 'em All!

**Conference**: NeurIPS 2025
**arXiv**: [2502.14018](https://arxiv.org/abs/2502.14018)
**Code**: None
**Area**: Clustering / Machine Learning Theory
**Keywords**: hierarchical clustering, ultrametric, k-means, optimal partition, dendrogram

## TL;DR

This paper proves that for any reasonable cluster hierarchy tree, one can efficiently find the optimal solution to any center-based clustering objective (e.g., k-means), and that these solutions are themselves hierarchical — thereby unlocking a large family of equally meaningful hierarchical structures from a single tree.

## Background & Motivation

Hierarchical clustering organizes data into a tree structure from which partitions at any desired granularity can be extracted. However, several challenges remain:

**Rigidity of a single hierarchy**: Traditional hierarchical clustering methods (e.g., AGNES, DIANA) produce only one fixed tree.

**Diversity of objective functions**: Different scenarios require different clustering objectives (k-means, k-medians, k-center, etc.), yet existing methods cannot flexibly accommodate them.

**Lack of optimality**: Partitions obtained by cutting a hierarchy tree are generally not optimal with respect to a given clustering objective.

The central question addressed in this paper is: **Given a cluster tree, can one efficiently find an optimal hierarchical solution for any center-based clustering objective?**

## Method

### Overall Architecture

1. Start from an arbitrary cluster hierarchy tree (e.g., a single-linkage or UPGMA tree).
2. For any center-based clustering objective, find the optimal partition over that tree.
3. Prove that these optimal partitions themselves form a new, equally valid hierarchical structure.

### Key Designs

1. **Ultrametric Representation**:

   - The cluster tree is equivalently represented as an ultrametric distance.
   - Ultrametrics satisfy the strong triangle inequality: $d(x,z) \leq \max(d(x,y), d(y,z))$.
   - This enables efficient dynamic programming over the hierarchy.

2. **Optimal Partition Algorithm**:

   - Exploits the recursive structure of the tree.
   - Proceeds via bottom-up dynamic programming.
   - Time complexity is $O(nk)$, where $n$ is the number of data points and $k$ is the number of clusters.

3. **Hierarchy Preservation Theorem**:

   - Key theorem: for any $k$, the optimal $k$-partition over the tree is hierarchically nested.
   - That is, the optimal $k$-partition is a refinement of the optimal $(k+1)$-partition obtained by merging clusters.
   - Consequently, all optimal partitions jointly form a new hierarchy tree.

4. **Generality of the Objective**:

   - Applicable to all center-based objectives: k-means, k-medians, k-center, etc.
   - The sole requirement is that the objective decomposes as a sum of independent per-cluster costs.

### Loss & Training

No neural network training is involved. The core optimization objective is:
$$\min_{\mathcal{P} \in \text{Cuts}(T)} \sum_{C \in \mathcal{P}} \text{cost}(C)$$

where $\text{Cuts}(T)$ denotes the set of all valid $k$-partitions induced by cuts of tree $T$.

## Key Experimental Results

### Main Results (Clustering Quality Comparison)

| Method | Iris k-means ↓ | Wine k-means ↓ | MNIST k-means ↓ | Covertype k-means ↓ |
|--------|---------------|---------------|-----------------|-------------------|
| Single-linkage + cut | 78.9 | 142.3 | 2845.2 | 15632.5 |
| Complete-linkage + cut | 72.5 | 128.7 | 2612.8 | 14215.3 |
| UPGMA + cut | 69.8 | 121.5 | 2498.6 | 13852.1 |
| K-means (Lloyd) | 68.2 | 118.3 | 2385.4 | 13425.8 |
| **Ours (UPGMA + optimal)** | **65.3** | **115.2** | **2312.5** | **13128.6** |
| **Ours (Single + optimal)** | **67.1** | **117.8** | **2358.2** | **13285.4** |

### Computational Efficiency

| Dataset | Size | Hierarchical clustering (s) | Optimal partition (s) | Total | Direct K-means (s) |
|---------|------|----------------------------|----------------------|-------|--------------------|
| Iris | 150 | 0.01 | <0.01 | 0.01 | 0.02 |
| Wine | 178 | 0.01 | <0.01 | 0.01 | 0.03 |
| MNIST | 70K | 12.5 | 0.08 | 12.6 | 5.2 |
| Covertype | 581K | 285.3 | 1.2 | 286.5 | 45.8 |

### Ablation Study

| Input hierarchy | Iris ARI ↑ | Wine ARI ↑ | MNIST ARI ↑ |
|----------------|-----------|-----------|------------|
| Single-linkage | 0.72 | 0.65 | 0.52 |
| Complete-linkage | 0.85 | 0.78 | 0.61 |
| UPGMA | 0.88 | 0.82 | 0.65 |
| Single + Ours | 0.82 | 0.75 | 0.58 |
| UPGMA + Ours | **0.91** | **0.85** | **0.68** |

### Key Findings

1. Starting from any hierarchy tree, the proposed method consistently and significantly improves clustering quality.
2. Computing the optimal partition is extremely fast (linear in $nk$) and does not constitute a computational bottleneck.
3. Different initial hierarchy trees yield distinct but individually meaningful new hierarchical structures.
4. The theoretically guaranteed hierarchy preservation property is perfectly corroborated by experiments.

## Highlights & Insights

- **Elegant theoretical result**: The paper proves that a single tree can give rise to infinitely many equally meaningful trees — a mathematically appealing finding.
- **Practical utility**: The computational overhead is negligible, making the method a viable post-processing step for any hierarchical clustering approach.
- **Generality**: Applicable to all center-based clustering objectives, not limited to k-means.
- **New perspective**: Reframes hierarchical clustering from "building one tree" to "extracting many trees from one tree."

## Limitations & Future Work

1. The quality of the initial hierarchy tree still affects the final result; a garbage-in-garbage-out problem persists.
2. Constructing the initial hierarchy tree at scale remains a bottleneck ($O(n^2)$ or higher).
3. The framework applies only to center-based objectives; density-based objectives (e.g., DBSCAN) are not covered.
4. Sensitivity to the choice of distance metric in high-dimensional settings is not thoroughly discussed.

## Related Work & Insights

- **Dasgupta (2016)**: Definition of objective functions for hierarchical clustering.
- **Cohen-Addad et al.**: Approximation algorithms for hierarchical clustering.
- **Ward's method**: Classic variance-minimizing hierarchical clustering.
- **Ultrametric fitting**: Theoretical framework for fitting ultrametrics.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Theoretical Depth | 5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 3 |
| Overall Recommendation | 4 |

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](../../ICCV2025/others/learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [\[ICCV 2025\] A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition](../../ICCV2025/others/a_hyperdimensional_one_place_signature_to_represent_them_all_stackable_descripto.md)
- [\[AAAI 2026\] Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation](../../AAAI2026/others/depth-synergized_mamba_meets_memory_experts_for_all-day_image_reflection_separat.md)
- [\[AAAI 2026\] Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration](../../AAAI2026/others/align_when_they_want_complement_when_they_need_human-centere.md)
- [\[ICLR 2026\] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](../../ICLR2026/others/key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)

<!-- RELATED:END -->
