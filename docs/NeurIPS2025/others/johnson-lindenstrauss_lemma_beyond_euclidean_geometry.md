---
title: >-
  [Paper Note] Johnson-Lindenstrauss Lemma Beyond Euclidean Geometry
description: >-
  [NeurIPS 2025][Dimensionality Reduction] This paper extends the Johnson-Lindenstrauss (JL) lemma from Euclidean space to general symmetric hollow dissimilarity matrices, proposing two complementary approaches — pseudo-Euclidean JL and generalized power distance JL — where the approximation error scales proportionally with the degree of deviation from Euclidean geometry.
tags:
  - NeurIPS 2025
  - Dimensionality Reduction
  - Johnson-Lindenstrauss Lemma
  - Non-Euclidean Geometry
  - Pseudo-Euclidean Space
  - Generalized Power Distance
date: 2026-05-08
content_hash: 36cbd1f736971781
---

# Johnson-Lindenstrauss Lemma Beyond Euclidean Geometry

**Conference**: NeurIPS 2025
**arXiv**: [2510.22401](https://arxiv.org/abs/2510.22401)
**Code**: [Anonymous GitHub](https://anonymous.4open.science/r/Non-Euclidean-Johnson-Lindenstrauss-1673)
**Area**: Others
**Keywords**: Dimensionality Reduction, Johnson-Lindenstrauss Lemma, Non-Euclidean Geometry, Pseudo-Euclidean Space, Generalized Power Distance

## TL;DR

This paper extends the Johnson-Lindenstrauss (JL) lemma from Euclidean space to general symmetric hollow dissimilarity matrices, proposing two complementary approaches — pseudo-Euclidean JL and generalized power distance JL — where the approximation error scales proportionally with the degree of deviation from Euclidean geometry.

## Background & Motivation

1. **State of the Field**: The JL lemma is a cornerstone of dimensionality reduction in Euclidean space, guaranteeing that random linear projections preserve pairwise distances within a $(1\pm\varepsilon)$ factor in $O(\log n/\varepsilon^2)$ dimensions. It is widely applied in approximate nearest neighbor search, clustering, and regression.

2. **Limitations of Prior Work**: The JL lemma requires data to reside in Euclidean space with accessible coordinates, whereas real-world dissimilarities are often non-Euclidean or non-metric (e.g., Minkowski distance, cosine similarity, KL divergence), and sometimes only pairwise dissimilarity matrices — without explicit coordinates — are available.

3. **Root Cause**: It is known that target dimensionality for $\ell_1$, $\ell_p$, and nuclear norm settings must be polynomial in $n$ for constant distortion, making $O(\log n)$ guarantees unattainable. This necessitates fine-grained error analysis rather than worst-case guarantees.

4. **Paper Goals**: To apply JL-type transformations to general symmetric hollow dissimilarity matrices and provide error guarantees that depend on the degree of deviation from Euclidean geometry.

5. **Starting Point**: Two approaches are pursued: (1) embed the dissimilarity matrix into a pseudo-Euclidean space $\mathbb{R}^{p,q}$, with error depending on the ratio of the $(p,q)$-norm to the Euclidean norm; (2) show that any symmetric hollow matrix can be expressed as a generalized power distance matrix, with additive error proportional to the deviation parameter $r^2$.

6. **Core Idea**: The JL transform can be extended to non-Euclidean settings, yielding error guarantees that degrade gracefully as a function of a parameter measuring the degree of non-Euclideanness of the data.

## Method

### Overall Architecture

Given an $n \times n$ symmetric hollow dissimilarity matrix $D$, two methods are proposed: (1) **Pseudo-Euclidean JL**: perform eigendecomposition of the Gram matrix to obtain coordinates in a $(p,q)$-space, then apply JL projections separately to the positive and negative parts; (2) **Power Distance JL**: express $D = E + 4r^2(I - J)$ where $E$ is a Euclidean distance matrix, then apply standard JL projection to the centered coordinates corresponding to $E$.

### Key Designs

**1. Pseudo-Euclidean JL Transform (Theorem 2.3)**

- **Function**: Extends the JL lemma to pseudo-Euclidean spaces of signature $(p, q)$.
- **Mechanism**: The Gram matrix of $D$ is eigendecomposed into positive and negative eigenvalue components, yielding coordinates in $\mathbb{R}^{p,q}$. Independent JL projections are applied to the positive part (dimension $p$) and the negative part (dimension $q$). The resulting error is $(1 \pm \varepsilon \cdot C_{ij})$ times the original distance, where $C_{ij} = |\|x_i - x_j\|_E^2 / \|x_i - x_j\|_{p,q}^2|$.
- **Design Motivation**: When $q$ is small relative to $p$ (i.e., data is close to Euclidean), $C_{ij}$ approaches 1 and the guarantee reduces to the standard JL bound.

**2. Generalized Power Distance JL Transform (Theorem 3.3)**

- **Function**: Provides a $(1 \pm \varepsilon)$ multiplicative plus $4\varepsilon r^2$ additive error guarantee.
- **Mechanism**: Lemma 3.1 establishes that any symmetric hollow matrix can be written as $D = E + 4r^2(I - J)$, where $r = \sqrt{|e_n|}/2$ and $e_n$ is the smallest eigenvalue of the Gram matrix. $E$ is a Euclidean distance matrix, and a standard JL projection is applied to its centered coordinates.
- **Design Motivation**: The additive error $4\varepsilon r^2$ vanishes when the data is Euclidean ($r = 0$). Power distances admit elegant geometric interpretations (as internal tangent lengths between circles) and statistical interpretations (as silhouette coefficients of Gaussian distributions).

**3. Algebraic Result: "Any Dissimilarity is a Power Distance" (Lemma 3.1)**

- **Function**: Establishes a bridge between general dissimilarity matrices and Euclidean geometry.
- **Mechanism**: Adding the term $4r^2(I - J)$ to $D$ renders the Gram matrix positive semidefinite, effectively "lifting" a non-Euclidean matrix into the power distance matrix of circles of equal radius embedded in Euclidean space.
- **Design Motivation**: Provides a canonical method for "Euclideanizing" arbitrary dissimilarity matrices.

### Loss & Training

No training is involved — this is a purely theoretical and algorithmic contribution. The computational bottleneck is SVD of the Gram matrix at $O(n^3)$, which can be accelerated via landmark MDS and related techniques.

## Key Experimental Results

### Main Results

Maximum relative error comparison across 10 datasets:

| Dataset | Standard JL | JL-PE (Pseudo-Euclidean) | JL-Power (Power Distance) |
|---------|-------------|--------------------------|---------------------------|
| Simplex | 6.47e5 | 1.11e6 | **109.18** |
| Ball | 5.97e5 | 5.31e5 | **12.10** |
| MNIST | inf | 8.47e5 | **85.76** |
| CIFAR-10 | inf | 2.24e6 | **55.74** |
| Email | 3.85e4 | 3.24e6 | **781.45** |
| Facebook | 1.18e6 | 2.51e7 | **82.74** |

### Ablation Study

| Validation Target | Result | Notes |
|-------------------|--------|-------|
| Pseudo-Euclidean JL approximation ratio within theoretical bounds | ✅ | Rare minor violations observed |
| Power Distance JL residual additive error $< 4\varepsilon r^2$ | ✅ | All sampled points below upper bound |
| Downstream k-means clustering task | JL-Power best | Clustering quality preserved |

### Key Findings

- JL-Power substantially outperforms standard JL on all non-Euclidean datasets, often by several orders of magnitude.
- Standard JL produces infinite error on MNIST and CIFAR-10, collapsing distinct images to identical points.
- JL-PE performs best on genomic datasets, where the underlying $(p,q)$ structure is better suited to the pseudo-Euclidean approach.
- Experimental results are in strong agreement with theoretical predictions.

## Highlights & Insights

- **"Any dissimilarity is a power distance"**: A deep algebraic result that connects non-Euclidean data to classical geometry.
- **Dual interpretation**: Power distances admit both a geometric meaning (tangent length between circles) and a statistical meaning (silhouette coefficient of Gaussian distributions).
- **Connection to special relativity**: Pseudo-Euclidean space $\mathbb{R}^{p,q}$ subsumes Minkowski spacetime as a special case.
- **Matching lower bounds**: The target dimensionality lower bound $\Omega(\log n / \varepsilon^2)$ for power distance JL matches the standard JL bound.

## Limitations & Future Work

- SVD computation at $O(n^3)$ is a bottleneck for large-scale datasets.
- The $C_{ij}$ factor in the pseudo-Euclidean method can be large for highly non-Euclidean data.
- The additive error $4\varepsilon r^2$ of the power distance method may not be tight for strongly non-Euclidean data.
- Integration with deep learning representation learning approaches remains an open direction.

## Related Work & Insights

- Classical MDS and PCA are data-dependent dimensionality reduction methods; the proposed methods are data-oblivious random projections.
- Non-Euclidean MDS (DGL+ 2024) considers pseudo-Euclidean settings but provides no error guarantees.
- **Insight**: When worst-case $O(\log n)$ guarantees are unattainable, parameterized fine-grained analysis serves as a valuable alternative.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic extension of the JL lemma to general non-Euclidean/non-metric settings.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Theoretical validation + 10 datasets + downstream tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical exposition with well-grounded geometric intuition.
- **Value**: ⭐⭐⭐⭐⭐ Significant contribution to dimensionality reduction theory with broad potential applications.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] On a Geometry of Interbrain Networks](on_a_geometry_of_interbrain_networks.md)
- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](../../ICLR2026/others/distributed_algorithms_for_euclidean_clustering.md)
- [\[NeurIPS 2025\] Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games](efficient_kernelized_learning_in_polyhedral_games_beyond_full-information_from_c.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[ICCV 2025\] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](../../ICCV2025/others/hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)

<!-- RELATED:END -->
