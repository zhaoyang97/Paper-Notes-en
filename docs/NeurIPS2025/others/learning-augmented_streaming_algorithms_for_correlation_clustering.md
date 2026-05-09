---
title: >-
  [Paper Note] Learning-Augmented Streaming Algorithms for Correlation Clustering
description: >-
  [NeurIPS 2025][Correlation Clustering] This paper proposes the first learning-augmented streaming algorithms for Correlation Clustering. By leveraging pairwise distance predictions, the proposed methods achieve a better-than-3 approximation ratio on complete graphs ($\tilde{O}(n)$ space) and an $O(\log|E^-|)$ approximation ratio on general graphs ($\tilde{O}(n)$ space), significantly improving the space–approximation tradeoff over existing prediction-free algorithms.
tags:
  - NeurIPS 2025
  - Correlation Clustering
  - Streaming Algorithms
  - Learning-Augmented
  - Space Efficiency
  - Approximation Algorithms
date: 2026-05-08
content_hash: 7ec6f21ebb40a11b
---

# Learning-Augmented Streaming Algorithms for Correlation Clustering

**Conference**: NeurIPS 2025

**arXiv**: [2510.10705](https://arxiv.org/abs/2510.10705)

**Code**: None

**Area**: Streaming Algorithms / Clustering

**Keywords**: Correlation Clustering, Streaming Algorithms, Learning-Augmented, Space Efficiency, Approximation Algorithms

## TL;DR

This paper proposes the first learning-augmented streaming algorithms for Correlation Clustering. By leveraging pairwise distance predictions, the proposed methods achieve a better-than-3 approximation ratio on complete graphs ($\tilde{O}(n)$ space) and an $O(\log|E^-|)$ approximation ratio on general graphs ($\tilde{O}(n)$ space), significantly improving the space–approximation tradeoff over existing prediction-free algorithms.

## Background & Motivation

**Correlation Clustering** is a classical graph clustering problem:

- **Input**: A graph $G=(V,E)$ where each edge is labeled as "positive" (endpoints should be in the same cluster) or "negative" (endpoints should be in different clusters)
- **Objective**: Partition vertices into disjoint clusters to minimize the number of **disagreements**—positive edges crossing clusters and negative edges within clusters
- **Streaming model**: Edges arrive in arbitrary order, and the algorithm is restricted to limited memory

Traditional streaming algorithms face the challenge of the **space–approximation tradeoff**:

| Algorithm | Graph Type | Approximation Ratio | Space |
|------|--------|--------|------|
| Cambus et al. (SODA'24) | Complete graph | 3 | $\tilde{O}(n)$ |
| Ahn et al. (ICML'15) | General graph | $O(\log n)$ | $\tilde{O}(n \cdot \log n)$ |

Core Problem: Can machine learning predictions be leveraged to simultaneously improve both the approximation ratio and space efficiency?

The **learning-augmented algorithms** paradigm augments classical algorithms with (possibly inaccurate) predictions, achieving superior performance when predictions are accurate and gracefully degrading to classical guarantees when predictions are poor.

## Method

### Overall Architecture

The algorithms use **pairwise distance predictions provided by a predictor** as auxiliary information:

- The predictor estimates, for each pair $(u,v)$, their "distance" in the optimal clustering (i.e., whether they belong to the same cluster)
- An initial clustering structure is constructed based on these predictions
- During stream processing, actual edge information is used to correct prediction errors

### Key Designs

#### Complete Graph Algorithm

**Design Motivation**:

1. **Preprocessing phase**: Pre-group vertices according to predicted distances
2. **Stream processing phase**: Update groupings upon reading each edge using its label (positive/negative)
3. **Key technique**: Builds on the framework of Cambus et al. (SODA'24), incorporating predictions to relax sampling requirements

**Theoretical Guarantees**:
- When prediction quality is high: approximation ratio **better than 3** (breaking the prediction-free lower bound)
- Space complexity: $\tilde{O}(n)$
- When predictions are entirely uninformative: degrades to a 3-approximation

**Prediction Quality Measure**:

The prediction error $\eta$ is defined as the degree of inconsistency between predicted distances and the true optimal distances. The approximation ratio varies smoothly with $\eta$:

$$\text{Approximation Ratio} = f(\eta), \quad f(0) < 3, \quad f(\infty) = 3$$

#### General Graph Algorithm

**Design Motivation**:

1. Builds on the linear sketching method of Ahn et al. (ICML'15)
2. Uses predicted distances to reduce the amount of edge information that must be stored
3. Achieves a logarithmic approximation with respect to the negative edge set $E^-$

**Theoretical Guarantees**:
- Approximation ratio: $O(\log|E^-|)$ (when prediction quality is high)
- Space complexity: $\tilde{O}(n)$
- Improves the space efficiency of known prediction-free algorithms

**Comparison with Prior Results**:

| Algorithm | Approximation Ratio | Space | Notes |
|------|--------|------|------|
| Ahn et al. (no learning) | $O(\log n)$ | $\tilde{O}(n \log n)$ | Baseline |
| Ours (learning-augmented) | $O(\log |E^-|)$ | $\tilde{O}(n)$ | Better space |

Note that $|E^-| \leq \binom{n}{2}$, so $\log|E^-| \leq 2\log n$; in practice, $|E^-|$ is typically much smaller than $n^2$.

### Loss & Training

This paper does not address predictor training; instead, it analyzes algorithm performance given a fixed predictor. The predictor can be:
- A classifier trained on historical data
- A heuristic based on graph features
- Any method capable of providing pairwise distance estimates

## Key Experimental Results

### Main Results

Experiments are conducted on both synthetic and real-world datasets.

**Synthetic Data (Complete Graph)**:

| Prediction Error $\eta$ | Ours | Cambus et al. (no learning) | Gain |
|:---:|:---:|:---:|:---:|
| 0% | Optimal | 3× | Significant |
| 10% | <3× | 3× | Notable |
| 30% | ≈3× | 3× | On par |
| 50%+ | 3× | 3× | Degraded |

**Real-World Data**:

| Dataset | Ours | Non-learning Baseline | Space Usage |
|--------|:---:|:---:|:---:|
| Synthetic random graph | Better than baseline | Baseline | $\tilde{O}(n)$ |
| Real social network | Significantly better | Baseline | $\tilde{O}(n)$ |

### Ablation Study

**Effect of Prediction Quality on Performance**:

| Component | Good Prediction | Moderate Prediction | Poor Prediction |
|------|:---:|:---:|:---:|
| Complete graph approx. ratio | <3 | ≈3 | =3 |
| General graph approx. ratio | $O(\log|E^-|)$ | Intermediate | $O(\log n)$ |
| Space usage | $\tilde{O}(n)$ | $\tilde{O}(n)$ | $\tilde{O}(n)$ |

Key observation: **The improvement in space efficiency is unconditional**—the $\tilde{O}(n)$ space bound is maintained regardless of prediction quality.

**Effect of Edge Arrival Order**:

| Edge Arrival Order | Performance Impact |
|-----------|---------|
| Random order | Optimal |
| Adversarial order | Slight degradation, but still better than non-learning methods |
| Natural order | Close to random |

### Key Findings

1. The learning-augmented approach is the first to break the 3-approximation barrier for streaming correlation clustering on complete graphs, demonstrating that prediction information can provide substantial benefit in the streaming setting.
2. The space improvement from $\tilde{O}(n \log n)$ to $\tilde{O}(n)$ is unconditional and does not depend on prediction quality.
3. On real-world datasets, the learning-augmented algorithm consistently outperforms its prediction-free counterpart even when predictions are imperfect.
4. The key algorithmic design insight is to use predictions to reduce the edge information that must be stored, rather than directly substituting predictions for algorithmic decisions.

## Highlights & Insights

1. **First application of learning-augmentation to streaming correlation clustering**: Opens a new research direction.
2. **Elegant degradation guarantee**: Naturally degrades to the known optimum when predictions are poor, without requiring knowledge of prediction quality.
3. **Unconditional space improvement**: The space efficiency gain is independent of prediction quality, which is uncommon among learning-augmented algorithms.
4. **Integration of theory and practice**: Combines rigorous theoretical analysis with empirical validation on real-world data.

## Limitations & Future Work

1. **Predictor design left unspecified**: The paper does not discuss how to construct effective distance predictors.
2. **Complete graph assumption**: The complete graph assumption may be unrealistic for large-scale data.
3. **No closed-form expression for approximation ratio**: The concrete better-than-3 approximation ratio on complete graphs depends on prediction quality without a closed-form characterization.
4. **Single-pass restriction**: The current algorithms may require multi-pass extensions to achieve stronger results.
5. **Potential direction**: Investigating learning-augmented clustering in the semi-streaming setting ($O(n \text{ polylog } n)$ space).

## Related Work & Insights

- **Cambus et al. (SODA'24)**: Foundational framework for the 3-approximation streaming algorithm on complete graphs.
- **Ahn et al. (ICML'15)**: Linear sketch-based correlation clustering on general graphs.
- **Learning-augmented online algorithms**: Pioneering work by Lykouris & Vassilvitskii et al.; this paper extends the paradigm to the streaming computation model.
- **Other correlation clustering algorithms**: Offline settings admit PTAS and other stronger results; the constraints of the streaming setting make learning augmentation particularly valuable.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First learning-augmented streaming algorithm for correlation clustering
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous theoretical analysis with meaningful improvements in the space–approximation tradeoff
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic and real-world data with multi-dimensional ablation
- **Practicality**: ⭐⭐⭐ — Potential applications in large-scale clustering scenarios
- **Overall**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)
- [\[NeurIPS 2025\] MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering](mutualvpr_a_mutual_learning_framework_for_resolving_supervision_inconsistencies_.md)
- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](../../ICLR2026/others/distributed_algorithms_for_euclidean_clustering.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)

</div>

<!-- RELATED:END -->
