---
title: >-
  [Paper Note] Distributed Algorithms for Euclidean Clustering
description: >-
  [ICLR2026][distributed clustering] This paper constructs $(1+\varepsilon)$-coresets for Euclidean $(k,z)$-clustering in the distributed setting…
tags:
  - "ICLR2026"
  - "distributed clustering"
  - "coreset"
  - "communication complexity"
  - "k-means"
  - "k-median"
date: 2026-05-08
content_hash: 34c7f6104d22fb4a
---

# Distributed Algorithms for Euclidean Clustering

**Conference**: ICLR2026
**arXiv**: [2603.08615](https://arxiv.org/abs/2603.08615)  
**Code**: None (theoretical work)  
**Area**: Other
**Keywords**: distributed clustering, coreset, communication complexity, k-means, k-median

## TL;DR

This paper constructs $(1+\varepsilon)$-coresets for Euclidean $(k,z)$-clustering in the distributed setting, achieving communication complexity that matches known lower bounds (up to polylogarithmic factors) in both the coordinator model and the blackboard model.

## Background & Motivation

- **Euclidean $(k,z)$-clustering** is a classical clustering problem: given $n$ points in $d$ dimensions, find $k$ centers minimizing the sum of $z$-th power distances from each point to its nearest center. The case $z=1$ corresponds to $k$-median and $z=2$ to $k$-means.
- Modern datasets are massive and naturally distributed across multiple machines, precluding centralized processing. **Distributed clustering** has become a core requirement: $n$ points are distributed across $s$ machines, which collaborate by exchanging compact summaries (coresets) to perform clustering while minimizing communication.
- Existing distributed methods (e.g., merge-and-reduce) exhibit multiplicative coupling among $s$, $d$, and $1/\varepsilon$ in their communication complexity (e.g., $\tilde{O}(skd/\varepsilon^4 \cdot \log(n\Delta))$), far from the known communication lower bounds.
- **Core Motivation**: Can one design communication-optimal protocols that decouple $s$ (number of machines), $d$ (dimensionality), and $1/\varepsilon$ (approximation accuracy) to match theoretical lower bounds?

## Core Problem

The paper aims to construct $(1+\varepsilon)$-strong coresets for $(k,z)$-clustering in two classical distributed communication models, while achieving optimal communication complexity:

1. **Coordinator Model**: $s$ machines communicate only through a coordinator relay, using private channels and private randomness.
2. **Blackboard Model**: $s$ machines communicate via a shared blackboard, where any message written by one machine is visible to all others.

## Method

### Overall Architecture

The protocol proceeds in two phases: (1) obtain a constant-factor approximation; (2) upgrade to a $(1+\varepsilon)$-coreset via sensitivity sampling. The key innovation lies in substantially reducing communication at each phase.

### Constant-Factor Approximation in the Blackboard Model

- Existing adaptive sampling (analogous to $k$-means++) requires all $s$ machines to report their distance sums $D_i$ after each round, incurring $O(sk\log n)$ communication—prohibitively expensive.
- **Lazy Sampling**: The blackboard maintains approximate values $\widehat{D_i}$ of each machine's distance sum $D_i$. Machines are selected for sampling proportionally to $\widehat{D_i}$. Provided $\widehat{D_i}$ is a constant-factor approximation of $D_i$, the sampling failure probability is a constant, and the total number of sampling rounds remains $O(k)$.
- When sampling fails—indicating that $D_i$ has decreased significantly—machine $i$ updates its value on the blackboard. Each machine updates at most $O(\log n)$ times, transmitting only $O(\log\log n)$ bits per update (by sending the exponent).
- **$L_1$ Sampling for Acceleration**: Rather than requiring all machines to report each round, a single machine $i$ is randomly selected with probability proportional to $\widehat{D_i}$. The estimator $D_i/p_i$ is used to estimate whether the global total weight $\sum D_i$ has decreased by more than $1/64$, triggering a global update only when necessary.
- **Exponentially Growing Batch Sampling**: When the global weight has not decreased significantly, $2^i$ samples are drawn in a single batch (with $i$ increasing incrementally), reducing the number of communication rounds to $O(\log n \log k)$.

### Key Innovations in the Coordinator Model

- **Coordinate-wise Sampling**: Rather than transmitting full high-dimensional center coordinates, the coordinator and a machine perform a distributed binary search over sorted coordinates, transmitting only a small offset. This decouples communication from dimensionality $d$.
- **Coordinate-wise Sensitivity Sampling**: Each point is decomposed along its coordinates, and dimensions are sampled according to their importance. The coordinator sends a compact summary, and machines request additional information only when necessary.
- The reconstructed samples may not correspond to any actual point in the dataset, but the paper proves that the distortion of the clustering cost remains controlled.

### $(1+\varepsilon)$-Coreset Construction

- Using the constant-factor approximation $C'$, the sensitivity $\mu(x)$ of each point is computed, and $\tilde{O}(k/\min(\varepsilon^4, \varepsilon^{2+z}))$ points are sampled according to the sensitivity distribution.
- **Compact Encoding**: Each sampled point is encoded as $(c'(x), y)$, where $c'(x)$ is the index of the nearest center and $y$ is the coordinate-wise logarithmic exponent of the residual vector. Each point requires only $O(\log k + d\log(1/\varepsilon, d, \log(n\Delta)))$ bits.

### Main Theorems

**Theorem 1.1 (Coordinator Model)**: Total communication is $\tilde{O}(sk + dk/\min(\varepsilon^4, \varepsilon^{2+z}) + dk\log(n\Delta))$ bits.

**Theorem 1.2 (Blackboard Model)**: Total communication is $\tilde{O}(s\log(n\Delta) + dk\log(n\Delta) + dk/\min(\varepsilon^4, \varepsilon^{2+z}))$ bits.

Both results match known lower bounds up to polylogarithmic factors.

## Key Experimental Results

This is a purely theoretical work with no experimental component. The primary contributions are the proofs of communication complexity upper bounds and their matching with known lower bounds.

| Model | Prior Best | Ours | Improvement |
|-------|-----------|------|-------------|
| Coordinator | $O(skd/\varepsilon^4 \cdot \log(n\Delta))$ | $\tilde{O}(sk + dk/\varepsilon^4 + dk\log(n\Delta))$ | Decouples $s$, $d$, $1/\varepsilon$; removes multiplicative dependence |
| Blackboard | $\tilde{O}((s+dk)\log^2(n\Delta))$ (constant approx. only) | $\tilde{O}(s\log(n\Delta) + dk\log(n\Delta) + dk/\varepsilon^4)$ | Upgrades from constant to $(1+\varepsilon)$ approx.; removes extra $\log$ factors |

## Highlights & Insights

- **Communication Matches Optimal Lower Bounds**: Both distributed models match the lower bounds of [Chen et al., NeurIPS 2016] and [Huang et al., STOC 2024], representing a complete theoretical resolution of the problem.
- **Parameter Decoupling**: In the coordinator model, $s$ (number of machines) is no longer multiplied by $d$ (dimensionality) and $1/\varepsilon$ (precision); in the blackboard model, $1/\varepsilon$ is no longer multiplied by $\log(n\Delta)$.
- **No Plaintext Coordinate Broadcast**: In the coordinator model, no site needs to broadcast full point coordinates to all machines—an unexpected and important result.
- **Lazy Sampling + $L_1$ Sampling**: An elegant variant of distributed adaptive sampling that avoids global communication at every round.
- **Coordinate-wise Sensitivity Sampling** is a novel technique that may be of independent value for distributed regression, low-rank approximation, and related problems.
- The results generalize to arbitrary connected communication topologies, not restricted to the coordinator or blackboard setting.

## Limitations & Future Work

- **Purely Theoretical**: No experiments validate improvements in actual communication volume or runtime; in particular, the polylogarithmic factors may be non-trivial in practice.
- **Round Complexity**: In the blackboard model, while the total bit complexity is optimal, the number of rounds is $O(\log n \log k)$, which may be undesirable in latency-sensitive settings.
- **Degradation When $\varepsilon$ Is Constant**: When $\varepsilon$ is a constant, the $1/\varepsilon^4$ term becomes a constant, and the dominant communication term becomes $dk\log(n\Delta)$ (transmitting the center coordinates themselves), which cannot be further optimized.
- **Finite-Precision Grid Assumption**: Input points are assumed to lie on a $\{1,\ldots,\Delta\}^d$ grid; continuous data in practice requires an additional discretization step.
- **Restricted to Euclidean Space**: Non-Euclidean metrics and non-clustering cost functions are outside the scope of this paper.

## Related Work & Insights

| Work | Approximation | Communication | Notes |
|------|--------------|---------------|-------|
| Merge-and-reduce + [BCP+24] | $(1+\varepsilon)$ | $\tilde{O}(skd/\varepsilon^4 \cdot \log(n\Delta))$ | Coupled parameters |
| [BEL13] + [BCP+24] | $(1+\varepsilon)$ | $O(dk/\varepsilon^4 \cdot \log(n\Delta) + sdk\log(sk)\log(n\Delta))$ | $s$ still multiplies $dk$ |
| [CSWZ16] (Blackboard) | $O(1)$ | $\tilde{O}((s+dk)\log^2(n\Delta))$ | Constant approx. only |
| **Ours (Coordinator)** | $(1+\varepsilon)$ | $\tilde{O}(sk + dk/\varepsilon^4 + dk\log(n\Delta))$ | **Optimal** |
| **Ours (Blackboard)** | $(1+\varepsilon)$ | $\tilde{O}(s\log(n\Delta) + dk\log(n\Delta) + dk/\varepsilon^4)$ | **Optimal** |

**Broader Connections**:

- The **distributed algorithms + coreset** paradigm—first obtaining a constant-factor approximation at low communication cost, then refining via sensitivity sampling—may transfer to other distributed optimization problems such as low-rank approximation and regression.
- The **Lazy Sampling** philosophy (tolerating stale information and updating only when deviation exceeds a threshold) is analogous to eventual consistency in distributed systems and may apply to gradient compression in distributed SGD.
- **Coordinate-wise Sampling** decomposes the high-dimensional communication problem into lower-dimensional per-coordinate subproblems, paralleling the role of quantization and sparsification in federated learning.
- The authors explicitly note natural connections between these techniques and data quantization and communication compression in LLM training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Resolves an open problem on communication optimality in distributed clustering)
- Experimental Thoroughness: ⭐⭐ (Purely theoretical; no experiments)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; technical overview is well explained)
- Value: ⭐⭐⭐⭐ (Complete theoretical resolution; techniques have broad transfer potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/others/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/others/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](../../NeurIPS2025/others/coresets_for_clustering_under_stochastic_noise.md)
- [\[NeurIPS 2025\] Johnson-Lindenstrauss Lemma Beyond Euclidean Geometry](../../NeurIPS2025/others/johnson-lindenstrauss_lemma_beyond_euclidean_geometry.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](../../AAAI2026/others/improved_differentially_private_algorithms_for_rank_aggregation.md)

</div>

<!-- RELATED:END -->
