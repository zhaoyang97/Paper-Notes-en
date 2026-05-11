---
title: >-
  [Paper Note] Optimized Learned Count-Min Sketch
description: >-
  [NeurIPS 2025 (Workshop: ML for Systems)][Count-Min Sketch] This paper proposes OptLCMS, which partitions the score space and analytically solves CMS parameters via KKT conditions while optimizing thresholds through dyna…
tags:
  - "NeurIPS 2025 (Workshop: ML for Systems)"
  - "Count-Min Sketch"
  - "frequency estimation"
  - "learned data structures"
  - "dynamic programming"
  - "KKT conditions"
date: 2026-05-08
content_hash: 1470a9ec85cf0f52
---

# Optimized Learned Count-Min Sketch

**Conference**: NeurIPS 2025 (Workshop: ML for Systems)
**arXiv**: [2512.12252](https://arxiv.org/abs/2512.12252)
**Code**: None
**Area**: Data Structures / Learning-Augmented Systems
**Keywords**: Count-Min Sketch, frequency estimation, learned data structures, dynamic programming, KKT conditions

## TL;DR

This paper proposes OptLCMS, which partitions the score space and analytically solves CMS parameters via KKT conditions while optimizing thresholds through dynamic programming, substantially accelerating the construction process and providing theoretical guarantees on the probability of intolerable error.

## Background & Motivation

Count-Min Sketch (CMS) is a probabilistic data structure for estimating element frequencies in a multiset under memory-efficient conditions, widely applied in network traffic monitoring, search query statistics, and related domains. CMS exhibits an inherent **error–memory** trade-off: reducing memory increases error, while reducing error demands more memory.

Learned CMS (LCMS) introduces a machine learning model to generate approximate frequency scores and uses a threshold to determine how elements are stored (dictionary vs. CMS), achieving lower estimation error under a fixed memory budget. However, LCMS has two critical limitations:

**Slow construction**: Requires empirical parameter tuning on validation data (exhaustive search over thresholds and CMS parameters).

**Lack of theoretical guarantees**: Cannot provide a theoretical upper bound on the probability of *intolerable error* (the probability that the estimation error exceeds a user-specified threshold).

## Method

### Overall Architecture

OptLCMS extends Partitioned Learned CMS (PL-CMS). The overall structure comprises three components:

- **Pre-trained ML model**: Generates frequency scores for each element.
- **$G$ CMS tables**: Corresponding to $G$ score intervals, each group with independent parameters $(\epsilon_g, \delta_g)$.
- **Unique Bucket (UB)**: For exact counting of high-frequency elements.

Elements are routed to components based on model scores: high-score elements enter the UB for exact counting, while others are directed to the CMS of the corresponding score group.

### Key Designs

**Optimization problem formulation**: Under a fixed memory budget $M$, minimize the upper bound on the intolerable error probability:

$$\min_{\boldsymbol{t}, \boldsymbol{\epsilon}, \boldsymbol{\delta}} \sum_{g=1}^{G} \delta_g q_g$$

Subject to:
- Memory constraint: total memory of all CMS tables plus UB equals budget $M$.
- CMS constraint: $\delta_g \leq 1$.
- Global error consistency: $\epsilon_g = \frac{\epsilon N}{N_g}$, ensuring each CMS's allowable error is consistent with the global specification.

**Analytical solution for $\delta$ (KKT conditions)**: When thresholds $\boldsymbol{t}$ are fixed, the problem is convex in $\boldsymbol{\delta}$, admitting a closed-form solution via KKT conditions:

$$\delta_g = \min\left\{1, \frac{1}{q_g \epsilon_g} \exp\left[-\frac{\frac{M-cn}{be} - I}{\sum_{g \in \mathcal{D}} \frac{1}{\epsilon_g}}\right]\right\}$$

where $I = \sum_{g \in \mathcal{D}} \frac{1}{\epsilon_g} \ln(q_g \epsilon_g)$ and $\mathcal{D}$ is the set of groups with $\delta_g < 1$.

**Threshold optimization (dynamic programming)**: Substituting the optimal $(\boldsymbol{\epsilon}, \boldsymbol{\delta})$ into the objective, determination of optimal thresholds reduces to maximizing the KL divergence between the data distribution and the query distribution, solved via dynamic programming.

### Loss & Training

OptLCMS does not involve conventional neural network training. The core procedure is:
- The ML model reuses the pre-trained model from LCMS.
- CMS parameters $(\epsilon, \delta)$ are computed directly via the analytical solution/DP.
- $G = 10$ partitions are used.
- $\epsilon$ is set to the minimum feasible value for CMS under the given memory: $\epsilon = e/M$.

## Key Experimental Results

### Main Results: Memory vs. Intolerable Error Probability

Evaluated on the AOL query log dataset (21 million queries, 3.8 million unique terms, Zipf distribution):

| Data Structure | Memory Efficiency | Intolerable Error Prob. | Mean Error |
|---|---|---|---|
| CMS | Baseline | Highest | Highest |
| LCMS | Better | Moderate | Lower |
| OptLCMS ($G=10$) | Better | **Lowest (~1/20 of LCMS)** | **Comparable to LCMS** |

Key finding: OptLCMS reduces the intolerable error probability by approximately **20×** compared to LCMS while maintaining comparable mean error.

### Construction Time Comparison

| Data Structure | Construction Time (s) — Frequency-weighted | Construction Time (s) — Uniform |
|---|---|---|
| LCMS | 10.712 | 10.250 |
| OptLCMS | **0.003** (−10.709) | **4.873** (−5.377) |

Construction speed improves by approximately **3500×** under frequency-weighted queries and approximately **2×** under uniform queries.

### Key Findings

1. **Analytical derivation eliminates the validation phase**: OptLCMS computes optimal parameters directly from the validation data distribution without iterative testing on actual data.
2. **Minimizing the intolerable error upper bound does not sacrifice mean error**: The two objectives are not in conflict.
3. **The partitioning strategy effectively exploits score-space heterogeneity**: Differences in data and query distributions across score groups are captured by KL divergence.

## Highlights & Insights

- **Elegant theoretical derivation**: The parameter tuning problem for learned data structures is reformulated as a convex optimization problem; KKT conditions yield an analytical solution, and DP optimizes the thresholds.
- **Strong practicality**: The substantial speedup in construction is particularly beneficial for online scenarios requiring frequent CMS rebuilding.
- **Theoretical guarantees**: OptLCMS fills the gap left by LCMS, providing formal bounds on the intolerable error probability.
- **Elegant handling of integer relaxation**: The ceiling constraints on CMS dimensions are relaxed to continuous variables; the resulting non-convexity is negligible in practice.

## Limitations & Future Work

1. **Evaluated on only the AOL dataset**: Not tested on the CAIDA network traffic dataset.
2. **No comparison with confidence-aware LCMS**: Lacks comparison against more recent improved variants.
3. **Number of partitions $G$ fixed at 10**: The effect of varying $G$ on performance is not explored.
4. **Depends on the quality of the pre-trained ML model**: The accuracy of model scores directly affects partitioning effectiveness.
5. **Workshop paper with limited scope**: Theoretical analysis is thorough, but experimental scale is small.

## Related Work & Insights

- **Learned Bloom Filter → Learned CMS**: Extends the partitioning and optimization ideas from learned Bloom filters to the CMS domain.
- **PLBF optimization framework**: The partitioning strategy of the Partitioned Learned Bloom Filter provides direct inspiration.
- **Data structures × convex optimization**: Demonstrates the effective application of classical convex optimization tools (KKT conditions, DP) in learning-augmented data structures.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introduces a convex optimization framework for parameter tuning in learned CMS.
- Theoretical Contribution: ⭐⭐⭐⭐⭐ — Complete analytical derivation with formal guarantees.
- Experimental Thoroughness: ⭐⭐⭐ — Single dataset, but comprehensive metrics.
- Value: ⭐⭐⭐⭐ — Improvements in construction speed and error guarantees have direct practical relevance.
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adjusted Count Quantification Learning on Graphs](adjusted_count_quantification_learning_on_graphs.md)
- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](semi-infinite_nonconvex_constrained_min-max_optimization.md)
- [\[ICCV 2025\] Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation](../../ICCV2025/others/stroke2sketch_harnessing_stroke_attributes_for_training-free_sketch_generation.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](../../ICCV2025/others/switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](../../ICCV2025/others/doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)

</div>

<!-- RELATED:END -->
