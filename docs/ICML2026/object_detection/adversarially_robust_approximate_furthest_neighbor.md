---
title: >-
  [Paper Note] Adversarially Robust Approximate Furthest Neighbor
description: >-
  [ICML 2026][Object Detection][Paper Note] This theoretical paper provides the first approximate furthest neighbor data structure resistant to adaptive query adversaries. While maintaining a query complexity with $n$-dependence similar to Indyk's classical oblivious algorithm, it demonstrates that traditional random projection furthest neighbor algorithms can b
tags:
  - ICML 2026
  - Object Detection
date: 2026-05-08
content_hash: 4ec7a363424d219b
---
# Adversarially Robust Approximate Furthest Neighbor

**Conference**: ICML 2026  
**arXiv**: [2605.16618](https://arxiv.org/abs/2605.16618)  
**Code**: No public code  
**Area**: Optimization / Theoretical Algorithms  
**Keywords**: Approximate Furthest Neighbor, Adaptive Query, Adversarially Robust Data Structures, Random Projection, High-dimensional Geometry  

## TL;DR
This theoretical paper provides the first approximate furthest neighbor data structure resistant to adaptive query adversaries. While maintaining a query complexity with $n$-dependence similar to Indyk's classical oblivious algorithm, it demonstrates that traditional random projection furthest neighbor algorithms can be broken by adaptive queries.

## Background & Motivation
**Background**: Nearest neighbor and furthest neighbor searches are fundamental geometric primitives in high-dimensional data analysis. Though less discussed than nearest neighbor, furthest neighbor is natural in diversity maximization, anomaly detection, hard negative mining, adversarial sample generation, reinforcement learning exploration, and clustering.

**Limitations of Prior Work**: Classical randomized data structures typically assume all queries are fixed before the data structure's randomness is determined, known as oblivious queries. Modern machine learning pipelines more commonly feature interactive or closed-loop scenarios: after the algorithm returns a point, the next query is adjusted based on previous answers. Such adaptive queries can leak the random blind spots of the data structure, invalidating classical Monte Carlo guarantees.

**Key Challenge**: Furthest neighbor is a search problem that returns concrete points rather than simple distance values. While distance estimation can be robustified using covering balls and stable estimators, the identity of furthest neighbor candidates can change abruptly with the query position, making it difficult to directly apply existing adaptive distance estimation frameworks.

**Goal**: The authors aim to answer a fundamental question: under a fully adaptive query model, can approximate furthest neighbor still achieve sublinear query time relative to the data scale $n$? Additionally, they seek to explain why the direct use of Indyk’s oblivious random projection algorithm is not robust.

**Key Insight**: The paper treats the classical random projection algorithm as an analyzable "white box": first strengthening the smooth success guarantee of a single query, then using query space covering and union bounds to ensure the guarantee holds for all possible queries simultaneously. Finally, only a few base data structures are sampled and filtered using robust distance estimation.

**Core Idea**: Use multiple independent random projection data structures to cover the entire query space, ensuring any adaptive query is a "good query" for at least half of the structures. Then, use a small amount of random sampling and robust distance estimation to select the approximate furthest neighbor from the candidate set.

## Method
This paper does not feature a conventional experimental system; its core contributions are algorithmic construction, complexity proofs, and attacks on oblivious algorithms. The method can be understood as upgrading Indyk’s random projection furthest neighbor algorithm from "high probability success for a fixed query" to "simultaneous success for all possible adaptive queries."

### Overall Architecture
Given a point set $P \subset \mathbb{R}^d$ and an approximation factor $c > 1$, the algorithm constructs $k$ independent base data structures during preprocessing. Each base structure consists of $N \approx \tilde{\Theta}(n^{1/c^2})$ Gaussian random projections and stores the top candidates for each projection direction. During querying, the algorithm randomly samples $m = \Theta(\log n)$ base structures, collects their returned candidate sets, and uses an adaptively robust distance estimation structure to estimate the distances from candidates to the query point, returning the candidate with the largest estimated distance.

### Key Designs

**1. Good query definition with slack: Allowing success properties to "propagate" to nearby queries**

Adaptive adversaries prefer placing the next query near the boundaries of the current structure's random blind spots, so proving success for a fixed query is insufficient. The authors strengthen the success condition with a "slack" version: if the true furthest neighbor $p^*$ of query $q$ is sufficiently prominent in a projection direction and the number of outlier projections for incorrect candidates does not exceed $8N$, $q$ is called $(c, \delta)$-good for that projection matrix. They prove that using $N = \tilde{\Theta}(n^{1/c^2})$ Gaussian projections, a fixed query satisfies this property with at least $3/4$ probability. The significance of slack is: as long as $q'$ is within $\Delta/n^3$ of $q$, the good property of $q$ transfers to $q'$. This transforms discrete success events for fixed queries into local stable events that can be covered by a grid.

**2. Query space covering + Union bound across multiple copies: Upgrading from "any fixed query" to "all queries simultaneously"**

Since adaptive query sequences can be infinitely long, one cannot rely on small failure probabilities and union bounds over the number of queries—a guarantee must be proven for the entire continuous query space at once. The authors first show that queries sufficiently far from the center of the point set can be solved approximately with a trivial answer, leaving only a bounded sphere to cover. A grid is constructed over this sphere, and $k = \tilde{\Theta}(d)$ independent base structures are selected. Chernoff bounds and union bounds prove that each grid point is "good" for at least $k/2$ structures. Using the smoothness from the previous design, this conclusion is extended from grid points to any query within the sphere, ensuring any adaptive query is good for at least half the structures. This is the key to resisting infinite adaptive sequences.

**3. Sampling a few candidates + Robust distance estimation: Avoiding full structure traversal while reducing dimensional dependence**

Since any query is good for at least half of the base structures, the query process does not need to access all $k$ structures. Sampling $m = \Theta(\log n)$ ensures a high probability of hitting at least one good structure; the fresh randomness of the sampling is not contaminated by past queries. The collected candidate set size is approximately $\tilde{O}(n^{1/c^2})$ or $\tilde{O}(n^{2/c^2})$. Direct distance calculation results in $\tilde{O}(d n^{1/c^2})$ query time; alternatively, using Cherapanamjeri-Nelson robust distance estimation as a black box on the candidate subset reduces query time to $\tilde{O}(\min\{n^{2/c^2}, n\} + d)$, at the cost of the approximation factor degrading to $(1+\epsilon)c$. Embedding robust distance estimation into the search algorithm is a crucial compositional technique—candidate generation and distance comparison are robustified separately.

### Loss & Training
This is a theoretical algorithm paper with no training loss. Preprocessing complexity is $\tilde{O}(d^2 n^{1+1/c})$. One version returns a $c$-approximate AFN with query time $\tilde{O}(d n^{1/c^2})$; another returns a $(1+\epsilon)c$-approximate AFN with query time $\tilde{O}(\min\{n^{2/c^2}, n\} + d)$. Space complexities involve $\tilde{O}(d \cdot \min\{n, d n^{2/c^2}\})$ or an additional $\tilde{O}(d^2)$ term.

## Key Experimental Results

### Main Results
The "Main Results" correspond to the primary theoretical results and complexity comparisons rather than empirical benchmarks.

| Method / Result | Query Model | Approx. Factor | Query Time | Space | Notes |
|-----------------|-------------|----------------|------------|-------|-------|
| Indyk 2003 AFN | oblivious | $\approx c$ | $\tilde{O}(d n^{1/c^2})$ | No adaptive guarantee | Efficient for fixed queries, broken by adaptive attack |
| Cherapanamjeri-Nelson ADE + scan | adaptive | $c$ | $\tilde{O}(n+d)$ | Robust | Handles adaptivity but close to linear scan |
| Ours (Version 1) | adaptive / white-box | $c$ | $\tilde{O}(d n^{1/c^2})$ | $\tilde{O}(d\min\{n,d n^{2/c^2}\})$ | Matches $n$-dependence of oblivious algorithm |
| Ours (Version 2) | adaptive / white-box | $(1+\epsilon)c$ | $\tilde{O}(\min\{n^{2/c^2},n\}+d)$ | $\tilde{O}(d^2+d\min\{n,d n^{2/c^2}\})$ | Uses robust distance estimation to lower $d$ multiplier |

### Ablation Study
Ablation here refers to the analysis of algorithmic components: removing a component causes the theoretical guarantee to degrade or fail.

| Component / Variant | Function | Effect if Missing |
|---------------------|----------|-------------------|
| Single Indyk-style structure | Sublinear candidate generation | Only guarantees oblivious queries; adaptive adversary can exploit projection directions |
| $(c,\delta)$-good + Inherited property | Stability against small perturbations | Cannot generalize from grid points to continuous query space |
| $k$ structures + Covering union bound | Global validity for all queries | Failure probability depends on query count; fails for infinite adaptive sequences |
| Randomly sampling $m=\Theta(\log n)$ structures | Avoid querying all $k$ structures | Increases query overhead; insufficient sampling leads to missing good structures |
| Robust distance estimation filtering | Secure distance comparison in candidate set | Plain distance calculation retains $d n^{1/c^2}$ dependence; standard JL is not robust |

### Key Findings
- Sublinear adaptive AFN is feasible: When $d = \mathrm{poly}(\log n)$, Version 1 is sublinear in $n$ for any $c > 1$; when $c > \sqrt{2}$ and $d = o(n)$, Version 2 also remains sublinear.
- The guarantee is stronger than black-box differential privacy style reconstruction: The authors prove the algorithm holds even under a white-box adversary, meaning leakage of past information does not destroy the high-probability success of all queries.
- Attack results show that classical oblivious guarantees cannot be simply translated to interactive ML pipelines. The authors construct a dataset where the algorithm returns a point at distance $d^{0.01}$ while the true furthest distance is at least $d^{0.5}$ when the query depends on random projections.

## Highlights & Insights
- The most inspiring aspect is decomposing the search problem's robustification into "robust candidate generation" and "robust candidate distance comparison," rather than attempting a one-time global stability proof for the returned point.
- The definition of $(c, \delta)$-good query with slack is crucial. It converts random projection success from a discrete event for fixed queries into a locally stable event suitable for grid covering, providing the core bridge for handling continuous query spaces.
- The paper demonstrates that robust algorithms can be combined as black boxes: the candidate set comes from a robustified random projection structure, and distance comparison utilizes adaptive distance estimation. This approach could be transferred to nearest neighbor search, clustering, or extremum retrieval.
- The attack portion warns practitioners: as long as a model or user can continue querying based on system responses, the "high probability correctness" of traditional randomized indices may not be the guarantee actually required at deployment.

## Limitations & Future Work
- The results are primarily asymptotic theory; hidden polylog, constants, and space terms may be large. Practical high-dimensional retrieval systems require engineering implementation and benchmarking.
- The algorithm relies on Euclidean space and Gaussian random projections; it is unclear if it generalizes directly to cosine distance, inner product search, non-Euclidean embeddings, or learned indices.
- Furthest neighbor is a specific extremum problem. Although the authors provide a transferable robustification recipe, more complex search problems like nearest neighbor, top-k, or diverse subset selection still require re-proving candidate identity stability.
- The attack proof targets Indyk-style oblivious AFN, showing classical algorithms are not robust; however, practical ANN/FN systems often use multi-layer heuristics, and systematic evaluation of adaptive attacks remains an open direction.

## Related Work & Insights
- **vs Indyk 2003**: Indyk uses random projections for oblivious approximate furthest neighbor with excellent query complexity; this paper retains the candidate generation idea but adds smoothness, covering, and multiple copies to resist adaptive queries.
- **vs Cherapanamjeri & Nelson 2020**: They provide adaptive distance estimation, which can robustly estimate distances but leads to near-linear scans for furthest neighbor; this paper applies ADE to a smaller candidate set to achieve sublinear search.
- **vs adaptive nearest neighbor work**: Nearest neighbor has several results in the adaptive setting but often requires large space or only guarantees non-adaptive query time; this paper shows furthest neighbor can achieve strong robust guarantees via scale-free covering.
- **vs differential privacy-inspired robustification**: DP-style methods typically maintain stability for a finite number of queries and require periodic reconstruction; this paper proves simultaneous validity for all queries, making it independent of query count and resistant to white-box leakage.

## Rating
- Novelty: ⭐⭐⭐⭐ Provides sublinear AFN in the adaptive query model and includes attacks on oblivious algorithms; the theoretical approach is clear.
- Experimental Thoroughness: ⭐⭐⭐ Theoretical proofs are complete, but lack systematic implementation or empirical evaluation; practical constants for application scenarios remain unknown.
- Writing Quality: ⭐⭐⭐⭐ The technical path from base structures to robustification and attack is coherent, though proof details are dense and may be challenging for non-theory readers.
- Value: ⭐⭐⭐⭐ Highly valuable for understanding the robustness of high-dimensional geometric data structures and interactive ML systems, particularly in highlighting the importance of query adaptivity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Distribution-Aligned Multimodal Fusion for Robust Object Detection](../../CVPR2026/object_detection/distribution-aligned_multimodal_fusion_for_robust_object_detection.md)
- [\[CVPR 2026\] RHCNet: Residual-Guided Hierarchical Calibration Network for Robust Underwater Object Detection](../../CVPR2026/object_detection/rhcnet_residual-guided_hierarchical_calibration_network_for_robust_underwater_ob.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](../../NeurIPS2025/object_detection/semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[NeurIPS 2025\] Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](../../NeurIPS2025/object_detection/scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)
- [\[ICML 2025\] Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](../../ICML2025/object_detection/causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)

</div>

<!-- RELATED:END -->
