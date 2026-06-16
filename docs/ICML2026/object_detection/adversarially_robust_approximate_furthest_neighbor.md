---
title: >-
  [Paper Note] Adversarially Robust Approximate Furthest Neighbor
description: >-
  [ICML 2026][Object Detection][Paper Note] This theoretical paper provides the first approximate furthest neighbor (AFN) data structure resistant to adaptive query adversaries. While maintaining query complexity dependencies on $n$ similar to Indyk's classical oblivious algorithm, the authors prove that traditional random projection AFN algorithms can be compro
tags:
  - ICML 2026
  - Object Detection
date: 2026-05-08
content_hash: 57a1597091a2fef9
---
# Adversarially Robust Approximate Furthest Neighbor

**Conference**: ICML 2026  
**arXiv**: [2605.16618](https://arxiv.org/abs/2605.16618)  
**Code**: No public code  
**Area**: Optimization / Theoretical Algorithms  
**Keywords**: Approximate Furthest Neighbor, Adaptive Query, Adversarially Robust Data Structures, Random Projection, High-dimensional Geometry

## TL;DR
This theoretical paper provides the first approximate furthest neighbor (AFN) data structure resistant to adaptive query adversaries. While maintaining query complexity dependencies on $n$ similar to Indyk's classical oblivious algorithm, the authors prove that traditional random projection AFN algorithms can be compromised by adaptive queries.

## Background & Motivation
**Background**: Nearest neighbor search and furthest neighbor search are fundamental geometric primitives in high-dimensional data analysis. Although furthest neighbor is discussed less frequently than nearest neighbor, it arises naturally in diversity maximization, anomaly detection, hard negative mining, adversarial example generation, reinforcement learning exploration, and clustering.

**Limitations of Prior Work**: Classical randomized data structures typically assume that all queries are fixed before the randomness of the data structure is determined, known as oblivious queries. Modern machine learning pipelines more commonly involve interactive or closed-loop scenarios: after the algorithm returns a point, the next query is adjusted based on previous answers. Such adaptive queries can leak the random "blind spots" of the data structure, rendering classical Monte Carlo guarantees invalid.

**Key Challenge**: Furthest neighbor is a search problem that returns specific points rather than simple distance values. Distance estimation can be robustified using covering balls and stable estimators, but the identity of furthest neighbor candidates can change abruptly with query positions, making it difficult to directly apply existing adaptive distance estimation frameworks.

**Goal**: The authors aim to answer a fundamental question: under a fully adaptive query model, can approximate furthest neighbor still achieve sublinear query time relative to the data size $n$? Simultaneously, they seek to explain why directly using Indyk's oblivious random projection algorithm is insufficiently robust.

**Key Insight**: The paper "opens" the classical random projection algorithm into an analyzable white-box: first strengthening the smooth success guarantee of a single query, then using query space covering and a union bound to create a guarantee that holds for all queries simultaneously. Finally, only a few base data structures are sampled, and candidates are screened using robust distance estimation.

**Core Idea**: Cover the entire query space with multiple independent random projection data structures such that any adaptive query is a "good query" for at least half of the structures. Then, use a small amount of random sampling and robust distance estimation to select the approximate furthest neighbor from the candidate set.

## Method
This paper does not feature a conventional experimental system; its core contributions are algorithmic construction, complexity proofs, and attacks on oblivious algorithms. The method can be understood as upgrading Indyk’s random projection AFN algorithm from "high probability success for a fixed query" to "simultaneous success for all possible adaptive queries."

### Overall Architecture
Given a point set $P\subset\mathbb{R}^d$ and an approximation factor $c>1$, the algorithm builds $k$ independent base data structures during preprocessing. Each base structure consists of $N\approx \tilde{\Theta}(n^{1/c^2})$ Gaussian random projections and stores several top candidate points for each projection direction. During query time, the algorithm randomly samples $m=\Theta(\log n)$ base structures, collects their sets of returned candidate furthest points, and uses an adaptively robust distance estimation structure to estimate distances from the candidates to the query point, returning the candidate with the largest estimated distance.

### Key Designs

**1. Good query definition with slack: Allowing success properties to "propagate" to nearby queries**

Adaptive adversaries prefer placing the next query near the boundaries of the current structure's random blind spots. Thus, proving "this fixed query can find a far point" is insufficient. The authors strengthen the success condition to a version with slack: if the true furthest neighbor $p^*$ of $q$ is sufficiently prominent in a certain projection direction, and the relative number of outlier projections for incorrect candidates does not exceed $8N$, $q$ is called $(c,\delta)$-good for that projection matrix. They prove that using $N=\tilde{\Theta}(n^{1/c^2})$ Gaussian projections, a fixed query satisfies this property with at least $3/4$ probability. The significance of slack is: as long as $q'$ is within $\Delta/n^3$ distance of $q$, the good property of $q$ transfers to $q'$—this transforms discrete "fixed query success events" into locally stable events that can be covered by a grid.

**2. Query space covering + Multi-replica union bound: Scaling from "arbitrary fixed query" to "all queries simultaneously"**

Since adaptive query sequences can be infinitely long, one cannot rely on "small failure probability per query" followed by a union bound over the number of queries—one must prove correctness for the entire continuous query space at once. The authors first demonstrate that queries sufficiently far from the center of the point set can be solved approximately with a trivial answer, leaving only a bounded sphere to cover. They construct a grid on this sphere, select $k=\tilde{\Theta}(d)$ independent base structures, and use Chernoff + union bounds to prove that every grid point is "good" for at least $k/2$ structures. Utilizing the aforementioned smoothness, they generalize the conclusion from grid points to any query within the sphere, ensuring any adaptive query is "good" for at least half the structures. This is the key to resisting infinitely long adaptive sequences.

**3. Sparse candidate sampling + Robust distance estimation: Avoiding full traversal while reducing dimensionality dependence**

Since any query is "good" for at least half of the base structures, the query process does not need to touch all $k$ structures. Sampling $m=\Theta(\log n)$ structures is sufficient to hit at least one "good" structure with high probability; the fresh randomness of the samples is not corrupted by past queries. The size of the collected candidate set is approximately $\tilde{O}(n^{1/c^2})$ or $\tilde{O}(n^{2/c^2})$. Direct distance calculation results in a $\tilde{O}(d n^{1/c^2})$ query time. Alternatively, one can treat the Cherapanamjeri-Nelson robust distance estimation as a black box applied to the candidate subset, changing the query time to $\tilde{O}(\min\{n^{2/c^2},n\}+d)$, with the trade-off of the approximation factor degrading to $(1+\epsilon)c$. Embedding robust distance estimation into the search algorithm is a crucial compositional technique—candidate generation and distance comparison are robustified separately, rather than attempting a one-time global stability proof for the returned point.

### Loss & Training
As a theoretical algorithm paper, there is no training loss. Preprocessing complexity is $\tilde{O}(d^2 n^{1+1/c})$. One version returns a $c$-approximate AFN with query time $\tilde{O}(d n^{1/c^2})$; another version returns a $(1+\epsilon)c$-approximate AFN with query time $\tilde{O}(\min\{n^{2/c^2},n\}+d)$. Space complexities include $\tilde{O}(d\cdot\min\{n,d n^{2/c^2}\})$ or an additional $\tilde{O}(d^2)$ term respectively.

## Key Experimental Results

### Main Results
The "main experiments" correspond to the core theoretical results and complexity comparisons rather than empirical benchmarks.

| Method / Result | Query Model | Approx. Factor | Query Time | Space | Notes |
|-----------------|--------------|----------------|------------|-------|-------|
| Indyk 2003 AFN | Oblivious | $\approx c$ | $\tilde{O}(d n^{1/c^2})$ | No adaptive guarantee | Efficient for fixed queries, but compromised by adaptive attacks |
| Cherapanamjeri-Nelson ADE + Scan | Adaptive | $c$ | $\tilde{O}(n+d)$ | Robust | Handles adaptivity but approaches linear scan |
| Ours (Version 1) | Adaptive / White-box | $c$ | $\tilde{O}(d n^{1/c^2})$ | $\tilde{O}(d\min\{n,d n^{2/c^2}\})$ | Matches $n$-dependency of oblivious algorithm |
| Ours (Version 2) | Adaptive / White-box | $(1+\epsilon)c$ | $\tilde{O}(\min\{n^{2/c^2},n\}+d)$ | $\tilde{O}(d^2+d\min\{n,d n^{2/c^2}\})$ | Uses robust distance estimation to reduce explicit $d$ multiplier |

### Ablation Study
The ablation here corresponds to the analysis of algorithmic components: how theoretical guarantees degrade or fail without specific parts.

| Component / Variant | Function | Effect if Missing |
|---------------------|----------|-------------------|
| Single Indyk-style structure | Sublinear candidate generation for fixed queries | Only oblivious guarantees; adversary can craft queries based on projection directions |
| $(c,\delta)$-good + nearby inheritance | Makes success property stable under small perturbations | Cannot generalize from grid points to continuous query space |
| $k$ independent structures + covering union bound | Ensure correctness for all queries simultaneously | Failure probability can only be controlled by query count; unfit for infinite adaptive sequences |
| Random sampling of $m=\Theta(\log n)$ structures | Avoids querying all $k$ structures | Large query overhead; however, if sampling is too few, may miss "good" structures |
| Robust distance estimation for screening | Safely compares distances within candidate set | Direct calculation keeps $d n^{1/c^2}$ dependence; ordinary JL is not necessarily robust to adaptive queries |

### Key Findings
- Sublinear adaptive AFN is feasible: When $d=\mathrm{poly}(\log n)$, Version 1 is sublinear in $n$ for any $c>1$; when $c>\sqrt{2}$ and $d=o(n)$, Version 2 also remains sublinear.
- The guarantees are stronger than black-box differential privacy-style reconstructions: The authors prove the algorithm holds even under a white-box adversary, meaning that even if the adversary sees internal randomness, past information leakage does not destroy the high-probability event of "correctness for all queries."
- Attack results demonstrate that classical oblivious guarantees cannot be simply moved to interactive ML pipelines. The authors construct a dataset with duplicate copies of points where a query directed based on random projections causes the algorithm to return a point at distance $d^{0.01}$, while the true furthest distance is at least $d^{0.5}$.

## Highlights & Insights
- The most enlightening aspect is the decomposition of search problem robustification into "robust candidate generation" and "robust candidate distance comparison," rather than attempting a one-time global stability proof for the returned point.
- The $(c,\delta)$-good query definition with slack is critical. It transforms randomized projection success from discrete fixed-query events into locally stable events coverable by a grid—the core bridge for handling continuous query spaces.
- The paper shows that robust algorithms can be combined as black boxes: candidate sets come from a robustified random projection structure, and distance comparison invokes adaptive distance estimation. This compositional approach may transfer to nearest neighbor, clustering, or extreme value retrieval.
- The attack section reminds practitioners: as long as a model or user can continue querying based on system responses, the "high probability correctness" of traditional randomized indices may not be the guarantee actually required at deployment.

## Limitations & Future Work
- The results are primarily asymptotic theory; hidden polylog, constants, and space terms might be large. Whether practical high-dimensional retrieval systems should adopt this requires engineering implementation and benchmarks.
- The algorithm relies on Euclidean space and Gaussian random projections; it is unclear if it can be directly generalized to cosine distance, inner product search, non-Euclidean embeddings, or learned indices.
- Furthest neighbor is a specific extreme-value problem. While the authors provide a transferable robustification recipe, more complex search problems like nearest neighbor, top-k, or diverse subset selection still require re-proving candidate identity stability.
- The attack proof targets Indyk-style oblivious AFN, showing classical algorithms are not robust; however, practical ANN/FN systems often use multi-layer heuristics, and systematically evaluating adaptive attacks remains an open direction.

## Related Work & Insights
- **vs Indyk 2003**: Indyk uses random projection for oblivious approximate furthest neighbor with excellent query complexity. This paper retains the candidate generation idea but adds smooth good queries, covering, and multiple replicas to resist adaptive queries.
- **vs Cherapanamjeri & Nelson 2020**: They provide adaptive distance estimation (ADE) for robust distance estimation, which would approach linear scan if used directly for furthest neighbor; this paper applies ADE to a smaller candidate set to achieve sublinear search.
- **vs Adaptive Nearest Neighbor Work**: Several results exist for nearest neighbor in adaptive settings, but they often require large space or only guarantee non-adaptive query times; this paper shows furthest neighbor can achieve strong robust guarantees via scale-free covering.
- **vs Differential Privacy-inspired Robustification**: DP-style methods usually maintain stability for a finite number of queries and require periodic reconstruction; this paper proves simultaneous correctness for all queries, thus not depending on query count or fearing white-box information leakage.

## Rating
- Novelty: ⭐⭐⭐⭐ Provides sublinear AFN under the adaptive query model with an attack on oblivious algorithms; theoretical framing is very clear.
- Experimental Thoroughness: ⭐⭐⭐ Complete theoretical proofs, but lacks systematic implementation or empirical evaluation; actual constants for application scenarios remain unknown.
- Writing Quality: ⭐⭐⭐⭐ The technical path from base structures to robustification and then to the attack is coherent, though proof details are dense and potentially difficult for non-theory readers.
- Value: ⭐⭐⭐⭐ Highly relevant for robustness in high-dimensional geometric data structures and interactive ML systems, particularly in highlighting the importance of query adaptivity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Distribution-Aligned Multimodal Fusion for Robust Object Detection](../../CVPR2026/object_detection/distribution-aligned_multimodal_fusion_for_robust_object_detection.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](../../NeurIPS2025/object_detection/semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[NeurIPS 2025\] Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](../../NeurIPS2025/object_detection/scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)
- [\[ICML 2025\] Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](../../ICML2025/object_detection/causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)
- [\[CVPR 2025\] Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](../../CVPR2025/object_detection/generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)

</div>

<!-- RELATED:END -->
