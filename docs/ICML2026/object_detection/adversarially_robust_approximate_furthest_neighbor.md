---
title: >-
  [Paper Note] Adversarially Robust Approximate Furthest Neighbor
description: >-
  [ICML 2026][Object Detection][Approximate Furthest Neighbor] This theoretical paper provides the first approximate furthest neighbor data structure resistant to adaptive query adversaries. While maintaining a query compl…
tags:
  - "ICML 2026"
  - "Object Detection"
  - "Approximate Furthest Neighbor"
  - "Adaptive Query"
  - "Adversarially Robust Data Structure"
  - "Random Projection"
  - "High-dimensional Geometry"
date: 2026-05-08
content_hash: 5a2ce2f637fdef19
---

# Adversarially Robust Approximate Furthest Neighbor

**Conference**: ICML 2026  
**arXiv**: [2605.16618](https://arxiv.org/abs/2605.16618)  
**Code**: No public code  
**Area**: Optimization / Theoretical Algorithms  
**Keywords**: Approximate Furthest Neighbor, Adaptive Query, Adversarially Robust Data Structure, Random Projection, High-dimensional Geometry  

## TL;DR
This theoretical paper provides the first approximate furthest neighbor data structure resistant to adaptive query adversaries. While maintaining a query complexity with $n$ dependence similar to Indyk's classic oblivious algorithm, it demonstrates that traditional random projection furthest neighbor algorithms can be compromised by adaptive queries.

## Background & Motivation
**Background**: Nearest neighbor search and furthest neighbor search are fundamental geometric primitives in high-dimensional data analysis. Although furthest neighbor is discussed less frequently than nearest neighbor, it arises naturally in diversity maximization, anomaly detection, hard negative mining, adversarial sample generation, reinforcement learning exploration, and clustering.

**Limitations of Prior Work**: Classic randomized data structures typically assume all queries are fixed before the randomness of the data structure is determined, known as oblivious queries. Modern machine learning pipelines more commonly involve interactive or closed-loop scenarios: after the algorithm returns a point, the next query is adjusted based on previous answers. Such adaptive queries can leak the random blind spots of the data structure, rendering classic Monte Carlo guarantees invalid.

**Key Challenge**: The furthest neighbor is a search problem that returns specific points rather than simple distance values. Distance estimation can be made robust using covering balls and stable estimators, but the identity of furthest neighbor candidates can change abruptly with the query position, making it impossible to directly apply existing adaptive distance estimation frameworks.

**Goal**: The authors aim to answer a fundamental question: Can approximate furthest neighbor search still achieve sublinear query time relative to data size $n$ under a fully adaptive query model? Simultaneously, they seek to explain why directly using Indyk's oblivious random projection algorithm is not robust.

**Key Insight**: The paper "opens" the classic random projection algorithm into an analyzable white box: it first strengthens the smooth success guarantee for a single query, then uses query space covering and a union bound to transform it into a guarantee that holds for all queries simultaneously. Finally, it samples only a few base data structures and uses robust distance estimation to filter candidates.

**Core Idea**: Use multiple independent random projection data structures to cover the entire query space, ensuring that any adaptive query is a good query for at least half of the structures. Then, use a small amount of random sampling and robust distance estimation to select the approximate furthest neighbor from the candidate set.

## Method
This paper does not follow a conventional experimental system; its core contributions are algorithmic construction, complexity proofs, and attacks on oblivious algorithms. The method can be understood as upgrading Indyk's random projection furthest neighbor algorithm from "high probability success for fixed queries" to "simultaneous success for all possible adaptive queries."

### Overall Architecture
Given a point set $P\subset\mathbb{R}^d$ and an approximation factor $c>1$, the algorithm builds $k$ independent base data structures during preprocessing. Each base structure consists of $N\approx \tilde{\Theta}(n^{1/c^2})$ Gaussian random projections and stores several top candidates for each projection direction. During query time, the algorithm randomly samples $m=\Theta(\log n)$ base structures, collects their returned sets of candidate furthest points, and uses an adaptively robust distance estimation structure to estimate distances from the candidates to the query point, returning the estimated furthest candidate.

### Key Designs
1. **Definition of good query with slack**:

    - **Function**: Strengthens the success condition of a single random projection structure from "this query can find a far point" to "nearby queries also inherit success."
    - **Mechanism**: If the true furthest neighbor $p^*$ of query $q$ is sufficiently prominent in a certain projection direction, and the number of outlier projections for incorrect candidates does not exceed $8N$, $q$ is called $(c,\delta)$-good for that projection matrix. The authors prove that using $N=\tilde{\Theta}(n^{1/c^2})$ Gaussian projections, a fixed query satisfies this property with at least $3/4$ probability.
    - **Design Motivation**: An adaptive adversary might place the next query near the boundary of the current structure. The definition with slack ensures that if $q'$ is close to $q$ by as little as $\Delta/n^3$, the good property of $q$ can be transferred to $q'$, laying the foundation for subsequent space covering.

2. **Query space covering + Union bound over multiple copies**:

    - **Function**: Boosts "high probability success for any fixed query" to "simultaneous success for all possible queries."
    - **Mechanism**: It is first proved that queries sufficiently far from the center of the point set can be solved approximately by a trivial answer; only a bounded ball needs to be covered. The authors establish a grid on this ball, choose $k=\tilde{\Theta}(d)$ independent base structures, and use Chernoff + union bounds to prove that each grid point is good for at least $k/2$ structures. Via smoothness, any query within the ball is also good for at least $k/2$ structures.
    - **Design Motivation**: The adaptive query sequence can be infinitely long; thus, one cannot rely on the probability of failure per query being small and applying a union bound over the number of queries. A guarantee must be proven for the entire continuous query space at once.

3. **Few-sample candidates + Robust distance estimation**:

    - **Function**: Avoids querying all $k$ structures every time while reducing dimensionality dependence in distance calculations.
    - **Mechanism**: Since any query is good for at least half of the base structures, randomly sampling $m=\Theta(\log n)$ structures provides a high probability of hitting at least one good structure. The candidate set size is approximately $\tilde{O}(n^{1/c^2})$ or $\tilde{O}(n^{2/c^2})$, allowing for direct distance calculation with $\tilde{O}(d n^{1/c^2})$ query time. Alternatively, one can use the Cherapanamjeri-Nelson robust distance estimation on the candidate subset to change the query time to $\tilde{O}(\min\{n^{2/c^2},n\}+d)$, at the cost of the approximation factor becoming $(1+\epsilon)c$.
    - **Design Motivation**: The fresh randomness of random sampling is not contaminated by past queries; embedding robust distance estimation as a black box into the search algorithm is a key compositional technique in the paper.

### Loss & Training
As a theoretical algorithm paper, there is no training loss. Preprocessing complexity is $\tilde{O}(d^2 n^{1+1/c})$. One version returns a $c$-approximate AFN with $\tilde{O}(d n^{1/c^2})$ query time; another returns a $(1+\epsilon)c$-approximate AFN with $\tilde{O}(\min\{n^{2/c^2},n\}+d)$ query time. Space complexity includes $\tilde{O}(d\cdot\min\{n,d n^{2/c^2}\})$ or an additional $\tilde{O}(d^2)$ term.

## Key Experimental Results

### Main Results
The "main experiments" in this paper correspond to the primary theoretical results and complexity comparisons rather than empirical benchmarks.

| Method / Result | Query Model | Approximation Factor | Query Time | Space | Remarks |
|-------------|----------|----------|----------|------|------|
| Indyk 2003 AFN | oblivious | approx. $c$ | $\tilde{O}(d n^{1/c^2})$ | No adaptive guarantee | Efficient for fixed queries, but compromised by this paper's adaptive attack |
| Cherapanamjeri-Nelson ADE + scan | adaptive | $c$ | $\tilde{O}(n+d)$ | Robust | Can handle adaptivity, but essentially close to linear scan |
| Ours Version 1 | adaptive / white-box | $c$ | $\tilde{O}(d n^{1/c^2})$ | $\tilde{O}(d\min\{n,d n^{2/c^2}\})$ | Matches $n$-dependence of the oblivious algorithm |
| Ours Version 2 | adaptive / white-box | $(1+\epsilon)c$ | $\tilde{O}(\min\{n^{2/c^2},n\}+d)$ | $\tilde{O}(d^2+d\min\{n,d n^{2/c^2}\})$ | Uses robust distance estimation to reduce explicit $d$ multiplier |

### Ablation Study
Ablation here corresponds to the analysis of algorithmic components: what happens if a component is missing.

| Component / Variant | Function | If Missing |
|-------------|------|----------------|
| Single Indyk-style projection structure | Provides sublinear candidate generation for fixed queries | Only guarantees oblivious queries; adaptive adversaries can construct failing queries based on projection directions |
| $(c,\delta)$-good + nearby query inheritance | Ensures success property is stable under small perturbations | Fails to generalize from grid points to a continuous query space |
| $k$ independent structures and covering union bound | Ensures simultaneous success for all queries | Can only control failure probability by query count; unsuitable for infinite adaptive sequences |
| Randomly sampling $m=\Theta(\log n)$ structures | Avoids querying all $k$ structures | Query overhead increases; if sampling too few, a good structure might not be hit |
| Robust distance estimation to filter candidates | Safely compares distances within the candidate set | Direct distance calculation retains $d n^{1/c^2}$ dependence; ordinary JL is not necessarily robust to adaptive queries |

### Key Findings
- Sublinear adaptive AFN is feasible: When $d=\mathrm{poly}(\log n)$, Version 1 is sublinear in $n$ for any $c>1$; when $c>\sqrt{2}$ and $d=o(n)$, Version 2 also remains sublinear.
- The guarantee is stronger than black-box differential privacy-style data structure reconstruction: The authors prove the algorithm holds even under a white-box adversary, meaning that even if the adversary sees the internal randomness, past information leakage does not destroy the high-probability event that "all queries are simultaneously correct."
- Attack results show that classical oblivious guarantees cannot be simply moved to interactive ML pipelines. The authors construct a dataset consisting only of repeated copies of two points, but because the query direction depends on random projections, the algorithm returns a point at distance $d^{0.01}$ from the query, while the true furthest distance is at least $d^{0.5}$.

## Highlights & Insights
- The most insightful aspect is splitting the robustification of the search problem into "robust candidate generation" and "robust candidate distance comparison," rather than trying to prove global stability for the returned point at once.
- The definition of a good query with slack is critical. It transforms the random projection success event from a discrete fixed-query event into a locally stable event that can be covered by a grid, which is the core bridge for handling continuous query spaces.
- The paper demonstrates that robust algorithms can be combined as black boxes: the candidate set comes from a robustified random projection structure, and distance comparison then calls adaptive distance estimation. This compositional strategy could transition to nearest neighbor, clustering, or extreme value retrieval.
- The attack section serves as a reminder to practitioners: as long as a model or user can continue to ask questions based on system answers, the "high probability correctness" of traditional randomized indexing might not be the guarantee actually needed at deployment.

## Limitations & Future Work
- Results are primarily asymptotic theory; hidden polylog, constants, and space terms may be large. Whether real-world high-dimensional retrieval systems should adopt them requires engineering implementation and benchmarks.
- The algorithm relies on Euclidean space and Gaussian random projections; it remains unclear if it can be directly generalized to cosine distance, inner product retrieval, non-Euclidean embeddings, or learned indices.
- Furthest neighbor is a relatively specific extreme value problem. Although the authors provide a transferable robustification recipe, proving candidate identity stability for more complex search problems like nearest neighbor, top-k, or diverse subset selection still requires further work.
- The attack proof targets Indyk-style oblivious AFN, showing classic algorithms are not robust; however, practical ANN/FN systems often use multi-layer heuristics. Systematically evaluating adaptive attacks remains an open direction.

## Related Work & Insights
- **vs Indyk 2003**: Indyk uses random projections for oblivious approximate furthest neighbor with excellent query complexity; this paper retains its candidate generation idea but adds smooth good queries, covering, and multiple copies to make it resistant to adaptive queries.
- **vs Cherapanamjeri & Nelson 2020**: They provide adaptive distance estimation, which can robustly estimate distance but would approach linear scanning if applied to furthest neighbor directly; this paper uses ADE on a smaller candidate set to achieve sublinear search.
- **vs adaptive nearest neighbor work**: Nearest neighbor has several results in the adaptive setting but often with large space or guarantees only for non-adaptive query times; this paper shows furthest neighbor can also achieve strong robust guarantees via scale-free covering.
- **vs differential privacy-inspired robustification**: DP-style methods usually maintain stability for a limited number of queries and require periodic reconstruction; this paper directly proves simultaneous correctness for all queries, thus not depending on the number of queries or fearing white-box information leakage.

## Rating
- Novelty: ⭐⭐⭐⭐ Provides sublinear AFN under the adaptive query model and provides an attack on oblivious algorithms; the theoretical entry point is very clear.
- Experimental Thoroughness: ⭐⭐⭐ Theoretical proof is complete, but there is no systematic implementation or empirical evaluation; actual constants for application scenarios remain unknown.
- Writing Quality: ⭐⭐⭐⭐ The technical path from base structure to robustification and then to attack is coherent, but the proof details are dense and may not be friendly to non-theoretical readers.
- Value: ⭐⭐⭐⭐ Highly valuable for robust high-dimensional geometric data structures and interactive ML systems, particularly for highlighting the importance of query adaptivity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[ICML 2026\] Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection](mixture_prototype_flow_matching_for_open-set_supervised_anomaly_detection.md)
- [\[ICML 2026\] EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding](earl_towards_a_unified_analysis-guided_reinforcement_learning_framework_for_egoc.md)
- [\[ICML 2026\] FOCUS: Forcing In-Context Object Localization through Visual Support Constraints and Policy Optimization](focus_forcing_in-context_object_localization_through_visual_support_constraints_.md)
- [\[ICML 2026\] OmniVerifier-M1: Multimodal Meta-Verifier with Explicit Structured Recalibration](omniverifier-m1_multimodal_meta-verifier_with_explicit_structured_recalibration.md)

</div>

<!-- RELATED:END -->
