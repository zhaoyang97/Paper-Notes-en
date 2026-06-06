---
title: >-
  [Paper Note] Estimating Correlation Clustering Cost in Node-Arrival Stream
description: >-
  [ICML 2026][Correlation Clustering] This paper studies the problem of approximating the cost of correlation clustering in the "node-arrival" data stream model. The authors propose the C4Approx algorithm…
tags:
  - "ICML 2026"
  - "Correlation Clustering"
  - "Node-Arrival Stream"
  - "Sublinear Space"
  - "Pivot Algorithm"
  - "Reference Set Sampling"
date: 2026-05-08
content_hash: 6e1b3b231fec2102
---

# Estimating Correlation Clustering Cost in Node-Arrival Stream

**Conference**: ICML 2026  
**arXiv**: [2605.07091](https://arxiv.org/abs/2605.07091)  
**Code**: None  
**Area**: Algorithm Theory / Data Stream Algorithms / Graph Clustering  
**Keywords**: Correlation Clustering, Node-Arrival Stream, Sublinear Space, Pivot Algorithm, Reference Set Sampling

## TL;DR
This paper studies the problem of approximating the cost of correlation clustering in the "node-arrival" data stream model. The authors propose the C4Approx algorithm, which achieves a $(O(1), n^{1-\alpha})$-approximation using $O(n^{(3+\alpha)/4}\log n)$ words of **sublinear** space and a constant number of passes. Two matching lower bounds are provided, showing that both multi-pass and additive error are unavoidable. On real data, storing only 2% of nodes suffices to match the performance of Pivot.

## Background & Motivation

**Background**: Correlation clustering is a classic NP/APX-hard problem. Given a complete graph with $\pm 1$ edge labels, the goal is to partition nodes into clusters to minimize the number of "positive edges across clusters + negative edges within clusters" (mismatches). Many $O(1)$-approximation algorithms exist (with the Pivot algorithm being a representative 3-approximation). In large-scale data scenarios, many edge-arrival streaming algorithms exist, but all require $O(n\,\text{polylog}\,n)$ space.

**Limitations of Prior Work**: Real-world data (images, tweets, vectors) naturally arrives as a "node stream," with edge labels computed on demand via similarity functions—no one explicitly stores all $\binom{n}{2}$ edges. In the node-arrival model, prior work is almost nonexistent; the only related work by Assadi et al. provides a $(O(1),\delta n^2)$-approximation, but the additive term $\delta n^2$ is too loose.

**Key Challenge**: Outputting the clustering requires $\Omega(n)$ space (since the number of clusters can be up to $n$), but if only the "clusterability" (i.e., OPT cost) is needed, it may be possible to break the $n$ barrier. However, in the node-arrival model, edges can only be queried when both endpoints are in memory, making even enumerating all edges impossible—a fundamental access limitation.

**Goal**: Achieve a $(O(1), n^{1-\alpha})$-approximation of the cost using $o(n)$ space and $O(1)$ passes; also, characterize the necessity of both "multi-pass" and "additive error" in this model.

**Key Insight**: The core observation is that it is unnecessary to find a pivot for every node. By maintaining a reference set $R$ consisting of a small number of nodes with the highest ranks in a random permutation $\pi$, most nodes (those with high degree or whose pivot falls in $R$) can have their pivot determined directly; the remaining few (necessarily low-degree) nodes can be estimated separately via sampling.

**Core Idea**: The combination of "reference set $R$ + high/low degree decomposition" allows the total number of PrunedPivot mismatches $|E^{\text{mis}}|$ to be split into two independently estimable parts, reducing space from $O(n)$ to sublinear.

## Method

### Overall Architecture
C4Approx is implemented as a 5-step pipeline.

First pass: Based on a random permutation $\pi$, the top $r=48k n^{1-\beta}\log n$ nodes are stored in the reference set $R$ ($\beta=(1-\alpha)/4$).

Then, two subroutines are executed in parallel: (i) Est-EA uses 3 passes to estimate $|E_A^{\text{mis}}|$ (mismatches with at least one endpoint in $A$), (ii) Est-EB uses $k+3$ passes to estimate $|E_B^{\text{mis}}|$ (mismatches with both endpoints in $B$). Here, $A$ is the set of nodes whose pivot can be determined directly via $R$, and $B=V\setminus A$ is the set of remaining low-degree nodes.

The final output is $(\tilde m_A+\tilde m_B + \frac{3}{8}\epsilon n^{1-\alpha})/(1-\epsilon/8)$, combined with Theorem 2.1 (Dalirrooyfard et al.)'s PrunedPivot $(9+\frac{24}{k-1})$-approximation, yielding an $(O(1),n^{1-\alpha})$-approximation to the OPT cost with probability at least $0.99$.

### Key Designs

1. **Reference Set $R$ + FindPivot Node Partitioning**:

    - **Function**: Uses a sublinear-sized $R$ as a partial oracle for pivot determination, partitioning nodes into $A$ (pivot can be determined) and $B$ (cannot be determined, but must be low-degree).
    - **Mechanism**: FindPivot recursively searches for higher-ranked neighbors within $R$ with a recursion budget of $k$; if successful, $\text{pivot}(u)\in R$ or $u$ is a singleton, so $u$ is assigned to $A$. If the search times out and all neighbors are outside $R$, $u$ is assigned to $B$. Lemma 2.5 ensures that all nodes in a cluster are either all in $A$ or all in $B$, allowing independent estimation on both sides.
    - **Design Motivation**: Storing all pivot information requires $\Omega(n)$ space; storing $|R|=\tilde O(n^{1-\beta})$ nodes suffices to determine pivots for all high-degree nodes (with high probability, the top $k$ high-ranked neighbors of a high-degree node fall into $R$, as shown by Lemma 2.6 via Chernoff bounds). The remaining $B$ nodes must be low-degree ($\le n^\beta$) and can be handled via sampling.

2. **High/Low Degree Decomposition for $E_A^{\text{mis}}$ Estimation**:

    - **Function**: Estimates the number of mismatches with at least one endpoint in $A$.
    - **Mechanism**: For each node $u$, $|E_A^{\text{mis}}|$ is equivalent to estimating the average degree in the mismatch subgraph $G_A^{\text{mis}}$. However, the degree range is $\{0,\dots,n-1\}$, so uniform sampling leads to high variance. The authors sample a small set $S_1$ as a "high-degree certificate," partitioning $V$ into a high-degree set $H$ (nodes with significant $|N_A^{\text{mis}}(u)\cap S_1|$) and a low-degree set $L$; $H$ is estimated via rescale-by-sampling, $L$ via direct subsampling. Lemma 2.8: $(1\pm\epsilon,\pm\epsilon n^{1-\alpha})$-approximation, $O(\frac{1}{\epsilon^2}(n^{1-\beta}+n^{\alpha+\beta})\log n)$ space, 3 passes.
    - **Design Motivation**: Separating high and low degrees addresses heavy/long tails, a classic variance control technique; in the node stream, sampling strategies must be redesigned to accommodate the restriction that edges can only be queried when both endpoints are in memory.

3. **Whole-Cluster Sampling for $E_B^{\text{mis}}$ Estimation**:

    - **Function**: Estimates the number of mismatches with both endpoints in $B$.
    - **Mechanism**: All clusters in $B$ are "small" (degree bounded by $n^\beta$), so clusters can be sampled directly from the set $\mathcal{C}(B)$, loaded entirely into memory, and their intra/inter-cluster edge counts are measured and rescaled. Pivot computation is performed via Algorithm 2, a streaming implementation of PrunedPivot, using $k$ passes and $O(k)$ space. Lemma 2.9: same approximation and confidence, $O(\frac{k}{\epsilon^2}n^{\alpha+3\beta}\log n)$ space, $k+3$ passes.
    - **Design Motivation**: In $B$, $R$ cannot be used for mismatch determination, but the "small cluster" property provides a bound—each sampled cluster's contribution is controlled by the cluster size, naturally reducing variance. Both parts together fully cover $E^{\text{mis}}$.

### Loss & Training
This is a purely combinatorial algorithm with no training. Key parameters: $k=37$, $\epsilon=1/10$, $\beta=(1-\alpha)/4$ yield the main theorem: $(O(1),n^{1-\alpha})$-approximation with $O(n^{(3+\alpha)/4}\log n)$ space.

## Key Experimental Results

### Main Results
The authors compare C4Approx with Pivot, PrunedPivot, and Assadi et al.'s algorithm on various real datasets. Representative results (see Section 4 of the original paper for detailed numbers):

| Dataset / Setting | Memory Fraction | C4Approx cost | Pivot cost | Notes |
|---|---|---|---|---|
| ImageNet-21K embedding + cosine threshold | 2% nodes | Matches Pivot | 100% nodes stored | 100× memory compression with no loss in accuracy |
| Sparse graph (clusters with uneven heights) | 2% | Significantly better than Assadi et al. | — | Sparse graphs are a weakness for Assadi's algorithm |
| Multiple runs averaged | 2% | Low variance | — | High/low degree decomposition suppresses variance |

### Ablation Study

| Configuration | Performance | Description |
|---|---|---|
| C4Approx (full) | Close to Pivot | Both high/low degree decomposition and whole-cluster sampling enabled |
| SimpleSampling only | Additive error $\Theta(n^2/\sqrt q)$ | Confirms the theoretical result that naive sampling cannot reduce additive error below $o(n^{1.5})$ in $o(n)$ space |
| Without high/low degree decomposition | Variance explodes | Confirms the critical role of variance reduction |
| Assadi et al. on sparse graphs | Unstable | Additive $\delta n^2$ cannot be made small simultaneously |

### Key Findings
- In the node stream model, additive error is **inevitable** (lower bound (2): $c$-approximation with $d=0$ requires $\Omega(n)$ bits); multi-pass is also **provably necessary** (lower bound (1): one-pass $(c,d)$-approximation requires $\Omega(n)$ bits). This provides a clear characterization of the model's inherent difficulty.
- Storing only $\sim 2\%$ of nodes suffices to approach Pivot—empirical results show that "sublinear memory + a few passes" is not an empty promise in the node stream model.
- Compared to Assadi et al.: To reduce their additive $\delta n^2$ to $n^{0.1}$ requires $\delta = n^{-1.9}$, implying space $\Omega(n^{9.5})$—completely infeasible.

## Highlights & Insights
- **Model Innovation**: Node-arrival, rather than edge-arrival, is a more realistic perspective for large-scale data streams; this line has been severely underestimated, and the authors formalize it and provide the first algorithm with matching lower bounds.
- **Transferable "Reference Set + High/Low Degree Decomposition" Paradigm**: This approach may be directly applicable to other streaming graph problems requiring "on-demand edge queries" (e.g., triangle counting, community detection, cut sparsifier).
- **Theory-Experiment Closure**: Upper and lower bounds are paired, and experiments confirm that theoretically chosen constants (e.g., $k=37$) are practical—a rare "proven and runnable" algorithmic paper.

## Limitations & Future Work
- The $(O(1), n^{1-\alpha})$ additive error ceiling has little impact for **dense ground truth** ($|E^{\text{mis}}|\gg n^{1-\alpha}$), but in low-cost scenarios ("nearly perfectly clusterable"), the additive term may overwhelm the true value.
- The algorithm relies on a one-time random permutation $\pi$; if the stream is adversarial (e.g., malicious nodes arrive first), the i.i.d. assumption is broken, and robustness remains for future work.
- Experiments are mainly conducted on "embedding + threshold" synthetic similarity graphs; scenarios where the similarity oracle itself is expensive (e.g., LLM judge) have not yet been evaluated.

## Related Work & Insights
- **vs Pivot / PrunedPivot**: This work inherits their $O(1)$-approximation guarantee, with the key contribution being the adaptation to the **sublinear memory + node stream** setting under strong constraints.
- **vs Assadi et al. 2023 (edge stream)**: Same output form, but the additive term $n^{1-\alpha}$ here is much tighter than their $\delta n^2$; the lower bounds also provide a more detailed characterization of the model.
- **vs Dynamic Algorithms (insertion/deletion streams)**: That line focuses on update time and still requires $\Omega(n)$ space; this work is complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic sublinear algorithm for node-arrival correlation clustering, with matching lower bounds, nearly foundational
- Experimental Thoroughness: ⭐⭐⭐ Real data used, but limited to similarity graphs generated by embedding thresholds, relatively single-modal
- Writing Quality: ⭐⭐⭐⭐ Theoretical exposition is rigorous, with clear definition–lemma–theorem structure
- Value: ⭐⭐⭐⭐ Directly applicable to "clusterability measurement" in large-scale similarity graphs, and the paradigm is transferable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/others/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/others/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[AAAI 2026\] Cost-Free Neutrality for the River Method](../../AAAI2026/others/cost-free_neutrality_for_the_river_method.md)
- [\[AAAI 2026\] Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning](../../AAAI2026/others/bilevel_mcts_for_amortized_o1_node_selection_in_classical_planning.md)
- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](../../ICLR2026/others/distributed_algorithms_for_euclidean_clustering.md)

</div>

<!-- RELATED:END -->
