---
title: >-
  [Paper Note] Estimating Correlation Clustering Cost in Node-Arrival Stream
description: >-
  [ICML 2026][Algorithmic Theory / Data Stream Algorithms / Graph Clustering][Correlation Clustering] This paper investigates the approximate estimation of correlation clustering costs under the "node-arrival" streaming mo…
tags:
  - "ICML 2026"
  - "Algorithmic Theory / Data Stream Algorithms / Graph Clustering"
  - "Correlation Clustering"
  - "Node-Arrival Stream"
  - "Sublinear Space"
  - "Pivot Algorithm"
  - "Reference Set Sampling"
date: 2026-05-08
content_hash: 7879efa9c500250a
---

# Estimating Correlation Clustering Cost in Node-Arrival Stream

**Conference**: ICML 2026  
**arXiv**: [2605.07091](https://arxiv.org/abs/2605.07091)  
**Code**: None  
**Area**: Algorithmic Theory / Data Stream Algorithms / Graph Clustering  
**Keywords**: Correlation Clustering, Node-Arrival Stream, Sublinear Space, Pivot Algorithm, Reference Set Sampling

## TL;DR
This paper investigates the approximate estimation of correlation clustering costs under the "node-arrival" streaming model. The authors propose the C4Approx algorithm, which achieves a $(O(1), n^{1-\alpha})$-approximation using **sublinear** space of $O(n^{(3+\alpha)/4}\log n)$ words and a constant number of passes. They also provide two matching lower bounds to prove that both multiple passes and additive errors are inevitable. On real-world data, the algorithm achieves performance comparable to the Pivot algorithm while storing only 2% of the nodes.

## Background & Motivation

**Background**: Correlation clustering is a classic NP/APX-hard problem. Given a complete graph with $\pm 1$ edges, the goal is to partition nodes into clusters to minimize the number of "positive edges across clusters + negative edges within clusters." Numerous $O(1)$-approximation algorithms exist, with the Pivot algorithm (3-approximation) being the most representative. In big data scenarios, many edge-arrival streaming algorithms exist, but they typically require $O(n\,\text{polylog}\,n)$ space.

**Limitations of Prior Work**: Real-world data (images, tweets, vectors) naturally appear as "node streams," where edge labels are computed on-demand via similarity functions—explicitly storing $\binom{n}{2}$ edges is impractical. In the node-arrival model, prior work is nearly non-existent. The only related work by Assadi et al. provides a $(O(1), \delta n^2)$ approximation, but the additive term $\delta n^2$ is too loose.

**Key Challenge**: Outputting a clustering requires $\Omega(n)$ space (as the number of clusters can reach $n$). However, if the goal is only to estimate "clusterability" (i.e., OPT cost), it might be possible to break the $n$ barrier. In the node-arrival model, edges can only be queried when both endpoints are in memory, meaning one cannot even enumerate all edges—this presents a fundamental access restriction.

**Goal**: To provide a $(O(1), n^{1-\alpha})$ cost approximation using $o(n)$ space and $O(1)$ passes, while characterizing the necessity of both "multiple passes" and "additive errors" in this model.

**Key Insight**: The core observation is that it is not necessary to find a pivot for every node. By maintaining a small reference set $R$ consisting of "top-ranked nodes according to a random permutation $\pi$," one can directly determine the pivot for most nodes (those whose degree is high enough or whose pivot falls into $R$). The remaining few nodes (which must be low-degree) can be estimated separately via sampling.

**Core Idea**: Combining a "Reference Set $R$ + High/Low Degree Decomposition" allows the total number of mismatches $|E^{\text{mis}}|$ in PrunedPivot to be split into two independently estimated components, reducing space from $O(n)$ to sublinear.

## Method

### Overall Architecture
C4Approx implements a 5-step pipeline.

First pass: Based on a random permutation $\pi$, the top $r=48k n^{1-\beta}\log n$ nodes are stored in the reference set $R$ (where $\beta=(1-\alpha)/4$).

Subsequently, two subroutines are executed in parallel: (i) Est-EA uses 3 passes to estimate $|E_A^{\text{mis}}|$ (mismatched pairs with at least one endpoint in $A$), and (ii) Est-EB uses $k+3$ passes to estimate $|E_B^{\text{mis}}|$ (both endpoints in $B$). Here, $A$ represents nodes whose pivots can be determined via $R$, and $B=V\setminus A$ contains the remaining low-degree nodes.

The final returned value is $(\tilde m_A+\tilde m_B + \frac{3}{8}\epsilon n^{1-\alpha})/(1-\epsilon/8)$. Combined with Theorem 2.1 (Dalirrooyfard et al.) regarding the $(9+\frac{24}{k-1})$-approximation of PrunedPivot, this yields a $(O(1), n^{1-\alpha})$-approximation of the OPT cost with probability at least $0.99$.

### Key Designs

1.  **Reference Set $R$ + FindPivot Node Partitioning**:
    *   **Function**: Uses a sublinear memory $R$ as a "partial oracle for pivot determination" to partition nodes into $A$ (determinable) and $B$ (indeterminable but guaranteed low-degree).
    *   **Mechanism**: FindPivot recursively searches for higher-ranked neighbors within $R$, with a recursion budget of $k$. If successful, $\text{pivot}(u)\in R$ or $u$ is a singleton, classifying it into $A$. If it times out and no neighbors are in $R$, it is classified into $B$. Lemma 2.5 ensures that "nodes in the same cluster are either all in $A$ or all in $B$," allowing independent estimations.
    *   **Design Motivation**: Storing all pivot information requires $\Omega(n)$. Storing $|R|=\tilde O(n^{1-\beta})$ nodes is sufficient to serve all determinations for high-degree nodes (the top $k$ neighbors of a high-degree node likely fall into $R$, proven via Chernoff in Lemma 2.6). The remaining nodes in $B$ must have degrees $\le n^\beta$, which can be handled by sampling.

2.  **Estimation of $E_A^{\text{mis}}$ via High/Low Degree Decomposition**:
    *   **Function**: Estimates the number of mismatched pairs with at least one endpoint in $A$.
    *   **Mechanism**: For each node $u$, $|E_A^{\text{mis}}|$ is equivalent to estimating the average degree of the mismatch subgraph $G_A^{\text{mis}}$. However, the degree range is $\{0,\dots,n-1\}$, and uniform sampling leads to high variance. The authors sample a small set $S_1$ as "high-degree certificates" to partition $V$ into a high-degree set $H$ and a low-degree set $L$. Degrees in $H$ are estimated via rescale-by-sampling, while $L$ is sub-sampled directly. Lemma 2.8 provides a $(1\pm\epsilon,\pm\epsilon n^{1-\alpha})$-approximation with 3 passes and $O(\frac{1}{\epsilon^2}(n^{1-\beta}+n^{\alpha+\beta})\log n)$ space.
    *   **Design Motivation**: Separating heavy-tail and long-tail distributions to control variance is a classic technique. In node streams, specific sampling strategies must be redesigned to accommodate the restriction that edges can only be queried for node pairs simultaneously in memory.

3.  **Estimation of $E_B^{\text{mis}}$ via Cluster Sampling**:
    *   **Function**: Estimates the number of mismatched pairs with both endpoints in $B$.
    *   **Mechanism**: Since all clusters in $B$ are "small" (degree upper bound $n^\beta$), one can sample entire clusters from the cluster set $\mathcal{C}(B)$. Sampled clusters are loaded into memory to count intra/inter-cluster edges and then rescaled. The actual pivots are computed using a streaming implementation of PrunedPivot (Algorithm 2) in $k$ passes and $O(k)$ space. Lemma 2.9 provides the same approximation and confidence with $k+3$ passes and $O(\frac{k}{\epsilon^2}n^{\alpha+3\beta}\log n)$ space.
    *   **Design Motivation**: In $B$, mismatch determination cannot rely on $R$, but the "small cluster" property provides an alternative bound—the contribution of each sampled cluster is capped, naturally reducing variance. Combining both parts covers $E^{\text{mis}}$ entirely.

### Loss & Training
This is a purely combinatorial algorithm with no training involved. Key parameters are $k=37$, $\epsilon=1/10$, and $\beta=(1-\alpha)/4$, which yield the main theorem's $(O(1), n^{1-\alpha})$ approximation and $O(n^{(3+\alpha)/4}\log n)$ space.

## Key Experimental Results

### Main Results
The authors compare C4Approx with Pivot, PrunedPivot, and the algorithm by Assadi et al. on various real-world datasets. Representative results include:

| Dataset / Setting | Memory % | C4Approx Cost | Pivot Cost | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| ImageNet-21K (embeddings + cosine threshold) | 2% nodes | Comparable to Pivot | 100% node storage | Maintains accuracy at 100x compression |
| Sparse Graphs (uneven cluster sizes) | 2% nodes | Significantly better than Assadi et al. | — | Sparse graphs are a weakness for Assadi's algorithm |
| Averaged multiple runs | 2% nodes | Low variance | — | Decomposition effectively suppresses variance |

### Ablation Study

| Configuration | Performance | Description |
| :--- | :--- | :--- |
| C4Approx (Full) | Close to Pivot | Both decomposition and cluster sampling are present |
| SimpleSampling Only | Additive error $\Theta(n^2/\sqrt q)$ | Confirms theoretical view that naive sampling cannot reduce error to $o(n^{1.5})$ in $o(n)$ space |
| No Degree Decomposition | High variance | Validates the criticality of Variance Reduction |
| Assadi et al. on Sparse | Unstable | Difficult to keep additive $\delta n^2$ small simultaneously |

### Key Findings
- In the node-arrival model, additive error is **inevitable** (Lower Bound 2: a $c$-approximation with $d=0$ requires $\Omega(n)$ bits). Multiple passes are also **necessary** (Lower Bound 1: a one-pass $(c,d)$-approximation requires $\Omega(n)$ bits). This provides a clear characterization of the model's inherent difficulty.
- Storing only $\sim 2\%$ of nodes is sufficient to approach Pivot's performance—empirical evidence shows that "sublinear memory + a few passes" is a viable strategy for node-arrival models.
- Comparison with Assadi et al.: To reduce their additive $\delta n^2$ to $n^{0.1}$, $\delta$ must be $n^{-1.9}$, which would require $\Omega(n^{9.5})$ space—making it impractical.

## Highlights & Insights
- **Model Innovation**: Node-arrival, rather than edge-arrival, is a more realistic perspective for big data streams. This direction was previously underestimated; the authors formalize it and provide the first algorithm with matching lower bounds.
- **Transferable Paradigm**: The "Reference Set + High/Low Degree Decomposition" framework could be directly applied to other streaming graph problems requiring on-demand edge queries (e.g., triangle counting, community detection, cut sparsifiers).
- **Theoretical & Empirical Alignment**: The matching upper and lower bounds, combined with experiments confirming that theoretical constants (like $k=37$) are practical, make this a rare "ready-to-run" algorithmic theory paper.

## Limitations & Future Work
- The $(O(1), n^{1-\alpha})$ additive error boundary has little impact on **dense ground truth** ($|E^{\text{mis}}|\gg n^{1-\alpha}$), but in "nearly perfectly clusterable" low-cost scenarios, the additive term might overshadow the true value.
- The algorithm relies on a predefined random permutation $\pi$. If the stream is adversarial (e.g., malicious nodes arriving first), the i.i.d. assumption might be violated; robustness is left for future work.
- Experiments focused on synthetic similarity graphs (embeddings + threshold); performance on scenarios where the similarity oracle itself is expensive (e.g., LLM-based judging) remains unexamined.

## Related Work & Insights
- **vs. Pivot / PrunedPivot**: This work inherits their $O(1)$-approximation guarantees. Its key contribution is porting these to the **sublinear memory + node-arrival** constraint model.
- **vs. Assadi et al. 2023 (edge stream)**: While the output format is similar, this paper's additive term $n^{1-\alpha}$ is much tighter than their $\delta n^2$. The lower bounds also provide a more granular characterization of the model.
- **vs. Dynamic Algorithms (Insertion/Deletion Streams)**: That line of research focuses on update time and still requires $\Omega(n)$ space; this work is complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first systematic sublinear algorithm for correlation clustering in node-arrival streams, complete with lower bounds.
- Experimental Thoroughness: ⭐⭐⭐ Real-world data is used, but limited to similarity graphs generated by embedding thresholds.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivation with clear definitions, lemmas, and theorems.
- Value: ⭐⭐⭐⭐ Directly applicable to "clusterability metrics" for large-scale similarity graphs; the paradigm is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/learning_theory/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[ICML 2026\] Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference](correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference.md)
- [\[ICML 2026\] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)

</div>

<!-- RELATED:END -->
