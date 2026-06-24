---
title: >-
  [Paper Note] Understanding Truncated Positional Encodings for Graph Neural Networks
description: >-
  [ICML 2026][Graph Learning][Positional Encoding] This paper theoretically demonstrates that while spectral and walk encodings are equivalent in their "full" forms, their expressive power differs significantly in the "truncated" versions used in practice. Truncated spectral encodings may not even be stronger than the 1-WL test. Consequently, the authors suggest the empirical rule: "Since truncation is necessary, mix positional encodings from different families…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Positional Encoding"
  - "Truncation"
  - "Spectral Encoding"
  - "$k$-harmonic distance"
  - "WL test"
date: 2026-05-08
content_hash: c1acc252ded40f84
---

# Understanding Truncated Positional Encodings for Graph Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2606.13671](https://arxiv.org/abs/2606.13671)  
**Code**: To be confirmed  
**Area**: Graph Learning / GNN Expressivity  
**Keywords**: Positional Encoding, Truncation, Spectral Encoding, $k$-harmonic distance, WL test

## TL;DR
This paper theoretically demonstrates that while spectral and walk encodings are equivalent in their "full" forms, their expressive power differs significantly in the "truncated" versions used in practice. Truncated spectral encodings may not even be stronger than the 1-WL test. Consequently, the authors suggest the empirical rule: "Since truncation is necessary, mix positional encodings from different families," and validate this on real-world datasets.

## Background & Motivation
**Background**: The expressivity of Message Passing Neural Networks (MPNNs) is bounded by the 1-WL test, failing to capture global structures. Graph Transformers (GTs) overcome this by injecting global structural information via Positional Encodings (PE). Two mainstream PE families are **spectral encodings** (Laplacian eigenspace projections, effective resistance, etc.) and **walk encodings** (powers of the adjacency matrix). Previous theory has proven that when walk encodings use $\Omega(n)$ powers, the two families are equivalent and fall between 1-WL and 3-WL.

**Limitations of Prior Work**: The aforementioned equivalence assumes the use of "full" PEs. However, full spectral/walk PEs require $O(n^3)$ time and space complexity due to eigendecomposition for each training graph, which is prohibitively expensive. In practice, researchers use **truncated** versions: the top-$k$ Laplacian eigenspaces or the first $k$ adjacency powers. The theoretical properties of these truncated PEs have remained unexplored.

**Key Challenge**: There is a default assumption that "truncation $\approx$ approximation of the full version," implying truncated spectral and walk encodings should behave similarly. This assumption has never been verified. If truncation breaks equivalence, the choice of truncated PE becomes a substantive model design decision rather than a negligible engineering detail.

**Goal**: (1) Characterize the expressive gap between truncated spectral encodings, truncated walk encodings, and effective resistance; (2) Identify a PE family that bridges spectral and walk encodings with smooth truncation; (3) Provide actionable PE selection advice for practitioners.

**Key Insight**: The authors use WL test hierarchies (and variants like EP-WL, Walk-WL, GD-WL) as proxies for "structural sensitivity" to compare which non-isomorphic graphs different truncated PEs can distinguish. This approach directly maps expressivity results to structural properties (centrality, motif counts, long-range dependencies) computable by the model.

**Core Idea**: Truncation is not a harmless approximation—it causes previously equivalent PE families to carry distinct information. Thus, mixing truncated PEs from different families provides complementary structural clues, whereas any single family is insufficient.

## Method
This paper is primarily **theoretical analysis supported by experimental validation**, focusing on theorems regarding the expressivity of truncated PEs rather than a pipeline architecture.

### Overall Architecture
The study focuses on **Relative Positional Encodings (RPE)**: associating a vector with every pair of nodes $(u,v)$ that remains invariant under graph isomorphism. Three types of truncated PEs are analyzed:

- **$k$-EP-WL**: Eigenspace projection encoding. For each eigenvalue $\lambda_i$ of the Laplacian $L=D-A$, let $\Pi_i=\sum_j z_{i,j}z_{i,j}^T$ be the projection matrix. The encoding is $\mathcal{P}^k(u,v)=\{(\lambda_i,\Pi_i(u,v)):1\le i\le k\}$, retaining only the $k$ smallest eigenvalues. Space complexity is reduced from $O(n^3)$ to $O(kn^2)$.
- **$k$-Walk-WL**: The first $k$ adjacency powers, where $(A^k)_{uv}$ counts walks of length $k$ between $u$ and $v$.
- **Effective Resistance / $k$-harmonic distance**: $R(u,v)=(1_u-1_v)^T L^+ (1_u-1_v)$ measures connectivity, where $L^+$ is the Laplacian pseudoinverse.

Expressivity is bounded by GD-WL ($\psi$-WL color refinement for RPE $\psi$). The analysis investigates where truncated $\psi$-WL sits in the WL hierarchy.

**Mechanism**: Full spectral/walk PEs are equivalent and trapped between 1-WL and 3-WL. Once truncated, they capture disparate aspects of graph structure; neither is strictly stronger, and they may even fall below 1-WL.

### Key Designs

**1. Truncation Breaks Equivalence: Spectral vs. Walk**

Theorem 4.1 constructs a pair of graphs $G,H$ distinguishable by 1-WL but **indistinguishable** by $k$-EP-WL even for $k\in\Omega(n)$. Thus, truncated spectral PEs can be weaker than basic MPNNs and are **incomparable** to the WL hierarchy. Theorem 4.2 provides the reverse: a pair indistinguishable by $\Omega(n)$-Walk-WL but distinguishable by 1-EP-WL ($k=1$). This implies that a constant-size truncated PE of one family can distinguish graphs that a linear-size truncated PE of another family cannot.

**2. Weakness of Effective Resistance: Incomplete Adjacency Encoding**

Effective resistance is often assumed to be strong due to its relation to commute times. However, Theorem 4.3 constructs weighted graphs where effective resistance cannot distinguish them while Adjacency-WL can. Corollary 4.1 shows that **effective resistance is strictly weaker than eigenspace projections**. This refutes previous conjectures that effective resistance-WL is stronger than shortest-path-WL and encodes the full spectrum; it cannot even fully recover adjacency information.

**3. $k$-harmonic distance Bridge: A Scalable PE Family**

The authors generalize effective resistance to $k$-harmonic distance $H^k(u,v)=\sqrt{(1_u-1_v)^T (L^+)^k (1_u-1_v)}$ (where $k=1$ squared is effective resistance). Theorem 4.6 proves that concatenating $[2n]$ $k$-harmonic distances is equivalent to EP-WL. Theorem 4.4/4.5 shows that sparse-$k$-harmonic-WL is strictly stronger than 1-WL but strictly bounded by 3-WL.

### A Concrete Example: Trees
On **trees**, truncated PEs reveal their specific "blind spots":
- **Effective Resistance**: On any tree, the effective resistance between two nodes equals their shortest path distance, and the resistance of all edges is 1. Thus, sparse-ER-WL fails to provide any information beyond 1-WL (Lemma 4.1).
- **Biharmonic Distance ($k=2$)**: The biharmonic distance on an edge $(s,t)$ depends on the number of nodes on either side of the edge. This provides additional topological information, allowing sparse-biharmonic-WL to distinguish certain trees in **one iteration** that ER-WL cannot distinguish in $o(n)$ iterations (Theorem 4.7).

## Key Experimental Results

### Main Results
The theory is validated using Graphormer-GD on **BREC** (expressivity benchmark) and **ZINC-12k**. On BREC, $k$-harmonic distances approach theoretical upper bounds using only 1D encodings, costing much less than multi-dimensional EP projections.

| BREC Subgroup | ER ($k=1$) | Biharmonic ($k=2$) | 4-Harmonic | $k$=5 EP | $k$=7 EP | 3-WL |
|-----------|---------|--------|--------|----------|----------|------|
| Basic | 0 | 46.6 | 86.7 | 91.6 | 100 | 100 |
| Regular | 0 | 27.1 | 32.1 | 30.7 | 35.7 | 35.7 |
| Extension | 0 | 59 | 83 | 92 | 100 | 100 |
| Total(%) | 0 | 32 | 46.5 | 48 | — | 60 |

### Ablation Study (ZINC-12k, Regression MAE, Lower is Better)
PE dimension is fixed at 8. Comparison of single vs. mixed families:

| PE Combination | Dim Ratio | Test MAE |
|---------|---------|----------|
| Projections only | 8 | $0.117\pm0.005$ |
| $k$-Harmonics only | 8 | $0.076\pm0.006$ |
| Walks only | 8 | $0.082\pm0.003$ |
| Projections + Walks | 4+4 | $0.075\pm0.005$ |
| Projections + $k$-Harmonics | 4+4 | $0.078\pm0.002$ |
| **Walks + $k$-Harmonics** | 4+4 | $\mathbf{0.064\pm0.002}$ |

### Key Findings
- **Mixing > Single**: The combination of Walks + $k$-Harmonics (0.064) significantly outperforms any single family, validating the core claim that truncation necessitates mixing.
- **Redundant Signals**: Projections + $k$-Harmonics (0.078) provided little gain over $k$-Harmonics alone, suggesting certain truncated spectral features are redundant.
- **Efficiency**: $k$-harmonic distances achieve high performance with 1D channels, offering low preprocessing and training overhead compared to multi-dimensional EP/Walks.

## Highlights & Insights
- **Paradigm Shift**: The reversal of "truncation = approximation" is the most valuable insight. Truncated PEs are an independent design space, explaining why switching PE types often changes downstream performance.
- **Principled Proxy**: The use of WL hierarchies as a proxy for "structural sensitivity" is well-argued, providing a grounded way to determine which structural signals are exposed.
- **Unified Perspective**: Viewing effective resistance as a $k=1$ special case of a continuous family allows for better PE selection and lightweight baselines.
- **Actionable Advice**: The conclusion clearly points to mixing "complementary" families (Walks + Harmonics) rather than redundant ones (Projections + Harmonics).

## Limitations & Future Work
- **Weighted Graph Counterexamples**: Key separation results like Theorem 4.3 rely on weighted graph constructions; whether these hold for unweighted graphs remains a conjecture.
- **Expressivity $\neq$ Performance**: Higher WL-rank does not always translate to better downstream metrics. The ZINC improvement (0.082 to 0.064) is notable but moderate.
- **Experimental Scale**: Validation is limited to BREC and ZINC-12k; broader benchmarks and node-level tasks would strengthen the robustness of the "mixing" rule.
- **Future Direction**: Combining the $k$-harmonic perspective with learnable PE selection (allowing the model to weight different families) could yield further improvements.

## Related Work & Insights
- **vs. Full PE Equivalence (Black et al. 2024; Gai et al. 2025)**: While they prove equivalence in the full form, this paper demonstrates that such equivalence **completely vanishes** upon truncation.
- **vs. Zhang et al. 2023 (ER as PE)**: Contrary to their conjecture that ER encodes the full spectrum, Corollary 4.1 proves ER is strictly weaker than EP-WL and fails to encode adjacency.
- **vs. Lim et al. 2023 / Huang et al. 2024 (Eigenspace Projections)**: This paper builds on their projection encoding but shifts focus to the $k$-truncated version, showing it is incomparable to the WL hierarchy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LEAP: Local ECT-Based Learnable Positional Encodings for Graphs](../../ICLR2026/graph_learning/leap_local_ect-based_learnable_positional_encodings_for_graphs.md)
- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](quantile-free_uncertainty_quantification_in_graph_neural_networks.md)
- [\[ICML 2026\] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)
- [\[ICML 2026\] Physics-Informed Coarsening for Multigrid Graph Neural Surrogates](physics-informed_coarsening_for_multigrid_graph_neural_surrogates.md)

</div>

<!-- RELATED:END -->
