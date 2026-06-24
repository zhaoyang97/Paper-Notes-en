---
title: >-
  [Paper Note] Online Sparsification of Bipartite-Like Clusters in Graphs
description: >-
  [ICML 2025][Graph Sparsification] Proposes a **near-linear time online graph sparsification algorithm** that compresses the number of edges to $\widetilde{O}(n)$ while preserving the bipartite-like cluster structure of the graph. It is applicable to both undirected and directed graphs, significantly accelerating existing clustering algorithms.
tags:
  - "ICML 2025"
  - "Graph Sparsification"
  - "Bipartite Clustering"
  - "Online Algorithms"
  - "Spectral Methods"
  - "Directed Graphs"
date: 2026-05-08
content_hash: 7595ddeb0cd8af64
---

# Online Sparsification of Bipartite-Like Clusters in Graphs

**Conference**: ICML 2025  
**arXiv**: [2508.05437](https://arxiv.org/abs/2508.05437)  
**Code**: [GitHub](https://github.com/suranjande4/Online-Sparsification-of-Bipartite-Like-Clusters-in-Graphs)  
**Area**: Graph Algorithms/Graph Clustering  
**Keywords**: Graph Sparsification, Bipartite Clustering, Online Algorithms, Spectral Methods, Directed Graphs

## TL;DR

Proposes a **near-linear time online graph sparsification algorithm** that compresses the number of edges to $\widetilde{O}(n)$ while preserving the bipartite-like cluster structure of the graph. It is applicable to both undirected and directed graphs, significantly accelerating existing clustering algorithms.

## Background & Motivation

### Limitations of Traditional Graph Clustering

Traditional graph clustering focuses on finding vertex sets with **low conductance**—where edges are dense within the set and sparse to the outside. However, in many real-world scenarios, it is more critical to mine the **dense interconnection relationships between two sets of vertices**, such as:

- **Migration/Trade Networks**: Population flows or trade transactions between regions form bipartite-like structures.
- **Financial Markets**: Lead-lag relationships between different types of assets.
- **Social Networks**: Chain-like structures in transportation and social graphs.

Such structures are called **bipartite-like clusters**—where edges between a pair of disjoint vertex sets $(A, B)$ occupy the vast majority of edges in $A \cup B$.

### Bottlenecks of Existing Algorithms

Existing clustering algorithms (such as the local algorithms of Macgregor & Sun 2021) are computationally expensive on large-scale graphs. Traditional **spectral sparsification** only preserves the cut values of $w(S, V \setminus S)$ and **does not guarantee preserving** cut values across two subsets like $w(A, B)$. Moreover, most sparsification algorithms rely on Laplacian solvers, which are complex to implement and only applicable to undirected graphs.

### Design Motivation

Can we design a **simple, online, near-linear time** sparsification algorithm that preserves the bipartite-like cluster structure and can be directly used as an acceleration subroutine for existing algorithms?

## Method

### Overall Architecture

The core of this work is divided into two parts:

1. **Undirected Graph Sparsification** (Theorem 1): An edge-sampling-based online algorithm that directly preserves $k$ bipartite-like clusters on undirected graphs.
2. **Directed Graph Sparsification** (Theorem 2): Reduces the directed graph problem to an undirected one using a "semi-double cover" reduction, and then reverse-constructs the sparse directed graph.

### Key Definitions

#### Dual Conductance

For an undirected graph $G=(V,E)$ and two disjoint subsets $A, B \subset V$, the **dual conductance** is defined as:

$$\overline{\phi}_G(A,B) \triangleq \frac{2 w_G(A,B)}{\text{vol}_G(A \cup B)}$$

A larger $\overline{\phi}_G(A,B)$ indicates a higher proportion of edges between $A$ and $B$, which means $(A,B)$ forms a stronger bipartite-like cluster.

#### k-way Dual Cheeger Constant

$$\bar{\rho}_G(k) \triangleq \max_{(A_1,B_1),\ldots,(A_k,B_k)} \min_{1 \le i \le k} \overline{\phi}_G(A_i, B_i)$$

Quantifies the strength of $k$ bipartite-like clusters in the graph $G$. A higher $\bar{\rho}_G(k)$ indicates that $G$ possesses more prominent $k$ bipartite-like structures.

### Key Designs

#### Undirected Graph Sparsification Algorithm

**Core Idea**: Sample each edge independently with a carefully designed sampling probability, where sampled edges are reweighted by their inverse probabilities.

For each edge $e = \{u,v\}$, the sampling probability is defined as:

$$p_e = p_u(v) + p_v(u) - p_u(v) \cdot p_v(u)$$

where:

$$p_u(v) = \min\left\{\frac{w_G(u,v) \cdot C \cdot \log^3 n}{d_G(u) \cdot (2 - \lambda_{n-k})}, 1\right\}$$

- $C$ is a constant, and $\lambda_{n-k}$ is the $(n-k)$-th eigenvalue of the normalized Laplacian $\mathcal{L}_G$.
- A smaller $(2 - \lambda_{n-k})$ (i.e., $\lambda_{n-k}$ closer to 2) indicates a more prominent bipartite-like structure, leading to higher sampling probabilities.
- The weight of the sampled edge is reset to $w_{G^*}(e) = w_G(e) / p_e$ (unbiased estimator).

**Key Guarantees** (Theorem 5):

- The output graph $G^*$ contains only $O\left(\frac{n \cdot \log^3 n}{2-\lambda_{n-k}}\right)$ edges.
- $\bar{\rho}_{G^*}(k) = \Omega(\bar{\rho}_G(k))$ — clustering quality is preserved.
- $\lambda_{k+1}(\mathcal{J}_{G^*}) = \Theta(\lambda_{k+1}(\mathcal{J}_G))$ — the number of clusters remains unchanged.

**Proof Core**: Uses Chebyshev's inequality to prove that the cut value $w_H(A_i, B_i) = \Omega(w_G(A_i, B_i))$ holds with high probability, which is then extended to all $k$ clusters using a union bound.

#### Directed Graph Sparsification (Three-Step Reduction)

The difficulty for directed graphs lies in: the natural matrix representation produces complex eigenvalues, and there is no directed graph version of the higher-order dual Cheeger inequality. This paper elegantly addresses this through a three-step reduction:

**Step 1: Semi-Double Cover**

Converts the directed graph $\overrightarrow{G}$ into an undirected bipartite graph $H$:
- Each vertex $v$ is duplicated into $v_1, v_2$.
- Each directed edge $(u,v)$ becomes an undirected edge $\{u_1, v_2\}$.

Key Lemma (Lemma 7): $f_{\overrightarrow{G}}(A,B) = \phi_H(A_1 \cup B_2)$, meaning that the flow ratio of the directed graph is equivalent to the conductance in the undirected cover graph.

**Step 2: Sparsify the Undirected Graph $H$**

Sparsify $H$ into $H^*$ using an existing cluster-preserving sparsification algorithm (Lemma 9, Sun & Zanetti 2019), keeping the $k$ low conductance sets.

**Step 3: Reverse Semi-Double Cover**

Reverse-construct the directed graph $\overrightarrow{G^*}$ from $H^*$:
- Each pair $(u_1, u_2)$ in $H^*$ is merged back into vertex $v$.
- The edge $\{u_1, v_2\} \in H^*$ is restored to the directed edge $(u,v)$.

Core Difficulty: During reverse construction, the optimal $k$-way partition in $H^*$ is not necessarily a simple set. The paper proves that for any (non-simple) set $S$, there exists a simple set $T$ such that $\phi_H(S) \ge (1-c) \cdot \phi_H(T)$, thereby establishing the chain of inequalities:

$$1 - \frac{1}{1-c} \cdot \rho_{H^*}(k) \le \bar{\rho}_{\overrightarrow{G^*}}(k) \le 1 - \rho_{H^*}(k)$$

### Loss & Training

This work is a theoretical algorithm-focused study and does not involve traditional loss functions or training strategies. Its core optimization objectives are:

- **Minimize the number of edges**: Output $\widetilde{O}(n)$ edges (compared to the input $m$ edges).
- **Maximize structure preservation**: $\bar{\rho}_{G^*}(k) = \Omega(\bar{\rho}_G(k))$.
- **Online processing**: The algorithm only requires a degree oracle and can perform sparsification on the fly during graph exploration.

Unlike methods based on Laplacian solvers, this work **uses random sampling only**, which is simple and efficient to implement.

## Key Experimental Results

### Main Results

#### Undirected Graphs - Synthetic Data (SBM, p=0.3, q=0.1p)

| Number of Vertices (per partition) | MS Runtime | Ours Runtime | MS Bipartiteness | Ours Bipartiteness |
|---|---|---|---|---|
| 500| ~0.3s | ~0.05s | ~0.18 | ~0.18 |
| 1500 | ~2.5s | ~0.3s | ~0.18 | ~0.18 |
| 2500 | ~8s | ~0.5s | ~0.18 | ~0.18 |

#### Undirected Graphs - Real-world Data (Militarised Interstate Disputes)

| Seed Country | MS Time | Ours Time | MS Bipartiteness | Ours Bipartiteness |
|---|---|---|---|---|
| USA | 0.034s | **0.0044s** (7.7×) | 0.292 | 0.285 |
| Netherlands | 0.0351s | **0.0042s** (8.4×) | 0.307 | 0.281 |
| Lithuania | 0.0336s | **0.0043s** (7.8×) | 0.303 | 0.165 |

#### Directed Graphs - Real-world Data (US Migration)

| Seed County | Target $\phi$ | ECD Time | Ours Time | ECD Flow-ratio | Ours Flow-ratio |
|---|---|---|---|---|---|
| Maricopa County | 0.2 | 20.66s | **13.43s** | 0.414 | 0.417 |
| Virginia Beach City | 0.2 | 15.31s | **12.29s** | 0.546 | 0.621 |
| Kanawha County | 0.2 | 9.32s | **8.48s** | 0.330 | 0.330 |

### Ablation Study

| Configuration | Key Metrics | Description |
|---|---|---|
| $\eta=0.7$ (Weak bipartite) | Speedup of ~1.3-2× | A lower $\eta$ means a weaker bipartite structure, leading to a relatively smaller speedup factor. |
| $\eta=0.8$ (Medium structure) | Speedup of ~2-3× | The structure is more prominent, yielding better sparsification. |
| $\eta=0.9$ (Strong bipartite) | Speedup of ~3-5× | Under strong structures, sampling probabilities are more concentrated, yielding the most significant effect. |
| Vertices 500 $\rightarrow$ 2500 | Speedup increases with scale | The advantage of sparsification becomes more pronounced on larger graphs. |

### Key Findings

1. **Significant Speedup**: On undirected synthetic data, as the number of vertices increases, the speedup factor of the proposed algorithm compared to MS grows from ~6× to ~16×.
2. **Quality Preservation**: The bipartiteness ratio and flow-ratio remain almost identical to the original algorithm.
3. **Higher $\eta$ Yields Better Performance**: The stronger the bipartite structure, the more information is preserved by sparsification, and the more significant the speedup.
4. **Real-world Validation**: The method is effective across multi-domain datasets, including military conflicts and population migration.

## Highlights & Insights

1. **Exquisite Sampling Probability Design**: $p_u(v)$ is proportional to $w(u,v)/d_G(u)$, which essentially weights the "importance of an edge from the perspective of its endpoints"—higher endpoint degree leads to a lower sampling probability, preventing edges of high-degree nodes from being excessively retained.
2. **Adaptability of $(2-\lambda_{n-k})$**: This factor automatically adapts to the strength of the graph's bipartite-like structure. Stronger structures lead to more aggressive sampling, which is highly elegant.
3. **Semi-Double Cover Reduction**: Mapping the directed graph problem perfectly to an undirected graph problem is the most creative technical contribution of this work. This reduction allows reusing the spectral theory tools of undirected graphs.
4. **Online Characteristics**: The algorithm only requires a degree oracle and can sparsify the graph while exploring it, making it naturally suitable for integration with local algorithms.
5. **Simple Implementation**: Compared to sparsification methods that rely on Laplacian solvers, this work uses only random sampling, making it highly engineering-friendly.

## Limitations & Future Work

1. **Dependency on $\lambda_{n-k}$**: The sampling probability depends on the eigenvalue $(2-\lambda_{n-k})$. In practice, it is usually approximated by $O(\log^c n)$, but accurate estimation might impact the performance.
2. **Preset $k$ Required**: The algorithm assumes $k$ is known, and automatically determining $k$ remains an open problem.
3. **Limited Sparsification Ratio**: For graphs without prominent bipartite structures ($\bar{\rho}_G(k)$ close to 0), the scope for sparsification is limited.
4. **Small Experimental Scale**: The maximum scale tested is 5000 vertices, and performance on million/billion-scale graphs remains to be verified.
5. **Strong Conditions on Directed Graphs**: Theorem 2 requires $\bar{\rho}_{\overrightarrow{G}}(k) = 1 - o(1/k)$, indicating a need for a very strong bipartite structure.
6. **Only Preserving Bipartite-Like Clusters**: It does not guarantee preserving other types of clustering structures (such as low conductance communities).

## Related Work & Insights

- **Trevisan (2009)**, **Soto (2015)**: Used spectral methods to find a single bipartite-like cluster; this work generalizes it to $k$ clusters.
- **Macgregor & Sun (2021a)**: Proposed LocBipartDC and ECD local algorithms; this work is directly used as their acceleration module.
- **Sun & Zanetti (2019)**: Cluster-preserving sparsification; this work extends it to the dual Cheeger setting.
- **Cucuringu et al. (2020)**: Spectral clustering of directed graphs using Hermitian matrices; this work provides a simpler perspective using semi-double cover reduction.
- **Liu (2015)**: Higher-order dual Cheeger inequality, which serves as a key tool for the theoretical foundation of this work.
- **Insight**: This three-step paradigm of "reduction-sparsification-reverse construction" could potentially be generalized to other types of structured clustering problems.

## Rating

| Dimension | Rating (1-5) | Description |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | The combination of semi-double cover reduction and online sparsification is highly novel. |
| Theoretical Depth | ⭐⭐⭐⭐⭐ | Rigorous theoretical guarantees with a complete proof chain. |
| Experimental Thoroughness | ⭐⭐⭐ | Contains both synthetic and real-world data, but the scale is relatively small. |
| Practicality | ⭐⭐⭐⭐ | Simple implementation, can directly accelerate existing algorithms. |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured, with rigorous definitions and theorem statements. |
| **Overall Rating** | **⭐⭐⭐⭐** | A solid theoretical work with clear practical value. |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structure-Aware Spectral Sparsification via Uniform Edge Sampling](../../NeurIPS2025/others/structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)
- [\[ICML 2025\] Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation](bipartite_ranking_from_multiple_labels_on_loss_versus_label_aggregation.md)
- [\[ICLR 2026\] t-SNE Exaggerates Clusters, Provably](../../ICLR2026/others/t-sne_exaggerates_clusters_provably.md)
- [\[NeurIPS 2025\] Adjusted Count Quantification Learning on Graphs](../../NeurIPS2025/others/adjusted_count_quantification_learning_on_graphs.md)
- [\[ACL 2025\] Beyond Position: the emergence of wavelet-like properties in Transformers](../../ACL2025/others/beyond_position_the_emergence_of_wavelet-like_properties_in_transformers.md)

</div>

<!-- RELATED:END -->
