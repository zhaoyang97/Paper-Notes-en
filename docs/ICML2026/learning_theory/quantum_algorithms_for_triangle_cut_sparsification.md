---
title: >-
  [Paper Note] Quantum Algorithms for Triangle Cut Sparsification
description: >-
  [ICML 2026][learning_theory][Paper Note] This work designs quantum algorithms for "triangle cut sparsification": it first provides the first provably accelerated quantum triangle listing algorithm (integrating heavy-light vertex partitioning, quantum walk, and Grover search, taking the best of the three). It then embeds this into the motif sparsification fram
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: a74cb184bc96bb50
---
# Quantum Algorithms for Triangle Cut Sparsification

**Conference**: ICML2026  
**arXiv**: [2606.06287](https://arxiv.org/abs/2606.06287)  
**Code**: None (Theoretical paper)  
**Area**: Learning Theory / Quantum Algorithms / Graph Sparsification  
**Keywords**: Triangle cut sparsification, triangle listing, quantum walk, Grover search, motif sparsification

## TL;DR
This work designs quantum algorithms for "triangle cut sparsification": it first provides the first provably accelerated quantum triangle listing algorithm (integrating heavy-light vertex partitioning, quantum walk, and Grover search, taking the best of the three). It then embeds this into the motif sparsification framework of Kapralov et al. and applies quantum acceleration to post-processing sampling. This allows the construction of an $\varepsilon$-triangle cut sparsifier of size $\widetilde{O}(n/\varepsilon^2)$ with an additional overhead of $\widetilde{O}(\sqrt{mn}/\varepsilon)$, accompanied by a matching lower bound of $\Omega(n/\varepsilon^2)$.

## Background & Motivation

**Background**: Graph sparsification (spanners, cut sparsifiers, spectral sparsifiers) approximates the key structures of the original graph by retaining a small number of weighted edges, serving as a fundamental tool for large-scale graph algorithms and machine learning. Recently, Kapralov et al. (2022) proposed the **motif cut sparsifier**, which maintains the weighted count of fixed subgraphs (motifs, e.g., triangles) crossing each cut instead of edge counts, to capture essential high-order structures in social networks, recommendation systems, and graph clustering.

**Limitations of Prior Work**: Although the framework of Kapralov et al. can produce linear-sized sparsifiers ($\widetilde{O}(n)$ edges), its classical implementation suffers from two computational bottlenecks: ① It requires **explicitly listing all triangles** (triangle listing), which is costly; ② it requires $\widetilde{O}(m)$ time for **post-sampling** edges in each iteration. Together, these steps dominate the overall runtime.

**Key Challenge**: High-order structures (triangles) provide expressiveness, but enumerating them is inherently expensive. Triangle listing is a well-known hard problem in fine-grained complexity, where the classical optimal bound $\widetilde{O}(m^{4/3}+mt^{1/3})$ is widely considered conditionally optimal. Accelerating sparsification necessitates accelerating triangle listing.

**Key Insight**: Quantum computing has shown promise in sparsification; Apers & De Wolf (2022) provided the first $\widetilde{O}(\sqrt{mn}/\varepsilon)$ quantum spectral sparsification algorithm, and Liu et al. (2025) extended it to hypergraphs. However, **triangle cut sparsification—the high-order version—has never been studied from a quantum perspective**. Furthermore, while quantum triangle **detection** (judging if at least one exists) has mature results at $\widetilde{O}(n^{5/4})$, quantum triangle **listing** (listing all $t$ triangles) is virtually unexplored. The authors identify this as a lever for acceleration.

**Core Idea**: The authors establish "quantum triangle listing" as a provably accelerated technical primitive. This primitive is used to simultaneously speed up the enumeration and post-sampling phases of the motif sparsification framework—replacing classical enumeration/sampling with quantum counterparts—thereby accelerating triangle cut sparsification and improving triangle-based clustering algorithms as a byproduct.

## Method

### Overall Architecture

The work follows the "strength-based motif sparsification" pipeline of Kapralov et al.: treating each triangle as a hyperedge, it iteratively retains edges with high estimated importance and samples the rest, resulting in a sparsifier of size $\widetilde{O}(n/\varepsilon^2)$. Quantum acceleration is injected into the enumeration and post-sampling bottlenecks:

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: Weighted graph G(V,E,w)<br/>QRAM Query Access"] --> B["Quantum Triangle Listing<br/>Parallel Execution of Three Approaches"]
    B --> B1["Heavy-Light Partitioning<br/>m + m^3/4 t^1/2"]
    B --> B2["Quantum Walk<br/>n^5/4 t^7/12 + n^7/6 t^7/9"]
    B --> B3["Grover Direct Search<br/>n^3/2 t^1/2"]
    B1 --> C["Quantum Post-sampling<br/>Critical edge detection + Implicit sampling"]
    B2 --> C
    B3 --> C
    C -->|Iterate until reach O~(n/ε²) size| C
    C --> D["Output: ε-triangle cut sparsifier<br/>+ Matching Ω(n/ε²) lower bound"]
```

The core technical contribution is `Q-TriangleListing` (Theorem 1.3), which runs three parallel algorithms and terminates with the fastest one, yielding a runtime of:

$$T_{\mathrm{q\text{-}list}}=\widetilde{O}\bigl(\min(n^{5/4}t^{7/12}+n^{7/6}t^{7/9},\ m+m^{3/4}t^{1/2},\ n^{3/2}t^{1/2})\bigr).$$

The total time for the final sparsification theorem (Theorem 1.2) is $T_{\mathrm{q\text{-}list}}+\widetilde{O}(\sqrt{mn}/\varepsilon)$. Algorithms assume the input graph is stored in classical Memory with Quantum Random Access (QRAM), supporting degree, neighbor, and vertex-pair queries through coherent access.

### Key Designs

**1. Heavy-Light Vertex Partitioning: Divide and Conquer via Degree Threshold**

To address the high cost of direct listing, this approach splits vertices into **heavy vertices** $V_H=\{v:\deg(v)>\frac{9}{10}\Delta\}$ and **light vertices** $V_L=\{v:\deg(v)\le\frac{11}{10}\Delta\}$ using a degree threshold $\Delta$. By the Handshaking Lemma, $|V_H|\le\frac{20m}{9\Delta}$. Triangles are classified as either having at least one light vertex or being composed entirely of heavy vertices. For each light vertex $v$, a **repetitive Grover search** is used to list all triangles passing through $v$ in $\widetilde{O}(\deg(v)\sqrt{t_v})$ time. For purely heavy triangles, Grover search is applied to all heavy vertex triples in $\widetilde{O}((m/\Delta)^{3/2}t^{1/2})$. Setting $\Delta=\sqrt{m}$ yields a total time of $\widetilde{O}(m+m^{3/4}t^{1/2})$. This is optimal for sparse graphs and demonstrates a quantum advantage over classical bounds when $t \le m^{3/2}$.

**2. Two-Layer Quantum Walk: Extending Detection to Listing without Duplication**

To handle cases with a moderate number of triangles, this approach generalizes the quantum triangle detection framework of Le Gall (2014). It involves two stages: **Pre-Listing**, which randomly samples a subset $S$ of vertices and uses Grover search to list all triangles containing vertices in $S$; and **Exhaustive Listing**, which repeatedly runs a **two-layer quantum walk** (`Q-WalkSearch`). The outer walk moves on a Johnson graph $J(V,h)$ to find subsets containing at least one edge of a remaining triangle, while the inner walk checks for specific edges. The technical challenge lies in bounding the ratio of "marked states" in the outer walk. The authors employ a **combinatorial packing argument** to provide a strict lower bound on the marked state ratio without assuming edge independence. This results in a time of $\widetilde{O}(n^{5/4}t^{7/12}+n^{7/6}t^{7/9})$ (Theorem 3.4).

**3. Grover Direct Search + Best-of-Three: Dense Graph Coverage**

For dense graphs with many triangles, the third approach applies Grover search directly to triples, resulting in $\widetilde{O}(n^{3/2}t^{1/2})$. This approaches Grover optimality. The finalized `Q-TriangleListing` (Theorem 1.3) runs the three strategies in parallel. For $t=1$, it degrades to $\widetilde{O}(n^{5/4})$, matching the best-known quantum detection bound. For $t=\Theta(n^3)$, it reaches the $\Theta(n^3)$ lower bound dictated by the output size.

**4. Quantum Post-sampling: Implicit Sampling Decisions**

To address the $\widetilde{O}(m)$ post-sampling bottleneck, the authors quantize two steps: "finding critical edges" (importance above threshold) and "sampling the rest." The former is accelerated by triangle listing and Grover search. For the latter, inspired by Apers & De Wolf (2022), the authors **avoid explicitly writing out every sampling result, instead encoding sampling decisions implicitly and retrieving only surviving edges via quantum search**. This reduces the post-sampling complexity from $\widetilde{O}(m)$ to $\widetilde{O}(\sqrt{mn}/\varepsilon)$ per round.

### Loss & Training
This is a purely theoretical algorithmic work. Correctness and runtimes are provided with "high probability" (at least $1-\mathrm{poly}(1/n)$). Parameter settings (e.g., $\Delta=\sqrt{m}$) are derived from balancing costs across stages. A "guessing schedule" is used to approximate the total triangle count $t$ when unknown.

## Key Experimental Results

As a theoretical paper, the "results" are time complexity bounds and lower bounds.

### Main Results: Runtime Comparison

| Task | Prev. SOTA | Ours (Quantum) | Condition for Quantum Advantage |
|------|-----------|-----------|-------------|
| Triangle Listing (Sparse) | $\widetilde{O}(m^{4/3}+mt^{1/3})$ ($\omega=2$) | $\widetilde{O}(m+m^{3/4}t^{1/2})$ | Superior when $t\le m^{3/2}$ |
| Triangle Listing (Dense) | $\widetilde{O}(n^2+nt^{2/3})$ ($\omega=2$ conj. lower bound) | $\widetilde{O}(\min(n^{5/4}t^{7/12}+n^{7/6}t^{7/9},\ n^{3/2}t^{1/2}))$ | Superior when $t\le n^{15/14}$ |
| Triangle Cut Sparsification | $\widetilde{O}(\min\{T_{\mathrm{c\text{-}list}},n^\omega\}+m)$ | $T_{\mathrm{q\text{-}list}}+\widetilde{O}(\sqrt{mn}/\varepsilon)$ | Listing + Sampling $\widetilde{O}(m)\!\to\!\widetilde{O}(\sqrt{mn}/\varepsilon)$ |

### Extremes and Lower Bounds

| Scenario | Result | Description |
|------|---------|------|
| $t=1$ (Single triangle) | $\widetilde{O}(n^{5/4})$ | Matches optimal quantum triangle **detection** (Le Gall 2014) |
| $t=\Theta(n^3)$ (Denser) | $\Theta(n^3)$ | Hard lower bound due to output size |
| $\widetilde{O}(n^{3/2}t^{1/2})$ Term | Near Grover Optimality | Cannot be improved beyond logarithmic factors |
| Sparsifier Size Lower Bound | $\Omega(n/\varepsilon^2)$ (Theorem 5.2) | First matching lower bound for triangle motifs |

### Key Findings
- **Three enumeration algorithms cover different regimes**: Heavy-light for sparse, Quantum walk for moderate, and Grover for dense graphs.
- **Acceleration from non-trivial primitives**: The gains come from intrinsic quantum mechanisms (quantum walk, variable-cost search) rather than black-box Grover wrapping.
- **Tight Upper/Lower Bounds**: The results align with detection bounds at $t=1$ and output bounds at $t=n^3$. Subgraph size $\Omega(n/\varepsilon^2)$ is nearly tight.
- **Clustering Applications**: Embedding the listing primitive into spectral clustering or PageRank clustering based on triangles yields immediate quantum speedups.

## Highlights & Insights
- **Upgrading Detection to Listing is Non-Trivial**: Moving from existence to enumeration requires managing "edge deletion" via oracle masks and a second-moment packing argument to bound marked states—this methodology is transferable to other motif listings ($k$-cliques).
- **Implicit Encoding for Post-Sampling**: Not materializing every sampling decision is a reusable pattern for scenarios where the final result is sparse.
- **Distinction from Hypergraph Sparsification**: The authors clarify that motif sparsification cannot be reduced to hypergraph sparsification because an edge can belong to both retained and discarded hyperedges, necessitating a return to the strength-based framework.

## Limitations & Future Work
- **QRAM Dependency**: Bounds rely on Memory with Quantum Random Access, which is a stronger assumption than standard oracle query models.
- **Asymptotic vs. Practical**: The polynomial advantage is relevant for massive graphs on fault-tolerant quantum computers and does not define a hardware crossover point.
- **Lack of Listing Lower Bound**: As no specific quantum lower bound for triangle listing exists yet, it is unknown if $T_{\mathrm{q\text{-}list}}$ is strictly optimal (except for the $n^{3/2}t^{1/2}$ term).
- **Lower Bound Generalization**: The $\Omega(n/\varepsilon^2)$ bound is specific to triangles; generalization to general $k$-vertex motifs remains open.

## Related Work & Insights
- **vs. Kapralov et al. (2022)**: Built upon their strength-based framework but resolved the listing and $O(m)$ sampling bottlenecks.
- **vs. Apers & De Wolf (2022)**: Extended the $\widetilde{O}(\sqrt{mn}/\varepsilon)$ quantum spectral (edge) sparsification to higher-order motifs.
- **vs. Liu et al. (2025)**: While they did hypergraphs, this work identifies the semantic mismatch in applying hypergraph sparsification to motifs.
- **vs. Le Gall (2014)**: Upgraded his $\widetilde{O}(n^{5/4})$ detection framework to an enumeration framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive theoretical bounds)
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)
- [\[ICML 2026\] Matroid Algorithms Under Size-Sensitive Independence Oracles](matroid_algorithms_under_size-sensitive_independence_oracles.md)
- [\[ICML 2026\] Online Learning with Recency: Algorithms for Sliding-window Streaming Multi-armed Bandits](online_learning_with_recency_algorithms_for_sliding-window_streaming_multi-armed.md)
- [\[ICLR 2026\] Transformers as Unsupervised Learning Algorithms: A study on Gaussian Mixtures](../../ICLR2026/learning_theory/transformers_as_unsupervised_learning_algorithms_a_study_on_gaussian_mixtures.md)
- [\[ICLR 2026\] Better Learning-Augmented Spanning Tree Algorithms via Metric Forest Completion](../../ICLR2026/learning_theory/better_learning-augmented_spanning_tree_algorithms_via_metric_forest_completion.md)

</div>

<!-- RELATED:END -->
