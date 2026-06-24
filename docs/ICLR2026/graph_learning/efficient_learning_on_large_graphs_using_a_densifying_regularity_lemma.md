---
title: >-
  [Paper Note] Efficient Learning on Large Graphs using a Densifying Regularity Lemma
description: >-
  [ICLR 2026][Graph Learning][Weak Regularity Lemma] This paper proposes "Intersecting Block Graphs" (IBG)—a low-rank decomposition representing large directed sparse graphs as a superposition of $K \ll N$ intersecting bipartite blocks. It proves a "densifying" version of the Weak Regularity Lemma, ensuring that the required number of blocks $K$ depends only on the approximation precision and is independent of graph scale or sparsity. Consequently…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Weak Regularity Lemma"
  - "Low-rank Graph Approximation"
  - "Directed Graphs"
  - "Sparse Graph Densification"
  - "Weighted Cut Norm"
  - "IBG-NN"
date: 2026-05-08
content_hash: 64259c6867d42d68
---

# Efficient Learning on Large Graphs using a Densifying Regularity Lemma

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=kK7PbRzqGk](https://openreview.net/forum?id=kK7PbRzqGk)  
**Code**: [https://anonymous.4open.science/r/IBGNN](https://anonymous.4open.science/r/IBGNN)  
**Area**: Graph Learning / Large-scale Graph Neural Networks  
**Keywords**: Weak Regularity Lemma, Low-rank Graph Approximation, Directed Graphs, Sparse Graph Densification, Weighted Cut Norm, IBG-NN  

## TL;DR
This paper proposes "Intersecting Block Graphs" (IBG)—a low-rank decomposition representing large directed sparse graphs as a superposition of $K \ll N$ intersecting bipartite blocks. It proves a "densifying" version of the Weak Regularity Lemma, ensuring that the required number of blocks $K$ depends only on the approximation precision and is independent of graph scale or sparsity. Consequently, IBG-based neural networks achieve SOTA performance on node classification, spatio-temporal prediction, and knowledge graph completion with $O(N)$ (rather than $O(E)$) time and space complexity.

## Background & Motivation
- **Background**: Message Passing Neural Networks (MPNNs) are the workhorses of graph learning, but their computational and memory overhead grows linearly with the number of edges $E$. Scenarios like social networks often involve $10^8 \sim 10^9$ nodes, with edge counts $10^2 \sim 10^3$ times the node count, making MPNNs difficult to scale. Graph reduction methods (sparsification, condensation, coarsening) attempt to mitigate this, but most except sparsification fail to solve the fundamental "whole graph cannot fit in memory" problem.
- **Limitations of Prior Work**: The predecessor work ICG (Intersecting Community Graph, Finkelshtein et al. 2024a) used low-rank graph approximation to reduce complexity from $O(E)$ to $O(N)$, but suffered from two major flaws: (1) it could only handle **undirected graphs**, losing crucial directional information for tasks like weather forecasting or causal inference; (2) it used an **unweighted cut norm**, where the number of non-edges dominates edges in sparse graphs, causing approximation quality to degrade sharply with sparsity—precisely where fraud detection, recommendation, and knowledge graphs operate.
- **Key Challenge**: The cut norm provides theoretical guarantees for dense graph approximations where the rank is independent of the number of nodes, but it is dominated by non-edges in sparse graphs. Directly downweighting non-edges requires re-proving that the regularity lemma still holds under the new metric. Bridging the gap between theory (scale-independent block counts) and practical optimization (gradient descent) for **directed and sparse** general graphs remained an open problem.
- **Goal**: Construct a similarity measure capable of approximating any directed sparse graph and prove a corresponding Weak Regularity Lemma providing a bound on $K$ dependent only on error $\epsilon$, thereby designing a network architecture with $O(N)$ complexity.
- **Core Idea**: **Active downweighting of non-edges (densification) via a weighted cut norm.** By "forcing" a sparse graph to be approximated as a dense low-rank graph, the required number of blocks $K = 1/\epsilon^2$ is completely decoupled from the number of nodes and sparsity. The NP-hard cut norm minimization is elegantly transformed into a gradient-descent-friendly **weighted Frobenius norm** minimization.

## Method

### Overall Architecture
The method consists of three layers: (1) Redefining graph similarity using a **weighted cut similarity** (densification metric) to downweight non-edges; (2) Decomposing the target graph into a superposition of $K$ intersecting bipartite blocks (IBG) and proving a **Densifying Weak Regularity Lemma** to guarantee that a low-rank IBG can approximate any graph—specifically bounding the non-optimizable cut error by the optimizable weighted Frobenius error; (3) Running **IBG-NN** on the fitted IBG, where each layer performs $O(NK)$ operations. The pipeline involves "fitting the IBG once ($O(E)$), then repeatedly training downstream tasks on the low-rank representation ($O(N)$)."

```mermaid
graph LR
    A["Directed Sparse Graph A<br/>N nodes, E edges"] -->|Weighted Cut Similarity<br/>Downweights non-edges| B["Fitting Objective:<br/>Weighted Frobenius Loss"]
    B -->|Gradient Descent<br/>O(E) Once| C["K-IBG Representation<br/>C=U·diag(r)·Vᵀ<br/>K≪N"]
    C -->|Densifying Weak Regularity Lemma<br/>Guarantees cut error ≤ O(1/√K)| C
    C -->|O(NK) per layer| D["IBG-NN<br/>Source/Target Community Synthesis"]
    D --> E["Downstream Tasks:<br/>Node Classification/Spatio-temporal/KGC"]
```

### Key Designs

**1. Densifying Cut Similarity: Downweighting non-edges to make sparse graph approximation tractable.** Standard cut norms fail on sparse graphs because non-edges vastly outnumber edges, causing the error to be dominated by non-edges. The authors define a weight matrix $Q_A = e_{E,\Gamma}\mathbf{1} + (1-e_{E,\Gamma})A$, assigning weight 1 to edges and a small weight $e_{E,\Gamma}=\frac{\Gamma E/N^2}{1-(E/N^2)}$ to non-edges, controlled by hyperparameter $\Gamma$. The densifying cut similarity is $\sigma_\square(A\|B):=(1+\Gamma)\|A-B\|_{\square;Q_A}$. This weighting is normalized such that $(1+\Gamma)\|A\|_{F;Q_A}=1$, keeping the similarity naturally at $O(1)$: bad approximations are near 1, while good ones are $\ll 1$. It recovers the standard cut metric when $\Gamma=N^2/E-1$. Downweighting non-edges results in an approximation that "densifies" the connectivity patterns, which benefits downstream node classification.

**2. Intersecting Block Graph (IBG): Low-rank bipartite block superposition for directed graphs.** A $K$-IBG consists of $K$ pairs of node communities $(U_j, V_j)$, each defining a weighted bipartite block connecting every node in $U_j$ to every node in $V_j$ with weight $r_j$. The adjacency matrix is written in low-rank form $C=\sum_{j=1}^{K} r_j \mathbf{1}_{U_j}\mathbf{1}_{V_j}^\top = U\,\mathrm{diag}(r)\,V^\top$, and the signal is $P=UF+VB$. Crucially, the **source community matrix $V$ and target community matrix $U$ are decoupled**, allowing IBG to represent directivity (unlike ICG where they are identical). For differentiable optimization, hard indicator functions $\mathbf{1}_U\in\{0,1\}^N$ are relaxed to soft membership vectors $U,V\in[0,1]^{N\times K}$ via Sigmoid.

**3. Densifying Weak Regularity Lemma: Bounding NP-hard cut error with optimizable Frobenius error.** Directly minimizing cut similarity is an NP-hard min-max problem. Theorem 4.1 provides a "semi-constructive" path: defining $\eta_k=(1+\delta)\min_{(C,P)\in[Q]^k}\|(A,X)-(C,P)\|^2_{F;Q_A,\dots}$ as the optimal weighted Frobenius error for $k$ blocks, any near-optimal Frobenius approximation $(C^*,P^*)$ satisfies $\sigma_\square\big((A,X)\|(C^*,P^*)\big)\le(\sqrt{\alpha(1+\Gamma)}+\sqrt{\beta})\sqrt{\tfrac{\eta_m-\eta_{m+1}}{1+\delta}}$. When $m$ is sampled from $[K]$, with probability $1-1/R$, $\sigma_\square\le\sqrt{2+\Gamma}\sqrt{\tfrac{\delta+R(1+\delta)}{K}}$. **The critical conclusion is that this bound depends only on $K$ and is independent of $N$ and sparsity.** This $K=O(1/\epsilon^2)$ requirement is the theoretical root of "densification." This lemma extends prior work by supporting directed IBGs and providing a scale-decoupled bound.

**4. Efficient Fitting via Gradient Descent: Reducing $O(N^2)$ to $O(K^2N+KE)$.** Computing $\|A-U\mathrm{diag}(r)V^\top\|^2_{F;Q_A}$ directly takes $O(N^2)$. Proposition 4.2 expands the loss into three terms: $\|A\|^2_{F;Q_A}$, a low-rank term $\mathrm{Tr}\big((V^\top V)\mathrm{diag}(r)(U^\top U)\mathrm{diag}(r)\big)$, and a sparse correction term summed only over the edges $\mathcal{N}(i)$. This **leverages both the sparsity of $A$ and the low-rank nature of $U\mathrm{diag}(r)V^\top$**, reducing time to $O(K^2N+KE)$ and space to $O(KN+E)$. Fitting is accelerated with SVD initialization, and randomized SVD with subgraph SGD sampling is provided for memory-constrained scenarios. IBG-NN layers then process signals via synthesis ($F\mapsto UF$, $B\mapsto VB$) in $O(NK)$.

## Key Experimental Results

### Main Results (Directed Node Classification, ROC AUC / Accuracy)

| Model | Squirrel | Chameleon | Tolokers |
|---|---|---|---|
| GCN | 53.43 | 64.82 | 83.64 |
| GAT | 40.72 | 66.82 | 83.70 |
| DirGNN (Directed-specific) | 75.13 | 79.74 | – |
| FaberNet | 76.71 | 80.33 | – |
| ICG-NN (Prior Work) | 64.02 | 63.9 | 83.73 |
| **IBG-NN** | **77.63** | **80.15** | **83.76** |

IBG-NN outperforms GNNs specifically designed for directed graphs that require quadratic complexity and significantly leads over the previous ICG-NN.

### Ablation Study (Decomposing Directionality vs Densification)

| Variant | Squirrel | Chameleon | Description |
|---|---|---|---|
| ICG-NN | 64.02 | 63.9 | Undirected + Unweighted |
| IBG-NN (undirected) | 70.02 | 75.15 | +Densification (+6%, +11.2%) |
| **IBG-NN** | **77.63** | **80.15** | +Directionality (additional +7.6%, +5%) |

Both densification and directionality contribute significantly. On Tolokers, gains are smaller because the graph is already dense and directional patterns are less pronounced.

### Key Findings
- **Efficiency**: Compared to DirGNN on sparse ER graphs, IBG-NN achieved $5.68\times/5.26\times$ speedups at $K=10/100$. On dense graphs, its runtime relative to DirGNN follows a square-root relationship, consistent with the $O(N)$ vs $O(E)$ theory. It remains faster even when $K=100$ exceeds the average degree 25.
- **Large Graph Condensation** (Flickr/Reddit/Arxiv/Products): The subgraph SGD version of IBG-NN reached SOTA at extremely low condensation ratios, outperforming GCOND, SFGC, and SimGC which require the full graph in GPU memory. It consistently beat ICG-NN (e.g., 92.3 vs 89.7 on Reddit at 0.1%).
- IBG fitting is a one-time $O(E)$ cost; downstream tasks can then be trained repeatedly at $O(N)$. In spatio-temporal tasks where the graph is fixed but signals vary, the efficiency advantage compounds.

## Highlights & Insights
- **Counter-intuitive Theoretical Insight**: Sparse graphs are typically the hardest to approximate. The authors take the opposite approach—actively "densifying" the graph to approximate it with a dense low-rank representation. By downweighting non-edges, the block count bound decouples from graph scale ($K=O(1/\epsilon^2)$), breaking the old notion that low-rank approximation only works for dense graphs.
- **Converting Non-optimizable to Optimizable**: While the cut norm is NP-hard, the lemma ensures that "Frobenius minimizers are also good cut approximations," allowing gradient descent to bypass the min-max problem.
- **Decoupled $U,V$ for Directionality**: Simply separating source and target community matrices elegantly extends undirected ICG to directed IBG.
- **Practical Complexity**: The $O(K^2N+KE)$ fitting and $O(NK)$ inference, coupled with IBG reusability, makes it ideal for hyperparameter sweeps ($O(SN)$ vs $O(SE)$) and temporal tasks.

## Limitations & Future Work
- **Non-convex Optimization**: IBG fitting uses gradient descent on a non-convex loss. While theory guarantees success for "near-optimal Frobenius solutions," reaching them depends on initialization (SVD helps but does not eliminate this).
- **Hyperparameter $\Gamma$ Selection**: The degree of densification is controlled by $\Gamma$. There is no deep discussion on adaptive selection across different graph sparsities.
- **Diminishing Returns on Dense Graphs**: On Tolokers, densification and directionality offered minimal gains. The method's strengths are specifically in sparse/directed scenarios.
- **Trade-off between $K$ and Expressivity**: While theory says $K=O(1)$, increasing communities in practice still improves performance. The optimal $K$ remains an empirical choice.
- **Future Work**: Extending densification to general graph kernels, temporal dynamic graphs, and validating end-to-end scalability on industrial graphs ($10^9$ nodes).

## Related Work & Insights
- **ICG** (Finkelshtein et al. 2024a): This paper is a non-trivial extension of ICG across three axes: directivity, densification, and scale-independent block counts.
- **Weak Regularity Lemma** (Frieze & Kannan 1999; Lovász 2007): Classic WRL gives a bound of $N/(\sqrt{E}\epsilon^2)$ communities (dependent on $N$ for sparse graphs). The densifying version in this paper improves this to $1/\epsilon^2$, a substantial advancement in regularity lemma theory.
- **Graph Condensation**: Unlike GCOND/SFGC, which require full graph storage, this low-rank approach proves that graph approximation can outperform condensation with lower overhead.
- **Insight**: Re-weighting at the metric level is often more fundamental than model-level patches. Downweighting non-edges simultaneously improved theoretical bounds, approximation quality, and downstream accuracy.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The densifying weak regularity lemma is a substantial theoretical contribution, and the IBG representation for directed sparse graphs is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers efficiency, node classification, condensation, and temporal tasks with ablations, though major benchmarks are slightly smaller scale; industrial-scale verification ($10^9$) is primarily theoretical.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous theory and clear motivation, though the density of theorems and weighted norm derivations poses a high barrier for non-theoretical readers.
- **Value**: ⭐⭐⭐⭐⭐ — Reducing large sparse directed graph learning from $O(E)$ to $O(N)$ while maintaining SOTA accuracy has direct practical value for large-scale recommendation and social graph systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs](tgm_a_modular_and_efficient_library_for_machine_learning_on_temporal_graphs.md)
- [\[ICLR 2026\] Global-Recent Semantic Reasoning on Dynamic Text-Attributed Graphs with Large Language Models](global-recent_semantic_reasoning_on_dynamic_text-attributed_graphs_with_large_la.md)
- [\[ICLR 2026\] Towards Quantifying Long-Range Interactions in Graph Machine Learning: A Large Graph Dataset and a Measurement](towards_quantifying_long-range_interactions_in_graph_machine_learning_a_large_gr.md)
- [\[ICLR 2026\] Contraction and Hourglass Persistence for Learning on Graphs, Simplices, and Cells](contraction_and_hourglass_persistence_for_learning_on_graphs_simplices_and_cells.md)
- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](../../AAAI2026/graph_learning/echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)

</div>

<!-- RELATED:END -->
