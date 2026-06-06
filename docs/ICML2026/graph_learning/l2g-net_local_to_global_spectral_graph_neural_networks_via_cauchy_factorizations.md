---
title: >-
  [Paper Note] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations
description: >-
  [ICML 2026][Graph Learning][Spectral GNN] The authors **exactly decompose** the Graph Fourier Transform (GFT) basis into "local GFTs for each subgraph $\times$ a sequence of Cauchy matrices…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Spectral GNN"
  - "Graph Fourier Transform"
  - "Cauchy Factorization"
  - "Hierarchical Partitioning"
  - "Long-range Dependencies"
date: 2026-05-08
content_hash: fb52ec46df22f3de
---

# L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations

**Conference**: ICML 2026  
**arXiv**: [2602.18837](https://arxiv.org/abs/2602.18837)  
**Code**: To be confirmed  
**Area**: Graph Learning / Spectral Graph Neural Networks  
**Keywords**: Spectral GNN, Graph Fourier Transform, Cauchy Factorization, Hierarchical Partitioning, Long-range Dependencies

## TL;DR
The authors **exactly decompose** the Graph Fourier Transform (GFT) basis into "local GFTs for each subgraph $\times$ a sequence of Cauchy matrices," reducing the $O(n^3)$ eigendecomposition cost to $O(kn^2)$ (where $k$ is the number of cut edges between subgraphs). By interleaving learnable spectral filters within the decomposition, they obtain a family of local-to-global spectral GNNs that can scale to 569k-node graphs with parameters orders of magnitude fewer than Transformers while achieving comparable performance.

## Background & Motivation

**Background**: Spectral GNNs process graph signals by projecting them onto the Laplacian eigenbasis (i.e., GFT), which theoretically characterizes global frequency structures. However, true GFT-based GNNs are rarely adopted in practice. The mainstream consists of polynomial Laplacian filters like ChebNet and MPNNs (Message Passing), which replace eigendecomposition with sparse Laplacian multiplications.

**Limitations of Prior Work**: Pure GFT methods face two fatal issues: the $O(n^3)$ complexity of eigendecomposition makes them unscalable beyond tens of thousands of nodes, and operations in the GFT domain are **global**, meaning a single spectral coefficient change affects all nodes, lacking the local inductive bias of the vertex domain. While polynomial/MPNN routes are inexpensive and offer $k$-hop locality, they approximate long-range dependencies through multiple message-passing steps, often triggering oversquashing and optimization instability. Graph Transformers use attention for global aggregation but suffer from massive parameter counts and loss of graph structural interpretability.

**Key Challenge**: There is a binary choice between **global spectral expressivity** and **local computation/inductive bias** in existing architectures—global expressivity requires $O(n^3)$ and loses locality, while local computation necessitates sacrificing exact spectral precision.

**Goal**: To find a spectral processing framework that **maintains exact GFT, enables local computation, and naturally incorporates local biases**, and to formulate it as stackable GNN layers.

**Key Insight**: The authors start from a mathematical observation: adding an edge to a graph is equivalent to a rank-one update of the Laplacian, and there is an **exact closed-form relationship in the form of Cauchy matrices** between the eigenbases before and after a rank-one update (Fasino 2023). By hierarchically partitioning a graph into subgraphs and "adding back" bridge edges one by one, the global GFT can be expressed as an exact product of "local subgraph GFTs $\times$ a sequence of Cauchy factors."

**Core Idea**: Use **Cauchy factorization** to rewrite the GFT as a chain structure of "block-wise GFT + local mixing," interleaving learnable spectral filters between each stage of mixing to construct a local-to-global spectral GNN.

## Method

### Overall Architecture
The input is a large graph $G$ and node features $X_0$. **Preprocessing stage** (data-independent, performed once before training): The graph $G$ is recursively partitioned into a hierarchical subgraph family $G \in \mathcal{F}(L, \{G_i\}, k)$ via greedy spectral bisection. Local GFT bases $U_i$ are computed for each leaf subgraph, and Cauchy factors $D_{r,p}$ are computed and cached by sequentially applying rank-one updates corresponding to all bridge edges. **Forward stage**: Starting from leaf subgraph features, each level $r$ first applies a learnable spectral filter $g_{r,p}(\lambda_{r,p})$ in the current subgraph's frequency domain, then multiplies it by a Cauchy factor $D_{r,p}$ to mix the spectra of two subgraphs. Finally, a global spectral filter $g_\theta(\lambda)$ is applied at the root level, followed by the inverse decomposition to return to the node domain. The entire process **never explicitly constructs the full GFT matrix $U$**.

### Key Designs

1.  **Cauchy Factorization of GFT (Theorem 3.1)**:
    - **Function**: Decomposes the GFT basis $U^\top$ of any graph $G \in \mathcal{F}(L, \{G_i\}, k)$ exactly as $U^\top = D(\lambda, \tilde\lambda_{K-1}) \cdots D(\tilde\lambda_1, \tilde\lambda_0) U_0^\top$, where $U_0$ is a block-diagonal matrix of subgraph GFTs and $D(\cdot,\cdot)$ are Cauchy factors.
    - **Mechanism**: Per Proposition 2.1, adding an edge $(i,j)$ is equivalent to a rank-one update $\tilde L = L + w_{ij}(e_i-e_j)(e_i-e_j)^\top$. The eigenbases before and after the update satisfy $\tilde U^\top = D(\tilde\lambda, \lambda) U^\top$, where $D$ is an Orthogonal Cauchy-like Matrix (OCLM) constructed from the new eigenvalues $\tilde\lambda$ obtained by solving the secular equation. Starting from the block-wise graph and adding back all $K = k(2L-1)$ bridge edges sequentially yields the global GFT. The authors extend Fasino 2023 to general cases **allowing multiple eigenvalues and degenerate updates** (using deflation in Definition 3.1 to select valid subspaces).
    - **Design Motivation**: Directly computing $U$ takes $O(n^3)$, but Cauchy factors are block-diagonal (each update affects only spectral components of the two involved subgraphs). Thus, the decomposition can be computed in $O(kn^2)$ (Theorem 3.2), where $k$ is the maximum number of cut edges between subgraphs. This rewrites a "global transform" as a "local update chain," serving as the mathematical cornerstone for the GNN design.

2.  **Greedy HGF Construction + Spectral Sparsification**:
    - **Function**: Finds a hierarchical partition that minimizes Cauchy factorization costs for any graph; when no such partition is found, it uses spectral sparsification to reduce bridge edges.
    - **Mechanism**: Candidate partitions are generated via spectral bisection using Fiedler vectors. A partition is accepted only if $n^2 k + \max_i f(G_i) < f(G)$ (where $f$ is the theoretical eigendecomposition cost); otherwise, the subgraph is marked as a leaf. If too many cut edges at a certain level violate this condition, Spielman-Srivastava spectral sparsification (Theorem 4.1) is applied to the **bridge edge subgraph** of that level, compressing cut edges to $O(\varepsilon^{-2})$ while guaranteeing $(1-\varepsilon) x^\top L x \le x^\top L' x \le (1+\varepsilon) x^\top L x$ with $O(|E| \log n)$ construction cost.
    - **Design Motivation**: The complexity in Theorem 3.2 is dominated by $k$ rather than $|E|$, so "balanced partitions with few cut edges" are critical. The greedy criterion links the decision to the theoretical cost function to avoid blind recursion, while sparsification provide a **worst-case fallback with controlled error**—for Lipschitz spectral filters, the layer output deviation is only $O(\varepsilon)$, generalizing the Cauchy framework from modular graphs to arbitrary graphs.

3.  **L2G-Net Spectral GNN Architecture**:
    - **Function**: **Interleaves** learnable spectral filters into each stage of the Cauchy factorization, resulting in a family of stackable layers with nonlinearity and residuals that fully cover the performance of standard spectral GNNs.
    - **Mechanism**: The $m$-th layer computes $X_{m+1,c} = U g_{\theta,c}(\lambda) U^\top Z_{m,c}$ ($Z_m = X_m W_m$), where the forward transform $U^\top$ is replaced by the recurrence $H_p^{(r)} = g_{r,p}(\lambda_{r,p}) D_{r,p} [H_{p_l}^{(r-1)}; H_{p_r}^{(r-1)}]^\top$. This mixes the spectra of left and right subgraphs via $D_{r,p}$ and immediately processes them with a learnable filter $g_{r,p}$ based on that subgraph's eigenvalues. The partitioning and factors are entirely determined by $G$; **learnable parameters only appear in $g_{r,p}$**, and the architecture is fixed by the graph structure before training.
    - **Design Motivation**: Standard global filters $g(L)$ apply the same frequency response to all nodes. L2G-Net uses different responses across subgraphs before mixing. Theorem 5.1 proves this strictly contains the $g(L)$ class. This corresponds to the "local-to-global interpolation" shown in Figure 2—where the energy decay shape over hop distance is determined by the graph structure, falling between ChebNet's hard truncation and GFT's global diffusion. Compared to Graph Transformers, structural inductive biases are encoded directly into the computation graph without relying on positional encodings, naturally reducing parameter counts by orders of magnitude.

### Loss & Training
Standard losses for each benchmark are used (cross-entropy/AUC for node classification, MAE for graph regression). Filters are parameterized using splines. In heterophilous graph experiments, **all feature channels and layers share the same filter** to maximize parameter efficiency. Preprocessing (HGF partitioning + Cauchy factor computation) is performed once and cached.

## Key Experimental Results

### Main Results

**Node Classification on Heterophilous Graphs** (Platonov 2023 large-scale benchmark; Acc for Roman-empire, AUC for Tolokers/Minesweeper):

| Dataset | Metric | L2G-Net | Polynormer (Prev. SOTA, attention) | St-ChebNet (Prev. SOTA, spectral) | Global GFT |
|--------|-------|---------|---------|---------|---------|
| Roman-empire | Acc | 92.12 (1.1) | **92.55** (0.3) | 92.03 (0.9) | — |
| Amazon-ratings | Acc | 53.39 (0.6) | **54.81** (0.5) | 53.15 (0.2) | — |
| Minesweeper | AUC | **97.50** (0.3) | 97.46 (0.4) | 95.71 (2.3) | — |
| Tolokers | AUC | 85.57 (0.6) | **85.91** (0.7) | 85.55 (3.4) | — |

**Long-range Large-scale Graphs** (Liang 2025 City-Networks, $>10^5$ nodes, runtime in minutes):

| Dataset | ED (Measured) | ED Extrapolated (cubic) | CF (Ours) | Speedup |
|--------|----------|----------|-----------|--------|
| Paris | OOM | 50,412 min | **17.91 min** | ~3000× |
| Shanghai | OOM | 121,932 min | **45.61 min** | ~2700× |
| LA | OOM | 131,544 min | **61.53 min** | ~2100× |
| London (569k) | OOM | 1,632,123 min | **144.22 min** | ~11000× |

### Ablation Study

**Memory Consumption (GB, 64-bit)**:

| Config | Mines. | Tolok. | Am.Rat. | R.Emp. | Paris | London |
|------|--------|--------|---------|--------|-------|--------|
| Full GFT | 0.80 | 1.11 | 4.80 | 4.11 | 103.97 | **2588.22** |
| CF (ours) | 0.20 | 0.28 | 1.21 | 1.03 | 12.21 | **40.96** |
| Saving | 4× | 4× | 4× | 4× | 8.5× | **63×** |

**Factorization vs. Eigendecomposition** (Platonov 2023, seconds): CF takes 232 s vs. ED 731 s on R.Emp; 281 s vs. 897 s on Am.Rat. In the worst case, Tolokers (dense graph, many cut edges after sparsification), it still achieves 91 s vs. 118 s.

**LRGB Long-range Inductive Benchmarks** (peptides-func AP↑ / peptides-struct MAE↓): L2G-Net achieves 72.14 / 0.2479 (+PE 72.46 / 0.2462), within one standard deviation of or exceeding GRIT (69.88 / 0.2460), Graph ViT (69.42 / 0.2449), MP-SSM (69.93 / 0.2458), and GMN (70.71 / 0.2473).

### Key Findings
- **Theoretical complexity is precisely confirmed by measurements**: On synthetic BA graphs with fixed $k=5$, CF follows $O(n^2)$ strictly; with fixed $n=8000$, CF grows linearly with $k$. Preprocessing (spectral bisection + sparsification) is nearly independent of $k$, representing a constant overhead—implying that "sparsifying to limit cut edges then decomposing" is a controllable engineering strategy.
- **Parameter efficiency is the true killer feature**: Figure 5 shows L2G-Net reaches almost the same accuracy as Polynormer with orders of magnitude fewer learnable parameters and consistently outperforms Global GFT on all datasets, indicating local bias is not an "accuracy-for-efficiency trade-off" but a **genuinely useful inductive bias**.
- **Interpretability byproduct**: Grad-CAM node attribution (Figure 6) shows L2G-Net concentrates predictive importance on a few nodes, whereas Global GFT spreads importance across the whole graph, and Polynormer is the most diffuse due to learned structure via attention. This confirms L2G-Net's local bias is "visible and aligns with graph geometry."
- **Failure/Boundary Scenarios**: For extremely dense graphs like Tolokers, CF's advantage over ED is minimal (91 vs 118 seconds), suggesting the method's benefits stem from graphs having partitionable hierarchical structures. The 1.4% lag behind Polynormer on Amazon-ratings suggests pure spectral inductive bias may not be optimal for all heterophilous patterns.

## Highlights & Insights
- **Extending the mathematical fact "adding an edge = rank-one update = Cauchy rotation" to hierarchical partitioning for stackable GNN layers**—this is a rare instance of "revamping deep learning architectures with classic numerical linear algebra," which has significant pedagogical value.
- The elegance of the **graph-determined architecture with graph-independent parameters**: Different graphs automatically result in different hierarchical structures and layer counts, yet the parameter count remains minimal and shared, making it naturally suited for "one model running on many different graphs."
- **Spectral sparsification is redefined here**: Traditionally used to "approximate graphs," here it acts as a bounded-error tool to "truncate the number of updates in a Cauchy chain." This perspective can be migrated to any scenario using rank-one update chains to express global operators (e.g., low-rank approximation, Krylov subspace methods).
- Enabling exact spectral processing "equivalent to full GFT" on large graphs (569k nodes) pulls spectral methods from "theoretically beautiful but unusable" into a league where they can compete with Transformers.

## Limitations & Future Work
- The benefits **strongly depend on the graph possessing hierarchical/modular structures** with small cut edges; for completely dense or random uniform graphs, HGF cannot find good cuts, necessitating $O(\varepsilon)$ error via sparsification.
- There is still a small gap compared to the attention-based Polynormer on heterophilous benchmarks (0.43% on Roman-empire, 1.42% on Amazon-ratings), suggesting pure spectral expressivity is less flexible than learned attention in some tasks; combining attention as an additional channel within L2G-Net could be considered.
- Experiments were primarily on CPU (i9-9900K + RTX 2070); efficient GPU kernels for Cauchy factors and parallel scheduling (Theorem 3.2 already provides a parallel complexity upper bound $T_m = O(kn^2 + \max_i f_i(n))$) are not yet fully developed, leaving room for significant engineering acceleration.
- The current architecture is static regarding graph structure—handling dynamic/evolving graphs would require re-running HGF partitioning. How to **incrementally update the Cauchy factorization** (adding edges implies adding Cauchy factors) is a natural extension.

## Related Work & Insights
- **vs. ChebNet / Polynomial Filtering**: ChebNet uses $K$-order Laplacian polynomials for $K$-hop hard truncation, which is inherently local. L2G-Net can exactly recover global GFT on top of locality, handling long-range dependencies without requiring excessive depth, thus avoiding oversquashing.
- **vs. Global GFT**: Pure GFT is a special case of L2G-Net when $L=0$ (no partitioning). The proposed method is strictly broader (Theorem 5.1) and outperforms Global GFT on all benchmarks, validating that "local bias is an asset, not a loss."
- **vs. Polynormer / Graph Transformer**: The Transformer route uses attention to learn global aggregation but has massive parameters and needs positional encodings. L2G-Net encodes inductive bias via the graph structure itself, with orders of magnitude fewer parameters and better interpretability via Grad-CAM.
- **vs. Fernández-Menduiña 2025a**: That work only provided Cauchy identities for **path graph Laplacians**; this paper generalizes it to **any symmetric matrix + arbitrary hierarchical partitioning** and turns it into a trainable GNN.
- **vs. MP-SSM / State-Space GNNs**: Both try to break the locality of polynomial filters, but MP-SSM is an implicit continuous-time dynamical system. L2G-Net provides an **explicit exact decomposition**, with controllable spectral expressivity and complexity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically extending and transforming rank-one updates + Cauchy matrices into a stackable GNN layer family is rare in the spectral GNN community.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic graphs (theory validation), heterophilous benchmarks, long-range induction, and city-scale graphs. However, GPU engineering optimization and detailed ablations (fine-tuned curves for different $L$ and $\varepsilon$) are relatively lacking.
- Writing Quality: ⭐⭐⭐⭐ Mathematical narrative is clear; Theorem-Proof-Algorithm structure is complete; Figures 1/2 are intuitive. A few sections (deflation in Definition 3.1) require appendix lookups.
- Value: ⭐⭐⭐⭐⭐ Pulling "exact spectral GNNs" from $O(n^3)$ down to $O(kn^2)$ and successfully running on 569k-node graphs provides a strong positive answer to whether spectral methods remain relevant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[ICML 2026\] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)
- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](quantile-free_uncertainty_quantification_in_graph_neural_networks.md)
- [\[NeurIPS 2025\] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](../../NeurIPS2025/graph_learning/duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
