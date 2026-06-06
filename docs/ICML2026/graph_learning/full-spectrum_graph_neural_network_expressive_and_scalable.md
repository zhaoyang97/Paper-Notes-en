---
title: >-
  [Paper Note] Full-Spectrum Graph Neural Network: Expressive and Scalable
description: >-
  [ICML 2026][Graph Learning][Spectral GNNs] This paper generalizes the univariate eigenvalue filter $g(\lambda_i)$ of classic spectral GNNs to a bivariate filter $g(\lambda_i, \lambda_j)$…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Spectral GNNs"
  - "Node-pair Domain"
  - "Bivariate Filtering"
  - "Heterophilic Graphs"
  - "Local 2-GNN"
  - "Kronecker Product"
date: 2026-05-08
content_hash: 163b726307b34ca1
---

# Full-Spectrum Graph Neural Network: Expressive and Scalable

**Conference**: ICML 2026  
**arXiv**: [2605.05759](https://arxiv.org/abs/2605.05759)  
**Code**: None  
**Area**: Graph Learning / Spectral Graph Neural Networks / Expressivity Theory  
**Keywords**: Spectral GNNs, Node-pair Domain, Bivariate Filtering, Heterophilic Graphs, Local 2-GNN, Kronecker Product

## TL;DR
This paper generalizes the univariate eigenvalue filter $g(\lambda_i)$ of classic spectral GNNs to a bivariate filter $g(\lambda_i, \lambda_j)$, lifting signals from the node domain to the node-pair domain. Theoretically, this approach can approximate Local 2-GNN (surpassing 1-WL). By utilizing low-rank tensor decomposition, it avoids explicit $n^2 \times n^2$ computations and achieves strong results in heterophilic node classification and substructure counting.

## Background & Motivation

**Background**: Spectral GNNs parameterize graph convolutions as Laplacian filtering $U g(\Lambda) U^\top x$. Although proven to be universal in approximating node signals, their ability to distinguish non-isomorphic graphs (another dimension of expressivity) is strictly bounded by the 1-WL test. To break the 1-WL barrier, spatial methods lift message passing from the node domain $V$ to the node-pair domain $V \times V$ or even $k$-tuples (e.g., higher-order GNNs). However, a corresponding "lifting" for spectral methods has been missing.

**Limitations of Prior Work**: (1) On heterophilic graphs, where adjacent nodes often have different labels, the diagonal spectral filter $g(L)$ of traditional spectral GNNs struggles to learn convolution patterns such as "cross-class suppression and intra-class enhancement"; (2) High-order spatial GNNs, while expressive, suffer from $O(n^k)$ complexity and poor scalability; (3) Spectral methods lack theoretical explanations regarding why off-diagonal spectral components are necessary.

**Key Challenge**: Spectral methods are naturally compact and can universally approximate node signals but are limited by 1-WL; spatial higher-order methods are expressive but non-scalable. There is no bridge between these two paradigms.

**Goal**: (1) Formulate a spectral counterpart for "lifting to the node-pair domain" and prove it reaches Local 2-GNN level discriminative power; (2) Provide a scalable implementation that avoids explicit construction of $n^2 \times n^2$ matrices; (3) Prove that the failure of classic spectral GNNs on heterophilic graphs is an inevitable consequence of missing diagonal spectral components and demonstrate how the new method naturally fixes this.

**Key Insight**: The Graph Fourier Transform (GFT) of a node signal $x \in \mathbb{R}^V$ is $U^\top x$. Naturally, the GFT of a node-pair signal $\varepsilon \in \mathbb{R}^{V \times V}$ is $(U \otimes U)^\top \varepsilon$, corresponding to the basis $\{u_i u_j^\top\}$. The filter is thus upgraded from a vector $g_\lambda = (g(\lambda_i))_i$ to a matrix $G_\lambda = (g(\lambda_i, \lambda_j))_{ij}$—representing the most natural second-order spectral generalization.

**Core Idea**: Replace the univariate spectral filter $g(\lambda_i)$ with a bivariate filter $g(\lambda_i, \lambda_j)$ as a second-order lifting for spectral methods, and utilize low-rank tensor decomposition to compress node-pair domain computations back to the node domain.

## Method

### Overall Architecture
The input consists of node features $X \in \mathbb{R}^{n \times d_1}$ and node-pair features $E \in \mathbb{R}^{n \times n \times d_2}$. An encoder $\phi$ first lifts each node pair to $H_{uv} = \phi(X_u, X_v, E_{uv})$, reshaped into $H \in \mathbb{R}^{n^2 \times d}$. This is followed by several full-spectrum convolutional layers $H' = \sigma(g(L \otimes I_n, I_n \otimes L) \cdot H \cdot W)$. Finally, a readout (node-pair / node / graph level) is selected based on the task. The core of the convolution lies in the parameterization of the bivariate function $g$, for which the paper provides three implementation routes, with Route III (low-rank tensor decomposition) being the most optimal.

### Key Designs

1.  **Bivariate Spectral Filtering on the Node-Pair Domain (Full-Spectrum Convolution)**:
    - **Function**: Generalizes traditional spectral convolution $\sum_i g(\lambda_i) u_i u_i^\top x$ to $\sum_{i,j} g(\lambda_i, \lambda_j) u_i u_i^\top \varepsilon u_j u_j^\top$, allowing each pair of eigenvalues to be modulated independently.
    - **Mechanism**: Specifically uses the Kronecker basis $\{u_i \otimes u_j\}$ as an orthogonal basis for $\mathbb{R}^{n^2}$. Node-pair convolution is defined as $G_\lambda \ast_G \varepsilon = g(L \otimes I_n, I_n \otimes L)\varepsilon$. Proposition 3.3 establishes consistency, showing classic spectral GNNs are a "diagonal embedding" case of full-spectrum GNNs: when $g(s,t)$ only takes diagonal values $g(\lambda_i, \lambda_i)$, it reverts exactly to $U g(\Lambda) U^\top x$. Theorem 3.4 proves linear FSpecGNN can universally approximate 1D node-pair signals; Theorem 3.8 proves that there exists a bivariate polynomial $q$ such that FSpecGNN reaches Local 2-GNN discriminative power (strictly exceeding 1-WL).
    - **Design Motivation**: The node-pair domain is the most natural lifting domain for "going beyond 1-WL." Retaining the Kronecker basis structure inherits the interpretability of spectral methods (frequency domain modulation) while unlocking richer filtering patterns, including the off-diagonal components required for heterophilic graphs.

2.  **Scalable Implementation via Low-Rank Tensor Decomposition**:
    - **Function**: Avoids explicit construction of the $n^2 \times n^2$ Kronecker product matrix by compressing node-pair convolution into several polynomial spectral filters on the node domain.
    - **Mechanism**: Parameterizes $g$ using a bivariate polynomial $P(s,t) = \sum_{i+j \le K} a_{ij} s^i t^j$. The key observation (Proposition 3.9) is that $P(L \otimes I_n, I_n \otimes L) = \sum_{r=1}^R f_r(L) \otimes h_r(L)$ if and only if $R \ge \mathrm{rank}(A)$, where $A = (a_{ij})$ is the coefficient matrix. Applying low-rank approximation to $A$ yields $\mathcal{T}_L^S \coloneqq \sum_{r=1}^S f_r(L) \otimes h_r(L)$, where $S \ll \mathrm{rank}(A)$ and each $f_r, h_r$ are univariate polynomials of degree $\le K$ (e.g., BernConv, Cheb). Using the property $(L^p \otimes L^q) \mathrm{vec}(\varepsilon) = \mathrm{vec}(L^q \varepsilon L^p)$, the Kronecker multiplication is converted into two $n \times n$ matrix multiplications. Total complexity is $O(SK \cdot n^2 d)$, which is the same order as first-order spectral GNNs.
    - **Design Motivation**: Directly learning $g(\lambda_i, \lambda_j)$ requires an $O(n^3)$ eigendecomposition, which is infeasible for large graphs. Polynomial parameterization writes $g$ as a low-rank Kronecker sum, allowing all node-pair domain operations to be equivalently transformed back to the node domain, granting second-order spectral methods scalability comparable to first-order methods for the first time.

3.  **Necessity Proof of Off-Diagonal Spectral Components via Heterophily**:
    - **Function**: Theoretically addresses the long-standing question of whether off-diagonal spectral components are redundant and demonstrates that FSpecGNN can implement optimal convolutions impossible for classic spectral GNNs.
    - **Mechanism**: Under a simplified model of "class-conditional features + intra-class compression," the class square error is defined as $\mathcal{L}(C) = \sum_a \frac{1}{n_a} \sum_{p \in V_a} \mathbb{E} \|Y_p - m_a\|_2^2$. Theorem 4.1 proves that the optimal convolution $C^*$ is asymptotically "block-diagonal by class"—with intra-class weights $1/(n_a + \tau_a)$ and inter-class weights of 0. Theorem 4.2 further proves that if $C = g(L)$ is any classic spectral filter satisfying zero cross-class entries, it must hold that $C = \alpha I_n$, meaning classic spectral GNNs cannot approximate this optimal operator. FSpecGNN can implement this optimal operator with minor modifications, characterizing heterophily essentially as a "second-order phenomenon."
    - **Design Motivation**: Elevates the necessity of the method from "empirical experience" to an "impossibility conclusion under algebraic constraints," providing theoretical support for the superiority of second-order spectral architectures on heterophilic graphs.

### Loss & Training
Supervised training (cross-entropy for node classification, MAE for substructure counting). Three spectral backbones are available: FSpecGNN(Cheb) / (ChebII) / (Bern). Low-rank parameter $S$ and polynomial order $K$ are selected via a validation set. For small graphs, Route I (explicit eigendecomposition + MLP $g_\theta$) can be used directly.

## Key Experimental Results

### Main Results

**Heterophilic Node Classification** (Higher is better):

| Model | Chameleon | Squirrel | Tolokers | Questions | Wisconsin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ChebNetII | 33.48 | 30.80 | 69.37 | 63.99 | 41.33 |
| GPRGNN | 30.44 | 24.33 | 67.05 | 53.76 | 40.79 |
| BernNet | 29.45 | 25.94 | 69.31 | 65.41 | 49.33 |
| **FSpecGNN(Cheb)** | 33.09 | **39.57** | 76.89 | 75.87 | 49.87 |
| **FSpecGNN(ChebII)** | **39.60** | 37.70 | 76.37 | **77.00** | 50.00 |
| **FSpecGNN(Bern)** | 37.91 | 37.59 | 74.50 | 77.11 | **54.58** |

The three variants of the proposed method increased the SOTA on Squirrel from 30.80 to 39.57 (+8.77) and by +11.6 points on Questions. All heterophilic datasets showed significant improvements over corresponding first-order spectral baselines, validating the theoretical conclusions of Theorems 4.1 and 4.2.

### Ablation Study

| Configuration | Substructure MAE | Heterophily Accuracy | Description |
| :--- | :--- | :--- | :--- |
| FSpecGNN (Full, $S$=Low rank) | Lowest | Highest | Full model |
| Degraded to Diagonal components ($g(s,t)=h(s+t)$) | Significant increase | Near BernNet | Equivalent to first-order spectral on Kronecker sum; validates off-diagonal necessity. |
| No Low-rank Approximation ($S=\mathrm{rank}(A)$) | Slightly lower than full | Comparable | Full-rank performance slightly better but GPU memory increases 5-10x. |
| Replaced with Spatial 2-GNN | Comparable | Slightly lower | Similar expressivity, but runtime is 5x slower with higher memory usage. |

### Key Findings
- On the chordal cycle substructure counting task, FSpecGNN aligns in expressivity with spatial Local 2-GNNs but is approximately 5x faster with the lowest peak GPU memory, realizing "second-order expressivity at first-order cost."
- The datasets with the largest gains (Squirrel/Questions) also have the highest heterophily $h(G)$, consistent with the observation in Figure 3 that "off-diagonal energy grows with heterophily."
- The low-rank $S$ is a critical hyperparameter: too small (e.g., $S=1$) degrades to a diagonal solution, while too large loses the efficiency advantage. Empirically, $S=4 \sim 8$ reaches over 95% of full-rank performance in most tasks.

## Highlights & Insights
- Found the "correct lifting" for spectral GNNs—node-pair domain + Kronecker basis—effectively bridging the long-standing divide between spectral and spatial methods: aligning with Local 2-GNN in expressivity while maintaining the sparse polynomial form of spectral methods.
- The framing of "heterophily as essentially a second-order phenomenon" is a strong theoretical insight. Theorem 4.2 transforms the empirical observation (GCN failing on heterophilic graphs) into an algebraic impossibility, going a layer deeper than previous "high/low-pass filtering" explanations.
- The trick of using low-rank tensor decomposition and $(L^p \otimes L^q) \mathrm{vec}(\varepsilon) = \mathrm{vec}(L^q \varepsilon L^p)$ to compress second-order convolutions back into $n \times n$ matrix multiplications is a generalizable template for other Kronecker-based graph algorithms.
- Simultaneously achieves universal approximation on the node-pair domain (Theorem 3.4) and discriminative power (Theorem 3.8), covering two usually orthogonal dimensions of expressivity with a clean structure.

## Limitations & Future Work
- The authors acknowledge that Theorem 3.8 concerns "existence" rather than "learnability"—while the polynomial $q$ exists, the optimizer may not always find it, so empirical expressivity may fluctuate relative to Local 2-GNN.
- Self-assessment: While node-pair domain input $E \in \mathbb{R}^{n \times n \times d_2}$ can be sparsified on certain graphs, memory remains $O(n^2 d)$. Feasibility on million-node graphs (OGB-LSC, Reddit) requires further sparse implementation details.
- The paper only evaluates node classification and substructure counting, lacking results for link prediction or graph-level regression.
- Selection of $S$ currently relies on validation set search; future work could make $S$ data-adaptive (analogous to NAS or structural sparsification).
- No comparison with recent hybrid GNN+attention architectures (like GraphGPS); a dual spectral+attention lifting is an obvious next step.

## Related Work & Insights
- **vs. Local 2-GNN**: This is its spectral counterpart, achieving second-order discriminative power via spectral filters. Advantage: avoids explicit node-pair traversal, lower cost. Disadvantage: "Existence" does not guarantee "learnability."
- **vs. BernNet / ChebNetII / GPRGNN**: These are first-order spectral polynomial filters. FSpecGNN views them as "diagonal embedding" special cases and can use the same spectral bases (Bern/Cheb) as $f_r$ and $h_r$ to construct second-order filters.
- **vs. Heterophily-specific methods (H2GCN, GBK-GNN)**: Those methods rely on local heuristics (distinguishing self/neighbor/far-neighbor). FSpecGNN provides a more general theoretical explanation focused on "off-diagonal spectral components."
- **Transferable Insight**: (1) Kronecker spectral basis + low-rank tensor decomposition is a general engineering template for models aiming for second-order expressivity without complexity explosion. (2) The "Optimal operator - Algebraic constraint - Architecture choice" argumentation pipeline is an elegant paradigm for justifying the necessity of new architectures.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First clean formulation of node-pair domain lifting in spectral form, supported by necessity theorems.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive wins in heterophilic node classification; alignment with spatial 2-GNN in speed/accuracy; missing link prediction and massive-scale graph experiments.
- Writing Quality: ⭐⭐⭐⭐ Rigorous derivations with clearly labeled propositions and theorems; however, density of lemmas/notations is high, requiring a strong spectral graph background.
- Value: ⭐⭐⭐⭐ Provides both a "scalable second-order spectral GNN baseline" and a new theoretical perspective on heterophily.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] On Stealing Graph Neural Network Models](../../AAAI2026/graph_learning/on_stealing_graph_neural_network_models.md)
- [\[ICML 2026\] Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design](beyond_model_base_retrieval_weaving_knowledge_to_master_fine-grained_neural_netw.md)
- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)
- [\[ACL 2026\] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](../../ACL2026/graph_learning/logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)
- [\[ICLR 2026\] On the Expressive Power of GNNs for Boolean Satisfiability](../../ICLR2026/graph_learning/on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)

</div>

<!-- RELATED:END -->
