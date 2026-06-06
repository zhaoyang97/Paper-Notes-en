---
title: >-
  [Paper Note] Full-Spectrum Graph Neural Network: Expressive and Scalable
description: >-
  [ICML 2026][Graph Learning][Spectral GNN] This work generalizes the classical spectral GNN’s univariate eigenvalue filter $g(\lambda_i)$ to a bivariate filter $g(\lambda_i,\lambda_j)$…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Spectral GNN"
  - "Node-Pair Domain"
  - "Bivariate Filtering"
  - "Heterophilic Graphs"
  - "Local 2-GNN"
  - "Kronecker Product"
date: 2026-05-08
content_hash: 6e9e67d844bcc9ee
---

# Full-Spectrum Graph Neural Network: Expressive and Scalable

**Conference**: ICML 2026  
**arXiv**: [2605.05759](https://arxiv.org/abs/2605.05759)  
**Code**: None  
**Area**: Graph Learning / Spectral Graph Neural Networks / Expressivity Theory  
**Keywords**: Spectral GNN, Node-Pair Domain, Bivariate Filtering, Heterophilic Graphs, Local 2-GNN, Kronecker Product

## TL;DR
This work generalizes the classical spectral GNN’s univariate eigenvalue filter $g(\lambda_i)$ to a bivariate filter $g(\lambda_i,\lambda_j)$, lifting the signal from the node domain to the node-pair domain. Theoretically, it can approximate Local 2-GNN (surpassing 1-WL), and avoids explicit $n^2\times n^2$ computation via low-rank tensor decomposition, achieving strong results on node classification for heterophilic graphs and substructure counting.

## Background & Motivation

**Background**: Spectral GNNs parameterize graph convolution as Laplacian filtering $U g(\Lambda) U^\top x$, and have been proven universal for node signal approximation. However, their ability to distinguish non-isomorphic graphs (another dimension of expressivity) is strictly limited below the 1-WL test. To surpass 1-WL, spatial methods lift message passing from the node domain $V$ to the node-pair domain $V\times V$ or even $k$-tuples (e.g., higher-order GNNs by Morris et al.), but the corresponding “lifting” in spectral methods has been missing.

**Limitations of Prior Work**: (1) On heterophilic graphs, adjacent nodes often have different labels, and traditional spectral GNNs’ diagonal spectral filtering $g(L)$ struggles to learn “cross-class suppression, intra-class enhancement” convolution patterns; (2) Higher-order spatial GNNs are expressive but computationally expensive at $O(n^k)$, lacking scalability; (3) Spectral methods lack theoretical explanation for “why non-diagonal spectral components are necessary.”

**Key Challenge**: Spectral methods are naturally compact and can universally approximate node signals, but their expressivity is capped by 1-WL; higher-order spatial methods are expressive but not scalable. There is no bridge between the two lines.

**Goal**: (1) Provide the spectral GNN counterpart of “lifting to the node-pair domain” and prove it achieves Local 2-GNN level discriminative power; (2) Offer a scalable engineering implementation without explicitly constructing $n^2\times n^2$ matrices; (3) Prove that the failure of classical spectral GNNs on heterophilic graphs is a necessary consequence of “lacking non-diagonal spectral components,” and show the new method naturally remedies this.

**Key Insight**: The GFT of a node signal $x\in\mathbb{R}^V$ is $U^\top x$; for a node-pair signal $\varepsilon\in\mathbb{R}^{V\times V}$, the GFT is naturally $(U\otimes U)^\top \varepsilon$, with basis $\{u_i u_j^\top\}$, and the filter upgrades from vector $g_\lambda=(g(\lambda_i))_i$ to matrix $G_\lambda=(g(\lambda_i,\lambda_j))_{ij}$—the most natural second-order spectral generalization.

**Core Idea**: Replace the univariate spectral filter $g(\lambda_i)$ with a bivariate filter $g(\lambda_i,\lambda_j)$ as the second-order lifting for spectral methods; use low-rank tensor decomposition to compress node-pair domain computation back to the node domain.

## Method

### Overall Architecture
Input consists of node features $X\in\mathbb{R}^{n\times d_1}$ and node-pair features $E\in\mathbb{R}^{n\times n\times d_2}$. An encoder $\phi$ first lifts each node pair to $H_{uv}=\phi(X_u,X_v,E_{uv})$, reshaped as $H\in\mathbb{R}^{n^2\times d}$. This is followed by several full-spectrum convolution layers $H'=\sigma(g(L\otimes I_n, I_n\otimes L)\cdot H\cdot W)$. The final readout (node-pair / node / graph level) depends on the task. The convolution core is the parameterization of the bivariate function $g$, with three implementation routes provided; Route III (low-rank tensor decomposition) is optimal.

### Key Designs

1. **Bivariate Spectral Filtering on Node-Pair Domain (Full-Spectrum Convolution)**:

    - **Function**: Generalizes traditional spectral convolution $\sum_i g(\lambda_i)u_iu_i^\top x$ to $\sum_{i,j} g(\lambda_i,\lambda_j)u_iu_i^\top \varepsilon u_ju_j^\top$, allowing each eigenvalue pair to be modulated independently.
    - **Mechanism**: Uses Kronecker basis $\{u_i\otimes u_j\}$ as an orthogonal basis for $\mathbb{R}^{n^2}$, defines bivariate spectral filtering $G_\lambda=(g(\lambda_i,\lambda_j))_{ij}$, and writes node-pair domain convolution as $G_\lambda \ast_G \varepsilon = g(L\otimes I_n, I_n\otimes L)\varepsilon$. Proposition 3.3 shows “classical spectral GNN is a diagonal-embedded special case of full-spectrum”: restricting $g(s,t)$ to diagonal values $g(\lambda_i,\lambda_i)$ recovers $U g(\Lambda) U^\top x$. Theorem 3.4 proves linear FSpecGNN can universally approximate 1D node-pair signals; Theorem 3.8 proves the existence of a bivariate polynomial $q$ such that FSpecGNN achieves Local 2-GNN discriminative power (strictly surpassing 1-WL).
    - **Design Motivation**: The node-pair domain is the most natural lifting domain to surpass 1-WL; retaining the Kronecker basis structure preserves the interpretability of spectral methods (frequency modulation) and unlocks richer filtering patterns, including the non-diagonal components needed for heterophilic graphs.

2. **Scalable Implementation via Low-Rank Tensor Decomposition**:

    - **Function**: Avoids explicit construction of $n^2\times n^2$ Kronecker product matrices, compressing node-pair domain convolution into several polynomial spectral filters on the node domain.
    - **Mechanism**: Parameterizes $g$ with a bivariate polynomial $P(s,t)=\sum_{i+j\le K} a_{ij} s^i t^j$. The key observation (Proposition 3.9) is $P(L\otimes I_n, I_n\otimes L)=\sum_{r=1}^R f_r(L)\otimes h_r(L)$ if and only if $R\ge\mathrm{rank}(A)$, where $A=(a_{ij})$ is the coefficient matrix. A low-rank approximation of $A$ yields $\mathcal{T}_L^S\coloneqq \sum_{r=1}^S f_r(L)\otimes h_r(L)$, $S\ll \mathrm{rank}(A)$, with each $f_r,h_r$ being univariate polynomials of degree $\le K$ (e.g., BernConv, Cheb). Using $(L^p\otimes L^q)\mathrm{vec}(\varepsilon)=\mathrm{vec}(L^q \varepsilon L^p)$, Kronecker multiplication is converted to two $n\times n$ matrix multiplications, with total computation $O(SK\cdot n^2 d)$, matching first-order spectral GNNs.
    - **Design Motivation**: Directly learning $g(\lambda_i,\lambda_j)$ requires $O(n^3)$ eigendecomposition, infeasible for large graphs; polynomial parameterization expresses $g$ as a low-rank Kronecker sum, enabling all node-pair domain operations to be equivalently converted to the node domain, granting second-order spectral methods scalability comparable to first-order methods.

3. **“Necessity Proof” for Non-Diagonal Spectral Components from Heterophilic Graphs**:

    - **Function**: Theoretically answers the long-standing question of whether non-diagonal spectral components are redundant, and demonstrates that FSpecGNN can realize the optimal convolution unattainable by classical spectral GNNs.
    - **Mechanism**: Under a simplified model of “class-conditional features + intra-class compression,” defines class MSE $\mathcal{L}(C)=\sum_a \frac{1}{n_a}\sum_{p\in V_a}\mathbb{E}\|Y_p-m_a\|_2^2$, and proves (Theorem 4.1) that the optimal convolution $C^*$ is asymptotically “block-diagonal by class”—intra-class weights $1/(n_a+\tau_a)$, inter-class zero. Further (Theorem 4.2), if $C=g(L)$ is any classical spectral filter with zero cross-class entries, then $C=\alpha I_n$, i.e., classical spectral GNNs cannot approximate such optimal operators. FSpecGNN, via full-spectrum convolution with minor modification, can realize this optimal operator, explicitly characterizing heterophilic graphs as “essentially second-order phenomena.”
    - **Design Motivation**: Elevates the necessity of the method from “engineering experience” to “impossibility under algebraic constraints,” providing first-hand theoretical support for the superiority of second-order spectral architectures on heterophilic graphs.

### Loss & Training
Supervised training (cross-entropy for node classification, MAE for substructure counting), with three spectral backbone options: FSpecGNN(Cheb) / (ChebII) / (Bern). Low-rank parameter $S$ and polynomial degree $K$ are selected via validation; for small graphs, Route I (explicit eigendecomposition + MLP $g_\theta$) can be used directly.

## Key Experimental Results

### Main Results

**Heterophilic Graph Node Classification** (higher is better):

| Model | Chameleon | Squirrel | Tolokers | Questions | Wisconsin |
|-------|-----------|----------|----------|-----------|-----------|
| ChebNetII | 33.48 | 30.80 | 69.37 | 63.99 | 41.33 |
| GPRGNN | 30.44 | 24.33 | 67.05 | 53.76 | 40.79 |
| BernNet | 29.45 | 25.94 | 69.31 | 65.41 | 49.33 |
| **FSpecGNN(Cheb)** | 33.09 | **39.57** | 76.89 | 75.87 | 49.87 |
| **FSpecGNN(ChebII)** | **39.60** | 37.70 | 76.37 | **77.00** | 50.00 |
| **FSpecGNN(Bern)** | 37.91 | 37.59 | 74.50 | 77.11 | **54.58** |

All three variants of this work raise SOTA on Squirrel from 30.80 to 39.57 (+8.77), and on Questions by +11.6 points, clearly outperforming corresponding first-order spectral baselines on all heterophilic datasets, validating the theoretical conclusions of Theorem 4.1/4.2.

### Ablation Study

| Configuration | Substructure Count MAE | Heterophilic Graph Classification | Notes |
|---------------|-----------------------|-----------------------------------|-------|
| FSpecGNN (full, $S$=low-rank) | lowest | highest | Full model |
| Degraded to diagonal ($g(s,t)=h(s+t)$) | significantly higher | close to BernNet | Equivalent to first-order spectral on Kronecker sum, verifies lifting must retain non-diagonal |
| No low-rank approx. ($S=\mathrm{rank}(A)$) | slightly lower than full | similar | Full-rank slightly better but GPU memory increases 5-10× |
| Replaced with spatial 2-GNN | similar | slightly lower | Comparable expressivity, but 5× slower runtime, highest memory |

### Key Findings
- On substructure (chordal cycle) counting, FSpecGNN matches the expressivity of spatial Local 2-GNN, but with ~5× lower runtime and lowest peak GPU memory—achieving “second-order expressivity at first-order cost.”
- The datasets with the largest gains on heterophilic graphs (Squirrel/Questions) also have the highest heterophily $h(G)$, consistent with Figure 3’s observation that “off-diagonal energy increases with heterophily,” aligning mechanism and phenomenon.
- Low-rank $S$ is a key hyperparameter: too small (e.g., 1) degenerates to diagonal solutions, too large loses efficiency; empirically, $S=4\sim 8$ achieves 95%+ of full-rank performance on most tasks.

## Highlights & Insights
- Provides the “correct lifting” for spectral GNNs—node-pair domain + Kronecker basis—brilliantly bridging the long-standing dichotomy of “spectral vs spatial methods”: matches Local 2-GNN in expressivity, while retaining the sparse polynomial form of spectral methods in engineering.
- The framing “heterophilic graphs are essentially second-order phenomena” is theoretically powerful; Theorem 4.2 turns the phenomenon (GCN collapse on heterophilic graphs) into an algebraic impossibility, a deeper explanation than previous “high/low-pass separation.”
- The low-rank tensor trick $(L^p\otimes L^q)\mathrm{vec}(\varepsilon)=\mathrm{vec}(L^q \varepsilon L^p)$ compresses second-order convolution to $n\times n$ matrix multiplication, and can be directly applied to other graph algorithms needing Kronecker operations (e.g., multi-view spectral clustering, Sinkhorn-Knopp on graphs).
- Simultaneously achieves universal approximation (Theorem 3.4) and discriminative power (Theorem 3.8) for node-pair domain, two previously orthogonal expressivity dimensions, with a clean structure.

## Limitations & Future Work
- The authors acknowledge Theorem 3.8 is an “existence” rather than “learnability” result—the polynomial $q$ exists but optimizers may not always find it, so empirical expressivity may be slightly stronger or weaker than Local 2-GNN.
- Self-assessment: While $E\in\mathbb{R}^{n\times n\times d_2}$ can be sparsified on sparse graphs, memory is still $O(n^2 d)$, so feasibility for million-node graphs (OGB-LSC, Reddit) requires more sparse implementation details.
- The paper only evaluates node classification and substructure counting, lacking results for link prediction, graph-level regression, and other tasks relying on node-pair signals.
- $S$ is currently selected via validation; future work could make $S$ data-adaptive (similar to NAS or structural sparsification).
- No discussion of comparison with recent GNN+attention hybrid architectures (e.g., GraphGPS); dual lifting with spectral+attention is an obvious next step.

## Related Work & Insights
- **vs Local 2-GNN (zhangbeyond)**: This work is its spectral version, achieving node-pair domain discriminative power via spectral filters; advantage is not explicitly traversing node pairs, thus lower cost; disadvantage is “existence” does not guarantee “always learnable.”
- **vs BernNet / ChebNetII / GPRGNN**: All are first-order spectral polynomial filters; FSpecGNN treats them as “diagonal-embedded special cases,” and can use the same spectral bases (Bern/Cheb) to construct second-order filters via $f_r,h_r$.
- **vs Heterophilic Graph Methods (H2GCN, GBK-GNN)**: Those rely on local heuristics (distinguishing self/neighbor/distant), while FSpecGNN provides a more general theoretical explanation for the necessity of non-diagonal spectral components, and naturally achieves cross-class suppression convolution in spectral form.
- **Transferable Insights**: (1) Kronecker spectral basis + low-rank tensor decomposition is a general engineering template for any model “wanting second-order but fearing complexity explosion” (including attention, Transformer); (2) The reasoning chain of “optimal operator–algebraic constraint–architecture choice” is a neat paradigm for establishing necessity of new architectures.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First clean spectral formulation of node-pair domain lifting, with necessity theorems; strong theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive win on heterophilic node classification, matches spatial 2-GNN on substructure counting and is faster; lacks link prediction and large-scale experiments.
- Writing Quality: ⭐⭐⭐⭐ Rigorous formula derivations, clear propositions and theorems; some lemma/notation density is high, requiring spectral graph background.
- Value: ⭐⭐⭐⭐ Provides the community with a “scalable second-order spectral GNN baseline” and a new theoretical perspective on heterophilic graphs, with both practical and theoretical impact.

## Related Papers

- [\[ICLR 2026\] A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](../../ICLR2026/graph_learning/a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)
- [\[ICML 2025\] Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification](../../ICML2025/graph_learning/mitigating_over-squashing_in_graph_neural_networks_by_spectrum-preserving_sparsi.md)
- [\[NeurIPS 2025\] GraphFaaS: Serverless GNN Inference for Burst-Resilient, Real-Time Intrusion Detection](../../NeurIPS2025/graph_learning/graphfaas_serverless_gnn_inference_for_burst-resilient_real-time_intrusion_detec.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](../../ICML2025/graph_learning/hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)
- [\[NeurIPS 2025\] GnnXemplar: Exemplars to Explanations -- Natural Language Rules for Global GNN Interpretability](../../NeurIPS2025/graph_learning/gnnxemplar_exemplars_to_explanations_--_natural_language_rules_for_global_gnn_in.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] On Stealing Graph Neural Network Models](../../AAAI2026/graph_learning/on_stealing_graph_neural_network_models.md)
- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)
- [\[ACL 2026\] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](../../ACL2026/graph_learning/logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)
- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](quantile-free_uncertainty_quantification_in_graph_neural_networks.md)
- [\[ICLR 2026\] On the Expressive Power of GNNs for Boolean Satisfiability](../../ICLR2026/graph_learning/on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)

</div>

<!-- RELATED:END -->
