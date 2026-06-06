---
title: >-
  [Paper Note] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning
description: >-
  [ICML 2026][Graph Learning][Multilayer graphs] T-GINEE combines **CP tensor decomposition** and **Generalized Estimating Equations (GEE)** to explicitly model **cross-layer dependencies** in multilayer networks. It offer…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Multilayer graphs"
  - "Tensor decomposition"
  - "Generalized Estimating Equations (GEE)"
  - "Cross-layer dependencies"
date: 2026-05-08
content_hash: 73ad8657991064c8
---

# T-GINEE: A Tensor-Based Multilayer Graph Representation Learning

**Conference**: ICML 2026  
**arXiv**: [2605.28300](https://arxiv.org/abs/2605.28300)  
**Code**: To be confirmed  
**Area**: Graph Learning / Representation Learning  
**Keywords**: Multilayer graphs, Tensor decomposition, Generalized Estimating Equations (GEE), Cross-layer dependencies

## TL;DR
T-GINEE combines **CP tensor decomposition** and **Generalized Estimating Equations (GEE)** to explicitly model **cross-layer dependencies** in multilayer networks. It offers theoretical guarantees and superior scalability, breaking the OOM (Out-of-Memory) limitations of other tensor methods on million-node graphs such as DBLP and Stack Overflow.

## Background & Motivation

**Background**: Multilayer networks are ubiquitous in the real world (e.g., social networks with friend/colleague/family relations; biological networks with protein co-expression/physical interactions). These systems require learning low-dimensional vector representations to capture complex cross-layer relationships.

**Limitations of Prior Work**: Traditional graph embedding methods (GNNs, Matrix Factorization) mostly **process layers independently** (ignoring inter-layer correlations) or use **simple aggregation strategies** (causing information loss) in multilayer settings. They lack a rigorous theoretical foundation to characterize how embeddings should encode subtle mutual influences between layers.

**Key Challenge**: Current multilayer graph learning lacks a systematic theoretical framework to **explicitly model cross-layer residual dependencies**.

**Goal**: Design a statistical framework for multilayer graph learning that possesses both theoretical guarantees and computational efficiency.

**Key Insight**: The **GEE framework** from biostatistics can model correlated data in a principled way, while **CP tensor decomposition** can elegantly represent multilayer graph structures. Combining these two allows for the explicit capture of inter-layer covariance.

**Core Idea**: Parameterize the multilayer graph parameter tensor via **CP tensor decomposition** and jointly model intra-layer and cross-layer dependencies through a **tensor-based GEE framework**.

## Method

### Overall Architecture
(1) Decompose the multilayer adjacency tensor $\mathcal{A}_{n \times n \times M}$ into a CP form consisting of node embeddings $\alpha_{n \times R}$ and layer-specific embeddings $\beta_{M \times R}$; (2) Obtain the marginal probabilities $\mathcal{P}$ by applying a link function $g^{-1}$ to the parameter tensor $\Theta$; (3) Solve for parameters under a **tensor-based GEE framework** weighted by a **working covariance matrix**.

### Key Designs

1. **Symmetric CP Tensor Decomposition**:

    - **Function**: Decomposes the parameter tensor as $\Theta = \sum_{r=1}^R \alpha^{(r)} \circ \alpha^{(r)} \circ \beta^{(r)}$.
    - **Mechanism**: The symmetry ensures the representation of undirected graphs. Shared latent factors $\alpha$ capture the **common roles** of nodes across different relationship types, while layer-specific factors $\beta$ model the heterogeneity of each layer. The total parameter count is reduced from $O(n^2 M)$ to $O((n + M) R)$.
    - **Design Motivation**: Real-world entities typically interact through a few latent features across multiple relations; CP decomposition captures this sparsity while maintaining interpretability.

2. **Tensor-Based Generalized Estimating Equations (T-GEE) Framework**:

    - **Function**: Estimates the parameter vector $\gamma$ by minimizing the weighted sum of squared residuals.
    - **Mechanism**: GEE only specifies the **first two moments** (mean and covariance) of the response without assuming a full distribution. It weights residuals using a **working covariance matrix** $\Sigma^w_{i, j} = \Gamma^{1/2}_{i, j} W \Gamma^{1/2}_{i, j}$ to explicitly model cross-layer correlations.
    - **Design Motivation**: Standard unweighted least squares ignore inter-layer correlations, leading to inefficient estimation. The GEE weighting strategy assigns more reasonable weights to correlated layers via co-variation data, improving parameter estimation efficiency.

3. **Working Covariance + Flexible Link Functions**:

    - **Function**: Estimates a shared correlation matrix $W$ from a **residual pool**. Flexible link functions $g^{-1}$ such as logit, probit, or sparsity-aware logit $g^{-1}(x) = \frac{s}{1 + e^{-x}}$ are supported.
    - **Mechanism**: Working covariance estimation allows for the misspecification of the correlation structure. Sparsity-aware link functions explicitly handle sparse networks.
    - **Design Motivation**: Practical networks are often sparse, where standard logit gradients may explode in sparse regions. The sparsity-aware design fits sparse observations naturally by introducing a coefficient $s \in (0, 1)$.

## Key Experimental Results

### Main Results

| Method | CP | Tucker | NMF | SVD | LSE | MASE | NNTUCK | SPECK | HOSVD | **T-GINEE** |
|------|-----|--------|-----|-----|-----|------|--------|-------|--------|----------|
| Synthetic AUC | 0.449 | 0.529 | 0.722 | 0.813 | 0.223 | 0.382 | 0.611 | 0.760 | 0.850 | **0.940** |

T-GINEE significantly outperforms the runner-up HOSVD (0.850). The substantial gap between the basic CP decomposition and T-GINEE (0.449 → 0.940) quantifies the benefits of the statistical regularization framework.

### Main Results (Real-world Data)

| Method | AUCS | Krackhardt | WAT | Yeast | DBLP (Million Nodes) | Stack Overflow (Million Nodes) |
|------|--------|----------|-----|-------|--------|-------------|
| HOSVD | 0.897 | 0.783 | 0.820 | 0.902 | OOM | OOM |
| SVD | 0.877 | 0.932 | 0.719 | 0.879 | 0.6093 | 0.9682 |
| **T-GINEE** | **0.920** | **0.948** | **0.838** | **0.921** | **0.6478** | **0.9831** |

CP, Tucker, and HOSVD all failed on DBLP and Stack Overflow due to OOM errors, whereas T-GINEE successfully handled these million-node graphs.

### Ablation Study & Generalization
- **Heterogeneous Synthesis**: Under low overdetermination ratios, the learned covariance matrix $W$ correctly recovers the inter-layer similarity order.
- **New Layer Generalization**: In zero-shot transfer experiments on DBLP-5K—training on the first four layers and predicting edges in the fifth—T-GINEE achieved an AUC of 0.7733, surpassing MGCN and MR-GCN even though the latter were retrained on the fifth layer.

## Highlights & Insights
- **Principled Cross-layer Modeling**: This work is the first to seamlessly combine the GEE framework (a standard tool in biostatistics) with tensor decomposition, providing a rigorous statistical foundation for multilayer graph learning.
- **Double Theoretical Guarantees**: Theorem 3.1 proves $\sqrt{N}$ consistency, and Theorem 3.2 establishes asymptotic normality.
- **Breakthrough Scalability**: The sparse tensor implementation and mini-batch sampling reduce complexity from $O(n^2 M)$ to $O(R |E|)$, maintaining competitiveness on million-node graphs.
- **Inter-layer Knowledge Transfer**: The learned $\alpha$ and $\beta$ decompositions naturally support **zero-shot cross-layer generalization**.

## Limitations & Future Work
- The theoretical analysis (Theorem 3.2) requires overdetermination conditions, which limits the applicability of asymptotic normality in extremely sparse scenarios.
- There is currently no data-driven method for the optimal selection of rank $R$ and the sparsity coefficient $s$.
- Layer Heterogeneity: The assumption of fully shared node embeddings $\alpha$ across layers may be too strong for completely heterogeneous relationship types.
- Future Work: Introduce hierarchical or soft-sharing mechanisms; develop provable mini-batch theory; design adaptive selection for rank and sparsity coefficients.

## Related Work & Insights
- **vs. Traditional Tensor Decomposition (CP/Tucker)**: Traditional methods assume edge independence without covariance modeling; T-GINEE explicitly learns a working covariance.
- **vs. Multilayer GNNs (MGCN/MR-GCN)**: GNNs rely on local neighborhood aggregation via graph convolution, while T-GINEE adopts a global tensor perspective and statistical inference.
- **Insight**: The combination of the GEE paradigm and tensor decomposition can be generalized to other multivariate data scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  The integration of GEE and tensor decomposition introduces a new theoretical paradigm for multilayer graphs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Comprehensive evaluation across synthetic data, small benchmarks, and million-node graphs compared against strong GNN baselines.
- Writing Quality: ⭐⭐⭐⭐⭐  Clear motivation, rigorous methodological presentation, and well-defined experimental conclusions.
- Value: ⭐⭐⭐⭐⭐  Fills a theoretical gap in multilayer graph learning, advancing both statistical frameworks and scalable designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] View Space: Representation Learning Across Arbitrary Graphs](view_space_learning_representation_across_arbitrary_graphs.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[ICML 2026\] Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs](unsat_core_prediction_through_polarity-aware_representation_learning_over_clause.md)
- [\[AAAI 2026\] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction](../../AAAI2026/graph_learning/unihr_hierarchical_representation_learning_for_unified_knowledge_graph_link_pred.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](../../AAAI2026/graph_learning/feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)

</div>

<!-- RELATED:END -->
