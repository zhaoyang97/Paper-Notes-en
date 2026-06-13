---
title: >-
  [Paper Note] AvAtar: Learning to Align via Active Optimal Transport
description: >-
  [ICML 2026][3D Vision][Optimal Transport] This paper proposes AvAtar, an active alignment framework based on Optimal Transport (OT). It quantifies the impact of candidate queries on global alignment results via gradient…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Optimal Transport"
  - "Active Learning"
  - "Alignment"
  - "Gradient Propagation"
  - "Conjugate Gradient Method"
date: 2026-05-08
content_hash: 66af9fbca476575c
---

# AvAtar: Learning to Align via Active Optimal Transport

**Conference**: ICML 2026  
**arXiv**: [2605.24395](https://arxiv.org/abs/2605.24395)  
**Code**: None  
**Area**: Machine Learning Theory  
**Keywords**: Optimal Transport, Active Learning, Alignment, Gradient Propagation, Conjugate Gradient Method  

## TL;DR

This paper proposes AvAtar, an active alignment framework based on Optimal Transport (OT). It quantifies the impact of candidate queries on global alignment results via gradient propagation and utilizes the adjoint state method and conjugate gradient method to achieve efficient solutions with linear complexity. AvAtar consistently outperforms existing active learning strategies in network alignment and cross-domain alignment tasks.

## Background & Motivation

**Background**: The alignment problem is a core step in many machine learning tasks, including multi-network analysis, multimodal learning, and point cloud registration. Recently, Optimal Transport (OT) has been widely used for alignment due to its ability to infer soft correspondences between distributions from a global perspective. By associating two sets of objects with discrete probability distributions and solving for the transport plan $\mathbf{T}$, OT methods have demonstrated superior performance across various alignment tasks.

**Limitations of Prior Work**: OT methods are highly sensitive to the quantity and quality of supervision signals—experiments show that increasing supervision can yield performance gains of up to 15%, while different query strategies can result in a 7% performance gap under the same budget. However, high-quality supervised annotations are expensive in practice (e.g., manual annotation of node correspondences across networks), and almost no existing work investigates how to actively acquire high-quality supervision within the OT framework.

**Key Challenge**: Existing active alignment methods face three critical limitations: (1) they are not designed for OT and fail to leverage core OT components such as cost functions and marginal constraints; (2) they lack a principled method to quantify how new supervision propagates through the OT formulation to affect alignment results; and (3) strategies are often designed for specific tasks (e.g., network alignment) and do not generalize easily to other tasks like cross-domain alignment.

**Goal**: To design a general active learning framework that maximizes the alignment performance of OT methods under a fixed query budget, applicable to various tasks such as network alignment, image-text retrieval, and image-text grounding.

**Key Insight**: The authors observe that global alignment quality can be encoded into a scalar using a utility function $f(\mathbf{T})$. Gradient propagation can then be used to calculate the impact of each candidate query on this utility function—a larger gradient indicates that the query is more valuable for improving the global alignment.

**Core Idea**: The difficult problem of "differentiating through OT" is transformed into an $(n+m)$-dimensional linear system via the adjoint state method. This system is solved using the conjugate gradient method with linear complexity and guaranteed convergence, efficiently quantifying the informativeness of each candidate query.

## Method

### Overall Architecture

The input to AvAtar consists of two sets of objects to be aligned $\mathcal{X}, \mathcal{Y}$, an OT alignment method (including cost function $\mathbf{C}$ and marginal distributions $\boldsymbol{\mu}, \boldsymbol{\nu}$), the current supervision matrix $\mathbf{H}$, and a query budget $k$. The output consists of $k$ optimal query objects and the updated alignment matrix $\mathbf{T}$. The core process iterates through: (1) computing the posterior query influence for all candidates; (2) selecting the $n_b$ candidates with the highest influence; (3) querying the oracle for ground-truth alignment; and (4) updating $\mathbf{H}$ and re-solving the OT alignment.

### Key Designs

1.  **Gradient-based Pairwise Query Influence Quantification**:
    - **Function**: Quantify the impact of querying a specific object pair $(x_i, y_j)$ on the global alignment result.
    - **Mechanism**: The gradient of the utility function $f(\mathbf{T})$ with respect to the supervision signal $\mathbf{H}_{i,j}$ is decomposed via the chain rule: $\nabla_{\mathbf{H}_{i,j}} f = \langle \nabla_{\tilde{\mathbf{C}}} f, \nabla_{\mathbf{H}_{i,j}} \tilde{\mathbf{C}} \rangle$. The latter term is computed directly as $-\beta \mathbf{C}_{i,j} \mathbf{E}$. The former term $\nabla_{\tilde{\mathbf{C}}} f$ requires differentiating through the OT solver. Since the transport plan $\mathbf{T}$ is an implicit function of the cost matrix, direct differentiation would require explicitly constructing and inverting a Jacobian matrix of size $(nm)^2$, which is computationally infeasible. The authors use the adjoint state method to transform this into solving an $(n+m)$-dimensional adjoint linear system $\mathbf{A}\mathbf{y} = \mathbf{b}$, where the coefficient matrix $\mathbf{A}$ is composed of marginal distributions and the transport plan. Although the system is singular ($\mathbf{A}$ is not invertible), its positive semi-definiteness and the fact that $\mathbf{b}$ lies in the column space of $\mathbf{A}$ allow the conjugate gradient method to guarantee convergence to a global optimum. Utilizing the sparsity of $\mathbf{T}$ achieves a linear time complexity of $\mathcal{O}(K(n+m))$.
    - **Design Motivation**: To avoid explicit construction and inversion of high-dimensional Jacobian matrices, making the differentiation of OT computationally efficient and feasible.

2.  **Posterior Object Query Influence Aggregation**:
    - **Function**: Aggregate pairwise influence to determine the query influence of a single source object.
    - **Mechanism**: For a source object $x_i$, its query influence is defined as the weighted sum of pairwise influences with all target objects, weighted by the transport plan: $\mathcal{I}(p_i) = \sum_{j=1}^{m} \mathbf{T}_{i,j} \mathcal{I}(p_{ij})$. Since the transport plan $\mathbf{T}_{i,j}$ encodes the posterior alignment probability between $x_i$ and $y_j$, using $\mathbf{T}$ as weights is equivalent to calculating the expected influence of querying $x_i$. Experiments show that posterior aggregation improves MRR by up to 16.8% compared to uniform aggregation (on the Douban dataset).
    - **Design Motivation**: Since the ground-truth alignment target is unknown before querying, the transport plan from OT is used as prior information for expectation calculation, which is both principled and fully utilizes the OT output.

3.  **General Utility Function Design**:
    - **Function**: Encode alignment results as a scalar to measure global quality.
    - **Mechanism**: Three utility functions are proposed: $f_{L_2} = \|\mathbf{T}\|_2^2$ and $f_{\text{entropy}} = \|\mathbf{T} \odot \log(\mathbf{T})\|_1$ are general functions that encourage the transport plan to converge toward a deterministic permutation matrix; $f_{\text{consist}}$ is specific to network alignment, using the graph Laplacian to measure neighborhood consistency. Different utility functions can be selected for different tasks while the framework remains unchanged.
    - **Design Motivation**: To achieve generalization across different alignment tasks through interchangeable utility functions without redesigning the query strategy for each task.

### Loss & Training

AvAtar employs a batch iterative query strategy: in each round, $n_b$ candidate queries are selected for the oracle, $\mathbf{H}$ is updated, and the OT solver is re-run until the budget $k$ is exhausted. The convergence rate of the conjugate gradient method is $\frac{\sqrt{\lambda_1/\lambda_r} - 1}{\sqrt{\lambda_1/\lambda_r} + 1}$, where $\lambda_1$ and $\lambda_r$ are the maximum and minimum non-zero eigenvalues of the coefficient matrix, respectively. Experiments indicate that CG converges faster than the Sinkhorn algorithm and is more robust to the regularization weight $\epsilon$.

## Key Experimental Results

### Main Results — Network Alignment (MRR)

| Dataset | Method | Query=5 | Query=10 | Gain vs. Random |
| :--- | :--- | :--- | :--- | :--- |
| Phone-Email | Random (JOENA) | 0.582 | 0.648 | — |
| Phone-Email | AvAtar-$L_2$ (JOENA) | 0.682 | 0.800 | +15.2% |
| Phone-Email | AvAtar-consist (JOENA) | **0.691** | **0.806** | +15.8% |
| ACM-DBLP-A | Random (JOENA) | 0.821 | 0.837 | — |
| ACM-DBLP-A | AvAtar-$L_2$ (JOENA) | **0.924** | **0.981** | +14.4% |
| Douban | Random (PARROT) | 0.730 | 0.751 | — |
| Douban | AvAtar-$L_2$ (PARROT) | **0.782** | **0.837** | +8.6% |

### Main Results — Cross-domain Alignment (Recall@1)

| Task | Dataset | Method | Query=5 | Query=10 |
| :--- | :--- | :--- | :--- | :--- |
| Image-Text Retrieval | ImageNet-C (GOT-W) | Random | 0.374 | 0.454 |
| Image-Text Retrieval | ImageNet-C (GOT-W) | AvAtar-entropy | **0.402** | **0.509** |
| Image-Text Grounding | Flickr30K (GOT-FGW) | Random | 0.550 | 0.628 |
| Image-Text Grounding | Flickr30K (GOT-FGW) | AvAtar-$L_2$ | **0.575** | **0.671** |
| Image-Text Grounding | COCO (GOT-FGW) | Random | 0.545 | 0.607 |
| Image-Text Grounding | COCO (GOT-FGW) | AvAtar-$L_2$ | **0.582** | **0.678** |

### Ablation Study

| Ablation Dimension | Configuration | MRR (Douban) | Description |
| :--- | :--- | :--- | :--- |
| Sparse vs. Dense | AvAtar-$L_2$ (Sparse) | 0.837 | 5.1s, 8.6× speedup |
| Sparse vs. Dense | AvAtar-$L_2$ (Dense) | 0.839 | 19s, comparable performance |
| Aggregation Type | AvAtar-$L_2$ (Posterior) | 0.837 | Transport plan weighting |
| Aggregation Type | AvAtar-$L_2$ (Uniform) | 0.669 | Uniform aggregation, MRR drops 16.8% |
| Utility Function | AvAtar-consist (PARROT) | 0.835 | Better for unattributed networks |
| Utility Function | AvAtar-$L_2$ (PARROT) | 0.837 | Better for attributed networks |

## Highlights & Insights

- The critical technical bottleneck of "differentiating through OT" is elegantly transformed into solving an adjoint linear system, avoiding $(nm)^2$ Jacobian inversion. This approach can be generalized to other scenarios requiring differentiation of constrained optimizations.
- The design of posterior weighted aggregation is highly sophisticated—the transport plan $\mathbf{T}$ itself encodes the posterior probability of alignment, and using it to weight pairwise influences is mathematically equivalent to computing conditional expectations.
- The framework exhibits strong generality: different tasks can be accommodated simply by replacing the utility function, while the core gradient calculation process is fully reused.
- The experiments cover 8 datasets, 4 OT methods, and 9 baseline strategies, achieving consistent SOTA performance across three major categories of alignment tasks, providing strong empirical evidence.

## Limitations & Future Work

- The datasets used for cross-domain alignment tasks (image-text retrieval/grounding) are relatively small and have not been validated on large-scale multimodal benchmarks.
- The selection of utility functions still requires guidance from domain knowledge; automatic selection or learning of utility functions remains a future research direction.
- The framework depends on the differentiability of entropy-regularized OT; extensions to non-regularized or unbalanced OT have not yet been discussed.
- The batch selection strategy is a greedy top-$n_b$ approach, which does not consider diversity or redundancy among candidates.

## Related Work & Insights

- **PARROT / JOENA**: Two representative OT-based network alignment methods; this work serves as an active learning layer directly integrated on top of them.
- **GOT**: A cross-domain alignment framework based on Wasserstein and Gromov-Wasserstein distances, upon which this work validates cross-domain active alignment.
- **Adjoint State Method**: An efficient differentiation technique originating from PDE-constrained optimization and Neural ODEs, applied here for the first time to gradient propagation in OT alignment.
- This research provides direct inspiration for researchers handling sparse annotation alignment problems: rather than random labeling, gradients should guide the selection of the most valuable objects to annotate.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to formalize the OT active alignment problem and provide a principled solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8 datasets × 4 OT methods × 9 baselines, covering three major tasks.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivation, unified notation, and complete paper structure.
- Value: ⭐⭐⭐⭐ — A general framework that can be directly applied to various OT alignment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Streaming Sliced Optimal Transport](streaming_sliced_optimal_transport.md)
- [\[ICML 2026\] Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation](convex_distance_operator_transport_a_convex_and_geometry-preserving_formulation.md)
- [\[ICLR 2026\] SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation](../../ICLR2026/3d_vision/scenetransporter_optimal_transport-guided_compositional_latent_diffusion_for_sin.md)
- [\[ICML 2026\] PLAID: A Unified Data Model for Machine Learning on Heterogeneous Physics Simulations](plaid_a_unified_data_model_for_machine_learning_on_heterogeneous_physics_simulat.md)
- [\[ICML 2026\] Geometry-Guided Modeling of Foundation Features Enables Generalizable Object Shape Deformation Learning](geometry-guided_modeling_of_foundation_features_enables_generalizable_object_sha.md)

</div>

<!-- RELATED:END -->
