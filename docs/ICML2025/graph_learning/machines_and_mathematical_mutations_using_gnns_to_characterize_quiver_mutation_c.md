---
title: >-
  [Paper Note] Machines and Mathematical Mutations: Using GNNs to Characterize Quiver Mutation Classes
description: >-
  [ICML2025][Graph Learning][GNN] This work investigates quiver mutation equivalence classes using graph neural networks (GNNs) and explainability techniques, **independently rediscovering** the combinatorial characterization theorem for mutation classes of type $\tilde{D}$ quivers, demonstrating the value of ML as a tool for mathematical research.
tags:
  - "ICML2025"
  - "Graph Learning"
  - "GNN"
  - "explainability"
  - "quiver mutation"
  - "cluster algebra"
  - "AI for Math"
date: 2026-05-08
content_hash: ed3888bb3d8c4c74
---

# Machines and Mathematical Mutations: Using GNNs to Characterize Quiver Mutation Classes

**Conference**: ICML2025  
**arXiv**: [2411.07467](https://arxiv.org/abs/2411.07467)  
**Code**: Not released  
**Area**: Graph Learning  
**Keywords**: GNN, explainability, quiver mutation, cluster algebra, AI for Math

## TL;DR

This work investigates quiver mutation equivalence classes using graph neural networks (GNNs) and explainability techniques, **independently rediscovering** the combinatorial characterization theorem for mutation classes of type $\tilde{D}$ quivers, demonstrating the value of ML as a tool for mathematical research.

## Background & Motivation

### Quivers and Quiver Mutations

A **quiver** is a directed multigraph without self-loops or 2-cycles, serving as a core object in cluster algebra theory. Given a quiver $Q$ and a vertex $j$, the **mutation operation** $\mu_j(Q)$ generates a new quiver according to the following rules:

1. For every path $i \to j \to k$ in $Q$, add an edge $i \to k$.
2. Reverse all edges incident to $j$.
3. Remove any 2-cycles created.

Mutation is an involution: $\mu_j(\mu_j(Q)) = Q$, and thus defines an equivalence relation on quivers. Determining whether two quivers are **mutation equivalent** is a challenging problem.

### Prior Theory

For specific types of quivers, prior works have provided elegant combinatorial characterizations:

- **Type $A_n$** (Buan & Vatne, 2008): Characterized by conditions such as directed 3-cycles and vertex degrees.
- **Type $D_n$** (Vatne, 2010): Classified into four subtypes, Types I–IV, each formed by gluing Type $A$ subquivers via connecting vertices.
- **Type $\tilde{D}$**: No prior direct subquiver characterization existed (Henrich 2011 provided an equivalent result in a different language).

### Core Problem

Can we train a GNN classifier and utilize explainability tools to extract from the model combinatorial characterization rules that have not yet been explicitly formulated by mathematicians?

## Method

### Data Generation

SageMath is used to enumerate quivers of 6 types ($A, D, E, \tilde{A}, \tilde{D}, \tilde{E}$). The training set contains approximately 70,000 quivers with 6–10 vertices, and the test set consists of quivers with 11 vertices (excluding type $\tilde{E}$).

### DirGINE Architecture

Based on improvements to the Graph Isomorphism Network (GIN), the authors propose **DirGINE** (Directed GIN with Edge features):

- **Directed Message Passing**: Independent message aggregation functions are used along both directions of each edge.
- **Edge Features**: Supports edge weights as feature inputs.
- **Architecture Parameters**: A 4-layer GNN with a hidden dimension of 32, obtaining graph-level representations through global pooling.

The message passing update formula for GIN is:

$$h_v^{(k)} = \text{MLP}^{(k)}\left((1 + \epsilon^{(k)}) \cdot h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)$$

DirGINE extends this to bidirectional message passing, aggregating neighbor information along incoming and outgoing edges respectively, and then fusing them.

### Explainability Analysis Pipeline

A three-step progressive analysis strategy is adopted:

1. **PGExplainer Edge Attribution**: Trains an explanation network to generate importance scores $\omega_{u,v}$ for each edge, identifying subgraph structures crucial for classification decisions (e.g., the central cycle and leaf vertices of type $D$).
2. **Latent Space Clustering**: Performs PCA dimensionality reduction and visualization on the 32-dimensional latent representations of the model's 3rd layer to observe the natural clustering of subtypes.
3. **Masking Experiments**: Examines prediction changes after removing high-attribution edges to verify that the model indeed relies on specific substructures.

### Discovery Process from Type $D$ to Type $\tilde{D}$

**Validation on Type $D$**: PGExplainer attribution results highly align with Vatne's (2010) Type I–IV classification; latent space PCA shows clear separation among the four subtypes, with a linear classifier achieving 99.7% accuracy.

**Discovery of Type $\tilde{D}$**:

1. Annotate Type $\tilde{D}$ quivers based on "pairing types" (two Type $D$ subquivers sharing a connecting vertex).
2. The "Other" class quivers form two distinct clusters in the latent space.
3. Inspection of the clustered samples reveals that the distinguishing criterion is the **number of central directed cycles**.
4. Consequently, the Type V family (single central cycle) and Type VI family (double central cycles) are proposed.
5. Finally, the complete combinatorial characterization theorem (Theorem 6.1) is proven.

## Key Experimental Results

### Classification Accuracy

The best training epoch achieves **99.2%** accuracy on the test set.

### Size Generalization Performance

| Number of Vertices $n$ | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Accuracy (%) | 99.6 | 98.7 | 97.7 | 95.5 | 94.3 | 92.0 | 91.1 | 89.4 | 89.1 |

The model's accuracy gradually decreases on larger quivers, possibly because the fixed 4-layer depth GNN cannot identify long cycle structures in larger sizes.

### Masking Experiments (Type $D$)

Prediction changes of 32,066 Type $D$ test samples after removing high-attribution edges identified by PGExplainer:

| Flipped Category | Count | Percentage |
|:---:|:---:|:---:|
| Type $A$ | 14,916 | 46.5% |
| Type $E$ | 14,238 | 44.4% |
| Unflipped ($D$) | 2,581 | 8.0% |
| Type $\tilde{A}$ | 264 | 0.8% |
| Type $\tilde{D}$ | 67 | 0.2% |

Nearly half flip to Type $A$, aligning with theoretical expectations (Type $D$ quivers degenerate into Type $A$ once crucial substructures are removed).

### Latent Space Analysis

- PCA visualization of Type $D$ latent representations shows clear separation among the four Type I–IV clusters.
- A linear classifier distinguishing subtypes using the 32-dimensional embedding of the 3rd layer achieves **99.7% ± 0.0%**.
- The "Other" cluster for Type $\tilde{D}$ is divided into single-central-cycle and double-central-cycle families via $k$-means ($k=2$).

## Highlights & Insights

- **AI-driven Theorem Discovery**: Without receiving any subtype labels, the model spontaneously learns the same substructure features as human mathematicians in the latent space. This guided the authors to independently discover and prove the complete combinatorial characterization of Type $\tilde{D}$ quivers.
- **Complete Explainability Methodology**: A closed-loop three-step process is formed: PGExplainer edge attribution $\to$ latent space PCA clustering $\to$ masking causal validation.
- **GNN-Algorithm Alignment**: Choosing the GIN architecture is well-justified, as its capability to recognize subgraph structures naturally aligns with the subquiver-based characterization of mathematical theorems.
- **Size Generalization**: The model trained on 6–10 vertices maintains high accuracy when generalizing to 11–20 vertices.

## Limitations & Future Work

- **Generalization Decay**: The fixed 4-layer depth limits the model's ability to recognize long cycle structures in large quivers, causing accuracy to drop to 89.1% when $n=20$.
- **Limited Precision of PGExplainer**: Around 44% of masking experiments flip to Type $E$ instead of the expected Type $A$, suggesting that edge attribution might not be sufficiently precise.
- **Incomplete Type Coverage**: Only simply-laced Dynkin / affine Dynkin types are covered; non-simply-laced or more general mutation-finite quivers are not addressed.
- **Code Not Publicly Available**: The paper does not release code or data, which limits reproducibility.
- **Questionable Novelty of the Theorem**: The Type $\tilde{D}$ characterization was subsequently found to be derivable from existing results by Henrich (2011), despite differences in formulation.

## Related Work & Insights

- **Davies et al. (2021)**: A pioneering work utilizing ML to guide mathematical intuition (published in Nature); this paper replicates a similar paradigm in the field of quivers.
- **Cheung et al. (2023) / Bao et al. (2020)**: ML applications in cluster algebras, but with a focus on model performance rather than theorem discovery.
- **GNN Explainability**: PGExplainer has demonstrated capabilities in identifying functional groups on molecular datasets like MUTAG, and this work transfers that capability to a pure mathematics scenario.

## Rating

- Novelty: ⭐⭐⭐⭐ — A typical paradigm for AI for Math, extracting provable mathematical theorems from GNNs.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multidimensional validation via classification, generalization, masking, and latent space clustering, but lacking comparison with other GNN architectures.
- Writing Quality: ⭐⭐⭐⭐⭐ — Smooth transition between mathematical background and ML methods with clear writing.
- Value: ⭐⭐⭐⭐ — Demonstrates the complete workflow for ML-assisted mathematics research, although the novelty of the proven theorem itself is somewhat compromised.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)
- [\[ICML 2025\] Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality](balancing_efficiency_and_expressiveness_subgraph_gnns_with_walk-based_centrality.md)
- [\[ICML 2025\] TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation](tined_gnns-to-mlps_by_teacher_injection_and_dirichlet_energy_distillation.md)
- [\[NeurIPS 2025\] Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](../../NeurIPS2025/graph_learning/making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)
- [\[ICLR 2026\] On the Expressive Power of GNNs for Boolean Satisfiability](../../ICLR2026/graph_learning/on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)

</div>

<!-- RELATED:END -->
