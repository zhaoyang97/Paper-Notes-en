---
title: >-
  [Paper Note] View Space: Representation Learning Across Arbitrary Graphs
description: >-
  [ICML 2026][Graph Learning][Graph Representation Learning] This paper introduces the concept of View Space, elevating graph representation from 2D (node-feature) to 3D (node-feature-view) to achieve a unified representat…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Graph Representation Learning"
  - "Feature Heterogeneity"
  - "Fully Inductive Learning"
  - "View Space"
date: 2026-05-08
content_hash: 5792430badf26b83
---

# View Space: Representation Learning Across Arbitrary Graphs

**Conference**: ICML 2026  
**arXiv**: [2512.11561](https://arxiv.org/abs/2512.11561)  
**Code**: To be confirmed  
**Area**: Graph Learning / Graph Neural Networks / Cross-domain Transfer  
**Keywords**: Graph Representation Learning, Feature Heterogeneity, Fully Inductive Learning, View Space

## TL;DR
This paper introduces the concept of View Space, elevating graph representation from 2D (node-feature) to 3D (node-feature-view) to achieve a unified representation for graphs with arbitrary feature dimensions and semantics. This allows graph models to perform zero-shot cross-domain reasoning for the first time, similar to NLP/CV foundation models, outperforming GraphAny by an average of 8.93% across 27 downstream tasks.

## Background & Motivation

**Background**: In NLP and CV, foundation models perform cross-dataset reasoning through large-scale pre-training followed by lightweight adaptation. This is possible due to standardized input formats—all text in NLP is tokenized into a shared vocabulary, and all images in CV can be resized to a fixed resolution.

**Limitations of Prior Work**: Standardizing graph data is extremely difficult. The dimensions and semantics of node features vary drastically across datasets. Existing GNNs handle this by learning feature transformation matrices, resulting in very weak generalization across feature spaces. While GraphAny partially addressed the fully inductive problem via relative distance spaces, it can only perform prediction and cannot learn representations.

**Key Challenge**: How to enable models to learn universal knowledge across graphs and features while ensuring feature equivariance? Traditional 2D representations cannot satisfy node permutation equivariance and feature permutation equivariance simultaneously.

**Goal**: (1) Formalize "Fully Inductive Node Representation Learning" (FI-NRL); (2) Discover the third representation axis of graphs: View Space; (3) Design the parameterized Graph View Transform (GVT) and prove its dual permutation equivariance; (4) Instantiate the recursive architecture RGVT to validate cross-task generalization.

**Key Insight**: Connectivity properties are shared across all graphs. Different adjacency matrix preprocessing methods emphasize different structural aspects of a graph. These "views" can be stacked to form a new dimension, enabling the learning of representations independent of feature dimensions within a unified view space.

**Core Idea**: Elevate from 2D representation to 3D—each node-feature pair $(n,f)$ is mapped to a $C$-dimensional "view vector," where the $C$ dimensions correspond to $C$ different structural views of the graph. A shared learnable function processes these view vectors to automatically adapt to arbitrary dimensions and semantic features.

## Method

### Overall Architecture
Two stages—(1) **View Stacking**: $C$ "view finders" $\{\nu_c\}_{c=1}^C$ are applied to the input adjacency matrix $\bm{A}$ to generate different propagated versions $\nu_c(\bm{A}) \bm{X}$, which are stacked along a new dimension to form a 3D tensor $\bm{\mathsf{X}} \in \mathbb{R}^{N \times F \times C}$; (2) **View Transformation & Representation Learning**: A learnable function $\phi$ is applied to each position $(n,f,:)$ to map the $C$-dimensional view vector to a scalar, ultimately resulting in $N \times F$ node representations.

### Key Designs

1. **Definition of View Space and Dual Permutation Equivariance**:

    - Function: Maps graphs with arbitrary feature dimensions to a unified $\mathbb{R}^C$ space while satisfying node permutation equivariance (R1) and feature permutation equivariance (R2).
    - Mechanism: The node-feature matrix $\bm{X} \in \mathbb{R}^{N \times F}$ naturally carries two orthogonal spatial structures. A third axis is introduced: multiple propagated node-feature matrices are stacked along a new dimension to form $\bm{\mathsf{X}} \in \mathbb{R}^{N \times F \times C}$. Each $(n,f)$ corresponds to a $C$-dimensional view vector $\bm{v}_{n,f} = \bm{\mathsf{X}}_{n,f,:}$, recording the "response values" at that position across $C$ different structural viewpoints.
    - Design Motivation: The view dimension $C$ is determined by a predefined set of view finders and is independent of the input graph size $N$ and feature dimension $F$. Any graph can be represented by $N \times F$ $C$-dimensional vectors, forming a standardized input format.

2. **Graph View Transform (GVT) and Dynamic Aggregation**:

    - Function: $\Psi(\bm{X}, \bm{A}) = [\phi(\bm{\mathsf{X}}_{n,f,:} \mid \theta)]_{n,f}$ defines a parameterized representation function that satisfies dual permutation equivariance.
    - Mechanism: GVT involves two steps—(i) elevating to 3D via view stacking; (ii) applying a learnable dimensionality reduction function $\phi$ to each position. Nonlinear GVT can be proven via Taylor expansion to implement "node-feature level dynamic aggregation"—where aggregation weights differ for each $(n,f)$.
    - Design Motivation: Avoids explicit parameterization in the feature space (e.g., $\bm{W}$) and parameterizes in the view space instead, naturally satisfying feature permutation equivariance. Non-linearity allows the representation capability to exceed that of static aggregation (e.g., GCN).

3. **Recurrent Architecture and Depth Decoupling**:

    - Function: Recurrent GVT (RGVT) repeatedly applies the same $\Psi$ function $L$ times with shared parameters: $\bm{Z} = \Psi(\cdot, \bm{A} \mid \theta)^L(\bm{X})$.
    - Mechanism: Inspired by RNNs, this decouples "parameterization" and "depth." Since different graphs may require different propagation depths, a recurrent model allows choosing an appropriate $L$ for each new graph after pre-training without re-optimizing the encoder.
    - Design Motivation: Solves the discrepancy in receptive field requirements across different graphs while avoiding parameter explosion caused by stacking multiple distinct layers.

## Key Experimental Results

### Main Results

Pre-trained on OGBN-Arxiv and transferred to 27 downstream node classification datasets:

| Dataset Group | OGBN-Arxiv | Signed Dense | Unsigned Dense | Sparse | Binary Dense | Binary Sparse | One-hot | Average |
|----------|-----------|-------------|-------------|--------|-----------|-----------|----------|-----|
| Linear Predictor | 52.44 | 53.29 | 75.67 | 66.41 | 72.18 | 57.11 | 38.86 | 59.41 |
| MLP Predictor | 53.80 | 55.08 | 75.86 | 69.02 | 72.88 | 57.65 | 39.34 | 60.43 |
| GraphAny (Wisconsin) | 57.77 | 59.12 | 71.78 | 81.61 | 83.44 | 55.25 | 52.68 | 64.72 |
| GraphAny (Cora) | 58.58 | 59.38 | 71.76 | 81.49 | 83.35 | 53.40 | 53.30 | 64.30 |
| GraphAny (Arxiv) | 58.63 | 59.70 | 72.62 | 81.68 | 83.56 | 54.18 | 53.02 | 64.71 |
| **RGVT + Linear** | **70.14** | **64.95** | **76.44** | **84.33** | **85.11** | **62.77** | **58.85** | **70.03** |
| **RGVT + MLP** | **71.11** | **66.37** | **77.12** | **83.98** | **84.86** | **63.87** | **62.48** | **71.13** |

RGVT outperforms the best GraphAny variant by an average gain of +8.93% (MLP) or +7.24% (Linear).

### Ablation Study

| Configuration | OGBN-Arxiv | Signed Dense | Unsigned Dense | Sparse | Binary Dense | Binary Sparse | One-hot | Average |
|----|-----------|---------|---------|-----|--------|--------|--------|-----|
| RGVT + MLP (Full) | 71.11 | 66.37 | 77.12 | 83.98 | 84.86 | 63.87 | 62.48 | 71.13 |
| w/o Non-linearity | 70.22 | 64.53 | 75.89 | 78.82 | 84.16 | 61.12 | 56.13 | 68.12 |
| w/o Recurrence | 70.91 | 63.73 | 73.79 | 82.61 | 83.90 | 53.29 | 54.53 | 65.73 |
| w/o Both | 70.53 | 61.69 | 75.10 | 77.52 | 84.57 | 53.41 | 54.73 | 64.96 |

### Key Findings
- Removing non-linearity leads to a 2.31 percentage point decrease.
- Removing recurrence leads to a 5.40 percentage point decrease.
- Compared with 12 dataset-specific GNNs, RGVT + MLP outperforms the strongest baseline UniMP by an average gain of +3.30% (71.13 vs 68.86).

## Highlights & Insights
- **The Third Representation Axis of Graphs**: Breaks the limitations of 2D representation by abstracting connectivity information into a "view" dimension, orthogonal to the feature dimension.
- **Necessary and Sufficient Conditions for Dual Permutation Equivariance**: The paper provides formal definitions and necessary conditions, establishing a theoretical benchmark for other cross-domain graph learning works.
- **Node-Feature Level Dynamic Aggregation**: Taylor expansion reveals the expressive power of nonlinear GVT—each node-feature pair can have its own aggregation weight distribution.
- **Parameterization-Depth Decoupling**: Inspired by RNNs, this allows the model to flexibly choose the recursive depth after pre-training.
- **Transferable Knowledge**: View space knowledge learned on arXiv can be directly transferred to 27 downstream tasks with entirely different feature sets.

## Limitations & Future Work
- Design trade-off—GVT learns independently across each feature dimension and cannot explicitly model cross-feature interactions.
- Predictor training cost—A lightweight predictor still needs to be trained for each downstream task.
- Recursive depth selection overhead—Requires training multiple predictors for each dataset to select the optimal $L$.
- Scope—Primarily focuses on node classification; extensions to edge/graph classification and hypergraphs require further exploration.

## Related Work & Insights
- **vs Traditional GNNs (GCN, GAT, GraphSAGE)**: These have poor generalization across graphs due to explicit feature transformation matrices; Ours avoids this by calculating directly in the view space.
- **vs GraphAny**: GraphAny performs prediction via attention in relative distance spaces and can only output labels; Ours provides a representation learning scheme that is more flexible for various downstream predictors.
- **vs Tabular Foundation Models (TabR, TabM)**: These generalize across feature spaces via pre-training on synthetic data but do not utilize graph structure; Ours utilizes connectivity as a new axis beyond the feature space.
- **Insights**: (1) The "dimension elevation" idea can be adapted to other cross-domain problems; (2) The formal framework for permutation equivariance helps in designing other fully inductive models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  The view space concept is innovative, formalizing fully inductive learning for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  27 downstream tasks + multiple feature types + detailed ablation + comparison with 12 GNNs.
- Writing Quality: ⭐⭐⭐⭐⭐  Clear logic and rigorous formalization.
- Value: ⭐⭐⭐⭐⭐  Addresses long-standing challenges in graph learning and lays the foundation for graph foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[ICML 2026\] Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs](unsat_core_prediction_through_polarity-aware_representation_learning_over_clause.md)
- [\[ICML 2026\] Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective](message_tuning_outshines_graph_prompt_tuning_a_prismatic_space_perspective.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](../../AAAI2026/graph_learning/feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)

</div>

<!-- RELATED:END -->
