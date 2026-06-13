---
title: >-
  [Paper Note] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization
description: >-
  [NeurIPS 2025][Graph Learning][Graph Neural Networks] This paper proposes Hierarchical Ego GNNs (HEGNNs), which generalize subgraph GNNs through a hierarchical node individualization mechanism…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Expressiveness"
  - "Hierarchical Node Individualization"
  - "Isomorphism Testing"
  - "Hybrid Logic"
date: 2026-05-08
content_hash: 220e6bea52dc87c3
---

# Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization

**Conference**: NeurIPS 2025

**arXiv**: [2506.13911](https://arxiv.org/abs/2506.13911)

**Code**: None

**Area**: Graph Learning / Graph Neural Network Theory

**Keywords**: Graph Neural Networks, Expressiveness, Hierarchical Node Individualization, Isomorphism Testing, Hybrid Logic

## TL;DR

This paper proposes Hierarchical Ego GNNs (HEGNNs), which generalize subgraph GNNs through a hierarchical node individualization mechanism, forming a hierarchy of models with strictly increasing expressive power. On bounded-degree graphs, the paper proves that the distinguishing power of HEGNN node classifiers is equivalent to graded hybrid logic ($\mathcal{HL}_k$), thereby unifying the expressiveness analysis of various GNN variants.

## Background & Motivation

The expressiveness of standard message-passing GNNs (MPNNs) is bounded by the 1-dimensional Weisfeiler-Leman (1-WL) color refinement algorithm, which fails to distinguish many pairs of non-isomorphic graphs. Several approaches have been proposed to enhance GNN expressiveness:

- **Higher-order GNNs** ($k$-WL GNNs): High computational complexity
- **Subgraph GNNs**: Construct ego-subgraphs centered at each node to increase distinguishing power
- **Local homomorphism count features**: Augment node features with subgraph occurrence counts
- **Individualization-Refinement (IR)**: A classical paradigm in graph isomorphism testing

**Core Problem**: What are the relationships among these methods? Can a unified hierarchical framework be established to precisely characterize their expressive power?

Traditional analyses lack tools to precisely connect GNN expressiveness to formal logical languages, resulting in the absence of a unified benchmark for comparing different GNN variants.

## Method

### Overall Architecture

HEGNNs are inspired by the Individualization-Refinement (IR) paradigm and extend standard GNNs through **hierarchical node individualization**:

- **Level 0**: Standard GNN (equivalent to 1-WL)
- **Level 1**: For each node $v$, run a GNN on the subgraph in which $v$ is "individualized" (equivalent to subgraph GNN)
- **Level $k$**: Recursively individualize $k$ nested layers of nodes
- **Limit**: As $k \to \infty$, all non-isomorphic graphs become distinguishable

Formal definition:

$$\text{HEGNN}_k(v) = \text{GNN}\left(G, v, \{\text{HEGNN}_{k-1}(u) : u \in V\}\right)$$

### Key Designs

#### Hierarchical Node Individualization

1. **Individualization**: Assign a unique marker to a selected node, breaking symmetry among nodes
2. **Hierarchical structure**: Apply individualization recursively, adding one "anchor" node per level
3. **Aggregation**: Aggregate information across levels, combining distinguishing power from different depths

**Relationship to Subgraph GNNs**:
- Subgraph GNN ≈ HEGNN-1 (single individualization layer)
- HEGNN-$k$ is strictly more expressive than HEGNN-$(k-1)$

#### Logical Characterization Theorem

**Core Theorem** (on bounded-degree graphs):

$$\text{Distinguishing power of } \text{HEGNN}_k = \text{Graded hybrid logic } \mathcal{HL}_k$$

Graded hybrid logic $\mathcal{HL}_k$ is an extension of modal logic with:
- **Graded modal operators**: $\langle \geq n \rangle \phi$ denotes "at least $n$ neighbors satisfy $\phi$"
- **Nominals**: Allow reference to specific individual nodes
- **Level $k$**: Permits $k$ nested layers of nominals

#### Relationship to Other GNN Variants

The logical characterization provides a unified basis for comparison:

| GNN Variant | Equivalent Logic | HEGNN Level |
|-------------|-----------------|-------------|
| Standard MPNN | Graded modal logic $\mathcal{ML}$ | HEGNN-0 |
| Subgraph GNN | $\mathcal{HL}_1$ | HEGNN-1 |
| $k$-WL GNN | $\mathcal{C}^k$ ($k$-variable counting logic) | No direct correspondence |
| HEGNN-$k$ | $\mathcal{HL}_k$ | HEGNN-$k$ |
| HEGNN-$\infty$ | Full distinguishability (isomorphism testing) | Limit |

### Loss & Training

HEGNNs follow standard graph learning training procedures:
- **Node classification**: Cross-entropy loss
- **Graph classification**: Cross-entropy after readout
- **Optimization**: Standard SGD/Adam

Key implementation considerations:
- Level-$k$ HEGNN requires $O(n^k)$ computation (each individualization level introduces an additional loop)
- In practice, $k=1$ or $k=2$ is typically used

## Key Experimental Results

### Main Results

#### Graph Isomorphism Discrimination

Evaluated on several standard graph isomorphism test benchmarks:

| Model | EXP | CSL | SR25 | 4×4 Rook | Discrimination Rate (%) |
|-------|:---:|:---:|:----:|:--------:|:-----------------------:|
| GCN | 50.0 | 10.0 | 0.0 | 0.0 | 15.0 |
| GIN | 50.0 | 10.0 | 0.0 | 0.0 | 15.0 |
| PPGN (3-WL) | 100.0 | 100.0 | 66.7 | 100.0 | 91.7 |
| DS (Subgraph GNN) | 100.0 | 100.0 | 0.0 | 16.7 | 54.2 |
| **HEGNN-1** | 100.0 | 100.0 | 33.3 | 66.7 | 75.0 |
| **HEGNN-2** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |

- HEGNN-2 achieves perfect discrimination on all datasets, matching or surpassing 3-WL GNNs
- HEGNN-1 already significantly outperforms standard GNNs and subgraph GNNs

#### Graph Classification

Graph classification accuracy on TU and OGB datasets:

| Model | MUTAG | PTC | PROTEINS | IMDB-B | Average |
|-------|:-----:|:---:|:--------:|:------:|:-------:|
| GCN | 85.6 | 64.2 | 76.0 | 74.1 | 75.0 |
| GIN | 89.4 | 64.6 | 76.2 | 75.1 | 76.3 |
| Subgraph GNN | 89.8 | 66.1 | 76.8 | 75.8 | 77.1 |
| **HEGNN-1** | 90.2 | 67.3 | 77.1 | 76.2 | 77.7 |
| **HEGNN-2** | **91.1** | **68.5** | **77.5** | **76.8** | **78.5** |

### Ablation Study

#### Effect of Local Homomorphism Count Features

| Model | No Count Features | +Triangle Count | +4-Cycle Count | All Counts |
|-------|:-----------------:|:---------------:|:--------------:|:----------:|
| GCN | 75.0 | 76.2 | 76.5 | 77.0 |
| GIN | 76.3 | 77.5 | 77.8 | 78.1 |
| **HEGNN-1** | 77.7 | 78.2 | 78.3 | 78.5 |
| **HEGNN-2** | **78.5** | **78.8** | **78.9** | **79.0** |

**Key Findings**:
- HEGNN-2 without homomorphism count features approaches GIN with all count features
- Count features yield smaller gains for HEGNN than for base GNNs, suggesting that HEGNNs implicitly capture partial substructure information

#### Computational Complexity Analysis

| Model | Time Complexity | Space Complexity | Actual Inference Time (EXP) |
|-------|:--------------:|:----------------:|:---------------------------:|
| GCN | $O(n \cdot d \cdot L)$ | $O(n \cdot d)$ | 1× |
| GIN | $O(n \cdot d \cdot L)$ | $O(n \cdot d)$ | 1.2× |
| HEGNN-1 | $O(n^2 \cdot d \cdot L)$ | $O(n^2 \cdot d)$ | 8× |
| HEGNN-2 | $O(n^3 \cdot d \cdot L)$ | $O(n^3 \cdot d)$ | 50× |

### Key Findings

1. HEGNNs form a strict expressiveness hierarchy: HEGNN-0 $\subset$ HEGNN-1 $\subset$ HEGNN-2 $\subset \ldots$
2. The logical characterization establishes a precise bridge between GNN expressiveness and mathematical logic
3. HEGNNs are practically feasible, though the computational cost of higher-order variants limits scalability
4. HEGNN-1 or HEGNN-2 suffices for most tasks

## Highlights & Insights

1. **Precise logical characterization**: Establishing an exact correspondence between GNN expressiveness and graded hybrid logic represents a significant theoretical advance in graph learning
2. **Unified framework**: Provides a common language for comparing the expressive power of different GNN variants
3. **Elegant transfer of the IR paradigm**: The classical IR paradigm from graph isomorphism testing is naturally translated into a GNN architecture
4. **Theory guiding practice**: The logical characterization assists practitioners in selecting GNN variants of appropriate complexity

## Limitations & Future Work

1. **Bounded-degree assumption**: The logical characterization theorem requires graphs to have bounded degree and does not directly apply to unbounded-degree graphs (e.g., power-law graphs)
2. **Computational scalability**: The $O(n^{k+1})$ complexity of HEGNN-$k$ limits the practical use of higher-order variants
3. **Lack of large-scale experiments**: Experiments on large-scale benchmarks such as OGB are limited
4. **Potential directions**:
    - Design efficient algorithms that approximate HEGNN-$k$ (e.g., via sampling strategies)
    - Relax the bounded-degree assumption
    - Explore integration with Transformer architectures

## Related Work & Insights

- **WL hierarchy**: The $k$-WL test serves as the baseline for GNN expressiveness analysis; this paper provides an orthogonal hierarchical structure
- **Subgraph GNNs**: Building on Bevilacqua et al. (2022) and Frasca et al. (2022), HEGNNs constitute a natural generalization
- **GNNs and logic**: Barceló et al. (2020) established connections between GNNs and FOC$_2$; this paper extends the analysis to hybrid logic
- **Local homomorphism counting**: Bouritsas et al. (2023) provide related work; this paper unifies the relationship through a logical framework

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Precisely characterizing GNN expressiveness hierarchies via logic is pioneering work
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Deep theoretical analysis integrating graph theory, modal logic, and deep learning
- **Experimental Thoroughness**: ⭐⭐⭐ — Experiments demonstrate feasibility but are limited in scale
- **Practicality**: ⭐⭐⭐ — Major theoretical contributions; practical applicability constrained by computational cost
- **Overall**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[AAAI 2026\] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](../../AAAI2026/graph_learning/enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)

</div>

<!-- RELATED:END -->
