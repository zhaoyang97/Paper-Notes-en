---
title: >-
  [Paper Note] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?
description: >-
  [ICLR 2026][Graph Learning][oversmoothing] This paper identifies that the widely adopted Dirichlet energy metric fails to correctly capture oversmoothing in GNNs under practical settings. It proposes the numerical/effect…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "oversmoothing"
  - "graph neural networks"
  - "Dirichlet energy"
  - "numerical rank"
  - "effective rank"
  - "rank collapse"
  - "GNN depth"
date: 2026-05-08
content_hash: ee504d90a54ac08d
---

# Are We Measuring Oversmoothing in Graph Neural Networks Correctly?

**Conference**: ICLR 2026
**arXiv**: [2502.04591](https://arxiv.org/abs/2502.04591)  
**Area**: Graph Neural Networks / Theoretical Analysis
**Keywords**: oversmoothing, graph neural networks, Dirichlet energy, numerical rank, effective rank, rank collapse, GNN depth

## TL;DR

This paper identifies that the widely adopted Dirichlet energy metric fails to correctly capture oversmoothing in GNNs under practical settings. It proposes the numerical/effective rank (Erank) of the feature representation matrix as an alternative measure. Empirically, Erank achieves an average correlation of 0.91 with accuracy (vs. 0.72 for Dirichlet energy), while on OGB-Arxiv, Dirichlet energy even exhibits an incorrect correlation direction. The paper further provides theoretical proofs that the numerical rank converges to 1 (rank collapse) for a broad family of GNN architectures, and redefines oversmoothing as rank collapse rather than feature vector alignment.

## Background & Motivation

**Oversmoothing as a core bottleneck in GNNs**: As the number of layers increases, GNN node representations tend to become indistinguishable, losing discriminability—widely regarded as the primary reason GNNs cannot benefit from depth as other deep networks do.

**Dominance of Dirichlet energy**: The current literature almost universally adopts Dirichlet energy (the sum of squared differences between representations of adjacent nodes) as the standard measure of oversmoothing, and a large body of mitigation methods has been designed around it.

**Decoupling between metric and performance**: In practice, the direction of change in Dirichlet energy sometimes contradicts model performance—a decrease in Dirichlet energy does not necessarily imply a performance drop, and vice versa.

**Insufficient theoretical understanding**: Existing oversmoothing theory primarily analyzes the convergence behavior of Dirichlet energy under specific graph topologies, without questioning the validity of the metric itself.

**Failure on heterophilic graphs**: On heterophilic graphs, where adjacent nodes have different labels, a high Dirichlet energy may actually indicate that the model has learned useful discriminative representations, whereas a low value may signal failure.

**Need for a better metric**: The GNN community requires an oversmoothing measure that is more consistent with downstream task performance in order to properly guide architecture design and regularization strategies.

## Method

### Overall Architecture
The paper revisits oversmoothing from a linear-algebraic perspective, redefining it as rank collapse of the feature matrix: as the number of layers increases, the effective rank of the node feature matrix approaches 1, meaning all node representations collapse into a low-dimensional subspace.

### Key Designs

1. **Effective Rank as the Measure**

    - **Function**: The effective rank is defined via the normalized entropy of the singular values of the feature matrix: $\text{Erank}(X) = \exp\left(-\sum_i \hat{\sigma}_i \log \hat{\sigma}_i\right)$, where $\hat{\sigma}_i = \sigma_i / \sum_j \sigma_j$.
    - **Mechanism**: Effective rank captures the number of "meaningful dimensions" in the feature matrix. A rank of 1 indicates that all node representations are linearly dependent (complete oversmoothing), while a high rank reflects rich representational diversity.
    - **Design Motivation**: Compared to Dirichlet energy, Erank directly measures the degree of dimensional collapse in the representation space and has a more direct causal connection to classification performance.

2. **Empirical Analysis of Dirichlet Energy Failures**

    - **Function**: A systematic comparison of the correlations between Erank, Dirichlet energy, and classification accuracy is conducted across multiple GNN architectures and datasets.
    - **Mechanism**: Pearson/Spearman correlation coefficients between each metric and node classification accuracy are computed across varying depths.
    - **Design Motivation**: If a metric claims to measure oversmoothing, it should exhibit a strong negative correlation with model performance—the more severe the oversmoothing, the worse the performance.

3. **Theoretical Convergence Proofs**

    - **Function**: Formal proofs are provided showing that for a broad family of GNN architectures—including GCN, GraphSAGE, and GIN—the numerical rank of the feature matrix converges to 1 as the number of layers tends to infinity.
    - **Mechanism**: Spectral analysis of matrix powers is employed to demonstrate that repeated application of the normalized adjacency matrix causes the singular value spectrum to collapse, ultimately retaining only the dominant singular value.
    - **Design Motivation**: To provide a theoretical foundation for rank collapse as a formal definition of oversmoothing.

## Key Experimental Results

### Main Results: Correlation Between Metrics and Accuracy

| Dataset | Dirichlet Energy Correlation | Erank Correlation | Difference |
|---------|------------------------------|-------------------|------------|
| Cora | 0.82 | **0.94** | +0.12 |
| Citeseer | 0.76 | **0.91** | +0.15 |
| PubMed | 0.79 | **0.93** | +0.14 |
| OGB-Arxiv | **-0.31** (wrong direction!) | **0.88** | +1.19 |
| OGB-Products | 0.45 | **0.89** | +0.44 |
| Average | 0.72 | **0.91** | +0.19 |

### Ablation Study: Different GNN Architectures

| Architecture | Layer Range | Erank@2L | Erank@64L | Accuracy Drop |
|-------------|-------------|----------|-----------|---------------|
| GCN | 2–128 | 45.2 | 1.3 | −42.1% |
| GAT | 2–128 | 52.8 | 2.1 | −38.6% |
| GraphSAGE | 2–128 | 61.3 | 3.7 | −31.2% |
| GIN | 2–128 | 48.9 | 1.8 | −39.8% |
| ResGCN | 2–128 | 47.6 | 8.4 | −22.3% |

### Key Findings

1. **Dirichlet energy fails on large-scale graphs**: On OGB-Arxiv, Dirichlet energy is negatively correlated with accuracy (−0.31), meaning accuracy actually improves as Dirichlet energy increases.
2. **Erank is consistently reliable**: Across all datasets and architectures, Erank maintains a high positive correlation with accuracy (>0.88).
3. **Rank collapse is universal**: All tested GNN architectures exhibit Erank approaching 1 at sufficient depth (≥64 layers).
4. **Residual connections slow but do not prevent collapse**: ResGCN exhibits a slower decline in Erank, but ultimately converges to a low-rank state.
5. **Metric choice influences research conclusions**: Relying on Dirichlet energy may lead researchers to draw incorrect conclusions about architectural improvements.

## Highlights & Insights

1. **Challenging a foundational assumption**: The paper questions the Dirichlet energy metric broadly accepted by the GNN community—a significant "emperor's new clothes" contribution.
2. **Superiority of the proposed alternative**: Erank substantially outperforms Dirichlet energy both theoretically and empirically, and is straightforward to implement.
3. **Conceptual redefinition**: Oversmoothing is redefined from "convergence of neighboring node features" (feature vector alignment) to "collapse of the feature space rank," which is more fundamental.
4. **Practical guidance**: A correct metric will guide the community toward developing more effective oversmoothing mitigation strategies.
5. **Importance of large-scale datasets**: The difference between the two metrics is modest on small datasets but becomes dramatic on large-scale real-world graphs.

## Limitations & Future Work

1. **Causality not fully established**: High correlation does not imply causation; whether rank collapse is the sole cause of performance degradation remains an open question.
2. **Limited exploration of heterophilic graphs**: Although mentioned, the experimental coverage of Erank behavior on heterophilic graphs is not sufficiently systematic.
3. **No mitigation strategies proposed**: The paper focuses on diagnostic measurement and does not propose concrete mitigation algorithms based on the rank perspective.
4. **Graph-level tasks not addressed**: All experiments target node classification; the manifestation of oversmoothing in graph-level classification tasks is not explored.

## Related Work & Insights

- **Oversmoothing theory**: Li et al. (2018) first formally defined oversmoothing; Cai & Wang (2020) introduced the Dirichlet energy framework.
- **Oversmoothing mitigation**: DropEdge (Rong et al., 2020), PairNorm (Zhao & Akoglu, 2020), DeeperGCN (Li et al., 2020).
- **Effective rank**: The definition of effective rank proposed by Roy & Vetterli (2007).
- **Deep GNNs**: Deep architecture designs such as GCNII (Chen et al., 2020) and RevGNN (Li et al., 2021).

## Rating

- Novelty: ⭐⭐⭐⭐ Challenges a foundational metric assumption and introduces the rank collapse perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple datasets and architectures, combining theoretical and empirical evidence.
- Writing Quality: ⭐⭐⭐⭐⭐ Argumentation is logically rigorous and experimental design is well-conceived.
- Value: ⭐⭐⭐⭐⭐ Provides a necessary correction to a foundational metric used throughout the GNN community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](cooperative_sheaf_neural_networks.md)
- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)
- [\[CVPR 2026\] Adaptive Learned Image Compression with Graph Neural Networks](../../CVPR2026/graph_learning/adaptive_learned_image_compression_with_graph_neural_networks.md)
- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](../../ICML2026/graph_learning/quantile-free_uncertainty_quantification_in_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
