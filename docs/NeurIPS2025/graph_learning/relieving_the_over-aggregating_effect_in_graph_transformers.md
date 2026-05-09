---
title: >-
  [Paper Note] Relieving the Over-Aggregating Effect in Graph Transformers
description: >-
  [NeurIPS 2025][Graph Learning][Over-Aggregating] This paper identifies the *over-aggregating* phenomenon in Graph Transformers—wherein a large number of nodes are aggregated with near-uniform attention scores, diluting critical information—and proposes Wideformer, which alleviates this issue via divided aggregation and guided attention. As a plug-and-play module, Wideformer consistently improves backbone model performance across 13 datasets.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Over-Aggregating
  - Graph Transformer
  - Attention Entropy
  - Wideformer
  - Linear Attention
date: 2026-05-08
content_hash: 66cc6ca81d5ef53d
---

# Relieving the Over-Aggregating Effect in Graph Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2510.21267](https://arxiv.org/abs/2510.21267)
**Code**: [https://github.com/sunjss/over-aggregating](https://github.com/sunjss/over-aggregating)
**Area**: Graph Learning / Graph Transformer
**Keywords**: Over-Aggregating, Graph Transformer, Attention Entropy, Wideformer, Linear Attention

## TL;DR
This paper identifies the *over-aggregating* phenomenon in Graph Transformers—wherein a large number of nodes are aggregated with near-uniform attention scores, diluting critical information—and proposes Wideformer, which alleviates this issue via divided aggregation and guided attention. As a plug-and-play module, Wideformer consistently improves backbone model performance across 13 datasets.

## Background & Motivation

**Background**: Graph Transformers leverage global attention mechanisms to capture long-range dependencies among nodes, overcoming the over-smoothing and over-squashing limitations of traditional GNNs. To handle large-scale graphs, two primary approaches exist: sparse attention and linear attention.

**Limitations of Prior Work**: Linear attention methods (e.g., Performer, SGFormer, Polynormer) preserve a global receptive field but suffer from severe information dilution—when all nodes participate in aggregation, attention scores tend toward uniformity (high attention entropy), preventing target nodes from distinguishing informative messages.

**Key Challenge**: In global attention computation, the more nodes involved, the more uniform the attention scores become (Theorem 3.1 proves that the entropy lower bound increases monotonically with $n$), causing critical messages to be diluted (over-aggregating). Sparse attention mitigates this issue but at the cost of a reduced receptive field.

**Goal**: Alleviate over-aggregating while preserving a global receptive field.

**Key Insight**: Rather than reducing the number of input nodes, the paper decomposes aggregation into multiple parallel sub-processes (cluster-wise aggregation) and increases the output dimensionality to retain more information.

**Core Idea**: Decompose the all-to-one global aggregation into multiple parallel cluster-to-one aggregations, and apply a guiding mechanism so that each target node focuses on the most informative subset.

## Method

### Overall Architecture
Wideformer is a plug-and-play module consisting of two steps:
1. **Dividing**: Source nodes are partitioned into $m$ clusters, each aggregated independently.
2. **Guiding**: The $m$ aggregation outputs are ranked and weighted so that target nodes focus on the most informative clusters.

Input: Standard Q, K, V features
Output: Enhanced representation for each target node

### Key Designs

1. **Cluster Center Selection (K-Means++-based variant)**:

   - **Function**: Selects $m$ representative cluster centers in the query space.
   - **Mechanism**: Following Algorithm 1, the node with the largest query feature norm is selected as the first center; subsequent centers are greedily chosen to maximize dissimilarity with existing centers.
   - **Design Motivation**: Maximizing inter-center diversity ensures that the cluster partition is semantically meaningful.

2. **Source Node Assignment**:

   - **Function**: Assigns each source node to the most similar cluster via $\mathbf{k}_i = \arg\max_j [(\mathbf{KC}^\top)_{i,j}]$.
   - **Mechanism**: Hard assignment based on key–center similarity.
   - **Design Motivation**: Grouping semantically similar nodes together yields more coherent intra-cluster aggregation and greater discriminability.

3. **Cluster-wise Aggregation + Guiding**:

   - **Function**: Each cluster performs attention aggregation independently, producing $m$ output vectors; these are then ranked and weighted according to their attention scores with the target node.
   - **Mechanism**: Reducing the aggregation input from $n$ to $n/m$ naturally lowers attention entropy; a secondary attention over the $m$ outputs preserves global information.
   - **Design Motivation**: Combining small-input aggregation with second-level ranking achieves both information retention and discriminability.

### Loss & Training
- Wideformer is embedded directly into the backbone training pipeline without additional loss terms.
- Computational complexity remains $O(n)$ (linear).
- Compatible with three backbone models: GraphGPS, SGFormer, and Polynormer.

## Key Experimental Results

### Main Results

| Dataset | Backbone | Original | + Wideformer | Gain |
|---|---|---|---|---|
| Cora | Polynormer | 86.03 | **86.23** | +0.20 |
| Citeseer | Polynormer | 77.96 | **78.19** | +0.23 |
| Amazon Photo | Polynormer | 95.47 | **95.52** | +0.05 |
| Minesweeper | Polynormer | 97.13 | **97.20** | +0.07 |

### Ablation Study

| Configuration | Key Findings |
|---|---|
| Direct entropy regularization | Effective but requires explicit computation of an $O(n^2)$ attention matrix; not scalable |
| Dividing only | Reduces entropy, validating the effectiveness of cluster partitioning |
| Guiding only | Ranking-based weighting improves information focus |
| Dividing + Guiding (Wideformer) | The combination yields the best overall performance |

### Key Findings
- Over-aggregating is a pervasive issue in linear-attention Graph Transformers; attention entropy increases with node count (Theorem 3.1).
- Gradient analysis (Eq. 3–4) explains why over-aggregating is difficult to self-correct during training: small attention scores produce weak gradient signals.
- Wideformer consistently reduces attention entropy and improves classification performance across all 13 datasets.

## Highlights & Insights
- **Discovery of a New Phenomenon (Over-Aggregating)**: Unlike over-smoothing (across GNN layers) and over-squashing (at bottleneck edges), over-aggregating occurs within a single-step global aggregation and is specific to Graph Transformers.
- **Dual Theoretical and Gradient Analysis**: Theorem 3.1 formally proves that the entropy lower bound increases monotonically with $n$, and gradient analysis further explains why training cannot self-correct this issue.
- **Plug-and-Play Design**: Wideformer requires no modification to backbone architectures and can be inserted as a standalone module, making it highly practical.
- **Clear Distinction from Over-Smoothing/Over-Squashing**: The paper clearly delineates over-aggregating from related phenomena, contributing to the community's understanding of distinct information-loss mechanisms on graphs.

## Limitations & Future Work
- **Modest Performance Gains**: Improvements on some datasets are below 1%, suggesting that over-aggregating is not the primary bottleneck in all settings.
- **Cluster Count $m$ Selection**: Requires tuning, and the optimal value may vary across datasets.
- **Hard Assignment**: The current argmax-based hard cluster assignment may lack flexibility.
- **Future Directions**: Exploring soft assignment; adaptive selection of $m$; integration with sparse attention methods.

## Related Work & Insights
- **vs. Over-Smoothing**: Over-smoothing refers to representational convergence across GNN layers; over-aggregating refers to information dilution in a single-step global aggregation. The two are orthogonal.
- **vs. Over-Squashing**: Over-squashing concerns information compression caused by bottleneck edges; over-aggregating concerns indiscriminate mixing in global aggregation.
- **vs. Sparse Attention**: Sparse attention alleviates the problem by restricting the receptive field; Wideformer preserves the global receptive field while decomposing the aggregation process.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel phenomenon identified with solid theoretical grounding
- Experimental Thoroughness: ⭐⭐⭐⭐ 13 datasets, 3 backbone models
- Writing Quality: ⭐⭐⭐⭐ Problem formulation and analytical chain are clearly presented
- Value: ⭐⭐⭐⭐ Practical plug-and-play module, though performance gains are limited

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)
- [\[NeurIPS 2025\] From Sequence to Structure: Uncovering Substructure Reasoning in Transformers](from_sequence_to_structure_uncovering_substructure_reasoning_in_transformers.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](../../ICLR2026/graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[AAAI 2026\] Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs](../../AAAI2026/graph_learning/are_graph_transformers_necessary_efficient_long-range_messag.md)
- [\[NeurIPS 2025\] GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks](graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
