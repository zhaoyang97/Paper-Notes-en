---
title: >-
  [Paper Note] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation
description: >-
  [AAAI 2026][Graph Learning][Logical Expressiveness] PN-GNN proposes aggregating neighbor node embeddings along reasoning paths on top of conditional message passing…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Logical Expressiveness"
  - "Knowledge Graph Reasoning"
  - "Path-Neighbor Aggregation"
  - "Conditional GNN"
  - "Labeling Trick"
date: 2026-05-08
content_hash: 8ae1b7fd42b54134
---

# Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation

**Conference**: AAAI 2026
**arXiv**: [2511.07994](https://arxiv.org/abs/2511.07994)  
**Code**: None  
**Area**: Graph Learning / Knowledge Graph Reasoning
**Keywords**: Logical Expressiveness, Knowledge Graph Reasoning, Path-Neighbor Aggregation, Conditional GNN, Labeling Trick

## TL;DR

PN-GNN proposes aggregating neighbor node embeddings along reasoning paths on top of conditional message passing, enhancing the logical rule expressiveness of GNNs (strictly beyond C-GNN) in a plug-and-play manner, while avoiding the generalization degradation caused by the labeling trick. The method achieves improvements on both synthetic datasets and real-world knowledge graph reasoning tasks.

## Background & Motivation

**Background**: GNNs have demonstrated strong performance in knowledge graph (KG) reasoning. Their expressiveness can be evaluated from two perspectives: the ability to distinguish non-isomorphic graphs (related to the WL test) and the ability to learn specific logical rule structures. R-GNNs (e.g., R-GCN, CompGCN) perform relation-aware message passing but have limited rule-learning capability; C-GNNs (e.g., NBFNet, RED-GNN) perform conditional message passing by marking the head entity, and are theoretically capable of learning all CML (Counting Modal Logic) formulas.

**Limitations of Prior Work**: The logical expressiveness of C-GNNs is equivalent to CML; however, CML cannot express certain important rule structures, most notably the U-structure—where two paths branch from the same intermediate node and then converge. C-GNNs cannot distinguish T-structures from U-structures, as both yield identical CML expressions.

**Key Challenge**: EL-GNN augments distinguishing ability by adding constant labels to nodes, enabling it to learn U-structures, but at the cost of reduced coverage—substituting variables with constants narrows the applicability of rules, weakening generalization, and making the approach difficult to apply in inductive settings.

## Method

### Overall Architecture

PN-GNN (Path-Neighbor enhanced GNN) is a plug-and-play module stacked on top of a C-GNN backbone (e.g., NBFNet):

1. Apply $L$ layers of conditional message passing via C-GNN to obtain entity-pair representations.
2. Aggregate representations of neighbor nodes along the reasoning path.
3. Fuse the aggregated path-neighbor information with the C-GNN representations.

### Key Designs

**1. Path-Neighbor Aggregation**

Given head entity $u$, tail entity $v$, and query relation $q$, PN-GNN aggregates neighbor nodes on the reasoning path $P_{uv}$:

$$h_{ij} = \text{POOL}\left(\{h_{w|u,q}^{(L)} \mid w \in P_{uv},\ d(u,w)=i,\ d(w,v)=j\}\right)$$

where $i$ denotes the distance from $w$ to the head entity and $j$ the distance from $w$ to the tail entity. POOL can be any standard pooling function such as max, min, or mean.

**2. Representation Fusion**

The path-neighbor aggregation result is fused with the C-GNN representation:

$$h_d = \text{MLP}_{11}(h_{11}) \cdot \text{MLP}_{12}(h_{12}) \cdot \text{MLP}_{21}(h_{21})$$

$$h_{v|u,q} = h_d \cdot h_{v|u,q}^{(L)}$$

To balance expressiveness and efficiency, 2-hop neighbors are used by default:
- $h_{11}$: neighbors 1 hop from both head and tail entities
- $h_{12}$: neighbors 1 hop from the head and 2 hops from the tail
- $h_{21}$: neighbors 2 hops from the head and 1 hop from the tail

**3. Why PN-GNN Can Distinguish U- and T-Structures**

- T-structure: two chains originate from the same head entity and reach the same tail entity, potentially through different intermediate nodes.
- U-structure: similar to T, but both chains are required to branch from the same intermediate node.

C-GNN cannot express the constraint "two edges originate from the same node" using CML. PN-GNN captures this by aggregating path-neighbor information: the resulting $h_{11}$ differs between T- and U-structures, as in the U-structure the aggregation at the branching node reflects the "shared branching point" structural feature.

**4. Theoretical Analysis**

- **Lemma 7**: Every CML formula learnable by C-GNN is also learnable by PN-GNN.
- **Lemma 8**: PN-GNN can learn U-structures (C-GNN cannot).
- **Theorem 9**: PN-GNN is strictly more expressive than C-GNN in terms of logical expressiveness.
- **Theorem 12**: $(k+1)$-hop PN-GNN is strictly more expressive than $k$-hop PN-GNN.

**5. Advantages over the Labeling Trick**

The labeling trick substitutes variables with constants, causing reduced coverage and incompatibility with inductive settings. PN-GNN introduces no constant labels, preserves the flexibility of the original variables, maintains full coverage, and naturally extends to inductive settings.

### Loss & Training

Built on the C-GNN backbone (NBFNet), PN-GNN uses sigmoid scoring to predict the conditional probability of the tail entity, with negative log-likelihood as the training loss. Negative samples are constructed under the PCA assumption by randomly corrupting head or tail entities.

## Key Experimental Results

### Logical Rule Learning on Synthetic Datasets (Hits@1)

| Method | C3 | C4 | I1 | I2 | T | U | T_label | U_label |
|--------|----|----|----|----|---|---|---------|---------|
| R-GCN | 1.6 | 3.1 | 4.4 | 2.4 | 6.7 | 1.4 | — | — |
| NBFNet | 100 | 100 | 100 | 100 | 100 | 54.1 | 60.0 | 56.8 |
| EL-GNN | 100 | 100 | 100 | 100 | 100 | 75.7 | 22.0 | 59.5 |
| **PN-GNN** | **100** | **100** | **100** | **100** | **100** | **69.9** | **60.0** | **68.9** |

PN-GNN improves Hits@1 on U by 15.8% over NBFNet, and substantially outperforms EL-GNN on T_label and U_label.

### Transductive Reasoning on Real-World Datasets

| Method | FB15K237 MRR | FB15K237 H@10 | WN18RR MRR | WN18RR H@10 |
|--------|-------------|--------------|------------|-------------|
| NBFNet | 0.415 | 59.9 | 0.551 | 66.6 |
| EL-GNN | 0.421 | 59.8 | 0.555 | 66.4 |
| **PN-GNN** | **0.423** | **60.2** | **0.555** | **66.9** |

### Ablation Study

| Method | U (H@1) | FB15K237v1 (H@10) | WN18RRv1 (H@10) |
|--------|---------|-------------------|-----------------|
| NBFNet | 54.1 | 17.1 | 60.1 |
| PN-GNN$_{11}$ (1-hop only) | 48.7 | 20.2 | **62.8** |
| PN-GNN$_{12\text{-}21}$ (2-hop only) | **69.9** | 18.5 | 62.5 |
| PN-GNN (full) | **69.9** | **22.1** | 62.5 |

### Key Findings

- Distinguishing U-structures relies primarily on 2-hop neighbors (PN-GNN$_{12\text{-}21}$ is equivalent to the full model on this task).
- For WN18RR, which features fewer relations and predominantly short paths, 1-hop neighbors are most effective.
- EL-GNN suffers a dramatic performance drop on the T_label dataset (22.0 vs. 100), confirming that the labeling trick harms generalization.
- PN-GNN remains effective in inductive settings, where EL-GNN is not applicable.

## Highlights & Insights

- The paper analyzes GNN limitations from the perspective of logical expressiveness rather than solely focusing on WL-test discrimination power.
- Path-neighbor aggregation is a plug-and-play enhancement that does not alter the fundamental training pipeline of C-GNNs.
- The concept of coverage formalizes the generalization ability of rules, providing a clear explanation of the side effects of the labeling trick.
- The theoretical result that $(k+1)$-hop is strictly more expressive than $k$-hop provides principled guidance for selecting the propagation range.

## Limitations & Future Work

- Computational cost grows with the number of hops in multi-hop settings; the current approach is limited to 2-hop neighbors to balance efficiency.
- The method is instantiated only on NBFNet; it should theoretically be extended to other C-GNNs such as RED-GNN and A*Net.
- The synthetic datasets are relatively simple; the actual prevalence and importance of U-structures in real-world KGs warrants further investigation.
- Path-neighbor selection relies on exact path distance computation, which may become a bottleneck on large-scale graphs.

## Related Work & Insights

- **NBFNet**: Conditional message passing based on the Bellman-Ford algorithm; serves as the backbone for PN-GNN.
- **EL-GNN**: Enhances logical expressiveness via the labeling trick at the cost of generalization.
- **R-WL test**: Extends the WL test to relational graphs, formalizing an upper bound on the expressiveness of R-GNNs.
- **GraIL / CoMPILE**: Subgraph-based methods for inductive reasoning; PN-GNN outperforms these baselines in inductive settings.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Theoretical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 3 |
| Practicality | 4 |
| Overall | 3.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Logical Characterizations of GNNs with Mean Aggregation](logical_characterizations_of_gnns_with_mean_aggregation.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)
- [\[AAAI 2026\] Adaptive Initial Residual Connections for GNNs with Theoretical Guarantees](adaptive_initial_residual_connections_for_gnns_with_theoretical_guarantees.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](../../NeurIPS2025/graph_learning/logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)

</div>

<!-- RELATED:END -->
