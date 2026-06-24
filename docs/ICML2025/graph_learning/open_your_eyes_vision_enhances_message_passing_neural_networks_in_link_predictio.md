---
title: >-
  [Paper Note] Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction
description: >-
  [ICML2025][Graph Learning][Graph Neural Networks] This work introduces visual perception into message-passing neural networks (MPNNs) for the first time. By visualizing subgraphs as images and leveraging a vision encoder to extract Visual Structural Features (VSFs), the proposed GVN/E-GVN framework achieves state-of-the-art (SOTA) performance across 7 link prediction benchmarks.
tags:
  - "ICML2025"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Link Prediction"
  - "Visual Perception"
  - "Multimodal Learning"
  - "Structural Features"
date: 2026-05-08
content_hash: 48eab36d5b5157e8
---

# Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction

**Conference**: ICML2025  
**arXiv**: [2505.08266](https://arxiv.org/abs/2505.08266)  
**Code**: [WEIYanbin1999/EGVN](https://github.com/WEIYanbin1999/EGVN)  
**Area**: Graph Learning / Link Prediction  
**Keywords**: Graph Neural Networks, Link Prediction, Visual Perception, Multimodal Learning, Structural Features

## TL;DR

This work introduces visual perception into message-passing neural networks (MPNNs) for the first time. By visualizing subgraphs as images and leveraging a vision encoder to extract Visual Structural Features (VSFs), the proposed GVN/E-GVN framework achieves state-of-the-art (SOTA) performance across 7 link prediction benchmarks.

## Background & Motivation

- **Link prediction** is a core task in graph machine learning, widely used in recommendation systems, drug-drug interaction prediction, and knowledge graph reasoning.
- The prevailing approaches are **structural feature-enhanced MPNNs** (e.g., SEAL, BUDDY, NCNC), which augment the expressiveness of MPNNs using hand-crafted topological heuristics (such as CN, SPD, etc.).
- However, one of the most intuitive ways for humans to understand graph data is **visual perception**—directly "seeing" the graph structure—a modality that has been completely overlooked by the MPNN community.
- Vision encoders (such as VGG, ResNet, ViT) have brought immense improvements to fields like NLP, making the introduction of vision into graph learning both reasonable and timely.
- The paper proposes three core research questions: (1) How to perceive graph structure from a visual perspective? (2) Does vision truly benefit link prediction? (3) How to effectively integrate vision with MPNNs?

## Method

### Overall Architecture: Graph Vision Network (GVN)

GVN integrates visual information into MPNNs in three steps:

**Step 1: Scope-Decoupled Subgraph Visualization**

- For a query edge $(u,v)$, its $k$-hop enclosing subgraph $S_{uv}^k$ is extracted.
- The subgraph is rendered into an image using tools like Graphviz: $I_{uv}^k = \text{GV}(S_{uv}^k, u, v)$.
- Key design principles:
    - **Style Consistency**: All subgraphs use the same layout algorithm, node colors, and shapes.
    - **Highlighting Query Edges**: Target nodes are highlighted with color, and the target edge is masked.
    - **Removing Node Labels**: This forces the model to focus on topological structures rather than label information.
- **Decoupling Vision and Message Passing Scopes**: The visual perception scope is fixed at $k \leq 3$ and does not change with the number of MPNN layers, avoiding oversmoothing and neighbor explosion.

**Step 2: Adaptive Extraction of VSFs**

- The visualized image is passed through a vision encoder (defaulting to ResNet50) to extract visual structural features:

$$\mathbf{v}_{uv} = \text{VE}_\psi(I_{uv}^k) \in \mathbb{R}^S$$

- Three major advantages of VSFs:
    - **Link Discrimination**: MPNNs cannot distinguish links between isomorphic nodes, but visualized subgraph images can be significantly different.
    - **Fine-Grained Substructure Perception**: VSF significantly improves the MPNN's ability to count substructures like triangles and 3-stars (reducing error by 6 to 8 orders of magnitude).
    - **Rich and Adaptive Information**: VSFs can reproduce various structural features like CN, RA, AA, SPD, DRNL, and DE (reproduction rates of 72%–83%) and can adaptively adjust information distribution through fine-tuning.

**Step 3: Compatible Features Integration**

Three fusion strategies are provided, all independent of the message passing process:

1. **Attention Fusion** (default): First linearly project $\tilde{\mathbf{v}}_{uv} = \text{Linear}(\mathbf{v}_{uv})$, then use cross-attention to update node representations: $\tilde{\mathbf{y}}_u = \text{CA}(\mathbf{y}_u, \tilde{\mathbf{v}}_{uv})$.
2. **Concatenation Fusion**: $\tilde{\mathbf{y}}_u = \mathbf{y}_u \| \mathbf{v}_{uv}$.
3. **Weighted Fusion**: $p(u,v) = \delta \cdot p_{\text{vision}} + (1-\delta) \cdot p_{\text{MPNN}}$, where $\delta$ is learnable.

### E-GVN: Efficient Variant

Scalability optimization for large-scale graphs:

| Design | GVN | E-GVN |
|------|-----|-------|
| Visualization Granularity | Edge-centric $S_{uv}^k$ | Node-centric $S_v^k$ |
| Number of Visualizations | $O(l)$ (number of edges) | $O(n)$ (number of nodes) |
| Vision Encoder | End-to-end Fine-tuning | Frozen + Trainable Adapter |
| Fusion Position | After MPNN | Before MPNN (integrated into node attributes) |

The Adapter mechanism of E-GVN: $\tilde{\mathbf{v}}_v = \text{Adaptor}_\phi(\mathbf{v}_v)$. Freezing the encoder reduces storage overhead, while the Adapter retains adaptive capacity.

## Key Experimental Results

### Main Results: 7 Link Prediction Benchmarks (Table 3)

| Method | Cora | Citeseer | Pubmed | ogbl-collab | ogbl-ppa | ogbl-citation2 | ogbl-ddi |
|------|------|----------|--------|-------------|----------|----------------|----------|
| GCN | 66.79 | 67.08 | 53.02 | 44.75 | 18.67 | 84.74 | 37.07 |
| SEAL| 81.71 | 83.89 | 75.54 | 64.74 | 48.80 | 87.67 | 30.56 |
| BUDDY | 88.00 | 92.93 | 74.10 | 65.94 | 49.85 | 87.56 | 78.51 |
| NCNC | 89.65 | 93.47 | 81.29 | 66.61 | 61.42 | 89.12 | 84.11 |
| **GVN_NCNC** | 90.70 | 94.12 | 82.17 | - | - | - | - |
| **E-GVN_NCNC** | **91.47** | **94.44** | **84.02** | **68.14** | **63.45** | **90.72** | **87.31** |

- E-GVN_NCNC achieves the best performance across all 7 datasets and scales effectively to large-scale OGB datasets.
- GVN is effective on small-scale datasets but fails to scale to large-scale graphs (>12h/epoch).

### Substructure Counting Experiment (Table 1)

| Method | Triangle (Best) | 3-Star (Best) |
|------|-------------|-------------|
| GCN | 0.69 | 0.49 |
| VSF+GCN | **6.76E-9** | **3.22E-8** |

VSF reduces the substructure counting error by approximately **8 orders of magnitude**.

### VSF Information Richness

On the ogbl-ddi dataset, VSF's reproduction rate of various structural features: CN=80.23%, RA=78.47%, AA=78.89%, SPD=75.85%, DRNL=73.34%, DE=72.12%.

## Highlights & Insights

1. **Pioneering Perspective**: Introduces the visual modality into MPNN link prediction for the first time, opening up an entirely new research direction.
2. **Solid Theoretical Analysis**: Demonstrates the advantages of VSFs from three aspects: link discrimination power, substructure perception, and information richness.
3. **Orthogonal Compatibility**: GVN/E-GVN can be seamlessly integrated with any existing MPNN methods, delivering consistent improvements.
4. **Scope Decoupling Design**: Decoupling the visual perception range from the message passing range elegantly avoids oversmoothing and neighbor explosion.
5. **Adaptive Feature Distribution**: VSFs can automatically adjust focus on different structural features via fine-tuning—downplaying path features on dense graphs while enhancing neighbor features on sparse graphs.

## Limitations & Future Work

1. **Computational Overhead**: Even with the optimization of E-GVN, generating visualized images for each node/edge still introduces additional preprocessing overhead.
2. **Non-Permutation Equivariance**: VSFs are not permutation equivariant. While they perform well empirically, theoretical guarantees remain limited.
3. **Vision Encoder Selection**: ResNet50 is used by default, leaving the potential of more powerful vision encoders like ViTs not fully explored.
4. **Visualizer Dependency**: The visual representation of graphs heavily depends on the layout algorithm of Graphviz, and different layouts may impact performance.
5. **Limited to Link Prediction**: The effects of vision-enhancement on other graph tasks, such as node classification and graph classification, remain unexplored.

## Related Work & Insights

- **NCNC** (Wang et al., 2024): The current SOTA structural feature-augmented MPNN and the primary baseline for GVN.
- **SEAL** (Zhang & Chen, 2018): An SPD-based subgraph method that inspired the concept of subgraph visualization.
- **BUDDY** (Chamberlain et al., 2023): High-order common neighbor modeling, offering complementary information to VSFs.
- **Insight**: The concept of using the visual modality as a "universal structural feature pool" can be extended to more graph learning tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introduces visual perception into MPNN link prediction for the first time, demonstrating exceptional pioneering quality.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 7 datasets, multiple baselines, ablation studies, substructure analysis, and adaptiveness analyses.
- Writing Quality: ⭐⭐⭐⭐ — Structurally clear with three RQs and detailed analysis, though somewhat heavy on notation.
- Value: ⭐⭐⭐⭐⭐ — Opens up a new direction for graph learning + vision with solid empirical results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Beyond Message Passing: Neural Graph Pattern Machine](beyond_message_passing_neural_graph_pattern_machine.md)
- [\[ICML 2025\] L-STEP: Learnable Spatial-Temporal Positional Encoding for Link Prediction](learnable_spatial-temporal_positional_encoding_for_link_prediction.md)
- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](../../NeurIPS2025/graph_learning/tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[NeurIPS 2025\] What Expressivity Theory Misses: Message Passing Complexity for GNNs](../../NeurIPS2025/graph_learning/what_expressivity_theory_misses_message_passing_complexity_for_gnns.md)
- [\[NeurIPS 2025\] OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](../../NeurIPS2025/graph_learning/ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)

</div>

<!-- RELATED:END -->
