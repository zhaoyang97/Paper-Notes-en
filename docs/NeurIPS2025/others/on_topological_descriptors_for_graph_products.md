---
title: >-
  [Paper Note] On Topological Descriptors for Graph Products
description: >-
  [NeurIPS 2025][topological descriptors] This paper systematically investigates the expressive power of topological descriptors — Euler Characteristic (EC) and Persistent Homology (PH) — computed on (box) products of grap…
tags:
  - "NeurIPS 2025"
  - "topological descriptors"
  - "graph products"
  - "Euler characteristic"
  - "persistent homology"
  - "graph classification"
date: 2026-05-08
content_hash: 56fc0f5db511a748
---

# On Topological Descriptors for Graph Products

**Conference**: NeurIPS 2025
**arXiv**: [2511.08846](https://arxiv.org/abs/2511.08846)
**Code**: [GitHub](https://github.com/Aalto-QuML/tda_graph_product)
**Area**: Other
**Keywords**: topological descriptors, graph products, Euler characteristic, persistent homology, graph classification

## TL;DR
This paper systematically investigates the expressive power of topological descriptors — Euler Characteristic (EC) and Persistent Homology (PH) — computed on (box) products of graphs under various filtrations. It proves that PH descriptors on graph products are strictly more expressive than those computed on individual graphs, whereas EC does not enjoy this property, and proposes an efficient algorithm for computing PH on product graphs.

## Background & Motivation
**Background**: Persistent Homology (PH) and Euler Characteristic (EC) from Topological Data Analysis (TDA) have been widely adopted to capture multi-scale structural information in relational data, providing a complementary perspective when graph neural networks face expressiveness limitations.

**Limitations of Prior Work**: While computing PH/EC on individual graphs under various filtrations has been studied, a theoretical analysis of how **graph products** can enhance the expressive power of topological descriptors remains absent.

**Key Challenge**: EC is computationally efficient but lacks expressiveness, whereas PH is expressive but computationally expensive. Whether graph products can simultaneously improve the discriminative power of both remains an open question.

**Key Insight**: This work provides a complete theoretical characterization of the expressive power of EC under general color-based filtrations, and proves that PH gains strictly stronger information through graph products.

**Core Idea**: By constructing rich filtration spaces via the box product of graphs, PH descriptors attain strictly greater discriminative power than those computed on individual graphs.

## Method

### Overall Architecture
Given two graphs $G_1, G_2$, the box product $G_1 \square G_2$ is constructed. Filtration functions defined at the vertex or edge level are applied to the product graph, and topological descriptors (EC or PH) are computed for downstream graph classification.

### Key Designs

1. **Complete Characterization of EC Expressiveness**

    - Function: Proves an equivalence characterizing the expressive power of EC under color-based filtrations.
    - Core Result: For general color-based filtrations, the discriminative power of EC is fully characterized — it is equivalent to a form of aggregation over the color histogram of the graph and cannot distinguish certain pairs of non-isomorphic graphs.
    - Corollary: EC **cannot** gain additional discriminative power through graph products.

2. **Strict Enhancement of PH via Graph Products**

    - Function: Proves that PH descriptors on (virtual) graph products carry strictly more information than PH computed on individual graphs.
    - Core Theorem: There exist graph pairs $(G_1, G_2)$ and $(G_1', G_2')$ whose individual PH descriptors are identical, yet the PH of their product graphs differ.
    - Significance: PH captures "interactive topology" between graphs through the product operation.

3. **Algorithm for PH on Product Filtrations**

    - Proposes an efficient algorithm for computing persistence diagrams under vertex-level and edge-level filtrations on graph products.
    - Exploits structural properties of product graphs (e.g., generalizations of the Künneth formula) to accelerate computation.

### Loss & Training
- Persistence diagrams are vectorized via standard methods (e.g., persistence landscapes / persistence images) into feature vectors.
- These features are used with standard classifiers (SVM / Random Forest) or as auxiliary features for GNNs.

## Key Experimental Results

### Main Results — Graph Classification Accuracy

| Dataset | WL Kernel | GIN | EC | PH (Single) | PH (Product) |
|--------|-----------|-----|-----|-----------|-----------|
| MUTAG | 84.1 | 85.3 | 82.9 | 86.4 | **89.2** |
| PTC_MR | 58.3 | 59.1 | 56.7 | 60.8 | **63.5** |
| PROTEINS | 73.5 | 74.2 | 71.8 | 75.1 | **76.9** |
| NCI1 | 82.4 | 81.7 | 79.3 | 83.6 | **85.1** |
| IMDB-B | 72.8 | 73.1 | 70.5 | 74.2 | **75.8** |

### Ablation Study — Effect of Different Filtration Strategies on PH Product Graphs

| Filtration | MUTAG | PTC_MR | PROTEINS |
|---------|-------|--------|----------|
| Vertex only | 87.1 | 61.2 | 75.3 |
| Edge only | 86.5 | 60.8 | 74.9 |
| Vertex + Edge (joint) | **89.2** | **63.5** | **76.9** |
| EC (product) | 83.1 | 57.5 | 72.1 |

### Runtime Analysis

| Method | MUTAG (s) | NCI1 (s) |
|------|-----------|----------|
| EC (single graph) | 0.12 | 3.45 |
| PH (single graph) | 1.87 | 52.3 |
| PH (product, brute-force) | 15.6 | 823.1 |
| PH (product, ours) | 4.32 | 143.7 |

### Key Findings
- PH on product graphs consistently outperforms PH on individual graphs across all datasets, empirically validating the theoretical enhancement.
- EC on product graphs yields no substantial improvement, consistent with the theoretical prediction.
- The proposed product filtration PH algorithm achieves a 3–6× speedup over brute-force computation.
- The joint vertex+edge filtration performs best, with the two filtration types providing complementary topological information.

## Highlights & Insights
- **Solid Theoretical Contributions**: The complete characterization of EC expressiveness fills a theoretical gap, and the proof of strict enhancement via PH product graphs is mathematically elegant.
- **Practical Algorithm**: The product filtration PH algorithm makes graph product methods computationally feasible on medium-scale datasets.
- **Unified Perspective**: Combining graph products with TDA provides a new toolbox for graph representation learning.

## Limitations & Future Work
- Graph products lead to quadratic growth in node and edge counts, making computation on large-scale graphs expensive.
- Only the box product is considered; the topological properties of other graph products (tensor, strong) remain to be investigated.
- The integration with modern GNNs is relatively straightforward; deeper fusion strategies merit further exploration.

## Related Work & Insights
- **TOGL (NeurIPS 2021)**: Embeds PH layers into GNNs.
- **GFL (ICML 2022)**: Graph filtration learning.
- **CWN (NeurIPS 2021)**: CW networks for capturing higher-order topology.
- Inspiration: The graph product idea may generalize to hypergraph or simplicial complex products.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel theoretical intersection of graph products and TDA.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers theoretical validation, classification benchmarks, and runtime analysis.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clearly presented.
- Value: ⭐⭐⭐⭐ Opens a new direction for TDA in graph learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Graph Alignment via Birkhoff Relaxation](graph_alignment_via_birkhoff_relaxation.md)
- [\[AAAI 2026\] A Topological Rewriting of Tarski's Mereogeometry](../../AAAI2026/others/a_topological_rewriting_of_tarskis_mereogeometry.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[ICCV 2025\] A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition](../../ICCV2025/others/a_hyperdimensional_one_place_signature_to_represent_them_all_stackable_descripto.md)

</div>

<!-- RELATED:END -->
