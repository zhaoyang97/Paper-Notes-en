---
title: >-
  [Paper Note] TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration
description: >-
  [ICML 2025][Graph Learning] TopInG proposes a topologically interpretable graph learning framework based on persistent homology. By learning a "rationale filtration", it identifies stable and persistent rationale subgraphs. It introduces a "topological discrepancy" constraint to reinforce the topological distinction between rationale and irrelevant subgraphs, significantly outperforming existing methods in handling variform rationale subgraphs.
tags:
  - "ICML 2025"
  - "Graph Learning"
date: 2026-05-08
content_hash: 43d854581e7ef462
---

# TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration

**Conference**: ICML 2025  
**arXiv**: [2510.05102](https://arxiv.org/abs/2510.05102)  
**Area**: Graph Learning  

## TL;DR

TopInG proposes a topologically interpretable graph learning framework based on persistent homology. By learning a "rationale filtration", it identifies stable and persistent rationale subgraphs. It introduces a "topological discrepancy" constraint to reinforce the topological distinction between rationale and irrelevant subgraphs, significantly outperforming existing methods in handling variform rationale subgraphs.

## Background & Motivation

GNNs have achieved remarkable success in scientific fields, but their lack of interpretability limits their adoption in critical decision-making. Existing inherently interpretable GNNs (such as GSAT, GMT, DIR) face a common issue:

**Overly strong implicit assumptions**: Existing methods typically assume that the rationale subgraphs of the same class of graphs are nearly invariant in form. In reality, however, **variform rationale subgraphs** are very common:

- In molecular biology, molecules with the same biological activity may possess different functional groups (aromatic rings, sulfonamide groups, heterocyclic compounds).
- In social networks, the structural reasons for user influence vary (high degree, high betweenness centrality, bridging nodes).

## Method

### Core Idea

A graph is viewed as a structure "growing" out of a core rationale subgraph $G_X^*$: key subgraphs are generated first, followed by the addition of auxiliary structures. Persistent homology is utilized to track the life cycles of topological features during this generation process.

### Rationale Filtration Learning

A backbone GNN is used as a learnable filtration function $f_\phi: G \to [0,1]^{|E|}$ to assign an importance score to each edge. These scores induce a **graph filtration**—a sequence of nested subgraphs:

$$\mathcal{F}(G) = \{G_0, G_1, \ldots, G_{|E|}\}$$

Edges are incrementally added in descending order of importance (more important edges are added first). The goal is to learn an importance ranking that is consistent with the partition of $G_X^*$ and $G_\epsilon^*$.

### Topological Discrepancy

The topological discrepancy is defined as the 1-Wasserstein distance between the persistent homology distributions:

$$d_{\text{topo}}(\mathbb{P}(\mathcal{T}_X), \mathbb{P}(\mathcal{T}_\epsilon)) \triangleq \inf_{\pi \in \Pi(\mathbb{P}(\mathcal{T}_X), \mathbb{P}(\mathcal{T}_\epsilon))} \mathbb{E}_{(P,Q) \sim \pi}[d_B(P,Q)]$$

where $d_B$ is the bottleneck distance.

### Loss & Training

$$\mathcal{L}(\phi) = \mathbb{E}_G[\mathcal{L}_{\text{ce}}(\hat{y}_G, y_G)] - \alpha \mathcal{L}_{\text{topo}}(\mathbb{P}(\mathcal{T}_X), \mathbb{P}(\mathcal{T}_\epsilon))$$

The topological distinction between rationale and complementary subgraphs is enhanced by maximizing the topological discrepancy.

### Computable Lower Bound

Kantorovich duality and Lipschitz continuous, learnable vectorization functions (Rational Hat structuring elements) are leveraged to provide a computable lower bound:

$$\varphi(p; \mathbf{c}, r) = \sum_{\mathbf{x} \in p} \frac{1}{1 + \|\mathbf{x} - \mathbf{c}\|_2} - \frac{1}{1 + ||r| - \|\mathbf{x} - \mathbf{c}\|_2|}$$

A 2-head attention mechanism is used to select the top-2 maximum values from $k=8$ Lipschitz representation functions.

### Prior Regularization

A two-component Gaussian mixture prior is introduced to cluster the edge filtration values:

$$\mathbb{P}_{prior} = w\mathcal{N}(\mu_1, r_1) + (1-w)\mathcal{N}(\mu_2, r_2)$$

By fixing $w=0.5$, $\mu_1=0.25$, and $\mu_2=0.75$, unsupervised clustering of $G_X$ and $G_\epsilon$ is achieved.

### Theoretical Guarantees

**Theorem 3.4**: Assuming that for all $G$, $|E_X| < |E_\epsilon|$ and $G_X^*$ is minimal with respect to $y_G$, the loss $\mathcal{L}(\phi)$ is **uniquely optimized** by $f_\phi^*(e) = \mathbb{1}\{e \in G_X^*\}$.

Key Advantage: This guarantee **does not depend** on assumptions about the stability or invariance of the rationale subgraphs, making it unaffected by variform subgraphs.

## Experiments

### Explanatory Performance (AUC)

| Method | BA-2Motifs | BA-HouseGrid | SPMotif0.9 | BA-HouseAndGrid | BA-HouseOrGrid | Benzene |
|---|---|---|---|---|---|---|
| GIN+GSAT | 98.85 | 98.55 | 65.25 | 92.92 | 83.56 | 91.57 |
| GIN+GMT-Lin | 97.72 | 85.68 | 69.08 | 76.12 | 74.36 | 83.90 |
| **GIN+TopInG** | **99.57** | **99.24** | **80.82** | **95.35** | **88.56** | **98.22** |
| CINpp+GSAT | 91.12 | 91.04 | 80.24 | 95.17 | 69.30 | 95.40 |
| **CINpp+TopInG** | **100.00** | **99.87** | **92.82** | **100.00** | **100.00** | **98.72** |

**Key Findings**:
- The improvements are particularly significant on variform rationale subgraph benchmarks (BA-HouseAndGrid, BA-HouseOrGrid).
- CINpp+TopInG achieves 100% AUC on multiple datasets.
- The improvement on SPMotif0.9 (which has high spurious correlation) indicates that the method effectively mitigates spurious correlations.

### Predictive Performance

While maintaining high interpretability, TopInG's predictive accuracy also outperforms or matches baseline methods, overcoming the traditional trade-off between performance and interpretability.

## Highlights & Insights

- **First to introduce persistent homology to inherently interpretable GNNs**: It provides a brand-new topological perspective to solve the rationale subgraph identification problem.
- **Unique ability to handle variform subgraphs**: It demonstrates a significant advantage over existing methods on synthetic datasets.
- **Theoretical guarantee independent of invariance assumptions**: Rationales can vary freely across different instances.
- **Adaptive topological constraints**: The prior regularization is insensitive to hyperparameters and requires no tuning.
- **Compatible with existing GNN backbones**: It can be paired with various architectures such as GIN and CINpp.

## Limitations & Future Work

- The computation of persistent homology increases training time.
- Only 0-dimensional and 1-dimensional persistent homology (connected components and cycles) are considered, leaving higher-order topological features unexploited.
- The extension from node filtration to edge filtration (star filtration) has limited information.
- Performance on the Mutag dataset is slightly lower than that of MAGE.
- The theoretical guarantee relies on the assumption $|E_X| < |E_\epsilon|$, which may not always hold.

## Rating

⭐⭐⭐⭐⭐ (5/5)

This is an outstanding work that excellently combines theory and practice. Introducing topological data analysis to interpretable GNNs is a clever and original idea. The theoretical guarantees are rigorous, and the experiments demonstrate significant advantages across multiple dimensions, especially in tackling the key challenge of variform subgraphs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Positional Encoding meets Persistent Homology on Graphs](positional_encoding_meets_persistent_homology_on_graphs.md)
- [\[ICML 2025\] Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees](towards_graph_foundation_models_learning_generalities_across_graphs_via_task-tre.md)
- [\[ACL 2026\] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](../../ACL2026/graph_learning/logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)
- [\[ICML 2025\] Banyan: Improved Representation Learning with Explicit Structure](banyan_improved_representation_learning_with_explicit_structure.md)
- [\[ICLR 2026\] Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs](../../ICLR2026/graph_learning/self-consistency_improves_the_trustworthiness_of_self-interpretable_gnns.md)

</div>

<!-- RELATED:END -->
