---
title: >-
  [Paper Note] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction
description: >-
  [AAAI2026][Graph Learning][Knowledge Graph] This paper proposes UniHR, a unified framework that converts hyper-relational, temporal, and nested KGs into a triple-based representation via Hierarchical Data Representation (HiDR), and designs a Hierarchical Structure Learning (HiSL) module for two-stage intra-fact and inter-fact message passing. UniHR achieves state-of-the-art or competitive link prediction results across 9 datasets spanning 5 KG types.
tags:
  - AAAI2026
  - Graph Learning
  - Knowledge Graph
  - Link Prediction
  - Hyper-relational KG
  - Temporal KG
  - Nested KG
  - Unified Representation Learning
  - Hierarchical Message Passing
date: 2026-05-08
content_hash: cd86915dad94ce77
---

# UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction

**Conference**: AAAI2026  
**arXiv**: [2411.07019](https://arxiv.org/abs/2411.07019)  
**Authors**: Zhiqiang Liu, Yin Hua, Mingyang Chen, Yichi Zhang, Zhuo Chen, Lei Liang, Wen Zhang (ZJU)  
**Code**: [zjukg/UniHR](https://github.com/zjukg/UniHR)  
**Area**: Graph Learning  
**Keywords**: Knowledge Graph, Link Prediction, Hyper-relational KG, Temporal KG, Nested KG, Unified Representation Learning, Hierarchical Message Passing  

## TL;DR

This paper proposes UniHR, a unified framework that converts hyper-relational, temporal, and nested KGs into a triple-based representation via Hierarchical Data Representation (HiDR), and designs a Hierarchical Structure Learning (HiSL) module for two-stage intra-fact and inter-fact message passing. UniHR achieves state-of-the-art or competitive link prediction results across 9 datasets spanning 5 KG types.

## Background & Motivation

### Heterogeneity of Real-World KGs

Large-scale real-world knowledge graphs (e.g., Wikidata, DBpedia) contain not only standard triples $(h, r, t)$ but also more complex fact forms: hyper-relational facts with auxiliary key-value pairs, temporal facts with timestamps, and nested facts expressing relationships between facts. For example, "Oppenheimer obtained a bachelor's degree in chemistry from Harvard" cannot be represented by a simple triple and requires the hyper-relational form $((h,r,t), \{(k_i:v_i)\}_{i=1}^m)$. These richer representations have attracted broad attention due to their enhanced semantic expressiveness.

### Two Key Limitations of Existing Methods

Prior work suffers from two main problems: (1) most methods are designed for a specific KG type (e.g., HKG, TKG, or NKG) and do not generalize to real-world scenarios containing multiple fact types; (2) the complexity of beyond-triple representations makes it difficult to achieve generalizable hierarchical modeling that captures both intra-fact semantics and inter-fact structural information. For instance, StarE customizes a GNN for HKG but cannot flexibly capture key-value information; NestE scores only within individual facts while ignoring global structure; ECEformer captures only inter-fact semantics for TKGs. Although HAHE begins to capture hierarchical semantics for HKGs, its heterogeneous representation limits extension to other KG types.

### Value of a Unified Approach

Developing a unified hierarchical representation learning method that simultaneously handles multiple fact types and performs hierarchical semantic modeling has both theoretical value and practical necessity—Wikidata itself contains mixed fact types. A unified representation also paves the way for KG pre-training and cross-type joint training.

## Core Problem

How to design a unified framework that converts hyper-relational KGs, temporal KGs, and nested KGs into a common representation, and performs effective hierarchical structure learning (intra-fact + inter-fact) over this representation to enable general-purpose link prediction across KG types?

## Method

### Overall Architecture

UniHR consists of three steps: (1) the HiDR module converts any KG type into a unified triple-based graph $\mathcal{G}^{\text{HiDR}}$; (2) the HiSL module performs two-stage intra-fact and inter-fact message passing over $\mathcal{G}^{\text{HiDR}}$ to enhance node embeddings; (3) a Transformer decoder is used for link prediction.

### HiDR: Hierarchical Data Representation

HiDR classifies nodes into three types (atomic node $\mathcal{V}_a$, relation node $\mathcal{V}_r$, fact node $\mathcal{V}_f$), relations into three types (atomic relation $\mathcal{R}_a$, nested relation $\mathcal{R}_n$, connected relation $\mathcal{R}_c = \{\text{has\_relation, has\_head\_entity, has\_tail\_entity}\}$), and facts into three types (atomic facts $\mathcal{F}_a$, connected facts $\mathcal{F}_c$, nested facts $\mathcal{F}_n$).

- **HKG → HiDR**: The main triple $(h,r,t)$ is decomposed into connected facts $(f, \text{has\_relation}, e_r)$, $(f, \text{has\_head}, h)$, $(f, \text{has\_tail}, t)$, and an atomic fact $(h,r,t)$; key-value pairs are converted to $(f, k_i, v_i)$.
- **NKG → HiDR**: Atomic facts are preserved; nested facts $(f_1, R, f_2)$ are converted to nested facts.
- **TKG → HiDR**: Timestamps are first converted to auxiliary key-value pairs (begin: $\tau_b$, end: $\tau_e$), then processed following the HKG conversion procedure.

### HiSL: Hierarchical Structure Learning

**Representation Initialization**: Atomic node embeddings $\mathbf{H}_a \in \mathbb{R}^{|\mathcal{V}_a| \times d}$; relation node embeddings are generated from relation edges via a projection matrix: $\mathbf{H}_r = \mathbf{E}_a \cdot \mathbf{W}_r$. Fact node embeddings are initialized from the main triple:
$$\mathbf{h}_f = f_m([\mathbf{h}_h; \mathbf{h}_r; \mathbf{h}_t])$$
where $f_m: \mathbb{R}^{3d} \rightarrow \mathbb{R}^d$ is a single-layer MLP. Timestamps are encoded using Time2Vec.

**Intra-fact Message Passing**: For each fact node $f_k$, a one-hop neighborhood subgraph is constructed and graph attention is used to aggregate local semantic information:
$$\alpha_{i,j}^l = \frac{\exp(\mathbf{W}^l(\sigma(\mathbf{W}_{in}^l \mathbf{h}_i^l + \mathbf{W}_{out}^l \mathbf{h}_j^l)))}{\sum_{j' \in \mathcal{N}_i} \exp(\mathbf{W}^l(\sigma(\mathbf{W}_{in}^l \mathbf{h}_i^l + \mathbf{W}_{out}^l \mathbf{h}_{j'}^l)))}$$

**Inter-fact Message Passing**: Message passing is performed over the full $\mathcal{G}^{\text{HiDR}}$ using the circular correlation operator $\phi(\mathbf{h}_j, \mathbf{e}_r) = \mathbf{h}_j \star \mathbf{e}_r$, with fine-grained aggregation conditioned on edge direction $\lambda(r)$ and edge type $\tau(r)$:
$$\mathbf{h}_i^{l+1} = \sum_{(r,j) \in \mathcal{N}(i)} \sigma(\omega_{\tau(r)}^l) \mathbf{W}_{\lambda(r)}^l \phi(\mathbf{h}_j^l, \mathbf{e}_r^l) + \mathbf{W}_{self}^l \mathbf{h}_i^l$$

**Decoder**: A Transformer decoder operates on serialized embeddings and is trained with cross-entropy loss.

## Key Experimental Results

### HKG Link Prediction (WikiPeople, WD50K)

| Model | WikiPeople MRR | WikiPeople H@10 | WD50K MRR | WD50K H@10 |
|-------|---------------|----------------|-----------|------------|
| StarE | 0.458 | 0.611 | 0.309 | 0.452 |
| GRAN | 0.477 | 0.596 | 0.329 | 0.465 |
| ShrinkE | 0.485 | 0.601 | 0.345 | 0.482 |
| HAHE | 0.498 | 0.610 | 0.343 | 0.484 |
| HyperSAT | 0.493 | 0.610 | 0.345 | 0.489 |
| **UniHR** | **0.496** | **0.619** | **0.348** | 0.482 |

### NKG Triple Prediction (FBH, FBHE, DBHE)

| Model | FBH MRR | FBH MR | FBHE MRR | DBHE MRR |
|-------|---------|--------|----------|----------|
| BiVE | 0.855 | 6.20 | 0.711 | 0.687 |
| NestE | 0.922 | 3.34 | 0.851 | 0.862 |
| GRADATE | 0.780 | 18.15 | 0.603 | 0.654 |
| **UniHR** | **0.946** | **2.46** | 0.793 | **0.862** |

### TKG Link Prediction (wikidata12k)

| Model | MRR | H@1 | H@3 | H@10 |
|-------|-----|-----|-----|------|
| TGeomE+ | 0.333 | 0.232 | 0.361 | 0.546 |
| 5EL | 0.311 | 0.237 | 0.355 | 0.546 |
| **UniHR** | **0.334** | **0.242** | **0.368** | 0.527 |

### Hyper-relational TKG (WIKI-hy, YAGO-hy)

| Model | WIKI-hy MRR | WIKI-hy H@10 | YAGO-hy MRR | YAGO-hy H@10 |
|-------|------------|-------------|------------|-------------|
| HypeTKG | 0.687 | 0.789 | 0.832 | 0.857 |
| **UniHR** | **0.692** | **0.792** | **0.841** | **0.862** |

### Ablation Study (FBHE / DB15K / wikidata12k)

| Variant | FBHE MRR | DB15K MRR | wikidata12k MRR |
|---------|----------|-----------|----------------|
| w/o intra-fact MP | 0.754 | 0.341 | 0.321 |
| w/o inter-fact MP | 0.776 | 0.338 | 0.319 |
| UniHR (full) | **0.793** | **0.348** | **0.334** |

## Highlights & Insights

- **First unified KG representation learning framework**: UniHR handles HKG, TKG, and NKG within a single model; HiDR converts diverse fact forms into triples without information loss.
- **Parameter-efficient hierarchical learning**: The number of trainable parameters in HiSL does not scale with graph size; relation node and fact node embeddings are derived from atomic elements, avoiding parameter explosion.
- **Demonstrated potential beyond the unified setting**: Joint training, composite KGs (hyper-relational temporal KGs), and multi-task scenarios all show performance gains; wikimix joint training improves MR by 17.1% on HKG and 39.7% on TKG.
- **State-of-the-art or competitive results across 5 KG types on 9 datasets**, with triple prediction MRR reaching 0.946 on the NKG FBH dataset, surpassing NestE's 0.922.

## Limitations & Future Work

- **HiDR conversion introduces additional nodes and edges**: Although the authors claim modest storage overhead, scalability on very large KGs (e.g., full Wikidata) remains to be verified.
- **Transformer-based decoder**: Decoding over serialized embeddings may become an efficiency bottleneck for long sequences (e.g., hyper-relational facts with many key-value pairs).
- **H@10 on TKG slightly below TGeomE+**: This suggests that learning temporal information purely through graph structure is not always superior to dedicated temporal embedding methods on certain metrics.
- **No large-scale pre-training**: While the potential of the unified representation for pre-training is discussed, large-scale cross-KG pre-training is not empirically explored.

## Related Work & Insights

- **vs StarE**: StarE customizes a GNN for HKG but cannot flexibly capture key-value information; UniHR improves MRR on WD50K by 12.6% via HiDR + HiSL.
- **vs NestE**: NestE scores only within individual facts; UniHR supplements global structural information through inter-fact MP, improving triple prediction MRR on FBH from 0.922 to 0.946.
- **vs HAHE**: HAHE achieves hierarchical modeling for HKG but is not generalizable by design; UniHR surpasses HAHE's H@10 of 0.610 with 0.619 on WikiPeople.
- **vs HypeTKG**: For composite hyper-relational temporal KGs, UniHR outperforms the specialized model on both WIKI-hy and YAGO-hy without complex module stacking.

The unified representation paradigm is transferable to other heterogeneous data modeling tasks such as multimodal KGs. The hierarchical message passing design (intra-fact + inter-fact) offers inspiration for graph learning tasks that require simultaneous modeling of local and global information. The joint training experiments provide empirical support for the intuition that data diversity benefits representation learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First unified framework to handle three types of beyond-triple KGs; HiDR design is concise and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 9 datasets, 5 KG types, ablation studies, joint training, and efficiency analysis provide comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with rigorous definitions of HiDR conversion rules.
- **Value**: ⭐⭐⭐⭐ — Fills a gap in unified KG representation learning; joint training experiments demonstrate practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](../../NeurIPS2025/graph_learning/tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)
- [\[NeurIPS 2025\] OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](../../NeurIPS2025/graph_learning/ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)
- [\[AAAI 2026\] Hyperbolic Continuous Structural Entropy for Hierarchical Clustering](hyperbolic_continuous_structural_entropy_for_hierarchical_clustering.md)
- [\[AAAI 2026\] Human Cognition Inspired RAG with Knowledge Graph for Complex Problem Solving](human_cognition_inspired_rag_with_knowledge_graph_for_complex_problem_solving.md)

</div>

<!-- RELATED:END -->
