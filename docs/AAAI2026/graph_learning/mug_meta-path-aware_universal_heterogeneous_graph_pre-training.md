---
title: >-
  [Paper Note] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training
description: >-
  [AAAI 2026][Graph Learning][Heterogeneous graph] MUG is the first universal heterogeneous graph pre-training method that requires no LLM. It unifies heterogeneous node/relation types via contextual structural encoding, aligns representation spaces across graphs with a dimension-aware encoder, and achieves transferable encoding and aggregation through a shared GNN encoder over meta-path views combined with global scatter regularization. MUG substantially outperforms existing methods on cross-domain and few-shot node classification.
tags:
  - AAAI 2026
  - Graph Learning
  - Heterogeneous graph
  - universal graph pre-training
  - meta-path
  - graph foundation model
  - cross-domain transfer
  - self-supervised learning
  - masked autoencoding
date: 2026-05-08
content_hash: 0040e5ff6480c4ab
---

# MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training

**Conference**: AAAI 2026
**arXiv**: [2602.22645](https://arxiv.org/abs/2602.22645)
**Code**: [github.com/slz1024/MUG](https://github.com/slz1024/MUG)
**Area**: Graph Learning / Heterogeneous Graph Pre-Training
**Keywords**: Heterogeneous graph, universal graph pre-training, meta-path, graph foundation model, cross-domain transfer, self-supervised learning, masked autoencoding

## TL;DR

MUG is the first universal heterogeneous graph pre-training method that requires no LLM. It unifies heterogeneous node/relation types via contextual structural encoding, aligns representation spaces across graphs with a dimension-aware encoder, and achieves transferable encoding and aggregation through a shared GNN encoder over meta-path views combined with global scatter regularization. MUG substantially outperforms existing methods on cross-domain and few-shot node classification.

## Background & Motivation

**Universal Graph Pre-training (UGP)** aims to train transferable graph encoders that generalize to unseen downstream tasks and datasets without retraining, and constitutes a core paradigm for graph foundation models.

However, existing UGP methods (e.g., FUG, SAMGPT) focus almost exclusively on **homogeneous graphs** (single node type, simple fixed relations), whereas real-world graphs are typically **heterogeneous**—containing multiple node and edge types, for example:

- **ACM**: paper–author–subject, with relations paper-author and paper-subject
- **Freebase**: movie–actor–director–writer, with relations movie-actor, movie-director, etc.

Extending UGP to heterogeneous graphs presents two core challenges:

**Input unification**: Different heterogeneous graphs have distinct node types, relation types, and attribute dimensions, making it infeasible to construct a unified representation space directly. Existing UGP methods assume a fixed set of entity and relation types and thus fail outright on heterogeneous graphs.

**Transfer of learned information**: Conventional heterogeneous graph methods (e.g., HAN, HeCo, HGMAE) employ type-specific encoders and meta-path-specific aggregation weights that are tightly coupled to a particular dataset and cannot generalize to new heterogeneous graphs.

The authors note that **no existing method specifically addresses universal pre-training for heterogeneous graphs**, representing a significant gap in graph foundation model research.

## Method

### Overall Architecture

MUG consists of two core modules:

1. **Heterogeneous Input Unification Module**: Encodes diverse node/relation types into a unified representation and aligns the representation spaces of different graphs via a dimension-aware encoder.
2. **Heterogeneous Information Transfer Module**: Performs universal encoding over meta-path views with a shared GNN encoder, and introduces global scatter regularization for universal aggregation.

The overall pipeline: heterogeneous graph → contextual structural encoding → concatenation with raw attributes → dimension-aware alignment → shared GNN encoding per meta-path view → joint optimization with three losses.

### Key Design 1: Contextual Structural Encoding (CSE)

**Goal**: Encode type and relation information of heterogeneous nodes into a unified embedding without type-specific transformation matrices.

**Approach**:

- For each meta-path $\mathcal{P}_\ell$, perform **meta-path-guided random walks** to sample structural context sequences for each node.
- Jointly train a unified structural embedding $\mathbf{z}_v^{\text{struct}}$ across all meta-paths using **skip-gram with negative sampling** (analogous to Metapath2vec).
- Optimization objective: encourage nodes that co-occur within the same meta-path context to be closer in the embedding space.
- After training, **freeze parameters** and concatenate the structural embedding with the raw node attributes: $\tilde{\mathbf{x}}_v = \text{concat}(\mathbf{x}_v, \mathbf{z}_v^{\text{struct}})$

**Advantage**: Implicitly encodes type semantics through structural context, eliminating the need for type-specific parameters.

### Key Design 2: Dimension-aware Alignment

**Goal**: The unified representations $\tilde{\mathbf{x}}_v$ from different heterogeneous graphs vary in dimensionality and semantic space, requiring alignment into a shared space.

**Approach**:

- Treat each attribute dimension as an independent semantic unit; randomly sample $n_s$ nodes and extract the $i$-th column vector $\tilde{\mathbf{X}}_{:,i}^s$.
- Encode it via an MLP into a semantic basis vector $\mathbf{s}_i \in \mathbb{R}^k$.
- The unified input for each node: $\mathbf{x}_v^{\text{unify}} = \sum_{i=1}^{d} \tilde{\mathbf{x}}_v[i] \cdot \mathbf{s}_i$
- A **mean-centering loss** $\mathcal{L}_{\text{align}}$ is applied to prevent local bias in the basis vectors.

### Key Design 3: Universal Encoding and Aggregation

**Universal Encoding**:

Through empirical analysis, the authors find that **the average homophily ratio across meta-path views in heterogeneous graphs is comparable to that of homogeneous graphs** (as shown in Figure 1), justifying the use of a **single shared GNN encoder** to process different meta-path views.

- Apply random edge masking to the adjacency matrix $\mathbf{A}^\phi$ of each meta-path.
- Encode the masked graph with a shared encoder $\text{GNN}_{\text{shared}}$: $\mathbf{Z}^\phi = \text{GNN}_{\text{shared}}(\tilde{\mathbf{A}}^\phi, \mathbf{X}^{\text{unify}})$
- Reconstruct the adjacency matrix via a GNN decoder, trained with a **scaled cosine loss** $\mathcal{L}^\phi$.

**Universal Aggregation**:

Conventional methods learn meta-path weights $\beta^\phi$ via semantic-level attention vectors that are coupled to the training dataset. MUG instead introduces **global scatter regularization**:

$$\mathcal{L}_{\text{scatter}} = -\frac{1}{|\mathcal{V}|}\sum_{v \in \mathcal{V}} \|\mathbf{z}_v - \bar{\mathbf{z}}\|_2^2$$

This loss encourages node embeddings to disperse away from the global mean, enhancing discriminability, reducing dependency on specific aggregation functions, and thereby improving cross-domain transferability.

### Loss & Training

The total loss is a weighted combination of three terms:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{align}} + \lambda_2 \sum_{\phi \in \Phi} \beta^\phi \mathcal{L}^\phi + \lambda_3 \mathcal{L}_{\text{scatter}}$$

- $\mathcal{L}_{\text{align}}$: dimension alignment loss, preventing basis vector bias.
- $\mathcal{L}^\phi$: meta-path masked reconstruction loss, capturing structural patterns.
- $\mathcal{L}_{\text{scatter}}$: global scatter regularization, enhancing cross-domain generalization.

After pre-training, all model parameters are frozen and transferred directly to unseen datasets for downstream tasks (zero parameter updates).

## Key Experimental Results

### Cross-domain Node Classification (trained on one dataset, evaluated on all four)

| Train Set | Method | ACM Ma-F1 | ACM Mi-F1 | DBLP Ma-F1 | DBLP Mi-F1 | AMiner Ma-F1 | AMiner Mi-F1 | Freebase Ma-F1 | Freebase Mi-F1 |
|-----------|--------|-----------|-----------|------------|------------|--------------|--------------|----------------|----------------|
| ACM | HeCo | 80.22 | 79.71 | 76.76 | 77.97 | 24.48 | 51.18 | 31.22 | 40.67 |
| ACM | HGMAE | 84.22 | 84.01 | 87.17 | 88.23 | 29.08 | 41.91 | 32.59 | 42.95 |
| ACM | HERO | 84.37 | 84.12 | 84.60 | 85.80 | 44.08 | 50.14 | 33.69 | 43.32 |
| ACM | **MUG** | **85.52** | **84.90** | **91.69** | **92.38** | **76.35** | **87.02** | **46.05** | **49.78** |
| Freebase | HeCo | 77.03 | 76.30 | 82.37 | 83.26 | 29.82 | 34.51 | 42.34 | 47.92 |
| Freebase | **MUG** | **85.21** | **85.22** | **91.79** | **92.24** | **78.10** | **87.94** | **52.33** | **57.50** |

MUG achieves state-of-the-art results across all training–evaluation combinations. The advantage is particularly pronounced on AMiner (Ma-F1 rises from ~44 to ~76), where one-hot attributes suffer severe information loss under SVD dimensionality reduction, whereas MUG's contextual structural encoding effectively preserves semantic information.

### Few-shot Cross-domain Node Classification (trained on ACM, evaluated at 1/3/5-shot)

| Shot | Method | ACM Ma-F1 | DBLP Ma-F1 | AMiner Ma-F1 | Freebase Ma-F1 |
|------|--------|-----------|------------|--------------|----------------|
| 1-shot | HGMAE | 73.17 | 61.46 | 20.65 | 30.65 |
| 1-shot | HERO | 51.39 | 40.49 | 44.18 | 32.20 |
| 1-shot | **MUG** | **79.49** | **84.24** | **49.12** | **33.24** |
| 3-shot | HGMAE | 79.42 | 71.39 | 23.73 | 32.47 |
| 3-shot | **MUG** | **84.39** | **90.56** | **66.80** | **35.01** |
| 5-shot | HGMAE | 81.68 | 79.03 | 24.84 | 33.45 |
| 5-shot | **MUG** | **83.83** | **90.76** | **68.30** | **39.36** |

At 5-shot, MUG's performance closely approaches that of full-label cross-domain classification, demonstrating highly effective transfer of pre-trained representations. On DBLP, 5-shot achieves 90.76 Ma-F1, within less than 1 point of the full-label result of 91.69.

## Ablation Study

Ablation experiments trained on Freebase and evaluated across four datasets show:

- **Removing CSE** (contextual structural encoding) causes the largest performance drop, especially on the cross-domain dataset AMiner, confirming the critical role of CSE in capturing universal heterogeneous semantics.
- **Removing $\mathcal{L}_{\text{align}}$** and **removing $\mathcal{L}_{\text{scatter}}$** also lead to notable degradation, demonstrating that both auxiliary losses are indispensable for mitigating domain-specific bias and improving cross-domain generalization.
- The full MUG model achieves the best results on all datasets.

## Highlights & Insights

1. **Pioneering contribution**: The first universal heterogeneous graph pre-training method without LLM dependency, filling a critical gap in graph foundation model research for heterogeneous graphs.
2. **Meta-path homophily finding**: Empirical evidence demonstrates that the homophily ratio of meta-path views in heterogeneous graphs is comparable to that of homogeneous graphs, providing both theoretical and empirical justification for the shared encoder design.
3. **Elegant input unification**: Replacing type-specific parameters with Metapath2vec-style structural encoding, combined with dimension-aware alignment, elegantly resolves the schema inconsistency problem across heterogeneous graphs.
4. **Zero-parameter-update transfer**: The encoder is fully frozen after pre-training and transferred directly to unseen datasets, representing a genuinely universal pre-training paradigm.
5. **Dominant advantage on AMiner** (Ma-F1 from ~44 to ~76) exposes the critical weakness of existing SVD-based unification approaches when applied to sparse one-hot attributes.

## Limitations & Future Work

1. **Meta-paths must be predefined**: The method still relies on manually defined meta-path sets; specifying meta-paths for new graphs requires domain knowledge, limiting full automation.
2. **Evaluation limited to node classification**: Downstream tasks include only node classification; transfer effectiveness on link prediction, graph classification, and other tasks remains unverified.
3. **Limited dataset scale**: All four evaluation datasets are small-to-medium academic benchmarks; the method has not been validated on industrial-scale large heterogeneous graphs.
4. **CSE pre-training overhead**: Contextual structural encoding requires skip-gram pre-training followed by parameter freezing, adding complexity through two-stage training.
5. **Insufficient comparison with LLM-augmented methods**: HiGPT is mentioned but not included in experiments, making it difficult to comprehensively assess the cost-effectiveness of the LLM-free approach.

## Related Work & Insights

- **Heterogeneous graph representation learning**: HAN (hierarchical attention), MAGNN (meta-path aggregation), HGT (Transformer for heterogeneous graphs), HeCo (contrastive learning with dual views), HGMAE (masked autoencoder), HGCL (attribute+topology view contrast).
- **Universal graph pre-training**: GCC (structural encoding transfer), FUG (attribute semantic basis learning), SAMGPT (structural transfer), GraphMAE (masked self-supervision).
- **Heterogeneous graphs + LLM**: HiGPT (LLM-assisted textualized attribute cross-domain transfer, but limited to graphs without non-textual attributes).

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | First LLM-free universal heterogeneous graph pre-training; pioneering problem formulation |
| Technical Depth | 4 | Three well-motivated modules; meta-path homophily analysis provides solid empirical grounding |
| Experimental Thoroughness | 3 | Comprehensive coverage with 4 datasets, cross-domain, and few-shot settings, but lacks large-scale and multi-task evaluation |
| Writing Quality | 4 | Clear problem motivation, rigorous method derivation, high-quality figures and tables |
| Value | 3 | Opens an important research direction, but meta-path predefinition and limited evaluation scenarios constrain practical deployment |
| Overall | 3.6 | An important first step toward heterogeneous graph foundation models; the direction is sound but considerable room for extension remains |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)
- [\[AAAI 2026\] Spiking Heterogeneous Graph Attention Networks](spiking_heterogeneous_graph_attention_networks.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)
- [\[AAAI 2026\] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)

</div>

<!-- RELATED:END -->
