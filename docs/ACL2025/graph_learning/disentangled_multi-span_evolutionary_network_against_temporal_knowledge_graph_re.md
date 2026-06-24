---
title: >-
  [Paper Note] Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning
description: >-
  [ACL 2025 Findings][Graph Learning][Temporal Knowledge Graph] The authors propose DiMNet, which separates the active/stable features of node semantics through a multi-span evolution strategy and a cross-time disentanglement mechanism. This significantly improves extrapolation reasoning performance on Temporal Knowledge Graphs (TKGs), achieving SOTA results across four benchmark datasets.
tags:
  - "ACL 2025 Findings"
  - "Graph Learning"
  - "Temporal Knowledge Graph"
  - "Extrapolation Reasoning"
  - "Graph Neural Network"
  - "Disentangled Representation"
  - "Multi-span Evolution"
date: 2026-05-08
content_hash: 1177cd9522bd6671
---

# Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning

**Conference**: ACL 2025 Findings  
**arXiv**: [2505.14020](https://arxiv.org/abs/2505.14020)  
**Code**: Not provided  
**Area**: Graph Learning  
**Keywords**: Temporal Knowledge Graph, Extrapolation Reasoning, Graph Neural Network, Disentangled Representation, Multi-span Evolution

## TL;DR

The authors propose DiMNet, which separates the active/stable features of node semantics through a multi-span evolution strategy and a cross-time disentanglement mechanism. This significantly improves extrapolation reasoning performance on Temporal Knowledge Graphs (TKGs), achieving SOTA results across four benchmark datasets.

## Background & Motivation

Temporal Knowledge Graphs (TKGs) represent timestamped facts as quadruplets $(s, r, o, t)$. Their reasoning tasks aim to predict missing future facts based on historical subgraph sequences. Existing evolution-based methods (e.g., RE-GCN, TiPNN) typically model historical subgraph sequences step-by-step, but suffer from the following limitations:

**Ignoring multi-span semantic evolution**: Existing methods only capture local structural semantics at each time step, failing to perceive intermediate updates of neighbor features across different time spans, which prevents learning fine-grained multi-span evolutionary patterns.

**No disentanglement between active and stable features**: The semantics of nodes during the evolutionary process contain both rapidly changing "active" components and relatively steady "stable" components. Existing methods do not distinguish between them, limiting precise modeling of semantic changes.

**Topology uncertainty during the inference phase**: When predicting future facts, the graph topology at future time steps is unknown, making the direct application of historically evolved representations potentially inaccurate.

## Method

### Overall Architecture

DiMNet is composed of three core components: the **Multi-span Evolution Module**, the **Cross-time Disentanglement Module**, and the **Sampling Virtual Subgraph Decoder**.

### 1. Multi-span Evolution

Given the historical subgraph sequence $\{G_{t-m}, ..., G_{t-1}\}$, DiMNet performs $\omega$ layers of GNN message passing at each time step $t_i$ to capture local structural semantics. The key innovation lies in: during the $l$-th layer of message aggregation at the current time step $t_i$, the model utilizes not only the neighbor information of the current subgraph but also perceives the neighbor features of the $l$-th layer in the previous time step $t_{i-1}$, achieving cross-step multi-span semantic awareness:

$$h_v^{(l)} = \text{Agg}\left(h_v^{(l-1)},\ \{h_u^{(l-1)}\}_{u \in \mathcal{N}(v, t_i)},\ \{h_u^{(l)}|_{t_{i-1}}\}\right)$$

This enables the model to capture semantic updates at different levels and spans during the evolution.

### 2. Cross-time Disentanglement

Between adjacent time steps, DiMNet adaptively disentangles node representations into an **active factor** $\mathcal{A}$ and a **stable factor** $\mathcal{B}$:

- **Active factor**: Captures semantic components that change significantly between adjacent time steps, extracted from the representation difference of two time steps via an attention mechanism.
- **Stable factor**: Captures semantic components that remain relatively unchanged during temporal evolution.

A GRU is used to iteratively update the active factor, modeling the continuous temporal variation patterns of active semantics. The disentanglement loss $\mathcal{L}_{dis}$ encourages orthogonal separation by minimizing the similarity between the active and stable factors:

$$\mathcal{L}_{dis} = \sum \text{sim}(\mathcal{A}, \mathcal{B})$$

### 3. Sampling Virtual Subgraph Decoder

To address the topological uncertainty of the future during inference, DiMNet designs a virtual subgraph $G_{\text{INF}}$:

1. Perform initial scoring of all candidate triples based on the evolved entity representations.
2. Sample the Top-$k$ highest-scoring triples to construct the virtual subgraph.
3. Perform GNN message passing again on the virtual subgraph to enhance representations using the predicted topology.
4. Obtain the final prediction scores based on the enhanced representations.

### Loss & Training

The total loss is a weighted combination of the cross-entropy loss and the disentanglement loss:

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{dis}$$

## Key Experimental Results

### Main Results

Compared with over 15 baseline methods (including traditional KG, interpolation TKG, and extrapolation TKG methods) on four datasets (ICEWS14, ICEWS05-15, ICEWS18, and GDELT), using time-aware filtered MRR and Hits@{1, 3, 10} metrics:

| Model | ICEWS14 MRR | ICEWS05-15 MRR | ICEWS18 MRR | GDELT MRR |
|------|------------|----------------|------------|----------|
| RE-GCN | 41.78 | 48.03 | 30.58 | 19.64 |
| CEN | 42.17 | 46.84 | 30.84 | 20.18 |
| TiPNN | 41.79 | 45.35 | 32.17 | 21.17 |
| DaeMon | 40.68 | 44.50 | 31.85 | 20.73 |
| **DiMNet** | **45.72** | **58.93** | **34.13** | **21.93** |

DiMNet outperforms the current best methods in MRR on the four datasets by **8.4%, 22.7%, 6.1%, and 3.6%** respectively, achieving a substantial lead, especially on ICEWS05-15.

### Ablation Study

| Variant | ICEWS14 MRR | ICEWS05-15 MRR | ICEWS18 MRR | GDELT MRR |
|------|------------|----------------|------------|----------|
| DiMNet (Full) | 45.72 | 58.93 | 34.13 | 21.93 |
| w/o Multi-span | 40.75 | 51.17 | 30.74 | 20.99 |
| w/o Disentangle | 34.34 | 53.22 | 33.60 | 20.51 |
| w/o Both | 36.09 | 50.36 | 30.88 | 20.71 |
| w/o $G_{\text{INF}}$ | 36.10 | 45.45 | 30.02 | 19.81 |

- Performance degrades significantly without the multi-span strategy, proving that cross-step neighbor awareness is crucial for fine-grained evolution modeling.
- Removing the disentanglement component leads to the largest drop on ICEWS14 (MRR decreases from 45.72 to 34.34), indicating that active/stable feature separation is a core contribution.
- Eliminating the virtual subgraph decoder also leads to substantial performance loss, validating the necessity of topological enhancement in the inference phase.

### Parameter Sensitivity Analysis

- **Historical Sequence Length $m$**: Tested on different $m$ values (2–15) across ICEWS14 and ICEWS18; the performance remains stable (especially on ICEWS18), validating the robustness of the model.
- **GNN Layers $\omega$**: Both datasets reach optimal performance at $\omega=3$. Performance stabilizes as the number of layers increases further.
- **Sample Size $k$**: $k=50$ is optimal on ICEWS14; excessively large $k$ (e.g., 80) introduces noise, thereby degrading performance.

## Highlights & Insights

- **Novel Multi-span Evolution Mechanism**: Introduces cross-time step neighbor feature awareness in GNN message passing, breaking the paradigm of step-by-step independent modeling in prior work, and capturing richer intermediate evolutionary features.
- **Well-justified Disentanglement Design**: Decomposes node semantics into active and stable factors with clear intuition, and the ablation study thoroughly validates its effectiveness.
- **Virtual Subgraph Decoder**: Alleviates uncertainty in the inference phase by constructing a predicted topology, representing a practical engineering design.
- **Comprehensive Experiments**: Covers four datasets, 15+ baselines, ablation studies, and complete parameter sensitivity analysis.

## Limitations & Future Work

- **Unexamined Computational Complexity**: Multi-span evolution requires maintaining intermediate features across time steps, and the virtual subgraph decoder requires scoring all candidates, sampling, and re-inferring, which may import high training and inference overhead.
- **Evaluation Limited to Event-based TKGs**: Both ICEWS and GDELT are political event datasets; generalization has not been verified on other types of TKGs (e.g., knowledge-based TKGs like Wikidata or YAGO).
- **Limited Interpretability of Disentanglement**: Although the concepts of active and stable factors are proposed, there is a lack of visualization analysis or semantic explanations for the disentangled results.
- **Code Not Open-sourced**: Replication and verification are not possible.

## Related Work & Insights

- **Traditional KG Embeddings**: DistMult, ComplEx, ConvE, RotatE — do not consider temporal dimensions.
- **Interpolated TKG Reasoning**: TTransE, TA-DistMult, DE-SimplE, TNTComplEx — complete missing facts within known time spans.
- **Extrapolated TKG Reasoning**: RE-NET, RE-GCN, CEN, TiPNN, DaeMon — predict future facts based on historical sequences, but lack multi-span awareness and feature disentanglement.
- **Graph Disentangled Learning**: DisenGCN, IPGDN, etc. — disentangle representations on static graphs, not extended to temporal graphs.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The combination of multi-span evolution and cross-time disentanglement is a meaningful improvement over the traditional TKG reasoning paradigm.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Comprehensive baseline coverage, rational ablation design, and thorough parameter analysis.
- **Writing Quality** ⭐⭐⭐⭐: Clear structure and standardized experimental presentation.
- **Impact** ⭐⭐⭐: TKG reasoning is an important sub-field of KG, but has a processed and relatively limited audience; Finding-tier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning](a_generative_adaptive_replay_continual_learning_model_for_temporal_knowledge_gra.md)
- [\[ICLR 2026\] Inductive Reasoning for Temporal Knowledge Graphs with Emerging Entities](../../ICLR2026/graph_learning/inductive_reasoning_for_temporal_knowledge_graphs_with_emerging_entities.md)
- [\[ICLR 2026\] Temporal Graph Thumbnail: Robust Representation Learning with Global Evolutionary Skeleton](../../ICLR2026/graph_learning/temporal_graph_thumbnail_robust_representation_learning_with_global_evolutionary.md)
- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)

</div>

<!-- RELATED:END -->
