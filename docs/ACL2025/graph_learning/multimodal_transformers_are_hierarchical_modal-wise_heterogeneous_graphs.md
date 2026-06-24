---
title: >-
  [Paper Note] Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs
description: >-
  [ACL 2025][Graph Learning][Multimodal Sentiment Analysis] Proving from a graph theory perspective that Multimodal Transformers (MulTs) are essentially Hierarchical Modal-wise Heterogeneous Graphs (HMHGs), this paper proposes the GsiT model. By employing an Interlaced Mask mechanism, GsiT achieves All-Modal-In-One fusion with only 1/3 of the parameters while significantly outperforming traditional MulTs.
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Multimodal Sentiment Analysis"
  - "Graph Structures"
  - "Transformer"
  - "Weight Sharing"
  - "Attention Mask"
date: 2026-05-08
content_hash: 203eac9680e6b938
---

# Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs

**Conference**: ACL 2025  
**arXiv**: [2505.01068](https://arxiv.org/abs/2505.01068)  
**Code**: [https://github.com/thuiar/MMSA](https://github.com/thuiar/MMSA)  
**Area**: Graph Learning  
**Keywords**: Multimodal Sentiment Analysis, Graph Structures, Transformer, Weight Sharing, Attention Mask  


## TL;DR

Proving from a graph theory perspective that Multimodal Transformers (MulTs) are essentially Hierarchical Modal-wise Heterogeneous Graphs (HMHGs), this paper proposes the GsiT model. By employing an Interlaced Mask mechanism, GsiT achieves All-Modal-In-One fusion with only 1/3 of the parameters while significantly outperforming traditional MulTs.

## Background & Motivation

Multimodal Sentiment Analysis (MSA) requires fusing heterogeneous modalities such as text, video, and audio to identify emotions. The Multimodal Transformer (MulT) and its subsequent variants (MulTs) are the current mainstream paradigm, achieving fusion through cross-modal attention (CMA) and multi-head self-attention (MHSA). However, MulTs suffer from severe efficiency issues:

- **Parameter Redundancy**: MulTs decomposes tri-modal inputs into pairwise combinations for separate processing, using independent weights for each pair (6 CMAs + 3 MHSAs + 9 MLPs), resulting in a massive parameter size.
- **End-to-End System Constraints**: MSA is an end-to-end discriminative task, requiring models to balance performance and efficiency in practical deployment.
- **Lack of Theoretical Analysis**: The structural redundancy of MulTs lacks formal theoretical analysis and optimization guidance.

The authors propose a key insight: for MSA systems, the resource savings achieved by designing low-cost, high-performance models are, in some aspects, more meaningful than the accuracy gains brought solely by scaling up models.

## Method

### 1. Theoretical Foundation: MulTs as Hierarchical Modal-wise Heterogeneous Graphs (HMHG)

The paper first establishes the equivalence relationship between attention mechanisms and graph structures:

- **Lemma 1**: Multi-head cross-modal attention (CMA) is equivalent to the aggregation of a unidirectional complete bipartite graph of bi-modal combinations; multi-head self-attention (MHSA) is equivalent to the aggregation of a single-modality directed complete graph.
- **Theorem 1**: MulTs are Hierarchical Modal-wise Heterogeneous Graphs (HMHGs).

Specifically, MulTs can be decomposed into:
- Bottom-layer subgraphs $G(i,j)$ and $G(i,p)$: Cross-modal bipartite graphs constructed by CMA.
- Intermediate concatenation layer: Concatenating cross-modal fusion vectors.
- Top-layer subgraph $G(i,i)$: Intra-modal directed complete graphs constructed by MHSA.

Viewed from a single dominant modality, it forms a tree; the combination of trees from multiple dominant modalities forms a forest structure.

### 2. Graph-Structured Interlaced-Masked Multimodal Transformer (GsiT)

Based on the parameter redundancy identified by the HMHG theorem, GsiT compresses the traditional forest structure into a single tree structure with shared weights. The core mechanism is the **Interlaced Mask (IM)**, which consists of two parts:

**Interlaced-Multimodal-Fusion Mask (IFM)**:
- forward mask: Allows attention in paths v->t, a->v, and t->a.
- backward mask: Allows attention in paths a->t, t->v, and v->a.
- Two counter-directional unidirectional loops form complete cross-modal fusion, avoiding information chaos.

**Interlaced-Intra-Enhancement Mask (IEM)**:
- Only allows intra-modal self-attention (t->t, v->v, a->a), masking cross-modal positions.

Graph structure constraints are implemented by adding negative infinity masks to the attention score matrix, making the shared-weight MHSA equivalent to independent CMA operations.

### 3. Decomposition Triton Kernel

The spatial complexity of GsiT's attention map after concatenating multimodal sequences is $O((T_t+T_v+T_a)^2)$, which is higher than the $O(T_i*T_j)$ of MulTs. To address this, a Decomposition Triton kernel was implemented: after the shared QKV projection, it decomposes the sequences according to their original lengths and independently performs attention on modality pairs specified by the IM. This keeps the runtime spatial complexity on par with MulTs while reducing static parameters to 1/3.

### 4. All-Modal-In-One Fusion

The final function system is compressed from the 6 CMAs + 3 MHSAs + 9 MLPs of MulTs to 3 MHSAs + 3 MLPs, reducing the parameter size to 1/3 of traditional methods and achieving fusion of all modalities within shared weights.

## Key Experimental Results

### Table 1: Main Results on CMU-MOSI and CMU-MOSEI

| Model | MOSI Acc-2 (NN/NP) | MOSI Acc-7 | MOSEI Acc-2 (NN/NP) | Params (M) | FLOPS (G) |
|------|-------------------|------------|---------------------|------------|-----------|
| MulT | 79.6 / 81.4 | 36.2 | 78.1 / 83.7 | 5.251 | 26.294 |
| **GsiT** | **83.7 / 85.8** | **47.4** | **84.5 / 85.6** | **1.695** | **26.224** |
| Delta | +4.1 / +4.4 | +11.2 | +6.4 / +1.9 | **-67.7%** | -0.3% |
| TETFN | 82.4 / 84.0 | 46.1 | 81.9 / 84.3 | 5.921 | 27.558 |
| TETFN w/ HMHG | 83.2 / 85.2 | 47.1 | 84.6 / 84.8 | 2.365 | 27.488 |
| ALMT | 82.1 / 83.3 | 45.5 | 81.4 / 83.5 | 2.604 | 19.876 |
| ALMT w/ HMHG | 83.2 / 84.6 | 47.1 | 82.9 / 86.4 | 2.506 | 19.876 |

Compared to MulT, GsiT reduces parameters by 67.7%, improves Acc-2 by 4+%, and increases Acc-7 by 11.2%.

### Table 2: Ablation Study on CMU-MOSI

| Structure | Acc-2 (NN/NP) | Acc-7 | MAE |
|------|---------------|-------|-----|
| **Original (Counter-directional loop)** | **83.7 / 85.8** | **47.4** | **0.713** |
| Structure-1 (Non-loop) | 83.5 / 85.5 | 46.5 | 0.721 |
| Structure-2 (Non-loop) | 83.2 / 84.9 | 43.8 | 0.729 |
| Structure-3 (Non-loop) | 83.4 / 85.2 | 45.5 | 0.726 |
| Self-Only (Information chaos) | 82.5 / 84.2 | 45.5 | 0.734 |

The counter-directional loop structure performs best; Self-Only, which violates the HMHG constraints, performs the worst, validating the existence of the information chaos issue.

## Highlights & Insights

- **Outstanding Theoretical Contribution**: Rigorously proves the equivalence relationship between MulTs and HMHGs, elevating the structural analysis of multimodal fusion to the height of graph theory.
- **High Parameter Efficiency**: Achieves or exceeds the performance of original MulTs with only 1/3 of the parameters, with an impressive 11.2% increase in Acc-7.
- **Strong Versatility**: The HMHG concept can be integrated as a plug-and-play component into various baseline models such as Self-MM, TETFN, and ALMT, yielding improvements.
- **Information Chaos Theory**: Explains why mask designs must follow specific constraints from the perspective of softmax probability distribution, providing clear theoretical guidance.
- **Deployment-Friendly Engineering**: The Decomposition Triton kernel ensures no additional computational overhead during runtime.

## Limitations & Future Work

- **Limited Task Scope**: Validated only on multimodal sentiment analysis, without scaling to broader multimodal tasks like video QA or multimodal retrieval.
- **No Consideration for Missing Modalities**: Does not discuss model robustness when certain modalities are missing.
- **Lack of Integration with Representation Learning**: No representation learning methods (e.g., contrastive learning) are introduced in the first-layer fusion encoder pairs, which is a direction for future exploration.
- **Limited to Tri-modality**: The current framework is designed for text/video/audio tri-modal inputs; the design of IM needs to be reconsidered if scaled to more modalities.
- **Small Dataset Scale**: Classic MSA datasets like CMU-MOSI/MOSEI are limited in scale and have not been validated on large-scale datasets.

## Related Work & Insights

- **MulT (Tsai et al., 2019)**: The direct theoretical foundation of GsiT. GsiT outperforms all of its metrics with only 1/3 of the parameters.
- **Self-MM (Yu et al., 2021)**: A self-supervised learning framework. Integrating GsiT improves performance but compromises efficiency (due to its originally extremely simplified fusion layer).
- **TETFN (Wang et al., 2023)**: A pure MulTs-based model. After embedding HMHG, parameters are reduced by 60.1%, with improvements across most metrics.
- **ALMT (Zhang et al., 2023)**: A MulTs-like architecture. After embedding HMHG, parameters are reduced by 3.8%, and Acc-2 on MOSEI is boosted by 2.9%.
- **GAT (Velickovic et al., 2018)**: This study proves the theoretical equivalence of CMA/MHSA and GAT, which serves as a bridge for constructing HMHG.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ --- Re-evaluating the MulTs architecture from a graph theory perspective with highly original theoretical proofs.
- Experimental Thoroughness: ⭐⭐⭐⭐ --- Relatively comprehensive, covering four datasets, multi-baseline integration, ablation studies, weight distribution analysis, and convergence analysis.
- Writing Quality: ⭐⭐⭐⭐ --- Rigorous theoretical derivation, though the large volume of LaTeX equations increases reading difficulty.
- Value: ⭐⭐⭐⭐ --- Provides a new theoretical framework and practical efficiency optimization solutions for multimodal fusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](../../NeurIPS2025/graph_learning/unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)
- [\[ACL 2025\] M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations](m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](../../ICLR2026/graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[ACL 2025\] Fast-and-Frugal Text-Graph Transformers are Effective Link Predictors](fast-and-frugal_text-graph_transformers_are_effective_link_predictors.md)
- [\[CVPR 2026\] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](../../CVPR2026/graph_learning/graph2eval_automatic_multimodal_task_generation_for_agents_via_knowledge_graphs.md)

</div>

<!-- RELATED:END -->
