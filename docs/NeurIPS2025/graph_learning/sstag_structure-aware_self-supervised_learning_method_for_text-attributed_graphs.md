---
title: >-
  [Paper Note] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs
description: >-
  [NeurIPS 2025][Graph Learning][Text-attributed graphs] This paper proposes SSTAG, which jointly distills complementary knowledge from LLMs and GNNs into a structure-aware MLP via dual knowledge distillation, and incorporates a memory bank mechanism to store prototype representations, enabling efficient and scalable cross-domain self-supervised pre-training on text-attributed graphs.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Text-attributed graphs
  - self-supervised learning
  - knowledge distillation
  - cross-domain transfer
  - graph foundation models
date: 2026-05-08
content_hash: bfae15b483c1f1c8
---

# SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs

**Conference**: NeurIPS 2025
**arXiv**: [2510.01248](https://arxiv.org/abs/2510.01248)
**Code**: N/A
**Area**: Graph Learning / Self-Supervised Learning
**Keywords**: Text-attributed graphs, self-supervised learning, knowledge distillation, cross-domain transfer, graph foundation models

## TL;DR

This paper proposes SSTAG, which jointly distills complementary knowledge from LLMs and GNNs into a structure-aware MLP via dual knowledge distillation, and incorporates a memory bank mechanism to store prototype representations, enabling efficient and scalable cross-domain self-supervised pre-training on text-attributed graphs.

## Background & Motivation

Large-scale pre-trained models in NLP and CV have demonstrated remarkable cross-domain generalization capabilities, yet graph learning remains largely confined to the **single-graph training** paradigm. This paradigm suffers from two major limitations: (1) models are restricted to single or narrow tasks, lacking the ability to transfer knowledge across graphs; and (2) model performance heavily relies on abundant labeled data, whereas high-quality annotations are costly and become a bottleneck in low-resource settings.

Building graph foundation models poses unique challenges: unlike the unified vocabulary space in NLP or the consistent pixel space in CV, graph data exhibits **domain heterogeneity** (different graph domains have distinct feature spaces and label taxonomies) and **structural diversity** (e.g., citation networks vs. knowledge graphs have fundamentally different structures).

The core insight of SSTAG is to exploit **text as a unified representational medium**, as many real-world graphs are naturally text-attributed graphs (TAGs). LLMs excel at textual understanding but are not adept at topological reasoning, while GNNs excel at structural modeling but lack open-world knowledge. SSTAG distills these complementary capabilities into a lightweight MLP.

## Method

### Overall Architecture

SSTAG consists of three core modules: (1) a Unified Graph Task (UGT) module that unifies node-, edge-, and graph-level tasks via subgraph sampling; (2) a Knowledge Extraction from LLMs (KEL) module that performs masked autoencoding pre-training by combining a language model with a GNN; and (3) a Knowledge Distillation (KD) module that distills knowledge from the LM+GNN teacher into a structure-aware MLP student.

### Key Designs

1. **Unified Graph Task (UGT)**: A Personalized PageRank (PPR)-based subgraph sampling strategy is adopted to construct a contextual subgraph for each target node or edge. For node $v$, the PPR importance score is $\pi_v = \alpha(\mathbf{I} - (1-\alpha)\tilde{\mathbf{A}})^{-1}\mathbf{e}_v$, and the sampling probability of a $k$-hop neighbor $u$ is proportional to $\pi_{vu}/\sum_{w \in \mathcal{N}_k(v)} \pi_{vw}$. This strategy eliminates structural discrepancies across graph domains and offers better scalability to large-scale graphs. Edge-level tasks take the union of the subgraphs of both endpoints, while graph-level tasks use the full graph directly.

2. **LLM Knowledge Extraction and Masked Pre-training**: The teacher model consists of a language model (Sentence Transformer) cascaded with a GCN. A masked language modeling (MLM) objective is employed—tokens in node text are randomly masked, and the model reconstructs them using both textual context and neighborhood information. During encoding, text is passed through the LM to obtain per-token embeddings $\mathbf{E}_v$; the [CLS] token is propagated through the GNN to yield $\mathbf{H}^{\text{cls}}$; the two are concatenated and fused via a linear layer: $\mathbf{H}_v = \text{Linear}(\mathbf{E}_v \oplus (\mathbf{H}_v^{\text{cls}} \otimes \mathbf{1}_{n_v+2}^\top))$; finally, an MLM Head predicts the masked tokens. This drives the model to jointly learn semantic associations and structural patterns.

3. **Knowledge Distillation into a Structure-Aware MLP**: The student model is a lightweight MLP that takes as input the concatenation of a node's [CLS] embedding and its PPR scores: $\tilde{\mathbf{H}}_v^{\text{cls}} = f_{\text{MLP}}([\tilde{\mathbf{E}}_v^{\text{cls}} \| p_v])$. Structural information is injected via PPR scores without explicit message passing, substantially reducing computational overhead. At inference time, only the LM and MLP are required; the GNN component is unnecessary.

4. **Memory Bank**: $L$ learnable prototype anchors $\{\mathbf{a}_j\}_{j=1}^L$ are maintained and interact with input graph representations via an attention mechanism. Each node representation is combined with anchors via softmax attention to produce a reconstructed embedding $\hat{\mathbf{H}}_v = \sum_j s'_{vj} \mathbf{a}_j$. The memory bank preserves invariant knowledge across training instances, encouraging the model to focus on stable and consistent features through alignment, thereby enhancing generalization.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{ST}} + \mathcal{L}_{\text{ME}}$:

- **Masking loss** $\mathcal{L}_{\text{mask}}$: Standard MLM cross-entropy, compelling the model to reconstruct masked tokens using neighboring nodes' textual information.
- **Student-teacher consistency loss** $\mathcal{L}_{\text{ST}}$: Cosine similarity alignment between student and teacher representations.
- **Memory consistency loss** $\mathcal{L}_{\text{ME}}$: L2 distance between node embeddings and memory bank reconstructions, guiding the memory anchors to capture invariant graph features.

After pre-training on ogbn-Paper100M, the model is evaluated on 12 cross-domain target datasets via a linear probing protocol.

## Key Experimental Results

### Main Results

**Node Classification (Cross-domain Transfer, Table 1, pre-trained on ogbn-Paper100M)**:

| Method | Cora | Pubmed | ogbn-Arxiv | WikiCS | Products |
|--------|------|--------|------------|--------|----------|
| GCN (supervised) | 57.62 | 55.18 | 60.85 | 53.24 | 61.95 |
| GraphMAE2 | 73.92 | 68.76 | 69.07 | 58.04 | 74.05 |
| UniGraph | 74.65 | 70.84 | 70.89 | 65.47 | 76.58 |
| **SSTAG** | **75.09** | **72.65** | **72.85** | **68.76** | **78.27** |

**Link Prediction + Graph Classification (Table 1 & 2)**:

| Method | FB15K237 | WN18RR | HIV | BACE |
|--------|----------|--------|-----|------|
| UniGraph | 85.01 | 80.55 | 77.27 | 79.23 |
| Graph-LLM | 82.47 | 73.46 | 76.43 | 80.65 |
| **SSTAG** | **88.64** | **82.42** | **79.52** | **82.06** |

### Ablation Study

**Key Component Ablation (Table 3)**:

| Configuration | WikiCS | ogbn-Arxiv | FB15K237 | MUV |
|---------------|--------|------------|----------|-----|
| SSTAG (full) | **68.76** | **72.85** | **88.64** | **79.86** |
| W/o $\mathcal{L}_{\text{mask}}$ | 67.02 | 70.51 | 85.84 | 76.22 |
| W/o $\mathcal{L}_{\text{ST}}$ | 67.75 | 71.86 | 87.12 | 78.65 |
| W/o $\mathcal{L}_{\text{ME}}$ | 66.53 | 71.14 | 85.96 | 76.43 |
| W/o GNN | 64.34 | 69.53 | 84.32 | 70.57 |

### Key Findings

- Removing the GNN component results in the largest performance degradation (WikiCS −4.42, MUV −9.29), demonstrating the critical importance of structural information.
- Removing the memory consistency loss ($\mathcal{L}_{\text{ME}}$) leads to a notable performance drop, validating the role of the memory bank in cross-domain generalization.
- Fine-tuning on BACE achieves 82.06%, surpassing the supervised GCN baseline by 12.21 percentage points.
- As a purely self-supervised pre-trained model, SSTAG matches or exceeds fully supervised methods on multiple datasets.

## Highlights & Insights

- The **subgraph-based representation for unifying multi-granularity tasks** is both practical and elegant; PPR sampling balances importance weighting with scalability.
- The **dual distillation design** is well-conceived: the teacher model captures the full capabilities of LM+GNN, while the student model injects structural information via PPR scores; at inference time, only LM+MLP is required, substantially reducing deployment cost.
- The **memory bank mechanism** is a novel contribution that provides invariant knowledge anchoring for cross-domain graph learning.

## Limitations & Future Work

- Pre-training is conducted solely on ogbn-Paper100M (a citation network); generalization to non-citation structures (e.g., social networks, molecular graphs) remains to be verified.
- The student model injects structural information implicitly via PPR scores, potentially losing some precise topological information.
- The selection of memory bank size $L$ lacks theoretical guidance.
- A more comprehensive comparison with recent graph foundation model methods (e.g., extended versions of GraphMAE) has not been conducted.

## Related Work & Insights

- SSTAG shares conceptual similarities with methods such as UniGraph but follows a different technical route; integrating existing graph prompt methods could be a promising direction.
- The idea of distilling into an MLP is consistent with GNN-to-MLP distillation works such as GLNN, but extends this line of research by incorporating the LLM dimension.
- The framework could potentially be extended to heterogeneous or dynamic graph settings.

## Rating

- Novelty: ⭐⭐⭐⭐ (Dual distillation of LLM+GNN into an MLP is novel; the memory bank mechanism further enhances the contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐ (12 cross-domain datasets, multi-task evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Clear method description, complete mathematical derivations)
- Value: ⭐⭐⭐⭐ (Provides a viable technical pathway toward graph foundation models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[AAAI 2026\] GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs](../../AAAI2026/graph_learning/gcl-ot_graph_contrastive_learning_with_optimal_transport_for_heterophilic_text-a.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)

</div>

<!-- RELATED:END -->
