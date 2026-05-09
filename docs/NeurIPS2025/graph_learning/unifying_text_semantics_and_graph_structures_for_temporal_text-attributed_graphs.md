---
title: >-
  [Paper Note] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs
description: >-
  [NeurIPS 2025][Graph Learning][Temporal Text-Attributed Graphs] This paper proposes the Cross framework, which employs LLMs to dynamically summarize the semantic evolution of node neighborhoods at strategically sampled temporal points (Temporal Reasoning Chain), then bidirectionally fuses text semantics and graph structural temporal information via a semantic-structural co-encoder. The approach achieves an average MRR improvement of 24.7% on temporal link prediction and a 3.7% AUC gain on an industrial dataset (WeChat).
tags:
  - NeurIPS 2025
  - Graph Learning
  - Temporal Text-Attributed Graphs
  - LLM
  - Semantic-Structural Co-Encoding
  - Link Prediction
  - Cross Framework
date: 2026-05-08
content_hash: 1e1f8e460edcdcc7
---

# Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2503.14411](https://arxiv.org/abs/2503.14411)
**Code**: To be confirmed
**Area**: Graph Learning / NLP
**Keywords**: Temporal Text-Attributed Graphs, LLM, Semantic-Structural Co-Encoding, Link Prediction, Cross Framework

## TL;DR
This paper proposes the Cross framework, which employs LLMs to dynamically summarize the semantic evolution of node neighborhoods at strategically sampled temporal points (Temporal Reasoning Chain), then bidirectionally fuses text semantics and graph structural temporal information via a semantic-structural co-encoder. The approach achieves an average MRR improvement of 24.7% on temporal link prediction and a 3.7% AUC gain on an industrial dataset (WeChat).

## Background & Motivation
**Background**: Temporal graph modeling is a core task in recommendation systems, social networks, and financial risk control. Methods such as TGAT, TGN, and DyGFormer employ attention or memory modules to capture structural dynamics in graphs.

**Limitations of Prior Work**: In temporal text-attributed graphs (TTAGs, e.g., social posts, news, or paper citation networks), nodes carry rich textual attributes. However, existing TGNNs encode text as static embeddings, entirely ignoring the temporal evolution of text semantics—a user's posting topics shift over time, and the context in which a paper is cited also changes.

**Key Challenge**: Structural dynamics and semantic dynamics in graphs are coupled (a user posting on a new topic attracts different interactions), yet existing methods model only one of these dimensions.

**Goal**: Enable TGNNs to simultaneously leverage the temporal evolution of text semantics and the temporal dynamics of graph structure.

**Key Insight**: LLMs inherently understand text semantics and temporal context. The paper uses LLMs to summarize a node's neighborhood semantic state at key temporal points, then bidirectionally fuses this information with structural information from TGNNs.

**Core Idea**: LLM-extracted semantic temporal chains + TGNN-extracted structural temporal information + cross-modal mixer for fusion = comprehensive modeling of temporal text-attributed graphs.

## Method

### Overall Architecture
Two parallel pathways: (1) **Semantic pathway**: LLMs generate neighborhood semantic summaries at strategically sampled temporal points → a Transformer encoder encodes the semantic temporal sequence; (2) **Structural pathway**: a standard TGNN encodes structural dynamics. A cross-modal MLP mixer fuses the final representations.

### Key Designs

1. **Temporal Reasoning Chain**:

    - Function: Strategically samples $m$ temporal points $\hat{t}_1, ..., \hat{t}_m$ from a node's interaction history; at each point, a multi-turn LLM prompt summarizes the semantic state of the node's neighborhood.
    - Sampling strategy: Uniform interval $\lceil n/m \rceil$ selection from $n$ interactions.
    - LLM output: $\hat{d}_u(\hat{t})$—a semantic summary of node $u$ at time $\hat{t}$.
    - Design Motivation: Invoking LLMs at every interaction is prohibitively expensive; strategic sampling balances efficiency and temporal granularity.

2. **Semantic-Structural Co-Encoder**:

    - **Semantic layer**: A Transformer encoder processes the LLM-generated summary sequence with time2vec temporal encoding.
    - **Structural layer**: Any TGNN (TGAT/TGN/DyGFormer) encodes structural features—the framework is agnostic to the specific TGNN architecture.
    - **Cross-modal mixer**: A 2-layer MLP fuses the most recent semantic representation with the structural representation.
    - Final representation: Concatenation of semantic (mean-pooled), structural, and mixed representations.

3. **Plug-and-Play Design**:

    - Function: Cross can be stacked on any TGNN (TGAT+Cross, TGN+Cross, DyGFormer+Cross).
    - Design Motivation: No new TGNN architecture is required; the semantic pathway is simply appended to existing backbones.

### Loss & Training
Standard binary cross-entropy loss for link prediction. LLM inference is performed as an offline preprocessing step.

## Key Experimental Results

### Main Results (Temporal Link Prediction MRR %, Transductive)

| Backbone | Original MRR | +Cross MRR | Gain |
|----------|-------------|-----------|------|
| TGAT | 66.06 | **95.58** | +29.5 |
| TGN | 73.05 | **95.84** | +22.8 |
| DyGFormer | 79.93 | **95.31** | +15.4 |

| Dataset | Max Gain | MRR after +Cross |
|---------|---------|-----------------|
| Enron (email) | +29.52 | 95.58 |
| GDELT (events) | +44.02 | 81.63 |
| Industrial (WeChat) | **+86.23%** | 86.97 |

### Ablation Study

| Configuration | Key Finding | Notes |
|--------------|-------------|-------|
| Inductive setting | Avg. +40.23% | Semantic information benefits new nodes more |
| LLM-only (DeepSeek zero/few-shot) | Far below TGNN | Pure semantics insufficient |
| Semantic pathway only | Below Cross | Structural information remains indispensable |
| Structural pathway only | = Original TGNN | No semantic contribution |
| Node classification (industrial) | AUC +3.7% | Validated on financial risk control task |
| No. of inference points $m$ | 5–10 optimal | Diminishing returns beyond this range |

### Key Findings
- The +86.23% gain on the industrial dataset is the most striking result, demonstrating that semantic information in real-world scenarios is critically important yet consistently underutilized.
- Cross achieves substantially larger gains in the inductive setting than in the transductive setting, indicating that semantic information aids generalization to unseen nodes.
- The LLM-only baseline performs poorly, confirming that pure semantics cannot substitute for structural information—the two are complementary.

## Highlights & Insights
- **Decoupled then fused semantic-structural pathways**: Decoupling the two information pathways before fusion proves more effective than end-to-end joint training, as LLMs and TGNNs each excel at capturing different types of information.
- **Persuasive industrial validation**: A +3.7% AUC improvement on WeChat's financial risk control data represents a meaningful gain in a real-world production environment.
- **Practical plug-and-play utility**: No replacement of the existing TGNN architecture is required—only the addition of the semantic pathway—lowering the barrier to practical adoption.

## Limitations & Future Work
- LLM inference must be invoked at multiple temporal points per node, incurring high computational cost.
- The temporal sampling granularity is governed by $m$, yet the optimal $m$ is dataset-dependent.
- Prompt engineering may require domain-specific adaptation across different application areas.
- The framework assumes LLMs can accurately capture semantic dynamics; multilingual LLMs may be necessary for non-English text.

## Related Work & Insights
- **vs. TGAT/TGN/DyGFormer**: These methods model only structural temporal dynamics; Cross adds a semantic temporal dimension.
- **vs. TAG methods (e.g., TextSAGE)**: TAG methods rely on static text embeddings; Cross employs dynamic LLM-generated summaries.
- **vs. GraphSAGE + LLM**: GraphSAGE does not handle temporality; Cross addresses semantic evolution through the temporal reasoning chain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of LLM temporal reasoning chains with TGNNs constitutes a new paradigm for temporal graph modeling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four public datasets + one industrial dataset + three TGNN backbones.
- Writing Quality: ⭐⭐⭐⭐ The dual-pathway architecture is described clearly.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to temporal text-attributed graph modeling, with industrial validation reinforcing practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[AAAI 2026\] GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs](../../AAAI2026/graph_learning/gcl-ot_graph_contrastive_learning_with_optimal_transport_for_heterophilic_text-a.md)
- [\[NeurIPS 2025\] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)

</div>

<!-- RELATED:END -->
