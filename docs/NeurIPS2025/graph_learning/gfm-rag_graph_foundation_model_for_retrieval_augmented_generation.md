---
title: >-
  [Paper Note] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation
description: >-
  [NeurIPS 2025][Graph Learning][Graph Foundation Model] This paper proposes GFM-RAG, the first graph foundation model-driven retrieval-augmented generation framework…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Graph Foundation Model"
  - "RAG"
  - "Knowledge Graph"
  - "Multi-hop Reasoning"
  - "GNN"
date: 2026-05-08
content_hash: 7d0595fd887a9425
---

# GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation

**Conference**: NeurIPS 2025
**arXiv**: [2502.01113](https://arxiv.org/abs/2502.01113)
**Code**: [https://github.com/rmanluo/gfm-rag](https://github.com/rmanluo/gfm-rag)
**Area**: Graph Learning / RAG / Knowledge Graphs
**Keywords**: Graph Foundation Model, RAG, Knowledge Graph, Multi-hop Reasoning, GNN

## TL;DR
This paper proposes GFM-RAG, the first graph foundation model-driven retrieval-augmented generation framework, which performs single-pass multi-hop reasoning over knowledge graphs via a query-dependent GNN. With only 8M parameters, GFM-RAG achieves zero-shot generalization to unseen datasets and substantially outperforms state-of-the-art methods on multi-hop QA retrieval benchmarks.

## Background & Motivation

**Background**: RAG is the dominant paradigm for injecting external knowledge into LLMs. Traditional RAG encodes documents as independent vectors for retrieval, which performs poorly on multi-hop questions requiring cross-document reasoning. GraphRAG methods (e.g., HippoRAG, LightRAG) explicitly model inter-knowledge relationships via graph structures.

**Limitations of Prior Work**: (a) Traditional vector retrieval fails to capture complex inter-document relationships; (b) multi-step retrieval methods (e.g., IRCoT) improve performance through iterative LLM reasoning but incur prohibitive computational overhead (several seconds per query); (c) existing GraphRAG methods (e.g., HippoRAG using Personalized PageRank) rely heavily on graph structure, which is often noisy and incomplete; (d) existing GNN-based methods require training from scratch for each new dataset, lacking generalizability.

**Key Challenge**: How can multi-hop reasoning capability be achieved within a single-step retrieval while maintaining cross-dataset generalizability?

**Goal**: To design a transferable Graph Foundation Model (GFM) that performs multi-hop reasoning retrieval in a single forward pass and generalizes directly to unseen datasets after pretraining.

**Key Insight**: The multi-hop message passing of a query-dependent GNN is theoretically equivalent to multi-hop logical reasoning on graphs. By mapping queries, entities, and relations into a unified semantic space, the model becomes universally applicable across different graphs.

**Core Idea**: A unified semantic space combined with query-dependent message-passing GNN is pretrained on large-scale KGs to produce a cross-dataset transferable graph foundation model retriever.

## Method

### Overall Architecture
GFM-RAG operates in three stages: (1) **KG-index construction**: entities and relations are extracted from documents to build a knowledge graph index; (2) **GFM Retriever**: a query-dependent GNN reasons over the KG and outputs relevance scores for each entity with respect to the query; (3) **Document ranking and generation**: documents are ranked by weighted entity scores and fed into an LLM to generate the final answer.

Input: user query $q$ + document corpus $\mathcal{D}$
Output: top-K relevant documents $\mathcal{D}^K$ and LLM-generated answer $a$

### Key Designs

1. **KG-index Construction**:

    - **Function**: Extracts (entity, relation, entity) triples from documents via LLM-driven OpenIE to construct the knowledge graph index.
    - **Mechanism**: In addition to directly extracted triples $\mathcal{T}$, equivalence edges $\mathcal{T}^+$ are added via entity resolution (embedding similarity), e.g., linking "USA" ↔ "United States of America", to enhance graph connectivity.
    - **Design Motivation**: Analogous to the hippocampal memory indexing theory, the KG-index serves as an "artificial hippocampus" that stores inter-knowledge associations, addressing the loss of relational structure inherent in independent vector encoding.

2. **Query-dependent GNN (GFM Retriever)**:

    - **Function**: Performs query-conditioned message passing over the KG to compute relevance scores of each entity with respect to the query.
    - **Mechanism**:
        - **Initialization**: The query is encoded as $\bm{q} \in \mathbb{R}^d$ by a sentence embedding model; entities mentioned in the query are initialized with $\bm{q}$, while all others are initialized to zero vectors.
        - **Message Passing**: $L$ layers of query-dependent message passing are applied; relation embeddings are initialized using the same sentence model and updated via layer-specific MLPs; the message function employs non-parametric DistMult, with sum aggregation and linear layer updates.
        - **Output**: A final MLP with sigmoid maps entity representations to relevance scores $P_q \in \mathbb{R}^{|\mathcal{E}| \times 1}$.
    - **Design Motivation**: Query-dependent message passing is theoretically proven to be equivalent to multi-hop logical reasoning (NBFNet), where $L$ layers of message passing correspond to $L$-hop reasoning. The unified semantic space (query/entity/relation initialized with the same embedding model) enables cross-graph generalization.
    - **Novelty**: Unlike conventional graph-specific GNNs, this approach achieves cross-graph transfer through semantic initialization.

3. **Two-Stage Training**:

    - **Function**: Self-supervised pretraining followed by supervised fine-tuning.
    - **Mechanism**:
        - **Stage 1 — KG Completion Pretraining**: Head or tail entities in triples are randomly masked, and the GNN is trained to predict the masked entity, enhancing graph reasoning capability.
        - **Stage 2 — Document Retrieval Fine-tuning**: The model is trained on annotated retrieval datasets where queries are natural language questions and target entities are derived from annotated supporting documents.
        - **Loss Function**: A weighted combination of BCE loss and ranking loss, $\mathcal{L} = \alpha \mathcal{L}_{BCE} + (1-\alpha) \mathcal{L}_{RANK}$, where ranking loss mitigates gradient vanishing caused by sparse positive samples.
    - **Training Scale**: 60 KGs, 14M+ triples, 700k documents.

4. **Document Ranking**:

    - **Function**: Converts entity-level scores into document-level scores.
    - **Mechanism**: Top-$T$ entities are selected and weighted by inverse document frequency (analogous to IDF); document scores are computed as $P_d = M^\top F_e$ via an entity-to-document inverted index $M$.
    - **Design Motivation**: High-frequency entities appearing in many documents have low discriminative power; inverse frequency weighting reduces their influence.

### Loss & Training
- Joint optimization with BCE + ranking loss, $\alpha = 0.3$.
- Negative samples are randomly drawn from the KG.
- Training uses 8 × A100 GPUs, batch size 4, learning rate $5 \times 10^{-4}$.
- The model has only 8M parameters, with 6 message-passing layers and hidden dimension 512.

## Key Experimental Results

### Main Results — Multi-hop Retrieval

| Dataset | Metric | GFM-RAG | IRCoT+HippoRAG (SOTA) | Gain |
|--------|------|---------|----------------------|------|
| HotpotQA | R@2 | **78.3** | 67.0 | +16.9% |
| MuSiQue | R@2 | **49.1** | 45.3 | +8.4% |
| 2Wiki | R@2 | **90.8** | 75.8 | +19.8% |
| HotpotQA | R@5 | **87.1** | 83.0 | +4.9% |
| MuSiQue | R@5 | **58.2** | 57.6 | +1.0% |
| 2Wiki | R@5 | **95.6** | 93.9 | +1.8% |

### Multi-hop QA

| Dataset | Metric | Ours | IRCoT+GFM-RAG | Prev. SOTA |
|--------|------|---------|---------------|---------|
| HotpotQA | EM | 51.6 | **56.0** | 48.7 (FLARE) |
| MuSiQue | EM | 30.2 | **36.6** | 21.9 (IRCoT+HippoRAG) |
| 2Wiki | EM | 69.8 | **72.5** | 48.9 (Adaptive-RAG) |

### Efficiency Analysis

| Method | HotpotQA Time (s) | R@5 |
|------|-----------------|-----|
| ColBERTv2 | 0.035 | 79.3 |
| HippoRAG | 0.255 | 77.7 |
| IRCoT+HippoRAG | 3.162 | 83.0 |
| **GFM-RAG** | **0.107** | **87.1** |

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| w/o pretraining | Significant performance drop; pretraining is critical for generalization |
| BCE loss only | Inferior to BCE+Ranking; positive sparsity issue confirmed |
| w/o entity resolution | Reduced KG connectivity hinders multi-hop reasoning |
| Different sentence models | Performance is insensitive; confirms framework generality |

### Key Findings
- GFM-RAG surpasses all multi-step methods in a single pass, achieving ~30× efficiency over IRCoT+HippoRAG.
- Zero-shot generalization across 7 domain-specific RAG datasets, outperforming HippoRAG by an average of 18.9%.
- Model performance follows a neural scaling law: $z \propto 0.24 x^{0.05} + 0.11 y^{0.03}$, indicating that larger data and model scales can yield further gains.

## Highlights & Insights
- **Elegant unified semantic space design**: Query, entity, and relation representations are all initialized using the same sentence embedding model, enabling the GNN to transfer naturally to any new graph — a key design for achieving a "graph foundation model."
- **Theoretical guarantee of single-step multi-hop equivalence**: $L$ layers of query-dependent message passing are theoretically equivalent to $L$-hop logical reasoning, avoiding the LLM overhead of multi-step retrieval.
- **Path interpretability**: Multi-hop reasoning paths can be extracted via gradient backpropagation, enhancing trustworthiness.
- **Inverse-frequency-weighted document ranking** draws on TF-IDF intuition to efficiently convert entity scores into document scores.

## Limitations & Future Work
- **KG construction depends on LLMs**: OpenIE extraction quality directly impacts KG quality; results vary across different LLMs and may degrade for low-resource languages.
- **8M parameters vs. scaling law**: Although a scaling law is demonstrated, training is only conducted at the 8M scale; whether larger-scale training encounters bottlenecks remains unknown.
- **Entity resolution is a bottleneck**: Current embedding-similarity-based resolution may fail for synonymous entities with dissimilar embeddings.
- **KG construction overhead**: Rebuilding the KG-index for each new dataset requires LLM calls, which is non-trivial in cost.
- **Potential improvements**: Combining GFM-RAG's graph reasoning with dense retrieval for hybrid retrieval; exploring larger-scale pretraining.

## Related Work & Insights
- **vs. HippoRAG**: HippoRAG uses Personalized PageRank for graph retrieval, relying entirely on graph structure; GFM-RAG learns to reason via GNN, making it more robust to noisy and incomplete graphs.
- **vs. IRCoT**: IRCoT requires multi-step iterative LLM reasoning with high overhead; GFM-RAG achieves equivalent multi-hop reasoning in a single GNN pass, at 30× greater efficiency.
- **vs. ULTRA/GFT and other graph foundation models**: These works focus on graph tasks (node classification, link prediction); GFM-RAG is the first graph foundation model targeting RAG.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to apply a graph foundation model to RAG, with an elegant unified semantic space design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 3 multi-hop QA benchmarks + 7 domain-specific datasets + efficiency analysis + scaling law + ablation study.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed method description, though notation is occasionally dense.
- Value: ⭐⭐⭐⭐⭐ Provides a powerful and generalizable solution for GraphRAG; zero-shot generalization achieved with only 8M parameters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](../../CVPR2026/graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[NeurIPS 2025\] ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG](remindrag_low-cost_llm-guided_knowledge_graph_traversal_for_efficient_rag.md)
- [\[NeurIPS 2025\] Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking](diagnosing_and_addressing_pitfalls_in_kg-rag_datasets_toward_more_reliable_bench.md)
- [\[NeurIPS 2025\] FastJAM: a Fast Joint Alignment Model for Images](fastjam_a_fast_joint_alignment_model_for_images.md)
- [\[NeurIPS 2025\] Reasoning Meets Representation: Envisioning Neuro-Symbolic Wireless Foundation Models](reasoning_meets_representation_envisioning_neuro-symbolic_wireless_foundation_mo.md)

</div>

<!-- RELATED:END -->
