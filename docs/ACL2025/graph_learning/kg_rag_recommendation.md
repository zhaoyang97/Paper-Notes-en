---
title: >-
  [Paper Note] Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)
description: >-
  [ACL 2025][Graph Learning][knowledge graph] The K-RagRec framework is proposed, which provides structured and reliable external knowledge for LLM-based recommender systems by retrieving multi-hop subgraphs from knowledge graphs. Combining a popularity-based selective retrieval strategy and a GNN encoder, it effectively mitigates hallucination and knowledge deficit issues in LLM recommendations.
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "knowledge graph"
  - "RAG"
  - "LLM recommendation"
  - "GNN"
  - "sub-graph retrieval"
date: 2026-05-08
content_hash: c4d0e10e1e9cec4c
---

# Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)

**Conference**: ACL 2025  
**arXiv**: [2501.02226](https://arxiv.org/abs/2501.02226)  
**Code**: Unavailable  
**Area**: Recommender Systems/Graph Learning  
**Keywords**: knowledge graph, RAG, LLM recommendation, GNN, sub-graph retrieval  

## TL;DR

The K-RagRec framework is proposed, which provides structured and reliable external knowledge for LLM-based recommender systems by retrieving multi-hop subgraphs from knowledge graphs. Combining a popularity-based selective retrieval strategy and a GNN encoder, it effectively mitigates hallucination and knowledge deficit issues in LLM recommendations.

## Background & Motivation

- **Problem Definition**: LLM-based recommender systems face three inherent limitations: (1) hallucination (recommending non-existent items); (2) knowledge obsolescence (training data cutoff prevents recommending new items); (3) lack of domain-specific knowledge (limited recommendation corpora during pre-training).
- **Limitations of RAG**: Traditional text-based RAG introduces noise and harmful interference while ignoring structural relationships between entities, thereby limiting LLM reasoning capability.
- **Advantages of Knowledge Graphs**: KGs provide structured, factual, and editable knowledge representations, making them a natural choice to combat hallucinations.
- **Key Challenge**: (1) Retrieving only first-order neighbors fails to capture high-order relationships; (2) indiscriminate retrieval reduces efficiency; (3) text serialization of KG triples fails to fully exploit structural information.

## Method

### Overall Architecture

K-RagRec comprises five core components: (1) multi-hop knowledge subgraph semantic indexing; (2) popularity-based selective retrieval strategy; (3) knowledge subgraph retrieval; (4) knowledge subgraph re-ranking; and (5) knowledge-augmented recommendation generation.

### Key Designs

- **Multi-hop Subgraph Indexing**: SentenceBERT is used to encode the textual attributes of KG nodes and edges $\to$ a GNN aggregates multi-hop neighbor information to obtain the subgraph representation $z_{g_o}$ $\to$ stored in a vector database. The $l$-hop GNN representation is equivalent to the $l$-hop neighborhood subgraph representation of the node, achieving flexible chunking from coarse to fine grain.
- **Popularity-based Selective Retrieval**: KG retrieval is determined by item popularity (e.g., sales, views)—retrieval is performed only for cold-start items whose popularity is below a threshold $p$. This aligns with the power-law distribution characteristic (a few popular items already have sufficient internal knowledge, whereas cold-start items need augmentation), significantly reducing retrieval latency.
- **Subgraph Re-ranking + GNN Encoding**: After retrieving the Top-K subgraphs, they are re-ranked using the recommendation prompt as the query to select the Top-N subgraphs $\to$ a second GNN encoder extracts structural information $\to$ an MLP projector aligns it to the LLM embedding space as a soft prompt prefix.

### Loss & Training

Cross-entropy loss $\mathcal{L}(Y, A)$, where $Y$ is the ground-truth recommended item and $A$ is the LLM prediction. Only the parameters of the two GNNs and the MLP projector are trained, while the LLM parameters are frozen.

## Experiments

### Main Results (LLaMA-2, Frozen LLM + Prompt Tuning, Candidate set $M=20$)

| Method | ML-1M ACC | ML-1M R@3 | ML-1M R@5 | ML-20M ACC | ML-20M R@3 | Amazon ACC | Amazon R@5 |
|------|-----------|-----------|-----------|------------|------------|------------|------------|
| KG-Text | 0.076 | - | - | 0.052 | - | 0.058 | - |
| KAPING | 0.079 | - | - | 0.069 | - | 0.063 | - |
| PT w/ KG-Text | 0.078 | 0.191 | 0.308 | 0.051 | 0.152 | 0.074 | 0.245 |
| GraphToken w/ RAG | 0.268 | 0.421 | 0.466 | 0.186 | 0.433 | 0.326 | 0.624 |
| G-retriever | 0.274 | 0.532 | 0.650 | 0.342 | 0.619 | 0.275 | 0.612 |
| **K-RagRec** | **0.435** | **0.725** | **0.831** | **0.600** | **0.850** | **0.508** | **0.780** |
| **Gain** | **+58.6%** | **+33.0%** | **+27.8%** | **+75.4%** | **+37.3%** | **+55.8%** | **+25.0%** |

### Ablation Study

| Variant | Impact |
|------|------|
| Remove multi-hop indexing (first-order neighbors only) | Performance drops significantly; unable to capture high-order relationships |
| Remove selective retrieval (retrieval for all) | Slight performance drop + significant decrease in retrieval efficiency |
| Remove re-ranking | Irrelevant subgraphs interfere with generation |
| Remove GNN encoding (KG text serialization) | Insufficient utilization of structural information, leading to performance degradation |
| Remove popularity strategy | Increased retrieval time; retrieval for popular items introduces noise |

### Key Findings

- K-RagRec significantly outperforms all baselines across all datasets and metrics, with ACC improving by 55-75%.
- Compared to retrieving only triples or first-order neighbors, the multi-hop subgraph indexing provides a more comprehensive view of item knowledge.
- The popularity-based selective retrieval strategy significantly reduces retrieval overhead while maintaining performance.
- The GNN encoder + projector approach outperforms KG text serialization, better preserving graph structural information.
- Fine-tuning (LoRA w/ K-RagRec) can further improve performance by 3-16%.

## Highlights & Insights

- A systematic framework for knowledge graph RAG in recommender systems, forming a complete closed-loop process from indexing, retrieval, and re-ranking to encoding.
- The popularity-based selective retrieval strategy balances both efficiency and effectiveness, featuring a simple design that aligns with the power-law distribution characteristic of recommender systems.
- GNNs are used as native encoders for graph structure instead of textual serialization, avoiding long context issues and loss of structural information.

## Limitations & Future Work

- The performance heavily relies on the quality and coverage of external KGs (e.g., Freebase); incomplete or outdated KGs may limit its effectiveness.
- Training GNNs and projectors increases deployment complexity and requires alignment between recommendation data and the KG.
- The popularity threshold $p$ requires dataset-specific tuning, and the optimal threshold may vary widely across different domains.
- Primarily validated on movie and book recommendations; more complex scenarios like e-commerce and news require further exploration.
- The two-stage GNN (indexing + encoding) increases model parameters and training complexity.
- Comparisons with the latest Graph RAG methods (such as GraphRAG, LightRAG) are missing.

## Related Work & Insights

- RAG: REALM (Guu et al. 2020), DPR (Karpukhin et al. 2020), RETRO (Borgeaud et al. 2022)
- Graph-Augmented RAG: G-Retriever (He et al. 2024), Retrieve-Rewrite-Answer (Wu et al. 2023b)
- LLM Recommendation: TALLRec (Bao et al. 2023), A First Look at RAG in Recommendation (Di Palma 2023)
- Knowledge Graph-based Recommendation: KGAT, KAPING (Baek et al. 2023)
- Graph Tokenization: GraphToken (Perozzi et al. 2024)

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[ACL 2025\] SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation](simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme.md)
- [\[ACL 2026\] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](../../ACL2026/graph_learning/megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)
- [\[ACL 2026\] STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](../../ACL2026/graph_learning/stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)
- [\[ACL 2025\] mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages](mrakl_multilingual_retrieval-augmented_knowledge_graph_construction_for_low-reso.md)

</div>

<!-- RELATED:END -->
