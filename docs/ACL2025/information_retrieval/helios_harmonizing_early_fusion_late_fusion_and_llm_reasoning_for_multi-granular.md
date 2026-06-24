---
title: >-
  [Paper Note] HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval
description: >-
  [ACL 2025][Information Retrieval & RAG][Table-Text Retrieval] Proposes HELIOS, a three-stage graph retrieval framework (edge-level early fusion $\rightarrow$ node-level late fusion $\rightarrow$ star-graph-level LLM refinement). Through multi-granular coordination, it addresses the three major challenges in table-text retrieval: retrieval unit granularity, query dependency relationship discovery, and advanced reasoning, achieving a 42.6% improvement in Answer Recall on OTT-QA…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Table-Text Retrieval"
  - "Bipartite Graph"
  - "Early Fusion"
  - "Late Fusion"
  - "LLM Reasoning"
  - "Multi-hop QA"
  - "ColBERT"
  - "Graph Retrieval"
date: 2026-05-08
content_hash: ad4446f5c7b605e4
---

# HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval

**Conference**: ACL 2025  
**arXiv**: [2603.02248](https://arxiv.org/abs/2603.02248)  
**Code**: Not released  
**Area**: Information Retrieval / Table-Text Retrieval  
**Keywords**: Table-Text Retrieval, Bipartite Graph, Early Fusion, Late Fusion, LLM Reasoning, Multi-hop QA, ColBERT, Graph Retrieval  

## TL;DR

Proposes HELIOS, a three-stage graph retrieval framework (edge-level early fusion $\rightarrow$ node-level late fusion $\rightarrow$ star-graph-level LLM refinement). Through multi-granular coordination, it addresses the three major challenges in table-text retrieval: retrieval unit granularity, query dependency relationship discovery, and advanced reasoning, achieving a 42.6% improvement in Answer Recall on OTT-QA.

## Background & Motivation

- **Problem Definition**: Open-domain table-text retrieval requires retrieving relevant information simultaneously from structured tables and unstructured text to support cross-modal multi-hop reasoning for answering complex questions.
- **Limitations of Prior Work**: (1) **Early fusion** (OTTeR, DoTTeR) pre-connects table rows and passages through entity linking to form fused blocks, which often contain query-irrelevant passages and fail to capture dynamic relationships; (2) **Late fusion** (CORE, COS) dynamically matches but suffers from error propagation in beam search; (3) Both approaches rely on semantic similarity, making them unable to handle queries requiring column-level aggregation or multi-hop logical reasoning.
- **Research Motivation**: A unified framework is needed to employ appropriate granularities (edge/node/star graph) at different retrieval stages, combining predefined relationships (early fusion) with dynamically discovered relationships (late fusion) while incorporating LLM reasoning capabilities to handle complex queries.
- **Core Innovation**: Formulates table-text relationships as a bipartite graph and designs a three-stage multi-granular retrieval framework where each stage operates at a different granularity.

## Method

### Overall Architecture

HELIOS models the corpus as a bipartite graph $G = (V, E)$, where nodes represent table segments and passages, and edges represent their relationships. The three-stage workflow is as follows:

1. **Edge-based Bipartite Subgraph Retrieval** $\rightarrow$ utilizes early fusion
2. **Query-relevant Node Expansion** $\rightarrow$ utilizes late fusion
3. **Star-based LLM Refinement** $\rightarrow$ utilizes LLM reasoning

### Key Designs

1. **Stage 1: Edge-level Retrieval**: A bipartite data graph $G_d$ is constructed offline via entity linking. Each edge (table segment + passage) is linearized and encoded into multi-vector representations using ColBERTv2 to compute late interaction similarity. It first retrieves the top-$k_1$ edges and then uses an all-to-all reranker to select the top-$k_2$ edges, which are merged and deduplicated into a candidate subgraph $G_c$. Using **edges** instead of nodes as the retrieval unit balances information completeness (avoiding the partial relevance issue of late fusion) and noise control (avoiding the excessively large fused blocks of early fusion).

2. **Stage 2: Node Expansion**: Expansion is conducted at the finest granularity (node-level) to address query-irrelevant nodes introduced by early fusion. A two-step beam search method is used: (1) **Seed Node Selection**—utilizes an all-to-all reranker to identify the top-$b$ nodes in $G_c$ most relevant to the query; (2) **Seed Node Expansion**—utilizes the query expansion technique $sim([q;\Gamma(u)], v)$ to retrieve the most relevant neighbors of the seed nodes from the complete bipartite graph, forming the expanded graph $G_l$.

3. **Stage 3: LLM Refinement**: The expanded graph is decomposed into star graph units and sent to the LLM (which outperforms whole-graph prompting by $+12.4\%$), executing two operations: (1) **Column-level Aggregation**—restores the complete table to let the LLM perform aggregation operations (e.g., "the latest record"); (2) **Passage Validation**—performs binary validation on each edge to remove query-irrelevant edges.

### Granularity Selection Principles

| Stage | Granularity | Reason |
|------|------|------|
| Early Fusion | Edge | Contains more context than nodes (reduces partial relevance), and is finer than fused blocks (reduces noise) |
| Late Fusion | Node | Finest granularity, precisely identifying query-relevant nodes and avoiding expanding irrelevant content |
| LLM Refinement | Star Graph | The smallest unit containing multi-hop relationships, more effective than the whole graph (reduces hallucinations) |

## Experimental Results

### Main Results (OTT-QA Dev Set)

| Model | Type | AR@2 | AR@5 | AR@10 | AR@50 | nDCG@50 |
|------|------|------|------|-------|-------|---------|
| OTTeR | Early | 31.4 | 49.7 | 62.0 | 82.0 | 25.9 |
| DoTTeR | Early | 31.5 | 51.0 | 61.5 | 80.8 | 26.7 |
| CORE | Late | 35.3 | 50.7 | 63.1 | 83.1 | 25.4 |
| COS | Late | 44.4 | 61.6 | 70.8 | 87.8 | 33.6 |
| COS w/ ColBERT & bge | Late | 49.6 | 68.2 | 78.7 | 91.7 | 36.5 |
| DoTTeR+COS+LLM | Combination | 50.0 | 62.4 | 70.0 | 84.7 | 34.7 |
| **HELIOS** | **Unified** | **63.3** | **76.7** | **85.0** | **94.2** | **47.0** |

### End-to-End QA Results

| Dataset | Model | EM | F1 |
|-------|------|------|------|
| OTT-QA Test | COS | 54.9 | 61.5 |
| OTT-QA Test | **HELIOS** | **57.0** | **64.3** |
| MMQA Dev | COS | 54.4 | 63.7 |
| MMQA Dev | **HELIOS** | **59.6** | **69.1** |

### Ablation Study

| Removed Component | Change in AR@2 | Change in nDCG@50 | Description |
|---------|----------|-------------|------|
| w/o Node Expansion | -4.1 | — | Dynamic relationship discovery in late fusion is crucial |
| w/o LLM Refinement | -2.9 | — | LLM reasoning is especially critical for aggregation queries |
| Full graph prompt vs. Star graph | — | -12.4% | Decomposing into star graphs is more effective at reducing hallucinations |
| w/o Edge-level Retrieval (using node instead) | Significant decrease | — | Edge-level retrieval is more effective than node-level |

### Key Findings

1. **HELIOS improves AR@2 by 42.6% and nDCG@50 by 39.9% over the SOTA COS** — a massive SOTA breakthrough.
2. **Simply stacking strong modules (DoTTeR+COS+LLM) is far inferior to HELIOS** — indicating that the key lies in the coordinated design of multi-granularity rather than simple combination.
3. **Cross-dataset generalization**: Still achieves an average gain of 20.9% AR on MMQA (which was not targeted in OTT-QA design).
4. **Comparison with HOLMES**: EM improved by 88.4%, F1 by 58.1% — the main advantage comes from edge-level seed retrieval, query-conditioned reasoning, and table structure preservation.

## Highlights & Insights

- Elegantly unifies the representation of early fusion and late fusion in the form of a bipartite graph, offering an elegant theoretical framework.
- The design of using different granularities across three stages (edge $\rightarrow$ node $\rightarrow$ star graph) has clear theoretical motivation and empirical validation.
- Achieves a dramatic SOTA improvement on OTT-QA, with a 42.6% increase in AR@2.
- Comparative experiments demonstrate that simply stacking modules cannot achieve the same effect, reflecting the importance of architectural design.
- Thorough ablation studies, with the contribution of each component independently validated.

## Limitations & Future Work

- High system complexity: involves multiple encoders (ColBERTv2 + multiple rerankers + LLM), leading to high training and inference costs.
- The LLM refinement stage introduces extra latency, potentially limiting real-time applications.
- Verified on only two datasets (OTT-QA and MMQA); datasets for table-text retrieval are limited.
- The quality of entity linking directly affects the construction of the initial bipartite graph, but the cascading impact of linking errors is not discussed in detail.
- The code is not yet publicly available.

## Related Work & Insights

- **Early Fusion**: Fusion-Retriever (Chen et al., 2020a), OTTeR (Huang et al., 2022), DoTTeR (Kang et al., 2024) pre-construct fused blocks through entity linking.
- **Late Fusion**: CORE (Ma et al., 2022), COS (Ma et al., 2023) dynamically form table-passage evidence chains.
- **Graph Methods**: DRAMA (Yuan et al., 2024), HOLMES (Panda et al., 2024) use graphs for multi-hop QA, but are limited to distractor settings.
- **Table Encoding**: GTR (Wang et al., 2021), MGNETS (Chen et al., 2021) use graph methods to improve table encoding.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 8 |
| Technical Depth | 9 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Value | 7 |
| Overall Score | 8.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] IF-GEO: Conflict-Aware Instruction Fusion for Multi-Query Generative Engine Optimization](../../ACL2026/information_retrieval/if-geo_conflict-aware_instruction_fusion_for_multi-query_generative_engine_optim.md)
- [\[ACL 2025\] InspireDebate: Multi-Dimensional Evaluation-Guided Reasoning for Debating](inspiredebate_multidim_evaluation_debating.md)
- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2025\] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)
- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](../../ACL2026/information_retrieval/enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)

</div>

<!-- RELATED:END -->
