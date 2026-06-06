---
title: >-
  [Paper Note] RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation
description: >-
  [ICLR 2026][Graph Learning][Retrieval-Augmented Generation] This paper proposes RAS, a framework that dynamically constructs query-specific knowledge graphs at inference time for each input question. Through three stages…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Retrieval-Augmented Generation"
  - "Knowledge Graph Construction"
  - "Iterative Retrieval"
  - "Graph-Structured Reasoning"
  - "LLM Generation"
date: 2026-05-08
content_hash: 391e75da51fcd0a9
---

# RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation

**Conference**: ICLR 2026
**arXiv**: [2502.10996](https://arxiv.org/abs/2502.10996)  
**Code**: Available  
**Area**: Graph Learning
**Keywords**: Retrieval-Augmented Generation, Knowledge Graph Construction, Iterative Retrieval, Graph-Structured Reasoning, LLM Generation

## TL;DR

This paper proposes RAS, a framework that dynamically constructs query-specific knowledge graphs at inference time for each input question. Through three stages—iterative retrieval planning, text-to-triple conversion, and graph-augmented answering—RAS achieves structured reasoning and attains improvements of up to 7.0% and 8.7% over prior methods on 7 knowledge-intensive benchmarks for open-source and closed-source LLMs, respectively.

## Background & Motivation

Although RAG provides LLMs with external knowledge, retrieved text remains unstructured, giving rise to several issues:

**Fragile implicit reasoning chains**: LLMs must internally bridge logical gaps across disparate passages; failures lead to hallucinations.

**Existing KG-RAG methods rely on static global graphs**: Approaches such as GraphRAG require indexing an entire corpus—processing Wikipedia 2018 alone demands millions of LLM calls and tens of thousands of dollars.

**Global graph quality problems**: Mixing evidence from multiple documents may introduce contradictory or ambiguous relations (e.g., conflicting associations for the same drug).

Interpretability research (Lindsey et al., 2025) indicates that LLM errors often stem from failures in implicit reasoning chains, reinforcing the necessity of explicit, structured intermediate knowledge.

**Core Idea**: Rather than pre-building a global KG, RAS constructs a lightweight, query-specific knowledge graph *on demand* at inference time for each query.

## Method

### Overall Architecture

The RAS inference pipeline consists of three iteratively applied stages:

1. **Planning (§3.1)**: Assesses the current knowledge state, decides whether retrieval is needed, and generates sub-queries.
2. **Text Retrieval & Structuring (§3.2)**: Retrieves documents → extracts triples → incrementally merges them into a query-specific KG.
3. **Answering (§3.3)**: Generates the final answer based on the accumulated structured knowledge.

The entire pipeline is driven by a unified Graph LLM combining graph neural network encoding and LoRA fine-tuning.

### Key Designs

#### Knowledge-Aware Planning

*Initial planning*: The model decides between [SUBQ] (retrieval required; initial sub-query equals the original question) and [NO_RETRIEVAL] (answer directly).

*Iterative planning*: Given the accumulated knowledge graph $G_i$ and the history of sub-query chains $[q_0, g_0, \ldots, q_i, g_i]$, the model generates either:
- [SUBQ] $q_{i+1}$: a new focused sub-query to continue retrieval, or
- [SUFFICIENT]: knowledge is adequate; proceed to answering.

$$p_{i+1} \leftarrow \mathcal{M}(\text{GNN}(G_i); \text{INST}_{\text{Plan}}; [q_0, g_0, \ldots, q_i, g_i]; Q)$$

#### Text Retrieval & Structuring

1. **Text retrieval**: A dense retriever (default: Contriever-MS MARCO) retrieves top-$k$ documents.
2. **Text-to-Triples model**: A lightweight model $f_{t2t}$ based on LLaMA-3.2-3B-Instruct, trained on the WikiOfGraph dataset, converts text into $(s, r, o)$ triples.
3. **Incremental knowledge enrichment**: Triples are converted into graph structures $g'_i = (V_i, E_i)$, node/edge attributes are encoded with Sentence-BERT, and the result is merged into the global query graph $G_Q$:

$$G_Q \leftarrow G_Q \cup g'_i$$

#### Knowledge-Augmented Answering

The final answer is generated based on the encoded $G_Q$ and the sub-query chain:

$$A \leftarrow \mathcal{M}(\text{GNN}(G_Q); \text{INST}_{\text{Ans}}; [q_0, g_0, \ldots, q_i, g_i]; Q)$$

The GNN encodes the graph, and the resulting graph representation is fed into the LLM as soft tokens.

#### Structure-Aware Multi-Task Learning

A single LLM is jointly trained on both Planning and Answering tasks using a standard next-token prediction objective. Parameter-efficient fine-tuning is performed via LoRA, while graph components are optimized simultaneously.

### Loss & Training

- **Training data**: The HotpotQA-SUBQ dataset (208K samples) is constructed from HotpotQA, containing iterative sub-queries and [SUFFICIENT] / [NO_RETRIEVAL] labels.
- **Backbone model**: LLaMA-2-7B or LLaMA-3-8B + Graph Transformer encoder.
- **Training procedure**: LoRA fine-tuning + graph component parameter training; Planning and Answering tasks are randomly sampled in a multi-task setup.
- **Triple extractor**: LLaMA-3.2-3B trained on WikiOfGraph, deployed with vLLM.
- **Retrieval corpus**: Wikipedia 2018 (faiss index, split into 5 segments); Wikipedia 2020 for PopQA.
- **Maximum iterations**: 5.

## Key Experimental Results

### Main Results

**7 benchmarks**: TriviaQA, 2WikiMultihopQA, PopQA (open-domain short-form QA), PubHealth, ARC-C (closed-book), ASQA, ELI5 (long-form generation).

| Model | TQA (acc) | 2WQA (F1) | PopQA (acc) | Pub (acc) | ARC (acc) | ASQA (rg/mv) | ELI5 (rg/mv) |
|-------|-----------|-----------|-------------|-----------|-----------|--------------|--------------|
| Self-RAG 7B | 66.4 | 25.1 | 54.9 | 72.4 | 67.3 | 35.7/74.3 | 17.9/35.6 |
| RPG 7B | 65.1 | 33.6 | 56.0 | 73.4 | 65.4 | 37.6/84.4 | 19.1/46.4 |
| **RAS 7B** | **72.7** | **42.1** | **58.3** | **74.7** | **68.5** | **37.2/95.2** | **19.7/47.8** |
| Sonnet-3.5+RAG | 72.5 | 53.7 | 57.3 | 53.9 | 87.1 | 38.8/61.6 | 20.2/32.3 |
| **RAS Sonnet-3.5** | **77.6** | **57.7** | **62.3** | **71.3** | **93.9** | **39.1/70.5** | **23.3/37.7** |

Compared with the previous SOTA (Self-RAG/RPG), RAS 7B achieves +9.7% on short-form QA and +7.9% on long-form generation.

### Ablation Study

| Variant | TQA | 2WQA | Pub | ASQA (rg/mv) |
|---------|-----|------|-----|--------------|
| RAS 7B (full) | 72.7 | 42.1 | 74.7 | 37.2/95.2 |
| w/o GraphEncode (train) | 70.2 | 38.4 | 66.4 | 33.1/85.0 |
| w/o LoRA | 71.5 | 37.8 | 54.8 | 32.8/84.8 |
| w/o Text-to-Triple | 70.4 | 38.2 | 71.4 | 36.2/73.8 |
| w/o Multi-Task | 68.6 | 39.2 | 65.5 | 36.7/88.9 |
| w/o Retrieval (inference) | 56.9 | 27.4 | 69.0 | 31.3/70.6 |
| w/o Planning (inference) | 66.7 | 37.8 | 71.5 | 37.2/95.2 |

### Key Findings

1. **Graph structuring is critical**: Removing Text-to-Triple causes ASQA MAUVE to drop from 95.2 to 73.8 (−22.4%); removing GraphEncode causes PubHealth to drop by 11.2%.
2. **Iterative planning yields significant gains**: Removing Planning leads to −8.8% on TQA and −9.0% on 2WQA.
3. **Role-swap experiment**: RAS 7B's planning capability is on par with Sonnet-3.5, but answering ability remains the primary bottleneck.
4. **Information scales linearly**: Retaining 30–50% of triples already yields noticeable improvements, with gains not saturating at 100%.
5. **Triple extractor selection**: Claude-3.5-Sonnet achieves the best quality but low efficiency (68 tokens/s); LLaMA-3.2-3B balances accuracy and efficiency (4,885 tokens/s).
6. **High data efficiency**: Using only 5% of training data (10K samples) already surpasses the previous SOTA on TQA, 2WQA, and ELI5.

## Highlights & Insights

- **Query-specific KG over global KG**: Eliminates the prohibitive cost of full-corpus graph construction and the noise inherent in global graphs; only a relevant subgraph is built per inference.
- **Unified retrieval–structuring–reasoning framework**: Planning, Structuring, and Answering are completed end-to-end by a single Graph LLM rather than being assembled from independent modules.
- **Exceptionally high MAUVE scores**: RAS 7B achieves MAUVE = 95.2 on ASQA, indicating that generated long-form text is both accurate and fluent.
- **Modular and flexible design**: Planning and Answering can be decoupled, enabling a stronger model for answering and a weaker model for planning.

## Limitations & Future Work

1. The gap between the open-source versions (7B/8B) and closed-source models remains large, particularly on ARC-C (68.5 vs. 93.9).
2. The triple extractor is a standalone model, adding system complexity and latency (end-to-end training is a potential alternative).
3. The maximum of 5 iterations may be insufficient for more complex multi-hop reasoning chains.
4. Graph encoding relies on a simple GNN; stronger graph Transformers or structured attention mechanisms remain unexplored.
5. Performance on ELI5 is unstable, likely due to training data distribution shift.

## Related Work & Insights

- **vs. GraphRAG / G-Retriever**: These methods depend on pre-built global KGs, which are costly and introduce noise; RAS constructs graphs dynamically on demand.
- **vs. Self-RAG / RPG**: Both share the self-reflection/iterative retrieval paradigm, but RAS additionally structures retrieved content into graphs.
- **vs. Chain-of-Thought**: RAS's sub-query chain can be viewed as an explicit reasoning chain, further grounded by structured knowledge.
- **Inspiration**: Future work could combine RAS's graph construction strategy with reinforcement learning (e.g., Search-Agent) to let an agent learn when to structuralize and when to answer directly.

## Rating

- **Novelty**: ★★★★☆ — Dynamic query-specific KG construction represents a valuable new paradigm.
- **Technical Depth**: ★★★★☆ — Multi-module integration is complete; multi-task training design is well-motivated.
- **Experimental Thoroughness**: ★★★★★ — 7 benchmarks, multiple settings, comprehensive ablations, and open-source vs. closed-source comparisons.
- **Writing Quality**: ★★★★☆ — Pipeline diagrams are clear; experimental organization is well-structured.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](../../ACL2026/graph_learning/megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)
- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](../../CVPR2026/graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](../../ACL2026/graph_learning/tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)
- [\[ACL 2026\] STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](../../ACL2026/graph_learning/stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)

</div>

<!-- RELATED:END -->
