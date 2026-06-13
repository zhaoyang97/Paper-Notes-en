---
title: >-
  [Paper Note] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge
description: >-
  [ICLR 2026][Information Retrieval & RAG][graph foundation model] This paper proposes G-reasoner, which standardizes heterogeneous knowledge sources via a four-layer unified graph interface called QuadGraph…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "graph foundation model"
  - "RAG"
  - "knowledge graph"
  - "GNN"
  - "LLM reasoning"
date: 2026-05-08
content_hash: bc44dcac26cccd3a
---

# G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge

**Conference**: ICLR 2026
**arXiv**: [2509.24276](https://arxiv.org/abs/2509.24276)  
**Code**: [Project Page](https://rmanluo.github.io/gfm-rag/)  
**Area**: Self-Supervised Learning / Graph Foundation Models / RAG
**Keywords**: graph foundation model, RAG, knowledge graph, GNN, LLM reasoning

## TL;DR
This paper proposes G-reasoner, which standardizes heterogeneous knowledge sources via a four-layer unified graph interface called QuadGraph, trains a 34M-parameter GNN-based graph foundation model to jointly reason over graph topology and textual semantics, and achieves state-of-the-art performance over existing GraphRAG methods across 6 benchmarks in conjunction with an LLM.

## Background & Motivation

**Background**: LLMs excel at reasoning but are constrained by static parametric knowledge. RAG augments LLMs with external knowledge. Graphs naturally model relational knowledge (knowledge graphs, document graphs, hierarchical graphs, etc.), and GraphRAG aims to combine both paradigms.

**Limitations of Prior Work**: Existing GraphRAG methods are tailored to specific graph structures (knowledge graphs, document graphs, and hierarchical graphs each require distinct designs), rely on heuristic search (e.g., PageRank), or involve costly agent pipelines (multiple LLM calls), resulting in poor generalizability and low efficiency.

**Key Challenge**: Different knowledge sources require different graph structures, yet no unified framework exists that can adapt to diverse graph structures and perform efficient reasoning.

**Goal**: Design a unified graph representation and reasoning framework that is compatible with multiple knowledge graph structures, efficient, and generalizable.

**Key Insight**: Define a four-layer standardized graph structure, QuadGraph, and use a GNN-based graph foundation model for unified reasoning.

**Core Idea**: Unify heterogeneous graphs into QuadGraph (attribute layer + knowledge graph layer + document layer + community layer), train a GFM to jointly reason over topology and semantics, and augment LLM generation.

## Method

### Overall Architecture
(1) QuadGraph normalizes diverse graph structures into a standardized 4-layer format; (2) a 34M-parameter GNN foundation model performs reasoning over QuadGraph; (3) the retrieved results are passed to an LLM to generate the final answer.

### Key Designs

1. **QuadGraph Unified Graph Interface**:

    - **Function**: Standardize heterogeneous knowledge sources.
    - **Four-layer structure**: Attribute layer (common node properties) → Knowledge graph layer (entity + relational triples) → Document layer (unstructured text) → Community layer (global information from semantic/structural clustering).
    - **Design Motivation**: Knowledge graphs, document graphs, and hierarchical graphs can all be mapped onto these four layers, eliminating dependence on specific graph structure designs.

2. **GNN Graph Foundation Model (GFM)**:

    - **Function**: Jointly reason over graph topology and textual semantics.
    - **Mechanism**: Employs a query-dependent GNN with a DistMult message function; node embeddings are initialized via a pretrained text encoder; after $L$ layers of message passing, type-specific predictors score the relevance of each node type.
    - **Weak supervision training**: A pretrained text encoder serves as a "teacher" providing pseudo-labels, which are distilled into the GFM via KL divergence, alleviating the scarcity of annotated data.

3. **Distributed Message Passing**:

    - **Function**: Enable large-scale training and inference.
    - **Mechanism**: The METIS algorithm partitions the graph across multiple GPUs, with each device storing a subgraph; message passing proceeds via local aggregation followed by cross-device communication.
    - Mixed-precision training yields a 2.1× throughput improvement and a 17.5% reduction in GPU memory usage.

### Loss & Training
Log-likelihood over labeled nodes + $\lambda \times$ KL distillation loss from teacher pseudo-labels; large-scale weakly supervised training across multiple datasets.

## Key Experimental Results

### Main Results: Multi-hop QA + G-bench

| Method | HotpotQA F1 | MuSiQue F1 | 2Wiki F1 |
|--------|-------------|------------|----------|
| BM25 | 63.4 | 28.8 | 51.2 |
| HippoRAG 2 | 71.1 | 49.3 | 69.7 |
| GFM-RAG | 69.5 | 49.2 | 77.7 |
| **G-reasoner** | **76.0** | **52.5** | **82.1** |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| w/o textual semantic integration | Significant drop; topology alone is insufficient |
| w/o weak supervision distillation | Performance drop due to insufficient labeled data |
| Single graph type training | Poor generalization |
| Full G-reasoner | Best performance |

### Key Findings
- Achieves comprehensive state-of-the-art results across all 6 benchmarks, including multi-hop QA and G-bench.
- Substantially more efficient than agent-based methods (ToG, KAG) — single forward pass vs. multiple LLM calls.
- Strong cross-graph-type generalization — performs well on unseen graph structures.

## Highlights & Insights
- QuadGraph's four-layer design elegantly unifies knowledge graphs, document graphs, and hierarchical graphs, demonstrating strong abstraction capability.
- The weak supervision distillation strategy addresses the practical challenge of annotation scarcity in graph reasoning.
- The 34M-parameter GFM significantly reduces inference costs compared to agent-based approaches.

## Limitations & Future Work
- Whether QuadGraph's four-layer structure covers all knowledge types remains an open question; temporal and multimodal knowledge may require extensions.
- The GFM is dependent on the quality of the pretrained text encoder.
- Validation is limited to the textual domain; multimodal reasoning remains unexplored.

## Related Work & Insights
- **vs. GFM-RAG**: The predecessor work, restricted to knowledge graphs; G-reasoner extends the framework to arbitrary graph structures.
- **vs. GraphRAG (MS)**: Relies on a specific hierarchical graph and LLM-generated summaries, resulting in poor generalizability.
- **vs. HippoRAG**: Uses PageRank for retrieval, without fully leveraging the capabilities of foundation models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The QuadGraph unified interface combined with joint GFM reasoning constitutes a meaningful systems contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 6 benchmarks, with ablations, efficiency analysis, and generalization tests.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework presentation with effective figures.
- **Value**: ⭐⭐⭐⭐ Provides a scalable and unified solution for GraphRAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models](synthworlds_controlled_parallel_worlds_for_disentangling_reasoning_and_knowledge.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)
- [\[ACL 2026\] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](../../ACL2026/information_retrieval/ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](../../NeurIPS2025/information_retrieval/hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)

</div>

<!-- RELATED:END -->
