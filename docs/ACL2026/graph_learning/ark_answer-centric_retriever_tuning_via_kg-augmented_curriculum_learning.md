---
title: >-
  [Paper Note] ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning
description: >-
  [ACL 2026][Graph Learning][Answer-Centric Retrieval] ARK filters positive samples through three-dimensional answer sufficiency scoring (Forward + Backward + Retriever alignment) and generates progressively difficult hard…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Answer-Centric Retrieval"
  - "Knowledge Graph Augmentation"
  - "Curriculum Learning"
  - "Contrastive Learning"
  - "Long-Context RAG"
content_hash: e66cb1747ae8b6e2
---

# ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning

**Conference**: ACL 2026
**arXiv**: [2511.16326](https://arxiv.org/abs/2511.16326)  
**Code**: [GitHub](https://github.com/valleysprings/ARK/)  
**Area**: Graph Learning
**Keywords**: Answer-Centric Retrieval, Knowledge Graph Augmentation, Curriculum Learning, Contrastive Learning, Long-Context RAG

## TL;DR
ARK filters positive samples through three-dimensional answer sufficiency scoring (Forward + Backward + Retriever alignment) and generates progressively difficult hard negatives via LLM-constructed knowledge graphs for curriculum contrastive learning, averaging +14.5% F1 across 10 datasets.

## Background & Motivation

**Key Challenge**: The gap between retriever training objective (query-document similarity) and RAG's ultimate goal (generating correct answers).

**Core Idea**: Use KG subgraph-generated augmented queries to mine progressively difficult hard negatives through curriculum contrastive learning, teaching the retriever to distinguish "sufficient" from "seemingly relevant but insufficient" evidence.

## Method

### Key Designs

1. **Three-Dimensional Answer Sufficiency Scoring**: Forward alignment $S_f$ = whether a chunk suffices to generate the answer; Backward alignment $S_b$ = whether the question can be reconstructed from answer + chunk; Parameter alignment $S_v$ = original retriever cosine similarity.

2. **KG-Driven Hard Negative Mining**: Large subgraph ($Q_L^{aug}$) generates easier negatives; small subgraph ($Q_S^{aug}$) generates harder negatives — more focused subgraphs produce queries closer to the correct answer's "semantic neighborhood."

3. **Curriculum Contrastive Learning**: Three-stage curriculum progressing from in-batch random negatives to hard negatives from $Q_L^{aug}$ then $Q_S^{aug}$.

## Key Experimental Results

- Average +14.5% F1 across 10 datasets
- SOTA on 8/10 datasets (Ultradomain + LongBench)

## Highlights & Insights
- Redefines KG's role in RAG — from "retrieval index" to "training signal generator" — drastically reducing KG usage cost
- Plug-and-play without changing retriever architecture

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](../../AAAI2026/graph_learning/feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[ICML 2026\] Structure-Centric Graph Foundation Model via Geometric Bases](../../ICML2026/graph_learning/structure-centric_graph_foundation_model_via_geometric_bases.md)
- [\[NeurIPS 2025\] Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking](../../NeurIPS2025/graph_learning/diagnosing_and_addressing_pitfalls_in_kg-rag_datasets_toward_more_reliable_bench.md)

</div>

<!-- RELATED:END -->
