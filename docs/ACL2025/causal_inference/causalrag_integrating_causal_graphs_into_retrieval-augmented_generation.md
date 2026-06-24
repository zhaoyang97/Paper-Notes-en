---
title: >-
  [Paper Note] CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation
description: >-
  [ACL 2025][Causal Inference][Retrieval-Augmented Generation] Proposes CausalRAG, which integrates causal graphs into the retrieval process of RAG. It builds a text graph from documents and identifies causal relationships. During querying, it retrieves context through causal path discovery and causal summary generation, significantly improving context precision (92.86%) and retrieval recall in document question answering.
tags:
  - "ACL 2025"
  - "Causal Inference"
  - "Retrieval-Augmented Generation"
  - "Causal Graphs"
  - "Knowledge Graphs"
  - "Document QA"
  - "Graph Indexing"
date: 2026-05-08
content_hash: 1477190a20def759
---

# CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2503.19878](https://arxiv.org/abs/2503.19878)  
**Code**: [https://github.com/Pwnb/CausalRAG](https://github.com/Pwnb/CausalRAG)  
**Area**: RAG / Causal Reasoning  
**Keywords**: Retrieval-Augmented Generation, Causal Graphs, Knowledge Graphs, Document QA, Graph Indexing

## TL;DR
Proposes CausalRAG, which integrates causal graphs into the retrieval process of RAG. It builds a text graph from documents and identifies causal relationships. During querying, it retrieves context through causal path discovery and causal summary generation, significantly improving context precision (92.86%) and retrieval recall in document question answering.

## Background & Motivation

**Background**: RAG enhances the factuality of LLMs by retrieving external knowledge. Standard RAG relies on text chunking and semantic similarity retrieval, while GraphRAG constructs knowledge graphs to assist retrieval.

**Limitations of Prior Work**: (1) Text chunking in standard RAG disrupts document structures and causal chains; (2) Semantic similarity does not equate to causal relevance—similar words do not imply causal relationships; (3) GraphRAG suffers from a tradeoff between retrieval precision and recall (Local has high precision but low recall, while Global is the opposite).

**Key Challenge**: Many complex questions require reasoning along causal chains (A causes B, which causes C), but semantic similarity retrieval cannot perceive such chained relationships.

**Goal**: How to enable RAG systems to understand and leverage causal relationships in documents for retrieval?

**Key Insight**: Identify causal relationships during the graph indexing phase (using LLMs), discover relevant causal paths during the querying phase, and generate causal summaries as the retrieved context.

**Core Idea**: Graph indexing $\rightarrow$ Causal path discovery $\rightarrow$ Causal summarization; a three-step approach to achieve causal-aware RAG.

## Method

### Overall Architecture
Three main steps: (1) **Graph Indexing**: Extract entities and relations from documents to construct a text graph, and use LLMs to identify causal relationships; (2) **Causal Path Discovery**: Find causal paths relevant to the query during querying; (3) **Causal Context Retrieval**: Generate causal summaries along the causal paths to serve as the generation context for the LLM.

### Key Designs

1. **Graph Indexing and Causal Relationship Identification**:

    - **Function**: Build a knowledge graph containing causal relationship annotations from documents.
    - **Mechanism**: Use an LLM to extract entity and relation triplets, then use an LLM to determine whether each relation is causal and annotate its direction.
    - **Design Motivation**: Causal relationships are the most valuable reasoning paths in documents. Explicit annotation allows retrieval to prioritize causal paths.

2. **Causal Path Discovery**:

    - **Function**: Find relevant causal paths in the causal graph given a user query.
    - **Mechanism**: Start from the query entities, expand $k$ hops along causal edges, collect $s$ candidate paths, and use an LLM to select the most relevant causal paths.
    - **Design Motivation**: Causal paths provide a more complete reasoning chain than isolated triplets; $k$ and $s$ control the depth and breadth of retrieval.

3. **Causal Context Retrieval and Summarization**:

    - **Function**: Convert causal paths into natural language summaries to serve as the generation context for the LLM.
    - **Mechanism**: The LLM generates a structured causal summary based on the selected causal paths, retaining the logical relationships of the causal chain.
    - **Design Motivation**: Compared to directly splicing retrieved text chunks, causal summaries are more focused and logical.

### Loss & Training
- Training-free: Entirely based on the in-context learning capabilities of LLMs to perform graph construction, causal identification, path selection, and summary generation.
- Hyperparameters $k$ (number of causal path hops) and $s$ (number of candidate paths) require tuning, with the optimal setting being $k=3, s=3$.

## Key Experimental Results

### Main Results (QA based on OpenAlex academic papers)

| Method | Answer Faithfulness↑ | Context Precision↑ | Context Recall↑ |
|------|---------------------|--------------------|-----------------|
| GraphRAG-Local | 78.18 | 89.18 | 41.54 |
| GraphRAG-Global | 55.27 | 66.67 | 47.22 |
| HippoRAG2 | 67.36 | 73.72 | 47.22 |
| **CausalRAG** | **78.00** | **92.86** | **49.46** |

### Ablation Study (Study on Parameters $k$ and $s$)

| k, s | Overall Performance |
|------|--------|
| k=1, s=1 | 0.534 |
| k=3, s=3 | 0.782 |
| k=5, s=5 | 0.824 |

### Key Findings
- **Causal path retrieval outperforms GraphRAG in both precision and recall**: Context Precision reached 92.86 (+3.68 vs GraphRAG-Local), and Context Recall reached 49.46 (+7.92 vs GraphRAG-Local).
- **The longer the document, the greater the advantage of CausalRAG**: From abstracts (72.43) to full texts (91.69), CausalRAG shows more significant improvements on long documents.
- **Causal reasoning reduces hallucinations**: Answer Faithfulness is 78.00, close to the best-performing GraphRAG-Local (78.18), but with higher context precision.

## Highlights & Insights
- **"Causality > Semantic Similarity" for retrieval**: This is the core insight. Many RAG failure cases occur because semantically related but causally irrelevant content is retrieved. CausalRAG effectively solves this problem via causal paths.
- **Training-free, plug-and-play solution**: Relying entirely on LLM capabilities, it requires no extra training and is easy to deploy.
- **Progressive analysis of $k$ and $s$**: Provides clear guidance on the tradeoff between retrieval depth/breadth and performance.

## Limitations & Future Work
- Reliance on internal LLM knowledge for causal identification may lead to inaccuracies in specialized domains (e.g., medical or legal).
- Causal path identification requires additional LLM calls, introducing computational overhead.
- The evaluation dataset is relatively small (QA on academic papers), lacking large-scale, multi-domain validation.
- The quality of causal relationship identification directly affects downstream performance, yet an evaluation of causal identification accuracy is lacking.

## Related Work & Insights
- **vs GraphRAG**: GraphRAG constructs general knowledge graphs, whereas CausalRAG focuses on causal subgraphs, making retrieval more targeted.
- **vs HippoRAG2**: HippoRAG2 simulates brain memory retrieval, while CausalRAG provides more structured reasoning through causal paths.
- **vs Standard RAG**: Semantic retrieval cannot handle causal reasoning demands, a gap that CausalRAG successfully fills.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of causal graphs and RAG is valuable, though the implementation relies heavily on LLM prompting.
- Experimental Thoroughness: ⭐⭐⭐ Small datasets and limited evaluation metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear elaboration of motivations and methodologies.
- Value: ⭐⭐⭐⭐ Provides a new direction for causal-aware retrieval in RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CaTs and DAGs: Integrating Directed Acyclic Graphs with Transformers for Causally Constrained Predictions](../../ICLR2026/causal_inference/cats_and_dags_integrating_directed_acyclic_graphs_with_transformers_for_causally.md)
- [\[ACL 2025\] FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation](fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example.md)
- [\[AAAI 2026\] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](../../AAAI2026/causal_inference/i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)
- [\[NeurIPS 2025\] Characterization and Learning of Causal Graphs from Hard Interventions](../../NeurIPS2025/causal_inference/characterization_and_learning_of_causal_graphs_from_hard_interventions.md)
- [\[ICLR 2026\] On the Identifiability of Causal Graphs with the Invariance Principle](../../ICLR2026/causal_inference/on_the_identifiability_of_causal_graphs_with_the_invariance_principle.md)

</div>

<!-- RELATED:END -->
