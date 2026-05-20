---
title: >-
  [Paper Note] Summaries as Centroids for Interpretable and Scalable Text Clustering
description: >-
  [ICLR 2026][Information Retrieval & RAG][Text Clustering] This paper proposes k-NLPmeans and k-LLMmeans, which periodically replace numeric centroids with textual summaries (summary-as-centroid) during k-means iterations…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "Text Clustering"
  - "k-means"
  - "Summary-as-Centroid"
  - "Interpretability"
  - "Streaming Clustering"
  - "LLM-optional"
date: 2026-05-08
content_hash: 614e6428da59611a
---

# Summaries as Centroids for Interpretable and Scalable Text Clustering

**Conference**: ICLR 2026
**arXiv**: [2502.09667](https://arxiv.org/abs/2502.09667)  
**Code**: None  
**Area**: Information Retrieval
**Keywords**: Text Clustering, k-means, Summary-as-Centroid, Interpretability, Streaming Clustering, LLM-optional

## TL;DR

This paper proposes k-NLPmeans and k-LLMmeans, which periodically replace numeric centroids with textual summaries (summary-as-centroid) during k-means iterations, achieving interpretable cluster prototypes while preserving the standard k-means objective. The number of LLM calls is independent of dataset size.

## Background & Motivation

- **Limitations of standard k-means on text**: numeric averaging blurs textual semantics, and centroids are not human-interpretable.
- **Problems with existing LLM-based clustering methods**:
  1. **Poor scalability**: the number of LLM calls grows with dataset size.
  2. **Opaque optimization**: relies on prompts, greedy merging, and similarity thresholds without a clear objective function.
- A clustering method that is both interpretable and scalable is needed.

## Method

### Core Idea: Summary-as-Centroid

Within the standard k-means loop, numeric centroids are replaced by textual summaries every $l$ iterations:

$$\boldsymbol{\mu}_j = \text{Embedding}(f_{\text{summarizer}}(C_j))$$

In the remaining iterations, the standard mean update is used: $\boldsymbol{\mu}_j = \frac{1}{|C_j|}\sum_{i \in [C_j]} \mathbf{x}_i$

### k-NLPmeans (LLM-free Variant)

Uses classical NLP summarization methods as $f_{\text{NLP}}^{(q)}$:

- **Centroid-based**: Computes the centroid of intra-cluster sentence embeddings and concatenates the top-$q$ most similar sentences.
- **TextRank**: Constructs a sentence similarity graph, ranks sentences via PageRank, and selects the top-$q$ sentences.
- **LSA-style SVD**: Applies SVD to sentence embeddings, scores sentences by principal component contribution, and selects accordingly.

Characteristics: fast, deterministic, LLM-free, and offline-capable.

### k-LLMmeans (LLM-assisted Variant)

$$\boldsymbol{\mu}_j = \text{Embedding}(f_{\text{LLM}}(p_j))$$

where $p_j = \text{Prompt}(I, \{d_{z_i} | z_i \sim [C_j]\}_{i=1}^{m_j})$

- The LLM processes **representative samples** of each cluster (sampled via k-means++) rather than all documents.
- Each summarization step requires exactly $k$ LLM calls → **call count is independent of dataset size**.

### Mini-batch Extension: Streaming Clustering

Summarization steps are inserted into the mini-batch k-means update rule:
- Batches $D_1, \ldots, D_b$ are processed sequentially.
- Each batch is handled by k-NLPmeans/k-LLMmeans, followed by incremental centroid updates.
- The low-memory property of mini-batch k-means is preserved.

### Loss & Training

The standard k-means objective remains unchanged between summarization steps:

$$\min_{C_1, \ldots, C_k} \sum_{j=1}^k \sum_{i \in [C_j]} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$$

When summarization fails, the method gracefully degrades to standard k-means.

## Key Experimental Results

### Static Clustering (text-embedding-3-small)

| Method | Bank77 ACC | CLINC ACC | GoEmo ACC | MASSIVE(D) ACC | MASSIVE(I) ACC |
|--------|-----------|----------|----------|---------------|---------------|
| k-means | ~65 | ~77 | ~20 | ~59 | ~52 |
| k-NLPmeans LSA-mult | **67.1** | **80.2** | **22.3** | **63.3** | **55.3** |
| k-LLMmeans single | 67.1 | 78.1 | **24.0** | — | — |
| k-LLMmeans mult | Higher | Higher | Higher | Higher | Higher |

### LLM Call Efficiency Comparison

| Method | LLM Call Complexity | Data Dependence |
|--------|-------------------|----------------|
| ClusterLLM | O(n) | Grows with data |
| LLMEdgeRefine | O(n) | Grows with data |
| k-NLPmeans | **O(0)** | **Zero LLM calls** |
| k-LLMmeans | **O(k · #summarization steps)** | **Independent of n** |

### Key Findings

1. Even a **single summarization step** ($l=60$) yields substantial improvements over standard k-means.
2. k-NLPmeans (zero LLM) approaches or matches k-LLMmeans on most benchmarks.
3. k-means++ sampled documents yield better LLM summaries than random sampling.
4. Consistent improvements are observed across 4 embedding models, 5 LLMs, and 3 classical NLP methods.
5. The proposed methods also outperform standard mini-batch k-means in streaming clustering scenarios.

## Highlights & Insights

- **Minimal modification, significant gain**: only the centroid update step of k-means is modified; everything else remains unchanged.
- **LLM-optional design**: k-NLPmeans captures most of the benefit without any LLM dependency.
- **Intrinsic interpretability**: each centroid is a human-readable textual summary.
- **Graceful degradation**: automatically falls back to standard k-means when summary quality is poor, ensuring no regression.
- **Fixed LLM budget**: total LLM calls are $k \times$ the number of summarization steps, making the approach tractable at scale.
- **Introduces a new StackExchange streaming clustering benchmark.**

## Limitations & Future Work

- Summarization quality is bounded by the capability of the underlying summarizer.
- For semantically overlapping clusters, summaries may fail to provide effective differentiation.
- The number of clusters $k$ must be specified in advance (inherited from k-means).
- The summarization frequency $l$ requires tuning, though experiments suggest low sensitivity to this hyperparameter.

## Related Work & Insights

- LLM-based clustering: ClusterLLM, IDAS, LLMEdgeRefine, etc.
- Classical text clustering: k-medoids, spectral clustering, BERTopic.
- Streaming clustering: mini-batch k-means, LLM-based online methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The summary-as-centroid concept is concise and novel.
- **Technical Depth**: ⭐⭐⭐ — The method is intuitively clear, but theoretical analysis is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely comprehensive: 4 datasets × 4 embeddings × 5 LLMs × 3 NLP methods.
- **Practicality**: ⭐⭐⭐⭐⭐ — Plug-and-play, interpretable, and scalable; high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hierarchical Concept-based Interpretable Models](hierarchical_concept-based_interpretable_models.md)
- [\[ICLR 2026\] Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning](hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning.md)
- [\[ICLR 2026\] LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference](lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que.md)
- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [\[AAAI 2026\] Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score](../../AAAI2026/information_retrieval/beyond_perplexity_let_the_reader_select_retrieval_summaries_via_spectrum_project.md)

</div>

<!-- RELATED:END -->
