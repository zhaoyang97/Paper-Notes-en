---
title: >-
  [Paper Note] MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness
description: >-
  [ACL 2025][LLM (Other)][Generative Retrieval] Proposed the MindRef framework, which mimics the human two-stage memory pattern of first recalling document titles and then locating specific passages. Through Trie and FM-Index constrained decoding, it enables LLMs to independently recall reference passages without the need for additional retrieval models or pre-chunking.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Generative Retrieval"
  - "Parametric Knowledge"
  - "Hierarchical Recall"
  - "Constrained Decoding"
  - "FM-Index"
date: 2026-05-08
content_hash: 2c802e8b2b99599f
---

# MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness

**Conference**: ACL 2025  
**arXiv**: [2402.17010](https://arxiv.org/abs/2402.17010)  
**Area**: Knowledge-Intensive Retrieval / Generative Retrieval  
**Keywords**: Generative Retrieval, Parametric Knowledge, Hierarchical Recall, Constrained Decoding, FM-Index

## TL;DR

Proposed the MindRef framework, which mimics the human two-stage memory pattern of first recalling document titles and then locating specific passages. Through Trie and FM-Index constrained decoding, it enables LLMs to independently recall reference passages without the need for additional retrieval models or pre-chunking.

## Background & Motivation

Knowledge-intensive tasks (e.g., question answering, fact verification) heavily rely on retrieving relevant passages from large-scale knowledge sources. Limitations of existing retrieval methods include:

**Sparse retrieval** (e.g., BM25) lacks semantic depth.

**Dense retrieval** (e.g., DPR) suffers from limited interaction between query and passage representations.

**All existing methods rely on pre-chunking**: requiring documents to be split into fixed-length chunks in advance, which limits the flexibility of reference passages.

The authors raise a core problem: **Can LLMs bypass document pre-chunking and directly recall reference passages from arbitrary locations?** This idea stems from the human two-step pattern of recalling information—first recalling easily remembered documents (e.g., the title of a book or an article), and then locating the specific passage within that document.

## Method

### Overall Architecture

MindRef is a two-stage generative retrieval framework:

**Stage 1: Coarse-Grained Document Recall**
- Store all Wikipedia titles in a prefix tree (Trie).
- Guide the LLM to generate relevant document titles using prompts.
- Trie constraints ensure that the generated titles must correspond to existing Wikipedia pages.
- Use beam search (beam size=15) to generate and retrieve top-$k$ ($k=2$) documents.
- Scoring formula: $score_1(t|prompt_t(x)) = \frac{\sum_{i=1}^{l_t} \log p_\theta(y_i|y_{<i}, prompt_t(x))}{l_t}$

**Stage 2: Fine-Grained Passage Recall**
- Build an FM-Index (a space-efficient substring search data structure) for the top-$k$ documents.
- Guide the LLM to generate reference passages using prompts.
- FM-Index constraints ensure that the generated text must be a substring actually existing in the target document.
- Key advantage: Supports retrieval from arbitrary locations in the document, breaking the limitation of pre-chunking.
- The scoring formula is similar, and a weighted score is finally used to integrate both stages.

### Key Designs

**1. Trie-Constrained Decoding (Stage 1)**

The Trie data structure uses all Wikipedia titles as leaf nodes. During the autoregressive generation of the LLM, each step only allows selecting valid successor tokens for the current prefix in the Trie. This guarantees that:
- The generated titles must correspond to valid, existing Wikipedia pages.
- Hallucinated titles are avoided.

**2. FM-Index Constrained Decoding (Stage 2)**

FM-Index is a compressed index structure based on the Burrows-Wheeler Transform that supports efficient substring search. In the second stage, FM-Index dynamically provides the set of feasible successor tokens based on the generated token sequence, ensuring that the generated passage is a continuous text actually existing in the target document.

**3. Short Prefix Recall Location (SPRL)**

To address the inference speed issue of long passage generation, SPRL (Short Prefix Recall Location) is proposed:
- Allow the LLM to generate only a short prefix ($l_{p_s}=16$ tokens).
- Locate the position of the prefix in the candidate document using the KMP algorithm.
- Extract $l_p=150$ tokens backwards from the located position as the complete reference passage.
- Achieves approximately 4x inference speedup while maintaining 95%+ accuracy.

**4. Two-Stage Weighted Scoring**

$$score(p|x) = \alpha \cdot score_1(t|prompt_t(x)) + (1-\alpha) \cdot score_2(p|prompt_p(x))$$

where $\alpha=0.9$, emphasizing the importance of document title recall.

## Key Experimental Results

### Main Results

On 6 knowledge-intensive tasks of the KILT benchmark (using LLaMA-2-13b):

**Coarse-Grained Page Retrieval R-Precision:**

| Method | NQ | TriviaQA | HotpotQA | FEVER | WoW |
|------|-----|----------|----------|-------|------|
| BM25 | 26.33 | 31.78 | 41.30 | 52.09 | 28.78 |
| DPR* | 54.74 | 45.68 | 25.46 | 56.61 | 26.62 |
| **MindRef** | **57.77** | **54.41** | **48.70** | **83.69** | **57.63** |

**Fine-Grained Passage Retrieval Answer/Entity in Context:**

| Method | NQ | TriviaQA | HotpotQA | FEVER | WoW |
|------|-----|----------|----------|-------|------|
| BM25 | 23.65 | 58.87 | 29.45 | 58.33 | 50.36 |
| DPR* | 47.94 | 66.60 | 20.29 | 41.22 | 45.38 |
| **MindRef** | **40.82** | **68.20** | **30.04** | **58.42** | **63.43** |

**Downstream Task Performance (LLaMA-2-13b serving as the reader model):**

| Method | NQ (EM) | FEVER (ACC) |
|------|---------|-------------|
| No Retrieval | 19.74 | 73.23 |
| BM25 | 25.84 | 77.54 |
| DPR* | 33.49 | 75.27 |
| **MindRef** | ~30 | **78.79** |

### Key Findings

1. **Page-level retrieval significantly leads**: Achieves 83.69% R-Precision on FEVER, far surpassing DPR (56.61%) and BM25 (52.09%).
2. **Retrieval without pre-chunking**: MindRef directly and flexibly locates passages from the complete document, breaking the limitation of relying on chunk boundaries.
3. **SPRL acceleration is effective**: Locates accurately by generating only a 16-token prefix, achieving a 4x speedup while maintaining high accuracy.
4. **Good consistency across different LLM sizes**: Across 7B to 13B, the LLaMA and LLaMA-2 series demonstrate stable retrieval capabilities.
5. **Particularly strong performance on fact-checking and dialogue tasks**: The advantages on FEVER and WoW tasks are most significant.

## Highlights & Insights

1. **Cognitively-inspired framework design**: Formalizes the human memory pattern of "first recalling document titles, then locating specific passages" into a two-stage generation process.
2. **Creative application of constrained decoding**: The combination of Trie and FM-Index ensures that the generated content must originate from real documents, fundamentally avoiding hallucinations.
3. **Chunkless retrieval**: Completely gets rid of pre-chunking restrictions, supporting flexible retrieval starting from any position in the document.
4. **Engineering value of SPRL**: The hybrid strategy of short prefix generation and string-matching location balances generation quality and inference efficiency.
5. **Open-source availability**: The code has been open-sourced on GitHub.

## Limitations & Future Work

- Requires pre-building FM-Index for all documents, which incurs large storage overhead (at Wikipedia scale).
- The setting of $\alpha=0.9$ is highly biased towards the first stage, compressing the contribution of the second stage.
- Slightly underperforms compared to fully-finetuned DPR on open-domain QA tasks like NQ.
- Only evaluated on English Wikipedia; its applicability to multilingual or other knowledge sources remains unknown.
- Limited by the quality of LLM's parametric knowledge—if certain documents are underrepresented in the pre-training data, the recall capability will be restricted.
- The inference cost of beam search remains high (beam size 10-15).

## Related Work & Insights

- **BM25 / DPR / Contriever**: Traditional sparse and dense retrieval models, serving as primary baselines.
- **Generative Retrieval (DSI / GENRE)**: Achieves retrieval by generating document identifiers; MindRef extends this by introducing fine-grained passage location.
- **KILT Benchmark**: A unified evaluation platform for knowledge-intensive tasks, covering QA, fact-checking, and dialogue tasks.
- **FM-Index**: A classic index structure in information retrieval, creatively utilized in this paper to constrain LLM decoding.
- **RAG / REALM**: Retrieval-augmented generation methods, which require an external retriever, whereas MindRef enables the LLM to perform retrieval autonomously.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Novel approach, pushing generative retrieval to chunkless fine-grained passage location.
- **Practicality**: ⭐⭐⭐⭐ — Demonstrated effective on the KILT benchmark, though index construction and inference costs still need optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 6 tasks, including three levels of metrics: page-level, passage-level, and downstream tasks.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly readable with a clear cognitively-inspired narrative and sufficient technical detail.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](retrollm_empowering_large_language_models_to_retrieve_fine-grained_evidence_with.md)
- [\[ACL 2025\] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)
- [\[ACL 2025\] ChartLens: Fine-Grained Visual Attribution in Charts](chartlens_fine-grained_visual_attribution_in_charts.md)
- [\[ACL 2025\] BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models](behaviorbox_automated_discovery_of_fine-grained_performance_differences_between_.md)
- [\[ACL 2025\] Hierarchical Attention Generates Better Proofs](hierarchical_attention_generates_better_proofs.md)

</div>

<!-- RELATED:END -->
