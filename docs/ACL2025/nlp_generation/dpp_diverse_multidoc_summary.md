---
title: >-
  [Paper Note] Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries
description: >-
  [ACL 2025][Text Generation][multi-document summarization] Proposes decoupling multi-document summarization into a three-step pipeline: key point extraction $\to$ DPP diversity selection $\to$ rewriting. By using Determinantal Point Processes (DPP) for principled content selection, the method significantly improves the source document coverage of LLMs in multi-document summarization.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "multi-document summarization"
  - "DPP"
  - "content selection"
  - "source coverage"
  - "personalized summary"
date: 2026-05-08
content_hash: 45f12200ee90c329
---

# Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries

**Conference**: ACL 2025  
**arXiv**: [2505.21859](https://arxiv.org/abs/2505.21859)  
**Code**: Not publicly available  
**Area**: Others  
**Keywords**: multi-document summarization, DPP, content selection, source coverage, personalized summary  

## TL;DR

Proposes decoupling multi-document summarization into a three-step pipeline: key point extraction $\to$ DPP diversity selection $\to$ rewriting. By using Determinantal Point Processes (DPP) for principled content selection, the method significantly improves the source document coverage of LLMs in multi-document summarization.

## Background & Motivation

- **Problem Definition**: Multi-Document Diverse Summarization (MDDS) requires generating a summary that covers diverse perspectives from multiple articles reporting on the same news event. Existing LLMs suffer from insufficient coverage due to the "lost in the middle" attention bias.
- **Limitations of Prior Work**: Directly concatenating all documents and prompting LLMs to generate summaries in a single step couples content selection with text generation. Consequently, the positional bias of LLMs leads to over-focusing on documents at the beginning or end.
- **Core Idea**: LLMs are not adept at content selection. Decoupling it from text generation and replacing LLMs with a principled method (DPP) for content selection can enhance coverage.
- **Goal**: Generate personalized summaries by integrating user interest relevance into the DPP kernel matrix.

## Method

### Overall Architecture

LLM + DPP three-step pipeline: (1) extracting atomic key points from each source document using an LLM; (2) selecting a diverse subset from all key points using DPP; (3) rewriting the selected key points into a coherent summary using an LLM.

### Key Designs

- **Key Point Extraction**: Using zero-shot prompting, the LLM decomposes each document $d_i$ into a set of atomic key points $K_i$. Each key point captures an independent informational unit, ensuring a sufficiently fine granularity of information.
- **DPP Diversity Selection**: Key points are encoded into vectors using a Transformer encoder (DeBERTa-V3 BERTScore embeddings) to construct a Gaussian kernel matrix $L$ that measures similarities between key points. DPP inference (exact sampling via spectral methods) is used to select the most diverse subset $K_{sel}$, where the size of the selected subset is automatically determined by the eigenvalues of the kernel matrix.
- **Relevance-Weighted DPP**: For query-focused tasks, e5-mistral-7b-instruct is used to compute the relevance score $R_i$ of each key point with respect to user intent. A new kernel matrix $L' = RLR^T$ is constructed to balance diversity and relevance.

### Loss & Training

Without requiring training, the DPP selection is based on combinatorial optimization of the kernel matrix (specifically, greedy approximation), and all LLM steps utilize zero-shot prompting. For evaluation, an LLM-as-a-judge (GPT-4o) determines whether the summary can correctly answer questions related to the source documents. Human verification shows agreement rates with the LLM judge of 86.4% (answerability) and 95.3% (correctness).

## Experiments

### Main Results

| Method | DiverseSumm Coverage | | | | DiverseSumm Augmented Coverage | | | |
|------|-------|-------|--------|--------|-------|-------|--------|--------|
| | GPT 3.5 | GPT 4o | Claude | Llama | GPT 3.5 | GPT 4o | Claude | Llama |
| Naive LLM | 0.332 | 0.552 | 0.478 | 0.243 | 0.267 | 0.481 | 0.425 | 0.219 |
| All KPs | 0.347 | 0.544 | 0.568 | 0.346 | 0.257 | 0.462 | 0.411 | 0.237 |
| LLM-Selected KPs | 0.437 | 0.575 | 0.537 | 0.338 | 0.385 | 0.541 | 0.514 | 0.309 |
| **LLM + DPP** | **0.471** | **0.581** | **0.592** | **0.365** | **0.385** | **0.554** | **0.547** | **0.323** |

### Ablation Study (DPP Kernel Selection)

| Kernel Function | GPT 3.5 | GPT 4o | Claude |
|--------|---------|--------|--------|
| Gaussian σ=0.1 | 0.449 | **0.615** | **0.635** |
| Gaussian σ=1 | **0.471** | 0.581 | 0.592 |
| Gaussian σ=10 | 0.434 | 0.591 | 0.520 |
| Linear | 0.465 | 0.589 | 0.586 |

### Key Findings

- LLM + DPP consistently achieves the highest coverage across all four LLMs, demonstrating the effectiveness of DPP-based content selection.
- Explicit key point selection (LLM-Selected KPs and LLM + DPP) generally outperforms using all key points (All KPs), indicating that simply shortening the context is insufficient and active selection is necessary.
- Key points selected by DPP cover more source documents (with a more uniform distribution) than those selected by LLMs.
- LLM + DPP effectively mitigates positional bias: the recency bias (tail bias) of Llama and primacy bias (head bias) of GPT-4o are both significantly reduced.
- The improvement in coverage is not driven by longer summaries, as there are no significant differences in average summary length across methods.

## Highlights & Insights

- Elegantly combines a principled statistical method (DPP) with LLM prompting pipelines, demonstrating that not all pipeline components need to be resolved by LLMs.
- Clearly reveals the positional bias patterns (primacy/recency/middle bias) of different LLMs in multi-document scenarios and their impact on coverage.
- The method is simple, modular, and plug-and-play, making it applicable to various LLM backends without additional training.
- Ensures evaluation reliability through synthetic question expansion and human verification.

## Limitations & Future Work

- Evaluation is restricted to the news-domain DiverseSumm benchmark; generalization to other domains (e.g., science, medical) remains unverified.
- The DPP kernel function and parameter (σ) require manual tuning, and optimal configurations may vary across different LLMs.
- Relevance evaluation for query-focused tasks relies on an external retrieval model (e5-mistral-7b), which increases system complexity and inference costs.
- The quality of key point extraction depends heavily on the capability of the LLM itself; weaker models may generate low-quality key points.
- Comparison against recent long-context optimization methods (such as RAG and context compression) is lacking.

## Related Work & Insights

- Multi-document summarization: DiverseSumm benchmark (Huang et al. 2024), hierarchical summarization (Chang et al. 2024)
- LLM attention bias: "Lost in the middle" (Liu et al. 2024)
- DPP applications in NLP: Document summarization (Kulesza et al. 2012), recommender systems diversity (Chen et al. 2018)
- Atomic claim decomposition: Kim et al. 2024, Krishna et al. 2023, Padmakumar & He
- Query-focused summarization: Daumé III & Marcu 2006, Vig et al. 2022
- LLM-as-Judge evaluation: Li et al. 2024, Balepur et al. 2024

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

> **Note**: This paper conducts experiments on 245 news events using four LLMs: GPT-3.5, GPT-4o, Claude-3-Sonnet, and LLaMA-3.1. The results display strong consistency and persuasiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs](event_graph_bias_mitigation_summarization.md)
- [\[ACL 2025\] Personalized Text Generation with Contrastive Activation Steering](personalized_text_generation_with_contrastive_activation_steering.md)
- [\[ACL 2026\] SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization](../../ACL2026/nlp_generation/scurank_ranking_multiple_candidate_summaries_with_summary_content_units_for_enha.md)
- [\[ACL 2025\] Context-Aware Hierarchical Merging for Long Document Summarization](context-aware_hierarchical_merging_for_long_document_summarization.md)
- [\[ACL 2025\] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](doc_level_mbr_optimal_transport.md)

</div>

<!-- RELATED:END -->
