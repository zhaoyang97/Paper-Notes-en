---
title: >-
  [Paper Note] Context-Aware Hierarchical Merging for Long Document Summarization
description: >-
  [ACL 2025][Text Generation][Long Document Summarization] This work proposes Context-Aware Hierarchical Merging (CAHM), which effectively mitigates LLM hallucinations during ultra-long document (>100K tokens) summarization by incorporating relevant source document context (via extractive, retrieval, or citation methods) into the hierarchical merging process.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Long Document Summarization"
  - "Hierarchical Merging"
  - "Context Enhancement"
  - "Hallucination Mitigation"
  - "Faithfulness"
date: 2026-05-08
content_hash: c5d826806dc077f3
---

# Context-Aware Hierarchical Merging for Long Document Summarization

**Conference**: ACL 2025  
**arXiv**: [2502.00977](https://arxiv.org/abs/2502.00977)  
**Code**: [https://github.com/Leonard907/CAHM](https://github.com/Leonard907/CAHM)  
**Area**: Text Generation  
**Keywords**: Long Document Summarization, Hierarchical Merging, Context Enhancement, Hallucination Mitigation, Faithfulness

## TL;DR

This work proposes Context-Aware Hierarchical Merging (CAHM), which effectively mitigates LLM hallucinations during ultra-long document (>100K tokens) summarization by incorporating relevant source document context (via extractive, retrieval, or citation methods) into the hierarchical merging process.

## Background & Motivation

**Background**: Summarization of ultra-long documents (such as legal documents, novels, etc., usually exceeding 100K tokens) is a vital task in NLP. Hierarchical Merging is a mainstream approach to processing ultra-long inputs—chunking the document, summarizing each chunk, and recursively merging them.

**Limitations of Prior Work**: Hierarchical merging **amplifies LLM hallucinations** during the recursive merging process. This occurs because intermediate summaries may contain factual errors, which are repeatedly referenced and reinforced in subsequent merges, leading to a decline in the faithfulness of the final summary.

**Key Challenge**: While hierarchical merging handles ultra-long inputs well, its connection to the source document gradually breaks during recursion, reducing faithfulness. How can one maintain factual consistency with the source document while preserving scalability?

**Goal**: To introduce source document context during the intermediate stages of hierarchical merging to ground intermediate summaries on stronger factual evidence, thereby improving the faithfulness of the final summary.

**Key Insight**: Drawing inspiration from RAG (Retrieval-Augmented Generation), this work designs an "Incorporate Context" (IC) module to explore combinations of three context acquisition methods and two context utilization strategies.

**Core Idea**: Incorporate relevant source document context at each level of hierarchical merging, enhancing the faithfulness of intermediate summaries through "Replace" or "Support" strategies.

## Method

### Overall Architecture

Building upon standard hierarchical merging, an **IC (Incorporate Context) module** is added to each layer to obtain both the abstractive summary and the relevant source document context. When generating the next layer's summary, the model can:
- **Support**: Input the context along with the previous level's summary as supporting evidence.
- **Replace**: Directly replace the previous level's abstractive summary with the retrieved context.

### Key Designs

1. **Extract (Extractive Summarization)**: MemSum (an extractive summarizer trained via reinforcement learning) is used to select key sentences from source document chunks. Extractive summaries naturally select sentences covering the same key information as the abstractive summaries, making them suitable as replacement or supporting contexts. Fine-tuning is performed on Multi-LexSum and BookSum, respectively.

2. **Retrieve (Information Retrieval)**: Using the intermediate abstractive summary as a query, relevant paragraphs (roughly 100 words each) are retrieved from source document chunks via BM25. This utilizes the LLM's distillation ability to generate concise queries, and retrieves information to enhance factual accuracy.

3. **Cite (Citation Generation)**: LLMs are instructed to cite source document paragraph numbers (e.g., [1], [2]) when generating intermediate summaries. Top-$k$ paragraphs are then selected as relevant context based on citation frequency. No additional retrieval or extraction steps are required.

4. **Support vs Replace**:
    - **Support**: Retains the abstractive summary and appends the context as evidence $\rightarrow$ high information density, broad coverage.
    - **Replace**: Directly replaces the abstractive summary with source document context $\rightarrow$ highest faithfulness but may lose the global perspective.

### Loss & Training

- All hierarchical merges use zero-shot prompting and **require no additional training** (except for MemSum fine-tuning).
- Chunk size and maximum merged context length are both set to 8K tokens.
- Llama-3.1 8B and 70B (GPTQ-INT4 quantized) are utilized with a 128K context length.

## Key Experimental Results

### Main Results (Llama-3.1-70B, Multi-LexSum)

| Method | ROUGE | BERTScore | SummaC | AlignScore | PRisma |
|------|-------|-----------|--------|------------|--------|
| Zero-shot | 23.6 | 60.7 | 43.5 | 77.6 | 41.5 |
| HMerge | 26.7 | 64.3 | 43.4 | 76.3 | 48.2 |
| Extract-Support | **27.6** | 64.1 | 43.2 | 79.0 | **49.7** |
| Retrieve-Support | 26.6 | **66.1** | 44.5 | 78.8 | 49.8 |
| Cite-Replace | 22.5 | 61.9 | **51.6** | **85.8** | 40.6 |
| Retrieve-Replace | 24.7 | 62.1 | 47.9 | 80.3 | 43.8 |

### Ablation Study (8B, Replace with increased context length, Retrieve-Replace, PRisma)

| Context Length | Multi-LexSum | SuperSummary |
|-----------|-------------|-------------|
| 8K | 41.8 | 23.3 |
| 16K | 43.7 (+1.9) | 26.5 (+3.2) |
| 32K | 44.1 (+2.3) | 27.9 (+4.6) |

Even when expanded to 32K, Replace is still inferior to Retrieve-Support (46.1/38.4), indicating that abstractive summaries are indispensable.

### Human Evaluation (70B, SuperSummary of a book)

| Method | Correct | Incorrect | Not Present |
|------|---------|-----------|-------------|
| Extract-Support | **72.7%** | 18.2% | 9.1% |
| HMerge | 59.1% | 27.3% | 13.6% |
| Zero-shot | 60.0% | 20.0% | 20.0% |
| Extract-Replace | 48.8% | 23.3% | 27.9% |

### Key Findings

- **Extract-Support achieves the best overall performance**: It achieves the highest average rank across all metrics, with its "Correct" ratio in human evaluation exceeding HMerge by 13.6 percentage points.
- **Replace yields substantial improvements on input-based metrics**: The AlignScore of Cite-Replace on SuperSummary is about 10 points higher than the baseline, as it directly uses segments from the source document.
- **Support is significantly better on reference-based metrics**: Abstractive summaries ensure information coverage, preventing over-focus on local details.
- **The Cite method performs the weakest**: Accurate citation generation remains difficult for LLMs, as it requires understanding and executing complex instructions.
- **Replace is prone to "drifting"**: Manual inspection reveals that Replace summaries tend to emphasize marginal details (dialogue, scene descriptions) rather than key events.
- **70B vs 8B**: 70B shows moderate improvements on reference-based metrics but minimal gains on input-based metrics.
- **Book summarization is more challenging than legal summarization**: Scores of all methods on SuperSummary are consistently lower than those on Multi-LexSum.

## Highlights & Insights

- Elegantly embeds RAG concepts into the hierarchical merging framework with a clear design space: 3 context acquisition methods $\times$ 2 utilization strategies = 6 variants.
- Human evaluation reveals biases in automatic metrics: input-based metrics favor 'Replace', whereas reference-based metrics favor 'Support'. In practice, 'Support' yields superior faithfulness.
- Core insight: **The information density of abstractive summaries is irreplaceable**—at the same length, abstractive summaries cover more critical events, whereas source document segments over-expand on single events.
- The lightweight combination of MemSum + BM25 brings consistent improvements without increasing training costs.

## Limitations & Future Work

1. **Limited scale of human evaluation**: Detailed annotations were done for only one book, costing USD 200–250/book + 10 hours/person.
2. **High inference cost of Support**: Supporting contexts make inputs longer in the merging phase. Future work could select contexts on demand.
3. **Restricted dataset domains**: The datasets are limited to legal and narrative domains. High-quality summarization datasets with >100K tokens remain scarce.
4. **Unreleased pointer of the Cite method**: Citation accuracy is a bottleneck, which could be improved through post-processing or specialized training.

## Related Work & Insights

- **Hierarchical Merging**: First proposed by Wu et al. 2021, and updated to zero-shot prompting by Chang et al. 2024. This work introduces context on top of these baselines.
- **RAG**: Retrieval-Augmented Generation has proven effective in QA. This work is the first to systematically apply it to ultra-long document summarization.
- **Insights**: Adaptive decision-making could be explored—dynamically deciding whether to invoke context enhancement based on intermediate summary quality; it could also combine with GraphRAG to handle document structure information.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Integrating RAG into hierarchical merging is a natural but important-innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covered two datasets, two model scales, six variants, and human evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, in-depth analysis, and intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ — Provides a practical and reproducible solution for ultra-long document summarization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs](event_graph_bias_mitigation_summarization.md)
- [\[ACL 2025\] Decomposed Opinion Summarization with Verified Aspect-Aware Modules](decomposed_opinion_summarization_with_verified_aspect-aware_modules.md)
- [\[ACL 2026\] XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](../../ACL2026/nlp_generation/xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)
- [\[ACL 2025\] Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries](dpp_diverse_multidoc_summary.md)
- [\[ACL 2025\] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](doc_level_mbr_optimal_transport.md)

</div>

<!-- RELATED:END -->
