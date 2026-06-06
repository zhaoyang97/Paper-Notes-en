---
title: >-
  [Paper Note] CRAFT: Training-Free Cascaded Retrieval for Tabular QA
description: >-
  [ACL 2026][Information Retrieval & RAG][Table Retrieval] This paper proposes CRAFT, a three-stage cascaded table retrieval framework that requires no dataset-specific training (SPLADE sparse filtering → semantic mini-tab…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Table Retrieval"
  - "Cascaded Retrieval"
  - "Zero-shot"
  - "Tabular QA"
  - "Training-free"
date: 2026-05-08
content_hash: 1e13f3649548bbe3
---

# CRAFT: Training-Free Cascaded Retrieval for Tabular QA

**Conference**: ACL 2026  
**arXiv**: [2505.14984](https://arxiv.org/abs/2505.14984)  
**Code**: [Project Page](https://coral-lab-asu.github.io/CRAFT/)  
**Area**: Information Retrieval / Tabular Question Answering  
**Keywords**: Table Retrieval, Cascaded Retrieval, Zero-shot, Tabular QA, Training-free

## TL;DR

This paper proposes CRAFT, a three-stage cascaded table retrieval framework that requires no dataset-specific training (SPLADE sparse filtering → semantic mini-table ranking → neural re-ranking). By enhancing table representations with Gemini-generated titles and descriptions, it achieves SOTA (R@1 49.84) on NQ-Tables and demonstrates strong zero-shot generalization on OTT-QA, while exhibiting significant robustness to query rephrasing.

## Background & Motivation

**Background**: Open-domain tabular question answering (TQA) requires first retrieving relevant tables from large-scale corpora before reasoning to derive answers. Existing methods include sparse retrieval (BM25), dense retrieval (DPR, DTR), and hybrid retrieval (THYME).

**Limitations of Prior Work**: (1) Dense retrieval models (DTR, DPR) are computationally expensive and require re-training or fine-tuning on new datasets, limiting adaptability; (2) Simple linearization of tables into text loses structural information of rows and columns; (3) Complex architectures (e.g., SSDR’s syntax-aware retriever) require sophisticated modeling and high training costs.

**Key Challenge**: The SOTA in table retrieval depends on expensive domain-specific fine-tuning, making systems inflexible when facing new domains or datasets. Can a carefully designed retrieval pipeline using pre-trained models achieve competitive performance?

**Goal**: Build a modular, scalable multi-stage retrieval framework that utilizes off-the-shelf pre-trained models to achieve competitive table retrieval and end-to-end QA performance in a zero-shot setting.

**Key Insight**: A three-stage cascaded design—progressively transitioning from high-recall sparse retrieval to high-precision semantic re-ranking, using stronger but slower models at each stage. Simultaneously, Gemini-generated table titles and descriptions are used to compensate for the semantic insufficiency of table representations.

**Core Idea**: Apply the "progressive refinement" concept of cascaded retrieval to table retrieval—efficient filtering with sparse models → reduced token overhead via mini-table construction → precise re-ranking with neural models, achieving SOTA without any training.

## Method

### Overall Architecture

Preprocessing (Gemini-1.5-Flash generates query sub-questions, table titles, and descriptions; Sentence Transformer ranks table rows by semantic relevance) → Stage 1 (SPLADE sparse retrieval, filtering Top-5000 from 169K/419K tables) → Stage 2 (Construct mini-tables with headers + top 5 rows, semantic matching via Sentence Transformer/Jina to get Top-K) → Stage 3 (Re-ranking with OpenAI/Gemini embeddings for final results) → End-to-end LLM answer generation.

### Key Designs

1.  **Three-stage cascaded retrieval**:
    - **Function**: Progressively transitions from high recall to high precision, balancing efficiency and effectiveness.
    - **Mechanism**: Stage 1 uses SPLADE (sparse lexical expansion) to efficiently scan all tables (utilizing titles, headers, cell values, and descriptions) and filter to 5,000 candidates. Stage 2 constructs mini-tables (headers + top 5 rows) and uses bi-encoder semantic matching to narrow down to Top-K. Stage 3 uses the strongest embedding models (text-embedding-3-large or gemini-embedding-001) for final re-ranking.
    - **Design Motivation**: Running semantic models on the full table corpus is computationally prohibitive; the cascaded design balances precision and efficiency at each stage.

2.  **Mini-table construction and table enhancement**:
    - **Function**: Reduces token overhead while preserving critical table information.
    - **Mechanism**: Each table retains only the headers and the top 5 most relevant rows (selected by Sentence Transformer based on semantic relevance) to form a mini-table. Additionally, Gemini-1.5-Flash generates descriptive titles and detailed descriptions for each table to enhance semantic matching.
    - **Design Motivation**: Mini-tables achieve up to 33× fewer online embedding calls and 70% shorter context without sacrificing retrieval precision.

3.  **Dataset-specific model selection**:
    - **Function**: Selects optimal pre-trained models based on dataset characteristics.
    - **Mechanism**: NQ-Tables (single-hop factual queries) uses all-mpnet-base-v2 + text-embedding-3-large; OTT-QA (multi-hop reasoning, hybrid text modes) uses Jina Embeddings v3 + gemini-embedding-001. Selection is based on model suitability for specific text types.
    - **Design Motivation**: Query and table characteristics vary across datasets, requiring model selection to match data features.

### Loss & Training

This paper involves no training. All models use pre-trained weights or APIs. End-to-end QA uses Llama3-8B, Qwen2.5-7B, and Mistral-7B to generate answers in a zero-shot or few-shot manner.

## Key Experimental Results

### Main Results

**NQ-Tables Retrieval Performance**

| Model | Training Required | R@1 | R@10 | R@50 |
| :--- | :--- | :--- | :--- | :--- |
| THYME (SOTA Hybrid) | Yes | 48.55 | 86.38 | 96.08 |
| DTR+HN | Yes | 47.33 | 80.96 | 91.51 |
| BIBERT+SPLADE | Yes | 45.62 | 86.72 | 95.62 |
| **CRAFT (Zero-shot)** | **None** | **49.84** | **86.83** | **97.17** |

**OTT-QA Zero-shot Retrieval Performance**

| Model | R@1 | R@10 | R@50 |
| :--- | :--- | :--- | :--- |
| THYME (Fine-tuned) | 66.67 | 91.10 | 96.16 |
| **CRAFT (Zero-shot)** | 55.56 | 89.88 | 96.07 |

### Ablation Study

**Query Robustness (Performance change Δ under rephrased queries)**

| Model | Original R@10 | Rephrased Δ(avg) |
| :--- | :--- | :--- |
| DTR (M) | 75.73 | -8.38 |
| DTR (S) | 73.88 | -11.82 |
| DTR (M)+HN | 80.96 | -5.80 |
| **CRAFT** | 87.16 | **-0.04** |

### Key Findings

- CRAFT surpasses all fine-tuned methods on NQ-Tables in a zero-shot setting (R@1 49.84 vs THYME 48.55), proving that a well-designed cascaded pipeline can replace expensive fine-tuning.
- On OTT-QA, CRAFT's zero-shot R@50 (96.07) is close to the fine-tuned SOTA (96.16), with a gap of only 0.09.
- CRAFT is nearly immune to query rephrasing (Δ=-0.04), while the fine-tuned DTR model suffers an 8-12 point performance drop, indicating significantly stronger generalization.
- Each stage in the cascaded design contributes: Stage 1→2 improves R@10 by approximately 10-21 points, and Stage 2→3 provides a further 5-8 point boost.
- Mini-table design reduces embedding calls by 33× without loss of precision.

## Highlights & Insights

- "Engineering wisdom" via cascaded retrieval and table enhancement defeated fine-tuning methods, suggesting that the general capabilities of pre-trained models are undervalued.
- Extreme robustness to query rephrasing (Δ=-0.04) is a highly practical feature, whereas fine-tuned models are fragile in this regard.
- Mini-table construction is a simple yet effective efficiency optimization; a 70% reduction in context is significant for real-world deployment.

## Limitations & Future Work

- Dependence on commercial APIs (Gemini, OpenAI embeddings) limits cost efficiency and reproducibility.
- Model selection (different models for NQ-Tables vs OTT-QA) introduces dataset-specific engineering choices.
- Performance on non-English tables or tables with complex formatting (e.g., merged cells) has not been evaluated.
- Preprocessing (generating titles/descriptions) requires additional offline LLM calls.

## Related Work & Insights

- **vs THYME**: THYME requires fine-tuning on target datasets and designs field-aware matching; CRAFT requires no training but achieves similar or better performance through cascading.
- **vs DTR**: DTR is a classic dense retriever but is sensitive to query rephrasing; CRAFT's cascaded design is naturally more robust.
- **vs T-RAG**: T-RAG combines retrieval and generation end-to-end; CRAFT remains modular, allowing for easy component replacement.

## Rating

- Novelty: ⭐⭐⭐ The combination of cascaded retrieval and table enhancement is effective but not a entirely new concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across two datasets, robustness tests, stage-wise ablation, and end-to-end QA.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions and detailed experimental analysis.
- Value: ⭐⭐⭐⭐ Demonstrates that training-free retrieval can reach SOTA, offering direct value for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Test-Time Training for Zero-Resource Dense Retrieval Reranking](test-time_training_for_zero-resource_dense_retrieval_reranking.md)
- [\[ACL 2026\] S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA](s2g-rag_structured_sufficiency_and_gap_judging_for_iterative_retrieval-augmented.md)
- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](../../ICLR2026/information_retrieval/q_rag_long_context_multi_step_retrieval.md)
- [\[ACL 2026\] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)
- [\[ICML 2026\] Ranking-Free RAG: Replacing Re-Ranking with Selection in RAG for Sensitive Domains](../../ICML2026/information_retrieval/ranking_free_rag_replacing_re-ranking_with_selection_in_rag_for_sensitive_domain.md)

</div>

<!-- RELATED:END -->
