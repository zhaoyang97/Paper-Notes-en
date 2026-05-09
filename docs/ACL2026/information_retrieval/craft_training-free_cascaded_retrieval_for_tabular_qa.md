---
title: >-
  [Paper Note] CRAFT: Training-Free Cascaded Retrieval for Tabular QA
description: >-
  [ACL 2026][Table Retrieval] This paper proposes CRAFT, a three-stage cascaded table retrieval framework requiring no dataset-specific training (SPLADE sparse filtering → semantic mini-table ranking → neural re-ranking). By augmenting table representations with Gemini-generated captions and descriptions, CRAFT achieves SOTA on NQ-Tables (R@1 49.84), demonstrates strong zero-shot generalization on OTT-QA, and exhibits remarkable robustness to query paraphrasing.
tags:
  - ACL 2026
  - Table Retrieval
  - Cascaded Retrieval
  - Zero-Shot
  - Tabular QA
  - Training-Free
date: 2026-05-08
content_hash: bcef5401a964348f
---

# CRAFT: Training-Free Cascaded Retrieval for Tabular QA

**Conference**: ACL 2026
**arXiv**: [2505.14984](https://arxiv.org/abs/2505.14984)
**Code**: [Project Page](https://coral-lab-asu.github.io/CRAFT/)
**Area**: Information Retrieval / Tabular Question Answering
**Keywords**: Table Retrieval, Cascaded Retrieval, Zero-Shot, Tabular QA, Training-Free

## TL;DR

This paper proposes CRAFT, a three-stage cascaded table retrieval framework requiring no dataset-specific training (SPLADE sparse filtering → semantic mini-table ranking → neural re-ranking). By augmenting table representations with Gemini-generated captions and descriptions, CRAFT achieves SOTA on NQ-Tables (R@1 49.84), demonstrates strong zero-shot generalization on OTT-QA, and exhibits remarkable robustness to query paraphrasing.

## Background & Motivation

**State of the Field**: Open-domain tabular question answering (TQA) requires first retrieving relevant tables from large-scale corpora, then reasoning over them to produce answers. Existing approaches include sparse retrieval (BM25), dense retrieval (DPR, DTR), and hybrid retrieval (THYME).

**Limitations of Prior Work**: (1) Dense retrieval models (DTR, DPR) incur high computational costs and require retraining or fine-tuning on new datasets, limiting adaptability to new domains; (2) naively linearizing tables into text loses row-column structural information; (3) complex architectures (e.g., SSDR's syntax-aware retrievers) demand elaborate modeling and expensive training.

**Root Cause**: SOTA table retrieval relies on costly domain-specific fine-tuning, which renders systems inflexible when facing new domains or datasets. The question is whether pretrained models, combined with a carefully designed retrieval pipeline, can achieve competitive performance.

**Paper Goals**: To construct a modular, extensible multi-stage retrieval framework that leverages off-the-shelf pretrained models to achieve competitive table retrieval and end-to-end QA performance in a zero-shot setting.

**Starting Point**: A three-stage cascaded design that progressively transitions from high-recall sparse retrieval to high-precision semantic re-ranking, employing progressively stronger but slower models at each stage. Gemini-generated table captions and descriptions are used to compensate for semantic deficiencies in raw table representations.

**Core Idea**: Applying the "progressive refinement" paradigm of cascaded retrieval to table retrieval — sparse models efficiently filter candidates → mini-table construction reduces token overhead → neural models perform precise re-ranking — achieving SOTA without any training.

## Method

### Overall Architecture

Preprocessing (Gemini-1.5-Flash generates query sub-questions, table captions, and descriptions; Sentence Transformer ranks table rows by semantic relevance) → Stage 1 (SPLADE sparse retrieval, filtering Top-5000 from 169K/419K tables) → Stage 2 (construct mini-tables comprising column headers + top-5 rows, semantic matching via Sentence Transformer/Jina to obtain Top-K) → Stage 3 (re-rank with OpenAI/Gemini embeddings for final results) → end-to-end LLM answer generation.

### Key Designs

1. **Three-Stage Cascaded Retrieval**:

    - Function: Progressively transitions from high recall to high precision, balancing efficiency and effectiveness.
    - Mechanism: Stage 1 uses SPLADE (sparse lexical expansion) to efficiently scan the full table corpus (leveraging captions, column headers, cell values, and descriptions), filtering to 5,000 candidates. Stage 2 constructs mini-tables (column headers + top-5 rows) and applies bi-encoder semantic matching to narrow down to Top-K. Stage 3 applies the strongest embedding model (text-embedding-3-large or gemini-embedding-001) for final re-ranking.
    - Design Motivation: Running semantic models over the full table corpus is computationally prohibitive; the cascaded design balances precision and efficiency at each stage.

2. **Mini-Table Construction and Table Augmentation**:

    - Function: Reduces token overhead while retaining critical table information.
    - Mechanism: Each table retains only its column headers and the most relevant top-5 rows (selected by semantic relevance ranking via Sentence Transformer), forming a mini-table. Gemini-1.5-Flash additionally generates descriptive captions and detailed descriptions for each table to enhance semantic matching.
    - Design Motivation: Mini-tables achieve up to 33× fewer online embedding calls and 70% shorter contexts without sacrificing retrieval precision.

3. **Dataset-Specific Model Selection**:

    - Function: Selects the optimal pretrained model tailored to the characteristics of each dataset.
    - Mechanism: NQ-Tables (single-hop factoid queries) uses all-mpnet-base-v2 + text-embedding-3-large; OTT-QA (multi-hop reasoning, hybrid-mode text) uses Jina Embeddings v3 + gemini-embedding-001. Selection is based on each model's suitability for the specific text characteristics of the dataset.
    - Design Motivation: Query and table characteristics differ across datasets; model selection should match the properties of the data.

### Loss & Training

No training is involved. All models use pretrained weights or APIs. End-to-end QA employs Llama3-8B, Qwen2.5-7B, and Mistral-7B in zero-shot or few-shot settings for answer generation.

## Key Experimental Results

### Main Results

**NQ-Tables Retrieval Performance**

| Model | Training Required | R@1 | R@10 | R@50 |
|-------|-------------------|-----|------|------|
| THYME (SOTA Hybrid) | Fine-tuning required | 48.55 | 86.38 | 96.08 |
| DTR+HN | Fine-tuning required | 47.33 | 80.96 | 91.51 |
| BIBERT+SPLADE | Fine-tuning required | 45.62 | 86.72 | 95.62 |
| **CRAFT (Zero-Shot)** | **None** | **49.84** | **86.83** | **97.17** |

**OTT-QA Zero-Shot Retrieval Performance**

| Model | R@1 | R@10 | R@50 |
|-------|-----|------|------|
| THYME (Fine-tuned) | 66.67 | 91.10 | 96.16 |
| **CRAFT (Zero-Shot)** | 55.56 | 89.88 | 96.07 |

### Ablation Study

**Query Robustness (Performance Change Δ Under Paraphrased Queries)**

| Model | Original R@10 | Paraphrased Δ (avg) |
|-------|--------------|----------------------|
| DTR (M) | 75.73 | -8.38 |
| DTR (S) | 73.88 | -11.82 |
| DTR (M)+HN | 80.96 | -5.80 |
| **CRAFT** | 87.16 | **-0.04** |

### Key Findings

- CRAFT surpasses all fine-tuned methods on NQ-Tables in a zero-shot setting (R@1 49.84 vs. THYME 48.55), demonstrating that a carefully designed cascaded pipeline can substitute expensive fine-tuning.
- On OTT-QA, CRAFT's zero-shot R@50 (96.07) approaches the fine-tuned SOTA (96.16), with a gap of only 0.09.
- CRAFT is nearly immune to query paraphrasing (Δ = -0.04), whereas fine-tuned DTR degrades by 8–12 points, indicating substantially stronger generalization.
- Each stage of the cascaded design contributes measurably: Stage 1→2 improves R@10 by approximately 10–21 points, and Stage 2→3 adds a further 5–8 points.
- Mini-table construction reduces embedding calls by 33× without loss of retrieval precision.

## Highlights & Insights

- CRAFT outperforms fine-tuned methods through "engineering wisdom" — cascaded retrieval combined with table augmentation — suggesting that the general-purpose capabilities of pretrained models have been underestimated.
- Near-perfect robustness to query paraphrasing (Δ = -0.04) is a highly practical property; fine-tuned models are notably fragile in this regard.
- Mini-table construction is a simple yet effective efficiency optimization; 70% shorter contexts carry significant practical value in real-world deployment.

## Limitations & Future Work

- Reliance on commercial APIs (Gemini, OpenAI embeddings) constrains cost and reproducibility.
- Model selection (different models for NQ-Tables vs. OTT-QA) introduces dataset-specific engineering choices.
- Performance on non-English tables or tables with complex formatting (e.g., merged cells) has not been evaluated.
- Preprocessing (caption/description generation) requires additional offline LLM calls.

## Related Work & Insights

- **vs. THYME**: THYME requires fine-tuning on the target dataset and employs field-aware matching, whereas CRAFT requires no training yet achieves comparable or superior performance through cascaded retrieval.
- **vs. DTR**: DTR is a classic dense retriever but is sensitive to query paraphrasing; CRAFT's cascaded design is inherently more robust.
- **vs. T-RAG**: T-RAG integrates retrieval and generation end-to-end, whereas CRAFT maintains modularity for easy component replacement.

## Rating

- Novelty: ⭐⭐⭐ The combination of cascaded retrieval and table augmentation is effective but not an entirely novel concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across two datasets, robustness testing, stage-wise ablation, and end-to-end QA assessment.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐ Demonstrates that training-free retrieval can achieve SOTA, with direct practical value for real-world deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](../../ICLR2026/information_retrieval/q_rag_long_context_multi_step_retrieval.md)
- [\[ACL 2026\] RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](racer_retrieval-augmented_contextual_rapid_speculative_decoding.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG](../../NeurIPS2025/information_retrieval/secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)

<!-- RELATED:END -->
