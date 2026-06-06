---
title: >-
  [Paper Note] RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora
description: >-
  [ACL 2026][Information Retrieval & RAG][redundancy-aware retrieval] This paper proposes the RARE framework, which tracks cross-document redundancy by decomposing documents into atomic facts and designs CRRF (Criterion-se…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "redundancy-aware retrieval"
  - "high-similarity corpora"
  - "multi-hop retrieval evaluation"
  - "enterprise-grade RAG"
  - "atomic fact decomposition"
date: 2026-05-08
content_hash: b2bf700ad3d8b7ba
---

# RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora

**Conference**: ACL 2026  
**arXiv**: [2604.19047](https://arxiv.org/abs/2604.19047)  
**Code**: None  
**Area**: Information Retrieval/RAG  
**Keywords**: redundancy-aware retrieval, high-similarity corpora, multi-hop retrieval evaluation, enterprise-grade RAG, atomic fact decomposition

## TL;DR

This paper proposes the RARE framework, which tracks cross-document redundancy by decomposing documents into atomic facts and designs CRRF (Criterion-separated Reciprocal Rank Fusion) to stabilize multi-criteria LLM judgments. By constructing the RedQA benchmark on high-redundancy enterprise corpora such as finance, law, and patents, the study reveals that the PerfRecall@10 of mainstream retrievers plunges from 66.4% to 5.0-27.9% in 4-hop high-overlap settings.

## Background & Motivation

**Background**: Existing QA benchmarks (e.g., HotpotQA, NQ, MS MARCO) assume extremely low information overlap between documents, where each answer corresponds to a unique gold passage. Prevailing retrieval evaluation schemes perform well on these "low-overlap" corpora, driving the rapid development of dense retrieval technologies.

**Limitations of Prior Work**: (1) Enterprise-grade RAG systems actually operate on corpora like annual financial reports, legal statutes, and patent documents, which are naturally high-redundancy and high-similarity—the same fact repeatedly appears in multiple passages in slightly different forms. (2) In high-redundancy scenarios, retrievers are unfairly penalized when returning "non-source passages" that nonetheless contain the correct answer information. (3) Superior performance on existing benchmarks overestimates the true robustness of models in enterprise deployments.

**Key Challenge**: The core assumption of existing retrieval evaluation—that each answer has a unique gold passage—does not hold in enterprise corpora. A framework is needed to systematically track cross-document information redundancy and incorporate it into evaluation labels.

**Goal**: (1) Construct a general framework that allows practitioners to build RAG evaluation benchmarks on their own domain corpora that truly reflect deployment conditions. (2) Quantify the gap between existing benchmarks and enterprise corpora.

**Key Insight**: Decomposing documents into minimal, indivisible "atomic fact" units allows for redundancy tracking at an atomic granularity. Atomic facts have lower noise in the embedding space than passage-level representations, resulting in a smaller gap between semantic similarity and factual equivalence, which makes LLM equivalence judgments more reliable.

**Core Idea**: Construct a redundancy-aware gold label set through atomic fact decomposition combined with two-stage redundancy detection (embedding retrieval + LLM verification). Simultaneously, use CRRF (Criterion-separated Reciprocal Rank Fusion) to stabilize multi-criteria LLM judgments, addressing quality control issues in data generation.

## Method

### Overall Architecture

The RARE framework consists of three stages: (1) Valid information selection—decomposing document chunks into atomic facts, filtering invalid units, and ranking them by multiple criteria; (2) Systematic redundancy tracking—detecting factual equivalence across documents at the atomic granularity; (3) QA generation—combining atomic facts into multi-hop reasoning chains and generating high-quality benchmark questions through rigorous logical filtering and multi-criteria ranking. The input is a domain document corpus, and the output is RedQA, a redundancy-aware multi-hop QA benchmark.

### Key Designs

1.  **Valid Information Selection**:
    -   **Function**: Extract high-quality minimal factual units from raw documents as base modules for multi-hop question generation.
    -   **Mechanism**: An LLM decomposes each document chunk into atomic information units $\mathcal{A} = f_{\text{LLM}}(C)$. After filtering by three minimum standards (integrity, non-triviality, factuality), the remaining units are ranked across five quality dimensions (validity, completeness, specificity, clarity, interrogability) via CRRF. The top-k units proceed to the subsequent pipeline.
    -   **Design Motivation**: Atomic granularity is more precise than passage granularity—while multiple facts are intertwined in a passage, atomic units isolate single statements, supporting both precise redundancy tracking and flexible multi-hop combination modules.

2.  **Systematic Redundancy Tracking**:
    -   **Function**: Identify semantically equivalent facts distributed across different document passages to construct a redundancy-aware gold evidence set.
    -   **Mechanism**: The first stage uses embedding similarity (threshold $\tau=0.5$, focusing on recall) to retrieve a candidate redundancy set $\mathcal{C}_\tau(a_t)$. The second stage uses LLM judgment $\phi(a_t, a_j)$ to precisely verify factual equivalence. Finally, each target atomic fact $a_t$ records its redundancy mapping $a_t \mapsto \mathcal{R}(a_t)$.
    -   **Design Motivation**: Pure embedding retrieval suffers from false positives (semantically similar but not equivalent), while pure LLM verification is too costly. The two-stage design balances recall and precision—a loose embedding threshold ensures no equivalent facts are missed, while LLM verification ensures precise annotation.

3.  **CRRF: Criterion-separated Reciprocal Rank Fusion**:
    -   **Function**: Stabilize LLM judgments in multi-criteria ranking tasks and solve the problem of unstable output in joint reasoning.
    -   **Mechanism**: Independent LLM calls are initiated for each quality criterion to obtain a per-criterion ranking $\text{rank}_i(x)$. A comprehensive score is then calculated via Reciprocal Rank Fusion: $s(x) = \sum_{i=1}^{N} \frac{1}{\text{rank}_i(x)}$. LLM confidence scores are completely discarded.
    -   **Design Motivation**: LLM output is unstable when performing joint reasoning across multiple criteria (needing to balance competing goals), and LLM confidence scores are poorly calibrated across criteria. CRRF relies only on ordinal preferences (LLM-generated rankings are more reliable than calibrated probabilities), and criterion separation reduces cross-interference.

### Loss & Training

RARE is a data construction framework and does not involve end-to-end training. LLMs are used in inference mode (GPT-5 Nano for judgment, GPT-5 for question generation), and text-embedding-3-large is used for similarity calculations.

## Key Experimental Results

### Main Results

**Cross-domain Retrieval Performance (Qwen3-8B)**

| Domain | Coverage@10 | PerfRecall@10 | Redundancy (%) | Similarity (%) |
| :--- | :--- | :--- | :--- | :--- |
| General-Wiki | 93.58 | 88.66 | 1.4 | 8.8 |
| Patent | 84.05 | 63.12 | 49.7 | 29.0 |
| Finance | 72.92 | 47.44 | 63.2 | 35.1 |
| Legal | 67.16 | 41.49 | 25.1 | 40.7 |

### Ablation Study

**CRRF Strategy Ablation (NDCG@3)**

| Prompting Strategy | Aggregation Method | GPT-5 Nano | GPT-5 |
| :--- | :--- | :--- | :--- |
| Vanilla | Base | 0.352 | 0.341 |
| Combined | RRF | 0.419 | 0.410 |
| Separate | Base | 0.391 | 0.387 |
| **Separate (CRRF)** | **RRF** | **0.463** | **0.467** |

### Key Findings

-   Retrieval performance degradation is primarily driven by document similarity rather than redundancy. The Legal domain has the highest similarity (40.7%) but the lowest redundancy (25.1%), yet it has the worst PerfRecall@10 (41.49%), indicating the "confusion effect" of similar documents is stronger than the "alternative path effect" of redundancy.
-   Performance degrades sharply as hop depth increases: Finance drops from 90.1% for 1-hop to 8.5% for 4-hop, while General-Wiki maintains 66.4% at 4-hop.
-   In CRRF, criterion separation improves performance by 11% over joint prompting (0.419 $\rightarrow$ 0.463), and RRF aggregation improves performance by 18% over score aggregation under separate prompting (0.391 $\rightarrow$ 0.463).
-   End-to-end RAG experiments show that retrieval quality is the dominant lever—the accuracy of hit units is significantly higher than that of missed units.

## Highlights & Insights

-   The idea of atomic fact decomposition is ingenious—it not only solves the granularity problem of redundancy tracking but also naturally provides combinatorial modules for multi-hop questions. This "decompose then compose" approach is transferable to any scenario requiring precise content tracking.
-   CRRF is a simple but effective recipe for stabilizing LLM judgments—the idea of criterion separation + rank fusion can be directly applied to any task requiring multi-criteria evaluation by an LLM (e.g., paper reviewing, data quality assessment).
-   The finding that document similarity predicts retrieval degradation better than redundancy provides important insights for RAG system design—inter-document similarity of the corpus should be prioritized for evaluation before deployment.

## Limitations & Future Work

-   Relies on LLM judgments (GPT-5/GPT-5 Nano) for generation and verification, inheriting model-specific biases.
-   The embedding similarity threshold $\tau=0.5$ is fixed; the optimal setting may vary by domain.
-   As hop depth increases, some generated questions become list-like—while logically valid, they are less natural.
-   Future work could extend to non-English corpora and more enterprise domains.

## Related Work & Insights

-   **vs HotpotQA/NQ**: These assume low overlap between documents and are unsuitable for enterprise-grade RAG evaluation. RARE explicitly models high-overlap scenarios.
-   **vs BEIR/MTEB**: These provide standardized retrieval evaluation but rely on static annotations, failing to reflect redundancy dynamics during deployment.
-   **vs PoisonedRAG**: This focuses on retrieval poisoning attacks, whereas RARE focuses on evaluation fairness—redundancy is seen as a characteristic to be correctly labeled rather than a threat.

## Rating

-   Novelty: ⭐⭐⭐⭐ Atomic fact redundancy tracking + CRRF is a novel combination, though components have precedents.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 domains, 9 retrievers + CRRF ablation + human evaluation + end-to-end RAG analysis.
-   Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, modular framework, and rigorous experimental design.
-   Value: ⭐⭐⭐⭐⭐ Fills a critical gap in enterprise-grade RAG evaluation; CRRF is widely reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reliable Evaluation Protocol for Low-Precision Retrieval](reliable_evaluation_protocol_for_low-precision_retrieval.md)
- [\[ACL 2026\] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](disco-rag_discourse-aware_retrieval-augmented_generation.md)
- [\[ACL 2026\] MTR-Suite: A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks](mtr-suite_a_framework_for_evaluating_and_synthesizing_conversational_retrieval_b.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](../../AAAI2026/information_retrieval/knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)

</div>

<!-- RELATED:END -->
