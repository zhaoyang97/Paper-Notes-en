---
title: >-
  [Paper Note] RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora
description: >-
  [ACL 2026][Video Understanding][Redundancy-aware retrieval] This paper proposes the RARE framework, which tracks cross-document redundancy by decomposing documents into atomic facts and introduces CRRF (Criterion-separat…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Redundancy-aware retrieval"
  - "high-similarity corpora"
  - "multi-hop retrieval evaluation"
  - "enterprise RAG"
  - "atomic fact decomposition"
date: 2026-05-08
content_hash: f48fdb9e0711cb52
---

# RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora

**Conference**: ACL 2026
**arXiv**: [2604.19047](https://arxiv.org/abs/2604.19047)
**Code**: None
**Area**: Video Understanding
**Keywords**: Redundancy-aware retrieval, high-similarity corpora, multi-hop retrieval evaluation, enterprise RAG, atomic fact decomposition

## TL;DR

This paper proposes the RARE framework, which tracks cross-document redundancy by decomposing documents into atomic facts and introduces CRRF (Criterion-separated Reciprocal Rank Fusion) to stabilize multi-criteria LLM judgments. The framework constructs the RedQA benchmark over high-redundancy enterprise corpora in finance, legal, and patent domains, revealing that mainstream retrievers suffer a dramatic collapse in PerfRecall@10 from 66.4% to 5.0–27.9% under 4-hop high-overlap settings.

## Background & Motivation

**State of the Field**: Existing QA benchmarks (e.g., HotpotQA, NQ, MS MARCO) assume minimal information overlap across documents, with each answer corresponding to a unique gold passage. Mainstream retrieval evaluation pipelines perform well on these "low-overlap" corpora, driving rapid advances in dense retrieval.

**Limitations of Prior Work**: (1) Enterprise-grade RAG systems operate on corpora such as financial annual reports, legal statutes, and patent documents, which are inherently highly redundant and similar—the same fact recurs across multiple passages in slightly different forms. (2) In high-redundancy settings, retrievers are unfairly penalized when they return "non-source passages" that nonetheless contain the correct answer information. (3) Strong performance on existing benchmarks overestimates the true robustness of models when deployed in enterprise environments.

**Root Cause**: The core assumption underlying existing retrieval evaluation—that each answer has a unique gold passage—fundamentally does not hold for enterprise corpora. A framework is needed that can systematically track cross-document information redundancy and incorporate it into evaluation labels.

**Paper Goals**: (1) Construct a general framework enabling practitioners to build RAG evaluation benchmarks that faithfully reflect deployment conditions in their own domain corpora. (2) Quantify the gap between existing benchmarks and enterprise corpora.

**Starting Point**: Documents are decomposed into minimal, indivisible "atomic fact" units, and redundancy is tracked at the atomic granularity. Atomic facts introduce less noise in the embedding space than passage-level representations, narrowing the gap between semantic similarity and factual equivalence and making LLM equivalence judgments more reliable.

**Core Idea**: A redundancy-aware gold label set is constructed via atomic fact decomposition combined with a two-stage redundancy detection pipeline (embedding retrieval + LLM verification). CRRF (criterion separation + reciprocal rank fusion) is introduced to stabilize LLM multi-criteria judgments, addressing quality control challenges in data generation.

## Method

### Overall Architecture

The RARE framework consists of three stages: (1) **Valid Information Selection**—decomposing document chunks into atomic facts, filtering invalid units, and ranking by multiple criteria; (2) **Systematic Redundancy Tracking**—detecting factual equivalence relations across documents at the atomic granularity; (3) **QA Generation**—combining atomic facts into multi-hop reasoning chains, applying rigorous logical filtering and multi-criteria ranking to produce high-quality benchmark questions. The input is a domain document corpus; the output is the redundancy-aware multi-hop QA benchmark RedQA.

### Key Designs

1. **Atomic Fact Decomposition and Multi-Criteria Ranking (Valid Information Selection)**:

    - Function: Extract high-quality minimal factual units from raw documents as building blocks for multi-hop question generation.
    - Mechanism: An LLM decomposes each document chunk into atomic information units $\mathcal{A} = f_{\text{LLM}}(C)$. After filtering by three minimum criteria (completeness, non-triviality, factuality), the remaining units are ranked by five quality dimensions (validity, completeness, specificity, clarity, questionability) via CRRF, and the top-$k$ units are passed to the downstream pipeline.
    - Design Motivation: Atomic granularity is more precise than passage granularity—passages interleave multiple facts, whereas atomic units isolate a single claim, enabling precise redundancy tracking and flexible multi-hop composition.

2. **Two-Stage Redundancy Detection (Systematic Redundancy Tracking)**:

    - Function: Identify semantically equivalent facts distributed across different document passages to construct a redundancy-aware gold evidence set.
    - Mechanism: Stage 1 retrieves candidate redundant sets $\mathcal{C}_\tau(a_t)$ via embedding similarity (threshold $\tau=0.5$, recall-oriented). Stage 2 applies LLM judgment $\phi(a_t, a_j)$ to precisely verify factual equivalence. Each target atomic fact $a_t$ is ultimately associated with its redundancy mapping $a_t \mapsto \mathcal{R}(a_t)$.
    - Design Motivation: Pure embedding retrieval produces false positives (semantically similar but not equivalent), while pure LLM verification is prohibitively expensive. The two-stage design balances recall and precision—the lenient embedding threshold avoids missing equivalent facts, while LLM verification ensures annotation accuracy.

3. **CRRF: Criterion-Separated Reciprocal Rank Fusion**:

    - Function: Stabilize LLM judgments in multi-criteria ranking tasks, addressing output instability from joint reasoning.
    - Mechanism: A separate LLM call is issued for each quality criterion to obtain a per-criterion ranking $\text{rank}_i(x)$; the aggregate score is then computed via reciprocal rank fusion as $s(x) = \sum_{i=1}^{N} \frac{1}{\text{rank}_i(x)}$, discarding LLM confidence scores entirely.
    - Design Motivation: LLM joint reasoning over multiple criteria is unstable (requiring simultaneous balancing of competing objectives), and LLM confidence scores are poorly calibrated across criteria. CRRF relies solely on ordinal preferences (rankings are more reliable than calibrated probabilities), and criterion separation reduces cross-criterion interference.

### Loss & Training

RARE is a data construction framework and does not involve end-to-end training. LLMs are used in inference mode (GPT-5 Nano for judgments, GPT-5 for question generation), and text-embedding-3-large is used for similarity computation.

## Key Experimental Results

### Main Results

**Cross-Domain Retrieval Performance (Qwen3-8B)**

| Domain | Coverage@10 | PerfRecall@10 | Redundancy (%) | Similarity (%) |
|--------|------------|---------------|----------------|----------------|
| General-Wiki | 93.58 | 88.66 | 1.4 | 8.8 |
| Patent | 84.05 | 63.12 | 49.7 | 29.0 |
| Finance | 72.92 | 47.44 | 63.2 | 35.1 |
| Legal | 67.16 | 41.49 | 25.1 | 40.7 |

### Ablation Study

**CRRF Strategy Ablation (NDCG@3)**

| Prompt Strategy | Aggregation | GPT-5 Nano | GPT-5 |
|----------------|-------------|-----------|-------|
| Vanilla | Base | 0.352 | 0.341 |
| Combined | RRF | 0.419 | 0.410 |
| Separate | Base | 0.391 | 0.387 |
| **Separate (CRRF)** | **RRF** | **0.463** | **0.467** |

### Key Findings

- Retrieval performance degradation is driven primarily by document similarity rather than redundancy—Legal has the highest similarity (40.7%) but the lowest redundancy (25.1%), yet yields the worst PerfRecall@10 (41.49%), indicating that the "confusion effect" of near-duplicate documents is stronger than the "alternative path effect" of redundancy.
- Performance degrades sharply with hop depth: Finance drops from 90.1% at 1-hop to 8.5% at 4-hop, whereas General-Wiki maintains 66.4% at 4-hop.
- Within CRRF, criterion separation improves performance by 11% over joint prompting (0.419→0.463), and RRF aggregation improves by 18% over score aggregation under separated prompting (0.391→0.463).
- End-to-end RAG experiments show that retrieval quality is the dominant lever—answer accuracy is substantially higher when the relevant unit is retrieved than when it is not.

## Highlights & Insights

- The atomic fact decomposition approach is particularly elegant—it not only resolves the granularity problem in redundancy tracking but also naturally provides modular building blocks for multi-hop question composition. This "decompose-then-compose" paradigm is transferable to any scenario requiring precise content tracking.
- CRRF offers a simple yet effective recipe for stabilizing LLM judgments—criterion separation combined with rank fusion can be directly applied to any task requiring multi-criteria LLM evaluation (e.g., paper reviewing, data quality assessment).
- The finding that document similarity is a stronger predictor of retrieval degradation than redundancy carries important implications for RAG system design—practitioners should prioritize measuring inter-document similarity rather than redundancy before deployment.

## Limitations & Future Work

- The framework relies on LLM judgments (GPT-5/GPT-5 Nano) for generation and verification, inheriting model-specific biases.
- The embedding similarity threshold $\tau=0.5$ is fixed; the optimal value may vary across domains.
- As hop depth increases, some generated questions become list-like—logically valid but less natural.
- Future work could extend the framework to non-English corpora and additional enterprise domains.

## Related Work & Insights

- **vs. HotpotQA/NQ**: These benchmarks assume low cross-document overlap and are unsuitable for enterprise RAG evaluation. RARE explicitly models high-overlap settings.
- **vs. BEIR/MTEB**: These provide standardized retrieval evaluation but rely on static annotations and cannot capture the redundancy dynamics encountered at deployment.
- **vs. PoisonedRAG**: PoisonedRAG focuses on retrieval poisoning attacks, whereas RARE addresses evaluation fairness—redundancy is not a threat but a property that should be correctly annotated.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of atomic-fact redundancy tracking and CRRF is novel, though individual components have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 domains, 9 retrievers, CRRF ablations, human evaluation, and end-to-end RAG analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clear, the framework is modular, and the experimental design is rigorous.
- Value: ⭐⭐⭐⭐⭐ Addresses a critical gap in enterprise RAG evaluation; CRRF is broadly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FuncBenchGen: A Contamination-Free Controllable Evaluation Framework for Reliable Benchmarking](../../ICLR2026/video_understanding/towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)
- [\[CVPR 2026\] RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation](../../CVPR2026/video_understanding/ragtrack_language-aware_rgbt_tracking_with_retrieval-augmented_generation.md)
- [\[ACL 2026\] ViLL-E: Video LLM Embeddings for Retrieval](vill-e_video_llm_embeddings_for_retrieval.md)
- [\[CVPR 2026\] SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning](../../CVPR2026/video_understanding/sail_similarity-aware_guidance_and_inter-caption_augmentation-based_learning_for.md)
- [\[ACL 2026\] VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis](vc-inspector_advancing_reference-free_evaluation_of_video_captions_with_factual_.md)

</div>

<!-- RELATED:END -->
