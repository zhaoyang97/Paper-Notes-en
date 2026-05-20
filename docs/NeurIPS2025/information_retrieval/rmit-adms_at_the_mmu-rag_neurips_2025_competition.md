---
title: >-
  [Paper Note] RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][Retrieval-Augmented Generation] This paper proposes the Routing-to-RAG (R2RAG) system, which employs an LLM-based query classifier to route simple queries to single-turn Vanill…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Query Classification"
  - "Adaptive Retrieval"
  - "Lightweight RAG"
  - "Dynamic Evaluation"
date: 2026-05-08
content_hash: e18dcc4ae0be7ae4
---

# RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition

**Conference**: NeurIPS 2025
**arXiv**: [2602.20735](https://arxiv.org/abs/2602.20735)  
**Code**: [GitHub](https://github.com/rmit-ir/NeurIPS-MMU-RAG)  
**Area**: Information Retrieval / RAG
**Keywords**: Retrieval-Augmented Generation, Query Classification, Adaptive Retrieval, Lightweight RAG, Dynamic Evaluation

## TL;DR

This paper proposes the Routing-to-RAG (R2RAG) system, which employs an LLM-based query classifier to route simple queries to single-turn Vanilla RAG and complex queries to iterative Vanilla Agent. All components are built upon two lightweight models — Qwen3-4B (unquantized) and Qwen3-Reranker-0.6B — running on a single consumer-grade GPU, and the system won the Best Dynamic Evaluation award in the open-source track of the NeurIPS 2025 MMU-RAG competition.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) has become the standard approach for improving LLM reliability. Competitions such as LiveRAG (ACM SIGIR 2025) and MMU-RAG (NeurIPS 2025) provide standardized evaluation settings. The MMU-RAG competition encompasses four evaluation dimensions: static/dynamic evaluation × open-source/closed-source systems, with final rankings derived from a robustness-aware aggregation of automatic metrics combined with human Likert scores.

**Limitations of Prior Work**:
- Existing RAG systems apply a uniform retrieval strategy to all queries, ignoring differences in query complexity — simple factoid questions and multi-step reasoning questions require fundamentally different processing pipelines.
- Synthetic LLM evaluation (LLM-as-a-Judge) suffers from distributional bias and evaluation artifacts, creating a gap with genuine user preferences.
- High-performing RAG systems typically rely on large models and expensive hardware, limiting reproducibility and practical deployment.

**Key Insight**: The proposed system extends G-RAG, the champion system of SIGIR 2025 LiveRAG, by introducing query complexity classification and adaptive routing. A qualitative user study with $N=20$ participants is incorporated to guide system refinement, achieving a lightweight yet effective dynamic RAG pipeline.

## Method

### Overall Architecture

R2RAG is a three-stage pipeline of "classify–route–generate": (1) an LLM-based query classifier determines whether a query is simple or complex; (2) simple queries are directed to the Vanilla RAG single-turn retrieval path, while complex queries are directed to the Vanilla Agent iterative retrieval path; (3) after sufficient evidence is collected, the LLM generates a response with inline citations. All components share Qwen3-4B (inference) and Qwen3-Reranker-0.6B (reranking), deployed via vLLM on a single GPU.

### Key Designs

1. **Query Classifier**:

    - Function: Classifies input queries into *simple* (factual/single-step) and *complex* (multi-faceted/multi-step reasoning) categories.
    - Design comparison: Two approaches were evaluated — an LR classifier (trained on 175,850 queries using 128-dimensional semantic embeddings + 19 linguistic features = 147-dimensional feature vectors) and an LLM-based classifier (using a structured prompt to guide Qwen3-4B in judging "whether multiple Google searches are required").
    - The LLM-based classifier was ultimately adopted because: (a) it reasons over query semantics rather than relying solely on surface features; (b) it handles out-of-distribution queries more robustly; (c) classification criteria can be flexibly adjusted by modifying the prompt. The trade-off is higher inference overhead, which is acceptable within the competition's 10-minute-per-query time limit.
    - Design Motivation: A qualitative user study found that Vanilla Agent handles complex queries well but produces unnecessarily verbose responses for simple queries, while Vanilla RAG is concise for simple queries but insufficient for complex tasks — motivating the routing mechanism to combine the strengths of both.

2. **Vanilla RAG (Simple Query Path)**:

    - A four-stage single-turn pipeline: query variant generation → parallel retrieval → reranking → answer generation.
    - Query variants: Qwen3-4B with thinking mode enabled generates 3 variants using different keywords/phrasings to expand retrieval coverage.
    - Retrieval: Each variant is retrieved in parallel over the ClueWeb22-A index, returning ≤10 documents per variant; after deduplication, approximately <30 documents remain.
    - Reranking: Qwen3-Reranker-0.6B is used as a pointwise reranker, predicting the probability of "yes" for each query–document pair as a relevance score. Top documents are selected and truncated to a 5K-word limit.
    - Answer generation: Qwen3-4B synthesizes a response grounded in the retrieved documents, with required inline citations in `[ID]` format; balanced perspectives are encouraged for controversial topics.

3. **Vanilla Agent (Complex Query Path)**:

    - Core mechanism: Extends Vanilla RAG with an iterative search loop using more aggressive parameters — up to 5 query variants (vs. 3) and a 25K-token document limit (vs. 5K words).
    - State maintenance: Accumulates across iterations (1) previously used queries (to avoid repetition) and (2) summaries of useful documents (to guide subsequent searches).
    - Stopping condition $S$: $(T > 20{,}000) \lor (Cov = 1) \lor (i \geq 5)$, i.e., termination when accumulated tokens exceed 20K, coverage is complete, or the number of iterations reaches 5.
    - Document review and coverage judgment ($Cov$): After each reranking round, Qwen3-4B evaluates five dimensions — whether multiple perspectives are needed, whether sub-questions are covered, whether controversial topics are balanced, identification of newly useful documents, and remaining knowledge gaps. If coverage is insufficient, new queries are generated targeting the identified gaps.

### Model Selection and Deployment Strategy

| Configuration | Setting | Notes |
|--------|---------|------|
| Inference model | Qwen3-4B (unquantized) | Prior experiments showed 4-bit quantized Qwen3-8B underperforms unquantized 4B |
| Reranking model | Qwen3-Reranker-0.6B | Size fits hardware budget; co-located with main model on single GPU |
| Inference framework | vLLM | Efficient memory management and inference acceleration |
| Context window | 25,000 tokens (native 32,768) | Reduced to conserve GPU memory |
| Decoding parameters | temperature=0.6, top_p=0.95 | Developer-recommended configuration |
| Hardware requirement | Single consumer-grade GPU | — |

### Loss & Training

The system requires no end-to-end training. The LR classifier was trained on 175,850 queries (TREC Deep Learning + Deep-Research Questions + TREC RAG 2025 + Natural Questions) but was ultimately not adopted. All core components are prompt-driven, and system optimization relies primarily on iterative prompt and parameter refinement guided by qualitative user study feedback.

## Key Experimental Results

### Qualitative User Study

- Participants: $N=20$ individuals from the CHAI Centre at RMIT University, ranging from master's students to professors, with backgrounds in IR, ML, HCI, NLP, and data science.
- Method: 2-hour free exploration; each participant submitted an average of 17 queries, providing binary feedback (👍/👎), open-ended comments, and verbal reflection.
- Analysis: Preference Ratio (👍/👎 ratio) + manual inspection of open-ended comments.

| System Variant | Simple Query Performance | Complex Query Performance | Main Issue |
|----------|-------------|-------------|---------|
| Vanilla RAG | Concise and accurate | Insufficient information | Inadequate coverage of multi-faceted questions |
| Vanilla Agent | Excessively verbose | Effective | Unnecessary multi-turn retrieval for simple questions |
| R2RAG (Routing) | Concise and accurate | Effective | Combines strengths of both |

### Competition Results

R2RAG won the **Best Dynamic Evaluation award in the open-source track** of the NeurIPS 2025 MMU-RAG competition. The competition evaluation framework combines:
- Robustness-aware normalized aggregation of automatic metrics
- Human ordinal scoring based on a Likert scale
- Both static and live dynamic (RAG-Arena) evaluation modes

### Key Findings

- **Small models are viable**: Qwen3-4B (4B parameters, unquantized) performs strongly across multiple tasks including query classification, query reformulation, document review, and answer generation; a larger model with 4-bit quantization (Qwen3-8B) yields inferior results.
- **Query routing is effective**: The Preference Ratio appears similar across two variants in aggregate, but qualitative analysis reveals complementary strengths — the routing mechanism enables each strategy to perform optimally on its respective query type.
- **Qualitative evaluation is irreplaceable**: User feedback surfaces dimensions that automatic metrics cannot capture — perceived system bias, response structure preferences, expectations around detail level, and tonal sensitivity.
- **Full-document reasoning outperforms chunking**: Retrieval and reasoning are performed over complete documents rather than chunk-level operations.

## Highlights & Insights

- Careful system engineering outweighs model scale — a 4B model with well-designed routing and prompts outperforms larger and more expensive systems in the competition.
- The stopping condition design (triple safeguard of token threshold + coverage + iteration cap) is concise and effective, preventing context overflow while ensuring sufficient information coverage.
- The methodology of using qualitative user studies to drive system improvement is noteworthy — actionable design refinements are extracted from feedback of $N=20$ participants rather than blindly optimizing automatic metrics.
- The findings align with the competition organizers' conclusion: "Live evaluation matters" — dynamic real-time evaluation surfaces qualitative differences that static metrics cannot capture.

## Limitations & Future Work

- As a competition system report, the paper lacks direct quantitative comparisons with other participating systems (no score comparison table).
- Query classification is binary (simple/complex); finer-grained multi-level classification could further improve routing effectiveness.
- The system relies on the competition-provided ClueWeb22-A retrieval API; optimization of the retrieval index itself is not addressed.
- Coverage judgment relies entirely on LLM subjective assessment, without quantifiable coverage metrics.
- Latency and throughput are not discussed — multi-turn iteration for complex queries under a single-GPU constraint may result in considerable response times.

## Related Work & Insights

- The system directly extends G-RAG (SIGIR 2025 LiveRAG champion), demonstrating a path for iterative improvement of competition systems across tracks.
- Query complexity classification draws on the definition from the Researchy Questions dataset — non-factoid questions requiring long-form reasoning or multi-step synthesis.
- The pointwise reranking approach is derived from "Beyond Yes and No" (NAACL 2024), replacing binary relevance judgments with fine-grained relevance labels.
- The combination of Qwen3 models (4B inference + 0.6B reranking) demonstrates that task-specialized small model combinations can outperform a single large model in resource-constrained settings.

## Rating

- Novelty: ⭐⭐⭐ Query routing is not new, but the engineering optimization in a competition setting and the qualitative evaluation-driven improvement methodology have practical value.
- Experimental Thoroughness: ⭐⭐⭐ Competition award validates effectiveness, but a quantitative comparison table with other systems is absent.
- Writing Quality: ⭐⭐⭐⭐ The competition report is clearly structured with well-justified design choices and fully disclosed prompts.
- Value: ⭐⭐⭐⭐ Strong practical reference value for resource-constrained RAG deployment; fully open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG](hifi-rag_hierarchical_content_filtering_and_two-pass_generation_for_open-domain_.md)
- [\[NeurIPS 2025\] RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering](rag-igbench_innovative_evaluation_for_rag-based_interleaved_generation_in_open-d.md)
- [\[NeurIPS 2025\] SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG](secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](../../ACL2026/information_retrieval/is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

</div>

<!-- RELATED:END -->
