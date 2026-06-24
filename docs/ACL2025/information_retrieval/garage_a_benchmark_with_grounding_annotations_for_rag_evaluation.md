---
title: >-
  [Paper Note] GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG benchmark] GaRAGe is a RAG benchmark consisting of 2,366 questions and over 35K human-annotated grounding passages. Through fine-grained grounding relevance annotations, it systematically evaluates LLMs' abilities to identify relevant information, deflect (refuse to answer), and attribute references in RAG scenarios.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG benchmark"
  - "grounding annotation"
  - "factuality"
  - "deflection"
  - "attribution"
date: 2026-05-08
content_hash: 24f83df8705851be
---

# GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation

**Conference**: ACL 2025  
**arXiv**: [2506.07671](https://arxiv.org/abs/2506.07671)  
**Code**: [Available](https://github.com/amazon-science/GaRAGe)  
**Area**: NLP / RAG Evaluation  
**Keywords**: RAG benchmark, grounding annotation, factuality, deflection, attribution

## TL;DR

GaRAGe is a RAG benchmark consisting of 2,366 questions and over 35K human-annotated grounding passages. Through fine-grained grounding relevance annotations, it systematically evaluates LLMs' abilities to identify relevant information, deflect (refuse to answer), and attribute references in RAG scenarios.

## Background & Motivation

Retrieval-Augmented Generation (RAG) is one of the most critical paradigms in current LLM applications, where users require LLMs to accurately locate relevant information within retrieved documents and generate well-supported answers. However, existing RAG benchmarks suffer from several key limitations:

**Conflated Evaluation Dimensions**: Most benchmarks evaluate query generation and answer generation together, failing to independently measure the LLMs' information filtering capabilities.

**Unlabeled Grounding Quality**: In existing benchmarks, retrieved passages are either all relevant (unrealistic) or have unknown relevance (preventing precise evaluation), lacking human relevance annotations for each grounding source.

**Overly Simplified Answer Formats**: A vast majority of benchmarks utilize short-form answers or multiple-choice questions, which is disconnected from real-world scenarios where users require comprehensive, long-form answers with proper citations.

**Single Source**: Few benchmarks include a mixture of public Web and private knowledge base grounding, which fails to simulate enterprise RAG scenarios.

GaRAGe is designed specifically to address these issues: it combines temporal sensitivity, multi-dimensional complexity, hybrid public-private sources, and human-written long-form answers, providing the most comprehensive evaluation platform for RAG systems to date.

## Method

### Overall Architecture

The construction process of GaRAGe consists of three major steps: (1) multi-stage complex question generation; (2) multi-source grounding passage collection; and (3) human annotation and verification.

### Key Designs

1. **Dynamic Multi-type Question Generation**: A LLM-driven four-step pipeline is adopted: generate search plan $\rightarrow$ execute Web search $\rightarrow$ generate questions via information fusion $\rightarrow$ filter and de-duplicate. Questions span four dimensions: temporal sensitivity (fast-changing/slow-changing/static), complexity (comparison/multi-hop/post-processing), popularity (head/torso/tail), and domain category. The Design Motivation is to simulate the diversity and difficulty of user queries in real-world RAG scenarios.

2. **Multi-source Grounding Collection**: For each question, complex queries are first decomposed into focused sub-queries, which are then retrieved separately from Web search engines and private knowledge bases (Enron emails, ArXiv abstracts, AWS DevOps guides, SEC filings). An STS classifier is used to filter out sub-queries deviating from the original query intent, followed by document reranking using a cross-encoder. Document reranking is intentionally bypassed for a subset of the queries to introduce noise and elevate difficulty.

3. **Fine-grained Human Annotation**: Professional annotators annotated 2,366 questions across four dimensions (temporal sensitivity/complexity/popularity/category), annotated the relevance of each grounding passage (answers the question/relevant information/outdated/unknown), and authored long-form reference answers with citation tags. 427 questions were designated as requiring deflection (refusing to answer) to simulate scenarios with insufficient grounding.

4. **Evaluation Metric System**:

    - **Eligibility Score**: Whether the answer fully addresses the user's request.
    - **Relevance-Aware Factuality (RAF)**: Whether the answer is based strictly on relevant passages and satisfies eligibility.
    - **Deflection Score**: Whether the LLM correctly refuses to answer when grounding is insufficient (True Positive Rate).
    - **Attribution Score**: Precision/Recall/F1 of citation tags.

### Loss & Training

GaRAGe itself is an evaluation benchmark rather than a model, and does not involve training. Evaluation is performed using GPT-4o as an auto-judge with temperature set to 0.2.

## Key Experimental Results

### Main Results

| Model | Eligibility | Factuality | RAF | Deflection TP |
|------|-----------|------------|-----|--------------|
| GPT-4o | **92.47** | 59.30 | 52.88 | **31.1** |
| Gemini 1.5 Flash | 84.88 | **70.50** | 59.43 | 27.2 |
| Nova Pro | 87.77 | 66.63 | **60.67** | 18.0 |
| Claude Sonnet | 86.07 | 64.67 | 48.91 | 25.3 |
| Qwen 32b | 90.50 | 61.00 | 52.90 | 21.5 |
| Mistral | 85.30 | 43.32 | 34.32 | 5.2 |

### Attribution Experiment

| Model | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Claude Haiku | 49.9 | **71.9** | **58.9** |
| GPT-4o | **57.9** | 59.0 | 58.4 |
| Gemini 1.5 | 54.7 | 56.3 | 55.5 |
| Nova Pro | 56.9 | 49.6 | 53.0 |

### Key Findings

1. **Suboptimal performance across all models in RAF**: Even the best model, Nova Pro, only scores 60.67%. This indicates a general tendency of LLMs toward "over-summarization" rather than strictly generating based on relevant passages.
2. **Extremely weak deflection capability**: The best model, GPT-4o, achieves a True Positive Rate (TPR) of only 31.1%, implying that LLMs still fabricate answers in nearly 70% of scenarios when grounding is unavailable.
3. **Temporal-sensitive questions are more challenging**: The RAF of fast-changing questions is approximately 10% lower than that of static or slow-changing questions.
4. **Private KB scenarios are significantly harder**: Compared to Web-based questions, performance drops by over 10% for private KB queries.
5. **Grounding noise directly impacts quality**: Low relevance ratio (< 33%) grounding leads to an approximate 30% drop in RAF compared to high relevance ratio (> 66%) grounding.

## Highlights & Insights

- **Core Contribution**: Providing fine-grained relevance annotations on each grounding passage, successfully enabling the evaluation to decouple "factual based on relevant info" from "faithful to the overall context" for the first time—two aspects that were conflated in previous benchmarks.
- The introduction of the RAF metric is highly significant: traditional factuality only measures whether an answer is grounded, whereas RAF further requires the grounding source to be relevant and fresh.
- The dataset covers hybrid retrieval scenarios spanning both Web search and private KBs, closely aligning with the practical needs of enterprise deployments.

## Limitations & Future Work

1. The dataset contains English data only, lacking multilingual support.
2. The evaluation leverages GPT-4o as an auto-judge, which may introduce preference bias toward the GPT series.
3. Subjectivity in annotations (e.g., topic popularity) may introduce some noise.
4. The dataset might be contaminated by future LLM training data, affecting the long-term validity of the evaluation.

## Related Work & Insights

- Compared to benchmarks like CRAG, MultiHop RAG, and Facts Grounding, GaRAGe is more comprehensive in grounding annotation, answer completeness, and multi-source support.
- Key takeaway for RAG system developers: Current LLMs' capability to distinguish relevant from irrelevant grounding is far from reliable. The practice of simply concatenating raw retrieval results as context needs urgent improvement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The fine-grained grounding annotations and RAF metric are significant contributions, addressing a critical blind spot of existing benchmarks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Very comprehensive, with evaluations spanning 11 models and slice analyses across multiple dimensions (temporal sensitivity, sources, noise levels, popularity).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-designed tables and figures, and well-articulated evaluation framework.
- **Value**: ⭐⭐⭐⭐⭐ — Holds direct, long-term practical value for the RAG community, successfully filling an evaluation gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables](mt-raig_novel_benchmark_and_evaluation_framework_for_retrieval-augmented_insight.md)
- [\[ACL 2025\] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)
- [\[ACL 2025\] Unanswerability Evaluation for Retrieval Augmented Generation](unanswerability_evaluation_for_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
