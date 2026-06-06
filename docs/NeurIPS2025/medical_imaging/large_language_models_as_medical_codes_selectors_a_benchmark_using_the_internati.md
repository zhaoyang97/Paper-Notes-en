---
title: >-
  [Paper Note] Large Language Models as Medical Codes Selectors: A Benchmark Using the International Classification of Primary Care
description: >-
  [NeurIPS 2025 (GenAI for Health Workshop)][Medical Imaging][Medical coding] This work constructs a medical coding benchmark based on an extract-retrieve-select framework…
tags:
  - "NeurIPS 2025 (GenAI for Health Workshop)"
  - "Medical Imaging"
  - "Medical coding"
  - "ICPC-2"
  - "LLM benchmark"
  - "extreme multi-label classification"
  - "semantic retrieval"
date: 2026-05-08
content_hash: 55091921e9cc58be
---

# Large Language Models as Medical Codes Selectors: A Benchmark Using the International Classification of Primary Care

**Conference**: NeurIPS 2025 (GenAI for Health Workshop)  
**arXiv**: [2507.14681](https://arxiv.org/abs/2507.14681)  
**Code**: Available  
**Area**: Medical Imaging / Medical Coding  
**Keywords**: Medical coding, ICPC-2, LLM benchmark, extreme multi-label classification, semantic retrieval

## TL;DR

This work constructs a medical coding benchmark based on an extract-retrieve-select framework, evaluating ICPC-2 code selection capability across 33 LLMs. Results show that 28 models achieve F1 > 0.8, demonstrating that LLMs can effectively automate primary care coding without fine-tuning.

## Background & Motivation

**Background**: Medical coding is the task of mapping clinical expressions to standardized classification systems (e.g., ICD-10, ICPC-2), which constitutes an extreme multi-label classification (XMC) problem. Traditional approaches are time-consuming and error-prone.

**Limitations of Prior Work**: ICPC-2 contains approximately 1,300 categories with severe class imbalance; existing automated coding methods predominantly target ICD, with limited research addressing ICPC-2 for primary care settings.

**Key Challenge**: Direct LLM-based coding is prone to hallucination (generating non-existent codes), whereas retrieval-augmented approaches can constrain the output space.

**Key Insight**: The problem is decomposed into three stages—extract, retrieve, and select. This paper focuses on evaluating the select stage: given retrieved candidates, can an LLM correctly identify the appropriate ICPC-2 code?

## Method

### Overall Architecture

(1) A semantic search engine (OpenAI text-embedding-3-large) retrieves candidates from 73,563 annotated concepts; (2) an LLM receives the query along with the retrieved results and selects the most appropriate ICPC-2 code.

### Key Designs

1. **Semantic Retrieval Engine**

    - Function: Matches clinical expressions against the ICPC-2 concept library.
    - Mechanism: Chroma DB with the HNSW algorithm for vector similarity search.
    - Design Motivation: Leverages pre-trained embeddings to capture semantic similarity, overcoming the limitations of exact matching.

2. **LLM Selection Evaluation**

    - Function: 33 LLMs perform code selection on identical retrieval results.
    - Mechanism: A unified prompt template provides the query and a list of retrieved candidates, instructing the LLM to select the best ICPC-2 code.
    - Evaluation Dimensions: F1-score, token consumption, cost, response latency, and format compliance.

### Loss & Training

No training is required. All evaluations are conducted in pure inference mode; all LLMs use the same zero-shot prompt.

## Key Experimental Results

### Main Results (437 Brazilian Portuguese Clinical Expressions)

| Model | F1-score | Token Usage | Format Compliance |
|------|---------|-----------|---------|
| gpt-4.5-preview | **0.89** | Moderate | 99% |
| o3 | **0.88** | High | 99% |
| gemini-2.5-pro | **0.87** | Moderate | 98% |
| gpt-4o | 0.85 | Moderate | 99% |
| Baseline (top-1 retrieval) | 0.81 | None | 100% |
| gpt-4o (no retrieval) | 0.72 | Moderate | 95% |
| Small models (<3B) | <0.60 | Low | <80% |

### Ablation Study (Impact of Retrieval Optimization)

| Retrieval Configuration | Best Model F1 | Gain |
|---------|-----------|------|
| Default retrieval ($k=10$) | 0.85 | Baseline |
| Optimized retrieval ($k=20$) | **0.89** | +4pp |
| No retrieval | 0.72 | −13pp |

### Key Findings

- 28/33 models achieve F1 > 0.8, demonstrating strong coding capability of LLMs under constrained conditions.
- Retrieval optimization yields up to 4pp improvement, indicating that retrieval quality directly affects selection quality.
- Small models (<3B) primarily fail in format compliance and long-input handling.
- Direct coding without retrieval leads to a significant F1 drop (−13pp), validating the necessity of the retrieve-then-select paradigm.

## Highlights & Insights

- **Modular Framework**: Each stage of extract-retrieve-select can be independently optimized, facilitating system upgrades.
- **Hallucination Suppression**: By constraining the output space through retrieval, LLMs rarely generate non-existent codes.
- **Multilingual Potential**: Evaluated on Brazilian Portuguese, most LLMs demonstrate strong cross-lingual performance.

## Limitations & Future Work

- The evaluation dataset contains only 437 instances, limiting scale.
- Only the select stage is evaluated; end-to-end assessment (from clinical notes to final codes) is absent.
- Evaluation is restricted to a single language (Portuguese); multilingual generalization requires further validation.
- Inter-annotator agreement is not accounted for.

## Related Work & Insights

- **vs. Infer-Retrieve-Rank**: D'Oosterlinck et al.'s general XMC framework was validated on non-medical benchmarks; this paper provides the first systematic evaluation on ICPC-2.
- **Insight**: This framework can be extended to ICD-10 coding (14,000+ categories), where the scale is larger but the pipeline remains generalizable.

## Rating
- Novelty: ⭐⭐⭐ The framework concept is not entirely new, but this is the first systematic evaluation on ICPC-2.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across 33 models.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-structured.
- Value: ⭐⭐⭐⭐ Practically significant for automating primary care coding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [\[NeurIPS 2025\] EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis](endobench_a_comprehensive_evaluation_of_multi-modal_large_language_models_for_en.md)
- [\[AAAI 2026\] Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models](../../AAAI2026/medical_imaging/coarse-to-fine_open-set_graph_node_classification_with_large_language_models.md)
- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](../../ACL2026/medical_imaging/beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)

</div>

<!-- RELATED:END -->
