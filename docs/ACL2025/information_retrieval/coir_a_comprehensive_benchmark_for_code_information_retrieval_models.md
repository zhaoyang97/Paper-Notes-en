---
title: >-
  [Paper Note] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models
description: >-
  [ACL 2025][Information Retrieval & RAG][Code Retrieval] This paper proposes CoIR, the first comprehensive benchmark for code information retrieval. Comprising 10 datasets across 4 major categories, 8 subtasks, and 14 programming languages, CoIR reveals that even state-of-the-art (SOTA) retrieval models underperform in code retrieval, and highlights that many models have overfitted to existing leaderboards.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Code Retrieval"
  - "Information Retrieval Benchmark"
  - "Embedding Models"
  - "Multilingual Programming"
  - "Code Understanding"
date: 2026-05-08
content_hash: fe5d3bbb3a4cca88
---

# CoIR: A Comprehensive Benchmark for Code Information Retrieval Models

**Conference**: ACL 2025  
**arXiv**: [2407.02883](https://arxiv.org/abs/2407.02883)  
**Code**: [https://github.com/CoIR-team/coir](https://github.com/CoIR-team/coir)  
**Area**: Information Retrieval  
**Keywords**: Code Retrieval, Information Retrieval Benchmark, Embedding Models, Multilingual Programming, Code Understanding

## TL;DR

This paper proposes CoIR, the first comprehensive benchmark for code information retrieval. Comprising 10 datasets across 4 major categories, 8 subtasks, and 14 programming languages, CoIR reveals that even state-of-the-art (SOTA) retrieval models underperform in code retrieval, and highlights that many models have overfitted to existing leaderboards.

## Background & Motivation

While Information Retrieval (IR) has achieved massive success in the textual domain, Code Retrieval—a critical function in daily developer workflows—remains severely undervalued and under-evaluated. Existing code retrieval benchmarks suffer from three major issues:

**Single Task**: CodeSearchNet and CosQA primarily focus on the single task of "text-to-code", neglecting diverse real-world scenarios such as "code-to-code" and "code-to-text". In practice, developers may need to input code snippets with bug information to retrieve explanations, summaries, or fixes.

**Lack of Domain Diversity**: CodeSearchNet only extracts code-comment pairs from GitHub, while XCodeEval focuses exclusively on competitive programming. Such narrow domain coverage fails to comprehensively evaluate model performance across broader coding scenarios.

**Lack of Standardized Evaluation Framework**: Different benchmarks utilize distinct evaluation metrics and formats, making across-benchmark performance comparisons highly challenging.

More critically, many models have already overfitted on commonly used benchmarks like CodeSearchNet—meaning that high scores on leaderboards may not reflect genuine code retrieval capabilities.

## Method

### Overall Architecture

The design of CoIR follows three core principles: diversity (4 major tasks $\times$ 8 subtasks $\times$ 10 datasets $\times$ 14 programming languages), ease of use (one-click evaluation via `pip install`), and mitigation of overfitting (combining diverse tasks and domains).

### Key Designs

1. **Four Major Task Systems**:

    - **Text-to-Code**: Competitive programming code retrieval (APPS), web query code retrieval (CosQA), and Text-to-SQL retrieval (Synthetic Text2SQL).
    - **Code-to-Text**: Code summarization retrieval (CodeSearchNet) — retrieving corresponding comments/summaries using code.
    - **Code-to-Code**: Code context retrieval (CodeSearchNet-CCR, self-constructed) — retrieving the second half of a code snippet given the first half; similar code retrieval (CodeTransOcean) — retrieving semantically equivalent code across different programming languages/frameworks.
    - **Hybrid Code**: Single-turn code QA (self-constructed StackOverflow QA + CodeFeedback-ST) and multi-turn code QA (CodeFeedback-MT) — where both queries and answers contain a mix of text and code.

2. **CodeSearchNet-CCR (Self-constructed Dataset)**: Each code snippet in CodeSearchNet is randomly split into two segments (40% to 70% as query, the rest as retrieval target) to simulate the retrieval demands in code completion. This represents the first large-scale dataset for code context retrieval.

3. **StackOverflow QA (Self-constructed Dataset)**: Questions are paired with their highest-voted answers from raw StackOverflow data, resulting in 19,931 pairs, with 1,202 samples selected for testing.

4. **Data Quality Assurance**: All datasets undergo manual inspection and filtering to remove invalid answers, ambiguous instances, and irrelevant information.

5. **Standardized Evaluation**: Aligned with BEIR and MTEB data formats, NDCG@10 is uniformly adopted as the primary metric, while MAP, Recall, and Precision are also supported. A Python framework `pip install coir` is provided for one-click evaluation.

### Diversity Analysis

By analyzing vocabulary overlap between datasets using weighted Jaccard similarity, it is found that the similarity between most dataset pairs is very low (except for CodeFeedback-ST and CodeFeedback-MT from the same source), confirming the challenging nature and diversity of the benchmark.

## Key Experimental Results

### Main Results (NDCG@10)

| Model (Params) | APPS | CosQA | Text2SQL | CodeSN | CSN-CCR | CodeTrans-C | CodeTrans-DL | SOQA | CF-ST | CF-MT | Average |
|------|------|-------|---------|--------|---------|-------------|-------------|------|-------|-------|------|
| BM25 | 0.95 | 13.96 | 16.92 | 26.75 | 34.69 | 50.13 | 8.69 | 56.80 | 54.32 | 34.73 | 29.79 |
| E5-Base (110M) | 11.52 | 32.59 | 52.31 | 67.99 | 56.87 | 62.50 | 21.87 | 86.86 | 74.52 | 41.99 | 50.90 |
| E5-Mistral (7B) | 21.33 | 31.27 | 65.98 | 54.25 | 65.27 | 82.55 | 33.24 | 91.54 | 72.71 | 33.65 | **55.18** |
| Voyage-Code-002 | 26.52 | 29.79 | 69.26 | 81.79 | 73.45 | 72.77 | 27.28 | 87.68 | 65.35 | 28.74 | **56.26** |
| OpenAI-Ada-002 | 8.70 | 28.88 | 58.32 | 74.21 | 69.13 | 53.34 | 26.04 | 72.40 | 47.12 | 17.74 | 45.59 |

### Efficiency Analysis (CodeFeedBack-ST, 156K corpus + 31K queries)

| Model | Embedding Dimension | Embedding Latency / Sample | Retrieval Latency / Query | Index Size |
|------|---------|-------------|-------------|---------|
| E5-Base | 768 | 7.4ms | 38.1µs | 0.3G |
| BGE-M3 | 1024 | 31.4ms | 42.9µs | 0.6G |
| E5-Mistral | 4096 | 1840ms | 115.5µs | 2.3G |

### Impact of Input Length

| Model (Input Length) | CodeFB-MT | CodeTO-DL | APPS | SOQA |
|------|-----------|-----------|------|------|
| GTE (512) | 28.48 | 28.80 | 3.24 | 62.71 |
| GTE (4k) | 51.32 | 27.33 | 5.08 | 78.63 |
| BGE-M3 (512) | 33.46 | 31.16 | 7.37 | 61.04 |
| BGE-M3 (4k) | 27.49 | 32.75 | 6.80 | 56.53 |

### Key Findings

- **No single model dominates across all tasks**: Voyage-Code-002 achieves the highest average (56.26) but exhibits high variance; E5-Mistral is the strongest on competitive programming and SOQA, but performs mediocrely on other tasks.
- **Code-specialized training is effective but not a silver bullet**: Voyage-Code-002 excels in Text-to-Data and Code-to-Text, but performs poorly on multi-turn QA (CodeFB-MT: 28.74).
- **BM25 almost fails on APPS** (0.95), indicating that semantic understanding requirements for competitive programming problems far exceed lexical matching.
- **Severe overfitting issues**: OpenAI-Ada-002 scores 74.21 on CodeSearchNet but averages only 45.59 on CoIR, a massive gap of 28.6 points.
- **Longer inputs are not always effective**: GTE benefits from long inputs ($512 \rightarrow 4\text{k}$ yields a 16-point improvement on SOQA), whereas BGE-M3 performs worse, indicating that code data possesses distinct characteristics compared to text data.
- **Efficiency-performance trade-off**: While E5-Mistral delivers strong performance, its embedding latency is 250 times that of E5-Base.

## Highlights & Insights

- CoIR is the first truly comprehensive benchmark for code retrieval, covering a broad spectrum of scenarios from code completion to cross-lingual code matching.
- It uncovers the critical issue of overfitting in existing leaderboards: excellent performance on CodeSearchNet does not translate to genuine code retrieval capability.
- The LLM-based retrieval model (E5-Mistral) exhibits the smallest performance gap between CodeSearchNet and CoIR, suggesting that LLMs have the potential to mitigate overfitting.
- Multi-turn code QA (queries up to 4K+ tokens) is identified as a highly challenging new direction, where all modern models currently perform poorly.
- Seamless integration with BEIR/MTEB lowers the barrier to entry.

## Limitations & Future Work

- All datasets are exclusively in English, lacking multilingual code retrieval evaluation.
- Each query corresponds to only one ground-truth answer ($n=1$), which does not reflect real-world scenarios where multiple relevant pieces of code might match a single query.
- Multi-dimensional retrieval based on code metadata (such as version numbers and library dependencies) is not addressed.
- CodeTransOcean has a small scale (hundreds to thousands of instances), offering limited statistical confidence.
- Recent code-specific embedding models (such as CodeSage) were not evaluated.

## Related Work & Insights

- BEIR and MTEB established standardized benchmarks for text retrieval; CoIR extends this philosophy to the code domain.
- The overfitting problem on CodeSearchNet serves as a reminder that evaluation benchmarks need continuous updates and diversification.
- Retrieval methods that incorporate code structural information (such as AST and control flow) represents a promising direction for future breakthroughs.
- Code-RAG systems (retrieval-augmented generation for code using LLMs) represent an important application pipeline.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first comprehensive code retrieval benchmark, featuring mature task design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Complete evaluation of 10 models across 10 datasets, accompanied by efficiency and overfitting analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured, with comprehensive task classification and dataset descriptions.
- **Value**: ⭐⭐⭐⭐⭐ — Fills a critical gap in code retrieval evaluation, providing long-term value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](air-bench_automated_heterogeneous_information_retrieval_benchmark.md)
- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)
- [\[ACL 2025\] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)
- [\[ACL 2025\] Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models](pandora_box_rag_noise.md)

</div>

<!-- RELATED:END -->
