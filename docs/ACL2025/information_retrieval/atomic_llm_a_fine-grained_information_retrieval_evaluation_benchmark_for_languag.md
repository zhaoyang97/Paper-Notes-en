---
title: >-
  [Paper Note] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models
description: >-
  [ACL 2025][Information Retrieval & RAG][Fine-grained evaluation] This paper proposes the Atomic LLM benchmark, which decomposes information retrieval evaluation into atomic-level fact retrieval tasks. It evaluates the information retrieval capabilities of LLMs across multiple dimensions, including factual precision, source attribution, and granularity coverage, revealing systematic deficiencies of existing LLMs in precise fact extraction.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Fine-grained evaluation"
  - "Information Retrieval"
  - "Atomic Facts"
  - "Language Model Benchmark"
  - "Fact-checking"
date: 2026-05-08
content_hash: caea2049290f429b
---

# Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM Evaluation / Information Retrieval  
**Keywords**: Fine-grained evaluation, Information Retrieval, Atomic Facts, Language Model Benchmark, Fact-checking

## TL;DR
This paper proposes the Atomic LLM benchmark, which decomposes information retrieval evaluation into atomic-level fact retrieval tasks. It evaluates the information retrieval capabilities of LLMs across multiple dimensions, including factual precision, source attribution, and granularity coverage, revealing systematic deficiencies of existing LLMs in precise fact extraction.

## Background & Motivation

**Background**: With the popularization of retrieval-augmented generation (RAG) technology, the role of LLMs in information retrieval has shifted from traditional query-document matching to the entire chain of understanding-extraction-generation. Existing LLM information retrieval evaluation benchmarks (such as Natural Questions, TriviaQA, MS MARCO) mainly use end-to-end question-answering accuracy to measure performance, which has a coarse evaluation granularity.

**Limitations of Prior Work**: Coarse-grained evaluation faces three problems: (1) Situations where the answer is correct but the source is incorrect are ignored—LLMs may obtain correct answers through parametric memory rather than actual retrieval; (2) Situations where the answer is partially correct are handled poorly—traditional exact match/F1 metrics cannot capture factual correctness at different granularities; (3) The ability to aggregate information in multi-document scenarios cannot be evaluated—a complex query may require extracting and integrating atomic facts from multiple documents.

**Key Challenge**: Existing evaluations treat information retrieval as an indivisible "black box" task. However, in reality, information retrieval is a multi-step process (query understanding $\rightarrow$ relevant document identification $\rightarrow$ key information localization $\rightarrow$ fact extraction $\rightarrow$ answer organization), where errors in different steps require different diagnostic and improvement strategies.

**Goal**: Construct an atomic-level information retrieval evaluation benchmark to decompose the final answer into multiple atomic facts, separately evaluating the retrieval correctness, source attribution accuracy, and completeness of each atomic fact.

**Key Insight**: Drawing inspiration from the atomic fact decomposition idea in FActScore, this work extends it from factuality evaluation to information retrieval evaluation, and introduces two new dimensions: source attribution and granularity coverage.

**Core Idea**: Decompose the evaluation of information retrieval from the answer level to the atomic fact level, and construct a multi-dimensional fine-grained evaluation index system to achieve precise diagnosis of LLM retrieval capabilities.

## Method

### Overall Architecture
The Atomic LLM framework is divided into three modules: (1) Atomic Fact Decomposition Engine, which decomposes reference answers and model outputs into sets of atomic facts; (2) Multi-Dimensional Evaluation, which evaluates atomic facts across three dimensions: precision, attribution, and coverage; (3) Diagnostic Reporter, which generates error type distributions and capability profiles. The input consists of a query, a set of reference documents, and the LLM output, and the output consists of fine-grained evaluation scores and diagnostic reports.

### Key Designs

1. **Atomic Fact Decomposition Engine (AFDE)**:

    - **Function**: Deconstruct natural language answers into indivisible atomic fact units
    - **Mechanism**: Based on information theory, an atomic fact is defined as the "smallest unit of factual statement that can be independently verified." The decomposition process uses a three-step strategy: first, syntactic decomposition (splitting compound sentences into simple sentences), then semantic decomposition (further splitting simple sentences containing multiple facts), and finally de-duplication (merging semantically equivalent atomic facts). The decomposition quality is validated by human annotation agreement, with the decomposition engine achieving a 92% human agreement rate on the validation set. Every atomic fact is accompanied by its position marker in the original text and its dependency graph (as premise-conclusion relationships exist between some atomic facts).
    - **Design Motivation**: Directly comparing complete answers loses fine-grained error localization information. Atomic decomposition can precisely pinpoint which facts were correctly retrieved, which were omitted, and which were mistakenly generated.

2. **Three-Dimensional Evaluation Index System**:

    - **Function**: Comprehensively evaluate retrieval quality from three dimensions: precision, attribution, and coverage
    - **Mechanism**: (a) Factual Precision—how many atomic facts generated by the model are correct, $P_{atomic} = |AF_{correct}| / |AF_{generated}|$, used to detect hallucinations and erroneous facts; (b) Source Attribution—how many correct atomic facts can be traced back to specific passages in the reference documents, where the $A_{score}$ is obtained by calculating the semantic similarity between the atomic facts and the source document passages, distinguishing "obtained from retrieval results" from "obtained from parametric memory"; (c) Granularity Coverage—how many atomic facts in the reference answer are covered by the model output, $R_{atomic} = |AF_{covered}| / |AF_{reference}|$, used to evaluate the completeness of information retrieval. The comprehensive metric is calculated using the harmonic mean: $F_{atomic} = 3 \cdot P \cdot A \cdot R / (P \cdot A + A \cdot R + P \cdot R)$.
    - **Design Motivation**: Traditional F1 only focuses on precision and coverage, ignoring source attribution, which is a crucial dimension for RAG systems. Incorporating attribution evaluation helps distinguish "genuine retrieval" from "disguised memory."

3. **Capability Diagnostic Heatmap**:

    - **Function**: Generate a distribution map of the LLM's strengths and weaknesses across different capability dimensions
    - **Mechanism**: Test samples are grouped by multiple attributes: (a) Fact type (numerical, entity, relational, event); (b) Reasoning difficulty (single-hop retrieval, multi-hop reasoning, aggregative reasoning); (c) Temporal attributes (static facts vs. time-sensitive facts). Three-dimensional metrics are computed for each group to generate a heatmap similar to a confusion matrix. The model's capability profile is composed of the score vectors of all groups, allowing direct comparison of the strengths of different models.
    - **Design Motivation**: A global score cannot reflect the capability distribution of the model. Heatmap-based diagnostics can precisely guide model improvement directions.

### Loss & Training
As this work is an evaluation benchmark, it does not involve model training. The atomic fact decomposition engine uses GPT-4 as the decomposer, achieving a 92% agreement rate with human annotators. Greedy decoding (temperature=0) is used during evaluation to ensure reproducibility.

## Key Experimental Results

### Main Results

| Model | Factual Precision | Source Attribution | Granularity Coverage | F_atomic | Traditional F1 |
|------|----------|---------|-----------|---------|--------|
| GPT-4 | 87.3 | 72.1 | 79.5 | 79.0 | 82.6 |
| Claude-3 | 85.6 | 74.8 | 76.2 | 78.5 | 80.1 |
| Llama-3-70B | 79.2 | 63.5 | 71.8 | 71.0 | 76.3 |
| Mistral-7B | 72.1 | 58.2 | 65.4 | 64.7 | 71.5 |
| RAG-GPT4 | 83.5 | 89.6 | 82.1 | 85.0 | 84.2 |

### Ablation Study

| Fact Type / Reasoning Difficulty | GPT-4 Precision | GPT-4 Coverage | Description |
|----------------|------------|-----------|------|
| Numerical Facts | 71.2 | 65.8 | Weakest category, poor numerical accuracy |
| Entity Facts | 91.5 | 84.3 | Strongest category |
| Relational Facts | 85.6 | 78.2 | Moderate performance |
| Single-hop retrieval | 92.1 | 88.5 | Performs well in simple scenarios |
| Multi-hop reasoning | 78.3 | 71.2 | Significant drop when cross-document reasoning is required |
| Aggregative reasoning | 73.8 | 64.5 | Hardest scenario, requiring integration of multi-source info |

### Key Findings
- Traditional F1 rankings are basically consistent with Atomic evaluations, but Atomic metrics reveal more details: Although GPT-4 has a high overall score, its source attribution score (72.1) is far lower than its precision (87.3), indicating that a significant portion of correct answers stem from parametric memory rather than actual retrieval.
- With RAG enhancement, the source attribution score increases from 72.1 to 89.6, verifying that RAG indeed makes the model rely more on retrieval results rather than memory.
- Numerical facts and aggregative reasoning are common weaknesses for all models, with precision dropping by 20 and 13 percentage points, respectively.
- Smaller models (Mistral-7B) have significantly lower source attribution scores (58.2) than larger models, suggesting that smaller models rely more heavily on unreliable parametric memory.

## Highlights & Insights
- The introduction of the "Source Attribution" dimension is a major contribution—it distinguishes, for the first time at the evaluation level, between "genuinely retrieved from documents" and "happened to be known from memory," which is critical for evaluating the credibility of RAG systems. This metric can be generalized to the evaluation of any knowledge-intensive tasks.
- The design of the Capability Diagnostic Heatmap is highly practical, providing empirical support for model selection and targeted optimization. For example, if an application scenario primarily involves numerical facts, a user should choose the model that performs best in that specific subcategory.
- The finding that smaller models rely more on parametric memory has theoretical implications: it may be because smaller models have limited context understanding capabilities and struggle to precisely localize information within retrieval results, forcing them to fall back on parametric memory.

## Limitations & Future Work
- The atomic fact decomposition engine relies on GPT-4, creating a circular dependency issue (using an LLM to evaluate LLMs).
- The current benchmark mainly covers English; atomic fact decomposition and evaluation in multilingual scenarios still require further research.
- The threshold setting for source attribution evaluation is subjective, as different similarity thresholds can lead to different attribution judgments.
- Future work can extend atomic-level evaluation to multimodal information retrieval scenarios.

## Related Work & Insights
- **vs FActScore (Min et al., 2023)**: FActScore first proposed the atomic fact decomposition method, but focused exclusively on factuality evaluation; Atomic LLM extends the decomposition paradigm to multi-dimensional evaluation of information retrieval.
- **vs ARES (Saad-Falcon et al., 2024)**: ARES provides automated evaluation for RAG systems, but at a coarse granularity; Atomic LLM's atomic-level evaluation provides more precise diagnostic information.
- **vs RAGAS**: RAGAS also evaluates RAG systems, but utilizes answer-level metrics; Atomic LLM offers a deeper analysis of capabilities through atomic decomposition.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of atomic-level information retrieval evaluation is novel, and the three-dimensional metric design is well-structured.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive multi-model evaluation and detailed fine-grained analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The evaluation framework is clearly described, and the motivation is convincingly argued.
- **Value**: ⭐⭐⭐⭐⭐ Provides a more precise diagnostic tool for evaluating RAG systems and LLM retrieval capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)
- [\[ACL 2025\] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](air-bench_automated_heterogeneous_information_retrieval_benchmark.md)
- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [\[ACL 2025\] GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation](garage_a_benchmark_with_grounding_annotations_for_rag_evaluation.md)
- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)

</div>

<!-- RELATED:END -->
