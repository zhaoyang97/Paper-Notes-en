---
title: >-
  [Paper Note] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework
description: >-
  [ACL 2025][Information Retrieval & RAG][Retrieval-Augmented Generation] RAGEval proposes a schema-based automated evaluation dataset generation framework. It can automatically generate high-quality document-question-answer-reference quadruplets for different vertical domains (finance, law, medicine, etc.) and introduces three new evaluation metrics—Completeness, Hallucination, and Irrelevance—to rigorously assess the factual accuracy of RAG systems.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Evaluation Framework"
  - "Scenario-specific"
  - "Dataset Generation"
  - "Factual Accuracy"
date: 2026-05-08
content_hash: d73849577839f44e
---

# RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework

**Conference**: ACL 2025  
**arXiv**: [2408.01262](https://arxiv.org/abs/2408.01262)  
**Code**: [https://github.com/OpenBMB/RAGEval](https://github.com/OpenBMB/RAGEval)  
**Area**: Information Retrieval / RAG Evaluation  
**Keywords**: Retrieval-Augmented Generation, Evaluation Framework, Scenario-specific, Dataset Generation, Factual Accuracy

## TL;DR

RAGEval proposes a schema-based automated evaluation dataset generation framework. It can automatically generate high-quality document-question-answer-reference quadruplets for different vertical domains (finance, law, medicine, etc.) and introduces three new evaluation metrics—Completeness, Hallucination, and Irrelevance—to rigorously assess the factual accuracy of RAG systems.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) has become the mainstream method for enabling LLMs to utilize external knowledge, and is widely deployed in scenarios such as question answering, customer service, and knowledge management. Existing RAG evaluations mainly rely on general knowledge benchmarks (e.g., NaturalQuestions, HotpotQA).

**Limitations of Prior Work**: (1) General benchmarks fail to reflect the actual performance of RAG in specific vertical domains—knowledge structures and reasoning patterns in domains like finance and medicine differ drastically from general scenarios. (2) Manually constructing domain-specific evaluation data is extremely costly, requiring domain expert annotation. (3) Existing evaluation metrics are too coarse-grained—commonly used metrics like F1, BLEU, and ROUGE only measure textual similarity, failing to distinguish between three distinct types of errors: "incomplete answers", "hallucinations", and "introduction of irrelevant information".

**Key Challenge**: RAG systems need to be deployed and evaluated in various professional domains, but there is a lack of low-cost, high-quality, scalable domain-specific evaluation data and fine-grained evaluation metrics.

**Goal**: To design an automated framework that can automatically generate domain-specific evaluation datasets based on a small number of seed documents, while proposing fine-grained evaluation metrics to more accurately diagnose issues in RAG systems.

**Key Insight**: The authors observe that while documents in different domains vary in content, they all have domain-specific knowledge structure patterns (schemas). For instance, financial reports have fixed financial indicator structures, and medical records have diagnosis-treatment workflow structures. Extracting and reusing these schemas can guide the large-scale generation of high-quality domain documents.

**Core Idea**: Utilizing a schema-based pipeline to extract domain knowledge structures from a few seed documents, automatically generating diverse documents accordingly, constructing QA pairs and reference answers based on these documents, and evaluating them using three fine-grained metrics.

## Method

### Overall Architecture

RAGEval consists of four core components: (1) Schema Summarization—extracting domain-specific knowledge structure schemas from seed documents; (2) Document Generation—generating diverse configurations based on the schema and generating a large volume of high-quality documents accordingly; (3) QRA Generation—constructing question-reference-answer triplets based on the generated documents; (4) Evaluation Metric—using Completeness, Hallucination, and Irrelevance metrics to evaluate RAG system outputs.

### Key Designs

1. **Schema-based Document Generation Pipeline**:

    - **Function**: Automatically extracting knowledge structures from a small number of seed documents and generating large-scale domain-specific documents based on them.
    - **Mechanism**: First, the LLM analyzes seed documents to extract the domain schema, which includes key entity types, attributes, relations, and event patterns. For instance, in the financial domain, a "company-financial report-metrics" structure is extracted. Then, based on the schema, different configurations are randomly sampled (e.g., different companies, timeframes, financial situations). Finally, the LLM generates complete documents based on these configurations. This approach guarantees both document diversity and domain consistency.
    - **Design Motivation**: Prompting the LLM to generate new documents merely by mimicking seed documents often leads to "similar forms but repetitive content." The schema abstraction layer enables the generation process to maximize content diversity while maintaining domain standards.

2. **QRA (Question-Reference-Answer) Generation Mechanism**:

    - **Function**: Automatically constructing high-quality evaluation triplets based on generated documents.
    - **Mechanism**: For each document, it first identifies key facts and reasoning chains, and then constructs three types of questions: (a) single-hop factual queries—directly retrieving a specific fact in the document; (b) multi-hop reasoning—requiring synthesis of information across multiple document segments; (c) comparative analysis—requiring comparison between different entities or points in time. Each question is equipped with reference segments precisely extracted from the source documents, along with standard answers generated based on the references. The existence of reference segments allows the evaluation to trace back to specific document evidence.
    - **Design Motivation**: QA pairs without reference segments cannot distinguish whether "the model failed to retrieve" or "the model retrieved it but misunderstood it." The QRA triplet structure allows evaluation to precisely pinpoint where the RAG system fails.

3. **Three-Dimensional Fine-Grained Evaluation Metrics**:

    - **Function**: Comprehensively assessing the quality of RAG system generated answers across three orthogonal dimensions.
    - **Mechanism**: After aligning each key point of the generated answer and the reference, three metrics are calculated: **Completeness** measures how many key points in the reference are covered by the answer ($C = \frac{|KP_{covered}|}{|KP_{ref}|}$); **Hallucination** measures how many key points in the answer lack support in the reference ($H = \frac{|KP_{hallucinated}|}{|KP_{answer}|}$); **Irrelevance** measures how many key points in the answer, while not hallucinated, are irrelevant to the question ($I = \frac{|KP_{irrelevant}|}{|KP_{answer}|}$). These three metrics portray answer quality from different perspectives, providing significantly stronger diagnostic capability than a single F1 score.
    - **Design Motivation**: Traditional metrics fail to differentiate between different error types. A "complete but hallucinated" answer and an "incomplete but hallucination-free" answer might yield similar F1 scores, but they reflect fundamentally different systemic issues.

### Loss & Training

RAGEval is an evaluation framework and does not involve model training. Document generation, QRA generation, and metric calculations in the framework are all executed via LLM inference. The paper also constructs the DRAGONBall dataset as a showcase product of the framework, covering both Chinese and English data across multiple domains like finance, law, and medicine.

## Key Experimental Results

### Main Results

Evaluating the RAG performance of 9 mainstream LLMs on the DRAGONBall dataset:

| Model | Completeness ↑ | Hallucination ↓ | Irrelevance ↓ | Total Score |
|------|---------------|-----------------|---------------|------|
| GPT-4o | **78.3** | **8.2** | **5.1** | **82.5** |
| GPT-4 | 75.6 | 9.7 | 6.8 | 79.3 |
| Claude-3 | 73.2 | 10.5 | 7.2 | 77.0 |
| Llama3-8B-Instruct | 62.4 | 15.3 | 11.8 | 63.1 |
| Llama3-70B-Instruct | 69.8 | 12.1 | 8.5 | 72.6 |
| Qwen-72B | 68.5 | 13.4 | 9.1 | 70.8 |
| Mistral-7B | 55.7 | 18.6 | 14.2 | 54.3 |
| ChatGLM3-6B | 51.2 | 20.1 | 16.5 | 48.9 |
| Yi-34B | 64.3 | 14.8 | 10.3 | 66.2 |

### Data Generation Quality Evaluation

| Evaluation Dimension | RAGEval | Zero-shot Generation | One-shot Generation |
|----------|---------|---------------|---------------|
| Clarity (1-5) | **4.52** | 3.87 | 4.12 |
| Safety (1-5) | **4.78** | 4.65 | 4.71 |
| Normativeness (1-5) | **4.41** | 3.42 | 3.89 |
| Richness (1-5) | **4.35** | 3.15 | 3.68 |
| LLM-Human Agreement (κ) | 0.82 | - | - |

### Key Findings

- GPT-4o achieves the best performance across all metrics, but the open-source Llama3-70B model has reached a competitive level.
- All models fall far short of perfect scores on Completeness, indicating that "incomplete answers" is a universal shortcoming of current RAG systems.
- Smaller models (7B/8B range) exhibit significantly higher hallucination rates than larger models, validating the positive impact of model scale on factual accuracy.
- The choice of retriever significantly affects final performance: BGE-M3 outperforms BM25 markedly in Chinese scenarios, though the gap is smaller in English.
- The Cohen's κ between LLM scoring and human grading reaches 0.82, validating the feasibility of automated evaluation using LLMs.

## Highlights & Insights

- **The schema abstraction layer is a key innovation**: Instead of simply prompting the LLM to mimic seed documents, it first distills the domain knowledge structure and then generates content based on the structure. This enables data generation to maintain domain consistency while maximizing diversity. This "abstract then generate" thinking can be transferred to any scenario requiring large-scale domain data generation.
- **Three-dimensional evaluation metrics target the core pain points of RAG**: Completeness, Hallucination, and Irrelevance orthogonally decompose the three key dimensions of RAG answer quality, providing a clear direction for system diagnosis and optimization.
- **Unified framework for evaluation and data generation**: The same framework simultaneously resolves two problems: "where the evaluation data comes from" and "how to evaluate," forming a self-consistent evaluation ecosystem.

## Limitations & Future Work

- Document generation quality still depends on LLM capabilities; in highly specialized domains (e.g., latest regulations, cutting-edge medicine), it may introduce factual errors.
- Schema extraction is currently limited to text documents and does not support multimodal content such as tables and charts.
- The calculation of the three evaluation metrics itself relies on LLM judgment, entailing a risk of inconsistency with human assessment.
- The DRAGONBall dataset currently only covers three domains (finance, law, medicine); it needs to be expanded to more vertical scenarios in the future.
- The actual utility of the evaluation metrics in guiding RAG system optimization remains unexplored.

## Related Work & Insights

- **vs RAGAS**: RAGAS also evaluates RAG systems but uses general datasets and metrics such as faithfulness/relevancy. RAGEval's strength lies in its ability to automatically generate domain-specific data and more fine-grained three-dimensional metrics.
- **vs RGB/RECALL**: These benchmarks focus on retrieval accuracy evaluation, whereas RAGEval evaluates both retrieval and generation stages.
- **vs FactScore**: FactScore focuses on factual evaluation (decomposing text into atomic facts). RAGEval's Hallucination metric shares a similar philosophy but additionally introduces Completeness and Irrelevance dimensions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Schema-based data generation and the three-dimensional evaluation metrics are both original, though the overall concept represents a natural evolution of evaluation frameworks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 9 models, multiple retrievers, various hyperparameter configurations, and includes human-LLM alignment verification.
- **Writing Quality**: ⭐⭐⭐⭐ The framework description is clear and comprehensive, with informative tables and figures.
- **Value**: ⭐⭐⭐⭐⭐ Provides a systematic solution for RAG system evaluation, with code open-sourced and directly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables](mt-raig_novel_benchmark_and_evaluation_framework_for_retrieval-augmented_insight.md)
- [\[ACL 2025\] Unanswerability Evaluation for Retrieval Augmented Generation](unanswerability_evaluation_for_retrieval_augmented_generation.md)
- [\[ACL 2025\] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)
- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](../../ACL2026/information_retrieval/domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2025\] GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation](garage_a_benchmark_with_grounding_annotations_for_rag_evaluation.md)

</div>

<!-- RELATED:END -->
