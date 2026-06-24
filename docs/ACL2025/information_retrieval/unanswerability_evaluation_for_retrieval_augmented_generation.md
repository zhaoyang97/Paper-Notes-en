---
title: >-
  [Paper Note] Unanswerability Evaluation for Retrieval Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG Evaluation] UAEval4RAG proposes a comprehensive evaluation framework for evaluating retrieval-augmented generation (RAG) systems on unanswerable queries. It defines six categories of unanswerability, automatically synthesizes test data based on any given knowledge base, and evaluates system refusal capabilities. Experiments reveal that no single configuration optimizes performance for both answerable and unanswerable queries across…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG Evaluation"
  - "Unanswerable Queries"
  - "Refusal Ability"
  - "Knowledge Base Customization"
  - "Evaluation Framework"
date: 2026-05-08
content_hash: 8981e98c38b31dc5
---

# Unanswerability Evaluation for Retrieval Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2412.12300](https://arxiv.org/abs/2412.12300)  
**Code**: [https://github.com/SalesforceAIResearch/Unanswerability_RAGE](https://github.com/SalesforceAIResearch/Unanswerability_RAGE)  
**Area**: Information Retrieval  
**Keywords**: RAG Evaluation, Unanswerable Queries, Refusal Ability, Knowledge Base Customization, Evaluation Framework

## TL;DR

UAEval4RAG proposes a comprehensive evaluation framework for evaluating retrieval-augmented generation (RAG) systems on unanswerable queries. It defines six categories of unanswerability, automatically synthesizes test data based on any given knowledge base, and evaluates system refusal capabilities. Experiments reveal that no single configuration optimizes performance for both answerable and unanswerable queries across all datasets.

## Background & Motivation

1. **Background**: Retrieval-Augmented Generation (RAG) has become a key technology for improving the reliability of LLMs. Existing evaluation frameworks (such as RAGAS, ARES, RGB) primarily focus on the accuracy and relevance of **answerable queries**.

2. **Limitations of Prior Work**: These frameworks largely overlook a critical capability: **properly refusing unanswerable requests**. In real-world scenarios, users frequently ask questions that cannot be answered by the knowledge base (e.g., due to insufficient information, false presuppositions, or out-of-scope topics), where the system should refuse rather than hallucinate.

3. **Key Challenge**: Existing unanswerability benchmarks (e.g., Brahman et al.) target LLMs in isolation using generic unanswerable requests, which cannot be customized to specific knowledge bases. Consequently, refusals often stem from a failure to retrieve relevant context rather than a genuine understanding that the request should not be answered. Additionally, the few studies evaluating RAG refusal capabilities (e.g., Ming et al.) focus only on a single type of unanswerable request.

4. **Goal**: How to automatically generate multi-category, high-quality unanswerable datasets for any knowledge base and systematically evaluate the refusal capability of RAG systems?

5. **Key Insight**: Inspired by Brahman et al., this work defines six unanswerability categories (ranging from underspecified to safety-concerned) and designs specialized synthesis pipelines and validation mechanisms for each category.

6. **Core Idea**: To construct an end-to-end evaluation framework capable of automatically synthesizing six categories of unanswerable queries based on any knowledge base, and automatically evaluating the refusal and acceptance rates of RAG systems.

## Method

### Overall Architecture

Given a knowledge base, the UAEval4RAG framework operates in three steps: (1) automatically synthesizing a dataset of six unanswerable query categories based on the knowledge base content; (2) feeding these queries into the target RAG system to obtain responses; and (3) employing LLM-based metrics to evaluate the refusal quality of the responses. All three steps are fully automated, requiring users only to provide the knowledge base.

### Key Designs

1. **Six categories of unanswerable queries taxonomy**:

    - Function: Comprehensive coverage of unanswerable scenarios that RAG systems may encounter.
    - Mechanism: Six categories are defined with associated difficulty levels: (1) Underspecified (missing critical information, Hard): e.g., "Are pets allowed?" without specifying the location; (2) False-presupposition (false premise, Easy): e.g., assuming Disney World is in Georgia; (3) Nonsensical (senseless, Medium): e.g., spelling errors or logical confusion; (4) Modality-limited (modality restriction, Medium): e.g., asking a text-based system to show images; (5) Safety-concerned (safety risk, Medium): e.g., harmful requests highly relevant to the knowledge base; (6) Out-of-Database (out-of-scope, Easy): e.g., questions relevant to the knowledge base but with answers not contained within it.
    - Design Motivation: Different categories test different dimensions of the RAG system's refusal capabilities, from understanding user intent to identifying safety risks.

2. **Automatic synthesis pipeline**:

    - Function: To automatically generate high-quality unanswerable test data for any knowledge base.
    - Mechanism: The first five categories (Underspecified to Safety-concerned) share a pipeline where a knowledge base chunk is randomly selected, and an LLM is prompted via category definitions and in-context learning (ICL) examples to generate a request and an explanation. This is followed by an LLM-based verification to ensure compliance with category criteria; only verified instances are included. The Out-of-Database category uses a separate pipeline where key phrases are extracted from the knowledge base, related news is crawled to construct Q&A pairs, and retrieval verification ensures the answers are indeed absent from the knowledge base.
    - Design Motivation: The two-step generate-and-verify mechanism ensures data quality; Out-of-Database queries are handled separately because they require ensuring the questions are highly relevant to the knowledge base but the answers are strictly non-existent in it.

3. **Three LLM-based evaluation metrics**:

    - Function: Quantitatively evaluating the capability of RAG systems to handle unanswerable queries.
    - Mechanism: (1) **Unanswered Ratio** (objective metric) measures the proportion of queries refused by the system under consistent definitions; (2) **Acceptable Ratio** (subjective metric) assesses whether responses align with human preferences based on specific standards for each category (e.g., the acceptable standard for Underspecified queries is to refuse, ask for clarification, or answer from multiple perspectives, while Modality-limited requires stating that the modality is unsupported); (3) **Joint Score** = $w_1 \times \text{Correctness} + w_2 \times \text{Acceptable Ratio}$ balances the performance on answerable and unanswerable queries.
    - Design Motivation: A single metric cannot provide a comprehensive evaluation. Both the refusal rate and refusal quality must be measured simultaneously. The Joint Score allows users to adjust weights according to their application needs.

### Loss & Training

This work presents an evaluation framework rather than a training method, and thus does not involve a loss function. The core technical contributions lie in the design of the synthesis pipeline and evaluation metrics.

## Key Experimental Results

### Main Results

| Dataset | Best Configuration (Embedding+Retrieval) | Answerable Correctness | Unanswerable Acceptable Ratio | Joint Score |
|--------|-------------------------------|--------------------|-----------------------|-------------|
| TriviaQA | Cohere+Vector+None | 88.0% | 54.8% | 78.04% |
| TriviaQA | BGE+Vector+Cohere | 87.6% | 55.5% | 77.97% |
| MuSiQue | Cohere+Vector+Cohere+HyDE | 48.0% | 62.7% | 52.41% |
| MuSiQue | BGE+Ensemble+Cohere | 47.2% | 62.8% | 51.88% |

**Impact of Prompt Design** (TriviaQA, GPT-4o):

| Prompt | Correctness | Acceptable Ratio | Joint Score |
|--------|-------------|-------------------|-------------|
| Default | 88.0% | 53.2% | 77.56% |
| Prompt #1 | 88.4% | 84.3% | 87.20% |
| Prompt #2 | 74.8% | 83.0% | 77.26% |

### Ablation Study

| LLM Evaluator | Unanswered Accuracy | Unanswered F1 | Acceptable Accuracy | Acceptable F1 |
|-----------|--------------------|--------------|--------------------|--------------|
| GPT-4o | 82.0% | 76.9% | 84.0% | 85.2% |
| Claude 3.5 Sonnet | 84.0% | 76.9% | 81.3% | 83.1% |
| Deepseek-R1 | 84.4% | 76.7% | 83.3% | 86.0% |

Synthesis Quality Verification: TriviaQA dataset achieved a 92% human annotation accuracy with an inter-annotator agreement of 0.85; MuSiQue achieved a 92% accuracy with an agreement of 0.88.

### Key Findings

- **No "one-size-fits-all" configuration**: Due to differences in knowledge base distributions, no single combination of components can simultaneously optimize performance on both answerable and unanswerable queries across all datasets.
- **Prompt design is critical**: The optimal prompt could increase the unanswerable Acceptable Ratio by around 80% while having minimal impact on answerable correctness.
- **LLM choice has a significant impact**: Claude 3.5 Sonnet improves on Correctness by 0.4% and on unanswerable Acceptable Ratio by 10.4% compared to GPT-4o.
- **Underspecified category is the most challenging**: Models struggle to judge whether the information is "sufficient," leading to the lowest refusal rate.
- **Varying difficulty across different knowledge bases**: Highly concentrated knowledge distributions make it easier to synthesize Out-of-Database requests.

## Highlights & Insights

- **Idea of knowledge base-customized evaluation**: Instead of using generic benchmarks, test data is dynamically generated according to the specific knowledge base. This concept is transferable to any NLP system requiring customized testing.
- **Comprehensive six-category coverage**: Ranging from information completeness (Underspecified) to safety (Safety-concerned), and from logical correctness (Nonsensical, False-presuppositions) to system capacity (Modality-limited), virtually all dimensions of RAG refusal capabilities are covered.
- **Design of the Joint Score**: Highly practical as it allows users to customize the weight balance between answerability and refusal rate based on their specific application scenarios.

## Limitations & Future Work

- Currently, only English datasets have been evaluated; performance on unanswerability may vary in multilingual scenarios.
- The synthesis pipeline depends on the generative and verification capabilities of LLMs, which may require domain adaptation for highly specialized domains (e.g., law, medicine).
- Although the six categories are comprehensive, they might lack fine-grained specificity; for instance, Underspecified could be further broken down into missing time, location, or subject.
- The work only evaluates refusal capabilities and does not explore methods to improve the refusal strategies of RAG systems.

## Related Work & Insights

- **vs RAGAS (Es et al.)**: RAGAS only evaluates retrieval relevance and generation quality, completely disregarding unanswerable scenarios. UAEval4RAG fills this critical gap.
- **vs Ming et al.**: Ming et al. only evaluate whether an LLM can refuse a single type of unanswerable request within inconsistent contexts; UAEval4RAG covers six categories and synthesizes data based on the original knowledge base.
- **vs Brahman et al.**: Brahman et al. evaluate the unanswerability capabilities of LLMs in isolation using generic requests; UAEval4RAG specifically targets RAG systems with data highly relevant to the knowledge base.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic RAG unanswerability evaluation framework, supported by a theoretically grounded six-category taxonomy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive experiments involving 27 component combinations × 3 LLMs × 3 prompts × 4 datasets.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clearly defined taxonomies.
- Value: ⭐⭐⭐⭐⭐ Represents a vital supplement to the completeness of RAG system evaluations, and is directly applicable to quality assurance for industrial RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)
- [\[ACL 2025\] MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables](mt-raig_novel_benchmark_and_evaluation_framework_for_retrieval-augmented_insight.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](gear_generation_augmented_retrieval.md)
- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)

</div>

<!-- RELATED:END -->
