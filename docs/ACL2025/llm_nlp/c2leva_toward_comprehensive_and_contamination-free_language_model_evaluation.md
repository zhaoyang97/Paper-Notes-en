---
title: >-
  [Paper Note] C²LEVA: Toward Comprehensive and Contamination-Free Language Model Evaluation
description: >-
  [ACL 2025 (Findings)][LLM (Other)][Language Model Evaluation] Proposes C²LEVA, a comprehensive Chinese-English bilingual evaluation benchmark containing 22 tasks. It systematically prevents data contamination through fully automated test data renewal and data protection mechanisms, demonstrating its effectiveness across 15 open-source and closed-source models.
tags:
  - "ACL 2025 (Findings)"
  - "LLM (Other)"
  - "Language Model Evaluation"
  - "Data Contamination"
  - "Bilingual Benchmark"
  - "Test Data Renewal"
  - "Evaluation Credibility"
date: 2026-05-08
content_hash: b9d455bbeb900f79
---

# C²LEVA: Toward Comprehensive and Contamination-Free Language Model Evaluation

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2412.04947](https://arxiv.org/abs/2412.04947)  
**Code**: Yes (Project Page)  
**Area**: LLM/NLP  
**Keywords**: Language Model Evaluation, Data Contamination, Bilingual Benchmark, Test Data Renewal, Evaluation Credibility

## TL;DR

Proposes C²LEVA, a comprehensive Chinese-English bilingual evaluation benchmark containing 22 tasks. It systematically prevents data contamination through fully automated test data renewal and data protection mechanisms, demonstrating its effectiveness across 15 open-source and closed-source models.

## Background & Motivation

**Background**: LLM evaluation is a core driving force behind model development. Current mainstream benchmarks such as MMLU, C-Eval, and SuperGLUE cover a wide range of tasks and capability dimensions, serving as standard tools to measure model performance.

**Limitations of Prior Work**: Data contamination poses the most severe threat to current LLM evaluation. This is because: (1) pre-training data is usually not disclosed, making it impossible to verify whether the model has "seen" the benchmark data; (2) many test sets remain static over long periods, making them easy to scrape and leakage into training data; (3) even inadvertent contamination (e.g., training corpora accidentally containing sources of benchmark data) can cause artificially inflated evaluation scores. Furthermore, the number of existing Chinese evaluation benchmarks is limited, and evaluations of bilingual Chinese-English capabilities are particularly scarce.

**Key Challenge**: The essence of evaluation requires models to demonstrate capabilities on "unseen" data, but current static benchmarks cannot guarantee this premise. Once the test data is included in the model's training data, the evaluation loses its meaning, while verifying whether contamination exists is extremely difficult.

**Goal**: To design a comprehensive evaluation benchmark that fundamentally solves the data contamination problem, while providing comprehensive support for bilingual capabilities and covering a wide range of task types.

**Key Insight**: The authors argue that combating data contamination should not rely solely on post-hoc detection. Instead, a systematic prevention mechanism should be established during the benchmark design phase—automatically renewing test data periodically and implementing data protection upon release.

**Core Idea**: Fundamentally resolve data contamination through a fully automated test data renewal pipeline (ensuring new data is used for each evaluation) and strict data protection strategies (preventing test data from leaking into training sets).

## Method

### Overall Architecture

The design of C²LEVA consists of two core pillars: (1) **Comprehensive Evaluation System**: 22 tasks covering 5 capability dimensions (knowledge, reasoning, language understanding, language generation, and safety), with each task targeting a specific LLM application or capability; (2) **Systematic Contamination Prevention**: A fully automated data renewal pipeline and multi-level data protection mechanisms to ensure test data does not leak into the model training process.

### Key Designs

1. **22-Task Comprehensive Evaluation Suite (Comprehensive Task Suite)**:

    - **Function**: Comprehensively evaluate LLM performance across different dimensions.
    - **Mechanism**: The 22 tasks are organized into 5 capability dimensions: (a) **Knowledge**: world knowledge QA, professional knowledge test, and common-sense reasoning; (b) **Reasoning**: logical reasoning, mathematical reasoning, and code generation and understanding; (c) **Language Understanding**: reading comprehension, sentiment analysis, natural language inference, and info extraction; (d) **Language Generation**: text summarization, translation, creative writing, and dialogue generation; (e) **Safety**: toxic content detection, bias evaluation, and factuality verification. Every task is bilingual (Chinese and English), ensuring the assessment of the models' cross-lingual capabilities.
    - **Design Motivation**: Existing benchmarks often cover only a subset of capability dimensions, failing to comprehensively reflect the overall capability of models. The bilingual design fills the gap in comparative Chinese-English evaluation.

2. **Fully Automated Data Renewal Pipeline (Automated Data Renewal Pipeline)**:

    - **Function**: Automatically generate brand new test data periodically to render obsolete test data invalid.
    - **Mechanism**: A data generator is designed for each task to automatically construct new test samples from source data (e.g., news, Wikipedia, academic papers). The specific pipeline includes: (a) fetching raw material from constantly updating data sources; (b) automatically constructing questions using task-specific templates and rules; (c) filtering for quality to ensure the difficulty distribution of new data is consistent with the old; (d) automated correctness validation. The entire process requires no human intervention and can be updated monthly or on demand.
    - **Design Motivation**: Static test sets are the root cause of data contamination—as long as the test set remains unchanged, there is always a risk of leakage. Automated renewal fundamentally solves this problem.

3. **Multi-Level Data Protection**:

    - **Function**: Prevent test data leakage during benchmark release and usage.
    - **Mechanism**: Implement three levels of protection: (a) **Release Level**: Raw text of the test data is not made public; instead, evaluations are conducted via API-based answer submission; (b) **Technical Level**: Data encryption and access control restrict direct access to the raw test data; (c) **Temporal Level**: The specific questions used in each evaluation round can only be viewed after the round is completed (at which point the next round of questions is already prepared).
    - **Design Motivation**: Data renewal alone is insufficient; it is also necessary to prevent test data from being scraped or leaked during release.

### Evaluation Methods

Supports two evaluation modes: (1) Multiple-choice mode: selects answers by comparing the probabilities of each option; (2) Generation mode: allows the model to generate answers directly, scored by rules or LLMs. These two modes ensure that the evaluation is fair to different types of models.

## Key Experimental Results

### Main Results

Overall performance of 15 models on C²LEVA (average score of each dimension):

| Model | Knowledge | Reasoning | Language Understanding | Language Generation | Safety | Overall |
|------|------|------|---------|---------|------|------|
| GPT-4 | 82.3 | 78.5 | 85.1 | 79.8 | 88.2 | 82.8 |
| Claude-3 Opus | 80.1 | 76.2 | 83.4 | 78.5 | 86.7 | 81.0 |
| GPT-3.5-Turbo | 71.2 | 65.8 | 74.3 | 71.0 | 78.5 | 72.2 |
| LLaMA-3-70B | 74.5 | 69.3 | 76.8 | 72.1 | 75.3 | 73.6 |
| Qwen-72B | 76.8 | 71.5 | 78.2 | 74.3 | 80.1 | 76.2 |
| Yi-34B | 68.3 | 62.1 | 70.5 | 66.8 | 72.4 | 68.0 |
| ChatGLM-4 | 72.1 | 66.5 | 73.8 | 70.2 | 77.3 | 72.0 |
| Mistral-7B | 58.2 | 52.3 | 61.4 | 55.7 | 63.8 | 58.3 |

### Ablation Study

| Experimental Configuration | Score Change | Notes |
|---------|---------|------|
| Evaluating twice on the same data version | <1% | High evaluation stability |
| Old test set vs. New test set | -3~5% | New data is indeed harder (unmemorized) |
| Scores on known leaked datasets | +8~15% | Contamination leads to artificially high scores |
| C²LEVA standard evaluation | Baseline | True capability after contamination prevention |
| Chinese vs. English average score difference | 5~12% | Most models perform worse in Chinese than English |

### Key Findings

- **Data contamination indeed exists and has a significant impact**: On known contaminated benchmarks, model scores are 8–15 percentage points higher than on C²LEVA, proving the necessity of contamination prevention.
- **Closed-source models still lead, but the gap is narrowing**: GPT-4 achieves the highest overall score, but open-source models like Qwen-72B are close to the level of GPT-3.5.
- **Chinese capability is generally weaker than English**: Almost all models score lower on Chinese tasks than on English tasks, with a gap of 5–12 percentage points.
- **Reasoning capability is the biggest bottleneck**: Among all capability dimensions, reasoning (especially mathematical reasoning and code understanding) yields the lowest scores.
- **Vast differences in the safety dimension**: Some models perform strongly overall but have relatively low safety scores, indicating that capability and safety are not fully positively correlated.
- **Data renewal mechanism is effective**: The score difference on the old and new versions of the test sets (3–5%) combined with the discrepancy under contaminated scenarios (8–15%) validates the effectiveness of the renewal mechanism.

## Highlights & Insights

- Systematically resolving data contamination at the benchmark design level, rather than relying on post-hoc detection, is a more fundamental approach.
- Bilingual Chinese-English coverage across 22 tasks makes it one of the most comprehensive bilingual evaluation benchmarks today.
- The fully automated data renewal pipeline gives the benchmark "self-renewal" capability, allowing it to be used indefinitely in theory.
- Evaluation results for 15 mainstream models provide valuable references for the community.

## Limitations & Future Work

- The publication in Findings of ACL 2025 suggests that reviewers may have had reservations regarding its completeness or certain other aspects.
- The quality of fully automatically generated test data might not match that of elaborately human-annotated data.
- Automatic evaluation metrics for certain tasks (such as creative writing) are inherently less reliable.
- Future work can explore more languages, additional capability dimensions, and dynamic difficulty adjustment.

## Related Work & Insights

- Closely related to Chinese evaluation benchmarks like C-Eval and CMMLU, but the contamination prevention mechanism of C²LEVA is its unique contribution.
- Shares the "dynamic evaluation" philosophy with LiveBench and Chatbot Arena, but C²LEVA achieves a higher level of automation.
- The design of the data renewal pipeline can be adopted by other benchmarks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The systematic design of the contamination prevention mechanism is novel, with automated data renewal being a highlight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Large-scale evaluation across 15 models and 22 tasks; the contamination detection experiments are persuasive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation of the problem and systematic description of the method.
- **Value**: ⭐⭐⭐⭐ — Provides a practical solution to the credibility issue of LLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference](can_llms_reason_about_program_semantics_a_comprehensive_evaluation_of_llms_on_fo.md)
- [\[ACL 2025\] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)
- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[ACL 2025\] Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)

</div>

<!-- RELATED:END -->
