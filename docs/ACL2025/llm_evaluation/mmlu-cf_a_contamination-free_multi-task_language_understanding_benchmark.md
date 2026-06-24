---
title: >-
  [Paper Note] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark
description: >-
  [ACL 2025][LLM Evaluation][Data Contamination] This paper proposes MMLU-CF, a contamination-free multi-task language understanding benchmark containing 20,000 questions. It avoids inadvertent and malicious data leakage by collecting data from broader sources and applying three decontamination rules (rephrasing questions, shuffling options, and randomly replacing options). On this benchmark, the strongest model GPT-4o achieves only 73.4% (compared to 88.0% on MMLU).
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Data Contamination"
  - "Benchmark Evaluation"
  - "MMLU"
  - "Multi-task Language Understanding"
  - "Decontamination"
date: 2026-05-08
content_hash: bd0b5792076c7be3
---

# MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark

**Conference**: ACL 2025  
**arXiv**: [2412.15194](https://arxiv.org/abs/2412.15194)  
**Code**: None (Test set is closed-source, validation set is open-source)  
**Area**: LLM Evaluation  
**Keywords**: Data Contamination, Benchmark Evaluation, MMLU, Multi-task Language Understanding, Decontamination

## TL;DR

This paper proposes MMLU-CF, a contamination-free multi-task language understanding benchmark containing 20,000 questions. It avoids inadvertent and malicious data leakage by collecting data from broader sources and applying three decontamination rules (rephrasing questions, shuffling options, and randomly replacing options). On this benchmark, the strongest model GPT-4o achieves only 73.4% (compared to 88.0% on MMLU).

## Background & Motivation

MMLU is the standard benchmark for evaluating the general knowledge understanding capability of LLMs, but it faces severe data contamination issues:

**Problem 1: Inadvertent Leakage**
- Because LLM training data is sourced extensively, it inevitably contains data from public benchmarks like MMLU.
- Models may answer questions through memorization rather than reasoning.

**Problem 2: Malicious Leakage**
- Since the benchmark is fully public, there is a risk of intentional leakage into training datasets.
- Experimental evidence: When only the questions of MMLU are provided, some LLMs can directly output the exact option letters matching the original answers, indicating that the models have memorized these questions.

**Insufficient Difficulty of MMLU**:
- Frontier models like GPT-4o, Gemini-1.5-Pro, and Claude have reached 86%-88% on MMLU.
- Nearly one-third of the questions have a difficulty level below 4 (out of 9), and a large number of simple questions account for the high scores.
- Evaluation has hit a bottleneck and cannot effectively distinguish the capabilities of next-generation models.

## Method

### Overall Architecture

The construction of MMLU-CF is a five-step process:
1. **MCQ Collection**
2. **MCQ Cleaning**
3. **Difficulty Sampling**
4. **LLMs Checking**
5. **Contamination-Free Processing**

Final output: 10,000 closed-source test questions + 10,000 open-source validation questions.

### Key Designs

**Diversification of Data Sources**:
- Raw multiple-choice questions (MCQs) are extracted from over 200 billion public web pages.
- Covers over 3,000 different website domains.
- Far more diverse than MMLU, which relies on only a few sources.
- Covers 14 domains (health, math, physics, business, chemistry, philosophy, law, engineering, etc.).

**Data Cleaning**:
- Scaled down from 2.7 million raw questions to 1.66 million through format standardization, deduplication, and filtering.
- Includes unifying option labels (A/B/C/D), filtering out non-four-choice questions, and removing low-quality brief questions.

**Difficulty Sampling**:
- GPT-4o was employed to categorize MMLU and candidate questions into difficulty levels from 0 to 9.
- Sampling was conducted using a normal distribution centered at difficulty level 6 to ensure the overall difficulty is higher than MMLU.
- 1.66 million questions $\rightarrow$ 50,000 questions.

**Three-Model Verification**:
- Three LLMs (GPT-4o, Gemini, Claude) were used to verify questions across both quality and safety dimensions.
- Quality dimensions: contextual clarity, logical consistency, factual accuracy, option mutual exclusivity, and presence of correct answers.
- Safety dimensions: free of hate expression, pornography, self-harm, and violence.
- Only questions with an average score of $>4$ (out of 5) across all three models were retained.
- 50,000 questions $\rightarrow$ 20,000 questions.

**Three Decontamination Rules**:

1. **Rule 1: Rephrase Question**

    - Reduces the models' dependence on questions encountered during training.
    - Alters the phrasing of the question without changing its core semantic meaning.

2. **Rule 2: Shuffle Choices**

    - Prevents models from answering by memorizing the option order.
    - Options like "None of the above" are kept at the very end.

3. **Rule 3: Random Replace Choices**

    - Replaces an arbitrary option with "None of the other choices" with a 50% probability.
    - If the correct option is replaced, it remains valid but requires more reasoning.
    - If an incorrect option is replaced, it serves as a distractor.

**Test Set / Validation Set Design**:
- Closed-source test set prevents malicious contamination.
- Open-source validation set facilitates independent verification.
- Both sets exhibit similar difficulty and category distributions.
- If the gap between the validation set and the test set widens in the future, it indicates that the validation set is being contaminated.

### Loss & Training

As this paper introduces an evaluation benchmark, it does not involve model training. Evaluation details:
- Standardized evaluation is conducted using the OpenCompass platform.
- Supports both 5-shot and 0-shot evaluations.
- Users can submit Hugging Face models or APIs via the project homepage for evaluation.

## Key Experimental Results

### Main Results

**Performance of over 40 models on MMLU-CF (5-shot Test %)**:

| Category | Model | MMLU | MMLU-CF (5-shot) | Decline |
|------|------|------|------------------|------|
| API | GPT-4o | 88.0 | 73.4 | -14.6 |
| API | GPT-4-Turbo | 86.5 | 70.4 | -16.1 |
| API | GPT-4o-mini | 81.8 | 65.5 | -16.3 |
| Large | Qwen2.5-72B-instruct | 85.3 | 71.6 | -13.7 |
| Large | Llama-3.3-70B-instruct | 86.3 | 68.8 | -17.5 |
| Medium | Qwen2.5-32B-instruct | 83.9 | 69.7 | -14.2 |
| Medium | Phi-4-14B | 84.8 | 67.8 | -17.0 |
| Small | Qwen2.5-7B-instruct | 75.4 | 61.3 | -14.1 |
| Mini | Phi-3.5-mini-3.8B | 69.1 | 57.9 | -11.2 |

The scores of all models on MMLU-CF are significantly lower than those on MMLU, with an average drop of about 14-17 percentage points.

**Consistency Between Validation and Test Sets**:
- For the vast majority of models, the performance difference between the two sets is $<1\%$ (the $\Delta$ column).
- This indicates that the difficulty and distribution of the two sets are indeed similar.
- It establishes a baseline to monitor whether the validation set gets contaminated in the future.

### Ablation Study

**Cumulative Effect of the Three Decontamination Rules (0-shot)**:

| Rule Combination | GPT-4o | GPT-3.5 | Llama-3.1-8B |
|----------|--------|---------|--------------|
| No Rules | 79.8 | 65.3 | 63.8 |
| +Rule 1 (Rephrase) | 78.6 | 63.1 | 62.3 |
| +Rule 1+2 (+Shuffle) | 77.9 | 62.8 | 61.8 |
| +Rule 1+2+3 (+Replace) | 73.4 | 58.2 | 57.1 |

Each rule steadily degrades the performance of the models:
- Rule 1 drops GPT-4o by 1.2 points, with weaker models dropping even more.
- Rule 3 has the most pronounced effect (GPT-4o drops by 4.5 points), demonstrating that random option replacement is the most effective decontamination strategy.

### Key Findings

1. **MMLU is severely contaminated**: The performance of all models on MMLU-CF decrease significantly, confirming that MMLU evaluation results are drastically inflated by data contamination.
2. **GPT-4o remains the strongest model**: It retains the lead even after decontamination, though its lead has narrowed (88% $\rightarrow$ 73.4%).
3. **Parameter-efficient models perform exceptionally well**: Phi-4-14B (67.8%) outperforms several larger models like Qwen2-72B (63.7%), demonstrating the importance of structural efficiency.
4. **Qwen2.5 series performs best overall**: Leading in almost all size categories.
5. **Small difference between 0-shot and 5-shot**: Most models show comparable performance under both settings, indicating that MMLU-CF primarily tests genuine knowledge understanding rather than few-shot adaptation.

## Highlights & Insights

1. **Systematic Decontamination Scheme**: Explicitly distinguishes between inadvertent and malicious leakage, addressing them recursively with rules and closed-source strategies.
2. **Cross-verification via Three Models**: Employs GPT-4o, Gemini, and Claude to jointly evaluate questions, avoiding individual model biases.
3. **Contamination Monitoring Mechanism**: The open-source validation + closed-source test set paradigm enables tracking validation set contamination over time by measuring the performance gap between them.
4. **Difficulty Calibration**: References the difficulty distribution of MMLU and intentionally raises the overall difficulty, making the benchmark more discriminative.
5. **Ultra-large Scale Data Sources**: Filtered from 200 billion documents across 3,000+ domains, significantly reducing the probability of data being pre-ingested in the training set.

## Limitations & Future Work

- Decontamination rules might inadvertently alter the cognitive demands of the questions (especially Rule 3's random replacement, which could shift the difficulty distribution).
- Although the closed-source test set prevents data leakage, it also restricts independent validation and reproduction by the academic community.
- Whether the 14 domains provide sufficiently comprehensive coverage remains questionable (e.g., missing programming-related fields).
- Relying on GPT-4o for difficulty classification and quality checking introduces dependency on a specific model.
- The differential impacts of decontamination processing on different question types (factual recall vs. reasoning) have not been analyzed.

## Related Work & Insights

- **MMLU** (Hendrycks et al.) and **MMLU-Pro** (Wang et al., 2024) are direct preceding works.
- **GSM1K** (Zhang et al., 2024) adopts a similar strategy to construct a contamination-free version for GSM8K.
- **LiveBench** (White et al., 2024) ensures zero contamination through constant updates, but at a high cost.
- **LatestEval** (Li et al., 2024) dynamically generates evaluation questions from the latest texts.
- Insights: Resolving the benchmark contamination problem fundamentally may require combining dynamic generation and closed-source evaluation.

## Rating

- **Novelty**: ⭐⭐⭐ — The decontamination method is intuitive but executed systematically.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated over 40 models with comprehensive and detailed analysis.
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses the core pain point in LLM evaluation; the test set is actively maintained.
- **Writing Quality**: ⭐⭐⭐⭐ — Structured clearly with abundant data and standard figures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)

</div>

<!-- RELATED:END -->
