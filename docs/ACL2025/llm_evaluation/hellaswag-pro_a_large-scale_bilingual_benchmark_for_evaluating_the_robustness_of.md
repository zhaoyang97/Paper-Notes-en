---
title: >-
  [Paper Note] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning
description: >-
  [ACL 2025][LLM Evaluation][commonsense reasoning] This work constructs HellaSwag-Pro, the first large-scale bilingual (Chinese-English) benchmark for evaluating the robustness of LLMs in commonsense reasoning. By generating 11,200 question variants from 1,600 original questions across 7 reasoning forms, systematic evaluation on 41 LLMs reveals that all models fall far short of robust commonsense reasoning—the average accuracy for negation transformation is only 9.01%…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "commonsense reasoning"
  - "robustness evaluation"
  - "bilingual benchmark"
  - "question variants"
  - "HellaSwag"
date: 2026-05-08
content_hash: 143b77e4d9d3d8d5
---

# HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning

**Conference**: ACL 2025  
**arXiv**: [2502.11393](https://arxiv.org/abs/2502.11393)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: commonsense reasoning, robustness evaluation, bilingual benchmark, question variants, HellaSwag

## TL;DR

This work constructs HellaSwag-Pro, the first large-scale bilingual (Chinese-English) benchmark for evaluating the robustness of LLMs in commonsense reasoning. By generating 11,200 question variants from 1,600 original questions across 7 reasoning forms, systematic evaluation on 41 LLMs reveals that all models fall far short of robust commonsense reasoning—the average accuracy for negation transformation is only 9.01%, highlighting a significant performance gap between models and humans.

## Background & Motivation

**High scores of LLMs on commonsense reasoning benchmarks do not imply true understanding**. GPT-4o can correctly answer standard HellaSwag questions like "A lady walks to a barbell, bends down, and grabs the bar, next she will...", but frequently fails on variants of the same knowledge, such as reversed conversion (inferring the context from the results) and negation transformation. This raises a core question: Do models truly understand commonsense knowledge, or do they merely memorize specific expression patterns?

**Limitations of Prior Work**: Most commonsense reasoning benchmarks (HellaSwag, CommonsenseQA, PIQA, etc.) only test accuracy under fixed reasoning formats. Few studies addressing robustness examine only a single simple variant (e.g., question paraphrasing), lacking a systematic evaluation of diverse and complex reasoning forms. Furthermore, existing benchmarks are almost entirely in English, failing to assess the commonsense reasoning capabilities of Chinese LLMs.

**Core Idea**: If a model truly understands a piece of commonsense knowledge, it should generalize to various reasoning forms of that knowledge (forward reasoning, reverse reasoning, causal reasoning, negation reasoning, etc.). Therefore, designing 7 types of variants spanning from memorization to high-order cognition can rigorously quantify the robustness of models in commonsense reasoning. To this end, the authors first construct a Chinese HellaSwag dataset with 12,000 questions, and then generate 7 variants for both the Chinese and English versions to form the HellaSwag-Pro benchmark.

## Method

### Overall Architecture

The framework consists of two main steps: (1) Construction of **Chinese HellaSwag**—employing a two-stage pipeline (initial data generation + adversarial hard exampler replacement) to yield 12,000 Chinese commonsense reasoning multiple-choice questions across 56 fine-grained categories; (2) Designing 7 question variants on both the Chinese and English versions of HellaSwag, validated thoroughly by humans to generate HellaSwag-Pro (11,200 variant questions derived from 1,600 original questions).

### Key Designs

1. **Two-Stage Construction of Chinese HellaSwag**:

    - **Function**: Establish a Chinese commonsense reasoning benchmark comparable in difficulty to the English HellaSwag.
    - **Mechanism**: Design a fine-grained taxonomy consisting of 56 categories ($7 \text{ major categories} \times 8 \text{ subcategories}$). In the first stage, Qwen-Max is utilized via in-context learning to overgenerate contexts and options, followed by human annotation to filter and retain high-quality samples (filtering from 12,960 to 12,000). The second stage performs adversarial filtering: overly simple distractors are rewritten using a generator LLM and verified using multiple discriminator LLMs—if a new option successfully misleads the discriminators, it replaces the original. This iterative process aligns the difficulty of the Chinese version with the English version (replacing 2,451 samples in total).
    - **Design Motivation**: To prevent the Chinese dataset from being too simple to be discriminative, and to ensure cross-lingual comparability through adversarial filtering.

2. **Design of 7 Question Variants**:

    - **Function**: Comprehensively test commonsense reasoning robustness across multiple cognitive dimensions (corresponding to Bloom's taxonomy).
    - **Mechanism**: (a) **Question Paraphrasing**—rephrasing the context and the correct answer while keeping the semantics intact; (b) **Reversed Conversion**—inferring the original context from the result; (c) **Causal Reasoning**—merging context and answer to ask for the cause of the behavior; (d) **Sentence Ordering**—shuffling the sentences and requiring the correct order; (e) **Scenario Rewriting**—making minimal modifications to the context so that an originally incorrect option becomes correct; (f) **Negation Transformation**—introducing negative semantics to make the least likely option correct; (g) **Critical Testing**—removing crucial information to render all original options invalid, where the correct answer is "None of the above". After generation using Qwen-Max, intensive human verification is conducted, filtering the initial 24,260 variants down to 11,200.
    - **Design Motivation**: The 7 variants cover the entire cognitive hierarchy from memorization (Question Paraphrasing) to high-order cognition (Critical Testing, Causal Reasoning).

3. **Evaluation Metric System**:

    - **Function**: Quantify the robustness of commonsense reasoning from multiple perspectives.
    - **Mechanism**: Includes OA (Original Accuracy), ARA (Average Robust Accuracy across all variants), RLA (Relative accuracy Loss of ARA = OA $-$ ARA), and **CRA (Complete Robustness Accuracy—the question is considered truly understood only if all 7 variants are answered correctly)**. Meanwhile, 9 prompting strategies are designed (zero-shot and few-shot variants of Direct, CN-CoT, EN-CoT, CN-XLT, EN-XLT, etc.).
    - **Design Motivation**: CRA serves as the most rigorous robustness metric—if a model "truly understands" a concept, it should correctly answer all reasoning forms of the same knowledge.

## Key Experimental Results

### Main Results

**Closed-source Model Performance** (Direct prompt strategy):

| Model | Chinese OA (%) | Chinese ARA (%) | Chinese CRA (%) | English OA (%) | English ARA (%) | English CRA (%) |
|------|-----------|------------|------------|-----------|------------|------------|
| Human | 96.41 | 97.79 | 92.03 | 95.56 | 96.04 | 90.02 |
| GPT-4o | 91.37 | 81.97 | **75.55** | 88.63 | 70.17 | **63.06** |
| Claude-3.5 | **95.37** | 80.15 | 75.04 | 85.11 | 66.02 | 57.20 |
| Gemini-1.5-Pro | 90.62 | 78.36 | 70.48 | 87.75 | 60.74 | 58.27 |
| Qwen-Max | 93.50 | 84.82 | 78.91 | 87.60 | 62.61 | 59.65 |

**Representative Results of Open-source Models**:

| Model | Chinese OA (%) | Chinese CRA (%) | English OA (%) | English CRA (%) | Average CRA (%) |
|------|-----------|------------|-----------|------------|------------|
| Qwen2.5-72B | 70.87 | 39.64 | 72.00 | 35.12 | 37.38 |
| Llama3-70B | 65.75 | 32.70 | 72.50 | 30.63 | 31.67 |
| Mixtral-8x22B | 66.00 | 34.32 | 72.12 | 30.61 | 32.47 |
| Yi1.5-34B | 71.00 | 38.09 | 71.00 | 29.91 | 34.00 |
| DeepSeek-67B | 71.50 | 35.89 | 71.37 | 29.71 | 32.80 |
| Random | 25.00 | — | 25.00 | 0.0015 | 0.0015 |

### Ablation Study

**Difficulty Ranking of Various Variants**:

| Variant Type | Design Principle | Average Accuracy | Difficulty |
|---------|---------|----------|------|
| Question Paraphrasing | Rephrasing expression while keeping semantics | Highest | Easiest |
| Reversed Conversion | Result $\rightarrow$ Context reasoning | Medium | Medium |
| Causal Reasoning | Asking for cause after merging | Medium | Medium |
| Sentence Ordering | Shuffling sentence order | Medium | Medium |
| Scenario Rewriting | Modifying context to flip correct option | Lower | Harder |
| Critical Testing | Removing critical information | Lower | Harder |
| **Negation Transformation** | Introducing negative semantics | **9.01%** | **Hardest** |

**Impact of Prompting Strategies**:

| Strategy | Effect | Description |
|------|------|------|
| Direct | Baseline | Computing log-likelihood to select the highest option |
| CoT (Native Language) | Improves robustness | Chain-of-thought reasoning helps in-depth understanding |
| Few-shot | Further improvement | Exemplar guidance improves variant performance |
| XLT (Cross-Lingual Translation) | Limited improvement | Translation might introduce noise |

### Key Findings

- **All LLMs fall far short of robustness in commonsense reasoning**: The best-performing GPT-4o achieves an average CRA of only 69.31%, compared to 91.03% for humans, representing a gap of over 20 percentage points.
- **Negation transformation is the Achilles' heel of LLMs**: The average accuracy is only 9.01% (below the random baseline of 25%), revealing a fundamental difficulty in LLMs' handling of negative semantics.
- **Question paraphrasing is the easiest**: Models exhibit some resistance to surface-level paraphrasing but fail to generalize to deeper variations in reasoning formats.
- **Language preference significantly affects robustness**: Chinese models (e.g., Qwen-Max) perform with higher robustness on the Chinese test (CRA 78.91% vs. English 59.65%), and vice versa.
- **Model capacity scales positively with robustness, but the gains are limited**: The average CRA of Qwen2.5 scales from 24.64% (0.5B) to only 37.38% (72B).

## Highlights & Insights

- The first systematic robustness evaluation framework for LLMs in commonsense reasoning, backed by theoretical grounds (Bloom's taxonomy of cognition) covering the complete cognitive hierarchy across 7 variants.
- The design of the CRA metric is highly original and rigorous—requiring all variant questions to be answered correctly to count as true understanding, effectively separating "memorized patterns" from "true understanding".
- The proposed methodology for constructing Chinese HellaSwag (two-stage + iterative adversarial filtering to align difficulty with the English version) is transferable to benchmarks in other languages.
- The finding of 9.01% accuracy in negation transformation is highly striking—falling significantly short of the 25% random guess rate, which demonstrates that models are not just incapable of handling negation but are systematically misled by it.
- Bilingual evaluation reveals the direct impact of language training adequacy on robustness, providing empirical support for scaling/training strategies of multilingual LLMs.

## Limitations & Future Work

- Variant generation primarily relies on Qwen-Max, which may introduce model-specific bias (notwithstanding human verification).
- The benchmark only covers Chinese and English; extending it to more languages (especially low-resource languages) would yield higher value.
- The abnormally low accuracy on negation transformation might partly stem from potential unfairness in question construction, which warrants further analysis.
- Lack of analysis on internal mechanisms—why are certain variants particularly difficult? How do attention patterns and hidden states change?
- Some variants (such as scenario rewriting) depend on choosing "relatively plausible" options from the original incorrect options, which might introduce subjective bias.

## Related Work & Insights

- **vs HellaSwag** (Zellers et al. 2019): The original benchmark only tests accuracy in fixed formats, whereas HellaSwag-Pro significantly enhances the discriminative power of the evaluation through 7 types of variants.
- **vs Balepur et al. (2024)**: Examined only a single variant, negation reasoning, while this work expands to 7 variants covering the entire cognitive hierarchy.
- **vs Zhou et al. (2021)**: Explored only question paraphrasing, whereas the variant designs in this study are far more diverse and profound.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic framework for multi-variant commonsense reasoning robustness evaluation, with an innovative Chinese HellaSwag construction methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 41 models, 9 prompting strategies, 7 variants, bilingual, and comprehensive ablation—a rare scale of evaluation in the field.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, solid data, and abundant figures and tables.
- Value: ⭐⭐⭐⭐ Offers a new paradigm for commonsense reasoning evaluation, with the 9.01% accuracy finding in negation transformation deserving of widespread attention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](com2_causal_commonsense.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](../../ACL2026/llm_evaluation/retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining](tic-lm_a_web-scale_benchmark_for_time-continual_llm_pretraining.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)

</div>

<!-- RELATED:END -->
