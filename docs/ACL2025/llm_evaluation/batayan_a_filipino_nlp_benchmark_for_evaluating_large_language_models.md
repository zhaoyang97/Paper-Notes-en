---
title: >-
  [Paper Note] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models
description: >-
  [ACL 2025][LLM Evaluation][Tagalog/Filipino] This work introduces Batayan, the first comprehensive NLP benchmark designed to evaluate LLMs on the Filipino language. It covers 8 tasks across three core capabilities (NLU, NLR, NLG), including 3 first-of-their-kind Filipino tasks. Constructed by native speakers to ensure linguistic authenticity, Batayan evaluates over 50 open-source and commercial LLMs, revealing that performance on Filipino significantly lags behind English. No…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Tagalog/Filipino"
  - "Low-resource languages"
  - "benchmark"
  - "multilingual LLM"
  - "morphology"
  - "code-switching"
date: 2026-05-08
content_hash: 6a24090fa616a953
---

# Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.14911](https://arxiv.org/abs/2502.14911)  
**Code**: [https://github.com/aisingapore/sea-helm](https://github.com/aisingapore/sea-helm)  
**Area**: LLM Evaluation  
**Keywords**: Tagalog/Filipino, Low-resource languages, benchmark, multilingual LLM, morphology, code-switching

## TL;DR

This work introduces Batayan, the first comprehensive NLP benchmark designed to evaluate LLMs on the Filipino language. It covers 8 tasks across three core capabilities (NLU, NLR, NLG), including 3 first-of-their-kind Filipino tasks. Constructed by native speakers to ensure linguistic authenticity, Batayan evaluates over 50 open-source and commercial LLMs, revealing that performance on Filipino significantly lags behind English. Notably, explicit Filipino linguistic support and model scale expansion consistently yield substantial performance gains.

## Background & Motivation

**Background**: LLMs exhibit superior performance on high-resource languages like English, and multilingual benchmarks (e.g., MMLU, HellaSwag) also predominantly focus on high-resource settings. Although Filipino has over 80 million speakers, it remains virtually absent from mainstream LLM evaluation suites.

**Limitations of Prior Work**: (a) Map-to-language corpora are mostly machine-translated, exhibiting severe "translationese," such as abnormal preference for DKA (ay-inversion) syntax and unnatural word choices; (b) Existing datasets contain a single domain, limited tasks, and are often constructed by non-native speakers; (c) Southeast Asian benchmarks like BHASA and SeaEval either omit Filipino or rely solely on machine translation.

**Key Challenge**: Filipino exhibits rich linguistic characteristics—an agglutinative morphological system (including prefixes, infixes, suffixes, and circumfixes), multilingual influences (Spanish, Chinese, and Malay loanwords), and pervasive English-Filipino code-switching (Taglish). However, there is a lack of high-quality evaluation databases to assess LLMs' comprehension of these features.

**Goal** To construct the first comprehensive, high-quality, native speaker-driven LLM evaluation benchmark for Filipino, covering the three core capabilities of understanding, reasoning, and generation.

**Key Insight**: Integrating 8 tasks (including 3 novel Filipino tasks) and grounding them in the KWF (Commission on the Filipino Language) official translation guidelines, emphasizing common word order (KA/karaniwang ayos) and natural colloquial vocabulary.

**Core Idea**: Constructing the first comprehensive Filipino LLM benchmark through a rigorous native-speaker-driven quality pipeline, exposing specific capability gaps in low-resource settings and providing a reproducible construction methodology.

## Method

### Overall Architecture

Batayan organizes 8 tasks across three capability dimensions, totaling 3,800 test samples:
- **NLU (Natural Language Understanding)**: Paraphrase Identification (PI; PAWS, 2,000 samples), Question Answering (QA; Belebele, 900 samples), Sentiment Analysis (SA; PH Elections, 5,160 samples), Topic Detection (TD; PH Elections, 5,160 samples)
- **NLR (Natural Language Reasoning)**: Causal Reasoning (CR; Balanced COPA, 500 samples), Natural Language Inference (NLI; XNLI, 5,010 samples)
- **NLG (Natural Language Generation)**: Abstract Summarization (AS; XL-Sum, 11,535 samples), Machine Translation (MT; FLORES-200, 1,012 samples)

### Key Designs

1. **Native-Speaker Translation and Three-Fold Quality Control**:
    - English-source datasets (PAWS, COPA, XNLI, XL-Sum) were first machine-translated by the Helsinki NLP OPUS model and then manually corrected by native Filipino speakers.
    - Each sample was evaluated by 3 native speakers based on three binary criteria: completeness, fluency, and semantic adequacy.
    - Only samples with a unanimous 3/3 agreement were kept in the final test set.
    - Strong agreement was achieved in sentiment analysis annotation (Cohen's kappa of 0.8202, Krippendorff's alpha of 0.8268).

2. **Native Re-Translation to Correct Erroneous Prior Data**:
    - Existing Filipino versions of QA (Belebele) and MT (FLORES-200) are of poor quality and suffer from translationese.
    - Native speakers re-translated these, prioritizing natural word order (KA/karaniwang ayos, predicate-first) over rigid DKA (ay-inversion).
    - Example: The original translation used unnatural DKA word order and incorrect verb collocations; the corrected version uses KA word order and more precise diction.

3. **Reservation of Native Filipino Data**:
    - SA and TD leverage Filipino political tweets (PH Elections), preserving genuine Taglish code-switching and non-standard spelling.
    - No manual normalization was performed, as these variations reflect how Filipino is naturally used.

4. **SEA-HELM Platform Integration**:
    - Released as the Filipino component of the SEA-HELM execution framework, complete with a leaderboard.
    - Provides 5-shot examples to support few-shot evaluations.

### Evaluation Protocol

- Instruction-tuned models are evaluated via zero-shot prompting by default, while base pre-trained models are evaluated via five-shot.
- NLU/NLR tasks use macro $F_1$, MT uses ChrF++ and MetricX-24, and AS uses BERTScore + ChrF++ + ROUGE-L.

## Key Experimental Results

### Main Results

Evaluated over 50 models ranging from 7B to 671B parameters, including GPT-4o, Gemini, Llama 3, Qwen 2.5, SEA-LION, etc.

| Model Type | NLU Macro F1 | NLR Macro F1 | NLG Overall | Characteristics |
|---------|-------------|-------------|---------|------|
| Gemma-SEA-LION-v3-9B-IT | 79.23 (best small model) | CR=92.75 | 60.35 | Fine-tuned on Filipino |
| Commercial models (e.g., GPT-4o) | 75.14-86.23 | High | MetricX > 88 | Large-scale pre-training |
| Llama 3 Series | Moderate | CR ~ 0 | Moderate | No specialized support for Filipino |

### Ablation Study

| Finding | Details |
|------|------|
| Importance of explicit Filipino support | SEA-LION achieves 92.75% on CR, while Llama 3 (without Filipino support) scores near 0% |
| Model scaling effect | Larger models in the same family consistently outperform smaller ones, but scaling alone is insufficient—regional language fine-tuning is necessary |
| Sensitivity of semantic metrics | ROUGE-L fails to distinguish whether a model has been fine-tuned on Filipino, whereas MetricX-24 exhibits significant differences (87+ vs lower) |
| Open-Source vs. Commercial | Open-source models fine-tuned on Filipino can match or even exceed the performance of commercial systems |

### Specific Challenges in Dataset Construction

- **Difficulties in Vocabulary Adaptation**: Technical terms like "rule of thirds" have no Filipino equivalents and must be kept in English.
- **Homonym Disambiguation**: Words like "right" require context to determine whether they refer to direction ("kanan") or correctness ("tama").
- **Idiomatic Localization**: Expressions such as "turned himself in" must be localized to natural Filipino idioms ("isinuko ang sarili").
- **Retention of Non-Standard Spellings**: Spelling variations in social media data are preserved as part of the language's authentic usage.
- **Addressing Class Imbalance**: Political tweets skew heavily negative; this is mitigated through resampling and filtering for high-agreement samples.

## Highlights & Insights

1. **Native Speaker-Driven Paradigm**: Native speakers remained in the loop throughout translation, annotation, and quality review, establishing a reproducible methodology template for low-resource benchmark construction.
2. **Systemic Analysis of Translationese**: The paper documents the phenomenon of machine translation's preference for DKA, revealing systemic limitations of automatic translators in preserving natural Filipino expression.
3. **Three First-of-its-Kind Filipino Tasks**: Abstractive Summarization (AS), Causal Reasoning (CR), and Paraphrase Identification (PI) are introduced for the first time in Filipino.
4. **Honest Documentation of Practical Challenges**: The paper details the real-world translation, annotation, and vocabulary-matching challenges encountered, offering valuable guidance for future low-resource NLP endeavors.
5. **The SEA-HELM Ecosystem**: Integrated into the SEA-HELM evaluation framework, it supports community-driven iterative development and cross-lingual performance comparison.

## Limitations & Future Work

1. **High Proportion of Translated Content**: 6 out of 8 tasks are adapted or translated from English datasets, which might introduce cultural bias and English-centric reasoning patterns even with native translation.
2. **Limited Dataset Size**: The final test set consists of 3,800 samples, and some tasks (e.g., AS, QA) are constrained to 100 samples, which limits statistical significance.
3. **Coverage Limited to Tagalog and Taglish**: More than 100 regional languages (e.g., Cebuano, Ilocano) are spoken in the Philippines, but remain unrepresented in this benchmark.
4. **Domain Constraints in SA/TD**: The sentiment and topic data are drawn only from political tweets, which may not generalize to broad-domain sentiment or toxicity patterns.
5. **Omission of Prosodic Features**: Written text lacks critical prosodic markers, making it difficult to capture spoken cues such as intonation and sarcasm.

## Related Work & Insights

- **vs. BHASA**: Evaluates regional Southeast Asian languages like Indonesian and Thai but omits Filipino; Batayan effectively bridges this gap.
- **vs. SeaEval**: Features Filipino but largely relies on machine-translated content and generic multilingual prompts, leading to subpar quality.
- **vs. XTREME/XGLUE**: Certain multilingual tasks encompass Filipino, but the task coverage is incomplete and lacks specialized quality control.
- **Insights**: The triad of a native-speaker-driven pipeline, translationese detection, and standardized evaluation execution is highly generalizable to other low-resource languages (e.g., Vietnamese, Burmese, Khmer). The construction methodology of Batayan itself represents a major contribution.

## Rating

- Novelty: ⭐⭐⭐⭐ (First comprehensive Filipino LLM benchmark, introducing 3 first-of-its-kind tasks)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Evaluation of 50+ models, spanning 7B to 671B parameters and including commercial systems)
- Writing Quality: ⭐⭐⭐⭐ (Detailed linguistic background and transparent discussion of dataset construction challenges)
- Value: ⭐⭐⭐⭐ (Fills the gap in Filipino LLM evaluation, offering a generalizable methodology for other low-resource settings)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[ACL 2025\] PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory](papersplease_a_benchmark_for_evaluating_motivational_values_of_large_language_mo.md)

</div>

<!-- RELATED:END -->
