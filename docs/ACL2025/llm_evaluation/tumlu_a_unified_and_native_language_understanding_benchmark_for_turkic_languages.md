---
title: >-
  [Paper Note] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages
description: >-
  [ACL 2025][LLM Evaluation][MMLU] Proposes TUMLU and TUMLU-mini, the first native multi-task language understanding benchmark for 9 Turkic languages. It contains 38,139 middle and high school multiple-choice questions spanning Latin, Cyrillic, and Arabic script systems. It systematically evaluates 13 open-source and closed-source LLMs, revealing the differentiated impacts of script systems, language resource sizes, and CoT on model performance.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "MMLU"
  - "Turkic languages"
  - "multilingual benchmark"
  - "native language evaluation"
  - "low-resource languages"
date: 2026-05-08
content_hash: 0e3299b5be8cbe8b
---

# TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages

**Conference**: ACL 2025  
**arXiv**: [2502.11020](https://arxiv.org/abs/2502.11020)  
**Code**: [ceferisbarov/TUMLU](https://github.com/ceferisbarov/TUMLU)  
**Area**: NLP / Multilingual NLU / Benchmark Evaluation  
**Keywords**: MMLU, Turkic languages, multilingual benchmark, native language evaluation, low-resource languages

## TL;DR

Proposes TUMLU and TUMLU-mini, the first native multi-task language understanding benchmark for 9 Turkic languages. It contains 38,139 middle and high school multiple-choice questions spanning Latin, Cyrillic, and Arabic script systems. It systematically evaluates 13 open-source and closed-source LLMs, revealing the differentiated impacts of script systems, language resource sizes, and CoT on model performance.

## Background & Motivation

**Core Problem**: How to construct a high-quality, translation-bias-free MMLU-style evaluation benchmark for the Turkic language family, which possesses unique morphosyntax?

**Translation Bias in Existing Multilingual Benchmarks**: Mainstream multilingual benchmarks like Global MMLU rely on machine translation from English, which introduces translationese and cultural mismatch, failing to reflect genuine reasoning scenarios in the target language. For example, translated math problems may lose idiomatic expressions in the original language.

**Severe Absence of Turkic Languages**: Before TUMLU, Uyghur, Karakalpak, Tatar, and Crimean Tatar had never been included in any MMLU-style benchmarks. The coverage of existing benchmarks for Azerbaijani, Kazakh, Kyrgyz, and Uzbek was also extremely limited, with current datasets predominantly focusing on Turkish (TurkishMMLU).

**Difficulties in Cross-Lingual Comparability**: Although native benchmarks like INCLUDE do not rely on translation, the distribution of subjects and knowledge levels varies greatly across different languages, making it difficult to conduct a fair comparison of model capabilities within the same language family. Additionally, the Turkic language family involves three script systems (Latin, Cyrillic, Arabic), and the impact of script systems on LLM performance has not been systematically investigated.

## Method

### Overall Architecture

TUMLU is a multilingual multi-task language understanding benchmark for the Turkic language family, covering 9 languages: Azerbaijani, Crimean Tatar, Karakalpak, Kazakh, Kyrgyz, Tatar, Turkish, Uyghur, and Uzbek. The dataset contains 38,139 four-choice multiple-choice questions sourced from publicly available university entrance exam sample questions and textbooks from various countries, spanning 11 middle and high school subjects. The evaluation employs two prompting methods: 5-shot and 5-shot CoT, with all prompts written in the target languages.

### Key Designs

1. **Native Data Collection and Multi-Layered Quality Control**: All questions are collected directly from public educational resources in each language (non-translated), with most being official or mock university entrance exam questions. They are unified into a 4-option format (removing one incorrect option if there are more than 4, and keeping them as-is if fewer than 4). Native annotators manually audited random samples from each subject, identifying and correcting about 10% of questions that were invalid or had incorrect answers. Option shuffling was applied to mitigate memorization effects.

2. **TUMLU-mini Compact Validation Subset**: To manage experimental costs and ensure balance, 100 questions were randomly sampled from each "subject × language" combination (if fewer than 100, all were used), all of which were manually verified. Subjects existing in only 1-2 languages (logic, man and society, philosophy, religious ethics) were removed, leaving 7 core subjects for the main experiments. Prior work has demonstrated that 100 questions per subject is sufficient to reliably estimate performance on large-scale datasets.

3. **Dual-Script Parallel Dataset**: For languages that simultaneously use two alphabetic systems (Latin/Cyrillic for Crimean Tatar, Cyrillic/Latin for Kazakh, Arabic/Latin for Uyghur, Latin/Cyrillic for Uzbek), parallel versions via automatic transliteration are provided. This enables a controlled-variable comparison of the impact of script systems on LLM comprehension under identical content conditions. FineWeb 2 corpus statistics were utilized to verify that the performance discrepancies correspond directly with the distribution of each script in the pre-training corpora.

### Evaluation Settings

| Configuration | Setting |
|-------|------|
| Prompting | 5-shot / 5-shot CoT (CoT evaluated only in Azerbaijani, Kazakh, Turkish, and Uzbek) |
| Original Prompt Language | Original text of target languages (non-English) |
| Temperature | 0.0 |
| Max tokens | 1,024 |
| Top-p | 1.0 |
| API | OpenAI / Anthropic / Google Cloud / Together AI / Deep Infra |

## Key Experimental Results

### Main Results: Average 5-shot Accuracy of Models (%)

| Model | Average | aze | crh | kaa | kaz | kir | tat | tur | uig | uzb |
|------|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Claude 3.5 Sonnet | **78.9** | 84.4 | 81.2 | 75.3 | 83.0 | 75.7 | 84.0 | 85.7 | 71.3 | 69.1 |
| GPT-4o | 74.9 | 82.4 | 70.5 | 70.8 | 81.0 | 72.9 | 80.5 | 83.7 | 66.5 | 65.4 |
| Gemini 1.5 Pro | 73.8 | 78.6 | 70.3 | 68.2 | 78.4 | 72.3 | 80.5 | 80.0 | 71.0 | 65.1 |
| Gemini 1.5 Flash | 65.4 | 72.4 | 68.0 | 61.2 | 68.6 | 63.2 | 68.3 | 76.6 | 57.8 | 52.1 |
| Claude 3.5 Haiku | 64.0 | 70.6 | 62.9 | 55.2 | 69.9 | 64.8 | 67.5 | 78.0 | 56.6 | 50.3 |
| Llama 3.1 405B | 63.0 | 65.9 | 69.5 | 60.0 | 69.0 | 64.1 | 70.4 | 59.7 | 58.2 | 50.4 |
| Qwen2.5 72B | 59.9 | 70.1 | 61.8 | 54.6 | 62.6 | 47.5 | 62.5 | 73.9 | 56.0 | 50.4 |
| Llama 3.3 70B | 58.6 | 66.0 | 58.7 | 49.2 | 60.0 | 60.2 | 69.5 | 68.4 | 51.6 | 44.1 |
| Llama 3.1 70B | 57.7 | 68.1 | 57.3 | 49.9 | 56.4 | 58.4 | 66.2 | 64.9 | 52.4 | 45.3 |
| Llama 3.1 8B | 40.3 | 48.4 | 35.7 | 33.4 | 46.4 | 41.8 | 44.1 | 47.7 | 35.0 | 29.9 |

### Script System Comparison: Accuracy across Different Alphabets for Identical Questions (%)

| Language | Claude 3.5 Sonnet | | Qwen2.5 72B | | Gemma 2 27B | |
|------|----------|-------|----------|-------|----------|-------|
| | Cyrillic | Latin/Arabic | Cyrillic | Latin/Arabic | Cyrillic | Latin/Arabic |
| Crimean Tatar | 66.1 | **80.0** | 47.6 | **61.8** | 43.5 | **49.8** |
| Kazakh | **82.7** | 78.0 | **64.3** | 54.1 | **58.5** | 46.3 |
| Uyghur | 64.5 | **70.8** (Arabic) | 53.4 | **56.1** (Arabic) | 36.0 | **42.2** (Arabic) |
| Uzbek | 67.9 | **68.6** | **51.1** | 50.4 | **39.4** | 36.9 |

### CoT Effect: Gains of 5-shot CoT vs. 5-shot (Claude 3.5 Sonnet)

| Language | 5-shot | 5-shot CoT | Gain |
|------|--------|-----------|------|
| Azerbaijani | 84.4 | 87.1 | +2.7 |
| Tatar | 84.0 | 87.9 | +3.9 |
| Turkish | 85.7 | 87.9 | +2.1 |
| Uzbek | 69.1 | 72.9 | +3.7 |
| Kazakh | 83.0 | 84.1 | +1.1 |

### Key Findings

1. **Closed-Source Models Dominate**: Claude 3.5 Sonnet performs the best across all 9 languages, and the top 5 models are all closed-source. The best open-source model, Qwen2.5 72B, logs 19 percentage points lower accuracy than Claude.
2. **Significant Scaling Effects**: Llama 3.1 scaling from 8B to 70B yields an improvement of approximately 17.4 pp, while Qwen2.5 scaling from 7B to 72B yields about 18.9 pp.
3. **Script System Impact Aligns with Corpus Distribution**: Models perform better in the script systems that hold a larger proportion in their pre-training corpora (e.g., Kazakh Cyrillic > Latin, Crimean Tatar Latin > Cyrillic).
4. **National Language and Literature is the Most Challenging**: Claude averages only ~68% on this subject, much lower than in other STEM subjects (~85%+), which requires deep cultural and literary knowledge.
5. **Cross-Lingual Transfer for Low-Resource Languages**: Surprisingly, performance on extremely low-resource languages like Crimean Tatar is comparable to medium-resource languages. However, language detection on the generated text reveals that the output is mostly a code-mixed hybrid of Turkish and the target language.
6. **CoT Yields Consistent Positive Gains for Closed-Source Models**: Claude CoT improves performance by +2.7 pp on average, but it conversely incurs a negative impact on small open-source models like Llama 3.1 8B (-5.1 pp).

## Highlights & Insights

- First MMLU-style benchmark to cover Uyghur, Karakalpak, Tatar, and Crimean Tatar, filling a crucial gap in LLM evaluation for the Turkic language family.
- Ingenious experimental design with dual-script parallel datasets: by controlling for content variables and only altering script systems, it reveals the decisive impact of pre-training corpus distribution on script preferences.
- Generated language detection analysis uncovers a deep-seated issue: "correctly answering questions" in low-resource languages does not imply "correctly using the language"; instead, models are performing cross-lingual code-mixed reasoning.
- The scale of 38,139 native questions ranks among the largest for regional language family benchmarks.

## Limitations & Future Work

- Subject difficulty is not uniform across all languages (Turkish and Uzbek are at college entrance exam level, while others are community-contrived), meaning cross-lingual comparisons should be approached with caution.
- CoT prompts only cover 4 languages, leaving the other 5 without CoT evaluation.
- Certain subjects (e.g., logic, philosophy) only contain data in 1-2 languages, limiting overall completeness.
- The dataset may partially overlap with the LLM pre-training corpora (despite option shuffling).
- The latest LLMs (such as newer models like GPT-4o-mini, Llama 3.2, etc.) have not been evaluated.

## Related Work & Insights

- The Turkish subset of TurkishMMLU (Yüksel et al., 2024) was directly integrated into TUMLU to form a more complete cross-lingual comparison.
- Shares the native data philosophy with INCLUDE (Romanou et al., 2025), but TUMLU achieves a more balanced distribution of subjects and difficulty levels by focusing on a single language family.
- Contrasts with the translation-based approach of Global MMLU (Singh et al., 2024), demonstrating that while native data is more expensive to construct, its quality is significantly superior.
- Insight: Regional language family benchmarks (e.g., Turkic, Dravidian) might design more effective assessments for low-resource languages compared to global benchmarks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first native MMLU for the Turkic language family, with parallel comparative experiments on three script systems offering unique value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 13 models (open & closed-source), 9 languages, 7 subjects, two prompting methods, and script system comparative analysis.
- **Writing Quality**: ⭐⭐⭐ — Highly detailed but some sections are verbose; discussions on the limitations of cross-lingual comparisons could be introduced earlier.
- **Value**: ⭐⭐⭐⭐ — Fills major gaps in Turkic-language LLM evaluation, with the dual-script parallel dataset carrying independent research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)
- [\[ACL 2025\] skLEP: A Slovak General Language Understanding Benchmark](sklep_a_slovak_general_language_understanding_benchmark.md)
- [\[ACL 2025\] NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark](noreval_a_norwegian_language_understanding_and_generation_evaluation_benchmark.md)
- [\[ACL 2025\] READoc: A Unified Benchmark for Realistic Document Structured Extraction](readoc_a_unified_benchmark_for_realistic_document_structured_extraction.md)
- [\[ACL 2025\] La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America](la_leaderboard_spanish.md)

</div>

<!-- RELATED:END -->
