---
title: >-
  [Paper Note] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan
description: >-
  [LLM (Other)] This paper proposes KazMMLU, the first MMLU-style bilingual (Kazakh + Russian) evaluation benchmark designed specifically for Kazakhstan. It contains 23,000 multiple-choice questions from authentic educational materials, covering various disciplines (such as STEM, humanities, and social sciences) across multiple educational levels. Using this benchmark to evaluate 27 multilingual LLMs, the study reveals significant deficiencies in current models' Kazakh capabili…
tags:
  - "LLM (Other)"
date: 2026-05-08
content_hash: 7f0c5b091556bd54
---

# KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan

## Basic Information

- **Conference**: ACL2025
- **arXiv**: [2502.12829](https://arxiv.org/abs/2502.12829)
- **Code**: [https://huggingface.co/datasets/MBZUAI/KazMMLU](https://huggingface.co/datasets/MBZUAI/KazMMLU)
- **Area**: LLM NLP
- **Keywords**: Kazakh, MMLU, Multilingual Evaluation, Low-Resource Languages, Bilingual Benchmark

## TL;DR

This paper proposes KazMMLU, the first MMLU-style bilingual (Kazakh + Russian) evaluation benchmark designed specifically for Kazakhstan. It contains 23,000 multiple-choice questions from authentic educational materials, covering various disciplines (such as STEM, humanities, and social sciences) across multiple educational levels. Using this benchmark to evaluate 27 multilingual LLMs, the study reveals significant deficiencies in current models' Kazakh capabilities.

## Background & Motivation

### Problem Definition
Despite Kazakhstan's population of over 20 million, its language and culture remain severely underrepresented in the NLP field. As a Turkic language spoken by over 14 million people, Kazakh suffers from a critical scarcity of dedicated LLMs and evaluation benchmarks.

### Limitations of Prior Work

**Existing multilingual benchmarks lack Kazakhstan-specific content**: Cross-lingual benchmarks like GlobalMMLU, XCOPA, and XGLUE either exclude the Kazakh language or lack regional cultural context.

**Limitations of translated benchmarks**: Most existing resources rely heavily on translation from English, lacking local cultural richness.

**Existing Kazakh NLP datasets are limited to single tasks**: Datasets like KazNERD (Named Entity Recognition), KazQAD (Question Answering), and KazSANDRA (Sentiment Analysis) focus on narrow tasks and do not evaluate reasoning and domain knowledge.

**TUMLU lacks Kazakhstan's context**: Although it covers Turkic languages, it lacks country-specific content.

### Motivation
To build a large-scale evaluation benchmark that reflects Kazakhstan's bilingual education system, realistically assessing the knowledge and reasoning capabilities of LLMs in both Kazakh and Russian.

## Method

### Overall Architecture

KazMMLU is designed following the MMLU framework, comprising multiple-choice questions with 4-5 options each, covering various subjects and educational levels.

### Dataset Construction

**Data Sources**:
- State exams (iTest.kz, ymnik.kz, oltest.kz)
- Textbooks (Book - Shyn Kitap)
- Professional certification materials

**Three Collection Strategies**:
1. Automated online crawling (accounting for 85%)
2. Manual transcription from scanned books
3. Manual extraction from online resources

**Language Distribution**:
- Kazakh: 10,969 questions (48%)
- Russian: 12,031 questions (52%)

**Educational Levels**:
- High School: Both Kazakh and Russian questions
- University/Professional: Russian questions only (reflecting the reality of Russian-dominated higher education in Kazakhstan)

**Subject Coverage**:

| Group | Sample Subjects |
|---|---|
| STEM | Biology, Chemistry, Informatics, Mathematics, Physics, Medicine |
| Humanities | Kazakh History, Kazakh Literature, World History, Philosophy & Psychology |
| Social Sciences | Economics, Law, Management & Marketing, Geography, Sociology |
| Languages | Kazakh Language, Russian Language, Reading Literacy |

### Quality Control

- Two professional annotators (holding at least a bachelor's degree and fluent in both Kazakh and Russian) manually reviewed all questions.
- Verified correctness and completeness, discarding questions with errors or missing components.
- Wrote scripts to detect duplicates, validate metadata, and eliminate formatting issues.
- All questions are in the Cyrillic script (the official educational standard of Kazakhstan).

### Key Designs

**Prompt Configuration**:
1. Kazakh prompt + English letter output
2. English prompt + English letter output

**Answer Selection Method**:
- Open-source models: Next-token prediction, applying softmax over the probabilities of each option to select the maximum.
- Closed-source models: Free-text generation + string matching to extract answers.

## Experiments

### Main Results

Performance of 27 models under the English prompt (Average Accuracy):

| Model | STEM | Social Sci. | Humanities | Language | Other | Average |
|---|---|---|---|---|---|---|
| GPT-4o | 70.0 | 81.9 | 83.3 | 73.4 | 62.1 | **76.6** |
| DeepSeek V3 | 77.6 | 81.3 | 78.9 | 61.2 | 65.1 | **76.9** |
| Gemma-2-27B-IT | 57.3 | 60.5 | 63.2 | 39.1 | 48.3 | 57.4 |
| Llama3.1-70B | 58.0 | 59.1 | 57.4 | 41.8 | 49.3 | 56.2 |
| YandexGPT | 54.8 | 70.6 | 63.7 | 42.6 | 57.0 | 60.2 |
| Sherkala-Chat-8B | 43.1 | 49.8 | 50.0 | 34.9 | 38.3 | 45.6 |
| KazakhLLM-8B | 40.3 | 45.6 | 44.0 | 31.3 | 38.6 | 41.7 |
| BLOOMZ-7B | 23.8 | 24.1 | 23.4 | 23.9 | 22.5 | 23.8 |

- GPT-4o and DeepSeek V3 performed the best (~76-77%), significantly outperforming other models.
- **The Language category was consistently the most difficult**, yielding the lowest scores across all models.
- The impact of instruction tuning varied by model: Llama3.1-70B-instruct performed 7.9% worse than its base counterpart, whereas the 8B instruct version saw a 4.9% improvement.

### Kazakh vs. Russian Performance
- All models performed slightly better in Russian than in Kazakh.
- GPT-4o achieved the best performance in Kazakh (76.90%), while DeepSeek V3 excelled in Russian (81.4%).
- The discrepancies likely stem from training data availability, linguistic complexity, or tokenization differences.

### Few-shot Analysis
- All models showed consistent improvements as the number of shots increased.
- Qwen-2.5-7B and Mistral-7B-v0.3 benefited the most.
- The largest jump in accuracy occurred between 0-shot and 1-shot.
- English prompts consistently outperformed Kazakh prompts under 1/2/3-shot settings.

### Negation Sensitivity Analysis
- A subset of 2,554 questions containing negative phrases (such as жоқ, емес, не) was filtered.
- DeepSeek V3 proved most robust to negation, while Llama3.1-70B suffered the largest drop on reading literacy (57.1% $\rightarrow$ 50.0%).

### Model Confidence Analysis
- Model confidence was strongly correlated with accuracy ($r > 0.9$).
- Question length had a negligible effect on confidence.

### Key Findings
1. Even the best models achieved only ~77% on KazMMLU, which is far below their performance on English MMLU.
2. Models specifically fine-tuned for Kazakh (such as KazakhLLM-8B and Sherkala-Chat-8B) underperformed compared to general LLMs.
3. English prompts consistently outperformed Kazakh prompts, indicating insufficient Kazakh comprehension capabilities in these models.
4. Educational levels affected performance: open-source models experienced a noticeable drop in accuracy on university-level questions.

## Highlights & Insights

1. **First localized MMLU**: Designed specifically for Kazakhstan, encompassing local history, traditions, and linguistics rather than simple translations.
2. **Reflecting the true bilingual education system**: The structural design is fitting, featuring Kazakh/Russian bilingual questions for high school and a Russian-dominant setup for university.
3. **Large-scale and reliable**: All 23,000 questions were manually verified, with data sourced from authoritative materials (e.g., state examinations).
4. **Comprehensive model evaluation**: The evaluation spanned 27 models across open/closed-source, various scales, and different language family optimizations.
5. **Negation sensitivity and model confidence analysis**: This provides a multi-dimensional evaluation perspective that goes beyond simple accuracy.

## Limitations & Future Work

1. **Text-only modality**: It does not cover multimodal questions involving images, audio, etc.
2. **Lack of explicit reasoning evaluation**: The multiple-choice format cannot assess the explicit reasoning process of models or their open-ended QA capability.
3. **Static evaluation**: The predefined questions cannot fully capture models' generalization capabilities in dynamic, real-world scenarios.
4. **Cyrillic script only**: It does not cover the Latin script transition currently underway in Kazakhstan.

## Related Work & Insights

- **Kazakh LLMs**: KazakhLLM (ISSAI, 2024), Sherkala (Koto et al., 2025)
- **Multilingual Benchmarks**: MMLU (Hendrycks et al., 2021), ArabicMMLU (Koto et al., 2024), CMMLU (Li et al., 2024), IndoMMLU (Koto et al., 2023)
- **Turkic Language Evaluation**: TUMLU (Isbarov et al., 2025), SIGTURK (Maxutov et al., 2024)
- **Kazakh NLP**: KazNERD (Yeshpanov et al., 2022), KazQAD (Yeshpanov et al., 2024)

## Rating ⭐⭐⭐⭐

- Novelty: ⭐⭐⭐⭐ — The first localized MMLU for Kazakhstan, representing pioneering work.
- Value: ⭐⭐⭐⭐⭐ — Plays a crucial role in promoting the evaluation of LLMs on low-resource languages.
- Methodological Novelty: ⭐⭐⭐ — The benchmark construction methodology is mature but lacks technical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Very comprehensive, with 27 models evaluated alongside multi-dimensional analyses (few-shot, negation, confidence, cross-lingual).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [\[ACL 2025\] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)

</div>

<!-- RELATED:END -->
