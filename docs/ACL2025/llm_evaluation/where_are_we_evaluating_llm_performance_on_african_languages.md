---
title: >-
  [Paper Note] Where Are We? Evaluating LLM Performance on African Languages
description: >-
  [ACL 2025][LLM Evaluation][African languages] Constructed the Sahara benchmark covering 517 African languages, 30 datasets, and 16 task categories to systematically evaluate 24 LLMs on African languages, revealing how language policy-driven data inequality directly impacts model performance.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "African languages"
  - "multilingual evaluation"
  - "language policy"
  - "low-resource NLP"
  - "benchmarking"
date: 2026-05-08
content_hash: 9d558769a294c690
---

# Where Are We? Evaluating LLM Performance on African Languages

**Conference**: ACL 2025  
**arXiv**: [2502.19582](https://arxiv.org/abs/2502.19582)  
**Code**: [GitHub](https://github.com/UBC-NLP/sahara)  
**Area**: LLM Evaluation  
**Keywords**: African languages, multilingual evaluation, language policy, low-resource NLP, benchmarking

## TL;DR

Constructed the Sahara benchmark covering 517 African languages, 30 datasets, and 16 task categories to systematically evaluate 24 LLMs on African languages, revealing how language policy-driven data inequality directly impacts model performance.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Africa is home to approximately 2000 languages, making it the most linguistically diverse continent globally, yet it remains severely underrepresented in NLP research:

1. **Impact of Historical Language Policies**: Most African nations use foreign languages introduced during the colonial era (English, French, Portuguese) as their official languages. For example, of Nigeria's 512 indigenous languages, only 3 receive regional recognition. Even when indigenous languages are officially recognized, their roles are often symbolic rather than functional.
2. **Extreme Data Imbalance**: Among 517 African languages, only 45 have more than one dataset, with the vast majority only having language identification data. Amharic leads with 11 datasets, while most languages have almost no available resources.
3. **Incomplete Existing Evaluations**: Prior efforts such as IROKOBench only cover a limited number of African languages and lack a comprehensive cross-lingual, cross-task evaluation benchmark to track overall progress.
4. **Data Availability $\neq$ Speaker Population**: Nigerian Pidgin (Naija), with 153 million speakers, is classified as a "forgotten language," whereas Catalan, with only 5 million speakers, is a high-resource language. This shows that language prestige, policy, and digitization are the critical factors.

## Method

### Overall Architecture

The Sahara benchmark adopts a modular design, collecting and integrating from existing public datasets to cover four major task clusters: classification, generation, multiple-choice/reasoning (MCCR), and token-level tasks. It supports 517 languages and 30 datasets. Simultaneously, it provides a dynamic leaderboard on HuggingFace for continuous tracking of model performance.

### Key Designs

1. **Broad and Diverse Coverage**: Covers 50 out of 54 African countries, including 5 writing systems (Arabic, Coptic, Ethiopic, Latin, Vai) and 5 language families. For each task, 1000 samples are randomly selected from the dataset for few-shot testing.

2. **Task Cluster Organization**:
    - Classification Cluster: Cross-lingual NLI, Language Identification (for 517 languages), news classification, sentiment analysis, topic classification.
    - Generation Cluster: Machine translation (for 29 languages), paraphrasing, summarization, headline generation.
    - MCCR Cluster: General knowledge (MMLU), math word problems (MGSM), reading comprehension, question answering.
    - Token-level Cluster: NER (for 27 languages), phrase chunking, part-of-speech (POS) tagging.

3. **Policy-Data-Performance Chain Analysis**: Beyond evaluating model performance, it systematically analyzes how language policies (educational, national, regional) influence data availability and ultimately determine model effectiveness on specific languages, forming a causal chain of "Policy $\to$ Data $\to$ Performance".

### Loss & Training

As this is an evaluation work, no model training is involved. Evaluation setup:
- Consistent use of few-shot settings (3-10 shots depending on the task).
- Evaluation metrics include Exact Match, F1, Accuracy, spBLEU1K, and RougeL.
- Evaluated 24 models: divided into SLMs ($<8$B) and LLMs ($\ge 8$B).

## Key Experimental Results

### Main Results

Average performance of 24 models across four task clusters (overall average score):

| Model | Classification Avg | Generation Avg | MCCR Avg | Token Avg | Overall Avg |
|------|---------|---------|----------|-----------|---------|
| Claude-4-Sonnet (Closed-source) | 47.28 | 10.59 | 60.53 | 44.86 | **40.82** |
| GPT-4.1 (Closed-source) | 48.07 | 11.06 | 50.98 | 34.05 | 36.04 |
| Command-A (111B) | 38.64 | 10.36 | 45.55 | 25.16 | 29.93 |
| Gemma3 (27B) | 44.44 | 8.19 | 43.20 | 16.45 | 28.07 |
| Llama3.1 (70B) | 35.96 | 11.15 | 43.67 | 15.51 | 26.57 |
| Phi-4 (3.8B) | 16.50 | 5.10 | 33.73 | 11.78 | 16.78 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| English $\to$ African language translation | spBLEU $\sim$12 | Translating to African languages is difficult |
| African language $\to$ English translation | spBLEU $\sim$19 | Translating from African languages is easier |
| French $\to$ African language translation | spBLEU $\sim$3.4 | Translating from French is more difficult, reflecting training data disparities |
| Language identification task | F1 $<5\%$ | Most models can barely identify African languages |
| MGSM mathematical reasoning | ExactM $<10\%$ | Open-source models perform extremely poorly in mathematical reasoning for African languages |

### Key Findings

1. **Closed-source Models Lead Significantly**: Claude-4-Sonnet significantly outperforms all open-source models with an overall average score of 40.82, followed closely by GPT-4.1.
2. **Comprehension is Easier than Generation**: Models perform relatively well on classification tasks (reaching $>80\%$ accuracy in some languages) but poorly on generation tasks (with BLEU/ROUGE $<15$ in most cases), indicating that the ability of models to comprehend African languages far exceeds their ability to generate them.
3. **Minority of Languages Benefit**: Models perform best on a few resource-rich languages such as Hausa, Swahili, Yorùbá, and Afrikaans, all of which enjoy official status and sufficient training data.
4. **Data Availability Drives Performance**: Performance differences strongly correlate with the volume of language data, rather than the intrinsic complexity of the language. Swahili stands out due to its standardized spelling, regular morphology, and abundant bilingual corpora.
5. **Small Models are Competitive on Specific Tasks**: Phi-4 (3.8B) performs best among SLMs in MCCR tasks, demonstrating that ultra-large models are not always necessary in certain scenarios.

## Highlights & Insights

- **Empirical Demonstration of the "Policy $\to$ Data $\to$ Performance" Causal Chain**: Provides the first large-scale empirical evidence of how language policy influences AI model performance through the mediation of data.
- **Comprehensive Coverage**: Conducts a comprehensive evaluation spanning 517 languages, 30 datasets, and 24 models, establishing the most extensive benchmark for African NLP.
- **Dynamic Leaderboard**: The public leaderboard hosted on HuggingFace supports ongoing evaluation and tracking, driving community development.
- **Actionable Policy Recommendations**: Goes beyond merely exposing issues by proposing concrete recommendations for data collection, policy reform, and community-driven annotation.
- **Asymmetric Translation Performance**: Reveals that translating *into* African languages is much harder than translating *from* them, highlighting the bottlenecks in target-side generation by these models.

## Limitations & Future Work

1. **Datasets are Mostly Translated**: Many datasets (such as AfriXLNI, AfriMMLU) are translated from English, which might not fully reflect the actual contexts of African language usage and could introduce label misalignment or loanword bias.
2. **Most Languages Only Have Language Identification Data**: The vast majority of the 517 languages only support language identification tasks, preventing the evaluation of more complex capabilities.
3. **Evaluation Method Constraints**: Sampling 1000 instances for few-shot evaluations may not capture the full variability of each language.
4. **Lack of Human Evaluation on Generation Quality**: Relies solely on automatic metrics without human evaluation to verify the actual quality of generated text.
5. **Insufficient Coverage of Dialectal Variations**: Dialectal differences within the same language can be vast, but dialects are not distinguished in this benchmark.

## Related Work & Insights

- **IROKOBench** (Adelani et al., 2024b) evaluated LLM performance on African languages first, but with limited coverage.
- **AfroLID** (Adebara et al., 2022) covers language identification for 517 African languages and serves as a major component of Sahara.
- **NLLB** (Meta) attempts to improve translation in low-resource languages, but performance remains inconsistent.
- The **Masakhane** community has made key contributions to advancing African NLP, driving the creation of multiple datasets.
- Core insight of this work: Technical progress must go hand-in-hand with policy reform; pure model optimization cannot resolve the fundamental issue of data scarcity.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 3 |
| Data Contribution | 5 |
| Experimental Thoroughness | 4 |
| Social Impact | 5 |
| Writing Quality | 4 |
| Total Score | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance](language_complexity_measurement_as_a_noisy_zero-shot_proxy_for_evaluating_llm_pe.md)
- [\[ACL 2025\] La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America](la_leaderboard_spanish.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[AAAI 2026\] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](../../AAAI2026/llm_evaluation/where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)

</div>

<!-- RELATED:END -->
