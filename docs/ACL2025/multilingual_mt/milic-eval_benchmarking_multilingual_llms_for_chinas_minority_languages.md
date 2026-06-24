---
title: >-
  [Paper Note] MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages
description: >-
  [Multilingual & Machine Translation] The authors construct MiLiC-Eval, the first standardized LLM evaluation benchmark for China's minority languages (Tibetan, Uyghur, Kazakh, and Mongolian). It contains 24k instances across 9 tasks, revealing the severe deficiencies of current LLMs in processing non-mainstream writing systems.
tags:
  - "Multilingual & Machine Translation"
date: 2026-05-08
content_hash: 8bc70fbf58c07bdc
---

# MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages

## Paper Information

- **Conference**: ACL 2025
- **arXiv**: [2503.01150](https://arxiv.org/abs/2503.01150)
- **Code**: [https://github.com/luciusssss/MiLiC-Eval](https://github.com/luciusssss/MiLiC-Eval)
- **Area**: Multilingual / Low-Resource Language Evaluation
- **Keywords**: Low-resource languages, minority languages, multilingual LLMs, evaluation benchmark, writing systems

## TL;DR

The authors construct MiLiC-Eval, the first standardized LLM evaluation benchmark for China's minority languages (Tibetan, Uyghur, Kazakh, and Mongolian). It contains 24k instances across 9 tasks, revealing the severe deficiencies of current LLMs in processing non-mainstream writing systems.

## Background & Motivation

- **Background**: LLMs exhibit outstanding performance in high-resource languages such as English and Chinese, but support for thousands of low-resource languages remains severely inadequate, particularly for those using non-mainstream writing systems.
- **Key Gap**: Despite having tens of millions of speakers, China's Tibetan, Uyghur, Kazakh, and Mongolian languages have been severely marginalized in NLP research, and they lack standardized evaluation benchmarks.
- **Limitations of Prior Work**: Existing multilingual benchmarks (e.g., XTREME, MEGA) exhibit insufficient coverage of low-resource writing systems, limited task diversity, and a lack of cross-task comparability. Furthermore, some rely on machine-translated data, leading to distorted evaluations.
- **Key Challenge**: These languages use non-Latin scripts (e.g., traditional Mongolian, Tibetan), posing extra challenges for tokenization and language modeling.

## Method

### Overall Architecture

MiLiC-Eval covers 9 task categories with a total of 24,000 instances across 4 minority languages (Tibetan [bo], Uyghur [ug], Kazakh [kk], and Mongolian [mn]). The benchmark design adheres to three core principles: focusing on non-mainstream writing systems, cross-lingual/cross-task alignment, and fine-grained skill assessment.

### Key Designs

1. **Focus on Non-Mainstream Writing Systems**: This work benchmarks Arabic-script Kazakh and traditional Mongolian (rather than the mainstream Cyrillic script) for the first time, addressing the weakest links of LLMs.
2. **Cross-Lingual and Cross-Task Alignment**: Parallel data for 6 tasks is provided across 6 languages (including Chinese and English). Using the same text for multiple tasks avoids evaluation bias introduced by a single task format.
3. **Hierarchical Skill Assessment Framework**: The 9 task categories are mapped into two dimensions: linguistic ability (vocabulary $\rightarrow$ grammar $\rightarrow$ pragmatics) and problem-solving capability (topic modeling $\rightarrow$ context understanding $\rightarrow$ generation $\rightarrow$ reasoning).

### Nine Evaluation Tasks

| Task | Instances per Language | Evaluated Skills |
|------|-----------|---------|
| Vocabulary Understanding | 1,000 | Lexical Knowledge |
| Topic Classification (Sentence) | 492 | Topic Modeling |
| Topic Classification (Document) | 600 | Topic Modeling |
| Reading Comprehension | 250 | Context Understanding |
| Response Selection | 507 | Pragmatic Reasoning |
| Headline Generation | 1,000 | Text Generation |
| Machine Translation (Article) | 1,012 | Translation Ability |
| Machine Translation (Dialogue) | 773 | Translation Ability |
| Mathematical Reasoning | 250 | Symbolic Reasoning |

## Experiments

### Main Results: Average Score per Language for Each Model

| Model | Tibetan (bo) | Uyghur (ug) | Kazakh (kk) | Mongolian (mn) | Average |
|------|---------|------------|------------|-----------|------|
| Qwen-2.5-7B | 29.4 | 48.0 | 37.0 | 24.9 | 34.8 |
| Qwen-3-8B | 34.5 | 56.5 | 46.7 | 28.7 | 41.6 |
| Gemma-3-12B | **53.3** | **63.7** | **57.5** | 25.1 | **49.9** |
| EMMA-500-7B | 25.3 | 42.5 | 27.4 | 17.8 | 28.2 |
| GPT-4.1 | 57.0 | 72.0 | 65.9 | 27.2 | 55.5 |
| Gemini-2.0-Flash | **72.9** | **75.0** | **70.9** | **66.8** | **71.4** |

### Ablation Study: Comparison of Evaluations Using Machine-Translated vs. Human-Translated Data

| Language | Reading Comprehension (Drop %) | Response Selection (Drop %) | Mathematical Reasoning (Drop %) |
|------|-------------|-------------|-------------|
| Tibetan | 40.0 (-21%) | 36.9 (-15%) | 11.9 (-52%) |
| Uyghur | 41.8 (-19%) | 42.2 (-17%) | 31.3 (-29%) |
| Kazakh | 40.3 (-19%) | 32.3 (-16%) | 19.3 (-22%) |

### Key Findings

1. **Mongolian Represents the Most Significant Deficiency**: Even the best open-source model, Gemma-3, performs only slightly better than random on Mongolian. The tokenization efficiency for traditional Mongolian is extremely low (GPT-4.1 requires 432 tokens per sentence, compared to only 54 for Cyrillic Mongolian).
2. **Specialized Multilingual Adaptation Models Struggle to Excel**: Although models like EMMA-500 and BayLing-2 are explicitly trained on minority language data, they underperform compared to native multilingual LLMs (e.g., Gemma-3, Qwen-3).
3. **Imbalanced Skill Profiles**: LLMs possess basic vocabulary understanding and topic modeling abilities, but remain highly deficient in tasks requiring syntactic knowledge, such as generation and translation.
4. **Severe Script-Switching Performance**: In Mongolian headline generation, GPT-4o-mini erroneously switches to the Cyrillic script in 36% of cases, and this figure rises to 95% for Kazakh.
5. **Distortion Caused by Machine-Translated Evaluation Data**: Evaluating with NLLB-translated data results in a performance drop of up to 52% (specifically in mathematical reasoning).

## Highlights & Insights

- The first LLM benchmark to systematically cover the non-mainstream writing systems of China's minority languages.
- A cross-task parallel design that prevents evaluation bias inherent to single task formats.
- Human-translated data that guarantees the authenticity and reliability of the evaluation.
- Uncovering a strong correlation between tokenization efficiency and model performance.

## Limitations & Future Work

- The benchmark only covers four minority languages, leaving out others such as Zhuang and Hmong.
- The dataset size for some tasks is relatively small (e.g., only 250 instances for mathematical reasoning), which may affect statistical significance.
- The benchmark primarily evaluates in-context learning (ICL) capabilities, while leaving the performance of fine-tuned models unexplored.
- The completeness of the skill classification framework requires further validation.

## Related Work & Insights

- **Multilingual Benchmarks**: XTREME (Hu et al., 2020), MEGA (Ahuja et al., 2023), Belebele (Bandarkar et al., 2024)
- **Low-Resource Language NLP**: EMMA-500 (Ji et al., 2024), LLaMAX-3 (Lu et al., 2024)
- **China's Minority Languages**: MC2 Corpus (Zhang et al., 2024b), WCM (Yang et al., 2022)
- **Translation Evaluation**: FLORES+ (NLLB Team et al., 2024), SIB-200 (Adelani et al., 2024)

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Total Score | 8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](../../ICLR2026/multilingual_mt/sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)
- [\[ACL 2025\] Did Translation Models Get More Robust Without Anyone Even Noticing?](translation_robustness.md)
- [\[ACL 2025\] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)
- [\[ICML 2026\] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](../../ICML2026/multilingual_mt/toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)
- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)

</div>

<!-- RELATED:END -->
