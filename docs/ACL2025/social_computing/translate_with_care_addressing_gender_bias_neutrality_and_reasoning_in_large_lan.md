---
title: >-
  [Paper Note] Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations
description: >-
  [ACL2025][Social Computing][Machine Translation] The Translate-with-Care (TWC) dataset (comprising 3,950 translation challenges across six genderless languages) is proposed to systematically reveal gender biases and reasoning errors in genderless-to-gendered language translation within models like GPT-4 and Google Translate. By fine-tuning mBART-50, this work substantially outperforms closed-source LLMs in bias mitigation and translation accuracy.
tags:
  - "ACL2025"
  - "Social Computing"
  - "Machine Translation"
  - "Gender Bias"
  - "Genderless Languages"
  - "Pronoun Disambiguation"
  - "Low-resource Languages"
  - "mBART-50"
date: 2026-05-08
content_hash: 200fcd80c64c7f7a
---

# Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations

**Conference**: ACL2025  
**arXiv**: [2506.00748](https://arxiv.org/abs/2506.00748)  
**Code**: [GitHub](https://github.com/PardisTagworksopen/TWC)  
**Area**: Social Computing  
**Keywords**: Machine Translation, Gender Bias, Genderless Languages, Pronoun Disambiguation, Low-resource Languages, mBART-50

## TL;DR

The Translate-with-Care (TWC) dataset (comprising 3,950 translation challenges across six genderless languages) is proposed to systematically reveal gender biases and reasoning errors in genderless-to-gendered language translation within models like GPT-4 and Google Translate. By fine-tuning mBART-50, this work substantially outperforms closed-source LLMs in bias mitigation and translation accuracy.

## Background & Motivation

**Semantic disambiguation remains a core challenge in machine translation**: Ambiguity and coreference resolution perform poorly in NMT systems, and while LLMs have made progress, shortcomings persist.

**Translation of genderless languages is neglected**: When translating genderless languages such as Persian, Indonesian, Finnish, and Turkish into English, pronoun gender selection encounters unique challenges.

**Existing benchmarks do not cover genderless languages**: Benchmarks like WinoMT and MT-GenEval mainly target gendered languages such as Spanish and French, resulting in a lack of evaluation for genderless languages.

**Models generally favor masculine pronouns**: All models tend to prefer masculine pronouns (he/his) when gender stereotypes influence translation decisions.

**Poorer performance in low-resource languages**: Scarcity of training data and complex grammatical structures lead to poor generalization of LLMs on low-resource languages.

**Social harms of biased translation**: Biased translations reinforce gender stereotypes, damage user trust, and hinder cross-cultural communication.

## Method

### Overall Architecture

Construct the TWC dataset $\rightarrow$ Evaluate various translation models using TWC $\rightarrow$ Fine-tune mBART-50 to eliminate biases and reasoning errors $\rightarrow$ Validate cross-lingual generalization capability. The dataset contains three categories of challenges: Bias, Neutrality, and Reasoning, covering six genderless languages.

### Key Design 1: TWC Dataset Construction

- **Function**: Constructs 3,950 translation challenge instances, each containing a source sentence, candidate antecedents, a target pronoun, the correct translation, and the challenge category.
- **Mechanism**: Employs Tree-of-Experts (ToE) prompting to guide GPT-4 in generating English sentences, which are then translated into the target languages and post-edited by humans. Additionally, 514 human-written instances are included to cover culture-specific scenarios.
- **Design Motivation**: Automated generation ensures scale and diversity, while human post-editing guarantees translation quality. The pronoun "one" is used instead of "they" to avoid singular/plural ambiguity. The types of antecedents include Personal Names, Titles, and Roles to ensure broad coverage.

### Key Design 2: Multi-Dimensional Evaluation System

- **Function**: Evaluates the pronoun accuracy and translation quality of GPT-4, Google Translate, mBART-50, NLLB-200, and SeamlessM4T v2 on TWC.
- **Mechanism**: Automatically extracts pronouns (such as he/she/they/one) from the translation output and compares them with labeled ground truth to calculate category-specific accuracies. Concurrently, metrics such as BLEU, ROUGE, METEOR, TER, and COMET are utilized to evaluate overall translation quality.
- **Design Motivation**: Pronoun accuracy directly reflects gender bias and reasoning ability, while general metrics complement the evaluation of overall translation quality. Multi-model comparisons reveal common issues across different architectures.

### Key Design 3: mBART-50 Fine-Tuning Strategy

- **Function**: Fine-tunes mBART-50 on the TWC training set to create two versions: mBART-ft-TWC (multilingual fine-tuning) and mBART-id-ft-TWC (Indonesian-only fine-tuning).
- **Mechanism**: Data augmentation via syntactic transformation expands 1,810 training instances to 5,430 (reordering antecedents, modifying punctuation and sentence structures), with early stopping applied to prevent overfitting. The test set contains unseen languages (Estonian, Azerbaijani), human-generated content, and novel semantic elements (Titles, Roles).
- **Design Motivation**: Data augmentation breaks the dependency on specific syntactic patterns. The Indonesian-only fine-tuning version is designed to validate the cross-lingual transfer hypothesis—whether pronoun resolution capability can transfer across different language families.

### Loss & Training

Uses the standard sequence-to-sequence cross-entropy loss, which is the original translation objective function of mBART-50, to optimize the likelihood of the model's translation outputs on TWC training samples during fine-tuning.

## Key Experimental Results

### Main Results: Model Overall Accuracy Comparison

| Model | TWC Overall Accuracy | Reasoning | Bias | Neutrality |
|------|-------------|-----------|------|------------|
| **mBART-ft-TWC** | **87.6%** | High | High | High |
| mBART-id-ft-TWC | 78.28% | — | — | — |
| GPT-4 | 35.4% | 89.3% | Low | Low |
| Google Translate | 22.8% | 55.5% | Low | Low |
| mBART-50 (Original) | 16.1% | 40.2% | — | — |
| NLLB-200 1.3B | 8.9% | 22.2% | — | — |

### Ablation Study: Cross-Lingual Transfer and Gender Bias Distribution

| Dimension of Analysis | Key Findings |
|---------|---------|
| Cross-lingual Transfer | Fine-tuning solely on Indonesian significantly improves Persian performance, with the reasoning category doubling across all languages. |
| Gender Preference | Google Translate uses masculine pronouns 4–6 times more often than feminine pronouns in leadership/occupational scenarios. |
| Content Omission | Models omit up to 32% of text in sentences that require reasoning-based disambiguation. |
| Unseen Languages | Estonian and Azerbaijani (unseen during training) still achieve high accuracy. |

### Key Findings

1. All models perform poorly on Bias and Neutrality categories, with near-zero accuracy prior to fine-tuning.
2. GPT-4 performs best in the Reasoning category (89.3%) but heavily favors masculine pronouns in Bias/Neutrality.
3. Fine-tuning solely on Indonesian data yields cross-lingual improvements (Indonesian $\rightarrow$ Persian), suggesting the cross-lingual transferability of pronoun disambiguation capabilities.
4. The fine-tuned open-source mBART-50 comprehensively outperforms closed-source systems such as GPT-4 and Google Translate.

## Highlights & Insights

1. **Precise Problem Definition**: Focusing on the "genderless $\rightarrow$ gendered" translation direction, filling an important gap in existing benchmarks.
2. **Unexpected Cross-Lingual Transfer Findings**: Fine-tuning only on Indonesian dramatically enhances Persian performance (despite completely different language families, scripts, and grammar), revealing the language-agnostic character of pronoun disambiguation.
3. **Open-Source Beats Closed-Source**: The fine-tuned mBART-50 (open-source) significantly outperforms GPT-4 (35.4%) with an overall accuracy of 87.6%, proving the value of targeted fine-tuning.
4. The **three-category challenge taxonomy (Bias/Neutrality/Reasoning)** is clear and actionable, facilitating targeted improvements in subsequent research.

## Limitations & Future Work

1. Translation quality depends on machine translation followed by human post-editing rather than being entirely human-translated, which may leave subtle errors.
2. The choice of "one" as a gender-neutral pronoun is relatively conservative; the feasibility of "they" or neopronouns is not explored.
3. Fine-tuning is only validated on the mBART-50 architecture and is not extended to larger open-source models (such as LLaMA or Mistral).
4. The evaluation focuses exclusively on pronoun accuracy and automated metrics, lacking human evaluation of the overall naturalness and fluency of translations.
5. The data primarily covers six genderless languages, whereas there are many more genderless languages (such as Japanese, or certain contexts in Chinese).

## Related Work & Insights

### vs WinoMT (Savoldi et al., 2021)

WinoMT focuses on gender bias evaluation in translations of gendered languages (Spanish, French, etc.). TWC fills the evaluation gap for genderless languages and adds a Neutrality category (which is absent in WinoMT). TWC's three-way classification framework is more granular.

### vs MT-GenEval (Currey et al., 2022)

MT-GenEval evaluates gender translation at sentence and paragraph levels but is similarly restricted to gendered languages. TWC's coverage of low-resource genderless languages is a unique contribution, and its findings regarding the cross-lingual transfer of the fine-tuning strategy go beyond the scope of MT-GenEval's evaluation.

### Insights

1. Targeted fine-tuning on a small dataset may resolve specific bias issues more effectively than expanding model scale.
2. Cross-lingual transfer learning shows great potential in pronoun/gender handling and warrants validation on more language pairs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Translating from genderless to gendered languages is an overlooked yet important problem, and the design of the three-category system is exquisite.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experiments covering six languages, multiple models, and cross-lingual transfer, though human evaluation is missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich examples, and intuitive data presentation.
- **Value**: ⭐⭐⭐⭐ — Both the dataset and fine-tuned models are open-source, directly driving research in translation fairness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[NeurIPS 2025\] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](../../NeurIPS2025/social_computing/any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [\[ACL 2025\] GG-BBQ: German Gender Bias Benchmark for Question Answering](gg-bbq_german_gender_bias_benchmark_for_question_answering.md)
- [\[ACL 2025\] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades](taz2024full_analysing_german_newspapers_for_gender_bias_and_discrimination_acros.md)

</div>

<!-- RELATED:END -->
