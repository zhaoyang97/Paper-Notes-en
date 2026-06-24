---
title: >-
  [Paper Note] GG-BBQ: German Gender Bias Benchmark for Question Answering
description: >-
  [ACL 2025][Social Computing][Gender Bias] This paper translates the gender subset of the English BBQ bias benchmark to German, creating the GG-BBQ German gender bias evaluation benchmark after manual review. It uncovers the limitations of machine translation in constructing bias evaluation datasets and evaluates the bias performance of multiple German LLMs.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Gender Bias"
  - "German LLMs"
  - "Bias Benchmark"
  - "Question Answering"
  - "Fairness Evaluation"
date: 2026-05-08
content_hash: 960d3678972df5dc
---

# GG-BBQ: German Gender Bias Benchmark for Question Answering

**Conference**: ACL 2025  
**arXiv**: [2507.16410](https://arxiv.org/abs/2507.16410)  
**Code**: [github.com/shalakasatheesh/GG-BBQ](https://github.com/shalakasatheesh/GG-BBQ)  
**Area**: Social Computing  
**Keywords**: Gender Bias, German LLMs, Bias Benchmark, Question Answering, Fairness Evaluation  

## TL;DR

This paper translates the gender subset of the English BBQ bias benchmark to German, creating the GG-BBQ German gender bias evaluation benchmark after manual review. It uncovers the limitations of machine translation in constructing bias evaluation datasets and evaluates the bias performance of multiple German LLMs.

## Background & Motivation

**Background**: The widespread application of LLMs across various fields has drawn attention to representational harm, making bias evaluation a core topic in trustworthy AI. Parrish et al. (2022) proposed the BBQ (Bias Benchmark for QA) benchmark, covering 9 social dimensions, but it is limited to US English contexts.

**Limitations of Prior Work**: NLP bias evaluation is heavily concentrated on English (Dhole et al., 2021; Hovy & Prabhumoye, 2021), and there is a severe lack of bias evaluation resources for other languages like German. Crucially, machine translation from English to German has fundamental issues in gender bias evaluation—German is a **grammatically gendered language**, where gender-neutral expressions in English often lose neutrality upon translation.

**Key Challenge**: The core requirement of bias evaluation datasets is to remain gender-neutral in certain contexts to test whether the model relies on stereotypes. However, the grammatical gender system in German prevents direct machine translation from maintaining this neutrality. For example, "Who is the secretary?" is consistently translated into "Wer ist die Sekretärin?" (female secretary) rather than a neutral expression.

**Goal**: To construct a reliable gender bias evaluation benchmark for German LLMs.

**Key Insight**: A systematic approach combining machine translation and manual review to create two sub-datasets: Subset-I (using gendered group terms) and Subset-II (using proper names).

**Core Idea**: Based on the machine translation of the English BBQ gender subset, grammatical gender issues are corrected through over 15 hours of manual review to create a German bias evaluation benchmark.

## Method

### Overall Architecture

1. **Machine Translation**: Translate 50 English templates into German using facebook/nllb-200-3.3B.
2. **Manual Review and Correction**: Bilingual language experts spent over 15 hours correcting translation errors.
3. **Template Expansion**: Due to grammatical gender requirements (e.g., friend → Freund/Freundin), templates were expanded from 50 to 167.
4. **Dataset Generation**: Final evaluation samples were generated from templates by substituting group terms and names.

### Key Designs

**Core Challenges in Translation**:
- **Loss of Gender Neutrality**: "reporters" was translated to "Berichterstatter" (masculine form), which needs to be modified to "Berichterstatter/Berichterstatterinnen".
- **Adjectival Endings**: "friendly woman/man" must be translated as "freundliche Frau/freundlicher Mann" respectively.
- **Non-binary Gender Expression**: German lacks a consensus on non-binary pronouns, requiring sentence rewrites to avoid pronouns.
- **Cultural Differences**: "middle school" has no direct equivalent in the German education system.
- **Gendered Occupations**: Questions must be rewritten to remain neutral, e.g., "Who is the secretary?" is changed to "Wer assistierte dem Vorstand?" (Who assisted the board?).

**Dataset Structure**: Each template generates 4 QA samples (as shown in Figure 1):
1. Ambiguous context + positive question
2. Ambiguous context + negative question
3. Ambiguous + disambiguated context + positive question
4. Ambiguous + disambiguated context + negative question

**Two Subsets**:
- **Subset-I**: Uses gendered group terms (e.g., Mann/Frau), with 484 ambiguous + 484 disambiguated samples.
- **Subset-II**: Uses proper names (e.g., Emma, Matteo, Kim), with 2484 ambiguous + 2484 disambiguated samples.

### Bias Evaluation Metrics

Accuracy: $\text{Acc}_{\text{amb}} = \frac{n_{au}}{n_a}$ (the proportion of correctly answering "unknown" in ambiguous contexts)

Bias score (adopting the method of Jin et al., 2024):

$$\text{diff-bias}_{\text{amb}} = \frac{n_{ab} - n_{ac}}{n_a}$$

$$\text{diff-bias}_{\text{disamb}} = \frac{n_{bb}}{n_b} - \frac{n_{cc}}{n_c}$$

An unbiased model should exhibit: Accuracy = 1.0, diff-bias = 0. A fully biased model: diff-bias = 1.0, ambiguous accuracy = 0.

## Key Experimental Results

### Subset-I Ambiguous Context Results

| Model | Acc_amb↑ | diff-bias_amb | bias_max |
|------|:---:|:---:|:---:|
| leo-hessianai-13b-chat | **0.684** | 0.124 | 0.316 |
| Mistral-7B-Instruct-v0.3 | 0.628 | 0.120 | 0.372 |
| Mistral-7B-v0.3 | 0.601 | 0.149 | 0.399 |
| Llama-3.2-3B-Instruct | 0.570 | 0.203 | 0.430 |
| Llama-3.1-70B-Instruct | 0.537 | 0.426 | 0.463 |
| leo-hessianai-13b | 0.496 | **0.076** | 0.504 |

### Subset-II Disambiguated Context Results

| Model | Acc_disamb↑ | diff-bias_disamb | bias_max |
|------|:---:|:---:|:---:|
| Llama-3.1-70B-Instruct | **0.980** | 0.040 | 0.041 |
| Llama-3.1-70B | 0.973 | 0.016 | 0.053 |
| Mistral-7B-Instruct-v0.3 | 0.738 | 0.125 | 0.524 |
| DiscoLeo-Instruct-8B | 0.701 | -0.551 | 0.599 |
| Llama-3.2-3B-Instruct | 0.612 | -0.206 | 0.776 |

### Key Findings

1. **All models exhibit bias**: Regardless of model size or whether they underwent instruction tuning, all evaluated LLMs exhibit gender bias.
2. **Bias direction varies with subsets**: All models in Subset-I show **positive bias** (aligning with stereotypes) in ambiguous contexts, whereas in Subset-II they all show **negative bias** (counter-stereotypes).
3. **Complex relationship between model size and bias**: Large models (Llama-3.1-70B) perform exceptionally well in disambiguated contexts (0.98 accuracy) but exhibit bias scores near the maximum in ambiguous contexts.
4. **Inconsistent effects of instruction tuning**: It improves performance for Llama-3.2-3B (accuracy↑, bias↓) but exaggerates bias in leo-hessianai-13b.
5. **Smaller models are sometimes superior**: Mistral-7B-v0.3 and leo-hessianai-13b outperform the 70B larger model in ambiguous contexts.
6. **Machine translation cannot be directly used for bias evaluation**: Original machine translations contain numerous gender-biased errors, making over 15 hours of manual correction indispensable.

## Highlights & Insights

- **Unveiling the systematic issue of machine translation**: Machine translation introduces additional bias when constructing gender bias evaluation datasets—a challenge not yet fully recognized by the research community.
- **Necessity of template expansion**: The expansion from 50 to 167 templates fully reflects the translation complexities of grammatically gendered languages.
- **Bias direction flip phenomenon**: Group terms (Mann/Frau) and proper names lead to opposite bias directions, a finding that warrants deeper investigation.
- **Improved bias calculation method**: Adopting the independent calculation of ambiguous/disambiguated bias scores from Jin et al. (2024) avoids the misrepresentation in Parrish's original method when bias directions differ.

## Limitations & Future Work

- Only the gender identity subset of BBQ is translated; the remaining 8 social dimensions are not covered.
- Reliance on a single language expert for review may introduce annotator bias.
- The translated dataset may not fully capture gender biases unique to the German cultural context.
- Intersectional bias (e.g., race × gender) is not considered.
- Decoding parameters (temperature, top_p) may affect bias performance, and only 5 prompts were tested.
- Future work needs to construct native bias evaluation datasets for the German cultural context from scratch.

## Related Work & Insights

- Multilingual extensions of BBQ have covered Dutch, Turkish, Spanish, Basque, Chinese, Korean, and Japanese, forming a systematic research ecosystem.
- Nie et al. (2024) directly used machine translation to evaluate Germanic language bias; this paper points out the risks of this approach.
- Bartl et al. (2020) analyzed German gender bias in BERT contextual embeddings—showing the transition from intrinsic to extrinsic evaluation.
- Zhou et al. (2019) evaluated word embedding bias for grammatically gendered languages such as Spanish and French—pioneering cross-lingual bias research.

## Rating

- **Novelty**: ⭐⭐⭐ — The core work is translating and correcting the BBQ dataset. While methodological innovation is limited, the data resource contribution is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 10 models (5 base + 5 instruct) × 2 subsets × 2 context conditions, providing a comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — The discussion of translation challenges is very detailed and inspiring, representing the most valuable part of the paper.
- **Value**: ⭐⭐⭐⭐ — Fills the gap in German bias evaluation. The findings during the translation process offer important warnings for the multilingual bias research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades](taz2024full_analysing_german_newspapers_for_gender_bias_and_discrimination_acros.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[ECCV 2024\] GRACE: Graph-Based Contextual Debiasing for Fair Visual Question Answering](../../ECCV2024/social_computing/grace_graph-based_contextual_debiasing_for_fair_visual_question_answering.md)
- [\[ACL 2025\] Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations](translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)
- [\[ACL 2026\] GKnow: Measuring the Entanglement of Gender Bias and Factual Gender](../../ACL2026/social_computing/gknow_measuring_the_entanglement_of_gender_bias_and_factual_gender.md)

</div>

<!-- RELATED:END -->
