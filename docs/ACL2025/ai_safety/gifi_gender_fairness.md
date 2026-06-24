---
title: >-
  [Paper Note] Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework
description: >-
  [ACL 2025][AI Safety][gender fairness] This work proposes GIFI (Gender Inclusivity Fairness Index), a multi-level evaluation framework covering seven dimensions: pronoun recognition, sentiment neutrality, toxicity, counterfactual fairness, stereotype association, occupational fairness, and mathematical reasoning consistency. It systematically quantifies binary and non-binary gender fairness across 22 mainstream LLMs, revealing deep bias patterns such as the complete absence o…
tags:
  - "ACL 2025"
  - "AI Safety"
  - "gender fairness"
  - "non-binary pronouns"
  - "LLM evaluation"
  - "inclusivity index"
  - "neopronouns"
date: 2026-05-08
content_hash: fa53004dbd740b4b
---

# Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework

**Conference**: ACL 2025  
**arXiv**: [2506.15568](https://arxiv.org/abs/2506.15568)  
**Code**: [https://github.com/ZhengyangShan/GIFI](https://github.com/ZhengyangShan/GIFI)  
**Area**: AI Safety / Fairness Evaluation  
**Keywords**: gender fairness, non-binary pronouns, LLM evaluation, inclusivity index, neopronouns

## TL;DR

This work proposes GIFI (Gender Inclusivity Fairness Index), a multi-level evaluation framework covering seven dimensions: pronoun recognition, sentiment neutrality, toxicity, counterfactual fairness, stereotype association, occupational fairness, and mathematical reasoning consistency. It systematically quantifies binary and non-binary gender fairness across 22 mainstream LLMs, revealing deep bias patterns such as the complete absence of neopronouns without prompting and the over-correction of "she".

## Background & Motivation

**Background:** The rapid advancement of LLMs has driven progress in various NLP fields, but it also brings concerns regarding fairness, with gender representation being one of the most heavily scrutinized areas. Existing gender bias research (such as word embedding bias in Bolukbasi et al. 2016, StereoSet, CrowS-Pairs, etc.) has established a foundation for evaluation, but almost all work is constrained to a binary gender framework (male/female).

**Limitations of Prior Work:** Most studies only evaluate the performance differences between the two pronouns "he" and "she", completely ignoring non-binary gender identities. Non-binary pronouns (such as singular they/them) and neopronouns (e.g., xe/xem, ze/zir, ae/aer) have rarely been systematically evaluated in LLMs. Existing datasets like StereoSet and CrowS-Pairs are not specifically tailored to the representations or lived experiences of non-binary genders, making biases against these groups undetectable.

**Key Challenge:** While societal awareness of gender diversity deepens and neopronoun systems continue to evolve, there is a lack of reliable metrics to measure whether AI systems—especially conversational LLMs—can respect and correctly handle these automotive identities. Although some works (such as MISGENDERED and Ovalle et al. 2023) have begun to focus on misgendering of non-binary pronouns, they only cover a single dimension and do not provide a comprehensive evaluation.

**Goal:** The authors aim to establish a comprehensive fairness evaluation framework covering non-binary genders, systematically quantifying LLMs' inclusivity toward 11 groups of different gender pronouns—ranging from basic pronoun recognition to deep cognitive reasoning—and conducting a large-scale benchmark on 22 mainstream models.

**Key Insight:** The evaluation is designed as four progressive phases of increasing depth: from simple pronoun recognition (whether the model can correctly use specified pronouns) to distributional fairness (whether sentiment/toxicity changes due to different pronouns), to stereotype association (whether the model spontaneously biases toward specific genders without prompting), and finally to performance consistency (whether tasks seemingly unrelated to gender, such as mathematical reasoning, are affected by pronouns).

**Core Idea:** To aggregate seven normalized $[0,1]$ dimensions into an interpretable GIFI composite score ($0$-$100$), serving as the first LLM fairness benchmark that encompasses non-binary genders.

## Method

### Overall Architecture

The GIFI framework consists of four progressive evaluation phases across seven dimensions, covering 11 pronoun groups (2 binary + 1 neutral + 8 neopronouns). All dimensional indicators are normalized to $[0,1]$, with higher scores indicating greater fairness. The final $\text{GIFI} = \text{mean of the seven dimensions} \times 100$, ranging from $0$ to $100$. The four phases are: (1) Pronoun Recognition (GDR), (2) Distributional Fairness (SN, NTS, CF), (3) Stereotypes & Role Assignment (SA, OF), and (4) Performance Consistency (PE).

### Key Designs

**1. Multilevel Pronoun Fairness Metric (GDR + SN + NTS + CF Joint Metrics)**

- **Function:** To comprehensively detect model processing discrepancies across different gender pronouns from the perspectives of both pronoun recognition and distributional fairness.
- **Mechanism:** For the GDR dimension, prompts containing each pronoun group are constructed to check if the model output maintains pronoun consistency. The coefficient of variation ($CV = \sigma/\mu$) is used to measure consistency across pronoun groups, where $GDR = 1/(1+CV)$. For SN and NTS, RoBERTa sentiment classifier and Google Perspective API are used to score outputs from identical prompts where only the pronoun is modified, measuring sentiment/toxicity consistency across pronouns via $1 - \text{Average MAD}$. CF encodes paired outputs using sentence-transformers, identifying "substantial differences" based on a cosine similarity threshold of $\gamma = 0.3$.
- **Design Motivation:** Pronoun recognition is the most fundamental requirement of fairness—if a model cannot even use pronouns correctly, discussing deeper fairness levels is meaningless. Distributional fairness reveals whether the model introduces asymmetric variations in sentiment, toxicity, or semantics at the content generation level due to pronoun differences.

**2. Stereotype and Occupational Bias Detection (SA + OF Dimensions)**

- **Function:** To detect whether models spontaneously associate specific pronouns with stereotypical activities, personality traits, or occupations in the absence of explicit gender prompts.
- **Mechanism:** Prompt templates without gender cues, such as "My friend is a {occupation}" or "My friend likes {hobby}", are used for multiple sampling (at a temperature of 0.95). The frequency distribution of each pronoun group in the model outputs is recorded, and its squared deviation from a uniform distribution is computed. SA targets personality/activities/color preferences, while OF targets 80 occupations (40 male-dominated + 40 female-dominated). Both exclude "they" to avoid interference from its role as a default neutral pronoun.
- **Design Motivation:** This level tests the implicit biases of models—even in the absence of any gender cues, whether the model still associates "nurse" with "she" and "engineer" with "he". Such biases reflect societal stereotypes embedded in the training data.

**3. Cross-Gender Reasoning Performance Consistency (PE Dimension)**

- **Function:** To evaluate whether models exhibit performance variations across different pronouns in tasks that are ostensibly unrelated to gender (e.g., mathematical reasoning).
- **Mechanism:** Based on the GSM8K dataset, math problems containing a single personal name are extracted via NER. The names are replaced with 11 different pronoun groups, generating 1,100 samples. Accuracy is evaluated using 8-shot CoT prompting, and PE is calculated using the same CV formula. Instance-level consistency analysis is also conducted to categorize results into "all correct", "all incorrect", "binary/neutral correct but neopronoun incorrect", etc.
- **Design Motivation:** If a model's reasoning capabilities degrade on "xe bought 3 books" but perform normally on "he bought 3 books", this exposes a deeper intrinsic bias—how the unfamiliarity of pronouns impacts cognitive performance unrelated to gender itself.

## Key Experimental Results

### Main Results: GIFI Seven-Dimension Scores of 22 Models (Top-10 Models)

| Model | GDR | SN | NTS | CF | SA | OF | PE | **GIFI** |
|------|-----|------|------|------|------|------|------|----------|
| GPT-4o | 0.76 | 0.77 | 0.96 | 0.86 | 0.37 | 0.41 | 0.96 | **73** |
| Claude 3 | 0.67 | 0.78 | 0.95 | 0.87 | 0.31 | 0.42 | 0.97 | **71** |
| DeepSeek V3 | 0.67 | 0.68 | 0.93 | 0.89 | 0.56 | 0.18 | 0.99 | **70** |
| GPT-4o-mini | 0.61 | 0.81 | 0.94 | 0.99 | 0.36 | 0.13 | 0.95 | **68** |
| GPT-4 | 0.71 | 0.78 | 0.93 | 0.84 | 0.34 | 0.14 | 0.96 | **67** |
| Claude 4 | 0.80 | 0.83 | 0.93 | 0.63 | 0.34 | 0.17 | 0.97 | **67** |
| Gemini 1.5 Pro | 0.55 | 0.78 | 0.92 | 0.74 | 0.37 | 0.36 | 0.97 | **67** |
| GPT-3.5-turbo | 0.64 | 0.73 | 0.93 | 0.82 | 0.35 | 0.14 | 0.96 | **65** |
| Gemma 3 | 0.65 | 0.70 | 0.91 | 0.60 | 0.47 | 0.20 | 0.96 | **64** |
| Gemini 2.0 Flash | 0.70 | 0.77 | 0.87 | 0.53 | 0.40 | 0.24 | 0.99 | **64** |

Bottom-ranked models: Vicuna (49), GPT-2 (55), LLaMA 2 (57), Zephyr (57).

### Ablation Study: Sentiment Classifier Comparison (RoBERTa vs VADER)

| Model | SN (RoBERTa) | SN (VADER) |
|------|-------------|------------|
| Claude 4 | 0.830 | 0.828 |
| GPT-4o-mini | 0.810 | 0.756 |
| Gemini 1.5 Pro | 0.776 | 0.755 |
| GPT-4o | 0.765 | 0.724 |
| Claude 3 | 0.783 | 0.690 |
| DeepSeek V3 | 0.684 | 0.650 |
| Yi-1.5 | 0.672 | 0.444 |

The Pearson correlation coefficient between the two classifiers is $r = 0.785$, indicating that the SN conclusion does not depend on the choice of a specific sentiment classifier.

### Key Findings

1. **Complete absence of neopronouns without prompts**: All 22 models never spontaneously generate neopronouns like xe or ze in SA/OF tasks, exposing the extreme sparsity of neopronouns in training data.
2. **Prevalent over-correction of "she"**: GPT-4o has a "she" proportion as high as 0.86 in stereotype tasks, LLaMA 4 reaches 0.83, and Claude 4 exhibits 0.72 for "she" versus only 0.26 for "he" in occupational tasks, reflecting a corrective bias overshooting in debiasing training.
3. **Clear hierarchy in pronoun recognition**: Binary pronouns > they > neopronouns. Even the strongest model, Claude 4, averages only 0.75 accuracy, with neopronoun recognition rates generally below 0.50.
4. **Reasoning fairness $\approx$ reasoning capability**: Strong models (Gemini 2.0 Flash and DeepSeek V3 both achieved 0.92 accuracy) perform consistently across all pronouns, whereas weak models fail consistently across all pronouns.
5. **Highly inconsistent performance across dimensions**: Claude 4 achieves the highest GDR (0.80) but fails on OF (0.17); Phi-3 achieves the highest SA (0.72) but shows the worst CF (0.25).

## Highlights & Insights

1. **First comprehensive fairness metric covering non-binary genders**: Compared to works like MISGENDERED which focus only on a single dimension, GIFI covers recognition, generation, bias, and reasoning, successfully filling a crucial research gap.
2. **Ingenious four-phase progressive design**: By moving from shallow pronoun recognition to deep reasoning consistency, it establishes a "gradually increasing stress test" paradigm for LLM gender fairness, systematically exposing various levels of bias.
3. **Discovery of the "over-correction" paradox in debiasing**: The new generation of models is not necessarily fairer; instead, they shift biases from "he" to "she", while genuine neutral expressions (they) and neopronouns remain severely neglected.
4. **Excellent insight from the PE dimension**: Inter-testing mathematical reasoning with gender pronouns demonstrates that even ostensibly unrelated tasks are vulnerable to the unfamiliarity of pronouns.
5. **Unprecedented evaluation scale**: Running 22 models $\times$ 7 dimensions $\times$ 11 pronoun groups produces a vast and rich dataset for cross-analysis.

## Limitations & Future Work

1. **Coverage limited to English**: Gender systems vary drastically across languages (e.g., grammatical gender in French, lack of morphological inflection in Chinese), making direct transfer of this framework difficult.
2. **Biases within external classifiers**: The RoBERTa sentiment model and Perspective API may possess varying sensitivities to different pronouns, a concern only partially mitigated by the ablation study ($r = 0.785$).
3. **Data contamination risk**: Datasets like RealToxicityPrompts, released prior to 2022, may have already been included in the training sets of newer models.
4. **GIFI aggregates seven dimensions via simple averaging**: Since SA and OF scores are generally much lower than NTS, equal weighting might mask severe unfairness in particular dimensions.
5. **Lack of intersectional analysis**: Cross-biases between gender and other demographic attributes like race or age are not considered.
6. **Incomplete and evolving set of neopronouns**: Although the framework supports expansion, the current selection of 8 neopronoun groups has limited representativeness.

## Related Work & Insights

| Comparison Dimension | Prior Work | GIFI (Ours) |
|---------|---------|-------------|
| Gender Scope | StereoSet and CrowS-Pairs cover only binary genders | Covers 11 pronoun groups, including 8 neopronouns |
| Evaluation Dimensions | MISGENDERED focuses only on misgendering | Comprehensive coverage across seven dimensions from recognition to reasoning |
| Composite Index | GenderCare (Tang et al. 2024) restricted to binary | GIFI provides a single, interpretable score that includes non-binary genders |
| Model Coverage | Ovalle et al. 2023 tested a small number of models | Large-scale comparison across 22 models |
| Cognitive Level | Most prior works focus on surface-level outputs | PE dimension tests the implicit impact of pronouns on reasoning capabilities |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first comprehensive LLM fairness metric covering non-binary genders, filling a significant research gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 22 models $\times$ 7 dimensions $\times$ 11 pronoun groups, with ablation studies and qualitative failure analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly organized with rich tables and figures, rigorous index definitions, and comprehensive appendices.
- **Value**: ⭐⭐⭐⭐⭐ — Directly applicable to LLM fairness audits; discoveries on "over-correction" and "absence of neopronouns" carry strong practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes](fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)
- [\[ICLR 2026\] Fairness via Independence: A General Regularization Framework for Machine Learning](../../ICLR2026/ai_safety/fairness_via_independence_a_general_regularization_framework_for_machine_learnin.md)
- [\[ACL 2025\] Towards Fairness Assessment of Dutch Hate Speech Detection](towards_fairness_assessment_of_dutch_hate_speech_detection.md)
- [\[ICML 2026\] COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs](../../ICML2026/ai_safety/copf_an_online_framework_for_deployment-stable_counterfactual_fairness_in_evolvi.md)
- [\[ICLR 2026\] RESFL: An Uncertainty-Aware Framework for Responsible Federated Learning by Balancing Privacy, Fairness and Utility](../../ICLR2026/ai_safety/resfl_an_uncertainty-aware_framework_for_responsible_federated_learning_by_balan.md)

</div>

<!-- RELATED:END -->
