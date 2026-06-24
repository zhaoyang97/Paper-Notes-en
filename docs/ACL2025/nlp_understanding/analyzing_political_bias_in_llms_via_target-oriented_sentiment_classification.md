---
title: >-
  [Paper Note] Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification
description: >-
  [ACL 2025][NLP Understanding][Political Bias] Proposes a political bias analysis framework for LLMs based on Target-Oriented Sentiment Classification (TSC). By substituting the names of 1,319 politicians into 450 political sentences and predicting sentiments using 7 models across 6 languages, this study defines an entropy-based inconsistency metric to quantify bias. The findings reveal that LLMs exhibit a positive bias toward left-wing and centrist politicians and a negative…
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Political Bias"
  - "Target-Oriented Sentiment Classification"
  - "LLM Bias"
  - "Multilingual Analysis"
  - "Bias Mitigation"
date: 2026-05-08
content_hash: e3311fde448a6cc8
---

# Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification

**Conference**: ACL 2025  
**arXiv**: [2505.19776](https://arxiv.org/abs/2505.19776)  
**Code**: Yes (coming soon)  
**Area**: NLP Understanding / LLM Bias Analysis  
**Keywords**: Political Bias, Target-Oriented Sentiment Classification, LLM Bias, Multilingual Analysis, Bias Mitigation

## TL;DR

Proposes a political bias analysis framework for LLMs based on Target-Oriented Sentiment Classification (TSC). By substituting the names of 1,319 politicians into 450 political sentences and predicting sentiments using 7 models across 6 languages, this study defines an entropy-based inconsistency metric to quantify bias. The findings reveal that LLMs exhibit a positive bias toward left-wing and centrist politicians and a negative bias toward the far-right, with larger models demonstrating stronger and more consistent biases.

## Background & Motivation

**Background**: LLMs are widely utilized in critical applications such as social media political discussion moderation and automated analysis of political events, but they simultaneously encode demographic and political biases.

**Limitations of Prior Work**: Existing research methods for LLM political bias suffer from significant limitations: (1) **Questionnaire-based approaches**—using questionnaires like the Political Compass to prompt LLM answers, which suffer from a low number of interactions, high sensitivity to prompt phrasing, and an inability to generalize to downstream applications; (2) **Generation-based approaches**—instructing LLMs to generate political texts and then analyzing the sentiment/stance, which is difficult to quantify and relies on the LLMs themselves for evaluation, thereby propagating bias.

**Key Challenge**: The bias analysis methodology itself should not introduce or propagate biases, yet existing approaches either lack statistical power or rely on the biased models themselves for evaluation.

**Goal**: How to systematically analyze LLM political bias in a large-scale, statistically robust manner that does not rely on self-evaluation by LLMs?

**Key Insight**: Cleverly leveraging the "inconsistency" of LLMs in TSC tasks—where sentiment predictions should not change when substituting different politicians' names in the same sentence, but actually become inconsistent due to changes in target entities—and using this inconsistency as the source of bias signals.

**Core Idea**: Utilizing the inconsistency of LLM sentiment classification (making different predictions after substituting names in the same sentence) as a probe for political bias, conducting a systematic analysis across multiple models and languages using approximately 25 million data points.

## Method

### Overall Architecture

A three-step pipeline: (1) Collect 1,319 diverse politicians (covering 8 political spectrums) and 450 multilingual political sentences from Wikidata + ParlGov → (2) Substitute each politician's name into each sentence, performing TSC predictions with 7 LLMs across 6 languages, generating approximately 25 million data points → (3) Utilize an entropy-based inconsistency metric and multi-level aggregation analysis (language/model/political spectrum/individual) to extract bias patterns.

### Key Designs

1. **Entropy-based Inconsistency Metric (IC)**
    - Function: Quantifies the variation in sentiment predictions of LLMs when target entities are substituted.
    - Mechanism: For each sentence $s_i$, collect the set of predictions $Y_i$ after substituting all politicians, calculate the entropy of its label distribution $H(Y_i)$, and average across all sentences to obtain $IC = \frac{1}{m}\sum_{i=1}^{m}H(Y_i)$.
    - Design Motivation: An unbiased LLM should produce the same sentiment prediction for any name (IC=0); a larger IC value indicates stronger entity-related bias.

2. **Multi-dimensional Politician Sampling and Representation**
    - Function: Ensures the representativeness of the analysis and controls for confounding factors.
    - Mechanism: Filter politicians from Wikidata → map them to 8 political spectrum categories (from far-left to far-right) using ParlGov → perform stratified sampling to ensure country and spectrum diversity → generate a control group using GPT-4 (retaining gender, birth year, and nationality but replacing names with fictional ones).
    - Design Motivation: It is necessary to isolate the influence of non-political attributes (e.g., gender, race) on bias.

3. **Bias Mitigation via Fictional Name Substitution**
    - Function: Validates the source of bias and provides a simple mitigation scheme.
    - Mechanism: Generate a fictional name for each politician that preserves non-political attributes, rerun the experiments using these fictional names, and compare the changes in inconsistency.
    - Design Motivation: If the IC decreases significantly after replacing with fictional names, it confirms that the bias primarily stems from political associations rather than demographic attributes.

### Loss & Training

This work does not involve training and is purely analytical. Key experimental setup: 7 models (Mistral-7B, Qwen-7B/72B, Llama3-8B/70B, Aya-Expanse-32B, GPT-4o-mini), 6 languages (English, French, Spanish, Russian, Arabic, Chinese), with temperature set to 0 to ensure deterministic outputs, generating approximately 25 million valid data points to guarantee statistical robustness.

## Key Experimental Results

### Main Results

| Dimension of Discovery | Key Results |
|---------|---------|
| Political Spectrum Bias | Left-wing (LL) and Centrist-Left (CL) receive positive sentiment, Far-Right (FR) and Right-wing (RR) receive negative sentiment |
| Language Effect | Biases in EN/FR/ES are stronger than in RU/AR/ZH |
| Scale Effect | Qwen-72B shows stronger and more consistent bias than Qwen-7B; Llama3-70B shows a similar trend |
| Multilingual Models | Aya-Expanse-32B (multilingually trained) exhibits stronger bias in non-Western languages |
| Individual Similarity | Politicians with similar political stances show highly similar sentiment predictions (Sánchez-Scholz: +0.92, Biden-Harris: +0.90) |

### Ablation Study

| Comparison | Original IC | Fake Name IC | Accuracy Change |
|------|-------------|-------------|-----------|
| Average across all models and languages | Higher | Significantly reduced | Slight increase |
| Residual sources of bias | - | Female names (+0.03) > Male names (-0.01); Russian/non-Western names receive negative scores | - |

### Key Findings

- **Consistent and cross-model left-wing bias**: All models and languages demonstrate a positive sentiment tendency toward left-wing/centrist politicians and a negative sentiment tendency toward the far-right.
- **Stronger bias in larger models**: Qwen-72B and Llama3-70B exhibit more pronounced political bias and higher cross-lingual consistency than their 7B/8B counterparts.
- **Political Compass analysis**: LLMs overall display a "left-liberal" bias, which aligns with previous ChatGPT political compass test results but is of a finer granularity.
- **Fictional name mitigation is effective**: Substituting with fictional names drastically reduces the IC and slightly boosts accuracy, confirming that the bias primarily originates from political associations.
- **Residual bias**: Even after removing political attributes, slight differences in sentiment ratings persist for female and non-Western names.

## Highlights & Insights

- **Methodological Innovation**: The idea of transforming TSC inconsistency into a bias signal is highly ingenious—detecting bias by leveraging the bias itself.
- **Unprecedented Scale**: Approximately 25 million data points ensure unprecedented statistical robustness, far exceeding existing questionnaire-based methods (which typically involve dozens to hundreds of interactions).
- **Multi-dimensional Findings**: Not only confirms the existence of bias, but also reveals propagation mechanisms across dimensions such as language, model scale, and multilingual training.
- **Counter-intuitive Finding**: Larger models not only exhibit stronger biases but also display more consistent biases across different languages—potentially because greater capacity better internalizes implicit patterns within training data.
- **Fictional Name Substitution** as a bias mitigation strategy is simple yet highly actionable.

## Limitations & Future Work

- The set of politicians is skewed toward figures frequently appearing in Western media, with underrepresentation of non-Western politicians.
- The sentences originate from European news corpora, which may fail to cover the nuances of global political discourse.
- Only the TSC task is tested; the performance of bias in other subjective tasks (such as stance detection or hate speech) remains unverified.
- The fictional name mitigation method loses some valuable contextual information.
- The temporal evolution of bias resulting from LLM version updates and changing political landscapes is not considered.

## Related Work & Insights

- Closest to Buyl et al. (2024) but in the opposite direction: the latter analyzes public figure descriptions generated by LLMs and uses LLMs to evaluate them (propagating bias); this work leverages the bias within the LLM's own TSC predictions.
- Distinction from Political Compass tests: This work provides a finer-grained analysis (at the level of individual politicians) with much greater statistical support.
- Insights: LLM biases can be "utilized"—the inconsistency itself is a signal that can be used for bias auditing and model comparison.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (Methodological innovation utilizing TSC inconsistency as a bias probe)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Extremely comprehensive analysis with 7 models $\times$ 6 languages $\times$ 1,319 entities $\times$ 450 sentences $\approx$ 25 million data points)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear methodology, rich visualizations, though the paper is quite long)
- **Value**: ⭐⭐⭐⭐ (Significant reference value for AI fairness and LLM robustness/trustworthiness research)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Variational Approach for Mitigating Entity Bias in Relation Extraction](a_variational_approach_for_mitigating_entity_bias_in_relation_extraction.md)
- [\[ACL 2025\] Sentiment Reasoning for Healthcare](sentiment_reasoning_for_healthcare.md)
- [\[ACL 2025\] Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)
- [\[ACL 2025\] Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](dot_absa_template.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)

</div>

<!-- RELATED:END -->
