---
title: >-
  [Paper Note] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language
description: >-
  [ACL 2025 (GeBNLP Workshop)][Social Computing][Gender Bias] This paper constructs five gender bias evaluation datasets specifically for German and systematically evaluates them across eight multilingual LLMs, revealing unique gender bias challenges in German—including the ambiguous interpretation of masculine occupational nouns and the influence of seemingly neutral nouns on gender perception.
tags:
  - "ACL 2025 (GeBNLP Workshop)"
  - "Social Computing"
  - "Gender Bias"
  - "German"
  - "Large Language Models"
  - "Bias Evaluation"
  - "Multilingual"
date: 2026-05-08
content_hash: 180343fddf57aef3
---

# Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language

**Conference**: ACL 2025 (GeBNLP Workshop)  
**arXiv**: [2507.16557](https://arxiv.org/abs/2507.16557)  
**Code**: [https://github.com/rwth-i6/Gender-Bias-in-German-LLMs](https://github.com/rwth-i6/Gender-Bias-in-German-LLMs)  
**Area**: Social Computing  
**Keywords**: Gender Bias, German, Large Language Models, Bias Evaluation, Multilingual

## TL;DR

This paper constructs five gender bias evaluation datasets specifically for German and systematically evaluates them across eight multilingual LLMs, revealing unique gender bias challenges in German—including the ambiguous interpretation of masculine occupational nouns and the influence of seemingly neutral nouns on gender perception.

## Background & Motivation

**Background**: Gender bias evaluation in LLMs has received significant attention in recent years, with various methods proposed to quantify and analyze bias, including the Word Embedding Association Test (WEAT), template-filling tests, and generation-based evaluation methods.

**Limitations of Prior Work**: The vast majority of existing bias evaluation methods are designed for English. Directly transferring these methods to other languages poses severe transferability challenges: grammatical structures (such as gender, number, and case systems), socio-cultural backgrounds, and occupational naming conventions vary significantly across languages, meaning English bias patterns cannot be simply mapped to other languages.

**Key Challenge**: As a language with a complex grammatical gender system (masculine, feminine, and neuter), German exhibits gender bias patterns that are fundamentally different from English. For instance, masculine occupational nouns in German (e.g., "Arzt") can refer specifically to male doctors or be used as a generic masculine representation. This ambiguity does not exist in English, meaning direct translations of English bias tests often yield misleading results.

**Goal**: To build specialized datasets and evaluation frameworks for evaluating bias in German LLMs, systematically revealing the unique challenges of gender bias in the German language environment.

**Key Insight**: Starting from German grammatical and sociolinguistic characteristics, the authors design evaluation schemes tailored to German features based on existing gender bias concepts (such as occupational stereotypes, pronoun resolution, and gender association).

**Core Idea**: Constructing five German datasets based on different bias concepts, combined with multiple evaluation methodologies, to conduct systematic testing on eight multilingual LLMs and uncover unique German bias patterns.

## Method

### Overall Architecture

The methodological framework of this paper is an evaluation system rather than a model. The overall workflow consists of: (1) designing five evaluation dimensions based on gender bias theories; (2) constructing German datasets for each dimension while fully considering German grammatical characteristics; (3) defining corresponding evaluation metrics and methodologies for each dataset; and (4) conducting systematic evaluations on eight multilingual LLMs and analyzing the results.

### Key Designs

1. **Occupational Stereotype Dataset**:

    - **Function**: Assessing whether LLMs exhibit stereotypes regarding occupational-gender associations.
    - **Mechanism**: Selecting a series of German occupational nouns, providing both masculine forms (e.g., "Arzt") and feminine forms (e.g., "Ärztin"), and designing prompt templates to instruct LLMs to generate gender-related attributes for occupational roles or complete sentences. The degree of stereotyping is quantified by analyzing the gender distribution in the LLM outputs.
    - **Design Motivation**: Addressing the generic masculine issue in German. When "Arzt" is used, LLMs may interpret it as a "male doctor" rather than a generic term, which itself acts as a signal of bias.

2. **Pronoun Resolution Dataset**:

    - **Function**: Testing whether LLMs are influenced by gender bias during pronoun resolution.
    - **Mechanism**: Constructing German sentences containing two characters of different genders and an ambiguous pronoun, and tasking the LLM with determining which character the pronoun refers to. German pronoun systems are more complex than English (e.g., "sie" can refer to both "she" and "they"); thus, sentence templates are specifically designed to avoid leaking grammatical cues.
    - **Design Motivation**: The complexity of the German pronoun system requires finer-grained design for pronoun resolution bias evaluation compared to English, as simple translation of Winogender/WinoBias would introduce substantial grammatical interference.

3. **Neutral Noun Gender Perception Dataset**:

    - **Function**: Detecting whether seemingly neutral nouns influence gender perception in LLMs.
    - **Mechanism**: Many inanimate nouns in German carry grammatical genders (e.g., "die Sonne/sun" is feminine, "der Mond/moon" is masculine). The goal is to test whether these grammatical genders leak into the LLM's gender judgments of associated human roles. Test templates are designed to combine neutral concepts with gender judgment tasks.
    - **Design Motivation**: This is a unique source of bias for German (and other grammatically gendered languages) that is entirely absent in English bias evaluation, representing a typical failure case of cross-lingual transfer.

### Loss & Training

This paper does not involve model training and focuses purely on evaluation. The evaluation metrics include gender distribution bias (the degree of deviation from a $50$-$50$ uniform distribution), stereotype consistency rate (the proportion of LLM outputs aligning with societal stereotypes), etc.

## Key Experimental Results

### Main Results

The performance of eight multilingual LLMs on five German bias datasets:

| Model | Occupational Stereotypes | Pronoun Resolution Bias | Neutral Noun Influence | Overall Bias Level |
|------|------------|------------|------------|------------|
| GPT-4 Series | Moderate Bias | Lower Bias | Influence Detected | Relatively Good |
| Llama Series | Stronger Bias | Moderate Bias | Significant Influence | Pronounced Bias |
| Multilingual Mid-sized Models | Stronger Bias | Higher Bias | Significant Influence | Most Pronounced Bias |
| Overall Trend | Large Variance Across Models | German-Specific Patterns | Grammatical Gender Leakage | Needs Targeted Evaluation |

### Ablation Study

| Evaluation Condition | Bias Change | Explanation |
|---------|---------|------|
| Generic Masculine vs. Feminine | Significant Difference | Masculine form triggers stronger stereotypes |
| With vs. Without Grammatical Gender Cues | Larger bias with cues | Grammatical gender indeed leaks into semantic judgment |
| German vs. English Under Same Conditions | Different patterns | Confirms English evaluations cannot be directly transferred |
| Different Prompt Templates | Fluctuations in results | Prompt sensitivity warrants attention |

### Key Findings

- **Ambiguity of the generic masculine** is the most central challenge in German bias evaluation: LLMs tend to interpret masculine occupational nouns specifically as male rather than generic, which inflates the quantified values of occupational-gender stereotypes.
- **Leakage from grammatical gender to semantic gender** is confirmed in multiple models: The grammatical gender (masculine/feminine/neuter) of nouns does indeed influence the LLM's gender judgment of associated human roles.
- Bias patterns vary greatly across models, with no single model performing best on all dimensions, indicating that bias is multidimensional.
- The evaluation results are relatively sensitive to the specific wording of prompt templates, highlighting the importance of standardizing evaluation methodologies.

## Highlights & Insights

- **The impact of grammatical gender on semantic bias** is a highly intriguing finding—this means that for languages with grammatical gender (French, German, Spanish, etc.), the sources of bias in LLMs are more complex than in English, necessitating specialized evaluation tools.
- The rigorous design of **five datasets covering different bias dimensions** is commendable. Each dataset targets a specific bias concept, avoiding the risk of a single metric masking multifaceted problems.
- The open-sourcing of code and datasets on GitHub provides a reusable infrastructure for subsequent bias research on German and other grammatically gendered languages.

## Limitations & Future Work

- As a workshop paper, the scope is limited, resulting in a lack of depth in some experimental analyses.
- It only covers German as a grammatically gendered language; whether the conclusions can generalize to French, Spanish, etc., remains to be verified.
- The dataset sizes are relatively small, which may introduce biases from specific templates or wording.
- The evaluation metrics focus primarily on binary gender (male/female), leaving non-binary gender identities unaddressed.
- Building on this, targeted bias mitigation (debiasing) strategies can be explored in future work.

## Related Work & Insights

- **vs. WinoBias/Winogender (English)**: Directly translating these classic bias evaluations to German fails due to grammatical gender differences. The datasets in this paper serve as specifically designed alternatives for German.
- **vs. BBQ/BOLD Benchmarks**: These large-scale bias benchmarks are mostly oriented towards English; this study provides a methodological exemplar for multilingual bias evaluation.
- The observed phenomenon of "grammatical gender leakage" has important implications for the alignment training of multilingual models.

## Rating

- Novelty: ⭐⭐⭐⭐ Extending bias evaluation to German has some novelty, though the methodology is primarily an adaptation of existing concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐ The coverage of eight models × five datasets is decent, but the depth of analysis is constrained by the workshop paper length.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem motivation is clear, and the explanation of German grammatical gender features is accessible to non-German speakers.
- Value: ⭐⭐⭐⭐ The primary contributions are the datasets and evaluation, which hold direct value for the German NLP community, though the broader impact is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GG-BBQ: German Gender Bias Benchmark for Question Answering](gg-bbq_german_gender_bias_benchmark_for_question_answering.md)
- [\[ACL 2025\] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades](taz2024full_analysing_german_newspapers_for_gender_bias_and_discrimination_acros.md)
- [\[ACL 2025\] Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations](translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](../../ACL2026/social_computing/spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)

</div>

<!-- RELATED:END -->
