---
title: >-
  [Paper Note] Can Language Models Reason about Individualistic Human Values and Preferences?
description: >-
  [ACL2025][LLM (Other)][individualistic alignment] This work proposes the paradigm of "individualistic alignment" and constructs the IndieValueCatalog dataset (based on real-world data from 93k individuals in the World Values Survey, WVS). It evaluates and trains language models to reason about an individual's value judgments in novel scenarios based on their stated value expressions, revealing that frontier LMs achieve only 55%-65% accuracy and exhibit significant inequity ac…
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "individualistic alignment"
  - "value reasoning"
  - "World Values Survey"
  - "pluralistic alignment"
  - "Value Inequity Index"
date: 2026-05-08
content_hash: b6a7a7db491a7ad7
---

# Can Language Models Reason about Individualistic Human Values and Preferences?

**Conference**: ACL2025  
**arXiv**: [2410.03868](https://arxiv.org/abs/2410.03868)  
**Code**: [liweijiang/indievalue](https://github.com/liweijiang/indievalue)  
**Area**: LLM/NLP  
**Keywords**: individualistic alignment, value reasoning, World Values Survey, pluralistic alignment, Value Inequity Index

## TL;DR

This work proposes the paradigm of "individualistic alignment" and constructs the IndieValueCatalog dataset (based on real-world data from 93k individuals in the World Values Survey, WVS). It evaluates and trains language models to reason about an individual's value judgments in novel scenarios based on their stated value expressions, revealing that frontier LMs achieve only 55%-65% accuracy and exhibit significant inequity across demographic groups.

## Background & Motivation

Pluralistic alignment has recently gained attention, emphasizing that AI should serve the diverse needs of all people. However, existing methods have fundamental limitations:

- **Predefined Bucket Problem**: Existing pluralistic alignment frameworks rely on pre-selected dimensions of diversity (demographics, personality, culture, writing style, etc.), forcing individuals into coarse-grained categories and erasing intra-individual differences.
- **Risk of Stereotyping**: Demographic-based grouping methods may reinforce stereotypes, such as assuming that individuals of the same age, gender, or region hold identical values.
- **Limitations of Group-Level Inference**: Even finer-grained evaluations (e.g., Durmus et al. 2024) only perform group-level distributional inference and fail to capture individual-level variation.

Key Insight: The way to truly respect diversity is bottom-up (starting from the individual) rather than top-down (starting from predefined categories). The authors propose "individualistic alignment," pushing pluralistic alignment to its limit by modeling diversity at the individual level.

The key challenge lies in the data: acquiring sufficiently rich, multi-dimensional data that represents an individual's global value preferences is extremely difficult.

## Method

### Overall Architecture: IndieValueCatalog Dataset

The authors convert raw questionnaire data from the seventh wave of the World Values Survey (WVS) into a unified natural language statement format to build the IndieValueCatalog:

- **Unified Questions**: Convert 253 raw questionnaires of various formats (multiple-choice, Likert scales, etc.) into standardized value statement representations. For example, the original question "How secure do you feel these days?" (options: "very secure", "quite secure", etc.) is converted to the statement "I feel very secure these days".
- **Two Granularities**: Refined (929 statements) and Polar (567 statements).
- **Scale**: 93,279 real respondents from over 70 countries, with an average of 242 value statements and 31 demographic statements per person.
- **Immense Combinatorial Space**: The refined version has $1.65 \times 10^{139}$ combinations, and the polar version has $3.94 \times 10^{86}$ combinations, making accurate prediction of individual choices highly challenging.

### Key Designs 1: Evaluation Framework and Value Inequity Index

**Evaluation Setup**: Each individual's statements are split into a demonstration set (50-200 statements) and a probe set (39 statements covering 13 WVS question categories). Given the demonstration statements, the model must select the statement that best aligns with the individual's values from the candidates of the probe questions. Three-fold cross-validation is used (200 demonstrations + 39 probes) and averaged to avoid overfitting to a specific probe set. An evaluation set of 800 demographically balanced individuals is sampled from the IndieValueCatalog.

**Value Inequity Index ($\sigma_{\text{Inequity}}$)**: Measures the fairness of models when reasoning about individual values across different demographic groups. Specifically, along 13 demographic dimensions (income, region, age, etc.), it projects the standard deviation of model accuracy across sub-groups and averages them across all dimensions:

$$\sigma\text{Inequity}_M = \frac{1}{|\mathbb{D}|} \sum_{\mathcal{D}^k \in \mathbb{D}} \sigma\left(\{Acc_M^{\mathbb{I}_{\text{eval}}^{g_{k_t}}} \mid \forall g_{k_t} \in \mathcal{D}^k\}\right)$$

A lower $\sigma_{\text{Inequity}}$ represents a fairer model. This metric is complementary to accuracy: two models with the same accuracy can exhibit vastly different $\sigma_{\text{Inequity}}$.

### Key Designs 2: IndieValueReasoner Training

Finetuned on top of Llama-3.1-8B, with training data sampled from $\mathbb{I}_{\text{train}}$. Each training sample contains $d$ demonstration statements of a single individual and candidates for a probe question, and the model outputs the individual's choice.

Ablations of Key Training Configurations:
- **Probe Granularity**: Polar / Refined / Mixed
- **Demonstration Count**: Fixed 200 / Random 50-200 (mixed) / A combination of both
- **Training Scale**: Data from 100/200/400/800/1600 individuals per question
- **Regional Data**: Trained continent-by-continent to observe cross-regional generalization ability

### Key Designs 3: Comparison with Statistical Baselines

To contextualize LM performance, statistical baselines that do not use LMs are designed:
- **Global Majority Vote**: Chooses the statement selected by the majority of global respondents.
- **Similar Individual Matching (Resemble)**: Finds the most similar individual/group in the training set based on demonstration statements, and uses their choice for prediction.
- The statistical baselines have oracle access to the data of all 92K individuals, serving as a contrast to the LM trained on only 1.6K individuals.

## Key Experimental Results

### Table: Zero-shot Individual Value Reasoning Accuracy of Frontier LMs

| Model | Accuracy (%) | $\sigma_{\text{Inequity}}$ ↓ |
|---|---|---|
| Random | ~36.6 | — |
| GPT-4o (0806, No Demo) | 54.8 | 3.03 |
| GPT-4o (0806, 200 Demo) | 63.5 | 3.03 |
| GPT-4o (0513) | 63.7 | 2.87 |
| GPT-4o-mini | ~58 | 2.55 |
| GPT-4-turbo | ~62 | 2.83 |
| Llama-3.1-8B | ~55 | 2.97 |
| Llama-3.1-70B | 63.7 | **1.94** |
| Claude-3.5 Sonnet | ~60 | 3.14 |
| Qwen2-72B | ~61 | 3.24 |

All models achieve only 55%-65% accuracy, leaving substantial room for improvement. GPT-4o (0513) and Llama-3.1-70B share the same accuracy (63.7%), but the latter exhibits a significantly better fairness metric (1.94 vs. 2.87).

### Table: IndieValueReasoner Results (200 Demonstrations, Evaluation Set)

| Method | Polar Acc | Refined Acc | Overall Acc |
|---|---|---|---|
| Random | 45.37 | 27.87 | 36.62 |
| Global Majority Vote | 64.95 | 48.70 | 56.83 |
| Resemble (All, top cluster) | 73.73 | 59.47 | 66.60 |
| GPT-4o (No Demo) | 54.79 | 33.06 | 43.93 |
| GPT-4o (Demographics only) | 60.31 | 40.69 | 50.50 |
| GPT-4o (200 Demo) | 63.46 | 35.59 | 49.52 |
| Llama-3.1-8B (200 Demo) | 54.34 | 37.97 | 46.16 |
| **IndieValueReasoner (Best)** | **74.74** | **60.60** | **67.67** |

IndieValueReasoner improves upon zero-shot Llama-3.1-8B by **46.6%** and outperforms the best GPT-4o configuration by **34.0%**. Remarkably, using training data from only 1.6K individuals per question surpasses the statistical baseline that leverages 92K individuals (67.67 vs. 66.60).

### Table: Impact of Demographics vs. Value Statements Training

| Training Data | Demographics Evaluation | Statements Evaluation | Average |
|---|---|---|---|
| Demographics Only (400 per question) | 73.74 | 62.84 | 68.14 |
| Value Statements Only (800 per question) | 63.81 | 67.42 | 65.62 |
| Mixed (400 of each per question) | **73.74** | **68.02** | **70.88** |

Mixed training performs the best under both evaluation settings, illustrating that demographic and value statement information are complementary.

## Key Findings

1. **Demographics Cannot Substitute Individual Values**: GPT-4o achieves 60.31% accuracy when provided with demographics only, whereas 50 value statements alone achieve 60.59%. As the number of value statements increases, accuracy continues to scale up, while the marginal utility of demographics is negligible.
2. **Mixed Demonstration Counts Enhance Generalization**: Using a random number of demonstrations between 50 and 200 during training (rather than a fixed 200) unexpectedly improves test performance, as variation in demonstration quantity fosters stronger generalization.
3. **Impact of Regional Training Data Is Significant**: Models trained on specific continents generally perform best on test data from the corresponding continent; however, the global model consistently matches or outperforms regional models.
4. **Oceania Is an Exception**: Since its data is entirely from New Zealand, it has the lowest Shannon evenness. The homogeneity of training data limits the model's capacity to capture diverse value patterns.
5. **Training Significantly Improves Fairness**: The $\sigma_{\text{Inequity}}$ of IndieValueReasoner drops from 2.97 (zero-shot) to 2.22, with the largest improvement (+18.24%) observed in South America, which initially had the worst performance.

## Highlights & Insights

- **Paradigm Innovation**: Shifting pluralistic alignment from "group alignment" to "individualistic alignment" avoids the stereotyping risks associated with predefined categorizations, offering a theoretically profound research direction.
- **Elegant Dataset Design**: For the first time, individual-level sequential survey data from the WVS is repurposed for LM evaluation. Unifying and converting questions makes them directly applicable to NLP tasks, while the massive combinatorial space ensures the task remains challenging.
- **The $\sigma_{\text{Inequity}}$ Metric**: Provides a quantifiable fairness measure, revealing substantial equity disparities between models with similar raw accuracy, filling a gap in existing evaluation frameworks.
- **Counterintuitive Discovery**: A smaller model, after training, can significantly outperform a larger untrained model (trained Llama-3.1-8B at 67.67% vs. zero-shot GPT-4o at 63.5%), surpassing a statistical baseline leveraging 92K individuals with just 1.6K individuals of training data.
- **Implications for Personalized AI**: Value statements capture individual uniqueness far better than demographics. Future personalized systems should rely more heavily on descriptive preference data rather than demographic labels.

## Limitations & Future Work

- **Data Limitations**: The WVS is based on static surveys with abstract questions that lack the complexity of real-world interactions. The standardized conversion might oversimplify the nuances of value expression.
- **Single Prompt Setup**: Due to computational constraints, evaluation is conducted using only a single carefully designed prompt, without exploring the impacts of diverse prompt designs.
- **Limited Training Scale**: The training uses data from only 200-1600 individuals per question. It remains unclear whether scaling up training data would yield further gains.
- **Privacy Risks**: Individualistic alignment inherently requires deep personal value data, which could potentially trigger privacy violations or information misuse.
- **Bias Transmission**: While avoiding demographic stereotyping, models may still propagate other forms of bias, such as confirmation bias, anchoring effects, or framing effects.
- **Application Gap**: Even if individual values are accurately modeled, translating them into appropriate system behavior is a non-trivial challenge.

## Related Work & Insights

- **Pluralistic Alignment** (Sorensen et al. 2024; Feng et al. 2024): Emphasizes that AI should serve diverse populations, but mostly relies on predefined dimensions to group individuals $\rightarrow$ This work pushes it to the extreme individual level.
- **Personalized LMs** (Han et al. 2024; Zhu et al. 2024): Tailor user experiences for specific applications (e.g., summarization, chatting, movie tagging) $\rightarrow$ This work focuses on fundamental value system modeling rather than task-specific personalization.
- **Cultural and Value Evaluation** (Durmus et al. 2024; Rao et al. 2024): Evaluate cultural bias in LMs $\rightarrow$ However, they rely on group-level distributions instead of individual-level variation.
- **Active Preference Elicitation** (Keswani et al. 2024; Piriyakulkij et al. 2024): Elicit preferences interactively and efficiently $\rightarrow$ Can be combined with individualistic value data collection in the future.
- **Insights**: Individualistic value reasoning might be the key capability toward truly decentralized, personalized AI. It holds both social science value (understanding human value dynamics) and engineering value (such as mental health chatbots and personalized recommendation).

## Rating

- Novelty: ⭐⭐⭐⭐ — Shifting the paradigm from pluralistic to individualistic alignment is a clear contribution, with the $\sigma_{\text{Inequity}}$ metric holding independent value.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Zero-shot evaluation across 10 frontier LMs + systematic ablation training + cross-regional analysis, although lacking validation on downstream applications.
- Writing Quality: ⭐⭐⭐⭐ — Thoroughly motivated and mathematically rigorous, though the notation is somewhat heavy and could be made more concise.
- Value: ⭐⭐⭐⭐ — Highly inspiring for personalized AI and fairness evaluation, with a highly reusable dataset, though still some distance from practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference](can_llms_reason_about_program_semantics_a_comprehensive_evaluation_of_llms_on_fo.md)
- [\[ACL 2025\] Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs](can_llms_help_uncover_insights_about_llms_a_large-scale_evolving_literature_anal.md)
- [\[ACL 2025\] Aligning Large Language Models with Implicit Preferences from User-Generated Content](pugc_align_implicit_pref_ugc.md)
- [\[ACL 2025\] Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)
- [\[ACL 2025\] Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items](value_portrait_assessing_language_models_values_through_psychometrically_and_eco.md)

</div>

<!-- RELATED:END -->
