---
title: >-
  [Paper Note] Diverging Preferences: When do Annotators Disagree and do Models Know?
description: >-
  [ICML 2025][LLM Alignment][Preference Divergence] This paper systematically analyzes the reasons behind annotator disagreement in RLHF preference datasets by taxonomizing them into 10 categories. It reveals that over 75% of disagreements stem from personal preference rather than annotation noise. To address this, the paper proposes a Mean-Var Reward Model to effectively differentiate between diverging and high-consensus preferences, and uncovers systematic biases in LLM-as-Ju…
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "Preference Divergence"
  - "Reward Modeling"
  - "Pluralistic Alignment"
  - "LLM-as-Judge"
  - "Distributional Reward Models"
date: 2026-05-08
content_hash: a7e8e21bbfdff290
---

# Diverging Preferences: When do Annotators Disagree and do Models Know?

**Conference**: ICML 2025  
**arXiv**: [2410.14632](https://arxiv.org/abs/2410.14632)  
**Code**: None  
**Area**: LLM Alignment/RLHF  
**Keywords**: Preference Divergence, Reward Modeling, Pluralistic Alignment, LLM-as-Judge, Distributional Reward Models

## TL;DR
This paper systematically analyzes the reasons behind annotator disagreement in RLHF preference datasets by taxonomizing them into 10 categories. It reveals that over 75% of disagreements stem from personal preference rather than annotation noise. To address this, the paper proposes a Mean-Var Reward Model to effectively differentiate between diverging and high-consensus preferences, and uncovers systematic biases in LLM-as-Judge evaluation methodologies when facing disagreement.

## Background & Motivation
**Background**: RLHF has become the standard approach for aligning LLMs, yet annotator disagreement remains widespread—affecting 39% of samples in MultiPref and 24% in HelpSteer2.

**Limitations of Prior Work**: Existing reward modeling workflows (e.g., Bradley-Terry) treat annotator disagreement as simple noise, aggregating labels via majority voting. This practice ignores that disagreements often reflect legitimate differences in user perspectives.

**Key Challenge**: Standard reward models predict similar reward margins for both diverging and high-consensus preferences, failing to distinguish between them. Consequently, LLMs trained via RLHF only learn to cater to a single user perspective, violating the objective of pluralistic alignment.

**Goal**: (1) Understand the root causes of annotator disagreement; (2) design reward models capable of identifying diverging preferences; and (3) discover and mitigate biases in LLM-as-Judge evaluation.

**Key Insight**: Begin with data analysis by liberating individual annotation data in MultiPref and HelpSteer2 to establish a taxonomy of disagreement causes, leveraging these findings to drive improvements in reward modeling and evaluation.

**Core Idea**: Model rewards as distributions (rather than single scalars), thereby both predicting preference direction and capturing the degree of disagreement among annotators.

## Method

### Overall Architecture
The method comprises three components: (1) Disagreement cause analysis and taxonomy construction; (2) distributional reward model design and training; and (3) LLM-as-Judge bias analysis and mitigation.

### Key Designs

1. **Disagreement Taxonomy**:

    - **Function**: Categorizes preference divergences into 4 major categories and 10 subcategories.
    - **Mechanism**: Summarizes the sources of disagreement by manually analyzing 200 diverging samples.
    - Four major categories:
        - **Task Underspecification** (20-22%): Prompt is underspecified, leading to multiple reasonable interpretations.
        - **Response Style** (Verbosity 38-44%, Format 20-32%, Complexity 10%, Aesthetic Taste 14-22%): Differences in individual preferences.
        - **Refusal** (Comply vs. Refuse 5%, Refuse vs. Refuse 20%): Divergence in safety judgments.
        - **Errors & Hallucinations** (14-24%): Disagreement on factual errors.
    - **Design Motivation**: Over 75% of disagreements arise from legitimate stylistic differences or task underspecifications rather than annotator errors.
    - Annotation Agreement: Cohen's κ = 0.58-0.59, Krippendorff's α = 0.62-0.68.

2. **Mean-Var Distributional Reward Model (KL)**:

    - **Function**: Models the reward of each response as a normal distribution $r_A \sim \mathcal{N}(\mu_A, \sigma_A^2)$, simultaneously predicting both mean and variance.
    - **Mechanism**: The mean reflects the overall preference direction, while the variance captures the degree of annotator disagreement.
    - **Key Formulas**: $r_A - r_B \sim \mathcal{N}(\mu_A - \mu_B, \sigma_A^2 + \sigma_B^2 - 2\rho\sigma_A\sigma_B)$
    - Where $\rho$ models the correlation between the two responses (based on tie frequency).
    - **Training Loss**: Uses KL divergence loss to map the value of $r_A - r_B$ to the probability distribution of annotated preference labels.
    - Preference Mapping Intervals: Ties correspond to $(-0.5, 0.5)$, slight preference to $[0.5, 1.5)$, and significant preference to $[1.5, \infty)$.
    - Disagreement Detection: $|\mu_A - \mu_B| - \lambda(\sigma_A + \sigma_B)$; classified as diverging when variance is large and the mean difference is small.
    - Difference from Bradley-Terry: The latter only outputs scalar rewards and cannot capture uncertainty.

3. **LLM-as-Judge Bias Analysis and Mitigation**:

    - **Function**: Analyzes LLM-as-Judge behavior on diverging preferences and proposes a filtering methodology.
    - **Core Finding**: The proportion of LLM-as-Judge choosing a "winner" on diverging preference samples (73.8%) is almost identical to that on high-consensus preference samples (73.1%).
    - Bias Types: Systematic preferences for verbose and highly formatted outputs, and favoring compliance over refusal.
    - Mitigation Strategy: Uses the distributional reward model to identify diverging samples and remove them from evaluation benchmarks.

### Loss & Training
- **Mean-Var (KL)**: Trained using KL divergence loss, mapping $r_A - r_B$ into 5 preference categories (Strongly prefer A / Marginally prefer A / Tie / Marginally prefer B / Strongly prefer B).
- **Baseline Mean-Var (NLL, Independent)**: Uses negative log-likelihood loss under independence assumption, yielding sub-optimal performance.
- **Classification (KL)**: A 5-way classifier based on Likert-5 scores, predicting the distribution of scores.
- All models are trained on Llama-3-8B-Instruct.

## Key Experimental Results

### Distributional vs. Scalar Reward Models

| Reward Model | MultiPref Pref Acc | MultiPref Div AUROC | HS2 Pref Acc | HS2 Div AUROC |
|---------|-------------------|-------------------|-------------|--------------|
| Skywork (27B) | 0.651 | 0.494 | — | — |
| Nemotron (70B) | 0.638 | 0.400 | — | — |
| Bradley-Terry (Agg) | 0.663 | 0.458 | 0.683 | 0.482 |
| Bradley-Terry (All) | 0.648 | 0.438 | 0.678 | 0.489 |
| **Mean-Var (KL, ours)** | **0.664** | **0.615** | **0.684** | **0.582** |
| Classification (KL) | — | — | 0.659 | **0.648** |

### LLM-as-Judge Performance across Different Preference Types

| Preference Type | MultiPref Winner Chosen % | HelpSteer2 Winner Chosen % |
|---------|-------------------|-------------------|
| High-Consensus Preference | 73.1% | 64.6% |
| High-Consensus Tie | 42.6% | 51.9% |
| Diverging Preference (All) | 73.8% | 57.3% |
| Diverging Preference (Significant) | 76.0% | 65.0% |

### Key Findings
- The Diverging ID AUROC of standard Bradley-Terry reward models is close to random (~0.5), failing to distinguish diverging from consistent preferences.
- **Mean-Var (KL)** improves the Div AUROC from 0.46 to 0.62 (+0.16) while maintaining preference accuracy.
- LLM-as-Judge exhibits almost the same "decisiveness" on diverging preferences as on high-consensus ones, showing a systematic bias towards specific styles.
- Verbosity is the largest source of disagreement (38-44%), rather than annotation noise as previously assumed.

## Highlights & Insights
- **Deep Data Insights**: This work represents the first systematic analysis of disagreement causes in real-world preference datasets, establishing an empirically grounded taxonomy of divergence.
- It uncovers a widely overlooked issue: current reward models treat diverging and consistent samples identically, hindering the progress of pluralistic alignment.
- The distributional reward model offers an elegant solution by simultaneously performing preference prediction and divergence identification within a single model.
- The findings regarding LLM-as-Judge bias serve as an important cautionary tale for popular evaluation benchmarks like Arena-Hard.

## Limitations & Future Work
- Validation is limited to two English datasets; divergence patterns may vary across different languages and cultures.
- The distributional reward model is not yet directly integrated into RLHF training pipelines; incorporating divergence knowledge into policy optimization remains an open question.
- The granularity of the disagreement taxonomy could be further refined, as categories like Verbosity and Format can overlap.
- The mitigation strategy for LLM-as-Judge bias relies on the reward model, introducing a potential risk of circular dependency.

## Related Work & Insights
- Closely relates to and provides empirical data support for the line of research on Pluralistic Alignment.
- The concept of distributional reward modeling can be generalized to other scenarios featuring annotation uncertainty, such as safety evaluation.
- The disagreement taxonomy provides direct guidance for building and quality-controlling preference datasets.
- Offers a new analytical perspective on reward hacking: some instances of "alignment failure" might simply indicate that the model has learned the preferences of a specific subgroup of annotators.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?](../../CVPR2025/llm_alignment/do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)
- [\[AAAI 2026\] When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](../../AAAI2026/llm_alignment/when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)
- [\[NeurIPS 2025\] Capturing Individual Human Preferences with Reward Features](../../NeurIPS2025/llm_alignment/capturing_individual_human_preferences_with_reward_features.md)
- [\[NeurIPS 2025\] Ask a Strong LLM Judge when Your Reward Model is Uncertain](../../NeurIPS2025/llm_alignment/ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)
- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)

</div>

<!-- RELATED:END -->
