---
title: >-
  [Paper Note] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs
description: >-
  [ACL 2026][LLM Evaluation][Demographic Alignment] This paper proposes an evaluation framework for LLM representativeness that goes beyond marginal distributions. By simultaneously examining marginal response distribution…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Demographic Alignment"
  - "Correlation Structure"
  - "Marginal Distributions"
  - "Value Surveys"
  - "Representativeness Evaluation"
date: 2026-05-08
content_hash: 535c0020ffeb8725
---

# Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs

**Conference**: ACL 2026  
**arXiv**: [2601.15755](https://arxiv.org/abs/2601.15755)  
**Code**: [https://github.com/tdw75/beyond-marginal-distributions](https://github.com/tdw75/beyond-marginal-distributions)  
**Area**: LLM Alignment  
**Keywords**: Demographic Alignment, Correlation Structure, Marginal Distributions, Value Surveys, Representativeness Evaluation

## TL;DR

This paper proposes an evaluation framework for LLM representativeness that goes beyond marginal distributions. By simultaneously examining marginal response distributions and cross-question correlation structures to evaluate demographic-aligned models, it finds that while fine-tuning and persona prompting can improve the approximation of marginal distributions, neither can faithfully reproduce the multivariate correlation patterns found in human value surveys.

## Background & Motivation

**Background**: LLMs are increasingly being used to simulate human opinions, values, and beliefs, making model steerability an active research direction. Existing work employs persona prompting or demographic fine-tuning to bring model outputs closer to specific groups.

**Limitations of Prior Work**: Current evaluations primarily focus on **marginal response distributions**—comparing response distributions for each question independently. While necessary, this approach may overlook the **deep latent structures** present in real populations. For example, a model might correctly approximate support rates for two separate policies but fail to capture the high correlation between supporting one and opposing the other observed in actual human populations.

**Key Challenge**: Cultural value theories in social sciences (e.g., Hofstede, Schwartz, Inglehart-Welzel) emphasize that multivariate correlation patterns between values are the core of cultural dimensions. However, LLM alignment evaluation has almost entirely ignored this dimension.

**Goal**: (1) Propose an evaluation framework that examines both marginal distributions and correlation structures; (2) Compare the performance of persona prompting and demographic fine-tuning across these two dimensions.

**Key Insight**: Using the World Values Survey (WVS) as ground truth, the authors diagnose model representativeness at both the marginal distribution level and the inter-question correlation matrix level.

**Core Idea**: Representativeness is an independent dimension of alignment. Evaluations relying solely on marginal distributions may mask structural failures, leading to overly optimistic conclusions regarding model representativeness.

## Method

### Overall Architecture

The framework consists of two complementary evaluation dimensions: (1) **Marginal Distribution Evaluation**—comparing the distance between simulated and real response distributions for each survey question (using Wasserstein-1 distance or Total Variation distance); (2) **Correlation Structure Evaluation**—constructing question-question or theme-theme correlation matrices to compare real and simulated data (using Pearson correlation and RMSE).

### Key Designs

1.  **Marginal Distribution Evaluation**:
    - **Function**: Measures model representativeness at the individual question level.
    - **Mechanism**: For each survey question $q$, the distance $d(P_m(\cdot|q), P_s(\cdot|q))$ between the real distribution $P_s$ and the simulated distribution $P_m$ is calculated. The average across all questions serves as the dissimilarity metric $\mathcal{D}$. Response diversity is also assessed by comparing the normalized variance of each question.
    - **Design Motivation**: Ensures comparability with existing literature while establishing a baseline for subsequent correlation structure analysis.

2.  **Correlation Structure Evaluation**:
    - **Function**: Evaluates whether the model preserves dependencies between questions.
    - **Mechanism**: (a) Calculate the mean response for each sub-population on each question to construct a mean matrix $A \in \mathbb{R}^{|S| \times |Q|}$; (b) Compute Pearson correlation coefficients between columns to derive the correlation matrix $C \in \mathbb{R}^{|Q| \times |Q|}$; (c) Extract the upper triangular elements and compare the Pearson correlation and RMSE between the true matrix $C^{\text{true}}$ and the simulated matrix $C^{\text{sim}}$.
    - **Design Motivation**: Correlation coefficients capture relative structures (which questions tend to co-vary), while RMSE captures magnitude matching. Together, they provide a comprehensive diagnosis.

3.  **Experimental Design: Comparison of Three Model Configurations**:
    - **Function**: Compares different steering strategies within a unified framework.
    - **Mechanism**: (a) Unsteered baseline Phi-3; (b) Phi-3 + persona prompting (10 demographic sub-populations); (c) OpinionGPT (demographic fine-tuning using LoRA adapters based on Reddit data). Using 193 questions from WVS Wave 7 across 10 sub-populations, with 500 samples per configuration.
    - **Design Motivation**: Compares parameter-level (fine-tuning) and prompt-level (persona) steering methods to reveal their differences across distinct evaluation dimensions.

### Loss & Training

OpinionGPT uses LoRA adapters fine-tuned on data from specific Reddit sub-populations. This paper focuses solely on evaluation and does not involve new model training.

## Key Experimental Results

### Main Results

**Question-Question Correlation Structure (95% Confidence Interval)**

| Model | Pearson $\rho$ | RMSE |
|-------|-----------|------|
| OpinionGPT | 0.090 [0.08, 0.10] | 0.638 [0.63, 0.64] |
| Persona Prompting | 0.158 [0.15, 0.17] | 0.679 [0.67, 0.68] |
| Permutation Baseline | −0.004 | 0.849 |
| Split-Half Upper Bound | 0.999 | 0.006 |

**Theme-Theme Correlation Structure**

| Model | Pearson $\rho$ | RMSE |
|-------|-----------|------|
| OpinionGPT | −0.018 [-0.02, 0.05] | 0.718 [0.71, 0.73] |
| Persona Prompting | 0.240 [0.21, 0.28] | 0.676 [0.67, 0.69] |

### Ablation Study

**Marginal Distribution Results**: OpinionGPT reduced marginal dissimilarity across all sub-populations, outperforming persona prompting. However, persona prompting exhibited poorer response diversity (tending to collapse into stereotypical single responses), while OpinionGPT occasionally over-amplified response diversity.

### Key Findings

- Improvement in marginal distribution $\neq$ improvement in correlation structure: OpinionGPT better approximates marginal distributions, yet persona prompting slightly better preserves correlation structures—indicating an "inversion" of evaluation dimensions.
- Both methods perform significantly below the empirical upper bound for correlation structure, suggesting that current steering techniques cannot faithfully reproduce the multivariate structure of human values.
- Persona prompting significantly compresses response diversity, tending to yield stereotypical answers.
- OpinionGPT completely loses correlation structure after theme-level aggregation ($\rho \approx -0.018$), suggesting that fine-tuning individual sub-population adapters may introduce representation drift across groups.

## Highlights & Insights

- The evaluation framework is ingeniously designed, with the permutation baseline and split-half upper bound providing clear references for the metrics.
- The finding that "marginal accuracy does not imply structural accuracy" serves as a critical methodological warning.
- Incorporating cultural value theories from social sciences into LLM alignment evaluation provides a unique interdisciplinary perspective.
- Demonstrates the importance of representativeness as an independent dimension of alignment.

## Limitations & Future Work

- The study only utilized the Phi-3 base model; the generalizability of the findings remains limited.
- WVS embeds Western-centric normative assumptions and is not a completely neutral benchmark.
- Evaluation was restricted to English, failing to cover multilingual contexts.
- Future research could replace independent sampling with trajectory-based sampling to construct more refined correlation matrices.

## Related Work & Insights

- Complementary to the marginal distribution evaluation work of Santurkar et al. and Durmus et al.
- Münker (2025) proposed a similar fingerprinting method, but this work explicitly defines correlation structure as a necessary condition for representativeness.
- Provides a theoretical basis for incorporating multivariate dependency structures into future alignment mechanisms.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically incorporate correlation structure into LLM representativeness evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-dimensional evaluation, confidence intervals, and robust baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear framework exposition and rigorous problem definition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions](large_language_models_are_bad_dice_players_llms_struggle_to_generate_random_numb.md)
- [\[ACL 2026\] Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)
- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)
- [\[ACL 2026\] LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics](llms_as_annotators_of_credibility_assessment_in_danish_asylum_decisions_evaluati.md)
- [\[ICLR 2026\] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](../../ICLR2026/llm_evaluation/unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)

</div>

<!-- RELATED:END -->
