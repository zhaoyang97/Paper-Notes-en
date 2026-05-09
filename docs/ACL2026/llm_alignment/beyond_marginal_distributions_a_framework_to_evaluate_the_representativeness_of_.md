---
title: >-
  [Paper Note] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs
description: >-
  [ACL 2026][LLM Alignment][demographic alignment] This paper proposes a representativeness evaluation framework for LLMs that goes beyond marginal distributions. By simultaneously examining marginal response distributions and cross-question correlation structures, the framework evaluates demographic-aligned models and finds that, while both fine-tuning and persona prompting improve approximation of marginal distributions, neither faithfully reproduces the multivariate correlation patterns observed in human values surveys.
tags:
  - ACL 2026
  - LLM Alignment
  - demographic alignment
  - correlation structure
  - marginal distributions
  - values survey
  - representativeness evaluation
date: 2026-05-08
content_hash: 44285593f253fb11
---

# Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs

**Conference**: ACL 2026
**arXiv**: [2601.15755](https://arxiv.org/abs/2601.15755)
**Code**: [https://github.com/tdw75/beyond-marginal-distributions](https://github.com/tdw75/beyond-marginal-distributions)
**Area**: LLM Alignment
**Keywords**: demographic alignment, correlation structure, marginal distributions, values survey, representativeness evaluation

## TL;DR

This paper proposes a representativeness evaluation framework for LLMs that goes beyond marginal distributions. By simultaneously examining marginal response distributions and cross-question correlation structures, the framework evaluates demographic-aligned models and finds that, while both fine-tuning and persona prompting improve approximation of marginal distributions, neither faithfully reproduces the multivariate correlation patterns observed in human values surveys.

## Background & Motivation

**State of the Field**: LLMs are increasingly used to simulate human opinions, values, and beliefs, and model steerability is an active research direction. Existing work employs persona prompting or demographic fine-tuning to align model outputs more closely with specific population groups.

**Limitations of Prior Work**: Existing evaluations focus primarily on **marginal response distributions**—comparing the response distribution of each question independently. While necessary, this approach may overlook the **deep latent structure** present in real populations. For instance, a model may correctly approximate support rates for two policies individually, yet fail to capture the strong correlation between supporting one policy and opposing the other that exists in real populations.

**Root Cause**: Cultural value theories in social science (e.g., Hofstede, Schwartz, Inglehart-Welzel) emphasize that multivariate correlation patterns among values are central to cultural dimensions, yet LLM alignment evaluation has almost entirely neglected this dimension.

**Paper Goals**: (1) Propose an evaluation framework that simultaneously examines marginal distributions and correlation structures; (2) Compare the performance of persona prompting and demographic fine-tuning along both dimensions.

**Starting Point**: The World Values Survey (WVS) is used as ground truth to diagnose model representativeness at the levels of both marginal distributions and inter-question correlation matrices.

**Core Idea**: Representativeness constitutes an independent dimension of alignment; evaluations that rely solely on marginal distributions may conceal structural failures, leading to overly optimistic conclusions about model representativeness.

## Method

### Overall Architecture

The framework comprises two complementary evaluation dimensions: (1) **Marginal Distribution Evaluation**—for each survey question, comparing the distance between the simulated response distribution and the true response distribution (using Wasserstein-1 distance or total variation distance); (2) **Correlation Structure Evaluation**—constructing question–question or topic–topic correlation matrices and comparing the correlation matrices of real and simulated data (using Pearson correlation and RMSE).

### Key Designs

1. **Marginal Distribution Evaluation**:

    - Function: Measures model representativeness at the level of individual questions.
    - Mechanism: For each survey question $q$, computes the distance $d(P_m(\cdot|q), P_s(\cdot|q))$ between the true distribution $P_s$ and the simulated distribution $P_m$, and takes the average over all questions as a dissimilarity metric $\mathcal{D}$. Response diversity is also assessed by comparing the normalized variance of each question.
    - Design Motivation: Maintains comparability with existing literature while providing a baseline for subsequent correlation structure analysis.

2. **Correlation Structure Evaluation**:

    - Function: Assesses whether the model preserves inter-question dependency relationships.
    - Mechanism: (a) Computes the mean response of each subgroup on each question to construct a mean matrix $A \in \mathbb{R}^{|S| \times |Q|}$; (b) Computes Pearson correlation coefficients between columns to obtain a correlation matrix $C \in \mathbb{R}^{|Q| \times |Q|}$; (c) Extracts the upper-triangular elements as a vector and compares the true matrix $C^{\text{true}}$ and simulated matrix $C^{\text{sim}}$ via Pearson correlation and RMSE.
    - Design Motivation: Correlation coefficients capture relative structure (which question pairs tend to co-vary), while RMSE captures magnitude matching; together they provide a comprehensive diagnostic.

3. **Experimental Design: Comparison of Three Model Configurations**:

    - Function: Compares different steering strategies within a unified framework.
    - Mechanism: (a) Unsteered baseline Phi-3; (b) Phi-3 with persona prompting (10 demographic subgroups); (c) OpinionGPT (demographically fine-tuned LoRA adapters trained on Reddit data). The WVS Wave 7 dataset with 193 questions and 10 subgroups is used; 500 samples are drawn per configuration.
    - Design Motivation: Compares the two mainstream steering approaches—parameter-level (fine-tuning) and prompt-level (persona)—to reveal their differences across evaluation dimensions.

### Loss & Training

OpinionGPT uses LoRA adapters fine-tuned on Reddit data from specific subgroups. This paper does not involve model training and focuses solely on evaluation.

## Key Experimental Results

### Main Results

**Question–Question Correlation Structure (95% Confidence Intervals)**

| Model | Pearson ρ | RMSE |
|-------|-----------|------|
| OpinionGPT | 0.090 [0.08, 0.10] | 0.638 [0.63, 0.64] |
| Persona Prompting | 0.158 [0.15, 0.17] | 0.679 [0.67, 0.68] |
| Permutation Null Baseline | −0.004 | 0.849 |
| Split-Half Upper Bound | 0.999 | 0.006 |

**Topic–Topic Correlation Structure**

| Model | Pearson ρ | RMSE |
|-------|-----------|------|
| OpinionGPT | −0.018 [-0.02, 0.05] | 0.718 [0.71, 0.73] |
| Persona Prompting | 0.240 [0.21, 0.28] | 0.676 [0.67, 0.69] |

### Ablation Study

**Marginal Distribution Results**: OpinionGPT reduces marginal dissimilarity across all subgroups, outperforming persona prompting. However, persona prompting performs worse on response diversity (tending to collapse toward stereotypical single responses), while OpinionGPT sometimes over-amplifies response diversity.

### Key Findings

- Improvement in marginal distributions does not imply improvement in correlation structure: OpinionGPT better approximates marginal distributions, while persona prompting slightly better preserves correlation structure—a reversal across evaluation dimensions.
- Both methods fall far below the empirical upper bound on correlation structure, indicating that current steering techniques cannot faithfully reproduce the multivariate structure of human values.
- Persona prompting significantly compresses response diversity and tends to produce stereotypical answers.
- OpinionGPT completely loses correlation structure after topic-level aggregation (ρ ≈ −0.018), suggesting that fine-tuning separate subgroup adapters may introduce cross-group representational drift.

## Highlights & Insights

- The evaluation framework is elegantly designed; the permutation null baseline and split-half upper bound provide clear reference points for the metrics.
- The finding that "marginal accuracy ≠ structural accuracy" carries important methodological implications as a cautionary signal.
- Incorporating cultural value theories from social science into LLM alignment evaluation reflects a distinctive interdisciplinary perspective.
- The paper highlights the importance of representativeness as an independent dimension of alignment.

## Limitations & Future Work

- Only Phi-3 is used as the base model, limiting the generalizability of the conclusions.
- WVS embeds Western-centric normative assumptions and is not a fully neutral benchmark.
- Evaluation is conducted in English only, without covering multilingual scenarios.
- Future work could replace independent sampling with trajectory-based sampling to construct more fine-grained correlation matrices.

## Related Work & Insights

- Complements marginal distribution evaluation work by Santurkar et al. and Durmus et al.
- Münker (2025) proposes a similar fingerprinting approach, but this paper explicitly positions correlation structure as a necessary condition for representativeness.
- Provides a theoretical foundation for future alignment mechanisms that incorporate multivariate dependency structures.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic incorporation of correlation structure into LLM representativeness evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation with confidence intervals and thorough baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Framework exposition is clear and problem definitions are rigorous.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](../../ICLR2026/llm_alignment/beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)
- [\[CVPR 2026\] Bias at the End of the Score: Demographic Biases in Reward Models for T2I](../../CVPR2026/llm_alignment/bias_reward_models_t2i.md)
- [\[ICLR 2026\] Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](../../ICLR2026/llm_alignment/beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)
- [\[AAAI 2026\] Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment](../../AAAI2026/llm_alignment/differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)
- [\[ICLR 2026\] CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](../../ICLR2026/llm_alignment/cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)

<!-- RELATED:END -->
