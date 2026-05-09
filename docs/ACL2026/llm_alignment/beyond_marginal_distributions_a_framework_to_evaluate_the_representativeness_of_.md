---
title: >-
  [Paper Note] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs
description: >-
  [ACL 2026][LLMAlignment][DemographicAlignment] This paper proposes an LLM representativeness evaluation framework that goes beyond marginal distributions, assessing demographic alignment models by jointly examining marginal response distributions and cross-question correlation structures. It finds that while fine-tuning and persona prompting improve marginal distribution approximation, neither faithfully reproduces the multivariate correlation patterns observed in human values surveys.
tags:
  - ACL 2026
  - LLM Alignment
  - Demographic Alignment
  - Correlation Structure
  - Marginal Distribution
  - Values Survey
  - Representativeness Evaluation
date: 2026-05-08
content_hash: 2bebcac0deb0affa
---
# Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs

**Conference**: ACL 2026
**arXiv**: [2601.15755](https://arxiv.org/abs/2601.15755)
**Code**: [https://github.com/tdw75/beyond-marginal-distributions](https://github.com/tdw75/beyond-marginal-distributions)
**Area**: LLM Alignment
**Keywords**: demographic alignment, correlation structure, marginal distributions, values surveys, representativeness evaluation

## TL;DR

This paper proposes a representativeness evaluation framework for LLMs that goes beyond marginal distributions by jointly examining marginal response distributions and cross-question correlation structures to assess demographic-aligned models. The findings reveal that while fine-tuning and persona prompting improve the approximation of marginal distributions, neither faithfully reproduces the multivariate correlation patterns observed in human values surveys.

## Background & Motivation

**State of the Field**: LLMs are increasingly used to simulate human opinions, values, and beliefs, and model steerability has become an active research direction. Existing work employs persona prompting or demographic fine-tuning to align model outputs more closely with specific population groups.

**Limitations of Prior Work**: Existing evaluations primarily focus on **marginal response distributions**—independently comparing the response distribution for each question. While necessary, this approach may overlook the **deep latent structure** present in real populations. For instance, a model may correctly approximate support rates for two policies individually, yet fail to capture the high correlation in the real population between supporting one policy and opposing the other.

**Root Cause**: Cultural value theories in social science (e.g., Hofstede, Schwartz, Inglehart-Welzel) emphasize that multivariate correlation patterns among values are central to cultural dimensions, yet LLM alignment evaluation has almost entirely neglected this dimension.

**Paper Goals**: (1) Propose an evaluation framework that jointly examines marginal distributions and correlation structures; (2) compare persona prompting and demographic fine-tuning across both dimensions.

**Starting Point**: The World Values Survey (WVS) is used as ground truth to diagnose model representativeness at the levels of both marginal distributions and inter-question correlation matrices.

**Core Idea**: Representativeness constitutes an independent dimension of alignment; evaluations relying solely on marginal distributions may conceal structural failures, leading to overly optimistic conclusions about model representativeness.

## Method

### Overall Architecture

The framework comprises two complementary evaluation dimensions: (1) **Marginal Distribution Evaluation**—for each survey question, the distance between simulated and real response distributions is compared (using Wasserstein-1 distance or total variation distance); (2) **Correlation Structure Evaluation**—question-question or topic-topic correlation matrices are constructed, and the correlation matrices of real and simulated data are compared (using Pearson correlation and RMSE).

### Key Designs

1. **Marginal Distribution Evaluation**:

    - **Function**: Measures model representativeness at the level of individual questions.
    - **Mechanism**: For each survey question $q$, the distance $d(P_m(\cdot|q), P_s(\cdot|q))$ between the real distribution $P_s$ and the simulated distribution $P_m$ is computed; the average across all questions serves as the dissimilarity metric $\mathcal{D}$. Response diversity is also assessed by comparing normalized variance per question.
    - **Design Motivation**: Maintains comparability with existing literature while providing a baseline for subsequent correlation structure analysis.

2. **Correlation Structure Evaluation**:

    - **Function**: Assesses whether the model preserves inter-question dependency relationships.
    - **Mechanism**: (a) The mean response of each subgroup on each question is computed to form a mean matrix $A \in \mathbb{R}^{|S| \times |Q|}$; (b) Pearson correlation coefficients between columns yield a correlation matrix $C \in \mathbb{R}^{|Q| \times |Q|}$; (c) Upper-triangular elements are extracted and the Pearson correlation and RMSE between the real matrix $C^{\text{true}}$ and simulated matrix $C^{\text{sim}}$ are computed.
    - **Design Motivation**: Pearson correlation captures relative structure (which question pairs tend to co-vary), while RMSE captures magnitude matching; together they provide a comprehensive diagnostic.

3. **Experimental Design: Comparison of Three Model Configurations**:

    - **Function**: Compares different steering strategies within a unified framework.
    - **Mechanism**: (a) Unsteered baseline Phi-3; (b) Phi-3 with persona prompting (10 demographic subgroups); (c) OpinionGPT (demographic fine-tuned LoRA adapters trained on Reddit data). Evaluation uses 193 questions from WVS Wave 7, 10 subgroups, and 500 samples per configuration.
    - **Design Motivation**: Compares the two dominant steering paradigms—parameter-level (fine-tuning) and prompt-level (persona)—to reveal their differences across evaluation dimensions.

### Loss & Training

OpinionGPT employs LoRA adapters fine-tuned on subgroup-specific Reddit data. This paper does not involve model training; it is exclusively an evaluation study.

## Key Experimental Results

### Main Results

**Question-Question Correlation Structure (95% Confidence Intervals)**

| Model | Pearson ρ | RMSE |
|-------|-----------|------|
| OpinionGPT | 0.090 [0.08, 0.10] | 0.638 [0.63, 0.64] |
| Persona Prompting | 0.158 [0.15, 0.17] | 0.679 [0.67, 0.68] |
| Permutation Null Baseline | −0.004 | 0.849 |
| Split-Half Upper Bound | 0.999 | 0.006 |

**Topic-Topic Correlation Structure**

| Model | Pearson ρ | RMSE |
|-------|-----------|------|
| OpinionGPT | −0.018 [-0.02, 0.05] | 0.718 [0.71, 0.73] |
| Persona Prompting | 0.240 [0.21, 0.28] | 0.676 [0.67, 0.69] |

### Ablation Study

**Marginal Distribution Results**: OpinionGPT reduces marginal dissimilarity across all subgroups, outperforming persona prompting. However, persona prompting performs worse on response diversity (tending to collapse toward stereotypical, uniform responses), while OpinionGPT sometimes over-amplifies response diversity.

### Key Findings

- Improvement in marginal distributions does not imply improvement in correlation structure: OpinionGPT better approximates marginal distributions, while persona prompting slightly better preserves correlation structure—a reversal across evaluation dimensions.
- Both methods fall far below the empirical upper bound on correlation structure, indicating that current steering techniques cannot faithfully reproduce the multivariate structure of human values.
- Persona prompting substantially compresses response diversity, tending to produce stereotypical answers.
- OpinionGPT entirely loses correlation structure after topic-level aggregation (ρ ≈ −0.018), suggesting that fine-tuning separate adapters per subgroup may introduce cross-group representational drift.

## Highlights & Insights

- The evaluation framework is elegantly designed; the permutation null baseline and split-half upper bound provide clear reference points for the metrics.
- The finding that "marginal correctness ≠ structural correctness" carries important methodological implications.
- Introducing cultural value theory from social science into LLM alignment evaluation offers a distinctive interdisciplinary perspective.
- The paper highlights the importance of representativeness as an independent dimension of alignment.

## Limitations & Future Work

- Only a single base model (Phi-3) is used, limiting the generalizability of the conclusions.
- WVS embeds Western-centric normative assumptions and is not a fully neutral benchmark.
- Evaluation is conducted in English only, with no coverage of multilingual settings.
- Future work could adopt trajectory-based sampling in place of independent sampling to construct more fine-grained correlation matrices.

## Related Work & Insights

- Complements marginal distribution evaluation work by Santurkar et al. and Durmus et al.
- Münker (2025) proposes a similar fingerprinting approach, but this paper explicitly positions correlation structure as a necessary condition for representativeness.
- Provides a theoretical foundation for future work integrating multivariate dependency structures into alignment mechanisms.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic incorporation of correlation structure into LLM representativeness evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dimensional evaluation with confidence intervals and comprehensive baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Framework is clearly articulated and problem definitions are rigorous.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2025\] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](../../ACL2025/llm_alignment/beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](../../ICLR2026/llm_alignment/beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)
- [\[CVPR 2026\] Bias at the End of the Score: Demographic Biases in Reward Models for T2I](../../CVPR2026/llm_alignment/bias_reward_models_t2i.md)
- [\[ICLR 2026\] Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](../../ICLR2026/llm_alignment/beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)
- [\[AAAI 2026\] Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment](../../AAAI2026/llm_alignment/differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)

<!-- RELATED:END -->
