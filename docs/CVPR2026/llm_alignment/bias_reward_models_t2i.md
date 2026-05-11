---
title: >-
  [Paper Note] Bias at the End of the Score: Demographic Biases in Reward Models for T2I
description: >-
  [CVPR 2026][LLM Alignment][Reward Model] This paper conducts a large-scale demographic bias audit of widely used reward models (PickScore, ImageReward, HPS, etc.) in text-to-image generation…
tags:
  - "CVPR 2026"
  - "LLM Alignment"
  - "Reward Model"
  - "Text-to-Image"
  - "Demographic Bias"
  - "Hypersexualization"
  - "Fairness"
date: 2026-05-08
content_hash: d345d1622a74f79b
---

# Bias at the End of the Score: Demographic Biases in Reward Models for T2I

**Conference**: CVPR 2026
**arXiv**: [2604.13305](https://arxiv.org/abs/2604.13305)
**Code**: N/A
**Area**: Alignment & RLHF
**Keywords**: Reward Model, Text-to-Image, Demographic Bias, Hypersexualization, Fairness

## TL;DR

This paper conducts a large-scale demographic bias audit of widely used reward models (PickScore, ImageReward, HPS, etc.) in text-to-image generation, revealing that reward-guided optimization disproportionately sexualizes female images, converges demographics toward white, and that reward scores correlate with real-world population frequency priors.

## Background & Motivation

**Current landscape**: Reward models (RMs) are ubiquitous in T2I pipelines—used for dataset filtering, evaluation metrics, optimization supervision signals, and post-generation filtering. PickScore, ImageReward, HPS, and others are trained on human preference data.

**Existing limitations**: RMs are designed and deployed as "quality metrics," yet their robustness and fairness regarding demographic biases remain virtually unstudied. Training data, human preferences, and model inductive biases can all inject biases into RMs.

**Core tension**: RMs are widely used as "quality" proxies, yet they may implicitly encode demographic biases that are exponentially amplified through various stages of the T2I pipeline.

**Objective**: Systematically audit the demographic bias behavior of RMs in fine-tuning and evaluation.

**Approach**: Use the ReNO framework for reward-guided optimization to observe demographic changes in images before and after optimization; employ counterfactual datasets for score-level bias analysis.

**Core idea**: RMs do not merely evaluate image quality—they implicitly reward images that conform to the dominant demographic characteristics in their training data.

## Method

### Overall Architecture

The analysis comprises two parts: Part I (optimization experiments) uses ReNO's reward-guided noise optimization to observe how RMs alter the demographic attributes and sexualization degree of generated images. Part II (scoring experiments) uses three counterfactual datasets (CausalFace, SocialCounterfactuals, PAIRS) to examine systematic demographic biases in RM scores via linear regression and ranking analysis.

### Key Designs

1. **Reward-Guided Optimization Experiments (Part I)**:

    - Purpose: Reveal the systematic effects of RM gradients on image demographic attributes
    - Core idea: Use the ReNO framework $\varepsilon^* = \arg\max_\varepsilon R(G_\theta(\varepsilon, p), p)$ to optimize initial noise vectors. Measure changes in NSFW classification rates, skin exposure, and demographic classifier outputs before and after optimization. Two prompt sets are used: with and without demographic identifiers
    - Design rationale: If RMs are neutral quality metrics, optimization should not systematically alter demographic attributes or increase sexualized content

2. **Counterfactual Scoring Analysis (Part II)**:

    - Purpose: Directly test whether RM scores systematically differ based on demographic attributes
    - Core idea: Use matched image sets that differ only in demographic attributes (race, gender). OLS regression: $s^R_{I,p} \approx \beta_0 + \beta_1 \rho_I + \beta_2 \gamma_I + \beta_3(\rho_I \times \gamma_I) + \epsilon_I$. Statistically significant coefficients indicate systematic RM preferences for specific demographic attributes. Ranking analysis supplements relative preference ordering
    - Design rationale: Counterfactual design controls all variables except demographic attributes, isolating pure demographic bias

3. **Real-World Frequency Correlation Analysis**:

    - Purpose: Reveal that RMs encode real-world population distribution priors
    - Core idea: Correlate RM scores for occupation prompts with female employment proportions reported by the U.S. Bureau of Labor Statistics across occupations
    - Design rationale: If RMs only evaluate quality, scores should not correlate with real-world population frequencies

### Loss Function / Training Strategy

This is an audit/analysis paper involving no new model training. Default ReNO hyperparameters are used for optimization experiments. Scores are normalized to zero mean and unit variance to ensure cross-model comparability.

## Key Experimental Results

### Main Experiments

| Finding | RM | Effect Size |
|---|---|---|
| Hypersexualization amplification | PickScore | Female NSFW rate increases 19% vs male 7% (2.7×) |
| Demographic convergence | ImageReward/HPS | >80% of Black images classified as white after optimization |
| Gender flipping | ImageReward | 39% of female images classified as male after optimization |
| Racial score bias | HPS/ImageReward | White images systematically receive highest scores |
| VQAScore reversal | VQAScore | Positive prompts prefer white; negative prompts prefer Black |

### Ablation Studies

| RM | White Rank | Black Rank | Gap |
|---|---|---|---|
| HPS | 1.2 | 3.1 | Largest bias |
| ImageReward | 1.4 | 2.8 | Significant bias |
| CLIP | 2.5 | 3.5 | Black consistently lowest |
| PickScore | 1.8 | 2.3 | Moderate bias |

### Key Findings

- PickScore exhibits the strongest hypersexualization effect: female impact is 2.7× that of male
- ImageReward and HPS cause the most severe demographic convergence: over 80% of Black images are classified as white after optimization
- RM scores significantly correlate with U.S. occupational gender ratios, indicating that RMs have learned real-world frequency priors
- VQAScore displays a "stereotype reinforcement" pattern: positive descriptions prefer white, negative descriptions prefer Black

## Highlights & Insights

- This is the most systematic fairness audit of T2I reward models to date, revealing that RMs are far from neutral quality metrics
- The "demographic convergence" phenomenon (optimization causes diverse images to converge toward white) is a critical finding: RMs may act as adversaries of diversity
- The conclusion that RMs encode "dominant demographic conformity" rather than "quality" has far-reaching implications for RM design and deployment

## Limitations & Future Work

- Only one optimization method (ReNO) is used; other optimization strategies may exhibit different behaviors
- Reliance on automatic classifiers for demographic attribute assessment introduces measurement noise
- The sources of bias (training data vs. annotator preferences vs. architecture) are not deeply analyzed
- Debiased RM training methods need to be developed

## Related Work & Inspiration

- **vs Concept2Concept**: C2C found that the Pick-a-Pic dataset contains CSAM; this paper focuses on systematic demographic biases in RMs
- **vs T2I fairness research**: Prior work examines biases in generative models themselves; this paper reveals that RM biases as evaluation and optimization tools are equally severe

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic audit of demographic biases in T2I RMs
- Experimental rigor: ⭐⭐⭐⭐⭐ 5 RMs × 3 counterfactual datasets × multiple analysis methods
- Writing quality: ⭐⭐⭐⭐ Findings are clearly articulated
- Impact: ⭐⭐⭐⭐⭐ Significant implications for AI safety and fairness

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Exploring the Effects of Alignment on Numerical Bias in Large Language Models](../../AAAI2026/llm_alignment/exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models.md)
- [\[AAAI 2026\] BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models](../../AAAI2026/llm_alignment/biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)
- [\[AAAI 2026\] GRAM-R²: Self-Training Generative Foundation Reward Models for Reward Reasoning](../../AAAI2026/llm_alignment/gram-r2_self-training_generative_foundation_reward_models_for_reward_reasoning.md)
- [\[ACL 2026\] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](../../ACL2026/llm_alignment/beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)
- [\[CVPR 2026\] MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models](mapreduce_lora_advancing_the_pareto_front_in_multi-preference_optimization_for_g.md)

</div>

<!-- RELATED:END -->
