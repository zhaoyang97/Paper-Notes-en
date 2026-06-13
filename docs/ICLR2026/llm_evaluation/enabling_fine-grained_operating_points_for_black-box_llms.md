---
title: >-
  [Paper Note] Enabling Fine-Grained Operating Points for Black-Box LLMs
description: >-
  [ICLR 2026][LLM Evaluation][Black-box LLM] This paper identifies that verbalized probabilities from black-box LLMs produce only 16–23 unique values (low-cardinality problem)…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Black-box LLM"
  - "operating points"
  - "probability calibration"
  - "PR curve"
  - "confidence estimation"
date: 2026-05-08
content_hash: a45df34dcf83d94a
---

# Enabling Fine-Grained Operating Points for Black-Box LLMs

**Conference**: ICLR 2026
**arXiv**: [2510.17727](https://arxiv.org/abs/2510.17727)  
**Code**: Not released (code snippets available in appendix)  
**Area**: LLM Evaluation
**Keywords**: Black-box LLM, operating points, probability calibration, PR curve, confidence estimation

## TL;DR
This paper identifies that verbalized probabilities from black-box LLMs produce only 16–23 unique values (low-cardinality problem), resulting in coarse PR/ROC curves that prevent fine-grained threshold tuning. By injecting parameterized noise and an optional MLP correction, the number of unique values increases from 16 to 20,000+, matching the performance of 20-sample ensembles with only 1–2 API calls.

## Background & Motivation

**Background**: When deploying LLMs as classifiers, practitioners must select an appropriate operating point on the precision-recall curve. A common approach is to prompt the LLM to output a verbalized probability in $[0,1]$ as a confidence score.

**Limitations of Prior Work**: LLMs exhibit a severe "rounding bias"—their output probabilities concentrate on a small set of values such as 0, 0.5, 0.85, 0.9, and 0.95 (with a tendency to end in 0 or 5), yielding PR/ROC curves with only a handful of discrete points and preventing fine-grained threshold control.

**Key Challenge**: The coarse spacing on the PR curve forces practitioners to choose between either high-precision/low-recall or low-precision/high-recall operating points, with no fine-grained trade-off available at deployment time.

**Goal**: How can black-box LLM probability outputs be made continuous without substantially increasing the number of API calls?

**Key Insight**: Inject parameterized noise onto discrete probabilities, effectively "diffusing" the discrete distribution into a continuous one.

**Core Idea**: Breaking rounding bias through noise injection—expanding the number of unique probability values from 16 to 20,000+ while preserving ranking performance.

## Method

### Overall Architecture
Three variants are proposed: (1) **Unsup**—unsupervised uniform noise injection; (2) **Sup-1call**—supervised MLP with noise, single API call; (3) **Sup-2call**—supervised MLP with noise, two API calls (at $T=0$ and $T=1$).

### Key Designs

1. **Unsupervised Noise (Ours-Unsup)**:

    - **Function**: Adds uniform noise to verbalized probabilities, maximizing noise magnitude while maintaining performance.
    - **Mechanism**: $\max w \text{ s.t. } \sum \text{loss}(y_i, \text{clip}(z_i \cdot w + y_{\text{vrb},i})) \leq \sum \text{loss}(y_i, y_{\text{vrb},i}),\ z \sim U(0,1)$. This finds the largest noise amplitude under the constraint that performance does not degrade.
    - **Design Motivation**: Requires no labeled data and is fully unsupervised. Increases cardinality from 16 to 5,614.

2. **Supervised Noise + MLP (Ours-Sup)**:

    - **Function**: Learns a correction function $f$ that maps discrete probabilities to better-calibrated outputs while simultaneously injecting noise.
    - **Mechanism**: $\min_{\theta_f, w} \sum \text{loss}(y_i, \text{sigmoid}(z_i / w + f(y_{\text{vrb}}; \theta_f))) + \lambda \cdot w,\ z \sim \mathcal{N}(0,1)$. Here $f$ is a 2-layer ReLU MLP that jointly learns the calibration correction and noise amplitude.
    - **Design Motivation**: The MLP correction addresses probability miscalibration (systematic over- or under-confidence), while noise injection resolves the cardinality problem (discretization).

## Key Experimental Results

### Cardinality Improvement

| Method | API Calls | Unique Values | Cardinality Gain |
|---|---|---|---|
| Prompt-Naive | 1 | 10 | 1× |
| Sample-Class (20×) | 20 | 97 | 10× |
| **Ours-Unsup** | **1** | **5,614** | **561×** |
| **Ours-Sup-2call** | **2** | **20,607** | **2,061×** |

### Performance Comparison (11 datasets combined)

| Method | API Calls | PRAUC | Precision Granularity |
|---|---|---|---|
| Prompt-Naive | 1 | 0.72 | 0.081 |
| Sample-Prob | 20 | 0.78 | 0.014 |
| **Ours-Sup-2call** | **2** | **0.79** | **0.016** |

### Key Findings
- Sup-2call surpasses 20-sample ensembling with only 2 API calls (PRAUC 0.79 vs. 0.78).
- Noise injection is necessary—MLP calibration alone without noise fails to resolve the cardinality problem.
- Results are consistent across 11 diverse datasets spanning sentiment classification to fact verification.

## Highlights & Insights
- **Engineering-driven problem discovery**: The paper identifies and quantifies the rounding bias in LLM probability outputs (16–23 unique values), which is a valuable observation in its own right.
- **Noise as regularization**: Adding noise does not degrade signal quality; rather, it smooths an over-discretized distribution into a continuous space, improving the flexibility of downstream decision-making.
- **Exceptional cost efficiency**: 2 API calls outperform 20-sample ensembling, reducing API costs by 90%.

## Limitations & Future Work
- The supervised variant requires labeled data for MLP training, making it unsuitable for cold-start scenarios.
- Validation is limited to Claude and a subset of open-source models; rounding bias may vary across different LLMs.
- The noise amplitude and MLP architecture are fixed; adaptive schemes may yield further improvements.

## Related Work & Insights
- **vs. Standard Sampling**: Multi-sample ensembling (20 calls) increases cardinality but incurs linearly growing costs; this work achieves comparable results with 1–2 calls.
- **vs. Probability Calibration**: Methods such as Platt Scaling improve calibration but do not address the cardinality problem; this work resolves both simultaneously.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel problem identification (rounding bias) with an intuitive and effective solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 11 datasets, multiple baselines, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation with rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Direct practical value for LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Predicting the Performance of Black-Box LLMs through Follow-Up Queries](../../NeurIPS2025/llm_evaluation/predicting_the_performance_of_black-box_llms_through_follow-up_queries.md)
- [\[ICML 2026\] Investigating Advanced Reasoning of Large Language Models via Black-Box Environment Interaction](../../ICML2026/llm_evaluation/investigating_advanced_reasoning_of_large_language_models_via_black-box_environm.md)
- [\[ICLR 2026\] When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)
- [\[ACL 2026\] IF-Critic: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation](../../ACL2026/llm_evaluation/if-critic_towards_a_fine-grained_llm_critic_for_instruction-following_evaluation.md)
- [\[ICML 2026\] REAL: Integrating Regression-Aware Rewards into RL for Fine-Grained Human-Centric LLM Evaluation](../../ICML2026/llm_evaluation/real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
