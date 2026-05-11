---
title: >-
  [Paper Note] GVPO: Group Variance Policy Optimization for Large Language Model Post-Training
description: >-
  [NeurIPS 2025][LLM Alignment][GRPO] GVPO is a more stable LLM post-training method than GRPO, derived by embedding the analytical solution of KL-constrained reward maximization into gradient weights (zero-sum weights eli…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "GRPO"
  - "Policy Optimization"
  - "KL Constraint"
  - "Training Stability"
  - "Post-Training"
date: 2026-05-08
content_hash: 8e63b26fac0c4c18
---

# GVPO: Group Variance Policy Optimization for Large Language Model Post-Training

**Conference**: NeurIPS 2025
**arXiv**: [2504.19599](https://arxiv.org/abs/2504.19599)
**Code**: [https://github.com/jszkc/GVPO](https://github.com/jszkc/GVPO)
**Area**: Alignment / RLHF
**Keywords**: GRPO, Policy Optimization, KL Constraint, Training Stability, Post-Training

## TL;DR
GVPO is a more stable LLM post-training method than GRPO, derived by embedding the analytical solution of KL-constrained reward maximization into gradient weights (zero-sum weights eliminate the partition function). It achieves 20.72% on AIME (vs. GRPO's 14.79%) and is proven to possess a unique global optimum.

## Background & Motivation

**Background**: Post-training methods such as GRPO have achieved strong performance through increased sampling and relative reward scoring, yet suffer from severe training instability and high sensitivity to hyperparameters (clipping threshold, KL coefficient).

**Limitations of Prior Work**: GRPO's instability stems from two sources: ① minimizing the log of negative probabilities can be numerically unstable; ② importance sampling weights in off-policy training cause gradient explosion when the policy deviates significantly.

**Key Challenge**: DPO admits a closed-form solution but may have multiple minima and does not guarantee convergence to the true optimal policy; GRPO is flexible but training is unstable.

**Goal**: Design a method that combines the theoretical advantages of DPO (closed-form optimal solution) while overcoming its weaknesses (convergence guarantees), and simultaneously supports flexible off-policy training.

**Key Insight**: A key observation — when the sum of gradient weights for all responses within a group is zero, the partition function $Z(x)$ becomes invariant across responses and is therefore eliminated.

**Core Idea**: Zero-sum weight design eliminates the partition function + variance/covariance regularization ensures stability = theoretically grounded, stable post-training.

## Method

### Overall Architecture
The gradient weights of GVPO equal the difference between the centered actual reward and the centered implicit reward, satisfying $\sum_i w_i = 0$, which naturally eliminates the partition function.

### Key Designs

1. **Zero-Sum Weights to Eliminate the Partition Function**:

    - Function: Design a gradient weighting scheme satisfying $\sum_i w_i = 0$
    - Mechanism: $w_i = (R(x,y_i) - \bar{R}) - \beta(\log\frac{\pi_\theta(y_i|x)}{\pi_{\theta'}(y_i|x)} - \overline{\log\frac{\pi_\theta}{\pi_{\theta'}}})$
    - Design Motivation: Directly exploit the analytical relationship of the KL-constrained optimal policy, avoiding estimation of the intractable $Z(x)$

2. **Three-Component Decomposition (RL Perspective)**:

    - Function: Decompose the loss into advantage maximization + variance regularization + covariance regularization
    - Mechanism: The advantage term prioritizes high-reward responses; the variance term balances exploration and exploitation; the covariance term acts as a trust-region constraint
    - Design Motivation: Ablations show that removing any single component causes training divergence — all three are indispensable

3. **Support for Flexible Sampling Distributions**:

    - Function: Guarantee optimality for any sampling distribution satisfying the support condition (Theorem 3.2)
    - Mechanism: On-policy sampling is not required; off-policy training, data reuse, and mixed data are all supported
    - Design Motivation: Avoids importance sampling weight explosion and is more flexible than conventional policy gradient methods

### Loss & Training
The GVPO loss is equivalent to MSE between the centered implicit reward and the centered actual reward, guaranteeing a unique global optimum $\pi^*(y|x) = \frac{1}{Z(x)}\pi_{\theta'}(y|x)e^{R(x,y)/\beta}$.

## Key Experimental Results

### Main Results (Mathematical Reasoning)

| Algorithm | AIME2024 | AMC | MATH500 | Minerva | OlympiadBench |
|-----------|----------|-----|---------|---------|---------------|
| Qwen2.5-Math-7B | 14.68 | 38.55 | 64.00 | 27.20 | 30.66 |
| + GRPO | 14.79 | 55.42 | 80.00 | 41.17 | 42.07 |
| + Dr.GRPO | 16.56 | 48.19 | 81.20 | 44.48 | 43.40 |
| **+ GVPO** | **20.72** | **62.65** | **83.80** | **45.95** | **46.96** |

### Ablation Study

| Configuration | Result |
|---------------|--------|
| Full GVPO | Converges, best performance |
| w/o variance regularization | Training fully diverges |
| w/o covariance regularization | Training fully diverges |
| w/o both | Converges initially, then diverges at ~10% of steps |

### Key Findings
- GVPO achieves the best results on all 5 benchmarks, with an absolute AIME gain of +5.93%
- Robust to $\beta \in [0.05, 0.2]$ with small variance (vs. GRPO's high hyperparameter sensitivity)
- Increasing the number of samples $k$ for a 1.5B model can match the performance of a 7B model — model scale is exchangeable with sampling

## Highlights & Insights
- **Clear Theoretical Advantages**: A unique global optimum is stronger than DPO's multiple minima, providing more robust convergence guarantees
- **Elegance of the Variance Decomposition**: The three components naturally achieve three distinct objectives without requiring manual tuning of complex coefficients
- **Off-Policy Flexibility Breakthrough**: Mixed data and historical data can both be leveraged, substantially reducing sampling costs

## Limitations & Future Work
- Validation is primarily on mathematical reasoning; evaluation on diverse tasks such as language understanding and safety alignment is lacking
- The respective contributions of the regularization terms and sampling flexibility to overall improvement are not fully disentangled

## Related Work & Insights
- **vs. GRPO**: More stable (low variance across 10 seeds), more flexible (no on-policy requirement), higher performing (AIME +5.93%)
- **vs. DPO**: Stronger convergence guarantees (unique optimum vs. possible multiple minima), supports flexible sampling distributions

## Rating
- Novelty: ⭐⭐⭐⭐ Zero-sum weights eliminating the partition function is a clever innovation; multi-perspective decomposition adds theoretical depth
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset validation with thorough ablations, though the evaluation domain is narrow
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and the three-perspective explanation is intuitive
- Value: ⭐⭐⭐⭐ A more stable post-training method has direct industrial applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](../../ICLR2026/llm_alignment/chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [\[NeurIPS 2025\] Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training](trajectory_balance_with_asynchrony_decoupling_exploration_and_learning_for_fast_.md)
- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)
- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[ICLR 2026\] PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](../../ICLR2026/llm_alignment/reinforcement_unlearning_via_group_relative_policy_optimization.md)

</div>

<!-- RELATED:END -->
