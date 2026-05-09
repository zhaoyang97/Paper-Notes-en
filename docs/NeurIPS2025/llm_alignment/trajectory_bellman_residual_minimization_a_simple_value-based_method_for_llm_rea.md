---
title: >-
  [Paper Note] Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning
description: >-
  [NeurIPS 2025][LLM Alignment][Bellman residual] TBRM minimizes trajectory-level Bellman residuals by treating LLM output logits as implicit Q-values, requiring only a single forward rollout per prompt during training. This yields substantially lower complexity than PPO/GRPO while achieving comparable or superior performance on mathematical reasoning benchmarks.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - Bellman residual
  - value learning
  - single-sample
  - critic-free
  - mathematical reasoning
date: 2026-05-08
content_hash: 76913e8d29cf6703
---

# Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2505.15311](https://arxiv.org/abs/2505.15311)
**Code**: To be released
**Area**: LLM Alignment
**Keywords**: Bellman residual, value learning, single-sample, critic-free, mathematical reasoning

## TL;DR
TBRM minimizes trajectory-level Bellman residuals by treating LLM output logits as implicit Q-values, requiring only a single forward rollout per prompt during training. This yields substantially lower complexity than PPO/GRPO while achieving comparable or superior performance on mathematical reasoning benchmarks.

## Background & Motivation

**Background**: PPO and GRPO have achieved notable success in LLM alignment, yet they rely on complex components such as critic models, multiple rollouts, and importance sampling.

**Limitations of Prior Work**: PPO requires 32 samples per prompt and GRPO requires 16, resulting in high computational cost, numerous hyperparameters, and training instability.

**Key Challenge**: Value-based methods in RL are generally more efficient and stable, but their application to the high-dimensional discrete action spaces of LLMs remains largely unexplored.

**Goal**: Design a simple value-learning method that requires only a single sample per prompt.

**Key Insight**: Treat LLM logits directly as Q-value estimators and minimize the trajectory-level Bellman residual.

**Core Idea**: $\mathcal{L}_{TBRM} = (\log \pi_m(y_w|x) - \log \pi_m(y_l|x) - R^*)^2$, driving the log-likelihood difference between correct and incorrect responses toward a target margin.

## Method

### Overall Architecture
Given a prompt $x$, the method generates a response pair $(y_w, y_l)$ (correct/incorrect) and minimizes the squared difference between the log-likelihood margin and the target reward $R^*$. No critic model, value normalization, or TD-$\lambda$ is required.

### Key Designs

1. **Trajectory-Level Bellman Residual**:

    - Function: Minimize $BR(x) = [\log \pi_m(y_w|x) - \log \pi_m(y_l|x) - R^*]^2$
    - Mechanism: Uses the log-likelihood of the entire response sequence as a proxy for the Q-value
    - Design Motivation: Trajectory-level targets are more stable than token-level TD and eliminate the need for intermediate value estimation

2. **Theoretical Convergence Guarantee**:

    - Convergence objective: $\pi_{opt} = \arg\min_\pi D_{KL}(\pi_{data} \| \pi) - \bar{R}(\pi)$
    - Consistent with the original RLHF motivation — staying close to the reference policy while maximizing reward

## Key Experimental Results

### Main Results

| Method | AIME24 | MATH | AMC | Minerva-Math | Avg. |
|--------|--------|------|-----|-------------|------|
| PPO | 27.3% | 41.2% | 48.7% | 59.4% | 44.2% |
| GRPO | 29.1% | 43.1% | 49.2% | 61.5% | 45.7% |
| **TBRM** | **30.5%** | **44.8%** | **50.3%** | **62.9%** | **47.1%** |

### Efficiency Comparison

| Metric | PPO | GRPO | TBRM |
|--------|-----|------|------|
| Samples per prompt | 32 | 16 | **1** |
| GPU memory (GB) | 48 | 40 | **32** |
| Training time (h) | 24.5 | 22.1 | **19.0** |

### Key Findings
- TBRM surpasses PPO by 3.2% on AIME24 while achieving a 32× improvement in sampling efficiency
- Eliminates the critic model, reducing GPU memory consumption by 33%
- The method is remarkably concise; the core implementation likely requires fewer than 50 lines of code

## Highlights & Insights
- **Extreme Simplicity**: No critic, no value normalization, no TD-$\lambda$ — only the trajectory log-likelihood margin
- **Single-Sample Efficiency**: Requires only one preference pair per prompt, yielding an 8–16× improvement in computational efficiency
- **Theoretical Consistency**: The convergence objective aligns with the original KL-regularized motivation of RLHF

## Limitations & Future Work
- The binary reward (correct/incorrect) assumption limits applicability to tasks with graded rewards, such as instruction following
- The selection of the target margin $R^*$ is heuristic
- When the initial policy is weak, the quality of generated data may be limited

## Related Work & Insights
- **vs. PPO/GRPO**: TBRM replaces complex policy gradient methods with a simpler value-based approach
- **vs. DPO**: DPO employs a contrastive loss whereas TBRM minimizes Bellman residuals — the theoretical motivations differ despite their formal similarity

## Rating
- Novelty: ⭐⭐⭐⭐ Adapts Bellman residual minimization to LLM reasoning
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four mathematical benchmarks with detailed efficiency analysis
- Writing Quality: ⭐⭐⭐⭐ Algorithm description is concise and reproducible
- Value: ⭐⭐⭐⭐⭐ Significant efficiency gains; strongly recommended for industrial applications
**Code**: To be confirmed

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training](trajectory_balance_with_asynchrony_decoupling_exploration_and_learning_for_fast_.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](../../ICLR2026/llm_alignment/slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)

</div>

<!-- RELATED:END -->
