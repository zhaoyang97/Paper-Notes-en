---
title: >-
  [Paper Note] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning
description: >-
  [ICLR 2026][LLM Alignment][Reinforcement Learning] This paper proposes SFPO (Slow-Fast Policy Optimization), which decomposes each training step into a three-stage structure of "fast trajectory — reposition — slow correc…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Reinforcement Learning"
  - "GRPO"
  - "Policy Optimization"
  - "Mathematical Reasoning"
  - "Sample Efficiency"
date: 2026-05-08
content_hash: 594ca1b8cdd3ec3b
---

# Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning

**Conference**: ICLR 2026
**arXiv**: [2510.04072](https://arxiv.org/abs/2510.04072)
**Code**: [slow-fast-po.github.io](https://slow-fast-po.github.io/)
**Area**: LLM Alignment
**Keywords**: Reinforcement Learning, GRPO, Policy Optimization, Mathematical Reasoning, Sample Efficiency

## TL;DR

This paper proposes SFPO (Slow-Fast Policy Optimization), which decomposes each training step into a three-stage structure of "fast trajectory — reposition — slow correction." Without modifying the objective function or rollout procedure, SFPO serves as a plug-and-play enhancement to GRPO, achieving up to 2.80-point average improvement on mathematical reasoning benchmarks and up to 4.93× reduction in rollouts.

## Background & Motivation

- Reinforcement learning (RL) has become a core technique for improving LLM reasoning, with GRPO being a widely adopted critic-free policy gradient method.
- **Limitations of GRPO**:
    - Poor rollout quality in early training leads to **high-variance gradients** due to random rewards, causing unstable updates.
    - Each batch of rollouts supports only a **single update step** (one-shot), underutilizing available gradient information.
    - Naively reusing rollout data introduces off-policy bias, which can degrade performance in later training.
- A mechanism is needed to stabilize gradient directions, improve sample utilization, and control distributional shift simultaneously.

## Method

### Overall Architecture

SFPO decomposes each training iteration into three coordinated stages:

1. **Stage I: Fast Trajectory** — Perform $K$ inner-loop update steps on the same batch of rollouts.
2. **Stage II: Reposition** — Interpolate back toward the starting point to control off-policy drift.
3. **Stage III: Slow Correction** — Perform one additional gradient update at the interpolated point.

### Stage I: Fast Trajectory

Starting from parameters $\theta^{s,0}$, perform $K$ inner-loop update steps:

$$\theta^{s,k+1} = \theta^{s,k} - \eta \nabla_\theta \mathcal{L}(\theta^{s,k}), \quad k=0,\ldots,K-1$$

- Unlike single-step updates, the displacement $\theta^{s,K} - \theta^{s,0} = -\eta \sum_{k=0}^{K-1} \nabla_\theta \mathcal{L}(\theta^{s,k})$ accumulates the effect of $K$ gradients.
- Under a second-order approximation, this is equivalent to a **curvature-aware low-pass filter**: making steady progress along flat directions while automatically suppressing oscillations along high-curvature directions.

### Stage II: Reposition

Inspired by the Lookahead Optimizer, the endpoint of the fast trajectory is interpolated back toward the starting point:

$$\widetilde{\theta}^{s,K} = \theta^{s,0} + \alpha(\theta^{s,K} - \theta^{s,0}), \quad \alpha \in [0,1]$$

- $\alpha$ controls the degree of off-policy drift: a smaller $\alpha$ implies stronger proximal regularization.
- This is equivalent to solving a linearized proximal subproblem centered at $\theta^{s,0}$, with $\alpha$ acting as an implicit trust-region radius.

### Stage III: Slow Correction

One additional gradient update is performed at the interpolated point:

$$\theta^{s+1} = \widetilde{\theta}^{s,K} - \eta \nabla_\theta \mathcal{L}(\widetilde{\theta}^{s,K})$$

This forms a predictor-corrector structure, ensuring the final update is aligned with local curvature.

### Unified Update Formula

$$\theta^{s+1} = \theta^{s,0} - \eta \left[\alpha \sum_{k=0}^{K-1} \nabla_\theta \mathcal{L}(\theta^{s,k}) + \nabla_\theta \mathcal{L}(\widetilde{\theta}^{s,K})\right]$$

### Adaptive $\alpha$ Scheduling

- Policy entropy $H_s$ is monitored via a rolling z-score $Z_s = (H_s - \mu_s) / \sigma_s$.
- When $|Z_s| \geq \tau$, $\alpha \to 0$ is triggered, reducing SFPO to standard GRPO.
- The fast trajectory accelerates convergence in early training, while pure on-policy updates maintain stability in later stages.

### Loss & Training

SFPO does not modify the underlying loss function and directly uses the GRPO objective:

$$\mathcal{J}_{GRPO}(\theta) = \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}) - \beta D_{KL}[\pi_\theta \| \pi_{ref}]$$

## Key Experimental Results

### Main Results: Mathematical Reasoning Benchmarks (DAPO+Math Training Set)

| Model | Method | Math-500 | AIME24 | AIME25 | AMC | Minerva | Olympiad | Avg |
|-------|--------|----------|--------|--------|-----|---------|----------|-----|
| Qwen2.5-Math-1.5B | GRPO | 77.15 | 16.67 | 11.67 | 53.31 | 31.89 | 39.42 | 38.35 |
| | **SFPO** | **78.35** | **20.00** | **15.00** | **56.02** | **32.07** | **39.72** | **40.19** |
| DS-Qwen-1.5B | GRPO | 84.65 | 30.00 | 23.33 | 66.86 | 31.71 | 49.85 | 47.73 |
| | **SFPO** | **86.10** | **32.50** | **30.83** | **70.28** | **32.81** | **50.67** | **50.53** |
| DS-Qwen-7B | GRPO | 91.70 | 50.00 | 35.83 | 80.42 | 43.65 | 61.24 | 60.47 |
| | **SFPO** | **92.60** | **54.17** | **37.50** | **83.75** | **44.49** | **65.73** | **63.04** |

### Ablation Study: Efficiency Analysis

| Model | Rollout Reduction | Training Time Reduction |
|-------|-------------------|------------------------|
| DS-Qwen-1.5B | 3.21× | 2.62× |
| Qwen3-4B-Base | 3.50× | 2.65× |
| DS-Qwen-7B | **4.93×** | **4.19×** |

### Key Findings

1. SFPO consistently outperforms GRPO across all 5 models and 6 benchmarks, with the largest gains on smaller models (+2.80 on DS-Qwen-1.5B).
2. Training dynamics analysis shows that SFPO avoids the response length collapse observed in GRPO.
3. SFPO introduces no additional GPU memory overhead, as no extra optimizer states need to be stored.
4. Consistent gains are maintained on the larger Skywork-OR1 training set (105K samples).

## Highlights & Insights

- **Plug-and-play design**: The loss function, rollout generation, and regularization are left entirely unchanged; SFPO directly replaces the update step of GRPO.
- **Clear theoretical intuition**: Fast trajectory = curvature-aware low-pass filter; reposition = implicit trust region; slow correction = local curvature alignment.
- **Adaptive exit mechanism**: The entropy-monitored $\alpha$ schedule automatically degrades to GRPO during convergence, balancing efficiency and stability.
- **Substantial sample efficiency gains**: Up to 4.93× fewer rollouts are required to reach equivalent accuracy.

## Limitations & Future Work

- Several additional hyperparameters are introduced ($K$, $\alpha_0$, $\omega$, $\tau$), though experiments suggest the method is robust to their selection.
- Theoretical analysis relies primarily on approximate derivations under the L-smooth assumption; the actual loss landscape of LLMs is considerably more complex.
- Validation is limited to mathematical reasoning tasks; the method has not been tested on other reasoning domains such as code generation or multimodal reasoning.

## Related Work & Insights

- Policy gradient enhancements: DAPO, Dr.GRPO, and related methods address GRPO improvements from different angles.
- Lookahead Optimizer: The reposition mechanism in SFPO is directly inspired by this work.
- Sample efficiency: Methods such as ReMax and RLOO also target improved rollout utilization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The three-stage fast–reposition–slow structure constitutes a novel policy optimization paradigm.
- **Technical Depth**: ⭐⭐⭐⭐ — Theoretical derivations are thorough, spanning curvature analysis, proximal optimization, and adaptive scheduling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 5 models, 6 benchmarks, 2 training sets, with efficiency and training dynamics analysis.
- **Practical Value**: ⭐⭐⭐⭐⭐ — Plug-and-play integration, no additional memory cost, and significant speedup make this highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](reinforcement_unlearning_via_group_relative_policy_optimization.md)
- [\[ICLR 2026\] Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)
- [\[ICLR 2026\] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[ICLR 2026\] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](avere_improving_audiovisual_emotion_reasoning_with_preference_optimization.md)

</div>

<!-- RELATED:END -->
