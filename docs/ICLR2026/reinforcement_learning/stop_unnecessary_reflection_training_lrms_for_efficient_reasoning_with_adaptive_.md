---
title: >-
  [Paper Note] Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty
description: >-
  [ICLR 2026][Reinforcement Learning][Large Reasoning Models] This paper proposes ARLCP (Adaptive Reflection and Length Coordinated Penalty), an adaptive reinforcement learning method that dynamically adjusts the weights of reflection and length penalties according to problem complexity, substantially reducing token consumption while maintaining or improving accuracy.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Large Reasoning Models
  - Over-reflection
  - Adaptive Penalty
  - Efficient Reasoning
  - RLVR
date: 2026-05-08
content_hash: 5581d27d7b07a397
---

# Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty

**Conference**: ICLR 2026
**arXiv**: [2602.12113](https://arxiv.org/abs/2602.12113)
**Code**: [https://github.com/ZeweiYu1/ARLCP](https://github.com/ZeweiYu1/ARLCP)
**Area**: Reinforcement Learning
**Keywords**: Large Reasoning Models, Over-reflection, Adaptive Penalty, Efficient Reasoning, RLVR

## TL;DR

This paper proposes ARLCP (Adaptive Reflection and Length Coordinated Penalty), an adaptive reinforcement learning method that dynamically adjusts the weights of reflection and length penalties according to problem complexity, substantially reducing token consumption while maintaining or improving accuracy.

## Background & Motivation

- **Over-reasoning problem**: Large reasoning models (LRMs) such as DeepSeek-R1 generate excessive redundant reflections in their chain-of-thought (e.g., repeated "wait" and "hmm" tokens), incurring high token consumption and computational overhead without improving accuracy.
- **Key observations**:
  1. **Reflection correlates with complexity**: Harder problems elicit more reflection tokens.
  2. **Over-reflection induces errors**: Incorrect responses contain on average far more reflection tokens than correct ones.
  3. **Accuracy declines with excessive reflection**: Beyond a certain threshold, additional reflection actually degrades accuracy.
- **Limitations of prior work**:
  - Inference-stage methods (e.g., Early Exit) do not alter model capabilities, yielding limited efficiency gains.
  - Training-stage methods (e.g., uniform length penalties) frequently sacrifice reasoning quality.
  - No existing mechanism dynamically adjusts penalties based on problem complexity.

## Method

### Overall Architecture

ARLCP introduces two coordinated penalty mechanisms into reinforcement learning training — an adaptive reflection penalty and a length penalty — optimized via RLOO (REINFORCE Leave One Out).

### 1. Complexity Estimation

Problem complexity as perceived by the model is estimated through a Reflection Token Count (RTC), categorized into three levels:
- **Easy**: $\text{RTC}(o_i^k) \leq n_1$, weight $\lambda_1$
- **Medium**: $n_1 < \text{RTC}(o_i^k) \leq n_2$, weight $\lambda_2$
- **Hard**: $\text{RTC}(o_i^k) > n_2$, weight $\lambda_3$

### 2. Adaptive Reflection Penalty

The reflection penalty coefficient $\alpha_1$ is dynamically adjusted according to complexity:

$$\alpha_1 = \begin{cases} \lambda_1, & \text{if } \text{RTC}(o_i^k) \leq n_1 \\ \lambda_2, & \text{if } n_1 < \text{RTC}(o_i^k) \leq n_2 \\ \lambda_3, & \text{if } \text{RTC}(o_i^k) > n_2 \end{cases}$$

The reflection penalty value is normalized via sigmoid:

$$f(\text{RTC}(o_i^k)) = \sigma\left(\frac{\text{RTC}(o_i^k) - \text{mean}(\text{RTC}(o_i))_{\text{correct}}}{\text{std}(\text{RTC}(o_i))_{\text{correct}}}\right)$$

### 3. Length Penalty

A complementary penalty suppresses non-reflective redundancy:

$$f(\text{LEN}(o_i^k)) = \sigma\left(\frac{\text{LEN}(o_i^k) - \text{mean}(\text{LEN}(o_i))_{\text{correct}}}{\text{std}(\text{LEN}(o_i))_{\text{correct}}}\right)$$

The length penalty coefficient is set as $\alpha_2 = \alpha - \alpha_1$, ensuring that the total penalty budget $\alpha$ is flexibly allocated between reflection and length penalties.

### 4. Composite Reward Function

$$r(o_i^k) = \mathcal{C}(o_i^k) \cdot \left(1 - \alpha_1 f(\text{RTC}(o_i^k)) - \alpha_2 f(\text{LEN}(o_i^k))\right)$$

where $\mathcal{C}(o_i^k) = \mathbf{1}\{\text{ANS}(o_i^k) = o^*(p_i)\}$ denotes the correctness reward.

### Key Designs

- **RLOO** is adopted over GRPO, as GRPO is unstable under non-standard length penalty settings.
- Statistical baselines (mean and std) are computed exclusively from **correct responses** to avoid noise.
- Hyperparameters: $\lambda_1=0.05,\ \lambda_2=0.1,\ \lambda_3=0.15,\ n_1=40,\ n_2=80,\ \alpha=0.2$.

## Key Experimental Results

### Main Results: DeepSeek-R1-Distill-Qwen-1.5B

| Method | AMC2023 Acc | AIME2024 Acc | AIME2025 Acc | GSM8K Acc | MATH500 Acc | ΔAcc | ΔLength |
|--------|------------|-------------|-------------|-----------|------------|------|---------|
| Vanilla | 66.72 | 30.00 | 21.40 | 78.46 | 80.20 | - | - |
| NoThinking | 49.22 | 14.38 | 9.79 | 69.98 | 69.20 | -12.84 | -81.04% |
| TLMRE | 72.10 | 25.80 | 19.60 | 84.30 | 82.10 | +1.42 | -58.10% |
| AdaptThink | 67.19 | 30.83 | 22.50 | 84.23 | 83.20 | +2.23 | -51.47% |
| LASER | 75.94 | 28.75 | 25.42 | 82.26 | 84.60 | +4.04 | -38.69% |
| **ARLCP** | **73.28** | **34.17** | **26.46** | **87.34** | **84.60** | **+5.81** | **-53.05%** |

### DeepSeek-R1-Distill-Qwen-7B

| Method | ΔAcc | ΔLength |
|--------|------|---------|
| Vanilla | - | - |
| AdaptThink | +1.87 | -34.68% |
| **ARLCP** | **+2.70** | **-35.00%** |

### Ablation Study

| Setting | ΔAcc | ΔLength |
|---------|------|---------|
| ARLCP (full) | +5.81 | -53.05% |
| Reflection penalty only (no length penalty) | +4.2 | -45.3% |
| Length penalty only (no reflection penalty) | +2.1 | -48.7% |
| Fixed penalty (non-adaptive) | +3.5 | -50.1% |

### Key Findings

- On the 1.5B model: length reduced by **53.1%**, accuracy improved by **5.8%**.
- On the 7B model: length reduced by **35.0%**, accuracy improved by **2.7%**.
- The adaptive mechanism substantially outperforms fixed penalties.
- The two penalty components are complementary and both essential.

## Highlights & Insights

1. **In-depth empirical analysis**: Systematically reveals the over-reflection phenomenon and its relationship to problem complexity.
2. **Reflection tokens as a complexity proxy**: The model's own reflective behavior is leveraged to estimate problem difficulty, eliminating the need for external complexity assessment.
3. **Dynamic penalty allocation**: The total budget $\alpha$ is automatically distributed between reflection and length penalties according to complexity.
4. **Dual gains in efficiency and accuracy**: Token consumption is substantially reduced while accuracy simultaneously improves.

## Limitations & Future Work

- The complexity classification thresholds $(n_1, n_2)$ require manual specification.
- Reflection tokens are detected via keyword matching ("wait", "hmm", "alternatively"), which may lack precision.
- Validation is limited to mathematical reasoning tasks; generalization to domains such as code reasoning remains unexplored.
- The method relies on DeepSeek-R1 distilled models; its effectiveness on non-distilled models warrants further investigation.

## Related Work & Insights

- **Efficient reasoning**: Early Exit, Model Switch, NoThinking (skipping the thinking phase).
- **Training-stage methods**: TLMRE (length-penalized RL), LASER (accuracy-based length constraint).
- **SFT methods**: SFT-Shortest (fine-tuning on the shortest correct responses).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Adaptive reflection penalty constitutes a novel starting point.
- **Technical Depth**: ⭐⭐⭐ — The method is relatively straightforward, yet well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple benchmarks and models with comprehensive comparisons.
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses the efficiency bottleneck in LRM deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)
- [\[ICLR 2026\] unsupervised learning of efficient exploration pre-training adaptive policies vi](unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)
- [\[ICLR 2026\] RM-R1: Reward Modeling as Reasoning](rm-r1_reward_modeling_as_reasoning.md)
- [\[ICLR 2026\] FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)
- [\[ICLR 2026\] Learning from Synthetic Data Improves Multi-hop Reasoning](learning_from_synthetic_data_improves_multi-hop_reasoning.md)

<!-- RELATED:END -->
