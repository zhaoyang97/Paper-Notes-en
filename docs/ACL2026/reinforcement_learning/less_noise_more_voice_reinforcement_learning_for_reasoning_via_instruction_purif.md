---
title: >-
  [Paper Note] LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification
description: >-
  [ACL 2026][Reinforcement Learning][RLVR] LENS identifies that many exploration failures in RLVR stem not from problem difficulty but from a small fraction (<5%) of distractor tokens in the prompt. By detecting and removing these tokens to improve rollout success rates, and transferring the learning signal from purified rollouts back to policy optimization on the original noisy prompts, LENS achieves an average improvement of 3.88% and a 1.6× training speedup.
tags:
  - ACL 2026
  - Reinforcement Learning
  - RLVR
  - distractor tokens
  - instruction purification
  - rollout efficiency
  - reasoning enhancement
date: 2026-05-08
content_hash: b1cb64e6f58315b4
---

# LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification

**Conference**: ACL 2026
**arXiv**: [2601.21244](https://arxiv.org/abs/2601.21244)
**Code**: [https://github.com/RUCBM/LENS](https://github.com/RUCBM/LENS)
**Area**: Reinforcement Learning / LLM Reasoning
**Keywords**: RLVR, distractor tokens, instruction purification, rollout efficiency, reasoning enhancement

## TL;DR

LENS identifies that many exploration failures in RLVR stem not from problem difficulty but from a small fraction (<5%) of distractor tokens in the prompt. By detecting and removing these tokens to improve rollout success rates, and transferring the learning signal from purified rollouts back to policy optimization on the original noisy prompts, LENS achieves an average improvement of 3.88% and a 1.6× training speedup.

## Background & Motivation

**Background**: RLVR methods (e.g., GRPO) have substantially enhanced LLM reasoning capabilities, yet face a core challenge on complex tasks: successful rollouts are extremely rare, leading to a scarcity of positive samples and causing low training efficiency or training collapse.

**Limitations of Prior Work**: Existing strategies take two directions — (1) scaling exploration (increasing rollout count), which is computationally expensive without improving efficiency; and (2) filtering zero-variance prompts (skipping prompts where all rollouts fail), which sacrifices exploration on hard samples. Neither approach addresses the root cause of exploration failure.

**Key Challenge**: Low-success-rate prompts contain valuable training signals, yet current methods either discard them (filtering) or brute-force them inefficiently (rollout scaling).

**Goal**: Identify the fundamental cause of exploration failure and design a targeted solution that improves rollout efficiency without increasing computational cost.

**Key Insight**: Token-level fine-grained analysis reveals that failures are often caused by a small number of tokens that introduce excessive distraction — pushing the policy far from the reference model in token space. Simply removing these tokens can improve rollout accuracy on failing prompts by over 20%.

**Core Idea**: A small number of distractor tokens in the prompt are the key cause of exploration failure. The proposed approach first "purifies" the prompt to obtain successful rollouts, then "transfers" the resulting learning signal back to the original prompt, training the model to ignore distractors rather than rely on purified inputs.

## Method

### Overall Architecture

LENS operates in two stages: (1) distractor token identification and purification — computing a distraction score for each token in the prompt and removing the highest-scoring ones; and (2) Calibrated Rollout Policy Optimization (CRPO) — replacing failed rollouts on the original prompt with successful rollouts from the purified prompt, and optimizing the policy on the original prompt via importance correction and sample reweighting.

### Key Designs

1. **Distractor Token Identification**:

    - Function: Precisely localize the small number of tokens in the prompt responsible for exploration failure.
    - Mechanism: The distraction score is defined as the absolute log-probability deviation between the current policy and the reference model at each token position: $S_I(s,a) = |\log \pi_\theta(a|s) - \log \pi_{\text{ref}}(a|s)|$. A high distraction score indicates that the token drives the policy to deviate substantially from the reference distribution in terms of KL divergence. Tokens are ranked in descending order of distraction score, and the top $k = \lceil \gamma \cdot |x_i| \rceil$ tokens are removed (with $\gamma$ set to 1%–5%) to produce the purified prompt.
    - Design Motivation: The reference model provides a stable distributional baseline learned from training data; large deviations typically arise from reward over-optimization or noisy/misleading signals.

2. **Calibrated Rollout Policy Optimization (CRPO)**:

    - Function: Transfer learning signals from purified prompts to policy optimization on original prompts.
    - Mechanism: For low-success-rate prompts (success rate < $\tau$), rollouts are sampled from the purified prompt. If purification yields an improved success rate, successful rollouts from the purified prompt replace an equal number of failed rollouts from the original. Crucially, all policy optimization is conducted on the original prompt via an importance ratio $\rho(y;\theta) = \frac{\pi_\theta(y|x_i)}{\tilde{w}(y) \pi_{\text{old}}(y|x^{\text{roll}}(y))}$ to correct for the distributional mismatch, so the model learns to reason correctly even under noisy conditions.
    - Design Motivation: Training directly on purified prompts amounts to learning in a clean environment, which does not transfer to real noisy settings. CRPO employs a transfer mechanism that teaches the model to identify and ignore distractors.

3. **Sample Reweighting**:

    - Function: Balance the contribution of original successful samples and purified successful samples.
    - Mechanism: The original success rate $\bar{a}_i$ is used as a scaling factor: original successful samples receive weight $\bar{a}_i$, while purified successful samples and unreplaced failed samples receive weight $1-\bar{a}_i$. Optimization uses a PPO-style clipped objective with KL regularization.
    - Design Motivation: When the original success rate is low, greater trust should be placed in signals from purified samples; when it is high, original samples should remain dominant.

### Loss & Training

PPO-style clipped objective: $\mathcal{L}(\theta) = -\sum_{y} \min(\rho(y;\theta)\hat{A}(y), \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\hat{A}(y)) + \beta D_{\text{KL}}$. Advantage values are computed in a group-relative manner over the reconstructed rollout set.

## Key Experimental Results

### Main Results

**Mathematical Reasoning Benchmarks Pass@1 (Llama3.2-3B-Instruct)**

| Method | MATH | Olympiad | AIME24 | Avg (7 benchmarks) |
|--------|------|----------|--------|-------------------|
| + GRPO | 51.60 | 44.68 | 6.25 | 23.98 |
| + DAPO | 53.00 | 47.01 | 9.79 | 25.32 |
| + GRPO_extended | 51.20 | 44.68 | 6.25 | 24.33 |
| + **LENS_GRPO** | **55.80** | **48.83** | **10.62** | **27.03** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Full LENS | Best | Identification + purification + CRPO |
| Purification only (no CRPO) | Second best | Model learns only in clean environment |
| Random deletion (replacing distraction score) | Degraded | Validates the effectiveness of the distraction score |
| Only ~20% of prompts benefit from deletion | — | Demonstrates the necessity of conditional activation in CRPO |

### Key Findings

- **Zero-reward prompt ratio substantially reduced**: LENS reduces the proportion of zero-reward prompts on DeepMath from ~80% (GRPO) to ~40%.
- Removing fewer than 5% of tokens improves rollout accuracy on failing prompts by over 20%, validating the hypothesis that a minority of tokens account for the majority of failures.
- LENS achieves the same performance as GRPO using only 60% of the training steps, corresponding to a 1.6× speedup.
- A 1.83% improvement is observed on four out-of-domain general reasoning benchmarks, indicating that robustness gains transfer beyond the training distribution.
- Compared to extended exploration (more rollouts) and prompt filtering, LENS achieves superior performance with lower computational cost.

## Highlights & Insights

- The finding that "a small number of distractor tokens causes exploration failure" is highly counterintuitive and compelling, opening a new research perspective for RLVR.
- The design philosophy of CRPO is elegant: rather than training the model in a clean environment and hoping it generalizes to noisy settings, CRPO uses signals from the clean environment to calibrate learning in the noisy one — effectively teaching the model to "ignore distractors."
- The distraction score (log-probability deviation between the policy and the reference model) is simple and efficient, requiring no additional models.

## Limitations & Future Work

- Validation is limited to models at the 3B–4B scale; effectiveness on larger models (7B+) remains unknown.
- The distraction score relies on the reference model; poor reference model quality may lead to misidentification.
- The removal ratio $\gamma$ requires manual tuning and may need to be adjusted for different datasets.
- Only approximately 20% of prompts benefit from token removal after purification, limiting the overall impact of CRPO's conditionally activated design.

## Related Work & Insights

- **vs. GRPO/DAPO**: LENS is a plug-and-play improvement that enhances rollout quality without modifying the underlying RL algorithm.
- **vs. exploration scaling methods**: Scaling exploration increases computational cost without improving efficiency, whereas LENS improves efficiency without additional cost.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Both the discovery of "distractor tokens" and the "purification + transfer" solution represent entirely novel perspectives.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-baseline comparisons and ablations are comprehensive, though model scale coverage is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, method descriptions are precise, and algorithmic pseudocode is well-presented.
- Value: ⭐⭐⭐⭐⭐ Provides a new perspective and practical solution for the exploration efficiency problem in RLVR.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Less is More: Clustered Cross-Covariance Control for Offline RL](../../ICLR2026/reinforcement_learning/less_is_more_clustered_cross-covariance_control_for_offline_rl.md)
- [\[ACL 2026\] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[NeurIPS 2025\] When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners](../../NeurIPS2025/reinforcement_learning/when_less_language_is_more_language-reasoning_disentanglement_makes_llms_better_.md)
- [\[ACL 2026\] Adaptive Instruction Composition for Automated LLM Red-Teaming](adaptive_instruction_composition_for_automated_llm_red-teaming.md)

<!-- RELATED:END -->
