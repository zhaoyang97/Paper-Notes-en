---
title: >-
  [Paper Note] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control
description: >-
  [NeurIPS 2025][Reinforcement Learning][mobile app control] This paper proposes the SoLS algorithm, which achieves sample-efficient RL fine-tuning of foundation models for mobile app control through an asymmetric policy u…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "mobile app control"
  - "off-policy learning"
  - "foundation model fine-tuning"
  - "sample efficiency"
date: 2026-05-08
content_hash: f75afa1de0ad6765
---

# Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control

**Conference**: NeurIPS 2025
**arXiv**: [2509.01720](https://arxiv.org/abs/2509.01720)
**Code**: Not yet available
**Area**: Reinforcement Learning
**Keywords**: reinforcement learning, mobile app control, off-policy learning, foundation model fine-tuning, sample efficiency

## TL;DR

This paper proposes the SoLS algorithm, which achieves sample-efficient RL fine-tuning of foundation models for mobile app control through an asymmetric policy update mechanism (aggressive learning on success, conservative regularization on failure) combined with Success Transition Replay (STR), attaining a 51.3% success rate on AndroidWorld.

## Background & Motivation

Mobile app control is a highly challenging multi-step interactive task: the action space is large and context-dependent, each simulation step takes several seconds, and most environments provide only sparse binary rewards (success/failure). Existing approaches exhibit clear bottlenecks:

**GPT-4o prompt engineering**: Although capable of leveraging strong prior knowledge, each step requires multiple API calls, incurring inference latency of 40–60 seconds and substantial operational costs.

**SFT fine-tuning**: Small models trained on human demonstration data have limited generalization, particularly when the target task distribution diverges significantly from the training data.

**Standard RL methods**: In the context of foundation model fine-tuning, these face two fundamental problems — policy gradient updates on negative-advantage samples disrupt learned representations and lead to performance degradation.

The paper's core insight is that **policy regularization is unnecessary for positive-advantage updates, whereas negative-advantage updates can harm model performance**. When the advantage function is negative, policy gradient reduces the probability of the current action token, uncontrollably increasing the probability of other tokens in the vocabulary and thereby corrupting the model's semantic structure.

## Method

### Overall Architecture

The system adopts a two-stage pipeline: first, SFT pre-training on the AndroidControl dataset (to acquire basic action prediction capability and output formatting), followed by online RL fine-tuning using SoLS+STR in the AndroidWorld environment. The backbone model is Llama-3-8B-Instruct, using only text UI trees as input.

### Key Designs

1. **Asymmetric Policy Updates (Core Mechanism of SoLS)**

   SoLS is built on an off-policy actor-critic framework but applies entirely different update strategies to positive- and negative-advantage samples. The advantage function is defined as $A(s,a) = R - V^{\pi_\theta}(s)$, where $R$ is the Monte Carlo return. The policy gradient is:

   $$\nabla\mathcal{L}_{ac} = \begin{cases} -\mathbb{E}_{s,a\sim\hat{D}}\left[A \cdot \frac{\nabla\pi_\theta(a|s)}{\pi_b(a|s)}\right] & \text{if } A > 0 \text{ or } 1-\epsilon \leq \frac{\pi_\theta(a|s)}{\pi_b(a|s)} \leq 1+\epsilon \\ 0 & \text{otherwise} \end{cases}$$

   For positive-advantage samples, unconstrained updates via the importance sampling ratio are applied directly, maximizing learning efficiency from successful experiences. For negative-advantage samples, updates are permitted only when the importance sampling ratio falls within $[1-\epsilon, 1+\epsilon]$; otherwise, the update is skipped entirely. This design augments standard PPO with a two-sided constraint — whereas PPO clips only the lower bound of the ratio, SoLS also clips the upper bound, preventing excessive policy drift induced by negative samples.

2. **Success Transition Replay (STR)**

   To address the high cost of trajectory generation and extremely low success rates in open-world environments, STR uses a hash table to map each task to its list of successful action transitions. Up to 50 recent successful time steps are retained per task, and during training, $n$ successful transitions are sampled from each task and mixed with online data:

   $$\mathcal{D} = \bigcup_{t \in \text{tasks}} \text{sample}(\mathcal{D}_{\text{STR}}(t), n) \cup \mathcal{D}_{on}$$

   STR bridges the SFT distribution and the target distribution by bootstrapping learning from sporadic successes, preventing the waste of scarce successful experiences.

3. **Value Function Design**

   The value network is implemented by appending an affine layer with sigmoid activation on top of the Transformer's final hidden layer, trained with a Monte Carlo target to avoid the need for a separate target network:

   $$\mathcal{L}_{cr} = \mathbb{E}_{R,s\sim\mathcal{D}}\left[(R - V_\phi^{\pi_\theta}(s))^2\right]$$

### Loss & Training

The actor and critic losses are jointly optimized: $\mathcal{L} = \mathcal{L}_{ac} + \lambda \cdot \mathcal{L}_{cr}$. Training uses a data-parallel architecture, with each parallel process maintaining an independent STR instance. The RL environment is the AndroidWorld simulator, with each step taking approximately 4–5 seconds.

## Key Experimental Results

### Main Results

| Method | Input Type | Easy | Medium | Hard | Overall |
|--------|-----------|------|--------|------|---------|
| SeeAct (GPT-4o) | screen+UI tree | 36.1 | 17.9 | 0.0 | 22.5 |
| T3A (GPT-4o) | UI tree | 66.7 | 21.4 | 12.5 | 40.0 |
| GPT-4o+UGround-7B | screen | 69.4 | 28.6 | 12.5 | 43.8 |
| GPT-4o+AriaUI | screen+UI tree | 66.7 | 28.6 | 6.3 | 41.3 |
| SFT | UI tree | 38.9 | 9.5 | 4.2 | 22.1±2.7 |
| PPO | UI tree | 53.7 | 8.3 | 6.3 | 28.3±0.7 |
| DigiRL-STR | UI tree | 55.6 | 32.1 | 12.5 | 38.8±0.0 |
| **SoLS-STR** | UI tree | **68.5** | **40.5** | **16.6** | **51.3±1.2** |

### Ablation Study

| Configuration | Success Rate | Notes |
|---------------|-------------|-------|
| SoLS (w/o STR) | ~38.8% | On par with DigiRL-STR |
| SoLS-STR | 51.3% | STR yields substantial gains |
| A2C-STR | 32.1% | Without asymmetric updates; 60% relative degradation |
| PPO-STR | ~35% | Off-policy PPO variant |
| DigiRL (w/o STR) | ~30% | STR also benefits DigiRL |

### Key Findings

1. SoLS-STR achieves a 32.5% relative improvement over DigiRL-STR, surpassing all GPT-4o-based methods.
2. Asymmetric updates contribute the most: the 60% relative gain of SoLS-STR over A2C-STR directly validates the core hypothesis.
3. Inference speed is approximately 0.9 seconds per step, 60× faster than UGround-7B.
4. Gains on Medium-difficulty tasks are most pronounced (40.5% vs. 32.1%), indicating that RL training facilitates acquisition of in-domain knowledge.
5. Categories such as Files, Maps, and Markor exhibit the largest success rate improvements over the course of training.

## Highlights & Insights

1. **Precise problem analysis**: The paper explains performance degradation in standard RL fine-tuning of foundation models through the lens of token probability redistribution induced by negative-advantage gradients — an insight with broad applicability.
2. **Simple yet effective design**: Only minor modifications to PPO's clipping scheme (two-sided constraints with no clipping on positive samples) yield substantial performance gains.
3. **Strong practical utility**: An 8B model completes tasks with a single forward pass, achieving 0.9-second latency — far superior to multi-step prompt engineering pipelines.

## Limitations & Future Work

- The asymmetric constraint may cause premature convergence to local optima.
- The approach is strongly dependent on the quality of the SFT initialization policy.
- It cannot handle tasks requiring visual input, long-term memory, or entirely unseen interaction patterns.
- Policy oscillation may occur in highly stochastic environments.

## Related Work & Insights

- **DigiRL**: Online RL for mobile app control using rejection sampling to avoid negative updates.
- **GRPO (DeepSeek-R1)**: A critic-free RL algorithm requiring 64 responses per prompt, with far greater overhead than SoLS.
- **WoLF**: The naming inspiration; adjusts learning rates based on win/loss outcomes.

## Rating

- Novelty: ⭐⭐⭐⭐☆ — The asymmetric update idea offers deep insight without excessive complexity.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated on the realistic AndroidWorld environment with comprehensive multi-baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐☆ — Clear structure with well-articulated motivation.
- Value: ⭐⭐⭐⭐⭐ — A small model with RL outperforming GPT-4o carries significant practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model](bootstrap_off-policy_with_world_model.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](bootstrap_off-policy_with_world_model.md)
- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)
- [\[NeurIPS 2025\] Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning](finite-sample_analysis_of_policy_evaluation_for_robust_average_reward_reinforcem.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](../../ICLR2026/reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)

</div>

<!-- RELATED:END -->
