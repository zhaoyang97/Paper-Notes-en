---
title: >-
  [Paper Note] Reinforcement Learning with Action Chunking
description: >-
  [NeurIPS 2025][Reinforcement Learning][Action chunking] This paper proposes Q-chunking, which extends action chunking from imitation learning to TD-based reinforcement learning by running RL directly over a "chunked" action space, thereby improving exploration and sample efficiency in long-horizon sparse-reward tasks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Action chunking
  - Q-learning
  - offline-to-online RL
  - sparse rewards
  - manipulation tasks
date: 2026-05-08
content_hash: 1bc2ff6047ded700
---

# Reinforcement Learning with Action Chunking

**Conference**: NeurIPS 2025

**arXiv**: [2507.07969](https://arxiv.org/abs/2507.07969)

**Code**: None

**Area**: Reinforcement Learning

**Keywords**: Action chunking, Q-learning, offline-to-online RL, sparse rewards, manipulation tasks

## TL;DR

This paper proposes Q-chunking, which extends action chunking from imitation learning to TD-based reinforcement learning by running RL directly over a "chunked" action space, thereby improving exploration and sample efficiency in long-horizon sparse-reward tasks.

## Background & Motivation

In the offline-to-online RL setting, a central challenge is how to leverage offline prior data to maximize sample efficiency during online learning. Action chunking — predicting a sequence of future actions rather than a single-step action — is a technique widely used in imitation learning.

Key motivations:

**Exploration difficulty**: In long-horizon sparse-reward tasks, random exploration is unlikely to reach the goal.

**Insufficient use of offline data**: Existing methods are not effective at obtaining good exploration policies from offline data.

**Untapped potential of action chunking in RL**: Methods such as ACT have demonstrated the value of action chunking in imitation learning, yet its application in TD learning has not been systematically studied.

**Temporal consistency**: Predicting actions step by step leads to temporally inconsistent behavior, which is detrimental to tasks such as robotic manipulation.

## Method

### Overall Architecture

Q-chunking extends the action space from single-step actions $a_t$ to action sequences $\mathbf{a}_t = (a_t, a_{t+1}, \ldots, a_{t+H-1})$, and runs Q-learning directly over this chunked space.

### Key Designs

**1. Chunked Action Space**

- $H$ consecutive actions are packed into a single "macro-action": $\mathbf{a} = (a_0, a_1, \ldots, a_{H-1})$
- The Q-function is defined over chunked actions: $Q(s, \mathbf{a})$
- The policy outputs chunked actions: $\pi(\mathbf{a} | s)$

**2. Unbiased $n$-step Returns**

- Chunked actions naturally correspond to $H$-step TD targets:
$$Q(s_t, \mathbf{a}_t) \leftarrow \sum_{k=0}^{H-1} \gamma^k r_{t+k} + \gamma^H \max_{\mathbf{a}'} Q(s_{t+H}, \mathbf{a}')$$
- Unlike standard multi-step returns, the $H$-step return here is **unbiased** because the action sequence is executed in full, avoiding the bias introduced by importance sampling or truncation.

**3. Offline-to-Online Transition**

- Offline phase: the Q-function and policy are trained on action chunks from offline data.
- Online phase: temporally consistent behavioral patterns learned from offline data are leveraged for exploration.
- Key insight: chunked actions from offline data provide a more temporally consistent exploration policy.

### Loss & Training

- Offline phase: conservative Q-learning in the CQL style applied to the chunked action space.
- Online phase: online fine-tuning in the SAC/TD3 style.
- Chunk size $H$: treated as a hyperparameter, typically set to 5–20.

## Key Experimental Results

### Main Results

Long-horizon manipulation tasks (normalized success rate, 100K online steps):

| Method | Nut Assembly | Pick-Place | Stack | Can | Average |
|--------|-------------|-----------|-------|-----|---------|
| CQL → SAC | 12% | 25% | 8% | 35% | 20.0% |
| IQL → SAC | 18% | 32% | 12% | 42% | 26.0% |
| Cal-QL | 22% | 38% | 15% | 48% | 30.8% |
| RLPD | 28% | 42% | 18% | 52% | 35.0% |
| Q-chunking (Ours) | **45%** | **62%** | **35%** | **72%** | **53.5%** |

Offline-only performance comparison:

| Method | Nut Assembly | Pick-Place | Stack | Can |
|--------|-------------|-----------|-------|-----|
| CQL | 10% | 22% | 6% | 30% |
| IQL | 15% | 28% | 10% | 38% |
| Q-chunking (offline) | **25%** | **40%** | **18%** | **52%** |

### Ablation Study

Effect of chunk size $H$ (Nut Assembly success rate):

| H | Offline | Online 100K | Online 500K |
|---|---------|-------------|-------------|
| 1 (no chunking) | 10% | 18% | 35% |
| 5 | 18% | 35% | 55% |
| 10 | 25% | **45%** | **68%** |
| 20 | 22% | 42% | 65% |
| 50 | 15% | 30% | 50% |

### Key Findings

1. Q-chunking improves online sample efficiency by approximately 50% over the best baseline (53.5% vs. 35.0%).
2. Chunked actions already yield a better initial policy during the offline phase.
3. The optimal chunk size is approximately 10; excessively long chunks reduce adaptability.
4. Temporally consistent exploration is the key driver of improvement — chunking eliminates the "jittery" exploratory behavior characteristic of step-wise policies.

## Highlights & Insights

- **Simple yet effective**: The core modification is merely a redefinition of the action space, introducing no new losses or architectural components.
- **Dual benefits**: The approach obtains better initialization from offline data while simultaneously achieving better exploration during online fine-tuning.
- **Unbiased multi-step returns**: Chunked actions naturally provide unbiased $n$-step returns, circumventing the bias issues of conventional multi-step methods.

## Limitations & Future Work

1. Chunk size $H$ is a critical hyperparameter that requires task-specific tuning.
2. In highly dynamic tasks demanding rapid reactions, chunking may reduce responsiveness.
3. High-dimensional chunked action spaces increase the difficulty of learning the Q-function.
4. Validation on physical robots has not yet been conducted.

## Related Work & Insights

- **ACT** (Zhao et al.): Pioneering work on action chunking in imitation learning.
- **Cal-QL, RLPD**: Foundational methods for offline-to-online RL.
- **Temporal Abstraction**: Options and macro-actions in hierarchical RL.

## Rating

- ⭐ Novelty: 8/10 — Extending action chunking to TD learning is a natural and effective idea.
- ⭐ Value: 8/10 — Directly relevant to practical tasks such as robotic manipulation.
- ⭐ Writing Quality: 8/10 — 36 pages with thorough experiments and in-depth analysis.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning](beast_efficient_tokenization_of_b-splines_encoded_action_sequences_for_imitation.md)
- [\[NeurIPS 2025\] Learning Human-Like RL Agents through Trajectory Optimization with Action Quantization](learning_human-like_rl_agents_through_trajectory_optimization_with_action_quanti.md)
- [\[NeurIPS 2025\] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies](act_to_see_see_to_act_diffusion-driven_perception-action_interplay_for_adaptive_.md)
- [\[AAAI 2026\] Object-Centric Latent Action Learning](../../AAAI2026/reinforcement_learning/object-centric_latent_action_learning.md)
- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](risk-averse_total-reward_reinforcement_learning.md)

<!-- RELATED:END -->
