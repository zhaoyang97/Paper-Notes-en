---
title: >-
  [Paper Note] Targeted Exploration via Unified Entropy Control for Reinforcement Learning
description: >-
  [ACL 2026 Findings][Reinforcement Learning][Entropy Control] This paper proposes UEC-RL, a unified bidirectional entropy control framework that solves the prevalent issues of entropy collapse and training instability in…
tags:
  - "ACL 2026 Findings"
  - "Reinforcement Learning"
  - "Entropy Control"
  - "GRPO"
  - "Exploration Strategy"
  - "Reasoning Enhancement"
date: 2026-05-08
content_hash: 13f2c62c03e6879e
---

# Targeted Exploration via Unified Entropy Control for Reinforcement Learning

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.14646](https://arxiv.org/abs/2604.14646)  
**Code**: [GitHub](https://github.com/597358816/UEC-RL)  
**Area**: Multimodal VLM  
**Keywords**: Entropy Control, GRPO, Exploration Strategy, Reinforcement Learning, Reasoning Enhancement

## TL;DR

This paper proposes UEC-RL, a unified bidirectional entropy control framework that solves the prevalent issues of entropy collapse and training instability in GRPO. It achieves this by performing high-temperature targeted exploration (increasing entropy) on difficult prompts and stabilizing high-quality trajectories via an experience replay stabilizer (decreasing entropy), resulting in a 37.9% relative gain on Geometry3K.

## Background & Motivation

**Background**: Reinforcement Learning (RL) has become the core paradigm for LLM/VLM post-training. GRPO is widely adopted as a lightweight alternative to PPO—removing the critic network and estimating advantages via group-relative rewards, which offers high computational efficiency and competitive reasoning performance.

**Limitations of Prior Work**: GRPO faces two prominent issues in complex reasoning tasks: (1) **Entropy collapse**—policy entropy drops rapidly, causing the model to converge prematurely to low-diversity behaviors and fail to discover low-probability but valuable reasoning paths; (2) **Training instability**—in complex scenarios like multimodal reasoning, the correctness of sampled outputs varies significantly, and group-normalized rewards fail to provide sufficient variance reduction, leading to fragile gradient updates.

**Key Challenge**: GRPO lacks a bidirectional entropy regulation mechanism—it can neither actively increase entropy to enhance exploration nor stabilize entropy in high-variance environments to ensure convergence. Existing remedies either introduce update variance (DAPO's clip-higher) or optimization bias (entropy rewards/entropy shaping optimize entropy-related objectives rather than task rewards).

**Goal**: Design a unified framework to increase entropy for deep exploration when needed and decrease entropy to ensure convergence when exploration becomes ineffective.

**Key Insight**: Based on the entropy change theorem of Natural Policy Gradient—policy updates increase entropy when high-advantage actions have low probability under the current policy (negative covariance) and decrease it otherwise. Therefore, the negative covariance effect can be amplified by increasing sampling temperature, selectively enhancing exploration for difficult prompts.

**Core Idea**: Implement bidirectional entropy control through two synergistic components: a targeted exploration mechanism (sampling with elevated temperature for hard prompts + filtering valuable trajectories) and a controllable entropy stabilizer (repeatedly reinforcing high-quality trajectories via experience replay to reduce entropy), dynamically balancing exploration and exploitation throughout training.

## Method

### Overall Architecture

UEC-RL is built upon GRPO. For each batch of prompts, $G$ trajectories are first sampled using a standard temperature. If all $G$ trajectories for a specific prompt yield zero reward (marked as "difficult"), an additional $G'$ trajectories are sampled using an increased temperature $t' > 1$. Valuable trajectories are filtered for gradient updates, while high-quality trajectories are stored in a replay buffer. Regular replay from the buffer is performed to stabilize training.

### Key Designs

1. **Targeted Exploration Mechanism**:

    - **Function**: Adaptively expand the exploration space for difficult prompts to discover low-probability but valuable reasoning trajectories.
    - **Mechanism**: If all $G$ trajectories from standard sampling fail ($\max_i R_i = 0$), an additional $G'$ trajectories are sampled using a softened distribution (temperature $t' > 1$). Only two types of trajectories are retained: regular samples $O_R$ (non-zero advantage) and exploration samples $O_H$ (trajectories with positive advantage from extended sampling), filtering out low-advantage noisy samples.
    - **Design Motivation**: Based on the entropy change theorem, increasing temperature $t'$ narrows the gap between high and low probability actions, making negative covariance more likely and thus controlledly increasing policy entropy. Crucially, this is only activated for difficult prompts, leaving simple prompts unaffected.

2. **Controllable Entropy Stabilizer (Experience Replay)**:

    - **Function**: Prevent uncontrolled entropy growth and guide the policy toward stable convergence by repeatedly reinforcing high-quality trajectories.
    - **Mechanism**: Positive advantage trajectories (advantage > threshold $A_0$) discovered during exploration are stored in a fixed-size replay buffer $\mathcal{B}_{replay}$, keeping only the latest $s'$ trajectories. A replay update is performed every $f_{replay}$ steps by sampling from the buffer. Repeatedly reinforcing high-advantage trajectories shifts probability mass toward correct reasoning patterns, which, according to Theorem 4.2, produces positive covariance and reduces entropy.
    - **Design Motivation**: High-quality trajectories discovered through exploration initially have very low probabilities, and using them once has limited impact. Replay strengthens their gradient signals and produces a stabilizing entropy-reduction effect, enabling a natural transition from exploration to convergence.

3. **Selective Trajectory Retention**:

    - **Function**: Filter noisy samples, retaining only trajectories valuable for optimization.
    - **Mechanism**: For regular samples, all trajectories with non-zero advantage ($\hat{A}_{i,t} \neq 0$) are kept. For exploration samples, only those with positive advantage ($\hat{A}_{i,t} \geq 0$) are retained. Low-advantage exploration samples introduce noise gradients that hinder optimization and generalization.
    - **Design Motivation**: Effective exploration should emphasize informative and high-quality diversity rather than indiscriminately encouraging randomness.

### Loss & Training

Based on the GRPO objective function, the framework uses a clipped surrogate objective + KL divergence regularization. Hyperparameters include exploration temperature $t'$, exploration group size $G'$, replay size $s'$, and replay frequency $f_{replay}$.

## Key Experimental Results

### Main Results (Text Reasoning, Qwen2.5-math-7B)

| Method | AIME24 | MATH | GSM8K | MMLU | Average |
|--------|--------|------|-------|------|------|
| GRPO | 25.8 | 77.6 | 87.1 | 45.0 | 50.34 |
| DAPO | 24.3 | 78.3 | 87.6 | 48.5 | 51.77 |
| **Ours** | **28.5** | **80.4** | **87.9** | **50.2** | **53.62** |

Geometry3K Multimodal Reasoning (Qwen2.5-VL-7B):

| Method | Accuracy | vs Ours |
|--------|--------|-----------|
| Baseline | 38.44 | -16.97 |
| GRPO | 50.75 | -4.66 |
| **Ours** | **55.41** | - |

### Ablation Study

| Configuration | Geometry3K | Description |
|------|-----------|------|
| UEC-RL (full) | 55.41 | Full model |
| w/o Exploration | ~50 | Drops to GRPO level |
| w/o Replay | ~52 | Training becomes unstable |
| w/o Selective Retention | ~51 | Noisy samples affect optimization |

### Key Findings
- UEC-RL consistently outperforms all RL baselines in both text and multimodal reasoning, with average gains of +2.88% (Qwen2.5-math-7B) and +1.07% (VLM).
- It achieves a 37.9% relative gain on Geometry3K (38.44 $\to$ 55.41) while maintaining higher training efficiency (step time is only 0.79$\times$ that of GRPO).
- Pass@k evaluations show UEC-RL consistently leading, indicating not only higher top-1 accuracy but also superior generation diversity.
- Both exploration and stabilization components are necessary: neither can achieve optimal results individually.

## Highlights & Insights
- **The concept of bidirectional entropy control** is ingenious: instead of merely "slowing entropy decay" (like DAPO), it actively increases entropy when needed and decreases it for convergence. This theoretical guidance based on the entropy change theorem makes the design more principled.
- **"Difficult prompt detection $\to$ targeted exploration"** is a practical strategy: detecting difficulty costs nearly zero (all samples fail = difficult), and compute budget is only increased for difficult samples, leading to high resource efficiency.
- **Application of experience replay in LLM RL**: Introducing the replay concept from classical RL into GRPO training not only improves sample efficiency but also provides a theoretically guaranteed entropy stabilization effect through repeated reinforcement.

## Limitations & Future Work
- Validated only on 7B/8B scale models; effectiveness on larger models remains unknown.
- Hyperparameters such as exploration temperature $t'$ and replay frequency $f_{replay}$ require tuning.
- Difficult prompt detection (all samples fail) is binary and might miss "partially difficult" samples.
- Theoretical analysis is based on an approximation of Natural Policy Gradient, leaving a gap between theory and the actual clipped objective.
- The replay buffer might introduce distribution shift issues, though keeping only the latest trajectories mitigates this.

## Related Work & Insights
- **vs GRPO**: GRPO lacks entropy regulation; UEC-RL adds bidirectional entropy control. Both use the same base objective function.
- **vs DAPO**: DAPO slows entropy decay via clip-higher but introduces update variance; UEC-RL controls entropy more precisely through targeted exploration and an experience replay stabilizer.
- **vs Entropy Reward/KL-cov**: These methods introduce bias by optimizing entropy-related terms; UEC-RL indirectly influences entropy by adjusting sampling strategies without changing the optimization target.

## Rating
- Novelty: ⭐⭐⭐⭐ Clear concept of a bidirectional entropy control framework with solid theoretical support.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers text + multimodal, multiple benchmarks, Pass@1, and Pass@k.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with coherent theoretical derivation and experimental analysis.
- Value: ⭐⭐⭐⭐ Practical guidance for improving GRPO training; open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](../../AAAI2026/reinforcement_learning/reasoning_with_exploration_an_entropy_perspective.md)
- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ICLR 2026\] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](../../ICLR2026/reinforcement_learning/exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)
- [\[ICLR 2026\] Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](../../ICLR2026/reinforcement_learning/entropy-preserving_reinforcement_learning.md)
- [\[ACL 2026\] NaviMaster: Learning a Unified Policy for GUI and Embodied Navigation Tasks](navimaster_learning_a_unified_policy_for_gui_and_embodied_navigation_tasks.md)

</div>

<!-- RELATED:END -->
