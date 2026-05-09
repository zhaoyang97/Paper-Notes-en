---
title: >-
  [Paper Note] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][meta-RL] This paper proposes SISL (Self-Improving Skill Learning), which decouples the high-level exploitation policy from a dedicated skill improvement policy, and incorporates a maximum return relabeling mechanism for skill prioritization. SISL achieves robust skill learning under noisy offline demonstration data and substantially improves the performance of skill-based meta-reinforcement learning on long-horizon tasks.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - meta-RL
  - skill learning
  - noisy demonstrations
  - self-improvement
  - maximum return relabeling
date: 2026-05-08
content_hash: c44042256bd8c998
---

# Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2502.03752](https://arxiv.org/abs/2502.03752)
**Code**: [github.com/epsilog/SISL](https://github.com/epsilog/SISL)
**Area**: Reinforcement Learning / Meta-Learning / Skill Learning
**Keywords**: meta-RL, skill learning, noisy demonstrations, self-improvement, maximum return relabeling

## TL;DR

This paper proposes SISL (Self-Improving Skill Learning), which decouples the high-level exploitation policy from a dedicated skill improvement policy, and incorporates a maximum return relabeling mechanism for skill prioritization. SISL achieves robust skill learning under noisy offline demonstration data and substantially improves the performance of skill-based meta-reinforcement learning on long-horizon tasks.

## Background & Motivation

**State of the Field**: Skill-based meta-RL methods (e.g., SiMPL) decompose long state-action sequences into reusable skills and achieve success on long-horizon tasks through hierarchical decision-making. These methods rely on offline demonstration data to learn a low-level skill library, which is then leveraged by a high-level policy to select skills online.

**Limitations of Prior Work**: Existing methods are highly dependent on high-quality offline demonstrations; however, real-world data are often corrupted by factors such as hardware degradation, environmental perturbations, and sensor drift. When offline data quality degrades, the learned skill library becomes contaminated, and this degradation propagates to the high-level policy, ultimately impairing adaptation performance.

**Root Cause**: Existing methods treat all trajectories uniformly (uniform sampling), causing low-quality samples to dominate skill learning. For example, skills learned from noisy data in the Kitchen microwave-opening task fail to even complete the grasping motion.

**Starting Point**: The paper designs a self-improvement mechanism by decoupling the high-level exploitation policy from an independent skill improvement policy. The improvement policy explores superior behaviors in the vicinity of the offline data distribution, while a return relabeling scheme prioritizes high-value trajectories.

## Method

### Overall Architecture

SISL consists of two alternating phases:
1. **Decoupled Policy Learning**: The high-level policy $\pi_h$ maximizes returns using the current skill library; the skill improvement policy $\pi_{\text{imp}}$ explores for superior behaviors in the neighborhood of the offline data distribution.
2. **Skill Learning**: Every $K_{\text{iter}}$ iterations, the skill encoder $q$, skill prior $p$, and low-level policy $\pi_l$ are retrained using high-quality data.

### Key Design 1: Decoupled Skill Self-Improvement

The training objective of the skill improvement policy $\pi_{\text{imp}}$ combines an RL loss with a KL divergence constraint:

$$\sum_i \mathbb{E}_{\tau^i \sim \mathcal{B}_{\text{imp}}^i \cup \mathcal{B}_{\text{on}}^i} [\mathcal{L}_{\text{imp}}^{\text{RL}}(\pi_{\text{imp}})] + \lambda_{\text{imp}}^{\text{kld}} \mathbb{E}_{\tau^i \sim \mathcal{B}_{\text{on}}^i} \mathcal{D}_{\text{KL}}(\hat{\pi}_d^i \| \pi_{\text{imp}})$$

The prioritized online buffer $\mathcal{B}_{\text{on}}^i$ retains high-return trajectories, providing both a self-supervised signal for $\pi_{\text{imp}}$ and high-quality samples for skill refinement.

### Key Design 2: Skill Prioritization via Maximum Return Relabeling

A reward model $\hat{R}(s_t, a_t, i)$ is trained to compute the maximum hypothetical return across tasks for each offline trajectory:

$$\hat{G}(\tilde{\tau}) = \max_i \left\{ \sum_t \gamma^t \hat{R}(s_t, a_t, i) \right\}$$

Offline data are sampled according to a softmax distribution $P_{\mathcal{B}_{\text{off}}}(\tilde{\tau}) = \text{Softmax}(\hat{G}(\tilde{\tau}) / T)$, which suppresses noisy samples.

### Loss & Training

The final skill learning objective dynamically mixes offline and online data:

$$\mathcal{L}_{\text{skill}} = (1 - \beta) \mathbb{E}_{\tilde{\tau} \sim P_{\mathcal{B}_{\text{off}}}} [\mathcal{L}(\pi_l, q, p, z)] + \frac{\beta}{N_{\mathcal{T}}} \sum_i \mathbb{E}_{\tau^i \sim \mathcal{B}_{\text{on}}^i} [\mathcal{L}(\pi_l, q, p, z)]$$

The mixing coefficient $\beta$ is computed adaptively based on the average returns of online and offline data:

$$\beta = \frac{\exp(\bar{G}_{\text{on}} / T)}{\exp(\bar{G}_{\text{on}} / T) + \exp(\bar{G}_{\text{off}} / T)}$$

## Key Experimental Results

### Main Results: Final Test Average Return across Four Long-Horizon Environments

| Environment (Noise) | SAC | PEARL | SPiRL | SiMPL | **SISL** |
|---------------------|-----|-------|-------|-------|----------|
| Kitchen (Expert) | 0.01 | 0.23 | 3.11 | 3.40 | **3.97** |
| Kitchen (σ=0.2) | - | - | 2.06 | 2.18 | **3.73** |
| Kitchen (σ=0.3) | - | - | 0.83 | 0.81 | **3.48** |
| Office (Expert) | 0.00 | 0.01 | 0.65 | 2.50 | **2.86** |
| Office (σ=0.3) | - | - | 0.42 | 0.11 | **1.68** |
| Maze2D (Expert) | 0.20 | 0.10 | 0.77 | 0.80 | **0.87** |
| Maze2D (σ=1.5) | - | - | 0.81 | 0.68 | **0.99** |
| AntMaze (Expert) | 0.00 | 0.00 | 0.64 | 0.67 | **0.81** |

### Ablation Study: Contribution of Each Component (Kitchen σ=0.3)

| Variant | Final Return |
|---------|-------------|
| **SISL (Full)** | **3.48** |
| w/o $\mathcal{B}_{\text{off}}$ | Significant drop |
| w/o $P_{\mathcal{B}_{\text{off}}}$ (uniform sampling) | Notable drop |
| w/o $\mathcal{B}_{\text{on}}$ | Significant drop |
| w/o $\pi_{\text{imp}}$ | Notable drop |

### Key Findings

- SPiRL and SiMPL suffer sharp performance degradation as noise increases, whereas SISL remains robust across all noise levels.
- At Kitchen σ=0.3, SiMPL achieves a return of only 0.81, while SISL reaches 3.48 (a 4.3× improvement).
- At Maze2D σ=1.5, SISL achieves a near-perfect success rate of 0.99.
- SISL incurs only approximately 16% additional training computation, with no change to meta-test cost.

## Highlights & Insights

1. **Insightful Problem Identification**: This work is the first to systematically identify the propagation chain from skill library contamination by noise to high-level policy degradation.
2. **Decoupled Design**: The high-level policy is responsible for exploitation, while the improvement policy handles exploration, avoiding conflicts between the two objectives.
3. **Adaptive Mixing Coefficient**: $\beta$ dynamically adjusts based on the relative quality of online and offline data, forming an automatic curriculum learning scheme.
4. **Lightweight Enhancement**: Only a 16% increase in additional computation is introduced, the meta-test procedure remains unchanged, and the approach is easily integrated into existing frameworks.

## Limitations & Future Work

- Meta-test still requires fine-tuning (0.5K iterations); zero-shot skill transfer remains an important direction for future improvement.
- The reward model relies on simple subtask completion rewards and may require per-task normalization in settings with more complex reward functions.
- The temperature parameter $T$ requires environment-specific tuning (1.0 for Kitchen, 0.5 for Maze2D); ideally, this should be determined adaptively.
- Evaluation is limited to four simulated environments; validation on real robotic systems is still lacking.

## Related Work & Insights

- SPiRL/SiMPL serve as direct baselines; augmenting their framework with a self-improvement mechanism is a natural extension.
- Distinction from offline-to-online RL: the latter assumes reward-labeled offline data for pretraining, whereas SISL requires only reward-free offline data for skill learning.
- The maximum return relabeling idea is generalizable to other scenarios requiring learning from low-quality data (e.g., learning from human demonstrations).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of a decoupled improvement policy and return relabeling is both effective and novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four environments × multiple noise levels, with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — Addresses the critical practical problem of uncontrollable data quality in real-world settings.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SUSD: Structured Unsupervised Skill Discovery through State Factorization](susd_structured_unsupervised_skill_discovery_through_state_factorization.md)
- [\[ICLR 2026\] AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification](amped_adaptive_multi-objective_projection_for_balancing_exploration_and_skill_di.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)
- [\[ICLR 2026\] Understanding and Improving Hyperbolic Deep Reinforcement Learning](understanding_and_improving_hyperbolic_deep_reinforcement_learning.md)
- [\[NeurIPS 2025\] Periodic Skill Discovery](../../NeurIPS2025/reinforcement_learning/periodic_skill_discovery.md)

<!-- RELATED:END -->
