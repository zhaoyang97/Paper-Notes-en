---
title: >-
  [Paper Note] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization
description: >-
  [NeurIPS 2025][Self-Supervised Learning][self-supervised reinforcement learning] To address policy collapse and entropy collapse in self-supervised reinforcement learning for LLMs, this paper proposes M-GRPO, a momentum-anchored GRPO framework combined with an IQR-based low-entropy trajectory filtering method, achieving stable training and state-of-the-art performance.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - self-supervised reinforcement learning
  - policy collapse
  - momentum anchoring
  - GRPO
  - entropy filtering
date: 2026-05-08
content_hash: c24d733371c6c267
---

# M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2512.13070](https://arxiv.org/abs/2512.13070)
**Code**: [GitHub](https://github.com/Bai-Bizhe/M_GRPO)
**Area**: Self-Supervised Learning
**Keywords**: self-supervised reinforcement learning, policy collapse, momentum anchoring, GRPO, entropy filtering

## TL;DR

To address policy collapse and entropy collapse in self-supervised reinforcement learning for LLMs, this paper proposes M-GRPO, a momentum-anchored GRPO framework combined with an IQR-based low-entropy trajectory filtering method, achieving stable training and state-of-the-art performance.

## Background & Motivation

**Background**: Reinforcement learning with verifiable rewards (RLVR) has become a core paradigm for post-training LLMs to enhance reasoning capabilities. However, it relies heavily on large-scale human-annotated data and reward model infrastructure, making it costly and domain-restricted. This has motivated exploration of self-supervised or label-free RL signals, such as self-consistency and self-certainty as reward signals (e.g., SRT, TTRL, Intuitor).

**Limitations of Prior Work**: Self-supervised RLVR (SS-RLVR) suffers from a critical **policy collapse** phenomenon during extended training — training rewards first increase and then sharply decline, accompanied by simultaneous degradation in validation accuracy. This is further compounded by **entropy collapse**, where the model becomes overconfident prematurely.

**Key Challenge**: Increasing the number of rollouts can only delay, not prevent, collapse. The self-reward mechanism is inherently unstable, lacking a stable training objective.

**Goal**: To stabilize the training process of SS-RLVR without relying on ground-truth labels, thereby preventing policy collapse and entropy collapse.

**Key Insight**: Drawing on the success of momentum encoders in contrastive learning (MoCo), this work introduces the momentum mechanism into policy optimization.

**Core Idea**: A slowly evolving momentum model provides stable pseudo-label training targets, while IQR-based filtering of low-entropy trajectories preserves policy diversity.

## Method

### Overall Architecture

M-GRPO consists of two key components:
1. **Momentum-anchored self-supervised RL framework**: A dual-model setup (current policy $\pi_{\theta_q}$ + momentum model $\pi_{\theta_k}$) jointly generates rollouts and produces pseudo-labels via majority voting.
2. **IQR adaptive entropy filter**: Dynamically prunes low-entropy trajectories to prevent premature policy convergence.

### Key Designs

1. **Momentum Model Update Mechanism**:

    - The momentum model parameters $\theta_k$ are not updated via backpropagation; instead, they are the exponential moving average of the current policy parameters:
    $\pi_{\theta_k} \leftarrow m \cdot \pi_{\theta_k} + (1-m) \cdot \pi_{\theta_q}$
    - $m \in [0,1)$ is the momentum coefficient (e.g., 0.99), ensuring the momentum model evolves slowly and provides a stable reference.
    - Inspired by the MoCo series of contrastive learning, this work is the first to transfer the momentum-contrastive idea to RL policy optimization.

2. **Joint Rollout and Majority Voting**:

    - The current policy generates $M$ responses $\{y_i^q\}_{i=1}^M$, and the momentum model generates $N$ responses $\{y_j^k\}_{j=1}^N$.
    - These are merged into a pool of size $G = M + N$.
    - Majority voting selects the pseudo ground truth $y_v$: the response with the highest answer consistency.
    - Including the momentum model's rollouts in the voting pool is critical — it reduces the noise inherent in pseudo-labels generated solely by the rapidly changing current policy.

3. **Normalized Advantage Estimation**: Binary rewards are computed for the current policy's $M$ rollouts based on the pseudo ground truth (consistent = 1, inconsistent = 0), and normalized per prompt following the GRPO framework:
    $\hat{A}_i = \frac{r(y_v, y_i^q) - \text{mean}(\{r(y_v, y_j^q)\}_{j=1}^M)}{\text{std}(\{r(y_v, y_j^q)\}_{j=1}^M)}$

4. **IQR Adaptive Entropy Filtering**:

    - Trajectory-level entropy is computed for each of the $G$ trajectories per input.
    - The interquartile range method is used to detect low-entropy outliers: $T_{IQR} = Q_1 - k \cdot (Q_3 - Q_1)$, with $k=0.75$.
    - Trajectories with entropy below the threshold are pruned.
    - Compared to static thresholds (e.g., removing the bottom 10%), the IQR method adaptively responds to dynamic changes in the entropy distribution throughout training.

### Loss & Training

The learning objective is to maximize the advantage-weighted log-likelihood:
$$\mathcal{J}(\theta_q) = \mathbb{E}_{x \sim D, \{y_i^q\} \sim \pi_{\theta_q}}\left[\sum_{i=1}^M \hat{A}_i \log \pi_{\theta_q}(y_i^q | x)\right]$$

Training details:
- Backbone: Qwen3-4B-Base
- Training data: MATH training set (no ground-truth labels)
- Batch size: 8 questions, 32 rollouts per question, temperature 1.1
- Optimizer: AdamW (lr=1e-6, cosine warmup)
- KL loss coefficient: 0.005
- Momentum model rollout count: $N = G/4$

## Key Experimental Results

### Main Results

M-GRPO performance on Qwen3-4B-Base (trained without ground-truth labels):

| Method | MATH500 | AIME24 | AIME25 | GPQA Dia | GPQA | LiveCode |
|--------|---------|--------|--------|----------|------|----------|
| Base Model | 61.50% | 0.83% | 5.00% | 34.41% | 29.91% | 9.61% |
| SRT_Best (manually selected) | 79.20% | 12.50% | 11.67% | 38.26% | 35.04% | 19.69% |
| SRT_Final (post-collapse) | 47.50% | 7.50% | 8.75% | 28.54% | 25.89% | 16.12% |
| **M-GRPO+IQR_Final** | **79.75%** | **14.58%** | **14.17%** | **39.65%** | 35.49% | **27.12%** |

The final checkpoint of M-GRPO outperforms the best manually selected SRT checkpoint without any human intervention.

### Ablation Study

Rollout scaling analysis (M-GRPO+IQR):

| Config | MATH500 | AIME24 | AIME25 | GPQA Dia | mbpp |
|--------|---------|--------|--------|----------|------|
| G=8 | 77.60% | 11.25% | 10.42% | 39.02% | 68.60% |
| G=16 | 79.75% | 14.43% | 10.00% | 39.65% | 70.40% |
| G=32 | 79.75% | 14.58% | 14.17% | 39.65% | 70.60% |
| G=256 | 79.50% | 16.67% | 14.17% | 40.66% | 70.40% |

Performance improves significantly from G=8 to G=32 and then saturates.

### Key Findings

- SRT eventually collapses under all rollout configurations; increasing rollouts only delays the inevitable.
- M-GRPO maintains training stability across three model scales: Qwen3-1.7B, 4B, and 8B.
- Gains of +5.05% on GPQA and +7.43% on LiveCode relative to SRT_Best.
- The IQR filter effectively maintains higher policy entropy levels, with a slower and more gradual entropy decline.

## Highlights & Insights

- **Clear problem diagnosis**: The paper systematically reveals the phenomena of policy collapse and entropy collapse in SS-RLVR and their causal relationship.
- **Elegant transfer of the momentum mechanism**: The momentum idea from MoCo in visual contrastive learning is transferred to RL policy stabilization with a clear and compelling analogy.
- **Strong practicality**: No manual checkpoint selection is required; training is stable throughout, and the final model is the best model.
- **IQR adaptive filtering**: More robust than fixed thresholds, dynamically adjusting to the entropy distribution as training progresses.

## Limitations & Future Work

- Validation is limited to the MATH dataset; generalization to other self-supervised RL scenarios (e.g., code generation, dialogue) has not been explored.
- The impact of the momentum coefficient $m$ on performance is not discussed in detail.
- Experiments are conducted exclusively on the Qwen3 model family; generalizability to other model families remains unknown.
- The dual-model architecture introduces approximately 25% additional inference overhead due to the momentum model's $N$ rollouts.
- Majority voting assumes that the correct answer is the most frequent one, which may fail on out-of-distribution or difficult samples.

## Related Work & Insights

- **SRT** (Sheikh et al.): A self-training RL method serving as the direct baseline; this paper exposes its collapse problem.
- **GRPO/DAPO**: Group relative policy optimization framework, upon which M-GRPO introduces the momentum mechanism.
- **MoCo** (He et al., 2020): Momentum contrastive learning, the primary source of inspiration for M-GRPO.
- The proposed approach offers insights for other RL scenarios requiring self-supervised signals, such as multimodal reasoning and tool-use learning.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of momentum anchoring and IQR filtering is concise and effective, with in-depth problem analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark, multi-scale validation with comprehensive scaling analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ Addresses a critical practical barrier in SS-RLVR, with significant implications for self-evolving LLM training.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[NeurIPS 2025\] Continuous Subspace Optimization for Continual Learning (CoSO)](continuous_subspace_optimization_for_continual_learning.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)

<!-- RELATED:END -->
