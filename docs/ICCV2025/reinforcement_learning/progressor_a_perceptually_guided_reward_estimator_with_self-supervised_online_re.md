---
title: >-
  [Paper Note] Progressor: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement
description: >-
  [ICCV 2025][Reinforcement Learning][Visual reward learning] This paper proposes Progressor, a framework that learns task-agnostic reward functions from unannotated videos via self-supervision. It provides dense reward signals by predicting task progress distributions and addresses distribution shift during online RL training through an adversarial push-back strategy.
tags:
  - ICCV 2025
  - Reinforcement Learning
  - Visual reward learning
  - self-supervised
  - goal-conditioned RL
  - adversarial online refinement
  - task progress estimation
date: 2026-05-08
content_hash: 9451e607de041b76
---

# Progressor: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement

**Conference**: ICCV 2025
**arXiv**: [2411.17764](https://arxiv.org/abs/2411.17764)
**Code**: [https://ripl.github.io/progressor](https://ripl.github.io/progressor)
**Area**: Reinforcement Learning / Robot Learning
**Keywords**: Visual reward learning, self-supervised, goal-conditioned RL, adversarial online refinement, task progress estimation

## TL;DR

This paper proposes Progressor, a framework that learns task-agnostic reward functions from unannotated videos via self-supervision. It provides dense reward signals by predicting task progress distributions and addresses distribution shift during online RL training through an adversarial push-back strategy.

## Background & Motivation

Designing reward functions remains a central challenge in practical reinforcement learning: manually crafted dense rewards are costly and may induce unintended behaviors, while sparse rewards suffer from poor sample efficiency. Learning reward functions from unannotated videos is a promising alternative, yet existing methods exhibit notable limitations:

**Temporal Contrastive Networks (TCN)** are sensitive to frame rate and assume reward symmetry.

**Generative methods (VIPER, Diffusion Reward)** require expensive generative processes for reward estimation.

**Rank2Reward** requires training separate models per task and relies on an additional classifier to handle distribution shift.

The core challenge is that pretrained reward models are trained exclusively on expert trajectories; non-expert observations generated during online policy exploration lie outside the training distribution, causing reward estimation to degrade.

## Method

### Overall Architecture

Progressor operates in two stages: (1) self-supervised pretraining of a reward model on expert videos to predict task progress distributions; and (2) online RL training, where the reward model is continuously refined via an adversarial push-back strategy while the policy network is simultaneously optimized.

### Key Designs

1. **Self-Supervised Progress Estimation**: Given a frame triplet $(o_i, o_j, o_g)$ (initial, current, and goal observations), progress labels are defined via normalized relative position: $\delta(o_i, o_j, o_g) = |j-i|/|g-i| \in [0,1]$. Progress is modeled as a Gaussian distribution $\mathcal{N}(\mu_{\tau_k}, \sigma_{\tau_k}^2)$, where the mean is the progress label and the standard deviation is $\sigma = \max(1/(g-i), \epsilon)$. A shared visual backbone encoder $E_\theta$ predicts the distribution parameters $(\mu, \sigma^2)$ and is optimized via KL divergence: $\mathcal{L}_{expert} = D_{KL}(p_{target} \| E_\theta)$.

2. **Reward Function Design**: The predicted progress distribution is converted into a reward signal: $r_\theta(o_i, o_j, o_g) = \mu - \alpha \mathcal{H}(\mathcal{N}(\mu, \sigma^2))$. The first term $\mu$ represents current progress, while the second term penalizes high-uncertainty predictions ($\alpha=0.4$). Since task progress is monotonically increasing, this progress estimate naturally serves as a dense reward.

3. **Adversarial Push-Back Online Refinement**: During online RL, random early-stage actions produce out-of-distribution observations. To address this, for frame triplets sampled from online rollouts, the current progress prediction is multiplied by a decay factor $\beta=0.9$ to construct a push-back target distribution $p_{push-back} = sg(\mathcal{N}(\beta\mu_{\tau'}, 1/(g-i)^2))$. The model is updated via $\mathcal{L}_{push-back} = D_{KL}(p_{push-back} \| E_\theta)$, combined with $\mathcal{L}_{expert}$ on expert data to prevent the model from being overly biased by non-expert observations.

### Loss & Training

- **Pretraining**: The progress estimation model is trained on expert videos using KL divergence.
- **Online Refinement**: Expert data loss and push-back loss are applied alternately.
- **Policy Optimization**: DrQ-v2 is used, with environment rewards replaced by Progressor rewards.
- **Real Robot Experiments**: A ResNet34 backbone is pretrained on EPIC-KITCHENS (~1.29M frames) with batch size 128, Adam optimizer, learning rate $2 \times 10^{-4}$, for 30,000 iterations.

## Key Experimental Results

### Main Results — Meta-World Simulation

Evaluated on 6 Meta-World manipulation tasks (door-open, drawer-open, hammer, peg-insert-side, pick-place, reach) over 1.5M training steps, compared against TCN, GAIL, and Rank2Reward:

| Method | door-open | drawer-open | hammer | peg-insert | pick-place | reach |
|--------|-----------|-------------|--------|------------|------------|-------|
| TCN | Fail | Fail | Fail | Fail | Fail | Fail |
| GAIL | Low | Med | Low | Fail | Low | Med |
| Rank2Reward | Low | Med | Med | Fail | Med | High |
| **Progressor** | **High** | **High** | **High** | **High** | **High** | **High** |

Progressor substantially outperforms all baselines across nearly all tasks, especially on difficult tasks (door-open, peg-insert-side) where competing methods fail entirely. On drawer-open and hammer, Progressor surpasses baselines with only 10% of the training steps.

### Ablation Study — Effect of Push-Back

| Task | With Push-Back | Without Push-Back |
|------|---------------|-------------------|
| door-open | High success rate | Low success rate |
| hammer | High success rate | Moderate success rate |
| drawer-open | High | High (comparable) |
| reach | High (slight late-stage drop) | High (stable) |

Push-back contributes substantially on difficult tasks. On the reach task, push-back may over-penalize behaviors already close to expert-level, causing a slight late-stage performance decrease.

### Real Robot Experiments

Evaluated on 4 UR5 tabletop manipulation tasks (20 success + 20 failure demos), using RWR-ACT for offline RL:

| Method | Drawer-Open | Drawer-Close | Push-Block | Pick-Place-Cup |
|--------|-------------|-------------|------------|----------------|
| Vanilla ACT | Low | Med | Very Low | Very Low |
| R3M-RWR-ACT | Med | High | Low | Low |
| VIP-RWR-ACT | Med | High | Low | Low |
| **Progressor-RWR-ACT** | **High** | **High** | **High** | **High** |

Progressor consistently outperforms all baselines across all tasks, with particularly pronounced advantages on difficult tasks (Push-Block, Pick-Place-Cup).

### Key Findings

- TCN trained solely on expert data without online updates fails on all tasks, demonstrating the importance of online training to combat distribution shift.
- Progressor provides more discriminative reward separation between successful and failed trajectories, with differences especially pronounced after step 125.
- A model pretrained on EPIC-KITCHENS human videos transfers zero-shot to robot tasks, generating reasonable reward predictions.

## Highlights & Insights

- **Task-Agnostic Unified Reward Model**: A single model handles diverse tasks without requiring per-task training as in Rank2Reward.
- **Progress Distribution over Point Estimates**: Modeling progress as a Gaussian distribution enables uncertainty estimation and explicitly penalizes high-variance predictions in the reward signal.
- **Elegant and Concise Push-Back Strategy**: Online reward refinement is achieved with a single decay factor $\beta$, requiring no additional classifier networks.
- **Zero-Shot Transfer from Human Videos to Robot Tasks**: The model pretrained on EPIC-KITCHENS is directly applied to robot manipulation tasks.

## Limitations & Future Work

- The framework assumes task progress is linearly monotonic, making it unsuitable for tasks with cyclic observations (e.g., certain environments in DeepMind Control Suite).
- Progress estimation is unimodal, and thus cannot handle tasks with multiple valid solution paths.
- The decay factor $\beta$ is fixed at 0.9; dynamic adjustment may further improve performance.
- The observed late-stage performance degradation caused by push-back on the reach task warrants further investigation.

## Related Work & Insights

- **Key distinction from Rank2Reward**: Both exploit temporal ordering information, but Progressor uses distributional estimation rather than ranking classification and operates as a single task-agnostic model.
- **Comparison with VIP/R3M**: These methods use contrastive learning for visual representation, but are less effective than progress estimation in distinguishing successful from failed trajectories.
- The push-back concept draws inspiration from domain adversarial training and may inform other visual reward learning methods that require online adaptation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Progress distribution estimation and adversarial push-back strategy are well-motivated and novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers both simulation and real robot settings with multiple baselines and ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and method motivation is well-articulated.
- **Value**: ⭐⭐⭐⭐ Provides a practical and effective framework for visual reward learning.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/roirl_efficient_self-supervised_reasoning_with_offline_iterative_reinforcement_l.md)
- [\[ICLR 2026\] Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](../../ICLR2026/reinforcement_learning/co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)
- [\[ICCV 2025\] RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment](reinforcement_learning-guided_data_selection_via_redundancy_assessment.md)
- [\[CVPR 2026\] Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](../../CVPR2026/reinforcement_learning/seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)
- [\[NeurIPS 2025\] TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning](../../NeurIPS2025/reinforcement_learning/trico_triadic_game-theoretic_co-training_for_robust_semi-supervised_learning.md)

<!-- RELATED:END -->
