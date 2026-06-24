---
title: >-
  [Paper Note] Online Pre-Training for Offline-to-Online Reinforcement Learning
description: >-
  [ICML2025][Reinforcement Learning][offline-to-online RL] This work proposes the OPT method, which introduces an "online pre-training" phase between offline pre-training and online fine-tuning. By introducing an independent value function trained with a meta-adaptation objective, OPT addresses the performance degradation in online fine-tuning caused by inaccurate value estimation of offline pre-trained agents, achieving an average improvement of around 30% on the D4RL benchmar…
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "offline-to-online RL"
  - "value function estimation"
  - "meta-adaptation"
  - "online pre-training"
  - "D4RL"
date: 2026-05-08
content_hash: e9997ca2912fb77d
---

# Online Pre-Training for Offline-to-Online Reinforcement Learning

**Conference**: ICML2025  
**arXiv**: [2507.08387](https://arxiv.org/abs/2507.08387)  
**Code**: To be confirmed  
**Area**: Offline-to-Online RL  
**Keywords**: offline-to-online RL, value function estimation, meta-adaptation, online pre-training, D4RL

## TL;DR

This work proposes the OPT method, which introduces an "online pre-training" phase between offline pre-training and online fine-tuning. By introducing an independent value function trained with a meta-adaptation objective, OPT addresses the performance degradation in online fine-tuning caused by inaccurate value estimation of offline pre-trained agents, achieving an average improvement of around 30% on the D4RL benchmark.

## Background & Motivation

Offline-to-online reinforcement learning (RL) aims to pre-train agents on offline datasets and then fine-tune them through online interactions to improve performance. However, recent studies have revealed a **counter-intuitive phenomenon**: offline pre-trained agents often perform **worse than training from scratch** during the online fine-tuning phase.

The core cause lies in **inaccurate value function estimation**:

- In offline RL, the value function is trained only on a fixed dataset, which introduces extrapolation errors for out-of-distribution (OOD) actions.
- Such inaccurate value estimations not only degrade offline performance but also propagate negatively during subsequent online fine-tuning.
- Existing methods (e.g., lower-bound value constraints in Cal-QL, value perturbation in Zhang et al.) rely on **the same value function** throughout both offline and online phases, failing to fundamentally solve the problem.

This paper raises two key research questions: (1) Can a newly added value function address the slow performance improvement? (2) How can this new value function be utilized most effectively?

## Method

The core idea of OPT (Online Pre-Training for Offline-to-Online RL) is to introduce a brand-new value function $Q^{\text{on-pt}}$ and design a three-phase training workflow:

### Phase 1: Offline Pre-Training

Identical to traditional offline RL, the value function $Q^{\text{off-pt}}$ and policy $\pi^{\text{off}}$ are jointly trained on the offline dataset $\mathcal{B}_{\text{off}}$. Using TD3+BC or SPOT as the backbone algorithm, the policy loss is:

$$\mathcal{L}_\pi(\phi) = \mathbb{E}_{s \sim B}\left[-Q_\theta(s, \pi_\phi(s)) + \alpha(\pi_\phi(s) - a)^2\right]$$

### Phase 2: Online Pre-Training (Core Innovation)

Initialize a brand-new value function $Q^{\text{on-pt}}_\psi$, and pre-train it using a meta-adaptation objective on both offline data and a small amount of online samples:

1. **Data Design**: First, collect $N_\tau$ online samples using the offline policy $\pi^{\text{off}}$ and store them in $\mathcal{B}_{\text{on}}$. Then, train using both $\mathcal{B}_{\text{off}}$ and $\mathcal{B}_{\text{on}}$.
2. **Meta-Adaptation Objective**:

$$\mathcal{L}_{Q^{\text{on-pt}}}^{\text{pretrain}}(\psi) = \mathcal{L}_{Q^{\text{on-pt}}}^{\text{off}}(\psi) + \mathcal{L}_{Q^{\text{on-pt}}}^{\text{on}}\left(\psi - \alpha \nabla \mathcal{L}_{Q^{\text{on-pt}}}^{\text{off}}(\psi)\right)$$

The first term learns on offline data, while the second term ensures that $Q^{\text{on-pt}}$ can quickly adapt to online data after a one-step gradient update on the offline data. This MAML-style meta-learning endows the new value function with an inherent capability for rapid adaptation to online samples.

### Phase 3: Online Fine-Tuning

The policy is improved using a weighted combination of both value functions:

$$\mathcal{L}_\pi^{\text{finetune}}(\phi) = \mathbb{E}_{s \sim B}\left[-\left\{(1-\kappa) Q^{\text{off-pt}}(s, \pi_\phi(s)) + \kappa Q^{\text{on-pt}}(s, \pi_\phi(s))\right\}\right]$$

where $\kappa \in (0, 1]$ is a weight coefficient with **dynamic scheduling**: the training initially biases towards the reliable $Q^{\text{off-pt}}$, and gradually increases $\kappa$ as online fine-tuning progresses to leverage the faster-adapting $Q^{\text{on-pt}}$. For low-quality datasets (such as "random"), the training heavily relies on $Q^{\text{on-pt}}$ from the very beginning.

## Key Experimental Results

Evaluated on the D4RL benchmark (MuJoCo / Antmaze / Adroit), with 1M steps for the offline phase and 300k steps for the online phase (where OPT uses the first 25k steps for online pre-training).

### MuJoCo Results (TD3+OPT vs. Baselines, 10 seeds)

| Environment | TD3 | Off2On | Cal-QL | **TD3+OPT** |
|------|-----|--------|--------|-------------|
| halfcheetah-r | 94.6 | 92.8 | 32.2 | **90.2** |
| hopper-r | 86.0 | 94.5 | 10.3 | **108.7** |
| walker2d-r | 0.1 | 29.4 | 10.9 | **88.0** |
| halfcheetah-m | 93.4 | 103.3 | 69.9 | **97.0** |
| hopper-m | 89.3 | 108.4 | 102.3 | **112.2** |
| walker2d-m | 103.5 | 112.3 | 96.1 | **116.1** |
| **Total** | 752.4 | 860.9 | 586.8 | **939.1** |

### Antmaze Results (SPOT+OPT vs. Baselines)

| Environment | SPOT | Cal-QL | **SPOT+OPT** |
|------|------|--------|--------------|
| umaze | 98.7 | 90.8 | **99.7** |
| umaze-diverse | 55.9 | 75.2 | **97.7** |
| medium-play | 91.1 | 94.6 | **97.6** |
| large-diverse | 70.0 | 72.9 | **90.1** |
| **Total** | 465.7 | 503.4 | **565.3** |

### Adroit Results (SPOT+OPT)

| Environment | SPOT | Cal-QL | **SPOT+OPT** |
|------|------|--------|--------------|
| pen-cloned | 114.8 | 0.21 | **136.2** |
| hammer-cloned | 84.0 | 0.23 | **121.9** |
| door-cloned | 1.6 | -0.33 | **51.1** |
| **Total** | 200.2 | -0.23 | **309.1** |

OPT achieves SOTA across all three domains. The IQM metric shows non-overlapping 95% confidence intervals, demonstrating statistical significance.

## Highlights & Insights

- **Novel Perspective**: Instead of repairing the old value function, a brand-new value function is introduced—simple yet highly effective in bypassing the pitfalls of offline value estimation bias.
- **Three-Phase Framework**: Inserting "online pre-training" between the traditional two phases prepares the new value function for adaptation using only a few online samples (25k steps).
- **Meta-Adaptation Training**: The MAML-style objective ensures that $Q^{\text{on-pt}}$ does not merely fit the data, but rather learns to adapt rapidly to online distribution shifts.
- **High Versatility**: OPT can serve as a plug-and-play module applicable to various backbone algorithms including TD3, SPOT, and IQL.
- **Adaptive $\kappa$ Scheduling**: Automatically adjusting the weights of the two value functions based on dataset quality (random vs. medium) reflects a design sensitive to data distributions.
- **Stunning Improvement on walker2d-random**: Jumping from a baseline maximum of 38.8 directly to 88.0, highlighting the massive advantage of the new value function under severe distribution shifts.

## Limitations & Future Work

- **Hyperparameter Tuning**: The scheduling strategy of $\kappa$ needs to be manually configured based on dataset quality, lacking an automated $\kappa$ adaptation mechanism.
- **Extra Computational Overhead**: Maintaining two value functions and computing meta-adaptation gradients increases both computational and memory costs.
- **Fixed Online Pre-training Steps**: The 25k steps of online pre-training are uniformly applied to all environments, which might not be optimal for some scenarios.
- **Evaluated Only on Value-Based Methods**: Adaptation to policy-gradient-based algorithms, such as SAC, remains unexplored.
- **Offline Dataset Assumptions**: Experiments are only conducted on D4RL; complex offline data distributions in real-world scenarios require further verification.
- **$N_\tau$ Sample Efficiency**: Collecting samples in the online pre-training phase without policy updates leads to some sample inefficiency.

## Related Work & Insights

- **Cal-QL** (Nakamoto et al., 2024): A value lower-bounding constraint method, but its conservatism limits the potential for policy improvement.
- **OEMA** (Guo et al., 2023): The source of the meta-adaptation idea; OPT extends it from policy adaptation to value function adaptation.
- **Off2On** (Lee et al., 2022): A balanced replay strategy; OPT also adopts this, but it becomes more effective when combined with dual value functions.
- **PEX** (Zhang et al., 2023): A multi-policy ensemble approach, which is orthogonal and complementary to the multi-value-function approach of OPT.
- **IQL** (Kostrikov et al., 2021): Extended experiments validate the generalizability of OPT.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introduces a third "online pre-training" phase and a dual value function design, presenting a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Full coverage across three major domains + multi-backbone validation + detailed ablations + IQM statistics.
- Writing Quality: ⭐⭐⭐⭐ — Problem-driven, clear narrative with rich illustrations.
- Value: ⭐⭐⭐⭐ — A plug-and-play general-purpose module that practically advances offline-to-online RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[ICML 2025\] Reward-free World Models for Online Imitation Learning](reward-free_world_models_for_online_imitation_learning.md)
- [\[ICML 2025\] Continual Reinforcement Learning by Planning with Online World Models](continual_reinforcement_learning_by_planning_with_online_world_models.md)
- [\[ICML 2025\] Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](leveraging_skills_from_unlabeled_prior_data_for_efficient_online_exploration.md)
- [\[ICML 2025\] Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability](non-stationary_online_learning_for_curved_losses_improved_dynamic_regret_via_mix.md)

</div>

<!-- RELATED:END -->
