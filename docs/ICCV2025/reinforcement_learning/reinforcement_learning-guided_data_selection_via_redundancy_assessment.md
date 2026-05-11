---
title: >-
  [Paper Note] RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment
description: >-
  [ICCV 2025][Reinforcement Learning][Data Selection] This paper proposes RL-Selector, which introduces the ε-sample cover concept to quantify sample redundancy and formulates data selection as a reinforcement learning pro…
tags:
  - "ICCV 2025"
  - "Reinforcement Learning"
  - "Data Selection"
  - "Data Redundancy"
  - "ε-sample cover"
  - "Coreset"
  - "A2C"
  - "Training Efficiency"
date: 2026-05-08
content_hash: e3e2f7e5b5b0f0cd
---

# RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment

**Conference**: ICCV 2025
**arXiv**: [2506.21037](https://arxiv.org/abs/2506.21037)
**Code**: To be confirmed
**Area**: Reinforcement Learning / Data Selection
**Keywords**: Data Selection, Data Redundancy, Reinforcement Learning, ε-sample cover, Coreset, A2C, Training Efficiency

## TL;DR

This paper proposes RL-Selector, which introduces the ε-sample cover concept to quantify sample redundancy and formulates data selection as a reinforcement learning problem. A lightweight A2C policy network adaptively optimizes the selection strategy, achieving generalization performance comparable to or surpassing full-data training with significantly fewer samples across multiple benchmark datasets.

## Background & Motivation

The success of deep learning models relies heavily on large-scale datasets, which impose substantial computational and storage costs. Moreover, real-world datasets often contain abundant redundant samples that waste resources and may induce overfitting. **Data selection** aims to identify the most representative subset of samples prior to training, achieving comparable model performance with less data.

Existing methods suffer from three main limitations:

**"Group effect" problem in importance scoring**: Hand-crafted per-sample scoring methods (e.g., EL2N, Forgetting score) overlook the collective effect of the selected subset—combinations of high- and low-scoring samples can significantly affect model performance.

**Neglect of training dynamics**: Most methods perform selection using a converged proxy model, biasing toward late-stage hard samples rather than samples that are truly informative throughout training.

**Lack of cross-ratio flexibility**: Coresets selected at a specific selection ratio are difficult to transfer to other ratios, requiring a complete re-selection.

To address these issues, this paper proposes modeling data selection as an RL problem, capturing training dynamics and inter-sample relationships through policy learning.

## Method

### Overall Architecture

The RL-Selector framework comprises two models:

- **Target model $f_\theta$** (e.g., ResNet-18): Extracts features and estimates sample redundancy; updated dynamically to capture training dynamics and discarded after selection is complete.
- **RL policy model (A2C)**: A lightweight Advantage Actor-Critic network, with both actor and critic consisting of only 3 linear layers, outputting a retain/discard decision for each sample.

### Key Design 1: ε-sample cover

**Definition**: If samples $\mathbf{x}_i$ and $\mathbf{x}_j$ belong to the same class and their feature distance satisfies $\|\tilde{\mathbf{x}}_i - \tilde{\mathbf{x}}_j\| \leq \epsilon$, then $\mathbf{x}_i$ is said to be ε-covered by $\mathbf{x}_j$.

Three key theoretical propositions:

- **Proposition 1**: Mutually ε-covered samples have model output differences of $\leq \mathcal{O}(\epsilon)$.
- **Proposition 2**: Mutually ε-covered samples have loss differences that tend to zero.
- **Proposition 3**: Mutually ε-covered samples produce nearly identical gradient updates to model weights.

Core conclusion: **Highly ε-covered samples are redundant, and their removal has negligible impact on model generalization.**

### Key Design 2: RL-Driven Selection Policy

Data selection is formulated as an MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, T)$:

- **State $s$**: Feature maps from the penultimate layer of the target model, varying with training dynamics.
- **Action $a$**: Sample-level selection scores $\pi \in \mathbb{R}^N$, continuous during training and binarized with a threshold of 0.5 at inference.

**Reward function with two components:**

1. **Selection ratio alignment reward $r_1$**: Penalizes deviation of the current ratio from the target ratio, normalized for symmetry.
2. **Redundancy reward $r_2$**: Encourages removal of highly ε-covered samples. The coverage of each sample is $E_c(\mathbf{x}_{ki}) = \sum_j D_{kij}$ (sum of intra-class feature distances); smaller values indicate higher redundancy.

Total reward: $r = r_1 + r_2$, where $r_2 = E_c(\mathbf{x}) \odot \pi(\mathbf{x})$.

### Policy Optimization

The A2C algorithm is adopted:

- Actor loss: $\mathcal{L}(\theta_a) = -\log \pi_{\theta_a}(a|s) A(s,a)$
- Critic loss: $\mathcal{L}(\theta_c) = \mathbb{E}[(A(s,a))^2]$
- Selection results are constrained within ±1% of the target ratio.

### Transfer Fine-Tuning

Starting from an existing selection policy, only 15 epochs of fine-tuning are required to adapt to a new selection ratio, substantially reducing selection cost.

### Generalization Analysis

Using influence functions, the paper proves that removed ε-covered samples and retained samples have similar effects on model parameters and test loss, providing theoretical guarantees that data selection does not harm generalization.

## Key Experimental Results

### Main Results

RL-Selector **consistently outperforms** 10 state-of-the-art baselines (Random, EL2N, MoSo, GraNd, Glister, Herding, CG-Score, Forgetting, Moderate-DS, Self-sup. prototypes) across all benchmark datasets (CIFAR-100, Tiny-ImageNet, ImageNet-1k) and all selection ratios. The advantage is particularly pronounced on large-scale ImageNet-1k.

### Cross-Architecture Generalization (CIFAR-10, Selected by ResNet-18 → Trained on ResNet-50)

| Method | 60% | 70% | 80% | 90% | 100% |
|--------|-----|-----|-----|-----|------|
| EL2N | 90.32 | 90.97 | 91.61 | 91.75 | 92.34 |
| MoSo | 90.73 | 91.13 | 91.50 | 92.23 | 92.34 |
| Moderate-DS | 90.42 | 90.84 | 90.91 | 91.88 | 92.34 |
| Self-sup. Proto | 90.11 | 90.85 | 91.82 | 91.98 | 92.34 |
| **RL-Selector** | **91.79** | **92.06** | **91.93** | **92.79** | 92.34 |

Key finding: At the 90% selection ratio, RL-Selector achieves **92.79%**, surpassing full-data training at 92.34%.

### Ablation Study

| Configuration | Key Finding |
|---------------|-------------|
| ε-sample cover vs. other metrics | ε-sample cover better quantifies redundancy |
| RL policy vs. static selection | RL dynamic optimization significantly outperforms static methods |
| A2C vs. other RL algorithms | A2C achieves the best cross-ratio stability |
| Transfer fine-tuning (15 epochs) | Only marginal performance drop with greatly improved efficiency |

### Out-of-Distribution Generalization

Models trained on subsets selected from ImageNet-1k **outperform models trained on the full dataset** on out-of-distribution test sets such as ImageNet-A/R/Hard, demonstrating that redundancy removal actually improves robust generalization.

## Highlights & Insights

1. **Theory-driven redundancy metric**: ε-sample cover defines redundancy from feature-space distances; three propositions establish the substitutability of redundant samples at the levels of output, loss, and gradients.
2. **Rationale for RL formulation**: Data selection requires accounting for sample combination effects and training dynamics, which the RL framework naturally addresses.
3. **90% selection surpassing full-data training**: This phenomenon is observed on both CIFAR-10 and ImageNet-1k, strongly demonstrating that redundancy removal enhances generalization.
4. **Lightweight RL module**: The A2C actor and critic each consist of only 3 linear layers, incurring low overhead.
5. **Cross-architecture generalization**: Subsets selected using ResNet-18 perform well when training ResNet-50, ViT, and Swin models.

## Limitations & Future Work

1. RL selection requires full training of the target model; selection cost remains high for very large datasets (e.g., LAION-5B).
2. ε-sample cover relies on Euclidean distance and is strongly dependent on the quality of the feature extractor.
3. Validation is limited to classification tasks; extension to detection, segmentation, and generation has not been explored.
4. Theoretical analysis is based on simplified assumptions involving linearized ReLU networks.
5. Transfer fine-tuning exhibits notable performance degradation at low selection ratios.

## Related Work & Insights

- **Importance scoring methods**: EL2N, Forgetting Score, GraNd, Memorization—per-sample scoring that ignores the group effect.
- **Data distribution methods**: Moderate-DS (median-based selection), CCS (distribution coverage), D2 pruning (graph sampling).
- **Optimization-based methods**: Glister (bilevel optimization), Self-supervised prototypes.
- **Reinforcement learning**: A2C outperforms DQN and PPO in efficiency and stability.

## Rating

| Dimension | Score (1–10) |
|-----------|-------------|
| Novelty | 7 |
| Theoretical Depth | 7 |
| Experimental Thoroughness | 8 |
| Value | 7 |
| Writing Quality | 7 |
| **Overall** | **7** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs](mdp3_a_training-free_approach_for_list-wise_frame_selection_in_video-llms.md)
- [\[ICLR 2026\] Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](../../ICLR2026/reinforcement_learning/reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)
- [\[ICCV 2025\] Progressor: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement](progressor_a_perceptually_guided_reward_estimator_with_self-supervised_online_re.md)
- [\[NeurIPS 2025\] Strategic Costs of Perceived Bias in Fair Selection](../../NeurIPS2025/reinforcement_learning/strategic_costs_of_perceived_bias_in_fair_selection.md)
- [\[AAAI 2026\] Enhancing Robustness of Offline RL Under Data Corruption via SAM](../../AAAI2026/reinforcement_learning/enhancing_robustness_of_offline_reinforcement_learning_under_data_corruption_via.md)

</div>

<!-- RELATED:END -->
