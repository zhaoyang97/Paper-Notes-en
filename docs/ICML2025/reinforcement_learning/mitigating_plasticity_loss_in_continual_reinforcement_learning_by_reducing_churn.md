---
title: >-
  [Paper Note] Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn
description: >-
  [ICML2025][Reinforcement Learning][plasticity loss] This work establishes a causal relationship between plasticity loss and churn (out-of-batch output drift) through the NTK matrix, and proposes the C-CHAIN method to continuously suppress churn during continual RL training. This mitigates plasticity loss and outperforms existing baselines across 24 continual RL environments.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "plasticity loss"
  - "churn"
  - "Neural Tangent Kernel"
  - "continual learning"
  - "gradient decorrelation"
date: 2026-05-08
content_hash: a2a24ddfac0058d2
---

# Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn

**Conference**: ICML2025  
**arXiv**: [2506.00592](https://arxiv.org/abs/2506.00592)  
**Code**: [bluecontra/C-CHAIN](https://github.com/bluecontra/C-CHAIN)  
**Area**: Continual Reinforcement Learning (Continual RL)  
**Keywords**: plasticity loss, churn, Neural Tangent Kernel, continual learning, gradient decorrelation

## TL;DR

This work establishes a causal relationship between plasticity loss and churn (out-of-batch output drift) through the NTK matrix, and proposes the C-CHAIN method to continuously suppress churn during continual RL training. This mitigates plasticity loss and outperforms existing baselines across 24 continual RL environments.

## Background & Motivation

### Problem Definition

In deep reinforcement learning, when agents are trained on non-stationary data distributions using non-linear function approximators, they gradually lose **plasticity**—the capacity to adapt to new tasks or new data distributions. This phenomenon is particularly severe in **continual RL** scenarios, where a sequence of tasks $\mathbb{T} = \{\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_k\}$ arrives sequentially.

### Limitations of Prior Work

Existing mitigation strategies include resetting dormant neurons (ReDo), parameter regularization (L2 Init), backpropagation variants (Continual BP), periodic parameter resetting (Shrink & Perturb), and dynamic sparse training. However, most of these methods intervene at the network architecture or parameter level, lacking a deep understanding of the **dynamical mechanisms** driving plasticity loss.

### Our Perspective: Churn

**Churn** is defined as the implicit change in network outputs for **out-of-batch** data caused by mini-batch training. Given that network parameters transition from $\theta$ to $\theta'$, the churn on a reference data point $\bar{x} \notin B_{\text{train}}$ is defined as:

$$C_f(\bar{x}, \theta, \Delta_\theta) = f_{\theta'}(\bar{x}) - f_\theta(\bar{x}) \approx \nabla_\theta f_\theta(\bar{x})^\top \Delta_\theta$$

The core observation of this work is that **plasticity loss is highly correlated with the aggravation of churn**, and they are linked via the NTK matrix.

## Method

### 1. NTK as a Bridge

The empirical NTK matrix is defined as the gradient dot product matrix of all data points:

$$N_\theta(i,j) = \nabla_\theta f_\theta(x_i)^\top \nabla_\theta f_\theta(x_j), \quad N_\theta = G_\theta^\top G_\theta$$

where $G_\theta = [g(x_1), g(x_2), \ldots]$ is the gradient matrix of all data points. Substituting parameter updates into the first-order approximation of churn yields the vector form of churn:

$$C_f(\theta, \Delta_\theta) \approx -\eta \, N_\theta \, S \, G_L$$

- $S$: Sampling matrix (diagonal 0/1) indicating which data points are in the training batch
- $G_L$: Gradient vector of the loss function
- $\eta$: Learning rate

**Key Conclusion**: The NTK matrix $N_\theta$ simultaneously determines both plasticity (via its rank) and the magnitude of churn, serving as a natural bridge between them.

### 2. NTK Rank Collapse Triggers Churn Deterioration

In continual learning, error dynamics satisfy the following iterative relation:

$$\mathcal{E}_i(\theta_{t+1}) = (I - \eta \, N_{\theta_t} S_i) \, \mathcal{E}_i(\theta_t)$$

- When $N_\theta$ is full-rank (positive on the diagonal, zero on the off-diagonals), the learning process is stable, resembling tabular approximation.
- In practice, SGD training drives gradients to correlate, causing off-diagonal terms of the NTK to increase, which leads to **rank decline**.
- Upon task switching, the function landscape implicitly shaped by prior churn fails to align with the new distribution, further aggravating churn.
- A **vicious cycle** is formed: NTK rank decline $\leftrightarrow$ churn aggravation $\rightarrow$ plasticity loss.

### 3. C-CHAIN: Continual Churn Approximated Reduction

**Core Idea**: Continuously minimize churn on out-of-batch reference data alongside routine RL training.

**Churn reduction loss function**:

$$L_f^{\text{cr}}(\theta) = \frac{1}{2} \mathbb{E}_{\bar{x} \in B_\text{ref}} \left[ C_f(\bar{x}, \theta, \Delta_\theta)^2 \right]$$

**Total training loss**:

$$L_{\text{total}} = L_{\text{RL}}(\theta) + \lambda \, L_f^{\text{cr}}(\theta)$$

where $\lambda$ is a trade-off coefficient, and $B_\text{ref}$ is a reference batch additionally sampled from the replay buffer.

### Theoretical Analysis of Dual Effects

The churn-reduction gradient in C-CHAIN plays a dual role in learning dynamics:

| Effect | Mechanism | Role |
|------|------|------|
| **Gradient Decorrelation** | Suppresses the off-diagonal terms of the NTK matrix $N_\theta$ | Prevents rank collapse and maintains plasticity |
| **Step-size Adjustment** | Projects gradients onto out-of-batch data | Adaptively controls update magnitude and stabilizes learning |

### Algorithmic Workflow

```
Input: Task sequence T = {T_1, ..., T_k}, environment interaction steps N per task
For each task T_j:
    for step = 1 to N:
        1. Collect experience (s, a, r, s') and store in replay buffer
        2. Sample training batch B_train and reference batch B_ref
        3. Compute RL loss L_RL and record network output f_θ(B_ref) on the reference batch
        4. Execute RL gradient update → θ'
        5. Compute churn: C = f_θ'(B_ref) - f_θ(B_ref)
        6. Compute churn reduction loss L_cr = (1/2) ||C||^2
        7. Perform an additional gradient update to minimize L_cr
```

## Key Experimental Results

### Experimental Setup

- **Four Major Benchmarks**: OpenAI Gym Control, ProcGen, DeepMind Control Suite, MinAtar
- **24 Continual RL Environments in Total**: Multiple task sequences are constructed for each benchmark
- **Baseline Methods**: Vanilla (no intervention), L2 Init, Shrink & Perturb (S&P), ReDo, Continual BP, PLASTIC

### Main Results

| Benchmark | Number of Environments | C-CHAIN Performance | Key Findings |
|------|--------|-------------|---------|
| OpenAI Gym Control | 6 | Optimal in most environments | Continuous churn reduction significantly improves performance on later tasks |
| ProcGen | 6 | Optimal in multiple environments | Evident advantage in tasks with high visual complexity |
| DeepMind Control Suite | 6 | Optimal in most environments | Effectively mitigates performance degradation in continuous control tasks |
| MinAtar | 6 | Optimal in most environments | Equally effective in discrete action spaces |

### Analytical Experiments

- **NTK Rank Tracking**: The effective rank of the NTK matrix under C-CHAIN is significantly higher than that of Vanilla during training, validating the gradient decorrelation effect.
- **Churn Quantification**: Under C-CHAIN, the magnitude of churn remains low as training progresses, whereas the churn in Vanilla continuously grows.
- **Later Task Performance**: In long sequences ($\ge 10$ tasks), the performance gap between C-CHAIN and Vanilla broadens over time, indicating cumulative retention of plasticity.

## Highlights & Insights

1. **Novel Theoretical Insight**: It unifies plasticity loss and churn—two seemingly independent phenomena—through the NTK matrix for the first time, revealing the underlying vicious cycle.
2. **Simple and Elegant Method**: C-CHAIN only requires sampling an additional reference batch and computing a churn reduction loss. It is simple to implement, plug-and-play, and compatible with any RL algorithm.
3. **Theoretical Guarantees of Dual Effects**: Beyond empirical effectiveness, formal analyses are provided from the dual perspectives of gradient decorrelation and step-size adjustment.
4. **Comprehensive Experimental Coverage**: The evaluation across 24 environments spans discrete/continuous actions, low-dimensional/high-dimensional observations, and different physics engines, rendering the results highly convincing.
5. **Open Source Code**: Publicly available on GitHub for reproducibility.

## Limitations & Future Work

1. **Additional Computational Overhead**: Each step requires an extra forward pass of the reference batch and calculation of the churn loss, increasing the computational workload by approximately 30-50%.
2. **Sensitivity to Hyperparameter $\lambda$**: The weight of the churn reduction loss requires tuning, and the optimal value may vary across different environments.
3. **Reference Batch Sampling Strategy**: Currently, the reference batch is sampled uniformly from the replay buffer. More targeted sampling strategies (such as prioritized experience replay) have not been explored.
4. **Focus Only on Forward Transfer**: This study focuses on plasticity (forward transfer) without addressing the impact on catastrophic forgetting (backward transfer).
5. **NTK Analysis Relies on First-Order Approximation**: The NTK expression of churn depends on a first-order Taylor expansion, where higher-order terms may not be negligible for large step sizes or deep networks.
6. **Task Boundary Assumptions**: Task transitions in the experiments are implemented as hard switches, leaving gradual non-stationarity untested.

## Related Work & Insights

- **Original Work on Churn** (Schaul et al., 2022; Tang & Berseth, 2024): Churn studies under a single MDP. This paper presents a natural expansion of those studies to continual RL.
- **NTK and Plasticity** (Lyle et al., 2024): Empirical findings of the NTK rank as an indicator of plasticity. This paper provides a theoretical explanation at the causal level.
- **Periodic Resetting** (Nikishin et al., 2022; Schwarzer et al., 2023): Restoring plasticity via "brute-force resetting". C-CHAIN provides a gentler alternative.
- **PLASTIC** (Lee et al., 2024): A combinatorial method integrating multiple techniques. C-CHAIN, departing from a single principle, is more interpretable.

## Rating

- Novelty: ⭐⭐⭐⭐ — Unifying plasticity and churn from an NTK perspective is a novel theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 24 environments, 4 major benchmarks, and including analysis experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations, with fluid logic from observation to methodology to verification.
- Value: ⭐⭐⭐⭐ — Provides theoretical understanding and a practical solution for the plasticity issue in continual RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Rank and Gradient Lost in Non-stationarity: Sample Weight Decay for Mitigating Plasticity Loss in Reinforcement Learning](../../ICLR2026/reinforcement_learning/the_rank_and_gradient_lost_in_non-stationarity_sample_weight_decay_for_mitigatin.md)
- [\[ICML 2026\] SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning](../../ICML2026/reinforcement_learning/sphere_mitigating_the_loss_of_spectral_plasticity_in_mixture-of-experts_for_deep.md)
- [\[CVPR 2026\] Resolving the Stability-Plasticity Dilemma in Reinforcement Learning via Complementary Continual Critics](../../CVPR2026/reinforcement_learning/resolving_the_stability-plasticity_dilemma_in_reinforcement_learning_via_complem.md)
- [\[ICML 2025\] Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning](position_lifetime_tuning_is_incompatible_with_continual_reinforcement_learning.md)
- [\[ICML 2025\] Continual Reinforcement Learning by Planning with Online World Models](continual_reinforcement_learning_by_planning_with_online_world_models.md)

</div>

<!-- RELATED:END -->
