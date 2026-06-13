---
title: >-
  [Paper Note] ANO: Faster is Better in Noisy Landscapes
description: >-
  [ICLR 2026][optimizer] This paper proposes the Ano optimizer, which decouples the update direction from its magnitude — the direction is determined by the sign of the momentum for noise robustness…
tags:
  - "ICLR 2026"
  - "optimizer"
  - "sign-based"
  - "noise robustness"
  - "reinforcement-learning"
  - "direction-magnitude decoupling"
date: 2026-05-08
content_hash: 0d4f72e7a587290a
---

# ANO: Faster is Better in Noisy Landscapes

**Conference**: ICLR 2026
**arXiv**: [2508.18258](https://arxiv.org/abs/2508.18258)  
**Code**: Available  
**Area**: Other
**Keywords**: optimizer, sign-based, noise robustness, reinforcement-learning, direction-magnitude decoupling

## TL;DR
This paper proposes the Ano optimizer, which decouples the update direction from its magnitude — the direction is determined by the sign of the momentum for noise robustness, while the magnitude is determined by the instantaneous gradient absolute value (rather than the momentum magnitude) for responsiveness. Combined with an improved Yogi-style variance estimator, Ano significantly outperforms Adam/Lion/Adan in noisy and non-stationary environments (e.g., RL), while remaining competitive on standard tasks.

## Background & Motivation
**Background**: Adam and its variants are the default optimizers in deep learning, but degrade in noisy or non-stationary settings (high gradient noise, label ambiguity, shifting RL objectives).

**Limitations of Prior Work**: Adam derives both direction and magnitude from the momentum $m_k$ — when large noise spikes occur, opposing directions partially cancel, reducing effective momentum magnitude and causing overly conservative updates. The exponential moving average of the second moment allows noise spikes to persist for many steps.

**Key Challenge**: Momentum smooths the directional signal well (reducing oscillations from noisy directions), but the *magnitude* of momentum is too sluggish — it responds too slowly to large gradient changes. What is needed is a combination of "stable direction + agile magnitude."

**Goal**: Design an optimizer that is more robust in noisy optimization environments while retaining the simplicity and efficiency of first-order methods.

**Key Insight**: Explicitly decouple direction and magnitude — direction = sign(momentum), magnitude = |gradient|, with second moment updated via an improved Yogi rule incorporating a decay factor to control memory length.

**Core Idea**: Use the sign of momentum to determine the direction and the absolute value of the current gradient to determine the step size — decoupling achieves the optimal balance between noise robustness and responsiveness.

## Method

### Overall Architecture
The Ano update rule: $x_{k+1} = x_k - \frac{\eta_k}{\sqrt{\hat{v}_k} + \epsilon} \cdot |g_k| \cdot \text{sign}(m_k) - \eta_k \lambda x_k$. The key difference from Adam lies in replacing $m_k$ with $|g_k| \cdot \text{sign}(m_k)$.

### Key Designs

1. **Sign-Magnitude Decoupling**:

    - Function: Direction is derived from the momentum sign $\text{sign}(m_k)$; magnitude is derived from the instantaneous gradient $|g_k|$.
    - vs. Adam: Adam uses $m_k = |m_k| \cdot \text{sign}(m_k)$, so both direction and magnitude come from momentum. Under high noise, $|m_k|$ is suppressed by averaging (directional oscillations cause cancellation), slowing updates.
    - vs. SignSGD/Lion: Pure sign methods discard magnitude information. Ano retains magnitude but uses the more responsive $|g_k|$ instead of the lagging $|m_k|$.

2. **Improved Second Moment Update**:

    - Formula: $v_k = \beta_2 v_{k-1} - (1-\beta_2) \cdot \text{sign}(v_{k-1} - g_k^2) \cdot g_k^2$
    - Inherits Yogi's asymmetric update (fast recovery) and adds $\beta_2$ decay to control memory length.
    - Design Motivation: Adam's EMA allows variance spikes to persist too long; Yogi recovers quickly but lacks decay. Adding decay enables both fast recovery and smooth forgetting.

3. **Anolog Variant (Adaptive β₁)**:

    - $\beta_{1,k} = 1 - 1/\log(k+2)$ — a logarithmic schedule that gradually enlarges the momentum window.
    - Eliminates the need to tune the $\beta_1$ hyperparameter.
    - More gradual than square-root or harmonic schedules — preserves adaptability in non-stationary environments.

### Loss & Training
Same memory and computational cost as Adam (maintains $m_k, v_k$). Default hyperparameters: $\beta_1=0.92$, $\beta_2=0.99$.

## Key Experimental Results

### Noise Robustness (CIFAR-10 + Gradient Noise Injection)

| Optimizer | σ=0 | σ=0.05 | σ=0.10 | σ=0.20 |
|-----------|------|--------|--------|--------|
| **Ano** | **82.10** | **70.88** | **65.93** | **59.54** |
| Adam | 80.67 | 66.86 | 60.83 | 52.46 |
| Lion | 81.04 | 69.62 | 64.02 | 56.82 |

### Key Findings
- The advantage of Ano over Adam grows with noise level: +1.4% at σ=0, +7.1% at σ=0.20.
- Improvements are most pronounced on RL tasks (non-stationary objectives), as RL gradients are inherently high-variance and non-stationary.
- Anolog sacrifices a small amount of peak performance but eliminates $\beta_1$ tuning — offering high practical value.
- On standard low-noise tasks (e.g., standard ImageNet training), Ano remains competitive with Adam.

### Theoretical Guarantees
- Non-convex convergence rate $\tilde{O}(K^{-1/4})$, matching sign-based methods such as Lion and Signum.
- Slower than SGD/Adam's $O(K^{-1/2})$, which is an inherent limitation of sign-based methods.

## Highlights & Insights
- **The decoupling principle of "momentum for direction, current gradient for magnitude"** is simple, intuitive, and effective. The modification over Adam is minimal yet yields significant gains.
- **Particular relevance to RL optimization**: The high variance and non-stationarity of RL gradients are longstanding pain points for the Adam family; Ano's decoupled design is naturally better suited to this regime.
- **Complementary to DRPO**: DRPO addresses reward design issues in GRPO, while Ano addresses noise problems in the optimizer itself — the two can be combined.

## Limitations & Future Work
- Theoretical convergence rate is slower than Adam ($K^{-1/4}$ vs. $K^{-1/2}$), although Ano converges faster in practice under noisy conditions.
- No clear advantage in extremely low-noise settings, where Adam's smooth updates are more beneficial.
- Validation is limited to CNN and RL tasks; performance on large-scale LLM training remains unknown.
- The improved Yogi update for $\beta_2$ increases the complexity of theoretical analysis.

## Related Work & Insights
- **vs. Adam**: Ano's direction-magnitude decoupling resolves Adam's conservatism in noisy environments.
- **vs. Lion**: Lion's pure sign discards magnitude information; Ano retains magnitude via $|g_k|$.
- **vs. Grams**: Grams uses the gradient sign for direction and momentum norm for magnitude; Ano inverts this — momentum sign for direction and gradient norm for magnitude.

## Rating
- Novelty: ⭐⭐⭐⭐ The direction/magnitude decoupling design is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Noise injection experiments are convincing; RL experiments validate the core scenario.
- Writing Quality: ⭐⭐⭐⭐ Algorithm description is clear with complete theoretical analysis.
- Value: ⭐⭐⭐⭐ Provides a practical alternative optimizer for noisy optimization environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)
- [\[AAAI 2026\] Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](../../AAAI2026/others/faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)
- [\[ICML 2026\] Envy-Free Allocation of Indivisible Goods via Noisy Queries](../../ICML2026/others/envy-free_allocation_of_indivisible_goods_via_noisy_queries.md)
- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[ICML 2026\] Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases](../../ICML2026/others/less_data_faster_training_repeating_smaller_datasets_speeds_up_learning_via_samp.md)

</div>

<!-- RELATED:END -->
