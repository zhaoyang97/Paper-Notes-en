---
title: >-
  [Paper Note] Controllable Feature Whitening for Hyperparameter-Free Bias Mitigation
description: >-
  [ICCV 2025][AI Safety][bias mitigation] This paper proposes the Controllable Feature Whitening (CFW) framework, which eliminates linear correlations between target features and bias features via whitening transformations to mitigate model bias. The approach requires neither adversarial training nor additional regularization hyperparameters, and supports smooth interpolation between demographic parity and equalized odds through a single weighting coefficient.
tags:
  - ICCV 2025
  - AI Safety
  - bias mitigation
  - feature whitening
  - fairness
  - demographic parity
  - equalized odds
date: 2026-05-08
content_hash: 71a688b0c91bbf42
---

# Controllable Feature Whitening for Hyperparameter-Free Bias Mitigation

**Conference**: ICCV 2025
**arXiv**: [2507.20284](https://arxiv.org/abs/2507.20284)
**Code**: N/A
**Area**: AI Safety / Fairness & Bias Mitigation
**Keywords**: bias mitigation, feature whitening, fairness, demographic parity, equalized odds

## TL;DR

This paper proposes the Controllable Feature Whitening (CFW) framework, which eliminates linear correlations between target features and bias features via whitening transformations to mitigate model bias. The approach requires neither adversarial training nor additional regularization hyperparameters, and supports smooth interpolation between demographic parity and equalized odds through a single weighting coefficient.

## Background & Motivation

Deep neural networks trained on biased datasets tend to exploit spurious correlations (e.g., recognizing objects via background cues), leading to severe performance degradation on bias-conflicting samples. Existing methods suffer from two key issues:

**Adversarial training instability**: Methods that train auxiliary networks to "forget" bias information via min-max games are inherently unstable.

**Regularization requires careful tuning**: Approaches measuring statistical dependence via mutual information or HSIC are sensitive to hyperparameters, and the accuracy of neural estimators is difficult to assess.

The core insight of this work is that, while zero covariance does not imply statistical independence, it does guarantee linear independence — i.e., one variable cannot be expressed as a linear combination of the other. Since the final layer of a deep network is typically a linear classifier, linear independence among features fed into that layer is sufficient for effective debiasing.

## Method

### Overall Architecture

1. A Vanilla network $f_t$ is trained with standard cross-entropy on biased data, yielding a biased target encoder $h_t$.
2. $h_t$ is frozen; a bias encoder $h_b$ is trained to predict bias attributes $B$.
3. Target features $z_t = h_t(X)$ and bias features $z_b = h_b(X)$ are concatenated and passed through the controllable whitening module $W_\lambda$.
4. The whitened target features $z_{wt}$ and bias features $z_{wb}$ are linearly independent.
5. Separate linear classifiers $g_{wt}$ and $g_{wb}$ are trained to predict target and bias attributes, respectively.

### Key Designs

1. **Feature Whitening for Debiasing**:

   - Whitening transform: $\tilde{X} = \Sigma^{-1/2} \cdot (X - \mu \cdot \mathbf{1}^\top)$
   - Target and bias features are concatenated as $z = [z_t; z_b]$ and jointly whitened.
   - After whitening, all channel pairs between $z_{wt}$ and $z_{wb}$ are orthogonal, preventing the linear classifier $g_{wt}$ from extracting bias information from $z_{wt}$ linearly.
   - $\Sigma^{-1/2}$ is computed via coupled Newton-Schultz iterations for numerical stability and computational efficiency.
   - The non-uniqueness of $\Sigma^{-1/2}$ is exploited to keep $z_{wt}$ as close as possible to the original $z_t$, preserving task-relevant information.

2. **Covariance Reweighting Strategy**:

   - Biased covariance $\Sigma_b$: estimated directly from biased training data.
   - Unbiased covariance $\Sigma_u$: computed by upweighting rare groups and downweighting majority groups such that $P(y,b|\mathcal{D}_u) = \frac{1}{N_Y \cdot N_B}$.
   - Key insight: under an unbiased distribution, demographic parity is equivalent to equalized odds.
   - Whitening with $\Sigma_b$ promotes demographic parity (unconditional independence of $\hat{Y}$ and $B$).
   - Whitening with $\Sigma_u$ promotes equalized odds (conditional independence of $\hat{Y}$ and $B$ given $Y$).

3. **Controllable Feature Whitening (CFW)**:

   - Mixed covariance: $\Sigma_\lambda = \lambda \cdot \Sigma_u + (1-\lambda) \cdot \Sigma_b$
   - $\lambda = 0$: whitening with pure biased covariance, reducing $\Delta_{DP}$ but potentially discarding target-relevant information.
   - $\lambda = 1$: whitening with pure unbiased covariance, reducing $\Delta_{EO}$ but susceptible to overfitting due to limited diversity in rare groups.
   - $\lambda = 0.25$: empirically optimal, consistently performing well across all datasets; the method can thus be regarded as hyperparameter-free in practice.
   - Training objective: $\min_{g_{wt}} \mathcal{L}_t + \min_{h_b, g_{wb}} \mathcal{L}_b$

### Loss & Training

- The target encoder $h_t$ (pretrained via Vanilla training) is frozen; only the bias encoder $h_b$ and two linear classifiers are trained.
- Loss reweighting is applied to $\mathcal{L}_t$, downweighting losses of bias-aligned samples to simulate an unbiased loss distribution.
- Training is stable, requiring no min-max adversarial optimization.

## Key Experimental Results

### Main Results

| Method | Bias Labels | CIFAR-10 (0.5%) | CIFAR-10 (5%) | bFFHQ (0.5%) | CelebA-Blond (WG) |
|--------|-------------|-----------------|---------------|--------------|-------------------|
| Vanilla | ✗ | 23.26 | 41.98 | 56.20 | 16.48 |
| LfF | ✗ | 28.57 | 50.27 | 62.20 | - |
| SelecMix+L (w/ GT) | ✓ | 37.02 | 53.47 | 75.00 | - |
| Ours+V | ✓ | 32.08 | 53.08 | 79.80 | - |
| **Ours+S** | ✓ | **42.51** | **59.05** | **82.77** | - |
| Ours (Res50) | ✓ | - | - | - | **91.02** |

### Ablation Study

| Configuration ($\lambda$) | $\Delta_{DP}$↓ | $\Delta_{EO}$↓ | Bias-Conflicting Acc.↑ | Notes |
|--------------------------|----------------|----------------|------------------------|-------|
| $\lambda=0$ (biased only) | Low | High | Moderate | Demographic parity but information loss |
| $\lambda=0.25$ (recommended) | Medium | Medium | **Highest** | Best trade-off |
| $\lambda=1$ (unbiased only) | High | Low | High but overfits | Equalized odds but overfits rare groups |

### Key Findings

- Eliminating only linear correlations yields significant fairness improvements without modeling higher-order dependencies.
- $\lambda=0.25$ is consistently optimal across all datasets, rendering the method effectively hyperparameter-free.
- t-SNE visualizations show that whitened features $z_{wt}$ cluster by target attribute for both bias-aligned and bias-conflicting samples.
- Integration with stronger target encoders (e.g., SelecMix) yields further gains (Ours+S vs. Ours+V).
- On CelebA-BlondHair with ResNet-50, the worst-group accuracy reaches 91.02%, surpassing GroupDRO (87.2) and LISA (89.3).

## Highlights & Insights

- **Theoretically grounded insight**: Linear independence does not imply statistical independence, yet it is sufficient for effective debiasing under the "final linear classifier" setting.
- **Minimal yet effective design**: The framework requires only a frozen encoder, a whitening module, and two linear classifiers — no adversarial training, no regularization.
- **Fine-grained controllability**: A single parameter $\lambda$ enables smooth interpolation between demographic parity and equalized odds.
- The work demonstrates an important observation: fine-tuning only the final linear layer is sufficient to achieve fairness.

## Limitations & Future Work

- Bias labels are required, limiting applicability in settings where bias attributes are unknown.
- Only linear dependencies are eliminated; higher-order nonlinear biases may theoretically persist.
- Whitening with pure unbiased covariance ($\lambda=1$) risks overfitting, indicating that reweighting alone cannot fully compensate for limited sample diversity in rare groups.
- Freezing the encoder may impose an upper bound on representation quality.

## Related Work & Insights

- **Adversarial methods (GRL, LNL, etc.)**: Achieve feature debiasing via gradient reversal and similar techniques, but suffer from training instability.
- **Data-side methods (SelecMix, CNC, etc.)**: Augment rare group samples through mixing or contrastive strategies.
- The proposed whitening approach is complementary to virtually all existing methods — any improvement to the target encoder directly translates to better debiasing performance.

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying whitening to bias mitigation is a novel perspective, though whitening itself is a well-established technique.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, multiple bias ratios, comprehensive baselines, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and experimental analysis is systematic.
- Value: ⭐⭐⭐⭐ — A hyperparameter-free debiasing method with strong practical utility for rapid industrial deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Backdoor Mitigation by Distance-Driven Detoxification](backdoor_mitigation_by_distance-driven_detoxification.md)
- [\[ICCV 2025\] FRET: Feature Redundancy Elimination for Test Time Adaptation](fret_feature_redundancy_elimination_for_test_time_adaptation.md)
- [\[ICCV 2025\] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)
- [\[NeurIPS 2025\] Fair Representation Learning with Controllable High Confidence Guarantees via Adversarial Inference](../../NeurIPS2025/ai_safety/fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)
- [\[NeurIPS 2025\] Robust Graph Condensation via Classification Complexity Mitigation](../../NeurIPS2025/ai_safety/robust_graph_condensation_via_classification_complexity_mitigation.md)

<!-- RELATED:END -->
