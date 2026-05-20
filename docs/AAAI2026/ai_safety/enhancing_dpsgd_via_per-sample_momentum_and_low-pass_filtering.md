---
title: >-
  [Paper Note] Enhancing DPSGD via Per-Sample Momentum and Low-Pass Filtering
description: >-
  [AAAI2026][AI Safety][differential privacy] This paper proposes DP-PMLF, which reduces clipping bias via per-sample momentum and suppresses high-frequency DP noise via a low-pass filter…
tags:
  - "AAAI2026"
  - "AI Safety"
  - "differential privacy"
  - "DPSGD"
  - "Per-Sample Momentum"
  - "Low-Pass Filtering"
  - "Privacy-Utility Trade-off"
date: 2026-05-08
content_hash: 8db8ac0d5a116602
---

# Enhancing DPSGD via Per-Sample Momentum and Low-Pass Filtering

**Conference**: AAAI2026
**arXiv**: [2511.08841](https://arxiv.org/abs/2511.08841)  
**Code**: To be confirmed  
**Area**: AI Safety
**Keywords**: differential privacy, DPSGD, Per-Sample Momentum, Low-Pass Filtering, Privacy-Utility Trade-off

## TL;DR
This paper proposes DP-PMLF, which reduces clipping bias via per-sample momentum and suppresses high-frequency DP noise via a low-pass filter, simultaneously addressing both sources of accuracy degradation in DPSGD for the first time.

## Background & Motivation
Differentially Private Stochastic Gradient Descent (DPSGD) provides formal privacy guarantees for deep learning through gradient clipping and noise injection, but incurs severe accuracy loss. The root cause lies in two conflicting factors:

1. **DP Noise**: Privacy protection requires injecting calibrated Gaussian noise into the aggregated gradient, with noise scale proportional to the clipping threshold $C$ — larger $C$ yields more noise.
2. **Clipping Bias**: Clipping per-sample gradient norms introduces bias — smaller $C$ yields larger bias.

Most existing methods address only one of these issues:
- **LP-DPSGD** (Zhang et al.) applies a low-pass filter to reduce DP noise but introduces an additional bias term, leading to worse performance when clipping bias dominates.
- **InnerOuter** (Xiao et al.) uses inner-outer momentum to reduce clipping bias but lacks noise suppression, and its unnormalized outer momentum accumulates noise, causing severe degradation at $\epsilon=1$ (strong privacy).

The authors observe that clipping bias is proportional not only to the threshold $C$ but also to the sampling variance $\sigma_{SGD}$, leaving room to simultaneously reduce both noise and bias.

## Core Problem
How can DP noise and clipping bias in DPSGD be **simultaneously** mitigated without consuming additional privacy budget?

## Method

### Overall Architecture: DP-PMLF
DP-PMLF consists of two complementary modules applied sequentially in the gradient processing pipeline.

### 1. Per-Sample Momentum
A momentum term is maintained for each sample $\xi$, computing an exponentially weighted moving average of gradients over the past $k$ steps:

$$v_t^{(\xi)} = \sum_{i=t-k+1}^{t} \hat{\beta}^{t-i} \nabla f^{(\xi)}(x_i)$$

where $\hat{\beta}^{t-i} = \beta^{t-i}/c_\beta$ and $c_\beta$ is a normalization constant ensuring the coefficients sum to 1.

**Function**: Smooths gradient estimates prior to clipping, reducing sampling variance $\sigma_{SGD}$. Theoretically, variance is reduced by a factor of $\rho^2$, where:

$$\rho = \sqrt{\frac{(1+\beta)(1-\beta^k)}{(1-\beta)(1+\beta^k)}}$$

As $\beta \to 1$, $\rho^2 \to k$, corresponding to uniform averaging. Normalization also prevents noise accumulation from overly large momentum coefficients — a known weakness of InnerOuter.

### 2. Low-Pass Filter
After clipping, gradients are aggregated with Gaussian noise, and a linear low-pass filter is applied:

$$m_t = -\sum_{r=1}^{n_a} a_r m_{t-r} + \sum_{r=0}^{n_b} b_r \bar{v}_{t-r}$$

The filter coefficients satisfy $-\sum a_r + \sum b_r = 1$ to preserve the signal mean.

**Mechanism**: DP noise is uniformly distributed across all frequencies, while the true gradient signal concentrates in the low-frequency band. The low-pass filter retains the gradient signal while suppressing high-frequency noise. Since the filter is applied as post-processing on already-noised outputs, by the **post-processing lemma** of differential privacy, it **consumes no additional privacy budget**.

### 3. Initialization Bias Correction
A normalization constant $c_{m,t}$ is computed recursively, and the output $\hat{m}_t = m_t / c_{m,t}$ corrects the transient bias during the early phase of filtering.

### Algorithm
1. Sample mini-batch $\mathcal{B}_t$
2. Compute per-sample momentum $v_t^{(\xi)}$ for each sample
3. Clip: $\tilde{v}_t^{(\xi)} = \text{clip}(v_t^{(\xi)}, C)$
4. Aggregate and add noise: $\bar{v}_t = \frac{1}{B}\sum \tilde{v}_t^{(\xi)} + w_t$
5. Apply low-pass filter + bias correction → $\hat{m}_t$
6. Update model: $x_{t+1} = x_t - \eta \hat{m}_t$

### Theoretical Guarantees
- **Convergence**: Under standard assumptions of $L$-smoothness and bounded variance/gradients, the convergence upper bound is:

$$\mathcal{O}\!\left(\frac{f(x_0)-f^*}{\eta T} + L\eta C^2 + \frac{L\eta \, d\sigma_{DP}^2}{\Gamma_{DP}} + \frac{\sigma_{SGD}^2}{\rho^2 \Gamma_{SGD}}\right)$$

where $\Gamma_{DP}$ and $\Gamma_{SGD}$ reflect the suppression factors of the low-pass filter on DP noise and clipping bias, respectively. Compared to vanilla DPSGD, the clipping bias term is additionally divided by $\rho^2$ and the DP noise term by $\Gamma_{DP}$.

- **Privacy**: The method satisfies $(\epsilon, \delta)$-DP via the Gaussian mechanism, privacy amplification by subsampling, and moments accountant.

## Key Experimental Results

### Image Classification (ViT, No Pretraining)

| Method | CIFAR-10 ($\epsilon$=1) | CIFAR-10 ($\epsilon$=8) | CIFAR-100 ($\epsilon$=1) | CIFAR-100 ($\epsilon$=8) |
|---|---|---|---|---|
| DPSGD | 35.74 | 47.74 | 7.52 | 18.27 |
| LP-DPSGD | 35.84 | 48.37 | 7.55 | 18.52 |
| InnerOuter | 11.55 | 33.53 | 1.13 | 13.93 |
| **DP-PMLF** | **40.96** | **51.47** | **11.40** | **23.15** |

- At $\epsilon=1$, DP-PMLF outperforms the best baseline by approximately **5%** on CIFAR-10 and **4%** on CIFAR-100.
- InnerOuter degrades severely at $\epsilon=1$ due to noise accumulation (11.55% on CIFAR-10).

### Sentence Classification (RoBERTa-base Fine-tuning, GLUE)

| Method | MNLI ($\epsilon$=1) | QNLI ($\epsilon$=1) | QQP ($\epsilon$=8) | SST-2 ($\epsilon$=8) |
|---|---|---|---|---|
| DPSGD | 51.36 | 65.59 | 80.38 | 90.83 |
| **DP-PMLF** | **56.81** | **72.38** | **83.42** | **90.39** |

- DP-PMLF surpasses the baseline by over 4% on MNLI and nearly 3% on QNLI at $\epsilon=1$.

### Multiple Architectures (CIFAR-10, $\epsilon=1$)
- CNN-5: DP-PMLF ~47%, outperforming the best baseline by ~9%
- ResNet-18: DP-PMLF ~50%, outperforming by ~1–2%
- ViT: DP-PMLF ~31%, outperforming by ~8%

### Ablation Study
- Removing per-sample momentum consistently degrades performance, validating its effectiveness in reducing clipping bias.
- Removing the low-pass filter reduces performance when $\epsilon \leq 6$ (higher noise); however, when $\epsilon > 6$, excessive smoothing slightly degrades performance by filtering out true gradient information (~0.5–0.7%).

## Highlights & Insights
1. **First simultaneous treatment of both degradation sources**: The combination of per-sample momentum (variance/bias reduction) and low-pass filtering (noise suppression) covers the failure modes of both LP-DPSGD and InnerOuter.
2. **Zero privacy cost post-processing**: The low-pass filter exploits the post-processing property of DP and consumes no privacy budget.
3. **Rigorous theoretical guarantees**: Complete convergence analysis and privacy proofs are provided; the convergence bound clearly delineates the contribution of each module.
4. **Cross-modal generalization**: The method is effective for both image classification (CNN/ResNet/ViT) and sentence classification (RoBERTa).

## Limitations & Future Work
1. **Hyperparameter sensitivity**: $\beta$, $k$, and filter coefficients $\{a_r\}, \{b_r\}$ require manual tuning; no adaptive selection strategy is proposed.
2. **Per-sample history storage overhead**: Maintaining gradients from the most recent $k$ steps for each sample incurs memory costs that scale with dataset size and $k$.
3. **Absolute accuracy under strong privacy remains low**: The best result on CIFAR-10 at $\epsilon=1$ is only 40.96%, which remains far from practical utility.
4. **Strong theoretical assumptions**: The analysis requires bounded gradients (Assumption 3) and gradient autocorrelation (Assumption 4), and is not extended to more general non-convex conditions such as the PL condition or $(L_0, L_1)$-smoothness.
5. **Over-smoothing risk**: Ablation results show that the low-pass filter slightly hurts performance when DP noise is small ($\epsilon > 6$).

## Related Work & Insights

| Method | Reduces DP Noise | Reduces Clipping Bias | Theoretical Guarantee | Extra Privacy Cost |
|---|---|---|---|---|
| LP-DPSGD | ✓ | ✗ (increases bias) | Yes (with extra bias term) | No |
| InnerOuter | ✗ (accumulates noise) | ✓ | No | No |
| DiceSGD | ✗ (requires more noise) | ✓ (error feedback) | Yes | Yes |
| Clipless DPSGD | ✓ | ✓ (no clipping) | Yes | No | Requires specific architecture |
| **DP-PMLF** | **✓** | **✓** | **Yes** | **No** |

**Insights and Connections**
- **Variance reduction as a general tool**: The core of per-sample momentum is reducing sampling variance; this idea can be extended to other DP optimizers such as DP-Adam.
- **Frequency-domain perspective worth further exploration**: The low-pass filter exploits the spectral difference between gradient signal and noise; higher-order or adaptive filters may further improve performance.
- **Connection to federated learning**: Federated learning similarly faces accuracy loss from noise and gradient compression; analogous momentum smoothing and frequency-domain filtering strategies may prove effective.

## Rating
- Novelty: 7/10 — An elegant combination of two known components; the core contribution lies in the insight of simultaneous treatment and the accompanying theoretical analysis
- Experimental Thoroughness: 7/10 — Covers multiple datasets, architectures, and modalities, but lacks comparisons on large-scale models and more DPSGD variants
- Writing Quality: 8/10 — Motivation is clear; theoretical and experimental sections are well-organized
- Value: 7/10 — Provides a practical and theoretically grounded improvement in the domain of DP training

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction](../../ICLR2026/ai_safety/sample-efficient_distributionally_robust_multi-agent_reinforcement_learning_via_.md)
- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](../../ICLR2026/ai_safety/toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[NeurIPS 2025\] Perturbation Bounds for Low-Rank Inverse Approximations under Noise](../../NeurIPS2025/ai_safety/perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)
- [\[NeurIPS 2025\] Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy](../../NeurIPS2025/ai_safety/spectral_perturbation_bounds_for_low-rank_approximation_with_applications_to_pri.md)
- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](../../NeurIPS2025/ai_safety/enhancing_graph_classification_robustness_with_singular_pooling.md)

</div>

<!-- RELATED:END -->
