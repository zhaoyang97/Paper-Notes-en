---
title: >-
  [Paper Note] Dynamic Momentum Recalibration in Online Gradient Learning
description: >-
  [CVPR 2026][Optimization][optimizer] From a signal processing perspective, this work identifies the inherent bias-variance tradeoff deficiencies of fixed momentum coefficients and proposes the SGDF optimizer, which dynamically balances noise suppression and signal preservation in gradient estimation by computing optimal time-varying gains online under the minimum mean squared error principle, outperforming SGD momentum and Adam variants across multiple vision tasks.
tags:
  - CVPR 2026
  - Optimization
  - optimizer
  - momentum
  - bias-variance tradeoff
  - optimal linear filter
  - gradient estimation
date: 2026-05-08
content_hash: 6ded7e008a5e741e
---

# Dynamic Momentum Recalibration in Online Gradient Learning

**Conference**: CVPR 2026
**arXiv**: [2603.06120](https://arxiv.org/abs/2603.06120)
**Code**: [GitHub](https://github.com/LilYau350/SGDF-Optimizer)
**Area**: Optimization
**Keywords**: optimizer, momentum, bias-variance tradeoff, optimal linear filter, gradient estimation

## TL;DR
From a signal processing perspective, this work identifies the inherent bias-variance tradeoff deficiencies of fixed momentum coefficients and proposes the SGDF optimizer, which dynamically balances noise suppression and signal preservation in gradient estimation by computing optimal time-varying gains online under the minimum mean squared error principle, outperforming SGD momentum and Adam variants across multiple vision tasks.

## Background & Motivation

**Background**: SGD and its momentum variants (EMA/CM) alongside adaptive methods (Adam/AdamW) form the foundation of deep learning optimization. Momentum methods smooth noise via historical gradients, while adaptive methods scale learning rates using second-order moments.

**Limitations of Prior Work**: Analysis under the SDE framework reveals that EMA ($u=1-\beta$), acting as a low-pass filter, reduces variance as $\beta \to 1$ but causes bias to diverge (accumulating stale gradients); CM ($u=1$) is more aggressive, with both bias and variance diverging as $\beta \to 1$. Both methods are locked into a preset bias-variance tradeoff via fixed coefficients and cannot adapt to the dynamically changing noise and curvature during training.

**Key Challenge**: Structurally reducing variance inevitably amplifies bias, while reducing bias inevitably exposes the estimator to higher variance — this is the fundamental dilemma of static momentum coefficients.

**Goal**: Design an adaptive gain that reduces momentum reliance during low-variance phases to minimize bias, while heavily leveraging momentum updates to filter noise during high-variance phases.

**Key Insight**: Drawing from optimal linear filtering (Kalman Filter principles), the historical gradient estimate and the current gradient observation are treated as two Gaussian sources with distinct uncertainties to be fused.

**Core Idea**: Apply the minimum mean squared error principle to compute a time-varying gain $K_t$ online, achieving optimal linear fusion of the momentum estimate and the current gradient.

## Method

### Overall Architecture
SGDF augments standard SGD with momentum by incorporating an online-computed gain $K_t$. At each step: (1) maintain the first moment $m_t$ and the "innovation" variance $s_t$ via EMA; (2) compute the optimal gain $K_t = \hat{s}_t / (\hat{s}_t + (g_t - \hat{m}_t)^2 + \epsilon)$; (3) fuse the momentum estimate and current gradient as $\hat{g}_t = \hat{m}_t + K_t^\gamma (g_t - \hat{m}_t)$; (4) update parameters using $\hat{g}_t$.

### Key Designs

1. **Optimal Time-Varying Gain**

    - **Function**: Computes online the optimal fusion weight between the current observation and the historical estimate.
    - **Mechanism**: The gradient estimate is expressed as a linear interpolation $\hat{g}_t = \hat{m}_t + K_t(g_t - \hat{m}_t)$, where $(g_t - \hat{m}_t)$ serves as the "innovation" term. Setting the derivative of $\text{Var}(\hat{g}_t)$ with respect to $K_t$ to zero yields the optimal gain $K_t^* = \text{Var}(\hat{m}_t) / (\text{Var}(\hat{m}_t) + \text{Var}(g_t))$. The EMA of $s_t$ estimates $\text{Var}(\hat{m}_t)$, while the squared current innovation estimates $\text{Var}(g_t)$.
    - **Design Motivation**: This corresponds precisely to the Kalman Filter update formula applied to gradient estimation — when the momentum estimate is highly uncertain (large $\hat{s}_t$), more weight is placed on the current gradient; when the current observation is noisy, more weight is placed on the historical momentum.

2. **Variance Correction Factor**

    - **Function**: Corrects the biased estimate of $s_t$.
    - **Mechanism**: A correction factor $(1-\beta_1)(1-\beta_1^{2t})/(1+\beta_1)$ is introduced to more accurately debias the EMA second moment (distinct from Adam's standard correction), yielding greater precision under the assumption of independent, bounded-variance gradients.
    - **Design Motivation**: Adam's standard bias correction is insufficiently accurate when estimating the innovation variance, which degrades the quality of $K_t$.

3. **Power Scaling ($\gamma=1/2$)**

    - **Function**: Replaces $K_t$ with $K_t^\gamma$ to enhance responsiveness in noisy environments.
    - **Mechanism**: Setting $\gamma=1/2$ is equivalent to modulating the effective observation variance to $\sqrt{\text{Var}(g_t)}$, expanding the signal-responsive range of the gain.
    - **Design Motivation**: The raw $K_t$ is overly conservative under high noise (almost fully trusting momentum); $K_t^{0.5}$ allows the estimator to retain meaningful responsiveness to the observed signal even when noise is high.

### Loss & Training
- Hyperparameters follow Adam's standard settings: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, with learning rates searched over the same range as SGD.
- Convergence rate is $O(\sqrt{T})$ in the convex case and $O(\log T / \sqrt{T})$ in the non-convex case, consistent with Adam-family methods.
- Extendable to the Adam framework (replacing Adam's first-moment estimate), improving generalization on certain tasks.

## Key Experimental Results

### Main Results

**CIFAR-10/100 Image Classification (VGG/ResNet/DenseNet)**

| Method | VGG11-C10 | ResNet34-C10 | DenseNet121-C100 |
|--------|-----------|--------------|------------------|
| SGD | ~93.5 | ~95.5 | ~77.0 |
| Adam | ~92.8 | ~94.8 | ~76.5 |
| AdaBelief | ~93.2 | ~95.3 | ~77.2 |
| **SGDF** | **~93.8** | **~95.8** | **~77.8** |

**ImageNet ResNet18 Top-1/Top-5**

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| SGD | 70.23 | 89.35 |
| AdaBelief | 70.08 | 89.37 |
| **SGDF** | **70.5+** | **89.6+** |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| SGDF (full) | Best | Includes variance correction + power scaling |
| w/o variance correction | Degraded | Correction factor improves $K_t$ estimation quality |
| w/o power scaling ($\gamma=1$) | Degraded | Raw $K_t$ is overly conservative under high noise |
| SGDF extended to Adam | Improved | Replacing Adam's first moment improves generalization |

### Key Findings
- SGDF shows a more pronounced advantage on VGG (without residual connections), suggesting greater benefit for networks with higher gradient noise and more difficult gradient propagation.
- SGDF can be seamlessly integrated into the Adam framework (replacing its first-moment estimate), improving Adam's generalization on certain tasks.
- The gain $K_t$ is relatively large in early training (trusting the current gradient more) and gradually decreases in later stages (relying more on historical momentum), which aligns with intuition.
- The theoretical bias-variance analysis (Table 1) serves as the best entry point for understanding the paper's core contributions.

## Highlights & Insights
- **SDE Framework Reveals the Essence of Momentum**: By unifying the analysis of EMA and CM under the stochastic differential equation framework, the work quantifies "parameter drift bias" — a bias that had previously been overlooked.
- **Elegant Correspondence with Optimal Linear Filtering**: The Kalman Filter principle is precisely mapped to gradient estimation — momentum prediction fused with current observation, with the gain determined by the ratio of uncertainties. This signal processing perspective provides a novel theoretical tool for optimizer design.
- **Statistical Interpretation via Gaussian Fusion**: SGDF is equivalent to the multiplicative fusion of two Gaussian distributions, where the fused variance is strictly smaller than either source variance — theoretically guaranteeing a monotonic improvement in estimation quality.

## Limitations & Future Work
- Maintaining $s_t$ and computing $K_t$ at each step incurs modest additional computational and memory overhead.
- The independence assumption (that $\hat{m}_t$ and $g_t$ are independent) does not strictly hold in practice.
- Experiments are primarily validated on CV tasks; effectiveness in NLP/LLM training remains unexplored.
- Whether $\gamma=1/2$ is optimal across all scenarios is an open question; adaptive $\gamma$ could be explored.

## Related Work & Insights
- **vs. Adam**: Adam adapts learning rates via second-order moments, while SGDF adapts the gain via the variance of the first moment — the two methods address different levels of the optimization problem.
- **vs. AdaBelief**: AdaBelief also uses the innovation $(g_t - m_t)^2$ as a variance estimate, but applies it to learning rate scaling; SGDF uses it to compute an optimal fusion gain, with different motivation and effect.
- **vs. Sophia**: Second-order methods rely on Hessian information at high computational cost; SGDF achieves comparable adaptivity using only first-order information.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The SDE analysis under a signal processing perspective and the correspondence with optimal linear filtering are highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple architectures and tasks, but lacks NLP and large-model experiments.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are thorough, though certain sections are somewhat verbose.
- Value: ⭐⭐⭐⭐ Provides both a novel theoretical tool and a practical method for optimizer design.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[AAAI 2026\] A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds](../../AAAI2026/optimization/a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)
- [\[NeurIPS 2025\] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](../../NeurIPS2025/optimization/dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)
- [\[NeurIPS 2025\] Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](../../NeurIPS2025/optimization/effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)
- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](../../NeurIPS2025/optimization/online_two-stage_submodular_maximization.md)

<!-- RELATED:END -->
