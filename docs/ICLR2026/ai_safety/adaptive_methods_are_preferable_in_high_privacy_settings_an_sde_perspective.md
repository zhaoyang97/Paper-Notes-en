---
title: >-
  [Paper Note] Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective
description: >-
  [ICLR 2026][AI Safety][Differential Privacy] This work is the first to analyze differentially private optimizers through a stochastic differential equation (SDE) framework, revealing fundamental behavioral differences between DP-SGD and DP-SignSGD under privacy noise: adaptive methods achieve a superior privacy-utility tradeoff of $\mathcal{O}(1/\varepsilon)$ vs. $\mathcal{O}(1/\varepsilon^2)$ in high-privacy regimes, and their hyperparameters transfer across privacy budgets.
tags:
  - ICLR 2026
  - AI Safety
  - Differential Privacy
  - SDE Analysis
  - DP-SGD
  - DP-SignSGD
  - Privacy-Utility Tradeoff
date: 2026-05-08
content_hash: 5c1852bbadb7a3f7
---

# Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective

**Conference**: ICLR 2026
**arXiv**: [2603.03226](https://arxiv.org/abs/2603.03226)
**Code**: None (uses Google's open-source DP² repository)
**Area**: AI Safety / Differentially Private Optimization
**Keywords**: Differential Privacy, SDE Analysis, DP-SGD, DP-SignSGD, Privacy-Utility Tradeoff

## TL;DR
This work is the first to analyze differentially private optimizers through a stochastic differential equation (SDE) framework, revealing fundamental behavioral differences between DP-SGD and DP-SignSGD under privacy noise: adaptive methods achieve a superior privacy-utility tradeoff of $\mathcal{O}(1/\varepsilon)$ vs. $\mathcal{O}(1/\varepsilon^2)$ in high-privacy regimes, and their hyperparameters transfer across privacy budgets.

## Background & Motivation

**Background**: Differential privacy (DP) has become the standard for large-scale private training. DP-SGD protects privacy via per-example gradient clipping and Gaussian noise injection. Adaptive DP optimizers (e.g., DP-Adam) are widely used in practice but remain theoretically underexplored. Prior work suggests DP-SGD and DP-Adam perform comparably under careful tuning, and which is superior remains an open question.

**Limitations of Prior Work**: (1) The interaction between DP noise and adaptivity lacks theoretical characterization; (2) hyperparameters must be re-searched for different privacy budgets $\varepsilon$, consuming additional privacy budget; (3) there is no consensus on whether adaptive methods confer advantages under DP.

**Key Challenge**: The mechanisms by which DP noise acts differ fundamentally between non-adaptive and adaptive methods, yet existing analyses cannot distinguish this difference.

**Goal**: (1) Establish SDE models for DP optimizers; (2) precisely characterize the effect of $\varepsilon$ on convergence rate and asymptotic neighborhood; (3) compare performance under fixed-hyperparameter and optimally-tuned protocols.

**Key Insight**: The SDE weak approximation framework can capture the effect of DP noise on continuous dynamics; SignSGD serves as a tractable theoretical proxy for Adam.

**Core Idea**: Although DP-SignSGD's convergence rate depends on $\varepsilon$, its privacy-utility tradeoff is only $\mathcal{O}(1/\varepsilon)$, whereas DP-SGD converges at a rate independent of $\varepsilon$ but incurs an $\mathcal{O}(1/\varepsilon^2)$ tradeoff. Adaptive methods are therefore preferable under strict privacy constraints.

## Method

### Overall Architecture
The paper derives continuous-time SDE models for DP-SGD and DP-SignSGD using SDE weak approximation theory. Two phases induced by per-example clipping are considered (Phase 1: all gradients clipped; Phase 2: no clipping), and convergence bounds are derived for each. Two analytical protocols are designed: Protocol A (fixed hyperparameters, varying $\varepsilon$) and Protocol B (independent hyperparameter tuning per $\varepsilon$).

### Key Designs

1. **SDE Analysis of DP-SGD (Protocol A)**:

    - **Function**: Characterizes the effect of privacy budget $\varepsilon$ on DP-SGD under fixed hyperparameters.
    - **Mechanism**: Under $\mu$-PL and $L$-smoothness conditions, it is proved that the DP-SGD loss satisfies $\mathbb{E}[f(X_t)] \lesssim f(X_0)e^{-\mu t} + (1-e^{-\mu t}) \cdot \mathcal{O}(1/\varepsilon^2)$. The decay term (convergence rate) is independent of $\varepsilon$, while the asymptotic neighborhood (privacy-utility term) scales as $1/\varepsilon^2$.
    - **Design Motivation**: Separating convergence rate from the asymptotic neighborhood precisely reveals that $\varepsilon$ affects only the latter.

2. **SDE Analysis of DP-SignSGD (Protocol A)**:

    - **Function**: Reveals fundamentally different behavior of adaptive methods under DP.
    - **Mechanism**: It is proved that the DP-SignSGD loss satisfies $\mathbb{E}[f(X_t)] \lesssim f(X_0)e^{-c\varepsilon t} + (1-e^{-c\varepsilon t}) \cdot \mathcal{O}(1/\varepsilon)$. The key distinction is that the decay term depends linearly on $\varepsilon$ (slow convergence at small $\varepsilon$), but the asymptotic neighborhood is only $\mathcal{O}(1/\varepsilon)$. This exploits the noise-compression effect of the sign operation: $\mathbb{E}[\text{sign}(g_k)] \approx \nabla f(x)/(\sigma_\gamma\sqrt{d})$.
    - **Design Motivation**: The sign operator inherently compresses noise magnitude, reducing the impact of DP noise from quadratic to linear.

3. **Hyperparameter Transfer Across Privacy Budgets (Protocol B)**:

    - **Function**: Compares asymptotic performance and hyperparameter sensitivity of both methods under optimal tuning.
    - **Mechanism**: Optimal learning rates are derived — DP-SGD requires $\eta^\star \propto \varepsilon$ (privacy-budget-dependent), whereas DP-SignSGD's $\eta^\star$ is independent of $\varepsilon$. Under optimal learning rates, asymptotic performance is comparable, but DP-SignSGD requires no re-tuning across different $\varepsilon$ values.
    - **Design Motivation**: In practice, hyperparameter search consumes additional privacy budget; methods insensitive to $\varepsilon$ are therefore more practical.

### Loss & Training
The theoretical analysis assumes $\mu$-PL or $L$-smooth loss functions. Experiments validate the theory on quadratic convex functions and logistic regression on IMDB and StackOverflow. Standard DP training with per-example clipping and Gaussian noise injection is used. Theoretical insights from DP-SignSGD are empirically extended to DP-Adam.

## Key Experimental Results

### Main Results (Privacy-Utility Tradeoff Verification)

| Method | Privacy-Utility Scaling | Convergence Rate vs. $\varepsilon$ | $\eta^\star$ vs. $\varepsilon$ |
|--------|------|------|----------|
| DP-SGD | $\mathcal{O}(1/\varepsilon^2)$ | Independent of $\varepsilon$ | $\eta^\star \propto \varepsilon$ |
| DP-SignSGD | $\mathcal{O}(1/\varepsilon)$ | Linear in $\varepsilon$ | Independent of $\varepsilon$ |
| DP-Adam | $\approx \mathcal{O}(1/\varepsilon)$ | Consistent with DP-SignSGD | Consistent with DP-SignSGD |

### Ablation Study (Effect of Batch Noise — IMDB Dataset)

| Batch Size $B$ | Threshold $\varepsilon^\star$ for DP-SignSGD Advantage | Notes |
|------|---------|------|
| 48 | Large | High batch noise; DP-SignSGD dominates throughout |
| 64 | Moderate | Transition regime |
| 80 | Small | Low batch noise; DP-SignSGD superior only under strict privacy |

### Key Findings
- On quadratic functions, theoretical predictions match experimental values exactly, validating the precision of the SDE analysis.
- On IMDB and StackOverflow, the $1/\varepsilon^2$ and $1/\varepsilon$ scalings for DP-SGD and DP-SignSGD respectively hold for both training and test loss.
- When batch noise is sufficiently large, DP-SignSGD outperforms DP-SGD across all $\varepsilon$; when batch noise is small, a critical threshold $\varepsilon^\star$ exists.
- DP-Adam behaves qualitatively consistently with DP-SignSGD, validating the use of SignSGD as a proxy for Adam.

## Highlights & Insights
- This work is the first to introduce SDE tools into DP optimization analysis, revealing structural differences between privacy noise and adaptivity that are invisible to all prior discrete-time analyses.
- The practical implications are clear: DP-Adam/DP-SignSGD should be preferred under strict privacy constraints — not only for superior asymptotic performance, but also because their hyperparameters transfer across $\varepsilon$ values, reducing the privacy cost of hyperparameter tuning.

## Limitations & Future Work
- The theory covers only DP-SGD and DP-SignSGD; DP-Adam is not directly analyzed (its treatment relies on empirical extension via the SignSGD proxy).
- Experiments are limited to logistic regression and simple convex problems; validation on deep networks is insufficient.
- The analysis assumes gradient noise follows Gaussian or Student-t distributions; the noise structure in practical deep learning may be more complex.

## Related Work & Insights
- **vs. Li et al. (2022b)**: That work finds comparable performance between DP-SGD and DP-Adam in LLM fine-tuning (Protocol B); the present paper's Protocol B theory is consistent, but additionally identifies a fundamental practical advantage of DP-Adam in terms of hyperparameter usability.
- **vs. Jin & Dai (2025)**: That work analyzes Noisy SignSGD from a privacy amplification perspective without accounting for clipping; this paper fully handles per-example clipping.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First SDE analysis of DP optimizers; solid theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ Experiments are relatively simple (logistic regression); deep network validation is insufficient.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are rigorous, notation is consistent, and figures are highly informative.
- **Value**: ⭐⭐⭐⭐ Provides theoretical justification for DP optimizer selection; offers practical guidance for privacy-aware ML.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[ICLR 2026\] Why Do Unlearnable Examples Work: A Novel Perspective of Mutual Information](why_do_unlearnable_examples_work_a_novel_perspective_of_mutual_information.md)
- [\[ICLR 2026\] Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization](unified_privacy_guarantees_for_decentralized_learning_via_matrix_factorization.md)
- [\[CVPR 2026\] A Unified Perspective on Adversarial Membership Manipulation in Vision Models](../../CVPR2026/ai_safety/a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)
- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](../../NeurIPS2025/ai_safety/mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)

</div>

<!-- RELATED:END -->
