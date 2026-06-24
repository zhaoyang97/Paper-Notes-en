---
title: >-
  [Paper Note] Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective
description: >-
  [ICLR 2026][AI Safety][Differential Privacy] This work introduces the first Stochastic Differential Equation (SDE) framework to analyze differential privacy (DP) optimizers, revealing fundamental differences between DP-SGD and DP-SignSGD under privacy noise. The analysis shows that adaptive methods achieve superior privacy-utility trade-offs of $\mathcal{O}(1/\varepsilon)$ compared to $\mathcal{O}(1/\varepsilon^2)$ in high privacy settings, and their hyperparameters remain tr…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "SDE Analysis"
  - "DP-SGD"
  - "DP-SignSGD"
  - "Privacy-Utility Trade-off"
date: 2026-05-08
content_hash: 2380a6ca42e572b8
---

# Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective

**Conference**: ICLR 2026  
**arXiv**: [2603.03226](https://arxiv.org/abs/2603.03226)  
**Code**: None (uses Google's open-source DP² repository)  
**Area**: AI Security / Differential Privacy Optimization  
**Keywords**: Differential Privacy, SDE Analysis, DP-SGD, DP-SignSGD, Privacy-Utility Trade-off

## TL;DR
This work introduces the first Stochastic Differential Equation (SDE) framework to analyze differential privacy (DP) optimizers, revealing fundamental differences between DP-SGD and DP-SignSGD under privacy noise. The analysis shows that adaptive methods achieve superior privacy-utility trade-offs of $\mathcal{O}(1/\varepsilon)$ compared to $\mathcal{O}(1/\varepsilon^2)$ in high privacy settings, and their hyperparameters remain transferable across varying privacy budgets.

## Background & Motivation

**Background**: Differential Privacy (DP) has become the standard for large-scale private training. DP-SGD protects privacy through per-example gradient clipping and Gaussian noise injection. While adaptive DP optimizers (e.g., DP-Adam) are frequently used in practice, their theoretical understanding remains limited. Existing studies suggest that DP-SGD and DP-Adam perform similarly after exhaustive hyperparameter tuning, leaving the question of which is superior open.

**Limitations of Prior Work**: (1) There is a lack of theoretical characterization of how DP noise interacts with adaptivity; (2) Hyperparameters often need to be re-searched for different privacy budgets $\varepsilon$, consuming additional privacy budget; (3) There is no consensus in the academic community regarding whether adaptive methods hold an advantage under DP.

**Key Challenge**: The mechanisms by which DP noise affects non-adaptive and adaptive methods differ, but existing analyses fail to distinguish these nuances.

**Goal**: (1) Establish SDE models for DP optimizers; (2) Precisely characterize the impact of $\varepsilon$ on convergence rates and asymptotic neighborhoods; (3) Compare performance under fixed hyperparameter and optimal tuning protocols.

**Key Insight**: The SDE weak approximation framework can capture the influence of DP noise on continuous dynamics. SignSGD serves as a theoretical proxy for Adam to facilitate analysis.

**Core Idea**: While the convergence rate of DP-SignSGD depends on $\varepsilon$, its privacy-utility trade-off is only $\mathcal{O}(1/\varepsilon)$. In contrast, while the convergence rate of DP-SGD is independent of $\varepsilon$, its trade-off is $\mathcal{O}(1/\varepsilon^2)$. Consequently, adaptive methods are superior in strict privacy regimes.

## Method

### Overall Architecture
This study does not propose new optimizers but rather constructs continuous-time Stochastic Differential Equation (SDE) models for both DP-SGD and DP-SignSGD. By treating the discrete-iteration privacy noise as a diffusion term, the analysis precisely identifies which part of the convergence dynamics the privacy budget $\varepsilon$ affects. The analysis distinguishes between two phases caused by per-example clipping (Phase 1: all gradients are clipped; Phase 2: gradients are no longer clipped) and defines two comparison protocols: Protocol A (fixing a set of hyperparameters and varying $\varepsilon$) and Protocol B (individually tuning to the optimum for each $\varepsilon$). The former evaluates the essential differences in dynamics, while the latter evaluates the cost of tuning in practical deployments.

### Key Designs

**1. SDE Analysis of DP-SGD: Locking $\varepsilon$ in the Asymptotic Neighborhood**

Does DP noise slow down convergence or merely increase the final error? This is often entangled in discrete analysis, but the SDE framework separates them clearly. Under $\mu$-PL and $L$-smoothness assumptions, the loss trajectory of DP-SGD satisfies $\mathbb{E}[f(X_t)] \lesssim f(X_0)e^{-\mu t} + (1-e^{-\mu t}) \cdot \mathcal{O}(1/\varepsilon^2)$. The first term on the right is the exponentially decaying transient phase, where the decay rate $\mu$ is entirely independent of $\varepsilon$, indicating that the privacy budget does not affect how fast DP-SGD converges. What is truly governed by privacy is the second term (stationary neighborhood), which expands at a rate of $1/\varepsilon^2$ as privacy tightens. In other words, stricter privacy pushes DP-SGD toward a larger error plateau, and the quadratic relationship implies this cost is quite steep.

**2. SDE Analysis of DP-SignSGD: Compressing Quadratic Costs to Linear with the Sign Operator**

Applying the same SDE tools to adaptive methods results in a qualitatively opposite conclusion. The loss for DP-SignSGD satisfies $\mathbb{E}[f(X_t)] \lesssim f(X_0)e^{-c\varepsilon t} + (1-e^{-c\varepsilon t}) \cdot \mathcal{O}(1/\varepsilon)$. The $\varepsilon$ dependencies of the two terms are exactly swapped compared to DP-SGD: the decay rate $c\varepsilon$ is linearly proportional to the privacy budget, meaning smaller $\varepsilon$ leads to slower convergence. However, the stationary neighborhood scales only as $\mathcal{O}(1/\varepsilon)$, which is an order of magnitude more moderate than the quadratic term in DP-SGD. The root of this difference lies in the compression of noise by the sign operation—taking only the gradient sign discards the magnitude information of the DP noise. In expectation, $\mathbb{E}[\text{sign}(g_k)] \approx \nabla f(x)/(\sigma_\gamma\sqrt{d})$, where the directional signal is preserved while the noise is normalized. Thus, the impact of privacy noise on the final error is reduced from quadratic to linear. The trade-off is slower convergence, but in high-privacy (small $\varepsilon$) regions, a lower error plateau is far more important than slightly slower convergence.

**3. Hyperparameter Transferability Across Privacy Budgets: Decoupling Optimal Learning Rate from $\varepsilon$**

Protocol B further explores whether differences remain if tuning is allowed for each privacy budget. The derived optimal learning rate provides the answer—for DP-SGD, $\eta^\star \propto \varepsilon$, meaning the learning rate must be re-searched whenever the privacy budget changes. Conversely, for DP-SignSGD, $\eta^\star$ is independent of $\varepsilon$, allowing a single learning rate to work across all privacy levels. While their asymptotic performance can be matched under their respective optimal learning rates, this highlights the practical advantage of adaptive methods: since every hyperparameter search in DP training consumes extra privacy budget, the $\varepsilon$-insensitive nature of DP-SignSGD eliminates the overhead of repeated tuning. This insight, verified experimentally, can be directly transferred to DP-Adam, as SignSGD is a proxy for Adam in theoretical analysis.

### Loss & Training
The theory is established under $\mu$-PL or $L$-smooth loss assumptions. Training follows the standard DP pipeline—per-example gradient clipping combined with Gaussian noise injection. Empirical validation is conducted on two types of problems: quadratic convex functions (to test the precision of SDE predictions) and logistic regression on IMDB and StackOverflow (to test scaling laws on real data). The validity of the sign proxy is confirmed by reproducing DP-SignSGD results using DP-Adam.

## Key Experimental Results

### Main Results (Privacy-Utility Trade-off Validation)

| Method | Privacy-Utility Scaling | Convergence Rate vs. $\varepsilon$ | $\eta^\star$ vs. $\varepsilon$ |
|--------|------|------|----------|
| DP-SGD | $\mathcal{O}(1/\varepsilon^2)$ | Independent of $\varepsilon$ | $\eta^\star \propto \varepsilon$ |
| DP-SignSGD | $\mathcal{O}(1/\varepsilon)$ | Linearly dependent on $\varepsilon$ | Independent of $\varepsilon$ |
| DP-Adam | $\approx \mathcal{O}(1/\varepsilon)$ | Consistent with DP-SignSGD | Consistent with DP-SignSGD |

### Ablation Study (Impact of Batch Noise - IMDB Dataset)

| Batch Size $B$ | DP-SignSGD Advantage Threshold $\varepsilon^\star$ | Description |
|------|---------|------|
| 48 | Large | High batch noise; DP-SignSGD is always superior |
| 64 | Moderate | Transition region |
| 80 | Small | Low batch noise; DP-SignSGD superior only under strict privacy |

### Key Findings
- On quadratic functions, theoretical predictions perfectly match experimental values, validating the precision of the SDE analysis.
- On IMDB and StackOverflow, the $1/\varepsilon^2$ scaling for DP-SGD and $1/\varepsilon$ scaling for DP-SignSGD hold for both training and test loss.
- When batch noise is sufficiently large, DP-SignSGD outperforms DP-SGD across all values of $\varepsilon$; when batch noise is low, a critical threshold $\varepsilon^\star$ exists.
- The behavior of DP-Adam is qualitatively consistent with DP-SignSGD, validating the use of SignSGD as a proxy for Adam.

## Highlights & Insights
- This work is the first to introduce SDE tools into the analysis of DP optimization, revealing structural differences between privacy noise and adaptivity that previous discrete analyses could not capture.
- The practical implications are clear: DP-Adam/DP-SignSGD should be prioritized in high-privacy settings, not only for superior asymptotic performance but also because hyperparameters are transferable across $\varepsilon$, saving the privacy budget otherwise spent on tuning.

## Limitations & Future Work
- The theory only covers DP-SGD and DP-SignSGD and does not directly analyze DP-Adam (relying instead on empirical extensions using SignSGD as a proxy).
- Experiments are limited to logistic regression and simple convex problems; validation on deep networks is not yet sufficiently comprehensive.
- The assumption that gradient noise follows Gaussian or Student-t distributions may not fully reflect the more complex noise structures in actual deep learning.

## Related Work & Insights
- **vs. Li et al. (2022b)**: That work found DP-SGD and DP-Adam perform similarly in LLM fine-tuning (Protocol B). The Protocol B theory in this paper is consistent but further points out that DP-Adam has a fundamental advantage in tuning practicality.
- **vs. Jin & Dai (2025)**: Analyzed Noisy SignSGD from a privacy amplification perspective but did not consider clipping. This paper provides a complete treatment of per-example clipping.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First SDE analysis of DP optimizers; solid theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ Experiments are relatively simple (logistic regression); lacks sufficient deep network validation.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivation, consistent notation, and highly informative charts.
- Value: ⭐⭐⭐⭐ Provides a theoretical basis for selecting DP optimizers, offering guidance for private ML practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[ICLR 2026\] Optimizing Canaries for Privacy Auditing with Metagradient Descent](optimizing_canaries_for_privacy_auditing_with_metagradient_descent.md)
- [\[ICLR 2026\] Concept-based Adversarial Attack: a Probabilistic Perspective](concept-based_adversarial_attack_a_probabilistic_perspective.md)
- [\[ICLR 2026\] Nasty Adversarial Training: A Probability Sparsity Perspective for Robustness Enhancement](nasty_adversarial_training_a_probability_sparsity_perspective_for_robustness_enh.md)
- [\[ICLR 2026\] Fairness-Aware Multi-view Evidential Learning with Adaptive Prior](fairness-aware_multi-view_evidential_learning_with_adaptive_prior.md)

</div>

<!-- RELATED:END -->
