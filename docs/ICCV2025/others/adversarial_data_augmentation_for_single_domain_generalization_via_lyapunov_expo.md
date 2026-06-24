---
title: >-
  [Paper Note] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents
description: >-
  [ICCV 2025][single domain generalization] This paper proposes LEAwareSGD, an optimizer that dynamically adjusts the learning rate using Lyapunov exponents (LE) to guide model training toward the edge of chaos, enabling broader exploration of the parameter space within an adversarial data augmentation framework and achieving significant improvements in single domain generalization (SDG).
tags:
  - "ICCV 2025"
  - "single domain generalization"
  - "adversarial data augmentation"
  - "Lyapunov exponents"
  - "edge of chaos"
  - "optimizer"
date: 2026-05-08
content_hash: 4fa408df88f16258
---

# Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents

**Conference**: ICCV 2025
**arXiv**: [2507.04302](https://arxiv.org/abs/2507.04302)  
**Code**: N/A  
**Area**: Other
**Keywords**: single domain generalization, adversarial data augmentation, Lyapunov exponents, edge of chaos, optimizer

## TL;DR

This paper proposes LEAwareSGD, an optimizer that dynamically adjusts the learning rate using Lyapunov exponents (LE) to guide model training toward the edge of chaos, enabling broader exploration of the parameter space within an adversarial data augmentation framework and achieving significant improvements in single domain generalization (SDG).

## Background & Motivation

Single domain generalization (SDG) aims to train a model on a single source domain that generalizes to unseen target domains. The core challenge lies in insufficient training data diversity and large domain shifts. Existing SDG methods primarily rely on data augmentation techniques:

**Adversarial data augmentation** (ADA, ME-ADA, AdvST, etc.): generates perturbed samples to simulate domain shifts and enhance robustness. However, these perturbations tend to be local and cannot effectively explore the global parameter space, limiting the model's ability to capture generalizable features.

**Generative model-based methods** (PDEN, etc.): expand the source domain distribution via synthetic samples, but incur high computational costs and offer no guarantee on generation quality.

The authors draw inspiration from dynamical systems theory by treating neural network training as a discrete-time dynamical system in parameter space, where each parameter update constitutes a state transition. The **edge of chaos** is a critical state between order and chaos where a system maintains both stability and adaptability. The Lyapunov exponent (LE) is a classical metric for quantifying the degree of chaos — $\text{LE} > 0$ indicates chaos (perturbations grow exponentially), while $\text{LE} < 0$ indicates stability (perturbations decay). At the edge of chaos ($\text{LE} \approx 0^-$), the model neither overfits nor diverges, which is most conducive to learning generalizable features.

Nevertheless, existing optimizers (SGD, Adam, etc.) do not account for the dynamical state of the system and lack mechanisms to actively adjust based on training stability.

## Method

### Overall Architecture

LEAwareSGD integrates Lyapunov exponent computation into adversarial data augmentation optimization: (1) LE is estimated by tracking the propagation of parameter perturbations; (2) the learning rate is dynamically adjusted based on LE changes; (3) joint optimization is performed within the adversarial data augmentation framework.

### Key Designs

1. **LE-based model perturbation analysis**: An initial perturbation $\delta\theta_0$ is introduced, yielding perturbed parameters $\tilde{\theta_t} = \theta_t + \delta\theta_t$. A first-order Taylor expansion gives the perturbation propagation formula:

$$\delta\theta_{t+1} = (I - \eta_t H[L(\theta_t)]) \delta\theta_t$$

Recursively expanding and substituting into the LE definition $LE = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\|\delta\theta_t\|}{\|\delta\theta_0\|}$ establishes the relationship between LE, the learning rate $\eta$, and the Hessian matrix $H$. The LE is jointly determined by these two quantities.

2. **LE-guided learning rate adjustment**: The core innovation. The LE change $\Delta LE_t = LE_t - LE_{t-1}$ is computed at each step:

    - When $\Delta LE_t > 0$ (the model approaches the edge of chaos), the learning rate is decreased to deeply explore that region:
     $\eta_{t+1} = \eta_t \cdot \exp(-\beta \cdot \Delta LE_t)$
    - When $\Delta LE_t \leq 0$, the learning rate is kept unchanged.

   The parameter $\beta$ controls adjustment sensitivity. This design ensures the model slows down to thoroughly explore the region when LE increases (i.e., when it approaches areas rich in generalizable features), rather than passing through rapidly.

3. **Joint optimization with adversarial data augmentation**: A standard minimax framework is adopted — the inner loop maximizes the loss on transformed samples (adversarial augmentation), and the outer loop minimizes the model loss on augmented samples. Weight decay regularization $\frac{\gamma}{2}\|\theta\|_2^2$ is incorporated to ensure the Hessian is approximately positive definite, promoting negative LE values and thus training stability.

### Loss & Training

$$\min_\theta \max_\omega \mathbb{E}_{(x,y)\sim\mathcal{D}_S} [\ell(\theta; \tau(x;\omega), y) - \lambda d_\theta(\tau(x;\omega), x)] + \frac{\gamma}{2}\|\theta\|_2^2$$

- $\ell$: prediction loss
- $d_\theta$: feature distance between original and transformed samples
- $\lambda$: trade-off between adversarial loss and feature consistency
- $\gamma$: weight decay coefficient

Training alternates between generating adversarial samples to simulate domain shifts and updating model parameters with LEAwareSGD.

## Key Experimental Results

### Main Results

Evaluated on three standard SDG benchmarks — PACS, OfficeHome, and DomainNet — using ResNet-18 as the backbone.

| Method | PACS Avg. | OfficeHome Avg. | DomainNet Avg. |
|--------|-----------|-----------------|----------------|
| ERM | 57.80 | 43.60 | 23.77 |
| ADA | 61.11 | 44.75 | 24.26 |
| ME-ADA | 60.22 | 45.35 | 24.63 |
| AdvST | 67.06 | 52.60 | 27.22 |
| PSDG | 67.14 | 47.05 | 26.28 |
| **LEAwareSGD (Ours)** | **69.46** | **54.38** | **28.15** |

*Achieves state-of-the-art results on all three benchmarks; outperforms the second-best method by 2.32% on PACS.*

### Ablation Study

| Optimizer | PACS A | C | P | S | Avg. |
|-----------|--------|---|---|---|------|
| Adam | 76.52 | 71.15 | 64.06 | 53.98 | 66.43 |
| AdamW | 76.68 | 71.93 | 62.03 | 56.68 | 66.83 |
| RMSprop | 71.62 | 71.08 | 59.09 | 47.55 | 62.34 |
| SGD | 76.65 | 74.92 | 62.47 | 54.18 | 67.06 |
| **LEAwareSGD** | **79.17** | **77.16** | **65.05** | **57.78** | **69.46** |

*LEAwareSGD outperforms the SGD baseline by 2.40%; adaptive optimizers such as Adam perform notably worse.*

**Low-data regime** (10% of PACS data): LEAwareSGD achieves 58.73% compared to 49.26% for AdvST, a gain of **9.47%**.

### Key Findings

- LE values converge toward slightly negative values near zero (the edge of chaos) during training, exhibiting smaller and more stable fluctuations compared to other methods.
- LEAwareSGD serves as a plug-and-play component that consistently improves different adversarial augmentation methods: +0.52% for ADA, +2.30% for ME-ADA, +2.40% for AdvST on PACS; +7.00% for ME-ADA on OfficeHome.
- Consistent gains are observed across ResNet backbones (18/34/50/101/152); ResNet-152 achieves 75.34%.
- Computational overhead is only marginally higher than AdvST (1.99h vs. 1.90h on PACS average), and lower than ADA (2.13h).

## Highlights & Insights

- This is the first work to introduce Lyapunov exponents from dynamical systems theory into single domain generalization optimization, establishing a theoretical connection between the edge of chaos and model generalization.
- The method is elegant in its simplicity: it augments standard SGD with a single LE feedback mechanism for learning rate adjustment.
- t-SNE visualizations intuitively demonstrate that LEAwareSGD explores a broader region of the parameter space compared to other methods.

## Limitations & Future Work

- LE computation relies on the Hessian matrix, which may introduce approximation errors in large-scale models.
- Validation is currently limited to classification tasks; extension to detection and segmentation remains to be explored.
- Performance on the DomainNet Quickdraw domain is slightly below SimDE (6.70% vs. 6.85%), suggesting that highly abstract domains may require dedicated augmentation strategies.

## Related Work & Insights

- SAM and GSAM promote generalization by minimizing loss surface sharpness; LEAwareSGD provides a complementary perspective from the viewpoint of dynamical systems.
- The idea of using LE as an optimization feedback signal can be generalized to other training scenarios that require balancing stability and exploration.
- The edge-of-chaos theory offers a novel theoretical framework for understanding what kind of training dynamics are conducive to generalization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Gradient-Guided Annealing for Domain Generalization](../../CVPR2025/others/gradient-guided_annealing_for_domain_generalization.md)
- [\[ICML 2025\] Curvature Enhanced Data Augmentation for Regression](../../ICML2025/others/curvature_enhanced_data_augmentation_for_regression.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](../../ICLR2026/others/noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)
- [\[ICML 2025\] Set-Valued Predictions for Robust Domain Generalization](../../ICML2025/others/set_valued_predictions_for_robust_domain_generalization.md)
- [\[ACL 2025\] Is Linguistically-Motivated Data Augmentation Worth It?](../../ACL2025/others/is_linguistically-motivated_data_augmentation_worth_it.md)

</div>

<!-- RELATED:END -->
