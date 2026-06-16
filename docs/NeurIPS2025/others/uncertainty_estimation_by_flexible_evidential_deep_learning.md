---
title: >-
  [Paper Note] Uncertainty Estimation by Flexible Evidential Deep Learning
description: >-
  [NeurIPS 2025][uncertainty quantification] This paper proposes $\mathcal{F}$-EDL, which generalizes the Dirichlet distribution in EDL to a Flexible Dirichlet (FD) distribution for modeling class probability distributions…
tags:
  - "NeurIPS 2025"
  - "uncertainty quantification"
  - "evidential deep learning"
  - "Flexible Dirichlet distribution"
  - "OOD detection"
  - "single forward pass"
date: 2026-05-08
content_hash: 8873f9c497589c74
---

# Uncertainty Estimation by Flexible Evidential Deep Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.18322](https://arxiv.org/abs/2510.18322)  
**Code**: [Available](https://github.com/TaeseongYoon/F-EDL)  
**Area**: Uncertainty Quantification
**Keywords**: uncertainty quantification, evidential deep learning, Flexible Dirichlet distribution, OOD detection, single forward pass

## TL;DR

This paper proposes $\mathcal{F}$-EDL, which generalizes the Dirichlet distribution in EDL to a Flexible Dirichlet (FD) distribution for modeling class probability distributions. This approach significantly enhances the generalization of uncertainty estimation under complex scenarios such as noise, long-tail distributions, and distribution shift, while preserving the efficiency of a single forward pass.

## Background & Motivation

Uncertainty quantification (UQ) is critical for deploying ML models in high-stakes applications such as autonomous driving and medical diagnosis. Effective UQ methods must simultaneously satisfy two requirements: (1) computational efficiency for real-time systems, and (2) strong generalization across diverse scenarios.

Classical UQ methods (Bayesian neural networks, MC Dropout, deep ensembles) are well-established but require multiple forward passes, incurring substantial computational cost. Evidential Deep Learning (EDL) quantifies uncertainty by predicting a Dirichlet distribution over class probabilities, requiring only a single forward pass and thus offering efficiency advantages.

However, the core assumption of EDL—that class probabilities follow a Dirichlet distribution—limits the model's expressive capacity. In noisy data (DMNIST) experiments, EDL produces severe overlap between in-distribution noisy data (AMNIST) and out-of-distribution data (FMNIST), failing to distinguish the two effectively. The authors argue that the unimodal nature of the Dirichlet distribution is the root cause of EDL's insufficient robustness in complex scenarios, motivating the development of a more flexible yet equally efficient UQ method.

## Method

### Overall Architecture

$\mathcal{F}$-EDL replaces the Dirichlet distribution in EDL with a Flexible Dirichlet (FD) distribution. A shared feature extractor $f_\theta$ and three prediction heads are used to predict the three parameter groups of the FD distribution: concentration parameters $\boldsymbol{\alpha}$, allocation probabilities $\mathbf{p}$, and dispersion $\tau$. The framework consists of three core components: model architecture, objective function, and a label-variance-based uncertainty measure.

### Key Designs

1. **Flexible Dirichlet Distribution**: The FD distribution is a generalization of the Dirichlet distribution, obtained by normalizing Flexible Gamma bases. It is constructed as $Y_k = W_k + Z_k U$, where $W_k \sim \text{Gamma}(\alpha_k)$ are independent Gamma variables, $U \sim \text{Gamma}(\tau)$ is a shared stochastic component, and $\mathbf{Z} \sim \text{Mu}(1, \mathbf{p})$ follows a multinomial distribution. The FD distribution can be expressed as a mixture of Dirichlet distributions and exhibits multimodal properties, enabling it to capture complex uncertainty patterns.

2. **Three-Head Prediction Structure**: Starting from features $\mathbf{z} = f_\theta(\mathbf{x})$, three neural network heads predict: $\boldsymbol{\alpha} = \exp(g_{\phi_1}(\mathbf{z}))$ (concentration parameters, with exp activation ensuring non-negativity), $\mathbf{p} = \text{softmax}(g_{\phi_2}(\mathbf{z}))$ (allocation probabilities), and $\tau = \text{softplus}(g_{\phi_3}(\mathbf{z}))$ (dispersion). Spectral normalization is applied to the feature extractor and the $\alpha$ prediction head to enforce Lipschitz continuity.

3. **Multimodal Class Probability Distribution (Theorem 4.4)**: The class probability distribution of $\mathcal{F}$-EDL is a Dirichlet mixture: $p_{\mathcal{F}\text{-EDL}}(\boldsymbol{\pi}|\mathbf{x}^*) = \sum_{k=1}^K p_k \text{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha} + \tau \mathbf{e}_k)$, with the number of modes determined by $\|\mathbf{p}\|_0$. This allows the model to express complex uncertainty corresponding to ambiguity among multiple plausible classes.

4. **EDL–Softmax Mixture Decomposition (Theorem 4.5)**: The predictive distribution of $\mathcal{F}$-EDL decomposes into an input-adaptive mixture of EDL and Softmax: $p_{\mathcal{F}\text{-EDL}}(y|\mathbf{x}^*) = w_{\text{EDL}} \cdot p_{\text{EDL}} + w_{\text{SM}} \cdot p_{\text{SM}}$, with weights $w_{\text{EDL}} = \alpha_0/(\alpha_0+\tau)$ and $w_{\text{SM}} = \tau/(\alpha_0+\tau)$ depending on the input. For clean in-distribution data, EDL dominates; for ambiguous or OOD data, the model interpolates between the two components.

### Loss & Training

The objective function comprises two terms:

$$\mathcal{L} = \mathbb{E}_{\boldsymbol{\pi} \sim \text{FD}} [\|\mathbf{y} - \boldsymbol{\pi}\|_2^2] + \|\mathbf{y} - \mathbf{p}\|_2^2$$

The first term is the expected MSE under the FD distribution, computed analytically using closed-form moments of the FD distribution without requiring sampling. The second term is a Brier score regularization that promotes input-dependent calibration of $\mathbf{p}$ and prevents degenerate solutions. Compared to the KL divergence regularization used in conventional EDL, this loss function reduces sensitivity to hyperparameters.

Uncertainty is measured via a label-variance-based approach, which decomposes predictive uncertainty into aleatoric uncertainty (AU) and epistemic uncertainty (EU) through the law of total variance.

## Key Experimental Results

### Main Results

**CIFAR-10/100 Standard Setting (Table 1)**:

| Method | CIFAR-10 Acc | CIFAR-10 OOD (SVHN) | CIFAR-100 Acc | CIFAR-100 OOD (SVHN) |
|--------|-------------|---------------------|---------------|---------------------|
| EDL | 83.55 | 79.12 | 45.91 | 56.21 |
| I-EDL | 89.20 | 82.96 | 66.38 | 67.51 |
| R-EDL | 90.09 | 85.00 | 63.53 | 61.80 |
| DAEDL | 91.11 | 85.54 | 66.01 | 72.07 |
| **F-EDL** | **91.19** | **91.20** | **69.40** | **75.35** |

**Noisy Setting DMNIST (Table 4)**:

| Method | Test Acc | Misclassification Detection (Conf.) | OOD Detection (FMNIST) |
|--------|----------|-------------------------------------|------------------------|
| DDU | 84.05 | 82.73 | 98.49 |
| DAEDL | 84.12 | 95.93 | 99.44 |
| **F-EDL** | **84.28** | **96.17** | **99.76** |

### Ablation Study

**FD Parameter Ablation (Table 5, DMNIST)**:

| Configuration | Test Acc | OOD Detection (FMNIST) | Note |
|---------------|----------|------------------------|------|
| Fix-p(U), τ | 83.34 | 97.22 | Fixed uniform p + fixed τ=1 |
| Fix-p(N), τ | 83.27 | 97.91 | Fixed normalized-α p + fixed τ=1 |
| Fix-τ | 83.39 | 98.46 | Only τ fixed to 1 |
| **F-EDL (full)** | **84.28** | **99.76** | Joint learning of p and τ |

### Key Findings

- F-EDL improves OOD detection on CIFAR-10 (SVHN) by approximately 5.7 percentage points over DAEDL.
- Under long-tail settings (CIFAR-10-LT, ρ=0.1), F-EDL also achieves the best OOD detection performance.
- F-EDL's epistemic uncertainty decreases monotonically as training data increases, consistent with theoretical expectations, whereas EDL and DAEDL exhibit inconsistent behavior.
- Inference speed is only 1.3% slower than EDL, while being more than 50% faster than DAEDL.

## Highlights & Insights

- Theoretical completeness: five theorems are proved, covering the conjugate prior property of the FD distribution for categorical likelihoods, the strict generalization of F-EDL over EDL, multimodality, and the EDL–Softmax mixture decomposition.
- The multimodal visualization is compelling: for ambiguous inputs (e.g., digits 9/7), F-EDL produces bimodal distributions, whereas EDL collapses to a unimodal, overconfident prediction.
- Additional parameter overhead is minimal (only 1.8% increase for VGG-16), with virtually no extra inference cost.

## Limitations & Future Work

- The method is currently limited to classification tasks; extension to regression is a natural direction.
- Disentanglement of aleatoric and epistemic uncertainty remains incomplete.
- The approach still relies on external regularization to control epistemic uncertainty, lacking an intrinsically stable training objective.
- Validation on large-scale datasets (e.g., ImageNet) has not been conducted.

## Related Work & Insights

- The method is complementary to the feature-space density approach of DAEDL; introducing the FD distribution into density estimation pipelines is worth exploring.
- Combining logit adjustment with F-EDL may further improve performance in long-tail scenarios.
- The FD distribution could replace Dirichlet components in trustworthy fusion tasks such as multi-view learning.

## Rating

- Novelty: ⭐⭐⭐⭐ (First application of the FD distribution to UQ, though the core idea is a generalization of a known distribution)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Covers standard, long-tail, noisy, and distribution-shift settings with sufficient ablation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Excellent integration of theory and experiments, clear structure)
- Value: ⭐⭐⭐⭐ (Constitutes a substantive advance within the EDL framework)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Asymmetric Duos: Sidekicks Improve Uncertainty](asymmetric_duos_sidekicks_improve_uncertainty.md)
- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](../../ICML2026/others/possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[NeurIPS 2025\] Position: There Is No Free Bayesian Uncertainty Quantification](position_there_is_no_free_bayesian_uncertainty_quantification.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[NeurIPS 2025\] Deep Legendre Transform](deep_legendre_transform.md)

</div>

<!-- RELATED:END -->
