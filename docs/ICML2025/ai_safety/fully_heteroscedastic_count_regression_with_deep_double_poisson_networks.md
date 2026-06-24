---
title: >-
  [Paper Note] Fully Heteroscedastic Count Regression with Deep Double Poisson Networks
description: >-
  [ICML 2025][AI Safety][Count Regression] This paper proposes Deep Double Poisson Networks (DDPN), which achieve full heteroscedasticity in discrete count regression by outputting parameters of the Double Poisson distribution. Supporting arbitrarily high or low predictive variances, DDPN comprehensively outperforms existing baselines in accuracy, calibration, and OOD detection.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Count Regression"
  - "Double Poisson Distribution"
  - "Heteroscedastic Regression"
  - "Deep Ensembles"
  - "Uncertainty Estimation"
  - "Out-of-Distribution Detection"
date: 2026-05-08
content_hash: 8fe3ab5d350de80b
---

# Fully Heteroscedastic Count Regression with Deep Double Poisson Networks

**Conference**: ICML 2025  
**arXiv**: [2406.09262](https://arxiv.org/abs/2406.09262)  
**Code**: [GitHub](https://github.com/porterjenkins/deep-double-poisson)  
**Area**: Statistical Modeling/Deep Learning/Uncertainty Quantification  
**Keywords**: Count Regression, Double Poisson Distribution, Heteroscedastic Regression, Deep Ensembles, Uncertainty Estimation, Out-of-Distribution Detection

## TL;DR

This paper proposes Deep Double Poisson Networks (DDPN), which achieve full heteroscedasticity in discrete count regression by outputting parameters of the Double Poisson distribution. Supporting arbitrarily high or low predictive variances, DDPN comprehensively outperforms existing baselines in accuracy, calibration, and OOD detection.

## Background & Motivation

Uncertainty quantification in neural networks is crucial for real-world AI systems, often decomposed into epistemic uncertainty (model parameter uncertainty) and aleatoric uncertainty (observation noise). For continuous regression, Deep Ensembles paired with Gaussian heteroscedastic networks are widely used and highly effective, with the key being that each member can output unrestricted predictive variance.

However, the scenario of **discrete count regression** lacks a similar methodology. Existing approaches face severe limitations:

- **Poisson DNN**: Bound by equi-dispersion, where the mean and variance are tied to the same value $\hat{\lambda} = \hat{\mu} = \hat{\sigma}^2$, preventing independent modeling of variance.
- **Negative Binomial (NB) DNN**: Although breaking equi-dispersion, it is constrained by over-dispersion $\hat{\sigma}^2 \geq \hat{\mu}$, failing to represent variances below the mean (under-dispersion).
- **Gaussian Models**: While possessing full heteroscedasticity, they lead to model misspecification for discrete count data by assigning probability mass to negative and non-integer values, thus lacking correct inductive biases.

The authors point out that full heteroscedasticity is critical for ensemble calibration. The total variance of an ensemble is $\text{Var}[y_i|\mathbf{x}_i] = \mathbb{E}_m[\sigma_m^{2(i)}] + \text{Var}_m[\mu_m^{(i)}]$. If the aleatoric uncertainty term of member models is misspecified, it causes miscalibration in the overall predictive distribution, which consequently degrades epistemic uncertainty estimation.

## Method

### Overall Architecture

DDPN is a neural network that takes arbitrary data $\mathbf{x}_i$ as input, extracts a latent representation $\mathbf{z}_i$ through a shared $L-1$ layer feature extractor, and then outputs the two parameters of the Double Poisson distribution using two independent linear heads:

$$\log(\hat{\mu}_i) = \mathbf{w}_\mu^T \mathbf{z}_i + b_\mu, \quad \log(\hat{\gamma}_i) = \mathbf{w}_\gamma^T \mathbf{z}_i + b_\gamma$$

where $\hat{\mu}_i > 0$ is the predicted mean, and $\hat{\gamma}_i > 0$ is the dispersion parameter. The probability mass function of the Double Poisson distribution is:

$$p(y|\mu, \gamma) = \frac{\gamma^{1/2} e^{-\gamma\mu}}{c(\mu, \gamma)} \cdot \frac{e^{-y} y^y}{y!} \cdot \left(\frac{e\mu}{y}\right)^{\gamma y}$$

Using Efron's moment approximation $\mathbb{E}[Z] = \mu$, $\text{Var}[Z] = \mu / \gamma$, the variance of this distribution can be adjusted to any value by independently tuning $\gamma$: under-dispersion when $\gamma > 1$ (variance less than mean), over-dispersion when $\gamma < 1$ (variance greater than mean), and degenerating to standard Poisson when $\gamma = 1$. This makes DDPN the **first fully heteroscedastic discrete count regression model**.

### Key Designs

#### Key Design 1: Learnable Loss Attenuation

The training objective of DDPN is to minimize the Double Poisson Negative Log-Likelihood (NLL):

$$\mathcal{L}_i = -\frac{\log \hat{\gamma}_i}{2} + \hat{\gamma}_i \hat{\mu}_i - \hat{\gamma}_i y_i (1 + \log \hat{\mu}_i - \log y_i)$$

The authors provide the first formal definition of learnable loss attenuation: a loss function can be decomposed as $\mathcal{L} = d(\hat{\phi}_i) + a(\hat{\phi}_i) \cdot r(\hat{\mu}_i, y_i)$, where $d$ is a divergence penalty term (tending to infinity), $a$ is an attenuation factor (tending to zero), and $r$ is a residual penalty term.

For DDPN, the specific form is:
- $d(\hat{\phi}_i) = \frac{1}{2} \log \hat{\phi}_i$ (divergence penalty)
- $a(\hat{\phi}_i) = 1/\hat{\phi}_i$ (attenuation factor)
- $r(\hat{\mu}_i, y_i) = (\hat{\mu}_i - y_i) - y_i(\log \hat{\mu}_i - \log y_i)$ (residual penalty)

This implies that, similar to Gaussian models, DDPN can adaptively reduce the impact of outliers on the loss by increasing the predicted dispersion, thereby achieving more robust regression.

#### Key Design 2: β-DDPN Controlled Loss Attenuation

Although loss attenuation brings robustness, excessive attenuation may cause the model to "give up" on fitting the mean in hard-to-fit regions, resorting to high uncertainty instead. The authors propose $\beta$-DDPN to modify the loss function:

$$\mathcal{L}_i^{(\beta)} = \lfloor \hat{\gamma}_i^{-\beta} \rfloor \cdot \left(-\frac{\log \hat{\gamma}_i}{2} + \hat{\gamma}_i \hat{\mu}_i - \hat{\gamma}_i y_i(1 + \log \hat{\mu}_i - \log y_i)\right)$$

where $\lfloor \cdot \rfloor$ denotes the stop-gradient operation. The modified partial derivative is:

$$\frac{\partial \mathcal{L}_i^{(\beta)}}{\partial \hat{\mu}_i} = (\hat{\gamma}_i^{1-\beta}) \left(1 - \frac{y_i}{\hat{\mu}_i}\right)$$

When $\beta = 0$, it degenerates to standard NLL; when $\beta = 1$, it completely eliminates the effect of dispersion on the mean gradient, guiding training towards fitting the mean. Experiments demonstrate that larger $\beta$ values speed up mean convergence.

### Ensemble Strategy

An ensemble of $M$ independently trained DDPNs yields a mixture prediction:

$$p(y_i|\mathbf{x}_i) = \frac{1}{M} \sum_{m=1}^M p(y_i | \mathbf{f}_{\Theta_m}(\mathbf{x}_i))$$

The total variance can be decomposed into aleatoric uncertainty (average variance of members) + epistemic uncertainty (variance of member means).

## Key Experimental Results

### Table 1: Accuracy (MAE↓) and Calibration (CRPS↓) on Four Real-World Datasets

| Method | Length of Stay MAE | Length of Stay CRPS | COCO-People MAE | COCO-People CRPS |
|---|---|---|---|---|
| Poisson DNN | 0.664 | 0.553 | 1.099 | 0.851 |
| NB DNN | 0.685 | 0.570 | 1.143 | 0.867 |
| β₀.₅-Gaussian | 0.600 | 0.427 | 1.055 | 0.786 |
| **DDPN** | **0.502** | 0.390 | 1.135 | 0.810 |
| **β₀.₅-DDPN** | 0.516 | **0.370** | 1.095 | 0.782 |
| **β₁.₀-DDPN** | 0.558 | 0.407 | **1.006** | **0.759** |
| DDPN Ensemble | **0.485** | 0.361 | 1.024 | 0.744 |
| β₁.₀-DDPN Ensemble | 0.543 | 0.393 | **0.959** | **0.712** |

### Table 2: OOD Detection Results (Amazon Reviews → Bible Texts)

| Method | AUROC↑ | AUPR↑ | FPR80↓ |
|---|---|---|---|
| Poisson DNN | 0.330 | 0.413 | 0.793 |
| NB DNN | 0.280 | 0.397 | 0.819 |
| Gaussian DNN | 0.840 | 0.812 | 0.318 |
| β₀.₅-Gaussian | — | — | — |
| **DDPN Ensemble** | — | — | — |
| **β-DDPN Ensemble** | **Best** | **Best** | **Best** |

DDPN and its β variants achieve the best performance across all OOD metrics. The AUROC of Poisson and NB DNNs is close to random (0.33 / 0.28), demonstrating that variance-constrained models completely fail to distinguish between in-distribution and out-of-distribution data.

## Key Findings

1. **Full Heteroscedasticity is Crucial**: DDPN or its β variants achieve the best accuracy and calibration across all four datasets (tabular, image, point cloud, text), often with a substantial margin.
2. **Strong Robustness to Misspecification**: Even if the true data originates from a Poisson or Negative Binomial distribution, DDPN recovers the correct distributional structure, performing on par with the correctly matched model.
3. **Ensemble Further Boosts Performance**: DDPN ensembles outperform all baseline ensembles across all metrics, proving that full heteroscedasticity indeed improves epistemic uncertainty estimation.
4. **Effectiveness of β Modification**: Increasing the β value accelerates mean convergence, while simultaneously improving CRPS on most datasets.

## Highlights & Insights

- **Perfect Blend of Theory and Practice**: The paper not only proposes the method but also provides a formal definition of "learnable loss attenuation" and proves DDPN satisfies it, establishing a solid theoretical foundation for discrete regression.
- **Importance of Inductive Biases**: While Gaussian models possess full heteroscedasticity, their inappropriate modeling of discrete data (assigning probability to negative and non-integer values) leads to measurable performance degradation, showing that selecting the correct output distribution family is critical.
- **Simple Yet Universal Method**: DDPN only requires adding an extra output head to the final layer of standard networks, allowing integration into almost any existing architecture with zero overhead.
- **Transfer of β-NLL from Continuous to Discrete**: The Seitzer et al. β-modification is cleverly adapted to the Double Poisson NLL, demonstrating the cross-distribution universality of heteroscedastic regression theory.

## Limitations & Future Work

1. **Unverified High-Count Scenarios**: The paper does not investigate behavior under extremely large count values (e.g., thousands or millions), where Gaussian approximations might suffice.
2. **Boundary Conditions of Moment Approximation**: Efron’s moment approximation degenerates when $\mu_0 \to 0$ and variance is large, although the authors show the error is close to zero in the vast majority of cases.
3. **Training Stability**: SGD and Adam optimizers might converge poorly when training DDPN, requiring the use of AdamW.
4. **Limited OOD Evaluation**: Only one OOD experiment (on Amazon Reviews) was conducted, requiring further verification on more datasets to establish generalizability.

## Related Work & Insights

- **Efron (1986)** proposed the Double Poisson distribution, originally for the GLM framework. This paper introduces it to deep learning.
- **Lakshminarayanan et al. (2017)**'s deep ensembles serve as the cornerstone method for epistemic uncertainty estimation.
- **Seitzer et al. (2022)**'s β-NLL modification solves the over-attenuation issue in Gaussian heteroscedastic regression. This paper adapts it to the discrete setting.
- **Kendall & Gal (2017)** first observed the loss attenuation phenomenon but did not provide a formal definition.

This work inspires an important direction: finding appropriate output distributions with full heteroscedasticity for different data types (count, ordinal, multi-class, etc.) rather than simply relying on Gaussian assumptions.

## Rating

⭐⭐⭐⭐ — Solid theory, comprehensive experiments, and a clean, practical method that fills an important gap in full heteroscedasticity for discrete count regression. The only drawbacks are the limited OOD experiments and lack of validation in high-count scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fisher-Rao Sensitivity for Out-of-Distribution Detection in Deep Neural Networks](../../ICLR2026/ai_safety/fisher-rao_sensitivity_for_out-of-distribution_detection_in_deep_neural_networks.md)
- [\[ICLR 2026\] Fingerprinting Deep Neural Networks for Ownership Protection: An Analytical Approach](../../ICLR2026/ai_safety/fingerprinting_deep_neural_networks_for_ownership_protection_an_analytical_appro.md)
- [\[ICLR 2026\] ULD-Net: Enabling Ultra-Low-Degree Fully Polynomial Networks for Homomorphically Encrypted Inference](../../ICLR2026/ai_safety/uld-net_enabling_ultra-low-degree_fully_polynomial_networks_for_homomorphically_.md)
- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](../../ICCV2025/ai_safety/a_framework_for_doubleblind_federated_adaptation_of_foundati.md)
- [\[ICLR 2026\] Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks](../../ICLR2026/ai_safety/benchmarking_stochastic_approximation_algorithms_for_fairness-constrained_traini.md)

</div>

<!-- RELATED:END -->
