---
title: >-
  [Paper Note] Position: There Is No Free Bayesian Uncertainty Quantification
description: >-
  [NeurIPS 2025][Bayesian Inference] This paper challenges the validity of Bayesian uncertainty quantification (UQ) from a frequentist perspective…
tags:
  - "NeurIPS 2025"
  - "Bayesian Inference"
  - "Uncertainty Quantification"
  - "Frequentist"
  - "PAC-Bayes"
  - "Prediction Intervals"
date: 2026-05-08
content_hash: 4e1760d12e8addb2
---

# Position: There Is No Free Bayesian Uncertainty Quantification

**Conference**: NeurIPS 2025
**arXiv**: [2506.03670](https://arxiv.org/abs/2506.03670)  
**Code**: None  
**Area**: Statistical Learning Theory / Uncertainty Quantification
**Keywords**: Bayesian Inference, Uncertainty Quantification, Frequentist, PAC-Bayes, Prediction Intervals

## TL;DR

This paper challenges the validity of Bayesian uncertainty quantification (UQ) from a frequentist perspective, reinterprets Bayesian updating as an optimization problem over model ensembles, and proposes a PAC-framework-based calibration algorithm for constructing prediction intervals with frequentist guarantees.

## Background & Motivation

Bayesian methods are widely adopted in machine learning and deep learning for their intuitive uncertainty quantification capabilities. Given a prior distribution over the parameter space, a posterior distribution can be directly obtained and is commonly interpreted as the model's uncertainty quantification. However, the authors identify fundamental issues with this interpretation:

**Difficulty in Prior Selection**: In overparameterized modern ML settings where the number of parameters far exceeds the amount of data, the prior exerts a substantial influence on the posterior, yet no principled guidance for prior selection exists.

**Inapplicability of the Bernstein–von Mises Theorem**: This theorem guarantees that the posterior asymptotically converges to a Gaussian centered at the MLE in finite-dimensional parameter spaces, but in ML settings where the data-to-parameter ratio is small or even less than one, the theorem's applicability is questionable.

**No Frequentist Coverage Guarantee**: Wenzel et al. (2020) have shown that Bayesian neural networks exhibit improved predictive performance under *cold posteriors*, violating the information-processing optimality implied by standard Bayesian updating.

**Limitations of Frequentist Alternatives**: Methods such as the bootstrap are also ill-suited to modern ML due to high training costs and limited data availability.

## Method

### Overall Architecture

The paper's central argument builds on the optimization perspective of Knoblauch et al. (2022), reformulating Bayesian updating as:

$$\min_{p \in \mathcal{P}} \{ E_{f \sim p}[l(f(x), y)] + D(p \| \pi_0) \}$$

where $\mathcal{P}$ is the space of probability measures over the function class, $l$ is a loss function, $\pi_0$ is the prior distribution, and $D$ is a divergence measure. When the negative log-likelihood is used as the loss and KL divergence as $D$, the solution coincides exactly with the Bayesian posterior.

**Key Insight**: This optimization problem contains no explicit uncertainty quantification — it merely minimizes expected loss in the vicinity of the prior. Consequently, any uncertainty claims derived from the posterior are open to question.

### Key Designs

#### 1. Prior Quality Assessment

The authors propose three frequentist metrics for evaluating prior quality:

**Average Quality**:
$$Q(\pi_0) := \mathbb{E}_{X^t, Y^t} \mathbb{E}_{Y, X^*}[\mathbb{1}_K(Y)]$$

Measures the average probability that the prediction interval covers the true value, taken over all possible training sets and new data points.

**Worst-case Quality**:
$$Q'(\pi_0) := \inf_{X^t, Y^t} \mathbb{E}_{Y, X^*}[\mathbb{1}_K(Y)]$$

Measures coverage under the least favorable training set, applicable to settings with low tolerance for uncertainty.

**Probabilistic Quality**:
$$Q''(\pi_0) := \mathbb{P}((X^t, Y^t) \text{ s.t. } \mathbb{E}_{Y, X^*}[\mathbb{1}_K(Y)]) \geq 1 - \alpha$$

Measures the fraction of possible datasets that achieve valid coverage, serving as a compromise between the two preceding metrics.

#### 2. Prediction Interval Calibration Algorithm

Given an existing model ensemble $p(f)$, the predictive distribution for a new input $X^* = x^*$ is:

$$P(Y^* | X^* = x^*) = \int P(Y^* | X^* = x^*, f) dp(f)$$

The algorithm employs a **quantile estimation dataset** $(X_i^v, Y_i^v)_{i=1}^m$ and reformulates the problem as binary classification: whether an observation falls within the prediction interval. The empirical risk is defined as:

$$\hat{R}((X^v, Y^v), q) := \frac{1}{m} \sum_{i=1}^m l(X_i^v, Y_i^v, q)$$

A grid search identifies the optimal quantile $\hat{q}$ such that $\hat{R} \leq \alpha$.

### Loss & Training

A 0–1 loss function is employed: zero loss when the observation falls within the interval, unit loss otherwise. The PAC framework guarantees that the calibrated prediction interval satisfies frequentist coverage requirements with high probability $1-\epsilon$:

$$\mathbb{P}((X^v, Y^v) \text{ s.t. } R((X,Y), \hat{q}) \leq \alpha + C(\epsilon)) \geq 1 - \epsilon(n)$$

where $C(\epsilon)$ decreases as the sample size $n$ grows. Since only a single parameter $q$ is estimated, the generalization gap is expected to be small.

## Key Experimental Results

### Main Results: Simulation Study 1 (Linear Regression, Correctly Specified Model)

| Prior Mean $i$ | Naïve Coverage | Calibrated Coverage | Target Coverage |
|:---:|:---:|:---:|:---:|
| $i \in [-5, 5]$ | 0.50–0.85 (below target) | **≈ 0.90** (on target) | 0.90 |
| $i \in [-10, -5) \cup (5, 10]$ | 0.50–0.65 | 0.70–0.85 (degraded by numerical limits) | 0.90 |
| $i = 0$ (optimal prior) | ≈ 0.85 | **≈ 0.90** | 0.90 |

- Setting: 20 parameters, 30 training samples, 300 test samples
- Prior: $\beta \sim \mathcal{N}_{20}(\mu = i \cdot \mathbf{1}, \Sigma = 2 \cdot I_{20})$, $i \in \{-10, \ldots, 10\}$
- The naïve method **never achieves** the target coverage of 0.90
- The calibrated method successfully meets the target for $i \in [-5, 5]$
- Calibrated intervals are wider but shrink as the prior moves closer to the true value

### Ablation Study: Simulation Study 2 (Misspecified Model)

| Missing Variable Coefficient $\beta_{20}$ | Naïve Coverage | Calibrated Coverage | Remarks |
|:---:|:---:|:---:|:---|
| $\beta_{20} = 1$ (mild misspecification) | 0.50–0.80 | **≈ 0.90** (on target) | Calibration remains effective |
| $\beta_{20} = 3$ (severe misspecification) | 0.40–0.70 | 0.60–0.80 (below target) | Both methods fail; calibration still outperforms naïve |

- Setting: identical to Study 1, with one additional unobservable feature
- Under mild misspecification ($\beta_{20}=1$), the calibrated method remains effective
- Under severe misspecification ($\beta_{20}=3$), the predictive distribution assigns negligible probability mass to the observed data region, causing numerical precision failures

### Key Findings

1. **Naïve Bayesian prediction intervals never achieve correct frequentist coverage**, particularly when the prior deviates from the true parameter values.
2. **The calibration method substantially improves coverage**, achieving the target level when the prior is within a reasonable range.
3. The cost of improved coverage is **wider prediction intervals** — a suboptimal prior necessitates greater uncertainty to compensate.
4. **Numerical precision** is the primary limitation of the calibration method — when the prior deviates severely, the predictive distribution may assign probability as small as $10^{-40}$ over the data support.

## Highlights & Insights

1. **Conceptual Contribution**: Reinterpreting the Bayesian posterior as an "optimal model ensemble" rather than "uncertainty quantification" represents a profound shift in perspective.
2. **Unification via Optimization**: Through the framework of Knoblauch et al., empirical risk minimization and Bayesian inference are unified within the same optimization problem — removing the divergence term recovers point estimation.
3. **Practical Calibration Scheme**: The proposed calibration algorithm is conceptually simple and general, reducing the coverage problem to a one-dimensional search.
4. **Bridging Theory and Practice**: The PAC framework provides frequentist validity guarantees for calibrated intervals while retaining the benefits of Bayesian ensembling.

## Limitations & Future Work

1. **Prior quality metrics are intractable**: $Q(\pi_0)$, $Q'(\pi_0)$, and $Q''(\pi_0)$ depend on the unknown data distribution and can only be approximated via lower bounds or resampling.
2. **Validation limited to simple linear models**: Simulation studies use only linear regression; the approach has not been evaluated on complex models such as neural networks.
3. **Numerical precision constraints**: When model fit is poor or the prior dominates, the estimated quantile $\hat{q}$ may be extremely small (e.g., $10^{-40}$), posing computational challenges.
4. **Generalization error not formally analyzed**: The generalization gap of the calibration algorithm lacks a formal theoretical proof, though the authors argue it should be small given that only a single parameter is estimated.
5. **Symmetric interval assumption**: The current algorithm constructs only symmetric prediction intervals; extension to asymmetric intervals would require two-dimensional optimization.
6. **No comparison with Conformal Prediction**: As a frequentist coverage calibration method, the paper does not compare against the widely adopted conformal prediction framework.

## Related Work & Insights

- **Knoblauch et al. (2022)**: The optimization perspective on Bayesian updating, which forms the theoretical foundation of the paper's central argument.
- **Wenzel et al. (2020)**: Discovery of the "cold posterior" effect in Bayesian neural networks, challenging the validity of BNN uncertainty quantification.
- **Grünwald (2011)**: The Safe Bayesian paradigm, providing an explanation of cold posteriors under model misspecification.
- **Park et al. (2019, 2020)**: Pioneering work on constructing frequentist-valid confidence intervals using PAC bounds.
- **McAllester (1998, 2003)**: Foundational work establishing the PAC-Bayes framework.

The central takeaway of this paper is: **the Bayesian posterior should not be directly equated with uncertainty quantification**. In finite-data regimes, the validity of Bayesian UQ depends critically on the choice of prior, and prior quality should be assessed through frequentist means. This finding carries important cautionary implications for all research that employs Bayesian neural networks for uncertainty estimation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Although a position paper, the argument for decoupling the Bayesian posterior from UQ is original and theoretically grounded.
- **Theoretical Depth**: ⭐⭐⭐⭐ — The mathematical definitions of prior quality metrics are rigorous, and the application of the PAC framework is appropriate.
- **Experimental Thoroughness**: ⭐⭐⭐ — Limited to linear regression simulations; validation on complex models and real-world data is absent.
- **Value**: ⭐⭐⭐ — The calibration algorithm is simple and general, though numerical issues in practice require resolution.
- **Writing Quality**: ⭐⭐⭐⭐ — The exposition is clear with a coherent logical structure throughout.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Asymmetric Duos: Sidekicks Improve Uncertainty](asymmetric_duos_sidekicks_improve_uncertainty.md)
- [\[NeurIPS 2025\] Adjusted Count Quantification Learning on Graphs](adjusted_count_quantification_learning_on_graphs.md)
- [\[NeurIPS 2025\] Uncertainty Estimation by Flexible Evidential Deep Learning](uncertainty_estimation_by_flexible_evidential_deep_learning.md)
- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](../../ICLR2026/others/bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](depth-supervised_fusion_network_for_seamless-free_image_stitching.md)

</div>

<!-- RELATED:END -->
