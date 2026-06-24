---
title: >-
  [Paper Note] Prediction-Powered Adaptive Shrinkage Estimation
description: >-
  [ICML2025][Prediction-Powered Inference] By organically combining Prediction-Powered Inference (PPI) with Empirical Bayes shrinkage, this work proposes the PAS two-stage estimation method. It first utilizes ML predictions for within-problem variance reduction, and then performs across-problem adaptive shrinkage with the ML predictions as shrinkage targets. The shrinkage parameters are automatically tuned via the Correlation-Unbiased Risk Estimator (CURE)…
tags:
  - "ICML2025"
  - "Prediction-Powered Inference"
  - "Empirical Bayes"
  - "Shrinkage Estimation"
  - "James-Stein"
  - "Compound Mean Estimation"
date: 2026-05-08
content_hash: c71998150b12ec10
---

# Prediction-Powered Adaptive Shrinkage Estimation

**Conference**: ICML2025  
**arXiv**: [2502.14166](https://arxiv.org/abs/2502.14166)  
**Code**: None  
**Area**: Statistical Inference / Empirical Bayes  
**Keywords**: Prediction-Powered Inference, Empirical Bayes, Shrinkage Estimation, James-Stein, Compound Mean Estimation

## TL;DR
By organically combining Prediction-Powered Inference (PPI) with Empirical Bayes shrinkage, this work proposes the PAS two-stage estimation method. It first utilizes ML predictions for within-problem variance reduction, and then performs across-problem adaptive shrinkage with the ML predictions as shrinkage targets. The shrinkage parameters are automatically tuned via the Correlation-Unbiased Risk Estimator (CURE), with theoretical guarantees of asymptotic optimality.

## Background & Motivation

**Background**: Prediction-Powered Inference (PPI) is a recently popular statistical inference framework that improves statistical estimation efficiency by combining a small amount of labeled data with a large amount of ML predictions. PPI++ further introduces a power-tuning parameter $\lambda$, ensuring that the estimator's variance is not worse than that of traditional classical estimators.

**Limitations of Prior Work**: Existing PPI methods focus on a single statistical problem, whereas modern applications often require answering a large number of parallel statistical problems simultaneously. For example, in Galaxy Zoo, instead of estimating only the overall proportion of spiral galaxies, one needs to estimate the proportion of spirals within each galaxy subgroup. Solving each problem individually wastes information shared across different problems.

**Key Challenge**: PPI (and PPI++) adheres to unbiasedness, which implies that even when the ML predictor is near-perfect ($\mathbb{E}[f(X_i)] \approx \mathbb{E}[Y_i]$), the MSE still has an irreducible lower bound of $\frac{1}{n}\mathbb{E}[\text{Var}[Y_i|X_i]]$. On the other hand, while the biased prediction mean $\tilde{Z}^f$ can have near-zero variance (as $N\to\infty$), its MSE equals the squared bias. Both have distinct advantages and limitations, making it challenging to achieve the best of both worlds.

**Key Insight**: The compound estimation setting allows further reduction of the MSE by introducing a controlled bias. The authors discover that ML predictions can play a dual role in PPI: (1) as a variance reduction tool (within-problem), and (2) as a shrinkage target (across-problem).

**Core Idea**: First, apply PPI++ for within-problem variance reduction to obtain the Power-Tuned (PT) estimators. Then, adaptively shrink these estimators toward the ML prediction means, automatically selecting the shrinkage intensity by minimizing a correlation-aware unbiased risk estimator.

## Method

### Overall Architecture
PAS is a two-stage method designed to handle $m$ parallel mean estimation problems. Each problem $j$ has $n_j$ labeled data points $(X_{ij}, Y_{ij})$ and $N_j$ unlabeled covariates $(\tilde{X}_{ij})$, along with a black-box predictor $f$. The objective is to estimate the true mean of each problem, $\theta_j = \mathbb{E}[Y_{ij}]$. The output is the PAS estimation vector $\hat{\boldsymbol{\theta}}^{\text{PAS}} = (\hat{\theta}_1^{\text{PAS}}, \ldots, \hat{\theta}_m^{\text{PAS}})$.

### Key Designs

1. **Stage 1: Within-Problem Power Tuning**:

    - **Function**: Utilizes ML predictions within each problem to achieve variance reduction, yielding unbiased PT estimators.
    - **Mechanism**: For each problem $j$, construct a family of PPI estimators $\hat{\theta}_{j,\lambda}^{\text{PPI}} = \bar{Y}_j + \lambda(\tilde{Z}_j^f - \bar{Z}_j^f)$, where $\bar{Y}_j$ is the classical mean estimate, while $\tilde{Z}_j^f$ and $\bar{Z}_j^f$ are the predicted means on unlabeled and labeled data, respectively. The optimal $\lambda$ minimizes the variance, yielding the analytical solution $\lambda_j^* = \frac{N_j}{n_j+N_j} \cdot \frac{\gamma_j}{\tau_j^2}$ (where $\gamma_j$ is the prediction-label covariance and $\tau_j^2$ is the prediction variance). This yields the PT estimator $\hat{\theta}_j^{\text{PT}}$, whose variance $\tilde{\sigma}_j^2 = \frac{\sigma_j^2}{n_j} - \frac{N_j}{n_j(n_j+N_j)}\frac{\gamma_j^2}{\tau_j^2}$ is guaranteed to be no greater than that of the classical estimator.
    - **Design Motivation**: Power tuning is performed individually for each problem because the optimal $\lambda$ depends on problem-specific second moments. This step ensures unbiasedness, establishing a solid foundation for the subsequent biased shrinkage.

2. **Stage 2: Across-Problem Adaptive Shrinkage**:

    - **Function**: Achieves further reduction in MSE by shrinking toward the ML prediction means, at the cost of introducing a controllable bias.
    - **Mechanism**: Define a family of shrinkage estimators $\hat{\theta}_{j,\omega}^{\text{PAS}} = \omega_j \hat{\theta}_j^{\text{PT}} + (1-\omega_j)\tilde{Z}_j^f$, where $\omega_j = \frac{\omega}{\omega + \tilde{\sigma}_j^2}$. Here, $\omega \geq 0$ is a global shrinkage parameter. The design of $\omega_j$ follows a parameterized form of Bayesian optimal weights: problems with larger variance (larger $\tilde{\sigma}_j^2$) receive smaller weights $\omega_j \to$ shrinking more toward $\tilde{Z}_j^f$; problems with smaller variance receive larger weights $\to$ retaining more of the PT estimate. As $\omega \to \infty$, it degenerates to PT; as $\omega = 0$, it degenerates to the prediction mean.
    - **Design Motivation**: This parameterized form perfectly matches the structure of the Bayesian posterior mean (regardless of the shrinkage target, the optimal weight can be written in the form of $\omega/(\omega+\sigma^2)$). Thus, adaptivity across all problems can be achieved by learning only a single scalar $\omega$.

3. **CURE: Correlation-Unbiased Risk Estimator**:

    - **Function**: Serves as a proxy for the true compound risk to enable data-driven selection of the optimal $\omega$.
    - **Mechanism**: $\text{CURE}(\hat{\boldsymbol{\theta}}_\omega^{\text{PAS}}) = \frac{1}{m}\sum_{j=1}^m [(2\omega_j-1)\tilde{\sigma}_j^2 + 2(1-\omega_j)\tilde{\gamma}_j + (1-\omega_j)^2(\hat{\theta}_j^{\text{PT}} - \tilde{Z}_j^f)^2]$, where $\tilde{\gamma}_j = \frac{\gamma_j}{n_j+N_j}$ is the covariance between the PT estimator and the prediction mean. Crucially, "correlation-awareness" means that the non-zero covariance $\tilde{\gamma}_j$ between the shrinkage source $\hat{\theta}_j^{\text{PT}}$ and target $\tilde{Z}_j^f$ must be accounted for. Theorem 4.1 proves that CURE is unbiased for the compound risk: $\mathbb{E}_{\boldsymbol{\eta}}[\text{CURE}] = \mathcal{R}_m(\hat{\boldsymbol{\theta}}_\omega^{\text{PAS}}, \boldsymbol{\theta})$.
    - **Design Motivation**: Classical SURE assumes independence between the shrinkage source and target. However, in PAS, both share the ML predictor, leading to correlation. CURE corrects for this by explicitly incorporating $\tilde{\gamma}_j$.

### Loss & Training
The final shrinkage parameter is selected via a one-dimensional grid search that minimizes CURE: $\hat{\omega} \in \arg\min_{\omega \geq 0} \text{CURE}(\hat{\boldsymbol{\theta}}_\omega^{\text{PAS}})$. No cross-validation is required since CURE is an unbiased risk estimator itself. In practice, the second moments $\sigma_j^2, \tau_j^2, \gamma_j$ are replaced by their standard sample estimates.

## Key Experimental Results

### Synthetic Experiments (m=200 problems, good/poor predictors)

| Estimator | MSE f₁(x)=x² (×10⁻³) | MSE f₂(x)=\|x\| (×10⁻³) |
|--------|----------------------|----------------------|
| Classical $\bar{Y}$ | 3.142 ± 0.033 | 3.142 ± 0.033 |
| Prediction Avg $\tilde{Z}^f$ | 0.273 ± 0.004 | 34.335 ± 0.147 |
| PPI | 2.689 ± 0.027 | 2.756 ± 0.027 |
| PT (PPI++) | 2.642 ± 0.027 | 2.659 ± 0.026 |
| Shrink Classical | 0.272 ± 0.003 | 2.863 ± 0.030 |
| **PAS (Ours)** | **0.272 ± 0.003** | **2.466 ± 0.026** |

### Real-World Data Experiments (K=200 Monte Carlo)

| Estimator | Amazon(base) MSE(×10⁻³) | Amazon(tuned) MSE(×10⁻³) | Galaxy MSE(×10⁻³) |
|--------|------------------------|--------------------------|-------------------|
| Classical | 24.305 ± 0.189 | 24.305 ± 0.189 | 2.073 ± 0.028 |
| PT | 10.633 ± 0.089 | 6.289 ± 0.050 | 1.026 ± 0.015 |
| Shrink Classical | 15.995 ± 0.121 | 3.828 ± 0.039 | 1.522 ± 0.016 |
| **PAS (Ours)** | **8.517 ± 0.071** | **3.287 ± 0.024** | **0.893 ± 0.011** |
| UniPAS (Ours) | 8.879 ± 0.073 | 3.356 ± 0.031 | 0.909 ± 0.011 |

### Key Findings
- PAS achieves optimal performance in both extreme cases: with the good predictor $f_1$, PAS matches the low MSE of the prediction average (via strong shrinkage), while with the poor predictor $f_2$, PAS retains the robustness of PT (via weak shrinkage). This demonstrates its adaptivity.
- On the Amazon reviews dataset: when the BERT-tuned predictor performs well, PAS applies substantial shrinkage (MSE 3.287 vs. PT 6.289); when the BERT-base predictor performs poorly, PAS applies moderate shrinkage (MSE 8.517 vs. PT 10.633, achieving improvement without being overly aggressive).
- PAS improves not only the average MSE but also the majority of individual problems—showing improvement in 80.8% of the problems on Amazon(tuned).
- The performance of UniPAS (a variant requiring no prior knowledge of second moments) is close to that of PAS, validating the practicality of the proposed method.

## Highlights & Insights
- Organically unifies two classical statistical concepts: PPI variance reduction and James-Stein shrinkage. In PPI, the ML prediction serves simultaneously as a "variance reducer" and a "shrinkage target," exploiting the same information source twice. This dual-use design scheme can be transferred to other scenarios requiring joint management of bias and variance.
- The "correlation-aware" design of CURE is a highlights theoretical contribution—revealing the easily overlooked source-target correlation issue in shrinkage estimation. While classical SURE assumes independence, CURE generalizes risk estimation to correlated scenarios.
- The asymptotic optimality guarantee (Theorem 5.2) implies that as the number of parallel problems grows, PAS automatically achieves the optimal bias-variance tradeoff without manual hyperparameter tuning.

## Limitations & Future Work
- Asymptotic optimality requires $m \to \infty$ (the number of problems tending to infinity), meaning that the theoretical guarantees are weaker when the number of problems is small.
- The global shrinkage parameter $\omega$ is shared across all problems. Although $\omega_j$ allows for problem-specific shrinkage intensity, the shrinkage target remains restricted to $\tilde{Z}_j^f$. More flexible non-linear shrinkage or problem-specific shrinkage targets could be superior.
- Systematic comparisons with non-parametric/deep learning methods (such as deep empirical Bayes) are currently lacking.
- The exchangeability assumption between problems (Assumption 2.1) may not hold in scenarios with systematic grouping.
- Estimating second moments is unstable when $n_j$ is extremely small. Although UniPAS alleviates this, it increases method complexity.

## Related Work & Insights
- **vs PPI++ (Angelopoulos et al. 2024)**: PPI++ is the Stage 1 building block of PAS. PAS further reduces MSE by incorporating across-problem shrinkage in Stage 2. The rigid adherence to unbiasedness in PPI++ is its limitation.
- **vs James-Stein / SURE (Xie et al. 2012)**: Classical shrinkage shrinks toward zero or the grand mean, whereas PAS shrinks toward ML prediction means. The latter serves as a more "informed" shrinkage target, leveraging prior knowledge encoded by the ML predictors.
- **vs StratPPI (Fisch et al.)**: StratPPI also leverages stratification but aims to estimate a single population parameter, whereas PAS aims to estimate the parameter vector of all sub-problems.
- **vs FAB-PPI (Cortinovis et al. 2025)**: FAB combines PPI and heavy-tailed priors for single problems but does not pursue empirical Bayes and across-problem information sharing.
- **Insight**: The PAS framework can be generalized to more complex estimation targets (such as quantiles or causal effects).
- **Insight**: The unbiased risk estimation concept of CURE can be transferred to other shrinkage or regularization scenarios requiring data-driven hyperparameter selection.
- **Insight**: The "dual-use" scheme of ML predictions (variance reduction + shrinkage target) may also prove valuable in other semi-supervised settings.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Organically unifies the separate fields of PPI and empirical Bayes; CURE is an original theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on synthetic, Amazon reviews, and Galaxy Zoo datasets with 200 Monte Carlo repetitions.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Extremely elegant presentation—building intuition from a simplified Gaussian model and then generalizing to the general cases.
- **Value**: ⭐⭐⭐⭐ Promising prospects for wide application in large-scale statistical inference and ML-assisted analysis.
- **Overall**: ⭐⭐⭐⭐ Solid theory, elegant formulation, and broad applicability; an outstanding interdisciplinary work bridging statistics and ML.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Time-Aware World Model for Adaptive Prediction and Control](time-aware_world_model_for_adaptive_prediction_and_control.md)
- [\[ICML 2026\] Industrializing Prediction-Powered Inference: The GLIDE Library for Reliable GenAI and Agentic Systems Evaluation](../../ICML2026/others/industrializing_prediction-powered_inference_the_glide_library_for_reliable_gena.md)
- [\[ICML 2025\] Prediction via Shapley Value Regression (ViaSHAP)](prediction_via_shapley_value_regression.md)
- [\[ICML 2025\] Latent Variable Estimation in Bayesian Black-Litterman Models](latent_variable_estimation_in_bayesian_black-litterman_models.md)
- [\[ACL 2025\] RePanda: Pandas-powered Tabular Verification and Reasoning](../../ACL2025/others/repanda_pandas-powered_tabular_verification_and_reasoning.md)

</div>

<!-- RELATED:END -->
