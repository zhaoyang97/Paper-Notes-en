---
title: >-
  [Paper Note] Risk and Cross Validation in Ridge Regression with Correlated Samples
description: >-
  [ICML2025][Time Series][Ridge Regression] Utilizing random matrix theory and free probability techniques, this work derives exact risk asymptotic formulas for high-dimensional ridge regression with training samples having arbitrary correlation, and proposes a corrected generalized cross-validation estimator, CorrGCV, which accurately predicts out-of-sample risk under sample-correlated conditions.
tags:
  - "ICML2025"
  - "Time Series"
  - "Ridge Regression"
  - "Cross-Validation"
  - "Correlated Samples"
  - "Random Matrix Theory"
  - "Free Probability"
  - "Time Series Forecasting"
  - "High-dimensional Asymptotics"
date: 2026-05-08
content_hash: dd041fa8eb6a1bee
---

# Risk and Cross Validation in Ridge Regression with Correlated Samples

**Conference**: ICML2025  
**arXiv**: [2408.04607](https://arxiv.org/abs/2408.04607)  
**Code**: [Pehlevan-Group/S_transform](https://github.com/Pehlevan-Group/S_transform)  
**Area**: Time Series  
**Keywords**: Ridge Regression, Cross-Validation, Correlated Samples, Random Matrix Theory, Free Probability, Time Series Forecasting, High-dimensional Asymptotics

## TL;DR

Utilizing random matrix theory and free probability techniques, this work derives exact risk asymptotic formulas for high-dimensional ridge regression with training samples having arbitrary correlation, and proposes a corrected generalized cross-validation estimator, CorrGCV, which accurately predicts out-of-sample risk under sample-correlated conditions.

## Background & Motivation

- **Failure of classical statistical assumptions**: Traditional ridge regression theory assumes training samples are i.i.d., but time series data (finance, climate, neuroscience) naturally possess inter-sample correlation, rendering existing theories inapplicable.
- **Theoretical progress in high-dimensional ridge regression**: In recent years, a large body of work (Hastie et al. 2022; Bordelon et al. 2020; Canatar et al. 2021, etc.) has provided exact out-of-sample risk asymptotics under high-dimensional proportional limits, but **nearly all assume i.i.d. samples**.
- **Limitations of GCV estimator**: Classical generalized cross-validation (GCV) is asymptotically exact in the i.i.d. setting, but suffers from **systematic estimation bias** when samples are correlated. Existing correction attempts (Altman 1990; Carmack et al. 2012) have not proven asymptotic exactness in high dimensions.
- **Practical demand**: Time series regression requires hyperparameter tuning methods (especially for regulating the regularization strength $\lambda$) that can correctly account for correlation.

## Method

### Problem Modeling

Consider ridge regression with training set $\mathcal{D} = \{(\mathbf{x}_t, y_t)\}_{t=1}^T$ and loss function:

$$L(\mathbf{w}) = \frac{1}{T}\sum_{t=1}^T (y_t - \mathbf{x}_t^\top \mathbf{w})^2 + \lambda \|\mathbf{w}\|^2$$

Data model: $\mathbf{y} = \mathbf{X}\bar{\mathbf{w}} + \boldsymbol{\epsilon}$, with the design matrix $\mathbf{X} = \mathbf{K}^{1/2}\mathbf{Z}\boldsymbol{\Sigma}^{1/2}$, where:

- $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$: feature-feature covariance
- $\mathbf{K} \in \mathbb{R}^{T \times T}$: sample-sample correlation matrix (**this is the core new element of this paper**)
- $\mathbf{Z}$: i.i.d. standard Gaussian matrix
- Noise covariance: $\mathbb{E}[\epsilon_t \epsilon_s] = \sigma_\epsilon^2 K'_{ts}$

### Theoretical Tool: Deterministic Equivalence

The core technique is establishing deterministic equivalence through the **S-transform in free probability**. Define the degrees of freedom:

$$\mathrm{df}_1 = \frac{1}{N}\mathrm{Tr}[\boldsymbol{\Sigma}(\boldsymbol{\Sigma}+\kappa)^{-1}], \quad \tilde{\mathrm{df}}_1 = \frac{1}{T}\mathrm{Tr}[\mathbf{K}(\mathbf{K}+\tilde{\kappa})^{-1}]$$

where the renormalized ridge parameters $\kappa, \tilde{\kappa}$ are determined by the self-consistent equations of the S-transform, satisfying the duality relation $\kappa\tilde{\kappa}/\lambda = 1/\tilde{\mathrm{df}}_1$.

**One-point strong deterministic equivalence** (Lemma 2.2): The resolvent of the sample covariance can be approximated by that of the population covariance:

$$\hat{\boldsymbol{\Sigma}}(\hat{\boldsymbol{\Sigma}}+\lambda)^{-1} \simeq \boldsymbol{\Sigma}(\boldsymbol{\Sigma}+\kappa)^{-1}$$

**Two-point strong deterministic equivalence** (Lemma 2.3, 2.4): Used to derive the exact asymptotics of variance terms.

### Main Results

**Result 1: Exact risk under matching correlation** ($\mathbf{K} = \mathbf{K}'$, Theorem 3.2)

$$R_g \simeq \frac{\kappa^2}{1-\gamma}\bar{\mathbf{w}}^\top \boldsymbol{\Sigma}(\boldsymbol{\Sigma}+\kappa)^{-2}\bar{\mathbf{w}} + \frac{\gamma}{1-\gamma}\sigma_\epsilon^2$$

where $\gamma = \frac{\mathrm{df}_2}{\mathrm{df}_1}\frac{\tilde{\mathrm{df}}_2}{\tilde{\mathrm{df}}_1}$ (note that $\gamma$ now depends on degrees of freedom from both the feature side and the sample side).

**Result 2: CorrGCV Estimator**

$$R_{out} = S(\mathrm{df}_1) \frac{\tilde{\mathrm{df}}_1}{\tilde{\mathrm{df}}_1 - \tilde{\mathrm{df}}_2} \hat{R}_{in}$$

- **Asymptotically unbiased and concentrated**: exact as $N, T \to \infty$.
- **Computable solely from training data**: $S, \tilde{\mathrm{df}}_1, \tilde{\mathrm{df}}_2$ can all be estimated from data.

**Result 3: General risk for mismatched correlation/covariate shift** (Theorem 3.3)

The risk is decomposed into three terms:

$$R_g \simeq \underbrace{\kappa^2 \bar{\mathbf{w}}^\top(\boldsymbol{\Sigma}+\kappa)^{-1}\boldsymbol{\Sigma}'(\boldsymbol{\Sigma}+\kappa)^{-1}\bar{\mathbf{w}}}_{\mathrm{Bias}^2} + \underbrace{\mathrm{Var}_{\mathbf{X}}}_\text{covariate variance} + \underbrace{\mathrm{Var}_{\mathbf{X}\boldsymbol{\epsilon}}}_\text{noise variance}$$

This reveals the **duality between covariate shift and noise-sample correlation mismatch**.

**Result 4: Correlated test points in time series**

When test points are correlated with the training set (e.g., near-future forecasting), the model's performance exhibits over-optimism. This work precisely characterizes the **decay of prediction accuracy with forecasting distance**.

### Scaling Law Invariance

Under power-law feature spectra $\lambda_k \sim k^{-\alpha}$, the scaling exponent of the optimal risk $R_g \sim T^{-2\alpha\min(r,1)}$ remains unaffected by the correlation structure of stationary processes.

## Key Experimental Results

### Main Contrast: CorrGCV vs. Naïve GCV

| Estimator | Weak Correlation ($\xi=10^{-2}$) | Strong Correlation ($\xi=10^2$) |
|---|---|---|
| Naïve GCV₁ $(1-q\cdot\mathrm{df}_1)^{-2}$ | ≈ Accurate | Severe bias |
| Naïve GCV₂ (Altman 1990) $S^2$ | ≈ Accurate | Underestimate risk |
| Carmack et al. (2012) | ≈ Accurate | Overestimate risk |
| **CorrGCV (Ours)** | **Accurate** | **Accurate** |

### Hyperparameter Tuning Results

- Only CorrGCV can correctly locate the **optimal regularization parameter $\lambda^*$**.
- Naïve GCV mislocates the optimal $\lambda$ under strong correlation, leading to sub-optimal generalization.

### Time Series Experiments

- Under exponential correlation $\mathbb{E}[\mathbf{x}_t \cdot \mathbf{x}_s] \propto e^{-|t-s|/\xi}$, theoretical predictions match perfectly with 10 repeated experimental trials.
- The risk of correlated test points monotonically increases with prediction distance, validating the theoretical prediction that near-future forecasting is overly optimistic.

### Ablation Study

| Ablation Dimension | Finding |
|---|---|
| $\mathbf{K} = \mathbf{K}'$ vs. $\mathbf{K} \neq \mathbf{K}'$ | CorrGCV is exact when matched; extra information is required when unmatched, making estimation solely from training data impossible. |
| Over-parameterization $q>1$ vs. Under-parameterization $q<1$ | Theory holds in both regimes |
| Power-law covariance spectrum | Scaling law exponent is unaffected by correlation structure |

## Highlights & Insights

1. **Fills a critical theoretical gap**: For the first time, exact asymptotic risk formulas and a computable unbiased risk estimator are established for high-dimensional ridge regression with correlated samples.
2. **Practical value of CorrGCV**: Can be computed using only training data, directly applicable to hyperparameter tuning under correlated data.
3. **Duality of covariate shift and noise correlation mismatch**: An elegant theoretical finding unifying two seemingly distinct problems.
4. **Renormalization perspective**: Stochastic fluctuation effects are absorbed into a renormalized ridge parameter $\kappa$. Even as $\lambda \to 0$, we may have $\kappa > 0$ (implicit regularization).
5. **Direct guidance for time series forecasting**: Precisely quantifies the optimistic bias in near-future forecasting.

## Limitations & Future Work

- **Gaussian assumption**: The data model assumes Gaussian design matrices; applicability to non-Gaussian time series (such as heavy-tailed distributions in financial returns) requires further validation.
- **Linear model**: Limited to ridge regression, not yet extended to kernel methods or neural networks.
- **Prior knowledge of correlation structure**: CorrGCV assumes $\mathbf{K}$ is known or can be estimated accurately. In practice, estimating the correlation structure itself can be noisy.
- **Stationarity assumption**: Principal results assume time series stationarity; non-stationary scenarios require further research.
- **No GCV for noise-sample mismatch**: When $\mathbf{K} \neq \mathbf{K}'$, there is no exact risk estimator that can be computed solely from training data.

## Related Work & Insights

- **High-dimensional ridge regression theory**: Hastie et al. (2022) summarized ridge regression under proportional limits; Bordelon et al. (2020) and Canatar et al. (2021) established the spectral bias theory; this work extends these results to correlated samples.
- **GCV theory**: Golub et al. (1979) and Craven & Wahba (1978) proposed classical GCV; Jacot et al. (2020) and Atanasov et al. (2024) proved its asymptotic exactness in high dimensions; this work details the connection between GCV and the S-transform and generalizes it.
- **CV on correlated data**: Altman (1990) and Carmack et al. (2012) proposed overcorrections but lacked proofs of high-dimensional asymptotic exactness.
- **Free probability**: Heavily draws from the technical framework of Potters & Bouchaud (2020).
- The theoretical framework established in this study lays a foundation for analyzing kernel regression, random feature models, etc., in time-series settings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to establish exact asymptotics and a computable estimator for high-dimensional ridge regression under correlated samples.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively verified under multiple correlation structures, showing exact agreement between theory and experiment.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous with clear physical intuition and well-structured presentation.
- Value: ⭐⭐⭐⭐⭐ Directly practical for hyperparameter tuning in time-series regressions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](../../ICLR2026/time_series/from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)
- [\[ACL 2025\] CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis](../../ACL2025/time_series/ctpd_cross-modal_temporal_pattern_discovery_for_enhanced_multimodal_electronic_h.md)
- [\[ICLR 2026\] STORM: Synergistic Cross-Scale Spatio-Temporal Modeling for Weather Forecasting](../../ICLR2026/time_series/storm_synergistic_cross-scale_spatio-temporal_modeling_for_weather_forecasting.md)
- [\[AAAI 2026\] HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction](../../AAAI2026/time_series/hydrodcm_hydrological_domain-conditioned_modulation_for_cross-reservoir_inflow_p.md)
- [\[ICML 2026\] HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](../../ICML2026/time_series/helix_hybrid_encoding_with_learnable_identity_and_cross-dimensional_synthesis_fo.md)

</div>

<!-- RELATED:END -->
