---
title: >-
  [Paper Note] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting
description: >-
  [ICLR 2026][Image Generation][probabilistic forecasting] This paper proposes CW-Gen (Conditionally Whitened Generative Models), which replaces the standard Gaussian terminal distribution in diffusion models and flow matching by jointly estimating the conditional mean and a sliding-window covariance matrix. The authors provide theoretical guarantees showing that sampling quality is necessarily improved when the estimator satisfies sufficient conditions, and demonstrate consistent improvements in multivariate time series probabilistic forecasting across 5 datasets × 6 generative models.
tags:
  - ICLR 2026
  - Image Generation
  - probabilistic forecasting
  - diffusion model
  - flow matching
  - conditional whitening
  - covariance estimation
date: 2026-05-08
content_hash: b30b69395ede7e3e
---

# Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting

**Conference**: ICLR 2026
**arXiv**: [2509.20928](https://arxiv.org/abs/2509.20928)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: probabilistic forecasting, diffusion model, flow matching, conditional whitening, covariance estimation

## TL;DR
This paper proposes CW-Gen (Conditionally Whitened Generative Models), which replaces the standard Gaussian terminal distribution in diffusion models and flow matching by jointly estimating the conditional mean and a sliding-window covariance matrix. The authors provide theoretical guarantees showing that sampling quality is necessarily improved when the estimator satisfies sufficient conditions, and demonstrate consistent improvements in multivariate time series probabilistic forecasting across 5 datasets × 6 generative models.

## Background & Motivation

**Background**: Diffusion models (TimeGrad, CSDI, SSSD, Diffusion-TS) and flow matching (FlowTS) have been applied to probabilistic forecasting of multivariate time series. CARD, TimeDiff, and TMDM introduce conditional mean regressors as priors to improve predictions, and NsDiff further incorporates conditional variance.

**Limitations of Prior Work**: (a) The terminal distribution of standard diffusion models is $\mathcal{N}(0, I)$, which completely discards conditional mean and covariance prior information—forcing the denoising process to learn non-stationary trends and inter-variable dependencies from scratch; (b) existing methods that introduce priors (TMDM, NsDiff) are overly complex in design and neglect inter-variable covariance; (c) there is no theoretical guarantee addressing when and why incorporating priors improves generation quality.

**Key Challenge**: The forward process of diffusion models corrupts data into standard Gaussian noise, discarding conditional mean and covariance information. If the terminal distribution could be brought closer to $P_{X|C}$ (i.e., with smaller KL divergence), generation quality would necessarily improve—but how accurate must the estimator be to yield a benefit?

**Goal**: (a) Theoretically characterize the conditions under which replacing the terminal distribution improves generation; (b) design a joint mean-covariance estimator; (c) propose a unified framework applicable to both diffusion models and flow matching.

**Key Insight**: Conditional whitening—centering data using the estimated conditional mean and normalizing with the inverse square root of the estimated conditional covariance, which is equivalent to a linear transformation that brings the data closer to a standard Gaussian.

**Core Idea**: Replace the terminal distribution $\mathcal{N}(0,I)$ in diffusion/flow matching with $\mathcal{N}(\hat{\mu}_{X|C}, \hat{\Sigma}_{X|C})$, so that through the conditional whitening transformation, the denoising network only needs to learn the residual.

## Method

### Overall Architecture
The framework consists of two stages: (1) JMCE (Joint Mean-Covariance Estimator), built upon a Non-stationary Transformer, jointly estimates the conditional mean and a sliding-window covariance; (2) the estimated quantities are injected into a diffusion model (CW-Diff) or flow matching (CW-Flow) via conditional whitening, replacing the terminal distribution.

### Key Designs

1. **Theorem 1 (Theoretical Guarantee)**:

    - Statement: When $(\min_i \hat{\lambda}_i)^{-1}(\|\mu - \hat{\mu}\|_2^2 + \|\Sigma - \hat{\Sigma}\|_N) + \sqrt{d}\|\Sigma - \hat{\Sigma}\|_F \leq \|\mu\|_2^2$, replacing the terminal distribution reduces the KL divergence.
    - Significance: The left-hand side represents estimation error (mean error plus covariance error under nuclear norm and Frobenius norm), and the right-hand side represents signal strength (the norm of the conditional mean). For non-stationary series, $\|\mu\|_2^2$ is large, making the condition easier to satisfy.
    - Design Motivation: This theorem directly guides the design of the JMCE loss function—minimizing the left-hand side makes the condition easier to fulfill.

2. **JMCE (Joint Mean-Covariance Estimator)**:

    - Function: Simultaneously outputs the conditional mean $\hat{\mu}_{X|C}$ and the sliding-window covariance $\hat{\Sigma}_{t|C}$ at each time step.
    - Mechanism: The covariance is parameterized via Cholesky decomposition $\hat{\Sigma}_t = \hat{L}_t \hat{L}_t^\top$ to ensure positive semi-definiteness. The loss function is $\mathcal{L}_{\text{JMCE}} = \mathcal{L}_2 + \mathcal{L}_{\text{SVD}} + \lambda_{\min}\sqrt{dT_f}\mathcal{L}_F + w_{\text{Eigen}}\sum_t \mathcal{R}_{\lambda_{\min}}$.
    - Minimum eigenvalue penalty: $\mathcal{R}_{\lambda_{\min}} = \sum_i \text{ReLU}(\lambda_{\min} - \hat{\lambda}_i)$, ensuring the covariance matrix does not degenerate.
    - Design Motivation: The left-hand side of Theorem 1 contains $(\min_i \hat{\lambda}_i)^{-1}$—a small minimum eigenvalue amplifies the error. The penalty term directly controls this risk factor.

3. **CW-Diff (Conditionally Whitened Diffusion)**:

    - Function: Injects conditional mean and covariance into the standard DDPM forward/reverse process.
    - Mechanism: Define $X_0^{\text{CW}} = \hat{\Sigma}^{-0.5} \circ (X_0 - \hat{\mu})$ (conditional whitening), apply standard DDPM on $X_0^{\text{CW}}$, and after sampling apply the inverse transform $X = \hat{\Sigma}^{0.5} \circ X^{\text{CW}} + \hat{\mu}$.
    - Design Motivation: Whitened data is closer to a standard Gaussian, improving the terminal approximation of the forward process and simplifying the reverse denoising task (learning only the residual).

4. **CW-Flow (Conditionally Whitened Flow Matching)**:

    - Function: Replaces the terminal noise from $\mathcal{N}(0,I)$ with $\mathcal{N}(\hat{\mu}, \hat{\Sigma})$.
    - Advantage: Matrix inversion is not required (unlike CW-Diff, which needs $\hat{\Sigma}^{-0.5}$), resulting in greater computational efficiency.

### Loss & Training
JMCE is pre-trained first; the whitened data is then used to train the diffusion or flow matching network. JMCE uses a Non-stationary Transformer as its backbone.

## Key Experimental Results

### Main Results (CW-Gen Win Rate)

| Dataset | Dimensions | CW-Gen Win Rate (6 models × 4 metrics) |
|---------|------------|----------------------------------------|
| ETTh1 | 7 | 22/24 ≈ 91.7% |
| ETTh2 | 7 | 22/24 ≈ 91.7% |
| ILI | 7 | 20/24 ≈ 83.3% |
| Weather | 21 | 22/24 ≈ 91.7% |
| Solar Energy | 137 | 19/24 ≈ 79.2% |

### Ablation Study (ETTh1, CRPS ↓)

| Model | Raw | + CW | Gain |
|-------|-----|------|------|
| TimeDiff | 0.787 | 0.505 | -35.8% |
| SSSD | 0.836 | 0.524 | -37.3% |
| Diffusion-TS | 0.626 | ~0.45 | ~-28% |
| FlowTS | ~0.7 | ~0.5 | ~-29% |

### Key Findings
- **Consistent effectiveness across models**: All 6 generative models (TimeDiff, SSSD, CSDI, Diffusion-TS, FlowTS, TMDM) consistently improve with CW, indicating that the method is model-agnostic.
- **Effective in high dimensions**: A 79% win rate on Solar Energy (137 dimensions) demonstrates that covariance estimation remains beneficial in high-dimensional settings.
- **CRPS improvement of 28–37%**: A substantial margin of improvement, indicating that the standard Gaussian terminal distribution indeed wastes a large amount of prior information.
- **CW-Flow vs. CW-Diff**: CW-Flow avoids matrix inversion, resulting in faster computation with comparable performance.
- **Robustness to distribution shift**: Experiments show that CW-Gen effectively mitigates distributional discrepancies between training and test sets.

## Highlights & Insights
- **Theorem 1 formalizes the intuition with a "sufficient condition"**: The larger the signal ($\|\mu\|_2^2$), the smaller the estimation error, and the farther the minimum eigenvalue from zero, the more beneficial it is to replace the terminal distribution. Non-stationary series naturally satisfy this condition.
- **Conditional whitening as a unified framework for prior injection in diffusion models**: CARD, TimeDiff, TMDM, and NsDiff can all be viewed as special cases of CW-Gen, a unifying perspective with both theoretical and practical value.
- **JMCE loss derived directly from theory**: Each loss term (L2, SVD norm, Frobenius norm, eigenvalue penalty) corresponds to a term in Theorem 1, with theory directly guiding design.
- **Plug-and-play property**: Conditional whitening operates as a data preprocessing step without modifying the diffusion/flow matching architecture, enabling seamless integration into any existing model.

## Limitations & Future Work
- **Covariance estimation may be unstable in very high dimensions**: At 137 dimensions it is already necessary to estimate a $137 \times 137$ covariance matrix; even higher-dimensional settings (e.g., thousands of variables) may require structured assumptions such as diagonal or sparse covariance.
- **JMCE pre-training introduces additional computational cost**: Training a separate mean-covariance estimator is required, though this is a one-time expense.
- **Validation limited to forecasting tasks**: Time series tasks such as generative imputation and anomaly detection have not been evaluated.
- **Future directions**: (a) Replace linear whitening with nonlinear whitening (e.g., normalizing flows); (b) jointly train JMCE and the diffusion model end-to-end.

## Related Work & Insights
- **vs. CARD/TimeDiff**: These methods inject only the conditional mean. CW-Gen jointly injects both mean and covariance, with theoretical proof of the covariance's contribution.
- **vs. NsDiff**: NsDiff incorporates mean and variance but neglects inter-variable covariance (assuming variable independence). CW-Gen uses a full covariance matrix to capture inter-variable dependencies.
- **vs. TMDM**: TMDM embeds mean regression within a variational inference framework, resulting in a complex design. CW-Gen's whitening operation is comparatively simpler and more elegant.

## Rating
- Novelty: ⭐⭐⭐⭐ The conditional whitening concept is novel, the theoretical analysis is thorough, and the framework unifies multiple prior methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 5 datasets × 6 models × 4 metrics; win-rate statistics are convincing.
- Writing Quality: ⭐⭐⭐⭐⭐ The derivation chain from theory to design to experiments is clear and complete.
- Value: ⭐⭐⭐⭐⭐ Directly plug-and-play into any time series diffusion or flow matching model; extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting](doflow_flow-based_generative_models_for_interventional_and_counterfactual_foreca.md)
- [\[AAAI 2026\] SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting](../../AAAI2026/image_generation/simdiff_simpler_yet_better_diffusion_model_for_time_series_point_forecasting.md)
- [\[NeurIPS 2025\] Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics](../../NeurIPS2025/image_generation/elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)
- [\[ICLR 2026\] FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching](flowcast_trajectory_forecasting_for_scalable_zero-cost_speculative_flow_matching.md)
- [\[ICLR 2026\] Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening](motion_prior_distillation_in_time_reversal_sampling_for_generative_inbetweening.md)

</div>

<!-- RELATED:END -->
