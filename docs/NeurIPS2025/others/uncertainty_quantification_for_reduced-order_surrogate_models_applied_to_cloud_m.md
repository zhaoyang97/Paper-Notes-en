---
title: >-
  [Paper Note] Uncertainty Quantification for Reduced-Order Surrogate Models Applied to Cloud Microphysics
description: >-
  [NeurIPS 2025][reduced-order models] This paper proposes the first post-hoc, model-agnostic uncertainty quantification framework for latent-space reduced-order models. By applying conformal prediction to the reconstructi…
tags:
  - "NeurIPS 2025"
  - "reduced-order models"
  - "conformal prediction"
  - "cloud microphysics"
  - "autoencoder-SINDy"
  - "surrogate models"
  - "prediction intervals"
date: 2026-05-08
content_hash: 30d039d058965b09
---

# Uncertainty Quantification for Reduced-Order Surrogate Models Applied to Cloud Microphysics

**Conference**: NeurIPS 2025
**arXiv**: [2511.04534](https://arxiv.org/abs/2511.04534)  
**Code**: [GitHub](https://github.com/jonaskat87/UQ_AE-SINDy)  
**Area**: Other
**Keywords**: reduced-order models, conformal prediction, cloud microphysics, autoencoder-SINDy, surrogate models, prediction intervals

## TL;DR

This paper proposes the first post-hoc, model-agnostic uncertainty quantification framework for latent-space reduced-order models. By applying conformal prediction to the reconstruction, latent dynamics, and end-to-end prediction components independently, it constructs distribution-free prediction intervals and reveals component-level uncertainty propagation in cloud microphysics ROMs — showing that structural errors in the autoencoder, rather than dynamics errors, dominate end-to-end prediction uncertainty.

## Background & Motivation

**Background**: Latent-space reduced-order models (ROMs) learn compact representations of high-dimensional physical systems in low-dimensional latent spaces and have demonstrated efficient simulation capabilities in complex fluid dynamics and related domains. In cloud microphysics, ROMs can replace traditional bulk parameterization schemes to simulate droplet collision–coalescence processes at significantly reduced computational cost.

**Limitations of Prior Work**: Existing UQ methods for ROMs suffer from three major issues: (1) they are tied to specific architectures (e.g., variational autoencoders require dedicated probabilistic frameworks); (2) they require expensive retraining (e.g., ensemble methods); and (3) they rely on parametric distributional assumptions (e.g., Gaussianity). These limitations hinder practitioners from trusting ROM predictions and obstruct deployment in safety-critical scientific applications.

**Key Challenge**: ROMs aggressively compress dimensionality to achieve computational efficiency, yet no unified framework exists to characterize how structural errors from compression and approximation errors from latent dynamics propagate and interact. Uncertainty in cloud microphysics parameterizations is recognized as a primary source of uncertainty in future climate projections, yet existing schemes broadly lack UQ capability.

**Goal**: To provide a post-hoc, model-agnostic UQ framework for arbitrary black-box latent-space ROMs — requiring no modification to the underlying architecture or training procedure — while enabling independent uncertainty quantification of each component in the ROM pipeline (encoder–decoder reconstruction, latent-space dynamics, and end-to-end prediction).

**Key Insight**: Conformal prediction (CP) is leveraged for its distribution-free coverage guarantees and exchangeability assumptions, enabling statistically valid prediction intervals to be constructed separately for each of the three ROM components — achieving component-level UQ analysis for ROMs for the first time.

**Core Idea**: Conformal prediction is applied as a post-hoc tool independently to each component of the ROM pipeline, simultaneously guaranteeing statistical coverage and enabling localization of uncertainty sources.

## Method

### Overall Architecture

The ROM consists of three components: an encoder $E: \mathcal{X} \to \mathcal{Z}$ that maps high-dimensional data (DSD with $d=64$ bins) to a low-dimensional latent space ($m=4$); a dynamics model $F: \mathcal{T} \times \mathcal{Z} \to \mathcal{Z}$ that evolves the system in latent space over time; and a decoder $D: \mathcal{Z} \to \mathcal{X}$ that maps predictions back to physical space. The UQ framework independently constructs conformal prediction intervals for the outputs of these three components:

- **Reconstruction UQ**: Evaluates the compression–reconstruction error of the autoencoder $D \circ E$
- **Dynamics UQ**: Evaluates the dynamic prediction error of the latent-space ODE model $F$
- **End-to-End UQ**: Evaluates the aggregate prediction error of the full pipeline $D \circ F \circ E$

Prediction intervals are constructed independently at each time step, enabling uncertainty to be tracked as it evolves over time.

### Key Designs

1. **Asymmetric Two-Sided Prediction Intervals (for DSD outputs)**

    - **Function**: Constructs prediction intervals for DSD values with potentially different upper and lower bound widths.
    - **Mechanism**: Uses the $\alpha/2$ and $1-\alpha/2$ empirical quantiles of the signed residuals $R = y - f(x)$ to form the interval $[f(x) + Q_{\alpha/2}(R),\; f(x) + Q_{1-\alpha/2}(R)]$, rather than symmetric intervals based on absolute residuals.
    - **Design Motivation**: DSDs are non-negative and frequently near zero, making the residual distribution inherently skewed. Symmetric intervals produce physically unrealistic negative lower bounds or excessively wide upper bounds; asymmetric design more faithfully reflects true uncertainty under physical constraints.

2. **Mahalanobis Distance Scalar Score (for latent-space outputs)**

    - **Function**: Compresses the joint prediction error of multivariate latent-space predictions into a single scalar nonconformity score.
    - **Mechanism**: Computes the Mahalanobis distance $S(z, \hat{z}) = r^\top \Sigma_r^{-1} r$ of the residual $r = z - \hat{z}$, where $\Sigma_r$ is the residual covariance matrix estimated via Ledoit–Wolf shrinkage, yielding ellipsoidal prediction sets in latent space.
    - **Design Motivation**: Latent variables are correlated; constructing intervals coordinate-wise ignores the joint error structure. The Mahalanobis distance is naturally covariance-aware, and the resulting ellipsoidal prediction sets accurately capture correlated multivariate uncertainty.

3. **Comparative Framework Across Three CP Variants**

    - **Function**: Provides coverage guarantees under different trade-offs between statistical efficiency and computational cost.
    - **Mechanism**: Vanilla CP (train–test split, scores computed on training data); Split CP (60–20–20 train–validation–test split, scores on validation set); CV+ CP ($k=20$ cross-validation, aggregating residuals across folds).
    - **Design Motivation**: Different applications impose different requirements on interval precision and computational budget — CV+ yields the tightest intervals but requires $k$ retraining runs, while Vanilla CP is simplest but may be overly wide. The comparative analysis guides practitioners in selecting the appropriate variant for their setting.

### Loss & Training

The UQ framework is entirely post-hoc and involves no additional training. The underlying AE-SINDy model is trained with a composite loss:

$$L = L_{\text{recon}} + w_{dx} L_{dx} + w_{dz} L_{dz}$$

- $L_{\text{recon}}$: KL divergence between input and reconstructed DSD
- $L_{dx}$: MSE of the decoder-projected DSD time derivatives
- $L_{dz}$: MSE of the SINDy latent-space time derivatives
- Weights are automatically scaled by the relative magnitudes of the DSD and its derivatives; other hyperparameters (batch size=25, lr=0.0042) are tuned via Optuna
- Training uses the AdamW optimizer with early stopping (patience=50, maximum 1000 epochs)

## Key Experimental Results

### Main Results

Empirical coverage rate validation (%, mean ± standard deviation, averaged across all time steps and output coordinates):

| Submodel | CP Method | 90% Target | 95% Target | 98% Target | 99% Target |
|----------|-----------|-----------|-----------|-----------|-----------|
| Reconstruction | Vanilla | 88.56±3.16 | 93.86±2.30 | 96.96±1.65 | 98.00±1.38 |
| Reconstruction | Split | 87.70±4.31 | 92.87±3.62 | 96.10±2.57 | 97.34±2.17 |
| Reconstruction | CV+ | 89.04±3.09 | 94.36±2.21 | 97.34±1.60 | 98.28±1.35 |
| Latent Dynamics | Vanilla | 89.38±5.41 | 95.16±3.36 | 97.88±1.64 | 98.98±1.04 |
| Latent Dynamics | CV+ | 95.23±3.45 | 98.36±1.93 | 99.44±1.19 | 99.83±0.53 |
| End-to-End | Vanilla | 88.65±3.35 | 93.73±2.55 | 96.79±1.96 | 97.94±1.54 |
| End-to-End | CV+ | 90.56±3.57 | 95.20±2.43 | 97.64±1.68 | 98.46±1.37 |

### Ablation Study

Component-level temporal behavior of uncertainty propagation:

| Component | Interval Width over Time | Physical Interpretation | Value of Improvement |
|-----------|--------------------------|------------------------|----------------------|
| Reconstruction (AE) | Constant | Structural compression error, time-independent | **Highest** — reducing AE structural error systematically improves all predictions |
| Latent Dynamics (SINDy) | Rapid early growth, then plateaus | High uncertainty during cloud-to-rain transition, saturation after stabilization | Moderate — dynamics errors are "smoothed" by the decoder |
| End-to-End | Approximately linear growth | Dynamics errors propagate through AE smoothing, yielding linear accumulation | — |

### Key Findings

- **Spatial migration of uncertainty**: Prediction interval peaks shift systematically from small droplet scales (<50 μm cloud droplets) to large droplet scales (raindrops). Even when the initial DSD is unimodal with no significant collision–coalescence growth, uncertainty in large raindrop bins increases over time — reflecting the inherent difficulty of predicting the onset of precipitation formation (the emergence of a secondary right peak).
- **AE error dominates end-to-end uncertainty**: Reconstruction error is constant but present at every time step, while dynamics error grows but is smoothed by the decoder — indicating that improving the autoencoder architecture is more valuable for reducing overall uncertainty than optimizing the latent-space dynamics model.
- **CV+ is overly conservative**: CV+ substantially exceeds the nominal coverage level for latent dynamics (empirical coverage of 95.23% at the 90% target), with the median saturating to 100% at high coverage targets, at the cost of $k$-fold computational overhead.
- **Coverage instability at small $\alpha$**: As $\alpha$ decreases (i.e., as the target coverage level increases), consistency across CP variants deteriorates, since estimation of extreme quantiles requires larger calibration sets.

## Highlights & Insights

- **Component-level UQ is the core contribution**: Unlike approaches that only assess end-to-end prediction quality, this framework precisely localizes the origin and propagation mode of errors within the ROM pipeline — providing direct guidance for model improvement, e.g., explicitly identifying that "the AE should be prioritized over SINDy."
- **Counter-intuitive finding**: Intuitively, growing dynamics errors should dominate end-to-end uncertainty over time; however, the decoder's smoothing effect means that the constant structural error of the AE is in fact the bottleneck. This insight is only accessible through component-level UQ.
- **Physically informed interval design**: Asymmetric intervals respect the non-negativity constraint of DSDs; Mahalanobis distance captures correlations in latent space — design choices are tightly coupled to physical properties of the system.
- **Minimal deployment overhead**: The framework is entirely post-hoc and model-agnostic, requiring no modification to the underlying model, and can be directly applied to any pre-trained black-box ROM.

## Limitations & Future Work

- **Non-adaptive interval width**: Standard CP methods produce fixed-width intervals for a given output dimension and time step across all input samples. Adaptive CP variants (e.g., CQR, localized CP) could adjust interval width based on input features to better reflect per-sample uncertainty.
- **Limited data scale**: Only 494 training and 124 test samples are used; estimation of extreme quantiles is unstable. Increasing diversity in LES initial conditions could improve coverage consistency.
- **Time-step independence assumption**: Prediction intervals are constructed independently at each time step, ignoring temporal autocorrelation — accounting for sequential dependence could yield tighter joint prediction intervals.
- **Single application validation**: Although the framework is general, it is validated only on AE-SINDy applied to cloud microphysics. Extension to other ROM architectures (e.g., VAEs, NODEs) and physical settings requires further empirical study.
- **Extensibility to other UQ paradigms**: The post-hoc philosophy is not limited to CP; it can be combined with parametric prediction intervals, Bayesian credible intervals, and related approaches to provide more flexible UQ under varying requirements.

## Related Work & Insights

- **De Jong et al. (2025)** is the companion paper developing the AE-SINDy surrogate model itself. The two papers are complementary — one focuses on surrogate model construction, the other on the UQ pipeline.
- **First application of conformal prediction to ROMs**: Prior work on CP has primarily targeted classification and regression UQ; extending it to reduced-order models in scientific computing represents a valuable cross-domain contribution.
- **SINDy family** (Brunton & Champion): Sparse identification of nonlinear dynamics provides an interpretable parameterization of latent-space ODEs; combined with autoencoders, it enables data-driven physical modeling.
- **Broader inspiration**: The component-level UQ paradigm generalizes to any multi-stage pipeline — for example, in VLMs, independent UQ applied to the visual encoder, language model, and alignment module could similarly localize sources of error.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The introduction of conformal prediction to the ROM domain is a first; the component-level UQ perspective offers distinct analytical value, though the core methodology (CP itself) is not novel.
- **Practicality**: ⭐⭐⭐⭐ — The post-hoc, model-agnostic, zero-additional-training design enables direct application to existing ROMs, though the small data scale limits the persuasiveness of the validation.
- **Experimental Thoroughness**: ⭐⭐⭐ — The comparison of three CP variants is thorough, and the temporal analysis of coverage and interval width is in-depth, but evaluation is limited to a single application domain and ROM architecture.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated with tight integration of methodology and physical context; visualizations of component-level analysis (Figures 2 & 3) are intuitive and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Position: There Is No Free Bayesian Uncertainty Quantification](position_there_is_no_free_bayesian_uncertainty_quantification.md)
- [\[NeurIPS 2025\] Adjusted Count Quantification Learning on Graphs](adjusted_count_quantification_learning_on_graphs.md)
- [\[NeurIPS 2025\] Uncertainty Estimation by Flexible Evidential Deep Learning](uncertainty_estimation_by_flexible_evidential_deep_learning.md)
- [\[NeurIPS 2025\] Radar: Benchmarking Language Models on Imperfect Tabular Data](radar_benchmarking_language_models_on_imperfect_tabular_data.md)
- [\[NeurIPS 2025\] Brain-Like Processing Pathways Form in Models With Heterogeneous Experts](brain-like_processing_pathways_form_in_models_with_heterogeneous_experts.md)

</div>

<!-- RELATED:END -->
