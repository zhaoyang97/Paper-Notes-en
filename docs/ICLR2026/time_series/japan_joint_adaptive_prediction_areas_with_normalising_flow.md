---
title: >-
  [Paper Note] JAPAN: Joint Adaptive Prediction Areas with Normalising-Flows
description: >-
  [ICLR 2026][Time Series][Conformal Prediction] JAPAN uses Normalising Flows (NF) to estimate (conditional) density and employs log-density as the conformal score. By thresholding the density, it constructs prediction areas that are **geometry-independent, potentially disconnected, and context-adaptive**. While maintaining finite-sample coverage guarantees, it compresses the prediction area volume significantly more than various residual-based baselines.
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Conformal Prediction"
  - "Normalising Flows"
  - "Density Thresholding"
  - "Multivariate Regression"
  - "Time Series Forecasting"
  - "Prediction Areas"
date: 2026-05-08
content_hash: 0b228bb0f36529c9
---

# JAPAN: Joint Adaptive Prediction Areas with Normalising-Flows

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=4SxAu9zMVC](https://openreview.net/forum?id=4SxAu9zMVC)  
**Code**: To be confirmed  
**Area**: Conformal Prediction / Uncertainty Quantification / Time Series Forecasting  
**Keywords**: Conformal Prediction, Normalising Flows, Density Thresholding, Multivariate Regression, Time Series Forecasting, Prediction Areas  

## TL;DR
JAPAN uses Normalising Flows (NF) to estimate (conditional) density and employs log-density as the conformal score. By thresholding the density, it constructs prediction areas that are **geometry-independent, potentially disconnected, and context-adaptive**. While maintaining finite-sample coverage guarantees, it compresses the prediction area volume significantly more than various residual-based baselines.

## Background & Motivation

**Background**: Conformal Prediction (CP) is a model-agnostic uncertainty quantification framework. Given a significance level $\epsilon$, it constructs a prediction set $\Gamma_\epsilon(x)$ such that $P(y\in\Gamma_\epsilon(x))\ge 1-\epsilon$. This coverage guarantee is finite-sample and distribution-free, making it popular in safety-critical scenarios. Inductive Conformal Prediction (ICP) splits data into training and calibration sets, computes non-conformity scores on the calibration set, and uses the quantile as a threshold, allowing CP to scale to large models.

**Limitations of Prior Work**: Traditional CP almost exclusively uses **residual-based non-conformity scores**, such as absolute error $|y-\hat y|$. While natural in 1D regression, once the response variable is multidimensional, residuals become multivariate vectors. One must manually choose a geometry to compress them into scalars—$\ell_2$ norms yield spherical regions, and $\ell_1$ norms yield rectangular regions. These geometric constraints may not reflect the true shape of uncertainty. Worse, residual scores naturally cluster around a single mean (mode); when the predictive distribution is **multimodal**, they produce overly conservative, bloated regions that cover large low-density areas (e.g., the spiral density example in Figure 1, where spherical/ellipsoid/rectangular methods expand the region excessively).

**Key Challenge**: CP provides coverage guarantees (validity), but utility requires **efficiency**—meaning the area should be as small as possible. Theoretically, Lei et al. (2013) long ago pointed out that regions obtained by thresholding the conditional density $p(y\mid x)$ are compact and even minimal when the threshold is independent of $x$. However, the true density is unknown, and residual-based scores cannot approximate this optimal "density-split" shape.

**Goal**: Switch the non-conformity score from "distance" to "density" to construct compact prediction regions that fit the true density shape, support multimodality, allow disconnectivity, and adapt to inputs, all while retaining finite-sample coverage guarantees.

**Core Idea**: **Estimate density using Normalising Flows (NF) and use log-density as the conformal score.** NF provides tractable likelihoods $\log\hat p(y\mid x)$ for scoring and leverages its bijective structure to shift "area volume calculation" to the latent space for efficiency, avoiding expensive KDE/Monte Carlo sampling in high dimensions. When $\hat p$ is sufficiently accurate, the resulting regions approach the optimal shape under the true density threshold without losing coverage guarantees.

## Method

### Overall Architecture
JAPAN decomposes uncertainty quantification into three steps: first, train a (conditional) density estimator $\hat p(y\mid x)$ using a Normalising Flow on the training set; second, use the log-density $\alpha_j=\log\hat p(y_j\mid x_j)$ of each point in the calibration set as the conformal score and take the $(1-\epsilon)$ quantile as the threshold $\tau_\epsilon$; finally, for testing, the prediction area consists of all $y$ with log-density above the threshold $\Gamma_\epsilon(x)=\{y:\log\hat p(y\mid x)\ge\tau_\epsilon\}$. The volume of this area is efficiently estimated via latent space sampling. This framework applies to both multivariate regression and "exchangeable time-series trajectories," differing only in how context is encoded.

```mermaid
flowchart LR
    A[Training Set D_train] --> B[Train Normalising Flow<br/>Learn p̂ y|x]
    B --> C[Score Calibration Set<br/>α_j = log p̂ y_j|x_j]
    C --> D[Take 1-ε Quantile<br/>Threshold τ_ε]
    E[Test Input x] --> F[Density Thresholding Region<br/>Γ = y: log p̂≥τ_ε]
    D --> F
    F --> G[Latent Space Volume Est.<br/>Proposition 3]
```

### Key Designs

**1. Log-density Conformal Score: Replacing "Distance" with "Density".** The fundamental action of JAPAN is using NF to compute log-density via the change-of-variables formula: $\log p(y\mid x)=\log p_Z(h(y,x))+\Phi(y,x)$, where $h$ is a bijection mapping data to a standard Gaussian base distribution and $\Phi$ is the log-volume change (log-determinant of the Jacobian for discrete flows, or divergence for continuous flows). Defining the prediction region as $\Gamma_\epsilon(x)=\{y:\log\hat p(y\mid x)\ge\tau_\epsilon\}$ provides three natural properties: **geometry-independence** (no assumption of ellipsoids/rectangles), **disconnectivity** (covering only high-density clusters in multimodal cases), and **context-adaptivity** (the region changes with $x$). Coverage remains $1-\epsilon$ by construction since scores are quantiles from an exchangeable calibration set.

**2. Rank-Preserving Optimality Theory: Why estimated density suffices.** A direct concern is whether inaccurate density estimation by NF destroys efficiency. The paper provides two propositions for assurance: if the estimated density $f_\theta$ is a strictly monotonic transformation of the true density $p$, i.e., $f_\theta=g(p)$ (meaning it **preserves ranking**), then the area of the resulting region is **exactly equal** to the optimal region under the true density threshold $\mathrm{Area}(\Gamma_\epsilon)=\mathrm{Area}(\Gamma^*_\epsilon)$ (Proposition 1). More realistically, if $f_\theta$ is only an approximate monotonic transformation within a uniform error $\delta$, the area difference is bounded by $C(\delta)$, where $C(\delta)\to0$ as $\delta\to0$ (Proposition 2). In other words, **JAPAN does not require absolute density accuracy, only that the density ranking of samples is correct**, significantly relaxing the requirements for the flow model.

**3. Latent Space Volume Estimation: Making expensive area calculation cheap.** While evaluating coverage is easy (computing likelihood vs. threshold), calculating the **volume** of the prediction region is hard—naive Monte Carlo in the label space is expensive when the support is unknown or high-dimensional. JAPAN utilizes the flow's bijection to move integration to the latent space: sampling $z\sim p_Z$ from the base distribution and mapping back $y=h^{-1}(z,x)$, the area is estimated as:
$$\widehat{\mathrm{Area}}(\Gamma_\epsilon(x))=\frac{1}{N}\sum_{i=1}^N \mathbf{1}\big(\hat p(h^{-1}(z_i,x)\mid x)\ge\tau_\epsilon\big)\cdot\frac{\exp(\phi(z_i,x))}{p_Z(z_i)}.$$
This only counts latent samples falling within the region after transformation, with weights $\exp(\phi)/p_Z$ correcting for the latent-to-data volume deformation. Since the base distribution is standard Gaussian and sampling is fast, this estimation remains feasible in medium-to-high dimensions and relates to importance sampling with low variance (Proposition 3; Appendix Table 10 shows it is much faster than naive MC).

**4. Time-Series Architecture + Unified Perspective on Density Scores.** For the "exchangeable time-series trajectory" setting (each full trajectory is a data point, trajectories are i.i.d.), JAPAN encodes the history $x^{(i)}_{1:T}$ into a context vector $c^{(i)}_T$ using an RNN/Transformer, then lets the flow model the density of the future trajectory conditioned on $c^{(i)}_T$. To respect causal structure, the authors adapted the TARFLOW architecture from the image domain. Furthermore, the paper demonstrates that this framework can utilize various density scores—unconditional $p(\hat y)$, conditional $p(y\mid\hat y)$, posterior $p(x\mid y)$, latent space density (where CONTRA becomes a special case of JAPAN), and adaptive thresholds $\tau_\epsilon(x)$—unifying many existing methods under the perspective of "thresholding some form of density."

## Key Experimental Results

### Main Results: Multidimension Regression (25 random splits, target coverage 0.9)
The table reports coverage (Cov.) and prediction area (Area, lower is better):

| Method | Energy Cov. / Area | RF2D Cov. / Area | RF4D Cov. / Area | SCM Cov. / Area(×10³) |
|------|------|------|------|------|
| CONTRA | 0.88 / 18.81 | 0.91 / 5.33 | 0.89 / 59.27 | 0.89 / 61.93 |
| PCP | 0.88 / 16.58 | 0.91 / 7.39 | 0.91 / 111.63 | 0.89 / 68.75 |
| NLE | 0.87 / 21.90 | 0.91 / 15.47 | 0.90 / 2732.15 | 0.90 / 102.13 |
| CQR | 0.88 / 31.12 | 0.91 / 12.50 | 0.91 / 1180.65 | 0.91 / 84.48 |
| CFRNN | 0.90 / 56.22 | 0.91 / 27.15 | 0.92 / 3322.47 | 0.91 / 83.60 |
| **JAPAN** | **0.88 / 16.32** | **0.91 / 5.06** | **0.91 / 24.11** | **0.90 / 61.28** |

JAPAN meets coverage targets on all four datasets and achieves the smallest area in nearly every case. **Its advantage is particularly stark in the higher-dimensional RF4D** (24.11 vs. the runner-up CONTRA's 59.27, and two orders of magnitude smaller than NLE/CQR).

### Main Results: Time Series Forecasting (25 random splits)

| Method | COVID-19 Cov. / Area | Particle-1 Cov. / Area | Drone Cov. / Area | Pedestrian Cov. / Area |
|------|------|------|------|------|
| CONTRA | 0.92 / 563.64 | 0.87 / 0.90 | 0.89 / 1.51 | 0.89 / 0.55 |
| PCP | 0.91 / 610.56 | 0.87 / 1.71 | 0.89 / 3.21 | 0.88 / 1.60 |
| MCQR | 0.91 / 1276.50 | 0.91 / 2.89 | 0.86 / 3.32 | 0.86 / 1.89 |
| CFRNN | 0.92 / 927.09 | 0.97 / 3.39 | 0.99 / 5.53 | 0.97 / 2.47 |
| **JAPAN** | **0.91 / 400.94** | **0.91 / 0.89** | 0.88 / 1.47 | **0.89 / 0.50** |

JAPAN's regions are generally the tightest. **On COVID-19, it is nearly an order of magnitude smaller than baselines** (400.94 vs. CFRNN's 927). On Drone, it is slightly behind CONTRA by 0.04 in area but remains near-optimal. RCP was missing on COVID-19 due to numerical instability/score divergence, highlighting JAPAN's robustness.

### Key Findings
- **Coverage targets are almost always met; the battle is over "Area"**: All methods approach 0.9 coverage, but JAPAN's density-threshold regions are systematically smaller and more informative.
- **Higher dimensions and complex distributions increase the advantage**: In high-dimensional/multimodal scenarios like RF4D and COVID-19, residual-based methods' areas explode, while JAPAN remains compact.
- **Robustness**: JANET and CopulaCPTS can "explode" (high standard deviation) when the denominator of the auxiliary model reaches near-zero using half the calibration set. RCP suffers numerical divergence. JAPAN avoids these issues.
- **Spiral Density Visualization**: Only JAPAN's region tightly follows the spiral. CONTRA (latent sphere), PCP (union of local spheres), and ellipsoid/rectangular methods all cover large zero-density regions.

## Highlights & Insights
- **Clean Paradigm Shift**: Switching from "residual distance" to "density thresholding" in one move captures geometry-independence, disconnectivity, and context-adaptivity—properties residual-based CP struggles to achieve simultaneously.
- **Theoretically Sound**: The rank-preservation proposition indicates that "absolute density accuracy is secondary to correct ranking," which is a pragmatic justification for using NF models in engineering.
- **Latent Space Volume Estimation is the finishing touch**: It elegantly solves the hardest part of CP—"how to calculate region volume"—using the flow's bijective structure and importance sampling, without which density thresholding would be unusable in high dimensions.
- **High Unification**: It subsumes existing methods like CONTRA (latent density) as special cases and provides a family of scores (unconditional/conditional/posterior/adaptive thresholds), making the framework highly extensible.

## Limitations & Future Work
- **Dependency on NF Quality**: Although theory only requires rank accuracy, severe underfitting still disrupts density ranking (Appendix A.7 discusses underfitting and assumption violations), which may distort the regions.
- **Marginal Coverage Only**: The main framework guarantees marginal coverage. Conditional coverage relies on adaptive $\epsilon(x)$ extensions and is not strictly conditionally valid.
- **Restricted Time Series Setting**: The exchangeable trajectory assumption requires "multiple i.i.d. trajectories." It is not directly applicable to common scenarios of "a single long time series where each time step is a data point" (where exchangeability is broken).
- **Scale of Experiments**: Dataset dimensions and sizes are relatively moderate. The variance and sample requirements of latent space volume estimation in ultra-high-dimensional label spaces require further verification.

## Related Work & Insights
- **Conformal Prediction Context**: CP for multivariate responses splits into three branches—i.i.d. multivariate regression, single long time series (where time dependence breaks exchangeability, e.g., Adaptive CP/EnbPI), and sets of exchangeable trajectories. JAPAN focuses on the first and third branches.
- **Multivariate Regression CP**: RCP uses covariance for ellipsoidal regions, NLE uses context-adaptive ellipsoids, and PCP uses local $\ell_2$ spheres after sampling from generative models. JAPAN differs by "thresholding directly on the data-space density" rather than imposing geometry.
- **Generative UQ Inspiration**: This paper treats Normalising Flows (and by extension CNFs or score-based diffusion via probability-flow ODEs) as the density engine for CP. It suggests that "any generative model with tractable likelihood + efficient sampling" can be plugged into this density-thresholding conformal framework. Advances in strong flow models like TARFLOW directly translate into tighter prediction regions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Replaces residual conformal scores with NF log-density and pairs it with latent space volume estimation. The perspective is unified, subsuming methods like CONTRA, with clear theoretical support.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 multivariate regression + 4 time-series datasets, 11 baselines, 25 random splits, and includes spiral density visualizations and various score extensions. However, data scales are moderate, lacking ultra-high-dimensional stress tests.
- **Writing Quality**: ⭐⭐⭐⭐ Clear hierarchy: Motivation—Theory—Algorithm—Experiments—Extensions. Propositions and algorithms are well-articulated, though individual notations are slightly overused.
- **Value**: ⭐⭐⭐⭐ Provides tighter, more distribution-aligned prediction regions for safety-critical scenarios. The framework's ability to benefit from future generative model improvements gives it high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reliable Probabilistic Forecasting of Irregular Time Series via Marginal Consistent Flows](reliable_probabilistic_forecasting_of_irregular_time_series_through_marginalizat.md)
- [\[ICLR 2026\] Online Time Series Prediction Using Feature Adjustment](online_time_series_prediction_using_feature_adjustment.md)
- [\[ICLR 2026\] DistDF: Time-series Forecasting Needs Joint-distribution Wasserstein Alignment](distdf_time-series_forecasting_needs_joint-distribution_wasserstein_alignment.md)
- [\[ICLR 2026\] Flow-based Conformal Prediction for Multi-dimensional Time Series](flow-based_conformal_prediction_for_multi-dimensional_time_series.md)
- [\[ICLR 2026\] ResCP: Reservoir Conformal Prediction for Time Series Forecasting](rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
