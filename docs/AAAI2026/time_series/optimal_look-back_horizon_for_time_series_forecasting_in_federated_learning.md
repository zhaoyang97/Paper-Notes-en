---
title: >-
  [Paper Note] Optimal Look-back Horizon for Time Series Forecasting in Federated Learning
description: >-
  [AAAI 2026][Time Series][time series forecasting] This paper proposes a theoretical framework for selecting the optimal look-back horizon in federated time series forecasting. By introducing a Synthetic Data Generator (S…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "time series forecasting"
  - "federated learning"
  - "look-back horizon"
  - "intrinsic space"
  - "Bayesian loss decomposition"
date: 2026-05-08
content_hash: cecceebb6ec3afdf
---

# Optimal Look-back Horizon for Time Series Forecasting in Federated Learning

**Conference**: AAAI 2026
**arXiv**: [2511.12791](https://arxiv.org/abs/2511.12791)  
**Code**: None  
**Area**: Time Series Forecasting / Federated Learning
**Keywords**: time series forecasting, federated learning, look-back horizon, intrinsic space, Bayesian loss decomposition

## TL;DR

This paper proposes a theoretical framework for selecting the optimal look-back horizon in federated time series forecasting. By introducing a Synthetic Data Generator (SDG) and an intrinsic space representation, the forecasting loss is decomposed into an irreducible Bayesian error and an approximation error. The paper proves that the total loss is unimodal with respect to the horizon length, and establishes that the minimum sufficient window is the optimal solution.

## Background & Motivation

The choice of look-back horizon $H$ is a central modeling decision in time series forecasting (TSF), directly affecting model complexity and prediction accuracy. Conventional practice treats $H$ as a hyperparameter tuned via cross-validation, without theoretical guidance.

The scaling law theory of Shi et al. (2024) embeds time series into an intrinsic representation space and decomposes the forecasting loss into Bayesian and approximation errors, but **assumes centralized, IID data and homogeneous model architectures**. In federated learning (FL), data are distributed across heterogeneous clients with varying distributions, sequence lengths, and domain characteristics, so a globally fixed horizon may cause a mismatch between local dynamics and model inputs.

**Core Problem**: *How can the optimal look-back horizon be adaptively determined for each client in a non-IID federated setting?*

This paper extends Shi's theory to the federated non-IID setting, introduces an SDG to capture client heterogeneity, constructs an intrinsic representation space satisfying geometric and statistical properties, and derives a closed-form expression for the optimal horizon.

## Method

### Overall Architecture

The framework consists of four levels: (1) an SDG that models the core structure of each client's time series (AR + seasonality + trend + noise); (2) a five-step transformation pipeline mapping time series windows to the intrinsic space; (3) loss decomposition into Bayesian (irreducible) and approximation terms; and (4) a proof of unimodality and derivation of the optimal horizon.

### Key Designs

**1. Synthetic Data Generator (SDG)**

The model for client $k$, feature $f$, and time step $t$ is:

$$\hat{x}_{f,t,k} = \sum_{j=1}^{J} A_{f,j,k} \cdot \sin\left(\frac{2\pi t}{T_{f,j,k}} + \theta_{f,j,k}\right) + \sum_{i=1}^{p} \phi_{k,i} x_{f,t-i,k} + \beta_{f,k} t + \epsilon_{f,t,k}$$

where seasonality is represented by a sum of sinusoids (amplitude $A$, period $T$, phase $\theta$), temporal dependence is modeled via AR(p) with client-specific coefficients $\phi_{k,i}$, the trend is linear, and noise is $\epsilon_{f,t,k} \sim \mathcal{N}(\mu_{f,k}, \sigma_{f,k}^2)$.

Client heterogeneity is introduced through an affine transformation that induces feature skew: $x_{f,t,k} = \Lambda_{f,k} \tilde{x}_{f,t,k} + \delta_{f,k}$.

The SDG is validated on real temperature data with strong fit: mean deviation $\approx 3.6 \times 10^{-3}$, ACF $L^2 \approx 3.7 \times 10^{-6}$, KS statistic = 0.042.

**2. Intrinsic Space Construction and Loss Decomposition**

The five-step transformation pipeline proceeds as follows: (1) client-level normalization to remove affine skew; (2) flattening windows into vectors; (3) global covariance estimation and eigendecomposition; (4) intrinsic dimensionality estimation based on the SDG; and (5) PCA projection onto the intrinsic space.

The intrinsic dimensionality is estimated as:

$$d_{I,k}(H) \approx F \cdot \left(\min\{H, \ell_{\mathrm{AR},k}\} + g_k(H) + 1\right)$$

where the effective AR memory length is $\ell_{\mathrm{AR},k} = \lceil \frac{\ln(1/(1-\epsilon))}{-\ln \rho_k} \rceil$, and the seasonal complexity is $g_k(H) = 2\sum_{j=1}^{J} w_{j,k} \cdot \min(1, H/T_{j,k}^*)$.

The federated loss decomposition (Theorem 1) is:

$$L(H,S;m) = L_{\mathrm{Bayes}}(H,S) + L_{\mathrm{approx}}(H,S;m)$$

with server-level aggregation: $L_{\mathrm{Bayes}}^{(\mathrm{server})} = \sum_{k=1}^{K} \pi_k L_{\mathrm{Bayes}}^{(k)}$.

**3. Optimal Horizon Theory**

**The Bayesian loss decreases monotonically in $H$ and saturates** (more history → better identification of seasonal/AR structure). The client-level decomposition yields three terms:

$$L_{\mathrm{Bayes}}^{(k)}(H,S) = L_{\mathrm{AR}}^{(k)}(S) + L_{\mathrm{seas}}^{(k)}(H) + L_{\mathrm{trend}}^{(k)}(H)$$

Upper bound on the AR term: $L_{\mathrm{AR}}^{(k)}(S) \leq \sum_f \sigma_{f,k}^2 \cdot \frac{1 - \rho_k^{2S}}{1-\rho_k^2}$

**The approximation loss increases monotonically in $H$** (higher intrinsic dimensionality + fewer effective samples):

$$L_{\mathrm{approx}}^{(k)}(H;m) \lesssim \left(K_2^2 d_{I,k}(H)^2\right)^{\frac{d_{I,k}(H)}{4+d_{I,k}(H)}} + \left(\frac{d_{I,k}(H) H}{D_k}\right)^{\frac{4}{4+d_{I,k}(H)}}$$

**Theorem 4 (Unimodality and Optimal Horizon)**: The total loss is strictly decreasing on $[1, H_k^*(\delta)]$ and strictly increasing on $[H_k^*(\delta), \infty)$. The minimum sufficient window $H_k^*(\delta)$ is therefore the optimal solution:

$$H_k^*(\delta) = \max\{\ell_{\mathrm{AR},k}, T_k^{(\tau)}\}$$

The federated global horizon is aggregated via a weighted trimmed mean: $H_{\mathrm{server}}^* = \mathrm{TrimMean}_\alpha(\{H_k^*(\delta)\}; \{w_k\})$

### Loss & Training

This paper is a purely theoretical work and does not involve specific training strategies. The loss analysis is based on the expected squared loss $\|V - m(U)\|^2$, derived within the intrinsic space.

## Key Experimental Results

### Main Results

This paper is primarily a theoretical analysis. The main experiments consist of SDG validation:

| Metric | Value |
|--------|-------|
| Mean deviation $\Delta\mu$ | $\approx 3.6 \times 10^{-3}$ |
| ACF $L^2$ gap (30 lags) | $\approx 3.7 \times 10^{-6}$ |
| Normalized PSD $L^2$ gap | $\approx 8.2 \times 10^{-3}$ |
| KS statistic | 0.042 |
| Random forest discrimination accuracy (window of 50) | 0.892 |

Experiments use 2020 Jena weather station temperature data (10-minute resolution, $N = 52{,}696$), with AR order $p=30$ and primary period $T_1 = 144$ (daily cycle).

### Ablation Study

**Intrinsic dimensionality as a function of horizon $H$** (theoretical analysis):

| $H$ range | Behavior of $d_{I,k}(H)$ |
|-----------|--------------------------|
| $H < \ell_{\mathrm{AR},k}$ | Linear growth (AR information not yet saturated) |
| $\ell_{\mathrm{AR},k} \leq H < \max T_{j,k}^*$ | Sublinear growth (seasonality progressively resolved) |
| $H \geq H_{\mathrm{id}}$ | Saturation, no further growth |

**Behavior of loss components as $H$ increases**:

| Loss term | Trend with increasing $H$ | Mechanism |
|-----------|--------------------------|-----------|
| $L_{\mathrm{Bayes}}$ | Monotonically decreasing → saturates | More history → better identification of AR/seasonality |
| $L_{\mathrm{approx}}$ | Monotonically increasing | Higher intrinsic dimensionality + fewer effective samples ($D_k/H$) |
| $L_{\mathrm{total}}$ | Decreases then increases (unimodal) | Trade-off is optimized at $H_k^*$ |

### Key Findings

- The look-back horizon has a theoretically optimal value; both excessively short (insufficient information) and excessively long (overfitting) horizons degrade performance.
- The optimal horizon is determined by two factors: the effective AR memory length and seasonal period coverage.
- The optimal horizon differs across clients, depending on local AR structure and seasonal characteristics.
- The federated setting requires robust aggregation (trimmed mean) to mitigate the influence of extreme clients.

## Highlights & Insights

- **First theoretical framework for horizon selection in federated TSF**: Extends the centralized theory of Shi et al. to the non-IID federated setting.
- **Interpretable closed-form optimal horizon**: $H_k^* = \max\{\ell_{\mathrm{AR},k}, T_k^{(\tau)}\}$ is determined directly from signal parameters (AR memory, seasonal period).
- **Rigorous proof of unimodality**: Provides a theoretical guarantee for horizon selection, eliminating the need for exhaustive search.

## Limitations & Future Work

- As a purely theoretical work, the paper lacks large-scale empirical validation on real federated datasets.
- The SDG assumes an additive structure (AR + seasonality + trend + Gaussian noise), and does not capture nonlinear interactions, regime switching, or other complex patterns.
- The framework assumes local stationarity and stable AR structure, which may break down in long-memory or near-unit-root settings.
- Global covariance estimation in the federated setting requires secure aggregation; privacy implications are not discussed.
- Overlapping windows are approximated as independent samples, which may overestimate the effective sample size.

## Related Work & Insights

- **vs. Shi et al. (2024)**: Shi et al. establish a scaling law in the centralized IID setting; this paper extends it to the federated non-IID setting by introducing client-specific SDGs and intrinsic spaces.
- **vs. FedProx**: FedProx stabilizes federated optimization via regularization but does not address horizon selection; this paper provides the theoretical foundation for that choice.
- **vs. iTransformer/NLinear**: Empirical findings show that different models favor different optimal horizons (channel-dependent models prefer shorter horizons; linear models prefer longer ones); this paper offers a theoretical explanation for this phenomenon.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First theoretical framework establishing optimal horizon selection in federated TSF
- Experimental Thoroughness: ⭐⭐⭐ Limited to SDG validation; no comparative experiments on real federated datasets
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with consistent notation
- Value: ⭐⭐⭐⭐ Meaningful theoretical contribution, though practical utility awaits empirical verification

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](../../ICLR2026/time_series/fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[AAAI 2026\] Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching](detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)
- [\[AAAI 2026\] Harmonic Dataset Distillation for Time Series Forecasting](harmonic_dataset_distillation_for_time_series_forecasting.md)
- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)

</div>

<!-- RELATED:END -->
