---
title: >-
  [Paper Note] Latent Variable Estimation in Bayesian Black-Litterman Models
description: >-
  [ICML2025][Black-Litterman model] By treating the subjective investor views $(q, \Omega)$ in the classical Black-Litterman portfolio optimization model as latent variables, this paper automatically infers them from market feature data via a Bayesian network. This eliminates reliance on manual subjective inputs, improving the Sharpe ratio by approximately 50% and reducing the turnover rate by around 55% on 30-year Dow Jones and 20-year ETF datasets.
tags:
  - "ICML2025"
  - "Black-Litterman model"
  - "portfolio optimization"
  - "Bayesian networks"
  - "latent variable models"
  - "uncertainty quantification"
date: 2026-05-08
content_hash: da38a14f0f7e7150
---

# Latent Variable Estimation in Bayesian Black-Litterman Models

**Conference**: ICML2025  
**arXiv**: [2505.02185](https://arxiv.org/abs/2505.02185)  
**Code**: Not released  
**Area**: Other/Bayesian, Financial Portfolio Optimization  
**Keywords**: Black-Litterman model, portfolio optimization, Bayesian networks, latent variable models, uncertainty quantification

## TL;DR

By treating the subjective investor views $(q, \Omega)$ in the classical Black-Litterman portfolio optimization model as latent variables, this paper automatically infers them from market feature data via a Bayesian network. This eliminates reliance on manual subjective inputs, improving the Sharpe ratio by approximately 50% and reducing the turnover rate by around 55% on 30-year Dow Jones and 20-year ETF datasets.

## Background & Motivation

**Limitations of Prior Work in Classical BL Models**: The Black-Litterman (1992) model builds on Markowitz mean-variance optimization by introducing a market equilibrium prior and investor views to generate more stable portfolio weights. However, the model requires investors to manually provide subjective forecast vectors $q \in \mathbb{R}^k$ and corresponding uncertainty matrices $\Omega \in \mathbb{R}^{k \times k}$, such as "Asset 2 will outperform Asset 1 by 9±3%". This dependence leads to:

**Subjective Bias**: Views provided by different investors vary widely, making results irreproducible.

**Error Propagation from External Estimators**: Existing works use external models like GARCH, LSTM, or SVM to generate $(q, \Omega)$, but embedding independent estimators into the BL framework triggers multi-stage error accumulation.

**Inconsistent Confidence Calibration**: Estimating $q$ and $\Omega$ independently can lead to overconfidence (low $\Omega$), amplifying errors in $q$.

**Core Idea**: Treat $(q, \Omega)$ as latent variables and directly infer their posterior distributions from feature data within a unified Bayesian network, achieving end-to-end data-driven portfolio optimization.

## Method

### Overall Architecture

Based on the BLB (Black-Litterman-Bayes) model, this paper constructs a feature-integrated Bayesian network and proposes three model variants based on two causal effects of features:

- **M-BL** (Mixed Effects): Leverages both effects simultaneously, applicable when views are known.
- **SLP-BL** (Shared Latent Parameterization): Features and asset returns share parameter $\theta$ (Effect 1), suitable for asset-specific features.
- **FIV-BL** (Feature-Influenced Views): Features act indirectly by influencing views (Effect 2), suitable for macro, non-asset-specific features.

### Core Modeling

**BLB Foundation**: Asset returns follow $r \sim N(\theta, \Sigma)$, parameter prior is $\theta \sim N(\theta_0, \Sigma_0)$, and view likelihood is $P\theta = q + \epsilon, \; \epsilon \sim N(0, \Omega)$. The posterior is:

$$p(\theta | q, \Omega) = N\!\left(\theta; G^{-1}(\Sigma_0^{-1}\theta_0 + P^\top \Omega^{-1} q),\; G^{-1}\right)$$

where $G = \Sigma_0^{-1} + P^\top \Omega^{-1} P$.

### SLP-BL Model (Main Experimental Model)

Introduce a $\theta \leftrightarrow F$ linear model: $\theta = \alpha^F + F\beta^F + \epsilon^F, \; \epsilon^F \sim N(0, \Omega^F)$.

The feature matrix $F = \text{diag}(f_1^\top, \ldots, f_m^\top) \in \mathbb{R}^{m \times dm}$ has a block-diagonal structure, with each asset having $d$-dimensional features.

Posterior estimation (closed-form):

$$p(\theta | F, \Omega^F) = N\!\left(\theta;\; (G^F)^{-1}[\Sigma_0^{-1}\theta_0 + (\Omega^F)^{-1}(\alpha^F + F\beta^F)],\; (G^F)^{-1}\right)$$

where $G^F = \Sigma_0^{-1} + (\Omega^F)^{-1}$. The predictive distribution is $\tilde{r} \sim N(\mu_\theta, \Sigma + (G^F)^{-1})$.

**Key Characteristic**: Degenerates to the classical BL when features recover original views ($\alpha^F + F\beta^F \to P^{-1}q$).

### FIV-BL Model

Introduce a $q \leftrightarrow F$ linear model: $q = P(\alpha + F\beta + \epsilon^F)$, and specify an Inverse-Wishart prior for the latent variable $\Omega$, yielding a multivariate $t$-distribution posterior after marginalization. Numerical approximation is required.

### Hyperparameter Estimation

- $\Sigma$: Sample covariance; $\Sigma_0 = \tau \Sigma$ where $\tau \in (0, 1]$.
- $\Omega^F$: Constructed using Silverman's bandwidth rule based on kernel density estimation as $\hat{\Omega}^F = B\tilde{H}B^\top$.
- $(\alpha^F, \beta^F)$: Maximum likelihood estimation via regression of historical returns on features.

## Key Experimental Results

### Datasets

| Dataset | Time Span | Number of Assets |
|---|---|---|
| SPDR Sector ETFs | 2004–2024 (20 years) | 11 sector ETFs |
| Dow Jones Index | 1994–2024 (30 years) | 41 constituents |

### Main Results: SPDR Sector ETFs

| Model | Cumulative Return (%) | CAGR (%) | Sharpe | Max Drawdown (%) | Annualized Volatility (%) |
|---|---|---|---|---|---|
| S&P500 | 545.77 | 6.69 | 0.59 | 55.19 | 19.03 |
| MV(100d) | 411.83 | 5.84 | 0.57 | 36.37 | 16.91 |
| **BL(100d)** | **602.75** | **7.01** | **0.70** | 46.05 | **15.91** |
| MV(150d) | 249.11 | 4.44 | 0.45 | 47.49 | 17.37 |
| **BL(150d)** | **556.13** | **6.75** | **0.68** | 44.54 | **15.91** |

### Main Results: Dow Jones Index

| Model | Cumulative Return (%) | CAGR (%) | Sharpe | Max Drawdown (%) | Annualized Volatility (%) |
|---|---|---|---|---|---|
| DJIA | 932.51 | 5.49 | 0.52 | 53.78 | 17.97 |
| MV(120d) | 1577.61 | 6.67 | 0.57 | 46.73 | 20.07 |
| **BL(120d)** | **4819.83** | **9.33** | **0.87** | **39.81** | **16.42** |
| MV(100d) | 1529.60 | 6.60 | 0.55 | 56.06 | 20.59 |
| **BL(100d)** | **4557.03** | **9.19** | **0.85** | 39.92 | **16.56** |

### Key Findings

- **Sharpe Ratio**: SLP-BL delivers an average improvement of 49.8% over Markowitz (0.66–0.70 vs 0.35–0.57 on ETFs; 0.78–0.87 vs 0.45–0.62 on DJIA).
- **Volatility**: The BL model exhibits lower volatility than the corresponding MV model across all window lengths (reducing it by approximately 1–4 percentage points).
- **Turnover Rate**: Reduced by approximately 55.1%, attributed to the more stable portfolio weights under the Bayesian framework.
- **Robustness to Hyperparameters**: The BL model consistently outperforms MV across all 5 window lengths (50/80/100/120/150 days).
- **Cumulative Return**: On the DJIA dataset, BL(120d) achieves a cumulative return of 4819% vs 1577% for MV(120d).

## Highlights & Insights

1. **Elegant Unified Framework**: Integrates feature integration and parameter inference into a single Bayesian network, avoiding error propagation in multi-stage pipelines. The closed-form solution ensures fast and stable inference.
2. **Theoretical Degeneration Properties**: Demonstrates that both classical BL and Markowitz models are special cases of the proposed model, and that true returns can be recovered under the limit of perfect information.
3. **Complementary Design of Two Configurations**: SLP-BL handles asset-specific features, whereas FIV-BL handles macro/non-asset-specific features (e.g., interest rates, CPI). They can be combined in practice.
4. **Complete Elimination of Subjective Inputs**: First to achieve end-to-end inference from feature data within the BL framework without requiring manually specified views.

## Limitations & Future Work

1. **Limited Feature Selection**: Utilizes only 9 generic technical indicators based on price/volume, without incorporating fundamental, macroeconomic, or alternative data.
2. **No Experiments on FIV-BL**: Experiments in the paper only validate SLP-BL; FIV-BL requires numerical approximation methods (e.g., MCMC), and its computational cost and practicality are not discussed.
3. **Linear Assumption**: The relationship between features and parameters is assumed to be linear, which may fail to capture non-linear market dynamics.
4. **Lack of Transaction Costs**: Backtesting does not account for real-world frictions such as transaction costs and slippage.
5. **Monthly Rebalancing Only**: High-frequency or dynamic rebalancing strategies are not explored.
6. **Weak Baselines**: No comparison with modern quantitative finance methods such as deep learning or reinforcement learning.

## Related Work & Insights

- **Black & Litterman (1992)**: The classical BL model, which is the direct target for improvement in this work.
- **Kolm & Ritter (2017, 2021)**: The BLB model, which reformulates BL into a Bayesian inference framework, though it still relies on subjective views.
- **Beach & Orlov (2007); Kara et al. (2019)**: Representative works using external models (e.g., GARCH, SVM) to generate views.
- **Markowitz (1952)**: Classical mean-variance optimization, the baseline in this paper.
- Insight: The concept of "converting heuristic inputs into latent variables" can be generalized to other Bayesian models requiring expert knowledge.

## Rating

- Novelty: ⭐⭐⭐⭐ (Treating BL views as latent variables is a natural yet under-explored direction)
- Experimental Thoroughness: ⭐⭐⭐ (Long-term real-world data is used, but modern baselines and FIV-BL experiments are lacking)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear, but LaTeX notations are dense, and the approximation handling of FIV-BL is somewhat rushed)
- Value: ⭐⭐⭐⭐ (Directly relevant to quantitative finance practice, with solid theoretical contributions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Feature Selection for Latent Factor Models](../../CVPR2025/others/feature_selection_for_latent_factor_models.md)
- [\[ICML 2025\] How Do Transformers Learn Variable Binding in Symbolic Programs?](how_do_transformers_learn_variable_binding_in_symbolic_programs.md)
- [\[ACL 2025\] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](../../ACL2025/others/latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)
- [\[ICML 2025\] Improving Generalization with Flat Hilbert Bayesian Inference](improving_generalization_with_flat_hilbert_bayesian_inference.md)
- [\[ICML 2025\] Prediction-Powered Adaptive Shrinkage Estimation](prediction-powered_adaptive_shrinkage_estimation.md)

</div>

<!-- RELATED:END -->
