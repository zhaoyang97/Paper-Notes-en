---
title: >-
  [Paper Note] ResCP: Reservoir Conformal Prediction for Time Series Forecasting
description: >-
  [ICLR 2026][Time Series][conformal prediction] This work is the first to integrate Reservoir Computing (Echo State Network) into conformal prediction. By using randomly initialized ESNs to encode the temporal dynamics of residual sequences, the method leverages state similarity to adaptively reweight historical residuals for constructing local prediction intervals—requiring no training—and achieves state-of-the-art Winkler scores on 4 real-world datasets while running 20–80× faster than HopCPT.
tags:
  - ICLR 2026
  - Time Series
  - conformal prediction
  - reservoir computing
  - echo state network
  - prediction interval
  - training-free
date: 2026-05-08
content_hash: b64985ca31572ac6
---

# ResCP: Reservoir Conformal Prediction for Time Series Forecasting

**Conference**: ICLR 2026
**arXiv**: [2510.05060](https://arxiv.org/abs/2510.05060)
**Code**: None
**Area**: Time Series / Uncertainty Quantification
**Keywords**: conformal prediction, reservoir computing, echo state network, prediction interval, training-free

## TL;DR
This work is the first to integrate Reservoir Computing (Echo State Network) into conformal prediction. By using randomly initialized ESNs to encode the temporal dynamics of residual sequences, the method leverages state similarity to adaptively reweight historical residuals for constructing local prediction intervals—requiring no training—and achieves state-of-the-art Winkler scores on 4 real-world datasets while running 20–80× faster than HopCPT.

## Background & Motivation

**Background**: Conformal Prediction (CP) is a powerful framework for constructing distribution-free prediction intervals, but it requires data exchangeability—a property that is inherently violated by the temporal dependencies in time series.

**Limitations of Prior Work**:
- Fixed-decay methods such as NexCP cannot adapt to local dynamics, resulting in overly conservative (wide) intervals.
- HopCPT uses Hopfield/Transformer attention for data-dependent reweighting, but training is expensive (4,574 seconds on the Solar dataset vs. 53 seconds for ResCP), and the model must be retrained under distribution shift.
- SPCI fits a quantile random forest at each step, limiting practical scalability.
- Training-based methods (CP-QRNN, ResCQR) suffer from severe undercoverage (>10%) on small datasets such as ACEA and Exchange.

**Key Challenge**: Data-dependent adaptive reweighting is needed to capture local dynamics, yet training a model for this purpose is costly and fragile under distribution shift.

**Goal**: Achieve local adaptivity in time series conformal prediction without introducing any training.

**Key Insight**: Reservoir Computing (RC) with Echo State Networks (ESNs)—randomly initialized RNNs that require no training yet map input sequences into a high-dimensional state space, producing meaningful dynamic representations.

**Core Idea**: Use similarity between ESN states as data-dependent weights for residual reweighting, effectively realizing locally adaptive conformal prediction via a "free" dynamic encoder.

## Method

### Overall Architecture

Given the residual sequence $\{r_t\}$ of a point forecasting model, ResCP constructs prediction intervals through the following pipeline: (1) encode the residual sequence into a state sequence $\{h_t\}$ via an ESN; (2) compute the similarity between the current state $h_T$ and each state in the calibration set as weights; (3) construct the prediction interval from the quantiles of the weighted empirical distribution.

### Key Designs

1. **ESN State Encoding (Reservoir Embedding)**:
    - Function: Map the residual sequence to a high-dimensional state space to capture local temporal dynamics.
    - Mechanism: ESN state update $\boldsymbol{h}_t = (1 - l)\boldsymbol{h}_{t-1} + l\,\sigma(\boldsymbol{W}_x \boldsymbol{x}_t + \boldsymbol{W}_h \boldsymbol{h}_{t-1} + \boldsymbol{b})$, where $\boldsymbol{W}_x, \boldsymbol{W}_h$ are randomly initialized and fixed, $l$ is the leak rate, and $\sigma = \tanh$.
    - Design Motivation: When the Echo State Property holds ($\rho(\boldsymbol{W}_h) < 1$), the ESN state asymptotically forgets initial conditions, produces similar states for similar input sequences, and constitutes a Lipschitz-continuous mapping—providing the foundation for theoretical guarantees.

2. **Similarity-Based Reweighting**:
    - Function: Assign weights to historical residuals based on reservoir state similarity, so that residuals from dynamically similar time steps receive higher weights.
    - Mechanism: Weights are computed via softmax-normalized similarity scores: $w_s(\boldsymbol{h}_t) = \text{SoftMax}\left(\frac{\text{Sim}(\boldsymbol{h}_t, \boldsymbol{h}_s)}{\tau}\right)$, where $\text{Sim}$ denotes cosine similarity and $\tau$ is a temperature hyperparameter. The weighted empirical CDF approximates the conditional distribution: $\hat{F}(r \mid \boldsymbol{h}_t) = \sum_{s} w_s(\boldsymbol{h}_t)\mathbb{1}(r_{s+H} \leq r)$.
    - Design Motivation: Temperature $\tau$ controls the bias–variance trade-off—low temperature concentrates mass on the most similar states (low bias), while high temperature approaches uniform weights, recovering vanilla SCP (low variance). The effective sample size $m_n = (\sum_i w_i^2)^{-1}$ must diverge as $n \to \infty$.

3. **Time-Dependent Weights and Distribution Shift Handling**:
    - Function: Superimpose temporal decay on similarity weights to handle non-stationary data.
    - Mechanism: $w_i(\boldsymbol{h}_t, t) = \gamma(\Delta(t,i)) \cdot w_i(\boldsymbol{h}_t)$, using linear decay $\gamma(\Delta) = 1/\Delta$ combined with a FIFO sliding window to update the calibration set.
    - Design Motivation: Linear decay is milder than exponential decay, preserving a sufficient effective sample size. The sliding window enables the calibration set to track distribution shifts, allowing ResCP to adapt without retraining.

### Loss & Training

ResCP **requires no training whatsoever**—ESN weights are randomly initialized and kept fixed. Hyperparameters (spectral radius, leak rate, input scaling, temperature, window size) are selected via grid search on a validation set by minimizing the Winkler score; because no training is involved, the search is extremely fast.

Prediction intervals are approximated via Monte Carlo sampling of weighted quantiles, with the interval width optimized using the optimal $\beta^*$: $\beta^* = \arg\min_{\beta \in [0,\alpha]} [\hat{Q}_{1-\alpha+\beta}(\boldsymbol{h}_t) - \hat{Q}_\beta(\boldsymbol{h}_t)]$.

## Key Experimental Results

### Main Results ($\alpha=0.1$, RNN base model)

| Dataset | Method | ΔCov(%) | PI Width↓ | Winkler↓ |
|--------|------|---------|---------|----------|
| Solar | HopCPT | -1.64 | 60.49 | 112.46 |
| Solar | CP-QRNN | -0.26 | 55.74 | **78.42** |
| Solar | **ResCP** | **0.74** | **62.25** | 104.24 |
| Exchange | HopCPT | 2.75 | 0.0404 | 0.0482 |
| Exchange | **ResCP** | **1.13** | **0.0210** | **0.0264** |
| ACEA | HopCPT | -2.18 | 18.90 | 27.56 |
| ACEA | CP-QRNN | -12.37 | 15.86 | 32.61 |
| ACEA | **ResCP** | **1.56** | **9.61** | **12.91** |

### Runtime Comparison (seconds, RNN base model)

| Dataset | SPCI | HopCPT | CP-QRNN | **ResCP** | SCP |
|--------|------|--------|---------|-----------|-----|
| Solar | 1040 | 4575 | 172 | **53** | 18 |
| Beijing | 351 | 1839 | 82 | **35** | 9 |
| Exchange | 51 | 318 | 37 | **7** | 2 |
| ACEA | 228 | 2263 | 95 | **71** | 7 |

### Ablation Study

| Configuration | Exchange Winkler↓ | ACEA Winkler↓ | Note |
|------|-------------------|---------------|------|
| ResCP (full) | **0.0264** | **12.91** | Temporal decay + sliding window |
| No decay | 0.0269 | 13.41 | Removing temporal decay worsens undercoverage |
| No window | 0.0284 | 14.80 | Uses all history instead of sliding window |
| No window, no decay | 0.0291 | 15.25 | Degenerates to global similarity |

### Key Findings
- ResCP substantially outperforms all methods (including training-based ones) on ACEA and Exchange in terms of Winkler score, and is competitive with training-based methods on Solar and Beijing.
- Training-based methods (CP-QRNN, ResCQR) suffer from severe undercoverage (−12% to −27%) on the small-data ACEA dataset, while ResCP consistently maintains valid coverage.
- Calibration curves show that ResCP provides accurate estimates across all coverage levels; NexCP is well-calibrated but produces intervals 1.5–2× wider.
- ResCP runs 20–80× faster than HopCPT and requires no GPU-intensive training.

## Highlights & Insights
- **Elegant use of reservoir computing**: The ESN serves as a free "temporal dynamic encoder"—no training is needed, yet it produces representations sufficiently discriminative for local dynamics. This is the paper's central insight.
- **Complete theoretical guarantees**: Under reasonable assumptions (α-mixing, Echo State Property, continuity of the conditional CDF), the paper proves consistency of the weighted empirical CDF (Theorem 3.6) and asymptotic conditional coverage (Corollary 3.7).
- **Natural robustness to distribution shift**: ResCP has no learnable parameters, so model updates are unnecessary under distribution shift; adaptation is handled automatically via the sliding window and temporal decay.

## Limitations & Future Work
- ESN hyperparameters (spectral radius, leak rate, temperature, etc.) require grid search tuning, which, though fast, adds user burden.
- Theoretical guarantees are asymptotic; coverage deviation under finite samples is not quantified.
- The method addresses only univariate time series with single-step prediction; multi-step joint prediction and extension to spatiotemporal data are promising future directions.
- In settings with large datasets and highly informative exogenous variables (e.g., Solar), training-based methods such as CP-QRNN may still be preferable.

## Related Work & Insights
- **vs. HopCPT**: Both use data-dependent attention-style weights, but HopCPT requires end-to-end Transformer training, whereas ResCP is entirely training-free and achieves superior performance.
- **vs. NexCP**: NexCP applies data-independent exponential decay; coverage is reliable but interval widths are 1.5–2× those of ResCP.
- **vs. SPCI**: SPCI fits a quantile random forest at each step, which is computationally expensive and difficult to scale; ResCP achieves comparable local adaptivity using a fixed ESN.

## Rating
- Novelty: ⭐⭐⭐⭐ First combination of reservoir computing and conformal prediction; conceptually simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 datasets × 3 base models × 3 coverage levels + full ablation + runtime analysis.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear; experimental design is systematic.
- Value: ⭐⭐⭐⭐ Provides a simple, fast, and theoretically grounded practical tool for uncertainty quantification in time series.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction](../../AAAI2026/time_series/hydrodcm_hydrological_domain-conditioned_modulation_for_cross-reservoir_inflow_p.md)
- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)

<!-- RELATED:END -->
