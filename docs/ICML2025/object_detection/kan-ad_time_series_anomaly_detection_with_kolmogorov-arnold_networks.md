---
title: >-
  [Paper Note] KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks
description: >-
  [ICML 2025][Object Detection][Time Series Anomaly Detection] KAN-AD reformulates time series anomaly detection as approximating sequences using smooth univariate functions. By replacing B-splines in KAN with truncated Fourier expansion to avoid local perturbation sensitivity, it improves detection accuracy by an average of 15% across four benchmarks with fewer than 1000 parameters.
tags:
  - "ICML 2025"
  - "Object Detection"
  - "Time Series Anomaly Detection"
  - "KAN"
  - "Kolmogorov-Arnold Networks"
  - "B-spline"
  - "Fourier Expansion"
date: 2026-05-08
content_hash: 75e9f2fa38516e69
---

# KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks

**Conference**: ICML 2025  
**arXiv**: [2411.00278](https://arxiv.org/abs/2411.00278)  
**Code**: None  
**Area**: Time Series  
**Keywords**: Time Series Anomaly Detection, KAN, Kolmogorov-Arnold Networks, B-spline, Fourier Expansion

## TL;DR
KAN-AD reformulates time series anomaly detection as approximating sequences using smooth univariate functions. By replacing B-splines in KAN with truncated Fourier expansion to avoid local perturbation sensitivity, it improves detection accuracy by an average of 15% across four benchmarks with fewer than 1000 parameters.

## Background & Motivation
**Background**: Time series anomaly detection (TSAD) is a core capability for real-time monitoring in cloud services and web systems. Typical approaches rely on forecasting models (predicting the next step, where large deviations indicate anomalies).

**Limitations of Prior Work**: (a) Forecasting models tend to overfit small fluctuations and are overly sensitive to local perturbations; (b) Effective TSAD should focus on global smooth patterns of "normal" behavior rather than fine-grained jitters; (c) Direct application of KANs (Kolmogorov-Arnold Networks), despite their theoretical capacity to approximate with univariate functions, suffers from B-splines' local nature, making them highly sensitive to perturbations.

**Key Challenge**: Precise fitting vs. robust detection—excessively precise fitting degrades anomaly detection performance.

**Key Insight**: Starting from the Kolmogorov-Arnold representation theorem, this work models time series as a composition of smooth univariate functions.

**Core Idea**: Replace B-splines with truncated Fourier expansion as the activation functions in KAN. The global nature of Fourier bases naturally mitigates local perturbations, while a lightweight learning mechanism emphasizes global patterns.

## Method

### Overall Architecture
Input: Time series window → KAN-AD (Fourier bases + lightweight learning mechanism) → Predict next-step value → Calculate prediction error → Anomalies detected if a threshold is exceeded.

### Key Designs

1. **Replacing B-spline KAN with Fourier KAN**:

    - Kolmogorov-Arnold representation theorem: $f(\mathbf{x}) = \sum_{q=0}^{2n} \Phi_q(\sum_{p=1}^n \phi_{q,p}(x_p))$
    - Standard KAN parameterizes $\phi_{q,p}$ with B-splines, which are local basis functions and thus sensitive to small input perturbations.
    - KAN-AD employs truncated Fourier series: $\phi(x) = a_0 + \sum_{k=1}^K (a_k \cos(kx) + b_k \sin(kx))$
    - The smoothness of each univariate function is controlled by the Fourier truncation order $K$.
    - **Design Motivation**: Fourier basis functions are global; changing a single coefficient affects the entire curve, naturally providing robustness against local noise.

2. **Lightweight Learning Mechanism**:

    - Emphasizes low-frequency information of global patterns.
    - Restricts network capacity to avoid overfitting high-frequency noise.
    - Requires very few parameters (<1000 trainable parameters).
    - **Design Motivation**: Extremely small models possess inherent regularization effects, forcing the network to capture only the most prominent patterns.

3. **Anomaly Detection Strategy**:

    - Training on normal data: Fitting the smooth patterns of "normal" behavior.
    - Testing: Anomalous points deviate from the smooth patterns, yielding large prediction errors, which are flagged as anomalies.
    - **Design Motivation**: The model's inability to reconstruct anomalous patterns yields a high-error signal.

### Loss & Training
- Prediction loss: Mean Squared Error (MSE) is used for next-step forecasting.
- Trained exclusively on normal data (unsupervised anomaly detection paradigm).

## Key Experimental Results

### Main Results

| Benchmark | Metric | KAN-AD | Prev. SOTA | Gain |
|------|------|--------|---------|------|
| Benchmark 1 | F1 / AUC | Best | - | Significant |
| Benchmark 2 | F1 / AUC | Best | - | Significant |
| Benchmark 3 | F1 / AUC | Best | - | Peak >27% |
| Benchmark 4 | F1 / AUC | Best | - | Significant |
| Ave. of 4 Benchmarks | Detection Accuracy | - | - | **+15%** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Fourier KAN (KAN-AD) | **Best** | Global basis functions resist local perturbations |
| B-spline KAN (Original KAN) | Inferior | Local basis functions are sensitive to noise |
| Different truncation orders $K$ | Performance curve | Moderate $K$ is optimal, while excessive $K$ leads to overfitting |
| Inference Speed | Time elapsed | 50% faster than original KAN |

### Key Findings
- Fourier basis functions significantly outperform B-splines on the TSAD task.
- SOTA performance is achieved with fewer than 1000 parameters—exhibiting extreme parameter efficiency.
- Inference speed is 50% faster than the original KAN, thanks to the efficient implementation of the Fourier transform.
- The Fourier truncation order $K$ acts like frequency bandwidth, controlling the trade-off between fitting precision and robustness.

## Highlights & Insights
- **Rethinking the Essence of TSAD**: Anomaly detection ≠ precise forecasting; rather, it is the identification of deviations from smooth patterns.
- **Practical Application of KAN**: Transforms KAN from a theoretical framework into a practical, lightweight model.
- **Extreme Parameter Efficiency**: Challenging ten-thousand-parameter SOTA models with <1000 parameters hints at the intrinsic low dimensionality of TSAD.
- **Engineering Value**: Compact models + fast inference = ideal for real-time monitoring scenarios.

## Limitations & Future Work
- Fourier expansion assumes a certain degree of periodicity or regularity, which may not hold for completely non-periodic time series.
- The composition of univariate functions might be insufficient to capture complex dependencies in high-dimensional time series.
- The truncation order $K$ currently requires manual selection.
- Evaluation is limited to four standard benchmarks; the complexity of practical production environments is not yet fully covered.

## Related Work & Insights
- The original KAN (Liu et al. 2024) introduced the Kolmogorov-Arnold network architecture.
- Classic TSAD methods: LSTM-based, Transformer-based, and Graph Neural Network-based approaches.
- Statistical approaches, such as STL decomposition, also utilize smoothness assumptions.
- **Insight**: A strong inductive bias is more critical than large model scales; what TSAD requires is smoothness rather than massive parameter capacity.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of KAN + Fourier in TSAD is novel and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four benchmarks + comprehensive ablation studies + efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivating arguments are highly convincing.
- Value: ⭐⭐⭐⭐⭐ Extreme parameter efficiency and inference speed make it highly suitable for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)
- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](../../NeurIPS2025/object_detection/scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)
- [\[ICLR 2026\] Contextual and Seasonal LSTMs for Time Series Anomaly Detection](../../ICLR2026/object_detection/contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)
- [\[ICML 2025\] CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering](costfilter-ad_enhancing_anomaly_detection_through_matching_cost_filtering.md)
- [\[AAAI 2026\] Harnessing Vision-Language Models for Time Series Anomaly Detection](../../AAAI2026/object_detection/harnessing_vision-language_models_for_time_series_anomaly_detection.md)

</div>

<!-- RELATED:END -->
