---
title: >-
  [Paper Note] Selective Learning for Deep Time Series Forecasting
description: >-
  [NeurIPS 2025][Time Series][time series forecasting] This paper proposes a Selective Learning strategy that employs a dual-mask mechanism—comprising an uncertainty mask and an anomaly mask—to identify generalizable time…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "time series forecasting"
  - "selective learning"
  - "overfitting"
  - "uncertainty"
  - "anomaly detection"
date: 2026-05-08
content_hash: 7bf123c503361332
---

# Selective Learning for Deep Time Series Forecasting

**Conference**: NeurIPS 2025
**arXiv**: [2510.25207](https://arxiv.org/abs/2510.25207)  
**Code**: Unavailable  
**Area**: Time Series Forecasting
**Keywords**: time series forecasting, selective learning, overfitting, uncertainty, anomaly detection

## TL;DR
This paper proposes a Selective Learning strategy that employs a dual-mask mechanism—comprising an uncertainty mask and an anomaly mask—to identify generalizable time steps for MSE loss computation. The approach achieves an average MSE reduction of 37.4% for Informer, 8.4% for TimesNet, and 6.5% for iTransformer across 8 benchmark datasets.

## Background & Motivation
**Background**: Deep learning has achieved remarkable progress in time series forecasting (TSF) by capturing complex temporal patterns.

**Limitations of Prior Work**: Deep models are susceptible to noise and anomalies in time series, leading to severe overfitting. The dominant deep learning paradigm optimizes all time steps uniformly using MSE loss, indiscriminately learning from uncertain and anomalous time steps.

**Key Challenge**: Models are required to learn from all time steps; however, some time steps correspond to noise or anomalies, and forcing the model to fit these steps causes overfitting.

**Key Insight**: Not all time steps are worth learning from; selectively masking non-generalizable time steps can improve model performance.

## Method

### Overall Architecture
The selective learning module is inserted into the training pipeline of standard deep TSF models: the model generates predictions → residuals are computed → non-generalizable time steps are identified via the dual-mask mechanism → MSE loss is computed only over the retained time steps → backpropagation is performed.

### Key Designs
1. **Uncertainty Mask**

    - **Function**: Filters out time steps with high predictive uncertainty.
    - **Mechanism**: Residual entropy is employed to quantify prediction uncertainty.
    - **Formula**: $M_u(t) = \mathbb{1}[H(r_t) < \tau_u]$, where $H(r_t)$ denotes the entropy estimate of the residual.
    - **Design Motivation**: High-uncertainty time steps likely correspond to noise; forcing the model to fit them induces overfitting.

2. **Anomaly Mask**

    - **Function**: Excludes anomalous time steps from loss computation.
    - **Mechanism**: Anomalies are detected via residual lower bound estimation.
    - **Formula**: $M_a(t) = \mathbb{1}[|r_t| < \text{LB}(r)]$
    - **Design Motivation**: Anomalous values distort gradient directions.

3. **Dual-Mask Fusion**

    - Final mask: $M(t) = M_u(t) \cdot M_a(t)$
    - Selective MSE: $\mathcal{L} = \frac{1}{|\{t: M(t)=1\}|} \sum_{t: M(t)=1} (y_t - \hat{y}_t)^2$

### Loss & Training
- **Plug-and-play**: Directly applicable to any deep TSF model without architectural modifications.
- Mask thresholds are determined adaptively using the validation set.
- A warm-up phase is applied at the beginning of training, during which masking is disabled to allow the model to first learn basic temporal patterns.

## Key Experimental Results

### Main Results: MSE Improvement (Average over 8 Datasets)

| Base Model | Original MSE | + Selective Learning | Gain (%) |
|------------|-------------|---------------------|----------|
| Informer | 0.847 | 0.530 | **37.4%** |
| Autoformer | 0.612 | 0.503 | 17.8% |
| FEDformer | 0.542 | 0.476 | 12.2% |
| TimesNet | 0.414 | 0.379 | **8.4%** |
| PatchTST | 0.386 | 0.363 | 6.0% |
| iTransformer | 0.371 | 0.347 | **6.5%** |

### Per-Dataset Results (iTransformer, Forecast Horizon 96)

| Dataset | Original MSE | +SL MSE | Gain (%) |
|---------|-------------|---------|----------|
| ETTh1 | 0.386 | 0.358 | 7.3 |
| ETTh2 | 0.340 | 0.318 | 6.5 |
| ETTm1 | 0.334 | 0.312 | 6.6 |
| ETTm2 | 0.180 | 0.168 | 6.7 |
| Weather | 0.174 | 0.164 | 5.7 |
| ECL | 0.168 | 0.157 | 6.5 |
| Traffic | 0.395 | 0.372 | 5.8 |
| Solar | 0.233 | 0.215 | 7.7 |

### Ablation Study

| Configuration | MSE (ETTh1) | Gain (%) |
|---------------|-------------|----------|
| Uncertainty mask only | 0.369 | 4.4 |
| Anomaly mask only | 0.372 | 3.6 |
| Dual mask (full) | **0.358** | **7.3** |
| Without warm-up | 0.365 | 5.4 |

### Key Findings
- The largest improvements are observed on weaker models (Informer, 37.4%), while consistent gains are also achieved on stronger models (iTransformer, 6.5%).
- The two mask components are complementary; using both jointly yields the best performance.
- The plug-and-play nature of the method enables broad applicability.

## Highlights & Insights
- **Elegant and effective idea**: The method modifies only the loss computation without altering the model architecture.
- **Strong generalizability**: Consistent improvements are demonstrated across six architecturally diverse models.
- More substantial gains on weaker models suggest that such models are more susceptible to the adverse effects of noise and anomalies.

## Limitations & Future Work
- Mask threshold selection still requires validation set tuning.
- Theoretical analysis of why selective learning improves generalization remains insufficient.
- Cross-variate selective learning (i.e., masking across variables in multivariate settings) has not been explored.

## Related Work & Insights
- Informer (Zhou et al. 2021), TimesNet (Wu et al. 2023), iTransformer (Liu et al. 2024)
- Conceptual connections to Curriculum Learning
- **Insight**: Importance-weighted training samples hold significant untapped potential in TSF.

## Rating
- **Novelty**: ⭐⭐⭐⭐ A concise yet effective new perspective
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 8 datasets, 6 models, and ablation studies
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and thorough experimental analysis
- **Value**: ⭐⭐⭐⭐⭐ Plug-and-play design with high practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_total_electron_con.md)
- [\[ICML 2026\] Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series](../../ICML2026/time_series/position_current_benchmarking_hinders_real_progress_in_deep_learning_for_time_se.md)
- [\[NeurIPS 2025\] TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)
- [\[NeurIPS 2025\] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)

</div>

<!-- RELATED:END -->
