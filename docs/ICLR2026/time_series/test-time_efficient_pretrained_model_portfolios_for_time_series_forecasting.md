---
title: >-
  [Paper Note] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting
description: >-
  [ICLR 2026][Time Series][Model portfolio] This paper proposes Chroma — a portfolio framework of small pretrained time series models: frequency/domain expert models are derived from a general model via post-training (achieving 10× training speedup), and at test time predictions are combined through model selection or greedy ensemble. A 4M-parameter portfolio matches the performance of 205M–500M parameter monolithic models on Chronos Benchmark II, while requiring far less inference computation than test-time fine-tuning.
tags:
  - ICLR 2026
  - Time Series
  - Model portfolio
  - mixture of experts
  - test-time selection
  - time series foundation models
  - Chronos-Bolt
date: 2026-05-08
content_hash: 725fc737d33d8c32
---

# Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting

**Conference**: ICLR 2026
**arXiv**: [2510.06419](https://arxiv.org/abs/2510.06419)
**Code**: None
**Area**: Time Series / Foundation Models
**Keywords**: Model portfolio, mixture of experts, test-time selection, time series foundation models, Chronos-Bolt

## TL;DR

This paper proposes Chroma — a portfolio framework of small pretrained time series models: frequency/domain expert models are derived from a general model via post-training (achieving 10× training speedup), and at test time predictions are combined through model selection or greedy ensemble. A 4M-parameter portfolio matches the performance of 205M–500M parameter monolithic models on Chronos Benchmark II, while requiring far less inference computation than test-time fine-tuning.

## Background & Motivation

**State of the Field**: Time series foundation models (Chronos, TimesFM, Moirai) follow the "bigger is better" scaling paradigm — increasing model parameters (10M–500M) and training data to improve zero-shot forecasting. However, large models incur high training and inference costs, limiting practical deployment.

**Limitations of Prior Work**:

1. Monolithic large models treat all domains/frequencies uniformly → yet time series from different domains (energy/retail/weather) and frequencies (minute/hourly/monthly) have vastly different characteristics.
2. Diminishing marginal returns of "bigger is better" → limited gains from Mini (9M) to Base (205M).
3. The only available mechanism for leveraging additional test-time computation is fine-tuning → costly and slow due to gradient updates.
4. Model combination strategies (ensemble/selection) well-validated in NLP/CV have not been applied to time series foundation models.

**Root Cause**: In the generalization error of foundation models, bias dominates over variance → traditional ensembling's variance reduction is of limited benefit → bias on specific sub-domains must be reduced through specialization, followed by intelligent combination to achieve overall bias reduction.

**Paper Goals**: Rather than training one large general-purpose model, train multiple small expert models → intelligently select or combine them at test time based on validation set performance.

## Method

### Overall Architecture

Chroma proceeds in two stages:

1. **Training stage**: Construct a diverse portfolio of small forecasting models.
2. **Test stage**: Combine predictions via model selection or ensemble.

The framework is implemented on top of Chronos-Bolt (T5 encoder-decoder architecture), with model sizes ranging from 1M to 9M parameters.

### Key Design 1: Specialized Portfolio Construction

Training data is partitioned along metadata dimensions:

| Partition Dimension | Partition Examples | Characteristic |
|---|---|---|
| **Frequency** | hourly / daily / weekly / monthly / subhour | Captures temporal scale differences |
| **Domain** | energy / retail / transport / weather / web / various | Captures application domain differences |

Each expert is trained on one partition → diversity stems from data differences (not random seed differences). Each portfolio also includes a general model trained on the full dataset.

Key finding: **Frequency**-based partitioning consistently outperforms **domain**-based partitioning (~4% lower WQL, ~5% lower MASE), consistent with prior work such as TTM.

### Key Design 2: Efficient Portfolio Construction via Post-Training

Training all experts from scratch requires 200K steps × $N$ experts. A two-stage strategy is adopted instead:

1. First train a general model (200K steps).
2. Fine-tune on different data partitions (only 1K steps each) → producing expert models.

$$\text{Speedup} = \frac{200\text{K} \times N}{200\text{K} + 1\text{K} \times N} \approx 10\times$$

Post-trained experts achieve accuracy comparable to from-scratch experts, while exhibiting better scaling behavior.

### Key Design 3: Test-Time Combination Strategies

| Strategy | Method | Computation Cost |
|---|---|---|
| Model selection | Select the best single model on the validation set | $N+1$ forward passes |
| Greedy ensemble | Ensemble selection algorithm of Caruana et al. (2004) | $N + 2.5$ forward passes |

$$\hat{y}_{\text{ens}} = \sum_{m=1}^M w_m \cdot \hat{y}_m$$

Weights are optimized on a validation window via the greedy ensemble selection algorithm.

**Important finding**: Simple averaging or performance-weighted averaging significantly degrades performance on pretrained portfolios (WQL increases by 0.14–0.18) → intelligent selection is essential, as pretrained model errors are bias-dominated.

## Experiments & Results

### Main Results: Performance on Chronos Benchmark II

| Model | Parameters | BM2 Relative WQL ↓ |
|---|:---:|:---:|
| Seasonal Naive | - | 1.000 |
| Auto ETS | - | 0.892 |
| Chronos-Bolt Mini (9M) | 9M | 0.835 |
| Moirai-1.1 Large | 311M | ~0.82 |
| Chronos-Bolt Base | 205M | ~0.80 |
| TimesFM-2.0 | 500M | ~0.79 |
| **Chroma 4M (freq, best)** | **4M** | **~0.81** |
| **Chroma tiny (freq, ens.)** | **9M** | **~0.80** |

Chroma with only 4M active parameters matches the performance of monolithic models with 200M+ parameters.

### Ablation Study: Portfolio Design Choices

| Method | WQL (relative to 1M general model) |
|---|:---:|
| Single general model (1M) | 1.000 |
| General model ensemble × 5 | 0.987 |
| Domain experts — selection | 0.963 |
| Domain experts — ensemble | 0.957 |
| **Frequency experts — selection** | **0.918** |
| **Frequency experts — ensemble** | **0.926** |

Key insights:

1. **General model ensembling is nearly ineffective** (0.987 vs. 1.000) → variance reduction is unhelpful when bias dominates.
2. **Frequency experts > domain experts** (~4% advantage for frequency).
3. **Model selection and ensembling perform comparably** → selection is computationally more efficient.

### Scaling Behavior Analysis

| Model Scale | Single General Model WQL | Chroma (freq best) WQL |
|---|:---:|:---:|
| 1M | 1.000 | 0.918 |
| 2M | 0.977 | 0.916 |
| 4M | 0.960 | 0.880 |
| 9M (tiny) | 0.958 | 0.909 |

The portfolio follows scaling laws similar to those of monolithic models (log-log fit), suggesting the approach generalizes to larger model scales.

### Test-Time Computational Efficiency

| Method | Test-time GFLOPs (relative) | Relative WQL Improvement |
|---|:---:|:---:|
| Zero-shot general model | 1× | Baseline |
| Chroma (selection) | ~7× | −8.2% |
| Chroma (ensemble) | ~9× | −7.4% |
| Fine-tuning 1K steps | ~80× | −6.5% |

Chroma's test-time computation is far lower than fine-tuning (~1/10), while achieving comparable or superior performance gains → favorable position on the accuracy–efficiency Pareto frontier.

### Bias-Variance Analysis

Bias and variance are estimated on synthetic data using 10 independently trained general models:

| Model Scale | Bias | Variance | Bias/Variance Ratio |
|---|:---:|:---:|:---:|
| 1M | 65.3 | 10.1 | 6.5× |
| 2M | 49.0 | 12.3 | 4.0× |
| 4M | 22.0 | 6.4 | 3.4× |
| 9M | 20.1 | 8.4 | 2.4× |

Bias dominates variance at all scales → conventional ensemble-based variance reduction is ineffective for pretrained models → Chroma's gains stem from reducing sub-domain bias through specialization combined with intelligent selection.

## Paper Evaluation

### Strengths

1. **Sharp insight**: The bias-variance analysis clearly explains why ensembling general models fails while expert portfolios succeed.
2. **Highly practical**: Post-training reduces portfolio construction cost to ~1/10 that of training a general model.
3. **Comprehensive evaluation**: Two large benchmarks (BM2 + GIFT-Eval), scaling analysis, computational efficiency analysis, and ablation studies.
4. **Good interpretability**: Expert activation heatmaps show which experts are selected → consistent with task metadata.

### Weaknesses

1. Validation is limited to the Chronos-Bolt (encoder-decoder T5) architecture → applicability to other architectures (decoder-only, Mamba) remains unknown.
2. Partitioning strategies are manually designed (frequency/domain) → automatic partition learning is not explored.
3. The largest model is only 9M parameters → scaling behavior at the 100M+ scale is not verified.
4. Multiple forward passes are still required at test time → potentially limiting in real-time, low-latency scenarios.

### Rating

⭐⭐⭐⭐

Chroma offers an elegant and practical alternative to the "bigger is better" paradigm in time series foundation models. The core contribution lies not in methodological complexity, but in the insight derived from bias-variance analysis — pretrained model errors are bias-dominated, requiring expert specialization rather than random ensembling for improvement. The result that a 4M-parameter portfolio matches 200M+ models validates the "many small experts > one large generalist" strategy in the time series domain. The 10× training speedup from post-training and 10× test-time computation savings make Chroma highly attractive for practical deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-ser.md)
- [\[NeurIPS 2025\] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting](../../NeurIPS2025/time_series/learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)
- [\[ICLR 2026\] ResCP: Reservoir Conformal Prediction for Time Series Forecasting](rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)

<!-- RELATED:END -->
