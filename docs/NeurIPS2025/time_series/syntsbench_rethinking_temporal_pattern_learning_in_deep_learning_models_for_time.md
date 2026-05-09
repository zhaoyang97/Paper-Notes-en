---
title: >-
  [Paper Note] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series
description: >-
  [NeurIPS 2025][Time Series][time series forecasting] This paper proposes SynTSBench, a synthetic data-driven evaluation paradigm that systematically assesses the actual modeling capabilities of time series forecasting models across dimensions such as trend, periodicity, dependency, and noise robustness, through programmable feature configurations and theoretically optimal benchmarks.
tags:
  - NeurIPS 2025
  - Time Series
  - time series forecasting
  - synthetic data
  - model evaluation
  - theoretical optimal benchmark
  - robustness analysis
date: 2026-05-08
content_hash: 179d6d147be3281a
---

# SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series

**Conference**: NeurIPS 2025
**arXiv**: [2510.20273](https://arxiv.org/abs/2510.20273)
**Code**: [GitHub](https://github.com/TanQitai/SynTSBench)
**Area**: Time Series Forecasting / Evaluation Benchmarks
**Keywords**: time series forecasting, synthetic data, model evaluation, theoretical optimal benchmark, robustness analysis

## TL;DR

This paper proposes SynTSBench, a synthetic data-driven evaluation paradigm that systematically assesses the actual modeling capabilities of time series forecasting models across dimensions such as trend, periodicity, dependency, and noise robustness, through programmable feature configurations and theoretically optimal benchmarks.

## Background & Motivation

### Limitations of Prior Work

The current evaluation practice in time series forecasting suffers from two core problems:

1. **Lack of feature isolation**: Real-world time series data contains intertwined components—trends, seasonality, dependencies of varying lengths—that cannot be disentangled for targeted evaluation. It remains unclear whether performance improvements stem from genuinely capturing specific patterns or from exploiting spurious correlations in training data.
2. **Lack of theoretical performance bounds**: Evaluation on observational data cannot establish a reference for what "optimal prediction" should look like, making it impossible to distinguish meaningful generalization gains from noise overfitting.

This reflects a fundamental methodological gap: existing approaches treat data complexity as the validation criterion, overlooking systematic characterization of time series properties and theoretical solution spaces.

## Method

### Overall Architecture

SynTSBench establishes three core analytical dimensions:

1. **Temporal feature decomposition and capability mapping**: Constructs synthetic data with known patterns and evaluates each model's ability to capture individual pattern types.
2. **Robustness analysis under data anomalies**: Injects progressive noise and various anomalies into clean signals to quantify noise tolerance thresholds and recovery capabilities.
3. **Theoretical optimal benchmarking**: Leverages the known generative process of synthetic data to derive theoretically optimal solutions for each pattern type, enabling direct comparison between model predictions and mathematical optima.

### Key Designs

1. **Programmable Temporal Feature Synthesis**:
   - *Function*: Designs synthetic datasets covering 11 trend functions and 10 periodic patterns, along with short- and long-range dependencies and multivariate correlations.
   - *Mechanism*: Generates short- and long-range dependencies via ARMA processes; uses random walks and white noise to test dependency-free baselines; constructs multivariate relationships including delayed coupling, linear additivity, conditional interaction, and nonlinear transformation.
   - *Design Motivation*: Isolates confounding factors so that model performance can be directly attributed to the capacity to capture specific temporal features.

2. **Progressive Noise Injection and Anomaly Testing**:
   - *Function*: Injects Gaussian noise at multiple SNR levels; tests diverse noise distributions including uniform, Laplace, $t$-distribution, and Lévy stable; introduces point anomalies, impulse anomalies, mean shifts, and trend changes.
   - *Mechanism*: Modifies clean synthetic signals under controlled conditions to quantify each model's noise tolerance threshold and anomaly recovery capability.
   - *Design Motivation*: Real-world data inevitably contains noise and anomalies; systematic robustness evaluation is necessary rather than comparison on clean data alone.

3. **Theoretical Optimal Benchmarking**:
   - *Function*: Computes theoretically optimal predictions for each pattern type by exploiting the known generative process of synthetic data.
   - *Mechanism*: For example, the optimal MSE for a linear trend is 0, and the optimal solution for an AR process can be derived analytically.
   - *Design Motivation*: Without a theoretical optimum, model optimization resembles "blind tuning"; with an optimal reference, performance gaps and improvement headroom become explicit.

### Loss & Training

All models follow a unified sliding-window protocol with an input length of 96 and prediction horizons of $\{10, 24, 48, 96, 192\}$, using a train/validation/test split of 7:1:2. Fifteen fine-tuning models (Autoformer, PatchTST, iTransformer, DLinear, TimesNet, etc.) and three zero-shot models (Chronos, TimeMoE, Moirai) are evaluated.

## Key Experimental Results

### Main Results

Trend signal forecasting (MSE/MAE, averaged over 4 prediction horizons):

| Trend Type | Best Model | Best MSE | Worst Model | Worst MSE | Theoretical Optimum |
|---|---|---|---|---|---|
| Exponential | DLinear | 3.40e-08 | Autoformer | 8.45 | 0 |
| Linear | PaiFilter | 1.34e-12 | Autoformer | 2.02e-03 | 0 |
| Gaussian | PaiFilter | 4.09e-06 | Autoformer | 2.00e-02 | 0 |
| Gompertz | TimeMixer | 2.85e-06 | Autoformer | 3.56e-03 | 0 |

### Ablation Study

- **Large variance in trend capability**: DLinear approaches the theoretical optimum on linear and exponential trends, whereas Autoformer's error on exponential trends exceeds DLinear's by 8 orders of magnitude.
- **Periodicity capture**: Transformer-based models do not consistently outperform MLP-based models on complex periodic patterns.
- **Noise robustness**: As SNR decreases, models degrade at markedly different rates.
- **Zero-shot models**: Chronos, TimeMoE, and Moirai fall far behind fine-tuning models on predictable patterns.

### Key Findings

- Current deep learning models **do not universally approach the theoretical optimum across all temporal feature types**.
- Different architectures (Transformer, MLP, CNN, RNN, KAN) each excel at specific pattern types; no single architecture is universally superior.
- The simple DLinear can outperform complex Transformers by 8 orders of magnitude on trend forecasting.
- Performance differences among models are far larger on synthetic data than on standard real-world datasets, indicating that existing benchmarks obscure important capability distinctions.

## Highlights & Insights

- **Methodological contribution**: Addresses a methodological gap in time series evaluation—shifting from "who performs better on real data" to "systematic analysis of individual capabilities."
- **Theoretical optimal benchmarks**: For the first time, computable theoretical optima are established for each temporal feature type.
- **Practical utility**: Guides practitioners in selecting appropriate model architectures based on task characteristics.
- Reveals the important insight that models with marginal differences on standard benchmarks may exhibit drastically different performance on specific capabilities.

## Limitations & Future Work

- The complexity of synthetic data remains limited and cannot fully replicate the intricate interactions found in real-world scenarios.
- Evaluation covers only univariate and a small number of multivariate settings; high-dimensional time series are not sufficiently tested.
- Theoretical optimal benchmarks apply only to synthetic data and cannot be directly extended to real-world data.
- In-depth analysis of large language model-based time series methods is absent.
- Design details for datasets simulating complex real-world scenarios (e.g., economic indicators) are insufficiently elaborated.

## Related Work & Insights

- **ProbTS**: Unifies evaluation across different forecasting paradigms but relies on real-world data.
- **TFB**: Provides a standardized pipeline with multi-domain coverage but similarly lacks feature isolation.
- This paper is complementary to these frameworks—SynTSBench diagnoses model capabilities, while standard benchmarks measure overall practical performance.
- The work offers important guidance for model selection practices within the time series community.

## Rating

⭐⭐⭐⭐ — Outstanding methodological innovation that uncovers blind spots in existing evaluation frameworks, supported by comprehensive and systematic experiments.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)
- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_dynamics.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[NeurIPS 2025\] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)

<!-- RELATED:END -->
