---
title: >-
  [Paper Note] How Foundational are Foundation Models for Time Series Forecasting?
description: >-
  [NeurIPS 2025][Time Series][Time series foundation models] Through systematic experiments on synthetic and real-world electricity consumption data…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Time series foundation models"
  - "zero-shot forecasting"
  - "fine-tuning"
  - "domain transfer"
  - "lightweight models"
date: 2026-05-08
content_hash: 59f91fa659182a9a
---

# How Foundational are Foundation Models for Time Series Forecasting?

**Conference**: NeurIPS 2025
**arXiv**: [2510.00742](https://arxiv.org/abs/2510.00742)
**Authors**: Nouha Karaouli, Denis Coquenet, Elisa Fromont, Martial Mermillod, Marina Reyboz
**Institutions**: Univ. Rennes / CNRS / Inria / IRISA; Univ. Grenoble Alpes / CEA
**Code**: Datasets publicly available (Zenodo)
**Area**: Time Series
**Keywords**: Time series foundation models, zero-shot forecasting, fine-tuning, domain transfer, lightweight models

## TL;DR

Through systematic experiments on synthetic and real-world electricity consumption data, this paper reveals that the zero-shot generalization capability of time series foundation models (TSFMs) is highly dependent on the pretraining data distribution. Under domain shift, SAMFormer—a lightweight specialized model with only 49.5K parameters trained from scratch—outperforms fine-tuned TimesFM with 500M+ parameters.

## Background & Motivation

### Success of Foundation Models in NLP/CV

Foundation models have achieved revolutionary success in NLP (e.g., BERT) and CV (e.g., ViT). Pretrained on large-scale diverse data, these models demonstrate strong zero-shot and few-shot transfer capabilities, consistently outperforming task-specific models trained from scratch across a wide range of downstream tasks.

### Challenges for Time Series Foundation Models

Inspired by the success in NLP/CV, researchers have proposed TSFMs such as TimesFM, TimeGPT, and TiReX, hoping to capture universal temporal pattern representations through large-scale pretraining. However, time series data presents unique challenges:

- **Domain-specific structure**: Seasonality, trends, and irregular sampling vary greatly across applications
- **Distribution shift**: Even data within the same broad category exhibits significant statistical differences
- **Individual behavioral variation**: Individual-level data in real-world scenarios (e.g., personal electricity usage patterns) differs substantially in distribution from the population-level aggregated data used during pretraining

### Core Research Questions

1. Can TSFMs generalize to scenarios beyond their pretraining distribution?
2. Are TSFMs competitive with lightweight specialized alternatives in practical applications?

## Method

### Evaluation Framework Design

The paper designs a systematic evaluation pipeline progressing from synthetic to real-world data, with progressively increasing task difficulty and degree of distribution shift.

### Model Selection

| Model | Type | Parameters | Description |
|-------|------|------------|-------------|
| TimesFM | Decoder-only Transformer TSFM | 500M+ | Released by Google; pretrained on large-scale synthetic and real data |
| TimeGPT | Transformer FM (hosted API) | Undisclosed | Nixtla; pretrained on 100B+ time series observations |
| TiReX | xLSTM-based TSFM | Undisclosed | Zero-shot short- and long-horizon forecasting |
| SAMFormer | Lightweight attention model | 49.5K | Channel-wise attention; trained from scratch |

### Synthetic Data Benchmarks

Four synthetic datasets are constructed to evaluate zero-shot capability, each containing 2,688 time steps (8 weeks, 30-minute sampling interval):

- **D1, D2**: Harmonically aligned sinusoids—fully observable periodic signals; tests the ability to recognize and extrapolate clean periodic patterns
- **D3, D4**: Randomly sampled non-harmonic sinusoids—complex and partially observable periodicity; tests generalization to incomplete patterns

### Real-World Evaluation

The **Elec_Consumption** dataset is used—a private small dataset of daily household electricity consumption for a single household spanning two years (2023–2024). This dataset reflects individual consumption behavior, creating a clear distribution shift relative to the general population-level datasets used during TSFM pretraining.

### Fine-Tuning Setup

- Optimizer: Adam, learning rate $10^{-4}$, weight decay 0.01, batch size 64
- Sliding window: context=128, horizon=128
- Training: up to 100 epochs, early stopping (patience=10)
- SAMFormer: results averaged over 5 different random seeds with standard deviation reported
- Hardware: NVIDIA Tesla V100 GPU

## Key Experimental Results

### Table 1: Zero-Shot MAE on Synthetic Data (D1–D4)

| Dataset | Horizon | TimeGPT | TiReX | TimesFM |
|---------|---------|---------|-------|---------|
| D1 | 128 | 0.89 | **0.11** | 0.13 |
| D1 | 256 | 1.08 | **0.21** | 0.22 |
| D1 | 512 | 1.09 | 0.37 | **0.34** |
| D2 | 128 | 0.80 | 0.29 | **0.15** |
| D2 | 256 | 1.25 | 0.72 | **0.35** |
| D2 | 512 | 1.57 | 1.11 | **0.72** |
| D3 | 128 | 1.86 | **1.10** | 1.13 |
| D3 | 512 | **2.29** | 3.30 | 3.50 |
| D4 | 128 | 1.30 | **0.78** | 0.89 |
| D4 | 512 | **2.31** | 2.80 | 2.98 |

**Key Findings**: TiReX and TimesFM perform well on simple periodic patterns (D1/D2); on complex non-harmonic patterns (D3/D4), all models degrade sharply at long horizons, with TimeGPT showing more stable performance due to its conservative prediction strategy.

### Table 2: Zero-Shot and Fine-Tuned MAE on Real-World Data (Elec_Consumption)

| Model | 15-7 | 30-7 | 60-30 | 128-128 | 365-365 |
|-------|------|------|-------|---------|---------|
| TimeGPT | 6.60 | 6.52 | 5.60 | 6.91 | 6.44 |
| TiReX | 6.94 | 5.71 | 4.61 | **3.78** | 5.90 |
| TimesFM | **5.07** | 5.83 | **4.08** | 4.63 | **5.30** |

**Fine-Tuning Comparison (context=128, horizon=128)**:

| Model | Parameters | MAE |
|-------|------------|-----|
| TimesFM (fine-tuned) | 500M+ | 4.49 ± 0.00 |
| SAMFormer (from scratch) | 49.5K | **4.28 ± 0.05** |

**Key Findings**: SAMFormer trained from scratch with only 49.5K parameters outperforms fine-tuned TimesFM with 500M+ parameters, representing a parameter efficiency improvement of over 10,000×.

## Highlights & Insights

1. **Elegant evaluation design**: The progressive evaluation from synthetic to real-world data systematically reveals the performance degradation patterns of TSFMs under increasing degrees of distribution shift.
2. **Practically actionable findings**: The paper explicitly demonstrates that the "one-size-fits-all" promise does not hold for time series, providing a concrete decision criterion for deployment—use TSFMs when pretraining–target similarity is high; otherwise, use lightweight specialized models.
3. **Striking parameter efficiency gap**: 49.5K vs. 500M+ parameters (a 10,000× difference), with the lightweight model still winning, providing strong evidence against the "bigger is better" pretraining paradigm.
4. **Focus on realistic deployment scenarios**: The use of private individual-level electricity consumption data, rather than public benchmarks, better reflects the domain shift challenges encountered in real-world applications.
5. **Public dataset release**: All datasets used in the study are made publicly available via Zenodo, supporting reproducibility.

## Limitations & Future Work

1. **Limited evaluation scope**: Only univariate time series forecasting is evaluated; other tasks such as multivariate forecasting, anomaly detection, and classification are not considered.
2. **Real-world dataset too narrow**: Only a single household's two-year daily electricity consumption is used (~730 data points), which is insufficient to represent large-scale real-world scenarios.
3. **Incomplete TSFM coverage**: Only three models (TimesFM, TimeGPT, TiReX) are tested; recent important TSFMs such as Chronos, Lag-Llama, and MOIRAI are not included.
4. **Limited fine-tuning strategies**: TimesFM is fine-tuned only with a default Adam configuration; parameter-efficient fine-tuning strategies such as learning rate scheduling and LoRA are not explored, potentially underestimating TSFM capacity.
5. **Lack of statistical significance testing**: TimesFM fine-tuning results show zero variance (deterministic), while SAMFormer reports standard deviation but no formal statistical test is conducted.
6. **Overly idealized synthetic data**: Sinusoidal compositions fail to capture complex characteristics of real time series such as abrupt changes, missing values, and noise.
7. **Computational cost not analyzed**: Despite noting parameter efficiency, the paper does not quantify training time, inference latency, or memory footprint—key metrics for practical deployment.

## Related Work & Insights

- **Time series foundation models**: TimesFM (Das et al., 2024), TimeGPT (Garza et al., 2024), TiReX (Auer et al., 2025), and FEDformer (Zhou et al., 2022) represent the large-scale pretraining paradigm.
- **TSFM benchmark evaluation**: GIFT-Eval (Aksu et al., 2024) for cross-domain generalization; OpenTS (2024) for reproducible test suites; Nixtla Arena (2024) for comprehensive evaluation.
- **Questioning TSFM effectiveness**: Xu et al. (2025, ICLR) show that simple autoregressive baselines can compete with TSFMs; Zhao et al. (2025) find that extended fine-tuning may degrade TSFM performance.
- **Lightweight specialized models**: SAMFormer (Ilbert et al., 2024), a compact Transformer leveraging sharpness-aware minimization and channel-wise attention.
- **General foundation model discourse**: Bommasani et al. (2021) discuss the opportunities and risks of foundation models; Yuan et al. (2025) challenge the "one-size-fits-all" evaluation paradigm.

## Rating

| Dimension | Score | Comment |
|-----------|-------|---------|
| Novelty | ⭐⭐⭐ | The research question is important but not entirely novel; the work is primarily empirical validation |
| Technical Depth | ⭐⭐ | No new methods or models are proposed; contributions are largely experimental comparison and observation |
| Experimental Quality | ⭐⭐⭐ | The synthetic-to-real progressive design is noteworthy, but scale is limited and model coverage is incomplete |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure, coherent argumentation, well-formatted figures and tables |
| Practical Value | ⭐⭐⭐⭐ | Directly informative for TSFM deployment decisions in practice |
| Overall | 3.2/5 | Timely and important research question with insightful conclusions, but limited technical contribution and insufficient experimental scale to support strong claims |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning](less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
