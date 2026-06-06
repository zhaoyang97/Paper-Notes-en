---
title: >-
  [Paper Note] Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity
description: >-
  [NeurIPS 2025][Time Series][neural activity prediction] The first systematic evaluation of 12 probabilistic time series forecasting models on mouse cortical calcium imaging data. PatchTST consistently achieves top perfor…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "neural activity prediction"
  - "probabilistic forecasting"
  - "calcium imaging"
  - "benchmarking"
  - "foundation models"
date: 2026-05-08
content_hash: ce92c59f9ee61d38
---

# Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity

**Conference**: NeurIPS 2025
**arXiv**: [2510.18037](https://arxiv.org/abs/2510.18037)  
**Code**: To be confirmed  
**Area**: Neuroscience / Time Series Forecasting
**Keywords**: neural activity prediction, probabilistic forecasting, calcium imaging, benchmarking, foundation models

## TL;DR
The first systematic evaluation of 12 probabilistic time series forecasting models on mouse cortical calcium imaging data. PatchTST consistently achieves top performance (informative prediction horizon up to 1.5 s), zero-shot foundation models (Chronos) fail entirely but become competitive after fine-tuning, and the intrinsic predictability ceiling of neural activity is found to be approximately 1.5 seconds.

## Background & Motivation

**Background**: Neural activity prediction is critical for closed-loop brain–computer interfaces (BCIs) and for understanding neural systems. Deep learning-based time series forecasting has achieved remarkable progress in finance and meteorology, yet has never been systematically evaluated on neural data.

**Limitations of Prior Work**: (a) Existing time series benchmarks (ETT, Weather, etc.) differ substantially from neural data in terms of millisecond-level sampling rates, absence of seasonality, and high noise levels; (b) neuroscience-specific methods lack probabilistic forecasting capabilities, despite noise being an intrinsic property of neural data; (c) the zero-shot capability of time series foundation models (Chronos, Moirai) on neural data remains unknown.

**Key Challenge**: Does neural activity possess sufficient temporal structure for deep learning models to exploit? If so, how far into the future can it be predicted?

**Goal**: Establish a standard benchmark for probabilistic forecasting of neural activity and evaluate the capabilities and limitations of classical, deep learning, and foundation models.

**Key Insight**: Twelve models plus two baselines are evaluated across five mouse cortical calcium imaging sessions (35 Hz, ~51K time steps), using mean weighted quantile loss (MWQL) for probabilistic assessment.

**Core Idea**: A comprehensive evaluation of 12 probabilistic forecasting models on neural calcium imaging data across multiple prediction horizons reveals that PatchTST achieves the best performance, that the predictability ceiling is approximately 1.5 seconds, and that zero-shot foundation models fail entirely.

## Method

### Overall Architecture
Five calcium imaging sessions (4 brain regions/session, 35 Hz) → 60%/20%/20% split → 12 models + AR/Naive baselines → prediction horizons of 0.5/1/2 s → MWQL-based probabilistic evaluation + step-wise error analysis.

Models span three categories: classical methods (AR, ARIMA, AR-HMM, Theta), deep learning methods (DeepAR, DLinear, TFT, PatchTST, TiDE, WaveNet), and foundation models (Chronos, Moirai).

### Key Designs

1. **Model Selection (4 categories, 12 models)**:

    - Classical: AR, ARIMA, AR-HMM, Theta
    - Deep learning: DeepAR, DLinear, TFT, PatchTST, TiDE, WaveNet
    - Foundation models: Chronos (T5 backbone, discretized time series), Moirai (mixture distribution)
    - Evaluation focus: direct multi-step forecasting vs. autoregressive single-step forecasting

2. **Probabilistic Evaluation (MWQL)**:

    - Function: assess the quality of predictive uncertainty via quantile loss
    - Mechanism: $\text{MWQL} = \frac{1}{H} \sum_{h=1}^{H} \sum_{q} w_q \cdot QL_q(y_{t+h}, \hat{y}_{t+h}^q)$ — jointly evaluates forecast accuracy and uncertainty interval coverage
    - Design Motivation: point predictions are insufficient — noisy neural data require reliable credible intervals for predictions

3. **Predictability Ceiling Analysis**:

    - Function: determine when forecasts degenerate to predicting the mean and standard deviation of the training set
    - Mechanism: step-wise error is computed; a forecast is deemed uninformative when model error reaches 95% of the error of the Average baseline (which predicts the training mean)
    - Design Motivation: reveal the intrinsic temporal structure window of neural activity — beyond 1.5 s, no model retains an informative advantage

### Loss & Training
- Non-overlapping sliding windows; all models use default hyperparameters with lightweight tuning.
- Chronos fine-tuning: the T5 backbone is fine-tuned on the target dataset.

## Key Experimental Results

### Main Results

| Model | 0.5 s MWQL | 1 s MWQL | 2 s MWQL | Informative Horizon |
|-------|-----------|---------|---------|-------------------|
| Naive | baseline | baseline | baseline | — |
| AR | good | good | degraded | 1.28 s |
| **PatchTST** | **best** | **best** | degraded | **1.80 s** |
| Chronos (zero-shot) | poor | poor | poor | — |
| Chronos (fine-tuned) | competitive | competitive | degraded | ~1.5 s |

### Ablation Study

| Analysis Dimension | Finding |
|-------------------|---------|
| Prediction horizon > 1.5 s | All models degenerate to predicting the training mean/std |
| Direct vs. autoregressive forecasting | PatchTST's direct multi-step forecasting outperforms autoregressive approaches at longer horizons |
| Zero-shot vs. fine-tuned | Chronos fails entirely in zero-shot setting (large domain gap); improves substantially after fine-tuning |
| Uncertainty intervals | Interval width increases with horizon and converges to the training data standard deviation after ~35 steps |
| Brain region differences | Predictability varies slightly across regions but patterns are consistent |

### Key Findings
- PatchTST consistently achieves the best performance — with an informative prediction horizon of 1.80 s (vs. 1.28 s for AR), indicating that Transformers capture longer temporal dependencies than linear models.
- 1.5 seconds represents an approximate predictability ceiling for neural activity — likely reflecting the intrinsic timescale of cortical dynamics rather than a model limitation.
- Zero-shot foundation models (Chronos, Moirai) perform poorly on neural data — the domain gap (finance/meteorology vs. neuroscience) is too large for pretrained knowledge to transfer.
- Fine-tuned Chronos becomes competitive with PatchTST — demonstrating that the T5 backbone has sufficient expressive capacity and requires only domain adaptation.
- Probabilistic forecast quality rankings are consistent with point forecast rankings — model uncertainty estimates are reliable.

## Highlights & Insights
- The **1.5-second predictability ceiling** has direct implications for BCIs — control algorithms should not rely on predictions beyond this horizon.
- The **failure of zero-shot foundation models** serves as a reminder to the community that general-purpose time series foundation models are not universally applicable; fine-tuning is necessary when the domain gap is large.
- The **advantage of PatchTST** demonstrates that direct multi-step forecasting outperforms autoregressive approaches on high-noise, short-temporal-structure data by avoiding error accumulation.

## Limitations & Future Work
- It remains unclear whether the predictability ceiling stems from model limitations or the intrinsic constraints of neural dynamics — electrophysiology or optogenetic control experiments are needed to disentangle these factors.
- Only four brain regions per session are studied, with no exploration of cross-subject generalization — individual variability may affect predictability.
- No comparison is made with neuroscience-specific methods (e.g., LFADS, NDT) — evaluating only general-purpose forecasting models may be an unfair assessment.
- Evaluation is limited to calcium imaging data; electrophysiology modalities (spike trains, LFP) and other recording types are not tested.
- Probabilistic forecasting is assessed solely via MWQL — quantile calibration is not evaluated.
- The 35 Hz sampling rate is relatively low — predictability may differ at higher temporal resolutions.

## Related Work & Insights
- **vs. standard time series benchmarks**: ETT/Weather datasets exhibit seasonality and long-term trends, whereas neural data constitute a fast stochastic process — a qualitatively different type of difficulty.
- **vs. neural-specific methods (e.g., LFADS)**: neuroscience methods perform latent dynamical inference rather than probabilistic forecasting; this work fills that gap.
- **Insight**: The intrinsic temporal structure of neural activity spans approximately 1.5 seconds — a finding with direct implications for latency design in BCI control algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic probabilistic forecasting benchmark for neural activity
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 models × multiple horizons × probabilistic evaluation
- Writing Quality: ⭐⭐⭐⭐ In-depth analysis; the 1.5-second ceiling is a valuable finding
- Value: ⭐⭐⭐⭐ Provides an important forecasting benchmark and predictability boundary for the neuroscience and BCI communities

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Learning Time-Scale Invariant Population-Level Neural Representations](learning_time-scale_invariant_population-level_neural_representations.md)

</div>

<!-- RELATED:END -->
