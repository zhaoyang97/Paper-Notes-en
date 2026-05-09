---
title: >-
  [Paper Note] IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?
description: >-
  [AAAI 2026][Time Series][time series forecasting] IdealTSF is a three-stage progressive framework that (1) uses negative sample pre-training on synthetic non-ideal data to enhance robustness, (2) trains on repaired positive samples to learn underlying trends, and (3) applies the ECOS optimizer to guide parameters toward flat minima — achieving approximately 10% MSE improvement on time series data containing noise and missing values.
tags:
  - AAAI 2026
  - Time Series
  - time series forecasting
  - negative sample pre-training
  - adversarial training
  - data robustness
  - non-ideal data
date: 2026-05-08
content_hash: 985cc2ee28741704
---

# IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?

**Conference**: AAAI 2026
**arXiv**: [2512.05442](https://arxiv.org/abs/2512.05442)
**Code**: [GitHub](https://github.com/LuckyLJH/IdealTSF)
**Area**: Time Series Forecasting
**Keywords**: time series forecasting, negative sample pre-training, adversarial training, data robustness, non-ideal data

## TL;DR
IdealTSF is a three-stage progressive framework that (1) uses negative sample pre-training on synthetic non-ideal data to enhance robustness, (2) trains on repaired positive samples to learn underlying trends, and (3) applies the ECOS optimizer to guide parameters toward flat minima — achieving approximately 10% MSE improvement on time series data containing noise and missing values.

## Background & Motivation

### State of the Field

**Background**: Deep models for time series forecasting (Transformer-based, MLP-based, etc.) typically assume complete, anomaly-free inputs and achieve strong performance on standard benchmarks such as ETT, Weather, and ECL. However, real-world time series data frequently contain missing values, outliers, and noise.

### Limitations of Prior Work

**Limitations of Prior Work**: (1) Simple interpolation methods (linear/spline) fail to recover complex nonlinear structures and perform poorly under large-scale missingness; (2) anomaly detection methods may incorrectly discard meaningful extreme events (e.g., sudden traffic surges); (3) existing forecasting models lack intrinsic robustness to input perturbations, suffering severe performance degradation under distribution shift at test time.

### Root Cause

**Key Challenge**: Non-ideal data (noise, missingness, anomalies) is conventionally treated as an obstacle to be "pre-processed and repaired," yet it carries valuable information — patterns of extreme events, system failure signatures, etc. Simply repairing or discarding such data results in information loss, while using it directly contaminates training.

### Solution

**Goal**: Transform non-ideal data from an obstacle into a training asset. **Key Insight**: Treat non-ideal data as "negative samples" for pre-training to build model immunity, generate "positive samples" for formal training, and further strengthen robustness via adversarial optimization. **Core Idea**: negative sample pre-training (synthetic non-ideal data for robustness) + positive sample training (learning trends from smoothed, repaired data) + ECOS adversarial optimizer (guiding parameters to flat minima for better generalization).

## Method

### Overall Architecture
Three-stage progressive training: (1) **Pre-training**: synthesize negative samples from raw data (with jumps/noise/missingness) to pre-train the attention module toward robust representations of non-ideal inputs; (2) **Training**: repair data into positive samples via hybrid interpolation and Z-score/IQR anomaly detection, then use the pre-trained attention module to extract features and produce forecasts; (3) **Optimization**: apply the ECOS optimizer, which injects adversarial perturbations to guide parameters toward flat minima.

### Key Designs

1. **Negative Sample Synthesis and Pre-training**:

    - Function: Train the model to handle diverse non-ideal data patterns.
    - Mechanism: Three synthesis strategies cover three non-ideal scenarios: (a) Stable-distribution jumps — heavy-tailed jump increments $\Delta x_i = R \cdot \cos(\theta)$ generated from an $\alpha$-stable distribution are superimposed on the time series, with $\alpha$ controlling tail thickness to simulate varying degrees of abrupt change; (b) Multi-scale noise — hierarchical noise injection combining low-frequency high-intensity and high-frequency low-intensity components to simulate multi-source perturbations such as sensor drift; (c) Structural deletion — randomly selected contiguous segments are set to missing to simulate data acquisition interruptions.
    - Design Motivation: The three strategies respectively simulate abrupt jumps/regime changes (e.g., financial crashes), observation noise (sensor drift), and data missingness (communication failure) — three archetypal anomaly types. Pre-training on synthetic data enhances robustness without requiring real-world annotations.

2. **Positive Sample Repair and Training**:

    - Function: Recover reliable training signals from non-ideal raw data.
    - Mechanism: Anomalous points are first identified via Z-score/IQR detection and marked as missing; hybrid smooth interpolation (weighted combination of linear interpolation and moving average) then repairs the missing values to produce "positive samples." The pre-trained attention module (with partial parameter freezing) extracts features, and the prediction head is trained with MSE loss.
    - Design Motivation: Since the pre-training stage has already equipped the attention module with noise-robust representations, training on positive samples ensures the model learns clean trend and periodicity signals.

3. **ECOS (Ecosystem) Optimizer**:

    - Function: Guide the model toward flat minima to improve generalization.
    - Mechanism: A three-phase optimization loop — (I) ascend along the gradient direction (add perturbation $e_\theta = \frac{\rho}{\|\nabla L\|} \cdot \nabla L$) to explore the worst-case direction on the loss surface; (II) perform multi-step fine-tuning with a small learning rate to optimize within the perturbed neighborhood; (III) restore original parameters and apply a standard optimization step. FGSM/PGD adversarial training is additionally incorporated.
    - Design Motivation: The approach resembles SAM (Sharpness-Aware Minimization) but achieves greater stability — multi-step fine-tuning avoids the single-step rebound oscillations inherent to SAM, while adversarial training further improves robustness in input space.

## Key Experimental Results

### Main Results

Evaluations are conducted on ETTh1/h2/m1/m2, Weather, ECL, and Traffic datasets.

| Dataset | Baseline Attention MSE | IdealTSF MSE | Gain |
|--------|:---:|:---:|:---:|
| ETTh1 (720 steps) | 0.456 | **0.411** | 9.9% |
| ETTm1 (720 steps) | 0.400 | **0.361** | 9.8% |
| Weather (720 steps) | 0.259 | **0.234** | 9.7% |
| ECL (720 steps) | 0.214 | **0.193** | 9.8% |

Average cross-dataset MSE improvement is approximately 10%.

### Ablation Study

| Configuration | ETTh1 MSE | Note |
|------|:---:|------|
| Full IdealTSF | **0.411** | Complete model |
| w/o negative sample pre-training | 0.442 | −7.5%; model is vulnerable to noise |
| w/o positive sample repair | 0.435 | −5.8%; training signal remains noisy |
| w/o ECOS | 0.428 | −4.1%; converges to sharp minima |
| w/o adversarial training | 0.421 | −2.4%; reduced generalization |

### Key Findings
- Negative sample pre-training contributes the most — even without positive sample repair, the pre-trained model outperforms training from scratch by 5%+.
- The ECOS "ascent–descent" strategy is more stable than vanilla SAM (multi-step fine-tuning avoids oscillation).
- Under adversarial scenarios with 30% artificially injected missingness/noise, the performance advantage roughly doubles (20%+ improvement).
- The three negative sample synthesis strategies are complementary — each contributes approximately 3% individually, while their combination yields 7.5%.

## Highlights & Insights
- The **"non-ideal data as negative samples"** perspective is novel — rather than repairing data, the framework trains model immunity using defective data.
- The **three-stage progressive design** follows a clear logic: pre-training → training → optimization, with each stage addressing a different level of the problem.
- The **ECOS optimizer** integrates the flat-minima principle from SAM with adversarial training, constituting a general optimization strategy transferable to other robust learning scenarios.

## Limitations & Future Work
- Hyperparameters for negative sample synthesis (stable distribution parameters $\alpha, \beta, \gamma$; noise scale; deletion length) require careful tuning.
- Validation is limited to attention-based architectures; other backbones such as MLP and full Transformer variants are not tested.
- Positive sample generation relies on linear interpolation and moving averages, which may be insufficient for complex nonlinear patterns.
- No direct comparison is made with dedicated robust time series forecasting methods (e.g., Robust-TSF).

## Related Work & Insights
- **vs. RevIN/DLinear**: These methods handle non-stationarity via normalization/decomposition but do not address missingness or anomalies; IdealTSF directly targets robustness.
- **vs. SAM optimizer**: ECOS extends the SAM paradigm by incorporating multi-step fine-tuning and adversarial training for greater stability.
- **vs. data augmentation methods**: Conventional augmentation (cropping/scaling) targets clean data; IdealTSF's negative sample synthesis is specifically designed for non-ideal scenarios.
- The negative sample pre-training paradigm is transferable to other noise-sensitive tasks, such as financial forecasting and medical time series analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of negative sample pre-training and the ECOS optimizer is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation, adversarial scenario testing, and comprehensive ablation study.
- Writing Quality: ⭐⭐⭐ Notation is occasionally inconsistent despite extensive mathematical formulation.
- Value: ⭐⭐⭐⭐ Provides a practical solution for time series forecasting under low-quality data conditions.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[AAAI 2026\] Finding Time Series Anomalies using Granular-ball Vector Data Description](finding_time_series_anomalies_using_granular-ball_vector_data_description.md)
- [\[AAAI 2026\] Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios](scaling_llm_speculative_decoding_non-autoregressive_forecasting_in_large-batch_s.md)
- [\[AAAI 2026\] Harmonic Dataset Distillation for Time Series Forecasting](harmonic_dataset_distillation_for_time_series_forecasting.md)

<!-- RELATED:END -->
