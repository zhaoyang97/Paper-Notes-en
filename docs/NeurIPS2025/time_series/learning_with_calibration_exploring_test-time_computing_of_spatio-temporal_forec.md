---
title: >-
  [Paper Note] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting
description: >-
  [NeurIPS 2025 (Spotlight)][Time Series][Spatio-temporal forecasting] This paper proposes ST-TTC, a lightweight test-time computing paradigm that corrects periodic biases in spatio-temporal forecasting during inference vi…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "Time Series"
  - "Spatio-temporal forecasting"
  - "test-time computing"
  - "frequency-domain calibration"
  - "non-stationarity"
  - "online adaptation"
date: 2026-05-08
content_hash: ba032644dce40b81
---

# Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting

**Conference**: NeurIPS 2025 (Spotlight)
**arXiv**: [2506.00635](https://arxiv.org/abs/2506.00635)  
**Code**: [https://github.com/Onedean/ST-TTC](https://github.com/Onedean/ST-TTC) (Open Source)  
**Area**: Time Series
**Keywords**: Spatio-temporal forecasting, test-time computing, frequency-domain calibration, non-stationarity, online adaptation

## TL;DR

This paper proposes ST-TTC, a lightweight test-time computing paradigm that corrects periodic biases in spatio-temporal forecasting during inference via a frequency-domain phase-amplitude calibrator and a flash gradient update mechanism, consistently improving the performance of diverse backbone models without modifying their architectures.

## Background & Motivation

**Background**: Spatio-temporal forecasting (traffic flow, weather, energy, etc.) has been addressed by numerous deep network architectures (Transformers, GNNs, MLPs, etc.), achieving strong results on standard benchmarks.

**Limitations of Prior Work**: In real-world deployments, sensor data is frequently corrupted by noise, outliers, and **non-stationary distribution shifts** (e.g., seasonal variations, sensor aging), causing performance degradation at test time.

**Key Challenge**: Existing solutions fall into three categories: (a) OOD learning—enhancing robustness during training, but assuming that training data covers all future distributional invariances, which rarely holds in practice; (b) continual fine-tuning—splitting the target domain into multiple periods for iterative training, which fails under data scarcity; (c) test-time training (TTT)—requiring auxiliary self-supervised tasks and architectural modifications, leading to high computational overhead.

**Goal**: How can test-time information be exploited at inference time with minimal overhead to correct prediction biases? Spatio-temporal forecasting offers a unique advantage—**label autocorrelation**: historical samples within a sliding window naturally acquire ground-truth labels after the window advances, enabling real-time calibration.

**Key Insight**: The paper operates in the spectral domain, decomposing the periodic biases induced by non-stationarity into amplitude and phase shifts, and performing node-level spectral calibration with very few parameters.

**Core Idea**: Freeze the backbone network and train only a frequency-domain amplitude-phase calibrator at test time, performing single-step gradient updates using streaming historical samples to achieve lightweight, plug-and-play test-time computing.

## Method

### Overall Architecture

ST-TTC consists of two cooperative components:
- **SD-Calibrator** (frequency-domain calibrator): addresses *what to compute*—applies frequency-domain calibration to the backbone's time-domain predictions.
- **Flash Gradient Update + Streaming Memory Queue**: addresses *how to compute*—efficiently updates calibrator parameters using historical test information.

### Key Designs

#### 1. Frequency-Domain Calibrator (SD-Calibrator)

**Function**: Performs frequency-domain amplitude and phase correction on the backbone prediction $\hat{y} \in \mathbb{R}^{B \times N \times T}$.

**Mechanism**:

- **Spatially-aware decomposition**: Applies rFFT independently to each spatial node, yielding the spectrum $Y_f = \text{rFFT}(\hat{y}) \in \mathbb{C}^{B \times N \times M}$, where $M = T/2 + 1$, decomposed into amplitude $A = |Y_f|$ and phase $P = \angle Y_f$.
- **Group modulation**: The $M$ frequency bins are divided into $G$ groups (default $G=4$), with learnable per-group per-node offset parameters $\lambda^\alpha \in \mathbb{R}^{G \times N \times 1}$ and $\lambda^\phi \in \mathbb{R}^{G \times N \times 1}$ (both initialized to zero).
- **Calibration formulas**: $A'_g = A_g \odot (1 + \lambda_g^\alpha)$, $P'_g = P_g + \lambda_g^\phi$.
- **Reconstruction**: $Y'_f = \bigcup_{g=1}^{G} A'_g \odot e^{jP'_g}$, followed by irFFT to recover the calibrated time-domain output $\hat{y}_{cal}$.

**Design Motivation**: Time-domain calibration requires a large number of parameters and is prone to overfitting noise. In the frequency domain, periodic variations manifest as amplitude and phase changes in specific frequency components, making frequency-domain calibration more direct and robust. The grouping strategy reduces the parameter count from $2NM$ (full spectrum) to $2NG$ ($G \ll M$), substantially lowering memory and gradient update costs.

#### 2. Flash Gradient Update + Streaming Memory Queue

**Function**: Efficiently updates SD-Calibrator parameters during inference.

**Mechanism**:

- **Streaming memory queue**: A FIFO queue $\mathcal{Q}$ is maintained with a maximum capacity equal to the forecasting horizon $T_f$. Each incoming test sample $(X_t, Y_t)$ is enqueued; when the queue is full, the oldest sample $(X_o, Y_o)$ is dequeued for gradient updates—**preventing information leakage**.
- **Flash gradient update**: A single gradient descent step is performed using the dequeued $(X_o, Y_o)$: $\hat{Y}_o^b = f_\theta(X_o)$ (backbone frozen), $\hat{Y}_o^{cal} = g_\theta(\hat{Y}_o^b)$, loss $\mathcal{L}(\hat{Y}_o^{cal}, Y_o)$ is computed, and $\lambda \leftarrow \lambda - \eta \nabla_\lambda \mathcal{L}$.

**Design Motivation**: Spatio-temporal forecasting demands **timeliness**—additional inference latency must remain smaller than the sliding window step. Single-sample, single-step gradient descent ensures flash-fast updates. The queue delays label usage by $T_f$ steps, effectively preventing information leakage.

### Loss & Training

- Backbone weights are completely frozen; only SD-Calibrator parameters ($\lambda^\alpha, \lambda^\phi$) are updated.
- Learning rate $lr = 1 \times 10^{-4}$; memory queue update sample count $n=1$; number of frequency groups $G=4$.
- All experiments share the same hyperparameters; each experiment is repeated 5 times, and mean ± standard deviation is reported.

## Key Experimental Results

### Main Results

| Dataset | Model | MAE (w/o TTC) | MAE (w/ TTC) | Gain |
|---------|-------|--------------|-------------|------|
| PEMS-03 | GWNet | 16.73 | 16.42 | ↓1.85% |
| PEMS-04 | STAEformer | 19.48 | 19.33 | ↓0.77% |
| PEMS-07 | STGCN | 24.26 | 23.83 | ↓1.77% |
| PEMS-08 | STAEformer | 14.84 | 14.73 | ↓0.74% |
| KnowAir | GWNet | 17.03 | 16.94 | ↓0.53% |
| UrbanEV | Various | — | — | More pronounced |

Across 6 backbone architectures (Transformer/Graph/MLP-based) × 6 datasets, ST-TTC delivers **consistent and universal improvements**.

### Ablation Study

| Component | PEMS-04 MAE | PEMS-08 MAE |
|-----------|------------|------------|
| Full ST-TTC | Best | Best |
| w/o phase calibration | Degraded | Degraded |
| w/o amplitude calibration | Degraded | Degraded |
| Time-domain calibration | Worse than freq. | Worse than freq. |
| Full spectrum vs. grouping | Overfitting | Overfitting |

### Key Findings

1. ST-TTC improves performance across **all** backbone architecture and dataset combinations, including further gains over current state-of-the-art models.
2. Datasets with larger distribution shifts (e.g., UrbanEV) exhibit more pronounced relative improvements.
3. Compared to existing test-time methods such as DOST, TTT-MAE, and TENT, ST-TTC achieves higher accuracy and faster inference.
4. ST-TTC remains effective on the large-scale LargeST benchmark (8,600+ nodes), adding only ~68K parameters (~0.02% overhead).
5. ST-TTC can be combined with OOD learning (STONE) and continual learning (EAC) methods for further gains.

## Highlights & Insights

- **Paradigm innovation**: This work is the first to formally define the test-time computing paradigm for spatio-temporal forecasting, with a clear formulation (Table 1 provides an exceptionally informative comparison).
- **Frequency-domain calibration as a core insight**: Non-stationarity primarily manifests as amplitude/phase drift in periodic patterns; frequency-domain calibration is more direct and robust than time-domain approaches.
- **Extreme lightweight design**: The calibrator introduces only $2NG$ parameters (e.g., 2,456 parameters when $G=4, N=307$), far fewer than the backbone network.
- **Plug-and-play**: No modification to backbone architecture or training procedure is required; any pretrained model can be directly enhanced.
- **Theoretical guarantee**: An approximate upper bound on the output perturbation of SD-Calibrator is provided (Theorem 1), ensuring the calibration does not overfit.

## Limitations & Future Work

1. The current approach assumes that periodic bias is the primary source of performance degradation; it may be less effective for entirely random distribution changes.
2. The number of frequency groups $G$ is a fixed hyperparameter; adaptive grouping strategies remain unexplored.
3. Single-step gradient updates are fast but limited in update magnitude, potentially insufficient for severe distribution shifts.
4. The possibility of using node-specific learning rates has not been explored.

## Related Work & Insights

- Test-time computing has proven successful in NLP (o1, r1) and CV (TTT, TENT); this work transfers the paradigm to the spatio-temporal forecasting domain.
- The frequency-domain processing approach is generalizable to other forecasting tasks with periodic patterns (e.g., periodic equipment monitoring, seasonal load forecasting).
- The FIFO queue + single-step update online learning framework is elegantly simple and effective, offering useful design principles for other streaming learning scenarios.

## Rating

⭐⭐⭐⭐⭐

NeurIPS 2025 Spotlight. The paper presents clear motivation, well-grounded design, elegant simplicity, and comprehensive experiments. The combination of frequency-domain calibration and flash gradient updates is highly elegant, addressing a practically important problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[NeurIPS 2025\] Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data](exploring_neural_granger_causality_with_xlstms_unveiling_temporal_dependencies_i.md)
- [\[NeurIPS 2025\] Probability Calibration for Precipitation Nowcasting](probability_calibration_for_precipitation_nowcasting.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[ICCV 2025\] V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction](../../ICCV2025/time_series/v2xpnp_vehicle-to-everything_spatio-temporal_fusion_for_multi-agent_perception_a.md)

</div>

<!-- RELATED:END -->
