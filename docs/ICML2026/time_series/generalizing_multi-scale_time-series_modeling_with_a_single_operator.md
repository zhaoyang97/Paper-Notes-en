---
title: >-
  [Paper Note] Generalizing Multi-scale Time-Series Modeling with a Single Operator
description: >-
  [ICML 2026][Time Series][Time Series Forecasting] The Sigma framework unifies existing discrete multi-scale operators by learning **Learned Discrete Gaussian (LDG) kernels** with **continuous…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time Series Forecasting"
  - "Multi-scale Modeling"
  - "Gaussian Kernels"
  - "Scale-space Theory"
date: 2026-05-08
content_hash: 956bda3f6a388dd2
---

# Generalizing Multi-scale Time-Series Modeling with a Single Operator

**Conference**: ICML 2026  
**arXiv**: [2605.31129](https://arxiv.org/abs/2605.31129)  
**Code**: To be confirmed  
**Area**: Time Series Forecasting / Multi-scale Modeling  
**Keywords**: Time Series Forecasting, Multi-scale Modeling, Gaussian Kernels, Scale-space Theory

## TL;DR
The Sigma framework unifies existing discrete multi-scale operators by learning **Learned Discrete Gaussian (LDG) kernels** with **continuous, distance-aware scale parameters**. It achieves SOTA performance on both long-term and short-term forecasting tasks while significantly reducing computational costs (5.3× faster training and 3.8× less GPU memory).

## Background & Motivation

**Background**: Multi-scale modeling has been established as an effective design principle for time-series forecasting, improving performance by capturing temporal dynamics across multiple resolutions. Existing methods include diverse strategies such as hierarchical decomposition (downsampling), frequency domain transforms (wavelet decomposition), and scale aggregation.

**Limitations of Prior Work**: Existing multi-scale methods rely on **fixed, discrete scale parameters** applied uniformly across all time steps. However: (1) The characteristic temporal scales of real-world time series (e.g., dominant frequencies, decay rates) vary continuously rather than discretely; (2) The optimal scale may differ across different time steps, a variation that discrete operators fail to accommodate.

**Key Challenge**: Discrete scale parameters introduce implicit boundaries in the representation space, preventing models from smoothly representing temporal dynamics across resolutions. Through the "predictability gap" theory (Theorem 4.2), it is proven that even the optimal discrete scale cannot reach the performance of the optimum in a continuous scale space.

**Goal**: Establish a mathematical foundation for multi-scale time-series modeling and design a unified framework capable of learning continuous, dynamic scale parameters.

**Key Insight**: Drawing from **scale-space theory** (originating from computer vision), this work adopts **Learned Discrete Gaussian (LDG) kernels** as an instance of a generalized scale operator family.

**Core Idea**: Replace multiple discrete scale operators with a single learnable Gaussian kernel operator. Use $L$ position-dependent continuous scale parameters $\mathbf{s}$ to dynamically control the degree of smoothing for each time step.

## Method

### Overall Architecture
The architecture consists of three layers: (1) **Unified Mathematical Foundation**: Formalizes the concept of a scale operator family via "non-extensivity" and "energy diminishing" axioms, unifying six categories of methods including average pooling, max pooling, moving average, downsampling, patching, and wavelet decomposition; (2) **Generalized Scale Operator Family**: Extends discrete scale operator families to continuous versions $\mathcal{F} = \{f(\mathbf{x} \mid \mathbf{s}) \mid \mathbf{s} \in \mathbb{R}_+^M\}$, ensuring consistency and differentiability; (3) **Lightweight Predictor with LDG Kernels**: Uses simple MLPs for prediction to avoid complex cross-scale interaction modules.

### Key Designs

1.  **Unified Framework for Scale Operator Families**:
    - **Function**: Establishes a rigorous mathematical foundation for multi-scale time-series modeling.
    - **Mechanism**: Defines two mathematical properties that a scale operator family $\mathcal{F}$ must satisfy: non-extensivity (the operator introduces no new information) and energy diminishing (coarser scales are simpler than finer scales). Theorem 3.2 proves that six common operations satisfy these, while many trivial operations (scalar multiplication, permutation) do not.
    - **Design Motivation**: Reveals the fundamental limitations of discrete scale parameters—Theorem 4.2 proves that the optimality of continuous scale spaces is strictly greater than their discrete counterparts.

2.  **Learned Discrete Gaussian (LDG) Kernels**:
    - **Function**: Implements distance-aware, position-dependent continuous scale parameters.
    - **Mechanism**: The $(i, j)$-th element of the kernel matrix is $[\mathbf{K}(\mathbf{s})]_{i, j} = e^{-s_d} I_d(s_d)$, where $d = |i - j|$ is the temporal distance and $I_d(\cdot)$ is the modified Bessel function of the first kind. Position-dependent $\mathbf{s} \in \mathbb{R}_+^L$ are learned—each position $i$ has a scale parameter $s_i$ controlling the degree of neighborhood aggregation.
    - **Design Motivation**: Theorem 4.3 guarantees that the LDG kernel family is a generalized scale operator family; Theorem 4.4 (stronger) proves that LDG is the unique symmetric kernel satisfying discrete scale-space axioms, eliminating the implicit boundaries of discrete operators.

3.  **Trend-Residual Decomposition + Lightweight MLP Predictor**:
    - **Function**: Leverages learned LDG representations for efficient forecasting.
    - **Mechanism**: Decomposes the embedding $\mathbf{X} = \text{Embed}(\mathbf{x})$ into a smooth component $\mathbf{K}(\mathbf{s}) \mathbf{X}$ and a residual component $(\mathbf{I} - \mathbf{K}(\mathbf{s})) \mathbf{X}$, concatenated as $\mathbf{H} \in \mathbb{R}^{2L \times d}$. Predictions are made via an MLP with skip connections: $\hat{\mathbf{y}} = \mathbf{W}_1 (\text{MLP}(\mathbf{H}) + \mathbf{H}) \mathbf{W}_2$.
    - **Design Motivation**: Trend-seasonal decomposition is inspired by classical time-series methods; skip connections stabilize optimization and preserve scale-specific information. This design is exceptionally concise compared to methods like AMD or TimeMixer that require multi-level downsampling and complex interactions.

## Key Experimental Results

### Main Results: Long-term Forecasting

| Dataset | Metric | **Sigma** | AMD | WPMixer | TimeMixer |
|---------|--------|-----------|-----|---------|-----------|
| Weather | MSE    | 0.247     | 0.263 | 0.255   | 0.246     |
| Electricity | MSE | **0.175** | 0.208 | 0.198   | 0.185     |
| Traffic | MSE    | **0.458** | 0.546 | 0.497   | 0.501     |
| Exchange| MSE    | **0.353** | 0.358 | 0.387   | 0.384     |
| ETTm2   | MSE    | **0.276** | 0.285 | 0.283   | 0.281     |

Sigma outperforms competitors in 13 out of 16 settings, with significant advantages on high-dimensional datasets.

### Ablation Study

| Configuration | MSE | MAE | Description |
|---------------|-----|-----|-------------|
| Sigma (Full)  | **0.480** | **0.468** | Baseline |
| ① Replace MLP with TimeMixer mixing | 0.486 | 0.467 | +0.6% error |
| ② LDG with single scale parameter | 0.489 | 0.473 | +1.9% error; position-dependence is important |
| ③ Sample-level scale parameters | 0.490 | 0.474 | +2.1% error; excessive flexibility introduces noise |
| ④ No scale operator, raw input only | 0.492 | 0.475 | +2.5% error |
| ⑤ Replace LDG with Moving Average | 0.493 | 0.475 | +2.7% error; learnability is key |
| ⑥ Unconstrained Conv (non-scale family) | 0.524 | 0.492 | +9.2% error; **Worst** |

### Efficiency Analysis

| Metric | Sigma | AMD | Gain |
|--------|-------|-----|------|
| Training Time | — | — | **5.3× Faster** |
| Memory Usage  | — | — | **3.8× Less** |

### Key Findings
- Position-dependence, learnability, and the constraints of the LDG kernel as a generalized scale operator family are all critical.
- The simplicity of the MLP is sufficient even when replacing other multi-scale strategies (Variant ①).
- Arbitrary convolutions that violate the axioms of scale operator families (Variant ⑥) cause performance collapse, confirming the necessity of the theoretical foundation.
- M4 Short-term Forecasting: Sigma wins in 11 out of 15 cases.

## Highlights & Insights
- **First Rigorous Application of Scale-space Theory**: Establishes a mathematical foundation for multi-scale time-series modeling for the first time, unifying six existing methods under the "scale operator family" concept.
- **Multi-scale Modeling via Continuous Optimization**: The core insight is shifting the "optimal scale parameter" from a problem hyperparameter to a **learned parameter**. By proving continuous scale space optimality strictly exceeds discrete versions, the performance gain of learning $\mathbf{s} \in \mathbb{R}_+^L$ is theoretically explained.
- **Minimalist and Efficient Architecture**: Sigma achieves SOTA using a single LDG kernel and an MLP, offering greater elegance than methods involving multi-layer interactions.
- **Alignment of Theory and Practice**: The sharp performance drop in Variant ⑥ (unconstrained convolution) directly validates the necessity of the "scale operator family" constraints.

## Limitations & Future Work
- **Dataset-level Scale Parameter Constraints**: Sharing $\mathbf{s}$ across samples makes learning difficult when training data is sparse, leading to mediocre performance in the M4 "Others" category (< 5% of data).
- **Computational Complexity of LDG Kernels**: The current implementation uses dense matrix multiplication with $O(L^2)$ complexity. Since the kernel matrix is Toeplitz, it could theoretically be reduced to $O(L \log L)$ using FFT or truncated convolutions.
- **Multivariate Interactions**: Adopts a channel-independent assumption, potentially overlooking inter-variable dependencies.

## Related Work & Insights
- **vs TimeMixer / AMD**: All are multi-scale methods, but TimeMixer fixes discrete scales and AMD introduces complex cross-scale mixing. Sigma achieves superior performance with a simpler architecture through learnable continuous parameters and mathematical constraints.
- **vs Scale-space Theory (CV)**: Sigma represents the first rigorous application of classical Witkin and Lindeberg scale-space ideas to time series.
- **vs Wavelet Decomposition**: Sigma’s LDG is theoretically more general (the scale operator family includes wavelets as a special case) and possesses stronger learning capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to rigorously formalize scale-space theory for time series, unifying existing methods with significant theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducts long-term forecasting (8 datasets × 4 lengths), short-term forecasting (M4), efficiency analysis, and in-depth ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain (Motivation → Definition → Theorem → Design → Experiment).
- Value: ⭐⭐⭐⭐⭐ Refreshes SOTA while establishing a mathematical foundation for multi-scale modeling; highly practical due to significant efficiency gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](../../AAAI2026/time_series/freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](../../NeurIPS2025/time_series/multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[ICML 2026\] IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection](impact_influence_modeling_for_open-set_time_series_anomaly_detection.md)
- [\[ICML 2026\] TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models](tsrbench_a_comprehensive_multi-task_multi-modal_time_series_reasoning_benchmark_.md)

</div>

<!-- RELATED:END -->
