---
title: >-
  [Paper Note] DBLoss: Decomposition-based Loss Function for Time Series Forecasting
description: >-
  [NeurIPS 2025][Autonomous Driving][Time Series Forecasting] This paper proposes DBLoss—a general-purpose loss function based on exponential moving average (EMA) decomposition. During loss computation…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Time Series Forecasting"
  - "Loss Function"
  - "Seasonality-Trend Decomposition"
  - "EMA"
  - "General-Purpose Loss"
date: 2026-05-08
content_hash: df9b634397ef14fe
---

# DBLoss: Decomposition-based Loss Function for Time Series Forecasting

**Conference**: NeurIPS 2025
**arXiv**: [2510.23672](https://arxiv.org/abs/2510.23672)  
**Code**: [https://github.com/decisionintelligence/DBLoss](https://github.com/decisionintelligence/DBLoss)  
**Area**: Autonomous Driving
**Keywords**: Time Series Forecasting, Loss Function, Seasonality-Trend Decomposition, EMA, General-Purpose Loss

## TL;DR
This paper proposes DBLoss—a general-purpose loss function based on exponential moving average (EMA) decomposition. During loss computation, both predictions and ground-truth values are decomposed into seasonal and trend components within the forecasting horizon, and losses are computed separately for each component. DBLoss serves as a plug-and-play replacement for MSE and consistently improves any deep learning forecasting model, with effectiveness validated across 8 benchmark datasets × 8 SOTA models.

## Background & Motivation

**Background**: Long-term time series forecasting is a critical task in economics, transportation, energy, and related domains. Mainstream deep models (DLinear, TimesNet, iTransformer, etc.) commonly incorporate seasonality-trend decomposition modules in their forward passes to extract effective representations.

**Limitations of Prior Work**: The standard MSE loss computes pointwise differences between predictions and ground truth directly, without explicitly constraining forecasting accuracy along the seasonal and trend dimensions. The authors identify three failure modes: (a) poor seasonal prediction but acceptable trend; (b) poor trend prediction but acceptable seasonality; (c) both components are poorly predicted. Even when a model performs decomposition in its forward pass, the loss function treats all errors uniformly.

**Key Challenge**: Decomposition in the forward pass provides useful inductive bias, yet the loss function does not exploit this prior—seasonal and trend components within the forecasting horizon are not supervised independently.

**Key Insight**: Since decomposition is beneficial in the forward pass, why not apply it at the loss computation stage as well? The proposed approach decomposes both predictions and ground truth via EMA within the forecasting horizon, computes losses for the trend and seasonal components separately, and combines them via a weighted fusion.

**Core Idea**: DBLoss = EMA decomposition + component-level independent losses + scale alignment weighting mechanism, with zero additional parameters and compatibility with any backbone.

## Method

### Overall Architecture
Given predictions $\hat{Y}$ and ground truth $Y$ (both in $\mathbb{R}^{N \times F}$, with $N$ channels and $F$ forecast steps) produced by an arbitrary backbone, DBLoss operates at the loss computation stage as follows: (1) EMA decomposition → seasonal and trend components; (2) separate computation of seasonal loss and trend loss; (3) scale-aligned weighted summation. The entire process does not modify the model architecture or incur additional inference overhead.

### Key Designs

1. **EMA Decomposition Module**:

    - Function: Decomposes predictions and ground truth into trend $Y_T$ and seasonality $Y_S = Y - Y_T$.
    - Mechanism: For a time series $X \in \mathbb{R}^{B \times T \times N}$, weights are computed as $W = [(1-\alpha)^{T-1}, (1-\alpha)^{T-2}, \ldots, 1]$, where $\alpha \in (0,1)$ is the smoothing factor. $W[1:]$ is multiplied by $\alpha$, then element-wise multiplied with the input; a cumulative sum $C = \text{cumsum}(X \times W)$ is computed and divided by a normalization factor $D_{div}$ to obtain the trend $\text{Trend} = C / D_{div}$; the residual yields the seasonal component $\text{Seasonality} = X - \text{Trend}$.
    - Design Motivation: EMA is more sensitive to recent changes than SMA, has $O(T)$ complexity, and requires no window size selection.

2. **Component-Level Loss Computation**:

    - Seasonal loss uses the L2 norm: $\mathcal{L}_S = |\hat{Y}_S - Y_S|_2$
    - Trend loss uses the L1 norm: $\mathcal{L}_T = |\hat{Y}_T - Y_T|_1$
    - Design Motivation: The seasonal component exhibits high variability, making L2 a suitable constraint; the trend component is relatively smooth, and L1 offers greater robustness.

3. **Scale Alignment Mechanism**:

    - Function: Prevents either component's loss from dominating optimization due to scale discrepancy.
    - Core Formula: $\mathcal{L}_T^{\text{aligned}} = \mathcal{L}_T \times \text{stopgrad}\left(\frac{\mathcal{L}_S}{\mathcal{L}_T + \epsilon}\right)$
    - The stop-gradient operator prevents gradient flow through the alignment ratio, avoiding gradient interference between the two loss components.

4. **Final Weighted Loss**:

    - $\mathcal{L} = \beta \cdot \mathcal{L}_S + (1-\beta) \cdot \mathcal{L}_T^{\text{aligned}}$
    - $\beta$ balances the contributions of the seasonal and trend components and can be tuned for different application scenarios.

### Loss & Training
- Only the loss function is replaced; model architecture, optimizer, and hyperparameters remain unchanged.
- The TFB unified evaluation framework is used to ensure fair comparison, without applying the "Drop Last" trick.

## Key Experimental Results

### Main Results — Comparison with Original Loss (8 datasets × 4 SOTA models, Avg MSE/MAE)

| Model | Original MSE/MAE | DBLoss MSE/MAE | Gain |
|-------|-----------------|---------------|------|
| iTransformer | 0.439/0.448 (ETTh1 Avg) | 0.423/0.430 | MSE↓3.6% |
| Amplifier | 0.428/0.435 (ETTh1 Avg) | 0.419/0.425 | MSE↓2.1% |
| PatchTST | 0.419/0.436 (ETTh1 Avg) | 0.402/0.420 | MSE↓4.1% |
| DLinear | 0.425/0.439 (ETTh1 Avg) | 0.412/0.425 | MSE↓3.1% |
| PatchTST | 0.351/0.395 (ETTh2 Avg) | 0.337/0.381 | MSE↓4.0% |
| DLinear | 0.470/0.468 (ETTh2 Avg) | 0.409/0.424 | MSE↓13.0% |

DBLoss outperforms MSE across **all** backbones and **the vast majority** of datasets. Models that already incorporate decomposition modules in the forward pass, such as DLinear, also benefit, demonstrating that forward decomposition and loss-side decomposition are complementary rather than redundant.

### Foundation Model Experiments

| Foundation Model | Original MSE | DBLoss MSE | Note |
|-----------------|-------------|-----------|------|
| CALF | baseline | improved | LLM-based methods also benefit |
| UniTS | baseline | improved | Pre-trained methods show improvement |
| TTM | baseline | improved | Lightweight foundation models benefit |
| GPT4TS | baseline | improved | GPT-based architectures also benefit |

### Key Findings
- DBLoss achieves the largest gain on ETTh2 + DLinear (MSE: 0.470→0.409, ↓13%), which is precisely the combination where decomposition models with MSE perform worst, confirming that MSE fails to effectively supervise decomposed components.
- DBLoss remains effective for models with existing decomposition modules (DLinear, DUET), establishing that forward decomposition and loss-side decomposition are complementary.
- Computational overhead is negligible—only one additional EMA computation and two norm evaluations are required.
- The default value of $\beta = 0.5$ yields strong performance across most scenarios, indicating low sensitivity to this hyperparameter.

## Highlights & Insights
- **Elegantly minimal design**: The entire method consists of a single EMA decomposition, two component losses, and one scale alignment mechanism. Despite minimal implementation complexity, it yields consistent and significant gains—a testament to the principle that a well-designed loss function can outperform a more complex model.
- **Scale alignment + stop-gradient** is a transferable technique applicable to any multi-component loss to mitigate scale imbalance.
- **Forward decomposition and loss-side decomposition are complementary**, not redundant—this finding challenges intuition and demonstrates that injecting inductive bias at the loss level is an effective design dimension independent of model architecture.

## Limitations & Future Work
- Validation is limited to long-term multivariate forecasting; short-term forecasting and univariate settings are not explored.
- The EMA smoothing factor $\alpha$ is fixed; in principle, it could be learned or selected adaptively.
- The trend definition via EMA may lack robustness for non-stationary series—the effectiveness of EMA decomposition under extreme distributional shift remains uncertain.
- Extensions involving multi-head losses (e.g., incorporating frequency-domain component constraints) are not explored.

## Related Work & Insights
- **vs. Soft-DTW / DILATE**: These shape-based losses focus on shape alignment but have high computational complexity ($O(T^2)$); DBLoss requires only $O(T)$ and approaches the problem from a decomposition perspective—the two are orthogonal and complementary.
- **vs. FreDF**: Frequency-domain losses target frequency dependencies; DBLoss approaches the problem from time-domain decomposition, and the two can be used in combination.
- **vs. PSLoss**: Patch-level structural losses focus on local statistics; DBLoss targets global seasonal/trend components.
- For autonomous driving: trajectory prediction is fundamentally a time series forecasting problem, and DBLoss may be explored as an auxiliary loss for motion prediction tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The design is concise yet effective; injecting decomposition priors at the loss level represents a novel perspective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 datasets × 8 models (including 4 foundation models), with comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure with intuitive motivational diagrams
- Value: ⭐⭐⭐⭐ A general-purpose plug-and-play loss with high practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)
- [\[NeurIPS 2025\] ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset](chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)
- [\[NeurIPS 2025\] Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion](layer-wise_modality_decomposition_for_interpretable_multimodal_sensor_fusion.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](../../ICCV2025/autonomous_driving/future-aware_interaction_network_for_motion_forecasting.md)
- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](../../ICCV2025/autonomous_driving/uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)

</div>

<!-- RELATED:END -->
