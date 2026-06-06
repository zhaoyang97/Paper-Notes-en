---
title: >-
  [Paper Note] DeepBooTS: Dual-Stream Residual Boosting for Drift-Resilient Time-Series Forecasting
description: >-
  [AAAI 2026][Time Series][time series forecasting] This paper proposes DeepBooTS, which leverages bias-variance decomposition theory to demonstrate that weighted ensembling reduces variance and thereby mitigates concept d…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "time series forecasting"
  - "concept drift"
  - "ensemble learning"
  - "residual boosting"
  - "bias-variance decomposition"
date: 2026-05-08
content_hash: e96a3233c073fe30
---

# DeepBooTS: Dual-Stream Residual Boosting for Drift-Resilient Time-Series Forecasting

**Conference**: AAAI 2026
**arXiv**: [2511.06893](https://arxiv.org/abs/2511.06893)  
**Code**: [https://github.com/Anoise/DeepBooTS](https://github.com/Anoise/DeepBooTS)  
**Area**: Time Series Forecasting
**Keywords**: time series forecasting, concept drift, ensemble learning, residual boosting, bias-variance decomposition

## TL;DR
This paper proposes DeepBooTS, which leverages bias-variance decomposition theory to demonstrate that weighted ensembling reduces variance and thereby mitigates concept drift. The method introduces a dual-stream residual-decreasing boosting architecture in which each block corrects the residual of the preceding block, achieving an average improvement of 15.8% across multiple datasets.

## Background & Motivation

**Background**: Mainstream time series forecasting methods (Transformers, MLPs, etc.) suffer from concept drift on non-stationary data—training loss decreases while validation loss rises, preventing generalization to distribution-shifted data.

**Limitations of Prior Work**: Methods such as RevIN (Reversible Instance Normalization) alleviate mean shift but leave variance instability unresolved. Experiments show that concept drift is pervasive in standard benchmarks including ETT, Traffic, and Weather, leading to poor model generalization.

**Key Challenge**: From a bias-variance perspective, when bias and noise are fixed, the degree of concept drift is governed by prediction variance. Existing models cannot effectively reduce prediction variance to cope with distributional drift.

**Goal**: Mitigate concept drift by reducing prediction variance through a theoretically grounded ensemble strategy.

**Key Insight**: The paper first proves that weighted ensembling does not increase bias while reducing variance, then implements a gradient-boosting-style residual-decreasing mechanism within a deep network.

**Core Idea**: Each block of the deep network serves as a learner in an ensemble. A dual-stream residual-decreasing architecture realizes boosting internally within the network, with theoretical guarantees of variance reduction.

## Method

### Overall Architecture
DeepBooTS comprises a dual-stream architecture:
- **Input stream**: Performs implicit decomposition of the input across layers—each block extracts a signal component and passes the remainder to the next block, $R_l = X_l - \hat{X}_l$.
- **Output stream**: The prediction of each block corrects the residual of all preceding blocks, with alternating addition and subtraction realizing a boosting ensemble.

### Key Designs

1. **Theoretical Foundation — Ensemble Variance Reduction**:

    - Function: Provides a theoretical guarantee for ensemble-based resistance to concept drift from a bias-variance decomposition perspective.
    - Mechanism: Theorem 1 proves that simple averaging $\bar{Y} = \frac{1}{N}\sum \hat{Y}_t$ satisfies $\text{Var}(\bar{Y}) \leq \text{Var}(\hat{Y})$ with unchanged bias. Theorem 2 further proves that weighted ensembling yields strictly lower MSE than a single model after distributional shift. Theorem 3 proves that the variance upper bound of DeepBooTS under subtractive aggregation is $\frac{4}{L}\alpha^2(\nu + \mu)$, whereas additive aggregation yields $\frac{4}{L}\alpha^2\nu + 3\alpha^2\mu$ (substantially larger).
    - Design Motivation: To provide not only empirical validation but also theoretical guarantees. Variance reduction → concept drift mitigation → lower test error.

2. **Dual-Stream Residual-Decreasing Architecture**:

    - Function: Implements gradient-boosting-style layer-wise residual correction within a deep network.
    - Mechanism: Each block adopts a fork structure—receiving $X_l$ and producing two outputs: (a) residual $R_l = X_l - \text{Block}_l(X_l)$, passed to the next block for further processing (input stream); (b) prediction $O_l = \text{Predictor}_l(\hat{X}_l)$, aggregated with predictions from preceding blocks via alternating subtraction (output stream). The final prediction is a weighted difference of predictions from odd- and even-indexed learners.
    - Design Motivation: Subtractive aggregation yields theoretically lower variance (Theorem 3) and corresponds to implicit decomposition of the input—each block processes information missed by previous blocks.

3. **Learnable Gating Coefficients**:

    - Function: Enables each learner to adaptively regulate the transmission rate of both input and output streams.
    - Mechanism: Input stream: $X_{l+1} = \varphi(\theta_1(R_{l,2})) \cdot \theta_2(R_{l,2})$; output stream: $O_{l+1} = \varphi(\theta_3([\hat{X}_{l,1}, \hat{X}_{l,2}])) \cdot \theta_4([\hat{X}_{l,1}, \hat{X}_{l,2}])$, where $\varphi$ denotes sigmoid gating.
    - Design Motivation: Different blocks require different weights to control information flow; fixed weights lack sufficient flexibility.

4. **Flexible Learner Design**:

    - Function: The base learner can be instantiated with different neural network architectures.
    - Mechanism: Supports attention layers (temporal or FFT frequency-domain) combined with feedforward layers; each layer also applies residual subtraction internally: $R_{l,1} = X_{l,1} - \delta\hat{X}_{l,1}$, where $\delta$ controls whether the attention layer is enabled.
    - Design Motivation: A modular design allows the framework to adapt to diverse tasks; FFT-based attention enables lightweight and efficient computation.

### Loss & Training
- MSE loss with end-to-end training.
- Auxiliary outputs from each block form highway connections to the final prediction, stabilizing gradient propagation.

## Key Experimental Results

### Main Results
Multivariate forecasting (average MSE/MAE across six major datasets):

| Model | ETT Avg | Traffic | ELC | Weather | Solar | PEMS |
|-------|---------|---------|-----|---------|-------|------|
| iTransformer | 0.383 | 0.428 | 0.178 | 0.258 | 0.233 | 0.113 |
| TimeMixer | 0.367 | 0.484 | 0.182 | 0.240 | 0.216 | 0.138 |
| PatchTST | 0.381 | 0.481 | 0.205 | 0.259 | 0.270 | 0.180 |
| **DeepBooTS** | **0.362** | **0.406** | **0.166** | **0.245** | **0.227** | **0.109** |
| **DeepBooTS*** | **0.346** | **0.373** | **0.158** | **0.227** | **0.197** | **0.075** |

Univariate forecasting results are also comprehensively superior—ETTh1 MSE 0.072 vs. Periodformer 0.093 (−22.6%).

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Replace subtractive aggregation in output stream with additive | Increased variance, higher MSE |
| w/o attention layer (feedforward only) | Slight performance drop, remains competitive |
| w/o gating (fixed weights) | Performance degradation |
| Increasing block count $L$ | Sustained performance improvement without saturation |

### Key Findings
- Subtraction vs. addition aggregation: experiments validate Theorem 3—subtractive aggregation yields lower variance and lower MSE.
- Increasing the number of blocks does not exacerbate concept drift (theoretical guarantee: variance decreases with $L$).
- The largest gain is observed on the large-scale PEMS dataset (0.075 vs. 0.113), demonstrating advantages in large-scale settings.
- Validation curves show that DeepBooTS does not exhibit the early rise in validation error seen in other methods, confirming effective mitigation of concept drift.

## Highlights & Insights
- **Theory-driven architecture design** is the primary contribution: the approach begins from bias-variance decomposition theory, proves that ensembling reduces variance, proves that subtraction outperforms addition, and then derives the network design—rather than resorting to empirical stacking.
- The concept of **"boosting within a deep network"** is elegant: rather than training multiple independent models, the multiple blocks of a single network serve as ensemble learners sharing gradient computation.
- An **average improvement of 15.8%** is a substantial margin in time series forecasting, indicating that concept drift has been severely underestimated as a problem.

## Limitations & Future Work
- The theoretical analysis assumes that estimation errors across blocks are i.i.d. Gaussian with identical noise levels, whereas in practice different blocks may exhibit distinct error characteristics.
- Although the alternating subtraction coefficients $\alpha_l$ are learnable, the equal-weight initialization assumption may not be appropriate for all scenarios.
- The paper does not thoroughly analyze behavior under extreme non-stationarity, such as abrupt changes or structural breaks.
- The choice between channel-independent and channel-mixing strategies is not discussed.

## Related Work & Insights
- **vs. iTransformer**: iTransformer models inter-variable correlations via inverted attention but does not address concept drift. DeepBooTS achieves 0.362 vs. 0.383 on ETT and 0.406 vs. 0.428 on Traffic.
- **vs. TimeMixer**: TimeMixer employs multi-scale mixing but still operates within a fixed window. DeepBooTS reduces variance at the architectural level through residual boosting.
- **vs. N-BEATS**: N-BEATS also uses residual learning and a fork architecture, but lacks theoretically grounded subtractive aggregation. The subtractive aggregation in DeepBooTS comes with a proven variance upper bound.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Theory-driven deep boosting design with rigorous bias-variance analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6+ datasets, multivariate and univariate settings, large-scale experiments, complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though dense notation requires careful reading.
- Value: ⭐⭐⭐⭐⭐ 15.8% average improvement with theoretical guarantees, establishing a new paradigm for drift-resilient time series forecasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](../../ICML2026/time_series/dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)
- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](../../ICML2026/time_series/dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)
- [\[AAAI 2026\] Harmonic Dataset Distillation for Time Series Forecasting](harmonic_dataset_distillation_for_time_series_forecasting.md)
- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[AAAI 2026\] Optimal Look-back Horizon for Time Series Forecasting in Federated Learning](optimal_look-back_horizon_for_time_series_forecasting_in_federated_learning.md)

</div>

<!-- RELATED:END -->
