---
title: >-
  [Paper Note] XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs
description: >-
  [AAAI2026][Time Series][time series forecasting] This paper proposes XLinear, a lightweight time series forecasting model based on MLP with sigmoid gating. Through a global token mechanism…
tags:
  - "AAAI2026"
  - "Time Series"
  - "time series forecasting"
  - "MLP"
  - "exogenous inputs"
  - "gating mechanism"
  - "lightweight model"
date: 2026-05-08
content_hash: ab7c367b5777fe09
---

# XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs

**Conference**: AAAI2026
**arXiv**: [2601.09237](https://arxiv.org/abs/2601.09237)  
**Code**: [Zaiwen/XLinear](https://github.com/Zaiwen/XLinear)  
**Area**: Time Series
**Keywords**: time series forecasting, MLP, exogenous inputs, gating mechanism, lightweight model

## TL;DR
This paper proposes XLinear, a lightweight time series forecasting model based on MLP with sigmoid gating. Through a global token mechanism, it efficiently integrates endogenous and exogenous variable information, achieving an optimal accuracy–efficiency trade-off across 12 datasets.

## Background & Motivation

### State of the Field

**Background**: Transformer-based models (e.g., TimeXer) achieve high accuracy but incur substantial computational overhead; their patch mechanisms are also limited by permutation invariance, which causes loss of temporal ordering information.

### Root Cause

**Key Challenge**: MLP-based models (e.g., DLinear) are efficient and lightweight but neglect cross-variate dependencies, and in particular fail to exploit exogenous inputs.

### Limitations of Prior Work

**Limitations of Prior Work**: In real-world scenarios, exogenous variables (e.g., meteorological data) exhibit unidirectional causal relationships with endogenous variables (e.g., water temperature); leveraging such relationships can substantially improve forecasting performance.

### Starting Point

**Key Insight**: Existing models either lack support for exogenous inputs or support them at prohibitive computational cost.

### Paper Goals

**Goal**: How can one effectively model the temporal patterns of endogenous variables and their cross-variate dependencies with exogenous variables while maintaining MLP-level efficiency, thereby achieving an optimal accuracy–efficiency trade-off?

## Method

### Overall Architecture
XLinear consists of four components: Embedding → TGM → VGM → Prediction Head

1. **Embedding**: Jointly embeds the endogenous sequence $X_{1:T}$ and exogenous sequence $E_{1:T}$, and introduces a learnable global token $X_{\text{glob}}$ for each endogenous variable.

2. **Time-wise Gating Module (TGM)**: Extracts temporal patterns via
$$[X'_{\text{endo}}, X'_{\text{glob}}] = \sigma(\text{Linear}_2(\phi(\text{Linear}_1(X_{\text{endo\_tok}})))) \odot X_{\text{endo\_tok}}$$
where $\phi$ denotes ReLU, $\sigma$ denotes sigmoid, and $\odot$ denotes element-wise multiplication.

3. **Variate-wise Gating Module (VGM)**: Concatenates the global token with the exogenous sequence and applies gating to extract cross-variate dependencies:
$$[E'_{\text{exo}}, X''_{\text{glob}}] = \sigma(\text{Linear}_4(\phi(\text{Linear}_3(X_{\text{exo\_tok}})))) \odot X_{\text{exo\_tok}}$$

4. **Prediction Head**: Concatenates $X'_{\text{endo}}$ and $X''_{\text{glob}}$, then applies a fully connected layer to produce the $S$-step ahead forecast.

### Key Designs
- **Global token as an information hub**: Prevents noise from direct interaction between endogenous and exogenous variables.
- **Sigmoid gating**: Achieves selective feature filtering with lower complexity than attention mechanisms.
- Loss: MSE $\mathcal{L} = \mathbb{E} \frac{1}{M} \sum_{i=1}^{M} \|\hat{X}^{(i)} - X^{(i)}\|_2^2$

## Key Experimental Results

### Main Results

| Model | Electricity MSE | ETTh1 MSE | Weather MSE | Training Speed |
|-------|----------------|-----------|-------------|----------------|
| TimeXer | 0.261 | 0.057 | 0.001 | Baseline |
| iTransformer | 0.299 | 0.057 | 0.001 | — |
| PatchTST | 0.339 | 0.055 | 0.001 | — |
| DLinear | 0.387 | 0.065 | 0.006 | Fastest |
| **XLinear** | **0.256** | **0.055** | **0.001** | **≥30% faster than Transformers** |

- Evaluated on 7 standard benchmarks and 5 real-world datasets containing exogenous inputs.
- On Electricity (horizon 96): XLinear MSE = 0.256 vs. TimeXer 0.261.
- Training speed matches DLinear; memory consumption is lower than all SOTA baselines.

## Highlights & Insights
- Extremely lightweight: only two MLP-based gating modules, with far fewer parameters than Transformer-based approaches.
- The global token design elegantly isolates cross-variate noise while preserving effective information transfer.
- Training speed exceeds that of efficient Transformers such as TimeXer by more than 30%.
- Demonstrates strong performance in real-world scenarios with exogenous inputs (dissolved oxygen, water temperature, crop yield).

## Limitations & Future Work
- The relatively simple architecture may underperform deep Transformers in scenarios with highly complex long-range dependencies.
- Only unidirectional causality (exogenous → endogenous) is considered; bidirectional interactions are not modeled.
- The input length is fixed at 96; the effect of varying look-back window sizes is not thoroughly explored.
- Comparison with foundation time series models is absent.

## Related Work & Insights

| Dimension | XLinear | TimeXer | DLinear | iTransformer |
|-----------|---------|---------|---------|-------------|
| Architecture | MLP + gating | Patch Transformer | Linear decomposition | Variate-level attention |
| Exogenous support | ✓ | ✓ | ✗ | ✗ |
| Cross-variate modeling | Global token + VGM | Cross-attention | None | Self-attention |
| Training efficiency | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Accuracy | SOTA | Runner-up | Low | Moderate |

## Related Work & Insights
- The "less is more" principle is once again validated in time series forecasting: a carefully designed MLP can surpass complex Transformer architectures.
- The global token concept (originally from TimeXer) is realized more efficiently here via MLP gating.
- Exogenous inputs are critically important in practical applications yet are overlooked by most standard benchmarks.

## Rating
- **Novelty**: ⭐⭐⭐ — The core idea is an efficient combination of existing modules; moderate novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 12 datasets, 10 baselines, and comprehensive efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and well-presented formulations.
- **Value**: ⭐⭐⭐⭐ — High practical value for real-world time series forecasting, especially in settings involving exogenous variables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CometNet: Contextual Motif-guided Long-term Time Series Forecasting](cometnet_contextual_motif-guided_long-term_time_series_forecasting.md)
- [\[AAAI 2026\] ReCast: Reliability-aware Codebook Assisted Lightweight Time Series Forecasting](recast_reliability-aware_codebook_assisted_lightweight_time_series_forecasting.md)
- [\[AAAI 2026\] ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition](probfm_probabilistic_time_series_foundation_model_with_uncertainty_decomposition.md)
- [\[ICML 2026\] DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](../../ICML2026/time_series/dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)
- [\[AAAI 2026\] A Unified Shape-Aware Foundation Model for Time Series Classification](a_unified_shape-aware_foundation_model_for_time_series_class.md)

</div>

<!-- RELATED:END -->
